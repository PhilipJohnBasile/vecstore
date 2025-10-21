//! gRPC server implementation using tonic

use super::types::{pb, *};
use crate::store::VecStore;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_stream;
use tonic::{Request, Response, Status};

/// gRPC server wrapper around VecStore
pub struct VecStoreGrpcServer {
    store: Arc<RwLock<VecStore>>,
}

impl VecStoreGrpcServer {
    /// Create a new gRPC server
    pub fn new(store: VecStore) -> Self {
        Self {
            store: Arc::new(RwLock::new(store)),
        }
    }

    /// Create a new gRPC server with shared store
    pub fn with_store(store: Arc<RwLock<VecStore>>) -> Self {
        Self { store }
    }

    /// Get the store reference (for sharing with HTTP server)
    pub fn store(&self) -> Arc<RwLock<VecStore>> {
        self.store.clone()
    }
}

#[tonic::async_trait]
impl pb::vec_store_service_server::VecStoreService for VecStoreGrpcServer {
    /// Type alias for the streaming query response
    type QueryStreamStream = std::pin::Pin<
        Box<dyn tokio_stream::Stream<Item = Result<pb::QueryResult, Status>> + Send + 'static>,
    >;

    /// Insert or update a vector
    async fn upsert(
        &self,
        request: Request<pb::UpsertRequest>,
    ) -> Result<Response<pb::UpsertResponse>, Status> {
        let req = request.into_inner();

        // Convert protobuf metadata to Metadata
        let metadata = pb_metadata_to_metadata(&req.metadata)
            .map_err(|e| Status::invalid_argument(format!("Invalid metadata: {}", e)))?;

        // Perform upsert
        let mut store = self.store.write().await;
        store
            .upsert(req.id, req.vector, metadata)
            .map_err(|e| Status::internal(format!("Upsert failed: {}", e)))?;

        Ok(Response::new(pb::UpsertResponse {
            success: true,
            error: None,
        }))
    }

    /// Batch insert/update multiple vectors
    async fn batch_upsert(
        &self,
        request: Request<pb::BatchUpsertRequest>,
    ) -> Result<Response<pb::BatchUpsertResponse>, Status> {
        let req = request.into_inner();

        let mut store = self.store.write().await;
        let mut inserted = 0;
        let mut errors = Vec::new();

        for upsert_req in req.records {
            match pb_metadata_to_metadata(&upsert_req.metadata) {
                Ok(metadata) => {
                    match store.upsert(upsert_req.id.clone(), upsert_req.vector, metadata) {
                        Ok(_) => inserted += 1,
                        Err(e) => errors.push(format!("{}: {}", upsert_req.id, e)),
                    }
                }
                Err(e) => errors.push(format!("{}: invalid metadata: {}", upsert_req.id, e)),
            }
        }

        Ok(Response::new(pb::BatchUpsertResponse {
            inserted,
            updated: 0, // We don't distinguish between insert and update
            errors,
        }))
    }

    /// Query for similar vectors
    async fn query(
        &self,
        request: Request<pb::QueryRequest>,
    ) -> Result<Response<pb::QueryResponse>, Status> {
        let req = request.into_inner();

        // Convert to Query
        let query = pb_query_to_query(&req)
            .map_err(|e| Status::invalid_argument(format!("Invalid query: {}", e)))?;

        // Execute query
        let store = self.store.read().await;
        let start = std::time::Instant::now();

        let neighbors = store
            .query(query)
            .map_err(|e| Status::internal(format!("Query failed: {}", e)))?;

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Convert results
        let results = neighbors.iter().map(neighbor_to_query_result).collect();

        let stats = Some(pb::QueryStats {
            total_candidates: neighbors.len() as i32,
            filtered_count: 0, // Filter tracking requires query execution instrumentation
            duration_ms,
            cache_hit: false, // Semantic cache integration is a future optimization
        });

        Ok(Response::new(pb::QueryResponse { results, stats }))
    }

    /// Stream query results (for large result sets)
    async fn query_stream(
        &self,
        request: Request<pb::QueryRequest>,
    ) -> Result<Response<Self::QueryStreamStream>, Status> {
        let req = request.into_inner();

        // Convert to Query
        let query = pb_query_to_query(&req)
            .map_err(|e| Status::invalid_argument(format!("Invalid query: {}", e)))?;

        // Execute query
        let store = self.store.read().await;
        let neighbors = store
            .query(query)
            .map_err(|e| Status::internal(format!("Query failed: {}", e)))?;

        // Create stream
        let results: Vec<pb::QueryResult> =
            neighbors.iter().map(neighbor_to_query_result).collect();

        let stream = tokio_stream::iter(results.into_iter().map(Ok));

        Ok(Response::new(Box::pin(stream)))
    }

    /// Hard delete a vector
    async fn delete(
        &self,
        request: Request<pb::DeleteRequest>,
    ) -> Result<Response<pb::DeleteResponse>, Status> {
        let req = request.into_inner();

        let mut store = self.store.write().await;
        store
            .remove(&req.id)
            .map_err(|e| Status::internal(format!("Delete failed: {}", e)))?;

        Ok(Response::new(pb::DeleteResponse {
            found: true,
            deleted: true,
        }))
    }

    /// Soft delete a vector
    async fn soft_delete(
        &self,
        request: Request<pb::SoftDeleteRequest>,
    ) -> Result<Response<pb::SoftDeleteResponse>, Status> {
        let req = request.into_inner();

        let mut store = self.store.write().await;
        let marked = store
            .soft_delete(&req.id)
            .map_err(|e| Status::internal(format!("Soft delete failed: {}", e)))?;

        Ok(Response::new(pb::SoftDeleteResponse {
            found: marked,
            marked_deleted: marked,
        }))
    }

    /// Restore a soft-deleted vector
    async fn restore(
        &self,
        request: Request<pb::RestoreRequest>,
    ) -> Result<Response<pb::RestoreResponse>, Status> {
        let req = request.into_inner();

        let mut store = self.store.write().await;
        let restored = store
            .restore(&req.id)
            .map_err(|e| Status::internal(format!("Restore failed: {}", e)))?;

        Ok(Response::new(pb::RestoreResponse {
            found: restored,
            restored,
        }))
    }

    /// Compact database (remove soft-deleted vectors)
    async fn compact(
        &self,
        _request: Request<pb::CompactRequest>,
    ) -> Result<Response<pb::CompactResponse>, Status> {
        let mut store = self.store.write().await;
        let removed_count = store
            .compact()
            .map_err(|e| Status::internal(format!("Compact failed: {}", e)))?;

        Ok(Response::new(pb::CompactResponse {
            removed_count: removed_count as i32,
            freed_bytes: 0, // Byte tracking requires storage layer instrumentation
        }))
    }

    /// Get database statistics
    async fn get_stats(
        &self,
        _request: Request<pb::StatsRequest>,
    ) -> Result<Response<pb::StatsResponse>, Status> {
        let store = self.store.read().await;

        Ok(Response::new(pb::StatsResponse {
            total_vectors: store.len() as i64 + store.deleted_count() as i64,
            active_vectors: store.active_count() as i64,
            deleted_vectors: store.deleted_count() as i64,
            dimension: store.dimension() as i32,
            storage_bytes: 0, // Storage size calculation requires persistence layer API extension
            cache_stats: None, // Semantic cache integration is a future optimization
        }))
    }

    /// Create a snapshot
    async fn create_snapshot(
        &self,
        request: Request<pb::SnapshotRequest>,
    ) -> Result<Response<pb::SnapshotResponse>, Status> {
        let req = request.into_inner();

        let store = self.store.read().await;
        store
            .create_snapshot(&req.name)
            .map_err(|e| Status::internal(format!("Snapshot failed: {}", e)))?;

        Ok(Response::new(pb::SnapshotResponse {
            success: true,
            path: format!("snapshots/{}", req.name),
        }))
    }

    /// List all snapshots
    async fn list_snapshots(
        &self,
        _request: Request<pb::ListSnapshotsRequest>,
    ) -> Result<Response<pb::ListSnapshotsResponse>, Status> {
        let store = self.store.read().await;
        let snapshots_info = store
            .list_snapshots()
            .map_err(|e| Status::internal(format!("List snapshots failed: {}", e)))?;

        let snapshots = snapshots_info
            .into_iter()
            .map(|(name, timestamp, size)| pb::SnapshotInfo {
                name,
                created_at: timestamp.parse::<i64>().unwrap_or(0),
                size_bytes: size as i64,
            })
            .collect();

        Ok(Response::new(pb::ListSnapshotsResponse { snapshots }))
    }

    /// Restore from snapshot
    async fn restore_snapshot(
        &self,
        request: Request<pb::RestoreSnapshotRequest>,
    ) -> Result<Response<pb::RestoreSnapshotResponse>, Status> {
        let req = request.into_inner();

        let mut store = self.store.write().await;
        store
            .restore_snapshot(&req.name)
            .map_err(|e| Status::internal(format!("Restore snapshot failed: {}", e)))?;

        Ok(Response::new(pb::RestoreSnapshotResponse {
            success: true,
            vectors_restored: store.len() as i64,
        }))
    }

    /// Hybrid search (vector + keyword)
    async fn hybrid_query(
        &self,
        request: Request<pb::HybridQueryRequest>,
    ) -> Result<Response<pb::QueryResponse>, Status> {
        let req = request.into_inner();

        let query = crate::store::HybridQuery {
            vector: req.vector,
            keywords: req.text_query,
            k: req.limit as usize,
            alpha: req.alpha.unwrap_or(0.7) as f32,
            filter: req.filter.and_then(|f| crate::store::parse_filter(&f).ok()),
        };

        let store = self.store.read().await;
        let start = std::time::Instant::now();

        let neighbors = store
            .hybrid_query(query)
            .map_err(|e| Status::internal(format!("Hybrid query failed: {}", e)))?;

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        let results = neighbors.iter().map(neighbor_to_query_result).collect();

        let stats = Some(pb::QueryStats {
            total_candidates: neighbors.len() as i32,
            filtered_count: 0,
            duration_ms,
            cache_hit: false,
        });

        Ok(Response::new(pb::QueryResponse { results, stats }))
    }

    /// Health check
    async fn health_check(
        &self,
        _request: Request<pb::HealthCheckRequest>,
    ) -> Result<Response<pb::HealthCheckResponse>, Status> {
        // Simple health check - just verify we can access the store
        let _ = self.store.read().await;

        Ok(Response::new(pb::HealthCheckResponse {
            status: pb::health_check_response::ServingStatus::Serving as i32,
            message: Some("Healthy".to_string()),
        }))
    }
}
