//! HTTP/REST API server implementation using axum

use crate::store::VecStore;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// HTTP server wrapper around VecStore
#[derive(Clone)]
pub struct VecStoreHttpServer {
    store: Arc<RwLock<VecStore>>,
}

impl VecStoreHttpServer {
    /// Create a new HTTP server
    pub fn new(store: VecStore) -> Self {
        Self {
            store: Arc::new(RwLock::new(store)),
        }
    }

    /// Create a new HTTP server with shared store
    pub fn with_store(store: Arc<RwLock<VecStore>>) -> Self {
        Self { store }
    }

    /// Build the router
    pub fn router(&self) -> Router {
        Router::new()
            // Vector operations
            .route("/v1/upsert", post(upsert))
            .route("/v1/batch-upsert", post(batch_upsert))
            .route("/v1/batch-execute", post(batch_execute))
            .route("/v1/query", post(query))
            .route("/v1/query-explain", post(query_explain))
            .route("/v1/query-estimate", post(query_estimate))
            .route("/v1/delete/:id", delete(delete_vector))
            .route("/v1/soft-delete/:id", post(soft_delete))
            .route("/v1/restore/:id", post(restore))
            // Database operations
            .route("/v1/compact", post(compact))
            .route("/v1/stats", get(get_stats))
            // Snapshot operations
            .route("/v1/snapshots", post(create_snapshot))
            .route("/v1/snapshots", get(list_snapshots))
            .route("/v1/snapshots/:name/restore", post(restore_snapshot))
            // Hybrid search
            .route("/v1/hybrid-query", post(hybrid_query))
            // WebSocket streaming
            .route("/ws/query-stream", get(query_stream_ws))
            // Metrics
            .route("/metrics", get(metrics_endpoint))
            // Health check
            .route("/health", get(health_check))
            .route("/ready", get(ready_check))
            .with_state(self.clone())
            .layer(CorsLayer::permissive())
            .layer(TraceLayer::new_for_http())
    }

    /// Get the store reference
    pub fn store(&self) -> Arc<RwLock<VecStore>> {
        self.store.clone()
    }
}

// ============================================================================
// Request/Response types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct UpsertRequest {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpsertResponse {
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchUpsertRequest {
    pub records: Vec<UpsertRequest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchUpsertResponse {
    pub inserted: i32,
    pub updated: i32,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryRequest {
    pub vector: Vec<f32>,
    pub limit: i32,
    pub filter: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResult {
    pub id: String,
    pub score: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    pub results: Vec<QueryResult>,
    pub stats: Option<QueryStats>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryStats {
    pub total_candidates: i32,
    pub filtered_count: i32,
    pub duration_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExplainedQueryResult {
    pub id: String,
    pub score: f32,
    pub metadata: HashMap<String, serde_json::Value>,
    pub explanation: ExplanationDto,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExplanationDto {
    pub raw_score: f32,
    pub distance_metric: String,
    pub filter_passed: bool,
    pub filter_details: Option<FilterEvaluationDto>,
    pub graph_stats: Option<GraphStatsDto>,
    pub rank: usize,
    pub explanation_text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FilterEvaluationDto {
    pub filter_expr: String,
    pub matched_conditions: Vec<String>,
    pub failed_conditions: Vec<String>,
    pub passed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphStatsDto {
    pub distance_calculations: usize,
    pub nodes_visited: usize,
    pub found_at_layer: Option<usize>,
    pub hops_from_entry: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryExplainResponse {
    pub results: Vec<ExplainedQueryResult>,
    pub stats: Option<QueryStats>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteResponse {
    pub found: bool,
    pub deleted: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SoftDeleteResponse {
    pub found: bool,
    pub marked_deleted: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RestoreResponse {
    pub found: bool,
    pub restored: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompactResponse {
    pub removed_count: i32,
    pub freed_bytes: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StatsResponse {
    pub total_vectors: i64,
    pub active_vectors: i64,
    pub deleted_vectors: i64,
    pub dimension: i32,
    pub storage_bytes: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotRequest {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotResponse {
    pub success: bool,
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotInfo {
    pub name: String,
    pub created_at: i64,
    pub size_bytes: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListSnapshotsResponse {
    pub snapshots: Vec<SnapshotInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RestoreSnapshotResponse {
    pub success: bool,
    pub vectors_restored: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HybridQueryRequest {
    pub vector: Vec<f32>,
    pub text_query: String,
    pub limit: i32,
    pub alpha: Option<f32>,
    pub filter: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    pub status: String,
    pub message: Option<String>,
}

// Batch operations DTOs
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum BatchOperationDto {
    Upsert {
        id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, serde_json::Value>,
    },
    Delete {
        id: String,
    },
    SoftDelete {
        id: String,
    },
    Restore {
        id: String,
    },
    UpdateMetadata {
        id: String,
        metadata: HashMap<String, serde_json::Value>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchExecuteRequest {
    pub operations: Vec<BatchOperationDto>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchExecuteResponse {
    pub succeeded: usize,
    pub failed: usize,
    pub errors: Vec<BatchErrorDto>,
    pub duration_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchErrorDto {
    pub index: usize,
    pub operation: String,
    pub error: String,
}

// Query estimation DTOs
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryEstimateRequest {
    pub vector: Vec<f32>,
    pub limit: i32,
    pub filter: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryEstimateResponse {
    pub valid: bool,
    pub errors: Vec<String>,
    pub cost_estimate: f32,
    pub estimated_distance_calculations: usize,
    pub estimated_nodes_visited: usize,
    pub will_overfetch: bool,
    pub recommendations: Vec<String>,
    pub estimated_duration_ms: f32,
}

// ============================================================================
// Error handling
// ============================================================================

struct ApiError(anyhow::Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let error_msg = format!("{}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": error_msg
            })),
        )
            .into_response()
    }
}

impl<E> From<E> for ApiError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

// ============================================================================
// Handler functions
// ============================================================================

async fn upsert(
    State(server): State<VecStoreHttpServer>,
    Json(req): Json<UpsertRequest>,
) -> Result<Json<UpsertResponse>, ApiError> {
    let start = std::time::Instant::now();

    let metadata = crate::store::Metadata {
        fields: req.metadata,
    };

    let mut store = server.store.write().await;
    store.upsert(req.id, req.vector, metadata)?;

    let duration = start.elapsed().as_secs_f64();
    super::metrics::record_upsert(false);
    super::metrics::record_request("/v1/upsert", "POST", duration);

    Ok(Json(UpsertResponse {
        success: true,
        error: None,
    }))
}

async fn batch_upsert(
    State(server): State<VecStoreHttpServer>,
    Json(req): Json<BatchUpsertRequest>,
) -> Result<Json<BatchUpsertResponse>, ApiError> {
    let start = std::time::Instant::now();

    let mut store = server.store.write().await;
    let mut inserted = 0;
    let mut errors = Vec::new();

    for upsert_req in req.records {
        let metadata = crate::store::Metadata {
            fields: upsert_req.metadata,
        };

        match store.upsert(upsert_req.id.clone(), upsert_req.vector, metadata) {
            Ok(_) => inserted += 1,
            Err(e) => errors.push(format!("{}: {}", upsert_req.id, e)),
        }
    }

    let duration = start.elapsed().as_secs_f64();
    super::metrics::record_upsert(true);
    super::metrics::record_request("/v1/batch-upsert", "POST", duration);

    Ok(Json(BatchUpsertResponse {
        inserted,
        updated: 0,
        errors,
    }))
}

async fn batch_execute(
    State(server): State<VecStoreHttpServer>,
    Json(req): Json<BatchExecuteRequest>,
) -> Result<Json<BatchExecuteResponse>, ApiError> {
    // Convert DTOs to internal BatchOperation types
    let operations: Vec<crate::store::BatchOperation> = req
        .operations
        .into_iter()
        .map(|op_dto| match op_dto {
            BatchOperationDto::Upsert {
                id,
                vector,
                metadata,
            } => crate::store::BatchOperation::Upsert {
                id,
                vector,
                metadata: crate::store::Metadata { fields: metadata },
            },
            BatchOperationDto::Delete { id } => crate::store::BatchOperation::Delete { id },
            BatchOperationDto::SoftDelete { id } => crate::store::BatchOperation::SoftDelete { id },
            BatchOperationDto::Restore { id } => crate::store::BatchOperation::Restore { id },
            BatchOperationDto::UpdateMetadata { id, metadata } => {
                crate::store::BatchOperation::UpdateMetadata {
                    id,
                    metadata: crate::store::Metadata { fields: metadata },
                }
            }
        })
        .collect();

    let mut store = server.store.write().await;
    let result = store.batch_execute(operations)?;

    // Convert result errors to DTOs
    let errors_dto: Vec<BatchErrorDto> = result
        .errors
        .into_iter()
        .map(|e| BatchErrorDto {
            index: e.index,
            operation: e.operation,
            error: e.error,
        })
        .collect();

    super::metrics::record_upsert(true);
    super::metrics::record_request("/v1/batch-execute", "POST", result.duration_ms / 1000.0);

    Ok(Json(BatchExecuteResponse {
        succeeded: result.succeeded,
        failed: result.failed,
        errors: errors_dto,
        duration_ms: result.duration_ms,
    }))
}

async fn query(
    State(server): State<VecStoreHttpServer>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    let start = std::time::Instant::now();

    let filter = if let Some(ref filter_str) = req.filter {
        Some(crate::store::parse_filter(filter_str)?)
    } else {
        None
    };

    let query = crate::store::Query {
        vector: req.vector,
        k: req.limit as usize,
        filter,
    };

    let store = server.store.read().await;

    let neighbors = store.query(query)?;

    let duration = start.elapsed().as_secs_f64();
    let duration_ms = duration * 1000.0;

    // Record metrics
    super::metrics::record_query("vector", neighbors.len(), duration);
    super::metrics::record_request("/v1/query", "POST", duration);

    let results = neighbors
        .iter()
        .map(|n| QueryResult {
            id: n.id.clone(),
            score: n.score,
            metadata: n.metadata.fields.clone(),
        })
        .collect();

    let stats = Some(QueryStats {
        total_candidates: neighbors.len() as i32,
        filtered_count: 0,
        duration_ms,
    });

    Ok(Json(QueryResponse { results, stats }))
}

async fn query_explain(
    State(server): State<VecStoreHttpServer>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryExplainResponse>, ApiError> {
    let start = std::time::Instant::now();

    let filter = if let Some(ref filter_str) = req.filter {
        Some(crate::store::parse_filter(filter_str)?)
    } else {
        None
    };

    let query = crate::store::Query {
        vector: req.vector,
        k: req.limit as usize,
        filter,
    };

    let store = server.store.read().await;

    let explained_neighbors = store.query_explain(query)?;

    let duration = start.elapsed().as_secs_f64();
    let duration_ms = duration * 1000.0;

    // Record metrics
    super::metrics::record_query("vector_explain", explained_neighbors.len(), duration);
    super::metrics::record_request("/v1/query-explain", "POST", duration);

    let results = explained_neighbors
        .iter()
        .map(|n| ExplainedQueryResult {
            id: n.id.clone(),
            score: n.score,
            metadata: n.metadata.fields.clone(),
            explanation: ExplanationDto {
                raw_score: n.explanation.raw_score,
                distance_metric: n.explanation.distance_metric.clone(),
                filter_passed: n.explanation.filter_passed,
                filter_details: n.explanation.filter_details.as_ref().map(|fd| {
                    FilterEvaluationDto {
                        filter_expr: fd.filter_expr.clone(),
                        matched_conditions: fd.matched_conditions.clone(),
                        failed_conditions: fd.failed_conditions.clone(),
                        passed: fd.passed,
                    }
                }),
                graph_stats: n.explanation.graph_stats.as_ref().map(|gs| GraphStatsDto {
                    distance_calculations: gs.distance_calculations,
                    nodes_visited: gs.nodes_visited,
                    found_at_layer: gs.found_at_layer,
                    hops_from_entry: gs.hops_from_entry,
                }),
                rank: n.explanation.rank,
                explanation_text: n.explanation.explanation_text.clone(),
            },
        })
        .collect();

    let stats = Some(QueryStats {
        total_candidates: explained_neighbors.len() as i32,
        filtered_count: 0,
        duration_ms,
    });

    Ok(Json(QueryExplainResponse { results, stats }))
}

async fn query_estimate(
    State(server): State<VecStoreHttpServer>,
    Json(req): Json<QueryEstimateRequest>,
) -> Result<Json<QueryEstimateResponse>, ApiError> {
    let filter = if let Some(ref filter_str) = req.filter {
        Some(crate::store::parse_filter(filter_str)?)
    } else {
        None
    };

    let query = crate::store::Query {
        vector: req.vector,
        k: req.limit as usize,
        filter,
    };

    let store = server.store.read().await;
    let estimate = store.estimate_query(&query);

    // Record metrics
    super::metrics::record_request("/v1/query-estimate", "POST", 0.001); // Estimate is very fast

    Ok(Json(QueryEstimateResponse {
        valid: estimate.valid,
        errors: estimate.errors,
        cost_estimate: estimate.cost_estimate,
        estimated_distance_calculations: estimate.estimated_distance_calculations,
        estimated_nodes_visited: estimate.estimated_nodes_visited,
        will_overfetch: estimate.will_overfetch,
        recommendations: estimate.recommendations,
        estimated_duration_ms: estimate.estimated_duration_ms,
    }))
}

async fn delete_vector(
    State(server): State<VecStoreHttpServer>,
    Path(id): Path<String>,
) -> Result<Json<DeleteResponse>, ApiError> {
    let mut store = server.store.write().await;
    store.remove(&id)?;

    Ok(Json(DeleteResponse {
        found: true,
        deleted: true,
    }))
}

async fn soft_delete(
    State(server): State<VecStoreHttpServer>,
    Path(id): Path<String>,
) -> Result<Json<SoftDeleteResponse>, ApiError> {
    let mut store = server.store.write().await;
    let marked = store.soft_delete(&id)?;

    Ok(Json(SoftDeleteResponse {
        found: marked,
        marked_deleted: marked,
    }))
}

async fn restore(
    State(server): State<VecStoreHttpServer>,
    Path(id): Path<String>,
) -> Result<Json<RestoreResponse>, ApiError> {
    let mut store = server.store.write().await;
    let restored = store.restore(&id)?;

    Ok(Json(RestoreResponse {
        found: restored,
        restored,
    }))
}

async fn compact(
    State(server): State<VecStoreHttpServer>,
) -> Result<Json<CompactResponse>, ApiError> {
    let mut store = server.store.write().await;
    let removed_count = store.compact()?;

    Ok(Json(CompactResponse {
        removed_count: removed_count as i32,
        freed_bytes: 0,
    }))
}

async fn get_stats(
    State(server): State<VecStoreHttpServer>,
) -> Result<Json<StatsResponse>, ApiError> {
    let store = server.store.read().await;

    Ok(Json(StatsResponse {
        total_vectors: store.len() as i64 + store.deleted_count() as i64,
        active_vectors: store.active_count() as i64,
        deleted_vectors: store.deleted_count() as i64,
        dimension: store.dimension() as i32,
        storage_bytes: 0,
    }))
}

async fn create_snapshot(
    State(server): State<VecStoreHttpServer>,
    Json(req): Json<SnapshotRequest>,
) -> Result<Json<SnapshotResponse>, ApiError> {
    let store = server.store.read().await;
    store.create_snapshot(&req.name)?;

    Ok(Json(SnapshotResponse {
        success: true,
        path: format!("snapshots/{}", req.name),
    }))
}

async fn list_snapshots(
    State(server): State<VecStoreHttpServer>,
) -> Result<Json<ListSnapshotsResponse>, ApiError> {
    let store = server.store.read().await;
    let snapshots_info = store.list_snapshots()?;

    let snapshots = snapshots_info
        .into_iter()
        .map(|(name, _timestamp, size)| SnapshotInfo {
            name,
            created_at: 0,
            size_bytes: size as i64,
        })
        .collect();

    Ok(Json(ListSnapshotsResponse { snapshots }))
}

async fn restore_snapshot(
    State(server): State<VecStoreHttpServer>,
    Path(name): Path<String>,
) -> Result<Json<RestoreSnapshotResponse>, ApiError> {
    let mut store = server.store.write().await;
    store.restore_snapshot(&name)?;

    Ok(Json(RestoreSnapshotResponse {
        success: true,
        vectors_restored: store.len() as i64,
    }))
}

async fn hybrid_query(
    State(server): State<VecStoreHttpServer>,
    Json(req): Json<HybridQueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    let query = crate::store::HybridQuery {
        vector: req.vector,
        keywords: req.text_query,
        k: req.limit as usize,
        alpha: req.alpha.unwrap_or(0.7),
        filter: req.filter.and_then(|f| crate::store::parse_filter(&f).ok()),
    };

    let store = server.store.read().await;
    let start = std::time::Instant::now();

    let neighbors = store.hybrid_query(query)?;

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

    let results = neighbors
        .iter()
        .map(|n| QueryResult {
            id: n.id.clone(),
            score: n.score,
            metadata: n.metadata.fields.clone(),
        })
        .collect();

    let stats = Some(QueryStats {
        total_candidates: neighbors.len() as i32,
        filtered_count: 0,
        duration_ms,
    });

    Ok(Json(QueryResponse { results, stats }))
}

async fn health_check() -> Result<Json<HealthCheckResponse>, ApiError> {
    Ok(Json(HealthCheckResponse {
        status: "healthy".to_string(),
        message: Some("VecStore server is running".to_string()),
    }))
}

async fn ready_check(
    State(server): State<VecStoreHttpServer>,
) -> Result<Json<HealthCheckResponse>, ApiError> {
    // Verify we can access the store
    let _ = server.store.read().await;

    Ok(Json(HealthCheckResponse {
        status: "ready".to_string(),
        message: Some("VecStore server is ready to accept requests".to_string()),
    }))
}

/// Prometheus metrics endpoint
async fn metrics_endpoint(State(server): State<VecStoreHttpServer>) -> Result<String, ApiError> {
    // Update database statistics
    let store = server.store.read().await;
    super::metrics::update_db_stats(
        store.len() + store.deleted_count(),
        store.active_count(),
        store.deleted_count(),
        store.dimension(),
    );
    drop(store);

    // Encode metrics
    super::metrics::encode_metrics()
        .map_err(|e| ApiError(anyhow::anyhow!("Failed to encode metrics: {}", e)))
}

// ============================================================================
// WebSocket streaming
// ============================================================================

/// WebSocket handler for streaming query results
async fn query_stream_ws(
    ws: WebSocketUpgrade,
    State(server): State<VecStoreHttpServer>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_query_stream(socket, server))
}

/// Handle WebSocket connection for query streaming
async fn handle_query_stream(mut socket: WebSocket, server: VecStoreHttpServer) {
    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(msg) => msg,
            Err(e) => {
                tracing::error!("WebSocket error: {}", e);
                break;
            }
        };

        match msg {
            Message::Text(text) => {
                // Parse query request from JSON
                let req: Result<QueryRequest, _> = serde_json::from_str(&text);

                match req {
                    Ok(query_req) => {
                        // Execute query
                        let filter = if let Some(ref filter_str) = query_req.filter {
                            match crate::store::parse_filter(filter_str) {
                                Ok(f) => Some(f),
                                Err(e) => {
                                    let error_msg = serde_json::json!({
                                        "error": format!("Invalid filter: {}", e)
                                    });
                                    if socket
                                        .send(Message::Text(error_msg.to_string().into()))
                                        .await
                                        .is_err()
                                    {
                                        break;
                                    }
                                    continue;
                                }
                            }
                        } else {
                            None
                        };

                        let query = crate::store::Query {
                            vector: query_req.vector,
                            k: query_req.limit as usize,
                            filter,
                        };

                        let store = server.store.read().await;
                        let start = std::time::Instant::now();

                        match store.query(query) {
                            Ok(neighbors) => {
                                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                                let total_results = neighbors.len();

                                // Stream results one by one
                                for neighbor in &neighbors {
                                    let result = QueryResult {
                                        id: neighbor.id.clone(),
                                        score: neighbor.score,
                                        metadata: neighbor.metadata.fields.clone(),
                                    };

                                    let result_json = match serde_json::to_string(&result) {
                                        Ok(json) => json,
                                        Err(e) => {
                                            tracing::error!("Failed to serialize result: {}", e);
                                            break;
                                        }
                                    };

                                    if socket
                                        .send(Message::Text(result_json.into()))
                                        .await
                                        .is_err()
                                    {
                                        break;
                                    }
                                }

                                // Send completion message with stats
                                let completion = serde_json::json!({
                                    "complete": true,
                                    "stats": {
                                        "duration_ms": duration_ms,
                                        "total_results": total_results
                                    }
                                });

                                if socket
                                    .send(Message::Text(completion.to_string().into()))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Err(e) => {
                                let error_msg = serde_json::json!({
                                    "error": format!("Query failed: {}", e)
                                });
                                if socket
                                    .send(Message::Text(error_msg.to_string().into()))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let error_msg = serde_json::json!({
                            "error": format!("Invalid query request: {}", e)
                        });
                        if socket
                            .send(Message::Text(error_msg.to_string().into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
            Message::Close(_) => {
                break;
            }
            _ => {
                // Ignore other message types (binary, ping, pong)
            }
        }
    }
}
