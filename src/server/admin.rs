//! Admin API implementation for namespace management
//!
//! Implements the VecStoreAdminService gRPC service and provides
//! HTTP endpoints for namespace lifecycle management.

use crate::namespace_manager::NamespaceManager;
use crate::server::types::pb::vec_store_admin_service_server::VecStoreAdminService;
use crate::server::types::pb::*;
use crate::server::types::{
    namespace_info_to_proto, namespace_quotas_from_proto, namespace_status_from_proto,
    namespace_status_to_proto,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

/// gRPC Admin Service implementation
pub struct AdminService {
    manager: Arc<RwLock<NamespaceManager>>,
}

impl AdminService {
    pub fn new(manager: Arc<RwLock<NamespaceManager>>) -> Self {
        Self { manager }
    }
}

#[tonic::async_trait]
impl VecStoreAdminService for AdminService {
    async fn create_namespace(
        &self,
        request: Request<CreateNamespaceRequest>,
    ) -> Result<Response<CreateNamespaceResponse>, Status> {
        let req = request.into_inner();

        let manager = self.manager.write().await;

        // Convert proto quotas to Rust type
        let quotas = req
            .quotas
            .map(namespace_quotas_from_proto)
            .transpose()
            .map_err(|e| Status::invalid_argument(format!("Invalid quotas: {}", e)))?;

        // Create namespace
        manager
            .create_namespace(req.id.clone(), req.name.clone(), quotas)
            .map_err(|e| Status::internal(format!("Failed to create namespace: {}", e)))?;

        // Get the created namespace for response
        let namespace = manager
            .get_namespace(&req.id)
            .map_err(|e| Status::internal(format!("Failed to retrieve namespace: {}", e)))?;

        Ok(Response::new(CreateNamespaceResponse {
            success: true,
            error: None,
            namespace: Some(namespace_info_to_proto(&namespace)),
        }))
    }

    async fn list_namespaces(
        &self,
        request: Request<ListNamespacesRequest>,
    ) -> Result<Response<ListNamespacesResponse>, Status> {
        let req = request.into_inner();

        let manager = self.manager.read().await;
        let mut namespaces = manager.list_namespaces();

        // Filter by status if requested
        if let Some(status_filter) = req.status_filter {
            let filter_status = namespace_status_from_proto(status_filter)
                .ok_or_else(|| Status::invalid_argument("Invalid status filter"))?;

            namespaces.retain(|ns| ns.status == filter_status);
        }

        let namespace_infos: Vec<NamespaceInfo> =
            namespaces.iter().map(namespace_info_to_proto).collect();

        Ok(Response::new(ListNamespacesResponse {
            namespaces: namespace_infos,
        }))
    }

    async fn get_namespace(
        &self,
        request: Request<GetNamespaceRequest>,
    ) -> Result<Response<GetNamespaceResponse>, Status> {
        let req = request.into_inner();

        let manager = self.manager.read().await;

        match manager.get_namespace(&req.namespace_id) {
            Ok(namespace) => Ok(Response::new(GetNamespaceResponse {
                namespace: Some(namespace_info_to_proto(&namespace)),
                error: None,
            })),
            Err(e) => Ok(Response::new(GetNamespaceResponse {
                namespace: None,
                error: Some(e.to_string()),
            })),
        }
    }

    async fn update_namespace_quotas(
        &self,
        request: Request<UpdateNamespaceQuotasRequest>,
    ) -> Result<Response<UpdateNamespaceQuotasResponse>, Status> {
        let req = request.into_inner();

        let quotas = req
            .quotas
            .ok_or_else(|| Status::invalid_argument("Quotas are required"))?;

        let quotas = namespace_quotas_from_proto(quotas)
            .map_err(|e| Status::invalid_argument(format!("Invalid quotas: {}", e)))?;

        let manager = self.manager.write().await;

        match manager.update_quotas(&req.namespace_id, quotas) {
            Ok(_) => Ok(Response::new(UpdateNamespaceQuotasResponse {
                success: true,
                error: None,
            })),
            Err(e) => Ok(Response::new(UpdateNamespaceQuotasResponse {
                success: false,
                error: Some(e.to_string()),
            })),
        }
    }

    async fn update_namespace_status(
        &self,
        request: Request<UpdateNamespaceStatusRequest>,
    ) -> Result<Response<UpdateNamespaceStatusResponse>, Status> {
        let req = request.into_inner();

        let status = namespace_status_from_proto(req.status)
            .ok_or_else(|| Status::invalid_argument("Invalid status"))?;

        let manager = self.manager.write().await;

        match manager.update_status(&req.namespace_id, status) {
            Ok(_) => Ok(Response::new(UpdateNamespaceStatusResponse {
                success: true,
                error: None,
            })),
            Err(e) => Ok(Response::new(UpdateNamespaceStatusResponse {
                success: false,
                error: Some(e.to_string()),
            })),
        }
    }

    async fn delete_namespace(
        &self,
        request: Request<DeleteNamespaceRequest>,
    ) -> Result<Response<DeleteNamespaceResponse>, Status> {
        let req = request.into_inner();

        let manager = self.manager.write().await;

        match manager.delete_namespace(&req.namespace_id) {
            Ok(_) => Ok(Response::new(DeleteNamespaceResponse {
                success: true,
                error: None,
            })),
            Err(e) => Ok(Response::new(DeleteNamespaceResponse {
                success: false,
                error: Some(e.to_string()),
            })),
        }
    }

    async fn get_namespace_stats(
        &self,
        request: Request<GetNamespaceStatsRequest>,
    ) -> Result<Response<GetNamespaceStatsResponse>, Status> {
        let req = request.into_inner();

        let manager = self.manager.read().await;

        let stats = manager
            .get_stats(&req.namespace_id)
            .map_err(|e| Status::not_found(format!("Namespace not found: {}", e)))?;

        let namespace = manager
            .get_namespace(&req.namespace_id)
            .map_err(|e| Status::not_found(format!("Namespace not found: {}", e)))?;

        Ok(Response::new(GetNamespaceStatsResponse {
            namespace_id: stats.namespace_id,
            vector_count: stats.vector_count as i64,
            active_count: stats.active_count as i64,
            deleted_count: stats.deleted_count as i64,
            dimension: stats.dimension as i32,
            quota_utilization: stats.quota_utilization,
            total_requests: stats.total_requests as i64,
            total_queries: stats.total_queries as i64,
            total_upserts: stats.total_upserts as i64,
            total_deletes: stats.total_deletes as i64,
            status: namespace_status_to_proto(namespace.status) as i32,
        }))
    }

    async fn get_aggregate_stats(
        &self,
        _request: Request<GetAggregateStatsRequest>,
    ) -> Result<Response<GetAggregateStatsResponse>, Status> {
        let manager = self.manager.read().await;

        let stats = manager.get_aggregate_stats();

        Ok(Response::new(GetAggregateStatsResponse {
            total_namespaces: stats.total_namespaces as i32,
            active_namespaces: stats.active_namespaces as i32,
            total_vectors: stats.total_vectors as i64,
            total_requests: stats.total_requests as i64,
        }))
    }
}
