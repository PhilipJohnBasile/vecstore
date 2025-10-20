//! HTTP/REST Admin API endpoints for namespace management

use crate::namespace::{NamespaceQuotas, NamespaceStatus};
use crate::namespace_manager::NamespaceManager;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// Admin HTTP server wrapper
#[derive(Clone)]
pub struct AdminHttpServer {
    manager: Arc<RwLock<NamespaceManager>>,
}

impl AdminHttpServer {
    pub fn new(manager: Arc<RwLock<NamespaceManager>>) -> Self {
        Self { manager }
    }

    /// Build the admin router
    pub fn router(&self) -> Router {
        Router::new()
            .route("/admin/namespaces", post(create_namespace))
            .route("/admin/namespaces", get(list_namespaces))
            .route("/admin/namespaces/:id", get(get_namespace))
            .route("/admin/namespaces/:id/quotas", put(update_quotas))
            .route("/admin/namespaces/:id/status", put(update_status))
            .route("/admin/namespaces/:id", delete(delete_namespace))
            .route("/admin/namespaces/:id/stats", get(get_namespace_stats))
            .route("/admin/stats", get(get_aggregate_stats))
            .route("/health", get(health_check))
            .route("/ready", get(ready_check))
            .with_state(self.clone())
            .layer(CorsLayer::permissive())
            .layer(TraceLayer::new_for_http())
    }
}

// ============================================================================
// Request/Response types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateNamespaceRequest {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub quotas: Option<NamespaceQuotasDto>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NamespaceQuotasDto {
    pub max_vectors: Option<usize>,
    pub max_storage_bytes: Option<u64>,
    pub max_requests_per_second: Option<f64>,
    pub max_concurrent_queries: Option<usize>,
    pub max_dimension: Option<usize>,
    pub max_results_per_query: Option<usize>,
    pub max_batch_size: Option<usize>,
}

impl From<NamespaceQuotasDto> for NamespaceQuotas {
    fn from(dto: NamespaceQuotasDto) -> Self {
        NamespaceQuotas {
            max_vectors: dto.max_vectors,
            max_storage_bytes: dto.max_storage_bytes,
            max_requests_per_second: dto.max_requests_per_second,
            max_concurrent_queries: dto.max_concurrent_queries,
            max_dimension: dto.max_dimension,
            max_results_per_query: dto.max_results_per_query,
            max_batch_size: dto.max_batch_size,
        }
    }
}

impl From<NamespaceQuotas> for NamespaceQuotasDto {
    fn from(quotas: NamespaceQuotas) -> Self {
        NamespaceQuotasDto {
            max_vectors: quotas.max_vectors,
            max_storage_bytes: quotas.max_storage_bytes,
            max_requests_per_second: quotas.max_requests_per_second,
            max_concurrent_queries: quotas.max_concurrent_queries,
            max_dimension: quotas.max_dimension,
            max_results_per_query: quotas.max_results_per_query,
            max_batch_size: quotas.max_batch_size,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateStatusRequest {
    pub status: String, // "active", "suspended", "read_only", "pending_deletion"
}

#[derive(Debug, Serialize)]
pub struct NamespaceInfoDto {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub quotas: NamespaceQuotasDto,
    pub status: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub quota_utilization: f64,
    pub is_near_quota: bool,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct NamespaceStatsDto {
    pub namespace_id: String,
    pub vector_count: usize,
    pub active_count: usize,
    pub deleted_count: usize,
    pub dimension: usize,
    pub quota_utilization: f64,
    pub total_requests: u64,
    pub total_queries: u64,
    pub total_upserts: u64,
    pub total_deletes: u64,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct AggregateStatsDto {
    pub total_namespaces: usize,
    pub active_namespaces: usize,
    pub total_vectors: usize,
    pub total_requests: u64,
}

// ============================================================================
// Admin HTTP handlers
// ============================================================================

async fn create_namespace(
    State(server): State<AdminHttpServer>,
    Json(req): Json<CreateNamespaceRequest>,
) -> Result<Json<NamespaceInfoDto>, AppError> {
    let manager = server.manager.write().await;

    let quotas = req.quotas.map(|q| q.into());

    manager
        .create_namespace(req.id.clone(), req.name.clone(), quotas)
        .map_err(|e| AppError::Internal(e.to_string()))?;

    let namespace = manager
        .get_namespace(&req.id)
        .map_err(|e| AppError::Internal(e.to_string()))?;

    Ok(Json(NamespaceInfoDto {
        id: namespace.id.clone(),
        name: namespace.name.clone(),
        description: namespace.description.clone(),
        quotas: namespace.quotas.clone().into(),
        status: format!("{:?}", namespace.status).to_lowercase(),
        created_at: namespace.created_at,
        updated_at: namespace.updated_at,
        quota_utilization: namespace.quota_utilization(),
        is_near_quota: namespace.is_near_quota(),
        metadata: namespace.metadata.clone(),
    }))
}

async fn list_namespaces(
    State(server): State<AdminHttpServer>,
) -> Result<Json<Vec<NamespaceInfoDto>>, AppError> {
    let manager = server.manager.read().await;

    let namespaces = manager.list_namespaces();

    let infos: Vec<NamespaceInfoDto> = namespaces
        .into_iter()
        .map(|ns| NamespaceInfoDto {
            id: ns.id.clone(),
            name: ns.name.clone(),
            description: ns.description.clone(),
            quotas: ns.quotas.clone().into(),
            status: format!("{:?}", ns.status).to_lowercase(),
            created_at: ns.created_at,
            updated_at: ns.updated_at,
            quota_utilization: ns.quota_utilization(),
            is_near_quota: ns.is_near_quota(),
            metadata: ns.metadata.clone(),
        })
        .collect();

    Ok(Json(infos))
}

async fn get_namespace(
    State(server): State<AdminHttpServer>,
    Path(namespace_id): Path<String>,
) -> Result<Json<NamespaceInfoDto>, AppError> {
    let manager = server.manager.read().await;

    let namespace = manager
        .get_namespace(&namespace_id)
        .map_err(|e| AppError::NotFound(e.to_string()))?;

    Ok(Json(NamespaceInfoDto {
        id: namespace.id.clone(),
        name: namespace.name.clone(),
        description: namespace.description.clone(),
        quotas: namespace.quotas.clone().into(),
        status: format!("{:?}", namespace.status).to_lowercase(),
        created_at: namespace.created_at,
        updated_at: namespace.updated_at,
        quota_utilization: namespace.quota_utilization(),
        is_near_quota: namespace.is_near_quota(),
        metadata: namespace.metadata.clone(),
    }))
}

async fn update_quotas(
    State(server): State<AdminHttpServer>,
    Path(namespace_id): Path<String>,
    Json(quotas): Json<NamespaceQuotasDto>,
) -> Result<Json<serde_json::Value>, AppError> {
    let manager = server.manager.write().await;

    manager
        .update_quotas(&namespace_id, quotas.into())
        .map_err(|e| AppError::Internal(e.to_string()))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Quotas updated successfully"
    })))
}

async fn update_status(
    State(server): State<AdminHttpServer>,
    Path(namespace_id): Path<String>,
    Json(req): Json<UpdateStatusRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let status = match req.status.to_lowercase().as_str() {
        "active" => NamespaceStatus::Active,
        "suspended" => NamespaceStatus::Suspended,
        "read_only" | "readonly" => NamespaceStatus::ReadOnly,
        "pending_deletion" => NamespaceStatus::PendingDeletion,
        _ => {
            return Err(AppError::BadRequest(format!(
                "Invalid status: {}",
                req.status
            )))
        }
    };

    let manager = server.manager.write().await;

    manager
        .update_status(&namespace_id, status)
        .map_err(|e| AppError::Internal(e.to_string()))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!("Status updated to {:?}", status)
    })))
}

async fn delete_namespace(
    State(server): State<AdminHttpServer>,
    Path(namespace_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let manager = server.manager.write().await;

    manager
        .delete_namespace(&namespace_id)
        .map_err(|e| AppError::Internal(e.to_string()))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!("Namespace '{}' deleted", namespace_id)
    })))
}

async fn get_namespace_stats(
    State(server): State<AdminHttpServer>,
    Path(namespace_id): Path<String>,
) -> Result<Json<NamespaceStatsDto>, AppError> {
    let manager = server.manager.read().await;

    let stats = manager
        .get_stats(&namespace_id)
        .map_err(|e| AppError::NotFound(e.to_string()))?;

    Ok(Json(NamespaceStatsDto {
        namespace_id: stats.namespace_id,
        vector_count: stats.vector_count,
        active_count: stats.active_count,
        deleted_count: stats.deleted_count,
        dimension: stats.dimension,
        quota_utilization: stats.quota_utilization,
        total_requests: stats.total_requests,
        total_queries: stats.total_queries,
        total_upserts: stats.total_upserts,
        total_deletes: stats.total_deletes,
        status: format!("{:?}", stats.status).to_lowercase(),
    }))
}

async fn get_aggregate_stats(
    State(server): State<AdminHttpServer>,
) -> Result<Json<AggregateStatsDto>, AppError> {
    let manager = server.manager.read().await;

    let stats = manager.get_aggregate_stats();

    Ok(Json(AggregateStatsDto {
        total_namespaces: stats.total_namespaces,
        active_namespaces: stats.active_namespaces,
        total_vectors: stats.total_vectors,
        total_requests: stats.total_requests,
    }))
}

// ============================================================================
// Error handling
// ============================================================================

#[derive(Debug)]
enum AppError {
    NotFound(String),
    BadRequest(String),
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(serde_json::json!({
            "error": message
        }));

        (status, body).into_response()
    }
}

// ============================================================================
// Health checks
// ============================================================================

async fn health_check(State(server): State<AdminHttpServer>) -> Json<serde_json::Value> {
    let manager = server.manager.read().await;
    let stats = manager.get_aggregate_stats();

    Json(serde_json::json!({
        "status": "healthy",
        "mode": "multi-tenant",
        "total_namespaces": stats.total_namespaces,
        "active_namespaces": stats.active_namespaces,
    }))
}

async fn ready_check(State(server): State<AdminHttpServer>) -> Json<serde_json::Value> {
    let manager = server.manager.read().await;
    let stats = manager.get_aggregate_stats();

    Json(serde_json::json!({
        "ready": true,
        "total_namespaces": stats.total_namespaces,
    }))
}
