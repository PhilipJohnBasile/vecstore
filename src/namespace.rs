//! Multi-tenant namespace support with quotas and resource limits
//!
//! Provides namespace isolation, per-namespace quotas, and resource management
//! for SaaS deployments.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Namespace identifier
pub type NamespaceId = String;

/// Namespace metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Namespace {
    /// Unique namespace identifier
    pub id: NamespaceId,

    /// Human-readable name
    pub name: String,

    /// Optional description
    pub description: Option<String>,

    /// Resource quotas
    pub quotas: NamespaceQuotas,

    /// Current resource usage
    pub usage: ResourceUsage,

    /// Namespace status
    pub status: NamespaceStatus,

    /// Creation timestamp (Unix epoch)
    pub created_at: u64,

    /// Last update timestamp
    pub updated_at: u64,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Namespace status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NamespaceStatus {
    /// Namespace is active and accepting requests
    Active,

    /// Namespace is suspended (quota exceeded or admin action)
    Suspended,

    /// Namespace is read-only (no writes allowed)
    ReadOnly,

    /// Namespace is marked for deletion
    PendingDeletion,
}

/// Resource quotas for a namespace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceQuotas {
    /// Maximum number of vectors
    pub max_vectors: Option<usize>,

    /// Maximum storage in bytes
    pub max_storage_bytes: Option<u64>,

    /// Maximum requests per second
    pub max_requests_per_second: Option<f64>,

    /// Maximum concurrent queries
    pub max_concurrent_queries: Option<usize>,

    /// Maximum vector dimension
    pub max_dimension: Option<usize>,

    /// Maximum results per query
    pub max_results_per_query: Option<usize>,

    /// Maximum batch size for upserts
    pub max_batch_size: Option<usize>,
}

impl Default for NamespaceQuotas {
    fn default() -> Self {
        Self {
            max_vectors: Some(1_000_000),                     // 1M vectors
            max_storage_bytes: Some(10 * 1024 * 1024 * 1024), // 10GB
            max_requests_per_second: Some(100.0),             // 100 req/s
            max_concurrent_queries: Some(10),                 // 10 concurrent queries
            max_dimension: Some(4096),                        // Max 4K dimensions
            max_results_per_query: Some(1000),                // Max 1K results
            max_batch_size: Some(1000),                       // Max 1K batch
        }
    }
}

impl NamespaceQuotas {
    /// Create unlimited quotas (for admin/premium namespaces)
    pub fn unlimited() -> Self {
        Self {
            max_vectors: None,
            max_storage_bytes: None,
            max_requests_per_second: None,
            max_concurrent_queries: None,
            max_dimension: None,
            max_results_per_query: None,
            max_batch_size: None,
        }
    }

    /// Create quotas for free tier
    pub fn free_tier() -> Self {
        Self {
            max_vectors: Some(10_000),                  // 10K vectors
            max_storage_bytes: Some(100 * 1024 * 1024), // 100MB
            max_requests_per_second: Some(10.0),        // 10 req/s
            max_concurrent_queries: Some(2),            // 2 concurrent
            max_dimension: Some(1536),                  // OpenAI embedding size
            max_results_per_query: Some(100),           // 100 results
            max_batch_size: Some(100),                  // 100 batch
        }
    }

    /// Create quotas for pro tier
    pub fn pro_tier() -> Self {
        Self {
            max_vectors: Some(1_000_000),                     // 1M vectors
            max_storage_bytes: Some(10 * 1024 * 1024 * 1024), // 10GB
            max_requests_per_second: Some(100.0),             // 100 req/s
            max_concurrent_queries: Some(20),                 // 20 concurrent
            max_dimension: Some(4096),                        // Large embeddings
            max_results_per_query: Some(1000),                // 1K results
            max_batch_size: Some(1000),                       // 1K batch
        }
    }
}

/// Current resource usage for a namespace
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Current number of vectors
    pub vector_count: usize,

    /// Current storage usage in bytes
    pub storage_bytes: u64,

    /// Total requests (all time)
    pub total_requests: u64,

    /// Requests in current window (for rate limiting)
    pub requests_current_window: usize,

    /// Window start time (Unix epoch)
    pub window_start: u64,

    /// Currently active queries
    pub active_queries: usize,

    /// Total queries executed
    pub total_queries: u64,

    /// Total upserts executed
    pub total_upserts: u64,

    /// Total deletes executed
    pub total_deletes: u64,

    /// Last request timestamp
    pub last_request_at: Option<u64>,
}

impl ResourceUsage {
    /// Check if current usage exceeds any quota limits
    pub fn check_quotas(&self, quotas: &NamespaceQuotas) -> Result<()> {
        // Check vector count
        if let Some(max_vectors) = quotas.max_vectors {
            if self.vector_count >= max_vectors {
                return Err(anyhow!(
                    "Vector quota exceeded: {} / {} vectors",
                    self.vector_count,
                    max_vectors
                ));
            }
        }

        // Check storage
        if let Some(max_storage) = quotas.max_storage_bytes {
            if self.storage_bytes >= max_storage {
                return Err(anyhow!(
                    "Storage quota exceeded: {} / {} bytes",
                    self.storage_bytes,
                    max_storage
                ));
            }
        }

        // Check concurrent queries
        if let Some(max_concurrent) = quotas.max_concurrent_queries {
            if self.active_queries >= max_concurrent {
                return Err(anyhow!(
                    "Concurrent query limit exceeded: {} / {}",
                    self.active_queries,
                    max_concurrent
                ));
            }
        }

        Ok(())
    }

    /// Update request rate tracking
    pub fn record_request(&mut self, quotas: &NamespaceQuotas) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Reset window if needed (1-second windows)
        if now > self.window_start {
            self.window_start = now;
            self.requests_current_window = 0;
        }

        // Check rate limit
        if let Some(max_rps) = quotas.max_requests_per_second {
            if self.requests_current_window as f64 >= max_rps {
                return Err(anyhow!("Rate limit exceeded: {} requests/second", max_rps));
            }
        }

        self.requests_current_window += 1;
        self.total_requests += 1;
        self.last_request_at = Some(now);

        Ok(())
    }

    /// Increment active query count
    pub fn start_query(&mut self) {
        self.active_queries += 1;
        self.total_queries += 1;
    }

    /// Decrement active query count
    pub fn end_query(&mut self) {
        if self.active_queries > 0 {
            self.active_queries -= 1;
        }
    }
}

impl Namespace {
    /// Create a new namespace
    pub fn new(id: NamespaceId, name: String, quotas: NamespaceQuotas) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id,
            name,
            description: None,
            quotas,
            usage: ResourceUsage::default(),
            status: NamespaceStatus::Active,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Check if namespace can accept new vectors
    pub fn can_upsert(&self, count: usize) -> Result<()> {
        if self.status != NamespaceStatus::Active {
            return Err(anyhow!("Namespace is not active: {:?}", self.status));
        }

        // Check if adding 'count' vectors would exceed quota
        if let Some(max_vectors) = self.quotas.max_vectors {
            if self.usage.vector_count + count > max_vectors {
                return Err(anyhow!(
                    "Upsert would exceed vector quota: {} + {} > {}",
                    self.usage.vector_count,
                    count,
                    max_vectors
                ));
            }
        }

        // Check batch size
        if let Some(max_batch) = self.quotas.max_batch_size {
            if count > max_batch {
                return Err(anyhow!(
                    "Batch size exceeds limit: {} > {}",
                    count,
                    max_batch
                ));
            }
        }

        Ok(())
    }

    /// Check if namespace can execute query with given parameters
    pub fn can_query(&self, k: usize) -> Result<()> {
        // Check namespace status (Major Issue #18 fix)
        match self.status {
            NamespaceStatus::PendingDeletion => {
                return Err(anyhow!("Namespace is pending deletion"));
            }
            NamespaceStatus::Suspended => {
                return Err(anyhow!(
                    "Namespace is suspended. Contact administrator to reactivate."
                ));
            }
            NamespaceStatus::Active | NamespaceStatus::ReadOnly => {
                // OK to query
            }
        }

        // Check concurrent query limit
        self.usage.check_quotas(&self.quotas)?;

        // Check result limit
        if let Some(max_results) = self.quotas.max_results_per_query {
            if k > max_results {
                return Err(anyhow!(
                    "Query limit exceeds maximum: {} > {}",
                    k,
                    max_results
                ));
            }
        }

        Ok(())
    }

    /// Update namespace status
    pub fn set_status(&mut self, status: NamespaceStatus) {
        self.status = status;
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Update quotas
    pub fn update_quotas(&mut self, quotas: NamespaceQuotas) {
        self.quotas = quotas;
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Get quota utilization percentage (0.0 - 1.0)
    /// Returns the maximum utilization across all quota types
    pub fn quota_utilization(&self) -> f64 {
        let mut utilizations = Vec::new();

        if let Some(max_vectors) = self.quotas.max_vectors {
            utilizations.push(self.usage.vector_count as f64 / max_vectors as f64);
        }

        if let Some(max_storage) = self.quotas.max_storage_bytes {
            utilizations.push(self.usage.storage_bytes as f64 / max_storage as f64);
        }

        if utilizations.is_empty() {
            0.0
        } else {
            // Use maximum utilization, not average
            // This ensures quota warnings trigger when ANY limit is near capacity
            utilizations.iter().copied().fold(0.0_f64, f64::max)
        }
    }

    /// Check if namespace is near quota limits (> 80%)
    pub fn is_near_quota(&self) -> bool {
        self.quota_utilization() > 0.8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_creation() {
        let ns = Namespace::new(
            "test-ns".to_string(),
            "Test Namespace".to_string(),
            NamespaceQuotas::free_tier(),
        );

        assert_eq!(ns.id, "test-ns");
        assert_eq!(ns.status, NamespaceStatus::Active);
        assert_eq!(ns.usage.vector_count, 0);
    }

    #[test]
    fn test_quota_enforcement() {
        let ns = Namespace::new(
            "test".to_string(),
            "Test".to_string(),
            NamespaceQuotas::free_tier(),
        );

        // Should allow upsert within quota
        assert!(ns.can_upsert(100).is_ok());

        // Should reject upsert exceeding quota
        assert!(ns.can_upsert(20_000).is_err());

        // Should reject query with too many results
        assert!(ns.can_query(200).is_err());
    }

    #[test]
    fn test_rate_limiting() {
        let quotas = NamespaceQuotas::free_tier();
        let mut usage = ResourceUsage::default();

        // Should allow requests within rate limit
        for _ in 0..10 {
            assert!(usage.record_request(&quotas).is_ok());
        }

        // Should reject request exceeding rate limit
        assert!(usage.record_request(&quotas).is_err());
    }

    #[test]
    fn test_quota_utilization() {
        let mut ns = Namespace::new(
            "test".to_string(),
            "Test".to_string(),
            NamespaceQuotas::free_tier(),
        );

        ns.usage.vector_count = 5000; // 50% of 10K quota
        let util = ns.quota_utilization();
        assert!((util - 0.5).abs() < 0.01);

        ns.usage.vector_count = 9000; // 90% of quota
        assert!(ns.is_near_quota());
    }
}
