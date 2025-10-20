//! Namespace manager for multi-tenant vector database
//!
//! Manages multiple isolated VecStore instances, one per namespace,
//! with quota enforcement and resource management.

use crate::namespace::{Namespace, NamespaceId, NamespaceQuotas, NamespaceStatus};
use crate::store::{Metadata, Neighbor, Query, VecStore};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Multi-tenant namespace manager
pub struct NamespaceManager {
    /// Root directory for all namespaces
    root_path: PathBuf,

    /// Active namespaces and their metadata
    namespaces: Arc<RwLock<HashMap<NamespaceId, Namespace>>>,

    /// VecStore instances per namespace
    stores: Arc<RwLock<HashMap<NamespaceId, VecStore>>>,

    /// Default quotas for new namespaces
    default_quotas: NamespaceQuotas,
}

impl NamespaceManager {
    /// Create a new namespace manager
    pub fn new<P: AsRef<Path>>(root_path: P) -> Result<Self> {
        let root_path = root_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&root_path)?;

        Ok(Self {
            root_path,
            namespaces: Arc::new(RwLock::new(HashMap::new())),
            stores: Arc::new(RwLock::new(HashMap::new())),
            default_quotas: NamespaceQuotas::default(),
        })
    }

    /// Create a new namespace manager with custom default quotas
    pub fn with_quotas<P: AsRef<Path>>(
        root_path: P,
        default_quotas: NamespaceQuotas,
    ) -> Result<Self> {
        let mut manager = Self::new(root_path)?;
        manager.default_quotas = default_quotas;
        Ok(manager)
    }

    /// Load existing namespaces from disk
    pub fn load_namespaces(&self) -> Result<Vec<NamespaceId>> {
        let mut loaded = Vec::new();

        for entry in std::fs::read_dir(&self.root_path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let ns_id = entry.file_name().to_string_lossy().to_string();
                let ns_path = entry.path();

                // Load namespace metadata
                let metadata_path = ns_path.join("namespace.json");
                if metadata_path.exists() {
                    let metadata = std::fs::read_to_string(&metadata_path)?;
                    let namespace: Namespace = serde_json::from_str(&metadata)?;

                    // Load VecStore
                    let store = VecStore::open(&ns_path)?;

                    let mut namespaces = self.namespaces.write().unwrap();
                    let mut stores = self.stores.write().unwrap();

                    namespaces.insert(ns_id.clone(), namespace);
                    stores.insert(ns_id.clone(), store);

                    loaded.push(ns_id);
                }
            }
        }

        Ok(loaded)
    }

    /// Create a new namespace
    pub fn create_namespace(
        &self,
        id: NamespaceId,
        name: String,
        quotas: Option<NamespaceQuotas>,
    ) -> Result<()> {
        let namespaces = self.namespaces.read().unwrap();
        if namespaces.contains_key(&id) {
            return Err(anyhow!("Namespace already exists: {}", id));
        }
        drop(namespaces);

        let quotas = quotas.unwrap_or_else(|| self.default_quotas.clone());
        let namespace = Namespace::new(id.clone(), name, quotas);

        // Create namespace directory
        let ns_path = self.root_path.join(&id);
        std::fs::create_dir_all(&ns_path)?;

        // Save namespace metadata
        let metadata_path = ns_path.join("namespace.json");
        let metadata = serde_json::to_string_pretty(&namespace)?;
        std::fs::write(metadata_path, metadata)?;

        // Create VecStore for this namespace
        let store = VecStore::open(&ns_path)?;

        let mut namespaces = self.namespaces.write().unwrap();
        let mut stores = self.stores.write().unwrap();

        namespaces.insert(id.clone(), namespace);
        stores.insert(id, store);

        Ok(())
    }

    /// Get namespace metadata
    pub fn get_namespace(&self, id: &NamespaceId) -> Result<Namespace> {
        let namespaces = self.namespaces.read().unwrap();
        namespaces
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("Namespace not found: {}", id))
    }

    /// List all namespaces
    pub fn list_namespaces(&self) -> Vec<Namespace> {
        let namespaces = self.namespaces.read().unwrap();
        namespaces.values().cloned().collect()
    }

    /// Update namespace quotas
    pub fn update_quotas(&self, id: &NamespaceId, quotas: NamespaceQuotas) -> Result<()> {
        let mut namespaces = self.namespaces.write().unwrap();
        let namespace = namespaces
            .get_mut(id)
            .ok_or_else(|| anyhow!("Namespace not found: {}", id))?;

        namespace.update_quotas(quotas);

        // Persist metadata
        let ns_path = self.root_path.join(id);
        let metadata_path = ns_path.join("namespace.json");
        let metadata = serde_json::to_string_pretty(namespace)?;
        std::fs::write(metadata_path, metadata)?;

        Ok(())
    }

    /// Update namespace status
    pub fn update_status(&self, id: &NamespaceId, status: NamespaceStatus) -> Result<()> {
        let mut namespaces = self.namespaces.write().unwrap();
        let namespace = namespaces
            .get_mut(id)
            .ok_or_else(|| anyhow!("Namespace not found: {}", id))?;

        namespace.set_status(status);

        // Persist metadata
        let ns_path = self.root_path.join(id);
        let metadata_path = ns_path.join("namespace.json");
        let metadata = serde_json::to_string_pretty(namespace)?;
        std::fs::write(metadata_path, metadata)?;

        Ok(())
    }

    /// Delete a namespace
    pub fn delete_namespace(&self, id: &NamespaceId) -> Result<()> {
        // Mark as pending deletion first
        self.update_status(id, NamespaceStatus::PendingDeletion)?;

        // Remove from in-memory maps
        let mut namespaces = self.namespaces.write().unwrap();
        let mut stores = self.stores.write().unwrap();

        namespaces.remove(id);
        stores.remove(id);

        // Delete directory
        let ns_path = self.root_path.join(id);
        std::fs::remove_dir_all(ns_path)?;

        Ok(())
    }

    /// Upsert a vector in a namespace
    pub fn upsert(
        &self,
        namespace_id: &NamespaceId,
        id: String,
        vector: Vec<f32>,
        metadata: Metadata,
    ) -> Result<()> {
        // Check namespace status and quotas
        {
            let mut namespaces = self.namespaces.write().unwrap();
            let namespace = namespaces
                .get_mut(namespace_id)
                .ok_or_else(|| anyhow!("Namespace not found: {}", namespace_id))?;

            namespace.can_upsert(1)?;
            namespace.usage.record_request(&namespace.quotas)?;
            namespace.usage.total_upserts += 1;
        }

        // Perform upsert
        let mut stores = self.stores.write().unwrap();
        let store = stores
            .get_mut(namespace_id)
            .ok_or_else(|| anyhow!("Store not found for namespace: {}", namespace_id))?;

        store.upsert(id, vector, metadata)?;

        // Persist changes to disk (Critical Issue #1 fix)
        store.save()?;

        // Update usage stats
        {
            let mut namespaces = self.namespaces.write().unwrap();
            if let Some(namespace) = namespaces.get_mut(namespace_id) {
                namespace.usage.vector_count = store.len();
                // Note: storage_bytes would need to be calculated from disk usage
            }
        }

        Ok(())
    }

    /// Query vectors in a namespace
    pub fn query(&self, namespace_id: &NamespaceId, query: Query) -> Result<Vec<Neighbor>> {
        // Check namespace status and quotas
        {
            let mut namespaces = self.namespaces.write().unwrap();
            let namespace = namespaces
                .get_mut(namespace_id)
                .ok_or_else(|| anyhow!("Namespace not found: {}", namespace_id))?;

            namespace.can_query(query.k)?;
            namespace.usage.record_request(&namespace.quotas)?;
            namespace.usage.start_query();
        }

        // Perform query
        let result = {
            let stores = self.stores.read().unwrap();
            let store = stores
                .get(namespace_id)
                .ok_or_else(|| anyhow!("Store not found for namespace: {}", namespace_id))?;

            store.query(query)
        };

        // Update usage stats
        {
            let mut namespaces = self.namespaces.write().unwrap();
            if let Some(namespace) = namespaces.get_mut(namespace_id) {
                namespace.usage.end_query();
            }
        }

        result
    }

    /// Delete a vector from a namespace
    pub fn remove(&self, namespace_id: &NamespaceId, id: &str) -> Result<()> {
        // Check namespace status
        {
            let mut namespaces = self.namespaces.write().unwrap();
            let namespace = namespaces
                .get_mut(namespace_id)
                .ok_or_else(|| anyhow!("Namespace not found: {}", namespace_id))?;

            if namespace.status != NamespaceStatus::Active {
                return Err(anyhow!("Namespace is not active"));
            }

            namespace.usage.record_request(&namespace.quotas)?;
            namespace.usage.total_deletes += 1;
        }

        // Perform delete
        let mut stores = self.stores.write().unwrap();
        let store = stores
            .get_mut(namespace_id)
            .ok_or_else(|| anyhow!("Store not found for namespace: {}", namespace_id))?;

        store.remove(id)?;

        // Persist changes to disk (Critical Issue #1 fix)
        store.save()?;

        // Update usage stats
        {
            let mut namespaces = self.namespaces.write().unwrap();
            if let Some(namespace) = namespaces.get_mut(namespace_id) {
                namespace.usage.vector_count = store.len();
            }
        }

        Ok(())
    }

    /// Get statistics for a namespace
    pub fn get_stats(&self, namespace_id: &NamespaceId) -> Result<NamespaceStats> {
        let namespaces = self.namespaces.read().unwrap();
        let stores = self.stores.read().unwrap();

        let namespace = namespaces
            .get(namespace_id)
            .ok_or_else(|| anyhow!("Namespace not found: {}", namespace_id))?;

        let store = stores
            .get(namespace_id)
            .ok_or_else(|| anyhow!("Store not found for namespace: {}", namespace_id))?;

        Ok(NamespaceStats {
            namespace_id: namespace_id.clone(),
            vector_count: store.len(),
            active_count: store.active_count(),
            deleted_count: store.deleted_count(),
            dimension: store.dimension(),
            quota_utilization: namespace.quota_utilization(),
            total_requests: namespace.usage.total_requests,
            total_queries: namespace.usage.total_queries,
            total_upserts: namespace.usage.total_upserts,
            total_deletes: namespace.usage.total_deletes,
            status: namespace.status,
        })
    }

    /// Get aggregate stats across all namespaces
    pub fn get_aggregate_stats(&self) -> AggregateStats {
        let namespaces = self.namespaces.read().unwrap();
        let stores = self.stores.read().unwrap();

        let total_namespaces = namespaces.len();
        let mut total_vectors = 0;
        let mut total_requests = 0;
        let mut active_namespaces = 0;

        for (ns_id, namespace) in namespaces.iter() {
            if namespace.status == NamespaceStatus::Active {
                active_namespaces += 1;
            }

            total_requests += namespace.usage.total_requests;

            if let Some(store) = stores.get(ns_id) {
                total_vectors += store.len();
            }
        }

        AggregateStats {
            total_namespaces,
            active_namespaces,
            total_vectors,
            total_requests,
        }
    }

    /// Persist all namespace metadata
    pub fn save_all(&self) -> Result<()> {
        let namespaces = self.namespaces.read().unwrap();

        for (id, namespace) in namespaces.iter() {
            let ns_path = self.root_path.join(id);
            let metadata_path = ns_path.join("namespace.json");
            let metadata = serde_json::to_string_pretty(namespace)?;
            std::fs::write(metadata_path, metadata)?;
        }

        Ok(())
    }

    /// Create a backup of a namespace
    ///
    /// This creates a snapshot of the namespace's VecStore and saves it.
    ///
    /// # Arguments
    /// * `namespace_id` - ID of the namespace to backup
    /// * `backup_name` - Name for the backup
    ///
    /// # Returns
    /// * `Ok(())` if backup was created successfully
    /// * `Err` if namespace doesn't exist or backup fails
    pub fn backup_namespace(&self, namespace_id: &NamespaceId, backup_name: &str) -> Result<()> {
        let mut namespace_path = self.root_path.clone();
        namespace_path.push(namespace_id);

        if !namespace_path.exists() {
            return Err(anyhow::anyhow!("Namespace '{}' not found", namespace_id));
        }

        // Use VecStore's snapshot functionality
        let store = VecStore::open(&namespace_path)?;
        store.create_snapshot(backup_name)?;

        Ok(())
    }

    /// Restore a namespace from a backup
    ///
    /// This restores a namespace from a previously created snapshot.
    ///
    /// # Arguments
    /// * `namespace_id` - ID of the namespace to restore
    /// * `backup_name` - Name of the backup to restore from
    ///
    /// # Returns
    /// * `Ok(())` if restore was successful
    /// * `Err` if namespace doesn't exist or restore fails
    pub fn restore_namespace(&self, namespace_id: &NamespaceId, backup_name: &str) -> Result<()> {
        let mut namespace_path = self.root_path.clone();
        namespace_path.push(namespace_id);

        if !namespace_path.exists() {
            return Err(anyhow::anyhow!("Namespace '{}' not found", namespace_id));
        }

        let mut store = VecStore::open(&namespace_path)?;
        store.restore_snapshot(backup_name)?;
        store.save()?;

        Ok(())
    }

    /// List available backups for a namespace
    ///
    /// # Arguments
    /// * `namespace_id` - ID of the namespace
    ///
    /// # Returns
    /// * List of (snapshot_name, created_at, record_count) tuples
    pub fn list_namespace_backups(
        &self,
        namespace_id: &NamespaceId,
    ) -> Result<Vec<(String, String, usize)>> {
        let mut namespace_path = self.root_path.clone();
        namespace_path.push(namespace_id);

        if !namespace_path.exists() {
            return Err(anyhow::anyhow!("Namespace '{}' not found", namespace_id));
        }

        let store = VecStore::open(&namespace_path)?;
        store.list_snapshots()
    }

    /// Delete a backup for a namespace
    ///
    /// # Arguments
    /// * `namespace_id` - ID of the namespace
    /// * `backup_name` - Name of the backup to delete
    pub fn delete_namespace_backup(
        &self,
        namespace_id: &NamespaceId,
        backup_name: &str,
    ) -> Result<()> {
        let mut namespace_path = self.root_path.clone();
        namespace_path.push(namespace_id);

        if !namespace_path.exists() {
            return Err(anyhow::anyhow!("Namespace '{}' not found", namespace_id));
        }

        let store = VecStore::open(&namespace_path)?;
        store.delete_snapshot(backup_name)
    }
}

/// Statistics for a single namespace
#[derive(Debug, Clone)]
pub struct NamespaceStats {
    pub namespace_id: NamespaceId,
    pub vector_count: usize,
    pub active_count: usize,
    pub deleted_count: usize,
    pub dimension: usize,
    pub quota_utilization: f64,
    pub total_requests: u64,
    pub total_queries: u64,
    pub total_upserts: u64,
    pub total_deletes: u64,
    pub status: NamespaceStatus,
}

/// Aggregate statistics across all namespaces
#[derive(Debug, Clone)]
pub struct AggregateStats {
    pub total_namespaces: usize,
    pub active_namespaces: usize,
    pub total_vectors: usize,
    pub total_requests: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_namespace() {
        let temp_dir = TempDir::new().unwrap();
        let manager = NamespaceManager::new(temp_dir.path()).unwrap();

        manager
            .create_namespace("test-ns".to_string(), "Test Namespace".to_string(), None)
            .unwrap();

        let ns = manager.get_namespace(&"test-ns".to_string()).unwrap();
        assert_eq!(ns.id, "test-ns");
        assert_eq!(ns.status, NamespaceStatus::Active);
    }

    #[test]
    fn test_namespace_isolation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = NamespaceManager::new(temp_dir.path()).unwrap();

        manager
            .create_namespace("ns1".to_string(), "NS1".to_string(), None)
            .unwrap();
        manager
            .create_namespace("ns2".to_string(), "NS2".to_string(), None)
            .unwrap();

        // Insert into ns1
        let metadata = Metadata {
            fields: std::collections::HashMap::new(),
        };
        manager
            .upsert(
                &"ns1".to_string(),
                "vec1".to_string(),
                vec![0.1, 0.2],
                metadata.clone(),
            )
            .unwrap();

        // Query ns1 - should find vector
        let query = Query {
            vector: vec![0.1, 0.2],
            k: 10,
            filter: None,
        };
        let results = manager.query(&"ns1".to_string(), query.clone()).unwrap();
        assert_eq!(results.len(), 1);

        // Query ns2 - should be empty
        let results = manager.query(&"ns2".to_string(), query).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_quota_enforcement() {
        let temp_dir = TempDir::new().unwrap();
        let manager = NamespaceManager::new(temp_dir.path()).unwrap();

        let mut quotas = NamespaceQuotas::default();
        quotas.max_vectors = Some(2);

        manager
            .create_namespace("limited".to_string(), "Limited".to_string(), Some(quotas))
            .unwrap();

        let metadata = Metadata {
            fields: std::collections::HashMap::new(),
        };

        // Should succeed
        manager
            .upsert(
                &"limited".to_string(),
                "vec1".to_string(),
                vec![0.1],
                metadata.clone(),
            )
            .unwrap();

        manager
            .upsert(
                &"limited".to_string(),
                "vec2".to_string(),
                vec![0.2],
                metadata.clone(),
            )
            .unwrap();

        // Should fail - quota exceeded
        let result = manager.upsert(
            &"limited".to_string(),
            "vec3".to_string(),
            vec![0.3],
            metadata,
        );
        assert!(result.is_err());
    }
}
