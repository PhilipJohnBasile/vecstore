//! Vector partitioning for data isolation and multi-tenancy
//!
//! This module provides partitioning capabilities to organize vectors into
//! isolated groups based on metadata. This is essential for:
//! - Multi-tenant applications (isolate by customer/org)
//! - Data organization (group by category, date, etc.)
//! - Performance optimization (smaller search spaces)
//! - Access control (restrict queries to specific partitions)
//!
//! # Features
//!
//! - **Automatic partitioning**: Route vectors based on metadata
//! - **Partition isolation**: Each partition has its own storage
//! - **Cross-partition queries**: Optional queries across partitions
//! - **Partition statistics**: Monitor size and performance per partition
//! - **Dynamic creation**: Partitions created on-demand
//!
//! # Example
//!
//! ```rust
//! use vecstore::partitioning::{PartitionedStore, PartitionConfig};
//!
//! let config = PartitionConfig {
//!     partition_field: "tenant_id".to_string(),
//!     auto_create: true,
//! };
//!
//! let mut store = PartitionedStore::new("data", config)?;
//!
//! // Insert with partition key
//! store.insert("doc1", vec![0.1, 0.2], metadata, "tenant_123")?;
//!
//! // Query within partition
//! let results = store.query_partition("tenant_123", query)?;
//! ```

use crate::store::{Metadata, Neighbor, Query, VecStore};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Partition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Metadata field to use for partitioning
    pub partition_field: String,
    /// Automatically create partitions on first insert
    pub auto_create: bool,
    /// Maximum vectors per partition (optional)
    pub max_vectors_per_partition: Option<usize>,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            partition_field: "partition".to_string(),
            auto_create: true,
            max_vectors_per_partition: None,
        }
    }
}

/// Partition metadata and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionInfo {
    /// Partition key/ID
    pub id: String,
    /// Number of vectors in partition
    pub vector_count: usize,
    /// Storage path
    pub path: PathBuf,
    /// Creation timestamp
    pub created_at: std::time::SystemTime,
    /// Last modified timestamp
    pub modified_at: std::time::SystemTime,
    /// Total size in bytes (estimate)
    pub size_bytes: u64,
}

/// Partitioned vector store
pub struct PartitionedStore {
    /// Base directory for all partitions
    base_path: PathBuf,
    /// Configuration
    config: PartitionConfig,
    /// Map of partition ID to VecStore
    partitions: HashMap<String, VecStore>,
    /// Partition metadata
    partition_info: HashMap<String, PartitionInfo>,
}

impl PartitionedStore {
    /// Create new partitioned store
    pub fn new<P: AsRef<Path>>(base_path: P, config: PartitionConfig) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Create base directory if it doesn't exist
        std::fs::create_dir_all(&base_path)
            .map_err(|e| anyhow::anyhow!("Failed to create directory: {}", e))?;

        let mut store = Self {
            base_path,
            config,
            partitions: HashMap::new(),
            partition_info: HashMap::new(),
        };

        // Load existing partitions
        store.load_existing_partitions()?;

        Ok(store)
    }

    /// Load existing partitions from disk
    fn load_existing_partitions(&mut self) -> Result<()> {
        let entries = std::fs::read_dir(&self.base_path)
            .map_err(|e| anyhow::anyhow!("Failed to read directory: {}", e))?;

        for entry in entries {
            let entry = entry.map_err(|e| anyhow::anyhow!("Read error: {}", e))?;
            let path = entry.path();

            if path.is_dir() {
                let partition_id = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();

                // Try to open the partition
                let db_path = path.join("vectors.db");
                if db_path.exists() {
                    match VecStore::open(&db_path) {
                        Ok(store) => {
                            let vector_count = store.len();
                            let metadata = std::fs::metadata(&db_path)
                                .map_err(|e| anyhow::anyhow!("Metadata error: {}", e))?;

                            let info = PartitionInfo {
                                id: partition_id.clone(),
                                vector_count,
                                path: path.clone(),
                                created_at: metadata
                                    .created()
                                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                                modified_at: metadata
                                    .modified()
                                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                                size_bytes: metadata.len(),
                            };

                            self.partitions.insert(partition_id.clone(), store);
                            self.partition_info.insert(partition_id, info);
                        }
                        Err(_) => {
                            // Skip corrupted partitions
                            continue;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get or create a partition
    fn get_or_create_partition(&mut self, partition_id: &str) -> Result<&mut VecStore> {
        if !self.partitions.contains_key(partition_id) {
            if !self.config.auto_create {
                return Err(anyhow::anyhow!(
                    "Partition '{}' does not exist and auto_create is disabled",
                    partition_id
                ));
            }

            // Create new partition
            let partition_path = self.base_path.join(partition_id);
            std::fs::create_dir_all(&partition_path)
                .map_err(|e| anyhow::anyhow!("Failed to create partition dir: {}", e))?;

            let db_path = partition_path.join("vectors.db");
            let store = VecStore::open(&db_path)?;

            let info = PartitionInfo {
                id: partition_id.to_string(),
                vector_count: 0,
                path: partition_path,
                created_at: std::time::SystemTime::now(),
                modified_at: std::time::SystemTime::now(),
                size_bytes: 0,
            };

            self.partitions.insert(partition_id.to_string(), store);
            self.partition_info.insert(partition_id.to_string(), info);
        }

        Ok(self.partitions.get_mut(partition_id).unwrap())
    }

    /// Insert vector into specific partition
    pub fn insert(
        &mut self,
        partition_id: &str,
        id: String,
        vector: Vec<f32>,
        metadata: Metadata,
    ) -> Result<()> {
        // Check partition size limit
        if let Some(max_size) = self.config.max_vectors_per_partition {
            if let Some(info) = self.partition_info.get(partition_id) {
                if info.vector_count >= max_size {
                    return Err(anyhow::anyhow!(
                        "Partition '{}' has reached maximum size of {} vectors",
                        partition_id,
                        max_size
                    ));
                }
            }
        }

        let partition = self.get_or_create_partition(partition_id)?;
        partition.upsert(id, vector, metadata)?;
        let new_count = partition.len();

        // Update partition info (after dropping partition reference)
        if let Some(info) = self.partition_info.get_mut(partition_id) {
            info.vector_count = new_count;
            info.modified_at = std::time::SystemTime::now();
        }

        Ok(())
    }

    /// Query within a specific partition
    pub fn query_partition(&self, partition_id: &str, query: Query) -> Result<Vec<Neighbor>> {
        let partition = self
            .partitions
            .get(partition_id)
            .ok_or_else(|| anyhow::anyhow!("Partition '{}' not found", partition_id))?;

        partition.query(query)
    }

    /// Query across all partitions (slower but comprehensive)
    pub fn query_all(&self, query: Query, limit: usize) -> Result<Vec<Neighbor>> {
        let mut all_results = Vec::new();

        for partition in self.partitions.values() {
            let results = partition.query(query.clone())?;
            all_results.extend(results);
        }

        // Sort by distance and limit
        all_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        all_results.truncate(limit);

        Ok(all_results)
    }

    /// Query across specific partitions
    pub fn query_partitions(
        &self,
        partition_ids: &[&str],
        query: Query,
        limit: usize,
    ) -> Result<Vec<Neighbor>> {
        let mut all_results = Vec::new();

        for &partition_id in partition_ids {
            if let Some(partition) = self.partitions.get(partition_id) {
                let results = partition.query(query.clone())?;
                all_results.extend(results);
            }
        }

        // Sort by distance and limit
        all_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        all_results.truncate(limit);

        Ok(all_results)
    }

    /// Delete a vector from a partition
    pub fn delete(&mut self, partition_id: &str, id: &str) -> Result<()> {
        let partition = self
            .partitions
            .get_mut(partition_id)
            .ok_or_else(|| anyhow::anyhow!("Partition '{}' not found", partition_id))?;

        partition.delete(id)?;

        // Update partition info
        if let Some(info) = self.partition_info.get_mut(partition_id) {
            info.vector_count = partition.len();
            info.modified_at = std::time::SystemTime::now();
        }

        Ok(())
    }

    /// Get partition information
    pub fn get_partition_info(&self, partition_id: &str) -> Option<&PartitionInfo> {
        self.partition_info.get(partition_id)
    }

    /// List all partitions
    pub fn list_partitions(&self) -> Vec<&PartitionInfo> {
        self.partition_info.values().collect()
    }

    /// Get total vector count across all partitions
    pub fn total_vectors(&self) -> usize {
        self.partition_info
            .values()
            .map(|info| info.vector_count)
            .sum()
    }

    /// Delete a partition
    pub fn delete_partition(&mut self, partition_id: &str) -> Result<()> {
        // Remove from memory
        self.partitions.remove(partition_id);

        if let Some(info) = self.partition_info.remove(partition_id) {
            // Delete from disk
            std::fs::remove_dir_all(&info.path)
                .map_err(|e| anyhow::anyhow!("Failed to delete partition: {}", e))?;
        }

        Ok(())
    }

    /// Compact a partition (remove deleted vectors)
    pub fn compact_partition(&mut self, partition_id: &str) -> Result<usize> {
        let partition = self
            .partitions
            .get_mut(partition_id)
            .ok_or_else(|| anyhow::anyhow!("Partition '{}' not found", partition_id))?;

        let removed = partition.compact()?;

        // Update partition info
        if let Some(info) = self.partition_info.get_mut(partition_id) {
            info.vector_count = partition.len();
            info.modified_at = std::time::SystemTime::now();
        }

        Ok(removed)
    }

    /// Get partition statistics
    pub fn partition_stats(&self) -> PartitionStats {
        let total_partitions = self.partitions.len();
        let total_vectors = self.total_vectors();
        let avg_vectors_per_partition = if total_partitions > 0 {
            total_vectors as f64 / total_partitions as f64
        } else {
            0.0
        };

        let mut largest_partition = None;
        let mut smallest_partition = None;
        let mut max_size = 0;
        let mut min_size = usize::MAX;

        for info in self.partition_info.values() {
            if info.vector_count > max_size {
                max_size = info.vector_count;
                largest_partition = Some(info.id.clone());
            }
            if info.vector_count < min_size {
                min_size = info.vector_count;
                smallest_partition = Some(info.id.clone());
            }
        }

        PartitionStats {
            total_partitions,
            total_vectors,
            avg_vectors_per_partition,
            largest_partition,
            smallest_partition,
            max_partition_size: max_size,
            min_partition_size: if min_size == usize::MAX { 0 } else { min_size },
        }
    }
}

/// Partition statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionStats {
    /// Total number of partitions
    pub total_partitions: usize,
    /// Total vectors across all partitions
    pub total_vectors: usize,
    /// Average vectors per partition
    pub avg_vectors_per_partition: f64,
    /// ID of largest partition
    pub largest_partition: Option<String>,
    /// ID of smallest partition
    pub smallest_partition: Option<String>,
    /// Size of largest partition
    pub max_partition_size: usize,
    /// Size of smallest partition
    pub min_partition_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_partitioned_store_creation() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = PartitionConfig::default();

        let store = PartitionedStore::new(temp_dir.path(), config)?;
        assert_eq!(store.list_partitions().len(), 0);

        Ok(())
    }

    #[test]
    fn test_insert_and_query() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = PartitionConfig::default();

        let mut store = PartitionedStore::new(temp_dir.path(), config)?;

        // Insert into partition A
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("partition".to_string(), serde_json::json!("A"));

        store.insert("A", "vec1".to_string(), vec![0.1, 0.2, 0.3], metadata)?;

        // Insert into partition B
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("partition".to_string(), serde_json::json!("B"));

        store.insert("B", "vec2".to_string(), vec![0.4, 0.5, 0.6], metadata)?;

        assert_eq!(store.list_partitions().len(), 2);
        assert_eq!(store.total_vectors(), 2);

        // Query partition A
        let query = Query::new(vec![0.1, 0.2, 0.3]).with_limit(10);
        let results = store.query_partition("A", query)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "vec1");

        Ok(())
    }

    #[test]
    fn test_cross_partition_query() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = PartitionConfig::default();

        let mut store = PartitionedStore::new(temp_dir.path(), config)?;

        // Insert into multiple partitions
        for i in 0..3 {
            let partition_id = format!("partition_{}", i);
            let metadata = Metadata {
                fields: HashMap::new(),
            };

            store.insert(
                &partition_id,
                format!("vec_{}", i),
                vec![i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3],
                metadata,
            )?;
        }

        // Query across all partitions
        let query = Query::new(vec![0.0, 0.0, 0.0]).with_limit(10);
        let results = store.query_all(query, 10)?;

        assert_eq!(results.len(), 3);

        Ok(())
    }

    #[test]
    fn test_partition_size_limit() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = PartitionConfig {
            partition_field: "tenant".to_string(),
            auto_create: true,
            max_vectors_per_partition: Some(2),
        };

        let mut store = PartitionedStore::new(temp_dir.path(), config)?;

        // Insert 2 vectors (should succeed)
        for i in 0..2 {
            let metadata = Metadata {
                fields: HashMap::new(),
            };
            store.insert("tenant1", format!("vec_{}", i), vec![i as f32], metadata)?;
        }

        // Try to insert 3rd vector (should fail)
        let metadata = Metadata {
            fields: HashMap::new(),
        };
        let result = store.insert("tenant1", "vec_3".to_string(), vec![3.0], metadata);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_partition_deletion() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = PartitionConfig::default();

        let mut store = PartitionedStore::new(temp_dir.path(), config)?;

        // Create partition
        let metadata = Metadata {
            fields: HashMap::new(),
        };
        store.insert("test_partition", "vec1".to_string(), vec![1.0], metadata)?;

        assert_eq!(store.list_partitions().len(), 1);

        // Delete partition
        store.delete_partition("test_partition")?;
        assert_eq!(store.list_partitions().len(), 0);

        Ok(())
    }

    #[test]
    fn test_partition_stats() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = PartitionConfig::default();

        let mut store = PartitionedStore::new(temp_dir.path(), config)?;

        // Create multiple partitions with different sizes
        let metadata = Metadata {
            fields: HashMap::new(),
        };
        store.insert("small", "vec1".to_string(), vec![1.0], metadata.clone())?;

        store.insert("large", "vec2".to_string(), vec![2.0], metadata.clone())?;
        store.insert("large", "vec3".to_string(), vec![3.0], metadata.clone())?;
        store.insert("large", "vec4".to_string(), vec![4.0], metadata)?;

        let stats = store.partition_stats();
        assert_eq!(stats.total_partitions, 2);
        assert_eq!(stats.total_vectors, 4);
        assert_eq!(stats.max_partition_size, 3);
        assert_eq!(stats.min_partition_size, 1);

        Ok(())
    }
}
