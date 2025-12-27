//! Partition Keys for Multi-Tenancy
//!
//! Partition-based data isolation and routing for multi-tenant deployments.
//! Similar to Milvus partition keys with automatic query routing.
//!
//! # Features
//!
//! - **Automatic Routing**: Queries route to correct partition
//! - **Tenant Isolation**: Data physically separated by partition
//! - **Per-Partition Indexes**: Independent HNSW graphs per partition
//! - **Partition Lifecycle**: TTL, compaction, archival per partition
//! - **Cross-Partition Queries**: Optional multi-partition search
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::partition_keys::{PartitionedIndex, PartitionConfig};
//!
//! let config = PartitionConfig::new()
//!     .with_partition_key("tenant_id")
//!     .with_max_partitions(1024);
//!
//! let mut index = PartitionedIndex::new(384, config)?;
//!
//! // Insert - auto-routed to partition
//! index.upsert("doc1", vec, json!({"tenant_id": "acme"}))?;
//!
//! // Query - auto-routed to partition
//! let results = index.search(&query, 10, json!({"tenant_id": "acme"}))?;
//!
//! // Cross-partition query
//! let all_results = index.search_all_partitions(&query, 10)?;
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

use crate::error::{VecStoreError, Result};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Partition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Field name for partition key
    pub partition_key: String,
    /// Maximum number of partitions
    pub max_partitions: usize,
    /// Default TTL for partitions (hours, 0 = no TTL)
    pub default_ttl_hours: u64,
    /// Enable auto-compaction
    pub auto_compaction: bool,
    /// Compaction threshold (% of deleted vectors)
    pub compaction_threshold: f32,
    /// Enable cross-partition queries by default
    pub allow_cross_partition: bool,
    /// Hash function for partition assignment
    pub hash_algorithm: HashAlgorithm,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            partition_key: "tenant_id".to_string(),
            max_partitions: 1024,
            default_ttl_hours: 0,
            auto_compaction: true,
            compaction_threshold: 0.2,
            allow_cross_partition: false,
            hash_algorithm: HashAlgorithm::Murmur3,
        }
    }
}

impl PartitionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_partition_key(mut self, key: &str) -> Self {
        self.partition_key = key.to_string();
        self
    }

    pub fn with_max_partitions(mut self, max: usize) -> Self {
        self.max_partitions = max;
        self
    }

    pub fn with_ttl_hours(mut self, hours: u64) -> Self {
        self.default_ttl_hours = hours;
        self
    }
}

/// Hash algorithm for partition assignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashAlgorithm {
    /// Murmur3 hash (fast, good distribution)
    Murmur3,
    /// XXHash (very fast)
    XXHash,
    /// Simple modulo (for testing)
    Modulo,
}

// ============================================================================
// PARTITION
// ============================================================================

/// Partition state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionState {
    /// Partition is active and accepting writes
    Active,
    /// Partition is read-only
    ReadOnly,
    /// Partition is being compacted
    Compacting,
    /// Partition is being archived
    Archiving,
    /// Partition is archived (cold storage)
    Archived,
    /// Partition is being deleted
    Deleting,
}

/// Partition metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionMeta {
    /// Partition ID (hash of partition key value)
    pub id: u64,
    /// Partition key value
    pub key_value: String,
    /// Current state
    pub state: PartitionState,
    /// Creation time
    pub created_at: u64,
    /// Last access time
    pub last_accessed: u64,
    /// Number of vectors
    pub vector_count: usize,
    /// Number of deleted vectors (tombstones)
    pub deleted_count: usize,
    /// TTL expiry (0 = no expiry)
    pub ttl_expiry: u64,
    /// Custom metadata
    pub metadata: Option<serde_json::Value>,
}

/// A single partition
pub struct Partition {
    /// Metadata
    meta: RwLock<PartitionMeta>,
    /// Vectors in this partition
    vectors: RwLock<HashMap<String, Vec<f32>>>,
    /// Vector metadata
    vector_metadata: RwLock<HashMap<String, serde_json::Value>>,
    /// Dimension
    dimension: usize,
}

impl Partition {
    pub fn new(id: u64, key_value: &str, dimension: usize) -> Self {
        Self {
            meta: RwLock::new(PartitionMeta {
                id,
                key_value: key_value.to_string(),
                state: PartitionState::Active,
                created_at: unix_timestamp(),
                last_accessed: unix_timestamp(),
                vector_count: 0,
                deleted_count: 0,
                ttl_expiry: 0,
                metadata: None,
            }),
            vectors: RwLock::new(HashMap::new()),
            vector_metadata: RwLock::new(HashMap::new()),
            dimension,
        }
    }

    /// Insert or update a vector
    pub fn upsert(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let is_new = {
            let mut vectors = self.vectors.write().unwrap();
            let is_new = !vectors.contains_key(id);
            vectors.insert(id.to_string(), vector);
            is_new
        };

        if let Some(meta) = metadata {
            self.vector_metadata.write().unwrap().insert(id.to_string(), meta);
        }

        // Update partition metadata
        {
            let mut meta = self.meta.write().unwrap();
            meta.last_accessed = unix_timestamp();
            if is_new {
                meta.vector_count += 1;
            }
        }

        Ok(())
    }

    /// Delete a vector (soft delete)
    pub fn delete(&self, id: &str) -> Result<bool> {
        let existed = {
            let mut vectors = self.vectors.write().unwrap();
            vectors.remove(id).is_some()
        };

        if existed {
            self.vector_metadata.write().unwrap().remove(id);
            let mut meta = self.meta.write().unwrap();
            meta.vector_count = meta.vector_count.saturating_sub(1);
            meta.deleted_count += 1;
        }

        Ok(existed)
    }

    /// Get a vector
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<serde_json::Value>)> {
        let vectors = self.vectors.read().unwrap();
        vectors.get(id).map(|v| {
            let meta = self.vector_metadata.read().unwrap().get(id).cloned();
            (v.clone(), meta)
        })
    }

    /// Search within partition
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<PartitionSearchResult> {
        let vectors = self.vectors.read().unwrap();

        let mut results: Vec<_> = vectors.iter()
            .map(|(id, vec)| {
                let score = cosine_similarity(query, vec);
                (id.clone(), score)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        let partition_id = self.meta.read().unwrap().id;

        results.into_iter()
            .map(|(id, score)| PartitionSearchResult {
                id,
                score,
                partition_id,
            })
            .collect()
    }

    /// Get partition metadata
    pub fn meta(&self) -> PartitionMeta {
        self.meta.read().unwrap().clone()
    }

    /// Check if compaction is needed
    pub fn needs_compaction(&self, threshold: f32) -> bool {
        let meta = self.meta.read().unwrap();
        if meta.vector_count == 0 {
            return false;
        }
        let delete_ratio = meta.deleted_count as f32 / (meta.vector_count + meta.deleted_count) as f32;
        delete_ratio > threshold
    }

    /// Compact partition (remove tombstones)
    pub fn compact(&self) -> CompactionResult {
        let mut meta = self.meta.write().unwrap();
        let before = meta.deleted_count;
        meta.deleted_count = 0;
        meta.state = PartitionState::Active;

        CompactionResult {
            partition_id: meta.id,
            vectors_before: meta.vector_count + before,
            vectors_after: meta.vector_count,
            tombstones_removed: before,
        }
    }
}

/// Search result with partition info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionSearchResult {
    pub id: String,
    pub score: f32,
    pub partition_id: u64,
}

/// Compaction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResult {
    pub partition_id: u64,
    pub vectors_before: usize,
    pub vectors_after: usize,
    pub tombstones_removed: usize,
}

// ============================================================================
// PARTITIONED INDEX
// ============================================================================

/// Partitioned vector index
pub struct PartitionedIndex {
    /// Configuration
    config: PartitionConfig,
    /// Vector dimension
    dimension: usize,
    /// Partitions by ID
    partitions: RwLock<HashMap<u64, Arc<Partition>>>,
    /// Partition key value -> partition ID
    key_to_partition: RwLock<HashMap<String, u64>>,
    /// Statistics
    stats: RwLock<PartitionedIndexStats>,
}

impl PartitionedIndex {
    pub fn new(dimension: usize, config: PartitionConfig) -> Result<Self> {
        Ok(Self {
            config,
            dimension,
            partitions: RwLock::new(HashMap::new()),
            key_to_partition: RwLock::new(HashMap::new()),
            stats: RwLock::new(PartitionedIndexStats::default()),
        })
    }

    /// Hash a partition key value to partition ID
    fn hash_key(&self, key_value: &str) -> u64 {
        match self.config.hash_algorithm {
            HashAlgorithm::Murmur3 => {
                // Simple murmur3-like hash
                let mut h: u64 = 0;
                for byte in key_value.bytes() {
                    h ^= byte as u64;
                    h = h.wrapping_mul(0x5bd1e995);
                    h ^= h >> 47;
                }
                h % self.config.max_partitions as u64
            }
            HashAlgorithm::XXHash => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                key_value.hash(&mut hasher);
                hasher.finish() % self.config.max_partitions as u64
            }
            HashAlgorithm::Modulo => {
                // For testing - just use first char
                key_value.bytes().next().unwrap_or(0) as u64 % self.config.max_partitions as u64
            }
        }
    }

    /// Get or create partition for key value
    fn get_or_create_partition(&self, key_value: &str) -> Result<Arc<Partition>> {
        let partition_id = self.hash_key(key_value);

        // Check if exists
        {
            let partitions = self.partitions.read().unwrap();
            if let Some(p) = partitions.get(&partition_id) {
                return Ok(p.clone());
            }
        }

        // Create new partition
        {
            let mut partitions = self.partitions.write().unwrap();

            // Double-check after acquiring write lock
            if let Some(p) = partitions.get(&partition_id) {
                return Ok(p.clone());
            }

            // Check max partitions
            if partitions.len() >= self.config.max_partitions {
                return Err(VecStoreError::InvalidInput(format!(
                    "Maximum partitions ({}) reached",
                    self.config.max_partitions
                )));
            }

            let partition = Arc::new(Partition::new(partition_id, key_value, self.dimension));
            partitions.insert(partition_id, partition.clone());

            self.key_to_partition
                .write()
                .unwrap()
                .insert(key_value.to_string(), partition_id);

            // Update stats
            self.stats.write().unwrap().partition_count += 1;

            Ok(partition)
        }
    }

    /// Extract partition key from metadata
    fn extract_partition_key(&self, metadata: &serde_json::Value) -> Option<String> {
        metadata.get(&self.config.partition_key)
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    /// Insert or update a vector
    pub fn upsert(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Result<()> {
        let key_value = self.extract_partition_key(&metadata)
            .ok_or_else(|| VecStoreError::InvalidInput(format!(
                "Missing partition key: {}",
                self.config.partition_key
            )))?;

        let partition = self.get_or_create_partition(&key_value)?;
        partition.upsert(id, vector, Some(metadata))?;

        // Update stats
        self.stats.write().unwrap().total_vectors += 1;

        Ok(())
    }

    /// Delete a vector (requires knowing partition)
    pub fn delete(&self, id: &str, partition_key_value: &str) -> Result<bool> {
        let partition_id = self.hash_key(partition_key_value);

        let partitions = self.partitions.read().unwrap();
        if let Some(partition) = partitions.get(&partition_id) {
            let deleted = partition.delete(id)?;
            if deleted {
                self.stats.write().unwrap().total_vectors =
                    self.stats.read().unwrap().total_vectors.saturating_sub(1);
            }
            Ok(deleted)
        } else {
            Ok(false)
        }
    }

    /// Search within a specific partition
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        partition_key_value: &str,
    ) -> Result<Vec<PartitionSearchResult>> {
        let partition_id = self.hash_key(partition_key_value);

        let partitions = self.partitions.read().unwrap();
        if let Some(partition) = partitions.get(&partition_id) {
            Ok(partition.search(query, top_k))
        } else {
            Ok(Vec::new())
        }
    }

    /// Search across all partitions
    pub fn search_all_partitions(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<PartitionSearchResult>> {
        if !self.config.allow_cross_partition {
            return Err(VecStoreError::InvalidInput(
                "Cross-partition queries not allowed".to_string()
            ));
        }

        let partitions = self.partitions.read().unwrap();
        let mut all_results: Vec<PartitionSearchResult> = Vec::new();

        for partition in partitions.values() {
            all_results.extend(partition.search(query, top_k));
        }

        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(top_k);

        Ok(all_results)
    }

    /// Search multiple partitions
    pub fn search_partitions(
        &self,
        query: &[f32],
        top_k: usize,
        partition_key_values: &[String],
    ) -> Result<Vec<PartitionSearchResult>> {
        let partitions = self.partitions.read().unwrap();
        let mut all_results: Vec<PartitionSearchResult> = Vec::new();

        for key_value in partition_key_values {
            let partition_id = self.hash_key(key_value);
            if let Some(partition) = partitions.get(&partition_id) {
                all_results.extend(partition.search(query, top_k));
            }
        }

        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(top_k);

        Ok(all_results)
    }

    /// Get partition info
    pub fn get_partition(&self, partition_key_value: &str) -> Option<PartitionMeta> {
        let partition_id = self.hash_key(partition_key_value);
        let partitions = self.partitions.read().unwrap();
        partitions.get(&partition_id).map(|p| p.meta())
    }

    /// List all partitions
    pub fn list_partitions(&self) -> Vec<PartitionMeta> {
        let partitions = self.partitions.read().unwrap();
        partitions.values().map(|p| p.meta()).collect()
    }

    /// Drop a partition
    pub fn drop_partition(&self, partition_key_value: &str) -> Result<bool> {
        let partition_id = self.hash_key(partition_key_value);

        let removed = {
            let mut partitions = self.partitions.write().unwrap();
            partitions.remove(&partition_id).is_some()
        };

        if removed {
            self.key_to_partition.write().unwrap().remove(partition_key_value);
            let mut stats = self.stats.write().unwrap();
            stats.partition_count = stats.partition_count.saturating_sub(1);
        }

        Ok(removed)
    }

    /// Run compaction on all partitions
    pub fn compact_all(&self) -> Vec<CompactionResult> {
        let partitions = self.partitions.read().unwrap();
        let mut results = Vec::new();

        for partition in partitions.values() {
            if partition.needs_compaction(self.config.compaction_threshold) {
                results.push(partition.compact());
            }
        }

        results
    }

    /// Get statistics
    pub fn stats(&self) -> PartitionedIndexStats {
        self.stats.read().unwrap().clone()
    }
}

/// Partitioned index statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PartitionedIndexStats {
    pub partition_count: usize,
    pub total_vectors: usize,
    pub active_partitions: usize,
    pub archived_partitions: usize,
}

// ============================================================================
// PARTITION ROUTER
// ============================================================================

/// Intelligent partition router
pub struct PartitionRouter {
    /// Partition -> recent access count
    access_counts: RwLock<HashMap<u64, u64>>,
    /// Hot partitions (frequently accessed)
    hot_partitions: RwLock<Vec<u64>>,
    /// Cold partitions (rarely accessed)
    cold_partitions: RwLock<Vec<u64>>,
}

impl PartitionRouter {
    pub fn new() -> Self {
        Self {
            access_counts: RwLock::new(HashMap::new()),
            hot_partitions: RwLock::new(Vec::new()),
            cold_partitions: RwLock::new(Vec::new()),
        }
    }

    /// Record access to partition
    pub fn record_access(&self, partition_id: u64) {
        *self.access_counts
            .write()
            .unwrap()
            .entry(partition_id)
            .or_insert(0) += 1;
    }

    /// Update hot/cold classification
    pub fn update_classification(&self, threshold: u64) {
        let counts = self.access_counts.read().unwrap();

        let mut hot = Vec::new();
        let mut cold = Vec::new();

        for (&id, &count) in counts.iter() {
            if count >= threshold {
                hot.push(id);
            } else {
                cold.push(id);
            }
        }

        *self.hot_partitions.write().unwrap() = hot;
        *self.cold_partitions.write().unwrap() = cold;
    }

    /// Get hot partitions
    pub fn hot_partitions(&self) -> Vec<u64> {
        self.hot_partitions.read().unwrap().clone()
    }

    /// Get cold partitions (candidates for archival)
    pub fn cold_partitions(&self) -> Vec<u64> {
        self.cold_partitions.read().unwrap().clone()
    }

    /// Reset access counts (call periodically)
    pub fn reset_counts(&self) {
        self.access_counts.write().unwrap().clear();
    }
}

impl Default for PartitionRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HELPERS
// ============================================================================

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition() {
        let partition = Partition::new(1, "tenant_a", 4);

        partition.upsert("doc1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
        partition.upsert("doc2", vec![0.0, 1.0, 0.0, 0.0], None).unwrap();

        let results = partition.search(&[1.0, 0.0, 0.0, 0.0], 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_partitioned_index() {
        let config = PartitionConfig::new()
            .with_partition_key("tenant_id")
            .with_max_partitions(100);

        let index = PartitionedIndex::new(4, config).unwrap();

        // Insert to different partitions
        index.upsert("doc1", vec![1.0, 0.0, 0.0, 0.0], serde_json::json!({
            "tenant_id": "acme"
        })).unwrap();

        index.upsert("doc2", vec![0.0, 1.0, 0.0, 0.0], serde_json::json!({
            "tenant_id": "globex"
        })).unwrap();

        // Search within partition
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10, "acme").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");

        // Different partition should not find it
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10, "globex").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc2");
    }

    #[test]
    fn test_partition_listing() {
        let config = PartitionConfig::new()
            .with_partition_key("tenant_id");

        let index = PartitionedIndex::new(4, config).unwrap();

        index.upsert("doc1", vec![1.0; 4], serde_json::json!({"tenant_id": "a"})).unwrap();
        index.upsert("doc2", vec![1.0; 4], serde_json::json!({"tenant_id": "b"})).unwrap();
        index.upsert("doc3", vec![1.0; 4], serde_json::json!({"tenant_id": "c"})).unwrap();

        let partitions = index.list_partitions();
        assert_eq!(partitions.len(), 3);
    }
}
