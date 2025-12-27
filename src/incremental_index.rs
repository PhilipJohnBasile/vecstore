// Incremental Index Updates - Update indexes without full rebuilds
// Supports online insertions, deletions, and modifications with minimal downtime

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Incremental index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Maximum pending operations before triggering merge
    pub merge_threshold: usize,
    /// Maximum delta index size (vectors) before merge
    pub delta_size_threshold: usize,
    /// Background merge interval
    pub merge_interval: Duration,
    /// Enable lazy deletion (mark deleted, don't remove immediately)
    pub lazy_deletion: bool,
    /// Tombstone cleanup interval
    pub tombstone_cleanup_interval: Duration,
    /// Maximum tombstone ratio before forced cleanup
    pub max_tombstone_ratio: f32,
    /// Enable write-ahead logging
    pub wal_enabled: bool,
    /// WAL flush interval
    pub wal_flush_interval: Duration,
    /// Enable compression for delta indexes
    pub compress_delta: bool,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            merge_threshold: 10_000,
            delta_size_threshold: 100_000,
            merge_interval: Duration::from_secs(60),
            lazy_deletion: true,
            tombstone_cleanup_interval: Duration::from_secs(300),
            max_tombstone_ratio: 0.2,
            wal_enabled: true,
            wal_flush_interval: Duration::from_millis(100),
            compress_delta: true,
        }
    }
}

/// Operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    Insert {
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    },
    Update {
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    },
    Delete {
        id: String,
    },
    BatchInsert {
        items: Vec<(String, Vec<f32>, Option<serde_json::Value>)>,
    },
}

/// WAL entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    sequence: u64,
    timestamp: u64,
    operation: Operation,
    checksum: u32,
}

/// Delta index (small, mutable index for recent changes)
#[derive(Debug)]
struct DeltaIndex {
    /// Vectors in delta
    vectors: RwLock<HashMap<String, DeltaEntry>>,
    /// Insertion order for search priority
    insertion_order: RwLock<Vec<String>>,
    /// Size in vectors
    size: AtomicUsize,
    /// Created at
    created_at: Instant,
}

/// Entry in delta index
#[derive(Debug, Clone)]
struct DeltaEntry {
    id: String,
    vector: Vec<f32>,
    metadata: Option<serde_json::Value>,
    version: u64,
    inserted_at: u64,
}

/// Tombstone for deleted vectors
#[derive(Debug, Clone)]
struct Tombstone {
    id: String,
    deleted_at: u64,
    version: u64,
}

/// Base index reference (read-only after creation)
#[derive(Debug)]
struct BaseIndex {
    /// Index ID
    id: String,
    /// Vector count
    vector_count: usize,
    /// Creation timestamp
    created_at: u64,
    /// Vector data (in real implementation, this would be on disk)
    vectors: HashMap<String, BaseEntry>,
    /// Index structure (simplified - in production, this is HNSW/IVF graph)
    index_built: bool,
}

/// Entry in base index
#[derive(Debug, Clone)]
struct BaseEntry {
    id: String,
    vector: Vec<f32>,
    metadata: Option<serde_json::Value>,
}

/// Incremental index manager
pub struct IncrementalIndex {
    config: IncrementalConfig,
    /// Dimensions
    dimensions: usize,
    /// Base indexes (immutable after merge)
    base_indexes: RwLock<Vec<Arc<BaseIndex>>>,
    /// Active delta index
    delta: Arc<DeltaIndex>,
    /// Tombstones (deleted vectors)
    tombstones: RwLock<HashMap<String, Tombstone>>,
    /// WAL sequence counter
    wal_sequence: AtomicU64,
    /// WAL buffer
    wal_buffer: RwLock<VecDeque<WalEntry>>,
    /// Pending operations count
    pending_ops: AtomicUsize,
    /// Merge in progress flag
    merge_in_progress: AtomicBool,
    /// Last merge time
    last_merge: RwLock<Instant>,
    /// Statistics
    stats: IncrementalStats,
}

/// Incremental index statistics
#[derive(Debug, Default)]
struct IncrementalStats {
    inserts: AtomicU64,
    updates: AtomicU64,
    deletes: AtomicU64,
    merges: AtomicU64,
    searches: AtomicU64,
    tombstones_cleaned: AtomicU64,
    wal_entries_written: AtomicU64,
}

impl IncrementalIndex {
    /// Create a new incremental index
    pub fn new(dimensions: usize, config: IncrementalConfig) -> Self {
        Self {
            config,
            dimensions,
            base_indexes: RwLock::new(Vec::new()),
            delta: Arc::new(DeltaIndex {
                vectors: RwLock::new(HashMap::new()),
                insertion_order: RwLock::new(Vec::new()),
                size: AtomicUsize::new(0),
                created_at: Instant::now(),
            }),
            tombstones: RwLock::new(HashMap::new()),
            wal_sequence: AtomicU64::new(0),
            wal_buffer: RwLock::new(VecDeque::new()),
            pending_ops: AtomicUsize::new(0),
            merge_in_progress: AtomicBool::new(false),
            last_merge: RwLock::new(Instant::now()),
            stats: IncrementalStats::default(),
        }
    }

    /// Insert a vector
    pub fn insert(&self, id: String, vector: Vec<f32>, metadata: Option<serde_json::Value>) -> Result<()> {
        self.validate_vector(&vector)?;

        let op = Operation::Insert {
            id: id.clone(),
            vector: vector.clone(),
            metadata: metadata.clone(),
        };

        // Write to WAL
        if self.config.wal_enabled {
            self.write_wal(op.clone())?;
        }

        // Add to delta index
        let entry = DeltaEntry {
            id: id.clone(),
            vector,
            metadata,
            version: self.wal_sequence.load(Ordering::Relaxed),
            inserted_at: current_timestamp(),
        };

        {
            let mut vectors = self.delta.vectors.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?;
            let is_new = !vectors.contains_key(&id);
            vectors.insert(id.clone(), entry);

            if is_new {
                self.delta.size.fetch_add(1, Ordering::Relaxed);
                self.delta.insertion_order.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.push(id.clone());
            }
        }

        // Remove from tombstones if exists
        self.tombstones.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.remove(&id);

        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        self.pending_ops.fetch_add(1, Ordering::Relaxed);

        // Check if merge needed
        self.maybe_trigger_merge()?;

        Ok(())
    }

    /// Update a vector
    pub fn update(&self, id: String, vector: Vec<f32>, metadata: Option<serde_json::Value>) -> Result<()> {
        self.validate_vector(&vector)?;

        // Check if exists
        if !self.exists(&id) {
            return Err(VecStoreError::NotFound(format!("Vector {} not found", id)));
        }

        let op = Operation::Update {
            id: id.clone(),
            vector: vector.clone(),
            metadata: metadata.clone(),
        };

        if self.config.wal_enabled {
            self.write_wal(op)?;
        }

        // Update in delta (overwrites or adds)
        let entry = DeltaEntry {
            id: id.clone(),
            vector,
            metadata,
            version: self.wal_sequence.load(Ordering::Relaxed),
            inserted_at: current_timestamp(),
        };

        {
            let mut vectors = self.delta.vectors.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?;
            let is_new = !vectors.contains_key(&id);
            vectors.insert(id.clone(), entry);

            if is_new {
                self.delta.size.fetch_add(1, Ordering::Relaxed);
                self.delta.insertion_order.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.push(id);
            }
        }

        self.stats.updates.fetch_add(1, Ordering::Relaxed);
        self.pending_ops.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Delete a vector
    pub fn delete(&self, id: String) -> Result<()> {
        if !self.exists(&id) {
            return Err(VecStoreError::NotFound(format!("Vector {} not found", id)));
        }

        let op = Operation::Delete { id: id.clone() };

        if self.config.wal_enabled {
            self.write_wal(op)?;
        }

        if self.config.lazy_deletion {
            // Add tombstone
            let tombstone = Tombstone {
                id: id.clone(),
                deleted_at: current_timestamp(),
                version: self.wal_sequence.load(Ordering::Relaxed),
            };
            self.tombstones.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.insert(id.clone(), tombstone);
        }

        // Remove from delta if present
        let mut vectors = self.delta.vectors.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?;
        if vectors.remove(&id).is_some() {
            self.delta.size.fetch_sub(1, Ordering::Relaxed);
        }

        self.stats.deletes.fetch_add(1, Ordering::Relaxed);
        self.pending_ops.fetch_add(1, Ordering::Relaxed);

        // Check if cleanup needed
        self.maybe_trigger_cleanup()?;

        Ok(())
    }

    /// Batch insert
    pub fn batch_insert(&self, items: Vec<(String, Vec<f32>, Option<serde_json::Value>)>) -> Result<BatchResult> {
        let mut succeeded = 0;
        let mut failed = 0;
        let mut errors = Vec::new();

        for (id, vector, metadata) in items {
            match self.insert(id.clone(), vector, metadata) {
                Ok(_) => succeeded += 1,
                Err(e) => {
                    failed += 1;
                    errors.push((id, e.to_string()));
                }
            }
        }

        Ok(BatchResult { succeeded, failed, errors })
    }

    /// Check if vector exists
    pub fn exists(&self, id: &str) -> bool {
        // Check tombstones first
        if let Ok(tombstones) = self.tombstones.read() {
            if tombstones.contains_key(id) {
                return false;
            }
        }

        // Check delta
        if let Ok(delta) = self.delta.vectors.read() {
            if delta.contains_key(id) {
                return true;
            }
        }

        // Check base indexes
        if let Ok(bases) = self.base_indexes.read() {
            for base in bases.iter() {
                if base.vectors.contains_key(id) {
                    return true;
                }
            }
        }

        false
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Option<VectorEntry> {
        // Check tombstones
        if self.tombstones.read().ok()?.contains_key(id) {
            return None;
        }

        // Check delta first (most recent)
        if let Some(entry) = self.delta.vectors.read().ok()?.get(id) {
            return Some(VectorEntry {
                id: entry.id.clone(),
                vector: entry.vector.clone(),
                metadata: entry.metadata.clone(),
                version: entry.version,
            });
        }

        // Check base indexes (from newest to oldest)
        let bases = self.base_indexes.read().ok()?;
        for base in bases.iter().rev() {
            if let Some(entry) = base.vectors.get(id) {
                return Some(VectorEntry {
                    id: entry.id.clone(),
                    vector: entry.vector.clone(),
                    metadata: entry.metadata.clone(),
                    version: 0,
                });
            }
        }

        None
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.validate_vector(query)?;
        self.stats.searches.fetch_add(1, Ordering::Relaxed);

        let tombstones = self.tombstones.read().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?;
        let mut results = Vec::new();

        // Search delta index
        let delta_vectors = self.delta.vectors.read().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?;
        for (id, entry) in delta_vectors.iter() {
            if tombstones.contains_key(id) {
                continue;
            }
            let distance = euclidean_distance(query, &entry.vector);
            results.push(SearchResult {
                id: id.clone(),
                distance,
                metadata: entry.metadata.clone(),
                source: SearchSource::Delta,
            });
        }

        // Search base indexes
        for base in self.base_indexes.read().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.iter() {
            for (id, entry) in &base.vectors {
                if tombstones.contains_key(id) || delta_vectors.contains_key(id) {
                    continue; // Skip deleted or overridden
                }
                let distance = euclidean_distance(query, &entry.vector);
                results.push(SearchResult {
                    id: id.clone(),
                    distance,
                    metadata: entry.metadata.clone(),
                    source: SearchSource::Base,
                });
            }
        }

        // Sort by distance and take top k
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Trigger a merge operation
    pub fn merge(&self) -> Result<MergeResult> {
        // Check if merge already in progress
        if self.merge_in_progress.swap(true, Ordering::SeqCst) {
            return Err(VecStoreError::InvalidInput("Merge already in progress".into()));
        }

        let start = Instant::now();
        let delta_size = self.delta.size.load(Ordering::Relaxed);

        // Collect all vectors (delta + base - tombstones)
        let mut all_vectors = HashMap::new();
        let tombstones = self.tombstones.read().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?;

        // Add base vectors
        for base in self.base_indexes.read().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.iter() {
            for (id, entry) in &base.vectors {
                if !tombstones.contains_key(id) {
                    all_vectors.insert(id.clone(), BaseEntry {
                        id: id.clone(),
                        vector: entry.vector.clone(),
                        metadata: entry.metadata.clone(),
                    });
                }
            }
        }

        // Add/override with delta vectors
        for (id, entry) in self.delta.vectors.read().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.iter() {
            if !tombstones.contains_key(id) {
                all_vectors.insert(id.clone(), BaseEntry {
                    id: id.clone(),
                    vector: entry.vector.clone(),
                    metadata: entry.metadata.clone(),
                });
            }
        }
        drop(tombstones);

        // Create new base index
        let new_base = Arc::new(BaseIndex {
            id: format!("base_{}", current_timestamp()),
            vector_count: all_vectors.len(),
            created_at: current_timestamp(),
            vectors: all_vectors,
            index_built: true,
        });

        // Replace indexes
        {
            let mut bases = self.base_indexes.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?;
            let _old_count = bases.len();
            bases.clear();
            bases.push(new_base);

            // Clear delta
            self.delta.vectors.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.clear();
            self.delta.insertion_order.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.clear();
            self.delta.size.store(0, Ordering::Relaxed);

            // Clear tombstones
            let cleaned = self.tombstones.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.len();
            self.tombstones.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.clear();

            self.stats.tombstones_cleaned.fetch_add(cleaned as u64, Ordering::Relaxed);
        }

        // Update state
        *self.last_merge.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))? = Instant::now();
        self.pending_ops.store(0, Ordering::Relaxed);
        self.merge_in_progress.store(false, Ordering::SeqCst);
        self.stats.merges.fetch_add(1, Ordering::Relaxed);

        Ok(MergeResult {
            vectors_merged: delta_size,
            new_index_size: self.base_indexes.read().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?[0].vector_count,
            duration: start.elapsed(),
            tombstones_removed: 0,
        })
    }

    /// Force cleanup of tombstones
    pub fn cleanup_tombstones(&self) -> Result<usize> {
        if !self.config.lazy_deletion {
            return Ok(0);
        }

        let tombstone_count = self.tombstones.read().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.len();
        let total_vectors = self.total_vectors();

        if total_vectors == 0 {
            return Ok(0);
        }

        let ratio = tombstone_count as f32 / (total_vectors + tombstone_count) as f32;

        if ratio < self.config.max_tombstone_ratio {
            return Ok(0);
        }

        // Trigger merge to clean up
        let result = self.merge()?;
        Ok(result.tombstones_removed)
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(VecStoreError::InvalidInput(format!(
                "Expected {} dimensions, got {}",
                self.dimensions,
                vector.len()
            )));
        }
        Ok(())
    }

    fn write_wal(&self, operation: Operation) -> Result<()> {
        let sequence = self.wal_sequence.fetch_add(1, Ordering::SeqCst);
        let entry = WalEntry {
            sequence,
            timestamp: current_timestamp(),
            operation,
            checksum: 0, // In production: compute actual checksum
        };

        self.wal_buffer.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.push_back(entry);
        self.stats.wal_entries_written.fetch_add(1, Ordering::Relaxed);

        // Flush if buffer is large
        if self.wal_buffer.read().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.len() > 1000 {
            self.flush_wal()?;
        }

        Ok(())
    }

    fn flush_wal(&self) -> Result<()> {
        // In production: write to disk
        let mut buffer = self.wal_buffer.write().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?;
        buffer.clear();
        Ok(())
    }

    fn maybe_trigger_merge(&self) -> Result<()> {
        let delta_size = self.delta.size.load(Ordering::Relaxed);
        let pending = self.pending_ops.load(Ordering::Relaxed);

        if delta_size >= self.config.delta_size_threshold
            || pending >= self.config.merge_threshold
        {
            // In production: spawn background task
            // For now, merge synchronously
            let _ = self.merge();
        }

        Ok(())
    }

    fn maybe_trigger_cleanup(&self) -> Result<()> {
        let tombstone_count = self.tombstones.read().map_err(|_| VecStoreError::LockError("lock poisoned".into()))?.len();
        let total = self.total_vectors();

        if total > 0 {
            let ratio = tombstone_count as f32 / (total + tombstone_count) as f32;
            if ratio >= self.config.max_tombstone_ratio {
                let _ = self.cleanup_tombstones();
            }
        }

        Ok(())
    }

    /// Get total vector count
    pub fn total_vectors(&self) -> usize {
        let delta_size = self.delta.size.load(Ordering::Relaxed);
        let base_size: usize = self.base_indexes.read()
            .map(|b| b.iter().map(|b| b.vector_count).sum())
            .unwrap_or(0);
        let tombstone_count = self.tombstones.read().map(|t| t.len()).unwrap_or(0);

        delta_size + base_size - tombstone_count.min(delta_size + base_size)
    }

    /// Get index statistics
    pub fn get_stats(&self) -> IndexStats {
        IndexStats {
            total_vectors: self.total_vectors(),
            delta_size: self.delta.size.load(Ordering::Relaxed),
            base_index_count: self.base_indexes.read().map(|b| b.len()).unwrap_or(0),
            tombstone_count: self.tombstones.read().map(|t| t.len()).unwrap_or(0),
            pending_operations: self.pending_ops.load(Ordering::Relaxed),
            inserts: self.stats.inserts.load(Ordering::Relaxed),
            updates: self.stats.updates.load(Ordering::Relaxed),
            deletes: self.stats.deletes.load(Ordering::Relaxed),
            merges: self.stats.merges.load(Ordering::Relaxed),
            searches: self.stats.searches.load(Ordering::Relaxed),
            wal_sequence: self.wal_sequence.load(Ordering::Relaxed),
        }
    }
}

/// Vector entry returned from get
#[derive(Debug, Clone, Serialize)]
pub struct VectorEntry {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
    pub version: u64,
}

/// Search result
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<serde_json::Value>,
    pub source: SearchSource,
}

/// Source of search result
#[derive(Debug, Clone, Serialize)]
pub enum SearchSource {
    Delta,
    Base,
}

/// Batch operation result
#[derive(Debug, Clone, Serialize)]
pub struct BatchResult {
    pub succeeded: usize,
    pub failed: usize,
    pub errors: Vec<(String, String)>,
}

/// Merge operation result
#[derive(Debug, Clone, Serialize)]
pub struct MergeResult {
    pub vectors_merged: usize,
    pub new_index_size: usize,
    pub duration: Duration,
    pub tombstones_removed: usize,
}

/// Index statistics
#[derive(Debug, Clone, Serialize)]
pub struct IndexStats {
    pub total_vectors: usize,
    pub delta_size: usize,
    pub base_index_count: usize,
    pub tombstone_count: usize,
    pub pending_operations: usize,
    pub inserts: u64,
    pub updates: u64,
    pub deletes: u64,
    pub merges: u64,
    pub searches: u64,
    pub wal_sequence: u64,
}

// Helper functions

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// WAL recovery helper
pub struct WalRecovery;

impl WalRecovery {
    /// Replay WAL entries to recover index state
    pub fn replay(index: &IncrementalIndex, entries: Vec<WalEntry>) -> Result<RecoveryResult> {
        let start = Instant::now();
        let mut replayed = 0;
        let mut errors = 0;

        for entry in entries {
            match entry.operation {
                Operation::Insert { id, vector, metadata } => {
                    if index.insert(id, vector, metadata).is_ok() {
                        replayed += 1;
                    } else {
                        errors += 1;
                    }
                }
                Operation::Update { id, vector, metadata } => {
                    // For updates, treat as insert if not exists
                    if index.update(id.clone(), vector.clone(), metadata.clone()).is_err() {
                        let _ = index.insert(id, vector, metadata);
                    }
                    replayed += 1;
                }
                Operation::Delete { id } => {
                    if index.delete(id).is_ok() {
                        replayed += 1;
                    } else {
                        errors += 1;
                    }
                }
                Operation::BatchInsert { items } => {
                    for (id, vector, metadata) in items {
                        if index.insert(id, vector, metadata).is_ok() {
                            replayed += 1;
                        } else {
                            errors += 1;
                        }
                    }
                }
            }
        }

        Ok(RecoveryResult {
            entries_replayed: replayed,
            errors,
            duration: start.elapsed(),
        })
    }
}

/// WAL recovery result
#[derive(Debug, Clone, Serialize)]
pub struct RecoveryResult {
    pub entries_replayed: usize,
    pub errors: usize,
    pub duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let index = IncrementalIndex::new(3, IncrementalConfig::default());

        index.insert("v1".to_string(), vec![1.0, 0.0, 0.0], None).unwrap();
        index.insert("v2".to_string(), vec![0.0, 1.0, 0.0], None).unwrap();
        index.insert("v3".to_string(), vec![0.0, 0.0, 1.0], None).unwrap();

        let results = index.search(&[1.0, 0.1, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1"); // Closest to query
    }

    #[test]
    fn test_update() {
        let index = IncrementalIndex::new(3, IncrementalConfig::default());

        index.insert("v1".to_string(), vec![1.0, 0.0, 0.0], None).unwrap();

        let entry1 = index.get("v1").unwrap();
        assert_eq!(entry1.vector, vec![1.0, 0.0, 0.0]);

        index.update("v1".to_string(), vec![0.0, 1.0, 0.0], None).unwrap();

        let entry2 = index.get("v1").unwrap();
        assert_eq!(entry2.vector, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_delete() {
        let index = IncrementalIndex::new(3, IncrementalConfig::default());

        index.insert("v1".to_string(), vec![1.0, 0.0, 0.0], None).unwrap();
        assert!(index.exists("v1"));

        index.delete("v1".to_string()).unwrap();
        assert!(!index.exists("v1"));
        assert!(index.get("v1").is_none());
    }

    #[test]
    fn test_merge() {
        let config = IncrementalConfig {
            merge_threshold: 1000, // High threshold to prevent auto-merge
            ..Default::default()
        };
        let index = IncrementalIndex::new(3, config);

        // Insert vectors
        for i in 0..100 {
            index.insert(
                format!("v{}", i),
                vec![i as f32, 0.0, 0.0],
                None,
            ).unwrap();
        }

        let stats_before = index.get_stats();
        assert_eq!(stats_before.delta_size, 100);
        assert_eq!(stats_before.base_index_count, 0);

        // Trigger merge
        let result = index.merge().unwrap();
        assert_eq!(result.vectors_merged, 100);

        let stats_after = index.get_stats();
        assert_eq!(stats_after.delta_size, 0);
        assert_eq!(stats_after.base_index_count, 1);
        assert_eq!(stats_after.total_vectors, 100);
    }

    #[test]
    fn test_tombstone_cleanup() {
        let config = IncrementalConfig {
            lazy_deletion: true,
            max_tombstone_ratio: 0.1,
            ..Default::default()
        };
        let index = IncrementalIndex::new(3, config);

        // Insert vectors
        for i in 0..10 {
            index.insert(format!("v{}", i), vec![i as f32, 0.0, 0.0], None).unwrap();
        }

        // Delete some
        for i in 0..3 {
            index.delete(format!("v{}", i)).unwrap();
        }

        let stats = index.get_stats();
        assert_eq!(stats.tombstone_count, 3);
        assert_eq!(stats.total_vectors, 7);
    }

    #[test]
    fn test_batch_insert() {
        let index = IncrementalIndex::new(3, IncrementalConfig::default());

        let items = vec![
            ("v1".to_string(), vec![1.0, 0.0, 0.0], None),
            ("v2".to_string(), vec![0.0, 1.0, 0.0], None),
            ("v3".to_string(), vec![0.0, 0.0, 1.0], None),
        ];

        let result = index.batch_insert(items).unwrap();
        assert_eq!(result.succeeded, 3);
        assert_eq!(result.failed, 0);
        assert_eq!(index.total_vectors(), 3);
    }
}
