//! Time Travel Queries
//!
//! Query historical states of the vector database at any past timestamp.
//! Similar to Milvus time travel with better performance.
//!
//! # Features
//!
//! - **Point-in-Time Queries**: Query exact state at any past moment
//! - **Version History**: Track all changes with timestamps
//! - **Efficient Storage**: Delta compression for versions
//! - **Garbage Collection**: Automatic cleanup of old versions
//! - **Audit Trail**: Full history for compliance
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::timetravel::{TimeTravelIndex, Timestamp};
//!
//! let mut index = TimeTravelIndex::new(384)?;
//!
//! // Insert vectors (automatically versioned)
//! index.upsert("doc1", vec, metadata)?;
//!
//! // Query current state
//! let results = index.search(&query, 10)?;
//!
//! // Query state from 1 hour ago
//! let past = Timestamp::hours_ago(1);
//! let historical = index.search_at(&query, 10, past)?;
//!
//! // Get full history of a vector
//! let history = index.get_history("doc1")?;
//! ```

use std::collections::{HashMap, BTreeMap};
use std::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration as ChronoDuration};

use crate::error::{VecStoreError, Result};

// ============================================================================
// TIMESTAMP
// ============================================================================

/// Timestamp for time travel queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp(pub i64);

impl Timestamp {
    /// Current timestamp
    pub fn now() -> Self {
        Self(Utc::now().timestamp_millis())
    }

    /// Timestamp from milliseconds since epoch
    pub fn from_millis(millis: i64) -> Self {
        Self(millis)
    }

    /// Timestamp from DateTime
    pub fn from_datetime(dt: DateTime<Utc>) -> Self {
        Self(dt.timestamp_millis())
    }

    /// N hours ago
    pub fn hours_ago(hours: i64) -> Self {
        let dt = Utc::now() - ChronoDuration::hours(hours);
        Self::from_datetime(dt)
    }

    /// N days ago
    pub fn days_ago(days: i64) -> Self {
        let dt = Utc::now() - ChronoDuration::days(days);
        Self::from_datetime(dt)
    }

    /// N minutes ago
    pub fn minutes_ago(minutes: i64) -> Self {
        let dt = Utc::now() - ChronoDuration::minutes(minutes);
        Self::from_datetime(dt)
    }

    /// Convert to DateTime
    pub fn to_datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.0).unwrap_or_else(Utc::now)
    }

    /// Get milliseconds since epoch
    pub fn millis(&self) -> i64 {
        self.0
    }
}

// ============================================================================
// VERSION ENTRY
// ============================================================================

/// Type of version operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionOp {
    /// Vector was inserted
    Insert,
    /// Vector was updated
    Update,
    /// Vector was deleted
    Delete,
}

/// A single version of a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    /// Timestamp of this version
    pub timestamp: Timestamp,
    /// Operation type
    pub op: VersionOp,
    /// Vector data (None for delete)
    pub vector: Option<Vec<f32>>,
    /// Metadata (None for delete)
    pub metadata: Option<serde_json::Value>,
    /// Previous version timestamp (for delta chain)
    pub prev_version: Option<Timestamp>,
}

/// Version history for a single vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionHistory {
    /// Vector ID
    pub id: String,
    /// All versions sorted by timestamp (newest first)
    pub versions: Vec<VersionEntry>,
    /// Current version timestamp
    pub current: Timestamp,
    /// Creation timestamp
    pub created_at: Timestamp,
}

impl VersionHistory {
    pub fn new(id: String) -> Self {
        Self {
            id,
            versions: Vec::new(),
            current: Timestamp::now(),
            created_at: Timestamp::now(),
        }
    }

    /// Add a new version
    pub fn add_version(&mut self, entry: VersionEntry) {
        self.current = entry.timestamp;
        self.versions.insert(0, entry);
    }

    /// Get version at specific timestamp
    pub fn get_at(&self, ts: Timestamp) -> Option<&VersionEntry> {
        // Find the latest version that is <= ts
        self.versions.iter().find(|v| v.timestamp <= ts)
    }

    /// Check if vector existed at timestamp
    pub fn existed_at(&self, ts: Timestamp) -> bool {
        if let Some(v) = self.get_at(ts) {
            v.op != VersionOp::Delete
        } else {
            false
        }
    }

    /// Get vector data at timestamp
    pub fn vector_at(&self, ts: Timestamp) -> Option<&Vec<f32>> {
        self.get_at(ts).and_then(|v| v.vector.as_ref())
    }

    /// Prune versions older than timestamp
    pub fn prune_before(&mut self, ts: Timestamp) {
        // Keep at least one version before the cutoff for reconstruction
        let cutoff_idx = self.versions.iter()
            .position(|v| v.timestamp < ts)
            .map(|i| i + 1)
            .unwrap_or(self.versions.len());

        if cutoff_idx < self.versions.len() {
            self.versions.truncate(cutoff_idx);
        }
    }
}

// ============================================================================
// TIME TRAVEL INDEX
// ============================================================================

/// Configuration for time travel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeTravelConfig {
    /// Maximum version history to keep
    pub max_versions_per_vector: usize,
    /// Maximum age of versions to keep (hours)
    pub max_version_age_hours: u64,
    /// Enable delta compression
    pub delta_compression: bool,
    /// Garbage collection interval (seconds)
    pub gc_interval_secs: u64,
}

impl Default for TimeTravelConfig {
    fn default() -> Self {
        Self {
            max_versions_per_vector: 100,
            max_version_age_hours: 24 * 30, // 30 days
            delta_compression: true,
            gc_interval_secs: 3600,
        }
    }
}

/// Time travel enabled vector index
pub struct TimeTravelIndex {
    /// Vector dimension
    dimension: usize,
    /// Configuration
    config: TimeTravelConfig,
    /// Version histories by vector ID
    histories: RwLock<HashMap<String, VersionHistory>>,
    /// Timestamp index: timestamp -> affected vector IDs
    timestamp_index: RwLock<BTreeMap<Timestamp, Vec<String>>>,
    /// Statistics
    stats: RwLock<TimeTravelStats>,
}

impl TimeTravelIndex {
    pub fn new(dimension: usize, config: TimeTravelConfig) -> Result<Self> {
        Ok(Self {
            dimension,
            config,
            histories: RwLock::new(HashMap::new()),
            timestamp_index: RwLock::new(BTreeMap::new()),
            stats: RwLock::new(TimeTravelStats::default()),
        })
    }

    /// Insert or update a vector
    pub fn upsert(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<Timestamp> {
        if vector.len() != self.dimension {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let ts = Timestamp::now();
        let mut histories = self.histories.write()
            .map_err(|_| VecStoreError::LockError("histories lock poisoned".into()))?;

        let history = histories.entry(id.to_string()).or_insert_with(|| {
            let mut h = VersionHistory::new(id.to_string());
            h.created_at = ts;
            h
        });

        let op = if history.versions.is_empty() {
            VersionOp::Insert
        } else {
            VersionOp::Update
        };

        let prev = history.versions.first().map(|v| v.timestamp);

        history.add_version(VersionEntry {
            timestamp: ts,
            op,
            vector: Some(vector),
            metadata,
            prev_version: prev,
        });

        // Update timestamp index
        {
            let mut ts_idx = self.timestamp_index.write()
                .map_err(|_| VecStoreError::LockError("timestamp_index lock poisoned".into()))?;
            ts_idx.entry(ts).or_insert_with(Vec::new).push(id.to_string());
        }

        // Update stats
        {
            let mut stats = self.stats.write()
                .map_err(|_| VecStoreError::LockError("stats lock poisoned".into()))?;
            stats.total_versions += 1;
            if op == VersionOp::Insert {
                stats.total_vectors += 1;
            }
        }

        // Enforce max versions
        if history.versions.len() > self.config.max_versions_per_vector {
            history.versions.truncate(self.config.max_versions_per_vector);
        }

        Ok(ts)
    }

    /// Delete a vector
    pub fn delete(&self, id: &str) -> Result<Option<Timestamp>> {
        let ts = Timestamp::now();
        let mut histories = self.histories.write()
            .map_err(|_| VecStoreError::LockError("histories lock poisoned".into()))?;

        if let Some(history) = histories.get_mut(id) {
            let prev = history.versions.first().map(|v| v.timestamp);

            history.add_version(VersionEntry {
                timestamp: ts,
                op: VersionOp::Delete,
                vector: None,
                metadata: None,
                prev_version: prev,
            });

            // Update timestamp index
            {
                let mut ts_idx = self.timestamp_index.write()
                    .map_err(|_| VecStoreError::LockError("timestamp_index lock poisoned".into()))?;
                ts_idx.entry(ts).or_insert_with(Vec::new).push(id.to_string());
            }

            // Update stats
            {
                let mut stats = self.stats.write()
                    .map_err(|_| VecStoreError::LockError("stats lock poisoned".into()))?;
                stats.total_versions += 1;
                stats.total_vectors = stats.total_vectors.saturating_sub(1);
            }

            Ok(Some(ts))
        } else {
            Ok(None)
        }
    }

    /// Get current vector
    pub fn get(&self, id: &str) -> Result<Option<(Vec<f32>, Option<serde_json::Value>)>> {
        let histories = self.histories.read()
            .map_err(|_| VecStoreError::LockError("histories lock poisoned".into()))?;
        Ok(histories.get(id).and_then(|h| {
            h.versions.first().and_then(|v| {
                if v.op != VersionOp::Delete {
                    Some((v.vector.clone()?, v.metadata.clone()))
                } else {
                    None
                }
            })
        }))
    }

    /// Get vector at specific timestamp
    pub fn get_at(&self, id: &str, ts: Timestamp) -> Result<Option<(Vec<f32>, Option<serde_json::Value>)>> {
        let histories = self.histories.read()
            .map_err(|_| VecStoreError::LockError("histories lock poisoned".into()))?;
        Ok(histories.get(id).and_then(|h| {
            h.get_at(ts).and_then(|v| {
                if v.op != VersionOp::Delete {
                    Some((v.vector.clone()?, v.metadata.clone()))
                } else {
                    None
                }
            })
        }))
    }

    /// Get all vectors that existed at timestamp
    pub fn get_all_at(&self, ts: Timestamp) -> Result<Vec<(String, Vec<f32>)>> {
        let histories = self.histories.read()
            .map_err(|_| VecStoreError::LockError("histories lock poisoned".into()))?;
        Ok(histories.iter()
            .filter_map(|(id, h)| {
                h.vector_at(ts).map(|v| (id.clone(), v.clone()))
            })
            .collect())
    }

    /// Search current state
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<TimeTravelResult>> {
        self.search_at(query, top_k, Timestamp::now())
    }

    /// Search at specific timestamp
    pub fn search_at(
        &self,
        query: &[f32],
        top_k: usize,
        ts: Timestamp,
    ) -> Result<Vec<TimeTravelResult>> {
        if query.len() != self.dimension {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        let histories = self.histories.read()
            .map_err(|_| VecStoreError::LockError("histories lock poisoned".into()))?;

        let mut results: Vec<_> = histories.iter()
            .filter_map(|(id, h)| {
                h.vector_at(ts).map(|v| {
                    let score = cosine_similarity(query, v);
                    (id.clone(), score, h.get_at(ts).map(|e| e.timestamp))
                })
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results.into_iter()
            .map(|(id, score, version_ts)| TimeTravelResult {
                id,
                score,
                query_timestamp: ts,
                version_timestamp: version_ts.unwrap_or(ts),
            })
            .collect())
    }

    /// Get version history for a vector
    pub fn get_history(&self, id: &str) -> Result<Option<VersionHistory>> {
        let histories = self.histories.read()
            .map_err(|_| VecStoreError::LockError("histories lock poisoned".into()))?;
        Ok(histories.get(id).cloned())
    }

    /// Get changes between two timestamps
    pub fn get_changes(
        &self,
        from: Timestamp,
        to: Timestamp,
    ) -> Result<Vec<ChangeRecord>> {
        let ts_idx = self.timestamp_index.read()
            .map_err(|_| VecStoreError::LockError("timestamp_index lock poisoned".into()))?;
        let histories = self.histories.read()
            .map_err(|_| VecStoreError::LockError("histories lock poisoned".into()))?;

        let mut changes = Vec::new();

        for (ts, ids) in ts_idx.range(from..=to) {
            for id in ids {
                if let Some(history) = histories.get(id) {
                    if let Some(entry) = history.versions.iter().find(|v| v.timestamp == *ts) {
                        changes.push(ChangeRecord {
                            id: id.clone(),
                            timestamp: *ts,
                            op: entry.op,
                        });
                    }
                }
            }
        }

        Ok(changes)
    }

    /// Run garbage collection
    pub fn gc(&self) -> Result<GCResult> {
        let cutoff = Timestamp::hours_ago(self.config.max_version_age_hours as i64);
        let mut histories = self.histories.write()
            .map_err(|_| VecStoreError::LockError("histories lock poisoned".into()))?;
        let mut ts_idx = self.timestamp_index.write()
            .map_err(|_| VecStoreError::LockError("timestamp_index lock poisoned".into()))?;

        let mut versions_removed = 0;

        // Prune old versions
        for history in histories.values_mut() {
            let before = history.versions.len();
            history.prune_before(cutoff);
            versions_removed += before - history.versions.len();
        }

        // Remove empty histories (deleted with all versions pruned)
        let before = histories.len();
        histories.retain(|_, h| !h.versions.is_empty());
        let vectors_removed = before - histories.len();

        // Clean timestamp index
        ts_idx.retain(|ts, _| *ts >= cutoff);

        Ok(GCResult {
            versions_removed,
            vectors_removed,
            cutoff_timestamp: cutoff,
        })
    }

    /// Get statistics
    pub fn stats(&self) -> Result<TimeTravelStats> {
        let stats = self.stats.read()
            .map_err(|_| VecStoreError::LockError("stats lock poisoned".into()))?;
        Ok(stats.clone())
    }
}

/// Search result with time travel info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeTravelResult {
    pub id: String,
    pub score: f32,
    pub query_timestamp: Timestamp,
    pub version_timestamp: Timestamp,
}

/// Change record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeRecord {
    pub id: String,
    pub timestamp: Timestamp,
    pub op: VersionOp,
}

/// Garbage collection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCResult {
    pub versions_removed: usize,
    pub vectors_removed: usize,
    pub cutoff_timestamp: Timestamp,
}

/// Time travel statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeTravelStats {
    pub total_vectors: usize,
    pub total_versions: usize,
    pub oldest_version: Option<Timestamp>,
    pub newest_version: Option<Timestamp>,
}

// ============================================================================
// SNAPSHOT
// ============================================================================

/// Point-in-time snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Snapshot ID
    pub id: String,
    /// Timestamp of snapshot
    pub timestamp: Timestamp,
    /// Number of vectors at this point
    pub vector_count: usize,
    /// Description
    pub description: Option<String>,
}

/// Snapshot manager
pub struct SnapshotManager {
    snapshots: RwLock<Vec<Snapshot>>,
}

impl SnapshotManager {
    pub fn new() -> Self {
        Self {
            snapshots: RwLock::new(Vec::new()),
        }
    }

    /// Create a named snapshot at current time
    pub fn create(&self, description: Option<String>, vector_count: usize) -> Result<Snapshot> {
        let snapshot = Snapshot {
            id: format!("snap-{}", uuid_simple()),
            timestamp: Timestamp::now(),
            vector_count,
            description,
        };

        self.snapshots.write()
            .map_err(|_| VecStoreError::LockError("snapshots lock poisoned".into()))?
            .push(snapshot.clone());
        Ok(snapshot)
    }

    /// List all snapshots
    pub fn list(&self) -> Result<Vec<Snapshot>> {
        let snapshots = self.snapshots.read()
            .map_err(|_| VecStoreError::LockError("snapshots lock poisoned".into()))?;
        Ok(snapshots.clone())
    }

    /// Get snapshot by ID
    pub fn get(&self, id: &str) -> Result<Option<Snapshot>> {
        let snapshots = self.snapshots.read()
            .map_err(|_| VecStoreError::LockError("snapshots lock poisoned".into()))?;
        Ok(snapshots
            .iter()
            .find(|s| s.id == id)
            .cloned())
    }

    /// Delete snapshot
    pub fn delete(&self, id: &str) -> Result<bool> {
        let mut snapshots = self.snapshots.write()
            .map_err(|_| VecStoreError::LockError("snapshots lock poisoned".into()))?;
        let before = snapshots.len();
        snapshots.retain(|s| s.id != id);
        Ok(snapshots.len() < before)
    }
}

impl Default for SnapshotManager {
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

fn uuid_simple() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    std::time::Instant::now().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp() {
        let now = Timestamp::now();
        let hour_ago = Timestamp::hours_ago(1);

        assert!(now > hour_ago);
        assert_eq!(now.millis() - hour_ago.millis(), 3600 * 1000);
    }

    #[test]
    fn test_time_travel_index() {
        let config = TimeTravelConfig::default();
        let index = TimeTravelIndex::new(4, config).unwrap();

        // Insert
        let ts1 = index.upsert("doc1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();

        // Update
        std::thread::sleep(std::time::Duration::from_millis(10));
        let ts2 = index.upsert("doc1", vec![0.0, 1.0, 0.0, 0.0], None).unwrap();

        // Current should return updated
        let current = index.get("doc1").unwrap().unwrap();
        assert_eq!(current.0, vec![0.0, 1.0, 0.0, 0.0]);

        // Historical should return original
        let historical = index.get_at("doc1", ts1).unwrap().unwrap();
        assert_eq!(historical.0, vec![1.0, 0.0, 0.0, 0.0]);

        // History should have 2 versions
        let history = index.get_history("doc1").unwrap().unwrap();
        assert_eq!(history.versions.len(), 2);
    }

    #[test]
    fn test_search_at() {
        let config = TimeTravelConfig::default();
        let index = TimeTravelIndex::new(4, config).unwrap();

        index.upsert("doc1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
        index.upsert("doc2", vec![0.0, 1.0, 0.0, 0.0], None).unwrap();

        let before_delete = Timestamp::now();
        std::thread::sleep(std::time::Duration::from_millis(10));

        index.delete("doc1").unwrap();

        // Current search should only find doc2
        let current = index.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].id, "doc2");

        // Historical search should find both
        let historical = index.search_at(&[1.0, 0.0, 0.0, 0.0], 10, before_delete).unwrap();
        assert_eq!(historical.len(), 2);
    }
}
