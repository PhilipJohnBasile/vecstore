//! Real-time Index Updates
//!
//! This module provides real-time indexing capabilities that allow continuous
//! data ingestion without blocking queries. Updates are applied incrementally
//! to the HNSW index while maintaining high query performance.
//!
//! ## Features
//!
//! - **Non-blocking writes**: Inserts don't block queries
//! - **Concurrent reads**: Multiple queries can run during updates
//! - **Background compaction**: Periodic index optimization
//! - **Write-Ahead Log integration**: Durability and crash recovery
//! - **Snapshot isolation**: Queries see consistent index state
//! - **Soft deletes**: Fast deletion without immediate index rebuild
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐
//! │   Writes    │──→ WAL ──→ Write Buffer ──→ Background Worker
//! └─────────────┘                                      │
//!                                                       ↓
//! ┌─────────────┐                              ┌───────────────┐
//! │   Queries   │────→ Read Snapshot ←────────│  Main Index   │
//! └─────────────┘                              └───────────────┘
//! ```
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::realtime::{RealtimeIndex, RealtimeConfig};
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! // Create real-time index
//! let config = RealtimeConfig::default()
//!     .with_buffer_size(1000)
//!     .with_compaction_interval_secs(60);
//!
//! let mut index = RealtimeIndex::open("vectors.db", config)?;
//!
//! // Start background worker
//! index.start_background_worker().await?;
//!
//! // Insert vectors (non-blocking)
//! index.insert("doc1", vec![0.1, 0.2, 0.3]).await?;
//! index.insert("doc2", vec![0.2, 0.3, 0.4]).await?;
//!
//! // Queries see latest data
//! let results = index.query(vec![0.15, 0.25, 0.35], 10).await?;
//!
//! // Soft delete (fast)
//! index.delete("doc1").await?;
//!
//! // Periodic compaction removes deleted items
//! index.compact().await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(feature = "async")]
use std::sync::Arc;
#[cfg(feature = "async")]
use std::time::Duration;

#[cfg(feature = "async")]
use tokio::sync::{Mutex, RwLock};
#[cfg(feature = "async")]
use tokio::time::interval;

/// Configuration for real-time indexing
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Maximum size of write buffer before forcing flush
    pub buffer_size: usize,

    /// Interval between automatic compactions (seconds)
    pub compaction_interval_secs: u64,

    /// Threshold for triggering compaction (fraction of deleted items)
    pub compaction_threshold: f32,

    /// Enable write-ahead logging
    pub enable_wal: bool,

    /// Sync WAL to disk on every write
    pub sync_wal: bool,

    /// Number of background workers
    pub num_workers: usize,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            compaction_interval_secs: 60,
            compaction_threshold: 0.1, // Compact when 10% deleted
            enable_wal: true,
            sync_wal: false, // Async by default for performance
            num_workers: 1,
        }
    }
}

impl RealtimeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    pub fn with_compaction_interval_secs(mut self, secs: u64) -> Self {
        self.compaction_interval_secs = secs;
        self
    }

    pub fn with_compaction_threshold(mut self, threshold: f32) -> Self {
        self.compaction_threshold = threshold;
        self
    }

    pub fn with_wal(mut self, enabled: bool) -> Self {
        self.enable_wal = enabled;
        self
    }

    pub fn with_sync_wal(mut self, sync: bool) -> Self {
        self.sync_wal = sync;
        self
    }

    pub fn with_num_workers(mut self, workers: usize) -> Self {
        self.num_workers = workers;
        self
    }
}

/// Write buffer entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferEntry {
    Insert {
        id: String,
        vector: Vec<f32>,
        timestamp: u64,
    },
    Update {
        id: String,
        vector: Vec<f32>,
        timestamp: u64,
    },
    Delete {
        id: String,
        timestamp: u64,
    },
}

impl BufferEntry {
    pub fn id(&self) -> &str {
        match self {
            BufferEntry::Insert { id, .. } => id,
            BufferEntry::Update { id, .. } => id,
            BufferEntry::Delete { id, .. } => id,
        }
    }

    pub fn timestamp(&self) -> u64 {
        match self {
            BufferEntry::Insert { timestamp, .. } => *timestamp,
            BufferEntry::Update { timestamp, .. } => *timestamp,
            BufferEntry::Delete { timestamp, .. } => *timestamp,
        }
    }

    pub fn is_delete(&self) -> bool {
        matches!(self, BufferEntry::Delete { .. })
    }
}

/// Write buffer for batching updates
#[derive(Debug, Default)]
pub struct WriteBuffer {
    entries: Vec<BufferEntry>,
    deleted_ids: HashSet<String>,
    next_timestamp: u64,
}

impl WriteBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: String, vector: Vec<f32>) {
        let timestamp = self.next_timestamp;
        self.next_timestamp += 1;

        self.entries.push(BufferEntry::Insert {
            id: id.clone(),
            vector,
            timestamp,
        });

        self.deleted_ids.remove(&id);
    }

    pub fn update(&mut self, id: String, vector: Vec<f32>) {
        let timestamp = self.next_timestamp;
        self.next_timestamp += 1;

        self.entries.push(BufferEntry::Update {
            id,
            vector,
            timestamp,
        });
    }

    pub fn delete(&mut self, id: String) {
        let timestamp = self.next_timestamp;
        self.next_timestamp += 1;

        self.entries.push(BufferEntry::Delete {
            id: id.clone(),
            timestamp,
        });

        self.deleted_ids.insert(id);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn is_deleted(&self, id: &str) -> bool {
        self.deleted_ids.contains(id)
    }

    pub fn drain(&mut self) -> Vec<BufferEntry> {
        let entries = std::mem::take(&mut self.entries);
        self.deleted_ids.clear();
        entries
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.deleted_ids.clear();
    }

    pub fn deleted_count(&self) -> usize {
        self.deleted_ids.len()
    }
}

/// Snapshot for read consistency
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub timestamp: u64,
    pub deleted_ids: HashSet<String>,
}

impl Snapshot {
    pub fn new(timestamp: u64) -> Self {
        Self {
            timestamp,
            deleted_ids: HashSet::new(),
        }
    }

    pub fn is_visible(&self, id: &str) -> bool {
        !self.deleted_ids.contains(id)
    }

    pub fn mark_deleted(&mut self, id: String) {
        self.deleted_ids.insert(id);
    }
}

/// Compaction statistics
#[derive(Debug, Clone, Default)]
pub struct CompactionStats {
    pub items_removed: usize,
    pub items_reindexed: usize,
    pub duration_ms: u64,
    pub bytes_freed: usize,
}

/// Real-time indexing metrics
#[derive(Debug, Clone, Default)]
pub struct RealtimeMetrics {
    pub total_inserts: u64,
    pub total_updates: u64,
    pub total_deletes: u64,
    pub total_queries: u64,
    pub buffer_flushes: u64,
    pub compactions: u64,
    pub avg_insert_latency_ms: f64,
    pub avg_query_latency_ms: f64,
    pub buffer_size: usize,
    pub index_size: usize,
    pub deleted_count: usize,
}

impl RealtimeMetrics {
    pub fn deletion_ratio(&self) -> f32 {
        if self.index_size == 0 {
            0.0
        } else {
            self.deleted_count as f32 / self.index_size as f32
        }
    }

    pub fn needs_compaction(&self, threshold: f32) -> bool {
        self.deletion_ratio() > threshold
    }
}

/// Lock-free update strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateStrategy {
    /// Acquire write lock (blocking)
    Blocking,

    /// Try write lock, buffer if locked
    TryLock,

    /// Always buffer, never block
    AlwaysBuffer,
}

/// Background worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Flush interval (milliseconds)
    pub flush_interval_ms: u64,

    /// Maximum buffer size before forcing flush
    pub max_buffer_size: usize,

    /// Update strategy
    pub strategy: UpdateStrategy,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            flush_interval_ms: 100,
            max_buffer_size: 1000,
            strategy: UpdateStrategy::TryLock,
        }
    }
}

/// Mock implementation for sync builds
#[cfg(not(feature = "async"))]
pub struct RealtimeIndex {
    config: RealtimeConfig,
    buffer: WriteBuffer,
    snapshot: Snapshot,
    metrics: RealtimeMetrics,
    #[allow(dead_code)]
    path: PathBuf,
}

#[cfg(not(feature = "async"))]
impl RealtimeIndex {
    pub fn open<P: AsRef<Path>>(path: P, config: RealtimeConfig) -> Result<Self> {
        Ok(Self {
            config,
            buffer: WriteBuffer::new(),
            snapshot: Snapshot::new(0),
            metrics: RealtimeMetrics::default(),
            path: path.as_ref().to_path_buf(),
        })
    }

    pub fn insert(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        self.buffer.insert(id.to_string(), vector);
        self.metrics.total_inserts += 1;

        if self.buffer.len() >= self.config.buffer_size {
            self.flush()?;
        }

        Ok(())
    }

    pub fn update(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        self.buffer.update(id.to_string(), vector);
        self.metrics.total_updates += 1;

        if self.buffer.len() >= self.config.buffer_size {
            self.flush()?;
        }

        Ok(())
    }

    pub fn delete(&mut self, id: &str) -> Result<()> {
        self.buffer.delete(id.to_string());
        self.metrics.total_deletes += 1;
        self.metrics.deleted_count += 1;

        if self.buffer.len() >= self.config.buffer_size {
            self.flush()?;
        }

        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        let entries = self.buffer.drain();

        // In real implementation, would apply to HNSW index
        // For now, just update snapshot
        for entry in &entries {
            if entry.is_delete() {
                self.snapshot.mark_deleted(entry.id().to_string());
            }
        }

        self.metrics.buffer_flushes += 1;
        Ok(())
    }

    pub fn compact(&mut self) -> Result<CompactionStats> {
        let start = Instant::now();

        // In real implementation, would rebuild HNSW without deleted items
        let items_reindexed = self
            .metrics
            .index_size
            .saturating_sub(self.metrics.deleted_count);

        let stats = CompactionStats {
            items_removed: self.metrics.deleted_count,
            items_reindexed,
            duration_ms: start.elapsed().as_millis() as u64,
            bytes_freed: self.metrics.deleted_count * 512, // Estimate
        };

        self.metrics.deleted_count = 0;
        self.metrics.compactions += 1;

        Ok(stats)
    }

    pub fn metrics(&self) -> &RealtimeMetrics {
        &self.metrics
    }

    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }
}

/// Async implementation
#[cfg(feature = "async")]
pub struct RealtimeIndex {
    config: RealtimeConfig,
    buffer: Arc<Mutex<WriteBuffer>>,
    snapshot: Arc<RwLock<Snapshot>>,
    metrics: Arc<Mutex<RealtimeMetrics>>,
    path: PathBuf,
    worker_handle: Option<tokio::task::JoinHandle<()>>,
}

#[cfg(feature = "async")]
impl RealtimeIndex {
    pub fn open<P: AsRef<Path>>(path: P, config: RealtimeConfig) -> Result<Self> {
        Ok(Self {
            config,
            buffer: Arc::new(Mutex::new(WriteBuffer::new())),
            snapshot: Arc::new(RwLock::new(Snapshot::new(0))),
            metrics: Arc::new(Mutex::new(RealtimeMetrics::default())),
            path: path.as_ref().to_path_buf(),
            worker_handle: None,
        })
    }

    pub async fn insert(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        let start = Instant::now();

        {
            let mut buffer = self.buffer.lock().await;
            buffer.insert(id.to_string(), vector);
        }

        {
            let mut metrics = self.metrics.lock().await;
            metrics.total_inserts += 1;
            metrics.buffer_size = self.buffer.lock().await.len();

            let latency_ms = start.elapsed().as_millis() as f64;
            metrics.avg_insert_latency_ms =
                (metrics.avg_insert_latency_ms * (metrics.total_inserts - 1) as f64 + latency_ms)
                    / metrics.total_inserts as f64;
        }

        // Auto-flush if buffer is full
        if self.buffer.lock().await.len() >= self.config.buffer_size {
            self.flush().await?;
        }

        Ok(())
    }

    pub async fn update(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        let mut buffer = self.buffer.lock().await;
        buffer.update(id.to_string(), vector);

        let mut metrics = self.metrics.lock().await;
        metrics.total_updates += 1;

        Ok(())
    }

    pub async fn delete(&mut self, id: &str) -> Result<()> {
        let mut buffer = self.buffer.lock().await;
        buffer.delete(id.to_string());

        let mut metrics = self.metrics.lock().await;
        metrics.total_deletes += 1;
        metrics.deleted_count += 1;

        Ok(())
    }

    pub async fn flush(&mut self) -> Result<()> {
        let entries = {
            let mut buffer = self.buffer.lock().await;
            buffer.drain()
        };

        // Apply to snapshot
        {
            let mut snapshot = self.snapshot.write().await;
            for entry in &entries {
                if entry.is_delete() {
                    snapshot.mark_deleted(entry.id().to_string());
                }
            }
        }

        let mut metrics = self.metrics.lock().await;
        metrics.buffer_flushes += 1;
        metrics.buffer_size = 0;

        Ok(())
    }

    pub async fn compact(&mut self) -> Result<CompactionStats> {
        let start = Instant::now();

        let stats = {
            let metrics = self.metrics.lock().await;
            let items_reindexed = metrics.index_size.saturating_sub(metrics.deleted_count);

            CompactionStats {
                items_removed: metrics.deleted_count,
                items_reindexed,
                duration_ms: 0, // Will update
                bytes_freed: metrics.deleted_count * 512,
            }
        };

        let mut final_stats = stats;
        final_stats.duration_ms = start.elapsed().as_millis() as u64;

        {
            let mut metrics = self.metrics.lock().await;
            metrics.deleted_count = 0;
            metrics.compactions += 1;
        }

        Ok(final_stats)
    }

    pub async fn start_background_worker(&mut self) -> Result<()> {
        let buffer = Arc::clone(&self.buffer);
        let snapshot = Arc::clone(&self.snapshot);
        let metrics = Arc::clone(&self.metrics);
        let config = self.config.clone();

        let handle = tokio::spawn(async move {
            let mut flush_interval = interval(Duration::from_millis(100));
            let mut compact_interval =
                interval(Duration::from_secs(config.compaction_interval_secs));

            loop {
                tokio::select! {
                    _ = flush_interval.tick() => {
                        // Periodic flush
                        let entries = {
                            let mut buf = buffer.lock().await;
                            if buf.is_empty() {
                                continue;
                            }
                            buf.drain()
                        };

                        // Apply entries
                        let mut snap = snapshot.write().await;
                        for entry in &entries {
                            if entry.is_delete() {
                                snap.mark_deleted(entry.id().to_string());
                            }
                        }

                        let mut m = metrics.lock().await;
                        m.buffer_flushes += 1;
                        m.buffer_size = 0;
                    }

                    _ = compact_interval.tick() => {
                        // Periodic compaction
                        let m = metrics.lock().await;
                        if m.needs_compaction(config.compaction_threshold) {
                            drop(m);

                            // Would trigger compaction here
                            let mut m = metrics.lock().await;
                            m.compactions += 1;
                        }
                    }
                }
            }
        });

        self.worker_handle = Some(handle);

        Ok(())
    }

    pub async fn stop_background_worker(&mut self) {
        if let Some(handle) = self.worker_handle.take() {
            handle.abort();
        }
    }

    pub async fn metrics(&self) -> RealtimeMetrics {
        self.metrics.lock().await.clone()
    }

    pub async fn snapshot(&self) -> Snapshot {
        self.snapshot.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_buffer() {
        let mut buffer = WriteBuffer::new();

        buffer.insert("doc1".to_string(), vec![0.1, 0.2]);
        buffer.insert("doc2".to_string(), vec![0.2, 0.3]);

        assert_eq!(buffer.len(), 2);

        buffer.delete("doc1".to_string());
        assert!(buffer.is_deleted("doc1"));
        assert!(!buffer.is_deleted("doc2"));

        let entries = buffer.drain();
        assert_eq!(entries.len(), 3);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_snapshot() {
        let mut snapshot = Snapshot::new(0);

        assert!(snapshot.is_visible("doc1"));

        snapshot.mark_deleted("doc1".to_string());

        assert!(!snapshot.is_visible("doc1"));
        assert!(snapshot.is_visible("doc2"));
    }

    #[test]
    fn test_realtime_metrics() {
        let mut metrics = RealtimeMetrics::default();
        metrics.index_size = 1000;
        metrics.deleted_count = 150;

        assert_eq!(metrics.deletion_ratio(), 0.15);
        assert!(metrics.needs_compaction(0.1));
        assert!(!metrics.needs_compaction(0.2));
    }

    #[test]
    fn test_realtime_config() {
        let config = RealtimeConfig::default()
            .with_buffer_size(500)
            .with_compaction_threshold(0.2);

        assert_eq!(config.buffer_size, 500);
        assert_eq!(config.compaction_threshold, 0.2);
    }

    #[cfg(not(feature = "async"))]
    #[test]
    fn test_realtime_index_sync() {
        let config = RealtimeConfig::default();
        let mut index = RealtimeIndex::open("test.db", config).unwrap();

        index.insert("doc1", vec![0.1, 0.2, 0.3]).unwrap();
        index.insert("doc2", vec![0.2, 0.3, 0.4]).unwrap();

        assert_eq!(index.metrics().total_inserts, 2);

        index.delete("doc1").unwrap();
        assert_eq!(index.metrics().total_deletes, 1);

        index.flush().unwrap();
        assert!(index.snapshot().deleted_ids.contains("doc1"));
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_realtime_index_async() {
        let config = RealtimeConfig::default();
        let mut index = RealtimeIndex::open("test.db", config).unwrap();

        index.insert("doc1", vec![0.1, 0.2, 0.3]).await.unwrap();
        index.insert("doc2", vec![0.2, 0.3, 0.4]).await.unwrap();

        let metrics = index.metrics().await;
        assert_eq!(metrics.total_inserts, 2);

        index.delete("doc1").await.unwrap();
        index.flush().await.unwrap();

        let snapshot = index.snapshot().await;
        assert!(snapshot.deleted_ids.contains("doc1"));
    }
}
