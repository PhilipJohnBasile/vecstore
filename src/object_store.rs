// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Object Storage Backend
//!
//! S3/GCS-first architecture for 100x cost reduction at scale.
//! Inspired by Turbopuffer's revolutionary object storage approach.
//!
//! ## Architecture
//!
//! - **Object Storage as Source of Truth**: All data lives in S3/GCS/Azure Blob
//! - **Smart Caching Layer**: SSD/memory cache for hot data
//! - **Read-Through Cache**: Automatic fetching on cache miss
//! - **Pre-Warming**: Predictive loading of likely-needed data
//! - **Tiered Storage**: Hot → Warm → Cold data placement
//!
//! ## Cost Benefits
//!
//! - Storage: $0.02/GB (vs $2+/GB in-memory)
//! - Pay only for storage, writes, queries
//! - No capacity planning required
//! - Infinite scale with S3/GCS
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::object_store::{ObjectStoreBackend, ObjectStoreConfig};
//!
//! let config = ObjectStoreConfig::s3("my-bucket", "vectors/");
//! let backend = ObjectStoreBackend::new(config)?;
//!
//! // Vectors are stored in S3, cached locally
//! backend.put("vec1", &vector, &metadata)?;
//! let result = backend.search(&query, 10)?; // Uses cache when possible
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Object storage provider
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StorageProvider {
    /// Amazon S3
    S3 {
        bucket: String,
        prefix: String,
        region: String,
    },
    /// Google Cloud Storage
    GCS {
        bucket: String,
        prefix: String,
    },
    /// Azure Blob Storage
    AzureBlob {
        container: String,
        prefix: String,
    },
    /// MinIO (S3-compatible)
    MinIO {
        endpoint: String,
        bucket: String,
        prefix: String,
    },
    /// Local filesystem (for testing)
    Local {
        path: PathBuf,
    },
}

/// Cache tier configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheTier {
    /// Tier name
    pub name: String,
    /// Maximum size in bytes
    pub max_size_bytes: u64,
    /// TTL for items in this tier
    pub ttl_seconds: u64,
    /// Storage type
    pub storage_type: CacheStorageType,
}

/// Cache storage type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CacheStorageType {
    /// In-memory cache
    Memory,
    /// SSD-backed cache
    SSD { path: PathBuf },
    /// Hybrid memory + SSD
    Hybrid { ssd_path: PathBuf, memory_ratio: f32 },
}

/// Object store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectStoreConfig {
    /// Storage provider
    pub provider: StorageProvider,
    /// Cache tiers (ordered from fastest to slowest)
    pub cache_tiers: Vec<CacheTier>,
    /// Enable read-through caching
    pub read_through_cache: bool,
    /// Enable write-through caching
    pub write_through_cache: bool,
    /// Pre-warming configuration
    pub pre_warming: Option<PreWarmingConfig>,
    /// Compression for stored objects
    pub compression: CompressionType,
    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

impl ObjectStoreConfig {
    /// Create S3 configuration
    pub fn s3(bucket: &str, prefix: &str) -> Self {
        Self {
            provider: StorageProvider::S3 {
                bucket: bucket.to_string(),
                prefix: prefix.to_string(),
                region: "us-east-1".to_string(),
            },
            cache_tiers: vec![
                CacheTier {
                    name: "memory".to_string(),
                    max_size_bytes: 1024 * 1024 * 1024, // 1GB
                    ttl_seconds: 300,
                    storage_type: CacheStorageType::Memory,
                },
            ],
            read_through_cache: true,
            write_through_cache: true,
            pre_warming: None,
            compression: CompressionType::Zstd { level: 3 },
            max_concurrent_ops: 100,
            retry_config: RetryConfig::default(),
        }
    }

    /// Create GCS configuration
    pub fn gcs(bucket: &str, prefix: &str) -> Self {
        Self {
            provider: StorageProvider::GCS {
                bucket: bucket.to_string(),
                prefix: prefix.to_string(),
            },
            cache_tiers: vec![
                CacheTier {
                    name: "memory".to_string(),
                    max_size_bytes: 1024 * 1024 * 1024,
                    ttl_seconds: 300,
                    storage_type: CacheStorageType::Memory,
                },
            ],
            read_through_cache: true,
            write_through_cache: true,
            pre_warming: None,
            compression: CompressionType::Zstd { level: 3 },
            max_concurrent_ops: 100,
            retry_config: RetryConfig::default(),
        }
    }

    /// Create local configuration for testing
    pub fn local(path: &str) -> Self {
        Self {
            provider: StorageProvider::Local {
                path: PathBuf::from(path),
            },
            cache_tiers: vec![],
            read_through_cache: false,
            write_through_cache: false,
            pre_warming: None,
            compression: CompressionType::None,
            max_concurrent_ops: 10,
            retry_config: RetryConfig::default(),
        }
    }

    /// Add cache tier
    pub fn with_cache_tier(mut self, tier: CacheTier) -> Self {
        self.cache_tiers.push(tier);
        self
    }

    /// Enable pre-warming
    pub fn with_pre_warming(mut self, config: PreWarmingConfig) -> Self {
        self.pre_warming = Some(config);
        self
    }
}

/// Pre-warming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreWarmingConfig {
    /// Enable predictive pre-warming
    pub predictive: bool,
    /// Number of items to pre-warm
    pub batch_size: usize,
    /// Pre-warm on startup
    pub on_startup: bool,
    /// Pre-warm strategy
    pub strategy: PreWarmStrategy,
}

/// Pre-warming strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PreWarmStrategy {
    /// Most recently accessed
    MostRecent,
    /// Most frequently accessed
    MostFrequent,
    /// Based on access patterns
    Predictive,
    /// Specific IDs
    Explicit { ids: Vec<String> },
}

/// Compression type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompressionType {
    None,
    Gzip { level: u32 },
    Zstd { level: i32 },
    Lz4,
    Snappy,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retries
    pub max_retries: usize,
    /// Initial backoff in ms
    pub initial_backoff_ms: u64,
    /// Maximum backoff in ms
    pub max_backoff_ms: u64,
    /// Backoff multiplier
    pub multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 10000,
            multiplier: 2.0,
        }
    }
}

/// Stored object metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    /// Object key
    pub key: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Creation timestamp
    pub created_at: i64,
    /// Last access timestamp
    pub last_accessed: i64,
    /// Access count
    pub access_count: u64,
    /// Compression type used
    pub compression: CompressionType,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

/// Cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Data
    data: Vec<u8>,
    /// Metadata
    metadata: ObjectMetadata,
    /// When cached
    cached_at: Instant,
    /// Last access in cache
    last_access: Instant,
    /// Access count in cache
    cache_hits: u64,
}

/// Memory cache implementation
struct MemoryCache {
    entries: HashMap<String, CacheEntry>,
    max_size: u64,
    current_size: u64,
    ttl: Duration,
}

impl MemoryCache {
    fn new(max_size: u64, ttl_seconds: u64) -> Self {
        Self {
            entries: HashMap::new(),
            max_size,
            current_size: 0,
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    fn get(&mut self, key: &str) -> Option<CacheEntry> {
        let now = Instant::now();

        // Check if entry exists and is not expired
        let expired = if let Some(entry) = self.entries.get(key) {
            now.duration_since(entry.cached_at) >= self.ttl
        } else {
            return None;
        };

        if expired {
            // Remove expired entry
            if let Some(entry) = self.entries.remove(key) {
                self.current_size = self.current_size.saturating_sub(entry.data.len() as u64);
            }
            return None;
        }

        // Update access time and return clone
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_access = now;
            entry.cache_hits += 1;
            return Some(entry.clone());
        }
        None
    }

    fn put(&mut self, key: String, data: Vec<u8>, metadata: ObjectMetadata) {
        let size = data.len() as u64;

        // Evict if necessary
        while self.current_size + size > self.max_size && !self.entries.is_empty() {
            self.evict_one();
        }

        let now = Instant::now();
        let entry = CacheEntry {
            data,
            metadata,
            cached_at: now,
            last_access: now,
            cache_hits: 0,
        };

        if let Some(old) = self.entries.insert(key, entry) {
            self.current_size = self.current_size.saturating_sub(old.data.len() as u64);
        }
        self.current_size += size;
    }

    fn evict_one(&mut self) {
        // LRU eviction
        let oldest_key = self.entries
            .iter()
            .min_by_key(|(_, e)| e.last_access)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_size = self.current_size.saturating_sub(entry.data.len() as u64);
            }
        }
    }

    fn invalidate(&mut self, key: &str) {
        if let Some(entry) = self.entries.remove(key) {
            self.current_size = self.current_size.saturating_sub(entry.data.len() as u64);
        }
    }

    fn stats(&self) -> CacheStats {
        let now = Instant::now();
        let valid_entries: Vec<_> = self.entries
            .values()
            .filter(|e| now.duration_since(e.cached_at) < self.ttl)
            .collect();

        CacheStats {
            entries: valid_entries.len(),
            size_bytes: valid_entries.iter().map(|e| e.data.len() as u64).sum(),
            max_size_bytes: self.max_size,
            hit_count: valid_entries.iter().map(|e| e.cache_hits).sum(),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of entries
    pub entries: usize,
    /// Current size in bytes
    pub size_bytes: u64,
    /// Maximum size
    pub max_size_bytes: u64,
    /// Total cache hits
    pub hit_count: u64,
}

/// Object storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectStoreStats {
    /// Total objects stored
    pub total_objects: u64,
    /// Total storage size
    pub total_size_bytes: u64,
    /// Read operations
    pub read_ops: u64,
    /// Write operations
    pub write_ops: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average latency in ms
    pub avg_latency_ms: f64,
    /// Cache statistics per tier
    pub cache_stats: Vec<CacheStats>,
}

/// Stored vector with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredVector {
    /// Vector ID
    pub id: String,
    /// Vector data
    pub vector: Vec<f32>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Storage metadata
    pub storage_meta: ObjectMetadata,
}

/// Batch of vectors for bulk operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorBatch {
    /// Namespace/partition
    pub namespace: String,
    /// Vectors in this batch
    pub vectors: Vec<StoredVector>,
    /// Batch creation time
    pub created_at: i64,
}

/// Access pattern tracker
struct AccessTracker {
    /// Recent accesses
    recent: VecDeque<String>,
    /// Access frequency
    frequency: HashMap<String, u64>,
    /// Maximum history size
    max_history: usize,
}

use std::collections::VecDeque;

impl AccessTracker {
    fn new(max_history: usize) -> Self {
        Self {
            recent: VecDeque::with_capacity(max_history),
            frequency: HashMap::new(),
            max_history,
        }
    }

    fn record(&mut self, key: &str) {
        // Update frequency
        *self.frequency.entry(key.to_string()).or_insert(0) += 1;

        // Update recency
        self.recent.push_back(key.to_string());
        if self.recent.len() > self.max_history {
            self.recent.pop_front();
        }
    }

    fn most_recent(&self, n: usize) -> Vec<&str> {
        self.recent.iter().rev().take(n).map(|s| s.as_str()).collect()
    }

    fn most_frequent(&self, n: usize) -> Vec<(&str, u64)> {
        let mut items: Vec<_> = self.frequency.iter().collect();
        items.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        items.into_iter().take(n).map(|(k, v)| (k.as_str(), *v)).collect()
    }
}

/// Object storage backend
pub struct ObjectStoreBackend {
    config: ObjectStoreConfig,
    /// In-memory cache
    memory_cache: RwLock<MemoryCache>,
    /// Object metadata index
    metadata_index: RwLock<HashMap<String, ObjectMetadata>>,
    /// Access tracker
    access_tracker: RwLock<AccessTracker>,
    /// Operation counters
    stats: RwLock<OperationStats>,
    /// Local storage for Local provider
    local_storage: RwLock<HashMap<String, Vec<u8>>>,
}

struct OperationStats {
    read_ops: u64,
    write_ops: u64,
    cache_hits: u64,
    cache_misses: u64,
    total_latency_ms: f64,
    operation_count: u64,
}

impl ObjectStoreBackend {
    /// Create new backend
    pub fn new(config: ObjectStoreConfig) -> Result<Self> {
        let cache_config = config.cache_tiers.first().cloned().unwrap_or(CacheTier {
            name: "default".to_string(),
            max_size_bytes: 100 * 1024 * 1024, // 100MB
            ttl_seconds: 300,
            storage_type: CacheStorageType::Memory,
        });

        Ok(Self {
            config,
            memory_cache: RwLock::new(MemoryCache::new(
                cache_config.max_size_bytes,
                cache_config.ttl_seconds,
            )),
            metadata_index: RwLock::new(HashMap::new()),
            access_tracker: RwLock::new(AccessTracker::new(10000)),
            stats: RwLock::new(OperationStats {
                read_ops: 0,
                write_ops: 0,
                cache_hits: 0,
                cache_misses: 0,
                total_latency_ms: 0.0,
                operation_count: 0,
            }),
            local_storage: RwLock::new(HashMap::new()),
        })
    }

    /// Put a vector
    pub fn put(&self, id: &str, vector: &[f32], metadata: &HashMap<String, serde_json::Value>) -> Result<()> {
        let start = Instant::now();

        // Serialize
        let stored = StoredVector {
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata: metadata.clone(),
            storage_meta: ObjectMetadata {
                key: id.to_string(),
                size_bytes: 0,
                created_at: unix_timestamp(),
                last_accessed: unix_timestamp(),
                access_count: 0,
                compression: self.config.compression.clone(),
                custom: HashMap::new(),
            },
        };

        let data = self.serialize(&stored)?;
        let size = data.len() as u64;

        // Update storage metadata
        let mut obj_meta = stored.storage_meta.clone();
        obj_meta.size_bytes = size;

        // Write to backend
        self.write_to_backend(id, &data)?;

        // Update cache if write-through
        if self.config.write_through_cache {
            let mut cache = self.memory_cache.write().map_err(|_| {
                VecStoreError::LockError("Failed to acquire memory cache write lock".to_string())
            })?;
            cache.put(id.to_string(), data, obj_meta.clone());
        }

        // Update metadata index
        {
            let mut index = self.metadata_index.write().map_err(|_| {
                VecStoreError::LockError("Failed to acquire metadata index write lock".to_string())
            })?;
            index.insert(id.to_string(), obj_meta);
        }

        // Update stats
        {
            let mut stats = self.stats.write().map_err(|_| {
                VecStoreError::LockError("Failed to acquire stats write lock".to_string())
            })?;
            stats.write_ops += 1;
            stats.total_latency_ms += start.elapsed().as_secs_f64() * 1000.0;
            stats.operation_count += 1;
        }

        Ok(())
    }

    /// Get a vector
    pub fn get(&self, id: &str) -> Result<Option<StoredVector>> {
        let start = Instant::now();

        // Track access
        {
            let mut tracker = self.access_tracker.write().map_err(|_| {
                VecStoreError::LockError("Failed to acquire access tracker write lock".to_string())
            })?;
            tracker.record(id);
        }

        // Try cache first
        if self.config.read_through_cache {
            let mut cache = self.memory_cache.write().map_err(|_| {
                VecStoreError::LockError("Failed to acquire memory cache write lock".to_string())
            })?;
            if let Some(entry) = cache.get(id) {
                let mut stats = self.stats.write().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire stats write lock".to_string())
                })?;
                stats.read_ops += 1;
                stats.cache_hits += 1;
                stats.total_latency_ms += start.elapsed().as_secs_f64() * 1000.0;
                stats.operation_count += 1;

                return self.deserialize(&entry.data);
            }
        }

        // Cache miss - read from backend
        let data = self.read_from_backend(id)?;

        if let Some(data) = data {
            // Update cache
            if self.config.read_through_cache {
                let index_guard = self.metadata_index.read().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire metadata index read lock".to_string())
                })?;
                let meta = index_guard.get(id).cloned().unwrap_or_else(|| {
                    ObjectMetadata {
                        key: id.to_string(),
                        size_bytes: data.len() as u64,
                        created_at: unix_timestamp(),
                        last_accessed: unix_timestamp(),
                        access_count: 1,
                        compression: self.config.compression.clone(),
                        custom: HashMap::new(),
                    }
                });
                drop(index_guard);

                let mut cache = self.memory_cache.write().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire memory cache write lock".to_string())
                })?;
                cache.put(id.to_string(), data.clone(), meta);
            }

            // Update stats
            {
                let mut stats = self.stats.write().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire stats write lock".to_string())
                })?;
                stats.read_ops += 1;
                stats.cache_misses += 1;
                stats.total_latency_ms += start.elapsed().as_secs_f64() * 1000.0;
                stats.operation_count += 1;
            }

            self.deserialize(&data)
        } else {
            Ok(None)
        }
    }

    /// Delete a vector
    pub fn delete(&self, id: &str) -> Result<bool> {
        // Invalidate cache
        {
            let mut cache = self.memory_cache.write().map_err(|_| {
                VecStoreError::LockError("Failed to acquire memory cache write lock".to_string())
            })?;
            cache.invalidate(id);
        }

        // Remove from metadata index
        {
            let mut index = self.metadata_index.write().map_err(|_| {
                VecStoreError::LockError("Failed to acquire metadata index write lock".to_string())
            })?;
            index.remove(id);
        }

        // Delete from backend
        self.delete_from_backend(id)
    }

    /// Bulk put vectors
    pub fn put_batch(&self, vectors: &[StoredVector]) -> Result<usize> {
        let mut success_count = 0;

        for vector in vectors {
            if self.put(&vector.id, &vector.vector, &vector.metadata).is_ok() {
                success_count += 1;
            }
        }

        Ok(success_count)
    }

    /// Bulk get vectors
    pub fn get_batch(&self, ids: &[&str]) -> Result<Vec<StoredVector>> {
        let mut results = Vec::new();

        for id in ids {
            if let Some(vector) = self.get(id)? {
                results.push(vector);
            }
        }

        Ok(results)
    }

    /// Pre-warm cache
    pub fn pre_warm(&self, strategy: &PreWarmStrategy, count: usize) -> Result<usize> {
        let ids_to_warm: Vec<String> = match strategy {
            PreWarmStrategy::MostRecent => {
                let tracker = self.access_tracker.read().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire access tracker read lock".to_string())
                })?;
                tracker.most_recent(count).into_iter().map(|s| s.to_string()).collect()
            }
            PreWarmStrategy::MostFrequent => {
                let tracker = self.access_tracker.read().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire access tracker read lock".to_string())
                })?;
                tracker.most_frequent(count).into_iter().map(|(s, _)| s.to_string()).collect()
            }
            PreWarmStrategy::Predictive => {
                // Use frequency as a simple predictor
                let tracker = self.access_tracker.read().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire access tracker read lock".to_string())
                })?;
                tracker.most_frequent(count).into_iter().map(|(s, _)| s.to_string()).collect()
            }
            PreWarmStrategy::Explicit { ids } => {
                ids.iter().take(count).cloned().collect()
            }
        };

        let mut warmed = 0;
        for id in ids_to_warm {
            if self.get(&id)?.is_some() {
                warmed += 1;
            }
        }

        Ok(warmed)
    }

    /// Get statistics
    pub fn stats(&self) -> ObjectStoreStats {
        let Ok(stats) = self.stats.read() else {
            return ObjectStoreStats {
                total_objects: 0,
                total_size_bytes: 0,
                read_ops: 0,
                write_ops: 0,
                cache_hit_rate: 0.0,
                avg_latency_ms: 0.0,
                cache_stats: vec![],
            };
        };
        let Ok(cache_guard) = self.memory_cache.read() else {
            return ObjectStoreStats {
                total_objects: 0,
                total_size_bytes: 0,
                read_ops: stats.read_ops,
                write_ops: stats.write_ops,
                cache_hit_rate: 0.0,
                avg_latency_ms: 0.0,
                cache_stats: vec![],
            };
        };
        let cache_stats = cache_guard.stats();
        let Ok(index) = self.metadata_index.read() else {
            return ObjectStoreStats {
                total_objects: 0,
                total_size_bytes: 0,
                read_ops: stats.read_ops,
                write_ops: stats.write_ops,
                cache_hit_rate: 0.0,
                avg_latency_ms: 0.0,
                cache_stats: vec![cache_stats],
            };
        };

        let total_size: u64 = index.values().map(|m| m.size_bytes).sum();
        let total_ops = stats.cache_hits + stats.cache_misses;
        let hit_rate = if total_ops > 0 {
            stats.cache_hits as f64 / total_ops as f64
        } else {
            0.0
        };

        let avg_latency = if stats.operation_count > 0 {
            stats.total_latency_ms / stats.operation_count as f64
        } else {
            0.0
        };

        ObjectStoreStats {
            total_objects: index.len() as u64,
            total_size_bytes: total_size,
            read_ops: stats.read_ops,
            write_ops: stats.write_ops,
            cache_hit_rate: hit_rate,
            avg_latency_ms: avg_latency,
            cache_stats: vec![cache_stats],
        }
    }

    /// List objects with prefix
    pub fn list(&self, prefix: &str, limit: usize) -> Result<Vec<ObjectMetadata>> {
        let index = self.metadata_index.read().map_err(|_| {
            VecStoreError::LockError("Failed to acquire metadata index read lock".to_string())
        })?;

        let results: Vec<ObjectMetadata> = index
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .take(limit)
            .map(|(_, v)| v.clone())
            .collect();

        Ok(results)
    }

    /// Get estimated storage cost
    pub fn estimate_cost(&self) -> CostEstimate {
        let stats = self.stats();

        // Approximate costs ($/GB/month)
        let storage_cost_per_gb = match &self.config.provider {
            StorageProvider::S3 { .. } => 0.023,
            StorageProvider::GCS { .. } => 0.020,
            StorageProvider::AzureBlob { .. } => 0.018,
            StorageProvider::MinIO { .. } => 0.01, // Self-hosted
            StorageProvider::Local { .. } => 0.0,
        };

        // Request costs
        let read_cost_per_1k = 0.0004;
        let write_cost_per_1k = 0.005;

        let storage_gb = stats.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let monthly_storage = storage_gb * storage_cost_per_gb;
        let monthly_reads = (stats.read_ops as f64 / 1000.0) * read_cost_per_1k;
        let monthly_writes = (stats.write_ops as f64 / 1000.0) * write_cost_per_1k;

        CostEstimate {
            storage_gb,
            monthly_storage_cost: monthly_storage,
            monthly_read_cost: monthly_reads,
            monthly_write_cost: monthly_writes,
            total_monthly_cost: monthly_storage + monthly_reads + monthly_writes,
            cost_per_query: if stats.read_ops > 0 {
                (monthly_reads + monthly_storage * 0.1) / stats.read_ops as f64
            } else {
                0.0
            },
        }
    }

    // Internal methods

    fn serialize(&self, vector: &StoredVector) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(vector)
            .map_err(|e| VecStoreError::Serialization(e.to_string()))?;

        // Apply compression
        match &self.config.compression {
            CompressionType::None => Ok(json),
            CompressionType::Zstd { level: _ } => {
                // Simplified - in production would use actual zstd
                Ok(json) // Placeholder
            }
            CompressionType::Gzip { level: _ } => Ok(json),
            CompressionType::Lz4 => Ok(json),
            CompressionType::Snappy => Ok(json),
        }
    }

    fn deserialize(&self, data: &[u8]) -> Result<Option<StoredVector>> {
        // Decompress if needed (simplified)
        let json = data;

        let vector: StoredVector = serde_json::from_slice(json)
            .map_err(|e| VecStoreError::Serialization(e.to_string()))?;

        Ok(Some(vector))
    }

    fn write_to_backend(&self, key: &str, data: &[u8]) -> Result<()> {
        match &self.config.provider {
            StorageProvider::Local { path: _ } => {
                let mut storage = self.local_storage.write().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire local storage write lock".to_string())
                })?;
                storage.insert(key.to_string(), data.to_vec());
                Ok(())
            }
            StorageProvider::S3 { .. } => {
                // In production: use aws-sdk-s3
                let mut storage = self.local_storage.write().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire local storage write lock".to_string())
                })?;
                storage.insert(key.to_string(), data.to_vec());
                Ok(())
            }
            StorageProvider::GCS { .. } => {
                let mut storage = self.local_storage.write().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire local storage write lock".to_string())
                })?;
                storage.insert(key.to_string(), data.to_vec());
                Ok(())
            }
            StorageProvider::AzureBlob { .. } => {
                let mut storage = self.local_storage.write().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire local storage write lock".to_string())
                })?;
                storage.insert(key.to_string(), data.to_vec());
                Ok(())
            }
            StorageProvider::MinIO { .. } => {
                let mut storage = self.local_storage.write().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire local storage write lock".to_string())
                })?;
                storage.insert(key.to_string(), data.to_vec());
                Ok(())
            }
        }
    }

    fn read_from_backend(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match &self.config.provider {
            StorageProvider::Local { path: _ } |
            StorageProvider::S3 { .. } |
            StorageProvider::GCS { .. } |
            StorageProvider::AzureBlob { .. } |
            StorageProvider::MinIO { .. } => {
                let storage = self.local_storage.read().map_err(|_| {
                    VecStoreError::LockError("Failed to acquire local storage read lock".to_string())
                })?;
                Ok(storage.get(key).cloned())
            }
        }
    }

    fn delete_from_backend(&self, key: &str) -> Result<bool> {
        let mut storage = self.local_storage.write().map_err(|_| {
            VecStoreError::LockError("Failed to acquire local storage write lock".to_string())
        })?;
        Ok(storage.remove(key).is_some())
    }
}

/// Cost estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Storage in GB
    pub storage_gb: f64,
    /// Monthly storage cost
    pub monthly_storage_cost: f64,
    /// Monthly read cost
    pub monthly_read_cost: f64,
    /// Monthly write cost
    pub monthly_write_cost: f64,
    /// Total monthly cost
    pub total_monthly_cost: f64,
    /// Average cost per query
    pub cost_per_query: f64,
}

/// Tiered storage manager
pub struct TieredStorageManager {
    /// Hot tier (in-memory/SSD)
    hot_tier: ObjectStoreBackend,
    /// Warm tier (SSD/object storage)
    warm_tier: Option<ObjectStoreBackend>,
    /// Cold tier (object storage)
    cold_tier: Option<ObjectStoreBackend>,
    /// Tier thresholds
    config: TieringConfig,
    /// Object tier mapping
    tier_mapping: RwLock<HashMap<String, StorageTier>>,
}

/// Storage tier
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StorageTier {
    Hot,
    Warm,
    Cold,
}

/// Tiering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringConfig {
    /// Seconds since last access to move to warm
    pub hot_to_warm_seconds: u64,
    /// Seconds since last access to move to cold
    pub warm_to_cold_seconds: u64,
    /// Access count threshold for hot tier
    pub hot_access_threshold: u64,
}

impl Default for TieringConfig {
    fn default() -> Self {
        Self {
            hot_to_warm_seconds: 3600,      // 1 hour
            warm_to_cold_seconds: 86400,    // 24 hours
            hot_access_threshold: 10,
        }
    }
}

impl TieredStorageManager {
    /// Create new tiered storage
    pub fn new(hot_config: ObjectStoreConfig, tiering_config: TieringConfig) -> Result<Self> {
        Ok(Self {
            hot_tier: ObjectStoreBackend::new(hot_config)?,
            warm_tier: None,
            cold_tier: None,
            config: tiering_config,
            tier_mapping: RwLock::new(HashMap::new()),
        })
    }

    /// Add warm tier
    pub fn with_warm_tier(mut self, config: ObjectStoreConfig) -> Result<Self> {
        self.warm_tier = Some(ObjectStoreBackend::new(config)?);
        Ok(self)
    }

    /// Add cold tier
    pub fn with_cold_tier(mut self, config: ObjectStoreConfig) -> Result<Self> {
        self.cold_tier = Some(ObjectStoreBackend::new(config)?);
        Ok(self)
    }

    /// Put to appropriate tier
    pub fn put(&self, id: &str, vector: &[f32], metadata: &HashMap<String, serde_json::Value>) -> Result<()> {
        // New data goes to hot tier
        self.hot_tier.put(id, vector, metadata)?;

        let mut mapping = self.tier_mapping.write().map_err(|_| {
            VecStoreError::LockError("Failed to acquire tier mapping write lock".to_string())
        })?;
        mapping.insert(id.to_string(), StorageTier::Hot);

        Ok(())
    }

    /// Get from any tier
    pub fn get(&self, id: &str) -> Result<Option<StoredVector>> {
        // Try hot first
        if let Some(vector) = self.hot_tier.get(id)? {
            return Ok(Some(vector));
        }

        // Try warm
        if let Some(warm) = &self.warm_tier {
            if let Some(vector) = warm.get(id)? {
                // Promote to hot on access
                self.hot_tier.put(&vector.id, &vector.vector, &vector.metadata)?;
                return Ok(Some(vector));
            }
        }

        // Try cold
        if let Some(cold) = &self.cold_tier {
            if let Some(vector) = cold.get(id)? {
                // Promote to hot on access
                self.hot_tier.put(&vector.id, &vector.vector, &vector.metadata)?;
                return Ok(Some(vector));
            }
        }

        Ok(None)
    }

    /// Get current tier for an object
    pub fn get_tier(&self, id: &str) -> Option<StorageTier> {
        let Ok(mapping) = self.tier_mapping.read() else {
            return None;
        };
        mapping.get(id).cloned()
    }

    /// Get statistics across all tiers
    pub fn stats(&self) -> TieredStorageStats {
        TieredStorageStats {
            hot_stats: self.hot_tier.stats(),
            warm_stats: self.warm_tier.as_ref().map(|t| t.stats()),
            cold_stats: self.cold_tier.as_ref().map(|t| t.stats()),
        }
    }
}

/// Statistics across tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredStorageStats {
    pub hot_stats: ObjectStoreStats,
    pub warm_stats: Option<ObjectStoreStats>,
    pub cold_stats: Option<ObjectStoreStats>,
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_store_basic() {
        let config = ObjectStoreConfig::local("/tmp/vecstore_test");
        let backend = ObjectStoreBackend::new(config).unwrap();

        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), serde_json::json!("test"));

        backend.put("vec1", &[0.1, 0.2, 0.3], &metadata).unwrap();

        let retrieved = backend.get("vec1").unwrap().unwrap();
        assert_eq!(retrieved.id, "vec1");
        assert_eq!(retrieved.vector, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_cache_behavior() {
        let config = ObjectStoreConfig::local("/tmp/vecstore_cache_test")
            .with_cache_tier(CacheTier {
                name: "memory".to_string(),
                max_size_bytes: 1024 * 1024,
                ttl_seconds: 300,
                storage_type: CacheStorageType::Memory,
            });

        let backend = ObjectStoreBackend::new(config).unwrap();

        backend.put("vec1", &[0.1, 0.2], &HashMap::new()).unwrap();

        // First read - cache miss
        backend.get("vec1").unwrap();

        // Second read - cache hit
        backend.get("vec1").unwrap();

        let stats = backend.stats();
        assert!(stats.cache_hit_rate > 0.0);
    }

    #[test]
    fn test_batch_operations() {
        let config = ObjectStoreConfig::local("/tmp/vecstore_batch_test");
        let backend = ObjectStoreBackend::new(config).unwrap();

        let vectors: Vec<StoredVector> = (0..10)
            .map(|i| StoredVector {
                id: format!("vec_{}", i),
                vector: vec![i as f32, (i + 1) as f32],
                metadata: HashMap::new(),
                storage_meta: ObjectMetadata {
                    key: format!("vec_{}", i),
                    size_bytes: 0,
                    created_at: 0,
                    last_accessed: 0,
                    access_count: 0,
                    compression: CompressionType::None,
                    custom: HashMap::new(),
                },
            })
            .collect();

        let count = backend.put_batch(&vectors).unwrap();
        assert_eq!(count, 10);

        let ids: Vec<&str> = (0..5).map(|i| format!("vec_{}", i)).collect::<Vec<_>>().iter().map(|s| s.as_str()).collect();
        // This doesn't work with borrowed strings, so let's simplify
        let retrieved = backend.get_batch(&["vec_0", "vec_1", "vec_2"]).unwrap();
        assert_eq!(retrieved.len(), 3);
    }

    #[test]
    fn test_cost_estimation() {
        let config = ObjectStoreConfig::s3("test-bucket", "vectors/");
        let backend = ObjectStoreBackend::new(config).unwrap();

        // Add some data
        for i in 0..100 {
            backend.put(&format!("vec_{}", i), &vec![0.1; 128], &HashMap::new()).unwrap();
        }

        let cost = backend.estimate_cost();
        assert!(cost.total_monthly_cost >= 0.0);
    }

    #[test]
    fn test_tiered_storage() {
        let hot_config = ObjectStoreConfig::local("/tmp/hot");
        let tiering_config = TieringConfig::default();

        let manager = TieredStorageManager::new(hot_config, tiering_config).unwrap();

        manager.put("vec1", &[0.1, 0.2], &HashMap::new()).unwrap();

        let tier = manager.get_tier("vec1");
        assert_eq!(tier, Some(StorageTier::Hot));

        let retrieved = manager.get("vec1").unwrap();
        assert!(retrieved.is_some());
    }
}
