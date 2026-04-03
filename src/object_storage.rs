//! Object Storage Tier for Cost-Effective Vector Storage
//!
//! S3-native architecture for 10x cheaper storage compared to local disk.
//! Inspired by Turbopuffer and LanceDB's object-storage-first designs.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Query Path                                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  [Query] → [Hot Cache] → [Warm Cache] → [Cold Storage (S3)]     │
//! │            (Memory)      (Local SSD)     (Object Store)         │
//! │            < 1ms         < 10ms          < 100ms                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Tiering Strategy
//!
//! - **Hot**: Recently/frequently accessed vectors in memory
//! - **Warm**: Less frequent vectors on local SSD
//! - **Cold**: Archival vectors in object storage (S3/GCS/Azure)
//!
//! ## Features
//!
//! - Automatic tiering based on access patterns
//! - Lazy loading from cold storage on demand
//! - Background prefetching based on query patterns
//! - Compression for cold storage
//! - Parallel fetch for batch queries
//!
//! ## Example
//!
//! ```rust,no_run
//! use vecstore::object_storage::{TieredStore, TierConfig, S3Backend};
//!
//! let config = TierConfig {
//!     hot_capacity_mb: 1024,   // 1GB in memory
//!     warm_capacity_mb: 10240, // 10GB on SSD
//!     cold_backend: S3Backend::new("my-bucket", "vectors/"),
//!     ..Default::default()
//! };
//!
//! let store = TieredStore::new(config).await?;
//!
//! // Add vectors - automatically tiered
//! store.add("doc1", &embedding).await?;
//!
//! // Query seamlessly across tiers
//! let results = store.search(&query, 10).await?;
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Tiered storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    /// Hot tier capacity in MB (in-memory)
    pub hot_capacity_mb: usize,

    /// Warm tier capacity in MB (local SSD)
    pub warm_capacity_mb: usize,

    /// Local path for warm tier
    pub warm_path: PathBuf,

    /// Cold tier compression level (0-9, 0 = none)
    pub cold_compression_level: u32,

    /// Time before vector moves to colder tier (seconds)
    pub tier_down_threshold_secs: u64,

    /// Number of accesses to keep in hot tier
    pub hot_access_threshold: usize,

    /// Enable background prefetching
    pub enable_prefetch: bool,

    /// Prefetch lookahead count
    pub prefetch_count: usize,

    /// Maximum concurrent cold fetches
    pub max_concurrent_fetches: usize,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            hot_capacity_mb: 1024,
            warm_capacity_mb: 10240,
            warm_path: PathBuf::from("./vecstore_warm"),
            cold_compression_level: 6,
            tier_down_threshold_secs: 3600, // 1 hour
            hot_access_threshold: 10,
            enable_prefetch: true,
            prefetch_count: 100,
            max_concurrent_fetches: 10,
        }
    }
}

// ============================================================================
// STORAGE BACKENDS
// ============================================================================

/// Trait for cold storage backends
#[async_trait::async_trait]
pub trait ColdStorageBackend: Send + Sync {
    /// Store a chunk of vectors
    async fn put(&self, key: &str, data: &[u8]) -> Result<()>;

    /// Retrieve a chunk of vectors
    async fn get(&self, key: &str) -> Result<Vec<u8>>;

    /// Delete a chunk
    async fn delete(&self, key: &str) -> Result<()>;

    /// List all chunks with prefix
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Check if key exists
    async fn exists(&self, key: &str) -> Result<bool>;

    /// Get storage statistics
    async fn stats(&self) -> Result<BackendStats>;
}

/// Storage backend statistics
#[derive(Debug, Clone, Default)]
pub struct BackendStats {
    pub total_bytes: u64,
    pub total_objects: u64,
    pub get_requests: u64,
    pub put_requests: u64,
}

/// S3-compatible backend
pub struct S3Backend {
    bucket: String,
    prefix: String,
    // In production: aws_sdk_s3::Client
    // For now, simulate with in-memory storage
    storage: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    stats: Arc<RwLock<BackendStats>>,
}

impl S3Backend {
    /// Create new S3 backend
    pub fn new(bucket: &str, prefix: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            storage: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(BackendStats::default())),
        }
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}{}", self.prefix, key)
    }
}

#[async_trait::async_trait]
impl ColdStorageBackend for S3Backend {
    async fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        let full_key = self.full_key(key);
        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        storage.insert(full_key, data.to_vec());
        stats.put_requests += 1;
        stats.total_bytes += data.len() as u64;
        stats.total_objects += 1;

        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>> {
        let full_key = self.full_key(key);
        let storage = self.storage.read().await;
        let mut stats = self.stats.write().await;

        stats.get_requests += 1;

        storage
            .get(&full_key)
            .cloned()
            .ok_or_else(|| anyhow!("Key not found: {}", key))
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let full_key = self.full_key(key);
        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        if let Some(data) = storage.remove(&full_key) {
            stats.total_bytes -= data.len() as u64;
            stats.total_objects -= 1;
        }

        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.full_key(prefix);
        let storage = self.storage.read().await;

        let keys: Vec<String> = storage
            .keys()
            .filter(|k| k.starts_with(&full_prefix))
            .cloned()
            .collect();

        Ok(keys)
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let full_key = self.full_key(key);
        let storage = self.storage.read().await;
        Ok(storage.contains_key(&full_key))
    }

    async fn stats(&self) -> Result<BackendStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }
}

/// Local filesystem backend (for testing/development)
pub struct LocalBackend {
    base_path: PathBuf,
    stats: Arc<RwLock<BackendStats>>,
}

impl LocalBackend {
    pub fn new(base_path: PathBuf) -> Self {
        Self {
            base_path,
            stats: Arc::new(RwLock::new(BackendStats::default())),
        }
    }
}

#[async_trait::async_trait]
impl ColdStorageBackend for LocalBackend {
    async fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        let path = self.base_path.join(key);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, data).await?;

        let mut stats = self.stats.write().await;
        stats.put_requests += 1;
        stats.total_bytes += data.len() as u64;
        stats.total_objects += 1;

        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.base_path.join(key);
        let mut stats = self.stats.write().await;
        stats.get_requests += 1;

        Ok(tokio::fs::read(&path).await?)
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let path = self.base_path.join(key);
        if path.exists() {
            let metadata = tokio::fs::metadata(&path).await?;
            tokio::fs::remove_file(&path).await?;

            let mut stats = self.stats.write().await;
            stats.total_bytes -= metadata.len();
            stats.total_objects -= 1;
        }
        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let search_path = self.base_path.join(prefix);
        let mut keys = Vec::new();

        if search_path.exists() {
            let mut entries = tokio::fs::read_dir(&search_path).await?;
            while let Some(entry) = entries.next_entry().await? {
                if let Some(name) = entry.file_name().to_str() {
                    keys.push(format!("{}/{}", prefix, name));
                }
            }
        }

        Ok(keys)
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let path = self.base_path.join(key);
        Ok(path.exists())
    }

    async fn stats(&self) -> Result<BackendStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }
}

// ============================================================================
// VECTOR CHUNK
// ============================================================================

/// A chunk of vectors stored together
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorChunk {
    /// Chunk ID
    pub id: String,

    /// Vector IDs in this chunk
    pub vector_ids: Vec<String>,

    /// Vectors data (flattened)
    pub vectors: Vec<f32>,

    /// Vector dimension
    pub dimension: usize,

    /// Chunk metadata
    pub metadata: ChunkMetadata,
}

/// Chunk metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Number of vectors
    pub count: usize,

    /// Uncompressed size in bytes
    pub uncompressed_size: usize,

    /// Compressed size in bytes (if applicable)
    pub compressed_size: Option<usize>,

    /// Creation timestamp
    pub created_at: i64,

    /// Last access timestamp
    pub last_accessed: i64,

    /// Access count
    pub access_count: u64,
}

impl VectorChunk {
    /// Create new chunk
    pub fn new(id: &str, dimension: usize) -> Self {
        Self {
            id: id.to_string(),
            vector_ids: Vec::new(),
            vectors: Vec::new(),
            dimension,
            metadata: ChunkMetadata {
                count: 0,
                uncompressed_size: 0,
                compressed_size: None,
                created_at: chrono::Utc::now().timestamp(),
                last_accessed: chrono::Utc::now().timestamp(),
                access_count: 0,
            },
        }
    }

    /// Add vector to chunk
    pub fn add(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!(
                "Vector dimension {} doesn't match chunk dimension {}",
                vector.len(),
                self.dimension
            ));
        }

        self.vector_ids.push(id.to_string());
        self.vectors.extend(vector);
        self.metadata.count += 1;
        self.metadata.uncompressed_size += vector.len() * 4;

        Ok(())
    }

    /// Get vector by ID
    pub fn get(&self, id: &str) -> Option<Vec<f32>> {
        self.vector_ids
            .iter()
            .position(|vid| vid == id)
            .map(|idx| {
                let start = idx * self.dimension;
                let end = start + self.dimension;
                self.vectors[start..end].to_vec()
            })
    }

    /// Serialize chunk
    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize chunk
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(data)?)
    }

    /// Compress chunk data
    pub fn compress(&self, level: u32) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let serialized = self.serialize()?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level));
        encoder.write_all(&serialized)?;
        Ok(encoder.finish()?)
    }

    /// Decompress chunk data
    pub fn decompress(data: &[u8]) -> Result<Self> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Self::deserialize(&decompressed)
    }
}

// ============================================================================
// TIER METADATA
// ============================================================================

/// Which tier a vector is currently in
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    Hot,
    Warm,
    Cold,
}

/// Location information for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorLocation {
    pub tier: Tier,
    pub chunk_id: String,
    pub last_accessed: i64,
    pub access_count: u64,
}

// ============================================================================
// TIERED STORE
// ============================================================================

/// Tiered vector store with hot/warm/cold tiers
pub struct TieredStore {
    config: TierConfig,

    /// Vector dimension
    dimension: usize,

    /// Hot tier: in-memory vectors
    hot: Arc<RwLock<HashMap<String, Vec<f32>>>>,

    /// Warm tier chunks
    warm_chunks: Arc<RwLock<HashMap<String, VectorChunk>>>,

    /// Cold storage backend
    cold: Arc<dyn ColdStorageBackend>,

    /// Vector location index
    locations: Arc<RwLock<HashMap<String, VectorLocation>>>,

    /// LRU tracking for hot tier
    hot_lru: Arc<RwLock<VecDeque<String>>>,

    /// Access statistics
    stats: Arc<RwLock<TieredStoreStats>>,
}

/// Store statistics
#[derive(Debug, Clone, Default)]
pub struct TieredStoreStats {
    pub hot_count: usize,
    pub hot_bytes: usize,
    pub warm_count: usize,
    pub warm_bytes: usize,
    pub cold_count: usize,
    pub cold_bytes: usize,
    pub hot_hits: u64,
    pub warm_hits: u64,
    pub cold_hits: u64,
    pub tier_promotions: u64,
    pub tier_demotions: u64,
}

impl TieredStore {
    /// Create new tiered store
    pub async fn new(
        config: TierConfig,
        dimension: usize,
        cold_backend: Arc<dyn ColdStorageBackend>,
    ) -> Result<Self> {
        // Create warm tier directory if needed
        if !config.warm_path.exists() {
            tokio::fs::create_dir_all(&config.warm_path).await?;
        }

        Ok(Self {
            config,
            dimension,
            hot: Arc::new(RwLock::new(HashMap::new())),
            warm_chunks: Arc::new(RwLock::new(HashMap::new())),
            cold: cold_backend,
            locations: Arc::new(RwLock::new(HashMap::new())),
            hot_lru: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(TieredStoreStats::default())),
        })
    }

    /// Add vector (goes to hot tier)
    pub async fn add(&self, id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!(
                "Vector dimension {} doesn't match store dimension {}",
                vector.len(),
                self.dimension
            ));
        }

        // Add to hot tier
        {
            let mut hot = self.hot.write().await;
            hot.insert(id.to_string(), vector.to_vec());
        }

        // Update LRU
        {
            let mut lru = self.hot_lru.write().await;
            lru.push_back(id.to_string());
        }

        // Update location
        {
            let mut locations = self.locations.write().await;
            locations.insert(
                id.to_string(),
                VectorLocation {
                    tier: Tier::Hot,
                    chunk_id: String::new(),
                    last_accessed: chrono::Utc::now().timestamp(),
                    access_count: 0,
                },
            );
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.hot_count += 1;
            stats.hot_bytes += vector.len() * 4;
        }

        // Check if we need to tier down
        self.maybe_tier_down().await?;

        Ok(())
    }

    /// Get vector from any tier
    pub async fn get(&self, id: &str) -> Result<Option<Vec<f32>>> {
        // Check hot tier first
        {
            let hot = self.hot.read().await;
            if let Some(vector) = hot.get(id) {
                let mut stats = self.stats.write().await;
                stats.hot_hits += 1;
                self.update_access(id).await?;
                return Ok(Some(vector.clone()));
            }
        }

        // Check warm tier
        {
            let warm = self.warm_chunks.read().await;
            let locations = self.locations.read().await;

            if let Some(loc) = locations.get(id) {
                if loc.tier == Tier::Warm {
                    if let Some(chunk) = warm.get(&loc.chunk_id) {
                        if let Some(vector) = chunk.get(id) {
                            let mut stats = self.stats.write().await;
                            stats.warm_hits += 1;
                            drop(warm);
                            drop(locations);
                            self.update_access(id).await?;
                            self.maybe_promote(id, &vector).await?;
                            return Ok(Some(vector));
                        }
                    }
                }
            }
        }

        // Check cold tier
        {
            let locations = self.locations.read().await;
            if let Some(loc) = locations.get(id) {
                if loc.tier == Tier::Cold {
                    let chunk_data = self.cold.get(&loc.chunk_id).await?;
                    let chunk = VectorChunk::decompress(&chunk_data)?;

                    if let Some(vector) = chunk.get(id) {
                        let mut stats = self.stats.write().await;
                        stats.cold_hits += 1;
                        drop(locations);
                        self.update_access(id).await?;
                        self.maybe_promote(id, &vector).await?;
                        return Ok(Some(vector));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Search across all tiers
    pub async fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        let mut all_results = Vec::new();

        // Search hot tier
        {
            let hot = self.hot.read().await;
            for (id, vector) in hot.iter() {
                let distance = self.l2_distance(query, vector);
                all_results.push((id.clone(), distance));
            }
        }

        // Search warm tier
        {
            let warm = self.warm_chunks.read().await;
            for chunk in warm.values() {
                for (i, vid) in chunk.vector_ids.iter().enumerate() {
                    let start = i * chunk.dimension;
                    let end = start + chunk.dimension;
                    let vector = &chunk.vectors[start..end];
                    let distance = self.l2_distance(query, vector);
                    all_results.push((vid.clone(), distance));
                }
            }
        }

        // Search cold tier (load all chunks - in production would use index)
        {
            let keys = self.cold.list("chunk_").await?;
            for key in keys {
                if let Ok(chunk_data) = self.cold.get(&key).await {
                    if let Ok(chunk) = VectorChunk::decompress(&chunk_data) {
                        for (i, vid) in chunk.vector_ids.iter().enumerate() {
                            let start = i * chunk.dimension;
                            let end = start + chunk.dimension;
                            let vector = &chunk.vectors[start..end];
                            let distance = self.l2_distance(query, vector);
                            all_results.push((vid.clone(), distance));
                        }
                    }
                }
            }
        }

        // Sort and return top-k
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);

        Ok(all_results)
    }

    /// L2 distance
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Update access time and count
    async fn update_access(&self, id: &str) -> Result<()> {
        let mut locations = self.locations.write().await;
        if let Some(loc) = locations.get_mut(id) {
            loc.last_accessed = chrono::Utc::now().timestamp();
            loc.access_count += 1;
        }
        Ok(())
    }

    /// Maybe promote vector to hotter tier
    async fn maybe_promote(&self, id: &str, vector: &[f32]) -> Result<()> {
        let should_promote = {
            let locations = self.locations.read().await;
            locations
                .get(id)
                .map(|l| l.access_count >= self.config.hot_access_threshold as u64)
                .unwrap_or(false)
        };

        if should_promote {
            // Add to hot tier
            {
                let mut hot = self.hot.write().await;
                hot.insert(id.to_string(), vector.to_vec());
            }

            // Update location
            {
                let mut locations = self.locations.write().await;
                if let Some(loc) = locations.get_mut(id) {
                    loc.tier = Tier::Hot;
                }
            }

            // Update stats
            {
                let mut stats = self.stats.write().await;
                stats.tier_promotions += 1;
            }
        }

        Ok(())
    }

    /// Check if hot tier is over capacity and tier down
    async fn maybe_tier_down(&self) -> Result<()> {
        let hot_bytes = {
            let stats = self.stats.read().await;
            stats.hot_bytes
        };

        let hot_capacity_bytes = self.config.hot_capacity_mb * 1024 * 1024;

        if hot_bytes > hot_capacity_bytes {
            // Find LRU vectors to demote
            let to_demote: Vec<String> = {
                let mut lru = self.hot_lru.write().await;
                let demote_count = (hot_bytes - hot_capacity_bytes) / (self.dimension * 4) + 1;
                (0..demote_count)
                    .filter_map(|_| lru.pop_front())
                    .collect()
            };

            // Demote vectors to warm tier
            for id in to_demote {
                self.demote_to_warm(&id).await?;
            }
        }

        Ok(())
    }

    /// Demote vector from hot to warm tier
    async fn demote_to_warm(&self, id: &str) -> Result<()> {
        let vector = {
            let mut hot = self.hot.write().await;
            hot.remove(id)
        };

        if let Some(vector) = vector {
            // Create or update warm chunk
            let chunk_id = format!("warm_chunk_{}", id.chars().next().unwrap_or('0'));

            {
                let mut warm = self.warm_chunks.write().await;
                let chunk = warm
                    .entry(chunk_id.clone())
                    .or_insert_with(|| VectorChunk::new(&chunk_id, self.dimension));
                chunk.add(id, &vector)?;
            }

            // Update location
            {
                let mut locations = self.locations.write().await;
                if let Some(loc) = locations.get_mut(id) {
                    loc.tier = Tier::Warm;
                    loc.chunk_id = chunk_id;
                }
            }

            // Update stats
            {
                let mut stats = self.stats.write().await;
                stats.hot_count -= 1;
                stats.hot_bytes -= vector.len() * 4;
                stats.warm_count += 1;
                stats.warm_bytes += vector.len() * 4;
                stats.tier_demotions += 1;
            }
        }

        Ok(())
    }

    /// Flush warm tier to cold storage
    pub async fn flush_to_cold(&self) -> Result<usize> {
        let mut flushed = 0;

        let chunks_to_flush: Vec<(String, VectorChunk)> = {
            let warm = self.warm_chunks.read().await;
            warm.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };

        for (chunk_id, chunk) in chunks_to_flush {
            // Compress and upload
            let compressed = chunk.compress(self.config.cold_compression_level)?;
            let cold_key = format!("chunk_{}.gz", chunk_id);
            self.cold.put(&cold_key, &compressed).await?;

            // Update locations
            {
                let mut locations = self.locations.write().await;
                for vid in &chunk.vector_ids {
                    if let Some(loc) = locations.get_mut(vid) {
                        loc.tier = Tier::Cold;
                        loc.chunk_id = cold_key.clone();
                    }
                }
            }

            // Remove from warm
            {
                let mut warm = self.warm_chunks.write().await;
                warm.remove(&chunk_id);
            }

            // Update stats
            {
                let mut stats = self.stats.write().await;
                stats.warm_count -= chunk.metadata.count;
                stats.warm_bytes -= chunk.metadata.uncompressed_size;
                stats.cold_count += chunk.metadata.count;
                stats.cold_bytes += compressed.len();
            }

            flushed += chunk.metadata.count;
        }

        Ok(flushed)
    }

    /// Get store statistics
    pub async fn stats(&self) -> TieredStoreStats {
        self.stats.read().await.clone()
    }

    /// Get tier for a vector
    pub async fn get_tier(&self, id: &str) -> Option<Tier> {
        let locations = self.locations.read().await;
        locations.get(id).map(|l| l.tier)
    }

    /// Get total vector count
    pub async fn len(&self) -> usize {
        let locations = self.locations.read().await;
        locations.len()
    }

    /// Check if empty
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

// ============================================================================
// PREFETCHER
// ============================================================================

/// Background prefetcher for cold data
pub struct Prefetcher {
    store: Arc<TieredStore>,
    queue: Arc<RwLock<VecDeque<String>>>,
    running: Arc<RwLock<bool>>,
}

impl Prefetcher {
    pub fn new(store: Arc<TieredStore>) -> Self {
        Self {
            store,
            queue: Arc::new(RwLock::new(VecDeque::new())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Add IDs to prefetch queue
    pub async fn prefetch(&self, ids: Vec<String>) {
        let mut queue = self.queue.write().await;
        for id in ids {
            if !queue.contains(&id) {
                queue.push_back(id);
            }
        }
    }

    /// Start background prefetching
    pub async fn start(&self) {
        let mut running = self.running.write().await;
        if *running {
            return;
        }
        *running = true;
        drop(running);

        let store = self.store.clone();
        let queue = self.queue.clone();
        let running = self.running.clone();

        tokio::spawn(async move {
            loop {
                let is_running = *running.read().await;
                if !is_running {
                    break;
                }

                let id = {
                    let mut q = queue.write().await;
                    q.pop_front()
                };

                if let Some(id) = id {
                    // Prefetch by accessing (will promote if accessed enough)
                    let _ = store.get(&id).await;
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        });
    }

    /// Stop prefetching
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::rng();
        (0..dim).map(|_| rng.random::<f32>()).collect()
    }

    #[tokio::test]
    async fn test_s3_backend() {
        let backend = S3Backend::new("test-bucket", "vectors/");

        // Put
        let data = vec![1, 2, 3, 4, 5];
        backend.put("test_key", &data).await.unwrap();

        // Get
        let retrieved = backend.get("test_key").await.unwrap();
        assert_eq!(retrieved, data);

        // Exists
        assert!(backend.exists("test_key").await.unwrap());
        assert!(!backend.exists("nonexistent").await.unwrap());

        // List
        let keys = backend.list("").await.unwrap();
        assert!(!keys.is_empty());

        // Delete
        backend.delete("test_key").await.unwrap();
        assert!(!backend.exists("test_key").await.unwrap());
    }

    #[tokio::test]
    async fn test_vector_chunk() {
        let mut chunk = VectorChunk::new("test_chunk", 128);

        // Add vectors
        for i in 0..10 {
            let vector = generate_random_vector(128);
            chunk.add(&format!("vec_{}", i), &vector).unwrap();
        }

        assert_eq!(chunk.metadata.count, 10);

        // Get vector
        let vec = chunk.get("vec_5");
        assert!(vec.is_some());
        assert_eq!(vec.unwrap().len(), 128);

        // Serialize/deserialize
        let serialized = chunk.serialize().unwrap();
        let deserialized = VectorChunk::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.metadata.count, 10);

        // Compress/decompress
        let compressed = chunk.compress(6).unwrap();
        let decompressed = VectorChunk::decompress(&compressed).unwrap();
        assert_eq!(decompressed.metadata.count, 10);

        // Compression should reduce size
        assert!(compressed.len() < serialized.len());
    }

    #[tokio::test]
    async fn test_tiered_store_basic() {
        let config = TierConfig {
            hot_capacity_mb: 1, // Small for testing
            ..Default::default()
        };

        let backend = Arc::new(S3Backend::new("test", "vectors/"));
        let store = TieredStore::new(config, 64, backend).await.unwrap();

        // Add vectors
        for i in 0..10 {
            let vector = generate_random_vector(64);
            store.add(&format!("vec_{}", i), &vector).await.unwrap();
        }

        assert_eq!(store.len().await, 10);

        // Get vector
        let vec = store.get("vec_5").await.unwrap();
        assert!(vec.is_some());
        assert_eq!(vec.unwrap().len(), 64);

        // Check tier
        let tier = store.get_tier("vec_5").await;
        assert!(tier.is_some());
    }

    #[tokio::test]
    async fn test_tiered_store_search() {
        let config = TierConfig::default();
        let backend = Arc::new(S3Backend::new("test", "vectors/"));
        let store = TieredStore::new(config, 64, backend).await.unwrap();

        // Add vectors
        let mut vectors = Vec::new();
        for i in 0..20 {
            let vector = generate_random_vector(64);
            store.add(&format!("vec_{}", i), &vector).await.unwrap();
            vectors.push(vector);
        }

        // Search
        let results = store.search(&vectors[5], 5).await.unwrap();
        assert_eq!(results.len(), 5);

        // First result should be the query itself
        assert_eq!(results[0].0, "vec_5");
        assert!(results[0].1 < 0.001); // Distance should be ~0
    }

    #[tokio::test]
    async fn test_tier_demotion() {
        let config = TierConfig {
            hot_capacity_mb: 0, // Force immediate demotion
            ..Default::default()
        };

        let backend = Arc::new(S3Backend::new("test", "vectors/"));
        let store = TieredStore::new(config, 64, backend).await.unwrap();

        // Add vectors (should trigger demotion)
        for i in 0..5 {
            let vector = generate_random_vector(64);
            store.add(&format!("vec_{}", i), &vector).await.unwrap();
        }

        let stats = store.stats().await;
        // Some vectors should have been demoted
        assert!(stats.warm_count > 0 || stats.tier_demotions > 0);
    }

    #[tokio::test]
    async fn test_flush_to_cold() {
        let config = TierConfig::default();
        let backend = Arc::new(S3Backend::new("test", "vectors/"));
        let store = TieredStore::new(config, 64, backend.clone()).await.unwrap();

        // Add and demote some vectors
        for i in 0..10 {
            let vector = generate_random_vector(64);
            store.add(&format!("vec_{}", i), &vector).await.unwrap();
            store.demote_to_warm(&format!("vec_{}", i)).await.unwrap();
        }

        // Flush to cold
        let flushed = store.flush_to_cold().await.unwrap();
        assert!(flushed > 0);

        // Verify in cold storage
        let cold_stats = backend.stats().await.unwrap();
        assert!(cold_stats.total_objects > 0);
    }
}
