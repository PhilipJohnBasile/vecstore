// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Multi-Tier Adaptive Cache
//!
//! Intelligent query caching with bloom filters, LRU eviction, and prefetching.
//! Provides L1 (hot), L2 (warm), and L3 (disk) caching tiers.
//!
//! ## Features
//!
//! - **Bloom Filters**: Fast cache miss detection
//! - **Multi-Tier Architecture**: L1 hot / L2 warm / L3 disk
//! - **Adaptive Prefetching**: Predict and pre-warm common queries
//! - **Semantic Similarity**: Cache hits for similar (not exact) queries
//! - **TTL & LRU Eviction**: Automatic cache management
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::adaptive_cache::{AdaptiveCache, CacheConfig};
//!
//! let config = CacheConfig::default()
//!     .with_l1_size(1000)
//!     .with_l2_size(10000);
//!
//! let cache = AdaptiveCache::new(config);
//! cache.put("query_hash", results);
//! let cached = cache.get("query_hash");
//! ```

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{RwLock, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// L1 (hot) cache size (number of entries)
    pub l1_size: usize,
    /// L2 (warm) cache size (number of entries)
    pub l2_size: usize,
    /// L3 (disk) cache size in bytes
    pub l3_size_bytes: usize,
    /// L1 TTL in seconds
    pub l1_ttl_seconds: u64,
    /// L2 TTL in seconds
    pub l2_ttl_seconds: u64,
    /// L3 TTL in seconds
    pub l3_ttl_seconds: u64,
    /// Bloom filter expected insertions
    pub bloom_expected_items: usize,
    /// Bloom filter false positive rate
    pub bloom_fp_rate: f64,
    /// Enable semantic matching
    pub semantic_matching: bool,
    /// Semantic similarity threshold (0.0-1.0)
    pub similarity_threshold: f32,
    /// Enable prefetching
    pub enable_prefetch: bool,
    /// Prefetch window size
    pub prefetch_window: usize,
    /// Enable disk cache
    pub enable_l3: bool,
    /// L3 cache path
    pub l3_path: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size: 1000,
            l2_size: 10000,
            l3_size_bytes: 1024 * 1024 * 1024, // 1GB
            l1_ttl_seconds: 60,
            l2_ttl_seconds: 300,
            l3_ttl_seconds: 3600,
            bloom_expected_items: 100000,
            bloom_fp_rate: 0.01,
            semantic_matching: true,
            similarity_threshold: 0.95,
            enable_prefetch: true,
            prefetch_window: 100,
            enable_l3: false,
            l3_path: None,
        }
    }
}

impl CacheConfig {
    /// Set L1 size
    #[inline]
    #[must_use]
    pub const fn with_l1_size(mut self, size: usize) -> Self {
        self.l1_size = size;
        self
    }

    /// Set L2 size
    #[inline]
    #[must_use]
    pub const fn with_l2_size(mut self, size: usize) -> Self {
        self.l2_size = size;
        self
    }

    /// Enable L3 disk cache
    #[inline]
    #[must_use]
    pub fn with_l3(mut self, path: &str, size_bytes: usize) -> Self {
        self.enable_l3 = true;
        self.l3_path = Some(path.to_string());
        self.l3_size_bytes = size_bytes;
        self
    }

    /// Set TTLs
    #[inline]
    #[must_use]
    pub const fn with_ttl(mut self, l1_seconds: u64, l2_seconds: u64, l3_seconds: u64) -> Self {
        self.l1_ttl_seconds = l1_seconds;
        self.l2_ttl_seconds = l2_seconds;
        self.l3_ttl_seconds = l3_seconds;
        self
    }
}

/// Bloom filter for fast cache miss detection
pub struct BloomFilter {
    bits: Vec<AtomicU64>,
    num_bits: usize,
    num_hashes: usize,
}

impl BloomFilter {
    /// Create new bloom filter
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        // Calculate optimal size and hash count
        let num_bits = Self::optimal_bits(expected_items, fp_rate);
        let num_hashes = Self::optimal_hashes(num_bits, expected_items);

        let num_words = (num_bits + 63) / 64;
        let bits = (0..num_words).map(|_| AtomicU64::new(0)).collect();

        Self {
            bits,
            num_bits,
            num_hashes,
        }
    }

    #[inline]
    fn optimal_bits(n: usize, p: f64) -> usize {
        (-(n as f64) * p.ln() / (2.0_f64.ln().powi(2))).ceil() as usize
    }

    #[inline]
    fn optimal_hashes(m: usize, n: usize) -> usize {
        ((m as f64 / n as f64) * 2.0_f64.ln()).ceil() as usize
    }

    fn hash(&self, key: &str, seed: usize) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        seed.hash(&mut hasher);
        (hasher.finish() as usize) % self.num_bits
    }

    /// Add key to filter
    pub fn add(&self, key: &str) {
        for i in 0..self.num_hashes {
            let bit = self.hash(key, i);
            let word = bit / 64;
            let bit_in_word = bit % 64;
            self.bits[word].fetch_or(1 << bit_in_word, Ordering::Relaxed);
        }
    }

    /// Check if key might exist
    pub fn might_contain(&self, key: &str) -> bool {
        for i in 0..self.num_hashes {
            let bit = self.hash(key, i);
            let word = bit / 64;
            let bit_in_word = bit % 64;
            if self.bits[word].load(Ordering::Relaxed) & (1 << bit_in_word) == 0 {
                return false;
            }
        }
        true
    }

    /// Clear filter
    pub fn clear(&self) {
        for word in &self.bits {
            word.store(0, Ordering::Relaxed);
        }
    }
}

/// Cache entry
#[derive(Clone)]
pub struct CacheEntry<T: Clone> {
    /// Cached value
    pub value: T,
    /// Creation time
    pub created_at: Instant,
    /// Last access time
    pub last_access: Instant,
    /// Access count
    pub access_count: u64,
    /// Size in bytes (estimated)
    pub size_bytes: usize,
    /// Query vector (for semantic matching)
    pub query_vector: Option<Vec<f32>>,
}

impl<T: Clone> CacheEntry<T> {
    /// Check if entry is expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    /// Update access stats
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;
    }
}

/// Cache tier statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierStats {
    /// Number of entries
    pub entries: usize,
    /// Total size in bytes
    pub size_bytes: usize,
    /// Hit count
    pub hits: u64,
    /// Miss count
    pub misses: u64,
    /// Eviction count
    pub evictions: u64,
}

impl TierStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Search result type for caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    /// Result IDs
    pub ids: Vec<String>,
    /// Scores
    pub scores: Vec<f32>,
    /// Metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Query pattern for prefetching
#[derive(Debug, Clone)]
struct QueryPattern {
    /// Query hash
    query_hash: String,
    /// Occurrence count
    count: usize,
    /// Last seen
    last_seen: Instant,
    /// Query vector (for semantic grouping)
    vector: Option<Vec<f32>>,
}

/// Prefetch predictor
struct PrefetchPredictor {
    /// Recent query patterns
    patterns: VecDeque<QueryPattern>,
    /// Pattern window size
    window_size: usize,
    /// Minimum count to trigger prefetch
    min_count: usize,
}

impl PrefetchPredictor {
    fn new(window_size: usize) -> Self {
        Self {
            patterns: VecDeque::new(),
            window_size,
            min_count: 3,
        }
    }

    fn record(&mut self, query_hash: &str, vector: Option<Vec<f32>>) {
        // Check if pattern exists
        for pattern in &mut self.patterns {
            if pattern.query_hash == query_hash {
                pattern.count += 1;
                pattern.last_seen = Instant::now();
                return;
            }
        }

        // Add new pattern
        self.patterns.push_back(QueryPattern {
            query_hash: query_hash.to_string(),
            count: 1,
            last_seen: Instant::now(),
            vector,
        });

        // Trim old patterns
        while self.patterns.len() > self.window_size {
            self.patterns.pop_front();
        }
    }

    fn should_prefetch(&self, query_hash: &str) -> bool {
        self.patterns.iter()
            .find(|p| p.query_hash == query_hash)
            .map(|p| p.count >= self.min_count)
            .unwrap_or(false)
    }

    fn get_frequent_patterns(&self) -> Vec<String> {
        self.patterns.iter()
            .filter(|p| p.count >= self.min_count)
            .map(|p| p.query_hash.clone())
            .collect()
    }
}

/// Multi-tier adaptive cache
pub struct AdaptiveCache {
    /// Configuration
    config: CacheConfig,
    /// L1 (hot) cache
    l1: RwLock<HashMap<String, CacheEntry<CachedResult>>>,
    /// L2 (warm) cache
    l2: RwLock<HashMap<String, CacheEntry<CachedResult>>>,
    /// L3 (disk) cache - simulated as in-memory for demo
    l3: RwLock<HashMap<String, CacheEntry<CachedResult>>>,
    /// Bloom filter
    bloom: BloomFilter,
    /// Prefetch predictor
    prefetch: RwLock<PrefetchPredictor>,
    /// L1 stats
    l1_stats: RwLock<TierStats>,
    /// L2 stats
    l2_stats: RwLock<TierStats>,
    /// L3 stats
    l3_stats: RwLock<TierStats>,
    /// Total operations
    total_ops: AtomicU64,
}

impl AdaptiveCache {
    /// Create new adaptive cache
    pub fn new(config: CacheConfig) -> Self {
        let bloom = BloomFilter::new(config.bloom_expected_items, config.bloom_fp_rate);
        let prefetch = PrefetchPredictor::new(config.prefetch_window);

        Self {
            config,
            l1: RwLock::new(HashMap::new()),
            l2: RwLock::new(HashMap::new()),
            l3: RwLock::new(HashMap::new()),
            bloom,
            prefetch: RwLock::new(prefetch),
            l1_stats: RwLock::new(TierStats::default()),
            l2_stats: RwLock::new(TierStats::default()),
            l3_stats: RwLock::new(TierStats::default()),
            total_ops: AtomicU64::new(0),
        }
    }

    /// Get from cache
    pub fn get(&self, key: &str) -> Option<CachedResult> {
        self.total_ops.fetch_add(1, Ordering::Relaxed);

        // Quick bloom filter check
        if !self.bloom.might_contain(key) {
            if let Ok(mut stats) = self.l1_stats.write() {
                stats.misses += 1;
            }
            return None;
        }

        // Check L1 (hot)
        {
            if let Ok(mut l1) = self.l1.write() {
                if let Some(entry) = l1.get_mut(key) {
                    if !entry.is_expired(Duration::from_secs(self.config.l1_ttl_seconds)) {
                        entry.touch();
                        if let Ok(mut stats) = self.l1_stats.write() {
                            stats.hits += 1;
                        }
                        return Some(entry.value.clone());
                    } else {
                        l1.remove(key);
                    }
                }
            }
        }

        // Check L2 (warm) and promote to L1
        {
            if let Ok(mut l2) = self.l2.write() {
                if let Some(entry) = l2.remove(key) {
                    if !entry.is_expired(Duration::from_secs(self.config.l2_ttl_seconds)) {
                        if let Ok(mut stats) = self.l2_stats.write() {
                            stats.hits += 1;
                        }
                        let value = entry.value.clone();
                        let _ = self.promote_to_l1(key, entry);
                        return Some(value);
                    }
                }
            }
        }

        // Check L3 (disk) and promote to L2
        if self.config.enable_l3 {
            if let Ok(mut l3) = self.l3.write() {
                if let Some(entry) = l3.remove(key) {
                    if !entry.is_expired(Duration::from_secs(self.config.l3_ttl_seconds)) {
                        if let Ok(mut stats) = self.l3_stats.write() {
                            stats.hits += 1;
                        }
                        let value = entry.value.clone();
                        let _ = self.promote_to_l2(key, entry);
                        return Some(value);
                    }
                }
            }
        }

        // Miss
        if let Ok(mut stats) = self.l1_stats.write() {
            stats.misses += 1;
        }
        None
    }

    /// Get with semantic matching
    pub fn get_semantic(&self, key: &str, query_vector: &[f32]) -> Option<CachedResult> {
        // Try exact match first
        if let Some(result) = self.get(key) {
            return Some(result);
        }

        if !self.config.semantic_matching {
            return None;
        }

        // Try semantic matching in L1
        let l1 = self.l1.read().ok()?;
        for (_, entry) in l1.iter() {
            if let Some(ref cached_vector) = entry.query_vector {
                let similarity = cosine_similarity(query_vector, cached_vector);
                if similarity >= self.config.similarity_threshold {
                    if let Ok(mut stats) = self.l1_stats.write() {
                        stats.hits += 1;
                    }
                    return Some(entry.value.clone());
                }
            }
        }

        None
    }

    /// Put into cache
    pub fn put(&self, key: &str, value: CachedResult) {
        self.put_with_vector(key, value, None);
    }

    /// Put with query vector (for semantic matching)
    pub fn put_with_vector(&self, key: &str, value: CachedResult, query_vector: Option<Vec<f32>>) {
        let size_bytes = estimate_size(&value);

        let entry = CacheEntry {
            value,
            created_at: Instant::now(),
            last_access: Instant::now(),
            access_count: 1,
            size_bytes,
            query_vector: query_vector.clone(),
        };

        // Add to bloom filter
        self.bloom.add(key);

        // Record for prefetching
        if self.config.enable_prefetch {
            if let Ok(mut prefetch) = self.prefetch.write() {
                prefetch.record(key, query_vector);
            }
        }

        // Evict if necessary and insert into L1
        let _ = self.evict_l1_if_needed();

        if let Ok(mut l1) = self.l1.write() {
            l1.insert(key.to_string(), entry);

            if let Ok(mut stats) = self.l1_stats.write() {
                stats.entries = l1.len();
                stats.size_bytes += size_bytes;
            }
        }
    }

    /// Warm cache with prefetch
    pub fn warm(&self, key: &str, value: CachedResult) {
        // Insert directly into L2
        let size_bytes = estimate_size(&value);

        let entry = CacheEntry {
            value,
            created_at: Instant::now(),
            last_access: Instant::now(),
            access_count: 0,
            size_bytes,
            query_vector: None,
        };

        self.bloom.add(key);

        if let Ok(mut l2) = self.l2.write() {
            l2.insert(key.to_string(), entry);
        }
    }

    /// Invalidate cache entry
    pub fn invalidate(&self, key: &str) {
        if let Ok(mut l1) = self.l1.write() {
            l1.remove(key);
        }
        if let Ok(mut l2) = self.l2.write() {
            l2.remove(key);
        }
        if let Ok(mut l3) = self.l3.write() {
            l3.remove(key);
        }
    }

    /// Clear all caches
    pub fn clear(&self) {
        if let Ok(mut l1) = self.l1.write() {
            l1.clear();
        }
        if let Ok(mut l2) = self.l2.write() {
            l2.clear();
        }
        if let Ok(mut l3) = self.l3.write() {
            l3.clear();
        }
        self.bloom.clear();

        if let Ok(mut stats) = self.l1_stats.write() {
            *stats = TierStats::default();
        }
        if let Ok(mut stats) = self.l2_stats.write() {
            *stats = TierStats::default();
        }
        if let Ok(mut stats) = self.l3_stats.write() {
            *stats = TierStats::default();
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            l1: self.l1_stats.read().map(|s| s.clone()).unwrap_or_default(),
            l2: self.l2_stats.read().map(|s| s.clone()).unwrap_or_default(),
            l3: self.l3_stats.read().map(|s| s.clone()).unwrap_or_default(),
            total_operations: self.total_ops.load(Ordering::Relaxed),
        }
    }

    /// Get frequent query patterns for prefetching
    pub fn get_prefetch_candidates(&self) -> Vec<String> {
        self.prefetch.read().map(|p| p.get_frequent_patterns()).unwrap_or_default()
    }

    fn promote_to_l1(&self, key: &str, mut entry: CacheEntry<CachedResult>) -> bool {
        entry.touch();
        let _ = self.evict_l1_if_needed();

        if let Ok(mut l1) = self.l1.write() {
            l1.insert(key.to_string(), entry);
            true
        } else {
            false
        }
    }

    fn promote_to_l2(&self, key: &str, mut entry: CacheEntry<CachedResult>) -> bool {
        entry.touch();
        let _ = self.evict_l2_if_needed();

        if let Ok(mut l2) = self.l2.write() {
            l2.insert(key.to_string(), entry);
            true
        } else {
            false
        }
    }

    fn evict_l1_if_needed(&self) -> bool {
        let Ok(mut l1) = self.l1.write() else {
            return false;
        };

        while l1.len() >= self.config.l1_size {
            // Find LRU entry
            let lru_key = l1.iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| k.clone());

            if let Some(key) = lru_key {
                if let Some(entry) = l1.remove(&key) {
                    // Demote to L2
                    let _ = self.demote_to_l2(&key, entry);
                    if let Ok(mut stats) = self.l1_stats.write() {
                        stats.evictions += 1;
                    }
                }
            } else {
                break;
            }
        }
        true
    }

    fn evict_l2_if_needed(&self) -> bool {
        let Ok(mut l2) = self.l2.write() else {
            return false;
        };

        while l2.len() >= self.config.l2_size {
            let lru_key = l2.iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| k.clone());

            if let Some(key) = lru_key {
                if let Some(entry) = l2.remove(&key) {
                    if self.config.enable_l3 {
                        let _ = self.demote_to_l3(&key, entry);
                    }
                    if let Ok(mut stats) = self.l2_stats.write() {
                        stats.evictions += 1;
                    }
                }
            } else {
                break;
            }
        }
        true
    }

    fn demote_to_l2(&self, key: &str, entry: CacheEntry<CachedResult>) -> bool {
        let _ = self.evict_l2_if_needed();

        if let Ok(mut l2) = self.l2.write() {
            l2.insert(key.to_string(), entry);
            true
        } else {
            false
        }
    }

    fn demote_to_l3(&self, key: &str, entry: CacheEntry<CachedResult>) -> bool {
        // In production, this would write to disk
        if let Ok(mut l3) = self.l3.write() {
            l3.insert(key.to_string(), entry);
            true
        } else {
            false
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// L1 statistics
    pub l1: TierStats,
    /// L2 statistics
    pub l2: TierStats,
    /// L3 statistics
    pub l3: TierStats,
    /// Total operations
    pub total_operations: u64,
}

impl CacheStats {
    /// Overall hit rate
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.l1.hits + self.l2.hits + self.l3.hits;
        let total_misses = self.l1.misses;
        let total = total_hits + total_misses;

        if total > 0 {
            total_hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Query cache key generator
pub struct CacheKeyGenerator;

impl CacheKeyGenerator {
    /// Generate cache key from query parameters
    pub fn generate(
        collection: &str,
        vector: &[f32],
        k: usize,
        filter: Option<&str>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        collection.hash(&mut hasher);
        k.hash(&mut hasher);

        // Hash vector (sample for performance)
        for (i, &v) in vector.iter().enumerate() {
            if i % 8 == 0 {
                (v.to_bits()).hash(&mut hasher);
            }
        }

        if let Some(f) = filter {
            f.hash(&mut hasher);
        }

        format!("{:016x}", hasher.finish())
    }

    /// Generate semantic key (for fuzzy matching)
    pub fn generate_semantic(collection: &str, vector: &[f32]) -> String {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        collection.hash(&mut hasher);

        // Quantize vector for fuzzy matching
        for &v in vector.iter().take(32) {
            let quantized = (v * 100.0) as i32;
            quantized.hash(&mut hasher);
        }

        format!("sem_{:016x}", hasher.finish())
    }
}

fn estimate_size(result: &CachedResult) -> usize {
    let ids_size: usize = result.ids.iter().map(|s| s.len()).sum();
    let scores_size = result.scores.len() * 4;
    let meta_size = result.metadata.as_ref()
        .map(|m| m.len() * 64)
        .unwrap_or(0);

    ids_size + scores_size + meta_size + 64 // overhead
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter() {
        let bloom = BloomFilter::new(1000, 0.01);

        bloom.add("key1");
        bloom.add("key2");

        assert!(bloom.might_contain("key1"));
        assert!(bloom.might_contain("key2"));
        // Note: might have false positives but should work most of the time
    }

    #[test]
    fn test_cache_put_get() {
        let cache = AdaptiveCache::new(CacheConfig::default());

        let result = CachedResult {
            ids: vec!["id1".to_string(), "id2".to_string()],
            scores: vec![0.9, 0.8],
            metadata: None,
        };

        cache.put("test_key", result.clone());

        let cached = cache.get("test_key").unwrap();
        assert_eq!(cached.ids, result.ids);
    }

    #[test]
    fn test_cache_miss() {
        let cache = AdaptiveCache::new(CacheConfig::default());
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let config = CacheConfig::default().with_l1_size(3);
        let cache = AdaptiveCache::new(config);

        for i in 0..5 {
            cache.put(&format!("key{}", i), CachedResult {
                ids: vec![],
                scores: vec![],
                metadata: None,
            });
        }

        // L1 should have at most 3 entries
        assert!(cache.l1.read().map(|l| l.len()).unwrap_or(0) <= 3);
    }

    #[test]
    fn test_semantic_matching() {
        let config = CacheConfig {
            semantic_matching: true,
            similarity_threshold: 0.99,
            ..Default::default()
        };
        let cache = AdaptiveCache::new(config);

        let vector1 = vec![1.0, 0.0, 0.0];
        let vector2 = vec![0.999, 0.001, 0.0]; // Very similar

        cache.put_with_vector("key1", CachedResult {
            ids: vec!["id1".to_string()],
            scores: vec![1.0],
            metadata: None,
        }, Some(vector1.clone()));

        // Should find via semantic matching
        let result = cache.get_semantic("key2", &vector2);
        assert!(result.is_some());
    }

    #[test]
    fn test_cache_key_generation() {
        let vector = vec![0.1, 0.2, 0.3, 0.4];

        let key1 = CacheKeyGenerator::generate("collection1", &vector, 10, None);
        let key2 = CacheKeyGenerator::generate("collection1", &vector, 10, None);
        let key3 = CacheKeyGenerator::generate("collection1", &vector, 20, None);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_stats() {
        let cache = AdaptiveCache::new(CacheConfig::default());

        cache.put("key1", CachedResult {
            ids: vec![],
            scores: vec![],
            metadata: None,
        });

        cache.get("key1");
        cache.get("nonexistent");

        let stats = cache.stats();
        assert!(stats.l1.hits > 0);
        assert!(stats.l1.misses > 0);
    }
}
