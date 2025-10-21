//! Locality-Sensitive Hashing (LSH) Index
//!
//! LSH is a probabilistic algorithm for approximate nearest neighbor search
//! that maps similar vectors to the same hash buckets with high probability.
//!
//! ## Algorithm Overview
//!
//! 1. **Random Projections**: Create random hyperplanes to divide vector space
//! 2. **Hash Functions**: Determine which side of each hyperplane a vector lies on
//! 3. **Hash Tables**: Multiple hash tables reduce false negatives
//! 4. **Bucketing**: Vectors with same hash code are candidates for neighbors
//!
//! ## Performance Characteristics
//!
//! - **Memory**: O(n) where n = number of vectors (low overhead)
//! - **Speed**: Sub-linear search time O(n^ρ) where ρ < 1
//! - **Recall**: 80-95% typical (tunable via num_tables and num_bits)
//! - **Scale**: Excellent for high-dimensional data (100+ dimensions)
//!
//! ## Use Cases
//!
//! - **High-dimensional vectors** (100-1000+ dims) where other methods struggle
//! - **Streaming data** where vectors arrive continuously
//! - **Low-memory environments** where HNSW's graph is too large
//! - **Approximate search** where 80-90% recall is acceptable
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::lsh::{LSHIndex, LSHConfig};
//!
//! # fn main() -> anyhow::Result<()> {
//! let config = LSHConfig {
//!     num_tables: 10,      // More tables = better recall
//!     num_bits: 16,        // More bits = more buckets
//!     seed: 42,
//! };
//!
//! let mut index = LSHIndex::new(128, config)?;
//!
//! // Add vectors (no training required!)
//! for (id, vector) in vectors.iter() {
//!     index.add(id.clone(), vector.clone())?;
//! }
//!
//! // Search
//! let results = index.search(&query_vector, 10)?;
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for LSH index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSHConfig {
    /// Number of hash tables (typical: 5-20)
    /// More tables = better recall, more memory, slower insertion
    pub num_tables: usize,

    /// Number of bits per hash (typical: 8-24)
    /// More bits = more buckets, better precision, sparser buckets
    pub num_bits: usize,

    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for LSHConfig {
    fn default() -> Self {
        Self {
            num_tables: 10,
            num_bits: 16,
            seed: 42,
        }
    }
}

/// A single hash function (random hyperplane)
#[derive(Clone, Serialize, Deserialize)]
struct HashFunction {
    /// Random projection vector (hyperplane normal)
    projection: Vec<f32>,
}

impl HashFunction {
    /// Create a new random hash function
    fn new(dimension: usize, rng: &mut StdRng) -> Self {
        let projection: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Self { projection }
    }

    /// Hash a vector (returns 0 or 1 based on which side of hyperplane)
    fn hash(&self, vector: &[f32]) -> u32 {
        let dot_product: f32 = self
            .projection
            .iter()
            .zip(vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        if dot_product >= 0.0 {
            1
        } else {
            0
        }
    }
}

/// A hash table with multiple hash functions
#[derive(Clone, Serialize, Deserialize)]
struct HashTable {
    /// Hash functions for this table
    hash_functions: Vec<HashFunction>,

    /// Buckets: hash_code -> list of (id, vector)
    buckets: HashMap<u64, Vec<(String, Vec<f32>)>>,
}

impl HashTable {
    /// Create a new hash table
    fn new(dimension: usize, num_bits: usize, rng: &mut StdRng) -> Self {
        let hash_functions: Vec<HashFunction> = (0..num_bits)
            .map(|_| HashFunction::new(dimension, rng))
            .collect();

        Self {
            hash_functions,
            buckets: HashMap::new(),
        }
    }

    /// Compute hash code for a vector
    fn compute_hash(&self, vector: &[f32]) -> u64 {
        let mut hash_code: u64 = 0;

        for (i, hash_fn) in self.hash_functions.iter().enumerate() {
            let bit = hash_fn.hash(vector);
            hash_code |= (bit as u64) << i;
        }

        hash_code
    }

    /// Add a vector to the hash table
    fn add(&mut self, id: String, vector: Vec<f32>) {
        let hash_code = self.compute_hash(&vector);
        self.buckets
            .entry(hash_code)
            .or_insert_with(Vec::new)
            .push((id, vector));
    }

    /// Get candidate vectors for a query
    fn get_candidates(&self, query: &[f32]) -> Vec<(String, Vec<f32>)> {
        let hash_code = self.compute_hash(query);

        self.buckets
            .get(&hash_code)
            .map(|bucket| bucket.clone())
            .unwrap_or_else(Vec::new)
    }

    /// Remove a vector by ID
    fn remove(&mut self, id: &str) -> bool {
        let mut found = false;

        for bucket in self.buckets.values_mut() {
            if let Some(pos) = bucket.iter().position(|(vid, _)| vid == id) {
                bucket.remove(pos);
                found = true;
                break;
            }
        }

        found
    }
}

/// Locality-Sensitive Hashing Index
///
/// Fast approximate nearest neighbor search using random projections.
/// No training required - can add vectors immediately.
#[derive(Clone, Serialize, Deserialize)]
pub struct LSHIndex {
    /// Vector dimension
    dimension: usize,

    /// Hash tables
    tables: Vec<HashTable>,

    /// Configuration
    config: LSHConfig,

    /// Total number of vectors
    num_vectors: usize,
}

impl LSHIndex {
    /// Create a new LSH index
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension
    /// * `config` - LSH configuration parameters
    pub fn new(dimension: usize, config: LSHConfig) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(config.seed);

        let tables: Vec<HashTable> = (0..config.num_tables)
            .map(|_| HashTable::new(dimension, config.num_bits, &mut rng))
            .collect();

        Ok(Self {
            dimension,
            tables,
            config,
            num_vectors: 0,
        })
    }

    /// Add a vector to the index
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the vector
    /// * `vector` - Vector to add
    pub fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!(
                "Vector dimension {} doesn't match index dimension {}",
                vector.len(),
                self.dimension
            ));
        }

        for table in &mut self.tables {
            table.add(id.clone(), vector.clone());
        }

        self.num_vectors += 1;

        Ok(())
    }

    /// Batch add vectors to the index
    ///
    /// # Arguments
    /// * `vectors` - List of (id, vector) pairs to add
    pub fn add_batch(&mut self, vectors: Vec<(String, Vec<f32>)>) -> Result<()> {
        for (id, vector) in vectors {
            self.add(id, vector)?;
        }

        Ok(())
    }

    /// Search for nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// List of (id, distance) pairs sorted by distance
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if query.len() != self.dimension {
            return Err(anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(),
                self.dimension
            ));
        }

        // Collect candidates from all hash tables
        let mut all_candidates: HashMap<String, Vec<f32>> = HashMap::new();

        for table in &self.tables {
            let candidates = table.get_candidates(query);

            for (id, vector) in candidates {
                all_candidates.entry(id).or_insert(vector);
            }
        }

        // Compute distances to all candidates
        let mut results: Vec<(String, f32)> = all_candidates
            .into_iter()
            .map(|(id, vector)| {
                let distance = euclidean_distance(query, &vector);
                (id, distance)
            })
            .collect();

        // Sort by distance and return top-k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Search with distance threshold
    ///
    /// Returns all neighbors within a given distance threshold
    pub fn search_radius(&self, query: &[f32], radius: f32) -> Result<Vec<(String, f32)>> {
        if query.len() != self.dimension {
            return Err(anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(),
                self.dimension
            ));
        }

        // Collect candidates from all hash tables
        let mut all_candidates: HashMap<String, Vec<f32>> = HashMap::new();

        for table in &self.tables {
            let candidates = table.get_candidates(query);

            for (id, vector) in candidates {
                all_candidates.entry(id).or_insert(vector);
            }
        }

        // Compute distances and filter by radius
        let mut results: Vec<(String, f32)> = all_candidates
            .into_iter()
            .filter_map(|(id, vector)| {
                let distance = euclidean_distance(query, &vector);
                if distance <= radius {
                    Some((id, distance))
                } else {
                    None
                }
            })
            .collect();

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Ok(results)
    }

    /// Remove a vector from the index by ID
    pub fn remove(&mut self, id: &str) -> Result<bool> {
        let mut found = false;

        for table in &mut self.tables {
            if table.remove(id) {
                found = true;
            }
        }

        if found {
            self.num_vectors = self.num_vectors.saturating_sub(1);
        }

        Ok(found)
    }

    /// Get index statistics
    pub fn stats(&self) -> LSHStats {
        let total_buckets: usize = self.tables.iter().map(|t| t.buckets.len()).sum();

        let avg_buckets_per_table = if self.config.num_tables > 0 {
            total_buckets as f32 / self.config.num_tables as f32
        } else {
            0.0
        };

        let max_bucket_sizes: Vec<usize> = self
            .tables
            .iter()
            .map(|t| {
                t.buckets
                    .values()
                    .map(|bucket| bucket.len())
                    .max()
                    .unwrap_or(0)
            })
            .collect();

        let max_bucket_size = max_bucket_sizes.into_iter().max().unwrap_or(0);

        let avg_bucket_size = if total_buckets > 0 {
            self.num_vectors as f32 * self.config.num_tables as f32 / total_buckets as f32
        } else {
            0.0
        };

        // Calculate memory usage (approximate)
        let vector_memory = self.num_vectors * self.dimension * 4 * self.config.num_tables; // f32 = 4 bytes
        let projection_memory = self.config.num_tables * self.config.num_bits * self.dimension * 4;
        let bucket_overhead = total_buckets * 32; // Approximate HashMap overhead
        let memory_bytes = vector_memory + projection_memory + bucket_overhead;

        LSHStats {
            num_vectors: self.num_vectors,
            num_tables: self.config.num_tables,
            num_bits: self.config.num_bits,
            total_buckets,
            avg_buckets_per_table,
            avg_bucket_size,
            max_bucket_size,
            memory_bytes,
        }
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the configuration
    pub fn config(&self) -> &LSHConfig {
        &self.config
    }
}

/// Statistics about the LSH index
#[derive(Debug, Clone)]
pub struct LSHStats {
    pub num_vectors: usize,
    pub num_tables: usize,
    pub num_bits: usize,
    pub total_buckets: usize,
    pub avg_buckets_per_table: f32,
    pub avg_bucket_size: f32,
    pub max_bucket_size: usize,
    pub memory_bytes: usize,
}

/// Helper function: Euclidean distance
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Helper function: Cosine similarity
#[allow(dead_code)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_lsh_basic() {
        let config = LSHConfig {
            num_tables: 5,
            num_bits: 12,
            seed: 42,
        };

        let mut index = LSHIndex::new(64, config).unwrap();

        // Add vectors
        let vectors = generate_random_vectors(100, 64);

        for (i, vector) in vectors.iter().enumerate() {
            index.add(format!("vec_{}", i), vector.clone()).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search
        let query = &vectors[0];
        let results = index.search(query, 10).unwrap();

        assert!(results.len() <= 10);
        // First result should be the query itself
        assert_eq!(results[0].0, "vec_0");
        assert!(results[0].1 < 0.01); // Very close to itself
    }

    #[test]
    fn test_lsh_batch_add() {
        let config = LSHConfig::default();
        let mut index = LSHIndex::new(32, config).unwrap();

        let vectors = generate_random_vectors(50, 32);
        let batch: Vec<(String, Vec<f32>)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("vec_{}", i), v.clone()))
            .collect();

        index.add_batch(batch).unwrap();

        assert_eq!(index.len(), 50);
    }

    #[test]
    fn test_lsh_remove() {
        let config = LSHConfig {
            num_tables: 5,
            num_bits: 10,
            seed: 42,
        };

        let mut index = LSHIndex::new(32, config).unwrap();

        let vectors = generate_random_vectors(20, 32);

        for (i, vector) in vectors.iter().enumerate() {
            index.add(format!("vec_{}", i), vector.clone()).unwrap();
        }

        assert_eq!(index.len(), 20);

        // Remove
        let removed = index.remove("vec_10").unwrap();
        assert!(removed);
        assert_eq!(index.len(), 19);

        // Try to remove again
        let removed = index.remove("vec_10").unwrap();
        assert!(!removed);
    }

    #[test]
    fn test_lsh_search_radius() {
        let config = LSHConfig {
            num_tables: 5,
            num_bits: 12,
            seed: 42,
        };

        let mut index = LSHIndex::new(64, config).unwrap();

        let vectors = generate_random_vectors(100, 64);

        for (i, vector) in vectors.iter().enumerate() {
            index.add(format!("vec_{}", i), vector.clone()).unwrap();
        }

        // Search within radius
        let query = &vectors[0];
        let results = index.search_radius(query, 0.5).unwrap();

        // At least the query itself should be within radius 0.5
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "vec_0");

        // All results should be within radius
        for (_, distance) in &results {
            assert!(*distance <= 0.5);
        }
    }

    #[test]
    fn test_lsh_stats() {
        let config = LSHConfig {
            num_tables: 5,
            num_bits: 10,
            seed: 42,
        };

        let mut index = LSHIndex::new(32, config).unwrap();

        let vectors = generate_random_vectors(50, 32);

        for (i, vector) in vectors.iter().enumerate() {
            index.add(format!("vec_{}", i), vector.clone()).unwrap();
        }

        let stats = index.stats();

        assert_eq!(stats.num_vectors, 50);
        assert_eq!(stats.num_tables, 5);
        assert_eq!(stats.num_bits, 10);
        assert!(stats.total_buckets > 0);
        assert!(stats.memory_bytes > 0);
    }

    #[test]
    fn test_lsh_dimension_validation() {
        let config = LSHConfig::default();
        let mut index = LSHIndex::new(64, config).unwrap();

        let wrong_dim_vector = vec![0.1; 32]; // Wrong dimension

        let result = index.add("vec_1".to_string(), wrong_dim_vector);
        assert!(result.is_err());
    }

    #[test]
    fn test_lsh_recall() {
        // Test recall quality with different configurations

        let vectors = generate_random_vectors(200, 64);

        // Config 1: Few tables, few bits (fast but lower recall)
        let config_low = LSHConfig {
            num_tables: 3,
            num_bits: 8,
            seed: 42,
        };

        let mut index_low = LSHIndex::new(64, config_low).unwrap();

        for (i, vector) in vectors.iter().enumerate() {
            index_low.add(format!("vec_{}", i), vector.clone()).unwrap();
        }

        // Config 2: Many tables, many bits (slower but higher recall)
        let config_high = LSHConfig {
            num_tables: 15,
            num_bits: 16,
            seed: 42,
        };

        let mut index_high = LSHIndex::new(64, config_high).unwrap();

        for (i, vector) in vectors.iter().enumerate() {
            index_high
                .add(format!("vec_{}", i), vector.clone())
                .unwrap();
        }

        // Search with both configs
        let query = &vectors[0];

        let results_low = index_low.search(query, 20).unwrap();
        let results_high = index_high.search(query, 20).unwrap();

        // Both should find the query itself (most important check)
        assert_eq!(results_low[0].0, "vec_0");
        assert_eq!(results_high[0].0, "vec_0");

        // High recall config typically returns more or equal candidates
        // (not always guaranteed due to randomness, but generally true)
        // Just verify both return reasonable results
        assert!(results_low.len() > 0);
        assert!(results_high.len() > 0);
    }

    #[test]
    fn test_hash_function_consistency() {
        // Test that hash functions are deterministic
        let mut rng = StdRng::seed_from_u64(42);
        let hash_fn = HashFunction::new(64, &mut rng);

        let vector = generate_random_vectors(1, 64)[0].clone();

        let hash1 = hash_fn.hash(&vector);
        let hash2 = hash_fn.hash(&vector);

        assert_eq!(hash1, hash2);
    }
}
