// Product Quantization (PQ) for memory compression
//
// Product Quantization is a lossy compression technique that reduces memory usage
// by 8-32x with minimal accuracy loss. Used by production systems like FAISS.
//
// How it works:
// 1. Split each D-dimensional vector into M subvectors of D/M dimensions
// 2. Learn a codebook of K centroids for each subspace using k-means
// 3. Encode each subvector as the index of its nearest centroid (1-2 bytes)
// 4. During search, use asymmetric distance computation with precomputed tables

use super::types::Id;
use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Product Quantizer configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PQConfig {
    /// Number of subvectors (M)
    /// Common values: 8, 16, 32
    /// Higher = better accuracy, more memory
    pub num_subvectors: usize,

    /// Number of centroids per subspace (K)
    /// Common values: 256 (1 byte), 65536 (2 bytes)
    /// Must be power of 2 for optimal packing
    pub num_centroids: usize,

    /// Number of training iterations for k-means
    pub training_iterations: usize,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            num_subvectors: 16,      // Split into 16 subvectors
            num_centroids: 256,      // 1 byte per subvector
            training_iterations: 20, // K-means iterations
        }
    }
}

/// Product Quantizer for vector compression
#[derive(Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Configuration
    config: PQConfig,

    /// Vector dimension
    dimension: usize,

    /// Dimension of each subvector (D/M)
    subvector_dim: usize,

    /// Codebooks: M codebooks, each with K centroids of subvector_dim dimensions
    /// Shape: [num_subvectors][num_centroids][subvector_dim]
    codebooks: Vec<Vec<Vec<f32>>>,

    /// Whether the quantizer has been trained
    trained: bool,
}

impl ProductQuantizer {
    /// Create a new product quantizer
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension
    /// * `config` - PQ configuration
    pub fn new(dimension: usize, config: PQConfig) -> Result<Self> {
        if !dimension.is_multiple_of(config.num_subvectors) {
            return Err(anyhow!(
                "Dimension {} must be divisible by num_subvectors {}",
                dimension,
                config.num_subvectors
            ));
        }

        let subvector_dim = dimension / config.num_subvectors;

        Ok(Self {
            config,
            dimension,
            subvector_dim,
            codebooks: Vec::new(),
            trained: false,
        })
    }

    /// Train the quantizer on a set of vectors
    ///
    /// This learns the codebooks using k-means clustering on each subspace.
    ///
    /// # Arguments
    /// * `training_vectors` - Vectors to train on (should be representative sample)
    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> Result<()> {
        if training_vectors.is_empty() {
            return Err(anyhow!("Need training vectors"));
        }

        if training_vectors.len() < self.config.num_centroids {
            return Err(anyhow!(
                "Need at least {} training vectors (got {})",
                self.config.num_centroids,
                training_vectors.len()
            ));
        }

        #[cfg(not(target_arch = "wasm32"))]
        println!(
            "Training PQ: {} subvectors, {} centroids, {} training vectors",
            self.config.num_subvectors,
            self.config.num_centroids,
            training_vectors.len()
        );

        #[cfg(target_arch = "wasm32")]
        {
            // In WASM, use web_sys console logging if available
            #[cfg(feature = "wasm")]
            web_sys::console::log_1(
                &format!(
                    "Training PQ: {} subvectors, {} centroids, {} training vectors",
                    self.config.num_subvectors,
                    self.config.num_centroids,
                    training_vectors.len()
                )
                .into(),
            );
        }

        // Initialize codebooks
        self.codebooks = Vec::with_capacity(self.config.num_subvectors);

        // Train codebook for each subspace
        #[cfg(not(target_arch = "wasm32"))]
        let codebooks: Vec<Vec<Vec<f32>>> = {
            // Use parallel processing on native platforms
            (0..self.config.num_subvectors)
                .into_par_iter()
                .map(|m| {
                    let start_dim = m * self.subvector_dim;
                    let end_dim = start_dim + self.subvector_dim;

                    // Extract subvectors for this subspace
                    let subvectors: Vec<Vec<f32>> = training_vectors
                        .iter()
                        .map(|v| v[start_dim..end_dim].to_vec())
                        .collect();

                    // Run k-means
                    self.kmeans(&subvectors, self.config.num_centroids).unwrap()
                })
                .collect()
        };

        #[cfg(target_arch = "wasm32")]
        let codebooks: Vec<Vec<Vec<f32>>> = {
            // Use sequential processing on WASM
            (0..self.config.num_subvectors)
                .map(|m| {
                    let start_dim = m * self.subvector_dim;
                    let end_dim = start_dim + self.subvector_dim;

                    // Extract subvectors for this subspace
                    let subvectors: Vec<Vec<f32>> = training_vectors
                        .iter()
                        .map(|v| v[start_dim..end_dim].to_vec())
                        .collect();

                    // Run k-means
                    self.kmeans(&subvectors, self.config.num_centroids).unwrap()
                })
                .collect()
        };

        self.codebooks = codebooks;
        self.trained = true;

        #[cfg(not(target_arch = "wasm32"))]
        println!("✅ PQ training complete");

        #[cfg(target_arch = "wasm32")]
        {
            #[cfg(feature = "wasm")]
            web_sys::console::log_1(&"✅ PQ training complete".into());
        }

        Ok(())
    }

    /// K-means clustering for a set of subvectors
    fn kmeans(&self, vectors: &[Vec<f32>], k: usize) -> Result<Vec<Vec<f32>>> {
        let dim = vectors[0].len();
        let mut rng = rand::thread_rng();

        // Initialize centroids randomly from data
        let mut centroids: Vec<Vec<f32>> = vectors.choose_multiple(&mut rng, k).cloned().collect();

        // K-means iterations
        for _ in 0..self.config.training_iterations {
            // Assign each vector to nearest centroid
            let assignments: Vec<usize> = vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (i, euclidean_distance(v, c)))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap()
                        .0
                })
                .collect();

            // Update centroids
            for (i, centroid) in centroids.iter_mut().enumerate().take(k) {
                let cluster: Vec<&Vec<f32>> = vectors
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| assignments[*idx] == i)
                    .map(|(_, v)| v)
                    .collect();

                if !cluster.is_empty() {
                    *centroid = compute_mean(&cluster, dim);
                }
            }
        }

        Ok(centroids)
    }

    /// Encode a vector into PQ codes
    ///
    /// # Arguments
    /// * `vector` - Full-precision vector
    ///
    /// # Returns
    /// Vector of codes (one per subvector)
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if !self.trained {
            return Err(anyhow!("Quantizer not trained"));
        }

        if vector.len() != self.dimension {
            return Err(anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            ));
        }

        let mut codes = Vec::with_capacity(self.config.num_subvectors);

        for m in 0..self.config.num_subvectors {
            let start_dim = m * self.subvector_dim;
            let end_dim = start_dim + self.subvector_dim;
            let subvector = &vector[start_dim..end_dim];

            // Find nearest centroid in this subspace
            let code = self.codebooks[m]
                .iter()
                .enumerate()
                .map(|(i, centroid)| (i, euclidean_distance(subvector, centroid)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;

            codes.push(code as u8);
        }

        Ok(codes)
    }

    /// Decode PQ codes back to approximate vector
    ///
    /// # Arguments
    /// * `codes` - PQ codes
    ///
    /// # Returns
    /// Reconstructed (approximate) vector
    pub fn decode(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if !self.trained {
            return Err(anyhow!("Quantizer not trained"));
        }

        if codes.len() != self.config.num_subvectors {
            return Err(anyhow!("Invalid number of codes"));
        }

        let mut vector = Vec::with_capacity(self.dimension);

        for (m, &code) in codes.iter().enumerate() {
            let centroid = &self.codebooks[m][code as usize];
            vector.extend_from_slice(centroid);
        }

        Ok(vector)
    }

    /// Compute asymmetric distance from query to encoded vector
    ///
    /// This is the key to fast PQ search. We precompute distances from
    /// query subvectors to all centroids, then lookup during search.
    ///
    /// # Arguments
    /// * `query` - Full-precision query vector
    /// * `codes` - PQ codes of database vector
    /// * `distance_table` - Precomputed distances (from compute_distance_table)
    pub fn asymmetric_distance(&self, codes: &[u8], distance_table: &[Vec<f32>]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(m, &code)| distance_table[m][code as usize])
            .sum()
    }

    /// Precompute distance table for asymmetric distance computation
    ///
    /// Computes distances from query subvectors to all codebook centroids.
    /// This table is reused for all database vectors in a query.
    ///
    /// # Arguments
    /// * `query` - Full-precision query vector
    ///
    /// # Returns
    /// Distance table: \[num_subvectors\]\[num_centroids\]
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut table = Vec::with_capacity(self.config.num_subvectors);

        for m in 0..self.config.num_subvectors {
            let start_dim = m * self.subvector_dim;
            let end_dim = start_dim + self.subvector_dim;
            let query_subvector = &query[start_dim..end_dim];

            let distances: Vec<f32> = self.codebooks[m]
                .iter()
                .map(|centroid| euclidean_distance(query_subvector, centroid))
                .collect();

            table.push(distances);
        }

        table
    }

    /// Get memory usage reduction factor
    ///
    /// Returns how much smaller PQ codes are compared to full vectors.
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.dimension * 4; // 4 bytes per float
        let compressed_size = self.config.num_subvectors; // 1 byte per code
        original_size as f32 / compressed_size as f32
    }

    /// Check if quantizer is trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get configuration
    pub fn config(&self) -> &PQConfig {
        &self.config
    }
}

/// Compressed vector store using Product Quantization
pub struct PQVectorStore {
    /// Product quantizer
    quantizer: ProductQuantizer,

    /// Compressed codes: id -> codes
    codes: HashMap<Id, Vec<u8>>,

    /// Whether the store has been trained
    trained: bool,
}

impl PQVectorStore {
    /// Create a new PQ vector store
    pub fn new(dimension: usize, config: PQConfig) -> Result<Self> {
        Ok(Self {
            quantizer: ProductQuantizer::new(dimension, config)?,
            codes: HashMap::new(),
            trained: false,
        })
    }

    /// Train the quantizer on a set of vectors
    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> Result<()> {
        self.quantizer.train(training_vectors)?;
        self.trained = true;
        Ok(())
    }

    /// Add a vector to the store (after training)
    pub fn add(&mut self, id: Id, vector: &[f32]) -> Result<()> {
        if !self.trained {
            return Err(anyhow!("Store not trained"));
        }

        let codes = self.quantizer.encode(vector)?;
        self.codes.insert(id, codes);
        Ok(())
    }

    /// Search for nearest neighbors using PQ codes
    ///
    /// # Arguments
    /// * `query` - Full-precision query vector
    /// * `k` - Number of results
    ///
    /// # Returns
    /// Vector of (id, distance) pairs, sorted by distance
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(Id, f32)>> {
        if !self.trained {
            return Err(anyhow!("Store not trained"));
        }

        // Precompute distance table (once per query)
        let distance_table = self.quantizer.compute_distance_table(query);

        // Compute distances to all vectors
        #[cfg(not(target_arch = "wasm32"))]
        let mut results: Vec<(Id, f32)> = self
            .codes
            .par_iter()
            .map(|(id, codes)| {
                let distance = self.quantizer.asymmetric_distance(codes, &distance_table);
                (id.clone(), distance)
            })
            .collect();

        #[cfg(target_arch = "wasm32")]
        let mut results: Vec<(Id, f32)> = self
            .codes
            .iter()
            .map(|(id, codes)| {
                let distance = self.quantizer.asymmetric_distance(codes, &distance_table);
                (id.clone(), distance)
            })
            .collect();

        // Sort by distance and return top-k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.quantizer.compression_ratio()
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.codes.len() * self.quantizer.config.num_subvectors
    }
}

// Helper functions

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn compute_mean(vectors: &[&Vec<f32>], dim: usize) -> Vec<f32> {
    let mut mean = vec![0.0; dim];
    let n = vectors.len() as f32;

    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            mean[i] += val / n;
        }
    }

    mean
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| (0..dim).map(|_| rand::random::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_pq_basic() {
        let config = PQConfig {
            num_subvectors: 8,
            num_centroids: 16,
            training_iterations: 5,
        };

        let mut pq = ProductQuantizer::new(64, config).unwrap();

        // Generate training data
        let training_vectors = generate_random_vectors(100, 64);

        // Train
        pq.train(&training_vectors).unwrap();
        assert!(pq.is_trained());

        // Encode and decode
        let vector = &training_vectors[0];
        let codes = pq.encode(vector).unwrap();
        assert_eq!(codes.len(), 8);

        let decoded = pq.decode(&codes).unwrap();
        assert_eq!(decoded.len(), 64);
    }

    #[test]
    fn test_pq_store() {
        let config = PQConfig {
            num_subvectors: 8,
            num_centroids: 256,
            training_iterations: 10,
        };

        let mut store = PQVectorStore::new(64, config).unwrap();

        // Generate data
        let training_vectors = generate_random_vectors(500, 64);

        // Train
        store.train(&training_vectors).unwrap();

        // Add vectors
        for (i, vec) in training_vectors.iter().take(100).enumerate() {
            store.add(format!("vec_{}", i), vec).unwrap();
        }

        assert_eq!(store.len(), 100);

        // Search
        let query = &training_vectors[0];
        let results = store.search(query, 10).unwrap();

        assert_eq!(results.len(), 10);
        assert_eq!(results[0].0, "vec_0"); // Query vector should be closest
    }

    #[test]
    fn test_compression_ratio() {
        let config = PQConfig {
            num_subvectors: 16,
            num_centroids: 256,
            training_iterations: 5,
        };

        let pq = ProductQuantizer::new(128, config).unwrap();

        // 128 floats * 4 bytes = 512 bytes
        // 16 codes * 1 byte = 16 bytes
        // Ratio = 512 / 16 = 32x
        assert_eq!(pq.compression_ratio(), 32.0);
    }
}
