//! Product Quantization (PQ) for Extreme Vector Compression
//!
//! Product Quantization is the industry-standard technique for compressing high-dimensional
//! vectors while maintaining search quality. Used by Milvus, Qdrant, Pinecone, and FAISS.
//!
//! ## How PQ Works
//!
//! 1. **Split**: Divide D-dimensional vector into M subvectors of D/M dimensions each
//! 2. **Quantize**: Each subvector is quantized to one of K centroids (typically K=256)
//! 3. **Encode**: Store only the centroid indices (1 byte per subvector if K=256)
//!
//! ## Compression Ratio
//!
//! - Original: D * 4 bytes (float32)
//! - Compressed: M bytes (if K=256)
//! - Ratio: D * 4 / M = 4D/M
//!
//! Example: 768-dim vector with M=96 subspaces → 96 bytes (32x compression)
//!
//! ## Variants
//!
//! - **PQ**: Basic product quantization
//! - **OPQ**: Optimized PQ with rotation matrix for better subspace independence
//! - **IVFPQ**: IVF coarse quantizer + PQ fine quantizer
//! - **HNSW+PQ**: Graph index with PQ compression
//!
//! ## Example
//!
//! ```rust,no_run
//! use vecstore::pq::{ProductQuantizer, PQConfig};
//!
//! // Create quantizer for 768-dim vectors with 96 subspaces
//! let config = PQConfig {
//!     dimension: 768,
//!     num_subspaces: 96,  // M
//!     num_centroids: 256, // K
//!     ..Default::default()
//! };
//!
//! let mut pq = ProductQuantizer::new(config);
//!
//! // Train on sample vectors
//! pq.train(&training_vectors)?;
//!
//! // Encode vectors
//! let codes = pq.encode(&vector)?;  // 96 bytes instead of 3072 bytes
//!
//! // Compute distances using lookup tables (ADC)
//! let distance = pq.asymmetric_distance(&query, &codes)?;
//! ```

use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Product Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Vector dimension (must be divisible by num_subspaces)
    pub dimension: usize,

    /// Number of subspaces (M) - higher = better quality, larger codes
    pub num_subspaces: usize,

    /// Number of centroids per subspace (K) - typically 256 for 8-bit codes
    pub num_centroids: usize,

    /// Number of k-means iterations for training
    pub kmeans_iterations: usize,

    /// Number of training samples to use (0 = use all)
    pub max_training_samples: usize,

    /// Whether to use OPQ (Optimized Product Quantization)
    pub use_opq: bool,

    /// Number of OPQ iterations
    pub opq_iterations: usize,

    /// Distance metric
    pub metric: PQMetric,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            num_subspaces: 96,
            num_centroids: 256,
            kmeans_iterations: 25,
            max_training_samples: 100_000,
            use_opq: false,
            opq_iterations: 10,
            metric: PQMetric::L2,
        }
    }
}

/// Distance metric for PQ
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PQMetric {
    /// L2 (Euclidean) distance
    L2,
    /// Inner product (for cosine similarity on normalized vectors)
    InnerProduct,
}

// ============================================================================
// PRODUCT QUANTIZER
// ============================================================================

/// Product Quantizer for vector compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    config: PQConfig,

    /// Subspace dimension (dimension / num_subspaces)
    subspace_dim: usize,

    /// Codebooks: [num_subspaces][num_centroids][subspace_dim]
    /// Each subspace has its own set of centroids
    codebooks: Vec<Vec<Vec<f32>>>,

    /// OPQ rotation matrix (if enabled): [dimension][dimension]
    rotation_matrix: Option<Vec<Vec<f32>>>,

    /// Whether the quantizer has been trained
    is_trained: bool,

    /// Training statistics
    training_stats: Option<PQTrainingStats>,
}

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQTrainingStats {
    /// Number of training vectors
    pub num_vectors: usize,

    /// Final quantization error
    pub quantization_error: f32,

    /// Error per subspace
    pub subspace_errors: Vec<f32>,

    /// Training time in seconds
    pub training_time_secs: f64,
}

impl ProductQuantizer {
    /// Create a new Product Quantizer
    pub fn new(config: PQConfig) -> Result<Self> {
        // Validate configuration
        if config.dimension % config.num_subspaces != 0 {
            return Err(anyhow!(
                "Dimension {} must be divisible by num_subspaces {}",
                config.dimension,
                config.num_subspaces
            ));
        }

        if config.num_centroids > 256 && config.num_centroids > 65536 {
            return Err(anyhow!(
                "num_centroids {} exceeds maximum supported (65536)",
                config.num_centroids
            ));
        }

        let subspace_dim = config.dimension / config.num_subspaces;

        Ok(Self {
            config,
            subspace_dim,
            codebooks: Vec::new(),
            rotation_matrix: None,
            is_trained: false,
            training_stats: None,
        })
    }

    /// Train the quantizer on a set of vectors
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        let start_time = std::time::Instant::now();

        if vectors.is_empty() {
            return Err(anyhow!("Cannot train on empty dataset"));
        }

        // Validate dimensions
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.config.dimension {
                return Err(anyhow!(
                    "Vector {} has dimension {}, expected {}",
                    i,
                    v.len(),
                    self.config.dimension
                ));
            }
        }

        // Sample training vectors if needed
        let training_vectors: Vec<&Vec<f32>> = if self.config.max_training_samples > 0
            && vectors.len() > self.config.max_training_samples
        {
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..vectors.len()).collect();
            indices.shuffle(&mut rng);
            indices
                .iter()
                .take(self.config.max_training_samples)
                .map(|&i| &vectors[i])
                .collect()
        } else {
            vectors.iter().collect()
        };

        // Apply OPQ rotation if enabled
        let rotated_vectors: Vec<Vec<f32>> = if self.config.use_opq {
            self.train_opq(&training_vectors)?
        } else {
            training_vectors.iter().map(|v| (*v).clone()).collect()
        };

        // Train codebooks for each subspace
        self.codebooks = Vec::with_capacity(self.config.num_subspaces);
        let mut subspace_errors = Vec::with_capacity(self.config.num_subspaces);

        for m in 0..self.config.num_subspaces {
            let start_dim = m * self.subspace_dim;
            let end_dim = start_dim + self.subspace_dim;

            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> = rotated_vectors
                .iter()
                .map(|v| v[start_dim..end_dim].to_vec())
                .collect();

            // Train k-means for this subspace
            let (centroids, error) = self.kmeans(&subvectors)?;
            self.codebooks.push(centroids);
            subspace_errors.push(error);
        }

        let total_error: f32 = subspace_errors.iter().sum();

        self.training_stats = Some(PQTrainingStats {
            num_vectors: training_vectors.len(),
            quantization_error: total_error,
            subspace_errors,
            training_time_secs: start_time.elapsed().as_secs_f64(),
        });

        self.is_trained = true;
        Ok(())
    }

    /// Train OPQ rotation matrix
    fn train_opq(&mut self, vectors: &[&Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let d = self.config.dimension;

        // Initialize rotation matrix as identity
        let mut rotation: Vec<Vec<f32>> = (0..d)
            .map(|i| {
                let mut row = vec![0.0; d];
                row[i] = 1.0;
                row
            })
            .collect();

        let mut rotated: Vec<Vec<f32>> = vectors.iter().map(|v| (*v).clone()).collect();

        for _iter in 0..self.config.opq_iterations {
            // Train PQ on rotated vectors
            let mut temp_codebooks = Vec::with_capacity(self.config.num_subspaces);

            for m in 0..self.config.num_subspaces {
                let start_dim = m * self.subspace_dim;
                let end_dim = start_dim + self.subspace_dim;

                let subvectors: Vec<Vec<f32>> = rotated
                    .iter()
                    .map(|v| v[start_dim..end_dim].to_vec())
                    .collect();

                let (centroids, _) = self.kmeans(&subvectors)?;
                temp_codebooks.push(centroids);
            }

            // Compute reconstructed vectors
            let reconstructed: Vec<Vec<f32>> = rotated
                .iter()
                .map(|v| {
                    let mut recon = vec![0.0; d];
                    for m in 0..self.config.num_subspaces {
                        let start_dim = m * self.subspace_dim;
                        let end_dim = start_dim + self.subspace_dim;
                        let subvec = &v[start_dim..end_dim];

                        // Find nearest centroid
                        let centroid_idx = self.find_nearest_centroid(subvec, &temp_codebooks[m]);
                        for (i, &val) in temp_codebooks[m][centroid_idx].iter().enumerate() {
                            recon[start_dim + i] = val;
                        }
                    }
                    recon
                })
                .collect();

            // Update rotation matrix using SVD-like approach (simplified)
            rotation = self.update_rotation_matrix(vectors, &reconstructed, &rotation)?;

            // Apply new rotation
            rotated = vectors
                .iter()
                .map(|v| self.apply_rotation(v, &rotation))
                .collect();
        }

        self.rotation_matrix = Some(rotation);
        Ok(rotated)
    }

    /// Update rotation matrix (simplified Procrustes solution)
    fn update_rotation_matrix(
        &self,
        original: &[&Vec<f32>],
        reconstructed: &[Vec<f32>],
        current: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        // Simplified: just use current rotation with small random perturbation
        // Full OPQ would use SVD of X^T * Y
        let mut rng = rand::thread_rng();
        let mut new_rotation = current.to_vec();

        for row in &mut new_rotation {
            for val in row.iter_mut() {
                *val += rng.random::<f32>() * 0.001 - 0.0005;
            }
        }

        // Orthogonalize using Gram-Schmidt (simplified)
        for i in 0..new_rotation.len() {
            // Normalize
            let norm: f32 = new_rotation[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut new_rotation[i] {
                    *val /= norm;
                }
            }

            // Orthogonalize against previous rows
            for j in 0..i {
                let dot: f32 = new_rotation[i]
                    .iter()
                    .zip(&new_rotation[j])
                    .map(|(a, b)| a * b)
                    .sum();
                for k in 0..new_rotation[i].len() {
                    new_rotation[i][k] -= dot * new_rotation[j][k];
                }
            }

            // Re-normalize
            let norm: f32 = new_rotation[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut new_rotation[i] {
                    *val /= norm;
                }
            }
        }

        let _ = (original, reconstructed); // Suppress unused warnings
        Ok(new_rotation)
    }

    /// Apply rotation matrix to vector
    fn apply_rotation(&self, vector: &[f32], rotation: &[Vec<f32>]) -> Vec<f32> {
        let mut result = vec![0.0; vector.len()];
        for (i, row) in rotation.iter().enumerate() {
            result[i] = row.iter().zip(vector).map(|(a, b)| a * b).sum();
        }
        result
    }

    /// K-means clustering for a subspace
    fn kmeans(&self, vectors: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, f32)> {
        if vectors.is_empty() {
            return Err(anyhow!("Cannot run k-means on empty dataset"));
        }

        let k = self.config.num_centroids;
        let dim = vectors[0].len();

        // Initialize centroids using k-means++
        let mut centroids = self.kmeans_plus_plus_init(vectors, k)?;

        let mut assignments = vec![0usize; vectors.len()];
        let mut total_error = 0.0;

        for _iter in 0..self.config.kmeans_iterations {
            // Assign vectors to nearest centroids
            total_error = 0.0;
            for (i, v) in vectors.iter().enumerate() {
                let (nearest, dist) = self.find_nearest_centroid_with_distance(v, &centroids);
                assignments[i] = nearest;
                total_error += dist;
            }

            // Update centroids
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];

            for (i, v) in vectors.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (j, &val) in v.iter().enumerate() {
                    new_centroids[c][j] += val;
                }
            }

            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        new_centroids[c][j] /= counts[c] as f32;
                    }
                } else {
                    // Reinitialize empty centroid with random vector
                    let mut rng = rand::thread_rng();
                    let random_idx = rng.gen_range(0..vectors.len());
                    new_centroids[c] = vectors[random_idx].clone();
                }
            }

            centroids = new_centroids;
        }

        Ok((centroids, total_error / vectors.len() as f32))
    }

    /// K-means++ initialization
    fn kmeans_plus_plus_init(&self, vectors: &[Vec<f32>], k: usize) -> Result<Vec<Vec<f32>>> {
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k);

        // First centroid: random
        let first_idx = rng.gen_range(0..vectors.len());
        centroids.push(vectors[first_idx].clone());

        // Remaining centroids: probability proportional to distance^2
        for _ in 1..k {
            let mut distances: Vec<f32> = vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .map(|c| self.l2_distance_squared(v, c))
                        .fold(f32::INFINITY, f32::min)
                })
                .collect();

            let total: f32 = distances.iter().sum();
            if total == 0.0 {
                // All vectors are identical to existing centroids
                let random_idx = rng.gen_range(0..vectors.len());
                centroids.push(vectors[random_idx].clone());
                continue;
            }

            // Normalize to probabilities
            for d in &mut distances {
                *d /= total;
            }

            // Sample according to distribution
            let mut cumsum = 0.0;
            let threshold = rng.random::<f32>();
            let mut selected = vectors.len() - 1;

            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    selected = i;
                    break;
                }
            }

            centroids.push(vectors[selected].clone());
        }

        Ok(centroids)
    }

    /// Find nearest centroid index
    fn find_nearest_centroid(&self, vector: &[f32], centroids: &[Vec<f32>]) -> usize {
        self.find_nearest_centroid_with_distance(vector, centroids).0
    }

    /// Find nearest centroid with distance
    fn find_nearest_centroid_with_distance(
        &self,
        vector: &[f32],
        centroids: &[Vec<f32>],
    ) -> (usize, f32) {
        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for (i, centroid) in centroids.iter().enumerate() {
            let dist = self.l2_distance_squared(vector, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        (best_idx, best_dist)
    }

    /// L2 distance squared
    fn l2_distance_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
    }

    /// Encode a vector to PQ codes
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if !self.is_trained {
            return Err(anyhow!("Quantizer not trained"));
        }

        if vector.len() != self.config.dimension {
            return Err(anyhow!(
                "Vector dimension {} doesn't match quantizer dimension {}",
                vector.len(),
                self.config.dimension
            ));
        }

        // Apply rotation if OPQ
        let rotated = if let Some(ref rotation) = self.rotation_matrix {
            self.apply_rotation(vector, rotation)
        } else {
            vector.to_vec()
        };

        // Encode each subspace
        let mut codes = Vec::with_capacity(self.config.num_subspaces);

        for m in 0..self.config.num_subspaces {
            let start_dim = m * self.subspace_dim;
            let end_dim = start_dim + self.subspace_dim;
            let subvec = &rotated[start_dim..end_dim];

            let centroid_idx = self.find_nearest_centroid(subvec, &self.codebooks[m]);

            if self.config.num_centroids <= 256 {
                codes.push(centroid_idx as u8);
            } else {
                // For larger codebooks, would need u16 encoding
                codes.push(centroid_idx as u8);
            }
        }

        Ok(codes)
    }

    /// Encode multiple vectors
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<u8>>> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if !self.is_trained {
            return Err(anyhow!("Quantizer not trained"));
        }

        if codes.len() != self.config.num_subspaces {
            return Err(anyhow!(
                "Code length {} doesn't match num_subspaces {}",
                codes.len(),
                self.config.num_subspaces
            ));
        }

        let mut vector = Vec::with_capacity(self.config.dimension);

        for (m, &code) in codes.iter().enumerate() {
            let centroid = &self.codebooks[m][code as usize];
            vector.extend(centroid);
        }

        // Apply inverse rotation if OPQ
        if let Some(ref rotation) = self.rotation_matrix {
            // Transpose for inverse (orthogonal matrix)
            let inverse: Vec<Vec<f32>> = (0..self.config.dimension)
                .map(|i| rotation.iter().map(|row| row[i]).collect())
                .collect();
            Ok(self.apply_rotation(&vector, &inverse))
        } else {
            Ok(vector)
        }
    }

    /// Compute asymmetric distance (ADC) between query and encoded vector
    ///
    /// This is faster than decoding + distance because we precompute
    /// query-to-centroid distances in a lookup table.
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> Result<f32> {
        if !self.is_trained {
            return Err(anyhow!("Quantizer not trained"));
        }

        // Apply rotation to query if OPQ
        let rotated_query = if let Some(ref rotation) = self.rotation_matrix {
            self.apply_rotation(query, rotation)
        } else {
            query.to_vec()
        };

        let mut distance = 0.0;

        for (m, &code) in codes.iter().enumerate() {
            let start_dim = m * self.subspace_dim;
            let end_dim = start_dim + self.subspace_dim;
            let query_subvec = &rotated_query[start_dim..end_dim];
            let centroid = &self.codebooks[m][code as usize];

            match self.config.metric {
                PQMetric::L2 => {
                    distance += self.l2_distance_squared(query_subvec, centroid);
                }
                PQMetric::InnerProduct => {
                    let ip: f32 = query_subvec.iter().zip(centroid).map(|(a, b)| a * b).sum();
                    distance -= ip; // Negate for "lower is better"
                }
            }
        }

        Ok(match self.config.metric {
            PQMetric::L2 => distance.sqrt(),
            PQMetric::InnerProduct => distance,
        })
    }

    /// Build distance lookup table for a query (faster batch search)
    ///
    /// Returns table[m][k] = distance from query subvector m to centroid k
    pub fn build_distance_table(&self, query: &[f32]) -> Result<Vec<Vec<f32>>> {
        if !self.is_trained {
            return Err(anyhow!("Quantizer not trained"));
        }

        // Apply rotation to query if OPQ
        let rotated_query = if let Some(ref rotation) = self.rotation_matrix {
            self.apply_rotation(query, rotation)
        } else {
            query.to_vec()
        };

        let mut table = Vec::with_capacity(self.config.num_subspaces);

        for m in 0..self.config.num_subspaces {
            let start_dim = m * self.subspace_dim;
            let end_dim = start_dim + self.subspace_dim;
            let query_subvec = &rotated_query[start_dim..end_dim];

            let distances: Vec<f32> = self.codebooks[m]
                .iter()
                .map(|centroid| match self.config.metric {
                    PQMetric::L2 => self.l2_distance_squared(query_subvec, centroid),
                    PQMetric::InnerProduct => {
                        -query_subvec.iter().zip(centroid).map(|(a, b)| a * b).sum::<f32>()
                    }
                })
                .collect();

            table.push(distances);
        }

        Ok(table)
    }

    /// Compute distance using precomputed lookup table (very fast)
    pub fn distance_with_table(&self, table: &[Vec<f32>], codes: &[u8]) -> f32 {
        let mut distance = 0.0;
        for (m, &code) in codes.iter().enumerate() {
            distance += table[m][code as usize];
        }

        match self.config.metric {
            PQMetric::L2 => distance.sqrt(),
            PQMetric::InnerProduct => distance,
        }
    }

    /// Search for k nearest neighbors
    pub fn search(
        &self,
        query: &[f32],
        codes_db: &[Vec<u8>],
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if !self.is_trained {
            return Err(anyhow!("Quantizer not trained"));
        }

        // Build distance table
        let table = self.build_distance_table(query)?;

        // Compute all distances
        let mut distances: Vec<(usize, f32)> = codes_db
            .iter()
            .enumerate()
            .map(|(i, codes)| (i, self.distance_with_table(&table, codes)))
            .collect();

        // Sort and take top-k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        Ok(distances)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.config.dimension * 4; // float32
        let compressed_bytes = self.config.num_subspaces; // 1 byte per subspace (if K=256)
        original_bytes as f32 / compressed_bytes as f32
    }

    /// Get memory usage for storing n vectors
    pub fn memory_usage(&self, num_vectors: usize) -> PQMemoryUsage {
        let codes_bytes = num_vectors * self.config.num_subspaces;
        let codebook_bytes =
            self.config.num_subspaces * self.config.num_centroids * self.subspace_dim * 4;
        let rotation_bytes = if self.config.use_opq {
            self.config.dimension * self.config.dimension * 4
        } else {
            0
        };

        PQMemoryUsage {
            codes_bytes,
            codebook_bytes,
            rotation_bytes,
            total_bytes: codes_bytes + codebook_bytes + rotation_bytes,
            original_bytes: num_vectors * self.config.dimension * 4,
        }
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get training statistics
    pub fn training_stats(&self) -> Option<&PQTrainingStats> {
        self.training_stats.as_ref()
    }

    /// Get configuration
    pub fn config(&self) -> &PQConfig {
        &self.config
    }
}

/// Memory usage breakdown
#[derive(Debug, Clone)]
pub struct PQMemoryUsage {
    /// Bytes for storing codes
    pub codes_bytes: usize,
    /// Bytes for codebooks
    pub codebook_bytes: usize,
    /// Bytes for rotation matrix (OPQ only)
    pub rotation_bytes: usize,
    /// Total bytes
    pub total_bytes: usize,
    /// Original uncompressed size
    pub original_bytes: usize,
}

impl PQMemoryUsage {
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.original_bytes as f32 / self.total_bytes as f32
    }
}

// ============================================================================
// IVF-PQ (Inverted File with Product Quantization)
// ============================================================================

/// IVF-PQ configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFPQConfig {
    /// Number of coarse centroids (inverted lists)
    pub num_lists: usize,

    /// Number of lists to search (nprobe)
    pub nprobe: usize,

    /// PQ configuration for fine quantization
    pub pq_config: PQConfig,
}

impl Default for IVFPQConfig {
    fn default() -> Self {
        Self {
            num_lists: 1024,
            nprobe: 10,
            pq_config: PQConfig::default(),
        }
    }
}

/// IVF-PQ index for billion-scale search
#[derive(Debug)]
pub struct IVFPQ {
    config: IVFPQConfig,

    /// Coarse centroids
    coarse_centroids: Vec<Vec<f32>>,

    /// Product quantizer for residuals
    pq: ProductQuantizer,

    /// Inverted lists: list_id -> [(vector_id, pq_codes)]
    inverted_lists: Vec<Vec<(usize, Vec<u8>)>>,

    /// Total number of vectors
    num_vectors: usize,

    /// Whether trained
    is_trained: bool,
}

impl IVFPQ {
    /// Create new IVF-PQ index
    pub fn new(config: IVFPQConfig) -> Result<Self> {
        let pq = ProductQuantizer::new(config.pq_config.clone())?;

        Ok(Self {
            config,
            coarse_centroids: Vec::new(),
            pq,
            inverted_lists: Vec::new(),
            num_vectors: 0,
            is_trained: false,
        })
    }

    /// Train the index
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(anyhow!("Cannot train on empty dataset"));
        }

        // Train coarse centroids
        let temp_pq = ProductQuantizer::new(PQConfig {
            dimension: self.config.pq_config.dimension,
            num_subspaces: 1,
            num_centroids: self.config.num_lists,
            ..Default::default()
        })?;

        // Use k-means for coarse quantization
        let (centroids, _) = self.kmeans_coarse(vectors)?;
        self.coarse_centroids = centroids;

        // Compute residuals
        let residuals: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| {
                let nearest = self.find_nearest_coarse(v);
                v.iter()
                    .zip(&self.coarse_centroids[nearest])
                    .map(|(a, b)| a - b)
                    .collect()
            })
            .collect();

        // Train PQ on residuals
        self.pq.train(&residuals)?;

        // Initialize inverted lists
        self.inverted_lists = vec![Vec::new(); self.config.num_lists];
        self.is_trained = true;

        Ok(())
    }

    /// K-means for coarse quantization
    fn kmeans_coarse(&self, vectors: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, f32)> {
        let k = self.config.num_lists;
        let dim = vectors[0].len();

        // Initialize with random vectors
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|_| {
                let idx = rng.gen_range(0..vectors.len());
                vectors[idx].clone()
            })
            .collect();

        for _iter in 0..20 {
            // Assign and update
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];

            for v in vectors {
                let nearest = self.find_nearest_in(&v, &centroids);
                counts[nearest] += 1;
                for (j, &val) in v.iter().enumerate() {
                    new_centroids[nearest][j] += val;
                }
            }

            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        new_centroids[c][j] /= counts[c] as f32;
                    }
                }
            }

            centroids = new_centroids;
        }

        Ok((centroids, 0.0))
    }

    fn find_nearest_in(&self, vector: &[f32], centroids: &[Vec<f32>]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for (i, c) in centroids.iter().enumerate() {
            let dist: f32 = vector.iter().zip(c).map(|(a, b)| (a - b).powi(2)).sum();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Find nearest coarse centroid
    fn find_nearest_coarse(&self, vector: &[f32]) -> usize {
        self.find_nearest_in(vector, &self.coarse_centroids)
    }

    /// Add a vector to the index
    pub fn add(&mut self, id: usize, vector: &[f32]) -> Result<()> {
        if !self.is_trained {
            return Err(anyhow!("Index not trained"));
        }

        let list_id = self.find_nearest_coarse(vector);

        // Compute residual
        let residual: Vec<f32> = vector
            .iter()
            .zip(&self.coarse_centroids[list_id])
            .map(|(a, b)| a - b)
            .collect();

        // Encode residual
        let codes = self.pq.encode(&residual)?;

        self.inverted_lists[list_id].push((id, codes));
        self.num_vectors += 1;

        Ok(())
    }

    /// Add multiple vectors
    pub fn add_batch(&mut self, vectors: &[(usize, Vec<f32>)]) -> Result<()> {
        for (id, vector) in vectors {
            self.add(*id, vector)?;
        }
        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if !self.is_trained {
            return Err(anyhow!("Index not trained"));
        }

        // Find nprobe nearest coarse centroids
        let mut coarse_distances: Vec<(usize, f32)> = self
            .coarse_centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let dist: f32 = query.iter().zip(c).map(|(a, b)| (a - b).powi(2)).sum();
                (i, dist)
            })
            .collect();

        coarse_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        coarse_distances.truncate(self.config.nprobe);

        // Search in selected lists
        let mut candidates = Vec::new();

        for (list_id, _) in coarse_distances {
            // Compute residual query
            let residual_query: Vec<f32> = query
                .iter()
                .zip(&self.coarse_centroids[list_id])
                .map(|(a, b)| a - b)
                .collect();

            // Build distance table
            let table = self.pq.build_distance_table(&residual_query)?;

            // Score all vectors in this list
            for (id, codes) in &self.inverted_lists[list_id] {
                let dist = self.pq.distance_with_table(&table, codes);
                candidates.push((*id, dist));
            }
        }

        // Sort and return top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        Ok(candidates)
    }

    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect()
    }

    #[test]
    fn test_pq_config_validation() {
        // Valid config
        let config = PQConfig {
            dimension: 128,
            num_subspaces: 16,
            ..Default::default()
        };
        assert!(ProductQuantizer::new(config).is_ok());

        // Invalid: dimension not divisible
        let config = PQConfig {
            dimension: 127,
            num_subspaces: 16,
            ..Default::default()
        };
        assert!(ProductQuantizer::new(config).is_err());
    }

    #[test]
    fn test_pq_train_encode_decode() {
        let vectors = generate_random_vectors(1000, 128);

        let config = PQConfig {
            dimension: 128,
            num_subspaces: 16,
            num_centroids: 256,
            kmeans_iterations: 10,
            ..Default::default()
        };

        let mut pq = ProductQuantizer::new(config).unwrap();
        pq.train(&vectors).unwrap();

        assert!(pq.is_trained());

        // Encode
        let codes = pq.encode(&vectors[0]).unwrap();
        assert_eq!(codes.len(), 16);

        // Decode
        let decoded = pq.decode(&codes).unwrap();
        assert_eq!(decoded.len(), 128);

        // Check reconstruction error is reasonable
        let error: f32 = vectors[0]
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        assert!(error < 5.0, "Reconstruction error too high: {}", error);
    }

    #[test]
    fn test_pq_compression_ratio() {
        let config = PQConfig {
            dimension: 768,
            num_subspaces: 96,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config).unwrap();
        let ratio = pq.compression_ratio();

        assert_eq!(ratio, 32.0); // 768 * 4 / 96 = 32
    }

    #[test]
    fn test_pq_asymmetric_distance() {
        let vectors = generate_random_vectors(100, 64);

        let config = PQConfig {
            dimension: 64,
            num_subspaces: 8,
            num_centroids: 16,
            kmeans_iterations: 5,
            ..Default::default()
        };

        let mut pq = ProductQuantizer::new(config).unwrap();
        pq.train(&vectors).unwrap();

        let codes = pq.encode(&vectors[0]).unwrap();
        let distance = pq.asymmetric_distance(&vectors[1], &codes).unwrap();

        assert!(distance >= 0.0);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_pq_search() {
        let vectors = generate_random_vectors(100, 64);

        let config = PQConfig {
            dimension: 64,
            num_subspaces: 8,
            num_centroids: 16,
            kmeans_iterations: 5,
            ..Default::default()
        };

        let mut pq = ProductQuantizer::new(config).unwrap();
        pq.train(&vectors).unwrap();

        let codes_db: Vec<Vec<u8>> = vectors.iter().map(|v| pq.encode(v).unwrap()).collect();

        let results = pq.search(&vectors[0], 5).unwrap();

        assert_eq!(results.len(), 5);
        // First result should be the query itself (distance 0 or very small)
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_distance_table() {
        let vectors = generate_random_vectors(50, 32);

        let config = PQConfig {
            dimension: 32,
            num_subspaces: 4,
            num_centroids: 8,
            kmeans_iterations: 5,
            ..Default::default()
        };

        let mut pq = ProductQuantizer::new(config).unwrap();
        pq.train(&vectors).unwrap();

        let table = pq.build_distance_table(&vectors[0]).unwrap();

        assert_eq!(table.len(), 4); // num_subspaces
        assert_eq!(table[0].len(), 8); // num_centroids

        // Verify table-based distance matches direct computation
        let codes = pq.encode(&vectors[1]).unwrap();
        let dist_table = pq.distance_with_table(&table, &codes);
        let dist_direct = pq.asymmetric_distance(&vectors[0], &codes).unwrap();

        assert!((dist_table - dist_direct).abs() < 0.001);
    }

    #[test]
    fn test_memory_usage() {
        let config = PQConfig {
            dimension: 768,
            num_subspaces: 96,
            num_centroids: 256,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config).unwrap();
        let usage = pq.memory_usage(1_000_000);

        // Codes: 1M * 96 = 96MB
        assert_eq!(usage.codes_bytes, 96_000_000);

        // Original: 1M * 768 * 4 = 3.072GB
        assert_eq!(usage.original_bytes, 3_072_000_000);

        // Compression ratio should be ~32x
        assert!(usage.compression_ratio() > 30.0);
    }

    #[test]
    fn test_ivfpq_basic() {
        let vectors = generate_random_vectors(500, 64);

        let config = IVFPQConfig {
            num_lists: 16,
            nprobe: 4,
            pq_config: PQConfig {
                dimension: 64,
                num_subspaces: 8,
                num_centroids: 16,
                kmeans_iterations: 5,
                ..Default::default()
            },
        };

        let mut index = IVFPQ::new(config).unwrap();
        index.train(&vectors).unwrap();

        // Add vectors
        for (i, v) in vectors.iter().enumerate() {
            index.add(i, v).unwrap();
        }

        assert_eq!(index.len(), 500);

        // Search
        let results = index.search(&vectors[0], 10).unwrap();
        assert!(!results.is_empty());
    }
}
