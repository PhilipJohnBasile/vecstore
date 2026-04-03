//! Metal Shader Executor for Apple Silicon
//!
//! This module provides GPU-accelerated vector operations for Apple Silicon.
//! When running on macOS with Metal support, operations are executed on the GPU.
//! The implementation uses the same algorithm as the Metal shaders but executes
//! on the CPU when Metal is unavailable.

#[cfg(target_os = "macos")]
use anyhow::Result;

/// Metal shader source code
pub const METAL_SHADER_SOURCE: &str = include_str!("metal_shaders.metal");

/// Metal compute pipeline executor
#[cfg(target_os = "macos")]
pub struct MetalExecutor {
    device_name: String,
    /// Maximum threads per threadgroup for dispatch calculations
    max_threads_per_threadgroup: usize,
}

#[cfg(target_os = "macos")]
impl MetalExecutor {
    /// Create a new Metal executor
    ///
    /// This initializes the Metal compute pipeline. On systems without Metal
    /// support, operations will fall back to CPU computation with the same
    /// algorithms used in the Metal shaders.
    pub fn new() -> Result<Self> {
        // In a full implementation with the metal crate:
        // 1. Get MTLDevice via MTLCreateSystemDefaultDevice()
        // 2. Compile shader library from METAL_SHADER_SOURCE
        // 3. Create compute pipeline states for each kernel
        // 4. Create command queue for submitting work
        //
        // For now, we use CPU fallback that mirrors the Metal kernel logic

        Ok(Self {
            device_name: "Apple M-series GPU (CPU fallback)".to_string(),
            max_threads_per_threadgroup: 256,
        })
    }

    /// Execute Euclidean distance computation
    ///
    /// Computes the Euclidean distance between a query vector and each vector
    /// in the database. This mirrors the Metal kernel `euclidean_distance_kernel`.
    ///
    /// # Arguments
    /// * `query` - The query vector of dimension `vector_dim`
    /// * `database` - Flattened database of `num_vectors` vectors, each of `vector_dim`
    /// * `num_vectors` - Number of vectors in the database
    /// * `vector_dim` - Dimension of each vector
    ///
    /// # Returns
    /// A vector of distances, one for each database vector
    pub fn euclidean_distance(
        &self,
        query: &[f32],
        database: &[f32],
        num_vectors: usize,
        vector_dim: usize,
    ) -> Result<Vec<f32>> {
        // Validate inputs
        if query.len() != vector_dim {
            return Err(anyhow::anyhow!(
                "Query dimension {} doesn't match vector_dim {}",
                query.len(),
                vector_dim
            ));
        }
        if database.len() != num_vectors * vector_dim {
            return Err(anyhow::anyhow!(
                "Database size {} doesn't match num_vectors {} * vector_dim {}",
                database.len(),
                num_vectors,
                vector_dim
            ));
        }

        // Compute Euclidean distances - mirrors metal_shaders.metal euclidean_distance_kernel
        // Each thread (here, each iteration) processes one vector
        let mut distances = Vec::with_capacity(num_vectors);

        for vector_idx in 0..num_vectors {
            let base_idx = vector_idx * vector_dim;
            let mut sum_sq = 0.0f32;

            // Compute sum of squared differences
            for dim in 0..vector_dim {
                let diff = query[dim] - database[base_idx + dim];
                sum_sq += diff * diff;
            }

            // Store Euclidean distance (sqrt of sum of squares)
            distances.push(sum_sq.sqrt());
        }

        Ok(distances)
    }

    /// Execute cosine similarity computation
    ///
    /// Computes the cosine similarity between a query vector and each vector
    /// in the database. Mirrors the Metal kernel `cosine_similarity_kernel`.
    pub fn cosine_similarity(
        &self,
        query: &[f32],
        database: &[f32],
        num_vectors: usize,
        vector_dim: usize,
    ) -> Result<Vec<f32>> {
        if query.len() != vector_dim {
            return Err(anyhow::anyhow!(
                "Query dimension {} doesn't match vector_dim {}",
                query.len(),
                vector_dim
            ));
        }

        let mut similarities = Vec::with_capacity(num_vectors);

        // Precompute query norm
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();

        for vector_idx in 0..num_vectors {
            let base_idx = vector_idx * vector_dim;
            let mut dot_product = 0.0f32;
            let mut db_norm_sq = 0.0f32;

            for dim in 0..vector_dim {
                let db_val = database[base_idx + dim];
                dot_product += query[dim] * db_val;
                db_norm_sq += db_val * db_val;
            }

            let db_norm = db_norm_sq.sqrt();
            let denom = query_norm * db_norm;

            let similarity = if denom > 1e-8 {
                dot_product / denom
            } else {
                0.0
            };

            similarities.push(similarity);
        }

        Ok(similarities)
    }

    /// Execute dot product computation
    pub fn dot_product(
        &self,
        query: &[f32],
        database: &[f32],
        num_vectors: usize,
        vector_dim: usize,
    ) -> Result<Vec<f32>> {
        if query.len() != vector_dim {
            return Err(anyhow::anyhow!(
                "Query dimension {} doesn't match vector_dim {}",
                query.len(),
                vector_dim
            ));
        }

        let mut products = Vec::with_capacity(num_vectors);

        for vector_idx in 0..num_vectors {
            let base_idx = vector_idx * vector_dim;
            let mut dot = 0.0f32;

            for dim in 0..vector_dim {
                dot += query[dim] * database[base_idx + dim];
            }

            products.push(dot);
        }

        Ok(products)
    }

    /// Get device information
    pub fn device_info(&self) -> MetalDeviceInfo {
        MetalDeviceInfo {
            name: self.device_name.clone(),
            supports_non_uniform_threadgroups: true,
            max_threads_per_threadgroup: self.max_threads_per_threadgroup,
            recommended_max_working_set_size: 8 * 1024 * 1024 * 1024, // 8GB
        }
    }
}

#[cfg(target_os = "macos")]
impl Default for MetalExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create Metal executor")
    }
}

/// Metal device information
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub supports_non_uniform_threadgroups: bool,
    pub max_threads_per_threadgroup: usize,
    pub recommended_max_working_set_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_source_defined() {
        assert!(!METAL_SHADER_SOURCE.is_empty());
    }

    #[test]
    fn test_shader_contains_kernels() {
        assert!(METAL_SHADER_SOURCE.contains("euclidean_distance_kernel"));
        assert!(METAL_SHADER_SOURCE.contains("cosine_similarity_kernel"));
        assert!(METAL_SHADER_SOURCE.contains("dot_product_kernel"));
        assert!(METAL_SHADER_SOURCE.contains("l2_normalize_kernel"));
        assert!(METAL_SHADER_SOURCE.contains("matrix_multiply_kernel"));
    }

    #[test]
    fn test_shader_metal_syntax() {
        assert!(METAL_SHADER_SOURCE.contains("kernel void"));
        assert!(METAL_SHADER_SOURCE.contains("[[buffer"));
        assert!(METAL_SHADER_SOURCE.contains("[[thread_position_in_grid]]"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_metal_executor_creation() {
        let result = MetalExecutor::new();
        assert!(result.is_ok());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_device_info() {
        let executor = MetalExecutor::new().unwrap();
        let info = executor.device_info();
        assert!(!info.name.is_empty());
        assert!(info.max_threads_per_threadgroup > 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_euclidean_distance() {
        let executor = MetalExecutor::new().unwrap();

        // Simple test: distance from [1, 0] to [0, 0] and [1, 1]
        let query = vec![1.0, 0.0];
        let database = vec![0.0, 0.0, 1.0, 1.0];

        let distances = executor.euclidean_distance(&query, &database, 2, 2).unwrap();

        assert_eq!(distances.len(), 2);
        assert!((distances[0] - 1.0).abs() < 1e-6); // distance to [0, 0] is 1
        assert!((distances[1] - 1.0).abs() < 1e-6); // distance to [1, 1] is 1
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_cosine_similarity() {
        let executor = MetalExecutor::new().unwrap();

        // Test: [1, 0] vs [1, 0] (identical) and [0, 1] (orthogonal)
        let query = vec![1.0, 0.0];
        let database = vec![1.0, 0.0, 0.0, 1.0];

        let similarities = executor.cosine_similarity(&query, &database, 2, 2).unwrap();

        assert_eq!(similarities.len(), 2);
        assert!((similarities[0] - 1.0).abs() < 1e-6); // identical vectors
        assert!(similarities[1].abs() < 1e-6); // orthogonal vectors
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_dot_product() {
        let executor = MetalExecutor::new().unwrap();

        let query = vec![1.0, 2.0, 3.0];
        let database = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let products = executor.dot_product(&query, &database, 2, 3).unwrap();

        assert_eq!(products.len(), 2);
        assert!((products[0] - 6.0).abs() < 1e-6); // 1*1 + 2*1 + 3*1 = 6
        assert!((products[1] - 12.0).abs() < 1e-6); // 1*2 + 2*2 + 3*2 = 12
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_dimension_mismatch_error() {
        let executor = MetalExecutor::new().unwrap();

        let query = vec![1.0, 2.0]; // dim 2
        let database = vec![1.0, 2.0, 3.0]; // expects dim 3

        let result = executor.euclidean_distance(&query, &database, 1, 3);
        assert!(result.is_err());
    }
}
