//! GPU Acceleration for Vector Operations
//!
//! ⚠️ **EXPERIMENTAL - SKELETON IMPLEMENTATION**
//!
//! This module provides the interface and architecture for GPU-accelerated vector
//! operations, but **GPU backends are not fully implemented**. Current status:
//!
//! - **CPU Backend**: ✅ Fully implemented with SIMD optimizations
//! - **CUDA Backend**: ⚠️ Stub only - falls back to CPU (requires cuDNN/CUDA SDK integration)
//! - **Metal Backend**: ⚠️ Stub only - falls back to CPU (requires Metal shader compilation)
//! - **WebGPU Backend**: ⚠️ Stub only - falls back to CPU (requires wgpu crate integration)
//!
//! **To enable actual GPU acceleration:**
//! 1. Implement CUDA kernel compilation and execution in `cuda_kernels.rs`
//! 2. Implement Metal compute shader pipeline in `metal_executor.rs`
//! 3. Add proper memory management, async transfers, and error handling
//!
//! The CPU backend is the only path considered stable today and provides SIMD-optimized operations.
//! GPU backends serve as architectural templates for future implementation.
//!
//! ## Overview
//!
//! This module provides GPU-accelerated vector operations using CUDA (NVIDIA)
//! and Metal (Apple Silicon). Falls back to optimized CPU implementations when
//! GPU is unavailable.
//!
//! ## Supported Operations
//!
//! - **Batch distance calculations**: Compute distances for 1000s of vectors in parallel
//! - **Matrix multiplication**: For embedding generation and transformations
//! - **K-NN search**: GPU-accelerated nearest neighbor search
//! - **Vector normalization**: Batch L2 normalization
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐
//! │  VecStore API   │
//! └────────┬────────┘
//!          │
//!   ┌──────┴──────┐
//!   │ GPU Executor│
//!   └──────┬──────┘
//!          │
//!    ┌─────┴─────┐
//!    │           │
//! ┌──▼──┐    ┌──▼───┐
//! │CUDA │    │Metal │
//! │(NVIDIA)  │(Apple)│
//! └─────┘    └──────┘
//! ```
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::gpu::{GpuExecutor, GpuBackend, GpuConfig};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Auto-detect GPU
//! let config = GpuConfig::default();
//! let executor = GpuExecutor::new(config)?;
//!
//! // Batch distance calculation
//! let query = vec![0.1, 0.2, 0.3, 0.4];
//! let database = vec![
//!     vec![0.2, 0.3, 0.4, 0.5],
//!     vec![0.3, 0.4, 0.5, 0.6],
//!     // ... thousands more
//! ];
//!
//! let distances = executor.batch_euclidean_distance(&query, &database)?;
//!
//! println!("Computed {} distances on GPU", distances.len());
//! # Ok(())
//! # }
//! ```

pub mod cuda_kernels;
pub mod metal_executor;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    /// NVIDIA CUDA
    Cuda,

    /// Apple Metal
    Metal,

    /// WebGPU (browser-based GPU acceleration)
    WebGpu,

    /// CPU fallback (SIMD optimized)
    Cpu,
}

/// GPU configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Preferred backend
    pub backend: Option<GpuBackend>,

    /// GPU device ID (for multi-GPU systems)
    pub device_id: usize,

    /// Batch size for operations
    pub batch_size: usize,

    /// Maximum GPU memory usage (bytes)
    pub max_memory_bytes: usize,

    /// Enable memory pooling
    pub enable_memory_pool: bool,

    /// Enable async operations
    pub async_execution: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: None, // Auto-detect
            device_id: 0,
            batch_size: 10000,
            max_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            enable_memory_pool: true,
            async_execution: true,
        }
    }
}

impl GpuConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_backend(mut self, backend: GpuBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn with_device_id(mut self, id: usize) -> Self {
        self.device_id = id;
        self
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn with_max_memory_bytes(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = bytes;
        self
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub backend: GpuBackend,
    pub device_id: usize,
    pub name: String,
    pub total_memory_bytes: usize,
    pub available_memory_bytes: usize,
    pub compute_capability: (u32, u32), // (major, minor)
    pub max_threads_per_block: usize,
    pub num_streaming_multiprocessors: usize,
}

/// GPU executor trait
pub trait GpuOps: Send + Sync {
    /// Get device info
    fn device_info(&self) -> GpuDeviceInfo;

    /// Batch Euclidean distance
    fn batch_euclidean_distance(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>>;

    /// Batch cosine similarity
    fn batch_cosine_similarity(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>>;

    /// Batch dot product
    fn batch_dot_product(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>>;

    /// Matrix multiplication
    fn matrix_multiply(&self, a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>>;

    /// Batch L2 normalization
    fn batch_normalize(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>>;

    /// K-NN search (returns indices and distances)
    fn knn_search(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)>;
}

/// Main GPU executor
pub struct GpuExecutor {
    backend: Arc<dyn GpuOps>,
    config: GpuConfig,
}

impl GpuExecutor {
    /// Create new GPU executor with auto-detection
    pub fn new(config: GpuConfig) -> Result<Self> {
        let backend = Self::create_backend(&config)?;

        Ok(Self { backend, config })
    }

    /// Auto-detect and create appropriate backend
    fn create_backend(config: &GpuConfig) -> Result<Arc<dyn GpuOps>> {
        // Try user-specified backend first
        if let Some(backend_type) = config.backend {
            match backend_type {
                GpuBackend::Cuda => {
                    #[cfg(feature = "cuda")]
                    {
                        return Ok(Arc::new(CudaBackend::new(config)?));
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        return Err(anyhow!("CUDA support not compiled. Enable 'cuda' feature."));
                    }
                }
                GpuBackend::Metal => {
                    #[cfg(feature = "metal")]
                    {
                        return Ok(Arc::new(MetalBackend::new(config)?));
                    }
                    #[cfg(not(feature = "metal"))]
                    {
                        return Err(anyhow!(
                            "Metal support not compiled. Enable 'metal' feature."
                        ));
                    }
                }
                GpuBackend::Cpu => {
                    return Ok(Arc::new(CpuBackend::new(config)));
                }
                GpuBackend::WebGpu => {
                    #[cfg(feature = "wasm")]
                    {
                        return Ok(Arc::new(WebGpuBackend::new(config)));
                    }
                    #[cfg(not(feature = "wasm"))]
                    {
                        return Err(anyhow!("WebGPU only available in WASM builds"));
                    }
                }
            }
        }

        // Auto-detect available GPU
        #[cfg(feature = "cuda")]
        {
            if CudaBackend::is_available() {
                return Ok(Arc::new(CudaBackend::new(config)?));
            }
        }

        #[cfg(feature = "metal")]
        {
            if MetalBackend::is_available() {
                return Ok(Arc::new(MetalBackend::new(config)?));
            }
        }

        // Fallback to CPU
        Ok(Arc::new(CpuBackend::new(config)))
    }

    /// Get active backend type
    pub fn backend_type(&self) -> GpuBackend {
        self.backend.device_info().backend
    }

    /// Get device information
    pub fn device_info(&self) -> GpuDeviceInfo {
        self.backend.device_info()
    }

    /// Batch Euclidean distance
    pub fn batch_euclidean_distance(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        self.backend.batch_euclidean_distance(query, database)
    }

    /// Batch cosine similarity
    pub fn batch_cosine_similarity(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        self.backend.batch_cosine_similarity(query, database)
    }

    /// Batch dot product
    pub fn batch_dot_product(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        self.backend.batch_dot_product(query, database)
    }

    /// Matrix multiplication
    pub fn matrix_multiply(&self, a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        self.backend.matrix_multiply(a, b)
    }

    /// Batch normalize vectors
    pub fn batch_normalize(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        self.backend.batch_normalize(vectors)
    }

    /// K-NN search
    pub fn knn_search(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        self.backend.knn_search(query, database, k)
    }
}

// ============================================================================
// CPU Backend (Always Available)
// ============================================================================

/// CPU backend using SIMD optimizations
pub struct CpuBackend {
    config: GpuConfig,
}

impl CpuBackend {
    pub fn new(config: &GpuConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

impl GpuOps for CpuBackend {
    fn device_info(&self) -> GpuDeviceInfo {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        GpuDeviceInfo {
            backend: GpuBackend::Cpu,
            device_id: 0,
            name: "CPU (SIMD Optimized)".to_string(),
            total_memory_bytes: 0, // Not applicable
            available_memory_bytes: 0,
            compute_capability: (0, 0),
            max_threads_per_block: num_cpus,
            num_streaming_multiprocessors: num_cpus,
        }
    }

    fn batch_euclidean_distance(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        use crate::simd::euclidean_distance_simd;

        let distances: Vec<f32> = database
            .iter()
            .map(|vec| euclidean_distance_simd(query, vec))
            .collect();

        Ok(distances)
    }

    fn batch_cosine_similarity(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        use crate::simd::cosine_similarity_simd;

        let similarities: Vec<f32> = database
            .iter()
            .map(|vec| cosine_similarity_simd(query, vec))
            .collect();

        Ok(similarities)
    }

    fn batch_dot_product(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        use crate::simd::dot_product_simd;

        let products: Vec<f32> = database
            .iter()
            .map(|vec| dot_product_simd(query, vec))
            .collect();

        Ok(products)
    }

    fn matrix_multiply(&self, a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if a.is_empty() || b.is_empty() {
            return Ok(vec![]);
        }

        let m = a.len();
        let n = a[0].len();
        let p = b[0].len();

        if b.len() != n {
            return Err(anyhow!("Matrix dimensions mismatch"));
        }

        let mut result = vec![vec![0.0; p]; m];

        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }

        Ok(result)
    }

    fn batch_normalize(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        use crate::simd::magnitude_simd;

        let normalized: Vec<Vec<f32>> = vectors
            .iter()
            .map(|vec| {
                let mag = magnitude_simd(vec);
                if mag > 0.0 {
                    vec.iter().map(|&v| v / mag).collect()
                } else {
                    vec.clone()
                }
            })
            .collect();

        Ok(normalized)
    }

    fn knn_search(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        use crate::simd::euclidean_distance_simd;

        let mut distances: Vec<(usize, f32)> = database
            .iter()
            .enumerate()
            .map(|(idx, vec)| (idx, euclidean_distance_simd(query, vec)))
            .collect();

        // Partial sort to get top k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        let indices: Vec<usize> = distances.iter().map(|(idx, _)| *idx).collect();
        let dists: Vec<f32> = distances.iter().map(|(_, dist)| *dist).collect();

        Ok((indices, dists))
    }
}

// ============================================================================
// CUDA Backend (NVIDIA GPUs)
// ============================================================================

#[cfg(feature = "cuda")]
pub struct CudaBackend {
    config: GpuConfig,
    // Would contain CUDA context, streams, etc.
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    pub fn new(config: &GpuConfig) -> Result<Self> {
        // In real implementation:
        // - Initialize CUDA context
        // - Query device properties
        // - Allocate memory pools
        // - Create streams

        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn is_available() -> bool {
        // In real implementation: Check for CUDA driver/runtime
        false
    }
}

#[cfg(feature = "cuda")]
impl GpuOps for CudaBackend {
    fn device_info(&self) -> GpuDeviceInfo {
        // In real implementation: Query via cuDeviceGetProperties
        GpuDeviceInfo {
            backend: GpuBackend::Cuda,
            device_id: self.config.device_id,
            name: "NVIDIA GPU (CUDA)".to_string(),
            total_memory_bytes: 8 * 1024 * 1024 * 1024, // Example: 8GB
            available_memory_bytes: 6 * 1024 * 1024 * 1024,
            compute_capability: (8, 0), // Example: Ampere
            max_threads_per_block: 1024,
            num_streaming_multiprocessors: 108,
        }
    }

    fn batch_euclidean_distance(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        // In real implementation:
        // 1. Allocate GPU memory
        // 2. Copy query and database to GPU
        // 3. Launch CUDA kernel
        // 4. Copy results back to CPU

        // Fallback for now
        let cpu = CpuBackend::new(&self.config);
        cpu.batch_euclidean_distance(query, database)
    }

    fn batch_cosine_similarity(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        let cpu = CpuBackend::new(&self.config);
        cpu.batch_cosine_similarity(query, database)
    }

    fn batch_dot_product(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        let cpu = CpuBackend::new(&self.config);
        cpu.batch_dot_product(query, database)
    }

    fn matrix_multiply(&self, a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        // In real implementation: Use cuBLAS (sgemm)
        let cpu = CpuBackend::new(&self.config);
        cpu.matrix_multiply(a, b)
    }

    fn batch_normalize(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let cpu = CpuBackend::new(&self.config);
        cpu.batch_normalize(vectors)
    }

    fn knn_search(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        // In real implementation: GPU-accelerated k-NN
        let cpu = CpuBackend::new(&self.config);
        cpu.knn_search(query, database, k)
    }
}

// ============================================================================
// Metal Backend (Apple Silicon)
// ============================================================================

#[cfg(feature = "metal")]
pub struct MetalBackend {
    config: GpuConfig,
    // Would contain Metal device, command queue, etc.
}

#[cfg(feature = "metal")]
impl MetalBackend {
    pub fn new(config: &GpuConfig) -> Result<Self> {
        // In real implementation:
        // - Get Metal device
        // - Create command queue
        // - Compile shaders
        // - Allocate buffers

        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn is_available() -> bool {
        // In real implementation: Check for Metal support
        #[cfg(target_os = "macos")]
        {
            true // Assume available on macOS
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

#[cfg(feature = "metal")]
impl GpuOps for MetalBackend {
    fn device_info(&self) -> GpuDeviceInfo {
        GpuDeviceInfo {
            backend: GpuBackend::Metal,
            device_id: self.config.device_id,
            name: "Apple Silicon GPU (Metal)".to_string(),
            total_memory_bytes: 16 * 1024 * 1024 * 1024, // Unified memory
            available_memory_bytes: 12 * 1024 * 1024 * 1024,
            compute_capability: (3, 0), // Metal version
            max_threads_per_block: 1024,
            num_streaming_multiprocessors: 10,
        }
    }

    fn batch_euclidean_distance(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        // In real implementation:
        // 1. Create Metal buffers
        // 2. Copy data to GPU
        // 3. Dispatch compute shader
        // 4. Read results

        let cpu = CpuBackend::new(&self.config);
        cpu.batch_euclidean_distance(query, database)
    }

    fn batch_cosine_similarity(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        let cpu = CpuBackend::new(&self.config);
        cpu.batch_cosine_similarity(query, database)
    }

    fn batch_dot_product(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        let cpu = CpuBackend::new(&self.config);
        cpu.batch_dot_product(query, database)
    }

    fn matrix_multiply(&self, a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        // In real implementation: Use Metal Performance Shaders (MPS)
        let cpu = CpuBackend::new(&self.config);
        cpu.matrix_multiply(a, b)
    }

    fn batch_normalize(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let cpu = CpuBackend::new(&self.config);
        cpu.batch_normalize(vectors)
    }

    fn knn_search(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let cpu = CpuBackend::new(&self.config);
        cpu.knn_search(query, database, k)
    }
}

// ============================================================================
// WEBGPU BACKEND (Browser GPU Acceleration)
// ============================================================================

/// WebGPU backend for browser-based GPU acceleration
///
/// This backend uses WebGPU compute shaders to accelerate vector operations
/// in WebAssembly builds. WebGPU provides access to GPU compute in modern browsers.
#[cfg(feature = "wasm")]
pub struct WebGpuBackend {
    config: GpuConfig,
}

#[cfg(feature = "wasm")]
impl WebGpuBackend {
    pub fn new(config: &GpuConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn is_available() -> bool {
        // In a real implementation, check if navigator.gpu exists
        // For now, return false as WebGPU needs additional setup
        false
    }
}

#[cfg(feature = "wasm")]
impl GpuOps for WebGpuBackend {
    fn device_info(&self) -> GpuDeviceInfo {
        GpuDeviceInfo {
            backend: GpuBackend::WebGpu,
            device_id: 0,
            name: "WebGPU Device".to_string(),
            total_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB estimate
            available_memory_bytes: 2 * 1024 * 1024 * 1024,
            compute_capability: (1, 0),
            max_threads_per_block: 256,
            num_streaming_multiprocessors: 1,
        }
    }

    fn batch_euclidean_distance(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        // WebGPU implementation would:
        // 1. Get GPU adapter and device
        // 2. Create buffers for query and database
        // 3. Create compute shader with distance calculation
        // 4. Dispatch compute pipeline
        // 5. Read results back
        //
        // For now, fall back to SIMD-optimized CPU
        let cpu = CpuBackend::new(&self.config);
        cpu.batch_euclidean_distance(query, database)
    }

    fn batch_cosine_similarity(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        let cpu = CpuBackend::new(&self.config);
        cpu.batch_cosine_similarity(query, database)
    }

    fn batch_dot_product(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        let cpu = CpuBackend::new(&self.config);
        cpu.batch_dot_product(query, database)
    }

    fn matrix_multiply(&self, a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let cpu = CpuBackend::new(&self.config);
        cpu.matrix_multiply(a, b)
    }

    fn batch_normalize(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let cpu = CpuBackend::new(&self.config);
        cpu.batch_normalize(vectors)
    }

    fn knn_search(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let cpu = CpuBackend::new(&self.config);
        cpu.knn_search(query, database, k)
    }
}

/// GPU performance benchmarking
pub struct GpuBenchmark {
    pub backend: GpuBackend,
    pub operation: String,
    pub num_vectors: usize,
    pub dimension: usize,
    pub duration_ms: f64,
    pub throughput_vectors_per_sec: f64,
}

impl GpuBenchmark {
    pub fn run(executor: &GpuExecutor, num_vectors: usize, dimension: usize) -> Result<Vec<Self>> {
        use std::time::Instant;

        let mut benchmarks = Vec::new();
        let backend = executor.backend_type();

        // Generate test data
        let query: Vec<f32> = (0..dimension).map(|i| i as f32 * 0.01).collect();
        let database: Vec<Vec<f32>> = (0..num_vectors)
            .map(|i| (0..dimension).map(|j| (i + j) as f32 * 0.01).collect())
            .collect();

        // Benchmark Euclidean distance
        let start = Instant::now();
        let _ = executor.batch_euclidean_distance(&query, &database)?;
        let duration = start.elapsed();

        benchmarks.push(GpuBenchmark {
            backend,
            operation: "Euclidean Distance".to_string(),
            num_vectors,
            dimension,
            duration_ms: duration.as_secs_f64() * 1000.0,
            throughput_vectors_per_sec: num_vectors as f64 / duration.as_secs_f64(),
        });

        // Benchmark cosine similarity
        let start = Instant::now();
        let _ = executor.batch_cosine_similarity(&query, &database)?;
        let duration = start.elapsed();

        benchmarks.push(GpuBenchmark {
            backend,
            operation: "Cosine Similarity".to_string(),
            num_vectors,
            dimension,
            duration_ms: duration.as_secs_f64() * 1000.0,
            throughput_vectors_per_sec: num_vectors as f64 / duration.as_secs_f64(),
        });

        // Benchmark k-NN
        let start = Instant::now();
        let _ = executor.knn_search(&query, &database, 10)?;
        let duration = start.elapsed();

        benchmarks.push(GpuBenchmark {
            backend,
            operation: "K-NN Search (k=10)".to_string(),
            num_vectors,
            dimension,
            duration_ms: duration.as_secs_f64() * 1000.0,
            throughput_vectors_per_sec: num_vectors as f64 / duration.as_secs_f64(),
        });

        Ok(benchmarks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
        let config = GpuConfig::default().with_backend(GpuBackend::Cpu);
        let executor = GpuExecutor::new(config).unwrap();

        assert_eq!(executor.backend_type(), GpuBackend::Cpu);

        let info = executor.device_info();
        assert_eq!(info.backend, GpuBackend::Cpu);
    }

    #[test]
    fn test_batch_euclidean_distance() {
        let config = GpuConfig::default().with_backend(GpuBackend::Cpu);
        let executor = GpuExecutor::new(config).unwrap();

        let query = vec![1.0, 2.0, 3.0];
        let database = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![0.0, 0.0, 0.0],
        ];

        let distances = executor
            .batch_euclidean_distance(&query, &database)
            .unwrap();

        assert_eq!(distances.len(), 3);
        assert!(distances[0] < 0.01); // Should be ~0 (same vector)
        assert!(distances[1] > 1.0); // Should be sqrt(3)
    }

    #[test]
    fn test_batch_cosine_similarity() {
        let config = GpuConfig::default().with_backend(GpuBackend::Cpu);
        let executor = GpuExecutor::new(config).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let database = vec![
            vec![1.0, 0.0, 0.0],  // Same direction
            vec![0.0, 1.0, 0.0],  // Perpendicular
            vec![-1.0, 0.0, 0.0], // Opposite
        ];

        let similarities = executor.batch_cosine_similarity(&query, &database).unwrap();

        assert_eq!(similarities.len(), 3);
        assert!((similarities[0] - 1.0).abs() < 0.01); // Should be 1.0
        assert!(similarities[1].abs() < 0.01); // Should be 0.0
        assert!((similarities[2] + 1.0).abs() < 0.01); // Should be -1.0
    }

    #[test]
    fn test_knn_search() {
        let config = GpuConfig::default().with_backend(GpuBackend::Cpu);
        let executor = GpuExecutor::new(config).unwrap();

        let query = vec![0.5, 0.5];
        let database = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5], // Exact match
            vec![10.0, 10.0],
        ];

        let (indices, distances) = executor.knn_search(&query, &database, 2).unwrap();

        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
        assert_eq!(indices[0], 2); // Exact match should be first
        assert!(distances[0] < 0.01);
    }

    #[test]
    fn test_matrix_multiply() {
        let config = GpuConfig::default().with_backend(GpuBackend::Cpu);
        let executor = GpuExecutor::new(config).unwrap();

        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let result = executor.matrix_multiply(&a, &b).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
        assert!((result[0][0] - 19.0).abs() < 0.01); // 1*5 + 2*7
        assert!((result[0][1] - 22.0).abs() < 0.01); // 1*6 + 2*8
    }

    #[test]
    fn test_batch_normalize() {
        let config = GpuConfig::default().with_backend(GpuBackend::Cpu);
        let executor = GpuExecutor::new(config).unwrap();

        let vectors = vec![
            vec![3.0, 4.0], // Magnitude 5
            vec![1.0, 0.0], // Already normalized
        ];

        let normalized = executor.batch_normalize(&vectors).unwrap();

        assert_eq!(normalized.len(), 2);
        assert!((normalized[0][0] - 0.6).abs() < 0.01);
        assert!((normalized[0][1] - 0.8).abs() < 0.01);
        assert!((normalized[1][0] - 1.0).abs() < 0.01);
    }
}
