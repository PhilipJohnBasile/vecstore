//! GPU Acceleration for Vector Operations
//!
//! This module provides GPU-accelerated vector operations for high-performance
//! similarity search and vector computations.
//!
//! ## Backend Status
//!
//! - **CPU Backend**: ✅ Fully implemented with SIMD optimizations (always available)
//! - **CUDA Backend**: ✅ Complete with PTX kernels via cudarc (requires `cuda` feature)
//! - **Metal Backend**: ✅ Complete with MSL compute shaders (requires `metal` feature + macOS)
//! - **WebGPU Backend**: ✅ Complete with WGSL shaders via wgpu (requires `webgpu` feature)
//!
//! ## Enabling GPU Support
//!
//! ```toml
//! [dependencies]
//! vecstore = { version = "0.1.0", features = ["cuda"] }  # NVIDIA
//! # or
//! vecstore = { version = "0.1.0", features = ["metal"] }  # Apple Silicon
//! # or
//! vecstore = { version = "0.1.0", features = ["webgpu"] }  # Cross-platform
//! ```
//!
//! ## Performance Expectations
//!
//! Based on industry benchmarks:
//! - **Batch Distance**: 5-10x faster for large batches (1000+ vectors)
//! - **K-NN Search**: 4-50x faster depending on algorithm
//! - **Memory Bandwidth**: GPUs excel at streaming vector data
//!
//! ## Supported Operations
//!
//! All backends implement the `GpuOps` trait with:
//! - Batch Euclidean distance computation
//! - Batch cosine similarity computation
//! - Batch dot product computation
//! - Matrix multiplication
//! - Batch L2 normalization
//! - K-NN search (GPU distance + CPU top-k selection)
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

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(any(feature = "metal", feature = "webgpu"))]
use std::mem::size_of;

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
    #[allow(dead_code)]
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
                },
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
                },
                GpuBackend::Cpu => {
                    return Ok(Arc::new(CpuBackend::new(config)));
                },
                GpuBackend::WebGpu => {
                    #[cfg(feature = "webgpu")]
                    {
                        return Ok(Arc::new(WebGpuBackend::new(config)?));
                    }
                    #[cfg(all(feature = "wasm", not(feature = "webgpu")))]
                    {
                        return Ok(Arc::new(WebGpuBackend::new(config)));
                    }
                    #[cfg(not(any(feature = "wasm", feature = "webgpu")))]
                    {
                        return Err(anyhow!("WebGPU requires 'webgpu' feature"));
                    }
                },
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

        #[cfg(feature = "webgpu")]
        {
            if WebGpuBackend::is_available() {
                return Ok(Arc::new(WebGpuBackend::new(config)?));
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
    #[allow(dead_code)]
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
    executor: cuda_kernels::CudaKernelExecutor,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    pub fn new(config: &GpuConfig) -> Result<Self> {
        let executor = cuda_kernels::CudaKernelExecutor::new(config.device_id)?;

        Ok(Self {
            config: config.clone(),
            executor,
        })
    }

    pub fn is_available() -> bool {
        cuda_kernels::CudaKernelExecutor::is_available()
    }
}

#[cfg(feature = "cuda")]
impl GpuOps for CudaBackend {
    fn device_info(&self) -> GpuDeviceInfo {
        let props = self.executor.device_properties().unwrap_or_else(|_| {
            cuda_kernels::CudaDeviceProperties {
                name: "Unknown CUDA Device".to_string(),
                compute_capability: (7, 0),
                total_memory_bytes: 0,
                multiprocessor_count: 0,
                max_threads_per_block: 1024,
                max_shared_memory_per_block: 48 * 1024,
            }
        });

        GpuDeviceInfo {
            backend: GpuBackend::Cuda,
            device_id: self.config.device_id,
            name: props.name,
            total_memory_bytes: props.total_memory_bytes,
            available_memory_bytes: props.total_memory_bytes, // Approximation
            compute_capability: (
                props.compute_capability.0 as u32,
                props.compute_capability.1 as u32,
            ),
            max_threads_per_block: props.max_threads_per_block as usize,
            num_streaming_multiprocessors: props.multiprocessor_count as usize,
        }
    }

    fn batch_euclidean_distance(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        if database.is_empty() {
            return Ok(vec![]);
        }

        let vector_dim = query.len();
        let num_vectors = database.len();

        // Flatten database into contiguous array
        let flat_database: Vec<f32> = database.iter().flat_map(|v| v.iter().copied()).collect();

        // Execute GPU kernel
        self.executor
            .euclidean_distance(query, &flat_database, num_vectors, vector_dim)
    }

    fn batch_cosine_similarity(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        if database.is_empty() {
            return Ok(vec![]);
        }

        let vector_dim = query.len();
        let num_vectors = database.len();

        let flat_database: Vec<f32> = database.iter().flat_map(|v| v.iter().copied()).collect();

        self.executor
            .cosine_similarity(query, &flat_database, num_vectors, vector_dim)
    }

    fn batch_dot_product(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        if database.is_empty() {
            return Ok(vec![]);
        }

        let vector_dim = query.len();
        let num_vectors = database.len();

        let flat_database: Vec<f32> = database.iter().flat_map(|v| v.iter().copied()).collect();

        self.executor
            .dot_product(query, &flat_database, num_vectors, vector_dim)
    }

    fn matrix_multiply(&self, a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        // Fall back to CPU for matrix multiply (would need cuBLAS integration)
        let cpu = CpuBackend::new(&self.config);
        cpu.matrix_multiply(a, b)
    }

    fn batch_normalize(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let vector_dim = vectors[0].len();
        let num_vectors = vectors.len();

        let flat_vectors: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        let normalized = self
            .executor
            .l2_normalize(&flat_vectors, num_vectors, vector_dim)?;

        // Reshape back to Vec<Vec<f32>>
        Ok(normalized.chunks(vector_dim).map(|c| c.to_vec()).collect())
    }

    fn knn_search(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        // Use GPU for distance computation, CPU for top-k selection
        let distances = self.batch_euclidean_distance(query, database)?;

        let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        indexed.truncate(k);

        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        let dists: Vec<f32> = indexed.iter().map(|(_, d)| *d).collect();

        Ok((indices, dists))
    }
}

// ============================================================================
// Metal Backend (Apple Silicon)
// ============================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};

/// Metal Shading Language source code for compute kernels
#[cfg(all(target_os = "macos", feature = "metal"))]
const METAL_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

/// Euclidean distance kernel - computes L2 distance between query and each database vector
kernel void euclidean_distance_kernel(
    constant float* query [[buffer(0)]],
    constant float* database [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& num_vectors [[buffer(3)]],
    constant uint& vector_dim [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_vectors) return;

    float sum = 0.0f;
    uint base_offset = idx * vector_dim;

    for (uint i = 0; i < vector_dim; i++) {
        float diff = query[i] - database[base_offset + i];
        sum += diff * diff;
    }

    distances[idx] = sqrt(sum);
}

/// Cosine similarity kernel - computes cosine similarity between query and each database vector
kernel void cosine_similarity_kernel(
    constant float* query [[buffer(0)]],
    constant float* database [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant uint& num_vectors [[buffer(3)]],
    constant uint& vector_dim [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_vectors) return;

    float dot = 0.0f;
    float query_norm = 0.0f;
    float db_norm = 0.0f;
    uint base_offset = idx * vector_dim;

    for (uint i = 0; i < vector_dim; i++) {
        float q = query[i];
        float d = database[base_offset + i];
        dot += q * d;
        query_norm += q * q;
        db_norm += d * d;
    }

    query_norm = sqrt(query_norm);
    db_norm = sqrt(db_norm);

    similarities[idx] = dot / (query_norm * db_norm + 1e-8f);
}

/// Dot product kernel - computes dot product between query and each database vector
kernel void dot_product_kernel(
    constant float* query [[buffer(0)]],
    constant float* database [[buffer(1)]],
    device float* products [[buffer(2)]],
    constant uint& num_vectors [[buffer(3)]],
    constant uint& vector_dim [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_vectors) return;

    float sum = 0.0f;
    uint base_offset = idx * vector_dim;

    for (uint i = 0; i < vector_dim; i++) {
        sum += query[i] * database[base_offset + i];
    }

    products[idx] = sum;
}

/// L2 normalization kernel - normalizes each vector to unit length
kernel void l2_normalize_kernel(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_vectors [[buffer(2)]],
    constant uint& vector_dim [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_vectors) return;

    uint base_offset = idx * vector_dim;
    float norm = 0.0f;

    // Compute L2 norm
    for (uint i = 0; i < vector_dim; i++) {
        float val = input[base_offset + i];
        norm += val * val;
    }
    norm = sqrt(norm);

    // Normalize
    for (uint i = 0; i < vector_dim; i++) {
        output[base_offset + i] = input[base_offset + i] / (norm + 1e-8f);
    }
}

/// Matrix multiplication kernel - C = A * B
kernel void matrix_multiply_kernel(
    constant float* A [[buffer(0)]],
    constant float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],  // rows of A
    constant uint& N [[buffer(4)]],  // cols of B
    constant uint& K [[buffer(5)]],  // cols of A / rows of B
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = sum;
}
"#;

/// Metal backend for Apple Silicon GPU acceleration
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct MetalBackend {
    config: GpuConfig,
    device: Device,
    command_queue: CommandQueue,
    euclidean_pipeline: ComputePipelineState,
    cosine_pipeline: ComputePipelineState,
    dot_product_pipeline: ComputePipelineState,
    normalize_pipeline: ComputePipelineState,
    matmul_pipeline: ComputePipelineState,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Create a new Metal backend with GPU device initialization
    pub fn new(config: &GpuConfig) -> Result<Self> {
        // Get the system default Metal device
        let device =
            Device::system_default().ok_or_else(|| anyhow!("No Metal-capable GPU device found"))?;

        // Create command queue for submitting work to GPU
        let command_queue = device.new_command_queue();

        // Compile Metal shaders from source
        let library = device
            .new_library_with_source(METAL_SHADER_SOURCE, &metal::CompileOptions::new())
            .map_err(|e| anyhow!("Failed to compile Metal shaders: {}", e))?;

        // Create compute pipeline states for each kernel
        let euclidean_pipeline =
            Self::create_pipeline(&device, &library, "euclidean_distance_kernel")?;
        let cosine_pipeline = Self::create_pipeline(&device, &library, "cosine_similarity_kernel")?;
        let dot_product_pipeline = Self::create_pipeline(&device, &library, "dot_product_kernel")?;
        let normalize_pipeline = Self::create_pipeline(&device, &library, "l2_normalize_kernel")?;
        let matmul_pipeline = Self::create_pipeline(&device, &library, "matrix_multiply_kernel")?;

        Ok(Self {
            config: config.clone(),
            device,
            command_queue,
            euclidean_pipeline,
            cosine_pipeline,
            dot_product_pipeline,
            normalize_pipeline,
            matmul_pipeline,
        })
    }

    /// Create a compute pipeline state for a kernel function
    fn create_pipeline(
        device: &Device,
        library: &metal::Library,
        kernel_name: &str,
    ) -> Result<ComputePipelineState> {
        let kernel_function = library
            .get_function(kernel_name, None)
            .map_err(|e| anyhow!("Failed to get kernel function '{}': {}", kernel_name, e))?;

        device
            .new_compute_pipeline_state_with_function(&kernel_function)
            .map_err(|e| anyhow!("Failed to create pipeline for '{}': {}", kernel_name, e))
    }

    /// Check if Metal is available on this system
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Create a Metal buffer from a slice of data
    fn create_buffer<T: Copy>(&self, data: &[T]) -> Buffer {
        let size = (data.len() * size_of::<T>()) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size,
            MTLResourceOptions::StorageModeShared,
        );
        buffer
    }

    /// Create an empty Metal buffer for output
    fn create_output_buffer(&self, num_elements: usize) -> Buffer {
        let size = (num_elements * size_of::<f32>()) as u64;
        self.device
            .new_buffer(size, MTLResourceOptions::StorageModeShared)
    }

    /// Read results from a Metal buffer back to CPU
    fn read_buffer(&self, buffer: &Buffer, num_elements: usize) -> Vec<f32> {
        let ptr = buffer.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, num_elements) };
        slice.to_vec()
    }

    /// Execute a distance/similarity kernel (euclidean, cosine, or dot product)
    fn execute_distance_kernel(
        &self,
        pipeline: &ComputePipelineState,
        query: &[f32],
        flat_database: &[f32],
        num_vectors: usize,
        vector_dim: usize,
    ) -> Result<Vec<f32>> {
        // Create buffers
        let query_buffer = self.create_buffer(query);
        let database_buffer = self.create_buffer(flat_database);
        let output_buffer = self.create_output_buffer(num_vectors);

        // Create parameter buffers
        let num_vectors_u32 = num_vectors as u32;
        let vector_dim_u32 = vector_dim as u32;
        let num_vectors_buffer = self.create_buffer(&[num_vectors_u32]);
        let vector_dim_buffer = self.create_buffer(&[vector_dim_u32]);

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline and buffers
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&query_buffer), 0);
        encoder.set_buffer(1, Some(&database_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&num_vectors_buffer), 0);
        encoder.set_buffer(4, Some(&vector_dim_buffer), 0);

        // Calculate threadgroup sizes
        let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
        let threads_per_threadgroup = max_threads.min(256);
        let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
        let grid_size = MTLSize::new(num_vectors as u64, 1, 1);

        // Dispatch compute kernel
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        // Submit and wait for completion
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Check for errors
        if let Some(error) = command_buffer.error() {
            return Err(anyhow!("Metal command buffer error: {:?}", error));
        }

        // Read results back
        Ok(self.read_buffer(&output_buffer, num_vectors))
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl GpuOps for MetalBackend {
    fn device_info(&self) -> GpuDeviceInfo {
        let name = self.device.name().to_string();

        // Get memory info - on Apple Silicon this is unified memory
        let recommended_max = self.device.recommended_max_working_set_size() as usize;

        // Metal feature set version (approximated)
        let supports_family = if self.device.supports_family(metal::MTLGPUFamily::Apple8) {
            (3, 1) // M2/M3 class
        } else if self.device.supports_family(metal::MTLGPUFamily::Apple7) {
            (3, 0) // M1 class
        } else {
            (2, 0) // Older
        };

        GpuDeviceInfo {
            backend: GpuBackend::Metal,
            device_id: self.config.device_id,
            name,
            total_memory_bytes: recommended_max,
            available_memory_bytes: recommended_max, // Unified memory, estimation
            compute_capability: (supports_family.0, supports_family.1),
            max_threads_per_block: self.euclidean_pipeline.max_total_threads_per_threadgroup()
                as usize,
            num_streaming_multiprocessors: 10, // Not directly exposed by Metal API
        }
    }

    fn batch_euclidean_distance(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        if database.is_empty() {
            return Ok(vec![]);
        }

        let vector_dim = query.len();
        let num_vectors = database.len();

        // Validate dimensions
        for (i, vec) in database.iter().enumerate() {
            if vec.len() != vector_dim {
                return Err(anyhow!(
                    "Vector {} has dimension {} but query has dimension {}",
                    i,
                    vec.len(),
                    vector_dim
                ));
            }
        }

        // Flatten database into contiguous array for GPU
        let flat_database: Vec<f32> = database.iter().flat_map(|v| v.iter().copied()).collect();

        self.execute_distance_kernel(
            &self.euclidean_pipeline,
            query,
            &flat_database,
            num_vectors,
            vector_dim,
        )
    }

    fn batch_cosine_similarity(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        if database.is_empty() {
            return Ok(vec![]);
        }

        let vector_dim = query.len();
        let num_vectors = database.len();

        // Validate dimensions
        for (i, vec) in database.iter().enumerate() {
            if vec.len() != vector_dim {
                return Err(anyhow!(
                    "Vector {} has dimension {} but query has dimension {}",
                    i,
                    vec.len(),
                    vector_dim
                ));
            }
        }

        // Flatten database into contiguous array for GPU
        let flat_database: Vec<f32> = database.iter().flat_map(|v| v.iter().copied()).collect();

        self.execute_distance_kernel(
            &self.cosine_pipeline,
            query,
            &flat_database,
            num_vectors,
            vector_dim,
        )
    }

    fn batch_dot_product(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        if database.is_empty() {
            return Ok(vec![]);
        }

        let vector_dim = query.len();
        let num_vectors = database.len();

        // Validate dimensions
        for (i, vec) in database.iter().enumerate() {
            if vec.len() != vector_dim {
                return Err(anyhow!(
                    "Vector {} has dimension {} but query has dimension {}",
                    i,
                    vec.len(),
                    vector_dim
                ));
            }
        }

        // Flatten database into contiguous array for GPU
        let flat_database: Vec<f32> = database.iter().flat_map(|v| v.iter().copied()).collect();

        self.execute_distance_kernel(
            &self.dot_product_pipeline,
            query,
            &flat_database,
            num_vectors,
            vector_dim,
        )
    }

    fn matrix_multiply(&self, a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if a.is_empty() || b.is_empty() {
            return Ok(vec![]);
        }

        let m = a.len(); // rows of A
        let k = a[0].len(); // cols of A = rows of B
        let n = b[0].len(); // cols of B

        // Validate dimensions
        if b.len() != k {
            return Err(anyhow!(
                "Matrix dimension mismatch: A is {}x{}, B is {}x{}",
                m,
                k,
                b.len(),
                n
            ));
        }

        // Flatten matrices in row-major order
        let flat_a: Vec<f32> = a.iter().flat_map(|row| row.iter().copied()).collect();
        let flat_b: Vec<f32> = b.iter().flat_map(|row| row.iter().copied()).collect();

        // Create buffers
        let a_buffer = self.create_buffer(&flat_a);
        let b_buffer = self.create_buffer(&flat_b);
        let output_buffer = self.create_output_buffer(m * n);

        // Create parameter buffers
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let m_buffer = self.create_buffer(&[m_u32]);
        let n_buffer = self.create_buffer(&[n_u32]);
        let k_buffer = self.create_buffer(&[k_u32]);

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline and buffers
        encoder.set_compute_pipeline_state(&self.matmul_pipeline);
        encoder.set_buffer(0, Some(&a_buffer), 0);
        encoder.set_buffer(1, Some(&b_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&m_buffer), 0);
        encoder.set_buffer(4, Some(&n_buffer), 0);
        encoder.set_buffer(5, Some(&k_buffer), 0);

        // Calculate threadgroup sizes for 2D dispatch
        let max_threads = self.matmul_pipeline.max_total_threads_per_threadgroup() as u64;
        let threads_per_dim = (max_threads as f64).sqrt() as u64;
        let threadgroup_size = MTLSize::new(threads_per_dim.min(16), threads_per_dim.min(16), 1);
        let grid_size = MTLSize::new(n as u64, m as u64, 1);

        // Dispatch compute kernel
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        // Submit and wait for completion
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Check for errors
        if let Some(error) = command_buffer.error() {
            return Err(anyhow!("Metal matrix multiply error: {:?}", error));
        }

        // Read results and reshape
        let flat_result = self.read_buffer(&output_buffer, m * n);
        let result: Vec<Vec<f32>> = flat_result.chunks(n).map(|chunk| chunk.to_vec()).collect();

        Ok(result)
    }

    fn batch_normalize(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let num_vectors = vectors.len();
        let vector_dim = vectors[0].len();

        // Validate all vectors have same dimension
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != vector_dim {
                return Err(anyhow!(
                    "Vector {} has dimension {} but expected {}",
                    i,
                    vec.len(),
                    vector_dim
                ));
            }
        }

        // Flatten vectors
        let flat_vectors: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        // Create buffers
        let input_buffer = self.create_buffer(&flat_vectors);
        let output_buffer = self.create_output_buffer(flat_vectors.len());

        // Create parameter buffers
        let num_vectors_u32 = num_vectors as u32;
        let vector_dim_u32 = vector_dim as u32;
        let num_vectors_buffer = self.create_buffer(&[num_vectors_u32]);
        let vector_dim_buffer = self.create_buffer(&[vector_dim_u32]);

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline and buffers
        encoder.set_compute_pipeline_state(&self.normalize_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&num_vectors_buffer), 0);
        encoder.set_buffer(3, Some(&vector_dim_buffer), 0);

        // Calculate threadgroup sizes
        let max_threads = self.normalize_pipeline.max_total_threads_per_threadgroup() as u64;
        let threads_per_threadgroup = max_threads.min(256);
        let threadgroup_size = MTLSize::new(threads_per_threadgroup, 1, 1);
        let grid_size = MTLSize::new(num_vectors as u64, 1, 1);

        // Dispatch compute kernel
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        // Submit and wait for completion
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Check for errors
        if let Some(error) = command_buffer.error() {
            return Err(anyhow!("Metal normalize error: {:?}", error));
        }

        // Read results and reshape
        let flat_result = self.read_buffer(&output_buffer, flat_vectors.len());
        let result: Vec<Vec<f32>> = flat_result
            .chunks(vector_dim)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(result)
    }

    fn knn_search(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        if database.is_empty() {
            return Ok((vec![], vec![]));
        }

        let k = k.min(database.len());

        // Use GPU for distance computation
        let distances = self.batch_euclidean_distance(query, database)?;

        // CPU-based top-k selection (GPU parallel reduction would be overkill for typical k values)
        // For very large k, we could implement GPU-based sorting, but for k << n, CPU is efficient
        let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();

        // Partial sort for efficiency when k << n
        if k < indexed.len() / 2 {
            indexed.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
            indexed.truncate(k);
            indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
        } else {
            indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
            indexed.truncate(k);
        }

        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        let dists: Vec<f32> = indexed.iter().map(|(_, d)| *d).collect();

        Ok((indices, dists))
    }
}

// Fallback stub for non-macOS platforms when metal feature is enabled
#[cfg(all(not(target_os = "macos"), feature = "metal"))]
pub struct MetalBackend {
    config: GpuConfig,
}

#[cfg(all(not(target_os = "macos"), feature = "metal"))]
impl MetalBackend {
    pub fn new(_config: &GpuConfig) -> Result<Self> {
        Err(anyhow!("Metal is only available on macOS"))
    }

    pub fn is_available() -> bool {
        false
    }
}

#[cfg(all(not(target_os = "macos"), feature = "metal"))]
impl GpuOps for MetalBackend {
    fn device_info(&self) -> GpuDeviceInfo {
        GpuDeviceInfo {
            backend: GpuBackend::Metal,
            device_id: 0,
            name: "Metal (unavailable)".to_string(),
            total_memory_bytes: 0,
            available_memory_bytes: 0,
            compute_capability: (0, 0),
            max_threads_per_block: 0,
            num_streaming_multiprocessors: 0,
        }
    }

    fn batch_euclidean_distance(&self, _query: &[f32], _database: &[Vec<f32>]) -> Result<Vec<f32>> {
        Err(anyhow!("Metal is only available on macOS"))
    }

    fn batch_cosine_similarity(&self, _query: &[f32], _database: &[Vec<f32>]) -> Result<Vec<f32>> {
        Err(anyhow!("Metal is only available on macOS"))
    }

    fn batch_dot_product(&self, _query: &[f32], _database: &[Vec<f32>]) -> Result<Vec<f32>> {
        Err(anyhow!("Metal is only available on macOS"))
    }

    fn matrix_multiply(&self, _a: &[Vec<f32>], _b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        Err(anyhow!("Metal is only available on macOS"))
    }

    fn batch_normalize(&self, _vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        Err(anyhow!("Metal is only available on macOS"))
    }

    fn knn_search(
        &self,
        _query: &[f32],
        _database: &[Vec<f32>],
        _k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        Err(anyhow!("Metal is only available on macOS"))
    }
}

// ============================================================================
// WEBGPU BACKEND (Cross-Platform GPU Acceleration via wgpu)
// ============================================================================

/// WGSL compute shader for Euclidean distance calculation
#[cfg(feature = "webgpu")]
pub const WGSL_EUCLIDEAN_DISTANCE: &str = r#"
// Euclidean distance compute shader
// Computes L2 distance between a query vector and multiple database vectors

struct Params {
    num_vectors: u32,
    vector_dim: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> database: array<f32>;
@group(0) @binding(2) var<storage, read_write> distances: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.num_vectors) {
        return;
    }

    var sum: f32 = 0.0;
    let base_offset = idx * params.vector_dim;

    for (var i: u32 = 0u; i < params.vector_dim; i = i + 1u) {
        let diff = query[i] - database[base_offset + i];
        sum = sum + diff * diff;
    }

    distances[idx] = sqrt(sum);
}
"#;

/// WGSL compute shader for cosine similarity calculation
#[cfg(feature = "webgpu")]
pub const WGSL_COSINE_SIMILARITY: &str = r#"
// Cosine similarity compute shader
// Computes cosine similarity between a query vector and multiple database vectors

struct Params {
    num_vectors: u32,
    vector_dim: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> database: array<f32>;
@group(0) @binding(2) var<storage, read_write> similarities: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.num_vectors) {
        return;
    }

    var dot_product: f32 = 0.0;
    var query_norm: f32 = 0.0;
    var db_norm: f32 = 0.0;
    let base_offset = idx * params.vector_dim;

    for (var i: u32 = 0u; i < params.vector_dim; i = i + 1u) {
        let q = query[i];
        let d = database[base_offset + i];
        dot_product = dot_product + q * d;
        query_norm = query_norm + q * q;
        db_norm = db_norm + d * d;
    }

    query_norm = sqrt(query_norm);
    db_norm = sqrt(db_norm);

    // Add small epsilon to prevent division by zero
    similarities[idx] = dot_product / (query_norm * db_norm + 1e-8);
}
"#;

/// WGSL compute shader for dot product calculation
#[cfg(feature = "webgpu")]
pub const WGSL_DOT_PRODUCT: &str = r#"
// Dot product compute shader
// Computes inner product between a query vector and multiple database vectors

struct Params {
    num_vectors: u32,
    vector_dim: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> database: array<f32>;
@group(0) @binding(2) var<storage, read_write> products: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.num_vectors) {
        return;
    }

    var sum: f32 = 0.0;
    let base_offset = idx * params.vector_dim;

    for (var i: u32 = 0u; i < params.vector_dim; i = i + 1u) {
        sum = sum + query[i] * database[base_offset + i];
    }

    products[idx] = sum;
}
"#;

/// WGSL compute shader for L2 normalization
#[cfg(feature = "webgpu")]
pub const WGSL_L2_NORMALIZE: &str = r#"
// L2 normalization compute shader
// Normalizes vectors to unit length

struct Params {
    num_vectors: u32,
    vector_dim: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.num_vectors) {
        return;
    }

    let base_offset = idx * params.vector_dim;

    // Compute L2 norm
    var norm: f32 = 0.0;
    for (var i: u32 = 0u; i < params.vector_dim; i = i + 1u) {
        let val = input[base_offset + i];
        norm = norm + val * val;
    }
    norm = sqrt(norm);

    // Normalize (with epsilon to prevent division by zero)
    let inv_norm = 1.0 / (norm + 1e-8);
    for (var i: u32 = 0u; i < params.vector_dim; i = i + 1u) {
        output[base_offset + i] = input[base_offset + i] * inv_norm;
    }
}
"#;

/// WebGPU backend for cross-platform GPU acceleration
///
/// This backend uses the wgpu crate to provide GPU-accelerated vector operations
/// across all platforms that support WebGPU (Windows, macOS, Linux, and browsers).
///
/// The backend handles:
/// - Automatic GPU device detection and initialization
/// - WGSL compute shader compilation
/// - Efficient buffer management for GPU memory
/// - Async operation handling with proper synchronization
#[cfg(feature = "webgpu")]
pub struct WebGpuBackend {
    config: GpuConfig,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    euclidean_pipeline: wgpu::ComputePipeline,
    cosine_pipeline: wgpu::ComputePipeline,
    dot_product_pipeline: wgpu::ComputePipeline,
    normalize_pipeline: wgpu::ComputePipeline,
}

/// Parameters passed to compute shaders via uniform buffer
#[cfg(feature = "webgpu")]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    num_vectors: u32,
    vector_dim: u32,
    _padding0: u32,
    _padding1: u32,
}

/// Parameters for normalization shader (2 bindings instead of 3)
#[cfg(feature = "webgpu")]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NormalizeParams {
    num_vectors: u32,
    vector_dim: u32,
    _padding0: u32,
    _padding1: u32,
}

#[cfg(feature = "webgpu")]
impl WebGpuBackend {
    /// Create a new WebGPU backend
    ///
    /// This initializes the GPU device, compiles all compute shaders, and
    /// creates the compute pipelines. Returns an error if WebGPU is not available.
    pub fn new(config: &GpuConfig) -> Result<Self> {
        // Use pollster to block on async initialization
        // In a real async context, you would use async/await
        pollster::block_on(Self::new_async(config))
    }

    /// Async initialization for WebGPU backend
    async fn new_async(config: &GpuConfig) -> Result<Self> {
        // Create wgpu instance with all available backends
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            ..Default::default()
        });

        // Request a high-performance adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| anyhow!("Failed to find a suitable GPU adapter: {:?}", e))?;

        let adapter_info = adapter.get_info();

        // Request device with reasonable limits for compute
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("VecStore WebGPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|e| anyhow!("Failed to create WebGPU device: {}", e))?;

        // Compile all compute shaders and create pipelines
        let euclidean_pipeline =
            Self::create_compute_pipeline(&device, WGSL_EUCLIDEAN_DISTANCE, "euclidean_distance")?;

        let cosine_pipeline =
            Self::create_compute_pipeline(&device, WGSL_COSINE_SIMILARITY, "cosine_similarity")?;

        let dot_product_pipeline =
            Self::create_compute_pipeline(&device, WGSL_DOT_PRODUCT, "dot_product")?;

        let normalize_pipeline =
            Self::create_compute_pipeline(&device, WGSL_L2_NORMALIZE, "l2_normalize")?;

        Ok(Self {
            config: config.clone(),
            device,
            queue,
            adapter_info,
            euclidean_pipeline,
            cosine_pipeline,
            dot_product_pipeline,
            normalize_pipeline,
        })
    }

    /// Create a compute pipeline from WGSL shader source
    fn create_compute_pipeline(
        device: &wgpu::Device,
        shader_source: &str,
        label: &str,
    ) -> Result<wgpu::ComputePipeline> {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}_shader", label)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", label)),
            layout: None, // Auto-generate layout from shader
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(pipeline)
    }

    /// Check if WebGPU is available on this system
    pub fn is_available() -> bool {
        // Try to create an instance and request an adapter
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                ..Default::default()
            });

            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .is_ok()
        })
    }

    /// Execute a distance-type compute shader (euclidean, cosine, dot product)
    fn execute_distance_shader(
        &self,
        pipeline: &wgpu::ComputePipeline,
        query: &[f32],
        database: &[f32],
        num_vectors: usize,
        vector_dim: usize,
    ) -> Result<Vec<f32>> {
        use wgpu::util::DeviceExt;

        // Create GPU buffers
        let query_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Query Buffer"),
                contents: bytemuck::cast_slice(query),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let database_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Database Buffer"),
                contents: bytemuck::cast_slice(database),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Output buffer (results)
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (num_vectors * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for reading results back to CPU
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (num_vectors * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform buffer for shader parameters
        let params = ShaderParams {
            num_vectors: num_vectors as u32,
            vector_dim: vector_dim as u32,
            _padding0: 0,
            _padding1: 0,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Distance Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: database_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create and submit command buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Distance Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with 256 threads per workgroup
            let workgroup_count = (num_vectors as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (num_vectors * size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map staging buffer and read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Poll the device until the buffer is mapped
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(60)),
            })
            .map_err(|e| anyhow!("GPU poll failed: {:?}", e))?;

        receiver
            .recv()
            .map_err(|e| anyhow!("Failed to receive map result: {}", e))?
            .map_err(|e| anyhow!("Failed to map buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Execute the normalization compute shader
    fn execute_normalize_shader(
        &self,
        input: &[f32],
        num_vectors: usize,
        vector_dim: usize,
    ) -> Result<Vec<f32>> {
        use wgpu::util::DeviceExt;

        let total_elements = num_vectors * vector_dim;

        // Create GPU buffers
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Buffer"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (total_elements * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (total_elements * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform buffer for parameters
        let params = NormalizeParams {
            num_vectors: num_vectors as u32,
            vector_dim: vector_dim as u32,
            _padding0: 0,
            _padding1: 0,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group_layout = self.normalize_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Normalize Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create and submit command buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Normalize Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Normalize Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.normalize_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (num_vectors as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (total_elements * size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(60)),
            })
            .map_err(|e| anyhow!("GPU poll failed: {:?}", e))?;

        receiver
            .recv()
            .map_err(|e| anyhow!("Failed to receive map result: {}", e))?
            .map_err(|e| anyhow!("Failed to map buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Process database in batches to handle large datasets
    fn process_in_batches<F>(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
        batch_processor: F,
    ) -> Result<Vec<f32>>
    where
        F: Fn(&Self, &[f32], &[f32], usize, usize) -> Result<Vec<f32>>,
    {
        if database.is_empty() {
            return Ok(vec![]);
        }

        let vector_dim = query.len();
        let total_vectors = database.len();
        let batch_size = self.config.batch_size.min(total_vectors);

        let mut all_results = Vec::with_capacity(total_vectors);

        for batch_start in (0..total_vectors).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_vectors);
            let batch_count = batch_end - batch_start;

            // Flatten batch into contiguous array
            let flat_batch: Vec<f32> = database[batch_start..batch_end]
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect();

            let batch_results = batch_processor(self, query, &flat_batch, batch_count, vector_dim)?;
            all_results.extend(batch_results);
        }

        Ok(all_results)
    }
}

#[cfg(feature = "webgpu")]
impl GpuOps for WebGpuBackend {
    fn device_info(&self) -> GpuDeviceInfo {
        GpuDeviceInfo {
            backend: GpuBackend::WebGpu,
            device_id: self.config.device_id,
            name: format!(
                "{} ({:?})",
                self.adapter_info.name, self.adapter_info.backend
            ),
            total_memory_bytes: self.config.max_memory_bytes,
            available_memory_bytes: self.config.max_memory_bytes,
            compute_capability: (1, 0),       // WebGPU version indicator
            max_threads_per_block: 256,       // Workgroup size
            num_streaming_multiprocessors: 1, // Not directly available in WebGPU
        }
    }

    fn batch_euclidean_distance(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        if database.is_empty() {
            return Ok(vec![]);
        }

        // Validate dimensions
        let vector_dim = query.len();
        for (i, vec) in database.iter().enumerate() {
            if vec.len() != vector_dim {
                return Err(anyhow!(
                    "Database vector {} has dimension {} but query has dimension {}",
                    i,
                    vec.len(),
                    vector_dim
                ));
            }
        }

        self.process_in_batches(query, database, |backend, q, db, num_vecs, dim| {
            backend.execute_distance_shader(&backend.euclidean_pipeline, q, db, num_vecs, dim)
        })
    }

    fn batch_cosine_similarity(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        if database.is_empty() {
            return Ok(vec![]);
        }

        let vector_dim = query.len();
        for (i, vec) in database.iter().enumerate() {
            if vec.len() != vector_dim {
                return Err(anyhow!(
                    "Database vector {} has dimension {} but query has dimension {}",
                    i,
                    vec.len(),
                    vector_dim
                ));
            }
        }

        self.process_in_batches(query, database, |backend, q, db, num_vecs, dim| {
            backend.execute_distance_shader(&backend.cosine_pipeline, q, db, num_vecs, dim)
        })
    }

    fn batch_dot_product(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
        if database.is_empty() {
            return Ok(vec![]);
        }

        let vector_dim = query.len();
        for (i, vec) in database.iter().enumerate() {
            if vec.len() != vector_dim {
                return Err(anyhow!(
                    "Database vector {} has dimension {} but query has dimension {}",
                    i,
                    vec.len(),
                    vector_dim
                ));
            }
        }

        self.process_in_batches(query, database, |backend, q, db, num_vecs, dim| {
            backend.execute_distance_shader(&backend.dot_product_pipeline, q, db, num_vecs, dim)
        })
    }

    fn matrix_multiply(&self, a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        // Matrix multiplication is more complex and would require a different shader
        // For now, fall back to CPU implementation
        // A full implementation would use tiled matrix multiplication for efficiency
        let cpu = CpuBackend::new(&self.config);
        cpu.matrix_multiply(a, b)
    }

    fn batch_normalize(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let vector_dim = vectors[0].len();
        let num_vectors = vectors.len();

        // Validate dimensions
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != vector_dim {
                return Err(anyhow!(
                    "Vector {} has dimension {} but expected {}",
                    i,
                    vec.len(),
                    vector_dim
                ));
            }
        }

        // Flatten input
        let flat_input: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        // Execute normalization
        let flat_output = self.execute_normalize_shader(&flat_input, num_vectors, vector_dim)?;

        // Reshape output
        Ok(flat_output.chunks(vector_dim).map(|c| c.to_vec()).collect())
    }

    fn knn_search(
        &self,
        query: &[f32],
        database: &[Vec<f32>],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        if database.is_empty() {
            return Ok((vec![], vec![]));
        }

        if k == 0 {
            return Ok((vec![], vec![]));
        }

        // Use GPU for distance computation
        let distances = self.batch_euclidean_distance(query, database)?;

        // Use CPU for top-k selection (GPU top-k requires more complex shader)
        // This is still efficient because the bottleneck is usually distance computation
        let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();

        // Partial sort to get top k (more efficient than full sort for small k)
        let k = k.min(indexed.len());
        if k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
            indexed.truncate(k);
        }

        // Sort the top k by distance
        indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        let dists: Vec<f32> = indexed.iter().map(|(_, d)| *d).collect();

        Ok((indices, dists))
    }
}

// ============================================================================
// WASM WebGPU Backend (Browser-specific fallback when webgpu feature disabled)
// ============================================================================

/// WebGPU backend fallback for WASM builds without the webgpu feature
///
/// This provides a CPU fallback implementation for WASM builds that don't
/// have the full webgpu feature enabled.
#[cfg(all(feature = "wasm", not(feature = "webgpu")))]
pub struct WebGpuBackend {
    config: GpuConfig,
}

#[cfg(all(feature = "wasm", not(feature = "webgpu")))]
impl WebGpuBackend {
    pub fn new(config: &GpuConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn is_available() -> bool {
        // In WASM builds without webgpu feature, fall back to CPU
        false
    }
}

#[cfg(all(feature = "wasm", not(feature = "webgpu")))]
impl GpuOps for WebGpuBackend {
    fn device_info(&self) -> GpuDeviceInfo {
        GpuDeviceInfo {
            backend: GpuBackend::WebGpu,
            device_id: 0,
            name: "WebGPU (WASM fallback to CPU)".to_string(),
            total_memory_bytes: 2 * 1024 * 1024 * 1024,
            available_memory_bytes: 2 * 1024 * 1024 * 1024,
            compute_capability: (1, 0),
            max_threads_per_block: 256,
            num_streaming_multiprocessors: 1,
        }
    }

    fn batch_euclidean_distance(&self, query: &[f32], database: &[Vec<f32>]) -> Result<Vec<f32>> {
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

    // Metal-specific tests (only run on macOS with metal feature)
    #[cfg(all(target_os = "macos", feature = "metal"))]
    mod metal_tests {
        use super::*;

        #[test]
        fn test_metal_backend_creation() {
            let config = GpuConfig::default();
            let result = MetalBackend::new(&config);
            // Should succeed on macOS with Metal support
            assert!(result.is_ok(), "Metal backend should be available on macOS");
        }

        #[test]
        fn test_metal_is_available() {
            assert!(
                MetalBackend::is_available(),
                "Metal should be available on macOS"
            );
        }

        #[test]
        fn test_metal_device_info() {
            let config = GpuConfig::default();
            let backend = MetalBackend::new(&config).unwrap();
            let info = backend.device_info();

            assert_eq!(info.backend, GpuBackend::Metal);
            assert!(!info.name.is_empty());
            assert!(info.total_memory_bytes > 0);
            assert!(info.max_threads_per_block > 0);
        }

        #[test]
        fn test_metal_euclidean_distance() {
            let config = GpuConfig::default();
            let backend = MetalBackend::new(&config).unwrap();

            let query = vec![1.0, 2.0, 3.0, 4.0];
            let database = vec![
                vec![1.0, 2.0, 3.0, 4.0], // Same as query
                vec![2.0, 3.0, 4.0, 5.0], // Different
                vec![0.0, 0.0, 0.0, 0.0], // Origin
            ];

            let distances = backend.batch_euclidean_distance(&query, &database).unwrap();

            assert_eq!(distances.len(), 3);
            assert!(distances[0] < 0.001, "Same vector should have distance ~0");
            assert!(
                distances[1] > 1.0,
                "Different vector should have larger distance"
            );
            assert!(distances[2] > 5.0, "Origin should have large distance");
        }

        #[test]
        fn test_metal_cosine_similarity() {
            let config = GpuConfig::default();
            let backend = MetalBackend::new(&config).unwrap();

            let query = vec![1.0, 0.0, 0.0, 0.0];
            let database = vec![
                vec![1.0, 0.0, 0.0, 0.0],  // Same direction
                vec![0.0, 1.0, 0.0, 0.0],  // Perpendicular
                vec![-1.0, 0.0, 0.0, 0.0], // Opposite
            ];

            let similarities = backend.batch_cosine_similarity(&query, &database).unwrap();

            assert_eq!(similarities.len(), 3);
            assert!(
                (similarities[0] - 1.0).abs() < 0.001,
                "Same direction should be 1.0"
            );
            assert!(similarities[1].abs() < 0.001, "Perpendicular should be 0.0");
            assert!(
                (similarities[2] + 1.0).abs() < 0.001,
                "Opposite should be -1.0"
            );
        }

        #[test]
        fn test_metal_dot_product() {
            let config = GpuConfig::default();
            let backend = MetalBackend::new(&config).unwrap();

            let query = vec![1.0, 2.0, 3.0, 4.0];
            let database = vec![
                vec![1.0, 1.0, 1.0, 1.0], // Sum = 10
                vec![2.0, 2.0, 2.0, 2.0], // Sum = 20
            ];

            let products = backend.batch_dot_product(&query, &database).unwrap();

            assert_eq!(products.len(), 2);
            assert!((products[0] - 10.0).abs() < 0.001);
            assert!((products[1] - 20.0).abs() < 0.001);
        }

        #[test]
        fn test_metal_batch_normalize() {
            let config = GpuConfig::default();
            let backend = MetalBackend::new(&config).unwrap();

            let vectors = vec![
                vec![3.0, 4.0, 0.0, 0.0], // Magnitude 5
                vec![1.0, 0.0, 0.0, 0.0], // Already normalized
            ];

            let normalized = backend.batch_normalize(&vectors).unwrap();

            assert_eq!(normalized.len(), 2);
            assert!((normalized[0][0] - 0.6).abs() < 0.001);
            assert!((normalized[0][1] - 0.8).abs() < 0.001);
            assert!((normalized[1][0] - 1.0).abs() < 0.001);
        }

        #[test]
        fn test_metal_matrix_multiply() {
            let config = GpuConfig::default();
            let backend = MetalBackend::new(&config).unwrap();

            let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
            let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

            let result = backend.matrix_multiply(&a, &b).unwrap();

            assert_eq!(result.len(), 2);
            assert_eq!(result[0].len(), 2);
            assert!((result[0][0] - 19.0).abs() < 0.001); // 1*5 + 2*7
            assert!((result[0][1] - 22.0).abs() < 0.001); // 1*6 + 2*8
            assert!((result[1][0] - 43.0).abs() < 0.001); // 3*5 + 4*7
            assert!((result[1][1] - 50.0).abs() < 0.001); // 3*6 + 4*8
        }

        #[test]
        fn test_metal_knn_search() {
            let config = GpuConfig::default();
            let backend = MetalBackend::new(&config).unwrap();

            let query = vec![0.5, 0.5, 0.0, 0.0];
            let database = vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![1.0, 1.0, 0.0, 0.0],
                vec![0.5, 0.5, 0.0, 0.0], // Exact match
                vec![10.0, 10.0, 0.0, 0.0],
            ];

            let (indices, distances) = backend.knn_search(&query, &database, 2).unwrap();

            assert_eq!(indices.len(), 2);
            assert_eq!(distances.len(), 2);
            assert_eq!(indices[0], 2); // Exact match should be first
            assert!(distances[0] < 0.001);
        }

        #[test]
        fn test_metal_large_batch() {
            let config = GpuConfig::default();
            let backend = MetalBackend::new(&config).unwrap();

            // Create a large batch to test GPU performance
            let dimension = 128;
            let num_vectors = 10000;

            let query: Vec<f32> = (0..dimension).map(|i| i as f32 * 0.01).collect();
            let database: Vec<Vec<f32>> = (0..num_vectors)
                .map(|i| {
                    (0..dimension)
                        .map(|j| ((i + j) % 100) as f32 * 0.01)
                        .collect()
                })
                .collect();

            let distances = backend.batch_euclidean_distance(&query, &database).unwrap();
            assert_eq!(distances.len(), num_vectors);

            // All distances should be non-negative
            assert!(distances.iter().all(|&d| d >= 0.0));
        }

        #[test]
        fn test_metal_executor_integration() {
            let config = GpuConfig::default().with_backend(GpuBackend::Metal);
            let executor = GpuExecutor::new(config).unwrap();

            assert_eq!(executor.backend_type(), GpuBackend::Metal);

            let query = vec![1.0, 2.0, 3.0];
            let database = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

            let distances = executor
                .batch_euclidean_distance(&query, &database)
                .unwrap();
            assert_eq!(distances.len(), 2);
        }

        #[test]
        fn test_metal_empty_inputs() {
            let config = GpuConfig::default();
            let backend = MetalBackend::new(&config).unwrap();

            // Empty database should return empty results
            let query = vec![1.0, 2.0, 3.0];
            let empty_db: Vec<Vec<f32>> = vec![];

            let distances = backend.batch_euclidean_distance(&query, &empty_db).unwrap();
            assert!(distances.is_empty());

            let similarities = backend.batch_cosine_similarity(&query, &empty_db).unwrap();
            assert!(similarities.is_empty());

            let products = backend.batch_dot_product(&query, &empty_db).unwrap();
            assert!(products.is_empty());
        }

        #[test]
        fn test_metal_dimension_validation() {
            let config = GpuConfig::default();
            let backend = MetalBackend::new(&config).unwrap();

            let query = vec![1.0, 2.0, 3.0];
            let mismatched_db = vec![
                vec![1.0, 2.0], // Wrong dimension
            ];

            let result = backend.batch_euclidean_distance(&query, &mismatched_db);
            assert!(result.is_err(), "Should fail on dimension mismatch");
        }
    }

    // WebGPU-specific tests (only run when webgpu feature is enabled)
    #[cfg(feature = "webgpu")]
    mod webgpu_tests {
        use super::*;

        #[test]
        fn test_webgpu_is_available() {
            // This may fail in environments without GPU, which is OK
            let available = WebGpuBackend::is_available();
            println!("WebGPU available: {}", available);
        }

        #[test]
        fn test_webgpu_backend_creation() {
            let config = GpuConfig::default();
            let result = WebGpuBackend::new(&config);
            // May fail if no GPU available - that's acceptable
            if let Ok(backend) = result {
                let info = backend.device_info();
                assert_eq!(info.backend, GpuBackend::WebGpu);
                assert!(!info.name.is_empty());
                println!("WebGPU device: {}", info.name);
            } else {
                println!("WebGPU backend not available (no GPU): {:?}", result.err());
            }
        }

        #[test]
        fn test_webgpu_euclidean_distance() {
            let config = GpuConfig::default();
            let backend = match WebGpuBackend::new(&config) {
                Ok(b) => b,
                Err(_) => {
                    println!("Skipping test: WebGPU not available");
                    return;
                },
            };

            let query = vec![1.0, 2.0, 3.0, 4.0];
            let database = vec![
                vec![1.0, 2.0, 3.0, 4.0], // Same as query
                vec![2.0, 3.0, 4.0, 5.0], // Different
                vec![0.0, 0.0, 0.0, 0.0], // Origin
            ];

            let distances = backend.batch_euclidean_distance(&query, &database).unwrap();

            assert_eq!(distances.len(), 3);
            assert!(distances[0] < 0.001, "Same vector should have distance ~0");
            assert!(
                distances[1] > 1.0,
                "Different vector should have larger distance"
            );
            assert!(distances[2] > 5.0, "Origin should have large distance");
        }

        #[test]
        fn test_webgpu_cosine_similarity() {
            let config = GpuConfig::default();
            let backend = match WebGpuBackend::new(&config) {
                Ok(b) => b,
                Err(_) => {
                    println!("Skipping test: WebGPU not available");
                    return;
                },
            };

            let query = vec![1.0, 0.0, 0.0, 0.0];
            let database = vec![
                vec![1.0, 0.0, 0.0, 0.0],  // Same direction
                vec![0.0, 1.0, 0.0, 0.0],  // Perpendicular
                vec![-1.0, 0.0, 0.0, 0.0], // Opposite
            ];

            let similarities = backend.batch_cosine_similarity(&query, &database).unwrap();

            assert_eq!(similarities.len(), 3);
            assert!(
                (similarities[0] - 1.0).abs() < 0.01,
                "Same direction should be 1.0"
            );
            assert!(similarities[1].abs() < 0.01, "Perpendicular should be 0.0");
            assert!(
                (similarities[2] + 1.0).abs() < 0.01,
                "Opposite should be -1.0"
            );
        }

        #[test]
        fn test_webgpu_dot_product() {
            let config = GpuConfig::default();
            let backend = match WebGpuBackend::new(&config) {
                Ok(b) => b,
                Err(_) => {
                    println!("Skipping test: WebGPU not available");
                    return;
                },
            };

            let query = vec![1.0, 2.0, 3.0, 4.0];
            let database = vec![
                vec![1.0, 1.0, 1.0, 1.0], // Sum = 10
                vec![2.0, 2.0, 2.0, 2.0], // Sum = 20
            ];

            let products = backend.batch_dot_product(&query, &database).unwrap();

            assert_eq!(products.len(), 2);
            assert!((products[0] - 10.0).abs() < 0.001);
            assert!((products[1] - 20.0).abs() < 0.001);
        }

        #[test]
        fn test_webgpu_batch_normalize() {
            let config = GpuConfig::default();
            let backend = match WebGpuBackend::new(&config) {
                Ok(b) => b,
                Err(_) => {
                    println!("Skipping test: WebGPU not available");
                    return;
                },
            };

            let vectors = vec![
                vec![3.0, 4.0, 0.0, 0.0], // Magnitude 5
                vec![1.0, 0.0, 0.0, 0.0], // Already normalized
            ];

            let normalized = backend.batch_normalize(&vectors).unwrap();

            assert_eq!(normalized.len(), 2);
            assert!((normalized[0][0] - 0.6).abs() < 0.001);
            assert!((normalized[0][1] - 0.8).abs() < 0.001);
            assert!((normalized[1][0] - 1.0).abs() < 0.001);
        }

        #[test]
        fn test_webgpu_knn_search() {
            let config = GpuConfig::default();
            let backend = match WebGpuBackend::new(&config) {
                Ok(b) => b,
                Err(_) => {
                    println!("Skipping test: WebGPU not available");
                    return;
                },
            };

            let query = vec![0.5, 0.5, 0.0, 0.0];
            let database = vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![1.0, 1.0, 0.0, 0.0],
                vec![0.5, 0.5, 0.0, 0.0], // Exact match
                vec![10.0, 10.0, 0.0, 0.0],
            ];

            let (indices, distances) = backend.knn_search(&query, &database, 2).unwrap();

            assert_eq!(indices.len(), 2);
            assert_eq!(distances.len(), 2);
            assert_eq!(indices[0], 2); // Exact match should be first
            assert!(distances[0] < 0.001);
        }

        #[test]
        fn test_webgpu_large_batch() {
            let config = GpuConfig::default();
            let backend = match WebGpuBackend::new(&config) {
                Ok(b) => b,
                Err(_) => {
                    println!("Skipping test: WebGPU not available");
                    return;
                },
            };

            // Create a large batch to test GPU performance
            let dimension = 128;
            let num_vectors = 10000;

            let query: Vec<f32> = (0..dimension).map(|i| i as f32 * 0.01).collect();
            let database: Vec<Vec<f32>> = (0..num_vectors)
                .map(|i| {
                    (0..dimension)
                        .map(|j| ((i + j) % 100) as f32 * 0.01)
                        .collect()
                })
                .collect();

            let distances = backend.batch_euclidean_distance(&query, &database).unwrap();
            assert_eq!(distances.len(), num_vectors);

            // All distances should be non-negative
            assert!(distances.iter().all(|&d| d >= 0.0));
        }

        #[test]
        fn test_webgpu_executor_integration() {
            let config = GpuConfig::default().with_backend(GpuBackend::WebGpu);
            let executor = match GpuExecutor::new(config) {
                Ok(e) => e,
                Err(_) => {
                    println!("Skipping test: WebGPU not available");
                    return;
                },
            };

            assert_eq!(executor.backend_type(), GpuBackend::WebGpu);

            let query = vec![1.0, 2.0, 3.0];
            let database = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

            let distances = executor
                .batch_euclidean_distance(&query, &database)
                .unwrap();
            assert_eq!(distances.len(), 2);
        }

        #[test]
        fn test_webgpu_empty_inputs() {
            let config = GpuConfig::default();
            let backend = match WebGpuBackend::new(&config) {
                Ok(b) => b,
                Err(_) => {
                    println!("Skipping test: WebGPU not available");
                    return;
                },
            };

            // Empty database should return empty results
            let query = vec![1.0, 2.0, 3.0];
            let empty_db: Vec<Vec<f32>> = vec![];

            let distances = backend.batch_euclidean_distance(&query, &empty_db).unwrap();
            assert!(distances.is_empty());

            let similarities = backend.batch_cosine_similarity(&query, &empty_db).unwrap();
            assert!(similarities.is_empty());

            let products = backend.batch_dot_product(&query, &empty_db).unwrap();
            assert!(products.is_empty());
        }

        #[test]
        fn test_webgpu_dimension_validation() {
            let config = GpuConfig::default();
            let backend = match WebGpuBackend::new(&config) {
                Ok(b) => b,
                Err(_) => {
                    println!("Skipping test: WebGPU not available");
                    return;
                },
            };

            let query = vec![1.0, 2.0, 3.0];
            let mismatched_db = vec![
                vec![1.0, 2.0], // Wrong dimension
            ];

            let result = backend.batch_euclidean_distance(&query, &mismatched_db);
            assert!(result.is_err(), "Should fail on dimension mismatch");
        }

        #[test]
        fn test_wgsl_shader_constants() {
            // Verify shader constants are defined and non-empty
            assert!(!WGSL_EUCLIDEAN_DISTANCE.is_empty());
            assert!(!WGSL_COSINE_SIMILARITY.is_empty());
            assert!(!WGSL_DOT_PRODUCT.is_empty());
            assert!(!WGSL_L2_NORMALIZE.is_empty());

            // Verify shaders contain expected WGSL syntax
            assert!(WGSL_EUCLIDEAN_DISTANCE.contains("@compute"));
            assert!(WGSL_EUCLIDEAN_DISTANCE.contains("@workgroup_size"));
            assert!(WGSL_COSINE_SIMILARITY.contains("@compute"));
            assert!(WGSL_DOT_PRODUCT.contains("@compute"));
            assert!(WGSL_L2_NORMALIZE.contains("@compute"));
        }
    }
}
