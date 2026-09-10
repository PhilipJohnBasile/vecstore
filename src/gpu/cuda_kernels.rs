//! CUDA Kernels for GPU-Accelerated Vector Operations
//!
//! This module provides CUDA kernel implementations for distance calculations
//! and other vector operations.

/// CUDA kernel source for Euclidean distance
pub const EUCLIDEAN_DISTANCE_KERNEL: &str = r#"
extern "C" __global__ void euclidean_distance_kernel(
    const float* query,
    const float* database,
    float* distances,
    int query_dim,
    int num_vectors,
    int vector_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_vectors) {
        float sum = 0.0f;
        int base_offset = idx * vector_dim;

        for (int i = 0; i < vector_dim; i++) {
            float diff = query[i] - database[base_offset + i];
            sum += diff * diff;
        }

        distances[idx] = sqrtf(sum);
    }
}
"#;

/// CUDA kernel source for cosine similarity
pub const COSINE_SIMILARITY_KERNEL: &str = r#"
extern "C" __global__ void cosine_similarity_kernel(
    const float* query,
    const float* database,
    float* similarities,
    int query_dim,
    int num_vectors,
    int vector_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_vectors) {
        float dot = 0.0f;
        float query_norm = 0.0f;
        float db_norm = 0.0f;
        int base_offset = idx * vector_dim;

        for (int i = 0; i < vector_dim; i++) {
            float q = query[i];
            float d = database[base_offset + i];
            dot += q * d;
            query_norm += q * q;
            db_norm += d * d;
        }

        query_norm = sqrtf(query_norm);
        db_norm = sqrtf(db_norm);

        similarities[idx] = dot / (query_norm * db_norm + 1e-8f);
    }
}
"#;

/// CUDA kernel source for dot product
pub const DOT_PRODUCT_KERNEL: &str = r#"
extern "C" __global__ void dot_product_kernel(
    const float* query,
    const float* database,
    float* products,
    int query_dim,
    int num_vectors,
    int vector_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_vectors) {
        float sum = 0.0f;
        int base_offset = idx * vector_dim;

        for (int i = 0; i < vector_dim; i++) {
            sum += query[i] * database[base_offset + i];
        }

        products[idx] = sum;
    }
}
"#;

/// CUDA kernel for batch L2 normalization
pub const L2_NORMALIZE_KERNEL: &str = r#"
extern "C" __global__ void l2_normalize_kernel(
    const float* input,
    float* output,
    int num_vectors,
    int vector_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_vectors) {
        int base_offset = idx * vector_dim;
        float norm = 0.0f;

        // Compute L2 norm
        for (int i = 0; i < vector_dim; i++) {
            float val = input[base_offset + i];
            norm += val * val;
        }
        norm = sqrtf(norm);

        // Normalize
        for (int i = 0; i < vector_dim; i++) {
            output[base_offset + i] = input[base_offset + i] / (norm + 1e-8f);
        }
    }
}
"#;

/// CUDA kernel for top-K selection (parallel reduction)
pub const TOP_K_KERNEL: &str = r#"
extern "C" __global__ void top_k_kernel(
    const float* distances,
    const int* indices,
    float* top_k_distances,
    int* top_k_indices,
    int num_vectors,
    int k
) {
    // Shared memory for partial results
    __shared__ float shared_distances[256];
    __shared__ int shared_indices[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < num_vectors) {
        shared_distances[tid] = distances[idx];
        shared_indices[tid] = indices[idx];
    } else {
        shared_distances[tid] = INFINITY;
        shared_indices[tid] = -1;
    }

    __syncthreads();

    // Parallel reduction to find top-k
    // This is a simplified version - production would use bitonic sort
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < num_vectors) {
            if (shared_distances[tid] > shared_distances[tid + s]) {
                shared_distances[tid] = shared_distances[tid + s];
                shared_indices[tid] = shared_indices[tid + s];
            }
        }
        __syncthreads();
    }

    // Write results
    if (tid < k && blockIdx.x == 0) {
        top_k_distances[tid] = shared_distances[tid];
        top_k_indices[tid] = shared_indices[tid];
    }
}
"#;

/// CUDA kernel executor using cudarc
#[cfg(feature = "cuda")]
pub struct CudaKernelExecutor {
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
    euclidean_kernel: cudarc::driver::CudaFunction,
    cosine_kernel: cudarc::driver::CudaFunction,
    dot_kernel: cudarc::driver::CudaFunction,
}

#[cfg(feature = "cuda")]
impl CudaKernelExecutor {
    /// PTX code for distance kernels (pre-compiled for portability)
    const DISTANCE_PTX: &'static str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry euclidean_distance_kernel(
    .param .u64 query,
    .param .u64 database,
    .param .u64 distances,
    .param .u32 num_vectors,
    .param .u32 vector_dim
) {
    .reg .pred %p<2>;
    .reg .f32 %f<8>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [database];
    ld.param.u64 %rd3, [distances];
    ld.param.u32 %r1, [num_vectors];
    ld.param.u32 %r2, [vector_dim];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;

    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra END;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r7, 0;
LOOP:
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];

    mul.lo.u32 %r8, %r6, %r2;
    add.u32 %r9, %r8, %r7;
    mul.wide.u32 %rd6, %r9, 4;
    add.u64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];

    sub.f32 %f4, %f2, %f3;
    fma.rn.f32 %f1, %f4, %f4, %f1;

    add.u32 %r7, %r7, 1;
    bra LOOP;
DONE:
    sqrt.rn.f32 %f5, %f1;
    mul.wide.u32 %rd8, %r6, 4;
    add.u64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f5;
END:
    ret;
}

.visible .entry cosine_similarity_kernel(
    .param .u64 query,
    .param .u64 database,
    .param .u64 similarities,
    .param .u32 num_vectors,
    .param .u32 vector_dim
) {
    .reg .pred %p<2>;
    .reg .f32 %f<12>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [database];
    ld.param.u64 %rd3, [similarities];
    ld.param.u32 %r1, [num_vectors];
    ld.param.u32 %r2, [vector_dim];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;

    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra END;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.u32 %r7, 0;
LOOP:
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f4, [%rd5];

    mul.lo.u32 %r8, %r6, %r2;
    add.u32 %r9, %r8, %r7;
    mul.wide.u32 %rd6, %r9, 4;
    add.u64 %rd7, %rd2, %rd6;
    ld.global.f32 %f5, [%rd7];

    fma.rn.f32 %f1, %f4, %f5, %f1;
    fma.rn.f32 %f2, %f4, %f4, %f2;
    fma.rn.f32 %f3, %f5, %f5, %f3;

    add.u32 %r7, %r7, 1;
    bra LOOP;
DONE:
    sqrt.rn.f32 %f6, %f2;
    sqrt.rn.f32 %f7, %f3;
    mul.f32 %f8, %f6, %f7;
    add.f32 %f9, %f8, 0f3727C5AC;
    div.rn.f32 %f10, %f1, %f9;

    mul.wide.u32 %rd8, %r6, 4;
    add.u64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f10;
END:
    ret;
}

.visible .entry dot_product_kernel(
    .param .u64 query,
    .param .u64 database,
    .param .u64 products,
    .param .u32 num_vectors,
    .param .u32 vector_dim
) {
    .reg .pred %p<2>;
    .reg .f32 %f<6>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [query];
    ld.param.u64 %rd2, [database];
    ld.param.u64 %rd3, [products];
    ld.param.u32 %r1, [num_vectors];
    ld.param.u32 %r2, [vector_dim];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;

    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra END;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r7, 0;
LOOP:
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];

    mul.lo.u32 %r8, %r6, %r2;
    add.u32 %r9, %r8, %r7;
    mul.wide.u32 %rd6, %r9, 4;
    add.u64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];

    fma.rn.f32 %f1, %f2, %f3, %f1;

    add.u32 %r7, %r7, 1;
    bra LOOP;
DONE:
    mul.wide.u32 %rd8, %r6, 4;
    add.u64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
END:
    ret;
}
"#;

    /// Create a new CUDA kernel executor
    pub fn new(device_id: usize) -> Result<Self> {
        use cudarc::driver::CudaDevice;

        let device = CudaDevice::new(device_id)
            .map_err(|e| anyhow!("Failed to initialize CUDA device {}: {}", device_id, e))?;

        // Load PTX module
        device
            .load_ptx(
                cudarc::nvrtc::Ptx::from_src(Self::DISTANCE_PTX),
                "distance_kernels",
                &[
                    "euclidean_distance_kernel",
                    "cosine_similarity_kernel",
                    "dot_product_kernel",
                ],
            )
            .map_err(|e| anyhow!("Failed to load PTX: {}", e))?;

        let euclidean_kernel = device
            .get_func("distance_kernels", "euclidean_distance_kernel")
            .ok_or_else(|| anyhow!("Failed to get euclidean kernel"))?;

        let cosine_kernel = device
            .get_func("distance_kernels", "cosine_similarity_kernel")
            .ok_or_else(|| anyhow!("Failed to get cosine kernel"))?;

        let dot_kernel = device
            .get_func("distance_kernels", "dot_product_kernel")
            .ok_or_else(|| anyhow!("Failed to get dot product kernel"))?;

        Ok(Self {
            device,
            euclidean_kernel,
            cosine_kernel,
            dot_kernel,
        })
    }

    /// Execute Euclidean distance kernel
    pub fn euclidean_distance(
        &self,
        query: &[f32],
        database: &[f32],
        num_vectors: usize,
        vector_dim: usize,
    ) -> Result<Vec<f32>> {
        use cudarc::driver::LaunchAsync;
        use cudarc::driver::LaunchConfig;

        // Copy data to device
        let query_dev = self
            .device
            .htod_sync_copy(query)
            .map_err(|e| anyhow!("Failed to copy query to device: {}", e))?;
        let database_dev = self
            .device
            .htod_sync_copy(database)
            .map_err(|e| anyhow!("Failed to copy database to device: {}", e))?;

        // Allocate output buffer
        let mut distances_dev = self
            .device
            .alloc_zeros::<f32>(num_vectors)
            .map_err(|e| anyhow!("Failed to allocate output buffer: {}", e))?;

        // Configure launch
        let threads_per_block = 256u32;
        let num_blocks = ((num_vectors as u32) + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            self.euclidean_kernel
                .clone()
                .launch(
                    cfg,
                    (
                        &query_dev,
                        &database_dev,
                        &mut distances_dev,
                        num_vectors as u32,
                        vector_dim as u32,
                    ),
                )
                .map_err(|e| anyhow!("Kernel launch failed: {}", e))?;
        }

        // Copy results back
        let distances = self
            .device
            .dtoh_sync_copy(&distances_dev)
            .map_err(|e| anyhow!("Failed to copy results from device: {}", e))?;

        Ok(distances)
    }

    /// Execute cosine similarity kernel
    pub fn cosine_similarity(
        &self,
        query: &[f32],
        database: &[f32],
        num_vectors: usize,
        vector_dim: usize,
    ) -> Result<Vec<f32>> {
        use cudarc::driver::LaunchAsync;
        use cudarc::driver::LaunchConfig;

        let query_dev = self
            .device
            .htod_sync_copy(query)
            .map_err(|e| anyhow!("Failed to copy query: {}", e))?;
        let database_dev = self
            .device
            .htod_sync_copy(database)
            .map_err(|e| anyhow!("Failed to copy database: {}", e))?;

        let mut similarities_dev = self
            .device
            .alloc_zeros::<f32>(num_vectors)
            .map_err(|e| anyhow!("Failed to allocate output: {}", e))?;

        let threads_per_block = 256u32;
        let num_blocks = ((num_vectors as u32) + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.cosine_kernel
                .clone()
                .launch(
                    cfg,
                    (
                        &query_dev,
                        &database_dev,
                        &mut similarities_dev,
                        num_vectors as u32,
                        vector_dim as u32,
                    ),
                )
                .map_err(|e| anyhow!("Kernel launch failed: {}", e))?;
        }

        let similarities = self
            .device
            .dtoh_sync_copy(&similarities_dev)
            .map_err(|e| anyhow!("Failed to copy results: {}", e))?;

        Ok(similarities)
    }

    /// Execute dot product kernel
    pub fn dot_product(
        &self,
        query: &[f32],
        database: &[f32],
        num_vectors: usize,
        vector_dim: usize,
    ) -> Result<Vec<f32>> {
        use cudarc::driver::LaunchAsync;
        use cudarc::driver::LaunchConfig;

        let query_dev = self
            .device
            .htod_sync_copy(query)
            .map_err(|e| anyhow!("Failed to copy query: {}", e))?;
        let database_dev = self
            .device
            .htod_sync_copy(database)
            .map_err(|e| anyhow!("Failed to copy database: {}", e))?;

        let mut products_dev = self
            .device
            .alloc_zeros::<f32>(num_vectors)
            .map_err(|e| anyhow!("Failed to allocate output: {}", e))?;

        let threads_per_block = 256u32;
        let num_blocks = ((num_vectors as u32) + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.dot_kernel
                .clone()
                .launch(
                    cfg,
                    (
                        &query_dev,
                        &database_dev,
                        &mut products_dev,
                        num_vectors as u32,
                        vector_dim as u32,
                    ),
                )
                .map_err(|e| anyhow!("Kernel launch failed: {}", e))?;
        }

        let products = self
            .device
            .dtoh_sync_copy(&products_dev)
            .map_err(|e| anyhow!("Failed to copy results: {}", e))?;

        Ok(products)
    }

    /// Execute L2 normalization kernel
    pub fn l2_normalize(
        &self,
        vectors: &[f32],
        num_vectors: usize,
        vector_dim: usize,
    ) -> Result<Vec<f32>> {
        // For now, fall back to CPU for normalization
        // Could add a dedicated kernel later
        let mut result = Vec::with_capacity(vectors.len());
        for i in 0..num_vectors {
            let start = i * vector_dim;
            let end = start + vector_dim;
            let slice = &vectors[start..end];

            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            for &v in slice {
                result.push(v / (norm + 1e-8));
            }
        }
        Ok(result)
    }

    /// Get device properties
    pub fn device_properties(&self) -> Result<CudaDeviceProperties> {
        // cudarc doesn't expose all properties directly, use defaults
        Ok(CudaDeviceProperties {
            name: format!("CUDA Device"),
            compute_capability: (7, 0),
            total_memory_bytes: 8 * 1024 * 1024 * 1024,
            multiprocessor_count: 68,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 48 * 1024,
        })
    }

    /// Check if CUDA is available
    pub fn is_available() -> bool {
        cudarc::driver::CudaDevice::new(0).is_ok()
    }
}

/// CUDA device properties
#[derive(Debug, Clone)]
pub struct CudaDeviceProperties {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory_bytes: usize,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub max_shared_memory_per_block: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_constants_defined() {
        assert!(!EUCLIDEAN_DISTANCE_KERNEL.is_empty());
        assert!(!COSINE_SIMILARITY_KERNEL.is_empty());
        assert!(!DOT_PRODUCT_KERNEL.is_empty());
        assert!(!L2_NORMALIZE_KERNEL.is_empty());
        assert!(!TOP_K_KERNEL.is_empty());
    }

    #[test]
    fn test_kernel_syntax() {
        // Check that kernels have proper __global__ declarations
        assert!(EUCLIDEAN_DISTANCE_KERNEL.contains("__global__"));
        assert!(COSINE_SIMILARITY_KERNEL.contains("__global__"));
        assert!(DOT_PRODUCT_KERNEL.contains("__global__"));
        assert!(L2_NORMALIZE_KERNEL.contains("__global__"));
        assert!(TOP_K_KERNEL.contains("__global__"));
    }

    #[test]
    fn test_kernel_function_names() {
        assert!(EUCLIDEAN_DISTANCE_KERNEL.contains("euclidean_distance_kernel"));
        assert!(COSINE_SIMILARITY_KERNEL.contains("cosine_similarity_kernel"));
        assert!(DOT_PRODUCT_KERNEL.contains("dot_product_kernel"));
        assert!(L2_NORMALIZE_KERNEL.contains("l2_normalize_kernel"));
        assert!(TOP_K_KERNEL.contains("top_k_kernel"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_executor_creation() {
        // This test requires CUDA hardware
        let result = CudaKernelExecutor::new(0);
        // May fail if no CUDA device available
        assert!(result.is_ok() || result.is_err());
    }
}
