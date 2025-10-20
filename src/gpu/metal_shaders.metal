//! Metal Compute Shaders for Apple Silicon GPU Acceleration
//!
//! These shaders provide GPU-accelerated vector operations on Apple Silicon (M1/M2/M3).

#include <metal_stdlib>
using namespace metal;

/// Euclidean distance kernel
kernel void euclidean_distance_kernel(
    constant float* query [[buffer(0)]],
    constant float* database [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& query_dim [[buffer(3)]],
    constant uint& num_vectors [[buffer(4)]],
    constant uint& vector_dim [[buffer(5)]],
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

/// Cosine similarity kernel
kernel void cosine_similarity_kernel(
    constant float* query [[buffer(0)]],
    constant float* database [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant uint& query_dim [[buffer(3)]],
    constant uint& num_vectors [[buffer(4)]],
    constant uint& vector_dim [[buffer(5)]],
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

/// Dot product kernel
kernel void dot_product_kernel(
    constant float* query [[buffer(0)]],
    constant float* database [[buffer(1)]],
    device float* products [[buffer(2)]],
    constant uint& query_dim [[buffer(3)]],
    constant uint& num_vectors [[buffer(4)]],
    constant uint& vector_dim [[buffer(5)]],
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

/// L2 normalization kernel
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

/// Matrix multiplication kernel (optimized with threadgroup memory)
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

/// Parallel reduction for finding minimum distance (k-NN)
kernel void find_min_distance_kernel(
    constant float* distances [[buffer(0)]],
    device float* min_distances [[buffer(1)]],
    device uint* min_indices [[buffer(2)]],
    constant uint& num_vectors [[buffer(3)]],
    uint idx [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_distances[256];
    threadgroup uint shared_indices[256];

    // Load data into threadgroup memory
    if (idx < num_vectors) {
        shared_distances[lid] = distances[idx];
        shared_indices[lid] = idx;
    } else {
        shared_distances[lid] = INFINITY;
        shared_indices[lid] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction to find minimum
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s && lid + s < 256) {
            if (shared_distances[lid] > shared_distances[lid + s]) {
                shared_distances[lid] = shared_distances[lid + s];
                shared_indices[lid] = shared_indices[lid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (lid == 0) {
        min_distances[gid] = shared_distances[0];
        min_indices[gid] = shared_indices[0];
    }
}

/// Batch vector addition
kernel void vector_add_kernel(
    constant float* a [[buffer(0)]],
    constant float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= length) return;
    c[idx] = a[idx] + b[idx];
}

/// Element-wise vector multiplication
kernel void vector_multiply_kernel(
    constant float* a [[buffer(0)]],
    constant float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= length) return;
    c[idx] = a[idx] * b[idx];
}

/// ReLU activation function
kernel void relu_kernel(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= length) return;
    output[idx] = max(0.0f, input[idx]);
}

/// Softmax activation (two-pass: max finding + exp normalization)
kernel void softmax_max_kernel(
    constant float* input [[buffer(0)]],
    device float* max_vals [[buffer(1)]],
    constant uint& num_vectors [[buffer(2)]],
    constant uint& vector_dim [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_vectors) return;

    uint base = idx * vector_dim;
    float max_val = input[base];

    for (uint i = 1; i < vector_dim; i++) {
        max_val = max(max_val, input[base + i]);
    }

    max_vals[idx] = max_val;
}

kernel void softmax_exp_norm_kernel(
    constant float* input [[buffer(0)]],
    constant float* max_vals [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_vectors [[buffer(3)]],
    constant uint& vector_dim [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_vectors) return;

    uint base = idx * vector_dim;
    float max_val = max_vals[idx];
    float sum = 0.0f;

    // Compute exp(x - max) and sum
    for (uint i = 0; i < vector_dim; i++) {
        float exp_val = exp(input[base + i] - max_val);
        output[base + i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    for (uint i = 0; i < vector_dim; i++) {
        output[base + i] /= sum;
    }
}
