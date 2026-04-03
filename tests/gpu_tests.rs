// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! Tests for GPU acceleration module

use vecstore::gpu::{CpuBackend, GpuBackend, GpuConfig, GpuOps};

/// Test GPU config defaults
#[test]
fn test_gpu_config_defaults() {
    let config = GpuConfig::default();

    // Default should auto-detect backend (None)
    assert!(config.backend.is_none());
    assert!(config.batch_size > 0);
    assert!(config.max_memory_bytes > 0);
    assert!(config.enable_memory_pool);
    assert!(config.async_execution);
}

/// Test GPU config builder pattern
#[test]
fn test_gpu_config_builder() {
    let config = GpuConfig::default()
        .with_backend(GpuBackend::Cpu)
        .with_batch_size(1024)
        .with_max_memory_bytes(1024 * 1024 * 1024); // 1GB

    assert_eq!(config.backend, Some(GpuBackend::Cpu));
    assert_eq!(config.batch_size, 1024);
    assert_eq!(config.max_memory_bytes, 1024 * 1024 * 1024);
}

/// Test GPU config with device id
#[test]
fn test_gpu_config_device_id() {
    let config = GpuConfig::default().with_device_id(1);

    assert_eq!(config.device_id, 1);
}

/// Test CPU backend creation
#[test]
fn test_cpu_backend_creation() {
    let config = GpuConfig::default().with_backend(GpuBackend::Cpu);
    let backend = CpuBackend::new(&config);

    // CPU backend should always succeed
    let info = backend.device_info();
    assert!(info.name.contains("CPU") || !info.name.is_empty());
}

/// Test CPU backend batch euclidean distance
#[test]
fn test_cpu_euclidean_distance() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);

    let query = vec![1.0, 0.0, 0.0];
    let database = vec![
        vec![1.0, 0.0, 0.0], // Distance 0
        vec![0.0, 1.0, 0.0], // Distance sqrt(2)
        vec![0.0, 0.0, 1.0], // Distance sqrt(2)
        vec![2.0, 0.0, 0.0], // Distance 1
    ];

    let distances = backend.batch_euclidean_distance(&query, &database).unwrap();

    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 0.0).abs() < 0.001); // Same vector
    assert!((distances[1] - 1.414).abs() < 0.01); // sqrt(2)
    assert!((distances[2] - 1.414).abs() < 0.01); // sqrt(2)
    assert!((distances[3] - 1.0).abs() < 0.001); // Distance 1
}

/// Test CPU backend batch cosine similarity
#[test]
fn test_cpu_cosine_similarity() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);

    let query = vec![1.0, 0.0, 0.0];
    let database = vec![
        vec![1.0, 0.0, 0.0],  // Similarity 1.0
        vec![-1.0, 0.0, 0.0], // Similarity -1.0
        vec![0.0, 1.0, 0.0],  // Similarity 0.0
    ];

    let similarities = backend.batch_cosine_similarity(&query, &database).unwrap();

    assert_eq!(similarities.len(), 3);
    assert!((similarities[0] - 1.0).abs() < 0.001);
    assert!((similarities[1] - (-1.0)).abs() < 0.001);
    assert!(similarities[2].abs() < 0.001);
}

/// Test CPU backend batch dot product
#[test]
fn test_cpu_dot_product() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);

    let query = vec![1.0, 2.0, 3.0];
    let database = vec![
        vec![1.0, 1.0, 1.0], // Dot = 6
        vec![2.0, 0.0, 0.0], // Dot = 2
        vec![0.0, 0.0, 1.0], // Dot = 3
    ];

    let products = backend.batch_dot_product(&query, &database).unwrap();

    assert_eq!(products.len(), 3);
    assert!((products[0] - 6.0).abs() < 0.001);
    assert!((products[1] - 2.0).abs() < 0.001);
    assert!((products[2] - 3.0).abs() < 0.001);
}

/// Test CPU backend batch normalization
#[test]
fn test_cpu_normalize() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);

    let vectors = vec![
        vec![3.0, 4.0, 0.0], // Norm = 5
        vec![1.0, 0.0, 0.0], // Already normalized
        vec![0.0, 0.0, 2.0], // Norm = 2
    ];

    let normalized = backend.batch_normalize(&vectors).unwrap();

    assert_eq!(normalized.len(), 3);

    // Check each vector is unit length
    for vec in &normalized {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.001,
            "Vector not normalized: norm = {}",
            norm
        );
    }

    // Check first vector: [3/5, 4/5, 0]
    assert!((normalized[0][0] - 0.6).abs() < 0.001);
    assert!((normalized[0][1] - 0.8).abs() < 0.001);
}

/// Test CPU backend KNN search
#[test]
fn test_cpu_knn_search() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);

    let query = vec![0.0, 0.0, 0.0];
    let database = vec![
        vec![1.0, 0.0, 0.0], // Distance 1
        vec![2.0, 0.0, 0.0], // Distance 2
        vec![0.5, 0.0, 0.0], // Distance 0.5 (closest)
        vec![3.0, 0.0, 0.0], // Distance 3
    ];

    let (indices, distances) = backend.knn_search(&query, &database, 2).unwrap();

    assert_eq!(indices.len(), 2);
    assert_eq!(distances.len(), 2);

    // Should return indices of closest vectors
    assert_eq!(indices[0], 2); // 0.5 is closest
    assert_eq!(indices[1], 0); // 1.0 is second closest

    // Distances should be sorted
    assert!(distances[0] <= distances[1]);
}

/// Test empty database handling
#[test]
fn test_empty_database() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);

    let query = vec![1.0, 0.0, 0.0];
    let database: Vec<Vec<f32>> = vec![];

    let distances = backend.batch_euclidean_distance(&query, &database).unwrap();
    assert!(distances.is_empty());

    let similarities = backend.batch_cosine_similarity(&query, &database).unwrap();
    assert!(similarities.is_empty());

    let products = backend.batch_dot_product(&query, &database).unwrap();
    assert!(products.is_empty());
}

/// Test single vector database
#[test]
fn test_single_vector_database() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);

    let query = vec![1.0, 0.0, 0.0];
    let database = vec![vec![1.0, 0.0, 0.0]];

    let distances = backend.batch_euclidean_distance(&query, &database).unwrap();
    assert_eq!(distances.len(), 1);
    assert!((distances[0] - 0.0).abs() < 0.001);
}

/// Test high dimensional vectors
#[test]
fn test_high_dimensional_vectors() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);

    let dim = 384; // Common embedding dimension
    let query: Vec<f32> = (0..dim).map(|i| (i as f32) / (dim as f32)).collect();
    let database = vec![
        query.clone(),
        (0..dim)
            .map(|i| (i as f32) / (dim as f32) + 0.1)
            .collect(),
    ];

    let distances = backend.batch_euclidean_distance(&query, &database).unwrap();
    assert_eq!(distances.len(), 2);
    assert!((distances[0] - 0.0).abs() < 0.001); // Same vector
    assert!(distances[1] > 0.0); // Different vector
}

/// Test matrix multiplication
#[test]
fn test_matrix_multiply() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);

    // 2x3 matrix
    let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

    // 3x2 matrix
    let b = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];

    let c = backend.matrix_multiply(&a, &b).unwrap();

    // Result should be 2x2
    assert_eq!(c.len(), 2);
    assert_eq!(c[0].len(), 2);

    // C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    assert!((c[0][0] - 58.0).abs() < 0.001);

    // C[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    assert!((c[0][1] - 64.0).abs() < 0.001);

    // C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    assert!((c[1][0] - 139.0).abs() < 0.001);

    // C[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    assert!((c[1][1] - 154.0).abs() < 0.001);
}

/// Test GPU backend enum serialization
#[test]
fn test_gpu_backend_serialization() {
    let backends = vec![
        GpuBackend::Cpu,
        GpuBackend::Cuda,
        GpuBackend::Metal,
        GpuBackend::WebGpu,
    ];

    for backend in backends {
        let json = serde_json::to_string(&backend).unwrap();
        let deserialized: GpuBackend = serde_json::from_str(&json).unwrap();
        assert_eq!(backend, deserialized);
    }
}

/// Test device info structure
#[test]
fn test_device_info() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);
    let info = backend.device_info();

    assert!(!info.name.is_empty());
    // CPU backend returns 0 for memory (not applicable)
    assert_eq!(info.backend, GpuBackend::Cpu);
}

/// Test empty matrix multiplication
#[test]
fn test_empty_matrix_multiply() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);

    let empty: Vec<Vec<f32>> = vec![];
    let result = backend.matrix_multiply(&empty, &empty).unwrap();
    assert!(result.is_empty());
}

/// Test zero vector normalization
#[test]
fn test_zero_vector_normalize() {
    let config = GpuConfig::default();
    let backend = CpuBackend::new(&config);

    let vectors = vec![vec![0.0, 0.0, 0.0]]; // Zero vector

    let normalized = backend.batch_normalize(&vectors).unwrap();
    assert_eq!(normalized.len(), 1);
    // Zero vector should remain zero (no division by zero)
    assert!((normalized[0][0]).abs() < 0.001);
    assert!((normalized[0][1]).abs() < 0.001);
    assert!((normalized[0][2]).abs() < 0.001);
}

/// Test GpuBackend variants
#[test]
fn test_gpu_backend_variants() {
    let backends = vec![
        GpuBackend::Cpu,
        GpuBackend::Cuda,
        GpuBackend::Metal,
        GpuBackend::WebGpu,
    ];

    for backend in backends {
        match backend {
            GpuBackend::Cpu | GpuBackend::Cuda | GpuBackend::Metal | GpuBackend::WebGpu => {}
        }
    }
}
