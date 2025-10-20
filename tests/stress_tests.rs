// Comprehensive stress and concurrency tests
// Tests high load, concurrent operations, large datasets, and edge cases

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use vecstore::{Metadata, Query, VecStore};

#[test]
fn test_large_dataset_insertion() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert 10,000 vectors
    for i in 0..10_000 {
        let vector: Vec<f32> = vec![
            (i as f32 * 0.001).sin(),
            (i as f32 * 0.002).cos(),
            (i as f32 * 0.003).sin(),
        ];

        store
            .upsert(format!("doc{}", i), vector, meta.clone())
            .unwrap();
    }

    assert_eq!(store.count(), 10_000);
}

#[test]
fn test_large_dataset_query_performance() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert 5,000 vectors
    for i in 0..5_000 {
        let vector: Vec<f32> = vec![
            (i as f32 * 0.01).sin(),
            (i as f32 * 0.02).cos(),
            (i as f32 * 0.03).sin(),
        ];

        store
            .upsert(format!("doc{}", i), vector, meta.clone())
            .unwrap();
    }

    // Query should complete in reasonable time
    let query = Query {
        vector: vec![0.5, 0.5, 0.5],
        k: 100,
        filter: None,
    };

    let start = std::time::Instant::now();
    let results = store.query(query).unwrap();
    let duration = start.elapsed();

    assert!(results.len() <= 100);
    assert!(
        duration.as_secs() < 5,
        "Query took too long: {:?}",
        duration
    );
}

#[test]
fn test_concurrent_reads() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert initial data
    for i in 0..100 {
        store
            .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    let store = Arc::new(Mutex::new(store));
    let mut handles = vec![];

    // Spawn multiple reader threads
    for thread_id in 0..10 {
        let store_clone = Arc::clone(&store);

        let handle = thread::spawn(move || {
            for _ in 0..20 {
                let query = Query {
                    vector: vec![thread_id as f32, 0.0, 0.0],
                    k: 10,
                    filter: None,
                };

                let store = store_clone.lock().unwrap();
                let results = store.query(query);
                drop(store);

                assert!(results.is_ok());
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_writes() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = VecStore::open(temp_dir.path()).unwrap();
    let store = Arc::new(Mutex::new(store));

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let mut handles = vec![];

    // Spawn multiple writer threads
    for thread_id in 0..5 {
        let store_clone = Arc::clone(&store);
        let meta_clone = meta.clone();

        let handle = thread::spawn(move || {
            for i in 0..20 {
                let id = format!("thread{}_doc{}", thread_id, i);
                let vector = vec![thread_id as f32, i as f32, 0.0];

                let mut store = store_clone.lock().unwrap();
                let result = store.upsert(id, vector, meta_clone.clone());
                drop(store);

                assert!(result.is_ok());
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let store = store.lock().unwrap();
    assert_eq!(store.count(), 100); // 5 threads * 20 docs each
}

#[test]
fn test_concurrent_mixed_operations() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = VecStore::open(temp_dir.path()).unwrap();
    let store = Arc::new(Mutex::new(store));

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Pre-populate
    {
        let mut store = store.lock().unwrap();
        for i in 0..50 {
            store
                .upsert(format!("init{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
                .unwrap();
        }
    }

    let mut handles = vec![];

    // Writers
    for thread_id in 0..3 {
        let store_clone = Arc::clone(&store);
        let meta_clone = meta.clone();

        let handle = thread::spawn(move || {
            for i in 0..10 {
                let mut store = store_clone.lock().unwrap();
                store
                    .upsert(
                        format!("write{}_{}", thread_id, i),
                        vec![thread_id as f32, i as f32, 0.0],
                        meta_clone.clone(),
                    )
                    .unwrap();
            }
        });

        handles.push(handle);
    }

    // Readers
    for thread_id in 0..3 {
        let store_clone = Arc::clone(&store);

        let handle = thread::spawn(move || {
            for _ in 0..10 {
                let store = store_clone.lock().unwrap();
                let query = Query {
                    vector: vec![thread_id as f32, 0.0, 0.0],
                    k: 10,
                    filter: None,
                };
                let _ = store.query(query);
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_batch_operations_large_scale() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let mut operations = vec![];

    // Create 1,000 batch operations
    for i in 0..1_000 {
        let id = format!("batch{}", i);
        let vector = vec![i as f32 * 0.1, (i * 2) as f32 * 0.1, (i * 3) as f32 * 0.1];

        operations.push(vecstore::BatchOperation::Upsert {
            id,
            vector,
            metadata: meta.clone(),
        });
    }

    let result = store.batch_execute(operations);

    assert!(result.is_ok());
    assert_eq!(store.count(), 1_000);
}

#[test]
fn test_high_dimensional_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let dimension = 1536; // OpenAI embedding size

    // Insert high-dimensional vectors
    for i in 0..100 {
        let vector: Vec<f32> = (0..dimension)
            .map(|j| ((i + j) as f32 * 0.001).sin())
            .collect();

        store
            .upsert(format!("doc{}", i), vector, meta.clone())
            .unwrap();
    }

    // Query
    let query_vec: Vec<f32> = (0..dimension).map(|i| (i as f32 * 0.001).cos()).collect();

    let query = Query {
        vector: query_vec,
        k: 10,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert!(results.len() <= 10);
}

#[test]
fn test_rapid_updates() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Rapidly update same document
    for i in 0..1_000 {
        store
            .upsert("doc1".into(), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    // Should still have only one document
    assert_eq!(store.count(), 1);
}

#[test]
fn test_many_deletes() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert many documents
    for i in 0..1_000 {
        store
            .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    assert_eq!(store.count(), 1_000);

    // Delete half
    for i in 0..500 {
        store.remove(&format!("doc{}", i)).unwrap();
    }

    assert_eq!(store.count(), 500);
}

#[test]
fn test_query_with_large_k() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert 100 vectors
    for i in 0..100 {
        store
            .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    // Query with k larger than dataset
    let query = Query {
        vector: vec![50.0, 0.0, 0.0],
        k: 1000,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // Should return all available vectors
    assert!(results.len() <= 100);
}

#[test]
fn test_extreme_vector_values() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Very large values
    store
        .upsert("large".into(), vec![1e6, 1e6, 1e6], meta.clone())
        .unwrap();

    // Very small values
    store
        .upsert("small".into(), vec![1e-6, 1e-6, 1e-6], meta.clone())
        .unwrap();

    // Mixed
    store
        .upsert("mixed".into(), vec![1e6, 1e-6, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 1.0, 1.0],
        k: 3,
        filter: None,
    };

    let results = store.query(query);
    assert!(results.is_ok());
}

#[test]
fn test_persistence_under_load() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().to_path_buf();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert data
    {
        let mut store = VecStore::open(&path).unwrap();

        for i in 0..500 {
            store
                .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
                .unwrap();
        }

        // Explicitly save before dropping
        store.save().unwrap();
    }

    // Reopen and verify
    {
        let store = VecStore::open(&path).unwrap();
        assert_eq!(store.count(), 500);
    }

    // Add more data
    {
        let mut store = VecStore::open(&path).unwrap();

        for i in 500..1_000 {
            store
                .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
                .unwrap();
        }

        // Explicitly save before dropping
        store.save().unwrap();
    }

    // Final verification
    {
        let store = VecStore::open(&path).unwrap();
        assert_eq!(store.count(), 1_000);
    }
}

#[test]
fn test_memory_efficiency_many_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert many small vectors
    for i in 0..5_000 {
        store
            .upsert(format!("doc{}", i), vec![i as f32 % 100.0], meta.clone())
            .unwrap();
    }

    assert_eq!(store.count(), 5_000);

    // Query should still work efficiently
    let query = Query {
        vector: vec![50.0],
        k: 10,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert!(results.len() <= 10);
}

#[test]
fn test_sequential_id_generation() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Use sequential integer IDs
    for i in 0..1_000 {
        store
            .upsert(i.to_string(), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    assert_eq!(store.count(), 1_000);
}

#[test]
fn test_very_long_ids() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Very long ID (1000 characters)
    let long_id = "a".repeat(1000);

    store
        .upsert(long_id.clone(), vec![1.0, 2.0, 3.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 2.0, 3.0],
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, long_id);
}

#[test]
fn test_stress_test_mixed_workload() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert
    for i in 0..100 {
        store
            .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    // Query
    for _ in 0..50 {
        let query = Query {
            vector: vec![50.0, 0.0, 0.0],
            k: 10,
            filter: None,
        };
        store.query(query).unwrap();
    }

    // Update
    for i in 0..50 {
        store
            .upsert(
                format!("doc{}", i),
                vec![(i + 100) as f32, 0.0, 0.0],
                meta.clone(),
            )
            .unwrap();
    }

    // Delete
    for i in 50..75 {
        store.remove(&format!("doc{}", i)).unwrap();
    }

    // Query again
    let query = Query {
        vector: vec![25.0, 0.0, 0.0],
        k: 20,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert!(results.len() <= 20);
}

#[test]
fn test_sparse_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Vectors with mostly zeros
    for i in 0..100 {
        let mut vector = vec![0.0; 100];
        vector[i] = 1.0; // Only one non-zero element

        store
            .upsert(format!("sparse{}", i), vector, meta.clone())
            .unwrap();
    }

    assert_eq!(store.count(), 100);
}

#[test]
fn test_identical_vectors_different_ids() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let same_vector = vec![1.0, 2.0, 3.0];

    // Insert same vector with different IDs
    for i in 0..100 {
        store
            .upsert(format!("doc{}", i), same_vector.clone(), meta.clone())
            .unwrap();
    }

    assert_eq!(store.count(), 100);

    let query = Query {
        vector: same_vector,
        k: 10,
        filter: None,
    };

    let results = store.query(query).unwrap();
    // Should return 10 results, all with high similarity
    assert_eq!(results.len(), 10);
    for result in results {
        assert!(result.score > 0.99);
    }
}
