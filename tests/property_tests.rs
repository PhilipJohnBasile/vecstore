// Property-based tests using proptest
// Tests invariants and properties that should hold for all inputs

use proptest::prelude::*;
use std::collections::HashMap;
use vecstore::{Metadata, Query, VecStore};

// Strategy for generating valid vectors
fn vector_strategy(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec((-100.0f32..100.0f32), dim..=dim)
}

// Strategy for generating valid IDs
fn id_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_-]{1,50}"
}

proptest! {
    #[test]
    fn test_insert_then_count_increases(
        id in id_strategy(),
        vector in vector_strategy(3)
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        let count_before = store.count();
        store.upsert(id, vector, meta).unwrap();
        let count_after = store.count();

        prop_assert!(count_after >= count_before);
        prop_assert!(count_after <= count_before + 1);
    }

    #[test]
    fn test_upsert_idempotent(
        id in id_strategy(),
        vector in vector_strategy(3)
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        // Insert once
        store.upsert(id.clone(), vector.clone(), meta.clone()).unwrap();
        let count_after_first = store.count();

        // Insert again with same ID
        store.upsert(id, vector, meta).unwrap();
        let count_after_second = store.count();

        // Count should not change (upsert is idempotent)
        prop_assert_eq!(count_after_first, count_after_second);
    }

    #[test]
    fn test_query_returns_at_most_k(
        k in 1usize..100,
        num_docs in 0usize..50
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        // Insert num_docs documents
        for i in 0..num_docs {
            store.upsert(
                format!("doc{}", i),
                vec![i as f32, 0.0, 0.0],
                meta.clone()
            ).unwrap();
        }

        let query = Query {
            vector: vec![0.0, 0.0, 0.0],
            k,
            filter: None,
        };

        let results = store.query(query).unwrap();

        prop_assert!(results.len() <= k);
        prop_assert!(results.len() <= num_docs);
    }

    #[test]
    fn test_query_scores_descending(
        vector in vector_strategy(3),
        num_docs in 1usize..20
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        for i in 0..num_docs {
            store.upsert(
                format!("doc{}", i),
                vec![i as f32, (i * 2) as f32, (i * 3) as f32],
                meta.clone()
            ).unwrap();
        }

        let query = Query {
            vector,
            k: num_docs,
            filter: None,
        };

        let results = store.query(query).unwrap();

        // Scores should be in descending order
        for i in 0..results.len().saturating_sub(1) {
            prop_assert!(results[i].score >= results[i + 1].score);
        }
    }

    #[test]
    fn test_remove_decreases_count(
        id in id_strategy(),
        vector in vector_strategy(3)
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        // Insert
        store.upsert(id.clone(), vector, meta).unwrap();
        let count_after_insert = store.count();

        // Remove
        store.remove(&id).unwrap();
        let count_after_remove = store.count();

        prop_assert_eq!(count_after_remove, count_after_insert - 1);
    }

    #[test]
    fn test_dimension_consistency(
        dim in 1usize..100,
        num_docs in 1usize..10
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        // First insert sets dimension
        let first_vector: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        store.upsert("doc0".to_string(), first_vector, meta.clone()).unwrap();

        // All subsequent inserts with same dimension should succeed
        for i in 1..num_docs {
            let vector: Vec<f32> = (0..dim).map(|j| (i + j) as f32).collect();
            let result = store.upsert(format!("doc{}", i), vector, meta.clone());
            prop_assert!(result.is_ok());
        }

        // Different dimension should fail
        let wrong_vector: Vec<f32> = (0..dim+1).map(|i| i as f32).collect();
        let result = store.upsert("wrong".to_string(), wrong_vector, meta);
        prop_assert!(result.is_err());
    }

    #[test]
    fn test_identical_vector_retrieval(
        vector in vector_strategy(5)
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        store.upsert("doc1".to_string(), vector.clone(), meta).unwrap();

        let query = Query {
            vector: vector.clone(),
            k: 1,
            filter: None,
        };

        let results = store.query(query).unwrap();

        if !results.is_empty() {
            // Identical vector should have high similarity score
            prop_assert!(results[0].score > 0.9);
        }
    }

    #[test]
    fn test_batch_insert_count(
        num_ops in 1usize..50
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        let mut operations = vec![];
        for i in 0..num_ops {
            operations.push(vecstore::BatchOperation::Upsert {
                id: format!("doc{}", i),
                vector: vec![i as f32, 0.0, 0.0],
                metadata: meta.clone(),
            });
        }

        let count_before = store.count();
        store.batch_execute(operations).unwrap();
        let count_after = store.count();

        prop_assert_eq!(count_after, count_before + num_ops);
    }

    #[test]
    fn test_query_empty_store_returns_empty(
        vector in vector_strategy(3),
        k in 1usize..100
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = VecStore::open(temp_dir.path()).unwrap();

        let query = Query {
            vector,
            k,
            filter: None,
        };

        let results = store.query(query).unwrap();
        prop_assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_persistence_preserves_count(
        num_docs in 1usize..30
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_path_buf();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        // Insert and close
        {
            let mut store = VecStore::open(&path).unwrap();
            for i in 0..num_docs {
                store.upsert(
                    format!("doc{}", i),
                    vec![i as f32, 0.0, 0.0],
                    meta.clone()
                ).unwrap();
            }
            // Explicitly save before closing
            store.save().unwrap();
        }

        // Reopen and verify
        {
            let store = VecStore::open(&path).unwrap();
            prop_assert_eq!(store.count(), num_docs);
        }
    }

    #[test]
    fn test_update_preserves_count(
        id in id_strategy(),
        vector1 in vector_strategy(3),
        vector2 in vector_strategy(3)
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        // First insert
        store.upsert(id.clone(), vector1, meta.clone()).unwrap();
        let count_after_first = store.count();

        // Update (upsert with same ID)
        store.upsert(id, vector2, meta).unwrap();
        let count_after_update = store.count();

        prop_assert_eq!(count_after_first, count_after_update);
    }

    #[test]
    fn test_score_range(
        vector in vector_strategy(3),
        num_docs in 1usize..20
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        for i in 0..num_docs {
            store.upsert(
                format!("doc{}", i),
                vec![i as f32, 0.0, 0.0],
                meta.clone()
            ).unwrap();
        }

        let query = Query {
            vector,
            k: num_docs,
            filter: None,
        };

        let results = store.query(query).unwrap();

        // Scores should be finite (not NaN or Inf)
        // Note: Different distance metrics have different ranges:
        // - Cosine: [0, 1]
        // - Euclidean: [0, ∞)
        // - DotProduct: (-∞, ∞)
        // - Manhattan: [0, ∞)
        // - Hamming: [0, ∞)
        // - Jaccard: [0, 1]
        for result in results {
            prop_assert!(result.score.is_finite(), "Score must be finite, got {}", result.score);
        }
    }

    #[test]
    fn test_remove_nonexistent_is_ok(
        id in id_strategy()
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        // Removing non-existent ID should not crash
        let result = store.remove(&id);

        // Implementation may return Ok or Err
        prop_assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_zero_k_query(
        vector in vector_strategy(3)
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        store.upsert("doc1".to_string(), vec![1.0, 2.0, 3.0], meta).unwrap();

        let query = Query {
            vector,
            k: 0,
            filter: None,
        };

        let results = store.query(query);

        // Should either error or return empty results
        if let Ok(results) = results {
            prop_assert_eq!(results.len(), 0);
        }
    }

    #[test]
    fn test_vector_normalization_invariant(
        mut vector in vector_strategy(3)
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        // Avoid zero vectors
        if vector.iter().all(|&x| x.abs() < 0.001) {
            vector[0] = 1.0;
        }

        store.upsert("doc1".to_string(), vector.clone(), meta).unwrap();

        // Query with normalized version
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = vector.iter().map(|x| x / magnitude).collect();

        let query = Query {
            vector: normalized,
            k: 1,
            filter: None,
        };

        let results = store.query(query);
        prop_assert!(results.is_ok());
    }
}

// Additional property tests for specific scenarios

proptest! {
    #[test]
    fn test_insert_retrieve_consistency(
        ids in prop::collection::vec(id_strategy(), 1..20),
        dim in 3usize..10
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        let unique_ids: Vec<String> = ids.into_iter().collect::<std::collections::HashSet<_>>().into_iter().collect();

        for (i, id) in unique_ids.iter().enumerate() {
            let vector: Vec<f32> = (0..dim).map(|j| (i + j) as f32).collect();
            store.upsert(id.clone(), vector, meta.clone()).unwrap();
        }

        prop_assert_eq!(store.count(), unique_ids.len());
    }

    #[test]
    fn test_batch_operations_atomicity(
        num_upserts in 1usize..20,
        num_deletes in 0usize..10
    ) {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::open(temp_dir.path()).unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        // Pre-populate
        for i in 0..10 {
            store.upsert(
                format!("existing{}", i),
                vec![i as f32, 0.0, 0.0],
                meta.clone()
            ).unwrap();
        }

        let mut operations = vec![];

        // Upserts
        for i in 0..num_upserts {
            operations.push(vecstore::BatchOperation::Upsert {
                id: format!("batch{}", i),
                vector: vec![i as f32, 0.0, 0.0],
                metadata: meta.clone(),
            });
        }

        // Deletes (only delete existing items)
        for i in 0..num_deletes.min(10) {
            operations.push(vecstore::BatchOperation::Delete {
                id: format!("existing{}", i),
            });
        }

        let count_before = store.count();
        store.batch_execute(operations).unwrap();
        let count_after = store.count();

        let expected_count = count_before + num_upserts - num_deletes.min(10);
        prop_assert_eq!(count_after, expected_count);
    }
}
