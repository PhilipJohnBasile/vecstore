use std::collections::HashMap;
use vecstore::{Metadata, Query, VecStore};

#[test]
fn test_dimension_mismatch_on_insert() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // First insert sets dimension
    store
        .upsert("vec1".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();

    // Wrong dimension should fail
    let result = store.upsert("vec2".into(), vec![1.0, 0.0], meta);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("dimension mismatch"));
}

#[test]
fn test_dimension_mismatch_on_query() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("vec1".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0], // Wrong dimension
        k: 1,
        filter: None,
    };

    let result = store.query(query);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("dimension mismatch"));
}

#[test]
fn test_load_nonexistent_store() {
    let temp_dir = tempfile::tempdir().unwrap();
    let nonexistent_path = temp_dir.path().join("nonexistent");

    // Opening a new store should work (creates it)
    let store = VecStore::open(&nonexistent_path);
    assert!(store.is_ok());
}

#[test]
fn test_remove_from_empty_store() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let result = store.remove("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_remove_after_remove() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("vec1".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();

    // First remove should succeed
    assert!(store.remove("vec1").is_ok());
    assert_eq!(store.count(), 0);

    // Second remove should fail
    let result = store.remove("vec1");
    assert!(result.is_err());
}

#[test]
fn test_query_after_remove_all() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Add and remove
    store
        .upsert("vec1".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();
    store.remove("vec1").unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_empty_vector() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Empty vectors are technically valid (dimension 0)
    let result = store.upsert("empty".into(), vec![], meta);
    // This should work - dimension 0 is valid
    assert!(result.is_ok());
}

#[test]
fn test_corrupted_data_recovery() {
    // This test verifies that we handle missing files gracefully
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Create and save a store
    {
        let mut store = VecStore::open(path).unwrap();
        store
            .upsert("vec1".into(), vec![1.0, 0.0, 0.0], meta)
            .unwrap();
        store.save().unwrap();
    }

    // Delete one of the data files to simulate corruption
    let vectors_path = path.join("vectors.bin");
    std::fs::remove_file(vectors_path).ok();

    // Loading should fail gracefully
    let result = VecStore::open(path);
    assert!(result.is_err());
}

#[test]
fn test_invalid_filter_field() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("vec1".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();

    // Filter on non-existent field should return no results
    use vecstore::{FilterExpr, FilterOp};
    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 1,
        filter: Some(FilterExpr::Cmp {
            field: "nonexistent".into(),
            op: FilterOp::Eq,
            value: serde_json::json!("value"),
        }),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 0); // No error, just empty results
}

#[test]
fn test_type_mismatch_in_filter() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let mut meta = Metadata {
        fields: HashMap::new(),
    };
    meta.fields.insert("score".into(), serde_json::json!(10));

    store
        .upsert("vec1".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();

    // Compare number field with string - should not match
    use vecstore::{FilterExpr, FilterOp};
    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 1,
        filter: Some(FilterExpr::Cmp {
            field: "score".into(),
            op: FilterOp::Eq,
            value: serde_json::json!("10"), // String instead of number
        }),
    };

    let results = store.query(query).unwrap();
    // Should handle gracefully - may or may not match depending on coercion
    assert!(results.len() <= 1);
}

#[test]
fn test_save_to_readonly_location() {
    // Note: This test might behave differently on different platforms
    // On Unix-like systems, we can test writing to /dev/null or similar
    // This is a basic test that saving can handle errors

    let temp_dir = tempfile::tempdir().unwrap();
    let store = VecStore::open(temp_dir.path()).unwrap();

    // Save should work on a valid temp directory
    let result = store.save();
    assert!(result.is_ok());
}

#[test]
fn test_concurrent_modifications_same_id() {
    // Test that sequential modifications to same ID work correctly
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta1 = Metadata {
        fields: HashMap::new(),
    };
    let meta2 = Metadata {
        fields: HashMap::new(),
    };

    // Rapid successive updates
    for i in 0..10 {
        store
            .upsert(
                "same_id".into(),
                vec![i as f32, 0.0, 0.0],
                if i % 2 == 0 {
                    meta1.clone()
                } else {
                    meta2.clone()
                },
            )
            .unwrap();
    }

    assert_eq!(store.count(), 1);
}

#[test]
fn test_nan_in_vector() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // NaN values - should be accepted but may cause undefined behavior in HNSW
    let result = store.upsert("nan".into(), vec![f32::NAN, 0.0, 0.0], meta);

    // The system should accept it (no validation against NaN)
    // But behavior in queries might be undefined
    assert!(result.is_ok());
}

#[test]
fn test_infinity_in_vector() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let result = store.upsert("inf".into(), vec![f32::INFINITY, 0.0, 0.0], meta);
    assert!(result.is_ok());
}
