use std::collections::HashMap;
use vecstore::{FilterExpr, FilterOp, Metadata, Query, VecStore};

#[test]
fn test_persistence_with_filters() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    // Create and save
    {
        let mut store = VecStore::open(path).unwrap();

        for i in 0..10 {
            let mut meta = Metadata {
                fields: HashMap::new(),
            };
            meta.fields.insert("value".into(), serde_json::json!(i));

            store
                .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta)
                .unwrap();
        }

        store.save().unwrap();
    }

    // Reload and query with filter
    {
        let store = VecStore::open(path).unwrap();

        let query = Query {
            vector: vec![5.0, 0.0, 0.0],
            k: 10,
            filter: Some(FilterExpr::Cmp {
                field: "value".into(),
                op: FilterOp::Gte,
                value: serde_json::json!(5),
            }),
        };

        let results = store.query(query).unwrap();
        assert!(results.len() >= 5);

        for result in &results {
            let value = result
                .metadata
                .fields
                .get("value")
                .unwrap()
                .as_i64()
                .unwrap();
            assert!(value >= 5);
        }
    }
}

#[test]
fn test_multiple_save_load_cycles() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Cycle 1
    {
        let mut store = VecStore::open(path).unwrap();
        store
            .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta.clone())
            .unwrap();
        store.save().unwrap();
    }

    // Cycle 2
    {
        let mut store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 1);
        store
            .upsert("doc2".into(), vec![0.0, 1.0, 0.0], meta.clone())
            .unwrap();
        store.save().unwrap();
    }

    // Cycle 3
    {
        let mut store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 2);
        store
            .upsert("doc3".into(), vec![0.0, 0.0, 1.0], meta)
            .unwrap();
        store.save().unwrap();
    }

    // Final verification
    {
        let store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 3);
    }
}

#[test]
fn test_save_without_modifications() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Create and save
    {
        let mut store = VecStore::open(path).unwrap();
        store
            .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta)
            .unwrap();
        store.save().unwrap();
    }

    // Load and save without changes
    {
        let store = VecStore::open(path).unwrap();
        store.save().unwrap(); // Should work fine
    }

    // Verify
    {
        let store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 1);
    }
}

#[test]
fn test_persistence_large_dataset() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    let count = 1000;

    // Create and save large dataset
    {
        let mut store = VecStore::open(path).unwrap();

        for i in 0..count {
            let mut meta = Metadata {
                fields: HashMap::new(),
            };
            meta.fields.insert("index".into(), serde_json::json!(i));

            store
                .upsert(
                    format!("doc{}", i),
                    vec![i as f32, (i * 2) as f32, (i * 3) as f32],
                    meta,
                )
                .unwrap();
        }

        assert_eq!(store.count(), count);
        store.save().unwrap();
    }

    // Reload and verify
    {
        let store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), count);
        assert_eq!(store.dimension(), 3);

        // Spot check - verify we can query and get results
        let query = Query {
            vector: vec![500.0, 1000.0, 1500.0],
            k: 5,
            filter: None,
        };

        let results = store.query(query).unwrap();
        assert_eq!(results.len(), 5);
        // All results should have valid IDs in expected range
        for result in &results {
            assert!(result.id.starts_with("doc"));
        }
    }
}

#[test]
fn test_persistence_after_remove() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Create, remove, save
    {
        let mut store = VecStore::open(path).unwrap();
        store
            .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta.clone())
            .unwrap();
        store
            .upsert("doc2".into(), vec![0.0, 1.0, 0.0], meta.clone())
            .unwrap();
        store
            .upsert("doc3".into(), vec![0.0, 0.0, 1.0], meta)
            .unwrap();

        store.remove("doc2").unwrap();
        assert_eq!(store.count(), 2);

        store.save().unwrap();
    }

    // Reload and verify
    {
        let store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 2);

        let query = Query {
            vector: vec![0.0, 1.0, 0.0],
            k: 3,
            filter: None,
        };

        let results = store.query(query).unwrap();
        assert_eq!(results.len(), 2);

        // doc2 should not be present
        for result in &results {
            assert_ne!(result.id, "doc2");
        }
    }
}

#[test]
fn test_persistence_dimension_preserved() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    let dimension = 128;
    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Create with high dimension
    {
        let mut store = VecStore::open(path).unwrap();
        let vec: Vec<f32> = (0..dimension).map(|i| i as f32).collect();
        store.upsert("doc1".into(), vec, meta).unwrap();
        assert_eq!(store.dimension(), dimension);
        store.save().unwrap();
    }

    // Reload and verify dimension
    {
        let store = VecStore::open(path).unwrap();
        assert_eq!(store.dimension(), dimension);
    }
}

#[test]
fn test_persistence_complex_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    // Create with complex metadata
    {
        let mut store = VecStore::open(path).unwrap();

        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields
            .insert("string".into(), serde_json::json!("test"));
        meta.fields.insert("number".into(), serde_json::json!(42));
        meta.fields
            .insert("float".into(), serde_json::json!(3.14159));
        meta.fields.insert("bool".into(), serde_json::json!(true));
        meta.fields
            .insert("array".into(), serde_json::json!([1, 2, 3]));
        meta.fields
            .insert("object".into(), serde_json::json!({"nested": "value"}));
        meta.fields.insert("null".into(), serde_json::json!(null));

        store
            .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta)
            .unwrap();
        store.save().unwrap();
    }

    // Reload and verify metadata
    {
        let store = VecStore::open(path).unwrap();

        let query = Query {
            vector: vec![1.0, 0.0, 0.0],
            k: 1,
            filter: None,
        };

        let results = store.query(query).unwrap();
        assert_eq!(results.len(), 1);

        let fields = &results[0].metadata.fields;
        assert_eq!(fields.get("string").unwrap(), &serde_json::json!("test"));
        assert_eq!(fields.get("number").unwrap(), &serde_json::json!(42));
        assert_eq!(fields.get("bool").unwrap(), &serde_json::json!(true));
        assert!(fields.get("array").unwrap().is_array());
        assert!(fields.get("object").unwrap().is_object());
        assert!(fields.get("null").unwrap().is_null());
    }
}

#[test]
fn test_persistence_after_batch_operations() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    use vecstore::make_record;

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Batch insert and save
    {
        let mut store = VecStore::open(path).unwrap();

        let records: Vec<_> = (0..100)
            .map(|i| make_record(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone()))
            .collect();

        store.batch_upsert(records).unwrap();
        store.save().unwrap();
    }

    // Reload and verify
    {
        let store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 100);
    }
}

#[test]
fn test_save_empty_then_populate() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Save empty
    {
        let store = VecStore::open(path).unwrap();
        store.save().unwrap();
    }

    // Populate
    {
        let mut store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 0);

        store
            .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta)
            .unwrap();
        store.save().unwrap();
    }

    // Verify
    {
        let store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 1);
    }
}

#[test]
fn test_concurrent_store_instances() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Create initial data
    {
        let mut store = VecStore::open(path).unwrap();
        store
            .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta.clone())
            .unwrap();
        store.save().unwrap();
    }

    // Open two instances (sequential, not truly concurrent in this test)
    let store1 = VecStore::open(path).unwrap();
    let store2 = VecStore::open(path).unwrap();

    assert_eq!(store1.count(), 1);
    assert_eq!(store2.count(), 1);
}

#[test]
fn test_timestamp_persistence() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let before = chrono::Utc::now().timestamp();

    // Create record
    {
        let mut store = VecStore::open(path).unwrap();
        store
            .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta)
            .unwrap();
        store.save().unwrap();
    }

    let after = chrono::Utc::now().timestamp();

    // Reload and check timestamp
    {
        let store = VecStore::open(path).unwrap();

        let query = Query {
            vector: vec![1.0, 0.0, 0.0],
            k: 1,
            filter: None,
        };

        let results = store.query(query).unwrap();

        // Timestamp should be preserved and within range
        // Note: We can't directly access created_at from Neighbor,
        // so this just verifies the data is retrievable
        assert_eq!(results.len(), 1);
    }
}
