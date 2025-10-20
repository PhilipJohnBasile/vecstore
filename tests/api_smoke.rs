use std::collections::HashMap;
use vecstore::{Metadata, Query, VecStore};

#[test]
fn test_basic_operations() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    // Initially empty
    assert_eq!(store.count(), 0);

    // Insert a vector
    let mut meta = Metadata {
        fields: HashMap::new(),
    };
    meta.fields.insert("type".into(), serde_json::json!("test"));

    store
        .upsert("vec1".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    assert_eq!(store.count(), 1);

    // Query
    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 1,
        filter: None,
    };
    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "vec1");

    // Update existing
    store
        .upsert("vec1".into(), vec![0.0, 1.0, 0.0], meta)
        .unwrap();
    assert_eq!(store.count(), 1);

    // Remove
    store.remove("vec1").unwrap();
    assert_eq!(store.count(), 0);
}

#[test]
fn test_dimension_validation() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // First insert sets dimension
    store
        .upsert("vec1".into(), vec![1.0, 2.0, 3.0], meta.clone())
        .unwrap();

    // Wrong dimension should fail
    let result = store.upsert("vec2".into(), vec![1.0, 2.0], meta);
    assert!(result.is_err());
}

#[test]
fn test_query_with_k() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert multiple vectors
    for i in 0..10 {
        let vec = vec![i as f32, 0.0, 0.0];
        store
            .upsert(format!("vec{}", i), vec, meta.clone())
            .unwrap();
    }

    // Query with k=3
    let query = Query {
        vector: vec![5.0, 0.0, 0.0],
        k: 3,
        filter: None,
    };
    let results = store.query(query).unwrap();
    assert!(results.len() <= 3);
}
