use std::collections::HashMap;
use vecstore::{FilterExpr, FilterOp, Metadata, Query, VecStore};

#[test]
fn test_empty_store_query() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = VecStore::open(temp_dir.path()).unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 5,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_single_vector() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("only".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 5,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "only");
}

#[test]
fn test_k_larger_than_store_size() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    for i in 0..3 {
        store
            .upsert(format!("vec{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 10, // More than we have
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 3); // Should return all 3
}

#[test]
fn test_zero_k() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("vec1".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 0,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_high_dimensional_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let dim = 1024;
    let vec1: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
    let vec2: Vec<f32> = (0..dim).map(|i| (i + 1) as f32 / dim as f32).collect();

    store
        .upsert("vec1".into(), vec1.clone(), meta.clone())
        .unwrap();
    store.upsert("vec2".into(), vec2.clone(), meta).unwrap();

    let query = Query {
        vector: vec1,
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "vec1");
}

#[test]
fn test_duplicate_id_upsert() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let mut meta1 = Metadata {
        fields: HashMap::new(),
    };
    meta1.fields.insert("version".into(), serde_json::json!(1));

    let mut meta2 = Metadata {
        fields: HashMap::new(),
    };
    meta2.fields.insert("version".into(), serde_json::json!(2));

    // First insert
    store
        .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta1)
        .unwrap();
    assert_eq!(store.count(), 1);

    // Second insert with same ID should update
    store
        .upsert("doc1".into(), vec![0.0, 1.0, 0.0], meta2)
        .unwrap();
    assert_eq!(store.count(), 1);

    let query = Query {
        vector: vec![0.0, 1.0, 0.0],
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "doc1");
    assert_eq!(
        results[0].metadata.fields.get("version").unwrap(),
        &serde_json::json!(2)
    );
}

#[test]
fn test_remove_nonexistent() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let result = store.remove("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_all_vectors_filtered_out() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let mut meta = Metadata {
        fields: HashMap::new(),
    };
    meta.fields
        .insert("category".into(), serde_json::json!("A"));

    for i in 0..5 {
        store
            .upsert(format!("vec{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    let query = Query {
        vector: vec![2.0, 0.0, 0.0],
        k: 10,
        filter: Some(FilterExpr::Cmp {
            field: "category".into(),
            op: FilterOp::Eq,
            value: serde_json::json!("B"), // None match
        }),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_very_large_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let mut meta = Metadata {
        fields: HashMap::new(),
    };

    // Add lots of metadata fields
    for i in 0..100 {
        meta.fields
            .insert(format!("field_{}", i), serde_json::json!(i));
    }
    meta.fields.insert(
        "text".into(),
        serde_json::json!(
            "This is a very long text field with lots of content to test large metadata storage"
        ),
    );

    store
        .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].metadata.fields.len(), 101);
}

#[test]
fn test_zero_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("zero".into(), vec![0.0, 0.0, 0.0], meta.clone())
        .unwrap();

    let query = Query {
        vector: vec![0.0, 0.0, 0.0],
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_negative_values_in_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("neg".into(), vec![-1.0, -2.0, -3.0], meta.clone())
        .unwrap();
    store
        .upsert("pos".into(), vec![1.0, 2.0, 3.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![-1.0, -2.0, -3.0],
        k: 2,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "neg");
}

#[test]
fn test_special_float_values() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Very small values
    store
        .upsert("small".into(), vec![1e-10, 1e-10, 1e-10], meta.clone())
        .unwrap();

    // Very large values
    store
        .upsert("large".into(), vec![1e10, 1e10, 1e10], meta)
        .unwrap();

    assert_eq!(store.count(), 2);
}

#[test]
fn test_unicode_in_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let mut meta = Metadata {
        fields: HashMap::new(),
    };
    meta.fields
        .insert("text".into(), serde_json::json!("Hello 世界 🌍"));
    meta.fields
        .insert("emoji".into(), serde_json::json!("🚀🔥💯"));

    store
        .upsert("unicode".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 1,
        filter: Some(FilterExpr::Cmp {
            field: "emoji".into(),
            op: FilterOp::Contains,
            value: serde_json::json!("🚀"),
        }),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_empty_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("empty_meta".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].metadata.fields.is_empty());
}
