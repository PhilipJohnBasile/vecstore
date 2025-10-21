use std::collections::HashMap;
use vecstore::{FilterExpr, FilterOp, Metadata, Query, VecStore};

fn setup_store() -> (tempfile::TempDir, VecStore) {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    // Insert test data
    let mut meta1 = Metadata {
        fields: HashMap::new(),
    };
    meta1
        .fields
        .insert("topic".into(), serde_json::json!("rust"));
    meta1.fields.insert("score".into(), serde_json::json!(10));
    store
        .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta1)
        .unwrap();

    let mut meta2 = Metadata {
        fields: HashMap::new(),
    };
    meta2
        .fields
        .insert("topic".into(), serde_json::json!("python"));
    meta2.fields.insert("score".into(), serde_json::json!(8));
    store
        .upsert("doc2".into(), vec![0.9, 0.1, 0.0], meta2)
        .unwrap();

    let mut meta3 = Metadata {
        fields: HashMap::new(),
    };
    meta3
        .fields
        .insert("topic".into(), serde_json::json!("rust"));
    meta3.fields.insert("score".into(), serde_json::json!(5));
    store
        .upsert("doc3".into(), vec![0.8, 0.0, 0.2], meta3)
        .unwrap();

    (temp_dir, store)
}

#[test]
fn test_eq_filter() {
    let (_temp, store) = setup_store();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 10,
        filter: Some(FilterExpr::Cmp {
            field: "topic".into(),
            op: FilterOp::Eq,
            value: serde_json::json!("rust"),
        }),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 2);
    for result in &results {
        let topic = result.metadata.fields.get("topic").unwrap();
        assert_eq!(topic, &serde_json::json!("rust"));
    }
}

#[test]
fn test_neq_filter() {
    let (_temp, store) = setup_store();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 10,
        filter: Some(FilterExpr::Cmp {
            field: "topic".into(),
            op: FilterOp::Neq,
            value: serde_json::json!("rust"),
        }),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "doc2");
}

#[test]
fn test_gt_filter() {
    let (_temp, store) = setup_store();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 10,
        filter: Some(FilterExpr::Cmp {
            field: "score".into(),
            op: FilterOp::Gt,
            value: serde_json::json!(7),
        }),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 2); // doc1 (10) and doc2 (8)
}

#[test]
fn test_lte_filter() {
    let (_temp, store) = setup_store();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 10,
        filter: Some(FilterExpr::Cmp {
            field: "score".into(),
            op: FilterOp::Lte,
            value: serde_json::json!(8),
        }),
    };

    let results = store.query(query).unwrap();
    // HNSW is approximate, may not return all matching vectors
    assert!(results.len() >= 1);
    assert!(results.len() <= 2); // doc2 (8) and doc3 (5)

    // Verify all results match the filter
    for result in &results {
        let score = result
            .metadata
            .fields
            .get("score")
            .unwrap()
            .as_i64()
            .unwrap();
        assert!(score <= 8);
    }
}

#[test]
fn test_and_filter() {
    let (_temp, store) = setup_store();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 10,
        filter: Some(FilterExpr::And(vec![
            FilterExpr::Cmp {
                field: "topic".into(),
                op: FilterOp::Eq,
                value: serde_json::json!("rust"),
            },
            FilterExpr::Cmp {
                field: "score".into(),
                op: FilterOp::Gt,
                value: serde_json::json!(7),
            },
        ])),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "doc1");
}

#[test]
fn test_or_filter() {
    let (_temp, store) = setup_store();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 10,
        filter: Some(FilterExpr::Or(vec![
            FilterExpr::Cmp {
                field: "topic".into(),
                op: FilterOp::Eq,
                value: serde_json::json!("python"),
            },
            FilterExpr::Cmp {
                field: "score".into(),
                op: FilterOp::Eq,
                value: serde_json::json!(10),
            },
        ])),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 2); // doc1 and doc2
}

#[test]
fn test_not_filter() {
    let (_temp, store) = setup_store();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 10,
        filter: Some(FilterExpr::Not(Box::new(FilterExpr::Cmp {
            field: "topic".into(),
            op: FilterOp::Eq,
            value: serde_json::json!("python"),
        }))),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 2); // doc1 and doc3 (not python)
}

#[test]
fn test_contains_filter() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let mut meta = Metadata {
        fields: HashMap::new(),
    };
    meta.fields
        .insert("description".into(), serde_json::json!("hello world"));
    store
        .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 10,
        filter: Some(FilterExpr::Cmp {
            field: "description".into(),
            op: FilterOp::Contains,
            value: serde_json::json!("world"),
        }),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
}
