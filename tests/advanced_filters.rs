use std::collections::HashMap;
use vecstore::{FilterExpr, FilterOp, Metadata, Query, VecStore};

fn setup_filtered_store() -> (tempfile::TempDir, VecStore) {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    // Create diverse metadata for testing
    for i in 0..20 {
        let mut meta = Metadata {
            fields: HashMap::new(),
        };

        meta.fields.insert("id_num".into(), serde_json::json!(i));
        meta.fields.insert(
            "category".into(),
            serde_json::json!(if i % 3 == 0 {
                "A"
            } else if i % 3 == 1 {
                "B"
            } else {
                "C"
            }),
        );
        meta.fields
            .insert("score".into(), serde_json::json!(i * 10));
        meta.fields
            .insert("active".into(), serde_json::json!(i % 2 == 0));
        meta.fields.insert(
            "tags".into(),
            serde_json::json!(vec![format!("tag{}", i % 5), format!("tag{}", (i + 1) % 5)]),
        );
        meta.fields.insert(
            "description".into(),
            serde_json::json!(format!("Document number {}", i)),
        );

        store
            .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta)
            .unwrap();
    }

    (temp_dir, store)
}

#[test]
fn test_complex_and_or_filter() {
    let (_temp, store) = setup_filtered_store();

    // (category = "A" OR category = "B") AND score > 50
    let filter = FilterExpr::And(vec![
        FilterExpr::Or(vec![
            FilterExpr::Cmp {
                field: "category".into(),
                op: FilterOp::Eq,
                value: serde_json::json!("A"),
            },
            FilterExpr::Cmp {
                field: "category".into(),
                op: FilterOp::Eq,
                value: serde_json::json!("B"),
            },
        ]),
        FilterExpr::Cmp {
            field: "score".into(),
            op: FilterOp::Gt,
            value: serde_json::json!(50),
        },
    ]);

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();

    // Verify all results match the filter
    for result in &results {
        let category = result.metadata.fields.get("category").unwrap();
        let score = result
            .metadata
            .fields
            .get("score")
            .unwrap()
            .as_i64()
            .unwrap();

        let cat_match = category == &serde_json::json!("A") || category == &serde_json::json!("B");
        let score_match = score > 50;

        assert!(cat_match && score_match);
    }
}

#[test]
fn test_nested_not_filters() {
    let (_temp, store) = setup_filtered_store();

    // NOT (category = "A" AND score < 100)
    let filter = FilterExpr::Not(Box::new(FilterExpr::And(vec![
        FilterExpr::Cmp {
            field: "category".into(),
            op: FilterOp::Eq,
            value: serde_json::json!("A"),
        },
        FilterExpr::Cmp {
            field: "score".into(),
            op: FilterOp::Lt,
            value: serde_json::json!(100),
        },
    ])));

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();

    // Verify results: should exclude docs where (category=A AND score<100)
    for result in &results {
        let category = result.metadata.fields.get("category").unwrap();
        let score = result
            .metadata
            .fields
            .get("score")
            .unwrap()
            .as_i64()
            .unwrap();

        let excluded = category == &serde_json::json!("A") && score < 100;
        assert!(!excluded);
    }
}

#[test]
fn test_multiple_or_conditions() {
    let (_temp, store) = setup_filtered_store();

    let filter = FilterExpr::Or(vec![
        FilterExpr::Cmp {
            field: "category".into(),
            op: FilterOp::Eq,
            value: serde_json::json!("A"),
        },
        FilterExpr::Cmp {
            field: "category".into(),
            op: FilterOp::Eq,
            value: serde_json::json!("B"),
        },
        FilterExpr::Cmp {
            field: "category".into(),
            op: FilterOp::Eq,
            value: serde_json::json!("C"),
        },
    ]);

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();
    // Should match most/all (HNSW is approximate)
    assert!(results.len() >= 15);
    assert!(results.len() <= 20);
}

#[test]
fn test_range_filter() {
    let (_temp, store) = setup_filtered_store();

    // score >= 50 AND score <= 150
    let filter = FilterExpr::And(vec![
        FilterExpr::Cmp {
            field: "score".into(),
            op: FilterOp::Gte,
            value: serde_json::json!(50),
        },
        FilterExpr::Cmp {
            field: "score".into(),
            op: FilterOp::Lte,
            value: serde_json::json!(150),
        },
    ]);

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();

    for result in &results {
        let score = result
            .metadata
            .fields
            .get("score")
            .unwrap()
            .as_i64()
            .unwrap();
        assert!(score >= 50 && score <= 150);
    }
}

#[test]
fn test_boolean_field_filter() {
    let (_temp, store) = setup_filtered_store();

    let filter = FilterExpr::Cmp {
        field: "active".into(),
        op: FilterOp::Eq,
        value: serde_json::json!(true),
    };

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();

    for result in &results {
        let active = result
            .metadata
            .fields
            .get("active")
            .unwrap()
            .as_bool()
            .unwrap();
        assert!(active);
    }
}

#[test]
fn test_contains_in_string() {
    let (_temp, store) = setup_filtered_store();

    let filter = FilterExpr::Cmp {
        field: "description".into(),
        op: FilterOp::Contains,
        value: serde_json::json!("number 1"),
    };

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();

    for result in &results {
        let desc = result
            .metadata
            .fields
            .get("description")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(desc.contains("number 1"));
    }
}

#[test]
fn test_contains_in_array() {
    let (_temp, store) = setup_filtered_store();

    let filter = FilterExpr::Cmp {
        field: "tags".into(),
        op: FilterOp::Contains,
        value: serde_json::json!("tag0"),
    };

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();

    for result in &results {
        let tags = result
            .metadata
            .fields
            .get("tags")
            .unwrap()
            .as_array()
            .unwrap();
        assert!(tags.contains(&serde_json::json!("tag0")));
    }
}

#[test]
fn test_deeply_nested_filters() {
    let (_temp, store) = setup_filtered_store();

    // ((A OR B) AND (score > 50)) OR (NOT C AND active)
    let filter = FilterExpr::Or(vec![
        FilterExpr::And(vec![
            FilterExpr::Or(vec![
                FilterExpr::Cmp {
                    field: "category".into(),
                    op: FilterOp::Eq,
                    value: serde_json::json!("A"),
                },
                FilterExpr::Cmp {
                    field: "category".into(),
                    op: FilterOp::Eq,
                    value: serde_json::json!("B"),
                },
            ]),
            FilterExpr::Cmp {
                field: "score".into(),
                op: FilterOp::Gt,
                value: serde_json::json!(50),
            },
        ]),
        FilterExpr::And(vec![
            FilterExpr::Not(Box::new(FilterExpr::Cmp {
                field: "category".into(),
                op: FilterOp::Eq,
                value: serde_json::json!("C"),
            })),
            FilterExpr::Cmp {
                field: "active".into(),
                op: FilterOp::Eq,
                value: serde_json::json!(true),
            },
        ]),
    ]);

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();
    assert!(results.len() > 0); // Should match some records
}

#[test]
fn test_filter_with_zero_results() {
    let (_temp, store) = setup_filtered_store();

    // Impossible condition
    let filter = FilterExpr::And(vec![
        FilterExpr::Cmp {
            field: "score".into(),
            op: FilterOp::Gt,
            value: serde_json::json!(1000),
        },
        FilterExpr::Cmp {
            field: "score".into(),
            op: FilterOp::Lt,
            value: serde_json::json!(0),
        },
    ]);

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_filter_neq_multiple() {
    let (_temp, store) = setup_filtered_store();

    // NOT A AND NOT B (only C should match)
    let filter = FilterExpr::And(vec![
        FilterExpr::Cmp {
            field: "category".into(),
            op: FilterOp::Neq,
            value: serde_json::json!("A"),
        },
        FilterExpr::Cmp {
            field: "category".into(),
            op: FilterOp::Neq,
            value: serde_json::json!("B"),
        },
    ]);

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();

    for result in &results {
        let category = result.metadata.fields.get("category").unwrap();
        assert_eq!(category, &serde_json::json!("C"));
    }
}

#[test]
fn test_numeric_string_coercion() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let mut meta = Metadata {
        fields: HashMap::new(),
    };
    meta.fields.insert("count".into(), serde_json::json!("42")); // String number

    store
        .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta)
        .unwrap();

    // Try to compare as number
    let filter = FilterExpr::Cmp {
        field: "count".into(),
        op: FilterOp::Gt,
        value: serde_json::json!(40),
    };

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 1,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();
    // Should work due to coercion
    assert_eq!(results.len(), 1);
}

#[test]
fn test_empty_and_filter() {
    let (_temp, store) = setup_filtered_store();

    // Empty AND should match all (vacuous truth)
    let filter = FilterExpr::And(vec![]);

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();
    // Should match most/all (HNSW is approximate)
    assert!(results.len() >= 15);
    assert!(results.len() <= 20);
}

#[test]
fn test_empty_or_filter() {
    let (_temp, store) = setup_filtered_store();

    // Empty OR should match none
    let filter = FilterExpr::Or(vec![]);

    let query = Query {
        vector: vec![10.0, 0.0, 0.0],
        k: 20,
        filter: Some(filter),
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 0); // Should match none
}
