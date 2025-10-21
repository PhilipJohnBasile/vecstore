use tempfile::TempDir;
use vecstore::{HNSWSearchParams, Metadata, PrefetchQuery, Query, QueryStage, VecStore};

#[test]
fn test_hnsw_parameter_tuning() {
    let temp_dir = TempDir::new().unwrap();
    let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();

    // Insert test data
    for i in 0..100 {
        let vector = vec![i as f32 / 100.0, (100 - i) as f32 / 100.0, 0.5];
        let mut meta = Metadata {
            fields: std::collections::HashMap::new(),
        };
        meta.fields.insert("id".into(), serde_json::json!(i));
        store.upsert(format!("doc{}", i), vector, meta).unwrap();
    }

    let query_vec = vec![0.5, 0.5, 0.5];

    // Test different HNSW search parameters
    let fast_results = store
        .query_with_params(
            Query::new(query_vec.clone()).with_limit(10),
            HNSWSearchParams::fast(),
        )
        .unwrap();

    let balanced_results = store
        .query_with_params(
            Query::new(query_vec.clone()).with_limit(10),
            HNSWSearchParams::balanced(),
        )
        .unwrap();

    let high_recall_results = store
        .query_with_params(
            Query::new(query_vec.clone()).with_limit(10),
            HNSWSearchParams::high_recall(),
        )
        .unwrap();

    let max_recall_results = store
        .query_with_params(
            Query::new(query_vec.clone()).with_limit(10),
            HNSWSearchParams::max_recall(),
        )
        .unwrap();

    // All should return results
    assert_eq!(fast_results.len(), 10);
    assert_eq!(balanced_results.len(), 10);
    assert_eq!(high_recall_results.len(), 10);
    assert_eq!(max_recall_results.len(), 10);

    // Higher recall settings may find better results (higher scores)
    // But for small datasets like this, they should be similar
    assert!(balanced_results[0].score > 0.0);
    assert!(high_recall_results[0].score > 0.0);
}

#[test]
fn test_query_planning() {
    let temp_dir = TempDir::new().unwrap();
    let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();

    // Insert test data
    for i in 0..1000 {
        let vector = vec![i as f32 / 1000.0, (1000 - i) as f32 / 1000.0, 0.5];
        let mut meta = Metadata {
            fields: std::collections::HashMap::new(),
        };
        meta.fields
            .insert("category".into(), serde_json::json!("tech"));
        meta.fields
            .insert("score".into(), serde_json::json!(i as f64 / 1000.0));
        store.upsert(format!("doc{}", i), vector, meta).unwrap();
    }

    // Test query planning without filter
    let query1 = Query::new(vec![0.5, 0.5, 0.5]).with_limit(10);
    let plan1 = store.explain_query(query1).unwrap();

    assert_eq!(plan1.query_type, "Vector Search");
    assert!(plan1.steps.len() >= 2); // HNSW + Top-K selection
    assert!(plan1.estimated_cost > 0.0);
    assert!(plan1.estimated_duration_ms > 0.0);

    // Test query planning with filter
    let query2 = Query::new(vec![0.5, 0.5, 0.5])
        .with_limit(10)
        .with_filter("score > 0.5");
    let plan2 = store.explain_query(query2).unwrap();

    assert_eq!(plan2.query_type, "Filtered Vector Search");
    assert!(plan2.steps.len() >= 3); // HNSW + Filter + Top-K
    assert!(plan2.estimated_cost > plan1.estimated_cost); // Filtering adds cost

    // Check for recommendations on large k
    let query3 = Query::new(vec![0.5, 0.5, 0.5]).with_limit(150);
    let plan3 = store.explain_query(query3).unwrap();

    assert!(plan3.recommendations.iter().any(|r| r.contains("Large k")));
}

#[test]
fn test_prefetch_query_vector_search() {
    let temp_dir = TempDir::new().unwrap();
    let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();

    // Insert test data
    for i in 0..50 {
        let vector = vec![i as f32 / 50.0, (50 - i) as f32 / 50.0, 0.5];
        let mut meta = Metadata {
            fields: std::collections::HashMap::new(),
        };
        meta.fields.insert("id".into(), serde_json::json!(i));
        store.upsert(format!("doc{}", i), vector, meta).unwrap();
    }

    // Single-stage prefetch query (vector search)
    let prefetch_query = PrefetchQuery {
        stages: vec![QueryStage::VectorSearch {
            vector: vec![0.5, 0.5, 0.5],
            k: 10,
            filter: None,
        }],
    };

    let results = store.prefetch_query(prefetch_query).unwrap();
    assert_eq!(results.len(), 10);
}

#[test]
fn test_prefetch_query_with_mmr() {
    let temp_dir = TempDir::new().unwrap();
    let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();

    // Insert test data with similar vectors
    for i in 0..50 {
        let vector = vec![0.5 + (i as f32 * 0.01), 0.5 - (i as f32 * 0.01), 0.5];
        let mut meta = Metadata {
            fields: std::collections::HashMap::new(),
        };
        meta.fields.insert("id".into(), serde_json::json!(i));
        store.upsert(format!("doc{}", i), vector, meta).unwrap();
    }

    // Two-stage prefetch: broad search + MMR for diversity
    let prefetch_query = PrefetchQuery {
        stages: vec![
            QueryStage::VectorSearch {
                vector: vec![0.5, 0.5, 0.5],
                k: 20, // Fetch 20 candidates
                filter: None,
            },
            QueryStage::MMR {
                k: 5,        // Select 5 diverse results
                lambda: 0.7, // 70% relevance, 30% diversity
            },
        ],
    };

    let results = store.prefetch_query(prefetch_query).unwrap();
    assert_eq!(results.len(), 5);

    // Results should exist and have metadata
    for result in &results {
        assert!(result.metadata.fields.contains_key("id"));
        assert!(result.score > 0.0);
    }

    // MMR should have successfully selected 5 results (diversity not strictly testable with small dataset)
    assert_eq!(results.len(), 5);
}

#[test]
fn test_prefetch_query_with_filter() {
    let temp_dir = TempDir::new().unwrap();
    let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();

    // Insert test data
    for i in 0..50 {
        let vector = vec![i as f32 / 50.0, (50 - i) as f32 / 50.0, 0.5];
        let mut meta = Metadata {
            fields: std::collections::HashMap::new(),
        };
        meta.fields.insert(
            "category".into(),
            serde_json::json!(if i % 2 == 0 { "even" } else { "odd" }),
        );
        meta.fields.insert("id".into(), serde_json::json!(i));
        store.upsert(format!("doc{}", i), vector, meta).unwrap();
    }

    // Multi-stage prefetch with filter
    let prefetch_query = PrefetchQuery {
        stages: vec![
            QueryStage::VectorSearch {
                vector: vec![0.5, 0.5, 0.5],
                k: 20,
                filter: None,
            },
            QueryStage::Filter {
                expr: vecstore::FilterExpr::Cmp {
                    field: "category".into(),
                    op: vecstore::FilterOp::Eq,
                    value: serde_json::json!("even"),
                },
            },
        ],
    };

    let results = store.prefetch_query(prefetch_query).unwrap();

    // All results should have category = "even"
    for result in results {
        assert_eq!(
            result.metadata.fields.get("category").unwrap(),
            &serde_json::json!("even")
        );
    }
}

#[test]
fn test_query_builder_api() {
    let temp_dir = TempDir::new().unwrap();
    let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();

    // Insert test data
    let mut meta = Metadata {
        fields: std::collections::HashMap::new(),
    };
    meta.fields
        .insert("category".into(), serde_json::json!("tech"));
    store
        .upsert("doc1".into(), vec![0.1, 0.2, 0.3], meta)
        .unwrap();

    // Test builder API
    let query = Query::new(vec![0.1, 0.2, 0.3])
        .with_limit(5)
        .with_filter("category = 'tech'");

    assert_eq!(query.k, 5);
    assert!(query.filter.is_some());

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_prefetch_empty_stages() {
    let temp_dir = TempDir::new().unwrap();
    let store = VecStore::open(temp_dir.path().join("test.db")).unwrap();

    let prefetch_query = PrefetchQuery { stages: vec![] };

    let result = store.prefetch_query(prefetch_query);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("at least one stage"));
}

#[test]
fn test_prefetch_mmr_without_previous_stage() {
    let temp_dir = TempDir::new().unwrap();
    let store = VecStore::open(temp_dir.path().join("test.db")).unwrap();

    // MMR stage without a previous search stage should fail
    let prefetch_query = PrefetchQuery {
        stages: vec![QueryStage::MMR { k: 5, lambda: 0.7 }],
    };

    let result = store.prefetch_query(prefetch_query);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("requires previous stage"));
}

#[test]
fn test_hnsw_params_values() {
    let fast = HNSWSearchParams::fast();
    assert_eq!(fast.ef_search, 20);

    let balanced = HNSWSearchParams::balanced();
    assert_eq!(balanced.ef_search, 50);

    let high_recall = HNSWSearchParams::high_recall();
    assert_eq!(high_recall.ef_search, 100);

    let max_recall = HNSWSearchParams::max_recall();
    assert_eq!(max_recall.ef_search, 200);

    let default = HNSWSearchParams::default();
    assert_eq!(default.ef_search, 50);
}
