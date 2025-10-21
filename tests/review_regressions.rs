use std::collections::HashMap;
use std::panic::{self, AssertUnwindSafe};

use tempfile::tempdir;
use vecstore::store::hybrid::{HybridQuery, TextIndex};
use vecstore::{
    distributed::{DistributedConfig, DistributedStore},
    parse_filter, Distance, HNSWSearchParams, Metadata, NamespaceManager, NamespaceStatus, Query,
    VecStore,
};

fn empty_metadata() -> Metadata {
    Metadata {
        fields: HashMap::new(),
    }
}

#[test]
fn namespace_manager_should_persist_upserts_without_manual_save() {
    let temp_dir = tempdir().expect("tempdir");
    let manager = NamespaceManager::new(temp_dir.path()).expect("manager");

    manager
        .create_namespace("ns1".to_string(), "Test Namespace".to_string(), None)
        .expect("create_namespace");

    let mut metadata = empty_metadata();
    metadata
        .fields
        .insert("tag".to_string(), serde_json::json!("v1"));

    manager
        .upsert(
            &"ns1".to_string(),
            "doc1".to_string(),
            vec![1.0, 0.0],
            metadata,
        )
        .expect("upsert");

    // Drop the manager without calling VecStore::save()
    drop(manager);

    let manager_reloaded = NamespaceManager::new(temp_dir.path()).expect("reloaded");
    manager_reloaded.load_namespaces().expect("load_namespaces");

    let results = manager_reloaded
        .query(&"ns1".to_string(), Query::new(vec![1.0, 0.0]).with_limit(1))
        .expect("query");

    assert!(
        !results.is_empty(),
        "NamespaceManager lost data across restart because VecStore::save() is never invoked."
    );
}

#[test]
fn namespace_manager_should_block_queries_when_suspended() {
    let temp_dir = tempdir().expect("tempdir");
    let manager = NamespaceManager::new(temp_dir.path()).expect("manager");

    manager
        .create_namespace("ns1".to_string(), "Test Namespace".to_string(), None)
        .expect("create_namespace");

    manager
        .upsert(
            &"ns1".to_string(),
            "doc".to_string(),
            vec![1.0, 0.0],
            empty_metadata(),
        )
        .expect("upsert");

    manager
        .update_status(&"ns1".to_string(), NamespaceStatus::Suspended)
        .expect("update_status");

    let result = manager.query(&"ns1".to_string(), Query::new(vec![1.0, 0.0]));

    assert!(
        result.is_err(),
        "Suspended namespace should reject queries, but query succeeded: {:?}",
        result
    );
}

#[test]
fn text_index_reindex_should_not_drift_bm25_scores() {
    let mut text_index = TextIndex::new();
    text_index.index_document("doc1".to_string(), "rust search vectors".to_string());

    let baseline = text_index
        .bm25_scores("rust")
        .get("doc1")
        .copied()
        .expect("baseline score");

    // Re-index the same document (common when metadata/text updates)
    text_index.index_document("doc1".to_string(), "rust search vectors".to_string());

    let after_reindex = text_index
        .bm25_scores("rust")
        .get("doc1")
        .copied()
        .unwrap_or_default();

    assert!(
        (baseline - after_reindex).abs() < 1e-6,
        "BM25 score drifted after reindex: baseline={baseline}, after={after_reindex}"
    );
}

#[test]
fn vecstore_should_reject_zero_length_vectors() {
    let temp_dir = tempdir().expect("tempdir");
    let mut store = VecStore::open(temp_dir.path()).expect("store");

    let result = store.upsert("empty".to_string(), vec![], empty_metadata());

    assert!(
        result.is_err(),
        "VecStore accepted zero-length vector; should return an error"
    );
}

#[test]
fn vecstore_should_require_minimum_dimension() {
    let temp_dir = tempdir().expect("tempdir");
    let mut store = VecStore::open(temp_dir.path()).expect("store");

    store
        .upsert("doc1".to_string(), vec![1.0, 0.0, 0.0], empty_metadata())
        .expect("upsert doc1");

    let result = store.upsert("doc2".to_string(), vec![1.0, 0.0], empty_metadata());

    assert!(
        result.is_err(),
        "VecStore accepted vector with mismatched dimension; should reject"
    );
}

#[test]
fn graph_visualizer_native_backend_should_error() {
    let temp_dir = tempdir().expect("tempdir");
    let store = VecStore::open(temp_dir.path()).expect("store");

    store
        .upsert("doc".to_string(), vec![1.0, 0.0], empty_metadata())
        .expect("upsert");

    let result = store.visualizer();

    assert!(
        result.is_err(),
        "Graph visualizer should error on native backend, but returned Ok"
    );
}

#[test]
fn removing_records_should_drop_text_index_state() {
    let temp_dir = tempdir().expect("tempdir");
    let mut store = VecStore::open(temp_dir.path()).expect("store");

    store
        .upsert("doc1".to_string(), vec![1.0, 0.0], empty_metadata())
        .expect("upsert");
    store
        .index_text("doc1", "rust search vectors")
        .expect("index_text");
    assert!(store.has_text("doc1"));

    store.remove("doc1").expect("remove");

    assert!(
        !store.has_text("doc1"),
        "Text index retained postings for a removed document."
    );
}

#[test]
fn query_explain_should_report_configured_distance_metric() {
    let temp_dir = tempdir().expect("tempdir");
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Euclidean)
        .build()
        .expect("build");

    store
        .upsert("doc_a".to_string(), vec![1.0, 0.0], empty_metadata())
        .expect("upsert a");
    store
        .upsert("doc_b".to_string(), vec![0.0, 1.0], empty_metadata())
        .expect("upsert b");

    let explanations = store
        .query_explain(Query::new(vec![1.0, 0.0]).with_limit(1))
        .expect("query explain");

    let metric = &explanations[0].explanation.distance_metric;

    assert_eq!(
        metric, "Euclidean",
        "Configured distance metric was ignored; explanation reports '{metric}'."
    );
}

#[test]
fn text_index_should_survive_store_reopen() {
    let temp_dir = tempdir().expect("tempdir");
    {
        let mut store = VecStore::open(temp_dir.path()).expect("store");
        store
            .upsert("doc1".to_string(), vec![1.0, 0.0], empty_metadata())
            .expect("upsert");
        store
            .index_text("doc1", "rust hybrid search")
            .expect("index_text");
        assert!(store.has_text("doc1"));
    }

    // Reopen the store from disk.
    let reopened = VecStore::open(temp_dir.path()).expect("reopen");

    assert!(
        reopened.has_text("doc1"),
        "Text index was not persisted; hybrid search loses data after restart."
    );
}

#[test]
fn store_configuration_should_persist_across_reopen() {
    let temp_dir = tempdir().expect("tempdir");
    {
        let _store = VecStore::builder(temp_dir.path())
            .distance(Distance::Manhattan)
            .hnsw_m(42)
            .hnsw_ef_construction(77)
            .build()
            .expect("build");
        // Drop store to flush to disk.
    }

    let reopened = VecStore::open(temp_dir.path()).expect("reopen");
    let config = reopened.config();

    assert_eq!(
        reopened.distance_metric(),
        Distance::Manhattan,
        "Store distance configuration was not persisted across reopen."
    );
    assert_eq!(
        config.hnsw_m, 42,
        "HNSW 'M' parameter reverted to default after reopen."
    );
    assert_eq!(
        config.hnsw_ef_construction, 77,
        "HNSW 'ef_construction' parameter reverted to default after reopen."
    );
}

#[test]
fn filter_parser_should_support_in_operator() {
    let filter_str = "status IN ['active', 'pending']";
    let result = parse_filter(filter_str);
    assert!(
        result.is_ok(),
        "parse_filter failed for valid IN expression: {result:?}"
    );
}

#[test]
fn filter_parser_should_support_startswith_operator() {
    let filter_str = "doc_id STARTSWITH 'rust_'";
    let result = parse_filter(filter_str);
    assert!(
        result.is_ok(),
        "parse_filter failed for STARTSWITH used in examples: {result:?}"
    );
}

#[test]
fn query_with_params_should_return_requested_k() {
    let temp_dir = tempdir().expect("tempdir");
    let mut store = VecStore::open(temp_dir.path()).expect("store");

    for i in 0..10 {
        store
            .upsert(
                format!("doc{i}"),
                vec![i as f32, 0.0, 0.0],
                empty_metadata(),
            )
            .expect("upsert");
    }

    let params = HNSWSearchParams { ef_search: 3 };
    let results = store
        .query_with_params(Query::new(vec![0.0, 0.0, 0.0]).with_limit(5), params)
        .expect("query_with_params");

    assert_eq!(
        results.len(),
        5,
        "query_with_params returned {} results instead of requested 5",
        results.len()
    );
}

#[test]
fn query_with_large_k_should_not_overflow_fetch_size() {
    let temp_dir = tempdir().expect("tempdir");
    let mut store = VecStore::open(temp_dir.path()).expect("store");

    store
        .upsert("doc".to_string(), vec![1.0, 0.0, 0.0], empty_metadata())
        .expect("upsert");

    let huge_k = usize::MAX / 2;
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = store
            .query(Query::new(vec![1.0, 0.0, 0.0]).with_limit(huge_k))
            .expect("query");
    }));

    assert!(
        result.is_ok(),
        "VecStore::query overflowed when computing fetch_size (k * 10) for large k"
    );
}

#[test]
fn explain_query_should_not_emit_nan_for_empty_store() {
    let temp_dir = tempdir().expect("tempdir");
    let store = VecStore::open(temp_dir.path()).expect("store");

    let plan = store
        .explain_query(Query::new(vec![]).with_limit(5))
        .expect("explain_query");

    assert!(
        plan.estimated_cost.is_finite(),
        "explain_query produced non-finite estimated_cost: {:?}",
        plan.estimated_cost
    );
    assert!(
        plan.steps.iter().all(|step| step.cost.is_finite()),
        "explain_query produced step cost that is not finite: {:?}",
        plan.steps
    );
}

#[test]
fn hybrid_query_should_skip_soft_deleted_records() {
    let temp_dir = tempdir().expect("tempdir");
    let mut store = VecStore::open(temp_dir.path()).expect("store");

    store
        .upsert("doc1".to_string(), vec![1.0, 0.0], empty_metadata())
        .expect("upsert");
    store
        .index_text("doc1", "rust vector search")
        .expect("index_text");

    // Soft delete the record
    assert!(store.soft_delete("doc1").expect("soft_delete"));

    let query = HybridQuery {
        vector: vec![1.0, 0.0],
        keywords: "rust".into(),
        k: 5,
        filter: None,
        alpha: 0.5,
    };

    let results = store.hybrid_query(query).expect("hybrid_query");

    assert!(
        results.is_empty(),
        "hybrid_query returned soft-deleted record(s): {:?}",
        results
    );
}

#[test]
fn query_with_params_should_reject_zero_ef_search() {
    let temp_dir = tempdir().expect("tempdir");
    let mut store = VecStore::open(temp_dir.path()).expect("store");

    store
        .upsert("doc".to_string(), vec![1.0, 0.0, 0.0], empty_metadata())
        .expect("upsert");

    let params = HNSWSearchParams { ef_search: 0 };
    let results = store
        .query_with_params(Query::new(vec![1.0, 0.0, 0.0]).with_limit(1), params)
        .expect("query_with_params");

    assert!(
        !results.is_empty(),
        "ef_search=0 caused query_with_params to return zero results; should be validated"
    );
}

#[test]
fn explain_query_should_fail_gracefully_when_vector_missing() {
    let temp_dir = tempdir().expect("tempdir");
    let store = VecStore::open(temp_dir.path()).expect("store");

    let result = store.explain_query(Query::new(vec![]).with_limit(1));

    assert!(
        result.is_err(),
        "explain_query succeeded with empty vector; should return an error"
    );
}

#[test]
fn prefetch_query_should_validate_stage_order() {
    use vecstore::{PrefetchQuery, QueryStage};

    let temp_dir = tempdir().expect("tempdir");
    let store = VecStore::open(temp_dir.path()).expect("store");

    let query = PrefetchQuery {
        stages: vec![QueryStage::Rerank { k: 10, model: None }],
    };

    let err = store
        .prefetch_query(query)
        .expect_err("prefetch_query allowed Rerank as the first stage");

    assert!(
        err.to_string().contains("first stage must be a search"),
        "prefetch_query returned confusing error message: {err}"
    );
}

#[test]
fn query_with_filter_should_not_silently_ignore_parse_errors() {
    let temp_dir = tempdir().expect("tempdir");
    let mut store = VecStore::open(temp_dir.path()).expect("store");

    let mut meta_admin = empty_metadata();
    meta_admin
        .fields
        .insert("role".into(), serde_json::json!("admin"));
    store
        .upsert("admin".to_string(), vec![1.0, 0.0, 0.0], meta_admin)
        .expect("upsert admin");

    let mut meta_user = empty_metadata();
    meta_user
        .fields
        .insert("role".into(), serde_json::json!("user"));
    store
        .upsert("user".to_string(), vec![0.0, 1.0, 0.0], meta_user)
        .expect("upsert user");

    // Attempt to filter using SQL syntax showcased in docs/examples, but not supported by parser.
    let query = Query::new(vec![0.0, 1.0, 0.0])
        .with_limit(10)
        .with_filter("role IN ['admin']");

    let results = store.query(query).expect("query");

    assert!(
        results
            .iter()
            .all(|neighbor| neighbor.metadata.fields.get("role") == Some(&serde_json::json!("admin"))),
        "Invalid filter syntax should not be silently ignored; expected only admin results, got {:?}",
        results
    );
}

#[test]
fn distributed_config_zero_shards_panics() {
    use vecstore::distributed::DistributedStore;

    let config = vecstore::distributed::DistributedConfig::new().with_num_shards(0);
    let store = DistributedStore::create(config).expect("create");

    let result = std::panic::catch_unwind(|| {
        let _ = store.get_shard_id("doc1");
    });

    assert!(
        result.is_err(),
        "DistributedStore::get_shard_id should reject zero-shard configuration instead of panicking"
    );
}
