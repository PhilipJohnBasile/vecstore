//! Integration tests for VecStore
//!
//! These tests verify that major components work together.

use vecstore::*;

#[test]
fn test_geospatial_integration() {
    let mut index = geospatial::GeoIndex::new();

    let sf = geospatial::GeoPoint::new(37.7749, -122.4194);
    let oakland = geospatial::GeoPoint::new(37.8044, -122.2712);

    index.add("sf", sf, vec![0.1; 128]);
    index.add("oakland", oakland, vec![0.2; 128]);

    let results = index.radius_search(&sf, 50000.0, 10).unwrap();
    assert!(results.len() >= 1);
}

#[test]
fn test_advanced_filtering_integration() {
    use advanced_filter::*;

    let filter = FilterBuilder::new()
        .basic("age", ">=", serde_json::json!(18))
        .unwrap()
        .and()
        .basic("status", "=", serde_json::json!("active"))
        .unwrap()
        .build();

    let metadata = serde_json::json!({
        "age": 25,
        "status": "active"
    });

    assert!(filter.matches(&metadata).unwrap());
}

#[test]
fn test_profiler_integration() {
    use profiler::*;
    use std::thread;
    use std::time::Duration;

    let mut profiler = QueryProfiler::new(ProfilerConfig::default());

    profiler.start_query("test");
    profiler.start_stage("stage1");
    thread::sleep(Duration::from_millis(10));
    profiler.end_stage();
    profiler.end_query();

    let summary = profiler.summary();
    assert_eq!(summary.total_queries, 1);
}

#[test]
fn test_splade_integration() {
    use splade::*;

    let mut index = SparseIndex::new();

    let vec1 = SparseVector {
        indices: vec![0, 5, 10],
        weights: vec![0.8, 0.6, 0.4],
        dim: 1000,
    };

    index.add("doc1".to_string(), vec1);

    let query = SparseVector {
        indices: vec![0, 5],
        weights: vec![0.9, 0.7],
        dim: 1000,
    };

    let results = index.search(&query, 10);
    assert!(!results.is_empty());
}

#[test]
fn test_multi_vector_integration() {
    use multi_vector::*;

    let mut index = MultiVectorIndex::new(128);

    let doc = MultiVectorDoc::new(
        "doc1",
        vec![vec![0.1; 128], vec![0.2; 128]],
        serde_json::json!({}),
    );

    index.add(doc).unwrap();

    let query_vectors = vec![vec![0.15; 128]];
    let results = index.search(&query_vectors, 10).unwrap();

    assert!(!results.is_empty());
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_async_python_api() {
    use python_async::*;

    let mut store = AsyncPyVecStore::new(128);

    store
        .upsert("doc1".to_string(), vec![0.1; 128], serde_json::json!({}))
        .await
        .unwrap();

    let results = store.query(vec![0.1; 128], 10, None).await.unwrap();
    assert_eq!(results.len(), 1);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_kafka_connector() {
    use kafka_connector::*;

    let config = KafkaConfig::default();
    let mut pipeline = StreamingPipeline::new(config).unwrap();

    let mut rx = pipeline.start().await.unwrap();

    // Receive a few messages
    let mut count = 0;
    while count < 3 {
        if rx.recv().await.is_some() {
            count += 1;
        }
    }

    pipeline.stop().await;
    assert_eq!(count, 3);
}
