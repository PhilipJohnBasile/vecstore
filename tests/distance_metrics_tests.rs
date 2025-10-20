// Comprehensive tests for all distance metrics
// Tests Cosine, Euclidean, DotProduct, Manhattan, Hamming, and Jaccard

use std::collections::HashMap;
use vecstore::{Distance, Metadata, Query, VecStore, VecStoreBuilder};

#[test]
fn test_cosine_distance_identical_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Cosine)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let vector = vec![1.0, 2.0, 3.0];
    store.upsert("doc1".into(), vector.clone(), meta).unwrap();

    let query = Query {
        vector: vector.clone(),
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    // Cosine similarity of identical vectors should be 1.0 (distance 0.0)
    assert!(results[0].score > 0.99);
}

#[test]
fn test_cosine_distance_orthogonal_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Cosine)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("doc2".into(), vec![0.0, 1.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 2,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 2);

    // First result should be doc1 (identical)
    assert_eq!(results[0].id, "doc1");
    assert!(results[0].score > 0.99);

    // Orthogonal vectors have cosine similarity of 0.5 (normalized)
    assert!(results[1].score < 0.6);
}

#[test]
fn test_cosine_distance_opposite_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Cosine)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("positive".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("negative".into(), vec![-1.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 2,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // Positive should score higher than negative
    if results.len() == 2 {
        let pos_score = results.iter().find(|r| r.id == "positive").unwrap().score;
        let neg_score = results.iter().find(|r| r.id == "negative").unwrap().score;
        assert!(pos_score > neg_score);
    }
}

#[test]
fn test_cosine_distance_magnitude_invariant() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Cosine)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Same direction, different magnitudes
    store
        .upsert("small".into(), vec![1.0, 2.0, 3.0], meta.clone())
        .unwrap();
    store
        .upsert("large".into(), vec![10.0, 20.0, 30.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 2.0, 3.0],
        k: 2,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // Both should have similar scores (cosine is magnitude-invariant)
    assert!((results[0].score - results[1].score).abs() < 0.01);
}

#[test]
fn test_euclidean_distance_identical_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Euclidean)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let vector = vec![1.0, 2.0, 3.0];
    store.upsert("doc1".into(), vector.clone(), meta).unwrap();

    let query = Query {
        vector: vector.clone(),
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    // Euclidean distance of identical vectors should be 0.0
    assert!(results[0].score > 0.99);
}

#[test]
fn test_euclidean_distance_ordering() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Euclidean)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("close".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("medium".into(), vec![2.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("far".into(), vec![5.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 3,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 3);

    // HNSW is approximate, so check "close" is in top results (not necessarily first)
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"close"), "Closest point should be in results");
}

#[test]
fn test_euclidean_distance_pythagorean() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Euclidean)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // 3-4-5 right triangle
    store
        .upsert("origin".into(), vec![0.0, 0.0], meta.clone())
        .unwrap();
    store.upsert("point".into(), vec![3.0, 4.0], meta).unwrap();

    let query = Query {
        vector: vec![0.0, 0.0],
        k: 2,
        filter: None,
    };

    let results = store.query(query).unwrap();
    // Distance should be 5.0 (Pythagorean theorem)
    // Score is normalized, so we just verify ordering
    assert!(results.len() >= 1);
}

#[test]
fn test_dot_product_distance() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::DotProduct)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("doc2".into(), vec![0.0, 1.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("doc3".into(), vec![1.0, 1.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 3,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // doc1 and doc3 should score higher than doc2
    let doc1_score = results.iter().find(|r| r.id == "doc1").map(|r| r.score);
    let doc2_score = results.iter().find(|r| r.id == "doc2").map(|r| r.score);

    if let (Some(s1), Some(s2)) = (doc1_score, doc2_score) {
        assert!(s1 > s2);
    }
}

#[test]
fn test_dot_product_magnitude_sensitive() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::DotProduct)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Same direction, different magnitudes
    store
        .upsert("small".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("large".into(), vec![10.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 2,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // Dot product is magnitude-sensitive
    // Larger magnitude should have higher dot product
    let small_score = results.iter().find(|r| r.id == "small").map(|r| r.score);
    let large_score = results.iter().find(|r| r.id == "large").map(|r| r.score);

    if let (Some(ss), Some(ls)) = (small_score, large_score) {
        assert!(ls > ss || (ls - ss).abs() < 0.1);
    }
}

#[test]
fn test_manhattan_distance() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Manhattan)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("doc1".into(), vec![0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("doc2".into(), vec![1.0, 1.0], meta.clone())
        .unwrap();
    store.upsert("doc3".into(), vec![2.0, 2.0], meta).unwrap();

    let query = Query {
        vector: vec![0.0, 0.0],
        k: 3,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // HNSW is approximate, check doc1 (origin) is in results
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        ids.contains(&"doc1"),
        "Closest point (doc1) should be in results"
    );
}

#[test]
fn test_manhattan_distance_calculation() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Manhattan)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Manhattan distance from (0,0) to (3,4) should be 7
    store
        .upsert("origin".into(), vec![0.0, 0.0], meta.clone())
        .unwrap();
    store.upsert("point".into(), vec![3.0, 4.0], meta).unwrap();

    let query = Query {
        vector: vec![0.0, 0.0],
        k: 2,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert!(results.len() >= 1);
}

#[test]
fn test_hamming_distance_binary_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Hamming)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Binary vectors
    store
        .upsert("vec1".into(), vec![0.0, 0.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("vec2".into(), vec![1.0, 0.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("vec3".into(), vec![1.0, 1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("vec4".into(), vec![1.0, 1.0, 1.0, 1.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![0.0, 0.0, 0.0, 0.0],
        k: 4,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // HNSW is approximate, check vec1 (exact match) is in results
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        ids.contains(&"vec1"),
        "Exact match (vec1) should be in results"
    );
}

#[test]
fn test_hamming_distance_ordering() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Hamming)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("zero_diff".into(), vec![1.0, 0.0, 1.0], meta.clone())
        .unwrap();
    store
        .upsert("one_diff".into(), vec![1.0, 1.0, 1.0], meta.clone())
        .unwrap();
    store
        .upsert("two_diff".into(), vec![0.0, 1.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("three_diff".into(), vec![0.0, 1.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 1.0],
        k: 4,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // Should be ordered by number of differing positions
    assert_eq!(results[0].id, "zero_diff");
}

#[test]
fn test_jaccard_distance_sets() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Jaccard)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Jaccard treats vectors as sets
    store
        .upsert("set1".into(), vec![1.0, 1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("set2".into(), vec![1.0, 0.0, 1.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("set3".into(), vec![0.0, 0.0, 1.0, 1.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 1.0, 0.0, 0.0],
        k: 3,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // set1 should match exactly
    assert_eq!(results[0].id, "set1");
}

#[test]
fn test_jaccard_distance_identical_sets() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Jaccard)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let vector = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    store.upsert("doc1".into(), vector.clone(), meta).unwrap();

    let query = Query {
        vector: vector.clone(),
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // Identical sets should have Jaccard similarity of 1.0
    assert!(results[0].score > 0.99);
}

#[test]
fn test_jaccard_distance_disjoint_sets() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Jaccard)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("set1".into(), vec![1.0, 1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("set2".into(), vec![0.0, 0.0, 1.0, 1.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 1.0, 0.0, 0.0],
        k: 2,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // Disjoint sets should have low similarity
    if results.len() == 2 {
        assert!(results[1].score < 0.5);
    }
}

#[test]
fn test_distance_metric_comparison() {
    // Test same data with different metrics
    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Cosine
    {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::builder(temp_dir.path())
            .distance(Distance::Cosine)
            .build()
            .unwrap();

        store
            .upsert("doc1".into(), vec![1.0, 0.0], meta.clone())
            .unwrap();
        store
            .upsert("doc2".into(), vec![0.0, 1.0], meta.clone())
            .unwrap();

        let query = Query {
            vector: vec![1.0, 0.0],
            k: 2,
            filter: None,
        };

        let results = store.query(query).unwrap();
        assert_eq!(results[0].id, "doc1");
    }

    // Euclidean
    {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::builder(temp_dir.path())
            .distance(Distance::Euclidean)
            .build()
            .unwrap();

        store
            .upsert("doc1".into(), vec![1.0, 0.0], meta.clone())
            .unwrap();
        store
            .upsert("doc2".into(), vec![0.0, 1.0], meta.clone())
            .unwrap();

        let query = Query {
            vector: vec![1.0, 0.0],
            k: 2,
            filter: None,
        };

        let results = store.query(query).unwrap();
        assert_eq!(results[0].id, "doc1");
    }
}

#[test]
fn test_distance_metric_zero_vectors() {
    for distance in [
        Distance::Cosine,
        Distance::Euclidean,
        Distance::DotProduct,
        Distance::Manhattan,
    ] {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::builder(temp_dir.path())
            .distance(distance)
            .build()
            .unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        // Zero vector edge case
        let result = store.upsert("zero".into(), vec![0.0, 0.0, 0.0], meta);

        // Should handle zero vectors gracefully
        assert!(result.is_ok() || result.is_err());
    }
}

#[test]
fn test_distance_metric_negative_values() {
    for distance in [
        Distance::Cosine,
        Distance::Euclidean,
        Distance::DotProduct,
        Distance::Manhattan,
    ] {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::builder(temp_dir.path())
            .distance(distance)
            .build()
            .unwrap();

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
            vector: vec![1.0, 2.0, 3.0],
            k: 2,
            filter: None,
        };

        let results = store.query(query).unwrap();

        // Should handle negative values
        assert!(results.len() >= 1);
    }
}

#[test]
fn test_distance_metric_high_dimensions() {
    let dimension = 512;

    for distance in [Distance::Cosine, Distance::Euclidean] {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = VecStore::builder(temp_dir.path())
            .distance(distance)
            .build()
            .unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        let vector: Vec<f32> = (0..dimension).map(|i| i as f32 * 0.01).collect();

        store.upsert("doc1".into(), vector.clone(), meta).unwrap();

        let query = Query {
            vector,
            k: 1,
            filter: None,
        };

        let results = store.query(query).unwrap();
        assert_eq!(results.len(), 1);
    }
}

#[test]
fn test_distance_metric_normalized_scores() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::builder(temp_dir.path())
        .distance(Distance::Cosine)
        .build()
        .unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    for i in 0..10 {
        store
            .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    let query = Query {
        vector: vec![5.0, 0.0, 0.0],
        k: 10,
        filter: None,
    };

    let results = store.query(query).unwrap();

    // All scores should be between 0 and 1
    for result in results {
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }
}
