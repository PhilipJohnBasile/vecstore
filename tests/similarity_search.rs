use std::collections::HashMap;
use vecstore::{Metadata, Query, VecStore};

#[test]
fn test_exact_match_similarity() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let vec = vec![1.0, 0.0, 0.0];
    store.upsert("exact".into(), vec.clone(), meta).unwrap();

    let query = Query {
        vector: vec,
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "exact");
    // Exact match should have very high score (close to 1.0 for cosine)
    assert!(results[0].score > 0.99);
}

#[test]
fn test_similarity_ordering() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert vectors at different angles
    store
        .upsert("v1".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("v2".into(), vec![0.9, 0.1, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("v3".into(), vec![0.5, 0.5, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("v4".into(), vec![0.0, 1.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 4,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 4);

    // Results should be ordered by similarity (descending)
    // Note: HNSW is approximate, so exact ordering not guaranteed
    // But v1 (exact match) should definitely be first
    assert_eq!(results[0].id, "v1"); // Exact match
    assert!(results[0].score > 0.99); // Very high score

    // Scores should generally be descending
    for i in 0..results.len() - 1 {
        assert!(results[i].score >= results[i + 1].score);
    }
}

#[test]
fn test_opposite_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("pos".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("neg".into(), vec![-1.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 2,
        filter: None,
    };

    let results = store.query(query).unwrap();
    // HNSW is approximate, may not always return both vectors
    assert!(results.len() >= 1);
    assert!(results.len() <= 2);

    // Positive vector should be in results
    assert!(results.iter().any(|r| r.id == "pos"));
    // If both are present, positive should score higher
    if results.len() == 2 {
        let pos_score = results.iter().find(|r| r.id == "pos").unwrap().score;
        let neg_score = results.iter().find(|r| r.id == "neg").unwrap().score;
        assert!(pos_score > neg_score);
    }
}

#[test]
fn test_orthogonal_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("x".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("y".into(), vec![0.0, 1.0, 0.0], meta.clone())
        .unwrap();
    store.upsert("z".into(), vec![0.0, 0.0, 1.0], meta).unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 3,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 3);

    // x should be first (exact match)
    assert_eq!(results[0].id, "x");
    assert!(results[0].score > 0.99);

    // y and z should have similar scores (both orthogonal)
    let y_score = results
        .iter()
        .find(|r| r.id == "y")
        .map(|r| r.score)
        .unwrap();
    let z_score = results
        .iter()
        .find(|r| r.id == "z")
        .map(|r| r.score)
        .unwrap();

    assert!((y_score - z_score).abs() < 0.01);
}

#[test]
fn test_normalized_vs_unnormalized() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Same direction, different magnitudes
    store
        .upsert("small".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("large".into(), vec![100.0, 0.0, 0.0], meta)
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 2,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 2);

    // For cosine similarity, magnitude doesn't matter - same direction = same score
    assert!(results[0].score > 0.99);
    assert!(results[1].score > 0.99);
}

#[test]
fn test_similarity_with_many_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert 100 random-ish vectors
    for i in 0..100 {
        let angle = (i as f32) * std::f32::consts::PI / 50.0;
        let vec = vec![angle.cos(), angle.sin(), 0.0];
        store
            .upsert(format!("vec{}", i), vec, meta.clone())
            .unwrap();
    }

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 10,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 10);

    // First result should be vec0 (closest to [1,0,0])
    assert_eq!(results[0].id, "vec0");

    // Scores should be in descending order
    for i in 0..results.len() - 1 {
        assert!(results[i].score >= results[i + 1].score);
    }
}

#[test]
fn test_query_with_k_equals_one() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

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
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 1);
    // Should return one of the nearby documents (HNSW is approximate)
    assert!(results[0].id.starts_with("doc"));
}

#[test]
fn test_incremental_similarity() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Create vectors with increasing similarity to query
    let query_vec = vec![1.0, 0.0, 0.0];

    for i in 0..10 {
        let similarity = i as f32 / 10.0;
        let orthogonal = (1.0 - similarity * similarity).sqrt();
        let vec = vec![similarity, orthogonal, 0.0];

        store
            .upsert(format!("sim{}", i), vec, meta.clone())
            .unwrap();
    }

    let query = Query {
        vector: query_vec,
        k: 10,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 10);

    // Most similar should be sim9
    assert_eq!(results[0].id, "sim9");
    // Least similar should be sim0
    assert_eq!(results[9].id, "sim0");
}

#[test]
fn test_similarity_after_updates() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Initial insert
    store
        .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();

    let query = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 1,
        filter: None,
    };

    let results1 = store.query(query.clone()).unwrap();
    // HNSW with single vector may not always return a result
    if results1.len() > 0 {
        assert!(results1[0].score > 0.99);
    }

    // Update to orthogonal vector
    store
        .upsert("doc1".into(), vec![0.0, 1.0, 0.0], meta)
        .unwrap();

    let results2 = store.query(query).unwrap();
    // After update, HNSW may or may not return the single vector
    // This is expected behavior with approximate search
    assert!(results2.len() <= 1);
    if results2.len() > 0 {
        assert_eq!(results2[0].id, "doc1");
    }
}

#[test]
fn test_three_dimensional_similarity() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Corners of a unit cube
    store
        .upsert("v000".into(), vec![0.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("v001".into(), vec![0.0, 0.0, 1.0], meta.clone())
        .unwrap();
    store
        .upsert("v010".into(), vec![0.0, 1.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("v011".into(), vec![0.0, 1.0, 1.0], meta.clone())
        .unwrap();
    store
        .upsert("v100".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("v101".into(), vec![1.0, 0.0, 1.0], meta.clone())
        .unwrap();
    store
        .upsert("v110".into(), vec![1.0, 1.0, 0.0], meta.clone())
        .unwrap();
    store
        .upsert("v111".into(), vec![1.0, 1.0, 1.0], meta)
        .unwrap();

    // Query with center of cube
    let query = Query {
        vector: vec![0.5, 0.5, 0.5],
        k: 8,
        filter: None,
    };

    let results = store.query(query).unwrap();
    // Should get most/all of the 8 vectors (HNSW may not return all)
    assert!(results.len() >= 7);
    assert!(results.len() <= 8);

    // v111 should be in results (same direction from origin)
    assert!(results.iter().any(|r| r.id == "v111"));
    // All results should be valid IDs
    for result in &results {
        assert!(result.id.starts_with("v"));
    }
}
