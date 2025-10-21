use std::collections::HashMap;
use vecstore::{make_record, Metadata, Query, VecStore};

#[test]
fn test_batch_upsert_basic() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let records: Vec<_> = (0..100)
        .map(|i| make_record(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone()))
        .collect();

    store.batch_upsert(records).unwrap();
    assert_eq!(store.count(), 100);
}

#[test]
fn test_batch_upsert_empty() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let records: Vec<_> = vec![];
    store.batch_upsert(records).unwrap();
    assert_eq!(store.count(), 0);
}

#[test]
fn test_batch_upsert_single() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let records = vec![make_record("doc1", vec![1.0, 0.0, 0.0], meta)];

    store.batch_upsert(records).unwrap();
    assert_eq!(store.count(), 1);
}

#[test]
fn test_batch_upsert_dimension_mismatch() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // First insert sets dimension to 3
    store
        .upsert("initial".into(), vec![1.0, 0.0, 0.0], meta.clone())
        .unwrap();

    // Batch with wrong dimension should fail
    let records = vec![
        make_record("doc1", vec![1.0, 0.0], meta.clone()), // Wrong dimension
        make_record("doc2", vec![1.0, 0.0, 0.0], meta),
    ];

    let result = store.batch_upsert(records);
    assert!(result.is_err());
}

#[test]
fn test_batch_upsert_large() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let records: Vec<_> = (0..1000)
        .map(|i| {
            make_record(
                format!("doc{}", i),
                vec![i as f32, (i * 2) as f32, (i * 3) as f32],
                meta.clone(),
            )
        })
        .collect();

    store.batch_upsert(records).unwrap();
    assert_eq!(store.count(), 1000);

    // Query should work correctly
    let query = Query {
        vector: vec![500.0, 1000.0, 1500.0],
        k: 5,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn test_batch_upsert_with_duplicates() {
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

    let records = vec![
        make_record("doc1", vec![1.0, 0.0, 0.0], meta1.clone()),
        make_record("doc2", vec![0.0, 1.0, 0.0], meta1),
        make_record("doc1", vec![1.0, 1.0, 0.0], meta2), // Duplicate ID
    ];

    store.batch_upsert(records).unwrap();
    assert_eq!(store.count(), 2); // Only 2 unique IDs

    let query = Query {
        vector: vec![1.0, 1.0, 0.0],
        k: 1,
        filter: None,
    };

    let results = store.query(query).unwrap();
    assert_eq!(results[0].id, "doc1");
    assert_eq!(
        results[0].metadata.fields.get("version").unwrap(),
        &serde_json::json!(2)
    );
}

#[test]
fn test_batch_then_individual_upsert() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Batch insert
    let records: Vec<_> = (0..50)
        .map(|i| make_record(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone()))
        .collect();

    store.batch_upsert(records).unwrap();
    assert_eq!(store.count(), 50);

    // Individual inserts
    for i in 50..60 {
        store
            .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    assert_eq!(store.count(), 60);
}

#[test]
fn test_batch_upsert_all_dimension_consistent() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let records = vec![
        make_record("doc1", vec![1.0, 0.0, 0.0], meta.clone()),
        make_record("doc2", vec![0.0, 1.0, 0.0], meta.clone()),
        make_record("doc3", vec![0.0, 0.0, 1.0], meta),
    ];

    store.batch_upsert(records).unwrap();
    assert_eq!(store.count(), 3);
    assert_eq!(store.dimension(), 3);
}

#[test]
fn test_multiple_batch_upserts() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // First batch
    let records1: Vec<_> = (0..100)
        .map(|i| {
            make_record(
                format!("batch1_{}", i),
                vec![i as f32, 0.0, 0.0],
                meta.clone(),
            )
        })
        .collect();

    store.batch_upsert(records1).unwrap();
    assert_eq!(store.count(), 100);

    // Second batch
    let records2: Vec<_> = (0..100)
        .map(|i| {
            make_record(
                format!("batch2_{}", i),
                vec![i as f32, 1.0, 0.0],
                meta.clone(),
            )
        })
        .collect();

    store.batch_upsert(records2).unwrap();
    assert_eq!(store.count(), 200);

    // Third batch with some overlaps (updates batch1_50..99 and creates batch3_0..49)
    let mut records3: Vec<_> = (50..100)
        .map(|i| {
            make_record(
                format!("batch1_{}", i),
                vec![i as f32, 2.0, 0.0],
                meta.clone(),
            )
        })
        .collect();
    records3.extend((0..50).map(|i| {
        make_record(
            format!("batch3_{}", i),
            vec![i as f32, 2.0, 0.0],
            meta.clone(),
        )
    }));

    store.batch_upsert(records3).unwrap();
    assert_eq!(store.count(), 250); // 100 (batch1) + 100 (batch2) + 50 (batch3)
}
