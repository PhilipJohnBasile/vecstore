use std::collections::HashMap;
use vecstore::{Metadata, Query, VecStore};

#[test]
fn test_save_and_reload() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    // Create store and add data
    {
        let mut store = VecStore::open(path).unwrap();

        let mut meta1 = Metadata {
            fields: HashMap::new(),
        };
        meta1
            .fields
            .insert("category".into(), serde_json::json!("A"));
        store
            .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta1)
            .unwrap();

        let mut meta2 = Metadata {
            fields: HashMap::new(),
        };
        meta2
            .fields
            .insert("category".into(), serde_json::json!("B"));
        store
            .upsert("doc2".into(), vec![0.0, 1.0, 0.0], meta2)
            .unwrap();

        assert_eq!(store.count(), 2);
        store.save().unwrap();
    }

    // Reload store
    {
        let store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 2);
        assert_eq!(store.dimension(), 3);

        // Query should work
        let query = Query {
            vector: vec![1.0, 0.0, 0.0],
            k: 1,
            filter: None,
        };
        let results = store.query(query).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
    }
}

#[test]
fn test_persistence_with_updates() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Initial save
    {
        let mut store = VecStore::open(path).unwrap();
        store
            .upsert("vec1".into(), vec![1.0, 0.0, 0.0], meta.clone())
            .unwrap();
        store.save().unwrap();
    }

    // Update and save again
    {
        let mut store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 1);

        store
            .upsert("vec2".into(), vec![0.0, 1.0, 0.0], meta.clone())
            .unwrap();
        assert_eq!(store.count(), 2);

        store.save().unwrap();
    }

    // Verify final state
    {
        let store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 2);
    }
}

#[test]
fn test_empty_store_persistence() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path();

    // Save empty store
    {
        let store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 0);
        store.save().unwrap();
    }

    // Reload empty store
    {
        let store = VecStore::open(path).unwrap();
        assert_eq!(store.count(), 0);
    }
}
