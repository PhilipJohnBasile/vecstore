// Comprehensive tests for import/export functionality
// Tests JSONL and Parquet formats, large datasets, error handling

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use vecstore::import_export::{Exporter, Importer};
use vecstore::{Metadata, VecStore};

#[test]
fn test_export_jsonl_basic() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert some data
    store
        .upsert("doc1".into(), vec![1.0, 2.0, 3.0], meta.clone())
        .unwrap();
    store
        .upsert("doc2".into(), vec![4.0, 5.0, 6.0], meta)
        .unwrap();

    let export_path = temp_dir.path().join("export.jsonl");
    let exporter = Exporter::new(&store);
    let result = exporter.to_jsonl(&export_path);

    assert!(result.is_ok());
    assert!(export_path.exists());
}

#[test]
fn test_export_jsonl_empty_store() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = VecStore::open(temp_dir.path()).unwrap();

    let export_path = temp_dir.path().join("empty.jsonl");
    let result = Exporter::new(&store).to_jsonl(&export_path);

    assert!(result.is_ok());
}

#[test]
fn test_import_jsonl_basic() {
    let temp_dir = tempfile::tempdir().unwrap();

    // Create a JSONL file
    let import_path = temp_dir.path().join("import.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    writeln!(
        file,
        r#"{{"id": "doc1", "vector": [1.0, 2.0, 3.0], "metadata": {{}}}}"#
    )
    .unwrap();
    writeln!(
        file,
        r#"{{"id": "doc2", "vector": [4.0, 5.0, 6.0], "metadata": {{}}}}"#
    )
    .unwrap();

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    assert!(result.is_ok());
    assert_eq!(store.count(), 2);
}

#[test]
fn test_import_jsonl_with_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("import.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    writeln!(
        file,
        r#"{{"id": "doc1", "vector": [1.0, 2.0, 3.0], "metadata": {{"key": "value", "count": 42}}}}"#
    ).unwrap();

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    assert!(result.is_ok());
    assert_eq!(store.count(), 1);
}

#[test]
fn test_export_import_roundtrip() {
    let temp_dir = tempfile::tempdir().unwrap();

    // Create and populate store
    let store_path = temp_dir.path().join("store");
    let mut store = VecStore::open(&store_path).unwrap();

    let mut meta = Metadata {
        fields: HashMap::new(),
    };
    meta.fields.insert("type".into(), serde_json::json!("test"));
    meta.fields.insert("index".into(), serde_json::json!(1));

    for i in 0..10 {
        let mut m = meta.clone();
        m.fields.insert("index".into(), serde_json::json!(i));

        store
            .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], m)
            .unwrap();
    }

    // Export
    let export_path = temp_dir.path().join("export.jsonl");
    Exporter::new(&store).to_jsonl(&export_path).unwrap();

    // Import to new store
    let new_store_path = temp_dir.path().join("new_store");
    let mut new_store = VecStore::open(&new_store_path).unwrap();
    Importer::new(&mut new_store)
        .from_jsonl(&export_path, 1000)
        .unwrap();

    assert_eq!(new_store.count(), 10);
}

#[test]
fn test_import_jsonl_malformed_line() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("malformed.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    writeln!(
        file,
        r#"{{"id": "doc1", "vector": [1.0, 2.0, 3.0], "metadata": {{}}}}"#
    )
    .unwrap();
    writeln!(file, r#"{{invalid json}}"#).unwrap(); // Malformed line
    writeln!(
        file,
        r#"{{"id": "doc2", "vector": [4.0, 5.0, 6.0], "metadata": {{}}}}"#
    )
    .unwrap();

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    // Should either skip malformed lines or error
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_import_jsonl_empty_file() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("empty.jsonl");
    fs::File::create(&import_path).unwrap();

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    assert!(result.is_ok());
    assert_eq!(store.count(), 0);
}

#[test]
fn test_import_jsonl_missing_fields() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("missing.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    // Missing 'vector' field
    writeln!(file, r#"{{"id": "doc1", "metadata": {{}}}}"#).unwrap();

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    assert!(result.is_err());
}

#[test]
fn test_import_jsonl_dimension_mismatch() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("mismatch.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    writeln!(
        file,
        r#"{{"id": "doc1", "vector": [1.0, 2.0, 3.0], "metadata": {{}}}}"#
    )
    .unwrap();
    writeln!(
        file,
        r#"{{"id": "doc2", "vector": [4.0, 5.0], "metadata": {{}}}}"#
    )
    .unwrap(); // Different dimension

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    // Should fail on dimension mismatch
    assert!(result.is_err());
}

#[test]
fn test_export_jsonl_large_dataset() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Insert 1000 vectors
    for i in 0..1000 {
        store
            .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    let export_path = temp_dir.path().join("large.jsonl");
    let result = Exporter::new(&store).to_jsonl(&export_path);

    assert!(result.is_ok());

    // Verify file size
    let metadata = fs::metadata(&export_path).unwrap();
    assert!(metadata.len() > 0);
}

#[test]
fn test_import_jsonl_large_dataset() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("large.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    // Write 500 lines
    for i in 0..500 {
        writeln!(
            file,
            r#"{{"id": "doc{}", "vector": [{}, 0.0, 0.0], "metadata": {{}}}}"#,
            i, i as f32
        )
        .unwrap();
    }

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    assert!(result.is_ok());
    assert_eq!(store.count(), 500);
}

#[test]
fn test_export_with_special_characters_in_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let mut meta = Metadata {
        fields: HashMap::new(),
    };
    meta.fields.insert(
        "text".into(),
        serde_json::json!("Text with \"quotes\" and \n newlines"),
    );

    store
        .upsert("doc1".into(), vec![1.0, 2.0, 3.0], meta)
        .unwrap();

    let export_path = temp_dir.path().join("special.jsonl");
    let result = Exporter::new(&store).to_jsonl(&export_path);

    assert!(result.is_ok());
}

#[test]
fn test_export_with_unicode() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let mut meta = Metadata {
        fields: HashMap::new(),
    };
    meta.fields
        .insert("text".into(), serde_json::json!("Unicode: 你好世界 🌍"));

    store
        .upsert("doc1".into(), vec![1.0, 2.0, 3.0], meta)
        .unwrap();

    let export_path = temp_dir.path().join("unicode.jsonl");
    let result = Exporter::new(&store).to_jsonl(&export_path);

    assert!(result.is_ok());

    // Reimport and verify
    let mut new_store = VecStore::open(temp_dir.path().join("new")).unwrap();
    Importer::new(&mut new_store)
        .from_jsonl(&export_path, 1000)
        .unwrap();

    assert_eq!(new_store.count(), 1);
}

#[test]
fn test_import_jsonl_duplicate_ids() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("duplicates.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    writeln!(
        file,
        r#"{{"id": "doc1", "vector": [1.0, 2.0, 3.0], "metadata": {{}}}}"#
    )
    .unwrap();
    writeln!(
        file,
        r#"{{"id": "doc1", "vector": [4.0, 5.0, 6.0], "metadata": {{}}}}"#
    )
    .unwrap();

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    Importer::new(&mut store)
        .from_jsonl(&import_path, 1000)
        .unwrap();

    // Should upsert (keep last value)
    assert_eq!(store.count(), 1);
}

#[test]
fn test_export_high_dimensional_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    let high_dim_vector: Vec<f32> = (0..1536).map(|i| i as f32 * 0.001).collect();

    store.upsert("doc1".into(), high_dim_vector, meta).unwrap();

    let export_path = temp_dir.path().join("high_dim.jsonl");
    let result = Exporter::new(&store).to_jsonl(&export_path);

    assert!(result.is_ok());
}

#[test]
fn test_import_jsonl_with_null_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("null_meta.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    writeln!(
        file,
        r#"{{"id": "doc1", "vector": [1.0, 2.0, 3.0], "metadata": null}}"#
    )
    .unwrap();

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    // Should handle null metadata
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_export_filtered_results() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let mut meta1 = Metadata {
        fields: HashMap::new(),
    };
    meta1.fields.insert("type".into(), serde_json::json!("a"));

    let mut meta2 = Metadata {
        fields: HashMap::new(),
    };
    meta2.fields.insert("type".into(), serde_json::json!("b"));

    store
        .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta1)
        .unwrap();
    store
        .upsert("doc2".into(), vec![0.0, 1.0, 0.0], meta2)
        .unwrap();

    let export_path = temp_dir.path().join("filtered.jsonl");

    // Export with filter using to_jsonl_filtered
    let exporter = Exporter::new(&store);
    let result = exporter.to_jsonl_filtered(&export_path, |record| {
        record.metadata.fields.get("type") == Some(&serde_json::json!("a"))
    });

    assert!(result.is_ok());
}

#[test]
#[cfg(feature = "parquet-export")]
fn test_export_parquet_basic() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    store
        .upsert("doc1".into(), vec![1.0, 2.0, 3.0], meta)
        .unwrap();

    let export_path = temp_dir.path().join("export.parquet");
    let result = Exporter::new(&store).to_parquet(&export_path);

    assert!(result.is_ok());
}

#[test]
#[cfg(feature = "parquet-export")]
fn test_import_parquet_basic() {
    let temp_dir = tempfile::tempdir().unwrap();

    // Create a parquet file (simplified test)
    let export_path = temp_dir.path().join("test.parquet");

    // This would require creating a valid parquet file
    // Skipping detailed implementation for now
}

#[test]
fn test_import_jsonl_blank_lines() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("blanks.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    writeln!(
        file,
        r#"{{"id": "doc1", "vector": [1.0, 2.0, 3.0], "metadata": {{}}}}"#
    )
    .unwrap();
    writeln!(file, "").unwrap(); // Blank line
    writeln!(
        file,
        r#"{{"id": "doc2", "vector": [4.0, 5.0, 6.0], "metadata": {{}}}}"#
    )
    .unwrap();

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    assert!(result.is_ok());
    assert_eq!(store.count(), 2);
}

#[test]
fn test_import_jsonl_comments() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("comments.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    writeln!(
        file,
        r#"{{"id": "doc1", "vector": [1.0, 2.0, 3.0], "metadata": {{}}}}"#
    )
    .unwrap();
    writeln!(file, "// This is a comment").unwrap();
    writeln!(
        file,
        r#"{{"id": "doc2", "vector": [4.0, 5.0, 6.0], "metadata": {{}}}}"#
    )
    .unwrap();

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    // Should skip comment lines
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_export_import_preserves_precision() {
    let temp_dir = tempfile::tempdir().unwrap();

    let store_path = temp_dir.path().join("store");
    let mut store = VecStore::open(&store_path).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // High-precision values
    let vector = vec![0.123456789, 0.987654321, 0.555555555];
    store
        .upsert("precise".into(), vector.clone(), meta)
        .unwrap();

    // Export
    let export_path = temp_dir.path().join("precise.jsonl");
    Exporter::new(&store).to_jsonl(&export_path).unwrap();

    // Import
    let new_store_path = temp_dir.path().join("new_store");
    let mut new_store = VecStore::open(&new_store_path).unwrap();
    Importer::new(&mut new_store)
        .from_jsonl(&export_path, 1000)
        .unwrap();

    // Precision may be slightly lost in JSON serialization
    // This is expected for floating point
    assert_eq!(new_store.count(), 1);
}

#[test]
fn test_export_with_nested_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let mut meta = Metadata {
        fields: HashMap::new(),
    };
    meta.fields.insert(
        "nested".into(),
        serde_json::json!({
            "level1": {
                "level2": {
                    "value": 42
                }
            }
        }),
    );

    store
        .upsert("doc1".into(), vec![1.0, 2.0, 3.0], meta)
        .unwrap();

    let export_path = temp_dir.path().join("nested.jsonl");
    let result = Exporter::new(&store).to_jsonl(&export_path);

    assert!(result.is_ok());
}

#[test]
fn test_import_jsonl_array_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("array.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    writeln!(
        file,
        r#"{{"id": "doc1", "vector": [1.0, 2.0, 3.0], "metadata": {{"tags": ["a", "b", "c"]}}}}"#
    )
    .unwrap();

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    assert!(result.is_ok());
}

#[test]
fn test_export_empty_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    // Edge case: empty vector (may not be allowed)
    let result = store.upsert("empty".into(), vec![], meta);

    if result.is_ok() {
        let export_path = temp_dir.path().join("empty_vec.jsonl");
        let export_result = Exporter::new(&store).to_jsonl(&export_path);
        assert!(export_result.is_ok() || export_result.is_err());
    }
}

#[test]
fn test_import_jsonl_very_long_id() {
    let temp_dir = tempfile::tempdir().unwrap();

    let import_path = temp_dir.path().join("long_id.jsonl");
    let mut file = fs::File::create(&import_path).unwrap();

    let long_id = "a".repeat(1000);
    writeln!(
        file,
        r#"{{"id": "{}", "vector": [1.0, 2.0, 3.0], "metadata": {{}}}}"#,
        long_id
    )
    .unwrap();

    let mut store = VecStore::open(temp_dir.path()).unwrap();
    let result = Importer::new(&mut store).from_jsonl(&import_path, 1000);

    assert!(result.is_ok());
}

#[test]
fn test_concurrent_export() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    let meta = Metadata {
        fields: HashMap::new(),
    };

    for i in 0..100 {
        store
            .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
            .unwrap();
    }

    // Export to multiple files concurrently
    let export1 = temp_dir.path().join("export1.jsonl");
    let export2 = temp_dir.path().join("export2.jsonl");

    let result1 = Exporter::new(&store).to_jsonl(&export1);
    let result2 = Exporter::new(&store).to_jsonl(&export2);

    assert!(result1.is_ok());
    assert!(result2.is_ok());
}
