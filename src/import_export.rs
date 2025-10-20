//! Data import/export in standard formats
//!
//! This module provides efficient import and export capabilities for vecstore data
//! in industry-standard formats like JSONL and Parquet.
//!
//! ## Features
//!
//! - **JSONL (JSON Lines)**: Human-readable, streaming-friendly format
//! - **Parquet**: Columnar format with high compression (optional feature)
//! - **Streaming**: Process large datasets without loading into memory
//! - **Batch processing**: Efficient bulk imports/exports
//!
//! ## Usage
//!
//! ### Export to JSONL
//!
//! ```no_run
//! use vecstore::{VecStore, Query};
//! use vecstore::import_export::Exporter;
//!
//! # fn main() -> anyhow::Result<()> {
//! let store = VecStore::open("vectors.db")?;
//! let exporter = Exporter::new(&store);
//!
//! // Export all vectors to JSONL
//! exporter.to_jsonl("export.jsonl")?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Import from JSONL
//!
//! ```no_run
//! use vecstore::VecStore;
//! use vecstore::import_export::Importer;
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut store = VecStore::open("vectors.db")?;
//! let mut importer = Importer::new(&mut store);
//!
//! // Import from JSONL (batch mode)
//! let count = importer.from_jsonl("data.jsonl", 1000)?;
//! println!("Imported {} vectors", count);
//! # Ok(())
//! # }
//! ```

use crate::store::{Metadata, Record, VecStore};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Export record format (JSONL compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRecord {
    /// Vector ID
    pub id: String,

    /// Vector data
    pub vector: Vec<f32>,

    /// Metadata (JSON object)
    pub metadata: serde_json::Value,
}

impl From<Record> for ExportRecord {
    fn from(record: Record) -> Self {
        // Convert Metadata struct to JSON object
        let metadata_json = serde_json::json!(record.metadata.fields);

        Self {
            id: record.id,
            vector: record.vector,
            metadata: metadata_json,
        }
    }
}

/// Convert serde_json::Value to Metadata
fn value_to_metadata(value: serde_json::Value) -> Metadata {
    match value {
        serde_json::Value::Object(map) => {
            let fields = map.into_iter().collect();
            Metadata { fields }
        }
        _ => Metadata {
            fields: HashMap::new(),
        },
    }
}

/// Exporter for writing vecstore data to files
pub struct Exporter<'a> {
    store: &'a VecStore,
}

impl<'a> Exporter<'a> {
    /// Create a new exporter
    pub fn new(store: &'a VecStore) -> Self {
        Self { store }
    }

    /// Export all vectors to JSONL format
    ///
    /// # Arguments
    /// * `path` - Output file path
    ///
    /// # Returns
    /// Number of records exported
    pub fn to_jsonl<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        let file = File::create(path.as_ref())
            .with_context(|| format!("Failed to create file: {:?}", path.as_ref()))?;

        let mut writer = BufWriter::new(file);
        let mut count = 0;

        // Get all records from store
        let records = self.store.list_all();

        for record in records {
            let export_record = ExportRecord::from(record);
            let json = serde_json::to_string(&export_record)
                .context("Failed to serialize record to JSON")?;

            writeln!(writer, "{}", json).context("Failed to write JSONL line")?;
            count += 1;
        }

        writer.flush().context("Failed to flush writer")?;
        Ok(count)
    }

    /// Export vectors to JSONL with filtering
    ///
    /// # Arguments
    /// * `path` - Output file path
    /// * `filter_fn` - Predicate function to select records
    pub fn to_jsonl_filtered<P, F>(&self, path: P, filter_fn: F) -> Result<usize>
    where
        P: AsRef<Path>,
        F: Fn(&Record) -> bool,
    {
        let file = File::create(path.as_ref())
            .with_context(|| format!("Failed to create file: {:?}", path.as_ref()))?;

        let mut writer = BufWriter::new(file);
        let mut count = 0;

        let records = self.store.list_all();

        for record in records.into_iter().filter(|r| filter_fn(r)) {
            let export_record = ExportRecord::from(record);
            let json = serde_json::to_string(&export_record)?;

            writeln!(writer, "{}", json)?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }

    /// Export to Parquet format (requires parquet-export feature)
    #[cfg(feature = "parquet-export")]
    pub fn to_parquet<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        use arrow::array::{ArrayRef, Float32Array, ListArray, StringArray};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::basic::Compression;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        let records = self.store.list_all();

        if records.is_empty() {
            return Ok(0);
        }

        // Get vector dimension from first record
        let dim = records[0].vector.len();

        // Build Arrow schema
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new("metadata", DataType::Utf8, true),
        ]));

        // Create Parquet writer
        let file = File::create(path.as_ref())?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        // Process in batches (1000 records at a time)
        const BATCH_SIZE: usize = 1000;
        let mut total_count = 0;

        for chunk in records.chunks(BATCH_SIZE) {
            let mut ids = Vec::with_capacity(chunk.len());
            let mut vector_values = Vec::with_capacity(chunk.len() * dim);
            let mut vector_offsets = vec![0i32];
            let mut metadatas = Vec::with_capacity(chunk.len());

            for record in chunk {
                ids.push(record.id.clone());

                // Flatten vectors
                vector_values.extend_from_slice(&record.vector);
                vector_offsets.push(vector_offsets.last().unwrap() + record.vector.len() as i32);

                // Serialize metadata
                let metadata_str =
                    serde_json::to_string(&record.metadata).unwrap_or_else(|_| "{}".to_string());
                metadatas.push(metadata_str);
            }

            // Build arrays
            let id_array = Arc::new(StringArray::from(ids)) as ArrayRef;

            let vector_array = Arc::new(ListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                arrow::buffer::OffsetBuffer::new(vector_offsets.into()),
                Arc::new(Float32Array::from(vector_values)),
                None,
            )?) as ArrayRef;

            let metadata_array = Arc::new(StringArray::from(metadatas)) as ArrayRef;

            // Create record batch
            let batch =
                RecordBatch::try_new(schema.clone(), vec![id_array, vector_array, metadata_array])?;

            writer.write(&batch)?;
            total_count += chunk.len();
        }

        writer.close()?;
        Ok(total_count)
    }
}

/// Importer for reading data into vecstore
pub struct Importer<'a> {
    store: &'a mut VecStore,
}

impl<'a> Importer<'a> {
    /// Create a new importer
    pub fn new(store: &'a mut VecStore) -> Self {
        Self { store }
    }

    /// Import vectors from JSONL format
    ///
    /// # Arguments
    /// * `path` - Input file path
    /// * `batch_size` - Number of records to insert at once (0 = one at a time)
    ///
    /// # Returns
    /// Number of records imported
    pub fn from_jsonl<P: AsRef<Path>>(&mut self, path: P, batch_size: usize) -> Result<usize> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;

        let reader = BufReader::new(file);
        let mut count = 0;
        let mut batch = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;

            if line.trim().is_empty() {
                continue; // Skip empty lines
            }

            let record: ExportRecord = serde_json::from_str(&line)
                .with_context(|| format!("Failed to parse JSON on line {}", line_num + 1))?;

            if batch_size > 0 {
                batch.push(record);

                if batch.len() >= batch_size {
                    self.flush_batch(&mut batch)?;
                    count += batch_size;
                    batch.clear();
                }
            } else {
                // Insert immediately
                let metadata = value_to_metadata(record.metadata);
                self.store.upsert(record.id, record.vector, metadata)?;
                count += 1;
            }
        }

        // Flush remaining batch
        if !batch.is_empty() {
            let remaining = batch.len();
            self.flush_batch(&mut batch)?;
            count += remaining;
        }

        Ok(count)
    }

    /// Import from Parquet format (requires parquet-export feature)
    #[cfg(feature = "parquet-export")]
    pub fn from_parquet<P: AsRef<Path>>(&mut self, path: P) -> Result<usize> {
        use arrow::array::{Array, AsArray};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = File::open(path.as_ref())?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let mut reader = builder.build()?;

        let mut count = 0;

        while let Some(batch) = reader.next() {
            let batch = batch?;

            // Extract columns
            let id_array = batch.column(0).as_string::<i32>();

            let vector_array = batch.column(1).as_list::<i32>();

            let metadata_array = batch.column(2).as_string::<i32>();

            // Process each row
            for row_idx in 0..batch.num_rows() {
                let id = id_array.value(row_idx).to_string();

                // Extract vector
                let vector_list = vector_array.value(row_idx);
                let vector_data = vector_list
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .context("Expected Float32Array for vector data")?;

                let vector: Vec<f32> = (0..vector_data.len())
                    .map(|i| vector_data.value(i))
                    .collect();

                // Extract metadata
                let metadata_str = metadata_array.value(row_idx);
                let metadata_value: serde_json::Value =
                    serde_json::from_str(metadata_str).unwrap_or(serde_json::json!({}));
                let metadata = value_to_metadata(metadata_value);

                self.store.upsert(id, vector, metadata)?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Flush a batch of records to the store
    fn flush_batch(&mut self, batch: &mut Vec<ExportRecord>) -> Result<()> {
        for record in batch.drain(..) {
            let metadata = value_to_metadata(record.metadata);
            self.store.upsert(record.id, record.vector, metadata)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{NamedTempFile, TempDir};

    fn create_test_store() -> (VecStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();

        let mut meta1 = Metadata {
            fields: HashMap::new(),
        };
        meta1
            .fields
            .insert("title".into(), serde_json::json!("Document 1"));
        store
            .upsert("doc1".into(), vec![1.0, 2.0, 3.0], meta1)
            .unwrap();

        let mut meta2 = Metadata {
            fields: HashMap::new(),
        };
        meta2
            .fields
            .insert("title".into(), serde_json::json!("Document 2"));
        store
            .upsert("doc2".into(), vec![4.0, 5.0, 6.0], meta2)
            .unwrap();

        let mut meta3 = Metadata {
            fields: HashMap::new(),
        };
        meta3
            .fields
            .insert("title".into(), serde_json::json!("Document 3"));
        store
            .upsert("doc3".into(), vec![7.0, 8.0, 9.0], meta3)
            .unwrap();

        (store, temp_dir)
    }

    #[test]
    fn test_export_jsonl() {
        let (store, _temp_dir) = create_test_store();
        let exporter = Exporter::new(&store);

        let temp_file = NamedTempFile::new().unwrap();
        let count = exporter.to_jsonl(temp_file.path()).unwrap();

        assert_eq!(count, 3);

        // Verify file contents
        let file = File::open(temp_file.path()).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<_> = reader.lines().collect();

        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_import_jsonl() {
        let temp_file = NamedTempFile::new().unwrap();

        // Write test data
        {
            let mut writer = BufWriter::new(File::create(temp_file.path()).unwrap());
            writeln!(
                writer,
                r#"{{"id":"test1","vector":[1.0,2.0,3.0],"metadata":{{"key":"value"}}}}"#
            )
            .unwrap();
            writeln!(
                writer,
                r#"{{"id":"test2","vector":[4.0,5.0,6.0],"metadata":{{"key":"value2"}}}}"#
            )
            .unwrap();
        }

        let temp_dir = TempDir::new().unwrap();
        let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();
        let mut importer = Importer::new(&mut store);

        let count = importer.from_jsonl(temp_file.path(), 0).unwrap();
        assert_eq!(count, 2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_import_with_batching() {
        let temp_file = NamedTempFile::new().unwrap();

        // Write test data
        {
            let mut writer = BufWriter::new(File::create(temp_file.path()).unwrap());
            for i in 0..10 {
                writeln!(
                    writer,
                    r#"{{"id":"doc{}","vector":[{}.0,{}.0,{}.0],"metadata":{{"index":{}}}}}"#,
                    i,
                    i,
                    i + 1,
                    i + 2,
                    i
                )
                .unwrap();
            }
        }

        let temp_dir = TempDir::new().unwrap();
        let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();
        let mut importer = Importer::new(&mut store);

        let count = importer.from_jsonl(temp_file.path(), 5).unwrap();
        assert_eq!(count, 10);
        assert_eq!(store.len(), 10);
    }

    #[test]
    fn test_export_filtered() {
        let (store, _temp_dir) = create_test_store();
        let exporter = Exporter::new(&store);

        let temp_file = NamedTempFile::new().unwrap();

        // Export only records with IDs containing "1" or "2"
        let count = exporter
            .to_jsonl_filtered(temp_file.path(), |r| {
                r.id.contains("1") || r.id.contains("2")
            })
            .unwrap();

        assert_eq!(count, 2);
    }

    #[test]
    fn test_roundtrip() {
        let (store, _temp_dir) = create_test_store();
        let exporter = Exporter::new(&store);

        let temp_file = NamedTempFile::new().unwrap();

        // Export
        exporter.to_jsonl(temp_file.path()).unwrap();

        // Import into new store
        let temp_dir2 = TempDir::new().unwrap();
        let mut new_store = VecStore::open(temp_dir2.path().join("test.db")).unwrap();
        let mut importer = Importer::new(&mut new_store);
        importer.from_jsonl(temp_file.path(), 0).unwrap();

        assert_eq!(new_store.len(), store.len());
    }

    #[test]
    fn test_empty_lines_ignored() {
        let temp_file = NamedTempFile::new().unwrap();

        // Write data with empty lines
        {
            let mut writer = BufWriter::new(File::create(temp_file.path()).unwrap());
            writeln!(
                writer,
                r#"{{"id":"test1","vector":[1.0,2.0,3.0],"metadata":{{}}}}"#
            )
            .unwrap();
            writeln!(writer, "").unwrap(); // Empty line
            writeln!(
                writer,
                r#"{{"id":"test2","vector":[4.0,5.0,6.0],"metadata":{{}}}}"#
            )
            .unwrap();
        }

        let temp_dir = TempDir::new().unwrap();
        let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();
        let mut importer = Importer::new(&mut store);

        let count = importer.from_jsonl(temp_file.path(), 0).unwrap();
        assert_eq!(count, 2);
    }
}
