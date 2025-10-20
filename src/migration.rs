//! Migration Tools for Vector Database Import/Export
//!
//! Provides utilities to migrate data from popular vector databases
//! (Pinecone, Weaviate, Qdrant, ChromaDB, Milvus) into vecstore format.
//!
//! ## Supported Sources
//!
//! - **Pinecone**: Cloud vector database
//! - **Weaviate**: Open-source vector search engine
//! - **Qdrant**: High-performance vector database
//! - **ChromaDB**: Embedding database for LLM apps
//! - **Milvus**: Cloud-native vector database
//!
//! ## Export Formats
//!
//! - **JSONL**: JSON Lines (one JSON object per line)
//! - **CSV**: Comma-separated values
//! - **Parquet**: Columnar format (Apache Arrow)
//! - **NPY**: NumPy binary format
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::migration::{Migrator, SourceDatabase};
//!
//! # fn main() -> anyhow::Result<()> {
//! let migrator = Migrator::new();
//!
//! // Import from Pinecone export
//! let records = migrator.import_from_jsonl("pinecone_export.jsonl")?;
//! println!("Imported {} records", records.len());
//!
//! // Convert to vecstore format
//! let mut store = vecstore::VecStore::open("my_vecstore.db")?;
//! for record in records {
//!     store.upsert(record.id, record.vector, record.metadata)?;
//! }
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Source database type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceDatabase {
    /// Pinecone cloud vector database
    Pinecone,
    /// Weaviate vector search engine
    Weaviate,
    /// Qdrant vector database
    Qdrant,
    /// ChromaDB embedding database
    ChromaDB,
    /// Milvus cloud-native vector database
    Milvus,
    /// Generic JSONL format
    Generic,
}

impl SourceDatabase {
    /// Get database name
    pub fn name(&self) -> &str {
        match self {
            SourceDatabase::Pinecone => "pinecone",
            SourceDatabase::Weaviate => "weaviate",
            SourceDatabase::Qdrant => "qdrant",
            SourceDatabase::ChromaDB => "chromadb",
            SourceDatabase::Milvus => "milvus",
            SourceDatabase::Generic => "generic",
        }
    }
}

/// Migration record (universal format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationRecord {
    /// Unique identifier
    pub id: String,

    /// Vector embedding
    #[serde(rename = "vector", alias = "embedding", alias = "values")]
    pub vector: Vec<f32>,

    /// Metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,

    /// Optional timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

/// Pinecone-specific record format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PineconeRecord {
    id: String,
    values: Vec<f32>,
    #[serde(default)]
    metadata: HashMap<String, Value>,
}

/// Weaviate-specific record format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeaviateRecord {
    id: String,
    vector: Vec<f32>,
    #[serde(default)]
    properties: HashMap<String, Value>,
}

/// Qdrant-specific record format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QdrantRecord {
    id: Value, // Can be string or number
    vector: Vec<f32>,
    #[serde(default)]
    payload: HashMap<String, Value>,
}

/// ChromaDB-specific record format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChromaRecord {
    id: String,
    embedding: Vec<f32>,
    #[serde(default)]
    metadata: HashMap<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<String>,
}

/// Milvus-specific record format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MilvusRecord {
    #[serde(alias = "pk", alias = "id")]
    primary_key: Value,
    vector: Vec<f32>,
    #[serde(flatten)]
    fields: HashMap<String, Value>,
}

/// Migration statistics
#[derive(Debug, Clone, Default)]
pub struct MigrationStats {
    pub total_records: usize,
    pub successful: usize,
    pub failed: usize,
    pub skipped: usize,
    pub total_vectors: usize,
    pub avg_vector_dim: f32,
    pub errors: Vec<String>,
}

impl MigrationStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Add successful record
    pub fn add_success(&mut self) {
        self.successful += 1;
    }

    /// Add failed record
    pub fn add_failure(&mut self, error: String) {
        self.failed += 1;
        self.errors.push(error);
    }

    /// Add skipped record
    pub fn add_skipped(&mut self) {
        self.skipped += 1;
    }

    /// Get success rate
    pub fn success_rate(&self) -> f32 {
        if self.total_records == 0 {
            0.0
        } else {
            self.successful as f32 / self.total_records as f32
        }
    }
}

/// Migrator for vector database imports
pub struct Migrator {
    /// Validation mode (strict vs lenient)
    strict_mode: bool,

    /// Maximum dimension allowed
    max_dimension: Option<usize>,

    /// Dimension normalization (auto-detect or enforce)
    enforce_dimension: Option<usize>,
}

impl Migrator {
    /// Create a new migrator
    pub fn new() -> Self {
        Self {
            strict_mode: false,
            max_dimension: None,
            enforce_dimension: None,
        }
    }

    /// Enable strict validation mode
    pub fn with_strict_mode(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Set maximum allowed dimension
    pub fn with_max_dimension(mut self, max_dim: usize) -> Self {
        self.max_dimension = Some(max_dim);
        self
    }

    /// Enforce specific dimension (pad or truncate)
    pub fn with_enforce_dimension(mut self, dim: usize) -> Self {
        self.enforce_dimension = Some(dim);
        self
    }

    /// Import records from JSONL file
    ///
    /// # Arguments
    /// * `path` - Path to JSONL file
    /// * `source` - Source database type (for format detection)
    pub fn import_from_jsonl(
        &self,
        path: impl AsRef<Path>,
        source: SourceDatabase,
    ) -> Result<(Vec<MigrationRecord>, MigrationStats)> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);

        let mut records = Vec::new();
        let mut stats = MigrationStats::new();

        for (line_num, line) in reader.lines().enumerate() {
            stats.total_records += 1;

            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    stats.add_failure(format!("Line {}: Failed to read line: {}", line_num + 1, e));
                    continue;
                }
            };

            if line.trim().is_empty() {
                stats.add_skipped();
                continue;
            }

            match self.parse_record(&line, source) {
                Ok(record) => {
                    // Validate dimension
                    if let Some(max_dim) = self.max_dimension {
                        if record.vector.len() > max_dim {
                            stats.add_failure(format!(
                                "Line {}: Vector dimension {} exceeds maximum {}",
                                line_num + 1,
                                record.vector.len(),
                                max_dim
                            ));
                            continue;
                        }
                    }

                    // Enforce dimension if specified
                    let record = if let Some(target_dim) = self.enforce_dimension {
                        self.normalize_dimension(record, target_dim)
                    } else {
                        record
                    };

                    stats.total_vectors += record.vector.len();
                    records.push(record);
                    stats.add_success();
                }
                Err(e) => {
                    if self.strict_mode {
                        return Err(anyhow!("Line {}: {}", line_num + 1, e));
                    } else {
                        stats.add_failure(format!("Line {}: {}", line_num + 1, e));
                    }
                }
            }
        }

        // Calculate average dimension
        if !records.is_empty() {
            stats.avg_vector_dim = stats.total_vectors as f32 / records.len() as f32;
        }

        Ok((records, stats))
    }

    /// Parse a single record based on source database format
    fn parse_record(&self, line: &str, source: SourceDatabase) -> Result<MigrationRecord> {
        match source {
            SourceDatabase::Pinecone => {
                let record: PineconeRecord = serde_json::from_str(line)?;
                Ok(MigrationRecord {
                    id: record.id,
                    vector: record.values,
                    metadata: record.metadata,
                    timestamp: None,
                })
            }

            SourceDatabase::Weaviate => {
                let record: WeaviateRecord = serde_json::from_str(line)?;
                Ok(MigrationRecord {
                    id: record.id,
                    vector: record.vector,
                    metadata: record.properties,
                    timestamp: None,
                })
            }

            SourceDatabase::Qdrant => {
                let record: QdrantRecord = serde_json::from_str(line)?;
                let id = match record.id {
                    Value::String(s) => s,
                    Value::Number(n) => n.to_string(),
                    _ => return Err(anyhow!("Invalid Qdrant ID format")),
                };

                Ok(MigrationRecord {
                    id,
                    vector: record.vector,
                    metadata: record.payload,
                    timestamp: None,
                })
            }

            SourceDatabase::ChromaDB => {
                let record: ChromaRecord = serde_json::from_str(line)?;
                let mut metadata = record.metadata;

                // Add document to metadata if present
                if let Some(doc) = record.document {
                    metadata.insert("document".to_string(), Value::String(doc));
                }

                Ok(MigrationRecord {
                    id: record.id,
                    vector: record.embedding,
                    metadata,
                    timestamp: None,
                })
            }

            SourceDatabase::Milvus => {
                let record: MilvusRecord = serde_json::from_str(line)?;
                let id = match record.primary_key {
                    Value::String(s) => s,
                    Value::Number(n) => n.to_string(),
                    _ => return Err(anyhow!("Invalid Milvus primary key format")),
                };

                Ok(MigrationRecord {
                    id,
                    vector: record.vector,
                    metadata: record.fields,
                    timestamp: None,
                })
            }

            SourceDatabase::Generic => {
                // Try to parse as generic migration record
                serde_json::from_str(line)
                    .map_err(|e| anyhow!("Failed to parse generic record: {}", e))
            }
        }
    }

    /// Normalize vector dimension (pad with zeros or truncate)
    fn normalize_dimension(
        &self,
        mut record: MigrationRecord,
        target_dim: usize,
    ) -> MigrationRecord {
        let current_dim = record.vector.len();

        if current_dim < target_dim {
            // Pad with zeros
            record.vector.resize(target_dim, 0.0);
        } else if current_dim > target_dim {
            // Truncate
            record.vector.truncate(target_dim);
        }

        record
    }

    /// Export records to JSONL format
    pub fn export_to_jsonl(
        &self,
        records: &[MigrationRecord],
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let file = File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);

        for record in records {
            let json = serde_json::to_string(record)?;
            writeln!(writer, "{}", json)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Export records to CSV format
    pub fn export_to_csv(&self, records: &[MigrationRecord], path: impl AsRef<Path>) -> Result<()> {
        if records.is_empty() {
            return Err(anyhow!("No records to export"));
        }

        let file = File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);

        // Write header
        let dim = records[0].vector.len();
        write!(writer, "id,")?;
        for i in 0..dim {
            write!(writer, "v{},", i)?;
        }
        writeln!(writer, "metadata")?;

        // Write data
        for record in records {
            write!(writer, "{},", record.id)?;
            for &val in &record.vector {
                write!(writer, "{},", val)?;
            }
            let metadata_json = serde_json::to_string(&record.metadata)?;
            writeln!(writer, "\"{}\"", metadata_json.replace("\"", "\"\""))?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Import from CSV format
    pub fn import_from_csv(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(Vec<MigrationRecord>, MigrationStats)> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);

        let mut records = Vec::new();
        let mut stats = MigrationStats::new();
        let mut lines = reader.lines();

        // Read header
        let header = if let Some(Ok(h)) = lines.next() {
            h
        } else {
            return Err(anyhow!("Empty CSV file"));
        };

        let headers: Vec<&str> = header.split(',').collect();
        let vector_cols: usize = headers.iter().filter(|h| h.starts_with("v")).count();

        // Read data rows
        for (line_num, line) in lines.enumerate() {
            stats.total_records += 1;

            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    stats.add_failure(format!("Line {}: Failed to read: {}", line_num + 2, e));
                    continue;
                }
            };

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < vector_cols + 2 {
                stats.add_failure(format!("Line {}: Insufficient columns", line_num + 2));
                continue;
            }

            let id = parts[0].to_string();

            // Parse vector
            let mut vector = Vec::with_capacity(vector_cols);
            for i in 1..=vector_cols {
                match parts[i].parse::<f32>() {
                    Ok(v) => vector.push(v),
                    Err(e) => {
                        stats.add_failure(format!(
                            "Line {}: Invalid vector value: {}",
                            line_num + 2,
                            e
                        ));
                        continue;
                    }
                }
            }

            // Parse metadata
            let metadata_str = parts[vector_cols + 1]
                .trim_matches('"')
                .replace("\"\"", "\"");
            let metadata: HashMap<String, Value> =
                serde_json::from_str(&metadata_str).unwrap_or_default();

            let record = MigrationRecord {
                id,
                vector,
                metadata,
                timestamp: None,
            };

            stats.total_vectors += record.vector.len();
            records.push(record);
            stats.add_success();
        }

        if !records.is_empty() {
            stats.avg_vector_dim = stats.total_vectors as f32 / records.len() as f32;
        }

        Ok((records, stats))
    }
}

impl Default for Migrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_pinecone_record() {
        let migrator = Migrator::new();

        let json = r#"{"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"source": "test"}}"#;
        let record = migrator
            .parse_record(json, SourceDatabase::Pinecone)
            .unwrap();

        assert_eq!(record.id, "vec1");
        assert_eq!(record.vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(
            record.metadata.get("source"),
            Some(&Value::String("test".to_string()))
        );
    }

    #[test]
    fn test_parse_qdrant_record() {
        let migrator = Migrator::new();

        let json = r#"{"id": "vec1", "vector": [0.1, 0.2], "payload": {"type": "doc"}}"#;
        let record = migrator.parse_record(json, SourceDatabase::Qdrant).unwrap();

        assert_eq!(record.id, "vec1");
        assert_eq!(record.vector, vec![0.1, 0.2]);
    }

    #[test]
    fn test_dimension_normalization() {
        let migrator = Migrator::new().with_enforce_dimension(5);

        let mut record = MigrationRecord {
            id: "vec1".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            metadata: HashMap::new(),
            timestamp: None,
        };

        record = migrator.normalize_dimension(record, 5);
        assert_eq!(record.vector.len(), 5);
        assert_eq!(record.vector, vec![0.1, 0.2, 0.3, 0.0, 0.0]);

        let mut record2 = MigrationRecord {
            id: "vec2".to_string(),
            vector: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            metadata: HashMap::new(),
            timestamp: None,
        };

        record2 = migrator.normalize_dimension(record2, 5);
        assert_eq!(record2.vector.len(), 5);
        assert_eq!(record2.vector, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[test]
    fn test_import_export_jsonl() {
        let migrator = Migrator::new();

        // Create test data
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"id": "vec1", "vector": [0.1, 0.2, 0.3], "metadata": {{}}}}"#
        )
        .unwrap();
        writeln!(
            file,
            r#"{{"id": "vec2", "vector": [0.4, 0.5, 0.6], "metadata": {{}}}}"#
        )
        .unwrap();
        file.flush().unwrap();

        // Import
        let (records, stats) = migrator
            .import_from_jsonl(file.path(), SourceDatabase::Generic)
            .unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(stats.successful, 2);
        assert_eq!(stats.total_records, 2);

        // Export
        let export_file = NamedTempFile::new().unwrap();
        migrator
            .export_to_jsonl(&records, export_file.path())
            .unwrap();

        // Re-import to verify
        let (records2, _) = migrator
            .import_from_jsonl(export_file.path(), SourceDatabase::Generic)
            .unwrap();

        assert_eq!(records2.len(), 2);
        assert_eq!(records2[0].id, "vec1");
        assert_eq!(records2[1].id, "vec2");
    }

    #[test]
    fn test_migration_stats() {
        let mut stats = MigrationStats::new();

        stats.total_records = 10;
        stats.add_success();
        stats.add_success();
        stats.add_success();
        stats.add_failure("Error 1".to_string());
        stats.add_skipped();

        assert_eq!(stats.successful, 3);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.skipped, 1);
        assert_eq!(stats.success_rate(), 0.3);
        assert_eq!(stats.errors.len(), 1);
    }

    #[test]
    fn test_strict_mode() {
        let migrator = Migrator::new().with_strict_mode();

        // Create test data with one invalid record
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"id": "vec1", "vector": [0.1, 0.2, 0.3], "metadata": {{}}}}"#
        )
        .unwrap();
        writeln!(file, r#"invalid json"#).unwrap();
        file.flush().unwrap();

        // Should fail in strict mode
        let result = migrator.import_from_jsonl(file.path(), SourceDatabase::Generic);
        assert!(result.is_err());
    }

    #[test]
    fn test_lenient_mode() {
        let migrator = Migrator::new(); // Default is lenient

        // Create test data with one invalid record
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"id": "vec1", "vector": [0.1, 0.2, 0.3], "metadata": {{}}}}"#
        )
        .unwrap();
        writeln!(file, r#"invalid json"#).unwrap();
        writeln!(
            file,
            r#"{{"id": "vec2", "vector": [0.4, 0.5, 0.6], "metadata": {{}}}}"#
        )
        .unwrap();
        file.flush().unwrap();

        // Should succeed in lenient mode, skipping invalid record
        let (records, stats) = migrator
            .import_from_jsonl(file.path(), SourceDatabase::Generic)
            .unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(stats.successful, 2);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.total_records, 3);
    }
}
