//! Command-line interface for VecStore
//!
//! Provides a complete CLI for managing vector databases, importing/exporting data,
//! running queries, and monitoring performance.
//!
//! # Commands
//!
//! - `info` - Show database information and statistics
//! - `query` - Search for similar vectors
//! - `insert` - Add vectors to the database
//! - `delete` - Remove vectors by ID
//! - `import` - Import vectors from JSON/CSV
//! - `export` - Export vectors to JSON/CSV
//! - `health` - Check database health
//! - `optimize` - Optimize database performance
//! - `benchmark` - Run performance benchmarks

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// TODO: Re-enable once API is updated to match current VecStore interface
// pub mod commands;

/// VecStore CLI - High-performance vector database tool
#[derive(Parser, Debug)]
#[command(name = "vecstore")]
#[command(about = "VecStore CLI - Manage vector databases from the command line", long_about = None)]
#[command(version)]
pub struct Cli {
    /// Path to the vector database
    #[arg(short, long, default_value = "vectors.db")]
    pub database: PathBuf,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Show database information and statistics
    Info {
        /// Show detailed statistics
        #[arg(short, long)]
        detailed: bool,
    },

    /// Search for similar vectors
    Query {
        /// Query vector as comma-separated floats (e.g., "0.1,0.2,0.3")
        #[arg(short, long)]
        vector: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value = "10")]
        limit: usize,

        /// Optional metadata filter (e.g., "category = 'tech'")
        #[arg(short, long)]
        filter: Option<String>,

        /// Output format (json, table, simple)
        #[arg(short, long, default_value = "table")]
        format: OutputFormat,
    },

    /// Insert a vector into the database
    Insert {
        /// Vector ID
        #[arg(short, long)]
        id: String,

        /// Vector values as comma-separated floats
        #[arg(short, long)]
        vector: String,

        /// Metadata as JSON (e.g., '{"category":"tech"}')
        #[arg(short, long)]
        metadata: Option<String>,
    },

    /// Delete vectors by ID or filter
    Delete {
        /// Vector IDs to delete (comma-separated)
        #[arg(short, long)]
        ids: Option<String>,

        /// Delete all vectors matching filter
        #[arg(short, long)]
        filter: Option<String>,

        /// Confirm deletion without prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Import vectors from a file
    Import {
        /// Input file path (JSON, JSONL, or CSV)
        #[arg(short, long)]
        file: PathBuf,

        /// File format (json, jsonl, csv, auto)
        #[arg(short = 't', long, default_value = "auto")]
        format: ImportFormat,

        /// Batch size for import
        #[arg(short, long, default_value = "1000")]
        batch_size: usize,

        /// Skip validation for faster import
        #[arg(long)]
        no_validate: bool,
    },

    /// Export vectors to a file
    Export {
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Export format (json, jsonl, csv)
        #[arg(short = 't', long, default_value = "json")]
        format: ExportFormat,

        /// Optional filter to limit export
        #[arg(short, long)]
        filter: Option<String>,

        /// Maximum vectors to export
        #[arg(short, long)]
        limit: Option<usize>,
    },

    /// Check database health and get recommendations
    Health {
        /// Run detailed health checks
        #[arg(short, long)]
        detailed: bool,

        /// Output format (json, table)
        #[arg(short, long, default_value = "table")]
        format: OutputFormat,
    },

    /// Optimize database performance
    Optimize {
        /// Run compaction
        #[arg(short, long)]
        compact: bool,

        /// Rebuild HNSW index
        #[arg(short, long)]
        rebuild_index: bool,

        /// Analyze and suggest optimizations
        #[arg(short, long)]
        analyze: bool,
    },

    /// Run performance benchmarks
    Benchmark {
        /// Number of vectors for benchmark
        #[arg(short, long, default_value = "10000")]
        vectors: usize,

        /// Vector dimensions
        #[arg(short, long, default_value = "128")]
        dimensions: usize,

        /// Number of queries to run
        #[arg(short, long, default_value = "1000")]
        queries: usize,

        /// Output format (json, table)
        #[arg(short = 'f', long, default_value = "table")]
        format: OutputFormat,
    },

    /// List all vectors or filter by criteria
    List {
        /// Filter expression
        #[arg(short, long)]
        filter: Option<String>,

        /// Maximum vectors to list
        #[arg(short, long, default_value = "100")]
        limit: usize,

        /// Show vector values
        #[arg(short, long)]
        show_vectors: bool,

        /// Output format (json, table, simple)
        #[arg(short = 'f', long, default_value = "table")]
        format: OutputFormat,
    },

    /// Create a new collection
    CreateCollection {
        /// Collection name
        name: String,

        /// Vector dimensions
        #[arg(short, long)]
        dimensions: usize,

        /// Distance metric (cosine, euclidean, dot)
        #[arg(short = 'm', long, default_value = "cosine")]
        metric: String,
    },

    /// Drop a collection
    DropCollection {
        /// Collection name
        name: String,

        /// Confirm deletion without prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    Table,
    Simple,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "table" => Ok(OutputFormat::Table),
            "simple" => Ok(OutputFormat::Simple),
            _ => Err(format!("Invalid format: {}. Use json, table, or simple", s)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportFormat {
    Json,
    Jsonl,
    Csv,
    Auto,
}

impl std::str::FromStr for ImportFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(ImportFormat::Json),
            "jsonl" => Ok(ImportFormat::Jsonl),
            "csv" => Ok(ImportFormat::Csv),
            "auto" => Ok(ImportFormat::Auto),
            _ => Err(format!(
                "Invalid format: {}. Use json, jsonl, csv, or auto",
                s
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Json,
    Jsonl,
    Csv,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(ExportFormat::Json),
            "jsonl" => Ok(ExportFormat::Jsonl),
            "csv" => Ok(ExportFormat::Csv),
            _ => Err(format!("Invalid format: {}. Use json, jsonl, or csv", s)),
        }
    }
}

/// Vector data for import/export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorData {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Helper to parse comma-separated floats
pub fn parse_vector(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|v| {
            v.trim()
                .parse::<f32>()
                .context(format!("Invalid float value: {}", v))
        })
        .collect()
}

/// Helper to parse comma-separated IDs
pub fn parse_ids(s: &str) -> Vec<String> {
    s.split(',').map(|id| id.trim().to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vector() {
        let vec = parse_vector("0.1,0.2,0.3").unwrap();
        assert_eq!(vec, vec![0.1, 0.2, 0.3]);

        let vec = parse_vector("1, 2, 3").unwrap();
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_ids() {
        let ids = parse_ids("doc1,doc2,doc3");
        assert_eq!(ids, vec!["doc1", "doc2", "doc3"]);

        let ids = parse_ids("doc1, doc2 , doc3 ");
        assert_eq!(ids, vec!["doc1", "doc2", "doc3"]);
    }

    #[test]
    fn test_output_format() {
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!(
            "table".parse::<OutputFormat>().unwrap(),
            OutputFormat::Table
        );
        assert_eq!(
            "simple".parse::<OutputFormat>().unwrap(),
            OutputFormat::Simple
        );
    }
}
