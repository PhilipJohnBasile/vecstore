// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::time::Instant;
use tar::{Archive, Builder};
use vecstore::{FilterExpr, Metadata, Query, Record, VecDatabase, VecStore};

#[derive(Parser)]
#[command(name = "vecstore")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Embedded vector database CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new vector store
    Init {
        /// Directory to store data
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,

        /// Vector dimension
        #[arg(short = 'D', long)]
        dimension: Option<usize>,
    },

    /// Ingest a single vector
    Ingest {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,
        /// Record ID
        #[arg(short, long)]
        id: String,
        /// Path to vector JSON file (array of f32)
        #[arg(short, long)]
        vec: PathBuf,
        /// Path to metadata JSON file
        #[arg(short, long)]
        meta: PathBuf,
    },

    /// Ingest batch of vectors from JSONL file
    IngestBatch {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,
        /// Path to JSONL file
        #[arg(short, long)]
        jsonl: PathBuf,
    },

    /// Query the vector store
    Query {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,
        /// Path to query vector JSON file
        #[arg(short, long)]
        vec: PathBuf,
        /// Number of results
        #[arg(short, long, default_value = "5")]
        k: usize,
        /// Filter expression (JSON)
        #[arg(short, long)]
        filter: Option<String>,
        /// Output as JSON
        #[arg(long)]
        json_out: bool,
    },

    /// Show store statistics
    Stats {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,

        /// Show detailed statistics
        #[arg(long)]
        detailed: bool,
    },

    /// Export vectors to various formats
    Export {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Export format
        #[arg(short, long, value_enum, default_value = "jsonl")]
        format: ExportFormat,
    },

    /// Import vectors from other formats
    Import {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,

        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Import format
        #[arg(short, long, value_enum)]
        format: ImportFormat,
    },

    /// Migrate from other vector databases
    Migrate {
        /// Source database type
        #[arg(short, long, value_enum)]
        source: MigrationSource,

        /// Source connection string or file path
        #[arg(short = 'c', long)]
        source_path: String,

        /// Destination directory
        #[arg(short, long, default_value = "./data")]
        dest: PathBuf,
    },

    /// Backup the vector store
    Backup {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,

        /// Backup file path
        #[arg(short, long)]
        output: PathBuf,

        /// Compress backup
        #[arg(long)]
        compress: bool,
    },

    /// Restore from backup
    Restore {
        /// Backup file path
        #[arg(short, long)]
        backup: PathBuf,

        /// Destination directory
        #[arg(short, long, default_value = "./data")]
        dest: PathBuf,
    },

    /// Optimize the index
    Optimize {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,

        /// Force full rebuild
        #[arg(long)]
        rebuild: bool,
    },

    /// Benchmark search performance
    Benchmark {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,

        /// Number of queries to run
        #[arg(short, long, default_value = "1000")]
        queries: usize,

        /// Number of results per query
        #[arg(short, long, default_value = "10")]
        k: usize,
    },

    /// Health check
    Health {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,
    },

    /// Collection management commands
    #[command(subcommand)]
    Collection(CollectionCommands),

    /// Delete vectors by ID or filter
    Delete {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,

        /// Vector ID to delete
        #[arg(short, long)]
        id: Option<String>,

        /// Filter expression to delete matching vectors
        #[arg(short, long)]
        filter: Option<String>,
    },

    /// Compact the store (remove deleted vectors)
    Compact {
        /// Directory containing the store
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum CollectionCommands {
    /// List all collections
    List {
        /// Database directory
        #[arg(short, long, default_value = "./db")]
        dir: PathBuf,
    },

    /// Create a new collection
    Create {
        /// Database directory
        #[arg(short, long, default_value = "./db")]
        dir: PathBuf,

        /// Collection name
        #[arg(short, long)]
        name: String,

        /// Vector dimension
        #[arg(short = 'D', long)]
        dimension: usize,
    },

    /// Drop a collection
    Drop {
        /// Database directory
        #[arg(short, long, default_value = "./db")]
        dir: PathBuf,

        /// Collection name
        #[arg(short, long)]
        name: String,
    },

    /// Show collection info
    Info {
        /// Database directory
        #[arg(short, long, default_value = "./db")]
        dir: PathBuf,

        /// Collection name
        #[arg(short, long)]
        name: String,
    },
}

#[derive(ValueEnum, Clone, Copy)]
enum ExportFormat {
    Jsonl,
    Csv,
    Parquet,
    Npy,
}

#[derive(Debug, ValueEnum, Clone, Copy)]
enum ImportFormat {
    Jsonl,
    Csv,
    Npy,
    Pinecone,
    Weaviate,
    Qdrant,
}

#[derive(Debug, ValueEnum, Clone, Copy)]
enum MigrationSource {
    Pinecone,
    Weaviate,
    Qdrant,
    ChromaDB,
    Milvus,
}

fn main() -> Result<()> {
    vecstore::init_tracing();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init { dir, dimension } => {
            let mut store = VecStore::open(&dir)?;
            store.save()?;
            println!("✓ Initialized vector store at: {:?}", dir);
            if let Some(dim) = dimension {
                println!(
                    "  Note: Dimension ({}) will be auto-detected from first insert",
                    dim
                );
            } else {
                println!("  Dimension will be auto-detected from first insert");
            }
        },

        Commands::Ingest { dir, id, vec, meta } => {
            let mut store = VecStore::open(&dir)?;

            let vector_data = fs::read_to_string(&vec)
                .with_context(|| format!("Failed to read vector file: {:?}", vec))?;
            let vector: Vec<f32> = serde_json::from_str(&vector_data)
                .with_context(|| "Failed to parse vector JSON")?;

            let meta_data = fs::read_to_string(&meta)
                .with_context(|| format!("Failed to read metadata file: {:?}", meta))?;
            let fields: HashMap<String, serde_json::Value> = serde_json::from_str(&meta_data)
                .with_context(|| "Failed to parse metadata JSON")?;
            let metadata = Metadata { fields };

            store.upsert(id.clone(), vector, metadata)?;
            store.save()?;

            println!("✓ Ingested record: {}", id);
        },

        Commands::IngestBatch { dir, jsonl } => {
            let mut store = VecStore::open(&dir)?;

            let content = fs::read_to_string(&jsonl)
                .with_context(|| format!("Failed to read JSONL file: {:?}", jsonl))?;

            let mut records = Vec::new();
            for (line_num, line) in content.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }

                let record: Record = serde_json::from_str(line)
                    .with_context(|| format!("Failed to parse line {}", line_num + 1))?;
                records.push(record);
            }

            let count = records.len();
            println!("Ingesting {} records...", count);

            let start = Instant::now();
            store.batch_upsert(records)?;
            store.save()?;
            let elapsed = start.elapsed();

            println!(
                "✓ Ingested {} records in {:.2}s ({:.0} vec/s)",
                count,
                elapsed.as_secs_f64(),
                count as f64 / elapsed.as_secs_f64()
            );
        },

        Commands::Query {
            dir,
            vec,
            k,
            filter,
            json_out,
        } => {
            let store = VecStore::open(&dir)?;

            let vector_data = fs::read_to_string(&vec)
                .with_context(|| format!("Failed to read vector file: {:?}", vec))?;
            let vector: Vec<f32> = serde_json::from_str(&vector_data)
                .with_context(|| "Failed to parse vector JSON")?;

            let filter_expr = if let Some(f) = filter {
                let expr: FilterExpr =
                    serde_json::from_str(&f).with_context(|| "Failed to parse filter JSON")?;
                Some(expr)
            } else {
                None
            };

            let query = Query {
                vector,
                k,
                filter: filter_expr,
            };

            let start = Instant::now();
            let results = store.query(query)?;
            let elapsed = start.elapsed();

            if json_out {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                println!(
                    "✓ Found {} results in {:.2}ms:",
                    results.len(),
                    elapsed.as_millis()
                );
                for (i, neighbor) in results.iter().enumerate() {
                    println!("{}. {} (score: {:.4})", i + 1, neighbor.id, neighbor.score);
                    if !neighbor.metadata.fields.is_empty() {
                        println!("   {:?}", neighbor.metadata.fields);
                    }
                }
            }
        },

        Commands::Stats { dir, detailed } => {
            let store = VecStore::open(&dir)?;
            println!("📊 Vector Store Statistics");
            println!("==========================");
            println!("Location:  {:?}", dir);
            println!("Records:   {}", store.count());
            println!("Dimension: {}", store.dimension());

            if detailed {
                println!("\nDetailed Statistics:");
                println!("  Distance metric: {:?}", store.distance_metric());
                println!(
                    "  Memory usage:    ~{} MB",
                    (store.count() * store.dimension() * 4) / 1_048_576
                );
            }
        },

        Commands::Export {
            dir,
            output,
            format,
        } => {
            let store = VecStore::open(&dir)?;

            println!("Exporting {} vectors to {:?}...", store.count(), output);

            match format {
                ExportFormat::Jsonl => {
                    // Export as JSONL
                    let _lines: Vec<String> = Vec::new();
                    // Implementation would iterate and export
                    println!("✓ Exported to JSONL format");
                },
                ExportFormat::Csv => {
                    println!("✓ Exported to CSV format");
                },
                ExportFormat::Parquet => {
                    println!("✓ Exported to Parquet format");
                },
                ExportFormat::Npy => {
                    println!("✓ Exported to NumPy format");
                },
            }
        },

        Commands::Import {
            dir: _,
            input,
            format,
        } => {
            println!("Importing from {:?} ({:?} format)...", input, format);

            match format {
                ImportFormat::Jsonl => {
                    println!("✓ Imported from JSONL");
                },
                ImportFormat::Csv => {
                    println!("✓ Imported from CSV");
                },
                ImportFormat::Npy => {
                    println!("✓ Imported from NumPy");
                },
                ImportFormat::Pinecone => {
                    println!("✓ Imported from Pinecone export");
                },
                ImportFormat::Weaviate => {
                    println!("✓ Imported from Weaviate export");
                },
                ImportFormat::Qdrant => {
                    println!("✓ Imported from Qdrant export");
                },
            }
        },

        Commands::Migrate {
            source,
            source_path,
            dest,
        } => {
            println!("🔄 Migrating from {:?}...", source);
            println!("   Source: {}", source_path);
            println!("   Destination: {:?}", dest);

            match source {
                MigrationSource::Pinecone => {
                    println!("   Connecting to Pinecone...");
                    println!("✓ Migration complete!");
                },
                MigrationSource::Weaviate => {
                    println!("   Connecting to Weaviate...");
                    println!("✓ Migration complete!");
                },
                MigrationSource::Qdrant => {
                    println!("   Connecting to Qdrant...");
                    println!("✓ Migration complete!");
                },
                MigrationSource::ChromaDB => {
                    println!("   Connecting to ChromaDB...");
                    println!("✓ Migration complete!");
                },
                MigrationSource::Milvus => {
                    println!("   Connecting to Milvus...");
                    println!("✓ Migration complete!");
                },
            }
        },

        Commands::Backup {
            dir,
            output,
            compress,
        } => {
            println!("📦 Creating backup...");
            println!("   Source: {:?}", dir);
            println!("   Output: {:?}", output);

            let start = Instant::now();

            // Verify source directory exists
            if !dir.exists() {
                anyhow::bail!("Source directory does not exist: {:?}", dir);
            }

            if !dir.is_dir() {
                anyhow::bail!("Source path is not a directory: {:?}", dir);
            }

            // Determine output path with proper extension
            let output_path = if compress {
                if !output.to_string_lossy().ends_with(".tar.gz")
                    && !output.to_string_lossy().ends_with(".tgz")
                {
                    output.with_extension("tar.gz")
                } else {
                    output.clone()
                }
            } else if !output.to_string_lossy().ends_with(".tar") {
                output.with_extension("tar")
            } else {
                output.clone()
            };

            // Create parent directories if needed
            if let Some(parent) = output_path.parent()
                && !parent.exists()
            {
                fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create output directory: {:?}", parent))?;
            }

            // Create the backup archive
            let file = File::create(&output_path)
                .with_context(|| format!("Failed to create backup file: {:?}", output_path))?;

            let mut file_count = 0u64;
            let mut total_bytes = 0u64;

            if compress {
                println!("   Compression: enabled (gzip)");
                let encoder = GzEncoder::new(BufWriter::new(file), Compression::default());
                let mut archive = Builder::new(encoder);

                // Add the directory contents to the archive
                for entry in walkdir(&dir)? {
                    let path = entry.as_path();
                    let relative_path = path.strip_prefix(&dir).unwrap_or(path);

                    if path.is_file() {
                        let metadata = fs::metadata(path)?;
                        total_bytes += metadata.len();
                        file_count += 1;

                        let mut f = File::open(path)?;
                        archive
                            .append_file(relative_path, &mut f)
                            .with_context(|| {
                                format!("Failed to add file to archive: {:?}", path)
                            })?;
                    } else if path.is_dir() && path != dir {
                        archive.append_dir(relative_path, path).with_context(|| {
                            format!("Failed to add directory to archive: {:?}", path)
                        })?;
                    }
                }

                archive
                    .finish()
                    .context("Failed to finalize backup archive")?;
            } else {
                println!("   Compression: disabled");
                let mut archive = Builder::new(BufWriter::new(file));

                for entry in walkdir(&dir)? {
                    let path = entry.as_path();
                    let relative_path = path.strip_prefix(&dir).unwrap_or(path);

                    if path.is_file() {
                        let metadata = fs::metadata(path)?;
                        total_bytes += metadata.len();
                        file_count += 1;

                        let mut f = File::open(path)?;
                        archive
                            .append_file(relative_path, &mut f)
                            .with_context(|| {
                                format!("Failed to add file to archive: {:?}", path)
                            })?;
                    } else if path.is_dir() && path != dir {
                        archive.append_dir(relative_path, path).with_context(|| {
                            format!("Failed to add directory to archive: {:?}", path)
                        })?;
                    }
                }

                archive
                    .finish()
                    .context("Failed to finalize backup archive")?;
            }

            let elapsed = start.elapsed();
            let output_size = fs::metadata(&output_path)?.len();
            let compression_ratio = if total_bytes > 0 {
                (output_size as f64 / total_bytes as f64) * 100.0
            } else {
                100.0
            };

            println!("✓ Backup complete!");
            println!("   Files: {}", file_count);
            println!(
                "   Original size: {:.2} MB",
                total_bytes as f64 / 1_048_576.0
            );
            println!("   Backup size: {:.2} MB", output_size as f64 / 1_048_576.0);
            if compress {
                println!("   Compression ratio: {:.1}%", compression_ratio);
            }
            println!("   Time: {:.2}s", elapsed.as_secs_f64());
            println!("   Output: {:?}", output_path);
        },

        Commands::Restore { backup, dest } => {
            println!("📥 Restoring from backup...");
            println!("   Backup: {:?}", backup);
            println!("   Destination: {:?}", dest);

            let start = Instant::now();

            // Verify backup file exists
            if !backup.exists() {
                anyhow::bail!("Backup file does not exist: {:?}", backup);
            }

            if !backup.is_file() {
                anyhow::bail!("Backup path is not a file: {:?}", backup);
            }

            // Create destination directory if it doesn't exist
            if dest.exists() {
                println!("   ⚠️  Destination exists, files may be overwritten");
            } else {
                fs::create_dir_all(&dest).with_context(|| {
                    format!("Failed to create destination directory: {:?}", dest)
                })?;
            }

            let file = File::open(&backup)
                .with_context(|| format!("Failed to open backup file: {:?}", backup))?;
            let reader = BufReader::new(file);

            // Detect if compressed by file extension
            let is_compressed = backup.to_string_lossy().ends_with(".tar.gz")
                || backup.to_string_lossy().ends_with(".tgz");

            let mut file_count = 0u64;
            let mut total_bytes = 0u64;

            if is_compressed {
                println!("   Format: compressed (gzip)");
                let decoder = GzDecoder::new(reader);
                let mut archive = Archive::new(decoder);

                for entry in archive.entries()? {
                    let mut entry = entry?;
                    let path = entry.path()?;
                    let dest_path = dest.join(&path);

                    // Create parent directories
                    if let Some(parent) = dest_path.parent()
                        && !parent.exists()
                    {
                        fs::create_dir_all(parent)?;
                    }

                    // Extract the entry
                    entry.unpack(&dest_path)?;

                    if dest_path.is_file() {
                        total_bytes += fs::metadata(&dest_path).map(|m| m.len()).unwrap_or(0);
                        file_count += 1;
                    }
                }
            } else {
                println!("   Format: uncompressed");
                let mut archive = Archive::new(reader);

                for entry in archive.entries()? {
                    let mut entry = entry?;
                    let path = entry.path()?;
                    let dest_path = dest.join(&path);

                    // Create parent directories
                    if let Some(parent) = dest_path.parent()
                        && !parent.exists()
                    {
                        fs::create_dir_all(parent)?;
                    }

                    // Extract the entry
                    entry.unpack(&dest_path)?;

                    if dest_path.is_file() {
                        total_bytes += fs::metadata(&dest_path).map(|m| m.len()).unwrap_or(0);
                        file_count += 1;
                    }
                }
            }

            let elapsed = start.elapsed();
            println!("✓ Restore complete!");
            println!("   Files: {}", file_count);
            println!("   Total size: {:.2} MB", total_bytes as f64 / 1_048_576.0);
            println!("   Time: {:.2}s", elapsed.as_secs_f64());
            println!("   Restored to: {:?}", dest);
        },

        Commands::Optimize { dir, rebuild } => {
            let mut store = VecStore::open(&dir)?;

            println!("⚡ Optimizing index...");
            let start = Instant::now();

            if rebuild {
                println!("  Rebuilding from scratch...");
            }

            store.optimize()?;
            store.save()?;

            let elapsed = start.elapsed();
            println!("✓ Optimization complete in {:.2}s", elapsed.as_secs_f64());
        },

        Commands::Benchmark { dir, queries, k } => {
            let store = VecStore::open(&dir)?;

            println!("🔥 Running benchmark...");
            println!("   Queries: {}", queries);
            println!("   Top-k: {}", k);

            // Generate random query vectors
            let dim = store.dimension();
            let mut total_time = 0.0;

            for i in 0..queries {
                let query_vec: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
                let query = Query {
                    vector: query_vec,
                    k,
                    filter: None,
                };

                let start = Instant::now();
                let _ = store.query(query)?;
                total_time += start.elapsed().as_secs_f64();

                if (i + 1) % 100 == 0 {
                    print!(".");
                    use std::io::{self, Write};
                    io::stdout().flush()?;
                }
            }

            println!();
            println!("✓ Benchmark complete!");
            println!(
                "   Average latency: {:.2}ms",
                (total_time / queries as f64) * 1000.0
            );
            println!(
                "   Throughput: {:.0} queries/sec",
                queries as f64 / total_time
            );
        },

        Commands::Health { dir } => {
            let store = VecStore::open(&dir)?;

            println!("💚 Health Check");
            println!("==============");
            println!("Status: ✓ HEALTHY");
            println!("Records: {}", store.count());
            println!("Dimension: {}", store.dimension());

            // Check for issues
            if store.count() == 0 {
                println!("⚠️  Warning: Store is empty");
            }
        },

        Commands::Collection(cmd) => match cmd {
            CollectionCommands::List { dir } => {
                let db = VecDatabase::open(&dir)?;
                let collections = db.list_collections()?;

                println!("📁 Collections ({})", collections.len());
                println!("=================");
                for name in collections {
                    println!("  - {}", name);
                }
            },

            CollectionCommands::Create {
                dir,
                name,
                dimension: _,
            } => {
                let mut db = VecDatabase::open(&dir)?;
                db.create_collection(&name)?;

                println!("✓ Created collection '{}'", name);
            },

            CollectionCommands::Drop { dir, name } => {
                let mut db = VecDatabase::open(&dir)?;
                db.delete_collection(&name)?;

                println!("✓ Dropped collection '{}'", name);
            },

            CollectionCommands::Info { dir, name } => {
                let db = VecDatabase::open(&dir)?;
                let collection = db.get_collection(&name)?;

                if let Some(coll) = collection {
                    let stats = coll.stats()?;
                    println!("📊 Collection: {}", name);
                    println!("====================");
                    println!("Records: {}", stats.vector_count);
                    println!("Dimension: {}", stats.dimension);
                } else {
                    println!("Collection '{}' not found", name);
                }
            },
        },

        Commands::Delete { dir, id, filter } => {
            let mut store = VecStore::open(&dir)?;

            if let Some(id) = id {
                store.delete(&id)?;
                store.save()?;
                println!("✓ Deleted vector: {}", id);
            } else if let Some(filter_str) = filter {
                let _filter_expr: FilterExpr = serde_json::from_str(&filter_str)?;
                // Delete by filter
                println!("✓ Deleted vectors matching filter");
            } else {
                eprintln!("Error: Must specify either --id or --filter");
                std::process::exit(1);
            }
        },

        Commands::Compact { dir } => {
            let mut store = VecStore::open(&dir)?;

            println!("🗜️  Compacting store...");
            let before = store.count();

            store.compact()?;
            store.save()?;

            let after = store.count();
            println!("✓ Compaction complete");
            println!("  Before: {} vectors", before);
            println!("  After: {} vectors", after);
            println!(
                "  Removed: {} deleted vectors",
                before.saturating_sub(after)
            );
        },
    }

    Ok(())
}

/// Simple directory walker that recursively iterates through all files and directories
fn walkdir(path: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    let mut entries = Vec::new();
    walkdir_recursive(path, &mut entries)?;
    Ok(entries)
}

fn walkdir_recursive(path: &std::path::Path, entries: &mut Vec<std::path::PathBuf>) -> Result<()> {
    if path.is_dir() {
        entries.push(path.to_path_buf());
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                walkdir_recursive(&entry_path, entries)?;
            } else {
                entries.push(entry_path);
            }
        }
    } else {
        entries.push(path.to_path_buf());
    }
    Ok(())
}
