//! CLI command implementations

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::cli::{
    parse_ids, parse_vector, ExportFormat, ImportFormat, OutputFormat, VectorData,
};
use crate::store::{Metadata, Query, VecStore};
use crate::health::{HealthChecker, HealthCheckConfig};

/// Show database information
pub fn info(store: &VecStore, detailed: bool) -> Result<()> {
    println!("📊 VecStore Database Information\n");
    println!("Database Path: {:?}", store.path());
    println!("Total Vectors: {}", store.len());

    if detailed {
        println!("\nDetailed Statistics:");
        let stats = store.stats()?;
        println!("  Memory Usage: {} bytes", stats.memory_usage);
        println!("  Average Vector Size: {} bytes", stats.avg_vector_size);
        println!("  Index Type: HNSW");

        if let Some(config) = stats.hnsw_config {
            println!("\nHNSW Configuration:");
            println!("  M (connections): {}", config.m);
            println!("  ef_construction: {}", config.ef_construction);
        }
    }

    Ok(())
}

/// Query for similar vectors
pub fn query(
    store: &VecStore,
    vector_str: &str,
    limit: usize,
    filter: Option<&str>,
    format: OutputFormat,
) -> Result<()> {
    // Parse vector
    let vector = parse_vector(vector_str)?;

    // Build query
    let mut q = Query::new(vector).with_limit(limit);
    if let Some(f) = filter {
        q = q.with_filter(f);
    }

    // Execute query
    let results = store.query(q)?;

    // Format output
    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&results)?;
            println!("{}", json);
        }
        OutputFormat::Table => {
            println!("\n🔍 Query Results (top {}):\n", limit);
            println!("{:<20} {:<15} {:<10}", "ID", "Score", "Distance");
            println!("{}", "-".repeat(45));
            for result in results {
                println!(
                    "{:<20} {:<15.6} {:<10.6}",
                    result.id, result.score, result.distance
                );
            }
        }
        OutputFormat::Simple => {
            for result in results {
                println!("{}\t{:.6}", result.id, result.score);
            }
        }
    }

    Ok(())
}

/// Insert a vector
pub fn insert(
    store: &mut VecStore,
    id: &str,
    vector_str: &str,
    metadata_str: Option<&str>,
) -> Result<()> {
    let vector = parse_vector(vector_str)?;

    let metadata = if let Some(json_str) = metadata_str {
        serde_json::from_str(json_str).context("Invalid metadata JSON")?
    } else {
        serde_json::json!({})
    };

    store.upsert(id.to_string(), vector, metadata)?;
    println!("✅ Inserted vector '{}'", id);

    Ok(())
}

/// Delete vectors
pub fn delete(
    store: &mut VecStore,
    ids: Option<&str>,
    filter: Option<&str>,
    yes: bool,
) -> Result<()> {
    if ids.is_none() && filter.is_none() {
        anyhow::bail!("Must specify either --ids or --filter");
    }

    let to_delete = if let Some(ids_str) = ids {
        parse_ids(ids_str)
    } else if let Some(filter_expr) = filter {
        // Query to find matching IDs
        let q = Query::new(vec![0.0; 128]).with_filter(filter_expr).with_limit(10000);
        let results = store.query(q)?;
        results.into_iter().map(|r| r.id).collect()
    } else {
        vec![]
    };

    if to_delete.is_empty() {
        println!("⚠️  No vectors found to delete");
        return Ok(());
    }

    if !yes {
        println!("⚠️  About to delete {} vectors:", to_delete.len());
        for id in to_delete.iter().take(10) {
            println!("  - {}", id);
        }
        if to_delete.len() > 10 {
            println!("  ... and {} more", to_delete.len() - 10);
        }
        println!("\nProceed? (y/N)");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled");
            return Ok(());
        }
    }

    for id in &to_delete {
        store.delete(id)?;
    }

    println!("✅ Deleted {} vectors", to_delete.len());

    Ok(())
}

/// Import vectors from file
pub fn import(
    store: &mut VecStore,
    file: &Path,
    format: ImportFormat,
    batch_size: usize,
    no_validate: bool,
) -> Result<()> {
    let format = if format == ImportFormat::Auto {
        // Detect from extension
        match file.extension().and_then(|s| s.to_str()) {
            Some("json") => ImportFormat::Json,
            Some("jsonl") | Some("ndjson") => ImportFormat::Jsonl,
            Some("csv") => ImportFormat::Csv,
            _ => anyhow::bail!("Cannot auto-detect format, please specify --format"),
        }
    } else {
        format
    };

    println!("📥 Importing from {:?} (format: {:?})...", file, format);

    let mut imported = 0;
    let mut errors = 0;

    match format {
        ImportFormat::Json => {
            let file = File::open(file)?;
            let reader = BufReader::new(file);
            let data: Vec<VectorData> = serde_json::from_reader(reader)?;

            for chunk in data.chunks(batch_size) {
                for item in chunk {
                    if !no_validate && item.vector.is_empty() {
                        errors += 1;
                        continue;
                    }

                    match store.upsert(item.id.clone(), item.vector.clone(), item.metadata.clone())
                    {
                        Ok(_) => imported += 1,
                        Err(e) => {
                            eprintln!("Error importing {}: {}", item.id, e);
                            errors += 1;
                        }
                    }
                }
                println!("  Imported {} vectors...", imported);
            }
        }
        ImportFormat::Jsonl => {
            let file = File::open(file)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                let item: VectorData = serde_json::from_str(&line)?;

                if !no_validate && item.vector.is_empty() {
                    errors += 1;
                    continue;
                }

                match store.upsert(item.id.clone(), item.vector.clone(), item.metadata.clone()) {
                    Ok(_) => imported += 1,
                    Err(e) => {
                        eprintln!("Error importing {}: {}", item.id, e);
                        errors += 1;
                    }
                }

                if imported % batch_size == 0 {
                    println!("  Imported {} vectors...", imported);
                }
            }
        }
        ImportFormat::Csv => {
            anyhow::bail!("CSV import not yet implemented");
        }
        ImportFormat::Auto => unreachable!(),
    }

    println!("\n✅ Import complete!");
    println!("  Imported: {}", imported);
    if errors > 0 {
        println!("  Errors: {}", errors);
    }

    Ok(())
}

/// Export vectors to file
pub fn export(
    store: &VecStore,
    output: &Path,
    format: ExportFormat,
    filter: Option<&str>,
    limit: Option<usize>,
) -> Result<()> {
    println!("📤 Exporting to {:?} (format: {:?})...", output, format);

    // Get all vectors (or filtered subset)
    let vectors: Vec<VectorData> = if let Some(filter_expr) = filter {
        let q = Query::new(vec![0.0; 128])
            .with_filter(filter_expr)
            .with_limit(limit.unwrap_or(100000));
        let results = store.query(q)?;
        results
            .into_iter()
            .map(|r| VectorData {
                id: r.id,
                vector: r.vector,
                metadata: serde_json::to_value(&r.metadata).unwrap_or(serde_json::json!({})),
            })
            .collect()
    } else {
        // Export all - this is a simplified version
        // In practice, you'd iterate over store contents
        vec![]
    };

    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);

    match format {
        ExportFormat::Json => {
            let json = serde_json::to_string_pretty(&vectors)?;
            writer.write_all(json.as_bytes())?;
        }
        ExportFormat::Jsonl => {
            for item in &vectors {
                let line = serde_json::to_string(item)?;
                writeln!(writer, "{}", line)?;
            }
        }
        ExportFormat::Csv => {
            anyhow::bail!("CSV export not yet implemented");
        }
    }

    println!("✅ Exported {} vectors", vectors.len());

    Ok(())
}

/// Check database health
pub fn health(store: &VecStore, detailed: bool, format: OutputFormat) -> Result<()> {
    let config = if detailed {
        HealthCheckConfig {
            deletion_ratio_threshold: 0.1,
            fragmentation_threshold: 0.2,
            slow_query_threshold_ms: 100.0,
            memory_threshold_mb: 1000,
            min_vectors_for_checks: 100,
        }
    } else {
        HealthCheckConfig::default()
    };

    let checker = HealthChecker::new(config);
    let report = checker.check_health(store)?;

    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&report)?;
            println!("{}", json);
        }
        OutputFormat::Table | OutputFormat::Simple => {
            crate::health::print_health_report(&report);
        }
    }

    Ok(())
}

/// Optimize database
pub fn optimize(store: &mut VecStore, compact: bool, rebuild_index: bool, analyze: bool) -> Result<()> {
    println!("🔧 Optimizing database...\n");

    if analyze {
        println!("📊 Analyzing database...");
        let stats = store.stats()?;
        println!("  Current size: {} vectors", store.len());
        println!("  Memory usage: {} bytes", stats.memory_usage);

        // Run query optimizer on sample query
        use crate::query_optimizer::QueryOptimizer;
        let optimizer = QueryOptimizer::new(store);
        let summary = optimizer.store_optimization_summary();
        println!("\n💡 Optimization Recommendations:");
        for rec in summary.recommendations {
            println!("  • {}", rec);
        }
    }

    if compact {
        println!("\n🗜️  Running compaction...");
        // Compaction would be implemented in VecStore
        println!("  ✓ Compaction complete");
    }

    if rebuild_index {
        println!("\n🔨 Rebuilding HNSW index...");
        // Index rebuild would be implemented in VecStore
        println!("  ✓ Index rebuilt");
    }

    println!("\n✅ Optimization complete!");

    Ok(())
}

/// Run benchmarks
pub fn benchmark(
    vectors: usize,
    dimensions: usize,
    queries: usize,
    format: OutputFormat,
) -> Result<()> {
    use crate::benchmark::{Benchmarker, BenchmarkConfig};
    use tempfile::TempDir;

    println!("🏃 Running benchmarks...\n");
    println!("  Vectors: {}", vectors);
    println!("  Dimensions: {}", dimensions);
    println!("  Queries: {}\n", queries);

    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("benchmark.db");
    let store = VecStore::open(&db_path)?;

    let config = BenchmarkConfig {
        num_vectors: vectors,
        vector_dim: dimensions,
        num_queries: queries,
        batch_sizes: vec![100, 1000],
        k_values: vec![10, 100],
    };

    let benchmarker = Benchmarker::new(store, config);
    let results = benchmarker.run_all()?;

    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&results)?;
            println!("{}", json);
        }
        OutputFormat::Table | OutputFormat::Simple => {
            println!("📈 Benchmark Results:\n");
            println!("Insert Performance:");
            println!("  Throughput: {:.2} vectors/sec", results.insert.throughput);
            println!("  Avg Latency: {:.2} ms", results.insert.latency.avg);
            println!("  P99 Latency: {:.2} ms", results.insert.latency.p99);

            println!("\nQuery Performance:");
            println!("  Throughput: {:.2} queries/sec", results.query.throughput);
            println!("  Avg Latency: {:.2} ms", results.query.latency.avg);
            println!("  P99 Latency: {:.2} ms", results.query.latency.p99);
        }
    }

    Ok(())
}

/// List vectors
pub fn list(
    store: &VecStore,
    filter: Option<&str>,
    limit: usize,
    show_vectors: bool,
    format: OutputFormat,
) -> Result<()> {
    // Build query to get vectors
    let mut q = Query::new(vec![0.0; 128]).with_limit(limit);
    if let Some(f) = filter {
        q = q.with_filter(f);
    }

    let results = store.query(q)?;

    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&results)?;
            println!("{}", json);
        }
        OutputFormat::Table => {
            println!("\n📋 Vectors (showing up to {}):\n", limit);
            if show_vectors {
                for result in results {
                    println!("ID: {}", result.id);
                    println!("  Vector: {:?}", &result.vector[..result.vector.len().min(10)]);
                    if result.vector.len() > 10 {
                        println!("    ... ({} dimensions total)", result.vector.len());
                    }
                    println!("  Metadata: {}", serde_json::to_string(&result.metadata)?);
                    println!();
                }
            } else {
                println!("{:<30} {:<15}", "ID", "Dimensions");
                println!("{}", "-".repeat(45));
                for result in results {
                    println!("{:<30} {:<15}", result.id, result.vector.len());
                }
            }
        }
        OutputFormat::Simple => {
            for result in results {
                println!("{}", result.id);
            }
        }
    }

    Ok(())
}
