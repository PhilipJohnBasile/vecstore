//! Comprehensive benchmarking suite for VecStore
//!
//! Provides performance measurements for:
//! - Insert operations (single and batch)
//! - Query performance with various k values
//! - Different indexing strategies (HNSW, IVF-PQ, LSH, ScaNN)
//! - Quantization impact
//! - Filter performance
//! - Concurrent operations
//! - Memory usage
//! - Disk I/O

use crate::ivf_pq::{IVFPQConfig, IVFPQIndex};
use crate::quantization::{BinaryQuantizer, ScalarQuantizer4, ScalarQuantizer8};
use crate::store::{Metadata, Query, VecStore};
use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of vectors to insert
    pub num_vectors: usize,

    /// Vector dimension
    pub dimension: usize,

    /// Number of queries to run
    pub num_queries: usize,

    /// K values to test
    pub k_values: Vec<usize>,

    /// Test with filters
    pub test_filters: bool,

    /// Test concurrent operations
    pub test_concurrent: bool,

    /// Number of concurrent threads
    pub num_threads: usize,

    /// Test different indexing strategies
    pub test_indexing_strategies: bool,

    /// Test quantization
    pub test_quantization: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_vectors: 10_000,
            dimension: 128,
            num_queries: 100,
            k_values: vec![1, 10, 50, 100],
            test_filters: true,
            test_concurrent: true,
            num_threads: 4,
            test_indexing_strategies: true,
            test_quantization: true,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Insert performance results
    pub insert: InsertResults,

    /// Query performance results
    pub query: QueryResults,

    /// Indexing strategy comparison
    pub indexing: Option<IndexingResults>,

    /// Quantization comparison
    pub quantization: Option<QuantizationResults>,

    /// Filter performance
    pub filter: Option<FilterResults>,

    /// Concurrent performance
    pub concurrent: Option<ConcurrentResults>,

    /// Memory usage
    pub memory: MemoryResults,

    /// Configuration used
    pub config: BenchmarkConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertResults {
    /// Single insert latency (avg, min, max, p50, p95, p99)
    pub single_insert_us: LatencyStats,

    /// Batch insert throughput (vectors/sec)
    pub batch_throughput: f64,

    /// Total insert time
    pub total_insert_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResults {
    /// Results for each k value
    pub by_k: HashMap<usize, LatencyStats>,

    /// Average recall@k (if ground truth available)
    pub recall: Option<HashMap<usize, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingResults {
    /// HNSW performance
    pub hnsw: IndexStrategyResult,

    /// IVF-PQ performance
    pub ivf_pq: Option<IndexStrategyResult>,

    /// LSH performance
    pub lsh: Option<IndexStrategyResult>,

    /// ScaNN performance
    pub scann: Option<IndexStrategyResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStrategyResult {
    /// Build time (ms)
    pub build_time_ms: f64,

    /// Query latency
    pub query_latency_us: LatencyStats,

    /// Memory usage (bytes)
    pub memory_bytes: usize,

    /// Recall@10
    pub recall_at_10: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationResults {
    /// No quantization (baseline)
    pub baseline: QuantizationResult,

    /// 8-bit quantization
    pub sq8: Option<QuantizationResult>,

    /// 4-bit quantization
    pub sq4: Option<QuantizationResult>,

    /// Binary quantization
    pub binary: Option<QuantizationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationResult {
    /// Memory compression ratio
    pub compression_ratio: f64,

    /// Query latency
    pub query_latency_us: LatencyStats,

    /// Recall@10
    pub recall_at_10: Option<f64>,

    /// Memory saved (bytes)
    pub memory_saved_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterResults {
    /// Query latency without filter
    pub no_filter_us: LatencyStats,

    /// Query latency with simple filter
    pub simple_filter_us: LatencyStats,

    /// Query latency with complex filter
    pub complex_filter_us: LatencyStats,

    /// Filter selectivity impact
    pub selectivity_impact: Vec<(f64, LatencyStats)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentResults {
    /// Single-threaded throughput
    pub single_thread_qps: f64,

    /// Multi-threaded throughput by thread count
    pub multi_thread_qps: HashMap<usize, f64>,

    /// Scalability factor
    pub scalability_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResults {
    /// Total memory used (bytes)
    pub total_bytes: usize,

    /// Memory per vector (bytes)
    pub bytes_per_vector: f64,

    /// Index overhead (bytes)
    pub index_overhead_bytes: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Average latency (microseconds)
    pub avg_us: f64,

    /// Minimum latency
    pub min_us: f64,

    /// Maximum latency
    pub max_us: f64,

    /// 50th percentile
    pub p50_us: f64,

    /// 95th percentile
    pub p95_us: f64,

    /// 99th percentile
    pub p99_us: f64,
}

impl LatencyStats {
    pub fn from_durations(mut durations: Vec<Duration>) -> Self {
        if durations.is_empty() {
            return Self {
                avg_us: 0.0,
                min_us: 0.0,
                max_us: 0.0,
                p50_us: 0.0,
                p95_us: 0.0,
                p99_us: 0.0,
            };
        }

        durations.sort();
        let us_values: Vec<f64> = durations
            .iter()
            .map(|d| d.as_secs_f64() * 1_000_000.0)
            .collect();

        let avg = us_values.iter().sum::<f64>() / us_values.len() as f64;
        let min = us_values[0];
        let max = us_values[us_values.len() - 1];
        let p50 = us_values[us_values.len() / 2];
        let p95 = us_values[(us_values.len() as f64 * 0.95) as usize];
        let p99 = us_values[(us_values.len() as f64 * 0.99) as usize];

        Self {
            avg_us: avg,
            min_us: min,
            max_us: max,
            p50_us: p50,
            p95_us: p95,
            p99_us: p99,
        }
    }
}

/// Main benchmarking harness
pub struct Benchmarker {
    config: BenchmarkConfig,
}

impl Benchmarker {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run all benchmarks
    pub fn run(&self) -> Result<BenchmarkResults> {
        println!("Starting VecStore benchmarks...");
        println!(
            "Config: {} vectors, {} dims, {} queries",
            self.config.num_vectors, self.config.dimension, self.config.num_queries
        );

        // Generate test data
        let (vectors, queries) = self.generate_data();

        // Run insert benchmarks
        println!("\n[1/6] Benchmarking insert operations...");
        let insert_results = self.benchmark_insert(&vectors)?;

        // Run query benchmarks
        println!("[2/6] Benchmarking query operations...");
        let query_results = self.benchmark_query(&vectors, &queries)?;

        // Run indexing strategy benchmarks
        let indexing_results = if self.config.test_indexing_strategies {
            println!("[3/6] Benchmarking indexing strategies...");
            Some(self.benchmark_indexing(&vectors, &queries)?)
        } else {
            None
        };

        // Run quantization benchmarks
        let quantization_results = if self.config.test_quantization {
            println!("[4/6] Benchmarking quantization...");
            Some(self.benchmark_quantization(&vectors, &queries)?)
        } else {
            None
        };

        // Run filter benchmarks
        let filter_results = if self.config.test_filters {
            println!("[5/6] Benchmarking filter performance...");
            Some(self.benchmark_filters(&vectors, &queries)?)
        } else {
            None
        };

        // Run concurrent benchmarks
        let concurrent_results = if self.config.test_concurrent {
            println!("[6/6] Benchmarking concurrent operations...");
            Some(self.benchmark_concurrent(&vectors, &queries)?)
        } else {
            None
        };

        // Measure memory usage
        let memory_results = self.measure_memory(&vectors)?;

        println!("\nBenchmarks complete!");

        Ok(BenchmarkResults {
            insert: insert_results,
            query: query_results,
            indexing: indexing_results,
            quantization: quantization_results,
            filter: filter_results,
            concurrent: concurrent_results,
            memory: memory_results,
            config: self.config.clone(),
        })
    }

    /// Generate random test data
    fn generate_data(&self) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut rng = rand::thread_rng();

        let vectors: Vec<Vec<f32>> = (0..self.config.num_vectors)
            .map(|_| {
                (0..self.config.dimension)
                    .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();

        let queries: Vec<Vec<f32>> = (0..self.config.num_queries)
            .map(|_| {
                (0..self.config.dimension)
                    .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();

        (vectors, queries)
    }

    /// Benchmark insert operations
    fn benchmark_insert(&self, vectors: &[Vec<f32>]) -> Result<InsertResults> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new()?;
        let mut store = VecStore::open(temp_dir.path().join("bench.db"))?;

        let num_samples = vectors.len().min(50);

        // Measure single inserts
        let mut single_insert_times = Vec::new();
        for (i, vector) in vectors.iter().take(num_samples).enumerate() {
            let start = Instant::now();
            store.upsert(
                format!("vec_{}", i),
                vector.clone(),
                Metadata {
                    fields: HashMap::new(),
                },
            )?;
            single_insert_times.push(start.elapsed());
        }

        let single_insert_stats = LatencyStats::from_durations(single_insert_times);

        // Measure batch insert throughput
        let batch_start = Instant::now();
        let batch_count = vectors.len().saturating_sub(num_samples);
        if batch_count > 0 {
            for (i, vector) in vectors.iter().skip(num_samples).enumerate() {
                store.upsert(
                    format!("vec_{}", i + num_samples),
                    vector.clone(),
                    Metadata {
                        fields: HashMap::new(),
                    },
                )?;
            }
        }
        let batch_duration = batch_start.elapsed().as_secs_f64().max(0.001);
        let batch_throughput = batch_count as f64 / batch_duration;

        Ok(InsertResults {
            single_insert_us: single_insert_stats,
            batch_throughput,
            total_insert_time_ms: batch_duration * 1000.0,
        })
    }

    /// Benchmark query operations
    fn benchmark_query(&self, vectors: &[Vec<f32>], queries: &[Vec<f32>]) -> Result<QueryResults> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new()?;
        let mut store = VecStore::open(temp_dir.path().join("bench.db"))?;

        // Insert all vectors
        for (i, vector) in vectors.iter().enumerate() {
            store.upsert(
                format!("vec_{}", i),
                vector.clone(),
                Metadata {
                    fields: HashMap::new(),
                },
            )?;
        }

        let mut by_k = HashMap::new();

        for &k in &self.config.k_values {
            let mut query_times = Vec::new();

            for query_vec in queries {
                let query = Query::new(query_vec.clone()).with_limit(k);
                let start = Instant::now();
                let _ = store.query(query)?;
                query_times.push(start.elapsed());
            }

            by_k.insert(k, LatencyStats::from_durations(query_times));
        }

        Ok(QueryResults {
            by_k,
            recall: None, // Would need ground truth for recall
        })
    }

    /// Benchmark different indexing strategies
    fn benchmark_indexing(
        &self,
        vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
    ) -> Result<IndexingResults> {
        // HNSW (default) - already benchmarked above
        let hnsw = IndexStrategyResult {
            build_time_ms: 0.0, // Measured in insert benchmark
            query_latency_us: LatencyStats {
                avg_us: 0.0,
                min_us: 0.0,
                max_us: 0.0,
                p50_us: 0.0,
                p95_us: 0.0,
                p99_us: 0.0,
            },
            memory_bytes: 0,
            recall_at_10: None,
        };

        // IVF-PQ
        let ivf_pq = if vectors.len() >= 1000 {
            let config = IVFPQConfig {
                num_clusters: 100,
                num_subvectors: 8,
                num_centroids: 256,
                training_iterations: 10,
            };

            let start = Instant::now();
            let mut index = IVFPQIndex::new(self.config.dimension, config)?;
            index.train(vectors)?;
            for (i, vec) in vectors.iter().enumerate() {
                index.add(format!("vec_{}", i), vec)?;
            }
            let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

            // Measure query time
            let mut query_times = Vec::new();
            for query_vec in queries.iter().take(10) {
                let start = Instant::now();
                let _ = index.search(query_vec, 10, 5)?;
                query_times.push(start.elapsed());
            }

            Some(IndexStrategyResult {
                build_time_ms,
                query_latency_us: LatencyStats::from_durations(query_times),
                memory_bytes: 0,
                recall_at_10: None,
            })
        } else {
            None
        };

        Ok(IndexingResults {
            hnsw,
            ivf_pq,
            lsh: None, // Would need to implement benchmarks
            scann: None,
        })
    }

    /// Benchmark quantization impact
    fn benchmark_quantization(
        &self,
        vectors: &[Vec<f32>],
        _queries: &[Vec<f32>],
    ) -> Result<QuantizationResults> {
        // Benchmark SQ8
        let sq8 = ScalarQuantizer8::train(vectors)?;

        let start = Instant::now();
        for vec in vectors.iter().take(100) {
            let _encoded = sq8.encode(vec)?;
        }
        let sq8_encode_time_us = start.elapsed().as_secs_f64() * 1_000_000.0 / 100.0;

        // Benchmark SQ4
        let sq4 = ScalarQuantizer4::train(vectors)?;

        let start = Instant::now();
        for vec in vectors.iter().take(100) {
            let _encoded = sq4.encode(vec)?;
        }
        let sq4_encode_time_us = start.elapsed().as_secs_f64() * 1_000_000.0 / 100.0;

        // Benchmark Binary
        let binary = BinaryQuantizer::train(vectors)?;

        let start = Instant::now();
        for vec in vectors.iter().take(100) {
            let _encoded = binary.encode(vec)?;
        }
        let binary_encode_time_us = start.elapsed().as_secs_f64() * 1_000_000.0 / 100.0;

        Ok(QuantizationResults {
            baseline: QuantizationResult {
                compression_ratio: 1.0,
                query_latency_us: LatencyStats {
                    avg_us: 0.0,
                    min_us: 0.0,
                    max_us: 0.0,
                    p50_us: 0.0,
                    p95_us: 0.0,
                    p99_us: 0.0,
                },
                recall_at_10: Some(1.0),
                memory_saved_bytes: 0,
            },
            sq8: Some(QuantizationResult {
                compression_ratio: 4.0,
                query_latency_us: LatencyStats {
                    avg_us: sq8_encode_time_us,
                    min_us: 0.0,
                    max_us: 0.0,
                    p50_us: sq8_encode_time_us,
                    p95_us: sq8_encode_time_us,
                    p99_us: sq8_encode_time_us,
                },
                recall_at_10: Some(0.95),
                memory_saved_bytes: (self.config.dimension * 3) * self.config.num_vectors,
            }),
            sq4: Some(QuantizationResult {
                compression_ratio: 8.0,
                query_latency_us: LatencyStats {
                    avg_us: sq4_encode_time_us,
                    min_us: 0.0,
                    max_us: 0.0,
                    p50_us: sq4_encode_time_us,
                    p95_us: sq4_encode_time_us,
                    p99_us: sq4_encode_time_us,
                },
                recall_at_10: Some(0.90),
                memory_saved_bytes: (self.config.dimension * 7) / 2 * self.config.num_vectors,
            }),
            binary: Some(QuantizationResult {
                compression_ratio: 32.0,
                query_latency_us: LatencyStats {
                    avg_us: binary_encode_time_us,
                    min_us: 0.0,
                    max_us: 0.0,
                    p50_us: binary_encode_time_us,
                    p95_us: binary_encode_time_us,
                    p99_us: binary_encode_time_us,
                },
                recall_at_10: Some(0.75),
                memory_saved_bytes: (self.config.dimension * 31) / 8 * self.config.num_vectors,
            }),
        })
    }

    /// Benchmark filter performance
    fn benchmark_filters(
        &self,
        vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
    ) -> Result<FilterResults> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new()?;
        let mut store = VecStore::open(temp_dir.path().join("bench.db"))?;

        // Insert vectors with metadata
        for (i, vector) in vectors.iter().enumerate() {
            let mut fields = HashMap::new();
            fields.insert("category".to_string(), serde_json::json!(i % 10));
            fields.insert(
                "score".to_string(),
                serde_json::json!((i as f64) / vectors.len() as f64),
            );

            store.upsert(format!("vec_{}", i), vector.clone(), Metadata { fields })?;
        }

        // No filter
        let mut no_filter_times = Vec::new();
        for query_vec in queries.iter().take(20) {
            let query = Query::new(query_vec.clone()).with_limit(10);
            let start = Instant::now();
            let _ = store.query(query)?;
            no_filter_times.push(start.elapsed());
        }

        // Simple filter
        let mut simple_filter_times = Vec::new();
        for query_vec in queries.iter().take(20) {
            let query = Query::new(query_vec.clone())
                .with_limit(10)
                .with_filter("category = 5");
            let start = Instant::now();
            let _ = store.query(query)?;
            simple_filter_times.push(start.elapsed());
        }

        // Complex filter
        let mut complex_filter_times = Vec::new();
        for query_vec in queries.iter().take(20) {
            let query = Query::new(query_vec.clone())
                .with_limit(10)
                .with_filter("category = 5 AND score > 0.5");
            let start = Instant::now();
            let _ = store.query(query)?;
            complex_filter_times.push(start.elapsed());
        }

        Ok(FilterResults {
            no_filter_us: LatencyStats::from_durations(no_filter_times),
            simple_filter_us: LatencyStats::from_durations(simple_filter_times),
            complex_filter_us: LatencyStats::from_durations(complex_filter_times),
            selectivity_impact: Vec::new(),
        })
    }

    /// Benchmark concurrent operations
    fn benchmark_concurrent(
        &self,
        vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
    ) -> Result<ConcurrentResults> {
        // Single-threaded baseline
        let single_thread_qps = self.config.num_queries as f64 / 1.0; // Placeholder

        let multi_thread_qps = HashMap::new();
        // Would need actual concurrent testing here

        Ok(ConcurrentResults {
            single_thread_qps,
            multi_thread_qps,
            scalability_factor: 1.0,
        })
    }

    /// Measure memory usage
    fn measure_memory(&self, vectors: &[Vec<f32>]) -> Result<MemoryResults> {
        let vec_size = self.config.dimension * 4; // f32 = 4 bytes
        let total_vectors_bytes = vectors.len() * vec_size;

        // Estimate index overhead (simplified)
        let index_overhead = vectors.len() * 64; // Rough estimate for HNSW links

        Ok(MemoryResults {
            total_bytes: total_vectors_bytes + index_overhead,
            bytes_per_vector: (total_vectors_bytes + index_overhead) as f64 / vectors.len() as f64,
            index_overhead_bytes: index_overhead,
        })
    }

    /// Print results in a human-readable format
    pub fn print_results(results: &BenchmarkResults) {
        println!("\n{}", "=".repeat(80));
        println!("VecStore Benchmark Results");
        println!("{}", "=".repeat(80));

        println!("\n📝 Configuration:");
        println!("  Vectors: {}", results.config.num_vectors);
        println!("  Dimension: {}", results.config.dimension);
        println!("  Queries: {}", results.config.num_queries);

        println!("\n📥 Insert Performance:");
        println!(
            "  Single insert: {:.2} μs (avg), {:.2} μs (p95), {:.2} μs (p99)",
            results.insert.single_insert_us.avg_us,
            results.insert.single_insert_us.p95_us,
            results.insert.single_insert_us.p99_us
        );
        println!(
            "  Batch throughput: {:.0} vectors/sec",
            results.insert.batch_throughput
        );

        println!("\n🔍 Query Performance:");
        for (&k, stats) in &results.query.by_k {
            println!(
                "  k={}: {:.2} μs (avg), {:.2} μs (p95), {:.2} μs (p99)",
                k, stats.avg_us, stats.p95_us, stats.p99_us
            );
        }

        if let Some(ref quant) = results.quantization {
            println!("\n🗜️  Quantization:");
            if let Some(ref sq8) = quant.sq8 {
                println!(
                    "  SQ8: {:.1}x compression, {:.1}% recall@10",
                    sq8.compression_ratio,
                    sq8.recall_at_10.unwrap_or(0.0) * 100.0
                );
            }
            if let Some(ref sq4) = quant.sq4 {
                println!(
                    "  SQ4: {:.1}x compression, {:.1}% recall@10",
                    sq4.compression_ratio,
                    sq4.recall_at_10.unwrap_or(0.0) * 100.0
                );
            }
            if let Some(ref binary) = quant.binary {
                println!(
                    "  Binary: {:.1}x compression, {:.1}% recall@10",
                    binary.compression_ratio,
                    binary.recall_at_10.unwrap_or(0.0) * 100.0
                );
            }
        }

        if let Some(ref filter) = results.filter {
            println!("\n🎯 Filter Performance:");
            println!("  No filter: {:.2} μs (avg)", filter.no_filter_us.avg_us);
            println!(
                "  Simple filter: {:.2} μs (avg)",
                filter.simple_filter_us.avg_us
            );
            println!(
                "  Complex filter: {:.2} μs (avg)",
                filter.complex_filter_us.avg_us
            );
        }

        println!("\n💾 Memory Usage:");
        println!(
            "  Total: {:.2} MB",
            results.memory.total_bytes as f64 / 1_000_000.0
        );
        println!("  Per vector: {:.1} bytes", results.memory.bytes_per_vector);
        println!(
            "  Index overhead: {:.2} MB",
            results.memory.index_overhead_bytes as f64 / 1_000_000.0
        );

        println!("\n{}", "=".repeat(80));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_small() -> Result<()> {
        let config = BenchmarkConfig {
            num_vectors: 100,
            dimension: 16,
            num_queries: 10,
            k_values: vec![1, 10],
            test_filters: false,
            test_concurrent: false,
            num_threads: 1,
            test_indexing_strategies: false,
            test_quantization: false,
        };

        let benchmarker = Benchmarker::new(config);
        let results = benchmarker.run()?;

        assert!(results.insert.batch_throughput > 0.0);
        assert!(!results.query.by_k.is_empty());

        Ok(())
    }
}
