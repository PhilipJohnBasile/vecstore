// Performance Benchmarks for OpenAI Embeddings Backend
//
// This benchmark suite measures the performance of various operations
// in the OpenAI embeddings backend, particularly focusing on operations
// that don't require actual API calls.
//
// Run with: cargo bench --features "embeddings,openai-embeddings" --bench openai_embeddings_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};

// Helper to create embedder
fn create_embedder(model: OpenAIModel) -> OpenAIEmbedding {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        OpenAIEmbedding::new("test-api-key".to_string(), model)
            .await
            .unwrap()
    })
}

// Benchmark: Cost estimation for various batch sizes
fn bench_cost_estimation_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cost_estimation_batch_sizes");

    let embedder = create_embedder(OpenAIModel::TextEmbedding3Small);
    let sample_text = "This is a sample text for benchmarking cost estimation performance.";

    for size in [1, 10, 100, 1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let texts: Vec<&str> = vec![sample_text; size];
            b.iter(|| embedder.estimate_cost(black_box(&texts)));
        });
    }

    group.finish();
}

// Benchmark: Cost estimation for various text lengths
fn bench_cost_estimation_text_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("cost_estimation_text_lengths");

    let embedder = create_embedder(OpenAIModel::TextEmbedding3Small);

    for length in [10, 100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Bytes(*length as u64));
        group.bench_with_input(BenchmarkId::from_parameter(length), length, |b, &length| {
            let text = "a".repeat(length);
            let texts = vec![text.as_str()];
            b.iter(|| embedder.estimate_cost(black_box(&texts)));
        });
    }

    group.finish();
}

// Benchmark: Cost estimation for different models
fn bench_cost_estimation_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("cost_estimation_models");

    let small = create_embedder(OpenAIModel::TextEmbedding3Small);
    let large = create_embedder(OpenAIModel::TextEmbedding3Large);
    let ada = create_embedder(OpenAIModel::Ada002);

    let sample_texts: Vec<&str> = vec!["Sample text 1", "Sample text 2", "Sample text 3"];

    group.bench_function("text-embedding-3-small", |b| {
        b.iter(|| small.estimate_cost(black_box(&sample_texts)))
    });

    group.bench_function("text-embedding-3-large", |b| {
        b.iter(|| large.estimate_cost(black_box(&sample_texts)))
    });

    group.bench_function("ada-002", |b| {
        b.iter(|| ada.estimate_cost(black_box(&sample_texts)))
    });

    group.finish();
}

// Benchmark: Model property access
fn bench_model_properties(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_properties");

    let embedder = create_embedder(OpenAIModel::TextEmbedding3Small);

    group.bench_function("dimension", |b| b.iter(|| embedder.model().dimension()));

    group.bench_function("cost_per_million_tokens", |b| {
        b.iter(|| embedder.model().cost_per_million_tokens())
    });

    group.bench_function("as_str", |b| b.iter(|| embedder.model().as_str()));

    group.finish();
}

// Benchmark: Embedder creation
fn bench_embedder_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedder_creation");

    let runtime = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("new", |b| {
        b.iter(|| {
            runtime.block_on(async {
                OpenAIEmbedding::new(
                    black_box("test-api-key".to_string()),
                    black_box(OpenAIModel::TextEmbedding3Small),
                )
                .await
                .unwrap()
            })
        })
    });

    group.finish();
}

// Benchmark: Builder pattern operations
fn bench_builder_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("builder_pattern");

    let runtime = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("with_rate_limit", |b| {
        b.iter(|| {
            let embedder = runtime.block_on(async {
                OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
                    .await
                    .unwrap()
            });
            embedder.with_rate_limit(black_box(100))
        })
    });

    group.bench_function("with_max_retries", |b| {
        b.iter(|| {
            let embedder = runtime.block_on(async {
                OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
                    .await
                    .unwrap()
            });
            embedder.with_max_retries(black_box(5))
        })
    });

    group.bench_function("chained_builders", |b| {
        b.iter(|| {
            let embedder = runtime.block_on(async {
                OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
                    .await
                    .unwrap()
            });
            embedder
                .with_rate_limit(black_box(100))
                .with_max_retries(black_box(5))
        })
    });

    group.finish();
}

// Benchmark: Cost estimation with varying text counts in 2048 boundary
fn bench_cost_estimation_batch_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("cost_estimation_batch_boundary");

    let embedder = create_embedder(OpenAIModel::TextEmbedding3Small);
    let sample_text = "Sample text for batch boundary testing.";

    // Test around the 2048 OpenAI API batch limit
    for size in [2000, 2048, 2049, 2100, 4096, 4097].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let texts: Vec<&str> = vec![sample_text; size];
            b.iter(|| embedder.estimate_cost(black_box(&texts)));
        });
    }

    group.finish();
}

// Benchmark: Cost estimation with unicode text
fn bench_cost_estimation_unicode(c: &mut Criterion) {
    let mut group = c.benchmark_group("cost_estimation_unicode");

    let embedder = create_embedder(OpenAIModel::TextEmbedding3Small);

    let ascii_text = "This is ASCII text for benchmarking.";
    let unicode_text = "これは日本語のテキストです。🎌";
    let mixed_text = "Mixed ASCII and Unicode: こんにちは世界！ Hello world! 🌍";

    group.bench_function("ascii", |b| {
        let texts = vec![ascii_text; 100];
        b.iter(|| embedder.estimate_cost(black_box(&texts)))
    });

    group.bench_function("unicode", |b| {
        let texts = vec![unicode_text; 100];
        b.iter(|| embedder.estimate_cost(black_box(&texts)))
    });

    group.bench_function("mixed", |b| {
        let texts = vec![mixed_text; 100];
        b.iter(|| embedder.estimate_cost(black_box(&texts)))
    });

    group.finish();
}

// Benchmark: Cost estimation with empty and whitespace
fn bench_cost_estimation_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("cost_estimation_edge_cases");

    let embedder = create_embedder(OpenAIModel::TextEmbedding3Small);

    group.bench_function("empty_strings", |b| {
        let texts: Vec<&str> = vec![""; 1000];
        b.iter(|| embedder.estimate_cost(black_box(&texts)))
    });

    group.bench_function("whitespace_only", |b| {
        let texts: Vec<&str> = vec!["   "; 1000];
        b.iter(|| embedder.estimate_cost(black_box(&texts)))
    });

    group.bench_function("single_chars", |b| {
        let texts: Vec<&str> = vec!["a"; 1000];
        b.iter(|| embedder.estimate_cost(black_box(&texts)))
    });

    group.finish();
}

// Benchmark: Cost estimation scalability
fn bench_cost_estimation_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("cost_estimation_scalability");
    group.sample_size(50); // Reduce sample size for large benchmarks

    let embedder = create_embedder(OpenAIModel::TextEmbedding3Small);
    let sample_text = "Sample text for scalability testing.";

    // Test very large batch sizes to see scalability
    for size in [10000, 50000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let texts: Vec<&str> = vec![sample_text; size];
            b.iter(|| embedder.estimate_cost(black_box(&texts)));
        });
    }

    group.finish();
}

// Benchmark: Model enum operations
fn bench_model_enum_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_enum_operations");

    group.bench_function("dimension_small", |b| {
        b.iter(|| OpenAIModel::TextEmbedding3Small.dimension())
    });

    group.bench_function("dimension_large", |b| {
        b.iter(|| OpenAIModel::TextEmbedding3Large.dimension())
    });

    group.bench_function("cost_small", |b| {
        b.iter(|| OpenAIModel::TextEmbedding3Small.cost_per_million_tokens())
    });

    group.bench_function("cost_large", |b| {
        b.iter(|| OpenAIModel::TextEmbedding3Large.cost_per_million_tokens())
    });

    group.bench_function("as_str_small", |b| {
        b.iter(|| OpenAIModel::TextEmbedding3Small.as_str())
    });

    group.bench_function("as_str_large", |b| {
        b.iter(|| OpenAIModel::TextEmbedding3Large.as_str())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cost_estimation_batch_sizes,
    bench_cost_estimation_text_lengths,
    bench_cost_estimation_models,
    bench_model_properties,
    bench_embedder_creation,
    bench_builder_pattern,
    bench_cost_estimation_batch_boundary,
    bench_cost_estimation_unicode,
    bench_cost_estimation_edge_cases,
    bench_cost_estimation_scalability,
    bench_model_enum_operations,
);

criterion_main!(benches);
