// Benchmark for WASM-compatible HNSW implementation
//
// This benchmark demonstrates the performance characteristics of the pure Rust HNSW
// implementation that works in WebAssembly environments.
//
// Run with: cargo bench --bench wasm_hnsw_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use vecstore::store::wasm_hnsw::WasmHnsw;
use vecstore::Distance;

fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasm_hnsw_insert");

    for size in [100, 1000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let vectors = generate_random_vectors(size, 128);

            b.iter(|| {
                let mut hnsw = WasmHnsw::with_params(128, Distance::Cosine, 16, 200);
                for (i, vector) in vectors.iter().enumerate() {
                    hnsw.insert(format!("v{}", i), vector.clone()).unwrap();
                }
                black_box(hnsw)
            });
        });
    }

    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasm_hnsw_search");

    for size in [1000, 10_000, 100_000].iter() {
        let mut hnsw = WasmHnsw::with_params(128, Distance::Cosine, 16, 200);
        let vectors = generate_random_vectors(*size, 128);

        // Build index
        for (i, vector) in vectors.iter().enumerate() {
            hnsw.insert(format!("v{}", i), vector.clone()).unwrap();
        }

        let query = generate_random_vectors(1, 128)[0].clone();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("k=10", size),
            &(hnsw, query.clone()),
            |b, (hnsw, query): &(WasmHnsw, Vec<f32>)| {
                b.iter(|| {
                    let results = hnsw.search(black_box(query), 10, 50).unwrap();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

fn bench_search_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasm_hnsw_search_k");

    let mut hnsw = WasmHnsw::with_params(128, Distance::Cosine, 16, 200);
    let vectors = generate_random_vectors(10_000, 128);

    // Build index
    for (i, vector) in vectors.iter().enumerate() {
        hnsw.insert(format!("v{}", i), vector.clone()).unwrap();
    }

    let query = generate_random_vectors(1, 128)[0].clone();

    for k in [1, 10, 50, 100].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, &k| {
            b.iter(|| {
                let results = hnsw.search(black_box(&query), k, 50).unwrap();
                black_box(results)
            });
        });
    }

    group.finish();
}

fn bench_search_varying_ef(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasm_hnsw_search_ef");

    let mut hnsw = WasmHnsw::with_params(128, Distance::Cosine, 16, 200);
    let vectors = generate_random_vectors(10_000, 128);

    // Build index
    for (i, vector) in vectors.iter().enumerate() {
        hnsw.insert(format!("v{}", i), vector.clone()).unwrap();
    }

    let query = generate_random_vectors(1, 128)[0].clone();

    for ef in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(ef), ef, |b, &ef| {
            b.iter(|| {
                let results = hnsw.search(black_box(&query), 10, ef).unwrap();
                black_box(results)
            });
        });
    }

    group.finish();
}

fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasm_hnsw_mixed_ops");

    group.bench_function("insert_and_search", |b| {
        let vectors = generate_random_vectors(1000, 128);

        b.iter(|| {
            let mut hnsw = WasmHnsw::with_params(128, Distance::Cosine, 16, 200);

            // Insert half the vectors
            for (i, vector) in vectors.iter().take(500).enumerate() {
                hnsw.insert(format!("v{}", i), vector.clone()).unwrap();
            }

            // Do some searches
            for i in 0..10 {
                let _ = hnsw.search(&vectors[i], 10, 50).unwrap();
            }

            // Insert more vectors
            for (i, vector) in vectors.iter().skip(500).enumerate() {
                hnsw.insert(format!("v{}", i + 500), vector.clone())
                    .unwrap();
            }

            // More searches
            for i in 0..10 {
                let _ = hnsw.search(&vectors[i + 10], 10, 50).unwrap();
            }

            black_box(hnsw)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_search,
    bench_search_varying_k,
    bench_search_varying_ef,
    bench_concurrent_operations
);
criterion_main!(benches);
