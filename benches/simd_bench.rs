use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use vecstore::simd::*;

fn generate_random_vector(dim: usize) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * 0.123) % 1.0).collect()
}

fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance");

    for size in [128, 384, 768, 1536].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = generate_random_vector(*size);
        let b = generate_random_vector(*size);

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bencher, _| {
            bencher.iter(|| black_box(euclidean_distance_simd(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [128, 384, 768, 1536].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = generate_random_vector(*size);
        let b = generate_random_vector(*size);

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bencher, _| {
            bencher.iter(|| black_box(dot_product_simd(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for size in [128, 384, 768, 1536].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = generate_random_vector(*size);
        let b = generate_random_vector(*size);

        group.bench_with_input(BenchmarkId::new("simd", size), size, |bencher, _| {
            bencher.iter(|| black_box(cosine_similarity_simd(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_euclidean_distance,
    bench_dot_product,
    bench_cosine_similarity
);
criterion_main!(benches);
