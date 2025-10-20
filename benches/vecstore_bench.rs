use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use vecstore::{make_record, FilterExpr, FilterOp, Metadata, Query, VecStore};

fn setup_store_with_data(n: usize, dim: usize) -> (tempfile::TempDir, VecStore) {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut store = VecStore::open(temp_dir.path()).unwrap();

    for i in 0..n {
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields.insert("index".into(), serde_json::json!(i));
        meta.fields.insert(
            "category".into(),
            serde_json::json!(format!("cat{}", i % 10)),
        );
        meta.fields
            .insert("score".into(), serde_json::json!(i % 100));

        let vector: Vec<f32> = (0..dim).map(|j| (i + j) as f32 / 100.0).collect();

        store.upsert(format!("doc{}", i), vector, meta).unwrap();
    }

    (temp_dir, store)
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let temp_dir = tempfile::tempdir().unwrap();
                let mut store = VecStore::open(temp_dir.path()).unwrap();

                for i in 0..size {
                    let meta = Metadata {
                        fields: HashMap::new(),
                    };
                    let vector: Vec<f32> = vec![i as f32, 0.0, 0.0];
                    store.upsert(format!("doc{}", i), vector, meta).unwrap();
                }
            });
        });
    }

    group.finish();
}

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");

    for size in [100, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let temp_dir = tempfile::tempdir().unwrap();
                let mut store = VecStore::open(temp_dir.path()).unwrap();

                let records: Vec<_> = (0..size)
                    .map(|i| {
                        let meta = Metadata {
                            fields: HashMap::new(),
                        };
                        let vector: Vec<f32> = vec![i as f32, 0.0, 0.0];
                        make_record(format!("doc{}", i), vector, meta)
                    })
                    .collect();

                store.batch_upsert(records).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");

    for (size, dim) in [(1000, 128), (5000, 128), (10000, 128)].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("dataset", format!("{}x{}", size, dim)),
            &(size, dim),
            |b, &(size, dim)| {
                let (_temp, store) = setup_store_with_data(*size, *dim);
                let query_vec: Vec<f32> = (0..*dim).map(|i| i as f32 / 100.0).collect();

                b.iter(|| {
                    let query = Query {
                        vector: query_vec.clone(),
                        k: 10,
                        filter: None,
                    };
                    black_box(store.query(query).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_query_with_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_with_filter");

    for size in [1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let (_temp, store) = setup_store_with_data(size, 128);
            let query_vec: Vec<f32> = (0..128).map(|i| i as f32 / 100.0).collect();

            b.iter(|| {
                let query = Query {
                    vector: query_vec.clone(),
                    k: 10,
                    filter: Some(FilterExpr::And(vec![
                        FilterExpr::Cmp {
                            field: "score".into(),
                            op: FilterOp::Gt,
                            value: serde_json::json!(50),
                        },
                        FilterExpr::Cmp {
                            field: "category".into(),
                            op: FilterOp::Eq,
                            value: serde_json::json!("cat5"),
                        },
                    ])),
                };
                black_box(store.query(query).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence");

    for size in [100, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Benchmark save
        group.bench_with_input(BenchmarkId::new("save", size), size, |b, &size| {
            let (_temp, store) = setup_store_with_data(size, 128);
            b.iter(|| {
                black_box(store.save().unwrap());
            });
        });

        // Benchmark load
        group.bench_with_input(BenchmarkId::new("load", size), size, |b, &size| {
            let (temp, store) = setup_store_with_data(size, 128);
            store.save().unwrap();
            let path = temp.path();

            b.iter(|| {
                black_box(VecStore::open(path).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_different_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimensions");

    for dim in [64, 128, 256, 512, 1024].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let (_temp, store) = setup_store_with_data(1000, dim);
            let query_vec: Vec<f32> = (0..dim).map(|i| i as f32 / 100.0).collect();

            b.iter(|| {
                let query = Query {
                    vector: query_vec.clone(),
                    k: 10,
                    filter: None,
                };
                black_box(store.query(query).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_complex_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_filters");
    let (_temp, store) = setup_store_with_data(5000, 128);
    let query_vec: Vec<f32> = (0..128).map(|i| i as f32 / 100.0).collect();

    group.bench_function("simple_eq", |b| {
        b.iter(|| {
            let query = Query {
                vector: query_vec.clone(),
                k: 10,
                filter: Some(FilterExpr::Cmp {
                    field: "category".into(),
                    op: FilterOp::Eq,
                    value: serde_json::json!("cat5"),
                }),
            };
            black_box(store.query(query).unwrap());
        });
    });

    group.bench_function("and_filter", |b| {
        b.iter(|| {
            let query = Query {
                vector: query_vec.clone(),
                k: 10,
                filter: Some(FilterExpr::And(vec![
                    FilterExpr::Cmp {
                        field: "score".into(),
                        op: FilterOp::Gt,
                        value: serde_json::json!(50),
                    },
                    FilterExpr::Cmp {
                        field: "score".into(),
                        op: FilterOp::Lt,
                        value: serde_json::json!(80),
                    },
                ])),
            };
            black_box(store.query(query).unwrap());
        });
    });

    group.bench_function("nested_or_and", |b| {
        b.iter(|| {
            let query = Query {
                vector: query_vec.clone(),
                k: 10,
                filter: Some(FilterExpr::Or(vec![
                    FilterExpr::And(vec![
                        FilterExpr::Cmp {
                            field: "score".into(),
                            op: FilterOp::Gt,
                            value: serde_json::json!(80),
                        },
                        FilterExpr::Cmp {
                            field: "category".into(),
                            op: FilterOp::Eq,
                            value: serde_json::json!("cat1"),
                        },
                    ]),
                    FilterExpr::And(vec![
                        FilterExpr::Cmp {
                            field: "score".into(),
                            op: FilterOp::Lt,
                            value: serde_json::json!(20),
                        },
                        FilterExpr::Cmp {
                            field: "category".into(),
                            op: FilterOp::Eq,
                            value: serde_json::json!("cat9"),
                        },
                    ]),
                ])),
            };
            black_box(store.query(query).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_batch_insert,
    bench_query,
    bench_query_with_filter,
    bench_persistence,
    bench_different_dimensions,
    bench_complex_filters,
);
criterion_main!(benches);
