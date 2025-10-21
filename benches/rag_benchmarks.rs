use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use vecstore::{
    rag_utils::MultiQueryRetrieval,
    text_splitter::{RecursiveCharacterTextSplitter, TextSplitter},
    Metadata, Query, VecStore,
};

// Mock embedding function for benchmarks
fn mock_embed(text: &str) -> Vec<f32> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut embedding = vec![0.0; 384];
    for (i, word) in words.iter().enumerate() {
        embedding[(word.len() * (i + 1)) % 384] += 1.0;
    }
    let mag: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 0.0 {
        for val in &mut embedding {
            *val /= mag;
        }
    }
    embedding
}

// Sample documents for benchmarking
fn sample_documents() -> Vec<String> {
    vec![
        "Rust is a systems programming language that runs blazingly fast and prevents segfaults.".to_string(),
        "The borrow checker in Rust enforces memory safety at compile time without garbage collection.".to_string(),
        "Cargo is Rust's package manager and build system, making dependency management easy.".to_string(),
        "Zero-cost abstractions in Rust mean you can use high-level features without runtime overhead.".to_string(),
        "Ownership in Rust ensures memory safety by tracking allocation and deallocation.".to_string(),
        "Rust's type system guarantees thread safety, preventing data races at compile time.".to_string(),
        "Pattern matching in Rust is exhaustive, ensuring all cases are handled.".to_string(),
        "Traits in Rust provide shared behavior similar to interfaces but more powerful.".to_string(),
        "VecStore is a high-performance vector database built in Rust for RAG applications.".to_string(),
        "HNSW indexing in VecStore provides fast approximate nearest neighbor search.".to_string(),
    ]
}

/// Benchmark: Document chunking with different text splitters
fn bench_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking");
    let documents = sample_documents();
    let combined_text = documents.join("\n\n");

    // Benchmark RecursiveCharacterTextSplitter
    group.bench_function("recursive_character_200", |b| {
        let splitter = RecursiveCharacterTextSplitter::new(200, 20);
        b.iter(|| splitter.split_text(black_box(&combined_text)).unwrap());
    });

    group.bench_function("recursive_character_500", |b| {
        let splitter = RecursiveCharacterTextSplitter::new(500, 50);
        b.iter(|| splitter.split_text(black_box(&combined_text)).unwrap());
    });

    group.finish();
}

/// Benchmark: Document indexing throughput
fn bench_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexing");
    let documents = sample_documents();

    for num_docs in [10, 50, 100] {
        group.throughput(Throughput::Elements(num_docs as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_docs),
            &num_docs,
            |b, &num_docs| {
                b.iter(|| {
                    let mut store =
                        VecStore::open(format!("./bench_data/index_{}", num_docs)).unwrap();
                    let splitter = RecursiveCharacterTextSplitter::new(200, 20);

                    for i in 0..num_docs {
                        let doc = &documents[i % documents.len()];
                        let chunks = splitter.split_text(doc).unwrap();

                        for (j, chunk) in chunks.iter().enumerate() {
                            let mut metadata = Metadata {
                                fields: HashMap::new(),
                            };
                            metadata
                                .fields
                                .insert("text".to_string(), serde_json::json!(chunk));

                            let chunk_id = format!("doc{}_{}", i, j);
                            let embedding = mock_embed(chunk);
                            store.upsert(chunk_id, embedding, metadata).unwrap();
                        }
                    }
                    store
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Query latency with different k values
fn bench_query_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_latency");

    // Setup: Create a pre-populated store
    let documents = sample_documents();
    let mut store = VecStore::open("./bench_data/query_latency").unwrap();
    let splitter = RecursiveCharacterTextSplitter::new(200, 20);

    for (i, doc) in documents.iter().enumerate() {
        let chunks = splitter.split_text(doc).unwrap();
        for (j, chunk) in chunks.iter().enumerate() {
            let mut metadata = Metadata {
                fields: HashMap::new(),
            };
            metadata
                .fields
                .insert("text".to_string(), serde_json::json!(chunk));
            store
                .upsert(format!("doc{}_{}", i, j), mock_embed(chunk), metadata)
                .unwrap();
        }
    }

    let query_text = "How does Rust ensure memory safety?";
    let query_embedding = mock_embed(query_text);

    for k in [1, 5, 10, 20] {
        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k| {
            b.iter(|| {
                store
                    .query(Query {
                        vector: black_box(query_embedding.clone()),
                        k: black_box(k),
                        filter: None,
                    })
                    .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark: Multi-query RAG with fusion
fn bench_multi_query_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_query_fusion");

    // Setup
    let documents = sample_documents();
    let mut store = VecStore::open("./bench_data/multi_query").unwrap();
    let splitter = RecursiveCharacterTextSplitter::new(200, 20);

    for (i, doc) in documents.iter().enumerate() {
        let chunks = splitter.split_text(doc).unwrap();
        for (j, chunk) in chunks.iter().enumerate() {
            let mut metadata = Metadata {
                fields: HashMap::new(),
            };
            metadata
                .fields
                .insert("text".to_string(), serde_json::json!(chunk));
            store
                .upsert(format!("doc{}_{}", i, j), mock_embed(chunk), metadata)
                .unwrap();
        }
    }

    let query_variants = vec![
        "How does Rust ensure memory safety?",
        "What are Rust's memory safety features?",
        "How does Rust prevent memory bugs?",
    ];

    group.bench_function("3_queries_with_fusion", |b| {
        b.iter(|| {
            let mut all_results = Vec::new();

            for variant in &query_variants {
                let results = store
                    .query(Query {
                        vector: mock_embed(variant),
                        k: 5,
                        filter: None,
                    })
                    .unwrap();
                all_results.push(results);
            }

            MultiQueryRetrieval::reciprocal_rank_fusion(black_box(all_results), 60)
        });
    });

    group.finish();
}

/// Benchmark: End-to-end RAG pipeline
fn bench_e2e_rag_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_pipeline");

    group.bench_function("complete_rag_workflow", |b| {
        let documents = sample_documents();
        let query = "What is VecStore?";

        b.iter(|| {
            // 1. Create store
            let mut store = VecStore::open("./bench_data/e2e").unwrap();

            // 2. Split documents
            let splitter = RecursiveCharacterTextSplitter::new(200, 20);

            // 3. Index documents
            for (i, doc) in documents.iter().enumerate() {
                let chunks = splitter.split_text(doc).unwrap();
                for (j, chunk) in chunks.iter().enumerate() {
                    let mut metadata = Metadata {
                        fields: HashMap::new(),
                    };
                    metadata
                        .fields
                        .insert("text".to_string(), serde_json::json!(chunk));
                    store
                        .upsert(format!("doc{}_{}", i, j), mock_embed(chunk), metadata)
                        .unwrap();
                }
            }

            // 4. Query
            let results = store
                .query(Query {
                    vector: mock_embed(black_box(query)),
                    k: 3,
                    filter: None,
                })
                .unwrap();

            // 5. Format context
            let _context: Vec<String> = results
                .iter()
                .filter_map(|r| r.metadata.fields.get("text").and_then(|v| v.as_str()))
                .map(|s| s.to_string())
                .collect();
        });
    });

    group.finish();
}

/// Benchmark: Text splitter comparison
fn bench_splitter_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("splitter_comparison");

    let long_document = sample_documents().join(" ").repeat(10);

    group.bench_function("recursive_char_small_chunks", |b| {
        let splitter = RecursiveCharacterTextSplitter::new(100, 10);
        b.iter(|| splitter.split_text(black_box(&long_document)).unwrap());
    });

    group.bench_function("recursive_char_medium_chunks", |b| {
        let splitter = RecursiveCharacterTextSplitter::new(300, 30);
        b.iter(|| splitter.split_text(black_box(&long_document)).unwrap());
    });

    group.bench_function("recursive_char_large_chunks", |b| {
        let splitter = RecursiveCharacterTextSplitter::new(500, 50);
        b.iter(|| splitter.split_text(black_box(&long_document)).unwrap());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_chunking,
    bench_indexing,
    bench_query_latency,
    bench_multi_query_fusion,
    bench_e2e_rag_pipeline,
    bench_splitter_comparison
);

criterion_main!(benches);
