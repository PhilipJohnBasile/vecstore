//! Sparse Vectors & Hybrid Search Demo
//!
//! This example demonstrates VecStore's new sparse vector capabilities:
//! - Sparse vectors (keyword/term vectors)
//! - Dense vectors (semantic embeddings)
//! - Hybrid vectors (dense + sparse combined)
//! - BM25 keyword scoring
//! - Fusion strategies (WeightedSum, RRF, Max, Min, HarmonicMean)
//!
//! Hybrid search combines the best of both worlds:
//! - Dense: Captures semantic meaning, synonyms, context
//! - Sparse: Captures exact keywords, terminology, rare terms
//!
//! Run with: cargo run --example sparse_vectors_demo

use std::collections::HashMap;
use vecstore::{
    bm25_score, hybrid_search_score, normalize_scores, BM25Config, BM25Stats, FusionStrategy,
    HybridQueryV2, HybridSearchConfig, Vector,
};

fn main() -> anyhow::Result<()> {
    println!("=== VecStore Sparse Vectors & Hybrid Search Demo ===\n");

    // ========== 1. Sparse Vectors ==========
    println!("1. Sparse Vectors (Keyword Vectors)\n");

    // Example: Documents represented as sparse term vectors
    // Vocabulary: ["rust", "python", "machine", "learning", "deep", "neural", "network"]
    // Indices:     [0,      1,        2,         3,         4,      5,       6]

    let doc1 = Vector::sparse(
        1000,                // vocabulary size
        vec![0, 2, 3],       // terms: rust, machine, learning
        vec![2.0, 1.0, 1.0], // term frequencies
    )?;

    let doc2 = Vector::sparse(
        1000,
        vec![1, 2, 3, 4, 5, 6], // terms: python, machine, learning, deep, neural, network
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )?;

    println!("   Document 1: 'rust machine learning'");
    println!("   - Dimension: {}", doc1.dimension());
    println!("   - Sparsity: {:.1}%", doc1.sparsity() * 100.0);
    println!(
        "   - Storage: {} elements (instead of {})",
        doc1.storage_size(),
        doc1.dimension()
    );
    println!("   - Memory: {} bytes", doc1.memory_usage());
    println!();

    println!("   Document 2: 'python machine learning deep neural network'");
    println!("   - Dimension: {}", doc2.dimension());
    println!("   - Sparsity: {:.1}%", doc2.sparsity() * 100.0);
    println!("   - Storage: {} elements", doc2.storage_size());
    println!();

    // ========== 2. Dense Vectors ==========
    println!("2. Dense Vectors (Semantic Embeddings)\n");

    // Simulated embeddings from a transformer model
    let embedding1 = Vector::dense(vec![0.8, 0.1, 0.2, 0.9]);
    let embedding2 = Vector::dense(vec![0.2, 0.9, 0.8, 0.1]);

    println!("   Embedding 1: {:?}", embedding1.dense_part().unwrap());
    println!("   Embedding 2: {:?}", embedding2.dense_part().unwrap());
    println!("   Dimension: {}", embedding1.dimension());
    println!();

    // ========== 3. Hybrid Vectors ==========
    println!("3. Hybrid Vectors (Dense + Sparse)\n");

    let hybrid_doc = Vector::hybrid(
        vec![0.1, 0.2, 0.3, 0.4], // semantic embedding
        vec![0, 2, 3],            // keyword indices
        vec![2.0, 1.0, 1.0],      // keyword weights
    )?;

    println!("   Hybrid document:");
    println!(
        "   - Has dense component: {}",
        hybrid_doc.has_dense_component()
    );
    println!(
        "   - Has sparse component: {}",
        hybrid_doc.has_sparse_component()
    );
    println!("   - Dense: {:?}", hybrid_doc.dense_part().unwrap());
    println!(
        "   - Sparse indices: {:?}",
        hybrid_doc.sparse_part().unwrap().0
    );
    println!(
        "   - Sparse values: {:?}",
        hybrid_doc.sparse_part().unwrap().1
    );
    println!();

    // ========== 4. BM25 Scoring ==========
    println!("4. BM25 Keyword Scoring\n");

    // Build corpus statistics for BM25
    let corpus_docs = vec![
        (vec![0, 2, 3], vec![2.0, 1.0, 1.0]), // doc1: rust, machine, learning
        (vec![1, 2, 3, 4, 5, 6], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), // doc2
        (vec![0, 1], vec![1.0, 2.0]),         // doc3: rust, python
    ];

    let docs_iter = corpus_docs
        .iter()
        .map(|(i, v)| (i.as_slice(), v.as_slice()));
    let bm25_stats = BM25Stats::from_corpus(docs_iter);

    println!("   Corpus statistics:");
    println!("   - Total documents: {}", bm25_stats.num_docs);
    println!("   - Average doc length: {:.2}", bm25_stats.avg_doc_length);
    println!("   - IDF scores:");
    println!("     - Term 0 (rust): {:.3}", bm25_stats.get_idf(0));
    println!("     - Term 1 (python): {:.3}", bm25_stats.get_idf(1));
    println!("     - Term 2 (machine): {:.3}", bm25_stats.get_idf(2));
    println!("     - Term 3 (learning): {:.3}", bm25_stats.get_idf(3));
    println!();

    // Query: "machine learning"
    let query_indices = vec![2, 3];
    let query_weights = vec![1.0, 1.0];

    let score1 = bm25_score(
        &query_indices,
        &query_weights,
        &corpus_docs[0].0,
        &corpus_docs[0].1,
        &bm25_stats,
        &BM25Config::default(),
    );

    let score2 = bm25_score(
        &query_indices,
        &query_weights,
        &corpus_docs[1].0,
        &corpus_docs[1].1,
        &bm25_stats,
        &BM25Config::default(),
    );

    println!("   Query: 'machine learning'");
    println!("   - Doc1 (rust machine learning) score: {:.4}", score1);
    println!("   - Doc2 (python ML deep learning) score: {:.4}", score2);
    println!();

    // ========== 5. Fusion Strategies ==========
    println!("5. Hybrid Search Fusion Strategies\n");

    // Simulate scores from dense (semantic) and sparse (keyword) search
    let dense_score = 0.85; // High semantic similarity
    let sparse_score = 0.45; // Lower keyword match

    println!("   Dense score (semantic): {:.2}", dense_score);
    println!("   Sparse score (keyword): {:.2}", sparse_score);
    println!();

    // Try different fusion strategies
    let strategies = vec![
        (
            FusionStrategy::WeightedSum,
            "WeightedSum (70% dense, 30% sparse)",
        ),
        (
            FusionStrategy::ReciprocalRankFusion,
            "Reciprocal Rank Fusion",
        ),
        (FusionStrategy::Max, "Max (take highest)"),
        (FusionStrategy::Min, "Min (both must match)"),
        (FusionStrategy::HarmonicMean, "Harmonic Mean (balanced)"),
    ];

    for (strategy, name) in strategies {
        let config = HybridSearchConfig {
            fusion_strategy: strategy,
            alpha: 0.7,
            rrf_k: 60.0,
            normalize_scores: false,
            autocut: None,
        };

        let fused_score = hybrid_search_score(dense_score, sparse_score, &config);
        println!("   {}: {:.4}", name, fused_score);
    }
    println!();

    // ========== 6. Alpha Parameter (WeightedSum) ==========
    println!("6. Alpha Parameter Effect (WeightedSum)\n");

    let alphas = vec![0.0, 0.3, 0.5, 0.7, 1.0];

    println!(
        "   Dense score: {:.2}, Sparse score: {:.2}",
        dense_score, sparse_score
    );
    println!();

    for alpha in alphas {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha,
            ..Default::default()
        };

        let score = hybrid_search_score(dense_score, sparse_score, &config);
        let dense_pct = (alpha * 100.0) as i32;
        let sparse_pct = ((1.0 - alpha) * 100.0) as i32;

        println!(
            "   alpha={:.1} ({}% dense, {}% sparse): {:.4}",
            alpha, dense_pct, sparse_pct, score
        );
    }
    println!();

    // ========== 7. Score Normalization ==========
    println!("7. Score Normalization\n");

    // Scores from different sources with different scales
    let dense_scores = vec![0.95, 0.82, 0.75, 0.60, 0.45]; // Cosine similarity
    let sparse_scores = vec![15.2, 8.5, 12.1, 3.2, 6.8]; // BM25 scores (different scale!)

    println!("   Original scores:");
    println!("   - Dense (cosine):  {:?}", dense_scores);
    println!("   - Sparse (BM25):   {:?}", sparse_scores);
    println!();

    let normalized_dense = normalize_scores(&dense_scores);
    let normalized_sparse = normalize_scores(&sparse_scores);

    println!("   Normalized scores (0-1 range):");
    println!(
        "   - Dense:  {:?}",
        normalized_dense
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "   - Sparse: {:?}",
        normalized_sparse
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );
    println!();

    // Fuse normalized scores
    println!("   Fused scores (70% dense, 30% sparse):");
    for i in 0..5 {
        let fused = 0.7 * normalized_dense[i] + 0.3 * normalized_sparse[i];
        println!("   - Result {}: {:.4}", i + 1, fused);
    }
    println!();

    // ========== 8. HybridQuery Builder ==========
    println!("8. HybridQuery Builder Pattern\n");

    // Create queries with different configurations
    let query1 = HybridQueryV2::new(
        vec![0.1, 0.2, 0.3], // dense embedding
        vec![0, 2, 3],       // sparse indices
        vec![1.0, 2.0, 1.5], // sparse values
    )
    .with_k(20)
    .with_alpha(0.8)
    .with_fusion_strategy(FusionStrategy::WeightedSum);

    println!("   Query 1 (Hybrid - WeightedSum):");
    println!("   - K: {}", query1.k);
    println!("   - Alpha: {:.2}", query1.config.alpha);
    println!("   - Strategy: {:?}", query1.config.fusion_strategy);
    println!();

    let query2 = HybridQueryV2::dense_only(vec![0.5, 0.6, 0.7])
        .with_k(10)
        .with_fusion_strategy(FusionStrategy::ReciprocalRankFusion);

    println!("   Query 2 (Dense-only - RRF):");
    println!("   - Has dense: {}", query2.dense.is_some());
    println!("   - Has sparse: {}", query2.sparse.is_some());
    println!("   - K: {}", query2.k);
    println!();

    let query3 = HybridQueryV2::sparse_only(vec![1, 2, 5], vec![1.0, 1.5, 2.0])
        .with_k(15)
        .with_alpha(0.3);

    println!("   Query 3 (Sparse-only - Keywords):");
    println!("   - Has dense: {}", query3.dense.is_none());
    println!("   - Has sparse: {}", query3.sparse.is_some());
    println!("   - K: {}", query3.k);
    println!();

    // ========== 9. Practical Example ==========
    println!("9. Practical Example: Document Search\n");

    println!("   Scenario: Search for 'rust programming tutorials'");
    println!();

    // Dense embedding (from transformer model)
    let query_dense = vec![0.8, 0.2, 0.1, 0.9, 0.3];

    // Sparse keywords (vocabulary indices)
    // vocab: ["rust": 0, "programming": 10, "tutorials": 25, ...]
    let query_sparse_indices = vec![0, 10, 25];
    let query_sparse_values = vec![2.0, 1.5, 1.0]; // term weights/frequencies

    println!("   Query representation:");
    println!("   - Dense embedding (5-dim): {:?}", query_dense);
    println!("   - Sparse keywords: [rust(2.0), programming(1.5), tutorials(1.0)]");
    println!();

    // Simulate search results
    let results = vec![
        ("doc_a", 0.92, 18.5), // High semantic similarity, high keyword match
        ("doc_b", 0.88, 5.2),  // High semantic, low keyword
        ("doc_c", 0.45, 22.1), // Low semantic, high keyword
        ("doc_d", 0.72, 12.8), // Medium both
    ];

    // Normalize and fuse
    let dense_results: Vec<f32> = results.iter().map(|(_, d, _)| *d).collect();
    let sparse_results: Vec<f32> = results.iter().map(|(_, _, s)| *s).collect();

    let norm_dense = normalize_scores(&dense_results);
    let norm_sparse = normalize_scores(&sparse_results);

    println!("   Search results (alpha=0.7):");
    println!("   ID      Dense   Sparse   Fused   Rank");
    println!("   ----------------------------------------");

    let mut fused_results: Vec<(usize, f32)> = norm_dense
        .iter()
        .zip(norm_sparse.iter())
        .enumerate()
        .map(|(i, (&d, &s))| {
            let fused = 0.7 * d + 0.3 * s;
            (i, fused)
        })
        .collect();

    // Sort by fused score descending
    fused_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (idx, fused)) in fused_results.iter().enumerate() {
        let (id, _d, _s) = results[*idx];
        let nd = norm_dense[*idx];
        let ns = norm_sparse[*idx];
        println!(
            "   {:<7} {:.2}    {:.2}     {:.4}   #{}",
            id,
            nd,
            ns,
            fused,
            rank + 1
        );
    }
    println!();

    println!(
        "   Winner: {} (balanced semantic + keyword match)",
        results[fused_results[0].0].0
    );
    println!();

    // ========== Summary ==========
    println!("=== Summary ===");
    println!();
    println!("VecStore now supports hybrid search:");
    println!("  ✓ Sparse vectors - Memory-efficient keyword vectors");
    println!("  ✓ Dense vectors - Semantic embeddings");
    println!("  ✓ Hybrid vectors - Combined dense + sparse");
    println!("  ✓ BM25 scoring - Industry-standard keyword scoring");
    println!("  ✓ 5 fusion strategies - WeightedSum, RRF, Max, Min, HarmonicMean");
    println!("  ✓ Score normalization - Handle different score scales");
    println!("  ✓ Builder pattern - Ergonomic query construction");
    println!();
    println!("Hybrid search combines the best of both worlds:");
    println!("  → Dense: Semantic meaning, synonyms, context");
    println!("  → Sparse: Exact keywords, terminology, rare terms");
    println!();
    println!("Perfect for: RAG, document search, e-commerce, code search");
    println!();
    println!("Memory savings example:");
    println!("  - Dense 10K-dim vector: 40KB (10,000 × 4 bytes)");
    println!("  - Sparse 10K-dim vector with 100 non-zero: 1.2KB (97% savings!)");

    Ok(())
}
