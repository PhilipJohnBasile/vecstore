//! Multi-Stage Reranking Pipeline
//!
//! Demonstrates building a sophisticated retrieval pipeline with:
//! - Initial vector search (Stage 1)
//! - Score-based filtering (Stage 2)
//! - Reranking with custom scorer (Stage 3)
//! - Result fusion and deduplication
//!
//! Run with: cargo run --example 06_reranking_pipeline

use anyhow::Result;
use std::collections::HashMap;
use vecstore::{
    text_splitter::{RecursiveCharacterTextSplitter, TextSplitter},
    Metadata, Neighbor, Query, VecStore,
};

fn main() -> Result<()> {
    println!("🎯 Multi-Stage Reranking Pipeline\n");

    // Build knowledge base
    println!("Step 1: Building knowledge base...");
    let documents = vec![
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.",
        "The Rust compiler uses a sophisticated borrow checker to enforce memory safety at compile time without garbage collection.",
        "Cargo is Rust's built-in package manager and build system, making it easy to manage dependencies and build projects.",
        "Rust achieves zero-cost abstractions, meaning you can use high-level features without runtime performance penalties.",
        "The ownership system in Rust ensures memory safety by tracking which part of code is responsible for allocating and freeing memory.",
        "Rust's type system and ownership model guarantee thread safety, preventing data races at compile time.",
        "Pattern matching in Rust is exhaustive, ensuring all possible cases are handled at compile time.",
        "Traits in Rust provide a way to define shared behavior, similar to interfaces in other languages but more powerful.",
    ];

    let mut store = VecStore::open("./data/06_reranking")?;
    let splitter = RecursiveCharacterTextSplitter::new(200, 20);

    for (i, doc) in documents.iter().enumerate() {
        let chunks = splitter.split_text(doc)?;
        for (j, chunk) in chunks.into_iter().enumerate() {
            let mut metadata = Metadata {
                fields: HashMap::new(),
            };
            metadata
                .fields
                .insert("text".to_string(), serde_json::json!(chunk));
            metadata
                .fields
                .insert("doc_id".to_string(), serde_json::json!(i));

            // Add synthetic quality scores for demonstration
            let quality_score = 0.5 + (i as f32 * 0.05) % 0.5;
            metadata.fields.insert(
                "quality_score".to_string(),
                serde_json::json!(quality_score),
            );

            store.upsert(format!("doc{}_{}", i, j), mock_embed(&chunk), metadata)?;
        }
    }
    println!(
        "   ✓ Knowledge base ready with {} documents\n",
        documents.len()
    );

    // Multi-stage retrieval
    let query = "How does Rust ensure memory safety?";
    println!("🔍 Query: {}\n", query);

    // STAGE 1: Initial vector search with high recall
    println!("Stage 1: Initial Vector Search (High Recall)");
    println!("   Retrieving top 20 candidates...");

    let stage1_results = store.query(Query {
        vector: mock_embed(query),
        k: 20,
        filter: None,
    })?;

    println!("   ✓ Retrieved {} candidates", stage1_results.len());
    println!(
        "   Score range: {:.3} - {:.3}\n",
        stage1_results.first().map(|r| r.score).unwrap_or(0.0),
        stage1_results.last().map(|r| r.score).unwrap_or(0.0)
    );

    // STAGE 2: Score-based filtering
    println!("Stage 2: Score-Based Filtering");
    println!("   Filtering results with score > 0.5...");

    let stage2_results: Vec<_> = stage1_results
        .into_iter()
        .filter(|r| r.score > 0.5)
        .collect();

    println!("   ✓ Filtered to {} results\n", stage2_results.len());

    // STAGE 3: Reranking with custom scorer
    println!("Stage 3: Custom Reranking");
    println!("   Reranking with BM25-style scoring...");

    let mut stage3_results = stage2_results;
    rerank_results(&mut stage3_results, query)?;

    println!("   ✓ Reranked {} results\n", stage3_results.len());

    // Display final results
    println!("🎯 Final Top 5 Results After Reranking:\n");
    for (i, result) in stage3_results.iter().take(5).enumerate() {
        let text = result
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");
        let quality = result
            .metadata
            .fields
            .get("quality_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        println!(
            "{}. Score: {:.3} | Quality: {:.2}",
            i + 1,
            result.score,
            quality
        );
        println!("   {}\n", text);
    }

    // Compare with original ordering
    println!("📊 Pipeline Impact Analysis:");
    println!("   Stage 1 (Vector Search): 20 candidates");
    println!(
        "   Stage 2 (Filtering):     {} candidates ({}% reduction)",
        stage3_results.len(),
        (1.0 - stage3_results.len() as f32 / 20.0) * 100.0
    );
    println!("   Stage 3 (Reranking):     Top 5 selected");

    if let Some(top) = stage3_results.first() {
        let text = top
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");
        println!("\n   🏆 Best Result:");
        println!("      {}", text);
    }

    println!("\n✅ Reranking Pipeline Example Complete!");
    println!("\n💡 Pipeline Benefits:");
    println!("   • Stage 1: High recall with fast vector search");
    println!("   • Stage 2: Filter low-quality or irrelevant results");
    println!("   • Stage 3: Precise reranking with expensive models");
    println!("   • Overall: Better relevance than single-stage retrieval");
    println!("\n🔧 Production Tips:");
    println!("   • Use cross-encoder models for Stage 3 reranking");
    println!("   • Consider ColBERT for token-level matching");
    println!("   • Add diversity scoring to avoid redundant results");
    println!("   • Cache reranking scores for repeated queries");
    println!("   • Monitor latency at each stage");

    Ok(())
}

/// Rerank results using a custom scoring function
/// In production: Use cross-encoder model or ColBERT
fn rerank_results(results: &mut Vec<Neighbor>, query: &str) -> Result<()> {
    let query_terms: Vec<&str> = query.split_whitespace().collect();

    // Calculate reranking score for each result
    for result in results.iter_mut() {
        let text = result
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let quality_score = result
            .metadata
            .fields
            .get("quality_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;

        // BM25-style term frequency scoring
        let mut term_score = 0.0;
        for term in &query_terms {
            let term_lower = term.to_lowercase();
            let text_lower = text.to_lowercase();
            let term_count = text_lower.matches(term_lower.as_str()).count() as f32;

            // BM25 formula: TF / (TF + k1 * (1 - b + b * doc_len / avg_doc_len))
            let k1 = 1.5;
            let b = 0.75;
            let doc_len = text.split_whitespace().count() as f32;
            let avg_doc_len = 20.0; // Approximate

            let tf_component =
                term_count / (term_count + k1 * (1.0 - b + b * doc_len / avg_doc_len));
            term_score += tf_component;
        }

        // Combine vector similarity, term matching, and quality score
        let combined_score = result.score * 0.5 +  // Vector similarity (50%)
                            term_score * 0.3 +      // Term matching (30%)
                            quality_score * 0.2; // Quality score (20%)

        result.score = combined_score;
    }

    // Sort by new scores (descending)
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    Ok(())
}

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
