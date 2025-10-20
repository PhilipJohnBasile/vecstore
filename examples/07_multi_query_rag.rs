//! Multi-Query RAG with Fusion
//!
//! Demonstrates advanced retrieval techniques:
//! - Query expansion
//! - Multiple query variants
//! - Result fusion (RRF)
//!
//! Run with: cargo run --example 07_multi_query_rag

use anyhow::Result;
use std::collections::HashMap;
use vecstore::{
    rag_utils::MultiQueryRetrieval,
    text_splitter::{RecursiveCharacterTextSplitter, TextSplitter},
    Metadata, Query, VecStore,
};

fn main() -> Result<()> {
    println!("🔀 Multi-Query RAG with Fusion\n");

    // Build knowledge base
    println!("Step 1: Building knowledge base...");
    let documents = vec![
        "Rust provides memory safety without garbage collection through its ownership system.",
        "The borrow checker enforces memory safety rules at compile time.",
        "Rust achieves zero-cost abstractions, meaning abstractions have no runtime overhead.",
        "Cargo is Rust's built-in package manager and build tool.",
    ];

    let mut store = VecStore::open("./data/07_multi_query")?;
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
            store.upsert(format!("doc{}_{}", i, j), mock_embed(&chunk), metadata)?;
        }
    }
    println!("   ✓ Knowledge base ready\n");

    // Multi-query RAG
    println!("Step 2: Query expansion...");
    let original_query = "How does Rust ensure memory safety?";
    println!("\n❓ Original Query: {}", original_query);

    // Generate query variants (in production, use LLM for this)
    let query_variants = vec![
        "How does Rust ensure memory safety?",
        "What are Rust's memory safety features?",
        "How does Rust prevent memory bugs?",
    ];

    println!("\n📝 Query Variants:");
    for (i, variant) in query_variants.iter().enumerate() {
        println!("   {}. {}", i + 1, variant);
    }

    // Execute all queries
    println!("\nStep 3: Executing multiple queries...");
    let mut all_results = Vec::new();

    for variant in &query_variants {
        let results = store.query(Query {
            vector: mock_embed(variant),
            k: 3,
            filter: None,
        })?;
        all_results.push(results);
    }

    println!("   ✓ Executed {} queries\n", query_variants.len());

    // Fuse results using RRF
    println!("Step 4: Fusing results with Reciprocal Rank Fusion...");
    let fused = MultiQueryRetrieval::reciprocal_rank_fusion(all_results, 60);

    println!("\n🎯 Top Fused Results:");
    for (i, result) in fused.iter().take(3).enumerate() {
        let text = result
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");
        println!("   {}. Score: {:.3}", i + 1, result.score);
        println!("      {}\n", text);
    }

    println!("✅ Multi-Query RAG Example Complete!");
    println!("\n💡 Benefits:");
    println!("   • Query expansion improves recall");
    println!("   • Multiple perspectives capture different aspects");
    println!("   • RRF fusion combines rankings effectively");
    println!("   • More robust than single-query retrieval");

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
