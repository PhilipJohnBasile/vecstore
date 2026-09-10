//! RAG Evaluation Demo
//!
//! Demonstrates using vecstore-eval to measure RAG quality.
//! This is a simplified version - see vecstore-eval/examples/evaluate_rag.rs
//! for the full implementation.
//!
//! Run with: cargo run --example 09_evaluation_demo

use anyhow::Result;
use std::collections::HashMap;
use vecstore::{
    Metadata, Query, VecStore,
    text_splitter::{RecursiveCharacterTextSplitter, TextSplitter},
};

fn main() -> Result<()> {
    println!("📊 RAG Evaluation Demo\n");
    println!("This example shows how to evaluate your RAG system quality.\n");

    // Build a simple RAG system
    println!("Step 1: Building RAG system...");
    let documents = vec![
        "VecStore is an embeddable vector database built in Rust.",
        "It provides HNSW indexing, persistence, and a complete RAG toolkit.",
        "The Rust implementation generally runs faster than pure Python loops.",
    ];

    let mut store = VecStore::open("./data/09_evaluation")?;
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
    println!("   ✓ RAG system ready\n");

    // Simulate RAG evaluation
    println!("Step 2: Evaluating RAG quality...\n");

    let test_cases = vec![
        (
            "What is VecStore?",
            "VecStore is a high-performance vector database built in Rust.",
        ),
        (
            "How fast is VecStore?",
            "Rust avoids the Python interpreter in tight loops.",
        ),
    ];

    let mut total_score = 0.0;
    for (query, _expected) in &test_cases {
        println!("❓ Query: {}", query);

        // Retrieve
        let results = store.query(Query {
            vector: mock_embed(query),
            k: 2,
            filter: None,
        })?;

        // Simple relevance score (in production, use vecstore-eval)
        let score = results.first().map(|r| r.score).unwrap_or(0.0);
        total_score += score;

        println!("   📄 Retrieved {} chunks", results.len());
        println!("   📈 Relevance Score: {:.3}\n", score);
    }

    let avg_score = total_score / test_cases.len() as f32;
    println!("Overall Average Score: {:.3}", avg_score);

    println!("\n💡 For complete evaluation with LLM-as-judge:");
    println!("   See: vecstore-eval/examples/evaluate_rag.rs");
    println!("   Metrics: Context Relevance, Answer Faithfulness, Answer Correctness");

    println!("\n✅ Evaluation Demo Complete!");

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
