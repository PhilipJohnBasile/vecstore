//! Local RAG Application Template
//!
//! A privacy-first RAG application that runs entirely offline.
//! No data leaves your machine - perfect for sensitive documents.
//!
//! Features:
//! - Local embeddings with ONNX models
//! - VecStore for vector storage
//! - Explainable search results
//! - Document chunking and retrieval
//!
//! Usage:
//!   cargo run --release --features embeddings
//!
//! Or with OpenAI (requires API key):
//!   OPENAI_API_KEY=your-key cargo run --release --features openai-embeddings

use anyhow::Result;
use std::path::PathBuf;
use vecstore::{VecStore, DistanceMetric};

/// Configuration for the RAG application
struct RagConfig {
    /// Path to store the vector database
    db_path: PathBuf,
    /// Embedding dimension (384 for MiniLM, 1536 for OpenAI)
    dimension: usize,
    /// Number of results to retrieve
    top_k: usize,
    /// Chunk size for document splitting
    chunk_size: usize,
    /// Overlap between chunks
    chunk_overlap: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("./local_rag.db"),
            dimension: 384,  // MiniLM-L6-v2
            top_k: 5,
            chunk_size: 512,
            chunk_overlap: 50,
        }
    }
}

/// Simple text chunker
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk = words[start..end].join(" ");
        chunks.push(chunk);

        if end >= words.len() {
            break;
        }
        start = end.saturating_sub(overlap);
    }

    chunks
}

/// Generate embeddings for text
/// In a real application, use vecstore's embedding features
fn generate_embedding(text: &str, dimension: usize) -> Vec<f32> {
    // Placeholder: In production, use:
    // - Local: vecstore with `embeddings` feature and ONNX models
    // - Cloud: vecstore with `openai-embeddings` feature

    // Simple hash-based embedding for demo
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut embedding = vec![0.0f32; dimension];
    let words: Vec<&str> = text.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        let mut hasher = DefaultHasher::new();
        word.to_lowercase().hash(&mut hasher);
        let hash = hasher.finish();

        for j in 0..dimension {
            let idx = (hash.wrapping_add(j as u64) as usize) % dimension;
            embedding[idx] += 1.0 / ((i + 1) as f32);
        }
    }

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    embedding
}

fn main() -> Result<()> {
    println!("===========================================");
    println!("  VecStore Local RAG Application");
    println!("  Privacy-First Document Search");
    println!("===========================================\n");

    let config = RagConfig::default();

    // Initialize VecStore
    let mut store = VecStore::new(config.dimension, DistanceMetric::Cosine);

    // Sample documents (replace with your document loading logic)
    let documents = vec![
        ("doc1", "VecStore is an embeddable vector database written in Rust. It provides fast similarity search using HNSW indexing."),
        ("doc2", "The Explainable Search feature helps you understand WHY results matched your query. This is unique among vector databases."),
        ("doc3", "Privacy-preserving search ensures your sensitive data stays secure. Differential privacy protects individual embeddings."),
        ("doc4", "VecStore supports WASM for browser deployment. Run vector search entirely in the browser without sending data to servers."),
        ("doc5", "Time-aware search lets you query how results would have looked at different points in time. Great for news and e-commerce."),
    ];

    println!("Indexing {} documents...\n", documents.len());

    // Index documents
    for (doc_id, content) in &documents {
        let chunks = chunk_text(content, config.chunk_size, config.chunk_overlap);

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let id = format!("{}_chunk_{}", doc_id, chunk_idx);
            let embedding = generate_embedding(chunk, config.dimension);

            store.add_with_metadata(
                &id,
                &embedding,
                serde_json::json!({
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "content": chunk,
                }),
            )?;
        }
    }

    println!("Indexed {} vectors\n", store.len());

    // Query loop
    let queries = vec![
        "What makes VecStore unique?",
        "How does privacy work?",
        "Can I use it in a browser?",
    ];

    for query in queries {
        println!("Query: {}", query);
        println!("{}", "-".repeat(50));

        let query_embedding = generate_embedding(query, config.dimension);
        let results = store.search(&query_embedding, config.top_k)?;

        for (i, result) in results.iter().enumerate() {
            let metadata = result.metadata.as_ref()
                .and_then(|m| m.as_object())
                .map(|m| m.get("content").and_then(|v| v.as_str()).unwrap_or(""))
                .unwrap_or("");

            println!(
                "  {}. [score: {:.4}] {}",
                i + 1,
                result.score,
                &metadata[..metadata.len().min(80)]
            );
        }
        println!();
    }

    // Save the database
    store.save(&config.db_path)?;
    println!("Database saved to {:?}", config.db_path);

    Ok(())
}
