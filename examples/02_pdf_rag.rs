//! PDF Document RAG Example
//!
//! Demonstrates RAG with PDF documents using vecstore-loaders.
//!
//! Run with: cargo run --example 02_pdf_rag --features loaders

use anyhow::Result;
use std::collections::HashMap;
use vecstore::{
    text_splitter::{RecursiveCharacterTextSplitter, TextSplitter},
    Metadata, Query, VecStore,
};
#[cfg(feature = "loaders")]
use vecstore_loaders::PDFLoader;

#[cfg(not(feature = "loaders"))]
fn main() {
    println!("❌ This example requires the 'loaders' feature.");
    println!("   Run with: cargo run --example 02_pdf_rag --features loaders");
}

#[cfg(feature = "loaders")]
fn main() -> Result<()> {
    println!("📄 PDF RAG Example\n");

    // Simulated PDF content (in production, load from actual PDF)
    let pdf_content = r#"
    VECSTORE: A MODERN VECTOR DATABASE

    Abstract
    VecStore is a high-performance vector database built in Rust. It provides
    HNSW indexing, persistence, and a complete RAG toolkit.

    1. Introduction
    Vector databases are essential for modern AI applications. They enable
    semantic search, recommendation systems, and retrieval-augmented generation.

    2. Features
    VecStore offers multiple distance metrics, quantization for memory efficiency,
    and full-text search capabilities.

    3. Performance
    The Rust implementation avoids the overhead associated with pure Python loops,
    delivering faster query performance in most scenarios.
    "#;

    println!("Step 1: Loading PDF content...");
    println!("   (Simulated - in production use PDFLoader with actual PDF files)");
    println!("   ✓ Loaded {} characters\n", pdf_content.len());

    // Step 2: Split into chunks
    println!("Step 2: Chunking document...");
    let splitter = RecursiveCharacterTextSplitter::new(300, 50);
    let chunks = splitter.split_text(pdf_content)?;
    println!("   ✓ Created {} chunks\n", chunks.len());

    // Step 3: Store chunks
    println!("Step 3: Storing in vector database...");
    let mut store = VecStore::open("./data/02_pdf_rag")?;

    for (i, chunk) in chunks.iter().enumerate() {
        let embedding = mock_embed(chunk);
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("text".to_string(), serde_json::json!(chunk));
        metadata.fields.insert(
            "source".to_string(),
            serde_json::json!("vecstore_paper.pdf"),
        );
        metadata
            .fields
            .insert("page".to_string(), serde_json::json!((i / 3) + 1));

        store.upsert(format!("chunk_{}", i), embedding, metadata)?;
    }
    println!("   ✓ Stored {} chunks\n", chunks.len());

    // Step 4: Query
    println!("Step 4: Querying PDF content...");
    let query_text = "What are the main features of VecStore?";
    println!("\n❓ Query: {}", query_text);

    let query_embedding = mock_embed(query_text);
    let results = store.query(Query {
        vector: query_embedding,
        k: 3,
        filter: None,
    })?;

    println!("   📄 Top {} results:", results.len());
    for (i, result) in results.iter().enumerate() {
        let text = result
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");
        let page = result
            .metadata
            .fields
            .get("page")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        println!("      {}. Page {}, Score: {:.3}", i + 1, page, result.score);
        println!("         {}\n", &text[..text.len().min(100)]);
    }

    println!("✅ PDF RAG Example Complete!");
    println!("\n💡 Next Steps:");
    println!("   • Use PDFLoader to load actual PDF files");
    println!("   • Try different chunk sizes for your documents");
    println!("   • Add page number extraction from PDF");
    println!("   • Implement citation tracking (chunk → page number)");

    Ok(())
}

#[cfg(feature = "loaders")]
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
