//! Basic RAG (Retrieval-Augmented Generation) Example
//!
//! Demonstrates the simplest RAG workflow:
//! 1. Load documents
//! 2. Split into chunks
//! 3. Embed and store
//! 4. Query and retrieve
//! 5. Generate answer (simulated)
//!
//! Run with: cargo run --example 01_basic_rag

use anyhow::Result;
use std::collections::HashMap;
use vecstore::{
    text_splitter::{RecursiveCharacterTextSplitter, TextSplitter},
    Metadata, Query, VecStore,
};

fn main() -> Result<()> {
    println!("📚 Basic RAG Example\n");
    println!("This example shows the simplest RAG workflow.\n");

    // Step 1: Prepare sample documents
    println!("Step 1: Loading sample documents...");
    let documents = vec![
        (
            "doc1",
            "Rust is a systems programming language that runs blazingly fast, \
                  prevents segfaults, and guarantees thread safety. It accomplishes \
                  these goals by being memory safe without using garbage collection.",
        ),
        (
            "doc2",
            "The Rust compiler is known for its helpful error messages. When you \
                  make a mistake, the compiler provides detailed explanations and suggestions \
                  for how to fix it.",
        ),
        (
            "doc3",
            "Rust's ownership system is unique among programming languages. Every value \
                  has a single owner, and when the owner goes out of scope, the value is dropped.",
        ),
        (
            "doc4",
            "Cargo is Rust's build system and package manager. It handles building code, \
                  downloading dependencies, and building those dependencies.",
        ),
    ];

    println!("   ✓ Loaded {} documents\n", documents.len());

    // Step 2: Split documents into chunks
    println!("Step 2: Splitting documents into chunks...");
    let splitter = RecursiveCharacterTextSplitter::new(200, 20);

    let mut all_chunks = Vec::new();
    for (doc_id, text) in &documents {
        let chunks = splitter.split_text(text)?;
        for (i, chunk) in chunks.into_iter().enumerate() {
            all_chunks.push((format!("{}_{}", doc_id, i), chunk));
        }
    }

    println!("   ✓ Created {} chunks\n", all_chunks.len());

    // Step 3: Create vector store and embed chunks
    println!("Step 3: Creating vector store...");
    let mut store = VecStore::open("./data/01_basic_rag")?;

    println!("Step 4: Embedding and storing chunks...");
    println!("   (In production, use real embeddings. We'll use mock embeddings here)");

    for (chunk_id, text) in &all_chunks {
        // In production: Use real embedding model (OpenAI, sentence-transformers, etc.)
        // For demo: Create simple mock embeddings
        let embedding = create_mock_embedding(text);

        // Store with metadata
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("text".to_string(), serde_json::json!(text));
        metadata
            .fields
            .insert("length".to_string(), serde_json::json!(text.len()));

        store.upsert(chunk_id.clone(), embedding, metadata)?;
    }

    println!("   ✓ Stored {} embeddings\n", all_chunks.len());

    // Step 5: Query the RAG system
    println!("Step 5: Querying the RAG system...");
    let queries = vec![
        "What is Rust's ownership system?",
        "How does Rust prevent memory bugs?",
        "What is Cargo?",
    ];

    for query_text in &queries {
        println!("\n❓ Query: {}", query_text);

        // In production: Embed the query with the same model
        let query_embedding = create_mock_embedding(query_text);

        // Retrieve top-k most similar chunks
        let query = Query {
            vector: query_embedding,
            k: 3,
            filter: None,
        };

        let results = store.query(query)?;

        println!("   📄 Retrieved {} relevant chunks:", results.len());
        for (i, result) in results.iter().enumerate() {
            let text = result
                .metadata
                .fields
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("N/A");
            println!(
                "      {}. Score: {:.3} - {}",
                i + 1,
                result.score,
                &text[..text.len().min(80)]
            );
        }

        // Step 6: Generate answer (simulated)
        println!("\n   🤖 Simulated LLM Answer:");
        println!(
            "      Based on the retrieved context, {}",
            generate_mock_answer(query_text)
        );
    }

    println!("\n\n✅ Basic RAG Example Complete!");
    println!("\n💡 Next Steps:");
    println!("   • Replace mock embeddings with real embeddings (OpenAI, etc.)");
    println!("   • Add actual LLM for answer generation");
    println!("   • Try different text splitters (semantic, markdown-aware)");
    println!("   • Experiment with different chunk sizes and overlap");
    println!("   • Add metadata filtering for more precise retrieval");

    Ok(())
}

// ============================================================================
// Mock Functions (Replace with real implementations in production)
// ============================================================================

/// Create mock embedding from text
/// In production: Use OpenAI, sentence-transformers, or other embedding models
fn create_mock_embedding(text: &str) -> Vec<f32> {
    // Simple mock: Create deterministic embedding based on text content
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut embedding = vec![0.0; 384]; // Typical embedding dimension

    for (i, word) in words.iter().enumerate() {
        let hash = word.len() * (i + 1);
        embedding[hash % 384] += 1.0;
    }

    // Normalize
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in &mut embedding {
            *val /= magnitude;
        }
    }

    embedding
}

/// Generate mock answer
/// In production: Use LLM (GPT-4, Claude, Llama, etc.) with retrieved context
fn generate_mock_answer(query: &str) -> &'static str {
    if query.contains("ownership") {
        "Rust's ownership system ensures memory safety by tracking who owns each value \
         and automatically cleaning up when owners go out of scope."
    } else if query.contains("memory") {
        "Rust prevents memory bugs through its ownership system and borrow checker, \
         which catch issues at compile time without needing garbage collection."
    } else if query.contains("Cargo") {
        "Cargo is Rust's build system and package manager that handles compilation, \
         dependencies, and project management."
    } else {
        "Rust is a fast, safe systems programming language with great tooling."
    }
}
