//! Text Chunking for RAG Demo
//!
//! This example demonstrates VecStore's text splitting capabilities,
//! essential for RAG (Retrieval-Augmented Generation) applications.
//!
//! Text chunking solves the problem of storing long documents in vector
//! databases with fixed-size embeddings. This example shows:
//! - Recursive character-based splitting
//! - Token-based splitting
//! - Overlap for context continuity
//! - Integration with VecStore
//!
//! Run with: cargo run --example text_chunking_demo

use std::collections::HashMap;
use vecstore::{
    text_splitter::{RecursiveCharacterTextSplitter, TextSplitter, TokenTextSplitter},
    Metadata, Query, VecStore,
};

const SAMPLE_DOCUMENT: &str = r#"
Rust is a multi-paradigm, high-level, general-purpose programming language. Rust emphasizes performance, type safety, and concurrency.

Rust enforces memory safety—meaning that all references point to valid memory—without a garbage collector. To simultaneously enforce memory safety and prevent data races, its "borrow checker" tracks the object lifetime of all references in a program during compilation.

Rust has been voted the "most loved programming language" in the Stack Overflow Developer Survey every year since 2016.

The primary design goals of Rust are:
- Memory safety without garbage collection
- Concurrency without data races
- Zero-cost abstractions
- Minimal runtime
- Efficient C bindings

Rust is syntactically similar to C++, but can guarantee memory safety using a borrow checker to validate references. Unlike other safe programming languages, Rust does not use garbage collection.

Rust has been used to build a variety of software, including:
- Operating systems (like Redox)
- Game engines
- Web browsers (Firefox's Servo engine)
- Command-line tools
- Embedded systems
- WebAssembly applications
"#;

fn main() -> anyhow::Result<()> {
    println!("=== VecStore Text Chunking Demo ===\n");

    // ========== 1. Character-Based Splitting ==========
    println!("1. Character-Based Text Splitting\n");

    let char_splitter = RecursiveCharacterTextSplitter::new(300, 50);
    let chunks = char_splitter.split_text(SAMPLE_DOCUMENT)?;

    println!("   Original document: {} chars", SAMPLE_DOCUMENT.len());
    println!(
        "   Split into {} chunks (300 chars, 50 overlap)\n",
        chunks.len()
    );

    for (i, chunk) in chunks.iter().take(3).enumerate() {
        println!("   Chunk {}:", i + 1);
        println!("   Length: {} chars", chunk.len());
        println!(
            "   Preview: {}...\n",
            &chunk.chars().take(80).collect::<String>()
        );
    }

    // ========== 2. Token-Based Splitting ==========
    println!("2. Token-Based Text Splitting\n");

    let token_splitter = TokenTextSplitter::new(100, 10); // ~100 tokens = 400 chars
    let token_chunks = token_splitter.split_text(SAMPLE_DOCUMENT)?;

    println!(
        "   Split into {} chunks (~100 tokens each)\n",
        token_chunks.len()
    );

    for (i, chunk) in token_chunks.iter().take(2).enumerate() {
        let estimated_tokens = chunk.len() / 4; // Approximation
        println!("   Chunk {}:", i + 1);
        println!(
            "   Length: {} chars (~{} tokens)",
            chunk.len(),
            estimated_tokens
        );
        println!(
            "   Preview: {}...\n",
            &chunk.chars().take(60).collect::<String>()
        );
    }

    // ========== 3. Custom Separators ==========
    println!("3. Custom Separators (Sentence-First)\n");

    let sentence_splitter = RecursiveCharacterTextSplitter::new(200, 20).with_separators(vec![
        ". ".to_string(),
        "! ".to_string(),
        "? ".to_string(),
        "\n\n".to_string(),
        "\n".to_string(),
        " ".to_string(),
    ]);

    let sentence_chunks = sentence_splitter.split_text(SAMPLE_DOCUMENT)?;

    println!(
        "   Sentence-first splitting: {} chunks",
        sentence_chunks.len()
    );
    println!(
        "   First chunk: {}...\n",
        &sentence_chunks[0].chars().take(100).collect::<String>()
    );

    // ========== 4. Chunk Overlap Demonstration ==========
    println!("4. Chunk Overlap for Context Continuity\n");

    println!("   Without overlap:");
    let no_overlap = RecursiveCharacterTextSplitter::new(150, 0);
    let chunks_no_overlap = no_overlap.split_text("The quick brown fox jumps over the lazy dog. The dog was sleeping under a tree. The tree provided shade on a hot summer day.")?;

    for (i, chunk) in chunks_no_overlap.iter().enumerate() {
        println!("     Chunk {}: {}", i + 1, chunk);
    }

    println!("\n   With 20 char overlap:");
    let with_overlap = RecursiveCharacterTextSplitter::new(150, 20);
    let chunks_with_overlap = with_overlap.split_text("The quick brown fox jumps over the lazy dog. The dog was sleeping under a tree. The tree provided shade on a hot summer day.")?;

    for (i, chunk) in chunks_with_overlap.iter().enumerate() {
        println!("     Chunk {}: {}", i + 1, chunk);
    }
    println!();

    // ========== 5. Integration with VecStore ==========
    println!("5. Integration with VecStore (RAG Pattern)\n");

    let mut store = VecStore::open("./demo_rag")?;

    println!("   Chunking document and storing with embeddings...\n");

    let splitter = RecursiveCharacterTextSplitter::new(300, 50);
    let doc_chunks = splitter.split_text(SAMPLE_DOCUMENT)?;

    for (i, chunk) in doc_chunks.iter().enumerate() {
        // In a real RAG application, you'd use an embedding model here
        // For demo purposes, we'll use simple random vectors
        let embedding = generate_mock_embedding(chunk);

        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields
            .insert("chunk_index".into(), serde_json::json!(i));
        meta.fields.insert("text".into(), serde_json::json!(chunk));
        meta.fields
            .insert("source".into(), serde_json::json!("rust_doc"));

        let chunk_id = format!("rust_doc_chunk_{}", i);
        store.upsert(chunk_id, embedding, meta)?;
    }

    println!(
        "   ✓ Stored {} document chunks in VecStore",
        doc_chunks.len()
    );

    // Query the chunked document
    let query_embedding = generate_mock_embedding("memory safety");
    let query = Query {
        vector: query_embedding,
        k: 3,
        filter: None,
    };

    let results = store.query(query)?;

    println!("\n   Query: 'memory safety'");
    println!("   Top {} relevant chunks:\n", results.len());

    for (i, result) in results.iter().enumerate() {
        let chunk_text = result
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");

        println!("   Result {}:", i + 1);
        println!("   Score: {:.4}", result.score);
        println!(
            "   Text: {}...\n",
            &chunk_text.chars().take(80).collect::<String>()
        );
    }

    // ========== 6. Chunk Statistics ==========
    println!("6. Chunk Statistics\n");

    let chunks_for_stats = char_splitter.split_text(SAMPLE_DOCUMENT)?;

    let total_chunks = chunks_for_stats.len();
    let avg_length = chunks_for_stats.iter().map(|c| c.len()).sum::<usize>() / total_chunks;
    let min_length = chunks_for_stats.iter().map(|c| c.len()).min().unwrap_or(0);
    let max_length = chunks_for_stats.iter().map(|c| c.len()).max().unwrap_or(0);

    println!("   Total chunks: {}", total_chunks);
    println!("   Average length: {} chars", avg_length);
    println!("   Min length: {} chars", min_length);
    println!("   Max length: {} chars", max_length);
    println!(
        "   Total coverage: {} chars",
        chunks_for_stats.iter().map(|c| c.len()).sum::<usize>()
    );
    println!("   Original doc: {} chars\n", SAMPLE_DOCUMENT.len());

    // ========== Summary ==========
    println!("=== Summary ===");
    println!();
    println!("Text chunking is essential for RAG applications:");
    println!("  ✓ RecursiveCharacterTextSplitter - Natural boundary splitting");
    println!("  ✓ TokenTextSplitter - LLM token-aware splitting");
    println!("  ✓ Overlap - Maintains context between chunks");
    println!("  ✓ VecStore integration - Store and search chunks");
    println!();
    println!("Typical RAG workflow:");
    println!("  1. Chunk documents (300-500 chars, 50-100 overlap)");
    println!("  2. Generate embeddings for each chunk");
    println!("  3. Store chunks in VecStore with metadata");
    println!("  4. Query with question embedding");
    println!("  5. Retrieve relevant chunks for LLM context");
    println!();
    println!("Perfect for: Documentation search, Q&A, content retrieval");

    // Cleanup
    std::fs::remove_dir_all("./demo_rag").ok();

    Ok(())
}

/// Generate a mock embedding (in real apps, use a model like sentence-transformers)
fn generate_mock_embedding(text: &str) -> Vec<f32> {
    // Simple deterministic "embedding" based on text content
    let mut vec = vec![0.0; 128];

    for (i, c) in text.chars().take(128).enumerate() {
        vec[i] = (c as u32 as f32) / 1000.0;
    }

    // Normalize
    let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for v in &mut vec {
            *v /= magnitude;
        }
    }

    vec
}
