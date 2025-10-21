//! Embedding Integration Demo
//!
//! This example demonstrates VecStore's Phase 5: Embedding Integration.
//! Shows how to use the TextEmbedder trait, SimpleEmbedder, and EmbeddingCollection
//! for seamless text-to-vector workflows.
//!
//! Features demonstrated:
//! - TextEmbedder trait abstraction
//! - SimpleEmbedder for testing (no ONNX required)
//! - EmbeddingCollection for text-based APIs
//! - Batch text processing
//! - Custom embedder implementation
//!
//! Run with: cargo run --example embedding_integration_demo --features embeddings

#![cfg_attr(not(feature = "embeddings"), allow(dead_code, unused_imports))]

#[cfg(feature = "embeddings")]
use vecstore::{
    embeddings::{EmbeddingCollection, SimpleEmbedder, TextEmbedder},
    Metadata, VecDatabase,
};

#[cfg(feature = "embeddings")]
use std::collections::HashMap;

#[cfg(feature = "embeddings")]
fn main() -> anyhow::Result<()> {
    println!("=== VecStore Embedding Integration Demo ===\n");

    // ========== 1. SimpleEmbedder Basics ==========
    println!("1. SimpleEmbedder - No ONNX Required\n");

    let simple_embedder = SimpleEmbedder::new(128);

    let text = "Rust is a systems programming language";
    let embedding = simple_embedder.embed(text)?;

    println!("   Text: \"{}\"", text);
    println!("   Embedding dimension: {}", embedding.len());
    println!("   First 5 values: {:?}\n", &embedding[0..5]);

    // ========== 2. TextEmbedder Trait ==========
    println!("2. TextEmbedder Trait Abstraction\n");

    // Can use as a trait object for pluggable embedders
    let embedder: Box<dyn TextEmbedder> = Box::new(SimpleEmbedder::new(128));

    let texts = vec!["First text", "Second text", "Third text"];
    let batch_embeddings = embedder.embed_batch(&texts)?;

    println!("   Batch embedded {} texts", batch_embeddings.len());
    println!(
        "   Each embedding has {} dimensions\n",
        embedder.dimension()?
    );

    // ========== 3. EmbeddingCollection - Text APIs ==========
    println!("3. EmbeddingCollection - Seamless Text-to-Vector\n");

    let mut db = VecDatabase::open("./demo_embedding")?;
    let collection = db.create_collection("documents")?;

    // Wrap collection with embedder
    let embedder = SimpleEmbedder::new(128);
    let mut emb_collection = EmbeddingCollection::new(collection, Box::new(embedder));

    println!("   Created EmbeddingCollection with SimpleEmbedder\n");

    // ========== 4. Inserting Text Documents ==========
    println!("4. Inserting Text Documents (Auto-Embedding)\n");

    // Insert documents with text - embeddings generated automatically
    let docs = vec![
        (
            "rust_doc",
            "Rust is a fast and memory-safe systems programming language",
            "programming",
            "rust",
        ),
        (
            "python_doc",
            "Python is a high-level interpreted programming language",
            "programming",
            "python",
        ),
        (
            "javascript_doc",
            "JavaScript is the programming language of the web",
            "programming",
            "javascript",
        ),
        (
            "database_doc",
            "Databases store and organize data efficiently",
            "data",
            "database",
        ),
        (
            "ml_doc",
            "Machine learning enables computers to learn from data",
            "ai",
            "machine-learning",
        ),
    ];

    for (id, text, category, tag) in &docs {
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields.insert("text".into(), serde_json::json!(text));
        meta.fields
            .insert("category".into(), serde_json::json!(category));
        meta.fields.insert("tag".into(), serde_json::json!(tag));

        emb_collection.upsert_text(*id, text, meta)?;
        println!("   ✓ Inserted: {} ({})", id, category);
    }

    println!("\n   Total documents: {}\n", emb_collection.count()?);

    // ========== 5. Querying with Text ==========
    println!("5. Querying with Text (Auto-Embedding)\n");

    let queries = vec![
        "programming language",
        "data storage",
        "artificial intelligence",
    ];

    for query in &queries {
        println!("   Query: \"{}\"", query);
        let results = emb_collection.query_text(query, 3, None)?;

        for (i, result) in results.iter().enumerate() {
            let text = result
                .metadata
                .fields
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("N/A");
            println!("     {}. {} (score: {:.4})", i + 1, result.id, result.score);
            println!("        \"{}\"", &text.chars().take(50).collect::<String>());
        }
        println!();
    }

    // ========== 6. Batch Insert ==========
    println!("6. Batch Text Insert (Efficient Bulk Loading)\n");

    let mut new_db = VecDatabase::open("./demo_embedding_batch")?;
    let new_collection = new_db.create_collection("batch_docs")?;
    let new_embedder = SimpleEmbedder::new(128);
    let mut batch_collection = EmbeddingCollection::new(new_collection, Box::new(new_embedder));

    let batch_docs = vec![
        ("doc1".to_string(), "First batch document".to_string(), {
            let mut m = Metadata {
                fields: HashMap::new(),
            };
            m.fields.insert("batch".into(), serde_json::json!(1));
            m
        }),
        ("doc2".to_string(), "Second batch document".to_string(), {
            let mut m = Metadata {
                fields: HashMap::new(),
            };
            m.fields.insert("batch".into(), serde_json::json!(1));
            m
        }),
        ("doc3".to_string(), "Third batch document".to_string(), {
            let mut m = Metadata {
                fields: HashMap::new(),
            };
            m.fields.insert("batch".into(), serde_json::json!(1));
            m
        }),
    ];

    batch_collection.batch_upsert_text(batch_docs)?;
    println!(
        "   ✓ Batch inserted {} documents",
        batch_collection.count()?
    );
    println!("   (More efficient than individual upserts)\n");

    // ========== 7. Custom Embedder Implementation ==========
    println!("7. Custom Embedder Implementation\n");

    struct FixedEmbedder;

    impl TextEmbedder for FixedEmbedder {
        fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            // Custom logic: create embedding based on text length
            let len = text.len() as f32;
            let mut vec = vec![len / 100.0; 128];

            // Add some variation based on first few chars
            for (i, c) in text.chars().take(10).enumerate() {
                vec[i] = (c as u32 as f32) / 1000.0;
            }

            // Normalize
            let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                for v in &mut vec {
                    *v /= magnitude;
                }
            }

            Ok(vec)
        }

        fn dimension(&self) -> anyhow::Result<usize> {
            Ok(128)
        }
    }

    let custom_embedder: Box<dyn TextEmbedder> = Box::new(FixedEmbedder);
    let custom_embedding = custom_embedder.embed("Test custom embedder")?;

    println!(
        "   Custom embedder dimension: {}",
        custom_embedder.dimension()?
    );
    println!(
        "   Generated embedding with {} values",
        custom_embedding.len()
    );
    println!("   Can be used with EmbeddingCollection!\n");

    // ========== 8. Comparison with Manual Embedding ==========
    println!("8. Comparison: Manual vs Automatic Embedding\n");

    println!("   Manual approach:");
    println!("     1. text → embedder.embed(text) → vector");
    println!("     2. collection.upsert(id, vector, metadata)");
    println!();
    println!("   With EmbeddingCollection:");
    println!("     1. emb_collection.upsert_text(id, text, metadata)");
    println!("     (Embedding happens automatically!)\n");

    // ========== 9. Collection Statistics ==========
    println!("9. Collection Statistics\n");

    let stats = emb_collection.stats()?;
    println!("   Total vectors: {}", stats.vector_count);
    println!("   Active vectors: {}", stats.active_count);

    // ========== Summary ==========
    println!("\n=== Summary ===");
    println!();
    println!("Phase 5 Embedding Integration provides:");
    println!("  ✓ TextEmbedder trait - Pluggable embedding models");
    println!("  ✓ SimpleEmbedder - Testing without ONNX Runtime");
    println!("  ✓ EmbeddingCollection - Text-based APIs for collections");
    println!("  ✓ Batch processing - Efficient bulk operations");
    println!("  ✓ Custom embedders - Implement your own embedding logic");
    println!();
    println!("Perfect for:");
    println!("  • RAG applications with automatic embedding");
    println!("  • Testing without external dependencies");
    println!("  • Custom embedding model integration");
    println!("  • Simplified text-to-vector workflows");
    println!();
    println!("Next steps:");
    println!("  • Use Embedder + ONNX Runtime for production models");
    println!("  • Combine with text_splitter for document chunking");
    println!("  • Integrate with your embedding API (OpenAI, Cohere, etc.)");

    // Cleanup
    std::fs::remove_dir_all("./demo_embedding").ok();
    std::fs::remove_dir_all("./demo_embedding_batch").ok();

    Ok(())
}

#[cfg(not(feature = "embeddings"))]
fn main() {
    eprintln!("This example requires the 'embeddings' feature.");
    eprintln!("Run with: cargo run --example embedding_integration_demo --features embeddings");
    std::process::exit(1);
}
