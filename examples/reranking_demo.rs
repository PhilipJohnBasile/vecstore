//! Reranking Demo
//!
//! Demonstrates how to use reranking to improve search result quality.
//!
//! ## What is Reranking?
//!
//! Reranking is a post-processing step that refines initial search results using
//! more sophisticated (but slower) models or algorithms.
//!
//! ## Why Reranking?
//!
//! 1. **Improves Precision**: Initial vector search casts a wide net, reranking refines
//! 2. **Balances Speed vs Quality**: Fast first-stage retrieval + slower reranking
//! 3. **Adds Diversity**: MMR prevents redundant results
//! 4. **Semantic Understanding**: Cross-encoders capture query-document interaction
//!
//! ## Run this example:
//! ```bash
//! cargo run --example reranking_demo --features embeddings
//! ```

use std::collections::HashMap;
use vecstore::{
    reranking::{CrossEncoderFn, IdentityReranker, MMRReranker, Reranker, ScoreReranker},
    Metadata, Query, VecDatabase,
};

fn main() -> anyhow::Result<()> {
    println!("=== VecStore Reranking Demo ===\n");

    // Setup
    let temp_dir = std::env::temp_dir().join("vecstore_reranking_demo");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let mut db = VecDatabase::open(&temp_dir)?;
    let mut collection = db.create_collection("documents")?;

    // Insert sample documents
    println!("📚 Inserting sample documents...\n");

    let documents = vec![
        (
            "doc1",
            vec![1.0, 0.0, 0.0],
            "Rust is a systems programming language focused on safety and performance",
        ),
        (
            "doc2",
            vec![0.9, 0.1, 0.0],
            "Rust programming language provides memory safety without garbage collection",
        ),
        (
            "doc3",
            vec![0.8, 0.2, 0.0],
            "The Rust compiler ensures thread safety and prevents data races",
        ),
        (
            "doc4",
            vec![0.5, 0.5, 0.0],
            "Python is a high-level programming language known for readability",
        ),
        (
            "doc5",
            vec![0.4, 0.6, 0.0],
            "JavaScript is the programming language of the web and Node.js",
        ),
        (
            "doc6",
            vec![0.95, 0.05, 0.0],
            "Rust async programming with tokio enables high-performance concurrent applications",
        ),
    ];

    for (id, vector, text) in &documents {
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields
            .insert("text".to_string(), serde_json::json!(text));
        collection.upsert(id.to_string(), vector.clone(), meta)?;
    }

    println!("✅ Inserted {} documents\n", documents.len());

    // Perform initial search
    let query_vector = vec![1.0, 0.0, 0.0]; // Looking for Rust-related documents
    let initial_results = collection.query(Query {
        vector: query_vector,
        k: 6, // Get all results
        filter: None,
    })?;

    println!("🔍 Initial Search Results (Vector Similarity Only):");
    println!("─────────────────────────────────────────────────────");
    for (i, result) in initial_results.iter().enumerate() {
        let text = result
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!(
            "{}. [Score: {:.3}] {}: {}",
            i + 1,
            result.score,
            result.id,
            &text[..text.len().min(60)]
        );
    }
    println!();

    // Demo 1: MMR Reranking (Diversity)
    println!("\n📊 Demo 1: MMR Reranking (Diversity-Based)");
    println!("═══════════════════════════════════════════");

    let mmr_reranker = MMRReranker::new(0.7); // 70% relevance, 30% diversity
    let mmr_results = mmr_reranker.rerank("rust programming", initial_results.clone(), 3)?;

    println!("MMR Results (λ=0.7 - balances relevance and diversity):");
    println!("─────────────────────────────────────────────────────");
    for (i, result) in mmr_results.iter().enumerate() {
        let text = result
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!(
            "{}. [Score: {:.3}] {}: {}",
            i + 1,
            result.score,
            result.id,
            &text[..text.len().min(60)]
        );
    }
    println!("\n💡 Note: MMR prevents redundant results by penalizing similarity to");
    println!("   already-selected documents, ensuring diverse coverage.\n");

    // Demo 2: Cross-Encoder Reranking (Semantic)
    println!("\n📊 Demo 2: Cross-Encoder Reranking (Semantic)");
    println!("═══════════════════════════════════════════");

    // Simple word overlap scorer (in production, use a real cross-encoder model)
    let cross_encoder = CrossEncoderFn::new(|query: &str, doc: &str| {
        let query_words: std::collections::HashSet<&str> = query.split_whitespace().collect();
        let doc_words: std::collections::HashSet<&str> = doc.split_whitespace().collect();

        // Weighted scoring: exact matches + partial matches
        let exact_matches = query_words.intersection(&doc_words).count() as f32;
        let doc_length = doc_words.len() as f32;

        // Score: (exact_matches / query_length) * (1 + 1/doc_length)
        // Favors documents with high word overlap and conciseness
        (exact_matches / query_words.len() as f32) * (1.0 + 1.0 / doc_length)
    });

    let ce_results =
        cross_encoder.rerank("rust programming language", initial_results.clone(), 3)?;

    println!("Cross-Encoder Results (word overlap scorer):");
    println!("─────────────────────────────────────────────");
    for (i, result) in ce_results.iter().enumerate() {
        let text = result
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!(
            "{}. [Score: {:.3}] {}: {}",
            i + 1,
            result.score,
            result.id,
            &text[..text.len().min(60)]
        );
    }
    println!("\n💡 Note: Cross-encoders process query+document together, capturing");
    println!("   semantic interactions that bi-encoders (vectors) miss.\n");

    // Demo 3: Score-Based Reranking (Custom Logic)
    println!("\n📊 Demo 3: Score-Based Reranking (Custom Logic)");
    println!("═══════════════════════════════════════════");

    // Boost recent or important documents
    let score_reranker = ScoreReranker::new(|neighbor| {
        let base_score = neighbor.score;

        // Boost if document mentions "safety" or "performance"
        let text = neighbor
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let safety_boost = if text.contains("safety") { 0.2 } else { 0.0 };
        let perf_boost = if text.contains("performance") {
            0.1
        } else {
            0.0
        };

        base_score + safety_boost + perf_boost
    });

    let score_results = score_reranker.rerank("rust", initial_results.clone(), 3)?;

    println!("Score-Based Results (boosting 'safety' and 'performance'):");
    println!("──────────────────────────────────────────────────────────");
    for (i, result) in score_results.iter().enumerate() {
        let text = result
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!(
            "{}. [Score: {:.3}] {}: {}",
            i + 1,
            result.score,
            result.id,
            &text[..text.len().min(60)]
        );
    }
    println!("\n💡 Note: Custom scoring functions allow domain-specific ranking logic");
    println!("   like recency, authority, user preferences, etc.\n");

    // Demo 4: Chained Reranking
    println!("\n📊 Demo 4: Chained Reranking (Multi-Stage)");
    println!("═══════════════════════════════════════════");

    // Stage 1: Vector search (already done)
    // Stage 2: MMR for diversity
    // Stage 3: Cross-encoder for final ranking

    let stage1_results = initial_results.clone();
    println!("Stage 1: Vector Search → {} results", stage1_results.len());

    let mmr = MMRReranker::new(0.6);
    let stage2_results = mmr.rerank("rust programming", stage1_results, 4)?;
    println!(
        "Stage 2: MMR (diversity) → {} results",
        stage2_results.len()
    );

    let ce = CrossEncoderFn::new(|query: &str, doc: &str| {
        query.split_whitespace().filter(|w| doc.contains(w)).count() as f32
    });
    let final_results = ce.rerank("rust programming", stage2_results, 2)?;
    println!(
        "Stage 3: Cross-Encoder (semantic) → {} results\n",
        final_results.len()
    );

    println!("Final Results (after 3-stage reranking):");
    println!("────────────────────────────────────────");
    for (i, result) in final_results.iter().enumerate() {
        let text = result
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!(
            "{}. [Score: {:.3}] {}: {}",
            i + 1,
            result.score,
            result.id,
            text
        );
    }
    println!("\n💡 Note: Multi-stage reranking combines strengths:");
    println!("   1. Fast vector search for recall");
    println!("   2. MMR for diversity");
    println!("   3. Cross-encoder for precision\n");

    // Demo 5: Trait Objects (Dynamic Dispatch)
    println!("\n📊 Demo 5: Dynamic Reranker Selection");
    println!("═══════════════════════════════════════");

    let rerankers: Vec<Box<dyn Reranker>> = vec![
        Box::new(IdentityReranker),
        Box::new(MMRReranker::new(0.7)),
        Box::new(ScoreReranker::new(|n| n.score)),
    ];

    for reranker in rerankers {
        let results = reranker.rerank("test query", initial_results.clone(), 2)?;
        println!("  {} → {} results", reranker.name(), results.len());
    }
    println!("\n💡 Note: Trait objects enable runtime reranker selection\n");

    // Cleanup
    std::fs::remove_dir_all(&temp_dir).ok();

    println!("\n✅ Demo Complete!\n");
    println!("Key Takeaways:");
    println!("─────────────");
    println!("1. MMR balances relevance and diversity");
    println!("2. Cross-encoders provide semantic understanding");
    println!("3. Score-based reranking allows custom business logic");
    println!("4. Multi-stage reranking combines multiple strategies");
    println!("5. Trait abstraction enables flexible composition");
    println!("\n📚 See PHASE-7-COMPLETE.md for detailed documentation\n");

    Ok(())
}
