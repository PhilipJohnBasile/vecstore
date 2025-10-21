//! Complete RAG pipeline using LangChain integration
//!
//! This example demonstrates:
//! - LangChain vector store integration
//! - Document ingestion with automatic chunking
//! - Retrieval-Augmented Generation (RAG)
//! - MMR (Maximal Marginal Relevance) for diverse results
//! - Health monitoring
//! - Performance benchmarking

use anyhow::Result;
use vecstore::error::VecStoreError;
use vecstore::{
    print_health_report, Document, HealthChecker, LangChainVectorStore, RetrieverConfig, VecStore,
    VectorStoreRetriever,
};

fn main() -> Result<()> {
    println!("🚀 LangChain RAG Pipeline Example\n");
    println!("{}", "=".repeat(80));

    // Step 1: Create a vector store
    println!("\n[1/6] Creating vector store...");
    let store = VecStore::open("data/langchain_rag.db")?;

    // Step 2: Create LangChain-compatible vector store with embedding function
    println!("[2/6] Setting up LangChain integration...");
    let mut lc_store = LangChainVectorStore::new(store).with_embedding_fn(|text: &str| {
        simple_embedding(text).map_err(|e| VecStoreError::Serialization(e.to_string()))
    });

    // Step 3: Ingest documents
    println!("[3/6] Ingesting documents...");
    let documents = vec![
        Document::new("Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.")
            .with_metadata("source", serde_json::json!("rust_intro.txt"))
            .with_metadata("category", serde_json::json!("programming")),

        Document::new("Vector databases are specialized databases designed to store and query high-dimensional vector embeddings efficiently.")
            .with_metadata("source", serde_json::json!("vector_db.txt"))
            .with_metadata("category", serde_json::json!("databases")),

        Document::new("Machine learning models can convert text, images, and other data into vector representations called embeddings.")
            .with_metadata("source", serde_json::json!("ml_embeddings.txt"))
            .with_metadata("category", serde_json::json!("machine_learning")),

        Document::new("RAG (Retrieval-Augmented Generation) combines retrieval from a knowledge base with language model generation.")
            .with_metadata("source", serde_json::json!("rag.txt"))
            .with_metadata("category", serde_json::json!("ai")),

        Document::new("HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search.")
            .with_metadata("source", serde_json::json!("hnsw.txt"))
            .with_metadata("category", serde_json::json!("algorithms")),

        Document::new("Semantic search uses vector embeddings to find conceptually similar content, not just keyword matches.")
            .with_metadata("source", serde_json::json!("semantic_search.txt"))
            .with_metadata("category", serde_json::json!("search")),
    ];

    let doc_ids = lc_store.add_documents(documents)?;
    println!("   ✓ Ingested {} documents", doc_ids.len());

    // Step 4: Perform similarity search
    println!("\n[4/6] Running similarity searches...");

    let query = "How do vector databases work?";
    println!("\n   Query: \"{}\"", query);

    let results = lc_store.similarity_search(query, 3, None)?;
    println!("   Top {} results:", results.len());
    for (i, doc) in results.iter().enumerate() {
        println!("     {}. {}", i + 1, truncate(&doc.page_content, 60));
        if let Some(source) = doc.get_metadata("source") {
            println!("        Source: {}", source);
        }
    }

    // Step 5: MMR search for diverse results
    println!("\n[5/6] Running MMR search for diverse results...");

    let mmr_results = lc_store.max_marginal_relevance_search(
        "Tell me about search algorithms",
        3,    // k results
        10,   // fetch k candidates
        0.5,  // lambda (balance relevance and diversity)
        None, // no filter
    )?;

    println!("   MMR results (balanced relevance + diversity):");
    for (i, doc) in mmr_results.iter().enumerate() {
        println!("     {}. {}", i + 1, truncate(&doc.page_content, 60));
    }

    // Step 6: Use retriever pattern
    println!("\n[6/6] Using retriever pattern...");

    let retriever = VectorStoreRetriever::new(lc_store).with_config(RetrieverConfig {
        k: 2,
        filter: Some("category = 'programming'".to_string()),
        use_mmr: false,
        mmr_lambda: 0.5,
        fetch_k: 10,
    });

    let filtered_results = retriever.get_relevant_documents("programming languages")?;
    println!("   Filtered results (category = 'programming'):");
    for (i, doc) in filtered_results.iter().enumerate() {
        println!("     {}. {}", i + 1, truncate(&doc.page_content, 60));
    }

    // Health check
    println!("\n{}", "=".repeat(80));
    println!("📊 Health Check");
    println!("{}", "=".repeat(80));

    let checker = HealthChecker::default();

    // Get a reference to the underlying store for health check
    let inner_store = VecStore::open("data/langchain_rag.db")?;
    let health_report = checker.check(&inner_store)?;
    print_health_report(&health_report);

    println!("\n✅ RAG pipeline complete!");
    println!("\n💡 Key Features Demonstrated:");
    println!("   • Document ingestion with metadata");
    println!("   • Similarity search");
    println!("   • MMR for diverse results");
    println!("   • Retriever pattern with filters");
    println!("   • Health monitoring");

    Ok(())
}

/// Simple embedding function for demonstration
/// In production, use a real embedding model (OpenAI, Sentence Transformers, etc.)
fn simple_embedding(text: &str) -> Result<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Create a simple but deterministic embedding based on text content
    let mut embedding = vec![0.0; 128];

    // Hash the text
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let hash = hasher.finish();

    // Use the hash to seed a simple pseudo-random generator
    let mut seed = hash;
    for i in 0..128 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        embedding[i] = ((seed >> 16) as f32 / 32768.0) - 1.0;
    }

    // Normalize
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in &mut embedding {
            *val /= magnitude;
        }
    }

    Ok(embedding)
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
