// OpenAI Embeddings - RAG Pipeline Example
//
// This example demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline
// using OpenAI embeddings for semantic search over documents.
//
// Requirements:
// - Set OPENAI_API_KEY environment variable
// - Run with: cargo run --example openai_rag --features "embeddings,openai-embeddings"

#![cfg_attr(
    not(all(feature = "embeddings", feature = "openai-embeddings")),
    allow(dead_code, unused_imports)
)]

use anyhow::Result;

#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};

#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
use vecstore::{FilterExpr, FilterOp, Metadata, Query, VecStore};

#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
use vecstore::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};

#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
#[tokio::main]
async fn main() -> Result<()> {
    println!("=============================================================");
    println!("OpenAI Embeddings - RAG Pipeline Example");
    println!("=============================================================\n");

    // 1. Check for API key
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("❌ Error: OPENAI_API_KEY environment variable not set");
            eprintln!("\nPlease set your OpenAI API key:");
            eprintln!("  export OPENAI_API_KEY='your-api-key-here'\n");
            std::process::exit(1);
        }
    };

    println!("✓ API key found\n");

    // 2. Create sample documents to index
    println!("Creating sample documents...");
    let documents = vec![
        ("rust_intro", "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. It accomplishes these goals without using a garbage collector."),
        ("rust_ownership", "Rust's ownership system is its most unique feature. The ownership rules ensure memory safety without needing a garbage collector. Each value has a variable that's its owner, and there can only be one owner at a time."),
        ("rust_cargo", "Cargo is Rust's build system and package manager. It handles building your code, downloading dependencies, and building those dependencies. Most Rustaceans use Cargo to manage their Rust projects."),
        ("python_intro", "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms and has a comprehensive standard library."),
        ("python_django", "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. It follows the model-template-views architectural pattern and includes an ORM."),
        ("javascript_intro", "JavaScript is a versatile programming language primarily used for web development. It runs in browsers and enables interactive web pages. Node.js allows JavaScript to run on servers too."),
        ("vecstore_intro", "VecStore is a high-performance vector database written in Rust. It provides efficient similarity search using HNSW indexing, supports multiple embedding backends, and offers both Rust and Python APIs."),
    ];

    println!("✓ Created {} documents", documents.len());
    for (id, _) in &documents {
        println!("  - {}", id);
    }
    println!();

    // 3. Initialize text splitter
    println!("Initializing text splitter...");
    let splitter = RecursiveCharacterTextSplitter::new(200, 50);
    println!("✓ Text splitter created (chunk_size: 200, overlap: 50)\n");

    // 4. Split documents into chunks
    println!("Splitting documents into chunks...");
    let mut all_chunks = Vec::new();
    let mut chunk_metadata = Vec::new();

    for (doc_id, text) in &documents {
        let chunks = splitter.split_text(text)?;
        println!("  {} → {} chunks", doc_id, chunks.len());

        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            chunk_metadata.push((doc_id.to_string(), chunk_idx, chunk.clone()));
            all_chunks.push(chunk);
        }
    }

    println!("✓ Total chunks: {}\n", all_chunks.len());

    // 5. Create OpenAI embedder
    println!("Creating OpenAI embedder...");
    let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small)
        .await?
        .with_rate_limit(500)
        .with_max_retries(3);

    println!("✓ Embedder created");
    println!("  Model: {}", embedder.model().as_str());
    println!("  Dimension: {}", embedder.model().dimension());
    println!(
        "  Cost per 1M tokens: ${}\n",
        embedder.model().cost_per_million_tokens()
    );

    // 6. Estimate cost for embedding all chunks
    println!("Estimating embedding costs...");
    let chunk_refs: Vec<&str> = all_chunks.iter().map(|s| s.as_str()).collect();
    let embedding_cost = embedder.estimate_cost(&chunk_refs);
    println!(
        "✓ Estimated cost for {} chunks: ${:.6}\n",
        all_chunks.len(),
        embedding_cost
    );

    // 7. Embed all chunks in batch
    println!("Embedding chunks (this may take a moment)...");
    let embeddings = embedder.embed_batch_async(&chunk_refs).await?;
    println!("✓ Generated {} embeddings\n", embeddings.len());

    // 8. Create VecStore and index chunks
    println!("Creating vector store and indexing chunks...");
    let mut store = VecStore::open("./openai_rag_db")?;

    for (i, (embedding, (doc_id, chunk_idx, chunk))) in
        embeddings.iter().zip(chunk_metadata.iter()).enumerate()
    {
        let mut fields = std::collections::HashMap::new();
        fields.insert("doc_id".to_string(), serde_json::json!(doc_id));
        fields.insert("chunk_idx".to_string(), serde_json::json!(chunk_idx));
        fields.insert("text".to_string(), serde_json::json!(chunk));
        let metadata = Metadata { fields };

        store.upsert(format!("chunk_{}", i), embedding.clone(), metadata)?;
    }

    println!("✓ Indexed {} chunks into vector store\n", embeddings.len());

    // 9. Perform semantic search queries
    println!("=============================================================");
    println!("Performing Semantic Searches");
    println!("=============================================================\n");

    let queries = vec![
        "How does memory management work?",
        "What web frameworks are available?",
        "Tell me about vector databases",
    ];

    for (q_idx, query) in queries.iter().enumerate() {
        println!("Query {}: \"{}\"", q_idx + 1, query);
        println!("{}", "-".repeat(60));

        // Embed the query
        let query_embedding = embedder.embed_async(query).await?;
        let query_cost = embedder.estimate_cost(&[query]);

        // Search the vector store
        let results = store.query(Query {
            vector: query_embedding.clone(),
            k: 3,
            filter: None,
        })?;

        println!("Top {} results:", results.len());
        for (i, result) in results.iter().enumerate() {
            let doc_id = result.metadata.fields.get("doc_id").unwrap();
            let chunk_idx = result.metadata.fields.get("chunk_idx").unwrap();
            let text = result
                .metadata
                .fields
                .get("text")
                .unwrap()
                .as_str()
                .unwrap();

            // Truncate text for display
            let display_text = if text.len() > 100 {
                format!("{}...", &text[..100])
            } else {
                text.to_string()
            };

            println!("\n  {}. Score: {:.4}", i + 1, result.score);
            println!("     Document: {} (chunk {})", doc_id, chunk_idx);
            println!("     Text: {}", display_text);
        }

        println!("\n  Query cost: ${:.8}", query_cost);
        println!();
    }

    // 10. Show total costs
    println!("=============================================================");
    println!("Cost Summary");
    println!("=============================================================\n");

    let total_query_cost = embedder.estimate_cost(&queries);
    let total_cost = embedding_cost + total_query_cost;

    println!("Document indexing cost: ${:.6}", embedding_cost);
    println!("Query cost:             ${:.6}", total_query_cost);
    println!("Total cost:             ${:.6}", total_cost);
    println!();

    // 11. Demonstrate filtered search
    println!("=============================================================");
    println!("Filtered Search (Rust documents only)");
    println!("=============================================================\n");

    let query = "programming language features";
    println!("Query: \"{}\"", query);

    let query_embedding = embedder.embed_async(query).await?;

    // Filter for only Rust documents
    use vecstore::parse_filter;
    let filter = parse_filter("doc_id STARTSWITH 'rust_'")?;

    let filtered_results = store.query(Query {
        vector: query_embedding.clone(),
        k: 3,
        filter: Some(filter),
    })?;

    println!("\nResults (filtered to Rust docs):");
    for (i, result) in filtered_results.iter().enumerate() {
        let doc_id = result.metadata.fields.get("doc_id").unwrap();
        let text = result
            .metadata
            .fields
            .get("text")
            .unwrap()
            .as_str()
            .unwrap();

        let display_text = if text.len() > 100 {
            format!("{}...", &text[..100])
        } else {
            text.to_string()
        };

        println!("\n  {}. Score: {:.4}", i + 1, result.score);
        println!("     Document: {}", doc_id);
        println!("     Text: {}", display_text);
    }
    println!();

    // 12. Cleanup
    std::fs::remove_dir_all("./openai_rag_db").ok();
    println!("✓ Cleaned up example database\n");

    println!("=============================================================");
    println!("✓ RAG Pipeline Example Complete!");
    println!("=============================================================");
    println!("\nKey Takeaways:");
    println!("  • Split documents into chunks for better retrieval");
    println!("  • Batch embedding is efficient and cost-effective");
    println!("  • Semantic search finds relevant content across documents");
    println!("  • Metadata enables filtering and source tracking");
    println!("  • Cost estimation helps with budget planning");
    println!();

    Ok(())
}

#[cfg(not(all(feature = "embeddings", feature = "openai-embeddings")))]
fn main() {
    eprintln!("This example requires the 'embeddings' and 'openai-embeddings' features.");
    eprintln!(
        "Run with: cargo run --example openai_rag --features \"embeddings,openai-embeddings\""
    );
    std::process::exit(1);
}
