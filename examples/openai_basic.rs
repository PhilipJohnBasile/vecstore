// OpenAI Embeddings - Basic Usage Example
//
// This example demonstrates basic usage of the OpenAI embeddings backend.
//
// Requirements:
// - Set OPENAI_API_KEY environment variable
// - Run with: cargo run --example openai_basic --features "embeddings,openai-embeddings"

#![cfg_attr(
    not(all(feature = "embeddings", feature = "openai-embeddings")),
    allow(dead_code, unused_imports)
)]

use anyhow::Result;

#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};

#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
use vecstore::{Metadata, Query, VecStore};

#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
#[tokio::main]
async fn main() -> Result<()> {
    println!("=============================================================");
    println!("OpenAI Embeddings - Basic Usage Example");
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

    // 2. Create OpenAI embedder with text-embedding-3-small model
    println!("Creating OpenAI embedder (text-embedding-3-small)...");
    let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small)
        .await?
        .with_rate_limit(500) // 500 requests per minute
        .with_max_retries(3); // Retry up to 3 times on failure

    println!("✓ Embedder created");
    println!("  Model: {}", embedder.model().as_str());
    println!("  Dimension: {}", embedder.model().dimension());
    println!(
        "  Cost: ${}/1M tokens\n",
        embedder.model().cost_per_million_tokens()
    );

    // 3. Embed a single text
    println!("Embedding single text...");
    let text = "Hello, this is a test of OpenAI embeddings!";
    println!("  Text: \"{}\"", text);

    let embedding = embedder.embed_async(text).await?;

    println!("✓ Embedding generated");
    println!("  Dimension: {}", embedding.len());
    println!("  First 5 values: {:?}\n", &embedding[0..5]);

    // 4. Embed multiple texts in batch
    println!("Embedding batch of texts...");
    let texts = vec![
        "Rust is a systems programming language",
        "Python is great for data science",
        "JavaScript runs in the browser",
    ];

    println!("  Texts:");
    for (i, t) in texts.iter().enumerate() {
        println!("    {}. \"{}\"", i + 1, t);
    }

    let embeddings = embedder.embed_batch_async(&texts).await?;

    println!("✓ Batch embeddings generated");
    println!("  Count: {}", embeddings.len());
    println!("  Each dimension: {}\n", embeddings[0].len());

    // 5. Compute similarity between embeddings
    println!("Computing cosine similarity...");
    let sim_1_2 = cosine_similarity(&embeddings[0], &embeddings[1]);
    let sim_1_3 = cosine_similarity(&embeddings[0], &embeddings[2]);
    let sim_2_3 = cosine_similarity(&embeddings[1], &embeddings[2]);

    println!("  Rust ↔ Python:     {:.4}", sim_1_2);
    println!("  Rust ↔ JavaScript: {:.4}", sim_1_3);
    println!("  Python ↔ JavaScript: {:.4}\n", sim_2_3);

    // 6. Estimate cost for embedding
    println!("Estimating costs...");
    let cost_single = embedder.estimate_cost(&[text]);
    let cost_batch = embedder.estimate_cost(&texts);

    println!("  Single text cost: ${:.8}", cost_single);
    println!("  Batch cost: ${:.8}", cost_batch);
    println!("  Total cost: ${:.8}\n", cost_single + cost_batch);

    // 7. Use with VecStore
    println!("Integrating with VecStore...");
    let mut store = VecStore::open("./openai_example_db")?;

    // Embed and store documents
    for (i, text) in texts.iter().enumerate() {
        let emb = embedder.embed_async(text).await?;
        let mut fields = std::collections::HashMap::new();
        fields.insert("text".to_string(), serde_json::json!(text));
        let metadata = Metadata { fields };
        store.upsert(format!("doc{}", i), emb, metadata)?;
    }

    println!("✓ Stored {} documents", texts.len());

    // Query with a new text
    let query_text = "Tell me about programming languages";
    println!("\nQuerying with: \"{}\"", query_text);

    let query_emb = embedder.embed_async(query_text).await?;
    let results = store.query(Query {
        vector: query_emb,
        k: 3,
        filter: None,
    })?;

    println!("✓ Top results:");
    for (i, result) in results.iter().enumerate() {
        let text = result.metadata.fields.get("text").unwrap();
        println!("  {}. Score: {:.4} - {}", i + 1, result.score, text);
    }

    // Cleanup
    std::fs::remove_dir_all("./openai_example_db").ok();
    println!("\n✓ Cleaned up example database");

    println!("\n=============================================================");
    println!("✓ Example complete!");
    println!("=============================================================");

    Ok(())
}

// Helper function to compute cosine similarity
#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (mag_a * mag_b)
}

#[cfg(not(all(feature = "embeddings", feature = "openai-embeddings")))]
fn main() {
    eprintln!("This example requires the 'embeddings' and 'openai-embeddings' features.");
    eprintln!(
        "Run with: cargo run --example openai_basic --features \"embeddings,openai-embeddings\""
    );
    std::process::exit(1);
}
