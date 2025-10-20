// Example demonstrating automatic model downloading and embedding
//
// This shows the "batteries included" approach where users don't need to
// manage embedding models manually.

#[cfg(feature = "embeddings")]
fn main() -> anyhow::Result<()> {
    use vecstore::embeddings::AutoEmbedder;

    println!("=== Auto-Downloading Embeddings Demo ===\n");

    println!("Creating embedder (will download model on first use)...");
    println!("Model: all-MiniLM-L6-v2 (384 dimensions, ~80MB)\n");

    // This will automatically download the model to ~/.vecstore/models/
    let embedder = AutoEmbedder::from_pretrained("all-MiniLM-L6-v2")?;

    println!("\n✓ Model loaded successfully!");
    println!("Cache directory: {}\n", embedder.cache_dir().display());

    // Encode some text
    println!("Encoding texts...");
    let texts = vec![
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming technology",
        "Rust is a systems programming language",
    ];

    for (i, text) in texts.iter().enumerate() {
        let embedding = embedder.encode(text)?;
        println!("  Text {}: \"{}\"", i + 1, &text[..text.len().min(40)]);
        println!(
            "    → Embedding: [{:.4}, {:.4}, ..., {:.4}] (dim={})",
            embedding[0],
            embedding[1],
            embedding[embedding.len() - 1],
            embedding.len()
        );
    }

    // Batch encoding (more efficient)
    println!("\nBatch encoding (faster for multiple texts)...");
    let batch_texts: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let start = std::time::Instant::now();
    let embeddings = embedder.encode_batch(&batch_texts)?;
    let elapsed = start.elapsed();

    println!("  Encoded {} texts in {:?}", embeddings.len(), elapsed);
    println!(
        "  Avg time per text: {:?}",
        elapsed / embeddings.len() as u32
    );

    println!("\n=== Demo Complete ===");
    println!("\nKey Features:");
    println!("  ✓ Automatic model download");
    println!("  ✓ Local caching (~/.vecstore/models/)");
    println!("  ✓ No manual model management");
    println!("  ✓ Pre-configured models available");
    println!("  ✓ Fast batch encoding");

    Ok(())
}

#[cfg(not(feature = "embeddings"))]
fn main() {
    eprintln!("This example requires the 'embeddings' feature.");
    eprintln!("Run with: cargo run --example auto_embeddings_demo --features embeddings");
    std::process::exit(1);
}
