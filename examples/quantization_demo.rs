// Product Quantization Demo - Memory Compression for Large Scale
//
// This example demonstrates how Product Quantization (PQ) can reduce memory
// usage by 8-32x with minimal accuracy loss. Critical for scaling to millions
// or billions of vectors!

use std::time::Instant;
use vecstore::{PQConfig, PQVectorStore, ProductQuantizer};

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                                                               ║");
    println!("║       Product Quantization - Memory Compression Demo         ║");
    println!("║                                                               ║");
    println!("║         Reduce Memory by 8-32x with Minimal Accuracy Loss    ║");
    println!("║                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let dimension = 128;
    let num_vectors = 10_000;
    let num_queries = 100;

    println!("📊 Configuration:");
    println!("   Vector dimension: {}", dimension);
    println!("   Number of vectors: {}", num_vectors);
    println!("   Number of queries: {}", num_queries);
    println!();

    // Demo 1: Understanding compression ratio
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 1: Compression Ratio Analysis");
    println!("═══════════════════════════════════════════════════════════════\n");

    let configs = vec![
        (
            "8 subvectors, 256 centroids",
            PQConfig {
                num_subvectors: 8,
                num_centroids: 256,
                training_iterations: 20,
            },
        ),
        (
            "16 subvectors, 256 centroids",
            PQConfig {
                num_subvectors: 16,
                num_centroids: 256,
                training_iterations: 20,
            },
        ),
        (
            "32 subvectors, 256 centroids",
            PQConfig {
                num_subvectors: 32,
                num_centroids: 256,
                training_iterations: 20,
            },
        ),
    ];

    for (name, config) in &configs {
        let pq = ProductQuantizer::new(dimension, config.clone())?;
        let original_size = dimension * 4; // 4 bytes per float
        let compressed_size = config.num_subvectors; // 1 byte per code

        println!("Configuration: {}", name);
        println!("   Original size: {} bytes", original_size);
        println!("   Compressed size: {} bytes", compressed_size);
        println!("   Compression ratio: {:.1}x", pq.compression_ratio());
        println!();
    }

    // Demo 2: Training and encoding
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 2: Training Product Quantizer");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Generate random training data
    println!("🔧 Generating training data...");
    let training_vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dimension).map(|_| rand::random::<f32>()).collect())
        .collect();

    let config = PQConfig {
        num_subvectors: 16,
        num_centroids: 256,
        training_iterations: 20,
    };

    let mut pq = ProductQuantizer::new(dimension, config.clone())?;

    println!("📚 Training quantizer...");
    let start = Instant::now();
    pq.train(&training_vectors)?;
    let training_time = start.elapsed();

    println!("✅ Training complete!");
    println!("   Time: {:.2}s", training_time.as_secs_f32());
    println!("   Compression ratio: {:.1}x", pq.compression_ratio());
    println!();

    // Demo 3: Encoding vectors
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 3: Encoding Vectors");
    println!("═══════════════════════════════════════════════════════════════\n");

    let test_vector = &training_vectors[0];

    let start = Instant::now();
    let codes = pq.encode(test_vector)?;
    let encode_time = start.elapsed();

    println!(
        "Original vector (first 10 values): {:?}...",
        &test_vector[..10]
    );
    println!(
        "Encoded codes: {:?}...",
        &codes[..std::cmp::min(10, codes.len())]
    );
    println!("Encoding time: {:.2}μs", encode_time.as_micros());
    println!();

    // Decode
    let decoded = pq.decode(&codes)?;
    println!("Decoded vector (first 10 values): {:?}...", &decoded[..10]);

    // Compute reconstruction error
    let error: f32 = test_vector
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    println!("Reconstruction error (L2 norm): {:.4}", error);
    println!();

    // Demo 4: Memory savings
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 4: Memory Savings Analysis");
    println!("═══════════════════════════════════════════════════════════════\n");

    let original_memory = num_vectors * dimension * 4; // bytes
    let compressed_memory = num_vectors * config.num_subvectors; // bytes

    println!(
        "Original vectors: {} vectors × {} dims × 4 bytes",
        num_vectors, dimension
    );
    println!("   Memory: {:.2} MB", original_memory as f32 / 1_000_000.0);
    println!();
    println!(
        "Compressed (PQ): {} vectors × {} codes × 1 byte",
        num_vectors, config.num_subvectors
    );
    println!(
        "   Memory: {:.2} MB",
        compressed_memory as f32 / 1_000_000.0
    );
    println!();
    println!(
        "💾 Memory saved: {:.2} MB ({:.1}x reduction)",
        (original_memory - compressed_memory) as f32 / 1_000_000.0,
        original_memory as f32 / compressed_memory as f32
    );
    println!();

    // Demo 5: PQ Vector Store
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 5: PQ Vector Store - Complete Example");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut pq_store = PQVectorStore::new(dimension, config)?;

    println!("Training store on {} vectors...", num_vectors);
    let start = Instant::now();
    pq_store.train(&training_vectors)?;
    println!(
        "✅ Training complete: {:.2}s",
        start.elapsed().as_secs_f32()
    );
    println!();

    println!("Adding vectors to store...");
    let start = Instant::now();
    for (i, vec) in training_vectors.iter().enumerate() {
        pq_store.add(format!("vec_{}", i), vec)?;
    }
    let add_time = start.elapsed();
    println!(
        "✅ Added {} vectors in {:.2}s ({:.1} vectors/sec)",
        num_vectors,
        add_time.as_secs_f32(),
        num_vectors as f32 / add_time.as_secs_f32()
    );
    println!();

    // Search performance
    println!("🔍 Search Performance:");
    let mut total_search_time = std::time::Duration::default();

    for i in 0..num_queries {
        let query = &training_vectors[i];
        let start = Instant::now();
        let results = pq_store.search(query, 10)?;
        total_search_time += start.elapsed();

        if i == 0 {
            println!("\nFirst query results:");
            for (rank, (id, dist)) in results.iter().enumerate().take(5) {
                println!("   {}. {} (distance: {:.4})", rank + 1, id, dist);
            }
        }
    }

    let avg_search_time = total_search_time / num_queries as u32;
    println!(
        "\n✅ Average search time: {:.2}ms ({} queries)",
        avg_search_time.as_micros() as f32 / 1000.0,
        num_queries
    );
    println!(
        "   Queries per second: {:.0}",
        1_000_000_000.0 / avg_search_time.as_nanos() as f32
    );
    println!();

    // Store statistics
    println!("📊 Store Statistics:");
    println!("   Vectors: {}", pq_store.len());
    println!(
        "   Memory usage: {:.2} MB",
        pq_store.memory_usage() as f32 / 1_000_000.0
    );
    println!("   Compression ratio: {:.1}x", pq_store.compression_ratio());
    println!();

    // Demo 6: Accuracy comparison
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 6: Accuracy vs Compression Trade-off");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Testing different PQ configurations...\n");

    let test_configs = vec![
        (
            "High accuracy",
            PQConfig {
                num_subvectors: 32,
                num_centroids: 256,
                training_iterations: 20,
            },
        ),
        (
            "Balanced",
            PQConfig {
                num_subvectors: 16,
                num_centroids: 256,
                training_iterations: 20,
            },
        ),
        (
            "High compression",
            PQConfig {
                num_subvectors: 8,
                num_centroids: 256,
                training_iterations: 20,
            },
        ),
    ];

    for (name, cfg) in test_configs {
        let mut pq = ProductQuantizer::new(dimension, cfg)?;
        pq.train(&training_vectors)?;

        // Measure reconstruction error
        let mut total_error = 0.0;
        for vec in training_vectors.iter().take(100) {
            let codes = pq.encode(vec)?;
            let decoded = pq.decode(&codes)?;

            let error: f32 = vec
                .iter()
                .zip(decoded.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();

            total_error += error;
        }

        let avg_error = total_error / 100.0;

        println!(
            "{:20} - Compression: {:>5.1}x   Avg Error: {:.4}",
            name,
            pq.compression_ratio(),
            avg_error
        );
    }

    println!();

    // Summary
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                  Summary & Best Practices                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("✨ What is Product Quantization?");
    println!("   • Lossy compression technique for vectors");
    println!("   • Reduces memory by 8-32x with minimal accuracy loss");
    println!("   • Used in production systems (FAISS, Milvus, etc.)");
    println!();
    println!("🎯 When to use PQ:");
    println!("   ✅ Large-scale deployments (millions of vectors)");
    println!("   ✅ Memory-constrained environments");
    println!("   ✅ When some accuracy loss is acceptable");
    println!("   ✅ Read-heavy workloads (search >> updates)");
    println!();
    println!("📊 Configuration Guidelines:");
    println!("   • num_subvectors: 8-32 (more = better accuracy, less compression)");
    println!("   • num_centroids: 256 (1 byte/code) or 65536 (2 bytes/code)");
    println!("   • training_iterations: 20-50 for convergence");
    println!("   • Training data: Use representative sample (10K-100K vectors)");
    println!();
    println!("⚡ Performance Characteristics:");
    println!("   • Training: One-time cost, can be slow");
    println!("   • Encoding: Very fast (~microseconds per vector)");
    println!("   • Search: Slower than full-precision but much less memory");
    println!("   • Memory: 8-32x reduction");
    println!();
    println!("💡 Pro Tips:");
    println!("   • Train on representative sample of your data");
    println!("   • Balance accuracy vs compression for your use case");
    println!("   • For huge datasets, combine PQ with HNSW for best results");
    println!("   • Monitor reconstruction error during development");
    println!();

    println!("╭───────────────────────────────────────────────────────────╮");
    println!("│  Product Quantization: Scale to Billions of Vectors! 🚀  │");
    println!("╰───────────────────────────────────────────────────────────╯");

    Ok(())
}
