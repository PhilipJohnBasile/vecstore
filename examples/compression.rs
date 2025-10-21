//! Index Compression Example
//!
//! This example demonstrates various compression techniques for reducing
//! index memory usage and storage size.
//!
//! ## Compression Techniques
//!
//! - Delta encoding: For sequential IDs
//! - Varint encoding: Variable-length integers
//! - Float quantization: Reduced precision for distances
//! - Run-length encoding: For repeated values
//!
//! ## Running
//!
//! ```bash
//! cargo run --example compression
//! ```

use vecstore::compression::{
    decode_rle, decode_varint, encode_rle, encode_varint, CompressedNeighborList,
    CompressionConfig, CompressionLevel, CompressionStats,
};

fn main() {
    println!("🗜️  Index Compression Example\n");

    // ============================================================
    // 1. Varint Encoding
    // ============================================================
    println!("📊 Variable-Length Integer Encoding (Varint):");
    println!("═══════════════════════════════════════════════\n");

    let numbers = vec![0, 1, 127, 128, 255, 256, 16383, 16384, 1_000_000];

    println!("Encoding examples:");
    for num in &numbers {
        let mut encoded = Vec::new();
        encode_varint(&mut encoded, *num).unwrap();
        println!(
            "  {} → {} bytes (vs 8 bytes uncompressed)",
            num,
            encoded.len()
        );
    }

    println!("\nVarint is most efficient for small integers!");

    // ============================================================
    // 2. Delta Encoding for Sequential IDs
    // ============================================================
    println!("\n\n🔢 Delta Encoding for Sequential IDs:");
    println!("═══════════════════════════════════════\n");

    let config = CompressionConfig::default();

    // Highly sequential IDs (best case)
    let sequential_ids = vec![100, 101, 102, 103, 104, 105, 106, 107, 108, 109];
    let compressed_seq = config.compress_ids(&sequential_ids).unwrap();

    println!("Sequential IDs (100-109):");
    println!("  Original size: {} bytes", sequential_ids.len() * 8);
    println!("  Compressed size: {} bytes", compressed_seq.len());
    println!(
        "  Compression ratio: {:.2}x",
        (sequential_ids.len() * 8) as f32 / compressed_seq.len() as f32
    );
    println!(
        "  Space savings: {:.1}%",
        (1.0 - compressed_seq.len() as f32 / (sequential_ids.len() * 8) as f32) * 100.0
    );

    // Verify decompression
    let decompressed_seq = config
        .decompress_ids(&compressed_seq, sequential_ids.len())
        .unwrap();
    assert_eq!(sequential_ids, decompressed_seq);
    println!("  ✓ Decompression verified");

    // Moderately sequential IDs
    println!("\nModerately sequential IDs (some gaps):");
    let moderate_ids = vec![10, 11, 15, 16, 20, 25, 30, 31, 32, 40];
    let compressed_mod = config.compress_ids(&moderate_ids).unwrap();

    println!("  Original size: {} bytes", moderate_ids.len() * 8);
    println!("  Compressed size: {} bytes", compressed_mod.len());
    println!(
        "  Compression ratio: {:.2}x",
        (moderate_ids.len() * 8) as f32 / compressed_mod.len() as f32
    );

    // Sparse IDs (worst case for delta encoding)
    println!("\nSparse IDs (large gaps):");
    let sparse_ids = vec![10, 500, 1000, 5000, 10000, 50000, 100000];
    let compressed_sparse = config.compress_ids(&sparse_ids).unwrap();

    println!("  Original size: {} bytes", sparse_ids.len() * 8);
    println!("  Compressed size: {} bytes", compressed_sparse.len());
    println!(
        "  Compression ratio: {:.2}x",
        (sparse_ids.len() * 8) as f32 / compressed_sparse.len() as f32
    );

    // ============================================================
    // 3. HNSW Neighbor List Compression
    // ============================================================
    println!("\n\n🕸️  HNSW Neighbor List Compression:");
    println!("═══════════════════════════════════\n");

    // Simulate HNSW neighbor lists at different layers
    let layer0_neighbors = vec![10, 11, 12, 13, 15, 18, 20, 21, 22, 25, 30, 32, 35, 40, 42];
    let layer1_neighbors = vec![5, 25, 50, 75];
    let layer2_neighbors = vec![100];

    println!(
        "Layer 0 (highly connected, {} neighbors):",
        layer0_neighbors.len()
    );
    let comp_l0 = CompressedNeighborList::compress(&layer0_neighbors, &config).unwrap();
    println!("  Original: {} bytes", layer0_neighbors.len() * 8);
    println!("  Compressed: {} bytes", comp_l0.data.len());
    println!("  Ratio: {:.2}x", comp_l0.compression_ratio());

    println!(
        "\nLayer 1 (moderate connectivity, {} neighbors):",
        layer1_neighbors.len()
    );
    let comp_l1 = CompressedNeighborList::compress(&layer1_neighbors, &config).unwrap();
    println!("  Original: {} bytes", layer1_neighbors.len() * 8);
    println!("  Compressed: {} bytes", comp_l1.data.len());
    println!("  Ratio: {:.2}x", comp_l1.compression_ratio());

    println!("\nLayer 2 (sparse, {} neighbor):", layer2_neighbors.len());
    let comp_l2 = CompressedNeighborList::compress(&layer2_neighbors, &config).unwrap();
    println!("  Original: {} bytes", layer2_neighbors.len() * 8);
    println!("  Compressed: {} bytes", comp_l2.data.len());
    println!("  Ratio: {:.2}x", comp_l2.compression_ratio());

    // ============================================================
    // 4. Float Compression (Quantization)
    // ============================================================
    println!("\n\n📉 Float Compression (Quantization):");
    println!("═══════════════════════════════════\n");

    let distances = vec![0.12, 0.25, 0.33, 0.45, 0.58, 0.67, 0.79, 0.88, 0.92];

    println!("Original distances (32-bit floats):");
    println!("  {:?}", &distances[..5]);
    println!("  Size: {} bytes\n", distances.len() * 4);

    // 8-bit quantization
    println!("8-bit quantization:");
    let compressed_8bit = config.compress_floats(&distances, 8).unwrap();
    println!("  Compressed size: {} bytes", compressed_8bit.len());
    println!(
        "  Compression ratio: {:.2}x",
        (distances.len() * 4) as f32 / compressed_8bit.len() as f32
    );

    let decompressed_8bit = config
        .decompress_floats(&compressed_8bit, distances.len(), 8)
        .unwrap();
    println!("  Decompressed: {:?}", &decompressed_8bit[..5]);

    // Calculate error
    let max_error_8bit = distances
        .iter()
        .zip(decompressed_8bit.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("  Max error: {:.4}", max_error_8bit);

    // 16-bit quantization
    println!("\n16-bit quantization:");
    let compressed_16bit = config.compress_floats(&distances, 16).unwrap();
    println!("  Compressed size: {} bytes", compressed_16bit.len());
    println!(
        "  Compression ratio: {:.2}x",
        (distances.len() * 4) as f32 / compressed_16bit.len() as f32
    );

    let decompressed_16bit = config
        .decompress_floats(&compressed_16bit, distances.len(), 16)
        .unwrap();
    println!("  Decompressed: {:?}", &decompressed_16bit[..5]);

    let max_error_16bit = distances
        .iter()
        .zip(decompressed_16bit.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("  Max error: {:.6}", max_error_16bit);

    println!(
        "\n💡 Trade-off: 8-bit is smaller but less precise, 16-bit is larger but more accurate"
    );

    // ============================================================
    // 5. Run-Length Encoding
    // ============================================================
    println!("\n\n🏃 Run-Length Encoding:");
    println!("═══════════════════════\n");

    let repeated = vec![5, 5, 5, 5, 7, 7, 7, 10, 10, 10, 10, 10, 15, 20, 20];

    println!("Original values: {:?}", repeated);
    println!("Original size: {} bytes", repeated.len() * 8);

    let rle_encoded = encode_rle(&repeated);
    println!("\nRLE encoded size: {} bytes", rle_encoded.len());
    println!(
        "Compression ratio: {:.2}x",
        (repeated.len() * 8) as f32 / rle_encoded.len() as f32
    );

    let rle_decoded = decode_rle(&rle_encoded).unwrap();
    assert_eq!(repeated, rle_decoded);
    println!("✓ Decompression verified");

    println!("\n💡 RLE works best for data with many consecutive repeated values");

    // ============================================================
    // 6. Compression Levels
    // ============================================================
    println!("\n\n⚙️  Compression Levels:");
    println!("═══════════════════════\n");

    let large_id_list: Vec<usize> = (1000..1500).collect();

    let levels = vec![
        ("None", CompressionLevel::None),
        ("Fast", CompressionLevel::Fast),
        ("Balanced", CompressionLevel::Balanced),
        ("Max", CompressionLevel::Max),
    ];

    for (name, level) in levels {
        let config = CompressionConfig::default().with_level(level);
        let compressed = config.compress_ids(&large_id_list).unwrap();

        println!("{} compression:", name);
        println!("  Compressed size: {} bytes", compressed.len());
        println!(
            "  Ratio: {:.2}x",
            (large_id_list.len() * 8) as f32 / compressed.len() as f32
        );
    }

    // ============================================================
    // 7. Configuration Options
    // ============================================================
    println!("\n\n🔧 Configuration Options:");
    println!("═════════════════════════\n");

    let test_ids = vec![100, 101, 102, 105, 110, 111, 112];

    println!("Test data: {:?}", test_ids);
    println!("Original size: {} bytes\n", test_ids.len() * 8);

    // Delta + Varint (default)
    let config1 = CompressionConfig::default();
    let comp1 = config1.compress_ids(&test_ids).unwrap();
    println!("Delta + Varint: {} bytes", comp1.len());

    // Varint only
    let config2 = CompressionConfig::default()
        .with_delta_encoding(false)
        .with_varint_encoding(true);
    let comp2 = config2.compress_ids(&test_ids).unwrap();
    println!("Varint only: {} bytes", comp2.len());

    // No compression
    let config3 = CompressionConfig::default()
        .with_delta_encoding(false)
        .with_varint_encoding(false);
    let comp3 = config3.compress_ids(&test_ids).unwrap();
    println!("No compression: {} bytes", comp3.len());

    // ============================================================
    // 8. Compression Statistics
    // ============================================================
    println!("\n\n📈 Compression Statistics:");
    println!("═══════════════════════════\n");

    let mut stats = CompressionStats::default();
    stats.original_bytes = 10000;
    stats.compressed_bytes = 2500;
    stats.num_lists = 100;
    stats.avg_list_length = 12.5;

    println!("Simulated HNSW index compression:");
    println!("  Number of neighbor lists: {}", stats.num_lists);
    println!("  Average list length: {:.1}", stats.avg_list_length);
    println!(
        "  Original size: {} bytes ({:.1} KB)",
        stats.original_bytes,
        stats.original_bytes as f32 / 1024.0
    );
    println!(
        "  Compressed size: {} bytes ({:.1} KB)",
        stats.compressed_bytes,
        stats.compressed_bytes as f32 / 1024.0
    );
    println!("  Compression ratio: {:.2}x", stats.ratio());
    println!("  Space savings: {:.1}%", stats.savings_percent());

    // ============================================================
    // 9. Real-world HNSW Index Example
    // ============================================================
    println!("\n\n🌍 Real-world HNSW Index Example:");
    println!("═══════════════════════════════════\n");

    println!("Scenario: 100,000 vectors with M=16");
    println!("  Average neighbors per node: ~16");
    println!("  Total neighbor list entries: ~1,600,000");
    println!("  Uncompressed size: ~12.2 MB (1.6M * 8 bytes)\n");

    // Simulate compression savings
    let typical_ratio = 4.5; // Typical for delta+varint on HNSW
    let compressed_size_mb = 12.2 / typical_ratio;

    println!("With delta + varint compression:");
    println!("  Compressed size: ~{:.1} MB", compressed_size_mb);
    println!("  Compression ratio: {:.1}x", typical_ratio);
    println!(
        "  Space savings: {:.1}%",
        (1.0 - 1.0 / typical_ratio) * 100.0
    );
    println!("  Memory saved: ~{:.1} MB", 12.2 - compressed_size_mb);

    // ============================================================
    // 10. Best Practices
    // ============================================================
    println!("\n\n💡 Best Practices:");
    println!("══════════════════\n");

    println!("1. Use delta + varint for HNSW neighbor lists");
    println!("   → Typically 3-5x compression for sequential IDs\n");

    println!("2. Use 8-bit quantization for approximate distances");
    println!("   → 4x compression with minimal impact on recall\n");

    println!("3. Use 16-bit quantization for precise distances");
    println!("   → 2x compression with negligible precision loss\n");

    println!("4. Use RLE for sparse data with repeated values");
    println!("   → Effective for metadata or layer information\n");

    println!("5. Consider bulk compression for large indices");
    println!("   → Additional 20-40% savings with ZSTD on top of delta+varint\n");

    println!("6. Benchmark on your specific data");
    println!("   → Compression effectiveness varies by use case");

    println!("\n✅ Compression example complete!\n");

    println!("🎯 Key Takeaways:");
    println!("  - Delta encoding is essential for HNSW neighbor lists");
    println!("  - Varint encoding reduces storage for small integers");
    println!("  - Float quantization trades precision for space");
    println!("  - Typical HNSW compression: 3-5x for graph structure");
    println!("  - Combined with PQ: 30-50x total memory reduction");
}
