//! Dimensionality reduction demonstration
//!
//! Shows how to reduce vector dimensions using PCA:
//! - Reduce high-dimensional data to 2D/3D for visualization
//! - Compress vectors to save storage
//! - Analyze explained variance
//! - Reconstruct original vectors

use anyhow::Result;
use vecstore::dim_reduction::{PCA, ReductionStats};

fn generate_high_dim_data() -> Vec<Vec<f32>> {
    let mut vectors = Vec::new();

    println!("   Generating synthetic high-dimensional data:");
    println!("   • Original dimensions: 128");
    println!("   • Vectors: 100");
    println!("   • Structure: 3 underlying factors with noise");

    // Generate data with 3 underlying factors but in 128 dimensions
    for i in 0..100 {
        let t1 = (i as f32 / 10.0).sin();
        let t2 = (i as f32 / 15.0).cos();
        let t3 = (i as f32 / 8.0).sin() * (i as f32 / 12.0).cos();

        let mut vector = Vec::with_capacity(128);
        for j in 0..128 {
            // Each dimension is a linear combination of 3 factors + noise
            let w1 = (j as f32 * 0.1).sin();
            let w2 = (j as f32 * 0.15).cos();
            let w3 = (j as f32 * 0.08).sin();
            let noise = ((i * 7 + j * 11) as f32).sin() * 0.1;

            let value = t1 * w1 + t2 * w2 + t3 * w3 + noise;
            vector.push(value);
        }
        vectors.push(vector);
    }

    vectors
}

fn main() -> Result<()> {
    println!("📐 VecStore Dimensionality Reduction Demo\n");
    println!("{}", "=".repeat(80));

    // Generate high-dimensional data
    println!("\n[1/5] Generating high-dimensional data...");
    let vectors = generate_high_dim_data();
    let original_size = vectors.len() * vectors[0].len() * 4; // 4 bytes per f32
    println!(
        "   ✓ Original storage: {} bytes ({:.1} KB)",
        original_size,
        original_size as f32 / 1024.0
    );

    // Reduce to 2D for visualization
    println!("\n[2/5] Reducing to 2D for visualization...");
    let mut pca_2d = PCA::new(2);
    let reduced_2d = pca_2d.fit_transform(&vectors)?;

    println!("   Reduced to 2 dimensions:");
    println!("   • First 5 vectors in 2D:");
    for (i, vec) in reduced_2d.iter().take(5).enumerate() {
        println!("     Vector {}: [{:.3}, {:.3}]", i, vec[0], vec[1]);
    }

    let var_ratio_2d = pca_2d.explained_variance_ratio().unwrap();
    println!(
        "   • Explained variance: {:.2}%",
        (var_ratio_2d[0] + var_ratio_2d[1]) * 100.0
    );

    // Reduce to 16D for compression
    println!("\n[3/5] Reducing to 16D for compression...");
    let mut pca_16d = PCA::new(16);
    let reduced_16d = pca_16d.fit_transform(&vectors)?;

    let compressed_size = vectors.len() * reduced_16d[0].len() * 4;
    let compression_ratio = original_size as f32 / compressed_size as f32;

    println!("   Compressed from 128D to 16D:");
    println!(
        "   • Compressed storage: {} bytes ({:.1} KB)",
        compressed_size,
        compressed_size as f32 / 1024.0
    );
    println!("   • Compression ratio: {:.1}x", compression_ratio);
    println!(
        "   • Space saved: {:.1}%",
        (1.0 - 1.0 / compression_ratio) * 100.0
    );

    let var_ratio_16d = pca_16d.explained_variance_ratio().unwrap();
    let total_var_16d: f32 = var_ratio_16d.iter().sum();
    println!("   • Information retained: {:.2}%", total_var_16d * 100.0);

    // Reconstruction error
    println!("\n[4/5] Testing reconstruction quality...");
    let reconstructed = pca_16d.inverse_transform(&reduced_16d)?;

    let mse: f32 = vectors
        .iter()
        .zip(&reconstructed)
        .map(|(orig, recon)| {
            orig.iter()
                .zip(recon)
                .map(|(&o, &r)| (o - r).powi(2))
                .sum::<f32>()
                / orig.len() as f32
        })
        .sum::<f32>()
        / vectors.len() as f32;

    let rmse = mse.sqrt();
    println!("   Reconstruction quality:");
    println!("   • Root Mean Squared Error (RMSE): {:.6}", rmse);
    println!(
        "   • Quality: {}",
        if rmse < 0.1 {
            "Excellent"
        } else if rmse < 0.5 {
            "Good"
        } else {
            "Fair"
        }
    );

    // Analyze variance explained
    println!("\n[5/5] Analyzing explained variance...");
    let stats = ReductionStats::from_pca(&pca_16d, 128);

    println!("   Reduction Statistics:");
    println!("   • Original dimensions: {}", stats.original_dims);
    println!("   • Reduced dimensions: {}", stats.reduced_dims);
    println!("   • Compression ratio: {:.1}x", stats.compression_ratio);

    println!("\n   Variance explained by component:");
    for (i, &var) in stats.explained_variance.iter().take(10).enumerate() {
        let cumul = stats.cumulative_variance[i];
        println!(
            "   • PC{}: {:.2}% (cumulative: {:.2}%)",
            i + 1,
            var * 100.0,
            cumul * 100.0
        );
    }

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("📊 Summary");
    println!("{}", "=".repeat(80));

    println!("\n✅ Dimensionality reduction working!");

    println!("\n💡 Use Cases:");
    println!("   • Visualization: Reduce to 2D/3D for plotting");
    println!("   • Compression: Save storage (128D → 16D = 8x smaller)");
    println!("   • Speed: Faster search with fewer dimensions");
    println!("   • Noise reduction: Remove low-variance components");
    println!("   • Feature engineering: Extract principal features");

    println!("\n🔍 PCA Characteristics:");
    println!("   • Method: Linear dimensionality reduction");
    println!("   • Preserves: Maximum variance directions");
    println!("   • Complexity: O(n*d² + d³) where n=vectors, d=dims");
    println!("   • Best for: Linearly correlated data");
    println!("   • Limitations: Linear only, assumes Gaussian");

    println!("\n⚙️  Configuration Guidelines:");
    println!("   Visualization:");
    println!("   • 2D: For simple plots, heatmaps");
    println!("   • 3D: For interactive 3D visualizations");
    println!("\n   Compression:");
    println!("   • Aim for 80-95% variance retained");
    println!("   • Balance compression vs. accuracy");
    println!("   • Test reconstruction quality");
    println!("\n   Search Acceleration:");
    println!("   • Reduce to 32-64 dims for large datasets");
    println!("   • Minimal quality loss if >95% variance retained");

    println!("\n📈 Component Selection:");
    println!("   • Elbow method: Plot variance vs. components");
    println!("   • Threshold: Keep components with >1% variance");
    println!("   • Target: Cumulative 90-95% variance");
    println!("   • Validate: Test on held-out data");

    println!("\n🎯 Best Practices:");
    println!("   1. Scale/normalize data before PCA");
    println!("   2. Remove outliers for stable components");
    println!("   3. Fit on training data, transform test data");
    println!("   4. Monitor reconstruction error");
    println!("   5. Visualize first 2-3 components");
    println!("   6. Consider non-linear methods (t-SNE, UMAP) for viz");

    println!("\n🔧 Example Usage:");
    println!("   # Reduce for visualization");
    println!("   let mut pca = PCA::new(2);");
    println!("   let reduced = pca.fit_transform(&vectors)?;");
    println!();
    println!("   # Compress for storage");
    println!("   let mut pca = PCA::new(32);");
    println!("   let compressed = pca.fit_transform(&vectors)?;");
    println!("   // Store compressed vectors");
    println!();
    println!("   # Reconstruct later");
    println!("   let reconstructed = pca.inverse_transform(&compressed)?;");

    Ok(())
}
