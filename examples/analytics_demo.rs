//! Vector Analytics and Statistics Demo
//!
//! Demonstrates comprehensive analytics and insights about vector data.

use vecstore::*;

fn main() -> anyhow::Result<()> {
    println!("\n📊 Vector Analytics and Statistics Demo\n");
    println!("{}", "=".repeat(70));

    // Create diverse test datasets
    println!("\n[1/4] Generating Test Datasets");
    println!("{}", "-".repeat(70));

    // Dataset 1: Well-clustered data
    let mut clustered_vectors = Vec::new();
    println!("\nDataset 1: Well-Clustered Data");
    for i in 0..30 {
        let cluster = i / 10;
        let base = cluster as f32 * 5.0;
        clustered_vectors.push(vec![
            base + (i % 10) as f32 * 0.1,
            base + (i % 10) as f32 * 0.1,
            base + (i % 10) as f32 * 0.1,
        ]);
    }
    println!(
        "  Created {} vectors in 3 clusters",
        clustered_vectors.len()
    );

    // Dataset 2: Random uniform data
    let mut random_vectors = Vec::new();
    println!("\nDataset 2: Random Uniform Data");
    for i in 0..30 {
        random_vectors.push(vec![
            (i as f32 * 0.1) % 1.0,
            (i as f32 * 0.2) % 1.0,
            (i as f32 * 0.3) % 1.0,
        ]);
    }
    println!("  Created {} random vectors", random_vectors.len());

    // Dataset 3: Data with outliers
    let mut outlier_vectors = Vec::new();
    println!("\nDataset 3: Data with Outliers");
    for i in 0..27 {
        outlier_vectors.push(vec![1.0, 1.0, 1.0]);
    }
    // Add outliers
    outlier_vectors.push(vec![10.0, 10.0, 10.0]);
    outlier_vectors.push(vec![-10.0, -10.0, -10.0]);
    outlier_vectors.push(vec![0.0, 0.0, 0.0]);
    println!(
        "  Created {} vectors with 3 outliers",
        outlier_vectors.len()
    );

    // Test 1: Analyze clustered data
    println!("\n[2/4] Analyzing Well-Clustered Data");
    println!("{}", "-".repeat(70));

    let analytics = VectorAnalytics::default();
    let report1 = analytics.analyze(&clustered_vectors)?;

    println!("\n📈 Distribution:");
    println!(
        "  Mean magnitude:   {:.4}",
        report1.distribution.mean_magnitude
    );
    println!("  Std deviation:    {:.4}", report1.distribution.std_dev);
    println!("  Skewness:         {:.4}", report1.distribution.skewness);
    println!("  Kurtosis:         {:.4}", report1.distribution.kurtosis);

    println!("\n🔗 Similarity:");
    println!("  Pairs analyzed:   {}", report1.similarity.pairs_analyzed);
    println!(
        "  Mean similarity:  {:.4}",
        report1.similarity.mean_similarity
    );
    println!(
        "  Min/Max:          [{:.4}, {:.4}]",
        report1.similarity.min_similarity, report1.similarity.max_similarity
    );

    println!("\n🎯 Clustering:");
    println!(
        "  Avg NN distance:  {:.4}",
        report1.cluster_tendency.avg_nn_distance
    );
    println!(
        "  Tendency score:   {:.4}",
        report1.cluster_tendency.tendency_score
    );

    println!("\n⚠️  Outliers:");
    println!("  Detected:         {}", report1.outliers.outlier_count);

    println!("\n✨ Quality Score:   {:.3}/1.0", report1.quality_score);

    // Test 2: Analyze random data
    println!("\n[3/4] Analyzing Random Uniform Data");
    println!("{}", "-".repeat(70));

    let report2 = analytics.analyze(&random_vectors)?;

    println!("\n📈 Distribution:");
    println!(
        "  Mean magnitude:   {:.4}",
        report2.distribution.mean_magnitude
    );
    println!("  Std deviation:    {:.4}", report2.distribution.std_dev);

    println!("\n🔗 Similarity:");
    println!(
        "  Mean similarity:  {:.4}",
        report2.similarity.mean_similarity
    );
    println!("  Variance:         {:.4}", report2.similarity.variance);

    println!("\n🎯 Clustering:");
    println!(
        "  Tendency score:   {:.4}",
        report2.cluster_tendency.tendency_score
    );

    println!("\n✨ Quality Score:   {:.3}/1.0", report2.quality_score);

    // Test 3: Analyze data with outliers
    println!("\n[4/4] Analyzing Data with Outliers");
    println!("{}", "-".repeat(70));

    let report3 = analytics.analyze(&outlier_vectors)?;

    println!("\n📈 Distribution:");
    println!(
        "  Mean magnitude:   {:.4}",
        report3.distribution.mean_magnitude
    );
    println!("  Std deviation:    {:.4}", report3.distribution.std_dev);
    println!(
        "  Min/Max:          [{:.4}, {:.4}]",
        report3.distribution.min_magnitude, report3.distribution.max_magnitude
    );

    println!("\n⚠️  Outliers:");
    println!("  Detected:         {}", report3.outliers.outlier_count);
    println!("  Threshold:        {:.1}σ", report3.outliers.threshold);
    if !report3.outliers.outlier_indices.is_empty() {
        println!(
            "  Indices:          {:?}",
            &report3.outliers.outlier_indices[..report3.outliers.outlier_indices.len().min(5)]
        );
        println!(
            "  Scores:           {:.2?}",
            &report3.outliers.outlier_scores[..report3.outliers.outlier_scores.len().min(5)]
        );
    }

    println!("\n✨ Quality Score:   {:.3}/1.0", report3.quality_score);

    // Test 4: Dimension analysis
    println!("\n[5/5] Per-Dimension Analysis");
    println!("{}", "-".repeat(70));

    println!("\nTop 3 Most Important Dimensions (Clustered Data):");
    let mut sorted_dims = report1.dimension_stats.clone();
    sorted_dims.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());

    for (i, dim) in sorted_dims.iter().take(3).enumerate() {
        println!("\n  Rank {}: Dimension {}", i + 1, dim.dimension);
        println!("    Importance:  {:.3}", dim.importance);
        println!("    Mean:        {:.3}", dim.mean);
        println!("    Std Dev:     {:.3}", dim.std_dev);
        println!("    Range:       [{:.3}, {:.3}]", dim.min, dim.max);
    }

    // Test 5: Similarity histogram
    println!("\n[6/6] Similarity Distribution Histogram");
    println!("{}", "-".repeat(70));

    println!("\nSimilarity histogram (clustered data):");
    let hist = &report1.similarity.histogram;
    let max_count = hist.iter().map(|(_, c)| c).max().unwrap_or(&1);

    for (bin_center, count) in hist.iter().take(10) {
        let bar_length = ((*count as f32 / *max_count as f32) * 40.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("  {:.2}: {:<40} {}", bin_center, bar, count);
    }

    // Comparison Summary
    println!("\n{}", "=".repeat(70));
    println!("📊 Comparison Summary");
    println!("{}", "=".repeat(70));

    println!(
        "\n{:<25} {:<15} {:<15} {:<15}",
        "Metric", "Clustered", "Random", "With Outliers"
    );
    println!("{}", "-".repeat(70));

    println!(
        "{:<25} {:<15.3} {:<15.3} {:<15.3}",
        "Quality Score", report1.quality_score, report2.quality_score, report3.quality_score
    );

    println!(
        "{:<25} {:<15.3} {:<15.3} {:<15.3}",
        "Mean Similarity",
        report1.similarity.mean_similarity,
        report2.similarity.mean_similarity,
        report3.similarity.mean_similarity
    );

    println!(
        "{:<25} {:<15.3} {:<15.3} {:<15.3}",
        "Cluster Tendency",
        report1.cluster_tendency.tendency_score,
        report2.cluster_tendency.tendency_score,
        report3.cluster_tendency.tendency_score
    );

    println!(
        "{:<25} {:<15} {:<15} {:<15}",
        "Outliers",
        report1.outliers.outlier_count,
        report2.outliers.outlier_count,
        report3.outliers.outlier_count
    );

    // Generate full text report
    println!("\n{}", "=".repeat(70));
    println!("📄 Full Text Report (Clustered Data)");
    println!("{}", "=".repeat(70));

    let text_report = analytics.generate_report(&report1);
    println!("{}", text_report);

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("✅ Demo Complete!");
    println!("{}", "=".repeat(70));

    println!("\n✨ Key Features Demonstrated:");
    println!("  ✓ Distribution analysis (mean, variance, skewness, kurtosis)");
    println!("  ✓ Similarity distribution and statistics");
    println!("  ✓ Per-dimension importance analysis");
    println!("  ✓ Cluster tendency detection");
    println!("  ✓ Statistical outlier detection");
    println!("  ✓ Quality score computation");
    println!("  ✓ Histogram generation");
    println!("  ✓ Text report generation");

    println!("\n🎯 Use Cases:");
    println!("  • Understanding vector data characteristics");
    println!("  • Detecting data quality issues");
    println!("  • Identifying natural clusters");
    println!("  • Finding outliers and anomalies");
    println!("  • Monitoring data drift");
    println!("  • Optimizing indexing strategies");

    println!();

    Ok(())
}
