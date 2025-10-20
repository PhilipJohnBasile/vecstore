//! Anomaly detection demonstration
//!
//! Shows how to detect outlier vectors using:
//! - Isolation Forest (tree-based ensemble)
//! - Local Outlier Factor (LOF) - density-based
//! - Z-Score (statistical method)
//! - Ensemble voting (combine multiple algorithms)

use anyhow::Result;
use vecstore::anomaly::{
    AnomalyDetector, AnomalyEnsemble, IsolationForest, LocalOutlierFactor, ZScoreDetector,
};

fn generate_dataset() -> Vec<Vec<f32>> {
    let mut vectors = Vec::new();

    println!("   Generating dataset:");

    // Normal cluster 1: around (1, 1)
    for i in 0..30 {
        vectors.push(vec![
            1.0 + (i % 6) as f32 * 0.15,
            1.0 + (i / 6) as f32 * 0.15,
        ]);
    }
    println!("   • 30 normal vectors around (1, 1)");

    // Normal cluster 2: around (5, 5)
    for i in 0..30 {
        vectors.push(vec![
            5.0 + (i % 6) as f32 * 0.15,
            5.0 + (i / 6) as f32 * 0.15,
        ]);
    }
    println!("   • 30 normal vectors around (5, 5)");

    // Add outliers (far from clusters)
    vectors.push(vec![10.0, 1.0]); // Outlier 1
    vectors.push(vec![1.0, 10.0]); // Outlier 2
    vectors.push(vec![10.0, 10.0]); // Outlier 3
    vectors.push(vec![-5.0, -5.0]); // Outlier 4
    println!("   • 4 outlier vectors far from clusters");

    vectors
}

fn main() -> Result<()> {
    println!("🔍 VecStore Anomaly Detection Demo\n");
    println!("{}", "=".repeat(80));

    // Generate test data
    println!("\n[1/5] Generating test data...");
    let vectors = generate_dataset();
    println!("   ✓ Total vectors: {}", vectors.len());

    // Method 1: Isolation Forest
    println!("\n[2/5] Isolation Forest detection...");
    let if_detector = IsolationForest::new(100, 32);
    let if_scores = if_detector.fit_predict(&vectors)?;

    println!("   Top 5 anomalies by Isolation Forest:");
    let mut if_scored: Vec<(usize, f32)> = if_scores
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();
    if_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (idx, score)) in if_scored.iter().take(5).enumerate() {
        println!(
            "   {}. Vector {} [{:.2}, {:.2}] - Score: {:.4}",
            rank + 1,
            idx,
            vectors[*idx][0],
            vectors[*idx][1],
            score
        );
    }

    // Method 2: Local Outlier Factor (LOF)
    println!("\n[3/5] Local Outlier Factor (LOF) detection...");
    let lof_detector = LocalOutlierFactor::new(20);
    let lof_scores = lof_detector.fit_predict(&vectors)?;

    println!("   Top 5 anomalies by LOF:");
    let mut lof_scored: Vec<(usize, f32)> = lof_scores
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();
    lof_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (idx, score)) in lof_scored.iter().take(5).enumerate() {
        println!(
            "   {}. Vector {} [{:.2}, {:.2}] - Score: {:.4}",
            rank + 1,
            idx,
            vectors[*idx][0],
            vectors[*idx][1],
            score
        );
    }

    // Method 3: Z-Score
    println!("\n[4/5] Z-Score detection...");
    let zscore_detector = ZScoreDetector::new(3.0);
    let zscore_scores = zscore_detector.fit_predict(&vectors)?;

    println!("   Top 5 anomalies by Z-Score:");
    let mut zscore_scored: Vec<(usize, f32)> = zscore_scores
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();
    zscore_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (idx, score)) in zscore_scored.iter().take(5).enumerate() {
        println!(
            "   {}. Vector {} [{:.2}, {:.2}] - Score: {:.4}",
            rank + 1,
            idx,
            vectors[*idx][0],
            vectors[*idx][1],
            score
        );
    }

    // Method 4: Ensemble (combining all methods)
    println!("\n[5/5] Ensemble detection (combining all methods)...");
    let ensemble = AnomalyEnsemble::new()
        .add_detector(IsolationForest::new(100, 32), 1.0)
        .add_detector(LocalOutlierFactor::new(20), 1.0)
        .add_detector(ZScoreDetector::new(3.0), 0.8);

    let results = ensemble.detect(&vectors, 0.5)?;

    let anomalies: Vec<_> = results.iter().filter(|r| r.is_anomaly).collect();
    println!(
        "   Detected {} anomalies (threshold: 0.5):",
        anomalies.len()
    );

    let mut sorted_anomalies = anomalies.clone();
    sorted_anomalies.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    for (rank, result) in sorted_anomalies.iter().take(10).enumerate() {
        println!(
            "   {}. Vector {} [{:.2}, {:.2}] - Ensemble Score: {:.4}",
            rank + 1,
            result.index,
            vectors[result.index][0],
            vectors[result.index][1],
            result.score
        );
    }

    // Verify detection accuracy
    let true_outliers = vec![60, 61, 62, 63]; // Known outlier indices
    let detected_outliers: Vec<usize> = anomalies.iter().map(|r| r.index).collect();

    let correct = true_outliers
        .iter()
        .filter(|&&idx| detected_outliers.contains(&idx))
        .count();

    println!(
        "\n   ✓ Detection accuracy: {}/{} outliers correctly identified",
        correct,
        true_outliers.len()
    );

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("📊 Summary");
    println!("{}", "=".repeat(80));

    println!("\n✅ Anomaly detection working!");

    println!("\n🔬 Algorithm Comparison:");
    println!("\n   Isolation Forest:");
    println!("   • Best for: High-dimensional data, large datasets");
    println!("   • Complexity: O(n log n) - Fast");
    println!("   • Approach: Tree-based isolation");
    println!("   • Strengths: Efficient, handles high dimensions");
    println!("   • Weaknesses: May miss local anomalies");

    println!("\n   Local Outlier Factor (LOF):");
    println!("   • Best for: Varying density clusters");
    println!("   • Complexity: O(n²) - Slower");
    println!("   • Approach: Local density deviation");
    println!("   • Strengths: Detects local outliers well");
    println!("   • Weaknesses: Expensive for large datasets");

    println!("\n   Z-Score:");
    println!("   • Best for: Quick screening, normally distributed data");
    println!("   • Complexity: O(n) - Fastest");
    println!("   • Approach: Statistical deviation");
    println!("   • Strengths: Fast, interpretable");
    println!("   • Weaknesses: Assumes normal distribution");

    println!("\n   Ensemble:");
    println!("   • Best for: Robust detection across scenarios");
    println!("   • Combines strengths of multiple algorithms");
    println!("   • Reduces false positives through voting");
    println!("   • Recommended for production use");

    println!("\n💡 Use Cases:");
    println!("   • Data quality control: Find corrupted/mislabeled vectors");
    println!("   • Security: Detect adversarial examples");
    println!("   • Monitoring: Alert on unusual patterns");
    println!("   • Data cleaning: Remove outliers before training");
    println!("   • Fraud detection: Identify unusual behavior");
    println!("   • System health: Detect anomalous system states");

    println!("\n⚙️  Configuration Tips:");
    println!("   • Isolation Forest: 100-200 trees, subsample 256");
    println!("   • LOF: k=20-50 neighbors");
    println!("   • Z-Score: threshold 2.5-3.5 std deviations");
    println!("   • Ensemble: Weight based on data characteristics");

    println!("\n🎯 Threshold Selection:");
    println!("   • Low threshold (0.3-0.4): More anomalies, higher recall");
    println!("   • Medium threshold (0.5-0.6): Balanced");
    println!("   • High threshold (0.7-0.8): Fewer anomalies, higher precision");
    println!("   • Calibrate on labeled data when available");

    Ok(())
}
