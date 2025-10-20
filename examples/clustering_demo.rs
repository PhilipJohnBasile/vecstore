//! Vector clustering demonstration
//!
//! Shows how to use different clustering algorithms:
//! - K-means: Partitioning into K clusters
//! - DBSCAN: Density-based clustering
//! - Hierarchical: Agglomerative clustering
//! - Cluster quality metrics

use anyhow::Result;
use vecstore::clustering::{
    ClusteringConfig, DBSCANClustering, DBSCANConfig, HierarchicalClustering, HierarchicalConfig,
    KMeansClustering, LinkageMethod,
};

fn generate_clustered_data() -> Vec<Vec<f32>> {
    let mut vectors = Vec::new();

    // Cluster 1: around (1, 1)
    for i in 0..20 {
        vectors.push(vec![
            1.0 + ((i % 5) as f32) * 0.2,
            1.0 + ((i / 5) as f32) * 0.2,
        ]);
    }

    // Cluster 2: around (8, 8)
    for i in 0..20 {
        vectors.push(vec![
            8.0 + ((i % 5) as f32) * 0.2,
            8.0 + ((i / 5) as f32) * 0.2,
        ]);
    }

    // Cluster 3: around (1, 8)
    for i in 0..20 {
        vectors.push(vec![
            1.0 + ((i % 5) as f32) * 0.2,
            8.0 + ((i / 5) as f32) * 0.2,
        ]);
    }

    // Add some noise points
    vectors.push(vec![4.5, 4.5]);
    vectors.push(vec![5.0, 5.0]);

    vectors
}

fn main() -> Result<()> {
    println!("📊 VecStore Clustering Demo\n");
    println!("{}", "=".repeat(80));

    // Generate test data
    println!("\n[1/4] Generating test data...");
    let vectors = generate_clustered_data();
    println!(
        "   ✓ Generated {} vectors in 3 clusters + 2 noise points",
        vectors.len()
    );

    // K-means clustering
    println!("\n[2/4] Running K-means clustering...");
    let kmeans_config = ClusteringConfig {
        k: 3,
        max_iterations: 100,
        tolerance: 0.001,
    };

    let kmeans = KMeansClustering::new(kmeans_config);
    let kmeans_result = kmeans.fit(&vectors)?;

    println!("   K-means results:");
    println!("   • Iterations: {}", kmeans_result.iterations);
    println!("   • Inertia: {:.2}", kmeans_result.inertia);
    if let Some(score) = kmeans_result.silhouette_score {
        println!("   • Silhouette score: {:.3}", score);
    }

    // Show cluster distribution
    let mut cluster_counts = vec![0; 3];
    for &label in &kmeans_result.labels {
        if label >= 0 && (label as usize) < 3 {
            cluster_counts[label as usize] += 1;
        }
    }
    println!("   • Cluster sizes: {:?}", cluster_counts);

    if let Some(ref centroids) = kmeans_result.centroids {
        println!("   • Centroids:");
        for (i, centroid) in centroids.iter().enumerate() {
            println!(
                "     - Cluster {}: [{:.2}, {:.2}]",
                i, centroid[0], centroid[1]
            );
        }
    }

    // DBSCAN clustering
    println!("\n[3/4] Running DBSCAN clustering...");
    let dbscan_config = DBSCANConfig {
        eps: 1.0,
        min_points: 5,
    };

    let dbscan = DBSCANClustering::new(dbscan_config);
    let dbscan_result = dbscan.fit(&vectors)?;

    // Count clusters and noise
    let max_label = dbscan_result.labels.iter().max().unwrap_or(&-1);
    let num_clusters = if *max_label >= 0 { *max_label + 1 } else { 0 };
    let noise_count = dbscan_result.labels.iter().filter(|&&l| l == -1).count();

    println!("   DBSCAN results:");
    println!("   • Number of clusters: {}", num_clusters);
    println!("   • Noise points: {}", noise_count);

    let mut dbscan_cluster_counts = vec![0; num_clusters as usize];
    for &label in &dbscan_result.labels {
        if label >= 0 {
            dbscan_cluster_counts[label as usize] += 1;
        }
    }
    if !dbscan_cluster_counts.is_empty() {
        println!("   • Cluster sizes: {:?}", dbscan_cluster_counts);
    }

    // Hierarchical clustering
    println!("\n[4/4] Running hierarchical clustering...");
    let hier_config = HierarchicalConfig {
        n_clusters: 3,
        linkage: LinkageMethod::Average,
    };

    let hierarchical = HierarchicalClustering::new(hier_config);
    let hier_result = hierarchical.fit(&vectors)?;

    let mut hier_cluster_counts = vec![0; 3];
    for &label in &hier_result.labels {
        if label >= 0 && (label as usize) < 3 {
            hier_cluster_counts[label as usize] += 1;
        }
    }

    println!("   Hierarchical clustering results:");
    println!("   • Iterations (merges): {}", hier_result.iterations);
    println!("   • Cluster sizes: {:?}", hier_cluster_counts);
    println!("   • Linkage method: Average");

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("📊 Summary");
    println!("{}", "=".repeat(80));

    println!("\n✅ Clustering algorithms working!");

    println!("\n💡 Algorithm Characteristics:");
    println!("   K-means:");
    println!("   • Fast and scalable");
    println!("   • Requires K to be specified");
    println!("   • Works best with spherical clusters");
    println!("   • Time complexity: O(n*k*i) where i = iterations");

    println!("\n   DBSCAN:");
    println!("   • Automatically finds number of clusters");
    println!("   • Can detect outliers/noise");
    println!("   • Works with arbitrary cluster shapes");
    println!("   • Time complexity: O(n²) or O(n log n) with index");

    println!("\n   Hierarchical:");
    println!("   • Creates cluster hierarchy");
    println!("   • Deterministic results");
    println!("   • Different linkage methods available");
    println!("   • Time complexity: O(n²) for naive implementation");

    println!("\n🚀 Use Cases:");
    println!("   • Customer segmentation: Group users by behavior");
    println!("   • Document organization: Group similar documents");
    println!("   • Image search: Cluster similar images");
    println!("   • Anomaly detection: Find outliers using DBSCAN");
    println!("   • Data exploration: Discover patterns in embeddings");
    println!("   • Search optimization: Reduce search space with clusters");

    println!("\n📊 Cluster Quality Metrics:");
    println!("   • Silhouette score: Measures cluster separation (-1 to 1)");
    println!("     - Close to 1: Well-separated clusters");
    println!("     - Close to 0: Overlapping clusters");
    println!("     - Negative: Misclassified points");
    println!("   • Inertia: Within-cluster sum of squares (lower is better)");

    Ok(())
}
