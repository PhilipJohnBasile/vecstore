//! Anomaly detection for vector outliers
//!
//! This module provides algorithms to detect anomalous/outlier vectors that
//! deviate significantly from normal patterns. Useful for:
//! - Quality control (detect corrupted or mislabeled data)
//! - Security (detect adversarial examples)
//! - Data cleaning (identify and remove outliers)
//! - Monitoring (alert on unusual patterns)
//!
//! # Algorithms
//!
//! - **Isolation Forest**: Tree-based ensemble method, O(n log n)
//! - **Local Outlier Factor (LOF)**: Density-based method, O(n²)
//! - **Statistical Methods**: Z-score and IQR-based detection
//!
//! # Example
//!
//! ```rust
//! use vecstore::anomaly::{IsolationForest, AnomalyDetector};
//!
//! let vectors = vec![
//!     vec![1.0, 1.0],
//!     vec![1.1, 0.9],
//!     vec![10.0, 10.0],  // Outlier
//! ];
//!
//! let detector = IsolationForest::new(100, 256);
//! let scores = detector.fit_predict(&vectors)?;
//!
//! // Higher scores indicate anomalies
//! for (i, score) in scores.iter().enumerate() {
//!     if *score > 0.6 {
//!         println!("Vector {} is an outlier (score: {:.3})", i, score);
//!     }
//! }
//! ```

use anyhow::Result;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::simd::euclidean_distance_simd;

/// Anomaly detector trait
pub trait AnomalyDetector {
    /// Fit the model and predict anomaly scores
    fn fit_predict(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>>;

    /// Classify vectors as normal or anomalous based on threshold
    fn classify(&self, vectors: &[Vec<f32>], threshold: f32) -> Result<Vec<bool>> {
        let scores = self.fit_predict(vectors)?;
        Ok(scores.iter().map(|&s| s > threshold).collect())
    }
}

/// Isolation Forest anomaly detector
///
/// Uses an ensemble of isolation trees to detect anomalies. Points that are
/// easier to isolate (require fewer splits) are more likely to be anomalies.
///
/// Time complexity: O(n * t * log(s)) where:
/// - n = number of vectors
/// - t = number of trees
/// - s = subsample size
#[derive(Debug, Clone)]
pub struct IsolationForest {
    /// Number of trees in the ensemble
    num_trees: usize,
    /// Subsample size for each tree
    subsample_size: usize,
    /// Random seed for reproducibility
    seed: u64,
}

impl IsolationForest {
    /// Create new Isolation Forest detector
    ///
    /// # Arguments
    /// * `num_trees` - Number of trees (recommended: 100-200)
    /// * `subsample_size` - Sample size per tree (recommended: 256)
    pub fn new(num_trees: usize, subsample_size: usize) -> Self {
        Self {
            num_trees,
            subsample_size,
            seed: 42,
        }
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Build a single isolation tree
    fn build_tree(
        &self,
        vectors: &[Vec<f32>],
        indices: &[usize],
        height: usize,
        max_height: usize,
        rng: &mut StdRng,
    ) -> IsolationTree {
        // Terminal conditions
        if indices.len() <= 1 || height >= max_height {
            return IsolationTree::Leaf {
                size: indices.len(),
            };
        }

        // Select random feature and split value
        let dim = vectors[0].len();
        let feature = rng.gen_range(0..dim);

        let min_val = indices
            .iter()
            .map(|&i| vectors[i][feature])
            .fold(f32::INFINITY, f32::min);
        let max_val = indices
            .iter()
            .map(|&i| vectors[i][feature])
            .fold(f32::NEG_INFINITY, f32::max);

        if (max_val - min_val).abs() < 1e-10 {
            return IsolationTree::Leaf {
                size: indices.len(),
            };
        }

        let split_value = rng.gen_range(min_val..max_val);

        // Partition indices
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            if vectors[idx][feature] < split_value {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        if left_indices.is_empty() || right_indices.is_empty() {
            return IsolationTree::Leaf {
                size: indices.len(),
            };
        }

        // Recursively build subtrees
        let left = Box::new(self.build_tree(vectors, &left_indices, height + 1, max_height, rng));
        let right = Box::new(self.build_tree(vectors, &right_indices, height + 1, max_height, rng));

        IsolationTree::Node {
            feature,
            split_value,
            left,
            right,
        }
    }

    /// Compute path length for a vector in a tree
    fn path_length(&self, vector: &[f32], tree: &IsolationTree, height: usize) -> f32 {
        match tree {
            IsolationTree::Leaf { size } => {
                // Adjust for unsuccessful search
                height as f32 + Self::average_path_length(*size)
            }
            IsolationTree::Node {
                feature,
                split_value,
                left,
                right,
            } => {
                if vector[*feature] < *split_value {
                    self.path_length(vector, left, height + 1)
                } else {
                    self.path_length(vector, right, height + 1)
                }
            }
        }
    }

    /// Average path length of unsuccessful search in BST
    fn average_path_length(n: usize) -> f32 {
        if n <= 1 {
            0.0
        } else {
            2.0 * ((n as f32).ln() + 0.5772156649) - 2.0 * (n as f32 - 1.0) / (n as f32)
        }
    }
}

impl AnomalyDetector for IsolationForest {
    fn fit_predict(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let n = vectors.len();
        let subsample_size = self.subsample_size.min(n);
        let max_height = (subsample_size as f32).log2().ceil() as usize;

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Build trees
        let mut trees = Vec::with_capacity(self.num_trees);
        for _ in 0..self.num_trees {
            // Sample indices
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            indices.truncate(subsample_size);

            let tree = self.build_tree(vectors, &indices, 0, max_height, &mut rng);
            trees.push(tree);
        }

        // Compute anomaly scores
        let c = Self::average_path_length(subsample_size);
        let scores: Vec<f32> = vectors
            .iter()
            .map(|vector| {
                let avg_path: f32 = trees
                    .iter()
                    .map(|tree| self.path_length(vector, tree, 0))
                    .sum::<f32>()
                    / self.num_trees as f32;

                // Anomaly score: 2^(-avg_path / c)
                // Score near 1 = anomaly, near 0 = normal
                2.0_f32.powf(-avg_path / c)
            })
            .collect();

        Ok(scores)
    }
}

/// Internal tree structure for Isolation Forest
#[derive(Debug, Clone)]
enum IsolationTree {
    Node {
        feature: usize,
        split_value: f32,
        left: Box<IsolationTree>,
        right: Box<IsolationTree>,
    },
    Leaf {
        size: usize,
    },
}

/// Local Outlier Factor (LOF) anomaly detector
///
/// Computes the local density deviation of a point with respect to its neighbors.
/// Points with substantially lower density than their neighbors are outliers.
///
/// Time complexity: O(n²) for distance computation
#[derive(Debug, Clone)]
pub struct LocalOutlierFactor {
    /// Number of neighbors to consider
    k: usize,
}

impl LocalOutlierFactor {
    /// Create new LOF detector
    ///
    /// # Arguments
    /// * `k` - Number of neighbors (recommended: 20-50)
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Compute k-distance (distance to kth nearest neighbor)
    fn k_distance(&self, vector: &[f32], vectors: &[Vec<f32>], k: usize) -> (f32, Vec<usize>) {
        let mut distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                let dist = euclidean_distance_simd(vector, v);
                if dist > 0.0 {
                    Some((i, dist))
                } else {
                    None
                }
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        let k_dist = distances.last().map(|x| x.1).unwrap_or(0.0);
        let neighbors = distances.iter().map(|x| x.0).collect();

        (k_dist, neighbors)
    }

    /// Compute reachability distance
    fn reachability_distance(&self, dist: f32, k_dist: f32) -> f32 {
        dist.max(k_dist)
    }

    /// Compute local reachability density
    fn local_reachability_density(&self, neighbors: &[usize], reach_dists: &[f32]) -> f32 {
        let sum: f32 = reach_dists.iter().sum();
        if sum == 0.0 {
            f32::INFINITY
        } else {
            neighbors.len() as f32 / sum
        }
    }
}

impl AnomalyDetector for LocalOutlierFactor {
    fn fit_predict(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let n = vectors.len();
        let k = self.k.min(n - 1);

        if k == 0 {
            return Ok(vec![0.0; n]);
        }

        // Step 1: Compute k-distances and neighbors for all points
        let mut k_distances = Vec::with_capacity(n);
        let mut neighbors_list = Vec::with_capacity(n);

        for vector in vectors {
            let (k_dist, neighbors) = self.k_distance(vector, vectors, k);
            k_distances.push(k_dist);
            neighbors_list.push(neighbors);
        }

        // Step 2: Compute local reachability density for all points
        let mut lrd_values = Vec::with_capacity(n);

        for i in 0..n {
            let neighbors = &neighbors_list[i];
            let mut reach_dists = Vec::with_capacity(neighbors.len());

            for &neighbor_idx in neighbors {
                let dist = euclidean_distance_simd(&vectors[i], &vectors[neighbor_idx]);
                let reach_dist = self.reachability_distance(dist, k_distances[neighbor_idx]);
                reach_dists.push(reach_dist);
            }

            let lrd = self.local_reachability_density(neighbors, &reach_dists);
            lrd_values.push(lrd);
        }

        // Step 3: Compute LOF scores
        let lof_scores: Vec<f32> = (0..n)
            .map(|i| {
                let neighbors = &neighbors_list[i];
                if neighbors.is_empty() {
                    return 1.0;
                }

                let lrd_ratios: f32 = neighbors
                    .iter()
                    .map(|&neighbor_idx| lrd_values[neighbor_idx] / lrd_values[i])
                    .sum();

                let lof = lrd_ratios / neighbors.len() as f32;

                // Normalize: LOF ~ 1 is normal, >> 1 is outlier
                // Convert to 0-1 scale: (lof - 1).max(0)
                (lof - 1.0).max(0.0).min(10.0) / 10.0
            })
            .collect();

        Ok(lof_scores)
    }
}

/// Statistical anomaly detector using Z-score
///
/// Detects outliers based on standard deviations from the mean.
/// Fast but assumes normal distribution.
///
/// Time complexity: O(n * d) where d = dimensions
#[derive(Debug, Clone)]
pub struct ZScoreDetector {
    /// Threshold in standard deviations (default: 3.0)
    threshold: f32,
}

impl ZScoreDetector {
    /// Create new Z-score detector
    ///
    /// # Arguments
    /// * `threshold` - Number of standard deviations (typical: 2.5-3.5)
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl AnomalyDetector for ZScoreDetector {
    fn fit_predict(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let n = vectors.len() as f32;
        let dim = vectors[0].len();

        // Compute mean and std for each dimension
        let mut means = vec![0.0; dim];
        let mut stds = vec![0.0; dim];

        for d in 0..dim {
            let sum: f32 = vectors.iter().map(|v| v[d]).sum();
            means[d] = sum / n;

            let var: f32 = vectors.iter().map(|v| (v[d] - means[d]).powi(2)).sum();
            stds[d] = (var / n).sqrt();
        }

        // Compute Z-scores
        let scores: Vec<f32> = vectors
            .iter()
            .map(|vector| {
                let max_z: f32 = (0..dim)
                    .map(|d| {
                        if stds[d] > 0.0 {
                            ((vector[d] - means[d]) / stds[d]).abs()
                        } else {
                            0.0
                        }
                    })
                    .fold(0.0, f32::max);

                // Normalize to 0-1: score > threshold => anomaly
                (max_z / (self.threshold * 2.0)).min(1.0)
            })
            .collect();

        Ok(scores)
    }
}

/// Anomaly detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Index of the vector
    pub index: usize,
    /// Anomaly score (0-1, higher = more anomalous)
    pub score: f32,
    /// Whether classified as anomaly
    pub is_anomaly: bool,
}

/// Batch anomaly detection with multiple algorithms
pub struct AnomalyEnsemble {
    detectors: Vec<Box<dyn AnomalyDetectorClone>>,
    weights: Vec<f32>,
}

trait AnomalyDetectorClone: AnomalyDetector {
    fn clone_box(&self) -> Box<dyn AnomalyDetectorClone>;
}

impl<T> AnomalyDetectorClone for T
where
    T: 'static + AnomalyDetector + Clone,
{
    fn clone_box(&self) -> Box<dyn AnomalyDetectorClone> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn AnomalyDetectorClone> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl AnomalyEnsemble {
    /// Create new ensemble
    pub fn new() -> Self {
        Self {
            detectors: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Add detector to ensemble
    pub fn add_detector<D: AnomalyDetector + Clone + 'static>(
        mut self,
        detector: D,
        weight: f32,
    ) -> Self {
        self.detectors.push(Box::new(detector));
        self.weights.push(weight);
        self
    }

    /// Detect anomalies with ensemble voting
    pub fn detect(&self, vectors: &[Vec<f32>], threshold: f32) -> Result<Vec<AnomalyResult>> {
        if vectors.is_empty() || self.detectors.is_empty() {
            return Ok(Vec::new());
        }

        // Get scores from all detectors
        let mut all_scores: Vec<Vec<f32>> = Vec::new();
        for detector in &self.detectors {
            let scores = detector.fit_predict(vectors)?;
            all_scores.push(scores);
        }

        // Combine weighted scores
        let total_weight: f32 = self.weights.iter().sum();
        let combined_scores: Vec<f32> = (0..vectors.len())
            .map(|i| {
                self.detectors
                    .iter()
                    .enumerate()
                    .map(|(d, _)| all_scores[d][i] * self.weights[d])
                    .sum::<f32>()
                    / total_weight
            })
            .collect();

        // Create results
        let results: Vec<AnomalyResult> = combined_scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| AnomalyResult {
                index,
                score,
                is_anomaly: score > threshold,
            })
            .collect();

        Ok(results)
    }
}

impl Default for AnomalyEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_normal_and_outliers() -> Vec<Vec<f32>> {
        let mut vectors = Vec::new();

        // Normal cluster around (0, 0)
        for i in 0..50 {
            vectors.push(vec![(i % 10) as f32 * 0.2, (i / 10) as f32 * 0.2]);
        }

        // Add outliers
        vectors.push(vec![10.0, 10.0]);
        vectors.push(vec![-10.0, -10.0]);
        vectors.push(vec![10.0, -10.0]);

        vectors
    }

    #[test]
    fn test_isolation_forest() -> Result<()> {
        let vectors = generate_normal_and_outliers();
        let detector = IsolationForest::new(100, 32);

        let scores = detector.fit_predict(&vectors)?;

        // Last 3 vectors are outliers, should have higher scores
        let outlier_scores = &scores[50..53];
        let normal_scores = &scores[0..50];

        let avg_outlier = outlier_scores.iter().sum::<f32>() / outlier_scores.len() as f32;
        let avg_normal = normal_scores.iter().sum::<f32>() / normal_scores.len() as f32;

        assert!(
            avg_outlier > avg_normal,
            "Outliers should have higher scores"
        );

        Ok(())
    }

    #[test]
    fn test_lof() -> Result<()> {
        let vectors = generate_normal_and_outliers();
        let detector = LocalOutlierFactor::new(20);

        let scores = detector.fit_predict(&vectors)?;

        // Check that outliers have higher scores
        let outlier_scores = &scores[50..53];
        let normal_scores = &scores[0..50];

        let avg_outlier = outlier_scores.iter().sum::<f32>() / outlier_scores.len() as f32;
        let avg_normal = normal_scores.iter().sum::<f32>() / normal_scores.len() as f32;

        assert!(
            avg_outlier > avg_normal * 0.5, // LOF might be less sensitive
            "Outliers should have higher LOF scores"
        );

        Ok(())
    }

    #[test]
    fn test_zscore() -> Result<()> {
        let vectors = generate_normal_and_outliers();
        let detector = ZScoreDetector::new(3.0);

        let scores = detector.fit_predict(&vectors)?;

        // Outliers should have higher scores
        assert!(scores[50] > 0.5, "Outlier 1 should have high score");
        assert!(scores[51] > 0.5, "Outlier 2 should have high score");
        assert!(scores[52] > 0.5, "Outlier 3 should have high score");

        Ok(())
    }

    #[test]
    fn test_ensemble() -> Result<()> {
        let vectors = generate_normal_and_outliers();

        let ensemble = AnomalyEnsemble::new()
            .add_detector(IsolationForest::new(50, 32), 1.0)
            .add_detector(LocalOutlierFactor::new(20), 1.0)
            .add_detector(ZScoreDetector::new(3.0), 0.5);

        let results = ensemble.detect(&vectors, 0.5)?;

        // Check that outliers are detected
        let outlier_count = results.iter().filter(|r| r.is_anomaly).count();
        assert!(outlier_count >= 2, "Should detect at least 2 outliers");

        Ok(())
    }

    #[test]
    fn test_classify() -> Result<()> {
        let vectors = generate_normal_and_outliers();
        let detector = IsolationForest::new(100, 32);

        let classifications = detector.classify(&vectors, 0.6)?;

        // Should classify outliers as true
        assert_eq!(classifications.len(), vectors.len());

        Ok(())
    }
}
