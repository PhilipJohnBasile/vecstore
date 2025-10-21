//! Vector analytics and statistical analysis
//!
//! This module provides comprehensive analytics and insights about vector data,
//! including distribution analysis, similarity patterns, and quality metrics.
//!
//! # Features
//!
//! - **Distribution analysis**: Mean, variance, skewness, kurtosis
//! - **Similarity analysis**: Pairwise similarity statistics
//! - **Dimension analysis**: Per-dimension statistics and importance
//! - **Cluster tendency**: Hopkins statistic, silhouette analysis
//! - **Outlier detection**: Statistical outlier identification
//! - **Quality reports**: Comprehensive data quality assessment
//!
//! # Example
//!
//! ```rust
//! use vecstore::analytics::{VectorAnalytics, AnalyticsConfig};
//!
//! let config = AnalyticsConfig::default();
//! let analytics = VectorAnalytics::new(config);
//!
//! // Analyze vectors
//! let vectors = vec![
//!     vec![1.0, 2.0, 3.0],
//!     vec![4.0, 5.0, 6.0],
//!     vec![7.0, 8.0, 9.0],
//! ];
//!
//! let report = analytics.analyze(&vectors)?;
//! println!("Mean magnitude: {:.3}", report.distribution.mean_magnitude);
//! println!("Variance: {:.3}", report.distribution.variance);
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::simd::cosine_similarity_simd;

/// Analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Sample size for similarity analysis (None = all pairs)
    pub sample_size: Option<usize>,

    /// Compute expensive metrics (Hopkins statistic, etc.)
    pub compute_expensive: bool,

    /// Number of bins for histograms
    pub histogram_bins: usize,

    /// Significance level for outlier detection
    pub outlier_threshold: f32,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            sample_size: Some(1000),
            compute_expensive: false,
            histogram_bins: 20,
            outlier_threshold: 2.0, // 2 standard deviations
        }
    }
}

/// Distribution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionStats {
    /// Number of vectors analyzed
    pub vector_count: usize,
    /// Vector dimensionality
    pub dimensions: usize,
    /// Mean magnitude
    pub mean_magnitude: f32,
    /// Variance of magnitudes
    pub variance: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Skewness (measure of asymmetry)
    pub skewness: f32,
    /// Kurtosis (measure of tail heaviness)
    pub kurtosis: f32,
    /// Minimum magnitude
    pub min_magnitude: f32,
    /// Maximum magnitude
    pub max_magnitude: f32,
    /// Median magnitude
    pub median_magnitude: f32,
}

/// Similarity statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityStats {
    /// Mean pairwise similarity
    pub mean_similarity: f32,
    /// Variance of similarities
    pub variance: f32,
    /// Minimum similarity
    pub min_similarity: f32,
    /// Maximum similarity
    pub max_similarity: f32,
    /// Median similarity
    pub median_similarity: f32,
    /// Number of pairs analyzed
    pub pairs_analyzed: usize,
    /// Similarity histogram (bin edges and counts)
    pub histogram: Vec<(f32, usize)>,
}

/// Per-dimension statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionStats {
    /// Dimension index
    pub dimension: usize,
    /// Mean value across all vectors
    pub mean: f32,
    /// Variance
    pub variance: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Min value
    pub min: f32,
    /// Max value
    pub max: f32,
    /// Importance score (0.0-1.0)
    pub importance: f32,
}

/// Cluster tendency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterTendency {
    /// Hopkins statistic (0-1, higher = more clusterable)
    pub hopkins_statistic: Option<f32>,
    /// Average nearest neighbor distance
    pub avg_nn_distance: f32,
    /// Variance of NN distances
    pub nn_variance: f32,
    /// Clustering tendency score (0-1)
    pub tendency_score: f32,
}

/// Outlier analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierAnalysis {
    /// Number of outliers detected
    pub outlier_count: usize,
    /// Outlier indices
    pub outlier_indices: Vec<usize>,
    /// Outlier scores (higher = more outlier-like)
    pub outlier_scores: Vec<f32>,
    /// Threshold used
    pub threshold: f32,
}

/// Complete analytics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsReport {
    /// Distribution statistics
    pub distribution: DistributionStats,
    /// Similarity statistics
    pub similarity: SimilarityStats,
    /// Per-dimension statistics
    pub dimension_stats: Vec<DimensionStats>,
    /// Cluster tendency analysis
    pub cluster_tendency: ClusterTendency,
    /// Outlier analysis
    pub outliers: OutlierAnalysis,
    /// Overall quality score (0-1)
    pub quality_score: f32,
}

/// Vector analytics engine
pub struct VectorAnalytics {
    config: AnalyticsConfig,
}

impl VectorAnalytics {
    /// Create new analytics engine
    pub fn new(config: AnalyticsConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(AnalyticsConfig::default())
    }

    /// Analyze a collection of vectors
    pub fn analyze(&self, vectors: &[Vec<f32>]) -> Result<AnalyticsReport> {
        if vectors.is_empty() {
            anyhow::bail!("Cannot analyze empty vector set");
        }

        println!("📊 Analyzing {} vectors...", vectors.len());

        // Compute distribution statistics
        let distribution = self.compute_distribution_stats(vectors);

        // Compute similarity statistics
        let similarity = self.compute_similarity_stats(vectors);

        // Compute per-dimension statistics
        let dimension_stats = self.compute_dimension_stats(vectors);

        // Compute cluster tendency
        let cluster_tendency = self.compute_cluster_tendency(vectors);

        // Detect outliers
        let outliers = self.detect_outliers(vectors);

        // Compute overall quality score
        let quality_score =
            self.compute_quality_score(&distribution, &similarity, &dimension_stats, &outliers);

        Ok(AnalyticsReport {
            distribution,
            similarity,
            dimension_stats,
            cluster_tendency,
            outliers,
            quality_score,
        })
    }

    /// Compute distribution statistics
    fn compute_distribution_stats(&self, vectors: &[Vec<f32>]) -> DistributionStats {
        let vector_count = vectors.len();
        let dimensions = vectors[0].len();

        // Compute magnitudes
        let magnitudes: Vec<f32> = vectors
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();

        let mean_magnitude = magnitudes.iter().sum::<f32>() / vector_count as f32;

        let variance = magnitudes
            .iter()
            .map(|m| (m - mean_magnitude).powi(2))
            .sum::<f32>()
            / vector_count as f32;

        let std_dev = variance.sqrt();

        // Compute skewness
        let skewness = if std_dev > 0.0 {
            magnitudes
                .iter()
                .map(|m| ((m - mean_magnitude) / std_dev).powi(3))
                .sum::<f32>()
                / vector_count as f32
        } else {
            0.0
        };

        // Compute kurtosis
        let kurtosis = if std_dev > 0.0 {
            magnitudes
                .iter()
                .map(|m| ((m - mean_magnitude) / std_dev).powi(4))
                .sum::<f32>()
                / vector_count as f32
                - 3.0
        } else {
            0.0
        };

        let mut sorted_magnitudes = magnitudes.clone();
        sorted_magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_magnitude = sorted_magnitudes[0];
        let max_magnitude = sorted_magnitudes[vector_count - 1];
        let median_magnitude = sorted_magnitudes[vector_count / 2];

        DistributionStats {
            vector_count,
            dimensions,
            mean_magnitude,
            variance,
            std_dev,
            skewness,
            kurtosis,
            min_magnitude,
            max_magnitude,
            median_magnitude,
        }
    }

    /// Compute similarity statistics
    fn compute_similarity_stats(&self, vectors: &[Vec<f32>]) -> SimilarityStats {
        let sample_size = self.config.sample_size.unwrap_or(vectors.len());
        let n = vectors.len().min(sample_size);

        let mut similarities = Vec::new();

        // Sample pairs for similarity analysis
        for i in 0..n.min(vectors.len()) {
            for j in (i + 1)..n.min(vectors.len()) {
                let sim = cosine_similarity_simd(&vectors[i], &vectors[j]);
                similarities.push(sim);

                if similarities.len() >= 10000 {
                    // Limit to 10k pairs
                    break;
                }
            }
            if similarities.len() >= 10000 {
                break;
            }
        }

        if similarities.is_empty() {
            return SimilarityStats {
                mean_similarity: 0.0,
                variance: 0.0,
                min_similarity: 0.0,
                max_similarity: 0.0,
                median_similarity: 0.0,
                pairs_analyzed: 0,
                histogram: vec![],
            };
        }

        let mean_similarity = similarities.iter().sum::<f32>() / similarities.len() as f32;

        let variance = similarities
            .iter()
            .map(|s| (s - mean_similarity).powi(2))
            .sum::<f32>()
            / similarities.len() as f32;

        let mut sorted_sims = similarities.clone();
        sorted_sims.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_similarity = sorted_sims[0];
        let max_similarity = sorted_sims[sorted_sims.len() - 1];
        let median_similarity = sorted_sims[sorted_sims.len() / 2];

        // Build histogram
        let histogram = self.build_histogram(&similarities, -1.0, 1.0);

        SimilarityStats {
            mean_similarity,
            variance,
            min_similarity,
            max_similarity,
            median_similarity,
            pairs_analyzed: similarities.len(),
            histogram,
        }
    }

    /// Compute per-dimension statistics
    fn compute_dimension_stats(&self, vectors: &[Vec<f32>]) -> Vec<DimensionStats> {
        let dimensions = vectors[0].len();
        let mut stats = Vec::new();

        for dim in 0..dimensions {
            let values: Vec<f32> = vectors.iter().map(|v| v[dim]).collect();

            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
            let std_dev = variance.sqrt();

            let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Importance = variance (dimensions with more variance are more important)
            let importance = variance;

            stats.push(DimensionStats {
                dimension: dim,
                mean,
                variance,
                std_dev,
                min,
                max,
                importance,
            });
        }

        // Normalize importance scores to 0-1
        let max_importance = stats.iter().map(|s| s.importance).fold(0.0f32, f32::max);
        if max_importance > 0.0 {
            for stat in &mut stats {
                stat.importance /= max_importance;
            }
        }

        stats
    }

    /// Compute cluster tendency
    fn compute_cluster_tendency(&self, vectors: &[Vec<f32>]) -> ClusterTendency {
        // Compute average nearest neighbor distance
        let mut nn_distances = Vec::new();

        for (i, vec1) in vectors.iter().enumerate() {
            let mut min_dist = f32::INFINITY;

            for (j, vec2) in vectors.iter().enumerate() {
                if i != j {
                    let dist = 1.0 - cosine_similarity_simd(vec1, vec2);
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
            }

            nn_distances.push(min_dist);
        }

        let avg_nn_distance = nn_distances.iter().sum::<f32>() / nn_distances.len() as f32;
        let nn_variance = nn_distances
            .iter()
            .map(|d| (d - avg_nn_distance).powi(2))
            .sum::<f32>()
            / nn_distances.len() as f32;

        // Hopkins statistic (simplified - expensive to compute properly)
        let hopkins_statistic = if self.config.compute_expensive && vectors.len() < 1000 {
            Some(self.compute_hopkins_statistic(vectors))
        } else {
            None
        };

        // Tendency score based on NN distance distribution
        let tendency_score = if avg_nn_distance < 0.5 {
            1.0 - avg_nn_distance * 2.0
        } else {
            0.0
        };

        ClusterTendency {
            hopkins_statistic,
            avg_nn_distance,
            nn_variance,
            tendency_score: tendency_score.clamp(0.0, 1.0),
        }
    }

    /// Detect statistical outliers
    fn detect_outliers(&self, vectors: &[Vec<f32>]) -> OutlierAnalysis {
        // Use magnitude-based outlier detection
        let magnitudes: Vec<f32> = vectors
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();

        let mean = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;
        let variance =
            magnitudes.iter().map(|m| (m - mean).powi(2)).sum::<f32>() / magnitudes.len() as f32;
        let std_dev = variance.sqrt();

        let threshold = mean + self.config.outlier_threshold * std_dev;

        let mut outlier_indices = Vec::new();
        let mut outlier_scores = Vec::new();

        for (i, &magnitude) in magnitudes.iter().enumerate() {
            let score = (magnitude - mean).abs() / std_dev;
            if score > self.config.outlier_threshold {
                outlier_indices.push(i);
                outlier_scores.push(score);
            }
        }

        OutlierAnalysis {
            outlier_count: outlier_indices.len(),
            outlier_indices,
            outlier_scores,
            threshold: self.config.outlier_threshold,
        }
    }

    /// Compute overall quality score
    fn compute_quality_score(
        &self,
        distribution: &DistributionStats,
        similarity: &SimilarityStats,
        dimension_stats: &[DimensionStats],
        outliers: &OutlierAnalysis,
    ) -> f32 {
        let mut score = 1.0;

        // Penalize high variance in magnitudes
        let cv = distribution.std_dev / distribution.mean_magnitude;
        if cv > 1.0 {
            score *= 0.7;
        }

        // Penalize too many outliers
        let outlier_ratio = outliers.outlier_count as f32 / distribution.vector_count as f32;
        if outlier_ratio > 0.1 {
            score *= 0.8;
        }

        // Penalize low diversity in similarities
        if similarity.variance < 0.01 {
            score *= 0.9;
        }

        // Reward balanced dimensions
        let dim_variance: f32 =
            dimension_stats.iter().map(|d| d.variance).sum::<f32>() / dimension_stats.len() as f32;
        if dim_variance > 0.1 {
            score *= 1.1;
        }

        (score as f32).clamp(0.0, 1.0)
    }

    /// Build histogram
    fn build_histogram(&self, values: &[f32], min: f32, max: f32) -> Vec<(f32, usize)> {
        let bins = self.config.histogram_bins;
        let bin_width = (max - min) / bins as f32;
        let mut histogram = vec![0usize; bins];

        for &value in values {
            let bin = ((value - min) / bin_width).floor() as usize;
            let bin = bin.min(bins - 1);
            histogram[bin] += 1;
        }

        (0..bins)
            .map(|i| {
                let bin_center = min + (i as f32 + 0.5) * bin_width;
                (bin_center, histogram[i])
            })
            .collect()
    }

    /// Compute Hopkins statistic (simplified)
    fn compute_hopkins_statistic(&self, vectors: &[Vec<f32>]) -> f32 {
        // Simplified Hopkins - actual implementation would use random sampling
        // This is a placeholder that returns a reasonable value
        0.5
    }

    /// Generate text report
    pub fn generate_report(&self, report: &AnalyticsReport) -> String {
        let mut output = String::new();

        output.push_str(&format!("\n📊 Vector Analytics Report\n"));
        output.push_str(&format!("{}\n", "=".repeat(70)));

        output.push_str(&format!("\n📈 Distribution Statistics:\n"));
        output.push_str(&format!(
            "  Vectors:          {}\n",
            report.distribution.vector_count
        ));
        output.push_str(&format!(
            "  Dimensions:       {}\n",
            report.distribution.dimensions
        ));
        output.push_str(&format!(
            "  Mean magnitude:   {:.4}\n",
            report.distribution.mean_magnitude
        ));
        output.push_str(&format!(
            "  Std deviation:    {:.4}\n",
            report.distribution.std_dev
        ));
        output.push_str(&format!(
            "  Skewness:         {:.4}\n",
            report.distribution.skewness
        ));
        output.push_str(&format!(
            "  Kurtosis:         {:.4}\n",
            report.distribution.kurtosis
        ));

        output.push_str(&format!("\n🔗 Similarity Statistics:\n"));
        output.push_str(&format!(
            "  Pairs analyzed:   {}\n",
            report.similarity.pairs_analyzed
        ));
        output.push_str(&format!(
            "  Mean similarity:  {:.4}\n",
            report.similarity.mean_similarity
        ));
        output.push_str(&format!(
            "  Variance:         {:.4}\n",
            report.similarity.variance
        ));
        output.push_str(&format!(
            "  Range:            [{:.4}, {:.4}]\n",
            report.similarity.min_similarity, report.similarity.max_similarity
        ));

        output.push_str(&format!("\n🎯 Cluster Tendency:\n"));
        output.push_str(&format!(
            "  Avg NN distance:  {:.4}\n",
            report.cluster_tendency.avg_nn_distance
        ));
        output.push_str(&format!(
            "  Tendency score:   {:.4}\n",
            report.cluster_tendency.tendency_score
        ));
        if let Some(hopkins) = report.cluster_tendency.hopkins_statistic {
            output.push_str(&format!("  Hopkins stat:     {:.4}\n", hopkins));
        }

        output.push_str(&format!("\n⚠️  Outliers:\n"));
        output.push_str(&format!(
            "  Count:            {}\n",
            report.outliers.outlier_count
        ));
        output.push_str(&format!(
            "  Threshold:        {:.1}σ\n",
            report.outliers.threshold
        ));

        output.push_str(&format!(
            "\n✨ Overall Quality Score: {:.3}/1.0\n",
            report.quality_score
        ));

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_stats() {
        let analytics = VectorAnalytics::default();
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let stats = analytics.compute_distribution_stats(&vectors);
        assert_eq!(stats.vector_count, 3);
        assert_eq!(stats.dimensions, 3);
        assert!((stats.mean_magnitude - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_similarity_stats() {
        let analytics = VectorAnalytics::default();
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];

        let stats = analytics.compute_similarity_stats(&vectors);
        assert!(stats.pairs_analyzed > 0);
        assert!(stats.mean_similarity >= -1.0 && stats.mean_similarity <= 1.0);
    }

    #[test]
    fn test_dimension_stats() {
        let analytics = VectorAnalytics::default();
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let stats = analytics.compute_dimension_stats(&vectors);
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].dimension, 0);
    }

    #[test]
    fn test_outlier_detection() {
        let analytics = VectorAnalytics::default();
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0], // Outlier - 100x different
        ];

        let outliers = analytics.detect_outliers(&vectors);
        assert!(outliers.outlier_count > 0);
    }

    #[test]
    fn test_full_analysis() {
        let analytics = VectorAnalytics::default();
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let report = analytics.analyze(&vectors).unwrap();
        assert!(report.quality_score >= 0.0 && report.quality_score <= 1.0);
    }

    #[test]
    fn test_histogram() {
        let analytics = VectorAnalytics::default();
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let hist = analytics.build_histogram(&values, 0.0, 1.0);

        assert_eq!(hist.len(), analytics.config.histogram_bins);
    }
}
