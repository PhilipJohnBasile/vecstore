//! Embedding Debugger: Diagnostic Tools for Vector Quality
//!
//! A unique VecStore capability that helps developers understand and fix
//! embedding quality issues. No competitor offers this level of insight.
//!
//! ## Capabilities
//!
//! - **Quality Analysis**: Detect degenerate, collapsed, or low-quality embeddings
//! - **Anomaly Detection**: Find outliers and unusual patterns
//! - **Drift Monitoring**: Track embedding distribution changes over time
//! - **Dimension Analysis**: Understand which dimensions carry signal vs noise
//! - **Comparison Tools**: Compare embeddings across models or versions
//!
//! ## Example
//!
//! ```rust,no_run
//! use vecstore::debugger::{EmbeddingDebugger, DebugConfig};
//!
//! let debugger = EmbeddingDebugger::new(DebugConfig::default());
//!
//! // Analyze embedding quality
//! let report = debugger.analyze(&embedding)?;
//! println!("Quality score: {}", report.quality_score);
//!
//! for issue in &report.issues {
//!     println!("Warning: {}", issue.message);
//! }
//!
//! // Check for anomalies in a batch
//! let anomalies = debugger.detect_anomalies(&embeddings)?;
//! ```

use anyhow::{Result};
use serde::{Deserialize, Serialize};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Debugger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Threshold for zero variance dimensions
    pub zero_variance_threshold: f32,

    /// Threshold for low norm embeddings
    pub low_norm_threshold: f32,

    /// Threshold for high norm embeddings
    pub high_norm_threshold: f32,

    /// Z-score threshold for anomaly detection
    pub anomaly_zscore_threshold: f32,

    /// Minimum entropy for non-degenerate embeddings
    pub min_entropy: f32,

    /// Enable drift monitoring
    pub monitor_drift: bool,

    /// Drift detection window size
    pub drift_window_size: usize,

    /// Drift alert threshold
    pub drift_threshold: f32,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            zero_variance_threshold: 1e-7,
            low_norm_threshold: 0.1,
            high_norm_threshold: 100.0,
            anomaly_zscore_threshold: 3.0,
            min_entropy: 0.5,
            monitor_drift: true,
            drift_window_size: 1000,
            drift_threshold: 0.1,
        }
    }
}

// ============================================================================
// QUALITY ANALYSIS
// ============================================================================

/// Quality analysis report for an embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// Overall quality score (0.0 - 1.0)
    pub quality_score: f32,

    /// Grade (A, B, C, D, F)
    pub grade: String,

    /// Detected issues
    pub issues: Vec<QualityIssue>,

    /// Dimension statistics
    pub dimension_stats: DimensionStats,

    /// Distribution analysis
    pub distribution: DistributionAnalysis,

    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// A quality issue detected in an embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue severity (Critical, Warning, Info)
    pub severity: IssueSeverity,

    /// Issue category
    pub category: IssueCategory,

    /// Human-readable message
    pub message: String,

    /// Affected dimensions (if applicable)
    pub affected_dimensions: Option<Vec<usize>>,

    /// Suggested fix
    pub suggestion: String,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    Warning,
    Info,
}

/// Issue categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueCategory {
    /// Embedding has collapsed or near-zero values
    Collapsed,
    /// Embedding has NaN or Inf values
    InvalidValues,
    /// Embedding has unusually low norm
    LowNorm,
    /// Embedding has unusually high norm
    HighNorm,
    /// Many dimensions have zero variance
    LowVariance,
    /// Embedding values are highly concentrated
    LowEntropy,
    /// Distribution is highly skewed
    Skewed,
    /// Unusual outlier dimensions
    Outliers,
}

/// Statistics about embedding dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionStats {
    /// Total dimensions
    pub total: usize,

    /// Dimensions with zero or near-zero variance
    pub zero_variance_dims: usize,

    /// Active (non-zero) dimensions
    pub active_dims: usize,

    /// Most significant dimensions (by absolute value)
    pub top_dimensions: Vec<(usize, f32)>,

    /// Dimensions with highest variance
    pub high_variance_dims: Vec<(usize, f32)>,

    /// Mean value per dimension
    pub mean: f32,

    /// Standard deviation
    pub std: f32,
}

/// Distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    /// Embedding norm (L2)
    pub norm: f32,

    /// Min value
    pub min: f32,

    /// Max value
    pub max: f32,

    /// Mean value
    pub mean: f32,

    /// Median value
    pub median: f32,

    /// Standard deviation
    pub std: f32,

    /// Skewness
    pub skewness: f32,

    /// Kurtosis
    pub kurtosis: f32,

    /// Entropy (information content)
    pub entropy: f32,

    /// Sparsity (fraction of near-zero values)
    pub sparsity: f32,
}

// ============================================================================
// EMBEDDING DEBUGGER
// ============================================================================

/// Embedding debugger for quality analysis and monitoring
pub struct EmbeddingDebugger {
    config: DebugConfig,

    /// Historical embeddings for drift detection
    history: Vec<HistoryEntry>,

    /// Baseline statistics (for comparison)
    baseline: Option<BaselineStats>,
}

/// Historical entry for drift detection
#[derive(Debug, Clone)]
struct HistoryEntry {
    timestamp: i64,
    mean_norm: f32,
    mean_per_dim: Vec<f32>,
    std_per_dim: Vec<f32>,
}

/// Baseline statistics for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineStats {
    pub dimension: usize,
    pub mean_norm: f32,
    pub std_norm: f32,
    pub mean_per_dim: Vec<f32>,
    pub std_per_dim: Vec<f32>,
    pub sample_count: usize,
}

impl EmbeddingDebugger {
    /// Create new debugger
    pub fn new(config: DebugConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            baseline: None,
        }
    }

    /// Analyze a single embedding
    pub fn analyze(&self, embedding: &[f32]) -> Result<QualityReport> {
        let mut issues = Vec::new();
        let mut quality_score = 1.0f32;

        // Check for invalid values
        let (has_nan, has_inf) = self.check_invalid_values(embedding);
        if has_nan || has_inf {
            issues.push(QualityIssue {
                severity: IssueSeverity::Critical,
                category: IssueCategory::InvalidValues,
                message: format!(
                    "Embedding contains invalid values (NaN: {}, Inf: {})",
                    has_nan, has_inf
                ),
                affected_dimensions: Some(
                    embedding
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| v.is_nan() || v.is_infinite())
                        .map(|(i, _)| i)
                        .collect(),
                ),
                suggestion: "Check your embedding model for numerical instability".to_string(),
            });
            quality_score -= 0.5;
        }

        // Compute distribution analysis
        let distribution = self.analyze_distribution(embedding);

        // Check norm
        if distribution.norm < self.config.low_norm_threshold {
            issues.push(QualityIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::LowNorm,
                message: format!(
                    "Embedding has unusually low norm ({:.4})",
                    distribution.norm
                ),
                affected_dimensions: None,
                suggestion: "The embedding may be collapsed. Try different input or model.".to_string(),
            });
            quality_score -= 0.2;
        } else if distribution.norm > self.config.high_norm_threshold {
            issues.push(QualityIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::HighNorm,
                message: format!(
                    "Embedding has unusually high norm ({:.4})",
                    distribution.norm
                ),
                affected_dimensions: None,
                suggestion: "Consider normalizing embeddings before storage.".to_string(),
            });
            quality_score -= 0.1;
        }

        // Check for collapsed embeddings
        if distribution.std < self.config.zero_variance_threshold {
            issues.push(QualityIssue {
                severity: IssueSeverity::Critical,
                category: IssueCategory::Collapsed,
                message: "Embedding appears to be collapsed (near-constant values)".to_string(),
                affected_dimensions: None,
                suggestion: "The model may be producing degenerate embeddings. Check model output.".to_string(),
            });
            quality_score -= 0.4;
        }

        // Check entropy
        if distribution.entropy < self.config.min_entropy {
            issues.push(QualityIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::LowEntropy,
                message: format!(
                    "Low information content (entropy: {:.3})",
                    distribution.entropy
                ),
                affected_dimensions: None,
                suggestion: "Embedding may not be capturing diverse features.".to_string(),
            });
            quality_score -= 0.15;
        }

        // Compute dimension statistics
        let dimension_stats = self.analyze_dimensions(embedding);

        // Check for low variance dimensions
        let low_var_ratio = dimension_stats.zero_variance_dims as f32 / dimension_stats.total as f32;
        if low_var_ratio > 0.5 {
            issues.push(QualityIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::LowVariance,
                message: format!(
                    "{:.1}% of dimensions have zero variance",
                    low_var_ratio * 100.0
                ),
                affected_dimensions: None,
                suggestion: "Many dimensions are not being used. Consider dimensionality reduction.".to_string(),
            });
            quality_score -= 0.1;
        }

        // Check skewness
        if distribution.skewness.abs() > 2.0 {
            issues.push(QualityIssue {
                severity: IssueSeverity::Info,
                category: IssueCategory::Skewed,
                message: format!("Distribution is highly skewed ({:.2})", distribution.skewness),
                affected_dimensions: None,
                suggestion: "This may be normal for your data type.".to_string(),
            });
        }

        // Clamp quality score
        quality_score = quality_score.clamp(0.0, 1.0);

        // Determine grade
        let grade = if quality_score >= 0.9 {
            "A"
        } else if quality_score >= 0.8 {
            "B"
        } else if quality_score >= 0.7 {
            "C"
        } else if quality_score >= 0.5 {
            "D"
        } else {
            "F"
        }
        .to_string();

        // Generate recommendations
        let recommendations = self.generate_recommendations(&issues, &distribution);

        Ok(QualityReport {
            quality_score,
            grade,
            issues,
            dimension_stats,
            distribution,
            recommendations,
        })
    }

    /// Check for NaN and Inf values
    fn check_invalid_values(&self, embedding: &[f32]) -> (bool, bool) {
        let has_nan = embedding.iter().any(|v| v.is_nan());
        let has_inf = embedding.iter().any(|v| v.is_infinite());
        (has_nan, has_inf)
    }

    /// Analyze distribution of embedding values
    fn analyze_distribution(&self, embedding: &[f32]) -> DistributionAnalysis {
        let n = embedding.len() as f32;

        // Basic stats
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let min = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = embedding.iter().sum();
        let mean = sum / n;

        // Median
        let mut sorted = embedding.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = if embedding.len().is_multiple_of(2) {
            (sorted[embedding.len() / 2 - 1] + sorted[embedding.len() / 2]) / 2.0
        } else {
            sorted[embedding.len() / 2]
        };

        // Standard deviation
        let variance = embedding.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        // Skewness
        let skewness = if std > 0.0 {
            embedding
                .iter()
                .map(|x| ((x - mean) / std).powi(3))
                .sum::<f32>()
                / n
        } else {
            0.0
        };

        // Kurtosis
        let kurtosis = if std > 0.0 {
            embedding
                .iter()
                .map(|x| ((x - mean) / std).powi(4))
                .sum::<f32>()
                / n
                - 3.0
        } else {
            0.0
        };

        // Entropy (discretize to buckets)
        let entropy = self.compute_entropy(embedding);

        // Sparsity
        let near_zero_count = embedding
            .iter()
            .filter(|x| x.abs() < self.config.zero_variance_threshold)
            .count();
        let sparsity = near_zero_count as f32 / n;

        DistributionAnalysis {
            norm,
            min,
            max,
            mean,
            median,
            std,
            skewness,
            kurtosis,
            entropy,
            sparsity,
        }
    }

    /// Compute entropy of embedding values
    fn compute_entropy(&self, embedding: &[f32]) -> f32 {
        // Discretize into 100 buckets
        let min = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        if range <= 0.0 {
            return 0.0;
        }

        let num_buckets = 100;
        let mut counts = vec![0usize; num_buckets];

        for &val in embedding {
            let bucket = ((val - min) / range * (num_buckets - 1) as f32) as usize;
            counts[bucket.min(num_buckets - 1)] += 1;
        }

        let n = embedding.len() as f32;
        let mut entropy = 0.0;

        for count in counts {
            if count > 0 {
                let p = count as f32 / n;
                entropy -= p * p.log2();
            }
        }

        // Normalize to [0, 1]
        entropy / (num_buckets as f32).log2()
    }

    /// Analyze individual dimensions
    fn analyze_dimensions(&self, embedding: &[f32]) -> DimensionStats {
        let total = embedding.len();
        let mean = embedding.iter().sum::<f32>() / total as f32;
        let variance = embedding.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / total as f32;
        let std = variance.sqrt();

        // Find zero variance dimensions
        let zero_variance_dims = embedding
            .iter()
            .filter(|v| v.abs() < self.config.zero_variance_threshold)
            .count();

        let active_dims = total - zero_variance_dims;

        // Find top dimensions by absolute value
        let mut indexed: Vec<(usize, f32)> = embedding
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top_dimensions: Vec<(usize, f32)> = indexed.iter().take(10).cloned().collect();

        // High variance dims (deviation from mean)
        let mut variance_indexed: Vec<(usize, f32)> = embedding
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, (v - mean).abs()))
            .collect();
        variance_indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let high_variance_dims: Vec<(usize, f32)> = variance_indexed.iter().take(10).cloned().collect();

        DimensionStats {
            total,
            zero_variance_dims,
            active_dims,
            top_dimensions,
            high_variance_dims,
            mean,
            std,
        }
    }

    /// Generate recommendations based on issues
    fn generate_recommendations(
        &self,
        issues: &[QualityIssue],
        distribution: &DistributionAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        let has_critical = issues.iter().any(|i| i.severity == IssueSeverity::Critical);
        let has_low_norm = issues.iter().any(|i| i.category == IssueCategory::LowNorm);
        let has_high_norm = issues.iter().any(|i| i.category == IssueCategory::HighNorm);

        if has_critical {
            recommendations.push("Critical issues detected. Review your embedding pipeline.".to_string());
        }

        if has_low_norm {
            recommendations.push("Consider checking if input text is empty or very short.".to_string());
        }

        if has_high_norm {
            recommendations.push("Apply L2 normalization before storing.".to_string());
        }

        if distribution.sparsity > 0.5 {
            recommendations.push("High sparsity detected. Consider using sparse vector storage.".to_string());
        }

        if issues.is_empty() {
            recommendations.push("Embedding quality looks good!".to_string());
        }

        recommendations
    }

    /// Detect anomalies in a batch of embeddings
    pub fn detect_anomalies(&self, embeddings: &[Vec<f32>]) -> Result<Vec<AnomalyResult>> {
        if embeddings.is_empty() {
            return Ok(Vec::new());
        }

        // Compute statistics across all embeddings
        let norms: Vec<f32> = embeddings
            .iter()
            .map(|e| e.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();

        let mean_norm = norms.iter().sum::<f32>() / norms.len() as f32;
        let std_norm = (norms.iter().map(|n| (n - mean_norm).powi(2)).sum::<f32>()
            / norms.len() as f32)
            .sqrt();

        // Detect outliers using Z-score
        let mut anomalies = Vec::new();

        for (idx, (embedding, norm)) in embeddings.iter().zip(&norms).enumerate() {
            let z_score = if std_norm > 0.0 {
                (norm - mean_norm) / std_norm
            } else {
                0.0
            };

            if z_score.abs() > self.config.anomaly_zscore_threshold {
                let report = self.analyze(embedding)?;

                anomalies.push(AnomalyResult {
                    index: idx,
                    z_score,
                    norm: *norm,
                    anomaly_type: if z_score > 0.0 {
                        AnomalyType::HighNorm
                    } else {
                        AnomalyType::LowNorm
                    },
                    quality_report: report,
                });
            }
        }

        // Sort by absolute Z-score
        anomalies.sort_by(|a, b| b.z_score.abs().total_cmp(&a.z_score.abs()));

        Ok(anomalies)
    }

    /// Set baseline for drift detection
    pub fn set_baseline(&mut self, embeddings: &[Vec<f32>]) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }

        let dimension = embeddings[0].len();
        let n = embeddings.len() as f32;

        // Compute norms
        let norms: Vec<f32> = embeddings
            .iter()
            .map(|e| e.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();

        let mean_norm = norms.iter().sum::<f32>() / n;
        let std_norm = (norms.iter().map(|n| (n - mean_norm).powi(2)).sum::<f32>() / n).sqrt();

        // Compute per-dimension statistics
        let mut mean_per_dim = vec![0.0f32; dimension];
        let mut std_per_dim = vec![0.0f32; dimension];

        for emb in embeddings {
            for (i, &val) in emb.iter().enumerate() {
                mean_per_dim[i] += val;
            }
        }

        for mean in &mut mean_per_dim {
            *mean /= n;
        }

        for emb in embeddings {
            for (i, &val) in emb.iter().enumerate() {
                std_per_dim[i] += (val - mean_per_dim[i]).powi(2);
            }
        }

        for std in &mut std_per_dim {
            *std = (*std / n).sqrt();
        }

        self.baseline = Some(BaselineStats {
            dimension,
            mean_norm,
            std_norm,
            mean_per_dim,
            std_per_dim,
            sample_count: embeddings.len(),
        });

        Ok(())
    }

    /// Check for drift compared to baseline
    pub fn check_drift(&mut self, embeddings: &[Vec<f32>]) -> Result<DriftReport> {
        let baseline = self.baseline.as_ref().ok_or_else(|| {
            anyhow::anyhow!("No baseline set. Call set_baseline first.")
        })?;

        if embeddings.is_empty() {
            return Ok(DriftReport {
                has_drift: false,
                norm_drift: 0.0,
                dimension_drifts: Vec::new(),
                overall_drift_score: 0.0,
                details: "No embeddings to check".to_string(),
            });
        }

        let n = embeddings.len() as f32;

        // Compute current norms
        let norms: Vec<f32> = embeddings
            .iter()
            .map(|e| e.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();

        let current_mean_norm = norms.iter().sum::<f32>() / n;

        // Norm drift
        let norm_drift = (current_mean_norm - baseline.mean_norm).abs()
            / (baseline.std_norm + 1e-10);

        // Per-dimension drift
        let mut current_mean_per_dim = vec![0.0f32; baseline.dimension];
        for emb in embeddings {
            for (i, &val) in emb.iter().enumerate() {
                if i < baseline.dimension {
                    current_mean_per_dim[i] += val;
                }
            }
        }
        for mean in &mut current_mean_per_dim {
            *mean /= n;
        }

        let mut dimension_drifts = Vec::new();
        for (i, (&current, (&baseline_mean, &baseline_std))) in current_mean_per_dim
            .iter()
            .zip(baseline.mean_per_dim.iter().zip(&baseline.std_per_dim))
            .enumerate()
        {
            let drift = (current - baseline_mean).abs() / (baseline_std + 1e-10);
            if drift > self.config.drift_threshold {
                dimension_drifts.push((i, drift));
            }
        }

        dimension_drifts.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Overall drift score
        let overall_drift_score = norm_drift
            + dimension_drifts.iter().map(|(_, d)| d).sum::<f32>()
                / (dimension_drifts.len() + 1) as f32;

        let has_drift = overall_drift_score > self.config.drift_threshold;

        // Record in history
        if self.config.monitor_drift {
            self.history.push(HistoryEntry {
                timestamp: chrono::Utc::now().timestamp(),
                mean_norm: current_mean_norm,
                mean_per_dim: current_mean_per_dim.clone(),
                std_per_dim: vec![0.0; baseline.dimension], // Simplified
            });

            // Trim history
            if self.history.len() > self.config.drift_window_size {
                self.history.remove(0);
            }
        }

        let details = if has_drift {
            format!(
                "Drift detected: norm drift = {:.3}, {} dimensions drifted",
                norm_drift,
                dimension_drifts.len()
            )
        } else {
            "No significant drift detected".to_string()
        };

        Ok(DriftReport {
            has_drift,
            norm_drift,
            dimension_drifts,
            overall_drift_score,
            details,
        })
    }

    /// Compare two embeddings
    pub fn compare(&self, emb1: &[f32], emb2: &[f32]) -> Result<ComparisonReport> {
        if emb1.len() != emb2.len() {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: {} vs {}",
                emb1.len(),
                emb2.len()
            ));
        }

        let report1 = self.analyze(emb1)?;
        let report2 = self.analyze(emb2)?;

        // Cosine similarity
        let dot: f32 = emb1.iter().zip(emb2).map(|(a, b)| a * b).sum();
        let norm1 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine_similarity = dot / (norm1 * norm2 + 1e-10);

        // L2 distance
        let l2_distance = emb1
            .iter()
            .zip(emb2)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Per-dimension differences
        let mut dimension_diffs: Vec<(usize, f32)> = emb1
            .iter()
            .zip(emb2)
            .enumerate()
            .map(|(i, (a, b))| (i, (a - b).abs()))
            .collect();
        dimension_diffs.sort_by(|a, b| b.1.total_cmp(&a.1));

        Ok(ComparisonReport {
            cosine_similarity,
            l2_distance,
            norm_difference: (norm1 - norm2).abs(),
            quality_difference: (report1.quality_score - report2.quality_score).abs(),
            top_differing_dimensions: dimension_diffs.into_iter().take(10).collect(),
            report1,
            report2,
        })
    }

    /// Get drift history
    pub fn drift_history(&self) -> Vec<(i64, f32)> {
        self.history
            .iter()
            .map(|h| (h.timestamp, h.mean_norm))
            .collect()
    }
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Index in the batch
    pub index: usize,

    /// Z-score
    pub z_score: f32,

    /// Embedding norm
    pub norm: f32,

    /// Type of anomaly
    pub anomaly_type: AnomalyType,

    /// Quality report for the anomalous embedding
    pub quality_report: QualityReport,
}

/// Types of anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    HighNorm,
    LowNorm,
    Collapsed,
    InvalidValues,
}

/// Drift detection report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// Whether significant drift was detected
    pub has_drift: bool,

    /// Norm drift (Z-score)
    pub norm_drift: f32,

    /// Dimensions with significant drift: (dim_index, drift_score)
    pub dimension_drifts: Vec<(usize, f32)>,

    /// Overall drift score
    pub overall_drift_score: f32,

    /// Human-readable details
    pub details: String,
}

/// Comparison report between two embeddings
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Cosine similarity
    pub cosine_similarity: f32,

    /// L2 distance
    pub l2_distance: f32,

    /// Absolute norm difference
    pub norm_difference: f32,

    /// Quality score difference
    pub quality_difference: f32,

    /// Top dimensions with largest differences
    pub top_differing_dimensions: Vec<(usize, f32)>,

    /// Quality report for first embedding
    pub report1: QualityReport,

    /// Quality report for second embedding
    pub report2: QualityReport,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_embedding(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::rng();
        (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    fn test_quality_analysis_good_embedding() {
        let debugger = EmbeddingDebugger::new(DebugConfig::default());
        let embedding = generate_random_embedding(128);

        let report = debugger.analyze(&embedding).unwrap();

        assert!(report.quality_score > 0.5);
        println!("Quality: {}, Grade: {}", report.quality_score, report.grade);
    }

    #[test]
    fn test_quality_analysis_collapsed() {
        let debugger = EmbeddingDebugger::new(DebugConfig::default());

        // Collapsed embedding (all same value)
        let embedding = vec![0.5; 128];

        let report = debugger.analyze(&embedding).unwrap();

        assert!(report.quality_score < 0.7);
        assert!(report
            .issues
            .iter()
            .any(|i| i.category == IssueCategory::LowEntropy));
    }

    #[test]
    fn test_quality_analysis_nan() {
        let debugger = EmbeddingDebugger::new(DebugConfig::default());

        let mut embedding = generate_random_embedding(128);
        embedding[10] = f32::NAN;

        let report = debugger.analyze(&embedding).unwrap();

        assert!(report
            .issues
            .iter()
            .any(|i| i.category == IssueCategory::InvalidValues));
        assert!(report.quality_score < 0.6);
    }

    #[test]
    fn test_anomaly_detection() {
        let debugger = EmbeddingDebugger::new(DebugConfig::default());

        // Generate normal embeddings
        let mut embeddings: Vec<Vec<f32>> = (0..50)
            .map(|_| generate_random_embedding(128))
            .collect();

        // Add an anomaly (very low norm)
        embeddings.push(vec![0.001; 128]);

        // Add an anomaly (very high norm)
        embeddings.push((0..128).map(|_| 50.0).collect());

        let anomalies = debugger.detect_anomalies(&embeddings).unwrap();

        assert!(anomalies.len() >= 1);
    }

    #[test]
    fn test_drift_detection() {
        let mut debugger = EmbeddingDebugger::new(DebugConfig::default());

        // Set baseline
        let baseline: Vec<Vec<f32>> = (0..100)
            .map(|_| generate_random_embedding(128))
            .collect();
        debugger.set_baseline(&baseline).unwrap();

        // Check similar embeddings (no drift)
        let similar: Vec<Vec<f32>> = (0..50)
            .map(|_| generate_random_embedding(128))
            .collect();
        let report = debugger.check_drift(&similar).unwrap();

        println!("Drift score: {}", report.overall_drift_score);

        // Check drifted embeddings
        let drifted: Vec<Vec<f32>> = (0..50)
            .map(|_| {
                (0..128).map(|_| 10.0).collect() // All 10s - very different
            })
            .collect();
        let report2 = debugger.check_drift(&drifted).unwrap();

        assert!(report2.overall_drift_score > report.overall_drift_score);
    }

    #[test]
    fn test_comparison() {
        let debugger = EmbeddingDebugger::new(DebugConfig::default());

        let emb1 = generate_random_embedding(128);
        let emb2 = generate_random_embedding(128);

        let report = debugger.compare(&emb1, &emb2).unwrap();

        assert!(report.cosine_similarity >= -1.0 && report.cosine_similarity <= 1.0);
        assert!(report.l2_distance >= 0.0);
    }

    #[test]
    fn test_distribution_analysis() {
        let debugger = EmbeddingDebugger::new(DebugConfig::default());
        let embedding = generate_random_embedding(128);

        let report = debugger.analyze(&embedding).unwrap();
        let dist = &report.distribution;

        assert!(dist.norm > 0.0);
        assert!(dist.entropy >= 0.0 && dist.entropy <= 1.0);
        assert!(dist.sparsity >= 0.0 && dist.sparsity <= 1.0);
    }
}
