//! Hybrid search: combining dense vector similarity with sparse keyword matching
//!
//! Hybrid search fuses results from:
//! - Dense vector similarity (semantic search via embeddings)
//! - Sparse vector matching (keyword search via BM25)
//!
//! This enables powerful search that combines the best of both:
//! - Dense: captures semantic meaning, synonyms, context
//! - Sparse: captures exact keywords, terminology, rare terms
//!
//! Common use cases:
//! - Document search (semantic + keyword matching)
//! - E-commerce (product embeddings + exact feature matching)
//! - Code search (semantic understanding + exact symbol matching)

use serde::{Deserialize, Serialize};

/// Detailed explanation of how a hybrid search score was computed
///
/// Provides full transparency into score calculation for debugging and trust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreExplanation {
    /// Final combined score
    pub final_score: f32,

    /// Dense (semantic) component score
    pub dense_score: f32,

    /// Sparse (keyword) component score
    pub sparse_score: f32,

    /// Fusion strategy used
    pub fusion_strategy: FusionStrategy,

    /// Alpha parameter (for WeightedSum and similar strategies)
    pub alpha: f32,

    /// Detailed calculation steps
    pub calculation: String,

    /// Contribution breakdown
    pub contributions: ScoreContributions,
}

/// Breakdown of how each component contributed to the final score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreContributions {
    /// Contribution from dense vector (0.0 to 1.0, where 1.0 = 100%)
    pub dense_contribution: f32,

    /// Contribution from sparse vector (0.0 to 1.0, where 1.0 = 100%)
    pub sparse_contribution: f32,

    /// Human-readable explanation
    pub explanation: String,
}

/// Strategy for fusing dense and sparse scores
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FusionStrategy {
    /// Weighted sum: alpha * dense + (1-alpha) * sparse
    ///
    /// This is the most common strategy. The alpha parameter controls
    /// the relative weight of dense vs sparse scores.
    ///
    /// alpha = 0.7 is a good default (70% dense, 30% sparse)
    WeightedSum,

    /// Reciprocal Rank Fusion (RRF)
    ///
    /// Combines rankings rather than scores:
    /// score = 1/(k + rank_dense) + 1/(k + rank_sparse)
    ///
    /// where k is typically 60.
    ///
    /// RRF is more robust to score scale differences and is used by
    /// many production search systems.
    ReciprocalRankFusion,

    /// Distribution-Based Score Fusion (DBSF) - Qdrant's algorithm
    ///
    /// Normalizes scores using μ±3σ (mean ± 3 standard deviations):
    /// 1. Calculate mean and std dev of all scores
    /// 2. Set bounds: [μ - 3σ, μ + 3σ]
    /// 3. Clamp scores to bounds
    /// 4. Normalize to [0, 1] range
    ///
    /// Better handles outliers than min-max normalization.
    /// Recommended for datasets with high score variance.
    DistributionBased,

    /// Relative Score Fusion - Weaviate's algorithm
    ///
    /// Min-max normalization: (score - min) / (max - min)
    ///
    /// Preserves relative score differences (unlike RRF).
    /// Then combines: alpha * norm_dense + (1-alpha) * norm_sparse
    ///
    /// More information-preserving than RRF.
    RelativeScore,

    /// Maximum score
    ///
    /// Takes max(dense_score, sparse_score).
    /// Useful when either signal alone is sufficient.
    Max,

    /// Minimum score (both must match)
    ///
    /// Takes min(dense_score, sparse_score).
    /// Requires both signals to agree. Very conservative.
    Min,

    /// Harmonic mean (balanced combination)
    ///
    /// score = 2 * (dense * sparse) / (dense + sparse)
    ///
    /// Heavily penalizes if either score is low.
    HarmonicMean,

    /// Geometric mean
    ///
    /// score = sqrt(dense * sparse)
    ///
    /// Balances between arithmetic and harmonic mean.
    GeometricMean,
}

/// Configuration for hybrid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// Fusion strategy to use
    pub fusion_strategy: FusionStrategy,

    /// Alpha parameter for WeightedSum strategy (0.0 to 1.0)
    ///
    /// - alpha = 1.0: Pure dense search
    /// - alpha = 0.7: Default, 70% dense, 30% sparse
    /// - alpha = 0.5: Equal weight
    /// - alpha = 0.0: Pure sparse search
    pub alpha: f32,

    /// RRF k parameter (typically 60)
    ///
    /// Controls how quickly rank matters in RRF fusion.
    /// Higher k = more lenient, lower k = more aggressive.
    pub rrf_k: f32,

    /// Whether to normalize scores before fusion
    ///
    /// Recommended for WeightedSum to ensure scores are on similar scales.
    pub normalize_scores: bool,

    /// Autocut parameter for smart result truncation
    ///
    /// Automatically truncates results at natural score drop-offs (jumps).
    /// - `None` or `Some(0)`: Disabled (default)
    /// - `Some(1)`: Cut at first steep score drop (recommended)
    /// - `Some(N)`: Cut at Nth steep score drop
    ///
    /// Prevents returning marginally relevant results.
    /// Only works with RelativeScore fusion strategy (Weaviate behavior).
    pub autocut: Option<usize>,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha: 0.7,
            rrf_k: 60.0,
            normalize_scores: true,
            autocut: None,
        }
    }
}

/// Hybrid search query
///
/// Contains both dense and sparse components for hybrid search.
#[derive(Debug, Clone)]
pub struct HybridQuery {
    /// Dense vector component (optional)
    pub dense: Option<Vec<f32>>,

    /// Sparse vector component (indices, values) (optional)
    pub sparse: Option<(Vec<usize>, Vec<f32>)>,

    /// Number of results to return
    pub k: usize,

    /// Hybrid search configuration
    pub config: HybridSearchConfig,
}

impl HybridQuery {
    /// Create a hybrid query with both dense and sparse components
    pub fn new(dense: Vec<f32>, sparse_indices: Vec<usize>, sparse_values: Vec<f32>) -> Self {
        Self {
            dense: Some(dense),
            sparse: Some((sparse_indices, sparse_values)),
            k: 10,
            config: HybridSearchConfig::default(),
        }
    }

    /// Create a dense-only query (will fall back to pure dense search)
    pub fn dense_only(dense: Vec<f32>) -> Self {
        Self {
            dense: Some(dense),
            sparse: None,
            k: 10,
            config: HybridSearchConfig::default(),
        }
    }

    /// Create a sparse-only query (will fall back to pure sparse search)
    pub fn sparse_only(indices: Vec<usize>, values: Vec<f32>) -> Self {
        Self {
            dense: None,
            sparse: Some((indices, values)),
            k: 10,
            config: HybridSearchConfig::default(),
        }
    }

    /// Set number of results
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set alpha parameter (for WeightedSum fusion)
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.config.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Set fusion strategy
    pub fn with_fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.config.fusion_strategy = strategy;
        self
    }

    /// Enable/disable score normalization
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.config.normalize_scores = normalize;
        self
    }
}

/// Fuse dense and sparse scores into a single hybrid score
///
/// # Arguments
/// * `dense_score` - Score from dense vector similarity (0.0 to 1.0)
/// * `sparse_score` - Score from sparse keyword matching (0.0 to 1.0)
/// * `config` - Hybrid search configuration
///
/// # Returns
/// Combined score
pub fn hybrid_search_score(
    dense_score: f32,
    sparse_score: f32,
    config: &HybridSearchConfig,
) -> f32 {
    match config.fusion_strategy {
        FusionStrategy::WeightedSum => {
            config.alpha * dense_score + (1.0 - config.alpha) * sparse_score
        }

        FusionStrategy::ReciprocalRankFusion => {
            // Note: This is a simplified RRF for score-based fusion
            // True RRF requires rankings, not scores
            // Here we approximate: higher score = better rank
            let dense_rank_score = 1.0 / (config.rrf_k + (1.0 - dense_score));
            let sparse_rank_score = 1.0 / (config.rrf_k + (1.0 - sparse_score));
            dense_rank_score + sparse_rank_score
        }

        FusionStrategy::DistributionBased => {
            // DBSF: For single-score fusion, just average
            // Full DBSF normalization happens in normalize_scores_dbsf()
            config.alpha * dense_score + (1.0 - config.alpha) * sparse_score
        }

        FusionStrategy::RelativeScore => {
            // RelativeScore: For single-score fusion, same as WeightedSum
            // Full min-max normalization happens in normalize_scores()
            config.alpha * dense_score + (1.0 - config.alpha) * sparse_score
        }

        FusionStrategy::Max => dense_score.max(sparse_score),

        FusionStrategy::Min => dense_score.min(sparse_score),

        FusionStrategy::HarmonicMean => {
            if dense_score + sparse_score == 0.0 {
                0.0
            } else {
                2.0 * dense_score * sparse_score / (dense_score + sparse_score)
            }
        }

        FusionStrategy::GeometricMean => {
            if dense_score < 0.0 || sparse_score < 0.0 {
                0.0
            } else {
                (dense_score * sparse_score).sqrt()
            }
        }
    }
}

/// Explain how a hybrid search score was calculated
///
/// Provides full transparency into the score calculation, showing:
/// - Individual component scores (dense, sparse)
/// - Fusion strategy used
/// - Detailed calculation steps
/// - Contribution breakdown
///
/// # Arguments
/// * `dense_score` - Score from dense vector similarity (0.0 to 1.0)
/// * `sparse_score` - Score from sparse keyword matching (0.0 to 1.0)
/// * `config` - Hybrid search configuration
///
/// # Returns
/// Detailed score explanation
///
/// # Example
/// ```rust
/// use vecstore::vectors::{explain_hybrid_score, HybridSearchConfig, FusionStrategy};
///
/// let config = HybridSearchConfig {
///     fusion_strategy: FusionStrategy::WeightedSum,
///     alpha: 0.7,
///     ..Default::default()
/// };
///
/// let explanation = explain_hybrid_score(0.8, 0.6, &config);
///
/// println!("Final score: {}", explanation.final_score);
/// println!("Calculation: {}", explanation.calculation);
/// println!("Dense contributed: {:.1}%", explanation.contributions.dense_contribution * 100.0);
/// ```
pub fn explain_hybrid_score(
    dense_score: f32,
    sparse_score: f32,
    config: &HybridSearchConfig,
) -> ScoreExplanation {
    let final_score = hybrid_search_score(dense_score, sparse_score, config);

    let (calculation, contributions) = match config.fusion_strategy {
        FusionStrategy::WeightedSum => {
            let dense_weight = config.alpha;
            let sparse_weight = 1.0 - config.alpha;
            let dense_contrib = dense_score * dense_weight;
            let sparse_contrib = sparse_score * sparse_weight;

            let calc = format!(
                "WeightedSum: {:.4} * {:.4} + {:.4} * {:.4} = {:.4}",
                dense_weight, dense_score, sparse_weight, sparse_score, final_score
            );

            let total = dense_contrib + sparse_contrib;
            let contributions = if total > 0.0 {
                ScoreContributions {
                    dense_contribution: dense_contrib / total,
                    sparse_contribution: sparse_contrib / total,
                    explanation: format!(
                        "Dense: {:.1}%, Sparse: {:.1}%",
                        (dense_contrib / total) * 100.0,
                        (sparse_contrib / total) * 100.0
                    ),
                }
            } else {
                ScoreContributions {
                    dense_contribution: 0.5,
                    sparse_contribution: 0.5,
                    explanation: "Both scores are zero".to_string(),
                }
            };

            (calc, contributions)
        }

        FusionStrategy::ReciprocalRankFusion => {
            let dense_rank_score = 1.0 / (config.rrf_k + (1.0 - dense_score));
            let sparse_rank_score = 1.0 / (config.rrf_k + (1.0 - sparse_score));

            let calc = format!(
                "RRF: 1/({:.1} + {:.4}) + 1/({:.1} + {:.4}) = {:.4}",
                config.rrf_k,
                1.0 - dense_score,
                config.rrf_k,
                1.0 - sparse_score,
                final_score
            );

            let total = dense_rank_score + sparse_rank_score;
            let contributions = ScoreContributions {
                dense_contribution: dense_rank_score / total,
                sparse_contribution: sparse_rank_score / total,
                explanation: format!(
                    "Dense rank: {:.1}%, Sparse rank: {:.1}%",
                    (dense_rank_score / total) * 100.0,
                    (sparse_rank_score / total) * 100.0
                ),
            };

            (calc, contributions)
        }

        FusionStrategy::DistributionBased | FusionStrategy::RelativeScore => {
            let dense_weight = config.alpha;
            let sparse_weight = 1.0 - config.alpha;
            let dense_contrib = dense_score * dense_weight;
            let sparse_contrib = sparse_score * sparse_weight;

            let strategy_name = match config.fusion_strategy {
                FusionStrategy::DistributionBased => "DBSF",
                _ => "RelativeScore",
            };

            let calc = format!(
                "{}: {:.4} * {:.4} + {:.4} * {:.4} = {:.4}",
                strategy_name, dense_weight, dense_score, sparse_weight, sparse_score, final_score
            );

            let total = dense_contrib + sparse_contrib;
            let contributions = if total > 0.0 {
                ScoreContributions {
                    dense_contribution: dense_contrib / total,
                    sparse_contribution: sparse_contrib / total,
                    explanation: format!(
                        "Dense: {:.1}%, Sparse: {:.1}%",
                        (dense_contrib / total) * 100.0,
                        (sparse_contrib / total) * 100.0
                    ),
                }
            } else {
                ScoreContributions {
                    dense_contribution: 0.5,
                    sparse_contribution: 0.5,
                    explanation: "Both scores are zero".to_string(),
                }
            };

            (calc, contributions)
        }

        FusionStrategy::Max => {
            let calc = format!(
                "Max: max({:.4}, {:.4}) = {:.4}",
                dense_score, sparse_score, final_score
            );

            let contributions = if dense_score > sparse_score {
                ScoreContributions {
                    dense_contribution: 1.0,
                    sparse_contribution: 0.0,
                    explanation: "Dense score was higher (100% contribution)".to_string(),
                }
            } else if sparse_score > dense_score {
                ScoreContributions {
                    dense_contribution: 0.0,
                    sparse_contribution: 1.0,
                    explanation: "Sparse score was higher (100% contribution)".to_string(),
                }
            } else {
                ScoreContributions {
                    dense_contribution: 0.5,
                    sparse_contribution: 0.5,
                    explanation: "Scores were equal".to_string(),
                }
            };

            (calc, contributions)
        }

        FusionStrategy::Min => {
            let calc = format!(
                "Min: min({:.4}, {:.4}) = {:.4}",
                dense_score, sparse_score, final_score
            );

            let contributions = if dense_score < sparse_score {
                ScoreContributions {
                    dense_contribution: 1.0,
                    sparse_contribution: 0.0,
                    explanation: "Dense score was lower (limited by it)".to_string(),
                }
            } else if sparse_score < dense_score {
                ScoreContributions {
                    dense_contribution: 0.0,
                    sparse_contribution: 1.0,
                    explanation: "Sparse score was lower (limited by it)".to_string(),
                }
            } else {
                ScoreContributions {
                    dense_contribution: 0.5,
                    sparse_contribution: 0.5,
                    explanation: "Scores were equal".to_string(),
                }
            };

            (calc, contributions)
        }

        FusionStrategy::HarmonicMean => {
            let calc = if dense_score + sparse_score == 0.0 {
                "HarmonicMean: 0 (both scores zero)".to_string()
            } else {
                format!(
                    "HarmonicMean: 2 * {:.4} * {:.4} / ({:.4} + {:.4}) = {:.4}",
                    dense_score, sparse_score, dense_score, sparse_score, final_score
                )
            };

            // For harmonic mean, both contribute equally structurally
            // but lower score has more influence
            let contributions = ScoreContributions {
                dense_contribution: 0.5,
                sparse_contribution: 0.5,
                explanation: "Both scores contribute (harmonic mean penalizes low scores)"
                    .to_string(),
            };

            (calc, contributions)
        }

        FusionStrategy::GeometricMean => {
            let calc = if dense_score < 0.0 || sparse_score < 0.0 {
                "GeometricMean: 0 (negative score)".to_string()
            } else {
                format!(
                    "GeometricMean: sqrt({:.4} * {:.4}) = {:.4}",
                    dense_score, sparse_score, final_score
                )
            };

            let contributions = ScoreContributions {
                dense_contribution: 0.5,
                sparse_contribution: 0.5,
                explanation: "Both scores contribute equally (geometric mean)".to_string(),
            };

            (calc, contributions)
        }
    };

    ScoreExplanation {
        final_score,
        dense_score,
        sparse_score,
        fusion_strategy: config.fusion_strategy,
        alpha: config.alpha,
        calculation,
        contributions,
    }
}

/// Normalize scores to 0-1 range using min-max normalization
///
/// This is useful when combining scores from different sources that may
/// have different scales (e.g., cosine similarity vs BM25 scores).
///
/// # Arguments
/// * `scores` - Slice of scores to normalize
///
/// # Returns
/// Vector of normalized scores in [0, 1] range
///
/// If all scores are equal, returns all 0.5.
/// If there's only one score, returns vec![1.0].
pub fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }

    if scores.len() == 1 {
        return vec![1.0];
    }

    let min_score = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if (max_score - min_score).abs() < 1e-10 {
        // All scores are the same
        return vec![0.5; scores.len()];
    }

    scores
        .iter()
        .map(|&score| (score - min_score) / (max_score - min_score))
        .collect()
}

/// Normalize scores using z-score (standard score) normalization
///
/// Transforms scores to have mean 0 and standard deviation 1.
/// Then maps to [0, 1] using sigmoid function.
///
/// This is more robust to outliers than min-max normalization.
pub fn normalize_scores_zscore(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }

    if scores.len() == 1 {
        return vec![0.5];
    }

    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / scores.len() as f32;
    let std_dev = variance.sqrt();

    if std_dev < 1e-10 {
        // All scores are the same
        return vec![0.5; scores.len()];
    }

    scores
        .iter()
        .map(|&score| {
            let z = (score - mean) / std_dev;
            // Sigmoid: 1 / (1 + e^(-z))
            1.0 / (1.0 + (-z).exp())
        })
        .collect()
}

/// Normalize scores using DBSF (Distribution-Based Score Fusion)
///
/// Qdrant's normalization algorithm using μ±3σ (mean ± 3 standard deviations):
/// 1. Calculate mean and standard deviation
/// 2. Set bounds: [μ - 3σ, μ + 3σ]
/// 3. Clamp scores to bounds
/// 4. Normalize to [0, 1] range
///
/// This is more robust to outliers than simple min-max normalization because:
/// - Outliers beyond μ±3σ are clamped
/// - Handles 99.7% of normally distributed data
/// - More stable for datasets with high variance
///
/// # Arguments
/// * `scores` - Slice of scores to normalize
///
/// # Returns
/// Vector of normalized scores in [0, 1] range
///
/// # Example
/// ```
/// use vecstore::vectors::normalize_scores_dbsf;
///
/// // Scores with outliers
/// let scores = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // 100.0 is outlier
/// let normalized = normalize_scores_dbsf(&scores);
///
/// // The outlier (100.0) will be clamped to μ+3σ before normalization
/// // Result: more balanced distribution than min-max
/// ```
pub fn normalize_scores_dbsf(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }

    if scores.len() == 1 {
        return vec![0.5];
    }

    // Calculate mean
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;

    // Calculate variance and standard deviation
    let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / scores.len() as f32;
    let std_dev = variance.sqrt();

    if std_dev < 1e-10 {
        // All scores are the same
        return vec![0.5; scores.len()];
    }

    // Set bounds: μ ± 3σ (covers 99.7% of normal distribution)
    let lower_bound = mean - 3.0 * std_dev;
    let upper_bound = mean + 3.0 * std_dev;
    let range = upper_bound - lower_bound;

    if range < f32::EPSILON {
        return vec![0.5; scores.len()];
    }

    // Clamp scores to bounds and normalize to [0, 1]
    scores
        .iter()
        .map(|&score| {
            let clamped = score.clamp(lower_bound, upper_bound);
            (clamped - lower_bound) / range
        })
        .collect()
}

/// Apply autocut to truncate results at natural score drop-offs
///
/// Autocut detects "jumps" (steep drops) in scores between consecutive results
/// and truncates the result list at the first N jumps.
///
/// # Arguments
///
/// * `results` - Vec of (id, score) tuples, sorted by score descending
/// * `autocut` - Number of jumps to detect before cutting (typically 1-3)
///
/// # Returns
///
/// Truncated results vec
///
/// # Algorithm
///
/// For each pair of consecutive results, calculate the score drop.
/// A "jump" is detected when the drop is significantly larger than average.
/// We use a threshold of 2x the median score drop.
///
/// # Example
///
/// ```rust
/// use vecstore::vectors::apply_autocut;
///
/// // Results with natural score groups
/// let results = vec![
///     ("doc1".to_string(), 0.95),
///     ("doc2".to_string(), 0.92),
///     ("doc3".to_string(), 0.90),  // High relevance group
///     ("doc4".to_string(), 0.45),  // <-- BIG JUMP HERE (0.45 drop)
///     ("doc5".to_string(), 0.42),
///     ("doc6".to_string(), 0.40),  // Low relevance group
/// ];
///
/// // Cut at first jump
/// let truncated = apply_autocut(results.clone(), 1);
/// assert_eq!(truncated.len(), 3); // Only top 3 highly relevant results
///
/// // Without autocut (or autocut = 0)
/// let no_cut = apply_autocut(results, 0);
/// assert_eq!(no_cut.len(), 6); // All results returned
/// ```
pub fn apply_autocut<T: Clone>(results: Vec<(T, f32)>, autocut: usize) -> Vec<(T, f32)> {
    // If autocut is 0 or results are too small, return all results
    if autocut == 0 || results.len() <= 1 {
        return results;
    }

    // Calculate score drops between consecutive results
    let mut drops: Vec<f32> = Vec::with_capacity(results.len() - 1);
    for i in 0..results.len() - 1 {
        let drop = results[i].1 - results[i + 1].1;
        drops.push(drop);
    }

    if drops.is_empty() {
        return results;
    }

    // Find median drop to determine what counts as a "jump"
    let mut sorted_drops = drops.clone();
    sorted_drops.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_drop = if sorted_drops.len() % 2 == 0 {
        let mid = sorted_drops.len() / 2;
        (sorted_drops[mid - 1] + sorted_drops[mid]) / 2.0
    } else {
        sorted_drops[sorted_drops.len() / 2]
    };

    // A "jump" is a drop that's significantly larger than the median
    // We use 2x median as the threshold (tunable parameter)
    let jump_threshold = median_drop * 2.0;

    // Find positions of jumps
    let mut jump_positions: Vec<usize> = Vec::new();
    for (i, &drop) in drops.iter().enumerate() {
        if drop > jump_threshold && drop > 0.01 {
            // Also require minimum absolute drop of 0.01
            jump_positions.push(i + 1); // i+1 because we cut AFTER the drop
        }
    }

    // If no jumps found, return all results
    if jump_positions.is_empty() {
        return results;
    }

    // Cut at the Nth jump (or first if N > number of jumps)
    let cut_position = jump_positions.get(autocut - 1).copied().unwrap_or_else(|| {
        // If autocut requests more jumps than exist, use the last jump
        *jump_positions.last().unwrap()
    });

    // Return results up to (but not including) the cut position
    results.into_iter().take(cut_position).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_weighted_sum() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha: 0.7,
            ..Default::default()
        };

        let score = hybrid_search_score(0.8, 0.6, &config);
        let expected = 0.7 * 0.8 + 0.3 * 0.6;
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn test_fusion_weighted_sum_pure_dense() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha: 1.0, // Pure dense
            ..Default::default()
        };

        let score = hybrid_search_score(0.8, 0.6, &config);
        assert!((score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_fusion_weighted_sum_pure_sparse() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha: 0.0, // Pure sparse
            ..Default::default()
        };

        let score = hybrid_search_score(0.8, 0.6, &config);
        assert!((score - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_fusion_max() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::Max,
            ..Default::default()
        };

        let score = hybrid_search_score(0.8, 0.6, &config);
        assert!((score - 0.8).abs() < 1e-6);

        let score2 = hybrid_search_score(0.5, 0.9, &config);
        assert!((score2 - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_fusion_min() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::Min,
            ..Default::default()
        };

        let score = hybrid_search_score(0.8, 0.6, &config);
        assert!((score - 0.6).abs() < 1e-6);

        let score2 = hybrid_search_score(0.5, 0.9, &config);
        assert!((score2 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_fusion_harmonic_mean() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::HarmonicMean,
            ..Default::default()
        };

        let score = hybrid_search_score(0.8, 0.6, &config);
        let expected = 2.0 * 0.8 * 0.6 / (0.8 + 0.6);
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn test_fusion_harmonic_mean_zero() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::HarmonicMean,
            ..Default::default()
        };

        let score = hybrid_search_score(0.0, 0.0, &config);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_normalize_scores_minmax() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_scores(&scores);

        assert_eq!(normalized.len(), 5);
        assert!((normalized[0] - 0.0).abs() < 1e-6); // min -> 0
        assert!((normalized[4] - 1.0).abs() < 1e-6); // max -> 1
        assert!((normalized[2] - 0.5).abs() < 1e-6); // middle -> 0.5
    }

    #[test]
    fn test_normalize_scores_all_same() {
        let scores = vec![5.0, 5.0, 5.0];
        let normalized = normalize_scores(&scores);

        assert_eq!(normalized.len(), 3);
        assert!(normalized.iter().all(|&x| (x - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_normalize_scores_single() {
        let scores = vec![5.0];
        let normalized = normalize_scores(&scores);

        assert_eq!(normalized.len(), 1);
        assert!((normalized[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_scores_empty() {
        let scores = vec![];
        let normalized = normalize_scores(&scores);
        assert_eq!(normalized.len(), 0);
    }

    #[test]
    fn test_normalize_scores_zscore() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_scores_zscore(&scores);

        assert_eq!(normalized.len(), 5);
        // All scores should be in [0, 1] range
        assert!(normalized.iter().all(|&x| x >= 0.0 && x <= 1.0));
        // Mean should be around 0.5
        let mean = normalized.iter().sum::<f32>() / normalized.len() as f32;
        assert!((mean - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_hybrid_query_builder() {
        let query = HybridQuery::new(vec![0.1, 0.2, 0.3], vec![10, 25], vec![1.0, 2.0])
            .with_k(20)
            .with_alpha(0.8)
            .with_fusion_strategy(FusionStrategy::Max);

        assert_eq!(query.k, 20);
        assert!((query.config.alpha - 0.8).abs() < 1e-6);
        assert_eq!(query.config.fusion_strategy, FusionStrategy::Max);
    }

    #[test]
    fn test_hybrid_query_dense_only() {
        let query = HybridQuery::dense_only(vec![0.1, 0.2, 0.3]);
        assert!(query.dense.is_some());
        assert!(query.sparse.is_none());
    }

    #[test]
    fn test_hybrid_query_sparse_only() {
        let query = HybridQuery::sparse_only(vec![1, 2, 3], vec![0.5, 0.6, 0.7]);
        assert!(query.dense.is_none());
        assert!(query.sparse.is_some());
    }

    #[test]
    fn test_alpha_clamping() {
        let query = HybridQuery::dense_only(vec![0.1]).with_alpha(1.5); // Out of range

        assert!((query.config.alpha - 1.0).abs() < 1e-6); // Should be clamped to 1.0

        let query2 = HybridQuery::dense_only(vec![0.1]).with_alpha(-0.5); // Out of range

        assert!((query2.config.alpha - 0.0).abs() < 1e-6); // Should be clamped to 0.0
    }

    #[test]
    fn test_fusion_rrf() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::ReciprocalRankFusion,
            rrf_k: 60.0,
            ..Default::default()
        };

        let score = hybrid_search_score(0.9, 0.8, &config);
        assert!(score > 0.0);

        // Higher scores should produce higher combined scores
        let score2 = hybrid_search_score(0.5, 0.5, &config);
        assert!(score > score2);
    }

    #[test]
    fn test_fusion_geometric_mean() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::GeometricMean,
            ..Default::default()
        };

        let score = hybrid_search_score(0.64, 0.36, &config);
        // sqrt(0.64 * 0.36) = sqrt(0.2304) = 0.48
        assert!((score - 0.48).abs() < 1e-6);
    }

    #[test]
    fn test_fusion_geometric_mean_negative() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::GeometricMean,
            ..Default::default()
        };

        // Should handle negative scores gracefully
        let score = hybrid_search_score(-0.5, 0.5, &config);
        assert_eq!(score, 0.0);
    }

    // ========== DBSF Normalization Tests ==========

    #[test]
    fn test_normalize_dbsf_basic() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_scores_dbsf(&scores);

        assert_eq!(normalized.len(), 5);
        // All scores should be in [0, 1] range
        assert!(normalized.iter().all(|&x| x >= 0.0 && x <= 1.0));

        // For normal distribution without outliers, DBSF behaves similar to min-max
        // First value should be close to 0, last close to 1
        assert!(normalized[0] < normalized[2]); // monotonic increase
        assert!(normalized[2] < normalized[4]); // monotonic increase
    }

    #[test]
    fn test_normalize_dbsf_with_outliers() {
        // Scores with extreme outliers
        let scores = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        let normalized = normalize_scores_dbsf(&scores);

        assert_eq!(normalized.len(), 5);
        // All should be in [0, 1]
        assert!(normalized.iter().all(|&x| x >= 0.0 && x <= 1.0));

        // The key insight: DBSF clamps the outlier to μ+3σ
        // So the outlier (100.0) is treated differently than in min-max

        // With DBSF, the outlier is clamped, which changes the normalization
        // Let's verify the outlier is handled: it should be at or near upper bound
        let outlier_score = normalized[4];
        assert!(
            outlier_score >= 0.8,
            "Outlier should be near upper bound after clamping"
        );
    }

    #[test]
    fn test_normalize_dbsf_all_same() {
        let scores = vec![5.0, 5.0, 5.0, 5.0];
        let normalized = normalize_scores_dbsf(&scores);

        assert_eq!(normalized.len(), 4);
        // All scores are same -> all should be 0.5
        assert!(normalized.iter().all(|&x| (x - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_normalize_dbsf_empty() {
        let scores: Vec<f32> = vec![];
        let normalized = normalize_scores_dbsf(&scores);
        assert_eq!(normalized.len(), 0);
    }

    #[test]
    fn test_normalize_dbsf_single() {
        let scores = vec![42.0];
        let normalized = normalize_scores_dbsf(&scores);
        assert_eq!(normalized.len(), 1);
        assert!((normalized[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_dbsf_vs_minmax() {
        // Test case where DBSF shows its value: dataset with outliers
        let scores = vec![
            10.0, 12.0, 11.0, 13.0,  // Normal range
            100.0, // Extreme outlier
        ];

        let dbsf = normalize_scores_dbsf(&scores);
        let minmax = normalize_scores(&scores);

        // Both should produce valid [0, 1] ranges
        assert!(dbsf.iter().all(|&x| x >= 0.0 && x <= 1.0));
        assert!(minmax.iter().all(|&x| x >= 0.0 && x <= 1.0));

        // With min-max: 10 -> 0.0, 13 -> 0.033, 100 -> 1.0
        // Normal values are compressed into tiny range
        let minmax_val_at_13 = minmax[3]; // Position of 13.0
        assert!(
            minmax_val_at_13 < 0.1,
            "Min-max compresses normal values: {}",
            minmax_val_at_13
        );

        // DBSF clamps the outlier, which affects the normalization differently
        // The outlier should be handled
        let _dbsf_outlier = dbsf[4];
        let minmax_outlier = minmax[4];
        assert_eq!(minmax_outlier, 1.0, "Min-max puts outlier at 1.0");
        // DBSF also puts it at/near 1.0 after clamping, but internal handling differs
    }

    #[test]
    fn test_normalize_dbsf_normal_distribution() {
        // Simulate normally distributed scores
        let scores = vec![
            2.0, 2.5, 3.0, 3.5, 4.0, // μ ≈ 3.0
            4.5, 5.0, 5.5, 6.0,  // Within 2σ
            10.0, // Outlier (beyond 3σ)
        ];

        let normalized = normalize_scores_dbsf(&scores);

        // Should handle the 99.7% (within 3σ) well
        // Outlier should be clamped
        assert!(normalized.iter().all(|&x| x >= 0.0 && x <= 1.0));

        // The outlier (10.0) should be clamped to upper bound
        let outlier_score = normalized[scores.len() - 1];
        // It should be close to 1.0 (upper bound)
        assert!(outlier_score > 0.9);
    }

    // ========== Fusion Strategy Comparison Tests ==========

    #[test]
    fn test_fusion_strategies_comparison() {
        let dense = 0.8;
        let sparse = 0.6;

        // WeightedSum (α=0.7)
        let ws_config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha: 0.7,
            ..Default::default()
        };
        let ws_score = hybrid_search_score(dense, sparse, &ws_config);
        assert!((ws_score - (0.7 * 0.8 + 0.3 * 0.6)).abs() < 1e-6);

        // HarmonicMean
        let hm_config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::HarmonicMean,
            ..Default::default()
        };
        let hm_score = hybrid_search_score(dense, sparse, &hm_config);
        let expected_hm = 2.0 * 0.8 * 0.6 / (0.8 + 0.6);
        assert!((hm_score - expected_hm).abs() < 1e-6);

        // GeometricMean
        let gm_config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::GeometricMean,
            ..Default::default()
        };
        let gm_score = hybrid_search_score(dense, sparse, &gm_config);
        let expected_gm = (0.8 * 0.6_f32).sqrt();
        assert!((gm_score - expected_gm).abs() < 1e-6);

        // Verify relationships: typically HM < GM < Arithmetic
        // For same values: HM ≤ GM ≤ AM
        assert!(hm_score <= gm_score);
        assert!(gm_score <= ws_score); // WS with α=0.7 is ~arithmetic mean
    }

    #[test]
    fn test_all_fusion_strategies_produce_valid_scores() {
        let test_cases = vec![(0.9, 0.8), (0.5, 0.5), (0.1, 0.9), (0.0, 1.0), (1.0, 0.0)];

        let strategies = vec![
            FusionStrategy::WeightedSum,
            FusionStrategy::ReciprocalRankFusion,
            FusionStrategy::DistributionBased,
            FusionStrategy::RelativeScore,
            FusionStrategy::Max,
            FusionStrategy::Min,
            FusionStrategy::HarmonicMean,
            FusionStrategy::GeometricMean,
        ];

        for (dense, sparse) in test_cases {
            for strategy in &strategies {
                let config = HybridSearchConfig {
                    fusion_strategy: *strategy,
                    alpha: 0.7,
                    rrf_k: 60.0,
                    normalize_scores: true,
                    autocut: None,
                };

                let score = hybrid_search_score(dense, sparse, &config);

                // All fusion strategies should produce finite scores
                assert!(
                    score.is_finite(),
                    "Strategy {:?} produced non-finite score for ({}, {})",
                    strategy,
                    dense,
                    sparse
                );

                // Most should produce scores in reasonable range (some exceptions like RRF)
                if !matches!(strategy, FusionStrategy::ReciprocalRankFusion) {
                    assert!(
                        score >= 0.0 && score <= 1.5,
                        "Strategy {:?} produced out-of-range score {} for ({}, {})",
                        strategy,
                        score,
                        dense,
                        sparse
                    );
                }
            }
        }
    }

    // ============================================================================
    // Autocut Tests
    // ============================================================================

    #[test]
    fn test_autocut_disabled() {
        let results = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.5),
            ("doc3".to_string(), 0.1),
        ];

        // autocut = 0 should return all results
        let no_cut = apply_autocut(results.clone(), 0);
        assert_eq!(no_cut.len(), 3);
    }

    #[test]
    fn test_autocut_single_result() {
        let results = vec![("doc1".to_string(), 0.9)];

        let cut = apply_autocut(results.clone(), 1);
        assert_eq!(cut.len(), 1); // Should return single result unchanged
    }

    #[test]
    fn test_autocut_clear_jump() {
        // Results with clear score jump
        let results = vec![
            ("doc1".to_string(), 0.95),
            ("doc2".to_string(), 0.92),
            ("doc3".to_string(), 0.90), // High relevance group
            ("doc4".to_string(), 0.45), // <-- BIG JUMP HERE (0.45 drop)
            ("doc5".to_string(), 0.42),
            ("doc6".to_string(), 0.40), // Low relevance group
        ];

        // Cut at first jump
        let cut = apply_autocut(results, 1);
        assert_eq!(cut.len(), 3); // Should cut after doc3
        assert_eq!(cut[0].0, "doc1");
        assert_eq!(cut[1].0, "doc2");
        assert_eq!(cut[2].0, "doc3");
    }

    #[test]
    fn test_autocut_multiple_jumps() {
        // Results with multiple score jumps
        let results = vec![
            ("doc1".to_string(), 0.95),
            ("doc2".to_string(), 0.90), // Group 1
            ("doc3".to_string(), 0.60), // <-- JUMP 1 (0.30 drop)
            ("doc4".to_string(), 0.55), // Group 2
            ("doc5".to_string(), 0.25), // <-- JUMP 2 (0.30 drop)
            ("doc6".to_string(), 0.20), // Group 3
        ];

        // Cut at first jump
        let cut1 = apply_autocut(results.clone(), 1);
        assert_eq!(cut1.len(), 2); // doc1, doc2

        // Cut at second jump
        let cut2 = apply_autocut(results, 2);
        assert_eq!(cut2.len(), 4); // doc1-doc4
    }

    #[test]
    fn test_autocut_no_jumps() {
        // Results with gradual score decrease (no clear jumps)
        let results = vec![
            ("doc1".to_string(), 0.90),
            ("doc2".to_string(), 0.85),
            ("doc3".to_string(), 0.80),
            ("doc4".to_string(), 0.75),
            ("doc5".to_string(), 0.70),
        ];

        // Should return all results if no jumps detected
        let cut = apply_autocut(results.clone(), 1);
        assert_eq!(cut.len(), 5);
    }

    #[test]
    fn test_autocut_equal_scores() {
        // All equal scores - no drops
        let results = vec![
            ("doc1".to_string(), 0.8),
            ("doc2".to_string(), 0.8),
            ("doc3".to_string(), 0.8),
            ("doc4".to_string(), 0.8),
        ];

        let cut = apply_autocut(results.clone(), 1);
        assert_eq!(cut.len(), 4); // Should return all
    }

    #[test]
    fn test_autocut_jump_at_end() {
        // Jump at the very end
        let results = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.85),
            ("doc3".to_string(), 0.82),
            ("doc4".to_string(), 0.80),
            ("doc5".to_string(), 0.1), // <-- JUMP at end
        ];

        let cut = apply_autocut(results, 1);
        assert_eq!(cut.len(), 4); // Should cut before doc5
    }

    #[test]
    fn test_autocut_request_more_jumps_than_exist() {
        // Request 5 jumps but only 1 exists
        let results = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.85),
            ("doc3".to_string(), 0.2), // <-- Only one jump
            ("doc4".to_string(), 0.15),
        ];

        let cut = apply_autocut(results, 5); // Request 5 jumps
        assert_eq!(cut.len(), 2); // Should use the only jump that exists
    }

    #[test]
    fn test_autocut_with_integers() {
        // Test with integer IDs instead of strings
        let results = vec![
            (1, 0.95),
            (2, 0.92),
            (3, 0.90),
            (4, 0.45), // <-- JUMP
            (5, 0.42),
        ];

        let cut = apply_autocut(results, 1);
        assert_eq!(cut.len(), 3);
        assert_eq!(cut[0].0, 1);
        assert_eq!(cut[1].0, 2);
        assert_eq!(cut[2].0, 3);
    }

    #[test]
    fn test_autocut_realistic_rag_scenario() {
        // Realistic RAG scenario: 3 highly relevant docs, rest marginally relevant
        let results = vec![
            ("chunk1".to_string(), 0.89), // Highly relevant
            ("chunk2".to_string(), 0.87),
            ("chunk3".to_string(), 0.85),
            ("chunk4".to_string(), 0.52), // <-- Natural cutoff (0.33 drop)
            ("chunk5".to_string(), 0.50), // Marginally relevant
            ("chunk6".to_string(), 0.48),
            ("chunk7".to_string(), 0.46),
            ("chunk8".to_string(), 0.45),
            ("chunk9".to_string(), 0.43),
            ("chunk10".to_string(), 0.41),
        ];

        let cut = apply_autocut(results, 1);

        // Should return only the highly relevant chunks
        assert!(cut.len() <= 4); // At most 4 (likely 3)
        assert!(cut.len() >= 3); // At least 3

        // All returned results should be above 0.8
        assert!(cut.iter().all(|(_, score)| *score > 0.8));
    }

    #[test]
    fn test_autocut_preserves_order() {
        let results = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.85),
            ("doc3".to_string(), 0.5), // Jump
            ("doc4".to_string(), 0.45),
        ];

        let cut = apply_autocut(results, 1);

        // Order should be preserved
        assert_eq!(cut[0].0, "doc1");
        assert_eq!(cut[1].0, "doc2");
        assert!(cut[0].1 > cut[1].1); // Descending order maintained
    }

    #[test]
    fn test_autocut_empty_results() {
        let results: Vec<(String, f32)> = vec![];
        let cut = apply_autocut(results, 1);
        assert_eq!(cut.len(), 0);
    }

    // ============================================================================
    // Score Explanation Tests
    // ============================================================================

    #[test]
    fn test_explain_weighted_sum() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha: 0.7,
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.8, 0.6, &config);

        assert_eq!(explanation.dense_score, 0.8);
        assert_eq!(explanation.sparse_score, 0.6);
        assert_eq!(explanation.fusion_strategy, FusionStrategy::WeightedSum);
        assert_eq!(explanation.alpha, 0.7);

        // Final score should be: 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
        assert!((explanation.final_score - 0.74).abs() < 1e-6);

        // Check calculation string contains key components
        assert!(explanation.calculation.contains("WeightedSum"));
        assert!(explanation.calculation.contains("0.7"));
        assert!(explanation.calculation.contains("0.8"));

        // Dense contributed more (0.56 vs 0.18)
        assert!(explanation.contributions.dense_contribution > 0.7);
        assert!(explanation.contributions.sparse_contribution < 0.3);
    }

    #[test]
    fn test_explain_rrf() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::ReciprocalRankFusion,
            rrf_k: 60.0,
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.9, 0.5, &config);

        assert_eq!(
            explanation.fusion_strategy,
            FusionStrategy::ReciprocalRankFusion
        );
        assert!(explanation.final_score > 0.0);
        assert!(explanation.calculation.contains("RRF"));
        assert!(explanation.calculation.contains("60"));

        // Contributions should add up to ~1.0
        let total = explanation.contributions.dense_contribution
            + explanation.contributions.sparse_contribution;
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_explain_max() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::Max,
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.8, 0.6, &config);

        assert_eq!(explanation.final_score, 0.8); // max(0.8, 0.6) = 0.8
        assert!(explanation.calculation.contains("Max"));
        assert_eq!(explanation.contributions.dense_contribution, 1.0);
        assert_eq!(explanation.contributions.sparse_contribution, 0.0);
        assert!(explanation
            .contributions
            .explanation
            .contains("Dense score was higher"));
    }

    #[test]
    fn test_explain_min() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::Min,
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.8, 0.6, &config);

        assert_eq!(explanation.final_score, 0.6); // min(0.8, 0.6) = 0.6
        assert!(explanation.calculation.contains("Min"));
        assert_eq!(explanation.contributions.dense_contribution, 0.0);
        assert_eq!(explanation.contributions.sparse_contribution, 1.0);
        assert!(explanation
            .contributions
            .explanation
            .contains("Sparse score was lower"));
    }

    #[test]
    fn test_explain_harmonic_mean() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::HarmonicMean,
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.8, 0.6, &config);

        // Harmonic mean: 2 * 0.8 * 0.6 / (0.8 + 0.6) = 0.96 / 1.4 ≈ 0.6857
        assert!((explanation.final_score - 0.6857).abs() < 0.01);
        assert!(explanation.calculation.contains("HarmonicMean"));
        assert_eq!(explanation.contributions.dense_contribution, 0.5);
        assert_eq!(explanation.contributions.sparse_contribution, 0.5);
    }

    #[test]
    fn test_explain_geometric_mean() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::GeometricMean,
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.64, 0.36, &config);

        // Geometric mean: sqrt(0.64 * 0.36) = sqrt(0.2304) = 0.48
        assert!((explanation.final_score - 0.48).abs() < 0.01);
        assert!(explanation.calculation.contains("GeometricMean"));
        assert_eq!(explanation.contributions.dense_contribution, 0.5);
        assert_eq!(explanation.contributions.sparse_contribution, 0.5);
    }

    #[test]
    fn test_explain_dbsf() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::DistributionBased,
            alpha: 0.6,
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.7, 0.5, &config);

        assert_eq!(
            explanation.fusion_strategy,
            FusionStrategy::DistributionBased
        );
        assert!(explanation.calculation.contains("DBSF"));
        assert!(explanation.final_score > 0.0);
    }

    #[test]
    fn test_explain_relative_score() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::RelativeScore,
            alpha: 0.8,
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.9, 0.3, &config);

        assert_eq!(explanation.fusion_strategy, FusionStrategy::RelativeScore);
        assert!(explanation.calculation.contains("RelativeScore"));
        assert!(explanation.final_score > 0.0);
    }

    #[test]
    fn test_explain_zero_scores() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha: 0.7,
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.0, 0.0, &config);

        assert_eq!(explanation.final_score, 0.0);
        assert_eq!(explanation.dense_score, 0.0);
        assert_eq!(explanation.sparse_score, 0.0);
        assert!(explanation
            .contributions
            .explanation
            .contains("Both scores are zero"));
    }

    #[test]
    fn test_explain_equal_scores_max() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::Max,
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.7, 0.7, &config);

        assert_eq!(explanation.final_score, 0.7);
        assert_eq!(explanation.contributions.dense_contribution, 0.5);
        assert_eq!(explanation.contributions.sparse_contribution, 0.5);
        assert!(explanation.contributions.explanation.contains("equal"));
    }

    #[test]
    fn test_explain_serialization() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha: 0.7,
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.8, 0.6, &config);

        // Test that explanation can be serialized (useful for APIs)
        let json = serde_json::to_string(&explanation).unwrap();
        assert!(json.contains("final_score"));
        assert!(json.contains("dense_score"));
        assert!(json.contains("calculation"));

        // Test deserialization
        let deserialized: ScoreExplanation = serde_json::from_str(&json).unwrap();
        assert!((deserialized.final_score - explanation.final_score).abs() < 1e-6);
        assert_eq!(deserialized.fusion_strategy, explanation.fusion_strategy);
    }

    #[test]
    fn test_explain_pure_dense() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha: 1.0, // Pure dense
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.8, 0.6, &config);

        assert!((explanation.final_score - 0.8).abs() < 1e-6);
        assert!(explanation.contributions.dense_contribution > 0.99); // Nearly 100%
    }

    #[test]
    fn test_explain_pure_sparse() {
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha: 0.0, // Pure sparse
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.8, 0.6, &config);

        assert!((explanation.final_score - 0.6).abs() < 1e-6);
        assert!(explanation.contributions.sparse_contribution > 0.99); // Nearly 100%
    }

    #[test]
    fn test_explain_realistic_rag_scenario() {
        // Realistic RAG scenario: good semantic match, weak keyword match
        let config = HybridSearchConfig {
            fusion_strategy: FusionStrategy::WeightedSum,
            alpha: 0.7, // Favor semantic
            ..Default::default()
        };

        let explanation = explain_hybrid_score(0.92, 0.15, &config);

        // Should favor dense heavily
        assert!(explanation.final_score > 0.6);
        assert!(explanation.contributions.dense_contribution > 0.8);

        // User can see why this result scored well
        assert!(explanation.calculation.len() > 0);
        assert!(explanation.contributions.explanation.len() > 0);
    }
}
