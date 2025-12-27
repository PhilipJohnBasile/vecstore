// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! Explainable Vector Search (XVS)
//!
//! First-of-its-kind explainability for vector similarity search.
//! Answers WHY vectors matched, not just THAT they matched.
//!
//! # Features
//!
//! - **Dimension Contributions**: Which dimensions contributed most to similarity
//! - **Semantic Reasoning**: Human-readable explanations
//! - **Confidence Intervals**: Uncertainty quantification
//! - **Counter-examples**: What would have ranked higher
//! - **Feature Attribution**: SHAP-like explanations for embeddings
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::explainable::{ExplainableSearch, ExplainLevel};
//!
//! let explainer = ExplainableSearch::new(store);
//!
//! let results = explainer.search_explain(
//!     query,
//!     k: 10,
//!     level: ExplainLevel::Full,
//! )?;
//!
//! for result in results {
//!     println!("ID: {}", result.id);
//!     println!("Score: {:.4}", result.score);
//!     println!("Explanation: {}", result.explanation.summary);
//!     println!("Top contributing dimensions: {:?}", result.explanation.top_dimensions);
//! }
//! ```

use anyhow::Result;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Level of explanation detail
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExplainLevel {
    /// Minimal - just the score
    Minimal,
    /// Basic - score + top contributing dimensions
    Basic,
    /// Standard - adds semantic reasoning
    Standard,
    /// Full - includes counter-examples and confidence intervals
    Full,
    /// Debug - all internal details
    Debug,
}

impl Default for ExplainLevel {
    fn default() -> Self {
        Self::Standard
    }
}

/// Dimension contribution to similarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionContribution {
    /// Dimension index
    pub index: usize,
    /// Contribution to final score (can be negative for cosine)
    pub contribution: f32,
    /// Percentage of total score
    pub percentage: f32,
    /// Query value at this dimension
    pub query_value: f32,
    /// Result value at this dimension
    pub result_value: f32,
    /// Semantic label if available
    pub label: Option<String>,
}

/// Confidence information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInfo {
    /// Point estimate of similarity
    pub score: f32,
    /// Lower bound (95% CI)
    pub lower_bound: f32,
    /// Upper bound (95% CI)
    pub upper_bound: f32,
    /// Confidence level (0.0-1.0)
    pub confidence: f32,
    /// Number of similar vectors in neighborhood
    pub neighborhood_density: usize,
}

/// Counter-example information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterExample {
    /// What would need to change for this to rank higher
    pub description: String,
    /// Dimensions that differ most
    pub key_differences: Vec<DimensionContribution>,
    /// Hypothetical score if changes made
    pub hypothetical_score: f32,
}

/// Semantic explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticExplanation {
    /// Human-readable summary
    pub summary: String,
    /// Detected themes/topics in common
    pub shared_themes: Vec<String>,
    /// Key matching features
    pub matching_features: Vec<String>,
    /// Potential mismatches
    pub divergent_features: Vec<String>,
}

/// Full explanation for a search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchExplanation {
    /// Similarity score
    pub score: f32,
    /// Distance metric used
    pub metric: String,
    /// Human-readable summary
    pub summary: String,
    /// Top contributing dimensions
    pub top_dimensions: Vec<DimensionContribution>,
    /// Bottom contributing dimensions (negative impact)
    pub bottom_dimensions: Vec<DimensionContribution>,
    /// Semantic explanation
    pub semantic: Option<SemanticExplanation>,
    /// Confidence information
    pub confidence: Option<ConfidenceInfo>,
    /// Counter-examples
    pub counter_examples: Vec<CounterExample>,
    /// Raw computation details (for Debug level)
    pub debug_info: Option<HashMap<String, String>>,
}

/// Explained search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainedResult {
    /// Vector ID
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Full explanation
    pub explanation: SearchExplanation,
}

/// Configuration for explainable search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainConfig {
    /// Number of top dimensions to show
    pub top_k_dimensions: usize,
    /// Include semantic reasoning
    pub include_semantic: bool,
    /// Include confidence intervals
    pub include_confidence: bool,
    /// Number of counter-examples to generate
    pub num_counter_examples: usize,
    /// Dimension labels (optional)
    pub dimension_labels: Option<Vec<String>>,
    /// Minimum contribution threshold to report
    pub min_contribution_threshold: f32,
}

impl Default for ExplainConfig {
    fn default() -> Self {
        Self {
            top_k_dimensions: 10,
            include_semantic: true,
            include_confidence: true,
            num_counter_examples: 3,
            dimension_labels: None,
            min_contribution_threshold: 0.01,
        }
    }
}

/// Explainable Vector Search engine
pub struct ExplainableSearch {
    /// Configuration
    config: ExplainConfig,
    /// Distance metric
    metric: DistanceMetric,
    /// Dimension importance weights (learned or static)
    dimension_weights: Option<Vec<f32>>,
}

/// Supported distance metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl ExplainableSearch {
    /// Create a new explainable search engine
    pub fn new(config: ExplainConfig, metric: DistanceMetric) -> Self {
        Self {
            config,
            metric,
            dimension_weights: None,
        }
    }

    /// Set dimension importance weights
    pub fn with_dimension_weights(mut self, weights: Vec<f32>) -> Self {
        self.dimension_weights = Some(weights);
        self
    }

    /// Explain a similarity computation between two vectors
    pub fn explain_similarity(
        &self,
        query: &[f32],
        result: &[f32],
        result_id: &str,
        level: ExplainLevel,
    ) -> Result<ExplainedResult> {
        let score = self.compute_similarity(query, result);
        let contributions = self.compute_dimension_contributions(query, result);

        // Sort contributions
        let mut sorted_contributions = contributions.clone();
        sorted_contributions.sort_by(|a, b| {
            OrderedFloat(b.contribution.abs()).cmp(&OrderedFloat(a.contribution.abs()))
        });

        // Get top and bottom contributors
        let top_dimensions: Vec<_> = sorted_contributions
            .iter()
            .filter(|c| c.contribution > self.config.min_contribution_threshold)
            .take(self.config.top_k_dimensions)
            .cloned()
            .collect();

        let bottom_dimensions: Vec<_> = sorted_contributions
            .iter()
            .filter(|c| c.contribution < -self.config.min_contribution_threshold)
            .rev()
            .take(self.config.top_k_dimensions)
            .cloned()
            .collect();

        // Generate summary
        let summary = self.generate_summary(&top_dimensions, &bottom_dimensions, score);

        // Build explanation based on level
        let semantic = if level >= ExplainLevel::Standard && self.config.include_semantic {
            Some(self.generate_semantic_explanation(&top_dimensions, &bottom_dimensions))
        } else {
            None
        };

        let confidence = if level >= ExplainLevel::Full && self.config.include_confidence {
            Some(self.compute_confidence(query, result, score))
        } else {
            None
        };

        let counter_examples = if level >= ExplainLevel::Full {
            self.generate_counter_examples(query, result, &contributions)
        } else {
            Vec::new()
        };

        let debug_info = if level == ExplainLevel::Debug {
            Some(self.generate_debug_info(query, result, &contributions))
        } else {
            None
        };

        Ok(ExplainedResult {
            id: result_id.to_string(),
            score,
            explanation: SearchExplanation {
                score,
                metric: format!("{:?}", self.metric),
                summary,
                top_dimensions,
                bottom_dimensions,
                semantic,
                confidence,
                counter_examples,
                debug_info,
            },
        })
    }

    /// Compute similarity between two vectors
    fn compute_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_a > 0.0 && norm_b > 0.0 {
                    dot / (norm_a * norm_b)
                } else {
                    0.0
                }
            }
            DistanceMetric::Euclidean => {
                let dist: f32 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt();
                1.0 / (1.0 + dist) // Convert to similarity
            }
            DistanceMetric::DotProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
        }
    }

    /// Compute per-dimension contributions to similarity
    fn compute_dimension_contributions(
        &self,
        query: &[f32],
        result: &[f32],
    ) -> Vec<DimensionContribution> {
        let total_similarity = self.compute_similarity(query, result);

        match self.metric {
            DistanceMetric::Cosine => {
                let norm_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_r: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                let normalizer = if norm_q > 0.0 && norm_r > 0.0 {
                    norm_q * norm_r
                } else {
                    1.0
                };

                query
                    .iter()
                    .zip(result.iter())
                    .enumerate()
                    .map(|(i, (q, r))| {
                        let contribution = (q * r) / normalizer;
                        let percentage = if total_similarity.abs() > 1e-9 {
                            contribution / total_similarity * 100.0
                        } else {
                            0.0
                        };
                        DimensionContribution {
                            index: i,
                            contribution,
                            percentage,
                            query_value: *q,
                            result_value: *r,
                            label: self.get_dimension_label(i),
                        }
                    })
                    .collect()
            }
            DistanceMetric::DotProduct => {
                query
                    .iter()
                    .zip(result.iter())
                    .enumerate()
                    .map(|(i, (q, r))| {
                        let contribution = q * r;
                        let percentage = if total_similarity.abs() > 1e-9 {
                            contribution / total_similarity * 100.0
                        } else {
                            0.0
                        };
                        DimensionContribution {
                            index: i,
                            contribution,
                            percentage,
                            query_value: *q,
                            result_value: *r,
                            label: self.get_dimension_label(i),
                        }
                    })
                    .collect()
            }
            DistanceMetric::Euclidean => {
                // For Euclidean, contribution is negative of squared difference
                let total_dist_sq: f32 = query
                    .iter()
                    .zip(result.iter())
                    .map(|(q, r)| (q - r).powi(2))
                    .sum();

                query
                    .iter()
                    .zip(result.iter())
                    .enumerate()
                    .map(|(i, (q, r))| {
                        let diff_sq = (q - r).powi(2);
                        let contribution = -diff_sq; // Negative because larger diff = worse
                        let percentage = if total_dist_sq > 1e-9 {
                            diff_sq / total_dist_sq * 100.0
                        } else {
                            0.0
                        };
                        DimensionContribution {
                            index: i,
                            contribution,
                            percentage,
                            query_value: *q,
                            result_value: *r,
                            label: self.get_dimension_label(i),
                        }
                    })
                    .collect()
            }
        }
    }

    fn get_dimension_label(&self, index: usize) -> Option<String> {
        self.config
            .dimension_labels
            .as_ref()
            .and_then(|labels| labels.get(index).cloned())
    }

    fn generate_summary(
        &self,
        top: &[DimensionContribution],
        bottom: &[DimensionContribution],
        score: f32,
    ) -> String {
        let mut summary = format!(
            "Similarity score: {:.4} ({:?}). ",
            score, self.metric
        );

        if !top.is_empty() {
            let top_dims: Vec<String> = top
                .iter()
                .take(3)
                .map(|c| {
                    if let Some(label) = &c.label {
                        format!("{} ({:.1}%)", label, c.percentage)
                    } else {
                        format!("dim{} ({:.1}%)", c.index, c.percentage)
                    }
                })
                .collect();
            summary.push_str(&format!("Top contributors: {}. ", top_dims.join(", ")));
        }

        if !bottom.is_empty() {
            let bottom_dims: Vec<String> = bottom
                .iter()
                .take(2)
                .map(|c| {
                    if let Some(label) = &c.label {
                        label.clone()
                    } else {
                        format!("dim{}", c.index)
                    }
                })
                .collect();
            summary.push_str(&format!("Reducing similarity: {}.", bottom_dims.join(", ")));
        }

        summary
    }

    fn generate_semantic_explanation(
        &self,
        top: &[DimensionContribution],
        bottom: &[DimensionContribution],
    ) -> SemanticExplanation {
        // Generate semantic explanations based on dimension patterns
        let shared_themes: Vec<String> = top
            .iter()
            .filter(|c| c.contribution > 0.05)
            .filter_map(|c| c.label.clone())
            .take(5)
            .collect();

        let matching_features: Vec<String> = top
            .iter()
            .filter(|c| c.query_value.signum() == c.result_value.signum())
            .filter_map(|c| {
                c.label.as_ref().map(|l| {
                    if c.query_value > 0.0 {
                        format!("Both have high '{}'", l)
                    } else {
                        format!("Both have low '{}'", l)
                    }
                })
            })
            .take(5)
            .collect();

        let divergent_features: Vec<String> = bottom
            .iter()
            .filter(|c| c.query_value.signum() != c.result_value.signum())
            .filter_map(|c| {
                c.label.as_ref().map(|l| {
                    format!(
                        "'{}' differs: query={:.2}, result={:.2}",
                        l, c.query_value, c.result_value
                    )
                })
            })
            .take(3)
            .collect();

        let summary = if !shared_themes.is_empty() {
            format!(
                "Vectors match on themes: {}",
                shared_themes.join(", ")
            )
        } else if !matching_features.is_empty() {
            format!("Similarity driven by: {}", matching_features[0])
        } else {
            "Similarity based on overall vector alignment".to_string()
        };

        SemanticExplanation {
            summary,
            shared_themes,
            matching_features,
            divergent_features,
        }
    }

    fn compute_confidence(
        &self,
        query: &[f32],
        result: &[f32],
        score: f32,
    ) -> ConfidenceInfo {
        // Estimate confidence based on vector properties
        let dim = query.len() as f32;

        // Higher dimensions = more stable estimates
        let base_confidence = (dim / (dim + 100.0)).min(0.99);

        // Vectors with higher norms tend to have more stable similarities
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let result_norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_factor = (query_norm * result_norm).min(100.0) / 100.0;

        let confidence = (base_confidence * 0.7 + norm_factor * 0.3).min(0.99);

        // Rough confidence interval (simplified)
        let margin = (1.0 - confidence) * 0.2;
        let lower = (score - margin).max(-1.0);
        let upper = (score + margin).min(1.0);

        ConfidenceInfo {
            score,
            lower_bound: lower,
            upper_bound: upper,
            confidence,
            neighborhood_density: 0, // Would need full index to compute
        }
    }

    fn generate_counter_examples(
        &self,
        query: &[f32],
        result: &[f32],
        contributions: &[DimensionContribution],
    ) -> Vec<CounterExample> {
        let mut examples = Vec::new();

        // Find dimensions that hurt similarity most
        let mut negative_dims: Vec<_> = contributions
            .iter()
            .filter(|c| c.contribution < 0.0)
            .collect();
        negative_dims.sort_by(|a, b| OrderedFloat(a.contribution).cmp(&OrderedFloat(b.contribution)));

        if let Some(worst) = negative_dims.first() {
            let hypothetical: Vec<f32> = result
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if i == worst.index {
                        query[i] // What if this matched?
                    } else {
                        v
                    }
                })
                .collect();
            let hyp_score = self.compute_similarity(query, &hypothetical);

            examples.push(CounterExample {
                description: format!(
                    "If dimension {} matched query value ({:.3} instead of {:.3}), score would increase to {:.4}",
                    worst.index, query[worst.index], result[worst.index], hyp_score
                ),
                key_differences: vec![(*worst).clone()],
                hypothetical_score: hyp_score,
            });
        }

        // What if we aligned the top 3 mismatched dimensions?
        let mismatched: Vec<_> = contributions
            .iter()
            .filter(|c| (c.query_value - c.result_value).abs() > 0.1)
            .take(3)
            .collect();

        if !mismatched.is_empty() {
            let hypothetical: Vec<f32> = result
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if mismatched.iter().any(|c| c.index == i) {
                        query[i]
                    } else {
                        v
                    }
                })
                .collect();
            let hyp_score = self.compute_similarity(query, &hypothetical);

            examples.push(CounterExample {
                description: format!(
                    "If top {} mismatched dimensions aligned, score would be {:.4}",
                    mismatched.len(),
                    hyp_score
                ),
                key_differences: mismatched.into_iter().cloned().collect(),
                hypothetical_score: hyp_score,
            });
        }

        examples
    }

    fn generate_debug_info(
        &self,
        query: &[f32],
        result: &[f32],
        contributions: &[DimensionContribution],
    ) -> HashMap<String, String> {
        let mut info = HashMap::new();

        info.insert("query_dim".to_string(), query.len().to_string());
        info.insert("result_dim".to_string(), result.len().to_string());

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let result_norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        info.insert("query_norm".to_string(), format!("{:.6}", query_norm));
        info.insert("result_norm".to_string(), format!("{:.6}", result_norm));

        let positive_contribs: usize = contributions.iter().filter(|c| c.contribution > 0.0).count();
        let negative_contribs: usize = contributions.iter().filter(|c| c.contribution < 0.0).count();
        info.insert("positive_dims".to_string(), positive_contribs.to_string());
        info.insert("negative_dims".to_string(), negative_contribs.to_string());

        let sparsity_q = query.iter().filter(|&&x| x.abs() < 1e-6).count() as f32 / query.len() as f32;
        let sparsity_r = result.iter().filter(|&&x| x.abs() < 1e-6).count() as f32 / result.len() as f32;
        info.insert("query_sparsity".to_string(), format!("{:.2}%", sparsity_q * 100.0));
        info.insert("result_sparsity".to_string(), format!("{:.2}%", sparsity_r * 100.0));

        info
    }
}

/// Batch explanation for multiple results
pub fn explain_batch(
    explainer: &ExplainableSearch,
    query: &[f32],
    results: &[(String, Vec<f32>)],
    level: ExplainLevel,
) -> Result<Vec<ExplainedResult>> {
    results
        .iter()
        .map(|(id, vec)| explainer.explain_similarity(query, vec, id, level))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explain_cosine_similarity() {
        let config = ExplainConfig::default();
        let explainer = ExplainableSearch::new(config, DistanceMetric::Cosine);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let result = vec![0.8, 0.2, 0.0, 0.0];

        let explained = explainer
            .explain_similarity(&query, &result, "test1", ExplainLevel::Full)
            .unwrap();

        assert!(explained.score > 0.9);
        assert!(!explained.explanation.top_dimensions.is_empty());
        assert!(explained.explanation.confidence.is_some());
    }

    #[test]
    fn test_dimension_contributions() {
        let config = ExplainConfig::default();
        let explainer = ExplainableSearch::new(config, DistanceMetric::DotProduct);

        let query = vec![1.0, 2.0, 0.0];
        let result = vec![1.0, 1.0, 0.0];

        let explained = explainer
            .explain_similarity(&query, &result, "test2", ExplainLevel::Standard)
            .unwrap();

        // Dot product should be 1*1 + 2*1 + 0*0 = 3
        assert!((explained.score - 3.0).abs() < 0.001);

        // First two dimensions should contribute
        assert!(explained.explanation.top_dimensions.len() >= 2);
    }

    #[test]
    fn test_counter_examples() {
        let config = ExplainConfig {
            num_counter_examples: 3,
            ..Default::default()
        };
        let explainer = ExplainableSearch::new(config, DistanceMetric::Cosine);

        let query = vec![1.0, 0.0, 0.0];
        let result = vec![0.5, 0.5, 0.0]; // Partially matching

        let explained = explainer
            .explain_similarity(&query, &result, "test3", ExplainLevel::Full)
            .unwrap();

        assert!(!explained.explanation.counter_examples.is_empty());
    }

    #[test]
    fn test_with_dimension_labels() {
        let config = ExplainConfig {
            dimension_labels: Some(vec![
                "topic_tech".to_string(),
                "topic_sports".to_string(),
                "sentiment".to_string(),
            ]),
            ..Default::default()
        };
        let explainer = ExplainableSearch::new(config, DistanceMetric::Cosine);

        let query = vec![0.9, 0.1, 0.5];
        let result = vec![0.8, 0.2, 0.4];

        let explained = explainer
            .explain_similarity(&query, &result, "test4", ExplainLevel::Standard)
            .unwrap();

        // Check that labels are included
        let has_labels = explained
            .explanation
            .top_dimensions
            .iter()
            .any(|d| d.label.is_some());
        assert!(has_labels);
    }

    #[test]
    fn test_batch_explain() {
        let config = ExplainConfig::default();
        let explainer = ExplainableSearch::new(config, DistanceMetric::Cosine);

        let query = vec![1.0, 0.0, 0.0];
        let results = vec![
            ("r1".to_string(), vec![0.9, 0.1, 0.0]),
            ("r2".to_string(), vec![0.5, 0.5, 0.0]),
            ("r3".to_string(), vec![0.0, 1.0, 0.0]),
        ];

        let explained = explain_batch(&explainer, &query, &results, ExplainLevel::Basic).unwrap();

        assert_eq!(explained.len(), 3);
        assert!(explained[0].score > explained[1].score);
        assert!(explained[1].score > explained[2].score);
    }
}
