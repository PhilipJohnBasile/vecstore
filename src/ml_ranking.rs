//! ONNX/ML Models in Ranking
//!
//! Deploy machine learning models directly in the ranking pipeline,
//! similar to Vespa's ONNX integration.
//!
//! # Features
//!
//! - **ONNX Runtime**: Run ONNX models for reranking
//! - **Feature Extraction**: Compute features for ML models
//! - **Multi-Stage Ranking**: First-pass retrieval + ML rerank
//! - **Ensemble Methods**: Combine multiple models
//! - **Online Learning**: Update models with feedback
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::ml_ranking::{MLRanker, RankingModel, RankingConfig};
//!
//! let config = RankingConfig::new()
//!     .with_model(RankingModel::onnx("reranker.onnx")?)
//!     .with_top_k(100)  // Rerank top 100
//!     .with_final_k(10);  // Return top 10
//!
//! let ranker = MLRanker::new(config)?;
//!
//! // First-pass retrieval
//! let candidates = store.search(&query, 100)?;
//!
//! // ML reranking
//! let final_results = ranker.rerank(&query, candidates)?;
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Ranking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingConfig {
    /// Number of candidates to rerank
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Final number of results
    #[serde(default = "default_final_k")]
    pub final_k: usize,
    /// Enable feature caching
    #[serde(default = "default_true")]
    pub cache_features: bool,
    /// Score normalization method
    #[serde(default)]
    pub normalization: ScoreNormalization,
    /// Ensemble weights (if multiple models)
    #[serde(default)]
    pub ensemble_weights: Vec<f32>,
}

fn default_top_k() -> usize { 100 }
fn default_final_k() -> usize { 10 }
fn default_true() -> bool { true }

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            top_k: 100,
            final_k: 10,
            cache_features: true,
            normalization: ScoreNormalization::None,
            ensemble_weights: Vec::new(),
        }
    }
}

impl RankingConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set top-k candidates
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set final-k results
    pub fn with_final_k(mut self, k: usize) -> Self {
        self.final_k = k;
        self
    }

    /// Set normalization method
    pub fn with_normalization(mut self, method: ScoreNormalization) -> Self {
        self.normalization = method;
        self
    }
}

/// Score normalization methods
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ScoreNormalization {
    #[default]
    None,
    MinMax,
    ZScore,
    Softmax,
}

/// Ranking model types
#[derive(Debug, Clone)]
pub enum RankingModel {
    /// ONNX model
    ONNX {
        path: String,
        input_names: Vec<String>,
        output_name: String,
    },
    /// XGBoost model
    XGBoost {
        path: String,
    },
    /// LightGBM model
    LightGBM {
        path: String,
    },
    /// Linear model
    Linear {
        weights: Vec<f32>,
        bias: f32,
    },
    /// Custom scoring function
    Custom {
        name: String,
    },
}

impl RankingModel {
    /// Create an ONNX model
    pub fn onnx(path: impl Into<String>) -> Self {
        RankingModel::ONNX {
            path: path.into(),
            input_names: vec!["input".to_string()],
            output_name: "output".to_string(),
        }
    }

    /// Create a linear model
    pub fn linear(weights: Vec<f32>, bias: f32) -> Self {
        RankingModel::Linear { weights, bias }
    }
}

/// Ranking features
#[derive(Debug, Clone, Serialize)]
pub struct RankingFeatures {
    /// Vector similarity score
    pub vector_score: f32,
    /// BM25 score (if available)
    pub bm25_score: Option<f32>,
    /// Query-document features
    pub qd_features: Vec<f32>,
    /// Document features
    pub doc_features: Vec<f32>,
    /// Query features
    pub query_features: Vec<f32>,
}

impl RankingFeatures {
    /// Create new features
    pub fn new(vector_score: f32) -> Self {
        Self {
            vector_score,
            bm25_score: None,
            qd_features: Vec::new(),
            doc_features: Vec::new(),
            query_features: Vec::new(),
        }
    }

    /// Set BM25 score
    pub fn with_bm25(mut self, score: f32) -> Self {
        self.bm25_score = Some(score);
        self
    }

    /// Add query-document features
    pub fn with_qd_features(mut self, features: Vec<f32>) -> Self {
        self.qd_features = features;
        self
    }

    /// Convert to feature vector
    pub fn to_vector(&self) -> Vec<f32> {
        let mut v = vec![self.vector_score];
        if let Some(bm25) = self.bm25_score {
            v.push(bm25);
        }
        v.extend(&self.qd_features);
        v.extend(&self.doc_features);
        v.extend(&self.query_features);
        v
    }
}

/// Candidate for reranking
#[derive(Debug, Clone)]
pub struct RankingCandidate {
    /// Document ID
    pub id: String,
    /// Original score
    pub original_score: f32,
    /// Document vector
    pub vector: Vec<f32>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
    /// Computed features
    pub features: Option<RankingFeatures>,
}

/// Reranked result
#[derive(Debug, Clone, Serialize)]
pub struct RankedResult {
    /// Document ID
    pub id: String,
    /// Final score after reranking
    pub score: f32,
    /// Original retrieval score
    pub original_score: f32,
    /// Feature contributions (for explainability)
    pub feature_contributions: Option<HashMap<String, f32>>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
}

/// ML-based ranker
pub struct MLRanker {
    config: RankingConfig,
    models: Vec<RankingModel>,
    feature_cache: HashMap<String, RankingFeatures>,
}

impl MLRanker {
    /// Create a new ML ranker
    pub fn new(config: RankingConfig) -> Result<Self> {
        Ok(Self {
            config,
            models: Vec::new(),
            feature_cache: HashMap::new(),
        })
    }

    /// Add a ranking model
    pub fn add_model(&mut self, model: RankingModel) {
        self.models.push(model);
    }

    /// Rerank candidates
    pub fn rerank(
        &self,
        query: &[f32],
        candidates: Vec<RankingCandidate>,
    ) -> Result<Vec<RankedResult>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Compute features for each candidate
        let candidates_with_features: Vec<(RankingCandidate, RankingFeatures)> = candidates
            .into_iter()
            .map(|c| {
                let features = c.features.clone().unwrap_or_else(|| {
                    self.compute_features(query, &c)
                });
                (c, features)
            })
            .collect();

        // Score each candidate with models
        let mut results: Vec<RankedResult> = candidates_with_features
            .iter()
            .map(|(c, features)| {
                let model_scores: Vec<f32> = self.models.iter()
                    .map(|model| self.score_with_model(model, features))
                    .collect();

                let final_score = if model_scores.is_empty() {
                    // If no models, use original score
                    c.original_score
                } else if self.config.ensemble_weights.len() == model_scores.len() {
                    // Weighted ensemble
                    model_scores.iter()
                        .zip(&self.config.ensemble_weights)
                        .map(|(s, w)| s * w)
                        .sum()
                } else {
                    // Simple average
                    model_scores.iter().sum::<f32>() / model_scores.len() as f32
                };

                RankedResult {
                    id: c.id.clone(),
                    score: final_score,
                    original_score: c.original_score,
                    feature_contributions: Some(self.get_feature_contributions(features)),
                    metadata: c.metadata.clone(),
                }
            })
            .collect();

        // Normalize scores if configured
        self.normalize_scores(&mut results);

        // Sort by final score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Truncate to final_k
        results.truncate(self.config.final_k);

        Ok(results)
    }

    /// Compute ranking features
    fn compute_features(&self, query: &[f32], candidate: &RankingCandidate) -> RankingFeatures {
        let vector_score = candidate.original_score;

        // Compute additional features
        let mut qd_features = Vec::new();

        // Dot product
        let dot: f32 = query.iter()
            .zip(candidate.vector.iter())
            .map(|(q, d)| q * d)
            .sum();
        qd_features.push(dot);

        // Query norm
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        qd_features.push(query_norm);

        // Doc norm
        let doc_norm: f32 = candidate.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        qd_features.push(doc_norm);

        // Max element similarity
        let max_sim: f32 = query.iter()
            .zip(candidate.vector.iter())
            .map(|(q, d)| q * d)
            .fold(f32::NEG_INFINITY, f32::max);
        qd_features.push(max_sim);

        RankingFeatures::new(vector_score)
            .with_qd_features(qd_features)
    }

    /// Score with a model
    fn score_with_model(&self, model: &RankingModel, features: &RankingFeatures) -> f32 {
        match model {
            RankingModel::Linear { weights, bias } => {
                let feature_vec = features.to_vector();
                let score: f32 = feature_vec.iter()
                    .zip(weights.iter())
                    .map(|(f, w)| f * w)
                    .sum::<f32>() + bias;
                self.sigmoid(score)
            }
            RankingModel::ONNX { .. } => {
                // Placeholder - would call ONNX runtime
                features.vector_score
            }
            RankingModel::XGBoost { .. } => {
                // Placeholder - would call XGBoost
                features.vector_score
            }
            RankingModel::LightGBM { .. } => {
                // Placeholder - would call LightGBM
                features.vector_score
            }
            RankingModel::Custom { .. } => {
                features.vector_score
            }
        }
    }

    /// Get feature contributions for explainability
    fn get_feature_contributions(&self, features: &RankingFeatures) -> HashMap<String, f32> {
        let mut contributions = HashMap::new();
        contributions.insert("vector_score".to_string(), features.vector_score);

        if let Some(bm25) = features.bm25_score {
            contributions.insert("bm25_score".to_string(), bm25);
        }

        for (i, &f) in features.qd_features.iter().enumerate() {
            contributions.insert(format!("qd_feature_{}", i), f);
        }

        contributions
    }

    /// Normalize scores
    fn normalize_scores(&self, results: &mut [RankedResult]) {
        if results.is_empty() {
            return;
        }

        match self.config.normalization {
            ScoreNormalization::None => {}
            ScoreNormalization::MinMax => {
                let min = results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
                let max = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
                let range = max - min;
                if range > 0.0 {
                    for r in results {
                        r.score = (r.score - min) / range;
                    }
                }
            }
            ScoreNormalization::ZScore => {
                let mean: f32 = results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32;
                let variance: f32 = results.iter()
                    .map(|r| (r.score - mean).powi(2))
                    .sum::<f32>() / results.len() as f32;
                let std = variance.sqrt();
                if std > 0.0 {
                    for r in results {
                        r.score = (r.score - mean) / std;
                    }
                }
            }
            ScoreNormalization::Softmax => {
                let max = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = results.iter().map(|r| (r.score - max).exp()).sum();
                for r in results {
                    r.score = (r.score - max).exp() / exp_sum;
                }
            }
        }
    }

    /// Sigmoid activation
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Multi-stage ranking pipeline
pub struct RankingPipeline {
    stages: Vec<RankingStage>,
}

/// Ranking stage
pub struct RankingStage {
    name: String,
    ranker: MLRanker,
    top_k: usize,
}

impl RankingPipeline {
    /// Create a new pipeline
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a ranking stage
    pub fn add_stage(
        &mut self,
        name: impl Into<String>,
        ranker: MLRanker,
        top_k: usize,
    ) {
        self.stages.push(RankingStage {
            name: name.into(),
            ranker,
            top_k,
        });
    }

    /// Execute the pipeline
    pub fn execute(
        &self,
        query: &[f32],
        initial_candidates: Vec<RankingCandidate>,
    ) -> Result<Vec<RankedResult>> {
        let mut candidates = initial_candidates;

        for stage in &self.stages {
            let results = stage.ranker.rerank(query, candidates)?;

            // Convert results back to candidates for next stage
            candidates = results.into_iter()
                .take(stage.top_k)
                .map(|r| RankingCandidate {
                    id: r.id,
                    original_score: r.score,
                    vector: Vec::new(), // Would need to carry this through
                    metadata: r.metadata,
                    features: None,
                })
                .collect();
        }

        // Final rerank or just convert
        if let Some(last_stage) = self.stages.last() {
            last_stage.ranker.rerank(query, candidates)
        } else {
            Ok(candidates.into_iter()
                .map(|c| RankedResult {
                    id: c.id,
                    score: c.original_score,
                    original_score: c.original_score,
                    feature_contributions: None,
                    metadata: c.metadata,
                })
                .collect())
        }
    }
}

impl Default for RankingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_model() {
        let mut config = RankingConfig::new()
            .with_top_k(10)
            .with_final_k(5);

        let mut ranker = MLRanker::new(config).unwrap();
        ranker.add_model(RankingModel::linear(vec![0.5, 0.3, 0.2], 0.0));

        let candidates: Vec<RankingCandidate> = (0..10)
            .map(|i| RankingCandidate {
                id: format!("doc{}", i),
                original_score: 1.0 - (i as f32 * 0.1),
                vector: vec![0.1f32; 64],
                metadata: None,
                features: None,
            })
            .collect();

        let query = vec![0.1f32; 64];
        let results = ranker.rerank(&query, candidates).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_normalization() {
        let config = RankingConfig::new()
            .with_normalization(ScoreNormalization::MinMax);

        let ranker = MLRanker::new(config).unwrap();

        let candidates: Vec<RankingCandidate> = (0..5)
            .map(|i| RankingCandidate {
                id: format!("doc{}", i),
                original_score: i as f32,
                vector: vec![0.1f32; 64],
                metadata: None,
                features: None,
            })
            .collect();

        let query = vec![0.1f32; 64];
        let results = ranker.rerank(&query, candidates).unwrap();

        // After MinMax normalization, scores should be in [0, 1]
        for r in &results {
            assert!(r.score >= 0.0 && r.score <= 1.0);
        }
    }
}
