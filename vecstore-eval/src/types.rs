//! Core types for RAG evaluation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Input data for RAG evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationInput {
    /// The user's query/question
    pub query: String,

    /// Retrieved context documents
    pub contexts: Vec<String>,

    /// Generated answer (optional, required for faithfulness/correctness)
    pub answer: Option<String>,

    /// Ground truth answer (optional, required for correctness)
    pub ground_truth: Option<String>,
}

/// Result of evaluating a single metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    /// Name of the metric
    pub metric_name: String,

    /// Score (typically 0.0-1.0)
    pub score: f32,

    /// Additional details/explanations
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub details: HashMap<String, serde_json::Value>,
}

/// Complete evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Overall score (average of all metrics)
    pub overall_score: f32,

    /// Individual metric scores
    pub metric_scores: HashMap<String, f32>,

    /// Detailed results for each metric
    pub results: Vec<MetricResult>,

    /// Timestamp of evaluation (Unix timestamp)
    pub timestamp: u64,
}

/// Trait for evaluation metrics
pub trait Metric: Send + Sync {
    /// Name of this metric
    fn name(&self) -> &str;

    /// Evaluate the metric on the given input
    fn evaluate(&self, input: &EvaluationInput) -> anyhow::Result<MetricResult>;
}
