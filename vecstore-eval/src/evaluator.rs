//! Evaluation suite orchestrator

use crate::types::{EvaluationInput, EvaluationReport, Metric, MetricResult};
use anyhow::Result;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Orchestrates evaluation across multiple metrics
///
/// # Example
///
/// ```no_run
/// use vecstore_eval::{Evaluator, ContextRelevance, AnswerCorrectness, EvaluationInput};
/// # struct MyLLM;
/// # impl vecstore_eval::LLM for MyLLM {
/// #     fn generate(&self, _: &str) -> anyhow::Result<String> { Ok("Yes".to_string()) }
/// # }
/// # struct MyEmbedder;
/// # impl vecstore_eval::Embedder for MyEmbedder {
/// #     fn embed(&self, _: &str) -> anyhow::Result<Vec<f32>> { Ok(vec![1.0]) }
/// # }
///
/// let mut evaluator = Evaluator::new();
/// evaluator.add_metric(Box::new(ContextRelevance::new(Box::new(MyLLM))));
/// evaluator.add_metric(Box::new(AnswerCorrectness::new(Box::new(MyEmbedder))));
///
/// let input = EvaluationInput {
///     query: "What is Rust?".to_string(),
///     contexts: vec!["Rust is a systems programming language.".to_string()],
///     answer: Some("Rust is a systems language.".to_string()),
///     ground_truth: Some("Rust is a memory-safe systems language.".to_string()),
/// };
///
/// let report = evaluator.evaluate(&input)?;
/// println!("Overall score: {:.2}", report.overall_score);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct Evaluator {
    metrics: Vec<Box<dyn Metric>>,
}

impl Evaluator {
    /// Create a new evaluator with no metrics
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    /// Add a metric to the evaluator
    pub fn add_metric(&mut self, metric: Box<dyn Metric>) {
        self.metrics.push(metric);
    }

    /// Evaluate a single input with all metrics
    pub fn evaluate(&self, input: &EvaluationInput) -> Result<EvaluationReport> {
        let mut results = Vec::new();
        let mut metric_scores = HashMap::new();
        let mut total_score = 0.0;
        let mut count = 0;

        for metric in &self.metrics {
            let result = metric.evaluate(input)?;
            total_score += result.score;
            count += 1;

            metric_scores.insert(result.metric_name.clone(), result.score);
            results.push(result);
        }

        let overall_score = if count > 0 {
            total_score / count as f32
        } else {
            0.0
        };

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(EvaluationReport {
            overall_score,
            metric_scores,
            results,
            timestamp,
        })
    }

    /// Evaluate multiple inputs in batch
    ///
    /// Returns a vector of reports, one for each input.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vecstore_eval::{Evaluator, EvaluationInput};
    /// # let evaluator = Evaluator::new();
    /// let test_cases = vec![
    ///     EvaluationInput {
    ///         query: "What is Rust?".to_string(),
    ///         contexts: vec!["Rust is a systems programming language.".to_string()],
    ///         answer: Some("Rust is a systems language.".to_string()),
    ///         ground_truth: Some("Rust is a memory-safe systems language.".to_string()),
    ///     },
    ///     // More test cases...
    /// ];
    ///
    /// let reports = evaluator.evaluate_batch(&test_cases)?;
    /// for (i, report) in reports.iter().enumerate() {
    ///     println!("Test case {}: score = {:.2}", i, report.overall_score);
    /// }
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluate_batch(&self, inputs: &[EvaluationInput]) -> Result<Vec<EvaluationReport>> {
        inputs.iter().map(|input| self.evaluate(input)).collect()
    }

    /// Calculate aggregate statistics across multiple reports
    ///
    /// Returns average scores for each metric plus overall average.
    pub fn aggregate_reports(&self, reports: &[EvaluationReport]) -> AggregateStats {
        if reports.is_empty() {
            return AggregateStats {
                count: 0,
                average_overall_score: 0.0,
                average_metric_scores: HashMap::new(),
                min_score: 0.0,
                max_score: 0.0,
            };
        }

        let mut total_overall = 0.0;
        let mut metric_totals: HashMap<String, f32> = HashMap::new();
        let mut min_score = f32::MAX;
        let mut max_score = f32::MIN;

        for report in reports {
            total_overall += report.overall_score;
            min_score = min_score.min(report.overall_score);
            max_score = max_score.max(report.overall_score);

            for (name, score) in &report.metric_scores {
                *metric_totals.entry(name.clone()).or_insert(0.0) += score;
            }
        }

        let count = reports.len();
        let average_overall_score = total_overall / count as f32;

        let average_metric_scores = metric_totals
            .into_iter()
            .map(|(name, total)| (name, total / count as f32))
            .collect();

        AggregateStats {
            count,
            average_overall_score,
            average_metric_scores,
            min_score,
            max_score,
        }
    }

    /// Get the number of metrics in this evaluator
    pub fn metric_count(&self) -> usize {
        self.metrics.len()
    }

    /// Get the names of all metrics in this evaluator
    pub fn metric_names(&self) -> Vec<String> {
        self.metrics.iter().map(|m| m.name().to_string()).collect()
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate statistics across multiple evaluation reports
#[derive(Debug, Clone)]
pub struct AggregateStats {
    /// Number of reports aggregated
    pub count: usize,

    /// Average overall score across all reports
    pub average_overall_score: f32,

    /// Average score for each metric
    pub average_metric_scores: HashMap<String, f32>,

    /// Minimum overall score
    pub min_score: f32,

    /// Maximum overall score
    pub max_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{AnswerCorrectness, ContextRelevance, Embedder, LLM};

    struct MockLLM;
    impl LLM for MockLLM {
        fn generate(&self, _prompt: &str) -> Result<String> {
            Ok("Yes".to_string())
        }
    }

    struct MockEmbedder;
    impl Embedder for MockEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            let len = text.len() as f32;
            Ok(vec![len / 100.0, 1.0])
        }
    }

    #[test]
    fn test_evaluator_new() {
        let evaluator = Evaluator::new();
        assert_eq!(evaluator.metric_count(), 0);
    }

    #[test]
    fn test_evaluator_add_metric() {
        let mut evaluator = Evaluator::new();
        evaluator.add_metric(Box::new(ContextRelevance::new(Box::new(MockLLM))));
        assert_eq!(evaluator.metric_count(), 1);
    }

    #[test]
    fn test_evaluator_evaluate() {
        let mut evaluator = Evaluator::new();
        evaluator.add_metric(Box::new(ContextRelevance::new(Box::new(MockLLM))));
        evaluator.add_metric(Box::new(AnswerCorrectness::new(Box::new(MockEmbedder))));

        let input = EvaluationInput {
            query: "What is Rust?".to_string(),
            contexts: vec!["Rust is a systems programming language.".to_string()],
            answer: Some("Rust is a systems language.".to_string()),
            ground_truth: Some("Rust is a memory-safe systems language.".to_string()),
        };

        let report = evaluator.evaluate(&input).unwrap();
        assert_eq!(report.results.len(), 2);
        assert!(report.overall_score >= 0.0 && report.overall_score <= 1.0);
        assert_eq!(report.metric_scores.len(), 2);
    }

    #[test]
    fn test_evaluator_batch() {
        let mut evaluator = Evaluator::new();
        evaluator.add_metric(Box::new(ContextRelevance::new(Box::new(MockLLM))));

        let inputs = vec![
            EvaluationInput {
                query: "Query 1".to_string(),
                contexts: vec!["Context 1".to_string()],
                answer: None,
                ground_truth: None,
            },
            EvaluationInput {
                query: "Query 2".to_string(),
                contexts: vec!["Context 2".to_string()],
                answer: None,
                ground_truth: None,
            },
        ];

        let reports = evaluator.evaluate_batch(&inputs).unwrap();
        assert_eq!(reports.len(), 2);
    }

    #[test]
    fn test_aggregate_reports() {
        let reports = vec![
            EvaluationReport {
                overall_score: 0.8,
                metric_scores: [("metric1".to_string(), 0.8)].iter().cloned().collect(),
                results: vec![],
                timestamp: 0,
            },
            EvaluationReport {
                overall_score: 0.6,
                metric_scores: [("metric1".to_string(), 0.6)].iter().cloned().collect(),
                results: vec![],
                timestamp: 0,
            },
        ];

        let evaluator = Evaluator::new();
        let stats = evaluator.aggregate_reports(&reports);

        assert_eq!(stats.count, 2);
        assert!((stats.average_overall_score - 0.7).abs() < 0.001); // Floating point tolerance
        assert_eq!(stats.min_score, 0.6);
        assert_eq!(stats.max_score, 0.8);
    }

    #[test]
    fn test_aggregate_empty() {
        let evaluator = Evaluator::new();
        let stats = evaluator.aggregate_reports(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.average_overall_score, 0.0);
    }

    #[test]
    fn test_metric_names() {
        let mut evaluator = Evaluator::new();
        evaluator.add_metric(Box::new(ContextRelevance::new(Box::new(MockLLM))));
        evaluator.add_metric(Box::new(AnswerCorrectness::new(Box::new(MockEmbedder))));

        let names = evaluator.metric_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"context_relevance".to_string()));
        assert!(names.contains(&"answer_correctness".to_string()));
    }
}
