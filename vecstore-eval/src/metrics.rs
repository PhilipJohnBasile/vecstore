//! RAG Evaluation Metrics
//!
//! Three core metrics for measuring RAG quality:
//!
//! 1. **Context Relevance**: Are retrieved contexts relevant to the query?
//! 2. **Answer Faithfulness**: Is the answer grounded in the context?
//! 3. **Answer Correctness**: How close is the answer to ground truth?

use crate::types::{EvaluationInput, Metric, MetricResult};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

// ============================================================================
// Trait Definitions
// ============================================================================

/// Trait for Large Language Models used as judges
///
/// Implement this trait to use any LLM (OpenAI, Anthropic, local models, etc.)
/// for LLM-as-judge evaluation.
pub trait LLM: Send + Sync {
    /// Generate text from a prompt
    fn generate(&self, prompt: &str) -> Result<String>;
}

/// Trait for embedding models
///
/// Implement this trait to use any embedding model for semantic similarity.
pub trait Embedder: Send + Sync {
    /// Embed text into a vector
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

// ============================================================================
// Context Relevance Metric (LLM-as-Judge)
// ============================================================================

/// Measures whether retrieved contexts are relevant to the query
///
/// Uses an LLM to judge whether each retrieved context is relevant
/// to answering the query. Score is the fraction of relevant contexts.
///
/// # Example
///
/// ```no_run
/// use vecstore_eval::{ContextRelevance, EvaluationInput, Metric};
/// # struct MyLLM;
/// # impl vecstore_eval::LLM for MyLLM {
/// #     fn generate(&self, prompt: &str) -> anyhow::Result<String> { Ok("Yes".to_string()) }
/// # }
///
/// let llm = Box::new(MyLLM);
/// let metric = ContextRelevance::new(llm);
///
/// let input = EvaluationInput {
///     query: "What is Rust?".to_string(),
///     contexts: vec![
///         "Rust is a systems programming language.".to_string(),
///         "Python is an interpreted language.".to_string(),
///     ],
///     answer: None,
///     ground_truth: None,
/// };
///
/// let result = metric.evaluate(&input)?;
/// assert!(result.score >= 0.0 && result.score <= 1.0);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct ContextRelevance {
    llm: Box<dyn LLM>,
}

impl ContextRelevance {
    /// Create a new context relevance metric
    pub fn new(llm: Box<dyn LLM>) -> Self {
        Self { llm }
    }

    /// Judge whether a single context is relevant
    fn is_relevant(&self, query: &str, context: &str) -> Result<bool> {
        let prompt = format!(
            "Query: {}\n\nContext: {}\n\n\
             Is this context relevant for answering the query? \
             Answer only 'Yes' or 'No'.",
            query, context
        );

        let response = self.llm.generate(&prompt)?;
        let normalized = response.trim().to_lowercase();

        Ok(normalized.contains("yes"))
    }
}

impl Metric for ContextRelevance {
    fn name(&self) -> &str {
        "context_relevance"
    }

    fn evaluate(&self, input: &EvaluationInput) -> Result<MetricResult> {
        if input.contexts.is_empty() {
            return Ok(MetricResult {
                metric_name: self.name().to_string(),
                score: 0.0,
                details: HashMap::new(),
            });
        }

        let mut relevant_count = 0;
        let mut context_relevance = Vec::new();

        for (i, context) in input.contexts.iter().enumerate() {
            let is_relevant = self.is_relevant(&input.query, context)?;
            if is_relevant {
                relevant_count += 1;
            }
            context_relevance.push((i, is_relevant));
        }

        let score = relevant_count as f32 / input.contexts.len() as f32;

        let mut details = HashMap::new();
        details.insert(
            "relevant_count".to_string(),
            serde_json::json!(relevant_count),
        );
        details.insert(
            "total_contexts".to_string(),
            serde_json::json!(input.contexts.len()),
        );
        details.insert(
            "context_relevance".to_string(),
            serde_json::json!(context_relevance),
        );

        Ok(MetricResult {
            metric_name: self.name().to_string(),
            score,
            details,
        })
    }
}

// ============================================================================
// Answer Faithfulness Metric (LLM-as-Judge)
// ============================================================================

/// Measures whether the answer is faithful to (supported by) the context
///
/// Uses an LLM to judge whether the generated answer is grounded in the
/// retrieved context (no hallucination). Score: 0.0-1.0.
///
/// # Example
///
/// ```no_run
/// use vecstore_eval::{AnswerFaithfulness, EvaluationInput, Metric};
/// # struct MyLLM;
/// # impl vecstore_eval::LLM for MyLLM {
/// #     fn generate(&self, prompt: &str) -> anyhow::Result<String> { Ok("1.0".to_string()) }
/// # }
///
/// let llm = Box::new(MyLLM);
/// let metric = AnswerFaithfulness::new(llm);
///
/// let input = EvaluationInput {
///     query: "What is Rust?".to_string(),
///     contexts: vec!["Rust is a systems programming language.".to_string()],
///     answer: Some("Rust is a systems language.".to_string()),
///     ground_truth: None,
/// };
///
/// let result = metric.evaluate(&input)?;
/// assert!(result.score >= 0.0 && result.score <= 1.0);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct AnswerFaithfulness {
    llm: Box<dyn LLM>,
}

impl AnswerFaithfulness {
    /// Create a new answer faithfulness metric
    pub fn new(llm: Box<dyn LLM>) -> Self {
        Self { llm }
    }
}

impl Metric for AnswerFaithfulness {
    fn name(&self) -> &str {
        "answer_faithfulness"
    }

    fn evaluate(&self, input: &EvaluationInput) -> Result<MetricResult> {
        let answer = input
            .answer
            .as_ref()
            .ok_or_else(|| anyhow!("Answer required for faithfulness metric"))?;

        if input.contexts.is_empty() {
            return Ok(MetricResult {
                metric_name: self.name().to_string(),
                score: 0.0,
                details: HashMap::new(),
            });
        }

        let context = input.contexts.join("\n\n");

        let prompt = format!(
            "Context:\n{}\n\nAnswer:\n{}\n\n\
             Is the answer fully supported by the context? \
             Rate the faithfulness from 0.0 (completely unfaithful/hallucinated) \
             to 1.0 (fully faithful/grounded). \
             Respond with only a number between 0.0 and 1.0.",
            context, answer
        );

        let response = self.llm.generate(&prompt)?;

        // Parse score from response
        let score = response
            .trim()
            .split_whitespace()
            .next()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        let mut details = HashMap::new();
        details.insert("llm_response".to_string(), serde_json::json!(response));

        Ok(MetricResult {
            metric_name: self.name().to_string(),
            score,
            details,
        })
    }
}

// ============================================================================
// Answer Correctness Metric (Embedding Similarity)
// ============================================================================

/// Measures semantic similarity between generated answer and ground truth
///
/// Uses embeddings to calculate cosine similarity between the generated
/// answer and the ground truth answer. Score: 0.0-1.0.
///
/// # Example
///
/// ```no_run
/// use vecstore_eval::{AnswerCorrectness, EvaluationInput, Metric};
/// # struct MyEmbedder;
/// # impl vecstore_eval::Embedder for MyEmbedder {
/// #     fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> { Ok(vec![1.0, 0.0, 0.0]) }
/// # }
///
/// let embedder = Box::new(MyEmbedder);
/// let metric = AnswerCorrectness::new(embedder);
///
/// let input = EvaluationInput {
///     query: "What is Rust?".to_string(),
///     contexts: vec![],
///     answer: Some("Rust is a systems programming language.".to_string()),
///     ground_truth: Some("Rust is a memory-safe systems language.".to_string()),
/// };
///
/// let result = metric.evaluate(&input)?;
/// assert!(result.score >= 0.0 && result.score <= 1.0);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct AnswerCorrectness {
    embedder: Box<dyn Embedder>,
}

impl AnswerCorrectness {
    /// Create a new answer correctness metric
    pub fn new(embedder: Box<dyn Embedder>) -> Self {
        Self { embedder }
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }
}

impl Metric for AnswerCorrectness {
    fn name(&self) -> &str {
        "answer_correctness"
    }

    fn evaluate(&self, input: &EvaluationInput) -> Result<MetricResult> {
        let answer = input
            .answer
            .as_ref()
            .ok_or_else(|| anyhow!("Answer required for correctness metric"))?;

        let ground_truth = input
            .ground_truth
            .as_ref()
            .ok_or_else(|| anyhow!("Ground truth required for correctness metric"))?;

        // Embed both texts
        let answer_embedding = self.embedder.embed(answer)?;
        let truth_embedding = self.embedder.embed(ground_truth)?;

        // Calculate cosine similarity
        let similarity = Self::cosine_similarity(&answer_embedding, &truth_embedding);

        // Normalize to 0-1 range (cosine similarity is -1 to 1)
        let score = ((similarity + 1.0) / 2.0).clamp(0.0, 1.0);

        let mut details = HashMap::new();
        details.insert("cosine_similarity".to_string(), serde_json::json!(similarity));
        details.insert(
            "answer_length".to_string(),
            serde_json::json!(answer.len()),
        );
        details.insert(
            "ground_truth_length".to_string(),
            serde_json::json!(ground_truth.len()),
        );

        Ok(MetricResult {
            metric_name: self.name().to_string(),
            score,
            details,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock LLM that always returns "Yes"
    struct MockLLMYes;
    impl LLM for MockLLMYes {
        fn generate(&self, _prompt: &str) -> Result<String> {
            Ok("Yes".to_string())
        }
    }

    // Mock LLM that returns a score
    struct MockLLMScore(f32);
    impl LLM for MockLLMScore {
        fn generate(&self, _prompt: &str) -> Result<String> {
            Ok(format!("{}", self.0))
        }
    }

    // Mock embedder that returns fixed vectors
    struct MockEmbedder;
    impl Embedder for MockEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            // Simple mock: use text length as a feature
            let len = text.len() as f32;
            Ok(vec![len / 100.0, 1.0, 0.5])
        }
    }

    #[test]
    fn test_context_relevance_all_relevant() {
        let metric = ContextRelevance::new(Box::new(MockLLMYes));
        let input = EvaluationInput {
            query: "What is Rust?".to_string(),
            contexts: vec![
                "Rust is a systems programming language.".to_string(),
                "Rust provides memory safety.".to_string(),
            ],
            answer: None,
            ground_truth: None,
        };

        let result = metric.evaluate(&input).unwrap();
        assert_eq!(result.score, 1.0);
        assert_eq!(result.metric_name, "context_relevance");
    }

    #[test]
    fn test_context_relevance_empty_contexts() {
        let metric = ContextRelevance::new(Box::new(MockLLMYes));
        let input = EvaluationInput {
            query: "What is Rust?".to_string(),
            contexts: vec![],
            answer: None,
            ground_truth: None,
        };

        let result = metric.evaluate(&input).unwrap();
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_answer_faithfulness() {
        let metric = AnswerFaithfulness::new(Box::new(MockLLMScore(0.8)));
        let input = EvaluationInput {
            query: "What is Rust?".to_string(),
            contexts: vec!["Rust is a systems programming language.".to_string()],
            answer: Some("Rust is a systems language.".to_string()),
            ground_truth: None,
        };

        let result = metric.evaluate(&input).unwrap();
        assert_eq!(result.score, 0.8);
        assert_eq!(result.metric_name, "answer_faithfulness");
    }

    #[test]
    fn test_answer_correctness() {
        let metric = AnswerCorrectness::new(Box::new(MockEmbedder));
        let input = EvaluationInput {
            query: "What is Rust?".to_string(),
            contexts: vec![],
            answer: Some("Rust is a systems programming language.".to_string()),
            ground_truth: Some("Rust is a memory-safe systems language.".to_string()),
        };

        let result = metric.evaluate(&input).unwrap();
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert_eq!(result.metric_name, "answer_correctness");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(AnswerCorrectness::cosine_similarity(&a, &b), 1.0);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(AnswerCorrectness::cosine_similarity(&a, &b), 0.0);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert_eq!(AnswerCorrectness::cosine_similarity(&a, &b), -1.0);
    }
}
