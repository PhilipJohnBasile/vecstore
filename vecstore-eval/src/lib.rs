//! # VecStore Evaluation Toolkit
//!
//! Measure and improve RAG (Retrieval-Augmented Generation) pipeline quality.
//!
//! ## Overview
//!
//! This crate provides metrics and evaluation tools for RAG systems:
//!
//! - **Context Relevance**: Are retrieved documents relevant to the query?
//! - **Answer Faithfulness**: Is the answer supported by the retrieved context?
//! - **Answer Correctness**: How similar is the answer to ground truth?
//!
//! ## Quick Start
//!
//! ```ignore
//! use vecstore_eval::{EvaluationInput, Evaluator, ContextRelevance, AnswerCorrectness};
//!
//! // Create evaluator with metrics
//! let mut evaluator = Evaluator::new();
//! evaluator.add_metric(Box::new(ContextRelevance::new(my_llm))); // my_llm: impl LLM
//! evaluator.add_metric(Box::new(AnswerCorrectness::new(my_embedder))); // my_embedder: impl Embedder
//!
//! // Evaluate a single test case
//! let input = EvaluationInput {
//!     query: "What is Rust?".to_string(),
//!     contexts: vec!["Rust is a systems programming language...".to_string()],
//!     answer: Some("Rust is a fast, safe systems language.".to_string()),
//!     ground_truth: Some("Rust is a memory-safe systems programming language.".to_string()),
//! };
//!
//! let report = evaluator.evaluate(&input)?;
//! println!("Overall score: {:.2}", report.overall_score);
//! ```
//!
//! ## Metrics
//!
//! ### Context Relevance (LLM-as-Judge)
//!
//! Measures whether retrieved contexts are relevant to answering the query.
//! Uses an LLM to judge relevance. Score: 0.0-1.0 (fraction of relevant contexts).
//!
//! ### Answer Faithfulness (LLM-as-Judge)
//!
//! Measures whether the answer is supported by the retrieved context (no hallucination).
//! Uses an LLM to judge faithfulness. Score: 0.0-1.0.
//!
//! ### Answer Correctness (Embedding Similarity)
//!
//! Measures semantic similarity between generated answer and ground truth.
//! Uses embeddings to calculate similarity. Score: 0.0-1.0.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │     RAG System Under Test           │
//! │  (Your VecStore + LLM pipeline)     │
//! └─────────────────────────────────────┘
//!          │
//!          ▼
//! ┌─────────────────────────────────────┐
//! │      Evaluator                      │
//! │  • Runs test cases                  │
//! │  • Aggregates metrics               │
//! └─────────────────────────────────────┘
//!          │
//!          ├──► Context Relevance Metric
//!          ├──► Answer Faithfulness Metric
//!          └──► Answer Correctness Metric
//!          │
//!          ▼
//! ┌─────────────────────────────────────┐
//! │    EvaluationReport                 │
//! │  • Overall score                    │
//! │  • Per-metric scores                │
//! │  • Detailed results                 │
//! └─────────────────────────────────────┘
//! ```

pub mod metrics;
pub mod types;
pub mod evaluator;

pub use types::{EvaluationInput, EvaluationReport, Metric};
pub use evaluator::Evaluator;
pub use metrics::{ContextRelevance, AnswerFaithfulness, AnswerCorrectness};

// Re-export for convenience
pub use metrics::{LLM, Embedder};
