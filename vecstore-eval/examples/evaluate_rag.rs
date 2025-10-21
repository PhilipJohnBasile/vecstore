//! RAG Evaluation Example
//!
//! Demonstrates how to evaluate a RAG system using vecstore-eval.
//!
//! Run with: cargo run --example evaluate_rag

use vecstore_eval::{
    AnswerCorrectness, AnswerFaithfulness, ContextRelevance, Embedder, EvaluationInput,
    Evaluator, LLM,
};
use anyhow::Result;

// ============================================================================
// Mock LLM (replace with real LLM in production)
// ============================================================================

struct MockLLM;

impl LLM for MockLLM {
    fn generate(&self, prompt: &str) -> Result<String> {
        // In production, call OpenAI, Anthropic, or local LLM
        // For demo, we simulate responses

        if prompt.contains("Is this context relevant") {
            // Context relevance check
            if prompt.contains("Rust") && prompt.contains("systems programming") {
                Ok("Yes".to_string())
            } else {
                Ok("No".to_string())
            }
        } else if prompt.contains("faithfulness") {
            // Answer faithfulness check
            Ok("0.9".to_string())
        } else {
            Ok("0.8".to_string())
        }
    }
}

// ============================================================================
// Mock Embedder (replace with real embeddings in production)
// ============================================================================

struct MockEmbedder;

impl Embedder for MockEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // In production, use actual embeddings (OpenAI, sentence-transformers, etc.)
        // For demo, we create simple mock embeddings

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut embedding = vec![0.0; 384]; // Typical embedding dimension

        // Simple hash-based mock embedding
        for (i, word) in words.iter().enumerate() {
            let hash = word.len() * (i + 1);
            embedding[hash % 384] += 1.0;
        }

        // Normalize
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut embedding {
                *val /= magnitude;
            }
        }

        Ok(embedding)
    }
}

// ============================================================================
// Main Example
// ============================================================================

fn main() -> Result<()> {
    println!("🎯 VecStore RAG Evaluation Example\n");
    println!("This example demonstrates how to evaluate RAG system quality.\n");

    // Step 1: Create evaluator with all metrics
    println!("Step 1: Setting up evaluator with 3 metrics...");
    let mut evaluator = Evaluator::new();

    // Add Context Relevance metric (LLM-as-judge)
    evaluator.add_metric(Box::new(ContextRelevance::new(Box::new(MockLLM))));

    // Add Answer Faithfulness metric (LLM-as-judge)
    evaluator.add_metric(Box::new(AnswerFaithfulness::new(Box::new(MockLLM))));

    // Add Answer Correctness metric (embedding similarity)
    evaluator.add_metric(Box::new(AnswerCorrectness::new(Box::new(MockEmbedder))));

    println!("   ✓ Added {} metrics\n", evaluator.metric_count());

    // Step 2: Evaluate a single test case
    println!("Step 2: Evaluating a single test case...");

    let test_case = EvaluationInput {
        query: "What is Rust and why is it popular?".to_string(),
        contexts: vec![
            "Rust is a systems programming language that runs blazingly fast, \
             prevents segfaults, and guarantees thread safety."
                .to_string(),
            "Rust has been voted the most loved programming language in the \
             Stack Overflow Developer Survey for several years."
                .to_string(),
        ],
        answer: Some(
            "Rust is a systems programming language known for its performance \
             and safety. It's popular because it prevents common bugs like \
             segfaults while maintaining C-like speed."
                .to_string(),
        ),
        ground_truth: Some(
            "Rust is a fast, memory-safe systems programming language that \
             has gained popularity due to its performance and safety guarantees."
                .to_string(),
        ),
    };

    let report = evaluator.evaluate(&test_case)?;

    println!("\n📊 Evaluation Results:");
    println!("   Overall Score: {:.2}", report.overall_score);
    println!("\n   Per-Metric Scores:");
    for (metric, score) in &report.metric_scores {
        println!("     • {}: {:.2}", metric, score);
    }

    // Step 3: Batch evaluation
    println!("\n\nStep 3: Batch evaluation of multiple test cases...");

    let test_cases = vec![
        EvaluationInput {
            query: "How does Rust handle memory management?".to_string(),
            contexts: vec![
                "Rust uses ownership and borrowing to manage memory without \
                 a garbage collector."
                    .to_string(),
            ],
            answer: Some("Rust manages memory through ownership rules.".to_string()),
            ground_truth: Some(
                "Rust uses an ownership system with borrowing rules to \
                 manage memory safely."
                    .to_string(),
            ),
        },
        EvaluationInput {
            query: "What are Rust's key features?".to_string(),
            contexts: vec![
                "Rust features include zero-cost abstractions, move semantics, \
                 guaranteed memory safety, and threads without data races."
                    .to_string(),
            ],
            answer: Some("Rust has memory safety, speed, and concurrency.".to_string()),
            ground_truth: Some(
                "Rust's key features are memory safety, zero-cost abstractions, \
                 and fearless concurrency."
                    .to_string(),
            ),
        },
        EvaluationInput {
            query: "Is Rust suitable for web development?".to_string(),
            contexts: vec![
                "Rust has several web frameworks like Actix, Rocket, and Axum \
                 for building web applications."
                    .to_string(),
            ],
            answer: Some(
                "Yes, Rust has web frameworks like Actix and Rocket.".to_string(),
            ),
            ground_truth: Some(
                "Rust is suitable for web development with frameworks like \
                 Actix and Rocket."
                    .to_string(),
            ),
        },
    ];

    let reports = evaluator.evaluate_batch(&test_cases)?;

    println!("   Evaluated {} test cases", reports.len());

    // Step 4: Aggregate statistics
    println!("\n   Calculating aggregate statistics...");
    let stats = evaluator.aggregate_reports(&reports);

    println!("\n📈 Aggregate Statistics:");
    println!("   Test Cases: {}", stats.count);
    println!("   Average Overall Score: {:.2}", stats.average_overall_score);
    println!("   Score Range: {:.2} - {:.2}", stats.min_score, stats.max_score);
    println!("\n   Average Per-Metric Scores:");
    for (metric, avg_score) in &stats.average_metric_scores {
        println!("     • {}: {:.2}", metric, avg_score);
    }

    // Step 5: Interpret results
    println!("\n\n💡 Interpretation:");
    println!("   • Context Relevance: Measures if retrieved docs are relevant");
    println!("   • Answer Faithfulness: Checks if answer is grounded in context");
    println!("   • Answer Correctness: Similarity to ground truth answer");
    println!("\n   Scores range from 0.0 (poor) to 1.0 (perfect)");
    println!("   Overall score > 0.7 is generally considered good for RAG systems.");

    if stats.average_overall_score >= 0.7 {
        println!("\n   ✅ Your RAG system is performing well!");
    } else if stats.average_overall_score >= 0.5 {
        println!("\n   ⚠️  Your RAG system needs improvement.");
    } else {
        println!("\n   ❌ Your RAG system needs significant work.");
    }

    println!("\n\n🎓 Next Steps:");
    println!("   1. Replace MockLLM with real LLM (OpenAI, Anthropic, etc.)");
    println!("   2. Replace MockEmbedder with real embeddings");
    println!("   3. Create test cases from your actual use cases");
    println!("   4. Use results to tune your RAG pipeline");
    println!("   5. Track improvements over time");

    println!("\n✅ Example complete!");

    Ok(())
}
