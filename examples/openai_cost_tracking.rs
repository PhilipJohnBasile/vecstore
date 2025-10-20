// OpenAI Embeddings - Cost Tracking and Optimization Example
//
// This example demonstrates cost estimation, tracking, and optimization strategies
// when using OpenAI embeddings at scale.
//
// Requirements:
// - Set OPENAI_API_KEY environment variable
// - Run with: cargo run --example openai_cost_tracking --features "embeddings,openai-embeddings"

#![cfg_attr(
    not(all(feature = "embeddings", feature = "openai-embeddings")),
    allow(dead_code, unused_imports)
)]

use anyhow::Result;

#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};

#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
use std::time::Instant;

#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
#[tokio::main]
async fn main() -> Result<()> {
    println!("=============================================================");
    println!("OpenAI Embeddings - Cost Tracking & Optimization");
    println!("=============================================================\n");

    // 1. Check for API key
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("❌ Error: OPENAI_API_KEY environment variable not set");
            eprintln!("\nPlease set your OpenAI API key:");
            eprintln!("  export OPENAI_API_KEY='your-api-key-here'\n");
            std::process::exit(1);
        }
    };

    println!("✓ API key found\n");

    // 2. Compare model costs
    println!("=============================================================");
    println!("Model Cost Comparison");
    println!("=============================================================\n");

    let models = vec![
        OpenAIModel::TextEmbedding3Small,
        OpenAIModel::TextEmbedding3Large,
        OpenAIModel::Ada002,
    ];

    let sample_texts = vec![
        "This is a sample document for cost estimation.",
        "Another document to embed and calculate costs.",
        "Testing the cost differences between models.",
    ];

    println!("Sample workload: {} documents", sample_texts.len());
    println!(
        "Average document length: {} chars\n",
        sample_texts.iter().map(|t| t.len()).sum::<usize>() / sample_texts.len()
    );

    println!(
        "{:<30} {:<12} {:<15} {:<20}",
        "Model", "Dimension", "Cost/1M tok", "Estimated Cost"
    );
    println!("{}", "-".repeat(80));

    for model in &models {
        // Create temporary embedder for cost estimation
        let embedder = OpenAIEmbedding::new(api_key.clone(), *model).await?;
        let cost = embedder.estimate_cost(&sample_texts);

        println!(
            "{:<30} {:<12} ${:<14.2} ${:<.8}",
            model.as_str(),
            model.dimension(),
            model.cost_per_million_tokens(),
            cost
        );
    }
    println!();

    // 3. Demonstrate batch size impact
    println!("=============================================================");
    println!("Batch Size Optimization");
    println!("=============================================================\n");

    let embedder = OpenAIEmbedding::new(api_key.clone(), OpenAIModel::TextEmbedding3Small)
        .await?
        .with_rate_limit(500);

    println!("Testing different batch sizes for 10 documents:\n");

    // Generate 10 sample documents
    let documents: Vec<String> = (0..10)
        .map(|i| {
            format!(
                "Document number {} with some sample content for embedding.",
                i
            )
        })
        .collect();
    let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();

    // Single embedding approach
    println!("Approach 1: Individual Embeddings (10 API calls)");
    let start = Instant::now();
    let individual_cost = embedder.estimate_cost(&doc_refs[..1]) * 10.0;
    println!("  • Estimated cost: ${:.8}", individual_cost);
    println!("  • Estimated time: ~{} seconds (with rate limiting)", 10);
    println!("  • API calls: 10");
    println!();

    // Batch embedding approach
    println!("Approach 2: Batch Embedding (1 API call)");
    let batch_cost = embedder.estimate_cost(&doc_refs);
    println!("  • Estimated cost: ${:.8}", batch_cost);
    println!("  • Estimated time: <1 second");
    println!("  • API calls: 1");
    println!();

    let time_saved = 9; // seconds
    let cost_same = batch_cost;
    println!("Savings:");
    println!("  • Time saved: ~{} seconds", time_saved);
    println!("  • Cost: Same (${:.8})", cost_same);
    println!("  • API call reduction: 90%");
    println!();

    // 4. Scale analysis
    println!("=============================================================");
    println!("Scale Analysis: Cost at Different Workload Sizes");
    println!("=============================================================\n");

    let workload_sizes = vec![100, 1_000, 10_000, 100_000, 1_000_000];
    let avg_doc_length = 500; // characters

    println!(
        "{:<15} {:<20} {:<20} {:<20}",
        "Documents", "Small Model", "Large Model", "Savings (vs Large)"
    );
    println!("{}", "-".repeat(80));

    for &size in &workload_sizes {
        // Estimate tokens: ~1 token per 4 chars
        let estimated_tokens = (size * avg_doc_length) / 4;

        let small_cost = estimated_tokens as f64
            * OpenAIModel::TextEmbedding3Small.cost_per_million_tokens()
            / 1_000_000.0;
        let large_cost = estimated_tokens as f64
            * OpenAIModel::TextEmbedding3Large.cost_per_million_tokens()
            / 1_000_000.0;
        let savings = large_cost - small_cost;

        println!(
            "{:<15} ${:<19.2} ${:<19.2} ${:<.2}",
            format_number(size),
            small_cost,
            large_cost,
            savings
        );
    }
    println!();

    // 5. Cost tracking simulation
    println!("=============================================================");
    println!("Real-time Cost Tracking Simulation");
    println!("=============================================================\n");

    struct CostTracker {
        total_cost: f64,
        total_documents: usize,
        total_tokens_estimate: usize,
    }

    impl CostTracker {
        fn new() -> Self {
            Self {
                total_cost: 0.0,
                total_documents: 0,
                total_tokens_estimate: 0,
            }
        }

        fn track_batch(&mut self, texts: &[&str], cost: f64) {
            let tokens = texts.iter().map(|t| t.len() / 4).sum::<usize>();
            self.total_cost += cost;
            self.total_documents += texts.len();
            self.total_tokens_estimate += tokens;
        }

        fn report(&self) {
            println!("Cost Tracker Summary:");
            println!(
                "  • Total documents processed: {}",
                format_number(self.total_documents)
            );
            println!(
                "  • Estimated tokens: {}",
                format_number(self.total_tokens_estimate)
            );
            println!("  • Total cost: ${:.6}", self.total_cost);
            println!(
                "  • Average cost per document: ${:.8}",
                self.total_cost / self.total_documents as f64
            );
        }
    }

    let mut tracker = CostTracker::new();

    // Simulate processing batches
    let batches = vec![
        vec!["Document 1", "Document 2", "Document 3"],
        vec!["Document 4", "Document 5"],
        vec!["Document 6", "Document 7", "Document 8", "Document 9"],
    ];

    for (i, batch) in batches.iter().enumerate() {
        let cost = embedder.estimate_cost(batch);
        tracker.track_batch(batch, cost);
        println!(
            "Batch {} processed: {} documents, ${:.8}",
            i + 1,
            batch.len(),
            cost
        );
    }

    println!();
    tracker.report();
    println!();

    // 6. Budget planning
    println!("=============================================================");
    println!("Budget Planning Calculator");
    println!("=============================================================\n");

    let monthly_budget = 100.0; // $100/month
    let avg_chars_per_doc = 1000;
    let estimated_tokens_per_doc = avg_chars_per_doc / 4;

    println!("Budget: ${}/month", monthly_budget);
    println!(
        "Average document: {} characters (~{} tokens)\n",
        avg_chars_per_doc, estimated_tokens_per_doc
    );

    for model in &models {
        let cost_per_doc =
            (estimated_tokens_per_doc as f64 * model.cost_per_million_tokens()) / 1_000_000.0;
        let docs_per_budget = (monthly_budget / cost_per_doc) as usize;
        let docs_per_day = docs_per_budget / 30;

        println!("{}:", model.as_str());
        println!("  • Cost per document: ${:.8}", cost_per_doc);
        println!(
            "  • Documents per month: {}",
            format_number(docs_per_budget)
        );
        println!("  • Documents per day: {}", format_number(docs_per_day));
        println!();
    }

    // 7. Optimization recommendations
    println!("=============================================================");
    println!("Cost Optimization Recommendations");
    println!("=============================================================\n");

    println!("1. Choose the Right Model:");
    println!("   • text-embedding-3-small: Best for most use cases");
    println!("   • 6.5x cheaper than text-embedding-3-large");
    println!("   • Only use Large if you need maximum accuracy\n");

    println!("2. Always Use Batch Processing:");
    println!("   • Batch up to 2048 texts per API call");
    println!("   • Same cost, but 90%+ fewer API calls");
    println!("   • Faster processing with rate limits\n");

    println!("3. Cache Embeddings:");
    println!("   • Store embeddings in VecStore for reuse");
    println!("   • Only re-embed when content changes");
    println!("   • Avoid duplicate embeddings\n");

    println!("4. Optimize Document Length:");
    println!("   • Split long documents into chunks");
    println!("   • Remove boilerplate/metadata before embedding");
    println!("   • Shorter texts = fewer tokens = lower cost\n");

    println!("5. Monitor and Track:");
    println!("   • Use estimate_cost() before large batches");
    println!("   • Track actual API usage in OpenAI dashboard");
    println!("   • Set budget alerts\n");

    println!("6. Rate Limiting:");
    println!("   • Default: 500 requests/minute");
    println!("   • Adjust based on your API tier");
    println!("   • Use .with_rate_limit() to configure\n");

    // 8. Example: Calculate costs for a real project
    println!("=============================================================");
    println!("Example Project: Documentation Search System");
    println!("=============================================================\n");

    let project_docs = 1_000; // 1000 documentation pages
    let avg_page_chars = 3_000; // 3000 chars per page
    let chunks_per_page = 3; // Split into 3 chunks
    let total_chunks = project_docs * chunks_per_page;
    let avg_chunk_chars = avg_page_chars / chunks_per_page;

    println!("Project Specs:");
    println!("  • Documentation pages: {}", format_number(project_docs));
    println!(
        "  • Average page length: {} characters",
        format_number(avg_page_chars)
    );
    println!("  • Chunks per page: {}", chunks_per_page);
    println!(
        "  • Total chunks to embed: {}\n",
        format_number(total_chunks)
    );

    let chunk_samples: Vec<String> = (0..total_chunks)
        .map(|_| "x".repeat(avg_chunk_chars))
        .collect();
    let chunk_refs: Vec<&str> = chunk_samples.iter().map(|s| s.as_str()).collect();

    let indexing_cost = embedder.estimate_cost(&chunk_refs);
    let queries_per_month = 10_000;
    let avg_query_chars = 50;
    let query_sample = "x".repeat(avg_query_chars);
    let query_cost_per = embedder.estimate_cost(&[query_sample.as_str()]);
    let monthly_query_cost = query_cost_per * queries_per_month as f64;

    println!("Cost Breakdown:");
    println!("  • One-time indexing: ${:.4}", indexing_cost);
    println!(
        "  • Monthly queries ({}/month): ${:.4}",
        format_number(queries_per_month),
        monthly_query_cost
    );
    println!(
        "  • Total first month: ${:.4}",
        indexing_cost + monthly_query_cost
    );
    println!("  • Ongoing monthly: ${:.4}\n", monthly_query_cost);

    println!("=============================================================");
    println!("✓ Cost Tracking Example Complete!");
    println!("=============================================================");

    Ok(())
}

// Helper function to format numbers with commas
#[cfg(all(feature = "embeddings", feature = "openai-embeddings"))]
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;

    for c in s.chars().rev() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(c);
        count += 1;
    }

    result.chars().rev().collect()
}

#[cfg(not(all(feature = "embeddings", feature = "openai-embeddings")))]
fn main() {
    eprintln!("This example requires the 'embeddings' and 'openai-embeddings' features.");
    eprintln!("Run with: cargo run --example openai_cost_tracking --features \"embeddings,openai-embeddings\"");
    std::process::exit(1);
}
