//! ColBERT Late Interaction Reranking Example
//!
//! This example demonstrates how to use ColBERT for high-accuracy reranking.
//!
//! ColBERT uses token-level interactions (late interaction) which provides
//! better accuracy than traditional reranking approaches.
//!
//! Run with:
//! ```bash
//! cargo run --example colbert_reranking
//! ```

use vecstore::reranking::colbert::{ColBERTBatchReranker, ColBERTConfig, ColBERTReranker};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== ColBERT Late Interaction Reranking ===\n");

    // Configure ColBERT
    let config = ColBERTConfig {
        max_query_tokens: 32,
        max_doc_tokens: 128,
        embedding_dim: 128,
        ..Default::default()
    };

    println!("Config:");
    println!("  Max query tokens: {}", config.max_query_tokens);
    println!("  Max doc tokens: {}", config.max_doc_tokens);
    println!("  Embedding dim: {}\n", config.embedding_dim);

    // Create reranker
    let reranker = ColBERTReranker::new(config.clone())?;

    // Example 1: Single document reranking
    println!("Example 1: Single Document Reranking");
    println!("-------------------------------------");

    let query = "What is Rust programming language?";
    let document = "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.";

    println!("Query: {}", query);
    println!("Document: {}\n", document);

    // Encode query and document
    let query_tokens = reranker.encode_query(query).await?;
    let doc_tokens = reranker.encode_document(document).await?;

    println!("Query tokens: {}", query_tokens.num_tokens());
    println!("Document tokens: {}", doc_tokens.num_tokens());

    // Compute late interaction score
    let score = reranker.compute_score(&query_tokens, &doc_tokens)?;
    println!("ColBERT Score: {:.4}\n", score);

    // Example 2: Batch reranking
    println!("Example 2: Batch Reranking");
    println!("--------------------------");

    let query = "machine learning frameworks";
    let documents = vec![
        "TensorFlow is an open-source machine learning framework developed by Google.".to_string(),
        "PyTorch is a popular deep learning framework that provides flexibility and speed.".to_string(),
        "Scikit-learn is a machine learning library for Python that features various classification, regression and clustering algorithms.".to_string(),
        "Rust is a systems programming language focused on safety and performance.".to_string(),
        "JavaScript is a high-level programming language commonly used for web development.".to_string(),
    ];

    println!("Query: {}", query);
    println!("Documents to rerank: {}\n", documents.len());

    let mut batch_reranker = ColBERTBatchReranker::new(config)?;
    let reranked = batch_reranker.rerank(query, &documents, 3).await?;

    println!("Top 3 Results:");
    for (rank, (idx, score)) in reranked.iter().enumerate() {
        println!("  {}. Score: {:.4}", rank + 1, score);
        println!(
            "     Doc[{}]: {}",
            idx,
            &documents[*idx][..80.min(documents[*idx].len())]
        );
        println!();
    }

    // Example 3: Comparing different similarity metrics
    println!("Example 3: Similarity Metrics Comparison");
    println!("----------------------------------------");

    let query = "vector database";
    let doc = "A vector database is a specialized database designed to store and query high-dimensional vectors.";

    println!("Query: {}", query);
    println!("Document: {}\n", doc);

    for &metric in &[
        vecstore::reranking::colbert::SimilarityMetric::Cosine,
        vecstore::reranking::colbert::SimilarityMetric::DotProduct,
        vecstore::reranking::colbert::SimilarityMetric::L2,
    ] {
        let config = ColBERTConfig {
            similarity_metric: metric,
            ..Default::default()
        };

        let reranker = ColBERTReranker::new(config)?;
        let query_tokens = reranker.encode_query(query).await?;
        let doc_tokens = reranker.encode_document(doc).await?;
        let score = reranker.compute_score(&query_tokens, &doc_tokens)?;

        println!("{:?} Score: {:.4}", metric, score);
    }

    println!("\n=== ColBERT Example Complete ===");
    Ok(())
}
