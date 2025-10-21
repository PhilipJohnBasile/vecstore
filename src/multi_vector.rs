//! Multi-Vector Document Storage (ColBERT-style)
//!
//! This module supports documents with multiple embeddings per document,
//! enabling late interaction models like ColBERT.
//!
//! ## Key Concepts
//!
//! - **Token-level embeddings**: Each token gets its own embedding
//! - **MaxSim**: Relevance score = max similarity across all token pairs
//! - **Late interaction**: Similarity computed at query time, not indexing time
//!
//! ## Architecture
//!
//! ```text
//! Document: "machine learning"
//!     │
//!     ▼
//! ┌─────────┬─────────┐
//! │ machine │ learning│
//! └────┬────┴────┬────┘
//!      │         │
//!   embed()   embed()
//!      │         │
//!      ▼         ▼
//!   [0.1,…]  [0.2,…]
//!
//! Query: "deep learning"
//!   MaxSim = max(sim(query, machine), sim(query, learning))
//! ```
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::multi_vector::{MultiVectorDoc, MultiVectorIndex, MaxSimAggregation};
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut index = MultiVectorIndex::new(128); // 128-dim embeddings
//!
//! // Add document with multiple token embeddings
//! let doc = MultiVectorDoc::new(
//!     "doc1",
//!     vec![
//!         vec![0.1; 128],  // "machine" embedding
//!         vec![0.2; 128],  // "learning" embedding
//!     ],
//!     serde_json::json!({"title": "ML Guide"}),
//! );
//!
//! index.add(doc)?;
//!
//! // Query with MaxSim aggregation
//! let query_tokens = vec![vec![0.15; 128]];
//! let results = index.search(&query_tokens, 10)?;
//!
//! println!("Found {} results", results.len());
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-vector document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorDoc {
    /// Document ID
    pub id: String,
    /// Multiple embeddings (one per token/chunk)
    pub vectors: Vec<Vec<f32>>,
    /// Metadata
    pub metadata: serde_json::Value,
}

impl MultiVectorDoc {
    /// Create a new multi-vector document
    pub fn new(id: impl Into<String>, vectors: Vec<Vec<f32>>, metadata: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            vectors,
            metadata,
        }
    }

    /// Get number of vectors
    pub fn num_vectors(&self) -> usize {
        self.vectors.len()
    }

    /// Get vector dimension
    pub fn dimension(&self) -> usize {
        self.vectors.first().map(|v| v.len()).unwrap_or(0)
    }

    /// Validate that all vectors have the same dimension
    pub fn validate(&self) -> Result<()> {
        if self.vectors.is_empty() {
            return Err(anyhow!("Document has no vectors"));
        }

        let dim = self.dimension();
        for (i, vec) in self.vectors.iter().enumerate() {
            if vec.len() != dim {
                return Err(anyhow!(
                    "Vector {} has dimension {}, expected {}",
                    i,
                    vec.len(),
                    dim
                ));
            }
        }

        Ok(())
    }
}

/// Aggregation method for multi-vector scores
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Maximum similarity (ColBERT)
    MaxSim,
    /// Average similarity
    AvgSim,
    /// Sum of similarities
    SumSim,
    /// First token only
    FirstToken,
}

/// Multi-vector index
pub struct MultiVectorIndex {
    /// Expected vector dimension
    dimension: usize,
    /// Documents indexed by ID
    documents: HashMap<String, MultiVectorDoc>,
    /// Flattened token index for fast retrieval
    /// Maps flat token ID -> (doc_id, token_index)
    token_index: Vec<(String, usize)>,
    /// All token vectors (flattened)
    token_vectors: Vec<Vec<f32>>,
    /// Aggregation method
    aggregation: AggregationMethod,
}

impl MultiVectorIndex {
    /// Create a new multi-vector index
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            documents: HashMap::new(),
            token_index: Vec::new(),
            token_vectors: Vec::new(),
            aggregation: AggregationMethod::MaxSim,
        }
    }

    /// Set aggregation method
    pub fn with_aggregation(mut self, aggregation: AggregationMethod) -> Self {
        self.aggregation = aggregation;
        self
    }

    /// Add a document
    pub fn add(&mut self, doc: MultiVectorDoc) -> Result<()> {
        doc.validate()?;

        if doc.dimension() != self.dimension {
            return Err(anyhow!(
                "Document dimension {} doesn't match index dimension {}",
                doc.dimension(),
                self.dimension
            ));
        }

        let doc_id = doc.id.clone();

        // Add all token vectors to flat index
        for (token_idx, vector) in doc.vectors.iter().enumerate() {
            self.token_index.push((doc_id.clone(), token_idx));
            self.token_vectors.push(vector.clone());
        }

        self.documents.insert(doc_id, doc);

        Ok(())
    }

    /// Search using multi-vector query
    pub fn search(&self, query_vectors: &[Vec<f32>], k: usize) -> Result<Vec<(String, f32)>> {
        if query_vectors.is_empty() {
            return Err(anyhow!("Query has no vectors"));
        }

        // Validate query dimensions
        for qv in query_vectors {
            if qv.len() != self.dimension {
                return Err(anyhow!(
                    "Query dimension {} doesn't match index dimension {}",
                    qv.len(),
                    self.dimension
                ));
            }
        }

        // Compute scores for each document
        let mut doc_scores: HashMap<String, Vec<f32>> = HashMap::new();

        // For each query vector
        for query_vec in query_vectors {
            // Compute similarity with all document tokens
            for (token_id, (doc_id, _token_idx)) in self.token_index.iter().enumerate() {
                let token_vec = &self.token_vectors[token_id];
                let sim = cosine_similarity(query_vec, token_vec);

                doc_scores
                    .entry(doc_id.clone())
                    .or_insert_with(Vec::new)
                    .push(sim);
            }
        }

        // Aggregate scores per document
        let mut results: Vec<(String, f32)> = doc_scores
            .into_iter()
            .map(|(doc_id, sims)| {
                let score = match self.aggregation {
                    AggregationMethod::MaxSim => {
                        sims.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                    }
                    AggregationMethod::AvgSim => sims.iter().sum::<f32>() / sims.len() as f32,
                    AggregationMethod::SumSim => sims.iter().sum(),
                    AggregationMethod::FirstToken => sims.first().copied().unwrap_or(0.0),
                };
                (doc_id, score)
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Get a document by ID
    pub fn get(&self, doc_id: &str) -> Option<&MultiVectorDoc> {
        self.documents.get(doc_id)
    }

    /// Get number of documents
    pub fn num_documents(&self) -> usize {
        self.documents.len()
    }

    /// Get total number of token vectors
    pub fn num_tokens(&self) -> usize {
        self.token_vectors.len()
    }

    /// Get index statistics
    pub fn stats(&self) -> MultiVectorStats {
        let avg_tokens_per_doc = if !self.documents.is_empty() {
            self.num_tokens() as f32 / self.num_documents() as f32
        } else {
            0.0
        };

        MultiVectorStats {
            num_documents: self.num_documents(),
            num_tokens: self.num_tokens(),
            dimension: self.dimension,
            avg_tokens_per_doc,
            aggregation: self.aggregation,
        }
    }
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct MultiVectorStats {
    pub num_documents: usize,
    pub num_tokens: usize,
    pub dimension: usize,
    pub avg_tokens_per_doc: f32,
    pub aggregation: AggregationMethod,
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// ColBERT-specific utilities
pub mod colbert {
    use super::*;

    /// ColBERT query encoder (wraps multi-vector with MaxSim)
    pub struct ColBERTQuery {
        /// Query token embeddings
        pub tokens: Vec<Vec<f32>>,
    }

    impl ColBERTQuery {
        /// Create a new ColBERT query
        pub fn new(tokens: Vec<Vec<f32>>) -> Self {
            Self { tokens }
        }

        /// Compute MaxSim score against a document
        pub fn score(&self, doc: &MultiVectorDoc) -> f32 {
            if self.tokens.is_empty() || doc.vectors.is_empty() {
                return 0.0;
            }

            let mut total_score = 0.0;

            // For each query token, find max similarity with any doc token
            for query_token in &self.tokens {
                let max_sim = doc
                    .vectors
                    .iter()
                    .map(|doc_token| cosine_similarity(query_token, doc_token))
                    .fold(f32::NEG_INFINITY, f32::max);

                total_score += max_sim;
            }

            total_score
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_vector_doc_creation() {
        let doc = MultiVectorDoc::new(
            "doc1",
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            serde_json::json!({}),
        );

        assert_eq!(doc.id, "doc1");
        assert_eq!(doc.num_vectors(), 2);
        assert_eq!(doc.dimension(), 2);
    }

    #[test]
    fn test_doc_validation() {
        let valid_doc = MultiVectorDoc::new(
            "doc1",
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            serde_json::json!({}),
        );
        assert!(valid_doc.validate().is_ok());

        let invalid_doc = MultiVectorDoc::new(
            "doc2",
            vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]], // Different dimensions
            serde_json::json!({}),
        );
        assert!(invalid_doc.validate().is_err());
    }

    #[test]
    fn test_index_add_and_get() {
        let mut index = MultiVectorIndex::new(2);

        let doc = MultiVectorDoc::new(
            "doc1",
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            serde_json::json!({}),
        );

        assert!(index.add(doc.clone()).is_ok());
        assert_eq!(index.num_documents(), 1);
        assert_eq!(index.num_tokens(), 2);

        let retrieved = index.get("doc1").unwrap();
        assert_eq!(retrieved.id, "doc1");
    }

    #[test]
    fn test_multi_vector_search_maxsim() {
        let mut index = MultiVectorIndex::new(2).with_aggregation(AggregationMethod::MaxSim);

        // Add documents
        let doc1 = MultiVectorDoc::new(
            "doc1",
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            serde_json::json!({}),
        );
        let doc2 = MultiVectorDoc::new(
            "doc2",
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
            serde_json::json!({}),
        );

        index.add(doc1).unwrap();
        index.add(doc2).unwrap();

        // Query
        let query = vec![vec![1.0, 0.0]];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // doc1 should rank higher (exact match with first token)
        assert_eq!(results[0].0, "doc1");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&c, &d) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_colbert_query() {
        use colbert::*;

        let query = ColBERTQuery::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        let doc = MultiVectorDoc::new(
            "doc1",
            vec![vec![1.0, 0.0], vec![0.5, 0.5]],
            serde_json::json!({}),
        );

        let score = query.score(&doc);
        assert!(score > 0.0);
    }

    #[test]
    fn test_index_stats() {
        let mut index = MultiVectorIndex::new(128);

        let doc1 = MultiVectorDoc::new(
            "doc1",
            vec![vec![0.0; 128], vec![0.1; 128]],
            serde_json::json!({}),
        );
        let doc2 = MultiVectorDoc::new(
            "doc2",
            vec![vec![0.2; 128], vec![0.3; 128], vec![0.4; 128]],
            serde_json::json!({}),
        );

        index.add(doc1).unwrap();
        index.add(doc2).unwrap();

        let stats = index.stats();
        assert_eq!(stats.num_documents, 2);
        assert_eq!(stats.num_tokens, 5); // 2 + 3
        assert_eq!(stats.dimension, 128);
        assert!((stats.avg_tokens_per_doc - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_aggregation_methods() {
        let mut index = MultiVectorIndex::new(2);

        let doc = MultiVectorDoc::new(
            "doc1",
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            serde_json::json!({}),
        );
        index.add(doc).unwrap();

        // Test MaxSim
        index.aggregation = AggregationMethod::MaxSim;
        let query = vec![vec![1.0, 0.0]];
        let results = index.search(&query, 1).unwrap();
        assert!(results[0].1 > 0.9); // Should be close to 1.0

        // Test AvgSim
        index.aggregation = AggregationMethod::AvgSim;
        let results = index.search(&query, 1).unwrap();
        assert!(results[0].1 > 0.0 && results[0].1 < 1.0); // Average of 1.0 and 0.0
    }
}
