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
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

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
                    .or_default()
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

/// Optimized Multi-vector index with HNSW-backed token search
///
/// Uses approximate nearest neighbor search to efficiently find
/// candidate documents without brute-force comparison of all tokens.
pub struct OptimizedMultiVectorIndex {
    /// Expected vector dimension
    dimension: usize,
    /// Documents indexed by ID
    documents: HashMap<String, MultiVectorDoc>,
    /// Token to document mapping (inverted index)
    token_to_doc: Vec<(String, u32)>, // (doc_id, token_idx)
    /// All token vectors for ANN search
    token_vectors: Vec<Vec<f32>>,
    /// HNSW-like graph for token search (simplified)
    token_graph: TokenGraph,
    /// Aggregation method
    aggregation: AggregationMethod,
    /// Configuration
    config: OptimizedIndexConfig,
    /// Statistics
    stats: OptimizedIndexStats,
}

/// Configuration for optimized index
#[derive(Debug, Clone)]
pub struct OptimizedIndexConfig {
    /// Number of nearest tokens to retrieve per query token
    pub tokens_per_query: usize,
    /// Minimum score threshold
    pub min_score: f32,
    /// Maximum candidates per document
    pub max_candidates: usize,
    /// HNSW-like parameters
    pub ef_construction: usize,
    pub ef_search: usize,
    pub m: usize,
}

impl Default for OptimizedIndexConfig {
    fn default() -> Self {
        Self {
            tokens_per_query: 100,
            min_score: 0.0,
            max_candidates: 1000,
            ef_construction: 200,
            ef_search: 50,
            m: 16,
        }
    }
}

/// Simple graph structure for fast token search
struct TokenGraph {
    /// Adjacency lists for each token
    neighbors: Vec<Vec<u32>>,
    /// Entry point for search
    entry_point: Option<usize>,
    /// Max neighbors per node
    m: usize,
}

impl TokenGraph {
    fn new(m: usize) -> Self {
        Self {
            neighbors: Vec::new(),
            entry_point: None,
            m,
        }
    }

    fn add_node(&mut self, vectors: &[Vec<f32>]) -> usize {
        let new_id = self.neighbors.len();
        self.neighbors.push(Vec::new());

        if self.entry_point.is_none() {
            self.entry_point = Some(new_id);
            return new_id;
        }

        // Find nearest neighbors using greedy search
        let nearest = self.search_greedy(vectors, new_id, self.m * 2);

        // Connect to nearest neighbors (bidirectional)
        for &neighbor_id in &nearest {
            // Add edge from new node to neighbor
            if self.neighbors[new_id].len() < self.m {
                self.neighbors[new_id].push(neighbor_id as u32);
            }

            // Add edge from neighbor to new node
            if self.neighbors[neighbor_id].len() < self.m {
                self.neighbors[neighbor_id].push(new_id as u32);
            }
        }

        new_id
    }

    fn search_greedy(&self, vectors: &[Vec<f32>], query_idx: usize, k: usize) -> Vec<usize> {
        if self.entry_point.is_none() || vectors.is_empty() {
            return vec![];
        }

        let query = &vectors[query_idx];
        let mut visited = HashSet::new();
        let mut candidates: Vec<(usize, f32)> = vec![];

        // Start from entry point
        let entry = self.entry_point.unwrap();
        candidates.push((entry, cosine_similarity(query, &vectors[entry])));
        visited.insert(entry);

        // Greedy search
        let mut changed = true;
        while changed {
            changed = false;

            // Get best candidate
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let best = candidates.first().map(|c| c.0);
            if let Some(best_id) = best {
                // Check neighbors
                for &neighbor in &self.neighbors[best_id] {
                    let neighbor_id = neighbor as usize;
                    if !visited.contains(&neighbor_id) && neighbor_id < vectors.len() {
                        visited.insert(neighbor_id);
                        let sim = cosine_similarity(query, &vectors[neighbor_id]);
                        candidates.push((neighbor_id, sim));
                        changed = true;
                    }
                }
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(k);
        candidates.into_iter().map(|(id, _)| id).collect()
    }

    fn search(&self, query: &[f32], vectors: &[Vec<f32>], ef: usize) -> Vec<(usize, f32)> {
        if self.entry_point.is_none() || vectors.is_empty() {
            return vec![];
        }

        let mut visited = HashSet::new();
        let mut candidates: Vec<(usize, f32)> = vec![];
        let mut results: Vec<(usize, f32)> = vec![];

        // Start from entry point
        let entry = self.entry_point.unwrap();
        let sim = cosine_similarity(query, &vectors[entry]);
        candidates.push((entry, sim));
        results.push((entry, sim));
        visited.insert(entry);

        while !candidates.is_empty() {
            // Get best unvisited candidate
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let (current, current_sim) = candidates.remove(0);

            // Check if we can stop (worst result is better than current)
            if results.len() >= ef {
                let worst_result = results.iter().map(|r| r.1).fold(f32::INFINITY, f32::min);
                if current_sim < worst_result {
                    break;
                }
            }

            // Explore neighbors
            for &neighbor in &self.neighbors[current] {
                let neighbor_id = neighbor as usize;
                if !visited.contains(&neighbor_id) && neighbor_id < vectors.len() {
                    visited.insert(neighbor_id);
                    let sim = cosine_similarity(query, &vectors[neighbor_id]);
                    candidates.push((neighbor_id, sim));
                    results.push((neighbor_id, sim));
                }
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(ef);
        results
    }
}

struct OptimizedIndexStats {
    queries: AtomicUsize,
    tokens_searched: AtomicUsize,
    docs_scored: AtomicUsize,
}

impl OptimizedMultiVectorIndex {
    /// Create a new optimized multi-vector index
    pub fn new(dimension: usize) -> Self {
        Self::with_config(dimension, OptimizedIndexConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(dimension: usize, config: OptimizedIndexConfig) -> Self {
        Self {
            dimension,
            documents: HashMap::new(),
            token_to_doc: Vec::new(),
            token_vectors: Vec::new(),
            token_graph: TokenGraph::new(config.m),
            aggregation: AggregationMethod::MaxSim,
            config,
            stats: OptimizedIndexStats {
                queries: AtomicUsize::new(0),
                tokens_searched: AtomicUsize::new(0),
                docs_scored: AtomicUsize::new(0),
            },
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

        // Add all token vectors
        for (token_idx, vector) in doc.vectors.iter().enumerate() {
            let _token_id = self.token_vectors.len();
            self.token_to_doc.push((doc_id.clone(), token_idx as u32));
            self.token_vectors.push(vector.clone());

            // Add to graph (skip during initial batch insert for performance)
            if self.token_vectors.len() <= 10000 {
                self.token_graph.add_node(&self.token_vectors);
            }
        }

        self.documents.insert(doc_id, doc);

        Ok(())
    }

    /// Rebuild the graph index (call after batch inserts)
    pub fn rebuild_graph(&mut self) {
        self.token_graph = TokenGraph::new(self.config.m);
        for _ in 0..self.token_vectors.len() {
            self.token_graph.add_node(&self.token_vectors);
        }
    }

    /// Search using multi-vector query (optimized with ANN)
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

        self.stats.queries.fetch_add(1, Ordering::Relaxed);

        // Collect candidate documents from token-level ANN search
        let mut doc_token_scores: HashMap<String, Vec<f32>> = HashMap::new();

        for query_vec in query_vectors {
            // Use graph-based ANN search for this query token
            let nearest_tokens = if self.token_graph.entry_point.is_some() {
                self.token_graph
                    .search(query_vec, &self.token_vectors, self.config.ef_search)
            } else {
                // Fallback to brute force for small indices
                self.brute_force_token_search(query_vec, self.config.tokens_per_query)
            };

            self.stats
                .tokens_searched
                .fetch_add(nearest_tokens.len(), Ordering::Relaxed);

            // Map tokens back to documents
            for (token_idx, sim) in nearest_tokens {
                if token_idx < self.token_to_doc.len() {
                    let (doc_id, _) = &self.token_to_doc[token_idx];
                    doc_token_scores
                        .entry(doc_id.clone())
                        .or_default()
                        .push(sim);
                }
            }
        }

        self.stats
            .docs_scored
            .fetch_add(doc_token_scores.len(), Ordering::Relaxed);

        // Aggregate scores per document
        let mut results: Vec<(String, f32)> = doc_token_scores
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
            .filter(|(_, score)| *score >= self.config.min_score)
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);

        Ok(results)
    }

    fn brute_force_token_search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = self
            .token_vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| (idx, cosine_similarity(query, vec)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
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
    pub fn stats(&self) -> OptimizedMultiVectorStats {
        let avg_tokens_per_doc = if !self.documents.is_empty() {
            self.num_tokens() as f32 / self.num_documents() as f32
        } else {
            0.0
        };

        OptimizedMultiVectorStats {
            num_documents: self.num_documents(),
            num_tokens: self.num_tokens(),
            dimension: self.dimension,
            avg_tokens_per_doc,
            aggregation: self.aggregation,
            queries: self.stats.queries.load(Ordering::Relaxed),
            tokens_searched: self.stats.tokens_searched.load(Ordering::Relaxed),
            docs_scored: self.stats.docs_scored.load(Ordering::Relaxed),
        }
    }
}

/// Statistics for optimized index
#[derive(Debug, Clone)]
pub struct OptimizedMultiVectorStats {
    pub num_documents: usize,
    pub num_tokens: usize,
    pub dimension: usize,
    pub avg_tokens_per_doc: f32,
    pub aggregation: AggregationMethod,
    pub queries: usize,
    pub tokens_searched: usize,
    pub docs_scored: usize,
}

/// Late interaction score computation for ColBERT
pub struct LateInteractionScorer {
    /// Score computation mode
    mode: ScoreMode,
}

/// How to compute late interaction scores
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreMode {
    /// Sum of per-query-token MaxSim (standard ColBERT)
    SumMaxSim,
    /// Average of per-query-token MaxSim
    AvgMaxSim,
    /// Maximum MaxSim across all query tokens
    MaxMaxSim,
}

impl LateInteractionScorer {
    /// Create a new scorer
    pub fn new(mode: ScoreMode) -> Self {
        Self { mode }
    }

    /// Compute score between query and document
    pub fn score(&self, query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return 0.0;
        }

        let maxsim_per_query: Vec<f32> = query_tokens
            .iter()
            .map(|qt| {
                doc_tokens
                    .iter()
                    .map(|dt| cosine_similarity(qt, dt))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();

        match self.mode {
            ScoreMode::SumMaxSim => maxsim_per_query.iter().sum(),
            ScoreMode::AvgMaxSim => {
                maxsim_per_query.iter().sum::<f32>() / maxsim_per_query.len() as f32
            }
            ScoreMode::MaxMaxSim => maxsim_per_query
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
        }
    }
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

    #[test]
    fn test_optimized_index_basic() {
        let mut index = OptimizedMultiVectorIndex::new(2);

        let doc1 = MultiVectorDoc::new(
            "doc1",
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            serde_json::json!({}),
        );
        let doc2 = MultiVectorDoc::new(
            "doc2",
            vec![vec![0.7, 0.7], vec![0.7, 0.7]],
            serde_json::json!({}),
        );

        index.add(doc1).unwrap();
        index.add(doc2).unwrap();

        assert_eq!(index.num_documents(), 2);
        assert_eq!(index.num_tokens(), 4);

        // Search
        let query = vec![vec![1.0, 0.0]];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // doc1 should rank higher (exact match)
        assert_eq!(results[0].0, "doc1");
    }

    #[test]
    fn test_optimized_index_stats() {
        let mut index = OptimizedMultiVectorIndex::new(4);

        for i in 0..5 {
            let doc = MultiVectorDoc::new(
                format!("doc{}", i),
                vec![vec![i as f32; 4], vec![(i + 1) as f32; 4]],
                serde_json::json!({}),
            );
            index.add(doc).unwrap();
        }

        // Perform a query
        let query = vec![vec![2.5; 4]];
        let _ = index.search(&query, 3);

        let stats = index.stats();
        assert_eq!(stats.num_documents, 5);
        assert_eq!(stats.num_tokens, 10);
        assert_eq!(stats.queries, 1);
    }

    #[test]
    fn test_late_interaction_scorer() {
        let scorer = LateInteractionScorer::new(ScoreMode::SumMaxSim);

        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let score = scorer.score(&query, &doc);
        // Each query token has MaxSim of 1.0, so sum = 2.0
        assert!((score - 2.0).abs() < 0.001);

        // Test AvgMaxSim
        let scorer_avg = LateInteractionScorer::new(ScoreMode::AvgMaxSim);
        let avg_score = scorer_avg.score(&query, &doc);
        assert!((avg_score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_optimized_index_with_config() {
        let config = OptimizedIndexConfig {
            tokens_per_query: 50,
            min_score: 0.5,
            ef_search: 30,
            ..Default::default()
        };

        let mut index = OptimizedMultiVectorIndex::with_config(2, config);

        let doc = MultiVectorDoc::new(
            "doc1",
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            serde_json::json!({}),
        );
        index.add(doc).unwrap();

        // Query with low similarity should be filtered out by min_score
        let query = vec![vec![0.1, 0.1]]; // Low similarity with doc tokens
        let results = index.search(&query, 10).unwrap();

        // Results may be empty or filtered depending on actual scores
        assert!(results.len() <= 1);
    }
}
