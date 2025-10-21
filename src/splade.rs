//! SPLADE: Sparse Lexical AnD Expansion Model
//!
//! SPLADE is a neural sparse retrieval model that learns sparse representations
//! by expanding queries and documents with learned term weights.
//!
//! ## Key Features
//!
//! - **Learned sparsity**: Neural model learns which terms are important
//! - **Expansion**: Adds related terms beyond exact matches
//! - **Efficient storage**: Significant compression compared to dense vectors
//! - **Better than BM25**: Outperforms traditional sparse retrieval on BEIR
//!
//! ## Architecture
//!
//! ```text
//! Input Text
//!     │
//!     ▼
//! ┌─────────────┐
//! │  Tokenizer  │
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │ BERT/RoBERTa│
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │ Log(1+ReLU) │  ← Activation
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │ Max Pooling │
//! └──────┬──────┘
//!        │
//!        ▼
//! Sparse Vector
//! ```
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::splade::{SpladeEncoder, SpladeConfig, ActivationType};
//!
//! # fn main() -> anyhow::Result<()> {
//! let config = SpladeConfig {
//!     vocab_size: 30522,
//!     activation: ActivationType::Log1p,
//!     sparsity_threshold: 0.01,
//!     ..Default::default()
//! };
//!
//! let encoder = SpladeEncoder::new(config)?;
//!
//! // Encode text to sparse vector
//! let text = "machine learning with neural networks";
//! let sparse_vec = encoder.encode(text)?;
//!
//! println!("Sparse vector has {} non-zero terms", sparse_vec.indices.len());
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Activation function type for SPLADE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    /// log(1 + x) - original SPLADE
    Log1p,
    /// ReLU(x)
    Relu,
    /// log(1 + ReLU(x))
    Log1pRelu,
}

/// SPLADE configuration
#[derive(Debug, Clone)]
pub struct SpladeConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Activation function
    pub activation: ActivationType,
    /// Threshold for sparsity (terms below this are pruned)
    pub sparsity_threshold: f32,
    /// Maximum number of terms to keep
    pub max_terms: Option<usize>,
    /// Model name/path
    pub model_path: Option<String>,
}

impl Default for SpladeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522, // BERT vocab size
            activation: ActivationType::Log1pRelu,
            sparsity_threshold: 0.01,
            max_terms: Some(256),
            model_path: None,
        }
    }
}

/// Sparse vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Non-zero term indices (vocabulary IDs)
    pub indices: Vec<usize>,
    /// Corresponding weights
    pub weights: Vec<f32>,
    /// Dimension (vocab size)
    pub dim: usize,
}

impl SparseVector {
    /// Create a new sparse vector
    pub fn new(indices: Vec<usize>, weights: Vec<f32>, dim: usize) -> Self {
        assert_eq!(indices.len(), weights.len());
        Self {
            indices,
            weights,
            dim,
        }
    }

    /// Compute dot product with another sparse vector
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut score = 0.0;

        // Create hash map for faster lookup
        let other_map: HashMap<usize, f32> = other
            .indices
            .iter()
            .zip(other.weights.iter())
            .map(|(&idx, &w)| (idx, w))
            .collect();

        for (&idx, &weight) in self.indices.iter().zip(self.weights.iter()) {
            if let Some(&other_weight) = other_map.get(&idx) {
                score += weight * other_weight;
            }
        }

        score
    }

    /// Get the L1 norm (sum of absolute values)
    pub fn l1_norm(&self) -> f32 {
        self.weights.iter().map(|w| w.abs()).sum()
    }

    /// Get the L2 norm
    pub fn l2_norm(&self) -> f32 {
        self.weights.iter().map(|w| w * w).sum::<f32>().sqrt()
    }

    /// Prune terms below threshold
    pub fn prune(&mut self, threshold: f32) {
        let keep: Vec<usize> = self
            .weights
            .iter()
            .enumerate()
            .filter(|(_, &w)| w >= threshold)
            .map(|(i, _)| i)
            .collect();

        let new_indices: Vec<usize> = keep.iter().map(|&i| self.indices[i]).collect();
        let new_weights: Vec<f32> = keep.iter().map(|&i| self.weights[i]).collect();

        self.indices = new_indices;
        self.weights = new_weights;
    }

    /// Get top-k terms by weight
    pub fn top_k(&mut self, k: usize) {
        if self.indices.len() <= k {
            return;
        }

        // Create vec of (index, weight) pairs
        let mut pairs: Vec<(usize, f32)> = self
            .indices
            .iter()
            .zip(self.weights.iter())
            .map(|(&i, &w)| (i, w))
            .collect();

        // Partial sort to get top-k
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs.truncate(k);

        self.indices = pairs.iter().map(|&(i, _)| i).collect();
        self.weights = pairs.iter().map(|&(_, w)| w).collect();
    }

    /// Get sparsity ratio (fraction of zero elements)
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.indices.len() as f32 / self.dim as f32)
    }

    /// Convert to dense vector
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.dim];
        for (&idx, &weight) in self.indices.iter().zip(self.weights.iter()) {
            if idx < self.dim {
                dense[idx] = weight;
            }
        }
        dense
    }
}

/// SPLADE encoder
pub struct SpladeEncoder {
    config: SpladeConfig,
    // In real implementation, would hold:
    // - Tokenizer
    // - BERT/RoBERTa model
    // - Vocabulary mapping
}

impl SpladeEncoder {
    /// Create a new SPLADE encoder
    pub fn new(config: SpladeConfig) -> Result<Self> {
        // Real implementation would:
        // 1. Load pre-trained model (e.g., naver/splade-cocondenser-ensembledistil)
        // 2. Initialize tokenizer
        // 3. Set up inference pipeline

        Ok(Self { config })
    }

    /// Encode text to sparse vector
    pub fn encode(&self, text: &str) -> Result<SparseVector> {
        // Real implementation:
        // 1. Tokenize text
        // 2. Run through BERT
        // 3. Apply activation (log1p(relu(x)))
        // 4. Max pooling over tokens
        // 5. Prune low-weight terms

        // Placeholder: Create mock sparse vector
        let mock_indices = vec![100, 250, 500, 1000, 2000];
        let mock_weights = vec![2.5, 1.8, 1.2, 0.9, 0.5];

        let mut sparse = SparseVector::new(mock_indices, mock_weights, self.config.vocab_size);

        // Apply sparsity threshold
        sparse.prune(self.config.sparsity_threshold);

        // Apply max terms limit
        if let Some(max_terms) = self.config.max_terms {
            sparse.top_k(max_terms);
        }

        Ok(sparse)
    }

    /// Encode batch of texts
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<SparseVector>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    /// Apply activation function
    fn apply_activation(&self, logits: &[f32]) -> Vec<f32> {
        match self.config.activation {
            ActivationType::Log1p => logits.iter().map(|&x| (1.0 + x).ln()).collect(),
            ActivationType::Relu => logits.iter().map(|&x| x.max(0.0)).collect(),
            ActivationType::Log1pRelu => logits.iter().map(|&x| (1.0 + x.max(0.0)).ln()).collect(),
        }
    }
}

/// Sparse index for efficient SPLADE retrieval
pub struct SparseIndex {
    /// Inverted index: term_id -> list of (doc_id, weight)
    inverted_index: HashMap<usize, Vec<(String, f32)>>,
    /// Document vectors
    documents: HashMap<String, SparseVector>,
}

impl SparseIndex {
    /// Create a new sparse index
    pub fn new() -> Self {
        Self {
            inverted_index: HashMap::new(),
            documents: HashMap::new(),
        }
    }

    /// Add a document to the index
    pub fn add(&mut self, doc_id: String, vector: SparseVector) {
        // Build inverted index
        for (&term_id, &weight) in vector.indices.iter().zip(vector.weights.iter()) {
            self.inverted_index
                .entry(term_id)
                .or_insert_with(Vec::new)
                .push((doc_id.clone(), weight));
        }

        self.documents.insert(doc_id, vector);
    }

    /// Search for top-k documents
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<(String, f32)> {
        let mut scores: HashMap<String, f32> = HashMap::new();

        // Accumulate scores from posting lists
        for (&term_id, &query_weight) in query.indices.iter().zip(query.weights.iter()) {
            if let Some(postings) = self.inverted_index.get(&term_id) {
                for (doc_id, doc_weight) in postings {
                    *scores.entry(doc_id.clone()).or_insert(0.0) += query_weight * doc_weight;
                }
            }
        }

        // Sort by score and return top-k
        let mut results: Vec<(String, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);

        results
    }

    /// Get statistics about the index
    pub fn stats(&self) -> SparseIndexStats {
        let total_terms: usize = self.inverted_index.len();
        let total_postings: usize = self.inverted_index.values().map(|v| v.len()).sum();

        let avg_sparsity = if !self.documents.is_empty() {
            self.documents.values().map(|v| v.sparsity()).sum::<f32>() / self.documents.len() as f32
        } else {
            0.0
        };

        SparseIndexStats {
            num_documents: self.documents.len(),
            num_unique_terms: total_terms,
            total_postings,
            avg_sparsity,
        }
    }
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for sparse index
#[derive(Debug, Clone)]
pub struct SparseIndexStats {
    pub num_documents: usize,
    pub num_unique_terms: usize,
    pub total_postings: usize,
    pub avg_sparsity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_creation() {
        let indices = vec![10, 20, 30];
        let weights = vec![1.0, 2.0, 3.0];
        let sparse = SparseVector::new(indices.clone(), weights.clone(), 100);

        assert_eq!(sparse.indices, indices);
        assert_eq!(sparse.weights, weights);
        assert_eq!(sparse.dim, 100);
    }

    #[test]
    fn test_sparse_dot_product() {
        let v1 = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0], 10);
        let v2 = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0], 10);

        // Only index 2 overlaps: 2.0 * 2.0 = 4.0
        let dot = v1.dot(&v2);
        assert_eq!(dot, 4.0);
    }

    #[test]
    fn test_sparse_norms() {
        let sparse = SparseVector::new(vec![0, 1, 2], vec![3.0, 4.0, 0.0], 10);

        assert_eq!(sparse.l1_norm(), 7.0);
        assert_eq!(sparse.l2_norm(), 5.0); // sqrt(9 + 16)
    }

    #[test]
    fn test_sparse_pruning() {
        let mut sparse = SparseVector::new(vec![0, 1, 2, 3], vec![2.0, 0.5, 1.5, 0.1], 10);

        sparse.prune(1.0);

        assert_eq!(sparse.indices, vec![0, 2]);
        assert_eq!(sparse.weights, vec![2.0, 1.5]);
    }

    #[test]
    fn test_sparse_top_k() {
        let mut sparse = SparseVector::new(vec![0, 1, 2, 3, 4], vec![1.0, 5.0, 2.0, 4.0, 3.0], 10);

        sparse.top_k(3);

        assert_eq!(sparse.indices.len(), 3);
        assert_eq!(sparse.weights.len(), 3);
        // Should keep indices with weights 5.0, 4.0, 3.0
        assert!(sparse.weights.contains(&5.0));
        assert!(sparse.weights.contains(&4.0));
        assert!(sparse.weights.contains(&3.0));
    }

    #[test]
    fn test_sparse_sparsity() {
        let sparse = SparseVector::new(vec![0, 1], vec![1.0, 2.0], 100);
        assert_eq!(sparse.sparsity(), 0.98); // 98 zeros out of 100
    }

    #[test]
    fn test_sparse_to_dense() {
        let sparse = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0], 5);
        let dense = sparse.to_dense();

        assert_eq!(dense, vec![1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_splade_encoder_creation() {
        let config = SpladeConfig::default();
        let encoder = SpladeEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_splade_encode() {
        let config = SpladeConfig::default();
        let encoder = SpladeEncoder::new(config).unwrap();

        let sparse = encoder.encode("test query").unwrap();
        assert!(sparse.indices.len() > 0);
        assert_eq!(sparse.indices.len(), sparse.weights.len());
    }

    #[test]
    fn test_sparse_index_add_and_search() {
        let mut index = SparseIndex::new();

        // Add documents
        let doc1 = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 1.0], 100);
        let doc2 = SparseVector::new(vec![2, 3, 4], vec![1.5, 1.0, 2.0], 100);

        index.add("doc1".to_string(), doc1);
        index.add("doc2".to_string(), doc2);

        // Search
        let query = SparseVector::new(vec![2, 3], vec![1.0, 1.0], 100);
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
        // doc1 should score higher: 1.0*2.0 + 1.0*1.0 = 3.0
        // doc2 should score: 1.0*1.5 + 1.0*1.0 = 2.5
        assert_eq!(results[0].0, "doc1");
    }

    #[test]
    fn test_sparse_index_stats() {
        let mut index = SparseIndex::new();

        let doc1 = SparseVector::new(vec![1, 2], vec![1.0, 2.0], 100);
        let doc2 = SparseVector::new(vec![2, 3], vec![1.5, 1.0], 100);

        index.add("doc1".to_string(), doc1);
        index.add("doc2".to_string(), doc2);

        let stats = index.stats();
        assert_eq!(stats.num_documents, 2);
        assert!(stats.num_unique_terms >= 2);
        assert!(stats.avg_sparsity > 0.9); // Very sparse
    }

    #[test]
    fn test_activation_log1p() {
        let config = SpladeConfig {
            activation: ActivationType::Log1p,
            ..Default::default()
        };
        let encoder = SpladeEncoder::new(config).unwrap();

        let result = encoder.apply_activation(&[0.0, 1.0, 2.0]);
        assert!((result[0] - 0.0).abs() < 0.001); // ln(1)
        assert!((result[1] - 0.693).abs() < 0.01); // ln(2)
    }
}
