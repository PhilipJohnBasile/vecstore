///! ColBERT Late Interaction Reranking
///!
///! This module implements ColBERT (Contextualized Late Interaction over BERT), a state-of-the-art
///! neural reranking approach that uses token-level interactions for high-accuracy retrieval.
///!
///! ## How ColBERT Works
///!
///! Unlike traditional reranking that uses a single vector per document:
///! 1. **Multi-vector representation**: Each document/query is encoded as multiple vectors (one per token)
///! 2. **Late interaction**: Similarity is computed at the token level, not averaged upfront
///! 3. **MaxSim operation**: For each query token, find max similarity with any document token
///! 4. **Final score**: Sum of all query token max similarities
///!
///! ## Example
///!
///! ```no_run
///! use vecstore::reranking::colbert::{ColBERTReranker, ColBERTConfig};
///!
///! # async fn example() -> anyhow::Result<()> {
///! let config = ColBERTConfig {
///!     max_query_tokens: 32,
///!     max_doc_tokens: 128,
///!     ..Default::default()
///! };
///!
///! let reranker = ColBERTReranker::new(config)?;
///!
///! // Encode query and documents
///! let query_tokens = reranker.encode_query("what is rust?").await?;
///! let doc_tokens = reranker.encode_document("Rust is a systems programming language").await?;
///!
///! // Compute late interaction score
///! let score = reranker.compute_score(&query_tokens, &doc_tokens)?;
///! # Ok(())
///! # }
///! ```
use anyhow::Result;
use std::collections::HashMap;

/// ColBERT reranker configuration
#[derive(Debug, Clone)]
pub struct ColBERTConfig {
    /// Maximum number of query tokens to encode
    pub max_query_tokens: usize,

    /// Maximum number of document tokens to encode
    pub max_doc_tokens: usize,

    /// Token embedding dimension
    pub embedding_dim: usize,

    /// Similarity metric (typically cosine for ColBERT)
    pub similarity_metric: SimilarityMetric,

    /// Whether to normalize embeddings
    pub normalize: bool,
}

impl Default for ColBERTConfig {
    fn default() -> Self {
        Self {
            max_query_tokens: 32,
            max_doc_tokens: 128,
            embedding_dim: 128, // ColBERT typically uses smaller dims
            similarity_metric: SimilarityMetric::Cosine,
            normalize: true,
        }
    }
}

/// Similarity metrics for token-level comparison
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimilarityMetric {
    /// Cosine similarity (default for ColBERT)
    Cosine,
    /// Dot product similarity
    DotProduct,
    /// L2 (Euclidean) distance
    L2,
}

/// Multi-vector representation of a text (one vector per token)
#[derive(Debug, Clone)]
pub struct TokenEmbeddings {
    /// Token vectors (shape: [num_tokens, embedding_dim])
    pub embeddings: Vec<Vec<f32>>,

    /// Optional token IDs or text for debugging
    pub tokens: Option<Vec<String>>,
}

impl TokenEmbeddings {
    /// Create new token embeddings
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            embeddings,
            tokens: None,
        }
    }

    /// Create with token text for debugging
    pub fn with_tokens(embeddings: Vec<Vec<f32>>, tokens: Vec<String>) -> Self {
        Self {
            embeddings,
            tokens: Some(tokens),
        }
    }

    /// Number of tokens
    pub fn num_tokens(&self) -> usize {
        self.embeddings.len()
    }

    /// Embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embeddings.first().map(|v| v.len()).unwrap_or(0)
    }
}

/// ColBERT reranker for late interaction scoring
pub struct ColBERTReranker {
    config: ColBERTConfig,

    /// Cache for document embeddings (document_id -> token embeddings)
    doc_cache: HashMap<String, TokenEmbeddings>,
}

impl ColBERTReranker {
    /// Create a new ColBERT reranker
    pub fn new(config: ColBERTConfig) -> Result<Self> {
        Ok(Self {
            config,
            doc_cache: HashMap::new(),
        })
    }

    /// Encode a query into token-level embeddings
    ///
    /// Note: In a real implementation, this would use a ColBERT model (BERT + projection layer).
    /// For now, this is a placeholder that shows the interface.
    pub async fn encode_query(&self, query: &str) -> Result<TokenEmbeddings> {
        // TODO: Integrate with actual ColBERT model (e.g., via ONNX Runtime)
        // This is a simplified version for demonstration

        let tokens: Vec<String> = query.split_whitespace().map(|s| s.to_string()).collect();
        let num_tokens = tokens.len().min(self.config.max_query_tokens);

        // Placeholder: random embeddings (replace with actual model)
        let embeddings: Vec<Vec<f32>> = (0..num_tokens)
            .map(|_| {
                let vec: Vec<f32> = (0..self.config.embedding_dim)
                    .map(|_| rand::random::<f32>() - 0.5)
                    .collect();

                if self.config.normalize {
                    Self::normalize_vector(vec)
                } else {
                    vec
                }
            })
            .collect();

        Ok(TokenEmbeddings::with_tokens(
            embeddings,
            tokens[..num_tokens].to_vec(),
        ))
    }

    /// Encode a document into token-level embeddings
    pub async fn encode_document(&self, document: &str) -> Result<TokenEmbeddings> {
        // Similar to encode_query but with max_doc_tokens limit

        let tokens: Vec<String> = document.split_whitespace().map(|s| s.to_string()).collect();
        let num_tokens = tokens.len().min(self.config.max_doc_tokens);

        // Placeholder: random embeddings (replace with actual model)
        let embeddings: Vec<Vec<f32>> = (0..num_tokens)
            .map(|_| {
                let vec: Vec<f32> = (0..self.config.embedding_dim)
                    .map(|_| rand::random::<f32>() - 0.5)
                    .collect();

                if self.config.normalize {
                    Self::normalize_vector(vec)
                } else {
                    vec
                }
            })
            .collect();

        Ok(TokenEmbeddings::with_tokens(
            embeddings,
            tokens[..num_tokens].to_vec(),
        ))
    }

    /// Compute ColBERT late interaction score
    ///
    /// For each query token, find the maximum similarity with any document token,
    /// then sum these maximum similarities.
    ///
    /// Score = Σ_i max_j sim(q_i, d_j)
    /// where q_i is the i-th query token, d_j is the j-th document token
    pub fn compute_score(
        &self,
        query_tokens: &TokenEmbeddings,
        doc_tokens: &TokenEmbeddings,
    ) -> Result<f32> {
        if query_tokens.embedding_dim() != doc_tokens.embedding_dim() {
            anyhow::bail!(
                "Dimension mismatch: query={}, doc={}",
                query_tokens.embedding_dim(),
                doc_tokens.embedding_dim()
            );
        }

        let mut total_score = 0.0;

        // For each query token
        for query_emb in &query_tokens.embeddings {
            let mut max_sim = f32::NEG_INFINITY;

            // Find max similarity with any document token
            for doc_emb in &doc_tokens.embeddings {
                let sim = self.compute_token_similarity(query_emb, doc_emb);
                max_sim = max_sim.max(sim);
            }

            total_score += max_sim;
        }

        Ok(total_score)
    }

    /// Compute similarity between two token embeddings
    fn compute_token_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        match self.config.similarity_metric {
            SimilarityMetric::Cosine => Self::cosine_similarity(vec1, vec2),
            SimilarityMetric::DotProduct => Self::dot_product(vec1, vec2),
            SimilarityMetric::L2 => -Self::l2_distance(vec1, vec2), // Negate for "higher is better"
        }
    }

    /// Cosine similarity (assumes normalized vectors for performance)
    fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
        Self::dot_product(vec1, vec2) // Since normalized, dot product = cosine
    }

    /// Dot product
    fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
    }

    /// L2 distance
    fn l2_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize a vector to unit length
    fn normalize_vector(vec: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vec.into_iter().map(|x| x / norm).collect()
        } else {
            vec
        }
    }

    /// Cache document embeddings for faster repeated queries
    pub fn cache_document(&mut self, doc_id: String, embeddings: TokenEmbeddings) {
        self.doc_cache.insert(doc_id, embeddings);
    }

    /// Retrieve cached document embeddings
    pub fn get_cached_document(&self, doc_id: &str) -> Option<&TokenEmbeddings> {
        self.doc_cache.get(doc_id)
    }

    /// Clear the document cache
    pub fn clear_cache(&mut self) {
        self.doc_cache.clear();
    }
}

/// Batch reranking using ColBERT
///
/// Reranks a list of documents based on their late interaction scores with the query.
pub struct ColBERTBatchReranker {
    reranker: ColBERTReranker,
}

impl ColBERTBatchReranker {
    /// Create a new batch reranker
    pub fn new(config: ColBERTConfig) -> Result<Self> {
        Ok(Self {
            reranker: ColBERTReranker::new(config)?,
        })
    }

    /// Rerank a batch of documents
    ///
    /// Returns document indices sorted by score (descending)
    pub async fn rerank(
        &mut self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        // Encode query once
        let query_tokens = self.reranker.encode_query(query).await?;

        // Compute scores for all documents
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(documents.len());

        for (idx, doc) in documents.iter().enumerate() {
            let doc_tokens = self.reranker.encode_document(doc).await?;
            let score = self.reranker.compute_score(&query_tokens, &doc_tokens)?;
            scores.push((idx, score));
        }

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        scores.truncate(top_k);
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_embeddings_creation() {
        let embeddings = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];

        let token_embs = TokenEmbeddings::new(embeddings.clone());
        assert_eq!(token_embs.num_tokens(), 2);
        assert_eq!(token_embs.embedding_dim(), 3);
    }

    #[test]
    fn test_normalize_vector() {
        let vec = vec![3.0, 4.0]; // Length = 5
        let normalized = ColBERTReranker::normalize_vector(vec);

        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        // Check unit length
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];

        let dot = ColBERTReranker::dot_product(&vec1, &vec2);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_l2_distance() {
        let vec1 = vec![0.0, 0.0];
        let vec2 = vec![3.0, 4.0];

        let dist = ColBERTReranker::l2_distance(&vec1, &vec2);
        assert_eq!(dist, 5.0); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_compute_score() {
        let config = ColBERTConfig {
            embedding_dim: 3,
            normalize: false, // Disable for predictable test
            ..Default::default()
        };

        let reranker = ColBERTReranker::new(config).unwrap();

        let query_tokens = TokenEmbeddings::new(vec![
            vec![1.0, 0.0, 0.0], // Query token 1
            vec![0.0, 1.0, 0.0], // Query token 2
        ]);

        let doc_tokens = TokenEmbeddings::new(vec![
            vec![1.0, 0.0, 0.0], // Doc token 1 (matches query token 1)
            vec![0.0, 0.0, 1.0], // Doc token 2
            vec![0.0, 1.0, 0.0], // Doc token 3 (matches query token 2)
        ]);

        let score = reranker.compute_score(&query_tokens, &doc_tokens).unwrap();

        // Query token 1 max sim = 1.0 (with doc token 1)
        // Query token 2 max sim = 1.0 (with doc token 3)
        // Total = 2.0
        assert_eq!(score, 2.0);
    }

    #[tokio::test]
    async fn test_colbert_reranker_basic() {
        let config = ColBERTConfig::default();
        let reranker = ColBERTReranker::new(config).unwrap();

        let query_tokens = reranker.encode_query("test query").await.unwrap();
        let doc_tokens = reranker.encode_document("test document").await.unwrap();

        assert!(query_tokens.num_tokens() > 0);
        assert!(doc_tokens.num_tokens() > 0);

        let score = reranker.compute_score(&query_tokens, &doc_tokens).unwrap();
        assert!(score.is_finite());
    }
}
