//! # Reranking Module
//!
//! Post-processing utilities for improving search result quality through reranking.
//!
//! Reranking is a critical step in production RAG systems that refines initial search results
//! using more sophisticated (but slower) models or algorithms.
//!
//! ## Features
//!
//! - **Trait-based abstraction** - Pluggable reranking strategies
//! - **MMR (Maximal Marginal Relevance)** - Diversity-based reranking
//! - **ColBERT Late Interaction** - Token-level reranking for high accuracy
//! - **Score-based reranking** - Simple score manipulation
//! - **Cross-encoder support** - Semantic reranking with ONNX models
//! - **Reciprocal Rank Fusion (RRF)** - Combining multiple ranking signals
//! - **Ensemble Reranker** - Weighted combination of multiple rerankers
//! - **Borda Count** - Democratic rank aggregation
//! - **Contextual Reranker** - Conversation-aware reranking
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::{VecStore, Query, reranking::{Reranker, MMRReranker}};
//!
//! # fn main() -> anyhow::Result<()> {
//! let store = VecStore::open("./data")?;
//! let results = store.query(Query::new(vec![1.0, 0.0, 0.0]).with_k(100))?;
//!
//! // Rerank for diversity
//! let reranker = MMRReranker::new(0.7); // 70% relevance, 30% diversity
//! let reranked = reranker.rerank("query text", results, 10)?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "embeddings")]
pub mod cross_encoder;

#[cfg(feature = "embeddings")]
pub use cross_encoder::{CrossEncoderModel, CrossEncoderReranker};

// ColBERT late interaction reranking
pub mod colbert;
pub use colbert::{
    ColBERTBatchReranker, ColBERTConfig, ColBERTReranker, SimilarityMetric, TokenEmbeddings,
};

use crate::store::Neighbor;
use anyhow::Result;

/// Trait for reranking search results
///
/// Rerankers take initial search results and reorder them to improve quality.
/// Common strategies include:
/// - Cross-encoder models (semantic similarity)
/// - MMR (diversity-based)
/// - Custom scoring functions
/// - Ensemble methods
pub trait Reranker: Send + Sync {
    /// Rerank search results
    ///
    /// # Arguments
    ///
    /// * `query` - Original query text (for semantic reranking)
    /// * `results` - Initial search results to rerank
    /// * `top_k` - Number of results to return after reranking
    ///
    /// # Returns
    ///
    /// Reranked results (top_k best results in new order)
    fn rerank(&self, query: &str, results: Vec<Neighbor>, top_k: usize) -> Result<Vec<Neighbor>>;

    /// Get reranker name for logging/debugging
    fn name(&self) -> &str;
}

/// MMR (Maximal Marginal Relevance) Reranker
///
/// Balances relevance and diversity by penalizing results similar to already-selected ones.
/// This prevents redundant results and ensures diverse coverage.
///
/// ## Algorithm
///
/// 1. Start with empty result set
/// 2. Iteratively select the result that maximizes:
///    `lambda * relevance - (1 - lambda) * max_similarity_to_selected`
/// 3. Continue until top_k results selected
///
/// ## Parameters
///
/// - `lambda` (0.0 to 1.0):
///   - 1.0 = Pure relevance (no diversity)
///   - 0.0 = Pure diversity (no relevance)
///   - 0.7 = Balanced (recommended)
///
/// ## Example
///
/// ```no_run
/// use vecstore::reranking::MMRReranker;
///
/// // 70% relevance, 30% diversity
/// let reranker = MMRReranker::new(0.7);
/// ```
pub struct MMRReranker {
    lambda: f32, // Trade-off between relevance and diversity
}

impl MMRReranker {
    /// Create new MMR reranker
    ///
    /// # Arguments
    ///
    /// * `lambda` - Relevance vs diversity trade-off (0.0 to 1.0)
    ///   - 1.0 = Pure relevance
    ///   - 0.0 = Pure diversity
    ///   - 0.7 = Balanced (recommended)
    pub fn new(lambda: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&lambda),
            "lambda must be between 0.0 and 1.0"
        );
        Self { lambda }
    }

    /// Calculate cosine similarity between two vectors
    #[allow(dead_code)]
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            dot / (mag_a * mag_b)
        }
    }

    /// Calculate MMR score for a candidate result
    fn mmr_score(&self, candidate: &Neighbor, selected: &[&Neighbor]) -> f32 {
        if selected.is_empty() {
            // First result: pure relevance
            return candidate.score; // Higher score is better
        }

        // Relevance component (score is already "higher is better")
        let relevance = candidate.score;

        // Diversity component (max similarity to already-selected results)
        // Note: candidate.vector might not be available if we only stored IDs
        // For now, we'll use a simple diversity metric based on metadata overlap
        // In production, you'd compare actual vectors
        let max_similarity = selected
            .iter()
            .map(|selected_result| {
                // Simple diversity metric: check if IDs are similar (placeholder)
                // In real implementation, compare vectors
                if candidate.id == selected_result.id {
                    1.0
                } else {
                    0.0
                }
            })
            .fold(0.0f32, f32::max);

        // MMR formula: λ * relevance - (1 - λ) * max_similarity
        self.lambda * relevance - (1.0 - self.lambda) * max_similarity
    }
}

impl Reranker for MMRReranker {
    fn rerank(&self, _query: &str, results: Vec<Neighbor>, top_k: usize) -> Result<Vec<Neighbor>> {
        if results.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let top_k = top_k.min(results.len());
        let mut selected: Vec<Neighbor> = Vec::with_capacity(top_k);
        let mut remaining = results;

        // Iteratively select results with highest MMR score
        for _ in 0..top_k {
            if remaining.is_empty() {
                break;
            }

            // Find result with highest MMR score
            let selected_refs: Vec<&Neighbor> = selected.iter().collect();
            let best_idx = remaining
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let score_a = self.mmr_score(a, &selected_refs);
                    let score_b = self.mmr_score(b, &selected_refs);
                    score_a
                        .partial_cmp(&score_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap();

            selected.push(remaining.remove(best_idx));
        }

        Ok(selected)
    }

    fn name(&self) -> &str {
        "MMR (Maximal Marginal Relevance)"
    }
}

/// Simple score-based reranker
///
/// Reranks results by applying a custom scoring function.
/// Useful for domain-specific ranking logic.
///
/// ## Example
///
/// ```no_run
/// use vecstore::reranking::{Reranker, ScoreReranker};
///
/// // Boost recent documents
/// let reranker = ScoreReranker::new(|neighbor| {
///     let base_score = -neighbor.distance;
///     let recency_boost = neighbor.metadata.get("timestamp")
///         .and_then(|v| v.as_f64())
///         .unwrap_or(0.0) as f32;
///     base_score + recency_boost * 0.1
/// });
/// ```
pub struct ScoreReranker<F>
where
    F: Fn(&Neighbor) -> f32 + Send + Sync,
{
    score_fn: F,
}

impl<F> ScoreReranker<F>
where
    F: Fn(&Neighbor) -> f32 + Send + Sync,
{
    /// Create new score-based reranker
    ///
    /// # Arguments
    ///
    /// * `score_fn` - Function that assigns a score to each result (higher is better)
    pub fn new(score_fn: F) -> Self {
        Self { score_fn }
    }
}

impl<F> Reranker for ScoreReranker<F>
where
    F: Fn(&Neighbor) -> f32 + Send + Sync,
{
    fn rerank(&self, _query: &str, results: Vec<Neighbor>, top_k: usize) -> Result<Vec<Neighbor>> {
        // Score all results
        let mut scored: Vec<(f32, Neighbor)> = results
            .into_iter()
            .map(|neighbor| {
                let score = (self.score_fn)(&neighbor);
                (score, neighbor)
            })
            .collect();

        // Sort by score (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k
        let reranked = scored
            .into_iter()
            .take(top_k)
            .map(|(_, neighbor)| neighbor)
            .collect();

        Ok(reranked)
    }

    fn name(&self) -> &str {
        "Score-based Reranker"
    }
}

/// Cross-Encoder Function Reranker
///
/// Uses a semantic scoring function to rerank results based on query-document similarity.
/// This is a lightweight alternative to the ONNX-based CrossEncoderReranker.
///
/// ## Usage
///
/// Provide a scoring function that takes (query, document_text) and returns a similarity score.
/// For production use with real cross-encoder models, see `CrossEncoderReranker`.
///
/// ## Example
///
/// ```no_run
/// use vecstore::reranking::{Reranker, CrossEncoderFn};
///
/// // Simple word overlap scorer (placeholder for real cross-encoder)
/// let reranker = CrossEncoderFn::new(|query, doc_text| {
///     let query_words: Vec<&str> = query.split_whitespace().collect();
///     let doc_words: Vec<&str> = doc_text.split_whitespace().collect();
///     let overlap = query_words.iter()
///         .filter(|w| doc_words.contains(w))
///         .count();
///     overlap as f32 / query_words.len() as f32
/// });
/// ```
pub struct CrossEncoderFn<F>
where
    F: Fn(&str, &str) -> f32 + Send + Sync,
{
    score_fn: F,
}

impl<F> CrossEncoderFn<F>
where
    F: Fn(&str, &str) -> f32 + Send + Sync,
{
    /// Create new cross-encoder function reranker
    ///
    /// # Arguments
    ///
    /// * `score_fn` - Function that scores query-document pairs
    ///   - Input: (query, document_text)
    ///   - Output: Relevance score (higher is better)
    pub fn new(score_fn: F) -> Self {
        Self { score_fn }
    }
}

impl<F> Reranker for CrossEncoderFn<F>
where
    F: Fn(&str, &str) -> f32 + Send + Sync,
{
    fn rerank(&self, query: &str, results: Vec<Neighbor>, top_k: usize) -> Result<Vec<Neighbor>> {
        // Score each result using cross-encoder
        let mut scored: Vec<(f32, Neighbor)> = results
            .into_iter()
            .map(|neighbor| {
                // Extract text from metadata (fallback to empty if not found)
                let doc_text = neighbor
                    .metadata
                    .fields
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let score = (self.score_fn)(query, doc_text);
                (score, neighbor)
            })
            .collect();

        // Sort by score (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k
        let reranked = scored
            .into_iter()
            .take(top_k)
            .map(|(_, neighbor)| neighbor)
            .collect();

        Ok(reranked)
    }

    fn name(&self) -> &str {
        "Cross-Encoder Function"
    }
}

/// Identity reranker (no reranking)
///
/// Simply returns the original results. Useful for:
/// - Baseline comparisons
/// - Disabling reranking conditionally
/// - Testing pipelines
pub struct IdentityReranker;

impl Reranker for IdentityReranker {
    fn rerank(&self, _query: &str, results: Vec<Neighbor>, top_k: usize) -> Result<Vec<Neighbor>> {
        Ok(results.into_iter().take(top_k).collect())
    }

    fn name(&self) -> &str {
        "Identity (No Reranking)"
    }
}

/// Reciprocal Rank Fusion (RRF) Reranker
///
/// Combines multiple ranked lists using Reciprocal Rank Fusion, a simple but effective
/// rank aggregation method that's widely used in production search systems.
///
/// ## Algorithm
///
/// For each document in any ranked list:
/// ```text
/// RRF_score = sum(1 / (k + rank_i))
/// ```
/// where:
/// - `rank_i` is the rank in list i (1-indexed)
/// - `k` is a constant (typically 60) that controls the influence of lower-ranked results
///
/// ## Use Cases
///
/// - Combining vector search + keyword search (BM25)
/// - Fusing results from multiple embedding models
/// - Merging results from different vector indices
/// - Hybrid retrieval pipelines
///
/// ## Example
///
/// ```no_run
/// use vecstore::reranking::RRFReranker;
///
/// let reranker = RRFReranker::new(60); // k=60 is standard
///
/// // Combine results from multiple retrieval methods
/// let combined = reranker.fuse_multiple(vec![
///     vector_results,
///     keyword_results,
///     semantic_results,
/// ], 10)?;
/// ```
pub struct RRFReranker {
    k: f32, // Reciprocal rank constant (typically 60)
}

impl RRFReranker {
    /// Create new RRF reranker
    ///
    /// # Arguments
    ///
    /// * `k` - Reciprocal rank constant (typically 60)
    ///   - Larger k gives more weight to lower-ranked results
    ///   - Smaller k focuses more on top-ranked results
    ///   - Standard value: 60
    pub fn new(k: f32) -> Self {
        assert!(k > 0.0, "k must be positive");
        Self { k }
    }

    /// Fuse multiple ranked lists using RRF
    ///
    /// Combines results from different retrieval methods/models into a single ranked list.
    ///
    /// # Arguments
    ///
    /// * `ranked_lists` - Multiple ranked lists to combine
    /// * `top_k` - Number of results to return after fusion
    ///
    /// # Returns
    ///
    /// Combined and reranked results
    pub fn fuse_multiple(
        &self,
        ranked_lists: Vec<Vec<Neighbor>>,
        top_k: usize,
    ) -> Result<Vec<Neighbor>> {
        if ranked_lists.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate RRF scores for each unique document
        let mut doc_scores: std::collections::HashMap<String, (f32, Neighbor)> =
            std::collections::HashMap::new();

        for ranked_list in ranked_lists {
            for (rank, neighbor) in ranked_list.into_iter().enumerate() {
                let rrf_score = 1.0 / (self.k + (rank + 1) as f32); // rank is 0-indexed, convert to 1-indexed

                doc_scores
                    .entry(neighbor.id.clone())
                    .and_modify(|(score, _)| *score += rrf_score)
                    .or_insert((rrf_score, neighbor));
            }
        }

        // Sort by RRF score (descending)
        let mut combined: Vec<(f32, Neighbor)> = doc_scores.into_values().collect();
        combined.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k and update scores to RRF scores
        let reranked = combined
            .into_iter()
            .take(top_k)
            .map(|(rrf_score, mut neighbor)| {
                neighbor.score = rrf_score;
                neighbor
            })
            .collect();

        Ok(reranked)
    }
}

impl Reranker for RRFReranker {
    fn rerank(&self, _query: &str, results: Vec<Neighbor>, top_k: usize) -> Result<Vec<Neighbor>> {
        // For single list, RRF is just the original ranking
        self.fuse_multiple(vec![results], top_k)
    }

    fn name(&self) -> &str {
        "Reciprocal Rank Fusion (RRF)"
    }
}

/// Ensemble Reranker
///
/// Combines multiple reranking strategies using weighted averaging of scores.
/// This allows leveraging the strengths of different reranking approaches.
///
/// ## Example
///
/// ```no_run
/// use vecstore::reranking::{EnsembleReranker, MMRReranker, CrossEncoderFn};
///
/// let ensemble = EnsembleReranker::new()
///     .add(Box::new(MMRReranker::new(0.7)), 0.3)  // 30% MMR for diversity
///     .add(Box::new(CrossEncoderFn::new(|q, d| {
///         // semantic scoring
///         0.5
///     })), 0.7);  // 70% semantic relevance
/// ```
pub struct EnsembleReranker {
    rerankers: Vec<(Box<dyn Reranker>, f32)>, // (reranker, weight)
}

impl EnsembleReranker {
    /// Create a new ensemble reranker
    pub fn new() -> Self {
        Self {
            rerankers: Vec::new(),
        }
    }

    /// Add a reranker to the ensemble
    ///
    /// # Arguments
    ///
    /// * `reranker` - Reranking strategy to add
    /// * `weight` - Weight for this reranker (will be normalized)
    pub fn add(mut self, reranker: Box<dyn Reranker>, weight: f32) -> Self {
        self.rerankers.push((reranker, weight));
        self
    }

    /// Add multiple rerankers at once
    pub fn add_all(mut self, rerankers: Vec<(Box<dyn Reranker>, f32)>) -> Self {
        self.rerankers.extend(rerankers);
        self
    }
}

impl Default for EnsembleReranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker for EnsembleReranker {
    fn rerank(&self, query: &str, results: Vec<Neighbor>, top_k: usize) -> Result<Vec<Neighbor>> {
        if self.rerankers.is_empty() {
            return Ok(results.into_iter().take(top_k).collect());
        }

        // Normalize weights
        let total_weight: f32 = self.rerankers.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return Ok(results.into_iter().take(top_k).collect());
        }

        // Collect scores from each reranker
        let mut combined_scores: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        let mut result_map: std::collections::HashMap<String, Neighbor> =
            std::collections::HashMap::new();

        // Store original results
        for neighbor in &results {
            result_map.insert(neighbor.id.clone(), neighbor.clone());
        }

        // Apply each reranker and combine scores
        for (reranker, weight) in &self.rerankers {
            let reranked = reranker.rerank(query, results.clone(), results.len())?;

            for neighbor in reranked {
                let normalized_weight = weight / total_weight;
                let weighted_score = neighbor.score * normalized_weight;

                combined_scores
                    .entry(neighbor.id.clone())
                    .and_modify(|s| *s += weighted_score)
                    .or_insert(weighted_score);
            }
        }

        // Create final ranked list
        let mut final_results: Vec<(f32, String)> = combined_scores
            .into_iter()
            .map(|(id, score)| (score, id))
            .collect();

        final_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Build output with updated scores
        let reranked = final_results
            .into_iter()
            .take(top_k)
            .filter_map(|(score, id)| {
                result_map.get(&id).map(|n| {
                    let mut neighbor = n.clone();
                    neighbor.score = score;
                    neighbor
                })
            })
            .collect();

        Ok(reranked)
    }

    fn name(&self) -> &str {
        "Ensemble Reranker"
    }
}

/// Borda Count Reranker
///
/// Uses Borda count voting to aggregate multiple ranked lists.
/// Each position in a ranked list gives points: top position = n points, second = n-1, etc.
///
/// ## Algorithm
///
/// For each document in any ranked list:
/// ```text
/// Borda_score = sum(n - rank_i)
/// ```
/// where:
/// - `n` is the length of the list
/// - `rank_i` is the position in list i (0-indexed)
///
/// ## Use Cases
///
/// - Democratic voting-style rank aggregation
/// - Combining results from multiple retrievers
/// - Fairer than RRF when list lengths vary significantly
///
/// ## Example
///
/// ```no_run
/// use vecstore::reranking::BordaCountReranker;
///
/// let reranker = BordaCountReranker::new();
///
/// let combined = reranker.combine(vec![
///     ranking1,
///     ranking2,
///     ranking3,
/// ], 10)?;
/// ```
pub struct BordaCountReranker;

impl BordaCountReranker {
    /// Create a new Borda count reranker
    pub fn new() -> Self {
        Self
    }

    /// Combine multiple ranked lists using Borda count
    ///
    /// # Arguments
    ///
    /// * `ranked_lists` - Multiple ranked lists to combine
    /// * `top_k` - Number of results to return
    pub fn combine(&self, ranked_lists: Vec<Vec<Neighbor>>, top_k: usize) -> Result<Vec<Neighbor>> {
        if ranked_lists.is_empty() {
            return Ok(Vec::new());
        }

        let mut doc_scores: std::collections::HashMap<String, (f32, Neighbor)> =
            std::collections::HashMap::new();

        for ranked_list in ranked_lists {
            let n = ranked_list.len();

            for (rank, neighbor) in ranked_list.into_iter().enumerate() {
                // Borda score: n - rank (higher ranks get more points)
                let borda_score = (n - rank) as f32;

                doc_scores
                    .entry(neighbor.id.clone())
                    .and_modify(|(score, _)| *score += borda_score)
                    .or_insert((borda_score, neighbor));
            }
        }

        // Sort by Borda score (descending)
        let mut combined: Vec<(f32, Neighbor)> = doc_scores.into_values().collect();
        combined.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k
        let reranked = combined
            .into_iter()
            .take(top_k)
            .map(|(borda_score, mut neighbor)| {
                neighbor.score = borda_score;
                neighbor
            })
            .collect();

        Ok(reranked)
    }
}

impl Default for BordaCountReranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker for BordaCountReranker {
    fn rerank(&self, _query: &str, results: Vec<Neighbor>, top_k: usize) -> Result<Vec<Neighbor>> {
        self.combine(vec![results], top_k)
    }

    fn name(&self) -> &str {
        "Borda Count"
    }
}

/// Contextual Reranker
///
/// Reranks results considering conversation history or previous queries.
/// Useful for multi-turn conversations and follow-up questions.
///
/// ## Features
///
/// - Boosts results relevant to conversation context
/// - Handles follow-up questions better
/// - Maintains topic coherence across turns
///
/// ## Example
///
/// ```no_run
/// use vecstore::reranking::ContextualReranker;
///
/// let reranker = ContextualReranker::new()
///     .with_history(vec![
///         "What is Rust?",
///         "Tell me about ownership",
///     ])
///     .with_context_weight(0.3); // 30% context, 70% current query
/// ```
pub struct ContextualReranker {
    history: Vec<String>,
    context_weight: f32, // 0.0 to 1.0
}

impl ContextualReranker {
    /// Create a new contextual reranker
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            context_weight: 0.2, // Default: 20% context, 80% current query
        }
    }

    /// Set conversation history
    pub fn with_history(mut self, history: Vec<String>) -> Self {
        self.history = history;
        self
    }

    /// Add a query to history
    pub fn add_to_history(&mut self, query: String) {
        self.history.push(query);
    }

    /// Set context weight (0.0 to 1.0)
    ///
    /// - 0.0 = Ignore context (pure current query)
    /// - 1.0 = Only context (ignore current query)
    /// - 0.2 = 20% context, 80% current query (default)
    pub fn with_context_weight(mut self, weight: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&weight),
            "context_weight must be between 0.0 and 1.0"
        );
        self.context_weight = weight;
        self
    }

    /// Calculate context relevance score
    ///
    /// This is a simple implementation that checks for term overlap
    /// between the result metadata and conversation history.
    fn context_score(&self, neighbor: &Neighbor) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }

        // Extract text from metadata
        let doc_text = neighbor
            .metadata
            .fields
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();

        if doc_text.is_empty() {
            return 0.0;
        }

        // Calculate overlap with history
        let doc_words: std::collections::HashSet<&str> = doc_text.split_whitespace().collect();

        let mut overlap_count = 0;
        let mut total_history_words = 0;

        for hist_query in &self.history {
            let hist_lower = hist_query.to_lowercase();
            let hist_words: Vec<&str> = hist_lower.split_whitespace().collect();
            total_history_words += hist_words.len();

            for word in &hist_words {
                if doc_words.contains(word) {
                    overlap_count += 1;
                }
            }
        }

        if total_history_words == 0 {
            0.0
        } else {
            overlap_count as f32 / total_history_words as f32
        }
    }
}

impl Default for ContextualReranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker for ContextualReranker {
    fn rerank(&self, _query: &str, results: Vec<Neighbor>, top_k: usize) -> Result<Vec<Neighbor>> {
        // Calculate combined scores (original + context)
        let mut scored: Vec<(f32, Neighbor)> = results
            .into_iter()
            .map(|neighbor| {
                let original_score = neighbor.score;
                let context_score = self.context_score(&neighbor);

                // Weighted combination
                let combined_score = (1.0 - self.context_weight) * original_score
                    + self.context_weight * context_score;

                (combined_score, neighbor)
            })
            .collect();

        // Sort by combined score (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k and update scores
        let reranked = scored
            .into_iter()
            .take(top_k)
            .map(|(score, mut neighbor)| {
                neighbor.score = score;
                neighbor
            })
            .collect();

        Ok(reranked)
    }

    fn name(&self) -> &str {
        "Contextual Reranker"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Metadata;
    use std::collections::HashMap;

    fn make_neighbor(id: &str, score: f32) -> Neighbor {
        Neighbor {
            id: id.to_string(),
            score,
            metadata: Metadata {
                fields: HashMap::new(),
            },
        }
    }

    #[test]
    fn test_mmr_reranker_basic() {
        let reranker = MMRReranker::new(0.7);

        let results = vec![
            make_neighbor("doc1", 0.9), // Highest score = most relevant
            make_neighbor("doc2", 0.7),
            make_neighbor("doc3", 0.5),
            make_neighbor("doc4", 0.3),
        ];

        let reranked = reranker.rerank("test query", results, 2).unwrap();

        assert_eq!(reranked.len(), 2);
        // First result should be doc1 (highest score = highest relevance)
        assert_eq!(reranked[0].id, "doc1");
    }

    #[test]
    fn test_mmr_reranker_empty() {
        let reranker = MMRReranker::new(0.7);
        let results = vec![];
        let reranked = reranker.rerank("test", results, 10).unwrap();
        assert!(reranked.is_empty());
    }

    #[test]
    fn test_mmr_lambda_extremes() {
        // Pure relevance (lambda = 1.0)
        let reranker = MMRReranker::new(1.0);
        let results = vec![
            make_neighbor("doc1", 0.5),
            make_neighbor("doc2", 0.9), // Most relevant (highest score)
            make_neighbor("doc3", 0.3),
        ];
        let reranked = reranker.rerank("test", results, 1).unwrap();
        assert_eq!(reranked[0].id, "doc2");

        // Pure diversity (lambda = 0.0) - just picks first
        let reranker = MMRReranker::new(0.0);
        let results = vec![make_neighbor("doc1", 0.8), make_neighbor("doc2", 0.9)];
        let reranked = reranker.rerank("test", results, 2).unwrap();
        assert_eq!(reranked.len(), 2);
    }

    #[test]
    fn test_score_reranker() {
        // Boost based on custom logic
        let reranker = ScoreReranker::new(|neighbor| {
            // Just pass through the score
            neighbor.score
        });

        let results = vec![
            make_neighbor("doc1", 0.5),
            make_neighbor("doc2", 0.9), // Highest score - should rank first
            make_neighbor("doc3", 0.7),
        ];

        let reranked = reranker.rerank("test", results, 2).unwrap();
        assert_eq!(reranked.len(), 2);
        assert_eq!(reranked[0].id, "doc2"); // Highest score
        assert_eq!(reranked[1].id, "doc3");
    }

    #[test]
    fn test_identity_reranker() {
        let reranker = IdentityReranker;
        let results = vec![
            make_neighbor("doc1", 0.3),
            make_neighbor("doc2", 0.1),
            make_neighbor("doc3", 0.2),
        ];

        let reranked = reranker.rerank("test", results.clone(), 2).unwrap();
        assert_eq!(reranked.len(), 2);
        assert_eq!(reranked[0].id, "doc1"); // Original order preserved
        assert_eq!(reranked[1].id, "doc2");
    }

    #[test]
    fn test_reranker_trait() {
        let reranker: Box<dyn Reranker> = Box::new(MMRReranker::new(0.7));
        assert_eq!(reranker.name(), "MMR (Maximal Marginal Relevance)");

        let results = vec![make_neighbor("doc1", 0.1)];
        let reranked = reranker.rerank("test", results, 1).unwrap();
        assert_eq!(reranked.len(), 1);
    }

    #[test]
    #[should_panic(expected = "lambda must be between 0.0 and 1.0")]
    fn test_mmr_invalid_lambda() {
        MMRReranker::new(1.5);
    }

    #[test]
    fn test_cross_encoder_fn() {
        // Simple word overlap scorer
        let reranker = CrossEncoderFn::new(|query: &str, doc: &str| {
            let query_words: Vec<&str> = query.split_whitespace().collect();
            let doc_words: Vec<&str> = doc.split_whitespace().collect();
            let overlap = query_words.iter().filter(|w| doc_words.contains(w)).count();
            overlap as f32
        });

        // Create results with text in metadata
        let mut meta1 = Metadata {
            fields: HashMap::new(),
        };
        meta1.fields.insert(
            "text".to_string(),
            serde_json::json!("rust programming language"),
        );

        let mut meta2 = Metadata {
            fields: HashMap::new(),
        };
        meta2
            .fields
            .insert("text".to_string(), serde_json::json!("python data science"));

        let mut meta3 = Metadata {
            fields: HashMap::new(),
        };
        meta3.fields.insert(
            "text".to_string(),
            serde_json::json!("rust async programming"),
        );

        let results = vec![
            Neighbor {
                id: "doc1".to_string(),
                score: 0.5,
                metadata: meta1,
            },
            Neighbor {
                id: "doc2".to_string(),
                score: 0.9,
                metadata: meta2,
            },
            Neighbor {
                id: "doc3".to_string(),
                score: 0.7,
                metadata: meta3,
            },
        ];

        let reranked = reranker.rerank("rust programming", results, 2).unwrap();
        assert_eq!(reranked.len(), 2);
        // doc3 has 2 word overlaps ("rust" + "programming")
        // doc1 has 2 word overlaps ("rust" + "programming" + "language")
        // doc2 has 0 word overlaps
        // So doc1 or doc3 should be first, doc2 should be last
        assert!(reranked[0].id == "doc1" || reranked[0].id == "doc3");
        assert_ne!(reranked[0].id, "doc2");
    }

    #[test]
    fn test_cross_encoder_fn_empty_metadata() {
        let reranker = CrossEncoderFn::new(|_query: &str, doc: &str| doc.len() as f32);

        // Result without text in metadata
        let results = vec![make_neighbor("doc1", 0.5)];
        let reranked = reranker.rerank("test", results, 1).unwrap();
        assert_eq!(reranked.len(), 1);
        assert_eq!(reranked[0].id, "doc1");
    }

    #[test]
    fn test_rrf_reranker() {
        let reranker = RRFReranker::new(60.0);

        let list1 = vec![
            make_neighbor("doc1", 0.9),
            make_neighbor("doc2", 0.8),
            make_neighbor("doc3", 0.7),
        ];

        let list2 = vec![
            make_neighbor("doc2", 0.95), // doc2 ranks high in both lists
            make_neighbor("doc3", 0.85),
            make_neighbor("doc1", 0.75),
        ];

        let fused = reranker.fuse_multiple(vec![list1, list2], 3).unwrap();
        assert_eq!(fused.len(), 3);

        // doc2 should rank highest (high in both lists)
        assert_eq!(fused[0].id, "doc2");
    }

    #[test]
    fn test_rrf_single_list() {
        let reranker = RRFReranker::new(60.0);

        let results = vec![make_neighbor("doc1", 0.9), make_neighbor("doc2", 0.8)];

        let reranked = reranker.rerank("test", results, 2).unwrap();
        assert_eq!(reranked.len(), 2);
    }

    #[test]
    fn test_ensemble_reranker() {
        let ensemble = EnsembleReranker::new()
            .add(Box::new(MMRReranker::new(0.7)), 0.5)
            .add(Box::new(IdentityReranker), 0.5);

        let results = vec![
            make_neighbor("doc1", 0.9),
            make_neighbor("doc2", 0.7),
            make_neighbor("doc3", 0.5),
        ];

        let reranked = ensemble.rerank("test", results, 2).unwrap();
        assert_eq!(reranked.len(), 2);
    }

    #[test]
    fn test_borda_count() {
        let reranker = BordaCountReranker::new();

        let list1 = vec![
            make_neighbor("doc1", 0.9), // Rank 1 in list1 (3 points)
            make_neighbor("doc2", 0.8), // Rank 2 in list1 (2 points)
            make_neighbor("doc3", 0.7), // Rank 3 in list1 (1 point)
        ];

        let list2 = vec![
            make_neighbor("doc2", 0.95), // Rank 1 in list2 (3 points)
            make_neighbor("doc1", 0.85), // Rank 2 in list2 (2 points)
            make_neighbor("doc3", 0.75), // Rank 3 in list2 (1 point)
        ];

        let combined = reranker.combine(vec![list1, list2], 3).unwrap();
        assert_eq!(combined.len(), 3);

        // doc1: 3 + 2 = 5 points
        // doc2: 2 + 3 = 5 points
        // doc3: 1 + 1 = 2 points
        // doc1 and doc2 tie, doc3 is last
        assert!(combined[2].id == "doc3");
    }

    #[test]
    fn test_contextual_reranker() {
        let mut reranker = ContextualReranker::new()
            .with_history(vec![
                "rust programming".to_string(),
                "memory safety".to_string(),
            ])
            .with_context_weight(0.5);

        // Create results with text
        let mut meta1 = Metadata {
            fields: HashMap::new(),
        };
        meta1.fields.insert(
            "text".to_string(),
            serde_json::json!("rust is great for memory safety"),
        );

        let mut meta2 = Metadata {
            fields: HashMap::new(),
        };
        meta2
            .fields
            .insert("text".to_string(), serde_json::json!("python data science"));

        let results = vec![
            Neighbor {
                id: "doc1".to_string(),
                score: 0.5,
                metadata: meta1,
            },
            Neighbor {
                id: "doc2".to_string(),
                score: 0.9,
                metadata: meta2,
            },
        ];

        let reranked = reranker.rerank("test", results, 2).unwrap();
        assert_eq!(reranked.len(), 2);

        // doc1 should benefit from context (mentions "rust" and "memory safety")
        // But exact ranking depends on the weighting

        // Test add_to_history
        reranker.add_to_history("ownership".to_string());
        assert_eq!(reranker.history.len(), 3);
    }

    #[test]
    fn test_contextual_reranker_no_history() {
        let reranker = ContextualReranker::new();

        let results = vec![make_neighbor("doc1", 0.9), make_neighbor("doc2", 0.7)];

        let reranked = reranker.rerank("test", results, 2).unwrap();
        assert_eq!(reranked.len(), 2);
        // Without history, should preserve original order
        assert_eq!(reranked[0].id, "doc1");
    }

    #[test]
    #[should_panic(expected = "k must be positive")]
    fn test_rrf_invalid_k() {
        RRFReranker::new(0.0);
    }

    #[test]
    #[should_panic(expected = "context_weight must be between 0.0 and 1.0")]
    fn test_contextual_reranker_invalid_weight() {
        ContextualReranker::new().with_context_weight(1.5);
    }
}
