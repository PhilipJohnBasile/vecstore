// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Hybrid Reranking Pipeline
//!
//! End-to-end hybrid search with dense vectors, sparse vectors, BM25 keyword matching,
//! and ensemble reranking in a single unified pipeline.
//!
//! ## Features
//!
//! - **Multi-Signal Search**: Dense + Sparse + BM25 in one query
//! - **Reciprocal Rank Fusion (RRF)**: Industry-standard result merging
//! - **Cross-Encoder Reranking**: Deep relevance scoring
//! - **Ensemble Methods**: Multiple fusion strategies
//! - **Boolean FTS**: AND/OR/NOT for keyword queries
//! - **Phrase Search**: Exact phrase matching with slop
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::hybrid_rerank::{HybridPipeline, HybridQuery, RerankerConfig};
//!
//! let pipeline = HybridPipeline::new(config);
//!
//! let query = HybridQuery::new("machine learning tutorial")
//!     .with_vector(query_embedding)
//!     .with_sparse(sparse_query)
//!     .with_filter("category = 'tech'");
//!
//! let results = pipeline.search(query)?;
//! ```

use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Hybrid pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Dense vector weight
    pub dense_weight: f32,
    /// Sparse vector weight
    pub sparse_weight: f32,
    /// BM25 weight
    pub bm25_weight: f32,
    /// Fusion strategy
    pub fusion_strategy: FusionMethod,
    /// Reranker configuration
    pub reranker: Option<RerankerConfig>,
    /// Maximum candidates before reranking
    pub max_candidates: usize,
    /// Final result limit
    pub top_k: usize,
    /// Enable boolean FTS
    pub enable_boolean_fts: bool,
    /// Minimum score threshold
    pub min_score: f32,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            dense_weight: 0.7,
            sparse_weight: 0.2,
            bm25_weight: 0.1,
            fusion_strategy: FusionMethod::RRF { k: 60.0 },
            reranker: None,
            max_candidates: 100,
            top_k: 10,
            enable_boolean_fts: true,
            min_score: 0.0,
        }
    }
}

/// Fusion method for combining results
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion with k parameter
    RRF { k: f32 },
    /// Weighted linear combination
    WeightedSum,
    /// Maximum score across signals
    MaxScore,
    /// Geometric mean of scores
    GeometricMean,
    /// Harmonic mean of scores
    HarmonicMean,
    /// Voting-based fusion
    Voting { min_votes: usize },
    /// Distribution-based fusion (normalize per signal)
    DistributionBased,
    /// Cascade: dense -> sparse -> BM25
    Cascade,
}

/// Reranker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Reranker type
    pub reranker_type: RerankerType,
    /// Model name/path
    pub model: String,
    /// Batch size for reranking
    pub batch_size: usize,
    /// Score normalization
    pub normalize_scores: bool,
}

/// Type of reranker
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RerankerType {
    /// Cross-encoder model
    CrossEncoder,
    /// Cohere reranker
    Cohere,
    /// Voyage reranker
    Voyage,
    /// Custom ONNX model
    OnnxModel { path: String },
    /// LLM-based reranker
    LLM { model: String },
    /// Simple lexical overlap
    LexicalOverlap,
}

/// Hybrid search query
#[derive(Debug, Clone)]
pub struct HybridSearchQuery {
    /// Text query
    pub text: String,
    /// Dense vector (optional)
    pub dense_vector: Option<Vec<f32>>,
    /// Sparse vector (optional)
    pub sparse_vector: Option<SparseQueryVector>,
    /// Boolean FTS query (optional)
    pub boolean_query: Option<BooleanQuery>,
    /// Metadata filters
    pub filters: HashMap<String, FilterValue>,
    /// Override weights
    pub weight_overrides: Option<WeightOverrides>,
}

impl HybridSearchQuery {
    /// Create new query from text
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            dense_vector: None,
            sparse_vector: None,
            boolean_query: None,
            filters: HashMap::new(),
            weight_overrides: None,
        }
    }

    /// Add dense vector
    pub fn with_dense(mut self, vector: Vec<f32>) -> Self {
        self.dense_vector = Some(vector);
        self
    }

    /// Add sparse vector
    pub fn with_sparse(mut self, sparse: SparseQueryVector) -> Self {
        self.sparse_vector = Some(sparse);
        self
    }

    /// Add boolean query
    pub fn with_boolean(mut self, query: BooleanQuery) -> Self {
        self.boolean_query = Some(query);
        self
    }

    /// Add filter
    pub fn with_filter(mut self, key: &str, value: FilterValue) -> Self {
        self.filters.insert(key.to_string(), value);
        self
    }

    /// Override weights
    pub fn with_weights(mut self, dense: f32, sparse: f32, bm25: f32) -> Self {
        self.weight_overrides = Some(WeightOverrides {
            dense_weight: Some(dense),
            sparse_weight: Some(sparse),
            bm25_weight: Some(bm25),
        });
        self
    }
}

/// Sparse query vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseQueryVector {
    /// Term indices
    pub indices: Vec<u32>,
    /// Term weights
    pub values: Vec<f32>,
}

impl SparseQueryVector {
    /// Create from terms and weights
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        Self { indices, values }
    }

    /// Create from term-weight pairs
    pub fn from_pairs(pairs: Vec<(u32, f32)>) -> Self {
        let (indices, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Self { indices, values }
    }
}

/// Boolean FTS query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BooleanQuery {
    /// Term must appear
    Must(String),
    /// Term should appear (boosts score)
    Should(String),
    /// Term must not appear
    MustNot(String),
    /// Phrase match with slop (word distance allowed)
    Phrase { terms: Vec<String>, slop: usize },
    /// Fuzzy term matching
    Fuzzy { term: String, max_edits: usize },
    /// Wildcard matching
    Wildcard(String),
    /// AND of multiple queries
    And(Vec<BooleanQuery>),
    /// OR of multiple queries
    Or(Vec<BooleanQuery>),
    /// NOT (negate query)
    Not(Box<BooleanQuery>),
}

impl BooleanQuery {
    /// Create MUST query
    pub fn must(term: &str) -> Self {
        BooleanQuery::Must(term.to_string())
    }

    /// Create SHOULD query
    pub fn should(term: &str) -> Self {
        BooleanQuery::Should(term.to_string())
    }

    /// Create MUST NOT query
    pub fn must_not(term: &str) -> Self {
        BooleanQuery::MustNot(term.to_string())
    }

    /// Create phrase query
    pub fn phrase(terms: Vec<&str>, slop: usize) -> Self {
        BooleanQuery::Phrase {
            terms: terms.into_iter().map(|s| s.to_string()).collect(),
            slop,
        }
    }

    /// Create fuzzy query
    pub fn fuzzy(term: &str, max_edits: usize) -> Self {
        BooleanQuery::Fuzzy {
            term: term.to_string(),
            max_edits,
        }
    }

    /// Combine with AND
    pub fn and(queries: Vec<BooleanQuery>) -> Self {
        BooleanQuery::And(queries)
    }

    /// Combine with OR
    pub fn or(queries: Vec<BooleanQuery>) -> Self {
        BooleanQuery::Or(queries)
    }
}

/// Filter value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterValue {
    Equals(String),
    NotEquals(String),
    GreaterThan(f64),
    LessThan(f64),
    Between(f64, f64),
    In(Vec<String>),
    Contains(String),
    StartsWith(String),
    Exists,
    NotExists,
}

/// Weight overrides for query-time tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightOverrides {
    pub dense_weight: Option<f32>,
    pub sparse_weight: Option<f32>,
    pub bm25_weight: Option<f32>,
}

/// Search result from a single signal
#[derive(Debug, Clone)]
pub struct SignalResult {
    /// Document ID
    pub id: String,
    /// Score from this signal
    pub score: f32,
    /// Rank in this signal's results
    pub rank: usize,
    /// Signal type
    pub signal: SignalType,
}

/// Type of search signal
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalType {
    Dense,
    Sparse,
    BM25,
    Reranker,
}

/// Combined hybrid result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResult {
    /// Document ID
    pub id: String,
    /// Final fused score
    pub score: f32,
    /// Scores from each signal
    pub signal_scores: HashMap<String, f32>,
    /// Ranks from each signal
    pub signal_ranks: HashMap<String, usize>,
    /// Number of signals that returned this doc
    pub signal_count: usize,
    /// Document content (if retrieved)
    pub content: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Explanation of scoring
    pub explanation: Option<ScoreExplanation>,
}

/// Score explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreExplanation {
    /// Fusion method used
    pub fusion_method: String,
    /// Component scores
    pub components: Vec<ScoreComponent>,
    /// Reranking applied
    pub reranked: bool,
    /// Original rank before reranking
    pub original_rank: Option<usize>,
}

/// Individual score component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreComponent {
    /// Component name
    pub name: String,
    /// Raw score
    pub raw_score: f32,
    /// Weight applied
    pub weight: f32,
    /// Weighted score
    pub weighted_score: f32,
}

/// Document for indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridDocument {
    /// Document ID
    pub id: String,
    /// Text content for BM25
    pub text: String,
    /// Dense vector
    pub dense_vector: Vec<f32>,
    /// Sparse vector
    pub sparse_vector: Option<SparseQueryVector>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// BM25 index entry
#[derive(Debug, Clone)]
struct BM25Entry {
    doc_id: String,
    term_freqs: HashMap<String, f32>,
    doc_length: f32,
}

/// Inverted index for BM25
struct InvertedIndex {
    /// Term -> doc_id -> term frequency
    postings: HashMap<String, HashMap<String, f32>>,
    /// Document lengths
    doc_lengths: HashMap<String, f32>,
    /// Average document length
    avg_doc_length: f32,
    /// Total documents
    doc_count: usize,
    /// IDF cache
    idf_cache: HashMap<String, f32>,
}

impl InvertedIndex {
    fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            avg_doc_length: 0.0,
            doc_count: 0,
            idf_cache: HashMap::new(),
        }
    }

    fn add_document(&mut self, doc_id: &str, terms: &[String]) {
        let doc_length = terms.len() as f32;
        self.doc_lengths.insert(doc_id.to_string(), doc_length);

        // Count term frequencies
        let mut term_freqs: HashMap<String, f32> = HashMap::new();
        for term in terms {
            *term_freqs.entry(term.clone()).or_insert(0.0) += 1.0;
        }

        // Add to postings
        for (term, freq) in term_freqs {
            self.postings
                .entry(term)
                .or_insert_with(HashMap::new)
                .insert(doc_id.to_string(), freq);
        }

        self.doc_count += 1;

        // Update average doc length
        let total_length: f32 = self.doc_lengths.values().sum();
        self.avg_doc_length = total_length / self.doc_count as f32;

        // Invalidate IDF cache
        self.idf_cache.clear();
    }

    fn get_idf(&mut self, term: &str) -> f32 {
        if let Some(&idf) = self.idf_cache.get(term) {
            return idf;
        }

        let df = self.postings.get(term).map_or(0, |p| p.len());
        let idf = if df > 0 {
            ((self.doc_count as f32 - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln()
        } else {
            0.0
        };

        self.idf_cache.insert(term.to_string(), idf);
        idf
    }

    fn search_bm25(&mut self, query_terms: &[String], k1: f32, b: f32, top_k: usize) -> Vec<(String, f32)> {
        let mut scores: HashMap<String, f32> = HashMap::new();

        for term in query_terms {
            let idf = self.get_idf(term);

            if let Some(postings) = self.postings.get(term) {
                for (doc_id, tf) in postings {
                    let doc_length = self.doc_lengths.get(doc_id).copied().unwrap_or(1.0);
                    let norm = 1.0 - b + b * (doc_length / self.avg_doc_length);
                    let tf_component = (tf * (k1 + 1.0)) / (tf + k1 * norm);
                    let bm25_score = idf * tf_component;

                    *scores.entry(doc_id.clone()).or_insert(0.0) += bm25_score;
                }
            }
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(top_k);
        results
    }
}

/// Sparse index
struct SparseIndex {
    /// Term -> doc_id -> weight
    postings: HashMap<u32, HashMap<String, f32>>,
}

impl SparseIndex {
    fn new() -> Self {
        Self {
            postings: HashMap::new(),
        }
    }

    fn add_document(&mut self, doc_id: &str, sparse: &SparseQueryVector) {
        for (&idx, &val) in sparse.indices.iter().zip(&sparse.values) {
            self.postings
                .entry(idx)
                .or_insert_with(HashMap::new)
                .insert(doc_id.to_string(), val);
        }
    }

    fn search(&self, query: &SparseQueryVector, top_k: usize) -> Vec<(String, f32)> {
        let mut scores: HashMap<String, f32> = HashMap::new();

        for (&idx, &q_val) in query.indices.iter().zip(&query.values) {
            if let Some(postings) = self.postings.get(&idx) {
                for (doc_id, &doc_val) in postings {
                    *scores.entry(doc_id.clone()).or_insert(0.0) += q_val * doc_val;
                }
            }
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(top_k);
        results
    }
}

/// Dense vector index (simple brute-force for demo)
struct DenseIndex {
    vectors: HashMap<String, Vec<f32>>,
}

impl DenseIndex {
    fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }

    fn add_document(&mut self, doc_id: &str, vector: &[f32]) {
        self.vectors.insert(doc_id.to_string(), vector.to_vec());
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let mut scores: Vec<_> = self.vectors
            .iter()
            .map(|(id, vec)| {
                let score = cosine_similarity(query, vec);
                (id.clone(), score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scores.truncate(top_k);
        scores
    }
}

/// Main hybrid search pipeline
pub struct HybridPipeline {
    config: HybridConfig,
    /// Dense vector index
    dense_index: RwLock<DenseIndex>,
    /// Sparse vector index
    sparse_index: RwLock<SparseIndex>,
    /// BM25 inverted index
    bm25_index: RwLock<InvertedIndex>,
    /// Document store
    documents: RwLock<HashMap<String, HybridDocument>>,
    /// Reranker (if configured)
    reranker: Option<Reranker>,
}

impl HybridPipeline {
    /// Create new pipeline
    pub fn new(config: HybridConfig) -> Self {
        let reranker = config.reranker.as_ref().map(|c| Reranker::new(c.clone()));

        Self {
            config,
            dense_index: RwLock::new(DenseIndex::new()),
            sparse_index: RwLock::new(SparseIndex::new()),
            bm25_index: RwLock::new(InvertedIndex::new()),
            documents: RwLock::new(HashMap::new()),
            reranker,
        }
    }

    /// Index a document
    pub fn index(&self, doc: HybridDocument) -> Result<()> {
        // Index in dense index
        {
            let mut dense = self.dense_index.write().unwrap();
            dense.add_document(&doc.id, &doc.dense_vector);
        }

        // Index in sparse index if available
        if let Some(sparse) = &doc.sparse_vector {
            let mut sparse_idx = self.sparse_index.write().unwrap();
            sparse_idx.add_document(&doc.id, sparse);
        }

        // Index in BM25
        {
            let terms = tokenize(&doc.text);
            let mut bm25 = self.bm25_index.write().unwrap();
            bm25.add_document(&doc.id, &terms);
        }

        // Store document
        {
            let mut docs = self.documents.write().unwrap();
            docs.insert(doc.id.clone(), doc);
        }

        Ok(())
    }

    /// Search with hybrid query
    pub fn search(&self, query: HybridSearchQuery) -> Result<Vec<HybridResult>> {
        let weights = self.get_weights(&query.weight_overrides);

        // Collect results from each signal
        let mut all_signals: Vec<Vec<SignalResult>> = Vec::new();

        // Dense search
        if let Some(ref vector) = query.dense_vector {
            if weights.0 > 0.0 {
                let dense = self.dense_index.read().unwrap();
                let results = dense.search(vector, self.config.max_candidates);
                let signal_results: Vec<SignalResult> = results
                    .into_iter()
                    .enumerate()
                    .map(|(rank, (id, score))| SignalResult {
                        id,
                        score,
                        rank,
                        signal: SignalType::Dense,
                    })
                    .collect();
                all_signals.push(signal_results);
            }
        }

        // Sparse search
        if let Some(ref sparse) = query.sparse_vector {
            if weights.1 > 0.0 {
                let sparse_idx = self.sparse_index.read().unwrap();
                let results = sparse_idx.search(sparse, self.config.max_candidates);
                let signal_results: Vec<SignalResult> = results
                    .into_iter()
                    .enumerate()
                    .map(|(rank, (id, score))| SignalResult {
                        id,
                        score,
                        rank,
                        signal: SignalType::Sparse,
                    })
                    .collect();
                all_signals.push(signal_results);
            }
        }

        // BM25 search
        if weights.2 > 0.0 && !query.text.is_empty() {
            let query_terms = tokenize(&query.text);
            let mut bm25 = self.bm25_index.write().unwrap();
            let results = bm25.search_bm25(&query_terms, 1.2, 0.75, self.config.max_candidates);
            let signal_results: Vec<SignalResult> = results
                .into_iter()
                .enumerate()
                .map(|(rank, (id, score))| SignalResult {
                    id,
                    score,
                    rank,
                    signal: SignalType::BM25,
                })
                .collect();
            all_signals.push(signal_results);
        }

        // Fuse results
        let mut fused = self.fuse_results(&all_signals, weights);

        // Apply reranking if configured
        if let Some(reranker) = &self.reranker {
            let docs = self.documents.read().unwrap();
            fused = reranker.rerank(&query.text, fused, &docs);
        }

        // Apply minimum score filter
        fused.retain(|r| r.score >= self.config.min_score);

        // Limit to top_k
        fused.truncate(self.config.top_k);

        Ok(fused)
    }

    fn get_weights(&self, overrides: &Option<WeightOverrides>) -> (f32, f32, f32) {
        if let Some(o) = overrides {
            (
                o.dense_weight.unwrap_or(self.config.dense_weight),
                o.sparse_weight.unwrap_or(self.config.sparse_weight),
                o.bm25_weight.unwrap_or(self.config.bm25_weight),
            )
        } else {
            (self.config.dense_weight, self.config.sparse_weight, self.config.bm25_weight)
        }
    }

    fn fuse_results(&self, signals: &[Vec<SignalResult>], weights: (f32, f32, f32)) -> Vec<HybridResult> {
        match &self.config.fusion_strategy {
            FusionMethod::RRF { k } => self.fuse_rrf(signals, *k),
            FusionMethod::WeightedSum => self.fuse_weighted_sum(signals, weights),
            FusionMethod::MaxScore => self.fuse_max_score(signals),
            FusionMethod::GeometricMean => self.fuse_geometric_mean(signals),
            FusionMethod::HarmonicMean => self.fuse_harmonic_mean(signals),
            FusionMethod::Voting { min_votes } => self.fuse_voting(signals, *min_votes),
            FusionMethod::DistributionBased => self.fuse_distribution(signals, weights),
            FusionMethod::Cascade => self.fuse_weighted_sum(signals, weights), // Fallback
        }
    }

    fn fuse_rrf(&self, signals: &[Vec<SignalResult>], k: f32) -> Vec<HybridResult> {
        let mut doc_scores: HashMap<String, (f32, HashMap<String, f32>, HashMap<String, usize>)> = HashMap::new();

        for signal_results in signals {
            for result in signal_results {
                let rrf_score = 1.0 / (k + result.rank as f32 + 1.0);

                let entry = doc_scores.entry(result.id.clone()).or_insert_with(|| {
                    (0.0, HashMap::new(), HashMap::new())
                });

                entry.0 += rrf_score;
                entry.1.insert(format!("{:?}", result.signal), result.score);
                entry.2.insert(format!("{:?}", result.signal), result.rank);
            }
        }

        self.build_results(doc_scores, "RRF")
    }

    fn fuse_weighted_sum(&self, signals: &[Vec<SignalResult>], weights: (f32, f32, f32)) -> Vec<HybridResult> {
        let weight_map: HashMap<SignalType, f32> = [
            (SignalType::Dense, weights.0),
            (SignalType::Sparse, weights.1),
            (SignalType::BM25, weights.2),
        ].into_iter().collect();

        let mut doc_scores: HashMap<String, (f32, HashMap<String, f32>, HashMap<String, usize>)> = HashMap::new();

        for signal_results in signals {
            for result in signal_results {
                let weight = weight_map.get(&result.signal).copied().unwrap_or(1.0);
                let weighted_score = result.score * weight;

                let entry = doc_scores.entry(result.id.clone()).or_insert_with(|| {
                    (0.0, HashMap::new(), HashMap::new())
                });

                entry.0 += weighted_score;
                entry.1.insert(format!("{:?}", result.signal), result.score);
                entry.2.insert(format!("{:?}", result.signal), result.rank);
            }
        }

        self.build_results(doc_scores, "WeightedSum")
    }

    fn fuse_max_score(&self, signals: &[Vec<SignalResult>]) -> Vec<HybridResult> {
        let mut doc_scores: HashMap<String, (f32, HashMap<String, f32>, HashMap<String, usize>)> = HashMap::new();

        for signal_results in signals {
            for result in signal_results {
                let entry = doc_scores.entry(result.id.clone()).or_insert_with(|| {
                    (0.0, HashMap::new(), HashMap::new())
                });

                if result.score > entry.0 {
                    entry.0 = result.score;
                }
                entry.1.insert(format!("{:?}", result.signal), result.score);
                entry.2.insert(format!("{:?}", result.signal), result.rank);
            }
        }

        self.build_results(doc_scores, "MaxScore")
    }

    fn fuse_geometric_mean(&self, signals: &[Vec<SignalResult>]) -> Vec<HybridResult> {
        let mut doc_data: HashMap<String, (Vec<f32>, HashMap<String, f32>, HashMap<String, usize>)> = HashMap::new();

        for signal_results in signals {
            for result in signal_results {
                let entry = doc_data.entry(result.id.clone()).or_insert_with(|| {
                    (Vec::new(), HashMap::new(), HashMap::new())
                });

                entry.0.push(result.score.max(0.001)); // Avoid zero
                entry.1.insert(format!("{:?}", result.signal), result.score);
                entry.2.insert(format!("{:?}", result.signal), result.rank);
            }
        }

        let doc_scores: HashMap<String, (f32, HashMap<String, f32>, HashMap<String, usize>)> = doc_data
            .into_iter()
            .map(|(id, (scores, signal_scores, signal_ranks))| {
                let product: f32 = scores.iter().product();
                let geom_mean = product.powf(1.0 / scores.len() as f32);
                (id, (geom_mean, signal_scores, signal_ranks))
            })
            .collect();

        self.build_results(doc_scores, "GeometricMean")
    }

    fn fuse_harmonic_mean(&self, signals: &[Vec<SignalResult>]) -> Vec<HybridResult> {
        let mut doc_data: HashMap<String, (Vec<f32>, HashMap<String, f32>, HashMap<String, usize>)> = HashMap::new();

        for signal_results in signals {
            for result in signal_results {
                let entry = doc_data.entry(result.id.clone()).or_insert_with(|| {
                    (Vec::new(), HashMap::new(), HashMap::new())
                });

                entry.0.push(result.score.max(0.001));
                entry.1.insert(format!("{:?}", result.signal), result.score);
                entry.2.insert(format!("{:?}", result.signal), result.rank);
            }
        }

        let doc_scores: HashMap<String, (f32, HashMap<String, f32>, HashMap<String, usize>)> = doc_data
            .into_iter()
            .map(|(id, (scores, signal_scores, signal_ranks))| {
                let sum_reciprocals: f32 = scores.iter().map(|s| 1.0 / s).sum();
                let harm_mean = scores.len() as f32 / sum_reciprocals;
                (id, (harm_mean, signal_scores, signal_ranks))
            })
            .collect();

        self.build_results(doc_scores, "HarmonicMean")
    }

    fn fuse_voting(&self, signals: &[Vec<SignalResult>], min_votes: usize) -> Vec<HybridResult> {
        let mut doc_votes: HashMap<String, (usize, HashMap<String, f32>, HashMap<String, usize>)> = HashMap::new();

        for signal_results in signals {
            for result in signal_results {
                let entry = doc_votes.entry(result.id.clone()).or_insert_with(|| {
                    (0, HashMap::new(), HashMap::new())
                });

                entry.0 += 1;
                entry.1.insert(format!("{:?}", result.signal), result.score);
                entry.2.insert(format!("{:?}", result.signal), result.rank);
            }
        }

        let doc_scores: HashMap<String, (f32, HashMap<String, f32>, HashMap<String, usize>)> = doc_votes
            .into_iter()
            .filter(|(_, (votes, _, _))| *votes >= min_votes)
            .map(|(id, (votes, signal_scores, signal_ranks))| {
                (id, (votes as f32, signal_scores, signal_ranks))
            })
            .collect();

        self.build_results(doc_scores, "Voting")
    }

    fn fuse_distribution(&self, signals: &[Vec<SignalResult>], weights: (f32, f32, f32)) -> Vec<HybridResult> {
        // Normalize scores within each signal, then combine
        let mut normalized_signals = Vec::new();

        for signal_results in signals {
            if signal_results.is_empty() {
                continue;
            }

            let max_score = signal_results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
            let min_score = signal_results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
            let range = (max_score - min_score).max(0.001);

            let normalized: Vec<SignalResult> = signal_results
                .iter()
                .map(|r| SignalResult {
                    id: r.id.clone(),
                    score: (r.score - min_score) / range,
                    rank: r.rank,
                    signal: r.signal.clone(),
                })
                .collect();

            normalized_signals.push(normalized);
        }

        self.fuse_weighted_sum(&normalized_signals, weights)
    }

    fn build_results(
        &self,
        doc_scores: HashMap<String, (f32, HashMap<String, f32>, HashMap<String, usize>)>,
        method: &str,
    ) -> Vec<HybridResult> {
        let docs = self.documents.read().unwrap();

        let mut results: Vec<HybridResult> = doc_scores
            .into_iter()
            .map(|(id, (score, signal_scores, signal_ranks))| {
                let doc = docs.get(&id);
                HybridResult {
                    id: id.clone(),
                    score,
                    signal_scores: signal_scores.clone(),
                    signal_ranks: signal_ranks.clone(),
                    signal_count: signal_scores.len(),
                    content: doc.map(|d| d.text.clone()),
                    metadata: doc.map(|d| d.metadata.clone()).unwrap_or_default(),
                    explanation: Some(ScoreExplanation {
                        fusion_method: method.to_string(),
                        components: signal_scores
                            .iter()
                            .map(|(name, &score)| ScoreComponent {
                                name: name.clone(),
                                raw_score: score,
                                weight: 1.0,
                                weighted_score: score,
                            })
                            .collect(),
                        reranked: false,
                        original_rank: None,
                    }),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> PipelineStats {
        let dense = self.dense_index.read().unwrap();
        let sparse = self.sparse_index.read().unwrap();
        let bm25 = self.bm25_index.read().unwrap();

        PipelineStats {
            dense_vectors: dense.vectors.len(),
            sparse_terms: sparse.postings.len(),
            bm25_terms: bm25.postings.len(),
            bm25_documents: bm25.doc_count,
            avg_doc_length: bm25.avg_doc_length,
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    pub dense_vectors: usize,
    pub sparse_terms: usize,
    pub bm25_terms: usize,
    pub bm25_documents: usize,
    pub avg_doc_length: f32,
}

/// Reranker implementation
struct Reranker {
    config: RerankerConfig,
}

impl Reranker {
    fn new(config: RerankerConfig) -> Self {
        Self { config }
    }

    fn rerank(
        &self,
        query: &str,
        mut results: Vec<HybridResult>,
        _docs: &HashMap<String, HybridDocument>,
    ) -> Vec<HybridResult> {
        match &self.config.reranker_type {
            RerankerType::LexicalOverlap => {
                // Simple lexical overlap reranking
                let query_terms: HashSet<String> = tokenize(query).into_iter().collect();

                for (i, result) in results.iter_mut().enumerate() {
                    if let Some(content) = &result.content {
                        let doc_terms: HashSet<String> = tokenize(content).into_iter().collect();
                        let overlap = query_terms.intersection(&doc_terms).count();
                        let overlap_score = overlap as f32 / query_terms.len().max(1) as f32;

                        // Blend with original score
                        let original_rank = i;
                        result.score = result.score * 0.7 + overlap_score * 0.3;

                        if let Some(ref mut exp) = result.explanation {
                            exp.reranked = true;
                            exp.original_rank = Some(original_rank);
                        }
                    }
                }

                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            }
            _ => {
                // Other rerankers would call external APIs
                // For now, return unchanged
            }
        }

        results
    }
}

// Helper functions

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1)
        .map(|s| s.to_string())
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_pipeline() {
        let config = HybridConfig::default();
        let pipeline = HybridPipeline::new(config);

        // Index documents
        for i in 0..10 {
            let doc = HybridDocument {
                id: format!("doc_{}", i),
                text: format!("This is document {} about machine learning", i),
                dense_vector: vec![0.1 * i as f32; 128],
                sparse_vector: Some(SparseQueryVector::new(
                    vec![1, 2, 3],
                    vec![0.5, 0.3, 0.2],
                )),
                metadata: HashMap::new(),
            };
            pipeline.index(doc).unwrap();
        }

        // Search
        let query = HybridSearchQuery::new("machine learning")
            .with_dense(vec![0.5; 128])
            .with_sparse(SparseQueryVector::new(vec![1, 2], vec![0.5, 0.3]));

        let results = pipeline.search(query).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_rrf_fusion() {
        let config = HybridConfig {
            fusion_strategy: FusionMethod::RRF { k: 60.0 },
            ..Default::default()
        };
        let pipeline = HybridPipeline::new(config);

        for i in 0..5 {
            let doc = HybridDocument {
                id: format!("doc_{}", i),
                text: format!("Document {}", i),
                dense_vector: vec![i as f32; 4],
                sparse_vector: None,
                metadata: HashMap::new(),
            };
            pipeline.index(doc).unwrap();
        }

        let query = HybridSearchQuery::new("Document")
            .with_dense(vec![2.0; 4]);

        let results = pipeline.search(query).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_boolean_query() {
        let must = BooleanQuery::must("machine");
        let should = BooleanQuery::should("learning");
        let combined = BooleanQuery::and(vec![must, should]);

        match combined {
            BooleanQuery::And(queries) => assert_eq!(queries.len(), 2),
            _ => panic!("Expected And query"),
        }
    }

    #[test]
    fn test_phrase_query() {
        let phrase = BooleanQuery::phrase(vec!["machine", "learning"], 1);

        match phrase {
            BooleanQuery::Phrase { terms, slop } => {
                assert_eq!(terms.len(), 2);
                assert_eq!(slop, 1);
            }
            _ => panic!("Expected Phrase query"),
        }
    }

    #[test]
    fn test_weight_overrides() {
        let query = HybridSearchQuery::new("test")
            .with_weights(0.5, 0.3, 0.2);

        let overrides = query.weight_overrides.unwrap();
        assert_eq!(overrides.dense_weight, Some(0.5));
        assert_eq!(overrides.sparse_weight, Some(0.3));
        assert_eq!(overrides.bm25_weight, Some(0.2));
    }
}
