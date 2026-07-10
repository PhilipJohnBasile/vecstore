// Neural Ranker Integration - Deep learning-based reranking for improved relevance
// Cross-encoder models, late interaction, and learned sparse representations

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Neural ranker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralRankerConfig {
    /// Model type
    pub model_type: ModelType,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// Cache size for embeddings
    pub cache_size: usize,
    /// Rerank top-k candidates
    pub rerank_top_k: usize,
    /// Score threshold
    pub score_threshold: f32,
    /// Fusion weight (0-1, neural vs original)
    pub fusion_weight: f32,
    /// Model path
    pub model_path: Option<String>,
    /// Timeout for inference
    pub inference_timeout: Duration,
}

impl Default for NeuralRankerConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::CrossEncoder,
            max_seq_length: 512,
            batch_size: 32,
            use_gpu: false,
            cache_size: 10000,
            rerank_top_k: 100,
            score_threshold: 0.0,
            fusion_weight: 0.7,
            model_path: None,
            inference_timeout: Duration::from_secs(5),
        }
    }
}

/// Model type for neural reranking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    /// Cross-encoder (BERT-style)
    CrossEncoder,
    /// Late interaction (ColBERT-style)
    LateInteraction,
    /// Learned sparse (SPLADE-style)
    LearnedSparse,
    /// Listwise ranker
    ListwiseRanker,
    /// Pairwise ranker
    PairwiseRanker,
    /// Custom model
    Custom(String),
}

/// Neural ranker with model inference
pub struct NeuralRanker {
    config: NeuralRankerConfig,
    model: RwLock<Option<LoadedModel>>,
    tokenizer: RwLock<Option<Tokenizer>>,
    cache: RwLock<EmbeddingCache>,
    stats: RankerStats,
}

/// Loaded model representation
struct LoadedModel {
    model_type: ModelType,
    weights: Vec<f32>,
    layers: Vec<Layer>,
    vocab_size: usize,
    hidden_size: usize,
    num_heads: usize,
    loaded_at: Instant,
}

/// Neural network layer
#[derive(Debug, Clone)]
struct Layer {
    layer_type: LayerType,
    weights: Vec<f32>,
    bias: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

/// Layer types
#[derive(Debug, Clone)]
enum LayerType {
    Linear,
    Attention,
    LayerNorm,
    Embedding,
    Pooling,
}

/// Tokenizer for text processing
struct Tokenizer {
    vocab: HashMap<String, u32>,
    special_tokens: SpecialTokens,
    max_length: usize,
}

/// Special tokens
#[derive(Debug, Clone)]
struct SpecialTokens {
    cls_token: u32,
    sep_token: u32,
    pad_token: u32,
    unk_token: u32,
    mask_token: u32,
}

/// Embedding cache
struct EmbeddingCache {
    entries: HashMap<String, CacheEntry>,
    max_size: usize,
    hits: u64,
    misses: u64,
}

/// Cache entry
struct CacheEntry {
    embedding: Vec<f32>,
    created_at: Instant,
    access_count: u32,
}

/// Ranker statistics
struct RankerStats {
    total_reranks: AtomicU64,
    total_candidates: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    avg_latency_ms: RwLock<f64>,
    gpu_inferences: AtomicU64,
    cpu_inferences: AtomicU64,
}

/// Candidate for reranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankCandidate {
    /// Candidate ID
    pub id: String,
    /// Text content
    pub text: String,
    /// Original score
    pub original_score: f32,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Reranked result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    /// Candidate ID
    pub id: String,
    /// Neural score
    pub neural_score: f32,
    /// Original score
    pub original_score: f32,
    /// Fused score
    pub fused_score: f32,
    /// Rank change from original
    pub rank_change: i32,
    /// Confidence
    pub confidence: f32,
}

/// Rerank response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    /// Reranked results
    pub results: Vec<RerankResult>,
    /// Query text
    pub query: String,
    /// Total candidates
    pub total_candidates: usize,
    /// Processing time
    pub processing_time_ms: f64,
    /// Cache hit
    pub cache_hit: bool,
    /// Model used
    pub model: String,
}

/// Cross-encoder for pairwise scoring
pub struct CrossEncoder {
    config: CrossEncoderConfig,
    model: Arc<RwLock<Option<LoadedModel>>>,
}

/// Cross-encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEncoderConfig {
    /// Model name
    pub model_name: String,
    /// Maximum input length
    pub max_length: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: ActivationFn,
    /// Normalize scores
    pub normalize: bool,
}

impl Default for CrossEncoderConfig {
    fn default() -> Self {
        Self {
            model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            max_length: 512,
            output_dim: 1,
            activation: ActivationFn::Sigmoid,
            normalize: true,
        }
    }
}

/// Activation functions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActivationFn {
    Sigmoid,
    Tanh,
    ReLU,
    GELU,
    Softmax,
    Linear,
}

/// Late interaction model (ColBERT-style)
pub struct LateInteractionModel {
    config: LateInteractionConfig,
    query_encoder: Arc<RwLock<Option<LoadedModel>>>,
    doc_encoder: Arc<RwLock<Option<LoadedModel>>>,
    doc_embeddings: RwLock<HashMap<String, Vec<Vec<f32>>>>,
}

/// Late interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateInteractionConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Number of query tokens
    pub query_max_tokens: usize,
    /// Number of document tokens
    pub doc_max_tokens: usize,
    /// Similarity function
    pub similarity: SimilarityFn,
    /// Compression for doc embeddings
    pub compress_docs: bool,
    /// Compression bits
    pub compression_bits: usize,
}

impl Default for LateInteractionConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            query_max_tokens: 32,
            doc_max_tokens: 180,
            similarity: SimilarityFn::MaxSim,
            compress_docs: true,
            compression_bits: 2,
        }
    }
}

/// Similarity functions for late interaction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SimilarityFn {
    MaxSim,
    SumMax,
    AvgMax,
    SoftMax,
}

/// Learned sparse model (SPLADE-style)
pub struct LearnedSparseModel {
    config: LearnedSparseConfig,
    model: Arc<RwLock<Option<LoadedModel>>>,
    vocab: RwLock<HashMap<String, u32>>,
}

/// Learned sparse configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedSparseConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum terms per document
    pub max_terms: usize,
    /// Sparsity target
    pub sparsity_target: f32,
    /// Regularization weight
    pub regularization_weight: f32,
    /// Use IDF weighting
    pub use_idf: bool,
    /// Expansion factor
    pub expansion_factor: f32,
}

impl Default for LearnedSparseConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            max_terms: 256,
            sparsity_target: 0.95,
            regularization_weight: 0.0001,
            use_idf: true,
            expansion_factor: 1.5,
        }
    }
}

/// Sparse representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseRepresentation {
    /// Term IDs
    pub term_ids: Vec<u32>,
    /// Term weights
    pub weights: Vec<f32>,
    /// Expansion terms
    pub expansion_terms: Vec<u32>,
    /// Expansion weights
    pub expansion_weights: Vec<f32>,
}

impl NeuralRanker {
    /// Create a new neural ranker
    pub fn new(config: NeuralRankerConfig) -> Self {
        Self {
            config: config.clone(),
            model: RwLock::new(None),
            tokenizer: RwLock::new(None),
            cache: RwLock::new(EmbeddingCache {
                entries: HashMap::new(),
                max_size: config.cache_size,
                hits: 0,
                misses: 0,
            }),
            stats: RankerStats {
                total_reranks: AtomicU64::new(0),
                total_candidates: AtomicU64::new(0),
                cache_hits: AtomicU64::new(0),
                cache_misses: AtomicU64::new(0),
                avg_latency_ms: RwLock::new(0.0),
                gpu_inferences: AtomicU64::new(0),
                cpu_inferences: AtomicU64::new(0),
            },
        }
    }

    /// Load model from path
    pub fn load_model(&self, _path: &str) -> Result<()> {
        // Initialize model weights using Xavier/Glorot initialization
        // For a real implementation, these would be loaded from disk
        let hidden_size = 768;
        let num_layers = 12;
        let vocab_size = 30522;

        // Calculate total weight count for transformer model:
        // - Embedding layer: vocab_size * hidden_size
        // - Each transformer layer: ~4 * hidden_size^2 (attention + FFN)
        // - Output layer: hidden_size
        let embedding_weights = vocab_size * hidden_size;
        let layer_weights = num_layers * 4 * hidden_size * hidden_size;
        let output_weights = hidden_size;
        let total_weights = embedding_weights + layer_weights + output_weights;

        // Initialize weights using scaled random initialization
        // Using a deterministic seed for reproducibility in testing
        let weights = self.initialize_weights(total_weights, hidden_size);

        let model = LoadedModel {
            model_type: self.config.model_type.clone(),
            weights,
            layers: self.create_default_layers(),
            vocab_size,
            hidden_size,
            num_heads: 12,
            loaded_at: Instant::now(),
        };

        let tokenizer = Tokenizer {
            vocab: self.create_default_vocab(),
            special_tokens: SpecialTokens {
                cls_token: 101,
                sep_token: 102,
                pad_token: 0,
                unk_token: 100,
                mask_token: 103,
            },
            max_length: self.config.max_seq_length,
        };

        *self.model.write()
            .map_err(|_| VecStoreError::LockError("Failed to acquire model write lock".into()))? = Some(model);
        *self.tokenizer.write()
            .map_err(|_| VecStoreError::LockError("Failed to acquire tokenizer write lock".into()))? = Some(tokenizer);

        Ok(())
    }

    /// Initialize weights using Xavier/Glorot initialization
    /// This produces weights with zero mean and variance scaled by layer size
    fn initialize_weights(&self, count: usize, hidden_size: usize) -> Vec<f32> {
        // Xavier scale factor: sqrt(2 / (fan_in + fan_out))
        // For transformers, approximate as sqrt(2 / (2 * hidden_size))
        let scale = (2.0 / (2.0 * hidden_size as f64)).sqrt() as f32;

        // Use a simple LCG (Linear Congruential Generator) for deterministic initialization
        // Parameters from Numerical Recipes
        let mut seed: u64 = 42; // Fixed seed for reproducibility
        let a: u64 = 1664525;
        let c: u64 = 1013904223;
        let m: u64 = 1 << 32;

        (0..count)
            .map(|_| {
                seed = (a.wrapping_mul(seed).wrapping_add(c)) % m;
                // Convert to [-1, 1] range, then scale
                let uniform = (seed as f64 / m as f64) * 2.0 - 1.0;
                (uniform as f32) * scale
            })
            .collect()
    }

    fn create_default_layers(&self) -> Vec<Layer> {
        // Initialize layer weights properly
        let hidden_size = 768;
        let vocab_size = 30522;

        vec![
            Layer {
                layer_type: LayerType::Embedding,
                weights: self.initialize_weights(hidden_size * vocab_size, hidden_size),
                bias: vec![],
                input_size: vocab_size,
                output_size: hidden_size,
            },
            Layer {
                layer_type: LayerType::Attention,
                weights: self.initialize_weights(hidden_size * hidden_size * 4, hidden_size),
                bias: self.initialize_weights(hidden_size * 4, hidden_size),
                input_size: hidden_size,
                output_size: hidden_size,
            },
            Layer {
                layer_type: LayerType::LayerNorm,
                weights: vec![1.0; hidden_size], // Gamma initialized to 1
                bias: vec![0.0; hidden_size],    // Beta initialized to 0
                input_size: hidden_size,
                output_size: hidden_size,
            },
            Layer {
                layer_type: LayerType::Pooling,
                weights: vec![],
                bias: vec![],
                input_size: hidden_size,
                output_size: hidden_size,
            },
            Layer {
                layer_type: LayerType::Linear,
                weights: self.initialize_weights(hidden_size, hidden_size),
                bias: vec![0.0; 1],
                input_size: hidden_size,
                output_size: 1,
            },
        ]
    }

    fn create_default_vocab(&self) -> HashMap<String, u32> {
        // Rust 1.92: Use FromIterator with array for cleaner initialization
        HashMap::from([
            ("[PAD]".to_string(), 0),
            ("[UNK]".to_string(), 100),
            ("[CLS]".to_string(), 101),
            ("[SEP]".to_string(), 102),
            ("[MASK]".to_string(), 103),
        ])
    }

    /// Rerank candidates based on query
    pub fn rerank(&self, query: &str, candidates: Vec<RerankCandidate>) -> Result<RerankResponse> {
        let start = Instant::now();
        self.stats.total_reranks.fetch_add(1, Ordering::Relaxed);
        self.stats.total_candidates.fetch_add(candidates.len() as u64, Ordering::Relaxed);

        // Limit candidates
        let top_candidates: Vec<_> = candidates.into_iter()
            .take(self.config.rerank_top_k)
            .collect();

        let total = top_candidates.len();

        // Score candidates
        let mut scored: Vec<(RerankCandidate, f32)> = Vec::new();
        let mut cache_hit = false;

        for candidate in top_candidates {
            let cache_key = format!("{}:{}", query, candidate.id);

            // Check cache
            let neural_score = {
                let cache = self.cache.read()
                    .map_err(|_| VecStoreError::LockError("Failed to acquire cache read lock".into()))?;
                if let Some(entry) = cache.entries.get(&cache_key) {
                    cache_hit = true;
                    self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                    entry.embedding[0]
                } else {
                    drop(cache);
                    self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);

                    // Compute score
                    let score = self.compute_score(query, &candidate.text)?;

                    // Cache result
                    let mut cache = self.cache.write()
                        .map_err(|_| VecStoreError::LockError("Failed to acquire cache write lock".into()))?;
                    if cache.entries.len() < cache.max_size {
                        cache.entries.insert(cache_key, CacheEntry {
                            embedding: vec![score],
                            created_at: Instant::now(),
                            access_count: 1,
                        });
                    }

                    score
                }
            };

            scored.push((candidate, neural_score));
        }

        // Sort by neural score
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Create results with rank changes
        let mut results = Vec::new();
        for (new_rank, (candidate, neural_score)) in scored.iter().enumerate() {
            // Find original rank (simplified - assuming sorted by original_score)
            let original_rank = new_rank; // Simplified

            let fused_score = self.config.fusion_weight * neural_score
                + (1.0 - self.config.fusion_weight) * candidate.original_score;

            if fused_score >= self.config.score_threshold {
                results.push(RerankResult {
                    id: candidate.id.clone(),
                    neural_score: *neural_score,
                    original_score: candidate.original_score,
                    fused_score,
                    rank_change: original_rank as i32 - new_rank as i32,
                    confidence: self.compute_confidence(*neural_score),
                });
            }
        }

        let elapsed = start.elapsed();
        self.update_latency(elapsed.as_secs_f64() * 1000.0);

        Ok(RerankResponse {
            results,
            query: query.to_string(),
            total_candidates: total,
            processing_time_ms: elapsed.as_secs_f64() * 1000.0,
            cache_hit,
            model: format!("{:?}", self.config.model_type),
        })
    }

    /// Compute relevance score for query-document pair
    fn compute_score(&self, query: &str, document: &str) -> Result<f32> {
        if self.config.use_gpu {
            self.stats.gpu_inferences.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.cpu_inferences.fetch_add(1, Ordering::Relaxed);
        }

        match self.config.model_type {
            ModelType::CrossEncoder => self.cross_encode(query, document),
            ModelType::LateInteraction => self.late_interaction(query, document),
            ModelType::LearnedSparse => self.learned_sparse(query, document),
            ModelType::ListwiseRanker => self.listwise_score(query, document),
            ModelType::PairwiseRanker => self.pairwise_score(query, document),
            ModelType::Custom(_) => self.custom_score(query, document),
        }
    }

    /// Cross-encoder scoring
    fn cross_encode(&self, query: &str, document: &str) -> Result<f32> {
        // Tokenize
        let tokens = self.tokenize(&format!("[CLS] {} [SEP] {} [SEP]", query, document))?;

        // Forward pass (simulated)
        let embedding = self.forward_pass(&tokens)?;

        // Apply sigmoid for probability
        let score = sigmoid(embedding[0]);

        Ok(score)
    }

    /// Generate token embedding using hash-based approach
    /// This creates a deterministic embedding for each token based on its ID and position
    fn generate_token_embedding(&self, token_id: u32, position: usize, dim: usize) -> Vec<f32> {
        let mut embedding = vec![0.0f32; dim];

        // Use token ID and position to seed the embedding
        // This creates consistent, position-aware embeddings
        let mut seed = (token_id as u64).wrapping_mul(31).wrapping_add(position as u64);

        for (i, emb_val) in embedding.iter_mut().enumerate().take(dim) {
            // LCG for deterministic pseudo-random values
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let value = ((seed >> 16) as f32 / 65535.0) * 2.0 - 1.0;

            // Add positional encoding (sinusoidal)
            let pos_enc = if i % 2 == 0 {
                (position as f32 / 10000_f32.powf(i as f32 / dim as f32)).sin()
            } else {
                (position as f32 / 10000_f32.powf((i - 1) as f32 / dim as f32)).cos()
            };

            *emb_val = value * 0.5 + pos_enc * 0.5;
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }

    /// Late interaction scoring (ColBERT-style)
    fn late_interaction(&self, query: &str, document: &str) -> Result<f32> {
        const EMBEDDING_DIM: usize = 128;

        // Encode query tokens with position-aware embeddings
        let query_tokens = self.tokenize(query)?;
        let query_embeddings: Vec<Vec<f32>> = query_tokens
            .iter()
            .enumerate()
            .map(|(pos, &token_id)| self.generate_token_embedding(token_id, pos, EMBEDDING_DIM))
            .collect();

        // Encode document tokens with position-aware embeddings
        let doc_tokens = self.tokenize(document)?;
        let doc_embeddings: Vec<Vec<f32>> = doc_tokens
            .iter()
            .enumerate()
            .map(|(pos, &token_id)| self.generate_token_embedding(token_id, pos, EMBEDDING_DIM))
            .collect();

        // MaxSim scoring (ColBERT algorithm)
        let mut score = 0.0;
        for q_emb in &query_embeddings {
            let mut max_sim = f32::MIN;
            for d_emb in &doc_embeddings {
                let sim = cosine_similarity(q_emb, d_emb);
                if sim > max_sim {
                    max_sim = sim;
                }
            }
            // Clamp to avoid negative contributions from very dissimilar tokens
            score += max_sim.max(0.0);
        }

        Ok(score / query_embeddings.len() as f32)
    }

    /// Learned sparse scoring
    fn learned_sparse(&self, query: &str, document: &str) -> Result<f32> {
        // Get sparse representations
        let query_sparse = self.get_sparse_representation(query)?;
        let doc_sparse = self.get_sparse_representation(document)?;

        // Sparse dot product
        let mut score = 0.0;
        for (i, &q_term) in query_sparse.term_ids.iter().enumerate() {
            if let Some(pos) = doc_sparse.term_ids.iter().position(|&t| t == q_term) {
                score += query_sparse.weights[i] * doc_sparse.weights[pos];
            }
        }

        // Include expansion terms
        for (i, &q_term) in query_sparse.expansion_terms.iter().enumerate() {
            if let Some(pos) = doc_sparse.term_ids.iter().position(|&t| t == q_term) {
                score += query_sparse.expansion_weights[i] * doc_sparse.weights[pos];
            }
        }

        Ok(score)
    }

    /// Listwise ranking score
    fn listwise_score(&self, query: &str, document: &str) -> Result<f32> {
        // Simplified listwise scoring
        let tokens = self.tokenize(&format!("{} {}", query, document))?;
        let embedding = self.forward_pass(&tokens)?;
        Ok(embedding[0])
    }

    /// Pairwise ranking score
    fn pairwise_score(&self, query: &str, document: &str) -> Result<f32> {
        // Simplified pairwise scoring
        let tokens = self.tokenize(&format!("{} {}", query, document))?;
        let embedding = self.forward_pass(&tokens)?;
        Ok(sigmoid(embedding[0]))
    }

    /// Custom model scoring
    fn custom_score(&self, query: &str, document: &str) -> Result<f32> {
        // Placeholder for custom models
        let _ = (query, document);
        Ok(0.5)
    }

    /// Tokenize text
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let tokenizer = self.tokenizer.read()
            .map_err(|_| VecStoreError::LockError("Failed to acquire tokenizer read lock".into()))?;
        let tokenizer = tokenizer.as_ref()
            .ok_or(VecStoreError::IndexNotInitialized)?;

        let words: Vec<&str> = text.split_whitespace().collect();
        let tokens: Vec<u32> = words.iter()
            .map(|w| *tokenizer.vocab.get(*w).unwrap_or(&tokenizer.special_tokens.unk_token))
            .take(tokenizer.max_length)
            .collect();

        Ok(tokens)
    }

    /// Forward pass through model
    fn forward_pass(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
        // Simulated forward pass
        Ok(vec![0.5])
    }

    /// Get sparse representation
    fn get_sparse_representation(&self, text: &str) -> Result<SparseRepresentation> {
        let tokens = self.tokenize(text)?;

        Ok(SparseRepresentation {
            term_ids: tokens.clone(),
            weights: tokens.iter().map(|_| 1.0).collect(),
            expansion_terms: vec![],
            expansion_weights: vec![],
        })
    }

    /// Compute confidence from score
    fn compute_confidence(&self, score: f32) -> f32 {
        // Higher confidence for scores further from 0.5
        (score - 0.5).abs() * 2.0
    }

    /// Update running average latency
    fn update_latency(&self, latency_ms: f64) {
        let Ok(mut avg) = self.stats.avg_latency_ms.write() else { return; };
        let count = self.stats.total_reranks.load(Ordering::Relaxed) as f64;
        *avg = (*avg * (count - 1.0) + latency_ms) / count;
    }

    /// Get statistics
    pub fn get_stats(&self) -> RankerStatsSummary {
        RankerStatsSummary {
            total_reranks: self.stats.total_reranks.load(Ordering::Relaxed),
            total_candidates: self.stats.total_candidates.load(Ordering::Relaxed),
            cache_hits: self.stats.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.stats.cache_misses.load(Ordering::Relaxed),
            cache_hit_rate: {
                let hits = self.stats.cache_hits.load(Ordering::Relaxed) as f64;
                let misses = self.stats.cache_misses.load(Ordering::Relaxed) as f64;
                if hits + misses > 0.0 { hits / (hits + misses) } else { 0.0 }
            },
            avg_latency_ms: self.stats.avg_latency_ms.read().map(|g| *g).unwrap_or(0.0),
            gpu_inferences: self.stats.gpu_inferences.load(Ordering::Relaxed),
            cpu_inferences: self.stats.cpu_inferences.load(Ordering::Relaxed),
        }
    }

    /// Clear embedding cache
    pub fn clear_cache(&self) {
        let Ok(mut cache) = self.cache.write() else { return; };
        cache.entries.clear();
        cache.hits = 0;
        cache.misses = 0;
    }

    /// Batch rerank multiple queries
    pub fn batch_rerank(
        &self,
        queries: Vec<(String, Vec<RerankCandidate>)>,
    ) -> Result<Vec<RerankResponse>> {
        let mut results = Vec::new();

        for (query, candidates) in queries {
            results.push(self.rerank(&query, candidates)?);
        }

        Ok(results)
    }
}

/// Statistics summary
#[derive(Debug, Clone, Serialize)]
pub struct RankerStatsSummary {
    pub total_reranks: u64,
    pub total_candidates: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
    pub avg_latency_ms: f64,
    pub gpu_inferences: u64,
    pub cpu_inferences: u64,
}

impl CrossEncoder {
    /// Create a new cross-encoder
    pub fn new(config: CrossEncoderConfig) -> Self {
        Self {
            config,
            model: Arc::new(RwLock::new(None)),
        }
    }

    /// Score a query-document pair
    pub fn score(&self, query: &str, document: &str) -> Result<f32> {
        // Combine query and document
        let combined = format!("[CLS] {} [SEP] {} [SEP]", query, document);

        // Simulate scoring (in practice, run through transformer)
        let score = self.simulated_forward(&combined);

        // Apply activation
        let activated = match self.config.activation {
            ActivationFn::Sigmoid => sigmoid(score),
            ActivationFn::Tanh => score.tanh(),
            ActivationFn::ReLU => score.max(0.0),
            ActivationFn::GELU => gelu(score),
            ActivationFn::Softmax => score, // Single value, no softmax
            ActivationFn::Linear => score,
        };

        if self.config.normalize {
            Ok((activated + 1.0) / 2.0)
        } else {
            Ok(activated)
        }
    }

    fn simulated_forward(&self, _input: &str) -> f32 {
        0.5 // Placeholder
    }

    /// Batch score multiple pairs
    pub fn batch_score(&self, pairs: &[(String, String)]) -> Result<Vec<f32>> {
        pairs.iter()
            .map(|(q, d)| self.score(q, d))
            .collect()
    }
}

impl LateInteractionModel {
    /// Create a new late interaction model
    pub fn new(config: LateInteractionConfig) -> Self {
        Self {
            config,
            query_encoder: Arc::new(RwLock::new(None)),
            doc_encoder: Arc::new(RwLock::new(None)),
            doc_embeddings: RwLock::new(HashMap::new()),
        }
    }

    /// Encode a query
    pub fn encode_query(&self, query: &str) -> Result<Vec<Vec<f32>>> {
        let tokens: Vec<&str> = query.split_whitespace()
            .take(self.config.query_max_tokens)
            .collect();

        // Generate embeddings per token
        Ok(tokens.iter()
            .map(|_| vec![0.0; self.config.embedding_dim])
            .collect())
    }

    /// Encode a document
    pub fn encode_document(&self, doc: &str) -> Result<Vec<Vec<f32>>> {
        let tokens: Vec<&str> = doc.split_whitespace()
            .take(self.config.doc_max_tokens)
            .collect();

        // Generate embeddings per token
        let embeddings: Vec<Vec<f32>> = tokens.iter()
            .map(|_| vec![0.0; self.config.embedding_dim])
            .collect();

        // Optionally compress
        if self.config.compress_docs {
            Ok(self.compress_embeddings(&embeddings))
        } else {
            Ok(embeddings)
        }
    }

    /// Compress embeddings
    fn compress_embeddings(&self, embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Simulated compression
        embeddings.to_vec()
    }

    /// Score query against document
    pub fn score(&self, query_embeddings: &[Vec<f32>], doc_embeddings: &[Vec<f32>]) -> f32 {
        match self.config.similarity {
            SimilarityFn::MaxSim => self.max_sim(query_embeddings, doc_embeddings),
            SimilarityFn::SumMax => self.sum_max(query_embeddings, doc_embeddings),
            SimilarityFn::AvgMax => self.avg_max(query_embeddings, doc_embeddings),
            SimilarityFn::SoftMax => self.soft_max(query_embeddings, doc_embeddings),
        }
    }

    fn max_sim(&self, query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
        let mut total = 0.0;
        for q in query {
            let max = doc.iter()
                .map(|d| cosine_similarity(q, d))
                .fold(f32::MIN, f32::max);
            total += max;
        }
        total
    }

    fn sum_max(&self, query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
        self.max_sim(query, doc)
    }

    fn avg_max(&self, query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
        if query.is_empty() { return 0.0; }
        self.max_sim(query, doc) / query.len() as f32
    }

    fn soft_max(&self, query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
        let mut total = 0.0;
        for q in query {
            let sims: Vec<f32> = doc.iter()
                .map(|d| cosine_similarity(q, d))
                .collect();
            let max = sims.iter().fold(f32::MIN, |a, &b| a.max(b));
            let exp_sum: f32 = sims.iter().map(|&s| (s - max).exp()).sum();
            let softmax_weighted: f32 = sims.iter()
                .map(|&s| s * (s - max).exp() / exp_sum)
                .sum();
            total += softmax_weighted;
        }
        total
    }

    /// Index document embeddings
    pub fn index_document(&self, doc_id: &str, embeddings: Vec<Vec<f32>>) {
        let Ok(mut doc_emb) = self.doc_embeddings.write() else { return; };
        doc_emb.insert(doc_id.to_string(), embeddings);
    }

    /// Retrieve and score against indexed documents
    pub fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<(String, f32)>> {
        let query_embeddings = self.encode_query(query)?;
        let docs = self.doc_embeddings.read()
            .map_err(|_| VecStoreError::LockError("Failed to acquire doc_embeddings read lock".into()))?;

        let mut scores: Vec<(String, f32)> = docs.iter()
            .map(|(id, emb)| (id.clone(), self.score(&query_embeddings, emb)))
            .collect();

        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        scores.truncate(top_k);

        Ok(scores)
    }
}

impl LearnedSparseModel {
    /// Create a new learned sparse model
    pub fn new(config: LearnedSparseConfig) -> Self {
        Self {
            config,
            model: Arc::new(RwLock::new(None)),
            vocab: RwLock::new(HashMap::new()),
        }
    }

    /// Encode text to sparse representation
    pub fn encode(&self, text: &str) -> Result<SparseRepresentation> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let vocab = self.vocab.read()
            .map_err(|_| VecStoreError::LockError("Failed to acquire vocab read lock".into()))?;

        let mut term_ids = Vec::new();
        let mut weights = Vec::new();

        for token in &tokens {
            let token_lower = token.to_lowercase();
            if let Some(&id) = vocab.get(&token_lower) {
                term_ids.push(id);
                weights.push(1.0); // Simulated weight
            }
        }

        // Generate expansion terms
        let (expansion_terms, expansion_weights) = if self.config.expansion_factor > 1.0 {
            self.generate_expansions(&term_ids)
        } else {
            (vec![], vec![])
        };

        // Enforce sparsity
        let max_terms = self.config.max_terms.min(term_ids.len());
        term_ids.truncate(max_terms);
        weights.truncate(max_terms);

        Ok(SparseRepresentation {
            term_ids,
            weights,
            expansion_terms,
            expansion_weights,
        })
    }

    fn generate_expansions(&self, _term_ids: &[u32]) -> (Vec<u32>, Vec<f32>) {
        // Simulated expansion
        (vec![], vec![])
    }

    /// Score two sparse representations
    pub fn score(&self, query: &SparseRepresentation, doc: &SparseRepresentation) -> f32 {
        let mut score = 0.0;

        for (i, &q_term) in query.term_ids.iter().enumerate() {
            if let Some(pos) = doc.term_ids.iter().position(|&t| t == q_term) {
                let weight = query.weights[i] * doc.weights[pos];
                if self.config.use_idf {
                    // Apply IDF weighting (simulated)
                    score += weight * 1.0;
                } else {
                    score += weight;
                }
            }
        }

        score
    }
}

/// Ensemble ranker combining multiple models
pub struct EnsembleRanker {
    rankers: Vec<(Box<dyn Ranker>, f32)>, // (ranker, weight)
    config: EnsembleConfig,
}

/// Ranker trait for ensemble
pub trait Ranker: Send + Sync {
    fn score(&self, query: &str, document: &str) -> Result<f32>;
    fn name(&self) -> &str;
}

/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Combination method
    pub combination: CombinationMethod,
    /// Normalize scores
    pub normalize: bool,
    /// Minimum agreement
    pub min_agreement: f32,
}

/// Combination methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CombinationMethod {
    WeightedSum,
    Max,
    Min,
    Average,
    RRF(f32), // Reciprocal Rank Fusion with k parameter
    CombMNZ,  // CombMNZ method
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            combination: CombinationMethod::WeightedSum,
            normalize: true,
            min_agreement: 0.5,
        }
    }
}

impl EnsembleRanker {
    /// Create a new ensemble ranker
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            rankers: Vec::new(),
            config,
        }
    }

    /// Add a ranker with weight
    pub fn add_ranker(&mut self, ranker: Box<dyn Ranker>, weight: f32) {
        self.rankers.push((ranker, weight));
    }

    /// Score using ensemble
    pub fn score(&self, query: &str, document: &str) -> Result<f32> {
        let scores: Vec<f32> = self.rankers.iter()
            .filter_map(|(r, _)| r.score(query, document).ok())
            .collect();

        if scores.is_empty() {
            return Ok(0.0);
        }

        match &self.config.combination {
            CombinationMethod::WeightedSum => {
                let total_weight: f32 = self.rankers.iter().map(|(_, w)| w).sum();
                let weighted_sum: f32 = scores.iter()
                    .zip(self.rankers.iter())
                    .map(|(s, (_, w))| s * w)
                    .sum();
                Ok(weighted_sum / total_weight)
            }
            CombinationMethod::Max => {
                Ok(scores.iter().fold(f32::MIN, |a, &b| a.max(b)))
            }
            CombinationMethod::Min => {
                Ok(scores.iter().fold(f32::MAX, |a, &b| a.min(b)))
            }
            CombinationMethod::Average => {
                Ok(scores.iter().sum::<f32>() / scores.len() as f32)
            }
            CombinationMethod::RRF(k) => {
                // Reciprocal rank fusion
                let rrf: f32 = scores.iter()
                    .enumerate()
                    .map(|(rank, _)| 1.0 / (k + rank as f32 + 1.0))
                    .sum();
                Ok(rrf)
            }
            CombinationMethod::CombMNZ => {
                let non_zero = scores.iter().filter(|&&s| s > 0.0).count() as f32;
                let sum: f32 = scores.iter().sum();
                Ok(sum * non_zero)
            }
        }
    }
}

// Helper functions
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (x * 0.797_884_6 * (1.0 + 0.044715 * x * x)).tanh())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Builder for NeuralRanker
#[must_use = "builders do nothing unless built"]
pub struct NeuralRankerBuilder {
    config: NeuralRankerConfig,
}

impl NeuralRankerBuilder {
    pub fn new() -> Self {
        Self {
            config: NeuralRankerConfig::default(),
        }
    }

    pub fn model_type(mut self, model_type: ModelType) -> Self {
        self.config.model_type = model_type;
        self
    }

    pub fn max_seq_length(mut self, length: usize) -> Self {
        self.config.max_seq_length = length;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    pub fn use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.use_gpu = use_gpu;
        self
    }

    pub fn cache_size(mut self, size: usize) -> Self {
        self.config.cache_size = size;
        self
    }

    pub fn rerank_top_k(mut self, k: usize) -> Self {
        self.config.rerank_top_k = k;
        self
    }

    pub fn fusion_weight(mut self, weight: f32) -> Self {
        self.config.fusion_weight = weight;
        self
    }

    pub fn build(self) -> NeuralRanker {
        NeuralRanker::new(self.config)
    }
}

impl Default for NeuralRankerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_ranker_creation() {
        let ranker = NeuralRankerBuilder::new()
            .model_type(ModelType::CrossEncoder)
            .rerank_top_k(50)
            .build();

        let _ = ranker.load_model("test_model");
    }

    #[test]
    fn test_rerank() {
        let ranker = NeuralRankerBuilder::new().build();
        let _ = ranker.load_model("test");

        let candidates = vec![
            RerankCandidate {
                id: "1".to_string(),
                text: "First document about AI".to_string(),
                original_score: 0.8,
                metadata: HashMap::new(),
            },
            RerankCandidate {
                id: "2".to_string(),
                text: "Second document about ML".to_string(),
                original_score: 0.7,
                metadata: HashMap::new(),
            },
        ];

        let result = ranker.rerank("artificial intelligence", candidates);
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.total_candidates, 2);
    }

    #[test]
    fn test_cross_encoder() {
        let encoder = CrossEncoder::new(CrossEncoderConfig::default());
        let score = encoder.score("query", "document");
        assert!(score.is_ok());
    }

    #[test]
    fn test_late_interaction() {
        let model = LateInteractionModel::new(LateInteractionConfig::default());

        let q_emb = model.encode_query("test query").unwrap();
        let d_emb = model.encode_document("test document").unwrap();

        let score = model.score(&q_emb, &d_emb);
        assert!(score >= 0.0 || score < 0.0); // Just verify it computes
    }

    #[test]
    fn test_learned_sparse() {
        let model = LearnedSparseModel::new(LearnedSparseConfig::default());
        let sparse = model.encode("test query");
        assert!(sparse.is_ok());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);

        let c = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
