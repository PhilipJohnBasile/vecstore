//! ColBERT Late Interaction Reranking
//!
//! This module implements ColBERT (Contextualized Late Interaction over BERT), a state-of-the-art
//! neural reranking approach that uses token-level interactions for high-accuracy retrieval.
//!
//! ## How ColBERT Works
//!
//! Unlike traditional reranking that uses a single vector per document:
//! 1. **Multi-vector representation**: Each document/query is encoded as multiple vectors (one per token)
//! 2. **Late interaction**: Similarity is computed at the token level, not averaged upfront
//! 3. **MaxSim operation**: For each query token, find max similarity with any document token
//! 4. **Final score**: Sum of all query token max similarities
//!
//! ## Feature Flags
//!
//! This module requires the `embeddings` feature for full ONNX model inference support.
//! Without this feature, a fallback implementation using simple heuristics is available.
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::reranking::colbert::{ColBERTReranker, ColBERTConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = ColBERTConfig {
//!     max_query_tokens: 32,
//!     max_doc_tokens: 128,
//!     ..Default::default()
//! };
//!
//! let reranker = ColBERTReranker::new(config)?;
//!
//! // Encode query and documents
//! let query_tokens = reranker.encode_query("what is rust?").await?;
//! let doc_tokens = reranker.encode_document("Rust is a systems programming language").await?;
//!
//! // Compute late interaction score
//! let score = reranker.compute_score(&query_tokens, &doc_tokens)?;
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
#[cfg(feature = "embeddings")]
use anyhow::Context;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(feature = "embeddings")]
use std::sync::Arc;

#[cfg(feature = "embeddings")]
use ndarray::{Array2, CowArray};
#[cfg(feature = "embeddings")]
use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder, Value};
#[cfg(feature = "embeddings")]
use std::sync::RwLock;
#[cfg(feature = "embeddings")]
use tokenizers::tokenizer::Tokenizer;

/// ColBERT reranker configuration
#[derive(Debug, Clone)]
pub struct ColBERTConfig {
    /// Maximum number of query tokens to encode
    pub max_query_tokens: usize,

    /// Maximum number of document tokens to encode
    pub max_doc_tokens: usize,

    /// Token embedding dimension (output dimension after projection)
    pub embedding_dim: usize,

    /// Similarity metric (typically cosine for ColBERT)
    pub similarity_metric: SimilarityMetric,

    /// Whether to normalize embeddings
    pub normalize: bool,

    /// Path to ColBERT ONNX model file
    pub model_path: Option<PathBuf>,

    /// Path to tokenizer.json file
    pub tokenizer_path: Option<PathBuf>,

    /// Number of inference threads
    pub num_threads: usize,

    /// Whether to use query augmentation (prepend [Q] token marker)
    pub use_query_augmentation: bool,

    /// Whether to use document augmentation (prepend [D] token marker)
    pub use_doc_augmentation: bool,

    /// Batch size for document encoding
    pub batch_size: usize,

    /// Enable model warmup on initialization
    pub warmup_on_init: bool,
}

impl Default for ColBERTConfig {
    fn default() -> Self {
        Self {
            max_query_tokens: 32,
            max_doc_tokens: 180,
            embedding_dim: 128, // ColBERT typically uses 128-dim projections
            similarity_metric: SimilarityMetric::Cosine,
            normalize: true,
            model_path: None,
            tokenizer_path: None,
            num_threads: 4,
            use_query_augmentation: true,
            use_doc_augmentation: true,
            batch_size: 32,
            warmup_on_init: true,
        }
    }
}

/// Available pretrained ColBERT models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColBERTModel {
    /// ColBERTv2 - Standard ColBERT model
    /// - Embedding dim: 128
    /// - Base model: BERT
    ColBERTv2,

    /// ColBERT-tiny - Smaller, faster variant
    /// - Embedding dim: 128
    /// - Base model: DistilBERT
    ColBERTTiny,

    /// Custom model (user-provided)
    Custom,
}

impl ColBERTModel {
    /// Get the HuggingFace model ID
    pub fn model_id(&self) -> &'static str {
        match self {
            ColBERTModel::ColBERTv2 => "colbert-ir/colbertv2.0",
            ColBERTModel::ColBERTTiny => "sentence-transformers/msmarco-distilbert-base-tas-b",
            ColBERTModel::Custom => "custom",
        }
    }

    /// Get the local model directory name
    pub fn model_dir(&self) -> &'static str {
        match self {
            ColBERTModel::ColBERTv2 => "colbertv2",
            ColBERTModel::ColBERTTiny => "colbert-tiny",
            ColBERTModel::Custom => "custom",
        }
    }

    /// Get the default model cache directory
    pub fn cache_dir() -> PathBuf {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        Path::new(&home)
            .join(".cache")
            .join("vecstore")
            .join("colbert-models")
    }

    /// Get the expected embedding dimension for this model
    pub fn embedding_dim(&self) -> usize {
        match self {
            ColBERTModel::ColBERTv2 => 128,
            ColBERTModel::ColBERTTiny => 128,
            ColBERTModel::Custom => 128,
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

    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Option<Vec<u8>>,
}

impl TokenEmbeddings {
    /// Create new token embeddings
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            embeddings,
            tokens: None,
            attention_mask: None,
        }
    }

    /// Create with token text for debugging
    pub fn with_tokens(embeddings: Vec<Vec<f32>>, tokens: Vec<String>) -> Self {
        Self {
            embeddings,
            tokens: Some(tokens),
            attention_mask: None,
        }
    }

    /// Create with full metadata
    pub fn with_metadata(
        embeddings: Vec<Vec<f32>>,
        tokens: Vec<String>,
        attention_mask: Vec<u8>,
    ) -> Self {
        Self {
            embeddings,
            tokens: Some(tokens),
            attention_mask: Some(attention_mask),
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

    /// Get number of non-padding tokens
    pub fn num_real_tokens(&self) -> usize {
        match &self.attention_mask {
            Some(mask) => mask.iter().filter(|&&m| m == 1).count(),
            None => self.embeddings.len(),
        }
    }
}

/// ColBERT reranker for late interaction scoring
///
/// When the `embeddings` feature is enabled, this uses actual ONNX model inference.
/// Otherwise, it falls back to a simple heuristic-based approach for testing.
pub struct ColBERTReranker {
    config: ColBERTConfig,

    /// Cache for document embeddings (document_id -> token embeddings)
    doc_cache: HashMap<String, TokenEmbeddings>,

    /// ONNX Runtime session (when embeddings feature is enabled)
    #[cfg(feature = "embeddings")]
    session: Option<Arc<Session>>,

    /// Tokenizer (when embeddings feature is enabled)
    #[cfg(feature = "embeddings")]
    tokenizer: Option<Arc<Tokenizer>>,

    /// ONNX Runtime environment
    #[cfg(feature = "embeddings")]
    environment: Option<Arc<Environment>>,

    /// Query embedding cache for faster repeated queries
    #[cfg(feature = "embeddings")]
    query_cache: Arc<RwLock<HashMap<String, TokenEmbeddings>>>,
}

impl std::fmt::Debug for ColBERTReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColBERTReranker")
            .field("config", &self.config)
            .field("doc_cache_size", &self.doc_cache.len())
            .finish()
    }
}

impl ColBERTReranker {
    /// Create a new ColBERT reranker with default configuration
    ///
    /// This creates a reranker without model loading. Use `from_pretrained` or
    /// `from_dir` for full model-based inference.
    pub fn new(config: ColBERTConfig) -> Result<Self> {
        Ok(Self {
            config,
            doc_cache: HashMap::new(),
            #[cfg(feature = "embeddings")]
            session: None,
            #[cfg(feature = "embeddings")]
            tokenizer: None,
            #[cfg(feature = "embeddings")]
            environment: None,
            #[cfg(feature = "embeddings")]
            query_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Load a pretrained ColBERT model
    ///
    /// This will look for the model in the cache directory and download if necessary.
    ///
    /// # Arguments
    /// * `model` - The pretrained model to load
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::reranking::colbert::{ColBERTReranker, ColBERTModel, ColBERTConfig};
    ///
    /// let reranker = ColBERTReranker::from_pretrained(
    ///     ColBERTModel::ColBERTv2,
    ///     ColBERTConfig::default()
    /// )?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[cfg(feature = "embeddings")]
    pub fn from_pretrained(model: ColBERTModel, config: ColBERTConfig) -> Result<Self> {
        let cache_dir = ColBERTModel::cache_dir();
        let model_dir = cache_dir.join(model.model_dir());

        if !model_dir.exists() {
            return Err(anyhow!(
                "Model not found. Please download the ColBERT model from:\n\
                 https://huggingface.co/{}\n\
                 \n\
                 Required files:\n\
                 - model.onnx (ColBERT ONNX model with token embeddings output)\n\
                 - tokenizer.json (HuggingFace tokenizer)\n\
                 \n\
                 Place them in: {:?}\n\
                 \n\
                 To convert a ColBERT model to ONNX:\n\
                 1. pip install optimum[onnxruntime]\n\
                 2. optimum-cli export onnx --model {} --task feature-extraction {}",
                model.model_id(),
                model_dir,
                model.model_id(),
                model_dir.display()
            ));
        }

        Self::from_dir(&model_dir, config)
    }

    /// Load a ColBERT model from a local directory
    ///
    /// The directory should contain:
    /// - `model.onnx` - ONNX model file
    /// - `tokenizer.json` - HuggingFace tokenizer
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::reranking::colbert::{ColBERTReranker, ColBERTConfig};
    ///
    /// let reranker = ColBERTReranker::from_dir(
    ///     "./my_colbert_model",
    ///     ColBERTConfig::default()
    /// )?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[cfg(feature = "embeddings")]
    pub fn from_dir<P: AsRef<Path>>(model_dir: P, mut config: ColBERTConfig) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Determine model and tokenizer paths
        let model_path = config
            .model_path
            .clone()
            .unwrap_or_else(|| model_dir.join("model.onnx"));
        let tokenizer_path = config
            .tokenizer_path
            .clone()
            .unwrap_or_else(|| model_dir.join("tokenizer.json"));

        if !model_path.exists() {
            return Err(anyhow!("Model file not found: {:?}", model_path));
        }

        if !tokenizer_path.exists() {
            return Err(anyhow!("Tokenizer file not found: {:?}", tokenizer_path));
        }

        // Update config with resolved paths
        config.model_path = Some(model_path.clone());
        config.tokenizer_path = Some(tokenizer_path.clone());

        // Initialize ONNX Runtime environment
        let environment = Arc::new(
            Environment::builder()
                .with_name("colbert")
                .build()
                .context("Failed to create ONNX environment")?,
        );

        // Load ONNX model with optimizations
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.num_threads as i16)?
            .with_model_from_file(&model_path)
            .context("Failed to load ColBERT ONNX model")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let mut reranker = Self {
            config,
            doc_cache: HashMap::new(),
            session: Some(Arc::new(session)),
            tokenizer: Some(Arc::new(tokenizer)),
            environment: Some(environment),
            query_cache: Arc::new(RwLock::new(HashMap::new())),
        };

        // Warmup the model if configured
        if reranker.config.warmup_on_init {
            reranker.warmup()?;
        }

        Ok(reranker)
    }

    /// Warmup the model with dummy inputs
    ///
    /// This helps optimize the model execution graph and allocate memory upfront.
    #[cfg(feature = "embeddings")]
    pub fn warmup(&self) -> Result<()> {
        if self.session.is_none() {
            return Ok(());
        }

        tracing::info!("Warming up ColBERT model...");

        // Run a few inference passes with dummy data
        let warmup_texts = vec![
            "warmup query text",
            "another warmup text for optimization",
            "third warmup pass",
        ];

        for text in warmup_texts {
            let _ = self.encode_text_sync(text, self.config.max_query_tokens);
        }

        tracing::info!("ColBERT model warmup complete");
        Ok(())
    }

    /// Encode text into token-level embeddings (synchronous internal method)
    #[cfg(feature = "embeddings")]
    fn encode_text_sync(&self, text: &str, max_tokens: usize) -> Result<TokenEmbeddings> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| anyhow!("Model not loaded. Use from_pretrained or from_dir."))?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("Tokenizer not loaded."))?;

        // Tokenize the input
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        // Get input tensors
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Truncate or pad to max_tokens
        let seq_length = input_ids.len().min(max_tokens);

        // Prepare input arrays
        let mut padded_input_ids = vec![0i64; max_tokens];
        let mut padded_attention_mask = vec![0i64; max_tokens];

        for i in 0..seq_length {
            padded_input_ids[i] = input_ids[i] as i64;
            padded_attention_mask[i] = attention_mask[i] as i64;
        }

        // Create 2D arrays (batch_size=1, seq_length)
        let input_ids_array =
            Array2::from_shape_vec((1, max_tokens), padded_input_ids.clone())?;
        let attention_mask_array =
            Array2::from_shape_vec((1, max_tokens), padded_attention_mask.clone())?;

        // Convert to dynamic arrays for ONNX Runtime
        let input_ids_dyn = input_ids_array.into_dyn();
        let attention_mask_dyn = attention_mask_array.into_dyn();

        let input_ids_cow = CowArray::from(&input_ids_dyn);
        let attention_mask_cow = CowArray::from(&attention_mask_dyn);

        // Create ONNX input values
        let input_ids_value = Value::from_array(session.allocator(), &input_ids_cow)?;
        let attention_mask_value = Value::from_array(session.allocator(), &attention_mask_cow)?;

        // Run inference - ColBERT models typically take input_ids and attention_mask
        let outputs = session.run(vec![input_ids_value, attention_mask_value])?;

        // Extract token embeddings
        // ColBERT output shape: (batch_size, seq_length, embedding_dim)
        let token_embeddings = outputs[0]
            .try_extract::<f32>()
            .context("Failed to extract token embeddings")?
            .view()
            .to_owned();

        let shape = token_embeddings.shape();
        let _batch_size = shape[0];
        let output_seq_len = shape[1];
        let embedding_dim = shape[2];

        // Convert to Vec<Vec<f32>> format
        let mut embeddings = Vec::with_capacity(output_seq_len);
        let mut real_attention_mask = Vec::with_capacity(output_seq_len);

        for seq_idx in 0..output_seq_len {
            let mask_val = if seq_idx < padded_attention_mask.len() {
                padded_attention_mask[seq_idx] as u8
            } else {
                0
            };
            real_attention_mask.push(mask_val);

            // Only include embeddings for non-padding tokens if configured
            let mut token_emb = Vec::with_capacity(embedding_dim);
            for emb_idx in 0..embedding_dim {
                token_emb.push(token_embeddings[[0, seq_idx, emb_idx]]);
            }

            // Normalize if configured
            if self.config.normalize {
                let norm: f32 = token_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for val in &mut token_emb {
                        *val /= norm;
                    }
                }
            }

            embeddings.push(token_emb);
        }

        // Get token strings for debugging
        let token_strings: Vec<String> = encoding
            .get_tokens()
            .iter()
            .take(output_seq_len)
            .map(|s| s.to_string())
            .collect();

        Ok(TokenEmbeddings::with_metadata(
            embeddings,
            token_strings,
            real_attention_mask,
        ))
    }

    /// Encode a query into token-level embeddings
    ///
    /// For ColBERT, query encoding may use special query augmentation.
    pub async fn encode_query(&self, query: &str) -> Result<TokenEmbeddings> {
        #[cfg(feature = "embeddings")]
        {
            // Check query cache first
            {
                let cache = self.query_cache.read().map_err(|e| anyhow!("Cache lock error: {}", e))?;
                if let Some(cached) = cache.get(query) {
                    return Ok(cached.clone());
                }
            }

            // Prepare query text with optional augmentation
            let query_text = if self.config.use_query_augmentation {
                format!("[Q] {}", query)
            } else {
                query.to_string()
            };

            let embeddings = self.encode_text_sync(&query_text, self.config.max_query_tokens)?;

            // Cache the result
            {
                let mut cache = self.query_cache.write().map_err(|e| anyhow!("Cache lock error: {}", e))?;
                cache.insert(query.to_string(), embeddings.clone());
            }

            Ok(embeddings)
        }

        #[cfg(not(feature = "embeddings"))]
        {
            // Fallback: simple hash-based embeddings for testing
            self.encode_fallback(query, self.config.max_query_tokens)
        }
    }

    /// Encode a document into token-level embeddings
    pub async fn encode_document(&self, document: &str) -> Result<TokenEmbeddings> {
        #[cfg(feature = "embeddings")]
        {
            // Prepare document text with optional augmentation
            let doc_text = if self.config.use_doc_augmentation {
                format!("[D] {}", document)
            } else {
                document.to_string()
            };

            self.encode_text_sync(&doc_text, self.config.max_doc_tokens)
        }

        #[cfg(not(feature = "embeddings"))]
        {
            self.encode_fallback(document, self.config.max_doc_tokens)
        }
    }

    /// Encode multiple documents in a batch (more efficient than encoding one by one)
    #[cfg(feature = "embeddings")]
    pub async fn encode_documents_batch(
        &self,
        documents: &[&str],
    ) -> Result<Vec<TokenEmbeddings>> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| anyhow!("Model not loaded. Use from_pretrained or from_dir."))?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("Tokenizer not loaded."))?;

        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = documents.len().min(self.config.batch_size);
        let max_tokens = self.config.max_doc_tokens;

        // Process in batches
        let mut all_embeddings = Vec::with_capacity(documents.len());

        for batch_start in (0..documents.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(documents.len());
            let batch_docs = &documents[batch_start..batch_end];
            let current_batch_size = batch_docs.len();

            // Prepare texts with augmentation
            let batch_texts: Vec<String> = batch_docs
                .iter()
                .map(|doc| {
                    if self.config.use_doc_augmentation {
                        format!("[D] {}", doc)
                    } else {
                        (*doc).to_string()
                    }
                })
                .collect();

            // Tokenize batch
            let encodings = tokenizer
                .encode_batch(batch_texts.iter().map(|s| s.as_str()).collect::<Vec<_>>(), true)
                .map_err(|e| anyhow!("Batch tokenization failed: {}", e))?;

            // Prepare batch input tensors
            let mut all_input_ids = Vec::with_capacity(current_batch_size * max_tokens);
            let mut all_attention_masks = Vec::with_capacity(current_batch_size * max_tokens);

            for encoding in &encodings {
                let input_ids = encoding.get_ids();
                let attention_mask = encoding.get_attention_mask();
                let seq_length = input_ids.len().min(max_tokens);

                // Pad or truncate
                for i in 0..max_tokens {
                    if i < seq_length {
                        all_input_ids.push(input_ids[i] as i64);
                        all_attention_masks.push(attention_mask[i] as i64);
                    } else {
                        all_input_ids.push(0);
                        all_attention_masks.push(0);
                    }
                }
            }

            // Create batch arrays
            let input_ids_array =
                Array2::from_shape_vec((current_batch_size, max_tokens), all_input_ids)?;
            let attention_mask_array =
                Array2::from_shape_vec((current_batch_size, max_tokens), all_attention_masks.clone())?;

            let input_ids_dyn = input_ids_array.into_dyn();
            let attention_mask_dyn = attention_mask_array.into_dyn();

            let input_ids_cow = CowArray::from(&input_ids_dyn);
            let attention_mask_cow = CowArray::from(&attention_mask_dyn);

            let input_ids_value = Value::from_array(session.allocator(), &input_ids_cow)?;
            let attention_mask_value = Value::from_array(session.allocator(), &attention_mask_cow)?;

            // Run batch inference
            let outputs = session.run(vec![input_ids_value, attention_mask_value])?;

            // Extract and process embeddings
            let token_embeddings = outputs[0]
                .try_extract::<f32>()
                .context("Failed to extract batch token embeddings")?
                .view()
                .to_owned();

            let shape = token_embeddings.shape();
            let output_seq_len = shape[1];
            let embedding_dim = shape[2];

            // Process each item in the batch
            for batch_idx in 0..current_batch_size {
                let mut embeddings = Vec::with_capacity(output_seq_len);
                let mut attention_mask_vec = Vec::with_capacity(output_seq_len);

                for seq_idx in 0..output_seq_len {
                    let mask_val = all_attention_masks[batch_idx * max_tokens + seq_idx] as u8;
                    attention_mask_vec.push(mask_val);

                    let mut token_emb = Vec::with_capacity(embedding_dim);
                    for emb_idx in 0..embedding_dim {
                        token_emb.push(token_embeddings[[batch_idx, seq_idx, emb_idx]]);
                    }

                    // Normalize if configured
                    if self.config.normalize {
                        let norm: f32 = token_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 0.0 {
                            for val in &mut token_emb {
                                *val /= norm;
                            }
                        }
                    }

                    embeddings.push(token_emb);
                }

                let token_strings: Vec<String> = encodings[batch_idx]
                    .get_tokens()
                    .iter()
                    .take(output_seq_len)
                    .map(|s| s.to_string())
                    .collect();

                all_embeddings.push(TokenEmbeddings::with_metadata(
                    embeddings,
                    token_strings,
                    attention_mask_vec,
                ));
            }
        }

        Ok(all_embeddings)
    }

    /// Fallback encoding for when the embeddings feature is not enabled
    #[cfg(not(feature = "embeddings"))]
    fn encode_fallback(&self, text: &str, max_tokens: usize) -> Result<TokenEmbeddings> {
        // Simple word-based tokenization for fallback
        let tokens: Vec<String> = text
            .split_whitespace()
            .take(max_tokens)
            .map(|s| s.to_string())
            .collect();

        // Generate deterministic embeddings based on token hashes
        let embeddings: Vec<Vec<f32>> = tokens
            .iter()
            .map(|token| {
                let mut emb = vec![0.0f32; self.config.embedding_dim];

                // Use a simple hash-based approach for reproducibility
                let hash = Self::simple_hash(token);
                for (i, val) in emb.iter_mut().enumerate() {
                    let seed = hash.wrapping_add(i as u64);
                    *val = ((seed as f32 / u64::MAX as f32) * 2.0) - 1.0;
                }

                if self.config.normalize {
                    Self::normalize_vector(emb)
                } else {
                    emb
                }
            })
            .collect();

        // Handle empty input
        let embeddings = if embeddings.is_empty() {
            vec![vec![0.0f32; self.config.embedding_dim]]
        } else {
            embeddings
        };

        let tokens = if tokens.is_empty() {
            vec!["[PAD]".to_string()]
        } else {
            tokens
        };

        Ok(TokenEmbeddings::with_tokens(embeddings, tokens))
    }

    /// Simple deterministic hash function
    #[cfg(not(feature = "embeddings"))]
    fn simple_hash(s: &str) -> u64 {
        let mut hash: u64 = 5381;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }

    /// Compute ColBERT late interaction score (MaxSim)
    ///
    /// For each query token, find the maximum similarity with any document token,
    /// then sum these maximum similarities.
    ///
    /// Score = Sum_i max_j sim(q_i, d_j)
    /// where q_i is the i-th query token, d_j is the j-th document token
    pub fn compute_score(
        &self,
        query_tokens: &TokenEmbeddings,
        doc_tokens: &TokenEmbeddings,
    ) -> Result<f32> {
        if query_tokens.embedding_dim() != doc_tokens.embedding_dim() {
            return Err(anyhow!(
                "Dimension mismatch: query={}, doc={}",
                query_tokens.embedding_dim(),
                doc_tokens.embedding_dim()
            ));
        }

        if query_tokens.embeddings.is_empty() || doc_tokens.embeddings.is_empty() {
            return Ok(0.0);
        }

        let mut total_score = 0.0;

        // For each query token (considering attention mask)
        for (q_idx, query_emb) in query_tokens.embeddings.iter().enumerate() {
            // Skip padding tokens if attention mask is available
            if let Some(ref mask) = query_tokens.attention_mask {
                if q_idx < mask.len() && mask[q_idx] == 0 {
                    continue;
                }
            }

            let mut max_sim = f32::NEG_INFINITY;

            // Find max similarity with any document token
            for (d_idx, doc_emb) in doc_tokens.embeddings.iter().enumerate() {
                // Skip padding tokens if attention mask is available
                if let Some(ref mask) = doc_tokens.attention_mask {
                    if d_idx < mask.len() && mask[d_idx] == 0 {
                        continue;
                    }
                }

                let sim = self.compute_token_similarity(query_emb, doc_emb);
                max_sim = max_sim.max(sim);
            }

            if max_sim.is_finite() {
                total_score += max_sim;
            }
        }

        Ok(total_score)
    }

    /// Compute batched MaxSim scores for multiple query-document pairs
    ///
    /// This is more efficient when scoring the same query against many documents.
    pub fn compute_scores_batch(
        &self,
        query_tokens: &TokenEmbeddings,
        doc_tokens_batch: &[TokenEmbeddings],
    ) -> Result<Vec<f32>> {
        doc_tokens_batch
            .iter()
            .map(|doc_tokens| self.compute_score(query_tokens, doc_tokens))
            .collect()
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
        if vec1.len() != vec2.len() {
            return 0.0;
        }
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

    /// Clear the query cache
    #[cfg(feature = "embeddings")]
    pub fn clear_query_cache(&self) -> Result<()> {
        let mut cache = self.query_cache.write().map_err(|e| anyhow!("Cache lock error: {}", e))?;
        cache.clear();
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        #[cfg(feature = "embeddings")]
        let query_cache_size = self
            .query_cache
            .read()
            .map(|c| c.len())
            .unwrap_or(0);

        #[cfg(not(feature = "embeddings"))]
        let query_cache_size = 0;

        CacheStats {
            doc_cache_size: self.doc_cache.len(),
            query_cache_size,
        }
    }

    /// Check if model is loaded and ready for inference
    pub fn is_model_loaded(&self) -> bool {
        #[cfg(feature = "embeddings")]
        {
            self.session.is_some() && self.tokenizer.is_some()
        }

        #[cfg(not(feature = "embeddings"))]
        {
            false
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &ColBERTConfig {
        &self.config
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached document embeddings
    pub doc_cache_size: usize,
    /// Number of cached query embeddings
    pub query_cache_size: usize,
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

    /// Create a batch reranker from a pretrained model
    #[cfg(feature = "embeddings")]
    pub fn from_pretrained(model: ColBERTModel, config: ColBERTConfig) -> Result<Self> {
        Ok(Self {
            reranker: ColBERTReranker::from_pretrained(model, config)?,
        })
    }

    /// Create a batch reranker from a model directory
    #[cfg(feature = "embeddings")]
    pub fn from_dir<P: AsRef<Path>>(model_dir: P, config: ColBERTConfig) -> Result<Self> {
        Ok(Self {
            reranker: ColBERTReranker::from_dir(model_dir, config)?,
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
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Encode query once
        let query_tokens = self.reranker.encode_query(query).await?;

        // Encode all documents
        #[cfg(feature = "embeddings")]
        let doc_embeddings = if self.reranker.is_model_loaded() {
            let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
            self.reranker.encode_documents_batch(&doc_refs).await?
        } else {
            let mut embeddings = Vec::with_capacity(documents.len());
            for doc in documents {
                embeddings.push(self.reranker.encode_document(doc).await?);
            }
            embeddings
        };

        #[cfg(not(feature = "embeddings"))]
        let doc_embeddings = {
            let mut embeddings = Vec::with_capacity(documents.len());
            for doc in documents {
                embeddings.push(self.reranker.encode_document(doc).await?);
            }
            embeddings
        };

        // Compute scores for all documents
        let scores = self.reranker.compute_scores_batch(&query_tokens, &doc_embeddings)?;

        // Create indexed scores and sort
        let mut indexed_scores: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        indexed_scores.truncate(top_k);
        Ok(indexed_scores)
    }

    /// Rerank with caching - documents that have been seen before use cached embeddings
    pub async fn rerank_with_cache(
        &mut self,
        query: &str,
        documents: &[(String, String)], // (doc_id, doc_text)
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Encode query
        let query_tokens = self.reranker.encode_query(query).await?;

        // Encode documents, using cache when available
        let mut doc_embeddings = Vec::with_capacity(documents.len());
        let mut doc_ids = Vec::with_capacity(documents.len());

        for (doc_id, doc_text) in documents {
            doc_ids.push(doc_id.clone());

            if let Some(cached) = self.reranker.get_cached_document(doc_id) {
                doc_embeddings.push(cached.clone());
            } else {
                let embeddings = self.reranker.encode_document(doc_text).await?;
                self.reranker.cache_document(doc_id.clone(), embeddings.clone());
                doc_embeddings.push(embeddings);
            }
        }

        // Compute scores
        let scores = self.reranker.compute_scores_batch(&query_tokens, &doc_embeddings)?;

        // Create indexed scores and sort
        let mut indexed_scores: Vec<(String, f32)> = doc_ids
            .into_iter()
            .zip(scores.into_iter())
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        indexed_scores.truncate(top_k);
        Ok(indexed_scores)
    }

    /// Get a reference to the underlying reranker
    pub fn reranker(&self) -> &ColBERTReranker {
        &self.reranker
    }

    /// Get a mutable reference to the underlying reranker
    pub fn reranker_mut(&mut self) -> &mut ColBERTReranker {
        &mut self.reranker
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
    fn test_token_embeddings_with_metadata() {
        let embeddings = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let tokens = vec!["hello".to_string(), "world".to_string(), "[PAD]".to_string()];
        let mask = vec![1, 1, 0];

        let token_embs = TokenEmbeddings::with_metadata(embeddings, tokens, mask);
        assert_eq!(token_embs.num_tokens(), 3);
        assert_eq!(token_embs.num_real_tokens(), 2);
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

    #[test]
    fn test_compute_score_with_attention_mask() {
        let config = ColBERTConfig {
            embedding_dim: 3,
            normalize: false,
            ..Default::default()
        };

        let reranker = ColBERTReranker::new(config).unwrap();

        // Query with padding
        let query_tokens = TokenEmbeddings::with_metadata(
            vec![
                vec![1.0, 0.0, 0.0], // Real token
                vec![0.0, 1.0, 0.0], // Padding (should be ignored)
            ],
            vec!["hello".to_string(), "[PAD]".to_string()],
            vec![1, 0], // Second token is padding
        );

        let doc_tokens = TokenEmbeddings::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ]);

        let score = reranker.compute_score(&query_tokens, &doc_tokens).unwrap();

        // Only the first query token should be considered
        // Max sim = 1.0 (with doc token 1)
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_compute_scores_batch() {
        let config = ColBERTConfig {
            embedding_dim: 3,
            normalize: false,
            ..Default::default()
        };

        let reranker = ColBERTReranker::new(config).unwrap();

        let query_tokens = TokenEmbeddings::new(vec![vec![1.0, 0.0, 0.0]]);

        let doc_batch = vec![
            TokenEmbeddings::new(vec![vec![1.0, 0.0, 0.0]]), // Perfect match
            TokenEmbeddings::new(vec![vec![0.0, 1.0, 0.0]]), // No match
            TokenEmbeddings::new(vec![vec![0.5, 0.5, 0.0]]), // Partial match
        ];

        let scores = reranker.compute_scores_batch(&query_tokens, &doc_batch).unwrap();

        assert_eq!(scores.len(), 3);
        assert!(scores[0] > scores[2]); // Perfect match > partial
        assert!(scores[2] > scores[1]); // Partial > no match
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

    #[tokio::test]
    async fn test_batch_reranker() {
        let config = ColBERTConfig::default();
        let mut reranker = ColBERTBatchReranker::new(config).unwrap();

        let documents = vec![
            "First document about programming".to_string(),
            "Second document about cooking".to_string(),
            "Third document about programming languages".to_string(),
        ];

        let results = reranker.rerank("programming", &documents, 2).await.unwrap();

        assert_eq!(results.len(), 2);
        // Results should be sorted by score (descending)
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn test_cache_operations() {
        let config = ColBERTConfig::default();
        let mut reranker = ColBERTReranker::new(config).unwrap();

        let embeddings = TokenEmbeddings::new(vec![vec![1.0, 2.0, 3.0]]);
        reranker.cache_document("doc1".to_string(), embeddings.clone());

        assert!(reranker.get_cached_document("doc1").is_some());
        assert!(reranker.get_cached_document("doc2").is_none());

        let stats = reranker.cache_stats();
        assert_eq!(stats.doc_cache_size, 1);

        reranker.clear_cache();
        assert!(reranker.get_cached_document("doc1").is_none());
    }

    #[test]
    fn test_colbert_config_default() {
        let config = ColBERTConfig::default();
        assert_eq!(config.max_query_tokens, 32);
        assert_eq!(config.max_doc_tokens, 180);
        assert_eq!(config.embedding_dim, 128);
        assert!(config.normalize);
        assert!(config.warmup_on_init);
    }

    #[test]
    fn test_colbert_model_metadata() {
        assert_eq!(ColBERTModel::ColBERTv2.model_id(), "colbert-ir/colbertv2.0");
        assert_eq!(ColBERTModel::ColBERTv2.model_dir(), "colbertv2");
        assert_eq!(ColBERTModel::ColBERTv2.embedding_dim(), 128);

        let cache_dir = ColBERTModel::cache_dir();
        assert!(cache_dir.to_string_lossy().contains("colbert-models"));
    }

    #[test]
    fn test_similarity_metrics() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.707, 0.707, 0.0];

        // Dot product
        let dot = ColBERTReranker::dot_product(&vec1, &vec2);
        assert!((dot - 0.707).abs() < 0.001);

        // Cosine (same as dot for normalized vectors)
        let cosine = ColBERTReranker::cosine_similarity(&vec1, &vec2);
        assert!((cosine - 0.707).abs() < 0.001);

        // L2 distance
        let l2 = ColBERTReranker::l2_distance(&vec1, &vec2);
        assert!(l2 > 0.0);
    }

    #[test]
    fn test_is_model_loaded() {
        let config = ColBERTConfig::default();
        let reranker = ColBERTReranker::new(config).unwrap();

        // Without calling from_pretrained or from_dir, model should not be loaded
        assert!(!reranker.is_model_loaded());
    }

    // Integration tests that require actual model files and the embeddings feature
    #[cfg(feature = "embeddings")]
    #[tokio::test]
    #[ignore]
    async fn test_colbert_with_real_model() {
        let config = ColBERTConfig::default();
        let reranker = ColBERTReranker::from_pretrained(ColBERTModel::ColBERTv2, config)
            .expect("Failed to load model");

        let query_tokens = reranker.encode_query("what is rust").await.unwrap();
        let doc1_tokens = reranker
            .encode_document("Rust is a systems programming language")
            .await
            .unwrap();
        let doc2_tokens = reranker
            .encode_document("Python is great for data science")
            .await
            .unwrap();

        let score1 = reranker.compute_score(&query_tokens, &doc1_tokens).unwrap();
        let score2 = reranker.compute_score(&query_tokens, &doc2_tokens).unwrap();

        // Rust-related document should score higher
        assert!(score1 > score2);
    }
}
