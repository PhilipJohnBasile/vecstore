//! Embedded ONNX Models
//!
//! Built-in embedding generation without external API calls.
//! Similar to Chroma's embedding functions with local ONNX runtime.
//!
//! # Features
//!
//! - **Local Inference**: No external API dependencies
//! - **Multiple Models**: Support for various embedding models
//! - **Auto-Embed on Upsert**: Automatic embedding generation
//! - **Batch Processing**: Efficient batch embedding
//! - **Model Caching**: Cached model loading
//! - **Quantized Models**: Support for quantized ONNX models
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::embedded_onnx::{EmbeddingModel, EmbeddedStore};
//!
//! // Load a model
//! let model = EmbeddingModel::load("all-MiniLM-L6-v2")?;
//!
//! // Create store with auto-embedding
//! let mut store = EmbeddedStore::new(model)?;
//!
//! // Insert text - automatically embedded
//! store.insert_text("doc1", "Hello world", metadata)?;
//!
//! // Search with text query - automatically embedded
//! let results = store.search_text("greeting", 10)?;
//! ```

use std::collections::HashMap;
use std::sync::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{VecStoreError, Result};

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

/// Supported embedding models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// all-MiniLM-L6-v2 (384 dimensions)
    AllMiniLmL6V2,
    /// all-mpnet-base-v2 (768 dimensions)
    AllMpnetBaseV2,
    /// BGE-small-en-v1.5 (384 dimensions)
    BgeSmallEnV15,
    /// BGE-base-en-v1.5 (768 dimensions)
    BgeBaseEnV15,
    /// E5-small-v2 (384 dimensions)
    E5SmallV2,
    /// E5-base-v2 (768 dimensions)
    E5BaseV2,
    /// Custom ONNX model
    Custom { path: String, dimension: usize },
}

impl ModelType {
    /// Get the output dimension of this model
    pub fn dimension(&self) -> usize {
        match self {
            ModelType::AllMiniLmL6V2 => 384,
            ModelType::AllMpnetBaseV2 => 768,
            ModelType::BgeSmallEnV15 => 384,
            ModelType::BgeBaseEnV15 => 768,
            ModelType::E5SmallV2 => 384,
            ModelType::E5BaseV2 => 768,
            ModelType::Custom { dimension, .. } => *dimension,
        }
    }

    /// Get the model name for downloads
    pub fn model_name(&self) -> &str {
        match self {
            ModelType::AllMiniLmL6V2 => "all-MiniLM-L6-v2",
            ModelType::AllMpnetBaseV2 => "all-mpnet-base-v2",
            ModelType::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            ModelType::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            ModelType::E5SmallV2 => "intfloat/e5-small-v2",
            ModelType::E5BaseV2 => "intfloat/e5-base-v2",
            ModelType::Custom { path, .. } => path,
        }
    }

    /// Get HuggingFace model identifier for downloading
    pub fn huggingface_id(&self) -> &str {
        match self {
            ModelType::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            ModelType::AllMpnetBaseV2 => "sentence-transformers/all-mpnet-base-v2",
            ModelType::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            ModelType::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            ModelType::E5SmallV2 => "intfloat/e5-small-v2",
            ModelType::E5BaseV2 => "intfloat/e5-base-v2",
            ModelType::Custom { path, .. } => path,
        }
    }

    /// Check if this model type requires a query prefix (for BGE/E5 models)
    pub fn requires_query_prefix(&self) -> bool {
        matches!(
            self,
            ModelType::BgeSmallEnV15
                | ModelType::BgeBaseEnV15
                | ModelType::E5SmallV2
                | ModelType::E5BaseV2
        )
    }

    /// Get the default query prefix for this model
    pub fn default_query_prefix(&self) -> Option<&str> {
        match self {
            ModelType::BgeSmallEnV15 | ModelType::BgeBaseEnV15 => Some("Represent this sentence for searching relevant passages: "),
            ModelType::E5SmallV2 | ModelType::E5BaseV2 => Some("query: "),
            _ => None,
        }
    }

    /// Get the default document prefix for this model
    pub fn default_document_prefix(&self) -> Option<&str> {
        match self {
            ModelType::E5SmallV2 | ModelType::E5BaseV2 => Some("passage: "),
            _ => None,
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type
    pub model_type: ModelType,
    /// Maximum sequence length
    pub max_length: usize,
    /// Enable quantization
    pub quantize: bool,
    /// Use GPU if available
    pub use_gpu: bool,
    /// Number of threads
    pub num_threads: usize,
    /// Normalize embeddings
    pub normalize: bool,
    /// Prefix for queries
    pub query_prefix: Option<String>,
    /// Prefix for documents
    pub document_prefix: Option<String>,
    /// Batch size for inference
    pub batch_size: usize,
    /// Enable model warmup on load
    pub warmup: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::AllMiniLmL6V2,
            max_length: 512,
            quantize: false,
            use_gpu: false,
            num_threads: 4,
            normalize: true,
            query_prefix: None,
            document_prefix: None,
            batch_size: 32,
            warmup: true,
        }
    }
}

impl ModelConfig {
    /// Create config for a specific model type with default settings
    pub fn for_model(model_type: ModelType) -> Self {
        let query_prefix = model_type.default_query_prefix().map(String::from);
        let document_prefix = model_type.default_document_prefix().map(String::from);

        Self {
            model_type,
            query_prefix,
            document_prefix,
            ..Default::default()
        }
    }

    /// Builder: set max sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Builder: enable/disable quantization
    pub fn with_quantize(mut self, quantize: bool) -> Self {
        self.quantize = quantize;
        self
    }

    /// Builder: enable/disable GPU
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Builder: set number of threads
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Builder: enable/disable normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Builder: set query prefix
    pub fn with_query_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.query_prefix = Some(prefix.into());
        self
    }

    /// Builder: set document prefix
    pub fn with_document_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.document_prefix = Some(prefix.into());
        self
    }

    /// Builder: set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Builder: enable/disable warmup
    pub fn with_warmup(mut self, warmup: bool) -> Self {
        self.warmup = warmup;
        self
    }
}

// ============================================================================
// MODEL STATISTICS
// ============================================================================

/// Model statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelStats {
    pub total_embeddings: u64,
    pub total_tokens: u64,
    pub avg_latency_ms: f64,
    pub cache_hits: u64,
    pub warmup_completed: bool,
}

// ============================================================================
// EMBEDDING MODEL - Feature-gated implementations
// ============================================================================

/// Embedding model wrapper
#[cfg(feature = "embeddings")]
pub struct EmbeddingModel {
    /// Model configuration
    config: ModelConfig,
    /// Model dimension
    dimension: usize,
    /// ONNX Runtime session (uses RwLock for interior mutability as run() requires &mut)
    session: std::sync::Arc<std::sync::RwLock<ort::Session>>,
    /// Tokenizer
    tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
    /// Statistics
    stats: RwLock<ModelStats>,
}

/// Embedding model wrapper (fallback without embeddings feature)
#[cfg(not(feature = "embeddings"))]
pub struct EmbeddingModel {
    /// Model configuration
    config: ModelConfig,
    /// Model dimension
    dimension: usize,
    /// Simple vocabulary for tokenization (placeholder)
    vocab: HashMap<String, usize>,
    /// Statistics
    stats: RwLock<ModelStats>,
}

#[cfg(feature = "embeddings")]
impl EmbeddingModel {
    /// Load a model from the default cache directory
    pub fn load(model_type: ModelType) -> Result<Self> {
        let config = ModelConfig::for_model(model_type);
        Self::with_config(config)
    }

    /// Load with configuration
    pub fn with_config(config: ModelConfig) -> Result<Self> {
        use std::path::PathBuf;

        let dimension = config.model_type.dimension();

        // Get model directory
        let model_dir = Self::get_model_dir(&config.model_type)?;

        // Check if model files exist, download if not
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() || !tokenizer_path.exists() {
            Self::download_model(&config.model_type, &model_dir)?;
        }

        // Initialize ONNX Runtime session
        let session = Self::create_session(&model_path, &config)?;

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| VecStoreError::Tokenization(format!("Failed to load tokenizer: {}", e)))?;

        let model = Self {
            config: config.clone(),
            dimension,
            session: std::sync::Arc::new(std::sync::RwLock::new(session)),
            tokenizer: std::sync::Arc::new(tokenizer),
            stats: RwLock::new(ModelStats::default()),
        };

        // Warmup if configured
        if config.warmup {
            model.warmup()?;
        }

        Ok(model)
    }

    /// Load model from a specific directory
    pub fn from_directory(
        model_dir: impl AsRef<std::path::Path>,
        config: ModelConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() {
            return Err(VecStoreError::NotFound(format!(
                "Model file not found: {:?}",
                model_path
            )));
        }

        if !tokenizer_path.exists() {
            return Err(VecStoreError::NotFound(format!(
                "Tokenizer file not found: {:?}",
                tokenizer_path
            )));
        }

        let dimension = config.model_type.dimension();
        let session = Self::create_session(&model_path, &config)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| VecStoreError::Tokenization(format!("Failed to load tokenizer: {}", e)))?;

        let model = Self {
            config: config.clone(),
            dimension,
            session: std::sync::Arc::new(std::sync::RwLock::new(session)),
            tokenizer: std::sync::Arc::new(tokenizer),
            stats: RwLock::new(ModelStats::default()),
        };

        if config.warmup {
            model.warmup()?;
        }

        Ok(model)
    }

    /// Create ONNX session with configuration
    fn create_session(
        model_path: &std::path::Path,
        config: &ModelConfig,
    ) -> Result<ort::Session> {
        use ort::GraphOptimizationLevel;

        let mut session_builder = ort::Session::builder()?;

        // Set optimization level based on quantization
        if config.quantize {
            session_builder = session_builder
                .with_optimization_level(GraphOptimizationLevel::Level3)?;
        } else {
            session_builder = session_builder
                .with_optimization_level(GraphOptimizationLevel::Level2)?;
        }

        // Set number of threads
        session_builder = session_builder.with_intra_threads(config.num_threads)?;

        // Add execution providers
        #[cfg(feature = "cuda")]
        if config.use_gpu {
            session_builder = session_builder.with_execution_providers([
                ort::CUDAExecutionProvider::default().build(),
            ])?;
        }

        // Load model
        let session = session_builder
            .commit_from_file(model_path)
            .map_err(|e| VecStoreError::OnnxRuntime(format!("Failed to load model: {}", e)))?;

        Ok(session)
    }

    /// Get model cache directory
    fn get_model_dir(model_type: &ModelType) -> Result<std::path::PathBuf> {
        let home = directories::UserDirs::new()
            .ok_or_else(|| VecStoreError::Internal("Failed to get user home directory".into()))?;

        let cache_dir = home.home_dir().join(".vecstore").join("models");
        let model_dir = cache_dir.join(model_type.huggingface_id().replace('/', "_"));

        std::fs::create_dir_all(&model_dir)
            .map_err(|e| VecStoreError::Io(e))?;

        Ok(model_dir)
    }

    /// Download model files from HuggingFace
    fn download_model(
        model_type: &ModelType,
        model_dir: &std::path::Path,
    ) -> Result<()> {
        use std::io::Write;

        let base_url = format!(
            "https://huggingface.co/{}/resolve/main",
            model_type.huggingface_id()
        );

        // Files to download
        let files = vec![
            ("model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ];

        for (remote_name, local_name) in files {
            let url = format!("{}/{}", base_url, remote_name);
            let dest = model_dir.join(local_name);

            tracing::info!("Downloading {}...", remote_name);

            let response = ureq::get(&url)
                .call()
                .map_err(|e| VecStoreError::Internal(format!("Failed to download {}: {}", url, e)))?;

            let mut reader = response.into_reader();
            let mut file = std::fs::File::create(&dest)
                .map_err(|e| VecStoreError::Io(e))?;

            std::io::copy(&mut reader, &mut file)
                .map_err(|e| VecStoreError::Io(e))?;

            tracing::info!("Downloaded {} successfully", remote_name);
        }

        Ok(())
    }

    /// Warmup the model with a dummy inference
    fn warmup(&self) -> Result<()> {
        let _ = self.embed("warmup")?;
        if let Ok(mut stats) = self.stats.write() {
            stats.warmup_completed = true;
            // Reset counters after warmup
            stats.total_embeddings = 0;
            stats.total_tokens = 0;
        }
        Ok(())
    }

    /// Get model dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text.to_string()])?;
        embeddings.into_iter().next()
            .ok_or_else(|| VecStoreError::InvalidInput("No embedding returned".to_string()))
    }

    /// Embed a query (with query prefix if configured)
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let text = if let Some(prefix) = &self.config.query_prefix {
            format!("{}{}", prefix, query)
        } else {
            query.to_string()
        };
        self.embed(&text)
    }

    /// Embed a document (with document prefix if configured)
    pub fn embed_document(&self, document: &str) -> Result<Vec<f32>> {
        let text = if let Some(prefix) = &self.config.document_prefix {
            format!("{}{}", prefix, document)
        } else {
            document.to_string()
        };
        self.embed(&text)
    }

    /// Embed multiple texts in a batch
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();

        // Process in configured batch sizes
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.config.batch_size) {
            let batch_embeddings = self.embed_batch_internal(chunk)?;
            all_embeddings.extend(batch_embeddings);
        }

        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.total_embeddings += texts.len() as u64;
            let latency = start.elapsed().as_millis() as f64;
            stats.avg_latency_ms = if stats.total_embeddings == texts.len() as u64 {
                latency
            } else {
                (stats.avg_latency_ms + latency) / 2.0
            };
        }

        Ok(all_embeddings)
    }

    /// Internal batch embedding implementation
    fn embed_batch_internal(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        use ndarray::Array2;
        use ort::value::Tensor;

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| VecStoreError::Tokenization(format!("Tokenization failed: {}", e)))?;

        let batch_size = encodings.len();

        // Get the sequence length from the first encoding (truncate to max_length)
        let seq_length = encodings[0].get_ids().len().min(self.config.max_length);

        // Prepare input tensors
        let mut input_ids = Vec::with_capacity(batch_size * seq_length);
        let mut attention_mask = Vec::with_capacity(batch_size * seq_length);
        let mut token_type_ids = Vec::with_capacity(batch_size * seq_length);

        let mut total_tokens = 0u64;

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();

            total_tokens += ids.len().min(seq_length) as u64;

            // Truncate or pad to seq_length
            for i in 0..seq_length {
                input_ids.push(ids.get(i).copied().unwrap_or(0) as i64);
                attention_mask.push(mask.get(i).copied().unwrap_or(0) as i64);
                token_type_ids.push(type_ids.get(i).copied().unwrap_or(0) as i64);
            }
        }

        // Keep a copy of attention_mask for mean pooling later
        let attention_mask_for_pooling = attention_mask.clone();

        // Convert to ndarray
        let input_ids_array = Array2::from_shape_vec((batch_size, seq_length), input_ids)
            .map_err(|e| VecStoreError::Internal(format!("Array shape error: {}", e)))?;
        let attention_mask_array = Array2::from_shape_vec((batch_size, seq_length), attention_mask)
            .map_err(|e| VecStoreError::Internal(format!("Array shape error: {}", e)))?;
        let token_type_ids_array = Array2::from_shape_vec((batch_size, seq_length), token_type_ids)
            .map_err(|e| VecStoreError::Internal(format!("Array shape error: {}", e)))?;

        // Create ONNX input tensors using ort 2.0 API
        let input_ids_tensor = Tensor::from_array(input_ids_array)
            .map_err(|e| VecStoreError::OnnxRuntime(format!("Failed to create input_ids tensor: {}", e)))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array)
            .map_err(|e| VecStoreError::OnnxRuntime(format!("Failed to create attention_mask tensor: {}", e)))?;
        let token_type_ids_tensor = Tensor::from_array(token_type_ids_array)
            .map_err(|e| VecStoreError::OnnxRuntime(format!("Failed to create token_type_ids tensor: {}", e)))?;

        // Get mutable access to session for inference
        let mut session = self.session.write()
            .map_err(|e| VecStoreError::LockError(format!("Failed to acquire session lock: {}", e)))?;

        // Run inference using ort::inputs! macro
        let outputs = session
            .run(ort::inputs![
                input_ids_tensor,
                attention_mask_tensor,
                token_type_ids_tensor,
            ])
            .map_err(|e| VecStoreError::OnnxRuntime(format!("Inference failed: {}", e)))?;

        // Extract embeddings from output
        // Most sentence transformers output shape: (batch_size, seq_length, hidden_size)
        let output_value = &outputs[0];
        let embeddings_tensor = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| VecStoreError::OnnxRuntime(format!("Failed to extract output: {}", e)))?;

        let embeddings_view = embeddings_tensor.view();
        let shape = embeddings_view.shape();

        // Convert to owned array for processing
        let embeddings_array = if shape.len() == 3 {
            // Shape is (batch_size, seq_length, hidden_size) - need to process for pooling
            ndarray::ArrayD::from_shape_vec(
                shape.to_vec(),
                embeddings_view.iter().copied().collect(),
            ).map_err(|e| VecStoreError::Internal(format!("Array conversion error: {}", e)))?
        } else if shape.len() == 2 {
            // Shape is (batch_size, hidden_size) - already pooled by model
            ndarray::ArrayD::from_shape_vec(
                shape.to_vec(),
                embeddings_view.iter().copied().collect(),
            ).map_err(|e| VecStoreError::Internal(format!("Array conversion error: {}", e)))?
        } else {
            return Err(VecStoreError::OnnxRuntime(format!(
                "Unexpected output shape: {:?}",
                shape
            )));
        };

        // Apply mean pooling
        let embeddings = self.mean_pooling(&embeddings_array, &attention_mask_for_pooling)?;

        // Update token count
        if let Ok(mut stats) = self.stats.write() {
            stats.total_tokens += total_tokens;
        }

        Ok(embeddings)
    }

    /// Apply mean pooling to token embeddings
    fn mean_pooling(
        &self,
        token_embeddings: &ndarray::ArrayD<f32>,
        attention_mask: &[i64],
    ) -> Result<Vec<Vec<f32>>> {
        let shape = token_embeddings.shape();

        // Handle different output shapes
        let (batch_size, seq_length, hidden_size) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape.len() == 2 {
            // Some models output (batch_size, hidden_size) directly
            let batch_size = shape[0];
            let hidden_size = shape[1];

            let mut result = Vec::with_capacity(batch_size);
            for batch_idx in 0..batch_size {
                let mut embedding = Vec::with_capacity(hidden_size);
                for hidden_idx in 0..hidden_size {
                    embedding.push(token_embeddings[[batch_idx, hidden_idx]]);
                }

                // Normalize if configured
                if self.config.normalize {
                    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for val in &mut embedding {
                            *val /= norm;
                        }
                    }
                }

                result.push(embedding);
            }
            return Ok(result);
        } else {
            return Err(VecStoreError::OnnxRuntime(format!(
                "Unexpected output shape: {:?}",
                shape
            )));
        };

        let mut result = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let mut pooled = vec![0.0f32; hidden_size];
            let mut mask_sum = 0.0f32;

            for seq_idx in 0..seq_length {
                let mask_val = attention_mask[batch_idx * seq_length + seq_idx] as f32;
                mask_sum += mask_val;

                for hidden_idx in 0..hidden_size {
                    let token_embedding = token_embeddings[[batch_idx, seq_idx, hidden_idx]];
                    pooled[hidden_idx] += token_embedding * mask_val;
                }
            }

            // Normalize by the sum of attention mask
            if mask_sum > 0.0 {
                for val in &mut pooled {
                    *val /= mask_sum;
                }
            }

            // L2 normalization if configured
            if self.config.normalize {
                let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for val in &mut pooled {
                        *val /= norm;
                    }
                }
            }

            result.push(pooled);
        }

        Ok(result)
    }

    /// Get statistics
    pub fn stats(&self) -> ModelStats {
        self.stats.read().ok().map(|g| g.clone()).unwrap_or_default()
    }
}

/// Fallback implementation without the embeddings feature
#[cfg(not(feature = "embeddings"))]
impl EmbeddingModel {
    /// Load a model
    pub fn load(model_type: ModelType) -> Result<Self> {
        let config = ModelConfig {
            model_type: model_type.clone(),
            ..Default::default()
        };

        Self::with_config(config)
    }

    /// Load with configuration
    pub fn with_config(config: ModelConfig) -> Result<Self> {
        let dimension = config.model_type.dimension();

        // Build simple vocabulary (placeholder - would use tokenizers crate)
        let vocab = Self::build_simple_vocab();

        Ok(Self {
            config,
            dimension,
            vocab,
            stats: RwLock::new(ModelStats::default()),
        })
    }

    /// Build a simple vocabulary for demonstration
    fn build_simple_vocab() -> HashMap<String, usize> {
        // This is a placeholder - real implementation would use the tokenizers crate
        let words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
            "who", "whom", "where", "when", "why", "how", "hello", "world", "test",
            "document", "search", "query", "vector", "embedding", "model", "text",
        ];

        words.iter().enumerate()
            .map(|(i, &w)| (w.to_string(), i))
            .collect()
    }

    /// Get model dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text.to_string()])?;
        embeddings.into_iter().next()
            .ok_or_else(|| VecStoreError::InvalidInput("No embedding returned".to_string()))
    }

    /// Embed multiple texts
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let start = std::time::Instant::now();

        let embeddings: Vec<Vec<f32>> = texts.iter()
            .map(|text| self.embed_text(text))
            .collect();

        // Update stats
        {
            let mut stats = self.stats.write()?;
            stats.total_embeddings += texts.len() as u64;
            let latency = start.elapsed().as_millis() as f64;
            stats.avg_latency_ms = (stats.avg_latency_ms + latency) / 2.0;
        }

        Ok(embeddings)
    }

    /// Embed a single text (internal)
    fn embed_text(&self, text: &str) -> Vec<f32> {
        // Tokenize
        let tokens = self.tokenize(text);

        // Generate deterministic embedding based on tokens
        // This is a placeholder - real implementation would run ONNX inference
        let mut embedding = vec![0.0; self.dimension];

        for (i, token_id) in tokens.iter().enumerate() {
            // Create a pseudo-random but deterministic contribution from each token
            let seed = (*token_id as u64).wrapping_mul(31).wrapping_add(i as u64);
            for j in 0..self.dimension {
                let idx = (seed.wrapping_mul(j as u64 + 1)) % self.dimension as u64;
                embedding[idx as usize] += ((token_id % 100) as f32 - 50.0) / 100.0;
            }
        }

        // Normalize if configured
        if self.config.normalize {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut embedding {
                    *v /= norm;
                }
            }
        }

        embedding
    }

    /// Simple tokenization
    fn tokenize(&self, text: &str) -> Vec<usize> {
        text.to_lowercase()
            .split_whitespace()
            .map(|word| {
                // Clean word
                let clean: String = word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect();

                // Look up in vocabulary or use hash
                self.vocab.get(&clean)
                    .copied()
                    .unwrap_or_else(|| self.hash_word(&clean))
            })
            .take(self.config.max_length)
            .collect()
    }

    /// Hash unknown words
    fn hash_word(&self, word: &str) -> usize {
        let mut hash: usize = 0;
        for byte in word.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as usize);
        }
        hash % 30000 + 1000 // Keep in a reasonable range
    }

    /// Get statistics
    pub fn stats(&self) -> ModelStats {
        let Ok(guard) = self.stats.read() else { return ModelStats::default(); };
        guard.clone()
    }
}

// ============================================================================
// EMBEDDED STORE
// ============================================================================

/// Vector store with embedded embedding model
pub struct EmbeddedStore {
    /// Embedding model
    model: EmbeddingModel,
    /// Vectors by ID
    vectors: RwLock<HashMap<String, Vec<f32>>>,
    /// Original text by ID
    texts: RwLock<HashMap<String, String>>,
    /// Metadata by ID
    metadata: RwLock<HashMap<String, serde_json::Value>>,
    /// Embedding cache
    cache: RwLock<HashMap<String, Vec<f32>>>,
    /// Configuration
    config: StoreConfig,
}

/// Store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreConfig {
    /// Enable embedding cache
    pub enable_cache: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Store original text
    pub store_text: bool,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            max_cache_size: 10000,
            store_text: true,
        }
    }
}

impl EmbeddedStore {
    /// Create new store with model
    pub fn new(model: EmbeddingModel) -> Result<Self> {
        Self::with_config(model, StoreConfig::default())
    }

    /// Create with configuration
    pub fn with_config(model: EmbeddingModel, config: StoreConfig) -> Result<Self> {
        Ok(Self {
            model,
            vectors: RwLock::new(HashMap::new()),
            texts: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            cache: RwLock::new(HashMap::new()),
            config,
        })
    }

    /// Insert text - automatically embedded
    pub fn insert_text(
        &self,
        id: &str,
        text: &str,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        // Check cache first
        let embedding = if self.config.enable_cache {
            let cache = self.cache.read()?;
            cache.get(text).cloned()
        } else {
            None
        };

        let embedding = match embedding {
            Some(e) => {
                // Update cache hit stats
                if let Ok(mut stats) = self.model.stats.write() {
                    stats.cache_hits += 1;
                }
                e
            }
            None => {
                #[cfg(feature = "embeddings")]
                let e = self.model.embed_document(text)?;

                #[cfg(not(feature = "embeddings"))]
                let e = self.model.embed(text)?;

                // Update cache
                if self.config.enable_cache {
                    let mut cache = self.cache.write()?;
                    if cache.len() < self.config.max_cache_size {
                        cache.insert(text.to_string(), e.clone());
                    }
                }

                e
            }
        };

        // Store vector
        self.vectors.write()?.insert(id.to_string(), embedding);

        // Store text
        if self.config.store_text {
            self.texts.write()?.insert(id.to_string(), text.to_string());
        }

        // Store metadata
        if let Some(meta) = metadata {
            self.metadata.write()?.insert(id.to_string(), meta);
        }

        Ok(())
    }

    /// Insert pre-computed vector
    pub fn insert_vector(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        if vector.len() != self.model.dimension() {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.model.dimension(),
                got: vector.len(),
            });
        }

        self.vectors.write()?.insert(id.to_string(), vector);

        if let Some(meta) = metadata {
            self.metadata.write()?.insert(id.to_string(), meta);
        }

        Ok(())
    }

    /// Batch insert texts
    pub fn insert_texts_batch(
        &self,
        items: Vec<(&str, &str, Option<serde_json::Value>)>,
    ) -> Result<usize> {
        let texts: Vec<String> = items.iter().map(|(_, text, _)| text.to_string()).collect();

        #[cfg(feature = "embeddings")]
        let embeddings = {
            // Use document embedding for batch
            let prefixed_texts: Vec<String> = texts
                .iter()
                .map(|t| {
                    if let Some(prefix) = &self.model.config.document_prefix {
                        format!("{}{}", prefix, t)
                    } else {
                        t.clone()
                    }
                })
                .collect();
            self.model.embed_batch(&prefixed_texts)?
        };

        #[cfg(not(feature = "embeddings"))]
        let embeddings = self.model.embed_batch(&texts)?;

        let mut vectors = self.vectors.write()?;
        let mut stored_texts = self.texts.write()?;
        let mut metadata = self.metadata.write()?;

        for ((id, text, meta), embedding) in items.into_iter().zip(embeddings) {
            vectors.insert(id.to_string(), embedding);

            if self.config.store_text {
                stored_texts.insert(id.to_string(), text.to_string());
            }

            if let Some(m) = meta {
                metadata.insert(id.to_string(), m);
            }
        }

        Ok(texts.len())
    }

    /// Delete by ID
    pub fn delete(&self, id: &str) -> bool {
        let Ok(mut vectors) = self.vectors.write() else { return false; };
        let removed = vectors.remove(id).is_some();
        drop(vectors);

        if let Ok(mut texts) = self.texts.write() {
            texts.remove(id);
        }
        if let Ok(mut metadata) = self.metadata.write() {
            metadata.remove(id);
        }
        removed
    }

    /// Search with text query
    pub fn search_text(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        #[cfg(feature = "embeddings")]
        let query_embedding = self.model.embed_query(query)?;

        #[cfg(not(feature = "embeddings"))]
        let query_embedding = self.model.embed(query)?;

        self.search_vector(&query_embedding, top_k)
    }

    /// Search with vector query
    pub fn search_vector(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.model.dimension() {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.model.dimension(),
                got: query.len(),
            });
        }

        let vectors = self.vectors.read()?;
        let texts = self.texts.read()?;
        let metadata = self.metadata.read()?;

        let mut results: Vec<_> = vectors.iter()
            .map(|(id, vec)| {
                let score = cosine_similarity(query, vec);
                SearchResult {
                    id: id.clone(),
                    score,
                    text: texts.get(id).cloned(),
                    metadata: metadata.get(id).cloned(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
    }

    /// Get by ID
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<String>, Option<serde_json::Value>)> {
        let Ok(vectors) = self.vectors.read() else { return None; };
        vectors.get(id).map(|v| {
            let text = self.texts.read().ok().and_then(|t| t.get(id).cloned());
            let meta = self.metadata.read().ok().and_then(|m| m.get(id).cloned());
            (v.clone(), text, meta)
        })
    }

    /// Get model dimension
    pub fn dimension(&self) -> usize {
        self.model.dimension()
    }

    /// Get count
    pub fn len(&self) -> usize {
        let Ok(guard) = self.vectors.read() else { return 0; };
        guard.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        let Ok(guard) = self.vectors.read() else { return true; };
        guard.is_empty()
    }

    /// Get statistics
    pub fn stats(&self) -> StoreStats {
        let vector_count = self.vectors.read().ok().map_or(0, |g| g.len());
        let text_count = self.texts.read().ok().map_or(0, |g| g.len());
        let cache_size = self.cache.read().ok().map_or(0, |g| g.len());
        StoreStats {
            vector_count,
            text_count,
            cache_size,
            model_stats: self.model.stats(),
        }
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Get the embedding model
    pub fn model(&self) -> &EmbeddingModel {
        &self.model
    }
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub text: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Store statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStats {
    pub vector_count: usize,
    pub text_count: usize,
    pub cache_size: usize,
    pub model_stats: ModelStats,
}

// ============================================================================
// EMBEDDING FUNCTIONS (Chroma-style API)
// ============================================================================

/// Embedding function trait (Chroma-style)
pub trait EmbeddingFunction: Send + Sync {
    /// Get the name of this function
    fn name(&self) -> &str;

    /// Get output dimension
    fn dimension(&self) -> usize;

    /// Embed texts
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Embed a single text
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed(&[text.to_string()])?;
        embeddings.into_iter().next()
            .ok_or_else(|| VecStoreError::InvalidInput("No embedding returned".to_string()))
    }

    /// Embed a query (with appropriate prefix if needed)
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embed_one(query)
    }

    /// Embed documents (with appropriate prefix if needed)
    fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(documents)
    }
}

/// Default embedding function using ONNX
pub struct DefaultEmbeddingFunction {
    model: EmbeddingModel,
}

impl DefaultEmbeddingFunction {
    pub fn new() -> Result<Self> {
        let model = EmbeddingModel::load(ModelType::AllMiniLmL6V2)?;
        Ok(Self { model })
    }

    pub fn with_model(model_type: ModelType) -> Result<Self> {
        let model = EmbeddingModel::load(model_type)?;
        Ok(Self { model })
    }

    pub fn with_config(config: ModelConfig) -> Result<Self> {
        let model = EmbeddingModel::with_config(config)?;
        Ok(Self { model })
    }
}

impl EmbeddingFunction for DefaultEmbeddingFunction {
    fn name(&self) -> &str {
        self.model.config.model_type.model_name()
    }

    fn dimension(&self) -> usize {
        self.model.dimension()
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.model.embed_batch(texts)
    }

    #[cfg(feature = "embeddings")]
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.model.embed_query(query)
    }

    #[cfg(feature = "embeddings")]
    fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = documents
            .iter()
            .map(|d| {
                if let Some(prefix) = &self.model.config.document_prefix {
                    format!("{}{}", prefix, d)
                } else {
                    d.clone()
                }
            })
            .collect();
        self.model.embed_batch(&prefixed)
    }
}

// ============================================================================
// QUANTIZED MODEL SUPPORT
// ============================================================================

/// Configuration for quantized models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedModelConfig {
    /// Base model configuration
    pub base_config: ModelConfig,
    /// Quantization type
    pub quantization_type: QuantizationType,
    /// Use dynamic quantization
    pub dynamic_quantization: bool,
}

/// Quantization types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// INT8 quantization (4x smaller, ~2% accuracy loss)
    Int8,
    /// UINT8 quantization
    UInt8,
    /// FP16 quantization (2x smaller, minimal accuracy loss)
    Float16,
    /// Dynamic quantization (quantize at runtime)
    Dynamic,
}

impl Default for QuantizedModelConfig {
    fn default() -> Self {
        Self {
            base_config: ModelConfig::default(),
            quantization_type: QuantizationType::Int8,
            dynamic_quantization: false,
        }
    }
}

// ============================================================================
// MODEL CACHE / REGISTRY
// ============================================================================

/// Global model cache for reusing loaded models
#[cfg(feature = "embeddings")]
pub struct ModelRegistry {
    models: RwLock<HashMap<String, std::sync::Arc<EmbeddingModel>>>,
}

#[cfg(feature = "embeddings")]
impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
        }
    }

    /// Get or load a model
    pub fn get_or_load(&self, model_type: ModelType) -> Result<std::sync::Arc<EmbeddingModel>> {
        let key = model_type.model_name().to_string();

        // Check if already loaded
        if let Ok(models) = self.models.read() {
            if let Some(model) = models.get(&key) {
                return Ok(model.clone());
            }
        }

        // Load the model
        let model = EmbeddingModel::load(model_type)?;
        let model = std::sync::Arc::new(model);

        // Cache it
        if let Ok(mut models) = self.models.write() {
            models.insert(key, model.clone());
        }

        Ok(model)
    }

    /// Unload a model from cache
    pub fn unload(&self, model_type: ModelType) -> bool {
        let key = model_type.model_name().to_string();
        if let Ok(mut models) = self.models.write() {
            models.remove(&key).is_some()
        } else {
            false
        }
    }

    /// Get number of cached models
    pub fn cached_count(&self) -> usize {
        self.models.read().ok().map_or(0, |m| m.len())
    }

    /// Clear all cached models
    pub fn clear(&self) {
        if let Ok(mut models) = self.models.write() {
            models.clear();
        }
    }
}

#[cfg(feature = "embeddings")]
impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HELPERS
// ============================================================================

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_dimension() {
        assert_eq!(ModelType::AllMiniLmL6V2.dimension(), 384);
        assert_eq!(ModelType::AllMpnetBaseV2.dimension(), 768);
        assert_eq!(ModelType::BgeSmallEnV15.dimension(), 384);
        assert_eq!(ModelType::E5BaseV2.dimension(), 768);
    }

    #[test]
    fn test_model_type_prefixes() {
        assert!(ModelType::BgeSmallEnV15.requires_query_prefix());
        assert!(ModelType::E5SmallV2.requires_query_prefix());
        assert!(!ModelType::AllMiniLmL6V2.requires_query_prefix());

        assert!(ModelType::BgeSmallEnV15.default_query_prefix().is_some());
        assert!(ModelType::E5SmallV2.default_document_prefix().is_some());
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::for_model(ModelType::AllMiniLmL6V2)
            .with_max_length(256)
            .with_threads(8)
            .with_normalize(false)
            .with_batch_size(64);

        assert_eq!(config.max_length, 256);
        assert_eq!(config.num_threads, 8);
        assert!(!config.normalize);
        assert_eq!(config.batch_size, 64);
    }

    #[test]
    fn test_embedding_model_fallback() {
        // Test the fallback (non-ONNX) implementation
        #[cfg(not(feature = "embeddings"))]
        {
            let model = EmbeddingModel::load(ModelType::AllMiniLmL6V2).unwrap();

            let embedding = model.embed("Hello world").unwrap();
            assert_eq!(embedding.len(), 384);

            // Check normalization
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_embedded_store() {
        #[cfg(not(feature = "embeddings"))]
        {
            let model = EmbeddingModel::load(ModelType::AllMiniLmL6V2).unwrap();
            let store = EmbeddedStore::new(model).unwrap();

            store.insert_text("doc1", "Hello world", None).unwrap();
            store.insert_text("doc2", "Goodbye world", None).unwrap();

            let results = store.search_text("greeting hello", 10).unwrap();
            assert!(!results.is_empty());

            // "Hello world" should be more similar to "greeting hello"
            assert_eq!(results[0].id, "doc1");
        }
    }

    #[test]
    fn test_batch_insert() {
        #[cfg(not(feature = "embeddings"))]
        {
            let model = EmbeddingModel::load(ModelType::AllMiniLmL6V2).unwrap();
            let store = EmbeddedStore::new(model).unwrap();

            let items = vec![
                ("doc1", "Hello world", None),
                ("doc2", "Goodbye world", None),
                ("doc3", "Test document", None),
            ];

            let count = store.insert_texts_batch(items).unwrap();
            assert_eq!(count, 3);
            assert_eq!(store.len(), 3);
        }
    }

    #[test]
    fn test_embedding_function() {
        #[cfg(not(feature = "embeddings"))]
        {
            let func = DefaultEmbeddingFunction::new().unwrap();

            assert_eq!(func.name(), "all-MiniLM-L6-v2");
            assert_eq!(func.dimension(), 384);

            let texts = vec!["Hello".to_string(), "World".to_string()];
            let embeddings = func.embed(&texts).unwrap();

            assert_eq!(embeddings.len(), 2);
            assert_eq!(embeddings[0].len(), 384);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_store_cache() {
        #[cfg(not(feature = "embeddings"))]
        {
            let model = EmbeddingModel::load(ModelType::AllMiniLmL6V2).unwrap();
            let store = EmbeddedStore::new(model).unwrap();

            // Insert same text twice
            store.insert_text("doc1", "Hello world", None).unwrap();
            store.insert_text("doc2", "Hello world", None).unwrap();

            let stats = store.stats();
            assert_eq!(stats.cache_size, 1); // Same text cached once
        }
    }

    // Integration tests that require actual model files
    #[test]
    #[ignore]
    #[cfg(feature = "embeddings")]
    fn test_onnx_embedding_model() {
        let model = EmbeddingModel::load(ModelType::AllMiniLmL6V2).unwrap();

        let embedding = model.embed("This is a test sentence").unwrap();
        assert_eq!(embedding.len(), 384);

        // Check L2 normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    #[ignore]
    #[cfg(feature = "embeddings")]
    fn test_onnx_batch_embedding() {
        let model = EmbeddingModel::load(ModelType::AllMiniLmL6V2).unwrap();

        let texts = vec![
            "First sentence".to_string(),
            "Second sentence".to_string(),
            "Third sentence".to_string(),
        ];

        let embeddings = model.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 384);

        // All should be normalized
        for emb in &embeddings {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.01);
        }
    }

    #[test]
    #[ignore]
    #[cfg(feature = "embeddings")]
    fn test_bge_model_with_prefix() {
        let config = ModelConfig::for_model(ModelType::BgeSmallEnV15);
        assert!(config.query_prefix.is_some());

        let model = EmbeddingModel::with_config(config).unwrap();

        let query_emb = model.embed_query("search query").unwrap();
        let doc_emb = model.embed_document("document text").unwrap();

        assert_eq!(query_emb.len(), 384);
        assert_eq!(doc_emb.len(), 384);
    }

    #[test]
    #[ignore]
    #[cfg(feature = "embeddings")]
    fn test_model_registry() {
        let registry = ModelRegistry::new();

        // Load a model
        let model1 = registry.get_or_load(ModelType::AllMiniLmL6V2).unwrap();
        assert_eq!(registry.cached_count(), 1);

        // Get it again (should be cached)
        let model2 = registry.get_or_load(ModelType::AllMiniLmL6V2).unwrap();
        assert_eq!(registry.cached_count(), 1);

        // Same Arc reference
        assert!(std::sync::Arc::ptr_eq(&model1, &model2));

        // Unload
        assert!(registry.unload(ModelType::AllMiniLmL6V2));
        assert_eq!(registry.cached_count(), 0);
    }
}
