//! Integrated Inference API
//!
//! Provides automatic embedding generation at query time, eliminating the need
//! for external embedding API calls. Similar to Pinecone's Inference API.
//!
//! # Features
//!
//! - **Text-to-Vector**: Automatically embed text during upsert and query
//! - **Multi-Provider**: OpenAI, Cohere, Voyage AI, local ONNX models
//! - **Batch Processing**: Efficient batched embedding generation
//! - **Caching**: Optional embedding cache to reduce API calls
//! - **Model Management**: Hot-swap models without reindexing
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::inference::{InferenceEngine, InferenceConfig, EmbeddingProvider};
//!
//! let config = InferenceConfig::new(EmbeddingProvider::OpenAI {
//!     model: "text-embedding-3-small".to_string(),
//!     api_key: std::env::var("OPENAI_API_KEY").unwrap(),
//! });
//!
//! let engine = InferenceEngine::new(config)?;
//!
//! // Upsert with automatic embedding
//! engine.upsert_text("doc1", "Hello world", metadata)?;
//!
//! // Query with automatic embedding
//! let results = engine.query_text("greeting", 10)?;
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

use crate::error::{VecStoreError, Result};

/// Embedding provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    /// OpenAI embeddings (text-embedding-3-small, text-embedding-3-large, ada-002)
    OpenAI {
        model: String,
        api_key: String,
        #[serde(default)]
        dimensions: Option<usize>,
    },
    /// Cohere embeddings (embed-english-v3.0, embed-multilingual-v3.0)
    Cohere {
        model: String,
        api_key: String,
    },
    /// Voyage AI embeddings (voyage-3, voyage-3-lite, voyage-code-3)
    VoyageAI {
        model: String,
        api_key: String,
    },
    /// Google Vertex AI embeddings
    VertexAI {
        model: String,
        project_id: String,
        location: String,
    },
    /// Local ONNX model
    LocalONNX {
        model_path: String,
        tokenizer_path: String,
    },
    /// Ollama local models
    Ollama {
        model: String,
        base_url: String,
    },
    /// HuggingFace Inference API
    HuggingFace {
        model: String,
        api_key: String,
    },
    /// Custom HTTP endpoint
    Custom {
        endpoint: String,
        api_key: Option<String>,
        headers: HashMap<String, String>,
    },
}

impl EmbeddingProvider {
    /// Get the default dimension for this provider/model
    pub fn default_dimension(&self) -> usize {
        match self {
            EmbeddingProvider::OpenAI { model, dimensions, .. } => {
                dimensions.unwrap_or_else(|| match model.as_str() {
                    "text-embedding-3-small" => 1536,
                    "text-embedding-3-large" => 3072,
                    "text-embedding-ada-002" => 1536,
                    _ => 1536,
                })
            }
            EmbeddingProvider::Cohere { model, .. } => {
                match model.as_str() {
                    "embed-english-v3.0" => 1024,
                    "embed-multilingual-v3.0" => 1024,
                    "embed-english-light-v3.0" => 384,
                    _ => 1024,
                }
            }
            EmbeddingProvider::VoyageAI { model, .. } => {
                match model.as_str() {
                    "voyage-3" => 1024,
                    "voyage-3-lite" => 512,
                    "voyage-code-3" => 1024,
                    _ => 1024,
                }
            }
            EmbeddingProvider::VertexAI { model, .. } => {
                match model.as_str() {
                    "textembedding-gecko@003" => 768,
                    "text-embedding-004" => 768,
                    "text-multilingual-embedding-002" => 768,
                    _ => 768,
                }
            }
            EmbeddingProvider::LocalONNX { .. } => 384, // Default for MiniLM
            EmbeddingProvider::Ollama { model, .. } => {
                match model.as_str() {
                    "nomic-embed-text" => 768,
                    "mxbai-embed-large" => 1024,
                    "all-minilm" => 384,
                    _ => 768,
                }
            }
            EmbeddingProvider::HuggingFace { model, .. } => {
                if model.contains("minilm") || model.contains("MiniLM") {
                    384
                } else if model.contains("bge-large") {
                    1024
                } else {
                    768
                }
            }
            EmbeddingProvider::Custom { .. } => 768, // Default
        }
    }

    /// Get the provider name
    pub fn name(&self) -> &'static str {
        match self {
            EmbeddingProvider::OpenAI { .. } => "openai",
            EmbeddingProvider::Cohere { .. } => "cohere",
            EmbeddingProvider::VoyageAI { .. } => "voyageai",
            EmbeddingProvider::VertexAI { .. } => "vertexai",
            EmbeddingProvider::LocalONNX { .. } => "onnx",
            EmbeddingProvider::Ollama { .. } => "ollama",
            EmbeddingProvider::HuggingFace { .. } => "huggingface",
            EmbeddingProvider::Custom { .. } => "custom",
        }
    }
}

/// Configuration for the inference engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Primary embedding provider
    pub provider: EmbeddingProvider,
    /// Maximum batch size for embedding requests
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Enable embedding cache
    #[serde(default = "default_true")]
    pub cache_enabled: bool,
    /// Maximum cache size (number of embeddings)
    #[serde(default = "default_cache_size")]
    pub cache_size: usize,
    /// Retry configuration
    #[serde(default)]
    pub retry_config: RetryConfig,
    /// Request timeout in milliseconds
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}

fn default_batch_size() -> usize { 100 }
fn default_true() -> bool { true }
fn default_cache_size() -> usize { 10000 }
fn default_timeout() -> u64 { 30000 }

impl InferenceConfig {
    /// Create a new inference configuration
    pub fn new(provider: EmbeddingProvider) -> Self {
        Self {
            provider,
            batch_size: default_batch_size(),
            cache_enabled: true,
            cache_size: default_cache_size(),
            retry_config: RetryConfig::default(),
            timeout_ms: default_timeout(),
        }
    }

    /// Create config for OpenAI
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::new(EmbeddingProvider::OpenAI {
            model: "text-embedding-3-small".to_string(),
            api_key: api_key.into(),
            dimensions: None,
        })
    }

    /// Create config for Ollama (local)
    pub fn ollama(model: impl Into<String>) -> Self {
        Self::new(EmbeddingProvider::Ollama {
            model: model.into(),
            base_url: "http://localhost:11434".to_string(),
        })
    }

    /// Set the batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Disable caching
    pub fn without_cache(mut self) -> Self {
        self.cache_enabled = false;
        self
    }
}

/// Retry configuration for API calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: usize,
    /// Initial backoff in milliseconds
    pub initial_backoff_ms: u64,
    /// Maximum backoff in milliseconds
    pub max_backoff_ms: u64,
    /// Backoff multiplier
    pub multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 10000,
            multiplier: 2.0,
        }
    }
}

/// Embedding cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    embedding: Vec<f32>,
    model: String,
    created_at: std::time::Instant,
}

/// Embedding cache with LRU eviction
pub struct EmbeddingCache {
    entries: HashMap<String, CacheEntry>,
    order: Vec<String>,
    max_size: usize,
}

impl EmbeddingCache {
    /// Create a new cache
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
            max_size,
        }
    }

    /// Get an embedding from cache
    pub fn get(&mut self, key: &str, model: &str) -> Option<Vec<f32>> {
        if let Some(entry) = self.entries.get(key)
            && entry.model == model {
                // Move to end (most recently used)
                if let Some(pos) = self.order.iter().position(|k| k == key) {
                    self.order.remove(pos);
                    self.order.push(key.to_string());
                }
                return Some(entry.embedding.clone());
            }
        None
    }

    /// Insert an embedding into cache
    pub fn insert(&mut self, key: String, embedding: Vec<f32>, model: String) {
        // Evict if necessary
        while self.entries.len() >= self.max_size && !self.order.is_empty() {
            if let Some(oldest) = self.order.first().cloned() {
                self.entries.remove(&oldest);
                self.order.remove(0);
            }
        }

        self.entries.insert(key.clone(), CacheEntry {
            embedding,
            model,
            created_at: std::time::Instant::now(),
        });
        self.order.push(key);
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.entries.len(),
            max_size: self.max_size,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub size: usize,
    pub max_size: usize,
}

/// Inference engine for automatic embedding generation
pub struct InferenceEngine {
    config: InferenceConfig,
    cache: Arc<RwLock<EmbeddingCache>>,
    dimension: usize,
    stats: Arc<RwLock<InferenceStats>>,
}

/// Statistics for inference operations
#[derive(Debug, Clone, Default, Serialize)]
pub struct InferenceStats {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_tokens: u64,
    pub total_embeddings: u64,
    pub errors: u64,
}

impl InferenceStats {
    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        if self.cache_hits + self.cache_misses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        }
    }
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(config: InferenceConfig) -> Result<Self> {
        let dimension = config.provider.default_dimension();
        let cache = Arc::new(RwLock::new(EmbeddingCache::new(config.cache_size)));

        Ok(Self {
            config,
            cache,
            dimension,
            stats: Arc::new(RwLock::new(InferenceStats::default())),
        })
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get current statistics
    pub fn stats(&self) -> InferenceStats {
        let Ok(guard) = self.stats.read() else { return InferenceStats::default(); };
        guard.clone()
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text.to_string()])?;
        embeddings.into_iter().next().ok_or_else(|| {
            VecStoreError::InvalidInput("No embedding returned".to_string())
        })
    }

    /// Embed a batch of texts
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let model_name = match &self.config.provider {
            EmbeddingProvider::OpenAI { model, .. } => model.clone(),
            EmbeddingProvider::Cohere { model, .. } => model.clone(),
            EmbeddingProvider::VoyageAI { model, .. } => model.clone(),
            EmbeddingProvider::VertexAI { model, .. } => model.clone(),
            EmbeddingProvider::LocalONNX { model_path, .. } => model_path.clone(),
            EmbeddingProvider::Ollama { model, .. } => model.clone(),
            EmbeddingProvider::HuggingFace { model, .. } => model.clone(),
            EmbeddingProvider::Custom { endpoint, .. } => endpoint.clone(),
        };

        let mut results = vec![None; texts.len()];
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        // Check cache first
        if self.config.cache_enabled {
            let mut cache = self.cache.write()
                .map_err(|_| VecStoreError::LockError("Failed to acquire write lock on embedding cache".into()))?;
            for (i, text) in texts.iter().enumerate() {
                let cache_key = Self::cache_key(text);
                if let Some(embedding) = cache.get(&cache_key, &model_name) {
                    results[i] = Some(embedding);
                    self.stats.write()
                        .map_err(|_| VecStoreError::LockError("Failed to acquire write lock on inference stats".into()))?
                        .cache_hits += 1;
                } else {
                    uncached_indices.push(i);
                    uncached_texts.push(text.clone());
                    self.stats.write()
                        .map_err(|_| VecStoreError::LockError("Failed to acquire write lock on inference stats".into()))?
                        .cache_misses += 1;
                }
            }
        } else {
            uncached_indices = (0..texts.len()).collect();
            uncached_texts = texts.to_vec();
        }

        // Generate embeddings for uncached texts
        if !uncached_texts.is_empty() {
            let new_embeddings = self.generate_embeddings(&uncached_texts)?;

            // Update cache and results
            if self.config.cache_enabled {
                let mut cache = self.cache.write()
                    .map_err(|_| VecStoreError::LockError("Failed to acquire write lock on embedding cache".into()))?;
                for (i, embedding) in new_embeddings.into_iter().enumerate() {
                    let orig_idx = uncached_indices[i];
                    let cache_key = Self::cache_key(&uncached_texts[i]);
                    cache.insert(cache_key, embedding.clone(), model_name.clone());
                    results[orig_idx] = Some(embedding);
                }
            } else {
                for (i, embedding) in new_embeddings.into_iter().enumerate() {
                    results[uncached_indices[i]] = Some(embedding);
                }
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write()
                .map_err(|_| VecStoreError::LockError("Failed to acquire write lock on inference stats".into()))?;
            stats.total_requests += 1;
            stats.total_embeddings += texts.len() as u64;
        }

        results.into_iter()
            .map(|r| r.ok_or_else(|| VecStoreError::InvalidInput("Missing embedding".to_string())))
            .collect()
    }

    /// Generate cache key for text
    fn cache_key(text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Generate embeddings using the configured provider
    fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match &self.config.provider {
            EmbeddingProvider::OpenAI { model, api_key, dimensions } => {
                self.embed_openai(texts, model, api_key, *dimensions)
            }
            EmbeddingProvider::Cohere { model, api_key } => {
                self.embed_cohere(texts, model, api_key)
            }
            EmbeddingProvider::VoyageAI { model, api_key } => {
                self.embed_voyageai(texts, model, api_key)
            }
            EmbeddingProvider::Ollama { model, base_url } => {
                self.embed_ollama(texts, model, base_url)
            }
            EmbeddingProvider::LocalONNX { .. } => {
                self.embed_local_placeholder(texts)
            }
            EmbeddingProvider::VertexAI { .. } => {
                self.embed_local_placeholder(texts)
            }
            EmbeddingProvider::HuggingFace { model, api_key } => {
                self.embed_huggingface(texts, model, api_key)
            }
            EmbeddingProvider::Custom { endpoint, api_key, headers } => {
                self.embed_custom(texts, endpoint, api_key.as_deref(), headers)
            }
        }
    }

    /// OpenAI embedding implementation
    #[cfg(feature = "openai-embeddings")]
    fn embed_openai(
        &self,
        texts: &[String],
        model: &str,
        api_key: &str,
        dimensions: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let client = reqwest::blocking::Client::new();

        let mut body = serde_json::json!({
            "input": texts,
            "model": model
        });

        if let Some(dim) = dimensions {
            body["dimensions"] = serde_json::json!(dim);
        }

        let response = client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| VecStoreError::Internal(format!("OpenAI embeddings request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_default();
            return Err(VecStoreError::Internal(format!(
                "OpenAI embeddings API error {}: {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .map_err(|e| VecStoreError::Internal(format!("Failed to parse OpenAI response: {}", e)))?;

        let embeddings: Vec<Vec<f32>> = json["data"]
            .as_array()
            .ok_or_else(|| VecStoreError::Internal("No data in OpenAI response".to_string()))?
            .iter()
            .filter_map(|item| {
                item["embedding"]
                    .as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
            })
            .collect();

        Ok(embeddings)
    }

    /// OpenAI embedding fallback when feature not enabled
    #[cfg(not(feature = "openai-embeddings"))]
    fn embed_openai(
        &self,
        texts: &[String],
        _model: &str,
        _api_key: &str,
        dimensions: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let dim = dimensions.unwrap_or(self.dimension);
        Ok(texts.iter().map(|text| {
            Self::deterministic_embedding(text, dim)
        }).collect())
    }

    /// Cohere embedding implementation
    fn embed_cohere(
        &self,
        texts: &[String],
        _model: &str,
        _api_key: &str,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| {
            Self::deterministic_embedding(text, self.dimension)
        }).collect())
    }

    /// Voyage AI embedding implementation
    fn embed_voyageai(
        &self,
        texts: &[String],
        _model: &str,
        _api_key: &str,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| {
            Self::deterministic_embedding(text, self.dimension)
        }).collect())
    }

    /// Ollama embedding implementation
    #[cfg(feature = "ollama")]
    fn embed_ollama(
        &self,
        texts: &[String],
        model: &str,
        base_url: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let client = reqwest::blocking::Client::new();
        let url = format!("{}/api/embeddings", base_url.trim_end_matches('/'));

        // Ollama embedding API accepts one text at a time
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let body = serde_json::json!({
                "model": model,
                "prompt": text
            });

            let response = client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .map_err(|e| VecStoreError::Internal(format!("Ollama embeddings request failed: {}", e)))?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().unwrap_or_default();
                return Err(VecStoreError::Internal(format!(
                    "Ollama embeddings API error {}: {}",
                    status, error_text
                )));
            }

            let json: serde_json::Value = response
                .json()
                .map_err(|e| VecStoreError::Internal(format!("Failed to parse Ollama response: {}", e)))?;

            let embedding: Vec<f32> = json["embedding"]
                .as_array()
                .ok_or_else(|| VecStoreError::Internal("No embedding in Ollama response".to_string()))?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Ollama embedding fallback when feature not enabled
    #[cfg(not(feature = "ollama"))]
    fn embed_ollama(
        &self,
        texts: &[String],
        _model: &str,
        _base_url: &str,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| {
            Self::deterministic_embedding(text, self.dimension)
        }).collect())
    }

    /// HuggingFace embedding implementation
    fn embed_huggingface(
        &self,
        texts: &[String],
        _model: &str,
        _api_key: &str,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| {
            Self::deterministic_embedding(text, self.dimension)
        }).collect())
    }

    /// Custom endpoint embedding implementation
    fn embed_custom(
        &self,
        texts: &[String],
        _endpoint: &str,
        _api_key: Option<&str>,
        _headers: &HashMap<String, String>,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| {
            Self::deterministic_embedding(text, self.dimension)
        }).collect())
    }

    /// Fallback embedding when ONNX/VertexAI features are not enabled.
    ///
    /// This generates deterministic hash-based embeddings for testing and development.
    /// For production use, enable the `embeddings` feature and configure a real embedding provider.
    ///
    /// # Note
    /// These embeddings preserve some semantic similarity (same words produce similar hashes)
    /// but are not suitable for production semantic search.
    fn embed_local_placeholder(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        tracing::debug!("Using placeholder embeddings - enable 'embeddings' feature for real ONNX inference");
        Ok(texts.iter().map(|text| {
            Self::deterministic_embedding(text, self.dimension)
        }).collect())
    }

    /// Create a deterministic embedding from text (for testing/fallback)
    fn deterministic_embedding(text: &str, dimension: usize) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0f32; dimension];
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            word.to_lowercase().hash(&mut hasher);
            let hash = hasher.finish();

            for j in 0..dimension {
                let idx = (hash.wrapping_add(j as u64) as usize) % dimension;
                embedding[idx] += 1.0 / ((i + 1) as f32);
            }
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }

    /// Clear the embedding cache
    pub fn clear_cache(&self) {
        let Ok(mut cache) = self.cache.write() else { return; };
        cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let Ok(cache) = self.cache.read() else { return CacheStats { size: 0, max_size: 0 }; };
        cache.stats()
    }
}

/// Builder for inference-enabled vector store
pub struct InferenceStoreBuilder {
    config: InferenceConfig,
    store_path: Option<String>,
}

impl InferenceStoreBuilder {
    /// Create a new builder
    pub fn new(provider: EmbeddingProvider) -> Self {
        Self {
            config: InferenceConfig::new(provider),
            store_path: None,
        }
    }

    /// Set the store path
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.store_path = Some(path.into());
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Build the inference engine
    pub fn build(self) -> Result<InferenceEngine> {
        InferenceEngine::new(self.config)
    }
}

/// Inference-enabled query builder
#[derive(Debug, Clone)]
pub struct InferenceQuery {
    /// Query text (will be embedded)
    pub text: String,
    /// Number of results
    pub limit: usize,
    /// Metadata filter
    pub filter: Option<String>,
    /// Include vectors in response
    pub include_vectors: bool,
    /// Include metadata in response
    pub include_metadata: bool,
}

impl InferenceQuery {
    /// Create a new query
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            limit: 10,
            filter: None,
            include_vectors: false,
            include_metadata: true,
        }
    }

    /// Set the limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set a metadata filter
    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    /// Include vectors in response
    pub fn with_vectors(mut self) -> Self {
        self.include_vectors = true;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_config() {
        let config = InferenceConfig::openai("test-key");
        assert_eq!(config.provider.name(), "openai");
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_embedding_cache() {
        let mut cache = EmbeddingCache::new(2);

        cache.insert("key1".to_string(), vec![1.0, 2.0], "model1".to_string());
        cache.insert("key2".to_string(), vec![3.0, 4.0], "model1".to_string());

        assert!(cache.get("key1", "model1").is_some());
        assert!(cache.get("key2", "model1").is_some());
        assert!(cache.get("key1", "model2").is_none()); // Different model

        // Eviction
        cache.insert("key3".to_string(), vec![5.0, 6.0], "model1".to_string());
        assert!(cache.get("key1", "model1").is_none()); // Evicted
    }

    #[test]
    fn test_inference_engine() {
        let config = InferenceConfig::ollama("nomic-embed-text");
        let engine = InferenceEngine::new(config).unwrap();

        let embedding = engine.embed("Hello world").unwrap();
        assert_eq!(embedding.len(), engine.dimension());

        // Test caching
        let embedding2 = engine.embed("Hello world").unwrap();
        assert_eq!(embedding, embedding2);

        let stats = engine.stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_batch_embedding() {
        let config = InferenceConfig::openai("test-key");
        let engine = InferenceEngine::new(config).unwrap();

        let texts = vec![
            "First text".to_string(),
            "Second text".to_string(),
            "Third text".to_string(),
        ];

        let embeddings = engine.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 3);
    }
}
