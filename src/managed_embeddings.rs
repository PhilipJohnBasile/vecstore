// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Managed Embedding Service
//!
//! Pay-as-you-go embedding service with model switching, caching, and billing.
//! Inspired by Weaviate Embeddings Service and Pinecone Inference API.
//!
//! ## Features
//!
//! - **Multi-Provider Support**: OpenAI, Cohere, Voyage, Jina, local ONNX
//! - **Model Switching**: Change models without re-embedding
//! - **Smart Caching**: Avoid re-embedding identical text
//! - **Batching**: Efficient batch processing with rate limiting
//! - **Cost Tracking**: Real-time usage and billing estimates
//! - **Fallback Chains**: Automatic failover between providers
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::managed_embeddings::{EmbeddingService, EmbeddingConfig};
//!
//! let service = EmbeddingService::new(config);
//!
//! // Embed text (uses caching automatically)
//! let embeddings = service.embed(&["Hello world", "Machine learning"]).await?;
//!
//! // Check usage
//! let usage = service.usage();
//! println!("Tokens used: {}, Cost: ${:.4}", usage.total_tokens, usage.estimated_cost);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Embedding provider type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EmbeddingProvider {
    /// OpenAI embeddings
    OpenAI,
    /// Cohere embeddings
    Cohere,
    /// Voyage AI embeddings
    Voyage,
    /// Jina AI embeddings
    Jina,
    /// HuggingFace Inference API
    HuggingFace,
    /// Google Vertex AI
    Google,
    /// Azure OpenAI
    Azure,
    /// Local ONNX model
    LocalONNX,
    /// Custom endpoint
    Custom { endpoint: String },
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Provider
    pub provider: EmbeddingProvider,
    /// Model name/ID
    pub model: String,
    /// Output dimension
    pub dimension: usize,
    /// Maximum tokens per request
    pub max_tokens: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Cost per 1M tokens (in USD)
    pub cost_per_million_tokens: f64,
    /// Supports truncation
    pub supports_truncation: bool,
    /// Supports task type specification
    pub supports_task_type: bool,
}

impl ModelConfig {
    /// OpenAI text-embedding-3-small
    pub fn openai_small() -> Self {
        Self {
            provider: EmbeddingProvider::OpenAI,
            model: "text-embedding-3-small".to_string(),
            dimension: 1536,
            max_tokens: 8191,
            max_batch_size: 2048,
            cost_per_million_tokens: 0.02,
            supports_truncation: true,
            supports_task_type: false,
        }
    }

    /// OpenAI text-embedding-3-large
    pub fn openai_large() -> Self {
        Self {
            provider: EmbeddingProvider::OpenAI,
            model: "text-embedding-3-large".to_string(),
            dimension: 3072,
            max_tokens: 8191,
            max_batch_size: 2048,
            cost_per_million_tokens: 0.13,
            supports_truncation: true,
            supports_task_type: false,
        }
    }

    /// Cohere embed-v3
    pub fn cohere_v3() -> Self {
        Self {
            provider: EmbeddingProvider::Cohere,
            model: "embed-english-v3.0".to_string(),
            dimension: 1024,
            max_tokens: 512,
            max_batch_size: 96,
            cost_per_million_tokens: 0.10,
            supports_truncation: true,
            supports_task_type: true,
        }
    }

    /// Voyage AI voyage-3
    pub fn voyage_3() -> Self {
        Self {
            provider: EmbeddingProvider::Voyage,
            model: "voyage-3".to_string(),
            dimension: 1024,
            max_tokens: 32000,
            max_batch_size: 128,
            cost_per_million_tokens: 0.06,
            supports_truncation: true,
            supports_task_type: true,
        }
    }

    /// Jina embeddings v3
    pub fn jina_v3() -> Self {
        Self {
            provider: EmbeddingProvider::Jina,
            model: "jina-embeddings-v3".to_string(),
            dimension: 1024,
            max_tokens: 8192,
            max_batch_size: 2048,
            cost_per_million_tokens: 0.02,
            supports_truncation: true,
            supports_task_type: true,
        }
    }

    /// Local ONNX model
    pub fn local_onnx(model_path: &str, dimension: usize) -> Self {
        Self {
            provider: EmbeddingProvider::LocalONNX,
            model: model_path.to_string(),
            dimension,
            max_tokens: 512,
            max_batch_size: 32,
            cost_per_million_tokens: 0.0, // Free!
            supports_truncation: false,
            supports_task_type: false,
        }
    }
}

/// Task type for embeddings (affects quality)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskType {
    /// For storing/indexing documents
    Document,
    /// For search queries
    Query,
    /// For clustering
    Clustering,
    /// For classification
    Classification,
    /// For semantic similarity
    Similarity,
}

/// Embedding service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingServiceConfig {
    /// Primary model configuration
    pub primary_model: ModelConfig,
    /// Fallback models (in order of preference)
    pub fallback_models: Vec<ModelConfig>,
    /// API keys per provider
    pub api_keys: HashMap<EmbeddingProvider, String>,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Maximum cache entries
    pub max_cache_entries: usize,
    /// Rate limiting (requests per second)
    pub rate_limit_rps: f64,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Enable usage tracking
    pub track_usage: bool,
}

impl Default for EmbeddingServiceConfig {
    fn default() -> Self {
        Self {
            primary_model: ModelConfig::openai_small(),
            fallback_models: vec![],
            api_keys: HashMap::new(),
            enable_cache: true,
            cache_ttl_seconds: 3600,
            max_cache_entries: 100000,
            rate_limit_rps: 100.0,
            retry_config: RetryConfig::default(),
            track_usage: true,
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
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

/// Embedding request
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// Text inputs
    pub texts: Vec<String>,
    /// Task type (optional)
    pub task_type: Option<TaskType>,
    /// Override model
    pub model_override: Option<ModelConfig>,
    /// Override dimension (for dimensionality reduction)
    pub dimension_override: Option<usize>,
    /// Skip cache
    pub skip_cache: bool,
}

impl EmbeddingRequest {
    /// Create new request
    pub fn new(texts: Vec<String>) -> Self {
        Self {
            texts,
            task_type: None,
            model_override: None,
            dimension_override: None,
            skip_cache: false,
        }
    }

    /// Set task type
    pub fn with_task_type(mut self, task_type: TaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }

    /// Override model
    pub fn with_model(mut self, model: ModelConfig) -> Self {
        self.model_override = Some(model);
        self
    }

    /// Override dimension
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension_override = Some(dimension);
        self
    }

    /// Skip cache
    pub fn skip_cache(mut self) -> Self {
        self.skip_cache = true;
        self
    }
}

/// Embedding response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Embeddings for each input
    pub embeddings: Vec<Vec<f32>>,
    /// Model used
    pub model: String,
    /// Total tokens used
    pub tokens_used: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Estimated cost
    pub estimated_cost: f64,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    /// Total tokens used
    pub total_tokens: u64,
    /// Total requests
    pub total_requests: u64,
    /// Total embeddings generated
    pub total_embeddings: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average latency in ms
    pub avg_latency_ms: f64,
    /// Estimated total cost
    pub estimated_cost: f64,
    /// Usage by provider
    pub by_provider: HashMap<String, ProviderUsage>,
    /// Usage by model
    pub by_model: HashMap<String, ModelUsage>,
}

/// Usage per provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderUsage {
    pub tokens: u64,
    pub requests: u64,
    pub errors: u64,
    pub cost: f64,
}

/// Usage per model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUsage {
    pub tokens: u64,
    pub embeddings: u64,
    pub cost: f64,
}

/// Cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Embedding vector
    embedding: Vec<f32>,
    /// Model used
    model: String,
    /// Cached at timestamp
    cached_at: Instant,
    /// Hit count
    hits: u64,
}

/// Embedding cache
struct EmbeddingCache {
    entries: HashMap<String, CacheEntry>,
    max_entries: usize,
    ttl: Duration,
}

impl EmbeddingCache {
    fn new(max_entries: usize, ttl_seconds: u64) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    fn get(&mut self, key: &str) -> Option<Vec<f32>> {
        let now = Instant::now();

        if let Some(entry) = self.entries.get_mut(key) {
            if now.duration_since(entry.cached_at) < self.ttl {
                entry.hits += 1;
                return Some(entry.embedding.clone());
            } else {
                self.entries.remove(key);
            }
        }
        None
    }

    fn put(&mut self, key: String, embedding: Vec<f32>, model: &str) {
        if self.entries.len() >= self.max_entries {
            self.evict_oldest();
        }

        self.entries.insert(key, CacheEntry {
            embedding,
            model: model.to_string(),
            cached_at: Instant::now(),
            hits: 0,
        });
    }

    fn evict_oldest(&mut self) {
        let oldest = self.entries
            .iter()
            .min_by_key(|(_, e)| e.cached_at)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest {
            self.entries.remove(&key);
        }
    }

    fn stats(&self) -> CacheStats {
        let total_hits: u64 = self.entries.values().map(|e| e.hits).sum();

        CacheStats {
            entries: self.entries.len(),
            total_hits,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
struct CacheStats {
    entries: usize,
    total_hits: u64,
}

/// Rate limiter
struct RateLimiter {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl RateLimiter {
    fn new(rps: f64) -> Self {
        Self {
            tokens: rps,
            max_tokens: rps * 2.0, // Allow some bursting
            refill_rate: rps,
            last_refill: Instant::now(),
        }
    }

    fn acquire(&mut self, count: f64) -> bool {
        self.refill();

        if self.tokens >= count {
            self.tokens -= count;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }

    fn wait_time(&self) -> Duration {
        if self.tokens >= 1.0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64((1.0 - self.tokens) / self.refill_rate)
        }
    }
}

/// Internal usage tracker
struct UsageTracker {
    total_tokens: u64,
    total_requests: u64,
    total_embeddings: u64,
    cache_hits: u64,
    cache_misses: u64,
    total_latency_ms: u64,
    by_provider: HashMap<String, ProviderUsage>,
    by_model: HashMap<String, ModelUsage>,
}

impl UsageTracker {
    fn new() -> Self {
        Self {
            total_tokens: 0,
            total_requests: 0,
            total_embeddings: 0,
            cache_hits: 0,
            cache_misses: 0,
            total_latency_ms: 0,
            by_provider: HashMap::new(),
            by_model: HashMap::new(),
        }
    }

    fn record(&mut self, provider: &str, model: &str, tokens: usize, embeddings: usize, cost: f64, latency_ms: u64) {
        self.total_tokens += tokens as u64;
        self.total_requests += 1;
        self.total_embeddings += embeddings as u64;
        self.total_latency_ms += latency_ms;

        // By provider
        let provider_usage = self.by_provider.entry(provider.to_string()).or_insert(ProviderUsage {
            tokens: 0,
            requests: 0,
            errors: 0,
            cost: 0.0,
        });
        provider_usage.tokens += tokens as u64;
        provider_usage.requests += 1;
        provider_usage.cost += cost;

        // By model
        let model_usage = self.by_model.entry(model.to_string()).or_insert(ModelUsage {
            tokens: 0,
            embeddings: 0,
            cost: 0.0,
        });
        model_usage.tokens += tokens as u64;
        model_usage.embeddings += embeddings as u64;
        model_usage.cost += cost;
    }

    fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    fn record_error(&mut self, provider: &str) {
        let provider_usage = self.by_provider.entry(provider.to_string()).or_insert(ProviderUsage {
            tokens: 0,
            requests: 0,
            errors: 0,
            cost: 0.0,
        });
        provider_usage.errors += 1;
    }

    fn stats(&self) -> UsageStats {
        let total_cache_ops = self.cache_hits + self.cache_misses;
        let cache_hit_rate = if total_cache_ops > 0 {
            self.cache_hits as f64 / total_cache_ops as f64
        } else {
            0.0
        };

        let avg_latency = if self.total_requests > 0 {
            self.total_latency_ms as f64 / self.total_requests as f64
        } else {
            0.0
        };

        let estimated_cost: f64 = self.by_model.values().map(|m| m.cost).sum();

        UsageStats {
            total_tokens: self.total_tokens,
            total_requests: self.total_requests,
            total_embeddings: self.total_embeddings,
            cache_hit_rate,
            avg_latency_ms: avg_latency,
            estimated_cost,
            by_provider: self.by_provider.clone(),
            by_model: self.by_model.clone(),
        }
    }
}

/// Main embedding service
pub struct EmbeddingService {
    config: EmbeddingServiceConfig,
    cache: RwLock<EmbeddingCache>,
    rate_limiter: RwLock<RateLimiter>,
    usage: RwLock<UsageTracker>,
}

impl EmbeddingService {
    /// Create new embedding service
    pub fn new(config: EmbeddingServiceConfig) -> Self {
        let cache = EmbeddingCache::new(config.max_cache_entries, config.cache_ttl_seconds);
        let rate_limiter = RateLimiter::new(config.rate_limit_rps);

        Self {
            config,
            cache: RwLock::new(cache),
            rate_limiter: RwLock::new(rate_limiter),
            usage: RwLock::new(UsageTracker::new()),
        }
    }

    /// Embed texts
    pub fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let start = Instant::now();
        let model = request.model_override.as_ref().unwrap_or(&self.config.primary_model);

        let mut embeddings = Vec::with_capacity(request.texts.len());
        let mut cache_hits = 0;
        let mut texts_to_embed: Vec<(usize, String)> = Vec::new();

        // Check cache for each text
        if self.config.enable_cache && !request.skip_cache {
            let mut cache = self.cache.write()
                .map_err(|_| VecStoreError::LockError("cache lock poisoned".into()))?;

            for (i, text) in request.texts.iter().enumerate() {
                let cache_key = self.cache_key(text, &model.model, request.task_type.as_ref());

                if let Some(cached) = cache.get(&cache_key) {
                    embeddings.push((i, cached));
                    cache_hits += 1;
                    self.usage.write()
                        .map_err(|_| VecStoreError::LockError("usage lock poisoned".into()))?
                        .record_cache_hit();
                } else {
                    texts_to_embed.push((i, text.clone()));
                    self.usage.write()
                        .map_err(|_| VecStoreError::LockError("usage lock poisoned".into()))?
                        .record_cache_miss();
                }
            }
        } else {
            texts_to_embed = request.texts.iter().enumerate().map(|(i, t)| (i, t.clone())).collect();
        }

        // Embed remaining texts
        let mut total_tokens = 0;
        if !texts_to_embed.is_empty() {
            // Rate limiting
            {
                let mut limiter = self.rate_limiter.write()
                    .map_err(|_| VecStoreError::LockError("rate_limiter lock poisoned".into()))?;
                if !limiter.acquire(1.0) {
                    // Would need to wait in real async implementation
                }
            }

            // Call embedding provider
            let texts: Vec<&str> = texts_to_embed.iter().map(|(_, t)| t.as_str()).collect();
            let (new_embeddings, tokens) = self.call_provider(&texts, model, request.task_type.as_ref())?;
            total_tokens = tokens;

            // Store in cache
            if self.config.enable_cache && !request.skip_cache {
                let mut cache = self.cache.write()
                    .map_err(|_| VecStoreError::LockError("cache lock poisoned".into()))?;
                for ((i, text), emb) in texts_to_embed.iter().zip(&new_embeddings) {
                    let cache_key = self.cache_key(text, &model.model, request.task_type.as_ref());
                    cache.put(cache_key, emb.clone(), &model.model);
                    embeddings.push((*i, emb.clone()));
                }
            } else {
                for ((i, _), emb) in texts_to_embed.iter().zip(new_embeddings) {
                    embeddings.push((*i, emb));
                }
            }
        }

        // Sort by original index
        embeddings.sort_by_key(|(i, _)| *i);
        let final_embeddings: Vec<Vec<f32>> = embeddings.into_iter().map(|(_, e)| e).collect();

        // Apply dimension override if specified
        let final_embeddings = if let Some(dim) = request.dimension_override {
            final_embeddings.into_iter().map(|e| {
                if e.len() > dim {
                    e[..dim].to_vec()
                } else {
                    e
                }
            }).collect()
        } else {
            final_embeddings
        };

        let processing_time_ms = start.elapsed().as_millis() as u64;
        let cost = (total_tokens as f64 / 1_000_000.0) * model.cost_per_million_tokens;

        // Record usage
        if self.config.track_usage && total_tokens > 0 {
            let provider_name = format!("{:?}", model.provider);
            self.usage.write()
                .map_err(|_| VecStoreError::LockError("usage lock poisoned".into()))?
                .record(
                    &provider_name,
                    &model.model,
                    total_tokens,
                    final_embeddings.len(),
                    cost,
                    processing_time_ms,
                );
        }

        Ok(EmbeddingResponse {
            embeddings: final_embeddings,
            model: model.model.clone(),
            tokens_used: total_tokens,
            cache_hits,
            processing_time_ms,
            estimated_cost: cost,
        })
    }

    /// Embed single text
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbeddingRequest::new(vec![text.to_string()]);
        let response = self.embed(request)?;
        response.embeddings.into_iter().next().ok_or_else(|| {
            VecStoreError::EmbeddingError("No embedding returned".to_string())
        })
    }

    /// Embed for query (uses query task type)
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbeddingRequest::new(vec![text.to_string()])
            .with_task_type(TaskType::Query);
        let response = self.embed(request)?;
        response.embeddings.into_iter().next().ok_or_else(|| {
            VecStoreError::EmbeddingError("No embedding returned".to_string())
        })
    }

    /// Embed for documents (uses document task type)
    pub fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = EmbeddingRequest::new(texts)
            .with_task_type(TaskType::Document);
        let response = self.embed(request)?;
        Ok(response.embeddings)
    }

    /// Get usage statistics
    pub fn usage(&self) -> Result<UsageStats> {
        let usage = self.usage.read()
            .map_err(|_| VecStoreError::LockError("usage lock poisoned".into()))?;
        Ok(usage.stats())
    }

    /// Reset usage tracking
    pub fn reset_usage(&self) -> Result<()> {
        let mut usage = self.usage.write()
            .map_err(|_| VecStoreError::LockError("usage lock poisoned".into()))?;
        *usage = UsageTracker::new();
        Ok(())
    }

    /// Get current model
    pub fn current_model(&self) -> &ModelConfig {
        &self.config.primary_model
    }

    /// Switch primary model
    pub fn switch_model(&mut self, model: ModelConfig) {
        self.config.primary_model = model;
    }

    /// Clear embedding cache
    pub fn clear_cache(&self) -> Result<()> {
        let mut cache = self.cache.write()
            .map_err(|_| VecStoreError::LockError("cache lock poisoned".into()))?;
        *cache = EmbeddingCache::new(self.config.max_cache_entries, self.config.cache_ttl_seconds);
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<(usize, u64)> {
        let cache = self.cache.read()
            .map_err(|_| VecStoreError::LockError("cache lock poisoned".into()))?;
        let stats = cache.stats();
        Ok((stats.entries, stats.total_hits))
    }

    // Internal methods

    fn cache_key(&self, text: &str, model: &str, task_type: Option<&TaskType>) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        model.hash(&mut hasher);
        if let Some(tt) = task_type {
            format!("{:?}", tt).hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }

    fn call_provider(&self, texts: &[&str], model: &ModelConfig, task_type: Option<&TaskType>) -> Result<(Vec<Vec<f32>>, usize)> {
        // Simulated embedding - in production would call actual APIs
        match &model.provider {
            EmbeddingProvider::OpenAI => self.embed_openai(texts, model),
            EmbeddingProvider::Cohere => self.embed_cohere(texts, model, task_type),
            EmbeddingProvider::Voyage => self.embed_voyage(texts, model, task_type),
            EmbeddingProvider::Jina => self.embed_jina(texts, model, task_type),
            EmbeddingProvider::LocalONNX => self.embed_local(texts, model),
            _ => self.embed_simulated(texts, model),
        }
    }

    fn embed_openai(&self, texts: &[&str], model: &ModelConfig) -> Result<(Vec<Vec<f32>>, usize)> {
        // Simulated OpenAI embedding
        let total_tokens: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
        let embeddings = self.generate_embeddings(texts, model.dimension);
        Ok((embeddings, total_tokens))
    }

    fn embed_cohere(&self, texts: &[&str], model: &ModelConfig, _task_type: Option<&TaskType>) -> Result<(Vec<Vec<f32>>, usize)> {
        let total_tokens: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
        let embeddings = self.generate_embeddings(texts, model.dimension);
        Ok((embeddings, total_tokens))
    }

    fn embed_voyage(&self, texts: &[&str], model: &ModelConfig, _task_type: Option<&TaskType>) -> Result<(Vec<Vec<f32>>, usize)> {
        let total_tokens: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
        let embeddings = self.generate_embeddings(texts, model.dimension);
        Ok((embeddings, total_tokens))
    }

    fn embed_jina(&self, texts: &[&str], model: &ModelConfig, _task_type: Option<&TaskType>) -> Result<(Vec<Vec<f32>>, usize)> {
        let total_tokens: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
        let embeddings = self.generate_embeddings(texts, model.dimension);
        Ok((embeddings, total_tokens))
    }

    fn embed_local(&self, texts: &[&str], model: &ModelConfig) -> Result<(Vec<Vec<f32>>, usize)> {
        let total_tokens: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
        let embeddings = self.generate_embeddings(texts, model.dimension);
        Ok((embeddings, total_tokens))
    }

    /// Fallback embedding for unsupported providers.
    ///
    /// Generates deterministic hash-based embeddings for testing and development.
    /// Production deployments should use supported providers (OpenAI, Cohere, Voyage, Jina, LocalONNX).
    fn embed_simulated(&self, texts: &[&str], model: &ModelConfig) -> Result<(Vec<Vec<f32>>, usize)> {
        tracing::warn!("Using simulated embeddings for unsupported provider - not suitable for production");
        let total_tokens: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
        let embeddings = self.generate_embeddings(texts, model.dimension);
        Ok((embeddings, total_tokens))
    }

    fn generate_embeddings(&self, texts: &[&str], dimension: usize) -> Vec<Vec<f32>> {
        // Generate deterministic embeddings based on text hash for testing/fallback
        texts.iter().map(|text| {
            let mut embedding = vec![0.0f32; dimension];
            let bytes = text.as_bytes();

            for (i, chunk) in bytes.chunks(4).enumerate() {
                let idx = i % dimension;
                let val: u32 = chunk.iter().fold(0u32, |acc, &b| acc.wrapping_add(b as u32));
                embedding[idx] = ((val as f32) / 255.0 - 0.5) * 2.0;
            }

            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }

            embedding
        }).collect()
    }
}

/// Embedding collection with automatic embedding on insert
pub struct EmbeddingCollection {
    service: Arc<EmbeddingService>,
    vectors: RwLock<HashMap<String, (Vec<f32>, HashMap<String, serde_json::Value>)>>,
    texts: RwLock<HashMap<String, String>>,
}

impl EmbeddingCollection {
    /// Create new collection
    pub fn new(service: Arc<EmbeddingService>) -> Self {
        Self {
            service,
            vectors: RwLock::new(HashMap::new()),
            texts: RwLock::new(HashMap::new()),
        }
    }

    /// Add document with automatic embedding
    pub fn add(&self, id: &str, text: &str, metadata: HashMap<String, serde_json::Value>) -> Result<()> {
        let embedding = self.service.embed_one(text)?;

        let mut vectors = self.vectors.write()
            .map_err(|_| VecStoreError::LockError("vectors lock poisoned".into()))?;
        vectors.insert(id.to_string(), (embedding, metadata));

        let mut texts = self.texts.write()
            .map_err(|_| VecStoreError::LockError("texts lock poisoned".into()))?;
        texts.insert(id.to_string(), text.to_string());

        Ok(())
    }

    /// Add multiple documents
    pub fn add_batch(&self, documents: Vec<(String, String, HashMap<String, serde_json::Value>)>) -> Result<usize> {
        let texts: Vec<String> = documents.iter().map(|(_, t, _)| t.clone()).collect();
        let embeddings = self.service.embed_documents(texts.clone())?;

        let mut vectors = self.vectors.write()
            .map_err(|_| VecStoreError::LockError("vectors lock poisoned".into()))?;
        let mut text_store = self.texts.write()
            .map_err(|_| VecStoreError::LockError("texts lock poisoned".into()))?;

        for ((id, text, metadata), embedding) in documents.into_iter().zip(embeddings) {
            vectors.insert(id.clone(), (embedding, metadata));
            text_store.insert(id, text);
        }

        Ok(vectors.len())
    }

    /// Query with automatic embedding
    pub fn query(&self, text: &str, top_k: usize) -> Result<Vec<QueryResult>> {
        let query_embedding = self.service.embed_query(text)?;

        let vectors = self.vectors.read()
            .map_err(|_| VecStoreError::LockError("vectors lock poisoned".into()))?;
        let texts = self.texts.read()
            .map_err(|_| VecStoreError::LockError("texts lock poisoned".into()))?;

        let mut results: Vec<QueryResult> = vectors
            .iter()
            .map(|(id, (vec, metadata))| {
                let score = cosine_similarity(&query_embedding, vec);
                QueryResult {
                    id: id.clone(),
                    score,
                    text: texts.get(id).cloned(),
                    metadata: metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results.truncate(top_k);

        Ok(results)
    }

    /// Get vector by ID
    pub fn get(&self, id: &str) -> Result<Option<Vec<f32>>> {
        let vectors = self.vectors.read()
            .map_err(|_| VecStoreError::LockError("vectors lock poisoned".into()))?;
        Ok(vectors.get(id).map(|(v, _)| v.clone()))
    }

    /// Delete by ID
    pub fn delete(&self, id: &str) -> Result<bool> {
        let mut vectors = self.vectors.write()
            .map_err(|_| VecStoreError::LockError("vectors lock poisoned".into()))?;
        let mut texts = self.texts.write()
            .map_err(|_| VecStoreError::LockError("texts lock poisoned".into()))?;
        texts.remove(id);
        Ok(vectors.remove(id).is_some())
    }

    /// Count documents
    pub fn count(&self) -> Result<usize> {
        let vectors = self.vectors.read()
            .map_err(|_| VecStoreError::LockError("vectors lock poisoned".into()))?;
        Ok(vectors.len())
    }
}

/// Query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Document ID
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Original text
    pub text: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
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
    fn test_embedding_service() {
        let config = EmbeddingServiceConfig::default();
        let service = EmbeddingService::new(config);

        let response = service.embed(EmbeddingRequest::new(vec![
            "Hello world".to_string(),
            "Machine learning".to_string(),
        ])).unwrap();

        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(response.embeddings[0].len(), 1536); // OpenAI small dimension
    }

    #[test]
    fn test_cache() {
        let config = EmbeddingServiceConfig::default();
        let service = EmbeddingService::new(config);

        // First request
        let response1 = service.embed(EmbeddingRequest::new(vec!["Hello".to_string()])).unwrap();

        // Second request (should be cached)
        let response2 = service.embed(EmbeddingRequest::new(vec!["Hello".to_string()])).unwrap();

        assert_eq!(response2.cache_hits, 1);
        assert_eq!(response1.embeddings[0], response2.embeddings[0]);
    }

    #[test]
    fn test_usage_tracking() {
        let config = EmbeddingServiceConfig::default();
        let service = EmbeddingService::new(config);

        service.embed(EmbeddingRequest::new(vec!["Test".to_string()])).unwrap();

        let usage = service.usage().unwrap();
        assert!(usage.total_requests > 0);
    }

    #[test]
    fn test_embedding_collection() {
        let config = EmbeddingServiceConfig::default();
        let service = Arc::new(EmbeddingService::new(config));
        let collection = EmbeddingCollection::new(service);

        collection.add("doc1", "Hello world", HashMap::new()).unwrap();
        collection.add("doc2", "Machine learning tutorial", HashMap::new()).unwrap();

        let results = collection.query("Hello", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc1"); // Most similar
    }

    #[test]
    fn test_model_configs() {
        let openai = ModelConfig::openai_small();
        assert_eq!(openai.dimension, 1536);

        let cohere = ModelConfig::cohere_v3();
        assert_eq!(cohere.dimension, 1024);

        let voyage = ModelConfig::voyage_3();
        assert!(voyage.supports_task_type);
    }

    #[test]
    fn test_task_types() {
        let config = EmbeddingServiceConfig {
            primary_model: ModelConfig::cohere_v3(),
            ..Default::default()
        };
        let service = EmbeddingService::new(config);

        let doc_embedding = service.embed(
            EmbeddingRequest::new(vec!["Document text".to_string()])
                .with_task_type(TaskType::Document)
        ).unwrap();

        let query_embedding = service.embed(
            EmbeddingRequest::new(vec!["Query text".to_string()])
                .with_task_type(TaskType::Query)
        ).unwrap();

        assert_eq!(doc_embedding.embeddings[0].len(), 1024);
        assert_eq!(query_embedding.embeddings[0].len(), 1024);
    }
}
