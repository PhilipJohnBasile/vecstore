// OpenAI embeddings backend
//
// Provides text embedding using OpenAI's embedding API.
// Supports text-embedding-3-small, text-embedding-3-large, and ada-002 models.

use super::TextEmbedder;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// OpenAI embedding models
#[derive(Debug, Clone, Copy)]
pub enum OpenAIModel {
    /// text-embedding-3-small (1536 dimensions, $0.02/1M tokens)
    TextEmbedding3Small,
    /// text-embedding-3-large (3072 dimensions, $0.13/1M tokens)
    TextEmbedding3Large,
    /// text-embedding-ada-002 (1536 dimensions, legacy)
    Ada002,
}

impl OpenAIModel {
    /// Get the model name string
    pub fn as_str(&self) -> &'static str {
        match self {
            OpenAIModel::TextEmbedding3Small => "text-embedding-3-small",
            OpenAIModel::TextEmbedding3Large => "text-embedding-3-large",
            OpenAIModel::Ada002 => "text-embedding-ada-002",
        }
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        match self {
            OpenAIModel::TextEmbedding3Small => 1536,
            OpenAIModel::TextEmbedding3Large => 3072,
            OpenAIModel::Ada002 => 1536,
        }
    }

    /// Get the cost per 1M tokens in USD
    pub fn cost_per_million_tokens(&self) -> f64 {
        match self {
            OpenAIModel::TextEmbedding3Small => 0.02,
            OpenAIModel::TextEmbedding3Large => 0.13,
            OpenAIModel::Ada002 => 0.10,
        }
    }
}

/// OpenAI API request for embeddings
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

/// OpenAI API response for embeddings
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

/// Rate limiter for OpenAI API calls
struct RateLimiter {
    requests_per_minute: usize,
    last_requests: Arc<std::sync::Mutex<Vec<std::time::Instant>>>,
}

impl RateLimiter {
    fn new(requests_per_minute: usize) -> Self {
        Self {
            requests_per_minute,
            last_requests: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    async fn wait_if_needed(&self) {
        let mut requests = self.last_requests.lock().unwrap();

        // Remove requests older than 1 minute
        let now = std::time::Instant::now();
        requests.retain(|&time| now.duration_since(time) < Duration::from_secs(60));

        // If at limit, wait
        if requests.len() >= self.requests_per_minute {
            if let Some(&oldest) = requests.first() {
                let wait_time = Duration::from_secs(60)
                    .checked_sub(now.duration_since(oldest))
                    .unwrap_or(Duration::from_secs(0));

                if wait_time > Duration::from_secs(0) {
                    drop(requests); // Release lock before sleeping
                    tokio::time::sleep(wait_time).await;
                    requests = self.last_requests.lock().unwrap();
                }
            }
        }

        requests.push(now);
    }
}

/// OpenAI embedding backend
///
/// Provides text embeddings using OpenAI's API with rate limiting,
/// retry logic, and batch processing.
///
/// # Example
/// ```no_run
/// use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let api_key = std::env::var("OPENAI_API_KEY")?;
///     let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small).await?;
///
///     let embedding = embedder.embed_async("Hello world").await?;
///     assert_eq!(embedding.len(), 1536);
///     Ok(())
/// }
/// ```
pub struct OpenAIEmbedding {
    client: reqwest::Client,
    api_key: String,
    model: OpenAIModel,
    rate_limiter: RateLimiter,
    max_retries: usize,
}

impl OpenAIEmbedding {
    /// Create a new OpenAI embedding backend
    ///
    /// # Arguments
    /// * `api_key` - OpenAI API key
    /// * `model` - Model to use for embeddings
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let api_key = std::env::var("OPENAI_API_KEY")?;
    ///     let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn new(api_key: String, model: OpenAIModel) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            api_key,
            model,
            rate_limiter: RateLimiter::new(500), // 500 requests per minute (conservative)
            max_retries: 3,
        })
    }

    /// Configure rate limiting
    pub fn with_rate_limit(mut self, requests_per_minute: usize) -> Self {
        self.rate_limiter = RateLimiter::new(requests_per_minute);
        self
    }

    /// Configure max retries
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Embed a single text asynchronously
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// # let api_key = std::env::var("OPENAI_API_KEY")?;
    /// # let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small).await?;
    /// let embedding = embedder.embed_async("Hello world").await?;
    /// assert_eq!(embedding.len(), 1536);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch_async(&[text]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned"))
    }

    /// Embed multiple texts in batch asynchronously
    ///
    /// OpenAI API supports up to 2048 texts per request.
    /// This method automatically chunks larger batches.
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// # let api_key = std::env::var("OPENAI_API_KEY")?;
    /// # let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small).await?;
    /// let texts = vec!["Hello", "World", "Test"];
    /// let embeddings = embedder.embed_batch_async(&texts).await?;
    /// assert_eq!(embeddings.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // OpenAI API limit is 2048 texts per request
        const BATCH_SIZE: usize = 2048;

        let mut all_embeddings = Vec::new();

        for chunk in texts.chunks(BATCH_SIZE) {
            let chunk_embeddings = self.embed_batch_chunk(chunk).await?;
            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
    }

    async fn embed_batch_chunk(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut retries = 0;

        loop {
            // Wait for rate limiter
            self.rate_limiter.wait_if_needed().await;

            let request = EmbeddingRequest {
                model: self.model.as_str().to_string(),
                input: texts.iter().map(|s| s.to_string()).collect(),
                dimensions: None,
            };

            let response = self
                .client
                .post("https://api.openai.com/v1/embeddings")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let embedding_response: EmbeddingResponse = resp
                        .json()
                        .await
                        .context("Failed to parse OpenAI response")?;

                    // Sort by index to ensure correct order
                    let mut data = embedding_response.data;
                    data.sort_by_key(|d| d.index);

                    return Ok(data.into_iter().map(|d| d.embedding).collect());
                }
                Ok(resp) if resp.status().as_u16() == 429 && retries < self.max_retries => {
                    // Rate limit hit, retry with exponential backoff
                    retries += 1;
                    let wait_time = Duration::from_secs(2_u64.pow(retries as u32));
                    tokio::time::sleep(wait_time).await;
                    continue;
                }
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| String::from("(no body)"));
                    return Err(anyhow!("OpenAI API error {}: {}", status, body));
                }
                Err(_e) if retries < self.max_retries => {
                    // Network error, retry
                    retries += 1;
                    let wait_time = Duration::from_secs(2_u64.pow(retries as u32));
                    tokio::time::sleep(wait_time).await;
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Failed to call OpenAI API: {}", e));
                }
            }
        }
    }

    /// Estimate the cost for embedding given texts
    ///
    /// This is a rough estimate based on character count.
    /// Actual cost depends on token count which requires tokenization.
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// # let api_key = std::env::var("OPENAI_API_KEY")?;
    /// # let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small).await?;
    /// let texts = vec!["Hello world", "This is a test"];
    /// let estimated_cost = embedder.estimate_cost(&texts);
    /// println!("Estimated cost: ${:.6}", estimated_cost);
    /// # Ok(())
    /// # }
    /// ```
    pub fn estimate_cost(&self, texts: &[&str]) -> f64 {
        // Rough estimate: 1 token ≈ 4 characters
        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        let estimated_tokens = total_chars / 4;
        let cost_per_token = self.model.cost_per_million_tokens() / 1_000_000.0;
        estimated_tokens as f64 * cost_per_token
    }

    /// Get the model being used
    pub fn model(&self) -> OpenAIModel {
        self.model
    }
}

// Implement TextEmbedder for OpenAIEmbedding (synchronous wrapper)
// Note: This blocks the current thread. Prefer using embed_async when possible.
impl TextEmbedder for OpenAIEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Use tokio runtime to run async code
        let runtime = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
        runtime.block_on(self.embed_async(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let runtime = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
        runtime.block_on(self.embed_batch_async(texts))
    }

    fn dimension(&self) -> Result<usize> {
        Ok(self.model.dimension())
    }
}
