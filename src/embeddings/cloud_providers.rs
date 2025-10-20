// Cloud embedding providers
//
// Provides text embedding using multiple cloud APIs:
// - Cohere (embed-english-v3.0, embed-multilingual-v3.0)
// - Voyage AI (voyage-2, voyage-code-2, voyage-lite-02-instruct)
// - Mistral (mistral-embed)
// - Anthropic (Voyage via Anthropic)
// - Google Vertex AI (textembedding-gecko, text-embedding-004)
// - Azure OpenAI (text-embedding-ada-002, text-embedding-3-small/large)
// - HuggingFace Inference API (any model)
// - Jina AI (jina-embeddings-v2)
//
// Each provider implements the TextEmbedder trait for seamless integration.

use super::TextEmbedder;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// COHERE
// ============================================================================

/// Cohere embedding models
#[derive(Debug, Clone, Copy)]
pub enum CohereModel {
    /// embed-english-v3.0 (1024 dimensions, English only)
    EmbedEnglishV3,
    /// embed-multilingual-v3.0 (1024 dimensions, 100+ languages)
    EmbedMultilingualV3,
    /// embed-english-light-v3.0 (384 dimensions, faster)
    EmbedEnglishLightV3,
    /// embed-multilingual-light-v3.0 (384 dimensions, faster)
    EmbedMultilingualLightV3,
}

impl CohereModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            CohereModel::EmbedEnglishV3 => "embed-english-v3.0",
            CohereModel::EmbedMultilingualV3 => "embed-multilingual-v3.0",
            CohereModel::EmbedEnglishLightV3 => "embed-english-light-v3.0",
            CohereModel::EmbedMultilingualLightV3 => "embed-multilingual-light-v3.0",
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            CohereModel::EmbedEnglishV3 | CohereModel::EmbedMultilingualV3 => 1024,
            CohereModel::EmbedEnglishLightV3 | CohereModel::EmbedMultilingualLightV3 => 384,
        }
    }
}

#[derive(Debug, Serialize)]
struct CohereEmbedRequest {
    texts: Vec<String>,
    model: String,
    input_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CohereEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

/// Rate limiter for API calls
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

/// Cohere embedding backend
///
/// # Example
/// ```no_run
/// use vecstore::embeddings::cloud_providers::{CohereEmbedding, CohereModel};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let api_key = std::env::var("COHERE_API_KEY")?;
///     let embedder = CohereEmbedding::new(api_key, CohereModel::EmbedEnglishV3).await?;
///
///     let embedding = embedder.embed_async("Hello world").await?;
///     assert_eq!(embedding.len(), 1024);
///     Ok(())
/// }
/// ```
pub struct CohereEmbedding {
    client: reqwest::Client,
    api_key: String,
    model: CohereModel,
    rate_limiter: RateLimiter,
    max_retries: usize,
}

impl CohereEmbedding {
    pub async fn new(api_key: String, model: CohereModel) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            api_key,
            model,
            rate_limiter: RateLimiter::new(100), // Conservative default
            max_retries: 3,
        })
    }

    pub fn with_rate_limit(mut self, requests_per_minute: usize) -> Self {
        self.rate_limiter = RateLimiter::new(requests_per_minute);
        self
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch_async(&[text]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned"))
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Cohere API supports up to 96 texts per request
        const BATCH_SIZE: usize = 96;

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
            self.rate_limiter.wait_if_needed().await;

            let request = CohereEmbedRequest {
                texts: texts.iter().map(|s| s.to_string()).collect(),
                model: self.model.as_str().to_string(),
                input_type: "search_document".to_string(),
                truncate: Some("END".to_string()),
            };

            let response = self
                .client
                .post("https://api.cohere.ai/v1/embed")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let embed_response: CohereEmbedResponse = resp
                        .json()
                        .await
                        .context("Failed to parse Cohere response")?;

                    return Ok(embed_response.embeddings);
                }
                Ok(resp) if resp.status().as_u16() == 429 && retries < self.max_retries => {
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
                    return Err(anyhow!("Cohere API error {}: {}", status, body));
                }
                Err(_e) if retries < self.max_retries => {
                    retries += 1;
                    let wait_time = Duration::from_secs(2_u64.pow(retries as u32));
                    tokio::time::sleep(wait_time).await;
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Failed to call Cohere API: {}", e));
                }
            }
        }
    }

    pub fn model(&self) -> CohereModel {
        self.model
    }
}

impl TextEmbedder for CohereEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
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

// ============================================================================
// VOYAGE AI
// ============================================================================

/// Voyage AI embedding models
#[derive(Debug, Clone, Copy)]
pub enum VoyageModel {
    /// voyage-3 (1024 dimensions, latest general-purpose)
    Voyage3,
    /// voyage-3-lite (512 dimensions, faster)
    Voyage3Lite,
    /// voyage-code-2 (1536 dimensions, optimized for code)
    VoyageCode2,
    /// voyage-large-2 (1536 dimensions, high performance)
    VoyageLarge2,
}

impl VoyageModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            VoyageModel::Voyage3 => "voyage-3",
            VoyageModel::Voyage3Lite => "voyage-3-lite",
            VoyageModel::VoyageCode2 => "voyage-code-2",
            VoyageModel::VoyageLarge2 => "voyage-large-2",
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            VoyageModel::Voyage3 => 1024,
            VoyageModel::Voyage3Lite => 512,
            VoyageModel::VoyageCode2 => 1536,
            VoyageModel::VoyageLarge2 => 1536,
        }
    }
}

#[derive(Debug, Serialize)]
struct VoyageEmbedRequest {
    input: Vec<String>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VoyageEmbedResponse {
    data: Vec<VoyageEmbedData>,
}

#[derive(Debug, Deserialize)]
struct VoyageEmbedData {
    embedding: Vec<f32>,
    index: usize,
}

/// Voyage AI embedding backend
///
/// # Example
/// ```no_run
/// use vecstore::embeddings::cloud_providers::{VoyageEmbedding, VoyageModel};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let api_key = std::env::var("VOYAGE_API_KEY")?;
///     let embedder = VoyageEmbedding::new(api_key, VoyageModel::Voyage3).await?;
///
///     let embedding = embedder.embed_async("Hello world").await?;
///     assert_eq!(embedding.len(), 1024);
///     Ok(())
/// }
/// ```
pub struct VoyageEmbedding {
    client: reqwest::Client,
    api_key: String,
    model: VoyageModel,
    rate_limiter: RateLimiter,
    max_retries: usize,
}

impl VoyageEmbedding {
    pub async fn new(api_key: String, model: VoyageModel) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            api_key,
            model,
            rate_limiter: RateLimiter::new(300), // Conservative default
            max_retries: 3,
        })
    }

    pub fn with_rate_limit(mut self, requests_per_minute: usize) -> Self {
        self.rate_limiter = RateLimiter::new(requests_per_minute);
        self
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch_async(&[text]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned"))
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Voyage AI supports up to 128 texts per request
        const BATCH_SIZE: usize = 128;

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
            self.rate_limiter.wait_if_needed().await;

            let request = VoyageEmbedRequest {
                input: texts.iter().map(|s| s.to_string()).collect(),
                model: self.model.as_str().to_string(),
                input_type: Some("document".to_string()),
            };

            let response = self
                .client
                .post("https://api.voyageai.com/v1/embeddings")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let embed_response: VoyageEmbedResponse = resp
                        .json()
                        .await
                        .context("Failed to parse Voyage response")?;

                    // Sort by index to ensure correct order
                    let mut data = embed_response.data;
                    data.sort_by_key(|d| d.index);

                    return Ok(data.into_iter().map(|d| d.embedding).collect());
                }
                Ok(resp) if resp.status().as_u16() == 429 && retries < self.max_retries => {
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
                    return Err(anyhow!("Voyage API error {}: {}", status, body));
                }
                Err(_e) if retries < self.max_retries => {
                    retries += 1;
                    let wait_time = Duration::from_secs(2_u64.pow(retries as u32));
                    tokio::time::sleep(wait_time).await;
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Failed to call Voyage API: {}", e));
                }
            }
        }
    }

    pub fn model(&self) -> VoyageModel {
        self.model
    }
}

impl TextEmbedder for VoyageEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
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

// ============================================================================
// MISTRAL
// ============================================================================

/// Mistral embedding models
#[derive(Debug, Clone, Copy)]
pub enum MistralModel {
    /// mistral-embed (1024 dimensions)
    MistralEmbed,
}

impl MistralModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            MistralModel::MistralEmbed => "mistral-embed",
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            MistralModel::MistralEmbed => 1024,
        }
    }
}

#[derive(Debug, Serialize)]
struct MistralEmbedRequest {
    input: Vec<String>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MistralEmbedResponse {
    data: Vec<MistralEmbedData>,
}

#[derive(Debug, Deserialize)]
struct MistralEmbedData {
    embedding: Vec<f32>,
    index: usize,
}

/// Mistral AI embedding backend
///
/// # Example
/// ```no_run
/// use vecstore::embeddings::cloud_providers::{MistralEmbedding, MistralModel};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let api_key = std::env::var("MISTRAL_API_KEY")?;
///     let embedder = MistralEmbedding::new(api_key, MistralModel::MistralEmbed).await?;
///
///     let embedding = embedder.embed_async("Hello world").await?;
///     assert_eq!(embedding.len(), 1024);
///     Ok(())
/// }
/// ```
pub struct MistralEmbedding {
    client: reqwest::Client,
    api_key: String,
    model: MistralModel,
    rate_limiter: RateLimiter,
    max_retries: usize,
}

impl MistralEmbedding {
    pub async fn new(api_key: String, model: MistralModel) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            api_key,
            model,
            rate_limiter: RateLimiter::new(100), // Conservative default
            max_retries: 3,
        })
    }

    pub fn with_rate_limit(mut self, requests_per_minute: usize) -> Self {
        self.rate_limiter = RateLimiter::new(requests_per_minute);
        self
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch_async(&[text]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned"))
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Process in batches (conservative limit)
        const BATCH_SIZE: usize = 100;

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
            self.rate_limiter.wait_if_needed().await;

            let request = MistralEmbedRequest {
                input: texts.iter().map(|s| s.to_string()).collect(),
                model: self.model.as_str().to_string(),
                encoding_format: Some("float".to_string()),
            };

            let response = self
                .client
                .post("https://api.mistral.ai/v1/embeddings")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let embed_response: MistralEmbedResponse = resp
                        .json()
                        .await
                        .context("Failed to parse Mistral response")?;

                    // Sort by index to ensure correct order
                    let mut data = embed_response.data;
                    data.sort_by_key(|d| d.index);

                    return Ok(data.into_iter().map(|d| d.embedding).collect());
                }
                Ok(resp) if resp.status().as_u16() == 429 && retries < self.max_retries => {
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
                    return Err(anyhow!("Mistral API error {}: {}", status, body));
                }
                Err(_e) if retries < self.max_retries => {
                    retries += 1;
                    let wait_time = Duration::from_secs(2_u64.pow(retries as u32));
                    tokio::time::sleep(wait_time).await;
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Failed to call Mistral API: {}", e));
                }
            }
        }
    }

    pub fn model(&self) -> MistralModel {
        self.model
    }
}

impl TextEmbedder for MistralEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
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

// ============================================================================
// GOOGLE VERTEX AI
// ============================================================================

/// Google Vertex AI embedding models
#[derive(Debug, Clone, Copy)]
pub enum GoogleModel {
    /// textembedding-gecko@latest (768 dimensions)
    TextEmbeddingGecko,
    /// text-embedding-004 (768 dimensions, latest)
    TextEmbedding004,
    /// textembedding-gecko-multilingual@latest (768 dimensions)
    TextEmbeddingGeckoMultilingual,
}

impl GoogleModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            GoogleModel::TextEmbeddingGecko => "textembedding-gecko@latest",
            GoogleModel::TextEmbedding004 => "text-embedding-004",
            GoogleModel::TextEmbeddingGeckoMultilingual => {
                "textembedding-gecko-multilingual@latest"
            }
        }
    }

    pub fn dimension(&self) -> usize {
        768 // All Google models use 768 dimensions
    }
}

#[derive(Debug, Serialize)]
struct GoogleEmbedRequest {
    instances: Vec<GoogleInstance>,
}

#[derive(Debug, Serialize)]
struct GoogleInstance {
    content: String,
}

#[derive(Debug, Deserialize)]
struct GoogleEmbedResponse {
    predictions: Vec<GooglePrediction>,
}

#[derive(Debug, Deserialize)]
struct GooglePrediction {
    embeddings: GoogleEmbeddings,
}

#[derive(Debug, Deserialize)]
struct GoogleEmbeddings {
    values: Vec<f32>,
}

/// Google Vertex AI embedding client
pub struct GoogleEmbedding {
    project_id: String,
    location: String,
    model: GoogleModel,
    client: reqwest::Client,
    access_token: String,
    max_retries: usize,
}

impl GoogleEmbedding {
    /// Create new Google Vertex AI embedder
    ///
    /// # Arguments
    /// * `project_id` - Google Cloud project ID
    /// * `location` - GCP region (e.g., "us-central1")
    /// * `access_token` - OAuth2 access token (from gcloud auth print-access-token)
    /// * `model` - Model to use
    pub fn new(
        project_id: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
        model: GoogleModel,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            project_id: project_id.into(),
            location: location.into(),
            model,
            client,
            access_token: access_token.into(),
            max_retries: 3,
        })
    }

    pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch_async(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Empty response from Google"))
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let instances: Vec<GoogleInstance> = texts
            .iter()
            .map(|&text| GoogleInstance {
                content: text.to_string(),
            })
            .collect();

        let request = GoogleEmbedRequest { instances };

        let url = format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:predict",
            self.location, self.project_id, self.location, self.model.as_str()
        );

        let mut retries = 0;
        loop {
            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.access_token))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let embed_response: GoogleEmbedResponse = resp
                        .json()
                        .await
                        .context("Failed to parse Google response")?;

                    return Ok(embed_response
                        .predictions
                        .into_iter()
                        .map(|p| p.embeddings.values)
                        .collect());
                }
                Ok(resp) if resp.status().as_u16() == 429 && retries < self.max_retries => {
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
                    return Err(anyhow!("Google API error {}: {}", status, body));
                }
                Err(_e) if retries < self.max_retries => {
                    retries += 1;
                    let wait_time = Duration::from_secs(2_u64.pow(retries as u32));
                    tokio::time::sleep(wait_time).await;
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Failed to call Google API: {}", e));
                }
            }
        }
    }

    pub fn model(&self) -> GoogleModel {
        self.model
    }
}

impl TextEmbedder for GoogleEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
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
