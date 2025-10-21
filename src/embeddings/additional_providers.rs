// Additional cloud embedding providers
//
// Azure OpenAI, HuggingFace Inference API, Jina AI

use super::TextEmbedder;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ============================================================================
// AZURE OPENAI
// ============================================================================

/// Azure OpenAI embedding models
#[derive(Debug, Clone, Copy)]
pub enum AzureModel {
    /// text-embedding-ada-002 (1536 dimensions)
    TextEmbeddingAda002,
    /// text-embedding-3-small (1536 dimensions, improved)
    TextEmbedding3Small,
    /// text-embedding-3-large (3072 dimensions, best quality)
    TextEmbedding3Large,
}

impl AzureModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            AzureModel::TextEmbeddingAda002 => "text-embedding-ada-002",
            AzureModel::TextEmbedding3Small => "text-embedding-3-small",
            AzureModel::TextEmbedding3Large => "text-embedding-3-large",
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            AzureModel::TextEmbeddingAda002 | AzureModel::TextEmbedding3Small => 1536,
            AzureModel::TextEmbedding3Large => 3072,
        }
    }
}

#[derive(Debug, Serialize)]
struct AzureEmbedRequest {
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AzureEmbedResponse {
    data: Vec<AzureEmbedData>,
}

#[derive(Debug, Deserialize)]
struct AzureEmbedData {
    embedding: Vec<f32>,
    index: usize,
}

/// Azure OpenAI embedding client
pub struct AzureEmbedding {
    endpoint: String,
    deployment_name: String,
    api_key: String,
    api_version: String,
    model: AzureModel,
    client: reqwest::Client,
    max_retries: usize,
}

impl AzureEmbedding {
    /// Create new Azure OpenAI embedder
    ///
    /// # Arguments
    /// * `endpoint` - Azure OpenAI endpoint (e.g., "https://YOUR-RESOURCE.openai.azure.com")
    /// * `deployment_name` - Name of your deployment
    /// * `api_key` - Azure API key
    /// * `model` - Model to use
    pub fn new(
        endpoint: impl Into<String>,
        deployment_name: impl Into<String>,
        api_key: impl Into<String>,
        model: AzureModel,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            endpoint: endpoint.into(),
            deployment_name: deployment_name.into(),
            api_key: api_key.into(),
            api_version: "2023-05-15".to_string(),
            model,
            client,
            max_retries: 3,
        })
    }

    pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch_async(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Empty response from Azure"))
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let input: Vec<String> = texts.iter().map(|&s| s.to_string()).collect();
        let request = AzureEmbedRequest { input };

        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.endpoint, self.deployment_name, self.api_version
        );

        let mut retries = 0;
        loop {
            let response = self
                .client
                .post(&url)
                .header("api-key", &self.api_key)
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let embed_response: AzureEmbedResponse = resp
                        .json()
                        .await
                        .context("Failed to parse Azure response")?;

                    // Sort by index
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
                    return Err(anyhow!("Azure API error {}: {}", status, body));
                }
                Err(_e) if retries < self.max_retries => {
                    retries += 1;
                    let wait_time = Duration::from_secs(2_u64.pow(retries as u32));
                    tokio::time::sleep(wait_time).await;
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Failed to call Azure API: {}", e));
                }
            }
        }
    }

    pub fn model(&self) -> AzureModel {
        self.model
    }
}

impl TextEmbedder for AzureEmbedding {
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
// HUGGINGFACE INFERENCE API
// ============================================================================

/// HuggingFace Inference API embedding client
pub struct HuggingFaceEmbedding {
    model_id: String,
    api_key: String,
    dimension: usize,
    client: reqwest::Client,
    max_retries: usize,
}

impl HuggingFaceEmbedding {
    /// Create new HuggingFace embedder
    ///
    /// # Arguments
    /// * `model_id` - Model ID on HuggingFace (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    /// * `api_key` - HuggingFace API token
    /// * `dimension` - Expected embedding dimension
    pub fn new(
        model_id: impl Into<String>,
        api_key: impl Into<String>,
        dimension: usize,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            model_id: model_id.into(),
            api_key: api_key.into(),
            dimension,
            client,
            max_retries: 3,
        })
    }

    /// Create with common models (auto-detect dimension)
    pub fn from_model(model_name: &str, api_key: impl Into<String>) -> Result<Self> {
        let (model_id, dimension) = match model_name {
            "all-MiniLM-L6-v2" => ("sentence-transformers/all-MiniLM-L6-v2", 384),
            "all-mpnet-base-v2" => ("sentence-transformers/all-mpnet-base-v2", 768),
            "bge-small-en" => ("BAAI/bge-small-en", 384),
            "bge-base-en" => ("BAAI/bge-base-en", 768),
            "bge-large-en" => ("BAAI/bge-large-en", 1024),
            _ => return Err(anyhow!("Unknown model: {}", model_name)),
        };

        Self::new(model_id, api_key, dimension)
    }

    pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch_async(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Empty response from HuggingFace"))
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<String> = texts.iter().map(|&s| s.to_string()).collect();

        let url = format!(
            "https://api-inference.huggingface.co/models/{}",
            self.model_id
        );

        let mut retries = 0;
        loop {
            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({ "inputs": inputs }))
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let embeddings: Vec<Vec<f32>> = resp
                        .json()
                        .await
                        .context("Failed to parse HuggingFace response")?;

                    return Ok(embeddings);
                }
                Ok(resp) if resp.status().as_u16() == 429 && retries < self.max_retries => {
                    retries += 1;
                    let wait_time = Duration::from_secs(2_u64.pow(retries as u32));
                    tokio::time::sleep(wait_time).await;
                    continue;
                }
                Ok(resp) if resp.status().as_u16() == 503 && retries < self.max_retries => {
                    // Model loading, retry after longer wait
                    retries += 1;
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    continue;
                }
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| String::from("(no body)"));
                    return Err(anyhow!("HuggingFace API error {}: {}", status, body));
                }
                Err(_e) if retries < self.max_retries => {
                    retries += 1;
                    let wait_time = Duration::from_secs(2_u64.pow(retries as u32));
                    tokio::time::sleep(wait_time).await;
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Failed to call HuggingFace API: {}", e));
                }
            }
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl TextEmbedder for HuggingFaceEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let runtime = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
        runtime.block_on(self.embed_async(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let runtime = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
        runtime.block_on(self.embed_batch_async(texts))
    }

    fn dimension(&self) -> Result<usize> {
        Ok(self.dimension)
    }
}

// ============================================================================
// JINA AI
// ============================================================================

/// Jina AI embedding models
#[derive(Debug, Clone, Copy)]
pub enum JinaModel {
    /// jina-embeddings-v2-base-en (768 dimensions, 8192 tokens)
    EmbeddingsV2BaseEn,
    /// jina-embeddings-v2-small-en (512 dimensions, 8192 tokens)
    EmbeddingsV2SmallEn,
    /// jina-embeddings-v3 (1024 dimensions, multilingual)
    EmbeddingsV3,
}

impl JinaModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            JinaModel::EmbeddingsV2BaseEn => "jina-embeddings-v2-base-en",
            JinaModel::EmbeddingsV2SmallEn => "jina-embeddings-v2-small-en",
            JinaModel::EmbeddingsV3 => "jina-embeddings-v3",
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            JinaModel::EmbeddingsV2BaseEn => 768,
            JinaModel::EmbeddingsV2SmallEn => 512,
            JinaModel::EmbeddingsV3 => 1024,
        }
    }
}

#[derive(Debug, Serialize)]
struct JinaEmbedRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct JinaEmbedResponse {
    data: Vec<JinaEmbedData>,
}

#[derive(Debug, Deserialize)]
struct JinaEmbedData {
    embedding: Vec<f32>,
    index: usize,
}

/// Jina AI embedding client
pub struct JinaEmbedding {
    api_key: String,
    model: JinaModel,
    client: reqwest::Client,
    max_retries: usize,
}

impl JinaEmbedding {
    /// Create new Jina AI embedder
    ///
    /// # Arguments
    /// * `api_key` - Jina AI API key
    /// * `model` - Model to use
    pub fn new(api_key: impl Into<String>, model: JinaModel) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            api_key: api_key.into(),
            model,
            client,
            max_retries: 3,
        })
    }

    pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch_async(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Empty response from Jina"))
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let input: Vec<String> = texts.iter().map(|&s| s.to_string()).collect();
        let request = JinaEmbedRequest {
            input,
            model: self.model.as_str().to_string(),
        };

        let mut retries = 0;
        loop {
            let response = self
                .client
                .post("https://api.jina.ai/v1/embeddings")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let embed_response: JinaEmbedResponse =
                        resp.json().await.context("Failed to parse Jina response")?;

                    // Sort by index
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
                    return Err(anyhow!("Jina API error {}: {}", status, body));
                }
                Err(_e) if retries < self.max_retries => {
                    retries += 1;
                    let wait_time = Duration::from_secs(2_u64.pow(retries as u32));
                    tokio::time::sleep(wait_time).await;
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Failed to call Jina API: {}", e));
                }
            }
        }
    }

    pub fn model(&self) -> JinaModel {
        self.model
    }
}

impl TextEmbedder for JinaEmbedding {
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
