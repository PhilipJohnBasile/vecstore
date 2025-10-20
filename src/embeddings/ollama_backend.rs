//! Ollama Local LLM Integration
//!
//! Provides text embeddings using locally-running Ollama models.
//!
//! ## Supported Models
//!
//! - `nomic-embed-text` (768-dim) - Best general purpose
//! - `all-minilm` (384-dim) - Smaller, faster
//! - `mxbai-embed-large` (1024-dim) - Highest quality
//! - Any Ollama embedding model

use super::TextEmbedder;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Ollama embedding model
#[derive(Debug, Clone)]
pub enum OllamaModel {
    /// nomic-embed-text (768 dimensions)
    NomicEmbedText,
    /// all-minilm (384 dimensions)
    AllMiniLM,
    /// mxbai-embed-large (1024 dimensions)
    MxbaiEmbedLarge,
    /// Custom model by name
    Custom(String, usize),
}

impl OllamaModel {
    pub fn as_str(&self) -> &str {
        match self {
            OllamaModel::NomicEmbedText => "nomic-embed-text",
            OllamaModel::AllMiniLM => "all-minilm",
            OllamaModel::MxbaiEmbedLarge => "mxbai-embed-large",
            OllamaModel::Custom(name, _) => name.as_str(),
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            OllamaModel::NomicEmbedText => 768,
            OllamaModel::AllMiniLM => 384,
            OllamaModel::MxbaiEmbedLarge => 1024,
            OllamaModel::Custom(_, dim) => *dim,
        }
    }
}

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

/// Ollama local LLM embedder
///
/// # Example
///
/// ```no_run
/// use vecstore::embeddings::{OllamaEmbedding, OllamaModel, TextEmbedder};
///
/// # fn main() -> anyhow::Result<()> {
/// // Connect to local Ollama
/// let embedder = OllamaEmbedding::new("http://localhost:11434", OllamaModel::NomicEmbedText)?;
///
/// // Generate embeddings
/// let embedding = embedder.embed("Hello, world!")?;
/// assert_eq!(embedding.len(), 768);
/// # Ok(())
/// # }
/// ```
pub struct OllamaEmbedding {
    base_url: String,
    model: OllamaModel,
    client: reqwest::blocking::Client,
}

impl OllamaEmbedding {
    /// Create new Ollama embedder
    ///
    /// # Arguments
    /// * `base_url` - Ollama server URL (default: "http://localhost:11434")
    /// * `model` - Model to use
    pub fn new(base_url: impl Into<String>, model: OllamaModel) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            base_url: base_url.into(),
            model,
            client,
        })
    }

    /// Create with default localhost URL
    pub fn localhost(model: OllamaModel) -> Result<Self> {
        Self::new("http://localhost:11434", model)
    }

    /// Check if Ollama server is running
    pub fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        self.client.get(&url).send().is_ok()
    }

    /// List available models
    pub fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self.client.get(&url).send()?;

        #[derive(Deserialize)]
        struct TagsResponse {
            models: Vec<ModelInfo>,
        }

        #[derive(Deserialize)]
        struct ModelInfo {
            name: String,
        }

        let tags: TagsResponse = response.json()?;
        Ok(tags.models.into_iter().map(|m| m.name).collect())
    }

    /// Pull a model from Ollama registry
    pub fn pull_model(&self, model_name: &str) -> Result<()> {
        let url = format!("{}/api/pull", self.base_url);

        #[derive(Serialize)]
        struct PullRequest {
            name: String,
        }

        let request = PullRequest {
            name: model_name.to_string(),
        };

        let _response = self.client.post(&url).json(&request).send()?;

        Ok(())
    }

    fn embed_sync(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);

        let request = OllamaEmbedRequest {
            model: self.model.as_str().to_string(),
            prompt: text.to_string(),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .context("Failed to call Ollama API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .unwrap_or_else(|_| String::from("(no body)"));
            return Err(anyhow!("Ollama API error {}: {}", status, body));
        }

        let embed_response: OllamaEmbedResponse =
            response.json().context("Failed to parse Ollama response")?;

        Ok(embed_response.embedding)
    }

    pub fn model(&self) -> &OllamaModel {
        &self.model
    }
}

impl TextEmbedder for OllamaEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_sync(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Ollama doesn't have native batch API, so we process sequentially
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> Result<usize> {
        Ok(self.model.dimension())
    }
}

// Async version for Tokio applications
#[cfg(feature = "async")]
pub struct AsyncOllamaEmbedding {
    base_url: String,
    model: OllamaModel,
    client: reqwest::Client,
}

#[cfg(feature = "async")]
impl AsyncOllamaEmbedding {
    pub fn new(base_url: impl Into<String>, model: OllamaModel) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            base_url: base_url.into(),
            model,
            client,
        })
    }

    pub fn localhost(model: OllamaModel) -> Result<Self> {
        Self::new("http://localhost:11434", model)
    }

    pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);

        let request = OllamaEmbedRequest {
            model: self.model.as_str().to_string(),
            prompt: text.to_string(),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to call Ollama API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("(no body)"));
            return Err(anyhow!("Ollama API error {}: {}", status, body));
        }

        let embed_response: OllamaEmbedResponse = response
            .json()
            .await
            .context("Failed to parse Ollama response")?;

        Ok(embed_response.embedding)
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed_async(text).await?);
        }
        Ok(results)
    }

    pub async fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        self.client.get(&url).send().await.is_ok()
    }
}

#[cfg(feature = "async")]
impl TextEmbedder for AsyncOllamaEmbedding {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires Ollama running locally
    fn test_ollama_embedding() {
        let embedder = OllamaEmbedding::localhost(OllamaModel::NomicEmbedText).unwrap();

        if !embedder.is_available() {
            println!("Ollama not available, skipping test");
            return;
        }

        let embedding = embedder.embed("Hello, world!").unwrap();
        assert_eq!(embedding.len(), 768);

        // Check that embedding is normalized
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (magnitude - 1.0).abs() < 0.1,
            "Embedding should be normalized"
        );
    }

    #[test]
    #[ignore]
    fn test_ollama_batch() {
        let embedder = OllamaEmbedding::localhost(OllamaModel::AllMiniLM).unwrap();

        if !embedder.is_available() {
            return;
        }

        let texts = vec!["First text", "Second text", "Third text"];
        let embeddings = embedder.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 384);
        }
    }
}
