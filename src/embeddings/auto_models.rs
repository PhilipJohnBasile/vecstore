//! Auto-downloading embedding models from HuggingFace Hub
//!
//! This module provides "batteries included" embedding support by automatically
//! downloading pre-trained models from HuggingFace on first use.
//!
//! ## Features
//!
//! - Auto-download from HuggingFace Hub
//! - Local caching (~/.vecstore/models/)
//! - Progress bars for downloads
//! - Multiple pre-configured models
//! - Custom ONNX model support
//!
//! ## Usage
//!
//! ```no_run
//! use vecstore::embeddings::AutoEmbedder;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Downloads model on first use, cached afterwards
//! let embedder = AutoEmbedder::from_pretrained("all-MiniLM-L6-v2")?;
//!
//! let embedding = embedder.encode("Hello world")?;
//! println!("Dimension: {}", embedding.len());
//! # Ok(())
//! # }
//! ```

#![cfg(feature = "embeddings")]

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

use super::Embedder;

/// Pre-configured embedding models available for auto-download
#[derive(Debug, Clone, Copy)]
pub enum PretrainedModel {
    /// all-MiniLM-L6-v2 (384 dimensions, 80MB)
    /// Fast and efficient general-purpose model
    AllMiniLML6V2,

    /// all-mpnet-base-v2 (768 dimensions, 420MB)
    /// Higher quality, slower inference
    AllMpnetBaseV2,

    /// multilingual-e5-small (384 dimensions, 118MB)
    /// Supports 100+ languages
    MultilingualE5Small,

    /// all-MiniLM-L12-v2 (384 dimensions, 120MB)
    /// Balanced speed and quality
    AllMiniLML12V2,
}

impl PretrainedModel {
    /// Get the HuggingFace model identifier
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::AllMiniLML6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            Self::AllMpnetBaseV2 => "sentence-transformers/all-mpnet-base-v2",
            Self::MultilingualE5Small => "intfloat/multilingual-e5-small",
            Self::AllMiniLML12V2 => "sentence-transformers/all-MiniLM-L12-v2",
        }
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        match self {
            Self::AllMiniLML6V2 => 384,
            Self::AllMpnetBaseV2 => 768,
            Self::MultilingualE5Small => 384,
            Self::AllMiniLML12V2 => 384,
        }
    }

    /// Get approximate model size in MB
    pub fn size_mb(&self) -> usize {
        match self {
            Self::AllMiniLML6V2 => 80,
            Self::AllMpnetBaseV2 => 420,
            Self::MultilingualE5Small => 118,
            Self::AllMiniLML12V2 => 120,
        }
    }

    /// Parse model name from string
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_lowercase().as_str() {
            "all-minilm-l6-v2" | "minilm" => Ok(Self::AllMiniLML6V2),
            "all-mpnet-base-v2" | "mpnet" => Ok(Self::AllMpnetBaseV2),
            "multilingual-e5-small" | "e5-small" => Ok(Self::MultilingualE5Small),
            "all-minilm-l12-v2" | "minilm-l12" => Ok(Self::AllMiniLML12V2),
            _ => anyhow::bail!("Unknown model: {}", name),
        }
    }
}

/// Auto-downloading embedder that manages model cache
pub struct AutoEmbedder {
    embedder: Embedder,
    model_name: String,
    cache_dir: PathBuf,
}

impl AutoEmbedder {
    /// Load a pretrained model, downloading if necessary
    ///
    /// Models are cached in `~/.vecstore/models/` for future use.
    ///
    /// # Arguments
    /// * `model_name` - Name or identifier of the model
    ///
    /// # Supported Models
    /// - `"all-MiniLM-L6-v2"` - Fast, general purpose (384d, 80MB)
    /// - `"all-mpnet-base-v2"` - High quality (768d, 420MB)
    /// - `"multilingual-e5-small"` - Multilingual (384d, 118MB)
    /// - `"all-MiniLM-L12-v2"` - Balanced (384d, 120MB)
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::embeddings::AutoEmbedder;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let embedder = AutoEmbedder::from_pretrained("all-MiniLM-L6-v2")?;
    /// let embedding = embedder.encode("Hello world")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let cache_dir = Self::get_cache_dir()?;
        Self::from_pretrained_with_cache(model_name, cache_dir)
    }

    /// Load a pretrained model with custom cache directory
    pub fn from_pretrained_with_cache(
        model_name: &str,
        cache_dir: impl AsRef<Path>,
    ) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Parse model
        let model = PretrainedModel::from_name(model_name)?;

        // Get model files (download if needed)
        let model_dir = Self::ensure_model_cached(&cache_dir, model)?;

        // Load embedder
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let embedder =
            Embedder::new(model_path, tokenizer_path).context("Failed to load embedding model")?;

        Ok(Self {
            embedder,
            model_name: model_name.to_string(),
            cache_dir,
        })
    }

    /// Encode a single text to an embedding vector
    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        self.embedder.embed(text)
    }

    /// Encode multiple texts in a batch
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embedder.embed_batch(texts)
    }

    /// Get the model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get the cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get default cache directory (~/.vecstore/models/)
    fn get_cache_dir() -> Result<PathBuf> {
        let home = directories::UserDirs::new().context("Failed to get user home directory")?;

        let cache = home.home_dir().join(".vecstore").join("models");

        fs::create_dir_all(&cache).context("Failed to create cache directory")?;

        Ok(cache)
    }

    /// Ensure model is cached, downloading if necessary
    fn ensure_model_cached(cache_dir: &Path, model: PretrainedModel) -> Result<PathBuf> {
        let model_dir = cache_dir.join(model.model_id().replace('/', "_"));

        // Check if already cached
        if model_dir.exists()
            && model_dir.join("model.onnx").exists()
            && model_dir.join("tokenizer.json").exists()
        {
            println!("Using cached model: {}", model.model_id());
            return Ok(model_dir);
        }

        // Need to download
        println!(
            "Downloading model: {} (~{}MB)...",
            model.model_id(),
            model.size_mb()
        );

        fs::create_dir_all(&model_dir).context("Failed to create model directory")?;

        // Download model files
        Self::download_model_files(&model_dir, model)?;

        println!("Model downloaded and cached successfully!");

        Ok(model_dir)
    }

    /// Download model files from HuggingFace
    fn download_model_files(model_dir: &Path, model: PretrainedModel) -> Result<()> {
        let base_url = format!("https://huggingface.co/{}/resolve/main", model.model_id());

        // Files to download
        let files = vec![
            ("model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ];

        for (remote_name, local_name) in files {
            let url = format!("{}/{}", base_url, remote_name);
            let dest = model_dir.join(local_name);

            println!("  Downloading {}...", remote_name);

            // Simple download using ureq
            let response = ureq::get(&url)
                .call()
                .with_context(|| format!("Failed to download {}", url))?;

            let mut reader = response.into_reader();
            let mut file = fs::File::create(&dest)
                .with_context(|| format!("Failed to create file: {:?}", dest))?;

            std::io::copy(&mut reader, &mut file).context("Failed to write downloaded file")?;

            println!("  ✓ {} downloaded", remote_name);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_parsing() {
        assert!(PretrainedModel::from_name("all-MiniLM-L6-v2").is_ok());
        assert!(PretrainedModel::from_name("minilm").is_ok());
        assert!(PretrainedModel::from_name("mpnet").is_ok());
        assert!(PretrainedModel::from_name("unknown").is_err());
    }

    #[test]
    fn test_model_metadata() {
        let model = PretrainedModel::AllMiniLML6V2;
        assert_eq!(model.dimension(), 384);
        assert_eq!(model.size_mb(), 80);
        assert!(model.model_id().contains("MiniLM"));
    }

    #[test]
    fn test_cache_dir_creation() {
        let cache = AutoEmbedder::get_cache_dir().unwrap();
        assert!(cache.exists());
        assert!(cache.ends_with(".vecstore/models"));
    }

    // Integration test - requires network and ONNX Runtime
    #[test]
    #[ignore]
    fn test_auto_download_and_encode() {
        let embedder = AutoEmbedder::from_pretrained("all-MiniLM-L6-v2").unwrap();

        let embedding = embedder.encode("Hello world").unwrap();
        assert_eq!(embedding.len(), 384);

        let batch = embedder.encode_batch(&["test1", "test2"]).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].len(), 384);
    }
}
