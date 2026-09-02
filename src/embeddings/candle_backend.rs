//! Pure Rust embeddings using Candle
//!
//! This module provides embedding generation using the Candle ML framework,
//! enabling fully native Rust embeddings without Python dependencies.

use crate::embeddings::TextEmbedder;
use anyhow::{Context, Result, anyhow};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

/// Supported Candle embedding models
#[derive(Debug, Clone)]
pub enum CandleModel {
    /// all-MiniLM-L6-v2 (22M params, 384-dim)
    AllMiniLML6V2,
    /// BAAI/bge-small-en (33M params, 384-dim)
    BgeSmallEn,
    /// Custom model from HuggingFace Hub
    Custom { repo_id: String, revision: String },
}

impl CandleModel {
    fn repo_id(&self) -> &str {
        match self {
            CandleModel::AllMiniLML6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            CandleModel::BgeSmallEn => "BAAI/bge-small-en",
            CandleModel::Custom { repo_id, .. } => repo_id,
        }
    }
}

/// Candle-based text embedder
///
/// Provides pure Rust embedding generation using the Candle ML framework.
/// No Python dependencies required.
///
/// # Example
/// ```no_run
/// use vecstore::CandleEmbedder;
/// use vecstore::embeddings::candle_backend::CandleModel;
///
/// # fn main() -> anyhow::Result<()> {
/// let embedder = CandleEmbedder::new(CandleModel::AllMiniLML6V2)?;
///
/// let embedding = embedder.embed("Hello, world!")?;
/// assert_eq!(embedding.len(), 384);  // all-MiniLM-L6-v2 outputs 384-dim
/// # Ok(())
/// # }
/// ```
pub struct CandleEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    normalize: bool,
    /// Hidden size (embedding dimension) from model config
    hidden_size: usize,
}

impl CandleEmbedder {
    /// Create a new Candle embedder
    ///
    /// Downloads the model from HuggingFace Hub if not cached locally.
    ///
    /// # Arguments
    /// * `model` - Which model to use
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::CandleEmbedder;
    /// use vecstore::embeddings::candle_backend::CandleModel;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// // Use all-MiniLM-L6-v2 (22M params, fast)
    /// let embedder = CandleEmbedder::new(CandleModel::AllMiniLML6V2)?;
    ///
    /// // Or use BGE-small (33M params, better quality)
    /// let embedder = CandleEmbedder::new(CandleModel::BgeSmallEn)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model: CandleModel) -> Result<Self> {
        let device = Device::Cpu; // Use CPU for now, GPU support can be added later

        // Download model from HuggingFace Hub
        let api = Api::new().context("Failed to create HuggingFace API")?;
        let repo = Repo::new(model.repo_id().to_string(), RepoType::Model);
        let repo = api.repo(repo);

        // Load tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model config
        let config_path = repo
            .get("config.json")
            .context("Failed to download config")?;
        let config: Config = serde_json::from_reader(std::fs::File::open(config_path)?)
            .context("Failed to parse config")?;

        // Load model weights
        let weights_path = repo
            .get("pytorch_model.bin")
            .or_else(|_| repo.get("model.safetensors"))
            .context("Failed to download model weights")?;

        let vb = if weights_path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? }
        } else {
            VarBuilder::from_pth(&weights_path, DTYPE, &device)?
        };

        let hidden_size = config.hidden_size;
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            normalize: true, // Normalize embeddings by default (common for sentence transformers)
            hidden_size,
        })
    }

    /// Set whether to normalize embeddings (default: true)
    pub fn set_normalize(&mut self, normalize: bool) {
        self.normalize = normalize;
    }

    /// Mean pooling strategy for sentence embeddings
    ///
    /// Takes the mean of all token embeddings to get a single sentence embedding
    fn mean_pool(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (_batch_size, seq_len, hidden_size) = embeddings.dims3()?;

        // Expand attention mask to match embeddings shape
        let mask = attention_mask.unsqueeze(2)?;
        let mask = mask.expand(&[1, seq_len, hidden_size])?;

        // Zero out padding tokens
        let masked = embeddings.mul(&mask)?;

        // Sum over sequence length
        let sum = masked.sum(1)?;

        // Count non-padding tokens
        let mask_sum = mask.sum(1)?;

        // Divide to get mean
        let mean = sum.broadcast_div(&mask_sum)?;

        Ok(mean)
    }

    /// Normalize embeddings to unit length
    fn normalize_embeddings(&self, embeddings: &Tensor) -> Result<Tensor> {
        let norm = embeddings.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = embeddings.broadcast_div(&norm)?;
        Ok(normalized)
    }
}

impl TextEmbedder for CandleEmbedder {
    fn dimension(&self) -> Result<usize> {
        // Return the hidden size from the loaded model config
        Ok(self.hidden_size)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize input
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Convert to tensors
        let input_ids = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(attention_mask, &self.device)?.unsqueeze(0)?;

        // Run model
        let embeddings = self.model.forward(&input_ids, &attention_mask, None)?;

        // Mean pooling
        let pooled = self.mean_pool(&embeddings, &attention_mask)?;

        // Normalize if requested
        let final_embedding = if self.normalize {
            self.normalize_embeddings(&pooled)?
        } else {
            pooled
        };

        // Convert to Vec<f32>
        let embedding_vec = final_embedding.squeeze(0)?.to_vec1::<f32>()?;

        Ok(embedding_vec)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // For small batches, sequential processing avoids padding overhead
        if texts.len() <= 4 {
            return texts.iter().map(|text| self.embed(text)).collect();
        }

        // Tokenize all texts
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(*text, true)
                    .map_err(|e| anyhow!("Tokenization failed: {}", e))
            })
            .collect::<Result<Vec<_>>>()?;

        // Find max sequence length for padding
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Pad sequences and create batch tensors
        let batch_size = encodings.len();
        let mut input_ids_batch = vec![0u32; batch_size * max_len];
        let mut attention_mask_batch = vec![0u32; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let seq_len = ids.len();

            // Copy ids and mask, pad the rest with zeros
            for j in 0..seq_len {
                input_ids_batch[i * max_len + j] = ids[j];
                attention_mask_batch[i * max_len + j] = mask[j];
            }
            // Remaining positions are already zero (padding)
        }

        // Convert to tensors with shape [batch_size, max_len]
        let input_ids = Tensor::from_vec(input_ids_batch, (batch_size, max_len), &self.device)?;
        let attention_mask =
            Tensor::from_vec(attention_mask_batch, (batch_size, max_len), &self.device)?;

        // Run model on batch
        let embeddings = self.model.forward(&input_ids, &attention_mask, None)?;

        // Mean pooling for each item in batch
        let pooled = self.batch_mean_pool(&embeddings, &attention_mask)?;

        // Normalize if requested
        let final_embeddings = if self.normalize {
            self.normalize_embeddings(&pooled)?
        } else {
            pooled
        };

        // Extract individual embeddings
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let embedding = final_embeddings.get(i)?.to_vec1::<f32>()?;
            results.push(embedding);
        }

        Ok(results)
    }
}

impl CandleEmbedder {
    /// Batch mean pooling - operates on batched embeddings
    fn batch_mean_pool(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = embeddings.dims3()?;

        // Expand attention mask to match embeddings shape [batch_size, seq_len, hidden_size]
        let mask = attention_mask.unsqueeze(2)?;
        let mask = mask.expand(&[batch_size, seq_len, hidden_size])?;

        // Zero out padding tokens
        let masked = embeddings.mul(&mask)?;

        // Sum over sequence length dimension
        let sum = masked.sum(1)?;

        // Count non-padding tokens
        let mask_sum = mask.sum(1)?;

        // Divide to get mean
        let mean = sum.broadcast_div(&mask_sum)?;

        Ok(mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download, run with --ignored
    fn test_candle_embedder() {
        let embedder = CandleEmbedder::new(CandleModel::AllMiniLML6V2).unwrap();

        let text = "This is a test sentence.";
        let embedding = embedder.embed(text).unwrap();

        // all-MiniLM-L6-v2 outputs 384-dimensional embeddings
        assert_eq!(embedding.len(), 384);

        // Check that embedding is normalized (norm should be ~1.0)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");
    }

    #[test]
    #[ignore] // Requires model download
    fn test_batch_embed() {
        let embedder = CandleEmbedder::new(CandleModel::AllMiniLML6V2).unwrap();

        let texts = vec!["First sentence", "Second sentence", "Third sentence"];
        let embeddings = embedder.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 384);
    }
}
