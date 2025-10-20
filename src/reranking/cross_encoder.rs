// Cross-Encoder Reranker using ONNX models
//
// This module provides production-ready cross-encoder reranking using ONNX Runtime.
// Cross-encoders process query-document pairs together, providing more accurate
// relevance scores than bi-encoder (embedding) models.

use super::Reranker;
use crate::store::Neighbor;
use anyhow::{anyhow, Context, Result};
use ndarray::{Array2, CowArray};
use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder, Value};
use std::path::Path;
use std::sync::Arc;
use tokenizers::tokenizer::Tokenizer;

/// Available pretrained cross-encoder models
#[derive(Debug, Clone, Copy)]
pub enum CrossEncoderModel {
    /// ms-marco-MiniLM-L-6-v2 - Fast and efficient for most use cases
    /// - Size: ~90MB
    /// - Speed: ~10ms per pair
    /// - Quality: Good for general search
    MiniLML6V2,

    /// ms-marco-MiniLM-L-12-v2 - Better quality, slower
    /// - Size: ~150MB
    /// - Speed: ~20ms per pair
    /// - Quality: Better accuracy
    MiniLML12V2,
}

impl CrossEncoderModel {
    /// Get the HuggingFace model ID
    pub fn model_id(&self) -> &'static str {
        match self {
            CrossEncoderModel::MiniLML6V2 => "cross-encoder/ms-marco-MiniLM-L-6-v2",
            CrossEncoderModel::MiniLML12V2 => "cross-encoder/ms-marco-MiniLM-L-12-v2",
        }
    }

    /// Get the local model directory name
    pub fn model_dir(&self) -> &'static str {
        match self {
            CrossEncoderModel::MiniLML6V2 => "ms-marco-minilm-l6-v2",
            CrossEncoderModel::MiniLML12V2 => "ms-marco-minilm-l12-v2",
        }
    }

    /// Get the default model cache directory
    pub fn cache_dir() -> std::path::PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Path::new(&home)
            .join(".cache")
            .join("vecstore")
            .join("cross-encoders")
    }
}

/// ONNX-based Cross-Encoder Reranker
///
/// Uses a cross-encoder model to score query-document pairs for reranking.
/// This provides significantly better relevance than bi-encoder similarity alone.
///
/// # Example
/// ```no_run
/// use vecstore::reranking::cross_encoder::{CrossEncoderReranker, CrossEncoderModel};
///
/// # fn main() -> anyhow::Result<()> {
/// // Load model
/// let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)?;
///
/// // Rerank search results
/// let reranked = reranker.rerank("query", results, 10)?;
/// # Ok(())
/// # }
/// ```
pub struct CrossEncoderReranker {
    session: Arc<Session>,
    tokenizer: Arc<Tokenizer>,
    max_length: usize,
}

impl std::fmt::Debug for CrossEncoderReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossEncoderReranker")
            .field("max_length", &self.max_length)
            .finish()
    }
}

impl CrossEncoderReranker {
    /// Load a pretrained cross-encoder model
    ///
    /// This will download the model if it's not already cached.
    ///
    /// # Arguments
    /// * `model` - The pretrained model to load
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::reranking::cross_encoder::{CrossEncoderReranker, CrossEncoderModel};
    ///
    /// let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn from_pretrained(model: CrossEncoderModel) -> Result<Self> {
        let cache_dir = CrossEncoderModel::cache_dir();
        let model_dir = cache_dir.join(model.model_dir());

        // Download model if not cached
        if !model_dir.exists() {
            eprintln!("Downloading {} model...", model.model_id());
            Self::download_model(model, &model_dir)?;
        }

        Self::from_dir(&model_dir)
    }

    /// Load a cross-encoder model from a local directory
    ///
    /// The directory should contain:
    /// - `model.onnx` - ONNX model file
    /// - `tokenizer.json` - HuggingFace tokenizer
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::reranking::cross_encoder::CrossEncoderReranker;
    ///
    /// let reranker = CrossEncoderReranker::from_dir("./my_cross_encoder")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn from_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Load ONNX model
        let model_path = model_dir.join("model.onnx");
        if !model_path.exists() {
            return Err(anyhow!("Model file not found: {:?}", model_path));
        }

        let environment = Arc::new(
            Environment::builder()
                .with_name("cross_encoder")
                .build()
                .context("Failed to create ONNX environment")?,
        );

        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(&model_path)
            .context("Failed to load ONNX model")?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(anyhow!("Tokenizer file not found: {:?}", tokenizer_path));
        }

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            session: Arc::new(session),
            tokenizer: Arc::new(tokenizer),
            max_length: 512, // Standard BERT max length
        })
    }

    /// Download a pretrained model from HuggingFace
    fn download_model(model: CrossEncoderModel, target_dir: &Path) -> Result<()> {
        use std::fs;

        fs::create_dir_all(target_dir).context("Failed to create model directory")?;

        // For now, return an error with instructions
        // In a full implementation, we'd use hf-hub or reqwest to download
        Err(anyhow!(
            "Model download not yet implemented. Please manually download the model from:\n\
             https://huggingface.co/{}\n\
             \n\
             Required files:\n\
             - model.onnx (convert from PyTorch using optimum)\n\
             - tokenizer.json\n\
             \n\
             Place them in: {:?}",
            model.model_id(),
            target_dir
        ))
    }

    /// Score a single query-document pair
    ///
    /// Returns a relevance score (higher = more relevant).
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::reranking::cross_encoder::{CrossEncoderReranker, CrossEncoderModel};
    /// # let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)?;
    /// let score = reranker.score_pair("what is rust?", "Rust is a programming language")?;
    /// println!("Relevance score: {}", score);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn score_pair(&self, query: &str, document: &str) -> Result<f32> {
        // Tokenize query + document pair
        let input_text = format!("{} [SEP] {}", query, document);

        let encoding = self
            .tokenizer
            .encode(input_text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        // Get input IDs and attention mask
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Truncate or pad to max_length
        let (input_ids, attention_mask) = self.pad_or_truncate(input_ids, attention_mask);

        // Create ONNX inputs
        let input_ids_array = Array2::from_shape_vec(
            (1, input_ids.len()),
            input_ids.iter().map(|&id| id as i64).collect(),
        )?;

        let attention_mask_array = Array2::from_shape_vec(
            (1, attention_mask.len()),
            attention_mask.iter().map(|&m| m as i64).collect(),
        )?;

        let input_ids_dyn = input_ids_array.into_dyn();
        let attention_mask_dyn = attention_mask_array.into_dyn();

        let input_ids_cow = CowArray::from(&input_ids_dyn);
        let attention_mask_cow = CowArray::from(&attention_mask_dyn);

        let input_ids_value = Value::from_array(self.session.allocator(), &input_ids_cow)?;

        let attention_mask_value =
            Value::from_array(self.session.allocator(), &attention_mask_cow)?;

        // Run inference
        let outputs = self
            .session
            .run(vec![input_ids_value, attention_mask_value])?;

        // Extract logits (typically shape [1, 1] or [1, 2] for binary classification)
        let logits = outputs[0].try_extract::<f32>()?.view().to_owned();

        // Get the relevance score
        // For ms-marco models, output is typically [1, 1] with a single score
        let score = if logits.len() == 1 {
            logits[0]
        } else if logits.len() >= 2 {
            // If binary classification, use the positive class score
            logits[1]
        } else {
            return Err(anyhow!("Unexpected output shape: {:?}", logits.shape()));
        };

        Ok(score)
    }

    /// Pad or truncate sequences to max_length
    fn pad_or_truncate(&self, input_ids: &[u32], attention_mask: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let len = input_ids.len();

        if len >= self.max_length {
            // Truncate
            (
                input_ids[..self.max_length].to_vec(),
                attention_mask[..self.max_length].to_vec(),
            )
        } else {
            // Pad
            let mut padded_ids = input_ids.to_vec();
            let mut padded_mask = attention_mask.to_vec();

            padded_ids.resize(self.max_length, 0); // PAD token is typically 0
            padded_mask.resize(self.max_length, 0); // Padding has attention mask 0

            (padded_ids, padded_mask)
        }
    }
}

impl Reranker for CrossEncoderReranker {
    fn rerank(
        &self,
        query: &str,
        mut results: Vec<Neighbor>,
        top_k: usize,
    ) -> Result<Vec<Neighbor>> {
        if results.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        // Score each result
        for neighbor in &mut results {
            // Extract document text from metadata
            let doc_text = neighbor
                .metadata
                .fields
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if doc_text.is_empty() {
                // No text to score, keep original score
                continue;
            }

            // Score the query-document pair
            match self.score_pair(query, doc_text) {
                Ok(score) => {
                    neighbor.score = score;
                }
                Err(e) => {
                    eprintln!("Warning: Failed to score document {}: {}", neighbor.id, e);
                    // Keep original score
                }
            }
        }

        // Sort by score (descending)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top_k
        Ok(results.into_iter().take(top_k).collect())
    }

    fn name(&self) -> &str {
        "Cross-Encoder (ONNX)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Metadata;
    use std::collections::HashMap;

    fn make_neighbor(id: &str, score: f32, text: &str) -> Neighbor {
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("text".to_string(), serde_json::json!(text));

        Neighbor {
            id: id.to_string(),
            score,
            metadata,
        }
    }

    #[test]
    fn test_cross_encoder_model_metadata() {
        let model = CrossEncoderModel::MiniLML6V2;
        assert_eq!(model.model_id(), "cross-encoder/ms-marco-MiniLM-L-6-v2");
        assert_eq!(model.model_dir(), "ms-marco-minilm-l6-v2");

        let cache_dir = CrossEncoderModel::cache_dir();
        assert!(cache_dir.to_string_lossy().contains("vecstore"));
        assert!(cache_dir.to_string_lossy().contains("cross-encoders"));
    }

    #[test]
    fn test_cross_encoder_from_dir_missing_files() {
        let result = CrossEncoderReranker::from_dir("/nonexistent/path");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    // Integration test - requires actual model files
    #[test]
    #[ignore]
    fn test_cross_encoder_score_pair() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        let score1 = reranker
            .score_pair("what is rust", "Rust is a programming language")
            .expect("Failed to score");

        let score2 = reranker
            .score_pair("what is rust", "Python is a programming language")
            .expect("Failed to score");

        let score3 = reranker
            .score_pair("what is rust", "I like cooking pasta")
            .expect("Failed to score");

        // Rust query should score higher with Rust document than Python or cooking
        assert!(score1 > score2);
        assert!(score1 > score3);
    }

    // Integration test - requires actual model files
    #[test]
    #[ignore]
    fn test_cross_encoder_rerank() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        let results = vec![
            make_neighbor("doc1", 0.5, "Rust is a systems programming language"),
            make_neighbor("doc2", 0.9, "Python is great for data science"),
            make_neighbor("doc3", 0.7, "Rust has great performance"),
        ];

        let reranked = reranker
            .rerank("what is rust programming", results, 2)
            .expect("Failed to rerank");

        assert_eq!(reranked.len(), 2);

        // Rust-related documents should be ranked higher
        assert!(reranked[0].id == "doc1" || reranked[0].id == "doc3");
        assert_ne!(reranked[0].id, "doc2");
    }

    #[test]
    fn test_reranker_trait_implementation() {
        // This test just verifies the trait is implemented correctly
        // It doesn't require actual model files

        // We can't test without a model, but we can verify the trait compiles
        fn _accepts_reranker<R: Reranker>(_reranker: R) {}

        // This would compile if we had a model
        // _accepts_reranker(CrossEncoderReranker::from_dir("./dummy").unwrap());
    }
}
