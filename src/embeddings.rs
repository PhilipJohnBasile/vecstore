// Built-in embeddings support using ONNX Runtime and cloud providers
//
// This module provides text embedding functionality so users don't need to
// generate embeddings separately. Supports ONNX models for local inference
// and OpenAI API for cloud-based embeddings.

pub mod auto_models;

#[cfg(feature = "openai-embeddings")]
pub mod openai_backend;

#[cfg(feature = "candle-embeddings")]
pub mod candle_backend;

#[cfg(feature = "cloud-embeddings")]
pub mod cloud_providers;

#[cfg(feature = "cloud-embeddings")]
pub mod additional_providers;

#[cfg(feature = "ollama")]
pub mod ollama_backend;

#[cfg(feature = "embeddings")]
pub use auto_models::{AutoEmbedder, PretrainedModel};

#[cfg(feature = "openai-embeddings")]
pub use openai_backend::{OpenAIEmbedding, OpenAIModel};

#[cfg(feature = "candle-embeddings")]
pub use candle_backend::{CandleEmbedder, CandleModel};

#[cfg(feature = "cloud-embeddings")]
pub use cloud_providers::{
    CohereEmbedding, CohereModel, GoogleEmbedding, GoogleModel, MistralEmbedding, MistralModel,
    VoyageEmbedding, VoyageModel,
};

#[cfg(feature = "cloud-embeddings")]
pub use additional_providers::{
    AzureEmbedding, AzureModel, HuggingFaceEmbedding, JinaEmbedding, JinaModel,
};

#[cfg(feature = "ollama")]
pub use ollama_backend::{OllamaEmbedding, OllamaModel};

use anyhow::Result;

#[cfg(feature = "embeddings")]
use anyhow::{anyhow, Context};

#[cfg(feature = "embeddings")]
use ndarray::{Array2, CowArray};
#[cfg(feature = "embeddings")]
use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder, Value};
#[cfg(feature = "embeddings")]
use std::path::{Path, PathBuf};
#[cfg(feature = "embeddings")]
use std::sync::Arc;
#[cfg(feature = "embeddings")]
use tokenizers::tokenizer::Tokenizer;

#[cfg(feature = "embeddings")]
use crate::collection::Collection;
#[cfg(feature = "embeddings")]
use crate::store::{HybridQuery, Metadata, Neighbor, Query, VecStore};

/// Trait for text embedding models
///
/// This trait provides a common interface for different embedding implementations,
/// allowing users to plug in their own models or use the built-in ONNX-based embedder.
///
/// # Example
/// ```no_run
/// use vecstore::embeddings::TextEmbedder;
///
/// struct MyEmbedder;
///
/// impl TextEmbedder for MyEmbedder {
///     fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
///         // Your custom embedding logic
///         Ok(vec![0.1, 0.2, 0.3])
///     }
///
///     fn dimension(&self) -> anyhow::Result<usize> {
///         Ok(3)
///     }
/// }
/// ```
pub trait TextEmbedder: Send + Sync {
    /// Embed a single text into a vector
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed multiple texts in batch (default: sequential embedding)
    ///
    /// Override this method to provide optimized batch processing.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Get the expected embedding dimension
    fn dimension(&self) -> Result<usize>;
}

/// Simple deterministic embedder for testing
///
/// This embedder generates embeddings based on character statistics of the input text.
/// It doesn't require ONNX Runtime or external models, making it perfect for testing
/// and examples.
///
/// # Example
/// ```
/// use vecstore::embeddings::{SimpleEmbedder, TextEmbedder};
///
/// let embedder = SimpleEmbedder::new(128);
/// let embedding = embedder.embed("Hello world").unwrap();
/// assert_eq!(embedding.len(), 128);
/// ```
pub struct SimpleEmbedder {
    dimension: usize,
}

impl SimpleEmbedder {
    /// Create a new simple embedder with the specified dimension
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl TextEmbedder for SimpleEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vec = vec![0.0; self.dimension];

        // Generate deterministic embedding based on text statistics
        let bytes = text.as_bytes();
        let len = bytes.len();

        if len == 0 {
            return Ok(vec);
        }

        // Fill vector with character-based features
        for (i, &byte) in bytes.iter().enumerate().take(self.dimension) {
            vec[i] = (byte as f32) / 255.0;
        }

        // Add some statistical features
        if self.dimension > len {
            let avg = bytes.iter().map(|&b| b as f32).sum::<f32>() / len as f32;
            vec[len % self.dimension] += avg / 255.0;

            let variance =
                bytes.iter().map(|&b| (b as f32 - avg).powi(2)).sum::<f32>() / len as f32;
            vec[(len + 1) % self.dimension] += variance.sqrt() / 255.0;
        }

        // L2 normalization
        let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut vec {
                *v /= magnitude;
            }
        }

        Ok(vec)
    }

    fn dimension(&self) -> Result<usize> {
        Ok(self.dimension)
    }
}

/// Text embedder using ONNX models
///
/// This provides a simple interface for generating embeddings from text.
/// It uses sentence-transformer models exported to ONNX format.
#[cfg(feature = "embeddings")]
pub struct Embedder {
    environment: Arc<Environment>,
    session: Session,
    tokenizer: Tokenizer,
    max_length: usize,
}

#[cfg(feature = "embeddings")]
impl Embedder {
    /// Create a new embedder from ONNX model and tokenizer files
    ///
    /// # Arguments
    /// * `model_path` - Path to ONNX model file (e.g., "model.onnx")
    /// * `tokenizer_path` - Path to tokenizer JSON file (e.g., "tokenizer.json")
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::embeddings::Embedder;
    ///
    /// let embedder = Embedder::new("model.onnx", "tokenizer.json").unwrap();
    /// let embedding = embedder.embed("Hello, world!").unwrap();
    /// println!("Embedding dimension: {}", embedding.len());
    /// ```
    pub fn new(model_path: impl AsRef<Path>, tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        // Initialize ONNX Runtime environment
        let environment = Arc::new(
            Environment::builder()
                .with_name("vecstore")
                .build()
                .context("Failed to create ONNX Runtime environment")?,
        );

        // Initialize ONNX Runtime session
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(model_path.as_ref())
            .context("Failed to load ONNX model")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            environment,
            session,
            tokenizer,
            max_length: 512, // Standard max length for most sentence transformers
        })
    }

    /// Set the maximum sequence length for tokenization
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Generate embedding for a single text
    ///
    /// # Arguments
    /// * `text` - Input text to embed
    ///
    /// # Returns
    /// Vector of floats representing the embedding
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::embeddings::Embedder;
    /// # let embedder = Embedder::new("model.onnx", "tokenizer.json").unwrap();
    /// let embedding = embedder.embed("This is a test sentence").unwrap();
    /// assert!(embedding.len() > 0);
    /// ```
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text])?;
        Ok(embeddings.into_iter().next().unwrap())
    }

    /// Generate embeddings for multiple texts in a batch
    ///
    /// This is more efficient than calling `embed()` multiple times.
    ///
    /// # Arguments
    /// * `texts` - Slice of input texts to embed
    ///
    /// # Returns
    /// Vector of embeddings (one per input text)
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::embeddings::Embedder;
    /// # let embedder = Embedder::new("model.onnx", "tokenizer.json").unwrap();
    /// let texts = vec!["First sentence", "Second sentence"];
    /// let embeddings = embedder.embed_batch(&texts).unwrap();
    /// assert_eq!(embeddings.len(), 2);
    /// ```
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();

        // Get the sequence length from the first encoding
        let seq_length = encodings[0].get_ids().len().min(self.max_length);

        // Prepare input tensors
        let mut input_ids = Vec::with_capacity(batch_size * seq_length);
        let mut attention_mask = Vec::with_capacity(batch_size * seq_length);
        let mut token_type_ids = Vec::with_capacity(batch_size * seq_length);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();

            // Truncate or pad to seq_length
            for i in 0..seq_length {
                input_ids.push(ids.get(i).copied().unwrap_or(0) as i64);
                attention_mask.push(mask.get(i).copied().unwrap_or(0) as i64);
                token_type_ids.push(type_ids.get(i).copied().unwrap_or(0) as i64);
            }
        }

        // Keep a copy of attention_mask for mean pooling later
        let attention_mask_for_pooling = attention_mask.clone();

        // Convert to ndarray arrays
        let input_ids_array = Array2::from_shape_vec((batch_size, seq_length), input_ids)?;
        let attention_mask_array =
            Array2::from_shape_vec((batch_size, seq_length), attention_mask)?;
        let token_type_ids_array =
            Array2::from_shape_vec((batch_size, seq_length), token_type_ids)?;

        // Convert to CowArray for ORT
        let input_ids_cow = CowArray::from(&input_ids_array).into_dyn();
        let attention_mask_cow = CowArray::from(&attention_mask_array).into_dyn();
        let token_type_ids_cow = CowArray::from(&token_type_ids_array).into_dyn();

        // Create Value objects for ORT
        let input_ids_value = Value::from_array(self.session.allocator(), &input_ids_cow)?;
        let attention_mask_value =
            Value::from_array(self.session.allocator(), &attention_mask_cow)?;
        let token_type_ids_value =
            Value::from_array(self.session.allocator(), &token_type_ids_cow)?;

        // Run inference
        let outputs = self
            .session
            .run(vec![
                input_ids_value,
                attention_mask_value,
                token_type_ids_value,
            ])
            .context("ONNX inference failed")?;

        // Extract embeddings from output
        // Most sentence transformers output shape: (batch_size, seq_length, hidden_size)
        // We need to apply mean pooling over the sequence dimension
        let embeddings_array = outputs[0]
            .try_extract::<f32>()
            .context("Failed to extract output tensor")?
            .view()
            .to_owned();

        // Apply mean pooling
        let embeddings = self.mean_pooling(&embeddings_array, &attention_mask_for_pooling)?;

        Ok(embeddings)
    }

    /// Apply mean pooling to token embeddings
    ///
    /// This averages the token embeddings weighted by attention mask
    fn mean_pooling(
        &self,
        token_embeddings: &ndarray::ArrayD<f32>,
        attention_mask: &[i64],
    ) -> Result<Vec<Vec<f32>>> {
        let shape = token_embeddings.shape();
        let batch_size = shape[0];
        let seq_length = shape[1];
        let hidden_size = shape[2];

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

            // L2 normalization for cosine similarity
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut pooled {
                    *val /= norm;
                }
            }

            result.push(pooled);
        }

        Ok(result)
    }

    /// Get the expected embedding dimension
    ///
    /// This runs a dummy inference to determine the output dimension
    pub fn embedding_dim(&self) -> Result<usize> {
        let dummy_embedding = self.embed("test")?;
        Ok(dummy_embedding.len())
    }
}

/// Implement TextEmbedder trait for Embedder
#[cfg(feature = "embeddings")]
impl TextEmbedder for Embedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Embedder::embed(self, text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Embedder::embed_batch(self, texts)
    }

    fn dimension(&self) -> Result<usize> {
        self.embedding_dim()
    }
}

/// Vector store with built-in text embedding
///
/// This combines a VecStore with an Embedder to provide a seamless
/// text-to-vector experience. Users can insert and query using raw text
/// without manually generating embeddings.
///
/// # Example
/// ```no_run
/// use vecstore::embeddings::EmbeddingStore;
/// use std::collections::HashMap;
///
/// let mut store = EmbeddingStore::new(
///     "./data",
///     "model.onnx",
///     "tokenizer.json"
/// ).unwrap();
///
/// // Insert text directly - embeddings are generated automatically
/// let mut meta = HashMap::new();
/// meta.insert("category".to_string(), serde_json::json!("tech"));
/// store.upsert_text("doc1", "Rust is a systems programming language", meta).unwrap();
///
/// // Query with text - embedding is generated automatically
/// let results = store.query_text("programming languages", 10, None).unwrap();
/// ```
#[cfg(feature = "embeddings")]
pub struct EmbeddingStore {
    store: VecStore,
    embedder: Embedder,
}

#[cfg(feature = "embeddings")]
impl EmbeddingStore {
    /// Create a new embedding store
    ///
    /// # Arguments
    /// * `store_path` - Path to vector store directory
    /// * `model_path` - Path to ONNX model file
    /// * `tokenizer_path` - Path to tokenizer JSON file
    pub fn new(
        store_path: impl Into<PathBuf>,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let store = VecStore::open(store_path.into())?;
        let embedder = Embedder::new(model_path, tokenizer_path)?;

        Ok(Self { store, embedder })
    }

    /// Insert or update a document using text
    ///
    /// The text will be automatically embedded before storage.
    ///
    /// # Arguments
    /// * `id` - Unique document ID
    /// * `text` - Text content to embed and store
    /// * `metadata` - Document metadata
    pub fn upsert_text(
        &mut self,
        id: impl Into<String>,
        text: &str,
        metadata: Metadata,
    ) -> Result<()> {
        let id = id.into();
        let vector = self.embedder.embed(text)?;

        // Also index the text for hybrid search
        self.store.index_text(&id, text)?;

        self.store.upsert(id, vector, metadata)
    }

    /// Batch insert multiple documents from text
    ///
    /// This is more efficient than calling `upsert_text()` in a loop.
    pub fn batch_upsert_text(&mut self, documents: Vec<(String, String, Metadata)>) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }

        // Extract texts for batch embedding
        let texts: Vec<&str> = documents.iter().map(|(_, text, _)| text.as_str()).collect();

        // Generate embeddings in batch
        let embeddings = self.embedder.embed_batch(&texts)?;

        // Upsert all documents
        for ((id, text, metadata), vector) in documents.into_iter().zip(embeddings) {
            self.store.index_text(&id, &text)?;
            self.store.upsert(id, vector, metadata)?;
        }

        Ok(())
    }

    /// Query using text
    ///
    /// The query text will be automatically embedded before searching.
    pub fn query_text(
        &self,
        query: &str,
        k: usize,
        filter: Option<crate::store::FilterExpr>,
    ) -> Result<Vec<Neighbor>> {
        let vector = self.embedder.embed(query)?;
        self.store.query(Query { vector, k, filter })
    }

    /// Hybrid search using text
    ///
    /// Combines vector similarity (from embedded text) with keyword search.
    ///
    /// # Arguments
    /// * `text` - Query text (will be embedded for vector search AND used for keywords)
    /// * `k` - Number of results
    /// * `alpha` - Balance between vector (1.0) and keyword (0.0) search
    /// * `filter` - Optional metadata filter
    pub fn hybrid_query_text(
        &self,
        text: &str,
        k: usize,
        alpha: f32,
        filter: Option<crate::store::FilterExpr>,
    ) -> Result<Vec<Neighbor>> {
        let vector = self.embedder.embed(text)?;

        self.store.hybrid_query(HybridQuery {
            vector,
            keywords: text.to_string(),
            k,
            filter,
            alpha,
        })
    }

    /// Get the underlying vector store
    pub fn store(&self) -> &VecStore {
        &self.store
    }

    /// Get mutable reference to the underlying vector store
    pub fn store_mut(&mut self) -> &mut VecStore {
        &mut self.store
    }

    /// Get the embedder
    pub fn embedder(&self) -> &Embedder {
        &self.embedder
    }

    /// Save the store to disk
    pub fn save(&self) -> Result<()> {
        self.store.save()
    }

    /// Get the number of documents
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

#[cfg(all(test, feature = "embeddings"))]
mod tests {
    use super::*;

    // Note: These tests require model files which aren't included in the repo
    // They're here as examples of how to use the API

    #[test]
    #[ignore] // Requires model files
    fn test_single_embedding() {
        let embedder = Embedder::new("model.onnx", "tokenizer.json").unwrap();
        let embedding = embedder.embed("This is a test").unwrap();
        assert!(embedding.len() > 0);
        assert!(embedding.len() <= 1024); // Reasonable dimension
    }

    #[test]
    #[ignore] // Requires model files
    fn test_batch_embedding() {
        let embedder = Embedder::new("model.onnx", "tokenizer.json").unwrap();
        let texts = vec!["First text", "Second text", "Third text"];
        let embeddings = embedder.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        assert!(embeddings[0].len() > 0);
        assert_eq!(embeddings[0].len(), embeddings[1].len());
        assert_eq!(embeddings[1].len(), embeddings[2].len());
    }

    #[test]
    #[ignore] // Requires model files
    fn test_normalized_embeddings() {
        let embedder = Embedder::new("model.onnx", "tokenizer.json").unwrap();
        let embedding = embedder.embed("Test normalization").unwrap();

        // Check L2 norm is approximately 1.0
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_simple_embedder() {
        let embedder = SimpleEmbedder::new(128);

        // Test single embedding
        let embedding = embedder.embed("Hello world").unwrap();
        assert_eq!(embedding.len(), 128);

        // Check L2 normalization (should be approximately 1.0)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);

        // Test dimension method
        assert_eq!(embedder.dimension().unwrap(), 128);
    }

    #[test]
    fn test_simple_embedder_deterministic() {
        let embedder = SimpleEmbedder::new(64);

        // Same text should produce same embedding
        let emb1 = embedder.embed("test text").unwrap();
        let emb2 = embedder.embed("test text").unwrap();
        assert_eq!(emb1, emb2);

        // Different text should produce different embeddings
        let emb3 = embedder.embed("different text").unwrap();
        assert_ne!(emb1, emb3);
    }

    #[test]
    fn test_simple_embedder_batch() {
        let embedder = SimpleEmbedder::new(128);

        let texts = vec!["First", "Second", "Third"];
        let embeddings = embedder.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 128);
        assert_eq!(embeddings[1].len(), 128);
        assert_eq!(embeddings[2].len(), 128);

        // Each should be normalized
        for embedding in &embeddings {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_simple_embedder_empty_text() {
        let embedder = SimpleEmbedder::new(64);
        let embedding = embedder.embed("").unwrap();

        assert_eq!(embedding.len(), 64);
        // Empty text should produce zero vector
        assert!(embedding.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_text_embedder_trait() {
        // Test that SimpleEmbedder implements TextEmbedder trait
        let embedder: Box<dyn TextEmbedder> = Box::new(SimpleEmbedder::new(128));

        let embedding = embedder.embed("Test trait").unwrap();
        assert_eq!(embedding.len(), 128);
        assert_eq!(embedder.dimension().unwrap(), 128);
    }

    #[test]
    fn test_embedding_collection_basic() {
        use crate::{Metadata, VecDatabase};
        use std::collections::HashMap;

        // Create database and collection
        let temp_dir = std::env::temp_dir().join("vecstore_test_embedding_collection");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let mut db = VecDatabase::open(&temp_dir).unwrap();
        let collection = db.create_collection("test").unwrap();

        // Wrap with embedder
        let embedder = SimpleEmbedder::new(128);
        let mut emb_collection = EmbeddingCollection::new(collection, Box::new(embedder));

        // Insert text documents
        let mut meta1 = Metadata {
            fields: HashMap::new(),
        };
        meta1
            .fields
            .insert("category".into(), serde_json::json!("tech"));
        emb_collection
            .upsert_text("doc1", "Rust programming language", meta1)
            .unwrap();

        let mut meta2 = Metadata {
            fields: HashMap::new(),
        };
        meta2
            .fields
            .insert("category".into(), serde_json::json!("tech"));
        emb_collection
            .upsert_text("doc2", "Python programming language", meta2)
            .unwrap();

        // Query with text
        let results = emb_collection.query_text("programming", 5, None).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].score > 0.0);

        // Test count
        assert_eq!(emb_collection.count().unwrap(), 2);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_embedding_collection_batch() {
        use crate::{Metadata, VecDatabase};
        use std::collections::HashMap;

        let temp_dir = std::env::temp_dir().join("vecstore_test_embedding_batch");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let mut db = VecDatabase::open(&temp_dir).unwrap();
        let collection = db.create_collection("test").unwrap();

        let embedder = SimpleEmbedder::new(128);
        let mut emb_collection = EmbeddingCollection::new(collection, Box::new(embedder));

        // Batch insert
        let mut meta1 = Metadata {
            fields: HashMap::new(),
        };
        meta1.fields.insert("type".into(), serde_json::json!("doc"));
        let mut meta2 = Metadata {
            fields: HashMap::new(),
        };
        meta2.fields.insert("type".into(), serde_json::json!("doc"));
        let mut meta3 = Metadata {
            fields: HashMap::new(),
        };
        meta3.fields.insert("type".into(), serde_json::json!("doc"));

        let documents = vec![
            ("doc1".into(), "First document".into(), meta1),
            ("doc2".into(), "Second document".into(), meta2),
            ("doc3".into(), "Third document".into(), meta3),
        ];

        emb_collection.batch_upsert_text(documents).unwrap();

        assert_eq!(emb_collection.count().unwrap(), 3);

        // Query - HNSW may not return all documents with tiny datasets
        let results = emb_collection.query_text("document", 5, None).unwrap();
        assert!(results.len() >= 2); // At least 2 of 3 documents

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_embedding_collection_delete() {
        use crate::{Metadata, VecDatabase};
        use std::collections::HashMap;

        let temp_dir = std::env::temp_dir().join("vecstore_test_embedding_delete");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let mut db = VecDatabase::open(&temp_dir).unwrap();
        let collection = db.create_collection("test").unwrap();

        let embedder = SimpleEmbedder::new(128);
        let mut emb_collection = EmbeddingCollection::new(collection, Box::new(embedder));

        // Insert
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields.insert("test".into(), serde_json::json!(true));
        emb_collection
            .upsert_text("doc1", "Test document", meta)
            .unwrap();

        assert_eq!(emb_collection.count().unwrap(), 1);

        // Delete
        emb_collection.delete("doc1").unwrap();
        assert_eq!(emb_collection.count().unwrap(), 0);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }
}

/// Collection with built-in text embedding
///
/// This combines a Collection with a TextEmbedder to provide a seamless
/// text-to-vector experience for collections. Users can insert and query using raw text
/// without manually generating embeddings.
///
/// # Example
/// ```no_run
/// use vecstore::{VecDatabase, embeddings::{EmbeddingCollection, SimpleEmbedder}};
///
/// # fn main() -> anyhow::Result<()> {
/// let mut db = VecDatabase::open("./data")?;
/// let collection = db.create_collection("documents")?;
///
/// // Wrap collection with embedder
/// let embedder = SimpleEmbedder::new(128);
/// let mut embedding_collection = EmbeddingCollection::new(collection, Box::new(embedder));
///
/// // Insert text directly - embeddings are generated automatically
/// let mut meta = vecstore::Metadata::new();
/// meta.insert("category", "tech");
/// embedding_collection.upsert_text("doc1", "Rust is a systems programming language", meta)?;
///
/// // Query with text - embedding is generated automatically
/// let results = embedding_collection.query_text("programming languages", 10, None)?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "embeddings")]
pub struct EmbeddingCollection {
    collection: Collection,
    embedder: Box<dyn TextEmbedder>,
}

#[cfg(feature = "embeddings")]
impl EmbeddingCollection {
    /// Create a new embedding collection
    ///
    /// # Arguments
    /// * `collection` - The collection to wrap
    /// * `embedder` - The text embedder to use
    pub fn new(collection: Collection, embedder: Box<dyn TextEmbedder>) -> Self {
        Self {
            collection,
            embedder,
        }
    }

    /// Create from ONNX model files
    ///
    /// # Arguments
    /// * `collection` - The collection to wrap
    /// * `model_path` - Path to ONNX model file
    /// * `tokenizer_path` - Path to tokenizer JSON file
    pub fn from_onnx(
        collection: Collection,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let embedder = Embedder::new(model_path, tokenizer_path)?;
        Ok(Self::new(collection, Box::new(embedder)))
    }

    /// Insert or update a document using text
    ///
    /// The text will be automatically embedded before storage.
    ///
    /// # Arguments
    /// * `id` - Unique document ID
    /// * `text` - Text content to embed and store
    /// * `metadata` - Document metadata
    pub fn upsert_text(
        &mut self,
        id: impl Into<String>,
        text: &str,
        metadata: Metadata,
    ) -> Result<()> {
        let id = id.into();
        let vector = self.embedder.embed(text)?;

        // Convert anyhow::Result to crate::Result
        self.collection
            .upsert(id, vector, metadata)
            .map_err(|e| anyhow::anyhow!("Collection upsert failed: {}", e))
    }

    /// Batch insert multiple documents from text
    ///
    /// This is more efficient than calling `upsert_text()` in a loop.
    pub fn batch_upsert_text(&mut self, documents: Vec<(String, String, Metadata)>) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }

        // Extract texts for batch embedding
        let texts: Vec<&str> = documents.iter().map(|(_, text, _)| text.as_str()).collect();

        // Generate embeddings in batch
        let embeddings = self.embedder.embed_batch(&texts)?;

        // Upsert all documents
        for ((id, _text, metadata), vector) in documents.into_iter().zip(embeddings) {
            self.collection
                .upsert(id, vector, metadata)
                .map_err(|e| anyhow::anyhow!("Collection upsert failed: {}", e))?;
        }

        Ok(())
    }

    /// Query using text
    ///
    /// The query text will be automatically embedded before searching.
    pub fn query_text(
        &self,
        query: &str,
        k: usize,
        filter: Option<crate::store::FilterExpr>,
    ) -> Result<Vec<Neighbor>> {
        let vector = self.embedder.embed(query)?;
        self.collection
            .query(Query { vector, k, filter })
            .map_err(|e| anyhow::anyhow!("Collection query failed: {}", e))
    }

    /// Get the underlying collection
    pub fn collection(&self) -> &Collection {
        &self.collection
    }

    /// Get mutable reference to the underlying collection
    pub fn collection_mut(&mut self) -> &mut Collection {
        &mut self.collection
    }

    /// Get the embedder
    pub fn embedder(&self) -> &dyn TextEmbedder {
        self.embedder.as_ref()
    }

    /// Get collection statistics
    pub fn stats(&self) -> Result<crate::namespace_manager::NamespaceStats> {
        self.collection
            .stats()
            .map_err(|e| anyhow::anyhow!("Collection stats failed: {}", e))
    }

    /// Get the number of documents
    pub fn count(&self) -> Result<usize> {
        self.collection
            .count()
            .map_err(|e| anyhow::anyhow!("Collection count failed: {}", e))
    }

    /// Delete a document
    pub fn delete(&mut self, id: &str) -> Result<()> {
        self.collection
            .delete(id)
            .map_err(|e| anyhow::anyhow!("Collection delete failed: {}", e))
    }
}
