// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Multimodal Search
//!
//! Cross-modal search supporting text, images, video, and audio embeddings
//! with unified query interface and multimodal reranking.
//!
//! ## Features
//!
//! - **Cross-Modal Search**: Text query finds images, image query finds text
//! - **Unified Embedding Space**: All modalities in same vector space (CLIP-like)
//! - **Multimodal Reranking**: Score fusion across modalities
//! - **Late Fusion**: Combine results from modality-specific indexes
//! - **Modality-Aware Filtering**: Filter by content type
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::multimodal_search::{MultimodalIndex, MultimodalQuery, Modality};
//!
//! let index = MultimodalIndex::new(config);
//!
//! // Index different modalities
//! index.insert("img1", Modality::Image, &image_embedding, metadata)?;
//! index.insert("txt1", Modality::Text, &text_embedding, metadata)?;
//!
//! // Cross-modal search: text finds images
//! let results = index.search(
//!     MultimodalQuery::text("a cat sitting on a couch")
//!         .with_target_modality(Modality::Image)
//! )?;
//! ```

use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Content modality types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Text content
    Text,
    /// Image content
    Image,
    /// Video content
    Video,
    /// Audio content
    Audio,
    /// 3D model/point cloud
    ThreeD,
    /// Code
    Code,
    /// Tabular data
    Tabular,
    /// Mixed/multimodal
    Mixed,
    /// Custom modality
    Custom(String),
}

impl Modality {
    /// Get typical embedding dimension for modality
    pub fn typical_dimension(&self) -> usize {
        match self {
            Modality::Text => 768,
            Modality::Image => 512,
            Modality::Video => 512,
            Modality::Audio => 512,
            Modality::ThreeD => 256,
            Modality::Code => 768,
            Modality::Tabular => 128,
            Modality::Mixed => 768,
            Modality::Custom(_) => 512,
        }
    }
}

/// Multimodal document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalDocument {
    /// Document ID
    pub id: String,
    /// Primary modality
    pub modality: Modality,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Additional modality embeddings (for true multimodal docs)
    pub auxiliary_embeddings: HashMap<String, Vec<f32>>,
    /// Raw content reference (URL/path)
    pub content_ref: Option<String>,
    /// Text content (if available)
    pub text_content: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Creation timestamp
    pub created_at: i64,
}

/// Multimodal query
#[derive(Debug, Clone)]
pub struct MultimodalQuery {
    /// Query modality
    pub source_modality: Modality,
    /// Query embedding
    pub embedding: Option<Vec<f32>>,
    /// Text query (will be embedded)
    pub text: Option<String>,
    /// Image query (reference)
    pub image_ref: Option<String>,
    /// Target modality for results
    pub target_modality: Option<Modality>,
    /// Filter by modalities
    pub modality_filter: Option<Vec<Modality>>,
    /// Metadata filters
    pub filters: HashMap<String, serde_json::Value>,
    /// Top-k results
    pub top_k: usize,
    /// Enable cross-modal reranking
    pub rerank: bool,
}

impl MultimodalQuery {
    /// Create text query
    pub fn text(text: &str) -> Self {
        Self {
            source_modality: Modality::Text,
            embedding: None,
            text: Some(text.to_string()),
            image_ref: None,
            target_modality: None,
            modality_filter: None,
            filters: HashMap::new(),
            top_k: 10,
            rerank: false,
        }
    }

    /// Create image query
    pub fn image(image_ref: &str) -> Self {
        Self {
            source_modality: Modality::Image,
            embedding: None,
            text: None,
            image_ref: Some(image_ref.to_string()),
            target_modality: None,
            modality_filter: None,
            filters: HashMap::new(),
            top_k: 10,
            rerank: false,
        }
    }

    /// Create vector query
    pub fn vector(embedding: Vec<f32>, modality: Modality) -> Self {
        Self {
            source_modality: modality,
            embedding: Some(embedding),
            text: None,
            image_ref: None,
            target_modality: None,
            modality_filter: None,
            filters: HashMap::new(),
            top_k: 10,
            rerank: false,
        }
    }

    /// Set target modality
    pub fn with_target_modality(mut self, modality: Modality) -> Self {
        self.target_modality = Some(modality);
        self
    }

    /// Filter by modalities
    pub fn with_modality_filter(mut self, modalities: Vec<Modality>) -> Self {
        self.modality_filter = Some(modalities);
        self
    }

    /// Set top-k
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Enable reranking
    pub fn with_reranking(mut self) -> Self {
        self.rerank = true;
        self
    }

    /// Add metadata filter
    pub fn with_filter(mut self, key: &str, value: serde_json::Value) -> Self {
        self.filters.insert(key.to_string(), value);
        self
    }
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalResult {
    /// Document ID
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Document modality
    pub modality: Modality,
    /// Text content if available
    pub text_content: Option<String>,
    /// Content reference
    pub content_ref: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Cross-modal scores (if reranked)
    pub cross_modal_scores: Option<HashMap<String, f32>>,
}

/// Multimodal index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Enable cross-modal search
    pub cross_modal: bool,
    /// Modality-specific indexes
    pub per_modality_index: bool,
    /// Fusion strategy
    pub fusion_strategy: FusionStrategy,
    /// Reranking configuration
    pub reranker: Option<RerankerConfig>,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            cross_modal: true,
            per_modality_index: true,
            fusion_strategy: FusionStrategy::WeightedSum,
            reranker: None,
        }
    }
}

/// Fusion strategy for multimodal results
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FusionStrategy {
    /// Weighted sum of scores
    WeightedSum,
    /// Maximum score
    MaxScore,
    /// Average score
    Average,
    /// Late fusion with modality weights
    LateFusion { weights: HashMap<String, f32> },
}

/// Reranker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Reranker type
    pub reranker_type: String,
    /// Model path/name
    pub model: String,
    /// Cross-modal reranking
    pub cross_modal: bool,
}

/// Simple embedding function trait
pub trait Embedder: Send + Sync {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_image(&self, image_ref: &str) -> Result<Vec<f32>>;
    fn dimension(&self) -> usize;
}

/// Mock embedder for testing
struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Embedder for MockEmbedder {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // Generate deterministic embedding based on text hash
        let mut embedding = vec![0.0f32; self.dimension];
        let bytes = text.as_bytes();

        for (i, chunk) in bytes.chunks(4).enumerate() {
            let idx = i % self.dimension;
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

        Ok(embedding)
    }

    fn embed_image(&self, image_ref: &str) -> Result<Vec<f32>> {
        // Use same logic as text for demo
        self.embed_text(image_ref)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Main multimodal index
pub struct MultimodalIndex {
    config: MultimodalConfig,
    /// All documents
    documents: RwLock<HashMap<String, MultimodalDocument>>,
    /// Per-modality indexes
    modality_indexes: RwLock<HashMap<Modality, Vec<String>>>,
    /// Embedder
    embedder: Box<dyn Embedder>,
}

impl MultimodalIndex {
    /// Create new multimodal index
    pub fn new(config: MultimodalConfig) -> Self {
        let embedder = Box::new(MockEmbedder::new(config.dimension));

        Self {
            config,
            documents: RwLock::new(HashMap::new()),
            modality_indexes: RwLock::new(HashMap::new()),
            embedder,
        }
    }

    /// Create with custom embedder
    pub fn with_embedder(config: MultimodalConfig, embedder: Box<dyn Embedder>) -> Self {
        Self {
            config,
            documents: RwLock::new(HashMap::new()),
            modality_indexes: RwLock::new(HashMap::new()),
            embedder,
        }
    }

    /// Insert document
    pub fn insert(&self, doc: MultimodalDocument) -> Result<()> {
        let id = doc.id.clone();
        let modality = doc.modality.clone();

        // Add to documents
        {
            let mut docs = self.documents.write().unwrap();
            docs.insert(id.clone(), doc);
        }

        // Add to modality index
        {
            let mut indexes = self.modality_indexes.write().unwrap();
            indexes.entry(modality).or_insert_with(Vec::new).push(id);
        }

        Ok(())
    }

    /// Insert text document
    pub fn insert_text(&self, id: &str, text: &str, metadata: HashMap<String, serde_json::Value>) -> Result<()> {
        let embedding = self.embedder.embed_text(text)?;

        let doc = MultimodalDocument {
            id: id.to_string(),
            modality: Modality::Text,
            embedding,
            auxiliary_embeddings: HashMap::new(),
            content_ref: None,
            text_content: Some(text.to_string()),
            metadata,
            created_at: unix_timestamp(),
        };

        self.insert(doc)
    }

    /// Insert image document
    pub fn insert_image(&self, id: &str, image_ref: &str, caption: Option<&str>, metadata: HashMap<String, serde_json::Value>) -> Result<()> {
        let embedding = self.embedder.embed_image(image_ref)?;

        let mut aux = HashMap::new();
        if let Some(cap) = caption {
            let text_emb = self.embedder.embed_text(cap)?;
            aux.insert("caption".to_string(), text_emb);
        }

        let doc = MultimodalDocument {
            id: id.to_string(),
            modality: Modality::Image,
            embedding,
            auxiliary_embeddings: aux,
            content_ref: Some(image_ref.to_string()),
            text_content: caption.map(|s| s.to_string()),
            metadata,
            created_at: unix_timestamp(),
        };

        self.insert(doc)
    }

    /// Search
    pub fn search(&self, query: MultimodalQuery) -> Result<Vec<MultimodalResult>> {
        // Get query embedding
        let query_embedding = if let Some(emb) = &query.embedding {
            emb.clone()
        } else if let Some(text) = &query.text {
            self.embedder.embed_text(text)?
        } else if let Some(img) = &query.image_ref {
            self.embedder.embed_image(img)?
        } else {
            return Err(VecStoreError::InvalidInput("No query provided".to_string()));
        };

        let docs = self.documents.read().unwrap();
        let indexes = self.modality_indexes.read().unwrap();

        // Determine which documents to search
        let candidate_ids: Vec<&String> = if let Some(target) = &query.target_modality {
            // Search only target modality
            indexes.get(target).map(|ids| ids.iter().collect()).unwrap_or_default()
        } else if let Some(filter) = &query.modality_filter {
            // Search filtered modalities
            filter.iter()
                .flat_map(|m| indexes.get(m).map(|ids| ids.iter()).unwrap_or_else(|| [].iter()))
                .collect()
        } else {
            // Search all
            docs.keys().collect()
        };

        // Compute scores
        let mut results: Vec<MultimodalResult> = candidate_ids
            .iter()
            .filter_map(|id| {
                docs.get(*id).map(|doc| {
                    let score = cosine_similarity(&query_embedding, &doc.embedding);

                    // Cross-modal scoring
                    let cross_scores = if query.rerank && !doc.auxiliary_embeddings.is_empty() {
                        let mut scores = HashMap::new();
                        for (name, aux_emb) in &doc.auxiliary_embeddings {
                            let aux_score = cosine_similarity(&query_embedding, aux_emb);
                            scores.insert(name.clone(), aux_score);
                        }
                        Some(scores)
                    } else {
                        None
                    };

                    // Combine scores
                    let final_score = if let Some(ref cs) = cross_scores {
                        match &self.config.fusion_strategy {
                            FusionStrategy::WeightedSum => {
                                let aux_avg: f32 = cs.values().sum::<f32>() / cs.len().max(1) as f32;
                                score * 0.7 + aux_avg * 0.3
                            }
                            FusionStrategy::MaxScore => {
                                cs.values().fold(score, |max, &s| max.max(s))
                            }
                            FusionStrategy::Average => {
                                let sum: f32 = cs.values().sum::<f32>() + score;
                                sum / (cs.len() + 1) as f32
                            }
                            FusionStrategy::LateFusion { weights } => {
                                let mut weighted_sum = score;
                                for (name, &s) in cs {
                                    let w = weights.get(name).copied().unwrap_or(1.0);
                                    weighted_sum += s * w;
                                }
                                weighted_sum
                            }
                        }
                    } else {
                        score
                    };

                    // Apply metadata filters
                    let passes_filters = query.filters.iter().all(|(key, value)| {
                        doc.metadata.get(key).map_or(false, |v| v == value)
                    });

                    if passes_filters {
                        Some(MultimodalResult {
                            id: doc.id.clone(),
                            score: final_score,
                            modality: doc.modality.clone(),
                            text_content: doc.text_content.clone(),
                            content_ref: doc.content_ref.clone(),
                            metadata: doc.metadata.clone(),
                            cross_modal_scores: cross_scores,
                        })
                    } else {
                        None
                    }
                })
            })
            .flatten()
            .collect();

        // Sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(query.top_k);

        Ok(results)
    }

    /// Text-to-image search
    pub fn text_to_image(&self, text: &str, top_k: usize) -> Result<Vec<MultimodalResult>> {
        let query = MultimodalQuery::text(text)
            .with_target_modality(Modality::Image)
            .with_top_k(top_k);
        self.search(query)
    }

    /// Image-to-text search
    pub fn image_to_text(&self, image_ref: &str, top_k: usize) -> Result<Vec<MultimodalResult>> {
        let query = MultimodalQuery::image(image_ref)
            .with_target_modality(Modality::Text)
            .with_top_k(top_k);
        self.search(query)
    }

    /// Image-to-image search
    pub fn image_to_image(&self, image_ref: &str, top_k: usize) -> Result<Vec<MultimodalResult>> {
        let query = MultimodalQuery::image(image_ref)
            .with_target_modality(Modality::Image)
            .with_top_k(top_k);
        self.search(query)
    }

    /// Get document by ID
    pub fn get(&self, id: &str) -> Option<MultimodalDocument> {
        let docs = self.documents.read().unwrap();
        docs.get(id).cloned()
    }

    /// Delete document
    pub fn delete(&self, id: &str) -> bool {
        let mut docs = self.documents.write().unwrap();

        if let Some(doc) = docs.remove(id) {
            let mut indexes = self.modality_indexes.write().unwrap();
            if let Some(ids) = indexes.get_mut(&doc.modality) {
                ids.retain(|x| x != id);
            }
            true
        } else {
            false
        }
    }

    /// Get statistics
    pub fn stats(&self) -> MultimodalStats {
        let docs = self.documents.read().unwrap();
        let indexes = self.modality_indexes.read().unwrap();

        let mut by_modality = HashMap::new();
        for (modality, ids) in indexes.iter() {
            by_modality.insert(format!("{:?}", modality), ids.len());
        }

        MultimodalStats {
            total_documents: docs.len(),
            by_modality,
            dimension: self.config.dimension,
        }
    }
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalStats {
    pub total_documents: usize,
    pub by_modality: HashMap<String, usize>,
    pub dimension: usize,
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

fn unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_index() {
        let config = MultimodalConfig::default();
        let index = MultimodalIndex::new(config);

        // Insert text
        index.insert_text("txt1", "A beautiful sunset over the ocean", HashMap::new()).unwrap();

        // Insert image
        index.insert_image("img1", "/path/to/sunset.jpg", Some("sunset photo"), HashMap::new()).unwrap();

        // Search
        let results = index.search(MultimodalQuery::text("sunset")).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_cross_modal_search() {
        let config = MultimodalConfig::default();
        let index = MultimodalIndex::new(config);

        // Insert images
        for i in 0..5 {
            index.insert_image(
                &format!("img_{}", i),
                &format!("/images/{}.jpg", i),
                Some(&format!("Photo number {}", i)),
                HashMap::new(),
            ).unwrap();
        }

        // Text to image search
        let results = index.text_to_image("Photo", 3).unwrap();
        assert_eq!(results.len(), 3);

        for result in &results {
            assert_eq!(result.modality, Modality::Image);
        }
    }

    #[test]
    fn test_modality_filter() {
        let config = MultimodalConfig::default();
        let index = MultimodalIndex::new(config);

        index.insert_text("txt1", "Hello world", HashMap::new()).unwrap();
        index.insert_image("img1", "/img.jpg", None, HashMap::new()).unwrap();

        // Filter to only text
        let results = index.search(
            MultimodalQuery::text("hello")
                .with_modality_filter(vec![Modality::Text])
        ).unwrap();

        assert!(results.iter().all(|r| r.modality == Modality::Text));
    }

    #[test]
    fn test_stats() {
        let config = MultimodalConfig::default();
        let index = MultimodalIndex::new(config);

        index.insert_text("txt1", "Text document", HashMap::new()).unwrap();
        index.insert_text("txt2", "Another text", HashMap::new()).unwrap();
        index.insert_image("img1", "/img.jpg", None, HashMap::new()).unwrap();

        let stats = index.stats();
        assert_eq!(stats.total_documents, 3);
        assert_eq!(*stats.by_modality.get("Text").unwrap_or(&0), 2);
        assert_eq!(*stats.by_modality.get("Image").unwrap_or(&0), 1);
    }
}
