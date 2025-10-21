//! Multi-Modal Vector Search
//!
//! Enables searching across different modalities (text, images, audio, etc.)
//! in a unified embedding space. Supports cross-modal retrieval where you can:
//! - Query with text to find similar images
//! - Query with image to find related text
//! - Combine multiple modalities with weighted fusion
//!
//! ## Features
//!
//! - **Cross-Modal Search**: Query one modality, retrieve another
//! - **Multi-Modal Fusion**: Combine embeddings from different modalities
//! - **Modality Weighting**: Balance importance of different modalities
//! - **Late Fusion**: Merge results from separate modal searches
//! - **Early Fusion**: Combine embeddings before search
//!
//! ## Use Cases
//!
//! - **E-commerce**: Search products by text description OR product image
//! - **Media Libraries**: Find videos/images using text queries
//! - **Medical Imaging**: Search medical images with text symptoms
//! - **Social Media**: Search posts by image similarity or text content
//! - **Research**: Cross-reference papers, figures, and citations
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::multimodal::{MultiModalIndex, Modality, MultiModalQuery, MultiModalFusion};
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut index = MultiModalIndex::new()?;
//!
//! // Add text embedding
//! index.add_with_modality(
//!     "doc1",
//!     vec![0.1; 512],
//!     Modality::Text,
//!     None
//! )?;
//!
//! // Add image embedding
//! index.add_with_modality(
//!     "img1",
//!     vec![0.2; 512],
//!     Modality::Image,
//!     None
//! )?;
//!
//! // Cross-modal search: text query to find images
//! let query = MultiModalQuery::new(vec![0.15; 512], Modality::Text)
//!     .with_target_modality(Modality::Image) // Find images
//!     .with_limit(10);
//!
//! let results = index.search(&query)?;
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Text embeddings (BERT, GPT, etc.)
    Text,

    /// Image embeddings (CLIP, ResNet, ViT, etc.)
    Image,

    /// Audio embeddings (Wav2Vec, CLAP, etc.)
    Audio,

    /// Video embeddings (VideoMAE, TimeSformer, etc.)
    Video,

    /// Code embeddings (CodeBERT, GraphCodeBERT, etc.)
    Code,

    /// Custom modality
    Custom(u8),
}

impl Modality {
    /// Get modality name
    pub fn name(&self) -> &str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
            Modality::Video => "video",
            Modality::Code => "code",
            Modality::Custom(_) => "custom",
        }
    }
}

/// Fusion strategy for combining multi-modal embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiModalFusion {
    /// Average embeddings with equal weight
    Average,

    /// Weighted average (weights must sum to 1.0)
    WeightedAverage(Vec<f32>),

    /// Concatenate embeddings (dimension = sum of all)
    Concatenate,

    /// Maximum value per dimension across embeddings
    Max,

    /// Late fusion: search each modality separately, then merge results
    LateFusion {
        /// Weights for each modality's results
        weights: HashMap<Modality, f32>,
    },
}

/// Multi-modal entry with embeddings from different modalities
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiModalEntry {
    /// Unique identifier
    pub id: String,

    /// Embeddings for each modality
    pub embeddings: HashMap<Modality, Vec<f32>>,

    /// Optional metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl MultiModalEntry {
    /// Create a new multi-modal entry
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            embeddings: HashMap::new(),
            metadata: None,
        }
    }

    /// Add an embedding for a specific modality
    pub fn add_embedding(mut self, modality: Modality, embedding: Vec<f32>) -> Self {
        self.embeddings.insert(modality, embedding);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get embedding for a modality
    pub fn get_embedding(&self, modality: Modality) -> Option<&Vec<f32>> {
        self.embeddings.get(&modality)
    }

    /// Check if entry has a specific modality
    pub fn has_modality(&self, modality: Modality) -> bool {
        self.embeddings.contains_key(&modality)
    }
}

/// Multi-modal search query
#[derive(Clone)]
pub struct MultiModalQuery {
    /// Query embeddings (can have multiple modalities)
    pub embeddings: HashMap<Modality, Vec<f32>>,

    /// Target modalities to search (None = search all)
    pub target_modalities: Option<Vec<Modality>>,

    /// Maximum results
    pub limit: usize,

    /// Fusion strategy
    pub fusion: MultiModalFusion,

    /// Metadata filter (optional)
    pub filter: Option<HashMap<String, serde_json::Value>>,
}

impl MultiModalQuery {
    /// Create a new query with a single modality
    pub fn new(embedding: Vec<f32>, modality: Modality) -> Self {
        let mut embeddings = HashMap::new();
        embeddings.insert(modality, embedding);

        Self {
            embeddings,
            target_modalities: None,
            limit: 10,
            fusion: MultiModalFusion::Average,
            filter: None,
        }
    }

    /// Create query with multiple modalities
    pub fn multi(embeddings: HashMap<Modality, Vec<f32>>) -> Self {
        Self {
            embeddings,
            target_modalities: None,
            limit: 10,
            fusion: MultiModalFusion::Average,
            filter: None,
        }
    }

    /// Set target modality to search (cross-modal search)
    pub fn with_target_modality(mut self, modality: Modality) -> Self {
        self.target_modalities = Some(vec![modality]);
        self
    }

    /// Set multiple target modalities
    pub fn with_target_modalities(mut self, modalities: Vec<Modality>) -> Self {
        self.target_modalities = Some(modalities);
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set fusion strategy
    pub fn with_fusion(mut self, fusion: MultiModalFusion) -> Self {
        self.fusion = fusion;
        self
    }

    /// Set metadata filter
    pub fn with_filter(mut self, filter: HashMap<String, serde_json::Value>) -> Self {
        self.filter = Some(filter);
        self
    }
}

/// Multi-modal search result
#[derive(Debug, Clone)]
pub struct MultiModalResult {
    pub id: String,
    pub score: f32,
    pub distance: f32,
    pub modality: Modality,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Multi-modal vector search index
pub struct MultiModalIndex {
    /// All entries
    entries: Vec<MultiModalEntry>,

    /// Dimension per modality
    dimensions: HashMap<Modality, usize>,
}

impl MultiModalIndex {
    /// Create a new multi-modal index
    pub fn new() -> Result<Self> {
        Ok(Self {
            entries: Vec::new(),
            dimensions: HashMap::new(),
        })
    }

    /// Add an entry with a single modality
    pub fn add_with_modality(
        &mut self,
        id: impl Into<String>,
        embedding: Vec<f32>,
        modality: Modality,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<()> {
        // Track dimension for this modality
        let dim = embedding.len();
        if let Some(&existing_dim) = self.dimensions.get(&modality) {
            if existing_dim != dim {
                return Err(anyhow!(
                    "Dimension mismatch for modality {:?}: expected {}, got {}",
                    modality,
                    existing_dim,
                    dim
                ));
            }
        } else {
            self.dimensions.insert(modality, dim);
        }

        let mut entry = MultiModalEntry::new(id);
        entry.embeddings.insert(modality, embedding);
        if let Some(meta) = metadata {
            entry.metadata = Some(meta);
        }

        self.entries.push(entry);
        Ok(())
    }

    /// Add a multi-modal entry
    pub fn add_entry(&mut self, entry: MultiModalEntry) -> Result<()> {
        // Validate dimensions
        for (modality, embedding) in &entry.embeddings {
            let dim = embedding.len();
            if let Some(&existing_dim) = self.dimensions.get(modality) {
                if existing_dim != dim {
                    return Err(anyhow!(
                        "Dimension mismatch for modality {:?}: expected {}, got {}",
                        modality,
                        existing_dim,
                        dim
                    ));
                }
            } else {
                self.dimensions.insert(*modality, dim);
            }
        }

        self.entries.push(entry);
        Ok(())
    }

    /// Search the index
    pub fn search(&self, query: &MultiModalQuery) -> Result<Vec<MultiModalResult>> {
        if self.entries.is_empty() {
            return Ok(Vec::new());
        }

        let results = match &query.fusion {
            MultiModalFusion::LateFusion { weights } => self.late_fusion_search(query, weights)?,
            _ => self.early_fusion_search(query)?,
        };

        Ok(results)
    }

    /// Early fusion: combine query embeddings, then search
    fn early_fusion_search(&self, query: &MultiModalQuery) -> Result<Vec<MultiModalResult>> {
        // Determine which modalities to search
        let target_modalities = if let Some(ref targets) = query.target_modalities {
            targets.clone()
        } else {
            // Search all modalities present in entries
            self.dimensions.keys().copied().collect()
        };

        let mut results: Vec<MultiModalResult> = self
            .entries
            .par_iter()
            .filter(|entry| {
                // Filter by target modality
                target_modalities.iter().any(|m| entry.has_modality(*m))
            })
            .filter(|entry| {
                // Apply metadata filter if present
                if let Some(ref filter) = query.filter {
                    if let Some(ref entry_meta) = entry.metadata {
                        filter.iter().all(|(k, v)| entry_meta.get(k) == Some(v))
                    } else {
                        false
                    }
                } else {
                    true
                }
            })
            .flat_map(|entry| {
                // For each entry, compute similarity for each target modality
                target_modalities
                    .iter()
                    .filter_map(|&target_modality| {
                        let entry_embedding = entry.get_embedding(target_modality)?;

                        // Compute fused query embedding for this modality
                        let query_embedding = self
                            .fuse_query_embeddings(
                                &query.embeddings,
                                &query.fusion,
                                target_modality,
                            )
                            .ok()?;

                        // Compute similarity
                        let distance = euclidean_distance(&query_embedding, entry_embedding);
                        let score = 1.0 / (1.0 + distance);

                        Some(MultiModalResult {
                            id: entry.id.clone(),
                            score,
                            distance,
                            modality: target_modality,
                            metadata: entry.metadata.clone(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Sort by score and return top-k
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(query.limit);

        Ok(results)
    }

    /// Late fusion: search each modality separately, then merge results
    fn late_fusion_search(
        &self,
        query: &MultiModalQuery,
        weights: &HashMap<Modality, f32>,
    ) -> Result<Vec<MultiModalResult>> {
        let mut all_results: HashMap<String, (f32, MultiModalResult)> = HashMap::new();

        // Search each query modality
        for (query_modality, query_embedding) in &query.embeddings {
            let weight = weights.get(query_modality).copied().unwrap_or(1.0);

            // Find entries with matching modality
            for entry in &self.entries {
                if let Some(entry_embedding) = entry.get_embedding(*query_modality) {
                    // Apply metadata filter
                    if let Some(ref filter) = query.filter {
                        if let Some(ref entry_meta) = entry.metadata {
                            if !filter.iter().all(|(k, v)| entry_meta.get(k) == Some(v)) {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }

                    let distance = euclidean_distance(query_embedding, entry_embedding);
                    let score = (1.0 / (1.0 + distance)) * weight;

                    // Accumulate scores for same ID
                    all_results
                        .entry(entry.id.clone())
                        .and_modify(|(accumulated_score, _)| *accumulated_score += score)
                        .or_insert((
                            score,
                            MultiModalResult {
                                id: entry.id.clone(),
                                score,
                                distance,
                                modality: *query_modality,
                                metadata: entry.metadata.clone(),
                            },
                        ));
                }
            }
        }

        // Extract and sort results
        let mut results: Vec<MultiModalResult> = all_results
            .into_iter()
            .map(|(_, (accumulated_score, mut result))| {
                result.score = accumulated_score;
                result
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(query.limit);

        Ok(results)
    }

    /// Fuse query embeddings according to strategy
    fn fuse_query_embeddings(
        &self,
        query_embeddings: &HashMap<Modality, Vec<f32>>,
        strategy: &MultiModalFusion,
        target_modality: Modality,
    ) -> Result<Vec<f32>> {
        if query_embeddings.len() == 1 {
            // Single modality query - return as-is (may need projection)
            let (_, embedding) = query_embeddings.iter().next().unwrap();
            return Ok(embedding.clone());
        }

        match strategy {
            MultiModalFusion::Average => {
                // Average all embeddings (assume same dimension)
                let embeddings: Vec<&Vec<f32>> = query_embeddings.values().collect();
                let dim = embeddings[0].len();

                let mut fused = vec![0.0; dim];
                for embedding in &embeddings {
                    for (i, &val) in embedding.iter().enumerate() {
                        fused[i] += val;
                    }
                }

                let n = embeddings.len() as f32;
                for val in &mut fused {
                    *val /= n;
                }

                Ok(fused)
            }

            MultiModalFusion::WeightedAverage(weights) => {
                let embeddings: Vec<&Vec<f32>> = query_embeddings.values().collect();
                let dim = embeddings[0].len();

                let mut fused = vec![0.0; dim];
                for (embedding, &weight) in embeddings.iter().zip(weights.iter()) {
                    for (i, &val) in embedding.iter().enumerate() {
                        fused[i] += val * weight;
                    }
                }

                Ok(fused)
            }

            MultiModalFusion::Concatenate => {
                // Concatenate all embeddings
                let mut fused = Vec::new();
                for embedding in query_embeddings.values() {
                    fused.extend_from_slice(embedding);
                }
                Ok(fused)
            }

            MultiModalFusion::Max => {
                let embeddings: Vec<&Vec<f32>> = query_embeddings.values().collect();
                let dim = embeddings[0].len();

                let mut fused = vec![f32::NEG_INFINITY; dim];
                for embedding in &embeddings {
                    for (i, &val) in embedding.iter().enumerate() {
                        fused[i] = fused[i].max(val);
                    }
                }

                Ok(fused)
            }

            MultiModalFusion::LateFusion { .. } => {
                // Should not reach here (handled separately)
                Err(anyhow!("Late fusion not applicable in early fusion search"))
            }
        }
    }

    /// Remove entry by ID
    pub fn remove(&mut self, id: &str) -> Result<bool> {
        if let Some(pos) = self.entries.iter().position(|e| e.id == id) {
            self.entries.remove(pos);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get statistics
    pub fn stats(&self) -> MultiModalStats {
        let mut modality_counts: HashMap<Modality, usize> = HashMap::new();

        for entry in &self.entries {
            for modality in entry.embeddings.keys() {
                *modality_counts.entry(*modality).or_insert(0) += 1;
            }
        }

        MultiModalStats {
            total_entries: self.entries.len(),
            modality_counts,
            dimensions: self.dimensions.clone(),
        }
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for MultiModalIndex {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Multi-modal index statistics
#[derive(Debug, Clone)]
pub struct MultiModalStats {
    pub total_entries: usize,
    pub modality_counts: HashMap<Modality, usize>,
    pub dimensions: HashMap<Modality, usize>,
}

/// Helper: Euclidean distance
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_basic() {
        let mut index = MultiModalIndex::new().unwrap();

        // Add text entry
        index
            .add_with_modality("doc1", vec![0.1, 0.2, 0.3], Modality::Text, None)
            .unwrap();

        // Add image entry
        index
            .add_with_modality("img1", vec![0.4, 0.5, 0.6], Modality::Image, None)
            .unwrap();

        assert_eq!(index.len(), 2);

        // Search text for text
        let query = MultiModalQuery::new(vec![0.1, 0.2, 0.3], Modality::Text);
        let results = index.search(&query).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_cross_modal_search() {
        let mut index = MultiModalIndex::new().unwrap();

        // Add entries
        index
            .add_with_modality("doc1", vec![0.1; 64], Modality::Text, None)
            .unwrap();
        index
            .add_with_modality("img1", vec![0.2; 64], Modality::Image, None)
            .unwrap();
        index
            .add_with_modality("img2", vec![0.3; 64], Modality::Image, None)
            .unwrap();

        // Cross-modal: text query to find images
        let query = MultiModalQuery::new(vec![0.25; 64], Modality::Text)
            .with_target_modality(Modality::Image)
            .with_limit(10);

        let results = index.search(&query).unwrap();

        // Should only return images
        assert!(!results.is_empty());
        for result in &results {
            assert_eq!(result.modality, Modality::Image);
        }
    }

    #[test]
    fn test_multimodal_entry() {
        let entry = MultiModalEntry::new("item1")
            .add_embedding(Modality::Text, vec![0.1, 0.2])
            .add_embedding(Modality::Image, vec![0.3, 0.4]);

        assert!(entry.has_modality(Modality::Text));
        assert!(entry.has_modality(Modality::Image));
        assert!(!entry.has_modality(Modality::Audio));

        assert_eq!(entry.get_embedding(Modality::Text), Some(&vec![0.1, 0.2]));
    }

    #[test]
    fn test_late_fusion() {
        let mut index = MultiModalIndex::new().unwrap();

        // Add multi-modal entries
        let entry1 = MultiModalEntry::new("item1")
            .add_embedding(Modality::Text, vec![0.1; 32])
            .add_embedding(Modality::Image, vec![0.2; 32]);

        let entry2 = MultiModalEntry::new("item2")
            .add_embedding(Modality::Text, vec![0.5; 32])
            .add_embedding(Modality::Image, vec![0.6; 32]);

        index.add_entry(entry1).unwrap();
        index.add_entry(entry2).unwrap();

        // Query with both modalities
        let mut query_embeddings = HashMap::new();
        query_embeddings.insert(Modality::Text, vec![0.1; 32]);
        query_embeddings.insert(Modality::Image, vec![0.2; 32]);

        let mut weights = HashMap::new();
        weights.insert(Modality::Text, 0.7);
        weights.insert(Modality::Image, 0.3);

        let query = MultiModalQuery::multi(query_embeddings)
            .with_fusion(MultiModalFusion::LateFusion { weights })
            .with_limit(10);

        let results = index.search(&query).unwrap();

        assert!(!results.is_empty());
        // item1 should rank higher due to similarity on both modalities
        assert_eq!(results[0].id, "item1");
    }

    #[test]
    fn test_metadata_filter() {
        let mut index = MultiModalIndex::new().unwrap();

        let mut meta1 = HashMap::new();
        meta1.insert("category".to_string(), serde_json::json!("tech"));

        let mut meta2 = HashMap::new();
        meta2.insert("category".to_string(), serde_json::json!("sports"));

        index
            .add_with_modality("doc1", vec![0.1; 32], Modality::Text, Some(meta1))
            .unwrap();
        index
            .add_with_modality("doc2", vec![0.2; 32], Modality::Text, Some(meta2))
            .unwrap();

        // Filter for tech category only
        let mut filter = HashMap::new();
        filter.insert("category".to_string(), serde_json::json!("tech"));

        let query = MultiModalQuery::new(vec![0.15; 32], Modality::Text).with_filter(filter);

        let results = index.search(&query).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_remove() {
        let mut index = MultiModalIndex::new().unwrap();

        index
            .add_with_modality("doc1", vec![0.1; 32], Modality::Text, None)
            .unwrap();
        index
            .add_with_modality("doc2", vec![0.2; 32], Modality::Text, None)
            .unwrap();

        assert_eq!(index.len(), 2);

        let removed = index.remove("doc1").unwrap();
        assert!(removed);
        assert_eq!(index.len(), 1);

        let removed = index.remove("doc1").unwrap();
        assert!(!removed);
    }

    #[test]
    fn test_stats() {
        let mut index = MultiModalIndex::new().unwrap();

        index
            .add_with_modality("doc1", vec![0.1; 64], Modality::Text, None)
            .unwrap();
        index
            .add_with_modality("img1", vec![0.2; 128], Modality::Image, None)
            .unwrap();
        index
            .add_with_modality("img2", vec![0.3; 128], Modality::Image, None)
            .unwrap();

        let stats = index.stats();

        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.modality_counts.get(&Modality::Text), Some(&1));
        assert_eq!(stats.modality_counts.get(&Modality::Image), Some(&2));
        assert_eq!(stats.dimensions.get(&Modality::Text), Some(&64));
        assert_eq!(stats.dimensions.get(&Modality::Image), Some(&128));
    }

    #[test]
    fn test_fusion_strategies() {
        let mut index = MultiModalIndex::new().unwrap();

        // Add entry with both modalities
        let entry = MultiModalEntry::new("item1")
            .add_embedding(Modality::Text, vec![0.5; 64])
            .add_embedding(Modality::Image, vec![0.5; 64]);

        index.add_entry(entry).unwrap();

        // Test average fusion
        let mut query_embeddings = HashMap::new();
        query_embeddings.insert(Modality::Text, vec![0.6; 64]);
        query_embeddings.insert(Modality::Image, vec![0.4; 64]);

        let query =
            MultiModalQuery::multi(query_embeddings.clone()).with_fusion(MultiModalFusion::Average);

        let results = index.search(&query).unwrap();
        assert!(!results.is_empty());

        // Test weighted average
        let weights = vec![0.7, 0.3];
        let query = MultiModalQuery::multi(query_embeddings)
            .with_fusion(MultiModalFusion::WeightedAverage(weights));

        let results = index.search(&query).unwrap();
        assert!(!results.is_empty());
    }
}
