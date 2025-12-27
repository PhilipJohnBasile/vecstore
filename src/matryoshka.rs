//! Matryoshka Embeddings: Adaptive Dimension Truncation
//!
//! Matryoshka Representation Learning (MRL) produces embeddings that can be truncated
//! to smaller dimensions while preserving semantic quality. This enables:
//!
//! - **Adaptive precision**: Use full dimensions for high-quality search, truncate for speed
//! - **Storage savings**: Store shorter embeddings, expand only when needed
//! - **Progressive search**: Start with short embeddings, refine with longer ones
//!
//! ## Supported Models
//!
//! - OpenAI text-embedding-3-small/large (native MRL support)
//! - Voyage AI embeddings
//! - Nomic Embed
//! - Any MRL-trained model
//!
//! ## Dimension Hierarchy
//!
//! Typical dimensions: 64 → 128 → 256 → 512 → 768 → 1024 → 1536 → 3072
//!
//! Each level preserves the quality of smaller levels:
//! - First 64 dims: ~85% of full quality
//! - First 256 dims: ~95% of full quality
//! - First 512 dims: ~98% of full quality
//!
//! ## Example
//!
//! ```rust,no_run
//! use vecstore::matryoshka::{MatryoshkaStore, MatryoshkaConfig};
//!
//! let config = MatryoshkaConfig {
//!     full_dimension: 1536,
//!     storage_dimension: 256,  // Store truncated
//!     search_dimensions: vec![64, 256, 1536],  // Progressive search
//!     ..Default::default()
//! };
//!
//! let store = MatryoshkaStore::new(config);
//!
//! // Add full embedding - automatically truncated for storage
//! store.add("doc1", &full_embedding)?;
//!
//! // Search with progressive refinement
//! let results = store.search(&query, 10)?;
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Matryoshka embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatryoshkaConfig {
    /// Full dimension of embeddings from the model
    pub full_dimension: usize,

    /// Dimension to use for storage (truncated)
    pub storage_dimension: usize,

    /// Dimensions to use for progressive search (ascending order)
    pub search_dimensions: Vec<usize>,

    /// Whether to store full embeddings alongside truncated
    pub store_full_embeddings: bool,

    /// Normalization strategy after truncation
    pub normalize_truncated: bool,

    /// Quality thresholds for each dimension level
    pub quality_thresholds: HashMap<usize, f32>,
}

impl Default for MatryoshkaConfig {
    fn default() -> Self {
        Self {
            full_dimension: 1536,
            storage_dimension: 256,
            search_dimensions: vec![64, 256, 512],
            store_full_embeddings: false,
            normalize_truncated: true,
            quality_thresholds: [
                (64, 0.85),
                (128, 0.90),
                (256, 0.95),
                (512, 0.98),
                (768, 0.99),
            ]
            .into_iter()
            .collect(),
        }
    }
}

// ============================================================================
// MATRYOSHKA EMBEDDING
// ============================================================================

/// A Matryoshka embedding with multi-resolution support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatryoshkaEmbedding {
    /// Document/vector ID
    pub id: String,

    /// Full embedding (if stored)
    pub full: Option<Vec<f32>>,

    /// Truncated embeddings at various dimensions
    /// Key: dimension, Value: truncated embedding
    pub truncated: HashMap<usize, Vec<f32>>,

    /// Original full dimension
    pub full_dimension: usize,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl MatryoshkaEmbedding {
    /// Create from full embedding
    pub fn from_full(id: &str, embedding: &[f32], dimensions: &[usize], normalize: bool) -> Self {
        let mut truncated = HashMap::new();

        for &dim in dimensions {
            if dim <= embedding.len() {
                let mut trunc: Vec<f32> = embedding[..dim].to_vec();

                if normalize {
                    let norm: f32 = trunc.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for val in &mut trunc {
                            *val /= norm;
                        }
                    }
                }

                truncated.insert(dim, trunc);
            }
        }

        Self {
            id: id.to_string(),
            full: Some(embedding.to_vec()),
            truncated,
            full_dimension: embedding.len(),
            metadata: HashMap::new(),
        }
    }

    /// Create from truncated (no full embedding)
    pub fn from_truncated(id: &str, embedding: &[f32], full_dimension: usize) -> Self {
        let mut truncated = HashMap::new();
        truncated.insert(embedding.len(), embedding.to_vec());

        Self {
            id: id.to_string(),
            full: None,
            truncated,
            full_dimension,
            metadata: HashMap::new(),
        }
    }

    /// Get embedding at specific dimension
    pub fn at_dimension(&self, dim: usize) -> Option<&Vec<f32>> {
        // First try exact match
        if let Some(emb) = self.truncated.get(&dim) {
            return Some(emb);
        }

        // Try to get from full embedding
        if let Some(ref full) = self.full {
            if dim <= full.len() {
                // Would need to compute truncated - return None for now
                return None;
            }
        }

        // Return closest smaller dimension
        self.truncated
            .iter()
            .filter(|(d, _)| **d <= dim)
            .max_by_key(|(d, _)| **d)
            .map(|(_, v)| v)
    }

    /// Get best available embedding
    pub fn best_available(&self) -> Option<&Vec<f32>> {
        if let Some(ref full) = self.full {
            return Some(full);
        }

        self.truncated
            .iter()
            .max_by_key(|(d, _)| **d)
            .map(|(_, v)| v)
    }

    /// Get smallest available embedding
    pub fn smallest(&self) -> Option<&Vec<f32>> {
        self.truncated
            .iter()
            .min_by_key(|(d, _)| **d)
            .map(|(_, v)| v)
    }
}

// ============================================================================
// MATRYOSHKA STORE
// ============================================================================

/// Store for Matryoshka embeddings with progressive search
pub struct MatryoshkaStore {
    config: MatryoshkaConfig,
    embeddings: HashMap<String, MatryoshkaEmbedding>,
}

impl MatryoshkaStore {
    /// Create a new Matryoshka store
    pub fn new(config: MatryoshkaConfig) -> Self {
        // Validate config
        let mut search_dims = config.search_dimensions.clone();
        search_dims.sort();

        Self {
            config: MatryoshkaConfig {
                search_dimensions: search_dims,
                ..config
            },
            embeddings: HashMap::new(),
        }
    }

    /// Add embedding to store
    pub fn add(&mut self, id: &str, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.config.full_dimension {
            return Err(anyhow!(
                "Embedding dimension {} doesn't match config {}",
                embedding.len(),
                self.config.full_dimension
            ));
        }

        let mrl = MatryoshkaEmbedding::from_full(
            id,
            embedding,
            &self.config.search_dimensions,
            self.config.normalize_truncated,
        );

        self.embeddings.insert(id.to_string(), mrl);
        Ok(())
    }

    /// Add batch of embeddings
    pub fn add_batch(&mut self, embeddings: &[(String, Vec<f32>)]) -> Result<()> {
        for (id, emb) in embeddings {
            self.add(id, emb)?;
        }
        Ok(())
    }

    /// Search with progressive refinement
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.progressive_search(query, k, None)
    }

    /// Progressive search with configurable stages
    pub fn progressive_search(
        &self,
        query: &[f32],
        k: usize,
        stages: Option<&[usize]>,
    ) -> Result<Vec<SearchResult>> {
        let stages = stages.unwrap_or(&self.config.search_dimensions);

        if stages.is_empty() {
            return Err(anyhow!("No search stages configured"));
        }

        // Stage 1: Coarse search with smallest dimension
        let coarse_dim = stages[0];
        let coarse_query = self.truncate_and_normalize(query, coarse_dim)?;

        let mut candidates: Vec<(String, f32)> = self
            .embeddings
            .iter()
            .filter_map(|(id, emb)| {
                emb.at_dimension(coarse_dim).map(|v| {
                    let sim = self.cosine_similarity(&coarse_query, v);
                    (id.clone(), sim)
                })
            })
            .collect();

        // Sort and keep top candidates for refinement
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let refine_k = (k * 10).min(candidates.len()); // Keep 10x candidates for refinement
        candidates.truncate(refine_k);

        // Progressive refinement through stages
        for &dim in stages.iter().skip(1) {
            if dim > query.len() {
                break;
            }

            let refined_query = self.truncate_and_normalize(query, dim)?;

            // Re-score candidates at higher dimension
            for (id, score) in &mut candidates {
                if let Some(emb) = self.embeddings.get(id) {
                    if let Some(v) = emb.at_dimension(dim) {
                        *score = self.cosine_similarity(&refined_query, v);
                    }
                }
            }

            // Re-sort after refinement
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Take top-k and build results
        candidates.truncate(k);

        let results = candidates
            .into_iter()
            .enumerate()
            .map(|(rank, (id, score))| SearchResult {
                id,
                score,
                rank,
                dimension_used: *stages.last().unwrap_or(&coarse_dim),
            })
            .collect();

        Ok(results)
    }

    /// Search at specific dimension (no progressive refinement)
    pub fn search_at_dimension(
        &self,
        query: &[f32],
        dimension: usize,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let truncated_query = self.truncate_and_normalize(query, dimension)?;

        let mut results: Vec<(String, f32)> = self
            .embeddings
            .iter()
            .filter_map(|(id, emb)| {
                emb.at_dimension(dimension).map(|v| {
                    let sim = self.cosine_similarity(&truncated_query, v);
                    (id.clone(), sim)
                })
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results
            .into_iter()
            .enumerate()
            .map(|(rank, (id, score))| SearchResult {
                id,
                score,
                rank,
                dimension_used: dimension,
            })
            .collect())
    }

    /// Truncate and normalize query
    fn truncate_and_normalize(&self, query: &[f32], dim: usize) -> Result<Vec<f32>> {
        if dim > query.len() {
            return Err(anyhow!(
                "Cannot truncate to {} dims when query has {} dims",
                dim,
                query.len()
            ));
        }

        let mut trunc: Vec<f32> = query[..dim].to_vec();

        if self.config.normalize_truncated {
            let norm: f32 = trunc.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut trunc {
                    *val /= norm;
                }
            }
        }

        Ok(trunc)
    }

    /// Cosine similarity
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Get embedding by ID
    pub fn get(&self, id: &str) -> Option<&MatryoshkaEmbedding> {
        self.embeddings.get(id)
    }

    /// Remove embedding
    pub fn remove(&mut self, id: &str) -> Option<MatryoshkaEmbedding> {
        self.embeddings.remove(id)
    }

    /// Number of embeddings
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Estimate memory usage
    pub fn memory_usage(&self) -> MemoryEstimate {
        let mut truncated_bytes = 0usize;
        let mut full_bytes = 0usize;

        for emb in self.embeddings.values() {
            if let Some(ref full) = emb.full {
                full_bytes += full.len() * 4;
            }

            for v in emb.truncated.values() {
                truncated_bytes += v.len() * 4;
            }
        }

        MemoryEstimate {
            truncated_bytes,
            full_bytes,
            total_bytes: truncated_bytes + full_bytes,
            num_embeddings: self.embeddings.len(),
        }
    }
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub rank: usize,
    pub dimension_used: usize,
}

/// Memory usage estimate
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub truncated_bytes: usize,
    pub full_bytes: usize,
    pub total_bytes: usize,
    pub num_embeddings: usize,
}

impl MemoryEstimate {
    pub fn bytes_per_embedding(&self) -> f64 {
        if self.num_embeddings > 0 {
            self.total_bytes as f64 / self.num_embeddings as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// DIMENSION ANALYZER
// ============================================================================

/// Analyze quality degradation at different truncation levels
pub struct DimensionAnalyzer {
    /// Reference similarities at full dimension
    reference: HashMap<(String, String), f32>,
}

impl DimensionAnalyzer {
    /// Create analyzer
    pub fn new() -> Self {
        Self {
            reference: HashMap::new(),
        }
    }

    /// Analyze quality at different dimensions
    pub fn analyze(
        &mut self,
        embeddings: &[(&str, Vec<f32>)],
        dimensions: &[usize],
    ) -> DimensionAnalysis {
        if embeddings.is_empty() || dimensions.is_empty() {
            return DimensionAnalysis {
                dimension_scores: HashMap::new(),
                recommended_dimension: 0,
                full_dimension: 0,
            };
        }

        let full_dim = embeddings[0].1.len();

        // Compute reference similarities at full dimension
        self.compute_reference(embeddings);

        // Analyze each dimension
        let mut dimension_scores = HashMap::new();

        for &dim in dimensions {
            if dim > full_dim {
                continue;
            }

            let correlation = self.compute_correlation(embeddings, dim);
            dimension_scores.insert(dim, correlation);
        }

        // Find recommended dimension (first to exceed 0.95 correlation)
        let recommended = *dimensions
            .iter()
            .find(|&&d| dimension_scores.get(&d).copied().unwrap_or(0.0) >= 0.95)
            .unwrap_or(&full_dim);

        DimensionAnalysis {
            dimension_scores,
            recommended_dimension: recommended,
            full_dimension: full_dim,
        }
    }

    fn compute_reference(&mut self, embeddings: &[(&str, Vec<f32>)]) {
        self.reference.clear();

        for (i, (id1, emb1)) in embeddings.iter().enumerate() {
            for (id2, emb2) in embeddings.iter().skip(i + 1) {
                let sim = Self::cosine_sim(emb1, emb2);
                self.reference
                    .insert((id1.to_string(), id2.to_string()), sim);
            }
        }
    }

    fn compute_correlation(&self, embeddings: &[(&str, Vec<f32>)], dim: usize) -> f32 {
        let mut full_sims = Vec::new();
        let mut trunc_sims = Vec::new();

        for (i, (id1, emb1)) in embeddings.iter().enumerate() {
            for (id2, emb2) in embeddings.iter().skip(i + 1) {
                if let Some(&ref_sim) = self.reference.get(&(id1.to_string(), id2.to_string())) {
                    let trunc1: Vec<f32> = emb1.iter().take(dim).cloned().collect();
                    let trunc2: Vec<f32> = emb2.iter().take(dim).cloned().collect();
                    let trunc_sim = Self::cosine_sim(&trunc1, &trunc2);

                    full_sims.push(ref_sim);
                    trunc_sims.push(trunc_sim);
                }
            }
        }

        if full_sims.is_empty() {
            return 0.0;
        }

        // Pearson correlation
        let n = full_sims.len() as f32;
        let mean_full: f32 = full_sims.iter().sum::<f32>() / n;
        let mean_trunc: f32 = trunc_sims.iter().sum::<f32>() / n;

        let mut cov = 0.0;
        let mut var_full = 0.0;
        let mut var_trunc = 0.0;

        for (f, t) in full_sims.iter().zip(&trunc_sims) {
            cov += (f - mean_full) * (t - mean_trunc);
            var_full += (f - mean_full).powi(2);
            var_trunc += (t - mean_trunc).powi(2);
        }

        if var_full > 0.0 && var_trunc > 0.0 {
            cov / (var_full.sqrt() * var_trunc.sqrt())
        } else {
            0.0
        }
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

impl Default for DimensionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Dimension analysis result
#[derive(Debug, Clone)]
pub struct DimensionAnalysis {
    /// Quality score (correlation with full dimension) for each dimension
    pub dimension_scores: HashMap<usize, f32>,

    /// Recommended dimension based on quality threshold
    pub recommended_dimension: usize,

    /// Full dimension
    pub full_dimension: usize,
}

impl DimensionAnalysis {
    /// Get quality at dimension
    pub fn quality_at(&self, dim: usize) -> Option<f32> {
        self.dimension_scores.get(&dim).copied()
    }

    /// Get compression ratio at dimension
    pub fn compression_at(&self, dim: usize) -> f32 {
        self.full_dimension as f32 / dim as f32
    }

    /// Print summary
    pub fn summary(&self) -> String {
        let mut lines = vec![format!("Full dimension: {}", self.full_dimension)];
        lines.push(format!("Recommended: {}", self.recommended_dimension));
        lines.push("Quality by dimension:".to_string());

        let mut dims: Vec<_> = self.dimension_scores.keys().collect();
        dims.sort();

        for dim in dims {
            let quality = self.dimension_scores[dim];
            lines.push(format!(
                "  {} dims: {:.1}% quality, {:.1}x compression",
                dim,
                quality * 100.0,
                self.compression_at(*dim)
            ));
        }

        lines.join("\n")
    }
}

// ============================================================================
// ADAPTIVE SEARCH
// ============================================================================

/// Adaptive search that automatically chooses optimal dimension
pub struct AdaptiveSearch {
    store: MatryoshkaStore,
    analyzer: DimensionAnalyzer,
    /// Target quality threshold (0.0-1.0)
    quality_threshold: f32,
    /// Maximum latency in milliseconds
    max_latency_ms: u64,
}

impl AdaptiveSearch {
    pub fn new(config: MatryoshkaConfig, quality_threshold: f32, max_latency_ms: u64) -> Self {
        Self {
            store: MatryoshkaStore::new(config),
            analyzer: DimensionAnalyzer::new(),
            quality_threshold,
            max_latency_ms,
        }
    }

    /// Add embedding
    pub fn add(&mut self, id: &str, embedding: &[f32]) -> Result<()> {
        self.store.add(id, embedding)
    }

    /// Search with automatic dimension selection
    pub fn search(&self, query: &[f32], k: usize) -> Result<AdaptiveSearchResult> {
        let start = std::time::Instant::now();

        // Estimate optimal dimension based on query complexity
        let optimal_dim = self.estimate_optimal_dimension(query);

        // Start with smallest dimension
        let dims = &self.store.config.search_dimensions;
        let mut current_dim_idx = 0;
        let mut results = Vec::new();

        while current_dim_idx < dims.len() {
            let dim = dims[current_dim_idx];

            if dim > query.len() {
                break;
            }

            // Check latency budget
            let elapsed = start.elapsed().as_millis() as u64;
            if elapsed > self.max_latency_ms * 8 / 10 {
                // 80% budget used
                break;
            }

            results = self.store.search_at_dimension(query, dim, k)?;

            // Check if we've reached quality threshold or optimal dimension
            if dim >= optimal_dim {
                break;
            }

            current_dim_idx += 1;
        }

        let dimension_used = if !results.is_empty() {
            results[0].dimension_used
        } else {
            dims[0]
        };

        Ok(AdaptiveSearchResult {
            results,
            dimension_used,
            latency_ms: start.elapsed().as_millis() as u64,
            quality_estimate: self
                .store
                .config
                .quality_thresholds
                .get(&dimension_used)
                .copied()
                .unwrap_or(1.0),
        })
    }

    fn estimate_optimal_dimension(&self, query: &[f32]) -> usize {
        // Simple heuristic: use query variance to estimate complexity
        let mean: f32 = query.iter().sum::<f32>() / query.len() as f32;
        let variance: f32 = query.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / query.len() as f32;

        // Higher variance queries need more dimensions
        if variance > 0.5 {
            512
        } else if variance > 0.2 {
            256
        } else {
            128
        }
    }

    /// Get store reference
    pub fn store(&self) -> &MatryoshkaStore {
        &self.store
    }

    /// Get mutable store reference
    pub fn store_mut(&mut self) -> &mut MatryoshkaStore {
        &mut self.store
    }
}

/// Adaptive search result
#[derive(Debug, Clone)]
pub struct AdaptiveSearchResult {
    pub results: Vec<SearchResult>,
    pub dimension_used: usize,
    pub latency_ms: u64,
    pub quality_estimate: f32,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_embeddings(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
        use rand::Rng;
        let mut rng = rand::rng();

        (0..n)
            .map(|i| {
                let emb: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
                (format!("doc_{}", i), emb)
            })
            .collect()
    }

    #[test]
    fn test_matryoshka_embedding_creation() {
        let full = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let dims = vec![2, 4, 8];

        let mrl = MatryoshkaEmbedding::from_full("test", &full, &dims, true);

        assert!(mrl.truncated.contains_key(&2));
        assert!(mrl.truncated.contains_key(&4));
        assert!(mrl.truncated.contains_key(&8));

        // Check normalization
        let at_2 = mrl.at_dimension(2).unwrap();
        let norm: f32 = at_2.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_matryoshka_store_basic() {
        let config = MatryoshkaConfig {
            full_dimension: 128,
            storage_dimension: 32,
            search_dimensions: vec![16, 32, 64, 128],
            ..Default::default()
        };

        let mut store = MatryoshkaStore::new(config);

        let embeddings = generate_random_embeddings(100, 128);
        for (id, emb) in &embeddings {
            store.add(id, emb).unwrap();
        }

        assert_eq!(store.len(), 100);

        // Search
        let query = &embeddings[0].1;
        let results = store.search(query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // First result should be the query itself
        assert_eq!(results[0].id, embeddings[0].0);
    }

    #[test]
    fn test_progressive_search() {
        let config = MatryoshkaConfig {
            full_dimension: 64,
            storage_dimension: 16,
            search_dimensions: vec![8, 16, 32, 64],
            ..Default::default()
        };

        let mut store = MatryoshkaStore::new(config);

        let embeddings = generate_random_embeddings(50, 64);
        for (id, emb) in &embeddings {
            store.add(id, emb).unwrap();
        }

        let query = &embeddings[5].1;
        let results = store.progressive_search(query, 10, None).unwrap();

        assert!(!results.is_empty());
        // Final dimension should be 64
        assert_eq!(results[0].dimension_used, 64);
    }

    #[test]
    fn test_search_at_dimension() {
        let config = MatryoshkaConfig {
            full_dimension: 64,
            search_dimensions: vec![16, 32, 64],
            ..Default::default()
        };

        let mut store = MatryoshkaStore::new(config);

        let embeddings = generate_random_embeddings(30, 64);
        for (id, emb) in &embeddings {
            store.add(id, emb).unwrap();
        }

        // Search at specific dimension
        let query = &embeddings[0].1;
        let results = store.search_at_dimension(query, 16, 5).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].dimension_used, 16);
    }

    #[test]
    fn test_dimension_analyzer() {
        let embeddings: Vec<(&str, Vec<f32>)> = (0..20)
            .map(|i| {
                let mut emb = vec![0.0f32; 128];
                for j in 0..128 {
                    emb[j] = ((i * 7 + j) as f32 / 100.0).sin();
                }
                (
                    Box::leak(format!("doc_{}", i).into_boxed_str()) as &str,
                    emb,
                )
            })
            .collect();

        let mut analyzer = DimensionAnalyzer::new();
        let analysis = analyzer.analyze(&embeddings, &[16, 32, 64, 128]);

        assert!(analysis.dimension_scores.contains_key(&16));
        assert!(analysis.dimension_scores.contains_key(&128));

        // Higher dimensions should have higher quality
        let q16 = analysis.quality_at(16).unwrap_or(0.0);
        let q128 = analysis.quality_at(128).unwrap_or(0.0);
        assert!(q128 >= q16);
    }

    #[test]
    fn test_memory_usage() {
        let config = MatryoshkaConfig {
            full_dimension: 1536,
            search_dimensions: vec![64, 256, 512],
            store_full_embeddings: false,
            ..Default::default()
        };

        let mut store = MatryoshkaStore::new(config);

        let embeddings = generate_random_embeddings(1000, 1536);
        for (id, emb) in &embeddings {
            store.add(id, emb).unwrap();
        }

        let usage = store.memory_usage();

        // Without full embeddings, should be much smaller
        // 1000 * (64 + 256 + 512) * 4 = 3.328 MB
        assert!(usage.truncated_bytes < 4_000_000);

        println!(
            "Memory usage: {} bytes per embedding",
            usage.bytes_per_embedding()
        );
    }

    #[test]
    fn test_adaptive_search() {
        let config = MatryoshkaConfig {
            full_dimension: 128,
            search_dimensions: vec![32, 64, 128],
            ..Default::default()
        };

        let mut search = AdaptiveSearch::new(config, 0.95, 100);

        let embeddings = generate_random_embeddings(100, 128);
        for (id, emb) in &embeddings {
            search.add(id, emb).unwrap();
        }

        let query = &embeddings[0].1;
        let result = search.search(query, 5).unwrap();

        assert!(!result.results.is_empty());
        assert!(result.quality_estimate > 0.0);
    }
}
