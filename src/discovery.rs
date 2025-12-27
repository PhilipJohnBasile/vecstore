// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Discovery API
//!
//! Vector-only search constraints for exploring and navigating the embedding space
//! without requiring metadata filters. Inspired by Qdrant's Discovery API.
//!
//! ## Features
//!
//! - **Context Search**: Find vectors in a specific region of the embedding space
//! - **Positive/Negative Examples**: Guide search with example vectors
//! - **Discover Mode**: Explore dissimilar regions from known points
//! - **Recommend Mode**: Find similar items with diversity
//! - **Clustering-Based Discovery**: Navigate semantic clusters
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::discovery::{DiscoveryIndex, DiscoveryQuery};
//!
//! let index = DiscoveryIndex::new(config);
//!
//! // Context search: find vectors near these positive examples, away from negatives
//! let results = index.discover(
//!     DiscoveryQuery::new()
//!         .with_positive(&["doc1", "doc2"])
//!         .with_negative(&["doc5"])
//!         .with_top_k(10)
//! )?;
//!
//! // Exploration: find diverse results in a region
//! let explored = index.explore("doc1", 0.3, 20)?;
//! ```

use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Discovery query
#[derive(Debug, Clone)]
pub struct DiscoveryQuery {
    /// Positive example IDs (search toward these)
    pub positive_ids: Vec<String>,
    /// Positive example vectors
    pub positive_vectors: Vec<Vec<f32>>,
    /// Negative example IDs (search away from these)
    pub negative_ids: Vec<String>,
    /// Negative example vectors
    pub negative_vectors: Vec<Vec<f32>>,
    /// Target vector (optional, for combined search)
    pub target: Option<Vec<f32>>,
    /// Number of results
    pub top_k: usize,
    /// Negative weight (how much to penalize similarity to negatives)
    pub negative_weight: f32,
    /// Positive weight
    pub positive_weight: f32,
    /// Diversity factor (0 = no diversity, 1 = max diversity)
    pub diversity: f32,
    /// Search mode
    pub mode: DiscoveryMode,
}

impl DiscoveryQuery {
    /// Create new discovery query
    pub fn new() -> Self {
        Self {
            positive_ids: Vec::new(),
            positive_vectors: Vec::new(),
            negative_ids: Vec::new(),
            negative_vectors: Vec::new(),
            target: None,
            top_k: 10,
            negative_weight: 1.0,
            positive_weight: 1.0,
            diversity: 0.0,
            mode: DiscoveryMode::Context,
        }
    }

    /// Add positive examples by ID
    pub fn with_positive_ids(mut self, ids: &[&str]) -> Self {
        self.positive_ids = ids.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add positive example vectors
    pub fn with_positive_vectors(mut self, vectors: Vec<Vec<f32>>) -> Self {
        self.positive_vectors = vectors;
        self
    }

    /// Add negative examples by ID
    pub fn with_negative_ids(mut self, ids: &[&str]) -> Self {
        self.negative_ids = ids.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add negative example vectors
    pub fn with_negative_vectors(mut self, vectors: Vec<Vec<f32>>) -> Self {
        self.negative_vectors = vectors;
        self
    }

    /// Set target vector
    pub fn with_target(mut self, target: Vec<f32>) -> Self {
        self.target = Some(target);
        self
    }

    /// Set top-k
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set negative weight
    pub fn with_negative_weight(mut self, weight: f32) -> Self {
        self.negative_weight = weight;
        self
    }

    /// Set positive weight
    pub fn with_positive_weight(mut self, weight: f32) -> Self {
        self.positive_weight = weight;
        self
    }

    /// Set diversity factor
    pub fn with_diversity(mut self, diversity: f32) -> Self {
        self.diversity = diversity.clamp(0.0, 1.0);
        self
    }

    /// Set discovery mode
    pub fn with_mode(mut self, mode: DiscoveryMode) -> Self {
        self.mode = mode;
        self
    }
}

impl Default for DiscoveryQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Discovery mode
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiscoveryMode {
    /// Context search: find items similar to positives, dissimilar to negatives
    Context,
    /// Discover: explore new regions of the space
    Discover,
    /// Recommend: find similar with diversity
    Recommend,
    /// Contrast: maximize difference from negatives
    Contrast,
    /// Blend: weighted combination of all signals
    Blend,
}

/// Discovery result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResult {
    /// Vector ID
    pub id: String,
    /// Final score
    pub score: f32,
    /// Positive similarity (avg)
    pub positive_score: f32,
    /// Negative similarity (avg)
    pub negative_score: f32,
    /// Diversity score
    pub diversity_score: f32,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Exploration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationResult {
    /// Explored regions
    pub regions: Vec<Region>,
    /// Representative vectors per region
    pub representatives: HashMap<usize, Vec<String>>,
    /// Diversity score of exploration
    pub diversity_score: f32,
}

/// Region in the embedding space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    /// Region ID
    pub id: usize,
    /// Centroid vector
    pub centroid: Vec<f32>,
    /// Members
    pub members: Vec<String>,
    /// Radius (max distance from centroid)
    pub radius: f32,
    /// Density (members / volume)
    pub density: f32,
}

/// Discovery index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Dimension
    pub dimension: usize,
    /// Number of clusters for exploration
    pub num_clusters: usize,
    /// Enable MMR-style diversity
    pub enable_mmr: bool,
    /// Lambda for MMR
    pub mmr_lambda: f32,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            num_clusters: 10,
            enable_mmr: true,
            mmr_lambda: 0.5,
        }
    }
}

/// Discovery index
pub struct DiscoveryIndex {
    config: DiscoveryConfig,
    /// Vectors by ID
    vectors: RwLock<HashMap<String, Vec<f32>>>,
    /// Metadata by ID
    metadata: RwLock<HashMap<String, HashMap<String, serde_json::Value>>>,
    /// Cluster assignments
    clusters: RwLock<HashMap<String, usize>>,
    /// Cluster centroids
    centroids: RwLock<Vec<Vec<f32>>>,
}

impl DiscoveryIndex {
    /// Create new discovery index
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            config,
            vectors: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            clusters: RwLock::new(HashMap::new()),
            centroids: RwLock::new(Vec::new()),
        }
    }

    /// Insert vector
    pub fn insert(&self, id: &str, vector: &[f32], metadata: HashMap<String, serde_json::Value>) -> Result<()> {
        if vector.len() != self.config.dimension {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.config.dimension,
                got: vector.len(),
            });
        }

        let mut vectors = self.vectors.write().unwrap();
        vectors.insert(id.to_string(), vector.to_vec());

        let mut meta = self.metadata.write().unwrap();
        meta.insert(id.to_string(), metadata);

        Ok(())
    }

    /// Get vector by ID
    pub fn get(&self, id: &str) -> Option<Vec<f32>> {
        let vectors = self.vectors.read().unwrap();
        vectors.get(id).cloned()
    }

    /// Execute discovery query
    pub fn discover(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryResult>> {
        let vectors = self.vectors.read().unwrap();
        let metadata = self.metadata.read().unwrap();

        // Get positive vectors
        let positive_vecs: Vec<Vec<f32>> = query.positive_ids
            .iter()
            .filter_map(|id| vectors.get(id).cloned())
            .chain(query.positive_vectors.clone())
            .collect();

        // Get negative vectors
        let negative_vecs: Vec<Vec<f32>> = query.negative_ids
            .iter()
            .filter_map(|id| vectors.get(id).cloned())
            .chain(query.negative_vectors.clone())
            .collect();

        // Exclude positive and negative IDs from results
        let exclude_ids: HashSet<&str> = query.positive_ids.iter()
            .chain(&query.negative_ids)
            .map(|s| s.as_str())
            .collect();

        // Score all vectors
        let mut results: Vec<DiscoveryResult> = vectors
            .iter()
            .filter(|(id, _)| !exclude_ids.contains(id.as_str()))
            .map(|(id, vec)| {
                let pos_score = if !positive_vecs.is_empty() {
                    positive_vecs.iter()
                        .map(|pv| cosine_similarity(vec, pv))
                        .sum::<f32>() / positive_vecs.len() as f32
                } else {
                    0.0
                };

                let neg_score = if !negative_vecs.is_empty() {
                    negative_vecs.iter()
                        .map(|nv| cosine_similarity(vec, nv))
                        .sum::<f32>() / negative_vecs.len() as f32
                } else {
                    0.0
                };

                let target_score = query.target.as_ref()
                    .map(|t| cosine_similarity(vec, t))
                    .unwrap_or(0.0);

                // Compute final score based on mode
                let score = match query.mode {
                    DiscoveryMode::Context => {
                        pos_score * query.positive_weight - neg_score * query.negative_weight
                    }
                    DiscoveryMode::Discover => {
                        // Maximize novelty (low similarity to all examples)
                        let all_sim = (pos_score + neg_score) / 2.0;
                        1.0 - all_sim
                    }
                    DiscoveryMode::Recommend => {
                        pos_score * query.positive_weight
                    }
                    DiscoveryMode::Contrast => {
                        -neg_score * query.negative_weight
                    }
                    DiscoveryMode::Blend => {
                        pos_score * query.positive_weight
                            - neg_score * query.negative_weight
                            + target_score * 0.5
                    }
                };

                DiscoveryResult {
                    id: id.clone(),
                    score,
                    positive_score: pos_score,
                    negative_score: neg_score,
                    diversity_score: 0.0, // Will be set later if diversity enabled
                    metadata: metadata.get(id).cloned().unwrap_or_default(),
                }
            })
            .collect();

        // Sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        // Apply diversity if enabled
        if query.diversity > 0.0 && self.config.enable_mmr {
            results = self.apply_mmr(results, &vectors, query.diversity, query.top_k);
        } else {
            results.truncate(query.top_k);
        }

        Ok(results)
    }

    /// Apply MMR (Maximal Marginal Relevance) for diversity
    fn apply_mmr(
        &self,
        mut candidates: Vec<DiscoveryResult>,
        vectors: &HashMap<String, Vec<f32>>,
        diversity: f32,
        top_k: usize,
    ) -> Vec<DiscoveryResult> {
        let lambda = 1.0 - diversity;
        let mut selected: Vec<DiscoveryResult> = Vec::new();
        let mut selected_vecs: Vec<Vec<f32>> = Vec::new();

        while selected.len() < top_k && !candidates.is_empty() {
            let mut best_idx = 0;
            let mut best_mmr = f32::NEG_INFINITY;

            for (i, candidate) in candidates.iter().enumerate() {
                if let Some(vec) = vectors.get(&candidate.id) {
                    // Relevance component
                    let relevance = candidate.score;

                    // Diversity component (max similarity to already selected)
                    let diversity_penalty = if selected_vecs.is_empty() {
                        0.0
                    } else {
                        selected_vecs.iter()
                            .map(|sv| cosine_similarity(vec, sv))
                            .fold(f32::NEG_INFINITY, f32::max)
                    };

                    let mmr = lambda * relevance - (1.0 - lambda) * diversity_penalty;

                    if mmr > best_mmr {
                        best_mmr = mmr;
                        best_idx = i;
                    }
                }
            }

            let mut selected_item = candidates.remove(best_idx);
            selected_item.diversity_score = best_mmr;

            if let Some(vec) = vectors.get(&selected_item.id) {
                selected_vecs.push(vec.clone());
            }

            selected.push(selected_item);
        }

        selected
    }

    /// Explore from a starting point
    pub fn explore(&self, start_id: &str, radius: f32, max_results: usize) -> Result<Vec<DiscoveryResult>> {
        let vectors = self.vectors.read().unwrap();
        let metadata = self.metadata.read().unwrap();

        let start_vec = vectors.get(start_id)
            .ok_or_else(|| VecStoreError::NotFound(start_id.to_string()))?;

        let mut results: Vec<DiscoveryResult> = vectors
            .iter()
            .filter(|(id, _)| *id != start_id)
            .map(|(id, vec)| {
                let similarity = cosine_similarity(start_vec, vec);
                let distance = 1.0 - similarity;

                DiscoveryResult {
                    id: id.clone(),
                    score: similarity,
                    positive_score: similarity,
                    negative_score: 0.0,
                    diversity_score: distance,
                    metadata: metadata.get(id).cloned().unwrap_or_default(),
                }
            })
            .filter(|r| (1.0 - r.score) <= radius)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(max_results);

        Ok(results)
    }

    /// Find boundary vectors (transition points between clusters)
    pub fn find_boundaries(&self, num_boundaries: usize) -> Vec<DiscoveryResult> {
        let vectors = self.vectors.read().unwrap();
        let metadata = self.metadata.read().unwrap();

        if vectors.len() < 3 {
            return Vec::new();
        }

        // Compute average similarity for each vector
        let vec_list: Vec<(&String, &Vec<f32>)> = vectors.iter().collect();
        let mut boundary_scores: Vec<(String, f32)> = Vec::new();

        for (id, vec) in &vec_list {
            // Find k-nearest neighbors
            let mut similarities: Vec<f32> = vec_list
                .iter()
                .filter(|(other_id, _)| other_id != id)
                .map(|(_, other_vec)| cosine_similarity(vec, other_vec))
                .collect();

            similarities.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));

            // Boundary vectors have high variance in neighbor similarities
            let k = 5.min(similarities.len());
            if k > 0 {
                let top_k: Vec<f32> = similarities.into_iter().take(k).collect();
                let mean: f32 = top_k.iter().sum::<f32>() / k as f32;
                let variance: f32 = top_k.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / k as f32;

                // Boundary score = variance (high variance = boundary)
                boundary_scores.push((id.to_string(), variance));
            }
        }

        boundary_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        boundary_scores
            .into_iter()
            .take(num_boundaries)
            .map(|(id, score)| DiscoveryResult {
                id: id.clone(),
                score,
                positive_score: 0.0,
                negative_score: 0.0,
                diversity_score: score,
                metadata: metadata.get(&id).cloned().unwrap_or_default(),
            })
            .collect()
    }

    /// Find clusters and their representatives
    pub fn find_clusters(&self) -> ExplorationResult {
        let vectors = self.vectors.read().unwrap();

        if vectors.is_empty() {
            return ExplorationResult {
                regions: Vec::new(),
                representatives: HashMap::new(),
                diversity_score: 0.0,
            };
        }

        // Simple k-means clustering
        let k = self.config.num_clusters.min(vectors.len());
        let vec_list: Vec<(&String, &Vec<f32>)> = vectors.iter().collect();

        // Initialize centroids (first k vectors)
        let mut centroids: Vec<Vec<f32>> = vec_list
            .iter()
            .take(k)
            .map(|(_, v)| (*v).clone())
            .collect();

        // Run k-means for a few iterations
        let mut assignments: Vec<usize> = vec![0; vec_list.len()];

        for _ in 0..10 {
            // Assign each vector to nearest centroid
            for (i, (_, vec)) in vec_list.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_sim = f32::NEG_INFINITY;

                for (c, centroid) in centroids.iter().enumerate() {
                    let sim = cosine_similarity(vec, centroid);
                    if sim > best_sim {
                        best_sim = sim;
                        best_cluster = c;
                    }
                }

                assignments[i] = best_cluster;
            }

            // Update centroids
            let dim = self.config.dimension;
            let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
            let mut counts: Vec<usize> = vec![0; k];

            for (i, (_, vec)) in vec_list.iter().enumerate() {
                let cluster = assignments[i];
                for (j, &v) in vec.iter().enumerate() {
                    new_centroids[cluster][j] += v;
                }
                counts[cluster] += 1;
            }

            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        new_centroids[c][j] /= counts[c] as f32;
                    }
                    // Normalize
                    let norm: f32 = new_centroids[c].iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for x in &mut new_centroids[c] {
                            *x /= norm;
                        }
                    }
                }
            }

            centroids = new_centroids;
        }

        // Build regions
        let mut regions: Vec<Region> = Vec::new();
        let mut representatives: HashMap<usize, Vec<String>> = HashMap::new();

        for c in 0..k {
            let members: Vec<String> = vec_list
                .iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == c)
                .map(|(_, (id, _))| (*id).clone())
                .collect();

            if !members.is_empty() {
                // Calculate radius (max distance to centroid)
                let radius: f32 = members
                    .iter()
                    .filter_map(|id| vectors.get(id))
                    .map(|v| 1.0 - cosine_similarity(v, &centroids[c]))
                    .fold(0.0f32, f32::max);

                // Find representatives (closest to centroid)
                let mut member_sims: Vec<(String, f32)> = members
                    .iter()
                    .filter_map(|id| {
                        vectors.get(id).map(|v| {
                            (id.clone(), cosine_similarity(v, &centroids[c]))
                        })
                    })
                    .collect();

                member_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                let reps: Vec<String> = member_sims.into_iter().take(3).map(|(id, _)| id).collect();

                representatives.insert(c, reps);

                regions.push(Region {
                    id: c,
                    centroid: centroids[c].clone(),
                    members,
                    radius,
                    density: 0.0, // Could be calculated based on member distances
                });
            }
        }

        // Calculate diversity score (average inter-cluster distance)
        let mut total_dist = 0.0f32;
        let mut count = 0;

        for i in 0..regions.len() {
            for j in (i + 1)..regions.len() {
                total_dist += 1.0 - cosine_similarity(&regions[i].centroid, &regions[j].centroid);
                count += 1;
            }
        }

        let diversity_score = if count > 0 { total_dist / count as f32 } else { 0.0 };

        ExplorationResult {
            regions,
            representatives,
            diversity_score,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> DiscoveryStats {
        let vectors = self.vectors.read().unwrap();
        let clusters = self.clusters.read().unwrap();

        DiscoveryStats {
            vector_count: vectors.len(),
            dimension: self.config.dimension,
            cluster_count: clusters.values().collect::<HashSet<_>>().len(),
        }
    }
}

/// Discovery index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryStats {
    pub vector_count: usize,
    pub dimension: usize,
    pub cluster_count: usize,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovery_context() {
        let config = DiscoveryConfig { dimension: 4, ..Default::default() };
        let index = DiscoveryIndex::new(config);

        // Insert vectors
        index.insert("pos1", &[1.0, 0.0, 0.0, 0.0], HashMap::new()).unwrap();
        index.insert("pos2", &[0.9, 0.1, 0.0, 0.0], HashMap::new()).unwrap();
        index.insert("neg1", &[0.0, 0.0, 1.0, 0.0], HashMap::new()).unwrap();
        index.insert("candidate1", &[0.8, 0.2, 0.0, 0.0], HashMap::new()).unwrap();
        index.insert("candidate2", &[0.0, 0.0, 0.8, 0.2], HashMap::new()).unwrap();

        // Context search
        let query = DiscoveryQuery::new()
            .with_positive_ids(&["pos1", "pos2"])
            .with_negative_ids(&["neg1"])
            .with_top_k(2);

        let results = index.discover(query).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "candidate1"); // Should be closest to positive
    }

    #[test]
    fn test_discovery_with_diversity() {
        let config = DiscoveryConfig { dimension: 4, enable_mmr: true, ..Default::default() };
        let index = DiscoveryIndex::new(config);

        for i in 0..10 {
            let vec = vec![(i as f32) / 10.0, 1.0 - (i as f32) / 10.0, 0.0, 0.0];
            index.insert(&format!("vec_{}", i), &vec, HashMap::new()).unwrap();
        }

        let query = DiscoveryQuery::new()
            .with_positive_vectors(vec![vec![0.5, 0.5, 0.0, 0.0]])
            .with_diversity(0.5)
            .with_top_k(5);

        let results = index.discover(query).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_explore() {
        let config = DiscoveryConfig { dimension: 4, ..Default::default() };
        let index = DiscoveryIndex::new(config);

        index.insert("center", &[1.0, 0.0, 0.0, 0.0], HashMap::new()).unwrap();
        index.insert("near1", &[0.95, 0.05, 0.0, 0.0], HashMap::new()).unwrap();
        index.insert("near2", &[0.9, 0.1, 0.0, 0.0], HashMap::new()).unwrap();
        index.insert("far", &[0.0, 1.0, 0.0, 0.0], HashMap::new()).unwrap();

        let results = index.explore("center", 0.2, 10).unwrap();

        // near1 and near2 should be within radius
        assert!(results.iter().any(|r| r.id == "near1"));
        assert!(results.iter().any(|r| r.id == "near2"));
    }

    #[test]
    fn test_find_clusters() {
        let config = DiscoveryConfig {
            dimension: 4,
            num_clusters: 2,
            ..Default::default()
        };
        let index = DiscoveryIndex::new(config);

        // Create two clusters
        for i in 0..5 {
            let vec = vec![1.0 - (i as f32) * 0.05, (i as f32) * 0.05, 0.0, 0.0];
            index.insert(&format!("cluster1_{}", i), &vec, HashMap::new()).unwrap();
        }

        for i in 0..5 {
            let vec = vec![0.0, 0.0, 1.0 - (i as f32) * 0.05, (i as f32) * 0.05];
            index.insert(&format!("cluster2_{}", i), &vec, HashMap::new()).unwrap();
        }

        let exploration = index.find_clusters();

        assert!(!exploration.regions.is_empty());
        assert!(exploration.diversity_score > 0.0);
    }

    #[test]
    fn test_discovery_modes() {
        let config = DiscoveryConfig { dimension: 4, ..Default::default() };
        let index = DiscoveryIndex::new(config);

        index.insert("v1", &[1.0, 0.0, 0.0, 0.0], HashMap::new()).unwrap();
        index.insert("v2", &[0.0, 1.0, 0.0, 0.0], HashMap::new()).unwrap();
        index.insert("v3", &[0.0, 0.0, 1.0, 0.0], HashMap::new()).unwrap();

        // Discover mode (novelty)
        let query = DiscoveryQuery::new()
            .with_positive_ids(&["v1"])
            .with_mode(DiscoveryMode::Discover)
            .with_top_k(2);

        let results = index.discover(query).unwrap();
        assert!(!results.is_empty());
    }
}
