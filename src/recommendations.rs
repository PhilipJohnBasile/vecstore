//! Recommendations API
//!
//! Item-to-item and user-to-item recommendations using vector similarity.
//! Similar to Qdrant's recommendations API with personalization.
//!
//! # Features
//!
//! - **Item-to-Item**: Find similar items based on vectors
//! - **User-to-Item**: Personalized recommendations based on history
//! - **Collaborative Filtering**: Learn from user interactions
//! - **Diversity Control**: Avoid repetitive recommendations
//! - **Cold Start Handling**: Handle new users/items
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::recommendations::{RecommendationEngine, RecommendRequest};
//!
//! let engine = RecommendationEngine::new(384);
//!
//! // Item-to-item recommendations
//! let similar = engine.recommend_similar("item_123", 10)?;
//!
//! // User-to-item recommendations
//! let request = RecommendRequest::new()
//!     .with_positive(&["liked_item_1", "liked_item_2"])
//!     .with_negative(&["disliked_item"])
//!     .with_limit(10);
//!
//! let recommendations = engine.recommend(request)?;
//! ```

use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;
use serde::{Deserialize, Serialize};

use crate::error::{VecStoreError, Result};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Recommendation engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendConfig {
    /// Default number of recommendations
    pub default_limit: usize,
    /// Minimum similarity score
    pub min_score: f32,
    /// Enable diversity optimization
    pub enable_diversity: bool,
    /// Diversity factor (0-1, higher = more diverse)
    pub diversity_factor: f32,
    /// Enable collaborative filtering
    pub enable_collaborative: bool,
    /// Weight for collaborative signal
    pub collaborative_weight: f32,
}

impl Default for RecommendConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            min_score: 0.0,
            enable_diversity: true,
            diversity_factor: 0.3,
            enable_collaborative: true,
            collaborative_weight: 0.2,
        }
    }
}

// ============================================================================
// RECOMMENDATION REQUEST
// ============================================================================

/// Recommendation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendRequest {
    /// Positive examples (items to recommend similar to)
    pub positive: Vec<String>,
    /// Negative examples (items to avoid)
    pub negative: Vec<String>,
    /// Number of recommendations
    pub limit: usize,
    /// Filter by metadata
    pub filter: Option<serde_json::Value>,
    /// User ID for personalization
    pub user_id: Option<String>,
    /// Strategy to use
    pub strategy: RecommendStrategy,
    /// Override diversity factor
    pub diversity: Option<f32>,
}

impl RecommendRequest {
    pub fn new() -> Self {
        Self {
            positive: Vec::new(),
            negative: Vec::new(),
            limit: 10,
            filter: None,
            user_id: None,
            strategy: RecommendStrategy::AverageVector,
            diversity: None,
        }
    }

    pub fn with_positive(mut self, items: &[&str]) -> Self {
        self.positive = items.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn with_negative(mut self, items: &[&str]) -> Self {
        self.negative = items.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_user(mut self, user_id: &str) -> Self {
        self.user_id = Some(user_id.to_string());
        self
    }

    pub fn with_strategy(mut self, strategy: RecommendStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn with_diversity(mut self, diversity: f32) -> Self {
        self.diversity = Some(diversity);
        self
    }
}

impl Default for RecommendRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// Recommendation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendStrategy {
    /// Use average of positive vectors
    AverageVector,
    /// Use best matching positive
    BestMatch,
    /// Use all positives with voting
    Voting,
    /// Collaborative filtering only
    Collaborative,
    /// Hybrid (content + collaborative)
    Hybrid,
}

// ============================================================================
// RECOMMENDATION RESULT
// ============================================================================

/// Single recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub id: String,
    pub score: f32,
    pub content_score: f32,
    pub collaborative_score: Option<f32>,
    pub diversity_penalty: f32,
    pub explanation: Option<String>,
}

impl Eq for Recommendation {}

impl PartialEq for Recommendation {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialOrd for Recommendation {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for Recommendation {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Recommendation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendResult {
    pub recommendations: Vec<Recommendation>,
    pub strategy_used: RecommendStrategy,
    pub query_vector: Option<Vec<f32>>,
    pub processing_time_ms: u64,
}

// ============================================================================
// USER PROFILE
// ============================================================================

/// User profile for personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    /// Items the user has interacted with positively
    pub positive_items: Vec<String>,
    /// Items the user has interacted with negatively
    pub negative_items: Vec<String>,
    /// Aggregated preference vector
    pub preference_vector: Option<Vec<f32>>,
    /// Interaction counts by item
    pub item_interactions: HashMap<String, u32>,
    /// Last updated timestamp
    pub last_updated: u64,
}

impl UserProfile {
    pub fn new(user_id: &str) -> Self {
        Self {
            user_id: user_id.to_string(),
            positive_items: Vec::new(),
            negative_items: Vec::new(),
            preference_vector: None,
            item_interactions: HashMap::new(),
            last_updated: unix_timestamp(),
        }
    }

    /// Record a positive interaction
    pub fn record_positive(&mut self, item_id: &str) {
        if !self.positive_items.contains(&item_id.to_string()) {
            self.positive_items.push(item_id.to_string());
        }
        *self.item_interactions.entry(item_id.to_string()).or_insert(0) += 1;
        self.last_updated = unix_timestamp();
    }

    /// Record a negative interaction
    pub fn record_negative(&mut self, item_id: &str) {
        if !self.negative_items.contains(&item_id.to_string()) {
            self.negative_items.push(item_id.to_string());
        }
        self.last_updated = unix_timestamp();
    }
}

// ============================================================================
// RECOMMENDATION ENGINE
// ============================================================================

/// Recommendation engine
pub struct RecommendationEngine {
    /// Configuration
    config: RecommendConfig,
    /// Vector dimension
    dimension: usize,
    /// Item vectors
    items: HashMap<String, Vec<f32>>,
    /// Item metadata
    item_metadata: HashMap<String, serde_json::Value>,
    /// User profiles
    users: HashMap<String, UserProfile>,
    /// Item-to-item similarity cache
    similarity_cache: HashMap<String, Vec<(String, f32)>>,
    /// Co-occurrence matrix for collaborative filtering
    cooccurrence: HashMap<String, HashMap<String, u32>>,
}

impl RecommendationEngine {
    pub fn new(dimension: usize) -> Self {
        Self {
            config: RecommendConfig::default(),
            dimension,
            items: HashMap::new(),
            item_metadata: HashMap::new(),
            users: HashMap::new(),
            similarity_cache: HashMap::new(),
            cooccurrence: HashMap::new(),
        }
    }

    pub fn with_config(mut self, config: RecommendConfig) -> Self {
        self.config = config;
        self
    }

    /// Add an item
    pub fn add_item(
        &mut self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        self.items.insert(id.to_string(), vector);
        if let Some(meta) = metadata {
            self.item_metadata.insert(id.to_string(), meta);
        }

        // Invalidate similarity cache for this item
        self.similarity_cache.remove(id);

        Ok(())
    }

    /// Remove an item
    pub fn remove_item(&mut self, id: &str) -> bool {
        self.similarity_cache.remove(id);
        self.item_metadata.remove(id);
        self.items.remove(id).is_some()
    }

    /// Get or create user profile
    pub fn get_user(&mut self, user_id: &str) -> &mut UserProfile {
        if !self.users.contains_key(user_id) {
            self.users.insert(user_id.to_string(), UserProfile::new(user_id));
        }
        self.users.get_mut(user_id).unwrap()
    }

    /// Record user interaction
    pub fn record_interaction(
        &mut self,
        user_id: &str,
        item_id: &str,
        positive: bool,
    ) {
        let user = self.get_user(user_id);
        if positive {
            user.record_positive(item_id);
        } else {
            user.record_negative(item_id);
        }

        // Update co-occurrence matrix
        if positive {
            for other_item in &user.positive_items.clone() {
                if other_item != item_id {
                    *self.cooccurrence
                        .entry(item_id.to_string())
                        .or_insert_with(HashMap::new)
                        .entry(other_item.clone())
                        .or_insert(0) += 1;

                    *self.cooccurrence
                        .entry(other_item.clone())
                        .or_insert_with(HashMap::new)
                        .entry(item_id.to_string())
                        .or_insert(0) += 1;
                }
            }
        }
    }

    /// Get recommendations
    pub fn recommend(&self, request: RecommendRequest) -> Result<RecommendResult> {
        let start = std::time::Instant::now();

        // Build query vector from positive/negative examples
        let query_vector = self.build_query_vector(&request)?;

        // Get content-based candidates
        let mut candidates = self.get_content_candidates(&query_vector, &request)?;

        // Add collaborative scores if enabled
        if self.config.enable_collaborative {
            self.add_collaborative_scores(&mut candidates, &request);
        }

        // Apply diversity optimization
        if self.config.enable_diversity {
            let diversity = request.diversity.unwrap_or(self.config.diversity_factor);
            self.apply_diversity(&mut candidates, diversity);
        }

        // Sort by final score and take top-k
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        candidates.truncate(request.limit);

        Ok(RecommendResult {
            recommendations: candidates,
            strategy_used: request.strategy,
            query_vector: Some(query_vector),
            processing_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Recommend items similar to a single item
    pub fn recommend_similar(&self, item_id: &str, limit: usize) -> Result<Vec<Recommendation>> {
        let item_vector = self.items.get(item_id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Item: {}", item_id)))?;

        let mut results: Vec<Recommendation> = self.items.iter()
            .filter(|(id, _)| *id != item_id)
            .map(|(id, vec)| {
                let score = cosine_similarity(item_vector, vec);
                Recommendation {
                    id: id.clone(),
                    score,
                    content_score: score,
                    collaborative_score: None,
                    diversity_penalty: 0.0,
                    explanation: Some(format!("Similar to {}", item_id)),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Build query vector from request
    fn build_query_vector(&self, request: &RecommendRequest) -> Result<Vec<f32>> {
        if request.positive.is_empty() {
            // Use user profile if available
            if let Some(user_id) = &request.user_id {
                if let Some(user) = self.users.get(user_id) {
                    if let Some(pref) = &user.preference_vector {
                        return Ok(pref.clone());
                    }
                    // Build from user's positive items
                    let positive_ids: Vec<&str> = user.positive_items.iter()
                        .map(|s| s.as_str())
                        .collect();
                    return self.average_vectors(&positive_ids);
                }
            }
            return Err(VecStoreError::InvalidInput(
                "No positive examples or user profile".to_string()
            ));
        }

        match request.strategy {
            RecommendStrategy::AverageVector | RecommendStrategy::Hybrid => {
                let positive_refs: Vec<&str> = request.positive.iter()
                    .map(|s| s.as_str())
                    .collect();
                let mut avg = self.average_vectors(&positive_refs)?;

                // Subtract negative vectors
                if !request.negative.is_empty() {
                    let negative_refs: Vec<&str> = request.negative.iter()
                        .map(|s| s.as_str())
                        .collect();
                    if let Ok(neg_avg) = self.average_vectors(&negative_refs) {
                        for (i, v) in avg.iter_mut().enumerate() {
                            *v -= neg_avg[i] * 0.5;
                        }
                    }
                }

                // Normalize
                let norm: f32 = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut avg {
                        *v /= norm;
                    }
                }

                Ok(avg)
            }
            _ => {
                // For other strategies, use first positive
                let first = &request.positive[0];
                self.items.get(first)
                    .cloned()
                    .ok_or_else(|| VecStoreError::NotFound(format!("Item: {}", first)))
            }
        }
    }

    /// Average multiple item vectors
    fn average_vectors(&self, item_ids: &[&str]) -> Result<Vec<f32>> {
        if item_ids.is_empty() {
            return Err(VecStoreError::InvalidInput("Empty item list".to_string()));
        }

        let mut sum = vec![0.0; self.dimension];
        let mut count = 0;

        for id in item_ids {
            if let Some(vec) = self.items.get(*id) {
                for (i, v) in vec.iter().enumerate() {
                    sum[i] += v;
                }
                count += 1;
            }
        }

        if count == 0 {
            return Err(VecStoreError::NotFound("No valid items found".to_string()));
        }

        Ok(sum.into_iter().map(|v| v / count as f32).collect())
    }

    /// Get content-based candidates
    fn get_content_candidates(
        &self,
        query_vector: &[f32],
        request: &RecommendRequest,
    ) -> Result<Vec<Recommendation>> {
        let excluded: HashSet<_> = request.positive.iter()
            .chain(request.negative.iter())
            .collect();

        let candidates: Vec<Recommendation> = self.items.iter()
            .filter(|(id, _)| !excluded.contains(id))
            .map(|(id, vec)| {
                let score = cosine_similarity(query_vector, vec);
                Recommendation {
                    id: id.clone(),
                    score,
                    content_score: score,
                    collaborative_score: None,
                    diversity_penalty: 0.0,
                    explanation: None,
                }
            })
            .filter(|r| r.score >= self.config.min_score)
            .collect();

        Ok(candidates)
    }

    /// Add collaborative filtering scores
    fn add_collaborative_scores(
        &self,
        candidates: &mut [Recommendation],
        request: &RecommendRequest,
    ) {
        // Get collaborative scores from co-occurrence
        for candidate in candidates.iter_mut() {
            let mut collab_score = 0.0;
            let mut count = 0;

            for positive_id in &request.positive {
                if let Some(cooc) = self.cooccurrence.get(positive_id) {
                    if let Some(&freq) = cooc.get(&candidate.id) {
                        collab_score += freq as f32;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                // Normalize collaborative score
                collab_score = (collab_score / count as f32).min(1.0);
                candidate.collaborative_score = Some(collab_score);

                // Blend with content score
                candidate.score = candidate.content_score * (1.0 - self.config.collaborative_weight)
                    + collab_score * self.config.collaborative_weight;
            }
        }
    }

    /// Apply diversity optimization using MMR-like approach
    fn apply_diversity(&self, candidates: &mut [Recommendation], diversity_factor: f32) {
        if candidates.len() < 2 {
            return;
        }

        // Sort by content score first
        candidates.sort_by(|a, b| b.content_score.partial_cmp(&a.content_score).unwrap_or(Ordering::Equal));

        // Apply MMR-like reranking
        let mut selected: Vec<&Vec<f32>> = Vec::new();

        for candidate in candidates.iter_mut() {
            if let Some(vec) = self.items.get(&candidate.id) {
                // Calculate max similarity to already selected items
                let mut max_sim: f32 = 0.0;
                for sel_vec in &selected {
                    let sim = cosine_similarity(vec, sel_vec);
                    max_sim = max_sim.max(sim);
                }

                // Apply diversity penalty
                candidate.diversity_penalty = max_sim * diversity_factor;
                candidate.score = candidate.content_score * (1.0 - diversity_factor)
                    + (1.0 - max_sim) * diversity_factor;

                selected.push(vec);
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> RecommendationStats {
        RecommendationStats {
            total_items: self.items.len(),
            total_users: self.users.len(),
            cached_similarities: self.similarity_cache.len(),
            cooccurrence_pairs: self.cooccurrence.values()
                .map(|m| m.len())
                .sum(),
        }
    }
}

/// Recommendation engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationStats {
    pub total_items: usize,
    pub total_users: usize,
    pub cached_similarities: usize,
    pub cooccurrence_pairs: usize,
}

// ============================================================================
// HELPERS
// ============================================================================

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommend_similar() {
        let mut engine = RecommendationEngine::new(4);

        engine.add_item("item1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
        engine.add_item("item2", vec![0.9, 0.1, 0.0, 0.0], None).unwrap();
        engine.add_item("item3", vec![0.0, 1.0, 0.0, 0.0], None).unwrap();

        let results = engine.recommend_similar("item1", 10).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "item2"); // Most similar
    }

    #[test]
    fn test_recommend_with_positives() {
        let mut engine = RecommendationEngine::new(4);

        engine.add_item("liked1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
        engine.add_item("liked2", vec![0.8, 0.2, 0.0, 0.0], None).unwrap();
        engine.add_item("candidate1", vec![0.9, 0.1, 0.0, 0.0], None).unwrap();
        engine.add_item("candidate2", vec![0.0, 0.0, 1.0, 0.0], None).unwrap();

        let request = RecommendRequest::new()
            .with_positive(&["liked1", "liked2"])
            .with_limit(10);

        let result = engine.recommend(request).unwrap();
        assert!(!result.recommendations.is_empty());
        // candidate1 should be ranked higher (more similar to likes)
        assert_eq!(result.recommendations[0].id, "candidate1");
    }

    #[test]
    fn test_collaborative_filtering() {
        let mut engine = RecommendationEngine::new(4);

        // Add items
        engine.add_item("item1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
        engine.add_item("item2", vec![0.0, 1.0, 0.0, 0.0], None).unwrap();
        engine.add_item("item3", vec![0.0, 0.0, 1.0, 0.0], None).unwrap();

        // User1 likes item1 and item2
        engine.record_interaction("user1", "item1", true);
        engine.record_interaction("user1", "item2", true);

        // User2 also likes item1 and item2, and item3
        engine.record_interaction("user2", "item1", true);
        engine.record_interaction("user2", "item2", true);
        engine.record_interaction("user2", "item3", true);

        // Recommend based on item1 - should boost item2 and item3 collaboratively
        let request = RecommendRequest::new()
            .with_positive(&["item1"])
            .with_limit(10);

        let result = engine.recommend(request).unwrap();

        // item2 and item3 should have collaborative scores
        for rec in &result.recommendations {
            if rec.id == "item2" || rec.id == "item3" {
                assert!(rec.collaborative_score.is_some());
            }
        }
    }
}
