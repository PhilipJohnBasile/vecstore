//! Recommender system for vector-based recommendations
//!
//! Provides collaborative filtering and content-based recommendation engines
//! built on top of vector similarity. Useful for:
//! - Product recommendations
//! - Content discovery
//! - Personalization
//! - Similar item suggestions
//!
//! # Methods
//!
//! - **Content-Based**: Recommend items similar to user's preferences
//! - **Collaborative Filtering**: Recommend based on similar users' preferences
//! - **Hybrid**: Combine both approaches
//!
//! # Example
//!
//! ```rust
//! use vecstore::recommender::{ContentBasedRecommender, UserPreference};
//!
//! let mut recommender = ContentBasedRecommender::new();
//!
//! // Add items with feature vectors
//! recommender.add_item("movie1", vec![0.9, 0.1, 0.5], metadata);
//! recommender.add_item("movie2", vec![0.8, 0.2, 0.6], metadata);
//!
//! // Get recommendations based on preferences
//! let preferences = vec![
//!     UserPreference::new("movie1", 5.0),
//! ];
//! let recommendations = recommender.recommend(&preferences, 10)?;
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::simd::cosine_similarity_simd;
use crate::store::Metadata;

/// User preference/rating for an item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreference {
    /// Item ID
    pub item_id: String,
    /// Rating/score (typically 1-5 or 0-1)
    pub rating: f32,
}

impl UserPreference {
    /// Create new user preference
    pub fn new(item_id: impl Into<String>, rating: f32) -> Self {
        Self {
            item_id: item_id.into(),
            rating,
        }
    }
}

/// Recommended item with score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Item ID
    pub item_id: String,
    /// Recommendation score (higher = more recommended)
    pub score: f32,
    /// Explanation of why recommended
    pub reason: String,
    /// Item metadata
    pub metadata: Option<Metadata>,
}

impl Recommendation {
    /// Create new recommendation
    pub fn new(item_id: String, score: f32, reason: String) -> Self {
        Self {
            item_id,
            score,
            reason,
            metadata: None,
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Item with feature vector
#[derive(Debug, Clone)]
struct Item {
    id: String,
    vector: Vec<f32>,
    metadata: Metadata,
}

/// Content-based recommender using vector similarity
///
/// Recommends items similar to those the user has liked.
/// Fast and works well for cold-start problems.
#[derive(Debug, Clone)]
pub struct ContentBasedRecommender {
    /// Items in the catalog
    items: HashMap<String, Item>,
}

impl ContentBasedRecommender {
    /// Create new content-based recommender
    pub fn new() -> Self {
        Self {
            items: HashMap::new(),
        }
    }

    /// Add item to catalog
    pub fn add_item(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Metadata,
    ) -> Result<()> {
        let id = id.into();
        self.items.insert(
            id.clone(),
            Item {
                id,
                vector,
                metadata,
            },
        );
        Ok(())
    }

    /// Get recommendations based on user preferences
    ///
    /// # Arguments
    /// * `preferences` - User's rated items
    /// * `top_k` - Number of recommendations to return
    pub fn recommend(
        &self,
        preferences: &[UserPreference],
        top_k: usize,
    ) -> Result<Vec<Recommendation>> {
        if preferences.is_empty() {
            return Err(anyhow!("No preferences provided"));
        }

        // Build user profile as weighted average of liked items
        let mut profile = self.build_user_profile(preferences)?;

        // Normalize profile
        let norm = profile.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut profile {
                *val /= norm;
            }
        }

        // Score all items
        let mut scores: Vec<(String, f32)> = self
            .items
            .iter()
            .filter_map(|(id, item)| {
                // Skip items user has already rated
                if preferences.iter().any(|p| &p.item_id == id) {
                    return None;
                }

                let similarity = cosine_similarity_simd(&profile, &item.vector);
                Some((id.clone(), similarity))
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        // Create recommendations
        let recommendations: Vec<Recommendation> = scores
            .into_iter()
            .map(|(item_id, score)| {
                let reason = format!("Similar to items you liked (score: {:.3})", score);
                let metadata = self.items.get(&item_id).map(|item| item.metadata.clone());

                let mut rec = Recommendation::new(item_id, score, reason);
                if let Some(meta) = metadata {
                    rec = rec.with_metadata(meta);
                }
                rec
            })
            .collect();

        Ok(recommendations)
    }

    /// Build user profile from preferences
    fn build_user_profile(&self, preferences: &[UserPreference]) -> Result<Vec<f32>> {
        let mut profile: Option<Vec<f32>> = None;
        let mut total_weight = 0.0;

        for pref in preferences {
            let item = self
                .items
                .get(&pref.item_id)
                .ok_or_else(|| anyhow!("Item not found: {}", pref.item_id))?;

            // Weight by rating
            let weight = pref.rating;
            total_weight += weight;

            if let Some(ref mut p) = profile {
                for (i, &val) in item.vector.iter().enumerate() {
                    p[i] += val * weight;
                }
            } else {
                profile = Some(item.vector.iter().map(|&v| v * weight).collect());
            }
        }

        let mut profile = profile.ok_or_else(|| anyhow!("No valid items found"))?;

        // Average
        if total_weight > 0.0 {
            for val in &mut profile {
                *val /= total_weight;
            }
        }

        Ok(profile)
    }

    /// Get similar items to a given item
    pub fn similar_items(&self, item_id: &str, top_k: usize) -> Result<Vec<Recommendation>> {
        let item = self
            .items
            .get(item_id)
            .ok_or_else(|| anyhow!("Item not found: {}", item_id))?;

        let mut scores: Vec<(String, f32)> = self
            .items
            .iter()
            .filter_map(|(id, other_item)| {
                if id == item_id {
                    return None;
                }

                let similarity = cosine_similarity_simd(&item.vector, &other_item.vector);
                Some((id.clone(), similarity))
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        let recommendations: Vec<Recommendation> = scores
            .into_iter()
            .map(|(id, score)| {
                let reason = format!("Similar to {} (score: {:.3})", item_id, score);
                let metadata = self.items.get(&id).map(|item| item.metadata.clone());

                let mut rec = Recommendation::new(id, score, reason);
                if let Some(meta) = metadata {
                    rec = rec.with_metadata(meta);
                }
                rec
            })
            .collect();

        Ok(recommendations)
    }
}

impl Default for ContentBasedRecommender {
    fn default() -> Self {
        Self::new()
    }
}

/// Collaborative filtering recommender using user-user similarity
///
/// Recommends items that similar users have liked.
/// Requires interaction history from multiple users.
#[derive(Debug, Clone)]
pub struct CollaborativeRecommender {
    /// User ID -> Item ratings
    user_ratings: HashMap<String, HashMap<String, f32>>,
    /// Item ID -> Set of users who rated it
    item_users: HashMap<String, Vec<String>>,
}

impl CollaborativeRecommender {
    /// Create new collaborative recommender
    pub fn new() -> Self {
        Self {
            user_ratings: HashMap::new(),
            item_users: HashMap::new(),
        }
    }

    /// Add user rating
    pub fn add_rating(
        &mut self,
        user_id: impl Into<String>,
        item_id: impl Into<String>,
        rating: f32,
    ) {
        let user_id = user_id.into();
        let item_id = item_id.into();

        self.user_ratings
            .entry(user_id.clone())
            .or_insert_with(HashMap::new)
            .insert(item_id.clone(), rating);

        self.item_users
            .entry(item_id)
            .or_insert_with(Vec::new)
            .push(user_id);
    }

    /// Get recommendations for a user
    pub fn recommend(&self, user_id: &str, top_k: usize) -> Result<Vec<Recommendation>> {
        let user_ratings = self
            .user_ratings
            .get(user_id)
            .ok_or_else(|| anyhow!("User not found: {}", user_id))?;

        // Find similar users
        let similar_users = self.find_similar_users(user_id, 20)?;

        // Score items based on similar users' ratings
        let mut item_scores: HashMap<String, (f32, f32)> = HashMap::new(); // (weighted_sum, weight_sum)

        for (other_user, similarity) in similar_users {
            if let Some(other_ratings) = self.user_ratings.get(&other_user) {
                for (item_id, rating) in other_ratings {
                    // Skip items user has already rated
                    if user_ratings.contains_key(item_id) {
                        continue;
                    }

                    let entry = item_scores.entry(item_id.clone()).or_insert((0.0, 0.0));
                    entry.0 += rating * similarity;
                    entry.1 += similarity;
                }
            }
        }

        // Compute final scores
        let mut scores: Vec<(String, f32)> = item_scores
            .into_iter()
            .map(|(item_id, (weighted_sum, weight_sum))| {
                let score = if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    0.0
                };
                (item_id, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        let recommendations: Vec<Recommendation> = scores
            .into_iter()
            .map(|(item_id, score)| {
                let reason = format!("Users like you also liked this (score: {:.3})", score);
                Recommendation::new(item_id, score, reason)
            })
            .collect();

        Ok(recommendations)
    }

    /// Find similar users using Pearson correlation
    fn find_similar_users(&self, user_id: &str, top_k: usize) -> Result<Vec<(String, f32)>> {
        let user_ratings = self
            .user_ratings
            .get(user_id)
            .ok_or_else(|| anyhow!("User not found: {}", user_id))?;

        let mut similarities: Vec<(String, f32)> = self
            .user_ratings
            .iter()
            .filter_map(|(other_id, other_ratings)| {
                if other_id == user_id {
                    return None;
                }

                let similarity = self.pearson_correlation(user_ratings, other_ratings);
                if similarity > 0.0 {
                    Some((other_id.clone(), similarity))
                } else {
                    None
                }
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);

        Ok(similarities)
    }

    /// Compute Pearson correlation between two users
    fn pearson_correlation(
        &self,
        ratings1: &HashMap<String, f32>,
        ratings2: &HashMap<String, f32>,
    ) -> f32 {
        // Find common items
        let common_items: Vec<&String> = ratings1
            .keys()
            .filter(|item| ratings2.contains_key(*item))
            .collect();

        if common_items.len() < 2 {
            return 0.0;
        }

        let n = common_items.len() as f32;

        let sum1: f32 = common_items.iter().map(|item| ratings1[*item]).sum();
        let sum2: f32 = common_items.iter().map(|item| ratings2[*item]).sum();

        let sum1_sq: f32 = common_items
            .iter()
            .map(|item| ratings1[*item].powi(2))
            .sum();
        let sum2_sq: f32 = common_items
            .iter()
            .map(|item| ratings2[*item].powi(2))
            .sum();

        let sum_products: f32 = common_items
            .iter()
            .map(|item| ratings1[*item] * ratings2[*item])
            .sum();

        let numerator = sum_products - (sum1 * sum2) / n;
        let denominator = ((sum1_sq - sum1.powi(2) / n) * (sum2_sq - sum2.powi(2) / n)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

impl Default for CollaborativeRecommender {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid recommender combining content-based and collaborative filtering
pub struct HybridRecommender {
    content_based: ContentBasedRecommender,
    collaborative: CollaborativeRecommender,
    /// Weight for content-based (0-1, collaborative weight = 1 - content_weight)
    content_weight: f32,
}

impl HybridRecommender {
    /// Create new hybrid recommender
    pub fn new(content_weight: f32) -> Self {
        Self {
            content_based: ContentBasedRecommender::new(),
            collaborative: CollaborativeRecommender::new(),
            content_weight: content_weight.clamp(0.0, 1.0),
        }
    }

    /// Add item to content-based catalog
    pub fn add_item(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Metadata,
    ) -> Result<()> {
        self.content_based.add_item(id, vector, metadata)
    }

    /// Add rating for collaborative filtering
    pub fn add_rating(
        &mut self,
        user_id: impl Into<String>,
        item_id: impl Into<String>,
        rating: f32,
    ) {
        self.collaborative.add_rating(user_id, item_id, rating);
    }

    /// Get hybrid recommendations
    pub fn recommend(
        &self,
        user_id: &str,
        preferences: &[UserPreference],
        top_k: usize,
    ) -> Result<Vec<Recommendation>> {
        // Get recommendations from both systems
        let content_recs = self.content_based.recommend(preferences, top_k * 2)?;
        let collab_recs = self.collaborative.recommend(user_id, top_k * 2)?;

        // Combine scores
        let mut combined: HashMap<String, (f32, String)> = HashMap::new();

        for rec in content_recs {
            let score = rec.score * self.content_weight;
            combined.insert(rec.item_id, (score, "content-based".to_string()));
        }

        for rec in collab_recs {
            let collab_score = rec.score * (1.0 - self.content_weight);
            combined
                .entry(rec.item_id)
                .and_modify(|(score, reason)| {
                    *score += collab_score;
                    *reason = "hybrid (content + collaborative)".to_string();
                })
                .or_insert((collab_score, "collaborative".to_string()));
        }

        // Sort and take top K
        let mut results: Vec<(String, f32, String)> = combined
            .into_iter()
            .map(|(id, (score, reason))| (id, score, reason))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        let recommendations: Vec<Recommendation> = results
            .into_iter()
            .map(|(item_id, score, reason)| Recommendation::new(item_id, score, reason))
            .collect();

        Ok(recommendations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_metadata(title: &str) -> Metadata {
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), serde_json::json!(title));
        Metadata { fields }
    }

    #[test]
    fn test_content_based_recommender() -> Result<()> {
        let mut recommender = ContentBasedRecommender::new();

        // Add items (movies with genre vectors)
        recommender.add_item(
            "movie1",
            vec![1.0, 0.0, 0.0],
            create_test_metadata("Action Movie"),
        )?;
        recommender.add_item(
            "movie2",
            vec![0.9, 0.1, 0.0],
            create_test_metadata("Action Comedy"),
        )?;
        recommender.add_item(
            "movie3",
            vec![0.0, 1.0, 0.0],
            create_test_metadata("Comedy"),
        )?;
        recommender.add_item("movie4", vec![0.0, 0.0, 1.0], create_test_metadata("Drama"))?;

        // User liked action movie
        let preferences = vec![UserPreference::new("movie1", 5.0)];

        let recs = recommender.recommend(&preferences, 2)?;

        assert_eq!(recs.len(), 2);
        // Should recommend action comedy (most similar)
        assert_eq!(recs[0].item_id, "movie2");

        Ok(())
    }

    #[test]
    fn test_similar_items() -> Result<()> {
        let mut recommender = ContentBasedRecommender::new();

        recommender.add_item("item1", vec![1.0, 0.0], create_test_metadata("Item 1"))?;
        recommender.add_item("item2", vec![0.9, 0.1], create_test_metadata("Item 2"))?;
        recommender.add_item("item3", vec![0.0, 1.0], create_test_metadata("Item 3"))?;

        let similar = recommender.similar_items("item1", 2)?;

        assert_eq!(similar.len(), 2);
        assert_eq!(similar[0].item_id, "item2"); // Most similar

        Ok(())
    }

    #[test]
    fn test_collaborative_recommender() -> Result<()> {
        let mut recommender = CollaborativeRecommender::new();

        // User1 and User2 have similar tastes
        recommender.add_rating("user1", "item1", 5.0);
        recommender.add_rating("user1", "item2", 5.0);
        recommender.add_rating("user1", "item3", 1.0);

        recommender.add_rating("user2", "item1", 4.0);
        recommender.add_rating("user2", "item2", 5.0);
        recommender.add_rating("user2", "item3", 2.0);
        recommender.add_rating("user2", "item4", 5.0); // User2 also liked item4

        // Should recommend item4 to user1
        let recs = recommender.recommend("user1", 2)?;

        assert!(!recs.is_empty());
        // item4 should be recommended since similar user liked it
        assert!(recs.iter().any(|r| r.item_id == "item4"));

        Ok(())
    }

    #[test]
    fn test_hybrid_recommender() -> Result<()> {
        let mut recommender = HybridRecommender::new(0.5);

        // Add items
        recommender.add_item("item1", vec![1.0, 0.0], create_test_metadata("Item 1"))?;
        recommender.add_item("item2", vec![0.9, 0.1], create_test_metadata("Item 2"))?;
        recommender.add_item("item3", vec![0.0, 1.0], create_test_metadata("Item 3"))?;

        // Add ratings
        recommender.add_rating("user1", "item1", 5.0);

        let preferences = vec![UserPreference::new("item1", 5.0)];
        let recs = recommender.recommend("user1", &preferences, 2)?;

        assert!(!recs.is_empty());

        Ok(())
    }

    #[test]
    fn test_user_profile_building() -> Result<()> {
        let mut recommender = ContentBasedRecommender::new();

        recommender.add_item("item1", vec![1.0, 0.0], create_test_metadata("Item 1"))?;
        recommender.add_item("item2", vec![0.0, 1.0], create_test_metadata("Item 2"))?;

        let preferences = vec![
            UserPreference::new("item1", 4.0),
            UserPreference::new("item2", 2.0),
        ];

        let profile = recommender.build_user_profile(&preferences)?;

        // Profile should be weighted average: (4*[1,0] + 2*[0,1])/6 = [2/3, 1/3]
        assert!(profile[0] > profile[1]);

        Ok(())
    }
}
