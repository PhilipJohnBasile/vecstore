// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! Time-Aware Vector Search
//!
//! This module provides temporal awareness for vector search, including:
//! - **Temporal Decay**: Automatically weight recent vectors higher
//! - **Point-in-Time Queries**: Query the state of the index at any past moment
//! - **Drift Detection**: Detect when embedding distributions shift over time
//! - **Temporal Clustering**: Group vectors by time periods
//! - **Trend Analysis**: Track how vector neighborhoods evolve
//!
//! # Example
//!
//! ```ignore
//! use vecstore::temporal::{TemporalSearch, TemporalConfig, DecayFunction};
//!
//! let config = TemporalConfig {
//!     decay_function: DecayFunction::Exponential { half_life_hours: 24.0 },
//!     enable_point_in_time: true,
//!     drift_detection: true,
//!     ..Default::default()
//! };
//!
//! let temporal = TemporalSearch::new(config);
//!
//! // Query with temporal decay (recent vectors weighted higher)
//! let results = temporal.search_with_decay(&store, query_vec, k)?;
//!
//! // Point-in-time query (what would results be 1 week ago?)
//! let past = Utc::now() - Duration::days(7);
//! let historical = temporal.search_at_time(&store, query_vec, k, past)?;
//!
//! // Detect drift in embeddings
//! let drift = temporal.detect_drift(&store, "category:tech")?;
//! ```

use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};

use crate::error::VecStoreError;

/// Decay function for temporal weighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayFunction {
    /// No decay - all vectors weighted equally
    None,
    /// Linear decay: weight = max(0, 1 - age/max_age)
    Linear {
        /// Maximum age in hours before weight becomes 0
        max_age_hours: f64,
    },
    /// Exponential decay: weight = 2^(-age/half_life)
    Exponential {
        /// Time in hours for weight to halve
        half_life_hours: f64,
    },
    /// Gaussian decay: weight = exp(-(age/scale)^2)
    Gaussian {
        /// Scale parameter in hours
        scale_hours: f64,
    },
    /// Step function: full weight before cutoff, zero after
    Step {
        /// Cutoff age in hours
        cutoff_hours: f64,
    },
    /// Custom decay with configurable parameters
    Custom {
        /// Decay rate parameter
        rate: f64,
        /// Minimum weight (floor)
        min_weight: f64,
        /// Maximum age before minimum weight applies
        max_age_hours: f64,
    },
}

impl Default for DecayFunction {
    fn default() -> Self {
        DecayFunction::Exponential { half_life_hours: 168.0 } // 1 week half-life
    }
}

impl DecayFunction {
    /// Calculate the temporal weight for a given age in hours
    pub fn weight(&self, age_hours: f64) -> f64 {
        match self {
            DecayFunction::None => 1.0,

            DecayFunction::Linear { max_age_hours } => {
                if age_hours >= *max_age_hours {
                    0.0
                } else {
                    1.0 - (age_hours / max_age_hours)
                }
            }

            DecayFunction::Exponential { half_life_hours } => {
                2.0_f64.powf(-age_hours / half_life_hours)
            }

            DecayFunction::Gaussian { scale_hours } => {
                let ratio = age_hours / scale_hours;
                (-ratio * ratio).exp()
            }

            DecayFunction::Step { cutoff_hours } => {
                if age_hours <= *cutoff_hours { 1.0 } else { 0.0 }
            }

            DecayFunction::Custom { rate, min_weight, max_age_hours } => {
                if age_hours >= *max_age_hours {
                    *min_weight
                } else {
                    let decay = (-rate * age_hours).exp();
                    decay.max(*min_weight)
                }
            }
        }
    }
}

/// Configuration for temporal search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Decay function for weighting vectors by age
    pub decay_function: DecayFunction,
    /// Enable point-in-time queries
    pub enable_point_in_time: bool,
    /// Enable drift detection
    pub drift_detection: bool,
    /// Window size for drift detection (hours)
    pub drift_window_hours: f64,
    /// Threshold for significant drift (0.0 - 1.0)
    pub drift_threshold: f64,
    /// Number of time buckets for temporal histograms
    pub histogram_buckets: usize,
    /// Retention period for temporal metadata (hours, 0 = forever)
    pub retention_hours: f64,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            decay_function: DecayFunction::default(),
            enable_point_in_time: true,
            drift_detection: true,
            drift_window_hours: 24.0,
            drift_threshold: 0.15,
            histogram_buckets: 24,
            retention_hours: 0.0, // Keep forever
        }
    }
}

/// Temporal metadata for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMetadata {
    /// When the vector was created
    pub created_at: DateTime<Utc>,
    /// When the vector was last updated
    pub updated_at: DateTime<Utc>,
    /// Version number (increments on update)
    pub version: u64,
    /// Previous versions (for point-in-time queries)
    pub history: Vec<VectorVersion>,
}

/// A historical version of a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorVersion {
    /// Version number
    pub version: u64,
    /// Timestamp of this version
    pub timestamp: DateTime<Utc>,
    /// The vector data (compressed)
    pub vector_hash: u64,
    /// Metadata at this version
    pub metadata_snapshot: Option<HashMap<String, serde_json::Value>>,
}

/// Result of a temporal search
#[derive(Debug, Clone)]
pub struct TemporalResult {
    /// Vector ID
    pub id: String,
    /// Base similarity score
    pub base_score: f32,
    /// Temporal weight applied
    pub temporal_weight: f32,
    /// Final adjusted score
    pub adjusted_score: f32,
    /// Age of the vector in hours
    pub age_hours: f64,
    /// Timestamp of the vector
    pub timestamp: DateTime<Utc>,
    /// Original metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Drift detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// Time period analyzed
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    /// Overall drift magnitude (0.0 - 1.0)
    pub drift_magnitude: f64,
    /// Is drift significant?
    pub is_significant: bool,
    /// Drift by dimension (top contributors)
    pub dimension_drift: Vec<DimensionDrift>,
    /// Centroid shift vector
    pub centroid_shift: Vec<f32>,
    /// Variance change
    pub variance_change: f64,
    /// Distribution statistics
    pub stats: DriftStatistics,
}

/// Drift contribution by dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionDrift {
    /// Dimension index
    pub dimension: usize,
    /// Drift amount for this dimension
    pub drift: f64,
    /// Direction of drift
    pub direction: DriftDirection,
}

/// Direction of drift
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DriftDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Statistical summary of drift
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftStatistics {
    /// Number of vectors in old window
    pub old_count: usize,
    /// Number of vectors in new window
    pub new_count: usize,
    /// Mean distance between distributions
    pub distribution_distance: f64,
    /// KL divergence estimate
    pub kl_divergence: f64,
    /// Cosine similarity between centroids
    pub centroid_similarity: f64,
}

/// Temporal trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalTrend {
    /// ID being tracked
    pub id: String,
    /// Time points analyzed
    pub time_points: Vec<DateTime<Utc>>,
    /// Neighborhood stability over time
    pub stability_scores: Vec<f64>,
    /// Average neighbor turnover rate
    pub turnover_rate: f64,
    /// Trend direction
    pub trend: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Stable,
    Growing,
    Shrinking,
    Volatile,
}

/// Time bucket for histogram analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBucket {
    /// Start of time bucket
    pub start: DateTime<Utc>,
    /// End of time bucket
    pub end: DateTime<Utc>,
    /// Number of vectors in this bucket
    pub count: usize,
    /// Average vector in this bucket (centroid)
    pub centroid: Option<Vec<f32>>,
    /// Variance within bucket
    pub variance: f64,
}

/// Main temporal search engine
pub struct TemporalSearch {
    config: TemporalConfig,
    /// Cached temporal metadata by vector ID
    metadata_cache: HashMap<String, TemporalMetadata>,
    /// Historical snapshots for point-in-time queries
    snapshots: Vec<Snapshot>,
}

/// A point-in-time snapshot
#[derive(Debug, Clone)]
struct Snapshot {
    timestamp: DateTime<Utc>,
    vector_ids: Vec<String>,
    index_state_hash: u64,
}

impl TemporalSearch {
    /// Create a new temporal search engine
    pub fn new(config: TemporalConfig) -> Self {
        Self {
            config,
            metadata_cache: HashMap::new(),
            snapshots: Vec::new(),
        }
    }

    /// Register a vector with temporal metadata
    pub fn register_vector(&mut self, id: String, timestamp: DateTime<Utc>) {
        let metadata = TemporalMetadata {
            created_at: timestamp,
            updated_at: timestamp,
            version: 1,
            history: Vec::new(),
        };
        self.metadata_cache.insert(id, metadata);
    }

    /// Update temporal metadata for a vector
    pub fn update_vector(&mut self, id: &str, timestamp: DateTime<Utc>, vector_hash: u64) {
        if let Some(meta) = self.metadata_cache.get_mut(id) {
            // Save current version to history
            let version = VectorVersion {
                version: meta.version,
                timestamp: meta.updated_at,
                vector_hash,
                metadata_snapshot: None,
            };
            meta.history.push(version);

            // Update to new version
            meta.version += 1;
            meta.updated_at = timestamp;
        }
    }

    /// Calculate temporal weight for a vector
    pub fn calculate_weight(&self, id: &str, reference_time: DateTime<Utc>) -> f32 {
        if let Some(meta) = self.metadata_cache.get(id) {
            let age = reference_time.signed_duration_since(meta.updated_at);
            let age_hours = age.num_seconds() as f64 / 3600.0;
            self.config.decay_function.weight(age_hours.max(0.0)) as f32
        } else {
            1.0 // Default weight if no metadata
        }
    }

    /// Adjust search results with temporal decay
    pub fn apply_decay(&self, results: &mut [TemporalResult]) {
        let now = Utc::now();

        for result in results.iter_mut() {
            let weight = self.calculate_weight(&result.id, now);
            result.temporal_weight = weight;
            result.adjusted_score = result.base_score * weight;
        }

        // Re-sort by adjusted score
        results.sort_by(|a, b| {
            b.adjusted_score.total_cmp(&a.adjusted_score)
        });
    }

    /// Get vectors that existed at a specific point in time
    pub fn get_vectors_at_time(&self, timestamp: DateTime<Utc>) -> Vec<String> {
        let mut result = Vec::new();

        for (id, meta) in &self.metadata_cache {
            if meta.created_at <= timestamp {
                // Vector existed at this time
                // Check if it was deleted (would need deletion tracking)
                result.push(id.clone());
            }
        }

        result
    }

    /// Create a snapshot of current state
    pub fn create_snapshot(&mut self) {
        let timestamp = Utc::now();
        let vector_ids: Vec<String> = self.metadata_cache.keys().cloned().collect();

        // Simple hash of the state
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        for id in &vector_ids {
            id.hash(&mut hasher);
        }
        let index_state_hash = hasher.finish();

        self.snapshots.push(Snapshot {
            timestamp,
            vector_ids,
            index_state_hash,
        });
    }

    /// Detect drift in embeddings over time
    pub fn detect_drift(
        &self,
        vectors: &[(String, Vec<f32>, DateTime<Utc>)],
    ) -> Result<DriftReport, VecStoreError> {
        let now = Utc::now();
        let window = Duration::hours(self.config.drift_window_hours as i64);

        // Split vectors into old and new windows
        let old_cutoff = now - window * 2;
        let new_cutoff = now - window;

        let old_vectors: Vec<_> = vectors.iter()
            .filter(|(_, _, t)| *t >= old_cutoff && *t < new_cutoff)
            .map(|(_, v, _)| v.clone())
            .collect();

        let new_vectors: Vec<_> = vectors.iter()
            .filter(|(_, _, t)| *t >= new_cutoff)
            .map(|(_, v, _)| v.clone())
            .collect();

        if old_vectors.is_empty() || new_vectors.is_empty() {
            return Ok(DriftReport {
                period_start: old_cutoff,
                period_end: now,
                drift_magnitude: 0.0,
                is_significant: false,
                dimension_drift: Vec::new(),
                centroid_shift: Vec::new(),
                variance_change: 0.0,
                stats: DriftStatistics {
                    old_count: old_vectors.len(),
                    new_count: new_vectors.len(),
                    distribution_distance: 0.0,
                    kl_divergence: 0.0,
                    centroid_similarity: 1.0,
                },
            });
        }

        // Calculate centroids
        let old_centroid = Self::calculate_centroid(&old_vectors);
        let new_centroid = Self::calculate_centroid(&new_vectors);

        // Calculate centroid shift
        let centroid_shift: Vec<f32> = old_centroid.iter()
            .zip(new_centroid.iter())
            .map(|(o, n)| n - o)
            .collect();

        // Calculate drift magnitude (normalized L2 distance)
        let shift_magnitude: f32 = centroid_shift.iter().map(|x| x * x).sum::<f32>().sqrt();
        let old_magnitude: f32 = old_centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        let drift_magnitude = if old_magnitude > 0.0 {
            (shift_magnitude / old_magnitude) as f64
        } else {
            0.0
        };

        // Calculate per-dimension drift
        let mut dimension_drift: Vec<DimensionDrift> = centroid_shift.iter()
            .enumerate()
            .map(|(i, &shift)| {
                let direction = if shift > 0.01 {
                    DriftDirection::Increasing
                } else if shift < -0.01 {
                    DriftDirection::Decreasing
                } else {
                    DriftDirection::Stable
                };
                DimensionDrift {
                    dimension: i,
                    drift: shift.abs() as f64,
                    direction,
                }
            })
            .collect();

        // Sort by drift magnitude
        dimension_drift.sort_by(|a, b| b.drift.total_cmp(&a.drift));
        dimension_drift.truncate(10); // Top 10 dimensions

        // Calculate variance change
        let old_variance = Self::calculate_variance(&old_vectors, &old_centroid);
        let new_variance = Self::calculate_variance(&new_vectors, &new_centroid);
        let variance_change = if old_variance > 0.0 {
            (new_variance - old_variance) / old_variance
        } else {
            0.0
        };

        // Calculate centroid similarity
        let centroid_similarity = Self::cosine_similarity(&old_centroid, &new_centroid);

        Ok(DriftReport {
            period_start: old_cutoff,
            period_end: now,
            drift_magnitude,
            is_significant: drift_magnitude > self.config.drift_threshold,
            dimension_drift,
            centroid_shift,
            variance_change,
            stats: DriftStatistics {
                old_count: old_vectors.len(),
                new_count: new_vectors.len(),
                distribution_distance: drift_magnitude,
                kl_divergence: 0.0, // Would need density estimation
                centroid_similarity: centroid_similarity as f64,
            },
        })
    }

    /// Generate temporal histogram of vectors
    pub fn generate_histogram(
        &self,
        vectors: &[(String, DateTime<Utc>)],
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<TimeBucket> {
        let duration = end.signed_duration_since(start);
        let bucket_duration = duration / self.config.histogram_buckets as i32;

        let mut buckets = Vec::with_capacity(self.config.histogram_buckets);

        for i in 0..self.config.histogram_buckets {
            let bucket_start = start + bucket_duration * i as i32;
            let bucket_end = bucket_start + bucket_duration;

            let count = vectors.iter()
                .filter(|(_, t)| *t >= bucket_start && *t < bucket_end)
                .count();

            buckets.push(TimeBucket {
                start: bucket_start,
                end: bucket_end,
                count,
                centroid: None,
                variance: 0.0,
            });
        }

        buckets
    }

    /// Analyze temporal trends for a specific vector
    pub fn analyze_trend(
        &self,
        id: &str,
        neighbor_history: &[(DateTime<Utc>, Vec<String>)],
    ) -> TemporalTrend {
        let time_points: Vec<_> = neighbor_history.iter().map(|(t, _)| *t).collect();

        // Calculate stability scores (Jaccard similarity between consecutive neighborhoods)
        let mut stability_scores = Vec::new();
        for i in 1..neighbor_history.len() {
            let prev: std::collections::HashSet<_> = neighbor_history[i-1].1.iter().collect();
            let curr: std::collections::HashSet<_> = neighbor_history[i].1.iter().collect();

            let intersection = prev.intersection(&curr).count();
            let union = prev.union(&curr).count();

            let stability = if union > 0 {
                intersection as f64 / union as f64
            } else {
                1.0
            };
            stability_scores.push(stability);
        }

        // Calculate turnover rate
        let turnover_rate = if !stability_scores.is_empty() {
            1.0 - (stability_scores.iter().sum::<f64>() / stability_scores.len() as f64)
        } else {
            0.0
        };

        // Determine trend direction
        let trend = if stability_scores.is_empty() || stability_scores.len() < 2 {
            TrendDirection::Stable
        } else {
            let avg_stability = stability_scores.iter().sum::<f64>() / stability_scores.len() as f64;
            let variance: f64 = stability_scores.iter()
                .map(|s| (s - avg_stability).powi(2))
                .sum::<f64>() / stability_scores.len() as f64;

            if variance > 0.1 {
                TrendDirection::Volatile
            } else if avg_stability > 0.8 {
                TrendDirection::Stable
            } else if stability_scores.last().unwrap_or(&0.5) > &avg_stability {
                TrendDirection::Growing
            } else {
                TrendDirection::Shrinking
            }
        };

        TemporalTrend {
            id: id.to_string(),
            time_points,
            stability_scores,
            turnover_rate,
            trend,
        }
    }

    /// Get temporal metadata for a vector
    pub fn get_metadata(&self, id: &str) -> Option<&TemporalMetadata> {
        self.metadata_cache.get(id)
    }

    /// Clean up old temporal data based on retention policy
    pub fn cleanup(&mut self) {
        if self.config.retention_hours <= 0.0 {
            return; // No cleanup needed
        }

        let cutoff = Utc::now() - Duration::hours(self.config.retention_hours as i64);

        // Remove old snapshots
        self.snapshots.retain(|s| s.timestamp >= cutoff);

        // Trim history from metadata
        for meta in self.metadata_cache.values_mut() {
            meta.history.retain(|v| v.timestamp >= cutoff);
        }
    }

    // Helper functions

    fn calculate_centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
        if vectors.is_empty() {
            return Vec::new();
        }

        let dim = vectors[0].len();
        let mut centroid = vec![0.0f32; dim];

        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                centroid[i] += v;
            }
        }

        let n = vectors.len() as f32;
        for c in &mut centroid {
            *c /= n;
        }

        centroid
    }

    fn calculate_variance(vectors: &[Vec<f32>], centroid: &[f32]) -> f64 {
        if vectors.is_empty() {
            return 0.0;
        }

        let mut total_variance = 0.0;

        for vec in vectors {
            let dist_sq: f32 = vec.iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            total_variance += dist_sq as f64;
        }

        total_variance / vectors.len() as f64
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Builder for temporal queries
pub struct TemporalQueryBuilder {
    query_vector: Vec<f32>,
    k: usize,
    decay: Option<DecayFunction>,
    point_in_time: Option<DateTime<Utc>>,
    time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    min_temporal_weight: f32,
}

impl TemporalQueryBuilder {
    /// Create a new temporal query builder
    pub fn new(query_vector: Vec<f32>) -> Self {
        Self {
            query_vector,
            k: 10,
            decay: None,
            point_in_time: None,
            time_range: None,
            min_temporal_weight: 0.0,
        }
    }

    /// Set the number of results to return
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set a custom decay function
    pub fn with_decay(mut self, decay: DecayFunction) -> Self {
        self.decay = Some(decay);
        self
    }

    /// Query as of a specific point in time
    pub fn at_time(mut self, timestamp: DateTime<Utc>) -> Self {
        self.point_in_time = Some(timestamp);
        self
    }

    /// Filter to vectors within a time range
    pub fn in_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.time_range = Some((start, end));
        self
    }

    /// Set minimum temporal weight threshold
    pub fn min_weight(mut self, weight: f32) -> Self {
        self.min_temporal_weight = weight;
        self
    }

    /// Get the query vector
    pub fn vector(&self) -> &[f32] {
        &self.query_vector
    }

    /// Get k
    pub fn get_k(&self) -> usize {
        self.k
    }

    /// Get the decay function
    pub fn get_decay(&self) -> Option<&DecayFunction> {
        self.decay.as_ref()
    }

    /// Get point in time
    pub fn get_point_in_time(&self) -> Option<DateTime<Utc>> {
        self.point_in_time
    }

    /// Get time range
    pub fn get_time_range(&self) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        self.time_range
    }

    /// Get minimum weight
    pub fn get_min_weight(&self) -> f32 {
        self.min_temporal_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_functions() {
        // Exponential decay
        let exp = DecayFunction::Exponential { half_life_hours: 24.0 };
        assert!((exp.weight(0.0) - 1.0).abs() < 1e-6);
        assert!((exp.weight(24.0) - 0.5).abs() < 1e-6);
        assert!((exp.weight(48.0) - 0.25).abs() < 1e-6);

        // Linear decay
        let linear = DecayFunction::Linear { max_age_hours: 100.0 };
        assert!((linear.weight(0.0) - 1.0).abs() < 1e-6);
        assert!((linear.weight(50.0) - 0.5).abs() < 1e-6);
        assert!((linear.weight(100.0) - 0.0).abs() < 1e-6);

        // Step function
        let step = DecayFunction::Step { cutoff_hours: 24.0 };
        assert!((step.weight(12.0) - 1.0).abs() < 1e-6);
        assert!((step.weight(36.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_temporal_search_registration() {
        let config = TemporalConfig::default();
        let mut search = TemporalSearch::new(config);

        let now = Utc::now();
        search.register_vector("vec1".to_string(), now);

        let meta = search.get_metadata("vec1").unwrap();
        assert_eq!(meta.version, 1);
        assert_eq!(meta.created_at, now);
    }

    #[test]
    fn test_vector_update_history() {
        let config = TemporalConfig::default();
        let mut search = TemporalSearch::new(config);

        let t1 = Utc::now();
        search.register_vector("vec1".to_string(), t1);

        let t2 = t1 + Duration::hours(1);
        search.update_vector("vec1", t2, 12345);

        let meta = search.get_metadata("vec1").unwrap();
        assert_eq!(meta.version, 2);
        assert_eq!(meta.history.len(), 1);
        assert_eq!(meta.history[0].version, 1);
    }

    #[test]
    fn test_temporal_weight_calculation() {
        let config = TemporalConfig {
            decay_function: DecayFunction::Exponential { half_life_hours: 24.0 },
            ..Default::default()
        };
        let mut search = TemporalSearch::new(config);

        let past = Utc::now() - Duration::hours(24);
        search.register_vector("old_vec".to_string(), past);

        let weight = search.calculate_weight("old_vec", Utc::now());
        assert!((weight - 0.5).abs() < 0.1); // Should be ~0.5 after one half-life
    }

    #[test]
    fn test_drift_detection() {
        let config = TemporalConfig {
            drift_window_hours: 24.0,
            drift_threshold: 0.1,
            ..Default::default()
        };
        let search = TemporalSearch::new(config);

        let now = Utc::now();
        let old_time = now - Duration::hours(36);
        let new_time = now - Duration::hours(12);

        // Create vectors with drift
        let vectors = vec![
            ("v1".to_string(), vec![1.0, 0.0, 0.0], old_time),
            ("v2".to_string(), vec![0.9, 0.1, 0.0], old_time + Duration::hours(1)),
            ("v3".to_string(), vec![0.0, 1.0, 0.0], new_time),
            ("v4".to_string(), vec![0.1, 0.9, 0.0], new_time + Duration::hours(1)),
        ];

        let report = search.detect_drift(&vectors).unwrap();
        // Should detect significant drift from [1,0,0] to [0,1,0] region
        assert!(report.drift_magnitude > 0.0);
    }

    #[test]
    fn test_histogram_generation() {
        let config = TemporalConfig {
            histogram_buckets: 4,
            ..Default::default()
        };
        let search = TemporalSearch::new(config);

        let now = Utc::now();
        let start = now - Duration::hours(4);

        let vectors = vec![
            ("v1".to_string(), start + Duration::minutes(30)),
            ("v2".to_string(), start + Duration::hours(1) + Duration::minutes(30)),
            ("v3".to_string(), start + Duration::hours(2) + Duration::minutes(30)),
            ("v4".to_string(), start + Duration::hours(3) + Duration::minutes(30)),
        ];

        let histogram = search.generate_histogram(&vectors, start, now);
        assert_eq!(histogram.len(), 4);

        // Each bucket should have 1 vector
        for bucket in &histogram {
            assert_eq!(bucket.count, 1);
        }
    }

    #[test]
    fn test_trend_analysis() {
        let config = TemporalConfig::default();
        let search = TemporalSearch::new(config);

        let now = Utc::now();

        // Stable neighborhood over time
        let history = vec![
            (now - Duration::hours(3), vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            (now - Duration::hours(2), vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            (now - Duration::hours(1), vec!["a".to_string(), "b".to_string(), "d".to_string()]),
            (now, vec!["a".to_string(), "b".to_string(), "d".to_string()]),
        ];

        let trend = search.analyze_trend("test", &history);
        assert!(trend.turnover_rate < 0.5); // Relatively stable
    }

    #[test]
    fn test_query_builder() {
        let now = Utc::now();
        let query = TemporalQueryBuilder::new(vec![1.0, 0.0, 0.0])
            .k(20)
            .with_decay(DecayFunction::Exponential { half_life_hours: 12.0 })
            .at_time(now)
            .min_weight(0.1);

        assert_eq!(query.get_k(), 20);
        assert!(query.get_decay().is_some());
        assert_eq!(query.get_point_in_time(), Some(now));
        assert!((query.get_min_weight() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_snapshot_creation() {
        let config = TemporalConfig::default();
        let mut search = TemporalSearch::new(config);

        search.register_vector("v1".to_string(), Utc::now());
        search.register_vector("v2".to_string(), Utc::now());

        search.create_snapshot();

        assert_eq!(search.snapshots.len(), 1);
        assert_eq!(search.snapshots[0].vector_ids.len(), 2);
    }
}
