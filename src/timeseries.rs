//! Time-Series Vector Search
//!
//! Provides temporal-aware vector search capabilities for time-stamped embeddings.
//! Useful for event detection, temporal recommendations, and time-aware retrieval.
//!
//! ## Features
//!
//! - **Temporal Filtering**: Search within time ranges (before/after/between)
//! - **Time Decay**: Apply exponential/linear decay based on recency
//! - **Window Search**: Sliding window queries over time
//! - **Seasonal Patterns**: Group by hour/day/week for pattern detection
//! - **Efficient Indexing**: Time-based partitioning for fast queries
//!
//! ## Use Cases
//!
//! - **Event Detection**: Find similar events in recent history
//! - **Temporal Recommendations**: Prioritize recent/seasonal items
//! - **News/Social Media**: Search recent articles with recency boost
//! - **IoT/Monitoring**: Time-aware anomaly detection
//! - **Financial**: Pattern matching with time constraints
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::timeseries::{TimeSeriesIndex, TimeQuery, DecayFunction};
//! use chrono::Utc;
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut index = TimeSeriesIndex::new(128)?;
//!
//! // Add timestamped vectors
//! let now = Utc::now().timestamp();
//! index.add("event1", vec![0.1; 128], now)?;
//! index.add("event2", vec![0.2; 128], now - 3600)?; // 1 hour ago
//!
//! // Search with time decay
//! let query = TimeQuery::new(vec![0.15; 128])
//!     .with_limit(10)
//!     .with_time_decay(DecayFunction::Exponential { half_life: 3600.0 })
//!     .after(now - 7200); // Last 2 hours
//!
//! let results = index.search(&query)?;
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use chrono::{DateTime, Datelike, Timelike, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Time-series vector entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeSeriesEntry {
    pub id: String,
    pub vector: Vec<f32>,
    pub timestamp: i64, // Unix timestamp in seconds
}

/// Time decay function for relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayFunction {
    /// No decay (standard similarity search)
    None,

    /// Exponential decay: score * exp(-lambda * age)
    /// half_life: time in seconds for score to decay to 50%
    Exponential { half_life: f64 },

    /// Linear decay: score * max(0, 1 - age / max_age)
    /// max_age: maximum age in seconds
    Linear { max_age: f64 },

    /// Gaussian decay: score * exp(-(age^2) / (2 * sigma^2))
    /// sigma: standard deviation in seconds
    Gaussian { sigma: f64 },
}

impl DecayFunction {
    /// Apply decay to a similarity score based on age
    pub fn apply(&self, score: f32, age_seconds: f64) -> f32 {
        match self {
            DecayFunction::None => score,

            DecayFunction::Exponential { half_life } => {
                let lambda = 0.693147 / half_life; // ln(2) / half_life
                score * (-lambda * age_seconds).exp() as f32
            }

            DecayFunction::Linear { max_age } => {
                let decay = (1.0 - age_seconds / max_age).max(0.0);
                score * decay as f32
            }

            DecayFunction::Gaussian { sigma } => {
                let exponent = -(age_seconds.powi(2)) / (2.0 * sigma.powi(2));
                score * exponent.exp() as f32
            }
        }
    }
}

/// Temporal grouping for pattern detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalGroup {
    HourOfDay,   // 0-23
    DayOfWeek,   // 0-6 (Monday=0)
    DayOfMonth,  // 1-31
    MonthOfYear, // 1-12
}

/// Time query builder
#[derive(Clone)]
pub struct TimeQuery {
    /// Query vector
    pub vector: Vec<f32>,

    /// Maximum number of results
    pub limit: usize,

    /// Time range: start timestamp (inclusive)
    pub after: Option<i64>,

    /// Time range: end timestamp (inclusive)
    pub before: Option<i64>,

    /// Time decay function
    pub decay: DecayFunction,

    /// Reference timestamp for age calculation (default: now)
    pub reference_time: Option<i64>,
}

impl TimeQuery {
    /// Create a new time query
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            limit: 10,
            after: None,
            before: None,
            decay: DecayFunction::None,
            reference_time: None,
        }
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Filter results after timestamp (inclusive)
    pub fn after(mut self, timestamp: i64) -> Self {
        self.after = Some(timestamp);
        self
    }

    /// Filter results before timestamp (inclusive)
    pub fn before(mut self, timestamp: i64) -> Self {
        self.before = Some(timestamp);
        self
    }

    /// Apply time decay function
    pub fn with_time_decay(mut self, decay: DecayFunction) -> Self {
        self.decay = decay;
        self
    }

    /// Set reference time for age calculation (default: now)
    pub fn with_reference_time(mut self, timestamp: i64) -> Self {
        self.reference_time = Some(timestamp);
        self
    }
}

/// Time-series vector search index
///
/// Efficiently indexes vectors with timestamps for temporal queries.
/// Uses time-based partitioning for fast range scans.
pub struct TimeSeriesIndex {
    /// Vector dimension
    dimension: usize,

    /// All entries sorted by timestamp (BTreeMap for efficient range queries)
    entries: BTreeMap<i64, Vec<TimeSeriesEntry>>,

    /// Total number of vectors
    num_vectors: usize,
}

impl TimeSeriesIndex {
    /// Create a new time-series index
    pub fn new(dimension: usize) -> Result<Self> {
        Ok(Self {
            dimension,
            entries: BTreeMap::new(),
            num_vectors: 0,
        })
    }

    /// Add a timestamped vector to the index
    ///
    /// # Arguments
    /// * `id` - Unique identifier
    /// * `vector` - Vector to index
    /// * `timestamp` - Unix timestamp in seconds
    pub fn add(&mut self, id: impl Into<String>, vector: Vec<f32>, timestamp: i64) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!(
                "Vector dimension {} doesn't match index dimension {}",
                vector.len(),
                self.dimension
            ));
        }

        let entry = TimeSeriesEntry {
            id: id.into(),
            vector,
            timestamp,
        };

        self.entries
            .entry(timestamp)
            .or_insert_with(Vec::new)
            .push(entry);

        self.num_vectors += 1;

        Ok(())
    }

    /// Batch add multiple timestamped vectors
    pub fn add_batch(&mut self, entries: Vec<(String, Vec<f32>, i64)>) -> Result<()> {
        for (id, vector, timestamp) in entries {
            self.add(id, vector, timestamp)?;
        }
        Ok(())
    }

    /// Search for similar vectors with temporal constraints
    pub fn search(&self, query: &TimeQuery) -> Result<Vec<TimeSeriesResult>> {
        if query.vector.len() != self.dimension {
            return Err(anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.vector.len(),
                self.dimension
            ));
        }

        let reference_time = query
            .reference_time
            .unwrap_or_else(|| Utc::now().timestamp());

        // Get entries within time range
        let range_start = query.after.unwrap_or(i64::MIN);
        let range_end = query.before.unwrap_or(i64::MAX);

        let mut results: Vec<TimeSeriesResult> = self
            .entries
            .range(range_start..=range_end)
            .flat_map(|(_, entries)| entries)
            .par_bridge()
            .map(|entry| {
                // Compute similarity
                let distance = euclidean_distance(&query.vector, &entry.vector);
                let similarity = 1.0 / (1.0 + distance); // Convert distance to similarity [0, 1]

                // Apply time decay
                let age_seconds = (reference_time - entry.timestamp).abs() as f64;
                let score = query.decay.apply(similarity, age_seconds);

                TimeSeriesResult {
                    id: entry.id.clone(),
                    score,
                    distance,
                    timestamp: entry.timestamp,
                    age_seconds,
                }
            })
            .collect();

        // Sort by score (descending) and return top-k
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(query.limit);

        Ok(results)
    }

    /// Search within a sliding time window
    ///
    /// # Arguments
    /// * `query_vector` - Query vector
    /// * `window_size` - Window size in seconds
    /// * `k` - Results per window
    ///
    /// # Returns
    /// Results grouped by time window
    pub fn search_windows(
        &self,
        query_vector: &[f32],
        window_size: i64,
        k: usize,
    ) -> Result<Vec<WindowResult>> {
        if query_vector.len() != self.dimension {
            return Err(anyhow!("Query dimension mismatch"));
        }

        if self.entries.is_empty() {
            return Ok(Vec::new());
        }

        let mut windows = Vec::new();

        // Get time range
        let min_time = *self.entries.keys().next().unwrap();
        let max_time = *self.entries.keys().last().unwrap();

        // Process windows
        let mut window_start = min_time;
        while window_start <= max_time {
            let window_end = window_start + window_size;

            let query = TimeQuery::new(query_vector.to_vec())
                .with_limit(k)
                .after(window_start)
                .before(window_end);

            let results = self.search(&query)?;

            if !results.is_empty() {
                windows.push(WindowResult {
                    window_start,
                    window_end,
                    results,
                });
            }

            window_start += window_size;
        }

        Ok(windows)
    }

    /// Group vectors by temporal pattern
    ///
    /// Useful for detecting seasonal patterns, hourly trends, etc.
    pub fn group_by_pattern(&self, grouping: TemporalGroup) -> BTreeMap<i64, Vec<String>> {
        let mut groups: BTreeMap<i64, Vec<String>> = BTreeMap::new();

        for (_, entries) in &self.entries {
            for entry in entries {
                let dt = DateTime::from_timestamp(entry.timestamp, 0).unwrap_or_else(|| Utc::now());

                let group_key = match grouping {
                    TemporalGroup::HourOfDay => dt.hour() as i64,
                    TemporalGroup::DayOfWeek => dt.weekday().num_days_from_monday() as i64,
                    TemporalGroup::DayOfMonth => dt.day() as i64,
                    TemporalGroup::MonthOfYear => dt.month() as i64,
                };

                groups
                    .entry(group_key)
                    .or_insert_with(Vec::new)
                    .push(entry.id.clone());
            }
        }

        groups
    }

    /// Remove vectors by ID
    pub fn remove(&mut self, id: &str) -> Result<bool> {
        let mut found = false;

        for (_, entries) in &mut self.entries {
            if let Some(pos) = entries.iter().position(|e| e.id == id) {
                entries.remove(pos);
                found = true;
                self.num_vectors = self.num_vectors.saturating_sub(1);
                break;
            }
        }

        Ok(found)
    }

    /// Get statistics about the index
    pub fn stats(&self) -> TimeSeriesStats {
        let mut min_timestamp = i64::MAX;
        let mut max_timestamp = i64::MIN;
        let mut timestamps_with_data = 0;

        for (&timestamp, entries) in &self.entries {
            if !entries.is_empty() {
                min_timestamp = min_timestamp.min(timestamp);
                max_timestamp = max_timestamp.max(timestamp);
                timestamps_with_data += 1;
            }
        }

        let time_span_seconds = if min_timestamp != i64::MAX {
            (max_timestamp - min_timestamp).max(0)
        } else {
            0
        };

        let avg_vectors_per_timestamp = if timestamps_with_data > 0 {
            self.num_vectors as f32 / timestamps_with_data as f32
        } else {
            0.0
        };

        TimeSeriesStats {
            num_vectors: self.num_vectors,
            num_unique_timestamps: self.entries.len(),
            min_timestamp: if min_timestamp != i64::MAX {
                Some(min_timestamp)
            } else {
                None
            },
            max_timestamp: if max_timestamp != i64::MIN {
                Some(max_timestamp)
            } else {
                None
            },
            time_span_seconds,
            avg_vectors_per_timestamp,
        }
    }

    /// Get the number of vectors
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Time-series search result
#[derive(Debug, Clone)]
pub struct TimeSeriesResult {
    pub id: String,
    pub score: f32,       // Similarity score with decay applied
    pub distance: f32,    // Raw distance
    pub timestamp: i64,   // Result timestamp
    pub age_seconds: f64, // Age relative to reference time
}

/// Window search result
#[derive(Debug, Clone)]
pub struct WindowResult {
    pub window_start: i64,
    pub window_end: i64,
    pub results: Vec<TimeSeriesResult>,
}

/// Statistics about the time-series index
#[derive(Debug, Clone)]
pub struct TimeSeriesStats {
    pub num_vectors: usize,
    pub num_unique_timestamps: usize,
    pub min_timestamp: Option<i64>,
    pub max_timestamp: Option<i64>,
    pub time_span_seconds: i64,
    pub avg_vectors_per_timestamp: f32,
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

    fn generate_test_vectors(
        n: usize,
        dim: usize,
        start_time: i64,
    ) -> Vec<(String, Vec<f32>, i64)> {
        (0..n)
            .map(|i| {
                let vector = vec![i as f32 / n as f32; dim];
                let timestamp = start_time + (i as i64 * 3600); // 1 hour apart
                (format!("vec_{}", i), vector, timestamp)
            })
            .collect()
    }

    #[test]
    fn test_timeseries_basic() {
        let mut index = TimeSeriesIndex::new(64).unwrap();
        let now = Utc::now().timestamp();

        // Add vectors
        for i in 0..10 {
            let vector = vec![i as f32 / 10.0; 64];
            let timestamp = now - (i * 3600); // Going back in time
            index.add(format!("vec_{}", i), vector, timestamp).unwrap();
        }

        assert_eq!(index.len(), 10);

        // Search without time constraints
        let query = TimeQuery::new(vec![0.5; 64]).with_limit(5);
        let results = index.search(&query).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_timeseries_time_range() {
        let mut index = TimeSeriesIndex::new(32).unwrap();
        let now = Utc::now().timestamp();

        // Add vectors at different times
        for i in 0..10 {
            index
                .add(format!("vec_{}", i), vec![i as f32; 32], now - (i * 3600))
                .unwrap();
        }

        // Search last 5 hours (but vec_0 is now, so we get vec_0 through vec_5)
        let cutoff = now - (5 * 3600);
        let query = TimeQuery::new(vec![3.0; 32]).with_limit(10).after(cutoff);

        let results = index.search(&query).unwrap();

        // Should only get vectors from last 5 hours
        // vec_0 is at 'now', vec_5 is at 'now - 5*3600', so we expect 6 results (0-5)
        assert!(results.len() <= 6);
        for result in &results {
            assert!(result.timestamp >= cutoff);
        }
    }

    #[test]
    fn test_decay_functions() {
        // Test exponential decay
        let decay = DecayFunction::Exponential { half_life: 3600.0 };
        let score = decay.apply(1.0, 3600.0);
        assert!((score - 0.5).abs() < 0.01); // Should be ~0.5 at half-life

        // Test linear decay
        let decay = DecayFunction::Linear { max_age: 7200.0 };
        let score = decay.apply(1.0, 3600.0);
        assert!((score - 0.5).abs() < 0.01); // Should be 0.5 at half of max_age

        // Test no decay
        let decay = DecayFunction::None;
        let score = decay.apply(1.0, 10000.0);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_timeseries_with_decay() {
        let mut index = TimeSeriesIndex::new(64).unwrap();
        let now = Utc::now().timestamp();

        // Add recent and old vectors
        index.add("recent", vec![0.5; 64], now).unwrap();
        index.add("old", vec![0.5; 64], now - 7200).unwrap();

        // Search with exponential decay (favors recent)
        let query = TimeQuery::new(vec![0.5; 64])
            .with_limit(10)
            .with_time_decay(DecayFunction::Exponential { half_life: 3600.0 })
            .with_reference_time(now);

        let results = index.search(&query).unwrap();

        // Recent should score higher due to decay
        assert_eq!(results[0].id, "recent");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_batch_add() {
        let mut index = TimeSeriesIndex::new(32).unwrap();
        let now = Utc::now().timestamp();

        let batch = generate_test_vectors(20, 32, now);
        index.add_batch(batch).unwrap();

        assert_eq!(index.len(), 20);
    }

    #[test]
    fn test_remove() {
        let mut index = TimeSeriesIndex::new(32).unwrap();
        let now = Utc::now().timestamp();

        index.add("vec_1", vec![0.1; 32], now).unwrap();
        index.add("vec_2", vec![0.2; 32], now + 100).unwrap();

        assert_eq!(index.len(), 2);

        let removed = index.remove("vec_1").unwrap();
        assert!(removed);
        assert_eq!(index.len(), 1);

        let removed = index.remove("vec_1").unwrap();
        assert!(!removed);
    }

    #[test]
    fn test_stats() {
        let mut index = TimeSeriesIndex::new(64).unwrap();
        let now = Utc::now().timestamp();

        for i in 0..10 {
            index
                .add(format!("vec_{}", i), vec![i as f32; 64], now + (i * 1000))
                .unwrap();
        }

        let stats = index.stats();

        assert_eq!(stats.num_vectors, 10);
        assert_eq!(stats.num_unique_timestamps, 10);
        assert_eq!(stats.min_timestamp, Some(now));
        assert_eq!(stats.max_timestamp, Some(now + 9000));
        assert_eq!(stats.time_span_seconds, 9000);
    }

    #[test]
    fn test_window_search() {
        let mut index = TimeSeriesIndex::new(32).unwrap();
        let now = Utc::now().timestamp();

        // Add vectors across 10 hours
        for i in 0..10 {
            index
                .add(format!("vec_{}", i), vec![i as f32; 32], now + (i * 3600))
                .unwrap();
        }

        // Search with 2-hour windows
        let query_vector = vec![5.0; 32];
        let windows = index.search_windows(&query_vector, 7200, 5).unwrap();

        assert!(!windows.is_empty());

        // Verify windows don't overlap incorrectly
        for window in &windows {
            assert_eq!(window.window_end - window.window_start, 7200);
        }
    }

    #[test]
    fn test_temporal_grouping() {
        let mut index = TimeSeriesIndex::new(32).unwrap();

        // Add vectors at specific hours
        let base_time = DateTime::parse_from_rfc3339("2024-01-15T10:00:00Z")
            .unwrap()
            .timestamp();

        for i in 0..24 {
            index
                .add(
                    format!("vec_{}", i),
                    vec![i as f32; 32],
                    base_time + (i * 3600), // Every hour
                )
                .unwrap();
        }

        // Group by hour of day
        let groups = index.group_by_pattern(TemporalGroup::HourOfDay);

        // Should have 24 groups (one for each hour)
        assert_eq!(groups.len(), 24);
    }
}
