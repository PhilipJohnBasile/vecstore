// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! Learned Index Structures for Vector Search
//!
//! This module provides ML-optimized self-tuning index parameters:
//! - **Workload Learning**: Adapt parameters based on query patterns
//! - **Data Distribution Modeling**: Learn optimal partitioning from data
//! - **Recall Prediction**: Predict recall before executing queries
//! - **Cost Modeling**: Learn latency/recall tradeoffs automatically
//! - **Auto-Tuning**: Continuously optimize index configuration
//!
//! # Example
//!
//! ```ignore
//! use vecstore::learned_index::{LearnedIndex, LearnedIndexConfig};
//!
//! let mut index = LearnedIndex::new(LearnedIndexConfig::default());
//!
//! // Add vectors
//! index.add("vec1", vec![1.0, 0.0, 0.0])?;
//!
//! // Learn from query workload
//! index.observe_query(&query, &results, latency_ms);
//!
//! // Get optimized parameters
//! let params = index.recommend_params(&query)?;
//! println!("Recommended beam width: {}", params.beam_width);
//! ```

use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};

use crate::error::VecStoreError;

/// Configuration for learned index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedIndexConfig {
    /// Enable continuous learning
    pub continuous_learning: bool,
    /// Minimum samples before making predictions
    pub min_samples: usize,
    /// Learning rate for online updates
    pub learning_rate: f64,
    /// Target recall for optimization
    pub target_recall: f64,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f64,
    /// History window size for learning
    pub history_window: usize,
    /// Enable cost-based optimization
    pub cost_optimization: bool,
}

impl Default for LearnedIndexConfig {
    fn default() -> Self {
        Self {
            continuous_learning: true,
            min_samples: 100,
            learning_rate: 0.01,
            target_recall: 0.95,
            max_latency_ms: 100.0,
            history_window: 1000,
            cost_optimization: true,
        }
    }
}

/// Query characteristics for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryProfile {
    /// Query vector (optional, for clustering)
    pub vector: Option<Vec<f32>>,
    /// Requested k
    pub k: usize,
    /// Has filter
    pub has_filter: bool,
    /// Filter selectivity estimate (0-1)
    pub filter_selectivity: f32,
    /// Query category (if known)
    pub category: Option<String>,
}

/// Observed query result for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryObservation {
    /// Query profile
    pub profile: QueryProfile,
    /// Actual latency in milliseconds
    pub latency_ms: f64,
    /// Actual recall (if ground truth available)
    pub recall: Option<f64>,
    /// Number of results returned
    pub result_count: usize,
    /// Parameters used
    pub params_used: SearchParams,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Search parameters that can be tuned
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    /// Beam width for graph search
    pub beam_width: usize,
    /// Number of candidates to evaluate
    pub ef_search: usize,
    /// Number of probes for IVF
    pub nprobe: usize,
    /// Reranking depth
    pub rerank_depth: usize,
    /// Use approximate mode
    pub approximate: bool,
    /// Early termination threshold
    pub early_termination: f32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            beam_width: 64,
            ef_search: 100,
            nprobe: 10,
            rerank_depth: 100,
            approximate: true,
            early_termination: 0.0,
        }
    }
}

/// Recommended parameters with confidence
#[derive(Debug, Clone)]
pub struct ParamRecommendation {
    /// Recommended parameters
    pub params: SearchParams,
    /// Confidence in recommendation (0-1)
    pub confidence: f64,
    /// Predicted latency (ms)
    pub predicted_latency_ms: f64,
    /// Predicted recall (0-1)
    pub predicted_recall: f64,
    /// Reasoning for recommendation
    pub reasoning: String,
}

/// Data distribution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataDistribution {
    /// Number of vectors
    pub count: usize,
    /// Dimensionality
    pub dimensions: usize,
    /// Mean vector (centroid)
    pub centroid: Vec<f32>,
    /// Variance per dimension
    pub variance: Vec<f32>,
    /// Estimated intrinsic dimensionality
    pub intrinsic_dim: f64,
    /// Cluster structure estimate
    pub cluster_count: usize,
    /// Average distance to nearest neighbor
    pub avg_nn_distance: f32,
}

/// Cost model for search operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// Base cost in microseconds
    pub base_cost_us: f64,
    /// Cost per distance computation
    pub distance_cost_us: f64,
    /// Cost per candidate
    pub candidate_cost_us: f64,
    /// Cost per filter evaluation
    pub filter_cost_us: f64,
    /// Memory access cost
    pub memory_cost_us: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            base_cost_us: 50.0,
            distance_cost_us: 0.1,
            candidate_cost_us: 0.5,
            filter_cost_us: 0.2,
            memory_cost_us: 0.05,
        }
    }
}

impl CostModel {
    /// Estimate latency for given parameters
    pub fn estimate_latency(&self, params: &SearchParams, data: &DataDistribution) -> f64 {
        let distance_comps = params.ef_search * data.dimensions;
        let candidates = params.ef_search;

        let total_us = self.base_cost_us
            + (distance_comps as f64) * self.distance_cost_us
            + (candidates as f64) * self.candidate_cost_us
            + (data.dimensions as f64) * self.memory_cost_us;

        total_us / 1000.0 // Convert to ms
    }

    /// Update cost model based on observation
    pub fn update(&mut self, observation: &QueryObservation, learning_rate: f64) {
        let predicted = self.estimate_latency(
            &observation.params_used,
            &DataDistribution {
                count: 0,
                dimensions: observation.profile.vector.as_ref().map(|v| v.len()).unwrap_or(128),
                centroid: Vec::new(),
                variance: Vec::new(),
                intrinsic_dim: 0.0,
                cluster_count: 1,
                avg_nn_distance: 0.0,
            },
        );

        let error = observation.latency_ms - predicted;

        // Simple online update
        self.base_cost_us += learning_rate * error * 10.0;
    }
}

/// Recall predictor
#[derive(Debug, Clone)]
pub struct RecallPredictor {
    /// Learned coefficients for recall prediction
    coefficients: Vec<f64>,
    /// Feature names
    features: Vec<String>,
    /// Number of observations
    observation_count: usize,
}

impl RecallPredictor {
    /// Create a new recall predictor
    pub fn new() -> Self {
        Self {
            coefficients: vec![0.5, 0.01, 0.001, -0.0001], // Initial guesses
            features: vec![
                "base".to_string(),
                "ef_search".to_string(),
                "beam_width".to_string(),
                "filter_selectivity".to_string(),
            ],
            observation_count: 0,
        }
    }

    /// Predict recall for given parameters
    pub fn predict(&self, params: &SearchParams, profile: &QueryProfile) -> f64 {
        let features = vec![
            1.0, // base
            params.ef_search as f64,
            params.beam_width as f64,
            profile.filter_selectivity as f64,
        ];

        let raw: f64 = self.coefficients.iter()
            .zip(features.iter())
            .map(|(c, f)| c * f)
            .sum();

        // Sigmoid to bound between 0 and 1
        1.0 / (1.0 + (-raw).exp())
    }

    /// Update predictor with observation
    pub fn update(&mut self, params: &SearchParams, profile: &QueryProfile, actual_recall: f64, learning_rate: f64) {
        let predicted = self.predict(params, profile);
        let error = actual_recall - predicted;

        let features = vec![
            1.0,
            params.ef_search as f64,
            params.beam_width as f64,
            profile.filter_selectivity as f64,
        ];

        // Gradient descent update
        for (i, f) in features.iter().enumerate() {
            self.coefficients[i] += learning_rate * error * f * predicted * (1.0 - predicted);
        }

        self.observation_count += 1;
    }
}

impl Default for RecallPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Workload analyzer
#[derive(Debug, Clone)]
pub struct WorkloadAnalyzer {
    /// Query history
    history: Vec<QueryObservation>,
    /// Maximum history size
    max_history: usize,
    /// Query patterns detected
    patterns: Vec<QueryPattern>,
}

/// Detected query pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    /// Pattern name
    pub name: String,
    /// Frequency (queries per second)
    pub frequency: f64,
    /// Average k value
    pub avg_k: f64,
    /// Has filters percentage
    pub filter_percentage: f64,
    /// Best observed params for this pattern
    pub best_params: Option<SearchParams>,
}

impl WorkloadAnalyzer {
    /// Create a new workload analyzer
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history,
            patterns: Vec::new(),
        }
    }

    /// Record a query observation
    pub fn record(&mut self, observation: QueryObservation) {
        self.history.push(observation);

        // Maintain history window
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Analyze workload and detect patterns
    pub fn analyze(&mut self) -> Vec<QueryPattern> {
        if self.history.is_empty() {
            return Vec::new();
        }

        // Simple pattern detection: group by k value and filter presence
        let mut patterns: HashMap<String, Vec<&QueryObservation>> = HashMap::new();

        for obs in &self.history {
            let key = format!(
                "k={}_filter={}",
                if obs.profile.k <= 10 { "small" } else if obs.profile.k <= 100 { "medium" } else { "large" },
                obs.profile.has_filter
            );
            patterns.entry(key).or_default().push(obs);
        }

        let total_time = self.history.last()
            .and_then(|last| self.history.first().map(|first| {
                last.timestamp.duration_since(first.timestamp).unwrap_or_default().as_secs_f64()
            }))
            .unwrap_or(1.0);

        self.patterns = patterns.into_iter()
            .map(|(name, obs)| {
                let count = obs.len();
                let frequency = count as f64 / total_time.max(1.0);
                let avg_k = obs.iter().map(|o| o.profile.k as f64).sum::<f64>() / count as f64;
                let filter_pct = obs.iter().filter(|o| o.profile.has_filter).count() as f64 / count as f64;

                // Find best params (lowest latency meeting recall target)
                let best = obs.iter()
                    .filter(|o| o.recall.unwrap_or(1.0) >= 0.9)
                    .min_by(|a, b| a.latency_ms.partial_cmp(&b.latency_ms).unwrap());

                QueryPattern {
                    name,
                    frequency,
                    avg_k,
                    filter_percentage: filter_pct,
                    best_params: best.map(|o| o.params_used.clone()),
                }
            })
            .collect();

        self.patterns.clone()
    }

    /// Get summary statistics
    pub fn summary(&self) -> WorkloadSummary {
        if self.history.is_empty() {
            return WorkloadSummary::default();
        }

        let total = self.history.len();
        let avg_latency = self.history.iter().map(|o| o.latency_ms).sum::<f64>() / total as f64;
        let avg_k = self.history.iter().map(|o| o.profile.k as f64).sum::<f64>() / total as f64;
        let filter_ratio = self.history.iter().filter(|o| o.profile.has_filter).count() as f64 / total as f64;

        let p50_latency = percentile(&self.history.iter().map(|o| o.latency_ms).collect::<Vec<_>>(), 0.5);
        let p99_latency = percentile(&self.history.iter().map(|o| o.latency_ms).collect::<Vec<_>>(), 0.99);

        WorkloadSummary {
            total_queries: total,
            avg_latency_ms: avg_latency,
            p50_latency_ms: p50_latency,
            p99_latency_ms: p99_latency,
            avg_k,
            filter_ratio,
            patterns: self.patterns.clone(),
        }
    }
}

/// Workload summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkloadSummary {
    /// Total queries observed
    pub total_queries: usize,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// P50 latency (ms)
    pub p50_latency_ms: f64,
    /// P99 latency (ms)
    pub p99_latency_ms: f64,
    /// Average k value
    pub avg_k: f64,
    /// Ratio of queries with filters
    pub filter_ratio: f64,
    /// Detected patterns
    pub patterns: Vec<QueryPattern>,
}

/// Main learned index
pub struct LearnedIndex {
    config: LearnedIndexConfig,
    /// Vectors (ID -> vector)
    vectors: HashMap<String, Vec<f32>>,
    /// Data distribution model
    distribution: Option<DataDistribution>,
    /// Cost model
    cost_model: CostModel,
    /// Recall predictor
    recall_predictor: RecallPredictor,
    /// Workload analyzer
    workload: WorkloadAnalyzer,
    /// Current best parameters
    current_params: SearchParams,
    /// Auto-tune enabled
    auto_tune: bool,
}

impl LearnedIndex {
    /// Create a new learned index
    pub fn new(config: LearnedIndexConfig) -> Self {
        let history_window = config.history_window;
        Self {
            config,
            vectors: HashMap::new(),
            distribution: None,
            cost_model: CostModel::default(),
            recall_predictor: RecallPredictor::new(),
            workload: WorkloadAnalyzer::new(history_window),
            current_params: SearchParams::default(),
            auto_tune: true,
        }
    }

    /// Add a vector to the index
    pub fn add(&mut self, id: impl Into<String>, vector: Vec<f32>) -> Result<(), VecStoreError> {
        self.vectors.insert(id.into(), vector);

        // Invalidate distribution cache
        if self.vectors.len() % 100 == 0 {
            self.update_distribution();
        }

        Ok(())
    }

    /// Remove a vector from the index
    pub fn remove(&mut self, id: &str) -> Result<(), VecStoreError> {
        self.vectors.remove(id);
        Ok(())
    }

    /// Observe a query for learning
    pub fn observe_query(
        &mut self,
        profile: QueryProfile,
        latency_ms: f64,
        recall: Option<f64>,
        result_count: usize,
        params_used: SearchParams,
    ) {
        let observation = QueryObservation {
            profile: profile.clone(),
            latency_ms,
            recall,
            result_count,
            params_used: params_used.clone(),
            timestamp: std::time::SystemTime::now(),
        };

        self.workload.record(observation.clone());

        // Update models
        if self.config.continuous_learning {
            self.cost_model.update(&observation, self.config.learning_rate);

            if let Some(actual_recall) = recall {
                self.recall_predictor.update(
                    &params_used,
                    &profile,
                    actual_recall,
                    self.config.learning_rate,
                );
            }

            // Auto-tune if enabled
            if self.auto_tune && self.workload.history.len() >= self.config.min_samples {
                self.auto_tune_params();
            }
        }
    }

    /// Recommend parameters for a query
    pub fn recommend_params(&self, profile: &QueryProfile) -> ParamRecommendation {
        let base = self.current_params.clone();

        // Check if we have enough data to make predictions
        if self.workload.history.len() < self.config.min_samples {
            return ParamRecommendation {
                params: base,
                confidence: 0.0,
                predicted_latency_ms: 0.0,
                predicted_recall: 0.0,
                reasoning: "Insufficient data for prediction, using defaults".to_string(),
            };
        }

        // Try different parameter combinations
        let candidates = self.generate_param_candidates(profile);
        let mut best: Option<(SearchParams, f64, f64, f64)> = None;

        for params in candidates {
            let pred_latency = if let Some(ref dist) = self.distribution {
                self.cost_model.estimate_latency(&params, dist)
            } else {
                0.0
            };

            let pred_recall = self.recall_predictor.predict(&params, profile);

            // Score: maximize recall while keeping latency in bounds
            let score = if pred_latency <= self.config.max_latency_ms {
                pred_recall
            } else {
                pred_recall * (self.config.max_latency_ms / pred_latency)
            };

            if best.is_none() || score > best.as_ref().unwrap().3 {
                best = Some((params, pred_latency, pred_recall, score));
            }
        }

        let (params, latency, recall, _) = best.unwrap_or((base, 0.0, 0.0, 0.0));

        let confidence = (self.workload.history.len() as f64 / 1000.0).min(1.0);

        let reasoning = if profile.has_filter {
            format!(
                "Adjusted for filtered query with selectivity {:.2}",
                profile.filter_selectivity
            )
        } else {
            format!("Optimized for k={} query", profile.k)
        };

        ParamRecommendation {
            params,
            confidence,
            predicted_latency_ms: latency,
            predicted_recall: recall,
            reasoning,
        }
    }

    /// Get current search parameters
    pub fn current_params(&self) -> &SearchParams {
        &self.current_params
    }

    /// Manually set parameters
    pub fn set_params(&mut self, params: SearchParams) {
        self.current_params = params;
    }

    /// Enable/disable auto-tuning
    pub fn set_auto_tune(&mut self, enabled: bool) {
        self.auto_tune = enabled;
    }

    /// Get workload summary
    pub fn workload_summary(&mut self) -> WorkloadSummary {
        self.workload.analyze();
        self.workload.summary()
    }

    /// Get data distribution
    pub fn distribution(&self) -> Option<&DataDistribution> {
        self.distribution.as_ref()
    }

    /// Force update of data distribution model
    pub fn update_distribution(&mut self) {
        if self.vectors.is_empty() {
            self.distribution = None;
            return;
        }

        let vectors: Vec<&Vec<f32>> = self.vectors.values().collect();
        let count = vectors.len();
        let dimensions = vectors[0].len();

        // Compute centroid
        let mut centroid = vec![0.0f32; dimensions];
        for vec in &vectors {
            for (i, &v) in vec.iter().enumerate() {
                centroid[i] += v;
            }
        }
        for c in &mut centroid {
            *c /= count as f32;
        }

        // Compute variance
        let mut variance = vec![0.0f32; dimensions];
        for vec in &vectors {
            for (i, &v) in vec.iter().enumerate() {
                let diff = v - centroid[i];
                variance[i] += diff * diff;
            }
        }
        for v in &mut variance {
            *v /= count as f32;
        }

        // Estimate intrinsic dimensionality (simplified: ratio of explained variance)
        let total_var: f32 = variance.iter().sum();
        let sorted_var: Vec<f32> = {
            let mut v = variance.clone();
            v.sort_by(|a, b| b.partial_cmp(a).unwrap());
            v
        };
        let mut cumsum = 0.0;
        let mut intrinsic_dim = 0;
        for &v in &sorted_var {
            cumsum += v;
            intrinsic_dim += 1;
            if cumsum >= total_var * 0.95 {
                break;
            }
        }

        // Estimate average NN distance (sample-based)
        let sample_size = (count / 10).max(10).min(100);
        let mut nn_distances = Vec::new();
        for i in 0..sample_size.min(count) {
            let v1 = &vectors[i];
            let mut min_dist = f32::MAX;
            for (j, v2) in vectors.iter().enumerate() {
                if i != j {
                    let dist: f32 = v1.iter()
                        .zip(v2.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    min_dist = min_dist.min(dist);
                }
            }
            if min_dist < f32::MAX {
                nn_distances.push(min_dist);
            }
        }

        let avg_nn_distance = if nn_distances.is_empty() {
            0.0
        } else {
            nn_distances.iter().sum::<f32>() / nn_distances.len() as f32
        };

        self.distribution = Some(DataDistribution {
            count,
            dimensions,
            centroid,
            variance,
            intrinsic_dim: intrinsic_dim as f64,
            cluster_count: (count / 1000).max(1), // Rough estimate
            avg_nn_distance,
        });
    }

    /// Auto-tune parameters based on observations
    fn auto_tune_params(&mut self) {
        let summary = self.workload.summary();

        // Adjust ef_search based on latency
        if summary.p99_latency_ms > self.config.max_latency_ms {
            // Too slow, reduce ef_search
            self.current_params.ef_search = (self.current_params.ef_search as f64 * 0.9) as usize;
            self.current_params.ef_search = self.current_params.ef_search.max(10);
        } else if summary.p99_latency_ms < self.config.max_latency_ms * 0.5 {
            // Have headroom, increase for better recall
            self.current_params.ef_search = (self.current_params.ef_search as f64 * 1.1) as usize;
            self.current_params.ef_search = self.current_params.ef_search.min(500);
        }

        // Adjust beam width similarly
        let avg_recall = self.workload.history.iter()
            .filter_map(|o| o.recall)
            .sum::<f64>() / self.workload.history.len().max(1) as f64;

        if avg_recall < self.config.target_recall {
            self.current_params.beam_width = (self.current_params.beam_width as f64 * 1.2) as usize;
            self.current_params.beam_width = self.current_params.beam_width.min(256);
        }
    }

    fn generate_param_candidates(&self, profile: &QueryProfile) -> Vec<SearchParams> {
        let base = &self.current_params;

        let mut candidates = vec![base.clone()];

        // Variations on ef_search
        for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] {
            let ef = (base.ef_search as f64 * factor) as usize;
            if ef >= 10 && ef <= 500 {
                let mut params = base.clone();
                params.ef_search = ef;
                candidates.push(params);
            }
        }

        // Variations on beam width
        for factor in [0.5, 1.0, 1.5, 2.0] {
            let bw = (base.beam_width as f64 * factor) as usize;
            if bw >= 8 && bw <= 256 {
                let mut params = base.clone();
                params.beam_width = bw;
                candidates.push(params);
            }
        }

        // Adjust for filters
        if profile.has_filter && profile.filter_selectivity < 0.5 {
            // Low selectivity = need more candidates
            let mut params = base.clone();
            params.ef_search = (base.ef_search as f32 / profile.filter_selectivity.max(0.1)) as usize;
            params.ef_search = params.ef_search.min(500);
            candidates.push(params);
        }

        // Adjust for k
        if profile.k > 50 {
            let mut params = base.clone();
            params.ef_search = base.ef_search.max(profile.k * 2);
            params.rerank_depth = profile.k * 3;
            candidates.push(params);
        }

        candidates
    }
}

/// Calculate percentile of a sorted slice
fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((sorted.len() - 1) as f64 * p) as usize;
    sorted[idx]
}

/// Benchmark utility for parameter tuning
pub struct ParamBenchmark {
    /// Benchmark results
    results: Vec<BenchmarkResult>,
}

/// Result of a parameter benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Parameters tested
    pub params: SearchParams,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// P99 latency (ms)
    pub p99_latency_ms: f64,
    /// Recall (if ground truth available)
    pub recall: Option<f64>,
    /// Number of queries in benchmark
    pub query_count: usize,
}

impl ParamBenchmark {
    /// Create a new benchmark
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    /// Run benchmark with given parameters
    pub fn run<F>(&mut self, params: SearchParams, query_fn: F, iterations: usize) -> BenchmarkResult
    where
        F: Fn(&SearchParams) -> (Duration, Option<f64>),
    {
        let mut latencies = Vec::with_capacity(iterations);
        let mut recalls = Vec::new();

        for _ in 0..iterations {
            let (latency, recall) = query_fn(&params);
            latencies.push(latency.as_secs_f64() * 1000.0);
            if let Some(r) = recall {
                recalls.push(r);
            }
        }

        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let p99_latency = percentile(&latencies, 0.99);
        let avg_recall = if recalls.is_empty() {
            None
        } else {
            Some(recalls.iter().sum::<f64>() / recalls.len() as f64)
        };

        let result = BenchmarkResult {
            params,
            avg_latency_ms: avg_latency,
            p99_latency_ms: p99_latency,
            recall: avg_recall,
            query_count: iterations,
        };

        self.results.push(result.clone());
        result
    }

    /// Get all results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Find best parameters meeting constraints
    pub fn best_params(&self, min_recall: Option<f64>, max_latency: Option<f64>) -> Option<&BenchmarkResult> {
        self.results.iter()
            .filter(|r| {
                let recall_ok = min_recall.map(|mr| r.recall.unwrap_or(1.0) >= mr).unwrap_or(true);
                let latency_ok = max_latency.map(|ml| r.avg_latency_ms <= ml).unwrap_or(true);
                recall_ok && latency_ok
            })
            .min_by(|a, b| a.avg_latency_ms.partial_cmp(&b.avg_latency_ms).unwrap())
    }
}

impl Default for ParamBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learned_index_creation() {
        let config = LearnedIndexConfig::default();
        let index = LearnedIndex::new(config);

        assert!(index.vectors.is_empty());
        assert!(index.distribution.is_none());
    }

    #[test]
    fn test_add_vectors() {
        let config = LearnedIndexConfig::default();
        let mut index = LearnedIndex::new(config);

        index.add("v1", vec![1.0, 0.0, 0.0]).unwrap();
        index.add("v2", vec![0.0, 1.0, 0.0]).unwrap();

        assert_eq!(index.vectors.len(), 2);
    }

    #[test]
    fn test_distribution_update() {
        let config = LearnedIndexConfig::default();
        let mut index = LearnedIndex::new(config);

        for i in 0..100 {
            index.add(format!("v{}", i), vec![i as f32, 0.0, 0.0]).unwrap();
        }

        index.update_distribution();

        let dist = index.distribution().unwrap();
        assert_eq!(dist.count, 100);
        assert_eq!(dist.dimensions, 3);
        assert!(dist.avg_nn_distance > 0.0);
    }

    #[test]
    fn test_observe_query() {
        let config = LearnedIndexConfig::default();
        let mut index = LearnedIndex::new(config);

        let profile = QueryProfile {
            vector: Some(vec![1.0, 0.0, 0.0]),
            k: 10,
            has_filter: false,
            filter_selectivity: 1.0,
            category: None,
        };

        index.observe_query(
            profile,
            5.0,
            Some(0.95),
            10,
            SearchParams::default(),
        );

        let summary = index.workload_summary();
        assert_eq!(summary.total_queries, 1);
    }

    #[test]
    fn test_recommend_params_insufficient_data() {
        let config = LearnedIndexConfig::default();
        let index = LearnedIndex::new(config);

        let profile = QueryProfile {
            vector: Some(vec![1.0, 0.0, 0.0]),
            k: 10,
            has_filter: false,
            filter_selectivity: 1.0,
            category: None,
        };

        let rec = index.recommend_params(&profile);
        assert_eq!(rec.confidence, 0.0); // Insufficient data
    }

    #[test]
    fn test_recommend_params_with_data() {
        let config = LearnedIndexConfig {
            min_samples: 5, // Lower for testing
            ..Default::default()
        };
        let mut index = LearnedIndex::new(config);

        // Add some observations
        for i in 0..10 {
            let profile = QueryProfile {
                vector: Some(vec![i as f32, 0.0, 0.0]),
                k: 10,
                has_filter: false,
                filter_selectivity: 1.0,
                category: None,
            };

            index.observe_query(
                profile,
                5.0 + i as f64,
                Some(0.9 + (i as f64) * 0.01),
                10,
                SearchParams::default(),
            );
        }

        let profile = QueryProfile {
            vector: Some(vec![1.0, 0.0, 0.0]),
            k: 10,
            has_filter: false,
            filter_selectivity: 1.0,
            category: None,
        };

        let rec = index.recommend_params(&profile);
        assert!(rec.confidence > 0.0);
    }

    #[test]
    fn test_recall_predictor() {
        let mut predictor = RecallPredictor::new();

        let profile = QueryProfile {
            vector: None,
            k: 10,
            has_filter: false,
            filter_selectivity: 1.0,
            category: None,
        };

        let params = SearchParams {
            ef_search: 100,
            ..Default::default()
        };

        // Initial prediction
        let pred1 = predictor.predict(&params, &profile);

        // Update with observation
        predictor.update(&params, &profile, 0.95, 0.1);

        // Prediction should change
        let pred2 = predictor.predict(&params, &profile);
        assert!((pred2 - pred1).abs() > 0.0 || pred1 == 0.95);
    }

    #[test]
    fn test_cost_model() {
        let model = CostModel::default();

        let params = SearchParams::default();
        let dist = DataDistribution {
            count: 1000,
            dimensions: 128,
            centroid: vec![0.0; 128],
            variance: vec![1.0; 128],
            intrinsic_dim: 50.0,
            cluster_count: 10,
            avg_nn_distance: 0.1,
        };

        let latency = model.estimate_latency(&params, &dist);
        assert!(latency > 0.0);
    }

    #[test]
    fn test_workload_analyzer() {
        let mut analyzer = WorkloadAnalyzer::new(100);

        for i in 0..20 {
            let obs = QueryObservation {
                profile: QueryProfile {
                    vector: None,
                    k: if i < 10 { 10 } else { 100 },
                    has_filter: i % 2 == 0,
                    filter_selectivity: 0.5,
                    category: None,
                },
                latency_ms: 5.0 + i as f64,
                recall: Some(0.95),
                result_count: 10,
                params_used: SearchParams::default(),
                timestamp: std::time::SystemTime::now(),
            };
            analyzer.record(obs);
        }

        let patterns = analyzer.analyze();
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_param_benchmark() {
        let mut bench = ParamBenchmark::new();

        let result = bench.run(
            SearchParams::default(),
            |_params| (Duration::from_millis(5), Some(0.95)),
            10,
        );

        assert!(result.avg_latency_ms > 0.0);
        assert!(result.recall.is_some());
        assert_eq!(result.query_count, 10);
    }

    #[test]
    fn test_auto_tune() {
        let config = LearnedIndexConfig {
            min_samples: 5,
            continuous_learning: true,
            max_latency_ms: 10.0,
            target_recall: 0.95,
            ..Default::default()
        };
        let mut index = LearnedIndex::new(config);

        let initial_ef = index.current_params().ef_search;

        // Simulate slow queries
        for i in 0..10 {
            let profile = QueryProfile {
                vector: None,
                k: 10,
                has_filter: false,
                filter_selectivity: 1.0,
                category: None,
            };

            index.observe_query(
                profile,
                50.0, // Very slow
                Some(0.99),
                10,
                index.current_params().clone(),
            );
        }

        // ef_search should be reduced due to high latency
        assert!(index.current_params().ef_search <= initial_ef);
    }
}
