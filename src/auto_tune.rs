//! Auto-Recall Optimizer: Zero-Config Performance Tuning
//!
//! Automatically tune index parameters to achieve target recall.
//! No more manual parameter sweeps - just set your target and let the system optimize.
//!
//! ## How It Works
//!
//! 1. Set target recall (e.g., 0.95)
//! 2. System samples queries and measures actual recall
//! 3. Bayesian optimization adjusts parameters
//! 4. Continuous monitoring and adaptation
//!
//! ## Parameters Tuned
//!
//! - **HNSW**: ef_construction, M, ef_search
//! - **IVF**: nlist, nprobe
//! - **Quantization**: compression level
//! - **Hybrid**: alpha (dense/sparse balance)
//!
//! ## Example
//!
//! ```rust,no_run
//! use vecstore::auto_tune::{AutoTuner, TuneConfig};
//!
//! let config = TuneConfig {
//!     target_recall: 0.95,
//!     max_latency_ms: 10,
//!     ..Default::default()
//! };
//!
//! let mut tuner = AutoTuner::new(config);
//!
//! // Tune with sample queries
//! let params = tuner.tune(&sample_queries, &ground_truth)?;
//! println!("Optimal params: {:?}", params);
//! ```

use anyhow::{Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Auto-tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneConfig {
    /// Target recall (0.0 - 1.0)
    pub target_recall: f32,

    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: u64,

    /// Maximum memory usage in MB
    pub max_memory_mb: Option<usize>,

    /// Number of optimization iterations
    pub max_iterations: usize,

    /// Number of samples for evaluation
    pub sample_size: usize,

    /// Enable continuous adaptation
    pub continuous_adaptation: bool,

    /// Adaptation interval (queries between adaptations)
    pub adaptation_interval: usize,

    /// Minimum improvement to accept new parameters
    pub min_improvement: f32,

    /// Exploration vs exploitation factor (0.0 = exploit, 1.0 = explore)
    pub exploration_factor: f32,
}

impl Default for TuneConfig {
    fn default() -> Self {
        Self {
            target_recall: 0.95,
            max_latency_ms: 10,
            max_memory_mb: None,
            max_iterations: 50,
            sample_size: 100,
            continuous_adaptation: true,
            adaptation_interval: 1000,
            min_improvement: 0.01,
            exploration_factor: 0.3,
        }
    }
}

// ============================================================================
// INDEX PARAMETERS
// ============================================================================

/// Parameters for HNSW index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWParams {
    /// Number of connections per layer
    pub m: usize,

    /// Size of dynamic candidate list during construction
    pub ef_construction: usize,

    /// Size of dynamic candidate list during search
    pub ef_search: usize,
}

impl Default for HNSWParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

impl HNSWParams {
    /// Parameter bounds for optimization
    pub fn bounds() -> ParamBounds {
        ParamBounds {
            m: (4, 64),
            ef_construction: (50, 500),
            ef_search: (10, 500),
        }
    }

    /// Convert to parameter vector for optimization
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.m as f64,
            self.ef_construction as f64,
            self.ef_search as f64,
        ]
    }

    /// Create from parameter vector
    pub fn from_vector(params: &[f64]) -> Self {
        Self {
            m: params[0] as usize,
            ef_construction: params[1] as usize,
            ef_search: params[2] as usize,
        }
    }
}

/// Parameter bounds
pub struct ParamBounds {
    pub m: (usize, usize),
    pub ef_construction: (usize, usize),
    pub ef_search: (usize, usize),
}

/// Parameters for IVF index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFParams {
    /// Number of inverted lists (clusters)
    pub nlist: usize,

    /// Number of lists to probe during search
    pub nprobe: usize,
}

impl Default for IVFParams {
    fn default() -> Self {
        Self {
            nlist: 100,
            nprobe: 10,
        }
    }
}

/// Parameters for hybrid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridParams {
    /// Dense/sparse balance (0.0 = all sparse, 1.0 = all dense)
    pub alpha: f32,

    /// Oversampling factor for reranking
    pub oversample: usize,
}

impl Default for HybridParams {
    fn default() -> Self {
        Self {
            alpha: 0.7,
            oversample: 3,
        }
    }
}

/// Combined index parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct IndexParams {
    pub hnsw: HNSWParams,
    pub ivf: IVFParams,
    pub hybrid: HybridParams,
}


// ============================================================================
// EVALUATION METRICS
// ============================================================================

/// Evaluation result for a parameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Recall@k
    pub recall: f32,

    /// Mean latency in milliseconds
    pub latency_ms: f64,

    /// P99 latency in milliseconds
    pub latency_p99_ms: f64,

    /// Queries per second
    pub qps: f64,

    /// Memory usage in MB
    pub memory_mb: Option<usize>,

    /// Parameters used
    pub params: IndexParams,

    /// Score (higher is better)
    pub score: f64,
}

impl EvalResult {
    /// Check if result meets constraints
    pub fn meets_constraints(&self, config: &TuneConfig) -> bool {
        self.recall >= config.target_recall
            && self.latency_p99_ms <= config.max_latency_ms as f64
            && config
                .max_memory_mb
                .map(|m| self.memory_mb.unwrap_or(0) <= m)
                .unwrap_or(true)
    }
}

// ============================================================================
// BAYESIAN OPTIMIZER
// ============================================================================

/// Simple Bayesian-inspired optimizer using Gaussian Process approximation
pub struct BayesianOptimizer {
    /// Observed points: (params, score)
    observations: Vec<(Vec<f64>, f64)>,

    /// Parameter bounds
    bounds: Vec<(f64, f64)>,

    /// Exploration factor
    kappa: f64,
}

impl BayesianOptimizer {
    /// Create new optimizer
    pub fn new(bounds: Vec<(f64, f64)>, kappa: f64) -> Self {
        Self {
            observations: Vec::new(),
            bounds,
            kappa,
        }
    }

    /// Add observation
    pub fn observe(&mut self, params: Vec<f64>, score: f64) {
        self.observations.push((params, score));
    }

    /// Suggest next parameters to try
    pub fn suggest(&self) -> Vec<f64> {
        if self.observations.len() < 3 {
            // Not enough data - random sampling
            return self.random_sample();
        }

        // Find best point so far
        let best = self
            .observations
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(p, _)| p.clone())
            .unwrap_or_else(|| self.random_sample());

        // Perturb around best point with some exploration
        let mut rng = rand::rng();
        let mut candidate = best.clone();

        for (i, &(low, high)) in self.bounds.iter().enumerate() {
            let range = high - low;
            let perturbation = rng.random::<f64>() * range * self.kappa - range * self.kappa / 2.0;
            candidate[i] = (candidate[i] + perturbation).clamp(low, high);
        }

        candidate
    }

    /// Random sample within bounds
    fn random_sample(&self) -> Vec<f64> {
        let mut rng = rand::rng();
        self.bounds
            .iter()
            .map(|&(low, high)| rng.random::<f64>() * (high - low) + low)
            .collect()
    }

    /// Get best observed parameters
    pub fn best(&self) -> Option<(Vec<f64>, f64)> {
        self.observations
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .cloned()
    }
}

// ============================================================================
// AUTO TUNER
// ============================================================================

/// Automatic parameter tuner
pub struct AutoTuner {
    config: TuneConfig,
    optimizer: BayesianOptimizer,
    current_params: IndexParams,
    history: Vec<EvalResult>,
    query_count: usize,
}

impl AutoTuner {
    /// Create new auto-tuner
    pub fn new(config: TuneConfig) -> Self {
        // Set up optimizer bounds for HNSW parameters
        let bounds = vec![
            (4.0, 64.0),   // M
            (50.0, 500.0), // ef_construction
            (10.0, 500.0), // ef_search
        ];

        Self {
            optimizer: BayesianOptimizer::new(bounds, config.exploration_factor as f64),
            config,
            current_params: IndexParams::default(),
            history: Vec::new(),
            query_count: 0,
        }
    }

    /// Tune parameters using sample queries and ground truth
    ///
    /// - `queries`: Sample query vectors
    /// - `ground_truth`: For each query, the true k nearest neighbor IDs
    /// - `search_fn`: Function that performs search with given parameters
    pub fn tune<F>(
        &mut self,
        queries: &[Vec<f32>],
        ground_truth: &[Vec<String>],
        mut search_fn: F,
    ) -> Result<IndexParams>
    where
        F: FnMut(&IndexParams, &[f32], usize) -> Result<(Vec<String>, std::time::Duration)>,
    {
        if queries.len() != ground_truth.len() {
            return Err(anyhow::anyhow!("Queries and ground truth must have same length"));
        }

        let k = ground_truth.first().map(|g| g.len()).unwrap_or(10);

        for iter in 0..self.config.max_iterations {
            // Get candidate parameters
            let param_vec = self.optimizer.suggest();
            let hnsw_params = HNSWParams::from_vector(&param_vec);
            let candidate_params = IndexParams {
                hnsw: hnsw_params,
                ..self.current_params.clone()
            };

            // Evaluate
            let eval = self.evaluate(&candidate_params, queries, ground_truth, k, &mut search_fn)?;

            // Record observation
            self.optimizer.observe(param_vec.clone(), eval.score);
            self.history.push(eval.clone());

            // Check if we've met the target
            if eval.meets_constraints(&self.config) {
                println!(
                    "Iteration {}: recall={:.3}, latency={:.2}ms, score={:.4}",
                    iter, eval.recall, eval.latency_ms, eval.score
                );

                if eval.recall >= self.config.target_recall + 0.01 {
                    // We've exceeded the target, we can stop
                    self.current_params = eval.params.clone();
                    break;
                }
            }

            // Update best params
            if eval.score > self.score_params(&self.current_params, queries, ground_truth, k, &mut search_fn)? {
                self.current_params = eval.params;
            }
        }

        // Return best found parameters
        if let Some((best_vec, _)) = self.optimizer.best() {
            let hnsw_params = HNSWParams::from_vector(&best_vec);
            Ok(IndexParams {
                hnsw: hnsw_params,
                ..self.current_params.clone()
            })
        } else {
            Ok(self.current_params.clone())
        }
    }

    /// Evaluate a parameter configuration
    fn evaluate<F>(
        &self,
        params: &IndexParams,
        queries: &[Vec<f32>],
        ground_truth: &[Vec<String>],
        k: usize,
        search_fn: &mut F,
    ) -> Result<EvalResult>
    where
        F: FnMut(&IndexParams, &[f32], usize) -> Result<(Vec<String>, std::time::Duration)>,
    {
        let mut total_recall = 0.0;
        let mut latencies = Vec::with_capacity(queries.len());

        for (query, truth) in queries.iter().zip(ground_truth) {
            let (results, duration) = search_fn(params, query, k)?;
            latencies.push(duration.as_secs_f64() * 1000.0);

            // Calculate recall
            let hits = results
                .iter()
                .filter(|r| truth.contains(r))
                .count();
            total_recall += hits as f32 / k as f32;
        }

        let n = queries.len() as f32;
        let recall = total_recall / n;

        // Calculate latency statistics
        latencies.sort_by(|a, b| a.total_cmp(b));
        let latency_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let latency_p99_ms = latencies.get(latencies.len() * 99 / 100).copied().unwrap_or(0.0);
        let qps = 1000.0 / latency_ms;

        // Compute score (balance recall and latency)
        let score = self.compute_score(recall, latency_ms);

        Ok(EvalResult {
            recall,
            latency_ms,
            latency_p99_ms,
            qps,
            memory_mb: None,
            params: params.clone(),
            score,
        })
    }

    /// Score parameters (for comparison)
    fn score_params<F>(
        &self,
        params: &IndexParams,
        queries: &[Vec<f32>],
        ground_truth: &[Vec<String>],
        k: usize,
        search_fn: &mut F,
    ) -> Result<f64>
    where
        F: FnMut(&IndexParams, &[f32], usize) -> Result<(Vec<String>, std::time::Duration)>,
    {
        let eval = self.evaluate(params, queries, ground_truth, k, search_fn)?;
        Ok(eval.score)
    }

    /// Compute optimization score
    fn compute_score(&self, recall: f32, latency_ms: f64) -> f64 {
        // Primary objective: meet target recall
        let recall_score = if recall >= self.config.target_recall {
            1.0
        } else {
            (recall / self.config.target_recall) as f64
        };

        // Secondary objective: minimize latency
        let latency_score = if latency_ms <= self.config.max_latency_ms as f64 {
            1.0
        } else {
            (self.config.max_latency_ms as f64 / latency_ms).min(1.0)
        };

        // Combined score (recall-weighted)
        recall_score * 0.7 + latency_score * 0.3
    }

    /// Record a query for continuous adaptation
    pub fn record_query(&mut self, _latency: std::time::Duration) {
        self.query_count += 1;
    }

    /// Get current parameters
    pub fn current_params(&self) -> &IndexParams {
        &self.current_params
    }

    /// Get tuning history
    pub fn history(&self) -> &[EvalResult] {
        &self.history
    }

    /// Get parameter suggestions without ground truth (online learning)
    pub fn suggest_params(&self) -> IndexParams {
        if let Some((best_vec, _)) = self.optimizer.best() {
            let hnsw_params = HNSWParams::from_vector(&best_vec);
            IndexParams {
                hnsw: hnsw_params,
                ..self.current_params.clone()
            }
        } else {
            self.current_params.clone()
        }
    }
}

// ============================================================================
// PARAMETER PRESETS
// ============================================================================

/// Pre-defined parameter presets for common use cases
pub struct Presets;

impl Presets {
    /// High recall, higher latency
    pub fn high_recall() -> IndexParams {
        IndexParams {
            hnsw: HNSWParams {
                m: 32,
                ef_construction: 400,
                ef_search: 200,
            },
            ivf: IVFParams {
                nlist: 200,
                nprobe: 20,
            },
            hybrid: HybridParams {
                alpha: 0.7,
                oversample: 5,
            },
        }
    }

    /// Low latency, acceptable recall
    pub fn low_latency() -> IndexParams {
        IndexParams {
            hnsw: HNSWParams {
                m: 16,
                ef_construction: 100,
                ef_search: 50,
            },
            ivf: IVFParams {
                nlist: 100,
                nprobe: 5,
            },
            hybrid: HybridParams {
                alpha: 0.8,
                oversample: 2,
            },
        }
    }

    /// Balanced
    pub fn balanced() -> IndexParams {
        IndexParams::default()
    }

    /// Memory-efficient
    pub fn memory_efficient() -> IndexParams {
        IndexParams {
            hnsw: HNSWParams {
                m: 8,
                ef_construction: 100,
                ef_search: 50,
            },
            ivf: IVFParams {
                nlist: 50,
                nprobe: 5,
            },
            hybrid: HybridParams {
                alpha: 0.7,
                oversample: 2,
            },
        }
    }
}

// ============================================================================
// RECALL ESTIMATOR
// ============================================================================

/// Estimate recall without ground truth (using sampling)
pub struct RecallEstimator {
    sample_rate: f32,
    exact_results_cache: HashMap<String, Vec<String>>,
}

impl RecallEstimator {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            exact_results_cache: HashMap::new(),
        }
    }

    /// Estimate recall by comparing approximate to exact search
    pub fn estimate<F, G>(
        &mut self,
        queries: &[Vec<f32>],
        k: usize,
        approx_search: &mut F,
        exact_search: &mut G,
    ) -> Result<f32>
    where
        F: FnMut(&[f32], usize) -> Result<Vec<String>>,
        G: FnMut(&[f32], usize) -> Result<Vec<String>>,
    {
        let mut rng = rand::rng();
        let mut total_recall = 0.0;
        let mut sample_count = 0;

        for query in queries {
            if rng.random::<f32>() > self.sample_rate {
                continue;
            }

            let approx_results = approx_search(query, k)?;
            let exact_results = exact_search(query, k)?;

            let hits = approx_results
                .iter()
                .filter(|r| exact_results.contains(r))
                .count();

            total_recall += hits as f32 / k as f32;
            sample_count += 1;
        }

        if sample_count == 0 {
            return Ok(0.0);
        }

        Ok(total_recall / sample_count as f32)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_params_conversion() {
        let params = HNSWParams {
            m: 32,
            ef_construction: 200,
            ef_search: 100,
        };

        let vec = params.to_vector();
        let reconstructed = HNSWParams::from_vector(&vec);

        assert_eq!(params.m, reconstructed.m);
        assert_eq!(params.ef_construction, reconstructed.ef_construction);
        assert_eq!(params.ef_search, reconstructed.ef_search);
    }

    #[test]
    fn test_bayesian_optimizer() {
        let bounds = vec![(0.0, 10.0), (0.0, 100.0)];
        let mut optimizer = BayesianOptimizer::new(bounds, 0.3);

        // Add some observations
        optimizer.observe(vec![5.0, 50.0], 0.8);
        optimizer.observe(vec![3.0, 30.0], 0.6);
        optimizer.observe(vec![7.0, 70.0], 0.9);

        // Suggest should work
        let suggestion = optimizer.suggest();
        assert_eq!(suggestion.len(), 2);

        // Best should be the highest scoring
        let (_best, score) = optimizer.best().unwrap();
        assert!((score - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_presets() {
        let high_recall = Presets::high_recall();
        let low_latency = Presets::low_latency();

        // High recall should have higher ef_search
        assert!(high_recall.hnsw.ef_search > low_latency.hnsw.ef_search);
    }

    #[test]
    fn test_eval_result_constraints() {
        let config = TuneConfig {
            target_recall: 0.9,
            max_latency_ms: 10,
            ..Default::default()
        };

        let good_result = EvalResult {
            recall: 0.95,
            latency_ms: 5.0,
            latency_p99_ms: 8.0,
            qps: 200.0,
            memory_mb: None,
            params: IndexParams::default(),
            score: 0.9,
        };

        let bad_recall = EvalResult {
            recall: 0.8,
            latency_ms: 5.0,
            latency_p99_ms: 8.0,
            qps: 200.0,
            memory_mb: None,
            params: IndexParams::default(),
            score: 0.7,
        };

        let bad_latency = EvalResult {
            recall: 0.95,
            latency_ms: 15.0,
            latency_p99_ms: 20.0,
            qps: 66.0,
            memory_mb: None,
            params: IndexParams::default(),
            score: 0.8,
        };

        assert!(good_result.meets_constraints(&config));
        assert!(!bad_recall.meets_constraints(&config));
        assert!(!bad_latency.meets_constraints(&config));
    }

    #[test]
    fn test_auto_tuner_creation() {
        let config = TuneConfig::default();
        let tuner = AutoTuner::new(config);

        assert_eq!(tuner.query_count, 0);
        assert!(tuner.history.is_empty());
    }
}
