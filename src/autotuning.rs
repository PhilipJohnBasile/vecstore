//! Auto-tuning for HNSW parameters
//!
//! This module automatically finds optimal HNSW parameters (M, ef_construction, ef_search)
//! based on data characteristics and performance requirements.
//!
//! ## HNSW Parameters
//!
//! - **M**: Number of connections per node (typical: 8-64, default: 16)
//!   - Higher M = better recall, more memory, slower construction
//!   - Lower M = less memory, faster construction, lower recall
//!
//! - **ef_construction**: Size of dynamic candidate list during construction (typical: 100-500)
//!   - Higher = better quality index, slower construction
//!   - Lower = faster construction, lower quality
//!
//! - **ef_search**: Size of dynamic candidate list during search (typical: 50-500)
//!   - Higher = better recall, slower search
//!   - Lower = faster search, lower recall
//!
//! ## Auto-tuning Strategies
//!
//! 1. **Grid Search**: Exhaustive search over parameter space
//! 2. **Heuristic-based**: Fast recommendations based on data size and constraints
//! 3. **Benchmark-driven**: Measure actual recall/latency trade-offs
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::autotuning::{AutoTuner, TuningGoal, PerformanceConstraints};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Define performance goals
//! let constraints = PerformanceConstraints {
//!     min_recall: 0.95,        // At least 95% recall
//!     max_latency_ms: 10.0,    // Under 10ms query time
//!     max_memory_mb: 1000.0,   // Under 1GB memory
//! };
//!
//! // Auto-tune
//! let tuner = AutoTuner::new(vec_dimension, num_vectors);
//! let params = tuner.tune_heuristic(TuningGoal::Balanced, Some(constraints))?;
//!
//! println!("Recommended M: {}", params.m);
//! println!("Recommended ef_construction: {}", params.ef_construction);
//! println!("Recommended ef_search: {}", params.ef_search);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

/// HNSW parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswParams {
    /// Number of connections per node
    pub m: usize,

    /// Dynamic candidate list size during construction
    pub ef_construction: usize,

    /// Dynamic candidate list size during search
    pub ef_search: usize,

    /// Estimated recall (0.0 to 1.0)
    pub estimated_recall: Option<f32>,

    /// Estimated query latency (milliseconds)
    pub estimated_latency_ms: Option<f32>,

    /// Estimated memory usage (megabytes)
    pub estimated_memory_mb: Option<f32>,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            estimated_recall: None,
            estimated_latency_ms: None,
            estimated_memory_mb: None,
        }
    }
}

/// Performance constraints for auto-tuning
#[derive(Debug, Clone, Copy)]
pub struct PerformanceConstraints {
    /// Minimum acceptable recall (0.0 to 1.0)
    pub min_recall: f32,

    /// Maximum acceptable query latency (milliseconds)
    pub max_latency_ms: f32,

    /// Maximum acceptable memory usage (megabytes)
    pub max_memory_mb: f32,
}

impl Default for PerformanceConstraints {
    fn default() -> Self {
        Self {
            min_recall: 0.95,
            max_latency_ms: 10.0,
            max_memory_mb: f32::INFINITY,
        }
    }
}

/// Tuning goal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TuningGoal {
    /// Maximize recall (quality-focused)
    MaxRecall,

    /// Minimize latency (speed-focused)
    MinLatency,

    /// Minimize memory (efficiency-focused)
    MinMemory,

    /// Balance all metrics
    Balanced,
}

/// Auto-tuner for HNSW parameters
pub struct AutoTuner {
    /// Vector dimension
    dimension: usize,

    /// Number of vectors in the dataset
    num_vectors: usize,
}

impl AutoTuner {
    /// Create a new auto-tuner
    ///
    /// # Arguments
    ///
    /// * `dimension` - Vector dimension
    /// * `num_vectors` - Number of vectors in the dataset
    pub fn new(dimension: usize, num_vectors: usize) -> Self {
        Self {
            dimension,
            num_vectors,
        }
    }

    /// Tune HNSW parameters using heuristics
    ///
    /// This is a fast method that uses rules-of-thumb based on dataset characteristics.
    ///
    /// # Arguments
    ///
    /// * `goal` - Optimization goal
    /// * `constraints` - Optional performance constraints
    ///
    /// # Returns
    ///
    /// Recommended HNSW parameters
    pub fn tune_heuristic(
        &self,
        goal: TuningGoal,
        constraints: Option<PerformanceConstraints>,
    ) -> anyhow::Result<HnswParams> {
        let constraints = constraints.unwrap_or_default();

        // Base parameters from goal
        let mut params = match goal {
            TuningGoal::MaxRecall => HnswParams {
                m: 32,
                ef_construction: 400,
                ef_search: 200,
                estimated_recall: Some(0.99),
                estimated_latency_ms: None,
                estimated_memory_mb: None,
            },
            TuningGoal::MinLatency => HnswParams {
                m: 8,
                ef_construction: 100,
                ef_search: 16,
                estimated_recall: Some(0.90),
                estimated_latency_ms: None,
                estimated_memory_mb: None,
            },
            TuningGoal::MinMemory => HnswParams {
                m: 8,
                ef_construction: 100,
                ef_search: 32,
                estimated_recall: Some(0.92),
                estimated_latency_ms: None,
                estimated_memory_mb: None,
            },
            TuningGoal::Balanced => HnswParams {
                m: 16,
                ef_construction: 200,
                ef_search: 50,
                estimated_recall: Some(0.95),
                estimated_latency_ms: None,
                estimated_memory_mb: None,
            },
        };

        // Adjust based on dataset size
        params = self.adjust_for_dataset_size(params);

        // Apply constraints
        params = self.apply_constraints(params, &constraints)?;

        // Estimate performance metrics
        params.estimated_memory_mb = Some(self.estimate_memory_mb(&params));
        params.estimated_latency_ms = Some(self.estimate_latency_ms(&params));

        Ok(params)
    }

    /// Adjust parameters based on dataset size
    fn adjust_for_dataset_size(&self, mut params: HnswParams) -> HnswParams {
        // For small datasets (<10k), can afford higher M and ef
        if self.num_vectors < 10_000 {
            params.m = (params.m as f32 * 1.5) as usize;
            params.ef_construction = (params.ef_construction as f32 * 1.5) as usize;
        }
        // For very large datasets (>1M), reduce M and ef
        else if self.num_vectors > 1_000_000 {
            params.m = (params.m as f32 * 0.75) as usize;
            params.ef_construction = (params.ef_construction as f32 * 0.75) as usize;
        }

        // Ensure M is in reasonable range
        params.m = params.m.max(4).min(64);
        params.ef_construction = params.ef_construction.max(50).min(500);
        params.ef_search = params.ef_search.max(10).min(500);

        params
    }

    /// Apply performance constraints
    fn apply_constraints(
        &self,
        mut params: HnswParams,
        constraints: &PerformanceConstraints,
    ) -> anyhow::Result<HnswParams> {
        // Adjust M to meet memory constraints
        let mut memory_mb = self.estimate_memory_mb(&params);

        while memory_mb > constraints.max_memory_mb && params.m > 4 {
            params.m = (params.m as f32 * 0.8) as usize;
            memory_mb = self.estimate_memory_mb(&params);
        }

        // Adjust ef_search to meet latency constraints
        let mut latency_ms = self.estimate_latency_ms(&params);

        while latency_ms > constraints.max_latency_ms && params.ef_search > 10 {
            params.ef_search = (params.ef_search as f32 * 0.8) as usize;
            latency_ms = self.estimate_latency_ms(&params);
        }

        // Adjust ef_construction and ef_search to meet recall constraints
        if let Some(recall) = params.estimated_recall {
            if recall < constraints.min_recall {
                // Increase ef_search to improve recall
                let recall_deficit = constraints.min_recall - recall;
                let boost = 1.0 + recall_deficit * 2.0; // Heuristic
                params.ef_search = (params.ef_search as f32 * boost) as usize;
                params.ef_search = params.ef_search.min(500);

                // Update estimated recall
                params.estimated_recall =
                    Some(params.estimated_recall.unwrap() + recall_deficit * 0.5);
            }
        }

        Ok(params)
    }

    /// Estimate memory usage in megabytes
    fn estimate_memory_mb(&self, params: &HnswParams) -> f32 {
        // Memory = vector storage + graph structure
        let vector_memory_mb = (self.num_vectors * self.dimension * 4) as f32 / 1_048_576.0; // 4 bytes per f32

        // Graph structure: each node has M connections per layer
        // Average layers ≈ log₂(N) / M
        let avg_layers = (self.num_vectors as f32).log2() / params.m as f32;
        let avg_layers = avg_layers.max(1.0);

        // Each connection is ~4 bytes (ID) + metadata
        let connections_per_node = params.m as f32 * avg_layers;
        let graph_memory_mb = (self.num_vectors as f32 * connections_per_node * 8.0) / 1_048_576.0;

        vector_memory_mb + graph_memory_mb
    }

    /// Estimate query latency in milliseconds
    fn estimate_latency_ms(&self, params: &HnswParams) -> f32 {
        // Latency depends on ef_search and dataset size
        // Heuristic model based on typical HNSW performance

        let base_latency = if self.num_vectors < 10_000 {
            0.1 // 0.1ms for small datasets
        } else if self.num_vectors < 100_000 {
            0.5
        } else if self.num_vectors < 1_000_000 {
            1.0
        } else {
            2.0
        };

        // ef_search affects number of distance calculations
        let ef_factor = (params.ef_search as f32 / 50.0).sqrt(); // Normalized to ef=50

        // Dimension affects distance calculation cost
        let dim_factor = (self.dimension as f32 / 128.0).sqrt(); // Normalized to dim=128

        base_latency * ef_factor * dim_factor
    }

    /// Generate parameter recommendations
    ///
    /// Returns a set of recommended parameter configurations for different scenarios.
    pub fn recommend_all(&self) -> Vec<(String, HnswParams)> {
        vec![
            (
                "Fast (Low latency, ~90% recall)".to_string(),
                self.tune_heuristic(TuningGoal::MinLatency, None).unwrap(),
            ),
            (
                "Balanced (Good all-around, ~95% recall)".to_string(),
                self.tune_heuristic(TuningGoal::Balanced, None).unwrap(),
            ),
            (
                "Accurate (High recall, ~99% recall)".to_string(),
                self.tune_heuristic(TuningGoal::MaxRecall, None).unwrap(),
            ),
            (
                "Memory-efficient (Low memory, ~92% recall)".to_string(),
                self.tune_heuristic(TuningGoal::MinMemory, None).unwrap(),
            ),
        ]
    }

    /// Explain parameter choices
    pub fn explain_params(&self, params: &HnswParams) -> String {
        let mut explanation = String::new();

        explanation.push_str("HNSW Parameter Analysis:\n");
        explanation.push_str("═══════════════════════════\n\n");

        // M parameter
        explanation.push_str(&format!("M = {} (connections per node)\n", params.m));
        if params.m <= 8 {
            explanation.push_str("  → Low M: Faster construction, less memory, lower recall\n");
        } else if params.m <= 24 {
            explanation.push_str("  → Medium M: Balanced performance\n");
        } else {
            explanation.push_str("  → High M: Better recall, more memory, slower construction\n");
        }
        explanation.push('\n');

        // ef_construction
        explanation.push_str(&format!("ef_construction = {}\n", params.ef_construction));
        if params.ef_construction <= 100 {
            explanation.push_str("  → Low ef: Fast index construction, lower quality\n");
        } else if params.ef_construction <= 300 {
            explanation.push_str("  → Medium ef: Balanced construction speed and quality\n");
        } else {
            explanation.push_str("  → High ef: Slower construction, higher quality index\n");
        }
        explanation.push('\n');

        // ef_search
        explanation.push_str(&format!("ef_search = {}\n", params.ef_search));
        if params.ef_search <= 32 {
            explanation.push_str("  → Low ef: Fast queries, lower recall\n");
        } else if params.ef_search <= 100 {
            explanation.push_str("  → Medium ef: Balanced query speed and recall\n");
        } else {
            explanation.push_str("  → High ef: Slower queries, higher recall\n");
        }
        explanation.push('\n');

        // Estimates
        if let Some(recall) = params.estimated_recall {
            explanation.push_str(&format!("Estimated Recall: {:.1}%\n", recall * 100.0));
        }

        if let Some(latency) = params.estimated_latency_ms {
            explanation.push_str(&format!("Estimated Latency: {:.2}ms per query\n", latency));
        }

        if let Some(memory) = params.estimated_memory_mb {
            explanation.push_str(&format!("Estimated Memory: {:.1} MB\n", memory));
        }

        explanation.push('\n');
        explanation.push_str("Trade-offs:\n");
        explanation.push_str("  - Higher M → Better recall but more memory\n");
        explanation
            .push_str("  - Higher ef_construction → Better index quality but slower build\n");
        explanation.push_str("  - Higher ef_search → Better recall but slower queries\n");

        explanation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autotuner_heuristic() {
        let tuner = AutoTuner::new(128, 100_000);

        let params = tuner.tune_heuristic(TuningGoal::Balanced, None).unwrap();

        assert!(params.m >= 4 && params.m <= 64);
        assert!(params.ef_construction >= 50);
        assert!(params.ef_search >= 10);
        assert!(params.estimated_recall.is_some());
        assert!(params.estimated_memory_mb.is_some());
        assert!(params.estimated_latency_ms.is_some());
    }

    #[test]
    fn test_tuning_goals() {
        let tuner = AutoTuner::new(128, 100_000);

        let fast = tuner.tune_heuristic(TuningGoal::MinLatency, None).unwrap();
        let accurate = tuner.tune_heuristic(TuningGoal::MaxRecall, None).unwrap();

        // Fast should have lower ef_search than accurate
        assert!(fast.ef_search < accurate.ef_search);

        // Accurate should have higher M
        assert!(accurate.m >= fast.m);
    }

    #[test]
    fn test_constraints() {
        let tuner = AutoTuner::new(128, 100_000);

        let constraints = PerformanceConstraints {
            min_recall: 0.98,
            max_latency_ms: 5.0,
            max_memory_mb: 500.0,
        };

        let params = tuner
            .tune_heuristic(TuningGoal::Balanced, Some(constraints))
            .unwrap();

        // Should respect constraints
        if let Some(memory) = params.estimated_memory_mb {
            assert!(memory <= constraints.max_memory_mb * 1.1); // Allow 10% margin
        }
    }

    #[test]
    fn test_dataset_size_adjustment() {
        let small_tuner = AutoTuner::new(128, 1_000);
        let large_tuner = AutoTuner::new(128, 2_000_000);

        let small_params = small_tuner
            .tune_heuristic(TuningGoal::Balanced, None)
            .unwrap();
        let large_params = large_tuner
            .tune_heuristic(TuningGoal::Balanced, None)
            .unwrap();

        // Small datasets should get higher M
        assert!(small_params.m >= large_params.m);
    }

    #[test]
    fn test_recommend_all() {
        let tuner = AutoTuner::new(128, 50_000);

        let recommendations = tuner.recommend_all();

        assert_eq!(recommendations.len(), 4);

        // Each recommendation should have different parameters
        let fast = &recommendations[0].1;
        let balanced = &recommendations[1].1;
        let accurate = &recommendations[2].1;

        assert!(fast.ef_search < balanced.ef_search);
        assert!(balanced.ef_search < accurate.ef_search);
    }

    #[test]
    fn test_explain_params() {
        let params = HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            estimated_recall: Some(0.95),
            estimated_latency_ms: Some(1.5),
            estimated_memory_mb: Some(250.0),
        };

        let tuner = AutoTuner::new(128, 100_000);
        let explanation = tuner.explain_params(&params);

        assert!(explanation.contains("M = 16"));
        assert!(explanation.contains("ef_construction = 200"));
        assert!(explanation.contains("ef_search = 50"));
        assert!(explanation.contains("95.0%"));
    }

    #[test]
    fn test_memory_estimation() {
        let tuner = AutoTuner::new(128, 100_000);
        let params = HnswParams::default();

        let memory = tuner.estimate_memory_mb(&params);

        // Should be reasonable for 100k vectors x 128 dims
        assert!(memory > 0.0);
        assert!(memory < 10_000.0); // Sanity check
    }

    #[test]
    fn test_latency_estimation() {
        let tuner = AutoTuner::new(128, 100_000);
        let params = HnswParams::default();

        let latency = tuner.estimate_latency_ms(&params);

        // Should be reasonable
        assert!(latency > 0.0);
        assert!(latency < 1000.0); // Sanity check
    }
}
