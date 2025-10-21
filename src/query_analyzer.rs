//! Query performance analysis and explain plans
//!
//! This module provides tools for analyzing query performance, generating
//! explain plans, and identifying optimization opportunities.
//!
//! ## Features
//!
//! - Query explain plans
//! - HNSW traversal statistics
//! - Filter selectivity analysis
//! - Slow query identification
//! - Performance profiling
//!
//! ## Usage
//!
//! ```no_run
//! use vecstore::{VecStore, Query};
//! use vecstore::query_analyzer::QueryAnalyzer;
//!
//! # fn main() -> anyhow::Result<()> {
//! let store = VecStore::open("vectors.db")?;
//! let mut analyzer = QueryAnalyzer::new();
//!
//! let query = Query::new(vec![0.1, 0.2, 0.3]).with_limit(10);
//!
//! // Analyze query
//! let plan = analyzer.explain(&store, &query)?;
//!
//! println!("HNSW nodes visited: {}", plan.nodes_visited);
//! println!("Distance calculations: {}", plan.distance_calculations);
//! println!("Filter selectivity: {:.2}%", plan.filter_selectivity * 100.0);
//! println!("Estimated cost: {:.2}ms", plan.estimated_cost_ms);
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

/// Query execution plan with performance statistics
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Number of HNSW graph nodes visited during search
    pub nodes_visited: usize,

    /// Number of distance calculations performed
    pub distance_calculations: usize,

    /// Number of vectors evaluated (before filtering)
    pub vectors_evaluated: usize,

    /// Number of vectors after filtering
    pub vectors_after_filter: usize,

    /// Filter selectivity (0.0 to 1.0)
    /// 1.0 = all vectors pass, 0.0 = all rejected
    pub filter_selectivity: f64,

    /// Actual query execution time
    pub execution_time: Duration,

    /// Estimated cost in milliseconds
    pub estimated_cost_ms: f64,

    /// Whether SIMD was used
    pub simd_used: bool,

    /// Whether cache was hit
    pub cache_hit: bool,

    /// Query complexity score (0-100, higher = more complex)
    pub complexity_score: f64,
}

impl QueryPlan {
    /// Check if this is a slow query (execution time > threshold)
    pub fn is_slow(&self, threshold_ms: u64) -> bool {
        self.execution_time.as_millis() as u64 > threshold_ms
    }

    /// Get efficiency score (0-100, higher = more efficient)
    pub fn efficiency_score(&self) -> f64 {
        if self.distance_calculations == 0 {
            return 0.0;
        }

        // Ideal: few distance calculations, high selectivity
        let calc_efficiency = 1.0 - (self.distance_calculations as f64 / 10000.0).min(1.0);
        let filter_efficiency = self.filter_selectivity;

        ((calc_efficiency + filter_efficiency) / 2.0) * 100.0
    }

    /// Get optimization suggestions
    pub fn suggest_optimizations(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        if self.filter_selectivity < 0.1 {
            suggestions.push(
                "Low filter selectivity (<10%). Consider more selective filters or indexing."
                    .to_string(),
            );
        }

        if self.distance_calculations > 5000 {
            suggestions.push(
                "High number of distance calculations. Consider reducing search scope or using approximate search."
                    .to_string(),
            );
        }

        if !self.simd_used {
            suggestions.push(
                "SIMD not used. Ensure CPU features are enabled for 4-8x speedup.".to_string(),
            );
        }

        if !self.cache_hit && self.execution_time.as_millis() > 10 {
            suggestions.push(
                "Cache miss. For repeated queries, enable query caching for 2-3x speedup."
                    .to_string(),
            );
        }

        suggestions
    }

    /// Format as human-readable explain plan
    pub fn format_explain(&self) -> String {
        format!(
            r#"Query Execution Plan
=====================

Execution Statistics:
  Execution time:        {:?}
  Nodes visited:         {}
  Distance calculations: {}
  Vectors evaluated:     {}
  Vectors after filter:  {}

Performance Metrics:
  Filter selectivity:    {:.2}%
  Complexity score:      {:.1}/100
  Efficiency score:      {:.1}/100
  Est. cost:             {:.2}ms

Optimizations:
  SIMD acceleration:     {}
  Cache hit:             {}

{}
"#,
            self.execution_time,
            self.nodes_visited,
            self.distance_calculations,
            self.vectors_evaluated,
            self.vectors_after_filter,
            self.filter_selectivity * 100.0,
            self.complexity_score,
            self.efficiency_score(),
            self.estimated_cost_ms,
            if self.simd_used { "✓" } else { "✗" },
            if self.cache_hit { "✓" } else { "✗" },
            if self.suggest_optimizations().is_empty() {
                "No optimization suggestions.".to_string()
            } else {
                format!(
                    "Suggestions:\n  - {}",
                    self.suggest_optimizations().join("\n  - ")
                )
            }
        )
    }
}

/// Query analyzer for performance profiling
pub struct QueryAnalyzer {
    /// Track slow queries (threshold in ms)
    slow_query_threshold_ms: u64,

    /// Slow queries log
    slow_queries: Vec<SlowQueryRecord>,

    /// Maximum slow queries to keep
    max_slow_queries: usize,
}

impl QueryAnalyzer {
    /// Create a new query analyzer
    pub fn new() -> Self {
        Self {
            slow_query_threshold_ms: 100, // 100ms default
            slow_queries: Vec::new(),
            max_slow_queries: 100,
        }
    }

    /// Set slow query threshold
    pub fn with_slow_threshold(mut self, threshold_ms: u64) -> Self {
        self.slow_query_threshold_ms = threshold_ms;
        self
    }

    /// Generate an execution plan for a query (dry run, no actual execution)
    pub fn estimate_plan(&self, vector_count: usize, has_filter: bool, limit: usize) -> QueryPlan {
        // Rough estimates based on HNSW properties
        let nodes_visited = (vector_count as f64).log2().ceil() as usize * limit;
        let distance_calculations = nodes_visited * 10; // Approx

        let vectors_evaluated = distance_calculations;
        let filter_selectivity = if has_filter { 0.5 } else { 1.0 }; // Assume 50% pass
        let vectors_after_filter = (vectors_evaluated as f64 * filter_selectivity) as usize;

        // Estimate cost: ~20ns per distance calc with SIMD
        let estimated_cost_ms = (distance_calculations as f64 * 20.0) / 1_000_000.0;

        QueryPlan {
            nodes_visited,
            distance_calculations,
            vectors_evaluated,
            vectors_after_filter,
            filter_selectivity,
            execution_time: Duration::from_secs(0),
            estimated_cost_ms,
            simd_used: true,
            cache_hit: false,
            complexity_score: Self::calculate_complexity(nodes_visited, has_filter, vector_count),
        }
    }

    /// Record a query execution
    pub fn record_execution(&mut self, plan: QueryPlan, query_text: Option<String>) {
        if plan.is_slow(self.slow_query_threshold_ms) {
            let record = SlowQueryRecord {
                timestamp: std::time::SystemTime::now(),
                execution_time: plan.execution_time,
                query_text,
                plan: plan.clone(),
            };

            self.slow_queries.push(record);

            // Keep only recent slow queries
            if self.slow_queries.len() > self.max_slow_queries {
                self.slow_queries.remove(0);
            }
        }
    }

    /// Get slow queries log
    pub fn slow_queries(&self) -> &[SlowQueryRecord] {
        &self.slow_queries
    }

    /// Clear slow queries log
    pub fn clear_slow_queries(&mut self) {
        self.slow_queries.clear();
    }

    /// Calculate query complexity score
    fn calculate_complexity(nodes_visited: usize, has_filter: bool, vector_count: usize) -> f64 {
        let base_complexity = (nodes_visited as f64 / vector_count as f64) * 50.0;
        let filter_complexity = if has_filter { 20.0 } else { 0.0 };
        let graph_complexity = (vector_count as f64).log10() * 10.0;

        (base_complexity + filter_complexity + graph_complexity).min(100.0)
    }
}

impl Default for QueryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Slow query record
#[derive(Debug, Clone)]
pub struct SlowQueryRecord {
    pub timestamp: std::time::SystemTime,
    pub execution_time: Duration,
    pub query_text: Option<String>,
    pub plan: QueryPlan,
}

impl SlowQueryRecord {
    /// Format as human-readable log entry
    pub fn format_log(&self) -> String {
        format!(
            "[{:?}] {:?} - {}",
            self.timestamp,
            self.execution_time,
            self.query_text.as_deref().unwrap_or("(no query text)")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_plan_creation() {
        let plan = QueryPlan {
            nodes_visited: 100,
            distance_calculations: 500,
            vectors_evaluated: 500,
            vectors_after_filter: 250,
            filter_selectivity: 0.5,
            execution_time: Duration::from_millis(10),
            estimated_cost_ms: 10.0,
            simd_used: true,
            cache_hit: false,
            complexity_score: 50.0,
        };

        assert!(!plan.is_slow(20));
        assert!(plan.is_slow(5));
    }

    #[test]
    fn test_efficiency_score() {
        let efficient_plan = QueryPlan {
            nodes_visited: 50,
            distance_calculations: 100,
            vectors_evaluated: 100,
            vectors_after_filter: 90,
            filter_selectivity: 0.9,
            execution_time: Duration::from_millis(2),
            estimated_cost_ms: 2.0,
            simd_used: true,
            cache_hit: false,
            complexity_score: 20.0,
        };

        assert!(efficient_plan.efficiency_score() > 90.0);
    }

    #[test]
    fn test_suggestions() {
        let plan = QueryPlan {
            nodes_visited: 1000,
            distance_calculations: 10000,
            vectors_evaluated: 10000,
            vectors_after_filter: 100,
            filter_selectivity: 0.01, // Very low selectivity
            execution_time: Duration::from_millis(50),
            estimated_cost_ms: 50.0,
            simd_used: false,
            cache_hit: false,
            complexity_score: 80.0,
        };

        let suggestions = plan.suggest_optimizations();
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("selectivity")));
        assert!(suggestions.iter().any(|s| s.contains("SIMD")));
    }

    #[test]
    fn test_analyzer_estimate() {
        let analyzer = QueryAnalyzer::new();
        let plan = analyzer.estimate_plan(100_000, true, 10);

        assert!(plan.nodes_visited > 0);
        assert!(plan.distance_calculations > 0);
        assert!(plan.estimated_cost_ms > 0.0);
    }

    #[test]
    fn test_slow_query_recording() {
        let mut analyzer = QueryAnalyzer::new().with_slow_threshold(10);

        let slow_plan = QueryPlan {
            nodes_visited: 100,
            distance_calculations: 500,
            vectors_evaluated: 500,
            vectors_after_filter: 250,
            filter_selectivity: 0.5,
            execution_time: Duration::from_millis(50), // Slow!
            estimated_cost_ms: 50.0,
            simd_used: true,
            cache_hit: false,
            complexity_score: 50.0,
        };

        analyzer.record_execution(slow_plan, Some("test query".to_string()));

        assert_eq!(analyzer.slow_queries().len(), 1);
    }

    #[test]
    fn test_explain_format() {
        let plan = QueryPlan {
            nodes_visited: 100,
            distance_calculations: 500,
            vectors_evaluated: 500,
            vectors_after_filter: 250,
            filter_selectivity: 0.5,
            execution_time: Duration::from_millis(10),
            estimated_cost_ms: 10.0,
            simd_used: true,
            cache_hit: false,
            complexity_score: 50.0,
        };

        let explain = plan.format_explain();
        assert!(explain.contains("Query Execution Plan"));
        assert!(explain.contains("100")); // nodes visited
        assert!(explain.contains("50.00%")); // selectivity
    }
}
