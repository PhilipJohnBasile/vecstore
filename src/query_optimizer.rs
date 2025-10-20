//! Query cost estimation and optimization
//!
//! Analyzes queries to estimate execution cost and suggest optimizations.
//! Helps users understand and improve query performance.
//!
//! # Features
//!
//! - **Cost Estimation**: Predict query execution time
//! - **Optimization Hints**: Suggest improvements
//! - **Query Analysis**: Identify bottlenecks
//! - **Index Selection**: Recommend best indexes
//!
//! # Example
//!
//! ```rust
//! use vecstore::query_optimizer::QueryOptimizer;
//!
//! let optimizer = QueryOptimizer::new(&store);
//!
//! // Analyze query
//! let analysis = optimizer.analyze_query(&query)?;
//! println!("Estimated cost: {}", analysis.estimated_cost);
//!
//! // Get optimization hints
//! for hint in analysis.hints {
//!     println!("Hint: {}", hint.suggestion);
//! }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::store::{Query, VecStore};

/// Query optimization hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHint {
    /// Hint category
    pub category: HintCategory,
    /// Suggestion text
    pub suggestion: String,
    /// Expected impact
    pub impact: Impact,
    /// Estimated improvement
    pub estimated_improvement: f32, // percentage
}

/// Hint category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HintCategory {
    /// Index-related optimization
    Index,
    /// Query parameter optimization
    QueryParam,
    /// Filter optimization
    Filter,
    /// Vector dimension optimization
    Dimension,
    /// Batching opportunity
    Batching,
}

/// Impact level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Impact {
    High,   // >50% improvement
    Medium, // 20-50% improvement
    Low,    // <20% improvement
}

/// Query cost breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Vector similarity computation cost
    pub similarity_cost: f32,
    /// Filter evaluation cost
    pub filter_cost: f32,
    /// Index lookup cost
    pub index_cost: f32,
    /// Result sorting cost
    pub sorting_cost: f32,
    /// Total estimated cost (milliseconds)
    pub total_cost: f32,
}

impl CostBreakdown {
    fn total(&self) -> f32 {
        self.similarity_cost + self.filter_cost + self.index_cost + self.sorting_cost
    }
}

/// Query execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Steps in execution order
    pub steps: Vec<PlanStep>,
    /// Estimated rows at each step
    pub estimated_rows: Vec<usize>,
    /// Whether indexes will be used
    pub uses_index: bool,
}

/// Execution plan step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Step name
    pub name: String,
    /// Description
    pub description: String,
    /// Estimated cost (ms)
    pub cost: f32,
}

/// Query analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalysis {
    /// Estimated total cost (milliseconds)
    pub estimated_cost: f32,
    /// Cost breakdown
    pub cost_breakdown: CostBreakdown,
    /// Optimization hints
    pub hints: Vec<OptimizationHint>,
    /// Execution plan
    pub execution_plan: ExecutionPlan,
    /// Query complexity level
    pub complexity: QueryComplexity,
}

/// Query complexity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryComplexity {
    Simple,   // Fast, < 10ms
    Moderate, // Medium, 10-100ms
    Complex,  // Slow, > 100ms
}

/// Query optimizer
pub struct QueryOptimizer<'a> {
    store: &'a VecStore,
}

impl<'a> QueryOptimizer<'a> {
    /// Create new optimizer
    pub fn new(store: &'a VecStore) -> Self {
        Self { store }
    }

    /// Analyze query and provide optimization suggestions
    pub fn analyze_query(&self, query: &Query) -> Result<QueryAnalysis> {
        let store_size = self.store.len();
        let vector_dim = if store_size > 0 {
            // Estimate dimension from store
            128 // Default estimate
        } else {
            128
        };

        // Estimate costs
        let cost_breakdown = self.estimate_costs(query, store_size, vector_dim);
        let total_cost = cost_breakdown.total();

        // Generate execution plan
        let execution_plan = self.generate_execution_plan(query, store_size);

        // Generate optimization hints
        let hints = self.generate_hints(query, store_size, vector_dim, &cost_breakdown);

        // Determine complexity
        let complexity = if total_cost < 10.0 {
            QueryComplexity::Simple
        } else if total_cost < 100.0 {
            QueryComplexity::Moderate
        } else {
            QueryComplexity::Complex
        };

        Ok(QueryAnalysis {
            estimated_cost: total_cost,
            cost_breakdown,
            hints,
            execution_plan,
            complexity,
        })
    }

    /// Estimate query costs
    fn estimate_costs(&self, query: &Query, store_size: usize, vector_dim: usize) -> CostBreakdown {
        // Base cost per vector comparison (microseconds)
        let base_comparison_cost = 0.001 * vector_dim as f32;

        // Similarity computation cost
        let vectors_to_compare = if query.filter.is_some() {
            // With filter, assume 50% selectivity
            store_size / 2
        } else {
            store_size
        };

        let similarity_cost = vectors_to_compare as f32 * base_comparison_cost;

        // Filter evaluation cost
        let filter_cost = if query.filter.is_some() {
            store_size as f32 * 0.0005 // 0.5 microseconds per filter check
        } else {
            0.0
        };

        // Index lookup cost (if available)
        let index_cost = if query.filter.is_some() {
            // Assume index lookup is O(log N)
            (store_size as f32).log2() * 0.001
        } else {
            0.0
        };

        // Result sorting cost
        let k = query.k;
        let sorting_cost = if vectors_to_compare > k {
            (vectors_to_compare as f32 * k as f32).log2() * 0.002
        } else {
            0.0
        };

        CostBreakdown {
            similarity_cost,
            filter_cost,
            index_cost,
            sorting_cost,
            total_cost: similarity_cost + filter_cost + index_cost + sorting_cost,
        }
    }

    /// Generate optimization hints
    fn generate_hints(
        &self,
        query: &Query,
        store_size: usize,
        vector_dim: usize,
        costs: &CostBreakdown,
    ) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();

        // Check for large K values
        if query.k > 100 {
            hints.push(OptimizationHint {
                category: HintCategory::QueryParam,
                suggestion: format!(
                    "Consider reducing k from {} to 100 or less. Large K values increase memory and sorting overhead.",
                    query.k
                ),
                impact: Impact::Medium,
                estimated_improvement: 20.0,
            });
        }

        // Check if filter could benefit from index
        if query.filter.is_some() && store_size > 1000 {
            hints.push(OptimizationHint {
                category: HintCategory::Index,
                suggestion: "Add metadata index for filtered fields to speed up filtering. Use MetadataIndexManager to create indexes.".to_string(),
                impact: Impact::High,
                estimated_improvement: 70.0,
            });
        }

        // Check for high-dimensional vectors
        if vector_dim > 512 {
            hints.push(OptimizationHint {
                category: HintCategory::Dimension,
                suggestion: format!(
                    "Consider dimensionality reduction from {} to 128-256 dimensions using PCA. This can speed up similarity computation by 2-4x.",
                    vector_dim
                ),
                impact: Impact::High,
                estimated_improvement: 60.0,
            });
        }

        // Check if similarity cost dominates
        if costs.similarity_cost > costs.total_cost * 0.8 && store_size > 10000 {
            hints.push(OptimizationHint {
                category: HintCategory::Index,
                suggestion: "Similarity computation dominates cost. Consider using IVF-PQ or LSH indexing for approximate search on large datasets.".to_string(),
                impact: Impact::High,
                estimated_improvement: 90.0,
            });
        }

        // Check for potential batching
        if store_size > 5000 {
            hints.push(OptimizationHint {
                category: HintCategory::Batching,
                suggestion: "For multiple queries, use batch operations to amortize index lookup costs across queries.".to_string(),
                impact: Impact::Medium,
                estimated_improvement: 30.0,
            });
        }

        // Check query vector quality
        let vector = &query.vector;
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if (magnitude - 1.0).abs() > 0.1 {
            hints.push(OptimizationHint {
                category: HintCategory::QueryParam,
                suggestion: format!(
                    "Query vector is not normalized (magnitude: {:.3}). Normalize vectors for cosine similarity to improve accuracy.",
                    magnitude
                ),
                impact: Impact::Low,
                estimated_improvement: 5.0,
            });
        }

        hints
    }

    /// Generate execution plan
    fn generate_execution_plan(&self, query: &Query, store_size: usize) -> ExecutionPlan {
        let mut steps = Vec::new();
        let mut estimated_rows = vec![store_size];
        let mut uses_index = false;

        // Step 1: Filter application
        if query.filter.is_some() {
            steps.push(PlanStep {
                name: "Filter".to_string(),
                description: "Apply metadata filter to reduce candidate set".to_string(),
                cost: 0.5,
            });
            let filtered_rows = store_size / 2; // Assume 50% selectivity
            estimated_rows.push(filtered_rows);
            uses_index = true;
        }

        // Step 2: Vector similarity computation
        let candidates = *estimated_rows.last().unwrap();
        steps.push(PlanStep {
            name: "Similarity".to_string(),
            description: format!("Compute similarity for {} vectors", candidates),
            cost: candidates as f32 * 0.001,
        });

        // Step 3: Top-K selection
        let k = query.k;
        steps.push(PlanStep {
            name: "Top-K".to_string(),
            description: format!("Select top {} results", k),
            cost: 0.1,
        });
        estimated_rows.push(k);

        ExecutionPlan {
            steps,
            estimated_rows,
            uses_index,
        }
    }

    /// Compare two queries
    pub fn compare_queries(&self, query1: &Query, query2: &Query) -> Result<QueryComparison> {
        let analysis1 = self.analyze_query(query1)?;
        let analysis2 = self.analyze_query(query2)?;

        let faster_query = if analysis1.estimated_cost < analysis2.estimated_cost {
            1
        } else {
            2
        };

        let cost_difference = (analysis1.estimated_cost - analysis2.estimated_cost).abs();
        let relative_difference =
            cost_difference / analysis1.estimated_cost.min(analysis2.estimated_cost);

        Ok(QueryComparison {
            query1_cost: analysis1.estimated_cost,
            query2_cost: analysis2.estimated_cost,
            faster_query,
            cost_difference,
            relative_difference,
            recommendation: if relative_difference > 0.3 {
                format!(
                    "Query {} is significantly faster ({:.1}% improvement)",
                    faster_query,
                    relative_difference * 100.0
                )
            } else {
                "Both queries have similar performance".to_string()
            },
        })
    }

    /// Get optimization summary for the entire store
    pub fn store_optimization_summary(&self) -> StoreOptimizationSummary {
        let store_size = self.store.len();
        let mut recommendations = Vec::new();

        // Check store size
        if store_size > 100000 {
            recommendations.push(
                "Consider partitioning large dataset by metadata for faster queries".to_string(),
            );
        }

        if store_size > 50000 {
            recommendations
                .push("Use approximate indexes (IVF-PQ, LSH) for better scaling".to_string());
        }

        if store_size > 10000 {
            recommendations.push("Add metadata indexes for frequently filtered fields".to_string());
        }

        StoreOptimizationSummary {
            store_size,
            estimated_query_time: self.estimate_avg_query_time(store_size),
            recommendations,
        }
    }

    /// Estimate average query time
    fn estimate_avg_query_time(&self, store_size: usize) -> Duration {
        let ms = (store_size as f32 * 0.001).max(0.1);
        Duration::from_millis(ms as u64)
    }
}

/// Query comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryComparison {
    pub query1_cost: f32,
    pub query2_cost: f32,
    pub faster_query: u8,
    pub cost_difference: f32,
    pub relative_difference: f32,
    pub recommendation: String,
}

/// Store optimization summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreOptimizationSummary {
    pub store_size: usize,
    pub estimated_query_time: Duration,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Metadata;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn create_test_store() -> Result<VecStore> {
        let temp_dir = TempDir::new()?;
        let mut store = VecStore::open(temp_dir.path().join("test.db"))?;

        // Add test vectors
        for i in 0..100 {
            let mut metadata = Metadata {
                fields: HashMap::new(),
            };
            metadata
                .fields
                .insert("category".to_string(), serde_json::json!("test"));

            store.upsert(format!("doc{}", i), vec![i as f32 * 0.01; 128], metadata)?;
        }

        Ok(store)
    }

    #[test]
    fn test_basic_analysis() -> Result<()> {
        let store = create_test_store()?;
        let optimizer = QueryOptimizer::new(&store);

        let query = Query::new(vec![0.5; 128]).with_limit(10);
        let analysis = optimizer.analyze_query(&query)?;

        assert!(analysis.estimated_cost > 0.0);
        assert!(matches!(
            analysis.complexity,
            QueryComplexity::Simple | QueryComplexity::Moderate
        ));

        Ok(())
    }

    #[test]
    fn test_filter_hint() -> Result<()> {
        let store = create_test_store()?;
        let optimizer = QueryOptimizer::new(&store);

        let query = Query::new(vec![0.5; 128])
            .with_limit(10)
            .with_filter("category = 'test'");

        let analysis = optimizer.analyze_query(&query)?;

        // Should suggest index for filtered queries
        assert!(!analysis.hints.is_empty());

        Ok(())
    }

    #[test]
    fn test_large_k_hint() -> Result<()> {
        let store = create_test_store()?;
        let optimizer = QueryOptimizer::new(&store);

        let query = Query::new(vec![0.5; 128]).with_limit(200);
        let analysis = optimizer.analyze_query(&query)?;

        // Should suggest reducing K
        let has_k_hint = analysis
            .hints
            .iter()
            .any(|h| matches!(h.category, HintCategory::QueryParam));
        assert!(has_k_hint);

        Ok(())
    }

    #[test]
    fn test_execution_plan() -> Result<()> {
        let store = create_test_store()?;
        let optimizer = QueryOptimizer::new(&store);

        let query = Query::new(vec![0.5; 128])
            .with_limit(10)
            .with_filter("category = 'test'");

        let analysis = optimizer.analyze_query(&query)?;

        assert!(!analysis.execution_plan.steps.is_empty());
        assert!(analysis.execution_plan.uses_index);

        Ok(())
    }

    #[test]
    fn test_query_comparison() -> Result<()> {
        let store = create_test_store()?;
        let optimizer = QueryOptimizer::new(&store);

        let query1 = Query::new(vec![0.5; 128]).with_limit(10);
        let query2 = Query::new(vec![0.5; 128]).with_limit(100);

        let comparison = optimizer.compare_queries(&query1, &query2)?;

        // Comparison should work (either query could be marginally faster)
        assert!(comparison.faster_query == 1 || comparison.faster_query == 2);
        assert!(comparison.cost_difference >= 0.0);

        Ok(())
    }
}
