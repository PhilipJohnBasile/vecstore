// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

use anyhow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Id = String;

// ============================================================================
// QUANTIZATION CONFIGURATION
// ============================================================================

/// Quantization configuration for memory reduction
///
/// Quantization compresses vectors to reduce memory usage with minimal accuracy loss.
/// Following Qdrant's approach, scalar quantization is recommended as the default
/// for most use cases (4x compression, 99% recall retained).
///
/// ## Compression Comparison
///
/// | Method | Compression | Recall | Speed | Best For |
/// |--------|-------------|--------|-------|----------|
/// | None | 1x | 100% | 1x | Small datasets |
/// | Scalar8 | 4x | 99% | 2x | General use (recommended) |
/// | Scalar4 | 8x | 96% | 3x | Large datasets |
/// | Binary | 32x | 90% | 8x | High-dim embeddings (1024+) |
/// | ProductQuantization | 32x | 70-90% | 0.5x | Maximum compression |
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantizationConfig {
    /// No quantization - full float32 vectors
    None,

    /// 8-bit scalar quantization (INT8)
    /// 4x memory reduction, ~99% recall retained
    /// Best for: General use, recommended default for memory-constrained environments
    Scalar8 {
        /// Quantile to exclude outliers (0.99 = exclude 1% extreme values)
        #[serde(default = "default_quantile")]
        quantile: f32,
    },

    /// 4-bit scalar quantization
    /// 8x memory reduction, ~96% recall retained
    /// Best for: Very large datasets where memory is critical
    Scalar4 {
        /// Quantile to exclude outliers
        #[serde(default = "default_quantile")]
        quantile: f32,
    },

    /// Binary quantization (1-bit)
    /// 32x memory reduction, ~90% recall (higher with rescoring)
    /// Best for: High-dimensional embeddings (1024+), OpenAI/Cohere embeddings
    Binary {
        /// Use mean as threshold (true) or zero (false, sign-based)
        #[serde(default)]
        use_mean_threshold: bool,
    },

    /// Product Quantization
    /// Up to 32x memory reduction, variable recall (70-90%)
    /// Best for: Maximum compression when accuracy can be traded
    Product {
        /// Number of subvectors (M) - dimension must be divisible by this
        /// Common values: 8, 16, 32
        #[serde(default = "default_pq_subvectors")]
        num_subvectors: usize,

        /// Number of centroids per subspace (K)
        /// 256 = 1 byte per subvector, 65536 = 2 bytes
        #[serde(default = "default_pq_centroids")]
        num_centroids: usize,

        /// Number of training iterations for k-means
        #[serde(default = "default_pq_iterations")]
        training_iterations: usize,
    },
}

fn default_quantile() -> f32 {
    0.99
}

fn default_pq_subvectors() -> usize {
    16
}

fn default_pq_centroids() -> usize {
    256
}

fn default_pq_iterations() -> usize {
    20
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        QuantizationConfig::None
    }
}

/// Statistics about quantized vector storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationStats {
    /// Quantization method name
    pub method: String,
    /// Number of vectors that have been quantized
    pub vectors_quantized: usize,
    /// Original size in bytes (full precision)
    pub original_size_bytes: usize,
    /// Compressed size in bytes
    pub compressed_size_bytes: usize,
    /// Actual compression ratio achieved
    pub compression_ratio: f32,
}

impl QuantizationConfig {
    /// Create scalar 8-bit quantization with default settings
    pub fn scalar8() -> Self {
        QuantizationConfig::Scalar8 {
            quantile: default_quantile(),
        }
    }

    /// Create scalar 4-bit quantization with default settings
    pub fn scalar4() -> Self {
        QuantizationConfig::Scalar4 {
            quantile: default_quantile(),
        }
    }

    /// Create binary quantization with mean threshold
    pub fn binary() -> Self {
        QuantizationConfig::Binary {
            use_mean_threshold: true,
        }
    }

    /// Create product quantization with default settings
    pub fn product(num_subvectors: usize) -> Self {
        QuantizationConfig::Product {
            num_subvectors,
            num_centroids: default_pq_centroids(),
            training_iterations: default_pq_iterations(),
        }
    }

    /// Get the expected compression ratio
    pub fn compression_ratio(&self) -> f32 {
        match self {
            QuantizationConfig::None => 1.0,
            QuantizationConfig::Scalar8 { .. } => 4.0,
            QuantizationConfig::Scalar4 { .. } => 8.0,
            QuantizationConfig::Binary { .. } => 32.0,
            QuantizationConfig::Product { num_subvectors, .. } => {
                // Assuming 128-dim vectors as reference
                // Original: 128 * 4 bytes = 512 bytes
                // Compressed: num_subvectors * 1 byte
                512.0 / *num_subvectors as f32
            }
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            QuantizationConfig::None => "No quantization (full precision)",
            QuantizationConfig::Scalar8 { .. } => "8-bit scalar quantization (4x compression, 99% recall)",
            QuantizationConfig::Scalar4 { .. } => "4-bit scalar quantization (8x compression, 96% recall)",
            QuantizationConfig::Binary { .. } => "Binary quantization (32x compression, 90% recall)",
            QuantizationConfig::Product { .. } => "Product quantization (variable compression, 70-90% recall)",
        }
    }

    /// Check if quantization requires training
    pub fn requires_training(&self) -> bool {
        matches!(
            self,
            QuantizationConfig::Scalar8 { .. }
                | QuantizationConfig::Scalar4 { .. }
                | QuantizationConfig::Binary { use_mean_threshold: true }
                | QuantizationConfig::Product { .. }
        )
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum Distance {
    /// Cosine similarity (default) - measures angle between vectors
    /// Range: [-1, 1], higher is more similar
    /// Best for: Text embeddings, normalized vectors
    #[default]
    Cosine,

    /// Euclidean distance (L2) - measures straight-line distance
    /// Range: [0, ∞), lower is more similar
    /// Best for: Spatial data, unnormalized vectors
    Euclidean,

    /// Dot product - measures alignment and magnitude
    /// Range: (-∞, ∞), higher is more similar
    /// Best for: When magnitude matters, recommendation systems
    DotProduct,

    /// Manhattan distance (L1) - measures city-block distance
    /// Range: [0, ∞), lower is more similar
    /// Best for: Spatial data, robust to outliers, grid-based distances
    Manhattan,

    /// Hamming distance - counts differing elements
    /// Range: [0, n], lower is more similar (n = vector dimension)
    /// Best for: Binary vectors, categorical data, error detection
    /// Note: Converts f32 vectors to binary at threshold 0.5
    Hamming,

    /// Jaccard distance - measures set dissimilarity
    /// Range: [0, 1], lower is more similar
    /// Best for: Sparse vectors, tag vectors, one-hot encoding, set similarity
    Jaccard,

    /// Chebyshev distance (L∞) - maximum absolute difference
    /// Range: [0, ∞), lower is more similar
    /// Best for: Grid-based movement, game AI, chessboard distance
    Chebyshev,

    /// Canberra distance - weighted Manhattan distance
    /// Range: [0, ∞), lower is more similar
    /// Best for: Data with large outliers, non-negative data, ranked data
    Canberra,

    /// Bray-Curtis dissimilarity - ecological distance
    /// Range: [0, 1], lower is more similar
    /// Best for: Ecological data, compositional data, species abundance
    BrayCurtis,
}

impl Distance {
    /// Convert distance metric name to enum
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(Distance::Cosine),
            "euclidean" | "l2" => Ok(Distance::Euclidean),
            "dotproduct" | "dot" => Ok(Distance::DotProduct),
            "manhattan" | "l1" => Ok(Distance::Manhattan),
            "hamming" => Ok(Distance::Hamming),
            "jaccard" => Ok(Distance::Jaccard),
            "chebyshev" | "linf" | "l_inf" => Ok(Distance::Chebyshev),
            "canberra" => Ok(Distance::Canberra),
            "braycurtis" | "bray-curtis" | "bray_curtis" => Ok(Distance::BrayCurtis),
            _ => Err(anyhow::anyhow!("Unknown distance metric: {}", s)),
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Distance::Cosine => "Cosine",
            Distance::Euclidean => "Euclidean",
            Distance::DotProduct => "DotProduct",
            Distance::Manhattan => "Manhattan",
            Distance::Hamming => "Hamming",
            Distance::Jaccard => "Jaccard",
            Distance::Chebyshev => "Chebyshev",
            Distance::Canberra => "Canberra",
            Distance::BrayCurtis => "BrayCurtis",
        }
    }

    /// Get a brief description of the metric
    pub fn description(&self) -> &'static str {
        match self {
            Distance::Cosine => "Measures angle between vectors (text embeddings)",
            Distance::Euclidean => "Straight-line distance (spatial data)",
            Distance::DotProduct => "Alignment and magnitude (recommendations)",
            Distance::Manhattan => "City-block distance (robust to outliers)",
            Distance::Hamming => "Count of differing elements (binary data)",
            Distance::Jaccard => "Set dissimilarity (sparse vectors, tags, sets)",
            Distance::Chebyshev => "Maximum difference across dimensions (grid-based)",
            Distance::Canberra => "Weighted Manhattan for data with outliers",
            Distance::BrayCurtis => "Ecological dissimilarity (compositional data)",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)] // Major Issue #7 fix: add serialization
pub struct Config {
    /// Distance metric to use for similarity search
    pub distance: Distance,

    /// HNSW parameter: number of connections per layer (default: 16)
    /// Higher values improve recall but increase memory and build time.
    /// Typical range: 8-64
    pub hnsw_m: usize,

    /// HNSW parameter: size of dynamic candidate list during construction (default: 200)
    /// Higher values improve index quality but increase build time.
    /// Typical range: 100-500
    pub hnsw_ef_construction: usize,

    /// Maximum number of elements the HNSW index can hold (default: 100,000)
    /// This is pre-allocated, so set appropriately for your use case.
    /// Inserting beyond this limit will return an error.
    #[serde(default = "default_hnsw_max_elements")]
    pub hnsw_max_elements: usize,

    /// Enable write-ahead logging for crash recovery (default: false)
    /// When enabled, all mutations are logged before being applied,
    /// allowing recovery after crashes.
    #[serde(default)]
    pub wal_enabled: bool,

    /// Quantization configuration for memory reduction (default: None)
    ///
    /// Enable quantization to reduce memory usage at the cost of some accuracy:
    /// - `Scalar8`: 4x compression, 99% recall (recommended)
    /// - `Scalar4`: 8x compression, 96% recall
    /// - `Binary`: 32x compression, 90% recall
    /// - `Product`: variable compression, 70-90% recall
    #[serde(default)]
    pub quantization: QuantizationConfig,
}

fn default_hnsw_max_elements() -> usize {
    100_000
}

impl Default for Config {
    fn default() -> Self {
        Self {
            distance: Distance::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_max_elements: 100_000,
            wal_enabled: false,
            quantization: QuantizationConfig::None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Metadata {
    pub fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: Id,
    pub vector: Vec<f32>,
    pub metadata: Metadata,
    pub created_at: i64, // unix seconds

    /// Soft delete flag - if true, record is marked for deletion
    /// but not yet removed (allows undo and deferred cleanup)
    #[serde(default)]
    pub deleted: bool,

    /// Timestamp when record was soft-deleted (unix seconds)
    #[serde(default)]
    pub deleted_at: Option<i64>,

    /// Time-to-live: timestamp when record expires (unix seconds)
    /// None means no expiration
    #[serde(default)]
    pub expires_at: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct Query {
    pub vector: Vec<f32>,
    pub k: usize,
    pub filter: Option<FilterExpr>,
}

impl Query {
    /// Create a new query with the given vector
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            k: 10, // Default k
            filter: None,
        }
    }

    /// Set the number of results to return
    pub fn with_limit(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Add a filter expression
    pub fn with_filter_expr(mut self, filter: FilterExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Add a filter from a string (parsed)
    ///
    /// Note: This method ignores parse errors for builder pattern convenience.
    /// Use `try_with_filter()` if you need error handling.
    pub fn with_filter(self, filter_str: &str) -> Self {
        if let Ok(filter) = crate::store::parse_filter(filter_str) {
            self.with_filter_expr(filter)
        } else {
            // Log warning but don't fail (Major Issue #17 partial fix)
            #[cfg(feature = "tracing")]
            tracing::warn!(
                "Failed to parse filter: '{}'. Filter will be ignored.",
                filter_str
            );
            self // Ignore invalid filters in builder pattern
        }
    }

    /// Add a filter from a string, returning an error if parsing fails
    ///
    /// This is the strict version of `with_filter()` that returns errors instead
    /// of silently ignoring them. (Major Issue #17 fix)
    pub fn try_with_filter(self, filter_str: &str) -> Result<Self, anyhow::Error> {
        let filter = crate::store::parse_filter(filter_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse filter '{}': {}", filter_str, e))?;
        Ok(self.with_filter_expr(filter))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FilterOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    Contains,
    In,         // Value is in array
    NotIn,      // Value not in array
    StartsWith, // String starts with prefix (Major Issue #13 fix)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FilterExpr {
    And(Vec<FilterExpr>),
    Or(Vec<FilterExpr>),
    Not(Box<FilterExpr>),
    Cmp {
        field: String,
        op: FilterOp,
        value: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neighbor {
    pub id: Id,
    pub score: f32,
    pub metadata: Metadata,
}

/// Detailed explanation of why a result was returned and how it was scored
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainedNeighbor {
    pub id: Id,
    pub score: f32,
    pub metadata: Metadata,
    pub explanation: QueryExplanation,
}

/// Detailed breakdown of query execution for a specific result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExplanation {
    /// Raw similarity score before any adjustments
    pub raw_score: f32,

    /// Distance metric used
    pub distance_metric: String,

    /// Whether this result passed all filters (if any)
    pub filter_passed: bool,

    /// Filter evaluation details (if filters were applied)
    pub filter_details: Option<FilterEvaluation>,

    /// Graph traversal details (for HNSW)
    pub graph_stats: Option<GraphTraversalStats>,

    /// Ranking information
    pub rank: usize,

    /// Why this result was included
    pub explanation_text: String,
}

/// Details about filter evaluation for this result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterEvaluation {
    /// Filter expression that was applied
    pub filter_expr: String,

    /// Which filter conditions matched
    pub matched_conditions: Vec<String>,

    /// Which filter conditions failed (if any)
    pub failed_conditions: Vec<String>,

    /// Overall result
    pub passed: bool,
}

/// Statistics about HNSW graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTraversalStats {
    /// Number of distance calculations performed
    pub distance_calculations: usize,

    /// Number of graph nodes visited
    pub nodes_visited: usize,

    /// Layer at which this result was found
    pub found_at_layer: Option<usize>,

    /// Number of hops from entry point
    pub hops_from_entry: Option<usize>,
}

/// Batch operation types for mixed operation batches
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum BatchOperation {
    /// Upsert a vector
    Upsert {
        id: Id,
        vector: Vec<f32>,
        metadata: Metadata,
    },
    /// Delete a vector
    Delete { id: Id },
    /// Soft delete a vector
    SoftDelete { id: Id },
    /// Restore a soft-deleted vector
    Restore { id: Id },
    /// Update metadata only
    UpdateMetadata { id: Id, metadata: Metadata },
}

/// Result of a batch operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Number of successful operations
    pub succeeded: usize,

    /// Number of failed operations
    pub failed: usize,

    /// Detailed errors (if any)
    pub errors: Vec<BatchError>,

    /// Duration in milliseconds
    pub duration_ms: f64,
}

/// Error for a specific batch operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchError {
    /// Index of the operation that failed
    pub index: usize,

    /// Operation that failed
    pub operation: String,

    /// Error message
    pub error: String,
}

/// Query validation and cost estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEstimate {
    /// Whether the query is valid
    pub valid: bool,

    /// Validation errors (if any)
    pub errors: Vec<String>,

    /// Estimated computational cost (0.0 - 1.0, where 1.0 is most expensive)
    pub cost_estimate: f32,

    /// Estimated number of distance calculations
    pub estimated_distance_calculations: usize,

    /// Estimated number of nodes to visit
    pub estimated_nodes_visited: usize,

    /// Whether over-fetching will occur (for filtered queries)
    pub will_overfetch: bool,

    /// Recommendations for optimization
    pub recommendations: Vec<String>,

    /// Estimated query duration in milliseconds
    pub estimated_duration_ms: f32,
}

/// Auto-compaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Minimum number of deleted records before triggering compaction
    pub min_deleted_records: usize,

    /// Minimum ratio of deleted/total records (0.0 - 1.0) to trigger compaction
    pub min_deleted_ratio: f32,

    /// Whether auto-compaction is enabled
    pub enabled: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            min_deleted_records: 1000,
            min_deleted_ratio: 0.1, // 10% deleted
            enabled: false,
        }
    }
}

/// Result of a compaction operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResult {
    /// Number of records removed
    pub removed_count: usize,

    /// Duration in milliseconds
    pub duration_ms: f64,

    /// Whether compaction was triggered
    pub triggered: bool,

    /// Reason for compaction (or why it wasn't triggered)
    pub reason: String,
}

/// Multi-stage query for advanced RAG patterns (prefetch API)
///
/// Allows chaining multiple query stages for complex retrieval patterns:
/// 1. Initial broad search (high k)
/// 2. Reranking or MMR for diversity
/// 3. Final selection (low k)
///
/// Example: Hybrid search → Cross-encoder rerank → MMR diversity → Top 10
#[derive(Debug, Clone)]
pub struct PrefetchQuery {
    /// Stages of query execution (evaluated in order)
    pub stages: Vec<QueryStage>,
}

/// A single stage in a prefetch query pipeline
#[derive(Debug, Clone)]
pub enum QueryStage {
    /// Vector similarity search stage
    VectorSearch {
        vector: Vec<f32>,
        k: usize,
        filter: Option<FilterExpr>,
    },

    /// Hybrid search stage (vector + BM25)
    HybridSearch {
        vector: Vec<f32>,
        keywords: String,
        k: usize,
        alpha: f32,
        filter: Option<FilterExpr>,
    },

    /// Rerank using cross-encoder or other scoring function
    Rerank {
        /// Number of results to keep after reranking
        k: usize,
        /// Optional: model name for cross-encoder
        model: Option<String>,
    },

    /// Maximal Marginal Relevance for diversity
    MMR {
        /// Number of diverse results to keep
        k: usize,
        /// Trade-off between relevance and diversity (0.0 = all diversity, 1.0 = all relevance)
        lambda: f32,
    },

    /// Filter stage (apply additional filters)
    Filter { expr: FilterExpr },
}

/// HNSW search parameters (can be tuned per-query)
#[derive(Debug, Clone)]
pub struct HNSWSearchParams {
    /// Size of dynamic candidate list during search
    /// Higher = better recall, slower search
    /// Default: 50 (balanced)
    /// Range: 10 (fast) to 500 (high recall)
    pub ef_search: usize,
}

impl Default for HNSWSearchParams {
    fn default() -> Self {
        Self {
            ef_search: 50, // Balanced default
        }
    }
}

impl HNSWSearchParams {
    /// Fast search (ef_search = 20)
    pub fn fast() -> Self {
        Self { ef_search: 20 }
    }

    /// Balanced search (ef_search = 50) - default
    pub fn balanced() -> Self {
        Self::default()
    }

    /// High recall search (ef_search = 100)
    pub fn high_recall() -> Self {
        Self { ef_search: 100 }
    }

    /// Maximum recall search (ef_search = 200)
    pub fn max_recall() -> Self {
        Self { ef_search: 200 }
    }
}

/// Query plan - explains how a query will be executed and its estimated cost
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// Query type
    pub query_type: String,

    /// Execution steps
    pub steps: Vec<QueryStep>,

    /// Estimated total cost (0.0 - 1.0)
    pub estimated_cost: f32,

    /// Estimated duration in milliseconds
    pub estimated_duration_ms: f32,

    /// Recommendations for optimization
    pub recommendations: Vec<String>,

    /// Whether this query plan is optimal
    pub is_optimal: bool,
}

/// A single step in query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStep {
    /// Step number (execution order)
    pub step: usize,

    /// Description of what this step does
    pub description: String,

    /// Estimated cost for this step (0.0 - 1.0)
    pub cost: f32,

    /// Expected input size (number of candidates)
    pub input_size: usize,

    /// Expected output size (number of results)
    pub output_size: usize,
}
