//! Vector representations and operations
//!
//! This module provides flexible vector types and scoring algorithms:
//! - Dense vectors (traditional embeddings)
//! - Sparse vectors (keyword/term vectors)
//! - Hybrid vectors (dense + sparse)
//! - BM25 scoring for keyword search
//! - Hybrid search fusion strategies
//! - Vector arithmetic operations (add, subtract, normalize, etc.)
//! - K-means clustering

pub mod bm25;
pub mod hybrid_search;
pub mod ops;
pub mod vector_types;

pub use bm25::{
    bm25_score, bm25_score_simple, bm25f_score, parse_field_weight, parse_field_weights,
    BM25Config, BM25Stats, FieldWeights,
};
pub use hybrid_search::{
    apply_autocut, explain_hybrid_score, hybrid_search_score, normalize_scores,
    normalize_scores_dbsf, normalize_scores_zscore, FusionStrategy, HybridQuery,
    HybridSearchConfig, ScoreContributions, ScoreExplanation,
};
pub use ops::{KMeans, VectorOps};
pub use vector_types::Vector;
