//! Error types for vecstore
//!
//! This module provides a comprehensive error type system for all vecstore operations.
//! Instead of using generic `anyhow::Result`, we use strongly-typed errors that provide
//! better debugging information and allow users to handle specific error cases.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for vecstore operations
pub type Result<T> = std::result::Result<T, VecStoreError>;

/// Main error type for all vecstore operations
#[derive(Error, Debug)]
pub enum VecStoreError {
    /// I/O errors (file operations, network, etc.)
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid vector dimension
    #[error("Invalid vector dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Vector not found
    #[error("Vector with id '{id}' not found")]
    VectorNotFound { id: String },

    /// Invalid filter expression
    #[error("Invalid filter expression: {0}")]
    InvalidFilter(String),

    /// Filter parsing error
    #[error("Filter parse error at position {position}: {message}")]
    FilterParse { position: usize, message: String },

    /// Index not initialized
    #[error("Index not initialized or corrupted")]
    IndexNotInitialized,

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Database corruption
    #[error("Database corruption detected at {path:?}: {reason}")]
    Corruption { path: PathBuf, reason: String },

    /// Snapshot errors
    #[error("Snapshot error: {0}")]
    Snapshot(String),

    /// Snapshot not found
    #[error("Snapshot '{name}' not found")]
    SnapshotNotFound { name: String },

    /// HNSW index errors
    #[error("HNSW index error: {0}")]
    HnswError(String),

    /// Product Quantization errors
    #[error("Product Quantization error: {0}")]
    PqError(String),

    /// Quantizer not trained
    #[error("Quantizer not trained - call train() first")]
    QuantizerNotTrained,

    /// Insufficient training data
    #[error("Insufficient training data: need at least {required}, got {actual}")]
    InsufficientTrainingData { required: usize, actual: usize },

    /// Hybrid search error
    #[error("Hybrid search error: {0}")]
    HybridSearch(String),

    /// Text not indexed for hybrid search
    #[error("Text not indexed for id '{id}' - call index_text() first")]
    TextNotIndexed { id: String },

    /// Empty query
    #[error("Query cannot be empty")]
    EmptyQuery,

    /// Invalid parameter
    #[error("Invalid parameter '{param}': {reason}")]
    InvalidParameter { param: String, reason: String },

    /// Memory limit exceeded
    #[error("Memory limit exceeded: {current} bytes (limit: {limit} bytes)")]
    MemoryLimitExceeded { current: usize, limit: usize },

    /// Concurrent access error
    #[error("Concurrent access error: {0}")]
    ConcurrentAccess(String),

    /// Lock error (poisoned lock)
    #[error("Lock error: {0}")]
    LockError(String),

    /// Embedding errors
    #[cfg(feature = "embeddings")]
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// ONNX Runtime errors
    #[cfg(feature = "embeddings")]
    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(String),

    /// Tokenization errors
    #[cfg(feature = "embeddings")]
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// Python binding errors
    #[cfg(feature = "python")]
    #[error("Python binding error: {0}")]
    Python(String),

    /// WASM specific errors
    #[cfg(feature = "wasm")]
    #[error("WASM error: {0}")]
    Wasm(String),

    /// Feature not enabled
    #[error("Feature '{feature}' not enabled - compile with --features {feature}")]
    FeatureNotEnabled { feature: String },

    /// Other errors
    #[error("Error: {0}")]
    Other(String),
}

// Implement conversions from various error types

impl From<bincode::Error> for VecStoreError {
    fn from(err: bincode::Error) -> Self {
        VecStoreError::Serialization(err.to_string())
    }
}

impl From<serde_json::Error> for VecStoreError {
    fn from(err: serde_json::Error) -> Self {
        VecStoreError::Serialization(err.to_string())
    }
}

#[cfg(feature = "embeddings")]
impl From<ort::OrtError> for VecStoreError {
    fn from(err: ort::OrtError) -> Self {
        VecStoreError::OnnxRuntime(err.to_string())
    }
}

#[cfg(feature = "embeddings")]
impl From<tokenizers::Error> for VecStoreError {
    fn from(err: tokenizers::Error) -> Self {
        VecStoreError::Tokenization(err.to_string())
    }
}

impl<T> From<std::sync::PoisonError<T>> for VecStoreError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        VecStoreError::LockError(err.to_string())
    }
}

// Helper functions for creating common errors

impl VecStoreError {
    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        VecStoreError::DimensionMismatch { expected, actual }
    }

    /// Create a vector not found error
    pub fn vector_not_found(id: impl Into<String>) -> Self {
        VecStoreError::VectorNotFound { id: id.into() }
    }

    /// Create an invalid filter error
    pub fn invalid_filter(msg: impl Into<String>) -> Self {
        VecStoreError::InvalidFilter(msg.into())
    }

    /// Create a filter parse error
    pub fn filter_parse(position: usize, message: impl Into<String>) -> Self {
        VecStoreError::FilterParse {
            position,
            message: message.into(),
        }
    }

    /// Create an invalid config error
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        VecStoreError::InvalidConfig(msg.into())
    }

    /// Create a corruption error
    pub fn corruption(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        VecStoreError::Corruption {
            path: path.into(),
            reason: reason.into(),
        }
    }

    /// Create a snapshot error
    pub fn snapshot(msg: impl Into<String>) -> Self {
        VecStoreError::Snapshot(msg.into())
    }

    /// Create a snapshot not found error
    pub fn snapshot_not_found(name: impl Into<String>) -> Self {
        VecStoreError::SnapshotNotFound { name: name.into() }
    }

    /// Create an HNSW error
    pub fn hnsw_error(msg: impl Into<String>) -> Self {
        VecStoreError::HnswError(msg.into())
    }

    /// Create a PQ error
    pub fn pq_error(msg: impl Into<String>) -> Self {
        VecStoreError::PqError(msg.into())
    }

    /// Create insufficient training data error
    pub fn insufficient_training_data(required: usize, actual: usize) -> Self {
        VecStoreError::InsufficientTrainingData { required, actual }
    }

    /// Create a hybrid search error
    pub fn hybrid_search(msg: impl Into<String>) -> Self {
        VecStoreError::HybridSearch(msg.into())
    }

    /// Create a text not indexed error
    pub fn text_not_indexed(id: impl Into<String>) -> Self {
        VecStoreError::TextNotIndexed { id: id.into() }
    }

    /// Create an invalid parameter error
    pub fn invalid_parameter(param: impl Into<String>, reason: impl Into<String>) -> Self {
        VecStoreError::InvalidParameter {
            param: param.into(),
            reason: reason.into(),
        }
    }

    /// Create a memory limit exceeded error
    pub fn memory_limit_exceeded(current: usize, limit: usize) -> Self {
        VecStoreError::MemoryLimitExceeded { current, limit }
    }

    /// Create a concurrent access error
    pub fn concurrent_access(msg: impl Into<String>) -> Self {
        VecStoreError::ConcurrentAccess(msg.into())
    }

    /// Create a feature not enabled error
    pub fn feature_not_enabled(feature: impl Into<String>) -> Self {
        VecStoreError::FeatureNotEnabled {
            feature: feature.into(),
        }
    }

    /// Create a generic error
    pub fn other(msg: impl Into<String>) -> Self {
        VecStoreError::Other(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch() {
        let err = VecStoreError::dimension_mismatch(128, 256);
        assert_eq!(
            err.to_string(),
            "Invalid vector dimension: expected 128, got 256"
        );
    }

    #[test]
    fn test_vector_not_found() {
        let err = VecStoreError::vector_not_found("vec_123");
        assert_eq!(err.to_string(), "Vector with id 'vec_123' not found");
    }

    #[test]
    fn test_filter_parse() {
        let err = VecStoreError::filter_parse(42, "unexpected token");
        assert_eq!(
            err.to_string(),
            "Filter parse error at position 42: unexpected token"
        );
    }

    #[test]
    fn test_memory_limit_exceeded() {
        let err = VecStoreError::memory_limit_exceeded(1000000, 500000);
        assert_eq!(
            err.to_string(),
            "Memory limit exceeded: 1000000 bytes (limit: 500000 bytes)"
        );
    }

    #[test]
    fn test_feature_not_enabled() {
        let err = VecStoreError::feature_not_enabled("embeddings");
        assert_eq!(
            err.to_string(),
            "Feature 'embeddings' not enabled - compile with --features embeddings"
        );
    }
}
