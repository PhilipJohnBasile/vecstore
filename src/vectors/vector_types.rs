//! Vector types supporting dense, sparse, and hybrid representations
//!
//! This module provides flexible vector representations for VecStore:
//! - Dense: Traditional embedding vectors (Vec<f32>)
//! - Sparse: Keyword/term vectors stored as (index, weight) pairs
//! - Hybrid: Combination of dense embeddings + sparse keywords
//!
//! Sparse vectors are ideal for:
//! - BM25-style keyword search
//! - Tag-based search
//! - Memory-efficient storage when most values are zero
//!
//! Hybrid vectors enable powerful fusion search strategies.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Vector representation supporting dense, sparse, or hybrid
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Vector {
    /// Dense vector (traditional embeddings)
    ///
    /// Example: text embedding [0.1, 0.2, 0.3, ...]
    Dense(Vec<f32>),

    /// Sparse vector (keyword/term vectors)
    ///
    /// Stores only non-zero elements as (index, weight) pairs.
    /// Much more memory-efficient when most values are zero.
    ///
    /// Example: Document with terms at indices [5, 127, 943]
    ///          with weights [0.8, 1.2, 0.5] in a 10,000-dim vocabulary
    Sparse {
        /// Total dimensionality of the vector space
        dimension: usize,
        /// Indices of non-zero elements (must be < dimension)
        indices: Vec<usize>,
        /// Values at those indices
        values: Vec<f32>,
    },

    /// Hybrid vector (dense + sparse)
    ///
    /// Combines semantic similarity (dense) with keyword matching (sparse).
    /// Enables powerful fusion search strategies.
    ///
    /// Example: Document with both:
    ///   - Dense embedding from transformer model
    ///   - Sparse BM25 term weights for keyword search
    Hybrid {
        /// Dense component (e.g., semantic embedding)
        dense: Vec<f32>,
        /// Sparse component indices
        sparse_indices: Vec<usize>,
        /// Sparse component values
        sparse_values: Vec<f32>,
    },
}

impl Vector {
    /// Create a dense vector
    ///
    /// # Example
    /// ```
    /// use vecstore::vectors::Vector;
    ///
    /// let vec = Vector::dense(vec![0.1, 0.2, 0.3]);
    /// assert_eq!(vec.dimension(), 3);
    /// ```
    pub fn dense(values: Vec<f32>) -> Self {
        Vector::Dense(values)
    }

    /// Create a sparse vector from (index, value) pairs
    ///
    /// # Arguments
    /// * `dimension` - Total dimensionality of the vector space
    /// * `indices` - Indices of non-zero elements
    /// * `values` - Values at those indices
    ///
    /// # Errors
    /// Returns error if:
    /// - indices and values have different lengths
    /// - any index >= dimension
    ///
    /// # Example
    /// ```
    /// use vecstore::vectors::Vector;
    ///
    /// // Sparse vector in 1000-dimensional space
    /// // Only 3 non-zero elements at indices [5, 127, 943]
    /// let vec = Vector::sparse(1000, vec![5, 127, 943], vec![0.8, 1.2, 0.5]).unwrap();
    /// assert_eq!(vec.dimension(), 1000);
    /// assert_eq!(vec.sparsity(), 0.997); // 99.7% sparse
    /// ```
    pub fn sparse(dimension: usize, indices: Vec<usize>, values: Vec<f32>) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(anyhow!(
                "Sparse vector indices and values must have same length (got {} indices, {} values)",
                indices.len(),
                values.len()
            ));
        }
        if let Some(&max_idx) = indices.iter().max() {
            if max_idx >= dimension {
                return Err(anyhow!(
                    "Sparse vector index {} out of bounds (dimension: {})",
                    max_idx,
                    dimension
                ));
            }
        }
        Ok(Vector::Sparse {
            dimension,
            indices,
            values,
        })
    }

    /// Create a hybrid vector
    ///
    /// # Arguments
    /// * `dense` - Dense component (e.g., semantic embedding)
    /// * `sparse_indices` - Sparse component indices
    /// * `sparse_values` - Sparse component values
    ///
    /// # Errors
    /// Returns error if sparse_indices and sparse_values have different lengths
    ///
    /// # Example
    /// ```
    /// use vecstore::vectors::Vector;
    ///
    /// // Hybrid: 384-dim dense embedding + sparse keyword weights
    /// let dense_embedding = vec![0.1; 384];
    /// let keyword_indices = vec![10, 25, 100]; // vocabulary indices
    /// let keyword_weights = vec![1.5, 0.8, 2.1]; // BM25 weights
    ///
    /// let vec = Vector::hybrid(dense_embedding, keyword_indices, keyword_weights).unwrap();
    /// assert!(vec.dense_part().is_some());
    /// assert!(vec.sparse_part().is_some());
    /// ```
    pub fn hybrid(
        dense: Vec<f32>,
        sparse_indices: Vec<usize>,
        sparse_values: Vec<f32>,
    ) -> Result<Self> {
        if sparse_indices.len() != sparse_values.len() {
            return Err(anyhow!(
                "Sparse component indices and values must have same length (got {} indices, {} values)",
                sparse_indices.len(),
                sparse_values.len()
            ));
        }
        Ok(Vector::Hybrid {
            dense,
            sparse_indices,
            sparse_values,
        })
    }

    /// Get the total dimension of the vector
    pub fn dimension(&self) -> usize {
        match self {
            Vector::Dense(v) => v.len(),
            Vector::Sparse { dimension, .. } => *dimension,
            Vector::Hybrid { dense, .. } => dense.len(),
        }
    }

    /// Get the dense component (if any)
    ///
    /// Returns Some(&[f32]) for Dense and Hybrid vectors, None for Sparse
    pub fn dense_part(&self) -> Option<&[f32]> {
        match self {
            Vector::Dense(v) => Some(v),
            Vector::Hybrid { dense, .. } => Some(dense),
            Vector::Sparse { .. } => None,
        }
    }

    /// Get the sparse component (if any)
    ///
    /// Returns Some((indices, values)) for Sparse and Hybrid vectors, None for Dense
    pub fn sparse_part(&self) -> Option<(&[usize], &[f32])> {
        match self {
            Vector::Sparse {
                indices, values, ..
            } => Some((indices, values)),
            Vector::Hybrid {
                sparse_indices,
                sparse_values,
                ..
            } => Some((sparse_indices, sparse_values)),
            Vector::Dense(_) => None,
        }
    }

    /// Get sparsity ratio (0.0 = fully dense, 1.0 = fully sparse)
    ///
    /// Only meaningful for Sparse and Hybrid vectors.
    /// Dense vectors always return 0.0.
    pub fn sparsity(&self) -> f32 {
        match self {
            Vector::Dense(_) => 0.0,
            Vector::Sparse {
                dimension, indices, ..
            } => {
                if *dimension == 0 {
                    0.0
                } else {
                    1.0 - (indices.len() as f32 / *dimension as f32)
                }
            }
            Vector::Hybrid { sparse_indices, .. } => {
                // For hybrid, report sparsity of sparse component
                // (not well-defined since we don't know sparse dimension)
                if sparse_indices.is_empty() {
                    1.0
                } else {
                    // Just report presence of sparse component
                    0.5
                }
            }
        }
    }

    /// Check if vector is sparse or hybrid (has sparse component)
    pub fn has_sparse_component(&self) -> bool {
        matches!(self, Vector::Sparse { .. } | Vector::Hybrid { .. })
    }

    /// Check if vector is dense or hybrid (has dense component)
    pub fn has_dense_component(&self) -> bool {
        matches!(self, Vector::Dense(_) | Vector::Hybrid { .. })
    }

    /// Convert to dense representation (materialize sparse vectors)
    ///
    /// For sparse vectors, creates a full dense vector with zeros.
    /// For hybrid vectors, returns only the dense component.
    pub fn to_dense(&self) -> Vec<f32> {
        match self {
            Vector::Dense(v) => v.clone(),
            Vector::Sparse {
                dimension,
                indices,
                values,
            } => {
                let mut dense = vec![0.0; *dimension];
                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    dense[idx] = val;
                }
                dense
            }
            Vector::Hybrid { dense, .. } => dense.clone(),
        }
    }

    /// Get number of stored elements (not total dimension)
    ///
    /// For sparse vectors, returns number of non-zero elements.
    /// For dense vectors, returns dimension.
    pub fn storage_size(&self) -> usize {
        match self {
            Vector::Dense(v) => v.len(),
            Vector::Sparse { indices, .. } => indices.len(),
            Vector::Hybrid {
                dense,
                sparse_indices,
                ..
            } => dense.len() + sparse_indices.len(),
        }
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            Vector::Dense(v) => v.len() * std::mem::size_of::<f32>(),
            Vector::Sparse {
                indices, values, ..
            } => {
                indices.len() * std::mem::size_of::<usize>()
                    + values.len() * std::mem::size_of::<f32>()
                    + std::mem::size_of::<usize>() // dimension field
            }
            Vector::Hybrid {
                dense,
                sparse_indices,
                sparse_values,
            } => {
                dense.len() * std::mem::size_of::<f32>()
                    + sparse_indices.len() * std::mem::size_of::<usize>()
                    + sparse_values.len() * std::mem::size_of::<f32>()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_vector() {
        let vec = Vector::dense(vec![0.1, 0.2, 0.3]);
        assert_eq!(vec.dimension(), 3);
        assert_eq!(vec.dense_part(), Some(&[0.1, 0.2, 0.3][..]));
        assert_eq!(vec.sparse_part(), None);
        assert_eq!(vec.sparsity(), 0.0);
        assert!(vec.has_dense_component());
        assert!(!vec.has_sparse_component());
    }

    #[test]
    fn test_sparse_vector() {
        let vec = Vector::sparse(1000, vec![5, 127, 943], vec![0.8, 1.2, 0.5]).unwrap();
        assert_eq!(vec.dimension(), 1000);
        assert_eq!(vec.dense_part(), None);
        assert_eq!(
            vec.sparse_part(),
            Some((&[5, 127, 943][..], &[0.8, 1.2, 0.5][..]))
        );
        assert!((vec.sparsity() - 0.997).abs() < 0.001);
        assert!(!vec.has_dense_component());
        assert!(vec.has_sparse_component());
    }

    #[test]
    fn test_sparse_vector_invalid_length_mismatch() {
        let result = Vector::sparse(1000, vec![1, 2], vec![0.5]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("same length"));
    }

    #[test]
    fn test_sparse_vector_invalid_index_out_of_bounds() {
        let result = Vector::sparse(100, vec![5, 127], vec![0.8, 1.2]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_hybrid_vector() {
        let vec = Vector::hybrid(vec![0.1, 0.2, 0.3], vec![10, 25], vec![1.5, 0.8]).unwrap();

        assert_eq!(vec.dimension(), 3);
        assert_eq!(vec.dense_part(), Some(&[0.1, 0.2, 0.3][..]));
        assert_eq!(vec.sparse_part(), Some((&[10, 25][..], &[1.5, 0.8][..])));
        assert!(vec.has_dense_component());
        assert!(vec.has_sparse_component());
    }

    #[test]
    fn test_hybrid_vector_invalid() {
        let result = Vector::hybrid(vec![0.1, 0.2], vec![1, 2], vec![0.5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_dense_from_sparse() {
        let vec = Vector::sparse(10, vec![2, 5, 7], vec![1.0, 2.0, 3.0]).unwrap();
        let dense = vec.to_dense();

        assert_eq!(dense.len(), 10);
        assert_eq!(dense[2], 1.0);
        assert_eq!(dense[5], 2.0);
        assert_eq!(dense[7], 3.0);
        assert_eq!(dense[0], 0.0);
        assert_eq!(dense[9], 0.0);
    }

    #[test]
    fn test_storage_size() {
        let dense = Vector::dense(vec![0.1; 1000]);
        assert_eq!(dense.storage_size(), 1000);

        let sparse = Vector::sparse(1000, vec![1, 2, 3], vec![0.1, 0.2, 0.3]).unwrap();
        assert_eq!(sparse.storage_size(), 3);

        let hybrid = Vector::hybrid(vec![0.1; 100], vec![1, 2], vec![0.5, 0.6]).unwrap();
        assert_eq!(hybrid.storage_size(), 102);
    }

    #[test]
    fn test_memory_usage() {
        let dense = Vector::dense(vec![0.1; 1000]);
        assert_eq!(dense.memory_usage(), 1000 * 4); // 1000 * sizeof(f32)

        let sparse = Vector::sparse(1000, vec![1, 2, 3], vec![0.1, 0.2, 0.3]).unwrap();
        // 3 * sizeof(usize) + 3 * sizeof(f32) + sizeof(usize)
        let expected = 3 * 8 + 3 * 4 + 8; // assuming 64-bit platform
        assert_eq!(sparse.memory_usage(), expected);
    }

    #[test]
    fn test_serialization() {
        let vec = Vector::sparse(100, vec![5, 10], vec![1.0, 2.0]).unwrap();
        let json = serde_json::to_string(&vec).unwrap();
        let deserialized: Vector = serde_json::from_str(&json).unwrap();
        assert_eq!(vec, deserialized);
    }
}
