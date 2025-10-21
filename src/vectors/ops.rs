//! Vector arithmetic and operations
//!
//! This module provides common vector operations needed for RAG and ML applications:
//! - Addition, subtraction, scalar multiplication
//! - Centroid calculation
//! - Normalization
//! - K-means clustering

use anyhow::Result;

/// Vector operations
pub struct VectorOps;

impl VectorOps {
    /// Add two vectors element-wise
    pub fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            anyhow::bail!("Vector dimensions must match: {} vs {}", a.len(), b.len());
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    /// Subtract two vectors element-wise
    pub fn subtract(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            anyhow::bail!("Vector dimensions must match");
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x - y).collect())
    }

    /// Multiply vector by scalar
    pub fn scale(v: &[f32], scalar: f32) -> Vec<f32> {
        v.iter().map(|x| x * scalar).collect()
    }

    /// Calculate mean/centroid of multiple vectors
    pub fn centroid(vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            anyhow::bail!("Cannot calculate centroid of empty vector set");
        }

        let dim = vectors[0].len();
        let mut result = vec![0.0; dim];

        for v in vectors {
            if v.len() != dim {
                anyhow::bail!("All vectors must have same dimension");
            }
            for (i, &val) in v.iter().enumerate() {
                result[i] += val;
            }
        }

        let n = vectors.len() as f32;
        for val in &mut result {
            *val /= n;
        }

        Ok(result)
    }

    /// Normalize vector to unit length (L2 norm = 1)
    pub fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }

    /// L2 norm (Euclidean length)
    pub fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Dot product
    pub fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            anyhow::bail!("Vector dimensions must match");
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }

    /// Sparse dot product
    ///
    /// Computes dot product between two sparse vectors represented as (indices, values).
    /// Much more efficient than materializing to dense when vectors are sparse.
    ///
    /// # Arguments
    /// * `a_indices` - Indices of non-zero elements in vector a
    /// * `a_values` - Values at those indices in vector a
    /// * `b_indices` - Indices of non-zero elements in vector b
    /// * `b_values` - Values at those indices in vector b
    ///
    /// # Returns
    /// Dot product of the two sparse vectors
    ///
    /// # Performance
    /// O(n + m) where n and m are the number of non-zero elements.
    /// For vectors with k non-zero elements in d dimensions: O(k) vs O(d) for dense.
    ///
    /// # Example
    /// ```
    /// use vecstore::vectors::VectorOps;
    ///
    /// // Sparse vectors in 1000-dimensional space
    /// // a has non-zero values at indices [5, 100, 500]
    /// // b has non-zero values at indices [5, 200, 500]
    /// let a_indices = vec![5, 100, 500];
    /// let a_values = vec![1.0, 2.0, 3.0];
    /// let b_indices = vec![5, 200, 500];
    /// let b_values = vec![1.5, 2.5, 1.0];
    ///
    /// let dot = VectorOps::sparse_dot(&a_indices, &a_values, &b_indices, &b_values);
    /// // Result: 1.0*1.5 + 3.0*1.0 = 4.5 (indices 5 and 500 match)
    /// assert_eq!(dot, 4.5);
    /// ```
    pub fn sparse_dot(
        a_indices: &[usize],
        a_values: &[f32],
        b_indices: &[usize],
        b_values: &[f32],
    ) -> f32 {
        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        // Two-pointer algorithm: iterate through sorted indices
        while i < a_indices.len() && j < b_indices.len() {
            match a_indices[i].cmp(&b_indices[j]) {
                std::cmp::Ordering::Equal => {
                    // Indices match - multiply and add
                    result += a_values[i] * b_values[j];
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    // a's index is smaller - advance a
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    // b's index is smaller - advance b
                    j += 1;
                }
            }
        }

        result
    }

    /// Sparse vector L2 norm (Euclidean length)
    ///
    /// # Example
    /// ```
    /// use vecstore::vectors::VectorOps;
    ///
    /// let indices = vec![0, 5, 10];
    /// let values = vec![3.0, 4.0, 0.0]; // [3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, ...]
    ///
    /// let norm = VectorOps::sparse_norm(&values);
    /// assert_eq!(norm, 5.0); // sqrt(3^2 + 4^2) = 5
    /// ```
    pub fn sparse_norm(values: &[f32]) -> f32 {
        values.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Cosine similarity for sparse vectors
    ///
    /// # Arguments
    /// * `a_indices`, `a_values` - First sparse vector
    /// * `b_indices`, `b_values` - Second sparse vector
    ///
    /// # Returns
    /// Cosine similarity in [-1, 1] range. Returns 0.0 if either vector has zero norm.
    ///
    /// # Example
    /// ```
    /// use vecstore::vectors::VectorOps;
    ///
    /// let a_indices = vec![0, 1, 2];
    /// let a_values = vec![1.0, 0.0, 1.0];
    /// let b_indices = vec![0, 1];
    /// let b_values = vec![1.0, 0.0];
    ///
    /// let sim = VectorOps::sparse_cosine(&a_indices, &a_values, &b_indices, &b_values);
    /// // Should be close to 1.0 / sqrt(2) ≈ 0.707
    /// assert!((sim - 0.707).abs() < 0.01);
    /// ```
    pub fn sparse_cosine(
        a_indices: &[usize],
        a_values: &[f32],
        b_indices: &[usize],
        b_values: &[f32],
    ) -> f32 {
        let dot = Self::sparse_dot(a_indices, a_values, b_indices, b_values);
        let norm_a = Self::sparse_norm(a_values);
        let norm_b = Self::sparse_norm(b_values);

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }
}

/// K-means clustering for vectors
pub struct KMeans {
    k: usize,
    max_iterations: usize,
    tolerance: f32,
}

impl KMeans {
    /// Create new K-means clusterer
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iterations: 100,
            tolerance: 1e-4,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Run K-means clustering
    ///
    /// Returns (centroids, assignments) where assignments[i] is the cluster index for vectors[i]
    pub fn fit(&self, vectors: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
        if vectors.len() < self.k {
            anyhow::bail!("Need at least k={} vectors, got {}", self.k, vectors.len());
        }

        let _dim = vectors[0].len();

        // Initialize centroids with k random vectors
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Vec<f32>> =
            vectors.choose_multiple(&mut rng, self.k).cloned().collect();

        let mut assignments = vec![0; vectors.len()];
        let mut prev_centroids = centroids.clone();

        for _iter in 0..self.max_iterations {
            // Assign each vector to nearest centroid
            for (i, v) in vectors.iter().enumerate() {
                let mut min_dist = f32::MAX;
                let mut best_cluster = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = euclidean_distance(v, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }

                assignments[i] = best_cluster;
            }

            // Recalculate centroids
            for (j, centroid) in centroids.iter_mut().enumerate().take(self.k) {
                let cluster_vectors: Vec<Vec<f32>> = vectors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == j)
                    .map(|(_, v)| v.clone())
                    .collect();

                if !cluster_vectors.is_empty() {
                    *centroid = VectorOps::centroid(&cluster_vectors)?;
                }
            }

            // Check convergence
            let max_centroid_change = centroids
                .iter()
                .zip(prev_centroids.iter())
                .map(|(c1, c2)| euclidean_distance(c1, c2))
                .fold(0.0f32, |a, b| a.max(b));

            if max_centroid_change < self.tolerance {
                break;
            }

            prev_centroids = centroids.clone();
        }

        Ok((centroids, assignments))
    }
}

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
    use approx::assert_relative_eq;

    #[test]
    fn test_vector_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = VectorOps::add(&a, &b).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_subtract() {
        let a = vec![5.0, 7.0, 9.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = VectorOps::subtract(&a, &b).unwrap();
        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_vector_scale() {
        let v = vec![1.0, 2.0, 3.0];
        let result = VectorOps::scale(&v, 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_centroid() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = VectorOps::centroid(&vectors).unwrap();
        assert_relative_eq!(result[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let result = VectorOps::normalize(&v);
        let norm = VectorOps::norm(&result);
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_kmeans() {
        let vectors = vec![
            vec![1.0, 1.0],
            vec![1.5, 2.0],
            vec![10.0, 10.0],
            vec![10.5, 11.0],
        ];

        let kmeans = KMeans::new(2);
        let (centroids, assignments) = kmeans.fit(&vectors).unwrap();

        assert_eq!(centroids.len(), 2);
        assert_eq!(assignments.len(), 4);

        // First two vectors should be in same cluster
        assert_eq!(assignments[0], assignments[1]);
        // Last two vectors should be in same cluster
        assert_eq!(assignments[2], assignments[3]);
        // But different clusters overall
        assert_ne!(assignments[0], assignments[2]);
    }

    // ========== Sparse Vector Operations Tests ==========

    #[test]
    fn test_sparse_dot_basic() {
        let a_indices = vec![5, 100, 500];
        let a_values = vec![1.0, 2.0, 3.0];
        let b_indices = vec![5, 200, 500];
        let b_values = vec![1.5, 2.5, 1.0];

        let dot = VectorOps::sparse_dot(&a_indices, &a_values, &b_indices, &b_values);
        // Matches at indices 5 and 500: 1.0*1.5 + 3.0*1.0 = 4.5
        assert_relative_eq!(dot, 4.5, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_dot_no_overlap() {
        let a_indices = vec![0, 1, 2];
        let a_values = vec![1.0, 2.0, 3.0];
        let b_indices = vec![10, 20, 30];
        let b_values = vec![1.0, 2.0, 3.0];

        let dot = VectorOps::sparse_dot(&a_indices, &a_values, &b_indices, &b_values);
        assert_eq!(dot, 0.0); // No overlapping indices
    }

    #[test]
    fn test_sparse_dot_full_overlap() {
        let a_indices = vec![0, 1, 2];
        let a_values = vec![1.0, 2.0, 3.0];
        let b_indices = vec![0, 1, 2];
        let b_values = vec![2.0, 3.0, 4.0];

        let dot = VectorOps::sparse_dot(&a_indices, &a_values, &b_indices, &b_values);
        // 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
        assert_relative_eq!(dot, 20.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_dot_empty() {
        let a_indices: Vec<usize> = vec![];
        let a_values: Vec<f32> = vec![];
        let b_indices = vec![0, 1];
        let b_values = vec![1.0, 2.0];

        let dot = VectorOps::sparse_dot(&a_indices, &a_values, &b_indices, &b_values);
        assert_eq!(dot, 0.0);
    }

    #[test]
    fn test_sparse_dot_different_lengths() {
        let a_indices = vec![0, 5, 10, 15];
        let a_values = vec![1.0, 2.0, 3.0, 4.0];
        let b_indices = vec![5, 10];
        let b_values = vec![2.0, 3.0];

        let dot = VectorOps::sparse_dot(&a_indices, &a_values, &b_indices, &b_values);
        // Matches at 5 and 10: 2*2 + 3*3 = 13
        assert_relative_eq!(dot, 13.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_norm() {
        let values = vec![3.0, 4.0];
        let norm = VectorOps::sparse_norm(&values);
        assert_relative_eq!(norm, 5.0, epsilon = 1e-6); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_sparse_norm_empty() {
        let values: Vec<f32> = vec![];
        let norm = VectorOps::sparse_norm(&values);
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_sparse_cosine() {
        // Identical vectors should have cosine = 1.0
        let indices = vec![0, 1, 2];
        let values_a = vec![1.0, 2.0, 3.0];
        let values_b = vec![1.0, 2.0, 3.0];

        let sim = VectorOps::sparse_cosine(&indices, &values_a, &indices, &values_b);
        assert_relative_eq!(sim, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_cosine_orthogonal() {
        // Orthogonal vectors should have cosine = 0.0
        let a_indices = vec![0, 1];
        let a_values = vec![1.0, 0.0];
        let b_indices = vec![2, 3];
        let b_values = vec![0.0, 1.0];

        let sim = VectorOps::sparse_cosine(&a_indices, &a_values, &b_indices, &b_values);
        assert_relative_eq!(sim, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_cosine_zero_norm() {
        let indices = vec![0, 1];
        let values_a = vec![1.0, 2.0];
        let values_b = vec![0.0, 0.0];

        let sim = VectorOps::sparse_cosine(&indices, &values_a, &indices, &values_b);
        assert_eq!(sim, 0.0); // Should handle zero norm gracefully
    }

    #[test]
    fn test_sparse_vs_dense_equivalence() {
        // Sparse and dense should give same results
        let a_indices = vec![0, 2, 5];
        let a_values = vec![1.0, 2.0, 3.0];
        let b_indices = vec![1, 2, 5];
        let b_values = vec![1.5, 2.5, 1.0];

        // Sparse dot product
        let sparse_dot = VectorOps::sparse_dot(&a_indices, &a_values, &b_indices, &b_values);

        // Dense equivalent (materialize to 10-dim vectors)
        let mut dense_a = vec![0.0; 10];
        let mut dense_b = vec![0.0; 10];
        for (&idx, &val) in a_indices.iter().zip(&a_values) {
            dense_a[idx] = val;
        }
        for (&idx, &val) in b_indices.iter().zip(&b_values) {
            dense_b[idx] = val;
        }
        let dense_dot = VectorOps::dot(&dense_a, &dense_b).unwrap();

        assert_relative_eq!(sparse_dot, dense_dot, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_high_dimension() {
        // Test with very high dimension but few non-zero values
        // This demonstrates the efficiency gain of sparse representation
        let a_indices = vec![10, 1000, 10000, 100000];
        let a_values = vec![1.0, 2.0, 3.0, 4.0];
        let b_indices = vec![1000, 10000, 50000];
        let b_values = vec![2.0, 3.0, 1.0];

        let dot = VectorOps::sparse_dot(&a_indices, &a_values, &b_indices, &b_values);
        // Matches at 1000 and 10000: 2*2 + 3*3 = 13
        assert_relative_eq!(dot, 13.0, epsilon = 1e-6);
    }
}
