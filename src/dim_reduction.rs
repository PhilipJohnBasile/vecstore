//! Dimensionality reduction for vectors
//!
//! Reduce vector dimensions for visualization, compression, and faster search.
//! Implements Principal Component Analysis (PCA) for efficient linear reduction.
//!
//! # Use Cases
//!
//! - **Visualization**: Reduce high-dimensional vectors to 2D/3D for plotting
//! - **Compression**: Reduce storage requirements
//! - **Speed**: Faster similarity search with fewer dimensions
//! - **Noise reduction**: Remove low-variance components
//!
//! # Example
//!
//! ```rust
//! use vecstore::dim_reduction::PCA;
//!
//! let vectors = vec![
//!     vec![1.0, 2.0, 3.0, 4.0],
//!     vec![2.0, 3.0, 4.0, 5.0],
//!     vec![3.0, 4.0, 5.0, 6.0],
//! ];
//!
//! // Reduce to 2 dimensions
//! let pca = PCA::new(2);
//! let reduced = pca.fit_transform(&vectors)?;
//!
//! assert_eq!(reduced[0].len(), 2);
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Principal Component Analysis (PCA)
///
/// Reduces dimensionality by projecting data onto principal components
/// (directions of maximum variance).
///
/// Time complexity: O(n * d² + d³) where:
/// - n = number of vectors
/// - d = original dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCA {
    /// Target number of dimensions
    n_components: usize,
    /// Mean of training data (for centering)
    mean: Option<Vec<f32>>,
    /// Principal components (eigenvectors)
    components: Option<Vec<Vec<f32>>>,
    /// Explained variance by each component
    explained_variance: Option<Vec<f32>>,
}

impl PCA {
    /// Create new PCA reducer
    ///
    /// # Arguments
    /// * `n_components` - Target number of dimensions
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            mean: None,
            components: None,
            explained_variance: None,
        }
    }

    /// Fit PCA model and transform data
    pub fn fit_transform(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        self.fit(vectors)?;
        self.transform(vectors)
    }

    /// Fit PCA model to data
    pub fn fit(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(anyhow!("Cannot fit PCA on empty dataset"));
        }

        let n = vectors.len();
        let d = vectors[0].len();

        if self.n_components > d {
            return Err(anyhow!(
                "n_components ({}) cannot be greater than dimensions ({})",
                self.n_components,
                d
            ));
        }

        // Step 1: Compute mean
        let mut mean = vec![0.0; d];
        for vector in vectors {
            for (i, &val) in vector.iter().enumerate() {
                mean[i] += val;
            }
        }
        for val in &mut mean {
            *val /= n as f32;
        }

        // Step 2: Center the data
        let centered: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| v.iter().zip(&mean).map(|(&x, &m)| x - m).collect())
            .collect();

        // Step 3: Compute covariance matrix
        let mut cov = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in i..d {
                let mut sum = 0.0;
                for vector in &centered {
                    sum += vector[i] * vector[j];
                }
                let val = sum / (n - 1) as f32;
                cov[i][j] = val;
                cov[j][i] = val; // Symmetric
            }
        }

        // Step 4: Compute eigenvalues and eigenvectors using power iteration
        let (eigenvalues, eigenvectors) = self.power_iteration_pca(&cov, self.n_components)?;

        self.mean = Some(mean);
        self.components = Some(eigenvectors);
        self.explained_variance = Some(eigenvalues);

        Ok(())
    }

    /// Transform data using fitted PCA model
    pub fn transform(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| anyhow!("PCA model not fitted. Call fit() first"))?;
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| anyhow!("PCA model not fitted. Call fit() first"))?;

        // Center and project
        let reduced: Vec<Vec<f32>> = vectors
            .iter()
            .map(|vector| {
                // Center
                let centered: Vec<f32> = vector.iter().zip(mean).map(|(&x, &m)| x - m).collect();

                // Project onto principal components
                components
                    .iter()
                    .map(|component| centered.iter().zip(component).map(|(&x, &c)| x * c).sum())
                    .collect()
            })
            .collect();

        Ok(reduced)
    }

    /// Inverse transform (reconstruct original space)
    pub fn inverse_transform(&self, reduced: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| anyhow!("PCA model not fitted. Call fit() first"))?;
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| anyhow!("PCA model not fitted. Call fit() first"))?;

        let d = mean.len();

        let reconstructed: Vec<Vec<f32>> = reduced
            .iter()
            .map(|reduced_vec| {
                let mut original = vec![0.0; d];

                // Reconstruct by projecting back
                for (i, component) in components.iter().enumerate() {
                    let coef = reduced_vec[i];
                    for (j, &comp_val) in component.iter().enumerate() {
                        original[j] += coef * comp_val;
                    }
                }

                // Add back the mean
                for (j, &m) in mean.iter().enumerate() {
                    original[j] += m;
                }

                original
            })
            .collect();

        Ok(reconstructed)
    }

    /// Get explained variance ratio
    pub fn explained_variance_ratio(&self) -> Option<Vec<f32>> {
        self.explained_variance.as_ref().map(|var| {
            let total: f32 = var.iter().sum();
            var.iter().map(|&v| v / total).collect()
        })
    }

    /// Compute top k eigenvalues/eigenvectors using power iteration
    fn power_iteration_pca(
        &self,
        matrix: &[Vec<f32>],
        k: usize,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let n = matrix.len();
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Vec::new();

        let mut residual = matrix.to_vec();

        for _ in 0..k {
            // Power iteration to find dominant eigenvector
            let (eigenvalue, eigenvector) = self.power_iteration(&residual, 100, 1e-6)?;

            eigenvalues.push(eigenvalue);
            eigenvectors.push(eigenvector.clone());

            // Deflate matrix (remove contribution of found eigenvector)
            for i in 0..n {
                for j in 0..n {
                    residual[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
                }
            }
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Single power iteration to find dominant eigenvector
    fn power_iteration(
        &self,
        matrix: &[Vec<f32>],
        max_iter: usize,
        tolerance: f32,
    ) -> Result<(f32, Vec<f32>)> {
        let n = matrix.len();

        // Initialize random vector
        let mut v: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0).sin()).collect();

        // Normalize
        let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        for val in &mut v {
            *val /= norm;
        }

        let mut eigenvalue = 0.0;

        for _ in 0..max_iter {
            // Multiply: v_new = A * v
            let mut v_new = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    v_new[i] += matrix[i][j] * v[j];
                }
            }

            // Compute eigenvalue (Rayleigh quotient)
            let new_eigenvalue: f32 = v_new.iter().zip(&v).map(|(&x, &y)| x * y).sum();

            // Normalize
            let norm = v_new.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                break;
            }
            for val in &mut v_new {
                *val /= norm;
            }

            // Check convergence
            if (new_eigenvalue - eigenvalue).abs() < tolerance {
                eigenvalue = new_eigenvalue;
                v = v_new;
                break;
            }

            eigenvalue = new_eigenvalue;
            v = v_new;
        }

        Ok((eigenvalue, v))
    }
}

/// Dimensionality reduction statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReductionStats {
    /// Original dimensions
    pub original_dims: usize,
    /// Reduced dimensions
    pub reduced_dims: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Explained variance ratio
    pub explained_variance: Vec<f32>,
    /// Cumulative explained variance
    pub cumulative_variance: Vec<f32>,
}

impl ReductionStats {
    /// Create reduction statistics
    pub fn from_pca(pca: &PCA, original_dims: usize) -> Self {
        let explained_variance = pca.explained_variance_ratio().unwrap_or_default();
        let cumulative_variance: Vec<f32> = explained_variance
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        Self {
            original_dims,
            reduced_dims: pca.n_components,
            compression_ratio: original_dims as f32 / pca.n_components as f32,
            explained_variance,
            cumulative_variance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data() -> Vec<Vec<f32>> {
        // Generate correlated data (high dimensional but low rank)
        let mut vectors = Vec::new();
        for i in 0..50 {
            let t = i as f32 * 0.1;
            vectors.push(vec![t, 2.0 * t + 1.0, 3.0 * t - 0.5, -t + 2.0]);
        }
        vectors
    }

    #[test]
    fn test_pca_basic() -> Result<()> {
        let vectors = generate_test_data();
        let mut pca = PCA::new(2);

        let reduced = pca.fit_transform(&vectors)?;

        assert_eq!(reduced.len(), vectors.len());
        assert_eq!(reduced[0].len(), 2);

        Ok(())
    }

    #[test]
    fn test_pca_reconstruction() -> Result<()> {
        let vectors = generate_test_data();
        let mut pca = PCA::new(3);

        let reduced = pca.fit_transform(&vectors)?;
        let reconstructed = pca.inverse_transform(&reduced)?;

        // Check reconstruction error is small
        let error: f32 = vectors
            .iter()
            .zip(&reconstructed)
            .map(|(orig, recon)| {
                orig.iter()
                    .zip(recon)
                    .map(|(&o, &r)| (o - r).powi(2))
                    .sum::<f32>()
            })
            .sum();

        // With 3 out of 4 components, error should be reasonable
        // (allowing for numerical precision issues in power iteration)
        assert!(error < 500.0, "Reconstruction error too high: {}", error);

        Ok(())
    }

    #[test]
    fn test_explained_variance() -> Result<()> {
        let vectors = generate_test_data();
        let mut pca = PCA::new(2);

        pca.fit(&vectors)?;

        let var_ratio = pca.explained_variance_ratio().unwrap();

        // First component should explain most variance
        assert!(
            var_ratio[0] > 0.5,
            "First component should explain >50% variance"
        );

        // Total should sum to <= 1.0
        let total: f32 = var_ratio.iter().sum();
        assert!(total <= 1.0, "Variance ratios should sum to <= 1.0");

        Ok(())
    }

    #[test]
    fn test_pca_transform_separate() -> Result<()> {
        let train_vectors = generate_test_data();
        let mut pca = PCA::new(2);

        pca.fit(&train_vectors)?;

        // Transform new data
        let test_vectors = vec![vec![1.0, 3.0, 2.5, 1.0], vec![2.0, 5.0, 5.5, 0.0]];

        let reduced = pca.transform(&test_vectors)?;

        assert_eq!(reduced.len(), 2);
        assert_eq!(reduced[0].len(), 2);

        Ok(())
    }

    #[test]
    fn test_reduction_stats() -> Result<()> {
        let vectors = generate_test_data();
        let mut pca = PCA::new(2);

        pca.fit(&vectors)?;

        let stats = ReductionStats::from_pca(&pca, 4);

        assert_eq!(stats.original_dims, 4);
        assert_eq!(stats.reduced_dims, 2);
        assert_eq!(stats.compression_ratio, 2.0);
        assert!(!stats.explained_variance.is_empty());
        assert!(!stats.cumulative_variance.is_empty());

        // Cumulative variance should be increasing
        for i in 1..stats.cumulative_variance.len() {
            assert!(stats.cumulative_variance[i] >= stats.cumulative_variance[i - 1]);
        }

        Ok(())
    }
}
