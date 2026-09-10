// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! Privacy-Preserving Vector Search
//!
//! This module provides privacy protection for vector embeddings:
//! - **Differential Privacy**: Add calibrated noise to protect individual vectors
//! - **Local DP**: Apply privacy at the query level
//! - **Secure Aggregation**: Combine vectors without exposing individuals
//! - **Privacy Budgeting**: Track and enforce cumulative privacy loss
//! - **Anonymization**: Remove personally identifiable patterns from embeddings
//!
//! # Example
//!
//! ```ignore
//! use vecstore::privacy::{PrivacyEngine, PrivacyConfig, DPMechanism};
//!
//! let config = PrivacyConfig {
//!     epsilon: 1.0,  // Privacy parameter (lower = more private)
//!     delta: 1e-5,   // Failure probability
//!     mechanism: DPMechanism::Gaussian,
//!     ..Default::default()
//! };
//!
//! let engine = PrivacyEngine::new(config);
//!
//! // Apply differential privacy to a vector before indexing
//! let private_vector = engine.privatize(&original_vector)?;
//!
//! // Query with local DP
//! let private_query = engine.privatize_query(&query_vector)?;
//! ```

use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::error::VecStoreError;

/// Laplace distribution for differential privacy
/// Laplace(μ, b) where μ is location and b is scale
pub struct Laplace {
    location: f64,
    scale: f64,
}

impl Laplace {
    pub fn new(location: f64, scale: f64) -> Result<Self, &'static str> {
        if scale <= 0.0 {
            return Err("Scale must be positive");
        }
        Ok(Self { location, scale })
    }
}

impl Distribution<f64> for Laplace {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // Inverse CDF method for Laplace distribution
        let u: f64 = rng.random_range(-0.5..0.5);
        self.location - self.scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

/// Differential privacy mechanism
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum DPMechanism {
    /// Laplace mechanism (pure DP)
    Laplace,
    /// Gaussian mechanism (approximate DP)
    #[default]
    Gaussian,
    /// Discrete Laplace for integer values
    DiscreteLaplace,
    /// Randomized response for binary data
    RandomizedResponse,
}

/// Privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Privacy parameter epsilon (lower = more private)
    pub epsilon: f64,
    /// Failure probability delta (for approximate DP)
    pub delta: f64,
    /// DP mechanism to use
    pub mechanism: DPMechanism,
    /// Sensitivity of the data (L2 sensitivity for vectors)
    pub sensitivity: f64,
    /// Maximum privacy budget per entity
    pub max_budget: f64,
    /// Enable privacy budgeting
    pub track_budget: bool,
    /// Clip vectors to this L2 norm before adding noise
    pub clip_norm: Option<f64>,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
            mechanism: DPMechanism::Gaussian,
            sensitivity: 1.0,
            max_budget: 10.0,
            track_budget: true,
            clip_norm: Some(1.0),
        }
    }
}

/// Privacy budget tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudget {
    /// Total epsilon spent
    pub epsilon_spent: f64,
    /// Total delta spent
    pub delta_spent: f64,
    /// Maximum allowed epsilon
    pub max_epsilon: f64,
    /// Maximum allowed delta
    pub max_delta: f64,
    /// Number of queries made
    pub query_count: u64,
    /// Per-entity budgets
    pub entity_budgets: HashMap<String, EntityBudget>,
}

/// Budget for a single entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityBudget {
    pub epsilon_spent: f64,
    pub delta_spent: f64,
    pub query_count: u64,
}

impl PrivacyBudget {
    /// Create a new privacy budget
    pub fn new(max_epsilon: f64, max_delta: f64) -> Self {
        Self {
            epsilon_spent: 0.0,
            delta_spent: 0.0,
            max_epsilon,
            max_delta,
            query_count: 0,
            entity_budgets: HashMap::new(),
        }
    }

    /// Check if budget is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.epsilon_spent >= self.max_epsilon || self.delta_spent >= self.max_delta
    }

    /// Check remaining budget
    pub fn remaining(&self) -> (f64, f64) {
        (
            (self.max_epsilon - self.epsilon_spent).max(0.0),
            (self.max_delta - self.delta_spent).max(0.0),
        )
    }

    /// Spend budget
    pub fn spend(&mut self, epsilon: f64, delta: f64) -> Result<(), VecStoreError> {
        if self.epsilon_spent + epsilon > self.max_epsilon {
            return Err(VecStoreError::PrivacyBudgetExhausted {
                requested: epsilon,
                remaining: self.max_epsilon - self.epsilon_spent,
            });
        }
        if self.delta_spent + delta > self.max_delta {
            return Err(VecStoreError::PrivacyBudgetExhausted {
                requested: delta,
                remaining: self.max_delta - self.delta_spent,
            });
        }

        self.epsilon_spent += epsilon;
        self.delta_spent += delta;
        self.query_count += 1;

        Ok(())
    }

    /// Spend budget for a specific entity
    pub fn spend_entity(
        &mut self,
        entity_id: &str,
        epsilon: f64,
        delta: f64,
    ) -> Result<(), VecStoreError> {
        let entry = self
            .entity_budgets
            .entry(entity_id.to_string())
            .or_insert_with(|| EntityBudget {
                epsilon_spent: 0.0,
                delta_spent: 0.0,
                query_count: 0,
            });

        if entry.epsilon_spent + epsilon > self.max_epsilon {
            return Err(VecStoreError::PrivacyBudgetExhausted {
                requested: epsilon,
                remaining: self.max_epsilon - entry.epsilon_spent,
            });
        }

        entry.epsilon_spent += epsilon;
        entry.delta_spent += delta;
        entry.query_count += 1;

        // Also update global budget
        self.spend(epsilon, delta)
    }
}

/// Result of privacy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyAnalysis {
    /// Effective epsilon
    pub epsilon: f64,
    /// Effective delta
    pub delta: f64,
    /// Noise scale used
    pub noise_scale: f64,
    /// Average noise magnitude added
    pub avg_noise_magnitude: f64,
    /// Maximum noise magnitude
    pub max_noise_magnitude: f64,
    /// Signal-to-noise ratio
    pub snr: f64,
    /// Estimated utility loss (0-1)
    pub utility_loss: f64,
}

/// Main privacy engine
pub struct PrivacyEngine {
    config: PrivacyConfig,
    budget: Arc<RwLock<PrivacyBudget>>,
    rng: rand::rngs::ThreadRng,
}

impl PrivacyEngine {
    /// Create a new privacy engine
    pub fn new(config: PrivacyConfig) -> Self {
        let budget = PrivacyBudget::new(config.max_budget, config.delta * 100.0);

        Self {
            config,
            budget: Arc::new(RwLock::new(budget)),
            rng: rand::rng(),
        }
    }

    /// Apply differential privacy to a vector
    pub fn privatize(&self, vector: &[f32]) -> Result<Vec<f32>, VecStoreError> {
        // Track budget if enabled
        if self.config.track_budget {
            let mut budget = self
                .budget
                .write()
                .map_err(|_| VecStoreError::Internal("Failed to acquire budget lock".into()))?;
            budget.spend(self.config.epsilon, self.config.delta)?;
        }

        // Clip vector if needed
        let clipped = if let Some(max_norm) = self.config.clip_norm {
            clip_vector(vector, max_norm)
        } else {
            vector.to_vec()
        };

        // Add noise based on mechanism
        let noisy = match self.config.mechanism {
            DPMechanism::Gaussian => self.add_gaussian_noise(&clipped)?,
            DPMechanism::Laplace => self.add_laplace_noise(&clipped)?,
            DPMechanism::DiscreteLaplace => self.add_discrete_laplace_noise(&clipped)?,
            DPMechanism::RandomizedResponse => {
                return Err(VecStoreError::InvalidInput(
                    "RandomizedResponse not applicable to continuous vectors".into(),
                ));
            },
        };

        Ok(noisy)
    }

    /// Apply local DP to a query vector
    pub fn privatize_query(&self, query: &[f32]) -> Result<Vec<f32>, VecStoreError> {
        // Query privacy uses less budget (can use advanced composition)
        let query_epsilon = self.config.epsilon * 0.1; // Use 10% of vector budget

        if self.config.track_budget {
            let mut budget = self
                .budget
                .write()
                .map_err(|_| VecStoreError::Internal("Failed to acquire budget lock".into()))?;
            budget.spend(query_epsilon, self.config.delta * 0.1)?;
        }

        // Clip and add noise
        let clipped = if let Some(max_norm) = self.config.clip_norm {
            clip_vector(query, max_norm)
        } else {
            query.to_vec()
        };

        // Use reduced noise for queries
        let scale = self.config.sensitivity / query_epsilon;
        self.add_gaussian_noise_with_scale(&clipped, scale * 0.5)
    }

    /// Privatize a batch of vectors
    pub fn privatize_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, VecStoreError> {
        // Use advanced composition for batch privacy
        let n = vectors.len() as f64;
        let batch_epsilon = self.config.epsilon * n.sqrt(); // Sublinear in batch size

        if self.config.track_budget {
            let mut budget = self
                .budget
                .write()
                .map_err(|_| VecStoreError::Internal("Failed to acquire budget lock".into()))?;
            budget.spend(batch_epsilon, self.config.delta * n)?;
        }

        let mut result = Vec::with_capacity(vectors.len());
        for vec in vectors {
            let clipped = if let Some(max_norm) = self.config.clip_norm {
                clip_vector(vec, max_norm)
            } else {
                vec.clone()
            };

            let noisy = match self.config.mechanism {
                DPMechanism::Gaussian => self.add_gaussian_noise(&clipped)?,
                DPMechanism::Laplace => self.add_laplace_noise(&clipped)?,
                _ => self.add_gaussian_noise(&clipped)?,
            };

            result.push(noisy);
        }

        Ok(result)
    }

    /// Secure aggregation of vectors (e.g., for federated learning)
    pub fn secure_aggregate(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>, VecStoreError> {
        if vectors.is_empty() {
            return Err(VecStoreError::InvalidInput("Empty vector list".into()));
        }

        let dim = vectors[0].len();
        let n = vectors.len() as f32;

        // Compute average
        let mut sum = vec![0.0f32; dim];
        for vec in vectors {
            if vec.len() != dim {
                return Err(VecStoreError::DimensionMismatch {
                    expected: dim,
                    got: vec.len(),
                });
            }
            for (i, &v) in vec.iter().enumerate() {
                sum[i] += v;
            }
        }

        let avg: Vec<f32> = sum.iter().map(|s| s / n).collect();

        // Add noise for DP (sensitivity is 1/n for average)
        let scale = (self.config.sensitivity / n as f64) / self.config.epsilon;
        self.add_gaussian_noise_with_scale(&avg, scale)
    }

    /// Anonymize a vector by removing identifiable patterns
    pub fn anonymize(&self, vector: &[f32], k_anonymity: usize) -> Result<Vec<f32>, VecStoreError> {
        // Simple anonymization: add noise and round to reduce uniqueness
        let noisy = self.privatize(vector)?;

        // Round to reduce precision (increases k-anonymity)
        let precision = 1.0 / k_anonymity as f32;
        let anonymized: Vec<f32> = noisy
            .iter()
            .map(|&v| (v / precision).round() * precision)
            .collect();

        Ok(anonymized)
    }

    /// Calculate the noise scale for current configuration
    pub fn noise_scale(&self) -> f64 {
        match self.config.mechanism {
            DPMechanism::Gaussian => {
                // Gaussian mechanism: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
                let c = (2.0 * (1.25 / self.config.delta).ln()).sqrt();
                self.config.sensitivity * c / self.config.epsilon
            },
            DPMechanism::Laplace => {
                // Laplace mechanism: scale = sensitivity / epsilon
                self.config.sensitivity / self.config.epsilon
            },
            _ => self.config.sensitivity / self.config.epsilon,
        }
    }

    /// Analyze privacy-utility tradeoff
    pub fn analyze(&self, sample_vector: &[f32]) -> Result<PrivacyAnalysis, VecStoreError> {
        let scale = self.noise_scale();

        // Estimate noise magnitude
        let dim = sample_vector.len() as f64;
        let expected_noise_mag = match self.config.mechanism {
            DPMechanism::Gaussian => scale * dim.sqrt(),
            DPMechanism::Laplace => scale * dim,
            _ => scale * dim.sqrt(),
        };

        // Signal magnitude
        let signal_mag: f64 = sample_vector
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt();

        let snr = if expected_noise_mag > 0.0 {
            signal_mag / expected_noise_mag
        } else {
            f64::INFINITY
        };

        // Utility loss estimate (heuristic)
        let utility_loss = 1.0 / (1.0 + snr);

        Ok(PrivacyAnalysis {
            epsilon: self.config.epsilon,
            delta: self.config.delta,
            noise_scale: scale,
            avg_noise_magnitude: expected_noise_mag,
            max_noise_magnitude: expected_noise_mag * 3.0, // 3-sigma bound
            snr,
            utility_loss,
        })
    }

    /// Get remaining privacy budget
    pub fn remaining_budget(&self) -> Result<(f64, f64), VecStoreError> {
        let budget = self
            .budget
            .read()
            .map_err(|_| VecStoreError::Internal("Failed to acquire budget lock".into()))?;

        Ok(budget.remaining())
    }

    /// Check if privacy budget is exhausted
    pub fn is_budget_exhausted(&self) -> Result<bool, VecStoreError> {
        let budget = self
            .budget
            .read()
            .map_err(|_| VecStoreError::Internal("Failed to acquire budget lock".into()))?;

        Ok(budget.is_exhausted())
    }

    /// Reset privacy budget
    pub fn reset_budget(&self) -> Result<(), VecStoreError> {
        let mut budget = self
            .budget
            .write()
            .map_err(|_| VecStoreError::Internal("Failed to acquire budget lock".into()))?;

        *budget = PrivacyBudget::new(self.config.max_budget, self.config.delta * 100.0);
        Ok(())
    }

    /// Get privacy budget snapshot
    pub fn get_budget_snapshot(&self) -> Result<PrivacyBudget, VecStoreError> {
        let budget = self
            .budget
            .read()
            .map_err(|_| VecStoreError::Internal("Failed to acquire budget lock".into()))?;

        Ok(budget.clone())
    }

    // === Private Methods ===

    fn add_gaussian_noise(&self, vector: &[f32]) -> Result<Vec<f32>, VecStoreError> {
        let scale = self.noise_scale();
        self.add_gaussian_noise_with_scale(vector, scale)
    }

    fn add_gaussian_noise_with_scale(
        &self,
        vector: &[f32],
        scale: f64,
    ) -> Result<Vec<f32>, VecStoreError> {
        let normal = Normal::new(0.0, scale)
            .map_err(|e| VecStoreError::InvalidInput(format!("Invalid noise scale: {}", e)))?;

        let mut rng = rand::rng();
        let noisy: Vec<f32> = vector
            .iter()
            .map(|&v| v + normal.sample(&mut rng) as f32)
            .collect();

        Ok(noisy)
    }

    fn add_laplace_noise(&self, vector: &[f32]) -> Result<Vec<f32>, VecStoreError> {
        let scale = self.config.sensitivity / self.config.epsilon;
        let laplace = Laplace::new(0.0, scale)
            .map_err(|e| VecStoreError::InvalidInput(format!("Invalid noise scale: {}", e)))?;

        let mut rng = rand::rng();
        let noisy: Vec<f32> = vector
            .iter()
            .map(|&v| v + laplace.sample(&mut rng) as f32)
            .collect();

        Ok(noisy)
    }

    fn add_discrete_laplace_noise(&self, vector: &[f32]) -> Result<Vec<f32>, VecStoreError> {
        // Discretized Laplace for integer-like values
        let scale = self.config.sensitivity / self.config.epsilon;
        let laplace = Laplace::new(0.0, scale)
            .map_err(|e| VecStoreError::InvalidInput(format!("Invalid noise scale: {}", e)))?;

        let mut rng = rand::rng();
        let noisy: Vec<f32> = vector
            .iter()
            .map(|&v| {
                let noise = laplace.sample(&mut rng).round() as f32;
                v + noise
            })
            .collect();

        Ok(noisy)
    }
}

/// Clip vector to have at most the given L2 norm
fn clip_vector(vector: &[f32], max_norm: f64) -> Vec<f32> {
    let current_norm: f64 = vector
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();

    if current_norm <= max_norm {
        vector.to_vec()
    } else {
        let scale = (max_norm / current_norm) as f32;
        vector.iter().map(|&v| v * scale).collect()
    }
}

/// Privacy configuration builder
pub struct PrivacyConfigBuilder {
    config: PrivacyConfig,
}

impl PrivacyConfigBuilder {
    /// Start with default config
    pub fn new() -> Self {
        Self {
            config: PrivacyConfig::default(),
        }
    }

    /// Set epsilon (privacy parameter)
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Set delta (failure probability)
    pub fn delta(mut self, delta: f64) -> Self {
        self.config.delta = delta;
        self
    }

    /// Set the DP mechanism
    pub fn mechanism(mut self, mechanism: DPMechanism) -> Self {
        self.config.mechanism = mechanism;
        self
    }

    /// Set sensitivity
    pub fn sensitivity(mut self, sensitivity: f64) -> Self {
        self.config.sensitivity = sensitivity;
        self
    }

    /// Set clip norm
    pub fn clip_norm(mut self, norm: f64) -> Self {
        self.config.clip_norm = Some(norm);
        self
    }

    /// Disable clipping
    pub fn no_clipping(mut self) -> Self {
        self.config.clip_norm = None;
        self
    }

    /// Set maximum budget
    pub fn max_budget(mut self, budget: f64) -> Self {
        self.config.max_budget = budget;
        self
    }

    /// Enable/disable budget tracking
    pub fn track_budget(mut self, track: bool) -> Self {
        self.config.track_budget = track;
        self
    }

    /// Build the configuration
    pub fn build(self) -> PrivacyConfig {
        self.config
    }
}

impl Default for PrivacyConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset privacy configurations
pub struct PrivacyPresets;

impl PrivacyPresets {
    /// High privacy, lower utility (epsilon = 0.1)
    pub fn high_privacy() -> PrivacyConfig {
        PrivacyConfigBuilder::new()
            .epsilon(0.1)
            .delta(1e-6)
            .mechanism(DPMechanism::Gaussian)
            .build()
    }

    /// Balanced privacy and utility (epsilon = 1.0)
    pub fn balanced() -> PrivacyConfig {
        PrivacyConfigBuilder::new()
            .epsilon(1.0)
            .delta(1e-5)
            .mechanism(DPMechanism::Gaussian)
            .build()
    }

    /// Lower privacy, higher utility (epsilon = 5.0)
    pub fn high_utility() -> PrivacyConfig {
        PrivacyConfigBuilder::new()
            .epsilon(5.0)
            .delta(1e-4)
            .mechanism(DPMechanism::Laplace)
            .build()
    }

    /// GDPR-compliant setting (conservative)
    pub fn gdpr_compliant() -> PrivacyConfig {
        PrivacyConfigBuilder::new()
            .epsilon(0.5)
            .delta(1e-7)
            .mechanism(DPMechanism::Gaussian)
            .max_budget(5.0)
            .track_budget(true)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privatize_vector() {
        let config = PrivacyConfig::default();
        let engine = PrivacyEngine::new(config);

        let original = vec![1.0, 0.0, 0.0];
        let private = engine.privatize(&original).unwrap();

        // Should have same dimension
        assert_eq!(private.len(), original.len());

        // Should be different (with high probability)
        let diff: f32 = original
            .iter()
            .zip(private.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_vector_clipping() {
        let large_vec = vec![10.0, 10.0, 10.0];
        let clipped = clip_vector(&large_vec, 1.0);

        // Should have norm <= 1
        let norm: f64 = clipped
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt();

        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_noise_scale_calculation() {
        // Gaussian mechanism
        let config = PrivacyConfigBuilder::new()
            .epsilon(1.0)
            .delta(1e-5)
            .mechanism(DPMechanism::Gaussian)
            .sensitivity(1.0)
            .build();

        let engine = PrivacyEngine::new(config);
        let scale = engine.noise_scale();

        assert!(scale > 0.0);
        assert!(scale < 10.0); // Reasonable range

        // Laplace mechanism
        let config2 = PrivacyConfigBuilder::new()
            .epsilon(1.0)
            .mechanism(DPMechanism::Laplace)
            .sensitivity(1.0)
            .build();

        let engine2 = PrivacyEngine::new(config2);
        let scale2 = engine2.noise_scale();

        assert!((scale2 - 1.0).abs() < 1e-5); // sensitivity/epsilon = 1.0
    }

    #[test]
    fn test_privacy_budget_tracking() {
        let config = PrivacyConfigBuilder::new()
            .epsilon(1.0)
            .max_budget(3.0)
            .track_budget(true)
            .build();

        let engine = PrivacyEngine::new(config);

        // First few calls should succeed
        let vec = vec![1.0, 0.0];
        engine.privatize(&vec).unwrap();
        engine.privatize(&vec).unwrap();

        // Check budget
        let (remaining_eps, _) = engine.remaining_budget().unwrap();
        assert!(remaining_eps < 3.0);
        assert!(remaining_eps > 0.0);

        // Should eventually exhaust budget
        let mut exhausted = false;
        for _ in 0..10 {
            if engine.privatize(&vec).is_err() {
                exhausted = true;
                break;
            }
        }

        // Budget should be exhausted
        assert!(exhausted || engine.is_budget_exhausted().unwrap());
    }

    #[test]
    fn test_secure_aggregation() {
        let config = PrivacyConfig::default();
        let engine = PrivacyEngine::new(config);

        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let aggregate = engine.secure_aggregate(&vectors).unwrap();

        // Should be approximately average
        assert_eq!(aggregate.len(), 3);

        // Each component should be roughly 1/3 (plus noise)
        for &v in &aggregate {
            assert!(v > -1.0 && v < 2.0); // Reasonable range
        }
    }

    #[test]
    fn test_privacy_analysis() {
        let config = PrivacyConfig::default();
        let engine = PrivacyEngine::new(config);

        let vec = vec![1.0, 0.0, 0.0];
        let analysis = engine.analyze(&vec).unwrap();

        assert_eq!(analysis.epsilon, 1.0);
        assert!(analysis.noise_scale > 0.0);
        assert!(analysis.snr > 0.0);
        assert!(analysis.utility_loss >= 0.0 && analysis.utility_loss <= 1.0);
    }

    #[test]
    fn test_presets() {
        let high_priv = PrivacyPresets::high_privacy();
        let balanced = PrivacyPresets::balanced();
        let high_util = PrivacyPresets::high_utility();

        // Higher epsilon = lower privacy = more utility
        assert!(high_priv.epsilon < balanced.epsilon);
        assert!(balanced.epsilon < high_util.epsilon);
    }

    #[test]
    fn test_anonymize() {
        let config = PrivacyConfig::default();
        let engine = PrivacyEngine::new(config);

        let vec = vec![0.123456, 0.654321, 0.111111];
        let anon = engine.anonymize(&vec, 10).unwrap();

        // Should have reduced precision
        for &v in &anon {
            let remainder = (v * 10.0) % 1.0;
            assert!(remainder.abs() < 0.01 || (1.0 - remainder.abs()) < 0.01);
        }
    }

    #[test]
    fn test_privatize_query() {
        let config = PrivacyConfig::default();
        let engine = PrivacyEngine::new(config);

        let query = vec![1.0, 0.0, 0.0];
        let private_query = engine.privatize_query(&query).unwrap();

        // Should be perturbed
        assert_eq!(private_query.len(), 3);

        let diff: f32 = query
            .iter()
            .zip(private_query.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_batch_privatization() {
        let config = PrivacyConfig::default();
        let engine = PrivacyEngine::new(config);

        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];

        let private = engine.privatize_batch(&vectors).unwrap();

        assert_eq!(private.len(), 3);
        for (orig, priv_vec) in vectors.iter().zip(private.iter()) {
            assert_eq!(orig.len(), priv_vec.len());
        }
    }

    #[test]
    fn test_laplace_mechanism() {
        let config = PrivacyConfigBuilder::new()
            .mechanism(DPMechanism::Laplace)
            .epsilon(1.0)
            .build();

        let engine = PrivacyEngine::new(config);

        let vec = vec![1.0, 0.0, 0.0];
        let private = engine.privatize(&vec).unwrap();

        assert_eq!(private.len(), 3);
    }

    #[test]
    fn test_reset_budget() {
        let config = PrivacyConfigBuilder::new()
            .epsilon(1.0)
            .max_budget(2.0)
            .track_budget(true)
            .build();

        let engine = PrivacyEngine::new(config);

        let vec = vec![1.0, 0.0];
        engine.privatize(&vec).unwrap();

        let (remaining, _) = engine.remaining_budget().unwrap();
        assert!(remaining < 2.0);

        // Reset budget
        engine.reset_budget().unwrap();

        let (remaining_after, _) = engine.remaining_budget().unwrap();
        assert!((remaining_after - 2.0).abs() < 0.1);
    }
}
