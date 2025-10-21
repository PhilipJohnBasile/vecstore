//! Vector validation and quality checks
//!
//! This module provides comprehensive validation for vectors before insertion,
//! ensuring data quality and preventing common issues that can break search.
//!
//! # Features
//!
//! - **Value validation**: Check for NaN, infinity, and invalid values
//! - **Dimension validation**: Ensure consistent vector dimensions
//! - **Magnitude checks**: Detect zero vectors and magnitude issues
//! - **Normalization**: Auto-normalize vectors for cosine similarity
//! - **Quality metrics**: Compute vector quality scores
//! - **Batch validation**: Efficient validation of large datasets
//!
//! # Example
//!
//! ```rust
//! use vecstore::validation::{VectorValidator, ValidationConfig, ValidationResult};
//!
//! let config = ValidationConfig::strict();
//! let validator = VectorValidator::new(config);
//!
//! // Validate a single vector
//! let vector = vec![1.0, 2.0, 3.0];
//! match validator.validate(&vector) {
//!     ValidationResult::Valid => println!("Vector is valid"),
//!     ValidationResult::Invalid(errors) => println!("Errors: {:?}", errors),
//!     ValidationResult::Warning(warnings) => println!("Warnings: {:?}", warnings),
//! }
//!
//! // Auto-fix invalid vectors
//! let fixed = validator.auto_fix(&vector)?;
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Expected vector dimension (None = any dimension)
    pub expected_dimension: Option<usize>,

    /// Allow NaN values
    pub allow_nan: bool,

    /// Allow infinity values
    pub allow_infinity: bool,

    /// Allow zero vectors
    pub allow_zero_vectors: bool,

    /// Minimum magnitude (0.0 = no minimum)
    pub min_magnitude: f32,

    /// Maximum magnitude (0.0 = no maximum)
    pub max_magnitude: f32,

    /// Auto-normalize vectors
    pub auto_normalize: bool,

    /// Validation strictness
    pub strictness: ValidationStrictness,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            expected_dimension: None,
            allow_nan: false,
            allow_infinity: false,
            allow_zero_vectors: false,
            min_magnitude: 0.0,
            max_magnitude: 0.0,
            auto_normalize: false,
            strictness: ValidationStrictness::Standard,
        }
    }
}

impl ValidationConfig {
    /// Strict validation (reject any issues)
    pub fn strict() -> Self {
        Self {
            allow_nan: false,
            allow_infinity: false,
            allow_zero_vectors: false,
            min_magnitude: 0.001,
            max_magnitude: 1000.0,
            auto_normalize: false,
            strictness: ValidationStrictness::Strict,
            ..Default::default()
        }
    }

    /// Lenient validation (allow minor issues)
    pub fn lenient() -> Self {
        Self {
            allow_nan: false,
            allow_infinity: false,
            allow_zero_vectors: true,
            min_magnitude: 0.0,
            max_magnitude: 0.0,
            auto_normalize: true,
            strictness: ValidationStrictness::Lenient,
            ..Default::default()
        }
    }

    /// Permissive validation (only block critical issues)
    pub fn permissive() -> Self {
        Self {
            allow_nan: false,
            allow_infinity: false,
            allow_zero_vectors: true,
            min_magnitude: 0.0,
            max_magnitude: 0.0,
            auto_normalize: true,
            strictness: ValidationStrictness::Permissive,
            ..Default::default()
        }
    }
}

/// Validation strictness level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStrictness {
    /// Block all invalid vectors
    Strict,
    /// Block invalid, warn on suspicious
    Standard,
    /// Allow suspicious, block only critical
    Lenient,
    /// Allow almost everything
    Permissive,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationError {
    /// Vector contains NaN values
    ContainsNaN { indices: Vec<usize> },
    /// Vector contains infinity values
    ContainsInfinity { indices: Vec<usize> },
    /// Vector dimension mismatch
    DimensionMismatch { expected: usize, actual: usize },
    /// Vector is zero (all zeros)
    ZeroVector,
    /// Magnitude too small
    MagnitudeTooSmall { magnitude: f32, minimum: f32 },
    /// Magnitude too large
    MagnitudeTooLarge { magnitude: f32, maximum: f32 },
    /// All values are identical
    AllValuesIdentical { value: f32 },
    /// Vector is empty
    EmptyVector,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationWarning {
    /// Vector has very small magnitude
    SmallMagnitude { magnitude: f32 },
    /// Vector has very large magnitude
    LargeMagnitude { magnitude: f32 },
    /// Vector has high sparsity (many zeros)
    HighSparsity { sparsity: f32 },
    /// Vector values are very similar
    LowVariance { variance: f32 },
    /// Vector not normalized
    NotNormalized { magnitude: f32 },
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationResult {
    /// Vector is valid
    Valid,
    /// Vector has non-critical warnings
    Warning(Vec<ValidationWarning>),
    /// Vector has critical errors
    Invalid(Vec<ValidationError>),
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationResult::Valid)
    }

    pub fn is_warning(&self) -> bool {
        matches!(self, ValidationResult::Warning(_))
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, ValidationResult::Invalid(_))
    }

    pub fn errors(&self) -> Option<&Vec<ValidationError>> {
        match self {
            ValidationResult::Invalid(errors) => Some(errors),
            _ => None,
        }
    }

    pub fn warnings(&self) -> Option<&Vec<ValidationWarning>> {
        match self {
            ValidationResult::Warning(warnings) => Some(warnings),
            _ => None,
        }
    }
}

/// Vector quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Vector magnitude
    pub magnitude: f32,
    /// Variance of values
    pub variance: f32,
    /// Sparsity (ratio of zeros)
    pub sparsity: f32,
    /// Overall quality score (0.0-1.0)
    pub quality_score: f32,
    /// Number of unique values
    pub unique_values: usize,
    /// Min value
    pub min_value: f32,
    /// Max value
    pub max_value: f32,
}

/// Vector validator
pub struct VectorValidator {
    config: ValidationConfig,
}

impl VectorValidator {
    /// Create new validator with config
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Create strict validator
    pub fn strict() -> Self {
        Self::new(ValidationConfig::strict())
    }

    /// Create lenient validator
    pub fn lenient() -> Self {
        Self::new(ValidationConfig::lenient())
    }

    /// Validate a vector
    pub fn validate(&self, vector: &[f32]) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for empty vector
        if vector.is_empty() {
            errors.push(ValidationError::EmptyVector);
            return ValidationResult::Invalid(errors);
        }

        // Check dimension
        if let Some(expected_dim) = self.config.expected_dimension {
            if vector.len() != expected_dim {
                errors.push(ValidationError::DimensionMismatch {
                    expected: expected_dim,
                    actual: vector.len(),
                });
            }
        }

        // Check for NaN and infinity
        let nan_indices: Vec<usize> = vector
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_nan())
            .map(|(i, _)| i)
            .collect();

        let inf_indices: Vec<usize> = vector
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_infinite())
            .map(|(i, _)| i)
            .collect();

        if !nan_indices.is_empty() && !self.config.allow_nan {
            errors.push(ValidationError::ContainsNaN {
                indices: nan_indices,
            });
        }

        if !inf_indices.is_empty() && !self.config.allow_infinity {
            errors.push(ValidationError::ContainsInfinity {
                indices: inf_indices,
            });
        }

        // Calculate magnitude
        let magnitude = self.compute_magnitude(vector);

        // Check for zero vector
        if magnitude < 1e-9 && !self.config.allow_zero_vectors {
            errors.push(ValidationError::ZeroVector);
        }

        // Check magnitude range
        if self.config.min_magnitude > 0.0 && magnitude < self.config.min_magnitude {
            errors.push(ValidationError::MagnitudeTooSmall {
                magnitude,
                minimum: self.config.min_magnitude,
            });
        }

        if self.config.max_magnitude > 0.0 && magnitude > self.config.max_magnitude {
            errors.push(ValidationError::MagnitudeTooLarge {
                magnitude,
                maximum: self.config.max_magnitude,
            });
        }

        // Check if all values are identical
        if vector.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9) {
            let value = vector[0];
            if value.is_finite() {
                errors.push(ValidationError::AllValuesIdentical { value });
            }
        }

        // Generate warnings based on strictness
        if errors.is_empty() {
            self.generate_warnings(vector, magnitude, &mut warnings);
        }

        // Return result based on findings
        if !errors.is_empty() {
            ValidationResult::Invalid(errors)
        } else if !warnings.is_empty() && self.config.strictness != ValidationStrictness::Permissive
        {
            ValidationResult::Warning(warnings)
        } else {
            ValidationResult::Valid
        }
    }

    /// Auto-fix invalid vectors
    pub fn auto_fix(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.is_empty() {
            return Err(anyhow!("Cannot fix empty vector"));
        }

        let mut fixed = vector.to_vec();

        // Replace NaN and infinity with zeros
        for val in &mut fixed {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            }
        }

        // Normalize if requested or if magnitude is zero
        if self.config.auto_normalize {
            fixed = self.normalize(&fixed);
        } else {
            let magnitude = self.compute_magnitude(&fixed);
            if magnitude < 1e-9 {
                // Zero vector - replace with uniform small values
                let default_val = 1.0 / (fixed.len() as f32).sqrt();
                for val in &mut fixed {
                    *val = default_val;
                }
            }
        }

        Ok(fixed)
    }

    /// Normalize a vector
    pub fn normalize(&self, vector: &[f32]) -> Vec<f32> {
        let magnitude = self.compute_magnitude(vector);
        if magnitude < 1e-9 {
            // Return uniform vector
            let val = 1.0 / (vector.len() as f32).sqrt();
            vec![val; vector.len()]
        } else {
            vector.iter().map(|v| v / magnitude).collect()
        }
    }

    /// Compute quality metrics for a vector
    pub fn compute_quality(&self, vector: &[f32]) -> QualityMetrics {
        let magnitude = self.compute_magnitude(vector);
        let variance = self.compute_variance(vector);
        let sparsity = self.compute_sparsity(vector);
        let unique_values = self.count_unique_values(vector);

        let min_value = vector.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_value = vector.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute overall quality score (0.0-1.0)
        let mut quality_score = 1.0;

        // Penalize low magnitude
        if magnitude < 0.1 {
            quality_score *= 0.5;
        }

        // Penalize low variance
        if variance < 0.01 {
            quality_score *= 0.5;
        }

        // Penalize high sparsity
        if sparsity > 0.9 {
            quality_score *= 0.5;
        }

        // Reward diversity
        let diversity_ratio = unique_values as f32 / vector.len() as f32;
        quality_score *= diversity_ratio;

        QualityMetrics {
            magnitude,
            variance,
            sparsity,
            quality_score: quality_score.clamp(0.0, 1.0),
            unique_values,
            min_value,
            max_value,
        }
    }

    /// Batch validate multiple vectors
    pub fn validate_batch(&self, vectors: &[Vec<f32>]) -> Vec<(usize, ValidationResult)> {
        vectors
            .iter()
            .enumerate()
            .map(|(i, vec)| (i, self.validate(vec)))
            .collect()
    }

    /// Get validation statistics for a batch
    pub fn batch_statistics(&self, vectors: &[Vec<f32>]) -> BatchStatistics {
        let results = self.validate_batch(vectors);

        let total = results.len();
        let valid = results.iter().filter(|(_, r)| r.is_valid()).count();
        let warnings = results.iter().filter(|(_, r)| r.is_warning()).count();
        let invalid = results.iter().filter(|(_, r)| r.is_invalid()).count();

        let avg_quality = vectors
            .iter()
            .map(|v| self.compute_quality(v).quality_score)
            .sum::<f32>()
            / total as f32;

        BatchStatistics {
            total_vectors: total,
            valid_count: valid,
            warning_count: warnings,
            invalid_count: invalid,
            average_quality: avg_quality,
        }
    }

    // Helper methods

    /// Compute the magnitude (L2 norm) of a vector
    pub fn compute_magnitude(&self, vector: &[f32]) -> f32 {
        vector.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    fn compute_variance(&self, vector: &[f32]) -> f32 {
        let mean = vector.iter().sum::<f32>() / vector.len() as f32;
        let variance = vector.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vector.len() as f32;
        variance
    }

    fn compute_sparsity(&self, vector: &[f32]) -> f32 {
        let zeros = vector.iter().filter(|v| v.abs() < 1e-9).count();
        zeros as f32 / vector.len() as f32
    }

    fn count_unique_values(&self, vector: &[f32]) -> usize {
        let mut values: Vec<i32> = vector
            .iter()
            .map(|v| (v * 1000000.0) as i32) // Discretize for uniqueness
            .collect();
        values.sort_unstable();
        values.dedup();
        values.len()
    }

    fn generate_warnings(
        &self,
        vector: &[f32],
        magnitude: f32,
        warnings: &mut Vec<ValidationWarning>,
    ) {
        // Warn if magnitude is suspicious
        if magnitude < 0.01 {
            warnings.push(ValidationWarning::SmallMagnitude { magnitude });
        } else if magnitude > 100.0 {
            warnings.push(ValidationWarning::LargeMagnitude { magnitude });
        }

        // Warn about high sparsity
        let sparsity = self.compute_sparsity(vector);
        if sparsity > 0.9 {
            warnings.push(ValidationWarning::HighSparsity { sparsity });
        }

        // Warn about low variance
        let variance = self.compute_variance(vector);
        if variance < 0.01 {
            warnings.push(ValidationWarning::LowVariance { variance });
        }

        // Warn if not normalized
        if (magnitude - 1.0).abs() > 0.1 {
            warnings.push(ValidationWarning::NotNormalized { magnitude });
        }
    }
}

/// Batch validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    pub total_vectors: usize,
    pub valid_count: usize,
    pub warning_count: usize,
    pub invalid_count: usize,
    pub average_quality: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_vector() {
        let validator = VectorValidator::default();
        let vector = vec![1.0, 2.0, 3.0];
        let result = validator.validate(&vector);
        assert!(result.is_valid() || result.is_warning());
    }

    #[test]
    fn test_nan_detection() {
        let validator = VectorValidator::strict();
        let vector = vec![1.0, f32::NAN, 3.0];
        let result = validator.validate(&vector);
        assert!(result.is_invalid());
        if let ValidationResult::Invalid(errors) = result {
            assert!(errors
                .iter()
                .any(|e| matches!(e, ValidationError::ContainsNaN { .. })));
        }
    }

    #[test]
    fn test_infinity_detection() {
        let validator = VectorValidator::strict();
        let vector = vec![1.0, f32::INFINITY, 3.0];
        let result = validator.validate(&vector);
        assert!(result.is_invalid());
    }

    #[test]
    fn test_zero_vector() {
        let validator = VectorValidator::strict();
        let vector = vec![0.0, 0.0, 0.0];
        let result = validator.validate(&vector);
        assert!(result.is_invalid());
        if let ValidationResult::Invalid(errors) = result {
            assert!(errors
                .iter()
                .any(|e| matches!(e, ValidationError::ZeroVector)));
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = ValidationConfig {
            expected_dimension: Some(3),
            ..Default::default()
        };
        let validator = VectorValidator::new(config);
        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let result = validator.validate(&vector);
        assert!(result.is_invalid());
    }

    #[test]
    fn test_auto_fix_nan() {
        let validator = VectorValidator::lenient();
        let vector = vec![1.0, f32::NAN, 3.0];
        let fixed = validator.auto_fix(&vector).unwrap();
        assert!(fixed.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_normalize() {
        let validator = VectorValidator::default();
        let vector = vec![3.0, 4.0]; // magnitude = 5.0
        let normalized = validator.normalize(&vector);
        let magnitude = validator.compute_magnitude(&normalized);
        assert!((magnitude - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quality_metrics() {
        let validator = VectorValidator::default();
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let metrics = validator.compute_quality(&vector);
        assert!(metrics.quality_score > 0.0 && metrics.quality_score <= 1.0);
        assert!(metrics.variance > 0.0);
    }

    #[test]
    fn test_batch_validation() {
        let validator = VectorValidator::default();
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![f32::NAN, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let results = validator.validate_batch(&vectors);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_batch_statistics() {
        let validator = VectorValidator::strict();
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![f32::NAN, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let stats = validator.batch_statistics(&vectors);
        assert_eq!(stats.total_vectors, 3);
        assert!(stats.invalid_count > 0);
    }
}
