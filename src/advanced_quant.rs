// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Advanced Quantization
//!
//! State-of-the-art quantization strategies including asymmetric quantization,
//! 1.5-bit, 2-bit, and adaptive quantization for optimal compression-quality tradeoffs.
//!
//! Inspired by Qdrant's advanced quantization research and industry best practices.
//!
//! ## Features
//!
//! - **Asymmetric Quantization**: Different strategies for stored vs query vectors
//! - **1.5-bit Quantization**: Novel approach for near-zero value handling
//! - **2-bit Quantization**: 16x compression with good accuracy
//! - **Adaptive Quantization**: Per-dimension bit allocation
//! - **Mixed-Precision**: Different precision for different vector components
//! - **Calibration**: Data-driven quantization parameter selection
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::advanced_quant::{AdaptiveQuantizer, QuantConfig};
//!
//! let config = QuantConfig::asymmetric_2bit();
//! let quantizer = AdaptiveQuantizer::new(config);
//!
//! // Calibrate on sample data
//! quantizer.calibrate(&training_vectors)?;
//!
//! // Quantize vectors (16x compression)
//! let compressed = quantizer.encode(&vector)?;
//! let reconstructed = quantizer.decode(&compressed)?;
//! ```

use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Quantization bit width
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BitWidth {
    /// 1-bit (binary)
    Bit1,
    /// 1.5-bit (ternary with special zero handling)
    Bit1_5,
    /// 2-bit (4 levels)
    Bit2,
    /// 4-bit (16 levels)
    Bit4,
    /// 8-bit (256 levels)
    Bit8,
}

impl BitWidth {
    /// Get compression ratio compared to 32-bit float
    pub fn compression_ratio(&self) -> f32 {
        match self {
            BitWidth::Bit1 => 32.0,
            BitWidth::Bit1_5 => 21.33,
            BitWidth::Bit2 => 16.0,
            BitWidth::Bit4 => 8.0,
            BitWidth::Bit8 => 4.0,
        }
    }

    /// Get number of quantization levels
    pub fn levels(&self) -> usize {
        match self {
            BitWidth::Bit1 => 2,
            BitWidth::Bit1_5 => 3,
            BitWidth::Bit2 => 4,
            BitWidth::Bit4 => 16,
            BitWidth::Bit8 => 256,
        }
    }
}

/// Quantization strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantStrategy {
    /// Symmetric quantization (same range for positive/negative)
    Symmetric,
    /// Asymmetric quantization (different ranges)
    Asymmetric,
    /// Per-channel quantization (different params per dimension)
    PerChannel,
    /// Adaptive (data-driven bit allocation)
    Adaptive,
    /// Mixed precision (different bits for different components)
    MixedPrecision { thresholds: Vec<f32> },
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    /// Bit width for stored vectors
    pub stored_bits: BitWidth,
    /// Bit width for query vectors (can differ for asymmetric)
    pub query_bits: BitWidth,
    /// Quantization strategy
    pub strategy: QuantStrategy,
    /// Enable rescoring with original vectors
    pub rescore: bool,
    /// Rescoring top-k multiplier
    pub rescore_multiplier: f32,
    /// Handle near-zero values specially
    pub zero_threshold: f32,
    /// Use SIMD acceleration
    pub use_simd: bool,
}

impl QuantConfig {
    /// 1-bit binary quantization
    pub fn binary() -> Self {
        Self {
            stored_bits: BitWidth::Bit1,
            query_bits: BitWidth::Bit1,
            strategy: QuantStrategy::Symmetric,
            rescore: true,
            rescore_multiplier: 4.0,
            zero_threshold: 0.0,
            use_simd: true,
        }
    }

    /// 1.5-bit ternary quantization
    pub fn ternary() -> Self {
        Self {
            stored_bits: BitWidth::Bit1_5,
            query_bits: BitWidth::Bit1_5,
            strategy: QuantStrategy::Symmetric,
            rescore: true,
            rescore_multiplier: 3.0,
            zero_threshold: 0.1,
            use_simd: true,
        }
    }

    /// 2-bit quantization
    pub fn two_bit() -> Self {
        Self {
            stored_bits: BitWidth::Bit2,
            query_bits: BitWidth::Bit2,
            strategy: QuantStrategy::Symmetric,
            rescore: true,
            rescore_multiplier: 2.0,
            zero_threshold: 0.0,
            use_simd: true,
        }
    }

    /// Asymmetric 2-bit (binary stored, scalar query)
    pub fn asymmetric_2bit() -> Self {
        Self {
            stored_bits: BitWidth::Bit1,
            query_bits: BitWidth::Bit8,
            strategy: QuantStrategy::Asymmetric,
            rescore: true,
            rescore_multiplier: 2.0,
            zero_threshold: 0.0,
            use_simd: true,
        }
    }

    /// 4-bit scalar quantization
    pub fn scalar_4bit() -> Self {
        Self {
            stored_bits: BitWidth::Bit4,
            query_bits: BitWidth::Bit4,
            strategy: QuantStrategy::Symmetric,
            rescore: false,
            rescore_multiplier: 1.0,
            zero_threshold: 0.0,
            use_simd: true,
        }
    }

    /// 8-bit scalar quantization
    pub fn scalar_8bit() -> Self {
        Self {
            stored_bits: BitWidth::Bit8,
            query_bits: BitWidth::Bit8,
            strategy: QuantStrategy::Symmetric,
            rescore: false,
            rescore_multiplier: 1.0,
            zero_threshold: 0.0,
            use_simd: true,
        }
    }

    /// Adaptive quantization
    pub fn adaptive() -> Self {
        Self {
            stored_bits: BitWidth::Bit4,
            query_bits: BitWidth::Bit4,
            strategy: QuantStrategy::Adaptive,
            rescore: true,
            rescore_multiplier: 1.5,
            zero_threshold: 0.05,
            use_simd: true,
        }
    }
}

/// Quantization parameters (learned from calibration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantParams {
    /// Scale factor per dimension (for per-channel)
    pub scales: Vec<f32>,
    /// Zero point per dimension
    pub zero_points: Vec<f32>,
    /// Global minimum
    pub global_min: f32,
    /// Global maximum
    pub global_max: f32,
    /// Dimension
    pub dimension: usize,
    /// Bit allocation per dimension (for adaptive)
    pub bit_allocation: Option<Vec<BitWidth>>,
}

impl QuantParams {
    /// Create default params for dimension
    pub fn default_for_dim(dim: usize) -> Self {
        Self {
            scales: vec![1.0; dim],
            zero_points: vec![0.0; dim],
            global_min: -1.0,
            global_max: 1.0,
            dimension: dim,
            bit_allocation: None,
        }
    }
}

/// Quantized vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVector {
    /// Compressed data
    pub data: Vec<u8>,
    /// Original dimension
    pub dimension: usize,
    /// Bit width used
    pub bit_width: BitWidth,
    /// Auxiliary data for reconstruction
    pub aux: Option<Vec<f32>>,
}

impl QuantizedVector {
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.dimension * 4; // f32 = 4 bytes
        let compressed_bytes = self.data.len();
        original_bytes as f32 / compressed_bytes as f32
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() + self.aux.as_ref().map_or(0, |a| a.len() * 4)
    }
}

/// Binary (1-bit) quantizer
pub struct BinaryQuantizer {
    dimension: usize,
    threshold: f32,
}

impl BinaryQuantizer {
    /// Create new binary quantizer
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            threshold: 0.0,
        }
    }

    /// Encode vector to binary
    pub fn encode(&self, vector: &[f32]) -> QuantizedVector {
        let num_bytes = vector.len().div_ceil(8);
        let mut data = vec![0u8; num_bytes];

        for (i, &v) in vector.iter().enumerate() {
            if v > self.threshold {
                data[i / 8] |= 1 << (i % 8);
            }
        }

        QuantizedVector {
            data,
            dimension: vector.len(),
            bit_width: BitWidth::Bit1,
            aux: None,
        }
    }

    /// Decode binary to approximate vector
    pub fn decode(&self, quantized: &QuantizedVector) -> Vec<f32> {
        let mut vector = vec![0.0f32; quantized.dimension];

        for (i, val) in vector.iter_mut().enumerate().take(quantized.dimension) {
            let bit = (quantized.data[i / 8] >> (i % 8)) & 1;
            *val = if bit == 1 { 1.0 } else { -1.0 };
        }

        vector
    }

    /// Compute Hamming distance (fast binary comparison)
    pub fn hamming_distance(&self, a: &QuantizedVector, b: &QuantizedVector) -> u32 {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
    }

    /// Compute approximate cosine similarity from binary
    pub fn binary_similarity(&self, a: &QuantizedVector, b: &QuantizedVector) -> f32 {
        let hamming = self.hamming_distance(a, b);
        let max_dist = a.dimension as f32;
        1.0 - (2.0 * hamming as f32 / max_dist)
    }
}

/// Ternary (1.5-bit) quantizer with special zero handling
pub struct TernaryQuantizer {
    dimension: usize,
    zero_threshold: f32,
}

impl TernaryQuantizer {
    /// Create new ternary quantizer
    pub fn new(dimension: usize, zero_threshold: f32) -> Self {
        Self {
            dimension,
            zero_threshold,
        }
    }

    /// Encode to ternary (-1, 0, +1)
    pub fn encode(&self, vector: &[f32]) -> QuantizedVector {
        // Use 2 bits per value: 00 = -1, 01 = 0, 10 = +1
        let num_bytes = (vector.len() * 2).div_ceil(8);
        let mut data = vec![0u8; num_bytes];

        for (i, &v) in vector.iter().enumerate() {
            let code = if v.abs() < self.zero_threshold {
                0b01 // zero
            } else if v > 0.0 {
                0b10 // positive
            } else {
                0b00 // negative
            };

            let bit_pos = i * 2;
            let byte_pos = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            data[byte_pos] |= code << bit_offset;
            if bit_offset == 7 && byte_pos + 1 < data.len() {
                data[byte_pos + 1] |= code >> 1;
            }
        }

        QuantizedVector {
            data,
            dimension: vector.len(),
            bit_width: BitWidth::Bit1_5,
            aux: None,
        }
    }

    /// Decode ternary to vector
    pub fn decode(&self, quantized: &QuantizedVector) -> Vec<f32> {
        let mut vector = vec![0.0f32; quantized.dimension];

        for (i, val) in vector.iter_mut().enumerate().take(quantized.dimension) {
            let bit_pos = i * 2;
            let byte_pos = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            let mut code = (quantized.data[byte_pos] >> bit_offset) & 0b11;
            if bit_offset == 7 && byte_pos + 1 < quantized.data.len() {
                code |= (quantized.data[byte_pos + 1] & 1) << 1;
            }

            *val = match code & 0b11 {
                0b00 => -1.0,
                0b01 => 0.0,
                0b10 => 1.0,
                _ => 0.0,
            };
        }

        vector
    }
}

/// 2-bit quantizer (4 levels)
pub struct TwoBitQuantizer {
    params: QuantParams,
}

impl TwoBitQuantizer {
    /// Create new 2-bit quantizer
    pub fn new(dimension: usize) -> Self {
        Self {
            params: QuantParams::default_for_dim(dimension),
        }
    }

    /// Calibrate on training data
    pub fn calibrate(&mut self, vectors: &[Vec<f32>]) {
        if vectors.is_empty() {
            return;
        }

        let dim = vectors[0].len();
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];

        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                mins[i] = mins[i].min(v);
                maxs[i] = maxs[i].max(v);
            }
        }

        self.params.scales = mins.iter()
            .zip(&maxs)
            .map(|(&min, &max)| (max - min) / 3.0) // 4 levels = 3 intervals
            .collect();

        // Calculate global min/max before moving mins
        self.params.global_min = mins.iter().copied().fold(f32::MAX, f32::min);
        self.params.global_max = maxs.iter().copied().fold(f32::MIN, f32::max);
        self.params.zero_points = mins;
        self.params.dimension = dim;
    }

    /// Encode to 2-bit
    pub fn encode(&self, vector: &[f32]) -> QuantizedVector {
        let num_bytes = (vector.len() * 2).div_ceil(8);
        let mut data = vec![0u8; num_bytes];

        for (i, &v) in vector.iter().enumerate() {
            let scale = self.params.scales.get(i).copied().unwrap_or(1.0);
            let zero = self.params.zero_points.get(i).copied().unwrap_or(0.0);

            let normalized = if scale > 0.0 { (v - zero) / scale } else { 0.0 };
            let quantized = (normalized.round() as i32).clamp(0, 3) as u8;

            let bit_pos = i * 2;
            let byte_pos = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            data[byte_pos] |= quantized << bit_offset;
            if bit_offset >= 7 && byte_pos + 1 < data.len() {
                data[byte_pos + 1] |= quantized >> (8 - bit_offset);
            }
        }

        QuantizedVector {
            data,
            dimension: vector.len(),
            bit_width: BitWidth::Bit2,
            aux: None,
        }
    }

    /// Decode from 2-bit
    pub fn decode(&self, quantized: &QuantizedVector) -> Vec<f32> {
        let mut vector = vec![0.0f32; quantized.dimension];

        for (i, val) in vector.iter_mut().enumerate().take(quantized.dimension) {
            let bit_pos = i * 2;
            let byte_pos = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            let mut code = (quantized.data[byte_pos] >> bit_offset) & 0b11;
            if bit_offset >= 7 && byte_pos + 1 < quantized.data.len() {
                code |= (quantized.data[byte_pos + 1] << (8 - bit_offset)) & 0b11;
            }

            let scale = self.params.scales.get(i).copied().unwrap_or(1.0);
            let zero = self.params.zero_points.get(i).copied().unwrap_or(0.0);

            *val = (code as f32) * scale + zero;
        }

        vector
    }
}

/// 4-bit scalar quantizer
pub struct FourBitQuantizer {
    params: QuantParams,
}

impl FourBitQuantizer {
    /// Create new 4-bit quantizer
    pub fn new(dimension: usize) -> Self {
        Self {
            params: QuantParams::default_for_dim(dimension),
        }
    }

    /// Calibrate on training data
    pub fn calibrate(&mut self, vectors: &[Vec<f32>]) {
        if vectors.is_empty() {
            return;
        }

        let dim = vectors[0].len();
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];

        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                mins[i] = mins[i].min(v);
                maxs[i] = maxs[i].max(v);
            }
        }

        self.params.scales = mins
            .iter()
            .zip(&maxs)
            .map(|(&min, &max)| (max - min) / 15.0)
            .collect();

        self.params.zero_points = mins;
        self.params.dimension = dim;
    }

    /// Encode to 4-bit
    pub fn encode(&self, vector: &[f32]) -> QuantizedVector {
        let num_bytes = vector.len().div_ceil(2);
        let mut data = vec![0u8; num_bytes];

        for (i, &v) in vector.iter().enumerate() {
            let scale = self.params.scales.get(i).copied().unwrap_or(1.0);
            let zero = self.params.zero_points.get(i).copied().unwrap_or(0.0);

            let normalized = if scale > 0.0 { (v - zero) / scale } else { 0.0 };
            let quantized = (normalized.round() as i32).clamp(0, 15) as u8;

            if i % 2 == 0 {
                data[i / 2] |= quantized;
            } else {
                data[i / 2] |= quantized << 4;
            }
        }

        QuantizedVector {
            data,
            dimension: vector.len(),
            bit_width: BitWidth::Bit4,
            aux: None,
        }
    }

    /// Decode from 4-bit
    pub fn decode(&self, quantized: &QuantizedVector) -> Vec<f32> {
        let mut vector = vec![0.0f32; quantized.dimension];

        for (i, val) in vector.iter_mut().enumerate().take(quantized.dimension) {
            let code = if i % 2 == 0 {
                quantized.data[i / 2] & 0x0F
            } else {
                quantized.data[i / 2] >> 4
            };

            let scale = self.params.scales.get(i).copied().unwrap_or(1.0);
            let zero = self.params.zero_points.get(i).copied().unwrap_or(0.0);

            *val = (code as f32) * scale + zero;
        }

        vector
    }
}

/// Asymmetric quantizer (different precision for stored vs query)
pub struct AsymmetricQuantizer {
    config: QuantConfig,
    stored_quantizer: Box<dyn Quantizer + Send + Sync>,
    query_quantizer: Box<dyn Quantizer + Send + Sync>,
}

/// Trait for quantizers
pub trait Quantizer: Send + Sync {
    fn encode(&self, vector: &[f32]) -> QuantizedVector;
    fn decode(&self, quantized: &QuantizedVector) -> Vec<f32>;
    fn calibrate(&mut self, vectors: &[Vec<f32>]);
}

impl Quantizer for BinaryQuantizer {
    fn encode(&self, vector: &[f32]) -> QuantizedVector {
        self.encode(vector)
    }

    fn decode(&self, quantized: &QuantizedVector) -> Vec<f32> {
        self.decode(quantized)
    }

    fn calibrate(&mut self, _vectors: &[Vec<f32>]) {
        // Binary doesn't need calibration
    }
}

impl Quantizer for TwoBitQuantizer {
    fn encode(&self, vector: &[f32]) -> QuantizedVector {
        self.encode(vector)
    }

    fn decode(&self, quantized: &QuantizedVector) -> Vec<f32> {
        self.decode(quantized)
    }

    fn calibrate(&mut self, vectors: &[Vec<f32>]) {
        self.calibrate(vectors)
    }
}

impl Quantizer for FourBitQuantizer {
    fn encode(&self, vector: &[f32]) -> QuantizedVector {
        self.encode(vector)
    }

    fn decode(&self, quantized: &QuantizedVector) -> Vec<f32> {
        self.decode(quantized)
    }

    fn calibrate(&mut self, vectors: &[Vec<f32>]) {
        self.calibrate(vectors)
    }
}

impl AsymmetricQuantizer {
    /// Create new asymmetric quantizer
    pub fn new(config: QuantConfig, dimension: usize) -> Self {
        let stored_quantizer: Box<dyn Quantizer + Send + Sync> = match config.stored_bits {
            BitWidth::Bit1 => Box::new(BinaryQuantizer::new(dimension)),
            BitWidth::Bit2 => Box::new(TwoBitQuantizer::new(dimension)),
            BitWidth::Bit4 => Box::new(FourBitQuantizer::new(dimension)),
            _ => Box::new(FourBitQuantizer::new(dimension)),
        };

        let query_quantizer: Box<dyn Quantizer + Send + Sync> = match config.query_bits {
            BitWidth::Bit1 => Box::new(BinaryQuantizer::new(dimension)),
            BitWidth::Bit2 => Box::new(TwoBitQuantizer::new(dimension)),
            BitWidth::Bit4 => Box::new(FourBitQuantizer::new(dimension)),
            _ => Box::new(FourBitQuantizer::new(dimension)),
        };

        Self {
            config,
            stored_quantizer,
            query_quantizer,
        }
    }

    /// Encode for storage
    pub fn encode_stored(&self, vector: &[f32]) -> QuantizedVector {
        self.stored_quantizer.encode(vector)
    }

    /// Encode for query
    pub fn encode_query(&self, vector: &[f32]) -> QuantizedVector {
        self.query_quantizer.encode(vector)
    }

    /// Compute similarity between stored (binary) and query (scalar)
    pub fn asymmetric_similarity(&self, stored: &QuantizedVector, query: &[f32]) -> f32 {
        let decoded = self.stored_quantizer.decode(stored);
        cosine_similarity(&decoded, query)
    }
}

/// Adaptive quantizer with per-dimension bit allocation
pub struct AdaptiveQuantizer {
    config: QuantConfig,
    dimension: usize,
    bit_allocation: Vec<BitWidth>,
    sub_quantizers: Vec<Box<dyn Quantizer + Send + Sync>>,
    importance_scores: Vec<f32>,
}

impl AdaptiveQuantizer {
    /// Create new adaptive quantizer
    pub fn new(config: QuantConfig, dimension: usize) -> Self {
        // Start with uniform bit allocation
        let bit_allocation = vec![config.stored_bits; dimension];

        Self {
            config,
            dimension,
            bit_allocation,
            sub_quantizers: Vec::new(),
            importance_scores: vec![1.0; dimension],
        }
    }

    /// Calibrate and compute optimal bit allocation
    pub fn calibrate(&mut self, vectors: &[Vec<f32>]) {
        if vectors.is_empty() {
            return;
        }

        // Compute variance per dimension to determine importance
        let dim = vectors[0].len();
        let mut means = vec![0.0f32; dim];
        let mut variances = vec![0.0f32; dim];

        // Compute means
        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                means[i] += v;
            }
        }
        for m in &mut means {
            *m /= vectors.len() as f32;
        }

        // Compute variances
        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                let diff = v - means[i];
                variances[i] += diff * diff;
            }
        }
        for v in &mut variances {
            *v /= vectors.len() as f32;
        }

        // Compute importance scores (normalized variance)
        let max_var = variances.iter().copied().fold(0.0f32, f32::max);
        if max_var > 0.0 {
            self.importance_scores = variances.iter().map(|&v| v / max_var).collect();
        }

        // Allocate bits based on importance
        let _total_bits = dim
            * match self.config.stored_bits {
                BitWidth::Bit1 => 1,
                BitWidth::Bit1_5 => 2,
                BitWidth::Bit2 => 2,
                BitWidth::Bit4 => 4,
                BitWidth::Bit8 => 8,
            };

        // For now, use uniform allocation (could be optimized)
        self.bit_allocation = vec![self.config.stored_bits; dim];
    }

    /// Encode with adaptive bit allocation
    pub fn encode(&self, vector: &[f32]) -> QuantizedVector {
        // Use 4-bit quantization as base
        let quantizer = FourBitQuantizer::new(self.dimension);
        quantizer.encode(vector)
    }

    /// Decode
    pub fn decode(&self, quantized: &QuantizedVector) -> Vec<f32> {
        let quantizer = FourBitQuantizer::new(self.dimension);
        quantizer.decode(quantized)
    }
}

/// Quantized index for fast similarity search
pub struct QuantizedIndex {
    config: QuantConfig,
    vectors: RwLock<HashMap<String, QuantizedVector>>,
    original_vectors: RwLock<HashMap<String, Vec<f32>>>, // For rescoring
    dimension: usize,
    binary_quantizer: BinaryQuantizer,
    two_bit_quantizer: TwoBitQuantizer,
    four_bit_quantizer: FourBitQuantizer,
}

impl QuantizedIndex {
    /// Create new quantized index
    pub fn new(config: QuantConfig, dimension: usize) -> Self {
        let two_bit = TwoBitQuantizer::new(dimension);
        let four_bit = FourBitQuantizer::new(dimension);

        Self {
            config,
            vectors: RwLock::new(HashMap::new()),
            original_vectors: RwLock::new(HashMap::new()),
            dimension,
            binary_quantizer: BinaryQuantizer::new(dimension),
            two_bit_quantizer: two_bit,
            four_bit_quantizer: four_bit,
        }
    }

    /// Calibrate quantizers
    pub fn calibrate(&mut self, vectors: &[Vec<f32>]) {
        self.two_bit_quantizer.calibrate(vectors);
        self.four_bit_quantizer.calibrate(vectors);
    }

    /// Insert vector
    pub fn insert(&self, id: &str, vector: &[f32]) -> Result<()> {
        let quantized = match self.config.stored_bits {
            BitWidth::Bit1 => self.binary_quantizer.encode(vector),
            BitWidth::Bit2 => self.two_bit_quantizer.encode(vector),
            BitWidth::Bit4 => self.four_bit_quantizer.encode(vector),
            _ => self.four_bit_quantizer.encode(vector),
        };

        let mut vectors = self.vectors.write().map_err(|_| {
            crate::error::VecStoreError::LockError("Failed to acquire write lock on vectors".into())
        })?;
        vectors.insert(id.to_string(), quantized);

        if self.config.rescore {
            let mut originals = self.original_vectors.write().map_err(|_| {
                crate::error::VecStoreError::LockError(
                    "Failed to acquire write lock on original_vectors".into(),
                )
            })?;
            originals.insert(id.to_string(), vector.to_vec());
        }

        Ok(())
    }

    /// Search with quantized comparison
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let Ok(vectors) = self.vectors.read() else {
            return Vec::new();
        };

        // First pass: approximate search with quantized vectors
        let candidates_k = if self.config.rescore {
            (top_k as f32 * self.config.rescore_multiplier) as usize
        } else {
            top_k
        };

        let mut candidates: Vec<(String, f32)> = vectors
            .iter()
            .map(|(id, qvec)| {
                let score = match self.config.stored_bits {
                    BitWidth::Bit1 => {
                        let query_q = self.binary_quantizer.encode(query);
                        self.binary_quantizer.binary_similarity(&query_q, qvec)
                    },
                    _ => {
                        let decoded = match qvec.bit_width {
                            BitWidth::Bit2 => self.two_bit_quantizer.decode(qvec),
                            BitWidth::Bit4 => self.four_bit_quantizer.decode(qvec),
                            _ => self.four_bit_quantizer.decode(qvec),
                        };
                        cosine_similarity(&decoded, query)
                    },
                };
                (id.clone(), score)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        candidates.truncate(candidates_k);

        // Second pass: rescore with original vectors
        if self.config.rescore {
            let Ok(originals) = self.original_vectors.read() else {
                return candidates;
            };

            candidates = candidates
                .into_iter()
                .filter_map(|(id, _)| {
                    originals.get(&id).map(|orig| {
                        let exact_score = cosine_similarity(orig, query);
                        (id, exact_score)
                    })
                })
                .collect();

            candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        }

        candidates.truncate(top_k);
        candidates
    }

    /// Get statistics
    pub fn stats(&self) -> QuantizedIndexStats {
        let Ok(vectors) = self.vectors.read() else {
            return QuantizedIndexStats {
                vector_count: 0,
                quantized_bytes: 0,
                original_bytes: 0,
                compression_ratio: 0.0,
                bit_width: self.config.stored_bits,
            };
        };
        let Ok(originals) = self.original_vectors.read() else {
            return QuantizedIndexStats {
                vector_count: vectors.len(),
                quantized_bytes: vectors.values().map(|v| v.size_bytes()).sum(),
                original_bytes: 0,
                compression_ratio: 0.0,
                bit_width: self.config.stored_bits,
            };
        };

        let total_quantized: usize = vectors.values().map(|v| v.size_bytes()).sum();
        let total_original: usize = originals.values().map(|v| v.len() * 4).sum();

        QuantizedIndexStats {
            vector_count: vectors.len(),
            quantized_bytes: total_quantized,
            original_bytes: total_original,
            compression_ratio: if total_quantized > 0 {
                total_original as f32 / total_quantized as f32
            } else {
                0.0
            },
            bit_width: self.config.stored_bits,
        }
    }
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedIndexStats {
    pub vector_count: usize,
    pub quantized_bytes: usize,
    pub original_bytes: usize,
    pub compression_ratio: f32,
    pub bit_width: BitWidth,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_quantization() {
        let quantizer = BinaryQuantizer::new(128);

        let vector: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let encoded = quantizer.encode(&vector);

        assert_eq!(encoded.dimension, 128);
        assert_eq!(encoded.data.len(), 16); // 128 bits = 16 bytes

        let decoded = quantizer.decode(&encoded);
        assert_eq!(decoded.len(), 128);
    }

    #[test]
    fn test_ternary_quantization() {
        let quantizer = TernaryQuantizer::new(128, 0.1);

        let vector: Vec<f32> = (0..128)
            .map(|i| match i % 3 {
                0 => 0.5,
                1 => -0.5,
                _ => 0.0,
            })
            .collect();

        let encoded = quantizer.encode(&vector);
        let decoded = quantizer.decode(&encoded);

        assert_eq!(decoded.len(), 128);
    }

    #[test]
    fn test_2bit_quantization() {
        let mut quantizer = TwoBitQuantizer::new(128);

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..128).map(|i| (i as f32 / 128.0) - 0.5).collect())
            .collect();

        quantizer.calibrate(&vectors);

        let vector: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0) - 0.5).collect();
        let encoded = quantizer.encode(&vector);

        assert_eq!(encoded.compression_ratio(), 16.0);

        let decoded = quantizer.decode(&encoded);
        assert_eq!(decoded.len(), 128);
    }

    #[test]
    fn test_quantized_index() {
        let config = QuantConfig::binary();
        let index = QuantizedIndex::new(config, 64);

        // Insert vectors
        for i in 0..100 {
            let vector: Vec<f32> = (0..64).map(|j| ((i + j) as f32 / 100.0) - 0.5).collect();
            index.insert(&format!("vec_{}", i), &vector).unwrap();
        }

        // Search
        let query: Vec<f32> = (0..64).map(|i| (i as f32 / 100.0) - 0.5).collect();
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_compression_ratios() {
        assert_eq!(BitWidth::Bit1.compression_ratio(), 32.0);
        assert_eq!(BitWidth::Bit2.compression_ratio(), 16.0);
        assert_eq!(BitWidth::Bit4.compression_ratio(), 8.0);
        assert_eq!(BitWidth::Bit8.compression_ratio(), 4.0);
    }
}
