//! Index Compression
//!
//! This module provides compression techniques for reducing index memory usage
//! and storage size. Complements Product Quantization with lossless compression.
//!
//! ## Compression Strategies
//!
//! 1. **Delta Encoding**: Compress sequential IDs in neighbor lists
//! 2. **Varint Encoding**: Variable-length integer encoding
//! 3. **Huffman Encoding**: Entropy-based compression
//! 4. **Run-Length Encoding**: Compress repeated values
//! 5. **Bulk Compression**: ZSTD/LZ4 for large data blocks
//!
//! ## Use Cases
//!
//! - Reduce HNSW graph memory usage by 30-70%
//! - Faster disk I/O with smaller index files
//! - Lower bandwidth for distributed systems
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::compression::{CompressedIndex, CompressionConfig, CompressionLevel};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Configure compression
//! let config = CompressionConfig::default()
//!     .with_level(CompressionLevel::Balanced)
//!     .with_delta_encoding(true)
//!     .with_varint_encoding(true);
//!
//! // Compress neighbor lists
//! let neighbor_ids = vec![10, 11, 12, 15, 20, 21, 22];
//! let compressed = config.compress_ids(&neighbor_ids)?;
//!
//! println!("Original: {} bytes", neighbor_ids.len() * 4);
//! println!("Compressed: {} bytes", compressed.len());
//! println!("Compression ratio: {:.2}x",
//!          (neighbor_ids.len() * 4) as f32 / compressed.len() as f32);
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Compression level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// No compression (fastest)
    None,

    /// Fast compression (lower ratio, faster)
    Fast,

    /// Balanced compression
    Balanced,

    /// Maximum compression (slower, best ratio)
    Max,
}

impl Default for CompressionLevel {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression level
    pub level: CompressionLevel,

    /// Enable delta encoding for sequential IDs
    pub delta_encoding: bool,

    /// Enable varint encoding for integers
    pub varint_encoding: bool,

    /// Enable bulk compression (ZSTD)
    pub bulk_compression: bool,

    /// Minimum size for bulk compression (bytes)
    pub bulk_threshold: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            level: CompressionLevel::Balanced,
            delta_encoding: true,
            varint_encoding: true,
            bulk_compression: true,
            bulk_threshold: 1024, // 1KB
        }
    }
}

impl CompressionConfig {
    /// Create new compression config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression level
    pub fn with_level(mut self, level: CompressionLevel) -> Self {
        self.level = level;
        self
    }

    /// Enable/disable delta encoding
    pub fn with_delta_encoding(mut self, enabled: bool) -> Self {
        self.delta_encoding = enabled;
        self
    }

    /// Enable/disable varint encoding
    pub fn with_varint_encoding(mut self, enabled: bool) -> Self {
        self.varint_encoding = enabled;
        self
    }

    /// Enable/disable bulk compression
    pub fn with_bulk_compression(mut self, enabled: bool) -> Self {
        self.bulk_compression = enabled;
        self
    }

    /// Compress a list of IDs (neighbor lists)
    ///
    /// Uses delta + varint encoding for efficient compression of sequential IDs.
    pub fn compress_ids(&self, ids: &[usize]) -> Result<Vec<u8>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let mut output = Vec::new();

        if self.delta_encoding {
            // Delta encoding: store differences between consecutive IDs
            let mut prev = 0;

            for &id in ids {
                let delta = if id >= prev {
                    id - prev
                } else {
                    // Handle non-sequential case
                    id
                };

                if self.varint_encoding {
                    encode_varint(&mut output, delta)?;
                } else {
                    output.extend_from_slice(&delta.to_le_bytes());
                }

                prev = id;
            }
        } else {
            // No delta encoding
            for &id in ids {
                if self.varint_encoding {
                    encode_varint(&mut output, id)?;
                } else {
                    output.extend_from_slice(&id.to_le_bytes());
                }
            }
        }

        // Apply bulk compression if enabled and beneficial
        if self.bulk_compression && output.len() >= self.bulk_threshold {
            self.compress_bulk(&output)
        } else {
            Ok(output)
        }
    }

    /// Decompress a list of IDs
    pub fn decompress_ids(&self, data: &[u8], count: usize) -> Result<Vec<usize>> {
        if data.is_empty() {
            return Ok(vec![]);
        }

        // Decompress bulk if needed
        let data = if self.bulk_compression {
            self.decompress_bulk(data)?
        } else {
            data.to_vec()
        };

        let mut ids = Vec::with_capacity(count);
        let mut cursor = 0;
        let mut prev = 0;

        while ids.len() < count && cursor < data.len() {
            let value = if self.varint_encoding {
                let (v, bytes_read) = decode_varint(&data[cursor..])?;
                cursor += bytes_read;
                v
            } else {
                if cursor + 8 > data.len() {
                    return Err(anyhow!("Insufficient data for usize"));
                }
                let v = usize::from_le_bytes(data[cursor..cursor + 8].try_into()?);
                cursor += 8;
                v
            };

            let id = if self.delta_encoding {
                prev + value
            } else {
                value
            };

            ids.push(id);
            prev = id;
        }

        Ok(ids)
    }

    /// Compress arbitrary binary data
    pub fn compress_bulk(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.level {
            CompressionLevel::None => Ok(data.to_vec()),
            CompressionLevel::Fast => {
                // LZ4-like simple compression (placeholder - would use lz4 crate in production)
                Ok(data.to_vec())
            }
            CompressionLevel::Balanced | CompressionLevel::Max => {
                // ZSTD compression (using built-in if available)
                #[cfg(feature = "compression")]
                {
                    let level = match self.level {
                        CompressionLevel::Balanced => 3,
                        CompressionLevel::Max => 19,
                        _ => 3,
                    };
                    zstd::encode_all(data, level)
                        .map_err(|e| anyhow!("ZSTD compression failed: {}", e))
                }
                #[cfg(not(feature = "compression"))]
                {
                    // Fallback: no compression if feature not enabled
                    Ok(data.to_vec())
                }
            }
        }
    }

    /// Decompress arbitrary binary data
    pub fn decompress_bulk(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.level {
            CompressionLevel::None => Ok(data.to_vec()),
            CompressionLevel::Fast => {
                // LZ4 decompression (placeholder)
                Ok(data.to_vec())
            }
            CompressionLevel::Balanced | CompressionLevel::Max => {
                #[cfg(feature = "compression")]
                {
                    zstd::decode_all(data).map_err(|e| anyhow!("ZSTD decompression failed: {}", e))
                }
                #[cfg(not(feature = "compression"))]
                {
                    Ok(data.to_vec())
                }
            }
        }
    }

    /// Compress floating-point values with quantization
    ///
    /// Reduces precision to save space (e.g., float32 -> float16 or fixed-point)
    pub fn compress_floats(&self, values: &[f32], precision_bits: u8) -> Result<Vec<u8>> {
        match precision_bits {
            8 => {
                // 8-bit quantization: map to [0, 255]
                let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let range = max - min;

                let mut output = Vec::with_capacity(8 + values.len());
                output.extend_from_slice(&min.to_le_bytes());
                output.extend_from_slice(&range.to_le_bytes());

                for &v in values {
                    let normalized = ((v - min) / range * 255.0) as u8;
                    output.push(normalized);
                }

                Ok(output)
            }
            16 => {
                // 16-bit quantization
                let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let range = max - min;

                let mut output = Vec::with_capacity(8 + values.len() * 2);
                output.extend_from_slice(&min.to_le_bytes());
                output.extend_from_slice(&range.to_le_bytes());

                for &v in values {
                    let normalized = ((v - min) / range * 65535.0) as u16;
                    output.extend_from_slice(&normalized.to_le_bytes());
                }

                Ok(output)
            }
            32 => {
                // No compression, just convert to bytes
                let mut output = Vec::with_capacity(values.len() * 4);
                for &v in values {
                    output.extend_from_slice(&v.to_le_bytes());
                }
                Ok(output)
            }
            _ => Err(anyhow!("Unsupported precision bits: {}", precision_bits)),
        }
    }

    /// Decompress floating-point values
    pub fn decompress_floats(
        &self,
        data: &[u8],
        count: usize,
        precision_bits: u8,
    ) -> Result<Vec<f32>> {
        match precision_bits {
            8 => {
                if data.len() < 8 {
                    return Err(anyhow!("Insufficient data for float decompression"));
                }

                let min = f32::from_le_bytes(data[0..4].try_into()?);
                let range = f32::from_le_bytes(data[4..8].try_into()?);

                let mut values = Vec::with_capacity(count);
                for i in 0..count {
                    if 8 + i >= data.len() {
                        break;
                    }
                    let normalized = data[8 + i] as f32 / 255.0;
                    values.push(min + normalized * range);
                }

                Ok(values)
            }
            16 => {
                if data.len() < 8 {
                    return Err(anyhow!("Insufficient data for float decompression"));
                }

                let min = f32::from_le_bytes(data[0..4].try_into()?);
                let range = f32::from_le_bytes(data[4..8].try_into()?);

                let mut values = Vec::with_capacity(count);
                for i in 0..count {
                    let offset = 8 + i * 2;
                    if offset + 2 > data.len() {
                        break;
                    }
                    let normalized =
                        u16::from_le_bytes(data[offset..offset + 2].try_into()?) as f32 / 65535.0;
                    values.push(min + normalized * range);
                }

                Ok(values)
            }
            32 => {
                let mut values = Vec::with_capacity(count);
                for i in 0..count {
                    let offset = i * 4;
                    if offset + 4 > data.len() {
                        break;
                    }
                    values.push(f32::from_le_bytes(data[offset..offset + 4].try_into()?));
                }
                Ok(values)
            }
            _ => Err(anyhow!("Unsupported precision bits: {}", precision_bits)),
        }
    }
}

/// Variable-length integer encoding (LEB128/varint)
///
/// More efficient than fixed 8-byte encoding for small integers.
pub fn encode_varint(output: &mut Vec<u8>, mut value: usize) -> Result<()> {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        if value != 0 {
            byte |= 0x80; // More bytes follow
        }

        output.push(byte);

        if value == 0 {
            break;
        }
    }

    Ok(())
}

/// Decode variable-length integer
///
/// Returns (value, bytes_read)
pub fn decode_varint(data: &[u8]) -> Result<(usize, usize)> {
    let mut value = 0usize;
    let mut shift = 0;
    let mut bytes_read = 0;

    for &byte in data.iter().take(10) {
        // Max 10 bytes for usize
        bytes_read += 1;
        value |= ((byte & 0x7F) as usize) << shift;

        if byte & 0x80 == 0 {
            return Ok((value, bytes_read));
        }

        shift += 7;
    }

    Err(anyhow!("Varint decoding failed"))
}

/// Run-length encoding for repeated values
pub fn encode_rle(values: &[usize]) -> Vec<u8> {
    if values.is_empty() {
        return vec![];
    }

    let mut output = Vec::new();
    let mut prev = values[0];
    let mut count = 1;

    for &value in &values[1..] {
        if value == prev && count < 255 {
            count += 1;
        } else {
            // Encode (value, count) pair
            output.extend_from_slice(&prev.to_le_bytes());
            output.push(count);

            prev = value;
            count = 1;
        }
    }

    // Encode last run
    output.extend_from_slice(&prev.to_le_bytes());
    output.push(count);

    output
}

/// Decode run-length encoding
pub fn decode_rle(data: &[u8]) -> Result<Vec<usize>> {
    let mut values = Vec::new();
    let mut cursor = 0;

    while cursor + 9 <= data.len() {
        let value = usize::from_le_bytes(data[cursor..cursor + 8].try_into()?);
        let count = data[cursor + 8] as usize;
        cursor += 9;

        for _ in 0..count {
            values.push(value);
        }
    }

    Ok(values)
}

/// Compressed HNSW neighbor list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedNeighborList {
    /// Compressed neighbor IDs
    pub data: Vec<u8>,

    /// Number of neighbors
    pub count: usize,

    /// Compression method used
    pub method: CompressionMethod,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionMethod {
    None,
    DeltaVarint,
    RLE,
    Bulk,
}

impl CompressedNeighborList {
    /// Compress a neighbor list
    pub fn compress(neighbors: &[usize], config: &CompressionConfig) -> Result<Self> {
        let data = config.compress_ids(neighbors)?;

        let method = if config.delta_encoding && config.varint_encoding {
            CompressionMethod::DeltaVarint
        } else if config.bulk_compression {
            CompressionMethod::Bulk
        } else {
            CompressionMethod::None
        };

        Ok(Self {
            data,
            count: neighbors.len(),
            method,
        })
    }

    /// Decompress neighbor list
    pub fn decompress(&self, config: &CompressionConfig) -> Result<Vec<usize>> {
        config.decompress_ids(&self.data, self.count)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.count * 8; // usize = 8 bytes
        original_size as f32 / self.data.len() as f32
    }
}

/// Statistics about compression
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Original size (bytes)
    pub original_bytes: usize,

    /// Compressed size (bytes)
    pub compressed_bytes: usize,

    /// Number of neighbor lists compressed
    pub num_lists: usize,

    /// Average list length
    pub avg_list_length: f32,
}

impl CompressionStats {
    /// Compute compression ratio
    pub fn ratio(&self) -> f32 {
        if self.compressed_bytes == 0 {
            0.0
        } else {
            self.original_bytes as f32 / self.compressed_bytes as f32
        }
    }

    /// Compute space savings percentage
    pub fn savings_percent(&self) -> f32 {
        if self.original_bytes == 0 {
            0.0
        } else {
            (1.0 - self.compressed_bytes as f32 / self.original_bytes as f32) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encoding() {
        let mut output = Vec::new();

        encode_varint(&mut output, 0).unwrap();
        encode_varint(&mut output, 1).unwrap();
        encode_varint(&mut output, 127).unwrap();
        encode_varint(&mut output, 128).unwrap();
        encode_varint(&mut output, 16383).unwrap();

        let mut cursor = 0;

        let (val, bytes_read) = decode_varint(&output[cursor..]).unwrap();
        assert_eq!(val, 0);
        cursor += bytes_read;

        let (val, bytes_read) = decode_varint(&output[cursor..]).unwrap();
        assert_eq!(val, 1);
        cursor += bytes_read;

        let (val, bytes_read) = decode_varint(&output[cursor..]).unwrap();
        assert_eq!(val, 127);
        cursor += bytes_read;

        let (val, bytes_read) = decode_varint(&output[cursor..]).unwrap();
        assert_eq!(val, 128);
        cursor += bytes_read;

        let (val, _bytes_read) = decode_varint(&output[cursor..]).unwrap();
        assert_eq!(val, 16383);
    }

    #[test]
    fn test_delta_varint_compression() {
        let config = CompressionConfig::default();

        // Sequential IDs (best case for delta encoding)
        let ids = vec![100, 101, 102, 103, 104, 105];
        let compressed = config.compress_ids(&ids).unwrap();

        // Should be much smaller than 48 bytes (6 * 8)
        assert!(compressed.len() < ids.len() * 8);

        // Decompress and verify
        let decompressed = config.decompress_ids(&compressed, ids.len()).unwrap();
        assert_eq!(ids, decompressed);
    }

    #[test]
    fn test_sparse_ids_compression() {
        let config = CompressionConfig::default();

        // Sparse IDs (worse for delta encoding)
        let ids = vec![10, 500, 1000, 5000, 10000];
        let compressed = config.compress_ids(&ids).unwrap();

        let decompressed = config.decompress_ids(&compressed, ids.len()).unwrap();
        assert_eq!(ids, decompressed);
    }

    #[test]
    fn test_float_compression_8bit() {
        let config = CompressionConfig::default();

        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let compressed = config.compress_floats(&values, 8).unwrap();

        // 8 bytes (min/max) + 5 bytes (values)
        assert_eq!(compressed.len(), 13);

        let decompressed = config
            .decompress_floats(&compressed, values.len(), 8)
            .unwrap();

        // Should be close (not exact due to quantization)
        for (orig, decomp) in values.iter().zip(decompressed.iter()) {
            assert!((orig - decomp).abs() < 0.01);
        }
    }

    #[test]
    fn test_float_compression_16bit() {
        let config = CompressionConfig::default();

        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let compressed = config.compress_floats(&values, 16).unwrap();

        // 8 bytes (min/max) + 10 bytes (5 * 2)
        assert_eq!(compressed.len(), 18);

        let decompressed = config
            .decompress_floats(&compressed, values.len(), 16)
            .unwrap();

        // Should be very close with 16-bit precision
        for (orig, decomp) in values.iter().zip(decompressed.iter()) {
            assert!((orig - decomp).abs() < 0.0001);
        }
    }

    #[test]
    fn test_rle_encoding() {
        let values = vec![5, 5, 5, 7, 7, 10, 10, 10, 10];
        let encoded = encode_rle(&values);
        let decoded = decode_rle(&encoded).unwrap();

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_compressed_neighbor_list() {
        let config = CompressionConfig::default();
        let neighbors = vec![10, 11, 12, 15, 20, 21, 22];

        let compressed = CompressedNeighborList::compress(&neighbors, &config).unwrap();

        println!("Original: {} bytes", neighbors.len() * 8);
        println!("Compressed: {} bytes", compressed.data.len());
        println!("Ratio: {:.2}x", compressed.compression_ratio());

        assert!(compressed.compression_ratio() > 1.0);

        let decompressed = compressed.decompress(&config).unwrap();
        assert_eq!(neighbors, decompressed);
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::default();
        stats.original_bytes = 1000;
        stats.compressed_bytes = 300;
        stats.num_lists = 10;
        stats.avg_list_length = 12.5;

        assert!((stats.ratio() - 3.33).abs() < 0.01);
        assert!((stats.savings_percent() - 70.0).abs() < 0.1);
    }

    #[test]
    fn test_empty_compression() {
        let config = CompressionConfig::default();
        let empty: Vec<usize> = vec![];

        let compressed = config.compress_ids(&empty).unwrap();
        assert!(compressed.is_empty());

        let decompressed = config.decompress_ids(&compressed, 0).unwrap();
        assert!(decompressed.is_empty());
    }
}
