// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! Columnar Storage for Vector Data
//!
//! This module provides columnar (column-oriented) storage for vectors, enabling:
//! - **Better Compression**: Similar values cluster together, improving compression ratios
//! - **Faster Analytics**: Column scans are cache-efficient for aggregations
//! - **Dimension Selection**: Read only needed dimensions for partial vector operations
//! - **SIMD-Friendly**: Contiguous dimension data enables efficient vectorization
//! - **Memory Mapping**: Efficient loading of specific columns from disk
//!
//! # Storage Layout
//!
//! Traditional row-based: `[v0d0, v0d1, v0d2, v1d0, v1d1, v1d2, ...]`
//! Columnar layout:       `[v0d0, v1d0, v2d0, ...], [v0d1, v1d1, v2d1, ...], ...`
//!
//! # Example
//!
//! ```ignore
//! use vecstore::columnar::{ColumnarStore, ColumnarConfig, CompressionType};
//!
//! let config = ColumnarConfig {
//!     compression: CompressionType::Lz4,
//!     chunk_size: 4096,
//!     ..Default::default()
//! };
//!
//! let mut store = ColumnarStore::new(128, config)?;
//!
//! // Add vectors (automatically stored in columnar format)
//! store.add("vec1", &[0.1, 0.2, 0.3, ...])?;
//!
//! // Read specific dimensions only (efficient for partial distance)
//! let partial = store.read_dimensions("vec1", &[0, 1, 5, 10])?;
//!
//! // Column-level analytics
//! let stats = store.column_stats(0)?;
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

use crate::error::VecStoreError;

/// Compression type for columnar data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum CompressionType {
    /// No compression
    #[default]
    None,
    /// LZ4 fast compression
    Lz4,
    /// Zstandard compression (better ratio)
    Zstd,
    /// Delta encoding (for sorted data)
    Delta,
    /// Run-length encoding (for sparse data)
    Rle,
    /// Dictionary encoding (for repeated values)
    Dictionary,
}


/// Configuration for columnar storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnarConfig {
    /// Compression type
    pub compression: CompressionType,
    /// Number of vectors per chunk
    pub chunk_size: usize,
    /// Enable memory mapping for disk access
    pub use_mmap: bool,
    /// Pre-compute column statistics
    pub compute_stats: bool,
    /// Quantization bits per value (0 = no quantization)
    pub quantization_bits: u8,
    /// Stripe size for parallel I/O
    pub stripe_size: usize,
}

impl Default for ColumnarConfig {
    fn default() -> Self {
        Self {
            compression: CompressionType::None,
            chunk_size: 4096,
            use_mmap: true,
            compute_stats: true,
            quantization_bits: 0,
            stripe_size: 65536,
        }
    }
}

/// Statistics for a single column (dimension)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    /// Dimension index
    pub dimension: usize,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f64,
    /// Variance
    pub variance: f64,
    /// Number of non-zero values
    pub non_zero_count: usize,
    /// Number of values
    pub count: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

/// A chunk of columnar data (fixed number of vectors)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnarChunk {
    /// Chunk index
    pub index: usize,
    /// Start vector ID
    pub start_id: usize,
    /// Number of vectors in this chunk
    pub count: usize,
    /// Column data (one Vec<f32> per dimension)
    pub columns: Vec<Vec<f32>>,
    /// Compressed column data (if compression enabled)
    pub compressed_columns: Option<Vec<Vec<u8>>>,
    /// Per-column statistics
    pub stats: Option<Vec<ColumnStats>>,
}

impl ColumnarChunk {
    /// Create a new chunk
    pub fn new(index: usize, start_id: usize, dimensions: usize, capacity: usize) -> Self {
        let columns = (0..dimensions)
            .map(|_| Vec::with_capacity(capacity))
            .collect();

        Self {
            index,
            start_id,
            count: 0,
            columns,
            compressed_columns: None,
            stats: None,
        }
    }

    /// Add a vector to this chunk
    pub fn add_vector(&mut self, vector: &[f32]) -> Result<(), VecStoreError> {
        if vector.len() != self.columns.len() {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.columns.len(),
                got: vector.len(),
            });
        }

        for (col_idx, &value) in vector.iter().enumerate() {
            self.columns[col_idx].push(value);
        }

        self.count += 1;
        Ok(())
    }

    /// Get a vector from this chunk
    pub fn get_vector(&self, local_idx: usize) -> Result<Vec<f32>, VecStoreError> {
        if local_idx >= self.count {
            return Err(VecStoreError::NotFound(format!(
                "Vector index {} out of range", local_idx
            )));
        }

        let vector: Vec<f32> = self.columns
            .iter()
            .map(|col| col[local_idx])
            .collect();

        Ok(vector)
    }

    /// Get specific dimensions for a vector
    pub fn get_dimensions(&self, local_idx: usize, dims: &[usize]) -> Result<Vec<f32>, VecStoreError> {
        if local_idx >= self.count {
            return Err(VecStoreError::NotFound(format!(
                "Vector index {} out of range", local_idx
            )));
        }

        let values: Vec<f32> = dims.iter()
            .filter_map(|&d| {
                if d < self.columns.len() {
                    Some(self.columns[d][local_idx])
                } else {
                    None
                }
            })
            .collect();

        Ok(values)
    }

    /// Compute statistics for all columns
    pub fn compute_stats(&mut self) {
        let stats: Vec<ColumnStats> = self.columns
            .iter()
            .enumerate()
            .map(|(dim, col)| Self::compute_column_stats(dim, col))
            .collect();

        self.stats = Some(stats);
    }

    fn compute_column_stats(dimension: usize, column: &[f32]) -> ColumnStats {
        if column.is_empty() {
            return ColumnStats {
                dimension,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                variance: 0.0,
                non_zero_count: 0,
                count: 0,
                compression_ratio: 1.0,
            };
        }

        let min = column.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = column.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = column.iter().map(|&x| x as f64).sum();
        let mean = sum / column.len() as f64;

        let variance: f64 = column.iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / column.len() as f64;

        let non_zero_count = column.iter().filter(|&&x| x != 0.0).count();

        ColumnStats {
            dimension,
            min,
            max,
            mean,
            variance,
            non_zero_count,
            count: column.len(),
            compression_ratio: 1.0,
        }
    }

    /// Compress the chunk data
    pub fn compress(&mut self, compression: CompressionType) -> Result<(), VecStoreError> {
        match compression {
            CompressionType::None => {
                self.compressed_columns = None;
                Ok(())
            }
            CompressionType::Delta => {
                let compressed: Vec<Vec<u8>> = self.columns
                    .iter()
                    .map(|col| Self::delta_encode(col))
                    .collect();
                self.compressed_columns = Some(compressed);
                Ok(())
            }
            CompressionType::Rle => {
                let compressed: Vec<Vec<u8>> = self.columns
                    .iter()
                    .map(|col| Self::rle_encode(col))
                    .collect();
                self.compressed_columns = Some(compressed);
                Ok(())
            }
            _ => {
                // LZ4, Zstd would require external crates
                // For now, store raw bytes
                let compressed: Vec<Vec<u8>> = self.columns
                    .iter()
                    .map(|col| {
                        let bytes: Vec<u8> = col.iter()
                            .flat_map(|&f| f.to_le_bytes())
                            .collect();
                        bytes
                    })
                    .collect();
                self.compressed_columns = Some(compressed);
                Ok(())
            }
        }
    }

    fn delta_encode(column: &[f32]) -> Vec<u8> {
        if column.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(column.len() * 4);

        // First value as-is
        result.extend_from_slice(&column[0].to_le_bytes());

        // Delta values
        for i in 1..column.len() {
            let delta = column[i] - column[i - 1];
            result.extend_from_slice(&delta.to_le_bytes());
        }

        result
    }

    fn rle_encode(column: &[f32]) -> Vec<u8> {
        if column.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current = column[0];
        let mut count: u32 = 1;

        for &value in &column[1..] {
            if value == current && count < u32::MAX {
                count += 1;
            } else {
                result.extend_from_slice(&count.to_le_bytes());
                result.extend_from_slice(&current.to_le_bytes());
                current = value;
                count = 1;
            }
        }

        // Final run
        result.extend_from_slice(&count.to_le_bytes());
        result.extend_from_slice(&current.to_le_bytes());

        result
    }
}

/// Main columnar storage engine
pub struct ColumnarStore {
    /// Configuration
    config: ColumnarConfig,
    /// Vector dimensions
    dimensions: usize,
    /// Storage path (if persisted)
    path: Option<PathBuf>,
    /// Chunks
    chunks: Vec<ColumnarChunk>,
    /// ID to (chunk_index, local_index) mapping
    id_map: HashMap<String, (usize, usize)>,
    /// Total vector count
    count: usize,
    /// Current active chunk (being written to)
    active_chunk: Option<usize>,
    /// Global column statistics
    global_stats: Option<Vec<ColumnStats>>,
}

impl ColumnarStore {
    /// Create a new in-memory columnar store
    pub fn new(dimensions: usize, config: ColumnarConfig) -> Result<Self, VecStoreError> {
        if dimensions == 0 {
            return Err(VecStoreError::InvalidInput("Dimensions must be > 0".into()));
        }

        Ok(Self {
            config,
            dimensions,
            path: None,
            chunks: Vec::new(),
            id_map: HashMap::new(),
            count: 0,
            active_chunk: None,
            global_stats: None,
        })
    }

    /// Open or create a columnar store at path
    pub fn open(path: impl AsRef<Path>, dimensions: usize, config: ColumnarConfig) -> Result<Self, VecStoreError> {
        let path = path.as_ref().to_path_buf();

        if path.exists() {
            Self::load(&path, config)
        } else {
            let mut store = Self::new(dimensions, config)?;
            store.path = Some(path);
            Ok(store)
        }
    }

    /// Load from disk
    fn load(path: &Path, config: ColumnarConfig) -> Result<Self, VecStoreError> {
        let meta_path = path.join("columnar_meta.json");

        let file = std::fs::File::open(&meta_path)
            .map_err(VecStoreError::Io)?;

        let meta: ColumnarMeta = serde_json::from_reader(file)
            .map_err(|e| VecStoreError::Serialization(e.to_string()))?;

        let mut store = Self {
            config,
            dimensions: meta.dimensions,
            path: Some(path.to_path_buf()),
            chunks: Vec::new(),
            id_map: meta.id_map,
            count: meta.count,
            active_chunk: None,
            global_stats: None,
        };

        // Load chunks
        for i in 0..meta.chunk_count {
            let chunk_path = path.join(format!("chunk_{}.bin", i));
            let chunk = store.load_chunk(&chunk_path)?;
            store.chunks.push(chunk);
        }

        Ok(store)
    }

    fn load_chunk(&self, path: &Path) -> Result<ColumnarChunk, VecStoreError> {
        let file = std::fs::File::open(path)
            .map_err(VecStoreError::Io)?;

        let chunk: ColumnarChunk = bincode::deserialize_from(file)
            .map_err(|e| VecStoreError::Serialization(e.to_string()))?;

        Ok(chunk)
    }

    /// Add a vector
    pub fn add(&mut self, id: impl Into<String>, vector: &[f32]) -> Result<(), VecStoreError> {
        if vector.len() != self.dimensions {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        let id = id.into();

        // Get or create active chunk
        let chunk_idx = self.get_or_create_active_chunk();

        // Add to chunk
        let chunk = &mut self.chunks[chunk_idx];
        let local_idx = chunk.count;
        chunk.add_vector(vector)?;

        // Update mapping
        self.id_map.insert(id, (chunk_idx, local_idx));
        self.count += 1;

        // Check if chunk is full
        if chunk.count >= self.config.chunk_size {
            if self.config.compute_stats {
                chunk.compute_stats();
            }
            if self.config.compression != CompressionType::None {
                chunk.compress(self.config.compression)?;
            }
            self.active_chunk = None;
        }

        Ok(())
    }

    fn get_or_create_active_chunk(&mut self) -> usize {
        if let Some(idx) = self.active_chunk {
            idx
        } else {
            let idx = self.chunks.len();
            let start_id = self.count;
            let chunk = ColumnarChunk::new(idx, start_id, self.dimensions, self.config.chunk_size);
            self.chunks.push(chunk);
            self.active_chunk = Some(idx);
            idx
        }
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Result<Vec<f32>, VecStoreError> {
        let (chunk_idx, local_idx) = self.id_map.get(id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Vector {} not found", id)))?;

        self.chunks[*chunk_idx].get_vector(*local_idx)
    }

    /// Read specific dimensions for a vector
    pub fn read_dimensions(&self, id: &str, dims: &[usize]) -> Result<Vec<f32>, VecStoreError> {
        let (chunk_idx, local_idx) = self.id_map.get(id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Vector {} not found", id)))?;

        self.chunks[*chunk_idx].get_dimensions(*local_idx, dims)
    }

    /// Get all values for a specific dimension across all vectors
    pub fn read_column(&self, dimension: usize) -> Result<Vec<f32>, VecStoreError> {
        if dimension >= self.dimensions {
            return Err(VecStoreError::InvalidInput(format!(
                "Dimension {} out of range (max {})", dimension, self.dimensions
            )));
        }

        let mut result = Vec::with_capacity(self.count);

        for chunk in &self.chunks {
            result.extend_from_slice(&chunk.columns[dimension]);
        }

        Ok(result)
    }

    /// Get statistics for a specific column
    pub fn column_stats(&self, dimension: usize) -> Result<ColumnStats, VecStoreError> {
        if dimension >= self.dimensions {
            return Err(VecStoreError::InvalidInput(format!(
                "Dimension {} out of range", dimension
            )));
        }

        // Read all values for this dimension
        let values = self.read_column(dimension)?;
        Ok(ColumnarChunk::compute_column_stats(dimension, &values))
    }

    /// Get global statistics (across all dimensions)
    pub fn global_stats(&mut self) -> Result<&[ColumnStats], VecStoreError> {
        if self.global_stats.is_none() {
            let stats: Result<Vec<ColumnStats>, _> = (0..self.dimensions)
                .map(|d| self.column_stats(d))
                .collect();
            self.global_stats = Some(stats?);
        }

        Ok(self.global_stats.as_ref().unwrap())
    }

    /// Save to disk
    pub fn save(&self) -> Result<(), VecStoreError> {
        let path = self.path.as_ref()
            .ok_or_else(|| VecStoreError::InvalidInput("No path set for columnar store".into()))?;

        std::fs::create_dir_all(path).map_err(VecStoreError::Io)?;

        // Save metadata
        let meta = ColumnarMeta {
            dimensions: self.dimensions,
            count: self.count,
            chunk_count: self.chunks.len(),
            id_map: self.id_map.clone(),
        };

        let meta_path = path.join("columnar_meta.json");
        let file = std::fs::File::create(&meta_path)
            .map_err(VecStoreError::Io)?;
        serde_json::to_writer(file, &meta)
            .map_err(|e| VecStoreError::Serialization(e.to_string()))?;

        // Save chunks
        for (i, chunk) in self.chunks.iter().enumerate() {
            let chunk_path = path.join(format!("chunk_{}.bin", i));
            let file = std::fs::File::create(&chunk_path)
                .map_err(VecStoreError::Io)?;
            bincode::serialize_into(file, chunk)
                .map_err(|e| VecStoreError::Serialization(e.to_string()))?;
        }

        Ok(())
    }

    /// Get total count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Check if a vector exists
    pub fn contains(&self, id: &str) -> bool {
        self.id_map.contains_key(id)
    }

    /// Remove a vector (marks as deleted, doesn't reclaim space)
    pub fn remove(&mut self, id: &str) -> Result<(), VecStoreError> {
        if self.id_map.remove(id).is_some() {
            self.count -= 1;
            Ok(())
        } else {
            Err(VecStoreError::NotFound(format!("Vector {} not found", id)))
        }
    }

    /// Iterate over all IDs
    pub fn ids(&self) -> impl Iterator<Item = &String> {
        self.id_map.keys()
    }

    /// Compact storage (remove deleted entries)
    pub fn compact(&mut self) -> Result<(), VecStoreError> {
        let mut new_chunks = Vec::new();
        let mut new_id_map = HashMap::new();
        let mut new_count = 0;

        let mut current_chunk = ColumnarChunk::new(0, 0, self.dimensions, self.config.chunk_size);

        for (id, (chunk_idx, local_idx)) in &self.id_map {
            let vector = self.chunks[*chunk_idx].get_vector(*local_idx)?;

            if current_chunk.count >= self.config.chunk_size {
                if self.config.compute_stats {
                    current_chunk.compute_stats();
                }
                new_chunks.push(current_chunk);
                let new_idx = new_chunks.len();
                current_chunk = ColumnarChunk::new(new_idx, new_count, self.dimensions, self.config.chunk_size);
            }

            let local_idx = current_chunk.count;
            current_chunk.add_vector(&vector)?;
            new_id_map.insert(id.clone(), (new_chunks.len(), local_idx));
            new_count += 1;
        }

        // Add final chunk
        if current_chunk.count > 0 {
            if self.config.compute_stats {
                current_chunk.compute_stats();
            }
            new_chunks.push(current_chunk);
        }

        self.chunks = new_chunks;
        self.id_map = new_id_map;
        self.count = new_count;
        self.active_chunk = None;
        self.global_stats = None;

        Ok(())
    }

    /// Batch distance calculation (leverages columnar layout)
    pub fn batch_euclidean_distance(&self, query: &[f32]) -> Result<Vec<(String, f32)>, VecStoreError> {
        if query.len() != self.dimensions {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }

        let mut results = Vec::with_capacity(self.count);

        for chunk in &self.chunks {
            for local_idx in 0..chunk.count {
                let mut dist_sq = 0.0f32;

                // Access columns directly for better cache locality
                for (d, &q) in query.iter().enumerate() {
                    let v = chunk.columns[d][local_idx];
                    let diff = q - v;
                    dist_sq += diff * diff;
                }

                // Find the ID for this vector
                for (id, &(cidx, lidx)) in &self.id_map {
                    if cidx == chunk.index && lidx == local_idx {
                        results.push((id.clone(), dist_sq.sqrt()));
                        break;
                    }
                }
            }
        }

        Ok(results)
    }

    /// Efficient column range scan
    pub fn scan_column_range(
        &self,
        dimension: usize,
        min_val: f32,
        max_val: f32,
    ) -> Result<Vec<String>, VecStoreError> {
        if dimension >= self.dimensions {
            return Err(VecStoreError::InvalidInput(format!(
                "Dimension {} out of range", dimension
            )));
        }

        let mut results = Vec::new();

        for chunk in &self.chunks {
            for local_idx in 0..chunk.count {
                let val = chunk.columns[dimension][local_idx];
                if val >= min_val && val <= max_val {
                    // Find ID
                    for (id, &(cidx, lidx)) in &self.id_map {
                        if cidx == chunk.index && lidx == local_idx {
                            results.push(id.clone());
                            break;
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Get chunk count
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get storage size estimate
    pub fn storage_size(&self) -> usize {
        let mut size = 0;

        for chunk in &self.chunks {
            if let Some(ref compressed) = chunk.compressed_columns {
                size += compressed.iter().map(|c| c.len()).sum::<usize>();
            } else {
                size += chunk.columns.iter()
                    .map(|c| c.len() * size_of::<f32>())
                    .sum::<usize>();
            }
        }

        size
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let raw_size = self.count * self.dimensions * size_of::<f32>();
        let stored_size = self.storage_size();

        if stored_size > 0 {
            raw_size as f32 / stored_size as f32
        } else {
            1.0
        }
    }
}

/// Metadata for persistence
#[derive(Debug, Serialize, Deserialize)]
struct ColumnarMeta {
    dimensions: usize,
    count: usize,
    chunk_count: usize,
    id_map: HashMap<String, (usize, usize)>,
}

/// Iterator over columnar store
pub struct ColumnarIterator<'a> {
    store: &'a ColumnarStore,
    id_iter: std::collections::hash_map::Iter<'a, String, (usize, usize)>,
}

impl<'a> Iterator for ColumnarIterator<'a> {
    type Item = (String, Vec<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((id, (chunk_idx, local_idx))) = self.id_iter.next()
            && let Ok(vec) = self.store.chunks[*chunk_idx].get_vector(*local_idx) {
                return Some((id.clone(), vec));
            }
        None
    }
}

impl ColumnarStore {
    /// Iterate over all vectors
    pub fn iter(&self) -> ColumnarIterator<'_> {
        ColumnarIterator {
            store: self,
            id_iter: self.id_map.iter(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_columnar_store() {
        let config = ColumnarConfig::default();
        let store = ColumnarStore::new(3, config).unwrap();

        assert_eq!(store.dimensions(), 3);
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_add_and_get() {
        let config = ColumnarConfig::default();
        let mut store = ColumnarStore::new(3, config).unwrap();

        store.add("vec1", &[1.0, 2.0, 3.0]).unwrap();
        store.add("vec2", &[4.0, 5.0, 6.0]).unwrap();

        assert_eq!(store.count(), 2);

        let v1 = store.get("vec1").unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 3.0]);

        let v2 = store.get("vec2").unwrap();
        assert_eq!(v2, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_read_dimensions() {
        let config = ColumnarConfig::default();
        let mut store = ColumnarStore::new(5, config).unwrap();

        store.add("vec1", &[0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();

        let partial = store.read_dimensions("vec1", &[0, 2, 4]).unwrap();
        assert_eq!(partial, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_read_column() {
        let config = ColumnarConfig::default();
        let mut store = ColumnarStore::new(3, config).unwrap();

        store.add("vec1", &[1.0, 2.0, 3.0]).unwrap();
        store.add("vec2", &[4.0, 5.0, 6.0]).unwrap();
        store.add("vec3", &[7.0, 8.0, 9.0]).unwrap();

        let col0 = store.read_column(0).unwrap();
        assert_eq!(col0, vec![1.0, 4.0, 7.0]);

        let col1 = store.read_column(1).unwrap();
        assert_eq!(col1, vec![2.0, 5.0, 8.0]);
    }

    #[test]
    fn test_column_stats() {
        let config = ColumnarConfig::default();
        let mut store = ColumnarStore::new(2, config).unwrap();

        store.add("v1", &[1.0, 10.0]).unwrap();
        store.add("v2", &[2.0, 20.0]).unwrap();
        store.add("v3", &[3.0, 30.0]).unwrap();

        let stats = store.column_stats(0).unwrap();
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 3.0);
        assert_eq!(stats.count, 3);
    }

    #[test]
    fn test_chunk_creation() {
        let config = ColumnarConfig {
            chunk_size: 2,
            ..Default::default()
        };
        let mut store = ColumnarStore::new(2, config).unwrap();

        store.add("v1", &[1.0, 2.0]).unwrap();
        store.add("v2", &[3.0, 4.0]).unwrap();
        store.add("v3", &[5.0, 6.0]).unwrap();

        assert_eq!(store.chunk_count(), 2);
    }

    #[test]
    fn test_remove() {
        let config = ColumnarConfig::default();
        let mut store = ColumnarStore::new(2, config).unwrap();

        store.add("v1", &[1.0, 2.0]).unwrap();
        store.add("v2", &[3.0, 4.0]).unwrap();

        assert_eq!(store.count(), 2);

        store.remove("v1").unwrap();
        assert_eq!(store.count(), 1);
        assert!(!store.contains("v1"));
        assert!(store.contains("v2"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = ColumnarConfig::default();
        let mut store = ColumnarStore::new(3, config).unwrap();

        let result = store.add("v1", &[1.0, 2.0]); // Wrong dimension
        assert!(result.is_err());
    }

    #[test]
    fn test_delta_encoding() {
        let column = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let encoded = ColumnarChunk::delta_encode(&column);

        // Should have 5 * 4 bytes
        assert_eq!(encoded.len(), 20);
    }

    #[test]
    fn test_rle_encoding() {
        let column = vec![1.0, 1.0, 1.0, 2.0, 2.0];
        let encoded = ColumnarChunk::rle_encode(&column);

        // Two runs: (3, 1.0) and (2, 2.0)
        assert_eq!(encoded.len(), 16); // 2 * (4 + 4) bytes
    }

    #[test]
    fn test_batch_distance() {
        let config = ColumnarConfig::default();
        let mut store = ColumnarStore::new(3, config).unwrap();

        store.add("v1", &[1.0, 0.0, 0.0]).unwrap();
        store.add("v2", &[0.0, 1.0, 0.0]).unwrap();
        store.add("v3", &[0.0, 0.0, 1.0]).unwrap();

        let distances = store.batch_euclidean_distance(&[1.0, 0.0, 0.0]).unwrap();

        // v1 should have distance 0
        let v1_dist = distances.iter().find(|(id, _)| id == "v1").unwrap().1;
        assert!(v1_dist < 0.01);

        // v2 and v3 should have distance sqrt(2)
        let v2_dist = distances.iter().find(|(id, _)| id == "v2").unwrap().1;
        assert!((v2_dist - 2.0_f32.sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_column_range_scan() {
        let config = ColumnarConfig::default();
        let mut store = ColumnarStore::new(2, config).unwrap();

        store.add("v1", &[1.0, 10.0]).unwrap();
        store.add("v2", &[5.0, 20.0]).unwrap();
        store.add("v3", &[10.0, 30.0]).unwrap();

        let results = store.scan_column_range(0, 2.0, 8.0).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results.contains(&"v2".to_string()));
    }

    #[test]
    fn test_iterator() {
        let config = ColumnarConfig::default();
        let mut store = ColumnarStore::new(2, config).unwrap();

        store.add("v1", &[1.0, 2.0]).unwrap();
        store.add("v2", &[3.0, 4.0]).unwrap();

        let mut count = 0;
        for (_id, vec) in store.iter() {
            assert_eq!(vec.len(), 2);
            count += 1;
        }
        assert_eq!(count, 2);
    }
}
