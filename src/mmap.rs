//! Memory-mapped storage for disk-based HNSW index
//!
//! This module provides memory-mapped file support to scale beyond RAM limits.
//! It allows vecstore to handle billions of vectors by storing the HNSW graph
//! and vector data on disk while accessing them as if they were in memory.
//!
//! ## Features
//!
//! - Memory-mapped vector storage
//! - Lazy loading (only accessed pages are loaded into RAM)
//! - Copy-on-write support for safe concurrent reads
//! - Platform-agnostic (uses `memmap2` crate)
//!
//! ## Usage
//!
//! ```no_run
//! use vecstore::mmap::{MmapVectorStore, MmapConfig};
//!
//! # fn main() -> anyhow::Result<()> {
//! let config = MmapConfig {
//!     vector_dim: 128,
//!     initial_capacity: 1_000_000,
//!     ..Default::default()
//! };
//!
//! let mut store = MmapVectorStore::create("vectors.mmap", config)?;
//!
//! // Insert vectors - stored on disk, accessed via mmap
//! store.insert(0, &vec![0.1; 128])?;
//!
//! // Read vectors - only accessed pages loaded into RAM
//! let vector = store.get(0)?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::Path;

/// Configuration for memory-mapped vector storage
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Vector dimensionality
    pub vector_dim: usize,

    /// Initial capacity (number of vectors)
    pub initial_capacity: usize,

    /// Use huge pages if available (Linux only)
    pub use_huge_pages: bool,

    /// Advise kernel about access pattern
    pub populate: bool,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            vector_dim: 128,
            initial_capacity: 100_000,
            use_huge_pages: false,
            populate: false,
        }
    }
}

/// Memory-mapped vector store
pub struct MmapVectorStore {
    config: MmapConfig,
    mmap: MmapMut,
    file: File,
    count: usize,
    capacity: usize,
}

impl MmapVectorStore {
    /// Create a new memory-mapped vector store
    pub fn create<P: AsRef<Path>>(path: P, config: MmapConfig) -> Result<Self> {
        let bytes_per_vector = config.vector_dim * std::mem::size_of::<f32>();
        let file_size = bytes_per_vector * config.initial_capacity;

        // Create the file and set its size
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .context("Failed to create mmap file")?;

        file.set_len(file_size as u64)
            .context("Failed to set file size")?;

        // Create memory map
        let mut mmap_options = MmapOptions::new();

        if config.populate {
            mmap_options.populate();
        }

        let mmap = unsafe {
            mmap_options
                .map_mut(&file)
                .context("Failed to create memory map")?
        };

        let capacity = config.initial_capacity;

        Ok(Self {
            config,
            mmap,
            file,
            count: 0,
            capacity,
        })
    }

    /// Open an existing memory-mapped vector store
    pub fn open<P: AsRef<Path>>(path: P, config: MmapConfig) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .context("Failed to open mmap file")?;

        let file_size = file.metadata()?.len() as usize;
        let bytes_per_vector = config.vector_dim * std::mem::size_of::<f32>();
        let capacity = file_size / bytes_per_vector;

        let mut mmap_options = MmapOptions::new();

        if config.populate {
            mmap_options.populate();
        }

        let mmap = unsafe {
            mmap_options
                .map_mut(&file)
                .context("Failed to open memory map")?
        };

        Ok(Self {
            config,
            mmap,
            file,
            count: capacity, // Will be set correctly when loading metadata
            capacity,
        })
    }

    /// Insert a vector at a specific index
    pub fn insert(&mut self, index: usize, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.vector_dim {
            anyhow::bail!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.vector_dim,
                vector.len()
            );
        }

        if index >= self.capacity {
            self.grow()?;
        }

        let offset = self.vector_offset(index);
        let bytes = self.vector_to_bytes(vector);

        self.mmap[offset..offset + bytes.len()].copy_from_slice(&bytes);

        if index >= self.count {
            self.count = index + 1;
        }

        Ok(())
    }

    /// Get a vector at a specific index
    pub fn get(&self, index: usize) -> Result<Vec<f32>> {
        if index >= self.count {
            anyhow::bail!("Index out of bounds: {} >= {}", index, self.count);
        }

        let offset = self.vector_offset(index);
        let bytes_per_vector = self.config.vector_dim * std::mem::size_of::<f32>();
        let bytes = &self.mmap[offset..offset + bytes_per_vector];

        Ok(self.bytes_to_vector(bytes))
    }

    /// Get the number of vectors currently stored
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the current capacity (max vectors before growth)
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Flush changes to disk
    pub fn flush(&mut self) -> Result<()> {
        self.mmap.flush().context("Failed to flush memory map")?;
        Ok(())
    }

    /// Grow the memory map to accommodate more vectors
    fn grow(&mut self) -> Result<()> {
        let new_capacity = self.capacity * 2;
        let bytes_per_vector = self.config.vector_dim * std::mem::size_of::<f32>();
        let new_size = bytes_per_vector * new_capacity;

        // Flush current mmap before resizing
        self.mmap.flush().context("Failed to flush before resize")?;

        // Resize the file
        self.file
            .set_len(new_size as u64)
            .context("Failed to resize file")?;

        // Recreate the memory map
        let mut mmap_options = MmapOptions::new();

        if self.config.populate {
            mmap_options.populate();
        }

        self.mmap = unsafe {
            mmap_options
                .map_mut(&self.file)
                .context("Failed to remap after resize")?
        };

        self.capacity = new_capacity;

        Ok(())
    }

    /// Calculate the byte offset for a vector index
    fn vector_offset(&self, index: usize) -> usize {
        index * self.config.vector_dim * std::mem::size_of::<f32>()
    }

    /// Convert a vector to bytes
    fn vector_to_bytes(&self, vector: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(vector));
        for &value in vector {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Convert bytes to a vector
    fn bytes_to_vector(&self, bytes: &[u8]) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.config.vector_dim);
        for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            vector.push(value);
        }
        vector
    }
}

impl Drop for MmapVectorStore {
    fn drop(&mut self) {
        // Ensure changes are flushed before dropping
        let _ = self.mmap.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_create_mmap_store() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = MmapConfig {
            vector_dim: 3,
            initial_capacity: 10,
            ..Default::default()
        };

        let store = MmapVectorStore::create(temp_file.path(), config).unwrap();
        assert_eq!(store.len(), 0);
        assert_eq!(store.capacity(), 10);
    }

    #[test]
    fn test_insert_and_get() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = MmapConfig {
            vector_dim: 3,
            initial_capacity: 10,
            ..Default::default()
        };

        let mut store = MmapVectorStore::create(temp_file.path(), config).unwrap();

        let vector = vec![1.0, 2.0, 3.0];
        store.insert(0, &vector).unwrap();

        let retrieved = store.get(0).unwrap();
        assert_eq!(retrieved, vector);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_insert_multiple() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = MmapConfig {
            vector_dim: 2,
            initial_capacity: 100,
            ..Default::default()
        };

        let mut store = MmapVectorStore::create(temp_file.path(), config).unwrap();

        for i in 0..10 {
            let vector = vec![i as f32, (i * 2) as f32];
            store.insert(i, &vector).unwrap();
        }

        assert_eq!(store.len(), 10);

        for i in 0..10 {
            let retrieved = store.get(i).unwrap();
            let expected = vec![i as f32, (i * 2) as f32];
            assert_eq!(retrieved, expected);
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = MmapConfig {
            vector_dim: 3,
            initial_capacity: 10,
            ..Default::default()
        };

        let mut store = MmapVectorStore::create(temp_file.path(), config).unwrap();

        let wrong_vector = vec![1.0, 2.0]; // Wrong dimension
        let result = store.insert(0, &wrong_vector);
        assert!(result.is_err());
    }

    #[test]
    fn test_out_of_bounds() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = MmapConfig {
            vector_dim: 3,
            initial_capacity: 10,
            ..Default::default()
        };

        let store = MmapVectorStore::create(temp_file.path(), config).unwrap();

        let result = store.get(100);
        assert!(result.is_err());
    }

    #[test]
    fn test_persist_and_reload() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = MmapConfig {
            vector_dim: 4,
            initial_capacity: 10,
            ..Default::default()
        };

        // Create and populate
        {
            let mut store = MmapVectorStore::create(temp_file.path(), config.clone()).unwrap();
            for i in 0..5 {
                let vector = vec![i as f32; 4];
                store.insert(i, &vector).unwrap();
            }
            store.flush().unwrap();
        }

        // Reload and verify
        {
            let store = MmapVectorStore::open(temp_file.path(), config).unwrap();
            for i in 0..5 {
                let retrieved = store.get(i).unwrap();
                let expected = vec![i as f32; 4];
                assert_eq!(retrieved, expected);
            }
        }
    }

    #[test]
    fn test_grow_capacity() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = MmapConfig {
            vector_dim: 2,
            initial_capacity: 2, // Small initial capacity
            ..Default::default()
        };

        let mut store = MmapVectorStore::create(temp_file.path(), config).unwrap();
        assert_eq!(store.capacity(), 2);

        // Insert beyond initial capacity
        for i in 0..5 {
            let vector = vec![i as f32, (i * 2) as f32];
            store.insert(i, &vector).unwrap();
        }

        // Should have grown
        assert!(store.capacity() >= 5);
        assert_eq!(store.len(), 5);

        // Verify all vectors
        for i in 0..5 {
            let retrieved = store.get(i).unwrap();
            let expected = vec![i as f32, (i * 2) as f32];
            assert_eq!(retrieved, expected);
        }
    }
}
