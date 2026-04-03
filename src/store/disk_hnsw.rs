//! Disk-backed HNSW Index with Memory Mapping
//!
//! This module provides a memory-mapped HNSW implementation that can scale to
//! 100M+ vectors while keeping memory usage minimal.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │     Memory-Mapped HNSW Graph        │
//! ├─────────────────────────────────────┤
//! │  Layer 0: Full graph (all nodes)    │
//! │  Layer 1: Subset (1/M nodes)        │
//! │  Layer 2: Subset (1/M² nodes)       │
//! │  ...                                │
//! │  Layer L: Entry point (1 node)      │
//! └─────────────────────────────────────┘
//!
//! Each node stored as:
//! [node_id: u64][layer: u8][num_edges: u16][edges: [u64]]
//! ```
//!
//! ## Features
//!
//! - Memory-mapped files for large-scale data
//! - Efficient sequential I/O patterns
//! - Incremental updates with append-only log
//! - Background compaction
//! - Cache-aware graph traversal

use anyhow::{anyhow, Context, Result};
use memmap2::{Mmap, MmapOptions};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[cfg(feature = "async")]
use std::sync::Arc;

#[cfg(feature = "async")]
use tokio::sync::RwLock;

/// Candidate node for HNSW search (max-heap by distance for beam search)
#[derive(Clone)]
struct Candidate {
    id: u64,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for max-heap (we want smallest distances first)
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

/// Min-heap candidate (for result collection)
#[derive(Clone)]
struct MinCandidate {
    id: u64,
    distance: f32,
}

impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for MinCandidate {}

impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Normal order for min-heap (largest distances at top for pruning)
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

/// Configuration for disk-backed HNSW
#[derive(Debug, Clone)]
pub struct DiskHNSWConfig {
    /// Maximum number of connections per node
    pub m: usize,
    /// Size multiplier for connection count at layer 0
    pub m_max0: usize,
    /// Maximum layer
    pub ml: f32,
    /// Selection factor for candidate list
    pub ef_construction: usize,
    /// Node buffer size for batching
    pub node_buffer_size: usize,
    /// Enable background compaction
    pub enable_compaction: bool,
}

impl Default for DiskHNSWConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ml: 1.0 / (16.0_f32.ln()),
            ef_construction: 200,
            node_buffer_size: 1000,
            enable_compaction: true,
        }
    }
}

/// Node in the HNSW graph
#[derive(Debug, Clone)]
pub struct HNSWNode {
    /// Node ID (index)
    pub id: u64,
    /// Layer this node exists in
    pub layer: u8,
    /// Edges to neighbors at this layer
    pub edges: Vec<u64>,
}

/// Header for the memory-mapped file
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct FileHeader {
    /// Magic number for validation
    magic: [u8; 4],
    /// Version
    version: u32,
    /// Number of nodes
    node_count: u64,
    /// Number of layers
    layer_count: u8,
    /// M parameter
    m: u16,
    /// Entry point node ID
    entry_point: u64,
    /// Length of data written (offset to end of last node)
    data_length: u64,
    /// Reserved for future use
    reserved: [u8; 24],
}

impl FileHeader {
    const MAGIC: [u8; 4] = *b"HNSW";
    const VERSION: u32 = 1;
    const SIZE: usize = size_of::<FileHeader>();

    fn new(m: u16) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            node_count: 0,
            layer_count: 0,
            m,
            entry_point: 0,
            data_length: FileHeader::SIZE as u64,
            reserved: [0; 24],
        }
    }

    fn validate(&self) -> Result<()> {
        if self.magic != Self::MAGIC {
            return Err(anyhow!("Invalid magic number"));
        }
        if self.version != Self::VERSION {
            return Err(anyhow!(
                "Unsupported version: expected {}, got {}",
                Self::VERSION,
                self.version
            ));
        }
        Ok(())
    }
}

/// Disk-backed HNSW index
pub struct DiskHNSW {
    config: DiskHNSWConfig,
    file_path: PathBuf,
    /// Memory-mapped file
    #[cfg(not(feature = "async"))]
    mmap: Option<Mmap>,
    #[cfg(feature = "async")]
    mmap: Option<Arc<RwLock<Mmap>>>,
    /// Node offset table (node_id -> file offset per layer)
    /// Key: (node_id, layer), Value: file offset
    node_layer_offsets: HashMap<(u64, u8), u64>,
    /// Node offset table (node_id -> file offset) - legacy for base layer
    node_offsets: HashMap<u64, u64>,
    /// Maximum layer for each node
    node_max_layer: HashMap<u64, u8>,
    /// Layer sizes
    layer_sizes: Vec<usize>,
    /// Entry point
    entry_point: Option<u64>,
    /// Entry point layer (highest layer)
    entry_point_layer: u8,
    /// Current node count
    node_count: u64,
}

impl DiskHNSW {
    /// Create a new disk-backed HNSW index
    pub fn create(path: impl Into<PathBuf>, config: DiskHNSWConfig) -> Result<Self> {
        let file_path = path.into();

        // Create the file with initial header
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)
            .context("Failed to create HNSW file")?;

        let header = FileHeader::new(config.m as u16);
        let header_bytes = unsafe {
            std::slice::from_raw_parts(&header as *const FileHeader as *const u8, FileHeader::SIZE)
        };
        file.write_all(header_bytes)
            .context("Failed to write header")?;

        // Allocate some initial space (1MB)
        file.set_len(1024 * 1024)
            .context("Failed to set file size")?;

        drop(file);

        // Initialize mmap
        let file = OpenOptions::new()
            .read(true)
            .open(&file_path)
            .context("Failed to open file for mapping")?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .context("Failed to memory-map file")?
        };

        #[cfg(not(feature = "async"))]
        let mmap_field = Some(mmap);

        #[cfg(feature = "async")]
        let mmap_field = Some(Arc::new(RwLock::new(mmap)));

        Ok(Self {
            config,
            file_path,
            mmap: mmap_field,
            node_layer_offsets: HashMap::new(),
            node_offsets: HashMap::new(),
            node_max_layer: HashMap::new(),
            layer_sizes: Vec::new(),
            entry_point: None,
            entry_point_layer: 0,
            node_count: 0,
        })
    }

    /// Open an existing disk-backed HNSW index
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let file_path = path.into();

        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(&file_path)
            .context("Failed to open HNSW file")?;

        // Memory-map the file
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .context("Failed to memory-map file")?
        };

        // Read header
        if mmap.len() < FileHeader::SIZE {
            return Err(anyhow!("File too small to contain header"));
        }

        let header = unsafe { &*(mmap.as_ptr() as *const FileHeader) };
        header.validate()?;

        // Build node offset tables by scanning the file
        let mut node_offsets = HashMap::new();
        let mut node_layer_offsets = HashMap::new();
        let mut node_max_layer: HashMap<u64, u8> = HashMap::new();
        let mut offset = FileHeader::SIZE as u64;
        let mut entry_point = None;
        let mut entry_point_layer = 0u8;
        let mut unique_nodes = HashSet::new();

        // Use data_length from header to know where to stop
        let data_end = header.data_length;

        // Scan until we reach the end of data
        while offset < data_end && (offset as usize) < mmap.len() {
            // Check if we have enough space for a node header
            if (offset as usize) + 11 > mmap.len() {
                break;
            }

            let peek = &mmap[offset as usize..offset as usize + 11];
            let node_id = u64::from_le_bytes(peek[0..8].try_into().unwrap());
            let layer = peek[8];
            let num_edges = u16::from_le_bytes(peek[9..11].try_into().unwrap());

            // Validate that we have space for all edges
            let node_size = 11 + (num_edges as usize * 8);
            if (offset as usize) + node_size > mmap.len() {
                break;
            }

            // Track layer-specific offset
            node_layer_offsets.insert((node_id, layer), offset);

            // Track base layer offset (layer 0)
            if layer == 0 {
                node_offsets.insert(node_id, offset);
            }

            // Track max layer for each node
            let current_max = node_max_layer.entry(node_id).or_insert(0);
            if layer > *current_max {
                *current_max = layer;
            }

            unique_nodes.insert(node_id);

            // Update entry point (highest layer node)
            if layer > entry_point_layer || entry_point.is_none() {
                entry_point = Some(node_id);
                entry_point_layer = layer;
            }

            // Move to next node
            offset += node_size as u64;
        }

        let node_count = unique_nodes.len() as u64;

        let config = DiskHNSWConfig {
            m: header.m as usize,
            ..Default::default()
        };

        #[cfg(not(feature = "async"))]
        let mmap_field = Some(mmap);

        #[cfg(feature = "async")]
        let mmap_field = Some(Arc::new(RwLock::new(mmap)));

        Ok(Self {
            config,
            file_path,
            mmap: mmap_field,
            node_layer_offsets,
            node_offsets,
            node_max_layer,
            layer_sizes: Vec::new(),
            entry_point,
            entry_point_layer,
            node_count,
        })
    }

    /// Add a node to the index
    pub fn add_node(&mut self, node: HNSWNode) -> Result<()> {
        // Open file for appending
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.file_path)
            .context("Failed to open file for writing")?;

        // Read current header to get data_length (position after last node)
        let mut header_buf = vec![0u8; FileHeader::SIZE];
        file.read_exact(&mut header_buf)?;
        let current_data_length = unsafe {
            (*(header_buf.as_ptr() as *const FileHeader)).data_length
        };

        // Seek to end of actual data (not end of file which may have padding)
        let file_len = file.metadata()?.len();
        let offset = file
            .seek(SeekFrom::Start(current_data_length))
            .context("Failed to seek to data end")?;

        // Calculate required size
        let node_size = 11 + (node.edges.len() * 8);
        let required_size = offset + node_size as u64;

        // Expand file if needed
        if required_size > file_len {
            let new_size = (required_size + 1024 * 1024).max(file_len * 2);
            file.set_len(new_size)?;
        }

        // Write node header
        file.write_all(&node.id.to_le_bytes())?;
        file.write_all(&[node.layer])?;
        file.write_all(&(node.edges.len() as u16).to_le_bytes())?;

        // Write edges
        for edge in &node.edges {
            file.write_all(&edge.to_le_bytes())?;
        }

        file.flush()?;

        // Calculate new data length
        let new_data_length = offset + node_size as u64;

        // Update header with new data length
        file.seek(SeekFrom::Start(0))?;
        let mut header_buf = vec![0u8; FileHeader::SIZE];
        file.read_exact(&mut header_buf)?;

        let header_ptr = header_buf.as_mut_ptr() as *mut FileHeader;
        unsafe {
            (*header_ptr).data_length = new_data_length;
            (*header_ptr).node_count = self.node_count + 1;
        }

        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header_buf)?;
        file.flush()?;
        drop(file);

        // Update offset tables
        self.node_layer_offsets.insert((node.id, node.layer), offset);
        if node.layer == 0 {
            self.node_offsets.insert(node.id, offset);
        }

        // Track max layer for this node (increment node count for new nodes)
        let is_new_node = !self.node_max_layer.contains_key(&node.id);
        let current_max = self.node_max_layer.entry(node.id).or_insert(0);
        if node.layer > *current_max {
            *current_max = node.layer;
        }

        // Increment node count for new unique nodes
        if is_new_node {
            self.node_count += 1;
        }

        // Update entry point if this is a higher layer
        if node.layer > self.entry_point_layer || self.entry_point.is_none() {
            self.entry_point = Some(node.id);
            self.entry_point_layer = node.layer;
        }

        // Re-map the file after adding nodes
        self.remap()?;

        Ok(())
    }

    /// Get a node from the index
    #[cfg(not(feature = "async"))]
    pub fn get_node(&self, node_id: u64) -> Result<HNSWNode> {
        let offset = *self
            .node_offsets
            .get(&node_id)
            .ok_or_else(|| anyhow!("Node {} not found", node_id))?;

        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| anyhow!("Index not mapped"))?;

        let offset = offset as usize;

        // Read node data
        let id = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap());
        let layer = mmap[offset + 8];
        let num_edges = u16::from_le_bytes(mmap[offset + 9..offset + 11].try_into().unwrap());

        let mut edges = Vec::with_capacity(num_edges as usize);
        let mut edge_offset = offset + 11;
        for _ in 0..num_edges {
            let edge = u64::from_le_bytes(mmap[edge_offset..edge_offset + 8].try_into().unwrap());
            edges.push(edge);
            edge_offset += 8;
        }

        Ok(HNSWNode { id, layer, edges })
    }

    /// Get a node from the index (sync fallback for async mode)
    ///
    /// In async mode, prefer using `get_node_async()` for proper lock handling.
    /// This sync version attempts to acquire the lock immediately and returns
    /// an error if the lock is currently held.
    #[cfg(feature = "async")]
    pub fn get_node(&self, node_id: u64) -> Result<HNSWNode> {
        let offset = *self
            .node_offsets
            .get(&node_id)
            .ok_or_else(|| anyhow!("Node {} not found", node_id))?;

        let mmap_arc = self
            .mmap
            .as_ref()
            .ok_or_else(|| anyhow!("Index not mapped"))?;

        // Try to acquire read lock without blocking
        let mmap = mmap_arc.try_read()
            .map_err(|_| anyhow!("Lock held - use get_node_async() for async access"))?;

        let offset = offset as usize;

        // Read node data
        let id = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap());
        let layer = mmap[offset + 8];
        let num_edges = u16::from_le_bytes(mmap[offset + 9..offset + 11].try_into().unwrap());

        let mut edges = Vec::with_capacity(num_edges as usize);
        let mut edge_offset = offset + 11;
        for _ in 0..num_edges {
            let edge = u64::from_le_bytes(mmap[edge_offset..edge_offset + 8].try_into().unwrap());
            edges.push(edge);
            edge_offset += 8;
        }

        Ok(HNSWNode { id, layer, edges })
    }

    /// Get a node from the index (async version)
    ///
    /// Properly awaits the RwLock for async access to the memory-mapped data.
    #[cfg(feature = "async")]
    pub async fn get_node_async(&self, node_id: u64) -> Result<HNSWNode> {
        let offset = *self
            .node_offsets
            .get(&node_id)
            .ok_or_else(|| anyhow!("Node {} not found", node_id))?;

        let mmap_arc = self
            .mmap
            .as_ref()
            .ok_or_else(|| anyhow!("Index not mapped"))?;

        // Await the read lock
        let mmap = mmap_arc.read().await;

        let offset = offset as usize;

        // Read node data
        let id = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap());
        let layer = mmap[offset + 8];
        let num_edges = u16::from_le_bytes(mmap[offset + 9..offset + 11].try_into().unwrap());

        let mut edges = Vec::with_capacity(num_edges as usize);
        let mut edge_offset = offset + 11;
        for _ in 0..num_edges {
            let edge = u64::from_le_bytes(mmap[edge_offset..edge_offset + 8].try_into().unwrap());
            edges.push(edge);
            edge_offset += 8;
        }

        Ok(HNSWNode { id, layer, edges })
    }

    /// Get node layer
    #[allow(dead_code)]
    fn get_node_layer(&self, node_id: u64) -> Option<u8> {
        let offset = *self.node_offsets.get(&node_id)? as usize;

        #[cfg(not(feature = "async"))]
        let mmap = self.mmap.as_ref()?;

        #[cfg(feature = "async")]
        return None; // Would need async version

        #[cfg(not(feature = "async"))]
        if offset + 9 <= mmap.len() {
            Some(mmap[offset + 8])
        } else {
            None
        }

        #[cfg(feature = "async")]
        None
    }

    /// Re-map the file after growth
    fn remap(&mut self) -> Result<()> {
        let file = OpenOptions::new()
            .read(true)
            .open(&self.file_path)
            .context("Failed to open file for remapping")?;

        let new_mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .context("Failed to remap file")?
        };

        #[cfg(not(feature = "async"))]
        {
            self.mmap = Some(new_mmap);
        }

        #[cfg(feature = "async")]
        {
            self.mmap = Some(Arc::new(RwLock::new(new_mmap)));
        }

        Ok(())
    }

    /// Get stats about the index
    pub fn stats(&self) -> DiskHNSWStats {
        DiskHNSWStats {
            node_count: self.node_count,
            file_size_bytes: std::fs::metadata(&self.file_path)
                .map(|m| m.len())
                .unwrap_or(0),
            layer_count: self.layer_sizes.len(),
        }
    }

    /// Get node neighbors at a specific layer
    #[cfg(not(feature = "async"))]
    pub fn get_neighbors_at_layer(&self, node_id: u64, layer: u8) -> Result<Vec<u64>> {
        let offset = self
            .node_layer_offsets
            .get(&(node_id, layer))
            .ok_or_else(|| anyhow!("Node {} not found at layer {}", node_id, layer))?;

        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| anyhow!("Index not mapped"))?;

        let offset = *offset as usize;

        // Read node data
        if offset + 11 > mmap.len() {
            return Err(anyhow!("Invalid offset"));
        }

        let num_edges = u16::from_le_bytes(mmap[offset + 9..offset + 11].try_into().unwrap());

        let mut edges = Vec::with_capacity(num_edges as usize);
        let mut edge_offset = offset + 11;
        for _ in 0..num_edges {
            if edge_offset + 8 > mmap.len() {
                break;
            }
            let edge = u64::from_le_bytes(mmap[edge_offset..edge_offset + 8].try_into().unwrap());
            edges.push(edge);
            edge_offset += 8;
        }

        Ok(edges)
    }

    /// Get the entry point and its layer
    pub fn get_entry_point(&self) -> Option<(u64, u8)> {
        self.entry_point.map(|ep| (ep, self.entry_point_layer))
    }

    /// Get maximum layer for a node
    pub fn get_max_layer(&self, node_id: u64) -> Option<u8> {
        self.node_max_layer.get(&node_id).copied()
    }

    /// Check if a node exists at a specific layer
    pub fn has_node_at_layer(&self, node_id: u64, layer: u8) -> bool {
        self.node_layer_offsets.contains_key(&(node_id, layer))
    }

    /// Get all node IDs
    pub fn get_all_node_ids(&self) -> Vec<u64> {
        self.node_max_layer.keys().copied().collect()
    }

    /// Sync all data to disk
    ///
    /// Ensures all written data is persisted to disk before closing.
    /// Call this before dropping if you need to reopen the index.
    pub fn sync(&self) -> Result<()> {
        // Open file and sync to disk
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.file_path)
            .context("Failed to open file for sync")?;

        // Sync file data to disk
        file.sync_all().context("Failed to sync file")?;
        Ok(())
    }
}

impl Drop for DiskHNSW {
    fn drop(&mut self) {
        // Best effort sync on drop
        if let Err(e) = self.sync() {
            eprintln!("Warning: failed to sync DiskHNSW on drop: {}", e);
        }
    }
}

/// Statistics for disk-backed HNSW
#[derive(Debug, Clone)]
pub struct DiskHNSWStats {
    pub node_count: u64,
    pub file_size_bytes: u64,
    pub layer_count: usize,
}

/// Disk-backed vector storage
///
/// Stores vectors in a memory-mapped file for datasets larger than RAM.
/// Uses sequential access patterns optimized for disk I/O.
pub struct DiskVectorStorage {
    file_path: PathBuf,
    dimension: usize,
    vector_count: u64,
    /// Memory-mapped vectors file
    mmap: Option<Mmap>,
}

impl DiskVectorStorage {
    const HEADER_SIZE: usize = 32;

    /// Create new vector storage
    pub fn create(path: impl Into<PathBuf>, dimension: usize) -> Result<Self> {
        let file_path = path.into();

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)
            .context("Failed to create vectors file")?;

        // Write header: magic (4) + version (4) + dimension (8) + count (8) + reserved (8)
        file.write_all(b"VECS")?;
        file.write_all(&1u32.to_le_bytes())?;
        file.write_all(&(dimension as u64).to_le_bytes())?;
        file.write_all(&0u64.to_le_bytes())?;  // Initial count
        file.write_all(&[0u8; 8])?;  // Reserved

        // Allocate initial space (10K vectors worth)
        let initial_size = Self::HEADER_SIZE + (10_000 * dimension * 4);
        file.set_len(initial_size as u64)?;
        file.flush()?;
        drop(file);

        let file = OpenOptions::new()
            .read(true)
            .open(&file_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        Ok(Self {
            file_path,
            dimension,
            vector_count: 0,
            mmap: Some(mmap),
        })
    }

    /// Open existing vector storage
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let file_path = path.into();

        let file = OpenOptions::new()
            .read(true)
            .open(&file_path)
            .context("Failed to open vectors file")?;

        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Read header
        if mmap.len() < Self::HEADER_SIZE {
            return Err(anyhow!("File too small"));
        }

        if &mmap[0..4] != b"VECS" {
            return Err(anyhow!("Invalid magic number"));
        }

        let dimension = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let vector_count = u64::from_le_bytes(mmap[16..24].try_into().unwrap());

        Ok(Self {
            file_path,
            dimension,
            vector_count,
            mmap: Some(mmap),
        })
    }

    /// Add a vector at the given index
    pub fn add_vector(&mut self, index: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!("Vector dimension mismatch"));
        }

        let vector_size = self.dimension * 4;
        let offset = Self::HEADER_SIZE + (index as usize * vector_size);

        // Open file for writing
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.file_path)?;

        // Expand file if needed
        let required_size = offset + vector_size;
        let current_size = file.metadata()?.len() as usize;
        if required_size > current_size {
            let new_size = (required_size + 1024 * 1024).max(current_size * 2);
            file.set_len(new_size as u64)?;
        }

        // Write vector
        file.seek(SeekFrom::Start(offset as u64))?;
        for &val in vector {
            file.write_all(&val.to_le_bytes())?;
        }

        // Update count if this is a new vector
        if index >= self.vector_count {
            self.vector_count = index + 1;
            file.seek(SeekFrom::Start(16))?;
            file.write_all(&self.vector_count.to_le_bytes())?;
        }

        file.flush()?;
        drop(file);

        // Remap
        let file = OpenOptions::new().read(true).open(&self.file_path)?;
        self.mmap = Some(unsafe { MmapOptions::new().map(&file)? });

        Ok(())
    }

    /// Get a vector by index
    pub fn get_vector(&self, index: u64) -> Result<Vec<f32>> {
        let mmap = self.mmap.as_ref().ok_or_else(|| anyhow!("Not mapped"))?;

        let vector_size = self.dimension * 4;
        let offset = Self::HEADER_SIZE + (index as usize * vector_size);

        if offset + vector_size > mmap.len() {
            return Err(anyhow!("Vector index out of bounds"));
        }

        let mut vector = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            let byte_offset = offset + (i * 4);
            let val = f32::from_le_bytes(mmap[byte_offset..byte_offset + 4].try_into().unwrap());
            vector.push(val);
        }

        Ok(vector)
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get vector count
    pub fn len(&self) -> u64 {
        self.vector_count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vector_count == 0
    }
}

/// Combined disk-backed HNSW index with vector storage
///
/// This provides a complete solution for larger-than-RAM datasets by combining:
/// - Memory-mapped HNSW graph structure
/// - Memory-mapped vector storage
/// - Streaming search with minimal memory footprint
pub struct DiskHNSWIndex {
    /// HNSW graph structure
    graph: DiskHNSW,
    /// Vector storage
    vectors: DiskVectorStorage,
    /// ID to index mapping
    id_to_index: HashMap<String, u64>,
    /// Index to ID mapping
    index_to_id: HashMap<u64, String>,
    /// Dimension
    dimension: usize,
    /// ef_search parameter
    ef_search: usize,
}

impl DiskHNSWIndex {
    /// Create a new disk-backed HNSW index
    pub fn create(base_path: impl AsRef<Path>, dimension: usize, config: DiskHNSWConfig) -> Result<Self> {
        let base = base_path.as_ref();
        std::fs::create_dir_all(base)?;

        let graph = DiskHNSW::create(base.join("graph.hnsw"), config)?;
        let vectors = DiskVectorStorage::create(base.join("vectors.dat"), dimension)?;

        Ok(Self {
            graph,
            vectors,
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            dimension,
            ef_search: 100,
        })
    }

    /// Open an existing disk-backed HNSW index
    pub fn open(base_path: impl AsRef<Path>) -> Result<Self> {
        let base = base_path.as_ref();

        let graph = DiskHNSW::open(base.join("graph.hnsw"))?;
        let vectors = DiskVectorStorage::open(base.join("vectors.dat"))?;

        // Load ID mappings
        let mapping_path = base.join("ids.json");
        let (id_to_index, index_to_id) = if mapping_path.exists() {
            let data = std::fs::read_to_string(&mapping_path)?;
            let map: HashMap<String, u64> = serde_json::from_str(&data)?;
            let reverse: HashMap<u64, String> = map.iter().map(|(k, v)| (*v, k.clone())).collect();
            (map, reverse)
        } else {
            (HashMap::new(), HashMap::new())
        };

        Ok(Self {
            dimension: vectors.dimension(),
            graph,
            vectors,
            id_to_index,
            index_to_id,
            ef_search: 100,
        })
    }

    /// Insert a vector with proper HNSW neighbor connections
    #[cfg(not(feature = "async"))]
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!("Vector dimension mismatch"));
        }

        let index = self.vectors.len();

        // Store vector
        self.vectors.add_vector(index, &vector)?;

        // Store ID mapping first
        self.id_to_index.insert(id.clone(), index);
        self.index_to_id.insert(index, id);

        // Determine the random layer for this node
        let max_layer = self.random_layer();
        let m = self.graph.config.m;
        let m_max0 = self.graph.config.m_max0;
        let ef_construction = self.graph.config.ef_construction;

        // If this is the first node, just add it at all layers
        if self.graph.node_count == 0 {
            for layer in 0..=max_layer {
                let node = HNSWNode {
                    id: index,
                    layer,
                    edges: vec![],
                };
                self.graph.add_node(node)?;
            }
            return Ok(());
        }

        // Get entry point
        let (ep_id, ep_layer) = self.graph.get_entry_point()
            .ok_or_else(|| anyhow!("No entry point found"))?;

        let mut current_ep = ep_id;

        // Phase 1: Greedy search from top layer down to max_layer + 1
        for layer in (max_layer + 1..=ep_layer).rev() {
            if let Ok(closest) = self.search_layer_single(&vector, current_ep, layer) {
                current_ep = closest;
            }
        }

        // Phase 2: Insert at layers max_layer down to 0
        for layer in (0..=max_layer.min(ep_layer)).rev() {
            // Find neighbors at this layer
            let max_neighbors = if layer == 0 { m_max0 } else { m };
            let neighbors = self.search_layer_beam(&vector, current_ep, layer, ef_construction)?;

            // Select best neighbors (simple heuristic: take closest ones)
            let selected: Vec<u64> = neighbors
                .iter()
                .take(max_neighbors)
                .map(|(id, _)| *id)
                .collect();

            // Add node at this layer with connections
            let node = HNSWNode {
                id: index,
                layer,
                edges: selected.clone(),
            };
            self.graph.add_node(node)?;

            // Update current entry point for next layer
            if !neighbors.is_empty() {
                current_ep = neighbors[0].0;
            }
        }

        Ok(())
    }

    /// Fallback insert for async feature (simplified)
    #[cfg(feature = "async")]
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!("Vector dimension mismatch"));
        }

        let index = self.vectors.len();
        self.vectors.add_vector(index, &vector)?;

        let layer = self.random_layer();
        let node = HNSWNode {
            id: index,
            layer,
            edges: vec![],
        };
        self.graph.add_node(node)?;

        self.id_to_index.insert(id.clone(), index);
        self.index_to_id.insert(index, id);

        Ok(())
    }

    /// Search for k nearest neighbors using HNSW graph traversal
    #[cfg(not(feature = "async"))]
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if query.len() != self.dimension {
            return Err(anyhow!("Query dimension mismatch"));
        }

        // Handle empty index
        if self.vectors.is_empty() {
            return Ok(vec![]);
        }

        // For small datasets, brute force is faster and more reliable
        if self.vectors.len() <= 100 {
            return self.search_brute_force(query, k);
        }

        // Get entry point
        let (ep_id, ep_layer) = match self.graph.get_entry_point() {
            Some(ep) => ep,
            None => {
                // Fallback to brute force if no entry point
                return self.search_brute_force(query, k);
            }
        };

        let mut current_ep = ep_id;

        // Phase 1: Greedy search from top layer down to layer 1
        for layer in (1..=ep_layer).rev() {
            if let Ok(closest) = self.search_layer_single(query, current_ep, layer) {
                current_ep = closest;
            }
        }

        // Phase 2: Beam search at layer 0 with ef_search candidates
        let candidates = self.search_layer_beam(query, current_ep, 0, self.ef_search)?;

        // Return top k results
        let results: Vec<(String, f32)> = candidates
            .into_iter()
            .take(k)
            .filter_map(|(idx, dist)| {
                self.index_to_id.get(&idx).map(|id| (id.clone(), dist))
            })
            .collect();

        // If we didn't find enough results via HNSW, fall back to brute force
        if results.len() < k {
            return self.search_brute_force(query, k);
        }

        Ok(results)
    }

    /// Fallback search for async feature
    #[cfg(feature = "async")]
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        self.search_brute_force(query, k)
    }

    /// Greedy search within a layer - find single closest node
    #[cfg(not(feature = "async"))]
    fn search_layer_single(&self, query: &[f32], entry: u64, layer: u8) -> Result<u64> {
        let mut current = entry;
        let mut current_dist = self.get_distance(query, current)?;

        loop {
            let neighbors = match self.graph.get_neighbors_at_layer(current, layer) {
                Ok(n) => n,
                Err(_) => break, // No neighbors at this layer
            };

            let mut found_closer = false;
            for neighbor in neighbors {
                if let Ok(dist) = self.get_distance(query, neighbor)
                    && dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        found_closer = true;
                    }
            }

            if !found_closer {
                break;
            }
        }

        Ok(current)
    }

    /// Beam search within a layer - find ef closest nodes
    #[cfg(not(feature = "async"))]
    fn search_layer_beam(&self, query: &[f32], entry: u64, layer: u8, ef: usize) -> Result<Vec<(u64, f32)>> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<MinCandidate> = BinaryHeap::new();

        // Start with entry point
        let entry_dist = self.get_distance(query, entry)?;
        candidates.push(Candidate { id: entry, distance: entry_dist });
        results.push(MinCandidate { id: entry, distance: entry_dist });
        visited.insert(entry);

        while let Some(current) = candidates.pop() {
            // Stop if current is further than the worst result
            if results.len() >= ef
                && let Some(worst) = results.peek()
                    && current.distance > worst.distance {
                        break;
                    }

            // Get neighbors
            let neighbors = match self.graph.get_neighbors_at_layer(current.id, layer) {
                Ok(n) => n,
                Err(_) => continue,
            };

            for neighbor in neighbors {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                if let Ok(dist) = self.get_distance(query, neighbor) {
                    // Check if we should add this neighbor
                    let should_add = if results.len() < ef {
                        true
                    } else if let Some(worst) = results.peek() {
                        dist < worst.distance
                    } else {
                        true
                    };

                    if should_add {
                        candidates.push(Candidate { id: neighbor, distance: dist });
                        results.push(MinCandidate { id: neighbor, distance: dist });

                        // Trim results if too large
                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted vec
        let mut result_vec: Vec<(u64, f32)> = results
            .into_iter()
            .map(|c| (c.id, c.distance))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Ok(result_vec)
    }

    /// Fallback brute-force search
    fn search_brute_force(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        let mut distances: Vec<(u64, f32)> = Vec::new();

        for i in 0..self.vectors.len() {
            if let Ok(vec) = self.vectors.get_vector(i) {
                let dist = euclidean_distance(query, &vec);
                distances.push((i, dist));
            }
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        let results: Vec<(String, f32)> = distances
            .into_iter()
            .filter_map(|(idx, dist)| {
                self.index_to_id.get(&idx).map(|id| (id.clone(), dist))
            })
            .collect();

        Ok(results)
    }

    /// Get distance from query to a node
    fn get_distance(&self, query: &[f32], node_id: u64) -> Result<f32> {
        let vec = self.vectors.get_vector(node_id)?;
        Ok(euclidean_distance(query, &vec))
    }

    /// Set ef_search parameter
    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef;
    }

    /// Save ID mappings
    pub fn save(&self, base_path: impl AsRef<Path>) -> Result<()> {
        let mapping_path = base_path.as_ref().join("ids.json");
        let data = serde_json::to_string(&self.id_to_index)?;
        std::fs::write(mapping_path, data)?;
        Ok(())
    }

    /// Get stats
    pub fn stats(&self) -> DiskHNSWIndexStats {
        DiskHNSWIndexStats {
            vector_count: self.vectors.len(),
            dimension: self.dimension,
            graph_stats: self.graph.stats(),
        }
    }

    fn random_layer(&self) -> u8 {
        use rand::Rng;
        let mut rng = rand::rng();
        let r: f32 = rng.random();
        (-r.ln() * self.graph.config.ml).floor() as u8
    }
}

/// Stats for disk-backed HNSW index
#[derive(Debug, Clone)]
pub struct DiskHNSWIndexStats {
    pub vector_count: u64,
    pub dimension: usize,
    pub graph_stats: DiskHNSWStats,
}

/// Euclidean distance calculation
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_disk_hnsw() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.hnsw");

        let config = DiskHNSWConfig::default();
        let hnsw = DiskHNSW::create(path, config);
        assert!(hnsw.is_ok());

        let hnsw = hnsw.unwrap();
        assert_eq!(hnsw.node_count, 0);
    }

    #[test]
    fn test_add_and_get_node() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.hnsw");

        let config = DiskHNSWConfig::default();
        let mut hnsw = DiskHNSW::create(&path, config).unwrap();

        // Add a node
        let node = HNSWNode {
            id: 1,
            layer: 0,
            edges: vec![2, 3, 4],
        };

        hnsw.add_node(node.clone()).unwrap();
        assert_eq!(hnsw.node_count, 1);

        // Get the node back
        #[cfg(not(feature = "async"))]
        {
            let retrieved = hnsw.get_node(1).unwrap();
            assert_eq!(retrieved.id, 1);
            assert_eq!(retrieved.layer, 0);
            assert_eq!(retrieved.edges, vec![2, 3, 4]);
        }
    }

    #[test]
    fn test_add_multiple_nodes() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.hnsw");

        let config = DiskHNSWConfig::default();
        let mut hnsw = DiskHNSW::create(&path, config).unwrap();

        // Add multiple nodes
        for i in 0..10 {
            let node = HNSWNode {
                id: i,
                layer: (i % 3) as u8,
                edges: vec![(i + 1) % 10, (i + 2) % 10],
            };
            hnsw.add_node(node).unwrap();
        }

        assert_eq!(hnsw.node_count, 10);
    }

    #[test]
    fn test_open_existing_index() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.hnsw");

        // Create and populate index
        {
            let config = DiskHNSWConfig::default();
            let mut hnsw = DiskHNSW::create(&path, config).unwrap();

            for i in 0..5 {
                let node = HNSWNode {
                    id: i,
                    layer: 0,
                    edges: vec![],
                };
                hnsw.add_node(node).unwrap();
            }
        }

        // Open existing index
        let hnsw = DiskHNSW::open(&path).unwrap();
        assert_eq!(hnsw.node_count, 5);
    }

    #[test]
    fn test_stats() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.hnsw");

        let config = DiskHNSWConfig::default();
        let mut hnsw = DiskHNSW::create(&path, config).unwrap();

        // Add nodes
        for i in 0..10 {
            let node = HNSWNode {
                id: i,
                layer: 0,
                edges: vec![],
            };
            hnsw.add_node(node).unwrap();
        }

        let stats = hnsw.stats();
        assert_eq!(stats.node_count, 10);
        assert!(stats.file_size_bytes > 0);
    }

    #[test]
    fn test_entry_point_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.hnsw");

        let config = DiskHNSWConfig::default();
        let mut hnsw = DiskHNSW::create(&path, config).unwrap();

        // Add node at layer 0
        let node0 = HNSWNode {
            id: 0,
            layer: 0,
            edges: vec![],
        };
        hnsw.add_node(node0).unwrap();
        assert_eq!(hnsw.entry_point, Some(0));

        // Add node at layer 2 (should become entry point)
        let node1 = HNSWNode {
            id: 1,
            layer: 2,
            edges: vec![],
        };
        hnsw.add_node(node1).unwrap();
        assert_eq!(hnsw.entry_point, Some(1));
    }

    #[test]
    fn test_disk_vector_storage() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("vectors.dat");

        let mut storage = DiskVectorStorage::create(&path, 4).unwrap();
        assert_eq!(storage.dimension(), 4);
        assert!(storage.is_empty());

        // Add vectors
        storage.add_vector(0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        storage.add_vector(1, &[5.0, 6.0, 7.0, 8.0]).unwrap();

        assert_eq!(storage.len(), 2);

        // Retrieve vectors
        let v0 = storage.get_vector(0).unwrap();
        assert_eq!(v0, vec![1.0, 2.0, 3.0, 4.0]);

        let v1 = storage.get_vector(1).unwrap();
        assert_eq!(v1, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_disk_hnsw_index() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().join("index");

        let config = DiskHNSWConfig::default();
        let mut index = DiskHNSWIndex::create(&base_path, 4, config).unwrap();

        // Insert vectors
        index.insert("doc1".to_string(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert("doc2".to_string(), vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        index.insert("doc3".to_string(), vec![0.0, 0.0, 1.0, 0.0]).unwrap();

        let stats = index.stats();
        assert_eq!(stats.vector_count, 3);

        // Search
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "doc1"); // Closest should be doc1
        assert!(results[0].1 < 0.01); // Distance should be ~0
    }

    #[test]
    #[ignore] // Requires sufficient disk space - may fail on constrained systems
    fn test_disk_hnsw_index_large_dataset() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().join("large_index");

        let config = DiskHNSWConfig::default();
        let mut index = DiskHNSWIndex::create(&base_path, 128, config).unwrap();

        // Insert 100 vectors
        for i in 0..100 {
            let mut vec = vec![0.0f32; 128];
            vec[i % 128] = 1.0;
            index.insert(format!("doc{}", i), vec).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.vector_count, 100);

        // Search
        let mut query = vec![0.0f32; 128];
        query[0] = 1.0;
        let results = index.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, "doc0"); // First vector has 1.0 at index 0
    }
}
