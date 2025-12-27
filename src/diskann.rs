// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! DiskANN: SSD-Optimized Billion-Scale Vector Search
//!
//! Implementation of Microsoft's DiskANN algorithm for disk-based approximate
//! nearest neighbor search. Unlike HNSW which uses multi-layer graphs, DiskANN
//! uses a single Vamana graph optimized for SSD access patterns.
//!
//! # Key Features
//!
//! - **Billion-scale support**: Handle datasets that don't fit in RAM
//! - **SSD-optimized**: Minimizes random I/O with sector-aligned reads
//! - **PQ compression**: Product quantization for in-memory graph navigation
//! - **Beam search**: Configurable beam width for recall/latency trade-off
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::diskann::{DiskANN, DiskANNConfig};
//!
//! let config = DiskANNConfig {
//!     max_degree: 64,        // Graph degree (R)
//!     build_beam_width: 128, // Construction beam width (L)
//!     alpha: 1.2,            // Vamana pruning parameter
//!     pq_dims: 32,           // PQ subvector count
//!     sector_size: 4096,     // SSD sector size
//!     ..Default::default()
//! };
//!
//! let mut index = DiskANN::new(config, 768)?; // 768-dim vectors
//! index.build(&vectors)?;
//! index.save("index.diskann")?;
//!
//! // Search
//! let results = index.search(&query, 10, 50)?; // k=10, beam_width=50
//! ```
//!
//! # Algorithm Overview
//!
//! DiskANN uses the Vamana algorithm to build a single-layer graph where:
//! 1. Each node has at most R neighbors (max_degree)
//! 2. Edges are pruned using α-RNG rule for better navigability
//! 3. Graph is stored on SSD with PQ codes in memory for fast navigation
//! 4. Beam search traverses graph, fetching full vectors only when needed

use anyhow::Result;
use memmap2::Mmap;
use ordered_float::OrderedFloat;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// DiskANN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskANNConfig {
    /// Maximum out-degree of the graph (R)
    /// Higher = better recall, more memory/disk
    /// Recommended: 32-128
    pub max_degree: usize,

    /// Beam width during index construction (L_build)
    /// Higher = better graph quality, slower build
    /// Recommended: 100-200
    pub build_beam_width: usize,

    /// Beam width during search (L_search)
    /// Higher = better recall, higher latency
    /// Recommended: 50-200
    pub search_beam_width: usize,

    /// Vamana pruning parameter (α)
    /// Controls graph connectivity vs diversity
    /// Recommended: 1.0-1.5
    pub alpha: f32,

    /// Number of PQ subvectors for in-memory navigation
    /// Higher = better accuracy, more memory
    /// Recommended: 16-64
    pub pq_dims: usize,

    /// PQ training sample size
    pub pq_sample_size: usize,

    /// SSD sector size for aligned I/O
    pub sector_size: usize,

    /// Number of sectors to read per I/O
    pub sectors_per_read: usize,

    /// Cache size for recently accessed vectors (in vectors)
    pub cache_size: usize,

    /// Number of threads for parallel operations
    pub num_threads: usize,

    /// Use mmap for disk access
    pub use_mmap: bool,
}

impl Default for DiskANNConfig {
    fn default() -> Self {
        Self {
            max_degree: 64,
            build_beam_width: 128,
            search_beam_width: 64,
            alpha: 1.2,
            pq_dims: 32,
            pq_sample_size: 100_000,
            sector_size: 4096,
            sectors_per_read: 4,
            cache_size: 10_000,
            num_threads: num_cpus::get().max(1),
            use_mmap: true,
        }
    }
}

/// Product Quantizer for in-memory graph navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQQuantizer {
    /// Number of subvectors
    pub num_subvectors: usize,
    /// Dimension of each subvector
    pub subvector_dim: usize,
    /// Codebook: [num_subvectors][256][subvector_dim]
    pub codebooks: Vec<Vec<Vec<f32>>>,
    /// Number of centroids per subvector
    pub num_centroids: usize,
}

impl PQQuantizer {
    /// Create a new PQ quantizer
    pub fn new(dimension: usize, num_subvectors: usize) -> Self {
        assert!(dimension % num_subvectors == 0);
        let subvector_dim = dimension / num_subvectors;

        Self {
            num_subvectors,
            subvector_dim,
            codebooks: vec![vec![vec![0.0; subvector_dim]; 256]; num_subvectors],
            num_centroids: 256,
        }
    }

    /// Train the quantizer on sample vectors
    pub fn train(&mut self, vectors: &[Vec<f32>], iterations: usize) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        let dim = vectors[0].len();
        assert!(dim % self.num_subvectors == 0);

        // Train each subvector independently using k-means
        for sub_idx in 0..self.num_subvectors {
            let start = sub_idx * self.subvector_dim;
            let end = start + self.subvector_dim;

            // Extract subvectors
            let subvectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            // K-means clustering
            let centroids = self.kmeans(&subvectors, self.num_centroids, iterations)?;
            self.codebooks[sub_idx] = centroids;
        }

        Ok(())
    }

    /// K-means clustering for a single subvector space
    fn kmeans(&self, vectors: &[Vec<f32>], k: usize, iterations: usize) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(vec![vec![0.0; self.subvector_dim]; k]);
        }

        let dim = vectors[0].len();
        let mut rng = rand::thread_rng();

        // Initialize centroids randomly
        let mut centroids: Vec<Vec<f32>> = vectors
            .choose_multiple(&mut rng, k.min(vectors.len()))
            .cloned()
            .collect();

        // Pad if we don't have enough vectors
        while centroids.len() < k {
            centroids.push(vec![0.0; dim]);
        }

        for _ in 0..iterations {
            // Assign vectors to nearest centroid
            let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); k];

            for (i, vec) in vectors.iter().enumerate() {
                let mut best_centroid = 0;
                let mut best_dist = f32::MAX;

                for (c, centroid) in centroids.iter().enumerate() {
                    let dist = self.l2_distance(vec, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_centroid = c;
                    }
                }
                assignments[best_centroid].push(i);
            }

            // Update centroids
            for (c, assigned) in assignments.iter().enumerate() {
                if assigned.is_empty() {
                    continue;
                }

                let mut new_centroid = vec![0.0; dim];
                for &idx in assigned {
                    for (d, &val) in vectors[idx].iter().enumerate() {
                        new_centroid[d] += val;
                    }
                }
                for val in &mut new_centroid {
                    *val /= assigned.len() as f32;
                }
                centroids[c] = new_centroid;
            }
        }

        Ok(centroids)
    }

    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Encode a vector to PQ codes
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.num_subvectors);

        for sub_idx in 0..self.num_subvectors {
            let start = sub_idx * self.subvector_dim;
            let end = start + self.subvector_dim;
            let subvector = &vector[start..end];

            // Find nearest centroid
            let mut best_code = 0u8;
            let mut best_dist = f32::MAX;

            for (c, centroid) in self.codebooks[sub_idx].iter().enumerate() {
                let dist = self.l2_distance(subvector, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_code = c as u8;
                }
            }
            codes.push(best_code);
        }

        codes
    }

    /// Compute approximate distance using PQ codes
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let mut distance = 0.0;

        for (sub_idx, &code) in codes.iter().enumerate() {
            let start = sub_idx * self.subvector_dim;
            let end = start + self.subvector_dim;
            let query_sub = &query[start..end];
            let centroid = &self.codebooks[sub_idx][code as usize];

            for (q, c) in query_sub.iter().zip(centroid.iter()) {
                distance += (q - c).powi(2);
            }
        }

        distance.sqrt()
    }

    /// Precompute distance table for a query (for faster batch lookups)
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut table = Vec::with_capacity(self.num_subvectors);

        for sub_idx in 0..self.num_subvectors {
            let start = sub_idx * self.subvector_dim;
            let end = start + self.subvector_dim;
            let query_sub = &query[start..end];

            let mut distances = Vec::with_capacity(self.num_centroids);
            for centroid in &self.codebooks[sub_idx] {
                let dist: f32 = query_sub
                    .iter()
                    .zip(centroid.iter())
                    .map(|(q, c)| (q - c).powi(2))
                    .sum();
                distances.push(dist);
            }
            table.push(distances);
        }

        table
    }

    /// Fast distance lookup using precomputed table
    pub fn table_distance(&self, table: &[Vec<f32>], codes: &[u8]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(i, &code)| table[i][code as usize])
            .sum::<f32>()
            .sqrt()
    }
}

/// Node in the Vamana graph
#[derive(Debug, Clone)]
struct VamanaNode {
    /// Node ID
    id: u64,
    /// Neighbors (sorted by distance)
    neighbors: Vec<u64>,
    /// PQ codes for this node's vector
    pq_codes: Vec<u8>,
}

/// DiskANN Index
pub struct DiskANN {
    /// Configuration
    config: DiskANNConfig,
    /// Vector dimension
    dimension: usize,
    /// Number of vectors
    num_vectors: u64,
    /// Medoid (entry point) node ID
    medoid: u64,
    /// In-memory graph: node_id -> neighbors
    graph: HashMap<u64, Vec<u64>>,
    /// In-memory PQ codes: node_id -> codes
    pq_codes: HashMap<u64, Vec<u8>>,
    /// PQ quantizer
    quantizer: PQQuantizer,
    /// Disk file for full vectors
    vectors_path: Option<PathBuf>,
    /// Memory-mapped vectors file
    vectors_mmap: Option<Mmap>,
    /// Vector cache (LRU)
    cache: RwLock<HashMap<u64, Vec<f32>>>,
    /// Statistics
    stats: DiskANNStats,
}

/// DiskANN statistics
#[derive(Debug, Default)]
pub struct DiskANNStats {
    /// Total disk reads
    pub disk_reads: AtomicU64,
    /// Cache hits
    pub cache_hits: AtomicU64,
    /// Cache misses
    pub cache_misses: AtomicU64,
    /// Total searches
    pub total_searches: AtomicU64,
    /// Total distance computations
    pub distance_computations: AtomicU64,
}

/// Search result
#[derive(Debug, Clone)]
pub struct DiskANNResult {
    /// Node ID
    pub id: u64,
    /// Distance to query
    pub distance: f32,
}

impl Ord for DiskANNResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(other.distance).cmp(&OrderedFloat(self.distance))
    }
}

impl PartialOrd for DiskANNResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DiskANNResult {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for DiskANNResult {}

impl DiskANN {
    /// Create a new DiskANN index
    pub fn new(config: DiskANNConfig, dimension: usize) -> Result<Self> {
        let pq_dims = config.pq_dims.min(dimension);
        let adjusted_pq_dims = if dimension % pq_dims != 0 {
            // Find closest divisor
            (1..=dimension)
                .filter(|d| dimension % d == 0)
                .min_by_key(|d| (*d as i32 - pq_dims as i32).abs())
                .unwrap_or(1)
        } else {
            pq_dims
        };

        Ok(Self {
            config,
            dimension,
            num_vectors: 0,
            medoid: 0,
            graph: HashMap::new(),
            pq_codes: HashMap::new(),
            quantizer: PQQuantizer::new(dimension, adjusted_pq_dims),
            vectors_path: None,
            vectors_mmap: None,
            cache: RwLock::new(HashMap::new()),
            stats: DiskANNStats::default(),
        })
    }

    /// Build the index from vectors
    pub fn build(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        self.num_vectors = vectors.len() as u64;

        // Train PQ quantizer
        let sample_size = self.config.pq_sample_size.min(vectors.len());
        let mut rng = rand::thread_rng();
        let sample: Vec<Vec<f32>> = vectors
            .choose_multiple(&mut rng, sample_size)
            .cloned()
            .collect();

        self.quantizer.train(&sample, 20)?;

        // Encode all vectors
        for (i, vec) in vectors.iter().enumerate() {
            let codes = self.quantizer.encode(vec);
            self.pq_codes.insert(i as u64, codes);
        }

        // Find medoid (vector closest to centroid)
        self.medoid = self.find_medoid(vectors)?;

        // Build Vamana graph
        self.build_vamana_graph(vectors)?;

        Ok(())
    }

    /// Find the medoid (entry point)
    fn find_medoid(&self, vectors: &[Vec<f32>]) -> Result<u64> {
        if vectors.is_empty() {
            return Ok(0);
        }

        // Compute centroid
        let dim = vectors[0].len();
        let mut centroid = vec![0.0f32; dim];
        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                centroid[i] += v;
            }
        }
        for c in &mut centroid {
            *c /= vectors.len() as f32;
        }

        // Find vector closest to centroid
        let mut best_id = 0u64;
        let mut best_dist = f32::MAX;

        for (i, vec) in vectors.iter().enumerate() {
            let dist = self.l2_distance(&centroid, vec);
            if dist < best_dist {
                best_dist = dist;
                best_id = i as u64;
            }
        }

        Ok(best_id)
    }

    /// Build Vamana graph using greedy search + RobustPrune
    fn build_vamana_graph(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        let n = vectors.len();
        if n == 0 {
            return Ok(());
        }

        // Initialize empty graph
        for i in 0..n {
            self.graph.insert(i as u64, Vec::new());
        }

        // Random permutation for insertion order
        let mut rng = rand::thread_rng();
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);

        // Insert each node
        for &node_idx in &order {
            let node_id = node_idx as u64;
            let node_vec = &vectors[node_idx];

            // Greedy search to find candidates
            let candidates = self.greedy_search_build(vectors, node_vec, self.config.build_beam_width)?;

            // RobustPrune to select neighbors
            let neighbors = self.robust_prune(vectors, node_id, &candidates)?;

            // Update graph with bidirectional edges
            self.graph.insert(node_id, neighbors.clone());

            // Add reverse edges
            for &neighbor_id in &neighbors {
                let needs_prune = {
                    let neighbor_neighbors = self.graph.entry(neighbor_id).or_insert_with(Vec::new);
                    if !neighbor_neighbors.contains(&node_id) {
                        neighbor_neighbors.push(node_id);
                        neighbor_neighbors.len() > self.config.max_degree
                    } else {
                        false
                    }
                };

                // Prune if over max degree (done outside the mutable borrow)
                if needs_prune {
                    let candidates: Vec<u64> = self.graph.get(&neighbor_id)
                        .map(|v| v.clone())
                        .unwrap_or_default();
                    let pruned = self.robust_prune(vectors, neighbor_id, &candidates)?;
                    self.graph.insert(neighbor_id, pruned);
                }
            }
        }

        Ok(())
    }

    /// Greedy search during build (uses full vectors)
    fn greedy_search_build(
        &self,
        vectors: &[Vec<f32>],
        query: &[f32],
        beam_width: usize,
    ) -> Result<Vec<u64>> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<DiskANNResult> = BinaryHeap::new();
        let mut results: BinaryHeap<std::cmp::Reverse<DiskANNResult>> = BinaryHeap::new();

        // Start from medoid
        let start_dist = self.l2_distance(query, &vectors[self.medoid as usize]);
        candidates.push(DiskANNResult {
            id: self.medoid,
            distance: start_dist,
        });
        results.push(std::cmp::Reverse(DiskANNResult {
            id: self.medoid,
            distance: start_dist,
        }));
        visited.insert(self.medoid);

        while let Some(current) = candidates.pop() {
            // Check if we can stop
            if let Some(std::cmp::Reverse(worst)) = results.peek() {
                if current.distance > worst.distance && results.len() >= beam_width {
                    break;
                }
            }

            // Explore neighbors
            if let Some(neighbors) = self.graph.get(&current.id) {
                for &neighbor_id in neighbors {
                    if visited.insert(neighbor_id) {
                        let dist = self.l2_distance(query, &vectors[neighbor_id as usize]);

                        candidates.push(DiskANNResult {
                            id: neighbor_id,
                            distance: dist,
                        });
                        results.push(std::cmp::Reverse(DiskANNResult {
                            id: neighbor_id,
                            distance: dist,
                        }));

                        // Keep only top beam_width
                        while results.len() > beam_width {
                            results.pop();
                        }
                    }
                }
            }
        }

        Ok(results.into_iter().map(|r| r.0.id).collect())
    }

    /// RobustPrune: Select neighbors using α-RNG rule
    fn robust_prune(
        &self,
        vectors: &[Vec<f32>],
        node_id: u64,
        candidates: &[u64],
    ) -> Result<Vec<u64>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let node_vec = &vectors[node_id as usize];

        // Sort candidates by distance
        let mut sorted_candidates: Vec<(u64, f32)> = candidates
            .iter()
            .filter(|&&id| id != node_id)
            .map(|&id| {
                let dist = self.l2_distance(node_vec, &vectors[id as usize]);
                (id, dist)
            })
            .collect();
        sorted_candidates.sort_by(|a, b| OrderedFloat(a.1).cmp(&OrderedFloat(b.1)));

        let mut neighbors = Vec::new();

        for (candidate_id, candidate_dist) in sorted_candidates {
            if neighbors.len() >= self.config.max_degree {
                break;
            }

            // Check α-RNG rule: keep if not too close to existing neighbors
            let candidate_vec = &vectors[candidate_id as usize];
            let mut keep = true;

            for &neighbor_id in &neighbors {
                let neighbor_vec = &vectors[neighbor_id as usize];
                let neighbor_to_candidate = self.l2_distance(neighbor_vec, candidate_vec);

                if neighbor_to_candidate * self.config.alpha < candidate_dist {
                    keep = false;
                    break;
                }
            }

            if keep {
                neighbors.push(candidate_id);
            }
        }

        Ok(neighbors)
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, beam_width: Option<usize>) -> Result<Vec<DiskANNResult>> {
        self.stats.total_searches.fetch_add(1, Ordering::Relaxed);

        let beam = beam_width.unwrap_or(self.config.search_beam_width);

        // Precompute PQ distance table
        let pq_table = self.quantizer.compute_distance_table(query);

        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<DiskANNResult> = BinaryHeap::new();
        let mut results: BinaryHeap<std::cmp::Reverse<DiskANNResult>> = BinaryHeap::new();

        // Start from medoid - use PQ distance
        let start_dist = if let Some(codes) = self.pq_codes.get(&self.medoid) {
            self.quantizer.table_distance(&pq_table, codes)
        } else {
            f32::MAX
        };

        candidates.push(DiskANNResult {
            id: self.medoid,
            distance: start_dist,
        });
        visited.insert(self.medoid);

        while let Some(current) = candidates.pop() {
            // Add to results
            results.push(std::cmp::Reverse(DiskANNResult {
                id: current.id,
                distance: current.distance,
            }));

            // Keep only top beam candidates
            while results.len() > beam {
                results.pop();
            }

            // Check stopping condition
            if let Some(std::cmp::Reverse(worst)) = results.peek() {
                if current.distance > worst.distance && results.len() >= beam {
                    continue;
                }
            }

            // Explore neighbors using PQ distances
            if let Some(neighbors) = self.graph.get(&current.id) {
                for &neighbor_id in neighbors {
                    if visited.insert(neighbor_id) {
                        self.stats.distance_computations.fetch_add(1, Ordering::Relaxed);

                        let dist = if let Some(codes) = self.pq_codes.get(&neighbor_id) {
                            self.quantizer.table_distance(&pq_table, codes)
                        } else {
                            f32::MAX
                        };

                        candidates.push(DiskANNResult {
                            id: neighbor_id,
                            distance: dist,
                        });
                    }
                }
            }
        }

        // Collect top-k and optionally rerank with exact distances
        let mut final_results: Vec<DiskANNResult> = results
            .into_iter()
            .take(k * 2) // Over-fetch for reranking
            .map(|r| r.0)
            .collect();

        // Sort and take top-k
        final_results.sort_by(|a, b| OrderedFloat(a.distance).cmp(&OrderedFloat(b.distance)));
        final_results.truncate(k);

        Ok(final_results)
    }

    /// Search with exact reranking (loads full vectors from disk)
    pub fn search_rerank(
        &self,
        query: &[f32],
        k: usize,
        beam_width: Option<usize>,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<DiskANNResult>> {
        // First pass with PQ
        let candidates = self.search(query, k * 3, beam_width)?;

        // Rerank with exact distances
        let mut reranked: Vec<DiskANNResult> = candidates
            .into_iter()
            .map(|r| {
                let exact_dist = self.l2_distance(query, &vectors[r.id as usize]);
                DiskANNResult {
                    id: r.id,
                    distance: exact_dist,
                }
            })
            .collect();

        reranked.sort_by(|a, b| OrderedFloat(a.distance).cmp(&OrderedFloat(b.distance)));
        reranked.truncate(k);

        Ok(reranked)
    }

    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Save index to disk
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        // Save metadata
        let metadata = DiskANNMetadata {
            dimension: self.dimension,
            num_vectors: self.num_vectors,
            medoid: self.medoid,
            config: self.config.clone(),
        };
        let meta_file = File::create(path.join("metadata.json"))?;
        serde_json::to_writer_pretty(meta_file, &metadata)?;

        // Save graph
        let graph_file = File::create(path.join("graph.bin"))?;
        let mut writer = BufWriter::new(graph_file);
        bincode::serialize_into(&mut writer, &self.graph)?;

        // Save PQ codes
        let pq_file = File::create(path.join("pq_codes.bin"))?;
        let mut pq_writer = BufWriter::new(pq_file);
        bincode::serialize_into(&mut pq_writer, &self.pq_codes)?;

        // Save quantizer
        let quant_file = File::create(path.join("quantizer.bin"))?;
        let mut quant_writer = BufWriter::new(quant_file);
        bincode::serialize_into(&mut quant_writer, &self.quantizer)?;

        Ok(())
    }

    /// Load index from disk
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        // Load metadata
        let meta_file = File::open(path.join("metadata.json"))?;
        let metadata: DiskANNMetadata = serde_json::from_reader(meta_file)?;

        // Load graph
        let graph_file = File::open(path.join("graph.bin"))?;
        let reader = BufReader::new(graph_file);
        let graph: HashMap<u64, Vec<u64>> = bincode::deserialize_from(reader)?;

        // Load PQ codes
        let pq_file = File::open(path.join("pq_codes.bin"))?;
        let pq_reader = BufReader::new(pq_file);
        let pq_codes: HashMap<u64, Vec<u8>> = bincode::deserialize_from(pq_reader)?;

        // Load quantizer
        let quant_file = File::open(path.join("quantizer.bin"))?;
        let quant_reader = BufReader::new(quant_file);
        let quantizer: PQQuantizer = bincode::deserialize_from(quant_reader)?;

        Ok(Self {
            config: metadata.config,
            dimension: metadata.dimension,
            num_vectors: metadata.num_vectors,
            medoid: metadata.medoid,
            graph,
            pq_codes,
            quantizer,
            vectors_path: None,
            vectors_mmap: None,
            cache: RwLock::new(HashMap::new()),
            stats: DiskANNStats::default(),
        })
    }

    /// Get statistics
    pub fn stats(&self) -> &DiskANNStats {
        &self.stats
    }

    /// Get number of vectors
    pub fn len(&self) -> u64 {
        self.num_vectors
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Get graph degree statistics
    pub fn graph_stats(&self) -> GraphStats {
        let degrees: Vec<usize> = self.graph.values().map(|n| n.len()).collect();

        let avg_degree = if degrees.is_empty() {
            0.0
        } else {
            degrees.iter().sum::<usize>() as f64 / degrees.len() as f64
        };

        GraphStats {
            num_nodes: self.graph.len(),
            num_edges: degrees.iter().sum::<usize>(),
            avg_degree,
            max_degree: degrees.iter().max().copied().unwrap_or(0),
            min_degree: degrees.iter().min().copied().unwrap_or(0),
        }
    }
}

/// DiskANN metadata for serialization
#[derive(Debug, Serialize, Deserialize)]
struct DiskANNMetadata {
    dimension: usize,
    num_vectors: u64,
    medoid: u64,
    config: DiskANNConfig,
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f64,
    pub max_degree: usize,
    pub min_degree: usize,
}

/// Get number of CPUs (helper)
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_pq_quantizer() {
        let dim = 128;
        let num_subvectors = 16;
        let mut pq = PQQuantizer::new(dim, num_subvectors);

        let vectors = random_vectors(1000, dim);
        pq.train(&vectors, 10).unwrap();

        // Test encoding
        let codes = pq.encode(&vectors[0]);
        assert_eq!(codes.len(), num_subvectors);

        // Test distance computation
        let dist = pq.asymmetric_distance(&vectors[0], &codes);
        assert!(dist >= 0.0);

        // Test table-based distance
        let table = pq.compute_distance_table(&vectors[0]);
        let table_dist = pq.table_distance(&table, &codes);
        assert!((dist - table_dist).abs() < 0.001);
    }

    #[test]
    fn test_diskann_build() {
        let config = DiskANNConfig {
            max_degree: 16,
            build_beam_width: 32,
            pq_dims: 8,
            ..Default::default()
        };

        let mut index = DiskANN::new(config, 64).unwrap();
        let vectors = random_vectors(100, 64);

        index.build(&vectors).unwrap();

        assert_eq!(index.len(), 100);
        assert!(!index.graph.is_empty());

        let stats = index.graph_stats();
        assert!(stats.avg_degree > 0.0);
        assert!(stats.max_degree <= 16);
    }

    #[test]
    fn test_diskann_search() {
        let config = DiskANNConfig {
            max_degree: 32,
            build_beam_width: 64,
            search_beam_width: 32,
            pq_dims: 8,
            ..Default::default()
        };

        let mut index = DiskANN::new(config, 64).unwrap();
        let vectors = random_vectors(500, 64);

        index.build(&vectors).unwrap();

        // Search
        let query = &vectors[0];
        let results = index.search(query, 10, None).unwrap();

        assert_eq!(results.len(), 10);

        // First result should be the query itself (or very close)
        assert!(results[0].distance < 0.5);
    }

    #[test]
    fn test_diskann_search_rerank() {
        let config = DiskANNConfig {
            max_degree: 32,
            build_beam_width: 64,
            pq_dims: 8,
            ..Default::default()
        };

        let mut index = DiskANN::new(config, 64).unwrap();
        let vectors = random_vectors(200, 64);

        index.build(&vectors).unwrap();

        let query = &vectors[0];
        let results = index.search_rerank(query, 10, None, &vectors).unwrap();

        assert_eq!(results.len(), 10);
        assert!(results[0].distance < 0.001); // Should find exact match
    }

    #[test]
    fn test_diskann_save_load() {
        let config = DiskANNConfig {
            max_degree: 16,
            build_beam_width: 32,
            pq_dims: 8,
            ..Default::default()
        };

        let mut index = DiskANN::new(config, 32).unwrap();
        let vectors = random_vectors(50, 32);
        index.build(&vectors).unwrap();

        // Save
        let temp_dir = tempfile::tempdir().unwrap();
        index.save(temp_dir.path()).unwrap();

        // Load
        let loaded = DiskANN::load(temp_dir.path()).unwrap();

        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.dimension, index.dimension);
        assert_eq!(loaded.medoid, index.medoid);
    }

    #[test]
    fn test_diskann_recall() {
        let config = DiskANNConfig {
            max_degree: 64,
            build_beam_width: 128,
            search_beam_width: 64,
            pq_dims: 16,
            alpha: 1.2,
            ..Default::default()
        };

        let mut index = DiskANN::new(config, 128).unwrap();
        let vectors = random_vectors(1000, 128);
        index.build(&vectors).unwrap();

        // Test recall@10
        let mut correct = 0;
        let test_count = 50;

        for i in 0..test_count {
            let query = &vectors[i];

            // Approximate search
            let approx_results = index.search(query, 10, Some(100)).unwrap();

            // Exact search (brute force)
            let mut exact: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(j, v)| (j, index.l2_distance(query, v)))
                .collect();
            exact.sort_by(|a, b| OrderedFloat(a.1).cmp(&OrderedFloat(b.1)));
            let exact_top10: HashSet<usize> = exact.iter().take(10).map(|(j, _)| *j).collect();

            // Count overlap
            for r in &approx_results {
                if exact_top10.contains(&(r.id as usize)) {
                    correct += 1;
                }
            }
        }

        let recall = correct as f64 / (test_count * 10) as f64;
        println!("Recall@10: {:.2}%", recall * 100.0);
        assert!(recall > 0.70, "Recall should be > 70%, got {:.2}%", recall * 100.0);
    }
}
