// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

use super::types::{Distance, Id};
use anyhow::{Result, anyhow};
use hnsw_rs::api::AnnT;
use hnsw_rs::prelude::{
    DistCosine, DistDot, DistL1, DistL2, Distance as HnswDistance, Hnsw, Neighbour,
};
use std::collections::HashMap;
use std::path::Path;

/// Enumerate the index when the caller asks for every raw node. In a small
/// randomly layered HNSW graph, approximate traversal can omit a node even
/// when `k` and `ef` exceed the population. That is not a useful approximation
/// for a full-population request (including over-fetches used for deletions).
fn search_hnsw<D: HnswDistance<f32> + Send + Sync>(
    hnsw: &Hnsw<'_, f32, D>,
    vector: &[f32],
    k: usize,
    ef: usize,
) -> Vec<Neighbour> {
    if k == 0 || hnsw.get_nb_point() == 0 {
        return Vec::new();
    }
    if k < hnsw.get_nb_point() {
        return hnsw.search(vector, k, ef);
    }
    let mut neighbors: Vec<_> = hnsw
        .get_point_indexation()
        .into_iter()
        .map(|point| {
            Neighbour::new(
                point.get_origin_id(),
                hnsw.get_distance().eval(vector, point.get_v()),
                point.get_point_id(),
            )
        })
        .collect();
    neighbors.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.d_id.cmp(&b.d_id)));
    neighbors
}

// ============================================================================
// CUSTOM DISTANCE METRICS FOR F32 VECTORS
// ============================================================================

/// Hamming distance for f32 vectors - counts differing elements
/// Values are compared with epsilon tolerance, then normalized by vector length
#[derive(Default, Clone, Copy)]
pub struct DistHammingF32;

impl HnswDistance<f32> for DistHammingF32 {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        let threshold = 0.5;
        let diff_count: usize = a
            .iter()
            .zip(b.iter())
            .filter(|&(x, y)| {
                // Convert to binary: > threshold = 1, <= threshold = 0
                (*x > threshold) != (*y > threshold)
            })
            .count();
        diff_count as f32 / a.len() as f32
    }
}

/// Jaccard distance for f32 vectors - measures set dissimilarity
/// Treats vectors as presence/absence using 0.5 threshold
#[derive(Default, Clone, Copy)]
pub struct DistJaccardF32;

impl HnswDistance<f32> for DistJaccardF32 {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        let threshold = 0.5;
        let mut intersection = 0usize;
        let mut union = 0usize;

        for (&x, &y) in a.iter().zip(b.iter()) {
            let x_present = x > threshold;
            let y_present = y > threshold;
            if x_present || y_present {
                union += 1;
                if x_present && y_present {
                    intersection += 1;
                }
            }
        }

        if union == 0 {
            0.0
        } else {
            1.0 - (intersection as f32 / union as f32)
        }
    }
}

/// Chebyshev distance (L∞) - maximum absolute difference between elements
#[derive(Default, Clone, Copy)]
pub struct DistChebyshev;

impl HnswDistance<f32> for DistChebyshev {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, |max, diff| max.max(diff))
    }
}

/// Canberra distance - weighted Manhattan distance, sensitive to values near zero
#[derive(Default, Clone, Copy)]
pub struct DistCanberra;

impl HnswDistance<f32> for DistCanberra {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = (x - y).abs();
                let sum = x.abs() + y.abs();
                if sum > f32::EPSILON { diff / sum } else { 0.0 }
            })
            .sum()
    }
}

/// Bray-Curtis dissimilarity - ecological distance for compositional data
#[derive(Default, Clone, Copy)]
pub struct DistBrayCurtis;

impl HnswDistance<f32> for DistBrayCurtis {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        let (diff_sum, total_sum) = a
            .iter()
            .zip(b.iter())
            .fold((0.0_f32, 0.0_f32), |(d, t), (x, y)| {
                (d + (x - y).abs(), t + x.abs() + y.abs())
            });
        if total_sum > f32::EPSILON {
            diff_sum / total_sum
        } else {
            0.0
        }
    }
}

/// Default HNSW parameters
pub const DEFAULT_HNSW_M: usize = 16;
pub const DEFAULT_HNSW_EF_CONSTRUCTION: usize = 200;
pub const DEFAULT_MAX_ELEMENTS: usize = 100_000;

/// HNSW configuration parameters
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    /// Number of connections per layer (M parameter)
    pub m: usize,
    /// Size of dynamic candidate list during construction
    pub ef_construction: usize,
    /// Maximum number of elements the index can hold
    pub max_elements: usize,
}

impl HnswConfig {
    /// Create a new HNSW configuration with default values
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            m: DEFAULT_HNSW_M,
            ef_construction: DEFAULT_HNSW_EF_CONSTRUCTION,
            max_elements: DEFAULT_MAX_ELEMENTS,
        }
    }

    /// Set the M parameter (connections per layer)
    #[inline]
    #[must_use]
    pub const fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set the ef_construction parameter
    #[inline]
    #[must_use]
    pub const fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set the maximum elements
    #[inline]
    #[must_use]
    pub const fn with_max_elements(mut self, max: usize) -> Self {
        self.max_elements = max;
        self
    }
}

impl Default for HnswConfig {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// Enum to hold different HNSW instances for different distance metrics
enum HnswInstance {
    Cosine(Hnsw<'static, f32, DistCosine>),
    Euclidean(Hnsw<'static, f32, DistL2>),
    DotProduct(Hnsw<'static, f32, DistDot>),
    Manhattan(Hnsw<'static, f32, DistL1>),
    Hamming(Hnsw<'static, f32, DistHammingF32>),
    Jaccard(Hnsw<'static, f32, DistJaccardF32>),
    Chebyshev(Hnsw<'static, f32, DistChebyshev>),
    Canberra(Hnsw<'static, f32, DistCanberra>),
    BrayCurtis(Hnsw<'static, f32, DistBrayCurtis>),
}

pub struct HnswBackend {
    hnsw: HnswInstance,
    id_to_idx: HashMap<Id, usize>,
    idx_to_id: HashMap<usize, Id>,
    next_idx: usize,
    dimension: usize,
    distance: Distance,
    config: HnswConfig,
}

impl HnswBackend {
    /// Create a new HNSW backend with default configuration
    pub fn new(dimension: usize, distance: Distance) -> Result<Self> {
        Self::with_config(dimension, distance, HnswConfig::default())
    }

    /// Create a new HNSW backend with custom configuration
    pub fn with_config(dimension: usize, distance: Distance, config: HnswConfig) -> Result<Self> {
        let hnsw = match distance {
            Distance::Cosine => HnswInstance::Cosine(Hnsw::<f32, DistCosine>::new(
                config.m,               // max_nb_connection
                config.max_elements,    // max_elements
                config.m,               // max_layer (typically same as M)
                config.ef_construction, // ef_construction
                DistCosine,
            )),
            Distance::Euclidean => HnswInstance::Euclidean(Hnsw::<f32, DistL2>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistL2,
            )),
            Distance::DotProduct => HnswInstance::DotProduct(Hnsw::<f32, DistDot>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistDot,
            )),
            Distance::Manhattan => HnswInstance::Manhattan(Hnsw::<f32, DistL1>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistL1,
            )),
            Distance::Hamming => HnswInstance::Hamming(Hnsw::<f32, DistHammingF32>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistHammingF32,
            )),
            Distance::Jaccard => HnswInstance::Jaccard(Hnsw::<f32, DistJaccardF32>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistJaccardF32,
            )),
            Distance::Chebyshev => HnswInstance::Chebyshev(Hnsw::<f32, DistChebyshev>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistChebyshev,
            )),
            Distance::Canberra => HnswInstance::Canberra(Hnsw::<f32, DistCanberra>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistCanberra,
            )),
            Distance::BrayCurtis => HnswInstance::BrayCurtis(Hnsw::<f32, DistBrayCurtis>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistBrayCurtis,
            )),
        };

        Ok(Self {
            hnsw,
            id_to_idx: HashMap::new(),
            idx_to_id: HashMap::new(),
            next_idx: 0,
            dimension,
            distance,
            config,
        })
    }

    /// Get the current HNSW configuration
    #[inline]
    #[must_use]
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Get the maximum capacity of this index
    #[inline]
    #[must_use]
    pub fn max_capacity(&self) -> usize {
        self.config.max_elements
    }

    /// Get the current number of vectors in the index
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.id_to_idx.len()
    }

    /// Check if the index is empty
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id_to_idx.is_empty()
    }

    /// Check if the index is at capacity
    #[inline]
    #[must_use]
    pub fn is_at_capacity(&self) -> bool {
        self.id_to_idx.len() >= self.config.max_elements
    }

    pub fn insert(&mut self, id: Id, vector: &[f32]) -> Result<()> {
        if self.dimension > 0 && vector.len() != self.dimension {
            return Err(anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            ));
        }

        // Check capacity before inserting new vectors (not updates)
        let is_update = self.id_to_idx.contains_key(&id);
        if !is_update && self.id_to_idx.len() >= self.config.max_elements {
            return Err(anyhow!(
                "HNSW index is at capacity ({} vectors). Cannot insert more vectors. \
                 Consider increasing max_elements in HnswConfig or using a new index.",
                self.config.max_elements
            ));
        }

        // Remove old entry if exists
        if let Some(&old_idx) = self.id_to_idx.get(&id) {
            self.idx_to_id.remove(&old_idx);
        }

        let idx = self.next_idx;
        self.next_idx += 1;

        // Insert into appropriate HNSW instance
        match &mut self.hnsw {
            HnswInstance::Cosine(h) => h.insert((vector, idx)),
            HnswInstance::Euclidean(h) => h.insert((vector, idx)),
            HnswInstance::DotProduct(h) => h.insert((vector, idx)),
            HnswInstance::Manhattan(h) => h.insert((vector, idx)),
            HnswInstance::Hamming(h) => h.insert((vector, idx)),
            HnswInstance::Jaccard(h) => h.insert((vector, idx)),
            HnswInstance::Chebyshev(h) => h.insert((vector, idx)),
            HnswInstance::Canberra(h) => h.insert((vector, idx)),
            HnswInstance::BrayCurtis(h) => h.insert((vector, idx)),
        }

        self.id_to_idx.insert(id.clone(), idx);
        self.idx_to_id.insert(idx, id);

        Ok(())
    }

    /// Remove a vector's ID mapping from the backend.
    ///
    /// **Note**: This creates a "ghost node" in the HNSW graph. The underlying `hnsw_rs`
    /// library does not support true node deletion. This method only removes the ID
    /// mapping, leaving the vector data in the graph structure. The ghost node:
    /// - Cannot be found by ID lookup
    /// - Is excluded from search results (filtered by the ID mapping)
    /// - Still occupies memory in the graph
    ///
    /// To reclaim memory, use [`optimize()`](Self::optimize) to rebuild the graph.
    pub fn remove(&mut self, id: &str) -> Result<()> {
        if let Some(&idx) = self.id_to_idx.get(id) {
            // Remove ID mapping only - the vector remains as a ghost node in HNSW
            self.id_to_idx.remove(id);
            self.idx_to_id.remove(&idx);
            Ok(())
        } else {
            Err(anyhow!("ID not found: {}", id))
        }
    }

    #[inline]
    pub fn search(&self, vector: &[f32], k: usize) -> Vec<(Id, f32)> {
        if self.id_to_idx.is_empty() {
            return Vec::new();
        }

        let neighbors = match &self.hnsw {
            HnswInstance::Cosine(h) => search_hnsw(h, vector, k, 30),
            HnswInstance::Euclidean(h) => search_hnsw(h, vector, k, 30),
            HnswInstance::DotProduct(h) => search_hnsw(h, vector, k, 30),
            HnswInstance::Manhattan(h) => search_hnsw(h, vector, k, 30),
            HnswInstance::Hamming(h) => search_hnsw(h, vector, k, 30),
            HnswInstance::Jaccard(h) => search_hnsw(h, vector, k, 30),
            HnswInstance::Chebyshev(h) => search_hnsw(h, vector, k, 30),
            HnswInstance::Canberra(h) => search_hnsw(h, vector, k, 30),
            HnswInstance::BrayCurtis(h) => search_hnsw(h, vector, k, 30),
        };

        neighbors
            .into_iter()
            .filter_map(|neighbor| {
                let idx = neighbor.d_id;
                self.idx_to_id.get(&idx).map(|id| {
                    let score = match self.distance {
                        // Similarity metrics: higher raw value = more similar
                        Distance::Cosine | Distance::DotProduct => neighbor.distance,
                        // Distance metrics: lower raw value = more similar, invert for score
                        Distance::Euclidean
                        | Distance::Manhattan
                        | Distance::Chebyshev
                        | Distance::Canberra => 1.0 / (1.0 + neighbor.distance),
                        // Normalized distance metrics [0,1]: convert to similarity
                        Distance::Hamming | Distance::Jaccard | Distance::BrayCurtis => {
                            1.0 - neighbor.distance.clamp(0.0, 1.0)
                        },
                    };
                    (id.clone(), score)
                })
            })
            .collect()
    }

    pub fn save_index<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path_ref = path.as_ref();
        let parent = path_ref
            .parent()
            .ok_or_else(|| anyhow!("Invalid path: no parent directory"))?;
        let file_name = path_ref
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow!("Invalid path: no file name"))?;

        let _ = match &self.hnsw {
            HnswInstance::Cosine(h) => h.file_dump(parent, file_name)?,
            HnswInstance::Euclidean(h) => h.file_dump(parent, file_name)?,
            HnswInstance::DotProduct(h) => h.file_dump(parent, file_name)?,
            HnswInstance::Manhattan(h) => h.file_dump(parent, file_name)?,
            HnswInstance::Hamming(h) => h.file_dump(parent, file_name)?,
            HnswInstance::Jaccard(h) => h.file_dump(parent, file_name)?,
            HnswInstance::Chebyshev(h) => h.file_dump(parent, file_name)?,
            HnswInstance::Canberra(h) => h.file_dump(parent, file_name)?,
            HnswInstance::BrayCurtis(h) => h.file_dump(parent, file_name)?,
        };

        Ok(())
    }

    // Note: Index persistence is handled via save_index/restore pattern
    // Direct index loading is not supported due to distance metric polymorphism

    #[inline]
    pub fn get_id_to_idx_map(&self) -> &HashMap<Id, usize> {
        &self.id_to_idx
    }

    #[inline]
    pub fn get_idx_to_id_map(&self) -> &HashMap<usize, Id> {
        &self.idx_to_id
    }

    pub fn restore(
        dimension: usize,
        distance: Distance,
        id_to_idx: HashMap<Id, usize>,
        idx_to_id: HashMap<usize, Id>,
        next_idx: usize,
    ) -> Result<Self> {
        Self::restore_with_config(
            dimension,
            distance,
            id_to_idx,
            idx_to_id,
            next_idx,
            HnswConfig::default(),
        )
    }

    pub fn restore_with_config(
        dimension: usize,
        distance: Distance,
        id_to_idx: HashMap<Id, usize>,
        idx_to_id: HashMap<usize, Id>,
        next_idx: usize,
        config: HnswConfig,
    ) -> Result<Self> {
        let hnsw = match distance {
            Distance::Cosine => HnswInstance::Cosine(Hnsw::<f32, DistCosine>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistCosine,
            )),
            Distance::Euclidean => HnswInstance::Euclidean(Hnsw::<f32, DistL2>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistL2,
            )),
            Distance::DotProduct => HnswInstance::DotProduct(Hnsw::<f32, DistDot>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistDot,
            )),
            Distance::Manhattan => HnswInstance::Manhattan(Hnsw::<f32, DistL1>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistL1,
            )),
            Distance::Hamming => HnswInstance::Hamming(Hnsw::<f32, DistHammingF32>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistHammingF32,
            )),
            Distance::Jaccard => HnswInstance::Jaccard(Hnsw::<f32, DistJaccardF32>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistJaccardF32,
            )),
            Distance::Chebyshev => HnswInstance::Chebyshev(Hnsw::<f32, DistChebyshev>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistChebyshev,
            )),
            Distance::Canberra => HnswInstance::Canberra(Hnsw::<f32, DistCanberra>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistCanberra,
            )),
            Distance::BrayCurtis => HnswInstance::BrayCurtis(Hnsw::<f32, DistBrayCurtis>::new(
                config.m,
                config.max_elements,
                config.m,
                config.ef_construction,
                DistBrayCurtis,
            )),
        };

        Ok(Self {
            hnsw,
            id_to_idx,
            idx_to_id,
            next_idx,
            dimension,
            distance,
            config,
        })
    }

    #[inline]
    pub fn get_next_idx(&self) -> usize {
        self.next_idx
    }

    pub fn set_mappings(
        &mut self,
        id_to_idx: HashMap<Id, usize>,
        idx_to_id: HashMap<usize, Id>,
        next_idx: usize,
    ) {
        self.id_to_idx = id_to_idx;
        self.idx_to_id = idx_to_id;
        self.next_idx = next_idx;
    }

    pub fn rebuild_from_vectors(&mut self, vectors: &[(Id, Vec<f32>)]) -> Result<()> {
        for (id, vector) in vectors {
            self.insert(id.clone(), vector)?;
        }
        Ok(())
    }

    pub fn batch_insert(&mut self, items: Vec<(Id, Vec<f32>)>) -> Result<()> {
        for (id, vector) in items {
            self.insert(id, &vector)?;
        }
        Ok(())
    }

    /// Optimize the HNSW index by rebuilding it from scratch.
    ///
    /// This removes "ghost" nodes that accumulate after deletions.
    /// The HNSW graph structure doesn't support true node removal, so deleted
    /// nodes remain in the graph until optimize() is called.
    ///
    /// Returns the number of ghost nodes that were removed.
    pub fn optimize(&mut self, vectors: &[(Id, Vec<f32>)]) -> Result<usize> {
        if vectors.is_empty() {
            return Ok(0);
        }

        // Count ghost nodes (nodes in old index that aren't in the vectors list)
        let old_count = self.id_to_idx.len();
        let new_count = vectors.len();
        let ghost_count = old_count.saturating_sub(new_count);

        // Create a fresh HNSW instance with same configuration
        let new_hnsw = match self.distance {
            Distance::Cosine => HnswInstance::Cosine(Hnsw::<f32, DistCosine>::new(
                self.config.m,
                self.config.max_elements,
                self.config.m,
                self.config.ef_construction,
                DistCosine,
            )),
            Distance::Euclidean => HnswInstance::Euclidean(Hnsw::<f32, DistL2>::new(
                self.config.m,
                self.config.max_elements,
                self.config.m,
                self.config.ef_construction,
                DistL2,
            )),
            Distance::DotProduct => HnswInstance::DotProduct(Hnsw::<f32, DistDot>::new(
                self.config.m,
                self.config.max_elements,
                self.config.m,
                self.config.ef_construction,
                DistDot,
            )),
            Distance::Manhattan => HnswInstance::Manhattan(Hnsw::<f32, DistL1>::new(
                self.config.m,
                self.config.max_elements,
                self.config.m,
                self.config.ef_construction,
                DistL1,
            )),
            Distance::Hamming => HnswInstance::Hamming(Hnsw::<f32, DistHammingF32>::new(
                self.config.m,
                self.config.max_elements,
                self.config.m,
                self.config.ef_construction,
                DistHammingF32,
            )),
            Distance::Jaccard => HnswInstance::Jaccard(Hnsw::<f32, DistJaccardF32>::new(
                self.config.m,
                self.config.max_elements,
                self.config.m,
                self.config.ef_construction,
                DistJaccardF32,
            )),
            Distance::Chebyshev => HnswInstance::Chebyshev(Hnsw::<f32, DistChebyshev>::new(
                self.config.m,
                self.config.max_elements,
                self.config.m,
                self.config.ef_construction,
                DistChebyshev,
            )),
            Distance::Canberra => HnswInstance::Canberra(Hnsw::<f32, DistCanberra>::new(
                self.config.m,
                self.config.max_elements,
                self.config.m,
                self.config.ef_construction,
                DistCanberra,
            )),
            Distance::BrayCurtis => HnswInstance::BrayCurtis(Hnsw::<f32, DistBrayCurtis>::new(
                self.config.m,
                self.config.max_elements,
                self.config.m,
                self.config.ef_construction,
                DistBrayCurtis,
            )),
        };

        // Replace the old HNSW with the new empty one
        self.hnsw = new_hnsw;
        self.id_to_idx.clear();
        self.idx_to_id.clear();
        self.next_idx = 0;

        // Re-insert all vectors into the fresh index
        for (id, vector) in vectors {
            self.insert(id.clone(), vector)?;
        }

        Ok(ghost_count)
    }

    #[inline]
    pub fn search_with_ef(
        &self,
        vector: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(Id, f32)>> {
        if self.id_to_idx.is_empty() {
            return Ok(Vec::new());
        }

        let neighbors = match &self.hnsw {
            HnswInstance::Cosine(h) => search_hnsw(h, vector, k, ef_search),
            HnswInstance::Euclidean(h) => search_hnsw(h, vector, k, ef_search),
            HnswInstance::DotProduct(h) => search_hnsw(h, vector, k, ef_search),
            HnswInstance::Manhattan(h) => search_hnsw(h, vector, k, ef_search),
            HnswInstance::Hamming(h) => search_hnsw(h, vector, k, ef_search),
            HnswInstance::Jaccard(h) => search_hnsw(h, vector, k, ef_search),
            HnswInstance::Chebyshev(h) => search_hnsw(h, vector, k, ef_search),
            HnswInstance::Canberra(h) => search_hnsw(h, vector, k, ef_search),
            HnswInstance::BrayCurtis(h) => search_hnsw(h, vector, k, ef_search),
        };

        Ok(neighbors
            .into_iter()
            .filter_map(|neighbor| {
                let idx = neighbor.d_id;
                self.idx_to_id.get(&idx).map(|id| {
                    let score = match self.distance {
                        // Similarity metrics: higher raw value = more similar
                        Distance::Cosine | Distance::DotProduct => neighbor.distance,
                        // Distance metrics: lower raw value = more similar, invert for score
                        Distance::Euclidean
                        | Distance::Manhattan
                        | Distance::Chebyshev
                        | Distance::Canberra => 1.0 / (1.0 + neighbor.distance),
                        // Normalized distance metrics [0,1]: convert to similarity
                        Distance::Hamming | Distance::Jaccard | Distance::BrayCurtis => {
                            1.0 - neighbor.distance.clamp(0.0, 1.0)
                        },
                    };
                    (id.clone(), score)
                })
            })
            .collect())
    }

    #[cfg(target_arch = "wasm32")]
    pub fn to_visualizer(&self) -> Result<crate::graph_viz::HnswVisualizer> {
        // WASM implementation would go here
        Err(anyhow!(
            "Graph visualization not yet implemented for distance-aware backend"
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_visualizer(&self) -> Result<crate::graph_viz::HnswVisualizer> {
        Err(anyhow!(
            "Graph visualization is only supported in WASM builds. \
             Compile with --target wasm32-unknown-unknown to use this feature."
        ))
    }

    #[inline]
    pub fn distance(&self) -> Distance {
        self.distance
    }
}
