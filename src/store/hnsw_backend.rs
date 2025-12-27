// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

use super::types::{Distance, Id};
use anyhow::{anyhow, Result};
use hnsw_rs::prelude::*;
use std::collections::HashMap;
use std::path::Path;

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
                config.m,              // max_nb_connection
                config.max_elements,   // max_elements
                config.m,              // max_layer (typically same as M)
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
            _ => {
                return Err(anyhow!(
                    "Distance metric {:?} is not yet supported by the HNSW backend. \
                     Supported metrics: Cosine, Euclidean, DotProduct. \
                     See https://github.com/PhilipJohnBasile/vecstore/issues for updates.",
                    distance
                ))
            }
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

    pub fn search(&self, vector: &[f32], k: usize) -> Vec<(Id, f32)> {
        if self.id_to_idx.is_empty() {
            return Vec::new();
        }

        let neighbors = match &self.hnsw {
            HnswInstance::Cosine(h) => h.search(vector, k, 30),
            HnswInstance::Euclidean(h) => h.search(vector, k, 30),
            HnswInstance::DotProduct(h) => h.search(vector, k, 30),
        };

        neighbors
            .into_iter()
            .filter_map(|neighbor| {
                let idx = neighbor.d_id;
                self.idx_to_id.get(&idx).map(|id| {
                    let score = match self.distance {
                        Distance::Cosine | Distance::DotProduct => neighbor.distance,
                        Distance::Euclidean => {
                            // For Euclidean, invert so higher score = closer
                            1.0 / (1.0 + neighbor.distance)
                        }
                        _ => {
                            // This should never happen since we validate distance metric in new()
                            neighbor.distance
                        }
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

        match &self.hnsw {
            HnswInstance::Cosine(h) => {
                h.file_dump(parent, file_name)?;
            }
            HnswInstance::Euclidean(h) => {
                h.file_dump(parent, file_name)?;
            }
            HnswInstance::DotProduct(h) => {
                h.file_dump(parent, file_name)?;
            }
        }

        Ok(())
    }

    // Note: Index persistence is handled via save_index/restore pattern
    // Direct index loading is not supported due to distance metric polymorphism

    pub fn get_id_to_idx_map(&self) -> &HashMap<Id, usize> {
        &self.id_to_idx
    }

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
        Self::restore_with_config(dimension, distance, id_to_idx, idx_to_id, next_idx, HnswConfig::default())
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
            _ => {
                return Err(anyhow!(
                    "Distance metric {:?} is not yet supported by the HNSW backend. \
                     Supported metrics: Cosine, Euclidean, DotProduct. \
                     See https://github.com/PhilipJohnBasile/vecstore/issues for updates.",
                    distance
                ))
            }
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
        let ghost_count = if old_count > new_count { old_count - new_count } else { 0 };

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
            _ => {
                return Err(anyhow!(
                    "Distance metric {:?} is not supported for optimization",
                    self.distance
                ))
            }
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
            HnswInstance::Cosine(h) => h.search(vector, k, ef_search),
            HnswInstance::Euclidean(h) => h.search(vector, k, ef_search),
            HnswInstance::DotProduct(h) => h.search(vector, k, ef_search),
        };

        Ok(neighbors
            .into_iter()
            .filter_map(|neighbor| {
                let idx = neighbor.d_id;
                self.idx_to_id.get(&idx).map(|id| {
                    let score = match self.distance {
                        Distance::Cosine | Distance::DotProduct => neighbor.distance,
                        Distance::Euclidean => {
                            // For Euclidean, invert so higher score = closer
                            1.0 / (1.0 + neighbor.distance)
                        }
                        _ => {
                            // This should never happen since we validate distance metric in new()
                            neighbor.distance
                        }
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

    pub fn distance(&self) -> Distance {
        self.distance
    }
}
