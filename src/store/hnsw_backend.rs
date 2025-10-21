use super::types::{Distance, Id};
use anyhow::{anyhow, Result};
use hnsw_rs::prelude::*;
use std::collections::HashMap;
use std::path::Path;

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
}

impl HnswBackend {
    pub fn new(dimension: usize, distance: Distance) -> Result<Self> {
        let hnsw = match distance {
            Distance::Cosine => HnswInstance::Cosine(Hnsw::<f32, DistCosine>::new(
                16,      // max_nb_connection
                100_000, // max_elements
                16,      // max_layer
                200,     // ef_construction
                DistCosine,
            )),
            Distance::Euclidean => {
                HnswInstance::Euclidean(Hnsw::<f32, DistL2>::new(16, 100_000, 16, 200, DistL2))
            }
            Distance::DotProduct => {
                HnswInstance::DotProduct(Hnsw::<f32, DistDot>::new(16, 100_000, 16, 200, DistDot))
            }
            _ => {
                return Err(anyhow!(
                    "Distance metric {:?} is not yet supported by the HNSW backend. \
                     Supported metrics: Cosine, Euclidean, DotProduct. \
                     See https://github.com/yourusername/vecstore/issues for updates.",
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
        })
    }

    pub fn insert(&mut self, id: Id, vector: &[f32]) -> Result<()> {
        if self.dimension > 0 && vector.len() != self.dimension {
            return Err(anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
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

    pub fn remove(&mut self, id: &str) -> Result<()> {
        if let Some(&idx) = self.id_to_idx.get(id) {
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
        let hnsw = match distance {
            Distance::Cosine => HnswInstance::Cosine(Hnsw::<f32, DistCosine>::new(
                16, 100_000, 16, 200, DistCosine,
            )),
            Distance::Euclidean => {
                HnswInstance::Euclidean(Hnsw::<f32, DistL2>::new(16, 100_000, 16, 200, DistL2))
            }
            Distance::DotProduct => {
                HnswInstance::DotProduct(Hnsw::<f32, DistDot>::new(16, 100_000, 16, 200, DistDot))
            }
            _ => {
                return Err(anyhow!(
                    "Distance metric {:?} is not yet supported by the HNSW backend. \
                     Supported metrics: Cosine, Euclidean, DotProduct. \
                     See https://github.com/yourusername/vecstore/issues for updates.",
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

    pub fn optimize(&mut self, _vectors: &[(Id, Vec<f32>)]) -> Result<usize> {
        // HNSW doesn't need explicit optimization
        // Return number of vectors in index
        Ok(self.id_to_idx.len())
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
