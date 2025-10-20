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
    pub fn new(dimension: usize, distance: Distance) -> Self {
        let hnsw = match distance {
            Distance::Cosine => HnswInstance::Cosine(Hnsw::<f32, DistCosine>::new(
                16,      // max_nb_connection
                100_000, // max_elements
                16,      // max_layer
                200,     // ef_construction
                DistCosine,
            )),
            Distance::Euclidean => HnswInstance::Euclidean(Hnsw::<f32, DistL2>::new(
                16, 100_000, 16, 200, DistL2,
            )),
            Distance::DotProduct => HnswInstance::DotProduct(Hnsw::<f32, DistDot>::new(
                16, 100_000, 16, 200, DistDot,
            )),
        };

        Self {
            hnsw,
            id_to_idx: HashMap::new(),
            idx_to_id: HashMap::new(),
            next_idx: 0,
            dimension,
            distance,
        }
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
                    };
                    (id.clone(), score)
                })
            })
            .collect()
    }

    pub fn save_index<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let base_name = path.as_ref().to_str().ok_or_else(|| anyhow!("Invalid path"))?;

        match &self.hnsw {
            HnswInstance::Cosine(h) => h.file_dump(base_name)?,
            HnswInstance::Euclidean(h) => h.file_dump(base_name)?,
            HnswInstance::DotProduct(h) => h.file_dump(base_name)?,
        }

        Ok(())
    }

    pub fn load_index<P: AsRef<Path>>(
        path: P,
        dimension: usize,
        distance: Distance,
    ) -> Result<Hnsw<'static, f32, impl Distance<f32>>> {
        let base_name = path.as_ref().to_str().ok_or_else(|| anyhow!("Invalid path"))?;

        match distance {
            Distance::Cosine => {
                let hnsw = Hnsw::<f32, DistCosine>::file_load(base_name, DistCosine)?;
                Ok(hnsw)
            }
            Distance::Euclidean => {
                let hnsw = Hnsw::<f32, DistL2>::file_load(base_name, DistL2)?;
                Ok(hnsw)
            }
            Distance::DotProduct => {
                let hnsw = Hnsw::<f32, DistDot>::file_load(base_name, DistDot)?;
                Ok(hnsw)
            }
        }
    }

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
}
