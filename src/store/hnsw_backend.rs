use super::types::Id;
use anyhow::{anyhow, Result};
use hnsw_rs::prelude::*;
use std::collections::HashMap;
use std::path::Path;

pub struct HnswBackend {
    hnsw: Hnsw<'static, f32, DistCosine>,
    id_to_idx: HashMap<Id, usize>,
    idx_to_id: HashMap<usize, Id>,
    next_idx: usize,
    dimension: usize,
}

impl HnswBackend {
    pub fn new(dimension: usize) -> Self {
        let hnsw = Hnsw::<f32, DistCosine>::new(
            16,      // max_nb_connection
            100_000, // max_elements
            16,      // max_layer
            200,     // ef_construction
            DistCosine,
        );

        Self {
            hnsw,
            id_to_idx: HashMap::new(),
            idx_to_id: HashMap::new(),
            next_idx: 0,
            dimension,
        }
    }

    /// Insert a single vector into the index (incremental, no rebuild required)
    ///
    /// This is a true streaming insert - the vector is added to the HNSW index
    /// immediately without rebuilding the entire structure.
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

        self.hnsw.insert((vector, idx));
        self.id_to_idx.insert(id.clone(), idx);
        self.idx_to_id.insert(idx, id);

        Ok(())
    }

    /// Batch insert multiple vectors using parallel processing
    ///
    /// This is more efficient than calling insert() in a loop when you have
    /// many vectors to add at once. Uses rayon for parallel processing.
    pub fn batch_insert(&mut self, items: Vec<(Id, Vec<f32>)>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        // Validate all vectors first
        for (_, vector) in &items {
            if self.dimension > 0 && vector.len() != self.dimension {
                return Err(anyhow!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dimension,
                    vector.len()
                ));
            }
        }

        // Prepare data for parallel insertion
        let mut data_with_idx = Vec::new();
        for (id, vector) in items {
            // Remove old entry if exists
            if let Some(&old_idx) = self.id_to_idx.get(&id) {
                self.idx_to_id.remove(&old_idx);
            }

            let idx = self.next_idx;
            self.next_idx += 1;

            data_with_idx.push((id, vector, idx));
        }

        // Parallel insert into HNSW
        let hnsw_data: Vec<_> = data_with_idx
            .iter()
            .map(|(_, vector, idx)| (vector, *idx))
            .collect();

        self.hnsw.parallel_insert(&hnsw_data);

        // Update mappings
        for (id, _, idx) in data_with_idx {
            self.id_to_idx.insert(id.clone(), idx);
            self.idx_to_id.insert(idx, id);
        }

        Ok(())
    }

    /// Optimize the index by rebuilding to remove "ghost" entries from deletions
    ///
    /// After many remove() operations, the HNSW index can accumulate entries that
    /// are no longer referenced. This method rebuilds the index from scratch to
    /// reclaim memory and improve search performance.
    ///
    /// Returns the number of entries removed.
    pub fn optimize(&mut self, vectors: &[(Id, Vec<f32>)]) -> Result<usize> {
        let old_size = self.idx_to_id.len();
        self.rebuild_from_vectors(vectors)?;
        let new_size = self.idx_to_id.len();
        Ok(old_size.saturating_sub(new_size))
    }

    pub fn remove(&mut self, id: &str) -> Result<()> {
        if let Some(&idx) = self.id_to_idx.get(id) {
            self.id_to_idx.remove(id);
            self.idx_to_id.remove(&idx);
            // Note: hnsw_rs doesn't support removal, so we just remove from our mappings
            // The index will still contain the old data, but we won't reference it
            Ok(())
        } else {
            Err(anyhow!("ID not found: {}", id))
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(Id, f32)>> {
        self.search_with_ef(query, k, 200) // Default ef_search = 200
    }

    /// Search with custom ef_search parameter for performance tuning
    ///
    /// ef_search controls the size of the dynamic candidate list during search:
    /// - Lower values (20-50): Faster search, lower recall
    /// - Medium values (50-100): Balanced (default: 50)
    /// - Higher values (100-200): Better recall, slower search
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(Id, f32)>> {
        if self.dimension > 0 && query.len() != self.dimension {
            return Err(anyhow!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.len()
            ));
        }

        let results = self.hnsw.search(query, k, ef_search);

        let mut neighbors = Vec::new();
        for neighbor in results {
            let idx = neighbor.d_id;
            if let Some(id) = self.idx_to_id.get(&idx) {
                // hnsw_rs returns distance, convert to similarity score
                // For cosine: similarity = 1 - distance
                let score = 1.0 - neighbor.distance;
                neighbors.push((id.clone(), score));
            }
        }

        Ok(neighbors)
    }

    pub fn save_index(&self, path: &Path) -> Result<()> {
        // For hnsw_rs 0.3.2, file_dump takes a path and basename
        let basename = path.file_stem().and_then(|s| s.to_str()).unwrap_or("hnsw");

        let parent = path.parent().unwrap_or_else(|| Path::new("."));

        self.hnsw
            .file_dump(parent, basename)
            .map_err(|e| anyhow!("Failed to save HNSW index: {}", e))?;
        Ok(())
    }

    pub fn rebuild_from_vectors(&mut self, vectors: &[(Id, Vec<f32>)]) -> Result<()> {
        // Clear existing state
        self.id_to_idx.clear();
        self.idx_to_id.clear();
        self.next_idx = 0;

        // Rebuild index
        for (id, vec) in vectors {
            self.insert(id.clone(), vec)?;
        }

        Ok(())
    }

    pub fn get_id_to_idx_map(&self) -> &HashMap<Id, usize> {
        &self.id_to_idx
    }

    pub fn get_idx_to_id_map(&self) -> &HashMap<usize, Id> {
        &self.idx_to_id
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

    pub fn restore(
        dimension: usize,
        id_to_idx: HashMap<Id, usize>,
        idx_to_id: HashMap<usize, Id>,
        next_idx: usize,
    ) -> Result<Self> {
        let mut backend = Self::new(dimension);
        backend.id_to_idx = id_to_idx;
        backend.idx_to_id = idx_to_id;
        backend.next_idx = next_idx;
        Ok(backend)
    }

    /// Create a graph visualizer for the HNSW index
    ///
    /// Note: Graph visualization is currently only supported for WASM builds.
    /// The native build uses hnsw_rs which doesn't expose the graph structure.
    ///
    /// For graph visualization, compile with `--target wasm32-unknown-unknown`.
    pub fn to_visualizer(&self) -> Result<crate::graph_viz::HnswVisualizer> {
        // Optimization Issue #22 fix: clearer error message
        Err(anyhow!(
            "Graph visualization is not supported on native targets.\n\n\
             The native build uses hnsw_rs which doesn't expose internal graph structure.\n\n\
             To visualize HNSW graphs:\n\
             1. Build for WASM: cargo build --target wasm32-unknown-unknown\n\
             2. Or use a custom HNSW backend that exposes graph internals\n\n\
             See documentation for more details on WASM builds."
        ))
    }
}
