pub mod advanced_filters;
mod disk;
pub mod disk_hnsw;
mod filter_parser;
pub mod filters; // Public for WASM module

// HNSW backend only available on non-WASM targets (requires hnsw_rs which needs mmap)
#[cfg(not(target_arch = "wasm32"))]
pub mod hnsw_backend;

// WASM-compatible HNSW implementation (pure in-memory, no mmap)
// Available on all platforms for testing/benchmarking, but only used as backend for WASM
pub mod wasm_hnsw;

// WASM backend wrapper
#[cfg(target_arch = "wasm32")]
pub mod wasm_backend;

// Type alias for the backend based on target architecture
#[cfg(not(target_arch = "wasm32"))]
pub type VectorBackend = hnsw_backend::HnswBackend;

#[cfg(target_arch = "wasm32")]
pub type VectorBackend = wasm_backend::WasmVectorBackend;

pub mod hybrid;
pub mod quantization;
mod types;

pub use filter_parser::{parse_filter, ParseError as FilterParseError};
pub use hybrid::{HybridQuery, TextIndex};
pub use quantization::{PQConfig, PQVectorStore, ProductQuantizer};
pub use types::*;

use anyhow::{Context, Result};
use chrono::Utc;
use std::collections::HashMap;
use std::path::PathBuf;

pub struct VecStore {
    root: PathBuf,
    backend: VectorBackend,
    records: HashMap<Id, Record>,
    dimension: usize,
    text_index: hybrid::TextIndex,
    compaction_config: CompactionConfig,
    config: Config,
}

/// Builder for VecStore with customizable configuration
pub struct VecStoreBuilder {
    path: PathBuf,
    config: Config,
}

impl VecStoreBuilder {
    /// Create a new VecStore builder with default configuration
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            path: path.into(),
            config: Config::default(),
        }
    }

    /// Set the distance metric
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::{VecStore, Distance};
    /// let store = VecStore::builder("./data")
    ///     .distance(Distance::Manhattan)
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn distance(mut self, metric: Distance) -> Self {
        self.config.distance = metric;
        self
    }

    /// Set HNSW M parameter (number of connections per layer)
    ///
    /// Higher values = better recall, more memory usage
    /// Default: 16
    pub fn hnsw_m(mut self, m: usize) -> Self {
        self.config.hnsw_m = m;
        self
    }

    /// Set HNSW ef_construction parameter
    ///
    /// Higher values = better quality index, slower construction
    /// Default: 200
    pub fn hnsw_ef_construction(mut self, ef: usize) -> Self {
        self.config.hnsw_ef_construction = ef;
        self
    }

    /// Build the VecStore with the configured settings
    pub fn build(self) -> Result<VecStore> {
        VecStore::open_with_config(self.path, self.config)
    }
}

impl VecStore {
    /// Create a builder for VecStore
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::{VecStore, Distance};
    /// let store = VecStore::builder("./data")
    ///     .distance(Distance::Manhattan)
    ///     .hnsw_m(32)
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn builder<P: Into<PathBuf>>(path: P) -> VecStoreBuilder {
        VecStoreBuilder::new(path)
    }

    /// Open VecStore with custom configuration
    pub fn open_with_config<P: Into<PathBuf>>(root: P, config: Config) -> Result<Self> {
        let root = root.into();
        let layout = disk::DiskLayout::new(&root);

        // Validate distance metric configuration (Major Issue #5 partial fix)
        if config.distance != Distance::Cosine {
            #[cfg(feature = "tracing")]
            tracing::warn!(
                "Distance metric {:?} is configured but not yet supported. \
                 The store will use Cosine similarity. \
                 See GitHub issue for distance metric support roadmap.",
                config.distance
            );
        }

        if layout.exists() {
            // Load existing store
            let (
                records,
                id_to_idx,
                idx_to_id,
                next_idx,
                dimension,
                loaded_config,
                text_index_data,
            ) = layout.load_all().context("Failed to load existing store")?;

            // Use loaded config if available, otherwise use provided config (Major Issue #7 fix)
            let config = loaded_config.unwrap_or(config);

            #[cfg(not(target_arch = "wasm32"))]
            let mut backend = VectorBackend::new(dimension, config.distance)?;
            #[cfg(target_arch = "wasm32")]
            let mut backend = VectorBackend::new(dimension);
            backend.set_mappings(id_to_idx, idx_to_id, next_idx);

            // Rebuild HNSW index from vectors
            let vectors: Vec<(Id, Vec<f32>)> = records
                .values()
                .map(|r| (r.id.clone(), r.vector.clone()))
                .collect();
            backend.rebuild_from_vectors(&vectors)?;

            // Import text index if available (Major Issue #6 fix)
            let mut text_index = hybrid::TextIndex::new();
            if let Some(texts) = text_index_data {
                text_index.import_texts(texts);
            }

            Ok(Self {
                root,
                backend,
                records,
                dimension,
                text_index,
                compaction_config: CompactionConfig::default(),
                config,
            })
        } else {
            // Create new store - infer dimension from first insert
            layout.ensure_directory()?;

            #[cfg(not(target_arch = "wasm32"))]
            let backend = VectorBackend::new(0, config.distance)?;
            #[cfg(target_arch = "wasm32")]
            let backend = VectorBackend::new(0);

            Ok(Self {
                root,
                backend, // Will be set on first insert
                records: HashMap::new(),
                dimension: 0,
                text_index: hybrid::TextIndex::new(),
                compaction_config: CompactionConfig::default(),
                config,
            })
        }
    }

    pub fn open<P: Into<PathBuf>>(root: P) -> Result<Self> {
        Self::open_with_config(root, Config::default())
    }

    /// Get the distance metric configured for this store
    pub fn distance_metric(&self) -> Distance {
        self.config.distance
    }

    /// Get the full configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    #[tracing::instrument(skip(self, vector, metadata), fields(dimension = vector.len()))]
    pub fn upsert(&mut self, id: Id, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        // Validate vector is non-empty (Critical Issue #20 fix)
        if vector.is_empty() {
            return Err(anyhow::anyhow!(
                "Cannot insert zero-dimension vector. Vectors must have at least one dimension."
            ));
        }

        // Set dimension on first insert
        if self.dimension == 0 {
            self.dimension = vector.len();
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.backend = VectorBackend::new(self.dimension, self.config.distance)?;
            }
            #[cfg(target_arch = "wasm32")]
            {
                self.backend = VectorBackend::new(self.dimension);
            }
        }

        if vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            ));
        }

        let record = Record {
            id: id.clone(),
            vector: vector.clone(),
            metadata,
            created_at: Utc::now().timestamp(),
            deleted: false,
            deleted_at: None,
            expires_at: None,
        };

        self.backend.insert(id.clone(), &vector)?;
        self.records.insert(id, record);

        Ok(())
    }

    pub fn remove(&mut self, id: &str) -> Result<()> {
        self.backend.remove(id)?;
        self.records
            .remove(id)
            .ok_or_else(|| anyhow::anyhow!("Record not found: {}", id))?;

        // Clean up text index (Critical Issue #4 fix)
        self.text_index.remove_document(id);

        Ok(())
    }

    /// Batch insert multiple vectors using parallel processing
    ///
    /// This is significantly faster than calling upsert() in a loop when you have
    /// many vectors to add at once. The HNSW index is built in parallel using rayon.
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::{VecStore, make_record};
    /// # let mut store = VecStore::open("data").unwrap();
    /// let records = vec![
    ///     make_record("doc1", vec![0.1, 0.2, 0.3], vec![]),
    ///     make_record("doc2", vec![0.4, 0.5, 0.6], vec![]),
    /// ];
    /// store.batch_upsert(records).unwrap();
    /// ```
    pub fn batch_upsert(&mut self, items: impl IntoIterator<Item = Record>) -> Result<()> {
        use rayon::prelude::*;

        let items: Vec<_> = items.into_iter().collect();

        if items.is_empty() {
            return Ok(());
        }

        // Set dimension from first record if needed
        if self.dimension == 0 {
            if let Some(first) = items.first() {
                // Validate first vector is non-empty (Major Issue #21 fix)
                if first.vector.is_empty() {
                    return Err(anyhow::anyhow!(
                        "Cannot insert zero-dimension vector. Vectors must have at least one dimension."
                    ));
                }
                self.dimension = first.vector.len();
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.backend = VectorBackend::new(self.dimension, self.config.distance)?;
                }
                #[cfg(target_arch = "wasm32")]
                {
                    self.backend = VectorBackend::new(self.dimension);
                }
            }
        }

        // Validate all vectors in parallel (Major Issue #21 fix)
        items.par_iter().try_for_each(|record| {
            if record.vector.is_empty() {
                return Err(anyhow::anyhow!(
                    "Cannot insert zero-dimension vector for '{}'. Vectors must have at least one dimension.",
                    record.id
                ));
            }
            if record.vector.len() != self.dimension {
                return Err(anyhow::anyhow!(
                    "Vector dimension mismatch for {}: expected {}, got {}",
                    record.id,
                    self.dimension,
                    record.vector.len()
                ));
            }
            Ok(())
        })?;

        // Prepare data for batch insert
        let batch_data: Vec<(Id, Vec<f32>)> = items
            .iter()
            .map(|r| (r.id.clone(), r.vector.clone()))
            .collect();

        // Use parallel batch insert (much faster than sequential)
        self.backend.batch_insert(batch_data)?;

        // Update records
        for record in items {
            self.records.insert(record.id.clone(), record);
        }

        Ok(())
    }

    /// Optimize the index by rebuilding to remove "ghost" entries from deletions
    ///
    /// After many remove() operations, the HNSW index accumulates entries that
    /// are no longer referenced. This method rebuilds the index from scratch to
    /// reclaim memory and improve search performance.
    ///
    /// Returns the number of ghost entries removed.
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::VecStore;
    /// # let mut store = VecStore::open("data").unwrap();
    /// // After many deletions...
    /// let removed = store.optimize().unwrap();
    /// println!("Removed {} ghost entries", removed);
    /// ```
    pub fn optimize(&mut self) -> Result<usize> {
        let vectors: Vec<(Id, Vec<f32>)> = self
            .records
            .values()
            .map(|r| (r.id.clone(), r.vector.clone()))
            .collect();

        self.backend.optimize(&vectors)
    }

    #[tracing::instrument(skip(self, q), fields(k = q.k, has_filter = q.filter.is_some(), dimension = q.vector.len()))]
    pub fn query(&self, q: Query) -> Result<Vec<Neighbor>> {
        if self.dimension == 0 {
            return Ok(Vec::new());
        }

        if q.vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                q.vector.len()
            ));
        }

        // Determine fetch size for HNSW search
        let fetch_size = if q.filter.is_some() {
            // When filtering, we need to over-fetch to account for filtered-out results
            // Fetch all records (up to k*10) to ensure we have enough candidates
            let total_records = self.records.len();
            if total_records <= q.k {
                // If we have fewer records than k, fetch all
                total_records
            } else {
                // Otherwise, over-fetch by 10x (capped at total records) - using saturating_mul to prevent overflow (Critical Issue #10 fix)
                std::cmp::min(q.k.saturating_mul(10), total_records)
            }
        } else {
            // No filter, just fetch k (or all records if fewer than k)
            std::cmp::min(q.k, self.records.len())
        };
        #[cfg(not(target_arch = "wasm32"))]
        let candidates = self.backend.search(&q.vector, fetch_size);
        #[cfg(target_arch = "wasm32")]
        let candidates = self.backend.search(&q.vector, fetch_size)?;

        let mut results = Vec::new();
        for (id, score) in candidates {
            if let Some(record) = self.records.get(&id) {
                // Skip soft-deleted records
                if record.deleted {
                    continue;
                }

                // Apply filter if present
                if let Some(ref filter) = q.filter {
                    if !filters::evaluate_filter(filter, &record.metadata) {
                        continue;
                    }
                }

                results.push(Neighbor {
                    id: id.clone(),
                    score,
                    metadata: record.metadata.clone(),
                });

                if results.len() >= q.k {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Query with detailed explanations of why each result was returned
    ///
    /// This is useful for debugging, understanding search results, and optimizing queries.
    /// Returns the same results as `query()` but with additional explanation metadata.
    pub fn query_explain(&self, q: Query) -> Result<Vec<ExplainedNeighbor>> {
        if self.dimension == 0 {
            return Ok(Vec::new());
        }

        if q.vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                q.vector.len()
            ));
        }

        // Determine fetch size for HNSW search
        let fetch_size = if q.filter.is_some() {
            let total_records = self.records.len();
            if total_records <= q.k {
                total_records
            } else {
                // Using saturating_mul to prevent overflow (Major Issue #10 fix)
                std::cmp::min(q.k.saturating_mul(10), total_records)
            }
        } else {
            std::cmp::min(q.k, self.records.len())
        };

        // Track stats for explanation
        #[cfg(not(target_arch = "wasm32"))]
        let candidates = self.backend.search(&q.vector, fetch_size);
        #[cfg(target_arch = "wasm32")]
        let candidates = self.backend.search(&q.vector, fetch_size)?;
        let total_candidates = candidates.len();

        let distance_metric = self.config.distance.name().to_string();
        let has_filter = q.filter.is_some();

        let mut results = Vec::new();
        let mut rank = 1;
        let mut filtered_out_count = 0;
        let mut deleted_count = 0;

        for (id, score) in candidates {
            if let Some(record) = self.records.get(&id) {
                // Track if soft-deleted
                if record.deleted {
                    deleted_count += 1;
                    continue;
                }

                // Evaluate filter and collect details
                let (filter_passed, filter_details) = if let Some(ref filter) = q.filter {
                    let passed = filters::evaluate_filter(filter, &record.metadata);
                    let details = FilterEvaluation {
                        filter_expr: format!("{:?}", filter),
                        matched_conditions: if passed {
                            vec![format!("All conditions matched")]
                        } else {
                            vec![]
                        },
                        failed_conditions: if !passed {
                            vec![format!("Filter did not match")]
                        } else {
                            vec![]
                        },
                        passed,
                    };
                    (passed, Some(details))
                } else {
                    (true, None)
                };

                if !filter_passed {
                    filtered_out_count += 1;
                    continue;
                }

                // Build explanation text
                let explanation_text = if has_filter {
                    format!(
                        "Ranked #{} with score {:.4} ({}). Passed filters. {} candidates evaluated, {} filtered out, {} deleted.",
                        rank, score, distance_metric, total_candidates, filtered_out_count, deleted_count
                    )
                } else {
                    format!(
                        "Ranked #{} with score {:.4} ({}). {} candidates evaluated, {} deleted.",
                        rank, score, distance_metric, total_candidates, deleted_count
                    )
                };

                results.push(ExplainedNeighbor {
                    id: id.clone(),
                    score,
                    metadata: record.metadata.clone(),
                    explanation: QueryExplanation {
                        raw_score: score,
                        distance_metric: distance_metric.clone(),
                        filter_passed,
                        filter_details,
                        graph_stats: Some(GraphTraversalStats {
                            distance_calculations: total_candidates,
                            nodes_visited: total_candidates,
                            found_at_layer: None, // Would require HNSW backend changes
                            hops_from_entry: None, // Would require HNSW backend changes
                        }),
                        rank,
                        explanation_text,
                    },
                });

                rank += 1;

                if results.len() >= q.k {
                    break;
                }
            }
        }

        Ok(results)
    }

    pub fn save(&self) -> Result<()> {
        let layout = disk::DiskLayout::new(&self.root);

        // Export text index if any texts are indexed (Major Issue #6 fix)
        let text_index_data = if self.text_index.export_texts().is_empty() {
            None
        } else {
            Some(self.text_index.export_texts())
        };

        layout.save_all(
            &self.records,
            self.backend.get_id_to_idx_map(),
            self.backend.get_idx_to_id_map(),
            // Use actual next_idx counter, not map length (Critical Issue #2 fix)
            self.backend.get_next_idx(),
            self.dimension,
            &self.config,    // Major Issue #7 fix: persist config
            text_index_data, // Major Issue #6 fix: persist text index
        )?;

        // Save HNSW index
        if self.dimension > 0 {
            self.backend.save_index(&layout.hnsw_path())?;
        }

        Ok(())
    }

    /// Get the number of active (non-deleted) records
    pub fn count(&self) -> usize {
        self.active_count()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Query with a filter expression parsed from a SQL-like string
    ///
    /// # Example
    ///
    /// ```
    /// # use vecstore::{VecStore, Metadata};
    /// # use std::collections::HashMap;
    /// # let temp_dir = tempfile::tempdir().unwrap();
    /// # let mut store = VecStore::open(temp_dir.path()).unwrap();
    /// # let mut meta = Metadata { fields: HashMap::new() };
    /// # meta.fields.insert("age".into(), serde_json::json!(25));
    /// # meta.fields.insert("role".into(), serde_json::json!("admin"));
    /// # store.upsert("user1".into(), vec![1.0, 0.0, 0.0], meta).unwrap();
    /// let results = store.query_with_filter(
    ///     vec![1.0, 0.0, 0.0],
    ///     10,
    ///     "age > 18 AND role = 'admin'"
    /// ).unwrap();
    /// ```
    pub fn query_with_filter(
        &self,
        vector: Vec<f32>,
        k: usize,
        filter_str: &str,
    ) -> Result<Vec<Neighbor>> {
        let filter = filter_parser::parse_filter(filter_str)?;
        self.query(Query {
            vector,
            k,
            filter: Some(filter),
        })
    }

    /// Get the number of active (non-deleted) vectors in the store
    pub fn len(&self) -> usize {
        self.active_count()
    }

    /// Check if the store has no active (non-deleted) records
    pub fn is_empty(&self) -> bool {
        self.active_count() == 0
    }

    /// Create a named snapshot of the current store state
    ///
    /// # Example
    ///
    /// ```
    /// # use vecstore::VecStore;
    /// # let temp_dir = tempfile::tempdir().unwrap();
    /// # let store = VecStore::open(temp_dir.path()).unwrap();
    /// store.create_snapshot("backup-2024-01-15")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn create_snapshot(&self, name: &str) -> Result<()> {
        let snapshot_dir = self.root.join("snapshots").join(name);

        if snapshot_dir.exists() {
            return Err(anyhow::anyhow!(
                "Snapshot '{}' already exists. Use a different name or delete the old snapshot.",
                name
            ));
        }

        std::fs::create_dir_all(&snapshot_dir)
            .with_context(|| format!("Failed to create snapshot directory: {:?}", snapshot_dir))?;

        // Save to snapshot directory
        let layout = disk::DiskLayout::new(&snapshot_dir);

        // Export text index if any texts are indexed (Major Issue #6 fix)
        let text_index_data = if self.text_index.export_texts().is_empty() {
            None
        } else {
            Some(self.text_index.export_texts())
        };

        layout.save_all(
            &self.records,
            self.backend.get_id_to_idx_map(),
            self.backend.get_idx_to_id_map(),
            // Use actual next_idx counter, not map length (Critical Issue #2 fix)
            self.backend.get_next_idx(),
            self.dimension,
            &self.config,    // Major Issue #7 fix: persist config in snapshots
            text_index_data, // Major Issue #6 fix: persist text index in snapshots
        )?;

        // Save HNSW index
        if self.dimension > 0 {
            self.backend.save_index(&layout.hnsw_path())?;
        }

        // Write snapshot metadata
        let metadata = serde_json::json!({
            "name": name,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "record_count": self.records.len(),
            "dimension": self.dimension,
        });

        std::fs::write(
            snapshot_dir.join("snapshot.json"),
            serde_json::to_string_pretty(&metadata)?,
        )?;

        Ok(())
    }

    /// List all available snapshots
    ///
    /// Returns a vector of (snapshot_name, created_at, record_count)
    pub fn list_snapshots(&self) -> Result<Vec<(String, String, usize)>> {
        let snapshots_dir = self.root.join("snapshots");

        if !snapshots_dir.exists() {
            return Ok(Vec::new());
        }

        let mut snapshots = Vec::new();

        for entry in std::fs::read_dir(&snapshots_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let metadata_path = path.join("snapshot.json");
                if metadata_path.exists() {
                    let content = std::fs::read_to_string(&metadata_path)?;
                    let metadata: serde_json::Value = serde_json::from_str(&content)?;

                    let name = metadata["name"].as_str().unwrap_or("unknown").to_string();
                    let created_at = metadata["created_at"]
                        .as_str()
                        .unwrap_or("unknown")
                        .to_string();
                    let count = metadata["record_count"].as_u64().unwrap_or(0) as usize;

                    snapshots.push((name, created_at, count));
                }
            }
        }

        // Sort by creation time (most recent first)
        snapshots.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(snapshots)
    }

    /// Restore from a named snapshot
    ///
    /// This will replace the current store with the snapshot data.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vecstore::VecStore;
    /// # let temp_dir = tempfile::tempdir().unwrap();
    /// # let mut store = VecStore::open(temp_dir.path()).unwrap();
    /// store.restore_snapshot("backup-2024-01-15")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn restore_snapshot(&mut self, name: &str) -> Result<()> {
        let snapshot_dir = self.root.join("snapshots").join(name);

        if !snapshot_dir.exists() {
            return Err(anyhow::anyhow!(
                "Snapshot '{}' not found. Use list_snapshots() to see available snapshots.",
                name
            ));
        }

        // Load from snapshot directory
        let layout = disk::DiskLayout::new(&snapshot_dir);

        if !layout.manifest_path().exists() {
            return Err(anyhow::anyhow!(
                "Snapshot '{}' is corrupted (missing manifest)",
                name
            ));
        }

        let (records, id_to_idx, idx_to_id, next_idx, dimension, loaded_config, text_index_data) =
            layout.load_all()?;

        self.records = records;
        self.dimension = dimension;

        // Update config if loaded (Major Issue #7 fix)
        if let Some(config) = loaded_config {
            self.config = config;
        }

        // Restore text index if available (Major Issue #6 fix)
        self.text_index = hybrid::TextIndex::new();
        if let Some(texts) = text_index_data {
            self.text_index.import_texts(texts);
        }

        // Recreate backend
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.backend = hnsw_backend::HnswBackend::restore(
                dimension,
                self.config.distance,
                id_to_idx,
                idx_to_id,
                next_idx,
            )?;
        }

        #[cfg(target_arch = "wasm32")]
        {
            // WASM backend doesn't support restore (no persistence in browser)
            // Create a new backend and rebuild from records
            let mut backend = VectorBackend::new(dimension);
            backend.set_mappings(id_to_idx, idx_to_id, next_idx);
            self.backend = backend;
        }

        // Rebuild HNSW index from vectors
        if dimension > 0 {
            let vectors: Vec<(Id, Vec<f32>)> = self
                .records
                .iter()
                .map(|(id, record)| (id.clone(), record.vector.clone()))
                .collect();
            self.backend.rebuild_from_vectors(&vectors)?;
        }

        Ok(())
    }

    /// Delete a named snapshot
    pub fn delete_snapshot(&self, name: &str) -> Result<()> {
        let snapshot_dir = self.root.join("snapshots").join(name);

        if !snapshot_dir.exists() {
            return Err(anyhow::anyhow!("Snapshot '{}' not found", name));
        }

        std::fs::remove_dir_all(&snapshot_dir)
            .with_context(|| format!("Failed to delete snapshot: {}", name))?;

        Ok(())
    }

    /// Index text content for a document (enables hybrid search)
    ///
    /// # Example
    ///
    /// ```
    /// # use vecstore::VecStore;
    /// # use std::collections::HashMap;
    /// # let temp_dir = tempfile::tempdir().unwrap();
    /// # let mut store = VecStore::open(temp_dir.path()).unwrap();
    /// # let mut meta = vecstore::Metadata { fields: HashMap::new() };
    /// # store.upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta).unwrap();
    /// store.index_text("doc1", "Rust is a systems programming language")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn index_text(&mut self, id: &str, text: impl Into<String>) -> Result<()> {
        if !self.records.contains_key(id) {
            return Err(anyhow::anyhow!(
                "Cannot index text for non-existent document: {}",
                id
            ));
        }

        self.text_index.index_document(id.to_string(), text.into());
        Ok(())
    }

    /// Perform hybrid search combining vector similarity and keyword search
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vecstore::{VecStore, HybridQuery};
    /// # let temp_dir = tempfile::tempdir().unwrap();
    /// # let store = VecStore::open(temp_dir.path()).unwrap();
    /// let query = HybridQuery {
    ///     vector: vec![0.1, 0.2, 0.3],
    ///     keywords: "rust programming".into(),
    ///     k: 10,
    ///     filter: None,
    ///     alpha: 0.7, // 70% vector, 30% keyword
    /// };
    ///
    /// let results = store.hybrid_query(query)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[tracing::instrument(skip(self, query), fields(k = query.k, has_keywords = !query.keywords.is_empty(), alpha = query.alpha))]
    pub fn hybrid_query(&self, query: HybridQuery) -> Result<Vec<Neighbor>> {
        if self.dimension == 0 {
            return Ok(Vec::new());
        }

        if query.vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.vector.len()
            ));
        }

        // Get vector similarity scores
        let fetch_size = if query.filter.is_some() {
            // Using saturating_mul to prevent overflow (Major Issue #10 fix)
            std::cmp::min(query.k.saturating_mul(10), self.records.len())
        } else {
            // Using saturating_mul to prevent overflow (Major Issue #10 fix)
            std::cmp::min(query.k.saturating_mul(2), self.records.len()) // Fetch more for ranking
        };

        #[cfg(not(target_arch = "wasm32"))]
        let vector_results = self.backend.search(&query.vector, fetch_size);
        #[cfg(target_arch = "wasm32")]
        let vector_results = self.backend.search(&query.vector, fetch_size)?;

        // Get BM25 scores if keywords provided
        let bm25_scores = if !query.keywords.is_empty() {
            self.text_index.bm25_scores(&query.keywords)
        } else {
            HashMap::new()
        };

        // Combine scores
        let combined = hybrid::combine_scores(vector_results, bm25_scores, query.alpha);

        // Apply filter and build results
        let mut results = Vec::new();

        for (id, score) in combined.into_iter().take(query.k * 2) {
            if let Some(record) = self.records.get(&id) {
                // Skip soft-deleted records (Major Issue #12 fix)
                if record.deleted {
                    continue;
                }

                // Apply filter if present
                if let Some(ref filter) = query.filter {
                    if !filters::evaluate_filter(filter, &record.metadata) {
                        continue;
                    }
                }

                results.push(Neighbor {
                    id: id.clone(),
                    score,
                    metadata: record.metadata.clone(),
                });

                if results.len() >= query.k {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Check if a document has indexed text
    pub fn has_text(&self, id: &str) -> bool {
        self.text_index.has_text(id)
    }

    /// Get the indexed text for a document
    pub fn get_text(&self, id: &str) -> Option<&str> {
        self.text_index.get_text(id)
    }

    /// Get all records in the store
    ///
    /// Returns a vector of all records, useful for iteration and export.
    pub fn list_all(&self) -> Vec<Record> {
        self.records.values().cloned().collect()
    }

    /// Soft delete a record (mark as deleted without removing)
    ///
    /// Soft deletes allow deferred cleanup and potential recovery.
    /// Call `compact()` to permanently remove soft-deleted records.
    ///
    /// # Arguments
    /// * `id` - ID of the record to soft delete
    ///
    /// # Returns
    /// * `Ok(true)` if record was soft deleted
    /// * `Ok(false)` if record doesn't exist or was already deleted
    pub fn soft_delete(&mut self, id: &str) -> Result<bool> {
        if let Some(record) = self.records.get_mut(id) {
            if !record.deleted {
                record.deleted = true;
                record.deleted_at = Some(Utc::now().timestamp());
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Restore a soft-deleted record
    ///
    /// # Arguments
    /// * `id` - ID of the record to restore
    ///
    /// # Returns
    /// * `Ok(true)` if record was restored
    /// * `Ok(false)` if record doesn't exist or wasn't deleted
    pub fn restore(&mut self, id: &str) -> Result<bool> {
        if let Some(record) = self.records.get_mut(id) {
            if record.deleted {
                record.deleted = false;
                record.deleted_at = None;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Permanently remove all soft-deleted records (compaction)
    ///
    /// This frees up memory and disk space by removing records marked
    /// for deletion. Returns the number of records removed.
    ///
    /// # Returns
    /// Number of records permanently removed
    pub fn compact(&mut self) -> Result<usize> {
        // Find all soft-deleted record IDs
        let deleted_ids: Vec<String> = self
            .records
            .values()
            .filter(|r| r.deleted)
            .map(|r| r.id.clone())
            .collect();

        let count = deleted_ids.len();

        // Permanently remove them
        for id in deleted_ids {
            self.backend.remove(&id)?;
            self.records.remove(&id);

            // Clean up text index (Critical Issue #4 fix)
            self.text_index.remove_document(&id);
        }

        Ok(count)
    }

    /// Get count of soft-deleted records
    pub fn deleted_count(&self) -> usize {
        self.records.values().filter(|r| r.deleted).count()
    }

    /// Get count of active (non-deleted) records
    pub fn active_count(&self) -> usize {
        self.records.values().filter(|r| !r.deleted).count()
    }

    /// List all soft-deleted records
    pub fn list_deleted(&self) -> Vec<Record> {
        self.records
            .values()
            .filter(|r| r.deleted)
            .cloned()
            .collect()
    }

    /// List all active (non-deleted) records
    pub fn list_active(&self) -> Vec<Record> {
        self.records
            .values()
            .filter(|r| !r.deleted)
            .cloned()
            .collect()
    }

    /// Execute a batch of mixed operations (upsert, delete, soft delete, restore, update metadata)
    ///
    /// This is significantly faster than executing operations individually and provides
    /// transactional-like semantics with detailed error reporting for each failed operation.
    ///
    /// # Arguments
    /// * `operations` - Vector of BatchOperation variants to execute
    ///
    /// # Returns
    /// * `BatchResult` with success/failure counts, detailed errors, and duration
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::{VecStore, BatchOperation, Metadata};
    /// # use std::collections::HashMap;
    /// # let mut store = VecStore::open("./data").unwrap();
    /// let operations = vec![
    ///     BatchOperation::Upsert {
    ///         id: "doc1".into(),
    ///         vector: vec![0.1, 0.2, 0.3],
    ///         metadata: Metadata { fields: HashMap::new() },
    ///     },
    ///     BatchOperation::Delete { id: "doc2".into() },
    ///     BatchOperation::SoftDelete { id: "doc3".into() },
    /// ];
    ///
    /// let result = store.batch_execute(operations).unwrap();
    /// println!("Succeeded: {}, Failed: {}", result.succeeded, result.failed);
    /// ```
    pub fn batch_execute(&mut self, operations: Vec<BatchOperation>) -> Result<BatchResult> {
        let start = std::time::Instant::now();
        let mut succeeded = 0;
        let mut failed = 0;
        let mut errors = Vec::new();

        for (index, op) in operations.into_iter().enumerate() {
            let result = match &op {
                BatchOperation::Upsert {
                    id,
                    vector,
                    metadata,
                } => self
                    .upsert(id.clone(), vector.clone(), metadata.clone())
                    .map_err(|e| (format!("upsert({})", id), e)),
                BatchOperation::Delete { id } => {
                    self.remove(id).map_err(|e| (format!("delete({})", id), e))
                }
                BatchOperation::SoftDelete { id } => self
                    .soft_delete(id)
                    .map(|_| ())
                    .map_err(|e| (format!("soft_delete({})", id), e)),
                BatchOperation::Restore { id } => self
                    .restore(id)
                    .map(|_| ())
                    .map_err(|e| (format!("restore({})", id), e)),
                BatchOperation::UpdateMetadata { id, metadata } => {
                    if let Some(record) = self.records.get_mut(id) {
                        record.metadata = metadata.clone();
                        Ok(())
                    } else {
                        Err((
                            format!("update_metadata({})", id),
                            anyhow::anyhow!("Record not found: {}", id),
                        ))
                    }
                }
            };

            match result {
                Ok(_) => succeeded += 1,
                Err((operation, error)) => {
                    failed += 1;
                    errors.push(BatchError {
                        index,
                        operation,
                        error: error.to_string(),
                    });
                }
            }
        }

        Ok(BatchResult {
            succeeded,
            failed,
            errors,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Convenience method: Delete a record (alias for remove)
    ///
    /// This provides consistency with BatchOperation::Delete naming
    pub fn delete(&mut self, id: &str) -> Result<()> {
        self.remove(id)
    }

    /// Update only the metadata of an existing record
    ///
    /// # Arguments
    /// * `id` - ID of the record to update
    /// * `metadata` - New metadata to set
    ///
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err` if record not found
    pub fn update_metadata(&mut self, id: &str, metadata: Metadata) -> Result<()> {
        if let Some(record) = self.records.get_mut(id) {
            record.metadata = metadata;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Record not found: {}", id))
        }
    }

    /// Estimate query cost and validate query parameters
    ///
    /// This provides pre-flight validation and cost estimation for queries,
    /// helping users understand query complexity and catch errors early.
    ///
    /// # Arguments
    /// * `q` - The query to estimate
    ///
    /// # Returns
    /// * `QueryEstimate` with validation results and cost information
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::{VecStore, Query};
    /// # let store = VecStore::open("./data").unwrap();
    /// let query = Query {
    ///     vector: vec![0.1, 0.2, 0.3],
    ///     k: 100,
    ///     filter: None,
    /// };
    ///
    /// let estimate = store.estimate_query(&query);
    /// if !estimate.valid {
    ///     println!("Query errors: {:?}", estimate.errors);
    /// } else {
    ///     println!("Estimated cost: {}", estimate.cost_estimate);
    ///     println!("Estimated duration: {}ms", estimate.estimated_duration_ms);
    /// }
    /// ```
    pub fn estimate_query(&self, q: &Query) -> QueryEstimate {
        let mut errors = Vec::new();
        let mut recommendations = Vec::new();

        // Validate dimension
        if self.dimension == 0 {
            return QueryEstimate {
                valid: true,
                errors: vec![],
                cost_estimate: 0.0,
                estimated_distance_calculations: 0,
                estimated_nodes_visited: 0,
                will_overfetch: false,
                recommendations: vec!["Store is empty, query will return no results".to_string()],
                estimated_duration_ms: 0.0,
            };
        }

        if q.vector.len() != self.dimension {
            errors.push(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                q.vector.len()
            ));
        }

        // Validate k
        if q.k == 0 {
            errors.push("k must be greater than 0".to_string());
        }

        if q.k > 1000 {
            recommendations.push(format!(
                "Large k value ({}). Consider reducing for better performance",
                q.k
            ));
        }

        // Determine if query is valid
        let valid = errors.is_empty();

        // Estimate computational cost
        let total_records = self.records.len();
        let active_records = self.active_count();

        // Calculate fetch size (same logic as query())
        let fetch_size = if q.filter.is_some() {
            if total_records <= q.k {
                total_records
            } else {
                // Using saturating_mul to prevent overflow (Major Issue #10 fix)
                std::cmp::min(q.k.saturating_mul(10), total_records)
            }
        } else {
            std::cmp::min(q.k, total_records)
        };

        // Check for over-fetching
        let will_overfetch = q.filter.is_some() && fetch_size > q.k;
        if will_overfetch {
            recommendations.push(format!(
                "Filter will cause over-fetching: requesting {} candidates to find {} results",
                fetch_size, q.k
            ));
        }

        // Estimate distance calculations (HNSW typically visits ~log(n) * M nodes)
        let m = 16; // Default HNSW M parameter
        let estimated_nodes = if total_records <= 1 {
            1
        } else {
            ((total_records as f32).log2() * m as f32).min(fetch_size as f32) as usize
        };

        // Cost estimate (0.0 - 1.0)
        // Factors: k, filter presence, over-fetching, database size
        let base_cost = (q.k as f32 / 100.0).min(1.0) * 0.3;
        let filter_cost = if q.filter.is_some() { 0.3 } else { 0.0 };
        let overfetch_cost = if will_overfetch { 0.2 } else { 0.0 };
        let size_cost = (total_records as f32 / 100000.0).min(1.0) * 0.2;

        let cost_estimate = (base_cost + filter_cost + overfetch_cost + size_cost).min(1.0);

        // Estimate duration (very rough heuristic)
        // Base: 0.1ms per distance calculation
        // Filter overhead: +50%
        let base_duration = estimated_nodes as f32 * 0.1;
        let filter_overhead = if q.filter.is_some() {
            base_duration * 0.5
        } else {
            0.0
        };
        let estimated_duration_ms = base_duration + filter_overhead;

        // Add recommendations
        if active_records < total_records {
            let deleted = total_records - active_records;
            recommendations.push(format!(
                "{} soft-deleted records present. Consider running compact() to improve performance",
                deleted
            ));
        }

        if q.filter.is_some() {
            recommendations.push(
                "Filtered queries are slower. Consider reducing k or adding indexes for better performance"
                    .to_string(),
            );
        }

        QueryEstimate {
            valid,
            errors,
            cost_estimate,
            estimated_distance_calculations: estimated_nodes,
            estimated_nodes_visited: estimated_nodes,
            will_overfetch,
            recommendations,
            estimated_duration_ms,
        }
    }

    /// Set auto-compaction configuration
    ///
    /// # Arguments
    /// * `config` - Compaction configuration
    pub fn set_compaction_config(&mut self, config: CompactionConfig) {
        self.compaction_config = config;
    }

    /// Get current compaction configuration
    pub fn compaction_config(&self) -> &CompactionConfig {
        &self.compaction_config
    }

    /// Check if auto-compaction should run and execute it if needed
    ///
    /// This method should be called periodically or after operations that may
    /// generate deleted records (e.g., after soft_delete or TTL expiration).
    ///
    /// # Returns
    /// * `CompactionResult` with statistics about the compaction
    pub fn maybe_compact(&mut self) -> Result<CompactionResult> {
        if !self.compaction_config.enabled {
            return Ok(CompactionResult {
                removed_count: 0,
                duration_ms: 0.0,
                triggered: false,
                reason: "Auto-compaction is disabled".to_string(),
            });
        }

        let deleted_count = self.deleted_count();
        let total_count = self.records.len();

        let deleted_ratio = if total_count > 0 {
            deleted_count as f32 / total_count as f32
        } else {
            0.0
        };

        let should_compact = deleted_count >= self.compaction_config.min_deleted_records
            && deleted_ratio >= self.compaction_config.min_deleted_ratio;

        if !should_compact {
            return Ok(CompactionResult {
                removed_count: 0,
                duration_ms: 0.0,
                triggered: false,
                reason: format!(
                    "Thresholds not met: {} deleted records ({:.1}% ratio), need {} records and {:.1}% ratio",
                    deleted_count,
                    deleted_ratio * 100.0,
                    self.compaction_config.min_deleted_records,
                    self.compaction_config.min_deleted_ratio * 100.0
                ),
            });
        }

        // Run compaction
        let start = std::time::Instant::now();
        let removed_count = self.compact()?;
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(CompactionResult {
            removed_count,
            duration_ms,
            triggered: true,
            reason: format!(
                "Compaction triggered: {} deleted records ({:.1}% ratio)",
                deleted_count,
                deleted_ratio * 100.0
            ),
        })
    }

    /// Expire TTL records (soft delete them)
    ///
    /// This scans all records and soft-deletes any that have passed their expiration time.
    ///
    /// # Returns
    /// * Number of records expired
    pub fn expire_ttl_records(&mut self) -> Result<usize> {
        let now = Utc::now().timestamp();
        let mut expired_count = 0;

        for record in self.records.values_mut() {
            if let Some(expires_at) = record.expires_at {
                if !record.deleted && now >= expires_at {
                    record.deleted = true;
                    record.deleted_at = Some(now);
                    expired_count += 1;
                }
            }
        }

        Ok(expired_count)
    }

    /// Set TTL for an existing record
    ///
    /// # Arguments
    /// * `id` - Record ID
    /// * `ttl_seconds` - Time-to-live in seconds from now
    ///
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err` if record not found
    pub fn set_ttl(&mut self, id: &str, ttl_seconds: i64) -> Result<()> {
        if let Some(record) = self.records.get_mut(id) {
            let expires_at = Utc::now().timestamp() + ttl_seconds;
            record.expires_at = Some(expires_at);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Record not found: {}", id))
        }
    }

    /// Upsert with TTL
    ///
    /// # Arguments
    /// * `id` - Record ID
    /// * `vector` - Vector data
    /// * `metadata` - Metadata
    /// * `ttl_seconds` - Time-to-live in seconds from now
    pub fn upsert_with_ttl(
        &mut self,
        id: Id,
        vector: Vec<f32>,
        metadata: Metadata,
        ttl_seconds: i64,
    ) -> Result<()> {
        // Set dimension on first insert
        if self.dimension == 0 {
            self.dimension = vector.len();
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.backend = VectorBackend::new(self.dimension, self.config.distance)?;
            }
            #[cfg(target_arch = "wasm32")]
            {
                self.backend = VectorBackend::new(self.dimension);
            }
        }

        if vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            ));
        }

        let expires_at = Utc::now().timestamp() + ttl_seconds;

        let record = Record {
            id: id.clone(),
            vector: vector.clone(),
            metadata,
            created_at: Utc::now().timestamp(),
            deleted: false,
            deleted_at: None,
            expires_at: Some(expires_at),
        };

        self.backend.insert(id.clone(), &vector)?;
        self.records.insert(id, record);

        Ok(())
    }

    /// Execute a multi-stage prefetch query for advanced RAG patterns
    ///
    /// This enables complex retrieval pipelines like:
    /// 1. Broad hybrid search (fetch 100 candidates)
    /// 2. Rerank with cross-encoder (score top 20)
    /// 3. Apply MMR for diversity (final 10 results)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vecstore::{VecStore, PrefetchQuery, QueryStage};
    /// # let temp_dir = tempfile::tempdir().unwrap();
    /// # let store = VecStore::open(temp_dir.path()).unwrap();
    /// let query = PrefetchQuery {
    ///     stages: vec![
    ///         QueryStage::HybridSearch {
    ///             vector: vec![0.1, 0.2, 0.3],
    ///             keywords: "machine learning".into(),
    ///             k: 100,
    ///             alpha: 0.7,
    ///             filter: None,
    ///         },
    ///         QueryStage::MMR {
    ///             k: 10,
    ///             lambda: 0.7, // 70% relevance, 30% diversity
    ///         },
    ///     ],
    /// };
    ///
    /// let results = store.prefetch_query(query)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn prefetch_query(&self, query: PrefetchQuery) -> Result<Vec<Neighbor>> {
        if query.stages.is_empty() {
            return Err(anyhow::anyhow!(
                "Prefetch query must have at least one stage"
            ));
        }

        // Validate first stage is a search stage (Major Issue #16 fix)
        if let Some(first_stage) = query.stages.first() {
            match first_stage {
                QueryStage::VectorSearch { .. } | QueryStage::HybridSearch { .. } => {
                    // Valid first stage
                }
                QueryStage::Rerank { .. } => {
                    return Err(anyhow::anyhow!(
                        "First stage must be VectorSearch or HybridSearch, not Rerank. Rerank requires previous search results."
                    ));
                }
                QueryStage::MMR { .. } => {
                    return Err(anyhow::anyhow!(
                        "First stage must be VectorSearch or HybridSearch, not MMR. MMR requires previous search results."
                    ));
                }
                QueryStage::Filter { .. } => {
                    return Err(anyhow::anyhow!(
                        "First stage must be VectorSearch or HybridSearch, not Filter. Filter requires previous search results."
                    ));
                }
            }
        }

        let mut candidates = Vec::new();

        for (stage_idx, stage) in query.stages.iter().enumerate() {
            candidates = match stage {
                QueryStage::VectorSearch { vector, k, filter } => {
                    // Stage 1: Vector search
                    self.query(Query {
                        vector: vector.clone(),
                        k: *k,
                        filter: filter.clone(),
                    })?
                }

                QueryStage::HybridSearch {
                    vector,
                    keywords,
                    k,
                    alpha,
                    filter,
                } => {
                    // Stage 1: Hybrid search
                    self.hybrid_query(HybridQuery {
                        vector: vector.clone(),
                        keywords: keywords.clone(),
                        k: *k,
                        alpha: *alpha,
                        filter: filter.clone(),
                    })?
                }

                QueryStage::Rerank { k, model: _ } => {
                    // Stage 2+: Rerank existing candidates
                    if candidates.is_empty() {
                        return Err(anyhow::anyhow!(
                            "Rerank stage {} requires previous stage results",
                            stage_idx
                        ));
                    }

                    // For now, just take top K (actual reranking would use cross-encoder)
                    // TODO: Integrate cross-encoder model if specified
                    candidates.truncate(*k);
                    candidates
                }

                QueryStage::MMR { k, lambda } => {
                    // Stage 2+: Maximal Marginal Relevance for diversity
                    if candidates.is_empty() {
                        return Err(anyhow::anyhow!(
                            "MMR stage {} requires previous stage results",
                            stage_idx
                        ));
                    }

                    self.apply_mmr(&candidates, *k, *lambda)?
                }

                QueryStage::Filter { expr } => {
                    // Stage 2+: Additional filtering
                    if candidates.is_empty() {
                        return Err(anyhow::anyhow!(
                            "Filter stage {} requires previous stage results",
                            stage_idx
                        ));
                    }

                    candidates
                        .into_iter()
                        .filter(|n| {
                            if let Some(record) = self.records.get(&n.id) {
                                filters::evaluate_filter(expr, &record.metadata)
                            } else {
                                false
                            }
                        })
                        .collect()
                }
            };
        }

        Ok(candidates)
    }

    /// Apply Maximal Marginal Relevance for diversity
    fn apply_mmr(&self, candidates: &[Neighbor], k: usize, lambda: f32) -> Result<Vec<Neighbor>> {
        use crate::simd::cosine_similarity_simd;

        if candidates.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let k = std::cmp::min(k, candidates.len());
        let mut selected = Vec::new();
        let mut remaining: Vec<_> = candidates.iter().collect();

        // Select first result (highest score)
        selected.push(remaining.remove(0).clone());

        // Iteratively select diverse results
        while selected.len() < k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_mmr_score = f32::NEG_INFINITY;

            for (idx, candidate) in remaining.iter().enumerate() {
                let candidate_record = match self.records.get(&candidate.id) {
                    Some(r) => r,
                    None => continue,
                };

                // Relevance score (from original query)
                let relevance = candidate.score;

                // Max similarity to already selected results
                let mut max_sim: f32 = 0.0;
                for selected_neighbor in &selected {
                    if let Some(selected_record) = self.records.get(&selected_neighbor.id) {
                        let sim = cosine_similarity_simd(
                            &candidate_record.vector,
                            &selected_record.vector,
                        );
                        max_sim = max_sim.max(sim);
                    }
                }

                // MMR score: lambda * relevance - (1 - lambda) * max_similarity
                let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim;

                if mmr_score > best_mmr_score {
                    best_mmr_score = mmr_score;
                    best_idx = idx;
                }
            }

            selected.push(remaining.remove(best_idx).clone());
        }

        Ok(selected)
    }

    /// Query with custom HNSW search parameters for performance tuning
    ///
    /// Allows per-query control over the speed/accuracy tradeoff.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vecstore::{VecStore, Query, HNSWSearchParams};
    /// # let temp_dir = tempfile::tempdir().unwrap();
    /// # let store = VecStore::open(temp_dir.path()).unwrap();
    /// // Fast search (lower recall, faster)
    /// let results = store.query_with_params(
    ///     Query::new(vec![0.1, 0.2, 0.3]).with_limit(10),
    ///     HNSWSearchParams::fast(),
    /// )?;
    ///
    /// // High recall search (better accuracy, slower)
    /// let results = store.query_with_params(
    ///     Query::new(vec![0.1, 0.2, 0.3]).with_limit(10),
    ///     HNSWSearchParams::high_recall(),
    /// )?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn query_with_params(&self, q: Query, params: HNSWSearchParams) -> Result<Vec<Neighbor>> {
        if self.dimension == 0 {
            return Ok(Vec::new());
        }

        if q.vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                q.vector.len()
            ));
        }

        // Validate ef_search parameter (Major Issue #14 fix)
        if params.ef_search == 0 {
            return Err(anyhow::anyhow!(
                "Invalid ef_search parameter: must be at least 1, got 0"
            ));
        }

        // Use custom ef_search parameter
        let fetch_size = if q.filter.is_some() {
            // Using saturating_mul to prevent overflow (Major Issue #10 fix)
            std::cmp::min(q.k.saturating_mul(10), self.records.len())
        } else {
            // Ensure we fetch at least k neighbors (Major Issue #8 fix)
            std::cmp::min(std::cmp::max(q.k, params.ef_search), self.records.len())
        };

        #[cfg(not(target_arch = "wasm32"))]
        let backend_results = self
            .backend
            .search_with_ef(&q.vector, fetch_size, params.ef_search)
            .unwrap_or_else(|_| self.backend.search(&q.vector, fetch_size));

        #[cfg(target_arch = "wasm32")]
        let backend_results = self
            .backend
            .search_with_ef(&q.vector, fetch_size, params.ef_search)
            .or_else(|_| self.backend.search(&q.vector, fetch_size))?;

        // Convert (Id, f32) to Neighbor with metadata
        let mut results: Vec<Neighbor> = backend_results
            .into_iter()
            .filter_map(|(id, score)| {
                self.records.get(&id).map(|record| Neighbor {
                    id,
                    score,
                    metadata: record.metadata.clone(),
                })
            })
            .collect();

        // Apply filter if present
        if let Some(ref filter) = q.filter {
            results.retain(|n| {
                if let Some(record) = self.records.get(&n.id) {
                    filters::evaluate_filter(filter, &record.metadata)
                } else {
                    false
                }
            });
        }

        // Limit to k results
        results.truncate(q.k);

        Ok(results)
    }

    /// Explain how a query will be executed and estimate its cost
    ///
    /// This is useful for:
    /// - Understanding query performance
    /// - Optimizing complex queries
    /// - Debugging slow queries
    /// - Capacity planning
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vecstore::{VecStore, Query};
    /// # let temp_dir = tempfile::tempdir().unwrap();
    /// # let store = VecStore::open(temp_dir.path()).unwrap();
    /// let query = Query::new(vec![0.1, 0.2, 0.3])
    ///     .with_limit(10)
    ///     .with_filter("category = 'tech' AND score > 0.9");
    ///
    /// let plan = store.explain_query(query)?;
    ///
    /// println!("Query type: {}", plan.query_type);
    /// println!("Estimated cost: {:.2}", plan.estimated_cost);
    /// println!("Estimated duration: {:.2}ms", plan.estimated_duration_ms);
    ///
    /// for step in plan.steps {
    ///     println!("  Step {}: {} (cost: {:.2})", step.step, step.description, step.cost);
    /// }
    ///
/// for rec in plan.recommendations {
///     println!("Hint: {}", rec);
/// }
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn explain_query(&self, q: Query) -> Result<QueryPlan> {
        // Validate query vector is not empty (Major Issue #15 fix)
        if q.vector.is_empty() {
            return Err(anyhow::anyhow!(
                "Query vector cannot be empty. Provide a valid embedding vector."
            ));
        }

        // Validate dimension matches store dimension (if store is initialized)
        if self.dimension > 0 && q.vector.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                q.vector.len()
            ));
        }

        let total_records = self.records.len();
        let has_filter = q.filter.is_some();

        let mut steps = Vec::new();
        let mut estimated_cost = 0.0;
        let mut recommendations = Vec::new();

        // Handle empty or near-empty stores (Major Issue #11 fix)
        if total_records < 2 {
            recommendations.push(
                "Store has fewer than 2 records. Ingest more data for meaningful query planning."
                    .to_string(),
            );
            return Ok(QueryPlan {
                query_type: "vector".to_string(),
                steps,
                estimated_cost: 0.0,
                estimated_duration_ms: 0.0,
                recommendations,
                is_optimal: false,
            });
        }

        // Step 1: HNSW graph traversal
        let hnsw_search_size = if has_filter {
            // Using saturating_mul to prevent overflow (Major Issue #10 fix)
            std::cmp::min(q.k.saturating_mul(10), total_records)
        } else {
            // Using saturating_mul to prevent overflow (Major Issue #10 fix)
            std::cmp::min(q.k.saturating_mul(2), total_records)
        };

        let hnsw_cost = (hnsw_search_size as f32).ln() / (total_records as f32).ln();
        estimated_cost += hnsw_cost;

        steps.push(QueryStep {
            step: 1,
            description: format!(
                "HNSW graph traversal (ef_search=50, fetch={})",
                hnsw_search_size
            ),
            cost: hnsw_cost,
            input_size: total_records,
            output_size: hnsw_search_size,
        });

        // Step 2: Filter application (if present)
        if let Some(ref filter) = q.filter {
            let filter_selectivity = self.estimate_filter_selectivity(filter);
            let filter_cost = 0.1 * (1.0 - filter_selectivity); // More selective = cheaper

            estimated_cost += filter_cost;

            let expected_pass_filter = (hnsw_search_size as f32 * filter_selectivity) as usize;

            steps.push(QueryStep {
                step: 2,
                description: format!(
                    "Apply filter (selectivity: {:.1}%)",
                    filter_selectivity * 100.0
                ),
                cost: filter_cost,
                input_size: hnsw_search_size,
                output_size: expected_pass_filter,
            });

            // Recommendation: if selectivity is very low, warn about over-fetching
            if filter_selectivity < 0.01 {
                recommendations.push(
                    "⚠️ Very low filter selectivity (<1%). Consider indexing filter fields or using a different filter strategy.".into()
                );
            }

            // Recommendation: if we're fetching way more than k, suggest optimization
            if hnsw_search_size > q.k * 20 {
                recommendations.push(format!(
                    "💡 Fetching {}x more candidates than needed. Consider using filtered HNSW traversal.",
                    hnsw_search_size / q.k
                ));
            }
        }

        // Step 3: Top-K selection
        let final_step = steps.len() + 1;
        steps.push(QueryStep {
            step: final_step,
            description: format!("Select top-{} results", q.k),
            cost: 0.05,
            input_size: if has_filter {
                steps
                    .last()
                    .map(|s| s.output_size)
                    .unwrap_or(hnsw_search_size)
            } else {
                hnsw_search_size
            },
            output_size: q.k,
        });

        estimated_cost += 0.05;

        // Estimate duration (very rough heuristic)
        let estimated_duration_ms = if total_records < 1000 {
            0.5 + estimated_cost * 2.0
        } else if total_records < 100_000 {
            1.0 + estimated_cost * 5.0
        } else {
            2.0 + estimated_cost * 10.0
        };

        // Is this query optimal?
        let is_optimal = recommendations.is_empty();

        // Additional recommendations
        if q.k > 100 {
            recommendations.push(
                "💡 Large k value (>100). Consider using pagination or reducing k for better performance.".into()
            );
        }

        if !is_optimal {
            recommendations
                .push("💡 Run EXPLAIN again after optimizations to see improvement.".into());
        }

        Ok(QueryPlan {
            query_type: if has_filter {
                "Filtered Vector Search".into()
            } else {
                "Vector Search".into()
            },
            steps,
            estimated_cost,
            estimated_duration_ms,
            recommendations,
            is_optimal,
        })
    }

    /// Estimate the selectivity of a filter (what fraction of records pass)
    fn estimate_filter_selectivity(&self, _filter: &FilterExpr) -> f32 {
        // TODO: Implement actual selectivity estimation based on metadata statistics
        // For now, return a conservative estimate
        0.1 // Assume 10% of records pass filter
    }

    /// Create a graph visualizer for the HNSW index
    ///
    /// Allows exporting the HNSW graph structure to various visualization formats:
    /// - DOT format for Graphviz rendering
    /// - JSON format for D3.js interactive visualizations
    /// - Cytoscape.js format for web-based graph visualization
    ///
    /// Note: Graph visualization is currently only supported for WASM builds.
    /// The native build uses hnsw_rs which doesn't expose the graph structure.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vecstore::VecStore;
    /// # let temp_dir = tempfile::tempdir().unwrap();
    /// let store = VecStore::open(temp_dir.path())?;
    ///
    /// // Export to Graphviz DOT format
    /// let viz = store.visualizer()?;
    /// let dot = viz.export_dot()?;
    /// std::fs::write("graph.dot", dot)?;
    ///
    /// // Render with: dot -Tpng graph.dot -o graph.png
    ///
    /// // Or export to D3.js JSON
    /// let json = viz.export_json()?;
    /// std::fs::write("graph.json", json)?;
    ///
    /// // Get graph statistics
    /// let stats = viz.statistics();
    /// println!("Nodes: {}, Edges: {}, Layers: {}",
    ///          stats.node_count, stats.edge_count, stats.layer_count);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn visualizer(&self) -> Result<crate::graph_viz::HnswVisualizer> {
        self.backend.to_visualizer()
    }
}

pub fn make_record(id: impl Into<String>, vector: Vec<f32>, metadata: Metadata) -> Record {
    Record {
        id: id.into(),
        vector,
        metadata,
        created_at: Utc::now().timestamp(),
        deleted: false,
        deleted_at: None,
        expires_at: None,
    }
}

#[cfg(test)]
mod soft_delete_tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_store() -> (VecStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();

        let mut meta1 = Metadata {
            fields: std::collections::HashMap::new(),
        };
        meta1
            .fields
            .insert("title".into(), serde_json::json!("Document 1"));
        store
            .upsert("doc1".into(), vec![1.0, 2.0, 3.0], meta1)
            .unwrap();

        let mut meta2 = Metadata {
            fields: std::collections::HashMap::new(),
        };
        meta2
            .fields
            .insert("title".into(), serde_json::json!("Document 2"));
        store
            .upsert("doc2".into(), vec![4.0, 5.0, 6.0], meta2)
            .unwrap();

        let mut meta3 = Metadata {
            fields: std::collections::HashMap::new(),
        };
        meta3
            .fields
            .insert("title".into(), serde_json::json!("Document 3"));
        store
            .upsert("doc3".into(), vec![7.0, 8.0, 9.0], meta3)
            .unwrap();

        (store, temp_dir)
    }

    #[test]
    fn test_soft_delete_basic() {
        let (mut store, _temp_dir) = create_test_store();

        assert_eq!(store.len(), 3);
        assert_eq!(store.active_count(), 3);
        assert_eq!(store.deleted_count(), 0);

        // Soft delete a record
        let result = store.soft_delete("doc1").unwrap();
        assert!(result);

        // Count should decrease
        assert_eq!(store.active_count(), 2);
        assert_eq!(store.deleted_count(), 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_soft_delete_nonexistent() {
        let (mut store, _temp_dir) = create_test_store();

        // Try to delete non-existent record
        let result = store.soft_delete("nonexistent").unwrap();
        assert!(!result);

        assert_eq!(store.active_count(), 3);
        assert_eq!(store.deleted_count(), 0);
    }

    #[test]
    fn test_soft_delete_twice() {
        let (mut store, _temp_dir) = create_test_store();

        // Delete once
        assert!(store.soft_delete("doc1").unwrap());
        assert_eq!(store.deleted_count(), 1);

        // Delete again - should return false
        assert!(!store.soft_delete("doc1").unwrap());
        assert_eq!(store.deleted_count(), 1); // Count unchanged
    }

    #[test]
    fn test_restore_deleted() {
        let (mut store, _temp_dir) = create_test_store();

        // Delete and restore
        store.soft_delete("doc1").unwrap();
        assert_eq!(store.active_count(), 2);
        assert_eq!(store.deleted_count(), 1);

        let result = store.restore("doc1").unwrap();
        assert!(result);

        assert_eq!(store.active_count(), 3);
        assert_eq!(store.deleted_count(), 0);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_restore_nondeleted() {
        let (mut store, _temp_dir) = create_test_store();

        // Try to restore a record that isn't deleted
        let result = store.restore("doc1").unwrap();
        assert!(!result);

        assert_eq!(store.active_count(), 3);
    }

    #[test]
    fn test_query_excludes_deleted() {
        let (mut store, _temp_dir) = create_test_store();

        // Query before deletion
        let query = Query {
            vector: vec![1.0, 2.0, 3.0],
            k: 10,
            filter: None,
        };
        let results = store.query(query.clone()).unwrap();
        assert_eq!(results.len(), 3);

        // Soft delete a record
        store.soft_delete("doc1").unwrap();

        // Query should exclude deleted record
        let results = store.query(query).unwrap();
        assert_eq!(results.len(), 2);

        // Ensure doc1 is not in results
        assert!(!results.iter().any(|n| n.id == "doc1"));
    }

    #[test]
    fn test_compact() {
        let (mut store, _temp_dir) = create_test_store();

        // Soft delete two records
        store.soft_delete("doc1").unwrap();
        store.soft_delete("doc2").unwrap();

        assert_eq!(store.active_count(), 1);
        assert_eq!(store.deleted_count(), 2);

        // Compact (permanently remove)
        let removed = store.compact().unwrap();
        assert_eq!(removed, 2);

        // Deleted records should be gone
        assert_eq!(store.deleted_count(), 0);
        assert_eq!(store.active_count(), 1);

        // Total records should be 1
        assert_eq!(store.records.len(), 1);
    }

    #[test]
    fn test_list_deleted() {
        let (mut store, _temp_dir) = create_test_store();

        store.soft_delete("doc1").unwrap();
        store.soft_delete("doc2").unwrap();

        let deleted = store.list_deleted();
        assert_eq!(deleted.len(), 2);
        assert!(deleted.iter().any(|r| r.id == "doc1"));
        assert!(deleted.iter().any(|r| r.id == "doc2"));
    }

    #[test]
    fn test_list_active() {
        let (mut store, _temp_dir) = create_test_store();

        store.soft_delete("doc1").unwrap();

        let active = store.list_active();
        assert_eq!(active.len(), 2);
        assert!(!active.iter().any(|r| r.id == "doc1"));
        assert!(active.iter().any(|r| r.id == "doc2"));
        assert!(active.iter().any(|r| r.id == "doc3"));
    }

    #[test]
    fn test_deleted_at_timestamp() {
        let (mut store, _temp_dir) = create_test_store();

        store.soft_delete("doc1").unwrap();

        let deleted = store.list_deleted();
        assert_eq!(deleted.len(), 1);
        assert!(deleted[0].deleted);
        assert!(deleted[0].deleted_at.is_some());
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_builder_default() {
        let temp_dir = TempDir::new().unwrap();
        let store = VecStore::builder(temp_dir.path().join("test.db"))
            .build()
            .unwrap();

        // Default should be Cosine
        assert_eq!(store.distance_metric(), Distance::Cosine);
        assert_eq!(store.config().hnsw_m, 16);
        assert_eq!(store.config().hnsw_ef_construction, 200);
    }

    #[test]
    fn test_builder_manhattan() {
        // Manhattan is not yet supported by HNSW backend - test that it returns error
        let temp_dir = TempDir::new().unwrap();
        let result = VecStore::builder(temp_dir.path().join("test.db"))
            .distance(Distance::Manhattan)
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("not yet supported"));
        }
    }

    #[test]
    fn test_builder_hamming() {
        // Hamming is not yet supported by HNSW backend - test that it returns error
        let temp_dir = TempDir::new().unwrap();
        let result = VecStore::builder(temp_dir.path().join("test.db"))
            .distance(Distance::Hamming)
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("not yet supported"));
        }
    }

    #[test]
    fn test_builder_jaccard() {
        // Jaccard is not yet supported by HNSW backend - test that it returns error
        let temp_dir = TempDir::new().unwrap();
        let result = VecStore::builder(temp_dir.path().join("test.db"))
            .distance(Distance::Jaccard)
            .build();

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("not yet supported"));
        }
    }

    #[test]
    fn test_builder_custom_hnsw_params() {
        let temp_dir = TempDir::new().unwrap();
        let store = VecStore::builder(temp_dir.path().join("test.db"))
            .hnsw_m(32)
            .hnsw_ef_construction(400)
            .build()
            .unwrap();

        assert_eq!(store.config().hnsw_m, 32);
        assert_eq!(store.config().hnsw_ef_construction, 400);
    }

    #[test]
    fn test_builder_chained() {
        let temp_dir = TempDir::new().unwrap();
        let store = VecStore::builder(temp_dir.path().join("test.db"))
            .distance(Distance::Euclidean)
            .hnsw_m(64)
            .hnsw_ef_construction(500)
            .build()
            .unwrap();

        assert_eq!(store.distance_metric(), Distance::Euclidean);
        assert_eq!(store.config().hnsw_m, 64);
        assert_eq!(store.config().hnsw_ef_construction, 500);
    }

    #[test]
    fn test_builder_backward_compatibility() {
        let temp_dir = TempDir::new().unwrap();

        // Old way still works
        let store_old = VecStore::open(temp_dir.path().join("old.db")).unwrap();
        assert_eq!(store_old.distance_metric(), Distance::Cosine);

        // New way also works
        let store_new = VecStore::builder(temp_dir.path().join("new.db"))
            .build()
            .unwrap();
        assert_eq!(store_new.distance_metric(), Distance::Cosine);

        // Both should have the same default config
        assert_eq!(store_old.config().distance, store_new.config().distance);
    }

    #[test]
    fn test_distance_enum_from_str() {
        assert_eq!(Distance::from_str("cosine").unwrap(), Distance::Cosine);
        assert_eq!(
            Distance::from_str("euclidean").unwrap(),
            Distance::Euclidean
        );
        assert_eq!(Distance::from_str("l2").unwrap(), Distance::Euclidean);
        assert_eq!(
            Distance::from_str("manhattan").unwrap(),
            Distance::Manhattan
        );
        assert_eq!(Distance::from_str("l1").unwrap(), Distance::Manhattan);
        assert_eq!(Distance::from_str("hamming").unwrap(), Distance::Hamming);
        assert_eq!(Distance::from_str("jaccard").unwrap(), Distance::Jaccard);
        assert_eq!(Distance::from_str("dot").unwrap(), Distance::DotProduct);

        // Case insensitive
        assert_eq!(
            Distance::from_str("MANHATTAN").unwrap(),
            Distance::Manhattan
        );
        assert_eq!(Distance::from_str("Hamming").unwrap(), Distance::Hamming);

        // Invalid
        assert!(Distance::from_str("invalid").is_err());
    }

    #[test]
    fn test_distance_enum_name_and_description() {
        assert_eq!(Distance::Cosine.name(), "Cosine");
        assert_eq!(Distance::Manhattan.name(), "Manhattan");
        assert_eq!(Distance::Hamming.name(), "Hamming");
        assert_eq!(Distance::Jaccard.name(), "Jaccard");

        assert!(Distance::Cosine.description().contains("angle"));
        assert!(Distance::Manhattan.description().contains("outliers"));
        assert!(Distance::Hamming.description().contains("binary"));
        assert!(Distance::Jaccard.description().contains("set"));
    }
}

#[cfg(test)]
mod text_index_persistence_tests {
    use super::*;
    use tempfile::TempDir;

    /// Test that text index survives store reopen (Major Issue #6 fix)
    #[test]
    fn test_text_index_persists_across_reopen() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.db");

        // Create store, index text, and save
        {
            let mut store = VecStore::open(&path).unwrap();

            let mut meta = Metadata {
                fields: std::collections::HashMap::new(),
            };
            meta.fields
                .insert("title".into(), serde_json::json!("Test Doc"));

            store
                .upsert("doc1".into(), vec![1.0, 2.0, 3.0], meta)
                .unwrap();
            store
                .index_text("doc1", "hello world test".to_string())
                .unwrap();
            store.save().unwrap();

            // Verify text is indexed
            assert!(store.has_text("doc1"));
            assert_eq!(store.get_text("doc1"), Some("hello world test"));
        }

        // Reopen store and verify text survives
        {
            let store = VecStore::open(&path).unwrap();
            assert!(
                store.has_text("doc1"),
                "Text index should survive store reopen"
            );
            assert_eq!(store.get_text("doc1"), Some("hello world test"));
        }
    }

    /// Test that text index survives snapshot restore (Major Issue #6 fix)
    #[test]
    fn test_text_index_persists_in_snapshots() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.db");

        let mut store = VecStore::open(&path).unwrap();

        let mut meta = Metadata {
            fields: std::collections::HashMap::new(),
        };
        meta.fields
            .insert("title".into(), serde_json::json!("Snapshot Doc"));

        store
            .upsert("doc1".into(), vec![1.0, 2.0, 3.0], meta.clone())
            .unwrap();
        store
            .index_text("doc1", "important data".to_string())
            .unwrap();

        // Create snapshot
        store.create_snapshot("backup1").unwrap();

        // Modify store
        store
            .upsert("doc2".into(), vec![4.0, 5.0, 6.0], meta)
            .unwrap();
        store.index_text("doc2", "new data".to_string()).unwrap();

        // Restore snapshot
        store.restore_snapshot("backup1").unwrap();

        // Text index should be restored to snapshot state
        assert!(store.has_text("doc1"));
        assert_eq!(store.get_text("doc1"), Some("important data"));
        assert!(
            !store.has_text("doc2"),
            "doc2 should not exist after restore"
        );
    }

    /// Test that text index is properly cleared when store is new
    #[test]
    fn test_text_index_empty_on_new_store() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.db");

        let store = VecStore::open(&path).unwrap();
        assert!(!store.has_text("anything"));
    }

    /// Test that multiple documents can be indexed and persisted
    #[test]
    fn test_multiple_texts_persist() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.db");

        {
            let mut store = VecStore::open(&path).unwrap();

            for i in 1..=5 {
                let id = format!("doc{}", i);
                let text = format!("Document number {}", i);
                let mut meta = Metadata {
                    fields: std::collections::HashMap::new(),
                };
                meta.fields.insert("num".into(), serde_json::json!(i));

                store.upsert(id.clone(), vec![i as f32; 3], meta).unwrap();
                store.index_text(&id, text).unwrap();
            }
            store.save().unwrap();
        }

        {
            let store = VecStore::open(&path).unwrap();
            for i in 1..=5 {
                let id = format!("doc{}", i);
                let expected_text = format!("Document number {}", i);
                assert!(store.has_text(&id), "doc{} should have text", i);
                assert_eq!(store.get_text(&id), Some(expected_text.as_str()));
            }
        }
    }
}
