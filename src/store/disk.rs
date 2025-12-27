// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

use super::quantization::ProductQuantizer;
use super::types::{Config, Id, Record};
use crate::quantization::{BinaryQuantizer, ScalarQuantizer4, ScalarQuantizer8};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Serializable quantizer state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizerState {
    /// No quantization
    None,
    /// 8-bit scalar quantizer
    Scalar8(ScalarQuantizer8),
    /// 4-bit scalar quantizer
    Scalar4(ScalarQuantizer4),
    /// Binary quantizer
    Binary(BinaryQuantizer),
    /// Product quantizer
    Product(ProductQuantizer),
}

const SCHEMA_VERSION: u32 = 5; // Use JSON for records (bincode incompatible with serde_json::Value)

type LoadResult = (
    HashMap<Id, Record>,
    HashMap<Id, usize>,
    HashMap<usize, Id>,
    usize,
    usize,
    Option<Config>,              // Added config to load result (Major Issue #7 fix)
    Option<HashMap<Id, String>>, // Added text index data (Major Issue #6 fix)
);

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub dimension: usize,
    pub record_count: usize,
    pub next_idx: usize,

    /// Store configuration (Major Issue #7 fix)
    /// Optional for backward compatibility with schema_version 1
    #[serde(default)]
    pub config: Option<Config>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiskState {
    pub records: Vec<Record>,
    pub id_to_idx: HashMap<Id, usize>,
    pub idx_to_id: HashMap<usize, Id>,
    pub next_idx: usize,
}

pub struct DiskLayout {
    pub root: PathBuf,
}

impl DiskLayout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.json")
    }

    pub fn vectors_path(&self) -> PathBuf {
        self.root.join("vectors.bin")
    }

    pub fn meta_path(&self) -> PathBuf {
        self.root.join("meta.bin")
    }

    pub fn hnsw_path(&self) -> PathBuf {
        self.root.join("hnsw.idx")
    }

    pub fn text_index_path(&self) -> PathBuf {
        self.root.join("text_index.json")
    }

    /// Path for quantization codebooks (trained quantizer state)
    pub fn quantization_path(&self) -> PathBuf {
        self.root.join("quantization.bin")
    }

    pub fn ensure_directory(&self) -> Result<()> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("Failed to create directory: {:?}", self.root))?;
        Ok(())
    }

    pub fn exists(&self) -> bool {
        self.manifest_path().exists()
    }

    pub fn save_all(
        &self,
        records: &HashMap<Id, Record>,
        id_to_idx: &HashMap<Id, usize>,
        idx_to_id: &HashMap<usize, Id>,
        next_idx: usize,
        dimension: usize,
        config: &Config, // Major Issue #7 fix: persist config
        text_index_data: Option<&HashMap<Id, String>>, // Major Issue #6 fix: persist text index
    ) -> Result<()> {
        self.ensure_directory()?;

        // Prepare data
        let manifest = Manifest {
            schema_version: SCHEMA_VERSION,
            dimension,
            record_count: records.len(),
            next_idx,
            config: Some(config.clone()), // Major Issue #7 fix
        };

        let state = DiskState {
            records: records.values().cloned().collect(),
            id_to_idx: id_to_idx.clone(),
            idx_to_id: idx_to_id.clone(),
            next_idx,
        };

        // Atomic writes using temp files
        self.atomic_write(
            &self.manifest_path(),
            &serde_json::to_vec_pretty(&manifest)?,
        )?;
        // Use JSON for records (bincode is incompatible with serde_json::Value in Metadata)
        self.atomic_write(&self.vectors_path(), &serde_json::to_vec(&state.records)?)?;
        self.atomic_write(
            &self.meta_path(),
            &bincode::serialize(&(state.id_to_idx, state.idx_to_id, state.next_idx))?,
        )?;

        // Save text index if present (Major Issue #6 fix)
        if let Some(texts) = text_index_data {
            self.atomic_write(&self.text_index_path(), &serde_json::to_vec(texts)?)?;
        }

        Ok(())
    }

    pub fn load_all(&self) -> Result<LoadResult> {
        if !self.exists() {
            return Err(anyhow::anyhow!("Store does not exist at {:?}", self.root));
        }

        // Load manifest
        let manifest_data = fs::read(self.manifest_path()).context("Failed to read manifest")?;
        let manifest: Manifest =
            serde_json::from_slice(&manifest_data).context("Failed to parse manifest")?;

        // Support schema versions 1, 2, 3, 4, and 5 (backward compatibility)
        if manifest.schema_version > SCHEMA_VERSION {
            return Err(anyhow::anyhow!(
                "Unsupported schema version: {}. Maximum supported: {}",
                manifest.schema_version,
                SCHEMA_VERSION
            ));
        }

        // Load records - schema 5+ and 1-3 use JSON, schema 4 used bincode (deprecated)
        let records_data = fs::read(self.vectors_path()).context("Failed to read vectors")?;
        let records_vec: Vec<Record> = if manifest.schema_version == 4 {
            // Legacy binary format (bincode) - only schema 4
            // Note: This may fail if records contain serde_json::Value
            bincode::deserialize(&records_data)
                .or_else(|_| serde_json::from_slice(&records_data))
                .context("Failed to deserialize vectors")?
        } else {
            // JSON format for schema versions 1, 2, 3, and 5+
            serde_json::from_slice(&records_data).context("Failed to deserialize vectors (JSON)")?
        };

        let mut records = HashMap::new();
        for record in records_vec {
            records.insert(record.id.clone(), record);
        }

        // Load metadata
        let meta_data = fs::read(self.meta_path()).context("Failed to read metadata")?;
        let (id_to_idx, idx_to_id, next_idx): (HashMap<Id, usize>, HashMap<usize, Id>, usize) =
            bincode::deserialize(&meta_data).context("Failed to deserialize metadata")?;

        // Load text index if present (Major Issue #6 fix)
        // Only available in schema version 3+
        let text_index_data = if manifest.schema_version >= 3 && self.text_index_path().exists() {
            let text_data =
                fs::read(self.text_index_path()).context("Failed to read text index")?;
            let texts: HashMap<Id, String> =
                serde_json::from_slice(&text_data).context("Failed to deserialize text index")?;
            Some(texts)
        } else {
            None
        };

        // Return loaded config and text index (Major Issues #7 and #6 fixes)
        Ok((
            records,
            id_to_idx,
            idx_to_id,
            next_idx,
            manifest.dimension,
            manifest.config,
            text_index_data,
        ))
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()> {
        let temp_path = path.with_extension("tmp");
        fs::write(&temp_path, data)
            .with_context(|| format!("Failed to write temp file: {:?}", temp_path))?;
        fs::rename(&temp_path, path)
            .with_context(|| format!("Failed to rename temp file to: {:?}", path))?;
        Ok(())
    }

    /// Save quantizer state (trained codebooks) to disk
    pub fn save_quantizer(&self, state: &QuantizerState) -> Result<()> {
        // Only save if there's a trained quantizer
        if matches!(state, QuantizerState::None) {
            // Remove old quantizer file if exists
            let path = self.quantization_path();
            if path.exists() {
                fs::remove_file(&path).ok();
            }
            return Ok(());
        }

        let data = bincode::serialize(state).context("Failed to serialize quantizer state")?;
        self.atomic_write(&self.quantization_path(), &data)?;
        Ok(())
    }

    /// Load quantizer state from disk
    pub fn load_quantizer(&self) -> Result<QuantizerState> {
        let path = self.quantization_path();
        if !path.exists() {
            return Ok(QuantizerState::None);
        }

        let data = fs::read(&path).context("Failed to read quantizer state")?;
        let state: QuantizerState =
            bincode::deserialize(&data).context("Failed to deserialize quantizer state")?;
        Ok(state)
    }
}
