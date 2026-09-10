//! Edge Runtime
//!
//! Lightweight runtime for edge deployments on Cloudflare Workers,
//! Vercel Edge Functions, Deno Deploy, and other edge platforms.
//!
//! # Features
//!
//! - **WASM-Compatible**: Runs in WebAssembly environments
//! - **Stateless Mode**: No filesystem dependencies
//! - **Memory-Efficient**: Compact index format
//! - **Fast Cold Start**: Minimal initialization time
//! - **Cross-Platform**: Works on any edge platform
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::edge_runtime::{EdgeVectorStore, EdgeConfig};
//!
//! // Create an edge-optimized store
//! let config = EdgeConfig::new()
//!     .with_max_vectors(10000)
//!     .with_quantization(true);
//!
//! let mut store = EdgeVectorStore::new(384, config)?;
//!
//! // Load from serialized state
//! store.load_from_bytes(&state_bytes)?;
//!
//! // Search
//! let results = store.search(&query, 10)?;
//!
//! // Serialize state for persistence
//! let state = store.to_bytes()?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, VecStoreError};

/// Edge runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    /// Maximum number of vectors
    #[serde(default = "default_max_vectors")]
    pub max_vectors: usize,
    /// Enable quantization for smaller memory footprint
    #[serde(default)]
    pub enable_quantization: bool,
    /// Quantization bits (8, 4, or 2)
    #[serde(default = "default_quant_bits")]
    pub quantization_bits: u8,
    /// Enable index building
    #[serde(default = "default_true")]
    pub build_index: bool,
    /// HNSW M parameter (smaller = less memory)
    #[serde(default = "default_m")]
    pub hnsw_m: usize,
    /// HNSW ef_construction
    #[serde(default = "default_ef")]
    pub hnsw_ef: usize,
}

fn default_max_vectors() -> usize {
    10000
}
fn default_quant_bits() -> u8 {
    8
}
fn default_true() -> bool {
    true
}
fn default_m() -> usize {
    8
}
fn default_ef() -> usize {
    64
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            max_vectors: 10000,
            enable_quantization: false,
            quantization_bits: 8,
            build_index: true,
            hnsw_m: 8,
            hnsw_ef: 64,
        }
    }
}

impl EdgeConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum vectors
    pub fn with_max_vectors(mut self, max: usize) -> Self {
        self.max_vectors = max;
        self
    }

    /// Enable quantization
    pub fn with_quantization(mut self, enabled: bool) -> Self {
        self.enable_quantization = enabled;
        self
    }

    /// Set quantization bits
    pub fn with_quantization_bits(mut self, bits: u8) -> Self {
        self.quantization_bits = bits;
        self
    }

    /// Optimize for minimal memory
    pub fn minimal_memory() -> Self {
        Self {
            max_vectors: 1000,
            enable_quantization: true,
            quantization_bits: 4,
            build_index: false,
            hnsw_m: 4,
            hnsw_ef: 32,
        }
    }

    /// Optimize for low latency
    pub fn low_latency() -> Self {
        Self {
            max_vectors: 10000,
            enable_quantization: true,
            quantization_bits: 8,
            build_index: true,
            hnsw_m: 16,
            hnsw_ef: 128,
        }
    }
}

/// Compact vector storage for edge
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompactVector {
    id: String,
    /// Full-precision or quantized vector
    data: VectorData,
    /// Metadata (optional, compact)
    metadata: Option<Vec<u8>>,
}

/// Vector data (full or quantized)
#[derive(Debug, Clone, Serialize, Deserialize)]
enum VectorData {
    Full(Vec<f32>),
    Quantized8(Vec<u8>, f32, f32), // data, min, scale
    Quantized4(Vec<u8>, f32, f32), // data (packed), min, scale
}

impl VectorData {
    /// Convert to full precision
    fn to_f32(&self, dimension: usize) -> Vec<f32> {
        match self {
            VectorData::Full(v) => v.clone(),
            VectorData::Quantized8(data, min, scale) => {
                data.iter().map(|&b| min + (b as f32) * scale).collect()
            },
            VectorData::Quantized4(data, min, scale) => {
                let mut result = Vec::with_capacity(dimension);
                for byte in data {
                    let lo = byte & 0x0F;
                    let hi = (byte >> 4) & 0x0F;
                    result.push(min + (lo as f32) * scale);
                    if result.len() < dimension {
                        result.push(min + (hi as f32) * scale);
                    }
                }
                result
            },
        }
    }

    /// Memory size in bytes
    fn memory_size(&self) -> usize {
        match self {
            VectorData::Full(v) => v.len() * 4,
            VectorData::Quantized8(data, _, _) => data.len() + 8,
            VectorData::Quantized4(data, _, _) => data.len() + 8,
        }
    }
}

/// Edge vector store
pub struct EdgeVectorStore {
    dimension: usize,
    config: EdgeConfig,
    vectors: Vec<CompactVector>,
    id_to_index: HashMap<String, usize>,
}

impl EdgeVectorStore {
    /// Create a new edge store
    pub fn new(dimension: usize, config: EdgeConfig) -> Result<Self> {
        Ok(Self {
            dimension,
            config,
            vectors: Vec::new(),
            id_to_index: HashMap::new(),
        })
    }

    /// Add a vector
    pub fn add(
        &mut self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        if self.vectors.len() >= self.config.max_vectors {
            return Err(VecStoreError::InvalidInput("Capacity exceeded".to_string()));
        }

        let data = if self.config.enable_quantization {
            self.quantize(&vector)
        } else {
            VectorData::Full(vector)
        };

        let metadata_bytes = metadata.map(|m| serde_json::to_vec(&m).unwrap_or_default());

        let index = self.vectors.len();
        self.vectors.push(CompactVector {
            id: id.to_string(),
            data,
            metadata: metadata_bytes,
        });
        self.id_to_index.insert(id.to_string(), index);

        Ok(())
    }

    /// Quantize a vector
    fn quantize(&self, vector: &[f32]) -> VectorData {
        let min = vector.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vector.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        match self.config.quantization_bits {
            8 => {
                let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
                let data: Vec<u8> = vector
                    .iter()
                    .map(|&v| ((v - min) / scale).round() as u8)
                    .collect();
                VectorData::Quantized8(data, min, scale)
            },
            4 => {
                let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
                let mut data = Vec::with_capacity(vector.len().div_ceil(2));
                for chunk in vector.chunks(2) {
                    let lo = ((chunk[0] - min) / scale).round() as u8;
                    let hi = if chunk.len() > 1 {
                        ((chunk[1] - min) / scale).round() as u8
                    } else {
                        0
                    };
                    data.push(lo | (hi << 4));
                }
                VectorData::Quantized4(data, min, scale)
            },
            _ => VectorData::Full(vector.to_vec()),
        }
    }

    /// Search for similar vectors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<EdgeSearchResult>> {
        if query.len() != self.dimension {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        let mut results: Vec<EdgeSearchResult> = self
            .vectors
            .iter()
            .map(|v| {
                let vec = v.data.to_f32(self.dimension);
                let score = Self::cosine_similarity(query, &vec);

                let metadata = v
                    .metadata
                    .as_ref()
                    .and_then(|bytes| serde_json::from_slice(bytes).ok());

                EdgeSearchResult {
                    id: v.id.clone(),
                    score,
                    metadata,
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Option<EdgeSearchResult> {
        self.id_to_index.get(id).map(|&idx| {
            let v = &self.vectors[idx];
            let metadata = v
                .metadata
                .as_ref()
                .and_then(|bytes| serde_json::from_slice(bytes).ok());

            EdgeSearchResult {
                id: v.id.clone(),
                score: 1.0,
                metadata,
            }
        })
    }

    /// Delete a vector
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        if let Some(&idx) = self.id_to_index.get(id) {
            // Swap remove for efficiency
            let last_idx = self.vectors.len() - 1;
            if idx != last_idx {
                let last_id = self.vectors[last_idx].id.clone();
                self.vectors.swap(idx, last_idx);
                self.id_to_index.insert(last_id, idx);
            }
            self.vectors.pop();
            self.id_to_index.remove(id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let state = EdgeState {
            dimension: self.dimension,
            config: self.config.clone(),
            vectors: self.vectors.clone(),
        };

        bincode::serialize(&state).map_err(|e| VecStoreError::Serialization(e.to_string()))
    }

    /// Load from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let state: EdgeState =
            bincode::deserialize(bytes).map_err(|e| VecStoreError::Serialization(e.to_string()))?;

        let id_to_index: HashMap<String, usize> = state
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (v.id.clone(), i))
            .collect();

        Ok(Self {
            dimension: state.dimension,
            config: state.config,
            vectors: state.vectors,
            id_to_index,
        })
    }

    /// Load from bytes into existing store
    pub fn load_from_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let loaded = Self::from_bytes(bytes)?;
        self.dimension = loaded.dimension;
        self.config = loaded.config;
        self.vectors = loaded.vectors;
        self.id_to_index = loaded.id_to_index;
        Ok(())
    }

    /// Calculate cosine similarity
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get memory usage estimate
    pub fn memory_usage(&self) -> MemoryUsage {
        let vector_bytes: usize = self.vectors.iter().map(|v| v.data.memory_size()).sum();

        let metadata_bytes: usize = self
            .vectors
            .iter()
            .filter_map(|v| v.metadata.as_ref())
            .map(|m| m.len())
            .sum();

        let id_bytes: usize = self.vectors.iter().map(|v| v.id.len()).sum();

        MemoryUsage {
            vector_bytes,
            metadata_bytes,
            id_bytes,
            total_bytes: vector_bytes + metadata_bytes + id_bytes,
        }
    }
}

/// Serializable state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EdgeState {
    dimension: usize,
    config: EdgeConfig,
    vectors: Vec<CompactVector>,
}

/// Search result
#[derive(Debug, Clone, Serialize)]
pub struct EdgeSearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct MemoryUsage {
    pub vector_bytes: usize,
    pub metadata_bytes: usize,
    pub id_bytes: usize,
    pub total_bytes: usize,
}

/// Edge-optimized embedding (for WASM)
#[cfg(target_arch = "wasm32")]
pub mod wasm {
    use super::*;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmEdgeStore {
        inner: EdgeVectorStore,
    }

    #[wasm_bindgen]
    impl WasmEdgeStore {
        #[wasm_bindgen(constructor)]
        pub fn new(dimension: usize) -> Result<WasmEdgeStore, JsValue> {
            let config = EdgeConfig::minimal_memory();
            EdgeVectorStore::new(dimension, config)
                .map(|inner| WasmEdgeStore { inner })
                .map_err(|e| JsValue::from_str(&e.to_string()))
        }

        #[wasm_bindgen]
        pub fn add(&mut self, id: &str, vector: &[f32]) -> Result<(), JsValue> {
            self.inner
                .add(id, vector.to_vec(), None)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        }

        #[wasm_bindgen]
        pub fn search(&self, query: &[f32], k: usize) -> Result<JsValue, JsValue> {
            let results = self
                .inner
                .search(query, k)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
        }

        #[wasm_bindgen]
        pub fn len(&self) -> usize {
            self.inner.len()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_store() {
        let config = EdgeConfig::new().with_max_vectors(100);
        let mut store = EdgeVectorStore::new(64, config).unwrap();

        store.add("doc1", vec![0.1f32; 64], None).unwrap();
        store.add("doc2", vec![0.2f32; 64], None).unwrap();

        let results = store.search(&vec![0.15f32; 64], 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_quantization() {
        let config = EdgeConfig::new()
            .with_quantization(true)
            .with_quantization_bits(8);

        let mut store = EdgeVectorStore::new(64, config).unwrap();

        let original = vec![0.5f32; 64];
        store.add("doc1", original.clone(), None).unwrap();

        let usage = store.memory_usage();
        // 8-bit quantization: 64 bytes vs 256 bytes (4x compression)
        assert!(usage.vector_bytes < 100);
    }

    #[test]
    fn test_serialization() {
        let config = EdgeConfig::new();
        let mut store = EdgeVectorStore::new(64, config).unwrap();

        store.add("doc1", vec![0.1f32; 64], None).unwrap();
        store.add("doc2", vec![0.2f32; 64], None).unwrap();

        let bytes = store.to_bytes().unwrap();
        let loaded = EdgeVectorStore::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
    }
}
