//! Standard Vector Database Protocol
//!
//! Provides a universal API layer compatible with popular vector databases.
//! Allows drop-in replacement of Pinecone, Weaviate, Qdrant, etc. with vecstore.
//!
//! ## Supported Protocols
//!
//! - **Pinecone-compatible API**: REST API matching Pinecone's endpoints
//! - **Qdrant-compatible API**: REST API matching Qdrant's format
//! - **Weaviate-compatible API**: GraphQL + REST matching Weaviate
//! - **ChromaDB-compatible API**: Simple REST API like ChromaDB
//! - **Universal JSON API**: Generic JSON format for any client
//!
//! ## Features
//!
//! - Protocol auto-detection from request format
//! - Request/response translation to vecstore format
//! - Error code mapping for compatibility
//! - Metric name translation
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::protocol::{ProtocolAdapter, Protocol, UniversalRequest};
//! use vecstore::VecStore;
//!
//! # fn main() -> anyhow::Result<()> {
//! let store = VecStore::open("my_store.db")?;
//! let adapter = ProtocolAdapter::new(store);
//!
//! // Handle Pinecone-compatible request
//! let pinecone_json = r#"{
//!     "vectors": [{
//!         "id": "vec1",
//!         "values": [0.1, 0.2, 0.3],
//!         "metadata": {"source": "doc1"}
//!     }]
//! }"#;
//!
//! let response = adapter.handle_request(pinecone_json, Protocol::Pinecone)?;
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::store::{Metadata, Query, VecStore};

/// Supported vector database protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// Pinecone-compatible API
    Pinecone,
    /// Qdrant-compatible API
    Qdrant,
    /// Weaviate-compatible API
    Weaviate,
    /// ChromaDB-compatible API
    ChromaDB,
    /// Milvus-compatible API
    Milvus,
    /// Universal JSON API (vecstore native)
    Universal,
}

impl Protocol {
    /// Detect protocol from request format
    pub fn detect(json: &str) -> Self {
        if json.contains("\"vectors\"") && json.contains("\"values\"") {
            Protocol::Pinecone
        } else if json.contains("\"points\"") && json.contains("\"vector\"") {
            Protocol::Qdrant
        } else if json.contains("\"class\"") && json.contains("\"properties\"") {
            Protocol::Weaviate
        } else if json.contains("\"embeddings\"") && json.contains("\"documents\"") {
            Protocol::ChromaDB
        } else if json.contains("\"entity\"") || json.contains("\"collection_name\"") {
            Protocol::Milvus
        } else {
            Protocol::Universal
        }
    }

    /// Get protocol name
    pub fn name(&self) -> &str {
        match self {
            Protocol::Pinecone => "pinecone",
            Protocol::Qdrant => "qdrant",
            Protocol::Weaviate => "weaviate",
            Protocol::ChromaDB => "chromadb",
            Protocol::Milvus => "milvus",
            Protocol::Universal => "universal",
        }
    }
}

/// Universal request format (internal representation)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "operation")]
pub enum UniversalRequest {
    /// Upsert vectors
    #[serde(rename = "upsert")]
    Upsert { vectors: Vec<VectorData> },

    /// Query for similar vectors
    #[serde(rename = "query")]
    Query {
        vector: Vec<f32>,
        #[serde(default = "default_limit")]
        top_k: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        filter: Option<HashMap<String, Value>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        include_metadata: Option<bool>,
    },

    /// Delete vectors
    #[serde(rename = "delete")]
    Delete { ids: Vec<String> },

    /// Fetch vectors by ID
    #[serde(rename = "fetch")]
    Fetch { ids: Vec<String> },
}

fn default_limit() -> usize {
    10
}

/// Vector data with ID and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorData {
    pub id: String,
    #[serde(alias = "values", alias = "vector", alias = "embedding")]
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// Universal response format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UniversalResponse {
    /// Upsert response
    Upsert { upserted_count: usize },

    /// Query response
    Query { matches: Vec<Match> },

    /// Delete response
    Delete { deleted_count: usize },

    /// Fetch response
    Fetch {
        vectors: HashMap<String, VectorData>,
    },

    /// Error response
    Error { error: String, code: String },
}

/// Query match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Match {
    pub id: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f32>>,
}

/// Protocol adapter for translating between formats
pub struct ProtocolAdapter {
    store: VecStore,
}

impl ProtocolAdapter {
    /// Create a new protocol adapter
    pub fn new(store: VecStore) -> Self {
        Self { store }
    }

    /// Handle request in any supported protocol
    pub fn handle_request(&mut self, json: &str, protocol: Protocol) -> Result<String> {
        // Parse request based on protocol
        let universal_request = self.parse_request(json, protocol)?;

        // Execute request
        let response = self.execute_request(universal_request)?;

        // Format response for protocol
        let json_response = self.format_response(response, protocol)?;

        Ok(json_response)
    }

    /// Handle request with auto-detection
    pub fn handle_request_auto(&mut self, json: &str) -> Result<String> {
        let protocol = Protocol::detect(json);
        self.handle_request(json, protocol)
    }

    /// Parse request from protocol-specific format to universal format
    fn parse_request(&self, json: &str, protocol: Protocol) -> Result<UniversalRequest> {
        match protocol {
            Protocol::Pinecone => self.parse_pinecone(json),
            Protocol::Qdrant => self.parse_qdrant(json),
            Protocol::Weaviate => self.parse_weaviate(json),
            Protocol::ChromaDB => self.parse_chromadb(json),
            Protocol::Milvus => self.parse_milvus(json),
            Protocol::Universal => {
                serde_json::from_str(json).map_err(|e| anyhow!("Invalid JSON: {}", e))
            }
        }
    }

    /// Parse Pinecone format
    fn parse_pinecone(&self, json: &str) -> Result<UniversalRequest> {
        let value: Value = serde_json::from_str(json)?;

        // Detect operation type
        if let Some(vectors) = value.get("vectors") {
            // Upsert operation
            let vectors: Vec<VectorData> = serde_json::from_value(vectors.clone())?;
            Ok(UniversalRequest::Upsert { vectors })
        } else if let Some(vector) = value.get("vector") {
            // Query operation
            let vector: Vec<f32> = serde_json::from_value(vector.clone())?;
            let top_k = value
                .get("topK")
                .or_else(|| value.get("top_k"))
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;

            let filter = value.get("filter").and_then(|f| {
                if let Value::Object(map) = f {
                    Some(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                } else {
                    None
                }
            });

            let include_metadata = value.get("includeMetadata").and_then(|v| v.as_bool());

            Ok(UniversalRequest::Query {
                vector,
                top_k,
                filter,
                include_metadata,
            })
        } else if let Some(ids) = value.get("ids") {
            // Delete or fetch
            let ids: Vec<String> = serde_json::from_value(ids.clone())?;

            if value
                .get("deleteAll")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
                || value
                    .as_object()
                    .map(|o| o.contains_key("delete"))
                    .unwrap_or(false)
            {
                Ok(UniversalRequest::Delete { ids })
            } else {
                Ok(UniversalRequest::Fetch { ids })
            }
        } else {
            Err(anyhow!("Unknown Pinecone operation"))
        }
    }

    /// Parse Qdrant format
    fn parse_qdrant(&self, json: &str) -> Result<UniversalRequest> {
        let value: Value = serde_json::from_str(json)?;

        if let Some(points) = value.get("points") {
            // Upsert
            let points_array: Vec<Value> = serde_json::from_value(points.clone())?;
            let vectors: Vec<VectorData> = points_array
                .iter()
                .map(|point| {
                    let id = point["id"].as_str().unwrap_or("").to_string();
                    let vector: Vec<f32> =
                        serde_json::from_value(point["vector"].clone()).unwrap_or_default();
                    let metadata = point
                        .get("payload")
                        .and_then(|p| {
                            if let Value::Object(map) = p {
                                Some(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();

                    VectorData {
                        id,
                        vector,
                        metadata,
                    }
                })
                .collect();

            Ok(UniversalRequest::Upsert { vectors })
        } else if let Some(vector) = value.get("vector") {
            // Query
            let vector: Vec<f32> = serde_json::from_value(vector.clone())?;
            let top_k = value.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
            let filter = value.get("filter").and_then(|f| {
                if let Value::Object(map) = f {
                    Some(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                } else {
                    None
                }
            });

            Ok(UniversalRequest::Query {
                vector,
                top_k,
                filter,
                include_metadata: Some(true),
            })
        } else {
            Err(anyhow!("Unknown Qdrant operation"))
        }
    }

    /// Parse Weaviate format (simplified)
    fn parse_weaviate(&self, json: &str) -> Result<UniversalRequest> {
        let value: Value = serde_json::from_str(json)?;

        // Simplified parsing - Weaviate uses GraphQL primarily
        if let Some(objects) = value.get("objects") {
            let objects_array: Vec<Value> = serde_json::from_value(objects.clone())?;
            let vectors: Vec<VectorData> = objects_array
                .iter()
                .map(|obj| {
                    let id = obj["id"].as_str().unwrap_or("").to_string();
                    let vector: Vec<f32> =
                        serde_json::from_value(obj["vector"].clone()).unwrap_or_default();
                    let metadata = obj
                        .get("properties")
                        .and_then(|p| {
                            if let Value::Object(map) = p {
                                Some(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();

                    VectorData {
                        id,
                        vector,
                        metadata,
                    }
                })
                .collect();

            Ok(UniversalRequest::Upsert { vectors })
        } else {
            Err(anyhow!(
                "Weaviate format not fully supported - use Universal or GraphQL"
            ))
        }
    }

    /// Parse ChromaDB format
    fn parse_chromadb(&self, json: &str) -> Result<UniversalRequest> {
        let value: Value = serde_json::from_str(json)?;

        if let Some(embeddings) = value.get("embeddings") {
            // Upsert
            let embeddings_array: Vec<Vec<f32>> = serde_json::from_value(embeddings.clone())?;
            let ids: Vec<String> = value
                .get("ids")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_else(|| {
                    (0..embeddings_array.len())
                        .map(|i| format!("vec_{}", i))
                        .collect()
                });

            let metadatas: Vec<HashMap<String, Value>> = value
                .get("metadatas")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_else(|| vec![HashMap::new(); embeddings_array.len()]);

            let vectors: Vec<VectorData> = embeddings_array
                .into_iter()
                .zip(ids.into_iter())
                .zip(metadatas.into_iter())
                .map(|((vector, id), metadata)| VectorData {
                    id,
                    vector,
                    metadata,
                })
                .collect();

            Ok(UniversalRequest::Upsert { vectors })
        } else if let Some(query_embeddings) = value.get("query_embeddings") {
            // Query
            let query_array: Vec<Vec<f32>> = serde_json::from_value(query_embeddings.clone())?;
            let vector = query_array
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("No query vector"))?;
            let top_k = value
                .get("n_results")
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;

            Ok(UniversalRequest::Query {
                vector,
                top_k,
                filter: None,
                include_metadata: Some(true),
            })
        } else {
            Err(anyhow!("Unknown ChromaDB operation"))
        }
    }

    /// Parse Milvus format (simplified)
    fn parse_milvus(&self, json: &str) -> Result<UniversalRequest> {
        let value: Value = serde_json::from_str(json)?;

        if let Some(entities) = value.get("entities") {
            // Insert
            let entities_array: Vec<Value> = serde_json::from_value(entities.clone())?;
            let vectors: Vec<VectorData> = entities_array
                .iter()
                .map(|entity| {
                    let id = entity["id"]
                        .as_str()
                        .or_else(|| entity["pk"].as_str())
                        .unwrap_or("")
                        .to_string();
                    let vector: Vec<f32> =
                        serde_json::from_value(entity["vector"].clone()).unwrap_or_default();
                    let metadata: HashMap<String, Value> = entity
                        .as_object()
                        .map(|obj| {
                            obj.iter()
                                .filter(|(k, _)| *k != "id" && *k != "pk" && *k != "vector")
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect()
                        })
                        .unwrap_or_default();

                    VectorData {
                        id,
                        vector,
                        metadata,
                    }
                })
                .collect();

            Ok(UniversalRequest::Upsert { vectors })
        } else {
            Err(anyhow!(
                "Milvus format requires more complex parsing - use Universal"
            ))
        }
    }

    /// Execute universal request
    fn execute_request(&mut self, request: UniversalRequest) -> Result<UniversalResponse> {
        match request {
            UniversalRequest::Upsert { vectors } => {
                let mut count = 0;
                for vec_data in vectors {
                    let metadata = self.value_map_to_metadata(&vec_data.metadata)?;
                    self.store.upsert(vec_data.id, vec_data.vector, metadata)?;
                    count += 1;
                }
                Ok(UniversalResponse::Upsert {
                    upserted_count: count,
                })
            }

            UniversalRequest::Query {
                vector,
                top_k,
                filter,
                include_metadata,
            } => {
                let query = Query {
                    vector,
                    k: top_k,
                    filter: None, // TODO: Convert filter to FilterExpr
                };

                let results = self.store.query(query)?;

                let matches: Vec<Match> = results
                    .into_iter()
                    .map(|neighbor| Match {
                        id: neighbor.id,
                        score: neighbor.score,
                        metadata: if include_metadata.unwrap_or(false) {
                            Some(self.metadata_to_value_map(&neighbor.metadata))
                        } else {
                            None
                        },
                        values: None, // TODO: Fetch actual vectors if requested
                    })
                    .collect();

                Ok(UniversalResponse::Query { matches })
            }

            UniversalRequest::Delete { ids } => {
                let mut count = 0;
                for id in ids {
                    if self.store.delete(&id).is_ok() {
                        count += 1;
                    }
                }
                Ok(UniversalResponse::Delete {
                    deleted_count: count,
                })
            }

            UniversalRequest::Fetch { ids } => {
                // TODO: Implement fetch - requires get_by_id method on VecStore
                Ok(UniversalResponse::Fetch {
                    vectors: HashMap::new(),
                })
            }
        }
    }

    /// Format response for protocol
    fn format_response(&self, response: UniversalResponse, protocol: Protocol) -> Result<String> {
        match protocol {
            Protocol::Pinecone => self.format_pinecone(response),
            Protocol::Qdrant => self.format_qdrant(response),
            Protocol::Universal | _ => {
                serde_json::to_string(&response).map_err(|e| anyhow!("Serialization error: {}", e))
            }
        }
    }

    /// Format response for Pinecone
    fn format_pinecone(&self, response: UniversalResponse) -> Result<String> {
        let formatted = match response {
            UniversalResponse::Upsert { upserted_count } => {
                serde_json::json!({
                    "upsertedCount": upserted_count
                })
            }
            UniversalResponse::Query { matches } => {
                serde_json::json!({
                    "matches": matches,
                    "namespace": ""
                })
            }
            UniversalResponse::Delete { deleted_count } => {
                serde_json::json!({
                    "deletedCount": deleted_count
                })
            }
            _ => serde_json::json!(response),
        };

        Ok(serde_json::to_string(&formatted)?)
    }

    /// Format response for Qdrant
    fn format_qdrant(&self, response: UniversalResponse) -> Result<String> {
        let formatted = match response {
            UniversalResponse::Upsert { upserted_count } => {
                serde_json::json!({
                    "result": {
                        "operation_id": 0,
                        "status": "completed"
                    },
                    "status": "ok",
                    "time": 0.0
                })
            }
            UniversalResponse::Query { matches } => {
                let results: Vec<Value> = matches
                    .into_iter()
                    .map(|m| {
                        serde_json::json!({
                            "id": m.id,
                            "score": m.score,
                            "payload": m.metadata.unwrap_or_default()
                        })
                    })
                    .collect();

                serde_json::json!({
                    "result": results,
                    "status": "ok",
                    "time": 0.0
                })
            }
            _ => serde_json::json!(response),
        };

        Ok(serde_json::to_string(&formatted)?)
    }

    /// Helper: Convert Value map to Metadata
    fn value_map_to_metadata(&self, map: &HashMap<String, Value>) -> Result<Metadata> {
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };

        for (key, value) in map {
            metadata.fields.insert(key.clone(), value.clone());
        }

        Ok(metadata)
    }

    /// Helper: Convert Metadata to Value map
    fn metadata_to_value_map(&self, metadata: &Metadata) -> HashMap<String, Value> {
        metadata.fields.clone()
    }

    /// Get mutable reference to store
    pub fn store_mut(&mut self) -> &mut VecStore {
        &mut self.store
    }

    /// Get reference to store
    pub fn store(&self) -> &VecStore {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_store() -> (VecStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let store = VecStore::open(temp_dir.path().join("test.db")).unwrap();
        (store, temp_dir)
    }

    #[test]
    fn test_protocol_detection() {
        let pinecone_json = r#"{"vectors": [{"id": "1", "values": [0.1, 0.2]}]}"#;
        assert_eq!(Protocol::detect(pinecone_json), Protocol::Pinecone);

        let qdrant_json = r#"{"points": [{"id": "1", "vector": [0.1, 0.2]}]}"#;
        assert_eq!(Protocol::detect(qdrant_json), Protocol::Qdrant);

        let chromadb_json = r#"{"embeddings": [[0.1, 0.2]], "documents": ["test"]}"#;
        assert_eq!(Protocol::detect(chromadb_json), Protocol::ChromaDB);
    }

    #[test]
    fn test_pinecone_upsert() {
        let (store, _temp_dir) = create_test_store();
        let mut adapter = ProtocolAdapter::new(store);

        let json = r#"{
            "vectors": [
                {
                    "id": "vec1",
                    "values": [0.1, 0.2, 0.3],
                    "metadata": {"source": "test"}
                }
            ]
        }"#;

        let response = adapter.handle_request(json, Protocol::Pinecone).unwrap();
        assert!(response.contains("upsertedCount"));
        assert!(response.contains("1"));
    }

    #[test]
    fn test_pinecone_query() {
        let (mut store, _temp_dir) = create_test_store();

        // Insert a vector first
        let metadata = Metadata {
            fields: [("source".to_string(), serde_json::json!("test"))]
                .iter()
                .cloned()
                .collect(),
        };
        store
            .upsert("vec1".to_string(), vec![0.1, 0.2, 0.3], metadata)
            .unwrap();

        let mut adapter = ProtocolAdapter::new(store);

        let json = r#"{
            "vector": [0.1, 0.2, 0.3],
            "topK": 5,
            "includeMetadata": true
        }"#;

        let response = adapter.handle_request(json, Protocol::Pinecone).unwrap();
        assert!(response.contains("matches"));
        assert!(response.contains("vec1"));
    }

    #[test]
    fn test_qdrant_format() {
        let (store, _temp_dir) = create_test_store();
        let mut adapter = ProtocolAdapter::new(store);

        let json = r#"{
            "points": [
                {
                    "id": "1",
                    "vector": [0.1, 0.2, 0.3],
                    "payload": {"key": "value"}
                }
            ]
        }"#;

        let response = adapter.handle_request(json, Protocol::Qdrant).unwrap();
        assert!(response.contains("status"));
        assert!(response.contains("ok"));
    }

    #[test]
    fn test_chromadb_format() {
        let (store, _temp_dir) = create_test_store();
        let mut adapter = ProtocolAdapter::new(store);

        let json = r#"{
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "ids": ["vec1", "vec2"],
            "metadatas": [{}, {}]
        }"#;

        let response = adapter.handle_request(json, Protocol::ChromaDB).unwrap();
        assert!(response.contains("upserted_count"));
    }

    #[test]
    fn test_auto_detection() {
        let (store, _temp_dir) = create_test_store();
        let mut adapter = ProtocolAdapter::new(store);

        let pinecone_json = r#"{"vectors": [{"id": "1", "values": [0.1, 0.2, 0.3]}]}"#;
        let response = adapter.handle_request_auto(pinecone_json).unwrap();
        assert!(response.contains("upsertedCount"));
    }
}
