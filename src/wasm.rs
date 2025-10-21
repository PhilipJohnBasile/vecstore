// WebAssembly bindings for vecstore
//
// This module provides browser-compatible APIs for running vecstore in WASM.
// Note: This uses an in-memory store since file I/O works differently in browsers.

use crate::store::{
    hybrid::TextIndex, parse_filter, FilterExpr, Metadata, Query, Record, VectorBackend,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// In-memory vector store for WASM
///
/// This is optimized for browser usage and doesn't use file I/O.
/// All data is kept in memory.
#[wasm_bindgen]
pub struct WasmVecStore {
    backend: VectorBackend,
    records: HashMap<String, Record>,
    text_index: TextIndex,
    dimension: usize,
}

/// Search result for WASM
#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct WasmSearchResult {
    id: String,
    score: f32,
    metadata_json: String,
}

#[wasm_bindgen]
impl WasmSearchResult {
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f32 {
        self.score
    }

    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> String {
        self.metadata_json.clone()
    }
}

#[wasm_bindgen]
impl WasmVecStore {
    /// Create a new in-memory vector store
    ///
    /// # Arguments
    /// * `dimension` - Expected vector dimension (0 for auto-detect from first insert)
    ///
    /// # Example (JavaScript)
    /// ```javascript
    /// const store = new WasmVecStore(384); // For 384-dimensional embeddings
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        // Set panic hook for better error messages in browser
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        Self {
            backend: VectorBackend::new(dimension),
            records: HashMap::new(),
            text_index: TextIndex::new(),
            dimension,
        }
    }

    /// Insert or update a vector with metadata
    ///
    /// # Arguments
    /// * `id` - Unique identifier
    /// * `vector` - Float32Array or regular array of numbers
    /// * `metadata` - JavaScript object with metadata
    ///
    /// # Example (JavaScript)
    /// ```javascript
    /// store.upsert(
    ///     "doc1",
    ///     new Float32Array([0.1, 0.2, 0.3, ...]),
    ///     { title: "Document 1", category: "tech" }
    /// );
    /// ```
    #[wasm_bindgen]
    pub fn upsert(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: JsValue,
    ) -> Result<(), JsValue> {
        // Set dimension on first insert
        if self.dimension == 0 {
            self.dimension = vector.len();
            self.backend = VectorBackend::new(self.dimension);
        }

        if vector.len() != self.dimension {
            return Err(JsValue::from_str(&format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }

        // Convert JS metadata to Rust Metadata
        let metadata = js_value_to_metadata(metadata)?;

        let record = Record {
            id: id.clone(),
            vector: vector.clone(),
            metadata,
            created_at: js_sys::Date::now() as i64,
            deleted: false,
            deleted_at: None,
            expires_at: None,
        };

        self.backend
            .insert(id.clone(), &vector)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.records.insert(id, record);

        Ok(())
    }

    /// Search for similar vectors
    ///
    /// # Arguments
    /// * `vector` - Query vector (Float32Array or regular array)
    /// * `k` - Number of results to return
    /// * `filter` - Optional SQL-like filter string (e.g., "category = 'tech'")
    ///
    /// # Returns
    /// Array of search results with id, score, and metadata
    ///
    /// # Example (JavaScript)
    /// ```javascript
    /// const results = store.query(
    ///     new Float32Array([0.1, 0.2, 0.3, ...]),
    ///     10,
    ///     "category = 'tech'"
    /// );
    ///
    /// results.forEach(result => {
    ///     console.log(`${result.id}: ${result.score}`);
    ///     console.log(JSON.parse(result.metadata));
    /// });
    /// ```
    #[wasm_bindgen]
    pub fn query(
        &self,
        vector: Vec<f32>,
        k: usize,
        filter: Option<String>,
    ) -> Result<JsValue, JsValue> {
        if self.dimension == 0 {
            return Ok(serde_wasm_bindgen::to_value(&Vec::<WasmSearchResult>::new()).unwrap());
        }

        if vector.len() != self.dimension {
            return Err(JsValue::from_str(&format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }

        let filter_expr: Option<FilterExpr> = if let Some(filter_str) = filter {
            Some(
                parse_filter(&filter_str)
                    .map_err(|e| JsValue::from_str(&format!("Filter parse error: {}", e)))?,
            )
        } else {
            None
        };

        let query = Query {
            vector,
            k,
            filter: filter_expr.clone(),
        };

        // Get results from backend
        let fetch_size = if filter_expr.is_some() { k * 10 } else { k };

        let results = self
            .backend
            .search(&query.vector, fetch_size)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Apply filters and convert to WASM results
        let mut neighbors: Vec<WasmSearchResult> = Vec::new();

        for (id, score) in results {
            if let Some(record) = self.records.get(&id) {
                // Apply filter if present
                if let Some(ref filter) = filter_expr {
                    if !crate::store::filters::evaluate_filter(filter, &record.metadata) {
                        continue;
                    }
                }

                let metadata_json = serde_json::to_string(&record.metadata.fields)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;

                neighbors.push(WasmSearchResult {
                    id: id.clone(),
                    score,
                    metadata_json,
                });

                if neighbors.len() >= k {
                    break;
                }
            }
        }

        serde_wasm_bindgen::to_value(&neighbors).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Index text for keyword search (required for hybrid search)
    ///
    /// # Example (JavaScript)
    /// ```javascript
    /// store.index_text("doc1", "Rust is a systems programming language");
    /// ```
    #[wasm_bindgen]
    pub fn index_text(&mut self, id: String, text: String) -> Result<(), JsValue> {
        self.text_index.index_document(id, text);
        Ok(())
    }

    /// Hybrid search combining vector similarity and keyword matching
    ///
    /// # Arguments
    /// * `vector` - Query vector
    /// * `keywords` - Search keywords
    /// * `k` - Number of results
    /// * `alpha` - Balance between vector (1.0) and keyword (0.0) search
    /// * `filter` - Optional filter string
    ///
    /// # Example (JavaScript)
    /// ```javascript
    /// const results = store.hybrid_query(
    ///     queryVector,
    ///     "machine learning",
    ///     10,
    ///     0.7,  // 70% vector, 30% keyword
    ///     null
    /// );
    /// ```
    #[wasm_bindgen]
    pub fn hybrid_query(
        &self,
        vector: Vec<f32>,
        keywords: String,
        k: usize,
        alpha: f32,
        filter: Option<String>,
    ) -> Result<JsValue, JsValue> {
        if self.dimension == 0 {
            return Ok(serde_wasm_bindgen::to_value(&Vec::<WasmSearchResult>::new()).unwrap());
        }

        let filter_expr: Option<FilterExpr> = if let Some(filter_str) = filter {
            Some(
                parse_filter(&filter_str)
                    .map_err(|e| JsValue::from_str(&format!("Filter parse error: {}", e)))?,
            )
        } else {
            None
        };

        let fetch_size = if filter_expr.is_some() { k * 10 } else { k * 2 };

        // Get vector results
        let vector_results = self
            .backend
            .search(&vector, fetch_size)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Get BM25 scores
        let bm25_scores = if !keywords.is_empty() {
            self.text_index.bm25_scores(&keywords)
        } else {
            HashMap::new()
        };

        // Combine scores
        let combined = crate::store::hybrid::combine_scores(vector_results, bm25_scores, alpha);

        // Apply filters and convert to WASM results
        let mut neighbors: Vec<WasmSearchResult> = Vec::new();

        for (id, score) in combined {
            if let Some(record) = self.records.get(&id) {
                // Apply filter if present
                if let Some(ref filter) = filter_expr {
                    if !crate::store::filters::evaluate_filter(filter, &record.metadata) {
                        continue;
                    }
                }

                let metadata_json = serde_json::to_string(&record.metadata.fields)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;

                neighbors.push(WasmSearchResult {
                    id: id.clone(),
                    score,
                    metadata_json,
                });

                if neighbors.len() >= k {
                    break;
                }
            }
        }

        serde_wasm_bindgen::to_value(&neighbors).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Remove a vector by ID
    #[wasm_bindgen]
    pub fn remove(&mut self, id: String) -> Result<(), JsValue> {
        self.backend
            .remove(&id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.records.remove(&id);
        self.text_index.remove_document(&id);
        Ok(())
    }

    /// Get the number of vectors in the store
    #[wasm_bindgen]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if the store is empty
    #[wasm_bindgen]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Export store data as JSON (for persistence/backup)
    ///
    /// Returns a JSON string containing all vectors and metadata.
    /// Can be saved to localStorage or IndexedDB.
    ///
    /// # Example (JavaScript)
    /// ```javascript
    /// const data = store.export_json();
    /// localStorage.setItem('vecstore_backup', data);
    /// ```
    #[wasm_bindgen]
    pub fn export_json(&self) -> Result<String, JsValue> {
        #[derive(Serialize)]
        struct ExportData {
            dimension: usize,
            records: Vec<Record>,
        }

        let data = ExportData {
            dimension: self.dimension,
            records: self.records.values().cloned().collect(),
        };

        serde_json::to_string(&data).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Import store data from JSON
    ///
    /// Restores a store from previously exported JSON data.
    ///
    /// # Example (JavaScript)
    /// ```javascript
    /// const store = new WasmVecStore(0);
    /// const data = localStorage.getItem('vecstore_backup');
    /// store.import_json(data);
    /// ```
    #[wasm_bindgen]
    pub fn import_json(&mut self, json_data: String) -> Result<(), JsValue> {
        #[derive(Deserialize)]
        struct ImportData {
            dimension: usize,
            records: Vec<Record>,
        }

        let data: ImportData =
            serde_json::from_str(&json_data).map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.dimension = data.dimension;
        self.backend = VectorBackend::new(self.dimension);
        self.records.clear();
        self.text_index = TextIndex::new();

        for record in data.records {
            self.backend
                .insert(record.id.clone(), &record.vector)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            self.records.insert(record.id.clone(), record);
        }

        Ok(())
    }

    /// Log information to browser console
    #[wasm_bindgen]
    pub fn log_info(&self) {
        web_sys::console::log_1(
            &format!(
                "VecStore: {} vectors, dimension: {}",
                self.len(),
                self.dimension
            )
            .into(),
        );
    }
}

/// Helper function to convert JsValue to Rust Metadata
fn js_value_to_metadata(js_value: JsValue) -> Result<Metadata, JsValue> {
    let obj = js_sys::Object::from(js_value);
    let entries = js_sys::Object::entries(&obj);
    let mut fields = HashMap::new();

    for i in 0..entries.length() {
        let entry = entries.get(i);
        let entry_array: js_sys::Array = entry.into();

        let key: String = entry_array
            .get(0)
            .as_string()
            .ok_or_else(|| JsValue::from_str("Metadata key must be a string"))?;

        let value = entry_array.get(1);

        let json_value = if value.is_string() {
            serde_json::Value::String(
                value
                    .as_string()
                    .ok_or_else(|| JsValue::from_str("Failed to convert string"))?,
            )
        } else if let Some(num) = value.as_f64() {
            serde_json::json!(num)
        } else if let Some(b) = value.as_bool() {
            serde_json::Value::Bool(b)
        } else if value.is_null() || value.is_undefined() {
            serde_json::Value::Null
        } else {
            // Try to serialize as JSON
            let json_str = js_sys::JSON::stringify(&value)
                .map_err(|_| JsValue::from_str("Failed to stringify value"))?
                .as_string()
                .ok_or_else(|| JsValue::from_str("Failed to convert to string"))?;
            serde_json::from_str(&json_str)
                .map_err(|e| JsValue::from_str(&format!("Failed to parse JSON: {}", e)))?
        };

        fields.insert(key, json_value);
    }

    Ok(Metadata { fields })
}

/// Initialize WASM module (called automatically when module loads)
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}
