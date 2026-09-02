//! Payload Indexes (B-tree/Hash) for Fast Filtering
//!
//! Secondary indexes on metadata fields for O(1) or O(log n) lookups.
//! Similar to Qdrant's payload indexing with tenant optimization.
//!
//! # Features
//!
//! - **B-tree Index**: Range queries on numeric/timestamp fields
//! - **Hash Index**: Exact match on string/enum fields
//! - **Full-Text Index**: Keyword search on text fields
//! - **Geo Index**: Spatial queries on coordinates
//! - **Tenant Optimization**: Co-locate tenant data on disk
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::payload_index::{PayloadIndexManager, IndexType};
//!
//! let mut manager = PayloadIndexManager::new();
//!
//! // Create indexes
//! manager.create_index("category", IndexType::Hash)?;
//! manager.create_index("price", IndexType::BTree)?;
//! manager.create_index("location", IndexType::Geo)?;
//!
//! // Insert with auto-indexing
//! manager.insert("doc1", json!({
//!     "category": "electronics",
//!     "price": 299.99,
//!     "location": {"lat": 40.7, "lon": -74.0}
//! }))?;
//!
//! // Fast filtered search
//! let filter = Filter::and(vec![
//!     Filter::eq("category", "electronics"),
//!     Filter::range("price", 100.0, 500.0),
//! ]);
//! let matching_ids = manager.query(&filter)?;
//! ```

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::RwLock;

use crate::error::{Result, VecStoreError};

// ============================================================================
// INDEX TYPES
// ============================================================================

/// Type of payload index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// Hash index for exact match (O(1))
    Hash,
    /// B-tree index for range queries (O(log n))
    BTree,
    /// Full-text index for keyword search
    FullText,
    /// Geo index for spatial queries
    Geo,
    /// Keyword index (tokenized strings)
    Keyword,
}

/// Index statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    pub index_type: String,
    pub field_name: String,
    pub num_entries: usize,
    pub num_unique_values: usize,
    pub memory_bytes: usize,
}

// ============================================================================
// HASH INDEX
// ============================================================================

/// Hash index for exact match lookups
#[derive(Debug, Clone)]
pub struct HashIndex {
    /// Field name
    field: String,
    /// Value -> document IDs
    index: HashMap<String, HashSet<String>>,
    /// Document ID -> value (for updates/deletes)
    reverse: HashMap<String, String>,
}

impl HashIndex {
    pub fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            index: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Insert a document
    pub fn insert(&mut self, doc_id: &str, value: &str) {
        // Remove old value if exists
        if let Some(old_value) = self.reverse.remove(doc_id)
            && let Some(docs) = self.index.get_mut(&old_value)
        {
            docs.remove(doc_id);
            if docs.is_empty() {
                self.index.remove(&old_value);
            }
        }

        // Insert new value
        self.index
            .entry(value.to_string())
            .or_default()
            .insert(doc_id.to_string());
        self.reverse.insert(doc_id.to_string(), value.to_string());
    }

    /// Remove a document
    pub fn remove(&mut self, doc_id: &str) {
        if let Some(value) = self.reverse.remove(doc_id)
            && let Some(docs) = self.index.get_mut(&value)
        {
            docs.remove(doc_id);
            if docs.is_empty() {
                self.index.remove(&value);
            }
        }
    }

    /// Get documents with exact value
    pub fn get(&self, value: &str) -> HashSet<String> {
        self.index.get(value).cloned().unwrap_or_default()
    }

    /// Get documents with any of the values
    pub fn get_any(&self, values: &[String]) -> HashSet<String> {
        let mut result = HashSet::new();
        for value in values {
            if let Some(docs) = self.index.get(value) {
                result.extend(docs.clone());
            }
        }
        result
    }

    /// Get statistics
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            index_type: "hash".to_string(),
            field_name: self.field.clone(),
            num_entries: self.reverse.len(),
            num_unique_values: self.index.len(),
            memory_bytes: 0, // Would need size_of implementation
        }
    }
}

// ============================================================================
// B-TREE INDEX
// ============================================================================

/// B-tree index for range queries
#[derive(Debug, Clone)]
pub struct BTreeIndex {
    /// Field name
    field: String,
    /// Value -> document IDs (sorted by value)
    index: BTreeMap<OrderedFloat<f64>, HashSet<String>>,
    /// Document ID -> value
    reverse: HashMap<String, OrderedFloat<f64>>,
}

impl BTreeIndex {
    pub fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            index: BTreeMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Insert a document
    pub fn insert(&mut self, doc_id: &str, value: f64) {
        let value = OrderedFloat(value);

        // Remove old value if exists
        if let Some(old_value) = self.reverse.remove(doc_id)
            && let Some(docs) = self.index.get_mut(&old_value)
        {
            docs.remove(doc_id);
            if docs.is_empty() {
                self.index.remove(&old_value);
            }
        }

        // Insert new value
        self.index
            .entry(value)
            .or_default()
            .insert(doc_id.to_string());
        self.reverse.insert(doc_id.to_string(), value);
    }

    /// Remove a document
    pub fn remove(&mut self, doc_id: &str) {
        if let Some(value) = self.reverse.remove(doc_id)
            && let Some(docs) = self.index.get_mut(&value)
        {
            docs.remove(doc_id);
            if docs.is_empty() {
                self.index.remove(&value);
            }
        }
    }

    /// Get documents in range [min, max]
    pub fn range(&self, min: f64, max: f64) -> HashSet<String> {
        let min = OrderedFloat(min);
        let max = OrderedFloat(max);

        let mut result = HashSet::new();
        for (_, docs) in self.index.range(min..=max) {
            result.extend(docs.clone());
        }
        result
    }

    /// Get documents greater than value
    pub fn gt(&self, value: f64) -> HashSet<String> {
        let value = OrderedFloat(value);
        let mut result = HashSet::new();
        for (k, docs) in self.index.range(value..) {
            if *k > value {
                result.extend(docs.clone());
            }
        }
        result
    }

    /// Get documents less than value
    pub fn lt(&self, value: f64) -> HashSet<String> {
        let value = OrderedFloat(value);
        let mut result = HashSet::new();
        for (_k, docs) in self.index.range(..value) {
            result.extend(docs.clone());
        }
        result
    }

    /// Get documents with exact value
    pub fn eq(&self, value: f64) -> HashSet<String> {
        self.index
            .get(&OrderedFloat(value))
            .cloned()
            .unwrap_or_default()
    }

    /// Get statistics
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            index_type: "btree".to_string(),
            field_name: self.field.clone(),
            num_entries: self.reverse.len(),
            num_unique_values: self.index.len(),
            memory_bytes: 0,
        }
    }
}

// ============================================================================
// GEO INDEX
// ============================================================================

/// Geographic point
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
}

impl GeoPoint {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Haversine distance in kilometers
    pub fn distance_km(&self, other: &GeoPoint) -> f64 {
        let r = 6371.0; // Earth radius in km

        let lat1 = self.lat.to_radians();
        let lat2 = other.lat.to_radians();
        let dlat = (other.lat - self.lat).to_radians();
        let dlon = (other.lon - self.lon).to_radians();

        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();

        r * c
    }
}

/// Simple geo index using grid cells
#[derive(Debug, Clone)]
pub struct GeoIndex {
    /// Field name
    field: String,
    /// Grid cell -> document IDs
    grid: HashMap<(i32, i32), HashSet<String>>,
    /// Document ID -> point
    points: HashMap<String, GeoPoint>,
    /// Grid resolution in degrees
    resolution: f64,
}

impl GeoIndex {
    pub fn new(field: &str, resolution: f64) -> Self {
        Self {
            field: field.to_string(),
            grid: HashMap::new(),
            points: HashMap::new(),
            resolution,
        }
    }

    fn cell(&self, point: &GeoPoint) -> (i32, i32) {
        (
            (point.lat / self.resolution).floor() as i32,
            (point.lon / self.resolution).floor() as i32,
        )
    }

    /// Insert a document
    pub fn insert(&mut self, doc_id: &str, point: GeoPoint) {
        // Remove old point if exists
        self.remove(doc_id);

        // Insert new point
        let cell = self.cell(&point);
        self.grid
            .entry(cell)
            .or_default()
            .insert(doc_id.to_string());
        self.points.insert(doc_id.to_string(), point);
    }

    /// Remove a document
    pub fn remove(&mut self, doc_id: &str) {
        if let Some(point) = self.points.remove(doc_id) {
            let cell = self.cell(&point);
            if let Some(docs) = self.grid.get_mut(&cell) {
                docs.remove(doc_id);
                if docs.is_empty() {
                    self.grid.remove(&cell);
                }
            }
        }
    }

    /// Get documents within radius (km) of point
    pub fn within_radius(&self, center: &GeoPoint, radius_km: f64) -> Vec<(String, f64)> {
        // Calculate cells to check
        let lat_delta = radius_km / 111.0; // Approximate km per degree
        let lon_delta = radius_km / (111.0 * center.lat.to_radians().cos());

        let min_cell = self.cell(&GeoPoint::new(
            center.lat - lat_delta,
            center.lon - lon_delta,
        ));
        let max_cell = self.cell(&GeoPoint::new(
            center.lat + lat_delta,
            center.lon + lon_delta,
        ));

        let mut results = Vec::new();

        for lat_cell in min_cell.0..=max_cell.0 {
            for lon_cell in min_cell.1..=max_cell.1 {
                if let Some(docs) = self.grid.get(&(lat_cell, lon_cell)) {
                    for doc_id in docs {
                        if let Some(point) = self.points.get(doc_id) {
                            let dist = center.distance_km(point);
                            if dist <= radius_km {
                                results.push((doc_id.clone(), dist));
                            }
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results
    }

    /// Get documents in bounding box
    pub fn within_bbox(
        &self,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> HashSet<String> {
        let min_cell = self.cell(&GeoPoint::new(min_lat, min_lon));
        let max_cell = self.cell(&GeoPoint::new(max_lat, max_lon));

        let mut results = HashSet::new();

        for lat_cell in min_cell.0..=max_cell.0 {
            for lon_cell in min_cell.1..=max_cell.1 {
                if let Some(docs) = self.grid.get(&(lat_cell, lon_cell)) {
                    for doc_id in docs {
                        if let Some(point) = self.points.get(doc_id)
                            && point.lat >= min_lat
                            && point.lat <= max_lat
                            && point.lon >= min_lon
                            && point.lon <= max_lon
                        {
                            results.insert(doc_id.clone());
                        }
                    }
                }
            }
        }

        results
    }

    /// Get statistics
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            index_type: "geo".to_string(),
            field_name: self.field.clone(),
            num_entries: self.points.len(),
            num_unique_values: self.grid.len(),
            memory_bytes: 0,
        }
    }
}

// ============================================================================
// FULL-TEXT INDEX
// ============================================================================

/// Simple full-text index
#[derive(Debug, Clone)]
pub struct FullTextIndex {
    /// Field name
    field: String,
    /// Token -> document IDs
    index: HashMap<String, HashSet<String>>,
    /// Document ID -> tokens
    doc_tokens: HashMap<String, Vec<String>>,
}

impl FullTextIndex {
    pub fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            index: HashMap::new(),
            doc_tokens: HashMap::new(),
        }
    }

    /// Tokenize text
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 1)
            .map(String::from)
            .collect()
    }

    /// Insert a document
    pub fn insert(&mut self, doc_id: &str, text: &str) {
        // Remove old tokens if exists
        self.remove(doc_id);

        // Tokenize and insert
        let tokens = Self::tokenize(text);
        for token in &tokens {
            self.index
                .entry(token.clone())
                .or_default()
                .insert(doc_id.to_string());
        }
        self.doc_tokens.insert(doc_id.to_string(), tokens);
    }

    /// Remove a document
    pub fn remove(&mut self, doc_id: &str) {
        if let Some(tokens) = self.doc_tokens.remove(doc_id) {
            for token in tokens {
                if let Some(docs) = self.index.get_mut(&token) {
                    docs.remove(doc_id);
                    if docs.is_empty() {
                        self.index.remove(&token);
                    }
                }
            }
        }
    }

    /// Search for documents containing all terms (AND)
    pub fn search_all(&self, query: &str) -> HashSet<String> {
        let tokens = Self::tokenize(query);
        if tokens.is_empty() {
            return HashSet::new();
        }

        let mut result: Option<HashSet<String>> = None;
        for token in tokens {
            let docs = self.index.get(&token).cloned().unwrap_or_default();
            result = Some(match result {
                Some(r) => r.intersection(&docs).cloned().collect(),
                None => docs,
            });
        }

        result.unwrap_or_default()
    }

    /// Search for documents containing any term (OR)
    pub fn search_any(&self, query: &str) -> HashSet<String> {
        let tokens = Self::tokenize(query);
        let mut result = HashSet::new();
        for token in tokens {
            if let Some(docs) = self.index.get(&token) {
                result.extend(docs.clone());
            }
        }
        result
    }

    /// Get statistics
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            index_type: "fulltext".to_string(),
            field_name: self.field.clone(),
            num_entries: self.doc_tokens.len(),
            num_unique_values: self.index.len(),
            memory_bytes: 0,
        }
    }
}

// ============================================================================
// PAYLOAD INDEX MANAGER
// ============================================================================

/// Manages all payload indexes
pub struct PayloadIndexManager {
    /// Hash indexes
    hash_indexes: RwLock<HashMap<String, HashIndex>>,
    /// B-tree indexes
    btree_indexes: RwLock<HashMap<String, BTreeIndex>>,
    /// Geo indexes
    geo_indexes: RwLock<HashMap<String, GeoIndex>>,
    /// Full-text indexes
    fulltext_indexes: RwLock<HashMap<String, FullTextIndex>>,
    /// Tenant field (for optimization)
    tenant_field: Option<String>,
}

impl PayloadIndexManager {
    pub fn new() -> Self {
        Self {
            hash_indexes: RwLock::new(HashMap::new()),
            btree_indexes: RwLock::new(HashMap::new()),
            geo_indexes: RwLock::new(HashMap::new()),
            fulltext_indexes: RwLock::new(HashMap::new()),
            tenant_field: None,
        }
    }

    /// Set tenant field for optimization
    pub fn with_tenant_field(mut self, field: &str) -> Self {
        self.tenant_field = Some(field.to_string());
        self
    }

    /// Create an index
    pub fn create_index(&self, field: &str, index_type: IndexType) -> Result<()> {
        match index_type {
            IndexType::Hash | IndexType::Keyword => {
                self.hash_indexes
                    .write()?
                    .insert(field.to_string(), HashIndex::new(field));
            },
            IndexType::BTree => {
                self.btree_indexes
                    .write()?
                    .insert(field.to_string(), BTreeIndex::new(field));
            },
            IndexType::Geo => {
                self.geo_indexes
                    .write()?
                    .insert(field.to_string(), GeoIndex::new(field, 0.1));
            },
            IndexType::FullText => {
                self.fulltext_indexes
                    .write()?
                    .insert(field.to_string(), FullTextIndex::new(field));
            },
        }
        Ok(())
    }

    /// Drop an index
    pub fn drop_index(&self, field: &str) -> bool {
        let mut removed = false;
        if let Ok(mut guard) = self.hash_indexes.write() {
            removed |= guard.remove(field).is_some();
        }
        if let Ok(mut guard) = self.btree_indexes.write() {
            removed |= guard.remove(field).is_some();
        }
        if let Ok(mut guard) = self.geo_indexes.write() {
            removed |= guard.remove(field).is_some();
        }
        if let Ok(mut guard) = self.fulltext_indexes.write() {
            removed |= guard.remove(field).is_some();
        }
        removed
    }

    /// Index a document's payload
    pub fn index_payload(&self, doc_id: &str, payload: &serde_json::Value) -> Result<()> {
        if let serde_json::Value::Object(map) = payload {
            for (field, value) in map {
                // Hash index
                if let Some(idx) = self.hash_indexes.write()?.get_mut(field)
                    && let Some(s) = value.as_str()
                {
                    idx.insert(doc_id, s);
                }

                // B-tree index
                if let Some(idx) = self.btree_indexes.write()?.get_mut(field) {
                    if let Some(n) = value.as_f64() {
                        idx.insert(doc_id, n);
                    } else if let Some(n) = value.as_i64() {
                        idx.insert(doc_id, n as f64);
                    }
                }

                // Geo index
                if let Some(idx) = self.geo_indexes.write()?.get_mut(field)
                    && let serde_json::Value::Object(geo) = value
                    && let (Some(lat), Some(lon)) = (
                        geo.get("lat").and_then(|v| v.as_f64()),
                        geo.get("lon").and_then(|v| v.as_f64()),
                    )
                {
                    idx.insert(doc_id, GeoPoint::new(lat, lon));
                }

                // Full-text index
                if let Some(idx) = self.fulltext_indexes.write()?.get_mut(field)
                    && let Some(s) = value.as_str()
                {
                    idx.insert(doc_id, s);
                }
            }
        }
        Ok(())
    }

    /// Remove document from all indexes
    pub fn remove_document(&self, doc_id: &str) {
        if let Ok(mut guard) = self.hash_indexes.write() {
            for idx in guard.values_mut() {
                idx.remove(doc_id);
            }
        }
        if let Ok(mut guard) = self.btree_indexes.write() {
            for idx in guard.values_mut() {
                idx.remove(doc_id);
            }
        }
        if let Ok(mut guard) = self.geo_indexes.write() {
            for idx in guard.values_mut() {
                idx.remove(doc_id);
            }
        }
        if let Ok(mut guard) = self.fulltext_indexes.write() {
            for idx in guard.values_mut() {
                idx.remove(doc_id);
            }
        }
    }

    /// Query with filter
    pub fn query(&self, filter: &Filter) -> Result<HashSet<String>> {
        self.evaluate_filter(filter)
    }

    fn evaluate_filter(&self, filter: &Filter) -> Result<HashSet<String>> {
        match filter {
            Filter::Eq { field, value } => {
                let guard = self.hash_indexes.read()?;
                if let Some(idx) = guard.get(field) {
                    Ok(idx.get(value))
                } else {
                    Err(VecStoreError::InvalidInput(format!(
                        "No hash index on field: {}",
                        field
                    )))
                }
            },
            Filter::In { field, values } => {
                let guard = self.hash_indexes.read()?;
                if let Some(idx) = guard.get(field) {
                    Ok(idx.get_any(values))
                } else {
                    Err(VecStoreError::InvalidInput(format!(
                        "No hash index on field: {}",
                        field
                    )))
                }
            },
            Filter::Range { field, min, max } => {
                let guard = self.btree_indexes.read()?;
                if let Some(idx) = guard.get(field) {
                    Ok(idx.range(*min, *max))
                } else {
                    Err(VecStoreError::InvalidInput(format!(
                        "No btree index on field: {}",
                        field
                    )))
                }
            },
            Filter::Gt { field, value } => {
                let guard = self.btree_indexes.read()?;
                if let Some(idx) = guard.get(field) {
                    Ok(idx.gt(*value))
                } else {
                    Err(VecStoreError::InvalidInput(format!(
                        "No btree index on field: {}",
                        field
                    )))
                }
            },
            Filter::Lt { field, value } => {
                let guard = self.btree_indexes.read()?;
                if let Some(idx) = guard.get(field) {
                    Ok(idx.lt(*value))
                } else {
                    Err(VecStoreError::InvalidInput(format!(
                        "No btree index on field: {}",
                        field
                    )))
                }
            },
            Filter::GeoRadius {
                field,
                center,
                radius_km,
            } => {
                let guard = self.geo_indexes.read()?;
                if let Some(idx) = guard.get(field) {
                    let results = idx.within_radius(center, *radius_km);
                    Ok(results.into_iter().map(|(id, _)| id).collect())
                } else {
                    Err(VecStoreError::InvalidInput(format!(
                        "No geo index on field: {}",
                        field
                    )))
                }
            },
            Filter::FullText { field, query } => {
                let guard = self.fulltext_indexes.read()?;
                if let Some(idx) = guard.get(field) {
                    Ok(idx.search_all(query))
                } else {
                    Err(VecStoreError::InvalidInput(format!(
                        "No fulltext index on field: {}",
                        field
                    )))
                }
            },
            Filter::And(filters) => {
                let mut result: Option<HashSet<String>> = None;
                for f in filters {
                    let docs = self.evaluate_filter(f)?;
                    result = Some(match result {
                        Some(r) => r.intersection(&docs).cloned().collect(),
                        None => docs,
                    });
                }
                Ok(result.unwrap_or_default())
            },
            Filter::Or(filters) => {
                let mut result = HashSet::new();
                for f in filters {
                    result.extend(self.evaluate_filter(f)?);
                }
                Ok(result)
            },
            Filter::Not(_inner) => {
                // Would need all document IDs to compute NOT
                // For now, return empty (would need enhancement)
                Err(VecStoreError::InvalidInput(
                    "NOT filter requires full document list".to_string(),
                ))
            },
        }
    }

    /// Get all index statistics
    pub fn stats(&self) -> Vec<IndexStats> {
        let mut stats = Vec::new();

        if let Ok(guard) = self.hash_indexes.read() {
            for idx in guard.values() {
                stats.push(idx.stats());
            }
        }
        if let Ok(guard) = self.btree_indexes.read() {
            for idx in guard.values() {
                stats.push(idx.stats());
            }
        }
        if let Ok(guard) = self.geo_indexes.read() {
            for idx in guard.values() {
                stats.push(idx.stats());
            }
        }
        if let Ok(guard) = self.fulltext_indexes.read() {
            for idx in guard.values() {
                stats.push(idx.stats());
            }
        }

        stats
    }
}

impl Default for PayloadIndexManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// FILTER DSL
// ============================================================================

/// Filter expression for querying indexes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
    /// Exact match
    Eq { field: String, value: String },
    /// Match any of values
    In { field: String, values: Vec<String> },
    /// Range query [min, max]
    Range { field: String, min: f64, max: f64 },
    /// Greater than
    Gt { field: String, value: f64 },
    /// Less than
    Lt { field: String, value: f64 },
    /// Geo radius query
    GeoRadius {
        field: String,
        center: GeoPoint,
        radius_km: f64,
    },
    /// Full-text search
    FullText { field: String, query: String },
    /// AND of filters
    And(Vec<Filter>),
    /// OR of filters
    Or(Vec<Filter>),
    /// NOT filter
    Not(Box<Filter>),
}

impl Filter {
    pub fn eq(field: &str, value: &str) -> Self {
        Filter::Eq {
            field: field.to_string(),
            value: value.to_string(),
        }
    }

    pub fn in_values(field: &str, values: Vec<String>) -> Self {
        Filter::In {
            field: field.to_string(),
            values,
        }
    }

    pub fn range(field: &str, min: f64, max: f64) -> Self {
        Filter::Range {
            field: field.to_string(),
            min,
            max,
        }
    }

    pub fn gt(field: &str, value: f64) -> Self {
        Filter::Gt {
            field: field.to_string(),
            value,
        }
    }

    pub fn lt(field: &str, value: f64) -> Self {
        Filter::Lt {
            field: field.to_string(),
            value,
        }
    }

    pub fn geo_radius(field: &str, lat: f64, lon: f64, radius_km: f64) -> Self {
        Filter::GeoRadius {
            field: field.to_string(),
            center: GeoPoint::new(lat, lon),
            radius_km,
        }
    }

    pub fn fulltext(field: &str, query: &str) -> Self {
        Filter::FullText {
            field: field.to_string(),
            query: query.to_string(),
        }
    }

    pub fn and(filters: Vec<Filter>) -> Self {
        Filter::And(filters)
    }

    pub fn or(filters: Vec<Filter>) -> Self {
        Filter::Or(filters)
    }

    /// Create a NOT filter (negation)
    #[allow(clippy::should_implement_trait)]
    pub fn not(filter: Filter) -> Self {
        Filter::Not(Box::new(filter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_index() {
        let mut idx = HashIndex::new("category");

        idx.insert("doc1", "electronics");
        idx.insert("doc2", "electronics");
        idx.insert("doc3", "books");

        assert_eq!(idx.get("electronics").len(), 2);
        assert_eq!(idx.get("books").len(), 1);
        assert_eq!(idx.get("other").len(), 0);

        idx.remove("doc1");
        assert_eq!(idx.get("electronics").len(), 1);
    }

    #[test]
    fn test_btree_index() {
        let mut idx = BTreeIndex::new("price");

        idx.insert("doc1", 100.0);
        idx.insert("doc2", 200.0);
        idx.insert("doc3", 300.0);

        assert_eq!(idx.range(150.0, 250.0).len(), 1);
        assert_eq!(idx.gt(150.0).len(), 2);
        assert_eq!(idx.lt(150.0).len(), 1);
    }

    #[test]
    fn test_geo_index() {
        let mut idx = GeoIndex::new("location", 0.1);

        idx.insert("nyc", GeoPoint::new(40.7128, -74.0060));
        idx.insert("la", GeoPoint::new(34.0522, -118.2437));

        let results = idx.within_radius(&GeoPoint::new(40.7, -74.0), 50.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "nyc");
    }

    #[test]
    fn test_fulltext_index() {
        let mut idx = FullTextIndex::new("content");

        idx.insert("doc1", "The quick brown fox");
        idx.insert("doc2", "The lazy dog");
        idx.insert("doc3", "Quick brown dog");

        assert_eq!(idx.search_all("quick").len(), 2);
        assert_eq!(idx.search_all("quick brown").len(), 2);
        assert_eq!(idx.search_all("fox dog").len(), 0); // AND - no doc has both
        assert_eq!(idx.search_any("fox dog").len(), 3); // OR - all docs have one
    }

    #[test]
    fn test_manager() {
        let manager = PayloadIndexManager::new();

        manager.create_index("category", IndexType::Hash).unwrap();
        manager.create_index("price", IndexType::BTree).unwrap();

        manager
            .index_payload(
                "doc1",
                &serde_json::json!({
                    "category": "electronics",
                    "price": 299.99
                }),
            )
            .unwrap();

        let filter = Filter::and(vec![
            Filter::eq("category", "electronics"),
            Filter::range("price", 200.0, 400.0),
        ]);

        let results = manager.query(&filter).unwrap();
        assert!(results.contains("doc1"));
    }
}
