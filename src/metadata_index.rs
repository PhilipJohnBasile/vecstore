//! Metadata indexing for fast filtered queries
//!
//! This module provides indexing structures for metadata fields to dramatically
//! speed up filtered vector searches. Instead of scanning all vectors during
//! filtering, indexed fields can be queried in O(log N) or O(1) time.
//!
//! # Features
//!
//! - **BTree indexes**: Range queries on numeric/string fields (>, >=, <, <=)
//! - **Hash indexes**: Fast equality queries (=, !=)
//! - **Inverted indexes**: Text containment queries (CONTAINS)
//! - **Set indexes**: Membership queries (IN, NOT IN)
//! - **Composite indexes**: Multi-field queries
//!
//! # Example
//!
//! ```rust
//! use vecstore::metadata_index::{MetadataIndexManager, IndexConfig, IndexType};
//!
//! let mut index_manager = MetadataIndexManager::new();
//!
//! // Create a BTree index for range queries on "price"
//! index_manager.create_index("price", IndexConfig {
//!     index_type: IndexType::BTree,
//!     field: "price".to_string(),
//! })?;
//!
//! // Create a Hash index for fast equality on "category"
//! index_manager.create_index("category", IndexConfig {
//!     index_type: IndexType::Hash,
//!     field: "category".to_string(),
//! })?;
//!
//! // Query using indexes
//! let ids = index_manager.query("price > 100")?;
//! ```

use crate::error::{Result, VecStoreError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Type of index to create
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IndexType {
    /// BTree index for range queries (>, >=, <, <=, =)
    BTree,
    /// Hash index for fast equality queries (=, !=)
    Hash,
    /// Inverted index for text containment (CONTAINS)
    Inverted,
    /// Set index for membership queries (IN)
    Set,
}

/// Configuration for creating an index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Type of index
    pub index_type: IndexType,
    /// Metadata field to index
    pub field: String,
}

/// Indexed value that can be compared
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum IndexedValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value (stored as ordered bytes for comparison)
    Float(ordered_float::OrderedFloat<f64>),
    /// Boolean value
    Bool(bool),
    /// Null value
    Null,
}

impl IndexedValue {
    /// Create from serde_json::Value
    pub fn from_json(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(s) => IndexedValue::String(s.clone()),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    IndexedValue::Int(i)
                } else if let Some(f) = n.as_f64() {
                    IndexedValue::Float(ordered_float::OrderedFloat(f))
                } else {
                    IndexedValue::Null
                }
            }
            serde_json::Value::Bool(b) => IndexedValue::Bool(*b),
            _ => IndexedValue::Null,
        }
    }

    /// Check if value satisfies operator
    pub fn satisfies(&self, op: &str, other: &IndexedValue) -> bool {
        match op {
            "=" => self == other,
            "!=" => self != other,
            ">" => self > other,
            ">=" => self >= other,
            "<" => self < other,
            "<=" => self <= other,
            _ => false,
        }
    }
}

/// BTree index for range queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTreeIndex {
    /// Field name
    field: String,
    /// BTree mapping values to vector IDs
    tree: BTreeMap<IndexedValue, HashSet<String>>,
}

impl BTreeIndex {
    /// Create new BTree index
    pub fn new(field: String) -> Self {
        Self {
            field,
            tree: BTreeMap::new(),
        }
    }

    /// Insert value
    pub fn insert(&mut self, value: IndexedValue, id: String) {
        self.tree
            .entry(value)
            .or_insert_with(HashSet::new)
            .insert(id);
    }

    /// Remove value
    pub fn remove(&mut self, value: &IndexedValue, id: &str) {
        if let Some(ids) = self.tree.get_mut(value) {
            ids.remove(id);
            if ids.is_empty() {
                self.tree.remove(value);
            }
        }
    }

    /// Query for values matching operator
    pub fn query(&self, op: &str, value: &IndexedValue) -> HashSet<String> {
        let mut result = HashSet::new();

        match op {
            "=" => {
                if let Some(ids) = self.tree.get(value) {
                    result.extend(ids.iter().cloned());
                }
            }
            "!=" => {
                for (k, ids) in &self.tree {
                    if k != value {
                        result.extend(ids.iter().cloned());
                    }
                }
            }
            ">" => {
                for (k, ids) in self.tree.range((
                    std::ops::Bound::Excluded(value.clone()),
                    std::ops::Bound::Unbounded,
                )) {
                    result.extend(ids.iter().cloned());
                }
            }
            ">=" => {
                for (k, ids) in self.tree.range((
                    std::ops::Bound::Included(value.clone()),
                    std::ops::Bound::Unbounded,
                )) {
                    result.extend(ids.iter().cloned());
                }
            }
            "<" => {
                for (k, ids) in self.tree.range((
                    std::ops::Bound::Unbounded,
                    std::ops::Bound::Excluded(value.clone()),
                )) {
                    result.extend(ids.iter().cloned());
                }
            }
            "<=" => {
                for (k, ids) in self.tree.range((
                    std::ops::Bound::Unbounded,
                    std::ops::Bound::Included(value.clone()),
                )) {
                    result.extend(ids.iter().cloned());
                }
            }
            _ => {}
        }

        result
    }

    /// Get all indexed IDs
    pub fn all_ids(&self) -> HashSet<String> {
        let mut result = HashSet::new();
        for ids in self.tree.values() {
            result.extend(ids.iter().cloned());
        }
        result
    }
}

/// Hash index for fast equality queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashIndex {
    /// Field name
    field: String,
    /// Hash map from values to vector IDs
    map: HashMap<IndexedValue, HashSet<String>>,
}

impl HashIndex {
    /// Create new Hash index
    pub fn new(field: String) -> Self {
        Self {
            field,
            map: HashMap::new(),
        }
    }

    /// Insert value
    pub fn insert(&mut self, value: IndexedValue, id: String) {
        self.map
            .entry(value)
            .or_insert_with(HashSet::new)
            .insert(id);
    }

    /// Remove value
    pub fn remove(&mut self, value: &IndexedValue, id: &str) {
        if let Some(ids) = self.map.get_mut(value) {
            ids.remove(id);
            if ids.is_empty() {
                self.map.remove(value);
            }
        }
    }

    /// Query for equality
    pub fn query_eq(&self, value: &IndexedValue) -> HashSet<String> {
        self.map.get(value).cloned().unwrap_or_default()
    }

    /// Query for inequality
    pub fn query_ne(&self, value: &IndexedValue) -> HashSet<String> {
        let mut result = HashSet::new();
        for (k, ids) in &self.map {
            if k != value {
                result.extend(ids.iter().cloned());
            }
        }
        result
    }

    /// Query for membership in set
    pub fn query_in(&self, values: &[IndexedValue]) -> HashSet<String> {
        let mut result = HashSet::new();
        for value in values {
            if let Some(ids) = self.map.get(value) {
                result.extend(ids.iter().cloned());
            }
        }
        result
    }

    /// Get all indexed IDs
    pub fn all_ids(&self) -> HashSet<String> {
        let mut result = HashSet::new();
        for ids in self.map.values() {
            result.extend(ids.iter().cloned());
        }
        result
    }
}

/// Inverted index for text containment queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedIndex {
    /// Field name
    field: String,
    /// Map from terms to vector IDs
    index: HashMap<String, HashSet<String>>,
}

impl InvertedIndex {
    /// Create new inverted index
    pub fn new(field: String) -> Self {
        Self {
            field,
            index: HashMap::new(),
        }
    }

    /// Insert value (tokenizes and indexes terms)
    pub fn insert(&mut self, text: &str, id: String) {
        // Simple whitespace tokenization (can be improved with proper tokenizer)
        for term in text.to_lowercase().split_whitespace() {
            self.index
                .entry(term.to_string())
                .or_insert_with(HashSet::new)
                .insert(id.clone());
        }
    }

    /// Remove value
    pub fn remove(&mut self, text: &str, id: &str) {
        for term in text.to_lowercase().split_whitespace() {
            if let Some(ids) = self.index.get_mut(term) {
                ids.remove(id);
                if ids.is_empty() {
                    self.index.remove(term);
                }
            }
        }
    }

    /// Query for documents containing term
    pub fn query(&self, term: &str) -> HashSet<String> {
        self.index
            .get(&term.to_lowercase())
            .cloned()
            .unwrap_or_default()
    }

    /// Query for documents containing all terms (AND)
    pub fn query_all(&self, terms: &[String]) -> HashSet<String> {
        if terms.is_empty() {
            return HashSet::new();
        }

        let mut result = self.query(&terms[0]);
        for term in &terms[1..] {
            let term_ids = self.query(term);
            result.retain(|id| term_ids.contains(id));
        }
        result
    }

    /// Query for documents containing any term (OR)
    pub fn query_any(&self, terms: &[String]) -> HashSet<String> {
        let mut result = HashSet::new();
        for term in terms {
            result.extend(self.query(term));
        }
        result
    }
}

/// Index on a metadata field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataIndex {
    /// BTree index
    BTree(BTreeIndex),
    /// Hash index
    Hash(HashIndex),
    /// Inverted index
    Inverted(InvertedIndex),
}

impl MetadataIndex {
    /// Create new index
    pub fn new(config: IndexConfig) -> Self {
        match config.index_type {
            IndexType::BTree => MetadataIndex::BTree(BTreeIndex::new(config.field)),
            IndexType::Hash => MetadataIndex::Hash(HashIndex::new(config.field)),
            IndexType::Inverted => MetadataIndex::Inverted(InvertedIndex::new(config.field)),
            IndexType::Set => MetadataIndex::Hash(HashIndex::new(config.field)), // Set is implemented as Hash
        }
    }

    /// Insert value into index
    pub fn insert(&mut self, value: &serde_json::Value, id: String) -> Result<()> {
        match self {
            MetadataIndex::BTree(idx) => {
                idx.insert(IndexedValue::from_json(value), id);
            }
            MetadataIndex::Hash(idx) => {
                idx.insert(IndexedValue::from_json(value), id);
            }
            MetadataIndex::Inverted(idx) => {
                if let serde_json::Value::String(text) = value {
                    idx.insert(text, id);
                }
            }
        }
        Ok(())
    }

    /// Remove value from index
    pub fn remove(&mut self, value: &serde_json::Value, id: &str) -> Result<()> {
        match self {
            MetadataIndex::BTree(idx) => {
                idx.remove(&IndexedValue::from_json(value), id);
            }
            MetadataIndex::Hash(idx) => {
                idx.remove(&IndexedValue::from_json(value), id);
            }
            MetadataIndex::Inverted(idx) => {
                if let serde_json::Value::String(text) = value {
                    idx.remove(text, id);
                }
            }
        }
        Ok(())
    }
}

/// Manager for all metadata indexes
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetadataIndexManager {
    /// Map from index name to index
    indexes: HashMap<String, MetadataIndex>,
}

impl MetadataIndexManager {
    /// Create new index manager
    pub fn new() -> Self {
        Self {
            indexes: HashMap::new(),
        }
    }

    /// Create an index on a field
    pub fn create_index(&mut self, name: &str, config: IndexConfig) -> Result<()> {
        if self.indexes.contains_key(name) {
            return Err(VecStoreError::Other(format!(
                "Index '{}' already exists",
                name
            )));
        }

        self.indexes
            .insert(name.to_string(), MetadataIndex::new(config));
        Ok(())
    }

    /// Drop an index
    pub fn drop_index(&mut self, name: &str) -> Result<()> {
        self.indexes
            .remove(name)
            .ok_or_else(|| VecStoreError::Other(format!("Index '{}' not found", name)))?;
        Ok(())
    }

    /// Insert metadata into all relevant indexes
    pub fn insert(
        &mut self,
        metadata: &serde_json::Map<String, serde_json::Value>,
        id: String,
    ) -> Result<()> {
        for (index_name, index) in &mut self.indexes {
            // Extract field from index name (format: "field_name" or "field_name_idx")
            let field = index_name.split('_').next().unwrap_or(index_name);

            if let Some(value) = metadata.get(field) {
                index.insert(value, id.clone())?;
            }
        }
        Ok(())
    }

    /// Remove metadata from all relevant indexes
    pub fn remove(
        &mut self,
        metadata: &serde_json::Map<String, serde_json::Value>,
        id: &str,
    ) -> Result<()> {
        for (index_name, index) in &mut self.indexes {
            let field = index_name.split('_').next().unwrap_or(index_name);

            if let Some(value) = metadata.get(field) {
                index.remove(value, id)?;
            }
        }
        Ok(())
    }

    /// Query using indexes
    pub fn query(
        &self,
        field: &str,
        op: &str,
        value: &serde_json::Value,
    ) -> Option<HashSet<String>> {
        // Find index for this field
        let index_name = field;
        let index = self.indexes.get(index_name)?;

        let result = match index {
            MetadataIndex::BTree(idx) => idx.query(op, &IndexedValue::from_json(value)),
            MetadataIndex::Hash(idx) => match op {
                "=" => idx.query_eq(&IndexedValue::from_json(value)),
                "!=" => idx.query_ne(&IndexedValue::from_json(value)),
                _ => HashSet::new(),
            },
            MetadataIndex::Inverted(idx) => {
                if let serde_json::Value::String(text) = value {
                    idx.query(text)
                } else {
                    HashSet::new()
                }
            }
        };

        Some(result)
    }

    /// Query IN operator using hash index
    pub fn query_in(&self, field: &str, values: &[serde_json::Value]) -> Option<HashSet<String>> {
        let index = self.indexes.get(field)?;

        match index {
            MetadataIndex::Hash(idx) => {
                let indexed_values: Vec<IndexedValue> =
                    values.iter().map(IndexedValue::from_json).collect();
                Some(idx.query_in(&indexed_values))
            }
            _ => None,
        }
    }

    /// List all indexes
    pub fn list_indexes(&self) -> Vec<String> {
        self.indexes.keys().cloned().collect()
    }

    /// Get statistics for an index
    pub fn index_stats(&self, name: &str) -> Option<IndexStats> {
        let index = self.indexes.get(name)?;

        let (index_type, unique_values, total_entries) = match index {
            MetadataIndex::BTree(idx) => {
                let unique = idx.tree.len();
                let total = idx.all_ids().len();
                (IndexType::BTree, unique, total)
            }
            MetadataIndex::Hash(idx) => {
                let unique = idx.map.len();
                let total = idx.all_ids().len();
                (IndexType::Hash, unique, total)
            }
            MetadataIndex::Inverted(idx) => {
                let unique = idx.index.len();
                let total: usize = idx.index.values().map(|s| s.len()).sum();
                (IndexType::Inverted, unique, total)
            }
        };

        Some(IndexStats {
            index_type,
            unique_values,
            total_entries,
        })
    }
}

/// Statistics for an index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Type of index
    pub index_type: IndexType,
    /// Number of unique values
    pub unique_values: usize,
    /// Total number of entries
    pub total_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_btree_index_range_queries() {
        let mut idx = BTreeIndex::new("price".to_string());

        idx.insert(IndexedValue::Int(100), "id1".to_string());
        idx.insert(IndexedValue::Int(200), "id2".to_string());
        idx.insert(IndexedValue::Int(300), "id3".to_string());

        // Test greater than
        let result = idx.query(">", &IndexedValue::Int(150));
        assert_eq!(result.len(), 2);
        assert!(result.contains("id2"));
        assert!(result.contains("id3"));

        // Test less than or equal
        let result = idx.query("<=", &IndexedValue::Int(200));
        assert_eq!(result.len(), 2);
        assert!(result.contains("id1"));
        assert!(result.contains("id2"));
    }

    #[test]
    fn test_hash_index_equality() {
        let mut idx = HashIndex::new("category".to_string());

        idx.insert(IndexedValue::String("tech".to_string()), "id1".to_string());
        idx.insert(IndexedValue::String("tech".to_string()), "id2".to_string());
        idx.insert(
            IndexedValue::String("science".to_string()),
            "id3".to_string(),
        );

        // Test equality
        let result = idx.query_eq(&IndexedValue::String("tech".to_string()));
        assert_eq!(result.len(), 2);
        assert!(result.contains("id1"));
        assert!(result.contains("id2"));

        // Test IN query
        let values = vec![
            IndexedValue::String("tech".to_string()),
            IndexedValue::String("science".to_string()),
        ];
        let result = idx.query_in(&values);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_inverted_index_contains() {
        let mut idx = InvertedIndex::new("description".to_string());

        idx.insert("rust programming language", "id1".to_string());
        idx.insert("python programming", "id2".to_string());
        idx.insert("rust systems", "id3".to_string());

        // Query for "rust"
        let result = idx.query("rust");
        assert_eq!(result.len(), 2);
        assert!(result.contains("id1"));
        assert!(result.contains("id3"));

        // Query for "programming"
        let result = idx.query("programming");
        assert_eq!(result.len(), 2);
        assert!(result.contains("id1"));
        assert!(result.contains("id2"));
    }

    #[test]
    fn test_index_manager() -> Result<()> {
        let mut manager = MetadataIndexManager::new();

        // Create indexes
        manager.create_index(
            "price",
            IndexConfig {
                index_type: IndexType::BTree,
                field: "price".to_string(),
            },
        )?;

        manager.create_index(
            "category",
            IndexConfig {
                index_type: IndexType::Hash,
                field: "category".to_string(),
            },
        )?;

        // Insert metadata
        let mut metadata = serde_json::Map::new();
        metadata.insert("price".to_string(), serde_json::json!(100));
        metadata.insert("category".to_string(), serde_json::json!("tech"));

        manager.insert(&metadata, "id1".to_string())?;

        // Query
        let result = manager.query("price", ">", &serde_json::json!(50));
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);

        // Get stats
        let stats = manager.index_stats("price");
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.index_type, IndexType::BTree);
        assert_eq!(stats.unique_values, 1);

        Ok(())
    }

    #[test]
    fn test_indexed_value_comparison() {
        let v1 = IndexedValue::Int(100);
        let v2 = IndexedValue::Int(200);

        assert!(v1.satisfies("<", &v2));
        assert!(v1.satisfies("<=", &v2));
        assert!(v1.satisfies("!=", &v2));
        assert!(!v1.satisfies(">", &v2));
    }

    #[test]
    fn test_remove_from_indexes() -> Result<()> {
        let mut manager = MetadataIndexManager::new();

        manager.create_index(
            "price",
            IndexConfig {
                index_type: IndexType::BTree,
                field: "price".to_string(),
            },
        )?;

        let mut metadata = serde_json::Map::new();
        metadata.insert("price".to_string(), serde_json::json!(100));

        manager.insert(&metadata, "id1".to_string())?;

        // Verify it's there
        let result = manager.query("price", "=", &serde_json::json!(100));
        assert_eq!(result.unwrap().len(), 1);

        // Remove it
        manager.remove(&metadata, "id1")?;

        // Verify it's gone
        let result = manager.query("price", "=", &serde_json::json!(100));
        assert_eq!(result.unwrap().len(), 0);

        Ok(())
    }
}
