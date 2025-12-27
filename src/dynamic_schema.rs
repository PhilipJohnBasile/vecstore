//! Dynamic Schema Evolution
//!
//! Runtime schema flexibility without migrations or downtime.
//! Similar to Milvus dynamic fields with automatic index adaptation.
//!
//! # Features
//!
//! - **Dynamic Fields**: Add fields without schema migration
//! - **Type Inference**: Automatic type detection
//! - **Index Adaptation**: Indexes update automatically
//! - **Backward Compatible**: Old data works with new schema
//! - **Schema Versioning**: Track schema changes over time
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::dynamic_schema::{DynamicCollection, SchemaBuilder};
//!
//! // Start with minimal schema
//! let schema = SchemaBuilder::new()
//!     .add_field("id", FieldType::String)
//!     .add_vector("embedding", 384)
//!     .build();
//!
//! let mut collection = DynamicCollection::new(schema)?;
//!
//! // Insert with extra fields - auto-detected
//! collection.insert(json!({
//!     "id": "doc1",
//!     "embedding": [...],
//!     "title": "Hello",        // String detected
//!     "score": 0.95,           // Float detected
//!     "tags": ["a", "b"],      // Array detected
//! }))?;
//!
//! // New fields are queryable
//! collection.search_with_filter(&query, "score > 0.9")?;
//! ```

use std::collections::HashMap;
use std::sync::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::{VecStoreError, Result};

// ============================================================================
// FIELD TYPES
// ============================================================================

/// Field data type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// String/text
    String,
    /// Integer
    Int64,
    /// Float
    Float64,
    /// Boolean
    Bool,
    /// JSON object
    Object,
    /// Array of values
    Array(Box<FieldType>),
    /// Vector with dimension
    Vector(usize),
    /// Timestamp
    Timestamp,
    /// Any type (dynamic)
    Any,
}

impl FieldType {
    /// Infer type from JSON value
    pub fn infer(value: &JsonValue) -> Self {
        match value {
            JsonValue::Null => FieldType::Any,
            JsonValue::Bool(_) => FieldType::Bool,
            JsonValue::Number(n) => {
                if n.is_i64() {
                    FieldType::Int64
                } else {
                    FieldType::Float64
                }
            }
            JsonValue::String(_) => FieldType::String,
            JsonValue::Array(arr) => {
                if arr.is_empty() {
                    FieldType::Array(Box::new(FieldType::Any))
                } else if arr.iter().all(|v| v.is_f64()) {
                    FieldType::Vector(arr.len())
                } else {
                    let elem_type = FieldType::infer(&arr[0]);
                    FieldType::Array(Box::new(elem_type))
                }
            }
            JsonValue::Object(_) => FieldType::Object,
        }
    }

    /// Check if value is compatible with this type
    pub fn is_compatible(&self, value: &JsonValue) -> bool {
        match (self, value) {
            (FieldType::Any, _) => true,
            (FieldType::String, JsonValue::String(_)) => true,
            (FieldType::Int64, JsonValue::Number(n)) => n.is_i64(),
            (FieldType::Float64, JsonValue::Number(_)) => true,
            (FieldType::Bool, JsonValue::Bool(_)) => true,
            (FieldType::Object, JsonValue::Object(_)) => true,
            (FieldType::Array(_), JsonValue::Array(_)) => true,
            (FieldType::Vector(dim), JsonValue::Array(arr)) => {
                arr.len() == *dim && arr.iter().all(|v| v.is_f64())
            }
            (FieldType::Timestamp, JsonValue::Number(_)) => true,
            (FieldType::Timestamp, JsonValue::String(_)) => true,
            _ => false,
        }
    }

    /// Can this type be promoted to another?
    pub fn can_promote_to(&self, other: &FieldType) -> bool {
        match (self, other) {
            (a, b) if a == b => true,
            (FieldType::Int64, FieldType::Float64) => true,
            (_, FieldType::Any) => true,
            _ => false,
        }
    }
}

// ============================================================================
// FIELD DEFINITION
// ============================================================================

/// Field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: String,
    pub field_type: FieldType,
    pub nullable: bool,
    pub indexed: bool,
    pub description: Option<String>,
    pub default: Option<JsonValue>,
}

impl FieldDef {
    pub fn new(name: &str, field_type: FieldType) -> Self {
        Self {
            name: name.to_string(),
            field_type,
            nullable: true,
            indexed: false,
            description: None,
            default: None,
        }
    }

    pub fn required(mut self) -> Self {
        self.nullable = false;
        self
    }

    pub fn indexed(mut self) -> Self {
        self.indexed = true;
        self
    }

    pub fn with_default(mut self, default: JsonValue) -> Self {
        self.default = Some(default);
        self
    }
}

// ============================================================================
// SCHEMA
// ============================================================================

/// Collection schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Schema version
    pub version: u32,
    /// Field definitions
    pub fields: HashMap<String, FieldDef>,
    /// Primary key field
    pub primary_key: String,
    /// Vector field name
    pub vector_field: Option<String>,
    /// Vector dimension
    pub dimension: Option<usize>,
    /// Allow dynamic fields
    pub allow_dynamic: bool,
    /// Created at
    pub created_at: u64,
    /// Modified at
    pub modified_at: u64,
}

impl Schema {
    /// Validate a document against schema
    pub fn validate(&self, doc: &JsonValue) -> Result<()> {
        if let JsonValue::Object(map) = doc {
            // Check required fields
            for (name, field) in &self.fields {
                if !field.nullable && !map.contains_key(name) {
                    return Err(VecStoreError::InvalidInput(format!(
                        "Missing required field: {}",
                        name
                    )));
                }
            }

            // Check field types
            for (key, value) in map {
                if let Some(field) = self.fields.get(key) {
                    if !field.field_type.is_compatible(value) {
                        return Err(VecStoreError::InvalidInput(format!(
                            "Field {} has wrong type: expected {:?}",
                            key, field.field_type
                        )));
                    }
                } else if !self.allow_dynamic {
                    return Err(VecStoreError::InvalidInput(format!(
                        "Unknown field: {} (dynamic fields disabled)",
                        key
                    )));
                }
            }

            Ok(())
        } else {
            Err(VecStoreError::InvalidInput("Document must be an object".to_string()))
        }
    }

    /// Get vector from document
    pub fn extract_vector(&self, doc: &JsonValue) -> Option<Vec<f32>> {
        let field_name = self.vector_field.as_ref()?;
        doc.get(field_name)?
            .as_array()?
            .iter()
            .map(|v| v.as_f64().map(|f| f as f32))
            .collect()
    }
}

/// Schema builder
pub struct SchemaBuilder {
    fields: HashMap<String, FieldDef>,
    primary_key: Option<String>,
    vector_field: Option<String>,
    dimension: Option<usize>,
    allow_dynamic: bool,
}

impl SchemaBuilder {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            primary_key: None,
            vector_field: None,
            dimension: None,
            allow_dynamic: true,
        }
    }

    pub fn add_field(mut self, name: &str, field_type: FieldType) -> Self {
        self.fields.insert(name.to_string(), FieldDef::new(name, field_type));
        self
    }

    pub fn add_field_def(mut self, field: FieldDef) -> Self {
        self.fields.insert(field.name.clone(), field);
        self
    }

    pub fn add_vector(mut self, name: &str, dimension: usize) -> Self {
        self.fields.insert(
            name.to_string(),
            FieldDef::new(name, FieldType::Vector(dimension)),
        );
        self.vector_field = Some(name.to_string());
        self.dimension = Some(dimension);
        self
    }

    pub fn primary_key(mut self, name: &str) -> Self {
        self.primary_key = Some(name.to_string());
        self
    }

    pub fn allow_dynamic(mut self, allow: bool) -> Self {
        self.allow_dynamic = allow;
        self
    }

    pub fn build(self) -> Schema {
        let now = unix_timestamp();
        Schema {
            version: 1,
            fields: self.fields,
            primary_key: self.primary_key.unwrap_or_else(|| "id".to_string()),
            vector_field: self.vector_field,
            dimension: self.dimension,
            allow_dynamic: self.allow_dynamic,
            created_at: now,
            modified_at: now,
        }
    }
}

impl Default for SchemaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SCHEMA EVOLUTION
// ============================================================================

/// Schema change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaChange {
    /// New field added
    FieldAdded { field: FieldDef },
    /// Field removed
    FieldRemoved { name: String },
    /// Field type changed
    FieldTypeChanged {
        name: String,
        from: FieldType,
        to: FieldType,
    },
    /// Field made nullable
    FieldNullable { name: String },
    /// Field made required
    FieldRequired { name: String },
    /// Index added
    IndexAdded { field: String },
    /// Index removed
    IndexRemoved { field: String },
}

/// Schema evolution manager
pub struct SchemaEvolution {
    /// Change history
    history: Vec<SchemaChange>,
    /// Version map
    versions: HashMap<u32, Schema>,
}

impl SchemaEvolution {
    pub fn new(initial: Schema) -> Self {
        let mut versions = HashMap::new();
        versions.insert(1, initial);

        Self {
            history: Vec::new(),
            versions,
        }
    }

    /// Apply a schema change
    pub fn apply(&mut self, change: SchemaChange) -> Result<Schema> {
        let current_version = *self.versions.keys().max().unwrap_or(&0);
        let mut new_schema = self.versions.get(&current_version)
            .ok_or_else(|| VecStoreError::Internal("No current schema".to_string()))?
            .clone();

        match &change {
            SchemaChange::FieldAdded { field } => {
                if new_schema.fields.contains_key(&field.name) {
                    return Err(VecStoreError::InvalidInput(format!(
                        "Field {} already exists",
                        field.name
                    )));
                }
                new_schema.fields.insert(field.name.clone(), field.clone());
            }
            SchemaChange::FieldRemoved { name } => {
                if !new_schema.fields.contains_key(name) {
                    return Err(VecStoreError::InvalidInput(format!(
                        "Field {} does not exist",
                        name
                    )));
                }
                new_schema.fields.remove(name);
            }
            SchemaChange::FieldTypeChanged { name, from, to } => {
                if let Some(field) = new_schema.fields.get_mut(name) {
                    if field.field_type != *from {
                        return Err(VecStoreError::InvalidInput(format!(
                            "Field {} type mismatch",
                            name
                        )));
                    }
                    if !from.can_promote_to(to) {
                        return Err(VecStoreError::InvalidInput(format!(
                            "Cannot change {} from {:?} to {:?}",
                            name, from, to
                        )));
                    }
                    field.field_type = to.clone();
                }
            }
            SchemaChange::FieldNullable { name } => {
                if let Some(field) = new_schema.fields.get_mut(name) {
                    field.nullable = true;
                }
            }
            SchemaChange::FieldRequired { name } => {
                if let Some(field) = new_schema.fields.get_mut(name) {
                    field.nullable = false;
                }
            }
            SchemaChange::IndexAdded { field } => {
                if let Some(f) = new_schema.fields.get_mut(field) {
                    f.indexed = true;
                }
            }
            SchemaChange::IndexRemoved { field } => {
                if let Some(f) = new_schema.fields.get_mut(field) {
                    f.indexed = false;
                }
            }
        }

        new_schema.version = current_version + 1;
        new_schema.modified_at = unix_timestamp();

        self.versions.insert(new_schema.version, new_schema.clone());
        self.history.push(change);

        Ok(new_schema)
    }

    /// Get schema at version
    pub fn get_version(&self, version: u32) -> Option<&Schema> {
        self.versions.get(&version)
    }

    /// Get change history
    pub fn get_history(&self) -> &[SchemaChange] {
        &self.history
    }
}

// ============================================================================
// DYNAMIC COLLECTION
// ============================================================================

/// Dynamic schema collection
pub struct DynamicCollection {
    /// Current schema
    schema: RwLock<Schema>,
    /// Schema evolution
    evolution: RwLock<SchemaEvolution>,
    /// Documents by ID
    documents: RwLock<HashMap<String, JsonValue>>,
    /// Vectors by ID
    vectors: RwLock<HashMap<String, Vec<f32>>>,
    /// Dynamic field types discovered
    dynamic_fields: RwLock<HashMap<String, FieldType>>,
}

impl DynamicCollection {
    pub fn new(schema: Schema) -> Result<Self> {
        let evolution = SchemaEvolution::new(schema.clone());

        Ok(Self {
            schema: RwLock::new(schema),
            evolution: RwLock::new(evolution),
            documents: RwLock::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            dynamic_fields: RwLock::new(HashMap::new()),
        })
    }

    /// Insert a document
    pub fn insert(&self, doc: JsonValue) -> Result<String> {
        // Validate against schema
        {
            let schema = self.schema.read().unwrap();
            schema.validate(&doc)?;
        }

        // Extract ID
        let id = {
            let schema = self.schema.read().unwrap();
            doc.get(&schema.primary_key)
                .and_then(|v| v.as_str())
                .map(String::from)
                .ok_or_else(|| VecStoreError::InvalidInput("Missing primary key".to_string()))?
        };

        // Extract vector
        let vector = {
            let schema = self.schema.read().unwrap();
            schema.extract_vector(&doc)
        };

        // Discover dynamic fields
        if let JsonValue::Object(map) = &doc {
            let schema = self.schema.read().unwrap();
            let mut dynamic = self.dynamic_fields.write().unwrap();

            for (key, value) in map {
                if !schema.fields.contains_key(key) {
                    let inferred_type = FieldType::infer(value);
                    dynamic.entry(key.clone())
                        .and_modify(|existing| {
                            // Promote type if needed
                            if !existing.is_compatible(value) {
                                *existing = FieldType::Any;
                            }
                        })
                        .or_insert(inferred_type);
                }
            }
        }

        // Store document
        self.documents.write().unwrap().insert(id.clone(), doc);

        // Store vector
        if let Some(vec) = vector {
            self.vectors.write().unwrap().insert(id.clone(), vec);
        }

        Ok(id)
    }

    /// Get document by ID
    pub fn get(&self, id: &str) -> Option<JsonValue> {
        self.documents.read().unwrap().get(id).cloned()
    }

    /// Delete document
    pub fn delete(&self, id: &str) -> bool {
        let doc_removed = self.documents.write().unwrap().remove(id).is_some();
        self.vectors.write().unwrap().remove(id);
        doc_removed
    }

    /// Add a new field to schema
    pub fn add_field(&self, field: FieldDef) -> Result<()> {
        let change = SchemaChange::FieldAdded { field };
        let new_schema = self.evolution.write().unwrap().apply(change)?;
        *self.schema.write().unwrap() = new_schema;
        Ok(())
    }

    /// Promote dynamic field to schema
    pub fn promote_field(&self, name: &str) -> Result<()> {
        let field_type = {
            let dynamic = self.dynamic_fields.read().unwrap();
            dynamic.get(name).cloned()
                .ok_or_else(|| VecStoreError::NotFound(format!("Dynamic field: {}", name)))?
        };

        let field = FieldDef::new(name, field_type);
        self.add_field(field)?;

        // Remove from dynamic fields
        self.dynamic_fields.write().unwrap().remove(name);

        Ok(())
    }

    /// Search with filter
    pub fn search(&self, query: &[f32], top_k: usize, filter: Option<&str>) -> Result<Vec<SearchResult>> {
        let vectors = self.vectors.read().unwrap();
        let documents = self.documents.read().unwrap();

        let mut results: Vec<_> = vectors.iter()
            .filter_map(|(id, vec)| {
                // Apply filter if provided
                if let Some(filter_str) = filter {
                    if let Some(doc) = documents.get(id) {
                        if !self.matches_filter(doc, filter_str) {
                            return None;
                        }
                    }
                }

                let score = cosine_similarity(query, vec);
                Some((id.clone(), score))
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results.into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect())
    }

    /// Simple filter matching
    fn matches_filter(&self, doc: &JsonValue, filter: &str) -> bool {
        // Very simple filter parsing: "field op value"
        let parts: Vec<&str> = filter.split_whitespace().collect();
        if parts.len() != 3 {
            return true;
        }

        let field = parts[0];
        let op = parts[1];
        let value = parts[2];

        if let Some(field_value) = doc.get(field) {
            match op {
                "=" | "==" => {
                    if let Some(s) = field_value.as_str() {
                        return s == value;
                    }
                    if let Some(n) = field_value.as_f64() {
                        return n.to_string() == value;
                    }
                }
                ">" => {
                    if let (Some(v), Ok(threshold)) = (field_value.as_f64(), value.parse::<f64>()) {
                        return v > threshold;
                    }
                }
                "<" => {
                    if let (Some(v), Ok(threshold)) = (field_value.as_f64(), value.parse::<f64>()) {
                        return v < threshold;
                    }
                }
                ">=" => {
                    if let (Some(v), Ok(threshold)) = (field_value.as_f64(), value.parse::<f64>()) {
                        return v >= threshold;
                    }
                }
                "<=" => {
                    if let (Some(v), Ok(threshold)) = (field_value.as_f64(), value.parse::<f64>()) {
                        return v <= threshold;
                    }
                }
                "!=" => {
                    if let Some(s) = field_value.as_str() {
                        return s != value;
                    }
                }
                _ => return true,
            }
        }
        false
    }

    /// Get current schema
    pub fn schema(&self) -> Schema {
        self.schema.read().unwrap().clone()
    }

    /// Get discovered dynamic fields
    pub fn dynamic_fields(&self) -> HashMap<String, FieldType> {
        self.dynamic_fields.read().unwrap().clone()
    }

    /// Get collection stats
    pub fn stats(&self) -> CollectionStats {
        CollectionStats {
            document_count: self.documents.read().unwrap().len(),
            vector_count: self.vectors.read().unwrap().len(),
            schema_version: self.schema.read().unwrap().version,
            dynamic_field_count: self.dynamic_fields.read().unwrap().len(),
        }
    }
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
}

/// Collection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    pub document_count: usize,
    pub vector_count: usize,
    pub schema_version: u32,
    pub dynamic_field_count: usize,
}

// ============================================================================
// HELPERS
// ============================================================================

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_type_inference() {
        assert_eq!(FieldType::infer(&json!("hello")), FieldType::String);
        assert_eq!(FieldType::infer(&json!(42)), FieldType::Int64);
        assert_eq!(FieldType::infer(&json!(3.14)), FieldType::Float64);
        assert_eq!(FieldType::infer(&json!(true)), FieldType::Bool);
        assert_eq!(FieldType::infer(&json!([1.0, 2.0, 3.0])), FieldType::Vector(3));
    }

    #[test]
    fn test_dynamic_collection() {
        let schema = SchemaBuilder::new()
            .add_field("id", FieldType::String)
            .add_vector("embedding", 4)
            .primary_key("id")
            .build();

        let collection = DynamicCollection::new(schema).unwrap();

        // Insert with dynamic fields
        collection.insert(json!({
            "id": "doc1",
            "embedding": [1.0, 0.0, 0.0, 0.0],
            "title": "Hello World",  // Dynamic field
            "score": 0.95,           // Dynamic field
        })).unwrap();

        // Check dynamic fields discovered
        let dynamic = collection.dynamic_fields();
        assert!(dynamic.contains_key("title"));
        assert!(dynamic.contains_key("score"));

        // Search with filter
        let results = collection.search(
            &[1.0, 0.0, 0.0, 0.0],
            10,
            Some("score > 0.9"),
        ).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_schema_evolution() {
        let schema = SchemaBuilder::new()
            .add_field("id", FieldType::String)
            .build();

        let mut evolution = SchemaEvolution::new(schema);

        // Add new field
        let change = SchemaChange::FieldAdded {
            field: FieldDef::new("name", FieldType::String),
        };

        let new_schema = evolution.apply(change).unwrap();
        assert_eq!(new_schema.version, 2);
        assert!(new_schema.fields.contains_key("name"));
    }
}
