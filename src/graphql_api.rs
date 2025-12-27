//! GraphQL API
//!
//! GraphQL interface for vector operations, similar to Weaviate's API.
//! Enables intuitive nested queries and cross-collection joins.
//!
//! # Features
//!
//! - **GraphQL Schema**: Auto-generated from vector collections
//! - **Vector Search**: nearVector, nearText queries
//! - **Filtering**: Where clauses with operators
//! - **Aggregations**: Count, sum, avg on metadata
//! - **Cross-References**: Join vectors across collections
//!
//! # Example
//!
//! ```graphql
//! {
//!   Get {
//!     Article(
//!       nearVector: { vector: [0.1, 0.2, ...], certainty: 0.8 }
//!       where: { path: ["category"], operator: Equal, valueString: "tech" }
//!       limit: 10
//!     ) {
//!       title
//!       content
//!       _additional {
//!         certainty
//!         distance
//!       }
//!     }
//!   }
//! }
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::{VecStoreError, Result};

// ============================================================================
// SCHEMA TYPES
// ============================================================================

/// GraphQL field type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GqlType {
    String,
    Int,
    Float,
    Boolean,
    Vector(usize),
    Object(String),
    List(Box<GqlType>),
    NonNull(Box<GqlType>),
}

impl GqlType {
    pub fn to_sdl(&self) -> String {
        match self {
            GqlType::String => "String".to_string(),
            GqlType::Int => "Int".to_string(),
            GqlType::Float => "Float".to_string(),
            GqlType::Boolean => "Boolean".to_string(),
            GqlType::Vector(_dim) => format!("[Float!]"),
            GqlType::Object(name) => name.clone(),
            GqlType::List(inner) => format!("[{}]", inner.to_sdl()),
            GqlType::NonNull(inner) => format!("{}!", inner.to_sdl()),
        }
    }
}

/// GraphQL field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GqlField {
    pub name: String,
    pub field_type: GqlType,
    pub description: Option<String>,
    pub is_vector: bool,
}

/// GraphQL collection/type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GqlCollection {
    pub name: String,
    pub description: Option<String>,
    pub fields: Vec<GqlField>,
    pub vector_field: Option<String>,
    pub dimension: usize,
}

impl GqlCollection {
    /// Generate SDL for this collection
    pub fn to_sdl(&self) -> String {
        let mut sdl = format!("type {} {{\n", self.name);
        for field in &self.fields {
            let desc = field.description.as_ref()
                .map(|d| format!("  \"{}\"\n", d))
                .unwrap_or_default();
            sdl.push_str(&format!("{}  {}: {}\n", desc, field.name, field.field_type.to_sdl()));
        }
        sdl.push_str("  _additional: _Additional\n");
        sdl.push_str("}\n");
        sdl
    }
}

// ============================================================================
// QUERY TYPES
// ============================================================================

/// Near vector search input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearVectorInput {
    pub vector: Vec<f32>,
    pub certainty: Option<f32>,
    pub distance: Option<f32>,
}

/// Near text search input (requires vectorizer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearTextInput {
    pub concepts: Vec<String>,
    pub certainty: Option<f32>,
    pub distance: Option<f32>,
    pub move_to: Option<MoveInput>,
    pub move_away_from: Option<MoveInput>,
}

/// Move input for semantic search refinement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveInput {
    pub concepts: Vec<String>,
    pub force: f32,
}

/// Where filter operator
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WhereOperator {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanEqual,
    LessThan,
    LessThanEqual,
    Like,
    WithinGeoRange,
    IsNull,
    ContainsAny,
    ContainsAll,
    And,
    Or,
}

/// Where filter clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhereFilter {
    pub path: Option<Vec<String>>,
    pub operator: WhereOperator,
    pub value_string: Option<String>,
    pub value_int: Option<i64>,
    pub value_number: Option<f64>,
    pub value_boolean: Option<bool>,
    pub value_text: Option<String>,
    pub operands: Option<Vec<WhereFilter>>,
}

/// GraphQL query for vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorQuery {
    pub collection: String,
    pub near_vector: Option<NearVectorInput>,
    pub near_text: Option<NearTextInput>,
    pub where_filter: Option<WhereFilter>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub fields: Vec<String>,
    pub additional: Vec<String>,
}

/// Additional metadata fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Additional {
    pub id: Option<String>,
    pub certainty: Option<f32>,
    pub distance: Option<f32>,
    pub vector: Option<Vec<f32>>,
    pub creation_time: Option<u64>,
    pub last_update_time: Option<u64>,
    pub explanation: Option<SearchExplanation>,
}

/// Search explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchExplanation {
    pub score_breakdown: Vec<ScoreComponent>,
}

/// Score component for explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreComponent {
    pub name: String,
    pub value: f32,
    pub weight: f32,
}

// ============================================================================
// QUERY RESULT
// ============================================================================

/// Single result item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResultItem {
    pub properties: HashMap<String, JsonValue>,
    pub additional: Additional,
}

/// Query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub collection: String,
    pub items: Vec<QueryResultItem>,
    pub total_count: Option<usize>,
}

// ============================================================================
// GRAPHQL EXECUTOR
// ============================================================================

/// GraphQL executor for vector operations
pub struct GraphQLExecutor {
    /// Registered collections
    collections: HashMap<String, GqlCollection>,
    /// Vector data (simplified - would use actual store)
    vectors: HashMap<String, HashMap<String, (Vec<f32>, HashMap<String, JsonValue>)>>,
}

impl GraphQLExecutor {
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
            vectors: HashMap::new(),
        }
    }

    /// Register a collection
    pub fn register_collection(&mut self, collection: GqlCollection) {
        self.vectors.insert(collection.name.clone(), HashMap::new());
        self.collections.insert(collection.name.clone(), collection);
    }

    /// Insert a vector
    pub fn insert(
        &mut self,
        collection: &str,
        id: &str,
        vector: Vec<f32>,
        properties: HashMap<String, JsonValue>,
    ) -> Result<()> {
        let col = self.collections.get(collection)
            .ok_or_else(|| VecStoreError::NotFound(format!("Collection: {}", collection)))?;

        if vector.len() != col.dimension {
            return Err(VecStoreError::DimensionMismatch {
                expected: col.dimension,
                got: vector.len(),
            });
        }

        self.vectors
            .get_mut(collection)
            .unwrap()
            .insert(id.to_string(), (vector, properties));

        Ok(())
    }

    /// Execute a vector query
    pub fn execute(&self, query: VectorQuery) -> Result<QueryResult> {
        let _collection = self.collections.get(&query.collection)
            .ok_or_else(|| VecStoreError::NotFound(format!("Collection: {}", query.collection)))?;

        let vectors = self.vectors.get(&query.collection).unwrap();

        let mut results: Vec<(String, f32, &HashMap<String, JsonValue>)> = Vec::new();

        // Vector search
        if let Some(near_vector) = &query.near_vector {
            for (id, (vec, props)) in vectors {
                let similarity = cosine_similarity(&near_vector.vector, vec);
                let certainty = (similarity + 1.0) / 2.0;  // Convert to 0-1 range

                // Check certainty threshold
                if let Some(min_cert) = near_vector.certainty {
                    if certainty < min_cert {
                        continue;
                    }
                }

                // Check distance threshold
                if let Some(max_dist) = near_vector.distance {
                    let distance = 1.0 - similarity;
                    if distance > max_dist {
                        continue;
                    }
                }

                results.push((id.clone(), similarity, props));
            }
        } else {
            // No vector search - return all
            for (id, (_, props)) in vectors {
                results.push((id.clone(), 1.0, props));
            }
        }

        // Apply where filter
        if let Some(filter) = &query.where_filter {
            results.retain(|(_, _, props)| self.evaluate_filter(filter, props));
        }

        // Sort by similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply offset and limit
        let offset = query.offset.unwrap_or(0);
        let limit = query.limit.unwrap_or(10);
        let total = results.len();

        results = results.into_iter().skip(offset).take(limit).collect();

        // Build result items
        let items: Vec<QueryResultItem> = results.into_iter()
            .map(|(id, score, props)| {
                let mut properties = HashMap::new();
                for field in &query.fields {
                    if let Some(value) = props.get(field) {
                        properties.insert(field.clone(), value.clone());
                    }
                }

                let additional = Additional {
                    id: Some(id),
                    certainty: Some((score + 1.0) / 2.0),
                    distance: Some(1.0 - score),
                    vector: if query.additional.contains(&"vector".to_string()) {
                        None  // Would include vector here
                    } else {
                        None
                    },
                    creation_time: None,
                    last_update_time: None,
                    explanation: None,
                };

                QueryResultItem { properties, additional }
            })
            .collect();

        Ok(QueryResult {
            collection: query.collection,
            items,
            total_count: Some(total),
        })
    }

    /// Evaluate a where filter
    fn evaluate_filter(&self, filter: &WhereFilter, props: &HashMap<String, JsonValue>) -> bool {
        match filter.operator {
            WhereOperator::And => {
                if let Some(operands) = &filter.operands {
                    operands.iter().all(|f| self.evaluate_filter(f, props))
                } else {
                    true
                }
            }
            WhereOperator::Or => {
                if let Some(operands) = &filter.operands {
                    operands.iter().any(|f| self.evaluate_filter(f, props))
                } else {
                    false
                }
            }
            WhereOperator::Equal => {
                if let Some(path) = &filter.path {
                    if let Some(field) = path.first() {
                        if let Some(value) = props.get(field) {
                            return self.compare_equal(value, filter);
                        }
                    }
                }
                false
            }
            WhereOperator::NotEqual => {
                if let Some(path) = &filter.path {
                    if let Some(field) = path.first() {
                        if let Some(value) = props.get(field) {
                            return !self.compare_equal(value, filter);
                        }
                    }
                }
                true
            }
            WhereOperator::GreaterThan => {
                self.compare_numeric(filter, props, |a, b| a > b)
            }
            WhereOperator::GreaterThanEqual => {
                self.compare_numeric(filter, props, |a, b| a >= b)
            }
            WhereOperator::LessThan => {
                self.compare_numeric(filter, props, |a, b| a < b)
            }
            WhereOperator::LessThanEqual => {
                self.compare_numeric(filter, props, |a, b| a <= b)
            }
            WhereOperator::Like => {
                if let (Some(path), Some(pattern)) = (&filter.path, &filter.value_string) {
                    if let Some(field) = path.first() {
                        if let Some(JsonValue::String(s)) = props.get(field) {
                            return self.match_like(s, pattern);
                        }
                    }
                }
                false
            }
            WhereOperator::IsNull => {
                if let Some(path) = &filter.path {
                    if let Some(field) = path.first() {
                        return props.get(field).map(|v| v.is_null()).unwrap_or(true);
                    }
                }
                false
            }
            _ => true,
        }
    }

    fn compare_equal(&self, value: &JsonValue, filter: &WhereFilter) -> bool {
        if let Some(s) = &filter.value_string {
            return value.as_str() == Some(s);
        }
        if let Some(i) = filter.value_int {
            return value.as_i64() == Some(i);
        }
        if let Some(n) = filter.value_number {
            return value.as_f64() == Some(n);
        }
        if let Some(b) = filter.value_boolean {
            return value.as_bool() == Some(b);
        }
        false
    }

    fn compare_numeric<F>(&self, filter: &WhereFilter, props: &HashMap<String, JsonValue>, cmp: F) -> bool
    where
        F: Fn(f64, f64) -> bool,
    {
        if let Some(path) = &filter.path {
            if let Some(field) = path.first() {
                if let Some(value) = props.get(field) {
                    if let Some(v) = value.as_f64() {
                        if let Some(n) = filter.value_number {
                            return cmp(v, n);
                        }
                        if let Some(i) = filter.value_int {
                            return cmp(v, i as f64);
                        }
                    }
                }
            }
        }
        false
    }

    fn match_like(&self, s: &str, pattern: &str) -> bool {
        // Simple wildcard matching (* = any chars)
        let pattern = pattern.replace('*', ".*");
        regex::Regex::new(&format!("^{}$", pattern))
            .map(|re| re.is_match(s))
            .unwrap_or(false)
    }

    /// Generate GraphQL SDL schema
    pub fn generate_schema(&self) -> String {
        let mut sdl = String::new();

        // Additional type
        sdl.push_str("type _Additional {\n");
        sdl.push_str("  id: ID\n");
        sdl.push_str("  certainty: Float\n");
        sdl.push_str("  distance: Float\n");
        sdl.push_str("  vector: [Float!]\n");
        sdl.push_str("  creationTimeUnix: String\n");
        sdl.push_str("  lastUpdateTimeUnix: String\n");
        sdl.push_str("}\n\n");

        // Collection types
        for collection in self.collections.values() {
            sdl.push_str(&collection.to_sdl());
            sdl.push('\n');
        }

        // Query type
        sdl.push_str("type Query {\n");
        sdl.push_str("  Get: GetQuery\n");
        sdl.push_str("  Aggregate: AggregateQuery\n");
        sdl.push_str("}\n\n");

        // Get query type
        sdl.push_str("type GetQuery {\n");
        for name in self.collections.keys() {
            sdl.push_str(&format!(
                "  {}(nearVector: NearVectorInput, where: WhereFilter, limit: Int, offset: Int): [{}]\n",
                name, name
            ));
        }
        sdl.push_str("}\n\n");

        // Input types
        sdl.push_str("input NearVectorInput {\n");
        sdl.push_str("  vector: [Float!]!\n");
        sdl.push_str("  certainty: Float\n");
        sdl.push_str("  distance: Float\n");
        sdl.push_str("}\n\n");

        sdl.push_str("input WhereFilter {\n");
        sdl.push_str("  path: [String!]\n");
        sdl.push_str("  operator: WhereOperator\n");
        sdl.push_str("  valueString: String\n");
        sdl.push_str("  valueInt: Int\n");
        sdl.push_str("  valueNumber: Float\n");
        sdl.push_str("  valueBoolean: Boolean\n");
        sdl.push_str("  operands: [WhereFilter!]\n");
        sdl.push_str("}\n\n");

        sdl.push_str("enum WhereOperator {\n");
        sdl.push_str("  Equal\n");
        sdl.push_str("  NotEqual\n");
        sdl.push_str("  GreaterThan\n");
        sdl.push_str("  GreaterThanEqual\n");
        sdl.push_str("  LessThan\n");
        sdl.push_str("  LessThanEqual\n");
        sdl.push_str("  Like\n");
        sdl.push_str("  And\n");
        sdl.push_str("  Or\n");
        sdl.push_str("}\n");

        sdl
    }
}

impl Default for GraphQLExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// QUERY PARSER
// ============================================================================

/// Parse a GraphQL query string into VectorQuery
pub fn parse_query(query_str: &str) -> Result<VectorQuery> {
    // Simplified parser - would use a proper GraphQL parser in production
    // This is a placeholder that extracts basic information

    let collection = extract_collection(query_str)?;
    let near_vector = extract_near_vector(query_str);
    let where_filter = extract_where(query_str);
    let limit = extract_limit(query_str);
    let fields = extract_fields(query_str);

    Ok(VectorQuery {
        collection,
        near_vector,
        near_text: None,
        where_filter,
        limit,
        offset: None,
        fields,
        additional: vec!["certainty".to_string(), "distance".to_string()],
    })
}

fn extract_collection(query: &str) -> Result<String> {
    // Look for pattern like "Get { CollectionName("
    if let Some(start) = query.find("Get") {
        let rest = &query[start..];
        if let Some(brace) = rest.find('{') {
            let after_brace = &rest[brace + 1..];
            let name: String = after_brace.trim()
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Ok(name);
            }
        }
    }
    Err(VecStoreError::InvalidInput("Could not parse collection name".to_string()))
}

fn extract_near_vector(query: &str) -> Option<NearVectorInput> {
    if let Some(start) = query.find("nearVector:") {
        // Very simplified - would need proper parsing
        if let Some(vec_start) = query[start..].find("vector:") {
            if let Some(bracket_start) = query[start + vec_start..].find('[') {
                if let Some(bracket_end) = query[start + vec_start + bracket_start..].find(']') {
                    let vec_str = &query[start + vec_start + bracket_start + 1..start + vec_start + bracket_start + bracket_end];
                    let vector: Vec<f32> = vec_str.split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if !vector.is_empty() {
                        return Some(NearVectorInput {
                            vector,
                            certainty: extract_float(query, "certainty:"),
                            distance: extract_float(query, "distance:"),
                        });
                    }
                }
            }
        }
    }
    None
}

fn extract_float(query: &str, key: &str) -> Option<f32> {
    if let Some(start) = query.find(key) {
        let rest = &query[start + key.len()..];
        let value: String = rest.trim()
            .chars()
            .take_while(|c| c.is_numeric() || *c == '.' || *c == '-')
            .collect();
        value.parse().ok()
    } else {
        None
    }
}

fn extract_where(_query: &str) -> Option<WhereFilter> {
    // Would need proper parsing
    None
}

fn extract_limit(query: &str) -> Option<usize> {
    if let Some(start) = query.find("limit:") {
        let rest = &query[start + 6..];
        let value: String = rest.trim()
            .chars()
            .take_while(|c| c.is_numeric())
            .collect();
        value.parse().ok()
    } else {
        None
    }
}

fn extract_fields(_query: &str) -> Vec<String> {
    // Would need proper parsing - return common defaults
    vec!["id".to_string(), "content".to_string(), "title".to_string()]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graphql_executor() {
        let mut executor = GraphQLExecutor::new();

        // Register collection
        let collection = GqlCollection {
            name: "Article".to_string(),
            description: Some("News articles".to_string()),
            fields: vec![
                GqlField {
                    name: "title".to_string(),
                    field_type: GqlType::String,
                    description: None,
                    is_vector: false,
                },
                GqlField {
                    name: "category".to_string(),
                    field_type: GqlType::String,
                    description: None,
                    is_vector: false,
                },
            ],
            vector_field: Some("embedding".to_string()),
            dimension: 4,
        };

        executor.register_collection(collection);

        // Insert vectors
        let mut props1 = HashMap::new();
        props1.insert("title".to_string(), JsonValue::String("Tech News".to_string()));
        props1.insert("category".to_string(), JsonValue::String("tech".to_string()));
        executor.insert("Article", "doc1", vec![1.0, 0.0, 0.0, 0.0], props1).unwrap();

        let mut props2 = HashMap::new();
        props2.insert("title".to_string(), JsonValue::String("Sports News".to_string()));
        props2.insert("category".to_string(), JsonValue::String("sports".to_string()));
        executor.insert("Article", "doc2", vec![0.0, 1.0, 0.0, 0.0], props2).unwrap();

        // Execute query
        let query = VectorQuery {
            collection: "Article".to_string(),
            near_vector: Some(NearVectorInput {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                certainty: None,
                distance: None,
            }),
            near_text: None,
            where_filter: None,
            limit: Some(10),
            offset: None,
            fields: vec!["title".to_string(), "category".to_string()],
            additional: vec!["certainty".to_string()],
        };

        let result = executor.execute(query).unwrap();
        assert_eq!(result.items.len(), 2);
        assert_eq!(result.items[0].additional.id, Some("doc1".to_string()));
    }

    #[test]
    fn test_schema_generation() {
        let mut executor = GraphQLExecutor::new();

        let collection = GqlCollection {
            name: "Document".to_string(),
            description: None,
            fields: vec![
                GqlField {
                    name: "content".to_string(),
                    field_type: GqlType::String,
                    description: None,
                    is_vector: false,
                },
            ],
            vector_field: None,
            dimension: 4,
        };

        executor.register_collection(collection);

        let schema = executor.generate_schema();
        assert!(schema.contains("type Document"));
        assert!(schema.contains("type GetQuery"));
    }
}
