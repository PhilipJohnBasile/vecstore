//! GraphRAG Integration
//!
//! Combines knowledge graphs with vector search for enhanced RAG.
//! Links entities, relationships, and semantic context.
//!
//! # Features
//!
//! - **Entity Linking**: Connect documents to knowledge graph entities
//! - **Relationship Traversal**: Follow graph edges during retrieval
//! - **Hybrid Ranking**: Combine vector similarity with graph proximity
//! - **Context Expansion**: Automatically include related entities
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::graphrag::{GraphRAG, Entity, Relationship};
//!
//! let mut graph_rag = GraphRAG::new(384)?;
//!
//! // Add entities
//! graph_rag.add_entity(Entity::new("rust", "Programming Language"))?;
//! graph_rag.add_entity(Entity::new("vecstore", "Vector Database"))?;
//!
//! // Add relationships
//! graph_rag.add_relationship(Relationship::new("vecstore", "rust", "written_in"))?;
//!
//! // Add document linked to entities
//! graph_rag.add_document("doc1", "VecStore is written in Rust", &["vecstore", "rust"])?;
//!
//! // Query with graph expansion
//! let results = graph_rag.query("vector database languages", 5)?;
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};

use crate::error::{VecStoreError, Result};

/// Entity in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity ID
    pub id: String,
    /// Entity type (e.g., "Person", "Organization", "Concept")
    pub entity_type: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Properties
    pub properties: HashMap<String, serde_json::Value>,
    /// Embedding for semantic matching
    pub embedding: Option<Vec<f32>>,
}

impl Entity {
    /// Create a new entity
    pub fn new(id: impl Into<String>, entity_type: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            entity_type: entity_type.into(),
            description: None,
            properties: HashMap::new(),
            embedding: None,
        }
    }

    /// Set the name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a property
    pub fn with_property(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    /// Set embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Relationship between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Source entity ID
    pub from: String,
    /// Target entity ID
    pub to: String,
    /// Relationship type
    pub relationship_type: String,
    /// Relationship weight (0.0 - 1.0)
    pub weight: f32,
    /// Properties
    pub properties: HashMap<String, serde_json::Value>,
}

impl Relationship {
    /// Create a new relationship
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        rel_type: impl Into<String>,
    ) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            relationship_type: rel_type.into(),
            weight: 1.0,
            properties: HashMap::new(),
        }
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Add a property
    pub fn with_property(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

/// Document with entity links
#[derive(Debug, Clone)]
struct LinkedDocument {
    id: String,
    content: String,
    embedding: Vec<f32>,
    linked_entities: HashSet<String>,
    metadata: Option<serde_json::Value>,
}

/// GraphRAG query result
#[derive(Debug, Clone, Serialize)]
pub struct GraphRAGResult {
    /// Document ID
    pub id: String,
    /// Document content
    pub content: String,
    /// Combined score (vector + graph)
    pub score: f32,
    /// Vector similarity score
    pub vector_score: f32,
    /// Graph proximity score
    pub graph_score: f32,
    /// Linked entities
    pub entities: Vec<Entity>,
    /// Related entities (from graph expansion)
    pub related_entities: Vec<Entity>,
    /// Relationship path
    pub relationships: Vec<Relationship>,
}

/// Configuration for GraphRAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRAGConfig {
    /// Weight for vector similarity (0.0 - 1.0)
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,
    /// Weight for graph proximity (0.0 - 1.0)
    #[serde(default = "default_graph_weight")]
    pub graph_weight: f32,
    /// Maximum graph traversal depth
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
    /// Maximum entities to expand
    #[serde(default = "default_max_expansion")]
    pub max_expansion: usize,
    /// Include entity embeddings in search
    #[serde(default = "default_true")]
    pub use_entity_embeddings: bool,
}

fn default_vector_weight() -> f32 { 0.7 }
fn default_graph_weight() -> f32 { 0.3 }
fn default_max_depth() -> usize { 2 }
fn default_max_expansion() -> usize { 10 }
fn default_true() -> bool { true }

impl Default for GraphRAGConfig {
    fn default() -> Self {
        Self {
            vector_weight: 0.7,
            graph_weight: 0.3,
            max_depth: 2,
            max_expansion: 10,
            use_entity_embeddings: true,
        }
    }
}

/// GraphRAG search engine
pub struct GraphRAG {
    dimension: usize,
    config: GraphRAGConfig,
    /// Entities in the knowledge graph
    entities: HashMap<String, Entity>,
    /// Relationships (from_id -> [(to_id, relationship)])
    outgoing: HashMap<String, Vec<Relationship>>,
    /// Reverse relationships (to_id -> [(from_id, relationship)])
    incoming: HashMap<String, Vec<Relationship>>,
    /// Documents with entity links
    documents: HashMap<String, LinkedDocument>,
    /// Entity to document index
    entity_documents: HashMap<String, HashSet<String>>,
}

impl GraphRAG {
    /// Create a new GraphRAG engine
    pub fn new(dimension: usize) -> Result<Self> {
        Self::with_config(dimension, GraphRAGConfig::default())
    }

    /// Create with custom config
    pub fn with_config(dimension: usize, config: GraphRAGConfig) -> Result<Self> {
        Ok(Self {
            dimension,
            config,
            entities: HashMap::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            documents: HashMap::new(),
            entity_documents: HashMap::new(),
        })
    }

    /// Add an entity to the knowledge graph
    pub fn add_entity(&mut self, entity: Entity) -> Result<()> {
        self.entities.insert(entity.id.clone(), entity);
        Ok(())
    }

    /// Add a relationship between entities
    pub fn add_relationship(&mut self, rel: Relationship) -> Result<()> {
        // Verify entities exist
        if !self.entities.contains_key(&rel.from) {
            return Err(VecStoreError::NotFound(format!("Entity not found: {}", rel.from)));
        }
        if !self.entities.contains_key(&rel.to) {
            return Err(VecStoreError::NotFound(format!("Entity not found: {}", rel.to)));
        }

        // Add to outgoing edges
        self.outgoing
            .entry(rel.from.clone())
            .or_default()
            .push(rel.clone());

        // Add to incoming edges
        self.incoming
            .entry(rel.to.clone())
            .or_default()
            .push(rel);

        Ok(())
    }

    /// Add a document with entity links
    pub fn add_document(
        &mut self,
        id: &str,
        content: &str,
        embedding: Vec<f32>,
        entity_ids: &[&str],
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.dimension,
                got: embedding.len(),
            });
        }

        let linked_entities: HashSet<String> = entity_ids.iter()
            .filter(|&&id| self.entities.contains_key(id))
            .map(|&s| s.to_string())
            .collect();

        // Update entity-document index
        for entity_id in &linked_entities {
            self.entity_documents
                .entry(entity_id.clone())
                .or_default()
                .insert(id.to_string());
        }

        self.documents.insert(id.to_string(), LinkedDocument {
            id: id.to_string(),
            content: content.to_string(),
            embedding,
            linked_entities,
            metadata,
        });

        Ok(())
    }

    /// Query with graph expansion
    pub fn query(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<GraphRAGResult>> {
        if query_embedding.len() != self.dimension {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.dimension,
                got: query_embedding.len(),
            });
        }

        // Step 1: Find relevant entities (if entity embeddings are used)
        let relevant_entities = if self.config.use_entity_embeddings {
            self.find_relevant_entities(query_embedding, self.config.max_expansion)
        } else {
            Vec::new()
        };

        // Step 2: Expand entities through graph
        let expanded_entities = self.expand_entities(&relevant_entities);

        // Step 3: Find documents linked to expanded entities
        let entity_docs: HashSet<&String> = expanded_entities.iter()
            .filter_map(|e| self.entity_documents.get(&e.id))
            .flatten()
            .collect();

        // Step 4: Score all documents
        let mut results: Vec<GraphRAGResult> = Vec::new();

        for (doc_id, doc) in &self.documents {
            let vector_score = Self::cosine_similarity(query_embedding, &doc.embedding);

            // Calculate graph score based on entity overlap
            let graph_score = if entity_docs.contains(doc_id) {
                let overlap = doc.linked_entities.iter()
                    .filter(|e| expanded_entities.iter().any(|exp| exp.id == **e))
                    .count();
                overlap as f32 / doc.linked_entities.len().max(1) as f32
            } else {
                0.0
            };

            let combined_score = self.config.vector_weight * vector_score
                + self.config.graph_weight * graph_score;

            // Get linked entities
            let linked: Vec<Entity> = doc.linked_entities.iter()
                .filter_map(|id| self.entities.get(id).cloned())
                .collect();

            // Get related entities (from expansion, not directly linked)
            let related: Vec<Entity> = expanded_entities.iter()
                .filter(|e| !doc.linked_entities.contains(&e.id))
                .cloned()
                .collect();

            // Get relationships for linked entities
            let relationships = self.get_relationships_for_entities(&doc.linked_entities);

            results.push(GraphRAGResult {
                id: doc_id.clone(),
                content: doc.content.clone(),
                score: combined_score,
                vector_score,
                graph_score,
                entities: linked,
                related_entities: related,
                relationships,
            });
        }

        // Sort by combined score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(limit);

        Ok(results)
    }

    /// Find entities similar to query embedding
    fn find_relevant_entities(&self, query: &[f32], limit: usize) -> Vec<Entity> {
        let mut scored: Vec<(f32, &Entity)> = self.entities.values()
            .filter_map(|e| {
                e.embedding.as_ref().map(|emb| {
                    (Self::cosine_similarity(query, emb), e)
                })
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.into_iter()
            .take(limit)
            .map(|(_, e)| e.clone())
            .collect()
    }

    /// Expand entities through graph traversal (BFS)
    fn expand_entities(&self, seed_entities: &[Entity]) -> Vec<Entity> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut result: Vec<Entity> = Vec::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        // Initialize with seed entities
        for entity in seed_entities {
            if visited.insert(entity.id.clone()) {
                result.push(entity.clone());
                queue.push_back((entity.id.clone(), 0));
            }
        }

        // BFS expansion
        while let Some((entity_id, depth)) = queue.pop_front() {
            if depth >= self.config.max_depth {
                continue;
            }
            if result.len() >= self.config.max_expansion {
                break;
            }

            // Follow outgoing edges
            if let Some(rels) = self.outgoing.get(&entity_id) {
                for rel in rels {
                    if visited.insert(rel.to.clone())
                        && let Some(entity) = self.entities.get(&rel.to) {
                            result.push(entity.clone());
                            queue.push_back((rel.to.clone(), depth + 1));
                        }
                }
            }

            // Follow incoming edges
            if let Some(rels) = self.incoming.get(&entity_id) {
                for rel in rels {
                    if visited.insert(rel.from.clone())
                        && let Some(entity) = self.entities.get(&rel.from) {
                            result.push(entity.clone());
                            queue.push_back((rel.from.clone(), depth + 1));
                        }
                }
            }
        }

        result
    }

    /// Get relationships for a set of entities
    fn get_relationships_for_entities(&self, entity_ids: &HashSet<String>) -> Vec<Relationship> {
        let mut relationships = Vec::new();

        for id in entity_ids {
            if let Some(rels) = self.outgoing.get(id) {
                for rel in rels {
                    if entity_ids.contains(&rel.to) {
                        relationships.push(rel.clone());
                    }
                }
            }
        }

        relationships
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

    /// Get entity by ID
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// Get all entities of a type
    pub fn get_entities_by_type(&self, entity_type: &str) -> Vec<&Entity> {
        self.entities.values()
            .filter(|e| e.entity_type == entity_type)
            .collect()
    }

    /// Get relationships from an entity
    pub fn get_outgoing(&self, entity_id: &str) -> Vec<&Relationship> {
        self.outgoing.get(entity_id)
            .map(|rels| rels.iter().collect())
            .unwrap_or_default()
    }

    /// Get relationships to an entity
    pub fn get_incoming(&self, entity_id: &str) -> Vec<&Relationship> {
        self.incoming.get(entity_id)
            .map(|rels| rels.iter().collect())
            .unwrap_or_default()
    }

    /// Get statistics
    pub fn stats(&self) -> GraphStats {
        GraphStats {
            entity_count: self.entities.len(),
            relationship_count: self.outgoing.values().map(|v| v.len()).sum(),
            document_count: self.documents.len(),
        }
    }
}

/// Graph statistics
#[derive(Debug, Clone, Serialize)]
pub struct GraphStats {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub document_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_entity() {
        let mut graph = GraphRAG::new(64).unwrap();

        graph.add_entity(Entity::new("rust", "Language")).unwrap();
        graph.add_entity(Entity::new("vecstore", "Database")).unwrap();

        assert!(graph.get_entity("rust").is_some());
        assert!(graph.get_entity("vecstore").is_some());
    }

    #[test]
    fn test_add_relationship() {
        let mut graph = GraphRAG::new(64).unwrap();

        graph.add_entity(Entity::new("rust", "Language")).unwrap();
        graph.add_entity(Entity::new("vecstore", "Database")).unwrap();

        graph.add_relationship(
            Relationship::new("vecstore", "rust", "written_in")
        ).unwrap();

        let outgoing = graph.get_outgoing("vecstore");
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].relationship_type, "written_in");
    }

    #[test]
    fn test_query() {
        let mut graph = GraphRAG::new(64).unwrap();

        graph.add_entity(Entity::new("rust", "Language")).unwrap();
        graph.add_entity(Entity::new("vecstore", "Database")).unwrap();

        let embedding = vec![0.1f32; 64];
        graph.add_document(
            "doc1",
            "VecStore is written in Rust",
            embedding.clone(),
            &["rust", "vecstore"],
            None,
        ).unwrap();

        let results = graph.query(&embedding, 5).unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].entities.is_empty());
    }
}
