//! Graph RAG - Knowledge Graph + Vector Search Integration
//!
//! Combines structured knowledge graphs with vector embeddings for
//! enhanced retrieval-augmented generation (RAG).
//!
//! ## Features
//!
//! - **Entity Graphs**: Model entities and relationships
//! - **Vector + Graph**: Hybrid search combining embeddings and graph structure
//! - **Graph Traversal**: Navigate relationships during retrieval
//! - **Subgraph Extraction**: Get connected subgraphs for context
//! - **Entity Linking**: Connect text chunks to knowledge entities
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐      ┌──────────────┐
//! │  Documents  │──────│   Entities   │
//! │  (Vectors)  │      │   (Nodes)    │
//! └─────────────┘      └──────────────┘
//!        │                    │
//!        │             ┌──────┴──────┐
//!        │             │             │
//!        └────────┐    │   Relations │
//!                 │    │   (Edges)   │
//!                 ▼    │             │
//!           ┌──────────┴──────┐      │
//!           │  Graph RAG       │◄─────┘
//!           │  Query Engine    │
//!           └─────────────────┘
//! ```
//!
//! ## Use Cases
//!
//! - **Question Answering**: Use graph structure for multi-hop reasoning
//! - **Research**: Navigate paper citations and connections
//! - **Enterprise**: Link documents to organizational entities
//! - **Medical**: Connect symptoms, diseases, and treatments
//! - **Financial**: Model company relationships and events
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::graph_rag::{GraphRAG, Entity, Relation, GraphQuery};
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut graph = GraphRAG::new(384)?;
//!
//! // Add entities
//! graph.add_entity("rust", vec![0.1; 384], "programming language")?;
//! graph.add_entity("wasm", vec![0.2; 384], "web assembly")?;
//!
//! // Add relation
//! graph.add_relation("rust", "wasm", "compiles_to", 1.0)?;
//!
//! // Query: find entities and their neighbors
//! let query = GraphQuery::new(vec![0.15; 384])
//!     .with_max_hops(2)
//!     .with_limit(10);
//!
//! let results = graph.search(&query)?;
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Knowledge graph entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier
    pub id: String,

    /// Vector embedding
    pub embedding: Vec<f32>,

    /// Entity type (e.g., "person", "company", "concept")
    pub entity_type: String,

    /// Properties
    pub properties: HashMap<String, serde_json::Value>,
}

impl Entity {
    /// Create a new entity
    pub fn new(id: impl Into<String>, embedding: Vec<f32>, entity_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            embedding,
            entity_type: entity_type.into(),
            properties: HashMap::new(),
        }
    }

    /// Add property
    pub fn with_property(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

/// Relation between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Source entity ID
    pub from: String,

    /// Target entity ID
    pub to: String,

    /// Relation type (e.g., "works_at", "located_in", "similar_to")
    pub relation_type: String,

    /// Relation weight/confidence (0.0 to 1.0)
    pub weight: f32,

    /// Optional properties
    pub properties: HashMap<String, serde_json::Value>,
}

impl Relation {
    /// Create a new relation
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        relation_type: impl Into<String>,
        weight: f32,
    ) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            relation_type: relation_type.into(),
            weight,
            properties: HashMap::new(),
        }
    }

    /// Add property
    pub fn with_property(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

/// Graph search query
#[derive(Clone)]
pub struct GraphQuery {
    /// Query embedding
    pub embedding: Vec<f32>,

    /// Maximum number of results
    pub limit: usize,

    /// Maximum graph hops from matched entities
    pub max_hops: usize,

    /// Filter by entity type
    pub entity_type_filter: Option<String>,

    /// Filter by relation type
    pub relation_type_filter: Option<Vec<String>>,

    /// Minimum relation weight threshold
    pub min_relation_weight: f32,
}

impl GraphQuery {
    /// Create a new graph query
    pub fn new(embedding: Vec<f32>) -> Self {
        Self {
            embedding,
            limit: 10,
            max_hops: 1,
            entity_type_filter: None,
            relation_type_filter: None,
            min_relation_weight: 0.0,
        }
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set maximum hops for graph traversal
    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.max_hops = max_hops;
        self
    }

    /// Filter by entity type
    pub fn with_entity_type(mut self, entity_type: impl Into<String>) -> Self {
        self.entity_type_filter = Some(entity_type.into());
        self
    }

    /// Filter by relation types
    pub fn with_relation_types(mut self, types: Vec<String>) -> Self {
        self.relation_type_filter = Some(types);
        self
    }

    /// Set minimum relation weight
    pub fn with_min_relation_weight(mut self, weight: f32) -> Self {
        self.min_relation_weight = weight;
        self
    }
}

/// Graph search result
#[derive(Debug, Clone)]
pub struct GraphResult {
    /// Matched entity
    pub entity: Entity,

    /// Vector similarity score
    pub score: f32,

    /// Graph distance (hops from query match)
    pub hops: usize,

    /// Path from query match to this entity
    pub path: Vec<String>,

    /// Connected entities (neighbors)
    pub neighbors: Vec<Entity>,
}

/// Graph RAG - Knowledge Graph + Vector Search
pub struct GraphRAG {
    /// Vector dimension
    dimension: usize,

    /// Entities indexed by ID
    entities: HashMap<String, Entity>,

    /// Outgoing edges: from -> list of relations
    outgoing: HashMap<String, Vec<Relation>>,

    /// Incoming edges: to -> list of relations
    incoming: HashMap<String, Vec<Relation>>,
}

impl GraphRAG {
    /// Create a new Graph RAG instance
    pub fn new(dimension: usize) -> Result<Self> {
        Ok(Self {
            dimension,
            entities: HashMap::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
        })
    }

    /// Add an entity to the graph
    pub fn add_entity(
        &mut self,
        id: impl Into<String>,
        embedding: Vec<f32>,
        entity_type: impl Into<String>,
    ) -> Result<()> {
        let id = id.into();

        if embedding.len() != self.dimension {
            return Err(anyhow!(
                "Embedding dimension {} doesn't match graph dimension {}",
                embedding.len(),
                self.dimension
            ));
        }

        let entity = Entity::new(id.clone(), embedding, entity_type);
        self.entities.insert(id.clone(), entity);

        // Initialize edge lists
        self.outgoing.entry(id.clone()).or_insert_with(Vec::new);
        self.incoming.entry(id).or_insert_with(Vec::new);

        Ok(())
    }

    /// Add entity with properties
    pub fn add_entity_with_properties(&mut self, entity: Entity) -> Result<()> {
        if entity.embedding.len() != self.dimension {
            return Err(anyhow!(
                "Embedding dimension {} doesn't match graph dimension {}",
                entity.embedding.len(),
                self.dimension
            ));
        }

        let id = entity.id.clone();
        self.entities.insert(id.clone(), entity);

        // Initialize edge lists
        self.outgoing.entry(id.clone()).or_insert_with(Vec::new);
        self.incoming.entry(id).or_insert_with(Vec::new);

        Ok(())
    }

    /// Add a relation between entities
    pub fn add_relation(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        relation_type: impl Into<String>,
        weight: f32,
    ) -> Result<()> {
        let from = from.into();
        let to = to.into();

        // Validate entities exist
        if !self.entities.contains_key(&from) {
            return Err(anyhow!("Source entity '{}' not found", from));
        }
        if !self.entities.contains_key(&to) {
            return Err(anyhow!("Target entity '{}' not found", to));
        }

        let relation = Relation::new(from.clone(), to.clone(), relation_type, weight);

        // Add to outgoing edges
        self.outgoing
            .entry(from.clone())
            .or_insert_with(Vec::new)
            .push(relation.clone());

        // Add to incoming edges
        self.incoming
            .entry(to)
            .or_insert_with(Vec::new)
            .push(relation);

        Ok(())
    }

    /// Add relation with properties
    pub fn add_relation_with_properties(&mut self, relation: Relation) -> Result<()> {
        // Validate entities exist
        if !self.entities.contains_key(&relation.from) {
            return Err(anyhow!("Source entity '{}' not found", relation.from));
        }
        if !self.entities.contains_key(&relation.to) {
            return Err(anyhow!("Target entity '{}' not found", relation.to));
        }

        let from = relation.from.clone();
        let to = relation.to.clone();

        // Add to outgoing edges
        self.outgoing
            .entry(from)
            .or_insert_with(Vec::new)
            .push(relation.clone());

        // Add to incoming edges
        self.incoming
            .entry(to)
            .or_insert_with(Vec::new)
            .push(relation);

        Ok(())
    }

    /// Search the graph
    pub fn search(&self, query: &GraphQuery) -> Result<Vec<GraphResult>> {
        if query.embedding.len() != self.dimension {
            return Err(anyhow!(
                "Query embedding dimension {} doesn't match graph dimension {}",
                query.embedding.len(),
                self.dimension
            ));
        }

        // Step 1: Find nearest entities by vector similarity
        let mut candidates: Vec<(String, f32)> = self
            .entities
            .iter()
            .filter(|(_, entity)| {
                if let Some(ref filter) = query.entity_type_filter {
                    &entity.entity_type == filter
                } else {
                    true
                }
            })
            .map(|(id, entity)| {
                let distance = euclidean_distance(&query.embedding, &entity.embedding);
                let score = 1.0 / (1.0 + distance);
                (id.clone(), score)
            })
            .collect();

        // Sort by score
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(query.limit);

        // Step 2: Expand via graph traversal
        let mut results = Vec::new();
        let mut visited = HashSet::new();

        for (entity_id, score) in candidates {
            if visited.contains(&entity_id) {
                continue;
            }

            // BFS to expand neighbors
            let subgraph = self.expand_subgraph(&entity_id, query, &mut visited)?;

            // Add root entity
            let entity = self.entities.get(&entity_id).unwrap().clone();
            let neighbors: Vec<Entity> = subgraph
                .iter()
                .filter_map(|id| self.entities.get(id).cloned())
                .collect();

            results.push(GraphResult {
                entity,
                score,
                hops: 0,
                path: vec![entity_id.clone()],
                neighbors,
            });

            visited.insert(entity_id);
        }

        Ok(results)
    }

    /// Expand subgraph from an entity using BFS
    fn expand_subgraph(
        &self,
        start_id: &str,
        query: &GraphQuery,
        visited: &mut HashSet<String>,
    ) -> Result<Vec<String>> {
        let mut queue = VecDeque::new();
        let mut subgraph = Vec::new();

        queue.push_back((start_id.to_string(), 0));

        while let Some((entity_id, hops)) = queue.pop_front() {
            if hops >= query.max_hops {
                continue;
            }

            if let Some(relations) = self.outgoing.get(&entity_id) {
                for relation in relations {
                    // Apply filters
                    if relation.weight < query.min_relation_weight {
                        continue;
                    }

                    if let Some(ref filter) = query.relation_type_filter {
                        if !filter.contains(&relation.relation_type) {
                            continue;
                        }
                    }

                    if !visited.contains(&relation.to) {
                        subgraph.push(relation.to.clone());
                        queue.push_back((relation.to.clone(), hops + 1));
                        visited.insert(relation.to.clone());
                    }
                }
            }
        }

        Ok(subgraph)
    }

    /// Get entity by ID
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// Get outgoing relations for an entity
    pub fn get_outgoing(&self, id: &str) -> Vec<&Relation> {
        self.outgoing
            .get(id)
            .map(|rels| rels.iter().collect())
            .unwrap_or_default()
    }

    /// Get incoming relations for an entity
    pub fn get_incoming(&self, id: &str) -> Vec<&Relation> {
        self.incoming
            .get(id)
            .map(|rels| rels.iter().collect())
            .unwrap_or_default()
    }

    /// Get all neighbors of an entity
    pub fn get_neighbors(&self, id: &str) -> Vec<String> {
        let mut neighbors = HashSet::new();

        // Outgoing
        if let Some(relations) = self.outgoing.get(id) {
            for rel in relations {
                neighbors.insert(rel.to.clone());
            }
        }

        // Incoming
        if let Some(relations) = self.incoming.get(id) {
            for rel in relations {
                neighbors.insert(rel.from.clone());
            }
        }

        neighbors.into_iter().collect()
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let total_relations: usize = self.outgoing.values().map(|v| v.len()).sum();

        let mut entity_types: HashMap<String, usize> = HashMap::new();
        for entity in self.entities.values() {
            *entity_types.entry(entity.entity_type.clone()).or_insert(0) += 1;
        }

        let mut relation_types: HashMap<String, usize> = HashMap::new();
        for relations in self.outgoing.values() {
            for rel in relations {
                *relation_types.entry(rel.relation_type.clone()).or_insert(0) += 1;
            }
        }

        GraphStats {
            num_entities: self.entities.len(),
            num_relations: total_relations,
            entity_types,
            relation_types,
        }
    }

    /// Remove entity (and all its relations)
    pub fn remove_entity(&mut self, id: &str) -> Result<bool> {
        if !self.entities.contains_key(id) {
            return Ok(false);
        }

        // Remove entity
        self.entities.remove(id);

        // Remove outgoing relations
        self.outgoing.remove(id);

        // Remove from incoming relations
        for relations in self.incoming.values_mut() {
            relations.retain(|r| r.from != id);
        }

        // Remove incoming relations for this entity
        self.incoming.remove(id);

        // Remove from outgoing relations
        for relations in self.outgoing.values_mut() {
            relations.retain(|r| r.to != id);
        }

        Ok(true)
    }

    /// Get number of entities
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_entities: usize,
    pub num_relations: usize,
    pub entity_types: HashMap<String, usize>,
    pub relation_types: HashMap<String, usize>,
}

/// Helper: Euclidean distance
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_basic() {
        let mut graph = GraphRAG::new(64).unwrap();

        // Add entities
        graph.add_entity("rust", vec![0.1; 64], "language").unwrap();
        graph
            .add_entity("python", vec![0.2; 64], "language")
            .unwrap();
        graph.add_entity("wasm", vec![0.3; 64], "platform").unwrap();

        assert_eq!(graph.len(), 3);

        // Add relations
        graph
            .add_relation("rust", "wasm", "compiles_to", 1.0)
            .unwrap();
        graph
            .add_relation("python", "wasm", "compiles_to", 0.8)
            .unwrap();

        // Check relations
        let rust_out = graph.get_outgoing("rust");
        assert_eq!(rust_out.len(), 1);
        assert_eq!(rust_out[0].to, "wasm");

        let wasm_in = graph.get_incoming("wasm");
        assert_eq!(wasm_in.len(), 2);
    }

    #[test]
    fn test_graph_search() {
        let mut graph = GraphRAG::new(32).unwrap();

        // Add entities (make doc1 clearly closest to 0.1)
        graph.add_entity("doc1", vec![0.1; 32], "document").unwrap();
        graph.add_entity("doc2", vec![0.5; 32], "document").unwrap();
        graph.add_entity("topic1", vec![0.3; 32], "topic").unwrap();

        // Add relations
        graph.add_relation("doc1", "topic1", "about", 1.0).unwrap();
        graph.add_relation("doc2", "topic1", "about", 0.9).unwrap();

        // Search with query vector closer to doc1
        let query = GraphQuery::new(vec![0.1; 32])
            .with_limit(5)
            .with_max_hops(1);

        let results = graph.search(&query).unwrap();

        assert!(!results.is_empty());
        // doc1 should be closest to [0.1; 32]
        assert_eq!(results[0].entity.id, "doc1");
    }

    #[test]
    fn test_graph_traversal() {
        let mut graph = GraphRAG::new(32).unwrap();

        // Create chain: A -> B -> C
        graph.add_entity("A", vec![0.1; 32], "node").unwrap();
        graph.add_entity("B", vec![0.2; 32], "node").unwrap();
        graph.add_entity("C", vec![0.3; 32], "node").unwrap();

        graph.add_relation("A", "B", "connects", 1.0).unwrap();
        graph.add_relation("B", "C", "connects", 1.0).unwrap();

        // Query with 2 hops should reach C from A
        let query = GraphQuery::new(vec![0.1; 32])
            .with_limit(1)
            .with_max_hops(2);

        let results = graph.search(&query).unwrap();

        assert_eq!(results.len(), 1);
        // Should have neighbors due to traversal
        assert!(!results[0].neighbors.is_empty());
    }

    #[test]
    fn test_entity_type_filter() {
        let mut graph = GraphRAG::new(32).unwrap();

        graph.add_entity("rust", vec![0.1; 32], "language").unwrap();
        graph.add_entity("wasm", vec![0.2; 32], "platform").unwrap();
        graph
            .add_entity("python", vec![0.3; 32], "language")
            .unwrap();

        // Query only for languages
        let query = GraphQuery::new(vec![0.15; 32]).with_entity_type("language");

        let results = graph.search(&query).unwrap();

        // Should only get languages
        for result in &results {
            assert_eq!(result.entity.entity_type, "language");
        }
    }

    #[test]
    fn test_neighbors() {
        let mut graph = GraphRAG::new(32).unwrap();

        graph.add_entity("A", vec![0.1; 32], "node").unwrap();
        graph.add_entity("B", vec![0.2; 32], "node").unwrap();
        graph.add_entity("C", vec![0.3; 32], "node").unwrap();

        graph.add_relation("A", "B", "connects", 1.0).unwrap();
        graph.add_relation("C", "A", "connects", 1.0).unwrap();

        let neighbors = graph.get_neighbors("A");

        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&"B".to_string()));
        assert!(neighbors.contains(&"C".to_string()));
    }

    #[test]
    fn test_remove_entity() {
        let mut graph = GraphRAG::new(32).unwrap();

        graph.add_entity("A", vec![0.1; 32], "node").unwrap();
        graph.add_entity("B", vec![0.2; 32], "node").unwrap();

        graph.add_relation("A", "B", "connects", 1.0).unwrap();

        assert_eq!(graph.len(), 2);

        let removed = graph.remove_entity("A").unwrap();
        assert!(removed);
        assert_eq!(graph.len(), 1);

        // Relations should be cleaned up
        let b_in = graph.get_incoming("B");
        assert_eq!(b_in.len(), 0);
    }

    #[test]
    fn test_stats() {
        let mut graph = GraphRAG::new(32).unwrap();

        graph.add_entity("rust", vec![0.1; 32], "language").unwrap();
        graph
            .add_entity("python", vec![0.2; 32], "language")
            .unwrap();
        graph.add_entity("wasm", vec![0.3; 32], "platform").unwrap();

        graph
            .add_relation("rust", "wasm", "compiles_to", 1.0)
            .unwrap();
        graph
            .add_relation("python", "wasm", "compiles_to", 0.8)
            .unwrap();

        let stats = graph.stats();

        assert_eq!(stats.num_entities, 3);
        assert_eq!(stats.num_relations, 2);
        assert_eq!(stats.entity_types.get("language"), Some(&2));
        assert_eq!(stats.entity_types.get("platform"), Some(&1));
        assert_eq!(stats.relation_types.get("compiles_to"), Some(&2));
    }

    #[test]
    fn test_entity_with_properties() {
        let mut graph = GraphRAG::new(32).unwrap();

        let entity = Entity::new("rust", vec![0.1; 32], "language")
            .with_property("year", serde_json::json!(2010))
            .with_property("paradigm", serde_json::json!("systems"));

        graph.add_entity_with_properties(entity).unwrap();

        let retrieved = graph.get_entity("rust").unwrap();
        assert_eq!(
            retrieved.properties.get("year"),
            Some(&serde_json::json!(2010))
        );
    }

    #[test]
    fn test_relation_weight_filter() {
        let mut graph = GraphRAG::new(32).unwrap();

        graph.add_entity("A", vec![0.1; 32], "node").unwrap();
        graph.add_entity("B", vec![0.2; 32], "node").unwrap();
        graph.add_entity("C", vec![0.3; 32], "node").unwrap();

        graph.add_relation("A", "B", "strong", 0.9).unwrap();
        graph.add_relation("A", "C", "weak", 0.1).unwrap();

        // Query with min weight 0.5 should only traverse strong relation
        let query = GraphQuery::new(vec![0.1; 32])
            .with_max_hops(1)
            .with_min_relation_weight(0.5);

        let results = graph.search(&query).unwrap();

        // Should find A and expand to B (not C)
        assert!(results[0].neighbors.iter().any(|e| e.id == "B"));
    }
}
