//! LangChain and LlamaIndex integration layer
//!
//! Provides compatibility interfaces for popular AI/LLM frameworks:
//! - LangChain: Python framework for building LLM applications
//! - LlamaIndex: Framework for connecting LLMs with external data
//!
//! This module implements standard patterns used by these frameworks:
//! - Document ingestion with automatic text chunking
//! - Vector store interface for embedding storage/retrieval
//! - Retriever interface for RAG (Retrieval-Augmented Generation)
//! - Metadata filtering and search
//! - MMR (Maximal Marginal Relevance) for diverse results

use crate::error::{Result, VecStoreError};
use crate::store::{Metadata, Neighbor, Query, VecStore};
use crate::text_splitter::TextSplitter;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Document representation compatible with LangChain/LlamaIndex
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// The text content of the document
    pub page_content: String,

    /// Metadata associated with the document
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Document {
    /// Create a new document
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            page_content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the document
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }
}

/// Document with similarity score (returned from searches)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredDocument {
    /// The document
    pub document: Document,

    /// Similarity score (distance from query)
    pub score: f32,
}

/// Vector store interface compatible with LangChain patterns
pub struct LangChainVectorStore {
    /// Underlying vecstore
    store: VecStore,

    /// Text splitter for chunking documents
    text_splitter: Option<Box<dyn TextSplitter>>,

    /// Embedding function (user-provided)
    embedding_fn: Option<Box<dyn Fn(&str) -> Result<Vec<f32>> + Send + Sync>>,
}

impl LangChainVectorStore {
    /// Create a new LangChain-compatible vector store
    pub fn new(store: VecStore) -> Self {
        Self {
            store,
            text_splitter: None,
            embedding_fn: None,
        }
    }

    /// Set the text splitter for automatic chunking
    pub fn with_text_splitter<T: TextSplitter + 'static>(mut self, splitter: T) -> Self {
        self.text_splitter = Some(Box::new(splitter));
        self
    }

    /// Set the embedding function
    pub fn with_embedding_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Result<Vec<f32>> + Send + Sync + 'static,
    {
        self.embedding_fn = Some(Box::new(f));
        self
    }

    /// Add documents to the vector store
    ///
    /// If a text splitter is configured, documents will be automatically chunked.
    /// If an embedding function is set, embeddings will be computed automatically.
    pub fn add_documents(&mut self, documents: Vec<Document>) -> Result<Vec<String>> {
        let mut ids = Vec::new();

        for (idx, doc) in documents.into_iter().enumerate() {
            // Split document if splitter is configured
            let chunks = if let Some(splitter) = &self.text_splitter {
                splitter.split_text(&doc.page_content)?
            } else {
                vec![doc.page_content.clone()]
            };

            // Process each chunk
            for (chunk_idx, chunk_text) in chunks.into_iter().enumerate() {
                let chunk_id = format!("doc_{}_{}", idx, chunk_idx);

                // Compute embedding if function is set
                if let Some(embed_fn) = &self.embedding_fn {
                    let embedding = embed_fn(&chunk_text)?;

                    // Use document metadata
                    let mut metadata = doc.metadata.clone();
                    metadata.insert("chunk_index".to_string(), serde_json::json!(chunk_idx));
                    metadata.insert("text".to_string(), serde_json::json!(chunk_text));

                    self.store
                        .upsert(chunk_id.clone(), embedding, Metadata { fields: metadata })
                        .map_err(|e| VecStoreError::Serialization(e.to_string()))?;
                } else {
                    return Err(VecStoreError::InvalidConfig(
                        "No embedding function set. Use with_embedding_fn() or add_embeddings()"
                            .to_string(),
                    ));
                }

                ids.push(chunk_id);
            }
        }

        Ok(ids)
    }

    /// Add documents with pre-computed embeddings
    pub fn add_embeddings(
        &mut self,
        texts: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadatas: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        if texts.len() != embeddings.len() {
            return Err(VecStoreError::InvalidConfig(format!(
                "Texts and embeddings length mismatch: {} vs {}",
                texts.len(),
                embeddings.len()
            )));
        }

        let mut ids = Vec::new();

        for (idx, (text, embedding)) in texts.into_iter().zip(embeddings.into_iter()).enumerate() {
            let id = format!("doc_{}", idx);

            let mut metadata = metadatas
                .as_ref()
                .and_then(|m| m.get(idx))
                .cloned()
                .unwrap_or_default();

            metadata.insert("text".to_string(), serde_json::json!(text));

            self.store
                .upsert(id.clone(), embedding, Metadata { fields: metadata })
                .map_err(|e| VecStoreError::Serialization(e.to_string()))?;
            ids.push(id);
        }

        Ok(ids)
    }

    /// Similarity search - find k most similar documents
    pub fn similarity_search(
        &self,
        query: &str,
        k: usize,
        filter: Option<&str>,
    ) -> Result<Vec<Document>> {
        let scored_docs = self.similarity_search_with_score(query, k, filter)?;
        Ok(scored_docs.into_iter().map(|sd| sd.document).collect())
    }

    /// Similarity search with relevance scores
    pub fn similarity_search_with_score(
        &self,
        query_text: &str,
        k: usize,
        filter: Option<&str>,
    ) -> Result<Vec<ScoredDocument>> {
        // Compute query embedding
        let query_embedding = if let Some(embed_fn) = &self.embedding_fn {
            embed_fn(query_text)?
        } else {
            return Err(VecStoreError::InvalidConfig(
                "No embedding function set. Use with_embedding_fn()".to_string(),
            ));
        };

        // Build query
        let mut query = Query::new(query_embedding).with_limit(k);
        if let Some(f) = filter {
            query = query.with_filter(f);
        }

        // Execute search
        let results = self
            .store
            .query(query)
            .map_err(|e| VecStoreError::Serialization(e.to_string()))?;

        // Convert to scored documents
        Ok(results
            .into_iter()
            .map(|neighbor| self.neighbor_to_scored_document(neighbor))
            .collect::<Result<Vec<_>>>()?)
    }

    /// Similarity search by vector (pre-computed embedding)
    pub fn similarity_search_by_vector(
        &self,
        embedding: Vec<f32>,
        k: usize,
        filter: Option<&str>,
    ) -> Result<Vec<ScoredDocument>> {
        let mut query = Query::new(embedding).with_limit(k);
        if let Some(f) = filter {
            query = query.with_filter(f);
        }

        let results = self
            .store
            .query(query)
            .map_err(|e| VecStoreError::Serialization(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|neighbor| self.neighbor_to_scored_document(neighbor))
            .collect::<Result<Vec<_>>>()?)
    }

    /// MMR (Maximal Marginal Relevance) search for diverse results
    ///
    /// Balances relevance and diversity:
    /// - lambda=1.0: only relevance (same as similarity search)
    /// - lambda=0.0: only diversity
    /// - lambda=0.5: balanced
    pub fn max_marginal_relevance_search(
        &self,
        query_text: &str,
        k: usize,
        fetch_k: usize,
        lambda: f32,
        filter: Option<&str>,
    ) -> Result<Vec<Document>> {
        // Get initial candidate set (fetch more than k)
        let candidates =
            self.similarity_search_with_score(query_text, fetch_k.max(k * 3), filter)?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // MMR algorithm
        let mut selected = Vec::new();
        let mut remaining: Vec<_> = candidates.into_iter().collect();

        // Select first document (most relevant)
        if let Some(first) = remaining.first() {
            selected.push(first.clone());
            remaining.remove(0);
        }

        // Iteratively select documents balancing relevance and diversity
        while selected.len() < k && !remaining.is_empty() {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = 0;

            for (idx, candidate) in remaining.iter().enumerate() {
                // Relevance score (similarity to query)
                let relevance = 1.0 - candidate.score; // Convert distance to similarity

                // Diversity score (max similarity to already selected documents)
                let mut max_similarity: f32 = 0.0;
                for selected_doc in &selected {
                    let sim = self.compute_similarity(
                        &candidate.document.page_content,
                        &selected_doc.document.page_content,
                    )?;
                    max_similarity = max_similarity.max(sim);
                }

                // MMR score: balance relevance and diversity
                let mmr_score = lambda * relevance - (1.0 - lambda) * max_similarity;

                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = idx;
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        Ok(selected.into_iter().map(|sd| sd.document).collect())
    }

    /// Delete documents by IDs
    pub fn delete(&mut self, ids: Vec<String>) -> Result<()> {
        for id in ids {
            self.store
                .delete(&id)
                .map_err(|e| VecStoreError::Serialization(e.to_string()))?;
        }
        Ok(())
    }

    /// Get the underlying VecStore (for advanced operations)
    pub fn inner(&self) -> &VecStore {
        &self.store
    }

    /// Get mutable access to the underlying VecStore
    pub fn inner_mut(&mut self) -> &mut VecStore {
        &mut self.store
    }

    // Helper methods

    fn neighbor_to_scored_document(&self, neighbor: Neighbor) -> Result<ScoredDocument> {
        let metadata = neighbor.metadata.fields;

        let text = metadata
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(ScoredDocument {
            document: Document {
                page_content: text,
                metadata,
            },
            score: neighbor.score,
        })
    }

    fn compute_similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        if let Some(embed_fn) = &self.embedding_fn {
            let emb1 = embed_fn(text1)?;
            let emb2 = embed_fn(text2)?;

            // Cosine similarity
            let dot: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
            let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();

            Ok(dot / (norm1 * norm2))
        } else {
            // Fallback: use simple string similarity
            Ok(simple_string_similarity(text1, text2))
        }
    }
}

/// Retriever interface for RAG applications (LlamaIndex pattern)
pub struct VectorStoreRetriever {
    store: LangChainVectorStore,
    search_kwargs: RetrieverConfig,
}

#[derive(Debug, Clone)]
pub struct RetrieverConfig {
    /// Number of documents to retrieve
    pub k: usize,

    /// Metadata filter expression
    pub filter: Option<String>,

    /// Use MMR for diversity
    pub use_mmr: bool,

    /// MMR lambda parameter (only if use_mmr=true)
    pub mmr_lambda: f32,

    /// Number of docs to fetch for MMR (only if use_mmr=true)
    pub fetch_k: usize,
}

impl Default for RetrieverConfig {
    fn default() -> Self {
        Self {
            k: 4,
            filter: None,
            use_mmr: false,
            mmr_lambda: 0.5,
            fetch_k: 20,
        }
    }
}

impl VectorStoreRetriever {
    /// Create a new retriever
    pub fn new(store: LangChainVectorStore) -> Self {
        Self {
            store,
            search_kwargs: RetrieverConfig::default(),
        }
    }

    /// Set retriever configuration
    pub fn with_config(mut self, config: RetrieverConfig) -> Self {
        self.search_kwargs = config;
        self
    }

    /// Retrieve relevant documents for a query
    pub fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        if self.search_kwargs.use_mmr {
            self.store.max_marginal_relevance_search(
                query,
                self.search_kwargs.k,
                self.search_kwargs.fetch_k,
                self.search_kwargs.mmr_lambda,
                self.search_kwargs.filter.as_deref(),
            )
        } else {
            self.store.similarity_search(
                query,
                self.search_kwargs.k,
                self.search_kwargs.filter.as_deref(),
            )
        }
    }

    /// Retrieve relevant documents with scores
    pub fn get_relevant_documents_with_scores(&self, query: &str) -> Result<Vec<ScoredDocument>> {
        self.store.similarity_search_with_score(
            query,
            self.search_kwargs.k,
            self.search_kwargs.filter.as_deref(),
        )
    }
}

/// LlamaIndex-compatible node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique node ID
    pub id: String,

    /// Node text content
    pub text: String,

    /// Node embedding
    pub embedding: Option<Vec<f32>>,

    /// Node metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Relationships to other nodes
    pub relationships: HashMap<String, String>,
}

impl Node {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            embedding: None,
            metadata: HashMap::new(),
            relationships: HashMap::new(),
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn with_relationship(
        mut self,
        rel_type: impl Into<String>,
        node_id: impl Into<String>,
    ) -> Self {
        self.relationships.insert(rel_type.into(), node_id.into());
        self
    }
}

/// LlamaIndex-compatible vector store index
pub struct LlamaIndexVectorStore {
    store: LangChainVectorStore,
}

impl LlamaIndexVectorStore {
    pub fn new(store: VecStore) -> Self {
        Self {
            store: LangChainVectorStore::new(store),
        }
    }

    pub fn with_embedding_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Result<Vec<f32>> + Send + Sync + 'static,
    {
        self.store = self.store.with_embedding_fn(f);
        self
    }

    /// Add nodes to the index
    pub fn add_nodes(&mut self, nodes: Vec<Node>) -> Result<Vec<String>> {
        let mut ids = Vec::new();

        for node in nodes {
            let embedding = if let Some(emb) = node.embedding {
                emb
            } else if let Some(embed_fn) = &self.store.embedding_fn {
                embed_fn(&node.text)?
            } else {
                return Err(VecStoreError::InvalidConfig(
                    "Node has no embedding and no embedding function set".to_string(),
                ));
            };

            let mut metadata = node.metadata.clone();
            metadata.insert("text".to_string(), serde_json::json!(node.text.clone()));
            metadata.insert(
                "relationships".to_string(),
                serde_json::json!(node.relationships.clone()),
            );

            self.store
                .store
                .upsert(node.id.clone(), embedding, Metadata { fields: metadata })
                .map_err(|e| VecStoreError::Serialization(e.to_string()))?;
            ids.push(node.id);
        }

        Ok(ids)
    }

    /// Query the index
    pub fn query(&self, query_text: &str, similarity_top_k: usize) -> Result<Vec<ScoredDocument>> {
        self.store
            .similarity_search_with_score(query_text, similarity_top_k, None)
    }

    /// Delete nodes by IDs
    pub fn delete_nodes(&mut self, node_ids: Vec<String>) -> Result<()> {
        self.store
            .delete(node_ids)
            .map_err(|e| VecStoreError::Serialization(e.to_string()))
    }
}

// Helper function for simple string similarity
fn simple_string_similarity(s1: &str, s2: &str) -> f32 {
    let s1_words: std::collections::HashSet<&str> = s1.split_whitespace().collect();
    let s2_words: std::collections::HashSet<&str> = s2.split_whitespace().collect();

    let intersection = s1_words.intersection(&s2_words).count();
    let union = s1_words.union(&s2_words).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VecStore;
    use tempfile::TempDir;

    fn dummy_embedding(text: &str) -> Result<Vec<f32>> {
        // Simple character-based embedding for testing
        let chars: Vec<char> = text.chars().take(10).collect();
        let mut embedding = vec![0.0; 10];
        for (i, &c) in chars.iter().enumerate() {
            embedding[i] = (c as u32 as f32) / 1000.0;
        }
        Ok(embedding)
    }

    fn create_test_store() -> (VecStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let store = VecStore::open(temp_dir.path().join("test.db")).unwrap();
        (store, temp_dir)
    }

    #[test]
    fn test_add_documents() -> Result<()> {
        let (store, _temp_dir) = create_test_store();
        let mut lc_store = LangChainVectorStore::new(store).with_embedding_fn(dummy_embedding);

        let docs = vec![
            Document::new("Hello world").with_metadata("source", serde_json::json!("test1")),
            Document::new("Rust programming").with_metadata("source", serde_json::json!("test2")),
        ];

        let ids = lc_store.add_documents(docs)?;
        assert_eq!(ids.len(), 2);

        Ok(())
    }

    #[test]
    fn test_similarity_search() -> Result<()> {
        let (store, _temp_dir) = create_test_store();
        let mut lc_store = LangChainVectorStore::new(store).with_embedding_fn(dummy_embedding);

        let docs = vec![
            Document::new("Machine learning"),
            Document::new("Deep learning"),
            Document::new("Cooking recipes"),
        ];

        lc_store.add_documents(docs)?;

        let results = lc_store.similarity_search("learning", 2, None)?;
        assert!(results.len() <= 2);

        Ok(())
    }

    #[test]
    fn test_add_embeddings() -> Result<()> {
        let (store, _temp_dir) = create_test_store();
        let mut lc_store = LangChainVectorStore::new(store);

        let texts = vec!["doc1".to_string(), "doc2".to_string()];
        let embeddings = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];

        let ids = lc_store.add_embeddings(texts, embeddings, None)?;
        assert_eq!(ids.len(), 2);

        Ok(())
    }

    #[test]
    fn test_retriever() -> Result<()> {
        let (store, _temp_dir) = create_test_store();
        let mut lc_store = LangChainVectorStore::new(store).with_embedding_fn(dummy_embedding);

        let docs = vec![
            Document::new("Vector databases"),
            Document::new("Machine learning"),
        ];

        lc_store.add_documents(docs)?;

        let retriever = VectorStoreRetriever::new(lc_store);
        let results = retriever.get_relevant_documents("databases")?;

        assert!(!results.is_empty());

        Ok(())
    }

    #[test]
    fn test_llamaindex_nodes() -> Result<()> {
        let (store, _temp_dir) = create_test_store();
        let mut llama_store = LlamaIndexVectorStore::new(store).with_embedding_fn(dummy_embedding);

        let nodes = vec![
            Node::new("node1", "First node").with_metadata("type", serde_json::json!("text")),
            Node::new("node2", "Second node")
                .with_metadata("type", serde_json::json!("text"))
                .with_relationship("parent", "node1"),
        ];

        let ids = llama_store.add_nodes(nodes)?;
        assert_eq!(ids.len(), 2);

        let results = llama_store.query("node", 2)?;
        assert!(!results.is_empty());

        Ok(())
    }
}
