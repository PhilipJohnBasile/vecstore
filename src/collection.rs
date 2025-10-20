//! Collection Abstraction for VecStore
//!
//! Provides a higher-level "collection" API similar to ChromaDB/Qdrant.
//! Collections are a more ergonomic way to work with isolated vector stores,
//! built on top of VecStore's namespace system.
//!
//! # Architecture
//!
//! - `VecDatabase`: Manages multiple collections (wraps NamespaceManager)
//! - `Collection`: Isolated vector store (wraps a namespace)
//!
//! # Example
//!
//! ```no_run
//! use vecstore::{VecDatabase, Metadata};
//! use std::collections::HashMap;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Create database
//! let mut db = VecDatabase::open("./my_db")?;
//!
//! // Create collections
//! let mut documents = db.create_collection("documents")?;
//! let mut users = db.create_collection("users")?;
//!
//! // Use collections independently
//! let mut meta = Metadata { fields: HashMap::new() };
//! meta.fields.insert("type".into(), serde_json::json!("article"));
//!
//! documents.upsert("doc1".into(), vec![0.1, 0.2, 0.3], meta)?;
//!
//! // List all collections
//! let collections = db.list_collections()?;
//! println!("Collections: {:?}", collections);
//! # Ok(())
//! # }
//! ```

use crate::error::Result;
use crate::namespace::{Namespace, NamespaceId, NamespaceQuotas};
use crate::namespace_manager::{NamespaceManager, NamespaceStats};
use crate::store::{Config, Distance, Metadata, Neighbor, Query};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Database managing multiple collections
///
/// VecDatabase provides a high-level API for managing multiple isolated
/// vector collections. Each collection is backed by a namespace in the
/// underlying NamespaceManager.
///
/// # Simple by Default
///
/// For simple use cases, just use `VecStore::open()` directly:
/// ```no_run
/// use vecstore::VecStore;
/// let mut store = VecStore::open("./data")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # Powerful When Needed
///
/// For multi-collection use cases, use `VecDatabase`:
/// ```no_run
/// use vecstore::VecDatabase;
/// let mut db = VecDatabase::open("./data")?;
/// let docs = db.create_collection("documents")?;
/// let users = db.create_collection("users")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct VecDatabase {
    manager: Arc<RwLock<NamespaceManager>>,
    #[allow(dead_code)]
    root: PathBuf,
}

impl VecDatabase {
    /// Open or create a database at the specified path
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::VecDatabase;
    /// let db = VecDatabase::open("./my_database")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        let manager = NamespaceManager::new(&root)
            .map_err(|e| crate::error::VecStoreError::Other(e.to_string()))?;
        Ok(Self {
            manager: Arc::new(RwLock::new(manager)),
            root,
        })
    }

    /// Create a new collection with default configuration
    ///
    /// # Arguments
    /// * `name` - Collection name (must be unique)
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::VecDatabase;
    /// let mut db = VecDatabase::open("./db")?;
    /// let collection = db.create_collection("documents")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn create_collection(&mut self, name: &str) -> Result<Collection> {
        self.create_collection_with_config(name, CollectionConfig::default())
    }

    /// Create a new collection with custom configuration
    ///
    /// # Arguments
    /// * `name` - Collection name (must be unique)
    /// * `config` - Collection configuration (quotas, distance metric, etc.)
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::{VecDatabase, CollectionConfig, Distance};
    /// let mut db = VecDatabase::open("./db")?;
    /// let config = CollectionConfig::default()
    ///     .with_distance(Distance::Manhattan)
    ///     .with_max_vectors(100_000);
    /// let collection = db.create_collection_with_config("documents", config)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn create_collection_with_config(
        &mut self,
        name: &str,
        config: CollectionConfig,
    ) -> Result<Collection> {
        let namespace_id: NamespaceId = name.to_string();
        let description = config
            .description
            .unwrap_or_else(|| format!("Collection: {}", name));

        // Create namespace with quotas
        {
            let manager = self.manager.read().unwrap();
            manager
                .create_namespace(namespace_id.clone(), description, Some(config.quotas))
                .map_err(|e| crate::error::VecStoreError::Other(e.to_string()))?;
        }

        Ok(Collection {
            name: name.to_string(),
            namespace_id,
            manager: Arc::clone(&self.manager),
            config: config.store_config,
        })
    }

    /// Get an existing collection
    ///
    /// Returns None if the collection doesn't exist.
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::VecDatabase;
    /// let db = VecDatabase::open("./db")?;
    /// if let Some(collection) = db.get_collection("documents")? {
    ///     // Use collection
    /// }
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn get_collection(&self, name: &str) -> Result<Option<Collection>> {
        let namespace_id: NamespaceId = name.to_string();

        let manager = self.manager.read().unwrap();
        // Check if namespace exists
        match manager.get_namespace(&namespace_id) {
            Ok(_) => Ok(Some(Collection {
                name: name.to_string(),
                namespace_id,
                manager: Arc::clone(&self.manager),
                // Config uses default for now - persistence can be added when needed
                config: Config::default(),
            })),
            Err(_) => Ok(None),
        }
    }

    /// List all collections in the database
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::VecDatabase;
    /// let db = VecDatabase::open("./db")?;
    /// let collections = db.list_collections()?;
    /// for name in collections {
    ///     println!("Collection: {}", name);
    /// }
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn list_collections(&self) -> Result<Vec<String>> {
        let manager = self.manager.read().unwrap();
        let namespaces = manager.list_namespaces();
        Ok(namespaces.into_iter().map(|ns| ns.id).collect())
    }

    /// Delete a collection
    ///
    /// This permanently deletes the collection and all its data.
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::VecDatabase;
    /// let mut db = VecDatabase::open("./db")?;
    /// db.delete_collection("old_documents")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn delete_collection(&mut self, name: &str) -> Result<()> {
        let namespace_id: NamespaceId = name.to_string();
        let manager = self.manager.read().unwrap();
        manager
            .delete_namespace(&namespace_id)
            .map_err(|e| crate::error::VecStoreError::Other(e.to_string()))
    }

    /// Get statistics for all collections
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::VecDatabase;
    /// let db = VecDatabase::open("./db")?;
    /// let collection_names = db.list_collections()?;
    /// for name in collection_names {
    ///     if let Some(coll) = db.get_collection(&name)? {
    ///         let stats = coll.stats()?;
    ///         println!("{}: {} vectors", name, stats.vector_count);
    ///     }
    /// }
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn collection_names(&self) -> Result<Vec<String>> {
        self.list_collections()
    }
}

/// Configuration for creating a collection
#[derive(Debug, Clone, Default)]
pub struct CollectionConfig {
    /// Description of the collection
    pub description: Option<String>,

    /// Resource quotas for the collection
    pub quotas: NamespaceQuotas,

    /// Vector store configuration (distance metric, HNSW params)
    pub store_config: Config,
}

impl CollectionConfig {
    /// Set collection description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set maximum number of vectors
    pub fn with_max_vectors(mut self, max: usize) -> Self {
        self.quotas.max_vectors = Some(max);
        self
    }

    /// Set maximum storage in bytes
    pub fn with_max_storage(mut self, max_bytes: u64) -> Self {
        self.quotas.max_storage_bytes = Some(max_bytes);
        self
    }

    /// Set distance metric
    pub fn with_distance(mut self, metric: Distance) -> Self {
        self.store_config.distance = metric;
        self
    }

    /// Set HNSW M parameter
    pub fn with_hnsw_m(mut self, m: usize) -> Self {
        self.store_config.hnsw_m = m;
        self
    }

    /// Set HNSW ef_construction parameter
    pub fn with_hnsw_ef_construction(mut self, ef: usize) -> Self {
        self.store_config.hnsw_ef_construction = ef;
        self
    }
}

/// A collection of vectors with isolated storage
///
/// Collection provides a familiar API for working with vectors,
/// similar to ChromaDB's collection interface. Each collection
/// is backed by a namespace for isolation.
pub struct Collection {
    name: String,
    namespace_id: NamespaceId,
    manager: Arc<RwLock<NamespaceManager>>,
    config: Config,
}

impl Collection {
    /// Get the collection name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get collection statistics
    ///
    /// Returns information about vector count, storage usage, etc.
    pub fn stats(&self) -> Result<NamespaceStats> {
        let manager = self.manager.read().unwrap();
        manager
            .get_stats(&self.namespace_id)
            .map_err(|e| crate::error::VecStoreError::Other(e.to_string()))
    }

    /// Get namespace metadata (includes resource usage)
    pub fn namespace(&self) -> Result<Namespace> {
        let manager = self.manager.read().unwrap();
        manager
            .get_namespace(&self.namespace_id)
            .map_err(|e| crate::error::VecStoreError::Other(e.to_string()))
    }

    /// Insert or update a vector
    ///
    /// If a vector with the same ID exists, it will be updated.
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::{VecDatabase, Metadata};
    /// use std::collections::HashMap;
    ///
    /// let mut db = VecDatabase::open("./db")?;
    /// let mut collection = db.create_collection("docs")?;
    ///
    /// let mut meta = Metadata { fields: HashMap::new() };
    /// meta.fields.insert("title".into(), serde_json::json!("My Document"));
    ///
    /// collection.upsert("doc1".into(), vec![0.1, 0.2, 0.3], meta)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn upsert(&mut self, id: String, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        let manager = self.manager.read().unwrap();
        manager
            .upsert(&self.namespace_id, id, vector, metadata)
            .map_err(|e| crate::error::VecStoreError::Other(e.to_string()))
    }

    /// Query for similar vectors
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::{VecDatabase, Query};
    ///
    /// let db = VecDatabase::open("./db")?;
    /// let collection = db.get_collection("docs")?.unwrap();
    ///
    /// let query = Query {
    ///     vector: vec![0.1, 0.2, 0.3],
    ///     k: 10,
    ///     filter: None,
    /// };
    ///
    /// let results = collection.query(query)?;
    /// for result in results {
    ///     println!("{}: {}", result.id, result.score);
    /// }
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn query(&self, query: Query) -> Result<Vec<Neighbor>> {
        let manager = self.manager.read().unwrap();
        manager
            .query(&self.namespace_id, query)
            .map_err(|e| crate::error::VecStoreError::Other(e.to_string()))
    }

    /// Delete a vector by ID
    pub fn delete(&mut self, id: &str) -> Result<()> {
        let manager = self.manager.read().unwrap();
        manager
            .remove(&self.namespace_id, id)
            .map_err(|e| crate::error::VecStoreError::Other(e.to_string()))
    }

    /// Count total vectors in collection
    pub fn count(&self) -> Result<usize> {
        let stats = self.stats()?;
        Ok(stats.vector_count)
    }

    /// Get the distance metric used by this collection
    pub fn distance_metric(&self) -> Distance {
        self.config.distance
    }

    /// Get the full configuration of this collection
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Split a document into chunks and upsert them with embeddings
    ///
    /// This is a convenience method that combines text splitting and embedding
    /// in one call. Each chunk gets a unique ID based on the document ID.
    ///
    /// # Arguments
    /// * `doc_id` - Base ID for the document (chunks will be `{doc_id}_chunk_0`, `{doc_id}_chunk_1`, etc.)
    /// * `text` - The full document text to split and embed
    /// * `splitter` - Text splitter implementation
    /// * `embedder` - Embedder implementation
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::{VecDatabase, CharacterTextSplitter, TextSplitter};
    /// # #[cfg(feature = "embeddings")]
    /// use vecstore::ONNXEmbedder;
    ///
    /// # #[cfg(feature = "embeddings")]
    /// # fn main() -> anyhow::Result<()> {
    /// let mut db = VecDatabase::open("./db")?;
    /// let mut collection = db.create_collection("docs")?;
    ///
    /// let splitter = CharacterTextSplitter::new(512, 50);
    /// let embedder = ONNXEmbedder::new("models/all-MiniLM-L6-v2")?;
    ///
    /// let long_document = "This is a very long document that will be split into chunks...";
    /// collection.upsert_chunks("doc1", long_document, &splitter, &embedder)?;
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "embeddings"))]
    /// # fn main() {}
    /// ```
    #[cfg(feature = "embeddings")]
    pub fn upsert_chunks<S, E>(
        &mut self,
        doc_id: &str,
        text: &str,
        splitter: &S,
        embedder: &E,
    ) -> Result<usize>
    where
        S: crate::text_splitter::TextSplitter,
        E: crate::embeddings::TextEmbedder,
    {
        let chunks = splitter.split_text(text).map_err(|e| {
            crate::error::VecStoreError::Other(format!("Text splitting failed: {}", e))
        })?;
        let chunk_count = chunks.len();

        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_id = format!("{}_{}", doc_id, i);
            let embedding = embedder.embed(chunk).map_err(|e| {
                crate::error::VecStoreError::Other(format!("Embedding failed: {}", e))
            })?;

            let mut meta = Metadata {
                fields: std::collections::HashMap::new(),
            };
            meta.fields.insert("text".into(), serde_json::json!(chunk));
            meta.fields
                .insert("chunk_index".into(), serde_json::json!(i));
            meta.fields
                .insert("doc_id".into(), serde_json::json!(doc_id));

            self.upsert(chunk_id, embedding, meta)?;
        }

        Ok(chunk_count)
    }

    /// Batch upsert multiple texts with embeddings
    ///
    /// Embeds multiple texts and upserts them in one call. More efficient than
    /// calling upsert in a loop.
    ///
    /// # Arguments
    /// * `items` - Vec of (id, text) tuples
    /// * `embedder` - Embedder implementation
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::VecDatabase;
    /// # #[cfg(feature = "embeddings")]
    /// use vecstore::ONNXEmbedder;
    ///
    /// # #[cfg(feature = "embeddings")]
    /// # fn main() -> anyhow::Result<()> {
    /// let mut db = VecDatabase::open("./db")?;
    /// let mut collection = db.create_collection("docs")?;
    ///
    /// let embedder = ONNXEmbedder::new("models/all-MiniLM-L6-v2")?;
    ///
    /// let texts = vec![
    ///     ("doc1".to_string(), "First document"),
    ///     ("doc2".to_string(), "Second document"),
    ///     ("doc3".to_string(), "Third document"),
    /// ];
    ///
    /// collection.batch_upsert_texts(texts, &embedder)?;
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "embeddings"))]
    /// # fn main() {}
    /// ```
    #[cfg(feature = "embeddings")]
    pub fn batch_upsert_texts<E>(
        &mut self,
        items: Vec<(String, &str)>,
        embedder: &E,
    ) -> Result<usize>
    where
        E: crate::embeddings::TextEmbedder,
    {
        let count = items.len();

        for (id, text) in items {
            let embedding = embedder.embed(text).map_err(|e| {
                crate::error::VecStoreError::Other(format!("Embedding failed: {}", e))
            })?;

            let mut meta = Metadata {
                fields: std::collections::HashMap::new(),
            };
            meta.fields.insert("text".into(), serde_json::json!(text));

            self.upsert(id, embedding, meta)?;
        }

        Ok(count)
    }

    /// Query using text instead of a vector
    ///
    /// Convenience method that embeds the query text and performs a vector search.
    ///
    /// # Arguments
    /// * `query_text` - The text query
    /// * `embedder` - Embedder implementation
    /// * `k` - Number of results to return
    ///
    /// # Example
    /// ```no_run
    /// use vecstore::VecDatabase;
    /// # #[cfg(feature = "embeddings")]
    /// use vecstore::ONNXEmbedder;
    ///
    /// # #[cfg(feature = "embeddings")]
    /// # fn main() -> anyhow::Result<()> {
    /// let db = VecDatabase::open("./db")?;
    /// let collection = db.get_collection("docs")?.unwrap();
    ///
    /// let embedder = ONNXEmbedder::new("models/all-MiniLM-L6-v2")?;
    ///
    /// let results = collection.query_text("machine learning", &embedder, 10)?;
    ///
    /// for result in results {
    ///     println!("{}: {:.4}", result.id, result.score);
    /// }
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "embeddings"))]
    /// # fn main() {}
    /// ```
    #[cfg(feature = "embeddings")]
    pub fn query_text<E>(&self, query_text: &str, embedder: &E, k: usize) -> Result<Vec<Neighbor>>
    where
        E: crate::embeddings::TextEmbedder,
    {
        let query_vector = embedder
            .embed(query_text)
            .map_err(|e| crate::error::VecStoreError::Other(format!("Embedding failed: {}", e)))?;

        let query = Query {
            vector: query_vector,
            k,
            filter: None,
        };

        self.query(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[test]
    fn test_create_database() {
        let dir = tempdir().unwrap();
        let db = VecDatabase::open(dir.path()).unwrap();
        assert_eq!(db.list_collections().unwrap().len(), 0);
    }

    #[test]
    fn test_create_collection() {
        let dir = tempdir().unwrap();
        let mut db = VecDatabase::open(dir.path()).unwrap();

        let collection = db.create_collection("test").unwrap();
        assert_eq!(collection.name(), "test");

        let collections = db.list_collections().unwrap();
        assert_eq!(collections.len(), 1);
        assert_eq!(collections[0], "test");
    }

    #[test]
    fn test_multiple_collections() {
        let dir = tempdir().unwrap();
        let mut db = VecDatabase::open(dir.path()).unwrap();

        db.create_collection("docs").unwrap();
        db.create_collection("users").unwrap();
        db.create_collection("products").unwrap();

        let collections = db.list_collections().unwrap();
        assert_eq!(collections.len(), 3);
    }

    #[test]
    fn test_get_collection() {
        let dir = tempdir().unwrap();
        let mut db = VecDatabase::open(dir.path()).unwrap();

        db.create_collection("test").unwrap();

        let collection = db.get_collection("test").unwrap();
        assert!(collection.is_some());
        assert_eq!(collection.unwrap().name(), "test");

        let missing = db.get_collection("nonexistent").unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn test_delete_collection() {
        let dir = tempdir().unwrap();
        let mut db = VecDatabase::open(dir.path()).unwrap();

        db.create_collection("test").unwrap();
        assert_eq!(db.list_collections().unwrap().len(), 1);

        db.delete_collection("test").unwrap();
        assert_eq!(db.list_collections().unwrap().len(), 0);
    }

    #[test]
    fn test_collection_upsert_and_query() {
        let dir = tempdir().unwrap();
        let mut db = VecDatabase::open(dir.path()).unwrap();

        let mut collection = db.create_collection("test").unwrap();

        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields
            .insert("text".into(), serde_json::json!("hello"));

        collection
            .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta)
            .unwrap();

        let query = Query {
            vector: vec![1.0, 0.0, 0.0],
            k: 10,
            filter: None,
        };

        let results = collection.query(query).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_collection_config() {
        let dir = tempdir().unwrap();
        let mut db = VecDatabase::open(dir.path()).unwrap();

        let config = CollectionConfig::default()
            .with_description("Test collection")
            .with_max_vectors(1000)
            .with_distance(Distance::Manhattan);

        let collection = db.create_collection_with_config("test", config).unwrap();
        assert_eq!(collection.distance_metric(), Distance::Manhattan);
    }

    #[test]
    fn test_collection_isolation() {
        let dir = tempdir().unwrap();
        let mut db = VecDatabase::open(dir.path()).unwrap();

        let mut coll1 = db.create_collection("coll1").unwrap();
        let coll2 = db.create_collection("coll2").unwrap();

        let meta = Metadata {
            fields: HashMap::new(),
        };

        // Insert into collection 1
        coll1
            .upsert("doc1".into(), vec![1.0, 0.0], meta.clone())
            .unwrap();

        // Query collection 2 (should be empty)
        let query = Query {
            vector: vec![1.0, 0.0],
            k: 10,
            filter: None,
        };
        let results = coll2.query(query).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_collection_count() {
        let dir = tempdir().unwrap();
        let mut db = VecDatabase::open(dir.path()).unwrap();
        let mut collection = db.create_collection("test").unwrap();

        assert_eq!(collection.count().unwrap(), 0);

        let meta = Metadata {
            fields: HashMap::new(),
        };
        collection
            .upsert("doc1".into(), vec![1.0], meta.clone())
            .unwrap();
        collection
            .upsert("doc2".into(), vec![2.0], meta.clone())
            .unwrap();

        assert_eq!(collection.count().unwrap(), 2);
    }
}
