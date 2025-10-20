// Async API for vecstore
//
// This module provides async wrappers around the synchronous VecStore API,
// enabling use in Tokio-based async applications.
//
// The approach: spawn blocking tasks for CPU-intensive operations like
// HNSW search, while keeping the API async-friendly.

use crate::{Collection, HybridQuery, Metadata, Neighbor, Query, VecDatabase, VecStore};
use anyhow::Result;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Thread-safe async wrapper around VecStore
///
/// # Example
///
/// ```no_run
/// use vecstore::AsyncVecStore;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let store = AsyncVecStore::open("./data").await?;
///
///     let results = store.query_async(
///         vec![1.0, 0.0, 0.0],
///         10,
///         None
///     ).await?;
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct AsyncVecStore {
    inner: Arc<RwLock<VecStore>>,
}

impl AsyncVecStore {
    /// Open or create a vector store asynchronously
    pub async fn open<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let path = path.into();
        let store = tokio::task::spawn_blocking(move || VecStore::open(path)).await??;

        Ok(Self {
            inner: Arc::new(RwLock::new(store)),
        })
    }

    /// Insert or update a single vector asynchronously
    pub async fn upsert(&self, id: String, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = inner.write().unwrap();
            store.upsert(id, vector, metadata)
        })
        .await?
    }

    /// Batch insert vectors asynchronously (parallelized internally)
    pub async fn batch_upsert(&self, records: Vec<crate::Record>) -> Result<()> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = inner.write().unwrap();
            store.batch_upsert(records)
        })
        .await?
    }

    /// Query for similar vectors asynchronously
    pub async fn query(&self, query: Query) -> Result<Vec<Neighbor>> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let store = inner.read().unwrap();
            store.query(query)
        })
        .await?
    }

    /// Query with SQL-like filter string
    pub async fn query_with_filter(
        &self,
        vector: Vec<f32>,
        k: usize,
        filter: &str,
    ) -> Result<Vec<Neighbor>> {
        let filter = filter.to_string();
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let store = inner.read().unwrap();
            store.query_with_filter(vector, k, &filter)
        })
        .await?
    }

    /// Remove a vector by ID
    pub async fn remove(&self, id: &str) -> Result<()> {
        let id = id.to_string();
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = inner.write().unwrap();
            store.remove(&id)
        })
        .await?
    }

    /// Save the store to disk asynchronously
    pub async fn save(&self) -> Result<()> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let store = inner.read().unwrap();
            store.save()
        })
        .await?
    }

    /// Get the number of vectors in the store
    pub async fn count(&self) -> usize {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let store = inner.read().unwrap();
            store.count()
        })
        .await
        .unwrap_or(0)
    }

    /// Get the vector dimension
    pub async fn dimension(&self) -> usize {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let store = inner.read().unwrap();
            store.dimension()
        })
        .await
        .unwrap_or(0)
    }

    /// Create a named snapshot asynchronously
    pub async fn create_snapshot(&self, name: &str) -> Result<()> {
        let name = name.to_string();
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let store = inner.read().unwrap();
            store.create_snapshot(&name)
        })
        .await?
    }

    /// List all snapshots
    pub async fn list_snapshots(&self) -> Result<Vec<(String, String, usize)>> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let store = inner.read().unwrap();
            store.list_snapshots()
        })
        .await?
    }

    /// Restore from a snapshot
    pub async fn restore_snapshot(&self, name: &str) -> Result<()> {
        let name = name.to_string();
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = inner.write().unwrap();
            store.restore_snapshot(&name)
        })
        .await?
    }

    /// Delete a snapshot
    pub async fn delete_snapshot(&self, name: &str) -> Result<()> {
        let name = name.to_string();
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let store = inner.read().unwrap();
            store.delete_snapshot(&name)
        })
        .await?
    }

    /// Get access to the underlying synchronous store
    ///
    /// # Warning
    ///
    /// This provides direct access to the inner store. Use carefully to avoid
    /// blocking async tasks.
    pub fn with_sync<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&VecStore) -> Result<R>,
    {
        let store = self.inner.read().unwrap();
        f(&store)
    }

    /// Get mutable access to the underlying synchronous store
    ///
    /// # Warning
    ///
    /// This provides direct mutable access to the inner store. Use carefully
    /// to avoid blocking async tasks.
    pub fn with_sync_mut<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut VecStore) -> Result<R>,
    {
        let mut store = self.inner.write().unwrap();
        f(&mut store)
    }

    /// Hybrid search combining vector similarity and keyword matching
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vecstore::{AsyncVecStore, HybridQuery};
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// # let store = AsyncVecStore::open("./data").await?;
    /// let results = store.hybrid_query(HybridQuery {
    ///     vector: vec![1.0, 0.0, 0.0],
    ///     keywords: "rust programming".to_string(),
    ///     k: 10,
    ///     filter: None,
    ///     alpha: 0.7, // 70% vector, 30% keyword
    /// }).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn hybrid_query(&self, query: HybridQuery) -> Result<Vec<Neighbor>> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let store = inner.read().unwrap();
            store.hybrid_query(query)
        })
        .await?
    }

    /// Index text for keyword search
    pub async fn index_text(&self, id: &str, text: &str) -> Result<()> {
        let id = id.to_string();
        let text = text.to_string();
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = inner.write().unwrap();
            store.index_text(&id, &text)
        })
        .await?
    }
}

/// Async wrapper around VecDatabase for multi-collection support
///
/// # Example
///
/// ```no_run
/// use vecstore::AsyncVecDatabase;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut db = AsyncVecDatabase::open("./data").await?;
///     let mut docs = db.create_collection("documents").await?;
///
///     // Use async collection
///     let results = docs.query_async(vec![1.0, 0.0, 0.0], 10).await?;
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct AsyncVecDatabase {
    inner: Arc<RwLock<VecDatabase>>,
}

impl AsyncVecDatabase {
    /// Open or create a vector database asynchronously
    pub async fn open<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let path = path.into();
        let db = tokio::task::spawn_blocking(move || VecDatabase::open(path)).await??;

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Create a new collection asynchronously
    pub async fn create_collection(&self, name: &str) -> Result<AsyncCollection> {
        let name = name.to_string();
        let inner = self.inner.clone();
        let collection = tokio::task::spawn_blocking(move || {
            let mut db = inner.write().unwrap();
            db.create_collection(&name)
        })
        .await??;

        Ok(AsyncCollection::new(collection))
    }

    /// Get an existing collection
    pub async fn get_collection(&self, name: &str) -> Result<Option<AsyncCollection>> {
        let name = name.to_string();
        let inner = self.inner.clone();
        let collection_opt = tokio::task::spawn_blocking(move || {
            let db = inner.read().unwrap();
            db.get_collection(&name)
        })
        .await??;

        Ok(collection_opt.map(AsyncCollection::new))
    }

    /// List all collection names
    pub async fn list_collections(&self) -> Result<Vec<String>> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = inner.read().unwrap();
            db.list_collections()
                .map_err(|e| anyhow::anyhow!("List collections failed: {}", e))
        })
        .await?
    }

    /// Delete a collection
    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        let name = name.to_string();
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut db = inner.write().unwrap();
            db.delete_collection(&name)
                .map_err(|e| anyhow::anyhow!("Delete collection failed: {}", e))
        })
        .await?
    }
}

/// Async wrapper around Collection
#[derive(Clone)]
pub struct AsyncCollection {
    inner: Arc<RwLock<Collection>>,
}

impl AsyncCollection {
    fn new(collection: Collection) -> Self {
        Self {
            inner: Arc::new(RwLock::new(collection)),
        }
    }

    /// Insert or update a vector asynchronously
    pub async fn upsert(&self, id: String, vector: Vec<f32>, metadata: Metadata) -> Result<()> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut collection = inner.write().unwrap();
            collection
                .upsert(id, vector, metadata)
                .map_err(|e| anyhow::anyhow!("Collection upsert failed: {}", e))
        })
        .await?
    }

    /// Query for similar vectors asynchronously
    pub async fn query(&self, query: Query) -> Result<Vec<Neighbor>> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let collection = inner.read().unwrap();
            collection
                .query(query)
                .map_err(|e| anyhow::anyhow!("Collection query failed: {}", e))
        })
        .await?
    }

    /// Delete a vector by ID
    pub async fn delete(&self, id: &str) -> Result<()> {
        let id = id.to_string();
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut collection = inner.write().unwrap();
            collection
                .delete(&id)
                .map_err(|e| anyhow::anyhow!("Collection delete failed: {}", e))
        })
        .await?
    }

    /// Get the number of vectors
    pub async fn count(&self) -> Result<usize> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let collection = inner.read().unwrap();
            collection
                .count()
                .map_err(|e| anyhow::anyhow!("Collection count failed: {}", e))
        })
        .await?
    }

    /// Get collection statistics
    pub async fn stats(&self) -> Result<crate::namespace_manager::NamespaceStats> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let collection = inner.read().unwrap();
            collection
                .stats()
                .map_err(|e| anyhow::anyhow!("Collection stats failed: {}", e))
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_async_basic_operations() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = AsyncVecStore::open(temp_dir.path()).await.unwrap();

        // Insert
        let meta = Metadata {
            fields: HashMap::new(),
        };
        store
            .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta.clone())
            .await
            .unwrap();
        store
            .upsert("doc2".into(), vec![0.0, 1.0, 0.0], meta)
            .await
            .unwrap();

        // Count
        assert_eq!(store.count().await, 2);

        // Query
        let results = store
            .query(Query {
                vector: vec![1.0, 0.0, 0.0],
                k: 1,
                filter: None,
            })
            .await
            .unwrap();

        assert!(results.len() >= 1);
        assert_eq!(results[0].id, "doc1");
    }

    #[tokio::test]
    async fn test_async_concurrent_queries() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = AsyncVecStore::open(temp_dir.path()).await.unwrap();

        // Insert test data
        let meta = Metadata {
            fields: HashMap::new(),
        };
        for i in 0..10 {
            store
                .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta.clone())
                .await
                .unwrap();
        }

        // Run multiple queries concurrently
        let store1 = store.clone();
        let store2 = store.clone();
        let store3 = store.clone();

        let (r1, r2, r3) = tokio::join!(
            store1.query(Query {
                vector: vec![5.0, 0.0, 0.0],
                k: 3,
                filter: None,
            }),
            store2.query(Query {
                vector: vec![2.0, 0.0, 0.0],
                k: 3,
                filter: None,
            }),
            store3.query(Query {
                vector: vec![8.0, 0.0, 0.0],
                k: 3,
                filter: None,
            }),
        );

        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert!(r3.is_ok());
    }

    #[tokio::test]
    async fn test_async_filter_query() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = AsyncVecStore::open(temp_dir.path()).await.unwrap();

        // Insert with metadata
        for i in 0..10 {
            let mut meta = Metadata {
                fields: HashMap::new(),
            };
            meta.fields.insert("value".into(), serde_json::json!(i));

            store
                .upsert(format!("doc{}", i), vec![i as f32, 0.0, 0.0], meta)
                .await
                .unwrap();
        }

        // Query with filter
        let results = store
            .query_with_filter(vec![5.0, 0.0, 0.0], 10, "value >= 5")
            .await
            .unwrap();

        assert!(results.len() >= 1);
        for result in &results {
            let value = result
                .metadata
                .fields
                .get("value")
                .unwrap()
                .as_i64()
                .unwrap();
            assert!(value >= 5);
        }
    }

    #[tokio::test]
    async fn test_async_snapshots() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = AsyncVecStore::open(temp_dir.path()).await.unwrap();

        // Insert data
        let meta = Metadata {
            fields: HashMap::new(),
        };
        store
            .upsert("doc1".into(), vec![1.0, 0.0, 0.0], meta.clone())
            .await
            .unwrap();

        // Create snapshot
        store.create_snapshot("test-snapshot").await.unwrap();

        // Add more data
        store
            .upsert("doc2".into(), vec![2.0, 0.0, 0.0], meta)
            .await
            .unwrap();
        assert_eq!(store.count().await, 2);

        // Restore
        store.restore_snapshot("test-snapshot").await.unwrap();
        assert_eq!(store.count().await, 1);

        // List snapshots
        let snapshots = store.list_snapshots().await.unwrap();
        assert_eq!(snapshots.len(), 1);

        // Delete snapshot
        store.delete_snapshot("test-snapshot").await.unwrap();
        let snapshots = store.list_snapshots().await.unwrap();
        assert_eq!(snapshots.len(), 0);
    }
}
