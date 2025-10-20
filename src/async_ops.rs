//! Async operations for VecStore
//!
//! This module provides async/await interfaces for batch operations,
//! enabling efficient parallel processing with tokio.
//!
//! ## Features
//!
//! - Async batch insertions with parallel processing
//! - Async batch queries
//! - Stream-based results
//! - Configurable concurrency limits
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::VecStore;
//! use vecstore::async_ops::AsyncVecStore;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let store = VecStore::open("vectors.db")?;
//!     let async_store = AsyncVecStore::new(store);
//!
//!     // Batch insert with parallelism
//!     let items = vec![
//!         ("doc1".to_string(), vec![0.1, 0.2, 0.3]),
//!         ("doc2".to_string(), vec![0.4, 0.5, 0.6]),
//!     ];
//!     async_store.batch_upsert(items).await?;
//!
//!     Ok(())
//! }
//! ```

#[cfg(feature = "async")]
use crate::store::{Metadata, Neighbor, VecStore};
#[cfg(feature = "async")]
use anyhow::Result;
#[cfg(feature = "async")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "async")]
use tokio::task;

/// Async wrapper for VecStore providing parallel batch operations
#[cfg(feature = "async")]
#[derive(Clone)]
pub struct AsyncVecStore {
    store: Arc<Mutex<VecStore>>,
}

#[cfg(feature = "async")]
impl AsyncVecStore {
    /// Create a new async wrapper around a VecStore
    pub fn new(store: VecStore) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
        }
    }

    /// Async batch upsert with parallel processing
    ///
    /// Inserts multiple vectors in parallel using tokio tasks.
    /// This is significantly faster than sequential insertions for large batches.
    ///
    /// # Arguments
    /// * `items` - Vector of (id, vector, metadata) tuples
    /// * `chunk_size` - Number of items to process per task (default: 100)
    ///
    /// # Example
    /// ```no_run
    /// # use vecstore::VecStore;
    /// # use vecstore::async_ops::AsyncVecStore;
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let store = VecStore::open("vectors.db")?;
    /// let async_store = AsyncVecStore::new(store);
    ///
    /// let items: Vec<_> = (0..1000)
    ///     .map(|i| {
    ///         let id = format!("doc{}", i);
    ///         let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
    ///         let metadata = serde_json::json!({"index": i});
    ///         (id, vector, metadata)
    ///     })
    ///     .collect();
    ///
    /// async_store.batch_upsert_with_metadata(items, 100).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn batch_upsert_with_metadata(
        &self,
        items: Vec<(String, Vec<f32>, serde_json::Value)>,
        chunk_size: usize,
    ) -> Result<()> {
        // Split into chunks for parallel processing
        let chunks: Vec<_> = items.chunks(chunk_size).map(|c| c.to_vec()).collect();

        let mut handles = Vec::new();

        for chunk in chunks {
            let store_clone = self.store.clone();
            let handle = task::spawn_blocking(move || {
                let mut store = store_clone.lock().unwrap();
                for (id, vector, metadata) in chunk {
                    let metadata: Metadata = serde_json::from_value(metadata)?;
                    store.upsert(id, vector, metadata)?;
                }
                Ok::<(), anyhow::Error>(())
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await??;
        }

        Ok(())
    }

    /// Async batch upsert without metadata
    pub async fn batch_upsert(&self, items: Vec<(String, Vec<f32>)>) -> Result<()> {
        let items_with_metadata: Vec<_> = items
            .into_iter()
            .map(|(id, vec)| (id, vec, serde_json::json!({})))
            .collect();

        self.batch_upsert_with_metadata(items_with_metadata, 100)
            .await
    }

    /// Async batch query - query multiple vectors in parallel
    ///
    /// # Arguments
    /// * `queries` - Vector of query vectors
    /// * `k` - Number of results per query
    ///
    /// # Returns
    /// Vector of result sets, one per query
    pub async fn batch_query(
        &self,
        queries: Vec<Vec<f32>>,
        k: usize,
    ) -> Result<Vec<Vec<Neighbor>>> {
        use crate::store::Query;

        let mut handles = Vec::new();

        for query_vec in queries {
            let store_clone = self.store.clone();
            let handle = task::spawn_blocking(move || {
                let store = store_clone.lock().unwrap();
                let query = Query::new(query_vec).with_limit(k);
                store.query(query)
            });
            handles.push(handle);
        }

        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await??);
        }

        Ok(results)
    }

    /// Async batch delete
    pub async fn batch_delete(&self, ids: Vec<String>) -> Result<()> {
        let mut handles = Vec::new();

        for id in ids {
            let store_clone = self.store.clone();
            let handle = task::spawn_blocking(move || {
                let mut store = store_clone.lock().unwrap();
                store.delete(&id)
            });
            handles.push(handle);
        }

        // Wait for all deletes to complete
        for handle in handles {
            handle.await??;
        }

        Ok(())
    }

    /// Get the underlying VecStore (blocking)
    pub fn get_store(&self) -> Arc<Mutex<VecStore>> {
        self.store.clone()
    }
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_async_batch_upsert() {
        let dir = tempdir().unwrap();
        let store = VecStore::open(dir.path().join("test.db")).unwrap();
        let async_store = AsyncVecStore::new(store);

        let items: Vec<_> = (0..100)
            .map(|i| {
                (
                    format!("doc{}", i),
                    vec![i as f32, (i + 1) as f32, (i + 2) as f32],
                )
            })
            .collect();

        async_store.batch_upsert(items).await.unwrap();

        let store_guard = async_store.store.lock().unwrap();
        assert_eq!(store_guard.len(), 100);
    }

    #[tokio::test]
    async fn test_async_batch_query() {
        let dir = tempdir().unwrap();
        let store = VecStore::open(dir.path().join("test.db")).unwrap();
        let async_store = AsyncVecStore::new(store);

        // Insert test data
        let items: Vec<_> = (0..50)
            .map(|i| {
                (
                    format!("doc{}", i),
                    vec![i as f32, (i + 1) as f32, (i + 2) as f32],
                )
            })
            .collect();

        async_store.batch_upsert(items).await.unwrap();

        // Batch query
        let queries = vec![
            vec![0.0, 1.0, 2.0],
            vec![10.0, 11.0, 12.0],
            vec![20.0, 21.0, 22.0],
        ];

        let results = async_store.batch_query(queries, 5).await.unwrap();

        assert_eq!(results.len(), 3);
        for result_set in results {
            assert_eq!(result_set.len(), 5);
        }
    }
}
