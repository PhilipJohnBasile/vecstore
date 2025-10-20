//! Async Python API
//!
//! This module provides async/await support for Python applications.
//! Note: Full PyO3 async integration requires pyo3-asyncio crate.
//! This module provides the core async functionality that can be exposed to Python.
//!
//! ## Architecture
//!
//! The async API uses tokio for async operations and can be wrapped with
//! pyo3-asyncio for Python integration in production.
//!
//! ## Example Usage (Rust)
//!
//! ```no_run
//! use vecstore::python_async::AsyncPyVecStore;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut store = AsyncPyVecStore::new(128);
//!
//!     store.upsert(
//!         "doc1".to_string(),
//!         vec![0.1, 0.2, 0.3; 128],
//!         serde_json::json!({"title": "Document 1"})
//!     ).await?;
//!
//!     let results = store.query(vec![0.15; 128], 10, None).await?;
//!     println!("Found {} results", results.len());
//!
//!     Ok(())
//! }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Async VecStore for Python (core implementation)
///
/// This provides the async functionality that can be wrapped with PyO3
/// in production using pyo3-asyncio.
pub struct AsyncPyVecStore {
    // Using Arc<RwLock> to allow async access
    data: Arc<RwLock<HashMap<String, (Vec<f32>, serde_json::Value)>>>,
    dimension: usize,
}

impl AsyncPyVecStore {
    /// Create a new async vector store
    pub fn new(dimension: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            dimension,
        }
    }

    /// Insert or update a vector (async)
    pub async fn upsert(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!(
                "Vector dimension {} doesn't match store dimension {}",
                vector.len(),
                self.dimension
            ));
        }

        let mut store = self.data.write().await;
        store.insert(id, (vector, metadata));
        Ok(())
    }

    /// Query for similar vectors (async)
    pub async fn query(
        &self,
        vector: Vec<f32>,
        limit: usize,
        _filter: Option<String>,
    ) -> Result<Vec<AsyncSearchResult>> {
        if vector.len() != self.dimension {
            return Err(anyhow!(
                "Query vector dimension {} doesn't match store dimension {}",
                vector.len(),
                self.dimension
            ));
        }

        let store = self.data.read().await;
        let mut results = Vec::new();

        // Compute cosine similarity for each vector
        for (id, (vec, metadata)) in store.iter() {
            let similarity = cosine_similarity(&vector, vec);
            results.push(AsyncSearchResult {
                id: id.clone(),
                score: similarity,
                metadata: metadata.clone(),
            });
        }

        // Sort by similarity descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(limit);

        Ok(results)
    }

    /// Get a vector by ID (async)
    pub async fn get(&self, id: &str) -> Result<Option<(Vec<f32>, serde_json::Value)>> {
        let store = self.data.read().await;
        Ok(store.get(id).cloned())
    }

    /// Delete a vector by ID (async)
    pub async fn delete(&mut self, id: &str) -> Result<bool> {
        let mut store = self.data.write().await;
        Ok(store.remove(id).is_some())
    }

    /// Batch insert (async)
    pub async fn batch_upsert(
        &mut self,
        items: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> Result<()> {
        for (id, vector, metadata) in &items {
            if vector.len() != self.dimension {
                return Err(anyhow!(
                    "Vector dimension {} doesn't match store dimension {}",
                    vector.len(),
                    self.dimension
                ));
            }
        }

        let mut store = self.data.write().await;
        for (id, vector, metadata) in items {
            store.insert(id, (vector, metadata));
        }

        Ok(())
    }

    /// Count total vectors (async)
    pub async fn count(&self) -> usize {
        let store = self.data.read().await;
        store.len()
    }

    /// List all document IDs (async)
    pub async fn list_ids(&self) -> Vec<String> {
        let store = self.data.read().await;
        store.keys().cloned().collect()
    }

    /// Clear all vectors (async)
    pub async fn clear(&mut self) {
        let mut store = self.data.write().await;
        store.clear();
    }
}

/// Async search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncSearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: serde_json::Value,
}

/// Compute cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&c, &d) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_dimension_validation() {
        let store = AsyncPyVecStore::new(128);
        assert_eq!(store.dimension, 128);
    }

    #[tokio::test]
    async fn test_upsert_and_query() {
        let mut store = AsyncPyVecStore::new(3);

        // Insert a document
        store
            .upsert(
                "doc1".to_string(),
                vec![1.0, 0.0, 0.0],
                json!({"title": "Document 1"}),
            )
            .await
            .unwrap();

        // Query
        let results = store.query(vec![1.0, 0.0, 0.0], 10, None).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
        assert!((results[0].score - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_dimension_mismatch() {
        let mut store = AsyncPyVecStore::new(3);

        let result = store
            .upsert(
                "doc1".to_string(),
                vec![1.0, 0.0], // Wrong dimension
                json!({}),
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_and_delete() {
        let mut store = AsyncPyVecStore::new(2);

        // Insert
        store
            .upsert("doc1".to_string(), vec![0.1, 0.2], json!({"key": "value"}))
            .await
            .unwrap();

        // Get
        let doc = store.get("doc1").await.unwrap();
        assert!(doc.is_some());

        // Delete
        let deleted = store.delete("doc1").await.unwrap();
        assert!(deleted);

        // Verify deleted
        let doc = store.get("doc1").await.unwrap();
        assert!(doc.is_none());
    }

    #[tokio::test]
    async fn test_batch_upsert() {
        let mut store = AsyncPyVecStore::new(2);

        let items = vec![
            ("doc1".to_string(), vec![0.1, 0.2], json!({"idx": 1})),
            ("doc2".to_string(), vec![0.3, 0.4], json!({"idx": 2})),
            ("doc3".to_string(), vec![0.5, 0.6], json!({"idx": 3})),
        ];

        store.batch_upsert(items).await.unwrap();

        let count = store.count().await;
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_list_ids() {
        let mut store = AsyncPyVecStore::new(2);

        store
            .upsert("doc1".to_string(), vec![0.1, 0.2], json!({}))
            .await
            .unwrap();
        store
            .upsert("doc2".to_string(), vec![0.3, 0.4], json!({}))
            .await
            .unwrap();

        let ids = store.list_ids().await;
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"doc1".to_string()));
        assert!(ids.contains(&"doc2".to_string()));
    }

    #[tokio::test]
    async fn test_clear() {
        let mut store = AsyncPyVecStore::new(2);

        store
            .upsert("doc1".to_string(), vec![0.1, 0.2], json!({}))
            .await
            .unwrap();
        store
            .upsert("doc2".to_string(), vec![0.3, 0.4], json!({}))
            .await
            .unwrap();

        assert_eq!(store.count().await, 2);

        store.clear().await;
        assert_eq!(store.count().await, 0);
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let store = Arc::new(AsyncPyVecStore::new(3));

        // Spawn multiple tasks
        let mut handles = vec![];

        for i in 0..10 {
            let store_clone = store.clone();
            let handle = tokio::spawn(async move {
                let mut data = store_clone.data.write().await;
                data.insert(
                    format!("doc{}", i),
                    (vec![i as f32; 3], json!({"index": i})),
                );
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all inserted
        let count = store.count().await;
        assert_eq!(count, 10);
    }

    #[tokio::test]
    async fn test_query_limit() {
        let mut store = AsyncPyVecStore::new(2);

        // Insert 5 documents
        for i in 0..5 {
            store
                .upsert(
                    format!("doc{}", i),
                    vec![i as f32 * 0.1, (i + 1) as f32 * 0.1],
                    json!({"index": i}),
                )
                .await
                .unwrap();
        }

        // Query with limit 3
        let results = store.query(vec![0.0, 0.1], 3, None).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_result_serialization() {
        let result = AsyncSearchResult {
            id: "doc1".to_string(),
            score: 0.95,
            metadata: json!({"key": "value"}),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: AsyncSearchResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "doc1");
        assert!((deserialized.score - 0.95).abs() < 0.001);
    }
}
