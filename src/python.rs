// Python bindings for vecstore using PyO3
//
// This module provides Python-friendly wrappers around the Rust API.

use crate::collection::{Collection, VecDatabase};
use crate::store::{parse_filter, FilterExpr, HybridQuery, Metadata, Query, VecStore};
use crate::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::path::PathBuf;

/// Python wrapper for VecStore
#[pyclass(name = "VecStore")]
pub struct PyVecStore {
    inner: VecStore,
}

/// Python wrapper for Query
#[pyclass(name = "Query")]
#[derive(Clone)]
pub struct PyQuery {
    #[pyo3(get, set)]
    pub vector: Vec<f32>,
    #[pyo3(get, set)]
    pub k: usize,
    #[pyo3(get, set)]
    pub filter: Option<String>,
}

#[pymethods]
impl PyQuery {
    #[new]
    #[pyo3(signature = (vector, k, filter=None))]
    fn new(vector: Vec<f32>, k: usize, filter: Option<String>) -> Self {
        Self { vector, k, filter }
    }
}

/// Python wrapper for HybridQuery
#[pyclass(name = "HybridQuery")]
#[derive(Clone)]
pub struct PyHybridQuery {
    #[pyo3(get, set)]
    pub vector: Vec<f32>,
    #[pyo3(get, set)]
    pub keywords: String,
    #[pyo3(get, set)]
    pub k: usize,
    #[pyo3(get, set)]
    pub filter: Option<String>,
    #[pyo3(get, set)]
    pub alpha: f32,
}

#[pymethods]
impl PyHybridQuery {
    #[new]
    #[pyo3(signature = (vector, keywords, k, alpha, filter=None))]
    fn new(
        vector: Vec<f32>,
        keywords: String,
        k: usize,
        alpha: f32,
        filter: Option<String>,
    ) -> Self {
        Self {
            vector,
            keywords,
            k,
            filter,
            alpha,
        }
    }
}

/// Python wrapper for search results
#[pyclass(name = "SearchResult")]
pub struct PySearchResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub score: f32,
    metadata_dict: Py<PyDict>,
}

#[pymethods]
impl PySearchResult {
    #[getter]
    fn metadata(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.metadata_dict.clone_ref(py).into())
    }

    fn __repr__(&self) -> String {
        format!("SearchResult(id='{}', score={:.3})", self.id, self.score)
    }
}

// Helper function to convert Rust Metadata to Python dict
fn metadata_to_pydict(py: Python, metadata: &Metadata) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    for (key, value) in &metadata.fields {
        match value {
            serde_json::Value::String(s) => {
                dict.set_item(key, s.as_str())?;
            }
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    dict.set_item(key, i)?;
                } else if let Some(f) = n.as_f64() {
                    dict.set_item(key, f)?;
                }
            }
            serde_json::Value::Bool(b) => {
                dict.set_item(key, b)?;
            }
            serde_json::Value::Null => {
                dict.set_item(key, py.None())?;
            }
            _ => {} // Skip complex types
        }
    }
    Ok(dict.unbind())
}

// Helper function to convert Python dict to Rust Metadata
fn pydict_to_metadata(dict: &Bound<PyDict>) -> PyResult<Metadata> {
    let mut fields = HashMap::new();

    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;

        let json_value = if let Ok(s) = value.extract::<String>() {
            serde_json::Value::String(s)
        } else if let Ok(i) = value.extract::<i64>() {
            serde_json::Value::Number(i.into())
        } else if let Ok(f) = value.extract::<f64>() {
            serde_json::json!(f)
        } else if let Ok(b) = value.extract::<bool>() {
            serde_json::Value::Bool(b)
        } else if value.is_none() {
            serde_json::Value::Null
        } else {
            // Try to convert to string as fallback
            serde_json::Value::String(value.str()?.to_string())
        };

        fields.insert(key_str, json_value);
    }

    Ok(Metadata { fields })
}

#[pymethods]
impl PyVecStore {
    /// Create or open a vector store at the given path
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let store = VecStore::open(PathBuf::from(path))
            .map_err(|e| PyValueError::new_err(format!("Failed to open store: {}", e)))?;
        Ok(Self { inner: store })
    }

    /// Insert or update a vector with metadata
    ///
    /// Args:
    ///     id: Unique identifier for the vector
    ///     vector: List of floats representing the embedding
    ///     metadata: Dictionary of metadata (supports str, int, float, bool, None)
    ///
    /// Example:
    ///     >>> store.upsert("doc1", [0.1, 0.2, 0.3], {"title": "Document 1"})
    fn upsert(&mut self, id: String, vector: Vec<f32>, metadata: &Bound<PyDict>) -> PyResult<()> {
        let meta = pydict_to_metadata(metadata)?;
        self.inner
            .upsert(id, vector, meta)
            .map_err(|e| PyValueError::new_err(format!("Upsert failed: {}", e)))
    }

    /// Remove a vector by ID
    fn remove(&mut self, id: String) -> PyResult<()> {
        self.inner
            .remove(&id)
            .map_err(|e| PyValueError::new_err(format!("Remove failed: {}", e)))
    }

    /// Search for similar vectors
    ///
    /// Args:
    ///     vector: Query vector (list of floats)
    ///     k: Number of results to return
    ///     filter: Optional SQL-like filter string (e.g., "category = 'tech'")
    ///
    /// Returns:
    ///     List of SearchResult objects
    ///
    /// Example:
    ///     >>> results = store.query([0.1, 0.2, 0.3], k=10, filter="category = 'tech'")
    ///     >>> for result in results:
    ///     ...     print(f"{result.id}: {result.score}")
    #[pyo3(signature = (vector, k, filter=None))]
    fn query(
        &self,
        py: Python,
        vector: Vec<f32>,
        k: usize,
        filter: Option<String>,
    ) -> PyResult<Vec<PySearchResult>> {
        let filter_expr: Option<FilterExpr> = if let Some(filter_str) = filter {
            Some(
                parse_filter(&filter_str)
                    .map_err(|e| PyValueError::new_err(format!("Filter parse error: {}", e)))?,
            )
        } else {
            None
        };

        let query = Query {
            vector,
            k,
            filter: filter_expr,
        };

        let results = self
            .inner
            .query(query)
            .map_err(|e| PyValueError::new_err(format!("Query failed: {}", e)))?;

        results
            .into_iter()
            .map(|neighbor| {
                Ok(PySearchResult {
                    id: neighbor.id,
                    score: neighbor.score,
                    metadata_dict: metadata_to_pydict(py, &neighbor.metadata)?,
                })
            })
            .collect()
    }

    /// Hybrid search combining vector similarity and keyword search
    ///
    /// Args:
    ///     vector: Query vector (list of floats)
    ///     keywords: Search keywords (string)
    ///     k: Number of results to return
    ///     alpha: Weight for vector vs keyword (0.0 = pure keyword, 1.0 = pure vector)
    ///     filter: Optional SQL-like filter string
    ///
    /// Returns:
    ///     List of SearchResult objects
    ///
    /// Example:
    ///     >>> results = store.hybrid_query(
    ///     ...     vector=[0.1, 0.2, 0.3],
    ///     ...     keywords="machine learning",
    ///     ...     k=10,
    ///     ...     alpha=0.7
    ///     ... )
    #[pyo3(signature = (vector, keywords, k, alpha, filter=None))]
    fn hybrid_query(
        &self,
        py: Python,
        vector: Vec<f32>,
        keywords: String,
        k: usize,
        alpha: f32,
        filter: Option<String>,
    ) -> PyResult<Vec<PySearchResult>> {
        let filter_expr: Option<FilterExpr> = if let Some(filter_str) = filter {
            Some(
                parse_filter(&filter_str)
                    .map_err(|e| PyValueError::new_err(format!("Filter parse error: {}", e)))?,
            )
        } else {
            None
        };

        let query = HybridQuery {
            vector,
            keywords,
            k,
            filter: filter_expr,
            alpha,
        };

        let results = self
            .inner
            .hybrid_query(query)
            .map_err(|e| PyValueError::new_err(format!("Hybrid query failed: {}", e)))?;

        results
            .into_iter()
            .map(|neighbor| {
                Ok(PySearchResult {
                    id: neighbor.id,
                    score: neighbor.score,
                    metadata_dict: metadata_to_pydict(py, &neighbor.metadata)?,
                })
            })
            .collect()
    }

    /// Index text for keyword search (required for hybrid search)
    ///
    /// Args:
    ///     id: Vector ID
    ///     text: Text content to index
    fn index_text(&mut self, id: String, text: String) -> PyResult<()> {
        self.inner
            .index_text(&id, text)
            .map_err(|e| PyValueError::new_err(format!("Text indexing failed: {}", e)))
    }

    /// Save the store to disk
    fn save(&self) -> PyResult<()> {
        self.inner
            .save()
            .map_err(|e| PyValueError::new_err(format!("Save failed: {}", e)))
    }

    /// Get the number of vectors in the store
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the store is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Create a named snapshot
    fn create_snapshot(&self, name: String) -> PyResult<()> {
        self.inner
            .create_snapshot(&name)
            .map_err(|e| PyValueError::new_err(format!("Snapshot creation failed: {}", e)))
    }

    /// Restore from a named snapshot
    fn restore_snapshot(&mut self, name: String) -> PyResult<()> {
        self.inner
            .restore_snapshot(&name)
            .map_err(|e| PyValueError::new_err(format!("Snapshot restore failed: {}", e)))
    }

    /// List all available snapshots
    fn list_snapshots(&self, py: Python) -> PyResult<Py<PyList>> {
        let snapshots = self
            .inner
            .list_snapshots()
            .map_err(|e| PyValueError::new_err(format!("Failed to list snapshots: {}", e)))?;

        let py_list = PyList::new_bound(py, &snapshots);
        Ok(py_list.unbind())
    }

    /// Optimize the index by removing ghost entries from deletions
    fn optimize(&mut self) -> PyResult<usize> {
        self.inner
            .optimize()
            .map_err(|e| PyValueError::new_err(format!("Optimization failed: {}", e)))
    }

    fn __repr__(&self) -> String {
        format!("VecStore(vectors={})", self.inner.len())
    }

    fn __len__(&self) -> usize {
        self.len()
    }
}

/// Python wrapper for VecDatabase (multi-collection database)
#[pyclass(name = "VecDatabase")]
pub struct PyVecDatabase {
    inner: VecDatabase,
}

#[pymethods]
impl PyVecDatabase {
    /// Create or open a database at the specified path
    ///
    /// Example:
    ///     >>> db = VecDatabase("./my_db")
    ///     >>> collections = db.list_collections()
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let db = VecDatabase::open(PathBuf::from(path))
            .map_err(|e| PyValueError::new_err(format!("Failed to open database: {}", e)))?;
        Ok(Self { inner: db })
    }

    /// Create a new collection
    ///
    /// Args:
    ///     name: Collection name (must be unique)
    ///
    /// Returns:
    ///     Collection object
    ///
    /// Example:
    ///     >>> db = VecDatabase("./my_db")
    ///     >>> collection = db.create_collection("documents")
    fn create_collection(&mut self, name: String) -> PyResult<PyCollection> {
        let collection = self
            .inner
            .create_collection(&name)
            .map_err(|e| PyValueError::new_err(format!("Failed to create collection: {}", e)))?;

        Ok(PyCollection { inner: collection })
    }

    /// Get an existing collection
    ///
    /// Args:
    ///     name: Collection name
    ///
    /// Returns:
    ///     Collection object or None if not found
    ///
    /// Example:
    ///     >>> db = VecDatabase("./my_db")
    ///     >>> if collection := db.get_collection("documents"):
    ///     ...     print(f"Found collection: {collection.name()}")
    fn get_collection(&self, name: String) -> PyResult<Option<PyCollection>> {
        match self
            .inner
            .get_collection(&name)
            .map_err(|e| PyValueError::new_err(format!("Failed to get collection: {}", e)))?
        {
            Some(collection) => Ok(Some(PyCollection { inner: collection })),
            None => Ok(None),
        }
    }

    /// List all collections in the database
    ///
    /// Returns:
    ///     List of collection names
    ///
    /// Example:
    ///     >>> db = VecDatabase("./my_db")
    ///     >>> for name in db.list_collections():
    ///     ...     print(f"Collection: {name}")
    fn list_collections(&self, py: Python) -> PyResult<Py<PyList>> {
        let collections = self
            .inner
            .list_collections()
            .map_err(|e| PyValueError::new_err(format!("Failed to list collections: {}", e)))?;

        let py_list = PyList::new_bound(py, &collections);
        Ok(py_list.unbind())
    }

    /// Delete a collection and all its data
    ///
    /// Args:
    ///     name: Collection name to delete
    ///
    /// Example:
    ///     >>> db = VecDatabase("./my_db")
    ///     >>> db.delete_collection("old_documents")
    fn delete_collection(&mut self, name: String) -> PyResult<()> {
        self.inner
            .delete_collection(&name)
            .map_err(|e| PyValueError::new_err(format!("Failed to delete collection: {}", e)))
    }

    fn __repr__(&self) -> PyResult<String> {
        let collections = self
            .inner
            .list_collections()
            .map_err(|e| PyValueError::new_err(format!("Failed to list collections: {}", e)))?;
        Ok(format!("VecDatabase(collections={})", collections.len()))
    }
}

/// Python wrapper for Collection
#[pyclass(name = "Collection")]
pub struct PyCollection {
    inner: Collection,
}

#[pymethods]
impl PyCollection {
    /// Get the collection name
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Insert or update a vector with metadata
    ///
    /// Args:
    ///     id: Unique identifier for the vector
    ///     vector: List of floats representing the embedding
    ///     metadata: Dictionary of metadata
    ///
    /// Example:
    ///     >>> collection.upsert("doc1", [0.1, 0.2, 0.3], {"title": "Document 1"})
    fn upsert(&mut self, id: String, vector: Vec<f32>, metadata: &Bound<PyDict>) -> PyResult<()> {
        let meta = pydict_to_metadata(metadata)?;
        self.inner
            .upsert(id, vector, meta)
            .map_err(|e| PyValueError::new_err(format!("Upsert failed: {}", e)))
    }

    /// Search for similar vectors
    ///
    /// Args:
    ///     vector: Query vector (list of floats)
    ///     k: Number of results to return
    ///     filter: Optional SQL-like filter string
    ///
    /// Returns:
    ///     List of SearchResult objects
    ///
    /// Example:
    ///     >>> results = collection.query([0.1, 0.2, 0.3], k=10)
    ///     >>> for result in results:
    ///     ...     print(f"{result.id}: {result.score}")
    #[pyo3(signature = (vector, k, filter=None))]
    fn query(
        &self,
        py: Python,
        vector: Vec<f32>,
        k: usize,
        filter: Option<String>,
    ) -> PyResult<Vec<PySearchResult>> {
        let filter_expr: Option<FilterExpr> = if let Some(filter_str) = filter {
            Some(
                parse_filter(&filter_str)
                    .map_err(|e| PyValueError::new_err(format!("Filter parse error: {}", e)))?,
            )
        } else {
            None
        };

        let query = Query {
            vector,
            k,
            filter: filter_expr,
        };

        let results = self
            .inner
            .query(query)
            .map_err(|e| PyValueError::new_err(format!("Query failed: {}", e)))?;

        results
            .into_iter()
            .map(|neighbor| {
                Ok(PySearchResult {
                    id: neighbor.id,
                    score: neighbor.score,
                    metadata_dict: metadata_to_pydict(py, &neighbor.metadata)?,
                })
            })
            .collect()
    }

    /// Delete a vector by ID
    ///
    /// Args:
    ///     id: Vector ID to delete
    ///
    /// Example:
    ///     >>> collection.delete("doc1")
    fn delete(&mut self, id: String) -> PyResult<()> {
        self.inner
            .delete(&id)
            .map_err(|e| PyValueError::new_err(format!("Delete failed: {}", e)))
    }

    /// Get the number of vectors in the collection
    fn count(&self) -> PyResult<usize> {
        self.inner
            .count()
            .map_err(|e| PyValueError::new_err(format!("Count failed: {}", e)))
    }

    /// Get collection statistics
    ///
    /// Returns:
    ///     Dictionary with statistics (vector_count, active_count, deleted_count, etc.)
    fn stats(&self, py: Python) -> PyResult<PyObject> {
        let stats = self
            .inner
            .stats()
            .map_err(|e| PyValueError::new_err(format!("Stats failed: {}", e)))?;

        let dict = PyDict::new_bound(py);
        dict.set_item("vector_count", stats.vector_count)?;
        dict.set_item("active_count", stats.active_count)?;
        dict.set_item("deleted_count", stats.deleted_count)?;
        dict.set_item("dimension", stats.dimension)?;
        dict.set_item("quota_utilization", stats.quota_utilization)?;

        Ok(dict.into())
    }

    fn __repr__(&self) -> PyResult<String> {
        let count = self
            .inner
            .count()
            .map_err(|e| PyValueError::new_err(format!("Count failed: {}", e)))?;
        Ok(format!(
            "Collection(name='{}', vectors={})",
            self.inner.name(),
            count
        ))
    }
}

/// Python wrapper for RecursiveCharacterTextSplitter
#[pyclass(name = "RecursiveCharacterTextSplitter")]
pub struct PyRecursiveCharacterTextSplitter {
    inner: RecursiveCharacterTextSplitter,
}

#[pymethods]
impl PyRecursiveCharacterTextSplitter {
    /// Create a new recursive character text splitter
    ///
    /// Args:
    ///     chunk_size: Maximum characters per chunk
    ///     chunk_overlap: Characters to overlap between chunks
    ///
    /// Example:
    ///     >>> splitter = RecursiveCharacterTextSplitter(500, 50)
    ///     >>> chunks = splitter.split_text("Long document text...")
    #[new]
    fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            inner: RecursiveCharacterTextSplitter::new(chunk_size, chunk_overlap),
        }
    }

    /// Split text into chunks
    ///
    /// Args:
    ///     text: Text to split
    ///
    /// Returns:
    ///     List of text chunks
    ///
    /// Example:
    ///     >>> splitter = RecursiveCharacterTextSplitter(500, 50)
    ///     >>> chunks = splitter.split_text("Very long document...")
    ///     >>> for i, chunk in enumerate(chunks):
    ///     ...     print(f"Chunk {i}: {len(chunk)} chars")
    fn split_text(&self, py: Python, text: String) -> PyResult<Py<PyList>> {
        let chunks = self
            .inner
            .split_text(&text)
            .map_err(|e| PyValueError::new_err(format!("Split failed: {}", e)))?;

        let py_list = PyList::new_bound(py, &chunks);
        Ok(py_list.unbind())
    }

    fn __repr__(&self) -> String {
        "RecursiveCharacterTextSplitter()".to_string()
    }
}

/// vecstore - Lightweight vector database for RAG applications
///
/// This module provides Python bindings for the vecstore library,
/// offering fast vector similarity search with HNSW indexing,
/// hybrid search (vector + keyword), and metadata filtering.
#[pymodule]
fn vecstore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVecStore>()?;
    m.add_class::<PyQuery>()?;
    m.add_class::<PyHybridQuery>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyVecDatabase>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PyRecursiveCharacterTextSplitter>()?;
    Ok(())
}
