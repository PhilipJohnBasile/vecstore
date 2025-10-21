//! Python bindings for VecStore
//!
//! Provides a Pythonic interface to the high-performance Rust vector database.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use crate::{VecStore, Query, Metadata, VecDatabase, Collection};
use crate::store::Neighbor;
use crate::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
use crate::rag_utils::{ConversationMemory, PromptTemplate};

/// Convert Python dict to Rust Metadata
fn pydict_to_metadata(py_dict: &PyDict) -> PyResult<Metadata> {
    let mut fields = HashMap::new();

    for (key, value) in py_dict.iter() {
        let key_str = key.extract::<String>()?;
        let json_value = python_to_json(value)?;
        fields.insert(key_str, json_value);
    }

    Ok(Metadata { fields })
}

/// Convert Python value to serde_json::Value
fn python_to_json(obj: &PyAny) -> PyResult<serde_json::Value> {
    if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::Value::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            Ok(serde_json::Value::Number(n))
        } else {
            Err(PyValueError::new_err("Invalid float value"))
        }
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else {
        // Try as list or dict
        Ok(serde_json::Value::String(obj.to_string()))
    }
}

/// Convert Rust Metadata to Python dict
fn metadata_to_pydict(py: Python, metadata: &Metadata) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    for (key, value) in &metadata.fields {
        let py_value = json_to_python(py, value)?;
        dict.set_item(key, py_value)?;
    }

    Ok(dict.into())
}

/// Convert serde_json::Value to Python object
fn json_to_python(py: Python, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(n.to_string().to_object(py))
            }
        }
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_python(py, item)?)?;
            }
            Ok(list.into())
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (key, val) in obj {
                dict.set_item(key, json_to_python(py, val)?)?;
            }
            Ok(dict.into())
        }
    }
}

/// A query result from VecStore
#[pyclass]
#[derive(Clone)]
pub struct PyNeighbor {
    #[pyo3(get)]
    pub id: String,

    #[pyo3(get)]
    pub score: f32,

    pub metadata: Metadata,
}

#[pymethods]
impl PyNeighbor {
    /// Get metadata as Python dict
    #[getter]
    fn get_metadata(&self, py: Python) -> PyResult<PyObject> {
        metadata_to_pydict(py, &self.metadata)
    }

    fn __repr__(&self) -> String {
        format!("Neighbor(id='{}', score={:.4})", self.id, self.score)
    }
}

impl From<Neighbor> for PyNeighbor {
    fn from(neighbor: Neighbor) -> Self {
        PyNeighbor {
            id: neighbor.id,
            score: neighbor.score,
            metadata: neighbor.metadata,
        }
    }
}

/// High-performance vector store
///
/// VecStore provides fast similarity search using HNSW indexing.
///
/// Example:
///     >>> store = VecStore.open("./my_db")
///     >>> store.upsert("doc1", [0.1, 0.2, 0.3], {"text": "hello"})
///     >>> results = store.query([0.1, 0.2, 0.3], k=5)
#[pyclass]
pub struct PyVecStore {
    inner: VecStore,
}

#[pymethods]
impl PyVecStore {
    /// Open or create a vector store at the given path
    ///
    /// Args:
    ///     path: Directory path for the vector store
    ///
    /// Returns:
    ///     VecStore instance
    ///
    /// Example:
    ///     >>> store = VecStore.open("./my_db")
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let store = VecStore::open(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to open store: {}", e)))?;
        Ok(PyVecStore { inner: store })
    }

    /// Insert or update a vector with metadata
    ///
    /// Args:
    ///     id: Unique identifier for the vector
    ///     vector: Dense vector (list of floats)
    ///     metadata: Dictionary of metadata
    ///
    /// Example:
    ///     >>> store.upsert("doc1", [0.1, 0.2, 0.3], {"text": "hello", "category": "greeting"})
    fn upsert(&mut self, id: String, vector: Vec<f32>, metadata: &PyDict) -> PyResult<()> {
        let meta = pydict_to_metadata(metadata)?;
        self.inner.upsert(id, vector, meta)
            .map_err(|e| PyValueError::new_err(format!("Upsert failed: {}", e)))?;
        Ok(())
    }

    /// Query for similar vectors
    ///
    /// Args:
    ///     vector: Query vector (list of floats)
    ///     k: Number of results to return
    ///     filter: Optional metadata filter (not yet implemented)
    ///
    /// Returns:
    ///     List of Neighbor objects with id, score, and metadata
    ///
    /// Example:
    ///     >>> results = store.query([0.1, 0.2, 0.3], k=5)
    ///     >>> for r in results:
    ///     ...     print(f"{r.id}: {r.score}")
    #[pyo3(signature = (vector, k, filter=None))]
    fn query(&self, vector: Vec<f32>, k: usize, filter: Option<&PyDict>) -> PyResult<Vec<PyNeighbor>> {
        let query = Query {
            vector,
            k,
            filter: None, // TODO: implement filter conversion
        };

        let results = self.inner.query(query)
            .map_err(|e| PyValueError::new_err(format!("Query failed: {}", e)))?;

        Ok(results.into_iter().map(|n| n.into()).collect())
    }

    /// Delete a vector by ID
    ///
    /// Args:
    ///     id: Vector ID to delete
    ///
    /// Returns:
    ///     True if deleted, False if not found
    fn delete(&mut self, id: &str) -> PyResult<bool> {
        self.inner.delete(id)
            .map_err(|e| PyValueError::new_err(format!("Delete failed: {}", e)))
    }

    /// Get the number of vectors in the store
    ///
    /// Returns:
    ///     Total number of vectors (including soft-deleted)
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Get the number of active vectors
    fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    /// Get the number of deleted vectors
    fn deleted_count(&self) -> usize {
        self.inner.deleted_count()
    }

    /// Save the store to disk
    fn save(&self) -> PyResult<()> {
        self.inner.save()
            .map_err(|e| PyValueError::new_err(format!("Save failed: {}", e)))
    }

    /// Compact the store (remove soft-deleted vectors)
    ///
    /// Returns:
    ///     Number of vectors removed
    fn compact(&mut self) -> PyResult<usize> {
        self.inner.compact()
            .map_err(|e| PyValueError::new_err(format!("Compact failed: {}", e)))
    }

    /// Get vector dimension
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn __repr__(&self) -> String {
        format!("VecStore(vectors={}, dimension={})", self.inner.len(), self.inner.dimension())
    }
}

/// Multi-tenant vector database with collections
///
/// VecDatabase provides isolated namespaces (collections) for organizing vectors.
///
/// Example:
///     >>> db = VecDatabase.open("./my_db")
///     >>> docs = db.create_collection("documents")
///     >>> docs.upsert("doc1", [0.1, 0.2, 0.3], {"text": "hello"})
#[pyclass]
pub struct PyVecDatabase {
    inner: VecDatabase,
}

#[pymethods]
impl PyVecDatabase {
    /// Open or create a vector database
    ///
    /// Args:
    ///     path: Directory path for the database
    ///
    /// Returns:
    ///     VecDatabase instance
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let db = VecDatabase::open(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to open database: {}", e)))?;
        Ok(PyVecDatabase { inner: db })
    }

    /// Create a new collection
    ///
    /// Args:
    ///     name: Collection name
    ///
    /// Returns:
    ///     Collection instance
    fn create_collection(&mut self, name: &str) -> PyResult<PyCollection> {
        let collection = self.inner.create_collection(name)
            .map_err(|e| PyValueError::new_err(format!("Failed to create collection: {}", e)))?;
        Ok(PyCollection { inner: collection })
    }

    /// Get an existing collection
    ///
    /// Args:
    ///     name: Collection name
    ///
    /// Returns:
    ///     Collection instance or None if not found
    fn get_collection(&self, name: &str) -> PyResult<Option<PyCollection>> {
        let collection = self.inner.get_collection(name)
            .map_err(|e| PyValueError::new_err(format!("Failed to get collection: {}", e)))?;
        Ok(collection.map(|c| PyCollection { inner: c }))
    }

    /// List all collections
    ///
    /// Returns:
    ///     List of collection names
    fn list_collections(&self) -> PyResult<Vec<String>> {
        self.inner.list_collections()
            .map_err(|e| PyValueError::new_err(format!("Failed to list collections: {}", e)))
    }

    /// Delete a collection
    ///
    /// Args:
    ///     name: Collection name
    ///
    /// Returns:
    ///     True if deleted, False if not found
    fn delete_collection(&mut self, name: &str) -> PyResult<bool> {
        self.inner.delete_collection(name)
            .map_err(|e| PyValueError::new_err(format!("Failed to delete collection: {}", e)))
    }

    fn __repr__(&self) -> String {
        let collections = self.inner.list_collections().unwrap_or_default();
        format!("VecDatabase(collections={})", collections.len())
    }
}

/// A collection within a VecDatabase
///
/// Collections provide isolated namespaces for vectors with quota management.
#[pyclass]
pub struct PyCollection {
    inner: Collection,
}

#[pymethods]
impl PyCollection {
    /// Insert or update a vector
    fn upsert(&mut self, id: String, vector: Vec<f32>, metadata: &PyDict) -> PyResult<()> {
        let meta = pydict_to_metadata(metadata)?;
        self.inner.upsert(id, vector, meta)
            .map_err(|e| PyValueError::new_err(format!("Upsert failed: {}", e)))?;
        Ok(())
    }

    /// Query for similar vectors
    #[pyo3(signature = (vector, k, filter=None))]
    fn query(&self, vector: Vec<f32>, k: usize, filter: Option<&PyDict>) -> PyResult<Vec<PyNeighbor>> {
        let query = Query {
            vector,
            k,
            filter: None,
        };

        let results = self.inner.query(query)
            .map_err(|e| PyValueError::new_err(format!("Query failed: {}", e)))?;

        Ok(results.into_iter().map(|n| n.into()).collect())
    }

    /// Delete a vector
    fn delete(&mut self, id: &str) -> PyResult<bool> {
        self.inner.delete(id)
            .map_err(|e| PyValueError::new_err(format!("Delete failed: {}", e)))
    }

    /// Get collection statistics
    fn stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = self.inner.stats()
                .map_err(|e| PyValueError::new_err(format!("Failed to get stats: {}", e)))?;

            let dict = PyDict::new(py);
            dict.set_item("vector_count", stats.vector_count)?;
            dict.set_item("active_count", stats.active_count)?;
            dict.set_item("deleted_count", stats.deleted_count)?;
            dict.set_item("quota_utilization", stats.quota_utilization)?;

            Ok(dict.into())
        })
    }
}

/// Text splitter that recursively splits on separators
///
/// Example:
///     >>> splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
///     >>> chunks = splitter.split_text("Long document...")
#[pyclass]
pub struct PyRecursiveCharacterTextSplitter {
    inner: RecursiveCharacterTextSplitter,
}

#[pymethods]
impl PyRecursiveCharacterTextSplitter {
    #[new]
    fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        PyRecursiveCharacterTextSplitter {
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
    fn split_text(&self, text: &str) -> PyResult<Vec<String>> {
        self.inner.split_text(text)
            .map_err(|e| PyValueError::new_err(format!("Split failed: {}", e)))
    }
}

/// Conversation memory with token limit management
///
/// Example:
///     >>> memory = ConversationMemory(max_tokens=2048)
///     >>> memory.add_message("user", "Hello!")
///     >>> memory.add_message("assistant", "Hi there!")
///     >>> print(memory.format_messages())
#[pyclass]
pub struct PyConversationMemory {
    inner: ConversationMemory,
}

#[pymethods]
impl PyConversationMemory {
    #[new]
    fn new(max_tokens: usize) -> Self {
        PyConversationMemory {
            inner: ConversationMemory::new(max_tokens),
        }
    }

    /// Add a message to the conversation
    ///
    /// Args:
    ///     role: Message role (e.g., "user", "assistant", "system")
    ///     content: Message content
    fn add_message(&mut self, role: &str, content: &str) {
        self.inner.add_message(role, content);
    }

    /// Format all messages as a string
    ///
    /// Returns:
    ///     Formatted conversation history
    fn format_messages(&self) -> String {
        self.inner.format_messages()
    }

    /// Clear all messages
    fn clear(&mut self) {
        self.inner.clear();
    }
}

/// Prompt template with variable substitution
///
/// Example:
///     >>> template = PromptTemplate("Hello {name}, you are {age} years old")
///     >>> result = template.format({"name": "Alice", "age": "30"})
#[pyclass]
pub struct PyPromptTemplate {
    inner: PromptTemplate,
}

#[pymethods]
impl PyPromptTemplate {
    #[new]
    fn new(template: String) -> Self {
        PyPromptTemplate {
            inner: PromptTemplate::new(template),
        }
    }

    /// Format the template with variables
    ///
    /// Args:
    ///     variables: Dictionary of variable name -> value
    ///
    /// Returns:
    ///     Formatted string
    fn format(&self, variables: &PyDict) -> PyResult<String> {
        let mut vars = HashMap::new();
        for (key, value) in variables.iter() {
            let key_str = key.extract::<String>()?;
            let val_str = value.to_string();
            vars.insert(key_str, val_str);
        }

        Ok(self.inner.format(&vars))
    }
}

/// Python module initialization
#[pymodule]
fn vecstore(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyVecStore>()?;
    m.add_class::<PyVecDatabase>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PyNeighbor>()?;
    m.add_class::<PyRecursiveCharacterTextSplitter>()?;
    m.add_class::<PyConversationMemory>()?;
    m.add_class::<PyPromptTemplate>()?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
