# VecStore Distance Metrics & Sparse Vectors - Implementation Spec

**Date**: 2025-10-19
**Status**: Design Phase
**Implements**: ULTRATHINK-COMPETITIVE-ANALYSIS.md recommendations

---

## 1. Additional Distance Metrics

### 1.1 Overview

Add Manhattan, Hamming, and Jaccard distance metrics to VecStore, maintaining backwards compatibility.

### 1.2 Distance Metric Definitions

#### Manhattan Distance (L1)
```
d(a, b) = Σ|aᵢ - bᵢ|
```
- **Use case**: When all dimensions are equally important, robust to outliers
- **Range**: [0, ∞), lower is more similar
- **Best for**: Spatial data, city-block distances, feature vectors with independent dimensions

#### Hamming Distance
```
d(a, b) = count(aᵢ ≠ bᵢ)
```
- **Use case**: Binary vectors, categorical data
- **Range**: [0, n], lower is more similar (n = vector dimension)
- **Best for**: Binary embeddings, error detection, categorical features
- **Implementation**: For f32 vectors, threshold at 0.5 to convert to binary

#### Jaccard Distance
```
d(a, b) = 1 - (|A ∩ B| / |A ∪ B|)
```
Where:
- A = set of indices where a > 0
- B = set of indices where b > 0
- **Use case**: Set similarity, sparse vectors
- **Range**: [0, 1], lower is more similar
- **Best for**: Tag vectors, one-hot encoded features, sparse embeddings

### 1.3 Updated Types

**src/store/types.rs**:
```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Distance {
    /// Cosine similarity - measures angle between vectors
    /// Range: [-1, 1], higher is more similar
    /// Best for: Text embeddings, normalized vectors
    Cosine,

    /// Euclidean distance (L2) - measures straight-line distance
    /// Range: [0, ∞), lower is more similar
    /// Best for: Spatial data, unnormalized vectors
    Euclidean,

    /// Dot product - measures alignment and magnitude
    /// Range: (-∞, ∞), higher is more similar
    /// Best for: When magnitude matters, recommendation systems
    DotProduct,

    /// Manhattan distance (L1) - measures city-block distance
    /// Range: [0, ∞), lower is more similar
    /// Best for: Spatial data, robust to outliers
    Manhattan,

    /// Hamming distance - counts differing elements
    /// Range: [0, n], lower is more similar
    /// Best for: Binary vectors, categorical data
    /// Note: Converts f32 vectors to binary at threshold 0.5
    Hamming,

    /// Jaccard distance - measures set dissimilarity
    /// Range: [0, 1], lower is more similar
    /// Best for: Sparse vectors, tag vectors, one-hot encoding
    Jaccard,
}

impl Default for Distance {
    fn default() -> Self {
        Distance::Cosine  // Backwards compatible
    }
}

impl Distance {
    /// Convert distance metric name to enum
    pub fn from_str(s: &str) -> Result<Self, &'static str> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(Distance::Cosine),
            "euclidean" | "l2" => Ok(Distance::Euclidean),
            "dotproduct" | "dot" => Ok(Distance::DotProduct),
            "manhattan" | "l1" => Ok(Distance::Manhattan),
            "hamming" => Ok(Distance::Hamming),
            "jaccard" => Ok(Distance::Jaccard),
            _ => Err("Unknown distance metric"),
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Distance::Cosine => "Cosine",
            Distance::Euclidean => "Euclidean",
            Distance::DotProduct => "DotProduct",
            Distance::Manhattan => "Manhattan",
            Distance::Hamming => "Hamming",
            Distance::Jaccard => "Jaccard",
        }
    }
}
```

### 1.4 SIMD Implementations

**src/simd.rs additions**:

```rust
/// Calculate Manhattan (L1) distance using SIMD acceleration
#[inline]
pub fn manhattan_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return manhattan_distance_avx2(a, b);
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx2"),
        target_feature = "sse2"
    ))]
    unsafe {
        return manhattan_distance_sse2(a, b);
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    manhattan_distance_scalar(a, b)
}

#[inline]
fn manhattan_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

/// Calculate Hamming distance (count of differing elements)
/// For f32 vectors, converts to binary at threshold 0.5
#[inline]
pub fn hamming_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    // For f32 vectors, threshold at 0.5 to convert to binary
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return hamming_distance_avx2(a, b);
    }

    #[cfg(not(any(all(target_arch = "x86_64", target_feature = "avx2"))))]
    hamming_distance_scalar(a, b)
}

#[inline]
fn hamming_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| {
            let x_bit = *x > 0.5;
            let y_bit = *y > 0.5;
            x_bit != y_bit
        })
        .count() as f32
}

/// Calculate Jaccard distance (1 - Jaccard similarity)
#[inline]
pub fn jaccard_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let (intersection, union) = jaccard_counts_simd(a, b);

    if union == 0 {
        return 1.0; // Maximum distance for empty sets
    }

    1.0 - (intersection as f32 / union as f32)
}

#[inline]
fn jaccard_counts_simd(a: &[f32], b: &[f32]) -> (usize, usize) {
    let mut intersection = 0;
    let mut union = 0;

    for i in 0..a.len() {
        let a_nonzero = a[i] > 0.0;
        let b_nonzero = b[i] > 0.0;

        if a_nonzero || b_nonzero {
            union += 1;
            if a_nonzero && b_nonzero {
                intersection += 1;
            }
        }
    }

    (intersection, union)
}
```

### 1.5 VecStore Builder Pattern

**src/store/mod.rs additions**:

```rust
/// Builder for VecStore with customizable options
pub struct VecStoreBuilder {
    path: PathBuf,
    distance: Distance,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
}

impl VecStoreBuilder {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            path: path.into(),
            distance: Distance::Cosine,  // Default
            hnsw_m: 16,
            hnsw_ef_construction: 200,
        }
    }

    /// Set the distance metric
    pub fn distance(mut self, metric: Distance) -> Self {
        self.distance = metric;
        self
    }

    /// Set HNSW M parameter (connections per layer)
    pub fn hnsw_m(mut self, m: usize) -> Self {
        self.hnsw_m = m;
        self
    }

    /// Set HNSW ef_construction parameter
    pub fn hnsw_ef_construction(mut self, ef: usize) -> Self {
        self.hnsw_ef_construction = ef;
        self
    }

    /// Build the VecStore
    pub fn build(self) -> Result<VecStore> {
        VecStore::open_with_config(self.path, Config {
            distance: self.distance,
            hnsw_m: self.hnsw_m,
            hnsw_ef_construction: self.hnsw_ef_construction,
        })
    }
}

impl VecStore {
    /// Create a builder for VecStore
    pub fn builder<P: Into<PathBuf>>(path: P) -> VecStoreBuilder {
        VecStoreBuilder::new(path)
    }

    /// Open VecStore with custom configuration
    pub fn open_with_config<P: Into<PathBuf>>(root: P, config: Config) -> Result<Self> {
        // Implementation that respects the config
        // For now, falls back to existing open() logic
        // TODO: Make HNSW backend respect distance metric
        Self::open(root)
    }
}
```

### 1.6 Backwards Compatibility

**Existing code works unchanged**:
```rust
// Old code - still works
let store = VecStore::open("./data")?;

// New code - opt-in to new metrics
let store = VecStore::builder("./data")
    .distance(Distance::Manhattan)
    .hnsw_m(32)
    .build()?;
```

### 1.7 HNSW Backend Challenge

**Current issue**: `hnsw_rs` crate hardcodes the distance metric at compile time:
```rust
pub struct HnswBackend {
    hnsw: Hnsw<'static, f32, DistCosine>,  // DistCosine is hardcoded
    // ...
}
```

**Solution options**:
1. **Option A**: Make HnswBackend use an enum to dispatch to different HNSW instances (runtime overhead)
2. **Option B**: Fork/wrap hnsw_rs to support dynamic distance metrics
3. **Option C**: Use trait objects (some performance cost)

**Recommendation**: Start with Option A (enum dispatch) for simplicity and profile later.

### 1.8 Testing Plan

```rust
#[test]
fn test_manhattan_distance() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let dist = manhattan_distance_simd(&a, &b);
    assert_relative_eq!(dist, 9.0, epsilon = 1e-6);  // |1-4| + |2-5| + |3-6| = 9
}

#[test]
fn test_hamming_distance() {
    let a = vec![1.0, 0.0, 1.0, 1.0];
    let b = vec![1.0, 1.0, 1.0, 0.0];
    let dist = hamming_distance_simd(&a, &b);
    assert_eq!(dist, 2.0);  // 2 positions differ
}

#[test]
fn test_jaccard_distance() {
    let a = vec![1.0, 1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 1.0, 0.0];
    let dist = jaccard_distance_simd(&a, &b);
    // Intersection = 1, Union = 3, Distance = 1 - 1/3 = 0.666...
    assert_relative_eq!(dist, 0.666666, epsilon = 1e-5);
}

#[test]
fn test_builder_pattern() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = VecStore::builder(temp_dir.path())
        .distance(Distance::Manhattan)
        .hnsw_m(32)
        .build()
        .unwrap();

    // Verify configuration is applied
    assert_eq!(store.distance_metric(), Distance::Manhattan);
}
```

---

## 2. Sparse Vectors & Hybrid Search

### 2.1 Overview

Add support for sparse vectors (BM25-style) and hybrid dense+sparse search.

### 2.2 Vector Type Design

```rust
/// Vector representation supporting dense, sparse, or hybrid
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Vector {
    /// Dense vector (traditional embeddings)
    Dense(Vec<f32>),

    /// Sparse vector (keyword/term vectors)
    /// Stores (index, weight) pairs
    Sparse {
        dimension: usize,
        indices: Vec<usize>,
        values: Vec<f32>,
    },

    /// Hybrid vector (dense + sparse)
    Hybrid {
        dense: Vec<f32>,
        sparse_indices: Vec<usize>,
        sparse_values: Vec<f32>,
    },
}

impl Vector {
    /// Create a dense vector
    pub fn dense(values: Vec<f32>) -> Self {
        Vector::Dense(values)
    }

    /// Create a sparse vector from (index, value) pairs
    pub fn sparse(dimension: usize, indices: Vec<usize>, values: Vec<f32>) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(anyhow!("Sparse vector indices and values must have same length"));
        }
        if indices.iter().any(|&i| i >= dimension) {
            return Err(anyhow!("Sparse vector index out of bounds"));
        }
        Ok(Vector::Sparse { dimension, indices, values })
    }

    /// Create a hybrid vector
    pub fn hybrid(
        dense: Vec<f32>,
        sparse_indices: Vec<usize>,
        sparse_values: Vec<f32>,
    ) -> Result<Self> {
        if sparse_indices.len() != sparse_values.len() {
            return Err(anyhow!("Sparse component indices and values must have same length"));
        }
        Ok(Vector::Hybrid { dense, sparse_indices, sparse_values })
    }

    /// Get the dense component (if any)
    pub fn dense_part(&self) -> Option<&[f32]> {
        match self {
            Vector::Dense(v) => Some(v),
            Vector::Hybrid { dense, .. } => Some(dense),
            Vector::Sparse { .. } => None,
        }
    }

    /// Get the sparse component (if any)
    pub fn sparse_part(&self) -> Option<(&[usize], &[f32])> {
        match self {
            Vector::Sparse { indices, values, .. } => Some((indices, values)),
            Vector::Hybrid { sparse_indices, sparse_values, .. } => {
                Some((sparse_indices, sparse_values))
            }
            Vector::Dense(_) => None,
        }
    }
}
```

### 2.3 BM25 Scoring

```rust
/// BM25 parameters
pub struct BM25Config {
    /// k1 parameter (term frequency saturation)
    pub k1: f32,

    /// b parameter (length normalization)
    pub b: f32,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
        }
    }
}

/// Calculate BM25 score for a sparse vector against a document
pub fn bm25_score(
    query_sparse: &[usize],  // query term indices
    query_weights: &[f32],   // query term weights
    doc_sparse: &[usize],    // document term indices
    doc_weights: &[f32],     // document term weights (term frequencies)
    avg_doc_length: f32,
    config: &BM25Config,
) -> f32 {
    // Build document term map for quick lookup
    let doc_terms: HashMap<usize, f32> = doc_sparse
        .iter()
        .zip(doc_weights.iter())
        .map(|(&idx, &weight)| (idx, weight))
        .collect();

    let doc_length = doc_weights.iter().sum::<f32>();

    let mut score = 0.0;

    for (&term_idx, &query_weight) in query_sparse.iter().zip(query_weights.iter()) {
        if let Some(&term_freq) = doc_terms.get(&term_idx) {
            // BM25 formula
            let numerator = term_freq * (config.k1 + 1.0);
            let denominator = term_freq + config.k1 * (1.0 - config.b + config.b * doc_length / avg_doc_length);

            score += query_weight * (numerator / denominator);
        }
    }

    score
}
```

### 2.4 Hybrid Search Fusion

```rust
/// Hybrid search query
pub struct HybridQueryV2 {
    /// Dense vector component (optional)
    pub dense: Option<Vec<f32>>,

    /// Sparse vector component (optional)
    pub sparse: Option<(Vec<usize>, Vec<f32>)>,

    /// Number of results
    pub k: usize,

    /// Weight for dense component (0.0 - 1.0)
    /// sparse weight = 1.0 - alpha
    pub alpha: f32,

    /// Metadata filter
    pub filter: Option<FilterExpr>,
}

impl HybridQueryV2 {
    /// Pure dense query
    pub fn dense_only(vector: Vec<f32>, k: usize) -> Self {
        Self {
            dense: Some(vector),
            sparse: None,
            k,
            alpha: 1.0,
            filter: None,
        }
    }

    /// Pure sparse query
    pub fn sparse_only(indices: Vec<usize>, values: Vec<f32>, k: usize) -> Self {
        Self {
            dense: None,
            sparse: Some((indices, values)),
            k,
            alpha: 0.0,
            filter: None,
        }
    }

    /// Hybrid query with both dense and sparse
    pub fn hybrid(
        dense: Vec<f32>,
        sparse_indices: Vec<usize>,
        sparse_values: Vec<f32>,
        k: usize,
        alpha: f32,
    ) -> Self {
        Self {
            dense: Some(dense),
            sparse: Some((sparse_indices, sparse_values)),
            k,
            alpha,
            filter: None,
        }
    }
}

/// Combine dense and sparse scores using weighted sum
pub fn fuse_scores(
    dense_scores: HashMap<Id, f32>,
    sparse_scores: HashMap<Id, f32>,
    alpha: f32,
) -> Vec<(Id, f32)> {
    // Normalize scores to [0, 1] range
    let dense_max = dense_scores.values().cloned().fold(0.0, f32::max);
    let sparse_max = sparse_scores.values().cloned().fold(0.0, f32::max);

    let mut all_ids: HashSet<Id> = HashSet::new();
    all_ids.extend(dense_scores.keys().cloned());
    all_ids.extend(sparse_scores.keys().cloned());

    let mut combined: Vec<(Id, f32)> = all_ids
        .into_iter()
        .map(|id| {
            let dense_score = dense_scores.get(&id).copied().unwrap_or(0.0) / dense_max.max(1e-10);
            let sparse_score = sparse_scores.get(&id).copied().unwrap_or(0.0) / sparse_max.max(1e-10);

            let combined_score = alpha * dense_score + (1.0 - alpha) * sparse_score;
            (id, combined_score)
        })
        .collect();

    // Sort by score descending
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    combined
}
```

### 2.5 Backwards Compatibility

**Existing code unchanged**:
```rust
// Old code still works
store.upsert("doc1".into(), vec![0.1, 0.2, 0.3], metadata)?;
```

**New hybrid code**:
```rust
// Store dense vector
store.upsert_vector("doc1".into(), Vector::dense(vec![0.1, 0.2, 0.3]), metadata)?;

// Store sparse vector (e.g., from BM25 tokenization)
let sparse = Vector::sparse(10000, vec![42, 100, 523], vec![2.5, 1.2, 0.8])?;
store.upsert_vector("doc1".into(), sparse, metadata)?;

// Hybrid search
let query = HybridQueryV2::hybrid(
    dense_vec,
    sparse_indices,
    sparse_values,
    10,
    0.7,  // 70% dense, 30% sparse
);
let results = store.hybrid_query_v2(query)?;
```

---

## 3. Collection Abstraction

### 3.1 Overview

Provide a higher-level "collection" API similar to ChromaDB/Qdrant for better ergonomics.

### 3.2 API Design

```rust
/// Database wrapper managing multiple collections
pub struct VecDatabase {
    namespace_manager: NamespaceManager,
}

impl VecDatabase {
    /// Open/create a database at the specified path
    pub fn open<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let manager = NamespaceManager::new(path)?;
        Ok(Self { namespace_manager: manager })
    }

    /// Create a new collection
    pub fn create_collection(&mut self, name: &str) -> Result<Collection> {
        self.create_collection_with_config(name, CollectionConfig::default())
    }

    /// Create a collection with custom configuration
    pub fn create_collection_with_config(
        &mut self,
        name: &str,
        config: CollectionConfig,
    ) -> Result<Collection> {
        // Create namespace for this collection
        self.namespace_manager.create_namespace(
            name,
            name,
            config.quotas,
        )?;

        Ok(Collection {
            name: name.to_string(),
            manager: self.namespace_manager.clone(),
            config,
        })
    }

    /// Get an existing collection
    pub fn get_collection(&self, name: &str) -> Result<Collection> {
        // Verify namespace exists
        if !self.namespace_manager.list_namespaces()?.iter().any(|ns| ns.id == name) {
            return Err(anyhow!("Collection '{}' does not exist", name));
        }

        Ok(Collection {
            name: name.to_string(),
            manager: self.namespace_manager.clone(),
            config: CollectionConfig::default(),
        })
    }

    /// List all collections
    pub fn list_collections(&self) -> Result<Vec<String>> {
        Ok(self.namespace_manager.list_namespaces()?
            .into_iter()
            .map(|ns| ns.id)
            .collect())
    }

    /// Delete a collection
    pub fn delete_collection(&mut self, name: &str) -> Result<()> {
        self.namespace_manager.delete_namespace(name)
    }
}

/// Collection configuration
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    pub distance: Distance,
    pub quotas: Option<NamespaceQuotas>,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            distance: Distance::Cosine,
            quotas: None,
        }
    }
}

/// Collection - a logical grouping of vectors
pub struct Collection {
    name: String,
    manager: NamespaceManager,
    config: CollectionConfig,
}

impl Collection {
    /// Add a single item to the collection
    pub fn add(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Metadata,
    ) -> Result<()> {
        self.manager.upsert(&self.name, id.into(), vector, metadata)
    }

    /// Add multiple items in batch
    pub fn add_batch(&mut self, items: Vec<(String, Vec<f32>, Metadata)>) -> Result<()> {
        let operations: Vec<BatchOperation> = items
            .into_iter()
            .map(|(id, vector, metadata)| BatchOperation::Upsert { id, vector, metadata })
            .collect();

        self.manager.batch_execute(&self.name, operations)?;
        Ok(())
    }

    /// Query the collection
    pub fn query(
        &self,
        vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<Neighbor>> {
        self.manager.query(&self.name, Query {
            vector,
            k,
            filter: None,
        })
    }

    /// Query with metadata filter
    pub fn query_with_filter(
        &self,
        vector: Vec<f32>,
        k: usize,
        filter: FilterExpr,
    ) -> Result<Vec<Neighbor>> {
        self.manager.query(&self.name, Query {
            vector,
            k,
            filter: Some(filter),
        })
    }

    /// Delete an item
    pub fn delete(&mut self, id: &str) -> Result<()> {
        self.manager.delete(&self.name, id)
    }

    /// Get collection statistics
    pub fn stats(&self) -> Result<NamespaceStats> {
        self.manager.get_namespace_stats(&self.name)
    }

    /// Get the collection name
    pub fn name(&self) -> &str {
        &self.name
    }
}
```

### 3.3 Usage Example

```rust
// Create database
let mut db = VecDatabase::open("./my_database")?;

// Create collection
let mut collection = db.create_collection("documents")?;

// Add items
collection.add("doc1", vec![0.1, 0.2, 0.3], metadata)?;

// Batch add
collection.add_batch(vec![
    ("doc2".into(), vec![0.2, 0.3, 0.4], metadata2),
    ("doc3".into(), vec![0.3, 0.4, 0.5], metadata3),
])?;

// Query
let results = collection.query(vec![0.15, 0.25, 0.35], 10)?;

// Get stats
let stats = collection.stats()?;
println!("Collection has {} vectors", stats.vector_count);

// List all collections
let collections = db.list_collections()?;
```

---

## 4. Implementation Order

1. ✅ **Design Phase** (current task)
2. **Week 1**: Distance Metrics
   - [ ] Extend Distance enum
   - [ ] Add SIMD implementations (Manhattan, Hamming, Jaccard)
   - [ ] Add VecStoreBuilder
   - [ ] Tests
3. **Week 2-3**: Sparse Vectors & Hybrid Search
   - [ ] Add Vector enum
   - [ ] Implement BM25 scoring
   - [ ] Implement hybrid fusion
   - [ ] Update VecStore to handle Vector type
   - [ ] Tests
4. **Week 4**: Collection Abstraction
   - [ ] Implement VecDatabase
   - [ ] Implement Collection
   - [ ] Tests
5. **Week 5**: Documentation & Examples
   - [ ] Update all documentation
   - [ ] Create examples
   - [ ] Benchmarks

---

## 5. Backwards Compatibility Checklist

- [x] Existing `VecStore::open()` still works
- [x] Existing `upsert(id, Vec<f32>, metadata)` signature unchanged
- [x] Distance::Cosine remains the default
- [x] All existing tests pass
- [x] No breaking changes to public API

---

## 6. Success Criteria

- [ ] All 3 new distance metrics implemented with SIMD support
- [ ] Builder pattern functional
- [ ] Sparse vector support working
- [ ] Hybrid search with BM25 fusion working
- [ ] Collection abstraction API complete
- [ ] All tests passing (target: 150+ tests)
- [ ] Documentation complete
- [ ] Performance benchmarks meet targets
- [ ] Zero breaking changes
