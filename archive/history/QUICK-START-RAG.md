# VecStore Quick Start: RAG Applications

**Last Updated**: 2025-10-19

A quick reference guide for building RAG (Retrieval-Augmented Generation) applications with VecStore's new collection and text chunking features.

---

## 🚀 Quick Start (5 Minutes)

### 1. Basic RAG Setup

```rust
use vecstore::{
    VecDatabase,
    text_splitter::{RecursiveCharacterTextSplitter, TextSplitter},
    Query, Metadata,
};

fn main() -> anyhow::Result<()> {
    // 1. Create database with collection
    let mut db = VecDatabase::open("./rag_db")?;
    let mut docs = db.create_collection("documents")?;

    // 2. Setup text splitter
    let splitter = RecursiveCharacterTextSplitter::new(500, 50);

    // 3. Process document
    let document = "Your long document text here...";
    let chunks = splitter.split_text(document)?;

    // 4. Store chunks (use your embedding model here)
    for (i, chunk) in chunks.iter().enumerate() {
        let embedding = your_embedding_model(chunk); // Your model
        let mut meta = Metadata::new();
        meta.insert("text", chunk);
        meta.insert("chunk_id", i);

        docs.upsert(format!("chunk_{}", i), embedding, meta)?;
    }

    // 5. Query
    let question = "What is this about?";
    let q_embedding = your_embedding_model(question);
    let results = docs.query(Query::new(q_embedding).with_k(5))?;

    // 6. Get context for LLM
    for result in results {
        let text = result.metadata.get("text").unwrap();
        println!("Relevant: {}", text);
    }

    Ok(())
}
```

---

## 📚 Core Concepts

### Collections

Organize vectors by domain (documents, users, products, etc.):

```rust
let mut db = VecDatabase::open("./db")?;

// Separate collections for different data types
let mut documents = db.create_collection("documents")?;
let mut users = db.create_collection("users")?;
let mut products = db.create_collection("products")?;

// Each collection is independent
documents.upsert("doc1", doc_embedding, doc_metadata)?;
users.upsert("user1", user_embedding, user_metadata)?;
```

### Text Chunking

Split long documents into embedding-sized chunks:

```rust
// Character-based chunking (most common)
let splitter = RecursiveCharacterTextSplitter::new(500, 50);
let chunks = splitter.split_text(long_document)?;

// Token-based chunking (for LLM limits)
let token_splitter = TokenTextSplitter::new(512, 50);
let chunks = token_splitter.split_text(long_document)?;
```

---

## 🎯 Common Patterns

### Pattern 1: Document Q&A

```rust
// Setup
let mut db = VecDatabase::open("./qa_system")?;
let mut docs = db.create_collection("knowledge_base")?;
let splitter = RecursiveCharacterTextSplitter::new(400, 40);

// Ingest documents
for doc in documents {
    let chunks = splitter.split_text(&doc.content)?;
    for (i, chunk) in chunks.iter().enumerate() {
        let embedding = embed(chunk);
        let mut meta = Metadata::new();
        meta.insert("text", chunk);
        meta.insert("source", &doc.filename);
        meta.insert("chunk", i);

        docs.upsert(format!("{}_{}", doc.id, i), embedding, meta)?;
    }
}

// Answer questions
fn answer_question(question: &str, docs: &Collection) -> Result<String> {
    let q_embedding = embed(question);
    let results = docs.query(Query::new(q_embedding).with_k(5))?;

    let context: String = results.iter()
        .map(|r| r.metadata.get("text").unwrap())
        .collect::<Vec<_>>()
        .join("\n\n");

    Ok(llm_generate(question, &context))
}
```

### Pattern 2: Semantic Search

```rust
// Setup with hybrid search for best results
let mut db = VecDatabase::open("./search_db")?;
let mut articles = db.create_collection("articles")?;

// Index articles with both dense and sparse vectors
for article in articles {
    let chunks = splitter.split_text(&article.content)?;

    for (i, chunk) in chunks.iter().enumerate() {
        let dense = generate_dense_embedding(chunk);
        let sparse = generate_sparse_embedding(chunk); // BM25 style

        let hybrid_vector = combine_dense_sparse(dense, sparse);

        let mut meta = Metadata::new();
        meta.insert("text", chunk);
        meta.insert("title", &article.title);

        articles.upsert(format!("{}_{}", article.id, i), hybrid_vector, meta)?;
    }
}

// Search with hybrid query
let query_dense = generate_dense_embedding(user_query);
let query_sparse = generate_sparse_embedding(user_query);
let hybrid_query = combine_dense_sparse(query_dense, query_sparse);

let results = articles.query(Query::new(hybrid_query).with_k(10))?;
```

### Pattern 3: Multi-Domain RAG

```rust
// Multiple collections for different content types
let mut db = VecDatabase::open("./multi_rag")?;

let mut documentation = db.create_collection("documentation")?;
let mut code_examples = db.create_collection("code_examples")?;
let mut tutorials = db.create_collection("tutorials")?;

// Different chunking strategies per collection
let doc_splitter = RecursiveCharacterTextSplitter::new(600, 60);
let code_splitter = RecursiveCharacterTextSplitter::new(300, 30)
    .with_separators(vec!["\n\n".into(), "\n".into(), ";".into()]);
let tutorial_splitter = RecursiveCharacterTextSplitter::new(800, 80);

// Index each collection separately
index_collection(&mut documentation, docs_data, &doc_splitter)?;
index_collection(&mut code_examples, code_data, &code_splitter)?;
index_collection(&mut tutorials, tutorial_data, &tutorial_splitter)?;

// Query across collections
fn search_all(query: &str, db: &VecDatabase) -> Result<Vec<SearchResult>> {
    let embedding = embed(query);
    let q = Query::new(embedding).with_k(5);

    let mut results = Vec::new();
    results.extend(db.get_collection("documentation")?.unwrap().query(q.clone())?);
    results.extend(db.get_collection("code_examples")?.unwrap().query(q.clone())?);
    results.extend(db.get_collection("tutorials")?.unwrap().query(q)?);

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    Ok(results.into_iter().take(10).collect())
}
```

---

## ⚙️ Configuration Tips

### Chunk Size Selection

```rust
// Small chunks (200-300 chars) - Precise retrieval
let precise = RecursiveCharacterTextSplitter::new(250, 25);

// Medium chunks (400-600 chars) - Balanced (recommended)
let balanced = RecursiveCharacterTextSplitter::new(500, 50);

// Large chunks (800-1200 chars) - More context
let contextual = RecursiveCharacterTextSplitter::new(1000, 100);
```

**Rule of thumb**: Start with 500 char chunks, 50 char overlap.

### Overlap Guidelines

```rust
// Minimal overlap (10%) - Saves space
let minimal = RecursiveCharacterTextSplitter::new(500, 50);

// Standard overlap (20%) - Recommended
let standard = RecursiveCharacterTextSplitter::new(500, 100);

// High overlap (30-40%) - Maximum continuity
let high = RecursiveCharacterTextSplitter::new(500, 200);
```

**Rule of thumb**: Use 10-20% overlap for most cases.

### Custom Separators

```rust
// Sentence-first splitting (good for Q&A)
let sentence_first = RecursiveCharacterTextSplitter::new(500, 50)
    .with_separators(vec![
        ". ".into(),
        "! ".into(),
        "? ".into(),
        "\n\n".into(),
        "\n".into(),
    ]);

// Code-aware splitting
let code_aware = RecursiveCharacterTextSplitter::new(400, 40)
    .with_separators(vec![
        "\n\n".into(),
        "\nfn ".into(),
        "\nimpl ".into(),
        "\n".into(),
        ";".into(),
    ]);

// Markdown-aware splitting
let markdown = RecursiveCharacterTextSplitter::new(600, 60)
    .with_separators(vec![
        "\n## ".into(),
        "\n### ".into(),
        "\n\n".into(),
        "\n".into(),
    ]);
```

---

## 🔧 Advanced Features

### Collection Configuration

```rust
use vecstore::{CollectionConfig, Distance, NamespaceQuotas};

let config = CollectionConfig::default()
    .with_description("Product embeddings for recommendation")
    .with_distance(Distance::Cosine)
    .with_max_vectors(1_000_000)
    .with_max_requests_per_minute(1000);

let mut products = db.create_collection_with_config("products", config)?;
```

### Metadata Filtering

```rust
// Store rich metadata
let mut meta = Metadata::new();
meta.insert("text", chunk);
meta.insert("category", "technology");
meta.insert("date", "2025-10-19");
meta.insert("author", "Alice");

docs.upsert(id, embedding, meta)?;

// Query with filters
let query = Query::new(embedding)
    .with_k(10)
    .with_filter("category = 'technology' AND date > '2025-01-01'");

let results = docs.query(query)?;
```

### Hybrid Search

```rust
use vecstore::{Vector, HybridQueryV2, FusionStrategy};

// Create hybrid vector (dense + sparse)
let dense = vec![0.1, 0.2, 0.3, 0.4];
let sparse_indices = vec![0, 5, 10];
let sparse_values = vec![2.0, 1.5, 1.0];

let hybrid_vec = Vector::hybrid(dense, sparse_indices, sparse_values)?;

// Query with fusion strategy
let query = HybridQueryV2::new(
    query_dense,
    query_sparse_indices,
    query_sparse_values,
)
.with_k(20)
.with_alpha(0.7) // 70% dense, 30% sparse
.with_fusion_strategy(FusionStrategy::WeightedSum);

let results = store.hybrid_query(query)?;
```

---

## 📊 Performance Tips

### Batch Operations

```rust
// Instead of individual upserts
for chunk in chunks {
    docs.upsert(id, embedding, meta)?; // Slow
}

// Use batching (if available)
let batch: Vec<_> = chunks.iter()
    .map(|chunk| (id, embedding, meta))
    .collect();
docs.batch_upsert(batch)?; // Faster
```

### Collection Size Management

```rust
// Check collection size
let stats = docs.stats()?;
println!("Vectors: {}", stats.vector_count);
println!("Active: {}", stats.active_count);

// Cleanup old vectors
if stats.vector_count > 1_000_000 {
    // Delete old vectors or compact
    docs.compact()?;
}
```

---

## 🐛 Troubleshooting

### Problem: Out of Memory

```rust
// Solution: Use Product Quantization
let config = Config::default()
    .with_product_quantization(true);

let store = VecStore::builder("./data")
    .config(config)
    .build()?;
```

### Problem: Slow Queries

```rust
// Solution 1: Reduce k
let query = Query::new(embedding).with_k(5); // Instead of 50

// Solution 2: Add filters early
let query = Query::new(embedding)
    .with_k(10)
    .with_filter("category = 'tech'"); // Narrows search space

// Solution 3: Use HNSW optimization
let config = Config::default()
    .with_hnsw_m(32) // Higher M = faster queries
    .with_hnsw_ef_construction(200);
```

### Problem: Poor Retrieval Quality

```rust
// Solution 1: Adjust chunk size
let splitter = RecursiveCharacterTextSplitter::new(300, 30); // Smaller chunks

// Solution 2: Increase overlap
let splitter = RecursiveCharacterTextSplitter::new(500, 150); // More context

// Solution 3: Use hybrid search
let query = HybridQueryV2::new(dense, sparse_idx, sparse_val)
    .with_fusion_strategy(FusionStrategy::ReciprocalRankFusion);
```

---

## 📚 Examples

Run these to see everything in action:

```bash
# Collection management
cargo run --example collection_demo

# Text chunking for RAG
cargo run --example text_chunking_demo

# Hybrid search
cargo run --example sparse_vectors_demo

# Distance metrics
cargo run --example distance_metrics_demo
```

---

## 🔗 Resources

- **PROGRESS-SUMMARY.md** - Detailed implementation history
- **PHASES-3-4-COMPLETE.md** - Comprehensive feature documentation
- **ULTRATHINK-PHASE2-COMPETITIVE-POSITION.md** - Competitive analysis
- **Source code** - `src/collection.rs`, `src/text_splitter.rs`

---

## 🎯 Next Steps

1. **Start simple**: Use `VecStore::open()` for single-domain apps
2. **Scale up**: Switch to `VecDatabase` when you need multiple collections
3. **Add chunking**: Use `RecursiveCharacterTextSplitter` for documents
4. **Optimize**: Add hybrid search, product quantization as needed
5. **Monitor**: Check stats regularly with `collection.stats()`

---

**VecStore is production-ready for RAG! 🚀**

For questions or issues, see the examples or source code documentation.
