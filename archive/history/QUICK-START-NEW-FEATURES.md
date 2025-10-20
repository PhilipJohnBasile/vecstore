# VecStore Quick Start - New Features (Weeks 1-4)

Quick reference for using VecStore's new OpenAI embeddings and reranking features.

## Installation

```toml
[dependencies]
vecstore = { version = "1.0", features = ["embeddings", "openai-embeddings"] }
tokio = { version = "1.0", features = ["full"] }
```

## 1. OpenAI Embeddings (Week 3)

### Basic Usage

```rust
use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Set API key
    let api_key = std::env::var("OPENAI_API_KEY")?;

    // Create embedder
    let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small)
        .await?
        .with_rate_limit(500)   // 500 requests/minute
        .with_max_retries(3);    // Retry on failure

    // Single text
    let embedding = embedder.embed_async("Hello world").await?;

    // Batch (up to 2,048 texts)
    let texts = vec!["doc1", "doc2", "doc3"];
    let embeddings = embedder.embed_batch_async(&texts).await?;

    // Estimate cost
    let cost = embedder.estimate_cost(&texts);
    println!("Cost: ${:.6}", cost);

    Ok(())
}
```

### Models Available

| Model | Dimension | Cost/1M tokens | Best For |
|-------|-----------|----------------|----------|
| `TextEmbedding3Small` | 1,536 | $0.02 | Most use cases ⭐ |
| `TextEmbedding3Large` | 3,072 | $0.13 | Maximum accuracy |
| `Ada002` | 1,536 | $0.10 | Legacy |

### Full Documentation

See [`docs/OPENAI-EMBEDDINGS.md`](docs/OPENAI-EMBEDDINGS.md)

### Examples

- `examples/openai_basic.rs` - Basic usage
- `examples/openai_rag.rs` - Complete RAG pipeline
- `examples/openai_cost_tracking.rs` - Cost analysis

---

## 2. Reranking (Week 4)

### Strategy 1: MMR (Diversity)

```rust
use vecstore::reranking::{Reranker, MMRReranker};

// Create MMR reranker (lambda = 0.7 means 70% relevance, 30% diversity)
let reranker = MMRReranker::new(0.7);

// Get initial results
let results = store.query(&query_emb, 100, None)?;

// Rerank for diversity
let reranked = reranker.rerank("query text", results, 10)?;
```

### Strategy 2: Score-Based (Custom Logic)

```rust
use vecstore::reranking::ScoreReranker;

// Custom scoring function
let reranker = ScoreReranker::new(|neighbor| {
    let base = neighbor.score;

    // Boost recent documents
    let recency = neighbor.metadata.fields
        .get("timestamp")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32 * 0.1;

    base + recency
});

let reranked = reranker.rerank("query", results, 10)?;
```

### Strategy 3: Cross-Encoder (Best Quality)

```rust
use vecstore::reranking::cross_encoder::{CrossEncoderReranker, CrossEncoderModel};

// Load ONNX model (requires model files)
let reranker = CrossEncoderReranker::from_pretrained(
    CrossEncoderModel::MiniLML6V2
)?;

// Semantic reranking
let reranked = reranker.rerank("query", results, 10)?;
```

### Reranking Strategies Comparison

| Strategy | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| MMR | Fast (10ms) | Good | Diversity |
| Score-Based | Fast (2ms) | Custom | Business logic |
| Cross-Encoder | Slow (200ms) | Best | RAG/QA ⭐ |

### Full Documentation

See [`docs/RERANKING.md`](docs/RERANKING.md)

### Example

- `examples/reranking_demo.rs` - All strategies

---

## Complete RAG Pipeline

Combine everything for production RAG:

```rust
use vecstore::VecStore;
use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};
use vecstore::reranking::cross_encoder::{CrossEncoderReranker, CrossEncoderModel};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Setup
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small)
        .await?;
    let reranker = CrossEncoderReranker::from_pretrained(
        CrossEncoderModel::MiniLML6V2
    )?;
    let mut store = VecStore::open("./knowledge_base")?;

    // 2. Index documents
    let docs = vec!["doc1", "doc2", "doc3"];
    let embeddings = embedder.embed_batch_async(&docs).await?;

    for (i, (doc, emb)) in docs.iter().zip(embeddings.iter()).enumerate() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("text".to_string(), serde_json::json!(doc));
        store.upsert(&format!("doc{}", i), emb, metadata)?;
    }

    // 3. Query with reranking
    let query = "your question";
    let query_emb = embedder.embed_async(query).await?;

    // Stage 1: Fast vector search (100 candidates)
    let candidates = store.query(&query_emb, 100, None)?;

    // Stage 2: Semantic reranking (top 5)
    let final_results = reranker.rerank(query, candidates, 5)?;

    // 4. Use results
    for result in final_results {
        let text = result.metadata.get("text").unwrap();
        println!("Score: {:.4} - {}", result.score, text);
    }

    Ok(())
}
```

---

## Python Usage (Week 1-2)

```python
import vecstore_py

# Create store
store = vecstore_py.VecStore("./my_db")

# Add vectors
metadata = {"text": "document content"}
store.upsert("doc1", embedding, metadata)

# Query
results = store.query(query_embedding, k=10)
for result in results:
    print(f"Score: {result.score}, Text: {result.metadata['text']}")
```

See `python/examples/` for more examples.

---

## Running Examples

```bash
# OpenAI examples (requires OPENAI_API_KEY)
export OPENAI_API_KEY='sk-...'
cargo run --example openai_basic --features "embeddings,openai-embeddings"
cargo run --example openai_rag --features "embeddings,openai-embeddings"
cargo run --example openai_cost_tracking --features "embeddings,openai-embeddings"

# Reranking example
cargo run --example reranking_demo --features embeddings

# Python examples
cd python
python examples/basic_rag.py
```

---

## Running Tests

```bash
# OpenAI tests
cargo test --features "embeddings,openai-embeddings" --test openai_embeddings

# Reranking tests
cargo test --features embeddings --lib reranking
cargo test --features embeddings --test cross_encoder_reranking

# Python tests
cd python && pytest
```

---

## Documentation

- **OpenAI Embeddings:** [`docs/OPENAI-EMBEDDINGS.md`](docs/OPENAI-EMBEDDINGS.md)
- **Reranking:** [`docs/RERANKING.md`](docs/RERANKING.md)
- **Completion Report:** [`WEEKS-1-4-COMPLETE.md`](WEEKS-1-4-COMPLETE.md)

---

## Cost Estimation (OpenAI)

```rust
// Before embedding
let texts: Vec<&str> = load_documents();
let estimated_cost = embedder.estimate_cost(&texts);

if estimated_cost > budget {
    println!("Warning: ${:.2} exceeds ${:.2} budget", estimated_cost, budget);
    // Proceed with caution or reduce batch size
}
```

**Typical Costs (text-embedding-3-small):**
- 1,000 docs: $0.0025
- 100,000 docs: $0.50
- 1,000,000 docs: $5.00

---

## Tips

### OpenAI
✅ Always use batch processing
✅ Estimate costs before large jobs
✅ Use text-embedding-3-small for most cases
✅ Cache embeddings to avoid re-embedding

### Reranking
✅ Retrieve more (100), rerank to fewer (10)
✅ Use MMR for diversity
✅ Use Cross-Encoder for quality
✅ Combine strategies in multi-stage pipeline

### Performance
✅ Use async/await for concurrent operations
✅ Batch API calls when possible
✅ Cache models (load once, reuse)
✅ Monitor rate limits

---

## Troubleshooting

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY='sk-...'
```

### "Rate limit exceeded"
```rust
let embedder = embedder.with_rate_limit(100);  // Reduce from 500
```

### "Model file not found" (Cross-Encoder)
Download model using `optimum-cli` - see [`docs/RERANKING.md`](docs/RERANKING.md)

---

## What's New

**Week 1-2:** Python bindings validated (74 tests passing)
**Week 3:** OpenAI embeddings (3 models, cost tracking)
**Week 4:** Advanced reranking (5 strategies, ONNX cross-encoders)

**Total:** 13,341+ lines, 120 tests, 7 examples, 1,543 lines of docs

---

**Ready to build production RAG applications!** 🚀
