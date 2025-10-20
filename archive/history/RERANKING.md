# Reranking Guide

Complete guide to using reranking in VecStore to improve search quality for RAG applications.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reranking Strategies](#reranking-strategies)
- [Cross-Encoder Models](#cross-encoder-models)
- [Best Practices](#best-practices)
- [Performance Tuning](#performance-tuning)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Overview

### What is Reranking?

Reranking is a post-processing step that refines initial search results using more sophisticated models or algorithms. It's a critical component in production RAG (Retrieval-Augmented Generation) systems.

**Two-Stage Retrieval:**
```
Stage 1: Fast Vector Search (bi-encoder)
  ↓ Retrieve 100 candidates quickly
Stage 2: Reranking (cross-encoder or algorithm)
  ↓ Rerank to top 10 with better relevance
Result: High-quality, relevant documents
```

### Why Use Reranking?

✅ **Improved Accuracy** - Better relevance than embedding similarity alone
✅ **Diversity** - Avoid redundant results
✅ **Custom Logic** - Apply business rules and domain knowledge
✅ **Quality-Speed Tradeoff** - Fast initial retrieval + accurate final ranking

### When to Use Reranking

| Use Case | Recommended Strategy |
|----------|---------------------|
| Question Answering | Cross-Encoder (ONNX) |
| Document Discovery | MMR (diversity) |
| News/Recent Content | Score-Based (recency boost) |
| E-commerce Search | Score-Based (popularity + relevance) |
| Chat/Conversation | Cross-Encoder or MMR |

---

## Installation

Reranking requires the `embeddings` feature:

```toml
[dependencies]
vecstore = { version = "1.0", features = ["embeddings"] }
```

For cross-encoder models, you'll also need to download ONNX models (see [Cross-Encoder Models](#cross-encoder-models)).

---

## Quick Start

### 1. MMR Reranking (Diversity)

```rust
use vecstore::reranking::{Reranker, MMRReranker};

// Create MMR reranker (0.7 = 70% relevance, 30% diversity)
let reranker = MMRReranker::new(0.7);

// Get initial search results
let results = store.query(&query_emb, 100, None)?;

// Rerank to top 10 with diversity
let reranked = reranker.rerank("query text", results, 10)?;
```

### 2. Score-Based Reranking (Custom Logic)

```rust
use vecstore::reranking::{Reranker, ScoreReranker};

// Boost recent and long documents
let reranker = ScoreReranker::new(|neighbor| {
    let base_score = neighbor.score;

    // Add recency boost
    let recency = neighbor.metadata.fields
        .get("timestamp")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32 * 0.1;

    // Add length boost (comprehensive content)
    let length_boost = (neighbor.metadata.fields
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .len() as f32 / 1000.0)
        .min(0.2);

    base_score + recency + length_boost
});

let reranked = reranker.rerank("query", results, 10)?;
```

### 3. Cross-Encoder Reranking (Best Quality)

```rust
use vecstore::reranking::cross_encoder::{CrossEncoderReranker, CrossEncoderModel};

// Load ONNX cross-encoder model
let reranker = CrossEncoderReranker::from_pretrained(
    CrossEncoderModel::MiniLML6V2
)?;

// Rerank with semantic understanding
let reranked = reranker.rerank("query", results, 10)?;
```

---

## Reranking Strategies

VecStore provides multiple reranking strategies for different use cases.

### 1. Identity Reranker (Baseline)

**Use Case:** Baseline comparison, debugging
**Speed:** Instant
**Quality:** No change from original

```rust
use vecstore::reranking::IdentityReranker;

let reranker = IdentityReranker;
let results = reranker.rerank("query", results, 10)?;
// Returns original results unchanged
```

### 2. MMR (Maximal Marginal Relevance)

**Use Case:** Diverse result sets, avoiding redundancy
**Speed:** Fast (~5-10ms for 100 results)
**Quality:** Balances relevance and diversity

```rust
use vecstore::reranking::MMRReranker;

// Lambda controls tradeoff
let reranker = MMRReranker::new(0.7);
//   1.0 = pure relevance (no diversity)
//   0.0 = pure diversity (no relevance)
//   0.7 = balanced (recommended)
```

**Algorithm:**
1. Select most relevant document
2. For remaining selections, maximize: `λ × relevance - (1-λ) × similarity_to_selected`
3. Iteratively build diverse result set

**Best For:**
- Search results with many similar documents
- News/article recommendations
- Document browsing interfaces

### 3. Score-Based Reranker

**Use Case:** Custom business logic, domain-specific ranking
**Speed:** Very fast (~1-2ms)
**Quality:** Depends on scoring function

```rust
use vecstore::reranking::ScoreReranker;

let reranker = ScoreReranker::new(|neighbor| {
    // Your custom scoring logic
    let score = neighbor.score;

    // Example: Boost authoritative sources
    let authority_boost = if neighbor.metadata.fields
        .get("source")
        .and_then(|v| v.as_str())
        .map(|s| s == "official_docs")
        .unwrap_or(false)
    {
        0.2
    } else {
        0.0
    };

    score + authority_boost
});
```

**Best For:**
- E-commerce (price, ratings, inventory)
- Content platforms (views, likes, freshness)
- Enterprise search (permissions, departments)

### 4. Cross-Encoder Function

**Use Case:** Lightweight semantic reranking without ONNX
**Speed:** Fast (~2-5ms)
**Quality:** Good (but less than ONNX models)

```rust
use vecstore::reranking::CrossEncoderFn;

// Simple word overlap scorer
let reranker = CrossEncoderFn::new(|query: &str, doc: &str| {
    let query_words: Vec<&str> = query.split_whitespace().collect();
    let doc_words: Vec<&str> = doc.split_whitespace().collect();

    let overlap = query_words.iter()
        .filter(|w| doc_words.contains(w))
        .count();

    overlap as f32 / query_words.len() as f32
});
```

**Best For:**
- Prototyping
- Environments where ONNX isn't available
- Very low-latency requirements

### 5. Cross-Encoder (ONNX) ⭐ **RECOMMENDED**

**Use Case:** Maximum quality semantic reranking
**Speed:** Moderate (~100-200ms for 10 pairs)
**Quality:** Best

```rust
use vecstore::reranking::cross_encoder::{CrossEncoderReranker, CrossEncoderModel};

let reranker = CrossEncoderReranker::from_pretrained(
    CrossEncoderModel::MiniLML6V2  // or MiniLML12V2 for higher quality
)?;
```

**Best For:**
- Question answering
- High-quality search
- Production RAG systems
- When accuracy > speed

---

## Cross-Encoder Models

### Available Models

| Model | Dimension | Speed | Quality | Size | Best For |
|-------|-----------|-------|---------|------|----------|
| ms-marco-MiniLM-L-6-v2 | N/A | ~10ms/pair | Good | 90MB | Most use cases |
| ms-marco-MiniLM-L-12-v2 | N/A | ~20ms/pair | Better | 150MB | High accuracy |

### Model Setup

#### Option 1: From Pretrained (Recommended)

```rust
use vecstore::reranking::cross_encoder::{CrossEncoderReranker, CrossEncoderModel};

// Will use cached model if available
let reranker = CrossEncoderReranker::from_pretrained(
    CrossEncoderModel::MiniLML6V2
)?;
```

**Cache Location:**
`~/.cache/vecstore/cross-encoders/ms-marco-minilm-l6-v2/`

#### Option 2: From Local Directory

```rust
let reranker = CrossEncoderReranker::from_dir("./my_model")?;
```

**Required files in directory:**
- `model.onnx` - ONNX model file
- `tokenizer.json` - HuggingFace tokenizer

### Downloading Models

**Step 1:** Install conversion tools
```bash
pip install optimum onnx onnxruntime
```

**Step 2:** Convert model to ONNX
```bash
optimum-cli export onnx \
    --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --task text-classification \
    ./model_dir
```

**Step 3:** Place in cache directory
```bash
mkdir -p ~/.cache/vecstore/cross-encoders/ms-marco-minilm-l6-v2
cp model_dir/* ~/.cache/vecstore/cross-encoders/ms-marco-minilm-l6-v2/
```

### Using Cross-Encoders

```rust
// Load model
let reranker = CrossEncoderReranker::from_pretrained(
    CrossEncoderModel::MiniLML6V2
)?;

// Score a single query-document pair
let score = reranker.score_pair(
    "what is rust programming",
    "Rust is a systems programming language"
)?;

// Rerank search results
let results = store.query(&query_emb, 100, None)?;
let reranked = reranker.rerank(
    "what is rust programming",
    results,
    10  // Return top 10
)?;
```

---

## Best Practices

### 1. Two-Stage Retrieval

**Best Practice:** Retrieve more initially, rerank to fewer

```rust
// ✅ GOOD: Wide net → precise ranking
let initial = store.query(&query_emb, 100, None)?;  // Fast
let final_results = reranker.rerank(query_text, initial, 10)?;  // Accurate

// ❌ BAD: Reranking too few candidates
let initial = store.query(&query_emb, 10, None)?;  // Might miss good docs
let final_results = reranker.rerank(query_text, initial, 10)?;
```

**Recommended ratios:**
- Interactive search: 100 → 10
- Batch processing: 500 → 50
- High-precision: 200 → 5

### 2. Choose the Right Strategy

```rust
// For diversity (news, articles)
let reranker = MMRReranker::new(0.7);

// For custom logic (e-commerce)
let reranker = ScoreReranker::new(|n| {
    n.score + popularity_boost + inventory_penalty
});

// For maximum quality (RAG, QA)
let reranker = CrossEncoderReranker::from_pretrained(
    CrossEncoderModel::MiniLML6V2
)?;
```

### 3. Hybrid Approach

Combine multiple reranking stages:

```rust
// Stage 1: Fast retrieval
let results = store.query(&query_emb, 200, None)?;

// Stage 2: Custom business logic
let reranker1 = ScoreReranker::new(|n| {
    apply_business_rules(n)
});
let results = reranker1.rerank(query, results, 50)?;

// Stage 3: Semantic reranking
let reranker2 = CrossEncoderReranker::from_pretrained(
    CrossEncoderModel::MiniLML6V2
)?;
let final_results = reranker2.rerank(query, results, 10)?;
```

### 4. Cache Models

```rust
// ✅ GOOD: Load once, reuse
let reranker = Arc::new(
    CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)?
);

// Reuse across requests
for query in queries {
    let results = reranker.rerank(query, candidates, 10)?;
}

// ❌ BAD: Reload for each query
for query in queries {
    let reranker = CrossEncoderReranker::from_pretrained(
        CrossEncoderModel::MiniLML6V2
    )?;  // Slow!
}
```

### 5. Handle Edge Cases

```rust
use vecstore::reranking::Reranker;

let results = store.query(&query_emb, 100, None)?;

// Handle empty results
if results.is_empty() {
    return Ok(vec![]);
}

// Rerank with error handling
let reranked = match reranker.rerank(query, results, 10) {
    Ok(r) => r,
    Err(e) => {
        eprintln!("Reranking failed: {}, using original results", e);
        results  // Fallback to original
    }
};
```

---

## Performance Tuning

### Latency Characteristics

| Strategy | 10 Candidates | 100 Candidates | 1000 Candidates |
|----------|---------------|----------------|-----------------|
| Identity | <1ms | <1ms | <1ms |
| MMR | 2ms | 10ms | 100ms |
| Score-Based | 1ms | 5ms | 50ms |
| Cross-Encoder Fn | 2ms | 20ms | 200ms |
| Cross-Encoder ONNX | 100ms | 1s | 10s |

### Optimization Strategies

#### 1. Batch Size Selection

```rust
// Interactive applications
let initial_k = 100;  // Fast enough for <100ms total
let final_k = 10;

// Batch processing
let initial_k = 500;  // Slower but more thorough
let final_k = 50;
```

#### 2. Parallel Processing

```rust
use rayon::prelude::*;

// Process multiple queries in parallel
let results: Vec<_> = queries.par_iter()
    .map(|query| {
        let initial = store.query(&query.embedding, 100, None)?;
        reranker.rerank(&query.text, initial, 10)
    })
    .collect();
```

#### 3. Async Processing

```rust
use tokio::task;

// For I/O-bound operations
let handles: Vec<_> = queries.into_iter()
    .map(|query| {
        let reranker = reranker.clone();
        task::spawn(async move {
            // Reranking logic
        })
    })
    .collect();

let results = join_all(handles).await;
```

#### 4. Caching

```rust
use std::collections::HashMap;

// Cache reranked results
let mut cache: HashMap<String, Vec<Neighbor>> = HashMap::new();

fn get_reranked(query: &str, reranker: &impl Reranker) -> Result<Vec<Neighbor>> {
    if let Some(cached) = cache.get(query) {
        return Ok(cached.clone());
    }

    let results = store.query(&embed(query), 100, None)?;
    let reranked = reranker.rerank(query, results, 10)?;
    cache.insert(query.to_string(), reranked.clone());
    Ok(reranked)
}
```

---

## API Reference

### Reranker Trait

```rust
pub trait Reranker: Send + Sync {
    fn rerank(
        &self,
        query: &str,
        results: Vec<Neighbor>,
        top_k: usize
    ) -> Result<Vec<Neighbor>>;

    fn name(&self) -> &str;
}
```

### MMRReranker

```rust
impl MMRReranker {
    pub fn new(lambda: f32) -> Self;
    // lambda: 0.0 (pure diversity) to 1.0 (pure relevance)
}
```

### ScoreReranker

```rust
impl<F> ScoreReranker<F>
where F: Fn(&Neighbor) -> f32 + Send + Sync
{
    pub fn new(score_fn: F) -> Self;
}
```

### CrossEncoderFn

```rust
impl<F> CrossEncoderFn<F>
where F: Fn(&str, &str) -> f32 + Send + Sync
{
    pub fn new(score_fn: F) -> Self;
}
```

### CrossEncoderReranker

```rust
impl CrossEncoderReranker {
    pub fn from_pretrained(model: CrossEncoderModel) -> Result<Self>;
    pub fn from_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self>;
    pub fn score_pair(&self, query: &str, document: &str) -> Result<f32>;
}
```

---

## Examples

### Example 1: RAG Pipeline with Reranking

```rust
use vecstore::VecStore;
use vecstore::reranking::cross_encoder::{CrossEncoderReranker, CrossEncoderModel};

fn rag_pipeline(query: &str, query_emb: &[f32]) -> Result<Vec<String>> {
    // 1. Fast vector search
    let store = VecStore::new("./knowledge_base")?;
    let candidates = store.query(query_emb, 100, None)?;

    // 2. Semantic reranking
    let reranker = CrossEncoderReranker::from_pretrained(
        CrossEncoderModel::MiniLML6V2
    )?;
    let reranked = reranker.rerank(query, candidates, 5)?;

    // 3. Extract context for LLM
    let context: Vec<String> = reranked.iter()
        .map(|r| r.metadata.fields.get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string())
        .collect();

    Ok(context)
}
```

### Example 2: Multi-Stage Reranking

```rust
// Stage 1: Vector search (fast, broad)
let candidates = store.query(&query_emb, 500, None)?;

// Stage 2: Filter by recency (business rule)
let recent_filter = ScoreReranker::new(|n| {
    let timestamp = n.metadata.fields.get("timestamp")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    let age_days = (now() - timestamp) / 86400;
    if age_days < 7 {
        n.score + 0.3  // Boost recent
    } else {
        n.score
    }
});
let recent_results = recent_filter.rerank(query, candidates, 100)?;

// Stage 3: Diversity (avoid redundancy)
let mmr = MMRReranker::new(0.7);
let diverse_results = mmr.rerank(query, recent_results, 20)?;

// Stage 4: Semantic precision (final ranking)
let cross_encoder = CrossEncoderReranker::from_pretrained(
    CrossEncoderModel::MiniLML6V2
)?;
let final_results = cross_encoder.rerank(query, diverse_results, 5)?;
```

### Example 3: A/B Testing Reranking Strategies

```rust
// Compare different strategies
let strategies: Vec<(&str, Box<dyn Reranker>)> = vec![
    ("baseline", Box::new(IdentityReranker)),
    ("mmr_0.7", Box::new(MMRReranker::new(0.7))),
    ("cross_encoder", Box::new(
        CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)?
    )),
];

for (name, reranker) in strategies {
    let start = Instant::now();
    let results = reranker.rerank(query, candidates.clone(), 10)?;
    let duration = start.elapsed();

    println!("{}: {:?}, top_score: {:.4}",
             name, duration, results[0].score);
}
```

---

## Additional Resources

- [Example: Reranking Demo](../examples/reranking_demo.rs)
- [Tests: Cross-Encoder](../tests/cross_encoder_reranking.rs)
- [VecStore Main Documentation](./README.md)

---

## Troubleshooting

### Model Loading Fails

**Error:** "Model file not found"

**Solution:**
1. Download and convert model using `optimum-cli`
2. Place in `~/.cache/vecstore/cross-encoders/[model-name]/`
3. Ensure `model.onnx` and `tokenizer.json` exist

### Slow Reranking

**Issue:** Reranking takes too long

**Solutions:**
- Reduce candidate set size (100 instead of 1000)
- Use faster model (MiniLM-L-6 instead of L-12)
- Use MMR or Score-Based instead of Cross-Encoder
- Enable parallel processing
- Cache reranked results

### Poor Quality Results

**Issue:** Reranked results aren't better

**Solutions:**
- Ensure query text is provided (not just embeddings)
- Check metadata has "text" field for cross-encoders
- Try different lambda values for MMR (0.5-0.9)
- Verify score function logic for Score-Based
- Use Cross-Encoder ONNX for best quality

---

## License

Reranking module is part of VecStore and uses the same MIT license.
