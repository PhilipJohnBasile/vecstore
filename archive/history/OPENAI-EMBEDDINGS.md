# OpenAI Embeddings Integration

Complete guide to using OpenAI's embedding models with VecStore for cloud-powered semantic search.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Available Models](#available-models)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Cost Management](#cost-management)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

VecStore's OpenAI embeddings backend provides seamless integration with OpenAI's state-of-the-art text embedding models. This enables:

- **Cloud-powered embeddings** without local model management
- **Production-ready API** with built-in rate limiting and retry logic
- **Multiple models** optimized for different use cases and budgets
- **Async/await support** for efficient concurrent processing
- **Batch processing** to maximize throughput and minimize costs

### When to Use OpenAI Embeddings

✅ **Use OpenAI when:**
- You want the best embedding quality without local setup
- You're building a production application with reliable infrastructure
- You need to get started quickly without downloading models
- Your embedding workload is moderate (cost-effective up to millions of docs)
- You want automatic scaling without managing GPU resources

❌ **Consider alternatives when:**
- You need 100% offline/airgapped operation
- You're embedding billions of documents (cost may be high)
- You have strict data privacy requirements (data sent to OpenAI)
- You need deterministic embeddings across all environments

---

## Installation

### 1. Add VecStore with OpenAI Feature

```toml
# Cargo.toml
[dependencies]
vecstore = { version = "1.0", features = ["embeddings", "openai-embeddings"] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
```

### 2. Get Your OpenAI API Key

1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Navigate to **API Keys** section
3. Create a new secret key
4. Set it as an environment variable:

```bash
export OPENAI_API_KEY='sk-...'
```

### 3. Verify Installation

```bash
cargo run --example openai_basic --features "embeddings,openai-embeddings"
```

---

## Quick Start

### Rust

```rust
use anyhow::Result;
use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};
use vecstore::store::VecStore;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Create embedder
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small)
        .await?
        .with_rate_limit(500)  // 500 requests/minute
        .with_max_retries(3);   // Retry failed requests

    // 2. Embed text
    let embedding = embedder.embed_async("Hello, world!").await?;
    println!("Embedding dimension: {}", embedding.len());  // 1536

    // 3. Batch embed multiple texts
    let texts = vec!["Document 1", "Document 2", "Document 3"];
    let embeddings = embedder.embed_batch_async(&texts).await?;
    println!("Generated {} embeddings", embeddings.len());  // 3

    // 4. Store in VecStore
    let mut store = VecStore::open("./my_db")?;
    for (i, emb) in embeddings.iter().enumerate() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("text".to_string(), serde_json::json!(texts[i]));
        store.upsert(&format!("doc{}", i), emb, metadata)?;
    }

    // 5. Query
    let query_emb = embedder.embed_async("search query").await?;
    let results = store.query(&query_emb, 5, None)?;

    for result in results {
        println!("Score: {:.4} - {}", result.score, result.metadata["text"]);
    }

    Ok(())
}
```

### Python

```python
import os
import asyncio
from vecstore_py import VecStore
# Note: Python bindings for OpenAI backend coming in Week 4

# For now, use the Rust async client or wait for Python async support
```

---

## Available Models

OpenAI provides three embedding models with different tradeoffs:

| Model | Dimension | Cost/1M Tokens | Best For | Released |
|-------|-----------|----------------|----------|----------|
| `text-embedding-3-small` | 1,536 | $0.02 | Most use cases, best value | Jan 2024 |
| `text-embedding-3-large` | 3,072 | $0.13 | Maximum accuracy needed | Jan 2024 |
| `text-embedding-ada-002` | 1,536 | $0.10 | Legacy (use 3-small instead) | Dec 2022 |

### Model Selection Guide

#### text-embedding-3-small ⭐ **RECOMMENDED**

```rust
let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small).await?;
```

**When to use:**
- General-purpose semantic search
- RAG (Retrieval-Augmented Generation) systems
- Document similarity
- Question answering
- Most production applications

**Advantages:**
- 6.5x cheaper than `3-large`
- Excellent quality-to-cost ratio
- Fast inference
- Proven performance

**Benchmarks:**
- MTEB Score: 62.3%
- Suitable for 95% of use cases

#### text-embedding-3-large

```rust
let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Large).await?;
```

**When to use:**
- Maximum embedding quality required
- Complex domain-specific search
- Research applications
- When cost is not a primary concern

**Advantages:**
- Highest quality embeddings
- Better at nuanced semantic differences
- 2x the dimensions (3,072 vs 1,536)

**Benchmarks:**
- MTEB Score: 64.6%
- ~3.7% improvement over small

**Cost consideration:**
- 6.5x more expensive than `3-small`
- For 1M documents (1000 chars each): ~$32.50 vs ~$5.00

#### text-embedding-ada-002 (Legacy)

```rust
let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::Ada002).await?;
```

**When to use:**
- Legacy systems migration
- Compatibility with existing embeddings
- Not recommended for new projects

**Note:** Use `TextEmbedding3Small` instead - it's 5x cheaper and performs better.

---

## Configuration

### Rate Limiting

Control API request rate to avoid hitting OpenAI's limits:

```rust
let embedder = OpenAIEmbedding::new(api_key, model)
    .await?
    .with_rate_limit(500);  // Requests per minute
```

**Default:** 500 requests/minute (conservative)

**OpenAI Tier Limits:**
- **Free tier:** 3 RPM (requests per minute)
- **Tier 1:** 500 RPM
- **Tier 2:** 5,000 RPM
- **Tier 3+:** Higher limits

💡 Set your rate limit slightly below your tier limit to account for request variance.

### Retry Configuration

Automatic retry with exponential backoff on failures:

```rust
let embedder = OpenAIEmbedding::new(api_key, model)
    .await?
    .with_max_retries(3);  // Retry up to 3 times
```

**Default:** 3 retries

**Retry strategy:**
- 1st retry: Wait 2 seconds
- 2nd retry: Wait 4 seconds
- 3rd retry: Wait 8 seconds

**Retries on:**
- Rate limit errors (429)
- Network failures
- Temporary server errors (5xx)

**No retry on:**
- Invalid API key (401)
- Invalid request (400)
- Max retries exceeded

### Timeout Configuration

The HTTP client has a 30-second timeout by default. This is configured internally and should be sufficient for most workloads.

For very large batches (2000+ texts), the timeout is automatically handled by the retry mechanism.

---

## Usage Examples

### Example 1: Basic Embedding

```rust
use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let embedder = OpenAIEmbedding::new(
        std::env::var("OPENAI_API_KEY")?,
        OpenAIModel::TextEmbedding3Small
    ).await?;

    // Single text
    let embedding = embedder.embed_async("Rust is amazing!").await?;
    println!("Dimension: {}", embedding.len());
    println!("First 5 values: {:?}", &embedding[0..5]);

    Ok(())
}
```

### Example 2: Batch Processing

```rust
let texts = vec![
    "The quick brown fox",
    "jumps over the lazy dog",
    "Rust is a systems programming language",
    // ... up to 2048 texts per batch
];

// All texts embedded in a single API call
let embeddings = embedder.embed_batch_async(&texts).await?;

println!("Embedded {} texts", embeddings.len());
```

### Example 3: RAG Pipeline

See [`examples/openai_rag.rs`](../examples/openai_rag.rs) for a complete RAG implementation:

```rust
// 1. Split documents into chunks
let splitter = RecursiveCharacterTextSplitter::new(200, 50);
let chunks = splitter.split_text(&document);

// 2. Embed all chunks
let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
let embeddings = embedder.embed_batch_async(&chunk_refs).await?;

// 3. Store in VecStore
let mut store = VecStore::open("./rag_db")?;
for (i, (chunk, embedding)) in chunks.iter().zip(embeddings.iter()).enumerate() {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("text".to_string(), serde_json::json!(chunk));
    metadata.insert("chunk_id".to_string(), serde_json::json!(i));
    store.upsert(&format!("chunk_{}", i), embedding, metadata)?;
}

// 4. Query
let query = "What is Rust?";
let query_emb = embedder.embed_async(query).await?;
let results = store.query(&query_emb, 5, None)?;
```

### Example 4: Computing Similarity

```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (mag_a * mag_b)
}

let emb1 = embedder.embed_async("Rust programming").await?;
let emb2 = embedder.embed_async("Python programming").await?;
let emb3 = embedder.embed_async("Cooking recipes").await?;

println!("Rust ↔ Python: {:.4}", cosine_similarity(&emb1, &emb2));    // ~0.85
println!("Rust ↔ Cooking: {:.4}", cosine_similarity(&emb1, &emb3));   // ~0.30
```

### Example 5: Synchronous Wrapper

For non-async contexts:

```rust
use vecstore::embeddings::TextEmbedder;

// In a non-async function
fn process_text(embedder: &OpenAIEmbedding, text: &str) -> anyhow::Result<Vec<f32>> {
    // TextEmbedder trait provides synchronous methods
    embedder.embed(text)  // Blocks until complete
}
```

**Note:** The synchronous wrapper creates a new Tokio runtime for each call. Prefer async methods for better performance.

---

## Cost Management

### Understanding Costs

OpenAI charges based on **tokens**, not characters:
- **Rough estimate:** 1 token ≈ 4 characters
- **Exact count:** Requires tokenization (use tiktoken library)

**Cost formula:**
```
Cost = (tokens / 1,000,000) × cost_per_million_tokens
```

### Estimating Costs

```rust
let texts = vec!["Document 1", "Document 2", "Document 3"];
let estimated_cost = embedder.estimate_cost(&texts);
println!("Estimated cost: ${:.6}", estimated_cost);
```

**Note:** `estimate_cost()` uses character-based estimation (4 chars = 1 token). Actual cost may vary by ±20%.

### Cost Examples

#### Small Project (1,000 documents)

```
Documents: 1,000
Average length: 500 characters
Estimated tokens: 125,000

text-embedding-3-small: $0.0025  ✅
text-embedding-3-large: $0.0163
```

#### Medium Project (100,000 documents)

```
Documents: 100,000
Average length: 1,000 characters
Estimated tokens: 25,000,000

text-embedding-3-small: $0.50    ✅
text-embedding-3-large: $3.25
```

#### Large Project (1,000,000 documents)

```
Documents: 1,000,000
Average length: 1,000 characters
Estimated tokens: 250,000,000

text-embedding-3-small: $5.00    ✅
text-embedding-3-large: $32.50
```

### Cost Optimization Strategies

#### 1. Use text-embedding-3-small

```rust
// Save 85% compared to 3-large
let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small).await?;
```

#### 2. Always Batch Process

```rust
// ❌ BAD: Individual requests (10 API calls)
for text in &texts {
    embedder.embed_async(text).await?;
}

// ✅ GOOD: Batch request (1 API call)
let embeddings = embedder.embed_batch_async(&texts).await?;
```

**Savings:** Same cost, but 90%+ fewer API calls and faster processing.

#### 3. Cache Embeddings

```rust
// Store embeddings in VecStore
store.upsert(doc_id, &embedding, metadata)?;

// Reuse instead of re-embedding
if let Some(existing) = store.get(doc_id)? {
    return Ok(existing.embedding);
}
```

#### 4. Optimize Input Length

```rust
// Remove unnecessary content before embedding
let clean_text = text
    .trim()
    .replace("  ", " ")  // Remove extra spaces
    .lines()
    .filter(|line| !line.is_empty())
    .collect::<Vec<_>>()
    .join(" ");

let embedding = embedder.embed_async(&clean_text).await?;
```

#### 5. Chunk Wisely

```rust
use vecstore::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};

// Balance between granularity and cost
let splitter = RecursiveCharacterTextSplitter::new(
    500,  // Larger chunks = fewer embeddings = lower cost
    50    // Some overlap for context
);
```

#### 6. Track and Monitor

```rust
struct CostTracker {
    total_cost: f64,
    total_tokens: usize,
}

impl CostTracker {
    fn track(&mut self, texts: &[&str], embedder: &OpenAIEmbedding) {
        let cost = embedder.estimate_cost(texts);
        self.total_cost += cost;
        println!("Batch cost: ${:.6} | Total: ${:.4}", cost, self.total_cost);
    }
}
```

### Budget Planning

See [`examples/openai_cost_tracking.rs`](../examples/openai_cost_tracking.rs) for detailed cost analysis and planning tools.

---

## Best Practices

### 1. API Key Security

```rust
// ✅ GOOD: Environment variable
let api_key = std::env::var("OPENAI_API_KEY")?;

// ❌ BAD: Hardcoded in code
let api_key = "sk-...".to_string();  // Never do this!

// ✅ GOOD: From secure config
let api_key = load_from_secret_manager()?;
```

**Never commit API keys to version control!**

### 2. Error Handling

```rust
match embedder.embed_async(text).await {
    Ok(embedding) => {
        // Process embedding
    }
    Err(e) => {
        if e.to_string().contains("429") {
            eprintln!("Rate limit exceeded - consider reducing rate_limit");
        } else if e.to_string().contains("401") {
            eprintln!("Invalid API key - check OPENAI_API_KEY");
        } else {
            eprintln!("Embedding failed: {}", e);
        }
    }
}
```

### 3. Batch Sizing

```rust
// OpenAI limit: 2048 texts per request
// VecStore automatically chunks larger batches

let texts: Vec<String> = load_10000_documents();
let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

// Automatically split into multiple API calls
let embeddings = embedder.embed_batch_async(&text_refs).await?;
```

### 4. Rate Limit Configuration

```rust
// Match your OpenAI tier
let embedder = OpenAIEmbedding::new(api_key, model)
    .await?
    .with_rate_limit(match tier {
        "free" => 3,
        "tier1" => 500,
        "tier2" => 5000,
        _ => 500,
    });
```

### 5. Concurrent Processing

```rust
use futures::future::join_all;

// Process multiple batches concurrently
let batch1 = embedder.embed_batch_async(&texts1);
let batch2 = embedder.embed_batch_async(&texts2);
let batch3 = embedder.embed_batch_async(&texts3);

let results = join_all(vec![batch1, batch2, batch3]).await;
```

**Note:** Rate limiter handles concurrent requests automatically.

### 6. Monitoring and Logging

```rust
use std::time::Instant;

let start = Instant::now();
let embeddings = embedder.embed_batch_async(&texts).await?;
let duration = start.elapsed();

println!("Embedded {} texts in {:.2}s ({:.0} texts/sec)",
         texts.len(),
         duration.as_secs_f64(),
         texts.len() as f64 / duration.as_secs_f64());
```

---

## API Reference

### OpenAIModel

```rust
pub enum OpenAIModel {
    TextEmbedding3Small,  // 1536 dimensions, $0.02/1M tokens
    TextEmbedding3Large,  // 3072 dimensions, $0.13/1M tokens
    Ada002,               // 1536 dimensions, $0.10/1M tokens (legacy)
}
```

**Methods:**
- `as_str(&self) -> &'static str` - Get model name string
- `dimension(&self) -> usize` - Get embedding dimension
- `cost_per_million_tokens(&self) -> f64` - Get cost in USD

### OpenAIEmbedding

```rust
pub struct OpenAIEmbedding { /* fields omitted */ }
```

**Constructor:**
```rust
pub async fn new(api_key: String, model: OpenAIModel) -> Result<Self>
```

**Configuration:**
```rust
pub fn with_rate_limit(self, requests_per_minute: usize) -> Self
pub fn with_max_retries(self, max_retries: usize) -> Self
```

**Async Methods:**
```rust
pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>>
pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>
```

**Utility Methods:**
```rust
pub fn estimate_cost(&self, texts: &[&str]) -> f64
pub fn model(&self) -> OpenAIModel
```

**TextEmbedder Trait (Sync):**
```rust
fn embed(&self, text: &str) -> Result<Vec<f32>>
fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>
fn dimension(&self) -> Result<usize>
```

---

## Troubleshooting

### Error: "Invalid API key"

```
Error: OpenAI API error 401: Unauthorized
```

**Solution:**
```bash
# Check your API key is set correctly
echo $OPENAI_API_KEY

# Ensure it starts with 'sk-'
export OPENAI_API_KEY='sk-...'
```

### Error: "Rate limit exceeded"

```
Error: OpenAI API error 429: Rate limit exceeded
```

**Solution:**
```rust
// Reduce rate limit to match your tier
let embedder = embedder.with_rate_limit(3);  // For free tier

// Or upgrade your OpenAI tier
```

### Error: "Request timeout"

```
Error: Failed to call OpenAI API: timeout
```

**Solution:**
- Check your internet connection
- OpenAI API may be experiencing issues
- Try reducing batch size:

```rust
// Instead of 2048 texts, try smaller batches
let embeddings = embedder.embed_batch_async(&texts[0..100]).await?;
```

### Error: "Failed to parse response"

```
Error: Failed to parse OpenAI response
```

**Solution:**
- OpenAI API may have changed - check for VecStore updates
- Try with a single text to isolate the issue:

```rust
let embedding = embedder.embed_async("test").await?;
```

### Issue: Slow embedding speed

**Checklist:**
1. ✅ Are you using batch processing?
2. ✅ Is your rate limit set appropriately?
3. ✅ Is your internet connection stable?
4. ✅ Are you processing concurrently when possible?

**Example optimization:**
```rust
// Slow (10 seconds for 100 texts)
for text in &texts {
    embedder.embed_async(text).await?;
}

// Fast (<1 second for 100 texts)
embedder.embed_batch_async(&texts).await?;
```

### Issue: Unexpected costs

**Debug steps:**

```rust
// 1. Estimate before embedding
let estimated = embedder.estimate_cost(&texts);
println!("Estimated cost: ${:.6}", estimated);

// 2. Check actual usage in OpenAI dashboard
// platform.openai.com/usage

// 3. Track costs in your application
let mut total_cost = 0.0;
for batch in batches {
    let cost = embedder.estimate_cost(&batch);
    total_cost += cost;
    println!("Running total: ${:.4}", total_cost);
}
```

---

## Additional Resources

### Examples
- [`examples/openai_basic.rs`](../examples/openai_basic.rs) - Basic usage
- [`examples/openai_rag.rs`](../examples/openai_rag.rs) - Complete RAG pipeline
- [`examples/openai_cost_tracking.rs`](../examples/openai_cost_tracking.rs) - Cost analysis

### Documentation
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/embeddings)
- [VecStore Main Documentation](./README.md)

### Testing
- [`tests/openai_embeddings.rs`](../tests/openai_embeddings.rs) - Test suite

---

## License

OpenAI embeddings backend is part of VecStore and uses the same MIT license.

**Note:** Using OpenAI's API is subject to OpenAI's [Terms of Use](https://openai.com/policies/terms-of-use).
