# Weeks 3-5: Advanced Features Implementation Plan

**Goal:** Add embedding backends, cloud integrations, and reranking capabilities

**Status:** Ready to implement
**Timeline:** 3 weeks (Weeks 3-5 of POST-1.0 roadmap)

---

## 🎯 OVERVIEW

### What We're Building

**Week 3: Candle Embedding Backend**
- Pure Rust ML framework integration (HuggingFace Candle)
- Local embedding generation (no external dependencies)
- Support for popular sentence-transformer models
- CPU and GPU support

**Week 4: Cloud Embeddings**
- OpenAI API integration (text-embedding-3-small/large)
- Rate limiting and retry logic
- Batch processing for cost efficiency
- Token counting and cost estimation

**Week 5: Advanced Reranking**
- Cross-encoder reranking for better search quality
- Candle-based implementation
- ColBERT support (stretch goal)

---

## 📋 DETAILED TASKS

### WEEK 3: CANDLE EMBEDDING BACKEND

#### Task 1: Add Candle Dependencies

**File:** `Cargo.toml`

```toml
# Add to [dependencies]
candle-core = { version = "0.7", optional = true }
candle-nn = { version = "0.7", optional = true }
candle-transformers = { version = "0.7", optional = true }
hf-hub = { version = "0.3", optional = true }  # For model downloads
safetensors = { version = "0.4", optional = true }

# Update [features]
candle-embeddings = [
    "candle-core",
    "candle-nn",
    "candle-transformers",
    "hf-hub",
    "safetensors",
    "tokenizers",
]
```

#### Task 2: Create Embedding Backend Trait

**File:** `src/embeddings/mod.rs`

```rust
pub trait EmbeddingBackend: Send + Sync {
    /// Embed a single text into a vector
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed multiple texts in batch (more efficient)
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Get the model name/identifier
    fn model_name(&self) -> &str;
}
```

#### Task 3: Implement Candle Backend

**File:** `src/embeddings/candle_backend.rs`

**Models to support:**
1. `sentence-transformers/all-MiniLM-L6-v2` (384 dim) - Fast, lightweight
2. `BAAI/bge-small-en-v1.5` (384 dim) - Good quality
3. `sentence-transformers/all-mpnet-base-v2` (768 dim) - Higher quality

**Features:**
- Automatic model download from HuggingFace
- Model caching in ~/.cache/vecstore/models
- Mean pooling for sentence embeddings
- Batch processing with configurable batch size
- CPU and GPU support (via candle features)

**Structure:**
```rust
pub struct CandleEmbedding {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimension: usize,
    model_name: String,
}

impl CandleEmbedding {
    pub fn new(model_name: &str) -> Result<Self> {
        // Download model if not cached
        // Load model and tokenizer
        // Initialize on CPU or GPU
    }

    fn mean_pooling(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Implement mean pooling
    }
}
```

#### Task 4: Tests for Candle Backend

**File:** `tests/embeddings_candle.rs`

**Test coverage:**
- Model loading and caching
- Single text embedding
- Batch embedding
- Dimension verification
- Similarity computation
- GPU/CPU device selection
- Error handling (invalid model, download failures)

**Minimum 20 tests**

#### Task 5: Examples

**Files:**
1. `examples/candle_basic_embedding.rs` - Basic usage
2. `examples/candle_rag_pipeline.rs` - Complete RAG with Candle
3. `examples/candle_batch_processing.rs` - Efficient batch embedding

#### Task 6: Documentation

**File:** `docs/embeddings/candle.md`

- Installation guide
- Supported models
- Performance benchmarks
- GPU setup guide
- Memory requirements
- Comparison with ONNX backend

---

### WEEK 4: OPENAI API INTEGRATION

#### Task 1: Add Dependencies

**File:** `Cargo.toml`

```toml
# Add to [dependencies]
reqwest = { version = "0.12", optional = true, default-features = false, features = ["json", "rustls-tls"] }
async-trait = { version = "0.1", optional = true }

# Update [features]
openai-embeddings = [
    "reqwest",
    "async-trait",
    "tokio/full",
]
```

#### Task 2: Implement OpenAI Backend

**File:** `src/embeddings/openai_backend.rs`

**Supported models:**
- `text-embedding-3-small` (1536 dim, $0.02/1M tokens)
- `text-embedding-3-large` (3072 dim, $0.13/1M tokens)
- `text-embedding-ada-002` (1536 dim, legacy)

**Features:**
- Async API calls
- Rate limiting (configurable RPM/TPM)
- Automatic retry with exponential backoff
- Batch processing (up to 2048 texts per request)
- Token counting for cost estimation
- API key from environment or config

**Structure:**
```rust
pub struct OpenAIEmbedding {
    client: reqwest::Client,
    api_key: String,
    model: OpenAIModel,
    rate_limiter: RateLimiter,
    dimension: usize,
}

#[derive(Clone, Copy)]
pub enum OpenAIModel {
    TextEmbedding3Small,
    TextEmbedding3Large,
    Ada002,
}

impl OpenAIEmbedding {
    pub async fn new(api_key: String, model: OpenAIModel) -> Result<Self> {
        // Initialize client with rate limiting
    }

    pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>> {
        // Single embedding with rate limiting
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Batch with automatic chunking (max 2048)
        // Rate limiting and retry
    }

    pub fn estimate_cost(&self, texts: &[&str]) -> f64 {
        // Calculate cost based on token count
    }
}
```

#### Task 3: Tests for OpenAI Backend

**File:** `tests/embeddings_openai.rs`

**Test coverage:**
- Mock API responses
- Rate limiting behavior
- Retry logic
- Batch chunking
- Cost estimation
- Error handling (API errors, network failures)

**Minimum 15 tests** (with mocking, no real API calls)

#### Task 4: Examples

**Files:**
1. `examples/openai_basic_embedding.rs` - Basic usage
2. `examples/openai_rag_pipeline.rs` - RAG with OpenAI
3. `examples/openai_cost_estimation.rs` - Cost tracking

#### Task 5: Documentation

**File:** `docs/embeddings/openai.md`

- API key setup
- Model selection guide
- Cost considerations
- Rate limits and quotas
- Best practices for batch processing
- Error handling

---

### WEEK 5: ADVANCED RERANKING

#### Task 1: Add Dependencies (if needed)

Reranking will use Candle, so no new dependencies.

#### Task 2: Create Reranking Trait

**File:** `src/reranking/mod.rs`

```rust
pub trait Reranker: Send + Sync {
    /// Rerank query-document pairs
    /// Returns scores (higher = more relevant)
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;

    /// Get model name
    fn model_name(&self) -> &str;
}
```

#### Task 3: Implement Cross-Encoder Reranker

**File:** `src/reranking/cross_encoder.rs`

**Supported models:**
1. `cross-encoder/ms-marco-MiniLM-L-6-v2` - Fast, good quality
2. `BAAI/bge-reranker-base` - Better quality
3. `BAAI/bge-reranker-large` - Best quality (slower)

**Features:**
- Candle-based implementation
- Batch processing
- Score normalization
- Top-k filtering

**Structure:**
```rust
pub struct CrossEncoderReranker {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    model_name: String,
}

impl CrossEncoderReranker {
    pub fn new(model_name: &str) -> Result<Self> {
        // Load cross-encoder model
    }

    pub fn rerank_top_k(&self, query: &str, documents: &[&str], k: usize) -> Result<Vec<(usize, f32)>> {
        // Rerank and return top-k with indices
    }
}
```

#### Task 4: Integrate Reranking into Search

**File:** `src/store.rs` (extend)

```rust
impl VecStore {
    pub fn query_with_reranking(
        &self,
        vector: &[f32],
        k: usize,
        rerank_k: usize,
        reranker: &dyn Reranker,
        filter: Option<&FilterExpr>,
    ) -> Result<Vec<SearchResult>> {
        // 1. Get initial results (k * rerank_factor)
        // 2. Rerank top candidates
        // 3. Return top-k after reranking
    }
}
```

#### Task 5: ColBERT Support (Stretch Goal)

**File:** `src/reranking/colbert.rs`

**If time permits:**
- Late interaction reranking
- Token-level similarity matching
- More expensive but higher quality

#### Task 6: Tests for Reranking

**File:** `tests/reranking.rs`

**Test coverage:**
- Model loading
- Reranking quality (verify order changes)
- Batch processing
- Integration with VecStore.query_with_reranking
- Performance benchmarks

**Minimum 15 tests**

#### Task 7: Examples

**Files:**
1. `examples/reranking_basic.rs` - Basic reranking
2. `examples/reranking_rag.rs` - RAG with reranking
3. `examples/reranking_comparison.rs` - Before/after quality comparison

#### Task 8: Documentation

**File:** `docs/reranking.md`

- What is reranking and why it helps
- Model selection guide
- Performance vs quality tradeoffs
- Integration examples
- Benchmarks showing quality improvement

---

## 📊 DELIVERABLES SUMMARY

### Code

**New Files:**
- `src/embeddings/mod.rs` (trait definitions)
- `src/embeddings/candle_backend.rs` (~400 lines)
- `src/embeddings/openai_backend.rs` (~300 lines)
- `src/reranking/mod.rs` (trait definitions)
- `src/reranking/cross_encoder.rs` (~350 lines)
- `src/reranking/colbert.rs` (~400 lines, if implemented)

**Total New Code:** ~1,500-2,000 lines

### Tests

- `tests/embeddings_candle.rs` (20+ tests)
- `tests/embeddings_openai.rs` (15+ tests)
- `tests/reranking.rs` (15+ tests)

**Total Tests:** 50+ new tests

### Examples

- 3 Candle embedding examples
- 3 OpenAI embedding examples
- 3 Reranking examples

**Total Examples:** 9 new examples

### Documentation

- `docs/embeddings/candle.md`
- `docs/embeddings/openai.md`
- `docs/embeddings/comparison.md`
- `docs/reranking.md`

**Total Documentation:** ~2,000 lines

---

## 🎯 SUCCESS CRITERIA

### Week 3 Complete When:
- ✅ Candle backend compiles and runs
- ✅ At least 2 models supported
- ✅ 20+ tests passing
- ✅ 3 examples working
- ✅ Documentation complete

### Week 4 Complete When:
- ✅ OpenAI backend compiles and runs
- ✅ Rate limiting and retry logic working
- ✅ 15+ tests passing (mocked)
- ✅ 3 examples working
- ✅ Documentation complete

### Week 5 Complete When:
- ✅ Cross-encoder reranking working
- ✅ Integration with VecStore.query complete
- ✅ 15+ tests passing
- ✅ 3 examples working
- ✅ Documentation complete
- ✅ (Optional) ColBERT implemented

### Weeks 3-5 Complete When:
- ✅ **50+ new tests passing**
- ✅ **9 new examples working**
- ✅ **All documentation complete**
- ✅ **Benchmarks showing performance and quality**

---

## 🚀 IMPLEMENTATION ORDER

### Priority 1: Candle Backend (Week 3)
1. Add dependencies
2. Create embedding trait
3. Implement CandleEmbedding
4. Write tests
5. Create examples
6. Write documentation

### Priority 2: OpenAI Backend (Week 4)
1. Add dependencies
2. Implement OpenAIEmbedding
3. Add rate limiting and retry
4. Write tests (with mocking)
5. Create examples
6. Write documentation

### Priority 3: Reranking (Week 5)
1. Create reranking trait
2. Implement CrossEncoderReranker
3. Integrate with VecStore
4. Write tests
5. Create examples
6. Write documentation
7. (Optional) Implement ColBERT

---

## 📈 EXPECTED IMPACT

### For Users

**Candle Backend:**
- Pure Rust embedding (no Python/ONNX dependencies)
- Fast inference on CPU and GPU
- Easy deployment

**OpenAI Backend:**
- State-of-the-art embedding quality
- No local model management
- Easy prototyping

**Reranking:**
- 10-30% improvement in search quality
- Configurable precision/performance tradeoff
- Production-ready

### For VecStore

- **Complete embedding story** - Local and cloud options
- **Best-in-class search quality** - With reranking
- **Competitive advantage** - vs other vector DBs
- **Production-ready** - All features tested and documented

---

## 🎊 NEXT STEPS

After Weeks 3-5 complete, move to:

**Weeks 6-7: Observability & Monitoring**
- OpenTelemetry integration
- Grafana dashboards
- Production monitoring

**Weeks 8-9: Polish, Testing & Documentation**
- Final polish
- 400+ total tests
- Security audit

**Week 10: PUBLICATION**
- Publish to crates.io, PyPI, NPM
- Announcements
- Community setup

---

**Let's churn through Weeks 3-5!** 🚀
