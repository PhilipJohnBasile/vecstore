# VecStore Templates

Ready-to-use project templates for common VecStore use cases.

## Available Templates

### 1. Local RAG App (`local-rag-app/`)
Privacy-first RAG application that runs entirely offline.

**Best for:**
- Sensitive document search
- Offline-capable applications
- Edge deployments

**Features:**
- Local embeddings with ONNX
- No network required
- Full data privacy

```bash
cd local-rag-app
cargo run --release
```

### 2. Browser Search (`browser-search/`)
Client-side semantic search using WebAssembly.

**Best for:**
- Static websites
- Privacy-conscious web apps
- No-backend deployments

**Features:**
- 100% client-side
- Zero server costs
- Instant search

```bash
cd browser-search
# Open index.html in a browser
```

### 3. Enterprise Explainable (`enterprise-explainable/`)
Production-ready template for regulated industries.

**Best for:**
- Healthcare (HIPAA)
- Finance (SOX, PCI-DSS)
- Legal & Government

**Features:**
- Explainable search results
- Full audit logging
- Vector lineage tracking
- Privacy-preserving search

```bash
cd enterprise-explainable
cargo run --release
```

## Quick Start

1. Copy the template you need
2. Update `Cargo.toml` with your dependencies
3. Replace placeholder embeddings with real ones
4. Deploy!

## Adding Real Embeddings

### Local Embeddings (ONNX)
```toml
[dependencies]
vecstore = { version = "0.1", features = ["embeddings"] }
```

### OpenAI Embeddings
```toml
[dependencies]
vecstore = { version = "0.1", features = ["openai-embeddings"] }
tokio = { version = "1", features = ["full"] }
```

### Ollama (Local LLM)
```toml
[dependencies]
vecstore = { version = "0.1", features = ["ollama"] }
tokio = { version = "1", features = ["full"] }
```

## Need Help?

- [Documentation](https://docs.rs/vecstore)
- [GitHub Issues](https://github.com/PhilipJohnBasile/vecstore/issues)
- [Examples](../examples/)
