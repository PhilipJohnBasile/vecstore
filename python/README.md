# VecStore Python Bindings

High-performance vector database with RAG toolkit for Python, powered by Rust.

> **Status:** Python bindings track the 0.0.2 alpha release. APIs may change between versions.

## Installation

```bash
pip install vecstore-rs
```

**Note:** The package is published as `vecstore-rs` on PyPI, but imports as `vecstore` in Python.

## Quick Start

```python
from vecstore import VecStore, Query

# Create or open a vector store
store = VecStore.open("./my_db")

# Insert vectors with metadata
store.upsert(
    id="doc1",
    vector=[0.1, 0.2, 0.3, ...],
    metadata={"text": "Hello world", "category": "greeting"}
)

# Query for similar vectors
results = store.query(
    vector=[0.1, 0.2, 0.3, ...],
    k=5
)

for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
    print(f"Metadata: {result.metadata}")
```

## LangChain Integration

VecStore provides native LangChain-compatible classes for seamless integration with LLM applications:

```python
from vecstore import LangChainVectorStore, Document

# Create a LangChain-compatible vector store
store = LangChainVectorStore("./langchain_db")

# Add documents with embeddings (from your embedding model)
store.add_embeddings(
    texts=["Hello world", "Goodbye world"],
    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}]
)

# Similarity search
results = store.similarity_search_by_vector(
    embedding=[0.1, 0.2, 0.3],
    k=5
)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {doc.score}")
```

### With Your Embedding Model

```python
from vecstore import LangChainVectorStore
# Use with any embedding model (OpenAI, HuggingFace, etc.)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
store = LangChainVectorStore("./my_rag_db")

# Add documents
texts = ["Document 1 content", "Document 2 content"]
embeddings = model.encode(texts).tolist()
store.add_embeddings(texts=texts, embeddings=embeddings)

# Query
query_embedding = model.encode("search query").tolist()
results = store.similarity_search_by_vector(query_embedding, k=3)
```

## Features

- **Fast**: Rust core avoids Python hot loops for distance calculations
- **Complete RAG Toolkit**: Text splitting, reranking, evaluation
- **LangChain Compatible**: Native Document and VectorStore classes
- **Operational Features**: Persistence, namespaces, server mode
- **Pythonic API**: Type hints, familiar patterns
- **Zero Config**: Works out of the box

## Documentation

See the main repository documentation:

- [Quick Start](../QUICKSTART.md)
- [Documentation Index](../docs/README.md)
- [API Reference (Rust docs)](https://docs.rs/vecstore)
- [Examples](examples/)

## Examples

See the `examples/` directory for complete examples:

- `basic_rag.py` - Simple RAG workflow
- `fastapi_integration.py` - FastAPI REST API
- `evaluation.py` - RAG quality measurement
- `production.py` - Production deployment

## Development

Building from source:

```bash
# Install maturin
pip install maturin

# Build in development mode
maturin develop --features python

# Run tests
pytest tests/
```

## License

Apache-2.0 License - see LICENSE file for details
