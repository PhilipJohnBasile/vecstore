# VecStore Python Bindings

High-performance vector database with RAG toolkit for Python, powered by Rust.

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

## Features

- **Fast**: 10-100x faster than pure Python implementations
- **Complete RAG Toolkit**: Text splitting, reranking, evaluation
- **Production Ready**: Persistence, namespaces, server mode
- **Pythonic API**: Type hints, familiar patterns
- **Zero Config**: Works out of the box

## Documentation

See the main repository documentation:

- [Quick Start](../QUICKSTART.md)
- [Complete Documentation](../MASTER-DOCUMENTATION.md)
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

MIT License - see LICENSE file for details
