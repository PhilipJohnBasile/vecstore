"""
VecStore Python Integrations

This package provides integrations for popular Python ML frameworks:
- LlamaIndex: VecStoreVectorStore
- Haystack: VecStoreDocumentStore

Installation:
    pip install vecstore llama-index haystack-ai

Usage:
    # LlamaIndex
    from vecstore_llamaindex import VecStoreVectorStore, create_vecstore_index

    # Haystack
    from vecstore_haystack import VecStoreDocumentStore, VecStoreEmbeddingRetriever
"""

__version__ = "0.1.0"
__all__ = [
    "VecStoreVectorStore",
    "VecStoreDocumentStore",
    "VecStoreEmbeddingRetriever",
    "VecStoreBM25Retriever",
    "create_vecstore_index",
]
