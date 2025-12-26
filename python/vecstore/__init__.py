"""
VecStore - High-performance vector database with RAG toolkit

A lightweight, fast vector database built in Rust with Python bindings.
Well-suited for RAG (Retrieval-Augmented Generation) applications.

Basic usage:
    >>> from vecstore import VecStore
    >>> store = VecStore("./my_db")
    >>> store.upsert("doc1", [0.1, 0.2, 0.3], {"text": "Hello world"})
    >>> results = store.query([0.1, 0.2, 0.3], k=5)

Multi-collection usage:
    >>> from vecstore import VecDatabase
    >>> db = VecDatabase("./my_db")
    >>> collection = db.create_collection("documents")
    >>> collection.upsert("doc1", [0.1, 0.2, 0.3], {"text": "Hello"})

LangChain integration:
    >>> from vecstore import LangChainVectorStore, Document
    >>> store = LangChainVectorStore("./my_db")
    >>> store.add_embeddings(
    ...     texts=["Hello world"],
    ...     embeddings=[[0.1, 0.2, 0.3]],
    ...     metadatas=[{"source": "doc1"}]
    ... )
    >>> results = store.similarity_search_by_vector([0.1, 0.2, 0.3], k=5)

Text splitting:
    >>> from vecstore import RecursiveCharacterTextSplitter
    >>> splitter = RecursiveCharacterTextSplitter(500, 50)
    >>> chunks = splitter.split_text("Long document text...")
"""

# Import all classes from the Rust module
from .vecstore import (
    VecStore,
    VecDatabase,
    Collection,
    Query,
    HybridQuery,
    SearchResult,
    RecursiveCharacterTextSplitter,
    # LangChain integration classes
    Document,
    ScoredDocument,
    LangChainVectorStore,
)

__version__ = "0.0.2"

__all__ = [
    # Core classes
    "VecStore",
    "VecDatabase",
    "Collection",
    "Query",
    "HybridQuery",
    "SearchResult",
    "RecursiveCharacterTextSplitter",
    # LangChain integration
    "Document",
    "ScoredDocument",
    "LangChainVectorStore",
]
