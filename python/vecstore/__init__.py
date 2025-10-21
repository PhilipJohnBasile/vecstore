"""
VecStore - High-performance vector database with RAG toolkit

A lightweight, fast vector database built in Rust with Python bindings.
Perfect for RAG (Retrieval-Augmented Generation) applications.

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
)

__version__ = "0.1.0"

__all__ = [
    "VecStore",
    "VecDatabase",
    "Collection",
    "Query",
    "HybridQuery",
    "SearchResult",
    "RecursiveCharacterTextSplitter",
]
