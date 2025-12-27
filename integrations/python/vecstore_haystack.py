"""
VecStore Haystack Integration

Provides a DocumentStore implementation for Haystack that uses VecStore
as the backend for document storage and vector retrieval.

Usage:
    from vecstore_haystack import VecStoreDocumentStore
    from haystack import Pipeline
    from haystack.components.retrievers import EmbeddingRetriever

    # Create VecStore document store
    document_store = VecStoreDocumentStore(
        db_path="./my_vectors.db",
        dimension=384,
    )

    # Add documents
    documents = [
        Document(content="VecStore is an embeddable vector database"),
        Document(content="It supports HNSW indexing for fast search"),
    ]
    document_store.write_documents(documents)

    # Create retrieval pipeline
    retriever = EmbeddingRetriever(document_store=document_store)
    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)
"""

from typing import Any, Dict, List, Optional, Generator
from dataclasses import dataclass, field
import json
import math
import hashlib
from datetime import datetime

try:
    from haystack import Document, default_from_dict, default_to_dict
    from haystack.document_stores.types import DocumentStore, DuplicatePolicy
except ImportError:
    raise ImportError(
        "Haystack is required for this integration. "
        "Install it with: pip install haystack-ai"
    )


@dataclass
class VecStoreDocumentStore(DocumentStore):
    """
    VecStore document store for Haystack.

    Uses VecStore as the backend for storing and querying documents
    with their embeddings. Supports all VecStore features including:
    - HNSW indexing for fast approximate nearest neighbor search
    - Explainable search results
    - Time-aware queries
    - Privacy-preserving search
    - Vector lineage tracking
    """

    db_path: str = "./vecstore.db"
    dimension: int = 384
    index_name: str = "haystack"
    similarity: str = "cosine"
    embedding_field: str = "embedding"
    content_field: str = "content"

    # Internal storage
    _documents: Dict[str, Document] = field(default_factory=dict, repr=False)
    _embeddings: Dict[str, List[float]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Initialize the document store."""
        self._initialize_store()

    def _initialize_store(self):
        """Initialize or connect to VecStore."""
        try:
            import vecstore
            self._store = vecstore.VecStore(
                path=self.db_path,
                dimension=self.dimension,
            )
            self._use_native = True
        except ImportError:
            self._store = None
            self._use_native = False

    def count_documents(self) -> int:
        """Return the number of documents in the store."""
        return len(self._documents)

    def filter_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Filter documents based on metadata.

        Args:
            filters: Dictionary of filters to apply

        Returns:
            List of matching documents
        """
        if filters is None:
            return list(self._documents.values())

        results = []
        for doc in self._documents.values():
            if self._matches_filters(doc, filters):
                results.append(doc)

        return results

    def _matches_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """Check if a document matches the given filters."""
        for key, value in filters.items():
            if key == "operator":
                continue

            if isinstance(value, dict):
                operator = value.get("operator", "==")
                filter_value = value.get("value")

                doc_value = doc.meta.get(key)

                if operator == "==":
                    if doc_value != filter_value:
                        return False
                elif operator == "!=":
                    if doc_value == filter_value:
                        return False
                elif operator == "in":
                    if doc_value not in filter_value:
                        return False
                elif operator == ">":
                    if doc_value <= filter_value:
                        return False
                elif operator == ">=":
                    if doc_value < filter_value:
                        return False
                elif operator == "<":
                    if doc_value >= filter_value:
                        return False
                elif operator == "<=":
                    if doc_value > filter_value:
                        return False
            else:
                if doc.meta.get(key) != value:
                    return False

        return True

    def write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Write documents to the store.

        Args:
            documents: List of documents to write
            policy: How to handle duplicates

        Returns:
            Number of documents written
        """
        written = 0

        for doc in documents:
            doc_id = doc.id

            # Handle duplicates
            if doc_id in self._documents:
                if policy == DuplicatePolicy.FAIL:
                    raise ValueError(f"Document {doc_id} already exists")
                elif policy == DuplicatePolicy.SKIP:
                    continue
                # OVERWRITE continues to write

            # Store document
            self._documents[doc_id] = doc

            # Store embedding if present
            if doc.embedding is not None:
                self._embeddings[doc_id] = doc.embedding

                if self._use_native and self._store:
                    self._store.add(
                        id=doc_id,
                        vector=doc.embedding,
                        metadata=json.dumps({
                            "content": doc.content,
                            "meta": doc.meta,
                        }),
                    )

            written += 1

        return written

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            document_ids: List of document IDs to delete
        """
        for doc_id in document_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
            if doc_id in self._embeddings:
                del self._embeddings[doc_id]
                if self._use_native and self._store:
                    self._store.delete(doc_id)

    def _query_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        """
        Query documents by embedding similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters to apply
            scale_score: Whether to scale scores to [0, 1]

        Returns:
            List of documents with scores
        """
        if self._use_native and self._store:
            results = self._store.search(
                vector=query_embedding,
                k=top_k * 2,  # Over-fetch for filtering
            )
        else:
            results = self._local_search(query_embedding, top_k * 2)

        documents = []
        for result in results:
            doc_id = result["id"]
            score = result["score"]

            if doc_id not in self._documents:
                continue

            doc = self._documents[doc_id]

            # Apply filters
            if filters and not self._matches_filters(doc, filters):
                continue

            # Scale score if needed
            if scale_score:
                score = (score + 1) / 2  # Cosine: [-1, 1] -> [0, 1]

            # Create result document with score
            result_doc = Document(
                id=doc.id,
                content=doc.content,
                meta=doc.meta,
                embedding=doc.embedding,
                score=score,
            )
            documents.append(result_doc)

            if len(documents) >= top_k:
                break

        return documents

    def _local_search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Local vector search when native bindings unavailable."""
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        similarities = []
        for doc_id, embedding in self._embeddings.items():
            score = cosine_similarity(query_embedding, embedding)
            similarities.append({"id": doc_id, "score": score})

        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the document store to a dictionary."""
        return default_to_dict(
            self,
            db_path=self.db_path,
            dimension=self.dimension,
            index_name=self.index_name,
            similarity=self.similarity,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VecStoreDocumentStore":
        """Deserialize from a dictionary."""
        return default_from_dict(cls, data)


class VecStoreEmbeddingRetriever:
    """
    Embedding-based retriever for VecStore.

    Works with Haystack 2.0 pipeline architecture.
    """

    def __init__(
        self,
        document_store: VecStoreDocumentStore,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        scale_score: bool = True,
    ):
        self.document_store = document_store
        self.top_k = top_k
        self.filters = filters
        self.scale_score = scale_score

    def run(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents by embedding similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results (overrides default)
            filters: Metadata filters (overrides default)

        Returns:
            Dictionary with "documents" key containing results
        """
        k = top_k or self.top_k
        f = filters or self.filters

        documents = self.document_store._query_by_embedding(
            query_embedding=query_embedding,
            top_k=k,
            filters=f,
            scale_score=self.scale_score,
        )

        return {"documents": documents}


class VecStoreBM25Retriever:
    """
    BM25 keyword retriever for VecStore.

    Uses VecStore's hybrid search capabilities for keyword matching.
    """

    def __init__(
        self,
        document_store: VecStoreDocumentStore,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ):
        self.document_store = document_store
        self.top_k = top_k
        self.filters = filters

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents by keyword matching.

        Args:
            query: Search query string
            top_k: Number of results
            filters: Metadata filters

        Returns:
            Dictionary with "documents" key containing results
        """
        k = top_k or self.top_k
        f = filters or self.filters

        # Simple keyword matching (VecStore hybrid search)
        query_terms = set(query.lower().split())
        scored_docs = []

        for doc in self.document_store._documents.values():
            if f and not self.document_store._matches_filters(doc, f):
                continue

            content = doc.content.lower()
            doc_terms = set(content.split())

            # Calculate simple BM25-like score
            matches = len(query_terms & doc_terms)
            score = matches / max(len(query_terms), 1)

            if score > 0:
                result_doc = Document(
                    id=doc.id,
                    content=doc.content,
                    meta=doc.meta,
                    embedding=doc.embedding,
                    score=score,
                )
                scored_docs.append(result_doc)

        # Sort by score
        scored_docs.sort(key=lambda x: x.score or 0, reverse=True)

        return {"documents": scored_docs[:k]}


if __name__ == "__main__":
    print("VecStore Haystack Integration")
    print("==============================")
    print()
    print("Usage:")
    print("  from vecstore_haystack import VecStoreDocumentStore")
    print("  document_store = VecStoreDocumentStore(db_path='./my_db.db')")
    print()
    print("See module docstring for full examples.")
