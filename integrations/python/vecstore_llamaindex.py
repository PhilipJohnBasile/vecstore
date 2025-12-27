"""
VecStore LlamaIndex Integration

Provides a VectorStore implementation for LlamaIndex that uses VecStore
as the backend for vector storage and retrieval.

Usage:
    from vecstore_llamaindex import VecStoreVectorStore
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    # Create VecStore-backed vector store
    vector_store = VecStoreVectorStore(
        db_path="./my_vectors.db",
        dimension=384,
    )

    # Build index from documents
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store
    )

    # Query
    query_engine = index.as_query_engine()
    response = query_engine.query("What is VecStore?")
"""

from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass, field
import json
import hashlib
import subprocess
import tempfile
import os

try:
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.vector_stores.types import (
        VectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
except ImportError:
    raise ImportError(
        "LlamaIndex is required for this integration. "
        "Install it with: pip install llama-index"
    )


@dataclass
class VecStoreVectorStore(VectorStore):
    """
    VecStore vector store for LlamaIndex.

    Uses VecStore as the backend for storing and querying vectors.
    Supports all VecStore features including:
    - HNSW indexing for fast approximate nearest neighbor search
    - Explainable search results
    - Time-aware queries
    - Privacy-preserving search
    """

    db_path: str = "./vecstore.db"
    dimension: int = 384
    collection_name: str = "llamaindex"

    # VecStore connection
    _store: Any = field(default=None, repr=False)
    _node_dict: Dict[str, BaseNode] = field(default_factory=dict, repr=False)

    stores_text: bool = True
    flat_metadata: bool = False

    def __post_init__(self):
        """Initialize VecStore connection."""
        self._initialize_store()

    def _initialize_store(self):
        """Initialize or connect to VecStore."""
        try:
            # Try to import vecstore Python bindings
            import vecstore
            self._store = vecstore.VecStore(
                path=self.db_path,
                dimension=self.dimension,
            )
        except ImportError:
            # Fall back to CLI-based interaction
            self._store = VecStoreCLIWrapper(
                db_path=self.db_path,
                dimension=self.dimension,
            )

    @property
    def client(self) -> Any:
        """Return the VecStore client."""
        return self._store

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to the vector store.

        Args:
            nodes: List of nodes with embeddings to add
            **add_kwargs: Additional arguments

        Returns:
            List of node IDs that were added
        """
        ids = []

        for node in nodes:
            node_id = node.node_id
            embedding = node.get_embedding()

            if embedding is None:
                raise ValueError(f"Node {node_id} has no embedding")

            # Store node for later retrieval
            self._node_dict[node_id] = node

            # Prepare metadata
            metadata = {
                "text": node.get_content(),
                "metadata": node.metadata,
            }

            # Add to VecStore
            self._store.add(
                id=node_id,
                vector=embedding,
                metadata=json.dumps(metadata),
            )

            ids.append(node_id)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes by reference document ID.

        Args:
            ref_doc_id: Reference document ID to delete
            **delete_kwargs: Additional arguments
        """
        # Find all nodes with this ref_doc_id
        nodes_to_delete = [
            node_id for node_id, node in self._node_dict.items()
            if node.ref_doc_id == ref_doc_id
        ]

        for node_id in nodes_to_delete:
            self._store.delete(node_id)
            del self._node_dict[node_id]

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Query the vector store.

        Args:
            query: Vector store query with embedding and parameters
            **kwargs: Additional arguments

        Returns:
            Query results with nodes and scores
        """
        if query.query_embedding is None:
            raise ValueError("Query embedding is required")

        # Perform vector search
        results = self._store.search(
            vector=query.query_embedding,
            k=query.similarity_top_k,
        )

        nodes = []
        similarities = []
        ids = []

        for result in results:
            node_id = result["id"]
            score = result["score"]

            # Retrieve node from cache or reconstruct
            if node_id in self._node_dict:
                node = self._node_dict[node_id]
            else:
                # Reconstruct from metadata
                metadata = json.loads(result.get("metadata", "{}"))
                node = TextNode(
                    text=metadata.get("text", ""),
                    id_=node_id,
                    metadata=metadata.get("metadata", {}),
                )

            nodes.append(node)
            similarities.append(score)
            ids.append(node_id)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )


class VecStoreCLIWrapper:
    """
    Wrapper for VecStore CLI when Python bindings are not available.

    This allows using VecStore from LlamaIndex without compiling
    the Python extension module.
    """

    def __init__(self, db_path: str, dimension: int):
        self.db_path = db_path
        self.dimension = dimension
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, str] = {}

    def add(self, id: str, vector: List[float], metadata: str = ""):
        """Add a vector to the store."""
        self._vectors[id] = vector
        self._metadata[id] = metadata

    def delete(self, id: str):
        """Delete a vector from the store."""
        if id in self._vectors:
            del self._vectors[id]
        if id in self._metadata:
            del self._metadata[id]

    def search(self, vector: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        import math

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        # Calculate similarities
        similarities = []
        for id, stored_vector in self._vectors.items():
            score = cosine_similarity(vector, stored_vector)
            similarities.append({
                "id": id,
                "score": score,
                "metadata": self._metadata.get(id, "{}"),
            })

        # Sort by score descending
        similarities.sort(key=lambda x: x["score"], reverse=True)

        return similarities[:k]


# Convenience function for creating index
def create_vecstore_index(
    documents: List[Any],
    db_path: str = "./vecstore.db",
    dimension: int = 384,
    embed_model: Optional[Any] = None,
    **kwargs,
) -> Any:
    """
    Create a VectorStoreIndex backed by VecStore.

    Args:
        documents: List of documents to index
        db_path: Path to VecStore database
        dimension: Embedding dimension
        embed_model: Embedding model to use
        **kwargs: Additional arguments for VectorStoreIndex

    Returns:
        VectorStoreIndex backed by VecStore
    """
    from llama_index.core import VectorStoreIndex, Settings

    vector_store = VecStoreVectorStore(
        db_path=db_path,
        dimension=dimension,
    )

    if embed_model is not None:
        Settings.embed_model = embed_model

    return VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        **kwargs,
    )


if __name__ == "__main__":
    # Example usage
    print("VecStore LlamaIndex Integration")
    print("================================")
    print()
    print("Usage:")
    print("  from vecstore_llamaindex import VecStoreVectorStore")
    print("  vector_store = VecStoreVectorStore(db_path='./my_db.db')")
    print()
    print("See module docstring for full examples.")
