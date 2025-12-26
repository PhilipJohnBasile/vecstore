"""
Embedding integration for VecStore.

This module provides optional embedding model support using sentence-transformers.
Install with: pip install sentence-transformers

Example:
    >>> from vecstore import VecStoreWithEmbeddings
    >>> store = VecStoreWithEmbeddings("./my_db", model_name="all-MiniLM-L6-v2")
    >>> store.add_texts(["Hello world", "Goodbye world"])
    >>> results = store.search("greeting", k=5)
"""

from typing import List, Optional, Dict, Any, Union
import hashlib

# Check for sentence-transformers availability
_SENTENCE_TRANSFORMERS_AVAILABLE = False
_SentenceTransformer = None

try:
    from sentence_transformers import SentenceTransformer as _ST
    _SentenceTransformer = _ST
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def is_embedding_available() -> bool:
    """Check if sentence-transformers is installed."""
    return _SENTENCE_TRANSFORMERS_AVAILABLE


class EmbeddingModel:
    """
    Wrapper around sentence-transformers for generating embeddings.

    Supported models (examples):
        - "all-MiniLM-L6-v2" (384 dims, fast, good quality)
        - "all-mpnet-base-v2" (768 dims, high quality)
        - "multi-qa-MiniLM-L6-cos-v1" (384 dims, optimized for Q&A)
        - "paraphrase-multilingual-MiniLM-L12-v2" (384 dims, multilingual)

    Example:
        >>> model = EmbeddingModel("all-MiniLM-L6-v2")
        >>> embeddings = model.encode(["Hello world", "How are you?"])
        >>> print(len(embeddings[0]))  # 384
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run model on ("cpu", "cuda", "mps", or None for auto)
            normalize_embeddings: Whether to L2-normalize embeddings (recommended for cosine similarity)
        """
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embedding support. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self._model = _SentenceTransformer(model_name, device=device)
        self._dimension = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # Compute a test embedding to get dimension
            test = self._model.encode(["test"], normalize_embeddings=self.normalize_embeddings)
            self._dimension = len(test[0])
        return self._dimension

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for encoding
            show_progress_bar: Show progress during encoding

        Returns:
            List of embedding vectors (list of floats)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        """
        Encode a single query string.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        return self.encode([query])[0]

    def __repr__(self) -> str:
        return f"EmbeddingModel(model='{self.model_name}', dim={self.dimension})"


class VecStoreWithEmbeddings:
    """
    VecStore with built-in embedding generation.

    This class combines VecStore with sentence-transformers for automatic
    embedding generation, providing a simpler API for text-based operations.

    Example:
        >>> store = VecStoreWithEmbeddings("./my_db")
        >>> store.add_texts(
        ...     texts=["Hello world", "Machine learning is great"],
        ...     metadatas=[{"source": "a"}, {"source": "b"}]
        ... )
        >>> results = store.search("artificial intelligence", k=5)
        >>> for doc in results:
        ...     print(f"{doc.page_content} (score: {doc.score:.3f})")
    """

    def __init__(
        self,
        path: str,
        model_name: str = EmbeddingModel.DEFAULT_MODEL,
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize store with embedding model.

        Args:
            path: Path to the vector store
            model_name: Embedding model name (default: all-MiniLM-L6-v2)
            device: Device for model ("cpu", "cuda", "mps", or None for auto)
            normalize_embeddings: Normalize embeddings for cosine similarity
        """
        # Import here to avoid circular imports
        from .vecstore import LangChainVectorStore

        self._store = LangChainVectorStore(path)
        self._embedding_model = EmbeddingModel(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings
        )

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Get the embedding model."""
        return self._embedding_model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._embedding_model.dimension

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> List[str]:
        """
        Add texts with automatic embedding generation.

        Args:
            texts: List of text contents
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs (auto-generated if not provided)
            batch_size: Batch size for embedding generation
            show_progress_bar: Show progress during embedding

        Returns:
            List of document IDs
        """
        # Generate embeddings
        embeddings = self._embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )

        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_id(text) for text in texts]

        # Add to store
        return self._store.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None
    ) -> List:
        """
        Search for similar documents by text query.

        Args:
            query: Query text (will be embedded automatically)
            k: Number of results to return
            filter: Optional filter expression

        Returns:
            List of ScoredDocument objects
        """
        query_embedding = self._embedding_model.encode_query(query)
        return self._store.similarity_search_by_vector(
            embedding=query_embedding,
            k=k,
            filter=filter
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None
    ) -> List:
        """
        Alias for search() - LangChain-compatible method name.
        """
        return self.search(query, k=k, filter=filter)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[str] = None
    ) -> List:
        """
        Search by pre-computed embedding vector.

        Args:
            embedding: Query embedding vector
            k: Number of results to return
            filter: Optional filter expression

        Returns:
            List of ScoredDocument objects
        """
        return self._store.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter
        )

    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        self._store.delete(ids)

    def save(self) -> None:
        """Save the store to disk."""
        self._store.save()

    def count(self) -> int:
        """Get number of vectors in the store."""
        return self._store.count()

    def _generate_id(self, text: str) -> str:
        """Generate a deterministic ID from text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        return (
            f"VecStoreWithEmbeddings("
            f"vectors={self.count()}, "
            f"model='{self._embedding_model.model_name}')"
        )


class LangChainVectorStoreWithEmbeddings(VecStoreWithEmbeddings):
    """
    Alias for VecStoreWithEmbeddings with LangChain-compatible naming.

    This provides the same functionality but with naming that aligns
    with LangChain's VectorStore interface conventions.
    """
    pass


# Convenience function to create store with embeddings
def create_store(
    path: str,
    model_name: str = EmbeddingModel.DEFAULT_MODEL,
    device: Optional[str] = None
) -> VecStoreWithEmbeddings:
    """
    Create a VecStore with built-in embedding support.

    This is a convenience function for quickly creating a store
    that can handle text directly.

    Args:
        path: Path to the vector store
        model_name: Embedding model name
        device: Device for model

    Returns:
        VecStoreWithEmbeddings instance

    Example:
        >>> store = create_store("./my_db")
        >>> store.add_texts(["Document 1", "Document 2"])
        >>> results = store.search("query")
    """
    return VecStoreWithEmbeddings(
        path=path,
        model_name=model_name,
        device=device
    )
