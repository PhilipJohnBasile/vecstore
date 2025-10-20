"""
Type stubs for vecstore

This file provides type hints for IDE autocomplete and type checking.
"""

from typing import Any, Dict, List, Optional

class VecStore:
    """
    Vector store for similarity search with HNSW indexing.

    Example:
        >>> store = VecStore("./my_db")
        >>> store.upsert("doc1", [0.1, 0.2, 0.3], {"text": "Hello"})
        >>> results = store.query([0.1, 0.2, 0.3], k=5)
    """

    def __init__(self, path: str) -> None:
        """
        Create or open a vector store at the given path.

        Args:
            path: Directory path for the vector store
        """
        ...

    def upsert(
        self,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Insert or update a vector with metadata.

        Args:
            id: Unique identifier for the vector
            vector: Embedding vector (list of floats)
            metadata: Associated metadata (dict)
        """
        ...

    def remove(self, id: str) -> None:
        """
        Remove a vector by ID.

        Args:
            id: Vector ID to remove
        """
        ...

    def query(
        self,
        vector: List[float],
        k: int,
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            vector: Query vector
            k: Number of results to return
            filter: Optional SQL-like filter (e.g., "category = 'tech'")

        Returns:
            List of search results
        """
        ...

    def hybrid_query(
        self,
        vector: List[float],
        keywords: str,
        k: int,
        alpha: float,
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity and keyword matching.

        Args:
            vector: Query vector
            keywords: Search keywords
            k: Number of results to return
            alpha: Weight for vector vs keyword (0.0 = pure keyword, 1.0 = pure vector)
            filter: Optional SQL-like filter

        Returns:
            List of search results
        """
        ...

    def index_text(self, id: str, text: str) -> None:
        """
        Index text for keyword search (required for hybrid_query).

        Args:
            id: Vector ID
            text: Text content to index
        """
        ...

    def save(self) -> None:
        """Save the store to disk."""
        ...

    def len(self) -> int:
        """Get the number of vectors in the store."""
        ...

    def is_empty(self) -> bool:
        """Check if the store is empty."""
        ...

    def create_snapshot(self, name: str) -> None:
        """
        Create a named snapshot.

        Args:
            name: Snapshot name
        """
        ...

    def restore_snapshot(self, name: str) -> None:
        """
        Restore from a named snapshot.

        Args:
            name: Snapshot name
        """
        ...

    def list_snapshots(self) -> List[str]:
        """List all available snapshots."""
        ...

    def optimize(self) -> int:
        """
        Optimize the index by removing deleted entries.

        Returns:
            Number of entries removed
        """
        ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class VecDatabase:
    """
    Multi-collection database for managing isolated vector stores.

    Example:
        >>> db = VecDatabase("./my_db")
        >>> collection = db.create_collection("documents")
        >>> collection.upsert("doc1", [0.1, 0.2, 0.3], {"text": "Hello"})
    """

    def __init__(self, path: str) -> None:
        """
        Create or open a database at the given path.

        Args:
            path: Directory path for the database
        """
        ...

    def create_collection(self, name: str) -> Collection:
        """
        Create a new collection.

        Args:
            name: Collection name (must be unique)

        Returns:
            Collection object
        """
        ...

    def get_collection(self, name: str) -> Optional[Collection]:
        """
        Get an existing collection.

        Args:
            name: Collection name

        Returns:
            Collection object or None if not found
        """
        ...

    def list_collections(self) -> List[str]:
        """List all collection names."""
        ...

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection and all its data.

        Args:
            name: Collection name
        """
        ...

    def __repr__(self) -> str: ...

class Collection:
    """
    Isolated vector collection within a database.

    Example:
        >>> collection = db.create_collection("docs")
        >>> collection.upsert("doc1", [0.1, 0.2], {"text": "Hello"})
        >>> results = collection.query([0.1, 0.2], k=5)
    """

    def name(self) -> str:
        """Get the collection name."""
        ...

    def upsert(
        self,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Insert or update a vector.

        Args:
            id: Unique identifier
            vector: Embedding vector
            metadata: Associated metadata
        """
        ...

    def query(
        self,
        vector: List[float],
        k: int,
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            vector: Query vector
            k: Number of results
            filter: Optional SQL-like filter

        Returns:
            List of search results
        """
        ...

    def delete(self, id: str) -> None:
        """
        Delete a vector by ID.

        Args:
            id: Vector ID
        """
        ...

    def count(self) -> int:
        """Get the number of vectors in the collection."""
        ...

    def stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dictionary with stats (vector_count, active_count, etc.)
        """
        ...

    def __repr__(self) -> str: ...

class SearchResult:
    """
    Search result from a query.

    Attributes:
        id: Vector ID
        score: Similarity score
        metadata: Associated metadata
    """

    id: str
    score: float
    metadata: Dict[str, Any]

    def __repr__(self) -> str: ...

class Query:
    """
    Vector search query.

    Attributes:
        vector: Query vector
        k: Number of results
        filter: Optional filter expression
    """

    vector: List[float]
    k: int
    filter: Optional[str]

    def __init__(
        self,
        vector: List[float],
        k: int,
        filter: Optional[str] = None,
    ) -> None: ...

class HybridQuery:
    """
    Hybrid search query combining vector and keyword search.

    Attributes:
        vector: Query vector
        keywords: Search keywords
        k: Number of results
        alpha: Weight for vector vs keyword
        filter: Optional filter expression
    """

    vector: List[float]
    keywords: str
    k: int
    alpha: float
    filter: Optional[str]

    def __init__(
        self,
        vector: List[float],
        keywords: str,
        k: int,
        alpha: float,
        filter: Optional[str] = None,
    ) -> None: ...

class RecursiveCharacterTextSplitter:
    """
    Text splitter that recursively splits on natural boundaries.

    Splits on paragraphs, then sentences, then words, then characters,
    trying to keep chunks under the specified size.

    Example:
        >>> splitter = RecursiveCharacterTextSplitter(500, 50)
        >>> chunks = splitter.split_text("Long document...")
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Create a new text splitter.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        ...

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        ...

    def __repr__(self) -> str: ...

__version__: str
__all__: List[str]
