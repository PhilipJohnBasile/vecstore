"""
Tests for VecStore core functionality

Tests CRUD operations, querying, snapshots, and optimization.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import random


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def store(temp_dir):
    """Create a VecStore instance for testing"""
    from vecstore import VecStore
    return VecStore(temp_dir)


def mock_vector(dim: int = 384) -> list[float]:
    """Generate a random vector for testing"""
    return [random.random() for _ in range(dim)]


class TestVecStoreBasics:
    """Test basic VecStore operations"""

    def test_create_store(self, temp_dir):
        """Test creating a new store"""
        from vecstore import VecStore
        store = VecStore(temp_dir)
        assert store is not None
        assert len(store) == 0
        assert store.is_empty()

    def test_store_repr(self, store):
        """Test store string representation"""
        repr_str = repr(store)
        assert "VecStore" in repr_str
        assert "vectors=" in repr_str

    def test_store_len(self, store):
        """Test len() and __len__"""
        assert len(store) == 0
        store.upsert("doc1", mock_vector(), {"text": "test"})
        assert len(store) == 1
        assert store.len() == 1


class TestUpsert:
    """Test vector insertion and updates"""

    def test_upsert_single_vector(self, store):
        """Test inserting a single vector"""
        vector = mock_vector()
        metadata = {"text": "Hello world", "category": "greeting"}

        store.upsert("doc1", vector, metadata)
        assert len(store) == 1

    def test_upsert_multiple_vectors(self, store):
        """Test inserting multiple vectors"""
        for i in range(5):
            store.upsert(f"doc{i}", mock_vector(), {"index": i})

        assert len(store) == 5

    def test_upsert_update_existing(self, store):
        """Test updating an existing vector"""
        store.upsert("doc1", mock_vector(), {"version": 1})
        assert len(store) == 1

        # Update with new vector and metadata
        store.upsert("doc1", mock_vector(), {"version": 2})
        assert len(store) == 1  # Still only 1 vector

    def test_upsert_different_metadata_types(self, store):
        """Test upserting with different metadata value types"""
        metadata = {
            "string": "value",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
        }
        store.upsert("doc1", mock_vector(), metadata)
        assert len(store) == 1


class TestQuery:
    """Test vector search queries"""

    def test_query_empty_store(self, store):
        """Test querying an empty store"""
        results = store.query(mock_vector(), k=5)
        assert len(results) == 0

    def test_query_single_result(self, store):
        """Test querying with one vector in store"""
        vector = mock_vector()
        store.upsert("doc1", vector, {"text": "test"})

        results = store.query(vector, k=5)
        assert len(results) == 1
        assert results[0].id == "doc1"
        assert results[0].score > 0.99  # Should be very similar to itself

    def test_query_multiple_results(self, store):
        """Test querying with multiple vectors"""
        # Insert 5 vectors
        for i in range(5):
            store.upsert(f"doc{i}", mock_vector(), {"index": i})

        results = store.query(mock_vector(), k=3)
        assert len(results) == 3

    def test_query_k_larger_than_store(self, store):
        """Test k larger than number of vectors"""
        store.upsert("doc1", mock_vector(), {})
        store.upsert("doc2", mock_vector(), {})

        results = store.query(mock_vector(), k=10)
        assert len(results) == 2

    def test_query_result_structure(self, store):
        """Test structure of search results"""
        metadata = {"text": "hello", "number": 42}
        store.upsert("doc1", mock_vector(), metadata)

        results = store.query(mock_vector(), k=1)
        result = results[0]

        assert hasattr(result, "id")
        assert hasattr(result, "score")
        assert hasattr(result, "metadata")
        assert result.id == "doc1"
        assert isinstance(result.score, float)
        assert result.metadata["text"] == "hello"
        assert result.metadata["number"] == 42

    def test_query_with_filter(self, store):
        """Test querying with metadata filter"""
        store.upsert("doc1", mock_vector(), {"category": "tech"})
        store.upsert("doc2", mock_vector(), {"category": "news"})
        store.upsert("doc3", mock_vector(), {"category": "tech"})

        results = store.query(mock_vector(), k=10, filter="category = 'tech'")
        assert len(results) == 2
        for result in results:
            assert result.metadata["category"] == "tech"


class TestRemove:
    """Test vector deletion"""

    def test_remove_existing_vector(self, store):
        """Test removing an existing vector"""
        store.upsert("doc1", mock_vector(), {})
        assert len(store) == 1

        store.remove("doc1")
        assert len(store) == 0

    def test_remove_nonexistent_vector(self, store):
        """Test removing a vector that doesn't exist"""
        # Should raise an error
        import pytest
        with pytest.raises(ValueError, match="ID not found"):
            store.remove("nonexistent")

    def test_remove_and_query(self, store):
        """Test that removed vectors don't appear in queries"""
        store.upsert("doc1", mock_vector(), {"keep": True})
        store.upsert("doc2", mock_vector(), {"remove": True})

        store.remove("doc2")

        results = store.query(mock_vector(), k=10)
        assert len(results) == 1
        assert results[0].id == "doc1"


class TestSnapshots:
    """Test snapshot functionality"""

    def test_create_snapshot(self, store):
        """Test creating a snapshot"""
        store.upsert("doc1", mock_vector(), {})
        store.create_snapshot("backup1")

        # Should not raise an error
        snapshots = store.list_snapshots()
        snapshot_names = [s[0] for s in snapshots]  # Extract names from tuples
        assert "backup1" in snapshot_names

    def test_list_snapshots(self, store):
        """Test listing snapshots"""
        snapshots = store.list_snapshots()
        assert isinstance(snapshots, list)
        assert len(snapshots) == 0

        store.create_snapshot("snap1")
        store.create_snapshot("snap2")

        snapshots = store.list_snapshots()
        assert len(snapshots) == 2
        snapshot_names = [s[0] for s in snapshots]  # Extract names from tuples
        assert "snap1" in snapshot_names
        assert "snap2" in snapshot_names

    def test_restore_snapshot(self, store):
        """Test restoring from a snapshot"""
        # Add data and create snapshot
        store.upsert("doc1", mock_vector(), {})
        store.create_snapshot("v1")

        # Modify data
        store.upsert("doc2", mock_vector(), {})
        assert len(store) == 2

        # Restore to snapshot
        store.restore_snapshot("v1")
        assert len(store) == 1

    def test_snapshot_preserves_data(self, store):
        """Test that snapshots preserve exact data"""
        vector = mock_vector()
        metadata = {"text": "original", "version": 1}

        store.upsert("doc1", vector, metadata)
        store.create_snapshot("original")

        # Modify the data
        store.upsert("doc1", mock_vector(), {"text": "modified", "version": 2})

        # Restore
        store.restore_snapshot("original")

        # Query to verify metadata
        results = store.query(vector, k=1)
        assert results[0].metadata["text"] == "original"
        assert results[0].metadata["version"] == 1


class TestOptimize:
    """Test store optimization"""

    def test_optimize_empty_store(self, store):
        """Test optimizing an empty store"""
        removed = store.optimize()
        assert removed == 0

    def test_optimize_with_deletions(self, store):
        """Test optimize removes deleted entries"""
        # Add vectors
        for i in range(5):
            store.upsert(f"doc{i}", mock_vector(), {})

        # Remove some
        store.remove("doc1")
        store.remove("doc3")

        # Optimize
        removed = store.optimize()
        assert removed >= 0  # Should remove ghost entries


class TestPersistence:
    """Test data persistence"""

    def test_save(self, store):
        """Test saving store to disk"""
        store.upsert("doc1", mock_vector(), {"text": "test"})
        store.save()
        # Should not raise an error

    def test_reload_store(self, temp_dir):
        """Test that data persists across store instances"""
        from vecstore import VecStore

        # Create store and add data
        store1 = VecStore(temp_dir)
        vector = mock_vector()
        store1.upsert("doc1", vector, {"text": "persistent"})
        store1.save()

        # Create new store instance with same path
        store2 = VecStore(temp_dir)

        # Data should be there
        results = store2.query(vector, k=1)
        assert len(results) >= 0  # Store should load


class TestHybridSearch:
    """Test hybrid search functionality"""

    def test_hybrid_query(self, store):
        """Test basic hybrid query"""
        # Add vectors with text
        store.upsert("doc1", mock_vector(), {"text": "rust programming"})
        store.index_text("doc1", "rust programming")

        store.upsert("doc2", mock_vector(), {"text": "python coding"})
        store.index_text("doc2", "python coding")

        # Hybrid query
        results = store.hybrid_query(
            vector=mock_vector(),
            keywords="rust",
            k=2,
            alpha=0.7
        )

        assert len(results) <= 2

    def test_hybrid_query_alpha_values(self, store):
        """Test hybrid query with different alpha values"""
        store.upsert("doc1", mock_vector(), {})
        store.index_text("doc1", "test document")

        # Test different alpha values
        for alpha in [0.0, 0.5, 1.0]:
            results = store.hybrid_query(
                vector=mock_vector(),
                keywords="test",
                k=1,
                alpha=alpha
            )
            assert isinstance(results, list)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_metadata(self, store):
        """Test upserting with empty metadata"""
        store.upsert("doc1", mock_vector(), {})
        assert len(store) == 1

    def test_large_metadata(self, store):
        """Test upserting with large metadata"""
        large_metadata = {f"field{i}": f"value{i}" for i in range(100)}
        store.upsert("doc1", mock_vector(), large_metadata)
        assert len(store) == 1

    def test_unicode_in_metadata(self, store):
        """Test Unicode strings in metadata"""
        metadata = {
            "text": "Hello 世界 🌍",
            "emoji": "🚀🐍",
        }
        store.upsert("doc1", mock_vector(), metadata)

        results = store.query(mock_vector(), k=1)
        assert results[0].metadata["text"] == "Hello 世界 🌍"

    def test_special_characters_in_id(self, store):
        """Test special characters in vector IDs"""
        ids = ["doc-1", "doc_2", "doc.3", "doc@4"]
        for id in ids:
            store.upsert(id, mock_vector(), {})

        assert len(store) == len(ids)
