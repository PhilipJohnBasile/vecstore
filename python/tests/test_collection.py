"""
Tests for VecDatabase and Collection functionality

Tests multi-tenant collections, isolation, and collection management.
"""

import pytest
import tempfile
import shutil
import random


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def db(temp_dir):
    """Create a VecDatabase instance for testing"""
    from vecstore import VecDatabase
    return VecDatabase(temp_dir)


def mock_vector(dim: int = 384) -> list[float]:
    """Generate a random vector for testing"""
    return [random.random() for _ in range(dim)]


class TestVecDatabase:
    """Test VecDatabase operations"""

    def test_create_database(self, temp_dir):
        """Test creating a new database"""
        from vecstore import VecDatabase
        db = VecDatabase(temp_dir)
        assert db is not None

        collections = db.list_collections()
        assert len(collections) == 0

    def test_database_repr(self, db):
        """Test database string representation"""
        repr_str = repr(db)
        assert "VecDatabase" in repr_str
        assert "collections=" in repr_str

    def test_create_collection(self, db):
        """Test creating a single collection"""
        collection = db.create_collection("test")
        assert collection is not None
        assert collection.name() == "test"

        collections = db.list_collections()
        assert len(collections) == 1
        assert "test" in collections

    def test_create_multiple_collections(self, db):
        """Test creating multiple collections"""
        names = ["docs", "users", "products", "logs"]

        for name in names:
            db.create_collection(name)

        collections = db.list_collections()
        assert len(collections) == len(names)
        for name in names:
            assert name in collections

    def test_get_existing_collection(self, db):
        """Test getting an existing collection"""
        db.create_collection("test")

        collection = db.get_collection("test")
        assert collection is not None
        assert collection.name() == "test"

    def test_get_nonexistent_collection(self, db):
        """Test getting a collection that doesn't exist"""
        collection = db.get_collection("nonexistent")
        assert collection is None

    def test_delete_collection(self, db):
        """Test deleting a collection"""
        db.create_collection("test")
        assert len(db.list_collections()) == 1

        db.delete_collection("test")
        assert len(db.list_collections()) == 0

    def test_delete_nonexistent_collection(self, db):
        """Test deleting a collection that doesn't exist"""
        # Should raise an error
        import pytest
        with pytest.raises(ValueError, match="Namespace not found"):
            db.delete_collection("nonexistent")


class TestCollection:
    """Test Collection operations"""

    def test_collection_name(self, db):
        """Test getting collection name"""
        collection = db.create_collection("my_collection")
        assert collection.name() == "my_collection"

    def test_collection_repr(self, db):
        """Test collection string representation"""
        collection = db.create_collection("test")
        repr_str = repr(collection)
        assert "Collection" in repr_str
        assert "test" in repr_str

    def test_collection_upsert(self, db):
        """Test upserting vectors in a collection"""
        collection = db.create_collection("test")

        vector = mock_vector()
        metadata = {"text": "hello"}

        collection.upsert("doc1", vector, metadata)
        assert collection.count() == 1

    def test_collection_query(self, db):
        """Test querying a collection"""
        collection = db.create_collection("test")

        vector = mock_vector()
        collection.upsert("doc1", vector, {"text": "test"})

        results = collection.query(vector, k=5)
        assert len(results) == 1
        assert results[0].id == "doc1"

    def test_collection_delete(self, db):
        """Test deleting from a collection"""
        collection = db.create_collection("test")

        collection.upsert("doc1", mock_vector(), {})
        assert collection.count() == 1

        collection.delete("doc1")
        assert collection.count() == 0

    def test_collection_count(self, db):
        """Test counting vectors in a collection"""
        collection = db.create_collection("test")
        assert collection.count() == 0

        for i in range(5):
            collection.upsert(f"doc{i}", mock_vector(), {})

        assert collection.count() == 5

    def test_collection_stats(self, db):
        """Test getting collection statistics"""
        collection = db.create_collection("test")

        collection.upsert("doc1", mock_vector(dim=128), {})

        stats = collection.stats()
        assert isinstance(stats, dict)
        assert "vector_count" in stats
        assert "active_count" in stats
        assert "deleted_count" in stats
        assert "dimension" in stats
        assert stats["vector_count"] >= 1


class TestCollectionIsolation:
    """Test that collections are isolated from each other"""

    def test_collections_are_isolated(self, db):
        """Test that data in one collection doesn't appear in another"""
        coll1 = db.create_collection("coll1")
        coll2 = db.create_collection("coll2")

        # Add data to collection 1
        vector = mock_vector()
        coll1.upsert("doc1", vector, {"collection": "coll1"})

        # Query collection 2 (should be empty)
        results = coll2.query(vector, k=10)
        assert len(results) == 0

        # Query collection 1 (should find the vector)
        results = coll1.query(vector, k=10)
        assert len(results) == 1
        assert results[0].id == "doc1"

    def test_same_id_different_collections(self, db):
        """Test that same ID can exist in different collections"""
        coll1 = db.create_collection("coll1")
        coll2 = db.create_collection("coll2")

        # Use same ID in both collections
        coll1.upsert("doc1", mock_vector(), {"source": "coll1"})
        coll2.upsert("doc1", mock_vector(), {"source": "coll2"})

        # Each collection should have 1 vector
        assert coll1.count() == 1
        assert coll2.count() == 1

        # Query each collection
        results1 = coll1.query(mock_vector(), k=1)
        results2 = coll2.query(mock_vector(), k=1)

        assert results1[0].metadata["source"] == "coll1"
        assert results2[0].metadata["source"] == "coll2"

    def test_deleting_one_collection_preserves_others(self, db):
        """Test that deleting one collection doesn't affect others"""
        coll1 = db.create_collection("keep")
        coll2 = db.create_collection("delete")

        # Add data to both
        coll1.upsert("doc1", mock_vector(), {})
        coll2.upsert("doc2", mock_vector(), {})

        # Delete one collection
        db.delete_collection("delete")

        # First collection should still exist
        collections = db.list_collections()
        assert "keep" in collections
        assert "delete" not in collections

        # Data in first collection should still be there
        assert coll1.count() == 1


class TestMultiCollectionWorkflow:
    """Test realistic multi-collection workflows"""

    def test_multi_tenant_scenario(self, db):
        """Test multi-tenant use case"""
        # Create collections for different tenants
        org1 = db.create_collection("org_alpha")
        org2 = db.create_collection("org_beta")

        # Add data for org_alpha
        org1.upsert("doc1", mock_vector(), {"tenant": "alpha", "doc": "A"})
        org1.upsert("doc2", mock_vector(), {"tenant": "alpha", "doc": "B"})

        # Add data for org_beta
        org2.upsert("doc1", mock_vector(), {"tenant": "beta", "doc": "X"})
        org2.upsert("doc2", mock_vector(), {"tenant": "beta", "doc": "Y"})

        # Each tenant sees only their data
        assert org1.count() == 2
        assert org2.count() == 2

        # Verify tenant isolation
        results = org1.query(mock_vector(), k=10)
        for result in results:
            assert result.metadata["tenant"] == "alpha"

    def test_different_document_types(self, db):
        """Test organizing different document types in collections"""
        articles = db.create_collection("articles")
        code = db.create_collection("code_snippets")
        images = db.create_collection("image_embeddings")

        # Add different types of content
        articles.upsert("art1", mock_vector(), {"type": "article"})
        code.upsert("code1", mock_vector(), {"type": "code"})
        images.upsert("img1", mock_vector(), {"type": "image"})

        # Each collection has its own data
        assert articles.count() == 1
        assert code.count() == 1
        assert images.count() == 1

        # Total collections
        assert len(db.list_collections()) == 3


class TestCollectionEdgeCases:
    """Test edge cases for collections"""

    def test_collection_name_with_special_chars(self, db):
        """Test collection names with special characters"""
        # These should work
        valid_names = ["test-collection", "test_collection", "test123"]

        for name in valid_names:
            collection = db.create_collection(name)
            assert collection.name() == name

    def test_empty_collection_query(self, db):
        """Test querying an empty collection"""
        collection = db.create_collection("empty")
        results = collection.query(mock_vector(), k=5)
        assert len(results) == 0

    def test_collection_with_filter(self, db):
        """Test collection queries with filters"""
        collection = db.create_collection("test")

        collection.upsert("doc1", mock_vector(), {"category": "A"})
        collection.upsert("doc2", mock_vector(), {"category": "B"})
        collection.upsert("doc3", mock_vector(), {"category": "A"})

        results = collection.query(mock_vector(), k=10, filter="category = 'A'")
        assert len(results) == 2
        for result in results:
            assert result.metadata["category"] == "A"
