"""
Multi-Collection Database Example

This example demonstrates using VecDatabase to manage multiple
isolated collections, similar to how you'd use ChromaDB or Qdrant.

Use cases:
- Multi-tenant applications (one collection per user/organization)
- Different document types (articles, code, images)
- Isolated namespaces for different projects
"""

import random
from vecstore import VecDatabase


def mock_embed(text: str, dim: int = 384) -> list[float]:
    """Mock embedding function"""
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(dim)]


def main():
    print("=" * 60)
    print("VecStore - Multi-Collection Example")
    print("=" * 60)

    # Create database
    print("\n1. Creating multi-collection database...")
    db = VecDatabase("./multi_collection_db")
    print(f"   ✓ Database created: {db}")

    # Create separate collections for different data types
    print("\n2. Creating collections...")

    docs_collection = db.create_collection("documents")
    print(f"   ✓ Created 'documents' collection: {docs_collection}")

    code_collection = db.create_collection("code_snippets")
    print(f"   ✓ Created 'code_snippets' collection: {code_collection}")

    notes_collection = db.create_collection("personal_notes")
    print(f"   ✓ Created 'personal_notes' collection: {notes_collection}")

    # List all collections
    collections = db.list_collections()
    print(f"\n3. All collections: {collections}")

    # Add data to each collection
    print("\n4. Adding data to collections...")

    # Documents collection
    docs_collection.upsert(
        "article_1",
        mock_embed("Rust programming language overview"),
        {"title": "Intro to Rust", "type": "article", "author": "Jane Doe"}
    )
    docs_collection.upsert(
        "article_2",
        mock_embed("Python for data science"),
        {"title": "Python DS", "type": "article", "author": "John Smith"}
    )
    print(f"   ✓ Added {docs_collection.count()} documents")

    # Code collection
    code_collection.upsert(
        "snippet_1",
        mock_embed("fn main() { println!(\"Hello\"); }"),
        {"language": "rust", "description": "Hello world in Rust"}
    )
    code_collection.upsert(
        "snippet_2",
        mock_embed("def hello(): print('Hello')"),
        {"language": "python", "description": "Hello function in Python"}
    )
    print(f"   ✓ Added {code_collection.count()} code snippets")

    # Notes collection
    notes_collection.upsert(
        "note_1",
        mock_embed("Remember to buy groceries"),
        {"category": "todo", "priority": "high"}
    )
    print(f"   ✓ Added {notes_collection.count()} notes")

    # Query each collection independently
    print("\n5. Querying collections...")

    # Query documents
    query_vec = mock_embed("programming tutorials")
    doc_results = docs_collection.query(query_vec, k=2)
    print(f"\n   Documents matching 'programming tutorials':")
    for result in doc_results:
        print(f"   - {result.id}: {result.metadata['title']} (score: {result.score:.4f})")

    # Query code
    code_query = mock_embed("rust code examples")
    code_results = code_collection.query(code_query, k=2)
    print(f"\n   Code snippets matching 'rust code examples':")
    for result in code_results:
        print(f"   - {result.id}: {result.metadata['description']} (score: {result.score:.4f})")

    # Get collection statistics
    print("\n6. Collection statistics...")
    for name in collections:
        coll = db.get_collection(name)
        if coll:
            stats = coll.stats()
            print(f"\n   {name}:")
            print(f"   - Vectors: {stats['vector_count']}")
            print(f"   - Active: {stats['active_count']}")
            print(f"   - Deleted: {stats['deleted_count']}")
            print(f"   - Dimension: {stats['dimension']}")

    # Delete a collection
    print("\n7. Deleting 'personal_notes' collection...")
    db.delete_collection("personal_notes")
    print(f"   ✓ Collections remaining: {db.list_collections()}")

    print("\n" + "=" * 60)
    print("✓ Multi-collection demo complete!")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree("./multi_collection_db", ignore_errors=True)
    print("\n✓ Cleaned up demo database")


if __name__ == "__main__":
    main()
