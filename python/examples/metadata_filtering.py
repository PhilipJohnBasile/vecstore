"""
Metadata Filtering Example

This example demonstrates using SQL-like filters to narrow down
search results based on metadata fields.

Supported filter operators:
- Equality: field = 'value'
- Inequality: field != 'value'
- Comparison: field > 10, field < 100
- AND: condition1 AND condition2
- OR: condition1 OR condition2
- IN: field IN ('val1', 'val2')
"""

import random
from vecstore import VecStore


def mock_embed(text: str) -> list[float]:
    """Mock embedding function"""
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(384)]


def main():
    print("=" * 60)
    print("VecStore - Metadata Filtering Example")
    print("=" * 60)

    # Create store
    print("\n1. Creating vector store...")
    store = VecStore("./filtering_demo_db")
    print(f"   ✓ Store created: {store}")

    # Add documents with rich metadata
    print("\n2. Adding documents with metadata...")
    documents = [
        {
            "id": "article_1",
            "text": "Introduction to Rust programming",
            "metadata": {
                "text": "Introduction to Rust programming",
                "category": "programming",
                "language": "Rust",
                "difficulty": "beginner",
                "year": 2023,
                "rating": 4.5,
                "author": "Alice",
            },
        },
        {
            "id": "article_2",
            "text": "Advanced Rust concurrency",
            "metadata": {
                "text": "Advanced Rust concurrency",
                "category": "programming",
                "language": "Rust",
                "difficulty": "advanced",
                "year": 2024,
                "rating": 4.8,
                "author": "Bob",
            },
        },
        {
            "id": "article_3",
            "text": "Python data science basics",
            "metadata": {
                "text": "Python data science basics",
                "category": "data-science",
                "language": "Python",
                "difficulty": "beginner",
                "year": 2023,
                "rating": 4.2,
                "author": "Carol",
            },
        },
        {
            "id": "article_4",
            "text": "Machine learning with Python",
            "metadata": {
                "text": "Machine learning with Python",
                "category": "data-science",
                "language": "Python",
                "difficulty": "intermediate",
                "year": 2024,
                "rating": 4.6,
                "author": "Alice",
            },
        },
        {
            "id": "article_5",
            "text": "Web development with JavaScript",
            "metadata": {
                "text": "Web development with JavaScript",
                "category": "web",
                "language": "JavaScript",
                "difficulty": "beginner",
                "year": 2023,
                "rating": 4.0,
                "author": "Bob",
            },
        },
    ]

    for doc in documents:
        vector = mock_embed(doc["text"])
        store.upsert(doc["id"], vector, doc["metadata"])
        print(f"   ✓ Added {doc['id']}")

    print(f"\n   Total documents: {len(store)}")

    # Demonstrate different filter types
    query_vector = mock_embed("programming tutorial")

    # Example 1: Simple equality filter
    print("\n3. Simple equality filter:")
    print("   Filter: category = 'programming'")
    results = store.query(query_vector, k=10, filter="category = 'programming'")
    print(f"   Results ({len(results)}):")
    for r in results:
        print(f"     - {r.id}: {r.metadata['text']} (category: {r.metadata['category']})")

    # Example 2: Comparison filter
    print("\n4. Comparison filter:")
    print("   Filter: year >= 2024")
    results = store.query(query_vector, k=10, filter="year >= 2024")
    print(f"   Results ({len(results)}):")
    for r in results:
        print(f"     - {r.id}: {r.metadata['text']} (year: {r.metadata['year']})")

    # Example 3: AND filter
    print("\n5. AND filter:")
    print("   Filter: language = 'Rust' AND difficulty = 'beginner'")
    results = store.query(
        query_vector,
        k=10,
        filter="language = 'Rust' AND difficulty = 'beginner'"
    )
    print(f"   Results ({len(results)}):")
    for r in results:
        print(f"     - {r.id}: {r.metadata['text']}")
        print(f"       Language: {r.metadata['language']}, Difficulty: {r.metadata['difficulty']}")

    # Example 4: OR filter
    print("\n6. OR filter:")
    print("   Filter: difficulty = 'beginner' OR difficulty = 'intermediate'")
    results = store.query(
        query_vector,
        k=10,
        filter="difficulty = 'beginner' OR difficulty = 'intermediate'"
    )
    print(f"   Results ({len(results)}):")
    for r in results:
        print(f"     - {r.id}: {r.metadata['difficulty']}")

    # Example 5: Complex filter with multiple conditions
    print("\n7. Complex filter:")
    print("   Filter: category = 'programming' AND year >= 2024 AND rating > 4.5")
    results = store.query(
        query_vector,
        k=10,
        filter="category = 'programming' AND year >= 2024 AND rating > 4.5"
    )
    print(f"   Results ({len(results)}):")
    for r in results:
        print(f"     - {r.id}: {r.metadata['text']}")
        print(f"       Year: {r.metadata['year']}, Rating: {r.metadata['rating']}")

    # Example 6: Filter by author
    print("\n8. Filter by author:")
    print("   Filter: author = 'Alice'")
    results = store.query(query_vector, k=10, filter="author = 'Alice'")
    print(f"   Results ({len(results)}):")
    for r in results:
        print(f"     - {r.id}: {r.metadata['text']} (by {r.metadata['author']})")

    # Use case examples
    print("\n" + "=" * 60)
    print("Common Use Cases:")
    print("=" * 60)
    print("""
1. Multi-tenant filtering:
   filter="tenant_id = 'org_123'"

2. Time-based filtering:
   filter="created_at >= '2024-01-01' AND created_at < '2024-12-31'"

3. Access control:
   filter="visibility = 'public' OR owner_id = 'user_456'"

4. Content type filtering:
   filter="content_type IN ('article', 'blog_post', 'tutorial')"

5. Quality filtering:
   filter="verified = true AND rating >= 4.0"

6. Language filtering:
   filter="language = 'en'"

7. Combination queries:
   filter="category = 'tech' AND published = true AND author_tier IN ('expert', 'professional')"
    """)

    print("\n" + "=" * 60)
    print("✓ Metadata filtering demo complete!")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree("./filtering_demo_db", ignore_errors=True)
    print("\n✓ Cleaned up demo database")


if __name__ == "__main__":
    main()
