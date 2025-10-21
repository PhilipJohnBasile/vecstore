"""
Hybrid Search Example

This example demonstrates combining vector similarity search with
keyword (BM25) search for better retrieval quality.

Hybrid search is useful when:
- You want both semantic and lexical matching
- Handling proper nouns, acronyms, or specific terms
- Improving recall by combining multiple signals
"""

import random
from vecstore import VecStore


def mock_embed(text: str) -> list[float]:
    """Mock embedding function"""
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(384)]


def main():
    print("=" * 60)
    print("VecStore - Hybrid Search Example")
    print("=" * 60)

    # Create store
    print("\n1. Creating vector store...")
    store = VecStore("./hybrid_search_db")
    print(f"   ✓ Store created: {store}")

    # Add documents with both vectors and text
    documents = [
        {
            "id": "doc1",
            "text": "Rust is a systems programming language focused on safety and performance.",
        },
        {
            "id": "doc2",
            "text": "Python is a high-level programming language known for simplicity and readability.",
        },
        {
            "id": "doc3",
            "text": "JavaScript is the language of the web, running in browsers and Node.js servers.",
        },
        {
            "id": "doc4",
            "text": "Go is a statically typed language designed at Google for building scalable systems.",
        },
        {
            "id": "doc5",
            "text": "Rust guarantees memory safety without garbage collection using ownership.",
        },
    ]

    print("\n2. Indexing documents...")
    for doc in documents:
        # Index vector
        vector = mock_embed(doc["text"])
        metadata = {"text": doc["text"], "indexed": True}
        store.upsert(doc["id"], vector, metadata)

        # Index text for keyword search
        store.index_text(doc["id"], doc["text"])
        print(f"   ✓ Indexed {doc['id']}")

    # Pure vector search
    print("\n3. Pure vector search:")
    query_text = "memory safe language"
    query_vec = mock_embed(query_text)

    vector_results = store.query(query_vec, k=3)
    print(f"\n   Query: '{query_text}'")
    print(f"   Top 3 results (vector only):")
    for i, result in enumerate(vector_results, 1):
        print(f"   {i}. {result.id} (score: {result.score:.4f})")
        print(f"      {result.metadata['text'][:70]}...")

    # Hybrid search (vector + keywords)
    print("\n4. Hybrid search (vector + keywords):")
    query_keywords = "Rust memory"
    alpha = 0.7  # 0.7 = 70% vector, 30% keyword

    hybrid_results = store.hybrid_query(
        vector=query_vec,
        keywords=query_keywords,
        k=3,
        alpha=alpha
    )

    print(f"\n   Query: '{query_text}'")
    print(f"   Keywords: '{query_keywords}'")
    print(f"   Alpha: {alpha} (70% vector, 30% keyword)")
    print(f"   Top 3 results (hybrid):")
    for i, result in enumerate(hybrid_results, 1):
        print(f"   {i}. {result.id} (score: {result.score:.4f})")
        print(f"      {result.metadata['text'][:70]}...")

    # Compare different alpha values
    print("\n5. Comparing different alpha values...")
    for alpha_val in [0.0, 0.5, 1.0]:
        results = store.hybrid_query(
            vector=query_vec,
            keywords="Rust",
            k=2,
            alpha=alpha_val
        )
        weight_desc = {
            0.0: "pure keyword",
            0.5: "balanced",
            1.0: "pure vector"
        }
        print(f"\n   Alpha={alpha_val} ({weight_desc[alpha_val]}):")
        for r in results:
            print(f"   - {r.id}: {r.score:.4f}")

    print("\n" + "=" * 60)
    print("✓ Hybrid search demo complete!")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree("./hybrid_search_db", ignore_errors=True)
    print("\n✓ Cleaned up demo database")


if __name__ == "__main__":
    main()
