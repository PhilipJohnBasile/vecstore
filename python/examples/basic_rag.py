"""
Basic RAG (Retrieval-Augmented Generation) Example

This example demonstrates:
1. Creating a vector store
2. Splitting text into chunks
3. Indexing chunks with embeddings
4. Querying for relevant chunks
5. Using results for RAG

Note: This example uses mock embeddings. In production, use a real
embedding model like sentence-transformers, OpenAI, or Cohere.
"""

import random
from vecstore import VecStore, RecursiveCharacterTextSplitter


def mock_embed(text: str) -> list[float]:
    """
    Mock embedding function for demonstration.

    In production, replace with a real embedding model:
    - sentence-transformers (e.g., all-MiniLM-L6-v2)
    - OpenAI embeddings (text-embedding-3-small)
    - Cohere embeddings
    """
    # Simple hash-based mock embedding (not for production!)
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(384)]


def main():
    print("=" * 60)
    print("VecStore - Basic RAG Example")
    print("=" * 60)

    # Step 1: Create vector store
    print("\n1. Creating vector store...")
    store = VecStore("./rag_demo_db")
    print(f"   ✓ Store created: {store}")

    # Step 2: Prepare document
    document = """
    Rust is a systems programming language that runs blazingly fast,
    prevents segfaults, and guarantees thread safety. It achieves this
    through a novel ownership system that enforces memory safety without
    garbage collection.

    Key features of Rust include:
    - Zero-cost abstractions
    - Move semantics
    - Guaranteed memory safety
    - Threads without data races
    - Trait-based generics
    - Pattern matching
    - Type inference
    - Minimal runtime
    - Efficient C bindings

    Rust is used in production by companies like Mozilla, Dropbox,
    and Cloudflare for building fast and reliable systems.
    """

    # Step 3: Split document into chunks
    print("\n2. Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,    # Max 200 characters per chunk
        chunk_overlap=20   # 20 character overlap for context continuity
    )
    chunks = splitter.split_text(document)
    print(f"   ✓ Document split into {len(chunks)} chunks")

    # Step 4: Index chunks
    print("\n3. Indexing chunks with embeddings...")
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        embedding = mock_embed(chunk)
        metadata = {
            "text": chunk.strip(),
            "chunk_index": i,
            "doc_id": "rust_intro",
        }
        store.upsert(chunk_id, embedding, metadata)
        print(f"   ✓ Indexed chunk {i}: {chunk[:50]}...")

    print(f"\n   Total vectors in store: {len(store)}")

    # Step 5: Query for relevant chunks
    print("\n4. Querying for relevant information...")
    query_text = "What are the key features of Rust?"
    query_embedding = mock_embed(query_text)

    print(f"   Query: '{query_text}'")
    results = store.query(query_embedding, k=3)

    print(f"\n   Top {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   - ID: {result.id}")
        print(f"   - Score: {result.score:.4f}")
        print(f"   - Text: {result.metadata['text'][:100]}...")

    # Step 6: Use results for RAG
    print("\n5. Building RAG context...")
    context = "\n\n".join([
        result.metadata['text']
        for result in results
    ])

    print(f"\n   Context (for LLM):")
    print(f"   {'-' * 56}")
    print(f"   {context[:300]}...")
    print(f"   {'-' * 56}")

    # This context would be passed to an LLM like:
    # prompt = f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
    # answer = llm.generate(prompt)

    print("\n" + "=" * 60)
    print("✓ RAG pipeline complete!")
    print("=" * 60)

    # Cleanup (optional)
    import shutil
    shutil.rmtree("./rag_demo_db", ignore_errors=True)
    print("\n✓ Cleaned up demo database")


if __name__ == "__main__":
    main()
