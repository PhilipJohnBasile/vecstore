"""
Text Splitting Strategies Example

This example demonstrates different text splitting strategies
for breaking documents into chunks for RAG applications.

Key considerations:
- Chunk size (should fit in embedding model's context window)
- Chunk overlap (maintains context continuity)
- Natural boundaries (paragraphs, sentences, words)
"""

from vecstore import RecursiveCharacterTextSplitter


def main():
    print("=" * 70)
    print("VecStore - Text Splitting Strategies Example")
    print("=" * 70)

    # Sample document
    document = """
The Rust Programming Language

Rust is a systems programming language that runs blazingly fast, prevents
segfaults, and guarantees thread safety. It accomplishes this through a
sophisticated system of ownership with set of rules that the compiler checks.

Memory Safety

Rust's ownership system is its most distinctive feature. At any given time,
you can have either one mutable reference or any number of immutable references.
This prevents data races at compile time. When a variable goes out of scope,
Rust automatically cleans up the memory.

Zero-Cost Abstractions

Rust provides high-level abstractions like iterators, closures, and pattern
matching, but these abstractions compile down to code as efficient as if you
had written the low-level code by hand.

Concurrency Without Fear

Rust's type system and ownership model guarantee thread safety. You can write
concurrent code without worrying about data races. The compiler catches these
errors before your code even runs.

Growing Ecosystem

Cargo is Rust's build system and package manager. It makes managing dependencies,
running tests, and building projects incredibly easy. The crates.io registry
hosts thousands of packages that you can use in your projects.

Production Use

Major companies like Mozilla, Dropbox, Cloudflare, and Discord use Rust in
production for everything from browser engines to cloud infrastructure.
    """

    print("\n1. Small chunks (200 chars, 20 overlap):")
    print("   Use case: Precise matching, fine-grained search")
    print("   " + "-" * 66)
    splitter_small = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    small_chunks = splitter_small.split_text(document)
    print(f"   Generated {len(small_chunks)} chunks")
    for i, chunk in enumerate(small_chunks[:3], 1):  # Show first 3
        print(f"\n   Chunk {i} ({len(chunk)} chars):")
        print(f"   {chunk[:150]}...")

    print("\n\n2. Medium chunks (500 chars, 50 overlap):")
    print("   Use case: Balanced approach, good for most RAG applications")
    print("   " + "-" * 66)
    splitter_medium = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    medium_chunks = splitter_medium.split_text(document)
    print(f"   Generated {len(medium_chunks)} chunks")
    for i, chunk in enumerate(medium_chunks[:2], 1):  # Show first 2
        print(f"\n   Chunk {i} ({len(chunk)} chars):")
        print(f"   {chunk[:200]}...")

    print("\n\n3. Large chunks (1000 chars, 100 overlap):")
    print("   Use case: More context, fewer chunks, better for long-form content")
    print("   " + "-" * 66)
    splitter_large = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    large_chunks = splitter_large.split_text(document)
    print(f"   Generated {len(large_chunks)} chunks")
    for i, chunk in enumerate(large_chunks[:1], 1):  # Show first 1
        print(f"\n   Chunk {i} ({len(chunk)} chars):")
        print(f"   {chunk[:300]}...")

    # Demonstrate overlap
    print("\n\n4. Demonstrating chunk overlap:")
    print("   " + "-" * 66)
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    short_text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    overlap_chunks = splitter.split_text(short_text)

    print(f"   Text: \"{short_text}\"")
    print(f"   Chunk size: 100, Overlap: 20")
    print(f"\n   Generated {len(overlap_chunks)} chunks:")
    for i, chunk in enumerate(overlap_chunks, 1):
        print(f"\n   Chunk {i}: \"{chunk}\"")

    # Best practices
    print("\n\n" + "=" * 70)
    print("Text Splitting Best Practices:")
    print("=" * 70)
    print("""
1. Choose chunk size based on your embedding model:
   - sentence-transformers (all-MiniLM-L6-v2): ~512 tokens ≈ 300-400 chars
   - OpenAI text-embedding-3-small: ~8192 tokens ≈ 5000-6000 chars
   - Cohere embed-english-v3.0: ~512 tokens ≈ 300-400 chars

2. Overlap guidelines:
   - Small chunks (200-300): 10-15% overlap (20-40 chars)
   - Medium chunks (500-700): 10% overlap (50-70 chars)
   - Large chunks (1000+): 10% overlap (100+ chars)

3. Content-specific strategies:
   - Code: Split on function/class boundaries
   - Markdown: Split on header boundaries
   - Books: Split on chapter/section boundaries
   - Chat logs: Split on message boundaries

4. Quality checks:
   - Ensure chunks are semantically meaningful
   - Avoid splitting mid-sentence when possible
   - Test retrieval quality with different chunk sizes
   - Monitor chunk size distribution

5. Performance considerations:
   - Smaller chunks = more vectors = slower queries but better precision
   - Larger chunks = fewer vectors = faster queries but less precise
   - Find the sweet spot for your use case

6. Common chunk sizes in production:
   - Q&A systems: 200-400 chars (precise answers)
   - Documentation search: 500-800 chars (balanced)
   - Long-form content: 1000-1500 chars (more context)
   - Code search: Variable (function/class level)
    """)

    print("\n" + "=" * 70)
    print("Chunk Size Comparison:")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Chunks':<10} {'Avg Size':<10} {'Use Case'}")
    print("-" * 70)

    strategies = [
        ("Small (200)", small_chunks, "Precise search"),
        ("Medium (500)", medium_chunks, "General RAG"),
        ("Large (1000)", large_chunks, "Long context"),
    ]

    for name, chunks, use_case in strategies:
        avg_size = sum(len(c) for c in chunks) // len(chunks) if chunks else 0
        print(f"{name:<20} {len(chunks):<10} {avg_size:<10} {use_case}")

    print("\n" + "=" * 70)
    print("✓ Text splitting demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
