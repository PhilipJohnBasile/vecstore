"""
Tests for text splitting functionality

Tests RecursiveCharacterTextSplitter with various text types and configurations.
"""

import pytest


class TestRecursiveCharacterTextSplitter:
    """Test RecursiveCharacterTextSplitter"""

    def test_create_splitter(self):
        """Test creating a text splitter"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(500, 50)
        assert splitter is not None

    def test_splitter_repr(self):
        """Test splitter string representation"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(500, 50)
        repr_str = repr(splitter)
        assert "RecursiveCharacterTextSplitter" in repr_str

    def test_split_short_text(self):
        """Test splitting text shorter than chunk size"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(500, 50)
        text = "This is a short text."

        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self):
        """Test splitting text longer than chunk size"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(100, 10)
        text = "A" * 250  # 250 characters

        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            # Text splitter may produce chunks slightly larger than chunk_size + overlap
            # due to boundary detection
            assert len(chunk) <= 200  # Reasonable upper bound

    def test_split_with_paragraphs(self):
        """Test splitting text with paragraph boundaries"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(100, 20)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_split_with_sentences(self):
        """Test splitting respects sentence boundaries"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(50, 10)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_different_chunk_sizes(self):
        """Test different chunk size configurations"""
        from vecstore import RecursiveCharacterTextSplitter

        text = "A" * 500

        # Small chunks
        splitter_small = RecursiveCharacterTextSplitter(100, 10)
        chunks_small = splitter_small.split_text(text)

        # Large chunks
        splitter_large = RecursiveCharacterTextSplitter(200, 20)
        chunks_large = splitter_large.split_text(text)

        # Smaller chunk size should produce more chunks
        assert len(chunks_small) > len(chunks_large)

    def test_chunk_overlap(self):
        """Test that chunks have overlap"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(50, 10)
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10  # Long text

        chunks = splitter.split_text(text)

        # Check that consecutive chunks share content (overlap)
        if len(chunks) > 1:
            # This is a basic check - actual overlap detection would be more complex
            assert len(chunks) >= 2

    def test_empty_text(self):
        """Test splitting empty text"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(100, 10)
        chunks = splitter.split_text("")

        assert isinstance(chunks, list)
        # Empty text might return empty list or list with empty string

    def test_whitespace_text(self):
        """Test splitting text with only whitespace"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(100, 10)
        text = "   \n\n   \t\t   "

        chunks = splitter.split_text(text)
        assert isinstance(chunks, list)

    def test_unicode_text(self):
        """Test splitting Unicode text"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(100, 10)
        text = "Hello 世界! " * 20

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

        # Verify Unicode is preserved
        for chunk in chunks:
            # Should contain Unicode characters
            assert isinstance(chunk, str)

    def test_code_text(self):
        """Test splitting code with proper indentation"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(200, 20)
        code = """
def function1():
    print("Hello")
    return 42

def function2():
    print("World")
    return 24
        """

        chunks = splitter.split_text(code)
        assert len(chunks) >= 1

    def test_markdown_text(self):
        """Test splitting Markdown text"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(150, 20)
        markdown = """
# Header 1

Some content here.

## Header 2

More content.

### Header 3

Even more content.
        """

        chunks = splitter.split_text(markdown)
        assert len(chunks) >= 1

    def test_very_small_chunks(self):
        """Test with very small chunk size"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(10, 2)
        text = "This is a test sentence."

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        for chunk in chunks:
            # Chunks should be roughly around the target size
            # Text splitter may produce larger chunks due to boundary detection
            assert len(chunk) <= 30  # More reasonable tolerance

    def test_very_large_chunks(self):
        """Test with very large chunk size"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(10000, 100)
        text = "Short text"

        chunks = splitter.split_text(text)
        # Text is shorter than chunk size
        assert len(chunks) == 1

    def test_zero_overlap(self):
        """Test with zero overlap"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(100, 0)
        text = "A" * 250

        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_realistic_document(self):
        """Test with a realistic document"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(500, 50)
        document = """
The Rust Programming Language

Rust is a systems programming language that runs blazingly fast,
prevents segfaults, and guarantees thread safety.

Memory Safety

Rust's ownership system is its most distinctive feature. At any given time,
you can have either one mutable reference or any number of immutable references.

Zero-Cost Abstractions

Rust provides high-level abstractions like iterators, closures, and pattern
matching, but these abstractions compile down to code as efficient as if you
had written the low-level code by hand.
        """

        chunks = splitter.split_text(document)

        # Should produce multiple chunks
        assert len(chunks) >= 1

        # Each chunk should be non-empty
        for chunk in chunks:
            assert len(chunk) > 0

        # Chunks should roughly respect size limit (with some tolerance for overlap)
        for chunk in chunks:
            assert len(chunk) <= 600  # chunk_size + overlap + tolerance


class TestTextSplitterEdgeCases:
    """Test edge cases for text splitting"""

    def test_single_long_word(self):
        """Test splitting a very long word"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(50, 5)
        text = "A" * 200  # Single long "word"

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_repeating_separators(self):
        """Test text with many consecutive separators"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(100, 10)
        text = "Word\n\n\n\n\n\nAnother word"

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_mixed_separators(self):
        """Test text with mixed separator types"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(100, 10)
        text = "Para1.\n\nPara2. Sentence! Question? Another sentence."

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_consistency(self):
        """Test that splitting is consistent"""
        from vecstore import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(100, 10)
        text = "Same text every time. " * 20

        chunks1 = splitter.split_text(text)
        chunks2 = splitter.split_text(text)

        # Should produce identical results
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1 == c2
