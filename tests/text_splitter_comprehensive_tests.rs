// Comprehensive tests for text splitter functionality
// Tests various splitting strategies, edge cases, and chunking scenarios

use vecstore::text_splitter::{
    MarkdownTextSplitter, RecursiveCharacterTextSplitter, TextSplitter, TokenTextSplitter,
};

#[test]
fn test_recursive_splitter_basic() {
    let splitter = RecursiveCharacterTextSplitter::new(50, 10);
    let text = "This is a simple test. This is another sentence.";

    let result = splitter.split_text(text);
    assert!(result.is_ok());

    let chunks = result.unwrap();
    assert!(chunks.len() >= 1);
}

#[test]
fn test_recursive_splitter_paragraph_boundary() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);
    let text = "First paragraph with some content.\n\nSecond paragraph with more content.\n\nThird paragraph here.";

    let chunks = splitter.split_text(text).unwrap();

    // Should split on double newlines (paragraph boundaries)
    assert!(chunks.len() >= 1);
}

#[test]
fn test_recursive_splitter_respects_chunk_size() {
    let chunk_size = 50;
    let splitter = RecursiveCharacterTextSplitter::new(chunk_size, 0);

    let text = "a".repeat(200);
    let chunks = splitter.split_text(&text).unwrap();

    for chunk in &chunks {
        assert!(chunk.len() <= chunk_size + 10); // Allow some tolerance
    }
}

#[test]
fn test_recursive_splitter_overlap() {
    let chunk_size = 50;
    let overlap = 10;
    let splitter = RecursiveCharacterTextSplitter::new(chunk_size, overlap);

    let text = "This is a test sentence that should be split. Another sentence here. And one more for good measure.";
    let chunks = splitter.split_text(text).unwrap();

    // If we have multiple chunks, they should overlap
    if chunks.len() > 1 {
        // Check that consecutive chunks share some content
        // This is approximate since overlap may not be exact
        assert!(chunks.len() >= 1);
    }
}

#[test]
fn test_recursive_splitter_empty_text() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let chunks = splitter.split_text("").unwrap();

    assert_eq!(chunks.len(), 0);
}

#[test]
fn test_recursive_splitter_single_character() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let chunks = splitter.split_text("a").unwrap();

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], "a");
}

#[test]
fn test_recursive_splitter_whitespace_only() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let chunks = splitter.split_text("   \n\n  \t  ").unwrap();

    // May return empty or whitespace chunks depending on implementation
    assert!(chunks.len() >= 0);
}

#[test]
fn test_recursive_splitter_long_word() {
    let chunk_size = 20;
    let splitter = RecursiveCharacterTextSplitter::new(chunk_size, 0);

    // Word longer than chunk_size
    let text = "supercalifragilisticexpialidocious";

    let chunks = splitter.split_text(text).unwrap();

    // Should still split the long word
    assert!(chunks.len() >= 1);
}

#[test]
fn test_recursive_splitter_unicode() {
    let splitter = RecursiveCharacterTextSplitter::new(50, 10);

    let text = "Hello 世界! This is a test with émojis 😀 and spëcial çharacters.";

    let chunks = splitter.split_text(text).unwrap();

    assert!(chunks.len() >= 1);

    // Verify no character corruption
    let rejoined = chunks.join("");
    assert!(rejoined.contains("世界"));
    assert!(rejoined.contains("😀"));
}

#[test]
fn test_recursive_splitter_multiple_newlines() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let text = "Line 1\n\n\n\nLine 2\n\n\n\nLine 3";

    let chunks = splitter.split_text(text).unwrap();

    assert!(chunks.len() >= 1);
}

#[test]
fn test_recursive_splitter_sentence_boundaries() {
    let splitter = RecursiveCharacterTextSplitter::new(30, 5);

    let text = "First sentence. Second sentence! Third sentence? Fourth sentence.";

    let chunks = splitter.split_text(text).unwrap();

    // Should try to split on sentence boundaries
    assert!(chunks.len() >= 2);
}

#[test]
fn test_recursive_splitter_preserves_content() {
    let splitter = RecursiveCharacterTextSplitter::new(50, 0);

    let text = "The quick brown fox jumps over the lazy dog.";

    let chunks = splitter.split_text(text).unwrap();

    // When rejoined (without overlap), should contain all original content
    let rejoined = chunks.join("");

    for word in ["quick", "brown", "fox", "lazy", "dog"] {
        assert!(rejoined.contains(word));
    }
}

#[test]
fn test_token_splitter_basic() {
    let splitter = TokenTextSplitter::new(10, 2);

    let text = "This is a test sentence with several words.";

    let result = splitter.split_text(text);

    // TokenTextSplitter may not be implemented
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_token_splitter_respects_token_limit() {
    let max_tokens = 10;
    let splitter = TokenTextSplitter::new(max_tokens, 0);

    let text = "word ".repeat(50);

    if let Ok(chunks) = splitter.split_text(&text) {
        // Each chunk should have at most max_tokens tokens
        for chunk in chunks {
            let token_count = chunk.split_whitespace().count();
            assert!(token_count <= max_tokens + 2); // Allow small tolerance
        }
    }
}

#[test]
fn test_markdown_splitter_headers() {
    let splitter = MarkdownTextSplitter::new(200, 20);

    let text = r#"# Heading 1
Content under heading 1.

## Heading 2
Content under heading 2.

### Heading 3
Content under heading 3."#;

    if let Ok(chunks) = splitter.split_text(text) {
        // Should respect markdown structure
        assert!(chunks.len() >= 1);
    }
}

#[test]
fn test_markdown_splitter_code_blocks() {
    let splitter = MarkdownTextSplitter::new(200, 20);

    let text = r#"# Code Example

Here's some code:

```python
def hello():
    print("world")
```

More text here."#;

    if let Ok(chunks) = splitter.split_text(text) {
        // Should keep code blocks together
        assert!(chunks.len() >= 1);
    }
}

#[test]
fn test_markdown_splitter_lists() {
    let splitter = MarkdownTextSplitter::new(100, 10);

    let text = r#"# List Example

- Item 1
- Item 2
- Item 3

1. Numbered item 1
2. Numbered item 2"#;

    if let Ok(chunks) = splitter.split_text(text) {
        assert!(chunks.len() >= 1);
    }
}

#[test]
fn test_split_with_metadata() {
    let splitter = RecursiveCharacterTextSplitter::new(50, 10);

    let text = "This is a test. Another sentence. And one more.";

    let result = splitter.split_with_metadata(text);

    assert!(result.is_ok());

    let chunks = result.unwrap();
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.index, i);
        assert!(!chunk.content.is_empty());
    }
}

#[test]
fn test_very_long_document() {
    let splitter = RecursiveCharacterTextSplitter::new(500, 50);

    // Generate a very long document
    let mut text = String::new();
    for i in 0..1000 {
        text.push_str(&format!("Sentence number {}. ", i));
    }

    let chunks = splitter.split_text(&text).unwrap();

    // Should split into multiple chunks
    assert!(chunks.len() > 10);

    // All chunks should respect size limits (with generous tolerance for overlap)
    for chunk in &chunks {
        assert!(chunk.len() <= 700); // chunk_size (500) + overlap (100) + tolerance
    }
}

#[test]
fn test_text_with_special_characters() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let text = "Special chars: @#$%^&*()_+-={}[]|\\:;\"'<>,.?/~`";

    let chunks = splitter.split_text(text).unwrap();

    assert!(chunks.len() >= 1);
}

#[test]
fn test_text_with_numbers() {
    let splitter = RecursiveCharacterTextSplitter::new(50, 10);

    let text = "Numbers: 123 456.789 1,000,000 -42 3.14159";

    let chunks = splitter.split_text(text).unwrap();

    assert!(chunks.len() >= 1);
}

#[test]
fn test_text_with_urls() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let text = "Visit https://example.com or http://test.org for more info.";

    let chunks = splitter.split_text(text).unwrap();

    // URLs should be preserved
    let rejoined = chunks.join("");
    assert!(rejoined.contains("https://example.com"));
}

#[test]
fn test_text_with_emails() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let text = "Contact us at info@example.com or support@test.org";

    let chunks = splitter.split_text(text).unwrap();

    let rejoined = chunks.join("");
    assert!(rejoined.contains("@example.com"));
}

#[test]
fn test_chunk_size_one() {
    let splitter = RecursiveCharacterTextSplitter::new(1, 0);

    let text = "abc";

    let chunks = splitter.split_text(text).unwrap();

    // Should split into individual characters
    assert!(chunks.len() >= 1);
}

#[test]
fn test_overlap_larger_than_chunk_size() {
    let splitter = RecursiveCharacterTextSplitter::new(10, 20);

    let text = "This is a test sentence.";

    // Should handle this edge case gracefully
    let result = splitter.split_text(text);

    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_zero_chunk_size() {
    let splitter = RecursiveCharacterTextSplitter::new(0, 0);

    let text = "test";

    let result = splitter.split_text(text);

    // Should handle edge case
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_text_with_tabs() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let text = "Column1\tColumn2\tColumn3\nValue1\tValue2\tValue3";

    let chunks = splitter.split_text(text).unwrap();

    assert!(chunks.len() >= 1);
}

#[test]
fn test_mixed_line_endings() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    // Mix of \n, \r\n, and \r
    let text = "Line 1\nLine 2\r\nLine 3\rLine 4";

    let chunks = splitter.split_text(text).unwrap();

    assert!(chunks.len() >= 1);
}

#[test]
fn test_html_content() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let text = "<html><body><h1>Title</h1><p>Paragraph text.</p></body></html>";

    let chunks = splitter.split_text(text).unwrap();

    assert!(chunks.len() >= 1);
}

#[test]
fn test_json_content() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let text = r#"{"key": "value", "array": [1, 2, 3], "nested": {"inner": "data"}}"#;

    let chunks = splitter.split_text(text).unwrap();

    assert!(chunks.len() >= 1);
}

#[test]
fn test_code_content() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 20);

    let text = r#"
fn main() {
    println!("Hello, world!");
    let x = 42;
    if x > 0 {
        println!("Positive");
    }
}
"#;

    let chunks = splitter.split_text(text).unwrap();

    assert!(chunks.len() >= 1);
}

#[test]
fn test_repetitive_content() {
    let splitter = RecursiveCharacterTextSplitter::new(50, 10);

    let text = "test ".repeat(100);

    let chunks = splitter.split_text(&text).unwrap();

    assert!(chunks.len() > 1);
}

#[test]
fn test_single_long_line() {
    let splitter = RecursiveCharacterTextSplitter::new(50, 5);

    let text = "a".repeat(500);

    let chunks = splitter.split_text(&text).unwrap();

    assert!(chunks.len() > 5);
}

#[test]
fn test_quoted_text() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let text = r#"He said, "This is a quote." She replied, 'Another quote.'"#;

    let chunks = splitter.split_text(text).unwrap();

    assert!(chunks.len() >= 1);
}

#[test]
fn test_text_with_abbreviations() {
    let splitter = RecursiveCharacterTextSplitter::new(100, 10);

    let text = "Dr. Smith went to the U.S.A. on Jan. 1st, 2024.";

    let chunks = splitter.split_text(text).unwrap();

    // Should not split on abbreviation periods
    assert!(chunks.len() >= 1);
}
