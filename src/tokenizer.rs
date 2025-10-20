//! Pluggable tokenizer system for text processing
//!
//! Provides flexible tokenization for BM25 and hybrid search.
//!
//! # Examples
//!
//! ```rust
//! use vecstore::tokenizer::{Tokenizer, SimpleTokenizer, LanguageTokenizer};
//!
//! // Simple whitespace tokenizer
//! let tokenizer = SimpleTokenizer::new();
//! let tokens = tokenizer.tokenize("Hello, world!");
//! assert_eq!(tokens, vec!["hello", "world"]);
//!
//! // Language-aware tokenizer with stopwords
//! let tokenizer = LanguageTokenizer::english();
//! let tokens = tokenizer.tokenize("The quick brown fox");
//! // "the" is removed as stopword
//! assert_eq!(tokens, vec!["quick", "brown", "fox"]);
//! ```

use std::collections::HashSet;

/// Trait for text tokenization
///
/// Implementers provide different tokenization strategies:
/// - SimpleTokenizer: Basic whitespace + punctuation splitting
/// - LanguageTokenizer: Stopword removal + optional stemming
/// - WhitespaceTokenizer: Split on whitespace only
/// - NGramTokenizer: Character or word n-grams
pub trait Tokenizer: Send + Sync {
    /// Tokenize text into a vector of tokens
    fn tokenize(&self, text: &str) -> Vec<String>;

    /// Get a description of this tokenizer
    fn name(&self) -> &'static str {
        "Tokenizer"
    }
}

/// Simple tokenizer: lowercase + split on whitespace and punctuation
///
/// This is the default tokenizer, matching VecStore's original behavior.
/// Fast and works for most Latin-script languages.
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    /// Whether to convert to lowercase (default: true)
    pub lowercase: bool,
    /// Whether to remove punctuation (default: true)
    pub remove_punctuation: bool,
}

impl SimpleTokenizer {
    /// Create a new SimpleTokenizer with default settings
    pub fn new() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
        }
    }

    /// Create tokenizer that preserves case
    pub fn with_case_preserved() -> Self {
        Self {
            lowercase: false,
            remove_punctuation: true,
        }
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let text = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        if self.remove_punctuation {
            text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        } else {
            text.split_whitespace()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        }
    }

    fn name(&self) -> &'static str {
        "SimpleTokenizer"
    }
}

/// Whitespace-only tokenizer
///
/// Splits on whitespace only, preserving punctuation.
/// Useful when punctuation carries meaning (code, URLs, etc.)
#[derive(Debug, Clone)]
pub struct WhitespaceTokenizer {
    pub lowercase: bool,
}

impl WhitespaceTokenizer {
    pub fn new() -> Self {
        Self { lowercase: true }
    }
}

impl Default for WhitespaceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let text = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        text.split_whitespace().map(|s| s.to_string()).collect()
    }

    fn name(&self) -> &'static str {
        "WhitespaceTokenizer"
    }
}

/// Language-aware tokenizer with stopword removal
///
/// Supports multiple languages with built-in stopword lists.
/// Optionally applies stemming for better recall.
#[derive(Debug, Clone)]
pub struct LanguageTokenizer {
    lowercase: bool,
    remove_punctuation: bool,
    stopwords: HashSet<String>,
}

impl LanguageTokenizer {
    /// Create a new LanguageTokenizer with custom stopwords
    pub fn new(stopwords: HashSet<String>) -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
            stopwords,
        }
    }

    /// Create English tokenizer with common stopwords
    pub fn english() -> Self {
        Self::new(english_stopwords())
    }

    /// Create tokenizer with no stopwords
    pub fn no_stopwords() -> Self {
        Self::new(HashSet::new())
    }
}

impl Tokenizer for LanguageTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let text = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let tokens: Vec<String> = if self.remove_punctuation {
            text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        } else {
            text.split_whitespace().map(|s| s.to_string()).collect()
        };

        // Remove stopwords
        tokens
            .into_iter()
            .filter(|token| !self.stopwords.contains(token))
            .collect()
    }

    fn name(&self) -> &'static str {
        "LanguageTokenizer"
    }
}

/// English stopwords list (common function words)
fn english_stopwords() -> HashSet<String> {
    let words = vec![
        // Articles
        "a", "an", "the", // Conjunctions
        "and", "or", "but", "nor", // Prepositions (common)
        "in", "on", "at", "to", "for", "of", "with", "from", "by", "about", // Pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
        "your", "his", "its", "our", "their", "this", "that", "these", "those",
        // Auxiliary verbs
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "will", "would", "shall", "should", "can", "could", "may",
        "might", "must", // Common adverbs/adjectives
        "not", "no", "yes", "very", "too", "so", "just", "only", "all", "any", "some", "more",
        "most", "other", "such", // Question words
        "what", "which", "who", "when", "where", "why", "how", // Other common words
        "as", "if", "than", "then", "there", "here",
    ];

    words.into_iter().map(|s| s.to_string()).collect()
}

/// N-gram tokenizer for character or word n-grams
///
/// Useful for fuzzy matching, typo tolerance, and substring search.
#[derive(Debug, Clone)]
pub struct NGramTokenizer {
    /// Size of n-grams (e.g., 3 for trigrams)
    pub n: usize,
    /// Whether to generate character n-grams (true) or word n-grams (false)
    pub char_ngrams: bool,
}

impl NGramTokenizer {
    /// Create character n-gram tokenizer
    pub fn char_ngrams(n: usize) -> Self {
        Self {
            n,
            char_ngrams: true,
        }
    }

    /// Create word n-gram tokenizer
    pub fn word_ngrams(n: usize) -> Self {
        Self {
            n,
            char_ngrams: false,
        }
    }
}

impl Tokenizer for NGramTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        if self.char_ngrams {
            // Character n-grams
            let text = text.to_lowercase();
            let chars: Vec<char> = text.chars().collect();

            if chars.len() < self.n {
                return vec![text];
            }

            chars
                .windows(self.n)
                .map(|window| window.iter().collect())
                .collect()
        } else {
            // Word n-grams
            let words: Vec<String> = text
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();

            if words.len() < self.n {
                return vec![words.join(" ")];
            }

            words
                .windows(self.n)
                .map(|window| window.join(" "))
                .collect()
        }
    }

    fn name(&self) -> &'static str {
        if self.char_ngrams {
            "CharNGramTokenizer"
        } else {
            "WordNGramTokenizer"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer() {
        let tokenizer = SimpleTokenizer::new();
        let tokens = tokenizer.tokenize("Hello, world! This is a test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_simple_tokenizer_case_preserved() {
        let tokenizer = SimpleTokenizer::with_case_preserved();
        let tokens = tokenizer.tokenize("Hello World");
        assert_eq!(tokens, vec!["Hello", "World"]);
    }

    #[test]
    fn test_whitespace_tokenizer() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("hello@example.com test-data");
        assert_eq!(tokens, vec!["hello@example.com", "test-data"]);
    }

    #[test]
    fn test_language_tokenizer_english() {
        let tokenizer = LanguageTokenizer::english();
        let tokens = tokenizer.tokenize("The quick brown fox jumps");
        // "the" should be removed as stopword
        assert_eq!(tokens, vec!["quick", "brown", "fox", "jumps"]);
    }

    #[test]
    fn test_language_tokenizer_no_stopwords() {
        let tokenizer = LanguageTokenizer::no_stopwords();
        let tokens = tokenizer.tokenize("The quick brown fox");
        assert_eq!(tokens, vec!["the", "quick", "brown", "fox"]);
    }

    #[test]
    fn test_char_ngrams() {
        let tokenizer = NGramTokenizer::char_ngrams(3);
        let tokens = tokenizer.tokenize("hello");
        assert_eq!(tokens, vec!["hel", "ell", "llo"]);
    }

    #[test]
    fn test_word_ngrams() {
        let tokenizer = NGramTokenizer::word_ngrams(2);
        let tokens = tokenizer.tokenize("the quick brown fox");
        assert_eq!(tokens, vec!["the quick", "quick brown", "brown fox"]);
    }

    #[test]
    fn test_ngrams_short_text() {
        let tokenizer = NGramTokenizer::char_ngrams(5);
        let tokens = tokenizer.tokenize("hi");
        assert_eq!(tokens, vec!["hi"]);
    }

    #[test]
    fn test_empty_text() {
        let tokenizer = SimpleTokenizer::new();
        let tokens = tokenizer.tokenize("");
        assert_eq!(tokens, Vec::<String>::new());
    }

    #[test]
    fn test_unicode_text() {
        let tokenizer = SimpleTokenizer::new();
        let tokens = tokenizer.tokenize("Hello 世界 émojis 😀");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"世界".to_string()));
        assert!(tokens.contains(&"émojis".to_string()));
    }
}
