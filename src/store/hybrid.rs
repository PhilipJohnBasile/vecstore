// Hybrid Search - Vector + Keyword (BM25)
//
// This module implements hybrid search combining:
// 1. Vector similarity (semantic search via HNSW)
// 2. Keyword search (BM25 algorithm)
//
// This is THE killer feature for RAG applications!

use super::types::{FilterExpr, Id};
use crate::tokenizer::{SimpleTokenizer, Tokenizer};
use std::collections::HashMap;

/// Hybrid query combining vector and keyword search
#[derive(Debug, Clone)]
pub struct HybridQuery {
    /// Vector for semantic similarity
    pub vector: Vec<f32>,

    /// Keywords for text search
    pub keywords: String,

    /// Number of results to return
    pub k: usize,

    /// Optional metadata filter
    pub filter: Option<FilterExpr>,

    /// Weighting between vector (alpha) and keyword (1-alpha)
    /// Range: [0.0, 1.0]
    /// - 0.0 = pure keyword search
    /// - 1.0 = pure vector search
    /// - 0.7 = 70% vector, 30% keyword (recommended for RAG)
    pub alpha: f32,
}

impl Default for HybridQuery {
    fn default() -> Self {
        Self {
            vector: Vec::new(),
            keywords: String::new(),
            k: 10,
            filter: None,
            alpha: 0.7, // Default: 70% vector, 30% keyword
        }
    }
}

/// Posting entry with term frequency and positions
#[derive(Debug, Clone)]
pub struct Posting {
    /// Document ID
    pub doc_id: Id,
    /// Term frequency (how many times term appears)
    pub term_freq: usize,
    /// Positions where term appears (0-indexed)
    pub positions: Vec<usize>,
}

/// Text index for keyword search with pluggable tokenization and phrase support
pub struct TextIndex {
    /// Document texts (id -> text)
    texts: HashMap<Id, String>,

    /// Inverted index with positions (term -> list of postings)
    inverted_index: HashMap<String, Vec<Posting>>,

    /// Document lengths (id -> number of terms)
    doc_lengths: HashMap<Id, usize>,

    /// Average document length
    avg_doc_length: f32,

    /// Total number of documents
    num_docs: usize,

    /// Total document length (sum of all doc lengths) - Optimization Issue #23 fix
    total_doc_length: usize,

    /// Pluggable tokenizer for text processing
    tokenizer: Box<dyn Tokenizer>,
}

impl Default for TextIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl TextIndex {
    /// Create a new TextIndex with the default SimpleTokenizer
    pub fn new() -> Self {
        Self::with_tokenizer(Box::new(SimpleTokenizer::new()))
    }

    /// Create a new TextIndex with a custom tokenizer
    pub fn with_tokenizer(tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            texts: HashMap::new(),
            inverted_index: HashMap::new(),
            doc_lengths: HashMap::new(),
            avg_doc_length: 0.0,
            num_docs: 0,
            total_doc_length: 0, // Optimization Issue #23 fix
            tokenizer,
        }
    }

    /// Get the name of the current tokenizer
    pub fn tokenizer_name(&self) -> &'static str {
        self.tokenizer.name()
    }

    /// Index a document's text
    pub fn index_document(&mut self, id: Id, text: String) {
        // Remove old entry if exists and track whether this is a new document
        let (is_new_document, old_length) = if let Some(old_text) = self.texts.remove(&id) {
            let old_len = self.doc_lengths.get(&id).copied().unwrap_or(0);
            self.remove_from_index(&id, &old_text);
            (false, old_len) // Document already existed
        } else {
            (true, 0) // New document
        };

        // Tokenize using pluggable tokenizer
        let tokens = self.tokenizer.tokenize(&text);
        let doc_length = tokens.len();

        // Track term frequencies and positions
        let mut term_info: HashMap<String, Vec<usize>> = HashMap::new();
        for (position, token) in tokens.iter().enumerate() {
            term_info.entry(token.clone()).or_default().push(position);
        }

        // Update inverted index with positions
        for (term, positions) in term_info {
            let term_freq = positions.len();
            self.inverted_index.entry(term).or_default().push(Posting {
                doc_id: id.clone(),
                term_freq,
                positions,
            });
        }

        // Update document info
        self.texts.insert(id.clone(), text);
        self.doc_lengths.insert(id, doc_length);

        // Update total_doc_length and num_docs (Optimization Issue #23 fix)
        if is_new_document {
            self.num_docs += 1;
            self.total_doc_length += doc_length;
        } else {
            // Replace old length with new length
            self.total_doc_length = self.total_doc_length.saturating_sub(old_length) + doc_length;
        }

        // Update average document length
        self.update_avg_doc_length();
    }

    /// Remove a document from the index
    pub fn remove_document(&mut self, id: &str) {
        if let Some(text) = self.texts.remove(id) {
            self.remove_from_index(id, &text);

            // Update total_doc_length before removing (Optimization Issue #23 fix)
            if let Some(length) = self.doc_lengths.remove(id) {
                self.total_doc_length = self.total_doc_length.saturating_sub(length);
            }

            self.num_docs = self.num_docs.saturating_sub(1);
            self.update_avg_doc_length();
        }
    }

    fn remove_from_index(&mut self, id: &str, text: &str) {
        let tokens = self.tokenizer.tokenize(text);
        let unique_terms: std::collections::HashSet<_> = tokens.into_iter().collect();

        for term in unique_terms {
            if let Some(postings) = self.inverted_index.get_mut(&term) {
                postings.retain(|posting| posting.doc_id != id);
                if postings.is_empty() {
                    self.inverted_index.remove(&term);
                }
            }
        }
    }

    fn update_avg_doc_length(&mut self) {
        // Optimization Issue #23 fix: use cached total instead of recomputing
        if self.num_docs == 0 {
            self.avg_doc_length = 0.0;
        } else {
            self.avg_doc_length = self.total_doc_length as f32 / self.num_docs as f32;
        }
    }

    /// Compute BM25 scores for query terms
    ///
    /// BM25 formula:
    /// score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
    ///
    /// where:
    /// - IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
    /// - f(qi, D) = frequency of qi in document D
    /// - |D| = length of document D
    /// - avgdl = average document length
    /// - k1 = 1.2 (term frequency saturation)
    /// - b = 0.75 (length normalization)
    pub fn bm25_scores(&self, query: &str) -> HashMap<Id, f32> {
        let query_terms = self.tokenizer.tokenize(query);
        if query_terms.is_empty() {
            return HashMap::new();
        }

        let k1 = 1.2;
        let b = 0.75;

        let mut scores: HashMap<Id, f32> = HashMap::new();

        for term in &query_terms {
            if let Some(postings) = self.inverted_index.get(term) {
                let df = postings.len() as f32;
                let idf = ((self.num_docs as f32 - df + 0.5) / (df + 0.5)).ln();

                for posting in postings {
                    let doc_length = *self.doc_lengths.get(&posting.doc_id).unwrap_or(&0) as f32;
                    let tf = posting.term_freq as f32;

                    let numerator = tf * (k1 + 1.0);
                    let denominator = tf + k1 * (1.0 - b + b * (doc_length / self.avg_doc_length));

                    let score = idf * (numerator / denominator);
                    *scores.entry(posting.doc_id.clone()).or_insert(0.0) += score;
                }
            }
        }

        scores
    }

    /// Search for an exact phrase and return matching document IDs with scores
    ///
    /// Uses position information to verify that terms appear consecutively.
    /// Returns BM25 scores with a phrase boost for exact matches.
    pub fn phrase_search(&self, phrase: &str) -> HashMap<Id, f32> {
        let phrase_terms = self.tokenizer.tokenize(phrase);
        if phrase_terms.is_empty() {
            return HashMap::new();
        }

        // Single term - fallback to regular BM25
        if phrase_terms.len() == 1 {
            return self.bm25_scores(phrase);
        }

        // Get postings for first term
        let first_term = &phrase_terms[0];
        let Some(first_postings) = self.inverted_index.get(first_term) else {
            return HashMap::new();
        };

        let mut phrase_matches: HashMap<Id, f32> = HashMap::new();

        // For each document containing the first term
        for posting in first_postings {
            let doc_id = &posting.doc_id;

            // Check if this document contains all phrase terms
            let mut all_term_postings = Vec::new();
            let mut has_all_terms = true;

            for term in &phrase_terms {
                if let Some(term_postings) = self.inverted_index.get(term) {
                    if let Some(posting) = term_postings.iter().find(|p| &p.doc_id == doc_id) {
                        all_term_postings.push(posting);
                    } else {
                        has_all_terms = false;
                        break;
                    }
                } else {
                    has_all_terms = false;
                    break;
                }
            }

            if !has_all_terms {
                continue;
            }

            // Check if terms appear consecutively
            let first_posting = &all_term_postings[0];
            for start_pos in &first_posting.positions {
                let mut found_phrase = true;

                // Verify each subsequent term appears at the expected position
                for (i, posting) in all_term_postings.iter().enumerate().skip(1) {
                    let expected_pos = start_pos + i;
                    if !posting.positions.contains(&expected_pos) {
                        found_phrase = false;
                        break;
                    }
                }

                if found_phrase {
                    // Calculate BM25 score manually for this document
                    let k1 = 1.2;
                    let b = 0.75;
                    let doc_length = *self.doc_lengths.get(doc_id).unwrap_or(&0) as f32;
                    let mut base_score = 0.0;

                    for term in &phrase_terms {
                        if let Some(postings) = self.inverted_index.get(term) {
                            let df = postings.len() as f32;
                            let idf = ((self.num_docs as f32 - df + 0.5) / (df + 0.5)).ln();

                            if let Some(posting) = postings.iter().find(|p| &p.doc_id == doc_id) {
                                let tf = posting.term_freq as f32;
                                let numerator = tf * (k1 + 1.0);
                                let denominator =
                                    tf + k1 * (1.0 - b + b * (doc_length / self.avg_doc_length));
                                base_score += idf * (numerator / denominator);
                            }
                        }
                    }

                    // Apply phrase boost (2x for exact phrase match)
                    let phrase_boost = 2.0;
                    phrase_matches.insert(doc_id.clone(), base_score * phrase_boost);
                    break; // Found at least one occurrence
                }
            }
        }

        phrase_matches
    }

    pub fn has_text(&self, id: &str) -> bool {
        self.texts.contains_key(id)
    }

    pub fn get_text(&self, id: &str) -> Option<&str> {
        self.texts.get(id).map(|s| s.as_str())
    }

    /// Export text data for persistence (Major Issue #6 fix)
    ///
    /// Returns the texts HashMap which can be serialized and saved to disk.
    /// The inverted index, doc_lengths, and statistics are not exported since
    /// they can be rebuilt by re-tokenizing the texts on load.
    pub fn export_texts(&self) -> &HashMap<Id, String> {
        &self.texts
    }

    /// Import text data from disk and rebuild the index (Major Issue #6 fix)
    ///
    /// Takes a HashMap of texts loaded from disk and rebuilds the inverted
    /// index, doc_lengths, and statistics by re-tokenizing all texts.
    /// This allows the text index to survive store reopens and snapshots.
    pub fn import_texts(&mut self, texts: HashMap<Id, String>) {
        // Clear existing state
        self.texts.clear();
        self.inverted_index.clear();
        self.doc_lengths.clear();
        self.num_docs = 0;
        self.total_doc_length = 0;
        self.avg_doc_length = 0.0;

        // Rebuild index by re-indexing all texts
        for (id, text) in texts {
            self.index_document(id, text);
        }
    }
}

/// Combine vector and BM25 scores using weighted sum
pub fn combine_scores(
    vector_results: Vec<(Id, f32)>,
    bm25_scores: HashMap<Id, f32>,
    alpha: f32,
) -> Vec<(Id, f32)> {
    // Normalize scores to [0, 1] range
    let normalize = |scores: Vec<(Id, f32)>| -> HashMap<Id, f32> {
        if scores.is_empty() {
            return HashMap::new();
        }

        let max_score = scores
            .iter()
            .map(|(_, s)| *s)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        let min_score = scores
            .iter()
            .map(|(_, s)| *s)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let range = max_score - min_score;

        scores
            .into_iter()
            .map(|(id, score)| {
                let normalized = if range > 0.0 {
                    (score - min_score) / range
                } else {
                    1.0
                };
                (id, normalized)
            })
            .collect()
    };

    let normalized_vector: HashMap<Id, f32> = normalize(vector_results);

    let normalized_bm25: HashMap<Id, f32> = if bm25_scores.is_empty() {
        HashMap::new()
    } else {
        let bm25_vec: Vec<_> = bm25_scores.into_iter().collect();
        normalize(bm25_vec)
    };

    // Combine scores
    let mut all_ids: std::collections::HashSet<Id> = std::collections::HashSet::new();
    all_ids.extend(normalized_vector.keys().cloned());
    all_ids.extend(normalized_bm25.keys().cloned());

    let mut combined: Vec<(Id, f32)> = all_ids
        .into_iter()
        .map(|id| {
            let vec_score = normalized_vector.get(&id).copied().unwrap_or(0.0);
            let bm25_score = normalized_bm25.get(&id).copied().unwrap_or(0.0);

            let hybrid_score = alpha * vec_score + (1.0 - alpha) * bm25_score;

            (id, hybrid_score)
        })
        .collect();

    // Sort by combined score (descending)
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{LanguageTokenizer, NGramTokenizer, WhitespaceTokenizer};

    #[test]
    fn test_text_index_default_tokenizer() {
        let index = TextIndex::new();
        assert_eq!(index.tokenizer_name(), "SimpleTokenizer");
    }

    #[test]
    fn test_text_index_with_custom_tokenizer() {
        let index = TextIndex::with_tokenizer(Box::new(LanguageTokenizer::english()));
        assert_eq!(index.tokenizer_name(), "LanguageTokenizer");
    }

    #[test]
    fn test_text_index_language_tokenizer() {
        let mut index = TextIndex::with_tokenizer(Box::new(LanguageTokenizer::english()));

        // Index documents with stopwords
        index.index_document(
            "doc1".into(),
            "the quick brown fox jumps over the lazy dog".into(),
        );
        index.index_document("doc2".into(), "a fast cat runs quickly".into());
        index.index_document("doc3".into(), "the slow turtle walks".into());

        // Query with stopwords - should be removed by tokenizer
        let scores = index.bm25_scores("the quick");

        // "the" is a stopword, so only "quick" should match
        assert!(
            scores.contains_key("doc1"),
            "doc1 should match (contains 'quick')"
        );

        // doc2 and doc3 shouldn't match as strongly since they don't have "quick"
        let doc1_score = scores.get("doc1").copied().unwrap_or(0.0);
        let doc2_score = scores.get("doc2").copied().unwrap_or(0.0);

        assert!(
            doc1_score > doc2_score,
            "doc1 should score higher than doc2"
        );
    }

    #[test]
    fn test_text_index_whitespace_tokenizer() {
        let mut index = TextIndex::with_tokenizer(Box::new(WhitespaceTokenizer::new()));

        // Whitespace tokenizer preserves punctuation
        index.index_document("doc1".into(), "hello@example.com test".into());
        index.index_document("doc2".into(), "hello world test".into());

        let scores = index.bm25_scores("hello@example.com");

        assert!(scores.contains_key("doc1"), "doc1 should match exact email");
    }

    #[test]
    fn test_text_index_word_ngrams() {
        let mut index = TextIndex::with_tokenizer(Box::new(NGramTokenizer::word_ngrams(2)));

        index.index_document("doc1".into(), "machine learning rocks".into());
        index.index_document("doc2".into(), "deep learning is fun".into());

        // Query with bigram
        let scores = index.bm25_scores("machine learning");

        // doc1 should have the exact bigram "machine learning"
        assert!(scores.contains_key("doc1"), "doc1 should match bigram");
    }

    #[test]
    fn test_text_index() {
        let mut index = TextIndex::new();

        // Use more documents so query terms aren't overly common
        index.index_document("doc1".into(), "the quick brown fox jumps high".into());
        index.index_document("doc2".into(), "the lazy dog sleeps all day".into());
        index.index_document("doc3".into(), "quick brown dog runs fast".into());
        index.index_document("doc4".into(), "a cat and mouse play together".into());
        index.index_document("doc5".into(), "the bird flies over the tree".into());
        index.index_document("doc6".into(), "rabbits hop in the garden".into());

        // Test BM25 scores
        let scores = index.bm25_scores("quick dog");

        // All docs with matching terms should have scores
        assert!(scores.len() > 0);

        // doc3 should have a score (contains both "quick" and "dog")
        assert!(scores.contains_key("doc3"));

        // doc1 should have a score (contains "quick")
        assert!(scores.contains_key("doc1"));

        // doc2 should have a score (contains "dog")
        assert!(scores.contains_key("doc2"));

        // doc3 should have the highest score (contains both terms)
        let doc3_score = scores.get("doc3").unwrap();
        let doc1_score = scores.get("doc1").unwrap();
        let doc2_score = scores.get("doc2").unwrap();

        assert!(
            doc3_score > doc1_score,
            "doc3 ({}) should score higher than doc1 ({})",
            doc3_score,
            doc1_score
        );
        assert!(
            doc3_score > doc2_score,
            "doc3 ({}) should score higher than doc2 ({})",
            doc3_score,
            doc2_score
        );
    }

    #[test]
    fn test_remove_document() {
        let mut index = TextIndex::new();

        index.index_document("doc1".into(), "hello world".into());
        index.index_document("doc2".into(), "hello rust".into());

        assert_eq!(index.num_docs, 2);

        index.remove_document("doc1");

        assert_eq!(index.num_docs, 1);
        let scores = index.bm25_scores("hello");
        assert!(!scores.contains_key("doc1"));
        assert!(scores.contains_key("doc2"));
    }

    #[test]
    fn test_combine_scores() {
        let vector_results = vec![("doc1".into(), 0.9), ("doc2".into(), 0.5)];

        let mut bm25_scores = HashMap::new();
        bm25_scores.insert("doc2".into(), 10.0);
        bm25_scores.insert("doc3".into(), 5.0);

        let combined = combine_scores(vector_results, bm25_scores, 0.7);

        // Should have all 3 docs
        assert_eq!(combined.len(), 3);

        // Scores should be sorted descending
        assert!(combined[0].1 >= combined[1].1);
        assert!(combined[1].1 >= combined[2].1);
    }

    #[test]
    fn test_alpha_weighting() {
        let vector_results = vec![("doc1".into(), 1.0)];
        let mut bm25_scores = HashMap::new();
        bm25_scores.insert("doc2".into(), 1.0);

        // Pure vector (alpha = 1.0)
        let combined = combine_scores(vector_results.clone(), bm25_scores.clone(), 1.0);
        assert!(combined.iter().find(|(id, _)| id == "doc1").unwrap().1 > 0.5);

        // Pure keyword (alpha = 0.0)
        let combined = combine_scores(vector_results.clone(), bm25_scores.clone(), 0.0);
        assert!(combined.iter().find(|(id, _)| id == "doc2").unwrap().1 > 0.5);

        // Balanced (alpha = 0.5)
        let combined = combine_scores(vector_results, bm25_scores, 0.5);
        let doc1_score = combined.iter().find(|(id, _)| id == "doc1").unwrap().1;
        let doc2_score = combined.iter().find(|(id, _)| id == "doc2").unwrap().1;
        assert!((doc1_score - doc2_score).abs() < 0.1); // Should be similar
    }

    // =====================================================================
    // Phrase Matching Tests
    // =====================================================================

    #[test]
    fn test_phrase_search_exact_match() {
        let mut index = TextIndex::new();

        index.index_document("doc1".into(), "machine learning is amazing".into());
        index.index_document(
            "doc2".into(),
            "deep learning and machine intelligence".into(),
        );
        index.index_document("doc3".into(), "learning machine code".into());

        // Exact phrase "machine learning"
        let scores = index.phrase_search("machine learning");

        // doc1 has exact phrase
        assert!(
            scores.contains_key("doc1"),
            "doc1 should match exact phrase"
        );

        // doc2 has both words but not adjacent
        assert!(
            !scores.contains_key("doc2"),
            "doc2 should NOT match (words not adjacent)"
        );

        // doc3 has both words but in reverse order
        assert!(
            !scores.contains_key("doc3"),
            "doc3 should NOT match (reverse order)"
        );
    }

    #[test]
    fn test_phrase_search_multiple_occurrences() {
        let mut index = TextIndex::new();

        index.index_document(
            "doc1".into(),
            "the quick brown fox jumps over the lazy dog".into(),
        );
        index.index_document("doc2".into(), "a quick brown cat".into());
        index.index_document("doc3".into(), "brown quick animals".into());

        let scores = index.phrase_search("quick brown");

        assert!(scores.contains_key("doc1"), "doc1 should match");
        assert!(scores.contains_key("doc2"), "doc2 should match");
        assert!(
            !scores.contains_key("doc3"),
            "doc3 should NOT match (reverse order)"
        );
    }

    #[test]
    fn test_phrase_search_single_word() {
        let mut index = TextIndex::new();

        index.index_document("doc1".into(), "machine learning".into());
        index.index_document("doc2".into(), "deep learning".into());

        // Single word should fallback to BM25
        let scores = index.phrase_search("learning");

        assert!(scores.contains_key("doc1"));
        assert!(scores.contains_key("doc2"));
    }

    #[test]
    fn test_phrase_search_not_found() {
        let mut index = TextIndex::new();

        index.index_document("doc1".into(), "machine learning".into());
        index.index_document("doc2".into(), "deep learning".into());

        let scores = index.phrase_search("neural network");

        assert!(
            scores.is_empty(),
            "Should return empty for non-matching phrase"
        );
    }

    #[test]
    fn test_phrase_search_partial_match() {
        let mut index = TextIndex::new();

        index.index_document("doc1".into(), "natural language processing".into());
        index.index_document("doc2".into(), "natural and language models".into());

        let scores = index.phrase_search("natural language");

        // doc1 has exact phrase
        assert!(
            scores.contains_key("doc1"),
            "doc1 should match exact phrase"
        );

        // doc2 has both words but not consecutive
        assert!(
            !scores.contains_key("doc2"),
            "doc2 should NOT match (not consecutive)"
        );
    }

    #[test]
    fn test_phrase_search_boost() {
        let mut index = TextIndex::new();

        // Add multiple documents to get positive IDF scores
        index.index_document("doc1".into(), "machine learning is powerful".into());
        index.index_document("doc2".into(), "database systems are robust".into());
        index.index_document("doc3".into(), "artificial intelligence rocks".into());
        index.index_document("doc4".into(), "web development techniques".into());

        // Phrase search should have higher score than regular BM25
        let phrase_score = index
            .phrase_search("machine learning")
            .get("doc1")
            .copied()
            .unwrap_or(0.0);
        let bm25_score = index
            .bm25_scores("machine learning")
            .get("doc1")
            .copied()
            .unwrap_or(0.0);

        assert!(
            phrase_score > 0.0,
            "Phrase score should be positive, got {}",
            phrase_score
        );
        assert!(
            bm25_score > 0.0,
            "BM25 score should be positive, got {}",
            bm25_score
        );
        assert!(
            phrase_score > bm25_score,
            "Phrase score ({}) should be higher than BM25 score ({})",
            phrase_score,
            bm25_score
        );

        // Verify boost is approximately 2x
        let boost_ratio = phrase_score / bm25_score;
        assert!(
            (boost_ratio - 2.0).abs() < 0.1,
            "Boost ratio should be ~2.0, got {}",
            boost_ratio
        );
    }

    #[test]
    fn test_phrase_search_with_stopwords() {
        let mut index = TextIndex::with_tokenizer(Box::new(LanguageTokenizer::english()));

        index.index_document("doc1".into(), "The quick brown fox jumps".into());
        index.index_document("doc2".into(), "quick brown animals".into());

        // "the" is a stopword and will be removed
        let scores = index.phrase_search("the quick brown");

        // After stopword removal, becomes "quick brown"
        assert!(scores.contains_key("doc1"), "doc1 should match");
        assert!(scores.contains_key("doc2"), "doc2 should match");
    }

    #[test]
    fn test_phrase_search_case_insensitive() {
        let mut index = TextIndex::new();

        index.index_document("doc1".into(), "Machine Learning Algorithms".into());

        let scores = index.phrase_search("machine learning");

        assert!(
            scores.contains_key("doc1"),
            "Should match case-insensitively"
        );
    }

    #[test]
    fn test_phrase_search_with_punctuation() {
        let mut index = TextIndex::new();

        index.index_document("doc1".into(), "Hello, world! How are you?".into());

        // Punctuation removed by SimpleTokenizer
        let scores = index.phrase_search("hello world");

        assert!(
            scores.contains_key("doc1"),
            "Should match after punctuation removal"
        );
    }

    #[test]
    fn test_phrase_search_long_phrase() {
        let mut index = TextIndex::new();

        index.index_document(
            "doc1".into(),
            "artificial intelligence and machine learning are transforming technology".into(),
        );
        index.index_document(
            "doc2".into(),
            "machine learning in artificial intelligence".into(),
        );

        let scores = index.phrase_search("artificial intelligence and machine learning");

        assert!(scores.contains_key("doc1"), "doc1 should match long phrase");
        assert!(
            !scores.contains_key("doc2"),
            "doc2 should NOT match (different word order)"
        );
    }

    #[test]
    fn test_positional_index_accuracy() {
        let mut index = TextIndex::new();

        index.index_document("doc1".into(), "the the the test".into());

        // Verify positions are tracked correctly for repeated words
        let postings = index.inverted_index.get("the").unwrap();
        let doc1_posting = postings.iter().find(|p| p.doc_id == "doc1").unwrap();

        assert_eq!(
            doc1_posting.term_freq, 3,
            "Should have 3 occurrences of 'the'"
        );
        assert_eq!(
            doc1_posting.positions,
            vec![0, 1, 2],
            "Positions should be [0, 1, 2]"
        );
    }
}
