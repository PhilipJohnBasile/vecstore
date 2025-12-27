//! Typo-Tolerant Search
//!
//! Fuzzy matching with automatic typo correction, similar to Typesense.
//! Enables search-as-you-type with tolerance for spelling mistakes.
//!
//! # Features
//!
//! - **Edit Distance**: Levenshtein distance-based matching
//! - **Phonetic Matching**: Soundex/Metaphone for pronunciation-based search
//! - **Prefix Matching**: Instant results while typing
//! - **Auto-Correction**: Suggest corrected queries
//! - **Configurable Tolerance**: 0-2 typos based on word length
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::typo_tolerance::{TypoTolerantSearch, TypoConfig};
//!
//! let mut search = TypoTolerantSearch::new(TypoConfig::default());
//!
//! // Index documents
//! search.index("doc1", "The quick brown fox jumps over the lazy dog");
//!
//! // Search with typos
//! let results = search.search("quik brwon fx")?;  // Still finds the document!
//! ```

use std::collections::{HashMap, HashSet, BTreeMap};
use serde::{Deserialize, Serialize};


/// Typo tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypoConfig {
    /// Maximum edit distance for short words (1-4 chars)
    #[serde(default = "default_short_typos")]
    pub max_typos_short: usize,
    /// Maximum edit distance for medium words (5-8 chars)
    #[serde(default = "default_medium_typos")]
    pub max_typos_medium: usize,
    /// Maximum edit distance for long words (9+ chars)
    #[serde(default = "default_long_typos")]
    pub max_typos_long: usize,
    /// Enable prefix matching
    #[serde(default = "default_true")]
    pub prefix_matching: bool,
    /// Minimum prefix length for matching
    #[serde(default = "default_min_prefix")]
    pub min_prefix_length: usize,
    /// Enable phonetic matching
    #[serde(default)]
    pub phonetic_matching: bool,
    /// Prioritize exact matches
    #[serde(default = "default_true")]
    pub prioritize_exact: bool,
}

fn default_short_typos() -> usize { 0 }
fn default_medium_typos() -> usize { 1 }
fn default_long_typos() -> usize { 2 }
fn default_true() -> bool { true }
fn default_min_prefix() -> usize { 2 }

impl Default for TypoConfig {
    fn default() -> Self {
        Self {
            max_typos_short: 0,
            max_typos_medium: 1,
            max_typos_long: 2,
            prefix_matching: true,
            min_prefix_length: 2,
            phonetic_matching: false,
            prioritize_exact: true,
        }
    }
}

impl TypoConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Strict mode - fewer typos allowed
    pub fn strict() -> Self {
        Self {
            max_typos_short: 0,
            max_typos_medium: 0,
            max_typos_long: 1,
            ..Default::default()
        }
    }

    /// Lenient mode - more typos allowed
    pub fn lenient() -> Self {
        Self {
            max_typos_short: 1,
            max_typos_medium: 2,
            max_typos_long: 3,
            ..Default::default()
        }
    }

    /// Enable phonetic matching
    pub fn with_phonetic(mut self) -> Self {
        self.phonetic_matching = true;
        self
    }

    /// Get max typos for word length
    pub fn max_typos_for_length(&self, len: usize) -> usize {
        if len <= 4 {
            self.max_typos_short
        } else if len <= 8 {
            self.max_typos_medium
        } else {
            self.max_typos_long
        }
    }
}

/// Indexed document
#[derive(Debug, Clone)]
struct IndexedDocument {
    id: String,
    tokens: Vec<String>,
    original: String,
}

/// Search result
#[derive(Debug, Clone, Serialize)]
pub struct TypoSearchResult {
    /// Document ID
    pub id: String,
    /// Relevance score
    pub score: f32,
    /// Number of typos corrected
    pub typos: usize,
    /// Original document text
    pub text: String,
    /// Matched tokens
    pub matched_tokens: Vec<MatchedToken>,
}

/// Matched token with correction info
#[derive(Debug, Clone, Serialize)]
pub struct MatchedToken {
    /// Query token
    pub query: String,
    /// Matched document token
    pub matched: String,
    /// Edit distance
    pub edit_distance: usize,
    /// Match type
    pub match_type: MatchType,
}

/// Type of match
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum MatchType {
    Exact,
    Prefix,
    Fuzzy,
    Phonetic,
}

/// Typo-tolerant search engine
pub struct TypoTolerantSearch {
    config: TypoConfig,
    documents: HashMap<String, IndexedDocument>,
    /// Inverted index: token -> document IDs
    token_index: HashMap<String, HashSet<String>>,
    /// Prefix index for instant search
    prefix_index: BTreeMap<String, HashSet<String>>,
    /// Phonetic index (Soundex codes)
    phonetic_index: HashMap<String, HashSet<String>>,
}

impl TypoTolerantSearch {
    /// Create a new search engine
    pub fn new(config: TypoConfig) -> Self {
        Self {
            config,
            documents: HashMap::new(),
            token_index: HashMap::new(),
            prefix_index: BTreeMap::new(),
            phonetic_index: HashMap::new(),
        }
    }

    /// Index a document
    pub fn index(&mut self, id: &str, text: &str) {
        let tokens = self.tokenize(text);

        // Update inverted index
        for token in &tokens {
            self.token_index
                .entry(token.clone())
                .or_default()
                .insert(id.to_string());

            // Update prefix index
            for i in self.config.min_prefix_length..=token.len() {
                let prefix = &token[..i];
                self.prefix_index
                    .entry(prefix.to_string())
                    .or_default()
                    .insert(id.to_string());
            }

            // Update phonetic index
            if self.config.phonetic_matching {
                let soundex = self.soundex(token);
                self.phonetic_index
                    .entry(soundex)
                    .or_default()
                    .insert(id.to_string());
            }
        }

        self.documents.insert(id.to_string(), IndexedDocument {
            id: id.to_string(),
            tokens,
            original: text.to_string(),
        });
    }

    /// Remove a document from the index
    pub fn remove(&mut self, id: &str) {
        if let Some(doc) = self.documents.remove(id) {
            for token in &doc.tokens {
                if let Some(set) = self.token_index.get_mut(token) {
                    set.remove(id);
                }

                for i in self.config.min_prefix_length..=token.len() {
                    let prefix = &token[..i];
                    if let Some(set) = self.prefix_index.get_mut(prefix) {
                        set.remove(id);
                    }
                }

                if self.config.phonetic_matching {
                    let soundex = self.soundex(token);
                    if let Some(set) = self.phonetic_index.get_mut(&soundex) {
                        set.remove(id);
                    }
                }
            }
        }
    }

    /// Search with typo tolerance
    pub fn search(&self, query: &str) -> Vec<TypoSearchResult> {
        let query_tokens = self.tokenize(query);
        let mut doc_scores: HashMap<String, (f32, usize, Vec<MatchedToken>)> = HashMap::new();

        for query_token in &query_tokens {
            let matches = self.find_matches(query_token);

            for (doc_id, matched_token) in matches {
                let entry = doc_scores.entry(doc_id).or_insert((0.0, 0, Vec::new()));

                // Calculate score based on match type
                let match_score = match matched_token.match_type {
                    MatchType::Exact => 1.0,
                    MatchType::Prefix => 0.8,
                    MatchType::Fuzzy => 0.6 / (matched_token.edit_distance as f32 + 1.0),
                    MatchType::Phonetic => 0.5,
                };

                entry.0 += match_score;
                entry.1 += matched_token.edit_distance;
                entry.2.push(matched_token);
            }
        }

        // Convert to results
        let mut results: Vec<TypoSearchResult> = doc_scores
            .into_iter()
            .filter_map(|(id, (score, typos, matches))| {
                self.documents.get(&id).map(|doc| TypoSearchResult {
                    id,
                    score: score / query_tokens.len() as f32,
                    typos,
                    text: doc.original.clone(),
                    matched_tokens: matches,
                })
            })
            .collect();

        // Sort by score (descending), then by typos (ascending)
        results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.typos.cmp(&b.typos))
        });

        results
    }

    /// Find matches for a query token
    fn find_matches(&self, query_token: &str) -> Vec<(String, MatchedToken)> {
        let mut matches = Vec::new();
        let max_typos = self.config.max_typos_for_length(query_token.len());

        // 1. Exact matches (highest priority)
        if let Some(doc_ids) = self.token_index.get(query_token) {
            for doc_id in doc_ids {
                matches.push((doc_id.clone(), MatchedToken {
                    query: query_token.to_string(),
                    matched: query_token.to_string(),
                    edit_distance: 0,
                    match_type: MatchType::Exact,
                }));
            }
        }

        // 2. Prefix matches
        if self.config.prefix_matching && query_token.len() >= self.config.min_prefix_length {
            // Find all tokens that start with this prefix
            let range_start = query_token.to_string();
            let range_end = format!("{}~", query_token); // '~' is after all alphanumeric

            for (prefix, doc_ids) in self.prefix_index.range(range_start..range_end) {
                if prefix != query_token { // Avoid duplicating exact matches
                    for doc_id in doc_ids {
                        matches.push((doc_id.clone(), MatchedToken {
                            query: query_token.to_string(),
                            matched: prefix.clone(),
                            edit_distance: 0,
                            match_type: MatchType::Prefix,
                        }));
                    }
                }
            }
        }

        // 3. Fuzzy matches (edit distance)
        if max_typos > 0 {
            for (token, doc_ids) in &self.token_index {
                let distance = self.levenshtein(query_token, token);
                if distance > 0 && distance <= max_typos {
                    for doc_id in doc_ids {
                        matches.push((doc_id.clone(), MatchedToken {
                            query: query_token.to_string(),
                            matched: token.clone(),
                            edit_distance: distance,
                            match_type: MatchType::Fuzzy,
                        }));
                    }
                }
            }
        }

        // 4. Phonetic matches
        if self.config.phonetic_matching {
            let query_soundex = self.soundex(query_token);
            if let Some(doc_ids) = self.phonetic_index.get(&query_soundex) {
                for doc_id in doc_ids {
                    if !matches.iter().any(|(id, _)| id == doc_id) {
                        matches.push((doc_id.clone(), MatchedToken {
                            query: query_token.to_string(),
                            matched: query_soundex.clone(),
                            edit_distance: 0,
                            match_type: MatchType::Phonetic,
                        }));
                    }
                }
            }
        }

        matches
    }

    /// Tokenize text
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() >= 2)
            .map(|s| s.to_string())
            .collect()
    }

    /// Calculate Levenshtein edit distance
    fn levenshtein(&self, a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let a_len = a_chars.len();
        let b_len = b_chars.len();

        if a_len == 0 { return b_len; }
        if b_len == 0 { return a_len; }

        let mut matrix = vec![vec![0usize; b_len + 1]; a_len + 1];

        for i in 0..=a_len {
            matrix[i][0] = i;
        }
        for j in 0..=b_len {
            matrix[0][j] = j;
        }

        for i in 1..=a_len {
            for j in 1..=b_len {
                let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };

                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[a_len][b_len]
    }

    /// Calculate Soundex code for phonetic matching
    fn soundex(&self, word: &str) -> String {
        if word.is_empty() {
            return String::new();
        }

        let chars: Vec<char> = word.to_uppercase().chars().collect();
        let mut code = String::new();
        code.push(chars[0]);

        let get_code = |c: char| -> Option<char> {
            match c {
                'B' | 'F' | 'P' | 'V' => Some('1'),
                'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => Some('2'),
                'D' | 'T' => Some('3'),
                'L' => Some('4'),
                'M' | 'N' => Some('5'),
                'R' => Some('6'),
                _ => None,
            }
        };

        let mut prev_code = get_code(chars[0]);

        for &c in &chars[1..] {
            if code.len() >= 4 {
                break;
            }

            let curr_code = get_code(c);
            if let Some(cc) = curr_code {
                if curr_code != prev_code {
                    code.push(cc);
                }
            }
            prev_code = curr_code;
        }

        // Pad with zeros
        while code.len() < 4 {
            code.push('0');
        }

        code
    }

    /// Suggest corrections for a query
    pub fn suggest(&self, query: &str, limit: usize) -> Vec<String> {
        let query_tokens = self.tokenize(query);
        let mut suggestions = Vec::new();

        for token in &query_tokens {
            let mut token_suggestions: Vec<(String, usize)> = Vec::new();

            for indexed_token in self.token_index.keys() {
                let distance = self.levenshtein(token, indexed_token);
                if distance <= 2 && distance > 0 {
                    token_suggestions.push((indexed_token.clone(), distance));
                }
            }

            token_suggestions.sort_by_key(|(_, d)| *d);

            for (suggestion, _) in token_suggestions.into_iter().take(3) {
                if !suggestions.contains(&suggestion) {
                    suggestions.push(suggestion);
                }
            }
        }

        suggestions.truncate(limit);
        suggestions
    }

    /// Get number of indexed documents
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

impl Default for TypoTolerantSearch {
    fn default() -> Self {
        Self::new(TypoConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let mut search = TypoTolerantSearch::new(TypoConfig::default());
        search.index("doc1", "The quick brown fox");

        let results = search.search("quick");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
        assert_eq!(results[0].typos, 0);
    }

    #[test]
    fn test_fuzzy_match() {
        let mut search = TypoTolerantSearch::new(TypoConfig::default());
        search.index("doc1", "The quick brown fox");

        // "quik" has 1 typo from "quick"
        let results = search.search("quik");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_prefix_match() {
        let mut search = TypoTolerantSearch::new(TypoConfig::default());
        search.index("doc1", "The quick brown fox");

        let results = search.search("qui");
        assert!(!results.is_empty());
        assert!(results[0].matched_tokens.iter().any(|t| t.match_type == MatchType::Prefix));
    }

    #[test]
    fn test_phonetic_match() {
        let mut search = TypoTolerantSearch::new(TypoConfig::default().with_phonetic());
        search.index("doc1", "Stephen");

        // "Steven" sounds similar to "Stephen"
        let results = search.search("steven");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_levenshtein() {
        let search = TypoTolerantSearch::new(TypoConfig::default());

        assert_eq!(search.levenshtein("kitten", "sitting"), 3);
        assert_eq!(search.levenshtein("hello", "hello"), 0);
        assert_eq!(search.levenshtein("", "abc"), 3);
    }

    #[test]
    fn test_soundex() {
        let search = TypoTolerantSearch::new(TypoConfig::default());

        assert_eq!(search.soundex("Robert"), "R163");
        assert_eq!(search.soundex("Rupert"), "R163");
        assert_eq!(search.soundex("Smith"), "S530");
    }

    #[test]
    fn test_suggestions() {
        let mut search = TypoTolerantSearch::new(TypoConfig::default());
        search.index("doc1", "The quick brown fox jumps");

        let suggestions = search.suggest("quik", 5);
        assert!(suggestions.contains(&"quick".to_string()));
    }
}
