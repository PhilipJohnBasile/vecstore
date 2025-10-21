//! Fuzzy search with typo tolerance
//!
//! This module provides fuzzy string matching capabilities for handling typos,
//! misspellings, and approximate text matching in search queries.
//!
//! ## Features
//!
//! - **Levenshtein distance** - Classic edit distance metric
//! - **Damerau-Levenshtein distance** - Includes transpositions
//! - **Fuzzy matching** - Find strings within edit distance threshold
//! - **BK-tree** - Efficient fuzzy search data structure
//! - **Query correction** - Suggest corrections for misspelled queries
//!
//! ## Algorithms
//!
//! ### Levenshtein Distance
//! Minimum number of single-character edits (insertions, deletions, substitutions)
//! required to change one string into another.
//!
//! ### Damerau-Levenshtein Distance
//! Extends Levenshtein to include transpositions (swapping adjacent characters).
//! Handles common typos like "teh" → "the".
//!
//! ### BK-tree (Burkhard-Keller tree)
//! Metric tree for efficient fuzzy search. Allows finding all strings within
//! edit distance k in O(log n) average time.
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::fuzzy::{levenshtein_distance, FuzzyMatcher, BKTree};
//!
//! // Simple distance calculation
//! let dist = levenshtein_distance("kitten", "sitting");
//! assert_eq!(dist, 3);
//!
//! // Fuzzy matching
//! let matcher = FuzzyMatcher::new(2); // max edit distance = 2
//! assert!(matcher.is_match("hello", "helo"));
//!
//! // BK-tree for efficient search
//! let mut tree = BKTree::new();
//! tree.insert("hello".to_string());
//! tree.insert("world".to_string());
//!
//! let matches = tree.search("helo", 1); // Find within edit distance 1
//! ```

use std::cmp::min;
use std::collections::HashMap;

/// Calculate Levenshtein distance between two strings
///
/// The Levenshtein distance is the minimum number of single-character edits
/// (insertions, deletions, or substitutions) required to change one string into another.
///
/// ## Complexity
/// - Time: O(m * n) where m, n are string lengths
/// - Space: O(min(m, n)) using optimized implementation
///
/// ## Example
///
/// ```
/// use vecstore::fuzzy::levenshtein_distance;
///
/// assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
/// assert_eq!(levenshtein_distance("hello", "helo"), 1);
/// assert_eq!(levenshtein_distance("same", "same"), 0);
/// ```
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use two rows for space optimization
    let mut prev_row: Vec<usize> = (0..=n).collect();
    let mut curr_row: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr_row[0] = i;

        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            curr_row[j] = min(
                min(
                    prev_row[j] + 1,     // deletion
                    curr_row[j - 1] + 1, // insertion
                ),
                prev_row[j - 1] + cost, // substitution
            );
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
}

/// Calculate Damerau-Levenshtein distance between two strings
///
/// Extends Levenshtein distance to include transpositions (swapping adjacent characters).
/// This better handles common typos like "teh" → "the".
///
/// ## Example
///
/// ```
/// use vecstore::fuzzy::damerau_levenshtein_distance;
///
/// // Transposition: "ab" -> "ba" is 1 edit (not 2 as in Levenshtein)
/// assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
/// assert_eq!(damerau_levenshtein_distance("the", "teh"), 1);
/// ```
pub fn damerau_levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Create distance matrix
    let mut d = vec![vec![0; n + 1]; m + 1];

    // Initialize first row and column
    for i in 0..=m {
        d[i][0] = i;
    }
    for j in 0..=n {
        d[0][j] = j;
    }

    // Fill matrix
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            d[i][j] = min(
                min(
                    d[i - 1][j] + 1, // deletion
                    d[i][j - 1] + 1, // insertion
                ),
                d[i - 1][j - 1] + cost, // substitution
            );

            // Transposition
            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1);
            }
        }
    }

    d[m][n]
}

/// Calculate normalized similarity score (0.0 to 1.0)
///
/// Returns 1.0 for identical strings, approaching 0.0 as they differ more.
///
/// ## Formula
/// ```text
/// similarity = 1.0 - (distance / max_length)
/// ```
///
/// ## Example
///
/// ```
/// use vecstore::fuzzy::similarity_score;
///
/// assert_eq!(similarity_score("hello", "hello"), 1.0);
/// assert!(similarity_score("hello", "helo") > 0.8);
/// ```
pub fn similarity_score(a: &str, b: &str) -> f32 {
    let dist = levenshtein_distance(a, b);
    let max_len = a.len().max(b.len());

    if max_len == 0 {
        return 1.0;
    }

    1.0 - (dist as f32 / max_len as f32)
}

/// Fuzzy string matcher with configurable edit distance threshold
///
/// ## Example
///
/// ```
/// use vecstore::fuzzy::FuzzyMatcher;
///
/// let matcher = FuzzyMatcher::new(2);
/// assert!(matcher.is_match("hello", "helo"));   // distance = 1 ≤ 2
/// assert!(!matcher.is_match("hello", "world")); // distance = 4 > 2
/// ```
pub struct FuzzyMatcher {
    /// Maximum edit distance for a match
    max_distance: usize,

    /// Whether to use Damerau-Levenshtein (includes transpositions)
    use_damerau: bool,

    /// Case-sensitive matching
    case_sensitive: bool,
}

impl FuzzyMatcher {
    /// Create a new fuzzy matcher
    ///
    /// # Arguments
    ///
    /// * `max_distance` - Maximum edit distance for a match
    pub fn new(max_distance: usize) -> Self {
        Self {
            max_distance,
            use_damerau: true,
            case_sensitive: false,
        }
    }

    /// Enable/disable Damerau-Levenshtein distance (default: enabled)
    pub fn with_damerau(mut self, enabled: bool) -> Self {
        self.use_damerau = enabled;
        self
    }

    /// Enable/disable case-sensitive matching (default: disabled)
    pub fn with_case_sensitive(mut self, enabled: bool) -> Self {
        self.case_sensitive = enabled;
        self
    }

    /// Check if two strings match within the edit distance threshold
    pub fn is_match(&self, a: &str, b: &str) -> bool {
        let (a, b) = if !self.case_sensitive {
            (a.to_lowercase(), b.to_lowercase())
        } else {
            (a.to_string(), b.to_string())
        };

        let distance = if self.use_damerau {
            damerau_levenshtein_distance(&a, &b)
        } else {
            levenshtein_distance(&a, &b)
        };

        distance <= self.max_distance
    }

    /// Get the distance between two strings
    pub fn distance(&self, a: &str, b: &str) -> usize {
        let (a, b) = if !self.case_sensitive {
            (a.to_lowercase(), b.to_lowercase())
        } else {
            (a.to_string(), b.to_string())
        };

        if self.use_damerau {
            damerau_levenshtein_distance(&a, &b)
        } else {
            levenshtein_distance(&a, &b)
        }
    }

    /// Find the best match from a list of candidates
    ///
    /// Returns (index, distance) of the closest match, or None if no match within threshold.
    pub fn find_best_match(&self, query: &str, candidates: &[String]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;

        for (i, candidate) in candidates.iter().enumerate() {
            let dist = self.distance(query, candidate);

            if dist <= self.max_distance {
                if let Some((_, best_dist)) = best {
                    if dist < best_dist {
                        best = Some((i, dist));
                    }
                } else {
                    best = Some((i, dist));
                }
            }
        }

        best
    }

    /// Find all matches within the edit distance threshold
    ///
    /// Returns indices of all matching candidates.
    pub fn find_all_matches(&self, query: &str, candidates: &[String]) -> Vec<usize> {
        candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| self.is_match(query, candidate))
            .map(|(i, _)| i)
            .collect()
    }
}

/// BK-tree (Burkhard-Keller tree) for efficient fuzzy search
///
/// A metric tree that allows finding all strings within edit distance k in O(log n) average time.
///
/// ## Use Cases
///
/// - Spell checking
/// - Query correction
/// - Fuzzy autocomplete
/// - Duplicate detection
///
/// ## Example
///
/// ```
/// use vecstore::fuzzy::BKTree;
///
/// let mut tree = BKTree::new();
/// tree.insert("hello".to_string());
/// tree.insert("help".to_string());
/// tree.insert("world".to_string());
///
/// // Find all words within edit distance 1 of "helo"
/// let matches = tree.search("helo", 1);
/// assert!(matches.contains(&"hello".to_string()));
/// ```
pub struct BKTree {
    root: Option<Box<BKNode>>,
}

struct BKNode {
    word: String,
    children: HashMap<usize, Box<BKNode>>,
}

impl BKTree {
    /// Create a new empty BK-tree
    pub fn new() -> Self {
        Self { root: None }
    }

    /// Insert a word into the tree
    pub fn insert(&mut self, word: String) {
        if let Some(ref mut root) = self.root {
            root.insert(word);
        } else {
            self.root = Some(Box::new(BKNode {
                word,
                children: HashMap::new(),
            }));
        }
    }

    /// Search for words within edit distance k
    ///
    /// Returns all words in the tree that are within edit distance k of the query.
    pub fn search(&self, query: &str, max_distance: usize) -> Vec<String> {
        let mut results = Vec::new();

        if let Some(ref root) = self.root {
            root.search(query, max_distance, &mut results);
        }

        results
    }

    /// Find the closest word in the tree
    ///
    /// Returns (word, distance) of the closest match.
    pub fn find_closest(&self, query: &str) -> Option<(String, usize)> {
        if let Some(ref root) = self.root {
            let mut best: Option<(String, usize)> = None;
            root.find_closest(query, &mut best);
            best
        } else {
            None
        }
    }

    /// Get the number of words in the tree
    pub fn len(&self) -> usize {
        if let Some(ref root) = self.root {
            root.count()
        } else {
            0
        }
    }

    /// Check if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Build a BK-tree from a collection of words
    pub fn from_words<I>(words: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        let mut tree = Self::new();
        for word in words {
            tree.insert(word);
        }
        tree
    }
}

impl Default for BKTree {
    fn default() -> Self {
        Self::new()
    }
}

impl BKNode {
    fn insert(&mut self, word: String) {
        let distance = levenshtein_distance(&self.word, &word);

        if distance == 0 {
            // Duplicate word, ignore
            return;
        }

        if let Some(child) = self.children.get_mut(&distance) {
            child.insert(word);
        } else {
            self.children.insert(
                distance,
                Box::new(BKNode {
                    word,
                    children: HashMap::new(),
                }),
            );
        }
    }

    fn search(&self, query: &str, max_distance: usize, results: &mut Vec<String>) {
        let distance = levenshtein_distance(&self.word, query);

        if distance <= max_distance {
            results.push(self.word.clone());
        }

        // Triangle inequality: only search children within range
        let min_dist = distance.saturating_sub(max_distance);
        let max_dist = distance + max_distance;

        for (child_dist, child) in &self.children {
            if *child_dist >= min_dist && *child_dist <= max_dist {
                child.search(query, max_distance, results);
            }
        }
    }

    fn find_closest(&self, query: &str, best: &mut Option<(String, usize)>) {
        let distance = levenshtein_distance(&self.word, query);

        match best {
            Some((_, best_dist)) => {
                if distance < *best_dist {
                    *best = Some((self.word.clone(), distance));
                }
            }
            None => {
                *best = Some((self.word.clone(), distance));
            }
        }

        // Search children
        for child in self.children.values() {
            child.find_closest(query, best);
        }
    }

    fn count(&self) -> usize {
        1 + self.children.values().map(|c| c.count()).sum::<usize>()
    }
}

/// Suggest corrections for a misspelled query
///
/// Uses a dictionary of known words to find the closest matches.
///
/// ## Example
///
/// ```
/// use vecstore::fuzzy::suggest_corrections;
///
/// let dictionary = vec![
///     "hello".to_string(),
///     "help".to_string(),
///     "world".to_string(),
/// ];
///
/// let suggestions = suggest_corrections("helo", &dictionary, 1, 3);
/// assert!(suggestions.contains(&"hello".to_string()));
/// ```
pub fn suggest_corrections(
    query: &str,
    dictionary: &[String],
    max_distance: usize,
    max_suggestions: usize,
) -> Vec<String> {
    let mut candidates: Vec<(usize, String)> = dictionary
        .iter()
        .map(|word| {
            let dist = levenshtein_distance(query, word);
            (dist, word.clone())
        })
        .filter(|(dist, _)| *dist <= max_distance)
        .collect();

    // Sort by distance (ascending)
    candidates.sort_by_key(|(dist, _)| *dist);

    // Take top N
    candidates
        .into_iter()
        .take(max_suggestions)
        .map(|(_, word)| word)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("saturday", "sunday"), 3);
        assert_eq!(levenshtein_distance("hello", "helo"), 1);
    }

    #[test]
    fn test_damerau_levenshtein_distance() {
        // Transposition
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
        assert_eq!(damerau_levenshtein_distance("the", "teh"), 1);

        // Should give same results as Levenshtein for non-transposition cases
        assert_eq!(damerau_levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_similarity_score() {
        assert_eq!(similarity_score("hello", "hello"), 1.0);
        assert_eq!(similarity_score("", ""), 1.0);

        let score = similarity_score("hello", "helo");
        assert!(score >= 0.8 && score < 1.0); // distance=1, max_len=5, score=0.8
    }

    #[test]
    fn test_fuzzy_matcher() {
        let matcher = FuzzyMatcher::new(2);

        assert!(matcher.is_match("hello", "hello"));
        assert!(matcher.is_match("hello", "helo"));
        assert!(matcher.is_match("hello", "hallo"));
        assert!(!matcher.is_match("hello", "world"));
    }

    #[test]
    fn test_fuzzy_matcher_case_insensitive() {
        let matcher = FuzzyMatcher::new(1).with_case_sensitive(false);

        assert!(matcher.is_match("Hello", "hello"));
        assert!(matcher.is_match("HELLO", "helo"));
    }

    #[test]
    fn test_fuzzy_matcher_case_sensitive() {
        // With max_distance=0, only exact matches work
        let matcher = FuzzyMatcher::new(0).with_case_sensitive(true);

        assert!(!matcher.is_match("Hello", "hello")); // Different case, not exact match
        assert!(matcher.is_match("hello", "hello")); // Exact match
        assert!(matcher.is_match("Hello", "Hello")); // Exact match

        // With case-insensitive, "Hello" and "hello" should match with distance 0
        let matcher_ci = FuzzyMatcher::new(0).with_case_sensitive(false);
        assert!(matcher_ci.is_match("Hello", "hello"));
    }

    #[test]
    fn test_fuzzy_matcher_find_best_match() {
        let matcher = FuzzyMatcher::new(2);
        let candidates = vec!["hello".to_string(), "world".to_string(), "help".to_string()];

        let best = matcher.find_best_match("helo", &candidates);
        assert_eq!(best, Some((0, 1))); // "hello" at index 0, distance 1
    }

    #[test]
    fn test_fuzzy_matcher_find_all_matches() {
        let matcher = FuzzyMatcher::new(2);
        let candidates = vec!["hello".to_string(), "halo".to_string(), "world".to_string()];

        let matches = matcher.find_all_matches("helo", &candidates);
        assert_eq!(matches.len(), 2); // "hello" and "halo"
    }

    #[test]
    fn test_bk_tree_insert_and_search() {
        let mut tree = BKTree::new();

        tree.insert("hello".to_string());
        tree.insert("help".to_string());
        tree.insert("world".to_string());

        assert_eq!(tree.len(), 3);

        // "helo" is within distance 1 of both "hello" and "help"
        let matches = tree.search("helo", 1);
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&"hello".to_string()));
        assert!(matches.contains(&"help".to_string()));

        // "world" is not within distance 1
        let matches2 = tree.search("wrld", 1);
        assert_eq!(matches2.len(), 1);
        assert!(matches2.contains(&"world".to_string()));
    }

    #[test]
    fn test_bk_tree_find_closest() {
        let mut tree = BKTree::new();

        tree.insert("hello".to_string());
        tree.insert("world".to_string());
        tree.insert("help".to_string());

        let closest = tree.find_closest("helo");
        assert_eq!(closest, Some(("hello".to_string(), 1)));
    }

    #[test]
    fn test_bk_tree_from_words() {
        let words = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];

        let tree = BKTree::from_words(words);
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_bk_tree_empty() {
        let tree = BKTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_suggest_corrections() {
        let dictionary = vec![
            "hello".to_string(),
            "help".to_string(),
            "world".to_string(),
            "word".to_string(),
        ];

        let suggestions = suggest_corrections("helo", &dictionary, 2, 3);

        assert!(suggestions.contains(&"hello".to_string()));
        assert!(suggestions.len() <= 3);

        // Should be sorted by distance
        if suggestions.len() >= 2 {
            let dist1 = levenshtein_distance("helo", &suggestions[0]);
            let dist2 = levenshtein_distance("helo", &suggestions[1]);
            assert!(dist1 <= dist2);
        }
    }

    #[test]
    fn test_bk_tree_duplicates() {
        let mut tree = BKTree::new();

        tree.insert("hello".to_string());
        tree.insert("hello".to_string()); // Duplicate

        assert_eq!(tree.len(), 1); // Should not increase
    }
}
