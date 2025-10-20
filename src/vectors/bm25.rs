//! BM25 scoring algorithm for sparse vector keyword search
//!
//! BM25 (Best Matching 25) is a probabilistic ranking function used for
//! information retrieval. It's widely used in search engines and is the
//! foundation of many modern keyword search systems.
//!
//! This implementation is optimized for sparse vectors, making it efficient
//! for large vocabulary spaces where most documents contain only a small
//! subset of terms.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Field weights for BM25F (multi-field scoring)
///
/// Maps field names to boost factors. Higher boost = more important field.
///
/// # Example
/// ```
/// use std::collections::HashMap;
///
/// let mut field_weights = HashMap::new();
/// field_weights.insert("title".to_string(), 3.0);    // Title 3x more important
/// field_weights.insert("abstract".to_string(), 2.0); // Abstract 2x
/// field_weights.insert("content".to_string(), 1.0);  // Content baseline
/// ```
pub type FieldWeights = HashMap<String, f32>;

/// BM25 configuration parameters
///
/// These parameters control how BM25 scores documents:
/// - k1: Controls term frequency saturation (typical: 1.2-2.0)
/// - b: Controls document length normalization (typical: 0.75)
///
/// # References
/// Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Config {
    /// k1 parameter: Controls term frequency saturation
    ///
    /// Higher values give more weight to term frequency.
    /// - k1 = 0: Binary (term present/absent)
    /// - k1 = 1.2: Default, balanced
    /// - k1 = 2.0: More emphasis on frequency
    pub k1: f32,

    /// b parameter: Controls document length normalization
    ///
    /// Controls how much document length affects the score.
    /// - b = 0: No length normalization
    /// - b = 0.75: Default, balanced
    /// - b = 1.0: Full length normalization
    pub b: f32,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Statistics needed for BM25 scoring across a corpus
#[derive(Debug, Clone)]
pub struct BM25Stats {
    /// Average document length in the corpus (in number of terms)
    pub avg_doc_length: f32,

    /// Inverse document frequency (IDF) for each term
    /// Map: term_index -> IDF score
    pub idf: HashMap<usize, f32>,

    /// Total number of documents in corpus
    pub num_docs: usize,
}

impl BM25Stats {
    /// Create BM25 statistics from a corpus of sparse vectors
    ///
    /// # Arguments
    /// * `documents` - Iterator of (indices, values) pairs representing sparse documents
    ///
    /// # Returns
    /// BM25Stats with computed IDF scores and average document length
    pub fn from_corpus<'a, I>(documents: I) -> Self
    where
        I: Iterator<Item = (&'a [usize], &'a [f32])>,
    {
        let mut doc_count: HashMap<usize, usize> = HashMap::new();
        let mut total_doc_length = 0.0;
        let mut num_docs = 0;

        // Collect statistics
        for (indices, values) in documents {
            num_docs += 1;
            total_doc_length += values.iter().sum::<f32>();

            // Count documents containing each term
            for &term_idx in indices {
                *doc_count.entry(term_idx).or_insert(0) += 1;
            }
        }

        let avg_doc_length = if num_docs > 0 {
            total_doc_length / num_docs as f32
        } else {
            0.0
        };

        // Compute IDF for each term
        // IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        // where N = total docs, df(t) = docs containing term t
        let idf = doc_count
            .into_iter()
            .map(|(term_idx, df)| {
                let idf_score =
                    ((num_docs as f32 - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
                (term_idx, idf_score)
            })
            .collect();

        BM25Stats {
            avg_doc_length,
            idf,
            num_docs,
        }
    }

    /// Get IDF for a term, returning 0.0 if term not in corpus
    pub fn get_idf(&self, term_idx: usize) -> f32 {
        self.idf.get(&term_idx).copied().unwrap_or(0.0)
    }
}

/// Calculate BM25 score between a query and a document
///
/// # Arguments
/// * `query_indices` - Query term indices
/// * `query_weights` - Query term weights (typically 1.0 for each query term)
/// * `doc_indices` - Document term indices
/// * `doc_values` - Document term frequencies (raw counts or TF-IDF)
/// * `stats` - BM25 statistics from the corpus
/// * `config` - BM25 configuration parameters
///
/// # Returns
/// BM25 score (higher is better match)
///
/// # Example
/// ```
/// use vecstore::vectors::{bm25_score, BM25Config, BM25Stats};
/// use std::collections::HashMap;
///
/// // Simple corpus statistics
/// let mut idf = HashMap::new();
/// idf.insert(10, 2.0);  // Term 10 has IDF of 2.0
/// idf.insert(25, 1.5);  // Term 25 has IDF of 1.5
///
/// let stats = BM25Stats {
///     avg_doc_length: 100.0,
///     idf,
///     num_docs: 1000,
/// };
///
/// // Query: terms [10, 25]
/// let query_indices = vec![10, 25];
/// let query_weights = vec![1.0, 1.0];
///
/// // Document: terms [10, 25, 30] with frequencies [3.0, 2.0, 1.0]
/// let doc_indices = vec![10, 25, 30];
/// let doc_values = vec![3.0, 2.0, 1.0];
///
/// let score = bm25_score(
///     &query_indices,
///     &query_weights,
///     &doc_indices,
///     &doc_values,
///     &stats,
///     &BM25Config::default()
/// );
///
/// assert!(score > 0.0);
/// ```
pub fn bm25_score(
    query_indices: &[usize],
    query_weights: &[f32],
    doc_indices: &[usize],
    doc_values: &[f32],
    stats: &BM25Stats,
    config: &BM25Config,
) -> f32 {
    // Build document term map for O(1) lookup
    let doc_terms: HashMap<usize, f32> = doc_indices
        .iter()
        .zip(doc_values.iter())
        .map(|(&idx, &val)| (idx, val))
        .collect();

    // Document length (sum of all term frequencies)
    let doc_length = doc_values.iter().sum::<f32>();

    let mut score = 0.0;

    // For each query term, compute BM25 component
    for (&term_idx, &query_weight) in query_indices.iter().zip(query_weights.iter()) {
        // Skip if document doesn't contain this term
        let term_freq = match doc_terms.get(&term_idx) {
            Some(&tf) => tf,
            None => continue,
        };

        // Get IDF for this term
        let idf = stats.get_idf(term_idx);

        // BM25 formula:
        // score = IDF(t) * (f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
        //
        // where:
        // - IDF(t) = inverse document frequency of term t
        // - f(t,d) = frequency of term t in document d
        // - |d| = document length
        // - avgdl = average document length in corpus
        // - k1, b = tuning parameters

        let numerator = term_freq * (config.k1 + 1.0);
        let denominator =
            term_freq + config.k1 * (1.0 - config.b + config.b * doc_length / stats.avg_doc_length);

        // Multiply by query weight (typically 1.0, but can be used for query boosting)
        score += idf * query_weight * (numerator / denominator);
    }

    score
}

/// Calculate BM25 score with simplified interface (no pre-computed stats)
///
/// This is a convenience function for one-off scoring without building corpus statistics.
/// For batch scoring, use `bm25_score` with pre-computed `BM25Stats` for better performance.
///
/// # Arguments
/// * `query_indices` - Query term indices
/// * `doc_indices` - Document term indices
/// * `doc_values` - Document term frequencies
/// * `config` - BM25 configuration (or use `BM25Config::default()`)
///
/// # Returns
/// Simple frequency-based score (no IDF)
pub fn bm25_score_simple(
    query_indices: &[usize],
    doc_indices: &[usize],
    doc_values: &[f32],
    config: &BM25Config,
) -> f32 {
    let doc_terms: HashMap<usize, f32> = doc_indices
        .iter()
        .zip(doc_values.iter())
        .map(|(&idx, &val)| (idx, val))
        .collect();

    let doc_length = doc_values.iter().sum::<f32>();
    let avg_doc_length = doc_length; // Assume query doc is average

    let mut score = 0.0;

    for &term_idx in query_indices {
        let term_freq = match doc_terms.get(&term_idx) {
            Some(&tf) => tf,
            None => continue,
        };

        // Simplified BM25 without IDF (assumes IDF = 1.0)
        let numerator = term_freq * (config.k1 + 1.0);
        let denominator =
            term_freq + config.k1 * (1.0 - config.b + config.b * doc_length / avg_doc_length);

        score += numerator / denominator;
    }

    score
}

/// Calculate BM25F score with field boosting
///
/// BM25F extends BM25 to support multi-field documents where different fields
/// can have different importance weights (e.g., title more important than body).
///
/// This is the algorithm used by Weaviate, Elasticsearch, and other production systems.
///
/// # Arguments
/// * `query_indices` - Query term indices
/// * `query_weights` - Query term weights (typically 1.0 for each)
/// * `doc_fields` - Map of field_name -> (term_indices, term_values)
/// * `field_weights` - Map of field_name -> boost_factor (e.g., "title" -> 3.0)
/// * `stats` - BM25 statistics from the corpus
/// * `config` - BM25 configuration parameters
///
/// # Returns
/// BM25F score (higher is better match)
///
/// # Example
/// ```
/// use vecstore::vectors::{bm25f_score, BM25Config, BM25Stats, FieldWeights};
/// use std::collections::HashMap;
///
/// // Field weights: title is 3x more important than content
/// let mut field_weights: FieldWeights = HashMap::new();
/// field_weights.insert("title".to_string(), 3.0);
/// field_weights.insert("content".to_string(), 1.0);
///
/// // Document with multiple fields
/// let mut doc_fields = HashMap::new();
/// doc_fields.insert("title".to_string(), (vec![10, 25], vec![1.0, 1.0]));
/// doc_fields.insert("content".to_string(), (vec![10, 30, 40], vec![2.0, 1.0, 1.0]));
///
/// // Stats (simplified for example)
/// let mut idf = HashMap::new();
/// idf.insert(10, 2.0);
/// idf.insert(25, 1.5);
/// idf.insert(30, 1.0);
///
/// let stats = BM25Stats {
///     avg_doc_length: 10.0,
///     idf,
///     num_docs: 1000,
/// };
///
/// let query_indices = vec![10, 25];
/// let query_weights = vec![1.0, 1.0];
///
/// let score = bm25f_score(
///     &query_indices,
///     &query_weights,
///     &doc_fields,
///     &field_weights,
///     &stats,
///     &BM25Config::default()
/// );
///
/// assert!(score > 0.0);
/// ```
pub fn bm25f_score(
    query_indices: &[usize],
    query_weights: &[f32],
    doc_fields: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    field_weights: &FieldWeights,
    stats: &BM25Stats,
    config: &BM25Config,
) -> f32 {
    // BM25F algorithm:
    // 1. For each field, compute weighted term frequencies
    // 2. Combine weighted frequencies across fields
    // 3. Apply BM25 formula with combined frequencies

    // Build combined term frequency map across all fields
    let mut combined_tf: HashMap<usize, f32> = HashMap::new();
    let mut total_doc_length = 0.0;

    for (field_name, (indices, values)) in doc_fields {
        let boost = field_weights.get(field_name).copied().unwrap_or(1.0);
        let field_length: f32 = values.iter().sum();
        total_doc_length += field_length * boost;

        // Add weighted term frequencies from this field
        for (&term_idx, &freq) in indices.iter().zip(values.iter()) {
            *combined_tf.entry(term_idx).or_insert(0.0) += freq * boost;
        }
    }

    let mut score = 0.0;

    // For each query term, compute BM25F component
    for (&term_idx, &query_weight) in query_indices.iter().zip(query_weights.iter()) {
        // Get combined term frequency across all fields
        let term_freq = match combined_tf.get(&term_idx) {
            Some(&tf) => tf,
            None => continue,
        };

        // Get IDF for this term
        let idf = stats.get_idf(term_idx);

        // BM25F formula (same as BM25, but with field-weighted frequencies)
        let numerator = term_freq * (config.k1 + 1.0);
        let denominator = term_freq
            + config.k1 * (1.0 - config.b + config.b * total_doc_length / stats.avg_doc_length);

        score += idf * query_weight * (numerator / denominator);
    }

    score
}

/// Parse field weights from a string like "title^3" or "content^1.5"
///
/// # Example
/// ```
/// use vecstore::vectors::parse_field_weight;
///
/// assert_eq!(parse_field_weight("title^3"), ("title", 3.0));
/// assert_eq!(parse_field_weight("content^1.5"), ("content", 1.5));
/// assert_eq!(parse_field_weight("body"), ("body", 1.0)); // Default weight
/// ```
pub fn parse_field_weight(field_spec: &str) -> (&str, f32) {
    if let Some(pos) = field_spec.find('^') {
        let field = &field_spec[..pos];
        let weight_str = &field_spec[pos + 1..];
        let weight = weight_str.parse::<f32>().unwrap_or(1.0);
        (field, weight)
    } else {
        (field_spec, 1.0)
    }
}

/// Parse multiple field weight specifications
///
/// # Example
/// ```
/// use vecstore::vectors::parse_field_weights;
/// use std::collections::HashMap;
///
/// let fields = vec!["title^3", "abstract^2", "content"];
/// let weights = parse_field_weights(&fields);
///
/// assert_eq!(weights.get("title"), Some(&3.0));
/// assert_eq!(weights.get("abstract"), Some(&2.0));
/// assert_eq!(weights.get("content"), Some(&1.0));
/// ```
pub fn parse_field_weights(field_specs: &[&str]) -> FieldWeights {
    field_specs
        .iter()
        .map(|spec| {
            let (field, weight) = parse_field_weight(spec);
            (field.to_string(), weight)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_config_default() {
        let config = BM25Config::default();
        assert_eq!(config.k1, 1.2);
        assert_eq!(config.b, 0.75);
    }

    #[test]
    fn test_bm25_stats_from_corpus() {
        // Corpus: 3 documents
        // Doc 1: terms [1, 2, 3]
        // Doc 2: terms [1, 2]
        // Doc 3: terms [1, 4]
        let corpus = vec![
            (vec![1, 2, 3], vec![1.0, 1.0, 1.0]),
            (vec![1, 2], vec![1.0, 1.0]),
            (vec![1, 4], vec![1.0, 1.0]),
        ];

        let docs: Vec<(&[usize], &[f32])> = corpus
            .iter()
            .map(|(indices, values)| (indices.as_slice(), values.as_slice()))
            .collect();

        let stats = BM25Stats::from_corpus(docs.into_iter());

        assert_eq!(stats.num_docs, 3);
        assert_eq!(stats.avg_doc_length, (3.0 + 2.0 + 2.0) / 3.0);

        // Term 1 appears in all 3 docs
        let idf_1 = stats.get_idf(1);
        assert!(idf_1 > 0.0); // Should have some IDF

        // Term 2 appears in 2 docs
        let idf_2 = stats.get_idf(2);
        assert!(idf_2 > idf_1); // Should have higher IDF than term 1

        // Term 5 doesn't appear
        let idf_5 = stats.get_idf(5);
        assert_eq!(idf_5, 0.0);
    }

    #[test]
    fn test_bm25_score_exact_match() {
        // Query and document are identical
        let mut idf = HashMap::new();
        idf.insert(1, 1.0);
        idf.insert(2, 1.0);

        let stats = BM25Stats {
            avg_doc_length: 2.0,
            idf,
            num_docs: 100,
        };

        let query_indices = vec![1, 2];
        let query_weights = vec![1.0, 1.0];
        let doc_indices = vec![1, 2];
        let doc_values = vec![1.0, 1.0];

        let score = bm25_score(
            &query_indices,
            &query_weights,
            &doc_indices,
            &doc_values,
            &stats,
            &BM25Config::default(),
        );

        assert!(score > 0.0);
    }

    #[test]
    fn test_bm25_score_no_match() {
        // Query and document have no overlapping terms
        let mut idf = HashMap::new();
        idf.insert(1, 1.0);
        idf.insert(2, 1.0);
        idf.insert(3, 1.0);
        idf.insert(4, 1.0);

        let stats = BM25Stats {
            avg_doc_length: 2.0,
            idf,
            num_docs: 100,
        };

        let query_indices = vec![1, 2];
        let query_weights = vec![1.0, 1.0];
        let doc_indices = vec![3, 4];
        let doc_values = vec![1.0, 1.0];

        let score = bm25_score(
            &query_indices,
            &query_weights,
            &doc_indices,
            &doc_values,
            &stats,
            &BM25Config::default(),
        );

        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_bm25_score_partial_match() {
        // Query [1, 2], Document [1, 3]
        let mut idf = HashMap::new();
        idf.insert(1, 2.0);
        idf.insert(2, 2.0);
        idf.insert(3, 2.0);

        let stats = BM25Stats {
            avg_doc_length: 2.0,
            idf,
            num_docs: 100,
        };

        let query_indices = vec![1, 2];
        let query_weights = vec![1.0, 1.0];
        let doc_indices = vec![1, 3];
        let doc_values = vec![1.0, 1.0];

        let score = bm25_score(
            &query_indices,
            &query_weights,
            &doc_indices,
            &doc_values,
            &stats,
            &BM25Config::default(),
        );

        // Should score > 0 because term 1 matches
        assert!(score > 0.0);
    }

    #[test]
    fn test_bm25_score_frequency_matters() {
        // Higher term frequency should yield higher score
        let mut idf = HashMap::new();
        idf.insert(1, 2.0);

        let stats = BM25Stats {
            avg_doc_length: 5.0,
            idf,
            num_docs: 100,
        };

        let query_indices = vec![1];
        let query_weights = vec![1.0];

        // Document 1: term appears once
        let doc1_indices = vec![1];
        let doc1_values = vec![1.0];

        let score1 = bm25_score(
            &query_indices,
            &query_weights,
            &doc1_indices,
            &doc1_values,
            &stats,
            &BM25Config::default(),
        );

        // Document 2: term appears 5 times
        let doc2_indices = vec![1];
        let doc2_values = vec![5.0];

        let score2 = bm25_score(
            &query_indices,
            &query_weights,
            &doc2_indices,
            &doc2_values,
            &stats,
            &BM25Config::default(),
        );

        assert!(score2 > score1);
    }

    #[test]
    fn test_bm25_score_simple() {
        let query_indices = vec![1, 2];
        let doc_indices = vec![1, 2, 3];
        let doc_values = vec![2.0, 1.0, 1.0];

        let score = bm25_score_simple(
            &query_indices,
            &doc_indices,
            &doc_values,
            &BM25Config::default(),
        );

        assert!(score > 0.0);
    }

    #[test]
    fn test_bm25_k1_parameter() {
        // Test that k1 affects term frequency saturation
        let mut idf = HashMap::new();
        idf.insert(1, 1.0);

        let stats = BM25Stats {
            avg_doc_length: 10.0,
            idf,
            num_docs: 100,
        };

        let query_indices = vec![1];
        let query_weights = vec![1.0];
        let doc_indices = vec![1];
        let doc_values = vec![10.0]; // High frequency

        // Low k1 = more saturation
        let config_low = BM25Config { k1: 0.5, b: 0.75 };
        let score_low = bm25_score(
            &query_indices,
            &query_weights,
            &doc_indices,
            &doc_values,
            &stats,
            &config_low,
        );

        // High k1 = less saturation, more weight on frequency
        let config_high = BM25Config { k1: 3.0, b: 0.75 };
        let score_high = bm25_score(
            &query_indices,
            &query_weights,
            &doc_indices,
            &doc_values,
            &stats,
            &config_high,
        );

        assert!(score_high > score_low);
    }

    // ============================================================================
    // BM25F Tests (Field Boosting)
    // ============================================================================

    #[test]
    fn test_parse_field_weight_with_boost() {
        let (field, weight) = parse_field_weight("title^3");
        assert_eq!(field, "title");
        assert_eq!(weight, 3.0);
    }

    #[test]
    fn test_parse_field_weight_with_float_boost() {
        let (field, weight) = parse_field_weight("abstract^2.5");
        assert_eq!(field, "abstract");
        assert_eq!(weight, 2.5);
    }

    #[test]
    fn test_parse_field_weight_without_boost() {
        let (field, weight) = parse_field_weight("content");
        assert_eq!(field, "content");
        assert_eq!(weight, 1.0);
    }

    #[test]
    fn test_parse_field_weight_invalid_boost() {
        let (field, weight) = parse_field_weight("title^invalid");
        assert_eq!(field, "title");
        assert_eq!(weight, 1.0); // Should default to 1.0 on parse error
    }

    #[test]
    fn test_parse_field_weights_multiple() {
        let specs = vec!["title^3", "abstract^2", "content"];
        let weights = parse_field_weights(&specs);

        assert_eq!(weights.len(), 3);
        assert_eq!(weights.get("title"), Some(&3.0));
        assert_eq!(weights.get("abstract"), Some(&2.0));
        assert_eq!(weights.get("content"), Some(&1.0));
    }

    #[test]
    fn test_parse_field_weights_empty() {
        let specs: Vec<&str> = vec![];
        let weights = parse_field_weights(&specs);
        assert_eq!(weights.len(), 0);
    }

    #[test]
    fn test_bm25f_single_field_matches_regular_bm25() {
        // BM25F with single field should match regular BM25
        let mut idf = HashMap::new();
        idf.insert(1, 2.0);
        idf.insert(2, 1.5);

        let stats = BM25Stats {
            avg_doc_length: 10.0,
            idf,
            num_docs: 100,
        };

        let query_indices = vec![1, 2];
        let query_weights = vec![1.0, 1.0];
        let doc_indices = vec![1, 2, 3];
        let doc_values = vec![2.0, 1.0, 1.0];

        // Regular BM25 score
        let regular_score = bm25_score(
            &query_indices,
            &query_weights,
            &doc_indices,
            &doc_values,
            &stats,
            &BM25Config::default(),
        );

        // BM25F score with single field (weight=1.0)
        let mut doc_fields = HashMap::new();
        doc_fields.insert(
            "content".to_string(),
            (doc_indices.clone(), doc_values.clone()),
        );

        let mut field_weights = HashMap::new();
        field_weights.insert("content".to_string(), 1.0);

        let bm25f_score_result = bm25f_score(
            &query_indices,
            &query_weights,
            &doc_fields,
            &field_weights,
            &stats,
            &BM25Config::default(),
        );

        // Should be very close (allowing for floating point precision)
        assert!((regular_score - bm25f_score_result).abs() < 0.01);
    }

    #[test]
    fn test_bm25f_multiple_fields() {
        // Multi-field document
        let mut idf = HashMap::new();
        idf.insert(1, 2.0); // term "rust"
        idf.insert(2, 1.5); // term "database"
        idf.insert(3, 1.0); // term "vector"

        let stats = BM25Stats {
            avg_doc_length: 10.0,
            idf,
            num_docs: 100,
        };

        let query_indices = vec![1, 2]; // searching for "rust database"
        let query_weights = vec![1.0, 1.0];

        // Document has three fields
        let mut doc_fields = HashMap::new();

        // Title: "rust database" (both terms appear)
        doc_fields.insert("title".to_string(), (vec![1, 2], vec![1.0, 1.0]));

        // Abstract: "rust" (only first term)
        doc_fields.insert("abstract".to_string(), (vec![1, 3], vec![1.0, 1.0]));

        // Content: "database vector" (only second term)
        doc_fields.insert("content".to_string(), (vec![2, 3], vec![1.0, 1.0]));

        // All fields equal weight
        let mut field_weights = HashMap::new();
        field_weights.insert("title".to_string(), 1.0);
        field_weights.insert("abstract".to_string(), 1.0);
        field_weights.insert("content".to_string(), 1.0);

        let score = bm25f_score(
            &query_indices,
            &query_weights,
            &doc_fields,
            &field_weights,
            &stats,
            &BM25Config::default(),
        );

        assert!(score > 0.0);
    }

    #[test]
    fn test_bm25f_title_boost() {
        // Test that title boost increases score
        let mut idf = HashMap::new();
        idf.insert(1, 2.0);

        let stats = BM25Stats {
            avg_doc_length: 10.0,
            idf,
            num_docs: 100,
        };

        let query_indices = vec![1];
        let query_weights = vec![1.0];

        let mut doc_fields = HashMap::new();
        doc_fields.insert("title".to_string(), (vec![1], vec![1.0]));
        doc_fields.insert("content".to_string(), (vec![1], vec![1.0]));

        // No boost
        let mut field_weights_no_boost = HashMap::new();
        field_weights_no_boost.insert("title".to_string(), 1.0);
        field_weights_no_boost.insert("content".to_string(), 1.0);

        let score_no_boost = bm25f_score(
            &query_indices,
            &query_weights,
            &doc_fields,
            &field_weights_no_boost,
            &stats,
            &BM25Config::default(),
        );

        // Title boosted 3x
        let mut field_weights_with_boost = HashMap::new();
        field_weights_with_boost.insert("title".to_string(), 3.0);
        field_weights_with_boost.insert("content".to_string(), 1.0);

        let score_with_boost = bm25f_score(
            &query_indices,
            &query_weights,
            &doc_fields,
            &field_weights_with_boost,
            &stats,
            &BM25Config::default(),
        );

        // Boosted score should be higher
        assert!(score_with_boost > score_no_boost);
    }

    #[test]
    fn test_bm25f_missing_field_weight() {
        // Fields without explicit weights should default to 1.0
        let mut idf = HashMap::new();
        idf.insert(1, 2.0);

        let stats = BM25Stats {
            avg_doc_length: 10.0,
            idf,
            num_docs: 100,
        };

        let query_indices = vec![1];
        let query_weights = vec![1.0];

        let mut doc_fields = HashMap::new();
        doc_fields.insert("title".to_string(), (vec![1], vec![1.0]));
        doc_fields.insert("content".to_string(), (vec![1], vec![1.0]));

        // Only specify weight for title, not content
        let mut field_weights = HashMap::new();
        field_weights.insert("title".to_string(), 2.0);

        let score = bm25f_score(
            &query_indices,
            &query_weights,
            &doc_fields,
            &field_weights,
            &stats,
            &BM25Config::default(),
        );

        // Should still work, content defaults to 1.0
        assert!(score > 0.0);
    }

    #[test]
    fn test_bm25f_no_matching_terms() {
        let mut idf = HashMap::new();
        idf.insert(1, 2.0);
        idf.insert(2, 1.5);

        let stats = BM25Stats {
            avg_doc_length: 10.0,
            idf,
            num_docs: 100,
        };

        let query_indices = vec![1, 2];
        let query_weights = vec![1.0, 1.0];

        let mut doc_fields = HashMap::new();
        // Document has different terms
        doc_fields.insert("title".to_string(), (vec![3, 4], vec![1.0, 1.0]));

        let mut field_weights = HashMap::new();
        field_weights.insert("title".to_string(), 1.0);

        let score = bm25f_score(
            &query_indices,
            &query_weights,
            &doc_fields,
            &field_weights,
            &stats,
            &BM25Config::default(),
        );

        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_bm25f_empty_fields() {
        let mut idf = HashMap::new();
        idf.insert(1, 2.0);

        let stats = BM25Stats {
            avg_doc_length: 10.0,
            idf,
            num_docs: 100,
        };

        let query_indices = vec![1];
        let query_weights = vec![1.0];

        let doc_fields = HashMap::new(); // No fields
        let field_weights = HashMap::new();

        let score = bm25f_score(
            &query_indices,
            &query_weights,
            &doc_fields,
            &field_weights,
            &stats,
            &BM25Config::default(),
        );

        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_bm25f_realistic_document() {
        // Realistic example: searching for "rust vector database"
        let mut idf = HashMap::new();
        idf.insert(100, 2.5); // "rust" - moderately rare
        idf.insert(200, 2.0); // "vector" - less rare
        idf.insert(300, 1.8); // "database" - common

        let stats = BM25Stats {
            avg_doc_length: 50.0,
            idf,
            num_docs: 1000,
        };

        let query_indices = vec![100, 200, 300];
        let query_weights = vec![1.0, 1.0, 1.0];

        // Document: Title="Rust Vector Store", Abstract="A fast vector database", Content=long text
        let mut doc_fields = HashMap::new();
        doc_fields.insert("title".to_string(), (vec![100, 200], vec![1.0, 1.0])); // "rust vector"
        doc_fields.insert("abstract".to_string(), (vec![200, 300], vec![1.0, 1.0])); // "vector database"
        doc_fields.insert(
            "content".to_string(),
            (vec![100, 200, 300], vec![2.0, 3.0, 1.0]),
        ); // all terms

        // Parse field weights using our helper
        let field_weights = parse_field_weights(&["title^3", "abstract^2", "content"]);

        let score = bm25f_score(
            &query_indices,
            &query_weights,
            &doc_fields,
            &field_weights,
            &stats,
            &BM25Config::default(),
        );

        // Should have good score since all terms match and title is boosted
        assert!(score > 5.0); // Reasonable threshold for this setup
    }
}
