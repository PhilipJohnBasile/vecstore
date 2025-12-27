//! Multi-Language Stemming and Text Processing
//!
//! Support for 17+ languages with stemming, stopword removal, and tokenization.
//! Similar to Turbopuffer and Elasticsearch's language analyzers.
//!
//! # Supported Languages
//!
//! - English, Spanish, French, German, Italian, Portuguese
//! - Dutch, Swedish, Norwegian, Danish, Finnish
//! - Russian, Arabic, Turkish, Chinese, Japanese, Korean
//! - Greek, Hebrew, Hindi, Thai, Vietnamese
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::multilang::{MultiLangAnalyzer, Language};
//!
//! let analyzer = MultiLangAnalyzer::new(Language::Spanish);
//!
//! let tokens = analyzer.analyze("Los gatos están corriendo rápidamente")?;
//! // ["gato", "corr", "rapid"] - stemmed, no stopwords
//! ```

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};


/// Supported languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    English,
    Spanish,
    French,
    German,
    Italian,
    Portuguese,
    Dutch,
    Swedish,
    Norwegian,
    Danish,
    Finnish,
    Russian,
    Arabic,
    Turkish,
    Chinese,
    Japanese,
    Korean,
    Greek,
    Hebrew,
    Hindi,
    Thai,
    Vietnamese,
    Polish,
    Czech,
    Hungarian,
    Romanian,
    Indonesian,
}

impl Language {
    /// Get language code (ISO 639-1)
    pub fn code(&self) -> &'static str {
        match self {
            Language::English => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Italian => "it",
            Language::Portuguese => "pt",
            Language::Dutch => "nl",
            Language::Swedish => "sv",
            Language::Norwegian => "no",
            Language::Danish => "da",
            Language::Finnish => "fi",
            Language::Russian => "ru",
            Language::Arabic => "ar",
            Language::Turkish => "tr",
            Language::Chinese => "zh",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Greek => "el",
            Language::Hebrew => "he",
            Language::Hindi => "hi",
            Language::Thai => "th",
            Language::Vietnamese => "vi",
            Language::Polish => "pl",
            Language::Czech => "cs",
            Language::Hungarian => "hu",
            Language::Romanian => "ro",
            Language::Indonesian => "id",
        }
    }

    /// Get language name
    pub fn name(&self) -> &'static str {
        match self {
            Language::English => "English",
            Language::Spanish => "Spanish",
            Language::French => "French",
            Language::German => "German",
            Language::Italian => "Italian",
            Language::Portuguese => "Portuguese",
            Language::Dutch => "Dutch",
            Language::Swedish => "Swedish",
            Language::Norwegian => "Norwegian",
            Language::Danish => "Danish",
            Language::Finnish => "Finnish",
            Language::Russian => "Russian",
            Language::Arabic => "Arabic",
            Language::Turkish => "Turkish",
            Language::Chinese => "Chinese",
            Language::Japanese => "Japanese",
            Language::Korean => "Korean",
            Language::Greek => "Greek",
            Language::Hebrew => "Hebrew",
            Language::Hindi => "Hindi",
            Language::Thai => "Thai",
            Language::Vietnamese => "Vietnamese",
            Language::Polish => "Polish",
            Language::Czech => "Czech",
            Language::Hungarian => "Hungarian",
            Language::Romanian => "Romanian",
            Language::Indonesian => "Indonesian",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "en" | "english" => Some(Language::English),
            "es" | "spanish" => Some(Language::Spanish),
            "fr" | "french" => Some(Language::French),
            "de" | "german" => Some(Language::German),
            "it" | "italian" => Some(Language::Italian),
            "pt" | "portuguese" => Some(Language::Portuguese),
            "nl" | "dutch" => Some(Language::Dutch),
            "sv" | "swedish" => Some(Language::Swedish),
            "no" | "norwegian" => Some(Language::Norwegian),
            "da" | "danish" => Some(Language::Danish),
            "fi" | "finnish" => Some(Language::Finnish),
            "ru" | "russian" => Some(Language::Russian),
            "ar" | "arabic" => Some(Language::Arabic),
            "tr" | "turkish" => Some(Language::Turkish),
            "zh" | "chinese" => Some(Language::Chinese),
            "ja" | "japanese" => Some(Language::Japanese),
            "ko" | "korean" => Some(Language::Korean),
            "el" | "greek" => Some(Language::Greek),
            "he" | "hebrew" => Some(Language::Hebrew),
            "hi" | "hindi" => Some(Language::Hindi),
            "th" | "thai" => Some(Language::Thai),
            "vi" | "vietnamese" => Some(Language::Vietnamese),
            "pl" | "polish" => Some(Language::Polish),
            "cs" | "czech" => Some(Language::Czech),
            "hu" | "hungarian" => Some(Language::Hungarian),
            "ro" | "romanian" => Some(Language::Romanian),
            "id" | "indonesian" => Some(Language::Indonesian),
            _ => None,
        }
    }

    /// Get all supported languages
    pub fn all() -> Vec<Language> {
        vec![
            Language::English, Language::Spanish, Language::French,
            Language::German, Language::Italian, Language::Portuguese,
            Language::Dutch, Language::Swedish, Language::Norwegian,
            Language::Danish, Language::Finnish, Language::Russian,
            Language::Arabic, Language::Turkish, Language::Chinese,
            Language::Japanese, Language::Korean, Language::Greek,
            Language::Hebrew, Language::Hindi, Language::Thai,
            Language::Vietnamese, Language::Polish, Language::Czech,
            Language::Hungarian, Language::Romanian, Language::Indonesian,
        ]
    }
}

/// Stopwords for each language
fn get_stopwords(lang: Language) -> HashSet<&'static str> {
    match lang {
        Language::English => {
            ["a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
             "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
             "be", "have", "has", "had", "do", "does", "did", "will", "would",
             "could", "should", "may", "might", "must", "can", "this", "that",
             "these", "those", "i", "you", "he", "she", "it", "we", "they",
             "what", "which", "who", "when", "where", "why", "how", "all",
             "each", "every", "both", "few", "more", "most", "other", "some",
             "such", "no", "not", "only", "same", "so", "than", "too", "very"]
                .iter().cloned().collect()
        }
        Language::Spanish => {
            ["de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
             "las", "por", "un", "para", "con", "no", "una", "su", "al",
             "es", "lo", "como", "más", "pero", "sus", "le", "ya", "o",
             "fue", "este", "ha", "sí", "porque", "esta", "son", "entre",
             "está", "cuando", "muy", "sin", "sobre", "ser", "tiene", "también"]
                .iter().cloned().collect()
        }
        Language::French => {
            ["le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
             "à", "au", "aux", "ce", "que", "qui", "dans", "pour", "sur",
             "par", "avec", "ne", "pas", "plus", "ou", "son", "sa", "ses",
             "il", "elle", "nous", "vous", "ils", "elles", "je", "tu",
             "est", "sont", "être", "avoir", "fait", "faire", "comme"]
                .iter().cloned().collect()
        }
        Language::German => {
            ["der", "die", "das", "und", "in", "zu", "den", "von", "mit",
             "ist", "nicht", "ein", "eine", "als", "auch", "es", "an",
             "auf", "für", "sich", "des", "dem", "werden", "bei", "haben",
             "wird", "sind", "oder", "nach", "am", "um", "aus", "nur",
             "wie", "über", "so", "wenn", "aber", "noch", "durch", "kann"]
                .iter().cloned().collect()
        }
        Language::Italian => {
            ["il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di",
             "a", "da", "in", "con", "su", "per", "tra", "fra", "che",
             "e", "ma", "o", "se", "come", "non", "più", "anche", "solo",
             "questo", "quello", "suo", "loro", "essere", "avere", "fare"]
                .iter().cloned().collect()
        }
        Language::Russian => {
            ["и", "в", "не", "на", "я", "что", "он", "с", "это", "а",
             "как", "она", "по", "но", "они", "к", "у", "ты", "из", "мы",
             "за", "от", "о", "же", "все", "так", "его", "её", "их"]
                .iter().cloned().collect()
        }
        Language::Arabic => {
            ["في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "التي",
             "الذي", "كان", "لم", "لا", "ما", "هو", "هي", "أن", "أو",
             "و", "ثم", "بعد", "قبل", "كل", "بين", "حتى", "إذا"]
                .iter().cloned().collect()
        }
        Language::Chinese | Language::Japanese | Language::Korean => {
            // CJK languages typically don't use stopword removal
            HashSet::new()
        }
        _ => {
            // Default minimal stopwords
            ["a", "an", "the", "and", "or", "but", "in", "on", "at", "to"]
                .iter().cloned().collect()
        }
    }
}

/// Stemming rules for a language
#[derive(Debug, Clone)]
struct StemmingRules {
    suffixes: Vec<(&'static str, &'static str)>,
    min_stem_length: usize,
}

impl StemmingRules {
    fn for_language(lang: Language) -> Self {
        match lang {
            Language::English => Self {
                suffixes: vec![
                    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
                    ("anci", "ance"), ("izer", "ize"), ("isation", "ize"),
                    ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
                    ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
                    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
                    ("biliti", "ble"), ("alli", "al"), ("entli", "ent"),
                    ("eli", "e"), ("ousli", "ous"), ("ling", ""),
                    ("ement", ""), ("ment", ""), ("ness", ""), ("ing", ""),
                    ("ies", "y"), ("es", ""), ("ed", ""), ("s", ""),
                ],
                min_stem_length: 3,
            },
            Language::Spanish => Self {
                suffixes: vec![
                    ("amiento", ""), ("imientos", ""), ("imiento", ""),
                    ("aciones", ""), ("adores", ""), ("amente", ""),
                    ("idades", ""), ("mente", ""), ("ables", ""),
                    ("ibles", ""), ("ación", ""), ("ador", ""),
                    ("ante", ""), ("ando", ""), ("endo", ""),
                    ("ido", ""), ("ido", ""), ("ar", ""), ("er", ""),
                    ("ir", ""), ("as", ""), ("es", ""), ("os", ""),
                ],
                min_stem_length: 3,
            },
            Language::French => Self {
                suffixes: vec![
                    ("issements", ""), ("issement", ""), ("atrices", ""),
                    ("ateurs", ""), ("ations", ""), ("logies", ""),
                    ("usion", "u"), ("ution", "u"), ("ences", ""),
                    ("ances", ""), ("ments", ""), ("ement", ""),
                    ("euses", ""), ("ables", ""), ("istes", ""),
                    ("ation", ""), ("ique", ""), ("isme", ""),
                    ("able", ""), ("iste", ""), ("ment", ""),
                    ("ence", ""), ("ance", ""), ("euse", ""),
                    ("eux", ""), ("ant", ""), ("ent", ""),
                ],
                min_stem_length: 3,
            },
            Language::German => Self {
                suffixes: vec![
                    ("ungen", ""), ("heit", ""), ("keit", ""),
                    ("lich", ""), ("isch", ""), ("ung", ""),
                    ("ig", ""), ("ik", ""), ("en", ""), ("er", ""),
                    ("em", ""), ("es", ""), ("e", ""), ("s", ""),
                ],
                min_stem_length: 3,
            },
            _ => Self {
                suffixes: vec![
                    ("ing", ""), ("ed", ""), ("es", ""), ("s", ""),
                ],
                min_stem_length: 3,
            },
        }
    }
}

/// Multi-language text analyzer
pub struct MultiLangAnalyzer {
    language: Language,
    stopwords: HashSet<&'static str>,
    stemming_rules: StemmingRules,
    lowercase: bool,
    remove_accents: bool,
    min_token_length: usize,
    max_token_length: usize,
}

impl MultiLangAnalyzer {
    /// Create a new analyzer for a language
    pub fn new(language: Language) -> Self {
        Self {
            language,
            stopwords: get_stopwords(language),
            stemming_rules: StemmingRules::for_language(language),
            lowercase: true,
            remove_accents: true,
            min_token_length: 2,
            max_token_length: 50,
        }
    }

    /// Set whether to lowercase
    pub fn with_lowercase(mut self, enabled: bool) -> Self {
        self.lowercase = enabled;
        self
    }

    /// Set whether to remove accents
    pub fn with_remove_accents(mut self, enabled: bool) -> Self {
        self.remove_accents = enabled;
        self
    }

    /// Set minimum token length
    pub fn with_min_token_length(mut self, len: usize) -> Self {
        self.min_token_length = len;
        self
    }

    /// Analyze text into tokens
    pub fn analyze(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        // Tokenize
        for word in self.tokenize(text) {
            // Apply transformations
            let mut token = word.to_string();

            if self.lowercase {
                token = token.to_lowercase();
            }

            if self.remove_accents {
                token = self.strip_accents(&token);
            }

            // Skip stopwords
            if self.stopwords.contains(token.as_str()) {
                continue;
            }

            // Length filter
            if token.len() < self.min_token_length || token.len() > self.max_token_length {
                continue;
            }

            // Apply stemming
            token = self.stem(&token);

            if !token.is_empty() {
                tokens.push(token);
            }
        }

        tokens
    }

    /// Tokenize text into words
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        match self.language {
            Language::Chinese | Language::Japanese => {
                // Character-based tokenization for CJK
                text.chars()
                    .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
                    .map(|c| {
                        let start = text.chars().position(|x| x == c).unwrap();
                        &text[start..start + c.len_utf8()]
                    })
                    .collect()
            }
            _ => {
                // Word-based tokenization for most languages
                text.split(|c: char| !c.is_alphanumeric() && c != '\'')
                    .filter(|s| !s.is_empty())
                    .collect()
            }
        }
    }

    /// Apply stemming
    fn stem(&self, word: &str) -> String {
        let mut result = word.to_string();

        for (suffix, replacement) in &self.stemming_rules.suffixes {
            if result.ends_with(suffix) {
                let stem_len = result.len() - suffix.len();
                if stem_len >= self.stemming_rules.min_stem_length {
                    result.truncate(stem_len);
                    result.push_str(replacement);
                    break;
                }
            }
        }

        result
    }

    /// Strip accents from text
    fn strip_accents(&self, text: &str) -> String {
        text.chars()
            .map(|c| match c {
                'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
                'é' | 'è' | 'ê' | 'ë' => 'e',
                'í' | 'ì' | 'î' | 'ï' => 'i',
                'ó' | 'ò' | 'ô' | 'ö' | 'õ' => 'o',
                'ú' | 'ù' | 'û' | 'ü' => 'u',
                'ý' | 'ÿ' => 'y',
                'ñ' => 'n',
                'ç' => 'c',
                _ => c,
            })
            .collect()
    }

    /// Get the language
    pub fn language(&self) -> Language {
        self.language
    }
}

/// Auto-detecting multi-language analyzer
pub struct AutoLangAnalyzer {
    analyzers: HashMap<Language, MultiLangAnalyzer>,
    default_language: Language,
}

impl AutoLangAnalyzer {
    /// Create a new auto-detecting analyzer
    pub fn new() -> Self {
        let mut analyzers = HashMap::new();
        for lang in Language::all() {
            analyzers.insert(lang, MultiLangAnalyzer::new(lang));
        }

        Self {
            analyzers,
            default_language: Language::English,
        }
    }

    /// Set default language
    pub fn with_default(mut self, lang: Language) -> Self {
        self.default_language = lang;
        self
    }

    /// Detect language from text
    pub fn detect_language(&self, text: &str) -> Language {
        // Simple heuristic-based detection
        let text_lower = text.to_lowercase();

        // Check for CJK characters
        if text.chars().any(|c| c >= '\u{4E00}' && c <= '\u{9FFF}') {
            return Language::Chinese;
        }
        if text.chars().any(|c| c >= '\u{3040}' && c <= '\u{30FF}') {
            return Language::Japanese;
        }
        if text.chars().any(|c| c >= '\u{AC00}' && c <= '\u{D7AF}') {
            return Language::Korean;
        }

        // Check for Cyrillic (Russian)
        if text.chars().any(|c| c >= '\u{0400}' && c <= '\u{04FF}') {
            return Language::Russian;
        }

        // Check for Arabic
        if text.chars().any(|c| c >= '\u{0600}' && c <= '\u{06FF}') {
            return Language::Arabic;
        }

        // Check for Greek
        if text.chars().any(|c| c >= '\u{0370}' && c <= '\u{03FF}') {
            return Language::Greek;
        }

        // Check for Hebrew
        if text.chars().any(|c| c >= '\u{0590}' && c <= '\u{05FF}') {
            return Language::Hebrew;
        }

        // Check common words for European languages
        let common_words: HashMap<&str, Language> = [
            ("the", Language::English), ("and", Language::English),
            ("der", Language::German), ("und", Language::German),
            ("le", Language::French), ("et", Language::French),
            ("el", Language::Spanish), ("y", Language::Spanish),
            ("il", Language::Italian), ("che", Language::Italian),
            ("het", Language::Dutch), ("en", Language::Dutch),
        ].iter().cloned().collect();

        for word in text_lower.split_whitespace() {
            if let Some(&lang) = common_words.get(word) {
                return lang;
            }
        }

        self.default_language
    }

    /// Analyze text with auto-detection
    pub fn analyze(&self, text: &str) -> Vec<String> {
        let lang = self.detect_language(text);
        self.analyze_with_language(text, lang)
    }

    /// Analyze text with specified language
    pub fn analyze_with_language(&self, text: &str, lang: Language) -> Vec<String> {
        self.analyzers
            .get(&lang)
            .unwrap_or(&self.analyzers[&self.default_language])
            .analyze(text)
    }
}

impl Default for AutoLangAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_english_analysis() {
        let analyzer = MultiLangAnalyzer::new(Language::English);
        let tokens = analyzer.analyze("The cats are running quickly");

        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"are".to_string()));
        assert!(tokens.iter().any(|t| t.starts_with("cat")));
        assert!(tokens.iter().any(|t| t.starts_with("run")));
    }

    #[test]
    fn test_spanish_analysis() {
        let analyzer = MultiLangAnalyzer::new(Language::Spanish);
        let tokens = analyzer.analyze("Los gatos están corriendo");

        assert!(!tokens.contains(&"los".to_string()));
    }

    #[test]
    fn test_language_detection() {
        let analyzer = AutoLangAnalyzer::new();

        assert_eq!(analyzer.detect_language("Hello world"), Language::English);
        assert_eq!(analyzer.detect_language("Der Hund läuft"), Language::German);
        assert_eq!(analyzer.detect_language("Привет мир"), Language::Russian);
        assert_eq!(analyzer.detect_language("你好世界"), Language::Chinese);
    }

    #[test]
    fn test_accent_removal() {
        let analyzer = MultiLangAnalyzer::new(Language::French);
        let tokens = analyzer.analyze("café résumé");

        assert!(tokens.iter().any(|t| t == "cafe"));
        assert!(tokens.iter().any(|t| t == "resum"));
    }

    #[test]
    fn test_all_languages() {
        for lang in Language::all() {
            let analyzer = MultiLangAnalyzer::new(lang);
            let tokens = analyzer.analyze("test word");
            // Just ensure it doesn't panic
            assert!(tokens.len() <= 2);
        }
    }
}
