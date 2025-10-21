//! Tokenizer Demo
//!
//! Demonstrates VecStore's pluggable tokenizer system for BM25/hybrid search:
//! - SimpleTokenizer: Basic whitespace + punctuation
//! - LanguageTokenizer: Stopword removal for better precision
//! - WhitespaceTokenizer: Preserves punctuation (emails, URLs, code)
//! - NGramTokenizer: Fuzzy matching with character/word n-grams
//!
//! Run with: cargo run --example tokenizer_demo

use vecstore::tokenizer::{
    LanguageTokenizer, NGramTokenizer, SimpleTokenizer, Tokenizer, WhitespaceTokenizer,
};

fn main() {
    println!("=== VecStore Tokenizer Demo ===\n");

    let sample_text = "The quick brown fox jumps over the lazy dog!";
    let technical_text = "user@example.com signed in from 192.168.1.1";
    let code_snippet = "function getUserData() { return data; }";

    // =================================================================================
    // Demo 1: SimpleTokenizer (Default)
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 1: SimpleTokenizer (Default)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let tokenizer = SimpleTokenizer::new();
    println!("   Text: \"{}\"", sample_text);
    let tokens = tokenizer.tokenize(sample_text);
    println!("   Tokens: {:?}", tokens);
    println!("   Count: {}\n", tokens.len());

    println!("   Features:");
    println!("     ✓ Lowercase conversion");
    println!("     ✓ Punctuation removal");
    println!("     ✓ Whitespace splitting");
    println!("     ✓ Fast and simple\n");

    // =================================================================================
    // Demo 2: LanguageTokenizer with Stopword Removal
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 2: LanguageTokenizer (English Stopwords)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let tokenizer = LanguageTokenizer::english();
    println!("   Text: \"{}\"", sample_text);
    let tokens = tokenizer.tokenize(sample_text);
    println!("   Tokens: {:?}", tokens);
    println!("   Count: {}\n", tokens.len());

    println!("   Removed stopwords: \"the\", \"over\"");
    println!("   Kept content words: {:?}\n", tokens);

    println!("   Features:");
    println!("     ✓ 60+ English stopwords removed");
    println!("     ✓ Better precision for search");
    println!("     ✓ Reduces index size");
    println!("     ✓ Improves BM25 scoring\n");

    println!("   Use Cases:");
    println!("     • Document search (remove \"the\", \"a\", \"is\")");
    println!("     • Question answering");
    println!("     • Content-focused retrieval\n");

    // =================================================================================
    // Demo 3: WhitespaceTokenizer (Preserves Punctuation)
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 3: WhitespaceTokenizer (Preserves Punctuation)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let tokenizer = WhitespaceTokenizer::new();
    println!("   Text: \"{}\"", technical_text);
    let tokens = tokenizer.tokenize(technical_text);
    println!("   Tokens: {:?}", tokens);
    println!("   Count: {}\n", tokens.len());

    println!("   Preserved:");
    println!("     • Email: user@example.com");
    println!("     • IP: 192.168.1.1\n");

    println!("   Features:");
    println!("     ✓ Preserves special characters");
    println!("     ✓ Exact match for technical data");
    println!("     ✓ Ideal for structured text\n");

    println!("   Use Cases:");
    println!("     • Email search");
    println!("     • URL/domain filtering");
    println!("     • Code search");
    println!("     • Log analysis\n");

    // Code example
    println!("   Code Example:");
    println!("   Text: \"{}\"", code_snippet);
    let tokens = tokenizer.tokenize(code_snippet);
    println!("   Tokens: {:?}", tokens);
    println!("   Preserved: parentheses, braces, semicolons\n");

    // =================================================================================
    // Demo 4: Character N-grams (Fuzzy Matching)
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 4: Character N-grams (Trigrams)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let tokenizer = NGramTokenizer::char_ngrams(3);
    let word = "algorithm";
    println!("   Text: \"{}\"", word);
    let tokens = tokenizer.tokenize(word);
    println!("   Trigrams: {:?}", tokens);
    println!("   Count: {}\n", tokens.len());

    println!("   Features:");
    println!("     ✓ Substring matching");
    println!("     ✓ Typo tolerance");
    println!("     ✓ Fuzzy search\n");

    println!("   Use Cases:");
    println!("     • Spell correction");
    println!("     • Autocomplete");
    println!("     • Name matching with variations");
    println!("     • Search \"algoritm\" → finds \"algorithm\"\n");

    // =================================================================================
    // Demo 5: Word N-grams (Phrase Matching)
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 5: Word N-grams (Bigrams)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let tokenizer = NGramTokenizer::word_ngrams(2);
    let phrase = "machine learning model";
    println!("   Text: \"{}\"", phrase);
    let tokens = tokenizer.tokenize(phrase);
    println!("   Bigrams: {:?}", tokens);
    println!("   Count: {}\n", tokens.len());

    println!("   Features:");
    println!("     ✓ Phrase-aware indexing");
    println!("     ✓ Captures word relationships");
    println!("     ✓ Better context matching\n");

    println!("   Use Cases:");
    println!("     • Named entity search (\"New York\")");
    println!("     • Technical terms (\"deep learning\")");
    println!("     • Collocation detection\n");

    // =================================================================================
    // Demo 6: Case Sensitivity
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 6: Case Sensitivity");
    println!("═══════════════════════════════════════════════════════════════\n");

    let case_sensitive = SimpleTokenizer::with_case_preserved();
    let text = "NASA announced new Mars mission";
    println!("   Text: \"{}\"", text);
    let tokens = case_sensitive.tokenize(text);
    println!("   Tokens (case preserved): {:?}", tokens);
    println!();

    let case_insensitive = SimpleTokenizer::new();
    let tokens = case_insensitive.tokenize(text);
    println!("   Tokens (lowercase): {:?}", tokens);
    println!();

    println!("   Use Cases:");
    println!("     • Acronym search (NASA, API)");
    println!("     • Proper noun matching");
    println!("     • Code identifiers (CamelCase)\n");

    // =================================================================================
    // Performance Comparison
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Performance Characteristics");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("   Tokenizer              | Speed    | Index Size | Recall | Precision");
    println!("   ---------------------- | -------- | ---------- | ------ | ---------");
    println!("   SimpleTokenizer        | Fastest  | Medium     | Medium | Medium   ");
    println!("   LanguageTokenizer      | Fast     | Smallest   | Medium | High     ");
    println!("   WhitespaceTokenizer    | Fastest  | Medium     | High   | High     ");
    println!("   CharNGramTokenizer     | Slow     | Largest    | Highest| Low      ");
    println!("   WordNGramTokenizer     | Medium   | Large      | High   | Medium   ");
    println!();

    // =================================================================================
    // Integration with TextIndex
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Integration Example");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("   ```rust");
    println!("   use vecstore::{{VecStore, tokenizer::LanguageTokenizer}};");
    println!();
    println!("   // Create store with custom tokenizer for BM25");
    println!("   let mut store = VecStore::open(\"my_db\")?;");
    println!();
    println!("   // Use LanguageTokenizer for better search quality");
    println!("   let tokenizer = Box::new(LanguageTokenizer::english());");
    println!("   let mut text_index = TextIndex::with_tokenizer(tokenizer);");
    println!();
    println!("   // Index documents");
    println!("   text_index.index_document(\"doc1\", \"The quick brown fox\");");
    println!();
    println!("   // Query (stopwords automatically removed)");
    println!("   let scores = text_index.bm25_scores(\"the quick fox\");");
    println!("   // Matches on \"quick\" and \"fox\" only");
    println!("   ```\n");

    // =================================================================================
    // Best Practices
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Best Practices");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("   Document Search (general):");
    println!("     → SimpleTokenizer or LanguageTokenizer");
    println!();

    println!("   RAG Applications:");
    println!("     → LanguageTokenizer (removes noise, improves relevance)");
    println!();

    println!("   Code Search:");
    println!("     → WhitespaceTokenizer (preserves syntax)");
    println!();

    println!("   Email/URL Search:");
    println!("     → WhitespaceTokenizer (exact matching)");
    println!();

    println!("   Fuzzy Search / Typo Tolerance:");
    println!("     → CharNGramTokenizer (3-grams)");
    println!();

    println!("   Phrase-Aware Search:");
    println!("     → WordNGramTokenizer (2-3 grams)");
    println!();

    println!("   Multi-Language:");
    println!("     → Implement custom Tokenizer trait with language-specific rules");
    println!();

    // =================================================================================
    // Summary
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Summary");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("   ✅ 4 built-in tokenizers");
    println!("   ✅ Trait-based (implement your own!)");
    println!("   ✅ Zero-copy where possible");
    println!("   ✅ Production-ready performance");
    println!("   ✅ Matches Qdrant/Weaviate tokenization capabilities");
    println!();

    println!("   Competitive Position:");
    println!("     • Qdrant: Custom tokenizers via gRPC");
    println!("     • Weaviate: Fixed tokenization (word-based)");
    println!("     • Pinecone: No BM25 (vector-only)");
    println!("     • VecStore: ✅ Pluggable tokenizers in-process");
    println!();

    println!("✅ Tokenizer Demo Complete!\n");
}
