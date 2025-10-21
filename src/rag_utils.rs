//! # RAG Utilities Module
//!
//! Helper utilities and patterns for Retrieval-Augmented Generation (RAG) applications.
//!
//! This module provides common RAG patterns as reusable utilities:
//! - Query expansion
//! - HyDE (Hypothetical Document Embeddings)
//! - Multi-query retrieval
//! - Result fusion
//!
//! ## Design Philosophy
//!
//! These are **helper utilities**, not core features. They demonstrate common RAG patterns
//! that users can adapt for their specific needs. Keep VecStore focused on vector operations,
//! while providing guidance for application-level patterns.

use crate::store::Neighbor;

/// Query expansion strategies
///
/// Expands a single query into multiple related queries to improve recall.
pub struct QueryExpander;

impl QueryExpander {
    /// Expand query using synonym substitution
    ///
    /// This is a simple demonstration. In production, use:
    /// - WordNet for synonyms
    /// - LLM for semantic expansion
    /// - Domain-specific thesaurus
    ///
    /// # Example
    ///
    /// ```
    /// use vecstore::rag_utils::QueryExpander;
    ///
    /// let queries = QueryExpander::expand_with_synonyms(
    ///     "rust programming",
    ///     &[("rust", &["rustlang"]), ("programming", &["coding", "development"])]
    /// );
    /// assert!(queries.len() > 1);
    /// ```
    pub fn expand_with_synonyms(query: &str, synonyms: &[(&str, &[&str])]) -> Vec<String> {
        let mut expanded = vec![query.to_string()];

        for (word, syns) in synonyms {
            if query.contains(word) {
                for syn in syns.iter() {
                    expanded.push(query.replace(word, syn));
                }
            }
        }

        expanded
    }

    /// Generate sub-queries by decomposing complex query
    ///
    /// Breaks down complex questions into simpler sub-queries.
    ///
    /// # Example
    ///
    /// ```
    /// use vecstore::rag_utils::QueryExpander;
    ///
    /// // In production, use LLM to decompose
    /// let sub_queries = QueryExpander::decompose_query(
    ///     "What are the benefits and drawbacks of Rust?",
    ///     2
    /// );
    /// // Returns: ["benefits of Rust", "drawbacks of Rust"]
    /// ```
    pub fn decompose_query(query: &str, num_subqueries: usize) -> Vec<String> {
        // Simple demonstration: split on conjunctions
        let parts: Vec<&str> = query.split(" and ").collect();

        let mut subqueries = Vec::new();
        for part in parts.iter().take(num_subqueries) {
            let cleaned = part.trim_matches(|c: char| !c.is_alphanumeric() && c != ' ');
            if !cleaned.is_empty() {
                subqueries.push(cleaned.to_string());
            }
        }

        if subqueries.is_empty() {
            vec![query.to_string()]
        } else {
            subqueries
        }
    }

    /// Generate multiple query variations
    ///
    /// Creates different phrasings of the same question.
    ///
    /// # Example
    ///
    /// ```
    /// use vecstore::rag_utils::QueryExpander;
    ///
    /// let variations = QueryExpander::generate_variations(
    ///     "how to learn rust",
    ///     &["how to", "what is", "guide to"]
    /// );
    /// // Returns variations with different question starters
    /// ```
    pub fn generate_variations(query: &str, prefixes: &[&str]) -> Vec<String> {
        let mut variations = vec![query.to_string()];

        // Try replacing question words with alternatives
        for prefix in prefixes {
            for existing_prefix in &["how to", "what is", "why does", "when to"] {
                if query.starts_with(existing_prefix) {
                    let rest = query.trim_start_matches(existing_prefix).trim();
                    variations.push(format!("{} {}", prefix, rest));
                }
            }
        }

        variations
    }
}

/// HyDE (Hypothetical Document Embeddings) pattern
///
/// Generates a hypothetical answer to the query, then searches for similar documents.
/// Often more effective than searching with the question directly.
///
/// ## How it Works
///
/// 1. Given a question, generate a hypothetical answer (using LLM)
/// 2. Embed the hypothetical answer
/// 3. Search for real documents similar to the hypothetical answer
/// 4. Real documents that answer the question will be similar to the hypothetical answer
///
/// ## Example
///
/// ```no_run
/// use vecstore::rag_utils::HyDEHelper;
///
/// // In production, use actual LLM
/// let hypothetical_doc_generator = |query: &str| {
///     format!("Here is an answer to '{}': ...", query)
/// };
///
/// let hyde = HyDEHelper::new(hypothetical_doc_generator);
/// let hypothetical_doc = hyde.generate_hypothetical_document("What is Rust?");
/// // Then embed this hypothetical doc and search with it
/// ```
pub struct HyDEHelper<F>
where
    F: Fn(&str) -> String,
{
    generator: F,
}

impl<F> HyDEHelper<F>
where
    F: Fn(&str) -> String,
{
    /// Create new HyDE helper with document generator function
    ///
    /// # Arguments
    ///
    /// * `generator` - Function that generates hypothetical document from query
    ///   In production, this would call an LLM
    pub fn new(generator: F) -> Self {
        Self { generator }
    }

    /// Generate hypothetical document for query
    ///
    /// This document should be what an ideal answer would look like.
    /// Search for real documents similar to this.
    pub fn generate_hypothetical_document(&self, query: &str) -> String {
        (self.generator)(query)
    }
}

/// Multi-query retrieval with result fusion
///
/// Executes multiple query variations and fuses results.
pub struct MultiQueryRetrieval;

impl MultiQueryRetrieval {
    /// Fuse results from multiple queries using Reciprocal Rank Fusion (RRF)
    ///
    /// RRF is a simple but effective fusion method that works without scores.
    ///
    /// Formula: `score(doc) = sum(1 / (rank + k))` for each query where doc appears
    ///
    /// # Arguments
    ///
    /// * `result_sets` - Results from each query variant
    /// * `k` - RRF parameter (typically 60), prevents division by zero and reduces impact of high ranks
    ///
    /// # Example
    ///
    /// ```
    /// use vecstore::{rag_utils::MultiQueryRetrieval, Neighbor, Metadata};
    /// use std::collections::HashMap;
    ///
    /// let results1 = vec![/* results from query 1 */];
    /// let results2 = vec![/* results from query 2 */];
    ///
    /// let fused = MultiQueryRetrieval::reciprocal_rank_fusion(
    ///     vec![results1, results2],
    ///     60
    /// );
    /// ```
    pub fn reciprocal_rank_fusion(result_sets: Vec<Vec<Neighbor>>, k: usize) -> Vec<Neighbor> {
        use std::collections::HashMap;

        // Calculate RRF scores for each document
        let mut scores: HashMap<String, f32> = HashMap::new();
        let mut doc_lookup: HashMap<String, Neighbor> = HashMap::new();

        for results in result_sets {
            for (rank, neighbor) in results.into_iter().enumerate() {
                let rrf_score = 1.0 / ((rank + k) as f32);
                *scores.entry(neighbor.id.clone()).or_insert(0.0) += rrf_score;
                doc_lookup.entry(neighbor.id.clone()).or_insert(neighbor);
            }
        }

        // Sort by RRF score
        let mut scored_docs: Vec<(String, f32)> = scores.into_iter().collect();
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return documents with updated scores
        scored_docs
            .into_iter()
            .filter_map(|(id, score)| {
                doc_lookup.remove(&id).map(|mut neighbor| {
                    neighbor.score = score;
                    neighbor
                })
            })
            .collect()
    }

    /// Simple score-based fusion (average scores)
    ///
    /// Averages scores across result sets. Requires all queries to return scores.
    pub fn average_fusion(result_sets: Vec<Vec<Neighbor>>) -> Vec<Neighbor> {
        use std::collections::HashMap;

        let mut score_sums: HashMap<String, f32> = HashMap::new();
        let mut score_counts: HashMap<String, usize> = HashMap::new();
        let mut doc_lookup: HashMap<String, Neighbor> = HashMap::new();

        for results in result_sets {
            for neighbor in results {
                *score_sums.entry(neighbor.id.clone()).or_insert(0.0) += neighbor.score;
                *score_counts.entry(neighbor.id.clone()).or_insert(0) += 1;
                doc_lookup.entry(neighbor.id.clone()).or_insert(neighbor);
            }
        }

        // Calculate averages
        let mut averaged: Vec<(String, f32)> = score_sums
            .into_iter()
            .map(|(id, sum)| {
                let count = score_counts[&id] as f32;
                (id, sum / count)
            })
            .collect();

        averaged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        averaged
            .into_iter()
            .filter_map(|(id, score)| {
                doc_lookup.remove(&id).map(|mut neighbor| {
                    neighbor.score = score;
                    neighbor
                })
            })
            .collect()
    }
}

/// Context window management for RAG
///
/// Helps manage token limits when constructing prompts with retrieved documents.
pub struct ContextWindowManager {
    max_tokens: usize,
}

impl ContextWindowManager {
    /// Create new context window manager
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum tokens available for context (e.g., 4096 for GPT-3.5)
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    /// Fit documents into context window
    ///
    /// Greedily adds documents until hitting token limit.
    ///
    /// # Arguments
    ///
    /// * `documents` - Ranked documents to include
    /// * `token_estimator` - Function to estimate tokens in text
    /// * `reserved_tokens` - Tokens reserved for prompt template, query, etc.
    ///
    /// # Returns
    ///
    /// Documents that fit within the context window
    pub fn fit_documents<F>(
        &self,
        mut documents: Vec<Neighbor>,
        token_estimator: F,
        reserved_tokens: usize,
    ) -> Vec<Neighbor>
    where
        F: Fn(&str) -> usize,
    {
        let available_tokens = self.max_tokens.saturating_sub(reserved_tokens);
        let mut used_tokens = 0;
        let mut fitted = Vec::new();

        for doc in documents.drain(..) {
            let text = doc
                .metadata
                .fields
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let doc_tokens = token_estimator(text);

            if used_tokens + doc_tokens <= available_tokens {
                used_tokens += doc_tokens;
                fitted.push(doc);
            } else {
                break;
            }
        }

        fitted
    }

    /// Simple token estimator (words * 1.3)
    ///
    /// Rough approximation. For production, use tiktoken or similar.
    pub fn simple_token_estimator(text: &str) -> usize {
        (text.split_whitespace().count() as f32 * 1.3) as usize
    }
}

/// Conversation memory for chat-based RAG applications
///
/// Manages conversation history with automatic token-based trimming.
/// **HYBRID**: Simple by default, customizable when needed.
///
/// # Simple Usage (Default)
///
/// ```
/// use vecstore::rag_utils::ConversationMemory;
///
/// // Just works with default token estimator
/// let mut memory = ConversationMemory::new(4096);
/// memory.add_message("user", "Hello!");
/// memory.add_message("assistant", "Hi there!");
///
/// let messages = memory.get_messages();
/// assert_eq!(messages.len(), 2);
/// ```
///
/// # Advanced Usage (Custom Token Estimator)
///
/// ```
/// use vecstore::rag_utils::ConversationMemory;
///
/// // Custom token estimator for specific LLM
/// let custom_estimator = |text: &str| {
///     // Use tiktoken or your LLM's tokenizer
///     text.len() / 4
/// };
///
/// let mut memory = ConversationMemory::with_token_estimator(4096, Box::new(custom_estimator));
/// memory.add_message("user", "Question");
/// ```
pub struct ConversationMemory {
    messages: Vec<Message>,
    max_tokens: usize,
    token_estimator: Box<dyn Fn(&str) -> usize>,
}

/// A single conversation message
#[derive(Debug, Clone)]
pub struct Message {
    /// Role: "user", "assistant", or "system"
    pub role: String,
    /// Message content
    pub content: String,
}

impl ConversationMemory {
    /// Create new conversation memory with simple token estimator
    ///
    /// **HYBRID**: Simple by default - just works with reasonable approximation
    pub fn new(max_tokens: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_tokens,
            token_estimator: Box::new(ContextWindowManager::simple_token_estimator),
        }
    }

    /// Create with custom token estimator (advanced, opt-in)
    pub fn with_token_estimator(
        max_tokens: usize,
        token_estimator: Box<dyn Fn(&str) -> usize>,
    ) -> Self {
        Self {
            messages: Vec::new(),
            max_tokens,
            token_estimator,
        }
    }

    /// Add a message to the conversation
    pub fn add_message(&mut self, role: impl Into<String>, content: impl Into<String>) {
        self.messages.push(Message {
            role: role.into(),
            content: content.into(),
        });
        self.trim_to_fit();
    }

    /// Get all messages
    pub fn get_messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get messages as formatted string (for LLM prompts)
    pub fn format_messages(&self) -> String {
        self.messages
            .iter()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Clear all messages
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Get current token count
    pub fn token_count(&self) -> usize {
        self.messages
            .iter()
            .map(|msg| (self.token_estimator)(&msg.content))
            .sum()
    }

    /// Trim old messages to fit token limit (FIFO, keeps system messages)
    fn trim_to_fit(&mut self) {
        while self.token_count() > self.max_tokens && self.messages.len() > 1 {
            // Always keep system messages, remove oldest user/assistant messages
            if let Some(pos) = self.messages.iter().position(|msg| msg.role != "system") {
                self.messages.remove(pos);
            } else {
                break;
            }
        }
    }
}

/// Prompt template system with variable substitution
///
/// Simple yet powerful template system for RAG prompts.
/// **HYBRID**: Simple by default, no forced template engines.
///
/// # Simple Usage
///
/// ```
/// use vecstore::rag_utils::PromptTemplate;
///
/// let template = PromptTemplate::new(
///     "Answer the question: {question}\n\nContext: {context}"
/// );
///
/// let prompt = template
///     .fill("question", "What is Rust?")
///     .fill("context", "Rust is a systems programming language.")
///     .render();
///
/// assert!(prompt.contains("What is Rust?"));
/// ```
///
/// # Advanced Usage (with defaults)
///
/// ```
/// use vecstore::rag_utils::PromptTemplate;
///
/// let template = PromptTemplate::new("Question: {question}\nContext: {context}")
///     .with_default("context", "No context available");
///
/// let prompt = template.fill("question", "Hello?").render();
/// assert!(prompt.contains("No context available"));
/// ```
pub struct PromptTemplate {
    template: String,
    variables: std::collections::HashMap<String, String>,
    defaults: std::collections::HashMap<String, String>,
}

impl PromptTemplate {
    /// Create a new prompt template (simple, just works)
    ///
    /// Variables are marked with {variable_name}
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
            variables: std::collections::HashMap::new(),
            defaults: std::collections::HashMap::new(),
        }
    }

    /// Set a variable value
    pub fn fill(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.variables.insert(key.into(), value.into());
        self
    }

    /// Set default value for a variable (advanced, opt-in)
    pub fn with_default(mut self, key: impl Into<String>, default: impl Into<String>) -> Self {
        self.defaults.insert(key.into(), default.into());
        self
    }

    /// Render the template with current variables
    pub fn render(&self) -> String {
        let mut result = self.template.clone();

        // Replace variables
        for (key, value) in &self.variables {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }

        // Replace with defaults if variable not filled
        for (key, default) in &self.defaults {
            let placeholder = format!("{{{}}}", key);
            if result.contains(&placeholder) {
                result = result.replace(&placeholder, default);
            }
        }

        result
    }

    /// Render and clear (convenient for reuse)
    pub fn render_and_reset(&mut self) -> String {
        let rendered = self.render();
        self.variables.clear();
        rendered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Metadata;
    use std::collections::HashMap;

    fn make_neighbor(id: &str, score: f32) -> Neighbor {
        Neighbor {
            id: id.to_string(),
            score,
            metadata: Metadata {
                fields: HashMap::new(),
            },
        }
    }

    #[test]
    fn test_query_expansion_synonyms() {
        let expanded = QueryExpander::expand_with_synonyms(
            "rust programming",
            &[("rust", &["rustlang"]), ("programming", &["coding"])],
        );

        assert!(expanded.len() >= 3); // Original + 2 expansions
        assert!(expanded.contains(&"rust programming".to_string()));
        assert!(expanded.contains(&"rustlang programming".to_string()));
        assert!(expanded.contains(&"rust coding".to_string()));
    }

    #[test]
    fn test_query_decomposition() {
        let sub_queries = QueryExpander::decompose_query("benefits and drawbacks", 2);
        assert_eq!(sub_queries.len(), 2);
        assert!(sub_queries[0].contains("benefits"));
        assert!(sub_queries[1].contains("drawbacks"));
    }

    #[test]
    fn test_hyde_helper() {
        let generator = |query: &str| format!("Hypothetical answer to: {}", query);
        let hyde = HyDEHelper::new(generator);

        let doc = hyde.generate_hypothetical_document("What is Rust?");
        assert!(doc.contains("What is Rust?"));
    }

    #[test]
    fn test_reciprocal_rank_fusion() {
        let results1 = vec![
            make_neighbor("doc1", 0.9),
            make_neighbor("doc2", 0.8),
            make_neighbor("doc3", 0.7),
        ];

        let results2 = vec![
            make_neighbor("doc2", 0.95), // doc2 appears in both
            make_neighbor("doc4", 0.85),
            make_neighbor("doc1", 0.75), // doc1 appears in both
        ];

        let fused = MultiQueryRetrieval::reciprocal_rank_fusion(vec![results1, results2], 60);

        // doc2 and doc1 should rank high (appear in both)
        assert!(fused.len() >= 2);
        assert!(fused[0].id == "doc1" || fused[0].id == "doc2");
    }

    #[test]
    fn test_average_fusion() {
        let results1 = vec![make_neighbor("doc1", 0.9), make_neighbor("doc2", 0.8)];
        let results2 = vec![make_neighbor("doc1", 0.7), make_neighbor("doc2", 0.6)];

        let fused = MultiQueryRetrieval::average_fusion(vec![results1, results2]);

        assert_eq!(fused.len(), 2);
        // doc1: (0.9 + 0.7) / 2 = 0.8
        // doc2: (0.8 + 0.6) / 2 = 0.7
        assert_eq!(fused[0].id, "doc1");
        assert!((fused[0].score - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_context_window_manager() {
        let manager = ContextWindowManager::new(100);

        let mut meta1 = Metadata {
            fields: HashMap::new(),
        };
        meta1
            .fields
            .insert("text".to_string(), serde_json::json!("short text"));

        let mut meta2 = Metadata {
            fields: HashMap::new(),
        };
        meta2
            .fields
            .insert("text".to_string(), serde_json::json!("another short text"));

        let docs = vec![
            Neighbor {
                id: "doc1".to_string(),
                score: 0.9,
                metadata: meta1,
            },
            Neighbor {
                id: "doc2".to_string(),
                score: 0.8,
                metadata: meta2,
            },
        ];

        let fitted = manager.fit_documents(docs, ContextWindowManager::simple_token_estimator, 50);

        assert!(fitted.len() > 0);
    }

    // Conversation memory tests
    #[test]
    fn test_conversation_memory_basic() {
        let mut memory = ConversationMemory::new(1000);
        memory.add_message("user", "Hello");
        memory.add_message("assistant", "Hi there!");

        assert_eq!(memory.get_messages().len(), 2);
        assert_eq!(memory.get_messages()[0].role, "user");
        assert_eq!(memory.get_messages()[0].content, "Hello");
    }

    #[test]
    fn test_conversation_memory_trimming() {
        // Very small token limit to force trimming
        let mut memory = ConversationMemory::new(10);
        memory.add_message(
            "user",
            "First message with lots of content here that will use many tokens",
        );
        memory.add_message(
            "assistant",
            "Response to first message with even more content",
        );
        memory.add_message(
            "user",
            "Second message that will definitely exceed the tiny limit we set",
        );

        // Should have trimmed the first messages (keeps at least 1)
        assert!(memory.get_messages().len() < 3);
        // Trimming works - even if last message exceeds limit, it's kept
    }

    #[test]
    fn test_conversation_memory_keeps_system_messages() {
        let mut memory = ConversationMemory::new(50);
        memory.add_message("system", "You are a helpful assistant");
        memory.add_message("user", "Message 1 with enough content to fill tokens");
        memory.add_message("user", "Message 2 with enough content to fill tokens");

        // System message should always be kept
        assert!(memory.get_messages().iter().any(|m| m.role == "system"));
    }

    #[test]
    fn test_conversation_memory_format() {
        let mut memory = ConversationMemory::new(1000);
        memory.add_message("user", "Hello");
        memory.add_message("assistant", "Hi");

        let formatted = memory.format_messages();
        assert!(formatted.contains("user: Hello"));
        assert!(formatted.contains("assistant: Hi"));
    }

    #[test]
    fn test_conversation_memory_clear() {
        let mut memory = ConversationMemory::new(1000);
        memory.add_message("user", "Test");
        memory.clear();

        assert_eq!(memory.get_messages().len(), 0);
    }

    #[test]
    fn test_conversation_memory_custom_estimator() {
        let custom_estimator = |text: &str| text.len();
        let mut memory = ConversationMemory::with_token_estimator(100, Box::new(custom_estimator));

        memory.add_message("user", "Hello");
        assert!(memory.token_count() > 0);
    }

    // Prompt template tests
    #[test]
    fn test_prompt_template_basic() {
        let template = PromptTemplate::new("Hello {name}!");
        let result = template.fill("name", "World").render();

        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_prompt_template_multiple_variables() {
        let template = PromptTemplate::new("Question: {question}\nContext: {context}");
        let result = template
            .fill("question", "What is Rust?")
            .fill("context", "Rust is a language.")
            .render();

        assert!(result.contains("What is Rust?"));
        assert!(result.contains("Rust is a language."));
    }

    #[test]
    fn test_prompt_template_defaults() {
        let template = PromptTemplate::new("Question: {question}\nContext: {context}")
            .with_default("context", "No context");

        let result = template.fill("question", "Hello").render();

        assert!(result.contains("Hello"));
        assert!(result.contains("No context"));
    }

    #[test]
    fn test_prompt_template_override_default() {
        let template = PromptTemplate::new("Value: {value}").with_default("value", "default");

        // Override the default
        let result = template.fill("value", "custom").render();
        assert_eq!(result, "Value: custom");
    }

    #[test]
    fn test_prompt_template_render_and_reset() {
        let mut template = PromptTemplate::new("Name: {name}").fill("name", "Alice");

        let result1 = template.render();
        assert!(result1.contains("Alice"));

        // Reset and reuse
        let result2 = template.render_and_reset();
        assert!(result2.contains("Alice"));

        // Should be reset now (variables cleared)
        let result3 = template.render();
        assert!(result3.contains("{name}")); // Unfilled after reset
    }
}
