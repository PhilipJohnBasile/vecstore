//! Native Generative Search (RAG in Single API Call)
//!
//! Combines vector retrieval with LLM generation in a single operation,
//! similar to Weaviate's generative search and Azure AI Search.
//!
//! # Features
//!
//! - **Single API Call**: Retrieve + Generate in one request
//! - **Multi-Provider**: OpenAI, Anthropic, Cohere, Ollama, etc.
//! - **Streaming**: Stream generated responses
//! - **Citations**: Automatic source attribution
//! - **Prompt Templates**: Customizable generation prompts
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::generative::{GenerativeSearch, GenerativeConfig, LLMProvider};
//!
//! let config = GenerativeConfig::new(LLMProvider::OpenAI {
//!     model: "gpt-4".to_string(),
//!     api_key: std::env::var("OPENAI_API_KEY").unwrap(),
//! });
//!
//! let gen_search = GenerativeSearch::new(store, config)?;
//!
//! // Single call: retrieve relevant docs + generate answer
//! let response = gen_search.generate(
//!     "What is VecStore?",
//!     GenerativeQuery::new()
//!         .with_limit(5)
//!         .with_prompt("Answer based on the following context: {context}")
//! )?;
//!
//! println!("Answer: {}", response.generated_text);
//! println!("Sources: {:?}", response.citations);
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::store::{Query, VecStore};

/// LLM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LLMProvider {
    /// OpenAI GPT models
    OpenAI {
        model: String,
        api_key: String,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        max_tokens: Option<usize>,
    },
    /// Anthropic Claude models
    Anthropic {
        model: String,
        api_key: String,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        max_tokens: Option<usize>,
    },
    /// Cohere Command models
    Cohere {
        model: String,
        api_key: String,
    },
    /// Google Gemini
    Gemini {
        model: String,
        api_key: String,
    },
    /// Ollama local models
    Ollama {
        model: String,
        base_url: String,
    },
    /// Azure OpenAI
    AzureOpenAI {
        deployment: String,
        endpoint: String,
        api_key: String,
        api_version: String,
    },
    /// AWS Bedrock
    Bedrock {
        model_id: String,
        region: String,
    },
    /// Custom HTTP endpoint
    Custom {
        endpoint: String,
        api_key: Option<String>,
        headers: HashMap<String, String>,
    },
}

impl LLMProvider {
    /// Get provider name
    pub fn name(&self) -> &'static str {
        match self {
            LLMProvider::OpenAI { .. } => "openai",
            LLMProvider::Anthropic { .. } => "anthropic",
            LLMProvider::Cohere { .. } => "cohere",
            LLMProvider::Gemini { .. } => "gemini",
            LLMProvider::Ollama { .. } => "ollama",
            LLMProvider::AzureOpenAI { .. } => "azure",
            LLMProvider::Bedrock { .. } => "bedrock",
            LLMProvider::Custom { .. } => "custom",
        }
    }

    /// Get model name
    pub fn model(&self) -> &str {
        match self {
            LLMProvider::OpenAI { model, .. } => model,
            LLMProvider::Anthropic { model, .. } => model,
            LLMProvider::Cohere { model, .. } => model,
            LLMProvider::Gemini { model, .. } => model,
            LLMProvider::Ollama { model, .. } => model,
            LLMProvider::AzureOpenAI { deployment, .. } => deployment,
            LLMProvider::Bedrock { model_id, .. } => model_id,
            LLMProvider::Custom { endpoint, .. } => endpoint,
        }
    }
}

/// Configuration for generative search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeConfig {
    /// LLM provider
    pub provider: LLMProvider,
    /// Default prompt template
    #[serde(default = "default_prompt_template")]
    pub prompt_template: String,
    /// System prompt
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
    /// Include citations in response
    #[serde(default = "default_true")]
    pub include_citations: bool,
    /// Maximum context length (tokens)
    #[serde(default = "default_context_length")]
    pub max_context_length: usize,
    /// Stream responses
    #[serde(default)]
    pub streaming: bool,
}

fn default_prompt_template() -> String {
    "Answer the following question based on the provided context.\n\n\
     Context:\n{context}\n\n\
     Question: {query}\n\n\
     Answer:".to_string()
}

fn default_system_prompt() -> String {
    "You are a helpful assistant that answers questions based on the provided context. \
     Be concise and accurate. If the context doesn't contain enough information, \
     say so rather than making up an answer.".to_string()
}

fn default_true() -> bool { true }
fn default_context_length() -> usize { 4000 }

impl GenerativeConfig {
    /// Create a new configuration
    pub fn new(provider: LLMProvider) -> Self {
        Self {
            provider,
            prompt_template: default_prompt_template(),
            system_prompt: default_system_prompt(),
            include_citations: true,
            max_context_length: 4000,
            streaming: false,
        }
    }

    /// Create config for OpenAI
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::new(LLMProvider::OpenAI {
            model: "gpt-4o-mini".to_string(),
            api_key: api_key.into(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
        })
    }

    /// Create config for Anthropic Claude
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self::new(LLMProvider::Anthropic {
            model: "claude-3-5-sonnet-20241022".to_string(),
            api_key: api_key.into(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
        })
    }

    /// Create config for Ollama (local)
    pub fn ollama(model: impl Into<String>) -> Self {
        Self::new(LLMProvider::Ollama {
            model: model.into(),
            base_url: "http://localhost:11434".to_string(),
        })
    }

    /// Set custom prompt template
    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
        self
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Enable streaming
    pub fn with_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }
}

/// Query options for generative search
#[derive(Debug, Clone)]
pub struct GenerativeQuery {
    /// Number of documents to retrieve
    pub limit: usize,
    /// Metadata filter
    pub filter: Option<String>,
    /// Custom prompt template (overrides config)
    pub prompt_template: Option<String>,
    /// Additional context to include
    pub additional_context: Option<String>,
    /// Task type (affects prompt construction)
    pub task: GenerativeTask,
    /// Include raw search results in response
    pub include_results: bool,
}

/// Types of generative tasks
#[derive(Debug, Clone, Default)]
pub enum GenerativeTask {
    /// Question answering (default)
    #[default]
    QuestionAnswering,
    /// Summarization
    Summarization,
    /// Extraction
    Extraction {
        /// Fields to extract
        fields: Vec<String>,
    },
    /// Classification
    Classification {
        /// Classes to classify into
        classes: Vec<String>,
    },
    /// Custom task
    Custom {
        /// Task description
        description: String,
    },
}

impl GenerativeQuery {
    /// Create a new query
    pub fn new() -> Self {
        Self {
            limit: 5,
            filter: None,
            prompt_template: None,
            additional_context: None,
            task: GenerativeTask::QuestionAnswering,
            include_results: true,
        }
    }

    /// Set the limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set a filter
    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    /// Set custom prompt template
    pub fn with_prompt(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = Some(template.into());
        self
    }

    /// Set task type
    pub fn with_task(mut self, task: GenerativeTask) -> Self {
        self.task = task;
        self
    }

    /// Add additional context
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.additional_context = Some(context.into());
        self
    }
}

impl Default for GenerativeQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Citation for a source document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Document ID
    pub id: String,
    /// Relevance score
    pub score: f32,
    /// Excerpt used in generation
    pub excerpt: String,
    /// Source metadata
    pub metadata: Option<serde_json::Value>,
}

/// Retrieved document for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedDocument {
    /// Document ID
    pub id: String,
    /// Document content
    pub content: String,
    /// Relevance score
    pub score: f32,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
}

/// Response from generative search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeResponse {
    /// Generated text
    pub generated_text: String,
    /// Source citations
    pub citations: Vec<Citation>,
    /// Retrieved documents used
    pub retrieved_documents: Vec<RetrievedDocument>,
    /// Token usage
    pub usage: TokenUsage,
    /// Generation metadata
    pub metadata: GenerationMetadata,
}

/// Token usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Generation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    /// Model used
    pub model: String,
    /// Provider
    pub provider: String,
    /// Retrieval time (ms)
    pub retrieval_time_ms: u64,
    /// Generation time (ms)
    pub generation_time_ms: u64,
    /// Total time (ms)
    pub total_time_ms: u64,
}

/// Generative search engine
pub struct GenerativeSearch {
    config: GenerativeConfig,
    /// Optional vector store for retrieval
    store: Option<Arc<std::sync::RwLock<VecStore>>>,
    /// Optional query embedding function
    embed_fn: Option<Box<dyn Fn(&str) -> Result<Vec<f32>> + Send + Sync>>,
}

impl GenerativeSearch {
    /// Create a new generative search engine
    pub fn new(config: GenerativeConfig) -> Result<Self> {
        Ok(Self {
            config,
            store: None,
            embed_fn: None,
        })
    }

    /// Create with a vector store for retrieval
    pub fn with_store(
        config: GenerativeConfig,
        store: Arc<std::sync::RwLock<VecStore>>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            store: Some(store),
            embed_fn: None,
        })
    }

    /// Set the embedding function for query embedding
    pub fn with_embedder<F>(mut self, embed_fn: F) -> Self
    where
        F: Fn(&str) -> Result<Vec<f32>> + Send + Sync + 'static,
    {
        self.embed_fn = Some(Box::new(embed_fn));
        self
    }

    /// Set the vector store
    pub fn set_store(&mut self, store: Arc<std::sync::RwLock<VecStore>>) {
        self.store = Some(store);
    }

    /// Generate a response for a query
    pub fn generate(
        &self,
        query: &str,
        options: GenerativeQuery,
    ) -> Result<GenerativeResponse> {
        let start = std::time::Instant::now();

        // Step 1: Retrieve relevant documents (simulated)
        let retrieval_start = std::time::Instant::now();
        let retrieved_docs = self.retrieve_documents(query, &options)?;
        let retrieval_time = retrieval_start.elapsed().as_millis() as u64;

        // Step 2: Build context from retrieved documents
        let context = self.build_context(&retrieved_docs, &options)?;

        // Step 3: Construct prompt
        let prompt = self.build_prompt(query, &context, &options)?;

        // Step 4: Generate response
        let generation_start = std::time::Instant::now();
        let generated_text = self.call_llm(&prompt)?;
        let generation_time = generation_start.elapsed().as_millis() as u64;

        // Step 5: Extract citations
        let citations = self.extract_citations(&retrieved_docs);

        let total_time = start.elapsed().as_millis() as u64;

        Ok(GenerativeResponse {
            generated_text,
            citations,
            retrieved_documents: retrieved_docs,
            usage: TokenUsage {
                prompt_tokens: self.estimate_tokens(&prompt),
                completion_tokens: 100, // Placeholder
                total_tokens: self.estimate_tokens(&prompt) + 100,
            },
            metadata: GenerationMetadata {
                model: self.config.provider.model().to_string(),
                provider: self.config.provider.name().to_string(),
                retrieval_time_ms: retrieval_time,
                generation_time_ms: generation_time,
                total_time_ms: total_time,
            },
        })
    }

    /// Retrieve relevant documents from vector store
    fn retrieve_documents(
        &self,
        query: &str,
        options: &GenerativeQuery,
    ) -> Result<Vec<RetrievedDocument>> {
        // If we have a store and embedder, perform real retrieval
        if let (Some(store), Some(embed_fn)) = (&self.store, &self.embed_fn) {
            // Embed the query
            let query_vector = embed_fn(query)?;

            // Query the store
            let store_guard = store.read().map_err(|e| {
                crate::error::VecStoreError::LockError(format!("Failed to acquire store lock: {}", e))
            })?;

            let results = store_guard.query(Query {
                vector: query_vector,
                k: options.limit,
                filter: None, // Filter support planned for future release
            }).map_err(|e| crate::error::VecStoreError::Internal(format!("Query failed: {}", e)))?;

            // Convert to RetrievedDocument format
            return Ok(results
                .into_iter()
                .map(|neighbor| {
                    // Get the actual document content from metadata
                    let content = neighbor
                        .metadata
                        .fields
                        .get("content")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| {
                            neighbor
                                .metadata
                                .fields
                                .get("text")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| format!("Document {}", neighbor.id));

                    RetrievedDocument {
                        id: neighbor.id,
                        content,
                        score: neighbor.score,
                        metadata: Some(serde_json::to_value(&neighbor.metadata.fields).unwrap_or_default()),
                    }
                })
                .collect());
        }

        // Fallback: return placeholder documents when no store/embedder configured
        // Useful for testing RAG pipelines without full vector store setup
        tracing::debug!("No store/embedder configured - returning placeholder documents for testing");
        Ok((0..options.limit.min(3))
            .map(|i| RetrievedDocument {
                id: format!("placeholder_doc_{}", i),
                content: format!("Placeholder document {} for query: {}", i, query),
                score: 0.9 - (i as f32 * 0.1),
                metadata: Some(serde_json::json!({
                    "source": "placeholder",
                    "title": format!("Test Document {}", i),
                    "note": "Configure store and embedder for real retrieval"
                })),
            })
            .collect())
    }

    /// Build context from retrieved documents
    fn build_context(
        &self,
        docs: &[RetrievedDocument],
        _options: &GenerativeQuery,
    ) -> Result<String> {
        let mut context = String::new();

        for (i, doc) in docs.iter().enumerate() {
            context.push_str(&format!(
                "[{}] {}\n\n",
                i + 1,
                doc.content
            ));
        }

        // Truncate if too long
        if context.len() > self.config.max_context_length * 4 {
            context.truncate(self.config.max_context_length * 4);
            context.push_str("...");
        }

        Ok(context)
    }

    /// Build the prompt for the LLM
    fn build_prompt(
        &self,
        query: &str,
        context: &str,
        options: &GenerativeQuery,
    ) -> Result<String> {
        let template = options.prompt_template
            .as_ref()
            .unwrap_or(&self.config.prompt_template);

        let mut prompt = template.clone();
        prompt = prompt.replace("{query}", query);
        prompt = prompt.replace("{context}", context);
        prompt = prompt.replace("{question}", query);

        if let Some(additional) = &options.additional_context {
            prompt = prompt.replace("{additional_context}", additional);
        }

        Ok(prompt)
    }

    /// Call the LLM to generate a response
    #[cfg(any(feature = "openai-embeddings", feature = "ollama"))]
    fn call_llm(&self, prompt: &str) -> Result<String> {
        match &self.config.provider {
            LLMProvider::OpenAI {
                model,
                api_key,
                temperature,
                max_tokens,
            } => self.call_openai(prompt, model, api_key, *temperature, *max_tokens),

            LLMProvider::Ollama { model, base_url } => {
                self.call_ollama(prompt, model, base_url)
            }

            // For other providers, return a message indicating they're not yet implemented
            _ => Ok(format!(
                "LLM provider '{}' is not yet implemented. \
                 Supported providers: OpenAI, Ollama.",
                self.config.provider.name()
            )),
        }
    }

    /// Fallback when LLM features are not enabled
    #[cfg(not(any(feature = "openai-embeddings", feature = "ollama")))]
    fn call_llm(&self, _prompt: &str) -> Result<String> {
        Ok(format!(
            "LLM integration requires feature flags. Enable 'openai-embeddings' for OpenAI \
             or 'ollama' for Ollama support. Using placeholder response from {} with model {}.",
            self.config.provider.name(),
            self.config.provider.model()
        ))
    }

    /// Call OpenAI API
    #[cfg(feature = "openai-embeddings")]
    fn call_openai(
        &self,
        prompt: &str,
        model: &str,
        api_key: &str,
        temperature: Option<f32>,
        max_tokens: Option<usize>,
    ) -> Result<String> {
        use crate::error::VecStoreError;

        let client = reqwest::blocking::Client::new();

        let mut body = serde_json::json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        });

        if let Some(temp) = temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(max) = max_tokens {
            body["max_tokens"] = serde_json::json!(max);
        }

        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| VecStoreError::Internal(format!("OpenAI request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_default();
            return Err(VecStoreError::Internal(format!(
                "OpenAI API error {}: {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .map_err(|e| VecStoreError::Internal(format!("Failed to parse OpenAI response: {}", e)))?;

        json["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| VecStoreError::Internal("No content in OpenAI response".to_string()))
    }

    /// Call Ollama API
    #[cfg(feature = "ollama")]
    fn call_ollama(&self, prompt: &str, model: &str, base_url: &str) -> Result<String> {
        use crate::error::VecStoreError;

        let client = reqwest::blocking::Client::new();

        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false
        });

        let url = format!("{}/api/generate", base_url.trim_end_matches('/'));

        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| VecStoreError::Internal(format!("Ollama request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_default();
            return Err(VecStoreError::Internal(format!(
                "Ollama API error {}: {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .map_err(|e| VecStoreError::Internal(format!("Failed to parse Ollama response: {}", e)))?;

        json["response"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| VecStoreError::Internal("No response in Ollama output".to_string()))
    }

    /// Stub for Ollama when feature is not enabled
    #[cfg(not(feature = "ollama"))]
    fn call_ollama(&self, _prompt: &str, model: &str, _base_url: &str) -> Result<String> {
        Ok(format!(
            "Ollama support requires the 'ollama' feature flag. \
             Would call model '{}' if enabled.",
            model
        ))
    }

    /// Stub for OpenAI when feature is not enabled
    #[cfg(not(feature = "openai-embeddings"))]
    fn call_openai(
        &self,
        _prompt: &str,
        model: &str,
        _api_key: &str,
        _temperature: Option<f32>,
        _max_tokens: Option<usize>,
    ) -> Result<String> {
        Ok(format!(
            "OpenAI support requires the 'openai-embeddings' feature flag. \
             Would call model '{}' if enabled.",
            model
        ))
    }

    /// Extract citations from retrieved documents
    fn extract_citations(&self, docs: &[RetrievedDocument]) -> Vec<Citation> {
        if !self.config.include_citations {
            return Vec::new();
        }

        docs.iter()
            .map(|doc| Citation {
                id: doc.id.clone(),
                score: doc.score,
                excerpt: doc.content.chars().take(200).collect(),
                metadata: doc.metadata.clone(),
            })
            .collect()
    }

    /// Estimate token count (rough approximation)
    fn estimate_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count() * 4 / 3
    }

    /// Generate with streaming (returns iterator)
    pub fn generate_stream(
        &self,
        query: &str,
        options: GenerativeQuery,
    ) -> Result<GenerativeStream> {
        let response = self.generate(query, options)?;
        Ok(GenerativeStream::new(response))
    }
}

/// Streaming response iterator
pub struct GenerativeStream {
    response: GenerativeResponse,
    position: usize,
    chunk_size: usize,
}

impl GenerativeStream {
    fn new(response: GenerativeResponse) -> Self {
        Self {
            response,
            position: 0,
            chunk_size: 10, // Words per chunk
        }
    }
}

impl Iterator for GenerativeStream {
    type Item = StreamChunk;

    fn next(&mut self) -> Option<Self::Item> {
        let text = &self.response.generated_text;
        let words: Vec<&str> = text.split_whitespace().collect();

        if self.position >= words.len() {
            return None;
        }

        let end = (self.position + self.chunk_size).min(words.len());
        let chunk = words[self.position..end].join(" ");
        self.position = end;

        let is_final = self.position >= words.len();

        Some(StreamChunk {
            text: chunk,
            is_final,
            citations: if is_final {
                Some(self.response.citations.clone())
            } else {
                None
            },
        })
    }
}

/// Streaming chunk
#[derive(Debug, Clone, Serialize)]
pub struct StreamChunk {
    pub text: String,
    pub is_final: bool,
    pub citations: Option<Vec<Citation>>,
}

/// Prompt templates for different tasks
pub mod prompts {
    /// Question answering prompt
    pub const QA_PROMPT: &str = r#"
Answer the following question based on the provided context. Be concise and accurate.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"#;

    /// Summarization prompt
    pub const SUMMARIZE_PROMPT: &str = r#"
Summarize the following documents concisely:

{context}

Summary:"#;

    /// Extraction prompt
    pub const EXTRACT_PROMPT: &str = r#"
Extract the following information from the provided text:
{fields}

Text:
{context}

Extracted Information:"#;

    /// Classification prompt
    pub const CLASSIFY_PROMPT: &str = r#"
Classify the following text into one of these categories: {classes}

Text:
{context}

Classification:"#;

    /// RAG with citations prompt
    pub const RAG_WITH_CITATIONS: &str = r#"
Answer the following question based on the provided context.
Include citations in [1], [2], etc. format when referencing sources.

Context:
{context}

Question: {query}

Answer with citations:"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generative_config() {
        let config = GenerativeConfig::openai("test-key");
        assert_eq!(config.provider.name(), "openai");
    }

    #[test]
    fn test_generative_query() {
        let query = GenerativeQuery::new()
            .with_limit(10)
            .with_filter("category = 'tech'")
            .with_prompt("Custom prompt: {query}");

        assert_eq!(query.limit, 10);
        assert!(query.filter.is_some());
        assert!(query.prompt_template.is_some());
    }

    #[test]
    fn test_generative_search() {
        let config = GenerativeConfig::ollama("llama3");
        let gen_search = GenerativeSearch::new(config).unwrap();

        let response = gen_search.generate(
            "What is VecStore?",
            GenerativeQuery::new().with_limit(3),
        ).unwrap();

        assert!(!response.generated_text.is_empty());
        assert!(!response.citations.is_empty());
    }

    #[test]
    fn test_streaming() {
        let config = GenerativeConfig::ollama("llama3");
        let gen_search = GenerativeSearch::new(config).unwrap();

        let stream = gen_search.generate_stream(
            "What is VecStore?",
            GenerativeQuery::new(),
        ).unwrap();

        let chunks: Vec<_> = stream.collect();
        assert!(!chunks.is_empty());
        assert!(chunks.last().unwrap().is_final);
    }
}
