//! Text Splitting for RAG Applications
//!
//! Provides text chunking strategies for breaking down large documents into
//! embedding-sized chunks. Essential for RAG (Retrieval-Augmented Generation)
//! systems that need to store and search long documents.
//!
//! # Strategies
//!
//! - **RecursiveCharacterTextSplitter**: Splits on paragraph/sentence/word boundaries
//! - **TokenTextSplitter**: Splits based on token count (for LLMs with token limits)
//! - **MarkdownTextSplitter**: Markdown-aware splitting that respects header hierarchy
//! - **CodeTextSplitter**: Code-aware splitting that respects function/class boundaries
//! - **SemanticTextSplitter**: Embedding-based splitting that groups semantically similar content
//!
//! # Example
//!
//! ```no_run
//! use vecstore::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
//!
//! let splitter = RecursiveCharacterTextSplitter::new(500, 50);
//! let chunks = splitter.split_text("Long document text...")?;
//!
//! for (i, chunk) in chunks.iter().enumerate() {
//!     println!("Chunk {}: {} chars", i, chunk.len());
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::error::{Result, VecStoreError};

/// Trait for text splitting strategies
pub trait TextSplitter {
    /// Split text into chunks
    fn split_text(&self, text: &str) -> Result<Vec<String>>;

    /// Split text into chunks with metadata (position, length, etc.)
    fn split_with_metadata(&self, text: &str) -> Result<Vec<TextChunk>> {
        let chunks = self.split_text(text)?;
        Ok(chunks
            .into_iter()
            .enumerate()
            .map(|(i, content)| TextChunk {
                index: i,
                content,
                char_start: 0, // Simplified - could track actual positions
                char_end: 0,
            })
            .collect())
    }
}

/// A text chunk with metadata
#[derive(Debug, Clone, PartialEq)]
pub struct TextChunk {
    /// Chunk index in the original document
    pub index: usize,
    /// Chunk content
    pub content: String,
    /// Character start position in original text
    pub char_start: usize,
    /// Character end position in original text
    pub char_end: usize,
}

/// Recursive character-based text splitter
///
/// Tries to split on natural boundaries in this order:
/// 1. Double newlines (paragraphs)
/// 2. Single newlines (lines)
/// 3. Sentences (periods, question marks, exclamation points)
/// 4. Words (spaces)
/// 5. Characters (last resort)
///
/// # Example
///
/// ```no_run
/// use vecstore::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
///
/// let splitter = RecursiveCharacterTextSplitter::new(1000, 100);
/// let text = "First paragraph.\n\nSecond paragraph with more content...";
/// let chunks = splitter.split_text(text)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct RecursiveCharacterTextSplitter {
    /// Maximum chunk size in characters
    chunk_size: usize,
    /// Overlap between chunks in characters
    chunk_overlap: usize,
    /// Separators to try, in order of preference
    separators: Vec<String>,
}

impl RecursiveCharacterTextSplitter {
    /// Create a new recursive splitter
    ///
    /// # Arguments
    /// * `chunk_size` - Maximum characters per chunk
    /// * `chunk_overlap` - Characters to overlap between chunks (for context continuity)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vecstore::text_splitter::RecursiveCharacterTextSplitter;
    ///
    /// // 500 char chunks with 50 char overlap
    /// let splitter = RecursiveCharacterTextSplitter::new(500, 50);
    /// ```
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            separators: vec![
                "\n\n".to_string(), // Paragraphs
                "\n".to_string(),   // Lines
                ". ".to_string(),   // Sentences
                "! ".to_string(),
                "? ".to_string(),
                " ".to_string(), // Words
                "".to_string(),  // Characters
            ],
        }
    }

    /// Create with custom separators
    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        self.separators = separators;
        self
    }

    fn split_recursive(&self, text: &str, separators: &[String]) -> Vec<String> {
        if text.len() <= self.chunk_size {
            return vec![text.to_string()];
        }

        if separators.is_empty() {
            // Fallback: character-level split
            return self.split_by_chars(text);
        }

        let sep = &separators[0];
        let remaining_seps = &separators[1..];

        if sep.is_empty() {
            // Empty separator means character-level split
            return self.split_by_chars(text);
        }

        // Split by current separator
        let parts: Vec<&str> = text.split(sep).collect();

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        for (i, part) in parts.iter().enumerate() {
            let part_with_sep = if i < parts.len() - 1 {
                format!("{}{}", part, sep)
            } else {
                part.to_string()
            };

            // If this part alone is too big, recursively split it
            if part_with_sep.len() > self.chunk_size {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                }
                let sub_chunks = self.split_recursive(&part_with_sep, remaining_seps);
                chunks.extend(sub_chunks);
                continue;
            }

            // Try to add to current chunk
            if current_chunk.len() + part_with_sep.len() <= self.chunk_size {
                current_chunk.push_str(&part_with_sep);
            } else {
                // Current chunk is full, start a new one
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                }
                current_chunk = part_with_sep;
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        // Add overlap
        self.add_overlap(chunks)
    }

    fn split_by_chars(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut chunks = Vec::new();

        let mut i = 0;
        while i < chars.len() {
            let end = (i + self.chunk_size).min(chars.len());
            let chunk: String = chars[i..end].iter().collect();
            chunks.push(chunk);

            if end >= chars.len() {
                break;
            }

            // Move forward, accounting for overlap
            i += self.chunk_size - self.chunk_overlap;
        }

        chunks
    }

    fn add_overlap(&self, chunks: Vec<String>) -> Vec<String> {
        if self.chunk_overlap == 0 || chunks.len() <= 1 {
            return chunks;
        }

        let mut result = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            if i == 0 {
                result.push(chunk.clone());
                continue;
            }

            // Get overlap from previous chunk
            let prev_chunk = &chunks[i - 1];
            let overlap_chars: Vec<char> = prev_chunk.chars().collect();
            let overlap_start = overlap_chars.len().saturating_sub(self.chunk_overlap);
            let overlap: String = overlap_chars[overlap_start..].iter().collect();

            let new_chunk = format!("{}{}", overlap, chunk);
            result.push(new_chunk);
        }

        result
    }
}

impl TextSplitter for RecursiveCharacterTextSplitter {
    fn split_text(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        if self.chunk_size == 0 {
            return Err(VecStoreError::invalid_parameter(
                "chunk_size",
                "must be greater than 0",
            ));
        }

        if self.chunk_overlap >= self.chunk_size {
            return Err(VecStoreError::invalid_parameter(
                "chunk_overlap",
                "must be less than chunk_size",
            ));
        }

        Ok(self.split_recursive(text, &self.separators))
    }
}

/// Token-based text splitter
///
/// Splits text based on approximate token count rather than character count.
/// Useful for LLM applications with token limits.
///
/// Uses a simple heuristic: ~4 characters per token (approximation for English)
///
/// # Example
///
/// ```no_run
/// use vecstore::text_splitter::{TokenTextSplitter, TextSplitter};
///
/// // Split into ~512 token chunks
/// let splitter = TokenTextSplitter::new(512, 50);
/// let chunks = splitter.split_text("Long document...")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct TokenTextSplitter {
    /// Maximum tokens per chunk
    max_tokens: usize,
    /// Overlap in tokens
    token_overlap: usize,
    /// Characters per token (approximation)
    chars_per_token: usize,
}

impl TokenTextSplitter {
    /// Create a new token-based splitter
    ///
    /// # Arguments
    /// * `max_tokens` - Maximum tokens per chunk
    /// * `token_overlap` - Tokens to overlap between chunks
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vecstore::text_splitter::TokenTextSplitter;
    ///
    /// // 512 token chunks with 50 token overlap
    /// let splitter = TokenTextSplitter::new(512, 50);
    /// ```
    pub fn new(max_tokens: usize, token_overlap: usize) -> Self {
        Self {
            max_tokens,
            token_overlap,
            chars_per_token: 4, // Approximation for English
        }
    }

    /// Set characters per token (default: 4)
    pub fn with_chars_per_token(mut self, chars_per_token: usize) -> Self {
        self.chars_per_token = chars_per_token;
        self
    }
}

impl TextSplitter for TokenTextSplitter {
    fn split_text(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Convert token limits to character limits
        let chunk_size = self.max_tokens * self.chars_per_token;
        let chunk_overlap = self.token_overlap * self.chars_per_token;

        // Use recursive splitter with character-based limits
        let char_splitter = RecursiveCharacterTextSplitter::new(chunk_size, chunk_overlap);
        char_splitter.split_text(text)
    }
}

/// Markdown-aware text splitter
///
/// Splits markdown documents while respecting header hierarchy.
/// **HYBRID**: Simple by default, powerful when needed.
///
/// # Simple Usage (Default)
///
/// ```no_run
/// use vecstore::text_splitter::{MarkdownTextSplitter, TextSplitter};
///
/// // Just works - splits on markdown boundaries
/// let splitter = MarkdownTextSplitter::new(500, 50);
/// let chunks = splitter.split_text("# Title\n\nContent...")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # Advanced Usage (Optional)
///
/// ```no_run
/// use vecstore::text_splitter::{MarkdownTextSplitter, TextSplitter};
///
/// // Preserve header context in each chunk
/// let splitter = MarkdownTextSplitter::new(500, 50)
///     .with_preserve_headers(true);
/// let chunks = splitter.split_text("# Title\n## Section\nContent...")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct MarkdownTextSplitter {
    /// Maximum chunk size in characters
    chunk_size: usize,
    /// Overlap between chunks
    chunk_overlap: usize,
    /// Whether to preserve header hierarchy in chunks
    preserve_headers: bool,
}

impl MarkdownTextSplitter {
    /// Create a new markdown splitter (simple, just works)
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            preserve_headers: false, // Simple by default
        }
    }

    /// Preserve header context in chunks (advanced, opt-in)
    pub fn with_preserve_headers(mut self, preserve: bool) -> Self {
        self.preserve_headers = preserve;
        self
    }

    /// Parse markdown sections with header hierarchy
    fn parse_sections(&self, text: &str) -> Vec<MarkdownSection> {
        let mut sections = Vec::new();
        let mut current_section = MarkdownSection {
            level: 0,
            header: String::new(),
            content: String::new(),
            header_chain: Vec::new(),
        };

        let mut header_stack: Vec<(usize, String)> = Vec::new();

        for line in text.lines() {
            if let Some(level) = self.parse_header_level(line) {
                // Save previous section
                if !current_section.content.is_empty() || !current_section.header.is_empty() {
                    sections.push(current_section.clone());
                }

                // Parse header text
                let header_text = line.trim_start_matches('#').trim().to_string();

                // Update header stack (track hierarchy)
                header_stack.retain(|(l, _)| *l < level);
                header_stack.push((level, header_text.clone()));

                // Start new section
                current_section = MarkdownSection {
                    level,
                    header: header_text,
                    content: String::new(),
                    header_chain: header_stack.iter().map(|(_, h)| h.clone()).collect(),
                };
            } else {
                // Add content line
                if !current_section.content.is_empty() {
                    current_section.content.push('\n');
                }
                current_section.content.push_str(line);
            }
        }

        // Save final section
        if !current_section.content.is_empty() || !current_section.header.is_empty() {
            sections.push(current_section);
        }

        sections
    }

    /// Parse header level from line (e.g., "### Header" -> 3)
    fn parse_header_level(&self, line: &str) -> Option<usize> {
        let trimmed = line.trim_start();
        if !trimmed.starts_with('#') {
            return None;
        }

        let level = trimmed.chars().take_while(|&c| c == '#').count();
        if level > 0 && level <= 6 {
            // Valid markdown header (H1-H6)
            Some(level)
        } else {
            None
        }
    }
}

/// Markdown section with header hierarchy
#[derive(Debug, Clone)]
struct MarkdownSection {
    level: usize,
    header: String,
    content: String,
    header_chain: Vec<String>, // Full hierarchy: ["H1", "H2", "H3"]
}

impl TextSplitter for MarkdownTextSplitter {
    fn split_text(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        if self.chunk_size == 0 {
            return Err(VecStoreError::invalid_parameter(
                "chunk_size",
                "must be greater than 0",
            ));
        }

        // Parse into markdown sections
        let sections = self.parse_sections(text);

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_header_context = String::new();

        for section in sections {
            // Build header context if preserving headers
            if self.preserve_headers && !section.header_chain.is_empty() {
                current_header_context = section
                    .header_chain
                    .iter()
                    .enumerate()
                    .map(|(i, h)| format!("{} {}", "#".repeat(i + 1), h))
                    .collect::<Vec<_>>()
                    .join("\n");
                current_header_context.push_str("\n\n");
            }

            let section_text = if section.header.is_empty() {
                section.content.clone()
            } else {
                format!(
                    "{} {}\n\n{}",
                    "#".repeat(section.level),
                    section.header,
                    section.content
                )
            };

            // If section fits in current chunk, add it
            let chunk_with_section = if self.preserve_headers {
                format!(
                    "{}{}{}",
                    current_chunk, current_header_context, section_text
                )
            } else {
                format!("{}{}", current_chunk, section_text)
            };

            if chunk_with_section.len() <= self.chunk_size {
                current_chunk = chunk_with_section;
            } else {
                // Current chunk is full, save it and start new one
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                }

                // If section itself is too large, split it with RecursiveCharacterTextSplitter
                if section_text.len() > self.chunk_size {
                    let splitter = RecursiveCharacterTextSplitter::new(
                        self.chunk_size.saturating_sub(current_header_context.len()),
                        self.chunk_overlap,
                    );
                    let sub_chunks = splitter.split_text(&section_text)?;

                    for sub_chunk in sub_chunks {
                        if self.preserve_headers && !current_header_context.is_empty() {
                            chunks.push(format!("{}{}", current_header_context, sub_chunk));
                        } else {
                            chunks.push(sub_chunk);
                        }
                    }
                    current_chunk = String::new();
                } else {
                    current_chunk = if self.preserve_headers {
                        format!("{}{}", current_header_context, section_text)
                    } else {
                        section_text
                    };
                }
            }
        }

        // Save final chunk
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        Ok(chunks)
    }
}

/// Code-aware text splitter
///
/// Splits source code while respecting function and class boundaries.
/// **HYBRID**: Simple by default, language-aware when needed.
///
/// # Simple Usage (Default)
///
/// ```no_run
/// use vecstore::text_splitter::{CodeTextSplitter, TextSplitter};
///
/// // Just works - splits on smart boundaries
/// let splitter = CodeTextSplitter::new(800, 50);
/// let chunks = splitter.split_text("fn main() { ... }")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # Advanced Usage (Optional)
///
/// ```no_run
/// use vecstore::text_splitter::{CodeTextSplitter, TextSplitter};
///
/// // Language-specific splitting
/// let splitter = CodeTextSplitter::new(800, 50)
///     .with_language("rust");
/// let chunks = splitter.split_text("fn main() { ... }")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct CodeTextSplitter {
    /// Maximum chunk size in characters
    chunk_size: usize,
    /// Overlap between chunks
    chunk_overlap: usize,
    /// Optional language hint ("rust", "python", "javascript", etc.)
    language: Option<String>,
}

impl CodeTextSplitter {
    /// Create a new code splitter (simple, language-agnostic)
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            language: None, // Simple by default - works for all languages
        }
    }

    /// Set language for smarter splitting (advanced, opt-in)
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Detect if a line starts a code block (function, class, etc.)
    fn is_code_block_start(&self, line: &str) -> bool {
        let trimmed = line.trim_start();

        match self.language.as_deref() {
            Some("rust") => {
                trimmed.starts_with("fn ")
                    || trimmed.starts_with("pub fn ")
                    || trimmed.starts_with("struct ")
                    || trimmed.starts_with("pub struct ")
                    || trimmed.starts_with("enum ")
                    || trimmed.starts_with("pub enum ")
                    || trimmed.starts_with("impl ")
                    || trimmed.starts_with("trait ")
            }
            Some("python") => {
                trimmed.starts_with("def ")
                    || trimmed.starts_with("class ")
                    || trimmed.starts_with("async def ")
            }
            Some("javascript") | Some("typescript") => {
                trimmed.starts_with("function ")
                    || trimmed.starts_with("class ")
                    || trimmed.starts_with("const ")
                    || trimmed.starts_with("let ")
                    || trimmed.starts_with("async function ")
                    || trimmed.starts_with("export ")
            }
            Some("java") | Some("c") | Some("cpp") => {
                // Simple heuristic: look for function-like patterns
                (trimmed.contains('(')
                    && trimmed.contains(')')
                    && (trimmed.contains("public")
                        || trimmed.contains("private")
                        || trimmed.contains("void")
                        || trimmed.contains("int")))
                    || trimmed.starts_with("class ")
            }
            Some("go") => {
                trimmed.starts_with("func ")
                    || trimmed.starts_with("type ")
                    || trimmed.starts_with("struct ")
            }
            _ => {
                // Language-agnostic heuristics
                trimmed.starts_with("fn ")
                    || trimmed.starts_with("function ")
                    || trimmed.starts_with("def ")
                    || trimmed.starts_with("class ")
            }
        }
    }

    /// Get code-specific separators
    fn get_separators(&self) -> Vec<String> {
        vec![
            "\n\n".to_string(),  // Double newline (blank line between functions/blocks)
            "\n}\n".to_string(), // Closing brace (end of block)
            "\n\n".to_string(),  // Paragraphs
            "\n".to_string(),    // Lines
            "; ".to_string(),    // Statements
            " ".to_string(),     // Words
            "".to_string(),      // Characters
        ]
    }
}

impl TextSplitter for CodeTextSplitter {
    fn split_text(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        if self.chunk_size == 0 {
            return Err(VecStoreError::invalid_parameter(
                "chunk_size",
                "must be greater than 0",
            ));
        }

        // Use recursive splitter with code-aware separators
        let separators = self.get_separators();
        let splitter = RecursiveCharacterTextSplitter::new(self.chunk_size, self.chunk_overlap)
            .with_separators(separators);

        // If we have language hints, try to split on code block boundaries first
        if self.language.is_some() {
            let mut chunks = Vec::new();
            let mut current_chunk = String::new();
            let mut current_block = String::new();

            for line in text.lines() {
                let line_with_newline = format!("{}\n", line);

                // Check if this starts a new code block
                if self.is_code_block_start(line) && !current_block.is_empty() {
                    // Save previous block
                    if current_chunk.len() + current_block.len() <= self.chunk_size {
                        current_chunk.push_str(&current_block);
                        current_block.clear();
                    } else {
                        if !current_chunk.is_empty() {
                            chunks.push(current_chunk.clone());
                        }
                        current_chunk = current_block.clone();
                        current_block.clear();
                    }
                }

                current_block.push_str(&line_with_newline);

                // If block is getting too large, flush it
                if current_block.len() > self.chunk_size {
                    if !current_chunk.is_empty() {
                        chunks.push(current_chunk.clone());
                        current_chunk.clear();
                    }

                    // Split oversized block with standard splitter
                    let sub_chunks = splitter.split_text(&current_block)?;
                    chunks.extend(sub_chunks);
                    current_block.clear();
                }
            }

            // Save remaining content
            if !current_block.is_empty() {
                current_chunk.push_str(&current_block);
            }
            if !current_chunk.is_empty() {
                chunks.push(current_chunk);
            }

            return Ok(chunks);
        }

        // Fallback: use standard recursive splitter with code separators
        splitter.split_text(text)
    }
}

/// Simple trait for embedding text (used by SemanticTextSplitter)
///
/// **HYBRID**: Any embedder works - users provide their own implementation.
/// No forced dependencies on specific embedding libraries.
///
/// # Example Implementation
///
/// ```no_run
/// use vecstore::text_splitter::Embedder;
/// use anyhow::Result;
///
/// struct MyEmbedder;
///
/// impl Embedder for MyEmbedder {
///     fn embed(&self, text: &str) -> Result<Vec<f32>> {
///         // Your embedding logic here
///         Ok(vec![0.0; 384]) // Example: 384-dim vector
///     }
/// }
/// ```
pub trait Embedder {
    /// Embed a text into a vector
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Semantic text splitter
///
/// Splits text based on semantic similarity using embeddings.
/// Groups semantically similar content together.
/// **HYBRID**: Requires embedder (advanced), but composable with any embedding model.
///
/// # Usage
///
/// ```no_run
/// use vecstore::text_splitter::{SemanticTextSplitter, TextSplitter, Embedder};
/// use anyhow::Result;
///
/// // Provide your own embedder (no forced dependency)
/// struct MyEmbedder;
/// impl Embedder for MyEmbedder {
///     fn embed(&self, text: &str) -> Result<Vec<f32>> {
///         Ok(vec![0.0; 384])
///     }
/// }
///
/// let embedder = Box::new(MyEmbedder);
/// let splitter = SemanticTextSplitter::new(embedder, 500, 50);
/// let chunks = splitter.split_text("Long document...")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct SemanticTextSplitter {
    /// Embedder for computing semantic similarity
    embedder: Box<dyn Embedder>,
    /// Maximum chunk size in characters
    max_chunk_size: usize,
    /// Minimum chunk size in characters
    min_chunk_size: usize,
    /// Similarity threshold (0.0-1.0) for grouping sentences
    similarity_threshold: f32,
}

impl SemanticTextSplitter {
    /// Create a new semantic splitter
    ///
    /// # Arguments
    /// * `embedder` - Any embedder implementing the Embedder trait (HYBRID: bring your own)
    /// * `max_chunk_size` - Maximum characters per chunk
    /// * `min_chunk_size` - Minimum characters per chunk (avoid tiny chunks)
    pub fn new(embedder: Box<dyn Embedder>, max_chunk_size: usize, min_chunk_size: usize) -> Self {
        Self {
            embedder,
            max_chunk_size,
            min_chunk_size,
            similarity_threshold: 0.7, // Default: group similar content
        }
    }

    /// Set similarity threshold (advanced, opt-in)
    ///
    /// Higher = more similar content required for grouping
    /// Lower = more aggressive grouping
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Split text into sentences (simple heuristic)
    fn split_sentences(&self, text: &str) -> Vec<String> {
        // Simple sentence splitting on common boundaries
        text.split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect()
    }
}

impl TextSplitter for SemanticTextSplitter {
    fn split_text(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        if self.max_chunk_size == 0 {
            return Err(VecStoreError::invalid_parameter(
                "max_chunk_size",
                "must be greater than 0",
            ));
        }

        // Split into sentences
        let sentences = self.split_sentences(text);

        if sentences.is_empty() {
            return Ok(vec![]);
        }

        // Compute embeddings for all sentences
        let mut sentence_embeddings = Vec::new();
        for sentence in &sentences {
            let embedding = self.embedder.embed(sentence)?;
            sentence_embeddings.push(embedding);
        }

        // Group sentences into chunks based on semantic similarity
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_embedding: Option<Vec<f32>> = None;

        for (i, sentence) in sentences.iter().enumerate() {
            let sentence_with_space = if current_chunk.is_empty() {
                sentence.clone()
            } else {
                format!(" {}", sentence)
            };

            // Check if adding this sentence would exceed max size
            if current_chunk.len() + sentence_with_space.len() > self.max_chunk_size {
                // Save current chunk if it meets minimum size
                if current_chunk.len() >= self.min_chunk_size {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                    current_embedding = None;
                }
            }

            // Compute similarity with current chunk
            let should_add = if let Some(ref chunk_emb) = current_embedding {
                let similarity = self.cosine_similarity(chunk_emb, &sentence_embeddings[i]);
                similarity >= self.similarity_threshold
            } else {
                true // First sentence always added
            };

            if should_add || current_chunk.is_empty() {
                // Add sentence to current chunk
                current_chunk.push_str(&sentence_with_space);

                // Update chunk embedding (average of all sentence embeddings)
                if let Some(ref mut chunk_emb) = current_embedding {
                    // Simple averaging (could be weighted)
                    for (j, val) in sentence_embeddings[i].iter().enumerate() {
                        chunk_emb[j] = (chunk_emb[j] + val) / 2.0;
                    }
                } else {
                    current_embedding = Some(sentence_embeddings[i].clone());
                }
            } else {
                // Similarity too low - start new chunk
                if current_chunk.len() >= self.min_chunk_size {
                    chunks.push(current_chunk.clone());
                }
                current_chunk = sentence.clone();
                current_embedding = Some(sentence_embeddings[i].clone());
            }
        }

        // Save final chunk
        if !current_chunk.is_empty() && current_chunk.len() >= self.min_chunk_size {
            chunks.push(current_chunk);
        }

        // Fallback: if no chunks created, use character splitter
        if chunks.is_empty() {
            let fallback =
                RecursiveCharacterTextSplitter::new(self.max_chunk_size, self.min_chunk_size / 2);
            return fallback.split_text(text);
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_splitter_basic() {
        let splitter = RecursiveCharacterTextSplitter::new(20, 0);
        let text = "Short text.";
        let chunks = splitter.split_text(text).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_recursive_splitter_paragraphs() {
        let splitter = RecursiveCharacterTextSplitter::new(50, 0);
        let text = "First paragraph.\n\nSecond paragraph.";
        let chunks = splitter.split_text(text).unwrap();
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_recursive_splitter_overlap() {
        let splitter = RecursiveCharacterTextSplitter::new(20, 5);
        let text = "This is a longer text that should be split into multiple chunks.";
        let chunks = splitter.split_text(text).unwrap();
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_token_splitter() {
        let splitter = TokenTextSplitter::new(10, 2); // 10 tokens ~ 40 chars
        let text = "This is a test. This text should be split based on token count.";
        let chunks = splitter.split_text(text).unwrap();
        assert!(chunks.len() > 0);
    }

    #[test]
    fn test_empty_text() {
        let splitter = RecursiveCharacterTextSplitter::new(100, 10);
        let chunks = splitter.split_text("").unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_invalid_chunk_size() {
        let splitter = RecursiveCharacterTextSplitter::new(0, 0);
        let result = splitter.split_text("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_overlap() {
        let splitter = RecursiveCharacterTextSplitter::new(100, 100);
        let result = splitter.split_text("test");
        assert!(result.is_err());
    }

    // Markdown splitter tests
    #[test]
    fn test_markdown_splitter_basic() {
        let splitter = MarkdownTextSplitter::new(200, 20);
        let text = "# Header 1\n\nSome content here.\n\n## Header 2\n\nMore content.";
        let chunks = splitter.split_text(text).unwrap();
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_markdown_splitter_preserve_headers() {
        let splitter = MarkdownTextSplitter::new(200, 20).with_preserve_headers(true);
        let text = "# Main\n\nContent 1\n\n## Section\n\nContent 2";
        let chunks = splitter.split_text(text).unwrap();

        // When preserving headers, chunks should contain header context
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_markdown_header_parsing() {
        let splitter = MarkdownTextSplitter::new(100, 10);

        // Test various header levels
        assert_eq!(splitter.parse_header_level("# H1"), Some(1));
        assert_eq!(splitter.parse_header_level("## H2"), Some(2));
        assert_eq!(splitter.parse_header_level("### H3"), Some(3));
        assert_eq!(splitter.parse_header_level("Not a header"), None);
        assert_eq!(splitter.parse_header_level("####### Too many"), None);
    }

    #[test]
    fn test_markdown_simple_by_default() {
        // Default behavior: simple splitting without header preservation
        let splitter = MarkdownTextSplitter::new(500, 50);
        assert!(!splitter.preserve_headers);
    }

    // Code splitter tests
    #[test]
    fn test_code_splitter_basic() {
        let splitter = CodeTextSplitter::new(200, 20);
        let code = "fn main() {\n    println!(\"Hello\");\n}\n\nfn test() {\n    // test\n}";
        let chunks = splitter.split_text(code).unwrap();
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_code_splitter_with_language() {
        let splitter = CodeTextSplitter::new(300, 30).with_language("rust");
        let code =
            "fn main() {\n    println!(\"Hello\");\n}\n\nfn test() {\n    println!(\"Test\");\n}";
        let chunks = splitter.split_text(code).unwrap();
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_code_block_detection() {
        let splitter = CodeTextSplitter::new(100, 10).with_language("rust");
        assert!(splitter.is_code_block_start("fn main() {"));
        assert!(splitter.is_code_block_start("pub fn test() {"));
        assert!(splitter.is_code_block_start("struct Foo {"));
        assert!(!splitter.is_code_block_start("    let x = 5;"));
    }

    #[test]
    fn test_code_splitter_simple_by_default() {
        // Default behavior: language-agnostic
        let splitter = CodeTextSplitter::new(500, 50);
        assert!(splitter.language.is_none());
    }

    // Semantic splitter tests (using mock embedder)
    struct MockEmbedder;

    impl Embedder for MockEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            // Simple mock: use text length as "embedding"
            // In real use, this would call an actual embedding model
            let len = text.len() as f32;
            Ok(vec![len / 100.0, len / 50.0, len / 25.0])
        }
    }

    #[test]
    fn test_semantic_splitter_basic() {
        let embedder = Box::new(MockEmbedder);
        let splitter = SemanticTextSplitter::new(embedder, 200, 20);
        let text =
            "First sentence. Second sentence here. Third one is different. Fourth continues.";
        let chunks = splitter.split_text(text).unwrap();
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_semantic_splitter_with_threshold() {
        let embedder = Box::new(MockEmbedder);
        let splitter = SemanticTextSplitter::new(embedder, 300, 30).with_similarity_threshold(0.8);
        let text = "Sentence one. Sentence two. Sentence three.";
        let chunks = splitter.split_text(text).unwrap();
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_semantic_splitter_cosine_similarity() {
        let embedder = Box::new(MockEmbedder);
        let splitter = SemanticTextSplitter::new(embedder, 100, 10);

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let v3 = vec![0.0, 1.0, 0.0];

        // Identical vectors should have similarity 1.0
        let sim1 = splitter.cosine_similarity(&v1, &v2);
        assert!((sim1 - 1.0).abs() < 0.01);

        // Orthogonal vectors should have similarity 0.0
        let sim2 = splitter.cosine_similarity(&v1, &v3);
        assert!(sim2.abs() < 0.01);
    }

    #[test]
    fn test_embedder_trait_composable() {
        // Test that Embedder trait is composable (HYBRID principle)
        struct CustomEmbedder;
        impl Embedder for CustomEmbedder {
            fn embed(&self, _text: &str) -> Result<Vec<f32>> {
                Ok(vec![1.0, 2.0, 3.0])
            }
        }

        let embedder = Box::new(CustomEmbedder);
        let splitter = SemanticTextSplitter::new(embedder, 500, 50);

        let text = "Test text.";
        let result = splitter.split_text(text);
        assert!(result.is_ok());
    }
}
