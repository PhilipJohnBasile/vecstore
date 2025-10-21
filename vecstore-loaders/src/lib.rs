//! # vecstore-loaders
//!
//! Document loaders for VecStore - load documents from various formats for RAG applications.
//!
//! This companion crate provides loaders for common document formats:
//! - Plain text files
//! - Markdown
//! - PDF documents
//! - Web pages
//! - JSON/CSV data
//! - Source code (syntax-aware)
//!
//! ## Philosophy
//!
//! Following VecStore's hybrid approach, loaders are:
//! - **Simple** - Easy to use with sensible defaults
//! - **Flexible** - Customizable when needed
//! - **Optional** - Feature-gated, only pay for what you use
//!
//! ## Quick Start
//!
//! ```no_run
//! use vecstore_loaders::{TextLoader, DocumentLoader};
//!
//! let loader = TextLoader::new();
//! let document = loader.load("document.txt")?;
//!
//! println!("Loaded {} characters from {}", document.content.len(), document.source);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Feature Flags
//!
//! - `text` - Plain text loader (enabled by default)
//! - `markdown` - Markdown loader with pulldown-cmark (enabled by default)
//! - `pdf` - PDF loader with lopdf
//! - `web` - Web scraping with reqwest + scraper
//! - `json` - JSON loader (enabled by default)
//! - `csv` - CSV loader (enabled by default)
//! - `code` - Syntax-aware code loader with tree-sitter
//! - `all` - Enable all loaders

use std::collections::HashMap;
use std::path::Path;

mod error;
pub use error::{LoaderError, Result};

#[cfg(feature = "text")]
mod text;
#[cfg(feature = "text")]
pub use text::TextLoader;

#[cfg(feature = "markdown")]
mod markdown;
#[cfg(feature = "markdown")]
pub use markdown::MarkdownLoader;

#[cfg(feature = "pdf")]
mod pdf;
#[cfg(feature = "pdf")]
pub use pdf::PdfLoader;

#[cfg(feature = "web")]
mod web;
#[cfg(feature = "web")]
pub use web::WebLoader;

#[cfg(feature = "json")]
mod json_loader;
#[cfg(feature = "json")]
pub use json_loader::JsonLoader;

#[cfg(feature = "csv")]
mod csv_loader;
#[cfg(feature = "csv")]
pub use csv_loader::CsvLoader;

#[cfg(feature = "code")]
mod code;
#[cfg(feature = "code")]
pub use code::CodeLoader;

#[cfg(feature = "docx")]
mod docx_loader;
#[cfg(feature = "docx")]
pub use docx_loader::DocxLoader;

#[cfg(feature = "pptx")]
mod pptx_loader;
#[cfg(feature = "pptx")]
pub use pptx_loader::PptxLoader;

#[cfg(feature = "epub")]
mod epub_loader;
#[cfg(feature = "epub")]
pub use epub_loader::EpubLoader;

// Extended loaders
#[cfg(feature = "extended")]
mod extended_loaders;

#[cfg(feature = "extended")]
pub use extended_loaders::{
    XlsxLoader, OdsLoader, RtfLoader, LatexLoader, XmlLoader,
    YamlLoader, TomlLoader, SqlLoader, EmlLoader, JupyterLoader,
    ArchiveLoader,
};

/// Represents a loaded document with content and metadata
#[derive(Debug, Clone)]
pub struct Document {
    /// The text content of the document
    pub content: String,

    /// Source identifier (file path, URL, etc.)
    pub source: String,

    /// Optional metadata about the document
    pub metadata: HashMap<String, String>,
}

impl Document {
    /// Create a new document
    pub fn new(content: String, source: String) -> Self {
        Self {
            content,
            source,
            metadata: HashMap::new(),
        }
    }

    /// Create a document with metadata
    pub fn with_metadata(content: String, source: String, metadata: HashMap<String, String>) -> Self {
        Self {
            content,
            source,
            metadata,
        }
    }

    /// Add metadata key-value pair
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get the content length in characters
    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Check if the document is empty
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

/// Trait for loading documents from various sources
///
/// This trait provides a common interface for all document loaders.
/// Implementations handle format-specific parsing and extract text content.
///
/// # Example
///
/// ```no_run
/// use vecstore_loaders::{TextLoader, DocumentLoader};
///
/// let loader = TextLoader::new();
/// let document = loader.load("document.txt")?;
/// println!("Loaded: {}", document.content);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait DocumentLoader {
    /// Load a document from a file path
    fn load(&self, source: &str) -> Result<Document>;

    /// Load a document from a file path with custom options
    ///
    /// Default implementation calls `load()`. Override to provide options.
    fn load_with_options(&self, source: &str, _options: &LoaderOptions) -> Result<Document> {
        self.load(source)
    }

    /// Load multiple documents from a directory
    ///
    /// Default implementation loads all files with supported extensions.
    fn load_directory(&self, dir_path: &str) -> Result<Vec<Document>> {
        let path = Path::new(dir_path);
        if !path.is_dir() {
            return Err(LoaderError::InvalidPath(format!("{} is not a directory", dir_path)));
        }

        let mut documents = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(path_str) = path.to_str() {
                    // Try to load, skip files that can't be loaded
                    if let Ok(doc) = self.load(path_str) {
                        documents.push(doc);
                    }
                }
            }
        }

        Ok(documents)
    }

    /// Get the name of this loader
    fn name(&self) -> &str;

    /// Get supported file extensions (e.g., ["txt", "text"])
    fn supported_extensions(&self) -> &[&str];
}

/// Options for configuring document loading
#[derive(Debug, Clone, Default)]
pub struct LoaderOptions {
    /// Text encoding (default: UTF-8)
    pub encoding: Option<String>,

    /// Maximum file size in bytes (None = unlimited)
    pub max_size: Option<usize>,

    /// Whether to include metadata
    pub include_metadata: bool,

    /// Custom loader-specific options
    pub custom: HashMap<String, String>,
}

impl LoaderOptions {
    /// Create new loader options with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set encoding
    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = Some(encoding.into());
        self
    }

    /// Set maximum file size
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = Some(max_size);
        self
    }

    /// Enable metadata inclusion
    pub fn with_metadata(mut self) -> Self {
        self.include_metadata = true;
        self
    }

    /// Add custom option
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document::new("Hello world".to_string(), "test.txt".to_string());
        assert_eq!(doc.content, "Hello world");
        assert_eq!(doc.source, "test.txt");
        assert_eq!(doc.len(), 11);
        assert!(!doc.is_empty());
    }

    #[test]
    fn test_document_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("author".to_string(), "Test".to_string());

        let mut doc = Document::with_metadata(
            "Content".to_string(),
            "source.txt".to_string(),
            metadata,
        );

        doc.add_metadata("title", "Test Document");
        assert_eq!(doc.metadata.get("author"), Some(&"Test".to_string()));
        assert_eq!(doc.metadata.get("title"), Some(&"Test Document".to_string()));
    }

    #[test]
    fn test_loader_options() {
        let options = LoaderOptions::new()
            .with_encoding("utf-8")
            .with_max_size(1024 * 1024)
            .with_metadata()
            .with_custom("key", "value");

        assert_eq!(options.encoding, Some("utf-8".to_string()));
        assert_eq!(options.max_size, Some(1024 * 1024));
        assert!(options.include_metadata);
        assert_eq!(options.custom.get("key"), Some(&"value".to_string()));
    }
}
