//! Plain text document loader

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use std::fs;
use std::path::Path;

#[cfg(feature = "text")]
use encoding_rs::Encoding;

/// Loader for plain text files
///
/// Supports multiple text encodings and handles common text file formats.
///
/// # Example
///
/// ```no_run
/// use vecstore_loaders::{TextLoader, DocumentLoader};
///
/// let loader = TextLoader::new();
/// let document = loader.load("document.txt")?;
/// println!("Loaded {} characters", document.content.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TextLoader {
    default_encoding: String,
}

impl TextLoader {
    /// Create a new text loader with UTF-8 encoding
    pub fn new() -> Self {
        Self {
            default_encoding: "utf-8".to_string(),
        }
    }

    /// Create a text loader with specific encoding
    pub fn with_encoding(encoding: impl Into<String>) -> Self {
        Self {
            default_encoding: encoding.into(),
        }
    }

    /// Load text with encoding detection
    #[cfg(feature = "text")]
    fn load_with_encoding(&self, path: &Path, encoding_name: &str) -> Result<String> {
        let bytes = fs::read(path)?;

        let encoding = Encoding::for_label(encoding_name.as_bytes())
            .unwrap_or(encoding_rs::UTF_8);

        let (content, _encoding, _had_errors) = encoding.decode(&bytes);

        // Note: Encoding errors are gracefully handled by encoding_rs
        // Future: could add optional logging

        Ok(content.into_owned())
    }

    /// Load text assuming UTF-8 (fallback)
    #[cfg(not(feature = "text"))]
    fn load_with_encoding(&self, path: &Path, _encoding_name: &str) -> Result<String> {
        Ok(fs::read_to_string(path)?)
    }
}

impl Default for TextLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for TextLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let path = Path::new(source);

        if !path.exists() {
            return Err(LoaderError::InvalidPath(format!("File not found: {}", source)));
        }

        if !path.is_file() {
            return Err(LoaderError::InvalidPath(format!("{} is not a file", source)));
        }

        let content = self.load_with_encoding(path, &self.default_encoding)?;

        let mut document = Document::new(content, source.to_string());

        // Add metadata
        if let Ok(metadata) = fs::metadata(path) {
            document.add_metadata("file_size", metadata.len().to_string());
            if let Ok(modified) = metadata.modified() {
                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                    document.add_metadata("modified_timestamp", duration.as_secs().to_string());
                }
            }
        }

        if let Some(extension) = path.extension() {
            document.add_metadata("extension", extension.to_string_lossy().to_string());
        }

        Ok(document)
    }

    fn load_with_options(&self, source: &str, options: &LoaderOptions) -> Result<Document> {
        let path = Path::new(source);

        // Check file size if max_size is set
        if let Some(max_size) = options.max_size {
            let metadata = fs::metadata(path)?;
            let file_size = metadata.len() as usize;

            if file_size > max_size {
                return Err(LoaderError::FileTooLarge(file_size, max_size));
            }
        }

        // Use custom encoding if provided
        let encoding = options.encoding.as_deref().unwrap_or(&self.default_encoding);

        let content = self.load_with_encoding(path, encoding)?;

        let mut document = Document::new(content, source.to_string());

        // Add metadata if requested
        if options.include_metadata {
            if let Ok(metadata) = fs::metadata(path) {
                document.add_metadata("file_size", metadata.len().to_string());
                if let Ok(modified) = metadata.modified() {
                    if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                        document.add_metadata("modified_timestamp", duration.as_secs().to_string());
                    }
                }
            }

            if let Some(extension) = path.extension() {
                document.add_metadata("extension", extension.to_string_lossy().to_string());
            }

            document.add_metadata("encoding", encoding.to_string());
        }

        Ok(document)
    }

    fn name(&self) -> &str {
        "TextLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["txt", "text", "log", "md", "rst", "yaml", "yml", "toml", "ini", "cfg"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_text_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Hello, world!").unwrap();
        writeln!(temp_file, "This is a test.").unwrap();

        let loader = TextLoader::new();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("Hello, world!"));
        assert!(document.content.contains("This is a test."));
        assert!(!document.is_empty());
    }

    #[test]
    fn test_load_with_metadata() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Test content").unwrap();

        let loader = TextLoader::new();
        let options = LoaderOptions::new().with_metadata();

        let document = loader
            .load_with_options(temp_file.path().to_str().unwrap(), &options)
            .unwrap();

        assert!(document.metadata.contains_key("file_size"));
        assert!(document.metadata.contains_key("encoding"));
    }

    #[test]
    fn test_file_not_found() {
        let loader = TextLoader::new();
        let result = loader.load("/path/to/nonexistent/file.txt");

        assert!(result.is_err());
        match result {
            Err(LoaderError::InvalidPath(_)) => {}
            _ => panic!("Expected InvalidPath error"),
        }
    }

    #[test]
    fn test_max_size_limit() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "This is some test content that exceeds the limit").unwrap();

        let loader = TextLoader::new();
        let options = LoaderOptions::new().with_max_size(10); // Very small limit

        let result = loader.load_with_options(temp_file.path().to_str().unwrap(), &options);

        assert!(result.is_err());
        match result {
            Err(LoaderError::FileTooLarge(_, _)) => {}
            _ => panic!("Expected FileTooLarge error"),
        }
    }

    #[test]
    fn test_supported_extensions() {
        let loader = TextLoader::new();
        let extensions = loader.supported_extensions();

        assert!(extensions.contains(&"txt"));
        assert!(extensions.contains(&"md"));
        assert!(extensions.contains(&"log"));
    }
}
