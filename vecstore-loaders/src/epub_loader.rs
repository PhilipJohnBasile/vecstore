///! EPUB (Electronic Publication) document loader
///!
///! Extracts text content from .epub files using the epub crate.

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use epub::doc::EpubDoc;
use std::collections::HashMap;
use std::path::Path;

/// Loader for EPUB (Electronic Publication) files
///
/// # Example
/// ```no_run
/// use vecstore_loaders::{EpubLoader, DocumentLoader};
///
/// let loader = EpubLoader::new();
/// let document = loader.load("book.epub")?;
/// println!("Loaded {} characters", document.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct EpubLoader {
    /// Whether to extract metadata (title, author, etc.)
    extract_metadata: bool,
    /// Whether to include chapter markers
    include_chapters: bool,
}

impl EpubLoader {
    /// Create a new EPUB loader
    pub fn new() -> Self {
        Self {
            extract_metadata: false,
            include_chapters: false,
        }
    }

    /// Enable metadata extraction (title, author, etc.)
    pub fn with_metadata(mut self) -> Self {
        self.extract_metadata = true;
        self
    }

    /// Include chapter markers in the output
    pub fn with_chapters(mut self) -> Self {
        self.include_chapters = true;
        self
    }

    /// Extract text from EPUB file
    fn extract_text(&self, path: &Path) -> Result<(String, HashMap<String, String>)> {
        let mut doc = EpubDoc::new(path)
            .map_err(|e| LoaderError::ParseError(format!("Failed to open EPUB: {}", e)))?;

        let mut texts = Vec::new();
        let mut metadata = HashMap::new();

        // Extract metadata if requested
        if self.extract_metadata {
            if let Some(title) = doc.mdata("title") {
                metadata.insert("title".to_string(), title);
            }
            if let Some(creator) = doc.mdata("creator") {
                metadata.insert("author".to_string(), creator);
            }
            if let Some(publisher) = doc.mdata("publisher") {
                metadata.insert("publisher".to_string(), publisher);
            }
            if let Some(date) = doc.mdata("date") {
                metadata.insert("date".to_string(), date);
            }
            if let Some(language) = doc.mdata("language") {
                metadata.insert("language".to_string(), language);
            }
        }

        // Extract text from each page/spine item
        let spine_len = doc.spine.len();
        metadata.insert("pages".to_string(), spine_len.to_string());

        for (chapter_num, _) in doc.spine.clone().iter().enumerate() {
            if let Some((content, _mime)) = doc.get_current_str() {
                // Strip HTML tags to get plain text
                let plain_text = self.strip_html(&content);

                if !plain_text.trim().is_empty() {
                    if self.include_chapters {
                        texts.push(format!("=== Chapter {} ===\n{}", chapter_num + 1, plain_text));
                    } else {
                        texts.push(plain_text);
                    }
                }
            }

            // Move to next page
            doc.go_next();
        }

        let full_text = texts.join("\n\n");
        Ok((full_text, metadata))
    }

    /// Strip HTML tags to get plain text
    fn strip_html(&self, html: &str) -> String {
        // Basic HTML tag removal
        let mut result = String::new();
        let mut inside_tag = false;
        let mut inside_script_or_style = false;
        let mut tag_name = String::new();

        for ch in html.chars() {
            match ch {
                '<' => {
                    inside_tag = true;
                    tag_name.clear();
                }
                '>' => {
                    inside_tag = false;
                    // Check if we're entering/exiting script or style tags
                    let tag_lower = tag_name.to_lowercase();
                    if tag_lower == "script" || tag_lower == "style" {
                        inside_script_or_style = true;
                    } else if tag_lower == "/script" || tag_lower == "/style" {
                        inside_script_or_style = false;
                    }
                }
                _ => {
                    if inside_tag {
                        tag_name.push(ch);
                    } else if !inside_script_or_style {
                        result.push(ch);
                    }
                }
            }
        }

        // Clean up whitespace
        result
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for EpubLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for EpubLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let path = Path::new(source);

        if !path.exists() {
            return Err(LoaderError::InvalidPath(format!("File not found: {}", source)));
        }

        if !path.is_file() {
            return Err(LoaderError::InvalidPath(format!("{} is not a file", source)));
        }

        // Check file extension
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            if ext_lower != "epub" {
                return Err(LoaderError::UnsupportedFormat(
                    format!("Expected .epub file, got .{}", ext.to_string_lossy())
                ));
            }
        } else {
            return Err(LoaderError::UnsupportedFormat("No file extension".to_string()));
        }

        let (content, mut extracted_metadata) = self.extract_text(path)?;

        let mut document = Document::new(content, source.to_string());

        // Add extracted metadata
        if self.extract_metadata {
            document.metadata.extend(extracted_metadata.clone());
        }

        // Always add format metadata
        document.add_metadata("format", "epub");
        document.add_metadata("type", "book");

        // Add pages count if available
        if let Some(pages) = extracted_metadata.remove("pages") {
            document.add_metadata("pages", &pages);
        }

        Ok(document)
    }

    fn load_with_options(&self, source: &str, options: &LoaderOptions) -> Result<Document> {
        let mut loader = Self::new();

        if options.include_metadata {
            loader = loader.with_metadata();
        }

        // Check for custom option to include chapters
        if let Some(val) = options.custom.get("include_chapters") {
            if val == "true" {
                loader = loader.with_chapters();
            }
        }

        loader.load(source)
    }

    fn name(&self) -> &str {
        "EpubLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["epub"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epub_loader_creation() {
        let loader = EpubLoader::new();
        assert_eq!(loader.name(), "EpubLoader");
        assert_eq!(loader.supported_extensions(), &["epub"]);
    }

    #[test]
    fn test_epub_loader_with_options() {
        let loader = EpubLoader::new()
            .with_metadata()
            .with_chapters();

        assert!(loader.extract_metadata);
        assert!(loader.include_chapters);
    }

    #[test]
    fn test_html_stripping() {
        let loader = EpubLoader::new();
        let html = "<p>Hello <b>world</b>!</p><script>alert('test');</script><p>More text</p>";
        let plain = loader.strip_html(html);
        assert_eq!(plain, "Hello world! More text");
    }

    #[test]
    fn test_html_stripping_with_style() {
        let loader = EpubLoader::new();
        let html = "<div>Text<style>body { color: red; }</style>More</div>";
        let plain = loader.strip_html(html);
        assert_eq!(plain, "Text More");
    }
}
