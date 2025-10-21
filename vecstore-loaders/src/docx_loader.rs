///! DOCX (Microsoft Word) document loader
///!
///! Extracts text content from .docx files using the docx-rs crate.

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Loader for Microsoft Word (.docx) documents
///
/// # Example
/// ```no_run
/// use vecstore_loaders::{DocxLoader, DocumentLoader};
///
/// let loader = DocxLoader::new();
/// let document = loader.load("document.docx")?;
/// println!("Loaded {} characters", document.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct DocxLoader {
    /// Whether to extract metadata (title, author, etc.)
    extract_metadata: bool,
}

impl DocxLoader {
    /// Create a new DOCX loader
    pub fn new() -> Self {
        Self {
            extract_metadata: false,
        }
    }

    /// Enable metadata extraction
    pub fn with_metadata(mut self) -> Self {
        self.extract_metadata = true;
        self
    }

    /// Extract text from DOCX file
    fn extract_text(&self, path: &Path) -> Result<String> {
        // Read the entire file into memory
        let mut file = File::open(path)
            .map_err(|e| LoaderError::Io(e))?;

        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut buf)
            .map_err(|e| LoaderError::Io(e))?;

        // Read the DOCX file using docx-rs
        let docx = docx_rs::read_docx(&buf)
            .map_err(|e| LoaderError::ParseError(format!("Failed to parse DOCX: {:?}", e)))?;

        // Extract all paragraphs as text
        let mut text_parts = Vec::new();

        for child in docx.document.children {
            match child {
                docx_rs::DocumentChild::Paragraph(para) => {
                    let mut para_text = String::new();
                    for child in para.children {
                        if let docx_rs::ParagraphChild::Run(run) = child {
                            for child in run.children {
                                if let docx_rs::RunChild::Text(text) = child {
                                    para_text.push_str(&text.text);
                                }
                            }
                        }
                    }
                    if !para_text.is_empty() {
                        text_parts.push(para_text);
                    }
                }
                docx_rs::DocumentChild::Table(table) => {
                    // Extract text from table rows
                    for row_child in &table.rows {
                        let docx_rs::TableChild::TableRow(_row) = row_child;
                        // docx-rs table structure is complex, just skip tables for now
                        // or implement simple extraction
                        text_parts.push("[Table content]".to_string());
                    }
                }
                _ => {}
            }
        }

        Ok(text_parts.join("\n\n"))
    }

    /// Extract metadata from DOCX file
    fn extract_docx_metadata(&self, _path: &Path) -> HashMap<String, String> {
        // TODO: Extract core properties (title, author, subject, etc.)
        // This requires parsing the docProps/core.xml file from the ZIP
        HashMap::new()
    }
}

impl Default for DocxLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for DocxLoader {
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
            if ext.to_string_lossy().to_lowercase() != "docx" {
                return Err(LoaderError::UnsupportedFormat(
                    format!("Expected .docx file, got .{}", ext.to_string_lossy())
                ));
            }
        } else {
            return Err(LoaderError::UnsupportedFormat("No file extension".to_string()));
        }

        let content = self.extract_text(path)?;

        let mut document = Document::new(content, source.to_string());

        if self.extract_metadata {
            let metadata = self.extract_docx_metadata(path);
            document.metadata = metadata;
        }

        document.add_metadata("format", "docx");
        document.add_metadata("type", "document");

        Ok(document)
    }

    fn load_with_options(&self, source: &str, options: &LoaderOptions) -> Result<Document> {
        let mut loader = Self::new();

        if options.include_metadata {
            loader = loader.with_metadata();
        }

        loader.load(source)
    }

    fn name(&self) -> &str {
        "DocxLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["docx"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docx_loader_creation() {
        let loader = DocxLoader::new();
        assert_eq!(loader.name(), "DocxLoader");
        assert_eq!(loader.supported_extensions(), &["docx"]);
    }

    #[test]
    fn test_docx_loader_with_metadata() {
        let loader = DocxLoader::new().with_metadata();
        assert!(loader.extract_metadata);
    }
}
