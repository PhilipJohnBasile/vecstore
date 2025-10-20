//! PDF document loader

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use lopdf::Document as PdfDocument;
use std::path::Path;

/// Loader for PDF files
///
/// Extracts text content from PDF documents using lopdf.
///
/// # Example
///
/// ```no_run
/// use vecstore_loaders::{PdfLoader, DocumentLoader};
///
/// let loader = PdfLoader::new();
/// let document = loader.load("document.pdf")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PdfLoader {
    /// Whether to include page numbers in output
    include_page_numbers: bool,

    /// Page separator in output
    page_separator: String,
}

impl PdfLoader {
    /// Create a new PDF loader
    pub fn new() -> Self {
        Self {
            include_page_numbers: false,
            page_separator: "\n\n".to_string(),
        }
    }

    /// Include page numbers in output
    pub fn with_page_numbers(mut self) -> Self {
        self.include_page_numbers = true;
        self
    }

    /// Set page separator
    pub fn with_page_separator(mut self, separator: impl Into<String>) -> Self {
        self.page_separator = separator.into();
        self
    }

    /// Extract text from PDF document
    fn extract_text(&self, pdf: &PdfDocument) -> Result<String> {
        let mut all_text = Vec::new();
        let pages = pdf.get_pages();

        for (page_num, &(obj_id, _gen_id)) in pages.iter() {
            if let Ok(text) = pdf.extract_text(&[obj_id]) {
                let page_text = if self.include_page_numbers {
                    format!("--- Page {} ---\n{}", page_num, text)
                } else {
                    text
                };

                all_text.push(page_text);
            }
        }

        Ok(all_text.join(&self.page_separator))
    }
}

impl Default for PdfLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for PdfLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let path = Path::new(source);

        if !path.exists() {
            return Err(LoaderError::InvalidPath(format!("File not found: {}", source)));
        }

        if !path.is_file() {
            return Err(LoaderError::InvalidPath(format!("{} is not a file", source)));
        }

        // Load PDF
        let pdf = PdfDocument::load(path)?;

        // Extract text
        let content = self.extract_text(&pdf)?;

        let mut document = Document::new(content, source.to_string());

        // Add metadata
        document.add_metadata("format", "pdf");
        document.add_metadata("page_count", pdf.get_pages().len().to_string());

        // Extract PDF metadata if available
        if let Ok(info) = pdf.trailer.get(b"Info") {
            if let Ok(info_dict) = info.as_dict() {
                // Try to extract title
                if let Ok(title) = info_dict.get(b"Title") {
                    if let Ok(title_bytes) = title.as_str() {
                        let title_str = String::from_utf8_lossy(title_bytes).to_string();
                        document.add_metadata("title", title_str);
                    }
                }

                // Try to extract author
                if let Ok(author) = info_dict.get(b"Author") {
                    if let Ok(author_bytes) = author.as_str() {
                        let author_str = String::from_utf8_lossy(author_bytes).to_string();
                        document.add_metadata("author", author_str);
                    }
                }

                // Try to extract subject
                if let Ok(subject) = info_dict.get(b"Subject") {
                    if let Ok(subject_bytes) = subject.as_str() {
                        let subject_str = String::from_utf8_lossy(subject_bytes).to_string();
                        document.add_metadata("subject", subject_str);
                    }
                }

                // Try to extract creation date
                if let Ok(creation_date) = info_dict.get(b"CreationDate") {
                    if let Ok(date_bytes) = creation_date.as_str() {
                        let date_str = String::from_utf8_lossy(date_bytes).to_string();
                        document.add_metadata("creation_date", date_str);
                    }
                }
            }
        }

        Ok(document)
    }

    fn load_with_options(&self, source: &str, options: &LoaderOptions) -> Result<Document> {
        // Check file size if max_size is set
        if let Some(max_size) = options.max_size {
            let metadata = std::fs::metadata(source)?;
            let file_size = metadata.len() as usize;

            if file_size > max_size {
                return Err(LoaderError::FileTooLarge(file_size, max_size));
            }
        }

        self.load(source)
    }

    fn name(&self) -> &str {
        "PdfLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_loader_creation() {
        let loader = PdfLoader::new();
        assert_eq!(loader.name(), "PdfLoader");
        assert!(loader.supported_extensions().contains(&"pdf"));
    }

    #[test]
    fn test_pdf_loader_with_page_numbers() {
        let loader = PdfLoader::new().with_page_numbers();
        assert!(loader.include_page_numbers);
    }

    #[test]
    fn test_pdf_loader_custom_separator() {
        let loader = PdfLoader::new().with_page_separator("\n---\n");
        assert_eq!(loader.page_separator, "\n---\n");
    }

    // Note: Actual PDF loading tests require sample PDF files
    // These would be added in integration tests with test fixtures
}
