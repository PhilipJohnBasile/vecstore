///! PPTX (Microsoft PowerPoint) document loader
///!
///! Extracts text content from .pptx files by parsing the XML structure inside the ZIP archive.

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use zip::ZipArchive;

/// Loader for Microsoft PowerPoint (.pptx) documents
///
/// # Example
/// ```no_run
/// use vecstore_loaders::{PptxLoader, DocumentLoader};
///
/// let loader = PptxLoader::new();
/// let document = loader.load("presentation.pptx")?;
/// println!("Loaded {} characters from {} slides", document.len(), document.metadata.get("slides").unwrap_or(&"?".to_string()));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PptxLoader {
    /// Whether to include slide numbers in output
    include_slide_numbers: bool,
    /// Whether to extract metadata
    extract_metadata: bool,
}

impl PptxLoader {
    /// Create a new PPTX loader
    pub fn new() -> Self {
        Self {
            include_slide_numbers: false,
            extract_metadata: false,
        }
    }

    /// Include slide numbers in the extracted text
    pub fn with_slide_numbers(mut self) -> Self {
        self.include_slide_numbers = true;
        self
    }

    /// Enable metadata extraction
    pub fn with_metadata(mut self) -> Self {
        self.extract_metadata = true;
        self
    }

    /// Extract text from a single slide XML
    fn extract_slide_text(&self, xml_content: &str) -> Result<String> {
        let mut reader = Reader::from_str(xml_content);
        reader.config_mut().trim_text(true);

        let mut texts = Vec::new();
        let mut buf = Vec::new();
        let mut inside_text = false;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    // Look for <a:t> tags which contain text runs
                    if e.name().as_ref() == b"a:t" {
                        inside_text = true;
                    }
                }
                Ok(Event::Text(e)) => {
                    if inside_text {
                        let text = e.unescape().unwrap_or_default();
                        if !text.is_empty() {
                            texts.push(text.to_string());
                        }
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name().as_ref() == b"a:t" {
                        inside_text = false;
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => {
                    return Err(LoaderError::ParseError(format!("XML parsing error: {}", e)));
                }
                _ => {}
            }
            buf.clear();
        }

        Ok(texts.join(" "))
    }

    /// Extract text from all slides in PPTX
    fn extract_text(&self, path: &Path) -> Result<(String, usize)> {
        let file = File::open(path)
            .map_err(|e| LoaderError::Io(e))?;

        let reader = BufReader::new(file);
        let mut archive = ZipArchive::new(reader)
            .map_err(|e| LoaderError::ParseError(format!("Failed to open PPTX ZIP: {}", e)))?;

        let mut slide_texts = Vec::new();
        let mut slide_count = 0;

        // Iterate through all files in the ZIP
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| LoaderError::ParseError(format!("Failed to read ZIP entry: {}", e)))?;

            let name = file.name().to_string();

            // Look for slide XML files
            if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
                slide_count += 1;

                let mut xml_content = String::new();
                file.read_to_string(&mut xml_content)
                    .map_err(|e| LoaderError::Io(e))?;

                let slide_text = self.extract_slide_text(&xml_content)?;

                if !slide_text.is_empty() {
                    if self.include_slide_numbers {
                        slide_texts.push(format!("--- Slide {} ---\n{}", slide_count, slide_text));
                    } else {
                        slide_texts.push(slide_text);
                    }
                }
            }
        }

        let full_text = slide_texts.join("\n\n");
        Ok((full_text, slide_count))
    }
}

impl Default for PptxLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for PptxLoader {
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
            if ext_lower != "pptx" {
                return Err(LoaderError::UnsupportedFormat(
                    format!("Expected .pptx file, got .{}", ext.to_string_lossy())
                ));
            }
        } else {
            return Err(LoaderError::UnsupportedFormat("No file extension".to_string()));
        }

        let (content, slide_count) = self.extract_text(path)?;

        let mut document = Document::new(content, source.to_string());

        document.add_metadata("format", "pptx");
        document.add_metadata("type", "presentation");
        document.add_metadata("slides", &slide_count.to_string());

        Ok(document)
    }

    fn load_with_options(&self, source: &str, options: &LoaderOptions) -> Result<Document> {
        let mut loader = Self::new();

        if options.include_metadata {
            loader = loader.with_metadata();
        }

        // Check for custom option to include slide numbers
        if let Some(val) = options.custom.get("include_slide_numbers") {
            if val == "true" {
                loader = loader.with_slide_numbers();
            }
        }

        loader.load(source)
    }

    fn name(&self) -> &str {
        "PptxLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["pptx"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pptx_loader_creation() {
        let loader = PptxLoader::new();
        assert_eq!(loader.name(), "PptxLoader");
        assert_eq!(loader.supported_extensions(), &["pptx"]);
    }

    #[test]
    fn test_pptx_loader_with_options() {
        let loader = PptxLoader::new()
            .with_slide_numbers()
            .with_metadata();

        assert!(loader.include_slide_numbers);
        assert!(loader.extract_metadata);
    }
}
