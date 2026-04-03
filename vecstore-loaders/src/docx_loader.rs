///! DOCX (Microsoft Word) document loader
///!
///! Extracts text content and metadata from .docx files using the docx-rs crate.
///! Metadata extraction requires the `zip` and `quick-xml` crates.

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
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

    /// Extract text from a paragraph, handling nested runs
    fn extract_paragraph_text(para: &docx_rs::Paragraph) -> String {
        let mut para_text = String::new();
        for child in &para.children {
            if let docx_rs::ParagraphChild::Run(run) = child {
                for run_child in &run.children {
                    if let docx_rs::RunChild::Text(text) = run_child {
                        para_text.push_str(&text.text);
                    }
                }
            }
        }
        para_text
    }

    /// Extract text from DOCX file
    fn extract_text(&self, path: &Path) -> Result<String> {
        // Read the entire file into memory
        let mut file = File::open(path).map_err(LoaderError::Io)?;

        let mut buf = Vec::new();
        file.read_to_end(&mut buf).map_err(LoaderError::Io)?;

        // Read the DOCX file using docx-rs
        let docx = docx_rs::read_docx(&buf)
            .map_err(|e| LoaderError::ParseError(format!("Failed to parse DOCX: {:?}", e)))?;

        // Extract all paragraphs as text
        let mut text_parts = Vec::new();

        for child in &docx.document.children {
            match child {
                docx_rs::DocumentChild::Paragraph(para) => {
                    let para_text = Self::extract_paragraph_text(para);
                    if !para_text.is_empty() {
                        text_parts.push(para_text);
                    }
                }
                docx_rs::DocumentChild::Table(table) => {
                    // Extract text from table cells
                    let table_text = self.extract_table_text(table);
                    if !table_text.is_empty() {
                        text_parts.push(table_text);
                    }
                }
                _ => {}
            }
        }

        Ok(text_parts.join("\n\n"))
    }

    /// Extract text from a DOCX table
    fn extract_table_text(&self, table: &docx_rs::Table) -> String {
        let mut rows_text = Vec::new();

        for row_child in &table.rows {
            let docx_rs::TableChild::TableRow(row) = row_child;
            let mut cells_text = Vec::new();

            for cell_child in &row.cells {
                let docx_rs::TableRowChild::TableCell(cell) = cell_child;
                let mut cell_text_parts = Vec::new();

                for content in &cell.children {
                    if let docx_rs::TableCellContent::Paragraph(para) = content {
                        let para_text = Self::extract_paragraph_text(para);
                        if !para_text.is_empty() {
                            cell_text_parts.push(para_text);
                        }
                    }
                }

                cells_text.push(cell_text_parts.join(" "));
            }

            if !cells_text.is_empty() {
                rows_text.push(cells_text.join("\t"));
            }
        }

        rows_text.join("\n")
    }

    /// Extract metadata from DOCX file by reading docProps/core.xml
    fn extract_docx_metadata(&self, path: &Path) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        // Open DOCX as ZIP archive
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return metadata,
        };

        let mut archive = match zip::ZipArchive::new(file) {
            Ok(a) => a,
            Err(_) => return metadata,
        };

        // Read docProps/core.xml for Dublin Core metadata
        if let Ok(mut core_xml) = archive.by_name("docProps/core.xml") {
            let mut contents = String::new();
            if core_xml.read_to_string(&mut contents).is_ok() {
                self.parse_core_xml(&contents, &mut metadata);
            }
        }

        // Read docProps/app.xml for application metadata
        if let Ok(mut app_xml) = archive.by_name("docProps/app.xml") {
            let mut contents = String::new();
            if app_xml.read_to_string(&mut contents).is_ok() {
                self.parse_app_xml(&contents, &mut metadata);
            }
        }

        metadata
    }

    /// Parse core.xml (Dublin Core metadata)
    fn parse_core_xml(&self, xml: &str, metadata: &mut HashMap<String, String>) {
        use quick_xml::events::Event;
        use quick_xml::Reader;

        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut current_element = String::new();
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    let name = String::from_utf8_lossy(e.local_name().as_ref()).to_string();
                    current_element = name;
                }
                Ok(Event::Text(ref e)) => {
                    let text = e.unescape().map(|s| s.to_string()).unwrap_or_default();
                    if !text.is_empty() && !current_element.is_empty() {
                        let key = match current_element.as_str() {
                            "title" => "title",
                            "creator" => "author",
                            "subject" => "subject",
                            "description" => "description",
                            "keywords" => "keywords",
                            "lastModifiedBy" => "last_modified_by",
                            "created" => "created",
                            "modified" => "modified",
                            "revision" => "revision",
                            "category" => "category",
                            _ => "",
                        };
                        if !key.is_empty() {
                            metadata.insert(key.to_string(), text);
                        }
                    }
                }
                Ok(Event::End(_)) => {
                    current_element.clear();
                }
                Ok(Event::Eof) => break,
                Err(_) => break,
                _ => {}
            }
            buf.clear();
        }
    }

    /// Parse app.xml (application metadata)
    fn parse_app_xml(&self, xml: &str, metadata: &mut HashMap<String, String>) {
        use quick_xml::events::Event;
        use quick_xml::Reader;

        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut current_element = String::new();
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    let name = String::from_utf8_lossy(e.local_name().as_ref()).to_string();
                    current_element = name;
                }
                Ok(Event::Text(ref e)) => {
                    let text = e.unescape().map(|s| s.to_string()).unwrap_or_default();
                    if !text.is_empty() && !current_element.is_empty() {
                        let key = match current_element.as_str() {
                            "Application" => "application",
                            "AppVersion" => "app_version",
                            "Company" => "company",
                            "Template" => "template",
                            "TotalTime" => "total_editing_time",
                            "Pages" => "pages",
                            "Words" => "words",
                            "Characters" => "characters",
                            "CharactersWithSpaces" => "characters_with_spaces",
                            "Paragraphs" => "paragraphs",
                            "Lines" => "lines",
                            _ => "",
                        };
                        if !key.is_empty() {
                            metadata.insert(key.to_string(), text);
                        }
                    }
                }
                Ok(Event::End(_)) => {
                    current_element.clear();
                }
                Ok(Event::Eof) => break,
                Err(_) => break,
                _ => {}
            }
            buf.clear();
        }
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
            return Err(LoaderError::InvalidPath(format!(
                "File not found: {}",
                source
            )));
        }

        if !path.is_file() {
            return Err(LoaderError::InvalidPath(format!("{} is not a file", source)));
        }

        // Check file extension
        if let Some(ext) = path.extension() {
            if ext.to_string_lossy().to_lowercase() != "docx" {
                return Err(LoaderError::UnsupportedFormat(format!(
                    "Expected .docx file, got .{}",
                    ext.to_string_lossy()
                )));
            }
        } else {
            return Err(LoaderError::UnsupportedFormat(
                "No file extension".to_string(),
            ));
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

    #[test]
    fn test_docx_loader_default() {
        let loader = DocxLoader::default();
        assert!(!loader.extract_metadata);
    }

    #[test]
    fn test_parse_core_xml() {
        let loader = DocxLoader::new();
        let xml = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                           xmlns:dc="http://purl.org/dc/elements/1.1/"
                           xmlns:dcterms="http://purl.org/dc/terms/">
            <dc:title>Test Document</dc:title>
            <dc:creator>John Doe</dc:creator>
            <dc:subject>Testing</dc:subject>
            <cp:keywords>test, document, example</cp:keywords>
            <cp:lastModifiedBy>Jane Doe</cp:lastModifiedBy>
            <dcterms:created>2024-01-15T10:30:00Z</dcterms:created>
            <dcterms:modified>2024-12-29T14:00:00Z</dcterms:modified>
        </cp:coreProperties>"#;

        let mut metadata = HashMap::new();
        loader.parse_core_xml(xml, &mut metadata);

        assert_eq!(metadata.get("title"), Some(&"Test Document".to_string()));
        assert_eq!(metadata.get("author"), Some(&"John Doe".to_string()));
        assert_eq!(metadata.get("subject"), Some(&"Testing".to_string()));
        assert_eq!(
            metadata.get("keywords"),
            Some(&"test, document, example".to_string())
        );
        assert_eq!(
            metadata.get("last_modified_by"),
            Some(&"Jane Doe".to_string())
        );
    }

    #[test]
    fn test_parse_app_xml() {
        let loader = DocxLoader::new();
        let xml = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
            <Application>Microsoft Office Word</Application>
            <AppVersion>16.0000</AppVersion>
            <Company>Acme Corp</Company>
            <Pages>5</Pages>
            <Words>1234</Words>
            <Characters>7500</Characters>
            <Paragraphs>42</Paragraphs>
        </Properties>"#;

        let mut metadata = HashMap::new();
        loader.parse_app_xml(xml, &mut metadata);

        assert_eq!(
            metadata.get("application"),
            Some(&"Microsoft Office Word".to_string())
        );
        assert_eq!(metadata.get("app_version"), Some(&"16.0000".to_string()));
        assert_eq!(metadata.get("company"), Some(&"Acme Corp".to_string()));
        assert_eq!(metadata.get("pages"), Some(&"5".to_string()));
        assert_eq!(metadata.get("words"), Some(&"1234".to_string()));
    }
}
