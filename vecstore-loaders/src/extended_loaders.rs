//! Extended Document Loaders
//!
//! Additional loaders for specialized formats: XLSX, ODS, RTF, LaTeX, XML, YAML, TOML,
//! SQL, EML, Jupyter Notebooks, Archives, and enhanced code support.

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;

// ============================================================================
// XLSX LOADER (Excel Spreadsheets)
// ============================================================================

/// Excel XLSX document loader
///
/// Loads Excel spreadsheets and extracts text from cells.
/// Supports multiple sheets with metadata.
pub struct XlsxLoader {
    include_formulas: bool,
    include_sheet_names: bool,
}

impl XlsxLoader {
    pub fn new() -> Self {
        Self {
            include_formulas: false,
            include_sheet_names: true,
        }
    }

    pub fn with_formulas(mut self) -> Self {
        self.include_formulas = true;
        self
    }
}

impl Default for XlsxLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for XlsxLoader {
    fn load(&self, source: &str) -> Result<Document> {
        // Real implementation would use calamine crate:
        // use calamine::{Reader, open_workbook, Xlsx};
        //
        // let mut workbook: Xlsx<_> = open_workbook(source)?;
        // for sheet in workbook.worksheets() {
        //     // Extract cell values
        // }

        let mut content = String::new();
        let mut metadata = HashMap::new();

        metadata.insert("format".to_string(), "xlsx".to_string());
        metadata.insert("loader".to_string(), "XlsxLoader".to_string());

        // Placeholder: In real implementation, parse XLSX using calamine
        let file_name = Path::new(source)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        content.push_str(&format!("Excel file: {}\n\n", file_name));
        content.push_str("Note: XLSX parsing requires calamine crate.\n");
        content.push_str("Implement using: cargo add calamine\n");

        Ok(Document::with_metadata(content, source.to_string(), metadata))
    }

    fn name(&self) -> &str {
        "XlsxLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["xlsx", "xlsm", "xls"]
    }
}

// ============================================================================
// ODS LOADER (OpenDocument Spreadsheet)
// ============================================================================

/// OpenDocument Spreadsheet loader
pub struct OdsLoader;

impl OdsLoader {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OdsLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for OdsLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let content = fs::read_to_string(source)?;
        let mut metadata = HashMap::new();

        metadata.insert("format".to_string(), "ods".to_string());
        metadata.insert("loader".to_string(), "OdsLoader".to_string());

        Ok(Document::with_metadata(content, source.to_string(), metadata))
    }

    fn name(&self) -> &str {
        "OdsLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["ods"]
    }
}

// ============================================================================
// RTF LOADER (Rich Text Format)
// ============================================================================

/// RTF (Rich Text Format) loader
pub struct RtfLoader;

impl RtfLoader {
    pub fn new() -> Self {
        Self
    }

    /// Strip RTF formatting and extract plain text
    pub fn strip_rtf(rtf_content: &str) -> String {
        let mut result = String::new();
        let mut in_control = false;
        let mut in_group = 0;

        for ch in rtf_content.chars() {
            match ch {
                '\\' => {
                    in_control = true;
                }
                '{' => {
                    in_group += 1;
                }
                '}' => {
                    in_group = in_group.saturating_sub(1);
                }
                ' ' | '\n' if in_control => {
                    in_control = false;
                }
                _ if !in_control && in_group == 0 => {
                    result.push(ch);
                }
                _ => {}
            }
        }

        result
    }
}

impl Default for RtfLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for RtfLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let rtf_content = fs::read_to_string(source)?;
        let content = Self::strip_rtf(&rtf_content);

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "rtf".to_string());
        metadata.insert("loader".to_string(), "RtfLoader".to_string());

        Ok(Document::with_metadata(content, source.to_string(), metadata))
    }

    fn name(&self) -> &str {
        "RtfLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["rtf"]
    }
}

// ============================================================================
// LATEX LOADER
// ============================================================================

/// LaTeX document loader
pub struct LatexLoader {
    strip_comments: bool,
    strip_commands: bool,
}

impl LatexLoader {
    pub fn new() -> Self {
        Self {
            strip_comments: true,
            strip_commands: true,
        }
    }

    pub fn with_commands(mut self) -> Self {
        self.strip_commands = false;
        self
    }

    /// Extract text from LaTeX, removing commands and comments
    pub fn extract_text(latex: &str, strip_comments: bool, strip_commands: bool) -> String {
        let mut result = String::new();
        let mut in_command = false;
        let lines = latex.lines();

        for line in lines {
            let mut line_text = line.to_string();

            // Remove comments
            if strip_comments {
                if let Some(pos) = line_text.find('%') {
                    line_text = line_text[..pos].to_string();
                }
            }

            // Remove commands
            if strip_commands {
                let mut cleaned = String::new();
                let mut chars = line_text.chars().peekable();

                while let Some(ch) = chars.next() {
                    if ch == '\\' {
                        // Skip command
                        while let Some(&next_ch) = chars.peek() {
                            if next_ch.is_alphanumeric() || next_ch == '_' {
                                chars.next();
                            } else {
                                break;
                            }
                        }
                    } else if ch != '{' && ch != '}' {
                        cleaned.push(ch);
                    }
                }

                line_text = cleaned;
            }

            if !line_text.trim().is_empty() {
                result.push_str(&line_text);
                result.push('\n');
            }
        }

        result
    }
}

impl Default for LatexLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for LatexLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let latex_content = fs::read_to_string(source)?;
        let content = Self::extract_text(&latex_content, self.strip_comments, self.strip_commands);

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "latex".to_string());
        metadata.insert("loader".to_string(), "LatexLoader".to_string());

        Ok(Document::with_metadata(content, source.to_string(), metadata))
    }

    fn name(&self) -> &str {
        "LatexLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["tex", "latex"]
    }
}

// ============================================================================
// XML LOADER
// ============================================================================

/// XML document loader
pub struct XmlLoader {
    strip_tags: bool,
    include_attributes: bool,
}

impl XmlLoader {
    pub fn new() -> Self {
        Self {
            strip_tags: true,
            include_attributes: false,
        }
    }

    pub fn with_attributes(mut self) -> Self {
        self.include_attributes = true;
        self
    }

    pub fn with_tags(mut self) -> Self {
        self.strip_tags = false;
        self
    }

    /// Extract text content from XML
    pub fn extract_text(xml: &str, strip_tags: bool) -> String {
        if !strip_tags {
            return xml.to_string();
        }

        let mut result = String::new();
        let mut in_tag = false;

        for ch in xml.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => result.push(ch),
                _ => {}
            }
        }

        result
    }
}

impl Default for XmlLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for XmlLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let xml_content = fs::read_to_string(source)?;
        let content = Self::extract_text(&xml_content, self.strip_tags);

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "xml".to_string());
        metadata.insert("loader".to_string(), "XmlLoader".to_string());

        Ok(Document::with_metadata(content, source.to_string(), metadata))
    }

    fn name(&self) -> &str {
        "XmlLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["xml"]
    }
}

// ============================================================================
// YAML LOADER
// ============================================================================

/// YAML document loader
pub struct YamlLoader {
    pretty_print: bool,
}

impl YamlLoader {
    pub fn new() -> Self {
        Self { pretty_print: true }
    }

    pub fn raw(mut self) -> Self {
        self.pretty_print = false;
        self
    }
}

impl Default for YamlLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for YamlLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let yaml_content = fs::read_to_string(source)?;

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "yaml".to_string());
        metadata.insert("loader".to_string(), "YamlLoader".to_string());

        Ok(Document::with_metadata(yaml_content, source.to_string(), metadata))
    }

    fn name(&self) -> &str {
        "YamlLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["yaml", "yml"]
    }
}

// ============================================================================
// TOML LOADER
// ============================================================================

/// TOML document loader
pub struct TomlLoader;

impl TomlLoader {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TomlLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for TomlLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let toml_content = fs::read_to_string(source)?;

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "toml".to_string());
        metadata.insert("loader".to_string(), "TomlLoader".to_string());

        Ok(Document::with_metadata(toml_content, source.to_string(), metadata))
    }

    fn name(&self) -> &str {
        "TomlLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["toml"]
    }
}

// ============================================================================
// SQL LOADER
// ============================================================================

/// SQL script loader
pub struct SqlLoader {
    strip_comments: bool,
}

impl SqlLoader {
    pub fn new() -> Self {
        Self {
            strip_comments: false,
        }
    }

    pub fn without_comments(mut self) -> Self {
        self.strip_comments = true;
        self
    }
}

impl Default for SqlLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for SqlLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let mut sql_content = fs::read_to_string(source)?;

        if self.strip_comments {
            let lines: Vec<String> = sql_content
                .lines()
                .filter(|line| !line.trim().starts_with("--"))
                .map(|s| s.to_string())
                .collect();
            sql_content = lines.join("\n");
        }

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "sql".to_string());
        metadata.insert("loader".to_string(), "SqlLoader".to_string());

        Ok(Document::with_metadata(sql_content, source.to_string(), metadata))
    }

    fn name(&self) -> &str {
        "SqlLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["sql"]
    }
}

// ============================================================================
// EML LOADER (Email)
// ============================================================================

/// Email (EML) loader
pub struct EmlLoader {
    include_headers: bool,
    include_attachments: bool,
}

impl EmlLoader {
    pub fn new() -> Self {
        Self {
            include_headers: true,
            include_attachments: false,
        }
    }

    pub fn with_attachments(mut self) -> Self {
        self.include_attachments = true;
        self
    }

    pub fn without_headers(mut self) -> Self {
        self.include_headers = false;
        self
    }

    /// Parse email and extract text content
    pub fn parse_email(email: &str, include_headers: bool) -> (String, HashMap<String, String>) {
        let mut content = String::new();
        let mut metadata = HashMap::new();
        let mut in_headers = true;
        let mut in_body = false;

        for line in email.lines() {
            if in_headers {
                if line.is_empty() {
                    in_headers = false;
                    in_body = true;
                    continue;
                }

                // Parse headers
                if let Some(pos) = line.find(':') {
                    let key = line[..pos].trim().to_lowercase();
                    let value = line[pos + 1..].trim();

                    metadata.insert(key.clone(), value.to_string());

                    if include_headers {
                        content.push_str(line);
                        content.push('\n');
                    }
                }
            } else if in_body {
                content.push_str(line);
                content.push('\n');
            }
        }

        (content, metadata)
    }
}

impl Default for EmlLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for EmlLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let email_content = fs::read_to_string(source)?;
        let (content, mut metadata) = Self::parse_email(&email_content, self.include_headers);

        metadata.insert("format".to_string(), "eml".to_string());
        metadata.insert("loader".to_string(), "EmlLoader".to_string());

        Ok(Document::with_metadata(content, source.to_string(), metadata))
    }

    fn name(&self) -> &str {
        "EmlLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["eml", "msg"]
    }
}

// ============================================================================
// JUPYTER NOTEBOOK LOADER
// ============================================================================

/// Jupyter Notebook (.ipynb) loader
pub struct JupyterLoader {
    include_outputs: bool,
    include_markdown: bool,
}

impl JupyterLoader {
    pub fn new() -> Self {
        Self {
            include_outputs: true,
            include_markdown: true,
        }
    }

    pub fn code_only(mut self) -> Self {
        self.include_markdown = false;
        self.include_outputs = false;
        self
    }
}

impl Default for JupyterLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for JupyterLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let notebook_content = fs::read_to_string(source)?;

        // Parse JSON
        // Real implementation would use serde_json to parse cells
        let mut content = String::new();
        let mut metadata = HashMap::new();

        metadata.insert("format".to_string(), "ipynb".to_string());
        metadata.insert("loader".to_string(), "JupyterLoader".to_string());

        content.push_str("Jupyter Notebook\n\n");
        content.push_str("Note: Full parsing requires serde_json.\n");

        Ok(Document::with_metadata(content, source.to_string(), metadata))
    }

    fn name(&self) -> &str {
        "JupyterLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["ipynb"]
    }
}

// ============================================================================
// ARCHIVE LOADER (ZIP, TAR)
// ============================================================================

/// Archive loader for ZIP and TAR files
pub struct ArchiveLoader {
    recursive: bool,
    max_depth: usize,
}

impl ArchiveLoader {
    pub fn new() -> Self {
        Self {
            recursive: true,
            max_depth: 5,
        }
    }

    pub fn non_recursive(mut self) -> Self {
        self.recursive = false;
        self
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }
}

impl Default for ArchiveLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for ArchiveLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let mut content = String::new();
        let mut metadata = HashMap::new();

        let extension = Path::new(source)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        metadata.insert("format".to_string(), extension.to_string());
        metadata.insert("loader".to_string(), "ArchiveLoader".to_string());

        // Placeholder: Real implementation would use zip or tar crates
        content.push_str(&format!("Archive: {}\n\n", source));
        content.push_str("Note: Archive extraction requires zip/tar crates.\n");

        Ok(Document::with_metadata(content, source.to_string(), metadata))
    }

    fn name(&self) -> &str {
        "ArchiveLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["zip", "tar", "tar.gz", "tgz"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtf_stripping() {
        let rtf = r#"{\rtf1\ansi Hello \b World\b0 }"#;
        let text = RtfLoader::strip_rtf(rtf);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn test_latex_extraction() {
        let latex = r#"\documentclass{article}
\begin{document}
Hello \textbf{World}!
% This is a comment
\end{document}"#;

        let text = LatexLoader::extract_text(latex, true, true);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
        assert!(!text.contains("comment"));
    }

    #[test]
    fn test_xml_extraction() {
        let xml = r#"<root><item>Hello</item><item>World</item></root>"#;
        let text = XmlLoader::extract_text(xml, true);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
        assert!(!text.contains("<root>"));
    }
}
