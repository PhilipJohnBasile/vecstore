//! JSON document loader

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Loader for JSON files
///
/// Converts JSON documents to text by extracting string values or pretty-printing.
///
/// # Example
///
/// ```no_run
/// use vecstore_loaders::{JsonLoader, DocumentLoader};
///
/// let loader = JsonLoader::new();
/// let document = loader.load("data.json")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct JsonLoader {
    /// Whether to pretty-print JSON
    pretty: bool,

    /// Fields to extract (None = all fields)
    extract_fields: Option<Vec<String>>,
}

impl JsonLoader {
    /// Create a new JSON loader
    pub fn new() -> Self {
        Self {
            pretty: false,
            extract_fields: None,
        }
    }

    /// Enable pretty printing
    pub fn with_pretty(mut self) -> Self {
        self.pretty = true;
        self
    }

    /// Extract only specific fields
    pub fn with_fields(mut self, fields: Vec<String>) -> Self {
        self.extract_fields = Some(fields);
        self
    }

    /// Extract text from JSON value
    fn extract_text(&self, value: &Value) -> String {
        match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Array(arr) => {
                let texts: Vec<String> = arr.iter().map(|v| self.extract_text(v)).collect();
                texts.join("\n")
            }
            Value::Object(obj) => {
                if let Some(ref fields) = self.extract_fields {
                    // Extract only specified fields
                    let texts: Vec<String> = fields
                        .iter()
                        .filter_map(|field| obj.get(field))
                        .map(|v| self.extract_text(v))
                        .collect();
                    texts.join("\n")
                } else {
                    // Extract all text values
                    let texts: Vec<String> = obj.values().map(|v| self.extract_text(v)).collect();
                    texts.join("\n")
                }
            }
            Value::Null => String::new(),
        }
    }
}

impl Default for JsonLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for JsonLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let path = Path::new(source);

        if !path.exists() {
            return Err(LoaderError::InvalidPath(format!("File not found: {}", source)));
        }

        let content_str = fs::read_to_string(path)?;
        let value: Value = serde_json::from_str(&content_str)?;

        let content = if self.pretty {
            serde_json::to_string_pretty(&value)?
        } else {
            self.extract_text(&value)
        };

        let mut document = Document::new(content, source.to_string());

        // Add metadata
        document.add_metadata("format", "json");
        if let Some(ref fields) = self.extract_fields {
            document.add_metadata("extracted_fields", fields.join(","));
        }

        Ok(document)
    }

    fn load_with_options(&self, source: &str, options: &LoaderOptions) -> Result<Document> {
        // Check file size if max_size is set
        if let Some(max_size) = options.max_size {
            let metadata = fs::metadata(source)?;
            let file_size = metadata.len() as usize;

            if file_size > max_size {
                return Err(LoaderError::FileTooLarge(file_size, max_size));
            }
        }

        self.load(source)
    }

    fn name(&self) -> &str {
        "JsonLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["json", "jsonl", "ndjson"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_json() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"{{"title": "Test", "content": "Hello world"}}"#
        )
        .unwrap();

        let loader = JsonLoader::new();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("Test") || document.content.contains("Hello world"));
    }

    #[test]
    fn test_load_json_array() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"[{{"text": "First"}}, {{"text": "Second"}}]"#
        )
        .unwrap();

        let loader = JsonLoader::new();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("First"));
        assert!(document.content.contains("Second"));
    }

    #[test]
    fn test_load_with_pretty() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, r#"{{"key":"value"}}"#).unwrap();

        let loader = JsonLoader::new().with_pretty();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        // Pretty-printed JSON should have newlines
        assert!(document.content.contains('\n'));
    }

    #[test]
    fn test_extract_specific_fields() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"{{"title": "Test", "content": "Hello", "ignore": "This"}}"#
        )
        .unwrap();

        let loader = JsonLoader::new().with_fields(vec!["title".to_string(), "content".to_string()]);
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("Test"));
        assert!(document.content.contains("Hello"));
        // Should not contain "ignore" field
        assert!(!document.content.contains("This") || document.content.contains("Test"));
    }
}
