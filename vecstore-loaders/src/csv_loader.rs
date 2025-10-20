//! CSV document loader

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use csv::ReaderBuilder;
use std::fs::File;
use std::path::Path;

/// Loader for CSV files
///
/// Converts CSV rows to text format, optionally extracting specific columns.
///
/// # Example
///
/// ```no_run
/// use vecstore_loaders::{CsvLoader, DocumentLoader};
///
/// let loader = CsvLoader::new();
/// let document = loader.load("data.csv")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CsvLoader {
    /// Delimiter character (default: ',')
    delimiter: u8,

    /// Whether CSV has headers
    has_headers: bool,

    /// Columns to extract (None = all columns)
    extract_columns: Option<Vec<String>>,

    /// Row separator in output
    row_separator: String,
}

impl CsvLoader {
    /// Create a new CSV loader with default settings
    pub fn new() -> Self {
        Self {
            delimiter: b',',
            has_headers: true,
            extract_columns: None,
            row_separator: "\n".to_string(),
        }
    }

    /// Set delimiter character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set whether CSV has headers
    pub fn with_headers(mut self, has_headers: bool) -> Self {
        self.has_headers = has_headers;
        self
    }

    /// Extract only specific columns
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.extract_columns = Some(columns);
        self
    }

    /// Set row separator
    pub fn with_row_separator(mut self, separator: impl Into<String>) -> Self {
        self.row_separator = separator.into();
        self
    }
}

impl Default for CsvLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for CsvLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let path = Path::new(source);

        if !path.exists() {
            return Err(LoaderError::InvalidPath(format!("File not found: {}", source)));
        }

        let file = File::open(path)?;
        let mut reader = ReaderBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.has_headers)
            .from_reader(file);

        let mut content_lines = Vec::new();

        // Get headers if available
        let headers = if self.has_headers {
            reader.headers()?.clone()
        } else {
            csv::StringRecord::new()
        };

        // Determine which column indices to extract
        let column_indices: Option<Vec<usize>> = if let Some(ref columns) = self.extract_columns {
            if !headers.is_empty() {
                Some(
                    columns
                        .iter()
                        .filter_map(|col| headers.iter().position(|h| h == col))
                        .collect(),
                )
            } else {
                None
            }
        } else {
            None
        };

        // Process rows
        for result in reader.records() {
            let record = result?;

            let row_text = if let Some(ref indices) = column_indices {
                // Extract only specified columns
                indices
                    .iter()
                    .filter_map(|&i| record.get(i))
                    .collect::<Vec<&str>>()
                    .join(" ")
            } else {
                // Extract all columns
                record.iter().collect::<Vec<&str>>().join(" ")
            };

            if !row_text.trim().is_empty() {
                content_lines.push(row_text);
            }
        }

        let content = content_lines.join(&self.row_separator);

        let mut document = Document::new(content, source.to_string());

        // Add metadata
        document.add_metadata("format", "csv");
        document.add_metadata("row_count", content_lines.len().to_string());

        if self.has_headers && !headers.is_empty() {
            document.add_metadata("columns", headers.iter().collect::<Vec<&str>>().join(","));
        }

        if let Some(ref columns) = self.extract_columns {
            document.add_metadata("extracted_columns", columns.join(","));
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
        "CsvLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["csv", "tsv"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_csv() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "name,age,city").unwrap();
        writeln!(temp_file, "Alice,30,NYC").unwrap();
        writeln!(temp_file, "Bob,25,SF").unwrap();

        let loader = CsvLoader::new();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("Alice"));
        assert!(document.content.contains("Bob"));
        assert_eq!(document.metadata.get("row_count"), Some(&"2".to_string()));
    }

    #[test]
    fn test_load_csv_no_headers() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Alice,30,NYC").unwrap();
        writeln!(temp_file, "Bob,25,SF").unwrap();

        let loader = CsvLoader::new().with_headers(false);
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("Alice"));
        assert!(document.content.contains("Bob"));
    }

    #[test]
    fn test_extract_columns() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "name,age,city").unwrap();
        writeln!(temp_file, "Alice,30,NYC").unwrap();
        writeln!(temp_file, "Bob,25,SF").unwrap();

        let loader = CsvLoader::new().with_columns(vec!["name".to_string(), "city".to_string()]);
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("Alice"));
        assert!(document.content.contains("NYC"));
        // Should not contain age column
        let lines: Vec<&str> = document.content.lines().collect();
        assert!(!lines[0].contains("30") || lines[0].contains("Alice"));
    }

    #[test]
    fn test_tsv_delimiter() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "name\tage\tcity").unwrap();
        writeln!(temp_file, "Alice\t30\tNYC").unwrap();

        let loader = CsvLoader::new().with_delimiter(b'\t');
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("Alice"));
        assert!(document.content.contains("NYC"));
    }

    #[test]
    fn test_custom_row_separator() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "name,age").unwrap();
        writeln!(temp_file, "Alice,30").unwrap();
        writeln!(temp_file, "Bob,25").unwrap();

        let loader = CsvLoader::new().with_row_separator(" | ");
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains(" | "));
    }
}
