//! Source code loader

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use std::fs;
use std::path::Path;

/// Loader for source code files
///
/// Loads source code with syntax awareness, preserving structure and adding metadata.
///
/// # Example
///
/// ```no_run
/// use vecstore_loaders::{CodeLoader, DocumentLoader};
///
/// let loader = CodeLoader::new();
/// let document = loader.load("main.rs")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CodeLoader {
    /// Whether to include comments
    include_comments: bool,

    /// Whether to preserve indentation
    preserve_indentation: bool,

    /// Whether to add line numbers
    add_line_numbers: bool,
}

impl CodeLoader {
    /// Create a new code loader
    pub fn new() -> Self {
        Self {
            include_comments: true,
            preserve_indentation: true,
            add_line_numbers: false,
        }
    }

    /// Exclude comments from output
    pub fn without_comments(mut self) -> Self {
        self.include_comments = false;
        self
    }

    /// Remove indentation for more compact output
    pub fn without_indentation(mut self) -> Self {
        self.preserve_indentation = false;
        self
    }

    /// Add line numbers to output
    pub fn with_line_numbers(mut self) -> Self {
        self.add_line_numbers = true;
        self
    }

    /// Detect language from file extension
    fn detect_language(path: &Path) -> Option<String> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| match ext {
                "rs" => "rust",
                "py" => "python",
                "js" => "javascript",
                "ts" => "typescript",
                "go" => "go",
                "java" => "java",
                "c" => "c",
                "cpp" | "cc" | "cxx" => "cpp",
                "h" | "hpp" => "c_header",
                "rb" => "ruby",
                "php" => "php",
                "swift" => "swift",
                "kt" => "kotlin",
                "scala" => "scala",
                "r" => "r",
                "sh" | "bash" => "shell",
                "sql" => "sql",
                "html" => "html",
                "css" => "css",
                "json" => "json",
                "xml" => "xml",
                "yaml" | "yml" => "yaml",
                "toml" => "toml",
                "md" => "markdown",
                _ => "unknown",
            }
            .to_string())
    }

    /// Process code content
    fn process_code(&self, content: &str, language: &str) -> String {
        let lines: Vec<&str> = content.lines().collect();
        let mut processed_lines = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            let mut processed_line = line.to_string();

            // Remove comments if requested
            if !self.include_comments {
                processed_line = self.remove_comments(&processed_line, language);
            }

            // Remove indentation if requested
            if !self.preserve_indentation {
                processed_line = processed_line.trim_start().to_string();
            }

            // Add line numbers if requested
            if self.add_line_numbers {
                processed_line = format!("{:4} | {}", i + 1, processed_line);
            }

            // Only include non-empty lines or preserve structure
            if self.preserve_indentation || !processed_line.trim().is_empty() {
                processed_lines.push(processed_line);
            }
        }

        processed_lines.join("\n")
    }

    /// Remove comments from a line (simple implementation)
    fn remove_comments(&self, line: &str, language: &str) -> String {
        match language {
            "rust" | "javascript" | "typescript" | "java" | "c" | "cpp" | "go" | "swift"
            | "kotlin" | "scala" => {
                // Remove single-line comments
                if let Some(pos) = line.find("//") {
                    line[..pos].trim_end().to_string()
                } else {
                    line.to_string()
                }
            }
            "python" | "ruby" | "shell" => {
                // Remove # comments
                if let Some(pos) = line.find('#') {
                    // Check if # is in a string
                    line[..pos].trim_end().to_string()
                } else {
                    line.to_string()
                }
            }
            "sql" => {
                // Remove -- comments
                if let Some(pos) = line.find("--") {
                    line[..pos].trim_end().to_string()
                } else {
                    line.to_string()
                }
            }
            _ => line.to_string(),
        }
    }

    /// Extract code structure metadata (functions, classes, etc.)
    fn extract_structure(&self, content: &str, language: &str) -> Vec<String> {
        let mut structure = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();

            // Rust
            if language == "rust" {
                if trimmed.starts_with("fn ") {
                    structure.push(trimmed.to_string());
                } else if trimmed.starts_with("struct ") || trimmed.starts_with("enum ") {
                    structure.push(trimmed.to_string());
                }
            }
            // Python
            else if language == "python" {
                if trimmed.starts_with("def ") || trimmed.starts_with("class ") {
                    structure.push(trimmed.to_string());
                }
            }
            // JavaScript/TypeScript
            else if language == "javascript" || language == "typescript" {
                if trimmed.starts_with("function ")
                    || trimmed.starts_with("class ")
                    || trimmed.starts_with("const ")
                    || trimmed.starts_with("let ")
                {
                    structure.push(trimmed.to_string());
                }
            }
        }

        structure
    }
}

impl Default for CodeLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for CodeLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let path = Path::new(source);

        if !path.exists() {
            return Err(LoaderError::InvalidPath(format!("File not found: {}", source)));
        }

        if !path.is_file() {
            return Err(LoaderError::InvalidPath(format!("{} is not a file", source)));
        }

        let raw_content = fs::read_to_string(path)?;

        // Detect language
        let language = Self::detect_language(path).unwrap_or_else(|| "unknown".to_string());

        // Process code
        let content = self.process_code(&raw_content, &language);

        let mut document = Document::new(content, source.to_string());

        // Add metadata
        document.add_metadata("format", "code");
        document.add_metadata("language", &language);
        document.add_metadata("lines", raw_content.lines().count().to_string());

        // Extract structure
        let structure = self.extract_structure(&raw_content, &language);
        if !structure.is_empty() {
            document.add_metadata("functions_classes", structure.join("; "));
            document.add_metadata("structure_count", structure.len().to_string());
        }

        // Add file extension
        if let Some(extension) = path.extension() {
            document.add_metadata("extension", extension.to_string_lossy().to_string());
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
        "CodeLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &[
            "rs", "py", "js", "ts", "go", "java", "c", "cpp", "h", "hpp", "rb", "php", "swift",
            "kt", "scala", "r", "sh", "bash", "sql",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_rust_code() {
        let mut temp_file = tempfile::Builder::new()
            .suffix(".rs")
            .tempfile()
            .unwrap();
        writeln!(temp_file, "fn main() {{").unwrap();
        writeln!(temp_file, "    println!(\"Hello\");").unwrap();
        writeln!(temp_file, "}}").unwrap();

        let loader = CodeLoader::new();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("fn main"));
        assert!(document.content.contains("println"));
        assert_eq!(document.metadata.get("language"), Some(&"rust".to_string()));
    }

    #[test]
    fn test_detect_language() {
        assert_eq!(
            CodeLoader::detect_language(Path::new("test.rs")),
            Some("rust".to_string())
        );
        assert_eq!(
            CodeLoader::detect_language(Path::new("test.py")),
            Some("python".to_string())
        );
        assert_eq!(
            CodeLoader::detect_language(Path::new("test.js")),
            Some("javascript".to_string())
        );
    }

    #[test]
    fn test_without_comments() {
        let mut temp_file = tempfile::Builder::new()
            .suffix(".rs")
            .tempfile()
            .unwrap();
        writeln!(temp_file, "fn main() {{ // This is a comment").unwrap();
        writeln!(temp_file, "    let x = 5; // Another comment").unwrap();
        writeln!(temp_file, "}}").unwrap();

        let loader = CodeLoader::new().without_comments();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(!document.content.contains("This is a comment"));
        assert!(!document.content.contains("Another comment"));
        assert!(document.content.contains("fn main"));
    }

    #[test]
    fn test_with_line_numbers() {
        let mut temp_file = tempfile::Builder::new()
            .suffix(".py")
            .tempfile()
            .unwrap();
        writeln!(temp_file, "def hello():").unwrap();
        writeln!(temp_file, "    print('Hello')").unwrap();

        let loader = CodeLoader::new().with_line_numbers();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("1 |"));
        assert!(document.content.contains("2 |"));
    }

    #[test]
    fn test_extract_structure() {
        let loader = CodeLoader::new();
        let rust_code = "fn main() {}\nstruct Foo {}\nenum Bar {}";
        let structure = loader.extract_structure(rust_code, "rust");

        assert_eq!(structure.len(), 3);
        assert!(structure.iter().any(|s| s.contains("fn main")));
        assert!(structure.iter().any(|s| s.contains("struct Foo")));
        assert!(structure.iter().any(|s| s.contains("enum Bar")));
    }
}
