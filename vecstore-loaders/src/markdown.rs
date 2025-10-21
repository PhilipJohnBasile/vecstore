//! Markdown document loader

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use std::fs;
use std::path::Path;

/// Loader for Markdown files
///
/// Parses Markdown and extracts text content, optionally preserving structure.
///
/// # Example
///
/// ```no_run
/// use vecstore_loaders::{MarkdownLoader, DocumentLoader};
///
/// let loader = MarkdownLoader::new();
/// let document = loader.load("README.md")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct MarkdownLoader {
    /// Whether to preserve Markdown formatting in output
    preserve_formatting: bool,

    /// Whether to include link URLs
    include_links: bool,

    /// Whether to include image alt text
    include_images: bool,
}

impl MarkdownLoader {
    /// Create a new Markdown loader
    pub fn new() -> Self {
        Self {
            preserve_formatting: false,
            include_links: true,
            include_images: true,
        }
    }

    /// Preserve Markdown formatting (default: plain text)
    pub fn with_formatting(mut self) -> Self {
        self.preserve_formatting = true;
        self
    }

    /// Include link URLs in output
    pub fn with_links(mut self, include: bool) -> Self {
        self.include_links = include;
        self
    }

    /// Include image alt text in output
    pub fn with_images(mut self, include: bool) -> Self {
        self.include_images = include;
        self
    }

    /// Extract plain text from Markdown
    fn extract_text(&self, markdown: &str) -> String {
        let options = Options::all();
        let parser = Parser::new_ext(markdown, options);

        let mut output = String::new();
        let mut in_link = false;
        let mut link_url = String::new();
        let mut in_image = false;

        for event in parser {
            match event {
                Event::Text(text) => {
                    output.push_str(&text);
                    if in_link && self.include_links {
                        output.push_str(&format!(" ({})", link_url));
                        in_link = false;
                    }
                }
                Event::Code(code) => {
                    if self.preserve_formatting {
                        output.push('`');
                        output.push_str(&code);
                        output.push('`');
                    } else {
                        output.push_str(&code);
                    }
                }
                Event::Start(tag) => match tag {
                    Tag::Link { dest_url, .. } => {
                        link_url = dest_url.to_string();
                        in_link = true;
                    }
                    Tag::Image { dest_url, .. } => {
                        if self.include_images {
                            in_image = true;
                            link_url = dest_url.to_string();
                        }
                    }
                    Tag::Heading { .. } => {
                        if self.preserve_formatting {
                            output.push_str("\n# ");
                        } else {
                            output.push('\n');
                        }
                    }
                    Tag::Paragraph => {
                        output.push('\n');
                    }
                    Tag::CodeBlock(_) => {
                        if self.preserve_formatting {
                            output.push_str("\n```\n");
                        } else {
                            output.push('\n');
                        }
                    }
                    Tag::List(_) => {
                        output.push('\n');
                    }
                    Tag::Item => {
                        if self.preserve_formatting {
                            output.push_str("\n- ");
                        } else {
                            output.push_str("\n• ");
                        }
                    }
                    _ => {}
                },
                Event::End(tag_end) => match tag_end {
                    TagEnd::Link => {
                        in_link = false;
                    }
                    TagEnd::Image => {
                        if in_image && self.include_images {
                            output.push_str(&format!(" [image: {}]", link_url));
                            in_image = false;
                        }
                    }
                    TagEnd::CodeBlock => {
                        if self.preserve_formatting {
                            output.push_str("\n```\n");
                        }
                    }
                    TagEnd::Heading(_) => {
                        output.push('\n');
                    }
                    TagEnd::Paragraph => {
                        output.push('\n');
                    }
                    _ => {}
                },
                Event::SoftBreak | Event::HardBreak => {
                    output.push(' ');
                }
                Event::Rule => {
                    output.push_str("\n---\n");
                }
                _ => {}
            }
        }

        output.trim().to_string()
    }
}

impl Default for MarkdownLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for MarkdownLoader {
    fn load(&self, source: &str) -> Result<Document> {
        let path = Path::new(source);

        if !path.exists() {
            return Err(LoaderError::InvalidPath(format!("File not found: {}", source)));
        }

        if !path.is_file() {
            return Err(LoaderError::InvalidPath(format!("{} is not a file", source)));
        }

        let markdown = fs::read_to_string(path)?;
        let content = self.extract_text(&markdown);

        let mut document = Document::new(content, source.to_string());

        // Add metadata
        document.add_metadata("format", "markdown");
        document.add_metadata("original_size", markdown.len().to_string());

        // Extract title from first heading if present
        let lines: Vec<&str> = markdown.lines().collect();
        for line in lines.iter().take(10) {
            if line.starts_with("# ") {
                let title = line.trim_start_matches("# ").trim();
                document.add_metadata("title", title);
                break;
            }
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
        "MarkdownLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["md", "markdown", "mdown", "mkd"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_markdown() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# Test Document").unwrap();
        writeln!(temp_file, "").unwrap();
        writeln!(temp_file, "This is a **bold** test.").unwrap();

        let loader = MarkdownLoader::new();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("Test Document"));
        assert!(document.content.contains("bold"));
        assert_eq!(document.metadata.get("title"), Some(&"Test Document".to_string()));
    }

    #[test]
    fn test_markdown_with_links() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Check out [Rust](https://rust-lang.org)").unwrap();

        let loader = MarkdownLoader::new().with_links(true);
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("Rust"));
        assert!(document.content.contains("rust-lang.org"));
    }

    #[test]
    fn test_markdown_code_blocks() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "```rust").unwrap();
        writeln!(temp_file, "fn main() {{}}").unwrap();
        writeln!(temp_file, "```").unwrap();

        let loader = MarkdownLoader::new();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("fn main"));
    }

    #[test]
    fn test_markdown_lists() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "- Item 1").unwrap();
        writeln!(temp_file, "- Item 2").unwrap();
        writeln!(temp_file, "- Item 3").unwrap();

        let loader = MarkdownLoader::new();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        assert!(document.content.contains("Item 1"));
        assert!(document.content.contains("Item 2"));
        assert!(document.content.contains("Item 3"));
    }

    #[test]
    fn test_preserve_formatting() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# Heading").unwrap();
        writeln!(temp_file, "").unwrap();
        writeln!(temp_file, "Some `code` here").unwrap();

        let loader = MarkdownLoader::new().with_formatting();
        let document = loader.load(temp_file.path().to_str().unwrap()).unwrap();

        // Should preserve backticks for inline code
        assert!(document.content.contains("`code`"));
    }
}
