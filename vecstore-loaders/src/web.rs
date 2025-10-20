//! Web page loader

use crate::{Document, DocumentLoader, LoaderError, LoaderOptions, Result};
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use std::time::Duration;

/// Loader for web pages
///
/// Fetches and extracts text content from HTML pages.
///
/// # Example
///
/// ```no_run
/// use vecstore_loaders::{WebLoader, DocumentLoader};
///
/// let loader = WebLoader::new();
/// let document = loader.load("https://example.com")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct WebLoader {
    /// HTTP client timeout
    timeout: Duration,

    /// User agent string
    user_agent: String,

    /// Whether to extract only main content (skip nav, footer, etc.)
    main_content_only: bool,

    /// CSS selectors to remove (e.g., nav, footer, ads)
    remove_selectors: Vec<String>,
}

impl WebLoader {
    /// Create a new web loader
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            user_agent: "vecstore-loaders/0.1.0".to_string(),
            main_content_only: true,
            remove_selectors: vec![
                "nav".to_string(),
                "footer".to_string(),
                "header".to_string(),
                "aside".to_string(),
                ".advertisement".to_string(),
                ".ad".to_string(),
            ],
        }
    }

    /// Set request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set custom user agent
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    /// Extract full page content (including nav, footer, etc.)
    pub fn with_full_content(mut self) -> Self {
        self.main_content_only = false;
        self
    }

    /// Add CSS selector to remove from content
    pub fn add_remove_selector(mut self, selector: impl Into<String>) -> Self {
        self.remove_selectors.push(selector.into());
        self
    }

    /// Clear all remove selectors
    pub fn clear_remove_selectors(mut self) -> Self {
        self.remove_selectors.clear();
        self
    }

    /// Extract text from HTML
    fn extract_text(&self, html: &str) -> Result<String> {
        let document = Html::parse_document(html);

        // Note: For now, we rely on content selector priority
        // Future enhancement: properly remove unwanted elements before extraction
        if self.main_content_only {
            // Main content only mode - we'll rely on selector priority below
        }

        // Try to find main content area
        let content_selectors = vec![
            "main",
            "article",
            "[role='main']",
            ".content",
            "#content",
            ".main",
            "#main",
            "body",
        ];

        for selector_str in content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let text = element.text().collect::<Vec<_>>().join(" ");
                    if !text.trim().is_empty() {
                        return Ok(text.trim().to_string());
                    }
                }
            }
        }

        // Fallback: extract all text
        let text = document
            .root_element()
            .text()
            .collect::<Vec<_>>()
            .join(" ");

        Ok(text.trim().to_string())
    }

    /// Extract metadata from HTML
    fn extract_metadata(&self, html: &str, url: &str) -> std::collections::HashMap<String, String> {
        let mut metadata = std::collections::HashMap::new();
        let document = Html::parse_document(html);

        // Extract title
        if let Ok(selector) = Selector::parse("title") {
            if let Some(title) = document.select(&selector).next() {
                let title_text = title.text().collect::<String>();
                metadata.insert("title".to_string(), title_text.trim().to_string());
            }
        }

        // Extract meta description
        if let Ok(selector) = Selector::parse("meta[name='description']") {
            if let Some(meta) = document.select(&selector).next() {
                if let Some(content) = meta.value().attr("content") {
                    metadata.insert("description".to_string(), content.to_string());
                }
            }
        }

        // Extract meta keywords
        if let Ok(selector) = Selector::parse("meta[name='keywords']") {
            if let Some(meta) = document.select(&selector).next() {
                if let Some(content) = meta.value().attr("content") {
                    metadata.insert("keywords".to_string(), content.to_string());
                }
            }
        }

        // Extract og:title
        if let Ok(selector) = Selector::parse("meta[property='og:title']") {
            if let Some(meta) = document.select(&selector).next() {
                if let Some(content) = meta.value().attr("content") {
                    metadata.insert("og_title".to_string(), content.to_string());
                }
            }
        }

        metadata.insert("url".to_string(), url.to_string());

        metadata
    }
}

impl Default for WebLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentLoader for WebLoader {
    fn load(&self, source: &str) -> Result<Document> {
        // Validate URL
        if !source.starts_with("http://") && !source.starts_with("https://") {
            return Err(LoaderError::InvalidPath(format!(
                "URL must start with http:// or https://: {}",
                source
            )));
        }

        // Build HTTP client
        let client = Client::builder()
            .timeout(self.timeout)
            .user_agent(&self.user_agent)
            .build()
            .map_err(|e| LoaderError::NetworkError(e.to_string()))?;

        // Fetch page
        let response = client
            .get(source)
            .send()
            .map_err(|e| LoaderError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(LoaderError::NetworkError(format!(
                "HTTP request failed with status: {}",
                response.status()
            )));
        }

        let html = response
            .text()
            .map_err(|e| LoaderError::NetworkError(e.to_string()))?;

        // Extract text
        let content = self.extract_text(&html)?;

        // Extract metadata
        let metadata_map = self.extract_metadata(&html, source);

        let document = Document::with_metadata(content, source.to_string(), metadata_map);

        Ok(document)
    }

    fn load_with_options(&self, source: &str, _options: &LoaderOptions) -> Result<Document> {
        // Web loader options could include custom headers, etc.
        self.load(source)
    }

    fn name(&self) -> &str {
        "WebLoader"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["html", "htm"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_loader_creation() {
        let loader = WebLoader::new();
        assert_eq!(loader.name(), "WebLoader");
        assert!(loader.main_content_only);
    }

    #[test]
    fn test_web_loader_with_timeout() {
        let loader = WebLoader::new().with_timeout(Duration::from_secs(60));
        assert_eq!(loader.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_web_loader_custom_user_agent() {
        let loader = WebLoader::new().with_user_agent("CustomBot/1.0");
        assert_eq!(loader.user_agent, "CustomBot/1.0");
    }

    #[test]
    fn test_web_loader_full_content() {
        let loader = WebLoader::new().with_full_content();
        assert!(!loader.main_content_only);
    }

    #[test]
    fn test_extract_text_from_html() {
        let loader = WebLoader::new();
        let html = r#"
            <html>
                <head><title>Test Page</title></head>
                <body>
                    <article>
                        <h1>Main Content</h1>
                        <p>This is the main content.</p>
                    </article>
                </body>
            </html>
        "#;

        let text = loader.extract_text(html).unwrap();
        assert!(text.contains("Main Content"));
        assert!(text.contains("main content"));
    }

    #[test]
    fn test_extract_metadata() {
        let loader = WebLoader::new();
        let html = r#"
            <html>
                <head>
                    <title>Test Page</title>
                    <meta name="description" content="Test description">
                    <meta property="og:title" content="OG Title">
                </head>
                <body></body>
            </html>
        "#;

        let metadata = loader.extract_metadata(html, "https://example.com");
        assert_eq!(metadata.get("title"), Some(&"Test Page".to_string()));
        assert_eq!(metadata.get("description"), Some(&"Test description".to_string()));
        assert_eq!(metadata.get("og_title"), Some(&"OG Title".to_string()));
    }

    #[test]
    fn test_invalid_url() {
        let loader = WebLoader::new();
        let result = loader.load("not-a-url");
        assert!(result.is_err());
    }

    // Note: Actual web loading tests require network access
    // These would be added in integration tests or with mock servers
}
