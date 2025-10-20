//! Error types for vecstore-loaders

use std::io;
use thiserror::Error;

/// Result type for loader operations
pub type Result<T> = std::result::Result<T, LoaderError>;

/// Errors that can occur during document loading
#[derive(Error, Debug)]
pub enum LoaderError {
    /// I/O error occurred
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Invalid file path or source
    #[error("Invalid path: {0}")]
    InvalidPath(String),

    /// File format parsing error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Unsupported file format
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// File too large
    #[error("File too large: {0} bytes (max: {1})")]
    FileTooLarge(usize, usize),

    /// Encoding error
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// Network error (for web loader)
    #[cfg(feature = "web")]
    #[error("Network error: {0}")]
    NetworkError(String),

    /// PDF-specific error
    #[cfg(feature = "pdf")]
    #[error("PDF error: {0}")]
    PdfError(String),

    /// Generic error
    #[error("Error: {0}")]
    Other(String),
}

#[cfg(feature = "web")]
impl From<reqwest::Error> for LoaderError {
    fn from(err: reqwest::Error) -> Self {
        LoaderError::NetworkError(err.to_string())
    }
}

#[cfg(feature = "pdf")]
impl From<lopdf::Error> for LoaderError {
    fn from(err: lopdf::Error) -> Self {
        LoaderError::PdfError(err.to_string())
    }
}

impl From<serde_json::Error> for LoaderError {
    fn from(err: serde_json::Error) -> Self {
        LoaderError::ParseError(err.to_string())
    }
}

#[cfg(feature = "csv")]
impl From<csv::Error> for LoaderError {
    fn from(err: csv::Error) -> Self {
        LoaderError::ParseError(err.to_string())
    }
}
