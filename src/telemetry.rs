//! Distributed tracing and observability for VecStore
//!
//! This module provides comprehensive tracing instrumentation for VecStore operations,
//! enabling performance monitoring, debugging, and distributed tracing across services.
//!
//! VecStore uses the `tracing` crate which is backend-agnostic and works with:
//! - OpenTelemetry (Jaeger, Zipkin, Honeycomb, etc.)
//! - Console/stdout logging
//! - Custom exporters
//!
//! # Features
//! - Automatic span creation for all major operations (query, upsert, hybrid_query)
//! - Performance timing for slow query detection
//! - Error tracking and context
//! - Configurable sampling rates
//!
//! # Example
//! ```no_run
//! use vecstore::telemetry::init_telemetry;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Initialize with default console logging
//! init_telemetry()?;
//! # Ok(())
//! # }
//! ```
//!
//! # OpenTelemetry Integration
//! For production deployments with OpenTelemetry, use `tracing-opentelemetry`:
//!
//! ```ignore
//! use tracing_subscriber::layer::SubscriberExt;
//! use tracing_opentelemetry::OpenTelemetryLayer;
//!
//! let tracer = opentelemetry_jaeger::new_pipeline()
//!     .with_service_name("vecstore")
//!     .install_simple()?;
//!
//! let telemetry = OpenTelemetryLayer::new(tracer);
//! let subscriber = tracing_subscriber::registry().with(telemetry);
//! tracing::subscriber::set_global_default(subscriber)?;
//! ```

use anyhow::Result;
use tracing_subscriber::{fmt, EnvFilter};

/// Initialize basic tracing with console output
///
/// This sets up tracing with formatted console output and environment-based filtering.
/// For production deployments, integrate with OpenTelemetry or other backends.
///
/// # Example
/// ```no_run
/// use vecstore::telemetry::init_telemetry;
///
/// # fn example() -> anyhow::Result<()> {
/// init_telemetry()?;
/// # Ok(())
/// # }
/// ```
pub fn init_telemetry() -> Result<()> {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("vecstore=info"));

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .init();

    tracing::info!("Tracing initialized");
    Ok(())
}

/// Initialize telemetry with JSON formatting
///
/// Useful for structured logging in production environments.
///
/// # Example
/// ```no_run
/// use vecstore::telemetry::init_telemetry_json;
///
/// # fn example() -> anyhow::Result<()> {
/// init_telemetry_json()?;
/// # Ok(())
/// # }
/// ```
pub fn init_telemetry_json() -> Result<()> {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("vecstore=info"));

    fmt()
        .json()
        .with_env_filter(filter)
        .with_current_span(true)
        .with_span_list(true)
        .init();

    tracing::info!("JSON tracing initialized");
    Ok(())
}

/// Trace an async operation with automatic span creation
///
/// # Example
/// ```no_run
/// use vecstore::telemetry::traced_async;
///
/// # async fn example() -> anyhow::Result<()> {
/// let result = traced_async("my_operation", async {
///     // Your async logic here
///     Ok::<_, anyhow::Error>(42)
/// }).await?;
/// # Ok(())
/// # }
/// ```
pub async fn traced_async<F, T>(operation: &str, future: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    let span = tracing::info_span!("async_operation", operation = operation);
    let _guard = span.enter();
    future.await
}

/// Trace a synchronous operation with automatic span creation
///
/// # Example
/// ```no_run
/// use vecstore::telemetry::traced_sync;
///
/// # fn example() -> anyhow::Result<()> {
/// let result = traced_sync("my_operation", || {
///     // Your logic here
///     Ok::<_, anyhow::Error>(42)
/// })?;
/// # Ok(())
/// # }
/// ```
pub fn traced_sync<F, T>(operation: &str, f: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    let span = tracing::info_span!("sync_operation", operation = operation);
    let _guard = span.enter();
    f()
}

/// Record an event in the current span
///
/// # Example
/// ```no_run
/// use vecstore::telemetry::record_event;
///
/// # fn example() {
/// record_event("cache_hit");
/// # }
/// ```
pub fn record_event(event: &str) {
    tracing::info!(event = event);
}

/// Record an error in the current span
///
/// # Example
/// ```no_run
/// use vecstore::telemetry::record_error;
/// use anyhow::anyhow;
///
/// # fn example() {
/// let error = anyhow!("Something went wrong");
/// record_error(&error);
/// # }
/// ```
pub fn record_error(error: &dyn std::error::Error) {
    tracing::error!(error = %error);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traced_sync() {
        let result = traced_sync("test_operation", || Ok::<_, anyhow::Error>(42));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_traced_async() {
        let result =
            traced_async("test_async_operation", async { Ok::<_, anyhow::Error>(42) }).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_record_event() {
        // Should not panic
        record_event("test_event");
    }

    #[test]
    fn test_record_error() {
        let error = anyhow::anyhow!("test error");
        // Should not panic
        record_error(error.as_ref());
    }
}
