//! Distributed Tracing Demo
//!
//! This example demonstrates VecStore's built-in distributed tracing capabilities.
//! Shows how to enable tracing, configure output formats, and integrate with
//! OpenTelemetry backends like Jaeger, Zipkin, or Honeycomb.
//!
//! Run with:
//! ```bash
//! # Console output with default formatting
//! cargo run --example distributed_tracing_demo
//!
//! # JSON output for production logging
//! RUST_LOG=vecstore=debug cargo run --example distributed_tracing_demo -- --json
//!
//! # Trace only specific operations
//! RUST_LOG=vecstore::store=trace cargo run --example distributed_tracing_demo
//! ```
//!
//! To integrate with Jaeger (requires Jaeger running):
//! ```bash
//! docker run -d -p6831:6831/udp -p6832:6832/udp -p16686:16686 jaegertracing/all-in-one:latest
//! cargo run --example distributed_tracing_demo
//! # View traces at http://localhost:16686
//! ```

use anyhow::Result;
use std::env;
use vecstore::telemetry::{init_telemetry, init_telemetry_json, record_event, traced_sync};
use vecstore::{HybridQuery, Metadata, Query, VecStore};

fn main() -> Result<()> {
    // Check if JSON output is requested
    let use_json = env::args().any(|arg| arg == "--json");

    // Initialize tracing
    if use_json {
        println!("🔍 Initializing JSON tracing for production...");
        init_telemetry_json()?;
    } else {
        println!("🔍 Initializing console tracing...");
        init_telemetry()?;
    }

    println!("\n=== VecStore Distributed Tracing Demo ===\n");

    // Demo 1: Basic operations with automatic tracing
    demo_basic_tracing()?;

    // Demo 2: Hybrid search with tracing
    demo_hybrid_search_tracing()?;

    // Demo 3: Custom span creation
    demo_custom_spans()?;

    // Demo 4: Performance monitoring
    demo_performance_monitoring()?;

    println!("\n✅ All tracing demos complete!");
    println!("\n💡 Tips:");
    println!("   - Set RUST_LOG=vecstore=trace for detailed traces");
    println!("   - Use --json flag for structured logging");
    println!("   - Integrate with OpenTelemetry for production observability");

    Ok(())
}

/// Demo 1: Basic operations with automatic span creation
fn demo_basic_tracing() -> Result<()> {
    println!("📊 Demo 1: Basic Operations with Automatic Tracing");
    println!("   All query() and upsert() calls automatically create spans\n");

    let mut store = VecStore::open("tracing_demo.db")?;

    // Upsert operations are automatically instrumented
    tracing::info!("Starting data ingestion phase");

    for i in 0..10 {
        let vector = vec![i as f32 * 0.1, (i + 1) as f32 * 0.1, (i + 2) as f32 * 0.1];
        let mut fields = std::collections::HashMap::new();
        fields.insert("index".to_string(), serde_json::json!(i));
        fields.insert("category".to_string(), serde_json::json!("demo"));

        let metadata = Metadata { fields };

        // This upsert is automatically traced with dimension information
        store.upsert(format!("doc{}", i), vector, metadata)?;
    }

    record_event("data_ingestion_complete");

    // Query operations are automatically instrumented
    tracing::info!("Starting query phase");

    let query = Query::new(vec![0.5, 0.6, 0.7]).with_limit(5);

    // This query is automatically traced with k and filter information
    let results = store.query(query)?;

    println!("   Found {} results (traced automatically)", results.len());
    record_event("query_complete");

    Ok(())
}

/// Demo 2: Hybrid search with tracing
fn demo_hybrid_search_tracing() -> Result<()> {
    println!("\n📊 Demo 2: Hybrid Search Tracing");
    println!("   Hybrid queries show vector + keyword weighting in traces\n");

    let mut store = VecStore::open("tracing_demo.db")?;

    // First upsert some documents
    let docs = vec![
        (
            "doc1",
            vec![0.9, 0.8, 0.7],
            "machine learning and artificial intelligence",
        ),
        ("doc2", vec![0.8, 0.9, 0.6], "deep learning neural networks"),
        ("doc3", vec![0.7, 0.7, 0.8], "natural language processing"),
    ];

    for (id, vector, text) in docs {
        let mut fields = std::collections::HashMap::new();
        fields.insert("text".to_string(), serde_json::json!(text));
        let metadata = Metadata { fields };
        store.upsert(id.to_string(), vector, metadata)?;
        store.index_text(id, text)?;
    }

    // Hybrid query is automatically traced with alpha parameter
    let query = HybridQuery {
        vector: vec![0.1, 0.2, 0.3],
        keywords: "machine learning".to_string(),
        k: 3,
        filter: None,
        alpha: 0.7, // 70% vector, 30% keyword - shows in trace
    };

    tracing::info!("Executing hybrid query");
    let results = store.hybrid_query(query)?;

    println!("   Found {} hybrid results", results.len());
    println!("   Check traces to see vector/keyword weighting (alpha=0.7)");

    Ok(())
}

/// Demo 3: Custom span creation for complex operations
fn demo_custom_spans() -> Result<()> {
    println!("\n📊 Demo 3: Custom Span Creation");
    println!("   Use traced_sync() for custom business logic\n");

    // Wrap custom logic in traced spans
    let result = traced_sync("data_preprocessing", || {
        tracing::info!("Processing batch of documents");

        // Simulate preprocessing
        let documents = vec![
            "Document about AI",
            "Document about ML",
            "Document about NLP",
        ];

        for (i, doc) in documents.iter().enumerate() {
            tracing::debug!(doc_index = i, doc_length = doc.len(), "Processing document");
        }

        record_event("preprocessing_complete");
        Ok::<_, anyhow::Error>(documents.len())
    })?;

    println!("   Processed {} documents in custom span", result);

    Ok(())
}

/// Demo 4: Performance monitoring with spans
fn demo_performance_monitoring() -> Result<()> {
    println!("\n📊 Demo 4: Performance Monitoring");
    println!("   Spans automatically track operation duration\n");

    let store = VecStore::open("tracing_demo.db")?;

    // Run multiple queries and monitor performance
    for i in 0..5 {
        let query_vec = vec![i as f32 * 0.1, (i + 1) as f32 * 0.1, (i + 2) as f32 * 0.1];

        let query = Query::new(query_vec)
            .with_limit(10)
            .with_filter("category = 'demo'");

        // Each query creates a span with timing
        tracing::info!(query_iteration = i, "Executing monitored query");
        let results = store.query(query)?;

        tracing::info!(
            query_iteration = i,
            result_count = results.len(),
            "Query completed"
        );
    }

    println!("   Executed 5 queries - check traces for timing information");
    println!("   Slow queries will be visible in span durations");

    Ok(())
}
