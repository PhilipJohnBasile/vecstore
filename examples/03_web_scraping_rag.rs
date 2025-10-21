//! Web Scraping RAG Example
//!
//! Demonstrates building a RAG system from web content:
//! - Simulated web scraping
//! - HTML content extraction
//! - URL metadata tracking
//! - Source attribution
//!
//! Run with: cargo run --example 03_web_scraping_rag

use anyhow::Result;
use std::collections::HashMap;
use vecstore::{
    text_splitter::{RecursiveCharacterTextSplitter, TextSplitter},
    Metadata, Query, VecStore,
};

fn main() -> Result<()> {
    println!("🌐 Web Scraping RAG Example\n");

    // Simulate web scraping (in production, use reqwest + scraper)
    println!("Step 1: Scraping web pages...");
    let web_pages = scrape_websites()?;
    println!("   ✓ Scraped {} pages\n", web_pages.len());

    // Initialize store
    println!("Step 2: Setting up vector store...");
    let mut store = VecStore::open("./data/03_web_scraping")?;
    let splitter = RecursiveCharacterTextSplitter::new(400, 50);
    println!("   ✓ Store ready\n");

    // Process and index web content
    println!("Step 3: Processing and indexing web content...");
    let mut total_chunks = 0;

    for page in &web_pages {
        let chunks = splitter.split_text(&page.content)?;

        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_id = format!(
                "{}_{}",
                page.url.replace("https://", "").replace("/", "_"),
                i
            );
            let embedding = mock_embed(chunk);

            let mut metadata = Metadata {
                fields: HashMap::new(),
            };
            metadata
                .fields
                .insert("text".to_string(), serde_json::json!(chunk));
            metadata
                .fields
                .insert("url".to_string(), serde_json::json!(page.url));
            metadata
                .fields
                .insert("title".to_string(), serde_json::json!(page.title));
            metadata
                .fields
                .insert("scraped_at".to_string(), serde_json::json!(page.timestamp));
            metadata
                .fields
                .insert("chunk_index".to_string(), serde_json::json!(i));

            store.upsert(chunk_id, embedding, metadata)?;
            total_chunks += 1;
        }
    }

    println!(
        "   ✓ Indexed {} chunks from {} pages\n",
        total_chunks,
        web_pages.len()
    );

    // Query with source attribution
    println!("Step 4: Querying with source attribution...\n");
    let queries = vec![
        "What is VecStore?",
        "How do I use HNSW indexing?",
        "What are the performance benchmarks?",
    ];

    for query_text in &queries {
        println!("🔍 Query: {}", query_text);

        let results = store.query(Query {
            vector: mock_embed(query_text),
            k: 3,
            filter: None,
        })?;

        for (i, result) in results.iter().enumerate() {
            let text = result
                .metadata
                .fields
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("N/A");
            let url = result
                .metadata
                .fields
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or("N/A");
            let title = result
                .metadata
                .fields
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("N/A");

            println!("   {}. Score: {:.3}", i + 1, result.score);
            println!("      Source: {} ({})", title, url);
            println!("      {}\n", &text[..text.len().min(150)]);
        }
        println!();
    }

    println!("✅ Web Scraping RAG Example Complete!");
    println!("\n💡 Production Tips:");
    println!("   • Use reqwest for HTTP requests");
    println!("   • Use scraper or select.rs for HTML parsing");
    println!("   • Add rate limiting to respect robots.txt");
    println!("   • Store scraped_at timestamp for freshness tracking");
    println!("   • Consider incremental updates for large sites");
    println!("   • Filter out navigation, ads, and boilerplate content");

    Ok(())
}

#[derive(Debug)]
struct WebPage {
    url: String,
    title: String,
    content: String,
    timestamp: u64,
}

fn scrape_websites() -> Result<Vec<WebPage>> {
    // In production: Use reqwest + scraper
    // Example:
    //   let html = reqwest::blocking::get(url)?.text()?;
    //   let document = scraper::Html::parse_document(&html);
    //   let selector = scraper::Selector::parse("article").unwrap();
    //   let content = document.select(&selector).map(|el| el.text().collect()).collect();

    Ok(vec![
        WebPage {
            url: "https://docs.vecstore.io/intro".to_string(),
            title: "VecStore Introduction".to_string(),
            content: "VecStore is a Rust-based vector database that focuses on local deployments. \
                     It provides HNSW indexing for approximate nearest neighbour search, \
                     persistence to disk, and supporting tooling for RAG applications. \
                     Rust and SIMD implementations generally run faster than pure Python alternatives.".to_string(),
            timestamp: 1704067200, // 2024-01-01
        },
        WebPage {
            url: "https://docs.vecstore.io/features/hnsw".to_string(),
            title: "HNSW Indexing Guide".to_string(),
            content: "HNSW (Hierarchical Navigable Small World) is an efficient algorithm for \
                     approximate nearest neighbor search. VecStore implements HNSW with SIMD \
                     acceleration for optimal performance. Configuration options include M (number \
                     of connections), ef_construction (search depth during construction), and \
                     ef_search (search depth during queries).".to_string(),
            timestamp: 1704153600, // 2024-01-02
        },
        WebPage {
            url: "https://docs.vecstore.io/benchmarks".to_string(),
            title: "Performance Benchmarks".to_string(),
            content: "VecStore benchmarks will be republished with reproducible harnesses. \
                     Run the Criterion benches inside the repository to gather latency and throughput \
                     numbers on your hardware. Product quantization and SIMD acceleration are available \
                     to help balance memory usage and speed. Hybrid search combines dense and sparse \
                     vectors for improved relevance.".to_string(),
            timestamp: 1704240000, // 2024-01-03
        },
    ])
}

fn mock_embed(text: &str) -> Vec<f32> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut embedding = vec![0.0; 384];
    for (i, word) in words.iter().enumerate() {
        embedding[(word.len() * (i + 1)) % 384] += 1.0;
    }
    let mag: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 0.0 {
        for val in &mut embedding {
            *val /= mag;
        }
    }
    embedding
}
