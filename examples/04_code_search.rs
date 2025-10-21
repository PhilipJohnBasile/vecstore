//! Code Search RAG Example
//!
//! Demonstrates semantic code search using:
//! - CodeTextSplitter for intelligent code chunking
//! - Function/class-aware splitting
//! - Code repository search
//!
//! Run with: cargo run --example 04_code_search

use anyhow::Result;
use std::collections::HashMap;
use vecstore::{
    text_splitter::{CodeTextSplitter, TextSplitter},
    Metadata, Query, VecStore,
};

fn main() -> Result<()> {
    println!("💻 Code Search RAG Example\n");

    // Sample Rust code
    let rust_code = r#"
use std::collections::HashMap;

pub struct VecStore {
    vectors: HashMap<String, Vec<f32>>,
    dimension: usize,
}

impl VecStore {
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: HashMap::new(),
            dimension,
        }
    }

    pub fn upsert(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!("Dimension mismatch"));
        }
        self.vectors.insert(id, vector);
        Ok(())
    }

    pub fn query(&self, query: Vec<f32>, k: usize) -> Vec<String> {
        // HNSW search implementation
        Vec::new()
    }
}
"#;

    println!("Step 1: Splitting code with CodeTextSplitter...");
    let splitter = CodeTextSplitter::new(300, 30).with_language("rust");
    let chunks = splitter.split_text(rust_code)?;
    println!("   ✓ Split into {} code chunks\n", chunks.len());

    // Store code chunks
    println!("Step 2: Indexing code chunks...");
    let mut store = VecStore::open("./data/04_code_search")?;

    for (i, chunk) in chunks.iter().enumerate() {
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("code".to_string(), serde_json::json!(chunk));
        metadata
            .fields
            .insert("language".to_string(), serde_json::json!("rust"));

        store.upsert(format!("chunk_{}", i), mock_embed(chunk), metadata)?;
    }
    println!("   ✓ Indexed {} code chunks\n", chunks.len());

    // Search code
    println!("Step 3: Searching code...\n");
    let searches = vec![
        "How to create a new VecStore?",
        "How to insert vectors?",
        "How to query vectors?",
    ];

    for search_query in searches {
        println!("🔍 Search: {}", search_query);
        let results = store.query(Query {
            vector: mock_embed(search_query),
            k: 2,
            filter: None,
        })?;

        for (i, result) in results.iter().enumerate() {
            let code = result
                .metadata
                .fields
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or("N/A");
            println!("   {}. Score: {:.3}", i + 1, result.score);
            println!("      {}\n", &code[..code.len().min(150)]);
        }
    }

    println!("✅ Code Search Example Complete!");
    println!("\n💡 Key Features:");
    println!("   • CodeTextSplitter respects function/class boundaries");
    println!("   • Language-specific parsing (Rust, Python, JS, etc.)");
    println!("   • Maintains code context and syntax");
    println!("   • Useful for code documentation and search workflows");

    Ok(())
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
