// Hybrid Search Demo - THE Killer Feature for RAG!
//
// This example demonstrates combining vector similarity with keyword search (BM25)
// for the best of both worlds in retrieval-augmented generation.

use std::collections::HashMap;
use vecstore::{HybridQuery, Metadata, VecStore};

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                                                              ║");
    println!("║      vecstore Hybrid Search - Vector + Keyword Search       ║");
    println!("║                                                              ║");
    println!("║           THE Killer Feature for RAG Applications!          ║");
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let temp_dir = tempfile::tempdir()?;
    let mut store = VecStore::open(temp_dir.path())?;

    // Create sample documents about programming languages
    println!("📚 Creating Knowledge Base");
    println!("──────────────────────────────────────────────────────────");

    let documents = vec![
        (
            "rust_overview",
            vec![0.9, 0.1, 0.0], // High on "systems"
            "Rust is a systems programming language focused on safety and performance",
            "systems",
        ),
        (
            "rust_memory",
            vec![0.8, 0.0, 0.2], // High on "systems", some "web"
            "Rust provides memory safety without garbage collection using ownership",
            "systems",
        ),
        (
            "python_intro",
            vec![0.1, 0.9, 0.0], // High on "scripting"
            "Python is a high-level programming language known for readability",
            "scripting",
        ),
        (
            "python_web",
            vec![0.0, 0.5, 0.5], // Medium on "scripting" and "web"
            "Python is popular for web development with frameworks like Django and Flask",
            "scripting",
        ),
        (
            "javascript_web",
            vec![0.0, 0.1, 0.9], // High on "web"
            "JavaScript is the language of the web, running in browsers and Node.js",
            "web",
        ),
        (
            "go_concurrency",
            vec![0.7, 0.0, 0.3], // High on "systems", some "web"
            "Go is designed for concurrent programming with goroutines and channels",
            "systems",
        ),
        (
            "typescript_types",
            vec![0.2, 0.2, 0.6], // High on "web"
            "TypeScript adds static types to JavaScript for safer web development",
            "web",
        ),
        (
            "rust_async",
            vec![0.8, 0.0, 0.2], // High on "systems"
            "Rust has powerful async/await support for concurrent programming",
            "systems",
        ),
    ];

    for (id, vector, text, category) in &documents {
        let mut meta = Metadata {
            fields: HashMap::new(),
        };
        meta.fields
            .insert("category".into(), serde_json::json!(category));
        meta.fields.insert("text".into(), serde_json::json!(text));

        // Insert vector
        store.upsert(id.to_string(), vector.clone(), meta)?;

        // Index text for keyword search
        store.index_text(id, *text)?;
    }

    println!("✅ Indexed {} documents", documents.len());
    println!("   • Vector embeddings for semantic search");
    println!("   • Text content for keyword search (BM25)");
    println!();

    // Demo 1: Pure Vector Search (alpha = 1.0)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 1: Pure Vector Search (Semantic Only)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Query: [0.8, 0.0, 0.2] (systems programming)");
    println!("Alpha: 1.0 (100% vector, 0% keyword)\n");

    let results = store.hybrid_query(HybridQuery {
        vector: vec![0.8, 0.0, 0.2],
        keywords: String::new(),
        k: 5,
        filter: None,
        alpha: 1.0,
    })?;

    for (i, result) in results.iter().enumerate() {
        let text = result.metadata.fields.get("text").unwrap();
        println!("{}. {} (score: {:.3})", i + 1, result.id, result.score);
        println!("   {}\n", text);
    }

    // Demo 2: Pure Keyword Search (alpha = 0.0)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 2: Pure Keyword Search (BM25 Only)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Keywords: \"concurrent programming\"");
    println!("Alpha: 0.0 (0% vector, 100% keyword)\n");

    let results = store.hybrid_query(HybridQuery {
        vector: vec![0.5, 0.5, 0.0], // Ignored when alpha = 0
        keywords: "concurrent programming".into(),
        k: 5,
        filter: None,
        alpha: 0.0,
    })?;

    for (i, result) in results.iter().enumerate() {
        let text = result.metadata.fields.get("text").unwrap();
        println!("{}. {} (score: {:.3})", i + 1, result.id, result.score);
        println!("   {}\n", text);
    }

    // Demo 3: Hybrid Search - Best of Both Worlds!
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 3: Hybrid Search - THE Sweet Spot! 🎯");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Vector: [0.8, 0.0, 0.2] (systems programming)");
    println!("Keywords: \"memory safety\"");
    println!("Alpha: 0.7 (70% vector, 30% keyword) ← Recommended for RAG\n");

    let results = store.hybrid_query(HybridQuery {
        vector: vec![0.8, 0.0, 0.2],
        keywords: "memory safety".into(),
        k: 5,
        filter: None,
        alpha: 0.7,
    })?;

    for (i, result) in results.iter().enumerate() {
        let text = result.metadata.fields.get("text").unwrap();
        println!("{}. {} (score: {:.3})", i + 1, result.id, result.score);
        println!("   {}\n", text);
    }

    // Demo 4: Hybrid Search with Filters
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 4: Hybrid Search + Metadata Filters");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Vector: [0.5, 0.5, 0.0]");
    println!("Keywords: \"web development\"");
    println!("Filter: category = 'web'");
    println!("Alpha: 0.6\n");

    let results = store.hybrid_query(HybridQuery {
        vector: vec![0.5, 0.5, 0.0],
        keywords: "web development".into(),
        k: 5,
        filter: Some(vecstore::parse_filter("category = 'web'")?),
        alpha: 0.6,
    })?;

    for (i, result) in results.iter().enumerate() {
        let text = result.metadata.fields.get("text").unwrap();
        let category = result.metadata.fields.get("category").unwrap();
        println!(
            "{}. {} [{}] (score: {:.3})",
            i + 1,
            result.id,
            category,
            result.score
        );
        println!("   {}\n", text);
    }

    // Demo 5: Comparing Alpha Values
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 5: Effect of Alpha Parameter");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Query: \"Rust async programming\"");
    println!("Vector: [0.8, 0.0, 0.2]\n");

    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0] {
        println!(
            "Alpha = {} ({:.0}% vector, {:.0}% keyword):",
            alpha,
            alpha * 100.0,
            (1.0 - alpha) * 100.0
        );

        let results = store.hybrid_query(HybridQuery {
            vector: vec![0.8, 0.0, 0.2],
            keywords: "async programming".into(),
            k: 3,
            filter: None,
            alpha,
        })?;

        print!("   Top 3: ");
        for (i, result) in results.iter().enumerate() {
            print!("{}", result.id);
            if i < results.len() - 1 {
                print!(", ");
            }
        }
        println!("\n");
    }

    // Summary and Best Practices
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                  Summary & Best Practices                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("✨ What is Hybrid Search?");
    println!("   Combines:");
    println!("   • Vector similarity (semantic understanding)");
    println!("   • Keyword search (exact term matching via BM25)");
    println!();
    println!("🎯 When to use Hybrid Search:");
    println!("   ✅ RAG applications (retrieve relevant context)");
    println!("   ✅ Document search (user queries may have specific terms)");
    println!("   ✅ Question answering");
    println!("   ✅ Code search (function names + semantic understanding)");
    println!();
    println!("📊 Recommended Alpha Values:");
    println!("   • 0.7 - General RAG (70% semantic, 30% keyword)");
    println!("   • 0.8 - Embedding-heavy use cases");
    println!("   • 0.5 - Balanced search");
    println!("   • 0.3 - Keyword-heavy queries");
    println!();
    println!("🔥 Why This is THE Killer Feature:");
    println!("   1. Best of both worlds - semantic + exact matching");
    println!("   2. No other lightweight vector DB has this!");
    println!("   3. Critical for production RAG systems");
    println!("   4. Better than pure vector or pure keyword alone");
    println!();
    println!("💡 Pro Tips:");
    println!("   • Index text for all documents you want to keyword search");
    println!("   • Use filters to narrow by metadata first");
    println!("   • Tune alpha based on your use case");
    println!("   • Monitor which retrieves better results for your queries");
    println!();
    println!("📚 Example Use Case - RAG Chatbot:");
    println!("   User: \"How does Rust handle memory safety?\"");
    println!("   1. Embed question → vector");
    println!("   2. Extract keywords → \"Rust\" \"memory safety\"");
    println!("   3. Hybrid search with alpha=0.7");
    println!("   4. Get most relevant docs (semantic + keyword match)");
    println!("   5. Feed to LLM as context");
    println!();

    println!("╭────────────────────────────────────────────────────────────╮");
    println!("│  Hybrid Search: Making RAG Actually Work in Production! ✨ │");
    println!("╰────────────────────────────────────────────────────────────╯");

    Ok(())
}
