//! Advanced RAG Demo
//!
//! Demonstrates the new advanced features:
//! - Markdown-aware text splitting
//! - Code-aware text splitting
//! - Conversation memory
//! - Prompt templates
//!
//! Run with: cargo run --example advanced_rag_demo

use vecstore::rag_utils::{ConversationMemory, PromptTemplate};
use vecstore::text_splitter::{CodeTextSplitter, MarkdownTextSplitter, TextSplitter};

fn main() -> anyhow::Result<()> {
    println!("=== VecStore Advanced RAG Demo ===\n");

    // Demo 1: Markdown-Aware Splitting
    demo_markdown_splitting()?;

    // Demo 2: Code-Aware Splitting
    demo_code_splitting()?;

    // Demo 3: Conversation Memory
    demo_conversation_memory()?;

    // Demo 4: Prompt Templates
    demo_prompt_templates()?;

    println!("\n✅ All demos complete!");
    Ok(())
}

fn demo_markdown_splitting() -> anyhow::Result<()> {
    println!("📝 Demo 1: Markdown-Aware Text Splitting");
    println!("==========================================\n");

    let markdown = r#"
# VecStore Documentation

VecStore is a hybrid vector database for Rust.

## Features

- Fast vector search with HNSW
- Multiple distance metrics
- Namespace support

## Getting Started

First, install VecStore:

```toml
[dependencies]
vecstore = "0.1.0"
```

Then create a collection:

```rust
let mut collection = Collection::new();
```

## Advanced Usage

### Custom Embeddings

You can use your own embedding models.

### Batch Operations

Process multiple documents efficiently.
"#;

    // Simple usage - just works
    let splitter = MarkdownTextSplitter::new(200, 20);
    let chunks = splitter.split_text(markdown)?;

    println!("Split markdown into {} chunks (simple mode):", chunks.len());
    for (i, chunk) in chunks.iter().take(3).enumerate() {
        println!(
            "\nChunk {}:\n{}",
            i + 1,
            chunk.chars().take(100).collect::<String>()
        );
    }

    // Advanced usage - preserve headers
    let splitter = MarkdownTextSplitter::new(200, 20).with_preserve_headers(true);
    let chunks = splitter.split_text(markdown)?;

    println!("\n\nWith header preservation: {} chunks", chunks.len());
    println!("(Headers provide context in each chunk)\n");

    Ok(())
}

fn demo_code_splitting() -> anyhow::Result<()> {
    println!("💻 Demo 2: Code-Aware Text Splitting");
    println!("=====================================\n");

    let rust_code = r#"
use vecstore::Collection;

pub fn create_embeddings(texts: &[String]) -> Vec<Vec<f32>> {
    texts.iter()
        .map(|text| embed_text(text))
        .collect()
}

pub fn embed_text(text: &str) -> Vec<f32> {
    // Simple embedding logic
    vec![0.0; 384]
}

pub struct VectorStore {
    collection: Collection,
    dimension: usize,
}

impl VectorStore {
    pub fn new(dimension: usize) -> Self {
        Self {
            collection: Collection::new(),
            dimension,
        }
    }

    pub fn add(&mut self, id: &str, vector: Vec<f32>) {
        self.collection.add(id, vector, Default::default());
    }
}
"#;

    // Simple usage - language-agnostic
    let splitter = CodeTextSplitter::new(300, 30);
    let chunks = splitter.split_text(rust_code)?;

    println!("Split code into {} chunks (generic mode):", chunks.len());

    // Advanced usage - Rust-specific
    let splitter = CodeTextSplitter::new(300, 30).with_language("rust");
    let chunks = splitter.split_text(rust_code)?;

    println!("With Rust-specific splitting: {} chunks", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("\nChunk {} ({} chars):", i + 1, chunk.len());
        let preview = chunk.lines().take(3).collect::<Vec<_>>().join("\n");
        println!("{}", preview);
    }

    println!();
    Ok(())
}

fn demo_conversation_memory() -> anyhow::Result<()> {
    println!("💬 Demo 3: Conversation Memory");
    println!("===============================\n");

    // Simple usage - just works with default token estimator
    let mut memory = ConversationMemory::new(500);

    memory.add_message(
        "system",
        "You are a helpful assistant for VecStore documentation.",
    );
    memory.add_message("user", "How do I create a vector collection?");
    memory.add_message(
        "assistant",
        "You can create a collection with Collection::new()",
    );
    memory.add_message("user", "What about adding vectors?");
    memory.add_message(
        "assistant",
        "Use the add() method with an ID, vector, and metadata",
    );

    println!("Conversation has {} messages", memory.get_messages().len());
    println!("Total tokens: ~{}", memory.token_count());

    println!("\nFormatted conversation:");
    println!("{}", memory.format_messages());

    // Add many more messages to test trimming
    println!("\n\nAdding more messages to test auto-trimming...");
    for i in 0..10 {
        memory.add_message(
            "user",
            &format!("Question {} with some additional content", i),
        );
        memory.add_message(
            "assistant",
            &format!("Answer {} with detailed explanation", i),
        );
    }

    println!(
        "After many messages: {} remain (auto-trimmed)",
        memory.get_messages().len()
    );
    println!(
        "System message preserved: {}",
        memory.get_messages().iter().any(|m| m.role == "system")
    );

    println!();
    Ok(())
}

fn demo_prompt_templates() -> anyhow::Result<()> {
    println!("📋 Demo 4: Prompt Templates");
    println!("============================\n");

    // Simple RAG prompt template
    let template = PromptTemplate::new(
        r#"Answer the following question using the provided context.

Question: {question}

Context:
{context}

Instructions: {instructions}

Answer:"#,
    )
    .with_default("instructions", "Provide a clear and concise answer.");

    // Fill and render
    let prompt = template
        .fill("question", "What is VecStore?")
        .fill("context", "VecStore is a hybrid vector database for Rust. It provides fast similarity search with multiple distance metrics.")
        .render();

    println!("Generated prompt:");
    println!("{}", prompt);

    // Another example with custom instructions
    let prompt2 = PromptTemplate::new(
        "Question: {question}\nContext: {context}\nInstructions: {instructions}\n",
    )
    .with_default("instructions", "Answer briefly.")
    .fill("question", "How do I install it?")
    .fill("context", "Add vecstore to your Cargo.toml dependencies.")
    .fill("instructions", "Provide installation steps.") // Override default
    .render();

    println!("\n\nSecond prompt with custom instructions:");
    println!("{}", prompt2);

    println!();
    Ok(())
}
