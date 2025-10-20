//! Conversational RAG with Memory
//!
//! Demonstrates building a chatbot that maintains conversation history
//! and retrieves relevant context from a knowledge base.
//!
//! Run with: cargo run --example 08_conversation_rag

use anyhow::Result;
use std::collections::HashMap;
use vecstore::{
    rag_utils::{ConversationMemory, PromptTemplate},
    text_splitter::{RecursiveCharacterTextSplitter, TextSplitter},
    Metadata, Query, VecStore,
};

fn main() -> Result<()> {
    println!("💬 Conversational RAG Example\n");

    // Step 1: Build knowledge base
    println!("Step 1: Building knowledge base...");
    let documents = vec![
        "Rust is a systems programming language focused on safety, speed, and concurrency.",
        "The Rust compiler uses a borrow checker to ensure memory safety at compile time.",
        "Cargo is Rust's package manager and build system.",
        "Rust has no garbage collector but ensures memory safety through ownership.",
    ];

    let mut store = VecStore::open("./data/08_conversation_rag")?;
    let splitter = RecursiveCharacterTextSplitter::new(200, 20);

    for (i, doc) in documents.iter().enumerate() {
        let chunks = splitter.split_text(doc)?;
        for (j, chunk) in chunks.into_iter().enumerate() {
            let embedding = mock_embed(&chunk);
            let mut metadata = Metadata {
                fields: HashMap::new(),
            };
            metadata
                .fields
                .insert("text".to_string(), serde_json::json!(chunk));
            store.upsert(format!("doc{}_{}", i, j), embedding, metadata)?;
        }
    }
    println!("   ✓ Knowledge base ready\n");

    // Step 2: Setup conversation memory
    println!("Step 2: Initializing conversation memory...");
    let mut memory = ConversationMemory::new(2048);
    memory.add_message(
        "system",
        "You are a helpful assistant expert in Rust programming.",
    );
    println!("   ✓ Conversation memory ready\n");

    // Step 3: Simulate multi-turn conversation
    println!("Step 3: Simulating conversation...\n");

    let conversation_turns = vec![
        ("User", "What is Rust?"),
        (
            "Assistant",
            "Rust is a systems programming language focused on safety, speed, and concurrency.",
        ),
        ("User", "How does it ensure memory safety?"),
        (
            "Assistant",
            "The Rust compiler uses a borrow checker to ensure memory safety at compile time.",
        ),
        ("User", "What about package management?"),
    ];

    for (role, message) in conversation_turns {
        if role == "User" {
            println!("👤 User: {}", message);

            // Retrieve context
            let query_embedding = mock_embed(message);
            let results = store.query(Query {
                vector: query_embedding,
                k: 2,
                filter: None,
            })?;

            let context: Vec<String> = results
                .iter()
                .filter_map(|r| {
                    r.metadata
                        .fields
                        .get("text")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .collect();

            println!("   📚 Retrieved {} context chunks", context.len());

            // Add to conversation
            memory.add_message("user", message);
        } else {
            println!("🤖 Assistant: {}\n", message);
            memory.add_message("assistant", message);
        }
    }

    // Show conversation history
    println!("\n📜 Conversation History:");
    println!("{}", memory.format_messages());

    println!("\n✅ Conversation RAG Example Complete!");
    println!("\n💡 Key Points:");
    println!("   • ConversationMemory manages message history with token limits");
    println!("   • Auto-trimming keeps conversation within limits");
    println!("   • System messages are preserved");
    println!("   • Context is retrieved for each user query");

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
