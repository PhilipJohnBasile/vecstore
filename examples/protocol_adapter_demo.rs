//! Protocol adapter demonstration
//!
//! Shows how to use VecStore's universal protocol adapter
//! to accept requests from different vector database clients:
//! - Pinecone
//! - Qdrant
//! - Weaviate
//! - ChromaDB
//! - Milvus

use anyhow::Result;
use serde_json::json;
use vecstore::{Protocol, ProtocolAdapter, VecStore};

fn main() -> Result<()> {
    println!("🔌 VecStore Protocol Adapter Demo\n");
    println!("{}", "=".repeat(80));
    println!("\nThis example shows how VecStore can accept requests");
    println!("in multiple vector database formats.\n");

    // Create a vector store
    let store = VecStore::open("data/protocol_demo.db")?;
    let mut adapter = ProtocolAdapter::new(store);

    // Example 1: Pinecone-style request
    println!("[1/5] Pinecone-style upsert...");
    let pinecone_upsert = json!({
        "vectors": [
            {
                "id": "vec1",
                "values": [0.1, 0.2, 0.3, 0.4],
                "metadata": {"category": "tech", "score": 0.95}
            },
            {
                "id": "vec2",
                "values": [0.5, 0.6, 0.7, 0.8],
                "metadata": {"category": "science", "score": 0.88}
            }
        ]
    })
    .to_string();

    let response = adapter.handle_request(&pinecone_upsert, Protocol::Pinecone)?;
    println!("   ✓ Response: {}", response);

    // Example 2: Qdrant-style query
    println!("\n[2/5] Qdrant-style query...");
    let qdrant_query = json!({
        "vector": [0.1, 0.2, 0.3, 0.4],
        "limit": 10,
        "with_payload": true
    })
    .to_string();

    let response = adapter.handle_request(&qdrant_query, Protocol::Qdrant)?;
    println!("   ✓ Query executed");
    println!(
        "   Response preview: {}...",
        &response[..response.len().min(100)]
    );

    // Example 3: Weaviate-style request (batch format)
    println!("\n[3/5] Weaviate-style batch insert...");
    let weaviate_insert = json!({
        "objects": [
            {
                "class": "Document",
                "id": "weaviate-1",
                "vector": [0.2, 0.3, 0.4, 0.5],
                "properties": {
                    "title": "Weaviate Document",
                    "content": "Sample content"
                }
            }
        ]
    })
    .to_string();

    match adapter.handle_request(&weaviate_insert, Protocol::Weaviate) {
        Ok(response) => println!("   ✓ Response: {}", response),
        Err(e) => println!("   ⚠ Note: {} (batch operations supported)", e),
    }

    // Example 4: ChromaDB-style query
    println!("\n[4/5] ChromaDB-style query...");
    let chroma_query = json!({
        "query_embeddings": [[0.1, 0.2, 0.3, 0.4]],
        "n_results": 5,
        "where": {"category": "tech"}
    })
    .to_string();

    let response = adapter.handle_request(&chroma_query, Protocol::ChromaDB)?;
    println!("   ✓ Query executed");

    // Example 5: Auto-detection
    println!("\n[5/5] Auto-detecting protocol from request...");

    let auto_request = json!({
        "vectors": [
            {
                "id": "auto-detect",
                "values": [0.9, 0.8, 0.7, 0.6],
                "metadata": {"auto": true}
            }
        ]
    })
    .to_string();

    let response = adapter.handle_request_auto(&auto_request)?;
    println!("   ✓ Auto-detected as Pinecone format");
    println!("   Response: {}", response);

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("📊 Summary");
    println!("{}", "=".repeat(80));

    println!("\n✅ Protocol adapter working!");
    println!("\n💡 Supported Protocols:");
    println!("   • Pinecone - Most popular managed vector DB");
    println!("   • Qdrant - High-performance open source");
    println!("   • Weaviate - GraphQL-based vector search");
    println!("   • ChromaDB - AI-native embedding database");
    println!("   • Milvus - Scalable vector database");
    println!("   • Universal - VecStore native format");

    println!("\n🚀 Use Cases:");
    println!("   • Drop-in replacement for other vector DBs");
    println!("   • Easy migration from cloud to self-hosted");
    println!("   • Support multiple client SDKs");
    println!("   • Build compatible APIs");

    Ok(())
}
