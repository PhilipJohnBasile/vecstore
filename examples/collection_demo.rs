//! Collection API Demo
//!
//! This example demonstrates VecStore's collection-based API,
//! which provides a ChromaDB/Qdrant-like interface for managing
//! multiple isolated vector stores.
//!
//! Collections make it easy to:
//! - Organize vectors by domain (documents, users, products, etc.)
//! - Isolate data with independent quotas and configurations
//! - Manage multiple vector stores from a single database
//!
//! Run with: cargo run --example collection_demo

use std::collections::HashMap;
use vecstore::{CollectionConfig, Distance, Metadata, Query, VecDatabase};

fn main() -> anyhow::Result<()> {
    println!("=== VecStore Collection API Demo ===\n");

    // ========== 1. Create Database ==========
    println!("1. Creating Database\n");

    let mut db = VecDatabase::open("./demo_collections")?;
    println!("   ✓ Database created at ./demo_collections\n");

    // ========== 2. Create Collections ==========
    println!("2. Creating Collections\n");

    // Simple collection with defaults
    let mut documents = db.create_collection("documents")?;
    println!("   ✓ Created 'documents' collection");

    // Collection with custom configuration
    let config = CollectionConfig::default()
        .with_description("User profile embeddings")
        .with_distance(Distance::Cosine)
        .with_max_vectors(10_000);

    let mut users = db.create_collection_with_config("users", config)?;
    println!("   ✓ Created 'users' collection (cosine similarity, max 10K vectors)");

    let mut products = db.create_collection("products")?;
    println!("   ✓ Created 'products' collection\n");

    // ========== 3. Insert Data into Collections ==========
    println!("3. Inserting Data\n");

    // Insert documents
    let mut doc_meta = Metadata {
        fields: HashMap::new(),
    };
    doc_meta
        .fields
        .insert("title".into(), serde_json::json!("Rust Programming Guide"));
    doc_meta
        .fields
        .insert("category".into(), serde_json::json!("tech"));

    documents.upsert("doc1".into(), vec![0.1, 0.2, 0.3, 0.4], doc_meta.clone())?;
    println!("   ✓ Inserted document 'doc1'");

    doc_meta
        .fields
        .insert("title".into(), serde_json::json!("Machine Learning Basics"));
    documents.upsert("doc2".into(), vec![0.2, 0.3, 0.4, 0.5], doc_meta)?;
    println!("   ✓ Inserted document 'doc2'");

    // Insert users
    let mut user_meta = Metadata {
        fields: HashMap::new(),
    };
    user_meta
        .fields
        .insert("name".into(), serde_json::json!("Alice"));
    user_meta.fields.insert("age".into(), serde_json::json!(28));

    users.upsert("user1".into(), vec![0.8, 0.1, 0.2], user_meta.clone())?;
    println!("   ✓ Inserted user 'user1'");

    user_meta
        .fields
        .insert("name".into(), serde_json::json!("Bob"));
    user_meta.fields.insert("age".into(), serde_json::json!(34));
    users.upsert("user2".into(), vec![0.7, 0.2, 0.3], user_meta)?;
    println!("   ✓ Inserted user 'user2'");

    // Insert products
    let mut prod_meta = Metadata {
        fields: HashMap::new(),
    };
    prod_meta
        .fields
        .insert("name".into(), serde_json::json!("Laptop"));
    prod_meta
        .fields
        .insert("price".into(), serde_json::json!(1299.99));

    products.upsert("prod1".into(), vec![0.5, 0.5, 0.0], prod_meta.clone())?;
    println!("   ✓ Inserted product 'prod1'");

    prod_meta
        .fields
        .insert("name".into(), serde_json::json!("Mouse"));
    prod_meta
        .fields
        .insert("price".into(), serde_json::json!(29.99));
    products.upsert("prod2".into(), vec![0.6, 0.4, 0.0], prod_meta)?;
    println!("   ✓ Inserted product 'prod2'\n");

    // ========== 4. Query Collections ==========
    println!("4. Querying Collections\n");

    // Query documents
    let query = Query {
        vector: vec![0.15, 0.25, 0.35, 0.45],
        k: 2,
        filter: None,
    };

    let results = documents.query(query)?;
    println!("   Documents search results:");
    for result in &results {
        println!("     - {}: score = {:.4}", result.id, result.score);
    }
    println!();

    // Query users
    let query = Query {
        vector: vec![0.75, 0.15, 0.25],
        k: 2,
        filter: None,
    };

    let results = users.query(query)?;
    println!("   Users search results:");
    for result in &results {
        println!("     - {}: score = {:.4}", result.id, result.score);
    }
    println!();

    // ========== 5. Collection Statistics ==========
    println!("5. Collection Statistics\n");

    let doc_stats = documents.stats()?;
    println!("   Documents collection:");
    println!("     - Total vectors: {}", doc_stats.vector_count);
    println!("     - Dimension: {}", doc_stats.dimension);
    println!(
        "     - Distance metric: {:?}\n",
        documents.distance_metric()
    );

    let user_stats = users.stats()?;
    println!("   Users collection:");
    println!("     - Total vectors: {}", user_stats.vector_count);
    println!("     - Dimension: {}", user_stats.dimension);
    println!("     - Distance metric: {:?}\n", users.distance_metric());

    // ========== 6. List All Collections ==========
    println!("6. Listing All Collections\n");

    let collections = db.list_collections()?;
    println!("   Active collections:");
    for name in &collections {
        let coll = db.get_collection(name)?.unwrap();
        let stats = coll.stats()?;
        println!("     - {}: {} vectors", name, stats.vector_count);
    }
    println!();

    // ========== 7. Collection Isolation ==========
    println!("7. Collection Isolation\n");

    println!("   Each collection is independent:");
    println!("     - Documents: {} vectors", documents.count()?);
    println!("     - Users: {} vectors", users.count()?);
    println!("     - Products: {} vectors", products.count()?);
    println!();

    println!("   Deleting from 'documents' doesn't affect 'users':");
    documents.delete("doc1")?;
    println!("     - Documents after delete: {}", documents.count()?);
    println!("     - Users (unchanged): {}", users.count()?);
    println!();

    // ========== 8. Delete Collection ==========
    println!("8. Deleting Collections\n");

    db.delete_collection("products")?;
    println!("   ✓ Deleted 'products' collection");

    let collections = db.list_collections()?;
    println!("   Remaining collections: {:?}\n", collections);

    // ========== Summary ==========
    println!("=== Summary ===");
    println!();
    println!("Collections provide a high-level API for VecStore:");
    println!("  ✓ Organize vectors by domain");
    println!("  ✓ Independent configurations and quotas");
    println!("  ✓ ChromaDB/Qdrant-like ergonomics");
    println!("  ✓ Built on VecStore's namespace system");
    println!();
    println!("Compare to simple VecStore:");
    println!("  - Simple: VecStore::open() for single-purpose use");
    println!("  - Powerful: VecDatabase for multi-collection apps");
    println!();
    println!("Useful for: RAG apps, multi-tenant systems, organized data");

    // Cleanup
    std::fs::remove_dir_all("./demo_collections").ok();

    Ok(())
}
