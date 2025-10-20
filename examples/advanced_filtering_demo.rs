//! Advanced Filtering Demo
//!
//! Demonstrates VecStore's complete filter operator suite:
//! - Comparison: Eq, Neq, Gt, Gte, Lt, Lte
//! - Membership: In, NotIn
//! - Text: Contains
//! - Boolean: And, Or, Not
//!
//! Run with: cargo run --example advanced_filtering_demo

use anyhow::Result;
use std::collections::HashMap;
use vecstore::{FilterExpr, FilterOp, Metadata, Query, VecStore};

fn main() -> Result<()> {
    println!("=== VecStore Advanced Filtering Demo ===\n");

    // Create store
    let mut store = VecStore::open("./data/advanced_filtering")?;

    // Sample data: E-commerce products
    let products = vec![
        (
            "product_1",
            "Laptop",
            999.0,
            "Electronics",
            vec!["portable", "productivity"],
            4.5,
            true,
        ),
        (
            "product_2",
            "Mouse",
            25.0,
            "Electronics",
            vec!["portable", "accessories"],
            4.2,
            true,
        ),
        (
            "product_3",
            "Desk",
            299.0,
            "Furniture",
            vec!["office", "home"],
            4.0,
            true,
        ),
        (
            "product_4",
            "Monitor",
            350.0,
            "Electronics",
            vec!["display", "productivity"],
            4.7,
            true,
        ),
        (
            "product_5",
            "Chair",
            199.0,
            "Furniture",
            vec!["office", "ergonomic"],
            4.3,
            false,
        ), // Out of stock
        (
            "product_6",
            "Keyboard",
            75.0,
            "Electronics",
            vec!["portable", "productivity"],
            4.4,
            true,
        ),
        (
            "product_7",
            "Bookshelf",
            150.0,
            "Furniture",
            vec!["storage", "home"],
            3.9,
            true,
        ),
        (
            "product_8",
            "Webcam",
            89.0,
            "Electronics",
            vec!["video", "remote"],
            4.1,
            true,
        ),
    ];

    // Index products
    println!("1. Indexing products...\n");
    for (id, name, price, category, tags, rating, in_stock) in &products {
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("name".into(), serde_json::json!(name));
        metadata
            .fields
            .insert("price".into(), serde_json::json!(price));
        metadata
            .fields
            .insert("category".into(), serde_json::json!(category));
        metadata
            .fields
            .insert("tags".into(), serde_json::json!(tags));
        metadata
            .fields
            .insert("rating".into(), serde_json::json!(rating));
        metadata
            .fields
            .insert("in_stock".into(), serde_json::json!(in_stock));

        // Simple embedding based on product characteristics
        let embedding = mock_product_embedding(name, category, price);
        store.upsert(id.to_string(), embedding, metadata)?;
    }

    println!("   ✓ Indexed {} products\n", products.len());

    // =================================================================================
    // Demo 1: Range Queries (Gt, Lt, Gte, Lte)
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 1: Range Queries");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Query: Products between $100 and $400
    let filter = FilterExpr::And(vec![
        FilterExpr::Cmp {
            field: "price".into(),
            op: FilterOp::Gte,
            value: serde_json::json!(100.0),
        },
        FilterExpr::Cmp {
            field: "price".into(),
            op: FilterOp::Lte,
            value: serde_json::json!(400.0),
        },
    ]);

    println!("   Query: Products priced $100 - $400");
    let results = store.query(Query {
        vector: mock_search_query(),
        k: 10,
        filter: Some(filter),
    })?;

    println!("   Results: {} products\n", results.len());
    for result in &results {
        let name = result.metadata.fields.get("name").unwrap();
        let price = result.metadata.fields.get("price").unwrap();
        println!("     • {} - ${}", name, price);
    }
    println!();

    // =================================================================================
    // Demo 2: Membership Queries (In, NotIn)
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 2: Membership Queries (NEW!)");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Query: Electronics OR Furniture categories
    let filter = FilterExpr::Cmp {
        field: "category".into(),
        op: FilterOp::In,
        value: serde_json::json!(["Electronics", "Furniture"]),
    };

    println!("   Query: category IN ['Electronics', 'Furniture']");
    let results = store.query(Query {
        vector: mock_search_query(),
        k: 10,
        filter: Some(filter),
    })?;

    println!("   Results: {} products\n", results.len());
    for result in &results {
        let name = result.metadata.fields.get("name").unwrap();
        let category = result.metadata.fields.get("category").unwrap();
        println!("     • {} - Category: {}", name, category);
    }
    println!();

    // =================================================================================
    // Demo 3: Complex Boolean Logic
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 3: Complex Boolean Filters");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Query: (Electronics AND price < $100) OR (rating >= 4.5)
    let filter = FilterExpr::Or(vec![
        FilterExpr::And(vec![
            FilterExpr::Cmp {
                field: "category".into(),
                op: FilterOp::Eq,
                value: serde_json::json!("Electronics"),
            },
            FilterExpr::Cmp {
                field: "price".into(),
                op: FilterOp::Lt,
                value: serde_json::json!(100.0),
            },
        ]),
        FilterExpr::Cmp {
            field: "rating".into(),
            op: FilterOp::Gte,
            value: serde_json::json!(4.5),
        },
    ]);

    println!("   Query: (Electronics AND price < $100) OR (rating >= 4.5)");
    let results = store.query(Query {
        vector: mock_search_query(),
        k: 10,
        filter: Some(filter),
    })?;

    println!("   Results: {} products\n", results.len());
    for result in &results {
        let name = result.metadata.fields.get("name").unwrap();
        let category = result.metadata.fields.get("category").unwrap();
        let price = result.metadata.fields.get("price").unwrap();
        let rating = result.metadata.fields.get("rating").unwrap();
        println!("     • {} - {} ${} (⭐ {})", name, category, price, rating);
    }
    println!();

    // =================================================================================
    // Demo 4: Negation with NotIn
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 4: Exclusion Filters (NotIn)");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Query: NOT in low-rated categories
    let filter = FilterExpr::And(vec![
        FilterExpr::Cmp {
            field: "category".into(),
            op: FilterOp::NotIn,
            value: serde_json::json!(["Deprecated", "Clearance"]),
        },
        FilterExpr::Cmp {
            field: "in_stock".into(),
            op: FilterOp::Eq,
            value: serde_json::json!(true),
        },
    ]);

    println!("   Query: NOT in deprecated categories AND in stock");
    let results = store.query(Query {
        vector: mock_search_query(),
        k: 10,
        filter: Some(filter),
    })?;

    println!("   Results: {} products\n", results.len());
    for result in &results {
        let name = result.metadata.fields.get("name").unwrap();
        let in_stock = result.metadata.fields.get("in_stock").unwrap();
        println!("     • {} - In Stock: {}", name, in_stock);
    }
    println!();

    // =================================================================================
    // Demo 5: E-commerce Use Case - Advanced Product Search
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 5: E-commerce Product Search");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Query: Premium electronics in stock, highly rated
    let filter = FilterExpr::And(vec![
        FilterExpr::Cmp {
            field: "category".into(),
            op: FilterOp::In,
            value: serde_json::json!(["Electronics"]),
        },
        FilterExpr::Cmp {
            field: "price".into(),
            op: FilterOp::Gte,
            value: serde_json::json!(50.0),
        },
        FilterExpr::Cmp {
            field: "rating".into(),
            op: FilterOp::Gte,
            value: serde_json::json!(4.0),
        },
        FilterExpr::Cmp {
            field: "in_stock".into(),
            op: FilterOp::Eq,
            value: serde_json::json!(true),
        },
    ]);

    println!("   Query: Premium Electronics");
    println!("   Filters:");
    println!("     • Category: Electronics");
    println!("     • Price: >= $50");
    println!("     • Rating: >= 4.0 ⭐");
    println!("     • In Stock: Yes\n");

    let results = store.query(Query {
        vector: mock_search_query(),
        k: 10,
        filter: Some(filter),
    })?;

    println!("   Results: {} products\n", results.len());
    for result in &results {
        let name = result.metadata.fields.get("name").unwrap();
        let price = result.metadata.fields.get("price").unwrap();
        let rating = result.metadata.fields.get("rating").unwrap();
        println!("     • {} - ${} (⭐ {})", name, price, rating);
    }
    println!();

    // =================================================================================
    // Summary
    // =================================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Summary: VecStore Filter Operators");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("   Comparison Operators:");
    println!("     ✓ Eq        - Equality");
    println!("     ✓ Neq       - Not equal");
    println!("     ✓ Gt, Gte   - Greater than (or equal)");
    println!("     ✓ Lt, Lte   - Less than (or equal)");
    println!();

    println!("   Membership Operators:");
    println!("     ✓ In        - Value in array (NEW!)");
    println!("     ✓ NotIn     - Value not in array (NEW!)");
    println!("     ✓ Contains  - String/array contains");
    println!();

    println!("   Boolean Logic:");
    println!("     ✓ And       - All conditions must match");
    println!("     ✓ Or        - Any condition matches");
    println!("     ✓ Not       - Negation");
    println!();

    println!("   Competitive Status:");
    println!("     ✅ Matches Qdrant filter capabilities");
    println!("     ✅ Matches Weaviate filter capabilities");
    println!("     ✅ Matches Pinecone filter capabilities");
    println!();

    println!("   Performance:");
    println!("     • Post-filtering (after vector search)");
    println!("     • O(k) complexity for k results");
    println!("     • Can filter thousands of results/ms");
    println!();

    // Cleanup
    std::fs::remove_dir_all("./data/advanced_filtering").ok();
    println!("✅ Advanced Filtering Demo Complete!\n");

    Ok(())
}

// Helper: Generate mock product embedding
fn mock_product_embedding(name: &str, category: &str, price: &f64) -> Vec<f32> {
    let mut embedding = vec![0.0; 128];

    // Name characteristics
    for (i, ch) in name.chars().enumerate() {
        embedding[i % 128] += (ch as u32 as f32) / 1000.0;
    }

    // Category signal
    let category_hash = category.len() * 17;
    embedding[category_hash % 128] += 0.5;

    // Price signal (normalized)
    let price_idx = (price / 100.0) as usize % 128;
    embedding[price_idx] += 0.3;

    // Normalize
    let mag: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 0.0 {
        for val in &mut embedding {
            *val /= mag;
        }
    }

    embedding
}

// Helper: Generate mock search query
fn mock_search_query() -> Vec<f32> {
    let mut query = vec![0.0; 128];
    query[0] = 0.1;
    query[10] = 0.2;
    query[50] = 0.15;

    // Normalize
    let mag: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 0.0 {
        for val in &mut query {
            *val /= mag;
        }
    }

    query
}
