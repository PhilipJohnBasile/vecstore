use std::collections::HashMap;
use vecstore::{FilterExpr, FilterOp, Metadata, Query, VecStore};

fn main() -> anyhow::Result<()> {
    // Create a temporary directory for this example
    let temp_dir = tempfile::tempdir()?;
    let data_path = temp_dir.path().join("data");

    let mut store = VecStore::open(&data_path)?;

    // Insert some records
    let mut meta1 = Metadata {
        fields: HashMap::new(),
    };
    meta1
        .fields
        .insert("topic".into(), serde_json::json!("rust"));
    meta1
        .fields
        .insert("difficulty".into(), serde_json::json!(5));
    store.upsert("doc1".into(), vec![0.1, 0.2, 0.9], meta1)?;

    let mut meta2 = Metadata {
        fields: HashMap::new(),
    };
    meta2
        .fields
        .insert("topic".into(), serde_json::json!("python"));
    meta2
        .fields
        .insert("difficulty".into(), serde_json::json!(3));
    store.upsert("doc2".into(), vec![0.8, 0.1, 0.2], meta2)?;

    let mut meta3 = Metadata {
        fields: HashMap::new(),
    };
    meta3
        .fields
        .insert("topic".into(), serde_json::json!("rust"));
    meta3
        .fields
        .insert("difficulty".into(), serde_json::json!(8));
    store.upsert("doc3".into(), vec![0.15, 0.25, 0.85], meta3)?;

    println!("Inserted {} records", store.count());

    // Query with filter
    let query = Query {
        vector: vec![0.1, 0.2, 0.8],
        k: 3,
        filter: Some(FilterExpr::Cmp {
            field: "topic".into(),
            op: FilterOp::Eq,
            value: serde_json::json!("rust"),
        }),
    };

    let results = store.query(query)?;
    println!("\nQuery results (filtered by topic='rust'):");
    for (i, neighbor) in results.iter().enumerate() {
        println!("{}. {} (score: {:.4})", i + 1, neighbor.id, neighbor.score);
        println!("   {:?}", neighbor.metadata.fields);
    }

    // Query with complex filter
    let complex_query = Query {
        vector: vec![0.1, 0.2, 0.8],
        k: 3,
        filter: Some(FilterExpr::And(vec![
            FilterExpr::Cmp {
                field: "topic".into(),
                op: FilterOp::Eq,
                value: serde_json::json!("rust"),
            },
            FilterExpr::Cmp {
                field: "difficulty".into(),
                op: FilterOp::Lte,
                value: serde_json::json!(6),
            },
        ])),
    };

    let complex_results = store.query(complex_query)?;
    println!("\nComplex query results (topic='rust' AND difficulty<=6):");
    for (i, neighbor) in complex_results.iter().enumerate() {
        println!("{}. {} (score: {:.4})", i + 1, neighbor.id, neighbor.score);
        println!("   {:?}", neighbor.metadata.fields);
    }

    // Save and reload
    store.save()?;
    println!("\nStore saved to disk");

    drop(store);

    let reloaded = VecStore::open(&data_path)?;
    println!("Store reloaded, count: {}", reloaded.count());

    Ok(())
}
