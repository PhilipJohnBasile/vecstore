// Query explain demonstration
//
// Shows how query_explain provides detailed insights into search results

use vecstore::store::parse_filter;
use vecstore::store::{Query, VecStore};

fn main() -> anyhow::Result<()> {
    println!("🔍 Query Explain Demo\n");

    // Create a temporary store
    let temp_dir = tempfile::tempdir()?;
    let mut store = VecStore::open(temp_dir.path())?;

    // Add some sample data
    println!("📝 Adding sample vectors...");
    let mut metadata1 = vecstore::store::Metadata {
        fields: std::collections::HashMap::new(),
    };
    metadata1
        .fields
        .insert("category".to_string(), serde_json::json!("tech"));
    metadata1
        .fields
        .insert("score".to_string(), serde_json::json!(95));

    let mut metadata2 = vecstore::store::Metadata {
        fields: std::collections::HashMap::new(),
    };
    metadata2
        .fields
        .insert("category".to_string(), serde_json::json!("tech"));
    metadata2
        .fields
        .insert("score".to_string(), serde_json::json!(75));

    let mut metadata3 = vecstore::store::Metadata {
        fields: std::collections::HashMap::new(),
    };
    metadata3
        .fields
        .insert("category".to_string(), serde_json::json!("science"));
    metadata3
        .fields
        .insert("score".to_string(), serde_json::json!(85));

    store.upsert("doc1".to_string(), vec![1.0, 0.0, 0.0], metadata1)?;
    store.upsert("doc2".to_string(), vec![0.9, 0.1, 0.0], metadata2)?;
    store.upsert("doc3".to_string(), vec![0.0, 1.0, 0.0], metadata3)?;

    // Query without filter
    println!("\n=== Query 1: No filters ===");
    let query1 = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 2,
        filter: None,
    };

    let results = store.query_explain(query1)?;
    for result in results {
        println!("\n📄 Result: {}", result.id);
        println!("   Score: {:.4}", result.score);
        println!("   Rank: #{}", result.explanation.rank);
        println!("   Distance Metric: {}", result.explanation.distance_metric);
        println!("   Explanation: {}", result.explanation.explanation_text);
        if let Some(stats) = result.explanation.graph_stats {
            println!("   Graph Stats:");
            println!(
                "     - Distance calculations: {}",
                stats.distance_calculations
            );
            println!("     - Nodes visited: {}", stats.nodes_visited);
        }
    }

    // Query with filter
    println!("\n\n=== Query 2: With filter (category = 'tech') ===");
    let filter = parse_filter("category = 'tech'")?;
    let query2 = Query {
        vector: vec![1.0, 0.0, 0.0],
        k: 3,
        filter: Some(filter),
    };

    let results = store.query_explain(query2)?;
    for result in results {
        println!("\n📄 Result: {}", result.id);
        println!("   Score: {:.4}", result.score);
        println!("   Rank: #{}", result.explanation.rank);
        println!("   Filter Passed: {}", result.explanation.filter_passed);
        if let Some(filter_details) = result.explanation.filter_details {
            println!("   Filter Details:");
            println!("     - Expression: {}", filter_details.filter_expr);
            println!("     - Matched: {:?}", filter_details.matched_conditions);
            if !filter_details.failed_conditions.is_empty() {
                println!("     - Failed: {:?}", filter_details.failed_conditions);
            }
        }
        println!("   Explanation: {}", result.explanation.explanation_text);
    }

    println!("\n✅ Query explain demo complete!");
    Ok(())
}
