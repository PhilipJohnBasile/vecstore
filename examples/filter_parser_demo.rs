// Filter Parser Demo
//
// Shows how to use the SQL-like filter syntax for much cleaner queries

use std::collections::HashMap;
use vecstore::{Metadata, VecStore};

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  vecstore Filter Parser Demo                ║");
    println!("║  SQL-like syntax for clean metadata filters ║");
    println!("╚══════════════════════════════════════════════╝\n");

    let temp_dir = tempfile::tempdir()?;
    let mut store = VecStore::open(temp_dir.path())?;

    // Insert sample data
    println!("📊 Inserting sample data...\n");

    for i in 0..20 {
        let mut meta = Metadata {
            fields: HashMap::new(),
        };

        meta.fields.insert("id".into(), serde_json::json!(i));
        meta.fields
            .insert("age".into(), serde_json::json!(20 + (i % 40)));
        meta.fields.insert(
            "role".into(),
            serde_json::json!(if i % 3 == 0 {
                "admin"
            } else if i % 3 == 1 {
                "user"
            } else {
                "guest"
            }),
        );
        meta.fields
            .insert("active".into(), serde_json::json!(i % 2 == 0));
        meta.fields.insert(
            "department".into(),
            serde_json::json!(format!("dept{}", i % 5)),
        );
        meta.fields
            .insert("score".into(), serde_json::json!((i * 5) % 100));

        let vector: Vec<f32> = vec![i as f32 / 20.0, 0.5, 0.0];
        store.upsert(format!("user{}", i), vector, meta)?;
    }

    println!("✅ Inserted {} records\n", store.count());

    // Demo 1: Simple equality filter
    println!("═══════════════════════════════════════");
    println!("Demo 1: Simple Equality");
    println!("═══════════════════════════════════════");
    println!("Filter: role = 'admin'\n");

    let results = store.query_with_filter(vec![0.5, 0.5, 0.0], 10, "role = 'admin'")?;

    println!("Found {} results:", results.len());
    for (i, result) in results.iter().take(5).enumerate() {
        let role = result.metadata.fields.get("role").unwrap();
        println!("  {}. {} - role: {}", i + 1, result.id, role);
    }
    println!();

    // Demo 2: Numeric comparison
    println!("═══════════════════════════════════════");
    println!("Demo 2: Numeric Comparison");
    println!("═══════════════════════════════════════");
    println!("Filter: age > 30 AND score >= 50\n");

    let results = store.query_with_filter(vec![0.5, 0.5, 0.0], 10, "age > 30 AND score >= 50")?;

    println!("Found {} results:", results.len());
    for (i, result) in results.iter().take(5).enumerate() {
        let age = result.metadata.fields.get("age").unwrap();
        let score = result.metadata.fields.get("score").unwrap();
        println!(
            "  {}. {} - age: {}, score: {}",
            i + 1,
            result.id,
            age,
            score
        );
    }
    println!();

    // Demo 3: Boolean logic with OR
    println!("═══════════════════════════════════════");
    println!("Demo 3: Boolean OR");
    println!("═══════════════════════════════════════");
    println!("Filter: role = 'admin' OR role = 'user'\n");

    let results =
        store.query_with_filter(vec![0.5, 0.5, 0.0], 10, "role = 'admin' OR role = 'user'")?;

    println!("Found {} results:", results.len());
    for (i, result) in results.iter().take(5).enumerate() {
        let role = result.metadata.fields.get("role").unwrap();
        println!("  {}. {} - role: {}", i + 1, result.id, role);
    }
    println!();

    // Demo 4: NOT operator
    println!("═══════════════════════════════════════");
    println!("Demo 4: NOT Operator");
    println!("═══════════════════════════════════════");
    println!("Filter: NOT role = 'guest'\n");

    let results = store.query_with_filter(vec![0.5, 0.5, 0.0], 10, "NOT role = 'guest'")?;

    println!("Found {} results:", results.len());
    for (i, result) in results.iter().take(5).enumerate() {
        let role = result.metadata.fields.get("role").unwrap();
        println!("  {}. {} - role: {}", i + 1, result.id, role);
    }
    println!();

    // Demo 5: Complex nested expression
    println!("═══════════════════════════════════════");
    println!("Demo 5: Complex Nested Expression");
    println!("═══════════════════════════════════════");
    println!("Filter: (age > 35 AND active = true) OR (role = 'admin' AND score > 70)\n");

    let results = store.query_with_filter(
        vec![0.5, 0.5, 0.0],
        10,
        "(age > 35 AND active = true) OR (role = 'admin' AND score > 70)",
    )?;

    println!("Found {} results:", results.len());
    for (i, result) in results.iter().take(5).enumerate() {
        let age = result.metadata.fields.get("age").unwrap();
        let role = result.metadata.fields.get("role").unwrap();
        let active = result.metadata.fields.get("active").unwrap();
        let score = result.metadata.fields.get("score").unwrap();
        println!(
            "  {}. {} - age: {}, role: {}, active: {}, score: {}",
            i + 1,
            result.id,
            age,
            role,
            active,
            score
        );
    }
    println!();

    // Demo 6: CONTAINS operator
    println!("═══════════════════════════════════════");
    println!("Demo 6: CONTAINS Operator");
    println!("═══════════════════════════════════════");
    println!("Filter: department CONTAINS 'dept2'\n");

    let results =
        store.query_with_filter(vec![0.5, 0.5, 0.0], 10, "department CONTAINS 'dept2'")?;

    println!("Found {} results:", results.len());
    for (i, result) in results.iter().take(5).enumerate() {
        let dept = result.metadata.fields.get("department").unwrap();
        println!("  {}. {} - department: {}", i + 1, result.id, dept);
    }
    println!();

    // Comparison: Old way vs New way
    println!("╔══════════════════════════════════════════════╗");
    println!("║  Comparison: Old Way vs New Way             ║");
    println!("╚══════════════════════════════════════════════╝\n");

    println!("OLD WAY (Manual FilterExpr):");
    println!("────────────────────────────");
    println!(
        r#"FilterExpr::And(vec![
    FilterExpr::Cmp {{
        field: "age".into(),
        op: FilterOp::Gt,
        value: serde_json::json!(18),
    }},
    FilterExpr::Cmp {{
        field: "role".into(),
        op: FilterOp::Eq,
        value: serde_json::json!("admin"),
    }},
])"#
    );
    println!();

    println!("NEW WAY (SQL-like Parser):");
    println!("─────────────────────────");
    println!(
        r#"store.query_with_filter(
    vec![1.0, 0.0, 0.0],
    10,
    "age > 18 AND role = 'admin'"
)?;"#
    );
    println!();

    println!("✨ Much cleaner and easier to read!");
    println!();
    println!("╭───────────────────────────────────────╮");
    println!("│  Demo completed successfully! ✨       │");
    println!("╰───────────────────────────────────────╯");

    Ok(())
}
