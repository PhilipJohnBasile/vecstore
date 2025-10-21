//! Auto-tuning HNSW Parameters Example
//!
//! This example demonstrates automatic parameter tuning for HNSW indexes.
//! The auto-tuner finds optimal parameters based on dataset characteristics
//! and performance requirements.
//!
//! ## What Gets Tuned
//!
//! - **M**: Number of connections per node (affects memory and recall)
//! - **ef_construction**: Index build quality (affects construction time)
//! - **ef_search**: Search quality (affects query latency and recall)
//!
//! ## Running
//!
//! ```bash
//! cargo run --example autotuning
//! ```

use vecstore::autotuning::{AutoTuner, PerformanceConstraints, TuningGoal};

fn main() {
    println!("🎯 HNSW Auto-tuning Example\n");

    // ============================================================
    // 1. Basic Auto-tuning
    // ============================================================
    println!("📊 Basic Auto-tuning:");
    println!("═══════════════════════\n");

    // Define dataset characteristics
    let dimension = 384; // e.g., all-MiniLM-L6-v2 embeddings
    let num_vectors = 100_000; // 100k documents

    let tuner = AutoTuner::new(dimension, num_vectors);

    println!("Dataset characteristics:");
    println!("  - Dimension: {}", dimension);
    println!("  - Number of vectors: {}\n", num_vectors);

    // ============================================================
    // 2. Different Tuning Goals
    // ============================================================
    println!("\n🎯 Tuning Goals:");
    println!("═══════════════\n");

    // Fast: Minimize latency (good for real-time search)
    println!("1. FAST (Minimize Latency):");
    let fast = tuner.tune_heuristic(TuningGoal::MinLatency, None).unwrap();
    println!("   M: {}", fast.m);
    println!("   ef_construction: {}", fast.ef_construction);
    println!("   ef_search: {}", fast.ef_search);
    if let Some(recall) = fast.estimated_recall {
        println!("   Estimated recall: {:.1}%", recall * 100.0);
    }
    if let Some(latency) = fast.estimated_latency_ms {
        println!("   Estimated latency: {:.2}ms", latency);
    }
    if let Some(memory) = fast.estimated_memory_mb {
        println!("   Estimated memory: {:.1} MB", memory);
    }

    // Balanced: Good all-around performance
    println!("\n2. BALANCED:");
    let balanced = tuner.tune_heuristic(TuningGoal::Balanced, None).unwrap();
    println!("   M: {}", balanced.m);
    println!("   ef_construction: {}", balanced.ef_construction);
    println!("   ef_search: {}", balanced.ef_search);
    if let Some(recall) = balanced.estimated_recall {
        println!("   Estimated recall: {:.1}%", recall * 100.0);
    }
    if let Some(latency) = balanced.estimated_latency_ms {
        println!("   Estimated latency: {:.2}ms", latency);
    }
    if let Some(memory) = balanced.estimated_memory_mb {
        println!("   Estimated memory: {:.1} MB", memory);
    }

    // Accurate: Maximize recall (good for high-precision search)
    println!("\n3. ACCURATE (Maximize Recall):");
    let accurate = tuner.tune_heuristic(TuningGoal::MaxRecall, None).unwrap();
    println!("   M: {}", accurate.m);
    println!("   ef_construction: {}", accurate.ef_construction);
    println!("   ef_search: {}", accurate.ef_search);
    if let Some(recall) = accurate.estimated_recall {
        println!("   Estimated recall: {:.1}%", recall * 100.0);
    }
    if let Some(latency) = accurate.estimated_latency_ms {
        println!("   Estimated latency: {:.2}ms", latency);
    }
    if let Some(memory) = accurate.estimated_memory_mb {
        println!("   Estimated memory: {:.1} MB", memory);
    }

    // Memory-efficient: Minimize memory usage
    println!("\n4. MEMORY-EFFICIENT:");
    let memory_efficient = tuner.tune_heuristic(TuningGoal::MinMemory, None).unwrap();
    println!("   M: {}", memory_efficient.m);
    println!("   ef_construction: {}", memory_efficient.ef_construction);
    println!("   ef_search: {}", memory_efficient.ef_search);
    if let Some(recall) = memory_efficient.estimated_recall {
        println!("   Estimated recall: {:.1}%", recall * 100.0);
    }
    if let Some(latency) = memory_efficient.estimated_latency_ms {
        println!("   Estimated latency: {:.2}ms", latency);
    }
    if let Some(memory) = memory_efficient.estimated_memory_mb {
        println!("   Estimated memory: {:.1} MB", memory);
    }

    // ============================================================
    // 3. Tuning with Constraints
    // ============================================================
    println!("\n\n⚖️  Tuning with Constraints:");
    println!("═══════════════════════════\n");

    let constraints = PerformanceConstraints {
        min_recall: 0.95,     // Must achieve 95% recall
        max_latency_ms: 5.0,  // Must be under 5ms
        max_memory_mb: 500.0, // Must use less than 500MB
    };

    println!("Constraints:");
    println!("  - Minimum recall: {:.0}%", constraints.min_recall * 100.0);
    println!("  - Maximum latency: {:.1}ms", constraints.max_latency_ms);
    println!("  - Maximum memory: {:.0} MB\n", constraints.max_memory_mb);

    let constrained = tuner
        .tune_heuristic(TuningGoal::Balanced, Some(constraints))
        .unwrap();

    println!("Constrained parameters:");
    println!("  M: {}", constrained.m);
    println!("  ef_construction: {}", constrained.ef_construction);
    println!("  ef_search: {}", constrained.ef_search);

    if let Some(recall) = constrained.estimated_recall {
        println!("  Estimated recall: {:.1}% ✓", recall * 100.0);
    }
    if let Some(latency) = constrained.estimated_latency_ms {
        let status = if latency <= 5.0 { "✓" } else { "✗" };
        println!("  Estimated latency: {:.2}ms {}", latency, status);
    }
    if let Some(memory) = constrained.estimated_memory_mb {
        let status = if memory <= 500.0 { "✓" } else { "✗" };
        println!("  Estimated memory: {:.1} MB {}", memory, status);
    }

    // ============================================================
    // 4. Dataset Size Impact
    // ============================================================
    println!("\n\n📏 Dataset Size Impact:");
    println!("═══════════════════════\n");

    let datasets = vec![
        ("Small (1K vectors)", 1_000),
        ("Medium (50K vectors)", 50_000),
        ("Large (500K vectors)", 500_000),
        ("Very Large (5M vectors)", 5_000_000),
    ];

    for (name, count) in datasets {
        println!("{}:", name);
        let dataset_tuner = AutoTuner::new(384, count);
        let params = dataset_tuner
            .tune_heuristic(TuningGoal::Balanced, None)
            .unwrap();

        println!(
            "  M={}, ef_construction={}, ef_search={}",
            params.m, params.ef_construction, params.ef_search
        );

        if let (Some(latency), Some(memory)) =
            (params.estimated_latency_ms, params.estimated_memory_mb)
        {
            println!("  ~{:.2}ms latency, ~{:.0} MB memory", latency, memory);
        }
        println!();
    }

    // ============================================================
    // 5. Dimension Impact
    // ============================================================
    println!("\n📐 Dimension Impact:");
    println!("══════════════════\n");

    let dimensions = vec![
        ("Small (128-dim)", 128),
        ("Medium (384-dim)", 384),
        ("Large (768-dim)", 768),
        ("Very Large (1536-dim)", 1536),
    ];

    for (name, dim) in dimensions {
        println!("{}:", name);
        let dim_tuner = AutoTuner::new(dim, 100_000);
        let params = dim_tuner
            .tune_heuristic(TuningGoal::Balanced, None)
            .unwrap();

        if let (Some(latency), Some(memory)) =
            (params.estimated_latency_ms, params.estimated_memory_mb)
        {
            println!("  ~{:.2}ms latency, ~{:.0} MB memory", latency, memory);
        }
        println!();
    }

    // ============================================================
    // 6. Recommendations
    // ============================================================
    println!("\n💡 All Recommendations:");
    println!("═══════════════════════\n");

    let recommendations = tuner.recommend_all();

    for (name, params) in &recommendations {
        println!("{}:", name);
        println!(
            "  M={}, ef_construction={}, ef_search={}",
            params.m, params.ef_construction, params.ef_search
        );

        if let Some(recall) = params.estimated_recall {
            println!("  Recall: {:.1}%", recall * 100.0);
        }
        if let Some(latency) = params.estimated_latency_ms {
            println!("  Latency: {:.2}ms", latency);
        }
        if let Some(memory) = params.estimated_memory_mb {
            println!("  Memory: {:.1} MB", memory);
        }
        println!();
    }

    // ============================================================
    // 7. Parameter Explanation
    // ============================================================
    println!("\n📖 Parameter Explanation:");
    println!("═════════════════════════\n");

    let params = tuner.tune_heuristic(TuningGoal::Balanced, None).unwrap();
    let explanation = tuner.explain_params(&params);

    println!("{}", explanation);

    // ============================================================
    // 8. Real-world Scenarios
    // ============================================================
    println!("\n🌍 Real-world Scenarios:");
    println!("═══════════════════════\n");

    // Scenario 1: E-commerce product search
    println!("1. E-commerce Product Search (1M products, fast search required):");
    let ecommerce_tuner = AutoTuner::new(384, 1_000_000);
    let ecommerce_constraints = PerformanceConstraints {
        min_recall: 0.90,      // 90% recall is acceptable
        max_latency_ms: 20.0,  // Under 20ms for good UX
        max_memory_mb: 2000.0, // 2GB memory budget
    };
    let ecommerce = ecommerce_tuner
        .tune_heuristic(TuningGoal::MinLatency, Some(ecommerce_constraints))
        .unwrap();
    println!(
        "   Recommended: M={}, ef_construction={}, ef_search={}",
        ecommerce.m, ecommerce.ef_construction, ecommerce.ef_search
    );
    if let Some(latency) = ecommerce.estimated_latency_ms {
        println!("   Expected latency: {:.2}ms", latency);
    }

    // Scenario 2: Medical document search
    println!("\n2. Medical Document Search (100K docs, high accuracy required):");
    let medical_tuner = AutoTuner::new(768, 100_000);
    let medical_constraints = PerformanceConstraints {
        min_recall: 0.98,     // 98% recall for safety
        max_latency_ms: 50.0, // Latency less critical
        max_memory_mb: 1000.0,
    };
    let medical = medical_tuner
        .tune_heuristic(TuningGoal::MaxRecall, Some(medical_constraints))
        .unwrap();
    println!(
        "   Recommended: M={}, ef_construction={}, ef_search={}",
        medical.m, medical.ef_construction, medical.ef_search
    );
    if let Some(recall) = medical.estimated_recall {
        println!("   Expected recall: {:.1}%", recall * 100.0);
    }

    // Scenario 3: Mobile app (limited memory)
    println!("\n3. Mobile App Vector Search (10K items, memory constrained):");
    let mobile_tuner = AutoTuner::new(128, 10_000);
    let mobile_constraints = PerformanceConstraints {
        min_recall: 0.92,
        max_latency_ms: 10.0,
        max_memory_mb: 50.0, // Very limited memory
    };
    let mobile = mobile_tuner
        .tune_heuristic(TuningGoal::MinMemory, Some(mobile_constraints))
        .unwrap();
    println!(
        "   Recommended: M={}, ef_construction={}, ef_search={}",
        mobile.m, mobile.ef_construction, mobile.ef_search
    );
    if let Some(memory) = mobile.estimated_memory_mb {
        println!("   Expected memory: {:.1} MB", memory);
    }

    // ============================================================
    // 9. Integration with VecStore
    // ============================================================
    println!("\n\n🔧 Integration with VecStore:");
    println!("═════════════════════════════\n");

    println!("Example code:");
    println!(
        r#"
use vecstore::{{VecStoreBuilder, autotuning::{{AutoTuner, TuningGoal}}}};

// Auto-tune parameters
let tuner = AutoTuner::new(384, 100_000);
let params = tuner.tune_heuristic(TuningGoal::Balanced, None)?;

// Build VecStore with tuned parameters
let store = VecStoreBuilder::new()
    .dimension(384)
    .hnsw_m(params.m)
    .hnsw_ef_construction(params.ef_construction)
    .hnsw_ef_search(params.ef_search)
    .build("vectors.db")?;
"#
    );

    println!("\n✅ Auto-tuning example complete!\n");

    println!("💡 Key Takeaways:");
    println!("  - Use MinLatency for real-time search");
    println!("  - Use MaxRecall for high-precision applications");
    println!("  - Use MinMemory for resource-constrained environments");
    println!("  - Use Balanced for general-purpose search");
    println!("  - Always specify constraints for production use");
    println!("  - Larger datasets need lower M to save memory");
    println!("  - Higher dimensions increase latency and memory");
}
