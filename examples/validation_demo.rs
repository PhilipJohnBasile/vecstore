//! Vector Validation and Quality Checks Demo
//!
//! Demonstrates comprehensive vector validation before insertion.

use vecstore::*;

fn main() -> anyhow::Result<()> {
    println!("\n🔍 Vector Validation and Quality Checks Demo\n");
    println!("{}", "=".repeat(70));

    // Test vectors with various issues
    let test_vectors = vec![
        ("Valid vector", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        ("Contains NaN", vec![1.0, f32::NAN, 3.0, 4.0, 5.0]),
        ("Contains Infinity", vec![1.0, 2.0, f32::INFINITY, 4.0, 5.0]),
        ("Zero vector", vec![0.0, 0.0, 0.0, 0.0, 0.0]),
        ("All identical", vec![5.0, 5.0, 5.0, 5.0, 5.0]),
        ("High sparsity", vec![1.0, 0.0, 0.0, 0.0, 0.0]),
        ("Low variance", vec![1.0, 1.1, 1.0, 1.1, 1.0]),
        ("Large magnitude", vec![100.0, 200.0, 300.0, 400.0, 500.0]),
    ];

    // Test 1: Default Validation
    println!("\n[1/5] Default Validation (Standard Strictness)");
    println!("{}", "-".repeat(70));

    let validator = VectorValidator::default();

    for (name, vector) in &test_vectors {
        let result = validator.validate(vector);
        print!("{:<20} ", name);
        match result {
            ValidationResult::Valid => println!("✓ VALID"),
            ValidationResult::Warning(warnings) => {
                println!("⚠️  WARNING ({} issues)", warnings.len());
                for warning in warnings {
                    println!("     {:?}", warning);
                }
            },
            ValidationResult::Invalid(errors) => {
                println!("❌ INVALID ({} errors)", errors.len());
                for error in errors {
                    println!("     {:?}", error);
                }
            },
        }
    }

    // Test 2: Strict Validation
    println!("\n[2/5] Strict Validation");
    println!("{}", "-".repeat(70));

    let strict_validator = VectorValidator::strict();

    for (name, vector) in &test_vectors {
        let result = strict_validator.validate(vector);
        print!("{:<20} ", name);
        match result {
            ValidationResult::Valid => println!("✓ VALID"),
            ValidationResult::Warning(warnings) => {
                println!("⚠️  {} warnings", warnings.len());
            },
            ValidationResult::Invalid(errors) => {
                println!("❌ {} errors", errors.len());
            },
        }
    }

    // Test 3: Lenient Validation with Auto-Fix
    println!("\n[3/5] Auto-Fix Invalid Vectors");
    println!("{}", "-".repeat(70));

    let lenient_validator = VectorValidator::lenient();

    for (name, vector) in &test_vectors {
        println!("\n{}", name);
        println!("  Original: {:?}", &vector[..vector.len().min(5)]);

        match lenient_validator.auto_fix(vector) {
            Ok(fixed) => {
                println!("  Fixed:    {:?}", &fixed[..fixed.len().min(5)]);
                let result = lenient_validator.validate(&fixed);
                println!(
                    "  Result:   {:?}",
                    if result.is_valid() {
                        "✓ VALID"
                    } else if result.is_warning() {
                        "⚠️  WARNING"
                    } else {
                        "❌ INVALID"
                    }
                );
            },
            Err(e) => println!("  Error:    {}", e),
        }
    }

    // Test 4: Quality Metrics
    println!("\n[4/5] Vector Quality Metrics");
    println!("{}", "-".repeat(70));

    println!(
        "\n{:<20} {:<10} {:<10} {:<10} {:<10}",
        "Vector", "Quality", "Magnitude", "Variance", "Sparsity"
    );
    println!("{}", "-".repeat(70));

    for (name, vector) in &test_vectors {
        let metrics = validator.compute_quality(vector);
        println!(
            "{:<20} {:<10.3} {:<10.3} {:<10.3} {:<10.3}",
            name, metrics.quality_score, metrics.magnitude, metrics.variance, metrics.sparsity
        );
    }

    // Test 5: Batch Validation
    println!("\n[5/5] Batch Validation Statistics");
    println!("{}", "-".repeat(70));

    let vectors: Vec<Vec<f32>> = test_vectors.iter().map(|(_, v)| v.clone()).collect();

    let strict_stats = strict_validator.batch_statistics(&vectors);
    let lenient_stats = lenient_validator.batch_statistics(&vectors);

    println!("\nStrict Validation:");
    println!("  Total vectors:    {}", strict_stats.total_vectors);
    println!(
        "  Valid:            {} ({:.1}%)",
        strict_stats.valid_count,
        strict_stats.valid_count as f32 / strict_stats.total_vectors as f32 * 100.0
    );
    println!(
        "  Warnings:         {} ({:.1}%)",
        strict_stats.warning_count,
        strict_stats.warning_count as f32 / strict_stats.total_vectors as f32 * 100.0
    );
    println!(
        "  Invalid:          {} ({:.1}%)",
        strict_stats.invalid_count,
        strict_stats.invalid_count as f32 / strict_stats.total_vectors as f32 * 100.0
    );
    println!("  Average quality:  {:.3}", strict_stats.average_quality);

    println!("\nLenient Validation:");
    println!("  Total vectors:    {}", lenient_stats.total_vectors);
    println!(
        "  Valid:            {} ({:.1}%)",
        lenient_stats.valid_count,
        lenient_stats.valid_count as f32 / lenient_stats.total_vectors as f32 * 100.0
    );
    println!(
        "  Warnings:         {} ({:.1}%)",
        lenient_stats.warning_count,
        lenient_stats.warning_count as f32 / lenient_stats.total_vectors as f32 * 100.0
    );
    println!(
        "  Invalid:          {} ({:.1}%)",
        lenient_stats.invalid_count,
        lenient_stats.invalid_count as f32 / lenient_stats.total_vectors as f32 * 100.0
    );
    println!("  Average quality:  {:.3}", lenient_stats.average_quality);

    // Test 6: Normalization
    println!("\n[6/6] Vector Normalization");
    println!("{}", "-".repeat(70));

    let unnormalized = vec![3.0, 4.0]; // magnitude = 5.0
    println!("\nOriginal vector:  {:?}", unnormalized);
    println!(
        "Magnitude:        {:.3}",
        validator.compute_magnitude(&unnormalized)
    );

    let normalized = validator.normalize(&unnormalized);
    println!("\nNormalized:       {:?}", normalized);
    println!(
        "New magnitude:    {:.3}",
        validator.compute_magnitude(&normalized)
    );

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("📊 Demo Complete!");
    println!("{}", "=".repeat(70));

    println!("\n✨ Key Features Demonstrated:");
    println!("  ✓ NaN and infinity detection");
    println!("  ✓ Zero vector detection");
    println!("  ✓ Dimension validation");
    println!("  ✓ Magnitude range checks");
    println!("  ✓ Quality metrics computation");
    println!("  ✓ Auto-fix for invalid vectors");
    println!("  ✓ Batch validation statistics");
    println!("  ✓ Vector normalization");

    println!("\n💡 Validation Strictness Levels:");
    println!("  • Strict:      Reject any suspicious vectors");
    println!("  • Standard:    Block invalid, warn on suspicious");
    println!("  • Lenient:     Allow suspicious, block only critical");
    println!("  • Permissive:  Allow almost everything");

    println!("\n🎯 Use Cases:");
    println!("  • Data quality assurance before insertion");
    println!("  • Detecting corrupted embeddings");
    println!("  • Automatic data cleaning pipelines");
    println!("  • Quality monitoring and alerting");

    println!();

    Ok(())
}
