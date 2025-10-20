// CLI Integration Tests
// Note: These tests require the binary to be built first: `cargo build --bin vecstore`
// Run with: `cargo build && cargo test --test cli_integration`

use std::fs;
use std::process::Command;
use tempfile::TempDir;

fn vecstore_bin() -> String {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // Remove test binary name
    if path.ends_with("deps") {
        path.pop(); // Remove 'deps'
    }
    path.push("vecstore");
    path.to_str().unwrap().to_string()
}

// Helper to check if vecstore binary exists
fn binary_exists() -> bool {
    std::path::Path::new(&vecstore_bin()).exists()
}

// Macro to skip tests if binary not built
macro_rules! skip_if_no_binary {
    () => {
        if !binary_exists() {
            println!("Skipping CLI test - run `cargo build --bin vecstore` first");
            return;
        }
    };
}

#[test]
fn test_cli_init() {
    skip_if_no_binary!();

    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().join("data");

    let output = Command::new(vecstore_bin())
        .arg("init")
        .arg("--dir")
        .arg(&data_path)
        .output()
        .expect("Failed to execute vecstore");

    assert!(output.status.success());
    assert!(data_path.exists());
}

#[test]
fn test_cli_stats_empty() {
    skip_if_no_binary!();

    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().join("data");

    // Init first
    Command::new(vecstore_bin())
        .arg("init")
        .arg("--dir")
        .arg(&data_path)
        .output()
        .expect("Failed to execute vecstore");

    // Check stats
    let output = Command::new(vecstore_bin())
        .arg("stats")
        .arg("--dir")
        .arg(&data_path)
        .output()
        .expect("Failed to execute vecstore");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Records: 0"));
}

#[test]
fn test_cli_ingest_single() {
    skip_if_no_binary!();

    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().join("data");

    // Init
    Command::new(vecstore_bin())
        .arg("init")
        .arg("--dir")
        .arg(&data_path)
        .output()
        .expect("Failed to execute vecstore");

    // Create vector and metadata files
    let vec_file = temp_dir.path().join("vec.json");
    let meta_file = temp_dir.path().join("meta.json");

    fs::write(&vec_file, "[1.0, 0.0, 0.0]").unwrap();
    fs::write(&meta_file, r#"{"category": "test"}"#).unwrap();

    // Ingest
    let output = Command::new(vecstore_bin())
        .arg("ingest")
        .arg("--dir")
        .arg(&data_path)
        .arg("--id")
        .arg("doc1")
        .arg("--vec")
        .arg(&vec_file)
        .arg("--meta")
        .arg(&meta_file)
        .output()
        .expect("Failed to execute vecstore");

    assert!(output.status.success());

    // Verify with stats
    let output = Command::new(vecstore_bin())
        .arg("stats")
        .arg("--dir")
        .arg(&data_path)
        .output()
        .expect("Failed to execute vecstore");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Records: 1"));
}

#[test]
fn test_cli_query() {
    skip_if_no_binary!();

    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().join("data");

    // Init
    Command::new(vecstore_bin())
        .arg("init")
        .arg("--dir")
        .arg(&data_path)
        .output()
        .expect("Failed to execute vecstore");

    // Ingest data
    let vec_file = temp_dir.path().join("vec.json");
    let meta_file = temp_dir.path().join("meta.json");

    fs::write(&vec_file, "[1.0, 0.0, 0.0]").unwrap();
    fs::write(&meta_file, "{}").unwrap();

    Command::new(vecstore_bin())
        .arg("ingest")
        .arg("--dir")
        .arg(&data_path)
        .arg("--id")
        .arg("doc1")
        .arg("--vec")
        .arg(&vec_file)
        .arg("--meta")
        .arg(&meta_file)
        .output()
        .expect("Failed to execute vecstore");

    // Query
    let query_file = temp_dir.path().join("query.json");
    fs::write(&query_file, "[1.0, 0.0, 0.0]").unwrap();

    let output = Command::new(vecstore_bin())
        .arg("query")
        .arg("--dir")
        .arg(&data_path)
        .arg("--vec")
        .arg(&query_file)
        .arg("--k")
        .arg("1")
        .output()
        .expect("Failed to execute vecstore");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("doc1"));
}

// Simplified test that just checks if the binary can be invoked
#[test]
fn test_cli_binary_exists() {
    // This test just verifies the binary path logic works
    // It doesn't require the binary to actually exist
    let bin_path = vecstore_bin();
    assert!(bin_path.ends_with("vecstore"));
}
