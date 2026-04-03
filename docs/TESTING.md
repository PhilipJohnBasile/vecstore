# Testing Guide

This guide covers how to run tests, add new tests, and understand the test organization in VecStore.

## Test Organization

VecStore tests are organized into several categories:

```
tests/
├── integration_tests.rs      # Core functionality
├── agent_tests.rs            # Agentic vector search
├── gpu_tests.rs              # GPU acceleration
├── distributed_tests.rs      # Distributed system
├── selectivity_tests.rs      # Filter optimization
├── persistence.rs            # Storage & durability
├── property_tests.rs         # Property-based tests
├── stress_tests.rs           # Load & stress tests
├── hybrid_search_tests.rs    # Hybrid search
├── quantization_tests.rs     # Vector quantization
├── wal_tests.rs              # Write-ahead logging
└── ...                       # More specialized tests
```

## Running Tests

### All Tests

```bash
cargo test
```

### Specific Test File

```bash
cargo test --test agent_tests
cargo test --test gpu_tests
cargo test --test distributed_tests
```

### Specific Test Function

```bash
cargo test test_agent_config_defaults
cargo test test_cpu_euclidean_distance
```

### With Feature Flags

```bash
# Run with CUDA support
cargo test --features cuda

# Run with all features
cargo test --all-features
```

### Verbose Output

```bash
cargo test -- --nocapture
```

### Release Mode Tests

```bash
cargo test --release
```

## Test Categories

### Unit Tests

Located within source files (`src/*.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example() {
        assert_eq!(1 + 1, 2);
    }
}
```

### Integration Tests

Located in `tests/` directory:

```rust
// tests/my_integration_test.rs
use vecstore::VecStore;

#[test]
fn test_integration() {
    let store = VecStore::default();
    // Test complete workflows
}
```

### Property-Based Tests

Using proptest for exhaustive testing:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_insert_retrieve(vec in prop::collection::vec(any::<f32>(), 3..=3)) {
        let store = VecStore::default();
        store.insert("test", vec.clone(), None)?;
        let result = store.get("test")?;
        prop_assert_eq!(result.unwrap().vector, vec);
    }
}
```

### Async Tests

For async functionality:

```rust
#[tokio::test]
async fn test_async_operation() {
    let store = VecStore::default();
    let result = store.async_search(query).await?;
    assert!(!result.is_empty());
}
```

## Writing Good Tests

### Test Structure (AAA Pattern)

```rust
#[test]
fn test_vector_insertion() {
    // Arrange
    let config = Config::default();
    let store = VecStore::new(config);
    let vector = vec![0.1, 0.2, 0.3];

    // Act
    let result = store.insert("id1", vector.clone(), None);

    // Assert
    assert!(result.is_ok());
    assert_eq!(store.len(), 1);
}
```

### Testing Edge Cases

```rust
#[test]
fn test_empty_database() {
    let store = VecStore::default();
    let result = store.search(&[0.1, 0.2], 10);
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_zero_k_search() {
    let store = VecStore::default();
    store.insert("id1", vec![0.1, 0.2], None).unwrap();
    let result = store.search(&[0.1, 0.2], 0);
    assert!(result.unwrap().is_empty());
}
```

### Testing Error Conditions

```rust
#[test]
fn test_dimension_mismatch() {
    let config = Config::default().with_dimensions(3);
    let store = VecStore::new(config);
    store.insert("id1", vec![0.1, 0.2, 0.3], None).unwrap();

    // This should fail - wrong dimensions
    let result = store.insert("id2", vec![0.1, 0.2], None);
    assert!(result.is_err());
}
```

### Testing Serialization

```rust
#[test]
fn test_config_serialization() {
    let config = Config::default()
        .with_ef_construction(128)
        .with_m(16);

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: Config = serde_json::from_str(&json).unwrap();

    assert_eq!(config.ef_construction, deserialized.ef_construction);
    assert_eq!(config.m, deserialized.m);
}
```

## Test Coverage

### Generating Coverage Report

```bash
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out Html
```

### Coverage Goals

- Core modules: 90%+
- Public APIs: 100%
- Error paths: 80%+

## Benchmarks

### Running Benchmarks

```bash
cargo bench
```

### Criterion Benchmarks

Located in `benches/`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_search(c: &mut Criterion) {
    let store = setup_store_with_data();
    let query = vec![0.1; 384];

    c.bench_function("knn_search_k10", |b| {
        b.iter(|| store.search(&query, 10))
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
```

## Continuous Integration

Tests run automatically on:

- Push to main branch
- Pull requests
- Nightly scheduled runs

### CI Configuration

The CI pipeline runs:

1. `cargo check --all-features`
2. `cargo clippy --all-features -- -D warnings`
3. `cargo test --all-features`
4. `cargo fmt --check`
5. `cargo doc --all-features`

## Debugging Tests

### Running a Single Test with Output

```bash
cargo test test_name -- --nocapture --test-threads=1
```

### Using RUST_BACKTRACE

```bash
RUST_BACKTRACE=1 cargo test test_name
```

### Using RUST_LOG

```bash
RUST_LOG=vecstore=debug cargo test test_name -- --nocapture
```

## Test Utilities

### Temporary Directories

```rust
use tempfile::TempDir;

#[test]
fn test_with_temp_dir() {
    let temp = TempDir::new().unwrap();
    let config = Config::default()
        .with_storage_path(temp.path());
    let store = VecStore::new(config);

    // Test operations...
    // temp directory cleaned up automatically
}
```

### Random Test Data

```rust
use rand::{Rng, SeedableRng};

fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect()
}
```

## Common Test Patterns

### Setup and Teardown

```rust
struct TestContext {
    store: VecStore,
    temp_dir: TempDir,
}

impl TestContext {
    fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        let config = Config::default()
            .with_storage_path(temp_dir.path());
        let store = VecStore::new(config);
        Self { store, temp_dir }
    }
}

#[test]
fn test_with_context() {
    let ctx = TestContext::new();
    // Use ctx.store...
}
```

### Parameterized Tests

```rust
#[test]
fn test_various_dimensions() {
    for dim in [64, 128, 256, 512, 1024] {
        let config = Config::default().with_dimensions(dim);
        let store = VecStore::new(config);
        let vector: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
        assert!(store.insert("test", vector, None).is_ok());
    }
}
```
