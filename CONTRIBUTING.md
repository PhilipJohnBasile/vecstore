# Contributing to VecStore

Thank you for your interest in contributing to VecStore! We welcome contributions from everyone.

---

## Quick Start for Contributors

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/vecstore.git
cd vecstore
```

### 2. Set Up Development Environment

**Requirements:**
- Rust 1.70+ (`rustup update stable`)
- cargo (comes with Rust)

**Optional (for full features):**
- Python 3.8+ (for Python bindings)
- Docker (for testing deployment)

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version
cargo --version
```

### 3. Build and Test

```bash
# Build the project
cargo build

# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Format code
cargo fmt

# Check for common mistakes
cargo clippy
```

**All tests should pass:** Currently 349/349 tests passing.

### 4. Make Your Changes

```bash
# Create a feature branch
git checkout -b feat/your-amazing-feature

# Make your changes
# ... edit files ...

# Run tests
cargo test

# Format code
cargo fmt

# Commit your changes
git add .
git commit -m "Add amazing feature"
```

### 5. Submit a Pull Request

```bash
# Push to your fork
git push origin feat/your-amazing-feature

# Go to GitHub and create a Pull Request
```

---

## What to Contribute

We'd love help with:

### 🌟 High Priority
- **Language bindings** (Go, Java, C#, Ruby)
- **Document loaders** (Notion, Confluence, Google Docs)
- **Performance benchmarks** (vs Qdrant, Weaviate, Pinecone)
- **Real-world examples** (RAG apps, semantic search)
- **Bug fixes** and **performance improvements**

### 📚 Documentation
- **Tutorials** and **guides**
- **API documentation** improvements
- **Example code** and **use cases**
- **Translation** to other languages

### 🧪 Testing
- **Edge case tests**
- **Property-based tests**
- **Load tests** and **benchmarks**
- **Integration tests**

### 🎨 Features
- New **distance metrics**
- Additional **fusion strategies**
- More **tokenizer** implementations
- **Query optimizations**

---

## Code Style

VecStore follows standard Rust conventions:

```bash
# Format your code (required before PR)
cargo fmt

# Check for common issues
cargo clippy

# Run all checks
cargo fmt && cargo clippy && cargo test
```

**Key principles:**
- ✅ Write tests for new features
- ✅ Update documentation
- ✅ Keep functions small and focused
- ✅ Use meaningful variable names
- ✅ Add comments for complex logic
- ❌ Don't break existing tests
- ❌ Don't add unnecessary dependencies

---

## Pull Request Process

1. **Create a feature branch** from `main`
   ```bash
   git checkout -b feat/your-feature
   # or
   git checkout -b fix/your-bugfix
   ```

2. **Write tests** for your changes
   - Add tests in `tests/` directory
   - Or inline tests with `#[cfg(test)]`

3. **Update documentation**
   - Update README.md if adding features
   - Add doc comments (`///`) to public APIs
   - Update CHANGELOG.md

4. **Format and test**
   ```bash
   cargo fmt
   cargo clippy
   cargo test
   ```

5. **Commit with clear messages**
   ```
   feat: Add support for Go bindings
   fix: Resolve HNSW index corruption on crash
   docs: Improve quickstart guide
   test: Add property-based tests for filters
   ```

6. **Push and create PR**
   - Describe what your PR does
   - Reference any related issues
   - Include test results if applicable

7. **Respond to review feedback**
   - Address reviewer comments
   - Update your branch as needed
   - Be patient and respectful

---

## Testing Guidelines

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_query_planning

# Test with output
cargo test -- --show-output

# Integration tests only
cargo test --test test_final_optimizations
```

### Writing Tests

**Example unit test:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_feature() {
        // Arrange
        let mut store = VecStore::open("test.db").unwrap();

        // Act
        let result = store.my_feature().unwrap();

        // Assert
        assert_eq!(result.len(), 10);
    }
}
```

**Example integration test** (`tests/test_my_feature.rs`):
```rust
use vecstore::VecStore;
use tempfile::TempDir;

#[test]
fn test_my_feature_integration() {
    let temp_dir = TempDir::new().unwrap();
    let mut store = VecStore::open(temp_dir.path().join("test.db")).unwrap();

    // Test your feature
    // ...

    assert!(/* your assertion */);
}
```

---

## Documentation

### API Documentation

Add doc comments to public APIs:

```rust
/// Explains how a query will be executed and estimates its cost.
///
/// This is useful for:
/// - Understanding query performance
/// - Optimizing complex queries
/// - Debugging slow queries
///
/// # Example
///
/// ```no_run
/// # use vecstore::{VecStore, Query};
/// let store = VecStore::open("vectors.db")?;
/// let plan = store.explain_query(query)?;
/// println!("Estimated cost: {:.2}", plan.estimated_cost);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn explain_query(&self, q: Query) -> Result<QueryPlan> {
    // ...
}
```

### Generate Documentation

```bash
# Build and open documentation
cargo doc --open

# Build with private items
cargo doc --document-private-items --open
```

---

## Areas We Need Help

### Language Bindings
We have Rust and Python. Help us add:
- **Go** - `cgo` bindings
- **Java** - JNI bindings
- **C#** - P/Invoke bindings
- **Ruby** - `ffi` bindings

### Document Loaders
We support PDF, Markdown, HTML, etc. Add loaders for:
- **Notion** exports
- **Confluence** pages
- **Google Docs**
- **Microsoft Word** (`.docx`)

### Benchmarks
Help us document performance:
- Compare with Qdrant, Weaviate, Pinecone
- Measure latency at different scales
- Test different HNSW parameters
- Benchmark query planning overhead

### Examples
Create real-world examples:
- RAG chatbot with VecStore
- Semantic code search
- Document Q&A system
- Recommendation engine

---

## Development Tips

### Debugging

```bash
# Run with debug output
RUST_LOG=debug cargo test test_name -- --nocapture

# Use rust-gdb or rust-lldb
rust-gdb ./target/debug/vecstore
```

### Performance Profiling

```bash
# CPU profiling (requires `perf`)
cargo build --release
perf record target/release/vecstore
perf report

# Memory profiling (requires `valgrind`)
cargo build
valgrind --tool=massif target/debug/vecstore
```

### Benchmarking

```bash
# Run criterion benchmarks
cargo bench

# Specific benchmark
cargo bench --bench my_benchmark
```

---

## Getting Help

- **Issues:** Check [existing issues](https://github.com/yourusername/vecstore/issues)
- **Discussions:** Ask questions in [GitHub Discussions](https://github.com/yourusername/vecstore/discussions)
- **Documentation:** See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for detailed info

---

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for making VecStore better! 🎉

---

**For detailed development information, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**
