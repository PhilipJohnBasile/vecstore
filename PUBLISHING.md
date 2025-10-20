# Publishing VecStore

This document explains how to publish VecStore to various package registries.

---

## Publishing to PyPI (Python Package Index)

### Prerequisites
1. Install maturin: `pip install maturin`
2. Get PyPI API token from https://pypi.org/manage/account/token/
3. Configure: `maturin` will prompt for token on first publish

### Local Build & Test
```bash
# Build wheels for current platform
maturin build --release --features python

# Build and install locally for testing
maturin develop --release --features python

# Test the local installation
python -c "import vecstore; print(vecstore.__version__)"
```

### Manual Publish
```bash
# Build wheels for all platforms (requires Docker for Linux wheels)
maturin build --release --features python --target x86_64-unknown-linux-gnu
maturin build --release --features python --target aarch64-unknown-linux-gnu
maturin build --release --features python --target x86_64-apple-darwin
maturin build --release --features python --target aarch64-apple-darwin
maturin build --release --features python --target x86_64-pc-windows-msvc

# Publish to PyPI
maturin publish --features python
```

### Automated Publish (Recommended)
The `.github/workflows/python-publish.yml` GitHub Action will automatically:
1. Build wheels for Linux (x86_64, aarch64), Windows (x64, x86), macOS (x86_64, aarch64)
2. Build source distribution (sdist)
3. Publish to PyPI when you create a new GitHub release tag

**To trigger automated publish:**
```bash
# Create and push a tag
git tag v1.0.0
git push origin v1.0.0

# Create a GitHub release from the tag
# The workflow will automatically publish to PyPI
```

---

## Publishing to crates.io (Rust Package Registry)

### Prerequisites
1. Install cargo: comes with Rust
2. Get crates.io API token from https://crates.io/me
3. Login: `cargo login <your-token>`

### Publish
```bash
# Verify package is ready
cargo package --allow-dirty

# Inspect the package contents
cargo package --list

# Publish to crates.io
cargo publish

# Publish with specific features
cargo publish --features server,embeddings
```

**Note:** Once published to crates.io, versions cannot be unpublished (only yanked).

---

## Publishing to NPM (JavaScript/WASM)

### Prerequisites
1. Install wasm-pack: `cargo install wasm-pack`
2. Get NPM token from https://www.npmjs.com/settings/USERNAME/tokens
3. Login: `npm login`

### Build & Publish
```bash
# Build WASM package
wasm-pack build --target web --out-dir pkg --features wasm

# Test the package locally
cd pkg
npm link
cd ../examples/wasm
npm link vecstore

# Publish to NPM
cd pkg
npm publish
```

**Automated publishing** can be added via GitHub Actions similar to PyPI.

---

## Version Management

All three package ecosystems (PyPI, crates.io, NPM) use the version from their respective config files:

- **Python (PyPI):** `pyproject.toml` → `version = "1.0.0"`
- **Rust (crates.io):** `Cargo.toml` → `version = "1.0.0"`
- **JavaScript (NPM):** `pkg/package.json` → `"version": "1.0.0"`

**Ensure all three are synchronized before publishing!**

### Sync versions script:
```bash
#!/bin/bash
VERSION="1.0.0"

# Update Cargo.toml
sed -i '' "s/^version = .*/version = \"$VERSION\"/" Cargo.toml

# Update pyproject.toml
sed -i '' "s/^version = .*/version = \"$VERSION\"/" pyproject.toml

# Update package.json (after wasm-pack build)
cd pkg && npm version $VERSION
```

---

## Pre-Publish Checklist

Before publishing any package:

- [ ] All tests passing (`cargo test`)
- [ ] Documentation builds (`cargo doc --no-deps`)
- [ ] CHANGELOG.md updated with version and changes
- [ ] README.md has correct version numbers and badges
- [ ] Version numbers synchronized across Cargo.toml, pyproject.toml
- [ ] Git tag created for the release
- [ ] GitHub release created with changelog notes
- [ ] No uncommitted changes (`git status`)

---

## Post-Publish Verification

After publishing:

### PyPI
```bash
pip install vecstore==1.0.0
python -c "import vecstore; print(vecstore.__version__)"
```

### crates.io
```bash
cargo install vecstore --version 1.0.0
```

### NPM
```bash
npm install vecstore@1.0.0
```

---

## Rollback / Yanking

If you need to remove a published version:

### PyPI
- Cannot delete, but can "yank" to hide from pip: Contact PyPI support

### crates.io
```bash
cargo yank --vers 1.0.0
cargo yank --vers 1.0.0 --undo  # To undo a yank
```

### NPM
```bash
npm unpublish vecstore@1.0.0  # Only within 72 hours
npm deprecate vecstore@1.0.0 "Use version 1.0.1 instead"
```

---

## Troubleshooting

### "Package already exists"
- Increment version number in config files
- Ensure you're not re-publishing the same version

### "Build failed on Linux"
- Use manylinux Docker images for maximum compatibility
- The GitHub Action handles this automatically

### "Missing dependencies"
- Ensure all required features are enabled during build
- Check Cargo.toml dependencies match pyproject.toml requirements

---

**Ready to publish?** Follow the automated GitHub Action workflow for the smoothest experience!
