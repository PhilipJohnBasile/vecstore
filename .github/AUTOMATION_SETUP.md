# Automated Publishing Setup

This document explains how to set up automated publishing for VecStore across all platforms.

## Overview

VecStore uses GitHub Actions to automatically publish to:
- **crates.io** (Rust packages)
- **PyPI** (Python packages)
- **npm** (@vecstore/core - WASM)
- **Docker Hub** & **GitHub Container Registry** (Docker images)
- **GitHub Releases** (Binary downloads)

## Required Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

### 1. CARGO_REGISTRY_TOKEN
- **Purpose:** Publish to crates.io
- **How to get:**
  1. Go to https://crates.io/settings/tokens
  2. Click "New Token"
  3. Name: `GitHub Actions - VecStore`
  4. Copy the token

### 2. PYPI_API_TOKEN
- **Purpose:** Publish to PyPI
- **How to get:**
  1. Go to https://pypi.org/manage/account/token/
  2. Click "Add API token"
  3. Name: `GitHub Actions - VecStore`
  4. Scope: "Entire account" or limit to "vecstore-rs"
  5. Copy the token (starts with `pypi-`)

### 3. NPM_TOKEN
- **Purpose:** Publish to npm
- **How to get:**
  1. Go to https://www.npmjs.com/settings/~/tokens
  2. Click "Generate New Token" → "Automation"
  3. Copy the token

### 4. DOCKERHUB_USERNAME
- **Purpose:** Docker Hub username
- **Value:** Your Docker Hub username

### 5. DOCKERHUB_TOKEN
- **Purpose:** Publish to Docker Hub
- **How to get:**
  1. Go to https://hub.docker.com/settings/security
  2. Click "New Access Token"
  3. Description: `GitHub Actions - VecStore`
  4. Copy the token

## Automated Release Process

### Triggering a Release

**Option 1: Create a Git Tag**
```bash
# Tag the release
git tag v1.0.0
git push origin v1.0.0
```

**Option 2: Manual Workflow Dispatch**
1. Go to Actions → Release
2. Click "Run workflow"
3. Enter version (e.g., `1.0.0`)
4. Click "Run workflow"

### What Happens Automatically

When you push a tag (e.g., `v1.0.0`), GitHub Actions will:

1. **Create GitHub Release**
   - Generate release notes
   - Create downloadable binaries for:
     - Linux (x86_64, aarch64)
     - macOS (x86_64, aarch64)
     - Windows (x86_64)

2. **Publish to crates.io**
   - Build and verify package
   - Publish `vecstore` crate

3. **Publish to PyPI**
   - Build wheels for all platforms:
     - Linux (x86_64, aarch64) - manylinux
     - macOS (x86_64, aarch64)
     - Windows (x64, x86)
   - Publish `vecstore-rs` package

4. **Publish to npm**
   - Build WASM package with wasm-pack
   - Publish `@vecstore/core` package

5. **Publish to Docker**
   - Build multi-platform images (amd64, arm64)
   - Push to:
     - `ghcr.io/philipjohnbasile/vecstore:1.0.0`
     - `ghcr.io/philipjohnbasile/vecstore:latest`
     - `dockerhub/vecstore:1.0.0`
     - `dockerhub/vecstore:latest`

## Manual Publishing

If you need to publish manually:

### crates.io
```bash
cargo publish --token $CARGO_REGISTRY_TOKEN
```

### PyPI
```bash
maturin publish --username __token__ --password $PYPI_API_TOKEN
```

### npm
```bash
wasm-pack build --target web --features wasm
cd pkg && npm publish
```

### Docker
```bash
docker build -t vecstore:1.0.0 .
docker tag vecstore:1.0.0 ghcr.io/philipjohnbasile/vecstore:1.0.0
docker push ghcr.io/philipjohnbasile/vecstore:1.0.0
```

## Troubleshooting

### Release Failed
- Check the Actions tab for error messages
- Verify all secrets are set correctly
- Ensure version numbers match across files

### Package Already Exists
- crates.io: Cannot republish same version, bump version
- PyPI: Use `--skip-existing` flag (already in workflow)
- npm: Cannot republish, bump version

### Binary Build Fails
- Check Rust toolchain compatibility
- Verify target platform support
- Review build logs in Actions

## Version Bumping

Before creating a release, update version in:
1. `Cargo.toml` → `version = "1.0.1"`
2. `pyproject.toml` → `version = "1.0.1"`
3. `CHANGELOG.md` → Add new version section

Then create and push the tag:
```bash
git add Cargo.toml pyproject.toml CHANGELOG.md
git commit -m "chore: Bump version to 1.0.1"
git tag v1.0.1
git push origin main
git push origin v1.0.1
```

## Monitoring

- **GitHub Actions:** https://github.com/PhilipJohnBasile/vecstore/actions
- **crates.io:** https://crates.io/crates/vecstore
- **PyPI:** https://pypi.org/project/vecstore-rs/
- **npm:** https://www.npmjs.com/package/@vecstore/core
- **Docker Hub:** https://hub.docker.com/r/youruser/vecstore
- **GHCR:** https://github.com/PhilipJohnBasile/vecstore/pkgs/container/vecstore
