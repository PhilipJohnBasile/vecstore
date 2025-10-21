# VecStore Publishing Guide

Complete instructions for publishing VecStore to all package managers.

## Already Published ✅

- **crates.io**: https://crates.io/crates/vecstore
- **PyPI**: https://pypi.org/project/vecstore-rs/
- **npm**: https://www.npmjs.com/package/vecstore-wasm
- **Docker Hub**: https://hub.docker.com/r/philipjohnbasile/vecstore
- **GHCR**: https://github.com/PhilipJohnBasile/vecstore/pkgs/container/vecstore
- **docs.rs**: https://docs.rs/vecstore
- **lib.rs**: https://lib.rs/crates/vecstore
- **Homebrew**: https://github.com/PhilipJohnBasile/homebrew-vecstore
- **Scoop**: https://github.com/PhilipJohnBasile/scoop-vecstore

## Manual Publishing Needed

### 1. AUR (Arch User Repository)

**Prerequisites:**
- Arch Linux system (or Docker)
- AUR account
- SSH key uploaded to AUR

**Steps:**

#### A. Create AUR Account
1. Go to https://aur.archlinux.org/register
2. Create account with your email
3. Generate SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
4. Upload public key at https://aur.archlinux.org/account

#### B. Prepare PKGBUILD
```bash
cd /Users/pjb/Git/vecstore/aur

# Generate .SRCINFO
makepkg --printsrcinfo > .SRCINFO

# Verify PKGBUILD works
makepkg -si
```

#### C. Publish to AUR
```bash
# Clone AUR repo (will be empty first time)
git clone ssh://aur@aur.archlinux.org/vecstore.git aur-repo
cd aur-repo

# Copy files
cp ../PKGBUILD .
cp ../.SRCINFO .

# Commit and push
git add PKGBUILD .SRCINFO
git commit -m "Initial import: vecstore 1.0.0"
git push origin master
```

**Users can then install with:**
```bash
yay -S vecstore
# or
paru -S vecstore
```

**Update Process:**
```bash
cd aur-repo
# Update version in PKGBUILD
# Update sha256sum
makepkg --printsrcinfo > .SRCINFO
git add PKGBUILD .SRCINFO
git commit -m "Update to version X.Y.Z"
git push
```

---

### 2. Chocolatey (Windows)

**Prerequisites:**
- Windows system
- Chocolatey account
- API key from https://community.chocolatey.org/account

**Steps:**

#### A. Create Chocolatey Account
1. Go to https://community.chocolatey.org/account/Register
2. Verify email
3. Get API key from https://community.chocolatey.org/account

#### B. Build Windows Binaries (Required First!)

**Option 1: Cross-compile from macOS/Linux**
```bash
# Install cross-compilation tools
cargo install cross

# Build for Windows
cross build --release --target x86_64-pc-windows-msvc --features server --bin vecstore-server

# Package
cd target/x86_64-pc-windows-msvc/release
7z a vecstore-windows-x86_64.zip vecstore-server.exe
```

**Option 2: Use GitHub Actions**
Create `.github/workflows/windows-release.yml`:
```yaml
name: Build Windows Binaries

on:
  release:
    types: [published]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: x86_64-pc-windows-msvc
      - name: Build
        run: cargo build --release --features server --bin vecstore-server
      - name: Package
        run: |
          cd target/release
          7z a vecstore-windows-x86_64.zip vecstore-server.exe
      - name: Upload to Release
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ github.event.release.upload_url }}
          asset_path: target/release/vecstore-windows-x86_64.zip
          asset_name: vecstore-windows-x86_64.zip
          asset_content_type: application/zip
```

#### C. Update Chocolatey Manifest

Once binaries are available, update `chocolatey/vecstore.nuspec`:
```xml
<!-- Add URL to actual binary -->
<file src="tools\chocolateyInstall.ps1" target="tools" />
```

Create `chocolatey/tools/chocolateyInstall.ps1`:
```powershell
$ErrorActionPreference = 'Stop'

$packageName = 'vecstore'
$url64 = 'https://github.com/PhilipJohnBasile/vecstore/releases/download/v1.0.0/vecstore-windows-x86_64.zip'
$checksum64 = 'PASTE_SHA256_HERE'

$packageArgs = @{
  packageName   = $packageName
  unzipLocation = "$(Split-Path -parent $MyInvocation.MyCommand.Definition)"
  url64bit      = $url64
  checksum64    = $checksum64
  checksumType64= 'sha256'
}

Install-ChocolateyZipPackage @packageArgs
```

#### D. Package and Submit
```powershell
# Install Chocolatey packager
choco install chocolatey-core.extension

# Package
cd chocolatey
choco pack

# Test locally
choco install vecstore -source .

# Push to Chocolatey
choco apikey --key YOUR_API_KEY --source https://push.chocolatey.org/
choco push vecstore.1.0.0.nupkg --source https://push.chocolatey.org/
```

**Users can then install with:**
```powershell
choco install vecstore
```

---

### 3. Snap (Ubuntu/Linux)

**Prerequisites:**
- Ubuntu One account
- snapcraft installed

**Steps:**

#### A. Create Ubuntu One Account
1. Go to https://login.ubuntu.com/
2. Create account
3. Go to https://snapcraft.io/account
4. Register package name "vecstore"

#### B. Test Build Locally
```bash
# Install snapcraft
sudo snap install snapcraft --classic

# Build snap
cd /Users/pjb/Git/vecstore
snapcraft

# Test install
sudo snap install --dangerous vecstore_*.snap

# Test run
vecstore-server --help
```

#### C. Publish to Snap Store
```bash
# Login
snapcraft login

# Upload
snapcraft upload vecstore_1.0.0_amd64.snap --release=stable

# Check status
snapcraft status vecstore
```

**Users can then install with:**
```bash
sudo snap install vecstore
```

**Update Process:**
```bash
# Update version in snap/snapcraft.yaml
snapcraft
snapcraft upload vecstore_*.snap --release=stable
```

---

### 4. Flatpak (Flathub)

**Prerequisites:**
- GitHub account
- Flathub submission

**Steps:**

#### A. Fork Flathub Repository
```bash
# Fork https://github.com/flathub/flathub
git clone https://github.com/YOUR_USERNAME/flathub.git
```

#### B. Create Application Repository
```bash
# Create new repo for your app
gh repo create flathub/com.github.philipjohnbasile.vecstore --public

# Clone it
git clone https://github.com/flathub/com.github.philipjohnbasile.vecstore.git
cd com.github.philipjohnbasile.vecstore
```

#### C. Add Flatpak Manifest
```bash
# Copy your manifest
cp /Users/pjb/Git/vecstore/flatpak/com.github.philipjohnbasile.vecstore.json .

# Generate Cargo sources (required for Rust apps)
# Install flatpak-cargo-generator
git clone https://github.com/flatpak/flatpak-builder-tools.git
cd flatpak-builder-tools/cargo
./flatpak-cargo-generator.py /Users/pjb/Git/vecstore/Cargo.lock -o generated-sources.json

# Copy back to manifest repo
cp generated-sources.json ../../
cd ../../
```

#### D. Test Build
```bash
# Install flatpak-builder
sudo apt install flatpak-builder

# Build
flatpak-builder build-dir com.github.philipjohnbasile.vecstore.json

# Test
flatpak-builder --run build-dir com.github.philipjohnbasile.vecstore.json vecstore-server --help
```

#### E. Submit to Flathub
```bash
# Commit your manifest
git add .
git commit -m "Initial commit of VecStore"
git push

# Create PR to flathub/flathub
# Go to https://github.com/flathub/flathub/pulls
# Create PR with title: "Add com.github.philipjohnbasile.vecstore"
```

**Flathub Review Process:**
- Maintainers will review (usually 1-2 weeks)
- They'll ask for changes if needed
- Once approved, it's automatically built and published

**Users can then install with:**
```bash
flatpak install flathub com.github.philipjohnbasile.vecstore
```

---

## Summary Checklist

### Immediate (Can do now)
- [x] crates.io
- [x] PyPI
- [x] npm
- [x] Docker Hub
- [x] GHCR
- [x] Homebrew
- [x] Scoop (placeholder)
- [x] cargo-binstall (auto-supported via GitHub releases)

### Ready to Submit (Manifests Created)
- [ ] Winget - PR to microsoft/winget-pkgs (needs Windows binary)
- [ ] Nix - PR to NixOS/nixpkgs
- [ ] Conda-forge - PR to conda-forge/staged-recipes
- [ ] MacPorts - PR to macports/macports-ports

### Need Accounts
- [ ] AUR - Register at https://aur.archlinux.org/register
- [ ] Chocolatey - Register at https://community.chocolatey.org/account/Register
- [ ] Snap - Register at https://snapcraft.io/account
- [ ] Flatpak - Use GitHub account

### Need Binaries First
- [ ] Chocolatey (Windows x64 binary)
- [ ] Scoop (Windows x64 binary)
- [ ] Winget (Windows x64 binary)

### Estimated Time
- **cargo-binstall**: ✅ Already works!
- **Winget**: 1 hour (needs Windows binary + PR + review ~1 week)
- **Nix**: 1 hour (PR + review ~1 week)
- **Conda-forge**: 2 hours (PR + review ~1-2 weeks)
- **MacPorts**: 1 hour (PR + review ~1 week)
- **AUR**: 30 minutes (after account setup)
- **Chocolatey**: 2 hours (needs Windows binary + review)
- **Snap**: 1 hour (+ 1-2 weeks review)
- **Flatpak**: 2 hours (+ 1-2 weeks review)

---

## Quick Reference

**Package URLs After Publishing:**
- AUR: `https://aur.archlinux.org/packages/vecstore`
- Chocolatey: `https://community.chocolatey.org/packages/vecstore`
- Snap: `https://snapcraft.io/vecstore`
- Flatpak: `https://flathub.org/apps/com.github.philipjohnbasile.vecstore`
- Winget: `https://winget.run/pkg/PhilipJohnBasile/vecstore`
- Nix: `https://search.nixos.org/packages?query=vecstore`
- Conda-forge: `https://anaconda.org/conda-forge/vecstore`
- MacPorts: `https://ports.macports.org/port/vecstore/`

**Install Commands:**
```bash
# Rust / Cargo
cargo install vecstore --features server
cargo binstall vecstore  # Faster binary install

# AUR (Arch Linux)
yay -S vecstore

# Nix
nix-env -iA nixpkgs.vecstore

# Conda-forge
conda install -c conda-forge vecstore

# Homebrew (macOS)
brew tap PhilipJohnBasile/vecstore && brew install vecstore

# MacPorts (macOS)
sudo port install vecstore

# Winget (Windows)
winget install PhilipJohnBasile.vecstore

# Chocolatey (Windows)
choco install vecstore

# Scoop (Windows)
scoop bucket add vecstore https://github.com/PhilipJohnBasile/scoop-vecstore
scoop install vecstore

# Snap (Linux)
sudo snap install vecstore

# Flatpak (Linux)
flatpak install flathub com.github.philipjohnbasile.vecstore
```

---

### 5. Winget (Windows Package Manager)

**Prerequisites:**
- Windows 10/11
- GitHub account (for PR submission)

**Steps:**

#### A. Prepare Manifest Files

The Winget manifests are already in `winget/manifests/p/PhilipJohnBasile/vecstore/1.0.0/`:
- `PhilipJohnBasile.vecstore.yaml` (version manifest)
- `PhilipJohnBasile.vecstore.installer.yaml` (installer manifest)
- `PhilipJohnBasile.vecstore.locale.en-US.yaml` (locale manifest)

**Update the installer manifest with Windows binary SHA256:**
```bash
# After Windows binaries are available
# Get SHA256 of the Windows zip file
sha256sum vecstore-windows-x86_64.zip

# Update in PhilipJohnBasile.vecstore.installer.yaml:
# InstallerSha256: <actual_sha256>
```

#### B. Submit to Winget Repository

```bash
# Fork https://github.com/microsoft/winget-pkgs
git clone https://github.com/YOUR_USERNAME/winget-pkgs.git
cd winget-pkgs

# Create branch
git checkout -b vecstore-1.0.0

# Copy manifests to correct location
mkdir -p manifests/p/PhilipJohnBasile/vecstore/1.0.0
cp /Users/pjb/Git/vecstore/winget/manifests/p/PhilipJohnBasile/vecstore/1.0.0/* \
   manifests/p/PhilipJohnBasile/vecstore/1.0.0/

# Commit and push
git add .
git commit -m "Add PhilipJohnBasile.vecstore version 1.0.0"
git push origin vecstore-1.0.0

# Create PR at https://github.com/microsoft/winget-pkgs/pulls
```

**Users can then install with:**
```powershell
winget install PhilipJohnBasile.vecstore
```

---

### 6. Nix / nixpkgs

**Prerequisites:**
- Nix installed
- GitHub account

**Steps:**

#### A. Generate Hash

```bash
# Get source tarball hash
nix-prefetch-url --unpack https://github.com/PhilipJohnBasile/vecstore/archive/refs/tags/v1.0.0.tar.gz

# Update hash in nix/default.nix
```

#### B. Test Build Locally

```bash
# Build with Nix
nix-build nix/default.nix

# Test the binary
./result/bin/vecstore-server --help
```

#### C. Submit to nixpkgs

```bash
# Fork https://github.com/NixOS/nixpkgs
git clone https://github.com/YOUR_USERNAME/nixpkgs.git
cd nixpkgs

# Create branch
git checkout -b vecstore-1.0.0

# Add package to pkgs/by-name/ve/vecstore/package.nix
mkdir -p pkgs/by-name/ve/vecstore
cp /Users/pjb/Git/vecstore/nix/default.nix pkgs/by-name/ve/vecstore/package.nix

# Commit and push
git add .
git commit -m "vecstore: init at 1.0.0"
git push origin vecstore-1.0.0

# Create PR at https://github.com/NixOS/nixpkgs/pulls
```

**Users can then install with:**
```bash
nix-env -iA nixpkgs.vecstore
# or with flakes
nix profile install nixpkgs#vecstore
```

---

### 7. Conda-forge

**Prerequisites:**
- conda or mamba installed
- GitHub account

**Steps:**

#### A. Update SHA256 Hash

```bash
# Get source tarball hash
curl -sL https://github.com/PhilipJohnBasile/vecstore/archive/refs/tags/v1.0.0.tar.gz | sha256sum

# Update in conda/meta.yaml
```

#### B. Test Build Locally

```bash
# Install conda-build
conda install conda-build

# Build the package
cd /Users/pjb/Git/vecstore
conda build conda/

# Test install
conda install --use-local vecstore
vecstore-server --help
```

#### C. Submit to Conda-forge

```bash
# Fork https://github.com/conda-forge/staged-recipes
git clone https://github.com/YOUR_USERNAME/staged-recipes.git
cd staged-recipes

# Create branch
git checkout -b vecstore

# Copy recipe
mkdir -p recipes/vecstore
cp /Users/pjb/Git/vecstore/conda/meta.yaml recipes/vecstore/
cp /Users/pjb/Git/vecstore/conda/build.sh recipes/vecstore/

# Commit and push
git add .
git commit -m "Add vecstore recipe"
git push origin vecstore

# Create PR at https://github.com/conda-forge/staged-recipes/pulls
```

**After approval, users can install with:**
```bash
conda install -c conda-forge vecstore
# or
mamba install vecstore
```

---

### 8. MacPorts

**Prerequisites:**
- macOS
- MacPorts installed
- GitHub account

**Steps:**

#### A. Update Checksums

```bash
# Get checksums
curl -L https://github.com/PhilipJohnBasile/vecstore/archive/refs/tags/v1.0.0.tar.gz -o vecstore-1.0.0.tar.gz
openssl rmd160 vecstore-1.0.0.tar.gz
openssl sha256 vecstore-1.0.0.tar.gz
wc -c vecstore-1.0.0.tar.gz

# Update in macports/Portfile
```

#### B. Test Locally

```bash
# Copy Portfile to local repository
sudo cp /Users/pjb/Git/vecstore/macports/Portfile /opt/local/var/macports/sources/rsync.macports.org/macports/release/tarballs/ports/databases/vecstore/

# Test install
sudo port install vecstore

# Test binary
vecstore-server --help
```

#### C. Submit to MacPorts

```bash
# Fork https://github.com/macports/macports-ports
git clone https://github.com/YOUR_USERNAME/macports-ports.git
cd macports-ports

# Create branch
git checkout -b vecstore

# Add Portfile
mkdir -p databases/vecstore
cp /Users/pjb/Git/vecstore/macports/Portfile databases/vecstore/

# Commit and push
git add .
git commit -m "vecstore: new port, version 1.0.0"
git push origin vecstore

# Create PR at https://github.com/macports/macports-ports/pulls
```

**Users can then install with:**
```bash
sudo port install vecstore
```

---

### 9. cargo-binstall (Already Supported!)

**Prerequisites:**
- None! Just needs GitHub releases with binaries

**Status:** ✅ Already works because we have GitHub release with binaries at:
- https://github.com/PhilipJohnBasile/vecstore/releases/tag/v1.0.0

**Users can install with:**
```bash
cargo install cargo-binstall
cargo binstall vecstore
```

**How it works:**
1. Checks GitHub releases for binary assets
2. Downloads appropriate binary for user's platform
3. Installs directly (much faster than `cargo install`)

**No action needed** - this works automatically once binaries are in GitHub releases!
