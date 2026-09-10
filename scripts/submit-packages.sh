#!/bin/bash
#
# Helper script to submit VecStore to package repositories
# Usage: ./scripts/submit-packages.sh <version> <tag>
# Example: ./scripts/submit-packages.sh 0.0.1 v0.0.1
#

set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"

VERSION="${1:-}"
TAG="${2:-}"

if [ -z "$VERSION" ] || [ -z "$TAG" ]; then
    echo "Usage: $0 <version> <tag>"
    echo "Example: $0 0.0.1 v0.0.1"
    exit 1
fi

for package_kind in nix conda macports; do
    python3 scripts/validate-packaging.py --kind "$package_kind" --version "$VERSION" --ref "$TAG"
done

echo "📦 Preparing package submissions for VecStore $VERSION"
echo ""

# Get source tarball hash
echo "🔍 Calculating source tarball hash..."
TARBALL_URL="https://github.com/PhilipJohnBasile/vecstore/archive/$TAG.tar.gz"
TARBALL_SHA256=$(curl -fsSL "$TARBALL_URL" | shasum -a 256 | awk '{print $1}')
echo "   SHA256: $TARBALL_SHA256"
echo ""

# Check if Windows binary exists
WINDOWS_BINARY_URL="https://github.com/PhilipJohnBasile/vecstore/releases/download/$TAG/vecstore-x86_64-pc-windows-msvc.zip"
if curl --output /dev/null --silent --head --fail "$WINDOWS_BINARY_URL"; then
    WINDOWS_SHA256=$(curl -fsSL "$WINDOWS_BINARY_URL" | shasum -a 256 | awk '{print $1}')
    echo "✅ Windows binary found"
    echo "   SHA256: $WINDOWS_SHA256"
    WINDOWS_READY="yes"
else
    echo "⚠️  Windows binary not found yet"
    WINDOWS_READY="no"
fi
echo ""

# Function to update manifest files
update_manifests() {
    echo "📝 Updating manifest files..."

    # Update Winget (if Windows binary ready)
    if [ "$WINDOWS_READY" == "yes" ] && [ -f "winget/manifests/p/PhilipJohnBasile/vecstore/$VERSION/PhilipJohnBasile.vecstore.installer.yaml" ]; then
        sed -i.bak "s/PackageVersion: .*/PackageVersion: $VERSION/" "winget/manifests/p/PhilipJohnBasile/vecstore/$VERSION/PhilipJohnBasile.vecstore.yaml"
        sed -i.bak "s/PackageVersion: .*/PackageVersion: $VERSION/" "winget/manifests/p/PhilipJohnBasile/vecstore/$VERSION/PhilipJohnBasile.vecstore.installer.yaml"
        sed -i.bak "s|InstallerUrl: .*|InstallerUrl: $WINDOWS_BINARY_URL|" "winget/manifests/p/PhilipJohnBasile/vecstore/$VERSION/PhilipJohnBasile.vecstore.installer.yaml"
        sed -i.bak "s/InstallerSha256: .*/InstallerSha256: $WINDOWS_SHA256/" "winget/manifests/p/PhilipJohnBasile/vecstore/$VERSION/PhilipJohnBasile.vecstore.installer.yaml"
        sed -i.bak "s/PackageVersion: .*/PackageVersion: $VERSION/" "winget/manifests/p/PhilipJohnBasile/vecstore/$VERSION/PhilipJohnBasile.vecstore.locale.en-US.yaml"
        echo "   ✅ Updated winget manifests"
    else
        echo "   ⚠️ Winget manifest template for $VERSION not found; copy a previous release before running this script."
    fi


    echo ""
}

# Function to create submission instructions
create_instructions() {
    cat > PACKAGE_SUBMISSION.md <<EOF
# Package Submission Instructions for VecStore $VERSION ($TAG)

## Separate release workflows
Verify Homebrew and Scoop workflow results separately. This helper does not update either repository.

## Manual Submissions Needed

### 1. Nix/nixpkgs
\`\`\`bash
# Fork https://github.com/NixOS/nixpkgs
git clone https://github.com/YOUR_USERNAME/nixpkgs.git
cd nixpkgs
git checkout -b vecstore-$VERSION

# Add package
mkdir -p pkgs/by-name/ve/vecstore
cp /path/to/vecstore/nix/default.nix pkgs/by-name/ve/vecstore/package.nix

# Get hash
nix-prefetch-url --unpack $TARBALL_URL
# Update hash in package.nix

# Commit and create PR
git add .
git commit -m "vecstore: init at $VERSION"
git push origin vecstore-$VERSION
# Create PR at https://github.com/NixOS/nixpkgs/pulls
\`\`\`

### 2. Conda-forge
\`\`\`bash
# Fork https://github.com/conda-forge/staged-recipes
git clone https://github.com/YOUR_USERNAME/staged-recipes.git
cd staged-recipes
git checkout -b vecstore

# Copy recipe
mkdir -p recipes/vecstore
cp /path/to/vecstore/conda/* recipes/vecstore/

# Commit and create PR
git add .
git commit -m "Add vecstore recipe"
git push origin vecstore
# Create PR at https://github.com/conda-forge/staged-recipes/pulls
\`\`\`

### 3. MacPorts
\`\`\`bash
# Fork https://github.com/macports/macports-ports
git clone https://github.com/YOUR_USERNAME/macports-ports.git
cd macports-ports
git checkout -b vecstore

# Add Portfile
mkdir -p databases/vecstore
cp /path/to/vecstore/macports/Portfile databases/vecstore/

# Update checksums
curl -L $TARBALL_URL -o vecstore-$VERSION.tar.gz
openssl rmd160 vecstore-$VERSION.tar.gz
openssl sha256 vecstore-$VERSION.tar.gz
wc -c vecstore-$VERSION.tar.gz
# Update in Portfile

# Commit and create PR
git add .
git commit -m "vecstore: new port, version $VERSION"
git push origin vecstore
# Create PR at https://github.com/macports/macports-ports/pulls
\`\`\`

EOF

    if [ "$WINDOWS_READY" == "yes" ]; then
        cat >> PACKAGE_SUBMISSION.md <<EOF
### 4. Winget
\`\`\`bash
# Fork https://github.com/microsoft/winget-pkgs
git clone https://github.com/YOUR_USERNAME/winget-pkgs.git
cd winget-pkgs
git checkout -b vecstore-$VERSION

# Copy manifests
mkdir -p manifests/p/PhilipJohnBasile/vecstore/$VERSION
cp /path/to/vecstore/winget/manifests/p/PhilipJohnBasile/vecstore/$VERSION/* \\
   manifests/p/PhilipJohnBasile/vecstore/$VERSION/

# Commit and create PR
git add .
git commit -m "Add PhilipJohnBasile.vecstore version $VERSION"
git push origin vecstore-$VERSION
# Create PR at https://github.com/microsoft/winget-pkgs/pulls
\`\`\`

EOF
    else
        cat >> PACKAGE_SUBMISSION.md <<EOF
### 4. Winget
⚠️  Waiting for Windows binaries to be built...

EOF
    fi

    cat >> PACKAGE_SUBMISSION.md <<EOF
## Hashes for Reference
- Source tarball SHA256: \`$TARBALL_SHA256\`
EOF

    if [ "$WINDOWS_READY" == "yes" ]; then
        cat >> PACKAGE_SUBMISSION.md <<EOF
- Windows binary SHA256: \`$WINDOWS_SHA256\`
EOF
    fi

    cat >> PACKAGE_SUBMISSION.md <<EOF

## URLs
- Release: https://github.com/PhilipJohnBasile/vecstore/releases/tag/$TAG
- Source tarball: $TARBALL_URL
EOF

    if [ "$WINDOWS_READY" == "yes" ]; then
        cat >> PACKAGE_SUBMISSION.md <<EOF
- Windows binary: $WINDOWS_BINARY_URL
EOF
    fi

    echo "📄 Created PACKAGE_SUBMISSION.md with detailed instructions"
}

# Main execution
update_manifests
create_instructions

echo ""
echo "✅ Package submission preparation complete!"
echo ""
echo "Next steps:"
echo "1. Review PACKAGE_SUBMISSION.md for submission instructions"
echo "2. Commit updated manifest files"
echo "3. Follow the instructions to create PRs to each package repository"
echo ""
