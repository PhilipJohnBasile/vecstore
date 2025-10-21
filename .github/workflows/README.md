# Package Manager Automation Workflows

This directory contains GitHub Actions workflows for automating package distribution across multiple platforms.

## Workflows

### 1. build-binaries.yml
Builds cross-platform binaries on release:
- **Platforms**: macOS (x86_64, ARM64), Linux (x86_64, ARM64), Windows (x86_64, ARM64)
- **Trigger**: On release publish or manual workflow dispatch
- **Output**: Binary artifacts attached to GitHub releases

### 2. package-updates.yml
Auto-updates package managers we control:
- **Homebrew tap**: PhilipJohnBasile/homebrew-vecstore
- **Scoop bucket**: PhilipJohnBasile/scoop-vecstore
- **Trigger**: On release publish or manual workflow dispatch
- **Status**: ✅ Fully automated (uses GITHUB_TOKEN)

### 3. submit-to-package-managers.yml
Creates PRs to external package repositories:
- **nixpkgs**: NixOS/nixpkgs
- **conda-forge**: conda-forge/staged-recipes
- **MacPorts**: macports/macports-ports
- **Winget**: microsoft/winget-pkgs
- **Status**: ⚠️ Requires Personal Access Token (see below)

## Setup Requirements

### For Automated Submissions (submit-to-package-managers.yml)

The default `GITHUB_TOKEN` cannot fork external repositories or create PRs to them. To enable automation:

#### Step 1: Create a Personal Access Token
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a descriptive name: "VecStore Package Automation"
4. Select scopes:
   - `public_repo` - Access public repositories
   - `workflow` - Update GitHub Actions workflows
5. Set expiration (recommended: 90 days, then rotate)
6. Click "Generate token" and **copy it immediately**

#### Step 2: Add Token as Repository Secret
1. Go to your repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `PAT_TOKEN`
4. Value: paste your token
5. Click "Add secret"

#### Step 3: Update Workflow
In `.github/workflows/submit-to-package-managers.yml`, replace all instances of:
```yaml
GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

With:
```yaml
GH_TOKEN: ${{ secrets.PAT_TOKEN }}
```

Search for this pattern in lines 59, 150, 235, and 320.

### Manual Submission (Alternative)

If you prefer not to use a PAT, you can submit packages manually:

```bash
# Run the helper script
./scripts/submit-packages.sh 1.0.0 v1.0.0

# Then follow the instructions in PACKAGE_SUBMISSION.md
```

This will:
1. Calculate source hashes
2. Update manifest files
3. Create a detailed instruction file
4. Guide you through manual PR creation for each platform

## Triggering Workflows

### Automatic (on release)
Workflows trigger automatically when you publish a GitHub release.

### Manual
Trigger workflows manually via GitHub UI:
```bash
# Or via gh CLI:
gh workflow run "Build Release Binaries" -f tag=v1.0.0
gh workflow run "Update Package Managers" -f version=1.0.0 -f tag=v1.0.0
gh workflow run "Submit to Package Managers" -f version=1.0.0 -f tag=v1.0.0
```

## Security Considerations

- **PAT Security**: Personal Access Tokens have broad permissions. Only create tokens with minimum required scopes.
- **Token Rotation**: Regularly rotate your PAT (every 90 days recommended).
- **Secret Protection**: Never commit tokens to git. Use repository secrets only.
- **Alternative**: For maximum security, use manual submission process instead of automation.

## Troubleshooting

### "Resource not accessible by integration (HTTP 403)"
- This means you're using `GITHUB_TOKEN` instead of `PAT_TOKEN`
- Follow the setup steps above to add a PAT

### "Windows binary not found"
- The Windows binary build may not be complete yet
- Check the "Build Release Binaries" workflow status
- Winget submission will be skipped if binary isn't available

### Hash mismatches
- Ensure the release tag exists on GitHub
- Wait a few seconds after creating a release before triggering workflows
- Check that the tarball URL is accessible: `https://github.com/PhilipJohnBasile/vecstore/archive/refs/tags/TAG.tar.gz`

## Workflow Details

Each external package manager has specific requirements:

| Platform | Repo | PR Target | First Submission | Updates |
|----------|------|-----------|-----------------|---------|
| **Nix** | NixOS/nixpkgs | `pkgs/by-name/ve/vecstore/` | New package | Version updates |
| **Conda** | conda-forge/staged-recipes | `recipes/vecstore/` | Once (creates feedstock) | Update feedstock later |
| **MacPorts** | macports/macports-ports | `databases/vecstore/` | New port | Port updates |
| **Winget** | microsoft/winget-pkgs | `manifests/p/PhilipJohnBasile/vecstore/VERSION/` | New version | Each version |

After the first conda-forge submission is accepted, future updates go to the feedstock repo, not staged-recipes.
