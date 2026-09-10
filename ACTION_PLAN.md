# VecStore - Immediate Action Plan
**Based on:** Product Improvement Report (Dec 27, 2025)
**Timeline:** 30-Day Sprint to Stabilization
**Goal:** Fix critical issues and establish foundation for 1.0

> Historical proposal: this plan reflects the December 27, 2025 assessment. Re-baseline every task against the current repository before execution.

---

## Overview

This document provides a **concrete, actionable 30-day plan** to address the most critical issues identified in the Product Improvement Report.

**Current State:** ❌ Build failing, 101 warnings, unclear production status
**Target State:** ✅ Clean build, passing tests, clear roadmap, v0.2.0 released

---

## Week 1: Critical Fixes (Days 1-7)

### Day 1-2: Fix Compilation & Tests

**Owner:** Development Team
**Priority:** P0 - BLOCKER

**Tasks:**
1. [ ] Identify and fix compilation error in `src/store/mod.rs` or related modules
2. [ ] Run `cargo build --all-features` and fix all errors
3. [ ] Run `cargo test --lib` and fix test compilation
4. [ ] Document test pass rate (actual number, not claims)
5. [ ] Create issue for any remaining test failures

**Success Criteria:**
- ✅ `cargo build` completes with 0 errors
- ✅ `cargo test --lib` compiles successfully
- ✅ At least 80% of tests pass (document failures)

**Commands:**
```bash
# Fix compilation
cargo build --all-features 2>&1 | tee build.log

# Fix test compilation
cargo test --lib --no-run 2>&1 | tee test-build.log

# Run tests
cargo test --lib 2>&1 | tee test-run.log

# Count passing tests
grep -E "test result: (ok|FAILED)" test-run.log
```

---

### Day 3: Resolve Compiler Warnings

**Owner:** Development Team
**Priority:** P0

**Tasks:**
1. [ ] Fix all unused variable warnings (prefix with `_` if intentional)
2. [ ] Fix all unused import warnings
3. [ ] Fix all other clippy warnings
4. [ ] Run `cargo clippy` with no warnings
5. [ ] Update CI to enforce zero warnings

**Success Criteria:**
- ✅ `cargo clippy --all-features` produces 0 warnings
- ✅ All intentional unused variables prefixed with `_`

**Commands:**
```bash
# Check warnings
cargo clippy --all-features 2>&1 | tee clippy.log

# Count warnings
grep "warning:" clippy.log | wc -l

# Fix and verify
cargo clippy --all-features -- -D warnings
```

---

### Day 4: Standardize Versions

**Owner:** Development Team
**Priority:** P0

**Tasks:**
1. [ ] Decide on single version number (recommend: 0.2.0-alpha)
2. [ ] Update Cargo.toml: `version = "0.2.0-alpha"`
3. [ ] Update pyproject.toml: `version = "0.2.0a1"`
4. [ ] Update wasm-pkg/package.json: `version = "0.2.0-alpha.1"`
5. [ ] Update all README files to reference 0.2.0
6. [ ] Update CHANGELOG.md with accurate release info
7. [ ] Search codebase for hardcoded version strings and update

**Success Criteria:**
- ✅ Single version number across all files
- ✅ Version follows semantic versioning
- ✅ No stale version references in docs

**Commands:**
```bash
# Find all version references
grep -r "0\.0\.1" . --include="*.md" --include="*.toml" --include="*.json"
grep -r "0\.0\.2" . --include="*.md" --include="*.toml" --include="*.json"
grep -r "0\.1\.0" . --include="*.md" --include="*.toml" --include="*.json"

# Update systematically
# (Manual editing required)
```

---

### Day 5: Add Prominent Warnings & Create MATURITY.md

**Owner:** Documentation Team
**Priority:** P0

**Tasks:**
1. [ ] Update README.md with prominent alpha warning (top of file)
2. [ ] Create MATURITY.md feature stability matrix
3. [ ] Link to MATURITY.md from README
4. [ ] Add stability badges to each feature in docs
5. [ ] Update CONTRIBUTING.md with maturity model

**Success Criteria:**
- ✅ Clear warning visible within first 3 lines of README
- ✅ MATURITY.md published with all modules categorized
- ✅ Users can quickly determine what's safe to use

**Example MATURITY.md structure:**
```markdown
# Feature Maturity Matrix

## Legend
- ✅ **Stable** - Production-ready, API stable, well-tested
- 🟡 **Beta** - Feature-complete, API may change, use with caution
- 🔴 **Experimental** - Prototype only, may be removed, no guarantees
- ⚫ **Deprecated** - Will be removed in future version

## Core Features

| Feature | Maturity | Version | Notes |
|---------|----------|---------|-------|
| HNSW Indexing | ✅ Stable | 0.1.0 | Battle-tested, performant |
| Vector Operations | ✅ Stable | 0.1.0 | upsert, query, delete |
| Metadata Filtering | 🟡 Beta | 0.1.0 | SQL-like syntax, most operators work |
| Snapshots | 🟡 Beta | 0.1.0 | Backup works, restore needs more testing |
| Hybrid Search | 🟡 Beta | 0.2.0 | BM25 implementation, limited testing |

## Advanced Features

| Feature | Maturity | Version | Notes |
|---------|----------|---------|-------|
| Server Mode (gRPC/HTTP) | 🟡 Beta | 0.1.0 | Works but lacks auth, rate limiting |
| Multi-Tenant Namespaces | 🟡 Beta | 0.1.0 | Quota enforcement working, needs hardening |
| Product Quantization | 🔴 Experimental | 0.2.0 | Prototype, limited testing |
| DiskANN Index | 🔴 Experimental | 0.2.0 | Not fully integrated |

## Innovation Features (Experimental)

| Feature | Maturity | Version | Notes |
|---------|----------|---------|-------|
| Explainable Search | 🔴 Experimental | 0.2.0 | Prototype, API unstable |
| Time-Aware Search | 🔴 Experimental | 0.2.0 | Proof of concept |
| Privacy-Preserving | 🔴 Experimental | 0.2.0 | Research prototype |
| GPU Acceleration | 🔴 Experimental | 0.2.0 | Stubs only, not functional |
| ... (50+ other modules) | 🔴 Experimental | 0.2.0 | See individual module docs |
```

---

### Day 6-7: Set Up CI/CD

**Owner:** DevOps Team
**Priority:** P0

**Tasks:**
1. [ ] Create `.github/workflows/ci.yml`
2. [ ] Add build job (cargo build --all-features)
3. [ ] Add test job (cargo test --all-features)
4. [ ] Add clippy job (cargo clippy -- -D warnings)
5. [ ] Add format check job (cargo fmt -- --check)
6. [ ] Set up PR branch protection (require CI passing)
7. [ ] Add status badge to README

**Success Criteria:**
- ✅ CI runs on every PR and commit to main
- ✅ CI fails if tests fail or warnings exist
- ✅ Status badge shows "passing" in README

**Example `.github/workflows/ci.yml`:**
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
    - name: Build
      run: cargo build --all-features --verbose

  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
    - name: Run tests
      run: cargo test --all-features --verbose

  clippy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        components: clippy
        override: true
    - name: Clippy
      run: cargo clippy --all-features -- -D warnings

  format:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        components: rustfmt
        override: true
    - name: Check formatting
      run: cargo fmt -- --check
```

---

## Week 2: Documentation Audit (Days 8-14)

### Day 8-9: Audit All Documentation

**Owner:** Documentation Team
**Priority:** P1

**Tasks:**
1. [ ] Create spreadsheet of all claims in docs
2. [ ] Verify each claim against codebase/tests
3. [ ] Flag outdated/incorrect claims
4. [ ] Update or remove incorrect information
5. [ ] Add "last verified" date to technical docs

**Success Criteria:**
- ✅ All documentation claims verified or updated
- ✅ No references to "349 tests passing" (unless actually true)
- ✅ Performance claims backed by benchmarks or removed

---

### Day 10-11: Create Missing Documentation

**Owner:** Documentation Team
**Priority:** P1

**Tasks:**
1. [ ] Create TROUBLESHOOTING.md (common build errors, debugging)
2. [ ] Create VERSIONING.md (semantic versioning policy)
3. [ ] Update CONTRIBUTING.md with current test status
4. [ ] Create examples/ directory with working examples
5. [ ] Verify all code examples in docs actually compile

**Success Criteria:**
- ✅ Users can self-serve common issues via TROUBLESHOOTING.md
- ✅ All code examples compile and run
- ✅ Clear versioning expectations set

---

### Day 12-14: Reorganize Experimental Features

**Owner:** Development Team
**Priority:** P1

**Tasks:**
1. [ ] Create `vecstore-labs/` workspace crate
2. [ ] Move experimental modules to labs crate:
   - explainable.rs
   - temporal.rs
   - lineage.rs
   - privacy.rs
   - graph_vector.rs
   - learned_index.rs
   - gpu/ (until functional)
   - agent.rs
   - anomaly_detection.rs
   - ab_testing.rs
   - cost_optimizer.rs
   - federation.rs
   - neural_ranker.rs
   - ... (any other experimental modules)
3. [ ] Update Cargo.toml to reference labs crate as optional
4. [ ] Update documentation to clarify core vs. labs
5. [ ] Create vecstore-labs/README.md explaining purpose

**Success Criteria:**
- ✅ Core vecstore crate has <20 modules
- ✅ Experimental features isolated in labs crate
- ✅ Clear separation between stable and experimental
- ✅ Labs crate clearly marked as unstable

**Workspace structure:**
```
Cargo.toml (workspace root)
├── vecstore/           # Core library
│   ├── Cargo.toml
│   └── src/
│       ├── store/      # Stable modules only
│       ├── server/     # Beta server mode
│       └── lib.rs
└── vecstore-labs/      # Experimental features
    ├── Cargo.toml
    └── src/
        ├── explainable.rs
        ├── temporal.rs
        └── ... (all experimental)
```

---

## Week 3: Quality & Testing (Days 15-21)

### Day 15-17: Test Coverage Analysis

**Owner:** QA Team
**Priority:** P1

**Tasks:**
1. [ ] Install `tarpaulin` or `llvm-cov` for coverage
2. [ ] Run coverage analysis on core modules
3. [ ] Identify gaps in test coverage
4. [ ] Create issues for untested code paths
5. [ ] Add tests for critical paths first

**Success Criteria:**
- ✅ Coverage report generated for core modules
- ✅ >70% coverage on core store/ modules
- ✅ Critical paths (upsert, query, delete) >90% covered

**Commands:**
```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run coverage
cargo tarpaulin --all-features --out Html --output-dir coverage

# Open report
open coverage/index.html
```

---

### Day 18-19: Benchmark Creation

**Owner:** Development Team
**Priority:** P1

**Tasks:**
1. [ ] Create `benches/standard_workloads.rs`
2. [ ] Implement benchmarks:
   - Upsert 10K, 100K, 1M vectors
   - Query with different k values
   - Query with filters
   - Hybrid search
3. [ ] Document hardware specs
4. [ ] Run benchmarks and record baseline
5. [ ] Add benchmark CI job (on main branch only)

**Success Criteria:**
- ✅ Reproducible benchmarks exist
- ✅ Baseline numbers documented
- ✅ Can track performance regressions

**Example benchmark:**
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use vecstore::VecStore;

fn benchmark_upsert(c: &mut Criterion) {
    let mut group = c.benchmark_group("upsert");

    for size in [1_000, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut store = VecStore::open_in_memory(384).unwrap();
            b.iter(|| {
                for i in 0..size {
                    let id = format!("doc{}", i);
                    let vec: Vec<f32> = (0..384).map(|_| rand::random()).collect();
                    store.upsert(&id, &vec, serde_json::json!({})).unwrap();
                }
            });
        });
    }
}

criterion_group!(benches, benchmark_upsert);
criterion_main!(benches);
```

---

### Day 20-21: Data Safety Testing

**Owner:** QA Team
**Priority:** P1

**Tasks:**
1. [ ] Create integration test for crash recovery
2. [ ] Implement "kill switch" test (SIGKILL during write)
3. [ ] Test WAL recovery scenarios
4. [ ] Verify no data loss after crash
5. [ ] Document recovery procedures

**Success Criteria:**
- ✅ WAL recovery tested under crash scenarios
- ✅ Zero data loss demonstrated
- ✅ Recovery procedure documented

---

## Week 4: Release Preparation (Days 22-30)

### Day 22-24: Documentation Polish

**Owner:** Documentation Team
**Priority:** P1

**Tasks:**
1. [ ] Review all docs for consistency
2. [ ] Update CHANGELOG.md with accurate 0.2.0 notes
3. [ ] Write release announcement
4. [ ] Update README with current capabilities
5. [ ] Create migration guide (0.1.0 → 0.2.0)

**Success Criteria:**
- ✅ Documentation reflects actual state
- ✅ Clear release notes
- ✅ Migration path documented

---

### Day 25-27: Final Testing & Bug Fixes

**Owner:** Full Team
**Priority:** P0

**Tasks:**
1. [ ] Run full test suite on multiple platforms (Linux, macOS, Windows)
2. [ ] Test Python bindings
3. [ ] Test WASM build
4. [ ] Fix any critical bugs found
5. [ ] Create release checklist

**Success Criteria:**
- ✅ All tests pass on all platforms
- ✅ Python/WASM builds work
- ✅ No known critical bugs

---

### Day 28-29: Community Preparation

**Owner:** Community Team
**Priority:** P2

**Tasks:**
1. [ ] Set up Discord server
2. [ ] Create GitHub Discussions categories
3. [ ] Write "Getting Started" tutorial
4. [ ] Create issue templates
5. [ ] Draft blog post announcing 0.2.0

**Success Criteria:**
- ✅ Community channels ready
- ✅ Onboarding materials available
- ✅ Issue tracking organized

---

### Day 30: Release!

**Owner:** Release Manager
**Priority:** P0

**Tasks:**
1. [ ] Tag v0.2.0-alpha in git
2. [ ] Publish to crates.io
3. [ ] Publish to PyPI
4. [ ] Publish to npm (WASM)
5. [ ] Update documentation site
6. [ ] Announce on Discord, Twitter, Reddit
7. [ ] Submit to Hacker News / Reddit

**Success Criteria:**
- ✅ v0.2.0-alpha published to all registries
- ✅ Announcement distributed
- ✅ Community aware of release

---

## Success Metrics

### Week 1 Metrics
- [ ] Build: ❌ Failing → ✅ Passing
- [ ] Warnings: 101 → 0
- [ ] CI: None → ✅ Active
- [ ] Version: Inconsistent → Standardized

### Week 2 Metrics
- [ ] Documentation accuracy: <50% → 100%
- [ ] Code organization: Monolithic → Core + Labs
- [ ] User clarity: Unclear → MATURITY.md published

### Week 3 Metrics
- [ ] Test coverage: Unknown → >70% core
- [ ] Benchmarks: None → Baseline established
- [ ] Data safety: Untested → Crash recovery verified

### Week 4 Metrics
- [ ] Release readiness: Not ready → v0.2.0-alpha published
- [ ] Community: None → Discord + Discussions active
- [ ] Awareness: Low → Announcement distributed

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Can't fix compilation errors in time | Medium | High | Allocate extra dev time, seek help from community |
| Test failures reveal major bugs | High | Medium | Document known issues, defer non-critical to 0.3.0 |
| Benchmarks show poor performance | Medium | High | Be transparent, create performance roadmap |
| Community doesn't engage | Low | Low | Focus on quality over marketing initially |
| Scope creep continues | Medium | High | Enforce "no new features" rule strictly |

---

## Definition of Done

At end of 30 days, VecStore should have:

✅ **Clean build:** Zero errors, zero warnings
✅ **Passing tests:** >80% tests passing, failures documented
✅ **Clear positioning:** MATURITY.md published, alpha warnings visible
✅ **Organized code:** Core vs. Labs separation
✅ **CI/CD active:** Automated quality checks
✅ **Documented state:** All claims verified or corrected
✅ **Baseline metrics:** Test coverage, benchmarks, known issues
✅ **Community ready:** Discord, GitHub Discussions, issue templates
✅ **Release published:** v0.2.0-alpha on crates.io, PyPI, npm
✅ **Path forward:** Clear roadmap to 1.0.0

---

## Next Steps After This Sprint

1. **Review results:** Did we hit all targets?
2. **Gather feedback:** What do early users say?
3. **Plan Q1 2026:** Focus on stability → 1.0.0-beta
4. **Build community:** Start monthly calls, contributor onboarding
5. **Iterate:** Address feedback, fix bugs, maintain quality

---

**Document Owner:** Project Lead
**Review Cadence:** Weekly standup
**Update Frequency:** Daily progress tracking
**Completion Target:** January 26, 2026
