# Verification notes

Documentation refresh: September 10, 2026.

## Reproduce the focused checks

```bash
cargo test --locked --lib
cargo run --locked --example quickstart
```

## Local result

The initial macOS fresh-checkout attempt could not complete because native build scripts were terminated or stalled before program startup. The existing CI checks the Rust library; the quick-start example has been added to that same validation path.

## Hosted checks

[Workflow and current runs](https://github.com/PhilipJohnBasile/vecstore/actions/workflows/ci.yml). Inspect the commit and individual jobs when using a run as evidence; a successful earlier run does not validate later source changes.

## Release and coverage scope

The source version is 0.1.0. No GitHub release was published at the start of this refresh. Optional Python, WASM, server, and GPU configurations are separate from the default Rust library gate.
