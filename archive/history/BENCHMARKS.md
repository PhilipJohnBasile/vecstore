# vecstore Performance Benchmarks

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ⚡ vecstore - Lightning Fast Vector Search in Rust ⚡     ║
║                                                               ║
║   📊 Comprehensive Performance Benchmarks                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

## 🚀 Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark group
cargo bench --bench vecstore_bench -- insert

# Generate HTML reports (in target/criterion/)
cargo bench --bench vecstore_bench
open target/criterion/report/index.html
```

## 📈 Benchmark Suites

### 1. **Insert Performance** 🔨

Measures single-record insertion throughput.

```
┌─────────────────────────────────────────┐
│  Dataset Size  │  Throughput (ops/sec)  │
├─────────────────────────────────────────┤
│     10 docs    │       ~XXX,XXX         │
│    100 docs    │       ~XXX,XXX         │
│   1000 docs    │       ~XXX,XXX         │
└─────────────────────────────────────────┘
```

**What it tests**: Individual `upsert()` calls with HNSW index updates

### 2. **Batch Insert Performance** ⚡

Measures parallel batch ingestion throughput.

```
┌─────────────────────────────────────────┐
│  Batch Size    │  Throughput (ops/sec)  │
├─────────────────────────────────────────┤
│    100 docs    │       ~XXX,XXX         │
│   1000 docs    │       ~XXX,XXX         │
│   5000 docs    │       ~XXX,XXX         │
└─────────────────────────────────────────┘
```

**What it tests**: `batch_upsert()` with Rayon parallelization

### 3. **Query Performance** 🔍

Measures k-NN search latency across different dataset sizes.

```
┌──────────────────────────────────────────────┐
│  Dataset       │  Query Latency (k=10)       │
├──────────────────────────────────────────────┤
│  1K x 128D     │       ~XXX µs               │
│  5K x 128D     │       ~XXX µs               │
│ 10K x 128D     │       ~XXX µs               │
└──────────────────────────────────────────────┘
```

**What it tests**: HNSW approximate nearest neighbor search

### 4. **Filtered Query Performance** 🎯

Measures query performance with metadata filtering.

```
┌──────────────────────────────────────────────┐
│  Dataset       │  Filtered Query Latency     │
├──────────────────────────────────────────────┤
│  1K vectors    │       ~XXX µs               │
│  5K vectors    │       ~XXX µs               │
│ 10K vectors    │       ~XXX µs               │
└──────────────────────────────────────────────┘
```

**What it tests**: Combined HNSW search + metadata filter evaluation

### 5. **Persistence Performance** 💾

Measures save/load performance.

```
┌──────────────────────────────────────────────┐
│  Dataset       │  Save Time  │  Load Time    │
├──────────────────────────────────────────────┤
│   100 docs     │   ~XXX ms   │   ~XXX ms     │
│  1000 docs     │   ~XXX ms   │   ~XXX ms     │
│  5000 docs     │   ~XXX ms   │   ~XXX ms     │
└──────────────────────────────────────────────┘
```

**What it tests**: File-backed persistence (JSON + bincode)

### 6. **Dimensional Scaling** 📐

Measures query performance across different vector dimensions.

```
┌──────────────────────────────────────────────┐
│  Dimensions    │  Query Latency (1K vecs)    │
├──────────────────────────────────────────────┤
│     64D        │       ~XXX µs               │
│    128D        │       ~XXX µs               │
│    256D        │       ~XXX µs               │
│    512D        │       ~XXX µs               │
│   1024D        │       ~XXX µs               │
└──────────────────────────────────────────────┘
```

**What it tests**: Impact of vector dimensionality on search

### 7. **Complex Filters** 🧩

Measures filter complexity impact.

```
┌──────────────────────────────────────────────┐
│  Filter Type            │  Latency           │
├──────────────────────────────────────────────┤
│  Simple (field = val)   │   ~XXX µs          │
│  AND (2 conditions)     │   ~XXX µs          │
│  Nested (OR + AND)      │   ~XXX µs          │
└──────────────────────────────────────────────┘
```

**What it tests**: Boolean filter expression evaluation overhead

## 🎯 Performance Characteristics

### Memory Usage

```
    Memory (MB)
       │
   800 │                              ╱
       │                          ╱
   600 │                      ╱
       │                  ╱
   400 │              ╱
       │          ╱
   200 │      ╱
       │  ╱
     0 └──────────────────────────────────
       0    2K    5K    10K   20K   50K
                 Dataset Size
```

**Approximate**: ~16 bytes per vector dimension + metadata overhead

### Query Latency vs Dataset Size

```
    Latency (µs)
       │
  1000 │                              •
       │                          •
   800 │                      •
       │                  •
   600 │              •
       │          •
   400 │      •
       │  •
   200 └──────────────────────────────────
       0    2K    5K    10K   20K   50K
                 Dataset Size
```

**HNSW Characteristic**: Sub-linear scaling (log n complexity)

## 🏆 Competitive Comparison

_Coming soon: vecstore vs ChromaDB vs FAISS vs hnswlib_

## 📝 Notes

### HNSW Parameters
- Default M: 16 (connections per layer)
- Default ef_construction: 200
- Default ef_search: depends on k

### Test Environment
Run `cargo bench` to see your specific environment details.

### Reproducibility
All benchmarks use:
- Criterion.rs with statistical rigor
- Warm-up iterations
- Multiple samples for confidence intervals
- Outlier detection

## 🔧 Benchmark Configuration

Edit `benches/vecstore_bench.rs` to:
- Add custom test scenarios
- Adjust dataset sizes
- Test different HNSW parameters
- Compare distance metrics

## 📊 Viewing Results

```bash
# Generate HTML reports
cargo bench

# Open in browser
open target/criterion/report/index.html
```

The HTML reports include:
- 📈 Performance graphs over time
- 📊 Statistical analysis
- 🎯 Regression detection
- 📉 Outlier identification

---

```
╭───────────────────────────────────────────╮
│  Made with ⚡ by the vecstore team        │
│  Fast • Lightweight • In-Process          │
╰───────────────────────────────────────────╯
```
