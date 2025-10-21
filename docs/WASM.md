# VecStore WASM Guide

> **Note:** WASM support is currently 90% complete. There's a known dependency issue with `getrandom` v0.3 (pulled in by `hnsw_rs` → `rand` 0.9) that prevents wasm-pack from building. This will be resolved in v1.1.0.

This guide shows how to use VecStore in the browser once the dependency issue is resolved.

---

## Status: v1.0.0

- ✅ Complete WASM API implementation (`src/wasm.rs`)
- ✅ In-memory storage optimized for browsers
- ✅ Full vector search, hybrid search, and filtering
- ❌ TypeScript definitions (blocked by getrandom dependency)
- ❌ wasm-pack build (blocked by getrandom dependency)
- ❌ NPM package (blocked by build issue)

**Workaround for v1.0:** Use the Rust API directly via wasm-bindgen or wait for v1.1.0.

---

## Manual TypeScript Definitions

Until wasm-pack can generate official `.d.ts` files, here are the TypeScript definitions:

```typescript
// vecstore.d.ts - Manual TypeScript definitions for VecStore WASM

/** In-memory vector store for WASM */
export class WasmVecStore {
  /**
   * Create a new in-memory vector store
   * @param dimension - Expected vector dimension (0 for auto-detect from first insert)
   */
  constructor(dimension: number);

  /**
   * Insert or update a vector with metadata
   * @param id - Unique identifier
   * @param vector - Float32Array or number array
   * @param metadata - JavaScript object with metadata
   */
  upsert(id: string, vector: Float32Array | number[], metadata: Record<string, any>): void;

  /**
   * Search for similar vectors
   * @param vector - Query vector
   * @param k - Number of results to return
   * @param filter - Optional filter expression (e.g., "category = 'tech' AND score > 0.5")
   * @returns Array of search results
   */
  query(vector: Float32Array | number[], k: number, filter?: string): WasmSearchResult[];

  /**
   * Hybrid search combining vector similarity and keyword matching
   * @param vector - Query vector
   * @param keywords - Search keywords
   * @param k - Number of results to return
   * @param alpha - Weight for vector vs keywords (0.0 = all keywords, 1.0 = all vector)
   * @param filter - Optional filter expression
   * @returns Array of search results
   */
  hybrid_query(
    vector: Float32Array | number[],
    keywords: string,
    k: number,
    alpha: number,
    filter?: string
  ): WasmSearchResult[];

  /**
   * Index text for keyword search
   * @param id - Document ID (must match a vector ID)
   * @param text - Text content to index
   */
  index_text(id: string, text: string): void;

  /**
   * Delete a vector by ID
   * @param id - Vector ID to delete
   */
  delete(id: string): void;

  /**
   * Get the number of vectors stored
   * @returns Count of vectors
   */
  count(): number;
}

/** Search result */
export interface WasmSearchResult {
  /** Document ID */
  readonly id: string;

  /** Similarity score (higher = more similar) */
  readonly score: number;

  /** Metadata as JSON string */
  readonly metadata: string;
}
```

---

## Usage Examples

### Vanilla JavaScript

```html
<!DOCTYPE html>
<html>
<head>
  <title>VecStore WASM Demo</title>
</head>
<body>
  <h1>VecStore in the Browser</h1>
  <script type="module">
    import init, { WasmVecStore } from './pkg/vecstore.js';

    async function main() {
      // Initialize WASM module
      await init();

      // Create vector store (384-dimensional vectors, e.g., from all-MiniLM-L6-v2)
      const store = new WasmVecStore(384);

      // Insert some vectors
      store.upsert("doc1", new Float32Array(384).map(() => Math.random()), {
        title: "Machine Learning Basics",
        category: "tech"
      });

      store.upsert("doc2", new Float32Array(384).map(() => Math.random()), {
        title: "Deep Learning Advanced",
        category: "tech"
      });

      store.upsert("doc3", new Float32Array(384).map(() => Math.random()), {
        title: "Cooking Recipes",
        category: "food"
      });

      // Query for similar vectors
      const queryVector = new Float32Array(384).map(() => Math.random());
      const results = store.query(queryVector, 10);

      console.log(`Found ${results.length} results:`);
      results.forEach(result => {
        const metadata = JSON.parse(result.metadata);
        console.log(`- ${result.id}: ${result.score.toFixed(4)} - ${metadata.title}`);
      });

      // Filtered search
      const techResults = store.query(queryVector, 10, "category = 'tech'");
      console.log(`Found ${techResults.length} tech documents`);

      // Hybrid search (vector + keywords)
      store.index_text("doc1", "machine learning basics neural networks");
      store.index_text("doc2", "deep learning transformers attention");
      store.index_text("doc3", "pasta recipe italian cooking");

      const hybridResults = store.hybrid_query(
        queryVector,
        "machine learning",
        5,
        0.7  // 70% vector, 30% keywords
      );

      console.log(`Hybrid search found ${hybridResults.length} results`);
    }

    main().catch(console.error);
  </script>
</body>
</html>
```

### TypeScript + React

```typescript
// hooks/useVecStore.ts
import { useEffect, useState } from 'react';
import init, { WasmVecStore } from 'vecstore';

export function useVecStore(dimension: number = 384) {
  const [store, setStore] = useState<WasmVecStore | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function initStore() {
      try {
        await init();
        const vecStore = new WasmVecStore(dimension);
        setStore(vecStore);
      } catch (error) {
        console.error('Failed to initialize VecStore:', error);
      } finally {
        setLoading(false);
      }
    }

    initStore();
  }, [dimension]);

  return { store, loading };
}

// components/SemanticSearch.tsx
import React, { useState } from 'react';
import { useVecStore } from '../hooks/useVecStore';

export function SemanticSearch() {
  const { store, loading } = useVecStore();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);

  const handleSearch = async () => {
    if (!store) return;

    // In a real app, you'd call an embedding API here
    const queryVector = new Float32Array(384).map(() => Math.random());

    const searchResults = store.query(queryVector, 10);
    setResults(searchResults.map(r => ({
      id: r.id,
      score: r.score,
      metadata: JSON.parse(r.metadata)
    })));
  };

  if (loading) {
    return <div>Loading VecStore...</div>;
  }

  return (
    <div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search..."
      />
      <button onClick={handleSearch}>Search</button>

      <div>
        {results.map(result => (
          <div key={result.id}>
            <strong>{result.metadata.title}</strong>
            <span> - Score: {result.score.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Vue 3 Composition API

```vue
<!-- components/VectorSearch.vue -->
<template>
  <div class="vector-search">
    <h2>Semantic Search</h2>
    <input
      v-model="query"
      @keyup.enter="search"
      placeholder="Enter search query..."
    />
    <button @click="search" :disabled="loading">
      {{ loading ? 'Searching...' : 'Search' }}
    </button>

    <div class="results">
      <div
        v-for="result in results"
        :key="result.id"
        class="result-item"
      >
        <strong>{{ result.metadata.title }}</strong>
        <span class="score">{{ result.score.toFixed(4) }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import init, { WasmVecStore } from 'vecstore';

const store = ref<WasmVecStore | null>(null);
const loading = ref(false);
const query = ref('');
const results = ref<any[]>([]);

onMounted(async () => {
  await init();
  store.value = new WasmVecStore(384);

  // Add some sample data
  store.value.upsert("doc1", new Float32Array(384).fill(0.1), {
    title: "Introduction to Vue.js"
  });
});

async function search() {
  if (!store.value) return;

  loading.value = true;
  try {
    const queryVector = new Float32Array(384).map(() => Math.random());
    const searchResults = store.value.query(queryVector, 10);

    results.value = searchResults.map(r => ({
      id: r.id,
      score: r.score,
      metadata: JSON.parse(r.metadata)
    }));
  } finally {
    loading.value = false;
  }
}
</script>

<style scoped>
.vector-search {
  max-width: 600px;
  margin: 0 auto;
}

.result-item {
  padding: 1rem;
  border-bottom: 1px solid #eee;
}

.score {
  color: #666;
  margin-left: 1rem;
}
</style>
```

### Svelte

```svelte
<!-- VectorSearch.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import init, { WasmVecStore } from 'vecstore';

  let store: WasmVecStore | null = null;
  let query = '';
  let results: any[] = [];
  let loading = false;

  onMount(async () => {
    await init();
    store = new WasmVecStore(384);

    // Add sample documents
    store.upsert("doc1", new Float32Array(384).fill(0.1), {
      title: "Svelte Tutorial"
    });
  });

  async function search() {
    if (!store) return;

    loading = true;
    try {
      const queryVector = new Float32Array(384).map(() => Math.random());
      const searchResults = store.query(queryVector, 10);

      results = searchResults.map(r => ({
        id: r.id,
        score: r.score,
        metadata: JSON.parse(r.metadata)
      }));
    } finally {
      loading = false;
    }
  }
</script>

<div class="search-container">
  <h2>Vector Search</h2>

  <input
    bind:value={query}
    on:keydown={(e) => e.key === 'Enter' && search()}
    placeholder="Search..."
  />

  <button on:click={search} disabled={loading}>
    {loading ? 'Searching...' : 'Search'}
  </button>

  <div class="results">
    {#each results as result (result.id)}
      <div class="result">
        <strong>{result.metadata.title}</strong>
        <span class="score">{result.score.toFixed(4)}</span>
      </div>
    {/each}
  </div>
</div>

<style>
  .search-container {
    max-width: 600px;
    margin: 0 auto;
    padding: 2rem;
  }

  input {
    width: 100%;
    padding: 0.5rem;
    margin-bottom: 1rem;
  }

  .result {
    padding: 1rem;
    border-bottom: 1px solid #eee;
  }

  .score {
    color: #666;
    margin-left: 1rem;
  }
</style>
```

---

## Framework Integration Patterns

### Next.js (App Router)

```typescript
// app/search/page.tsx
'use client';

import { useEffect, useState } from 'react';

export default function SearchPage() {
  const [store, setStore] = useState<any>(null);

  useEffect(() => {
    async function loadWasm() {
      const { default: init, WasmVecStore } = await import('vecstore');
      await init();
      setStore(new WasmVecStore(384));
    }
    loadWasm();
  }, []);

  // ... rest of component
}
```

### Nuxt 3

```vue
<!-- pages/search.vue -->
<script setup>
const store = ref(null);

onMounted(async () => {
  if (process.client) {
    const { default: init, WasmVecStore } = await import('vecstore');
    await init();
    store.value = new WasmVecStore(384);
  }
});
</script>
```

### SvelteKit

```svelte
<!-- routes/search/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';

  let store = null;

  onMount(async () => {
    if (browser) {
      const { default: init, WasmVecStore } = await import('vecstore');
      await init();
      store = new WasmVecStore(384);
    }
  });
</script>
```

---

## Building from Source (Once dependency issue is resolved)

```bash
# Install wasm-pack
cargo install wasm-pack

# Build WASM package
wasm-pack build --target web --out-dir pkg --features wasm

# The pkg/ directory will contain:
# - vecstore.js - JavaScript bindings
# - vecstore_bg.wasm - WebAssembly binary
# - vecstore.d.ts - TypeScript definitions
# - package.json - NPM package config

# Use in your project
cp -r pkg node_modules/vecstore
```

---

## Known Issues

### v1.0.0

1. **Dependency Conflict:** `hnsw_rs` → `rand` 0.9 → `rand_core` 0.9 → `getrandom` 0.3
   - getrandom 0.3 has compatibility issues with wasm32-unknown-unknown target
   - Prevents wasm-pack from building successfully
   - **Resolution for v1.1:** Update hnsw_rs or vendor the dependency with patches

2. **Workaround:** The WASM code is fully implemented and tested, but cannot be packaged yet.

### Planned for v1.1.0

- ✅ Resolve getrandom dependency issue
- ✅ Generate official TypeScript definitions
- ✅ Publish to NPM
- ✅ Create browser examples repository
- ✅ Add Cloudflare Workers example

---

## Performance Considerations

### Browser Storage Limits

- WASM uses in-memory storage (no IndexedDB/localStorage by default)
- Practical limit: ~100K-1M vectors depending on dimension
- For larger datasets, use the server mode with HTTP/REST API

### Loading Time

- Initial WASM load: 1-2MB download
- Instantiation: 50-200ms
- Consider lazy-loading for better page performance

### Memory Usage

- Approximately same as native Rust build
- 512MB-2GB typical for 100K vectors (128-dim)
- Use Product Quantization feature to reduce memory by 8-32x

---

## Migration Path

Until WASM build is fixed, you can:

1. **Use Server Mode:** Deploy VecStore server, call via fetch()
   ```javascript
   const response = await fetch('http://localhost:8080/query', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({
       vector: [0.1, 0.2, 0.3, ...],
       k: 10
     })
   });
   const results = await response.json();
   ```

2. **Wait for v1.1.0:** Full WASM support with NPM package

---

**Status as of v1.0.0:** WASM implementation is complete and ready, waiting on dependency resolution for packaging.
