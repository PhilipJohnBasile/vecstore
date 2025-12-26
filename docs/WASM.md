# VecStore WASM Guide

Run high-performance vector search directly in the browser. No backend required.

```bash
npm install @vecstore/core
```

Or with built-in embedding support:

```bash
npm install @vecstore/embeddings @vecstore/core @xenova/transformers
```

---

## Why Browser Vector Search?

- **Privacy-first**: Data never leaves the user's device
- **Offline-capable**: Works without network connection
- **Low latency**: Sub-millisecond search on 100K+ vectors
- **Zero infrastructure**: No servers to deploy or maintain

---

## Quick Start

### Installation

```bash
npm install vecstore-wasm
# or
yarn add vecstore-wasm
# or
pnpm add vecstore-wasm
```

### Basic Usage

```javascript
import init, { WasmVecStore } from 'vecstore-wasm';

async function main() {
  // Initialize WASM module
  await init();

  // Create vector store (384-dim for all-MiniLM-L6-v2 embeddings)
  const store = new WasmVecStore(384);

  // Add vectors with metadata
  store.upsert("doc1", new Float32Array([0.1, 0.2, ...]), {
    title: "Introduction to Machine Learning",
    category: "tech"
  });

  // Search for similar vectors
  const results = store.query(queryVector, 10);

  results.forEach(result => {
    console.log(`${result.id}: ${result.score}`);
  });
}

main();
```

---

## Built-in Embeddings (Recommended)

The `@vecstore/embeddings` package provides automatic embedding generation using Transformers.js:

```typescript
import { VecStoreWithEmbeddings } from '@vecstore/embeddings';

// Create store with automatic embedding generation
const store = await VecStoreWithEmbeddings.create({
  model: 'Xenova/all-MiniLM-L6-v2',
  progressCallback: (p) => console.log(`Loading: ${p.status}`)
});

// Add texts - embeddings generated automatically
await store.addTexts(
  ['Hello world', 'Machine learning is great'],
  [{ source: 'a' }, { source: 'b' }]
);

// Search by text - query embedded automatically
const results = await store.search('AI', 5);
for (const doc of results) {
  console.log(`${doc.pageContent} (score: ${doc.score.toFixed(3)})`);
}
```

### Supported Models

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| `Xenova/all-MiniLM-L6-v2` (default) | 384 | Fast | Good |
| `Xenova/all-mpnet-base-v2` | 768 | Medium | High |
| `Xenova/bge-small-en-v1.5` | 384 | Fast | Good |
| `Xenova/bge-base-en-v1.5` | 768 | Medium | High |

---

## Manual Embeddings

If you need more control, use the core WASM module directly and generate embeddings manually:

### Option 1: Transformers.js (Local, Privacy-First)

```javascript
import { pipeline } from '@xenova/transformers';
import init, { WasmVecStore } from 'vecstore-wasm';

// Load embedding model (runs entirely in browser)
const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

await init();
const store = new WasmVecStore(384);

// Generate embeddings locally
async function embed(text) {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return new Float32Array(output.data);
}

// Index documents
const docs = [
  { id: "1", text: "Machine learning is a subset of AI", category: "tech" },
  { id: "2", text: "Neural networks mimic the human brain", category: "tech" },
  { id: "3", text: "Italian pasta recipes from Rome", category: "food" }
];

for (const doc of docs) {
  const vector = await embed(doc.text);
  store.upsert(doc.id, vector, { text: doc.text, category: doc.category });
  store.index_text(doc.id, doc.text); // Enable hybrid search
}

// Search
const queryVector = await embed("How does AI learn?");
const results = store.query(queryVector, 5);
```

### Option 2: OpenAI API

```javascript
async function embedWithOpenAI(text) {
  const response = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'text-embedding-3-small',
      input: text
    })
  });

  const data = await response.json();
  return new Float32Array(data.data[0].embedding);
}
```

### Option 3: Hugging Face Inference API

```javascript
async function embedWithHuggingFace(text) {
  const response = await fetch(
    'https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2',
    {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${HF_TOKEN}` },
      body: JSON.stringify({ inputs: text })
    }
  );

  const embeddings = await response.json();
  return new Float32Array(embeddings);
}
```

---

## Features

### Filtered Search

```javascript
// Filter by metadata fields
const techDocs = store.query(queryVector, 10, "category = 'tech'");

// Complex filters
const filtered = store.query(queryVector, 10,
  "category = 'tech' AND score > 0.5"
);
```

### Hybrid Search (Vector + Keywords)

```javascript
// Index text content
store.index_text("doc1", "machine learning neural networks");
store.index_text("doc2", "deep learning transformers attention");

// Hybrid search: 70% vector similarity, 30% keyword matching
const results = store.hybrid_query(
  queryVector,
  "machine learning",
  10,    // k results
  0.7    // alpha: vector weight
);
```

### Import/Export (Persistence)

```javascript
// Export store to JSON (for IndexedDB/localStorage)
const data = store.export_json();
localStorage.setItem('vectorStore', data);

// Import from JSON
const savedData = localStorage.getItem('vectorStore');
if (savedData) {
  store.import_json(savedData);
}
```

### IndexedDB Persistence Example

```javascript
async function saveToIndexedDB(store) {
  const data = store.export_json();

  return new Promise((resolve, reject) => {
    const request = indexedDB.open('VecStoreDB', 1);

    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      db.createObjectStore('stores', { keyPath: 'name' });
    };

    request.onsuccess = (e) => {
      const db = e.target.result;
      const tx = db.transaction('stores', 'readwrite');
      tx.objectStore('stores').put({ name: 'main', data });
      tx.oncomplete = resolve;
    };

    request.onerror = reject;
  });
}

async function loadFromIndexedDB(store) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('VecStoreDB', 1);

    request.onsuccess = (e) => {
      const db = e.target.result;
      const tx = db.transaction('stores', 'readonly');
      const getRequest = tx.objectStore('stores').get('main');

      getRequest.onsuccess = () => {
        if (getRequest.result) {
          store.import_json(getRequest.result.data);
        }
        resolve();
      };
    };

    request.onerror = reject;
  });
}
```

---

## Framework Examples

### React

```tsx
import { useEffect, useState } from 'react';
import init, { WasmVecStore } from 'vecstore-wasm';

function useVecStore(dimension = 384) {
  const [store, setStore] = useState<WasmVecStore | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    init().then(() => {
      setStore(new WasmVecStore(dimension));
      setReady(true);
    });
  }, [dimension]);

  return { store, ready };
}

function SearchComponent() {
  const { store, ready } = useVecStore();
  const [results, setResults] = useState([]);

  const search = async (query: string) => {
    if (!store) return;

    // Get embedding (use your preferred method)
    const vector = await getEmbedding(query);
    const searchResults = store.query(vector, 10);

    setResults(searchResults.map(r => ({
      id: r.id,
      score: r.score,
      metadata: JSON.parse(r.metadata)
    })));
  };

  if (!ready) return <div>Loading...</div>;

  return (
    <div>
      <input onChange={(e) => search(e.target.value)} />
      {results.map(r => (
        <div key={r.id}>{r.metadata.title} ({r.score.toFixed(3)})</div>
      ))}
    </div>
  );
}
```

### Vue 3

```vue
<script setup>
import { ref, onMounted } from 'vue';
import init, { WasmVecStore } from 'vecstore-wasm';

const store = ref(null);
const results = ref([]);

onMounted(async () => {
  await init();
  store.value = new WasmVecStore(384);
});

async function search(query) {
  const vector = await getEmbedding(query);
  const searchResults = store.value.query(vector, 10);
  results.value = searchResults.map(r => ({
    id: r.id,
    score: r.score,
    ...JSON.parse(r.metadata)
  }));
}
</script>

<template>
  <input @input="search($event.target.value)" />
  <div v-for="r in results" :key="r.id">
    {{ r.title }} ({{ r.score.toFixed(3) }})
  </div>
</template>
```

### Next.js (App Router)

```tsx
'use client';

import { useEffect, useState } from 'react';

export default function SearchPage() {
  const [store, setStore] = useState(null);

  useEffect(() => {
    // Dynamic import for client-side only
    import('vecstore-wasm').then(async ({ default: init, WasmVecStore }) => {
      await init();
      setStore(new WasmVecStore(384));
    });
  }, []);

  // ... rest of component
}
```

---

## TypeScript Definitions

Full TypeScript support is included:

```typescript
import init, { WasmVecStore, WasmSearchResult } from 'vecstore-wasm';

const store: WasmVecStore = new WasmVecStore(384);

// All methods are fully typed
store.upsert(id: string, vector: Float32Array, metadata: object): void;
store.query(vector: Float32Array, k: number, filter?: string): WasmSearchResult[];
store.hybrid_query(vector: Float32Array, keywords: string, k: number, alpha: number, filter?: string): WasmSearchResult[];
store.index_text(id: string, text: string): void;
store.delete(id: string): void;
store.count(): number;
store.export_json(): string;
store.import_json(data: string): void;
```

---

## Performance

Benchmarks on M1 MacBook Pro (128-dim vectors):

| Dataset Size | Search Latency | Throughput |
|--------------|----------------|------------|
| 1,000 | 0.3ms | 3,400 qps |
| 10,000 | 0.7ms | 1,400 qps |
| 100,000 | 0.2ms | 5,800 qps |
| 1,000,000 | 0.2ms | 5,000 qps |

### Memory Usage

| Vectors | Dimensions | Memory |
|---------|------------|--------|
| 10,000 | 384 | ~20MB |
| 100,000 | 384 | ~180MB |
| 100,000 | 1536 | ~650MB |

### Practical Limits

- **Recommended**: Up to 100K vectors for smooth browser experience
- **Maximum**: ~1M vectors (depends on available RAM)
- **For larger datasets**: Use VecStore server mode with HTTP API

---

## Use Cases

1. **Offline-first search**: Documentation, notes, personal knowledge bases
2. **Privacy-sensitive apps**: Medical, legal, financial documents
3. **Edge computing**: IoT dashboards, embedded systems
4. **Prototyping**: Quickly test semantic search without backend
5. **Educational**: Interactive ML/AI demos

---

## Building from Source

```bash
# Install wasm-pack
cargo install wasm-pack

# Build WASM package
wasm-pack build --target web --out-dir pkg --features wasm

# Test locally
cd pkg && npm link
cd ../your-app && npm link vecstore-wasm
```

---

## Troubleshooting

### "WASM module not initialized"

Make sure to call `await init()` before using the store:

```javascript
import init, { WasmVecStore } from 'vecstore-wasm';

await init();  // Required!
const store = new WasmVecStore(384);
```

### Large bundle size

Use dynamic imports to lazy-load:

```javascript
const loadVecStore = async () => {
  const { default: init, WasmVecStore } = await import('vecstore-wasm');
  await init();
  return new WasmVecStore(384);
};
```

### Memory issues

Monitor memory usage and consider:
- Reducing vector dimensions
- Using fewer vectors
- Implementing pagination for large result sets

---

## Related

- [VecStore GitHub](https://github.com/PhilipJohnBasile/vecstore)
- [npm package](https://www.npmjs.com/package/vecstore-wasm)
- [Transformers.js](https://huggingface.co/docs/transformers.js) - Browser embeddings
