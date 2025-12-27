# @vecstore/core

The Explainable Vector Database - WebAssembly build for browsers and edge environments.

## Features

- **Browser-native**: Run vector search entirely in the browser
- **Privacy-first**: No data leaves the client
- **Explainable**: Understand WHY results matched
- **Offline-capable**: Works without network connectivity
- **Edge-ready**: Deploy to Cloudflare Workers, Deno Deploy, etc.

## Installation

```bash
npm install @vecstore/core
```

## Usage

```javascript
import init, { VecStore } from '@vecstore/core';

// Initialize WASM module
await init();

// Create a store
const store = new VecStore(384); // dimension

// Add vectors
store.add("doc1", new Float32Array([0.1, 0.2, ...]));
store.add("doc2", new Float32Array([0.3, 0.4, ...]));

// Search
const results = store.search(queryVector, 10);
console.log(results);
```

## With Embeddings

Use with `@vecstore/embeddings` for automatic text embedding:

```bash
npm install @vecstore/core @vecstore/embeddings
```

```javascript
import init, { VecStore } from '@vecstore/core';
import { EmbeddingPipeline } from '@vecstore/embeddings';

await init();

const embedder = new EmbeddingPipeline('Xenova/all-MiniLM-L6-v2');
const store = new VecStore(384);

// Add documents
const embedding = await embedder.embed("Hello world");
store.add("doc1", embedding);

// Semantic search
const queryEmbed = await embedder.embed("greeting");
const results = store.search(queryEmbed, 5);
```

## Building from Source

```bash
# Install wasm-pack
cargo install wasm-pack

# Build WASM package
wasm-pack build --target web --out-dir wasm-pkg -- --features wasm
```

## Browser Support

- Chrome 89+
- Firefox 89+
- Safari 15+
- Edge 89+

## License

Apache-2.0
