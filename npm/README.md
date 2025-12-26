# @vecstore/embeddings

Embedding integration for VecStore using [Transformers.js](https://huggingface.co/docs/transformers.js).

This package provides automatic embedding generation for VecStore in the browser and Node.js.

## Installation

```bash
npm install @vecstore/embeddings @vecstore/core @xenova/transformers
```

## Quick Start

```typescript
import { VecStoreWithEmbeddings } from '@vecstore/embeddings';

// Create store with automatic embedding generation
const store = await VecStoreWithEmbeddings.create({
  model: 'Xenova/all-MiniLM-L6-v2',
  progressCallback: (p) => console.log(`Loading: ${p.status}`)
});

// Add texts - embeddings generated automatically
await store.addTexts(
  ['Hello world', 'Machine learning is great', 'AI revolution'],
  [{ source: 'a' }, { source: 'b' }, { source: 'c' }]
);

// Search by text - query embedded automatically
const results = await store.search('artificial intelligence', 5);
for (const doc of results) {
  console.log(`${doc.pageContent} (score: ${doc.score.toFixed(3)})`);
}
```

## Supported Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `Xenova/all-MiniLM-L6-v2` (default) | 384 | Fast | Good | General purpose |
| `Xenova/all-mpnet-base-v2` | 768 | Medium | High | Best quality |
| `Xenova/multi-qa-MiniLM-L6-cos-v1` | 384 | Fast | Good | Q&A optimized |
| `Xenova/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Fast | Good | Multilingual |
| `Xenova/bge-small-en-v1.5` | 384 | Fast | Good | BGE small |
| `Xenova/bge-base-en-v1.5` | 768 | Medium | High | BGE base |

Any model from [Hugging Face](https://huggingface.co/models?library=transformers.js&sort=downloads) that supports feature extraction works.

## API Reference

### VecStoreWithEmbeddings

#### `create(options?)`

Create a new VecStore with embedding support.

```typescript
const store = await VecStoreWithEmbeddings.create({
  model: 'Xenova/all-MiniLM-L6-v2',
  device: 'webgpu', // 'webgpu', 'wasm', or 'auto'
  useBrowserCache: true,
  progressCallback: (p) => console.log(p.status)
});
```

#### `addTexts(texts, metadatas?, ids?)`

Add texts with automatic embedding generation.

```typescript
const ids = await store.addTexts(
  ['Document 1', 'Document 2'],
  [{ author: 'Alice' }, { author: 'Bob' }]
);
```

#### `search(query, k?, filter?)`

Search for similar documents by text query.

```typescript
const results = await store.search('query text', 10, "author = 'Alice'");
```

#### `hybridSearch(query, keywords, k?, alpha?, filter?)`

Hybrid search combining vector similarity and keyword matching.

```typescript
const results = await store.hybridSearch(
  'query text',
  'specific keywords',
  10,
  0.7  // 70% vector, 30% keyword
);
```

### EmbeddingModel

Standalone embedding model wrapper.

```typescript
import { EmbeddingModel } from '@vecstore/embeddings';

const model = new EmbeddingModel({ model: 'Xenova/all-MiniLM-L6-v2' });
await model.initialize();

const embeddings = await model.encode(['Hello', 'World']);
console.log(embeddings[0].length); // 384
```

## Browser Usage

```html
<script type="module">
import { VecStoreWithEmbeddings } from '@vecstore/embeddings';

const store = await VecStoreWithEmbeddings.create();
await store.addTexts(['Browser text search!']);
const results = await store.search('search');
console.log(results);
</script>
```

## Persistence

Export and import store data:

```typescript
// Export to JSON
const json = store.exportJson();
localStorage.setItem('vecstore', json);

// Import from JSON
const store2 = await VecStoreWithEmbeddings.fromJson(
  localStorage.getItem('vecstore')!
);
```

## License

Apache-2.0
