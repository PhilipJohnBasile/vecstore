/**
 * @vecstore/embeddings - Embedding integration for VecStore using Transformers.js
 *
 * This module provides automatic embedding generation for VecStore in the browser
 * and Node.js using Transformers.js models.
 *
 * @example
 * ```typescript
 * import { VecStoreWithEmbeddings } from '@vecstore/embeddings';
 *
 * const store = await VecStoreWithEmbeddings.create('./my_store', {
 *   model: 'Xenova/all-MiniLM-L6-v2'
 * });
 *
 * await store.addTexts(['Hello world', 'Machine learning is great']);
 * const results = await store.search('AI', 5);
 * ```
 */

// Types for external dependencies
interface Pipeline {
  (texts: string | string[], options?: { pooling?: string; normalize?: boolean }): Promise<{
    data: Float32Array;
    dims: number[];
  }>;
}

interface TransformersModule {
  pipeline(task: string, model: string, options?: Record<string, unknown>): Promise<Pipeline>;
  env: {
    allowLocalModels: boolean;
    useBrowserCache: boolean;
  };
}

// Type for the WasmVecStore from @vecstore/core
interface WasmVecStore {
  new (dimension: number): WasmVecStore;
  upsert(id: string, vector: Float32Array | number[], metadata: Record<string, unknown>): void;
  query(vector: Float32Array | number[], k: number, filter?: string | null): SearchResult[];
  hybrid_query(
    vector: Float32Array | number[],
    keywords: string,
    k: number,
    alpha: number,
    filter?: string | null
  ): SearchResult[];
  index_text(id: string, text: string): void;
  remove(id: string): void;
  len(): number;
  is_empty(): boolean;
  export_json(): string;
  import_json(json: string): void;
}

interface SearchResult {
  id: string;
  score: number;
  metadata: string;
}

export interface EmbeddingModelOptions {
  /** Model name from HuggingFace (default: 'Xenova/all-MiniLM-L6-v2') */
  model?: string;
  /** Device to run on: 'webgpu', 'wasm', or 'auto' (default: 'auto') */
  device?: 'webgpu' | 'wasm' | 'auto';
  /** Whether to use browser cache for model files (default: true) */
  useBrowserCache?: boolean;
  /** Progress callback for model loading */
  progressCallback?: (progress: { status: string; progress?: number }) => void;
}

export interface Document {
  pageContent: string;
  metadata: Record<string, unknown>;
}

export interface ScoredDocument extends Document {
  score: number;
}

/**
 * Supported embedding models with their properties.
 */
export const SUPPORTED_MODELS = {
  'Xenova/all-MiniLM-L6-v2': { dimension: 384, speed: 'fast', quality: 'good' },
  'Xenova/all-mpnet-base-v2': { dimension: 768, speed: 'medium', quality: 'high' },
  'Xenova/multi-qa-MiniLM-L6-cos-v1': { dimension: 384, speed: 'fast', quality: 'good' },
  'Xenova/paraphrase-multilingual-MiniLM-L12-v2': { dimension: 384, speed: 'fast', quality: 'good' },
  'Xenova/bge-small-en-v1.5': { dimension: 384, speed: 'fast', quality: 'good' },
  'Xenova/bge-base-en-v1.5': { dimension: 768, speed: 'medium', quality: 'high' },
} as const;

export type ModelName = keyof typeof SUPPORTED_MODELS;

/**
 * Embedding model wrapper for Transformers.js
 */
export class EmbeddingModel {
  private pipeline: Pipeline | null = null;
  private modelName: string;
  private dimension: number | null = null;
  private options: EmbeddingModelOptions;

  constructor(options: EmbeddingModelOptions = {}) {
    this.modelName = options.model || 'Xenova/all-MiniLM-L6-v2';
    this.options = options;

    // Get dimension from known models
    const knownModel = SUPPORTED_MODELS[this.modelName as ModelName];
    if (knownModel) {
      this.dimension = knownModel.dimension;
    }
  }

  /**
   * Initialize the embedding model.
   * Must be called before using encode().
   */
  async initialize(): Promise<void> {
    // Dynamic import of transformers
    const transformers = (await import('@xenova/transformers')) as TransformersModule;

    // Configure environment
    if (this.options.useBrowserCache !== false) {
      transformers.env.useBrowserCache = true;
    }

    // Load pipeline
    this.pipeline = await transformers.pipeline(
      'feature-extraction',
      this.modelName,
      {
        progress_callback: this.options.progressCallback,
        device: this.options.device === 'auto' ? undefined : this.options.device,
      }
    );

    // Get dimension from test embedding if not known
    if (this.dimension === null) {
      const testResult = await this.pipeline(['test'], { pooling: 'mean', normalize: true });
      this.dimension = testResult.dims[testResult.dims.length - 1];
    }
  }

  /**
   * Get embedding dimension.
   */
  getDimension(): number {
    if (this.dimension === null) {
      throw new Error('Model not initialized. Call initialize() first.');
    }
    return this.dimension;
  }

  /**
   * Encode texts into embeddings.
   */
  async encode(texts: string | string[]): Promise<number[][]> {
    if (!this.pipeline) {
      throw new Error('Model not initialized. Call initialize() first.');
    }

    const inputTexts = Array.isArray(texts) ? texts : [texts];
    const result = await this.pipeline(inputTexts, {
      pooling: 'mean',
      normalize: true,
    });

    // Convert to 2D array
    const embeddings: number[][] = [];
    const dim = this.getDimension();

    for (let i = 0; i < inputTexts.length; i++) {
      const start = i * dim;
      const end = start + dim;
      embeddings.push(Array.from(result.data.slice(start, end)));
    }

    return embeddings;
  }

  /**
   * Encode a single query string.
   */
  async encodeQuery(query: string): Promise<number[]> {
    const embeddings = await this.encode([query]);
    return embeddings[0];
  }
}

/**
 * VecStore with built-in embedding support using Transformers.js.
 *
 * This class combines the WASM VecStore with automatic embedding generation.
 */
export class VecStoreWithEmbeddings {
  private store: WasmVecStore;
  private embeddingModel: EmbeddingModel;
  private initialized: boolean = false;

  private constructor(store: WasmVecStore, embeddingModel: EmbeddingModel) {
    this.store = store;
    this.embeddingModel = embeddingModel;
  }

  /**
   * Create a new VecStore with embedding support.
   *
   * @param options - Embedding model options
   * @returns Initialized VecStoreWithEmbeddings instance
   *
   * @example
   * ```typescript
   * const store = await VecStoreWithEmbeddings.create({
   *   model: 'Xenova/all-MiniLM-L6-v2',
   *   progressCallback: (p) => console.log(p.status)
   * });
   * ```
   */
  static async create(options: EmbeddingModelOptions = {}): Promise<VecStoreWithEmbeddings> {
    // Initialize embedding model
    const embeddingModel = new EmbeddingModel(options);
    await embeddingModel.initialize();

    // Dynamic import of vecstore core
    const vecstoreCore = await import('@vecstore/core');
    await vecstoreCore.default(); // Initialize WASM

    // Create store with appropriate dimension
    const store = new vecstoreCore.WasmVecStore(embeddingModel.getDimension());

    const instance = new VecStoreWithEmbeddings(store, embeddingModel);
    instance.initialized = true;

    return instance;
  }

  /**
   * Create from existing JSON data.
   */
  static async fromJson(
    jsonData: string,
    options: EmbeddingModelOptions = {}
  ): Promise<VecStoreWithEmbeddings> {
    const instance = await VecStoreWithEmbeddings.create(options);
    instance.store.import_json(jsonData);
    return instance;
  }

  /**
   * Add texts with automatic embedding generation.
   *
   * @param texts - Array of text contents
   * @param metadatas - Optional array of metadata objects
   * @param ids - Optional array of IDs (auto-generated if not provided)
   * @returns Array of document IDs
   */
  async addTexts(
    texts: string[],
    metadatas?: Record<string, unknown>[],
    ids?: string[]
  ): Promise<string[]> {
    // Generate embeddings
    const embeddings = await this.embeddingModel.encode(texts);

    // Generate IDs if not provided
    const docIds = ids || texts.map((text, i) => this.generateId(text, i));

    // Add to store
    for (let i = 0; i < texts.length; i++) {
      const metadata = {
        text: texts[i],
        ...(metadatas?.[i] || {}),
      };

      this.store.upsert(docIds[i], embeddings[i], metadata);
      this.store.index_text(docIds[i], texts[i]);
    }

    return docIds;
  }

  /**
   * Search for similar documents by text query.
   *
   * @param query - Query text (will be embedded automatically)
   * @param k - Number of results to return
   * @param filter - Optional filter expression
   * @returns Array of scored documents
   */
  async search(query: string, k: number = 4, filter?: string): Promise<ScoredDocument[]> {
    const queryEmbedding = await this.embeddingModel.encodeQuery(query);
    return this.searchByVector(queryEmbedding, k, filter);
  }

  /**
   * Search by pre-computed embedding vector.
   */
  searchByVector(embedding: number[], k: number = 4, filter?: string): ScoredDocument[] {
    const results = this.store.query(embedding, k, filter || null);
    return results.map((r) => this.resultToDocument(r));
  }

  /**
   * Hybrid search combining vector similarity and keyword matching.
   *
   * @param query - Query text
   * @param keywords - Search keywords
   * @param k - Number of results
   * @param alpha - Balance between vector (1.0) and keyword (0.0)
   * @param filter - Optional filter expression
   */
  async hybridSearch(
    query: string,
    keywords: string,
    k: number = 4,
    alpha: number = 0.7,
    filter?: string
  ): Promise<ScoredDocument[]> {
    const queryEmbedding = await this.embeddingModel.encodeQuery(query);
    const results = this.store.hybrid_query(queryEmbedding, keywords, k, alpha, filter || null);
    return results.map((r) => this.resultToDocument(r));
  }

  /**
   * Delete documents by IDs.
   */
  delete(ids: string[]): void {
    for (const id of ids) {
      this.store.remove(id);
    }
  }

  /**
   * Get number of documents in the store.
   */
  count(): number {
    return this.store.len();
  }

  /**
   * Check if store is empty.
   */
  isEmpty(): boolean {
    return this.store.is_empty();
  }

  /**
   * Export store to JSON string.
   */
  exportJson(): string {
    return this.store.export_json();
  }

  /**
   * Get the embedding model.
   */
  getEmbeddingModel(): EmbeddingModel {
    return this.embeddingModel;
  }

  private generateId(text: string, index: number): string {
    // Simple hash-based ID generation
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return `doc_${Math.abs(hash).toString(16)}_${index}`;
  }

  private resultToDocument(result: SearchResult): ScoredDocument {
    const metadata = JSON.parse(result.metadata);
    return {
      pageContent: metadata.text || '',
      metadata,
      score: result.score,
    };
  }
}

// Re-export types
export type { Pipeline, SearchResult, WasmVecStore };
