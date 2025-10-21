//! # vecstore
//!
//! Embeddable vector database with HNSW indexing, metadata-aware filtering, and optional
//! tooling for retrieval-augmented generation (RAG) workflows.
//!
//! VecStore aims to be the "SQLite of vector search": run it in-process, keep data on disk
//! next to your application, and opt into extra features only when you need them.
//!
//! ## Highlights
//!
//! - **HNSW index** with cosine/Euclidean/dot-product support.
//! - **Metadata filtering** via a SQL-inspired expression language.
//! - **Hybrid search** (vector + BM25 keyword scoring) using a pluggable tokenizer.
//! - **Snapshots** and namespace-aware collections built on the same file layout.
//! - **Async wrappers, Python bindings, and optional HTTP/gRPC servers** for integration.
//! - **RAG utilities** (text splitters, rerankers, evaluation helpers) available as
//!   standalone modules.
//!
//! See `docs/ARCHITECTURE.md` for an up-to-date map of implemented modules and current
//! limitations. Experimental subsystems such as the distributed prototype, realtime
//! indexer, GPU backends, and WAL live in the repository but are not activated by default.
//!
//! ## Quick Start
//!
//! ```no_run
//! use vecstore::{VecStore, Query};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Create or open a vector store
//! let mut store = VecStore::open("my_vectors.db")?;
//!
//! // Insert vectors with metadata
//! store.upsert(
//!     "doc1",
//!     vec![0.1, 0.2, 0.3, 0.4],
//!     serde_json::json!({
//!         "title": "First Document",
//!         "category": "tech",
//!         "score": 0.95
//!     }),
//! )?;
//!
//! // Search with filters
//! let query = Query::new(vec![0.15, 0.25, 0.35, 0.45])
//!     .with_limit(10)
//!     .with_filter("category = 'tech' AND score > 0.9");
//!
//! let results = store.query(query)?;
//!
//! for result in results {
//!     println!("{}: {:.4}", result.id, result.distance);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Examples
//!
//! ### Hybrid Search (Vector + Keyword)
//!
//! ```no_run
//! use vecstore::{VecStore, HybridQuery};
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut store = VecStore::open("vectors.db")?;
//!
//! // Index text for keyword search
//! store.index_text("doc1", "machine learning tutorial")?;
//!
//! // Hybrid search: 70% vector similarity + 30% keyword match
//! let query = HybridQuery::new(
//!     vec![0.1, 0.2, 0.3],
//!     "machine learning",
//! )
//! .with_limit(10)
//! .with_alpha(0.7);
//!
//! let results = store.hybrid_query(query)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Product Quantization (Memory Compression)
//!
//! ```no_run
//! use vecstore::{ProductQuantizer, PQConfig};
//!
//! # fn main() -> anyhow::Result<()> {
//! let config = PQConfig {
//!     num_subvectors: 16,
//!     num_centroids: 256,
//!     training_iterations: 20,
//! };
//!
//! let mut pq = ProductQuantizer::new(128, config)?;
//!
//! // Train on representative sample
//! pq.train(&training_vectors)?;
//!
//! // Encode vectors (128 floats -> 16 bytes = 32x compression)
//! let codes = pq.encode(&vector)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Async API
//!
//! ```no_run
//! use vecstore::{AsyncVecStore, Query};
//!
//! #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let store = AsyncVecStore::open("vectors.db").await?;
//!
//! let query = Query::new(vec![0.1, 0.2, 0.3]).with_limit(10);
//! let results = store.query(query).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature Flags
//!
//! - `async` – Enable the Tokio-based async façade (`AsyncVecStore`).
//! - `python` – Build PyO3 bindings.
//! - `embeddings` – Include optional embedding adapters (ONNX/Candle/Ollama, etc.).
//! - `wasm` – Compile the pure-Rust HNSW backend for WebAssembly.
//!
//! Benchmark numbers vary with hardware and configuration; run the Criterion benches in
//! `benches/` if you need reproducible performance data for your workload.
//!
//! ## Architecture
//!
//! vecstore uses:
//! - **HNSW** for approximate nearest neighbor search
//! - **Product Quantization** for memory-efficient vector compression
//! - **BM25** for keyword-based text search
//! - **JSON + bincode** for on-disk persistence
//! - **Rayon** for parallel processing during batch builds
//!
//! ## Use Cases
//!
//! - **RAG applications** (Retrieval-Augmented Generation)
//! - **Semantic search** over documents/images/code
//! - **Recommendation systems**
//! - **Duplicate detection**
//! - **Clustering and classification**

pub mod autotuning;
pub mod cache;
pub mod compression;
pub mod error;
pub mod fuzzy;
pub mod graph_viz;
pub mod import_export;
pub mod metrics;
pub mod mmap;
pub mod query_analyzer;
pub mod schema;
pub mod semantic_cache;
pub mod simd;
pub mod store;
pub mod stream;
pub mod vectors;
pub mod wal;

#[cfg(feature = "async")]
pub mod async_api;

#[cfg(feature = "async")]
pub mod async_ops;

#[cfg(feature = "async")]
pub mod async_stream;

#[cfg(feature = "python")]
pub mod python;

#[cfg(any(
    feature = "embeddings",
    feature = "cloud-embeddings",
    feature = "openai-embeddings",
    feature = "candle-embeddings",
    feature = "ollama"
))]
pub mod embeddings;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "server")]
pub mod server;

pub mod access_control;
pub mod advanced_filter;
pub mod analytics;
pub mod anomaly;
pub mod audit;
pub mod benchmark;
pub mod bulk_migration;
pub mod clustering;
pub mod collection;
pub mod deduplication;
pub mod dim_reduction;
pub mod distributed;
pub mod geospatial;
pub mod gpu;
pub mod graph_rag;
pub mod health;
pub mod ivf_pq;
pub mod langchain;
pub mod lsh;
pub mod metadata_index;
pub mod migration;
pub mod monitoring;
pub mod multi_vector;
pub mod multimodal;
pub mod namespace;
pub mod namespace_manager;
pub mod partitioning;
pub mod profiler;
pub mod protocol;
pub mod quantization;
pub mod query_optimizer;
pub mod rag_utils;
pub mod rate_limit;
pub mod realtime;
pub mod recommender;
pub mod reranking;
pub mod scann;
pub mod splade;
pub mod telemetry;
pub mod text_splitter;
pub mod timeseries;
pub mod tokenizer;
pub mod validation;
pub mod versioning;

#[cfg(feature = "async")]
pub mod kafka_connector;

#[cfg(feature = "async")]
pub mod python_async;

pub use collection::{Collection, CollectionConfig, VecDatabase};
pub use error::{Result, VecStoreError};
pub use graph_viz::{GraphEdge, GraphNode, GraphStatistics, HnswVisualizer};
pub use namespace::{Namespace, NamespaceId, NamespaceQuotas, NamespaceStatus, ResourceUsage};
pub use namespace_manager::{AggregateStats, NamespaceManager, NamespaceStats};
pub use schema::{FieldSchema, FieldType, Schema, ValidationError};
pub use store::{
    make_record, parse_filter, BatchError, BatchOperation, BatchResult, CompactionConfig,
    CompactionResult, Config, Distance, ExplainedNeighbor, FilterExpr, FilterOp, FilterParseError,
    HNSWSearchParams, HybridQuery, Metadata, Neighbor, PQConfig, PQVectorStore, PrefetchQuery,
    ProductQuantizer, Query, QueryEstimate, QueryExplanation, QueryPlan, QueryStage, QueryStep,
    Record, VecStore, VecStoreBuilder,
};
pub use text_splitter::{
    RecursiveCharacterTextSplitter, TextChunk, TextSplitter, TokenTextSplitter,
};

// Export fuzzy search types
pub use fuzzy::{
    damerau_levenshtein_distance, levenshtein_distance, similarity_score, suggest_corrections,
    BKTree, FuzzyMatcher,
};

// Export auto-tuning types
pub use autotuning::{AutoTuner, HnswParams, PerformanceConstraints, TuningGoal};

// Export compression types
pub use compression::{
    decode_rle, decode_varint, encode_rle, encode_varint, CompressedNeighborList,
    CompressionConfig, CompressionLevel, CompressionMethod, CompressionStats,
};

// Export real-time indexing types
pub use realtime::{
    BufferEntry, CompactionStats, RealtimeConfig, RealtimeIndex, RealtimeMetrics, Snapshot,
    UpdateStrategy, WorkerConfig, WriteBuffer,
};

// Export GPU acceleration types
pub use gpu::{GpuBackend, GpuBenchmark, GpuConfig, GpuDeviceInfo, GpuExecutor, GpuOps};

// Export distributed indexing types
pub use distributed::{
    ConsistencyLevel, ConsistentHashRing, DistributedConfig, DistributedStats, DistributedStore,
    NodeInfo, NodeStatus, ReplicationStrategy, ShardInfo, ShardingStrategy,
};

// Export quantization types
pub use quantization::{
    BinaryQuantizer, QuantizationBenchmark, ScalarQuantizer4, ScalarQuantizer8,
};

// Export advanced indexing types
pub use ivf_pq::{IVFPQConfig, IVFPQIndex, IVFPQStats};
pub use lsh::{LSHConfig, LSHIndex, LSHStats};
pub use scann::{ScaNNConfig, ScaNNIndex, ScaNNStats};

// Export time-series types
pub use timeseries::{
    DecayFunction, TemporalGroup, TimeQuery, TimeSeriesEntry, TimeSeriesIndex, TimeSeriesResult,
    TimeSeriesStats, WindowResult,
};

// Export multi-modal types
pub use multimodal::{
    Modality, MultiModalEntry, MultiModalFusion, MultiModalIndex, MultiModalQuery,
    MultiModalResult, MultiModalStats,
};

// Export migration types
pub use migration::{MigrationRecord, MigrationStats, Migrator, SourceDatabase};

// Export Graph RAG types
pub use graph_rag::{Entity, GraphQuery, GraphRAG, GraphResult, GraphStats, Relation};

// Export protocol adapter types
pub use protocol::{Protocol, ProtocolAdapter, UniversalRequest, UniversalResponse, VectorData};

// Export LangChain/LlamaIndex integration types
pub use langchain::{
    Document, LangChainVectorStore, LlamaIndexVectorStore, Node, RetrieverConfig, ScoredDocument,
    VectorStoreRetriever,
};

// Export benchmarking types
pub use benchmark::{
    BenchmarkConfig, BenchmarkResults, Benchmarker, ConcurrentResults, FilterResults,
    IndexingResults, InsertResults, LatencyStats, MemoryResults, QuantizationResults, QueryResults,
};

// Export health check types
pub use health::{
    print_health_report, Alert, AlertCategory, AlertSeverity, DatabaseHealth, HealthCheckConfig,
    HealthChecker, HealthReport, HealthStatus, IndexHealth, PerformanceHealth, ResourceHealth,
};

// Export metadata indexing types
pub use metadata_index::{
    BTreeIndex, HashIndex, IndexConfig, IndexStats, IndexType, IndexedValue, InvertedIndex,
    MetadataIndex, MetadataIndexManager,
};

// Export clustering types
pub use clustering::{
    ClusteringConfig, ClusteringResult, DBSCANClustering, DBSCANConfig, HierarchicalClustering,
    HierarchicalConfig, KMeansClustering, LinkageMethod,
};

// Export bulk migration types
pub use bulk_migration::{
    BulkMigrationStats, ChromaDBMigration, FormatConverter, MigrationConfig, PineconeMigration,
    QdrantMigration,
};

// Export partitioning types
pub use partitioning::{PartitionConfig, PartitionInfo, PartitionStats, PartitionedStore};

// Export anomaly detection types
pub use anomaly::{
    AnomalyDetector, AnomalyEnsemble, AnomalyResult, IsolationForest, LocalOutlierFactor,
    ZScoreDetector,
};

// Export dimensionality reduction types
pub use dim_reduction::{ReductionStats, PCA};

// Export recommender system types
pub use recommender::{
    CollaborativeRecommender, ContentBasedRecommender, HybridRecommender, Recommendation,
    UserPreference,
};

// Export versioning types
pub use versioning::{
    Snapshot as VersionSnapshot, Version, VersionDiff, VersionHistory, VersionedStore,
    VersioningStats,
};

// Export query optimizer types
pub use query_optimizer::{
    CostBreakdown, ExecutionPlan, HintCategory, Impact, OptimizationHint, QueryAnalysis,
    QueryComparison, QueryComplexity, QueryOptimizer, StoreOptimizationSummary,
};

// Export deduplication types
pub use deduplication::{
    BatchDeduplicator, DeduplicationConfig, DeduplicationStats, DeduplicationStrategy,
    Deduplicator, DuplicateGroup,
};

// Export validation types
pub use validation::{
    BatchStatistics, QualityMetrics, ValidationConfig, ValidationError as VectorValidationError,
    ValidationResult, ValidationStrictness, ValidationWarning, VectorValidator,
};

// Export analytics types
pub use analytics::{
    AnalyticsConfig, AnalyticsReport, ClusterTendency, DimensionStats, DistributionStats,
    OutlierAnalysis, SimilarityStats, VectorAnalytics,
};

// Export monitoring types
pub use monitoring::{
    Alert as MonitorAlert, AlertCategory as MonitorAlertCategory, AlertCondition, AlertPresets,
    AlertRule, AlertSeverity as MonitorAlertSeverity, MetricHistory, MetricPoint, MetricType,
    Monitor, MonitoringConfig, MonitoringReport, MonitoringStats,
};

// Export rate limiting types
pub use rate_limit::{
    MultiTierRateLimiter, RateLimitAlgorithm, RateLimitConfig, RateLimitResult, RateLimitScope,
    RateLimiter,
};

// Export audit logging types
pub use audit::{
    AuditBackend, AuditConfig, AuditEntry, AuditEventType, AuditLogger, AuditMetadata,
    AuditOutcome, AuditSeverity, FileBackend, MemoryBackend, StdoutBackend,
};

// Export access control types
pub use access_control::{
    AccessContext, AccessControl, Condition, Effect, Operator, Permission, Policy, Resource, Role,
    User,
};

// Export SPLADE sparse vector types
pub use splade::{SparseIndex, SparseIndexStats, SparseVector, SpladeConfig, SpladeEncoder};

// Export multi-vector document types
pub use multi_vector::{AggregationMethod, MultiVectorDoc, MultiVectorIndex, MultiVectorStats};

// Export geospatial types
pub use geospatial::{
    BoundingBox, GeoDocument, GeoIndex, GeoIndexStats, GeoPoint, GeoSearchResult,
};

// Export advanced filter types
pub use advanced_filter::{parse_advanced_filter, AdvancedFilter, FilterBuilder};

// Export profiler types
pub use profiler::{ProfileStage, ProfileSummary, ProfilerConfig, QueryProfile, QueryProfiler};

#[cfg(feature = "async")]
// Export Kafka connector types
pub use kafka_connector::{
    ConsumerStats, KafkaConfig, KafkaConsumer, KafkaProducer, Operation, PipelineStats,
    ProducerStats, StreamingPipeline, VectorMessage,
};

#[cfg(feature = "async")]
// Export async Python API types
pub use python_async::{AsyncPyVecStore, AsyncSearchResult};

// Export vectors module types
pub use vectors::{
    bm25_score, bm25_score_simple, hybrid_search_score, normalize_scores, normalize_scores_zscore,
    BM25Config, BM25Stats, FusionStrategy, HybridQuery as HybridQueryV2, HybridSearchConfig,
    KMeans, Vector, VectorOps,
};

// Export SIMD-accelerated distance functions
pub use simd::{
    cosine_similarity_simd, dot_product_simd, euclidean_distance_simd, hamming_distance_simd,
    jaccard_distance_simd, jaccard_similarity_simd, magnitude_simd, manhattan_distance_simd,
};

#[cfg(feature = "async")]
pub use async_api::{AsyncCollection, AsyncVecDatabase, AsyncVecStore};

#[cfg(feature = "embeddings")]
pub use embeddings::{Embedder, EmbeddingCollection, EmbeddingStore, SimpleEmbedder, TextEmbedder};

#[cfg(feature = "cloud-embeddings")]
pub use embeddings::{
    AzureEmbedding, AzureModel, CohereEmbedding, CohereModel, GoogleEmbedding, GoogleModel,
    HuggingFaceEmbedding, JinaEmbedding, JinaModel, MistralEmbedding, MistralModel,
    VoyageEmbedding, VoyageModel,
};

#[cfg(feature = "ollama")]
pub use embeddings::{OllamaEmbedding, OllamaModel};

#[cfg(feature = "candle-embeddings")]
pub use embeddings::{CandleEmbedder, CandleModel};

#[cfg(feature = "wasm")]
pub use wasm::{WasmSearchResult, WasmVecStore};

/// Initialize tracing subscriber for logging
pub fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    fmt().with_env_filter(filter).with_target(false).init();
}
