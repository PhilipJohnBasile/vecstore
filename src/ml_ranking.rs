//! ONNX/ML Models in Ranking
//!
//! Deploy machine learning models directly in the ranking pipeline,
//! similar to Vespa's ONNX integration.
//!
//! # Features
//!
//! - **ONNX Runtime**: Run ONNX models for reranking
//! - **Feature Extraction**: Compute features for ML models
//! - **Multi-Stage Ranking**: First-pass retrieval + ML rerank
//! - **Ensemble Methods**: Combine multiple models
//! - **Online Learning**: Update models with feedback
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::ml_ranking::{MLRanker, RankingModel, RankingConfig};
//!
//! let config = RankingConfig::new()
//!     .with_model(RankingModel::onnx("reranker.onnx")?)
//!     .with_top_k(100)  // Rerank top 100
//!     .with_final_k(10);  // Return top 10
//!
//! let ranker = MLRanker::new(config)?;
//!
//! // First-pass retrieval
//! let candidates = store.search(&query, 100)?;
//!
//! // ML reranking
//! let final_results = ranker.rerank(&query, candidates)?;
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;

#[cfg(feature = "embeddings")]
use std::path::Path;
#[cfg(feature = "embeddings")]
use std::sync::{Arc, Mutex, RwLock};
#[cfg(feature = "embeddings")]
use crate::error::VecStoreError;
#[cfg(feature = "embeddings")]
use ndarray::Array2;
#[cfg(feature = "embeddings")]
use ort::session::{builder::GraphOptimizationLevel, Session};

/// Default batch size for ONNX inference
const DEFAULT_BATCH_SIZE: usize = 32;

/// Maximum number of cached model sessions
const MAX_CACHED_SESSIONS: usize = 10;

/// ONNX Model Session Cache for efficient model reuse
#[cfg(feature = "embeddings")]
struct OnnxSessionCache {
    sessions: RwLock<HashMap<String, Arc<Mutex<Session>>>>,
}

#[cfg(feature = "embeddings")]
impl OnnxSessionCache {
    /// Create a new session cache
    fn new() -> Result<Self> {
        Ok(Self {
            sessions: RwLock::new(HashMap::new()),
        })
    }

    /// Get or create a session for a model path
    fn get_or_create_session(&self, model_path: &str, num_threads: usize) -> Result<Arc<Mutex<Session>>> {
        // Check if session already exists
        {
            let sessions = self.sessions.read()
                .map_err(|e| VecStoreError::LockError(e.to_string()))?;
            if let Some(session) = sessions.get(model_path) {
                return Ok(Arc::clone(session));
            }
        }

        // Create new session using ort 2.0 API
        let path = Path::new(model_path);
        if !path.exists() {
            return Err(VecStoreError::NotFound(format!("Model file not found: {}", model_path)));
        }

        let session = Session::builder()
            .map_err(|e| VecStoreError::Internal(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| VecStoreError::Internal(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(num_threads)
            .map_err(|e| VecStoreError::Internal(format!("Failed to set thread count: {}", e)))?
            .commit_from_file(path)
            .map_err(|e| VecStoreError::Internal(format!("Failed to load ONNX model: {}", e)))?;

        let session = Arc::new(Mutex::new(session));

        // Cache the session
        {
            let mut sessions = self.sessions.write()
                .map_err(|e| VecStoreError::LockError(e.to_string()))?;

            // Evict oldest if cache is full
            if sessions.len() >= MAX_CACHED_SESSIONS {
                if let Some(key) = sessions.keys().next().cloned() {
                    sessions.remove(&key);
                }
            }

            sessions.insert(model_path.to_string(), Arc::clone(&session));
        }

        Ok(session)
    }

    /// Warmup a model by running a dummy inference
    fn warmup_model(&self, model_path: &str, feature_dim: usize, num_threads: usize) -> Result<()> {
        let session = self.get_or_create_session(model_path, num_threads)?;

        // Create dummy input matching expected feature dimension
        let dummy_features = vec![0.0f32; feature_dim];
        let input_array = Array2::from_shape_vec((1, feature_dim), dummy_features)
            .map_err(|e| VecStoreError::Internal(format!("Failed to create dummy input: {}", e)))?;

        // Create Tensor object for ort 2.0
        let input_tensor = ort::value::Tensor::from_array(input_array)
            .map_err(|e| VecStoreError::Internal(format!("Failed to create input tensor: {}", e)))?;

        // Run warmup inference using ort 2.0 inputs! macro
        let mut session_guard = session.lock()
            .map_err(|e| VecStoreError::LockError(e.to_string()))?;
        let _ = session_guard.run(ort::inputs![input_tensor])
            .map_err(|e| VecStoreError::Internal(format!("Warmup inference failed: {}", e)))?;

        Ok(())
    }

    /// Clear all cached sessions
    fn clear(&self) -> Result<()> {
        let mut sessions = self.sessions.write()
            .map_err(|e| VecStoreError::LockError(e.to_string()))?;
        sessions.clear();
        Ok(())
    }
}

/// Global session cache (lazy initialized)
#[cfg(feature = "embeddings")]
fn get_session_cache() -> Result<&'static OnnxSessionCache> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<OnnxSessionCache> = OnceLock::new();
    if let Some(cache) = CACHE.get() {
        return Ok(cache);
    }
    let cache = OnnxSessionCache::new()?;
    // Ignore the result if another thread initialized first
    let _ = CACHE.set(cache);
    CACHE.get().ok_or_else(|| VecStoreError::Internal("Failed to initialize session cache".to_string()))
}

/// Ranking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingConfig {
    /// Number of candidates to rerank
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Final number of results
    #[serde(default = "default_final_k")]
    pub final_k: usize,
    /// Enable feature caching
    #[serde(default = "default_true")]
    pub cache_features: bool,
    /// Score normalization method
    #[serde(default)]
    pub normalization: ScoreNormalization,
    /// Ensemble weights (if multiple models)
    #[serde(default)]
    pub ensemble_weights: Vec<f32>,
    /// Batch size for ONNX inference
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Number of threads for ONNX inference
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
    /// Whether to warmup models on initialization
    #[serde(default = "default_true")]
    pub warmup_models: bool,
}

fn default_top_k() -> usize { 100 }
fn default_final_k() -> usize { 10 }
fn default_true() -> bool { true }
fn default_batch_size() -> usize { DEFAULT_BATCH_SIZE }
fn default_num_threads() -> usize { 4 }

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            top_k: 100,
            final_k: 10,
            cache_features: true,
            normalization: ScoreNormalization::None,
            ensemble_weights: Vec::new(),
            batch_size: DEFAULT_BATCH_SIZE,
            num_threads: 4,
            warmup_models: true,
        }
    }
}

impl RankingConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set top-k candidates
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set final-k results
    pub fn with_final_k(mut self, k: usize) -> Self {
        self.final_k = k;
        self
    }

    /// Set normalization method
    pub fn with_normalization(mut self, method: ScoreNormalization) -> Self {
        self.normalization = method;
        self
    }

    /// Set batch size for ONNX inference
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Set number of threads for ONNX inference
    pub fn with_num_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads.max(1);
        self
    }

    /// Enable or disable model warmup
    pub fn with_warmup(mut self, enabled: bool) -> Self {
        self.warmup_models = enabled;
        self
    }
}

/// Score normalization methods
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ScoreNormalization {
    #[default]
    None,
    MinMax,
    ZScore,
    Softmax,
}

/// Model type for ONNX inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum OnnxModelType {
    /// Generic ranking model (outputs a single relevance score)
    #[default]
    Ranker,
    /// XGBoost model exported to ONNX
    XGBoost,
    /// LightGBM model exported to ONNX
    LightGBM,
    /// Gradient boosting model (generic)
    GradientBoosting,
    /// Neural network ranker
    NeuralNetwork,
}


/// ONNX model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxModelConfig {
    /// Path to the ONNX model file
    pub path: String,
    /// Model type (affects output interpretation)
    #[serde(default)]
    pub model_type: OnnxModelType,
    /// Expected input feature dimension (0 = auto-detect)
    #[serde(default)]
    pub feature_dim: usize,
    /// Input tensor name (for models with named inputs)
    #[serde(default = "default_input_name")]
    pub input_name: String,
    /// Output tensor name (for models with named outputs)
    #[serde(default = "default_output_name")]
    pub output_name: String,
    /// Whether to apply sigmoid to output
    #[serde(default)]
    pub apply_sigmoid: bool,
    /// Whether to apply softmax to output (for classification models)
    #[serde(default)]
    pub apply_softmax: bool,
    /// Which output index to use (for multi-output models)
    #[serde(default)]
    pub output_index: usize,
}

fn default_input_name() -> String { "input".to_string() }
fn default_output_name() -> String { "output".to_string() }

impl OnnxModelConfig {
    /// Create a new ONNX model config
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            model_type: OnnxModelType::Ranker,
            feature_dim: 0,
            input_name: "input".to_string(),
            output_name: "output".to_string(),
            apply_sigmoid: false,
            apply_softmax: false,
            output_index: 0,
        }
    }

    /// Create an XGBoost ONNX config
    pub fn xgboost(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            model_type: OnnxModelType::XGBoost,
            feature_dim: 0,
            input_name: "input".to_string(),
            output_name: "output".to_string(),
            apply_sigmoid: true, // XGBoost binary classifiers typically need sigmoid
            apply_softmax: false,
            output_index: 0,
        }
    }

    /// Create a LightGBM ONNX config
    pub fn lightgbm(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            model_type: OnnxModelType::LightGBM,
            feature_dim: 0,
            input_name: "input".to_string(),
            output_name: "output".to_string(),
            apply_sigmoid: true, // LightGBM binary classifiers typically need sigmoid
            apply_softmax: false,
            output_index: 0,
        }
    }

    /// Set the expected feature dimension
    pub fn with_feature_dim(mut self, dim: usize) -> Self {
        self.feature_dim = dim;
        self
    }

    /// Set the input tensor name
    pub fn with_input_name(mut self, name: impl Into<String>) -> Self {
        self.input_name = name.into();
        self
    }

    /// Set the output tensor name
    pub fn with_output_name(mut self, name: impl Into<String>) -> Self {
        self.output_name = name.into();
        self
    }

    /// Enable sigmoid activation on output
    pub fn with_sigmoid(mut self) -> Self {
        self.apply_sigmoid = true;
        self
    }

    /// Enable softmax activation on output
    pub fn with_softmax(mut self) -> Self {
        self.apply_softmax = true;
        self
    }

    /// Set which output index to use
    pub fn with_output_index(mut self, index: usize) -> Self {
        self.output_index = index;
        self
    }
}

/// Ranking model types
#[derive(Debug, Clone)]
pub enum RankingModel {
    /// ONNX model with full configuration
    ONNX(OnnxModelConfig),
    /// XGBoost model (exported to ONNX format)
    XGBoost {
        path: String,
        /// Number of features expected
        num_features: usize,
    },
    /// LightGBM model (exported to ONNX format)
    LightGBM {
        path: String,
        /// Number of features expected
        num_features: usize,
    },
    /// Linear model
    Linear {
        weights: Vec<f32>,
        bias: f32,
    },
    /// Custom scoring function
    Custom {
        name: String,
    },
}

impl RankingModel {
    /// Create an ONNX model with default configuration
    pub fn onnx(path: impl Into<String>) -> Self {
        RankingModel::ONNX(OnnxModelConfig::new(path))
    }

    /// Create an ONNX model with full configuration
    pub fn onnx_with_config(config: OnnxModelConfig) -> Self {
        RankingModel::ONNX(config)
    }

    /// Create an XGBoost model (expects ONNX-exported XGBoost)
    pub fn xgboost(path: impl Into<String>, num_features: usize) -> Self {
        RankingModel::XGBoost {
            path: path.into(),
            num_features,
        }
    }

    /// Create a LightGBM model (expects ONNX-exported LightGBM)
    pub fn lightgbm(path: impl Into<String>, num_features: usize) -> Self {
        RankingModel::LightGBM {
            path: path.into(),
            num_features,
        }
    }

    /// Create a linear model
    pub fn linear(weights: Vec<f32>, bias: f32) -> Self {
        RankingModel::Linear { weights, bias }
    }

    /// Get the model path if this is an ONNX-based model
    pub fn model_path(&self) -> Option<&str> {
        match self {
            RankingModel::ONNX(config) => Some(&config.path),
            RankingModel::XGBoost { path, .. } => Some(path),
            RankingModel::LightGBM { path, .. } => Some(path),
            _ => None,
        }
    }

    /// Get the expected feature dimension for ONNX models
    pub fn feature_dim(&self) -> Option<usize> {
        match self {
            RankingModel::ONNX(config) if config.feature_dim > 0 => Some(config.feature_dim),
            RankingModel::XGBoost { num_features, .. } => Some(*num_features),
            RankingModel::LightGBM { num_features, .. } => Some(*num_features),
            RankingModel::Linear { weights, .. } => Some(weights.len()),
            _ => None,
        }
    }
}

/// Ranking features
#[derive(Debug, Clone, Serialize)]
pub struct RankingFeatures {
    /// Vector similarity score
    pub vector_score: f32,
    /// BM25 score (if available)
    pub bm25_score: Option<f32>,
    /// Query-document features
    pub qd_features: Vec<f32>,
    /// Document features
    pub doc_features: Vec<f32>,
    /// Query features
    pub query_features: Vec<f32>,
}

impl RankingFeatures {
    /// Create new features
    pub fn new(vector_score: f32) -> Self {
        Self {
            vector_score,
            bm25_score: None,
            qd_features: Vec::new(),
            doc_features: Vec::new(),
            query_features: Vec::new(),
        }
    }

    /// Set BM25 score
    pub fn with_bm25(mut self, score: f32) -> Self {
        self.bm25_score = Some(score);
        self
    }

    /// Add query-document features
    pub fn with_qd_features(mut self, features: Vec<f32>) -> Self {
        self.qd_features = features;
        self
    }

    /// Convert to feature vector
    pub fn to_vector(&self) -> Vec<f32> {
        let mut v = vec![self.vector_score];
        if let Some(bm25) = self.bm25_score {
            v.push(bm25);
        }
        v.extend(&self.qd_features);
        v.extend(&self.doc_features);
        v.extend(&self.query_features);
        v
    }
}

/// Candidate for reranking
#[derive(Debug, Clone)]
pub struct RankingCandidate {
    /// Document ID
    pub id: String,
    /// Original score
    pub original_score: f32,
    /// Document vector
    pub vector: Vec<f32>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
    /// Computed features
    pub features: Option<RankingFeatures>,
}

/// Reranked result
#[derive(Debug, Clone, Serialize)]
pub struct RankedResult {
    /// Document ID
    pub id: String,
    /// Final score after reranking
    pub score: f32,
    /// Original retrieval score
    pub original_score: f32,
    /// Feature contributions (for explainability)
    pub feature_contributions: Option<HashMap<String, f32>>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
}

/// ML-based ranker with full ONNX support
pub struct MLRanker {
    config: RankingConfig,
    models: Vec<RankingModel>,
    feature_cache: HashMap<String, RankingFeatures>,
    /// Whether models have been warmed up
    models_warmed_up: bool,
}

impl MLRanker {
    /// Create a new ML ranker
    pub fn new(config: RankingConfig) -> Result<Self> {
        Ok(Self {
            config,
            models: Vec::new(),
            feature_cache: HashMap::new(),
            models_warmed_up: false,
        })
    }

    /// Add a ranking model
    pub fn add_model(&mut self, model: RankingModel) {
        self.models.push(model);
        self.models_warmed_up = false; // Need to re-warmup
    }

    /// Warmup all ONNX-based models
    ///
    /// This pre-loads the models and runs a dummy inference to ensure
    /// the models are ready for fast inference.
    #[cfg(feature = "embeddings")]
    pub fn warmup(&mut self) -> Result<()> {
        if self.models_warmed_up {
            return Ok(());
        }

        let cache = get_session_cache()?;
        let default_feature_dim = 10; // Default feature dimension for warmup

        for model in &self.models {
            if let Some(path) = model.model_path() {
                let feature_dim = model.feature_dim().unwrap_or(default_feature_dim);
                cache.warmup_model(path, feature_dim, self.config.num_threads)?;
            }
        }

        self.models_warmed_up = true;
        Ok(())
    }

    /// Warmup stub for non-embeddings builds
    #[cfg(not(feature = "embeddings"))]
    pub fn warmup(&mut self) -> Result<()> {
        self.models_warmed_up = true;
        Ok(())
    }

    /// Rerank candidates using all configured models
    pub fn rerank(
        &mut self,
        query: &[f32],
        candidates: Vec<RankingCandidate>,
    ) -> Result<Vec<RankedResult>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Warmup models if needed
        if self.config.warmup_models && !self.models_warmed_up {
            self.warmup()?;
        }

        // Compute features for each candidate
        let candidates_with_features: Vec<(RankingCandidate, RankingFeatures)> = candidates
            .into_iter()
            .map(|c| {
                let features = c.features.clone().unwrap_or_else(|| {
                    self.compute_features(query, &c)
                });
                (c, features)
            })
            .collect();

        // Score candidates with models using batched inference
        let mut results = self.score_candidates_batched(&candidates_with_features)?;

        // Normalize scores if configured
        self.normalize_scores(&mut results);

        // Sort by final score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to final_k
        results.truncate(self.config.final_k);

        Ok(results)
    }

    /// Score candidates with batched inference for ONNX models
    fn score_candidates_batched(
        &self,
        candidates: &[(RankingCandidate, RankingFeatures)],
    ) -> Result<Vec<RankedResult>> {
        if self.models.is_empty() {
            // No models - just use original scores
            return Ok(candidates.iter().map(|(c, features)| {
                RankedResult {
                    id: c.id.clone(),
                    score: c.original_score,
                    original_score: c.original_score,
                    feature_contributions: Some(self.get_feature_contributions(features)),
                    metadata: c.metadata.clone(),
                }
            }).collect());
        }

        // Collect all feature vectors
        let feature_vectors: Vec<Vec<f32>> = candidates.iter()
            .map(|(_, features)| features.to_vector())
            .collect();

        // Get scores from each model
        let mut all_model_scores: Vec<Vec<f32>> = Vec::with_capacity(self.models.len());

        for model in &self.models {
            let scores = self.score_batch_with_model(model, &feature_vectors)?;
            all_model_scores.push(scores);
        }

        // Combine scores for each candidate
        let results: Vec<RankedResult> = candidates.iter()
            .enumerate()
            .map(|(idx, (c, features))| {
                let model_scores: Vec<f32> = all_model_scores.iter()
                    .map(|scores| scores[idx])
                    .collect();

                let final_score = if self.config.ensemble_weights.len() == model_scores.len() {
                    // Weighted ensemble
                    model_scores.iter()
                        .zip(&self.config.ensemble_weights)
                        .map(|(s, w)| s * w)
                        .sum()
                } else {
                    // Simple average
                    model_scores.iter().sum::<f32>() / model_scores.len() as f32
                };

                RankedResult {
                    id: c.id.clone(),
                    score: final_score,
                    original_score: c.original_score,
                    feature_contributions: Some(self.get_feature_contributions(features)),
                    metadata: c.metadata.clone(),
                }
            })
            .collect();

        Ok(results)
    }

    /// Score a batch of feature vectors with a model
    fn score_batch_with_model(
        &self,
        model: &RankingModel,
        feature_vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        match model {
            RankingModel::Linear { weights, bias } => {
                Ok(feature_vectors.iter().map(|fv| {
                    let score: f32 = fv.iter()
                        .zip(weights.iter())
                        .map(|(f, w)| f * w)
                        .sum::<f32>() + bias;
                    self.sigmoid(score)
                }).collect())
            }
            RankingModel::ONNX(config) => {
                self.score_batch_onnx(&config.path, feature_vectors, config.apply_sigmoid, config.apply_softmax, config.output_index)
            }
            RankingModel::XGBoost { path, .. } => {
                // XGBoost exported to ONNX - apply sigmoid for binary classification
                self.score_batch_onnx(path, feature_vectors, true, false, 0)
            }
            RankingModel::LightGBM { path, .. } => {
                // LightGBM exported to ONNX - apply sigmoid for binary classification
                self.score_batch_onnx(path, feature_vectors, true, false, 0)
            }
            RankingModel::Custom { .. } => {
                // Custom models just return original scores (first feature)
                Ok(feature_vectors.iter().map(|fv| fv.first().copied().unwrap_or(0.0)).collect())
            }
        }
    }

    /// Run batched ONNX inference
    #[cfg(feature = "embeddings")]
    fn score_batch_onnx(
        &self,
        model_path: &str,
        feature_vectors: &[Vec<f32>],
        apply_sigmoid: bool,
        apply_softmax: bool,
        output_index: usize,
    ) -> Result<Vec<f32>> {
        if feature_vectors.is_empty() {
            return Ok(Vec::new());
        }

        let cache = get_session_cache()?;
        let session = cache.get_or_create_session(model_path, self.config.num_threads)?;

        let batch_size = self.config.batch_size;
        let mut all_scores = Vec::with_capacity(feature_vectors.len());

        // Process in batches
        for chunk in feature_vectors.chunks(batch_size) {
            let mut session_guard = session.lock()
                .map_err(|e| VecStoreError::LockError(e.to_string()))?;
            let batch_scores = self.run_onnx_batch(&mut session_guard, chunk, apply_sigmoid, apply_softmax, output_index)?;
            all_scores.extend(batch_scores);
        }

        Ok(all_scores)
    }

    /// Run a single batch through ONNX
    #[cfg(feature = "embeddings")]
    fn run_onnx_batch(
        &self,
        session: &mut Session,
        batch: &[Vec<f32>],
        apply_sigmoid: bool,
        apply_softmax: bool,
        output_index: usize,
    ) -> Result<Vec<f32>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = batch.len();
        let feature_dim = batch[0].len();

        // Flatten features into a single vector
        let flat_features: Vec<f32> = batch.iter()
            .flat_map(|v| v.iter().copied())
            .collect();

        // Create input array
        let input_array = Array2::from_shape_vec((batch_size, feature_dim), flat_features)
            .map_err(|e| VecStoreError::Internal(format!("Failed to create input array: {}", e)))?;

        // Create Tensor object for ort 2.0
        let input_tensor = ort::value::Tensor::from_array(input_array)
            .map_err(|e| VecStoreError::Internal(format!("Failed to create input tensor: {}", e)))?;

        // Run inference using ort 2.0 inputs! macro
        let outputs = session.run(ort::inputs![input_tensor])
            .map_err(|e| VecStoreError::Internal(format!("ONNX inference failed: {}", e)))?;

        // Extract output tensor using ort 2.0 API
        let output_tensor = &outputs[0];

        let output_view = output_tensor.try_extract_array::<f32>()
            .map_err(|e| VecStoreError::Internal(format!("Failed to extract output: {}", e)))?;
        let output_array = output_view.to_owned();

        // Parse output based on shape
        let output_shape = output_array.shape();
        let scores: Vec<f32> = if output_shape.len() == 1 {
            // Shape: [batch_size] - one score per sample
            output_array.iter().copied().collect()
        } else if output_shape.len() == 2 {
            if output_shape[1] == 1 {
                // Shape: [batch_size, 1] - regression output
                output_array.iter().copied().collect()
            } else {
                // Shape: [batch_size, num_classes] - classification output
                // Extract the specified output index (usually positive class probability)
                let num_classes = output_shape[1];
                let actual_index = output_index.min(num_classes - 1);
                (0..batch_size)
                    .map(|i| output_array[[i, actual_index]])
                    .collect()
            }
        } else {
            // Unexpected shape - flatten and take first elements
            output_array.iter().take(batch_size).copied().collect()
        };

        // Apply activations
        let scores = if apply_softmax && output_shape.len() == 2 && output_shape[1] > 1 {
            // Apply softmax across classes, then extract target class
            let num_classes = output_shape[1];
            let actual_index = output_index.min(num_classes - 1);
            (0..batch_size).map(|i| {
                let row: Vec<f32> = (0..num_classes).map(|j| output_array[[i, j]]).collect();
                let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
                (row[actual_index] - max_val).exp() / exp_sum
            }).collect()
        } else if apply_sigmoid {
            scores.iter().map(|&x| self.sigmoid(x)).collect()
        } else {
            scores
        };

        Ok(scores)
    }

    /// Stub for non-embeddings builds
    #[cfg(not(feature = "embeddings"))]
    fn score_batch_onnx(
        &self,
        _model_path: &str,
        feature_vectors: &[Vec<f32>],
        _apply_sigmoid: bool,
        _apply_softmax: bool,
        _output_index: usize,
    ) -> Result<Vec<f32>> {
        // Without embeddings feature, just return the first feature (vector_score)
        Ok(feature_vectors.iter()
            .map(|fv| fv.first().copied().unwrap_or(0.0))
            .collect())
    }

    /// Compute ranking features for a query-candidate pair
    fn compute_features(&self, query: &[f32], candidate: &RankingCandidate) -> RankingFeatures {
        let vector_score = candidate.original_score;

        // Compute additional features
        let mut qd_features = Vec::new();

        if !query.is_empty() && !candidate.vector.is_empty() {
            // Dot product
            let dot: f32 = query.iter()
                .zip(candidate.vector.iter())
                .map(|(q, d)| q * d)
                .sum();
            qd_features.push(dot);

            // Query norm
            let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            qd_features.push(query_norm);

            // Doc norm
            let doc_norm: f32 = candidate.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            qd_features.push(doc_norm);

            // Cosine similarity (normalized dot product)
            let cosine = if query_norm > 0.0 && doc_norm > 0.0 {
                dot / (query_norm * doc_norm)
            } else {
                0.0
            };
            qd_features.push(cosine);

            // Max element-wise product
            let max_sim: f32 = query.iter()
                .zip(candidate.vector.iter())
                .map(|(q, d)| q * d)
                .fold(f32::NEG_INFINITY, f32::max);
            qd_features.push(max_sim);

            // Min element-wise product
            let min_sim: f32 = query.iter()
                .zip(candidate.vector.iter())
                .map(|(q, d)| q * d)
                .fold(f32::INFINITY, f32::min);
            qd_features.push(min_sim);

            // Mean element-wise product
            let mean_sim: f32 = query.iter()
                .zip(candidate.vector.iter())
                .map(|(q, d)| q * d)
                .sum::<f32>() / query.len().max(1) as f32;
            qd_features.push(mean_sim);

            // Std of element-wise products
            let products: Vec<f32> = query.iter()
                .zip(candidate.vector.iter())
                .map(|(q, d)| q * d)
                .collect();
            let variance: f32 = products.iter()
                .map(|&p| (p - mean_sim).powi(2))
                .sum::<f32>() / products.len().max(1) as f32;
            qd_features.push(variance.sqrt());

            // L1 distance (Manhattan)
            let l1_dist: f32 = query.iter()
                .zip(candidate.vector.iter())
                .map(|(q, d)| (q - d).abs())
                .sum();
            qd_features.push(l1_dist);

            // L2 distance (Euclidean)
            let l2_dist: f32 = query.iter()
                .zip(candidate.vector.iter())
                .map(|(q, d)| (q - d).powi(2))
                .sum::<f32>()
                .sqrt();
            qd_features.push(l2_dist);
        }

        RankingFeatures::new(vector_score)
            .with_qd_features(qd_features)
    }

    /// Get feature contributions for explainability
    fn get_feature_contributions(&self, features: &RankingFeatures) -> HashMap<String, f32> {
        let mut contributions = HashMap::new();
        contributions.insert("vector_score".to_string(), features.vector_score);

        if let Some(bm25) = features.bm25_score {
            contributions.insert("bm25_score".to_string(), bm25);
        }

        let qd_feature_names = [
            "dot_product", "query_norm", "doc_norm", "cosine_similarity",
            "max_product", "min_product", "mean_product", "std_product",
            "l1_distance", "l2_distance"
        ];

        for (i, &f) in features.qd_features.iter().enumerate() {
            let name = qd_feature_names.get(i)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("qd_feature_{}", i));
            contributions.insert(name, f);
        }

        for (i, &f) in features.doc_features.iter().enumerate() {
            contributions.insert(format!("doc_feature_{}", i), f);
        }

        for (i, &f) in features.query_features.iter().enumerate() {
            contributions.insert(format!("query_feature_{}", i), f);
        }

        contributions
    }

    /// Normalize scores
    fn normalize_scores(&self, results: &mut [RankedResult]) {
        if results.is_empty() {
            return;
        }

        match self.config.normalization {
            ScoreNormalization::None => {}
            ScoreNormalization::MinMax => {
                let min = results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
                let max = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
                let range = max - min;
                if range > 0.0 {
                    for r in results {
                        r.score = (r.score - min) / range;
                    }
                }
            }
            ScoreNormalization::ZScore => {
                let mean: f32 = results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32;
                let variance: f32 = results.iter()
                    .map(|r| (r.score - mean).powi(2))
                    .sum::<f32>() / results.len() as f32;
                let std = variance.sqrt();
                if std > 0.0 {
                    for r in results {
                        r.score = (r.score - mean) / std;
                    }
                }
            }
            ScoreNormalization::Softmax => {
                let max = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = results.iter().map(|r| (r.score - max).exp()).sum();
                for r in results {
                    r.score = (r.score - max).exp() / exp_sum;
                }
            }
        }
    }

    /// Sigmoid activation
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Clear model session cache
    #[cfg(feature = "embeddings")]
    pub fn clear_cache() -> Result<()> {
        get_session_cache()?.clear()
    }

    /// Clear model session cache (stub for non-embeddings builds)
    #[cfg(not(feature = "embeddings"))]
    pub fn clear_cache() -> Result<()> {
        Ok(())
    }
}

/// Multi-stage ranking pipeline
pub struct RankingPipeline {
    stages: Vec<RankingStage>,
}

/// Ranking stage
pub struct RankingStage {
    #[allow(dead_code)]
    name: String,
    ranker: MLRanker,
    top_k: usize,
}

impl RankingPipeline {
    /// Create a new pipeline
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a ranking stage
    pub fn add_stage(
        &mut self,
        name: impl Into<String>,
        ranker: MLRanker,
        top_k: usize,
    ) {
        self.stages.push(RankingStage {
            name: name.into(),
            ranker,
            top_k,
        });
    }

    /// Execute the pipeline
    pub fn execute(
        &mut self,
        query: &[f32],
        initial_candidates: Vec<RankingCandidate>,
    ) -> Result<Vec<RankedResult>> {
        let mut candidates = initial_candidates;

        for stage in &mut self.stages {
            let results = stage.ranker.rerank(query, candidates)?;

            // Convert results back to candidates for next stage
            candidates = results.into_iter()
                .take(stage.top_k)
                .map(|r| RankingCandidate {
                    id: r.id,
                    original_score: r.score,
                    vector: Vec::new(), // Would need to carry this through
                    metadata: r.metadata,
                    features: None,
                })
                .collect();
        }

        // Final rerank or just convert
        if let Some(last_stage) = self.stages.last_mut() {
            last_stage.ranker.rerank(query, candidates)
        } else {
            Ok(candidates.into_iter()
                .map(|c| RankedResult {
                    id: c.id,
                    score: c.original_score,
                    original_score: c.original_score,
                    feature_contributions: None,
                    metadata: c.metadata,
                })
                .collect())
        }
    }

    /// Warmup all rankers in the pipeline
    pub fn warmup(&mut self) -> Result<()> {
        for stage in &mut self.stages {
            stage.ranker.warmup()?;
        }
        Ok(())
    }
}

impl Default for RankingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_model() {
        let config = RankingConfig::new()
            .with_top_k(10)
            .with_final_k(5)
            .with_warmup(false);

        let mut ranker = MLRanker::new(config).unwrap();
        ranker.add_model(RankingModel::linear(vec![0.5, 0.3, 0.2], 0.0));

        let candidates: Vec<RankingCandidate> = (0..10)
            .map(|i| RankingCandidate {
                id: format!("doc{}", i),
                original_score: 1.0 - (i as f32 * 0.1),
                vector: vec![0.1f32; 64],
                metadata: None,
                features: None,
            })
            .collect();

        let query = vec![0.1f32; 64];
        let results = ranker.rerank(&query, candidates).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_normalization() {
        let config = RankingConfig::new()
            .with_normalization(ScoreNormalization::MinMax)
            .with_warmup(false);

        let mut ranker = MLRanker::new(config).unwrap();

        let candidates: Vec<RankingCandidate> = (0..5)
            .map(|i| RankingCandidate {
                id: format!("doc{}", i),
                original_score: i as f32,
                vector: vec![0.1f32; 64],
                metadata: None,
                features: None,
            })
            .collect();

        let query = vec![0.1f32; 64];
        let results = ranker.rerank(&query, candidates).unwrap();

        // After MinMax normalization, scores should be in [0, 1]
        for r in &results {
            assert!(r.score >= 0.0 && r.score <= 1.0);
        }
    }

    #[test]
    fn test_onnx_model_config() {
        let config = OnnxModelConfig::new("model.onnx")
            .with_feature_dim(10)
            .with_sigmoid()
            .with_input_name("features")
            .with_output_name("score");

        assert_eq!(config.path, "model.onnx");
        assert_eq!(config.feature_dim, 10);
        assert!(config.apply_sigmoid);
        assert_eq!(config.input_name, "features");
        assert_eq!(config.output_name, "score");
    }

    #[test]
    fn test_xgboost_model_config() {
        let config = OnnxModelConfig::xgboost("xgboost.onnx");
        assert_eq!(config.model_type, OnnxModelType::XGBoost);
        assert!(config.apply_sigmoid);
    }

    #[test]
    fn test_lightgbm_model_config() {
        let config = OnnxModelConfig::lightgbm("lgbm.onnx");
        assert_eq!(config.model_type, OnnxModelType::LightGBM);
        assert!(config.apply_sigmoid);
    }

    #[test]
    fn test_ranking_model_constructors() {
        let onnx = RankingModel::onnx("model.onnx");
        assert!(onnx.model_path().is_some());

        let xgboost = RankingModel::xgboost("xgb.onnx", 10);
        assert_eq!(xgboost.feature_dim(), Some(10));

        let lgbm = RankingModel::lightgbm("lgbm.onnx", 15);
        assert_eq!(lgbm.feature_dim(), Some(15));

        let linear = RankingModel::linear(vec![1.0, 2.0, 3.0], 0.5);
        assert_eq!(linear.feature_dim(), Some(3));
    }

    #[test]
    fn test_feature_extraction() {
        let config = RankingConfig::new().with_warmup(false);
        let ranker = MLRanker::new(config).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let candidate = RankingCandidate {
            id: "test".to_string(),
            original_score: 0.9,
            vector: vec![0.5, 0.5, 0.0],
            metadata: None,
            features: None,
        };

        let features = ranker.compute_features(&query, &candidate);

        // Check that features are computed
        assert_eq!(features.vector_score, 0.9);
        assert!(!features.qd_features.is_empty());

        // Verify feature vector can be converted
        let fv = features.to_vector();
        assert!(!fv.is_empty());
    }

    #[test]
    fn test_empty_candidates() {
        let config = RankingConfig::new().with_warmup(false);
        let mut ranker = MLRanker::new(config).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = ranker.rerank(&query, Vec::new()).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_no_models_uses_original_scores() {
        let config = RankingConfig::new()
            .with_final_k(10)
            .with_warmup(false);
        let mut ranker = MLRanker::new(config).unwrap();

        let candidates: Vec<RankingCandidate> = vec![
            RankingCandidate {
                id: "doc1".to_string(),
                original_score: 0.9,
                vector: vec![1.0],
                metadata: None,
                features: None,
            },
            RankingCandidate {
                id: "doc2".to_string(),
                original_score: 0.8,
                vector: vec![0.5],
                metadata: None,
                features: None,
            },
        ];

        let query = vec![1.0];
        let results = ranker.rerank(&query, candidates).unwrap();

        assert_eq!(results.len(), 2);
        // Without models, original scores should be used
        assert_eq!(results[0].id, "doc1");
        assert_eq!(results[0].score, 0.9);
    }

    #[test]
    fn test_batch_size_config() {
        let config = RankingConfig::new()
            .with_batch_size(64)
            .with_num_threads(8);

        assert_eq!(config.batch_size, 64);
        assert_eq!(config.num_threads, 8);
    }

    #[test]
    fn test_zscore_normalization() {
        let config = RankingConfig::new()
            .with_normalization(ScoreNormalization::ZScore)
            .with_warmup(false);

        let mut ranker = MLRanker::new(config).unwrap();

        let candidates: Vec<RankingCandidate> = (0..10)
            .map(|i| RankingCandidate {
                id: format!("doc{}", i),
                original_score: i as f32,
                vector: vec![0.1f32; 4],
                metadata: None,
                features: None,
            })
            .collect();

        let query = vec![0.1f32; 4];
        let results = ranker.rerank(&query, candidates).unwrap();

        // ZScore should center around 0
        let sum: f32 = results.iter().map(|r| r.score).sum();
        assert!((sum / results.len() as f32).abs() < 0.1); // Mean should be close to 0
    }

    #[test]
    fn test_softmax_normalization() {
        let config = RankingConfig::new()
            .with_normalization(ScoreNormalization::Softmax)
            .with_warmup(false);

        let mut ranker = MLRanker::new(config).unwrap();

        let candidates: Vec<RankingCandidate> = (0..5)
            .map(|i| RankingCandidate {
                id: format!("doc{}", i),
                original_score: i as f32,
                vector: vec![0.1f32; 4],
                metadata: None,
                features: None,
            })
            .collect();

        let query = vec![0.1f32; 4];
        let results = ranker.rerank(&query, candidates).unwrap();

        // Softmax scores should sum to 1
        let sum: f32 = results.iter().map(|r| r.score).sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ensemble_weights() {
        let config = RankingConfig {
            ensemble_weights: vec![0.7, 0.3],
            warmup_models: false,
            ..Default::default()
        };

        let mut ranker = MLRanker::new(config).unwrap();
        ranker.add_model(RankingModel::linear(vec![1.0], 0.0));
        ranker.add_model(RankingModel::linear(vec![0.5], 0.0));

        let candidates = vec![RankingCandidate {
            id: "test".to_string(),
            original_score: 0.5,
            vector: vec![0.5],
            metadata: None,
            features: Some(RankingFeatures::new(0.5)),
        }];

        let query = vec![0.5];
        let results = ranker.rerank(&query, candidates).unwrap();

        assert_eq!(results.len(), 1);
        // Score should be weighted combination
    }

    #[test]
    fn test_pipeline_execution() {
        let config1 = RankingConfig::new()
            .with_final_k(5)
            .with_warmup(false);
        let config2 = RankingConfig::new()
            .with_final_k(3)
            .with_warmup(false);

        let ranker1 = MLRanker::new(config1).unwrap();
        let ranker2 = MLRanker::new(config2).unwrap();

        let mut pipeline = RankingPipeline::new();
        pipeline.add_stage("first", ranker1, 5);
        pipeline.add_stage("second", ranker2, 3);

        let candidates: Vec<RankingCandidate> = (0..10)
            .map(|i| RankingCandidate {
                id: format!("doc{}", i),
                original_score: i as f32 / 10.0,
                vector: vec![0.1f32; 4],
                metadata: None,
                features: None,
            })
            .collect();

        let query = vec![0.1f32; 4];
        let results = pipeline.execute(&query, candidates).unwrap();

        assert_eq!(results.len(), 3);
    }
}
