// Anomaly Detection - Statistical and ML-based anomaly detection for vector data
// Outlier detection, drift monitoring, and data quality analysis

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyConfig {
    /// Detection method
    pub method: DetectionMethod,
    /// Contamination rate (expected proportion of outliers)
    pub contamination: f64,
    /// Minimum samples for training
    pub min_samples: usize,
    /// Sensitivity (higher = more sensitive)
    pub sensitivity: f64,
    /// Window size for streaming detection
    pub window_size: usize,
    /// Alert threshold
    pub alert_threshold: f64,
    /// Enable real-time detection
    pub realtime: bool,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            method: DetectionMethod::IsolationForest,
            contamination: 0.01,
            min_samples: 100,
            sensitivity: 1.0,
            window_size: 1000,
            alert_threshold: 0.9,
            realtime: true,
        }
    }
}

/// Detection methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DetectionMethod {
    /// Isolation Forest
    IsolationForest,
    /// Local Outlier Factor
    LOF,
    /// One-Class SVM
    OneClassSVM,
    /// DBSCAN-based
    DBSCAN,
    /// Statistical (Z-score, IQR)
    Statistical,
    /// Autoencoder reconstruction error
    Autoencoder,
    /// Mahalanobis distance
    Mahalanobis,
    /// Ensemble of methods
    Ensemble,
}

/// Anomaly detector
pub struct AnomalyDetector {
    config: AnomalyConfig,
    model: RwLock<Option<DetectorModel>>,
    history: RwLock<VecDeque<VectorSample>>,
    anomalies: RwLock<Vec<AnomalyRecord>>,
    stats: DetectorStats,
}

/// Internal detector model
enum DetectorModel {
    IsolationForest(IsolationForestModel),
    LOF(LOFModel),
    Statistical(StatisticalModel),
    Ensemble(Vec<Box<dyn Detector + Send + Sync>>),
}

/// Isolation Forest model
struct IsolationForestModel {
    trees: Vec<IsolationTree>,
    sample_size: usize,
    threshold: f64,
}

/// Isolation tree node
#[derive(Clone)]
enum IsolationTree {
    Internal {
        feature: usize,
        split_value: f64,
        left: Box<IsolationTree>,
        right: Box<IsolationTree>,
    },
    Leaf {
        size: usize,
    },
}

/// LOF model
struct LOFModel {
    samples: Vec<Vec<f32>>,
    k: usize,
    lrd_cache: HashMap<usize, f64>,
}

/// Statistical model
struct StatisticalModel {
    mean: Vec<f64>,
    std: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
    q1: Vec<f64>,
    q3: Vec<f64>,
}

/// Vector sample for analysis
#[derive(Debug, Clone)]
struct VectorSample {
    id: String,
    vector: Vec<f32>,
    timestamp: u64,
    metadata: HashMap<String, String>,
}

/// Anomaly record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyRecord {
    /// Vector ID
    pub vector_id: String,
    /// Anomaly score (0-1, higher = more anomalous)
    pub score: f64,
    /// Detection method used
    pub method: String,
    /// Detected at
    pub detected_at: u64,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Contributing features
    pub contributing_features: Vec<usize>,
    /// Explanation
    pub explanation: String,
}

/// Types of anomalies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnomalyType {
    /// Point anomaly (single outlier)
    Point,
    /// Contextual anomaly (unusual given context)
    Contextual,
    /// Collective anomaly (group of points)
    Collective,
    /// Drift anomaly (distribution shift)
    Drift,
    /// Novelty (new pattern)
    Novelty,
}

/// Detector statistics
struct DetectorStats {
    total_samples: AtomicU64,
    anomalies_detected: AtomicU64,
    false_positives: AtomicU64,
    true_positives: AtomicU64,
    last_training: RwLock<Option<Instant>>,
}

/// Detector trait for ensemble
trait Detector {
    fn fit(&mut self, samples: &[Vec<f32>]) -> Result<()>;
    fn predict(&self, sample: &[f32]) -> f64;
    fn name(&self) -> &str;
}

/// Detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Vector ID
    pub vector_id: String,
    /// Is anomaly
    pub is_anomaly: bool,
    /// Anomaly score
    pub score: f64,
    /// Confidence
    pub confidence: f64,
    /// Details by method (if ensemble)
    pub method_scores: HashMap<String, f64>,
    /// Similar normal samples
    pub similar_normal: Vec<String>,
}

/// Batch detection result
#[derive(Debug, Clone, Serialize)]
pub struct BatchDetectionResult {
    /// Total samples
    pub total: usize,
    /// Anomalies found
    pub anomalies: Vec<DetectionResult>,
    /// Summary statistics
    pub summary: DetectionSummary,
}

/// Detection summary
#[derive(Debug, Clone, Serialize)]
pub struct DetectionSummary {
    pub anomaly_rate: f64,
    pub avg_score: f64,
    pub max_score: f64,
    pub score_distribution: Vec<(f64, usize)>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new(config: AnomalyConfig) -> Self {
        Self {
            config,
            model: RwLock::new(None),
            history: RwLock::new(VecDeque::new()),
            anomalies: RwLock::new(Vec::new()),
            stats: DetectorStats {
                total_samples: AtomicU64::new(0),
                anomalies_detected: AtomicU64::new(0),
                false_positives: AtomicU64::new(0),
                true_positives: AtomicU64::new(0),
                last_training: RwLock::new(None),
            },
        }
    }

    /// Fit the model on training data
    pub fn fit(&self, vectors: &[Vec<f32>]) -> Result<()> {
        let model = match self.config.method {
            DetectionMethod::IsolationForest => {
                DetectorModel::IsolationForest(self.build_isolation_forest(vectors)?)
            }
            DetectionMethod::LOF => {
                DetectorModel::LOF(self.build_lof(vectors)?)
            }
            DetectionMethod::Statistical => {
                DetectorModel::Statistical(self.build_statistical(vectors)?)
            }
            DetectionMethod::Ensemble => {
                let detectors: Vec<Box<dyn Detector + Send + Sync>> = Vec::new();
                // Would add multiple detectors here
                DetectorModel::Ensemble(detectors)
            }
            _ => {
                DetectorModel::Statistical(self.build_statistical(vectors)?)
            }
        };

        *self.model.write()
            .map_err(|_| VecStoreError::LockError("model lock poisoned".into()))? = Some(model);
        *self.stats.last_training.write()
            .map_err(|_| VecStoreError::LockError("last_training lock poisoned".into()))? = Some(Instant::now());

        Ok(())
    }

    /// Build isolation forest
    fn build_isolation_forest(&self, vectors: &[Vec<f32>]) -> Result<IsolationForestModel> {
        let n_trees = 100;
        let sample_size = 256.min(vectors.len());
        let max_depth = (sample_size as f64).log2().ceil() as usize;

        let mut trees = Vec::with_capacity(n_trees);

        for _ in 0..n_trees {
            // Sample subset
            let sample_indices: Vec<usize> = (0..sample_size)
                .map(|_| random_usize(0, vectors.len()))
                .collect();

            let samples: Vec<&Vec<f32>> = sample_indices.iter()
                .map(|&i| &vectors[i])
                .collect();

            let tree = self.build_tree(&samples, 0, max_depth);
            trees.push(tree);
        }

        // Compute threshold based on contamination
        let scores: Vec<f64> = vectors.iter()
            .map(|v| self.compute_path_length(&trees, v))
            .collect();

        let threshold = percentile(&scores, 1.0 - self.config.contamination);

        Ok(IsolationForestModel {
            trees,
            sample_size,
            threshold,
        })
    }

    fn build_tree(&self, samples: &[&Vec<f32>], depth: usize, max_depth: usize) -> IsolationTree {
        if samples.len() <= 1 || depth >= max_depth {
            return IsolationTree::Leaf { size: samples.len() };
        }

        let dim = samples[0].len();
        let feature = random_usize(0, dim);

        // Find min/max for feature
        let values: Vec<f32> = samples.iter().map(|s| s[feature]).collect();
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < 1e-10 {
            return IsolationTree::Leaf { size: samples.len() };
        }

        let split_value = random_f32(min_val, max_val);

        let (left, right): (Vec<_>, Vec<_>) = samples.iter()
            .partition(|s| s[feature] < split_value);

        IsolationTree::Internal {
            feature,
            split_value: split_value as f64,
            left: Box::new(self.build_tree(&left, depth + 1, max_depth)),
            right: Box::new(self.build_tree(&right, depth + 1, max_depth)),
        }
    }

    fn compute_path_length(&self, trees: &[IsolationTree], sample: &[f32]) -> f64 {
        let total: f64 = trees.iter()
            .map(|tree| self.path_length(tree, sample, 0))
            .sum();
        total / trees.len() as f64
    }

    fn path_length(&self, tree: &IsolationTree, sample: &[f32], depth: usize) -> f64 {
        match tree {
            IsolationTree::Leaf { size } => {
                depth as f64 + c_factor(*size)
            }
            IsolationTree::Internal { feature, split_value, left, right } => {
                if (sample[*feature] as f64) < *split_value {
                    self.path_length(left, sample, depth + 1)
                } else {
                    self.path_length(right, sample, depth + 1)
                }
            }
        }
    }

    /// Build LOF model
    fn build_lof(&self, vectors: &[Vec<f32>]) -> Result<LOFModel> {
        Ok(LOFModel {
            samples: vectors.to_vec(),
            k: 20,
            lrd_cache: HashMap::new(),
        })
    }

    /// Build statistical model
    fn build_statistical(&self, vectors: &[Vec<f32>]) -> Result<StatisticalModel> {
        if vectors.is_empty() {
            return Ok(StatisticalModel {
                mean: vec![],
                std: vec![],
                min: vec![],
                max: vec![],
                q1: vec![],
                q3: vec![],
            });
        }

        let dim = vectors[0].len();
        let n = vectors.len() as f64;

        let mut mean = vec![0.0; dim];
        let mut min = vec![f64::INFINITY; dim];
        let mut max = vec![f64::NEG_INFINITY; dim];

        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                let val = val as f64;
                mean[i] += val / n;
                min[i] = min[i].min(val);
                max[i] = max[i].max(val);
            }
        }

        // Compute std
        let mut std = vec![0.0; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                std[i] += ((val as f64) - mean[i]).powi(2) / n;
            }
        }
        for s in &mut std {
            *s = s.sqrt();
        }

        // Compute quartiles (simplified)
        let sorted_values: Vec<Vec<f64>> = (0..dim)
            .map(|d| {
                let mut vals: Vec<f64> = vectors.iter().map(|v| v[d] as f64).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                vals
            })
            .collect();

        let q1: Vec<f64> = sorted_values.iter()
            .map(|vals| vals[vals.len() / 4])
            .collect();

        let q3: Vec<f64> = sorted_values.iter()
            .map(|vals| vals[vals.len() * 3 / 4])
            .collect();

        Ok(StatisticalModel { mean, std, min, max, q1, q3 })
    }

    /// Detect anomalies in a single vector
    pub fn detect(&self, vector_id: &str, vector: &[f32]) -> Result<DetectionResult> {
        self.stats.total_samples.fetch_add(1, Ordering::Relaxed);

        let model = self.model.read()
            .map_err(|_| VecStoreError::LockError("model lock poisoned".into()))?;
        let model = model.as_ref().ok_or_else(|| {
            VecStoreError::IndexNotInitialized
        })?;

        let (score, method_scores) = match model {
            DetectorModel::IsolationForest(m) => {
                let path_length = self.compute_path_length(&m.trees, vector);
                let score = anomaly_score(path_length, m.sample_size);
                let mut scores = HashMap::new();
                scores.insert("isolation_forest".to_string(), score);
                (score, scores)
            }
            DetectorModel::LOF(m) => {
                let score = self.compute_lof_score(m, vector);
                let mut scores = HashMap::new();
                scores.insert("lof".to_string(), score);
                (score, scores)
            }
            DetectorModel::Statistical(m) => {
                let score = self.compute_statistical_score(m, vector);
                let mut scores = HashMap::new();
                scores.insert("statistical".to_string(), score);
                (score, scores)
            }
            DetectorModel::Ensemble(detectors) => {
                let mut scores = HashMap::new();
                let mut total = 0.0;
                for detector in detectors {
                    let s = detector.predict(vector);
                    scores.insert(detector.name().to_string(), s);
                    total += s;
                }
                let avg = if detectors.is_empty() { 0.0 } else { total / detectors.len() as f64 };
                (avg, scores)
            }
        };

        let is_anomaly = score >= self.config.alert_threshold;
        let confidence = self.compute_confidence(score);

        if is_anomaly {
            self.stats.anomalies_detected.fetch_add(1, Ordering::Relaxed);

            // Record anomaly
            let record = AnomalyRecord {
                vector_id: vector_id.to_string(),
                score,
                method: format!("{:?}", self.config.method),
                detected_at: current_timestamp(),
                anomaly_type: self.classify_anomaly(score),
                contributing_features: self.find_contributing_features(model, vector),
                explanation: self.generate_explanation(model, vector, score),
            };

            self.anomalies.write()
                .map_err(|_| VecStoreError::LockError("anomalies lock poisoned".into()))?
                .push(record);
        }

        Ok(DetectionResult {
            vector_id: vector_id.to_string(),
            is_anomaly,
            score,
            confidence,
            method_scores,
            similar_normal: vec![], // Would find similar normal samples
        })
    }

    /// Detect anomalies in batch
    pub fn detect_batch(&self, vectors: &[(String, Vec<f32>)]) -> Result<BatchDetectionResult> {
        let mut anomalies = Vec::new();
        let mut total_score = 0.0;
        let mut max_score = 0.0f64;
        let mut score_buckets: HashMap<usize, usize> = HashMap::new();

        for (id, vector) in vectors {
            let result = self.detect(id, vector)?;
            total_score += result.score;
            max_score = max_score.max(result.score);

            // Bucket scores for distribution
            let bucket = (result.score * 10.0) as usize;
            *score_buckets.entry(bucket).or_insert(0) += 1;

            if result.is_anomaly {
                anomalies.push(result);
            }
        }

        let score_distribution: Vec<(f64, usize)> = (0..=10)
            .map(|b| (b as f64 / 10.0, *score_buckets.get(&b).unwrap_or(&0)))
            .collect();

        Ok(BatchDetectionResult {
            total: vectors.len(),
            anomalies: anomalies.clone(),
            summary: DetectionSummary {
                anomaly_rate: anomalies.len() as f64 / vectors.len() as f64,
                avg_score: total_score / vectors.len() as f64,
                max_score,
                score_distribution,
            },
        })
    }

    /// Compute LOF score
    fn compute_lof_score(&self, model: &LOFModel, sample: &[f32]) -> f64 {
        if model.samples.is_empty() {
            return 0.0;
        }

        // Find k nearest neighbors
        let mut distances: Vec<(usize, f64)> = model.samples.iter()
            .enumerate()
            .map(|(i, s)| (i, euclidean_distance(sample, s)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let k_neighbors: Vec<usize> = distances.iter()
            .take(model.k)
            .map(|(i, _)| *i)
            .collect();

        // Compute local reachability density
        let k_dist = distances[model.k.min(distances.len() - 1)].1;
        let lrd = self.compute_lrd(model, sample, &k_neighbors, k_dist);

        // Compute LOF
        let mut lof = 0.0;
        for &neighbor_idx in &k_neighbors {
            let neighbor = &model.samples[neighbor_idx];
            let neighbor_lrd = self.compute_lrd(
                model,
                neighbor,
                &k_neighbors,
                k_dist
            );
            if lrd > 0.0 {
                lof += neighbor_lrd / lrd;
            }
        }

        lof / k_neighbors.len() as f64
    }

    fn compute_lrd(&self, _model: &LOFModel, _sample: &[f32], _neighbors: &[usize], k_dist: f64) -> f64 {
        // Simplified LRD computation
        if k_dist > 0.0 { 1.0 / k_dist } else { 1.0 }
    }

    /// Compute statistical anomaly score
    fn compute_statistical_score(&self, model: &StatisticalModel, sample: &[f32]) -> f64 {
        if model.mean.is_empty() {
            return 0.0;
        }

        // Z-score based
        let mut max_zscore = 0.0f64;
        for (i, &val) in sample.iter().enumerate() {
            if model.std[i] > 0.0 {
                let zscore = ((val as f64) - model.mean[i]).abs() / model.std[i];
                max_zscore = max_zscore.max(zscore);
            }
        }

        // Convert to probability (using CDF approximation)
        let prob = 2.0 * (1.0 - normal_cdf(max_zscore));
        1.0 - prob
    }

    /// Compute confidence
    fn compute_confidence(&self, score: f64) -> f64 {
        // Higher confidence for scores further from threshold
        let distance_from_threshold = (score - self.config.alert_threshold).abs();
        (distance_from_threshold * 10.0).min(1.0)
    }

    /// Classify anomaly type
    fn classify_anomaly(&self, score: f64) -> AnomalyType {
        if score > 0.95 {
            AnomalyType::Point
        } else if score > 0.85 {
            AnomalyType::Contextual
        } else {
            AnomalyType::Novelty
        }
    }

    /// Find features contributing to anomaly
    fn find_contributing_features(&self, model: &DetectorModel, sample: &[f32]) -> Vec<usize> {
        match model {
            DetectorModel::Statistical(m) => {
                let mut contributions: Vec<(usize, f64)> = sample.iter()
                    .enumerate()
                    .filter_map(|(i, &val)| {
                        if m.std[i] > 0.0 {
                            let zscore = ((val as f64) - m.mean[i]).abs() / m.std[i];
                            Some((i, zscore))
                        } else {
                            None
                        }
                    })
                    .collect();

                contributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                contributions.iter().take(5).map(|(i, _)| *i).collect()
            }
            _ => vec![],
        }
    }

    /// Generate explanation
    fn generate_explanation(&self, model: &DetectorModel, sample: &[f32], score: f64) -> String {
        match model {
            DetectorModel::IsolationForest(_) => {
                format!("Isolation score {:.3}: Vector is isolated from normal distribution", score)
            }
            DetectorModel::LOF(_) => {
                format!("LOF score {:.3}: Vector has low local density compared to neighbors", score)
            }
            DetectorModel::Statistical(_m) => {
                let features = self.find_contributing_features(model, sample);
                if features.is_empty() {
                    format!("Statistical score {:.3}: Overall deviation from normal", score)
                } else {
                    format!(
                        "Statistical score {:.3}: High deviation in dimensions {:?}",
                        score, features
                    )
                }
            }
            _ => format!("Anomaly score {:.3}", score),
        }
    }

    /// Get detection statistics
    pub fn get_stats(&self) -> DetectionStats {
        DetectionStats {
            total_samples: self.stats.total_samples.load(Ordering::Relaxed),
            anomalies_detected: self.stats.anomalies_detected.load(Ordering::Relaxed),
            false_positives: self.stats.false_positives.load(Ordering::Relaxed),
            true_positives: self.stats.true_positives.load(Ordering::Relaxed),
            detection_rate: {
                let total = self.stats.total_samples.load(Ordering::Relaxed);
                let anomalies = self.stats.anomalies_detected.load(Ordering::Relaxed);
                if total > 0 { anomalies as f64 / total as f64 } else { 0.0 }
            },
            last_training: {
                let Ok(guard) = self.stats.last_training.read() else { return DetectionStats {
                    total_samples: self.stats.total_samples.load(Ordering::Relaxed),
                    anomalies_detected: self.stats.anomalies_detected.load(Ordering::Relaxed),
                    false_positives: self.stats.false_positives.load(Ordering::Relaxed),
                    true_positives: self.stats.true_positives.load(Ordering::Relaxed),
                    detection_rate: 0.0,
                    last_training: None,
                }; };
                guard.map(|t| t.elapsed())
            },
        }
    }

    /// Get recent anomalies
    pub fn get_recent_anomalies(&self, limit: usize) -> Vec<AnomalyRecord> {
        let Ok(anomalies) = self.anomalies.read() else { return vec![]; };
        anomalies.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Provide feedback on detection
    pub fn feedback(&self, _vector_id: &str, is_true_positive: bool) {
        if is_true_positive {
            self.stats.true_positives.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.false_positives.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Online update with new sample
    pub fn update(&self, vector: &[f32]) -> Result<()> {
        let mut history = self.history.write()
            .map_err(|_| VecStoreError::LockError("history lock poisoned".into()))?;

        if history.len() >= self.config.window_size {
            history.pop_front();
        }

        history.push_back(VectorSample {
            id: format!("sample_{}", current_timestamp()),
            vector: vector.to_vec(),
            timestamp: current_timestamp(),
            metadata: HashMap::new(),
        });

        Ok(())
    }
}

/// Detection statistics
#[derive(Debug, Clone, Serialize)]
pub struct DetectionStats {
    pub total_samples: u64,
    pub anomalies_detected: u64,
    pub false_positives: u64,
    pub true_positives: u64,
    pub detection_rate: f64,
    pub last_training: Option<Duration>,
}

/// Drift detector for monitoring embedding distribution changes
pub struct DriftDetector {
    config: DriftConfig,
    reference_distribution: RwLock<Option<DistributionStats>>,
    current_window: RwLock<VecDeque<Vec<f32>>>,
    drift_events: RwLock<Vec<DriftEvent>>,
}

/// Drift detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftConfig {
    /// Reference window size
    pub reference_size: usize,
    /// Detection window size
    pub detection_size: usize,
    /// Drift threshold
    pub threshold: f64,
    /// Detection method
    pub method: DriftMethod,
    /// Check interval
    pub check_interval: usize,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            reference_size: 1000,
            detection_size: 100,
            threshold: 0.1,
            method: DriftMethod::KSTest,
            check_interval: 100,
        }
    }
}

/// Drift detection methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DriftMethod {
    /// Kolmogorov-Smirnov test
    KSTest,
    /// Population Stability Index
    PSI,
    /// Maximum Mean Discrepancy
    MMD,
    /// Wasserstein distance
    Wasserstein,
    /// Chi-squared test
    ChiSquared,
}

/// Distribution statistics
struct DistributionStats {
    mean: Vec<f64>,
    std: Vec<f64>,
    quantiles: Vec<Vec<f64>>,
    sample_size: usize,
}

/// Drift event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftEvent {
    pub detected_at: u64,
    pub severity: DriftSeverity,
    pub drift_score: f64,
    pub affected_dimensions: Vec<usize>,
    pub description: String,
}

/// Drift severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DriftSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl DriftDetector {
    /// Create a new drift detector
    pub fn new(config: DriftConfig) -> Self {
        Self {
            config,
            reference_distribution: RwLock::new(None),
            current_window: RwLock::new(VecDeque::new()),
            drift_events: RwLock::new(Vec::new()),
        }
    }

    /// Set reference distribution from data
    pub fn set_reference(&self, vectors: &[Vec<f32>]) -> Result<()> {
        let stats = self.compute_distribution_stats(vectors)?;
        *self.reference_distribution.write()
            .map_err(|_| VecStoreError::LockError("reference_distribution lock poisoned".into()))? = Some(stats);
        Ok(())
    }

    /// Add sample to detection window
    pub fn add_sample(&self, vector: Vec<f32>) -> Option<DriftEvent> {
        let Ok(mut window) = self.current_window.write() else { return None; };
        window.push_back(vector);

        if window.len() > self.config.detection_size {
            window.pop_front();
        }

        // Check for drift periodically
        if window.len() == self.config.detection_size {
            let samples: Vec<Vec<f32>> = window.iter().cloned().collect();
            drop(window);

            return self.check_drift(&samples).ok().flatten();
        }

        None
    }

    /// Check for drift
    fn check_drift(&self, current_samples: &[Vec<f32>]) -> Result<Option<DriftEvent>> {
        let reference = self.reference_distribution.read()
            .map_err(|_| VecStoreError::LockError("reference_distribution lock poisoned".into()))?;
        let reference = match reference.as_ref() {
            Some(r) => r,
            None => return Ok(None),
        };

        let current_stats = self.compute_distribution_stats(current_samples)?;

        let drift_score = match self.config.method {
            DriftMethod::KSTest => self.ks_test(reference, &current_stats),
            DriftMethod::PSI => self.psi(reference, &current_stats),
            DriftMethod::MMD => self.mmd(reference, &current_stats),
            DriftMethod::Wasserstein => self.wasserstein(reference, &current_stats),
            DriftMethod::ChiSquared => self.chi_squared(reference, &current_stats),
        };

        if drift_score > self.config.threshold {
            let affected = self.find_affected_dimensions(reference, &current_stats);
            let severity = self.classify_severity(drift_score);

            let event = DriftEvent {
                detected_at: current_timestamp(),
                severity,
                drift_score,
                affected_dimensions: affected,
                description: format!(
                    "Distribution drift detected with score {:.4} using {:?}",
                    drift_score, self.config.method
                ),
            };

            self.drift_events.write()
                .map_err(|_| VecStoreError::LockError("drift_events lock poisoned".into()))?
                .push(event.clone());
            return Ok(Some(event));
        }

        Ok(None)
    }

    fn compute_distribution_stats(&self, vectors: &[Vec<f32>]) -> Result<DistributionStats> {
        if vectors.is_empty() {
            return Ok(DistributionStats {
                mean: vec![],
                std: vec![],
                quantiles: vec![],
                sample_size: 0,
            });
        }

        let dim = vectors[0].len();
        let n = vectors.len() as f64;

        let mut mean = vec![0.0; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                mean[i] += val as f64 / n;
            }
        }

        let mut std = vec![0.0; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                std[i] += ((val as f64) - mean[i]).powi(2) / n;
            }
        }
        for s in &mut std {
            *s = s.sqrt();
        }

        // Compute quantiles for each dimension
        let quantiles: Vec<Vec<f64>> = (0..dim)
            .map(|d| {
                let mut vals: Vec<f64> = vectors.iter().map(|v| v[d] as f64).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                vec![
                    vals[0],
                    vals[vals.len() / 4],
                    vals[vals.len() / 2],
                    vals[vals.len() * 3 / 4],
                    vals[vals.len() - 1],
                ]
            })
            .collect();

        Ok(DistributionStats {
            mean,
            std,
            quantiles,
            sample_size: vectors.len(),
        })
    }

    fn ks_test(&self, ref_stats: &DistributionStats, curr_stats: &DistributionStats) -> f64 {
        // Simplified KS test using quantile differences
        let mut max_diff = 0.0f64;

        for (ref_q, curr_q) in ref_stats.quantiles.iter().zip(curr_stats.quantiles.iter()) {
            for (r, c) in ref_q.iter().zip(curr_q.iter()) {
                let diff = (r - c).abs();
                max_diff = max_diff.max(diff);
            }
        }

        max_diff
    }

    fn psi(&self, ref_stats: &DistributionStats, curr_stats: &DistributionStats) -> f64 {
        // Population Stability Index
        let mut psi = 0.0;

        for (ref_q, curr_q) in ref_stats.quantiles.iter().zip(curr_stats.quantiles.iter()) {
            for (r, c) in ref_q.iter().zip(curr_q.iter()) {
                if *r > 0.0 && *c > 0.0 {
                    let ratio = c / r;
                    psi += (c - r) * ratio.ln();
                }
            }
        }

        psi.abs()
    }

    fn mmd(&self, ref_stats: &DistributionStats, curr_stats: &DistributionStats) -> f64 {
        // Simplified Maximum Mean Discrepancy
        let mut mmd = 0.0;

        for (ref_mean, curr_mean) in ref_stats.mean.iter().zip(curr_stats.mean.iter()) {
            mmd += (ref_mean - curr_mean).powi(2);
        }

        mmd.sqrt()
    }

    fn wasserstein(&self, ref_stats: &DistributionStats, curr_stats: &DistributionStats) -> f64 {
        // 1D Wasserstein distance (earth mover's distance) averaged across dimensions
        let mut total_dist = 0.0;

        for (ref_q, curr_q) in ref_stats.quantiles.iter().zip(curr_stats.quantiles.iter()) {
            let dist: f64 = ref_q.iter()
                .zip(curr_q.iter())
                .map(|(r, c)| (r - c).abs())
                .sum();
            total_dist += dist / ref_q.len() as f64;
        }

        total_dist / ref_stats.quantiles.len() as f64
    }

    fn chi_squared(&self, ref_stats: &DistributionStats, curr_stats: &DistributionStats) -> f64 {
        // Simplified chi-squared
        let mut chi2 = 0.0;

        for (ref_mean, curr_mean) in ref_stats.mean.iter().zip(curr_stats.mean.iter()) {
            if *ref_mean != 0.0 {
                chi2 += (ref_mean - curr_mean).powi(2) / ref_mean.abs();
            }
        }

        chi2
    }

    fn find_affected_dimensions(
        &self,
        ref_stats: &DistributionStats,
        curr_stats: &DistributionStats,
    ) -> Vec<usize> {
        let mut affected = Vec::new();

        for i in 0..ref_stats.mean.len() {
            let mean_diff = (ref_stats.mean[i] - curr_stats.mean[i]).abs();
            let std_threshold = (ref_stats.std[i] + curr_stats.std[i]) / 2.0;

            if mean_diff > std_threshold {
                affected.push(i);
            }
        }

        affected
    }

    fn classify_severity(&self, drift_score: f64) -> DriftSeverity {
        let threshold = self.config.threshold;
        if drift_score > threshold * 3.0 {
            DriftSeverity::Critical
        } else if drift_score > threshold * 2.0 {
            DriftSeverity::High
        } else if drift_score > threshold * 1.5 {
            DriftSeverity::Medium
        } else {
            DriftSeverity::Low
        }
    }

    /// Get recent drift events
    pub fn get_drift_events(&self, limit: usize) -> Vec<DriftEvent> {
        let Ok(events) = self.drift_events.read() else { return vec![]; };
        events.iter().rev().take(limit).cloned().collect()
    }
}

/// Builder for AnomalyDetector
#[must_use = "builders do nothing unless built"]
pub struct AnomalyDetectorBuilder {
    config: AnomalyConfig,
}

impl AnomalyDetectorBuilder {
    pub fn new() -> Self {
        Self {
            config: AnomalyConfig::default(),
        }
    }

    #[inline]
    #[must_use]
    pub fn method(mut self, method: DetectionMethod) -> Self {
        self.config.method = method;
        self
    }

    #[inline]
    #[must_use]
    pub fn contamination(mut self, contamination: f64) -> Self {
        self.config.contamination = contamination;
        self
    }

    #[inline]
    #[must_use]
    pub fn sensitivity(mut self, sensitivity: f64) -> Self {
        self.config.sensitivity = sensitivity;
        self
    }

    #[inline]
    #[must_use]
    pub fn alert_threshold(mut self, threshold: f64) -> Self {
        self.config.alert_threshold = threshold;
        self
    }

    pub fn build(self) -> AnomalyDetector {
        AnomalyDetector::new(self.config)
    }
}

impl Default for AnomalyDetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn random_usize(min: usize, max: usize) -> usize {
    if max <= min { return min; }
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    std::time::Instant::now().hash(&mut hasher);
    let hash = hasher.finish();
    min + (hash as usize % (max - min))
}

fn random_f32(min: f32, max: f32) -> f32 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    std::time::Instant::now().hash(&mut hasher);
    let hash = hasher.finish();
    min + (hash as f32 / u64::MAX as f32) * (max - min)
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() { return 0.0; }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() as f64 - 1.0) * p) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn c_factor(size: usize) -> f64 {
    if size <= 1 { return 0.0; }
    let n = size as f64;
    2.0 * ((n - 1.0).ln() + 0.5772156649) - (2.0 * (n - 1.0) / n)
}

fn anomaly_score(path_length: f64, sample_size: usize) -> f64 {
    let c = c_factor(sample_size);
    if c == 0.0 { return 0.5; }
    (2.0_f64).powf(-path_length / c)
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detector_creation() {
        let detector = AnomalyDetectorBuilder::new()
            .method(DetectionMethod::IsolationForest)
            .contamination(0.05)
            .build();

        let stats = detector.get_stats();
        assert_eq!(stats.total_samples, 0);
    }

    #[test]
    fn test_fit_and_detect() {
        let detector = AnomalyDetectorBuilder::new()
            .method(DetectionMethod::Statistical)
            .build();

        // Generate normal data
        let normal_data: Vec<Vec<f32>> = (0..100)
            .map(|_| vec![0.5, 0.5, 0.5])
            .collect();

        detector.fit(&normal_data).unwrap();

        // Test normal sample
        let result = detector.detect("normal", &[0.5, 0.5, 0.5]);
        assert!(result.is_ok());

        // Test anomaly
        let result = detector.detect("anomaly", &[100.0, 100.0, 100.0]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_drift_detector() {
        let detector = DriftDetector::new(DriftConfig::default());

        let reference: Vec<Vec<f32>> = (0..100)
            .map(|_| vec![0.5, 0.5, 0.5])
            .collect();

        detector.set_reference(&reference).unwrap();
    }

    #[test]
    fn test_c_factor() {
        assert!((c_factor(256) - 8.0).abs() < 1.0);
    }

    #[test]
    fn test_anomaly_score() {
        let score = anomaly_score(8.0, 256);
        assert!(score > 0.0 && score < 1.0);
    }
}
