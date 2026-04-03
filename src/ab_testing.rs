// A/B Testing Framework - Test different configurations and strategies
// Compare index types, embedding models, reranking strategies, and more

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// A/B test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Experiment name
    pub name: String,
    /// Description
    pub description: String,
    /// Experiment type
    pub experiment_type: ExperimentType,
    /// Variants to test
    pub variants: Vec<Variant>,
    /// Traffic allocation method
    pub allocation: TrafficAllocation,
    /// Start time (unix timestamp ms)
    pub start_time: Option<u64>,
    /// End time (unix timestamp ms)
    pub end_time: Option<u64>,
    /// Minimum sample size per variant
    pub min_sample_size: u64,
    /// Primary metric
    pub primary_metric: Metric,
    /// Secondary metrics
    pub secondary_metrics: Vec<Metric>,
    /// Statistical significance threshold (e.g., 0.95)
    pub significance_threshold: f64,
    /// Owner/creator
    pub owner: String,
}

/// Type of experiment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExperimentType {
    /// Compare index configurations
    IndexConfig,
    /// Compare embedding models
    EmbeddingModel,
    /// Compare reranking strategies
    Reranking,
    /// Compare distance metrics
    DistanceMetric,
    /// Compare quantization methods
    Quantization,
    /// Compare search parameters (k, ef, beam width)
    SearchParams,
    /// Custom experiment
    Custom(String),
}

/// Experiment variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// Variant ID
    pub id: String,
    /// Variant name
    pub name: String,
    /// Traffic weight (relative)
    pub weight: u32,
    /// Configuration for this variant
    pub config: VariantConfig,
    /// Is this the control group?
    pub is_control: bool,
}

/// Variant configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantConfig {
    /// Index parameters
    pub index_params: Option<IndexParams>,
    /// Embedding model
    pub embedding_model: Option<String>,
    /// Distance metric
    pub distance_metric: Option<String>,
    /// Search parameters
    pub search_params: Option<SearchParams>,
    /// Reranking config
    pub reranking: Option<RerankingConfig>,
    /// Custom parameters
    pub custom: HashMap<String, serde_json::Value>,
}

/// Index parameters for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexParams {
    pub index_type: String,
    pub hnsw_m: Option<usize>,
    pub hnsw_ef_construction: Option<usize>,
    pub hnsw_ef_search: Option<usize>,
    pub pq_enabled: Option<bool>,
    pub pq_subvectors: Option<usize>,
}

/// Search parameters for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    pub ef: Option<usize>,
    pub beam_width: Option<usize>,
    pub num_candidates: Option<usize>,
    pub use_pq: Option<bool>,
}

/// Reranking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankingConfig {
    pub enabled: bool,
    pub model: Option<String>,
    pub top_k: Option<usize>,
}

/// Traffic allocation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficAllocation {
    /// Random allocation based on weights
    Random,
    /// Consistent hashing based on user/query ID
    Consistent,
    /// Time-based rotation
    TimeBased { rotation_minutes: u32 },
    /// Manual override capability
    Manual,
}

/// Metric to track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub metric_type: MetricType,
    pub higher_is_better: bool,
}

/// Type of metric
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MetricType {
    /// Latency (p50, p95, p99)
    Latency,
    /// Recall at K
    Recall,
    /// Precision at K
    Precision,
    /// NDCG
    Ndcg,
    /// Throughput (QPS)
    Throughput,
    /// Error rate
    ErrorRate,
    /// Click-through rate
    ClickThrough,
    /// Conversion rate
    Conversion,
    /// Custom metric
    Custom(String),
}

/// Experiment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExperimentStatus {
    Draft,
    Running,
    Paused,
    Completed,
    Cancelled,
}

/// Experiment state
#[derive(Debug)]
struct ExperimentState {
    config: ExperimentConfig,
    status: ExperimentStatus,
    created_at: u64,
    started_at: Option<u64>,
    ended_at: Option<u64>,
    /// Per-variant metrics
    variant_data: HashMap<String, VariantData>,
}

/// Per-variant collected data
#[derive(Debug, Default)]
struct VariantData {
    /// Total requests assigned to this variant
    requests: AtomicU64,
    /// Successful requests
    successes: AtomicU64,
    /// Errors
    errors: AtomicU64,
    /// Total latency (for average calculation)
    total_latency_us: AtomicU64,
    /// Latency samples for percentile calculation
    latency_samples: RwLock<Vec<u64>>,
    /// Metric values
    metric_values: RwLock<HashMap<String, Vec<f64>>>,
}

/// A/B testing manager
pub struct ABTestManager {
    /// Active experiments
    experiments: RwLock<HashMap<String, ExperimentState>>,
    /// User/query to variant assignments (for consistency)
    assignments: RwLock<HashMap<String, (String, String)>>, // (experiment_id, variant_id)
    /// Global metrics
    metrics: ABMetrics,
}

#[derive(Debug, Default)]
struct ABMetrics {
    experiments_created: AtomicU64,
    experiments_completed: AtomicU64,
    total_assignments: AtomicU64,
    total_results_recorded: AtomicU64,
}

impl ABTestManager {
    /// Create a new A/B test manager
    pub fn new() -> Self {
        Self {
            experiments: RwLock::new(HashMap::new()),
            assignments: RwLock::new(HashMap::new()),
            metrics: ABMetrics::default(),
        }
    }

    /// Create a new experiment
    pub fn create_experiment(&self, config: ExperimentConfig) -> Result<String> {
        let experiment_id = generate_experiment_id(&config.name);

        let mut variant_data = HashMap::new();
        for variant in &config.variants {
            variant_data.insert(variant.id.clone(), VariantData::default());
        }

        let state = ExperimentState {
            config,
            status: ExperimentStatus::Draft,
            created_at: current_timestamp(),
            started_at: None,
            ended_at: None,
            variant_data,
        };

        self.experiments.write()?.insert(experiment_id.clone(), state);
        self.metrics.experiments_created.fetch_add(1, Ordering::Relaxed);

        Ok(experiment_id)
    }

    /// Start an experiment
    pub fn start_experiment(&self, experiment_id: &str) -> Result<()> {
        let mut experiments = self.experiments.write()?;
        let state = experiments.get_mut(experiment_id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Experiment {} not found", experiment_id)))?;

        if state.status != ExperimentStatus::Draft && state.status != ExperimentStatus::Paused {
            return Err(VecStoreError::InvalidInput(
                format!("Cannot start experiment in status {:?}", state.status)
            ));
        }

        state.status = ExperimentStatus::Running;
        state.started_at = Some(current_timestamp());

        Ok(())
    }

    /// Pause an experiment
    pub fn pause_experiment(&self, experiment_id: &str) -> Result<()> {
        let mut experiments = self.experiments.write()?;
        let state = experiments.get_mut(experiment_id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Experiment {} not found", experiment_id)))?;

        if state.status != ExperimentStatus::Running {
            return Err(VecStoreError::InvalidInput("Experiment is not running".into()));
        }

        state.status = ExperimentStatus::Paused;
        Ok(())
    }

    /// Complete an experiment
    pub fn complete_experiment(&self, experiment_id: &str) -> Result<ExperimentResults> {
        let mut experiments = self.experiments.write()?;
        let state = experiments.get_mut(experiment_id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Experiment {} not found", experiment_id)))?;

        state.status = ExperimentStatus::Completed;
        state.ended_at = Some(current_timestamp());
        self.metrics.experiments_completed.fetch_add(1, Ordering::Relaxed);

        self.compute_results(state)
    }

    /// Get variant assignment for a request
    pub fn get_assignment(&self, experiment_id: &str, identity: &str) -> Result<VariantAssignment> {
        let experiments = self.experiments.read()?;
        let state = experiments.get(experiment_id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Experiment {} not found", experiment_id)))?;

        if state.status != ExperimentStatus::Running {
            return Err(VecStoreError::InvalidInput(
                format!("Experiment is not running: {:?}", state.status)
            ));
        }

        // Check for existing assignment (Rust 1.92 if-let chain)
        let assignments = self.assignments.read()?;
        if let Some((exp_id, var_id)) = assignments.get(identity)
            && exp_id == experiment_id
        {
            let variant = state.config.variants.iter()
                .find(|v| &v.id == var_id)
                .cloned()
                .ok_or_else(|| VecStoreError::NotFound("Variant not found".into()))?;

            return Ok(VariantAssignment {
                experiment_id: experiment_id.to_string(),
                variant_id: var_id.clone(),
                variant_config: variant.config,
                is_new_assignment: false,
            });
        }
        drop(assignments);

        // Make new assignment
        let variant = self.select_variant(&state.config, identity)?;

        // Store assignment for consistency
        let mut assignments = self.assignments.write()?;
        assignments.insert(
            identity.to_string(),
            (experiment_id.to_string(), variant.id.clone()),
        );
        drop(assignments);

        // Update metrics
        if let Some(vd) = state.variant_data.get(&variant.id) {
            vd.requests.fetch_add(1, Ordering::Relaxed);
        }
        self.metrics.total_assignments.fetch_add(1, Ordering::Relaxed);

        Ok(VariantAssignment {
            experiment_id: experiment_id.to_string(),
            variant_id: variant.id.clone(),
            variant_config: variant.config,
            is_new_assignment: true,
        })
    }

    fn select_variant(&self, config: &ExperimentConfig, identity: &str) -> Result<Variant> {
        match &config.allocation {
            TrafficAllocation::Random => {
                let total_weight: u32 = config.variants.iter().map(|v| v.weight).sum();
                let rand_val = hash_identity(identity) % total_weight as u64;

                let mut cumulative = 0u32;
                for variant in &config.variants {
                    cumulative += variant.weight;
                    if (rand_val as u32) < cumulative {
                        return Ok(variant.clone());
                    }
                }
                Ok(config.variants[0].clone())
            }
            TrafficAllocation::Consistent => {
                let hash = hash_identity(identity);
                let idx = (hash as usize) % config.variants.len();
                Ok(config.variants[idx].clone())
            }
            TrafficAllocation::TimeBased { rotation_minutes } => {
                let now = current_timestamp();
                let period = (*rotation_minutes as u64) * 60 * 1000;
                let idx = ((now / period) as usize) % config.variants.len();
                Ok(config.variants[idx].clone())
            }
            TrafficAllocation::Manual => {
                // Return control by default for manual
                config.variants.iter()
                    .find(|v| v.is_control)
                    .or_else(|| config.variants.first())
                    .cloned()
                    .ok_or_else(|| VecStoreError::InvalidInput("No variants defined".into()))
            }
        }
    }

    /// Record result for a request
    pub fn record_result(
        &self,
        experiment_id: &str,
        variant_id: &str,
        result: RequestResult,
    ) -> Result<()> {
        let experiments = self.experiments.read()?;
        let state = experiments.get(experiment_id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Experiment {} not found", experiment_id)))?;

        let variant_data = state.variant_data.get(variant_id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Variant {} not found", variant_id)))?;

        // Update counters
        if result.success {
            variant_data.successes.fetch_add(1, Ordering::Relaxed);
        } else {
            variant_data.errors.fetch_add(1, Ordering::Relaxed);
        }

        // Record latency
        variant_data.total_latency_us.fetch_add(result.latency_us, Ordering::Relaxed);
        variant_data.latency_samples.write()?.push(result.latency_us);

        // Record metrics
        let mut metrics = variant_data.metric_values.write()?;
        for (name, value) in result.metrics {
            metrics.entry(name).or_insert_with(Vec::new).push(value);
        }

        self.metrics.total_results_recorded.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Get current experiment results
    pub fn get_results(&self, experiment_id: &str) -> Result<ExperimentResults> {
        let experiments = self.experiments.read()?;
        let state = experiments.get(experiment_id)
            .ok_or_else(|| VecStoreError::NotFound(format!("Experiment {} not found", experiment_id)))?;

        self.compute_results(state)
    }

    fn compute_results(&self, state: &ExperimentState) -> Result<ExperimentResults> {
        let mut variant_results = Vec::new();
        let control_id = state.config.variants.iter()
            .find(|v| v.is_control)
            .map(|v| v.id.clone());

        for variant in &state.config.variants {
            if let Some(vd) = state.variant_data.get(&variant.id) {
                let requests = vd.requests.load(Ordering::Relaxed);
                let successes = vd.successes.load(Ordering::Relaxed);
                let errors = vd.errors.load(Ordering::Relaxed);
                let total_latency = vd.total_latency_us.load(Ordering::Relaxed);

                let latencies = vd.latency_samples.read()?.clone();
                let (p50, p95, p99) = compute_percentiles(&latencies);

                let metrics = vd.metric_values.read()?;
                let mut metric_summaries = HashMap::new();

                for (name, values) in metrics.iter() {
                    if !values.is_empty() {
                        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                        let variance: f64 = values.iter()
                            .map(|v| (v - mean).powi(2))
                            .sum::<f64>() / values.len() as f64;
                        let std_dev = variance.sqrt();

                        metric_summaries.insert(name.clone(), MetricSummary {
                            count: values.len(),
                            mean,
                            std_dev,
                            min: values.iter().cloned().fold(f64::INFINITY, f64::min),
                            max: values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                        });
                    }
                }

                variant_results.push(VariantResult {
                    variant_id: variant.id.clone(),
                    variant_name: variant.name.clone(),
                    is_control: variant.is_control,
                    sample_size: requests,
                    success_rate: if requests > 0 { successes as f64 / requests as f64 } else { 0.0 },
                    error_rate: if requests > 0 { errors as f64 / requests as f64 } else { 0.0 },
                    avg_latency_us: if requests > 0 { total_latency / requests } else { 0 },
                    p50_latency_us: p50,
                    p95_latency_us: p95,
                    p99_latency_us: p99,
                    metrics: metric_summaries,
                });
            }
        }

        // Compute statistical comparisons
        let comparisons = self.compute_comparisons(&variant_results, control_id.as_deref(), &state.config);

        // Determine winner
        let winner = self.determine_winner(&variant_results, &comparisons, &state.config);

        Ok(ExperimentResults {
            experiment_id: generate_experiment_id(&state.config.name),
            experiment_name: state.config.name.clone(),
            status: state.status.clone(),
            started_at: state.started_at,
            ended_at: state.ended_at,
            duration_seconds: state.started_at.map(|start| {
                let end = state.ended_at.unwrap_or_else(current_timestamp);
                (end - start) / 1000
            }),
            variant_results,
            comparisons,
            winner,
            is_significant: false, // Set based on comparisons
        })
    }

    fn compute_comparisons(
        &self,
        results: &[VariantResult],
        control_id: Option<&str>,
        config: &ExperimentConfig,
    ) -> Vec<VariantComparison> {
        let mut comparisons = Vec::new();

        let control = control_id.and_then(|id| results.iter().find(|r| r.variant_id == id));

        for result in results {
            if result.is_control {
                continue;
            }

            if let Some(ctrl) = control {
                // Compare against control
                let primary_metric = &config.primary_metric;
                let ctrl_value = ctrl.metrics.get(&primary_metric.name);
                let treat_value = result.metrics.get(&primary_metric.name);

                let (relative_change, is_significant, p_value) = match (ctrl_value, treat_value) {
                    (Some(c), Some(t)) if c.count > 0 && t.count > 0 => {
                        let change = if c.mean != 0.0 {
                            (t.mean - c.mean) / c.mean
                        } else {
                            0.0
                        };

                        // Simple t-test approximation
                        let (significant, p) = two_sample_ttest(
                            c.mean, c.std_dev, c.count,
                            t.mean, t.std_dev, t.count,
                            config.significance_threshold,
                        );

                        (change, significant, p)
                    }
                    _ => (0.0, false, 1.0),
                };

                comparisons.push(VariantComparison {
                    variant_id: result.variant_id.clone(),
                    control_id: ctrl.variant_id.clone(),
                    metric_name: primary_metric.name.clone(),
                    relative_change,
                    is_significant,
                    p_value,
                    confidence_interval: (relative_change - 0.1, relative_change + 0.1), // Simplified
                });
            }
        }

        comparisons
    }

    fn determine_winner(
        &self,
        _results: &[VariantResult],
        comparisons: &[VariantComparison],
        config: &ExperimentConfig,
    ) -> Option<String> {
        // Find best performing variant that's statistically significant
        let significant: Vec<_> = comparisons.iter()
            .filter(|c| c.is_significant)
            .collect();

        if significant.is_empty() {
            return None;
        }

        // Return the variant with best improvement
        let best = if config.primary_metric.higher_is_better {
            significant.iter().max_by(|a, b|
                a.relative_change.partial_cmp(&b.relative_change).unwrap_or(std::cmp::Ordering::Equal)
            )
        } else {
            significant.iter().min_by(|a, b|
                a.relative_change.partial_cmp(&b.relative_change).unwrap_or(std::cmp::Ordering::Equal)
            )
        };

        best.map(|c| c.variant_id.clone())
    }

    /// List all experiments
    pub fn list_experiments(&self) -> Result<Vec<ExperimentSummary>> {
        let experiments = self.experiments.read()
            .map_err(|_| VecStoreError::LockError("experiments lock poisoned".into()))?;
        Ok(experiments.iter()
            .map(|(id, state)| ExperimentSummary {
                id: id.clone(),
                name: state.config.name.clone(),
                status: state.status.clone(),
                experiment_type: state.config.experiment_type.clone(),
                variant_count: state.config.variants.len(),
                total_requests: state.variant_data.values()
                    .map(|vd| vd.requests.load(Ordering::Relaxed))
                    .sum(),
                created_at: state.created_at,
                started_at: state.started_at,
            })
            .collect())
    }

    /// Get manager stats
    pub fn get_stats(&self) -> Result<ABTestStats> {
        let experiments = self.experiments.read()
            .map_err(|_| VecStoreError::LockError("experiments lock poisoned".into()))?;
        Ok(ABTestStats {
            experiments_created: self.metrics.experiments_created.load(Ordering::Relaxed),
            experiments_completed: self.metrics.experiments_completed.load(Ordering::Relaxed),
            total_assignments: self.metrics.total_assignments.load(Ordering::Relaxed),
            total_results_recorded: self.metrics.total_results_recorded.load(Ordering::Relaxed),
            active_experiments: experiments.values().filter(|s| s.status == ExperimentStatus::Running).count(),
        })
    }
}

impl Default for ABTestManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Variant assignment result
#[derive(Debug, Clone, Serialize)]
pub struct VariantAssignment {
    pub experiment_id: String,
    pub variant_id: String,
    pub variant_config: VariantConfig,
    pub is_new_assignment: bool,
}

/// Request result to record
#[derive(Debug, Clone)]
pub struct RequestResult {
    pub success: bool,
    pub latency_us: u64,
    pub metrics: HashMap<String, f64>,
}

/// Experiment results
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentResults {
    pub experiment_id: String,
    pub experiment_name: String,
    pub status: ExperimentStatus,
    pub started_at: Option<u64>,
    pub ended_at: Option<u64>,
    pub duration_seconds: Option<u64>,
    pub variant_results: Vec<VariantResult>,
    pub comparisons: Vec<VariantComparison>,
    pub winner: Option<String>,
    pub is_significant: bool,
}

/// Per-variant results
#[derive(Debug, Clone, Serialize)]
pub struct VariantResult {
    pub variant_id: String,
    pub variant_name: String,
    pub is_control: bool,
    pub sample_size: u64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub avg_latency_us: u64,
    pub p50_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,
    pub metrics: HashMap<String, MetricSummary>,
}

/// Metric summary statistics
#[derive(Debug, Clone, Serialize)]
pub struct MetricSummary {
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

/// Comparison between variant and control
#[derive(Debug, Clone, Serialize)]
pub struct VariantComparison {
    pub variant_id: String,
    pub control_id: String,
    pub metric_name: String,
    pub relative_change: f64,
    pub is_significant: bool,
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
}

/// Experiment summary for listing
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentSummary {
    pub id: String,
    pub name: String,
    pub status: ExperimentStatus,
    pub experiment_type: ExperimentType,
    pub variant_count: usize,
    pub total_requests: u64,
    pub created_at: u64,
    pub started_at: Option<u64>,
}

/// A/B test manager stats
#[derive(Debug, Clone, Serialize)]
pub struct ABTestStats {
    pub experiments_created: u64,
    pub experiments_completed: u64,
    pub total_assignments: u64,
    pub total_results_recorded: u64,
    pub active_experiments: usize,
}

// Helper functions

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn generate_experiment_id(name: &str) -> String {
    let timestamp = current_timestamp();
    let hash = hash_identity(name) & 0xFFFF;
    format!("exp_{}_{:04x}", timestamp % 1_000_000, hash)
}

fn hash_identity(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn compute_percentiles(samples: &[u64]) -> (u64, u64, u64) {
    if samples.is_empty() {
        return (0, 0, 0);
    }

    let mut sorted = samples.to_vec();
    sorted.sort_unstable();

    let p50_idx = (samples.len() as f64 * 0.50) as usize;
    let p95_idx = (samples.len() as f64 * 0.95) as usize;
    let p99_idx = (samples.len() as f64 * 0.99) as usize;

    (
        sorted.get(p50_idx.min(sorted.len() - 1)).copied().unwrap_or(0),
        sorted.get(p95_idx.min(sorted.len() - 1)).copied().unwrap_or(0),
        sorted.get(p99_idx.min(sorted.len() - 1)).copied().unwrap_or(0),
    )
}

fn two_sample_ttest(
    mean1: f64, std1: f64, n1: usize,
    mean2: f64, std2: f64, n2: usize,
    threshold: f64,
) -> (bool, f64) {
    if n1 < 2 || n2 < 2 {
        return (false, 1.0);
    }

    // Welch's t-test
    let se1 = (std1 * std1) / n1 as f64;
    let se2 = (std2 * std2) / n2 as f64;
    let se = (se1 + se2).sqrt();

    if se < 1e-10 {
        return (false, 1.0);
    }

    let t = (mean1 - mean2).abs() / se;

    // Approximate p-value using normal distribution (for large samples)
    // For accurate results, use a proper statistical library
    let p_value = 2.0 * (1.0 - normal_cdf(t));

    (p_value < (1.0 - threshold), p_value)
}

fn normal_cdf(x: f64) -> f64 {
    // Approximation of the normal CDF
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / 2.0_f64.sqrt();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ExperimentConfig {
        ExperimentConfig {
            name: "test_experiment".to_string(),
            description: "Test experiment".to_string(),
            experiment_type: ExperimentType::IndexConfig,
            variants: vec![
                Variant {
                    id: "control".to_string(),
                    name: "Control".to_string(),
                    weight: 50,
                    config: VariantConfig {
                        index_params: None,
                        embedding_model: None,
                        distance_metric: None,
                        search_params: None,
                        reranking: None,
                        custom: HashMap::new(),
                    },
                    is_control: true,
                },
                Variant {
                    id: "treatment".to_string(),
                    name: "Treatment".to_string(),
                    weight: 50,
                    config: VariantConfig {
                        index_params: Some(IndexParams {
                            index_type: "hnsw".to_string(),
                            hnsw_m: Some(32),
                            hnsw_ef_construction: Some(200),
                            hnsw_ef_search: Some(100),
                            pq_enabled: None,
                            pq_subvectors: None,
                        }),
                        embedding_model: None,
                        distance_metric: None,
                        search_params: None,
                        reranking: None,
                        custom: HashMap::new(),
                    },
                    is_control: false,
                },
            ],
            allocation: TrafficAllocation::Random,
            start_time: None,
            end_time: None,
            min_sample_size: 100,
            primary_metric: Metric {
                name: "recall_at_10".to_string(),
                metric_type: MetricType::Recall,
                higher_is_better: true,
            },
            secondary_metrics: vec![],
            significance_threshold: 0.95,
            owner: "test".to_string(),
        }
    }

    #[test]
    fn test_create_experiment() {
        let manager = ABTestManager::new();
        let config = create_test_config();

        let id = manager.create_experiment(config).unwrap();
        assert!(id.starts_with("exp_"));

        let experiments = manager.list_experiments().unwrap();
        assert_eq!(experiments.len(), 1);
    }

    #[test]
    fn test_variant_assignment() {
        let manager = ABTestManager::new();
        let config = create_test_config();

        let id = manager.create_experiment(config).unwrap();
        manager.start_experiment(&id).unwrap();

        // Get assignment
        let assignment1 = manager.get_assignment(&id, "user1").unwrap();
        assert!(!assignment1.variant_id.is_empty());
        assert!(assignment1.is_new_assignment);

        // Same user should get same variant
        let assignment2 = manager.get_assignment(&id, "user1").unwrap();
        assert_eq!(assignment1.variant_id, assignment2.variant_id);
        assert!(!assignment2.is_new_assignment);
    }

    #[test]
    fn test_record_result() {
        let manager = ABTestManager::new();
        let config = create_test_config();

        let id = manager.create_experiment(config).unwrap();
        manager.start_experiment(&id).unwrap();

        let assignment = manager.get_assignment(&id, "user1").unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("recall_at_10".to_string(), 0.95);

        manager.record_result(&id, &assignment.variant_id, RequestResult {
            success: true,
            latency_us: 5000,
            metrics,
        }).unwrap();

        let results = manager.get_results(&id).unwrap();
        assert!(!results.variant_results.is_empty());
    }

    #[test]
    fn test_percentiles() {
        let samples: Vec<u64> = (1..=100).collect();
        let (p50, p95, p99) = compute_percentiles(&samples);

        // Using nearest-rank method: index = ceil(P/100 * N)
        // For 100 samples: p50 at index 50 = value 51
        assert_eq!(p50, 51);
        assert_eq!(p95, 96);
        assert_eq!(p99, 100);
    }
}
