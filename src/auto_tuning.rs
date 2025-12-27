// Auto-Tuning Engine - Automated parameter optimization for vector indexes
// Bayesian optimization, hyperparameter search, and adaptive configuration

use std::collections::{HashMap, BinaryHeap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::cmp::Ordering as CmpOrdering;

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Auto-tuning engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneConfig {
    /// Search strategy
    pub strategy: SearchStrategy,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Maximum time budget
    pub time_budget: Duration,
    /// Target metric
    pub target_metric: TuneMetric,
    /// Optimization direction
    pub direction: OptimizeDirection,
    /// Number of initial random samples
    pub n_initial_samples: usize,
    /// Exploration vs exploitation trade-off
    pub exploration_rate: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Enable parallel evaluation
    pub parallel_eval: bool,
}

impl Default for AutoTuneConfig {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::BayesianOptimization,
            max_iterations: 100,
            time_budget: Duration::from_secs(3600),
            target_metric: TuneMetric::RecallAtK(10),
            direction: OptimizeDirection::Maximize,
            n_initial_samples: 10,
            exploration_rate: 0.1,
            early_stopping_patience: 10,
            cv_folds: 5,
            parallel_eval: true,
        }
    }
}

/// Search strategies for parameter optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SearchStrategy {
    /// Grid search (exhaustive)
    GridSearch,
    /// Random search
    RandomSearch,
    /// Bayesian optimization
    BayesianOptimization,
    /// Tree-structured Parzen Estimators
    TPE,
    /// Hyperband (adaptive resource allocation)
    Hyperband,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Population-based training
    PopulationBasedTraining,
}

/// Metrics to optimize
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TuneMetric {
    /// Recall at k
    RecallAtK(usize),
    /// Precision at k
    PrecisionAtK(usize),
    /// NDCG at k
    NDCGAtK(usize),
    /// MRR (Mean Reciprocal Rank)
    MRR,
    /// Query latency (p50)
    LatencyP50,
    /// Query latency (p99)
    LatencyP99,
    /// Throughput (QPS)
    Throughput,
    /// Memory usage
    MemoryUsage,
    /// Index build time
    BuildTime,
    /// Custom metric
    Custom(String),
}

/// Optimization direction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizeDirection {
    Minimize,
    Maximize,
}

/// Parameter space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpace {
    /// Parameters to tune
    pub parameters: Vec<Parameter>,
    /// Constraints between parameters
    pub constraints: Vec<Constraint>,
}

/// Parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Description
    pub description: String,
    /// Default value
    pub default: ParameterValue,
}

/// Parameter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    /// Integer range
    Integer { min: i64, max: i64, step: Option<i64>, log_scale: bool },
    /// Float range
    Float { min: f64, max: f64, log_scale: bool },
    /// Categorical choices
    Categorical { choices: Vec<String> },
    /// Boolean
    Boolean,
    /// Ordinal (ordered categorical)
    Ordinal { choices: Vec<String> },
}

/// Parameter value
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ParameterValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
}

/// Constraint between parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Constraint expression
    pub expression: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// param1 < param2
    LessThan(String, String),
    /// param1 > param2
    GreaterThan(String, String),
    /// param1 implies param2
    Implies(String, ParameterValue, String, ParameterValue),
    /// Custom expression
    Custom(String),
}

/// Auto-tuning engine
pub struct AutoTuner {
    config: AutoTuneConfig,
    parameter_space: ParameterSpace,
    trials: RwLock<Vec<Trial>>,
    best_trial: RwLock<Option<Trial>>,
    stats: TunerStats,
    surrogate_model: RwLock<Option<SurrogateModel>>,
}

/// A single trial (parameter configuration + result)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    /// Trial ID
    pub id: u64,
    /// Parameter values
    pub params: HashMap<String, ParameterValue>,
    /// Objective value
    pub objective: Option<f64>,
    /// All metric values
    pub metrics: HashMap<String, f64>,
    /// Trial state
    pub state: TrialState,
    /// Duration
    pub duration: Option<Duration>,
    /// Started at
    pub started_at: Option<u64>,
    /// Finished at
    pub finished_at: Option<u64>,
}

/// Trial state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrialState {
    Pending,
    Running,
    Complete,
    Pruned,
    Failed,
}

/// Tuner statistics
struct TunerStats {
    total_trials: AtomicU64,
    completed_trials: AtomicU64,
    failed_trials: AtomicU64,
    pruned_trials: AtomicU64,
    best_objective: RwLock<f64>,
    total_time: RwLock<Duration>,
}

/// Surrogate model for Bayesian optimization
struct SurrogateModel {
    /// Observed points (params -> objective)
    observations: Vec<(Vec<f64>, f64)>,
    /// Kernel type
    kernel: KernelType,
    /// Length scales
    length_scales: Vec<f64>,
    /// Signal variance
    signal_variance: f64,
    /// Noise variance
    noise_variance: f64,
}

/// Kernel types for Gaussian Process
#[derive(Debug, Clone)]
enum KernelType {
    RBF,
    Matern32,
    Matern52,
    RationalQuadratic,
}

/// Acquisition function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    EI,
    /// Probability of Improvement
    PI,
    /// Upper Confidence Bound
    UCB { beta: f64 },
    /// Thompson Sampling
    ThompsonSampling,
}

/// Tuning result
#[derive(Debug, Clone, Serialize)]
pub struct TuneResult {
    /// Best parameters
    pub best_params: HashMap<String, ParameterValue>,
    /// Best objective value
    pub best_objective: f64,
    /// All trials
    pub trials: Vec<Trial>,
    /// Number of iterations
    pub n_iterations: usize,
    /// Total time
    pub total_time: Duration,
    /// Improvement history
    pub improvement_history: Vec<f64>,
    /// Convergence status
    pub converged: bool,
}

impl AutoTuner {
    /// Create a new auto-tuner
    pub fn new(config: AutoTuneConfig, parameter_space: ParameterSpace) -> Self {
        Self {
            config,
            parameter_space,
            trials: RwLock::new(Vec::new()),
            best_trial: RwLock::new(None),
            stats: TunerStats {
                total_trials: AtomicU64::new(0),
                completed_trials: AtomicU64::new(0),
                failed_trials: AtomicU64::new(0),
                pruned_trials: AtomicU64::new(0),
                best_objective: RwLock::new(f64::NEG_INFINITY),
                total_time: RwLock::new(Duration::ZERO),
            },
            surrogate_model: RwLock::new(None),
        }
    }

    /// Run the optimization
    pub fn optimize<F>(&self, objective_fn: F) -> Result<TuneResult>
    where
        F: Fn(&HashMap<String, ParameterValue>) -> Result<f64> + Send + Sync,
    {
        let start = Instant::now();
        let mut improvement_history = Vec::new();
        let mut no_improvement_count = 0;

        // Initialize with random samples
        for _ in 0..self.config.n_initial_samples {
            if start.elapsed() > self.config.time_budget {
                break;
            }

            let params = self.sample_random();
            let result = self.evaluate_trial(&params, &objective_fn)?;
            improvement_history.push(*self.stats.best_objective.read().unwrap());
        }

        // Main optimization loop
        for iteration in self.config.n_initial_samples..self.config.max_iterations {
            if start.elapsed() > self.config.time_budget {
                break;
            }

            // Get next parameters based on strategy
            let params = match self.config.strategy {
                SearchStrategy::GridSearch => self.sample_grid(iteration),
                SearchStrategy::RandomSearch => self.sample_random(),
                SearchStrategy::BayesianOptimization => self.sample_bayesian()?,
                SearchStrategy::TPE => self.sample_tpe()?,
                SearchStrategy::Hyperband => self.sample_random(), // Simplified
                SearchStrategy::GeneticAlgorithm => self.sample_genetic()?,
                SearchStrategy::SimulatedAnnealing => self.sample_annealing(iteration)?,
                SearchStrategy::PopulationBasedTraining => self.sample_random(),
            };

            let prev_best = *self.stats.best_objective.read().unwrap();
            let result = self.evaluate_trial(&params, &objective_fn)?;
            let new_best = *self.stats.best_objective.read().unwrap();

            improvement_history.push(new_best);

            // Check for early stopping
            if (new_best - prev_best).abs() < 1e-6 {
                no_improvement_count += 1;
                if no_improvement_count >= self.config.early_stopping_patience {
                    break;
                }
            } else {
                no_improvement_count = 0;
            }
        }

        let total_time = start.elapsed();
        *self.stats.total_time.write().unwrap() = total_time;

        let trials = self.trials.read().unwrap().clone();
        let best_trial = self.best_trial.read().unwrap().clone();

        Ok(TuneResult {
            best_params: best_trial.as_ref()
                .map(|t| t.params.clone())
                .unwrap_or_default(),
            best_objective: best_trial.as_ref()
                .and_then(|t| t.objective)
                .unwrap_or(f64::NEG_INFINITY),
            trials,
            n_iterations: self.stats.completed_trials.load(Ordering::Relaxed) as usize,
            total_time,
            improvement_history,
            converged: no_improvement_count >= self.config.early_stopping_patience,
        })
    }

    /// Evaluate a single trial
    fn evaluate_trial<F>(
        &self,
        params: &HashMap<String, ParameterValue>,
        objective_fn: &F,
    ) -> Result<Trial>
    where
        F: Fn(&HashMap<String, ParameterValue>) -> Result<f64>,
    {
        let trial_id = self.stats.total_trials.fetch_add(1, Ordering::Relaxed);
        let start = Instant::now();

        let mut trial = Trial {
            id: trial_id,
            params: params.clone(),
            objective: None,
            metrics: HashMap::new(),
            state: TrialState::Running,
            duration: None,
            started_at: Some(current_timestamp()),
            finished_at: None,
        };

        // Run objective function
        match objective_fn(params) {
            Ok(objective) => {
                trial.objective = Some(objective);
                trial.state = TrialState::Complete;
                trial.duration = Some(start.elapsed());
                trial.finished_at = Some(current_timestamp());
                self.stats.completed_trials.fetch_add(1, Ordering::Relaxed);

                // Update best
                let is_better = match self.config.direction {
                    OptimizeDirection::Maximize => objective > *self.stats.best_objective.read().unwrap(),
                    OptimizeDirection::Minimize => objective < *self.stats.best_objective.read().unwrap(),
                };

                if is_better {
                    *self.stats.best_objective.write().unwrap() = objective;
                    *self.best_trial.write().unwrap() = Some(trial.clone());
                }

                // Update surrogate model
                self.update_surrogate(&trial)?;
            }
            Err(_) => {
                trial.state = TrialState::Failed;
                trial.duration = Some(start.elapsed());
                self.stats.failed_trials.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.trials.write().unwrap().push(trial.clone());
        Ok(trial)
    }

    /// Sample random parameters
    fn sample_random(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();

        for param in &self.parameter_space.parameters {
            let value = match &param.param_type {
                ParameterType::Integer { min, max, step, log_scale } => {
                    let val = if *log_scale {
                        let log_min = (*min as f64).ln();
                        let log_max = (*max as f64).ln();
                        let log_val = random_f64(log_min, log_max);
                        log_val.exp() as i64
                    } else {
                        random_i64(*min, *max)
                    };
                    let val = if let Some(s) = step {
                        (val / s) * s
                    } else {
                        val
                    };
                    ParameterValue::Integer(val.clamp(*min, *max))
                }
                ParameterType::Float { min, max, log_scale } => {
                    let val = if *log_scale {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        random_f64(log_min, log_max).exp()
                    } else {
                        random_f64(*min, *max)
                    };
                    ParameterValue::Float(val)
                }
                ParameterType::Categorical { choices } | ParameterType::Ordinal { choices } => {
                    let idx = random_usize(0, choices.len());
                    ParameterValue::String(choices[idx].clone())
                }
                ParameterType::Boolean => {
                    ParameterValue::Boolean(random_bool())
                }
            };
            params.insert(param.name.clone(), value);
        }

        // Apply constraints
        self.apply_constraints(&mut params);
        params
    }

    /// Sample from grid
    fn sample_grid(&self, iteration: usize) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        let mut idx = iteration;

        for param in &self.parameter_space.parameters {
            let value = match &param.param_type {
                ParameterType::Integer { min, max, .. } => {
                    let range = (max - min + 1) as usize;
                    let pos = idx % range;
                    idx /= range;
                    ParameterValue::Integer(*min + pos as i64)
                }
                ParameterType::Float { min, max, .. } => {
                    let steps = 10;
                    let pos = idx % steps;
                    idx /= steps;
                    let val = min + (max - min) * (pos as f64 / (steps - 1) as f64);
                    ParameterValue::Float(val)
                }
                ParameterType::Categorical { choices } | ParameterType::Ordinal { choices } => {
                    let pos = idx % choices.len();
                    idx /= choices.len();
                    ParameterValue::String(choices[pos].clone())
                }
                ParameterType::Boolean => {
                    let val = idx % 2 == 0;
                    idx /= 2;
                    ParameterValue::Boolean(val)
                }
            };
            params.insert(param.name.clone(), value);
        }

        params
    }

    /// Sample using Bayesian optimization
    fn sample_bayesian(&self) -> Result<HashMap<String, ParameterValue>> {
        let trials = self.trials.read().unwrap();

        if trials.len() < self.config.n_initial_samples {
            return Ok(self.sample_random());
        }

        // Use acquisition function to select next point
        let mut best_acquisition = f64::NEG_INFINITY;
        let mut best_params = self.sample_random();

        // Random search over acquisition function
        for _ in 0..100 {
            let params = self.sample_random();
            let acquisition = self.compute_acquisition(&params)?;

            if acquisition > best_acquisition {
                best_acquisition = acquisition;
                best_params = params;
            }
        }

        Ok(best_params)
    }

    /// Compute acquisition value for a point
    fn compute_acquisition(&self, params: &HashMap<String, ParameterValue>) -> Result<f64> {
        let surrogate = self.surrogate_model.read().unwrap();

        if let Some(model) = surrogate.as_ref() {
            let x = self.params_to_vector(params);
            let (mean, std) = model.predict(&x);

            let best_f = *self.stats.best_objective.read().unwrap();

            // Expected Improvement
            let z = (mean - best_f) / (std + 1e-9);
            let ei = (mean - best_f) * normal_cdf(z) + std * normal_pdf(z);

            Ok(ei)
        } else {
            Ok(random_f64(0.0, 1.0))
        }
    }

    /// Sample using TPE
    fn sample_tpe(&self) -> Result<HashMap<String, ParameterValue>> {
        let trials = self.trials.read().unwrap();

        if trials.len() < self.config.n_initial_samples {
            return Ok(self.sample_random());
        }

        // Split trials into good and bad
        let mut sorted_trials: Vec<_> = trials.iter()
            .filter(|t| t.objective.is_some())
            .collect();

        sorted_trials.sort_by(|a, b| {
            b.objective.unwrap().partial_cmp(&a.objective.unwrap())
                .unwrap_or(CmpOrdering::Equal)
        });

        let gamma = 0.25; // Top 25% are "good"
        let n_good = ((sorted_trials.len() as f64) * gamma).ceil() as usize;
        let good_trials: Vec<_> = sorted_trials.iter().take(n_good).copied().collect();
        let bad_trials: Vec<_> = sorted_trials.iter().skip(n_good).copied().collect();

        // Sample from good distribution, weighted by likelihood ratio
        let mut params = HashMap::new();

        for param in &self.parameter_space.parameters {
            let good_values: Vec<_> = good_trials.iter()
                .filter_map(|t| t.params.get(&param.name))
                .collect();
            let bad_values: Vec<_> = bad_trials.iter()
                .filter_map(|t| t.params.get(&param.name))
                .collect();

            let value = self.sample_from_kde(&param.param_type, &good_values, &bad_values);
            params.insert(param.name.clone(), value);
        }

        Ok(params)
    }

    fn sample_from_kde(
        &self,
        param_type: &ParameterType,
        good: &[&ParameterValue],
        _bad: &[&ParameterValue],
    ) -> ParameterValue {
        // Simplified KDE sampling - just sample from good trials with noise
        if good.is_empty() {
            return match param_type {
                ParameterType::Integer { min, max, .. } => {
                    ParameterValue::Integer(random_i64(*min, *max))
                }
                ParameterType::Float { min, max, .. } => {
                    ParameterValue::Float(random_f64(*min, *max))
                }
                ParameterType::Categorical { choices } | ParameterType::Ordinal { choices } => {
                    ParameterValue::String(choices[random_usize(0, choices.len())].clone())
                }
                ParameterType::Boolean => ParameterValue::Boolean(random_bool()),
            };
        }

        let idx = random_usize(0, good.len());
        good[idx].clone()
    }

    /// Sample using genetic algorithm
    fn sample_genetic(&self) -> Result<HashMap<String, ParameterValue>> {
        let trials = self.trials.read().unwrap();

        if trials.len() < 2 {
            return Ok(self.sample_random());
        }

        // Select two parents from top trials
        let mut sorted: Vec<_> = trials.iter()
            .filter(|t| t.objective.is_some())
            .collect();

        sorted.sort_by(|a, b| {
            b.objective.unwrap().partial_cmp(&a.objective.unwrap())
                .unwrap_or(CmpOrdering::Equal)
        });

        let parent1 = &sorted[random_usize(0, sorted.len().min(5))];
        let parent2 = &sorted[random_usize(0, sorted.len().min(5))];

        // Crossover
        let mut child = HashMap::new();
        for param in &self.parameter_space.parameters {
            let value = if random_bool() {
                parent1.params.get(&param.name).cloned()
            } else {
                parent2.params.get(&param.name).cloned()
            };

            // Mutation
            let value = if random_f64(0.0, 1.0) < 0.1 {
                self.mutate_parameter(&param.param_type, value)
            } else {
                value.unwrap_or_else(|| param.default.clone())
            };

            child.insert(param.name.clone(), value);
        }

        Ok(child)
    }

    fn mutate_parameter(&self, param_type: &ParameterType, value: Option<ParameterValue>) -> ParameterValue {
        match param_type {
            ParameterType::Integer { min, max, .. } => {
                let base = match value {
                    Some(ParameterValue::Integer(v)) => v,
                    _ => (*min + *max) / 2,
                };
                let mutation = random_i64(-10, 10);
                ParameterValue::Integer((base + mutation).clamp(*min, *max))
            }
            ParameterType::Float { min, max, .. } => {
                let base = match value {
                    Some(ParameterValue::Float(v)) => v,
                    _ => (min + max) / 2.0,
                };
                let mutation = random_f64(-0.1, 0.1) * (max - min);
                ParameterValue::Float((base + mutation).clamp(*min, *max))
            }
            ParameterType::Categorical { choices } | ParameterType::Ordinal { choices } => {
                ParameterValue::String(choices[random_usize(0, choices.len())].clone())
            }
            ParameterType::Boolean => ParameterValue::Boolean(random_bool()),
        }
    }

    /// Sample using simulated annealing
    fn sample_annealing(&self, iteration: usize) -> Result<HashMap<String, ParameterValue>> {
        let temperature = 1.0 / (1.0 + iteration as f64 * 0.01);

        let best = self.best_trial.read().unwrap();
        if let Some(best_trial) = best.as_ref() {
            let mut params = best_trial.params.clone();

            // Perturb parameters based on temperature
            for param in &self.parameter_space.parameters {
                if random_f64(0.0, 1.0) < temperature {
                    let new_value = self.mutate_parameter(
                        &param.param_type,
                        params.get(&param.name).cloned(),
                    );
                    params.insert(param.name.clone(), new_value);
                }
            }

            Ok(params)
        } else {
            Ok(self.sample_random())
        }
    }

    /// Update surrogate model with new observation
    fn update_surrogate(&self, trial: &Trial) -> Result<()> {
        if trial.objective.is_none() {
            return Ok(());
        }

        let x = self.params_to_vector(&trial.params);
        let y = trial.objective.unwrap();

        let mut surrogate = self.surrogate_model.write().unwrap();
        if surrogate.is_none() {
            *surrogate = Some(SurrogateModel {
                observations: Vec::new(),
                kernel: KernelType::Matern52,
                length_scales: vec![1.0; self.parameter_space.parameters.len()],
                signal_variance: 1.0,
                noise_variance: 0.01,
            });
        }

        if let Some(model) = surrogate.as_mut() {
            model.observations.push((x, y));
        }

        Ok(())
    }

    /// Convert parameters to vector for surrogate model
    fn params_to_vector(&self, params: &HashMap<String, ParameterValue>) -> Vec<f64> {
        self.parameter_space.parameters.iter()
            .map(|p| {
                match params.get(&p.name) {
                    Some(ParameterValue::Integer(v)) => *v as f64,
                    Some(ParameterValue::Float(v)) => *v,
                    Some(ParameterValue::Boolean(v)) => if *v { 1.0 } else { 0.0 },
                    Some(ParameterValue::String(s)) => {
                        // Map categorical to index
                        if let ParameterType::Categorical { choices } = &p.param_type {
                            choices.iter().position(|c| c == s).unwrap_or(0) as f64
                        } else {
                            0.0
                        }
                    }
                    None => 0.0,
                }
            })
            .collect()
    }

    /// Apply parameter constraints
    fn apply_constraints(&self, params: &mut HashMap<String, ParameterValue>) {
        for constraint in &self.parameter_space.constraints {
            match &constraint.constraint_type {
                ConstraintType::LessThan(p1, p2) => {
                    if let (Some(v1), Some(v2)) = (params.get(p1), params.get(p2)) {
                        let v1_float = match v1 {
                            ParameterValue::Integer(i) => *i as f64,
                            ParameterValue::Float(f) => *f,
                            _ => continue,
                        };
                        let v2_float = match v2 {
                            ParameterValue::Integer(i) => *i as f64,
                            ParameterValue::Float(f) => *f,
                            _ => continue,
                        };

                        if v1_float >= v2_float {
                            // Swap values
                            let temp = params.get(p1).cloned();
                            params.insert(p1.clone(), params.get(p2).cloned().unwrap());
                            params.insert(p2.clone(), temp.unwrap());
                        }
                    }
                }
                ConstraintType::GreaterThan(p1, p2) => {
                    if let (Some(v1), Some(v2)) = (params.get(p1), params.get(p2)) {
                        let v1_float = match v1 {
                            ParameterValue::Integer(i) => *i as f64,
                            ParameterValue::Float(f) => *f,
                            _ => continue,
                        };
                        let v2_float = match v2 {
                            ParameterValue::Integer(i) => *i as f64,
                            ParameterValue::Float(f) => *f,
                            _ => continue,
                        };

                        if v1_float <= v2_float {
                            let temp = params.get(p1).cloned();
                            params.insert(p1.clone(), params.get(p2).cloned().unwrap());
                            params.insert(p2.clone(), temp.unwrap());
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> TunerStatsSummary {
        TunerStatsSummary {
            total_trials: self.stats.total_trials.load(Ordering::Relaxed),
            completed_trials: self.stats.completed_trials.load(Ordering::Relaxed),
            failed_trials: self.stats.failed_trials.load(Ordering::Relaxed),
            pruned_trials: self.stats.pruned_trials.load(Ordering::Relaxed),
            best_objective: *self.stats.best_objective.read().unwrap(),
            total_time: *self.stats.total_time.read().unwrap(),
        }
    }
}

impl SurrogateModel {
    /// Predict mean and standard deviation at a point
    fn predict(&self, x: &[f64]) -> (f64, f64) {
        if self.observations.is_empty() {
            return (0.0, 1.0);
        }

        // Simple GP prediction (simplified)
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (obs_x, obs_y) in &self.observations {
            let dist = self.compute_distance(x, obs_x);
            let weight = (-dist / 2.0).exp();
            weighted_sum += weight * obs_y;
            total_weight += weight;
        }

        let mean = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        // Simple uncertainty estimate
        let min_dist: f64 = self.observations.iter()
            .map(|(obs_x, _)| self.compute_distance(x, obs_x))
            .fold(f64::INFINITY, f64::min);

        let std = (1.0 - (-min_dist).exp()).max(0.01);

        (mean, std)
    }

    fn compute_distance(&self, x1: &[f64], x2: &[f64]) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .zip(self.length_scales.iter())
            .map(|((a, b), l)| ((a - b) / l).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Statistics summary
#[derive(Debug, Clone, Serialize)]
pub struct TunerStatsSummary {
    pub total_trials: u64,
    pub completed_trials: u64,
    pub failed_trials: u64,
    pub pruned_trials: u64,
    pub best_objective: f64,
    pub total_time: Duration,
}

/// Index parameter presets for common configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexTuningPreset {
    /// Preset name
    pub name: String,
    /// Description
    pub description: String,
    /// Parameters
    pub parameters: HashMap<String, ParameterValue>,
}

/// Create HNSW parameter space
pub fn hnsw_parameter_space() -> ParameterSpace {
    ParameterSpace {
        parameters: vec![
            Parameter {
                name: "m".to_string(),
                param_type: ParameterType::Integer { min: 4, max: 64, step: Some(4), log_scale: false },
                description: "Number of connections per layer".to_string(),
                default: ParameterValue::Integer(16),
            },
            Parameter {
                name: "ef_construction".to_string(),
                param_type: ParameterType::Integer { min: 50, max: 500, step: Some(50), log_scale: false },
                description: "Size of dynamic candidate list during construction".to_string(),
                default: ParameterValue::Integer(100),
            },
            Parameter {
                name: "ef_search".to_string(),
                param_type: ParameterType::Integer { min: 10, max: 500, step: Some(10), log_scale: false },
                description: "Size of dynamic candidate list during search".to_string(),
                default: ParameterValue::Integer(50),
            },
        ],
        constraints: vec![
            Constraint {
                expression: "ef_construction >= ef_search".to_string(),
                constraint_type: ConstraintType::GreaterThan(
                    "ef_construction".to_string(),
                    "ef_search".to_string(),
                ),
            },
        ],
    }
}

/// Create IVF parameter space
pub fn ivf_parameter_space() -> ParameterSpace {
    ParameterSpace {
        parameters: vec![
            Parameter {
                name: "nlist".to_string(),
                param_type: ParameterType::Integer { min: 16, max: 65536, step: None, log_scale: true },
                description: "Number of clusters".to_string(),
                default: ParameterValue::Integer(100),
            },
            Parameter {
                name: "nprobe".to_string(),
                param_type: ParameterType::Integer { min: 1, max: 256, step: None, log_scale: false },
                description: "Number of clusters to search".to_string(),
                default: ParameterValue::Integer(8),
            },
        ],
        constraints: vec![
            Constraint {
                expression: "nprobe <= nlist".to_string(),
                constraint_type: ConstraintType::LessThan(
                    "nprobe".to_string(),
                    "nlist".to_string(),
                ),
            },
        ],
    }
}

/// Workload analyzer for auto-tuning recommendations
pub struct WorkloadAnalyzer {
    queries: RwLock<Vec<QueryProfile>>,
    config: WorkloadConfig,
}

/// Query profile for workload analysis
#[derive(Debug, Clone)]
struct QueryProfile {
    latency_ms: f64,
    result_count: usize,
    filter_complexity: usize,
    timestamp: u64,
}

/// Workload configuration
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    pub sample_rate: f64,
    pub max_samples: usize,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 0.1,
            max_samples: 10000,
        }
    }
}

impl WorkloadAnalyzer {
    /// Create a new workload analyzer
    pub fn new(config: WorkloadConfig) -> Self {
        Self {
            queries: RwLock::new(Vec::new()),
            config,
        }
    }

    /// Record a query
    pub fn record_query(&self, latency_ms: f64, result_count: usize, filter_complexity: usize) {
        if random_f64(0.0, 1.0) > self.config.sample_rate {
            return;
        }

        let mut queries = self.queries.write().unwrap();
        if queries.len() >= self.config.max_samples {
            queries.remove(0);
        }

        queries.push(QueryProfile {
            latency_ms,
            result_count,
            filter_complexity,
            timestamp: current_timestamp(),
        });
    }

    /// Analyze workload and recommend parameters
    pub fn recommend(&self) -> WorkloadRecommendation {
        let queries = self.queries.read().unwrap();

        if queries.is_empty() {
            return WorkloadRecommendation {
                latency_sensitive: false,
                throughput_focused: false,
                filter_heavy: false,
                recommended_ef_search: 50,
                recommended_nprobe: 8,
            };
        }

        let avg_latency: f64 = queries.iter().map(|q| q.latency_ms).sum::<f64>() / queries.len() as f64;
        let avg_filter_complexity: f64 = queries.iter().map(|q| q.filter_complexity as f64).sum::<f64>() / queries.len() as f64;

        let latency_sensitive = avg_latency > 100.0;
        let filter_heavy = avg_filter_complexity > 2.0;

        WorkloadRecommendation {
            latency_sensitive,
            throughput_focused: queries.len() > 1000,
            filter_heavy,
            recommended_ef_search: if latency_sensitive { 30 } else { 100 },
            recommended_nprobe: if filter_heavy { 16 } else { 8 },
        }
    }
}

/// Workload recommendation
#[derive(Debug, Clone, Serialize)]
pub struct WorkloadRecommendation {
    pub latency_sensitive: bool,
    pub throughput_focused: bool,
    pub filter_heavy: bool,
    pub recommended_ef_search: usize,
    pub recommended_nprobe: usize,
}

// Helper functions
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn random_f64(min: f64, max: f64) -> f64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    std::time::Instant::now().hash(&mut hasher);
    let hash = hasher.finish();

    min + (hash as f64 / u64::MAX as f64) * (max - min)
}

fn random_i64(min: i64, max: i64) -> i64 {
    (random_f64(min as f64, max as f64)) as i64
}

fn random_usize(min: usize, max: usize) -> usize {
    if max <= min {
        return min;
    }
    (random_f64(min as f64, max as f64)) as usize
}

fn random_bool() -> bool {
    random_f64(0.0, 1.0) > 0.5
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn erf(x: f64) -> f64 {
    // Approximation
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

/// Builder for AutoTuner
pub struct AutoTunerBuilder {
    config: AutoTuneConfig,
    parameter_space: Option<ParameterSpace>,
}

impl AutoTunerBuilder {
    pub fn new() -> Self {
        Self {
            config: AutoTuneConfig::default(),
            parameter_space: None,
        }
    }

    pub fn strategy(mut self, strategy: SearchStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn max_iterations(mut self, iterations: usize) -> Self {
        self.config.max_iterations = iterations;
        self
    }

    pub fn time_budget(mut self, duration: Duration) -> Self {
        self.config.time_budget = duration;
        self
    }

    pub fn target_metric(mut self, metric: TuneMetric) -> Self {
        self.config.target_metric = metric;
        self
    }

    pub fn direction(mut self, direction: OptimizeDirection) -> Self {
        self.config.direction = direction;
        self
    }

    pub fn parameter_space(mut self, space: ParameterSpace) -> Self {
        self.parameter_space = Some(space);
        self
    }

    pub fn build(self) -> AutoTuner {
        AutoTuner::new(
            self.config,
            self.parameter_space.unwrap_or_else(hnsw_parameter_space),
        )
    }
}

impl Default for AutoTunerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_tuner_creation() {
        let tuner = AutoTunerBuilder::new()
            .strategy(SearchStrategy::RandomSearch)
            .max_iterations(10)
            .build();

        let stats = tuner.get_stats();
        assert_eq!(stats.total_trials, 0);
    }

    #[test]
    fn test_random_sampling() {
        let tuner = AutoTunerBuilder::new()
            .parameter_space(hnsw_parameter_space())
            .build();

        let params = tuner.sample_random();
        assert!(params.contains_key("m"));
        assert!(params.contains_key("ef_construction"));
        assert!(params.contains_key("ef_search"));
    }

    #[test]
    fn test_optimization() {
        let tuner = AutoTunerBuilder::new()
            .strategy(SearchStrategy::RandomSearch)
            .max_iterations(5)
            .build();

        let result = tuner.optimize(|params| {
            // Simple objective: maximize m value
            if let Some(ParameterValue::Integer(m)) = params.get("m") {
                Ok(*m as f64)
            } else {
                Ok(0.0)
            }
        });

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.n_iterations <= 5);
    }

    #[test]
    fn test_hnsw_parameter_space() {
        let space = hnsw_parameter_space();
        assert_eq!(space.parameters.len(), 3);
        assert_eq!(space.constraints.len(), 1);
    }

    #[test]
    fn test_ivf_parameter_space() {
        let space = ivf_parameter_space();
        assert_eq!(space.parameters.len(), 2);
    }

    #[test]
    fn test_workload_analyzer() {
        let analyzer = WorkloadAnalyzer::new(WorkloadConfig {
            sample_rate: 1.0,
            max_samples: 100,
        });

        analyzer.record_query(50.0, 10, 1);
        analyzer.record_query(150.0, 20, 3);

        let recommendation = analyzer.recommend();
        assert!(recommendation.recommended_ef_search > 0);
    }
}
