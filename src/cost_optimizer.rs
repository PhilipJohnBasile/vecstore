// Cost Optimizer - Resource usage optimization and cost management
// Automatic scaling recommendations, resource right-sizing, and cost forecasting

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Cost optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizerConfig {
    /// Target cost per query (dollars)
    pub target_cost_per_query: f64,
    /// Maximum budget (dollars/month)
    pub max_monthly_budget: f64,
    /// Resource pricing
    pub pricing: ResourcePricing,
    /// Optimization aggressiveness (0-1)
    pub aggressiveness: f64,
    /// Minimum performance threshold
    pub min_performance_threshold: f64,
    /// Enable auto-optimization
    pub auto_optimize: bool,
    /// Sampling window for analysis
    pub analysis_window: Duration,
}

impl Default for CostOptimizerConfig {
    fn default() -> Self {
        Self {
            target_cost_per_query: 0.0001, // $0.0001 per query
            max_monthly_budget: 1000.0,    // $1000/month
            pricing: ResourcePricing::default(),
            aggressiveness: 0.5,
            min_performance_threshold: 0.9,
            auto_optimize: false,
            analysis_window: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Resource pricing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePricing {
    /// CPU cost per core-hour
    pub cpu_per_core_hour: f64,
    /// Memory cost per GB-hour
    pub memory_per_gb_hour: f64,
    /// Storage cost per GB-month
    pub storage_per_gb_month: f64,
    /// GPU cost per GPU-hour
    pub gpu_per_hour: f64,
    /// Network egress per GB
    pub network_egress_per_gb: f64,
    /// API calls per 1M requests
    pub api_per_million: f64,
}

impl Default for ResourcePricing {
    fn default() -> Self {
        Self {
            cpu_per_core_hour: 0.03,
            memory_per_gb_hour: 0.004,
            storage_per_gb_month: 0.02,
            gpu_per_hour: 0.50,
            network_egress_per_gb: 0.08,
            api_per_million: 0.20,
        }
    }
}

/// Cost optimizer engine
pub struct CostOptimizer {
    config: CostOptimizerConfig,
    resource_usage: RwLock<ResourceUsageHistory>,
    cost_history: RwLock<VecDeque<CostRecord>>,
    recommendations: RwLock<Vec<CostRecommendation>>,
    stats: OptimizerStats,
}

/// Resource usage history
struct ResourceUsageHistory {
    samples: VecDeque<ResourceSample>,
    max_samples: usize,
}

/// Resource sample
#[derive(Debug, Clone)]
struct ResourceSample {
    timestamp: u64,
    cpu_cores: f64,
    memory_gb: f64,
    storage_gb: f64,
    gpu_count: u32,
    network_gb: f64,
    queries: u64,
    latency_ms: f64,
}

/// Cost record
#[derive(Debug, Clone, Serialize)]
pub struct CostRecord {
    pub timestamp: u64,
    pub period: CostPeriod,
    pub breakdown: CostBreakdown,
    pub total_cost: f64,
    pub query_count: u64,
    pub cost_per_query: f64,
}

/// Cost period
#[derive(Debug, Clone, Serialize)]
pub enum CostPeriod {
    Hour,
    Day,
    Week,
    Month,
}

/// Cost breakdown
#[derive(Debug, Clone, Serialize)]
pub struct CostBreakdown {
    pub cpu_cost: f64,
    pub memory_cost: f64,
    pub storage_cost: f64,
    pub gpu_cost: f64,
    pub network_cost: f64,
    pub api_cost: f64,
}

/// Cost recommendation
#[derive(Debug, Clone, Serialize)]
pub struct CostRecommendation {
    pub id: String,
    pub category: OptimizationCategory,
    pub title: String,
    pub description: String,
    pub estimated_savings: f64,
    pub savings_percent: f64,
    pub risk: RiskLevel,
    pub effort: EffortLevel,
    pub action: OptimizationAction,
    pub created_at: u64,
    pub status: RecommendationStatus,
}

/// Optimization category
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum OptimizationCategory {
    ResourceRightSizing,
    IndexOptimization,
    CachingStrategy,
    QueryOptimization,
    StorageTiering,
    ReplicaManagement,
    AutoScaling,
}

/// Risk level
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

/// Effort level
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum EffortLevel {
    Minimal,
    Moderate,
    Significant,
}

/// Optimization action
#[derive(Debug, Clone, Serialize)]
pub enum OptimizationAction {
    ScaleDown { resource: String, current: f64, target: f64 },
    ScaleUp { resource: String, current: f64, target: f64 },
    EnableFeature { feature: String },
    DisableFeature { feature: String },
    ChangeConfig { key: String, current: String, target: String },
    MigrateStorage { from_tier: String, to_tier: String, size_gb: f64 },
    AdjustReplicas { current: u32, target: u32 },
}

/// Recommendation status
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum RecommendationStatus {
    Pending,
    Applied,
    Dismissed,
    Expired,
}

/// Optimizer statistics
struct OptimizerStats {
    total_cost: AtomicU64, // Stored as cents
    total_queries: AtomicU64,
    recommendations_generated: AtomicU64,
    recommendations_applied: AtomicU64,
    savings_achieved: RwLock<f64>,
}

/// Cost analysis result
#[derive(Debug, Clone, Serialize)]
pub struct CostAnalysis {
    pub current_costs: CostBreakdown,
    pub projected_monthly_cost: f64,
    pub cost_trend: CostTrend,
    pub efficiency_score: f64,
    pub recommendations: Vec<CostRecommendation>,
    pub potential_savings: f64,
}

/// Cost trend
#[derive(Debug, Clone, Serialize)]
pub struct CostTrend {
    pub direction: TrendDirection,
    pub change_percent: f64,
    pub forecast_30_days: f64,
    pub forecast_90_days: f64,
}

/// Trend direction
#[derive(Debug, Clone, Serialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Budget status
#[derive(Debug, Clone, Serialize)]
pub struct BudgetStatus {
    pub monthly_budget: f64,
    pub current_spend: f64,
    pub projected_spend: f64,
    pub remaining: f64,
    pub on_track: bool,
    pub projected_overage: f64,
    pub days_remaining: u32,
}

impl CostOptimizer {
    /// Create a new cost optimizer
    pub fn new(config: CostOptimizerConfig) -> Self {
        Self {
            config,
            resource_usage: RwLock::new(ResourceUsageHistory {
                samples: VecDeque::new(),
                max_samples: 1000,
            }),
            cost_history: RwLock::new(VecDeque::new()),
            recommendations: RwLock::new(Vec::new()),
            stats: OptimizerStats {
                total_cost: AtomicU64::new(0),
                total_queries: AtomicU64::new(0),
                recommendations_generated: AtomicU64::new(0),
                recommendations_applied: AtomicU64::new(0),
                savings_achieved: RwLock::new(0.0),
            },
        }
    }

    /// Record resource usage
    pub fn record_usage(&self, usage: ResourceUsage) {
        let sample = ResourceSample {
            timestamp: current_timestamp(),
            cpu_cores: usage.cpu_cores,
            memory_gb: usage.memory_gb,
            storage_gb: usage.storage_gb,
            gpu_count: usage.gpu_count,
            network_gb: usage.network_gb,
            queries: usage.queries,
            latency_ms: usage.latency_ms,
        };

        let Ok(mut history) = self.resource_usage.write() else { return; };
        if history.samples.len() >= history.max_samples {
            history.samples.pop_front();
        }
        history.samples.push_back(sample);

        // Calculate and record cost
        let cost = self.calculate_cost(&usage);
        self.stats.total_cost.fetch_add((cost * 100.0) as u64, Ordering::Relaxed);
        self.stats.total_queries.fetch_add(usage.queries, Ordering::Relaxed);
    }

    /// Calculate cost for resource usage
    pub fn calculate_cost(&self, usage: &ResourceUsage) -> f64 {
        let pricing = &self.config.pricing;
        let hours = 1.0; // Assume 1 hour sample

        let cpu_cost = usage.cpu_cores * pricing.cpu_per_core_hour * hours;
        let memory_cost = usage.memory_gb * pricing.memory_per_gb_hour * hours;
        let storage_cost = usage.storage_gb * pricing.storage_per_gb_month / 720.0; // Monthly to hourly
        let gpu_cost = usage.gpu_count as f64 * pricing.gpu_per_hour * hours;
        let network_cost = usage.network_gb * pricing.network_egress_per_gb;
        let api_cost = usage.queries as f64 * pricing.api_per_million / 1_000_000.0;

        cpu_cost + memory_cost + storage_cost + gpu_cost + network_cost + api_cost
    }

    /// Analyze current costs and generate recommendations
    pub fn analyze(&self) -> Result<CostAnalysis> {
        let history = self.resource_usage.read()
            .map_err(|_| VecStoreError::LockError("resource_usage lock poisoned".into()))?;

        if history.samples.is_empty() {
            return Ok(CostAnalysis {
                current_costs: CostBreakdown {
                    cpu_cost: 0.0,
                    memory_cost: 0.0,
                    storage_cost: 0.0,
                    gpu_cost: 0.0,
                    network_cost: 0.0,
                    api_cost: 0.0,
                },
                projected_monthly_cost: 0.0,
                cost_trend: CostTrend {
                    direction: TrendDirection::Stable,
                    change_percent: 0.0,
                    forecast_30_days: 0.0,
                    forecast_90_days: 0.0,
                },
                efficiency_score: 1.0,
                recommendations: vec![],
                potential_savings: 0.0,
            });
        }

        // Calculate average usage
        let avg_usage = self.calculate_average_usage(&history.samples);
        let current_costs = self.calculate_breakdown(&avg_usage);
        let projected_monthly = self.project_monthly_cost(&history.samples);
        let cost_trend = self.analyze_trend(&history.samples);
        let efficiency_score = self.calculate_efficiency(&history.samples);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&avg_usage, &history.samples);
        let potential_savings: f64 = recommendations.iter()
            .map(|r| r.estimated_savings)
            .sum();

        // Store recommendations
        *self.recommendations.write()
            .map_err(|_| VecStoreError::LockError("recommendations lock poisoned".into()))? = recommendations.clone();

        Ok(CostAnalysis {
            current_costs,
            projected_monthly_cost: projected_monthly,
            cost_trend,
            efficiency_score,
            recommendations,
            potential_savings,
        })
    }

    fn calculate_average_usage(&self, samples: &VecDeque<ResourceSample>) -> ResourceUsage {
        let n = samples.len() as f64;
        if n == 0.0 {
            return ResourceUsage::default();
        }

        ResourceUsage {
            cpu_cores: samples.iter().map(|s| s.cpu_cores).sum::<f64>() / n,
            memory_gb: samples.iter().map(|s| s.memory_gb).sum::<f64>() / n,
            storage_gb: samples.iter().map(|s| s.storage_gb).sum::<f64>() / n,
            gpu_count: (samples.iter().map(|s| s.gpu_count as f64).sum::<f64>() / n) as u32,
            network_gb: samples.iter().map(|s| s.network_gb).sum::<f64>() / n,
            queries: (samples.iter().map(|s| s.queries as f64).sum::<f64>() / n) as u64,
            latency_ms: samples.iter().map(|s| s.latency_ms).sum::<f64>() / n,
        }
    }

    fn calculate_breakdown(&self, usage: &ResourceUsage) -> CostBreakdown {
        let pricing = &self.config.pricing;

        CostBreakdown {
            cpu_cost: usage.cpu_cores * pricing.cpu_per_core_hour,
            memory_cost: usage.memory_gb * pricing.memory_per_gb_hour,
            storage_cost: usage.storage_gb * pricing.storage_per_gb_month / 720.0,
            gpu_cost: usage.gpu_count as f64 * pricing.gpu_per_hour,
            network_cost: usage.network_gb * pricing.network_egress_per_gb,
            api_cost: usage.queries as f64 * pricing.api_per_million / 1_000_000.0,
        }
    }

    fn project_monthly_cost(&self, samples: &VecDeque<ResourceSample>) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        let total_cost: f64 = samples.iter()
            .map(|s| {
                let usage = ResourceUsage {
                    cpu_cores: s.cpu_cores,
                    memory_gb: s.memory_gb,
                    storage_gb: s.storage_gb,
                    gpu_count: s.gpu_count,
                    network_gb: s.network_gb,
                    queries: s.queries,
                    latency_ms: s.latency_ms,
                };
                self.calculate_cost(&usage)
            })
            .sum();

        let avg_hourly = total_cost / samples.len() as f64;
        avg_hourly * 720.0 // 720 hours per month
    }

    fn analyze_trend(&self, samples: &VecDeque<ResourceSample>) -> CostTrend {
        if samples.len() < 2 {
            return CostTrend {
                direction: TrendDirection::Stable,
                change_percent: 0.0,
                forecast_30_days: 0.0,
                forecast_90_days: 0.0,
            };
        }

        let half = samples.len() / 2;
        let first_half: Vec<_> = samples.iter().take(half).collect();
        let second_half: Vec<_> = samples.iter().skip(half).collect();

        let first_avg: f64 = first_half.iter()
            .map(|s| self.calculate_cost(&ResourceUsage::from_sample(s)))
            .sum::<f64>() / first_half.len() as f64;

        let second_avg: f64 = second_half.iter()
            .map(|s| self.calculate_cost(&ResourceUsage::from_sample(s)))
            .sum::<f64>() / second_half.len() as f64;

        let change_percent = if first_avg > 0.0 {
            ((second_avg - first_avg) / first_avg) * 100.0
        } else {
            0.0
        };

        let direction = if change_percent > 5.0 {
            TrendDirection::Increasing
        } else if change_percent < -5.0 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        let current_monthly = second_avg * 720.0;
        let monthly_change_rate = change_percent / 100.0;

        CostTrend {
            direction,
            change_percent,
            forecast_30_days: current_monthly * (1.0 + monthly_change_rate),
            forecast_90_days: current_monthly * (1.0 + monthly_change_rate * 3.0),
        }
    }

    fn calculate_efficiency(&self, samples: &VecDeque<ResourceSample>) -> f64 {
        if samples.is_empty() {
            return 1.0;
        }

        let total_queries: u64 = samples.iter().map(|s| s.queries).sum();
        let total_cost: f64 = samples.iter()
            .map(|s| self.calculate_cost(&ResourceUsage::from_sample(s)))
            .sum();

        if total_cost > 0.0 && total_queries > 0 {
            let actual_cost_per_query = total_cost / total_queries as f64;
            let target = self.config.target_cost_per_query;

            if actual_cost_per_query <= target {
                1.0
            } else {
                target / actual_cost_per_query
            }
        } else {
            1.0
        }
    }

    fn generate_recommendations(
        &self,
        avg_usage: &ResourceUsage,
        samples: &VecDeque<ResourceSample>,
    ) -> Vec<CostRecommendation> {
        let mut recommendations = Vec::new();
        let now = current_timestamp();

        // Check CPU utilization
        let cpu_utilization = self.estimate_cpu_utilization(samples);
        if cpu_utilization < 0.3 {
            let target_cores = (avg_usage.cpu_cores * 0.5).max(1.0);
            let savings = (avg_usage.cpu_cores - target_cores)
                * self.config.pricing.cpu_per_core_hour * 720.0;

            recommendations.push(CostRecommendation {
                id: format!("cpu-rightsize-{}", now),
                category: OptimizationCategory::ResourceRightSizing,
                title: "CPU Over-provisioned".to_string(),
                description: format!(
                    "CPU utilization is only {:.0}%. Consider reducing cores from {:.1} to {:.1}.",
                    cpu_utilization * 100.0, avg_usage.cpu_cores, target_cores
                ),
                estimated_savings: savings,
                savings_percent: savings / self.project_monthly_cost(samples) * 100.0,
                risk: RiskLevel::Low,
                effort: EffortLevel::Minimal,
                action: OptimizationAction::ScaleDown {
                    resource: "cpu_cores".to_string(),
                    current: avg_usage.cpu_cores,
                    target: target_cores,
                },
                created_at: now,
                status: RecommendationStatus::Pending,
            });
        }

        // Check memory utilization
        let memory_utilization = self.estimate_memory_utilization(samples);
        if memory_utilization < 0.4 {
            let target_memory = (avg_usage.memory_gb * 0.6).max(1.0);
            let savings = (avg_usage.memory_gb - target_memory)
                * self.config.pricing.memory_per_gb_hour * 720.0;

            recommendations.push(CostRecommendation {
                id: format!("mem-rightsize-{}", now),
                category: OptimizationCategory::ResourceRightSizing,
                title: "Memory Over-provisioned".to_string(),
                description: format!(
                    "Memory utilization is only {:.0}%. Consider reducing from {:.1}GB to {:.1}GB.",
                    memory_utilization * 100.0, avg_usage.memory_gb, target_memory
                ),
                estimated_savings: savings,
                savings_percent: savings / self.project_monthly_cost(samples) * 100.0,
                risk: RiskLevel::Medium,
                effort: EffortLevel::Minimal,
                action: OptimizationAction::ScaleDown {
                    resource: "memory_gb".to_string(),
                    current: avg_usage.memory_gb,
                    target: target_memory,
                },
                created_at: now,
                status: RecommendationStatus::Pending,
            });
        }

        // Check storage tiering opportunity
        if avg_usage.storage_gb > 100.0 {
            let cold_data_ratio = 0.3; // Assume 30% could be cold
            let cold_storage_savings = avg_usage.storage_gb * cold_data_ratio
                * self.config.pricing.storage_per_gb_month * 0.7; // 70% savings on cold tier

            recommendations.push(CostRecommendation {
                id: format!("storage-tier-{}", now),
                category: OptimizationCategory::StorageTiering,
                title: "Storage Tiering Opportunity".to_string(),
                description: format!(
                    "Move infrequently accessed data ({:.1}GB) to cold storage tier.",
                    avg_usage.storage_gb * cold_data_ratio
                ),
                estimated_savings: cold_storage_savings,
                savings_percent: cold_storage_savings / self.project_monthly_cost(samples) * 100.0,
                risk: RiskLevel::Low,
                effort: EffortLevel::Moderate,
                action: OptimizationAction::MigrateStorage {
                    from_tier: "hot".to_string(),
                    to_tier: "cold".to_string(),
                    size_gb: avg_usage.storage_gb * cold_data_ratio,
                },
                created_at: now,
                status: RecommendationStatus::Pending,
            });
        }

        // Check cache hit rate (if we had this data)
        let estimated_cache_hit_rate = 0.5; // Placeholder
        if estimated_cache_hit_rate < 0.7 {
            let potential_query_reduction = 0.2; // 20% of queries could be cached
            let api_savings = avg_usage.queries as f64 * potential_query_reduction
                * self.config.pricing.api_per_million / 1_000_000.0 * 720.0;

            recommendations.push(CostRecommendation {
                id: format!("cache-improve-{}", now),
                category: OptimizationCategory::CachingStrategy,
                title: "Improve Cache Hit Rate".to_string(),
                description: "Increase cache size or adjust caching strategy to improve hit rate.".to_string(),
                estimated_savings: api_savings,
                savings_percent: api_savings / self.project_monthly_cost(samples) * 100.0,
                risk: RiskLevel::Low,
                effort: EffortLevel::Minimal,
                action: OptimizationAction::ChangeConfig {
                    key: "cache_size_mb".to_string(),
                    current: "256".to_string(),
                    target: "512".to_string(),
                },
                created_at: now,
                status: RecommendationStatus::Pending,
            });
        }

        // Sort by estimated savings
        recommendations.sort_by(|a, b| {
            b.estimated_savings.partial_cmp(&a.estimated_savings)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.stats.recommendations_generated.fetch_add(recommendations.len() as u64, Ordering::Relaxed);

        recommendations
    }

    fn estimate_cpu_utilization(&self, _samples: &VecDeque<ResourceSample>) -> f64 {
        // In real implementation, would track actual CPU usage
        0.45 // Placeholder
    }

    fn estimate_memory_utilization(&self, _samples: &VecDeque<ResourceSample>) -> f64 {
        // In real implementation, would track actual memory usage
        0.55 // Placeholder
    }

    /// Get budget status
    pub fn get_budget_status(&self) -> BudgetStatus {
        let total_cost = self.stats.total_cost.load(Ordering::Relaxed) as f64 / 100.0;
        let _projected = {
            let Ok(history) = self.resource_usage.read() else {
                return BudgetStatus {
                    monthly_budget: self.config.max_monthly_budget,
                    current_spend: total_cost,
                    projected_spend: 0.0,
                    remaining: (self.config.max_monthly_budget - total_cost).max(0.0),
                    on_track: true,
                    projected_overage: 0.0,
                    days_remaining: 30,
                };
            };
            self.project_monthly_cost(&history.samples)
        };

        let days_elapsed = 15; // Placeholder - would calculate from start of month
        let days_remaining = 30 - days_elapsed;
        let daily_rate = total_cost / days_elapsed as f64;
        let projected_spend = total_cost + (daily_rate * days_remaining as f64);

        BudgetStatus {
            monthly_budget: self.config.max_monthly_budget,
            current_spend: total_cost,
            projected_spend,
            remaining: (self.config.max_monthly_budget - total_cost).max(0.0),
            on_track: projected_spend <= self.config.max_monthly_budget,
            projected_overage: (projected_spend - self.config.max_monthly_budget).max(0.0),
            days_remaining: days_remaining as u32,
        }
    }

    /// Apply a recommendation
    pub fn apply_recommendation(&self, recommendation_id: &str) -> Result<bool> {
        let mut recommendations = self.recommendations.write()
            .map_err(|_| VecStoreError::LockError("recommendations lock poisoned".into()))?;

        if let Some(rec) = recommendations.iter_mut().find(|r| r.id == recommendation_id) {
            if rec.status != RecommendationStatus::Pending {
                return Ok(false);
            }

            rec.status = RecommendationStatus::Applied;
            self.stats.recommendations_applied.fetch_add(1, Ordering::Relaxed);
            *self.stats.savings_achieved.write()
                .map_err(|_| VecStoreError::LockError("savings_achieved lock poisoned".into()))? += rec.estimated_savings;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Dismiss a recommendation
    pub fn dismiss_recommendation(&self, recommendation_id: &str) -> Result<bool> {
        let mut recommendations = self.recommendations.write()
            .map_err(|_| VecStoreError::LockError("recommendations lock poisoned".into()))?;

        if let Some(rec) = recommendations.iter_mut().find(|r| r.id == recommendation_id) {
            rec.status = RecommendationStatus::Dismissed;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get pending recommendations
    pub fn get_recommendations(&self) -> Vec<CostRecommendation> {
        let Ok(recommendations) = self.recommendations.read() else { return Vec::new(); };
        recommendations
            .iter()
            .filter(|r| r.status == RecommendationStatus::Pending)
            .cloned()
            .collect()
    }

    /// Get optimizer statistics
    pub fn get_stats(&self) -> OptimizerStatsSummary {
        let Ok(savings) = self.stats.savings_achieved.read() else {
            return OptimizerStatsSummary {
                total_cost: self.stats.total_cost.load(Ordering::Relaxed) as f64 / 100.0,
                total_queries: self.stats.total_queries.load(Ordering::Relaxed),
                cost_per_query: 0.0,
                recommendations_generated: self.stats.recommendations_generated.load(Ordering::Relaxed),
                recommendations_applied: self.stats.recommendations_applied.load(Ordering::Relaxed),
                savings_achieved: 0.0,
            };
        };
        OptimizerStatsSummary {
            total_cost: self.stats.total_cost.load(Ordering::Relaxed) as f64 / 100.0,
            total_queries: self.stats.total_queries.load(Ordering::Relaxed),
            cost_per_query: {
                let cost = self.stats.total_cost.load(Ordering::Relaxed) as f64 / 100.0;
                let queries = self.stats.total_queries.load(Ordering::Relaxed);
                if queries > 0 { cost / queries as f64 } else { 0.0 }
            },
            recommendations_generated: self.stats.recommendations_generated.load(Ordering::Relaxed),
            recommendations_applied: self.stats.recommendations_applied.load(Ordering::Relaxed),
            savings_achieved: *savings,
        }
    }

    /// Forecast costs for future period
    pub fn forecast(&self, days: u32) -> CostForecast {
        let default_forecast = CostForecast {
            period_days: days,
            base_forecast: 0.0,
            optimistic_forecast: 0.0,
            pessimistic_forecast: 0.0,
            with_recommendations: 0.0,
        };
        let Ok(history) = self.resource_usage.read() else { return default_forecast; };
        let monthly_cost = self.project_monthly_cost(&history.samples);
        let daily_cost = monthly_cost / 30.0;

        let _hours = days as f64 * 24.0;

        CostForecast {
            period_days: days,
            base_forecast: daily_cost * days as f64,
            optimistic_forecast: daily_cost * days as f64 * 0.85,
            pessimistic_forecast: daily_cost * days as f64 * 1.15,
            with_recommendations: {
                let Ok(recommendations) = self.recommendations.read() else {
                    return CostForecast {
                        period_days: days,
                        base_forecast: daily_cost * days as f64,
                        optimistic_forecast: daily_cost * days as f64 * 0.85,
                        pessimistic_forecast: daily_cost * days as f64 * 1.15,
                        with_recommendations: daily_cost * days as f64,
                    };
                };
                let savings: f64 = recommendations
                    .iter()
                    .filter(|r| r.status == RecommendationStatus::Pending)
                    .map(|r| r.estimated_savings * days as f64 / 30.0)
                    .sum();
                (daily_cost * days as f64 - savings).max(0.0)
            },
        }
    }
}

/// Resource usage input
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    pub cpu_cores: f64,
    pub memory_gb: f64,
    pub storage_gb: f64,
    pub gpu_count: u32,
    pub network_gb: f64,
    pub queries: u64,
    pub latency_ms: f64,
}

impl ResourceUsage {
    fn from_sample(sample: &ResourceSample) -> Self {
        Self {
            cpu_cores: sample.cpu_cores,
            memory_gb: sample.memory_gb,
            storage_gb: sample.storage_gb,
            gpu_count: sample.gpu_count,
            network_gb: sample.network_gb,
            queries: sample.queries,
            latency_ms: sample.latency_ms,
        }
    }
}

/// Optimizer statistics summary
#[derive(Debug, Clone, Serialize)]
pub struct OptimizerStatsSummary {
    pub total_cost: f64,
    pub total_queries: u64,
    pub cost_per_query: f64,
    pub recommendations_generated: u64,
    pub recommendations_applied: u64,
    pub savings_achieved: f64,
}

/// Cost forecast
#[derive(Debug, Clone, Serialize)]
pub struct CostForecast {
    pub period_days: u32,
    pub base_forecast: f64,
    pub optimistic_forecast: f64,
    pub pessimistic_forecast: f64,
    pub with_recommendations: f64,
}

/// Auto-scaler based on cost optimization
pub struct CostAwareAutoScaler {
    optimizer: std::sync::Arc<CostOptimizer>,
    config: AutoScalerConfig,
    current_scale: RwLock<ScaleState>,
}

/// Auto-scaler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalerConfig {
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub target_cpu_utilization: f64,
    pub target_cost_per_query: f64,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub cooldown_period: Duration,
}

impl Default for AutoScalerConfig {
    fn default() -> Self {
        Self {
            min_replicas: 1,
            max_replicas: 10,
            target_cpu_utilization: 0.7,
            target_cost_per_query: 0.0001,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.4,
            cooldown_period: Duration::from_secs(300),
        }
    }
}

/// Current scale state
struct ScaleState {
    replicas: u32,
    last_scale_action: Option<Instant>,
}

/// Scaling decision
#[derive(Debug, Clone, Serialize)]
pub struct ScalingDecision {
    pub action: ScaleAction,
    pub current_replicas: u32,
    pub target_replicas: u32,
    pub reason: String,
    pub cost_impact: f64,
}

/// Scale action
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum ScaleAction {
    ScaleUp,
    ScaleDown,
    NoChange,
}

impl CostAwareAutoScaler {
    /// Create a new cost-aware auto-scaler
    pub fn new(optimizer: std::sync::Arc<CostOptimizer>, config: AutoScalerConfig) -> Self {
        Self {
            optimizer,
            config,
            current_scale: RwLock::new(ScaleState {
                replicas: 1,
                last_scale_action: None,
            }),
        }
    }

    /// Evaluate and return scaling decision
    pub fn evaluate(&self, current_load: f64, _current_latency: f64) -> ScalingDecision {
        let Ok(state) = self.current_scale.read() else {
            return ScalingDecision {
                action: ScaleAction::NoChange,
                current_replicas: 1,
                target_replicas: 1,
                reason: "Lock error".to_string(),
                cost_impact: 0.0,
            };
        };

        // Check cooldown
        if let Some(last_action) = state.last_scale_action {
            if last_action.elapsed() < self.config.cooldown_period {
                return ScalingDecision {
                    action: ScaleAction::NoChange,
                    current_replicas: state.replicas,
                    target_replicas: state.replicas,
                    reason: "In cooldown period".to_string(),
                    cost_impact: 0.0,
                };
            }
        }

        let current_replicas = state.replicas;
        drop(state);

        // Calculate utilization per replica
        let utilization_per_replica = current_load / current_replicas as f64;

        // Get current cost metrics
        let stats = self.optimizer.get_stats();
        let current_cost_per_query = stats.cost_per_query;

        // Scale up conditions
        if utilization_per_replica > self.config.scale_up_threshold
            && current_replicas < self.config.max_replicas
        {
            let target = (current_replicas + 1).min(self.config.max_replicas);
            let cost_increase = self.estimate_replica_cost() * (target - current_replicas) as f64;

            return ScalingDecision {
                action: ScaleAction::ScaleUp,
                current_replicas,
                target_replicas: target,
                reason: format!("High utilization: {:.0}%", utilization_per_replica * 100.0),
                cost_impact: cost_increase,
            };
        }

        // Scale down conditions
        if utilization_per_replica < self.config.scale_down_threshold
            && current_replicas > self.config.min_replicas
            && current_cost_per_query > self.config.target_cost_per_query
        {
            let target = (current_replicas - 1).max(self.config.min_replicas);
            let cost_savings = self.estimate_replica_cost() * (current_replicas - target) as f64;

            return ScalingDecision {
                action: ScaleAction::ScaleDown,
                current_replicas,
                target_replicas: target,
                reason: format!(
                    "Low utilization ({:.0}%) and high cost per query (${:.6})",
                    utilization_per_replica * 100.0, current_cost_per_query
                ),
                cost_impact: -cost_savings,
            };
        }

        ScalingDecision {
            action: ScaleAction::NoChange,
            current_replicas,
            target_replicas: current_replicas,
            reason: "Optimal scale".to_string(),
            cost_impact: 0.0,
        }
    }

    fn estimate_replica_cost(&self) -> f64 {
        // Estimate hourly cost per replica
        let pricing = &self.optimizer.config.pricing;
        let cpu_per_replica = 2.0;
        let memory_per_replica = 8.0;

        (cpu_per_replica * pricing.cpu_per_core_hour
            + memory_per_replica * pricing.memory_per_gb_hour) * 720.0
    }

    /// Apply scaling decision
    pub fn apply(&self, decision: &ScalingDecision) -> bool {
        if decision.action == ScaleAction::NoChange {
            return false;
        }

        let Ok(mut state) = self.current_scale.write() else { return false; };
        state.replicas = decision.target_replicas;
        state.last_scale_action = Some(Instant::now());

        true
    }
}

/// Builder for CostOptimizer
#[must_use = "builders do nothing unless built"]
pub struct CostOptimizerBuilder {
    config: CostOptimizerConfig,
}

impl CostOptimizerBuilder {
    pub fn new() -> Self {
        Self {
            config: CostOptimizerConfig::default(),
        }
    }

    pub fn target_cost_per_query(mut self, cost: f64) -> Self {
        self.config.target_cost_per_query = cost;
        self
    }

    pub fn max_monthly_budget(mut self, budget: f64) -> Self {
        self.config.max_monthly_budget = budget;
        self
    }

    pub fn pricing(mut self, pricing: ResourcePricing) -> Self {
        self.config.pricing = pricing;
        self
    }

    pub fn auto_optimize(mut self, enabled: bool) -> Self {
        self.config.auto_optimize = enabled;
        self
    }

    pub fn build(self) -> CostOptimizer {
        CostOptimizer::new(self.config)
    }
}

impl Default for CostOptimizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_optimizer_creation() {
        let optimizer = CostOptimizerBuilder::new()
            .target_cost_per_query(0.0001)
            .max_monthly_budget(500.0)
            .build();

        let stats = optimizer.get_stats();
        assert_eq!(stats.total_queries, 0);
    }

    #[test]
    fn test_cost_calculation() {
        let optimizer = CostOptimizerBuilder::new().build();

        let usage = ResourceUsage {
            cpu_cores: 4.0,
            memory_gb: 16.0,
            storage_gb: 100.0,
            gpu_count: 0,
            network_gb: 1.0,
            queries: 1000,
            latency_ms: 10.0,
        };

        let cost = optimizer.calculate_cost(&usage);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_record_usage() {
        let optimizer = CostOptimizerBuilder::new().build();

        let usage = ResourceUsage {
            cpu_cores: 4.0,
            memory_gb: 16.0,
            storage_gb: 100.0,
            gpu_count: 0,
            network_gb: 1.0,
            queries: 1000,
            latency_ms: 10.0,
        };

        optimizer.record_usage(usage);

        let stats = optimizer.get_stats();
        assert_eq!(stats.total_queries, 1000);
    }

    #[test]
    fn test_budget_status() {
        let optimizer = CostOptimizerBuilder::new()
            .max_monthly_budget(1000.0)
            .build();

        let status = optimizer.get_budget_status();
        assert_eq!(status.monthly_budget, 1000.0);
    }

    #[test]
    fn test_forecast() {
        let optimizer = CostOptimizerBuilder::new().build();

        // Record some usage
        for _ in 0..10 {
            optimizer.record_usage(ResourceUsage {
                cpu_cores: 2.0,
                memory_gb: 8.0,
                storage_gb: 50.0,
                gpu_count: 0,
                network_gb: 0.5,
                queries: 500,
                latency_ms: 15.0,
            });
        }

        let forecast = optimizer.forecast(30);
        assert!(forecast.base_forecast >= 0.0);
    }

    #[test]
    fn test_auto_scaler() {
        let optimizer = std::sync::Arc::new(CostOptimizerBuilder::new().build());
        let scaler = CostAwareAutoScaler::new(optimizer, AutoScalerConfig::default());

        let decision = scaler.evaluate(0.3, 10.0);
        assert_eq!(decision.action, ScaleAction::NoChange);
    }
}
