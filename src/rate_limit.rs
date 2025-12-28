//! Rate Limiting for VecStore
//!
//! This module provides rate limiting functionality to control throughput
//! for queries and writes. Useful for:
//!
//! - Preventing resource exhaustion in multi-tenant environments
//! - Ensuring fair resource allocation
//! - Protecting against accidental DoS from clients
//!
//! ## Algorithms
//!
//! - **Token Bucket**: Allows bursts while maintaining average rate
//! - **Sliding Window**: Precise rate limiting over a time window
//! - **Leaky Bucket**: Smooths traffic to constant rate
//!
//! ## Usage
//!
//! ```no_run
//! use vecstore::rate_limit::{RateLimiter, RateLimitConfig};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Create a rate limiter: 100 requests per second, burst of 20
//! let limiter = RateLimiter::new(RateLimitConfig {
//!     requests_per_second: 100.0,
//!     burst_size: 20,
//!     ..Default::default()
//! });
//!
//! // Check if request is allowed
//! if limiter.try_acquire() {
//!     // Process request
//! } else {
//!     // Rate limited - return 429 or queue
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Configuration for rate limiting
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per second (average rate)
    pub requests_per_second: f64,

    /// Maximum burst size (tokens)
    pub burst_size: usize,

    /// Algorithm to use
    pub algorithm: RateLimitAlgorithm,

    /// Whether to enable rate limiting
    pub enabled: bool,

    /// Optional separate limits for different operation types
    pub operation_limits: HashMap<String, f64>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 1000.0,
            burst_size: 100,
            algorithm: RateLimitAlgorithm::TokenBucket,
            enabled: true,
            operation_limits: HashMap::new(),
        }
    }
}

/// Rate limiting algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitAlgorithm {
    /// Token bucket - allows bursts, good for general use
    TokenBucket,
    /// Sliding window - more precise, higher memory usage
    SlidingWindow,
    /// Leaky bucket - smooths traffic to constant rate
    LeakyBucket,
}

/// Result of a rate limit check
#[derive(Debug, Clone)]
pub struct RateLimitResult {
    /// Whether the request is allowed
    pub allowed: bool,

    /// Remaining tokens/requests in this window
    pub remaining: usize,

    /// Time until the limit resets (for retry-after header)
    pub reset_after: Duration,

    /// Current rate (requests per second)
    pub current_rate: f64,
}

/// Thread-safe rate limiter
pub struct RateLimiter {
    inner: Arc<RateLimiterInner>,
}

struct RateLimiterInner {
    config: RateLimitConfig,
    state: RwLock<RateLimiterState>,
    total_allowed: AtomicU64,
    total_denied: AtomicU64,
}

struct RateLimiterState {
    /// Token bucket state
    tokens: f64,
    last_refill: Instant,

    /// Sliding window state
    window_requests: Vec<Instant>,

    /// Leaky bucket state
    queue_size: f64,
    last_drain: Instant,

    /// Per-operation limits
    operation_states: HashMap<String, OperationState>,
}

struct OperationState {
    tokens: f64,
    last_refill: Instant,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration
    pub fn new(config: RateLimitConfig) -> Self {
        let state = RateLimiterState {
            tokens: config.burst_size as f64,
            last_refill: Instant::now(),
            window_requests: Vec::new(),
            queue_size: 0.0,
            last_drain: Instant::now(),
            operation_states: HashMap::new(),
        };

        Self {
            inner: Arc::new(RateLimiterInner {
                config,
                state: RwLock::new(state),
                total_allowed: AtomicU64::new(0),
                total_denied: AtomicU64::new(0),
            }),
        }
    }

    /// Create a rate limiter with a simple rate (requests per second)
    pub fn with_rate(requests_per_second: f64) -> Self {
        Self::new(RateLimitConfig {
            requests_per_second,
            burst_size: (requests_per_second * 0.1).max(1.0) as usize,
            ..Default::default()
        })
    }

    /// Try to acquire a permit (non-blocking)
    ///
    /// Returns true if the request is allowed, false if rate limited
    pub fn try_acquire(&self) -> bool {
        self.try_acquire_result().allowed
    }

    /// Try to acquire with detailed result
    pub fn try_acquire_result(&self) -> RateLimitResult {
        if !self.inner.config.enabled {
            return RateLimitResult {
                allowed: true,
                remaining: self.inner.config.burst_size,
                reset_after: Duration::ZERO,
                current_rate: 0.0,
            };
        }

        match self.inner.config.algorithm {
            RateLimitAlgorithm::TokenBucket => self.token_bucket_acquire(),
            RateLimitAlgorithm::SlidingWindow => self.sliding_window_acquire(),
            RateLimitAlgorithm::LeakyBucket => self.leaky_bucket_acquire(),
        }
    }

    /// Try to acquire for a specific operation type
    pub fn try_acquire_for(&self, operation: &str) -> bool {
        if !self.inner.config.enabled {
            return true;
        }

        // Check if there's a specific limit for this operation
        let rate = self.inner.config.operation_limits
            .get(operation)
            .copied()
            .unwrap_or(self.inner.config.requests_per_second);

        let Ok(mut state) = self.inner.state.write() else { return false; };
        let now = Instant::now();

        let op_state = state.operation_states
            .entry(operation.to_string())
            .or_insert_with(|| OperationState {
                tokens: self.inner.config.burst_size as f64,
                last_refill: now,
            });

        // Refill tokens
        let elapsed = now.duration_since(op_state.last_refill).as_secs_f64();
        let refill = elapsed * rate;
        op_state.tokens = (op_state.tokens + refill).min(self.inner.config.burst_size as f64);
        op_state.last_refill = now;

        // Try to consume a token
        if op_state.tokens >= 1.0 {
            op_state.tokens -= 1.0;
            self.inner.total_allowed.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            self.inner.total_denied.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    /// Token bucket algorithm implementation
    fn token_bucket_acquire(&self) -> RateLimitResult {
        let Ok(mut state) = self.inner.state.write() else {
            return RateLimitResult {
                allowed: false,
                remaining: 0,
                reset_after: Duration::from_secs(1),
                current_rate: 0.0,
            };
        };
        let now = Instant::now();

        // Refill tokens based on time elapsed
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        let refill = elapsed * self.inner.config.requests_per_second;
        state.tokens = (state.tokens + refill).min(self.inner.config.burst_size as f64);
        state.last_refill = now;

        // Try to consume a token
        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            self.inner.total_allowed.fetch_add(1, Ordering::Relaxed);

            RateLimitResult {
                allowed: true,
                remaining: state.tokens as usize,
                reset_after: Duration::ZERO,
                current_rate: self.calculate_rate(&state),
            }
        } else {
            self.inner.total_denied.fetch_add(1, Ordering::Relaxed);

            // Calculate time until next token
            let time_for_token = 1.0 / self.inner.config.requests_per_second;
            let reset_after = Duration::from_secs_f64(time_for_token * (1.0 - state.tokens));

            RateLimitResult {
                allowed: false,
                remaining: 0,
                reset_after,
                current_rate: self.calculate_rate(&state),
            }
        }
    }

    /// Sliding window algorithm implementation
    fn sliding_window_acquire(&self) -> RateLimitResult {
        let Ok(mut state) = self.inner.state.write() else {
            return RateLimitResult {
                allowed: false,
                remaining: 0,
                reset_after: Duration::from_secs(1),
                current_rate: 0.0,
            };
        };
        let now = Instant::now();
        let window = Duration::from_secs(1);

        // Remove old requests
        state.window_requests.retain(|t| now.duration_since(*t) < window);

        let current_count = state.window_requests.len();
        let max_requests = self.inner.config.requests_per_second as usize;

        if current_count < max_requests {
            state.window_requests.push(now);
            self.inner.total_allowed.fetch_add(1, Ordering::Relaxed);

            RateLimitResult {
                allowed: true,
                remaining: max_requests - current_count - 1,
                reset_after: Duration::ZERO,
                current_rate: current_count as f64,
            }
        } else {
            self.inner.total_denied.fetch_add(1, Ordering::Relaxed);

            // Calculate reset time based on oldest request
            let oldest = state.window_requests.first().copied().unwrap_or(now);
            let reset_after = window.saturating_sub(now.duration_since(oldest));

            RateLimitResult {
                allowed: false,
                remaining: 0,
                reset_after,
                current_rate: current_count as f64,
            }
        }
    }

    /// Leaky bucket algorithm implementation
    fn leaky_bucket_acquire(&self) -> RateLimitResult {
        let Ok(mut state) = self.inner.state.write() else {
            return RateLimitResult {
                allowed: false,
                remaining: 0,
                reset_after: Duration::from_secs(1),
                current_rate: 0.0,
            };
        };
        let now = Instant::now();

        // Drain the bucket
        let elapsed = now.duration_since(state.last_drain).as_secs_f64();
        let drained = elapsed * self.inner.config.requests_per_second;
        state.queue_size = (state.queue_size - drained).max(0.0);
        state.last_drain = now;

        // Check if adding one more would exceed capacity
        // Use ceiling of queue_size to avoid floating-point edge cases
        let effective_queue = state.queue_size.ceil() as usize;
        if effective_queue < self.inner.config.burst_size {
            state.queue_size += 1.0;
            self.inner.total_allowed.fetch_add(1, Ordering::Relaxed);

            RateLimitResult {
                allowed: true,
                remaining: self.inner.config.burst_size - effective_queue - 1,
                reset_after: Duration::ZERO,
                current_rate: state.queue_size,
            }
        } else {
            self.inner.total_denied.fetch_add(1, Ordering::Relaxed);

            // Calculate time until space is available
            let time_for_drain = if self.inner.config.requests_per_second > 0.0 {
                1.0 / self.inner.config.requests_per_second
            } else {
                f64::MAX
            };
            let reset_after = Duration::from_secs_f64(time_for_drain.min(3600.0));

            RateLimitResult {
                allowed: false,
                remaining: 0,
                reset_after,
                current_rate: state.queue_size,
            }
        }
    }

    fn calculate_rate(&self, state: &RateLimiterState) -> f64 {
        // Calculate approximate current rate based on token consumption
        let available = state.tokens;
        let max = self.inner.config.burst_size as f64;
        let consumed_ratio = 1.0 - (available / max);
        consumed_ratio * self.inner.config.requests_per_second
    }

    /// Get current statistics
    pub fn stats(&self) -> RateLimitStats {
        let Ok(state) = self.inner.state.read() else {
            return RateLimitStats {
                total_allowed: self.inner.total_allowed.load(Ordering::Relaxed),
                total_denied: self.inner.total_denied.load(Ordering::Relaxed),
                current_tokens: 0,
                requests_per_second: self.inner.config.requests_per_second,
                burst_size: self.inner.config.burst_size,
            };
        };

        RateLimitStats {
            total_allowed: self.inner.total_allowed.load(Ordering::Relaxed),
            total_denied: self.inner.total_denied.load(Ordering::Relaxed),
            current_tokens: state.tokens as usize,
            requests_per_second: self.inner.config.requests_per_second,
            burst_size: self.inner.config.burst_size,
        }
    }

    /// Reset the rate limiter state
    pub fn reset(&self) {
        let Ok(mut state) = self.inner.state.write() else { return; };
        state.tokens = self.inner.config.burst_size as f64;
        state.last_refill = Instant::now();
        state.window_requests.clear();
        state.queue_size = 0.0;
        state.last_drain = Instant::now();
        state.operation_states.clear();

        self.inner.total_allowed.store(0, Ordering::Relaxed);
        self.inner.total_denied.store(0, Ordering::Relaxed);
    }

    /// Check if rate limiting is enabled
    pub fn is_enabled(&self) -> bool {
        self.inner.config.enabled
    }
}

impl Clone for RateLimiter {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Statistics for rate limiting
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    /// Total number of allowed requests
    pub total_allowed: u64,

    /// Total number of denied requests
    pub total_denied: u64,

    /// Current available tokens
    pub current_tokens: usize,

    /// Configured requests per second
    pub requests_per_second: f64,

    /// Configured burst size
    pub burst_size: usize,
}

impl RateLimitStats {
    /// Calculate the denial rate
    pub fn denial_rate(&self) -> f64 {
        let total = self.total_allowed + self.total_denied;
        if total == 0 {
            0.0
        } else {
            self.total_denied as f64 / total as f64
        }
    }
}

/// A rate limiter that can have different limits per key (e.g., per tenant)
pub struct KeyedRateLimiter {
    default_config: RateLimitConfig,
    limiters: RwLock<HashMap<String, RateLimiter>>,
}

impl KeyedRateLimiter {
    /// Create a new keyed rate limiter
    pub fn new(default_config: RateLimitConfig) -> Self {
        Self {
            default_config,
            limiters: RwLock::new(HashMap::new()),
        }
    }

    /// Try to acquire for a specific key
    pub fn try_acquire(&self, key: &str) -> bool {
        let Ok(limiters) = self.limiters.read() else { return false; };

        if let Some(limiter) = limiters.get(key) {
            limiter.try_acquire()
        } else {
            drop(limiters);

            // Create new limiter
            let Ok(mut limiters) = self.limiters.write() else { return false; };
            let limiter = limiters
                .entry(key.to_string())
                .or_insert_with(|| RateLimiter::new(self.default_config.clone()));
            limiter.try_acquire()
        }
    }

    /// Set a custom rate limit for a specific key
    pub fn set_limit(&self, key: &str, requests_per_second: f64) {
        let Ok(mut limiters) = self.limiters.write() else { return; };
        let config = RateLimitConfig {
            requests_per_second,
            ..self.default_config.clone()
        };
        limiters.insert(key.to_string(), RateLimiter::new(config));
    }

    /// Get stats for all keys
    pub fn all_stats(&self) -> HashMap<String, RateLimitStats> {
        let Ok(limiters) = self.limiters.read() else { return HashMap::new(); };
        limiters
            .iter()
            .map(|(k, v)| (k.clone(), v.stats()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_token_bucket_basic() {
        let limiter = RateLimiter::new(RateLimitConfig {
            requests_per_second: 10.0,
            burst_size: 5,
            algorithm: RateLimitAlgorithm::TokenBucket,
            enabled: true,
            ..Default::default()
        });

        // Should allow burst
        for _ in 0..5 {
            assert!(limiter.try_acquire(), "Should allow burst");
        }

        // Should deny after burst exhausted
        assert!(!limiter.try_acquire(), "Should deny after burst");
    }

    #[test]
    fn test_token_bucket_refill() {
        let limiter = RateLimiter::new(RateLimitConfig {
            requests_per_second: 100.0,
            burst_size: 5,
            algorithm: RateLimitAlgorithm::TokenBucket,
            enabled: true,
            ..Default::default()
        });

        // Exhaust burst
        for _ in 0..5 {
            limiter.try_acquire();
        }

        // Wait for refill
        thread::sleep(Duration::from_millis(100));

        // Should be able to acquire again
        assert!(limiter.try_acquire());
    }

    #[test]
    fn test_sliding_window() {
        let limiter = RateLimiter::new(RateLimitConfig {
            requests_per_second: 10.0,
            burst_size: 10,
            algorithm: RateLimitAlgorithm::SlidingWindow,
            enabled: true,
            ..Default::default()
        });

        // Should allow up to rate limit
        for _ in 0..10 {
            assert!(limiter.try_acquire());
        }

        // Should deny after limit
        assert!(!limiter.try_acquire());
    }

    #[test]
    fn test_leaky_bucket() {
        // Use extremely low rate so bucket doesn't drain during test
        // 1e-9 RPS means the bucket would take ~31 years to drain one item
        let limiter = RateLimiter::new(RateLimitConfig {
            requests_per_second: 1e-9,
            burst_size: 5,
            algorithm: RateLimitAlgorithm::LeakyBucket,
            enabled: true,
            ..Default::default()
        });

        // Should allow up to burst size
        for _ in 0..5 {
            assert!(limiter.try_acquire());
        }

        // Should deny when bucket full
        assert!(!limiter.try_acquire());
    }

    #[test]
    fn test_disabled() {
        let limiter = RateLimiter::new(RateLimitConfig {
            requests_per_second: 1.0,
            burst_size: 1,
            enabled: false,
            ..Default::default()
        });

        // Should always allow when disabled
        for _ in 0..100 {
            assert!(limiter.try_acquire());
        }
    }

    #[test]
    fn test_operation_limits() {
        // Use very low rates so tokens don't refill during test
        let mut config = RateLimitConfig {
            requests_per_second: 0.001,
            burst_size: 10,
            enabled: true,
            ..Default::default()
        };
        config.operation_limits.insert("query".to_string(), 0.001);
        config.operation_limits.insert("write".to_string(), 0.001);

        let limiter = RateLimiter::new(config);

        // Query uses burst_size (10)
        for _ in 0..10 {
            assert!(limiter.try_acquire_for("query"));
        }
        assert!(!limiter.try_acquire_for("query"));

        // Write also uses burst_size (10) - each operation is independent
        for _ in 0..10 {
            assert!(limiter.try_acquire_for("write"));
        }
        assert!(!limiter.try_acquire_for("write"));
    }

    #[test]
    fn test_keyed_rate_limiter() {
        let limiter = KeyedRateLimiter::new(RateLimitConfig {
            requests_per_second: 10.0,
            burst_size: 3,
            enabled: true,
            ..Default::default()
        });

        // Different keys have independent limits
        for _ in 0..3 {
            assert!(limiter.try_acquire("tenant_a"));
            assert!(limiter.try_acquire("tenant_b"));
        }

        assert!(!limiter.try_acquire("tenant_a"));
        assert!(!limiter.try_acquire("tenant_b"));
    }

    #[test]
    fn test_stats() {
        let limiter = RateLimiter::new(RateLimitConfig {
            requests_per_second: 10.0,
            burst_size: 5,
            enabled: true,
            ..Default::default()
        });

        for _ in 0..7 {
            limiter.try_acquire();
        }

        let stats = limiter.stats();
        assert_eq!(stats.total_allowed, 5);
        assert_eq!(stats.total_denied, 2);
        assert!(stats.denial_rate() > 0.2);
    }

    #[test]
    fn test_with_rate() {
        let limiter = RateLimiter::with_rate(100.0);
        assert!(limiter.try_acquire());
    }

    #[test]
    fn test_result_details() {
        let limiter = RateLimiter::new(RateLimitConfig {
            requests_per_second: 10.0,
            burst_size: 3,
            enabled: true,
            ..Default::default()
        });

        let result = limiter.try_acquire_result();
        assert!(result.allowed);
        assert_eq!(result.remaining, 2);
    }
}

// ============================================================================
// Additional Rate Limiting Types (for API compatibility)
// ============================================================================

/// Scope for rate limiting (global, per-tenant, per-operation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RateLimitScope {
    /// Global rate limit across all requests
    Global,
    /// Per-tenant rate limiting
    Tenant,
    /// Per-operation rate limiting
    Operation,
    /// Per-IP rate limiting
    Ip,
    /// Custom scope
    Custom,
}

impl Default for RateLimitScope {
    fn default() -> Self {
        Self::Global
    }
}

/// Multi-tier rate limiter that applies limits at different scopes
///
/// For example, you might want:
/// - 1000 req/s global limit
/// - 100 req/s per-tenant limit
/// - 10 req/s per-operation limit
pub struct MultiTierRateLimiter {
    /// Global limiter
    global: RateLimiter,
    /// Per-key limiters (tenant, IP, etc.)
    keyed: KeyedRateLimiter,
    /// Per-operation limiters
    operations: KeyedRateLimiter,
    /// Configuration
    #[allow(dead_code)]
    config: MultiTierConfig,
}

/// Configuration for multi-tier rate limiting
#[derive(Debug, Clone)]
pub struct MultiTierConfig {
    /// Global rate limit
    pub global_rps: f64,
    /// Per-key rate limit
    pub per_key_rps: f64,
    /// Per-operation rate limit
    pub per_operation_rps: f64,
    /// Burst size multiplier
    pub burst_multiplier: f64,
}

impl Default for MultiTierConfig {
    fn default() -> Self {
        Self {
            global_rps: 10000.0,
            per_key_rps: 1000.0,
            per_operation_rps: 100.0,
            burst_multiplier: 0.1,
        }
    }
}

impl MultiTierRateLimiter {
    /// Create a new multi-tier rate limiter
    pub fn new(config: MultiTierConfig) -> Self {
        let burst = |rps: f64| -> usize {
            (rps * config.burst_multiplier).max(1.0) as usize
        };

        Self {
            global: RateLimiter::new(RateLimitConfig {
                requests_per_second: config.global_rps,
                burst_size: burst(config.global_rps),
                ..Default::default()
            }),
            keyed: KeyedRateLimiter::new(RateLimitConfig {
                requests_per_second: config.per_key_rps,
                burst_size: burst(config.per_key_rps),
                ..Default::default()
            }),
            operations: KeyedRateLimiter::new(RateLimitConfig {
                requests_per_second: config.per_operation_rps,
                burst_size: burst(config.per_operation_rps),
                ..Default::default()
            }),
            config,
        }
    }

    /// Check if a request is allowed
    ///
    /// Checks all applicable limits in order: global -> key -> operation
    pub fn try_acquire(&self, key: Option<&str>, operation: Option<&str>) -> bool {
        // Check global limit first
        if !self.global.try_acquire() {
            return false;
        }

        // Check per-key limit if key is provided
        if let Some(k) = key {
            if !self.keyed.try_acquire(k) {
                return false;
            }
        }

        // Check per-operation limit if operation is provided
        if let Some(op) = operation {
            if !self.operations.try_acquire(op) {
                return false;
            }
        }

        true
    }

    /// Get stats for all tiers
    pub fn stats(&self) -> MultiTierStats {
        MultiTierStats {
            global: self.global.stats(),
            per_key: self.keyed.all_stats(),
            per_operation: self.operations.all_stats(),
        }
    }
}

/// Statistics for multi-tier rate limiting
#[derive(Debug, Clone)]
pub struct MultiTierStats {
    /// Global tier stats
    pub global: RateLimitStats,
    /// Per-key tier stats
    pub per_key: HashMap<String, RateLimitStats>,
    /// Per-operation tier stats
    pub per_operation: HashMap<String, RateLimitStats>,
}

#[cfg(test)]
mod multi_tier_tests {
    use super::*;

    #[test]
    fn test_multi_tier_basic() {
        let limiter = MultiTierRateLimiter::new(MultiTierConfig {
            global_rps: 100.0,
            per_key_rps: 10.0,
            per_operation_rps: 5.0,
            burst_multiplier: 0.5,
        });

        // Should allow requests
        assert!(limiter.try_acquire(Some("tenant1"), Some("query")));
        assert!(limiter.try_acquire(Some("tenant1"), Some("query")));
    }

    #[test]
    fn test_multi_tier_per_key_limit() {
        let limiter = MultiTierRateLimiter::new(MultiTierConfig {
            global_rps: 1000.0,
            per_key_rps: 10.0,
            per_operation_rps: 100.0,
            burst_multiplier: 0.5,
        });

        // Per-key limit is 10 * 0.5 = 5
        for _ in 0..5 {
            assert!(limiter.try_acquire(Some("tenant1"), None));
        }

        // Should be rate limited for tenant1
        assert!(!limiter.try_acquire(Some("tenant1"), None));

        // But tenant2 should still work
        assert!(limiter.try_acquire(Some("tenant2"), None));
    }

    #[test]
    fn test_rate_limit_scope() {
        assert_eq!(RateLimitScope::default(), RateLimitScope::Global);
    }
}
