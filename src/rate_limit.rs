//! Rate Limiting and Throttling
//!
//! Provides protection against overload and abuse by limiting request rates
//! using multiple algorithms (token bucket, sliding window, fixed window).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Rate limit algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    /// Token bucket algorithm - smooth rate limiting with burst support
    TokenBucket,
    /// Sliding window - more accurate but higher memory usage
    SlidingWindow,
    /// Fixed window - simple and efficient but can have edge case bursts
    FixedWindow,
}

/// Rate limit scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RateLimitScope {
    /// Global limit across all requests
    Global,
    /// Per-user limit
    PerUser,
    /// Per-IP address limit
    PerIP,
    /// Per-API key limit
    PerAPIKey,
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests allowed
    pub max_requests: u32,

    /// Time window for the limit
    pub window: Duration,

    /// Algorithm to use
    pub algorithm: RateLimitAlgorithm,

    /// Scope of the limit
    pub scope: RateLimitScope,

    /// Allow bursts (for token bucket)
    pub allow_burst: bool,

    /// Burst size (for token bucket)
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window: Duration::from_secs(60),
            algorithm: RateLimitAlgorithm::TokenBucket,
            scope: RateLimitScope::Global,
            allow_burst: true,
            burst_size: 10,
        }
    }
}

impl RateLimitConfig {
    /// Create a per-second rate limit
    pub fn per_second(requests: u32) -> Self {
        Self {
            max_requests: requests,
            window: Duration::from_secs(1),
            ..Default::default()
        }
    }

    /// Create a per-minute rate limit
    pub fn per_minute(requests: u32) -> Self {
        Self {
            max_requests: requests,
            window: Duration::from_secs(60),
            ..Default::default()
        }
    }

    /// Create a per-hour rate limit
    pub fn per_hour(requests: u32) -> Self {
        Self {
            max_requests: requests,
            window: Duration::from_secs(3600),
            ..Default::default()
        }
    }
}

/// Rate limit result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitResult {
    /// Whether the request is allowed
    pub allowed: bool,

    /// Remaining requests in the current window
    pub remaining: u32,

    /// Total limit
    pub limit: u32,

    /// Time until the window resets
    pub reset_after: Duration,

    /// Time to wait before retry (if rate limited)
    pub retry_after: Option<Duration>,
}

/// Token bucket state
#[derive(Debug, Clone)]
struct TokenBucketState {
    tokens: f64,
    last_update: Instant,
    capacity: u32,
    refill_rate: f64, // tokens per second
}

impl TokenBucketState {
    fn new(capacity: u32, refill_rate: f64) -> Self {
        Self {
            tokens: capacity as f64,
            last_update: Instant::now(),
            capacity,
            refill_rate,
        }
    }

    fn try_consume(&mut self, tokens: u32) -> bool {
        self.refill();

        if self.tokens >= tokens as f64 {
            self.tokens -= tokens as f64;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        let new_tokens = elapsed * self.refill_rate;

        self.tokens = (self.tokens + new_tokens).min(self.capacity as f64);
        self.last_update = now;
    }

    fn remaining(&self) -> u32 {
        self.tokens.floor() as u32
    }

    fn time_until_available(&self, tokens: u32) -> Duration {
        if self.tokens >= tokens as f64 {
            return Duration::from_secs(0);
        }

        let needed = tokens as f64 - self.tokens;
        let secs = needed / self.refill_rate;
        Duration::from_secs_f64(secs)
    }
}

/// Sliding window state
#[derive(Debug, Clone)]
struct SlidingWindowState {
    requests: Vec<Instant>,
    window: Duration,
    max_requests: u32,
}

impl SlidingWindowState {
    fn new(window: Duration, max_requests: u32) -> Self {
        Self {
            requests: Vec::new(),
            window,
            max_requests,
        }
    }

    fn try_record(&mut self) -> bool {
        self.cleanup();

        if self.requests.len() < self.max_requests as usize {
            self.requests.push(Instant::now());
            true
        } else {
            false
        }
    }

    fn cleanup(&mut self) {
        let now = Instant::now();
        self.requests
            .retain(|&time| now.duration_since(time) < self.window);
    }

    fn remaining(&self) -> u32 {
        self.max_requests.saturating_sub(self.requests.len() as u32)
    }

    fn time_until_available(&self) -> Duration {
        if self.requests.len() < self.max_requests as usize {
            return Duration::from_secs(0);
        }

        if let Some(&oldest) = self.requests.first() {
            let elapsed = Instant::now().duration_since(oldest);
            self.window.saturating_sub(elapsed)
        } else {
            Duration::from_secs(0)
        }
    }
}

/// Fixed window state
#[derive(Debug, Clone)]
struct FixedWindowState {
    count: u32,
    window_start: Instant,
    window: Duration,
    max_requests: u32,
}

impl FixedWindowState {
    fn new(window: Duration, max_requests: u32) -> Self {
        Self {
            count: 0,
            window_start: Instant::now(),
            window,
            max_requests,
        }
    }

    fn try_record(&mut self) -> bool {
        self.maybe_reset();

        if self.count < self.max_requests {
            self.count += 1;
            true
        } else {
            false
        }
    }

    fn maybe_reset(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.window_start) >= self.window {
            self.count = 0;
            self.window_start = now;
        }
    }

    fn remaining(&self) -> u32 {
        self.max_requests.saturating_sub(self.count)
    }

    fn time_until_reset(&self) -> Duration {
        let elapsed = Instant::now().duration_since(self.window_start);
        self.window.saturating_sub(elapsed)
    }
}

/// Rate limiter state for a single key
#[derive(Debug, Clone)]
enum LimiterState {
    TokenBucket(TokenBucketState),
    SlidingWindow(SlidingWindowState),
    FixedWindow(FixedWindowState),
}

/// Rate limiter
pub struct RateLimiter {
    config: RateLimitConfig,
    states: Arc<Mutex<HashMap<String, LimiterState>>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            states: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }

    /// Check if a request is allowed
    pub fn check(&self, key: &str) -> RateLimitResult {
        let mut states = self.states.lock().unwrap();

        let state = states
            .entry(key.to_string())
            .or_insert_with(|| self.create_state());

        let allowed = self.try_consume(state);
        let remaining = self.get_remaining(state);
        let reset_after = self.get_reset_time(state);

        let retry_after = if !allowed {
            Some(self.get_retry_time(state))
        } else {
            None
        };

        RateLimitResult {
            allowed,
            remaining,
            limit: self.config.max_requests,
            reset_after,
            retry_after,
        }
    }

    /// Record a request (always consumes, doesn't check limit)
    pub fn record(&self, key: &str) {
        let mut states = self.states.lock().unwrap();

        let state = states
            .entry(key.to_string())
            .or_insert_with(|| self.create_state());

        let _ = self.try_consume(state);
    }

    /// Reset limits for a specific key
    pub fn reset(&self, key: &str) {
        let mut states = self.states.lock().unwrap();
        states.remove(key);
    }

    /// Reset all limits
    pub fn reset_all(&self) {
        let mut states = self.states.lock().unwrap();
        states.clear();
    }

    /// Get remaining requests for a key
    pub fn get_remaining_requests(&self, key: &str) -> u32 {
        let states = self.states.lock().unwrap();

        if let Some(state) = states.get(key) {
            self.get_remaining(state)
        } else {
            self.config.max_requests
        }
    }

    fn create_state(&self) -> LimiterState {
        match self.config.algorithm {
            RateLimitAlgorithm::TokenBucket => {
                let capacity = if self.config.allow_burst {
                    self.config.max_requests + self.config.burst_size
                } else {
                    self.config.max_requests
                };
                let refill_rate =
                    self.config.max_requests as f64 / self.config.window.as_secs_f64();
                LimiterState::TokenBucket(TokenBucketState::new(capacity, refill_rate))
            }
            RateLimitAlgorithm::SlidingWindow => LimiterState::SlidingWindow(
                SlidingWindowState::new(self.config.window, self.config.max_requests),
            ),
            RateLimitAlgorithm::FixedWindow => LimiterState::FixedWindow(FixedWindowState::new(
                self.config.window,
                self.config.max_requests,
            )),
        }
    }

    fn try_consume(&self, state: &mut LimiterState) -> bool {
        match state {
            LimiterState::TokenBucket(s) => s.try_consume(1),
            LimiterState::SlidingWindow(s) => s.try_record(),
            LimiterState::FixedWindow(s) => s.try_record(),
        }
    }

    fn get_remaining(&self, state: &LimiterState) -> u32 {
        match state {
            LimiterState::TokenBucket(s) => s.remaining(),
            LimiterState::SlidingWindow(s) => s.remaining(),
            LimiterState::FixedWindow(s) => s.remaining(),
        }
    }

    fn get_reset_time(&self, state: &LimiterState) -> Duration {
        match state {
            LimiterState::TokenBucket(_) => self.config.window,
            LimiterState::SlidingWindow(s) => s.time_until_available(),
            LimiterState::FixedWindow(s) => s.time_until_reset(),
        }
    }

    fn get_retry_time(&self, state: &LimiterState) -> Duration {
        match state {
            LimiterState::TokenBucket(s) => s.time_until_available(1),
            LimiterState::SlidingWindow(s) => s.time_until_available(),
            LimiterState::FixedWindow(s) => s.time_until_reset(),
        }
    }
}

/// Multi-tier rate limiter
pub struct MultiTierRateLimiter {
    limiters: Vec<(String, RateLimiter)>,
}

impl MultiTierRateLimiter {
    /// Create a new multi-tier rate limiter
    pub fn new() -> Self {
        Self {
            limiters: Vec::new(),
        }
    }

    /// Add a rate limiter tier
    pub fn add_tier(&mut self, name: impl Into<String>, limiter: RateLimiter) {
        self.limiters.push((name.into(), limiter));
    }

    /// Check all tiers and return the most restrictive result
    pub fn check(&self, key: &str) -> RateLimitResult {
        let mut most_restrictive = RateLimitResult {
            allowed: true,
            remaining: u32::MAX,
            limit: u32::MAX,
            reset_after: Duration::from_secs(0),
            retry_after: None,
        };

        for (_, limiter) in &self.limiters {
            let result = limiter.check(key);

            if !result.allowed {
                most_restrictive.allowed = false;
                if let Some(retry) = result.retry_after {
                    if let Some(current_retry) = most_restrictive.retry_after {
                        most_restrictive.retry_after = Some(retry.max(current_retry));
                    } else {
                        most_restrictive.retry_after = Some(retry);
                    }
                }
            }

            most_restrictive.remaining = most_restrictive.remaining.min(result.remaining);
        }

        most_restrictive
    }

    /// Reset all tiers for a key
    pub fn reset(&self, key: &str) {
        for (_, limiter) in &self.limiters {
            limiter.reset(key);
        }
    }

    /// Reset all tiers for all keys
    pub fn reset_all(&self) {
        for (_, limiter) in &self.limiters {
            limiter.reset_all();
        }
    }
}

impl Default for MultiTierRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_token_bucket_basic() {
        let config = RateLimitConfig {
            max_requests: 10,
            window: Duration::from_secs(1),
            algorithm: RateLimitAlgorithm::TokenBucket,
            allow_burst: false, // Disable burst for this test
            ..Default::default()
        };

        let limiter = RateLimiter::new(config);

        // First 10 requests should succeed
        for _ in 0..10 {
            let result = limiter.check("user1");
            assert!(result.allowed);
        }

        // 11th request should fail
        let result = limiter.check("user1");
        assert!(!result.allowed);
        assert!(result.retry_after.is_some());
    }

    #[test]
    fn test_token_bucket_refill() {
        let config = RateLimitConfig {
            max_requests: 5,
            window: Duration::from_secs(1),
            algorithm: RateLimitAlgorithm::TokenBucket,
            allow_burst: false,
            ..Default::default()
        };

        let limiter = RateLimiter::new(config);

        // Consume all tokens
        for _ in 0..5 {
            assert!(limiter.check("user1").allowed);
        }

        // Should be rate limited
        assert!(!limiter.check("user1").allowed);

        // Wait for refill
        thread::sleep(Duration::from_millis(250)); // 25% of window

        // Should have ~1 token now
        assert!(limiter.check("user1").allowed);
    }

    #[test]
    fn test_sliding_window() {
        let config = RateLimitConfig {
            max_requests: 5,
            window: Duration::from_millis(500),
            algorithm: RateLimitAlgorithm::SlidingWindow,
            ..Default::default()
        };

        let limiter = RateLimiter::new(config);

        // First 5 requests should succeed
        for _ in 0..5 {
            assert!(limiter.check("user1").allowed);
        }

        // 6th should fail
        assert!(!limiter.check("user1").allowed);

        // Wait for window to slide
        thread::sleep(Duration::from_millis(600));

        // Should work again
        assert!(limiter.check("user1").allowed);
    }

    #[test]
    fn test_fixed_window() {
        let config = RateLimitConfig {
            max_requests: 5,
            window: Duration::from_millis(500),
            algorithm: RateLimitAlgorithm::FixedWindow,
            ..Default::default()
        };

        let limiter = RateLimiter::new(config);

        // First 5 requests should succeed
        for _ in 0..5 {
            assert!(limiter.check("user1").allowed);
        }

        // 6th should fail
        assert!(!limiter.check("user1").allowed);

        // Wait for window reset
        thread::sleep(Duration::from_millis(600));

        // Should work again
        assert!(limiter.check("user1").allowed);
    }

    #[test]
    fn test_per_user_isolation() {
        let mut config = RateLimitConfig::per_second(5);
        config.allow_burst = false; // Disable burst
        let limiter = RateLimiter::new(config);

        // User1 consumes their limit
        for _ in 0..5 {
            assert!(limiter.check("user1").allowed);
        }
        assert!(!limiter.check("user1").allowed);

        // User2 should still have full quota
        for _ in 0..5 {
            assert!(limiter.check("user2").allowed);
        }
    }

    #[test]
    fn test_reset() {
        let mut config = RateLimitConfig::per_second(3);
        config.allow_burst = false;
        let limiter = RateLimiter::new(config);

        // Consume limit
        for _ in 0..3 {
            limiter.record("user1");
        }
        assert!(!limiter.check("user1").allowed);

        // Reset
        limiter.reset("user1");

        // Should work again
        assert!(limiter.check("user1").allowed);
    }

    #[test]
    fn test_remaining_requests() {
        let mut config = RateLimitConfig::per_second(10);
        config.allow_burst = false;
        let limiter = RateLimiter::new(config);

        // Initial remaining should be max_requests
        let initial = limiter.get_remaining_requests("user1");
        assert_eq!(initial, 10);

        // After recording one request
        limiter.record("user1");
        let remaining = limiter.get_remaining_requests("user1");
        assert!(remaining < initial);
    }

    #[test]
    fn test_multi_tier() {
        let mut multi = MultiTierRateLimiter::new();

        // Add per-second tier
        let mut config1 = RateLimitConfig::per_second(5);
        config1.allow_burst = false;
        multi.add_tier("per_second", RateLimiter::new(config1));

        // Add per-minute tier
        let mut config2 = RateLimitConfig::per_minute(20);
        config2.allow_burst = false;
        multi.add_tier("per_minute", RateLimiter::new(config2));

        // First 5 requests should succeed
        for _ in 0..5 {
            let result = multi.check("user1");
            assert!(result.allowed);
        }

        // 6th should fail (per-second limit)
        let result = multi.check("user1");
        assert!(!result.allowed);
    }

    #[test]
    fn test_burst_mode() {
        let config = RateLimitConfig {
            max_requests: 10,
            window: Duration::from_secs(1),
            algorithm: RateLimitAlgorithm::TokenBucket,
            allow_burst: true,
            burst_size: 5,
            ..Default::default()
        };

        let limiter = RateLimiter::new(config);

        // Should allow up to max_requests + burst_size
        for _ in 0..15 {
            let result = limiter.check("user1");
            assert!(result.allowed);
        }

        // 16th should fail
        assert!(!limiter.check("user1").allowed);
    }

    #[test]
    fn test_helper_constructors() {
        let per_sec = RateLimitConfig::per_second(100);
        assert_eq!(per_sec.window, Duration::from_secs(1));

        let per_min = RateLimitConfig::per_minute(1000);
        assert_eq!(per_min.window, Duration::from_secs(60));

        let per_hour = RateLimitConfig::per_hour(10000);
        assert_eq!(per_hour.window, Duration::from_secs(3600));
    }
}
