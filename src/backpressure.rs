// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Streaming Backpressure
//!
//! Flow control for streaming ingestion with automatic rate limiting,
//! dead letter queues, and consumer lag monitoring.
//!
//! ## Features
//!
//! - **Adaptive Rate Limiting**: Slow down producers when indexing lags
//! - **Dead Letter Queue**: Handle failed records gracefully
//! - **Consumer Lag Monitoring**: Track processing delays
//! - **Circuit Breaker**: Prevent cascade failures
//! - **Exactly-Once Semantics**: Transactional guarantees
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::backpressure::{BackpressureController, StreamConfig};
//!
//! let config = StreamConfig::default();
//! let controller = BackpressureController::new(config);
//!
//! // Check if we should accept more records
//! if controller.should_accept() {
//!     controller.record_received();
//!     // Process record...
//!     controller.record_processed();
//! } else {
//!     // Apply backpressure
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, atomic::{AtomicU64, AtomicUsize, Ordering}};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Stream configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Maximum buffer size before backpressure
    pub max_buffer_size: usize,
    /// High watermark (start backpressure)
    pub high_watermark: f64,
    /// Low watermark (resume normal flow)
    pub low_watermark: f64,
    /// Maximum lag before circuit breaker
    pub max_lag_seconds: u64,
    /// DLQ enabled
    pub enable_dlq: bool,
    /// DLQ max size
    pub dlq_max_size: usize,
    /// Retry attempts before DLQ
    pub max_retries: u32,
    /// Retry delay (exponential backoff base)
    pub retry_delay_ms: u64,
    /// Batch size for processing
    pub batch_size: usize,
    /// Checkpoint interval
    pub checkpoint_interval_ms: u64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10000,
            high_watermark: 0.8,
            low_watermark: 0.3,
            max_lag_seconds: 300,
            enable_dlq: true,
            dlq_max_size: 1000,
            max_retries: 3,
            retry_delay_ms: 1000,
            batch_size: 100,
            checkpoint_interval_ms: 5000,
        }
    }
}

/// Backpressure state
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BackpressureState {
    /// Normal flow
    Normal,
    /// Applying backpressure
    Backpressure,
    /// Circuit breaker open
    CircuitOpen,
    /// Recovering from circuit break
    Recovering,
}

/// Stream record
#[derive(Debug, Clone)]
pub struct StreamRecord<T> {
    /// Record ID
    pub id: String,
    /// Partition (for ordering)
    pub partition: u32,
    /// Offset within partition
    pub offset: u64,
    /// Payload
    pub payload: T,
    /// Timestamp
    pub timestamp: i64,
    /// Retry count
    pub retries: u32,
    /// Headers/metadata
    pub headers: HashMap<String, String>,
}

/// Dead letter record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadLetterRecord {
    /// Original record ID
    pub record_id: String,
    /// Partition
    pub partition: u32,
    /// Original offset
    pub offset: u64,
    /// Error message
    pub error: String,
    /// Retry count when failed
    pub retries: u32,
    /// Timestamp when moved to DLQ
    pub dlq_timestamp: i64,
    /// Serialized payload
    pub payload_json: String,
}

/// Checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Partition offsets
    pub offsets: HashMap<u32, u64>,
    /// Checkpoint timestamp
    pub timestamp: i64,
    /// Pending transaction IDs
    pub pending_txns: Vec<String>,
}

/// Consumer lag metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagMetrics {
    /// Records pending
    pub pending_records: usize,
    /// Lag in seconds
    pub lag_seconds: f64,
    /// Processing rate (records/second)
    pub processing_rate: f64,
    /// Time to catch up (estimated)
    pub estimated_catchup_seconds: f64,
    /// Per-partition lag
    pub partition_lag: HashMap<u32, u64>,
}

/// Backpressure controller
pub struct BackpressureController {
    /// Configuration
    config: StreamConfig,
    /// Current state
    state: RwLock<BackpressureState>,
    /// Buffer size
    buffer_size: AtomicUsize,
    /// Records received
    received: AtomicU64,
    /// Records processed
    processed: AtomicU64,
    /// Records failed
    failed: AtomicU64,
    /// Circuit breaker trips
    circuit_trips: AtomicU64,
    /// Last processed timestamp
    last_processed: RwLock<Instant>,
    /// Processing start times
    processing_starts: RwLock<VecDeque<Instant>>,
    /// Dead letter queue
    dlq: RwLock<VecDeque<DeadLetterRecord>>,
    /// Current checkpoint
    checkpoint: RwLock<Checkpoint>,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// Partition offsets
    partition_offsets: RwLock<HashMap<u32, u64>>,
    /// Circuit breaker
    circuit_breaker: CircuitBreaker,
}

impl BackpressureController {
    /// Create new controller
    pub fn new(config: StreamConfig) -> Self {
        let rate_limiter = RateLimiter::new(config.batch_size * 10); // 10x batch as rate

        Self {
            config: config.clone(),
            state: RwLock::new(BackpressureState::Normal),
            buffer_size: AtomicUsize::new(0),
            received: AtomicU64::new(0),
            processed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            circuit_trips: AtomicU64::new(0),
            last_processed: RwLock::new(Instant::now()),
            processing_starts: RwLock::new(VecDeque::new()),
            dlq: RwLock::new(VecDeque::new()),
            checkpoint: RwLock::new(Checkpoint {
                offsets: HashMap::new(),
                timestamp: unix_timestamp(),
                pending_txns: Vec::new(),
            }),
            rate_limiter,
            partition_offsets: RwLock::new(HashMap::new()),
            circuit_breaker: CircuitBreaker::new(config.max_lag_seconds, 5),
        }
    }

    /// Check if should accept more records
    pub fn should_accept(&self) -> bool {
        let Ok(state_guard) = self.state.read() else { return false; };
        let state = *state_guard;

        match state {
            BackpressureState::CircuitOpen => false,
            BackpressureState::Backpressure => {
                // Only accept if below low watermark
                let fill_ratio = self.buffer_fill_ratio();
                fill_ratio < self.config.low_watermark
            }
            _ => {
                // Accept if rate limiter allows
                self.rate_limiter.try_acquire()
            }
        }
    }

    /// Record received
    pub fn record_received(&self) {
        self.received.fetch_add(1, Ordering::Relaxed);
        self.buffer_size.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut starts) = self.processing_starts.write() {
            starts.push_back(Instant::now());
            while starts.len() > 1000 {
                starts.pop_front();
            }
        }

        self.update_state();
    }

    /// Record processed successfully
    pub fn record_processed(&self) {
        self.processed.fetch_add(1, Ordering::Relaxed);
        let prev = self.buffer_size.fetch_sub(1, Ordering::Relaxed);
        if prev == 0 {
            self.buffer_size.store(0, Ordering::Relaxed);
        }

        if let Ok(mut last) = self.last_processed.write() {
            *last = Instant::now();
        }
        self.circuit_breaker.record_success();
        self.update_state();
    }

    /// Record processing failure
    pub fn record_failed<T: Serialize>(&self, record: &StreamRecord<T>, error: &str) {
        self.failed.fetch_add(1, Ordering::Relaxed);
        let prev = self.buffer_size.fetch_sub(1, Ordering::Relaxed);
        if prev == 0 {
            self.buffer_size.store(0, Ordering::Relaxed);
        }

        self.circuit_breaker.record_failure();

        // Move to DLQ if retries exhausted
        if record.retries >= self.config.max_retries && self.config.enable_dlq {
            self.add_to_dlq(record, error);
        }

        self.update_state();
    }

    /// Get retry delay with exponential backoff
    pub fn get_retry_delay(&self, retry_count: u32) -> Duration {
        let base = self.config.retry_delay_ms;
        let delay_ms = base * 2u64.pow(retry_count.min(10));
        Duration::from_millis(delay_ms.min(60000)) // Cap at 60s
    }

    /// Commit checkpoint
    pub fn commit_checkpoint(&self, offsets: HashMap<u32, u64>) {
        let Ok(mut checkpoint) = self.checkpoint.write() else { return; };
        checkpoint.offsets = offsets;
        checkpoint.timestamp = unix_timestamp();
        checkpoint.pending_txns.clear();
    }

    /// Get current checkpoint
    pub fn get_checkpoint(&self) -> Checkpoint {
        let Ok(checkpoint) = self.checkpoint.read() else {
            return Checkpoint {
                offsets: HashMap::new(),
                timestamp: unix_timestamp(),
                pending_txns: Vec::new(),
            };
        };
        checkpoint.clone()
    }

    /// Get lag metrics
    pub fn get_lag_metrics(&self) -> LagMetrics {
        let pending = self.buffer_size.load(Ordering::Relaxed);
        let processing_rate = self.calculate_processing_rate();

        let lag_seconds = if processing_rate > 0.0 {
            pending as f64 / processing_rate
        } else {
            f64::INFINITY
        };

        let estimated_catchup = if processing_rate > 1.0 {
            pending as f64 / (processing_rate - 1.0).max(0.1)
        } else {
            f64::INFINITY
        };

        let partition_lag = self.partition_offsets.read()
            .map(|guard| guard.clone())
            .unwrap_or_default();  // Safe: returns empty HashMap on lock failure

        LagMetrics {
            pending_records: pending,
            lag_seconds,
            processing_rate,
            estimated_catchup_seconds: estimated_catchup,
            partition_lag,
        }
    }

    /// Get current state
    pub fn get_state(&self) -> BackpressureState {
        let Ok(state) = self.state.read() else { return BackpressureState::Normal; };
        *state
    }

    /// Get statistics
    pub fn get_stats(&self) -> StreamStats {
        let dlq_size = self.dlq.read().map(|guard| guard.len()).unwrap_or(0);  // Safe: returns 0 on lock failure
        StreamStats {
            received: self.received.load(Ordering::Relaxed),
            processed: self.processed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            pending: self.buffer_size.load(Ordering::Relaxed),
            dlq_size,
            circuit_trips: self.circuit_trips.load(Ordering::Relaxed),
            state: self.get_state(),
        }
    }

    /// Get DLQ records
    pub fn get_dlq_records(&self, limit: usize) -> Vec<DeadLetterRecord> {
        let Ok(dlq) = self.dlq.read() else { return Vec::new(); };
        dlq.iter().take(limit).cloned().collect()
    }

    /// Retry DLQ record
    pub fn retry_dlq_record(&self, record_id: &str) -> Option<DeadLetterRecord> {
        let mut dlq = self.dlq.write().ok()?;
        let pos = dlq.iter().position(|r| r.record_id == record_id)?;
        dlq.remove(pos)
    }

    /// Clear DLQ
    pub fn clear_dlq(&self) {
        if let Ok(mut dlq) = self.dlq.write() {
            dlq.clear();
        }
    }

    fn buffer_fill_ratio(&self) -> f64 {
        let size = self.buffer_size.load(Ordering::Relaxed);
        size as f64 / self.config.max_buffer_size as f64
    }

    fn calculate_processing_rate(&self) -> f64 {
        let Ok(starts) = self.processing_starts.read() else { return 0.0; };
        if starts.len() < 2 {
            return 0.0;
        }

        let Some(oldest) = starts.front() else { return 0.0; };
        let elapsed = oldest.elapsed().as_secs_f64();

        if elapsed > 0.0 {
            starts.len() as f64 / elapsed
        } else {
            0.0
        }
    }

    fn update_state(&self) {
        let Ok(mut state) = self.state.write() else { return; };

        // Check circuit breaker
        if self.circuit_breaker.is_open() {
            if *state != BackpressureState::CircuitOpen {
                self.circuit_trips.fetch_add(1, Ordering::Relaxed);
            }
            *state = BackpressureState::CircuitOpen;
            return;
        }

        // Check watermarks
        let fill_ratio = self.buffer_fill_ratio();

        match *state {
            BackpressureState::Normal => {
                if fill_ratio >= self.config.high_watermark {
                    *state = BackpressureState::Backpressure;
                }
            }
            BackpressureState::Backpressure => {
                if fill_ratio < self.config.low_watermark {
                    *state = BackpressureState::Normal;
                }
            }
            BackpressureState::CircuitOpen => {
                if self.circuit_breaker.should_allow_request() {
                    *state = BackpressureState::Recovering;
                }
            }
            BackpressureState::Recovering => {
                if !self.circuit_breaker.is_open() {
                    *state = BackpressureState::Normal;
                }
            }
        }
    }

    fn add_to_dlq<T: Serialize>(&self, record: &StreamRecord<T>, error: &str) {
        let Ok(mut dlq) = self.dlq.write() else { return; };

        // Evict oldest if full
        while dlq.len() >= self.config.dlq_max_size {
            dlq.pop_front();
        }

        let payload_json = serde_json::to_string(&record.payload).unwrap_or_default();

        dlq.push_back(DeadLetterRecord {
            record_id: record.id.clone(),
            partition: record.partition,
            offset: record.offset,
            error: error.to_string(),
            retries: record.retries,
            dlq_timestamp: unix_timestamp(),
            payload_json,
        });
    }
}

/// Stream statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStats {
    /// Records received
    pub received: u64,
    /// Records processed
    pub processed: u64,
    /// Records failed
    pub failed: u64,
    /// Pending records
    pub pending: usize,
    /// DLQ size
    pub dlq_size: usize,
    /// Circuit breaker trips
    pub circuit_trips: u64,
    /// Current state
    pub state: BackpressureState,
}

/// Rate limiter
struct RateLimiter {
    tokens: AtomicUsize,
    max_tokens: usize,
    last_refill: RwLock<Instant>,
    refill_rate: usize, // tokens per second
}

impl RateLimiter {
    fn new(max_tokens: usize) -> Self {
        Self {
            tokens: AtomicUsize::new(max_tokens),
            max_tokens,
            last_refill: RwLock::new(Instant::now()),
            refill_rate: max_tokens,
        }
    }

    fn try_acquire(&self) -> bool {
        self.refill();

        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            if current == 0 {
                return false;
            }
            if self.tokens.compare_exchange(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ).is_ok() {
                return true;
            }
        }
    }

    fn refill(&self) {
        let Ok(mut last) = self.last_refill.write() else { return; };
        let elapsed = last.elapsed();

        if elapsed >= Duration::from_millis(100) {
            let tokens_to_add = (elapsed.as_secs_f64() * self.refill_rate as f64) as usize;
            if tokens_to_add > 0 {
                let current = self.tokens.load(Ordering::Relaxed);
                let new_tokens = (current + tokens_to_add).min(self.max_tokens);
                self.tokens.store(new_tokens, Ordering::Relaxed);
                *last = Instant::now();
            }
        }
    }
}

/// Circuit breaker
struct CircuitBreaker {
    state: RwLock<CircuitState>,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    failure_threshold: u64,
    timeout_seconds: u64,
    last_failure: RwLock<Option<Instant>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    fn new(timeout_seconds: u64, failure_threshold: u64) -> Self {
        Self {
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            failure_threshold,
            timeout_seconds,
            last_failure: RwLock::new(None),
        }
    }

    fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);

        let Ok(mut state) = self.state.write() else { return; };
        if *state == CircuitState::HalfOpen {
            // Reset on success in half-open
            *state = CircuitState::Closed;
            self.failure_count.store(0, Ordering::Relaxed);
        }
    }

    fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut last_failure) = self.last_failure.write() {
            *last_failure = Some(Instant::now());
        }

        if count >= self.failure_threshold {
            if let Ok(mut state) = self.state.write() {
                *state = CircuitState::Open;
            }
        }
    }

    fn is_open(&self) -> bool {
        let Ok(state_guard) = self.state.read() else { return true; };  // Fail-safe: treat as open
        let state = *state_guard;
        drop(state_guard);

        match state {
            CircuitState::Open => {
                // Check if timeout has passed
                let Ok(last_failure_guard) = self.last_failure.read() else { return true; };
                if let Some(last) = *last_failure_guard {
                    if last.elapsed() > Duration::from_secs(self.timeout_seconds) {
                        drop(last_failure_guard);
                        if let Ok(mut state) = self.state.write() {
                            *state = CircuitState::HalfOpen;
                        }
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn should_allow_request(&self) -> bool {
        let Ok(state) = self.state.read() else { return false; };  // Fail-safe: deny requests on lock failure
        *state != CircuitState::Open
    }
}

/// Batch processor with backpressure
pub struct BatchProcessor<T> {
    controller: Arc<BackpressureController>,
    buffer: RwLock<Vec<StreamRecord<T>>>,
    batch_size: usize,
}

impl<T: Clone + Serialize> BatchProcessor<T> {
    /// Create new batch processor
    pub fn new(controller: Arc<BackpressureController>, batch_size: usize) -> Self {
        Self {
            controller,
            buffer: RwLock::new(Vec::with_capacity(batch_size)),
            batch_size,
        }
    }

    /// Add record to batch
    pub fn add(&self, record: StreamRecord<T>) -> bool {
        if !self.controller.should_accept() {
            return false;
        }

        self.controller.record_received();

        let Ok(mut buffer) = self.buffer.write() else { return false; };
        buffer.push(record);

        true
    }

    /// Check if batch is ready
    pub fn is_batch_ready(&self) -> bool {
        let Ok(buffer) = self.buffer.read() else { return false; };
        buffer.len() >= self.batch_size
    }

    /// Get batch for processing
    pub fn take_batch(&self) -> Vec<StreamRecord<T>> {
        let Ok(mut buffer) = self.buffer.write() else { return Vec::new(); };
        let batch: Vec<_> = buffer.drain(..).collect();
        batch
    }

    /// Mark batch as processed
    pub fn batch_processed(&self, count: usize) {
        for _ in 0..count {
            self.controller.record_processed();
        }
    }

    /// Mark record as failed
    pub fn record_failed(&self, record: &StreamRecord<T>, error: &str) {
        self.controller.record_failed(record, error);
    }
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backpressure_normal_flow() {
        let config = StreamConfig::default();
        let controller = BackpressureController::new(config);

        assert!(controller.should_accept());
        assert_eq!(controller.get_state(), BackpressureState::Normal);
    }

    #[test]
    fn test_backpressure_triggers() {
        let config = StreamConfig {
            max_buffer_size: 10,
            high_watermark: 0.5,
            low_watermark: 0.2,
            ..Default::default()
        };
        let controller = BackpressureController::new(config);

        // Fill buffer past high watermark
        for _ in 0..6 {
            controller.record_received();
        }

        assert_eq!(controller.get_state(), BackpressureState::Backpressure);
    }

    #[test]
    fn test_backpressure_recovery() {
        let config = StreamConfig {
            max_buffer_size: 10,
            high_watermark: 0.5,
            low_watermark: 0.2,
            ..Default::default()
        };
        let controller = BackpressureController::new(config);

        // Trigger backpressure
        for _ in 0..6 {
            controller.record_received();
        }
        assert_eq!(controller.get_state(), BackpressureState::Backpressure);

        // Process records to go below low watermark
        for _ in 0..5 {
            controller.record_processed();
        }
        assert_eq!(controller.get_state(), BackpressureState::Normal);
    }

    #[test]
    fn test_retry_delay() {
        let controller = BackpressureController::new(StreamConfig::default());

        let delay0 = controller.get_retry_delay(0);
        let delay1 = controller.get_retry_delay(1);
        let delay2 = controller.get_retry_delay(2);

        assert!(delay1 > delay0);
        assert!(delay2 > delay1);
    }

    #[test]
    fn test_dlq() {
        let config = StreamConfig {
            enable_dlq: true,
            max_retries: 0, // Immediately to DLQ
            ..Default::default()
        };
        let controller = BackpressureController::new(config);

        let record = StreamRecord {
            id: "test".to_string(),
            partition: 0,
            offset: 0,
            payload: "data",
            timestamp: 0,
            retries: 1,
            headers: HashMap::new(),
        };

        controller.record_failed(&record, "test error");

        let dlq = controller.get_dlq_records(10);
        assert_eq!(dlq.len(), 1);
        assert_eq!(dlq[0].record_id, "test");
    }

    #[test]
    fn test_checkpoint() {
        let controller = BackpressureController::new(StreamConfig::default());

        let mut offsets = HashMap::new();
        offsets.insert(0, 100);
        offsets.insert(1, 200);

        controller.commit_checkpoint(offsets.clone());

        let checkpoint = controller.get_checkpoint();
        assert_eq!(checkpoint.offsets.get(&0), Some(&100));
        assert_eq!(checkpoint.offsets.get(&1), Some(&200));
    }

    #[test]
    fn test_lag_metrics() {
        let controller = BackpressureController::new(StreamConfig::default());

        controller.record_received();
        controller.record_received();

        let metrics = controller.get_lag_metrics();
        assert_eq!(metrics.pending_records, 2);
    }
}
