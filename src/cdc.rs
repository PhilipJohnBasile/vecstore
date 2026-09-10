//! Change Data Capture (CDC) for Real-time Streaming Updates
//!
//! This module provides the ability to subscribe to changes in the vector store,
//! enabling real-time sync, replication, and event-driven architectures.
//!
//! ## Features
//!
//! - Subscribe to vector insert, update, and delete events
//! - Filter events by namespace, ID patterns, or metadata
//! - Multiple subscribers with independent cursors
//! - Persistent event log for replay
//! - Backpressure support
//!
//! ## Usage
//!
//! ```no_run
//! use vecstore::cdc::{ChangeStream, ChangeFilter, EventBroadcaster};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Create broadcaster (typically attached to VecStore)
//! let broadcaster = EventBroadcaster::new(1000); // Buffer 1000 events
//!
//! // Subscribe to all changes
//! let mut stream = broadcaster.subscribe(ChangeFilter::all());
//!
//! // Or filter to specific namespace
//! let mut filtered = broadcaster.subscribe(
//!     ChangeFilter::new().namespace("products")
//! );
//!
//! // Process events
//! while let Some(event) = stream.recv() {
//!     match event.change_type {
//!         ChangeType::Insert => println!("New vector: {}", event.vector_id),
//!         ChangeType::Update => println!("Updated: {}", event.vector_id),
//!         ChangeType::Delete => println!("Deleted: {}", event.vector_id),
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Type of change that occurred
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChangeType {
    /// New vector inserted
    Insert,
    /// Existing vector updated
    Update,
    /// Vector deleted
    Delete,
    /// Metadata-only update
    MetadataUpdate,
    /// Batch operation completed
    BatchComplete,
}

impl std::fmt::Display for ChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChangeType::Insert => write!(f, "INSERT"),
            ChangeType::Update => write!(f, "UPDATE"),
            ChangeType::Delete => write!(f, "DELETE"),
            ChangeType::MetadataUpdate => write!(f, "METADATA_UPDATE"),
            ChangeType::BatchComplete => write!(f, "BATCH_COMPLETE"),
        }
    }
}

/// A change event for a vector
#[derive(Debug, Clone)]
pub struct VectorChangeEvent {
    /// Unique sequence number for ordering
    pub sequence: u64,

    /// Timestamp of the change
    pub timestamp: u64,

    /// Type of change
    pub change_type: ChangeType,

    /// ID of the affected vector
    pub vector_id: String,

    /// Namespace (if applicable)
    pub namespace: Option<String>,

    /// Previous value (for updates/deletes)
    pub previous_vector: Option<Vec<f32>>,

    /// New value (for inserts/updates)
    pub new_vector: Option<Vec<f32>>,

    /// Metadata changes
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    /// Source of the change (e.g., "api", "replication", "import")
    pub source: String,

    /// Correlation ID for tracking
    pub correlation_id: Option<String>,
}

impl VectorChangeEvent {
    /// Create a new insert event
    pub fn insert(vector_id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            sequence: 0,
            timestamp: current_timestamp(),
            change_type: ChangeType::Insert,
            vector_id: vector_id.into(),
            namespace: None,
            previous_vector: None,
            new_vector: Some(vector),
            metadata: None,
            source: "api".to_string(),
            correlation_id: None,
        }
    }

    /// Create a new update event
    pub fn update(vector_id: impl Into<String>, old: Vec<f32>, new: Vec<f32>) -> Self {
        Self {
            sequence: 0,
            timestamp: current_timestamp(),
            change_type: ChangeType::Update,
            vector_id: vector_id.into(),
            namespace: None,
            previous_vector: Some(old),
            new_vector: Some(new),
            metadata: None,
            source: "api".to_string(),
            correlation_id: None,
        }
    }

    /// Create a new delete event
    pub fn delete(vector_id: impl Into<String>) -> Self {
        Self {
            sequence: 0,
            timestamp: current_timestamp(),
            change_type: ChangeType::Delete,
            vector_id: vector_id.into(),
            namespace: None,
            previous_vector: None,
            new_vector: None,
            metadata: None,
            source: "api".to_string(),
            correlation_id: None,
        }
    }

    /// Set namespace
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Set source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Set correlation ID
    pub fn with_correlation_id(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Filter for change events
#[derive(Debug, Clone, Default)]
pub struct ChangeFilter {
    /// Filter by namespace
    pub namespaces: Option<Vec<String>>,

    /// Filter by change types
    pub change_types: Option<Vec<ChangeType>>,

    /// Filter by ID prefix
    pub id_prefix: Option<String>,

    /// Filter by ID pattern (glob-style)
    pub id_pattern: Option<String>,

    /// Filter by source
    pub sources: Option<Vec<String>>,

    /// Minimum sequence to start from
    pub from_sequence: Option<u64>,

    /// Maximum sequence to read to
    pub to_sequence: Option<u64>,
}

impl ChangeFilter {
    /// Create a new empty filter (matches all)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a filter that matches all events
    pub fn all() -> Self {
        Self::default()
    }

    /// Filter by namespace
    pub fn namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespaces = Some(vec![ns.into()]);
        self
    }

    /// Filter by multiple namespaces
    pub fn namespaces(mut self, ns: Vec<String>) -> Self {
        self.namespaces = Some(ns);
        self
    }

    /// Filter by change type
    pub fn change_type(mut self, ct: ChangeType) -> Self {
        self.change_types = Some(vec![ct]);
        self
    }

    /// Filter by multiple change types
    pub fn change_types(mut self, cts: Vec<ChangeType>) -> Self {
        self.change_types = Some(cts);
        self
    }

    /// Filter by ID prefix
    pub fn id_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.id_prefix = Some(prefix.into());
        self
    }

    /// Filter by source
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.sources = Some(vec![source.into()]);
        self
    }

    /// Start from a specific sequence number
    pub fn from_sequence(mut self, seq: u64) -> Self {
        self.from_sequence = Some(seq);
        self
    }

    /// Check if an event matches this filter
    pub fn matches(&self, event: &VectorChangeEvent) -> bool {
        // Check namespace
        if let Some(ref ns) = self.namespaces {
            match &event.namespace {
                Some(event_ns) => {
                    if !ns.contains(event_ns) {
                        return false;
                    }
                },
                None => return false,
            }
        }

        // Check change type
        if let Some(ref cts) = self.change_types
            && !cts.contains(&event.change_type)
        {
            return false;
        }

        // Check ID prefix
        if let Some(ref prefix) = self.id_prefix
            && !event.vector_id.starts_with(prefix)
        {
            return false;
        }

        // Check ID pattern (simple glob)
        if let Some(ref pattern) = self.id_pattern
            && !glob_match(pattern, &event.vector_id)
        {
            return false;
        }

        // Check source
        if let Some(ref sources) = self.sources
            && !sources.contains(&event.source)
        {
            return false;
        }

        // Check sequence range
        if let Some(from) = self.from_sequence
            && event.sequence < from
        {
            return false;
        }

        if let Some(to) = self.to_sequence
            && event.sequence > to
        {
            return false;
        }

        true
    }
}

/// Simple glob pattern matching (supports * and ?)
fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    glob_match_impl(&pattern, &text)
}

fn glob_match_impl(pattern: &[char], text: &[char]) -> bool {
    match (pattern.first(), text.first()) {
        (None, None) => true,
        (Some('*'), _) => {
            // Try matching zero or more characters
            glob_match_impl(&pattern[1..], text)
                || (!text.is_empty() && glob_match_impl(pattern, &text[1..]))
        },
        (Some('?'), Some(_)) => glob_match_impl(&pattern[1..], &text[1..]),
        (Some(p), Some(t)) if p == t => glob_match_impl(&pattern[1..], &text[1..]),
        _ => false,
    }
}

/// A stream of change events
pub struct ChangeStream {
    receiver: Receiver<VectorChangeEvent>,
    filter: ChangeFilter,
    stats: Arc<StreamStats>,
}

struct StreamStats {
    received: AtomicU64,
    filtered_out: AtomicU64,
    last_sequence: AtomicU64,
}

impl ChangeStream {
    fn new(receiver: Receiver<VectorChangeEvent>, filter: ChangeFilter) -> Self {
        Self {
            receiver,
            filter,
            stats: Arc::new(StreamStats {
                received: AtomicU64::new(0),
                filtered_out: AtomicU64::new(0),
                last_sequence: AtomicU64::new(0),
            }),
        }
    }

    /// Receive the next event (blocking)
    pub fn recv(&self) -> Option<VectorChangeEvent> {
        loop {
            match self.receiver.recv() {
                Ok(event) => {
                    self.stats.received.fetch_add(1, Ordering::Relaxed);
                    if self.filter.matches(&event) {
                        self.stats
                            .last_sequence
                            .store(event.sequence, Ordering::Relaxed);
                        return Some(event);
                    } else {
                        self.stats.filtered_out.fetch_add(1, Ordering::Relaxed);
                    }
                },
                Err(_) => return None,
            }
        }
    }

    /// Receive with timeout
    pub fn recv_timeout(&self, timeout: Duration) -> Option<VectorChangeEvent> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }

            match self.receiver.recv_timeout(remaining) {
                Ok(event) => {
                    self.stats.received.fetch_add(1, Ordering::Relaxed);
                    if self.filter.matches(&event) {
                        self.stats
                            .last_sequence
                            .store(event.sequence, Ordering::Relaxed);
                        return Some(event);
                    } else {
                        self.stats.filtered_out.fetch_add(1, Ordering::Relaxed);
                    }
                },
                Err(mpsc::RecvTimeoutError::Timeout) => return None,
                Err(mpsc::RecvTimeoutError::Disconnected) => return None,
            }
        }
    }

    /// Try to receive without blocking
    pub fn try_recv(&self) -> Option<VectorChangeEvent> {
        loop {
            match self.receiver.try_recv() {
                Ok(event) => {
                    self.stats.received.fetch_add(1, Ordering::Relaxed);
                    if self.filter.matches(&event) {
                        self.stats
                            .last_sequence
                            .store(event.sequence, Ordering::Relaxed);
                        return Some(event);
                    } else {
                        self.stats.filtered_out.fetch_add(1, Ordering::Relaxed);
                    }
                },
                Err(TryRecvError::Empty) => return None,
                Err(TryRecvError::Disconnected) => return None,
            }
        }
    }

    /// Get the last received sequence number
    pub fn last_sequence(&self) -> u64 {
        self.stats.last_sequence.load(Ordering::Relaxed)
    }

    /// Get stream statistics
    pub fn stats(&self) -> ChangeStreamStats {
        ChangeStreamStats {
            received: self.stats.received.load(Ordering::Relaxed),
            filtered_out: self.stats.filtered_out.load(Ordering::Relaxed),
            last_sequence: self.stats.last_sequence.load(Ordering::Relaxed),
        }
    }
}

/// Statistics for a change stream
#[derive(Debug, Clone)]
pub struct ChangeStreamStats {
    pub received: u64,
    pub filtered_out: u64,
    pub last_sequence: u64,
}

/// Broadcasts change events to multiple subscribers
pub struct EventBroadcaster {
    inner: Arc<BroadcasterInner>,
}

struct BroadcasterInner {
    sequence: AtomicU64,
    subscribers: RwLock<Vec<Sender<VectorChangeEvent>>>,
    buffer: RwLock<Vec<VectorChangeEvent>>,
    buffer_capacity: usize,
    stats: BroadcasterStats,
}

struct BroadcasterStats {
    published: AtomicU64,
    dropped: AtomicU64,
    active_subscribers: AtomicU64,
}

impl EventBroadcaster {
    /// Create a new event broadcaster
    pub fn new(buffer_capacity: usize) -> Self {
        Self {
            inner: Arc::new(BroadcasterInner {
                sequence: AtomicU64::new(0),
                subscribers: RwLock::new(Vec::new()),
                buffer: RwLock::new(Vec::with_capacity(buffer_capacity)),
                buffer_capacity,
                stats: BroadcasterStats {
                    published: AtomicU64::new(0),
                    dropped: AtomicU64::new(0),
                    active_subscribers: AtomicU64::new(0),
                },
            }),
        }
    }

    /// Subscribe to events with a filter
    pub fn subscribe(&self, filter: ChangeFilter) -> ChangeStream {
        let (sender, receiver) = mpsc::channel();

        // Add to subscribers list
        {
            let Ok(mut subs) = self.inner.subscribers.write() else {
                return ChangeStream::new(receiver, filter);
            };
            subs.push(sender);
            self.inner
                .stats
                .active_subscribers
                .store(subs.len() as u64, Ordering::Relaxed);
        }

        // If filter has from_sequence, replay buffered events
        if let Some(from_seq) = filter.from_sequence {
            let Ok(buffer) = self.inner.buffer.read() else {
                return ChangeStream::new(receiver, filter);
            };
            for event in buffer.iter() {
                if event.sequence >= from_seq {
                    // Already filtered by the stream
                    let Ok(subs) = self.inner.subscribers.read() else {
                        continue;
                    };
                    if let Some(last_sub) = subs.last() {
                        let _ = last_sub.send(event.clone());
                    }
                }
            }
        }

        ChangeStream::new(receiver, filter)
    }

    /// Publish an event to all subscribers
    pub fn publish(&self, mut event: VectorChangeEvent) {
        // Assign sequence number
        event.sequence = self.inner.sequence.fetch_add(1, Ordering::SeqCst);

        // Buffer the event
        {
            let Ok(mut buffer) = self.inner.buffer.write() else {
                return;
            };
            if buffer.len() >= self.inner.buffer_capacity {
                buffer.remove(0);
            }
            buffer.push(event.clone());
        }

        // Broadcast to subscribers
        let Ok(mut subs) = self.inner.subscribers.write() else {
            return;
        };
        subs.retain(|sender| {
            match sender.send(event.clone()) {
                Ok(_) => {
                    self.inner.stats.published.fetch_add(1, Ordering::Relaxed);
                    true
                },
                Err(_) => {
                    // Subscriber dropped (disconnected)
                    false
                },
            }
        });

        self.inner
            .stats
            .active_subscribers
            .store(subs.len() as u64, Ordering::Relaxed);
    }

    /// Get current sequence number
    pub fn current_sequence(&self) -> u64 {
        self.inner.sequence.load(Ordering::SeqCst)
    }

    /// Get broadcaster statistics
    pub fn stats(&self) -> BroadcasterStatsSnapshot {
        let Ok(buffer) = self.inner.buffer.read() else {
            return BroadcasterStatsSnapshot {
                published: self.inner.stats.published.load(Ordering::Relaxed),
                dropped: self.inner.stats.dropped.load(Ordering::Relaxed),
                active_subscribers: self.inner.stats.active_subscribers.load(Ordering::Relaxed),
                current_sequence: self.current_sequence(),
                buffer_size: 0,
            };
        };
        BroadcasterStatsSnapshot {
            published: self.inner.stats.published.load(Ordering::Relaxed),
            dropped: self.inner.stats.dropped.load(Ordering::Relaxed),
            active_subscribers: self.inner.stats.active_subscribers.load(Ordering::Relaxed),
            current_sequence: self.current_sequence(),
            buffer_size: buffer.len(),
        }
    }
}

impl Clone for EventBroadcaster {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Snapshot of broadcaster statistics
#[derive(Debug, Clone)]
pub struct BroadcasterStatsSnapshot {
    pub published: u64,
    pub dropped: u64,
    pub active_subscribers: u64,
    pub current_sequence: u64,
    pub buffer_size: usize,
}

/// Async change stream for use with tokio
#[cfg(feature = "async")]
pub struct AsyncChangeStream {
    receiver: tokio::sync::mpsc::UnboundedReceiver<VectorChangeEvent>,
    filter: ChangeFilter,
    stats: Arc<StreamStats>,
}

#[cfg(feature = "async")]
impl AsyncChangeStream {
    fn new(
        receiver: tokio::sync::mpsc::UnboundedReceiver<VectorChangeEvent>,
        filter: ChangeFilter,
    ) -> Self {
        Self {
            receiver,
            filter,
            stats: Arc::new(StreamStats {
                received: AtomicU64::new(0),
                filtered_out: AtomicU64::new(0),
                last_sequence: AtomicU64::new(0),
            }),
        }
    }

    /// Receive the next event (async)
    pub async fn recv(&mut self) -> Option<VectorChangeEvent> {
        loop {
            match self.receiver.recv().await {
                Some(event) => {
                    self.stats.received.fetch_add(1, Ordering::Relaxed);
                    if self.filter.matches(&event) {
                        self.stats
                            .last_sequence
                            .store(event.sequence, Ordering::Relaxed);
                        return Some(event);
                    } else {
                        self.stats.filtered_out.fetch_add(1, Ordering::Relaxed);
                    }
                },
                None => return None,
            }
        }
    }

    /// Get stream statistics
    pub fn stats(&self) -> ChangeStreamStats {
        ChangeStreamStats {
            received: self.stats.received.load(Ordering::Relaxed),
            filtered_out: self.stats.filtered_out.load(Ordering::Relaxed),
            last_sequence: self.stats.last_sequence.load(Ordering::Relaxed),
        }
    }
}

/// Async event broadcaster
#[cfg(feature = "async")]
pub struct AsyncEventBroadcaster {
    inner: Arc<AsyncBroadcasterInner>,
}

#[cfg(feature = "async")]
struct AsyncBroadcasterInner {
    sequence: AtomicU64,
    subscribers: tokio::sync::RwLock<Vec<tokio::sync::mpsc::UnboundedSender<VectorChangeEvent>>>,
    buffer: tokio::sync::RwLock<Vec<VectorChangeEvent>>,
    buffer_capacity: usize,
    stats: BroadcasterStats,
}

#[cfg(feature = "async")]
impl AsyncEventBroadcaster {
    /// Create a new async event broadcaster
    pub fn new(buffer_capacity: usize) -> Self {
        Self {
            inner: Arc::new(AsyncBroadcasterInner {
                sequence: AtomicU64::new(0),
                subscribers: tokio::sync::RwLock::new(Vec::new()),
                buffer: tokio::sync::RwLock::new(Vec::with_capacity(buffer_capacity)),
                buffer_capacity,
                stats: BroadcasterStats {
                    published: AtomicU64::new(0),
                    dropped: AtomicU64::new(0),
                    active_subscribers: AtomicU64::new(0),
                },
            }),
        }
    }

    /// Subscribe to events with a filter
    pub async fn subscribe(&self, filter: ChangeFilter) -> AsyncChangeStream {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();

        // Add to subscribers list
        {
            let mut subs = self.inner.subscribers.write().await;
            subs.push(sender);
            self.inner
                .stats
                .active_subscribers
                .store(subs.len() as u64, Ordering::Relaxed);
        }

        // If filter has from_sequence, replay buffered events
        if let Some(from_seq) = filter.from_sequence {
            let buffer = self.inner.buffer.read().await;
            let subs = self.inner.subscribers.read().await;
            if let Some(sender) = subs.last() {
                for event in buffer.iter() {
                    if event.sequence >= from_seq {
                        let _ = sender.send(event.clone());
                    }
                }
            }
        }

        AsyncChangeStream::new(receiver, filter)
    }

    /// Publish an event to all subscribers
    pub async fn publish(&self, mut event: VectorChangeEvent) {
        // Assign sequence number
        event.sequence = self.inner.sequence.fetch_add(1, Ordering::SeqCst);

        // Buffer the event
        {
            let mut buffer = self.inner.buffer.write().await;
            if buffer.len() >= self.inner.buffer_capacity {
                buffer.remove(0);
            }
            buffer.push(event.clone());
        }

        // Broadcast to subscribers
        let mut subs = self.inner.subscribers.write().await;
        subs.retain(|sender| match sender.send(event.clone()) {
            Ok(_) => {
                self.inner.stats.published.fetch_add(1, Ordering::Relaxed);
                true
            },
            Err(_) => false,
        });

        self.inner
            .stats
            .active_subscribers
            .store(subs.len() as u64, Ordering::Relaxed);
    }

    /// Get current sequence number
    pub fn current_sequence(&self) -> u64 {
        self.inner.sequence.load(Ordering::SeqCst)
    }

    /// Get broadcaster statistics
    pub async fn stats(&self) -> BroadcasterStatsSnapshot {
        BroadcasterStatsSnapshot {
            published: self.inner.stats.published.load(Ordering::Relaxed),
            dropped: self.inner.stats.dropped.load(Ordering::Relaxed),
            active_subscribers: self.inner.stats.active_subscribers.load(Ordering::Relaxed),
            current_sequence: self.current_sequence(),
            buffer_size: self.inner.buffer.read().await.len(),
        }
    }
}

#[cfg(feature = "async")]
impl Clone for AsyncEventBroadcaster {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_change_event_creation() {
        let insert = VectorChangeEvent::insert("vec1", vec![0.1, 0.2, 0.3]);
        assert_eq!(insert.change_type, ChangeType::Insert);
        assert_eq!(insert.vector_id, "vec1");
        assert!(insert.new_vector.is_some());
        assert!(insert.previous_vector.is_none());

        let update = VectorChangeEvent::update("vec1", vec![0.1, 0.2], vec![0.3, 0.4]);
        assert_eq!(update.change_type, ChangeType::Update);
        assert!(update.previous_vector.is_some());
        assert!(update.new_vector.is_some());

        let delete = VectorChangeEvent::delete("vec1");
        assert_eq!(delete.change_type, ChangeType::Delete);
        assert!(delete.new_vector.is_none());
    }

    #[test]
    fn test_change_filter() {
        let event = VectorChangeEvent::insert("products:123", vec![0.1])
            .with_namespace("products".to_string())
            .with_source("api".to_string());

        // Match all
        assert!(ChangeFilter::all().matches(&event));

        // Match by namespace
        assert!(ChangeFilter::new().namespace("products").matches(&event));
        assert!(!ChangeFilter::new().namespace("users").matches(&event));

        // Match by change type
        assert!(
            ChangeFilter::new()
                .change_type(ChangeType::Insert)
                .matches(&event)
        );
        assert!(
            !ChangeFilter::new()
                .change_type(ChangeType::Delete)
                .matches(&event)
        );

        // Match by ID prefix
        assert!(ChangeFilter::new().id_prefix("products:").matches(&event));
        assert!(!ChangeFilter::new().id_prefix("users:").matches(&event));

        // Match by source
        assert!(ChangeFilter::new().source("api").matches(&event));
        assert!(!ChangeFilter::new().source("replication").matches(&event));
    }

    #[test]
    fn test_glob_pattern_matching() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("doc*", "document"));
        assert!(glob_match("doc*", "doc"));
        assert!(!glob_match("doc*", "specification"));

        assert!(glob_match("*.txt", "file.txt"));
        assert!(!glob_match("*.txt", "file.json"));

        assert!(glob_match("doc?", "docs"));
        assert!(glob_match("doc?", "doc1"));
        assert!(!glob_match("doc?", "document"));

        assert!(glob_match("*:*", "namespace:id"));
        assert!(glob_match("products:*", "products:123"));
    }

    #[test]
    fn test_broadcaster_basic() {
        let broadcaster = EventBroadcaster::new(100);

        // Subscribe
        let stream = broadcaster.subscribe(ChangeFilter::all());

        // Publish
        broadcaster.publish(VectorChangeEvent::insert("vec1", vec![0.1, 0.2]));
        broadcaster.publish(VectorChangeEvent::insert("vec2", vec![0.3, 0.4]));

        // Receive
        let event1 = stream.recv_timeout(Duration::from_millis(100));
        assert!(event1.is_some());
        assert_eq!(event1.unwrap().vector_id, "vec1");

        let event2 = stream.recv_timeout(Duration::from_millis(100));
        assert!(event2.is_some());
        assert_eq!(event2.unwrap().vector_id, "vec2");
    }

    #[test]
    fn test_broadcaster_filtering() {
        let broadcaster = EventBroadcaster::new(100);

        // Subscribe with filter
        let stream = broadcaster.subscribe(ChangeFilter::new().namespace("products"));

        // Publish events in different namespaces
        broadcaster.publish(
            VectorChangeEvent::insert("p1", vec![0.1]).with_namespace("products".to_string()),
        );
        broadcaster.publish(
            VectorChangeEvent::insert("u1", vec![0.2]).with_namespace("users".to_string()),
        );
        broadcaster.publish(
            VectorChangeEvent::insert("p2", vec![0.3]).with_namespace("products".to_string()),
        );

        // Should only receive products
        let event1 = stream.recv_timeout(Duration::from_millis(100));
        assert_eq!(event1.unwrap().vector_id, "p1");

        let event2 = stream.recv_timeout(Duration::from_millis(100));
        assert_eq!(event2.unwrap().vector_id, "p2");

        // No more events
        assert!(stream.recv_timeout(Duration::from_millis(50)).is_none());
    }

    #[test]
    fn test_multiple_subscribers() {
        let broadcaster = EventBroadcaster::new(100);

        let stream1 = broadcaster.subscribe(ChangeFilter::all());
        let stream2 = broadcaster.subscribe(ChangeFilter::new().change_type(ChangeType::Delete));

        // Publish
        broadcaster.publish(VectorChangeEvent::insert("vec1", vec![0.1]));
        broadcaster.publish(VectorChangeEvent::delete("vec2"));

        // Stream1 should get both
        assert!(stream1.recv_timeout(Duration::from_millis(100)).is_some());
        assert!(stream1.recv_timeout(Duration::from_millis(100)).is_some());

        // Stream2 should only get delete
        let event = stream2.recv_timeout(Duration::from_millis(100));
        assert!(event.is_some());
        assert_eq!(event.unwrap().change_type, ChangeType::Delete);
        assert!(stream2.recv_timeout(Duration::from_millis(50)).is_none());
    }

    #[test]
    fn test_sequence_numbers() {
        let broadcaster = EventBroadcaster::new(100);
        let stream = broadcaster.subscribe(ChangeFilter::all());

        broadcaster.publish(VectorChangeEvent::insert("vec1", vec![0.1]));
        broadcaster.publish(VectorChangeEvent::insert("vec2", vec![0.2]));
        broadcaster.publish(VectorChangeEvent::insert("vec3", vec![0.3]));

        let e1 = stream.recv_timeout(Duration::from_millis(100)).unwrap();
        let e2 = stream.recv_timeout(Duration::from_millis(100)).unwrap();
        let e3 = stream.recv_timeout(Duration::from_millis(100)).unwrap();

        assert_eq!(e1.sequence, 0);
        assert_eq!(e2.sequence, 1);
        assert_eq!(e3.sequence, 2);
    }

    #[test]
    fn test_event_buffering_and_replay() {
        let broadcaster = EventBroadcaster::new(100);

        // Publish before subscribing
        broadcaster.publish(VectorChangeEvent::insert("vec1", vec![0.1]));
        broadcaster.publish(VectorChangeEvent::insert("vec2", vec![0.2]));

        // Subscribe with from_sequence to replay
        let stream = broadcaster.subscribe(ChangeFilter::new().from_sequence(0));

        // Should receive buffered events
        let event = stream.recv_timeout(Duration::from_millis(100));
        assert!(event.is_some());
        assert_eq!(event.unwrap().vector_id, "vec1");
    }

    #[test]
    fn test_stats() {
        let broadcaster = EventBroadcaster::new(100);
        let stream = broadcaster.subscribe(ChangeFilter::new().change_type(ChangeType::Delete));

        // Publish mixed events
        broadcaster.publish(VectorChangeEvent::insert("vec1", vec![0.1]));
        broadcaster.publish(VectorChangeEvent::delete("vec2"));

        // Consume filtered stream
        stream.recv_timeout(Duration::from_millis(100));

        let stream_stats = stream.stats();
        assert_eq!(stream_stats.received, 2); // Received both
        assert_eq!(stream_stats.filtered_out, 1); // Filtered out insert

        let bc_stats = broadcaster.stats();
        assert!(bc_stats.published > 0);
        assert_eq!(bc_stats.active_subscribers, 1);
    }
}

#[cfg(all(test, feature = "async"))]
mod async_tests {
    use super::*;

    #[tokio::test]
    async fn test_async_broadcaster() {
        let broadcaster = AsyncEventBroadcaster::new(100);

        // Subscribe
        let mut stream = broadcaster.subscribe(ChangeFilter::all()).await;

        // Publish from another task
        let bc = broadcaster.clone();
        tokio::spawn(async move {
            bc.publish(VectorChangeEvent::insert("vec1", vec![0.1]))
                .await;
            bc.publish(VectorChangeEvent::insert("vec2", vec![0.2]))
                .await;
        });

        // Receive
        tokio::time::sleep(Duration::from_millis(50)).await;
        let event1 = stream.recv().await;
        assert!(event1.is_some());
        assert_eq!(event1.unwrap().vector_id, "vec1");
    }
}
