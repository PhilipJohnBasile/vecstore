// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Change Streams
//!
//! Real-time subscriptions for vector database changes. Get notified when
//! vectors are inserted, updated, or deleted.
//!
//! ## Features
//!
//! - **Real-time Notifications**: Subscribe to collection changes
//! - **Filtered Streams**: Watch specific IDs or metadata patterns
//! - **Resume Tokens**: Continue from where you left off
//! - **Batched Events**: Efficient delivery of multiple changes
//! - **Backpressure**: Handle slow consumers gracefully
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::change_streams::{ChangeStream, ChangeFilter};
//!
//! let stream = ChangeStream::new("my_collection");
//!
//! // Watch for all changes
//! let subscription = stream.watch(ChangeFilter::all())?;
//!
//! // Process changes
//! for event in subscription.events() {
//!     match event.operation {
//!         Operation::Insert => println!("Inserted: {}", event.document_id),
//!         Operation::Update => println!("Updated: {}", event.document_id),
//!         Operation::Delete => println!("Deleted: {}", event.document_id),
//!     }
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Change operation type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    /// New document inserted
    Insert,
    /// Existing document updated
    Update,
    /// Document deleted
    Delete,
    /// Document replaced (full update)
    Replace,
    /// Collection dropped
    Drop,
    /// Collection renamed
    Rename { new_name: String },
    /// Invalidate (something changed that affects cached state)
    Invalidate,
}

/// Change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEvent {
    /// Event ID (monotonically increasing)
    pub id: u64,
    /// Operation type
    pub operation: Operation,
    /// Collection name
    pub collection: String,
    /// Document ID
    pub document_id: String,
    /// Timestamp
    pub timestamp: i64,
    /// Resume token (for continuing after disconnect)
    pub resume_token: ResumeToken,
    /// Previous document state (for updates)
    pub before: Option<DocumentSnapshot>,
    /// Current document state
    pub after: Option<DocumentSnapshot>,
    /// Changed fields (for partial updates)
    pub changed_fields: Option<Vec<String>>,
    /// Metadata about the change
    pub metadata: ChangeMetadata,
}

/// Document snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSnapshot {
    /// Document ID
    pub id: String,
    /// Vector data
    pub vector: Option<Vec<f32>>,
    /// Document metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Snapshot timestamp
    pub timestamp: i64,
}

/// Change metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeMetadata {
    /// Transaction ID (if applicable)
    pub transaction_id: Option<String>,
    /// User/client that made the change
    pub user: Option<String>,
    /// Source of the change
    pub source: ChangeSource,
    /// Custom tags
    pub tags: Vec<String>,
}

/// Source of the change
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChangeSource {
    /// Direct API call
    Api,
    /// Replication from another node
    Replication,
    /// Import operation
    Import,
    /// Sync from external source
    Sync,
    /// Internal maintenance
    Maintenance,
    /// Unknown source
    Unknown,
}

/// Resume token for continuing stream
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResumeToken {
    /// Token value
    pub token: String,
    /// Cluster time
    pub cluster_time: i64,
    /// Collection
    pub collection: String,
}

impl ResumeToken {
    /// Create new resume token
    pub fn new(event_id: u64, collection: &str) -> Self {
        Self {
            token: format!("{}:{}", collection, event_id),
            cluster_time: unix_timestamp(),
            collection: collection.to_string(),
        }
    }

    /// Parse token string
    pub fn from_string(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() == 2 {
            Some(Self {
                token: s.to_string(),
                cluster_time: 0,
                collection: parts[0].to_string(),
            })
        } else {
            None
        }
    }
}

/// Change filter
#[derive(Debug, Clone, Default)]
pub struct ChangeFilter {
    /// Filter by operation types
    pub operations: Option<Vec<Operation>>,
    /// Filter by document IDs
    pub document_ids: Option<Vec<String>>,
    /// Filter by metadata fields
    pub metadata_filter: Option<HashMap<String, serde_json::Value>>,
    /// Include full document in events
    pub include_full_document: bool,
    /// Include previous document state
    pub include_previous: bool,
    /// Start after this resume token
    pub start_after: Option<ResumeToken>,
    /// Maximum batch size
    pub batch_size: usize,
}

impl ChangeFilter {
    /// Match all changes
    pub fn all() -> Self {
        Self {
            operations: None,
            document_ids: None,
            metadata_filter: None,
            include_full_document: true,
            include_previous: false,
            start_after: None,
            batch_size: 100,
        }
    }

    /// Filter by operations
    pub fn operations(ops: Vec<Operation>) -> Self {
        Self {
            operations: Some(ops),
            ..Default::default()
        }
    }

    /// Filter inserts only
    pub fn inserts() -> Self {
        Self::operations(vec![Operation::Insert])
    }

    /// Filter updates only
    pub fn updates() -> Self {
        Self::operations(vec![Operation::Update])
    }

    /// Filter deletes only
    pub fn deletes() -> Self {
        Self::operations(vec![Operation::Delete])
    }

    /// Filter by document IDs
    pub fn for_documents(ids: Vec<String>) -> Self {
        Self {
            document_ids: Some(ids),
            ..Default::default()
        }
    }

    /// Include full document
    pub fn with_full_document(mut self) -> Self {
        self.include_full_document = true;
        self
    }

    /// Include previous state
    pub fn with_previous(mut self) -> Self {
        self.include_previous = true;
        self
    }

    /// Resume from token
    pub fn resume_after(mut self, token: ResumeToken) -> Self {
        self.start_after = Some(token);
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Check if event matches filter
    fn matches(&self, event: &ChangeEvent) -> bool {
        // Check operation
        if let Some(ops) = &self.operations {
            if !ops.contains(&event.operation) {
                return false;
            }
        }

        // Check document ID
        if let Some(ids) = &self.document_ids {
            if !ids.contains(&event.document_id) {
                return false;
            }
        }

        // Check metadata
        if let Some(filter) = &self.metadata_filter {
            if let Some(after) = &event.after {
                for (key, value) in filter {
                    if after.metadata.get(key) != Some(value) {
                        return false;
                    }
                }
            } else {
                return false;
            }
        }

        true
    }
}

/// Subscription handle
pub struct Subscription {
    /// Subscription ID
    pub id: u64,
    /// Collection
    pub collection: String,
    /// Filter
    pub filter: ChangeFilter,
    /// Active flag
    active: Arc<AtomicBool>,
    /// Event queue
    events: Arc<RwLock<VecDeque<ChangeEvent>>>,
    /// Last resume token
    last_token: Arc<RwLock<Option<ResumeToken>>>,
}

impl Subscription {
    /// Check if subscription is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Get next event (non-blocking)
    pub fn try_next(&self) -> Option<ChangeEvent> {
        let mut events = self.events.write().unwrap();
        let event = events.pop_front();

        if let Some(ref e) = event {
            let mut token = self.last_token.write().unwrap();
            *token = Some(e.resume_token.clone());
        }

        event
    }

    /// Get batch of events
    pub fn try_batch(&self, max: usize) -> Vec<ChangeEvent> {
        let mut events = self.events.write().unwrap();
        let count = max.min(events.len());
        let batch: Vec<ChangeEvent> = events.drain(..count).collect();

        if let Some(last) = batch.last() {
            let mut token = self.last_token.write().unwrap();
            *token = Some(last.resume_token.clone());
        }

        batch
    }

    /// Get last resume token
    pub fn resume_token(&self) -> Option<ResumeToken> {
        self.last_token.read().unwrap().clone()
    }

    /// Close subscription
    pub fn close(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Check if events are pending
    pub fn has_events(&self) -> bool {
        !self.events.read().unwrap().is_empty()
    }

    /// Get pending event count
    pub fn pending_count(&self) -> usize {
        self.events.read().unwrap().len()
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.close();
    }
}

/// Change stream for a collection
pub struct ChangeStream {
    /// Collection name
    collection: String,
    /// Event counter
    event_counter: AtomicU64,
    /// Event log (for replay)
    event_log: RwLock<VecDeque<ChangeEvent>>,
    /// Maximum log size
    max_log_size: usize,
    /// Active subscriptions
    subscriptions: RwLock<HashMap<u64, Arc<Subscription>>>,
    /// Subscription counter
    subscription_counter: AtomicU64,
}

impl ChangeStream {
    /// Create new change stream
    pub fn new(collection: &str) -> Self {
        Self {
            collection: collection.to_string(),
            event_counter: AtomicU64::new(0),
            event_log: RwLock::new(VecDeque::new()),
            max_log_size: 10000,
            subscriptions: RwLock::new(HashMap::new()),
            subscription_counter: AtomicU64::new(0),
        }
    }

    /// Watch for changes
    pub fn watch(&self, filter: ChangeFilter) -> Arc<Subscription> {
        let id = self.subscription_counter.fetch_add(1, Ordering::Relaxed);

        // If resuming, replay events from log
        let events = if let Some(ref token) = filter.start_after {
            self.replay_from(token, &filter)
        } else {
            VecDeque::new()
        };

        let subscription = Arc::new(Subscription {
            id,
            collection: self.collection.clone(),
            filter,
            active: Arc::new(AtomicBool::new(true)),
            events: Arc::new(RwLock::new(events)),
            last_token: Arc::new(RwLock::new(None)),
        });

        let mut subs = self.subscriptions.write().unwrap();
        subs.insert(id, subscription.clone());

        subscription
    }

    /// Unwatch (close subscription)
    pub fn unwatch(&self, subscription_id: u64) {
        let mut subs = self.subscriptions.write().unwrap();
        if let Some(sub) = subs.remove(&subscription_id) {
            sub.close();
        }
    }

    /// Emit an event
    pub fn emit(&self, operation: Operation, document_id: &str, before: Option<DocumentSnapshot>, after: Option<DocumentSnapshot>) {
        let event_id = self.event_counter.fetch_add(1, Ordering::Relaxed);
        let timestamp = unix_timestamp();

        let event = ChangeEvent {
            id: event_id,
            operation,
            collection: self.collection.clone(),
            document_id: document_id.to_string(),
            timestamp,
            resume_token: ResumeToken::new(event_id, &self.collection),
            before,
            after,
            changed_fields: None,
            metadata: ChangeMetadata {
                transaction_id: None,
                user: None,
                source: ChangeSource::Api,
                tags: Vec::new(),
            },
        };

        // Add to log
        {
            let mut log = self.event_log.write().unwrap();
            log.push_back(event.clone());
            while log.len() > self.max_log_size {
                log.pop_front();
            }
        }

        // Notify subscriptions
        self.notify_subscriptions(&event);
    }

    /// Emit insert event
    pub fn emit_insert(&self, id: &str, vector: Vec<f32>, metadata: HashMap<String, serde_json::Value>) {
        let snapshot = DocumentSnapshot {
            id: id.to_string(),
            vector: Some(vector),
            metadata,
            timestamp: unix_timestamp(),
        };

        self.emit(Operation::Insert, id, None, Some(snapshot));
    }

    /// Emit update event
    pub fn emit_update(&self, id: &str, before: Option<DocumentSnapshot>, after: DocumentSnapshot) {
        self.emit(Operation::Update, id, before, Some(after));
    }

    /// Emit delete event
    pub fn emit_delete(&self, id: &str, before: Option<DocumentSnapshot>) {
        self.emit(Operation::Delete, id, before, None);
    }

    fn notify_subscriptions(&self, event: &ChangeEvent) {
        let subs = self.subscriptions.read().unwrap();

        for sub in subs.values() {
            if sub.is_active() && sub.filter.matches(event) {
                let mut events = sub.events.write().unwrap();

                // Apply backpressure
                if events.len() < sub.filter.batch_size * 10 {
                    events.push_back(event.clone());
                }
                // If queue is full, oldest events are dropped
            }
        }
    }

    fn replay_from(&self, token: &ResumeToken, filter: &ChangeFilter) -> VecDeque<ChangeEvent> {
        let log = self.event_log.read().unwrap();

        // Find starting position
        let start_id: u64 = token.token
            .split(':')
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        log.iter()
            .filter(|e| e.id > start_id && filter.matches(e))
            .cloned()
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> ChangeStreamStats {
        let subs = self.subscriptions.read().unwrap();
        let log = self.event_log.read().unwrap();

        ChangeStreamStats {
            collection: self.collection.clone(),
            total_events: self.event_counter.load(Ordering::Relaxed),
            log_size: log.len(),
            active_subscriptions: subs.values().filter(|s| s.is_active()).count(),
        }
    }
}

/// Change stream statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeStreamStats {
    pub collection: String,
    pub total_events: u64,
    pub log_size: usize,
    pub active_subscriptions: usize,
}

/// Multi-collection change stream manager
pub struct ChangeStreamManager {
    streams: RwLock<HashMap<String, Arc<ChangeStream>>>,
}

impl ChangeStreamManager {
    /// Create new manager
    pub fn new() -> Self {
        Self {
            streams: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create stream for collection
    pub fn stream(&self, collection: &str) -> Arc<ChangeStream> {
        let streams = self.streams.read().unwrap();

        if let Some(stream) = streams.get(collection) {
            return stream.clone();
        }

        drop(streams);

        let mut streams = self.streams.write().unwrap();
        streams
            .entry(collection.to_string())
            .or_insert_with(|| Arc::new(ChangeStream::new(collection)))
            .clone()
    }

    /// Watch all collections
    pub fn watch_all(&self, filter: ChangeFilter) -> Vec<Arc<Subscription>> {
        let streams = self.streams.read().unwrap();
        streams.values().map(|s| s.watch(filter.clone())).collect()
    }

    /// Get all statistics
    pub fn stats(&self) -> Vec<ChangeStreamStats> {
        let streams = self.streams.read().unwrap();
        streams.values().map(|s| s.stats()).collect()
    }
}

impl Default for ChangeStreamManager {
    fn default() -> Self {
        Self::new()
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
    fn test_change_stream() {
        let stream = ChangeStream::new("test_collection");

        // Subscribe
        let sub = stream.watch(ChangeFilter::all());

        // Emit events
        stream.emit_insert("doc1", vec![0.1, 0.2], HashMap::new());
        stream.emit_insert("doc2", vec![0.3, 0.4], HashMap::new());

        // Check events
        let event1 = sub.try_next().unwrap();
        assert_eq!(event1.document_id, "doc1");
        assert_eq!(event1.operation, Operation::Insert);

        let event2 = sub.try_next().unwrap();
        assert_eq!(event2.document_id, "doc2");
    }

    #[test]
    fn test_filtered_subscription() {
        let stream = ChangeStream::new("test");

        // Subscribe only to inserts
        let sub = stream.watch(ChangeFilter::inserts());

        // Emit mixed events
        stream.emit_insert("doc1", vec![0.1], HashMap::new());
        stream.emit_delete("doc2", None);
        stream.emit_insert("doc3", vec![0.2], HashMap::new());

        // Should only get inserts
        let events = sub.try_batch(10);
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| e.operation == Operation::Insert));
    }

    #[test]
    fn test_resume_token() {
        let stream = ChangeStream::new("test");

        // Emit some events
        stream.emit_insert("doc1", vec![0.1], HashMap::new());
        stream.emit_insert("doc2", vec![0.2], HashMap::new());
        stream.emit_insert("doc3", vec![0.3], HashMap::new());

        // First subscriber gets all events
        let sub1 = stream.watch(ChangeFilter::all());
        let events = sub1.try_batch(10);
        assert_eq!(events.len(), 3);

        let token = events[1].resume_token.clone();

        // Second subscriber resumes after doc2
        let sub2 = stream.watch(ChangeFilter::all().resume_after(token));

        // Should get doc3 only
        let events = sub2.try_batch(10);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].document_id, "doc3");
    }

    #[test]
    fn test_document_filter() {
        let stream = ChangeStream::new("test");

        // Subscribe to specific documents
        let sub = stream.watch(ChangeFilter::for_documents(vec!["doc1".to_string(), "doc3".to_string()]));

        // Emit events
        stream.emit_insert("doc1", vec![0.1], HashMap::new());
        stream.emit_insert("doc2", vec![0.2], HashMap::new());
        stream.emit_insert("doc3", vec![0.3], HashMap::new());
        stream.emit_insert("doc4", vec![0.4], HashMap::new());

        // Should only get doc1 and doc3
        let events = sub.try_batch(10);
        assert_eq!(events.len(), 2);
        assert!(events.iter().any(|e| e.document_id == "doc1"));
        assert!(events.iter().any(|e| e.document_id == "doc3"));
    }

    #[test]
    fn test_change_stream_manager() {
        let manager = ChangeStreamManager::new();

        let stream1 = manager.stream("collection1");
        let stream2 = manager.stream("collection2");

        stream1.emit_insert("doc1", vec![0.1], HashMap::new());
        stream2.emit_insert("doc2", vec![0.2], HashMap::new());

        let stats = manager.stats();
        assert_eq!(stats.len(), 2);
    }

    #[test]
    fn test_update_with_before() {
        let stream = ChangeStream::new("test");
        let sub = stream.watch(ChangeFilter::all().with_previous());

        let before = DocumentSnapshot {
            id: "doc1".to_string(),
            vector: Some(vec![0.1]),
            metadata: HashMap::new(),
            timestamp: unix_timestamp(),
        };

        let after = DocumentSnapshot {
            id: "doc1".to_string(),
            vector: Some(vec![0.2]),
            metadata: HashMap::new(),
            timestamp: unix_timestamp(),
        };

        stream.emit_update("doc1", Some(before.clone()), after);

        let event = sub.try_next().unwrap();
        assert_eq!(event.operation, Operation::Update);
        assert!(event.before.is_some());
        assert!(event.after.is_some());
    }
}
