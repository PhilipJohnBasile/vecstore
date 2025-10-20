//! Audit Logging
//!
//! Provides comprehensive audit trails for compliance and security monitoring.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Audit event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Vector insertion
    Insert,
    /// Vector update
    Update,
    /// Vector deletion
    Delete,
    /// Query operation
    Query,
    /// Batch operation
    Batch,
    /// Index creation
    IndexCreate,
    /// Index deletion
    IndexDelete,
    /// Authentication
    Auth,
    /// Authorization
    Authz,
    /// Configuration change
    ConfigChange,
    /// Backup operation
    Backup,
    /// Restore operation
    Restore,
    /// Export operation
    Export,
    /// Import operation
    Import,
}

/// Audit severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AuditSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// Audit outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditOutcome {
    Success,
    Failure,
    Denied,
}

/// Audit entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditMetadata {
    /// User identifier
    pub user_id: Option<String>,

    /// IP address
    pub ip_address: Option<String>,

    /// Session ID
    pub session_id: Option<String>,

    /// Request ID for tracing
    pub request_id: Option<String>,

    /// Additional custom fields
    #[serde(flatten)]
    pub custom: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for AuditMetadata {
    fn default() -> Self {
        Self {
            user_id: None,
            ip_address: None,
            session_id: None,
            request_id: None,
            custom: std::collections::HashMap::new(),
        }
    }
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry ID
    pub id: String,

    /// Timestamp
    pub timestamp: SystemTime,

    /// Event type
    pub event_type: AuditEventType,

    /// Severity level
    pub severity: AuditSeverity,

    /// Outcome
    pub outcome: AuditOutcome,

    /// Resource affected (e.g., vector ID, index name)
    pub resource: Option<String>,

    /// Action description
    pub action: String,

    /// Additional details
    pub details: Option<String>,

    /// Metadata (user, IP, etc.)
    pub metadata: AuditMetadata,

    /// Duration in milliseconds (for operations)
    pub duration_ms: Option<u64>,
}

impl AuditEntry {
    /// Create a new audit entry
    pub fn new(event_type: AuditEventType, action: impl Into<String>) -> Self {
        let timestamp = SystemTime::now();
        let micros = timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_micros();

        // Generate a simple unique ID based on timestamp and random component
        let random_component = (micros % 1000000) as u32;
        let id = format!("audit-{}-{}", micros, random_component);

        Self {
            id,
            timestamp,
            event_type,
            severity: AuditSeverity::Info,
            outcome: AuditOutcome::Success,
            resource: None,
            action: action.into(),
            details: None,
            metadata: AuditMetadata::default(),
            duration_ms: None,
        }
    }

    /// Set severity
    pub fn with_severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set outcome
    pub fn with_outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Set resource
    pub fn with_resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = Some(resource.into());
        self
    }

    /// Set details
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Set user ID
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.metadata.user_id = Some(user_id.into());
        self
    }

    /// Set IP address
    pub fn with_ip(mut self, ip: impl Into<String>) -> Self {
        self.metadata.ip_address = Some(ip.into());
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// Audit backend trait
pub trait AuditBackend: Send + Sync {
    /// Write an audit entry
    fn write(&mut self, entry: &AuditEntry) -> Result<(), String>;

    /// Flush any buffered entries
    fn flush(&mut self) -> Result<(), String>;
}

/// In-memory audit backend
pub struct MemoryBackend {
    entries: Arc<Mutex<VecDeque<AuditEntry>>>,
    max_size: usize,
}

impl MemoryBackend {
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Arc::new(Mutex::new(VecDeque::with_capacity(max_size))),
            max_size,
        }
    }

    pub fn get_entries(&self) -> Vec<AuditEntry> {
        self.entries.lock().unwrap().iter().cloned().collect()
    }

    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }
}

impl AuditBackend for MemoryBackend {
    fn write(&mut self, entry: &AuditEntry) -> Result<(), String> {
        let mut entries = self.entries.lock().unwrap();
        if entries.len() >= self.max_size {
            entries.pop_front();
        }
        entries.push_back(entry.clone());
        Ok(())
    }

    fn flush(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// File-based audit backend
pub struct FileBackend {
    file: Arc<Mutex<File>>,
    buffer: Arc<Mutex<Vec<String>>>,
    buffer_size: usize,
}

impl FileBackend {
    pub fn new(path: PathBuf) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("Failed to open audit log file: {}", e))?;

        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            buffer: Arc::new(Mutex::new(Vec::new())),
            buffer_size: 100,
        })
    }

    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
}

impl AuditBackend for FileBackend {
    fn write(&mut self, entry: &AuditEntry) -> Result<(), String> {
        let json = serde_json::to_string(entry)
            .map_err(|e| format!("Failed to serialize audit entry: {}", e))?;

        let mut buffer = self.buffer.lock().unwrap();
        buffer.push(json);

        if buffer.len() >= self.buffer_size {
            drop(buffer);
            self.flush()?;
        }

        Ok(())
    }

    fn flush(&mut self) -> Result<(), String> {
        let mut buffer = self.buffer.lock().unwrap();
        if buffer.is_empty() {
            return Ok(());
        }

        let mut file = self.file.lock().unwrap();

        for line in buffer.iter() {
            writeln!(file, "{}", line).map_err(|e| format!("Failed to write audit log: {}", e))?;
        }

        file.flush()
            .map_err(|e| format!("Failed to flush audit log: {}", e))?;

        buffer.clear();
        Ok(())
    }
}

/// Stdout audit backend (for development)
pub struct StdoutBackend;

impl AuditBackend for StdoutBackend {
    fn write(&mut self, entry: &AuditEntry) -> Result<(), String> {
        let json = serde_json::to_string(entry)
            .map_err(|e| format!("Failed to serialize audit entry: {}", e))?;
        println!("{}", json);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// Audit logger configuration
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,

    /// Minimum severity level to log
    pub min_severity: AuditSeverity,

    /// Event types to log
    pub event_types: Vec<AuditEventType>,

    /// Include stack traces for errors
    pub include_stack_traces: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_severity: AuditSeverity::Info,
            event_types: vec![
                AuditEventType::Insert,
                AuditEventType::Update,
                AuditEventType::Delete,
                AuditEventType::Query,
                AuditEventType::Auth,
                AuditEventType::Authz,
                AuditEventType::ConfigChange,
            ],
            include_stack_traces: false,
        }
    }
}

/// Audit logger
pub struct AuditLogger {
    config: AuditConfig,
    backends: Arc<Mutex<Vec<Box<dyn AuditBackend>>>>,
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(config: AuditConfig) -> Self {
        Self {
            config,
            backends: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(AuditConfig::default())
    }

    /// Add a backend
    pub fn add_backend(&self, backend: Box<dyn AuditBackend>) {
        self.backends.lock().unwrap().push(backend);
    }

    /// Log an audit entry
    pub fn log(&self, entry: AuditEntry) -> Result<(), String> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check severity filter
        if entry.severity < self.config.min_severity {
            return Ok(());
        }

        // Check event type filter
        if !self.config.event_types.is_empty()
            && !self.config.event_types.contains(&entry.event_type)
        {
            return Ok(());
        }

        // Write to all backends
        let mut backends = self.backends.lock().unwrap();
        for backend in backends.iter_mut() {
            backend.write(&entry)?;
        }

        Ok(())
    }

    /// Flush all backends
    pub fn flush(&self) -> Result<(), String> {
        let mut backends = self.backends.lock().unwrap();
        for backend in backends.iter_mut() {
            backend.flush()?;
        }
        Ok(())
    }

    /// Log an insert operation
    pub fn log_insert(&self, resource: &str, user_id: Option<&str>) -> Result<(), String> {
        let mut entry = AuditEntry::new(AuditEventType::Insert, "insert vector");
        entry = entry.with_resource(resource);
        if let Some(user) = user_id {
            entry = entry.with_user(user);
        }
        self.log(entry)
    }

    /// Log a query operation
    pub fn log_query(
        &self,
        query_type: &str,
        duration_ms: u64,
        user_id: Option<&str>,
    ) -> Result<(), String> {
        let mut entry = AuditEntry::new(AuditEventType::Query, format!("query: {}", query_type));
        entry = entry.with_duration(duration_ms);
        if let Some(user) = user_id {
            entry = entry.with_user(user);
        }
        self.log(entry)
    }

    /// Log a delete operation
    pub fn log_delete(&self, resource: &str, user_id: Option<&str>) -> Result<(), String> {
        let mut entry = AuditEntry::new(AuditEventType::Delete, "delete vector");
        entry = entry
            .with_resource(resource)
            .with_severity(AuditSeverity::Warning);
        if let Some(user) = user_id {
            entry = entry.with_user(user);
        }
        self.log(entry)
    }

    /// Log an authentication event
    pub fn log_auth(&self, user_id: &str, ip: &str, outcome: AuditOutcome) -> Result<(), String> {
        let entry = AuditEntry::new(AuditEventType::Auth, "authentication attempt")
            .with_user(user_id)
            .with_ip(ip)
            .with_outcome(outcome)
            .with_severity(if outcome == AuditOutcome::Success {
                AuditSeverity::Info
            } else {
                AuditSeverity::Warning
            });
        self.log(entry)
    }

    /// Log an authorization event
    pub fn log_authz(
        &self,
        user_id: &str,
        resource: &str,
        action: &str,
        outcome: AuditOutcome,
    ) -> Result<(), String> {
        let entry = AuditEntry::new(AuditEventType::Authz, format!("authz: {}", action))
            .with_user(user_id)
            .with_resource(resource)
            .with_outcome(outcome)
            .with_severity(if outcome == AuditOutcome::Denied {
                AuditSeverity::Warning
            } else {
                AuditSeverity::Info
            });
        self.log(entry)
    }
}

impl Drop for AuditLogger {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_audit_entry_creation() {
        let entry = AuditEntry::new(AuditEventType::Insert, "test action");
        assert_eq!(entry.event_type, AuditEventType::Insert);
        assert_eq!(entry.action, "test action");
        assert_eq!(entry.severity, AuditSeverity::Info);
        assert_eq!(entry.outcome, AuditOutcome::Success);
    }

    #[test]
    fn test_audit_entry_builder() {
        let entry = AuditEntry::new(AuditEventType::Query, "test query")
            .with_severity(AuditSeverity::Warning)
            .with_outcome(AuditOutcome::Failure)
            .with_resource("vector_123")
            .with_user("user_456")
            .with_ip("192.168.1.1")
            .with_duration(100);

        assert_eq!(entry.severity, AuditSeverity::Warning);
        assert_eq!(entry.outcome, AuditOutcome::Failure);
        assert_eq!(entry.resource, Some("vector_123".to_string()));
        assert_eq!(entry.metadata.user_id, Some("user_456".to_string()));
        assert_eq!(entry.metadata.ip_address, Some("192.168.1.1".to_string()));
        assert_eq!(entry.duration_ms, Some(100));
    }

    #[test]
    fn test_memory_backend() {
        let mut backend = MemoryBackend::new(10);

        let entry = AuditEntry::new(AuditEventType::Insert, "test");
        backend.write(&entry).unwrap();

        let entries = backend.get_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].action, "test");
    }

    #[test]
    fn test_memory_backend_overflow() {
        let mut backend = MemoryBackend::new(3);

        for i in 0..5 {
            let entry = AuditEntry::new(AuditEventType::Insert, format!("action_{}", i));
            backend.write(&entry).unwrap();
        }

        let entries = backend.get_entries();
        assert_eq!(entries.len(), 3);
        // Should have the last 3 entries
        assert_eq!(entries[0].action, "action_2");
        assert_eq!(entries[2].action, "action_4");
    }

    #[test]
    fn test_audit_logger() {
        let logger = AuditLogger::default();
        let backend = Box::new(MemoryBackend::new(100));
        let backend_ref = unsafe {
            let ptr = &*backend as *const MemoryBackend;
            &*ptr
        };
        logger.add_backend(backend);

        let entry = AuditEntry::new(AuditEventType::Query, "test query");
        logger.log(entry).unwrap();

        let entries = backend_ref.get_entries();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_severity_filtering() {
        let mut config = AuditConfig::default();
        config.min_severity = AuditSeverity::Warning;

        let logger = AuditLogger::new(config);
        let backend = Box::new(MemoryBackend::new(100));
        let backend_ref = unsafe {
            let ptr = &*backend as *const MemoryBackend;
            &*ptr
        };
        logger.add_backend(backend);

        // Info level - should be filtered
        let entry1 =
            AuditEntry::new(AuditEventType::Query, "info entry").with_severity(AuditSeverity::Info);
        logger.log(entry1).unwrap();

        // Warning level - should be logged
        let entry2 = AuditEntry::new(AuditEventType::Query, "warning entry")
            .with_severity(AuditSeverity::Warning);
        logger.log(entry2).unwrap();

        let entries = backend_ref.get_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].action, "warning entry");
    }

    #[test]
    fn test_event_type_filtering() {
        let mut config = AuditConfig::default();
        config.event_types = vec![AuditEventType::Insert, AuditEventType::Delete];

        let logger = AuditLogger::new(config);
        let backend = Box::new(MemoryBackend::new(100));
        let backend_ref = unsafe {
            let ptr = &*backend as *const MemoryBackend;
            &*ptr
        };
        logger.add_backend(backend);

        // Insert - should be logged
        logger.log_insert("vec_1", Some("user_1")).unwrap();

        // Query - should be filtered
        logger.log_query("knn", 100, Some("user_1")).unwrap();

        // Delete - should be logged
        logger.log_delete("vec_2", Some("user_1")).unwrap();

        let entries = backend_ref.get_entries();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_helper_methods() {
        let logger = AuditLogger::default();
        let backend = Box::new(MemoryBackend::new(100));
        let backend_ref = unsafe {
            let ptr = &*backend as *const MemoryBackend;
            &*ptr
        };
        logger.add_backend(backend);

        logger.log_insert("vec_1", Some("user_1")).unwrap();
        logger.log_query("knn", 50, Some("user_2")).unwrap();
        logger.log_delete("vec_3", Some("user_3")).unwrap();
        logger
            .log_auth("user_4", "192.168.1.1", AuditOutcome::Success)
            .unwrap();
        logger
            .log_authz("user_5", "vec_5", "read", AuditOutcome::Denied)
            .unwrap();

        let entries = backend_ref.get_entries();
        assert_eq!(entries.len(), 5);
    }

    #[test]
    fn test_disabled_logger() {
        let mut config = AuditConfig::default();
        config.enabled = false;

        let logger = AuditLogger::new(config);
        let backend = Box::new(MemoryBackend::new(100));
        let backend_ref = unsafe {
            let ptr = &*backend as *const MemoryBackend;
            &*ptr
        };
        logger.add_backend(backend);

        logger.log_insert("vec_1", Some("user_1")).unwrap();

        let entries = backend_ref.get_entries();
        assert_eq!(entries.len(), 0);
    }
}
