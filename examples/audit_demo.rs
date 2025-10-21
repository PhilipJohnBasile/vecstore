//! Audit Logging Demo
//!
//! Demonstrates comprehensive audit trails for compliance and security.

use std::fs;
use std::path::PathBuf;
use vecstore::*;

fn main() -> anyhow::Result<()> {
    println!("\n📋 Audit Logging Demo\n");
    println!("{}", "=".repeat(70));

    // Test 1: Basic Audit Logging
    println!("\n[1/6] Basic Audit Logging with Memory Backend");
    println!("{}", "-".repeat(70));

    let logger = AuditLogger::default();
    let backend = Box::new(MemoryBackend::new(1000));
    let backend_ref = unsafe {
        let ptr = &*backend as *const MemoryBackend;
        &*ptr
    };
    logger.add_backend(backend);

    println!("Configuration:");
    println!("  Backend:          Memory (1000 entries)");
    println!("  Min severity:     Info");

    // Log various operations
    logger.log_insert("vector_001", Some("alice")).unwrap();
    logger.log_query("knn_search", 45, Some("bob")).unwrap();
    logger.log_delete("vector_002", Some("charlie")).unwrap();

    println!("\nLogged 3 operations:");
    println!("  ✓ Insert by alice");
    println!("  ✓ Query by bob (45ms)");
    println!("  ✓ Delete by charlie");

    let entries = backend_ref.get_entries();
    println!("\nStored entries: {}", entries.len());

    for (i, entry) in entries.iter().take(3).enumerate() {
        println!("\n  Entry {}:", i + 1);
        println!("    Type:      {:?}", entry.event_type);
        println!("    Action:    {}", entry.action);
        println!("    User:      {:?}", entry.metadata.user_id);
        println!("    Outcome:   {:?}", entry.outcome);
        if let Some(duration) = entry.duration_ms {
            println!("    Duration:  {}ms", duration);
        }
    }

    // Test 2: Authentication Events
    println!("\n[2/6] Authentication and Authorization Logging");
    println!("{}", "-".repeat(70));

    logger
        .log_auth("alice", "192.168.1.100", AuditOutcome::Success)
        .unwrap();
    logger
        .log_auth("mallory", "10.0.0.50", AuditOutcome::Failure)
        .unwrap();

    logger
        .log_authz("alice", "vector_123", "read", AuditOutcome::Success)
        .unwrap();
    logger
        .log_authz("mallory", "vector_456", "write", AuditOutcome::Denied)
        .unwrap();

    println!("Logged authentication events:");
    println!("  ✓ alice login from 192.168.1.100 - Success");
    println!("  ✓ mallory login from 10.0.0.50 - Failure");

    println!("\nLogged authorization events:");
    println!("  ✓ alice read vector_123 - Success");
    println!("  ✓ mallory write vector_456 - Denied");

    let all_entries = backend_ref.get_entries();
    let auth_entries: Vec<_> = all_entries
        .iter()
        .filter(|e| e.event_type == AuditEventType::Auth)
        .collect();
    let authz_entries: Vec<_> = all_entries
        .iter()
        .filter(|e| e.event_type == AuditEventType::Authz)
        .collect();

    println!("\nAuth events:   {}", auth_entries.len());
    println!("Authz events:  {}", authz_entries.len());

    // Test 3: Custom Audit Entries
    println!("\n[3/6] Custom Audit Entries");
    println!("{}", "-".repeat(70));

    let custom_entry = AuditEntry::new(AuditEventType::ConfigChange, "update rate limit")
        .with_severity(AuditSeverity::Warning)
        .with_user("admin")
        .with_ip("192.168.1.1")
        .with_resource("rate_limiter_config")
        .with_details("Changed max_requests from 100 to 200");

    logger.log(custom_entry).unwrap();

    println!("Logged custom configuration change:");
    println!("  Type:      ConfigChange");
    println!("  Severity:  Warning");
    println!("  User:      admin");
    println!("  Resource:  rate_limiter_config");
    println!("  Details:   Changed max_requests from 100 to 200");

    // Test 4: Severity Filtering
    println!("\n[4/6] Severity-Based Filtering");
    println!("{}", "-".repeat(70));

    let mut warning_config = AuditConfig::default();
    warning_config.min_severity = AuditSeverity::Warning;

    let filtered_logger = AuditLogger::new(warning_config);
    let filtered_backend = Box::new(MemoryBackend::new(1000));
    let filtered_backend_ref = unsafe {
        let ptr = &*filtered_backend as *const MemoryBackend;
        &*ptr
    };
    filtered_logger.add_backend(filtered_backend);

    println!("Configuration: min_severity = Warning");

    // These should be filtered out
    filtered_logger
        .log_insert("vector_999", Some("user1"))
        .unwrap();
    filtered_logger
        .log_query("search", 30, Some("user2"))
        .unwrap();

    // These should be logged
    let warning_entry = AuditEntry::new(AuditEventType::Delete, "delete operation")
        .with_severity(AuditSeverity::Warning);
    filtered_logger.log(warning_entry).unwrap();

    let error_entry = AuditEntry::new(AuditEventType::Query, "failed query")
        .with_severity(AuditSeverity::Error)
        .with_outcome(AuditOutcome::Failure);
    filtered_logger.log(error_entry).unwrap();

    let filtered_entries = filtered_backend_ref.get_entries();
    println!("\nLogged 4 events (2 Info, 1 Warning, 1 Error)");
    println!(
        "Filtered result: {} entries (only Warning+)",
        filtered_entries.len()
    );

    for entry in &filtered_entries {
        println!("  ✓ {:?} - {:?}", entry.severity, entry.action);
    }

    // Test 5: Event Type Filtering
    println!("\n[5/6] Event Type Filtering");
    println!("{}", "-".repeat(70));

    let mut type_config = AuditConfig::default();
    type_config.event_types = vec![
        AuditEventType::Auth,
        AuditEventType::Authz,
        AuditEventType::Delete,
    ];

    let type_logger = AuditLogger::new(type_config);
    let type_backend = Box::new(MemoryBackend::new(1000));
    let type_backend_ref = unsafe {
        let ptr = &*type_backend as *const MemoryBackend;
        &*ptr
    };
    type_logger.add_backend(type_backend);

    println!("Configuration: event_types = [Auth, Authz, Delete]");

    // Should be logged
    type_logger
        .log_auth("user1", "10.0.0.1", AuditOutcome::Success)
        .unwrap();
    type_logger.log_delete("vec_1", Some("user2")).unwrap();

    // Should be filtered
    type_logger.log_insert("vec_2", Some("user3")).unwrap();
    type_logger.log_query("search", 10, Some("user4")).unwrap();

    let type_entries = type_backend_ref.get_entries();
    println!("\nLogged 4 events (2 allowed types, 2 filtered)");
    println!("Filtered result: {} entries", type_entries.len());

    for entry in &type_entries {
        println!("  ✓ {:?} - {}", entry.event_type, entry.action);
    }

    // Test 6: File Backend
    println!("\n[6/6] File-Based Audit Logging");
    println!("{}", "-".repeat(70));

    let audit_file = PathBuf::from("/tmp/vecstore_audit.log");

    // Clean up any existing file
    let _ = fs::remove_file(&audit_file);

    let file_logger = AuditLogger::default();
    let file_backend = Box::new(
        FileBackend::new(audit_file.clone())
            .unwrap()
            .with_buffer_size(5),
    );
    file_logger.add_backend(file_backend);

    println!("Configuration:");
    println!("  Backend:      File");
    println!("  Path:         /tmp/vecstore_audit.log");
    println!("  Buffer size:  5 entries");

    // Log some events
    for i in 1..=10 {
        file_logger
            .log_insert(&format!("vector_{:03}", i), Some("user_file"))
            .unwrap();
    }

    // Flush to ensure all entries are written
    file_logger.flush().unwrap();

    println!("\nLogged 10 insert operations");

    // Read and display file contents
    if let Ok(contents) = fs::read_to_string(&audit_file) {
        let line_count = contents.lines().count();
        println!("File entries:     {}", line_count);

        println!("\nFirst 3 entries:");
        for (i, line) in contents.lines().take(3).enumerate() {
            if let Ok(entry) = serde_json::from_str::<AuditEntry>(line) {
                println!(
                    "  {}. {:?} - {} by {:?}",
                    i + 1,
                    entry.event_type,
                    entry.action,
                    entry.metadata.user_id
                );
            }
        }
    }

    // Clean up
    let _ = fs::remove_file(&audit_file);

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("✅ Demo Complete!");
    println!("{}", "=".repeat(70));

    println!("\n✨ Key Features Demonstrated:");
    println!("  ✓ Multiple audit backends (Memory, File, Stdout)");
    println!("  ✓ Event types (Insert, Query, Delete, Auth, Authz, etc.)");
    println!("  ✓ Severity levels (Debug, Info, Warning, Error, Critical)");
    println!("  ✓ Outcome tracking (Success, Failure, Denied)");
    println!("  ✓ User and IP tracking");
    println!("  ✓ Duration tracking for operations");
    println!("  ✓ Severity-based filtering");
    println!("  ✓ Event type filtering");
    println!("  ✓ Custom audit entries");
    println!("  ✓ Structured JSON logging");
    println!("  ✓ Buffered file writing");

    println!("\n📊 Audit Event Types:");
    println!("  • Insert:        Vector insertion");
    println!("  • Update:        Vector updates");
    println!("  • Delete:        Vector deletion");
    println!("  • Query:         Search operations");
    println!("  • Auth:          Authentication attempts");
    println!("  • Authz:         Authorization checks");
    println!("  • ConfigChange:  Configuration modifications");
    println!("  • Backup:        Backup operations");
    println!("  • Export/Import: Data transfer");

    println!("\n🎯 Use Cases:");
    println!("  • Compliance and regulatory requirements");
    println!("  • Security monitoring and forensics");
    println!("  • Access tracking and accountability");
    println!("  • Debugging and troubleshooting");
    println!("  • Performance analysis");
    println!("  • User activity analytics");

    println!();

    Ok(())
}
