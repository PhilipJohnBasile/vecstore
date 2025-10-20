//! Vector Database Monitoring and Alerting Demo
//!
//! Demonstrates real-time monitoring with configurable alerts.

use std::thread;
use std::time::Duration;
use vecstore::*;

fn main() -> anyhow::Result<()> {
    println!("\n📊 Vector Database Monitoring and Alerting Demo\n");
    println!("{}", "=".repeat(70));

    // Create monitor with custom configuration
    println!("\n[1/6] Setting Up Monitor");
    println!("{}", "-".repeat(70));

    let config = MonitoringConfig {
        max_history_size: 100,
        collection_interval: Duration::from_secs(5),
        enable_alerts: true,
        alert_cooldown: Duration::from_millis(100),
        enable_aggregation: true,
    };

    let mut monitor = Monitor::new(config);
    println!("✓ Monitor initialized with custom configuration");
    println!("  Max history size: 100 samples");
    println!("  Alert cooldown:   100ms");

    // Add alert rules
    println!("\n[2/6] Configuring Alert Rules");
    println!("{}", "-".repeat(70));

    // Add preset rules
    monitor.add_rule(AlertPresets::high_query_latency(100.0));
    monitor.add_rule(AlertPresets::low_cache_hit_rate(0.7));
    monitor.add_rule(AlertPresets::high_error_rate(0.05));
    monitor.add_rule(AlertPresets::low_vector_quality(0.6));

    println!("✓ Added 4 alert rules:");
    println!("  • High query latency   (> 100ms)");
    println!("  • Low cache hit rate   (< 0.7)");
    println!("  • High error rate      (> 0.05)");
    println!("  • Low vector quality   (< 0.6)");

    // Simulate normal operations
    println!("\n[3/6] Simulating Normal Operations");
    println!("{}", "-".repeat(70));

    for i in 0..10 {
        // Good metrics - no alerts
        monitor.record(MetricType::QueryLatency, 50.0 + (i as f64 * 2.0));
        monitor.record(MetricType::CacheHitRate, 0.85);
        monitor.record(MetricType::ErrorRate, 0.01);
        monitor.record(MetricType::VectorQuality, 0.92);
    }

    println!("✓ Recorded 10 samples of normal metrics");
    println!("  Query latency:    50-68 ms");
    println!("  Cache hit rate:   0.85");
    println!("  Error rate:       0.01");
    println!("  Vector quality:   0.92");

    // Check metrics
    if let Some(history) = monitor.get_metric(MetricType::QueryLatency) {
        println!("\n📈 Query Latency Statistics:");
        println!("  Latest:           {:.2} ms", history.latest().unwrap());
        println!("  Average:          {:.2} ms", history.average().unwrap());
        println!(
            "  P50:              {:.2} ms",
            history.percentile(50.0).unwrap()
        );
        println!(
            "  P95:              {:.2} ms",
            history.percentile(95.0).unwrap()
        );
    }

    // Verify no alerts
    let alerts = monitor.get_alerts(10);
    println!("\n⚠️  Alerts triggered:   {}", alerts.len());

    // Simulate problematic operations
    println!("\n[4/6] Simulating Performance Issues");
    println!("{}", "-".repeat(70));

    // Trigger high latency alert
    monitor.record(MetricType::QueryLatency, 150.0);
    println!("⚡ Recorded high query latency: 150ms");

    // Trigger low cache hit rate alert
    monitor.record(MetricType::CacheHitRate, 0.5);
    println!("⚡ Recorded low cache hit rate: 0.50");

    // Trigger high error rate alert
    monitor.record(MetricType::ErrorRate, 0.1);
    println!("⚡ Recorded high error rate: 0.10");

    // Trigger low quality alert
    monitor.record(MetricType::VectorQuality, 0.4);
    println!("⚡ Recorded low vector quality: 0.40");

    // View triggered alerts
    println!("\n[5/6] Alert Summary");
    println!("{}", "-".repeat(70));

    let alerts = monitor.get_alerts(10);
    println!("\n✓ {} alerts triggered:\n", alerts.len());

    for (i, alert) in alerts.iter().enumerate() {
        let severity_icon = match alert.severity {
            MonitorAlertSeverity::Info => "ℹ️",
            MonitorAlertSeverity::Warning => "⚠️",
            MonitorAlertSeverity::Error => "❌",
            MonitorAlertSeverity::Critical => "🚨",
        };

        println!("{}. {} {:?}", i + 1, severity_icon, alert.severity);
        println!("   Category: {:?}", alert.category);
        println!("   Message:  {}", alert.message);
        println!();
    }

    // Filter alerts by severity
    let warnings = monitor.get_alerts_by_severity(MonitorAlertSeverity::Warning);
    let errors = monitor.get_alerts_by_severity(MonitorAlertSeverity::Error);

    println!("Alert breakdown:");
    println!("  Warnings:         {}", warnings.len());
    println!("  Errors:           {}", errors.len());

    // Test alert cooldown
    println!("\n[6/6] Testing Alert Cooldown");
    println!("{}", "-".repeat(70));

    let initial_count = monitor.get_alerts(100).len();

    // Immediate re-trigger - should be suppressed
    monitor.record(MetricType::QueryLatency, 200.0);
    let suppressed_count = monitor.get_alerts(100).len();

    println!("Initial alerts:       {}", initial_count);
    println!(
        "After re-trigger:     {} (suppressed by cooldown)",
        suppressed_count
    );

    // Wait for cooldown
    thread::sleep(Duration::from_millis(150));

    // Re-trigger after cooldown - should create new alert
    monitor.record(MetricType::QueryLatency, 250.0);
    let after_cooldown_count = monitor.get_alerts(100).len();

    println!(
        "After cooldown:       {} (new alert created)",
        after_cooldown_count
    );

    // Generate comprehensive report
    println!("\n{}", "=".repeat(70));
    println!("📄 Monitoring Report");
    println!("{}", "=".repeat(70));

    let report = monitor.generate_report();

    println!("\n📊 Current Metrics:");
    for (name, value) in &report.metrics {
        println!("  {:<25} {:.3}", name, value);
    }

    println!("\n📈 Statistics:");
    println!("  Total alerts:         {}", report.stats.total_alerts);
    println!("  Active rules:         {}", report.stats.active_rules);
    println!("  Metrics tracked:      {}", report.stats.metrics_tracked);
    println!("  Uptime:               {:?}", report.stats.uptime);

    println!("\n⚠️  Alerts by Severity:");
    for (severity, count) in &report.stats.alerts_by_severity {
        println!("  {:?}: {}", severity, count);
    }

    println!("\n📂 Alerts by Category:");
    for (category, count) in &report.stats.alerts_by_category {
        println!("  {:?}: {}", category, count);
    }

    // Test Prometheus export
    println!("\n{}", "=".repeat(70));
    println!("📊 Prometheus Export");
    println!("{}", "=".repeat(70));

    let prometheus_output = monitor.export_prometheus();
    println!("\n{}", prometheus_output);

    // Test rule management
    println!("{}", "=".repeat(70));
    println!("⚙️  Rule Management");
    println!("{}", "=".repeat(70));

    println!("\nRemoving 'high_query_latency' rule...");
    let removed = monitor.remove_rule("high_query_latency");
    println!("✓ Rule removed: {}", removed);

    // Record high latency - should not trigger alert
    monitor.record(MetricType::QueryLatency, 300.0);
    let final_count = monitor.get_alerts(100).len();
    println!("Recorded 300ms latency - no new alert (rule disabled)");
    println!("Final alert count: {}", final_count);

    // Test with default preset rules
    println!("\n{}", "=".repeat(70));
    println!("📋 Default Preset Rules");
    println!("{}", "=".repeat(70));

    let default_rules = AlertPresets::default_rules();
    println!("\nAvailable preset rules ({} total):", default_rules.len());

    for rule in &default_rules {
        println!("\n  • {}", rule.name);
        println!("    Category:  {:?}", rule.category);
        println!("    Severity:  {:?}", rule.severity);
        println!("    Threshold: {:.2}", rule.threshold);
    }

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("✅ Demo Complete!");
    println!("{}", "=".repeat(70));

    println!("\n✨ Key Features Demonstrated:");
    println!("  ✓ Real-time metric recording");
    println!("  ✓ Configurable alert rules");
    println!("  ✓ Alert severity levels (Info, Warning, Error, Critical)");
    println!("  ✓ Alert categories (Performance, DataQuality, Storage, etc.)");
    println!("  ✓ Alert cooldown to prevent spam");
    println!("  ✓ Metric history with statistics (avg, percentile)");
    println!("  ✓ Alert filtering by severity and category");
    println!("  ✓ Comprehensive monitoring reports");
    println!("  ✓ Prometheus export format");
    println!("  ✓ Dynamic rule management");
    println!("  ✓ Preset alert rules for common scenarios");

    println!("\n💡 Metric Types:");
    println!("  • Query/Insert Latency");
    println!("  • Throughput (QPS)");
    println!("  • Vector Quality");
    println!("  • Duplicate/Outlier Rate");
    println!("  • Storage/Memory Usage");
    println!("  • Index Fragmentation");
    println!("  • Cache Hit Rate");
    println!("  • Error Rate");

    println!("\n🎯 Use Cases:");
    println!("  • Production monitoring and observability");
    println!("  • Performance degradation detection");
    println!("  • Data quality tracking");
    println!("  • Capacity planning");
    println!("  • SLA compliance monitoring");
    println!("  • Integration with alerting systems");

    println!();

    Ok(())
}
