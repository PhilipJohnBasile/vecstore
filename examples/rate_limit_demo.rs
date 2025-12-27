//! Rate Limiting and Throttling Demo
//!
//! Demonstrates protection against overload and abuse with multiple algorithms.

use std::collections::HashMap;
use std::thread;
use std::time::Duration;
use vecstore::*;

fn main() -> anyhow::Result<()> {
    println!("\n Rate Limiting and Throttling Demo\n");
    println!("{}", "=".repeat(70));

    // Test 1: Token Bucket Algorithm
    println!("\n[1/5] Token Bucket Algorithm");
    println!("{}", "-".repeat(70));

    let token_config = RateLimitConfig {
        requests_per_second: 5.0,
        burst_size: 5,
        algorithm: RateLimitAlgorithm::TokenBucket,
        enabled: true,
        operation_limits: HashMap::new(),
    };

    let token_limiter = RateLimiter::new(token_config);

    println!("Configuration:");
    println!("  Algorithm:        Token Bucket");
    println!("  Max requests:     5 per second");
    println!("  Burst size:       5");

    println!("\nSending 7 requests:");
    for i in 1..=7 {
        let result = token_limiter.try_acquire_result();
        let status = if result.allowed {
            "ALLOWED"
        } else {
            "BLOCKED"
        };
        println!(
            "  Request {}: {} (remaining: {})",
            i, status, result.remaining
        );

        if !result.allowed {
            println!("    Retry after: {:?}", result.reset_after);
        }
    }

    // Test 2: Token Bucket with Larger Burst
    println!("\n[2/5] Token Bucket with Larger Burst");
    println!("{}", "-".repeat(70));

    let burst_config = RateLimitConfig {
        requests_per_second: 5.0,
        burst_size: 8,
        algorithm: RateLimitAlgorithm::TokenBucket,
        enabled: true,
        operation_limits: HashMap::new(),
    };

    let burst_limiter = RateLimiter::new(burst_config);

    println!("Configuration:");
    println!("  Algorithm:        Token Bucket");
    println!("  Max requests:     5 per second");
    println!("  Burst size:       8 requests");

    println!("\nSending 10 requests:");
    for i in 1..=10 {
        let result = burst_limiter.try_acquire_result();
        let status = if result.allowed {
            "ALLOWED"
        } else {
            "BLOCKED"
        };
        println!(
            "  Request {}: {} (remaining: {})",
            i, status, result.remaining
        );
    }

    // Test 3: Sliding Window Algorithm
    println!("\n[3/5] Sliding Window Algorithm");
    println!("{}", "-".repeat(70));

    let sliding_config = RateLimitConfig {
        requests_per_second: 10.0,
        burst_size: 10,
        algorithm: RateLimitAlgorithm::SlidingWindow,
        enabled: true,
        operation_limits: HashMap::new(),
    };

    let sliding_limiter = RateLimiter::new(sliding_config);

    println!("Configuration:");
    println!("  Algorithm:        Sliding Window");
    println!("  Max requests:     10 per second");

    println!("\nSending 12 requests:");
    for i in 1..=12 {
        let result = sliding_limiter.try_acquire_result();
        let status = if result.allowed {
            "ALLOWED"
        } else {
            "BLOCKED"
        };
        println!(
            "  Request {}: {} (remaining: {})",
            i, status, result.remaining
        );
    }

    // Wait for window to slide
    println!("\n  Waiting 1100ms for window to slide...");
    thread::sleep(Duration::from_millis(1100));

    println!("\n  After window slides:");
    for i in 13..=15 {
        let result = sliding_limiter.try_acquire_result();
        let status = if result.allowed {
            "ALLOWED"
        } else {
            "BLOCKED"
        };
        println!(
            "  Request {}: {} (remaining: {})",
            i, status, result.remaining
        );
    }

    // Test 4: Leaky Bucket Algorithm
    println!("\n[4/5] Leaky Bucket Algorithm");
    println!("{}", "-".repeat(70));

    let leaky_config = RateLimitConfig {
        requests_per_second: 5.0,
        burst_size: 5,
        algorithm: RateLimitAlgorithm::LeakyBucket,
        enabled: true,
        operation_limits: HashMap::new(),
    };

    let leaky_limiter = RateLimiter::new(leaky_config);

    println!("Configuration:");
    println!("  Algorithm:        Leaky Bucket");
    println!("  Leak rate:        5 per second");
    println!("  Queue size:       5");

    println!("\nSending 8 requests:");
    for i in 1..=8 {
        let result = leaky_limiter.try_acquire_result();
        let status = if result.allowed {
            "ALLOWED"
        } else {
            "BLOCKED"
        };
        println!(
            "  Request {}: {} (remaining: {})",
            i, status, result.remaining
        );
    }

    // Test 5: Per-Key Rate Limiting
    println!("\n[5/5] Per-Key Rate Limiting");
    println!("{}", "-".repeat(70));

    let per_key_config = RateLimitConfig {
        requests_per_second: 3.0,
        burst_size: 3,
        algorithm: RateLimitAlgorithm::TokenBucket,
        enabled: true,
        operation_limits: HashMap::new(),
    };

    let per_key_limiter = KeyedRateLimiter::new(per_key_config);

    println!("Configuration: 3 requests/sec per key");

    println!("\nKey 'userA' sends 4 requests:");
    for i in 1..=4 {
        let allowed = per_key_limiter.try_acquire("userA");
        let status = if allowed { "ALLOWED" } else { "BLOCKED" };
        println!("  Request {}: {}", i, status);
    }

    println!("\nKey 'userB' sends 4 requests:");
    for i in 1..=4 {
        let allowed = per_key_limiter.try_acquire("userB");
        let status = if allowed { "ALLOWED" } else { "BLOCKED" };
        println!("  Request {}: {}", i, status);
    }

    println!("\n Both keys have independent quotas!");

    // Performance test
    println!("\n{}", "=".repeat(70));
    println!(" Performance Test");
    println!("{}", "=".repeat(70));

    let perf_config = RateLimitConfig {
        requests_per_second: 100000.0,
        burst_size: 100000,
        algorithm: RateLimitAlgorithm::TokenBucket,
        enabled: true,
        operation_limits: HashMap::new(),
    };
    let perf_limiter = RateLimiter::new(perf_config);

    let start = std::time::Instant::now();
    let iterations = 100_000;

    for _ in 0..iterations {
        perf_limiter.try_acquire();
    }

    let elapsed = start.elapsed();
    let throughput = iterations as f64 / elapsed.as_secs_f64();

    println!("\nChecked {} requests", iterations);
    println!("Time elapsed:     {:?}", elapsed);
    println!("Throughput:       {:.0} checks/sec", throughput);

    // Summary
    println!("\n{}", "=".repeat(70));
    println!(" Demo Complete!");
    println!("{}", "=".repeat(70));

    println!("\n Key Features Demonstrated:");
    println!("   Token Bucket algorithm (smooth rate limiting)");
    println!("   Sliding Window algorithm (precise tracking)");
    println!("   Leaky Bucket algorithm (queue-based)");
    println!("   Per-key isolation for multi-tenant scenarios");
    println!("   Automatic token refilling");
    println!(
        "   High-performance ({}K+ checks/sec)",
        (throughput / 1000.0) as u32
    );

    println!("\n Algorithm Comparison:");
    println!("\n  Token Bucket:");
    println!("    + Smooth rate limiting");
    println!("    + Supports bursts");
    println!("    + Memory efficient");
    println!("    - Less precise at boundaries");

    println!("\n  Sliding Window:");
    println!("    + Most accurate");
    println!("    + No edge case bursts");
    println!("    - Higher memory usage");
    println!("    - Slightly slower");

    println!("\n  Leaky Bucket:");
    println!("    + Enforces strict rate");
    println!("    + Good for traffic shaping");
    println!("    - Queues requests");
    println!("    - Higher latency");

    println!("\n Use Cases:");
    println!("   API rate limiting");
    println!("   DDoS protection");
    println!("   Resource quota management");
    println!("   Fair usage policies");
    println!("   Cost control");
    println!("   Load shedding");

    println!();

    Ok(())
}
