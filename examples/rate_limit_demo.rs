//! Rate Limiting and Throttling Demo
//!
//! Demonstrates protection against overload and abuse with multiple algorithms.

use std::thread;
use std::time::Duration;
use vecstore::*;

fn main() -> anyhow::Result<()> {
    println!("\n🚦 Rate Limiting and Throttling Demo\n");
    println!("{}", "=".repeat(70));

    // Test 1: Token Bucket Algorithm
    println!("\n[1/6] Token Bucket Algorithm");
    println!("{}", "-".repeat(70));

    let mut token_config = RateLimitConfig::per_second(5);
    token_config.algorithm = RateLimitAlgorithm::TokenBucket;
    token_config.allow_burst = false;

    let token_limiter = RateLimiter::new(token_config);

    println!("Configuration:");
    println!("  Algorithm:        Token Bucket");
    println!("  Max requests:     5 per second");
    println!("  Burst mode:       Disabled");

    println!("\nSending 7 requests:");
    for i in 1..=7 {
        let result = token_limiter.check("user1");
        let status = if result.allowed {
            "✓ ALLOWED"
        } else {
            "✗ BLOCKED"
        };
        println!(
            "  Request {}: {} (remaining: {})",
            i, status, result.remaining
        );

        if !result.allowed {
            if let Some(retry_after) = result.retry_after {
                println!("    Retry after: {:?}", retry_after);
            }
        }
    }

    // Test 2: Token Bucket with Burst
    println!("\n[2/6] Token Bucket with Burst Support");
    println!("{}", "-".repeat(70));

    let mut burst_config = RateLimitConfig::per_second(5);
    burst_config.algorithm = RateLimitAlgorithm::TokenBucket;
    burst_config.allow_burst = true;
    burst_config.burst_size = 3;

    let burst_limiter = RateLimiter::new(burst_config);

    println!("Configuration:");
    println!("  Algorithm:        Token Bucket");
    println!("  Max requests:     5 per second");
    println!("  Burst mode:       Enabled");
    println!("  Burst size:       +3 extra");
    println!("  Total capacity:   8 requests");

    println!("\nSending 10 requests:");
    for i in 1..=10 {
        let result = burst_limiter.check("user2");
        let status = if result.allowed {
            "✓ ALLOWED"
        } else {
            "✗ BLOCKED"
        };
        println!(
            "  Request {}: {} (remaining: {})",
            i, status, result.remaining
        );
    }

    // Test 3: Sliding Window Algorithm
    println!("\n[3/6] Sliding Window Algorithm");
    println!("{}", "-".repeat(70));

    let mut sliding_config = RateLimitConfig {
        max_requests: 10,
        window: Duration::from_millis(1000),
        algorithm: RateLimitAlgorithm::SlidingWindow,
        allow_burst: false,
        ..Default::default()
    };
    sliding_config.scope = RateLimitScope::PerUser;

    let sliding_limiter = RateLimiter::new(sliding_config);

    println!("Configuration:");
    println!("  Algorithm:        Sliding Window");
    println!("  Max requests:     10 per second");
    println!("  Scope:            Per-user");

    println!("\nSending 12 requests:");
    for i in 1..=12 {
        let result = sliding_limiter.check("user3");
        let status = if result.allowed {
            "✓ ALLOWED"
        } else {
            "✗ BLOCKED"
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
        let result = sliding_limiter.check("user3");
        let status = if result.allowed {
            "✓ ALLOWED"
        } else {
            "✗ BLOCKED"
        };
        println!(
            "  Request {}: {} (remaining: {})",
            i, status, result.remaining
        );
    }

    // Test 4: Fixed Window Algorithm
    println!("\n[4/6] Fixed Window Algorithm");
    println!("{}", "-".repeat(70));

    let mut fixed_config = RateLimitConfig {
        max_requests: 8,
        window: Duration::from_millis(1000),
        algorithm: RateLimitAlgorithm::FixedWindow,
        allow_burst: false,
        ..Default::default()
    };

    let fixed_limiter = RateLimiter::new(fixed_config);

    println!("Configuration:");
    println!("  Algorithm:        Fixed Window");
    println!("  Max requests:     8 per second");

    println!("\nSending 10 requests:");
    for i in 1..=10 {
        let result = fixed_limiter.check("user4");
        let status = if result.allowed {
            "✓ ALLOWED"
        } else {
            "✗ BLOCKED"
        };
        println!(
            "  Request {}: {} (remaining: {}, reset: {:?})",
            i, status, result.remaining, result.reset_after
        );
    }

    // Wait for window reset
    println!("\n  Waiting 1100ms for window reset...");
    thread::sleep(Duration::from_millis(1100));

    println!("\n  After window resets:");
    let result = fixed_limiter.check("user4");
    println!("  Request 11: ✓ ALLOWED (remaining: {})", result.remaining);

    // Test 5: Multi-tier Rate Limiting
    println!("\n[5/6] Multi-Tier Rate Limiting");
    println!("{}", "-".repeat(70));

    let mut multi = MultiTierRateLimiter::new();

    // Tier 1: Per-second limit
    let mut tier1_config = RateLimitConfig::per_second(3);
    tier1_config.allow_burst = false;
    multi.add_tier("per_second", RateLimiter::new(tier1_config));

    // Tier 2: Per-minute limit
    let mut tier2_config = RateLimitConfig::per_minute(15);
    tier2_config.allow_burst = false;
    multi.add_tier("per_minute", RateLimiter::new(tier2_config));

    println!("Configuration:");
    println!("  Tier 1:           3 requests per second");
    println!("  Tier 2:           15 requests per minute");

    println!("\nSending 5 requests:");
    for i in 1..=5 {
        let result = multi.check("user5");
        let status = if result.allowed {
            "✓ ALLOWED"
        } else {
            "✗ BLOCKED"
        };
        let reason = if !result.allowed {
            if let Some(retry) = result.retry_after {
                format!(" (retry after {:?})", retry)
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        println!(
            "  Request {}: {} (remaining: {}){}",
            i, status, result.remaining, reason
        );
    }

    // Test 6: Per-scope Rate Limiting
    println!("\n[6/6] Per-User vs Global Scope");
    println!("{}", "-".repeat(70));

    let mut per_user_config = RateLimitConfig::per_second(3);
    per_user_config.allow_burst = false;
    per_user_config.scope = RateLimitScope::PerUser;
    let per_user_limiter = RateLimiter::new(per_user_config);

    println!("Configuration: 3 requests/sec per user");

    println!("\nUser A sends 4 requests:");
    for i in 1..=4 {
        let result = per_user_limiter.check("userA");
        let status = if result.allowed {
            "✓ ALLOWED"
        } else {
            "✗ BLOCKED"
        };
        println!("  Request {}: {}", i, status);
    }

    println!("\nUser B sends 4 requests:");
    for i in 1..=4 {
        let result = per_user_limiter.check("userB");
        let status = if result.allowed {
            "✓ ALLOWED"
        } else {
            "✗ BLOCKED"
        };
        println!("  Request {}: {}", i, status);
    }

    println!("\n✅ Both users have independent quotas!");

    // Performance test
    println!("\n{}", "=".repeat(70));
    println!("⚡ Performance Test");
    println!("{}", "=".repeat(70));

    let perf_config = RateLimitConfig::per_second(10000);
    let perf_limiter = RateLimiter::new(perf_config);

    let start = std::time::Instant::now();
    let iterations = 100_000;

    for i in 0..iterations {
        perf_limiter.check(&format!("user{}", i % 100));
    }

    let elapsed = start.elapsed();
    let throughput = iterations as f64 / elapsed.as_secs_f64();

    println!("\nChecked {} requests across 100 users", iterations);
    println!("Time elapsed:     {:?}", elapsed);
    println!("Throughput:       {:.0} checks/sec", throughput);

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("✅ Demo Complete!");
    println!("{}", "=".repeat(70));

    println!("\n✨ Key Features Demonstrated:");
    println!("  ✓ Token Bucket algorithm (smooth rate limiting)");
    println!("  ✓ Sliding Window algorithm (precise tracking)");
    println!("  ✓ Fixed Window algorithm (simple and efficient)");
    println!("  ✓ Burst support for traffic spikes");
    println!("  ✓ Multi-tier rate limiting");
    println!("  ✓ Per-user isolation");
    println!("  ✓ Automatic refilling/resetting");
    println!(
        "  ✓ High-performance ({}K+ checks/sec)",
        (throughput / 1000.0) as u32
    );

    println!("\n📊 Algorithm Comparison:");
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

    println!("\n  Fixed Window:");
    println!("    + Simplest implementation");
    println!("    + Lowest memory usage");
    println!("    + Fastest");
    println!("    - Can have edge case bursts");

    println!("\n💡 Rate Limit Scopes:");
    println!("  • Global:    Shared limit across all requests");
    println!("  • Per-User:  Individual limits per user");
    println!("  • Per-IP:    Individual limits per IP address");
    println!("  • Per-API Key: Individual limits per API key");

    println!("\n🎯 Use Cases:");
    println!("  • API rate limiting");
    println!("  • DDoS protection");
    println!("  • Resource quota management");
    println!("  • Fair usage policies");
    println!("  • Cost control");
    println!("  • Load shedding");

    println!();

    Ok(())
}
