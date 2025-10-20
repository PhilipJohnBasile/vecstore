//! SIMD-accelerated distance calculations
//!
//! This module provides vectorized implementations of distance metrics using SIMD instructions
//! for significant performance improvements (typically 4-8x faster than scalar code).
//!
//! The implementation uses architecture-specific intrinsics when available (x86 AVX/SSE, ARM NEON),
//! with automatic fallback to scalar implementations on unsupported platforms.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

/// Calculate Euclidean (L2) distance using SIMD acceleration
///
/// This function uses platform-specific SIMD instructions:
/// - x86/x86_64: AVX2 if available, otherwise SSE2
/// - ARM: NEON if available
/// - Fallback: Optimized scalar code
///
/// # Performance
/// - Typically 4-8x faster than naive scalar implementation
/// - Best performance with vectors that are multiples of SIMD width (8 floats for AVX)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Panics
/// Panics if vectors have different lengths
#[inline]
pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        euclidean_distance_avx2(a, b)
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx2"),
        target_feature = "sse2"
    ))]
    unsafe {
        euclidean_distance_sse2(a, b)
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        euclidean_distance_neon(a, b)
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse2"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    euclidean_distance_scalar(a, b)
}

/// Optimized scalar fallback for Euclidean distance
#[inline]
#[allow(dead_code)]
fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// AVX2 implementation for x86_64 (processes 8 floats at once)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 floats at a time
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        let squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
        i += 8;
    }

    // Horizontal sum of AVX register
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_high, sum_low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remaining elements
    while i < len {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result.sqrt()
}

/// SSE2 implementation for x86_64 (processes 4 floats at once)
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn euclidean_distance_sse2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(va, vb);
        let squared = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, squared);
        i += 4;
    }

    // Horizontal sum
    let sum_shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
    let sums = _mm_add_ps(sum, sum_shuf);
    let sums_shuf = _mm_movehl_ps(sums, sums);
    let result_ss = _mm_add_ss(sums, sums_shuf);
    let mut result = _mm_cvtss_f32(result_ss);

    // Handle remaining elements
    while i < len {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result.sqrt()
}

/// ARM NEON implementation (processes 4 floats at once)
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        let squared = vmulq_f32(diff, diff);
        sum = vaddq_f32(sum, squared);
        i += 4;
    }

    // Horizontal sum
    let mut result = vgetq_lane_f32(sum, 0)
        + vgetq_lane_f32(sum, 1)
        + vgetq_lane_f32(sum, 2)
        + vgetq_lane_f32(sum, 3);

    // Handle remaining elements
    while i < len {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result.sqrt()
}

/// Calculate cosine similarity using SIMD acceleration
///
/// Returns value in range [-1, 1] where 1 means identical direction,
/// 0 means orthogonal, and -1 means opposite direction.
#[inline]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let dot = dot_product_simd(a, b);
    let norm_a = magnitude_simd(a);
    let norm_b = magnitude_simd(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Calculate dot product using SIMD
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return dot_product_avx2(a, b);
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx2"),
        target_feature = "sse2"
    ))]
    unsafe {
        return dot_product_sse2(a, b);
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let prod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, prod);
        i += 8;
    }

    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_high, sum_low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
    let mut result = _mm_cvtss_f32(sum32);

    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn dot_product_sse2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let prod = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, prod);
        i += 4;
    }

    let sum_shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
    let sums = _mm_add_ps(sum, sum_shuf);
    let sums_shuf = _mm_movehl_ps(sums, sums);
    let result_ss = _mm_add_ss(sums, sums_shuf);
    let mut result = _mm_cvtss_f32(result_ss);

    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

/// Calculate vector magnitude (L2 norm) using SIMD
#[inline]
pub fn magnitude_simd(v: &[f32]) -> f32 {
    dot_product_simd(v, v).sqrt()
}

/// Calculate Manhattan (L1) distance using SIMD acceleration
///
/// Manhattan distance is the sum of absolute differences between corresponding
/// elements. It's more robust to outliers than Euclidean distance.
///
/// # Performance
/// - Typically 4-8x faster than naive scalar implementation with SIMD
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Panics
/// Panics if vectors have different lengths
#[inline]
pub fn manhattan_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return manhattan_distance_avx2(a, b);
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx2"),
        target_feature = "sse2"
    ))]
    unsafe {
        return manhattan_distance_sse2(a, b);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        manhattan_distance_neon(a, b)
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse2"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    manhattan_distance_scalar(a, b)
}

/// Optimized scalar fallback for Manhattan distance
#[inline]
#[allow(dead_code)]
fn manhattan_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// AVX2 implementation for Manhattan distance (processes 8 floats at once)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Sign bit mask for abs() - all bits set except sign bit
    let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

    // Process 8 floats at a time
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        // Absolute value using bitwise AND with sign mask
        let abs_diff = _mm256_and_ps(diff, sign_mask);
        sum = _mm256_add_ps(sum, abs_diff);
        i += 8;
    }

    // Horizontal sum of AVX register
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_high, sum_low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remaining elements
    while i < len {
        result += (a[i] - b[i]).abs();
        i += 1;
    }

    result
}

/// SSE2 implementation for Manhattan distance (processes 4 floats at once)
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn manhattan_distance_sse2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Sign bit mask for abs()
    let sign_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));

    // Process 4 floats at a time
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(va, vb);
        let abs_diff = _mm_and_ps(diff, sign_mask);
        sum = _mm_add_ps(sum, abs_diff);
        i += 4;
    }

    // Horizontal sum
    let sum_shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
    let sums = _mm_add_ps(sum, sum_shuf);
    let sums_shuf = _mm_movehl_ps(sums, sums);
    let result_ss = _mm_add_ss(sums, sums_shuf);
    let mut result = _mm_cvtss_f32(result_ss);

    // Handle remaining elements
    while i < len {
        result += (a[i] - b[i]).abs();
        i += 1;
    }

    result
}

/// ARM NEON implementation for Manhattan distance (processes 4 floats at once)
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn manhattan_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        let abs_diff = vabsq_f32(diff);
        sum = vaddq_f32(sum, abs_diff);
        i += 4;
    }

    // Horizontal sum
    let mut result = vgetq_lane_f32(sum, 0)
        + vgetq_lane_f32(sum, 1)
        + vgetq_lane_f32(sum, 2)
        + vgetq_lane_f32(sum, 3);

    // Handle remaining elements
    while i < len {
        result += (a[i] - b[i]).abs();
        i += 1;
    }

    result
}

/// Calculate Hamming distance (count of differing elements)
///
/// For f32 vectors, converts to binary at threshold 0.5 before comparison.
/// Useful for binary embeddings or categorical features.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Number of positions where the vectors differ (as f32)
///
/// # Panics
/// Panics if vectors have different lengths
#[inline]
pub fn hamming_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return hamming_distance_avx2(a, b);
    }

    #[cfg(not(any(all(target_arch = "x86_64", target_feature = "avx2"))))]
    hamming_distance_scalar(a, b)
}

#[inline]
#[allow(dead_code)]
fn hamming_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| {
            let x_bit = **x > 0.5;
            let y_bit = **y > 0.5;
            x_bit != y_bit
        })
        .count() as f32
}

/// AVX2 implementation for Hamming distance
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hamming_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let threshold = _mm256_set1_ps(0.5);
    let mut count = 0;
    let mut i = 0;

    // Process 8 floats at a time
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));

        // Convert to binary (> 0.5)
        let a_bits = _mm256_cmp_ps(va, threshold, _CMP_GT_OQ);
        let b_bits = _mm256_cmp_ps(vb, threshold, _CMP_GT_OQ);

        // XOR to find differences
        let diff = _mm256_xor_ps(a_bits, b_bits);

        // Count set bits using movemask
        let mask = _mm256_movemask_ps(diff);
        count += mask.count_ones();

        i += 8;
    }

    // Handle remaining elements
    while i < len {
        let a_bit = a[i] > 0.5;
        let b_bit = b[i] > 0.5;
        if a_bit != b_bit {
            count += 1;
        }
        i += 1;
    }

    count as f32
}

/// Calculate Jaccard distance (1 - Jaccard similarity)
///
/// Jaccard similarity measures overlap between two sets.
/// For vectors, treats non-zero elements as set membership.
///
/// # Formula
/// ```text
/// Jaccard distance = 1 - (|A ∩ B| / |A ∪ B|)
/// ```
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Jaccard distance in range [0, 1], where 0 means identical sets
///
/// # Panics
/// Panics if vectors have different lengths
#[inline]
pub fn jaccard_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let (intersection, union) = jaccard_counts_simd(a, b);

    if union == 0 {
        return 1.0; // Maximum distance for empty sets
    }

    1.0 - (intersection as f32 / union as f32)
}

/// Calculate Jaccard similarity (for cases where similarity is preferred over distance)
#[inline]
pub fn jaccard_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let (intersection, union) = jaccard_counts_simd(a, b);

    if union == 0 {
        return 0.0; // No similarity for empty sets
    }

    intersection as f32 / union as f32
}

#[inline]
fn jaccard_counts_simd(a: &[f32], b: &[f32]) -> (usize, usize) {
    let mut intersection = 0;
    let mut union = 0;

    for i in 0..a.len() {
        let a_nonzero = a[i] > 0.0;
        let b_nonzero = b[i] > 0.0;

        if a_nonzero || b_nonzero {
            union += 1;
            if a_nonzero && b_nonzero {
                intersection += 1;
            }
        }
    }

    (intersection, union)
}

/// Calculate Chebyshev distance (L∞) - maximum absolute difference
///
/// Chebyshev distance is the maximum absolute difference between corresponding elements.
/// Useful for grid-based movement, game AI, and measuring worst-case deviation.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Maximum absolute difference across all dimensions
///
/// # Panics
/// Panics if vectors have different lengths
#[inline]
pub fn chebyshev_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return chebyshev_distance_avx2(a, b);
    }

    #[cfg(not(any(all(target_arch = "x86_64", target_feature = "avx2"))))]
    chebyshev_distance_scalar(a, b)
}

#[inline]
#[allow(dead_code)]
fn chebyshev_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn chebyshev_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut max_vec = _mm256_setzero_ps();
    let mut i = 0;

    // Sign bit mask for abs()
    let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

    // Process 8 floats at a time
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        let abs_diff = _mm256_and_ps(diff, sign_mask);
        max_vec = _mm256_max_ps(max_vec, abs_diff);
        i += 8;
    }

    // Horizontal max of AVX register
    let max_high = _mm256_extractf128_ps(max_vec, 1);
    let max_low = _mm256_castps256_ps128(max_vec);
    let max128 = _mm_max_ps(max_high, max_low);
    let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 0x55));
    let mut result = _mm_cvtss_f32(max32);

    // Handle remaining elements
    while i < len {
        result = result.max((a[i] - b[i]).abs());
        i += 1;
    }

    result
}

/// Calculate Canberra distance - weighted Manhattan distance
///
/// Canberra distance is a weighted version of Manhattan distance where each dimension
/// is normalized by the sum of the absolute values. Useful for data with large outliers.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Canberra distance
///
/// # Panics
/// Panics if vectors have different lengths
#[inline]
pub fn canberra_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    canberra_distance_scalar(a, b)
}

#[inline]
fn canberra_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let numerator = (x - y).abs();
            let denominator = x.abs() + y.abs();
            if denominator > 1e-10 {
                numerator / denominator
            } else {
                0.0
            }
        })
        .sum()
}

/// Calculate Bray-Curtis dissimilarity - ecological distance
///
/// Bray-Curtis dissimilarity measures the dissimilarity between two samples,
/// commonly used in ecology and environmental science.
///
/// Formula: Σ|a_i - b_i| / Σ(a_i + b_i)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Bray-Curtis dissimilarity in range [0, 1]
///
/// # Panics
/// Panics if vectors have different lengths
#[inline]
pub fn braycurtis_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    braycurtis_distance_scalar(a, b)
}

#[inline]
fn braycurtis_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let numerator: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    let denominator: f32 = a.iter().zip(b.iter()).map(|(x, y)| x.abs() + y.abs()).sum();

    if denominator > 1e-10 {
        numerator / denominator
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_euclidean_distance_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let dist_simd = euclidean_distance_simd(&a, &b);
        let dist_scalar = euclidean_distance_scalar(&a, &b);

        assert_relative_eq!(dist_simd, dist_scalar, epsilon = 1e-6);
        assert_relative_eq!(dist_simd, 8.0_f32.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_similarity_simd() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_relative_eq!(cosine_similarity_simd(&a, &b), 1.0, epsilon = 1e-6);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_relative_eq!(cosine_similarity_simd(&a, &b), 0.0, epsilon = 1e-6);

        let a = vec![1.0, 1.0, 1.0];
        let b = vec![2.0, 2.0, 2.0];
        assert_relative_eq!(cosine_similarity_simd(&a, &b), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dot_product_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = dot_product_simd(&a, &b);
        assert_relative_eq!(result, 70.0, epsilon = 1e-6);
    }

    #[test]
    fn test_magnitude_simd() {
        let v = vec![3.0, 4.0];
        let mag = magnitude_simd(&v);
        assert_relative_eq!(mag, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_large_vectors() {
        let size = 1000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let dist_simd = euclidean_distance_simd(&a, &b);
        let dist_scalar = euclidean_distance_scalar(&a, &b);

        assert_relative_eq!(dist_simd, dist_scalar, epsilon = 1e-4);
    }

    // ========== Manhattan Distance Tests ==========

    #[test]
    fn test_manhattan_distance_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let dist_simd = manhattan_distance_simd(&a, &b);
        let dist_scalar = manhattan_distance_scalar(&a, &b);

        assert_relative_eq!(dist_simd, dist_scalar, epsilon = 1e-6);
        assert_relative_eq!(dist_simd, 8.0, epsilon = 1e-6); // Each element differs by 1
    }

    #[test]
    fn test_manhattan_distance_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let dist = manhattan_distance_simd(&a, &b);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_manhattan_distance_negative() {
        let a = vec![1.0, -2.0, 3.0];
        let b = vec![-1.0, 2.0, -3.0];

        let dist = manhattan_distance_simd(&a, &b);
        // |1-(-1)| + |-2-2| + |3-(-3)| = 2 + 4 + 6 = 12
        assert_relative_eq!(dist, 12.0, epsilon = 1e-6);
    }

    #[test]
    fn test_manhattan_large_vectors() {
        let size = 1000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let dist_simd = manhattan_distance_simd(&a, &b);
        let dist_scalar = manhattan_distance_scalar(&a, &b);

        assert_relative_eq!(dist_simd, dist_scalar, epsilon = 1e-4);
        assert_relative_eq!(dist_simd, 1000.0, epsilon = 1e-4); // Each element differs by 1
    }

    // ========== Hamming Distance Tests ==========

    #[test]
    fn test_hamming_distance_identical() {
        let a = vec![1.0, 1.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 0.0];

        let dist = hamming_distance_simd(&a, &b);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_hamming_distance_different() {
        let a = vec![1.0, 0.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 1.0, 0.0];

        let dist = hamming_distance_simd(&a, &b);
        assert_eq!(dist, 2.0); // Positions 1 and 3 differ
    }

    #[test]
    fn test_hamming_distance_all_different() {
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];

        let dist = hamming_distance_simd(&a, &b);
        assert_eq!(dist, 4.0);
    }

    #[test]
    fn test_hamming_distance_threshold() {
        // Values > 0.5 treated as 1, <= 0.5 treated as 0
        let a = vec![0.6, 0.4, 0.9, 0.1];
        let b = vec![0.7, 0.3, 0.2, 0.8];

        let dist = hamming_distance_simd(&a, &b);
        // a binary: [1, 0, 1, 0]
        // b binary: [1, 0, 0, 1]
        // Differences at positions 2 and 3
        assert_eq!(dist, 2.0);
    }

    #[test]
    fn test_hamming_large_vectors() {
        let size = 1000;
        let a: Vec<f32> = (0..size)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let b: Vec<f32> = (0..size)
            .map(|i| if i % 2 == 1 { 1.0 } else { 0.0 })
            .collect();

        let dist = hamming_distance_simd(&a, &b);
        assert_eq!(dist, 1000.0); // All positions differ
    }

    // ========== Jaccard Distance Tests ==========

    #[test]
    fn test_jaccard_distance_identical() {
        let a = vec![1.0, 1.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 0.0];

        let dist = jaccard_distance_simd(&a, &b);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-6); // Perfect similarity
    }

    #[test]
    fn test_jaccard_distance_disjoint() {
        let a = vec![1.0, 1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 1.0, 1.0];

        let dist = jaccard_distance_simd(&a, &b);
        assert_relative_eq!(dist, 1.0, epsilon = 1e-6); // No overlap
    }

    #[test]
    fn test_jaccard_distance_partial() {
        let a = vec![1.0, 1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 1.0, 0.0];

        let dist = jaccard_distance_simd(&a, &b);
        // Intersection: 1 (position 0)
        // Union: 3 (positions 0, 1, 2)
        // Jaccard similarity: 1/3
        // Jaccard distance: 1 - 1/3 = 2/3
        assert_relative_eq!(dist, 2.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_jaccard_similarity() {
        let a = vec![1.0, 1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 1.0, 0.0];

        let sim = jaccard_similarity_simd(&a, &b);
        // Intersection: 1, Union: 3
        assert_relative_eq!(sim, 1.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_jaccard_empty_sets() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];

        let dist = jaccard_distance_simd(&a, &b);
        assert_eq!(dist, 1.0); // Maximum distance for empty sets
    }

    #[test]
    fn test_jaccard_sparse_vectors() {
        // Simulate sparse vectors (many zeros)
        let mut a = vec![0.0; 100];
        let mut b = vec![0.0; 100];

        a[5] = 1.0;
        a[10] = 1.0;
        a[20] = 1.0;

        b[5] = 1.0;
        b[15] = 1.0;
        b[25] = 1.0;

        let dist = jaccard_distance_simd(&a, &b);
        // Intersection: 1 (position 5)
        // Union: 5 (positions 5, 10, 15, 20, 25)
        // Jaccard distance: 1 - 1/5 = 4/5
        assert_relative_eq!(dist, 4.0 / 5.0, epsilon = 1e-5);
    }

    // ========== Cross-metric Sanity Tests ==========

    #[test]
    fn test_all_metrics_with_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];

        assert_relative_eq!(euclidean_distance_simd(&a, &b), 0.0, epsilon = 1e-6);
        assert_relative_eq!(manhattan_distance_simd(&a, &b), 0.0, epsilon = 1e-6);
        assert_relative_eq!(hamming_distance_simd(&a, &b), 0.0, epsilon = 1e-6);
        assert_relative_eq!(jaccard_distance_simd(&a, &b), 0.0, epsilon = 1e-6);
        assert_relative_eq!(cosine_similarity_simd(&a, &b), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_distance_symmetry() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // All distance metrics should be symmetric
        assert_relative_eq!(
            euclidean_distance_simd(&a, &b),
            euclidean_distance_simd(&b, &a),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            manhattan_distance_simd(&a, &b),
            manhattan_distance_simd(&b, &a),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            hamming_distance_simd(&a, &b),
            hamming_distance_simd(&b, &a),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            jaccard_distance_simd(&a, &b),
            jaccard_distance_simd(&b, &a),
            epsilon = 1e-6
        );
    }
}
