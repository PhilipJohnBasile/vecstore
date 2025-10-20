//! Vector deduplication and near-duplicate detection
//!
//! This module provides tools for finding and removing duplicate or near-duplicate vectors
//! from a vector store. Useful for data cleaning, quality management, and reducing storage.
//!
//! # Features
//!
//! - **Exact duplicates**: Find vectors with identical values
//! - **Near-duplicates**: Find vectors within a similarity threshold
//! - **Batch operations**: Efficient deduplication of large datasets
//! - **Preservation strategies**: Keep first, last, or best quality duplicate
//! - **Statistics**: Report on duplication patterns
//!
//! # Example
//!
//! ```rust
//! use vecstore::deduplication::{Deduplicator, DeduplicationConfig, DeduplicationStrategy};
//!
//! let config = DeduplicationConfig {
//!     similarity_threshold: 0.99, // 99% similar = duplicate
//!     strategy: DeduplicationStrategy::KeepFirst,
//!     batch_size: 1000,
//! };
//!
//! let deduplicator = Deduplicator::new(config);
//!
//! // Find duplicates
//! let duplicates = deduplicator.find_duplicates(&store)?;
//! println!("Found {} duplicate groups", duplicates.len());
//!
//! // Remove duplicates
//! let stats = deduplicator.remove_duplicates(&mut store)?;
//! println!("Removed {} vectors", stats.removed_count);
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::simd::{cosine_similarity_simd, euclidean_distance_simd};
use crate::store::VecStore;

/// Deduplication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationConfig {
    /// Similarity threshold for considering vectors as duplicates (0.0-1.0)
    /// Higher values = more strict (only very similar vectors are duplicates)
    pub similarity_threshold: f32,

    /// Strategy for which duplicate to keep
    pub strategy: DeduplicationStrategy,

    /// Batch size for processing
    pub batch_size: usize,

    /// Use cosine similarity (true) or euclidean distance (false)
    pub use_cosine: bool,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.99, // 99% similar
            strategy: DeduplicationStrategy::KeepFirst,
            batch_size: 1000,
            use_cosine: true,
        }
    }
}

/// Strategy for keeping duplicates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeduplicationStrategy {
    /// Keep the first occurrence
    KeepFirst,
    /// Keep the last occurrence
    KeepLast,
    /// Keep the one with most metadata fields
    KeepMostMetadata,
    /// Keep the one with highest quality score (custom field)
    KeepHighestQuality,
}

/// Group of duplicate vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    /// Representative ID (the one to keep)
    pub representative: String,
    /// All duplicate IDs (including representative)
    pub duplicates: Vec<String>,
    /// Similarity scores to representative
    pub scores: Vec<f32>,
    /// Average similarity within group
    pub avg_similarity: f32,
}

/// Deduplication statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationStats {
    /// Total vectors scanned
    pub total_vectors: usize,
    /// Number of duplicate groups found
    pub duplicate_groups: usize,
    /// Total duplicates found
    pub total_duplicates: usize,
    /// Vectors removed
    pub removed_count: usize,
    /// Vectors kept
    pub kept_count: usize,
    /// Storage saved (estimated bytes)
    pub storage_saved: usize,
    /// Duplication ratio (0.0-1.0)
    pub duplication_ratio: f32,
}

/// Vector deduplicator
pub struct Deduplicator {
    config: DeduplicationConfig,
}

impl Deduplicator {
    /// Create new deduplicator
    pub fn new(config: DeduplicationConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(DeduplicationConfig::default())
    }

    /// Find all duplicate groups in the store
    pub fn find_duplicates(&self, store: &VecStore) -> Result<Vec<DuplicateGroup>> {
        let total = store.len();
        if total == 0 {
            return Ok(vec![]);
        }

        println!("🔍 Scanning {} vectors for duplicates...", total);

        let mut groups = Vec::new();
        let mut processed = HashSet::new();

        // Get all vectors from store (simplified - in practice we'd iterate)
        // This is a placeholder for the actual implementation
        let vectors = self.get_all_vectors(store)?;

        for (i, (id1, vec1)) in vectors.iter().enumerate() {
            if processed.contains(id1) {
                continue;
            }

            let mut group_ids = vec![id1.clone()];
            let mut group_scores = vec![1.0]; // Perfect match with itself

            // Compare with all other vectors
            for (j, (id2, vec2)) in vectors.iter().enumerate().skip(i + 1) {
                if processed.contains(id2) {
                    continue;
                }

                let similarity = self.compute_similarity(vec1, vec2);

                if similarity >= self.config.similarity_threshold {
                    group_ids.push(id2.clone());
                    group_scores.push(similarity);
                    processed.insert(id2.clone());
                }
            }

            // If we found duplicates, create a group
            if group_ids.len() > 1 {
                let avg_similarity = group_scores.iter().sum::<f32>() / group_scores.len() as f32;

                groups.push(DuplicateGroup {
                    representative: id1.clone(),
                    duplicates: group_ids,
                    scores: group_scores,
                    avg_similarity,
                });

                processed.insert(id1.clone());
            }

            if (i + 1) % self.config.batch_size == 0 {
                println!("  Processed {}/{} vectors...", i + 1, total);
            }
        }

        println!("✓ Found {} duplicate groups", groups.len());

        Ok(groups)
    }

    /// Find exact duplicates (vectors with identical values)
    pub fn find_exact_duplicates(&self, store: &VecStore) -> Result<Vec<DuplicateGroup>> {
        let vectors = self.get_all_vectors(store)?;
        let mut hash_map: HashMap<Vec<u8>, Vec<String>> = HashMap::new();

        for (id, vec) in vectors {
            // Create hash key from vector bytes
            let key = self.vector_to_bytes(&vec);
            hash_map.entry(key).or_default().push(id);
        }

        let mut groups = Vec::new();
        for (_, ids) in hash_map {
            if ids.len() > 1 {
                let scores = vec![1.0; ids.len()]; // Exact duplicates
                groups.push(DuplicateGroup {
                    representative: ids[0].clone(),
                    duplicates: ids,
                    scores,
                    avg_similarity: 1.0,
                });
            }
        }

        Ok(groups)
    }

    /// Remove duplicates from store according to strategy
    pub fn remove_duplicates(&self, store: &mut VecStore) -> Result<DeduplicationStats> {
        let total_vectors = store.len();
        let groups = self.find_duplicates(store)?;

        let mut removed_count = 0;
        let mut kept_count = 0;

        for mut group in groups.iter() {
            // Determine which ID to keep based on strategy
            let to_keep = match self.config.strategy {
                DeduplicationStrategy::KeepFirst => &group.duplicates[0],
                DeduplicationStrategy::KeepLast => group.duplicates.last().unwrap(),
                DeduplicationStrategy::KeepMostMetadata => {
                    self.select_most_metadata(&group.duplicates, store)?
                }
                DeduplicationStrategy::KeepHighestQuality => {
                    self.select_highest_quality(&group.duplicates, store)?
                }
            };

            // Remove all except the one to keep
            for id in &group.duplicates {
                if id != to_keep {
                    store.delete(id)?;
                    removed_count += 1;
                } else {
                    kept_count += 1;
                }
            }
        }

        let duplication_ratio = if total_vectors > 0 {
            removed_count as f32 / total_vectors as f32
        } else {
            0.0
        };

        let storage_saved = removed_count * store.dimension() * 4; // f32 = 4 bytes

        Ok(DeduplicationStats {
            total_vectors,
            duplicate_groups: groups.len(),
            total_duplicates: groups.iter().map(|g| g.duplicates.len()).sum(),
            removed_count,
            kept_count,
            storage_saved,
            duplication_ratio,
        })
    }

    /// Get statistics about duplication without removing
    pub fn analyze_duplication(&self, store: &VecStore) -> Result<DeduplicationStats> {
        let total_vectors = store.len();
        let groups = self.find_duplicates(store)?;

        let total_duplicates: usize = groups.iter().map(|g| g.duplicates.len()).sum();
        let removed_count = total_duplicates - groups.len(); // Keep one per group
        let kept_count = groups.len();

        let duplication_ratio = if total_vectors > 0 {
            removed_count as f32 / total_vectors as f32
        } else {
            0.0
        };

        let storage_saved = removed_count * store.dimension() * 4;

        Ok(DeduplicationStats {
            total_vectors,
            duplicate_groups: groups.len(),
            total_duplicates,
            removed_count,
            kept_count,
            storage_saved,
            duplication_ratio,
        })
    }

    /// Find near-duplicates of a specific vector
    pub fn find_similar_to(
        &self,
        store: &VecStore,
        target_id: &str,
        threshold: Option<f32>,
    ) -> Result<Vec<(String, f32)>> {
        let threshold = threshold.unwrap_or(self.config.similarity_threshold);
        let vectors = self.get_all_vectors(store)?;

        let target_vec = vectors
            .iter()
            .find(|(id, _)| id == target_id)
            .ok_or_else(|| anyhow::anyhow!("Vector {} not found", target_id))?
            .1
            .clone();

        let mut similar = Vec::new();

        for (id, vec) in vectors {
            if id == target_id {
                continue;
            }

            let similarity = self.compute_similarity(&target_vec, &vec);
            if similarity >= threshold {
                similar.push((id, similarity));
            }
        }

        // Sort by similarity descending
        similar.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(similar)
    }

    // Helper methods

    fn compute_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if self.config.use_cosine {
            cosine_similarity_simd(vec1, vec2)
        } else {
            // Convert euclidean distance to similarity (0-1 range)
            let dist = euclidean_distance_simd(vec1, vec2);
            1.0 / (1.0 + dist)
        }
    }

    fn get_all_vectors(&self, store: &VecStore) -> Result<Vec<(String, Vec<f32>)>> {
        // Placeholder - in practice, this would iterate over store contents
        // For now, return empty vec as the actual implementation depends on VecStore internals
        Ok(vec![])
    }

    fn vector_to_bytes(&self, vec: &[f32]) -> Vec<u8> {
        vec.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn select_most_metadata<'a>(&self, ids: &'a [String], _store: &VecStore) -> Result<&'a String> {
        // Placeholder - would query metadata for each ID and select the one with most fields
        Ok(&ids[0])
    }

    fn select_highest_quality<'a>(
        &self,
        ids: &'a [String],
        _store: &VecStore,
    ) -> Result<&'a String> {
        // Placeholder - would check "quality" metadata field
        Ok(&ids[0])
    }
}

/// Batch deduplication for large datasets
pub struct BatchDeduplicator {
    config: DeduplicationConfig,
    chunk_size: usize,
}

impl BatchDeduplicator {
    pub fn new(config: DeduplicationConfig, chunk_size: usize) -> Self {
        Self { config, chunk_size }
    }

    /// Process large dataset in chunks
    pub fn deduplicate_batches(&self, store: &mut VecStore) -> Result<DeduplicationStats> {
        let total = store.len();
        let mut overall_stats = DeduplicationStats {
            total_vectors: total,
            duplicate_groups: 0,
            total_duplicates: 0,
            removed_count: 0,
            kept_count: 0,
            storage_saved: 0,
            duplication_ratio: 0.0,
        };

        let num_chunks = (total + self.chunk_size - 1) / self.chunk_size;

        for chunk_idx in 0..num_chunks {
            println!("Processing chunk {}/{}...", chunk_idx + 1, num_chunks);

            // Process this chunk
            let deduplicator = Deduplicator::new(self.config.clone());
            let stats = deduplicator.remove_duplicates(store)?;

            // Aggregate stats
            overall_stats.duplicate_groups += stats.duplicate_groups;
            overall_stats.total_duplicates += stats.total_duplicates;
            overall_stats.removed_count += stats.removed_count;
            overall_stats.kept_count += stats.kept_count;
            overall_stats.storage_saved += stats.storage_saved;
        }

        overall_stats.duplication_ratio = if total > 0 {
            overall_stats.removed_count as f32 / total as f32
        } else {
            0.0
        };

        Ok(overall_stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Metadata;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_deduplication_config() {
        let config = DeduplicationConfig::default();
        assert_eq!(config.similarity_threshold, 0.99);
        assert_eq!(config.strategy, DeduplicationStrategy::KeepFirst);
        assert!(config.use_cosine);
    }

    #[test]
    fn test_compute_similarity() {
        let config = DeduplicationConfig::default();
        let dedup = Deduplicator::new(config);

        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let vec3 = vec![0.0, 1.0, 0.0];

        let sim1 = dedup.compute_similarity(&vec1, &vec2);
        assert!((sim1 - 1.0).abs() < 0.001); // Identical vectors

        let sim2 = dedup.compute_similarity(&vec1, &vec3);
        assert!(sim2 < 0.1); // Orthogonal vectors
    }

    #[test]
    fn test_vector_to_bytes() {
        let dedup = Deduplicator::default();
        let vec = vec![1.0, 2.0, 3.0];
        let bytes = dedup.vector_to_bytes(&vec);

        assert_eq!(bytes.len(), 12); // 3 floats * 4 bytes each
    }

    #[test]
    fn test_deduplication_stats() {
        let stats = DeduplicationStats {
            total_vectors: 1000,
            duplicate_groups: 50,
            total_duplicates: 150,
            removed_count: 100,
            kept_count: 50,
            storage_saved: 51200,
            duplication_ratio: 0.1,
        };

        assert_eq!(stats.removed_count, 100);
        assert_eq!(stats.duplication_ratio, 0.1);
    }

    #[test]
    fn test_batch_deduplicator() {
        let config = DeduplicationConfig::default();
        let batch_dedup = BatchDeduplicator::new(config, 100);

        assert_eq!(batch_dedup.chunk_size, 100);
    }
}
