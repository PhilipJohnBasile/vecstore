//! Vector versioning and rollback support
//!
//! Track changes to vectors over time and enable rollback to previous versions.
//! Useful for:
//! - Audit trails and compliance
//! - A/B testing different embeddings
//! - Debugging issues by comparing versions
//! - Experimentation with safe rollback
//!
//! # Features
//!
//! - **Version tracking**: Every update creates a new version
//! - **Rollback**: Restore to any previous version
//! - **Snapshots**: Create named checkpoints
//! - **History**: View all changes to a vector
//! - **Diff**: Compare versions
//!
//! # Example
//!
//! ```rust
//! use vecstore::versioning::VersionedStore;
//!
//! let mut store = VersionedStore::new("vectors.db")?;
//!
//! // Insert creates version 1
//! store.insert("doc1", vec![1.0, 2.0], metadata)?;
//!
//! // Update creates version 2
//! store.update("doc1", vec![1.1, 2.1], metadata)?;
//!
//! // Rollback to version 1
//! store.rollback("doc1", 1)?;
//!
//! // Create snapshot
//! store.create_snapshot("before_experiment")?;
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::store::{Metadata, VecStore};

/// Version information for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    /// Version number (incrementing)
    pub version: u64,
    /// Vector data
    pub vector: Vec<f32>,
    /// Metadata
    pub metadata: Metadata,
    /// Timestamp when created
    pub timestamp: SystemTime,
    /// Optional description of the change
    pub description: Option<String>,
}

/// Version history for a single vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionHistory {
    /// Vector ID
    pub id: String,
    /// All versions (ordered by version number)
    pub versions: Vec<Version>,
    /// Current active version
    pub current_version: u64,
}

impl VersionHistory {
    /// Create new version history
    pub fn new(id: String) -> Self {
        Self {
            id,
            versions: Vec::new(),
            current_version: 0,
        }
    }

    /// Add new version
    pub fn add_version(
        &mut self,
        vector: Vec<f32>,
        metadata: Metadata,
        description: Option<String>,
    ) -> u64 {
        let version_num = self.versions.len() as u64 + 1;

        self.versions.push(Version {
            version: version_num,
            vector,
            metadata,
            timestamp: SystemTime::now(),
            description,
        });

        self.current_version = version_num;
        version_num
    }

    /// Get current version
    pub fn get_current(&self) -> Option<&Version> {
        self.versions
            .iter()
            .find(|v| v.version == self.current_version)
    }

    /// Get specific version
    pub fn get_version(&self, version: u64) -> Option<&Version> {
        self.versions.iter().find(|v| v.version == version)
    }

    /// Rollback to version
    pub fn rollback(&mut self, version: u64) -> Result<()> {
        if self.get_version(version).is_some() {
            self.current_version = version;
            Ok(())
        } else {
            Err(anyhow!("Version {} not found", version))
        }
    }
}

/// Snapshot of the entire store at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Snapshot name/ID
    pub name: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Description
    pub description: Option<String>,
    /// Vector states at snapshot time
    pub states: HashMap<String, (Vec<f32>, Metadata, u64)>, // (vector, metadata, version)
}

/// Versioned vector store with rollback support
pub struct VersionedStore {
    /// Underlying vector store
    store: VecStore,
    /// Version history for each vector
    history: HashMap<String, VersionHistory>,
    /// Named snapshots
    snapshots: HashMap<String, Snapshot>,
    /// Storage path
    path: PathBuf,
}

impl VersionedStore {
    /// Create or open versioned store
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let store = VecStore::open(&path_buf)?;

        // Try to load existing history
        let history_path = path_buf.with_extension("history");
        let history = if history_path.exists() {
            let data = std::fs::read_to_string(&history_path)?;
            serde_json::from_str(&data)?
        } else {
            HashMap::new()
        };

        // Try to load snapshots
        let snapshots_path = path_buf.with_extension("snapshots");
        let snapshots = if snapshots_path.exists() {
            let data = std::fs::read_to_string(&snapshots_path)?;
            serde_json::from_str(&data)?
        } else {
            HashMap::new()
        };

        Ok(Self {
            store,
            history,
            snapshots,
            path: path_buf,
        })
    }

    /// Insert new vector (creates version 1)
    pub fn insert(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Metadata,
    ) -> Result<u64> {
        let id = id.into();

        // Add to store
        self.store
            .upsert(id.clone(), vector.clone(), metadata.clone())?;

        // Create version history
        let mut history = VersionHistory::new(id.clone());
        let version = history.add_version(vector, metadata, Some("Initial version".to_string()));

        self.history.insert(id, history);
        self.save_history()?;

        Ok(version)
    }

    /// Update vector (creates new version)
    pub fn update(
        &mut self,
        id: &str,
        vector: Vec<f32>,
        metadata: Metadata,
        description: Option<String>,
    ) -> Result<u64> {
        let history = self
            .history
            .get_mut(id)
            .ok_or_else(|| anyhow!("Vector {} not found", id))?;

        // Update store
        self.store
            .upsert(id.to_string(), vector.clone(), metadata.clone())?;

        // Add version
        let version = history.add_version(vector, metadata, description);
        self.save_history()?;

        Ok(version)
    }

    /// Rollback vector to specific version
    pub fn rollback(&mut self, id: &str, version: u64) -> Result<()> {
        let history = self
            .history
            .get_mut(id)
            .ok_or_else(|| anyhow!("Vector {} not found", id))?;

        // Get target version
        let target = history
            .get_version(version)
            .ok_or_else(|| anyhow!("Version {} not found", version))?;

        // Update store to that version
        self.store.upsert(
            id.to_string(),
            target.vector.clone(),
            target.metadata.clone(),
        )?;

        // Update current version in history
        history.rollback(version)?;
        self.save_history()?;

        Ok(())
    }

    /// Get version history for a vector
    pub fn get_history(&self, id: &str) -> Option<&VersionHistory> {
        self.history.get(id)
    }

    /// Get current version of a vector
    pub fn get_current_version(&self, id: &str) -> Option<&Version> {
        self.history.get(id).and_then(|h| h.get_current())
    }

    /// Create named snapshot of entire store
    pub fn create_snapshot(
        &mut self,
        name: impl Into<String>,
        description: Option<String>,
    ) -> Result<()> {
        let name = name.into();

        // Capture current state of all vectors
        let mut states = HashMap::new();
        for (id, history) in &self.history {
            if let Some(current) = history.get_current() {
                states.insert(
                    id.clone(),
                    (
                        current.vector.clone(),
                        current.metadata.clone(),
                        current.version,
                    ),
                );
            }
        }

        let snapshot = Snapshot {
            name: name.clone(),
            timestamp: SystemTime::now(),
            description,
            states,
        };

        self.snapshots.insert(name, snapshot);
        self.save_snapshots()?;

        Ok(())
    }

    /// Restore entire store to a snapshot
    pub fn restore_snapshot(&mut self, name: &str) -> Result<()> {
        let snapshot = self
            .snapshots
            .get(name)
            .ok_or_else(|| anyhow!("Snapshot {} not found", name))?
            .clone();

        // Restore each vector to its snapshot state
        for (id, (vector, metadata, version)) in snapshot.states {
            // Update store
            self.store
                .upsert(id.clone(), vector.clone(), metadata.clone())?;

            // Update history to reflect rollback
            if let Some(history) = self.history.get_mut(&id) {
                history.rollback(version)?;
            }
        }

        self.save_history()?;

        Ok(())
    }

    /// List all snapshots
    pub fn list_snapshots(&self) -> Vec<&Snapshot> {
        self.snapshots.values().collect()
    }

    /// Delete a snapshot
    pub fn delete_snapshot(&mut self, name: &str) -> Result<()> {
        self.snapshots
            .remove(name)
            .ok_or_else(|| anyhow!("Snapshot {} not found", name))?;
        self.save_snapshots()?;
        Ok(())
    }

    /// Compare two versions of a vector
    pub fn compare_versions(&self, id: &str, v1: u64, v2: u64) -> Result<VersionDiff> {
        let history = self
            .history
            .get(id)
            .ok_or_else(|| anyhow!("Vector {} not found", id))?;

        let version1 = history
            .get_version(v1)
            .ok_or_else(|| anyhow!("Version {} not found", v1))?;
        let version2 = history
            .get_version(v2)
            .ok_or_else(|| anyhow!("Version {} not found", v2))?;

        // Compute differences
        let vector_changed = version1.vector != version2.vector;
        let metadata_changed = serde_json::to_string(&version1.metadata)?
            != serde_json::to_string(&version2.metadata)?;

        let vector_distance = if vector_changed {
            let dist: f32 = version1
                .vector
                .iter()
                .zip(&version2.vector)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            Some(dist)
        } else {
            None
        };

        Ok(VersionDiff {
            id: id.to_string(),
            version1: v1,
            version2: v2,
            vector_changed,
            metadata_changed,
            vector_distance,
        })
    }

    /// Get underlying store (for queries)
    pub fn store(&self) -> &VecStore {
        &self.store
    }

    /// Get mutable store reference
    pub fn store_mut(&mut self) -> &mut VecStore {
        &mut self.store
    }

    /// Save history to disk
    fn save_history(&self) -> Result<()> {
        let path = self.path.with_extension("history");
        let data = serde_json::to_string(&self.history)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Save snapshots to disk
    fn save_snapshots(&self) -> Result<()> {
        let path = self.path.with_extension("snapshots");
        let data = serde_json::to_string(&self.snapshots)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Get total number of versions across all vectors
    pub fn total_versions(&self) -> usize {
        self.history.values().map(|h| h.versions.len()).sum()
    }

    /// Get statistics
    pub fn stats(&self) -> VersioningStats {
        let total_vectors = self.history.len();
        let total_versions = self.total_versions();
        let total_snapshots = self.snapshots.len();

        let avg_versions_per_vector = if total_vectors > 0 {
            total_versions as f32 / total_vectors as f32
        } else {
            0.0
        };

        VersioningStats {
            total_vectors,
            total_versions,
            total_snapshots,
            avg_versions_per_vector,
        }
    }
}

/// Difference between two versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// Vector ID
    pub id: String,
    /// First version
    pub version1: u64,
    /// Second version
    pub version2: u64,
    /// Whether vector changed
    pub vector_changed: bool,
    /// Whether metadata changed
    pub metadata_changed: bool,
    /// Euclidean distance if vector changed
    pub vector_distance: Option<f32>,
}

/// Versioning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningStats {
    /// Total vectors being tracked
    pub total_vectors: usize,
    /// Total versions across all vectors
    pub total_versions: usize,
    /// Number of snapshots
    pub total_snapshots: usize,
    /// Average versions per vector
    pub avg_versions_per_vector: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn create_metadata(value: &str) -> Metadata {
        let mut fields = HashMap::new();
        fields.insert("value".to_string(), serde_json::json!(value));
        Metadata { fields }
    }

    #[test]
    fn test_basic_versioning() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut store = VersionedStore::new(temp_dir.path().join("test.db"))?;

        // Insert creates version 1
        let v1 = store.insert("doc1", vec![1.0, 2.0], create_metadata("v1"))?;
        assert_eq!(v1, 1);

        // Update creates version 2
        let v2 = store.update(
            "doc1",
            vec![1.1, 2.1],
            create_metadata("v2"),
            Some("Updated".to_string()),
        )?;
        assert_eq!(v2, 2);

        // Check history
        let history = store.get_history("doc1").unwrap();
        assert_eq!(history.versions.len(), 2);
        assert_eq!(history.current_version, 2);

        Ok(())
    }

    #[test]
    fn test_rollback() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut store = VersionedStore::new(temp_dir.path().join("test.db"))?;

        store.insert("doc1", vec![1.0, 2.0], create_metadata("v1"))?;
        store.update("doc1", vec![2.0, 3.0], create_metadata("v2"), None)?;
        store.update("doc1", vec![3.0, 4.0], create_metadata("v3"), None)?;

        // Rollback to version 1
        store.rollback("doc1", 1)?;

        let current = store.get_current_version("doc1").unwrap();
        assert_eq!(current.version, 1);
        assert_eq!(current.vector, vec![1.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_snapshots() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut store = VersionedStore::new(temp_dir.path().join("test.db"))?;

        store.insert("doc1", vec![1.0, 2.0], create_metadata("v1"))?;
        store.insert("doc2", vec![3.0, 4.0], create_metadata("v1"))?;

        // Create snapshot
        store.create_snapshot("checkpoint1", Some("Before changes".to_string()))?;

        // Make changes
        store.update("doc1", vec![5.0, 6.0], create_metadata("v2"), None)?;
        store.update("doc2", vec![7.0, 8.0], create_metadata("v2"), None)?;

        // Restore snapshot
        store.restore_snapshot("checkpoint1")?;

        let doc1 = store.get_current_version("doc1").unwrap();
        assert_eq!(doc1.vector, vec![1.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_compare_versions() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut store = VersionedStore::new(temp_dir.path().join("test.db"))?;

        store.insert("doc1", vec![1.0, 0.0], create_metadata("v1"))?;
        store.update("doc1", vec![0.0, 1.0], create_metadata("v2"), None)?;

        let diff = store.compare_versions("doc1", 1, 2)?;

        assert!(diff.vector_changed);
        assert!(diff.vector_distance.unwrap() > 0.0);

        Ok(())
    }

    #[test]
    fn test_persistence() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db_path = temp_dir.path().join("test.db");

        // Create store and add versions
        {
            let mut store = VersionedStore::new(&db_path)?;
            store.insert("doc1", vec![1.0, 2.0], create_metadata("v1"))?;
            store.update("doc1", vec![2.0, 3.0], create_metadata("v2"), None)?;
            store.create_snapshot("snap1", None)?;
        }

        // Reopen and verify history persisted
        {
            let store = VersionedStore::new(&db_path)?;
            let history = store.get_history("doc1").unwrap();
            assert_eq!(history.versions.len(), 2);
            assert_eq!(store.list_snapshots().len(), 1);
        }

        Ok(())
    }
}
