//! Zero-Copy Versioning
//!
//! Git-like versioning for vector data with copy-on-write semantics.
//! Similar to LanceDB's Lance format with efficient branching.
//!
//! # Features
//!
//! - **Copy-on-Write**: Only changed data is duplicated
//! - **Branches**: Create lightweight branches for experiments
//! - **Commits**: Atomic commits with messages
//! - **Diff**: Compare versions efficiently
//! - **Merge**: Merge branches with conflict resolution
//!
//! # Example
//!
//! ```rust,ignore
//! use vecstore::zero_copy::{VersionedStore, Commit};
//!
//! let mut store = VersionedStore::new(384)?;
//!
//! // Insert data
//! store.upsert("doc1", vec, metadata)?;
//! let commit1 = store.commit("Initial data")?;
//!
//! // Create branch
//! store.branch("experiment")?;
//! store.checkout("experiment")?;
//!
//! // Make changes on branch
//! store.upsert("doc2", vec, metadata)?;
//! store.commit("Add doc2")?;
//!
//! // Merge back to main
//! store.checkout("main")?;
//! store.merge("experiment")?;
//! ```

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

use crate::error::{VecStoreError, Result};

// ============================================================================
// TYPES
// ============================================================================

/// Commit hash (simplified)
pub type CommitHash = String;

/// Branch name
pub type BranchName = String;

/// Vector data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorData {
    vector: Vec<f32>,
    metadata: Option<serde_json::Value>,
    created_at: u64,
    modified_at: u64,
}

/// Commit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    pub hash: CommitHash,
    pub parent: Option<CommitHash>,
    pub message: String,
    pub timestamp: u64,
    pub author: Option<String>,
    /// IDs changed in this commit
    pub changed_ids: Vec<String>,
    /// IDs deleted in this commit
    pub deleted_ids: Vec<String>,
}

/// Branch information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub name: BranchName,
    pub head: CommitHash,
    pub created_at: u64,
    pub protected: bool,
}

/// Diff between two commits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diff {
    pub from_commit: CommitHash,
    pub to_commit: CommitHash,
    pub added: Vec<String>,
    pub modified: Vec<String>,
    pub deleted: Vec<String>,
    pub stats: DiffStats,
}

/// Diff statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffStats {
    pub total_added: usize,
    pub total_modified: usize,
    pub total_deleted: usize,
    pub vectors_changed: usize,
}

/// Merge result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    pub success: bool,
    pub commit: Option<Commit>,
    pub conflicts: Vec<MergeConflict>,
}

/// Merge conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConflict {
    pub id: String,
    pub conflict_type: ConflictType,
    pub ours: Option<Vec<f32>>,
    pub theirs: Option<Vec<f32>>,
}

/// Type of merge conflict
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    /// Both modified the same vector
    ModifyModify,
    /// We modified, they deleted
    ModifyDelete,
    /// We deleted, they modified
    DeleteModify,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Keep our version
    Ours,
    /// Keep their version
    Theirs,
    /// Average the vectors
    Average,
    /// Abort on conflict
    Abort,
}

// ============================================================================
// STORAGE LAYER (Copy-on-Write)
// ============================================================================

/// Copy-on-write storage for vectors
struct CowStorage {
    /// Base data (immutable after commit)
    base: HashMap<String, Arc<VectorData>>,
    /// Working changes (mutable)
    working: HashMap<String, VectorData>,
    /// Deleted IDs in working set
    deleted: HashSet<String>,
}

impl CowStorage {
    fn new() -> Self {
        Self {
            base: HashMap::new(),
            working: HashMap::new(),
            deleted: HashSet::new(),
        }
    }

    /// Get a vector (checks working, then base)
    fn get(&self, id: &str) -> Option<&VectorData> {
        if self.deleted.contains(id) {
            return None;
        }
        self.working.get(id).or_else(|| self.base.get(id).map(|arc| arc.as_ref()))
    }

    /// Set a vector (copy-on-write)
    fn set(&mut self, id: &str, data: VectorData) {
        self.deleted.remove(id);
        self.working.insert(id.to_string(), data);
    }

    /// Delete a vector
    fn delete(&mut self, id: &str) {
        self.working.remove(id);
        if self.base.contains_key(id) {
            self.deleted.insert(id.to_string());
        }
    }

    /// Commit working changes to base
    fn commit(&mut self) -> (Vec<String>, Vec<String>) {
        let changed: Vec<String> = self.working.keys().cloned().collect();
        let deleted: Vec<String> = self.deleted.iter().cloned().collect();

        // Move working to base
        for (id, data) in self.working.drain() {
            self.base.insert(id, Arc::new(data));
        }

        // Remove deleted from base
        for id in &self.deleted {
            self.base.remove(id);
        }
        self.deleted.clear();

        (changed, deleted)
    }

    /// Reset working changes
    fn reset(&mut self) {
        self.working.clear();
        self.deleted.clear();
    }

    /// Clone storage (for branching)
    fn fork(&self) -> Self {
        // Copy-on-write: base is shared via Arc
        Self {
            base: self.base.clone(),
            working: HashMap::new(),
            deleted: HashSet::new(),
        }
    }

    /// Get all IDs
    fn all_ids(&self) -> Vec<String> {
        let mut ids: HashSet<_> = self.base.keys().cloned().collect();
        ids.extend(self.working.keys().cloned());
        for id in &self.deleted {
            ids.remove(id);
        }
        ids.into_iter().collect()
    }

    /// Check if has uncommitted changes
    fn is_dirty(&self) -> bool {
        !self.working.is_empty() || !self.deleted.is_empty()
    }
}

// ============================================================================
// VERSIONED STORE
// ============================================================================

/// Versioned vector store with Git-like operations
pub struct VersionedStore {
    /// Vector dimension
    dimension: usize,
    /// Current branch name
    current_branch: RwLock<BranchName>,
    /// Storage per branch
    storages: RwLock<HashMap<BranchName, CowStorage>>,
    /// All commits
    commits: RwLock<BTreeMap<CommitHash, Commit>>,
    /// Branch heads
    branches: RwLock<HashMap<BranchName, Branch>>,
    /// Commit counter for hash generation
    commit_counter: RwLock<u64>,
}

impl VersionedStore {
    pub fn new(dimension: usize) -> Result<Self> {
        let mut branches = HashMap::new();
        let mut storages = HashMap::new();

        // Initialize main branch
        let main_branch = Branch {
            name: "main".to_string(),
            head: "initial".to_string(),
            created_at: unix_timestamp(),
            protected: false,
        };
        branches.insert("main".to_string(), main_branch);
        storages.insert("main".to_string(), CowStorage::new());

        // Create initial commit
        let mut commits = BTreeMap::new();
        commits.insert("initial".to_string(), Commit {
            hash: "initial".to_string(),
            parent: None,
            message: "Initial commit".to_string(),
            timestamp: unix_timestamp(),
            author: None,
            changed_ids: Vec::new(),
            deleted_ids: Vec::new(),
        });

        Ok(Self {
            dimension,
            current_branch: RwLock::new("main".to_string()),
            storages: RwLock::new(storages),
            commits: RwLock::new(commits),
            branches: RwLock::new(branches),
            commit_counter: RwLock::new(1),
        })
    }

    /// Get current branch name
    pub fn current_branch(&self) -> String {
        self.current_branch.read().map(|b| b.clone()).unwrap_or_else(|_| "main".to_string())
    }

    /// Insert or update a vector
    pub fn upsert(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(VecStoreError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let branch = self.current_branch.read()?.clone();
        let mut storages = self.storages.write()?;
        let storage = storages.get_mut(&branch)
            .ok_or_else(|| VecStoreError::NotFound(format!("Branch: {}", branch)))?;

        let now = unix_timestamp();
        let data = VectorData {
            vector,
            metadata,
            created_at: storage.get(id).map(|d| d.created_at).unwrap_or(now),
            modified_at: now,
        };

        storage.set(id, data);
        Ok(())
    }

    /// Delete a vector
    pub fn delete(&self, id: &str) -> Result<bool> {
        let branch = self.current_branch.read()?.clone();
        let mut storages = self.storages.write()?;
        let storage = storages.get_mut(&branch)
            .ok_or_else(|| VecStoreError::NotFound(format!("Branch: {}", branch)))?;

        let existed = storage.get(id).is_some();
        storage.delete(id);
        Ok(existed)
    }

    /// Get a vector
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<serde_json::Value>)> {
        let branch = self.current_branch.read().ok()?.clone();
        let storages = self.storages.read().ok()?;
        let storage = storages.get(&branch)?;

        storage.get(id).map(|d| (d.vector.clone(), d.metadata.clone()))
    }

    /// Create a commit
    pub fn commit(&self, message: &str) -> Result<Commit> {
        let branch_name = self.current_branch.read()?.clone();

        // Generate commit hash
        let hash = {
            let mut counter = self.commit_counter.write()?;
            *counter += 1;
            format!("{:08x}", *counter)
        };

        // Commit storage changes
        let (changed, deleted) = {
            let mut storages = self.storages.write()?;
            let storage = storages.get_mut(&branch_name)
                .ok_or_else(|| VecStoreError::NotFound(format!("Branch: {}", branch_name)))?;
            storage.commit()
        };

        if changed.is_empty() && deleted.is_empty() {
            return Err(VecStoreError::InvalidInput("Nothing to commit".to_string()));
        }

        // Get parent commit
        let parent = {
            let branches = self.branches.read()?;
            branches.get(&branch_name).map(|b| b.head.clone())
        };

        // Create commit
        let commit = Commit {
            hash: hash.clone(),
            parent,
            message: message.to_string(),
            timestamp: unix_timestamp(),
            author: None,
            changed_ids: changed,
            deleted_ids: deleted,
        };

        // Store commit and update branch head
        {
            let mut commits = self.commits.write()?;
            commits.insert(hash.clone(), commit.clone());
        }

        {
            let mut branches = self.branches.write()?;
            if let Some(branch) = branches.get_mut(&branch_name) {
                branch.head = hash;
            }
        }

        Ok(commit)
    }

    /// Create a new branch
    pub fn branch(&self, name: &str) -> Result<Branch> {
        let current = self.current_branch.read()?.clone();

        // Check if branch exists
        {
            let branches = self.branches.read()?;
            if branches.contains_key(name) {
                return Err(VecStoreError::InvalidInput(format!(
                    "Branch {} already exists",
                    name
                )));
            }
        }

        // Get current branch head
        let head = {
            let branches = self.branches.read()?;
            branches.get(&current)
                .map(|b| b.head.clone())
                .unwrap_or_else(|| "initial".to_string())
        };

        // Fork storage
        let new_storage = {
            let storages = self.storages.read()?;
            storages.get(&current)
                .ok_or_else(|| VecStoreError::NotFound(format!("Branch: {}", current)))?
                .fork()
        };

        // Create branch
        let branch = Branch {
            name: name.to_string(),
            head,
            created_at: unix_timestamp(),
            protected: false,
        };

        // Store
        {
            let mut branches = self.branches.write()?;
            branches.insert(name.to_string(), branch.clone());
        }
        {
            let mut storages = self.storages.write()?;
            storages.insert(name.to_string(), new_storage);
        }

        Ok(branch)
    }

    /// Checkout a branch
    pub fn checkout(&self, name: &str) -> Result<()> {
        // Check if branch exists
        {
            let branches = self.branches.read()?;
            if !branches.contains_key(name) {
                return Err(VecStoreError::NotFound(format!("Branch: {}", name)));
            }
        }

        // Check for uncommitted changes
        {
            let current = self.current_branch.read()?.clone();
            let storages = self.storages.read()?;
            if storages.get(&current)
                .ok_or_else(|| VecStoreError::NotFound(format!("Branch: {}", current)))?
                .is_dirty()
            {
                return Err(VecStoreError::InvalidInput(
                    "Uncommitted changes - commit or reset first".to_string()
                ));
            }
        }

        *self.current_branch.write()? = name.to_string();
        Ok(())
    }

    /// Reset uncommitted changes
    pub fn reset(&self) {
        if let Ok(branch) = self.current_branch.read().map(|b| b.clone()) {
            if let Ok(mut storages) = self.storages.write() {
                if let Some(storage) = storages.get_mut(&branch) {
                    storage.reset();
                }
            }
        }
    }

    /// Delete a branch
    pub fn delete_branch(&self, name: &str) -> Result<bool> {
        if name == "main" {
            return Err(VecStoreError::InvalidInput("Cannot delete main branch".to_string()));
        }

        let current = self.current_branch.read()?.clone();
        if current == name {
            return Err(VecStoreError::InvalidInput("Cannot delete current branch".to_string()));
        }

        let removed = {
            let mut branches = self.branches.write()?;
            branches.remove(name).is_some()
        };

        if removed {
            self.storages.write()?.remove(name);
        }

        Ok(removed)
    }

    /// Get diff between two commits
    pub fn diff(&self, from: &str, to: &str) -> Result<Diff> {
        let commits = self.commits.read()?;

        let from_commit = commits.get(from)
            .ok_or_else(|| VecStoreError::NotFound(format!("Commit: {}", from)))?;
        let to_commit = commits.get(to)
            .ok_or_else(|| VecStoreError::NotFound(format!("Commit: {}", to)))?;

        // Collect changes between commits
        // Simplified: just compare the changed IDs
        let from_changes: HashSet<_> = from_commit.changed_ids.iter().collect();
        let to_changes: HashSet<_> = to_commit.changed_ids.iter().collect();

        let added: Vec<_> = to_changes.difference(&from_changes).map(|s| (*s).clone()).collect();
        let modified: Vec<_> = to_changes.intersection(&from_changes).map(|s| (*s).clone()).collect();
        let deleted = to_commit.deleted_ids.clone();

        Ok(Diff {
            from_commit: from.to_string(),
            to_commit: to.to_string(),
            added: added.clone(),
            modified: modified.clone(),
            deleted: deleted.clone(),
            stats: DiffStats {
                total_added: added.len(),
                total_modified: modified.len(),
                total_deleted: deleted.len(),
                vectors_changed: added.len() + modified.len() + deleted.len(),
            },
        })
    }

    /// Merge a branch into current
    pub fn merge(&self, from_branch: &str, resolution: ConflictResolution) -> Result<MergeResult> {
        let current = self.current_branch.read()?.clone();

        if current == from_branch {
            return Err(VecStoreError::InvalidInput("Cannot merge branch into itself".to_string()));
        }

        // Get both storages
        let storages = self.storages.read()?;
        let from_storage = storages.get(from_branch)
            .ok_or_else(|| VecStoreError::NotFound(format!("Branch: {}", from_branch)))?;
        let to_storage = storages.get(&current)
            .ok_or_else(|| VecStoreError::NotFound(format!("Branch: {}", current)))?;

        // Find conflicts
        let mut conflicts = Vec::new();
        let from_ids: HashSet<_> = from_storage.all_ids().into_iter().collect();
        let to_ids: HashSet<_> = to_storage.all_ids().into_iter().collect();

        for id in from_ids.intersection(&to_ids) {
            let from_data = from_storage.get(id);
            let to_data = to_storage.get(id);

            if let (Some(f), Some(t)) = (from_data, to_data) {
                if f.vector != t.vector {
                    conflicts.push(MergeConflict {
                        id: id.clone(),
                        conflict_type: ConflictType::ModifyModify,
                        ours: Some(t.vector.clone()),
                        theirs: Some(f.vector.clone()),
                    });
                }
            }
        }

        // Handle conflicts based on resolution strategy
        if !conflicts.is_empty() && resolution == ConflictResolution::Abort {
            return Ok(MergeResult {
                success: false,
                commit: None,
                conflicts,
            });
        }

        // Apply merge
        drop(storages);
        let mut storages = self.storages.write()?;

        // First, collect data from source branch
        let items_to_copy: Vec<(String, VectorData)> = {
            let from_storage = storages.get(from_branch)
                .ok_or_else(|| VecStoreError::NotFound(format!("Branch: {}", from_branch)))?;
            from_ids.difference(&to_ids)
                .filter_map(|id| from_storage.get(id).map(|data| (id.clone(), data.clone())))
                .collect()
        };

        // Now apply to target branch
        let to_storage = storages.get_mut(&current)
            .ok_or_else(|| VecStoreError::NotFound(format!("Branch: {}", current)))?;

        // Copy new items from source
        for (id, data) in items_to_copy {
            to_storage.set(&id, data);
        }

        // Resolve conflicts
        for conflict in &conflicts {
            let resolved = match resolution {
                ConflictResolution::Ours => conflict.ours.clone(),
                ConflictResolution::Theirs => conflict.theirs.clone(),
                ConflictResolution::Average => {
                    if let (Some(ours), Some(theirs)) = (&conflict.ours, &conflict.theirs) {
                        Some(ours.iter().zip(theirs).map(|(a, b)| (a + b) / 2.0).collect())
                    } else {
                        None
                    }
                }
                ConflictResolution::Abort => unreachable!(),
            };

            if let Some(vec) = resolved {
                to_storage.set(&conflict.id, VectorData {
                    vector: vec,
                    metadata: None,
                    created_at: unix_timestamp(),
                    modified_at: unix_timestamp(),
                });
            }
        }

        drop(storages);

        // Create merge commit
        let commit = self.commit(&format!("Merge {} into {}", from_branch, current))?;

        Ok(MergeResult {
            success: true,
            commit: Some(commit),
            conflicts,
        })
    }

    /// List all branches
    pub fn list_branches(&self) -> Vec<Branch> {
        self.branches.read().map(|b| b.values().cloned().collect()).unwrap_or_default()
    }

    /// List commits on current branch
    pub fn log(&self, limit: usize) -> Vec<Commit> {
        let Ok(branch_name) = self.current_branch.read().map(|b| b.clone()) else {
            return Vec::new();
        };
        let Ok(branches) = self.branches.read() else {
            return Vec::new();
        };
        let Ok(commits) = self.commits.read() else {
            return Vec::new();
        };

        let mut result = Vec::new();
        let mut current_hash = branches.get(&branch_name).map(|b| b.head.clone());

        while let Some(hash) = current_hash {
            if let Some(commit) = commits.get(&hash) {
                result.push(commit.clone());
                current_hash = commit.parent.clone();
            } else {
                break;
            }

            if result.len() >= limit {
                break;
            }
        }

        result
    }

    /// Get status (uncommitted changes)
    pub fn status(&self) -> StoreStatus {
        let branch = self.current_branch.read().map(|b| b.clone()).unwrap_or_else(|_| "main".to_string());
        let Ok(storages) = self.storages.read() else {
            return StoreStatus {
                branch,
                is_dirty: false,
                modified_count: 0,
                deleted_count: 0,
            };
        };
        let Some(storage) = storages.get(&branch) else {
            return StoreStatus {
                branch,
                is_dirty: false,
                modified_count: 0,
                deleted_count: 0,
            };
        };

        StoreStatus {
            branch: branch.clone(),
            is_dirty: storage.is_dirty(),
            modified_count: storage.working.len(),
            deleted_count: storage.deleted.len(),
        }
    }

    /// Search current branch
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let Ok(branch) = self.current_branch.read().map(|b| b.clone()) else {
            return Vec::new();
        };
        let Ok(storages) = self.storages.read() else {
            return Vec::new();
        };
        let Some(storage) = storages.get(&branch) else {
            return Vec::new();
        };

        let mut results: Vec<_> = storage.all_ids().into_iter()
            .filter_map(|id| {
                storage.get(&id).map(|data| {
                    let score = cosine_similarity(query, &data.vector);
                    (id, score)
                })
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        results.into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect()
    }
}

/// Store status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStatus {
    pub branch: String,
    pub is_dirty: bool,
    pub modified_count: usize,
    pub deleted_count: usize,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
}

// ============================================================================
// HELPERS
// ============================================================================

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let store = VersionedStore::new(4).unwrap();

        store.upsert("doc1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
        let commit = store.commit("Add doc1").unwrap();

        assert!(!commit.hash.is_empty());
        assert_eq!(commit.changed_ids, vec!["doc1"]);
    }

    #[test]
    fn test_branching() {
        let store = VersionedStore::new(4).unwrap();

        store.upsert("doc1", vec![1.0; 4], None).unwrap();
        store.commit("Initial").unwrap();

        // Create and checkout branch
        store.branch("feature").unwrap();
        store.checkout("feature").unwrap();

        // Make changes on branch
        store.upsert("doc2", vec![0.5; 4], None).unwrap();
        store.commit("Add doc2").unwrap();

        // doc2 exists on feature
        assert!(store.get("doc2").is_some());

        // Switch back to main
        store.checkout("main").unwrap();

        // doc2 doesn't exist on main
        assert!(store.get("doc2").is_none());
    }

    #[test]
    fn test_merge() {
        let store = VersionedStore::new(4).unwrap();

        store.upsert("doc1", vec![1.0; 4], None).unwrap();
        store.commit("Initial").unwrap();

        // Create branch and add doc
        store.branch("feature").unwrap();
        store.checkout("feature").unwrap();
        store.upsert("doc2", vec![0.5; 4], None).unwrap();
        store.commit("Add doc2").unwrap();

        // Merge back to main
        store.checkout("main").unwrap();
        let result = store.merge("feature", ConflictResolution::Theirs).unwrap();

        assert!(result.success);
        assert!(store.get("doc2").is_some());
    }
}
