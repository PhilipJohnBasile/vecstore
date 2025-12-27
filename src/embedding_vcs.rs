//! Embedding Version Control: Git-like Versioning for Embeddings
//!
//! Track embedding model versions, compare changes, and rollback.
//! Treats embeddings as code with full version control.
//!
//! ## Features
//!
//! - **Commits**: Snapshot embeddings with messages
//! - **Branches**: Experiment with different embedding models
//! - **Diff**: Compare embeddings between versions
//! - **Rollback**: Revert to previous versions
//! - **Blame**: Track which model produced each embedding
//!
//! ## Example
//!
//! ```rust,no_run
//! use vecstore::embedding_vcs::{EmbeddingVCS, CommitOptions};
//!
//! let mut vcs = EmbeddingVCS::new("./embeddings_repo")?;
//!
//! // Commit current embeddings
//! let commit_id = vcs.commit(&embeddings, CommitOptions {
//!     message: "Initial embeddings with text-embedding-3-small".to_string(),
//!     model: "text-embedding-3-small".to_string(),
//!     ..Default::default()
//! })?;
//!
//! // Later: update embeddings
//! vcs.commit(&new_embeddings, CommitOptions {
//!     message: "Upgrade to text-embedding-3-large".to_string(),
//!     model: "text-embedding-3-large".to_string(),
//!     ..Default::default()
//! })?;
//!
//! // Compare versions
//! let diff = vcs.diff(&commit_id, &latest_id)?;
//! println!("Changed: {} vectors", diff.changed.len());
//!
//! // Rollback if needed
//! vcs.checkout(&commit_id)?;
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

// ============================================================================
// CORE TYPES
// ============================================================================

/// Unique identifier for a commit
pub type CommitId = String;

/// Unique identifier for a branch
pub type BranchName = String;

/// A version control commit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    /// Unique commit ID (hash)
    pub id: CommitId,

    /// Parent commit ID (None for initial commit)
    pub parent: Option<CommitId>,

    /// Commit message
    pub message: String,

    /// Embedding model used
    pub model: String,

    /// Model version
    pub model_version: Option<String>,

    /// Timestamp
    pub timestamp: i64,

    /// Author
    pub author: String,

    /// Vector IDs included in this commit
    pub vector_ids: Vec<String>,

    /// Snapshot reference (path or key)
    pub snapshot_ref: String,

    /// Metadata
    pub metadata: HashMap<String, String>,

    /// Statistics
    pub stats: CommitStats,
}

/// Commit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitStats {
    pub total_vectors: usize,
    pub added: usize,
    pub modified: usize,
    pub removed: usize,
    pub dimension: usize,
    pub total_bytes: usize,
}

/// Commit options
#[derive(Debug, Clone, Default)]
pub struct CommitOptions {
    /// Commit message
    pub message: String,

    /// Embedding model name
    pub model: String,

    /// Model version
    pub model_version: Option<String>,

    /// Author name
    pub author: String,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Branch reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub name: BranchName,
    pub head: CommitId,
    pub created_at: i64,
}

/// Diff between two commits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitDiff {
    /// Source commit
    pub from_commit: CommitId,

    /// Target commit
    pub to_commit: CommitId,

    /// Added vector IDs
    pub added: Vec<String>,

    /// Removed vector IDs
    pub removed: Vec<String>,

    /// Changed vector IDs (same ID, different embedding)
    pub changed: Vec<VectorDiff>,

    /// Model change
    pub model_change: Option<ModelChange>,

    /// Statistics
    pub stats: DiffStats,
}

/// Diff for a single vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDiff {
    pub id: String,
    pub cosine_similarity: f32,
    pub l2_distance: f32,
    pub dimension_changes: Vec<(usize, f32, f32)>, // (dim, old, new)
}

/// Model change between commits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelChange {
    pub from_model: String,
    pub to_model: String,
    pub from_version: Option<String>,
    pub to_version: Option<String>,
}

/// Diff statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffStats {
    pub total_added: usize,
    pub total_removed: usize,
    pub total_changed: usize,
    pub avg_similarity: f32,
    pub min_similarity: f32,
    pub dimension_changed: bool,
}

// ============================================================================
// EMBEDDING SNAPSHOT
// ============================================================================

/// Snapshot of embeddings at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Commit ID this snapshot belongs to
    pub commit_id: CommitId,

    /// Vector data: id -> embedding
    pub vectors: HashMap<String, Vec<f32>>,

    /// Vector metadata
    pub metadata: HashMap<String, VectorMetadata>,

    /// Dimension
    pub dimension: usize,
}

/// Metadata for a single vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    /// Source document/item
    pub source: Option<String>,

    /// Creation timestamp
    pub created_at: i64,

    /// Last modified timestamp
    pub modified_at: i64,

    /// Model that generated this embedding
    pub model: String,

    /// Custom tags
    pub tags: Vec<String>,
}

// ============================================================================
// BLAME RESULT
// ============================================================================

/// Result of blame operation (who/what created each vector)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlameResult {
    pub vector_id: String,
    pub commit: CommitId,
    pub model: String,
    pub author: String,
    pub timestamp: i64,
    pub message: String,
}

// ============================================================================
// EMBEDDING VCS
// ============================================================================

/// Embedding Version Control System
pub struct EmbeddingVCS {
    /// Repository path
    repo_path: PathBuf,

    /// All commits
    commits: HashMap<CommitId, Commit>,

    /// Branches
    branches: HashMap<BranchName, Branch>,

    /// Current branch
    current_branch: BranchName,

    /// HEAD commit
    head: Option<CommitId>,

    /// Current working snapshot
    working: Option<Snapshot>,

    /// Vector to commit mapping (for blame)
    vector_commits: HashMap<String, CommitId>,
}

impl EmbeddingVCS {
    /// Create or open a VCS repository
    pub fn new(repo_path: impl Into<PathBuf>) -> Result<Self> {
        let repo_path = repo_path.into();

        // Create repo directory if needed
        if !repo_path.exists() {
            std::fs::create_dir_all(&repo_path)?;
        }

        // Try to load existing repository
        let state_path = repo_path.join("vcs_state.json");
        if state_path.exists() {
            let state_json = std::fs::read_to_string(&state_path)?;
            let state: VCSState = serde_json::from_str(&state_json)?;

            return Ok(Self {
                repo_path,
                commits: state.commits,
                branches: state.branches,
                current_branch: state.current_branch,
                head: state.head,
                working: None,
                vector_commits: state.vector_commits,
            });
        }

        // Create new repository
        let mut branches = HashMap::new();
        branches.insert(
            "main".to_string(),
            Branch {
                name: "main".to_string(),
                head: String::new(),
                created_at: chrono::Utc::now().timestamp(),
            },
        );

        Ok(Self {
            repo_path,
            commits: HashMap::new(),
            branches,
            current_branch: "main".to_string(),
            head: None,
            working: None,
            vector_commits: HashMap::new(),
        })
    }

    /// Commit embeddings
    pub fn commit(
        &mut self,
        embeddings: &HashMap<String, Vec<f32>>,
        options: CommitOptions,
    ) -> Result<CommitId> {
        let dimension = embeddings.values().next().map(|v| v.len()).unwrap_or(0);

        // Compute statistics
        let stats = self.compute_commit_stats(embeddings);

        // Generate commit ID
        let commit_id = self.generate_commit_id(&options.message);

        // Save snapshot
        let snapshot = Snapshot {
            commit_id: commit_id.clone(),
            vectors: embeddings.clone(),
            metadata: embeddings
                .keys()
                .map(|id| {
                    (
                        id.clone(),
                        VectorMetadata {
                            source: None,
                            created_at: chrono::Utc::now().timestamp(),
                            modified_at: chrono::Utc::now().timestamp(),
                            model: options.model.clone(),
                            tags: Vec::new(),
                        },
                    )
                })
                .collect(),
            dimension,
        };

        let snapshot_ref = format!("snapshots/{}.bin", commit_id);
        self.save_snapshot(&snapshot, &snapshot_ref)?;

        // Create commit
        let commit = Commit {
            id: commit_id.clone(),
            parent: self.head.clone(),
            message: options.message,
            model: options.model,
            model_version: options.model_version,
            timestamp: chrono::Utc::now().timestamp(),
            author: options.author,
            vector_ids: embeddings.keys().cloned().collect(),
            snapshot_ref,
            metadata: options.metadata,
            stats,
        };

        // Update vector commits for blame
        for id in embeddings.keys() {
            self.vector_commits.insert(id.clone(), commit_id.clone());
        }

        // Store commit
        self.commits.insert(commit_id.clone(), commit);

        // Update HEAD and branch
        self.head = Some(commit_id.clone());
        if let Some(branch) = self.branches.get_mut(&self.current_branch) {
            branch.head = commit_id.clone();
        }

        // Persist state
        self.save_state()?;

        Ok(commit_id)
    }

    /// Compute commit statistics
    fn compute_commit_stats(&self, embeddings: &HashMap<String, Vec<f32>>) -> CommitStats {
        let total_vectors = embeddings.len();
        let dimension = embeddings.values().next().map(|v| v.len()).unwrap_or(0);
        let total_bytes = total_vectors * dimension * 4;

        // Compare with previous commit
        let (added, modified, removed) = if let Some(ref head_id) = self.head {
            if let Some(prev_commit) = self.commits.get(head_id) {
                let prev_ids: HashSet<&String> = prev_commit.vector_ids.iter().collect();
                let curr_ids: HashSet<&String> = embeddings.keys().collect();

                let added = curr_ids.difference(&prev_ids).count();
                let removed = prev_ids.difference(&curr_ids).count();
                let common = curr_ids.intersection(&prev_ids).count();

                // For modified, we'd need to compare actual vectors
                // Simplified: assume common vectors might be modified
                (added, common, removed)
            } else {
                (total_vectors, 0, 0)
            }
        } else {
            (total_vectors, 0, 0)
        };

        CommitStats {
            total_vectors,
            added,
            modified,
            removed,
            dimension,
            total_bytes,
        }
    }

    /// Generate commit ID from message and timestamp
    fn generate_commit_id(&self, message: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        message.hash(&mut hasher);
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0).hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Save snapshot to disk
    fn save_snapshot(&self, snapshot: &Snapshot, path: &str) -> Result<()> {
        let full_path = self.repo_path.join(path);
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = bincode::serialize(snapshot)?;
        std::fs::write(full_path, data)?;
        Ok(())
    }

    /// Load snapshot from disk
    fn load_snapshot(&self, path: &str) -> Result<Snapshot> {
        let full_path = self.repo_path.join(path);
        let data = std::fs::read(full_path)?;
        Ok(bincode::deserialize(&data)?)
    }

    /// Save VCS state
    fn save_state(&self) -> Result<()> {
        let state = VCSState {
            commits: self.commits.clone(),
            branches: self.branches.clone(),
            current_branch: self.current_branch.clone(),
            head: self.head.clone(),
            vector_commits: self.vector_commits.clone(),
        };
        let state_json = serde_json::to_string_pretty(&state)?;
        std::fs::write(self.repo_path.join("vcs_state.json"), state_json)?;
        Ok(())
    }

    /// Get commit by ID
    pub fn get_commit(&self, commit_id: &str) -> Option<&Commit> {
        self.commits.get(commit_id)
    }

    /// Get current HEAD
    pub fn head(&self) -> Option<&CommitId> {
        self.head.as_ref()
    }

    /// Checkout a specific commit
    pub fn checkout(&mut self, commit_id: &str) -> Result<HashMap<String, Vec<f32>>> {
        let commit = self
            .commits
            .get(commit_id)
            .ok_or_else(|| anyhow!("Commit not found: {}", commit_id))?;

        let snapshot = self.load_snapshot(&commit.snapshot_ref)?;
        self.working = Some(snapshot.clone());

        Ok(snapshot.vectors)
    }

    /// Diff between two commits
    pub fn diff(&self, from_commit: &str, to_commit: &str) -> Result<CommitDiff> {
        let from = self
            .commits
            .get(from_commit)
            .ok_or_else(|| anyhow!("Commit not found: {}", from_commit))?;
        let to = self
            .commits
            .get(to_commit)
            .ok_or_else(|| anyhow!("Commit not found: {}", to_commit))?;

        let from_snapshot = self.load_snapshot(&from.snapshot_ref)?;
        let to_snapshot = self.load_snapshot(&to.snapshot_ref)?;

        let from_ids: HashSet<&String> = from_snapshot.vectors.keys().collect();
        let to_ids: HashSet<&String> = to_snapshot.vectors.keys().collect();

        // Find added, removed
        let added: Vec<String> = to_ids.difference(&from_ids).map(|s| (*s).clone()).collect();
        let removed: Vec<String> = from_ids.difference(&to_ids).map(|s| (*s).clone()).collect();

        // Find changed
        let common: Vec<&String> = from_ids.intersection(&to_ids).cloned().collect();
        let mut changed = Vec::new();
        let mut total_similarity = 0.0f32;
        let mut min_similarity = 1.0f32;

        for id in &common {
            let from_vec = &from_snapshot.vectors[*id];
            let to_vec = &to_snapshot.vectors[*id];

            let similarity = self.cosine_similarity(from_vec, to_vec);
            let l2_dist = self.l2_distance(from_vec, to_vec);

            if similarity < 0.9999 {
                // Consider changed if not identical
                let dim_changes: Vec<(usize, f32, f32)> = from_vec
                    .iter()
                    .zip(to_vec)
                    .enumerate()
                    .filter(|(_, (a, b))| (*a - *b).abs() > 0.001)
                    .map(|(i, (&a, &b))| (i, a, b))
                    .take(10)
                    .collect();

                changed.push(VectorDiff {
                    id: (*id).clone(),
                    cosine_similarity: similarity,
                    l2_distance: l2_dist,
                    dimension_changes: dim_changes,
                });

                total_similarity += similarity;
                min_similarity = min_similarity.min(similarity);
            }
        }

        let avg_similarity = if !changed.is_empty() {
            total_similarity / changed.len() as f32
        } else {
            1.0
        };

        let model_change = if from.model != to.model {
            Some(ModelChange {
                from_model: from.model.clone(),
                to_model: to.model.clone(),
                from_version: from.model_version.clone(),
                to_version: to.model_version.clone(),
            })
        } else {
            None
        };

        let total_added = added.len();
        let total_removed = removed.len();
        let total_changed = changed.len();

        Ok(CommitDiff {
            from_commit: from_commit.to_string(),
            to_commit: to_commit.to_string(),
            added,
            removed,
            changed,
            model_change,
            stats: DiffStats {
                total_added,
                total_removed,
                total_changed,
                avg_similarity,
                min_similarity,
                dimension_changed: from_snapshot.dimension != to_snapshot.dimension,
            },
        })
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b + 1e-10)
    }

    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Blame: find which commit created a vector
    pub fn blame(&self, vector_id: &str) -> Option<BlameResult> {
        let commit_id = self.vector_commits.get(vector_id)?;
        let commit = self.commits.get(commit_id)?;

        Some(BlameResult {
            vector_id: vector_id.to_string(),
            commit: commit_id.clone(),
            model: commit.model.clone(),
            author: commit.author.clone(),
            timestamp: commit.timestamp,
            message: commit.message.clone(),
        })
    }

    /// Create a new branch
    pub fn create_branch(&mut self, name: &str) -> Result<()> {
        if self.branches.contains_key(name) {
            return Err(anyhow!("Branch already exists: {}", name));
        }

        let head = self.head.clone().unwrap_or_default();

        self.branches.insert(
            name.to_string(),
            Branch {
                name: name.to_string(),
                head,
                created_at: chrono::Utc::now().timestamp(),
            },
        );

        self.save_state()?;
        Ok(())
    }

    /// Switch to a branch
    pub fn switch_branch(&mut self, name: &str) -> Result<()> {
        if !self.branches.contains_key(name) {
            return Err(anyhow!("Branch not found: {}", name));
        }

        self.current_branch = name.to_string();
        self.head = Some(self.branches[name].head.clone());

        self.save_state()?;
        Ok(())
    }

    /// Get all branches
    pub fn list_branches(&self) -> Vec<&Branch> {
        self.branches.values().collect()
    }

    /// Get current branch
    pub fn current_branch(&self) -> &str {
        &self.current_branch
    }

    /// Get commit log
    pub fn log(&self, max_count: usize) -> Vec<&Commit> {
        let mut commits = Vec::new();
        let mut current = self.head.clone();

        while let Some(ref commit_id) = current {
            if commits.len() >= max_count {
                break;
            }

            if let Some(commit) = self.commits.get(commit_id) {
                commits.push(commit);
                current = commit.parent.clone();
            } else {
                break;
            }
        }

        commits
    }

    /// Rollback to a specific commit (creates new commit with old state)
    pub fn rollback(&mut self, commit_id: &str) -> Result<CommitId> {
        let embeddings = self.checkout(commit_id)?;
        let original_commit = self.commits.get(commit_id).unwrap().clone();

        let new_commit_id = self.commit(
            &embeddings,
            CommitOptions {
                message: format!("Rollback to {}: {}", commit_id, original_commit.message),
                model: original_commit.model,
                model_version: original_commit.model_version,
                author: "VCS Rollback".to_string(),
                metadata: HashMap::new(),
            },
        )?;

        Ok(new_commit_id)
    }

    /// Get total number of commits
    pub fn commit_count(&self) -> usize {
        self.commits.len()
    }
}

/// Serializable VCS state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VCSState {
    commits: HashMap<CommitId, Commit>,
    branches: HashMap<BranchName, Branch>,
    current_branch: BranchName,
    head: Option<CommitId>,
    vector_commits: HashMap<String, CommitId>,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn generate_embeddings(n: usize, dim: usize) -> HashMap<String, Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::rng();

        (0..n)
            .map(|i| {
                let vec: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
                (format!("vec_{}", i), vec)
            })
            .collect()
    }

    #[test]
    fn test_vcs_creation() {
        let dir = TempDir::new().unwrap();
        let vcs = EmbeddingVCS::new(dir.path()).unwrap();

        assert_eq!(vcs.current_branch(), "main");
        assert!(vcs.head().is_none());
    }

    #[test]
    fn test_commit() {
        let dir = TempDir::new().unwrap();
        let mut vcs = EmbeddingVCS::new(dir.path()).unwrap();

        let embeddings = generate_embeddings(100, 128);
        let commit_id = vcs
            .commit(
                &embeddings,
                CommitOptions {
                    message: "Initial commit".to_string(),
                    model: "test-model".to_string(),
                    author: "test".to_string(),
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(!commit_id.is_empty());
        assert_eq!(vcs.head(), Some(&commit_id));

        let commit = vcs.get_commit(&commit_id).unwrap();
        assert_eq!(commit.vector_ids.len(), 100);
        assert_eq!(commit.stats.added, 100);
    }

    #[test]
    fn test_multiple_commits() {
        let dir = TempDir::new().unwrap();
        let mut vcs = EmbeddingVCS::new(dir.path()).unwrap();

        // First commit
        let emb1 = generate_embeddings(50, 64);
        let commit1 = vcs
            .commit(
                &emb1,
                CommitOptions {
                    message: "First".to_string(),
                    model: "model-v1".to_string(),
                    author: "test".to_string(),
                    ..Default::default()
                },
            )
            .unwrap();

        // Second commit
        let emb2 = generate_embeddings(75, 64);
        let commit2 = vcs
            .commit(
                &emb2,
                CommitOptions {
                    message: "Second".to_string(),
                    model: "model-v2".to_string(),
                    author: "test".to_string(),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_ne!(commit1, commit2);
        assert_eq!(vcs.commit_count(), 2);

        let log = vcs.log(10);
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].id, commit2);
        assert_eq!(log[1].id, commit1);
    }

    #[test]
    fn test_diff() {
        let dir = TempDir::new().unwrap();
        let mut vcs = EmbeddingVCS::new(dir.path()).unwrap();

        // First commit
        let mut emb1 = generate_embeddings(50, 64);
        let commit1 = vcs
            .commit(
                &emb1,
                CommitOptions {
                    message: "First".to_string(),
                    model: "model-v1".to_string(),
                    author: "test".to_string(),
                    ..Default::default()
                },
            )
            .unwrap();

        // Modify some vectors, add new ones
        emb1.insert("new_vec".to_string(), vec![0.5; 64]);
        emb1.remove("vec_0"); // Remove one

        let commit2 = vcs
            .commit(
                &emb1,
                CommitOptions {
                    message: "Second".to_string(),
                    model: "model-v2".to_string(),
                    author: "test".to_string(),
                    ..Default::default()
                },
            )
            .unwrap();

        let diff = vcs.diff(&commit1, &commit2).unwrap();

        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.removed.len(), 1);
        assert!(diff.model_change.is_some());
    }

    #[test]
    fn test_checkout() {
        let dir = TempDir::new().unwrap();
        let mut vcs = EmbeddingVCS::new(dir.path()).unwrap();

        let emb1 = generate_embeddings(50, 64);
        let commit1 = vcs
            .commit(
                &emb1,
                CommitOptions {
                    message: "First".to_string(),
                    model: "model".to_string(),
                    author: "test".to_string(),
                    ..Default::default()
                },
            )
            .unwrap();

        let emb2 = generate_embeddings(75, 64);
        let _commit2 = vcs
            .commit(
                &emb2,
                CommitOptions {
                    message: "Second".to_string(),
                    model: "model".to_string(),
                    author: "test".to_string(),
                    ..Default::default()
                },
            )
            .unwrap();

        // Checkout first commit
        let restored = vcs.checkout(&commit1).unwrap();
        assert_eq!(restored.len(), 50);
    }

    #[test]
    fn test_blame() {
        let dir = TempDir::new().unwrap();
        let mut vcs = EmbeddingVCS::new(dir.path()).unwrap();

        let emb = generate_embeddings(10, 64);
        let commit_id = vcs
            .commit(
                &emb,
                CommitOptions {
                    message: "Test commit".to_string(),
                    model: "test-model".to_string(),
                    author: "test-author".to_string(),
                    ..Default::default()
                },
            )
            .unwrap();

        let blame = vcs.blame("vec_5").unwrap();
        assert_eq!(blame.commit, commit_id);
        assert_eq!(blame.model, "test-model");
        assert_eq!(blame.author, "test-author");
    }

    #[test]
    fn test_branches() {
        let dir = TempDir::new().unwrap();
        let mut vcs = EmbeddingVCS::new(dir.path()).unwrap();

        // Initial commit on main
        let emb = generate_embeddings(10, 64);
        vcs.commit(
            &emb,
            CommitOptions {
                message: "Initial".to_string(),
                model: "model".to_string(),
                author: "test".to_string(),
                ..Default::default()
            },
        )
        .unwrap();

        // Create new branch
        vcs.create_branch("feature").unwrap();
        assert_eq!(vcs.list_branches().len(), 2);

        // Switch to new branch
        vcs.switch_branch("feature").unwrap();
        assert_eq!(vcs.current_branch(), "feature");
    }

    #[test]
    fn test_persistence() {
        let dir = TempDir::new().unwrap();

        // Create and commit
        {
            let mut vcs = EmbeddingVCS::new(dir.path()).unwrap();
            let emb = generate_embeddings(10, 64);
            vcs.commit(
                &emb,
                CommitOptions {
                    message: "Persisted".to_string(),
                    model: "model".to_string(),
                    author: "test".to_string(),
                    ..Default::default()
                },
            )
            .unwrap();
        }

        // Reopen and verify
        {
            let vcs = EmbeddingVCS::new(dir.path()).unwrap();
            assert_eq!(vcs.commit_count(), 1);
            assert!(vcs.head().is_some());
        }
    }
}
