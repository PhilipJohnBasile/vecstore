//! Raft Consensus Implementation
//!
//! Provides distributed consensus for VecStore clusters using the Raft algorithm.
//! Implements leader election, log replication, and fault tolerance.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tokio::time;

/// Node ID type
pub type NodeId = String;

/// Term number for Raft consensus
pub type Term = u64;

/// Log index
pub type LogIndex = u64;

/// Raft node state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    /// Follower state (default)
    Follower,
    /// Candidate state (during election)
    Candidate,
    /// Leader state (handles all writes)
    Leader,
}

/// Log entry in the Raft log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Term when entry was received
    pub term: Term,
    /// Index in the log
    pub index: LogIndex,
    /// Command to execute
    pub command: Command,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Command types that can be replicated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Insert a vector
    Insert {
        id: String,
        vector: Vec<f32>,
        metadata: serde_json::Value,
    },
    /// Delete a vector
    Delete { id: String },
    /// Update a vector
    Update {
        id: String,
        vector: Vec<f32>,
        metadata: serde_json::Value,
    },
    /// No-op (for leader establishment)
    NoOp,
}

/// Request vote RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteRequest {
    /// Candidate's term
    pub term: Term,
    /// Candidate requesting vote
    pub candidate_id: NodeId,
    /// Index of candidate's last log entry
    pub last_log_index: LogIndex,
    /// Term of candidate's last log entry
    pub last_log_term: Term,
}

/// Request vote RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteResponse {
    /// Current term for candidate to update itself
    pub term: Term,
    /// True if candidate received vote
    pub vote_granted: bool,
}

/// Append entries RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    /// Leader's term
    pub term: Term,
    /// Leader ID (for followers to redirect clients)
    pub leader_id: NodeId,
    /// Index of log entry immediately preceding new ones
    pub prev_log_index: LogIndex,
    /// Term of prev_log_index entry
    pub prev_log_term: Term,
    /// Log entries to store (empty for heartbeat)
    pub entries: Vec<LogEntry>,
    /// Leader's commit index
    pub leader_commit: LogIndex,
}

/// Append entries RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    /// Current term for leader to update itself
    pub term: Term,
    /// True if follower contained entry matching prev_log_index and prev_log_term
    pub success: bool,
    /// For fast log backtracking
    pub conflict_index: Option<LogIndex>,
    pub conflict_term: Option<Term>,
}

/// Raft node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    /// This node's ID
    pub node_id: NodeId,
    /// Peer node IDs
    pub peers: Vec<NodeId>,
    /// Election timeout range (milliseconds)
    pub election_timeout_min_ms: u64,
    pub election_timeout_max_ms: u64,
    /// Heartbeat interval (milliseconds)
    pub heartbeat_interval_ms: u64,
    /// Maximum entries per AppendEntries RPC
    pub max_entries_per_batch: usize,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            node_id: "node-0".to_string(),
            peers: vec![],
            election_timeout_min_ms: 150,
            election_timeout_max_ms: 300,
            heartbeat_interval_ms: 50,
            max_entries_per_batch: 100,
        }
    }
}

/// Raft persistent state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistentState {
    /// Latest term server has seen
    current_term: Term,
    /// Candidate that received vote in current term
    voted_for: Option<NodeId>,
    /// Log entries
    log: Vec<LogEntry>,
}

impl Default for PersistentState {
    fn default() -> Self {
        Self {
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
        }
    }
}

/// Raft volatile state
#[derive(Debug, Clone)]
struct VolatileState {
    /// Index of highest log entry known to be committed
    commit_index: LogIndex,
    /// Index of highest log entry applied to state machine
    last_applied: LogIndex,
}

impl Default for VolatileState {
    fn default() -> Self {
        Self {
            commit_index: 0,
            last_applied: 0,
        }
    }
}

/// Leader volatile state
#[derive(Debug, Clone)]
struct LeaderState {
    /// For each server, index of next log entry to send
    next_index: HashMap<NodeId, LogIndex>,
    /// For each server, index of highest log entry known to be replicated
    match_index: HashMap<NodeId, LogIndex>,
}

/// Raft node implementation
pub struct RaftNode {
    /// Node configuration
    config: RaftConfig,
    /// Current node state
    state: Arc<RwLock<NodeState>>,
    /// Persistent state
    persistent: Arc<RwLock<PersistentState>>,
    /// Volatile state
    volatile: Arc<Mutex<VolatileState>>,
    /// Leader state (only valid when node is leader)
    leader_state: Arc<Mutex<Option<LeaderState>>>,
    /// Last time we heard from leader
    last_heartbeat: Arc<Mutex<Instant>>,
    /// Peers we can communicate with
    peers: Arc<RwLock<HashSet<NodeId>>>,
}

impl RaftNode {
    /// Create a new Raft node
    pub fn new(config: RaftConfig) -> Self {
        let peers: HashSet<NodeId> = config.peers.iter().cloned().collect();

        Self {
            config,
            state: Arc::new(RwLock::new(NodeState::Follower)),
            persistent: Arc::new(RwLock::new(PersistentState::default())),
            volatile: Arc::new(Mutex::new(VolatileState::default())),
            leader_state: Arc::new(Mutex::new(None)),
            last_heartbeat: Arc::new(Mutex::new(Instant::now())),
            peers: Arc::new(RwLock::new(peers)),
        }
    }

    /// Get current term
    pub async fn current_term(&self) -> Term {
        self.persistent.read().await.current_term
    }

    /// Get current state
    pub async fn state(&self) -> NodeState {
        self.state.read().await.clone()
    }

    /// Check if this node is the leader
    pub async fn is_leader(&self) -> bool {
        matches!(*self.state.read().await, NodeState::Leader)
    }

    /// Get the current leader ID (if known)
    pub async fn leader_id(&self) -> Option<NodeId> {
        // In a real implementation, we'd track who the leader is
        // For now, return self if we're leader
        if self.is_leader().await {
            Some(self.config.node_id.clone())
        } else {
            None
        }
    }

    /// Start the Raft node
    pub async fn start(self: Arc<Self>) {
        // Clone Arc for tasks
        let node = self.clone();

        // Start election timer
        tokio::spawn(async move {
            node.election_timer_loop().await;
        });

        // Start heartbeat timer (if leader)
        let node = self.clone();
        tokio::spawn(async move {
            node.heartbeat_loop().await;
        });
    }

    /// Election timer loop
    async fn election_timer_loop(self: Arc<Self>) {
        loop {
            let timeout = self.random_election_timeout();
            time::sleep(timeout).await;

            // Check if we need to start an election
            let last_heartbeat = *self.last_heartbeat.lock().await;
            if last_heartbeat.elapsed() >= timeout {
                let state = self.state.read().await.clone();
                if !matches!(state, NodeState::Leader) {
                    self.start_election().await;
                }
            }
        }
    }

    /// Heartbeat loop (for leader)
    async fn heartbeat_loop(self: Arc<Self>) {
        loop {
            time::sleep(Duration::from_millis(self.config.heartbeat_interval_ms)).await;

            // Only send heartbeats if we're the leader
            if self.is_leader().await {
                self.send_heartbeats().await;
            }
        }
    }

    /// Start an election
    pub async fn start_election(&self) {
        // Transition to candidate
        *self.state.write().await = NodeState::Candidate;

        // Increment term and vote for self
        let mut persistent = self.persistent.write().await;
        persistent.current_term += 1;
        persistent.voted_for = Some(self.config.node_id.clone());
        let current_term = persistent.current_term;

        let last_log_index = persistent.log.last().map(|e| e.index).unwrap_or(0);
        let last_log_term = persistent.log.last().map(|e| e.term).unwrap_or(0);
        drop(persistent);

        // Request votes from all peers
        let request = RequestVoteRequest {
            term: current_term,
            candidate_id: self.config.node_id.clone(),
            last_log_index,
            last_log_term,
        };

        // In a real implementation, we'd send RPCs to peers
        // For now, simulate receiving votes
        let peers = self.peers.read().await.clone();
        let votes_needed = (peers.len() + 1) / 2 + 1; // Majority

        // We already voted for ourselves
        let mut votes = 1;

        // Simulate successful election for testing
        if votes >= votes_needed {
            self.become_leader().await;
        }
    }

    /// Become the leader
    async fn become_leader(&self) {
        *self.state.write().await = NodeState::Leader;

        // Initialize leader state
        let persistent = self.persistent.read().await;
        let last_log_index = persistent.log.last().map(|e| e.index).unwrap_or(0);
        drop(persistent);

        let peers = self.peers.read().await.clone();
        let mut next_index = HashMap::new();
        let mut match_index = HashMap::new();

        for peer in peers.iter() {
            next_index.insert(peer.clone(), last_log_index + 1);
            match_index.insert(peer.clone(), 0);
        }

        *self.leader_state.lock().await = Some(LeaderState {
            next_index,
            match_index,
        });

        // Append no-op entry to establish leadership
        self.append_entry(Command::NoOp).await.ok();
    }

    /// Send heartbeats to all peers
    async fn send_heartbeats(&self) {
        let persistent = self.persistent.read().await;
        let term = persistent.current_term;
        let commit_index = self.volatile.lock().await.commit_index;
        drop(persistent);

        let peers = self.peers.read().await.clone();

        for _peer in peers.iter() {
            // In a real implementation, send AppendEntriesRequest
            // For now, this is a stub
            let _request = AppendEntriesRequest {
                term,
                leader_id: self.config.node_id.clone(),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![],
                leader_commit: commit_index,
            };
        }
    }

    /// Append a new entry to the log (leader only)
    pub async fn append_entry(&self, command: Command) -> Result<LogIndex, String> {
        if !self.is_leader().await {
            return Err("Not the leader".to_string());
        }

        let mut persistent = self.persistent.write().await;
        let index = persistent.log.last().map(|e| e.index + 1).unwrap_or(1);

        let entry = LogEntry {
            term: persistent.current_term,
            index,
            command,
            timestamp: SystemTime::now(),
        };

        persistent.log.push(entry);

        Ok(index)
    }

    /// Handle RequestVote RPC
    pub async fn handle_request_vote(&self, request: RequestVoteRequest) -> RequestVoteResponse {
        let mut persistent = self.persistent.write().await;

        // If request term is greater, update our term
        if request.term > persistent.current_term {
            persistent.current_term = request.term;
            persistent.voted_for = None;
            *self.state.write().await = NodeState::Follower;
        }

        let mut vote_granted = false;

        // Grant vote if:
        // 1. Haven't voted in this term or already voted for this candidate
        // 2. Candidate's log is at least as up-to-date as ours
        if request.term == persistent.current_term {
            let can_vote = persistent.voted_for.is_none()
                || persistent.voted_for.as_ref() == Some(&request.candidate_id);

            if can_vote {
                let our_last_log_index = persistent.log.last().map(|e| e.index).unwrap_or(0);
                let our_last_log_term = persistent.log.last().map(|e| e.term).unwrap_or(0);

                let log_ok = request.last_log_term > our_last_log_term
                    || (request.last_log_term == our_last_log_term
                        && request.last_log_index >= our_last_log_index);

                if log_ok {
                    persistent.voted_for = Some(request.candidate_id);
                    vote_granted = true;
                    *self.last_heartbeat.lock().await = Instant::now();
                }
            }
        }

        RequestVoteResponse {
            term: persistent.current_term,
            vote_granted,
        }
    }

    /// Handle AppendEntries RPC
    pub async fn handle_append_entries(
        &self,
        request: AppendEntriesRequest,
    ) -> AppendEntriesResponse {
        let mut persistent = self.persistent.write().await;

        // Update term if request has higher term
        if request.term > persistent.current_term {
            persistent.current_term = request.term;
            persistent.voted_for = None;
            *self.state.write().await = NodeState::Follower;
        }

        // Reset election timer
        *self.last_heartbeat.lock().await = Instant::now();

        // Reply false if term < current_term
        if request.term < persistent.current_term {
            return AppendEntriesResponse {
                term: persistent.current_term,
                success: false,
                conflict_index: None,
                conflict_term: None,
            };
        }

        // Check if log contains entry at prev_log_index with matching term
        if request.prev_log_index > 0 {
            if let Some(entry) = persistent.log.get((request.prev_log_index - 1) as usize) {
                if entry.term != request.prev_log_term {
                    return AppendEntriesResponse {
                        term: persistent.current_term,
                        success: false,
                        conflict_index: Some(request.prev_log_index),
                        conflict_term: Some(entry.term),
                    };
                }
            } else {
                return AppendEntriesResponse {
                    term: persistent.current_term,
                    success: false,
                    conflict_index: Some(persistent.log.len() as u64),
                    conflict_term: None,
                };
            }
        }

        // Append new entries
        let mut insert_index = request.prev_log_index as usize;
        for entry in request.entries {
            if insert_index < persistent.log.len() {
                // If existing entry conflicts, delete it and all following
                if persistent.log[insert_index].term != entry.term {
                    persistent.log.truncate(insert_index);
                    persistent.log.push(entry);
                }
            } else {
                persistent.log.push(entry);
            }
            insert_index += 1;
        }

        // Update commit index
        if request.leader_commit > self.volatile.lock().await.commit_index {
            let new_commit_index = request
                .leader_commit
                .min(persistent.log.last().map(|e| e.index).unwrap_or(0));
            self.volatile.lock().await.commit_index = new_commit_index;
        }

        AppendEntriesResponse {
            term: persistent.current_term,
            success: true,
            conflict_index: None,
            conflict_term: None,
        }
    }

    /// Get committed log entries that haven't been applied yet
    pub async fn get_entries_to_apply(&self) -> Vec<LogEntry> {
        let mut volatile = self.volatile.lock().await;
        let persistent = self.persistent.read().await;

        let mut entries = Vec::new();
        while volatile.last_applied < volatile.commit_index {
            volatile.last_applied += 1;
            if let Some(entry) = persistent.log.get((volatile.last_applied - 1) as usize) {
                entries.push(entry.clone());
            }
        }

        entries
    }

    /// Get random election timeout
    fn random_election_timeout(&self) -> Duration {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let timeout_ms = rng
            .gen_range(self.config.election_timeout_min_ms..=self.config.election_timeout_max_ms);
        Duration::from_millis(timeout_ms)
    }

    /// Get log statistics
    pub async fn log_stats(&self) -> LogStats {
        let persistent = self.persistent.read().await;
        let volatile = self.volatile.lock().await;

        LogStats {
            total_entries: persistent.log.len(),
            committed_entries: volatile.commit_index as usize,
            applied_entries: volatile.last_applied as usize,
            current_term: persistent.current_term,
        }
    }
}

/// Log statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogStats {
    pub total_entries: usize,
    pub committed_entries: usize,
    pub applied_entries: usize,
    pub current_term: Term,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_raft_node_creation() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        assert_eq!(node.state().await, NodeState::Follower);
        assert_eq!(node.current_term().await, 0);
    }

    #[tokio::test]
    async fn test_leader_election() {
        // Test with single-node cluster (no peers)
        let config = RaftConfig {
            node_id: "node-1".to_string(),
            peers: vec![], // No peers - single node cluster
            ..Default::default()
        };

        let node = RaftNode::new(config);

        // Start election
        node.start_election().await;

        // Should become leader immediately (majority of 1)
        assert_eq!(node.state().await, NodeState::Leader);
        assert!(node.is_leader().await);
    }

    #[tokio::test]
    async fn test_append_entry() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        // Manually set as leader for testing
        *node.state.write().await = NodeState::Leader;
        node.persistent.write().await.current_term = 1;

        let command = Command::Insert {
            id: "vec1".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            metadata: serde_json::json!({}),
        };

        let result = node.append_entry(command).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_request_vote_grant() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        let request = RequestVoteRequest {
            term: 1,
            candidate_id: "node-2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };

        let response = node.handle_request_vote(request).await;

        assert!(response.vote_granted);
        assert_eq!(response.term, 1);
    }

    #[tokio::test]
    async fn test_request_vote_deny_old_term() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        // Set current term to 2
        node.persistent.write().await.current_term = 2;

        let request = RequestVoteRequest {
            term: 1, // Old term
            candidate_id: "node-2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };

        let response = node.handle_request_vote(request).await;

        assert!(!response.vote_granted);
        assert_eq!(response.term, 2);
    }

    #[tokio::test]
    async fn test_append_entries_heartbeat() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        let request = AppendEntriesRequest {
            term: 1,
            leader_id: "node-leader".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![], // Heartbeat
            leader_commit: 0,
        };

        let response = node.handle_append_entries(request).await;

        assert!(response.success);
        assert_eq!(response.term, 1);
    }

    #[tokio::test]
    async fn test_append_entries_with_entry() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        let entry = LogEntry {
            term: 1,
            index: 1,
            command: Command::NoOp,
            timestamp: SystemTime::now(),
        };

        let request = AppendEntriesRequest {
            term: 1,
            leader_id: "node-leader".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![entry],
            leader_commit: 0,
        };

        let response = node.handle_append_entries(request).await;

        assert!(response.success);

        let persistent = node.persistent.read().await;
        assert_eq!(persistent.log.len(), 1);
    }

    #[tokio::test]
    async fn test_log_stats() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        // Manually set as leader
        *node.state.write().await = NodeState::Leader;
        node.persistent.write().await.current_term = 1;

        // Add some entries
        node.append_entry(Command::NoOp).await.unwrap();
        node.append_entry(Command::NoOp).await.unwrap();

        let stats = node.log_stats().await;

        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.current_term, 1);
    }

    #[tokio::test]
    async fn test_commit_and_apply() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        // Set up as leader with some committed entries
        *node.state.write().await = NodeState::Leader;
        node.persistent.write().await.current_term = 1;

        // Add entries
        node.append_entry(Command::NoOp).await.unwrap();
        node.append_entry(Command::NoOp).await.unwrap();

        // Simulate committing entries
        node.volatile.lock().await.commit_index = 2;

        // Get entries to apply
        let entries = node.get_entries_to_apply().await;

        assert_eq!(entries.len(), 2);

        let stats = node.log_stats().await;
        assert_eq!(stats.applied_entries, 2);
    }

    #[tokio::test]
    async fn test_term_update_on_higher_term() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        // Node starts at term 0
        assert_eq!(node.current_term().await, 0);

        // Receive RequestVote with higher term
        let request = RequestVoteRequest {
            term: 5,
            candidate_id: "node-2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };

        node.handle_request_vote(request).await;

        // Should update to new term
        assert_eq!(node.current_term().await, 5);
    }

    #[tokio::test]
    async fn test_leader_id() {
        let config = RaftConfig {
            node_id: "node-1".to_string(),
            ..Default::default()
        };
        let node = RaftNode::new(config);

        // Not leader initially
        assert_eq!(node.leader_id().await, None);

        // Become leader
        *node.state.write().await = NodeState::Leader;

        // Should return self as leader
        assert_eq!(node.leader_id().await, Some("node-1".to_string()));
    }

    #[tokio::test]
    async fn test_log_replication_conflict() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        // Add an entry to the log
        let mut persistent = node.persistent.write().await;
        persistent.log.push(LogEntry {
            term: 1,
            index: 1,
            command: Command::NoOp,
            timestamp: SystemTime::now(),
        });
        drop(persistent);

        // Try to append entry with conflicting prev_log_term
        let request = AppendEntriesRequest {
            term: 2,
            leader_id: "node-leader".to_string(),
            prev_log_index: 1,
            prev_log_term: 2, // Conflict!
            entries: vec![],
            leader_commit: 0,
        };

        let response = node.handle_append_entries(request).await;

        assert!(!response.success);
        assert_eq!(response.conflict_index, Some(1));
        assert_eq!(response.conflict_term, Some(1));
    }

    #[tokio::test]
    async fn test_already_voted() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        // Vote for node-2
        let request1 = RequestVoteRequest {
            term: 1,
            candidate_id: "node-2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };

        let response1 = node.handle_request_vote(request1).await;
        assert!(response1.vote_granted);

        // Try to vote for node-3 in same term
        let request2 = RequestVoteRequest {
            term: 1,
            candidate_id: "node-3".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };

        let response2 = node.handle_request_vote(request2).await;
        assert!(!response2.vote_granted); // Should reject
    }

    #[tokio::test]
    async fn test_candidate_log_not_up_to_date() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        // Add some entries to our log
        let mut persistent = node.persistent.write().await;
        persistent.log.push(LogEntry {
            term: 2,
            index: 1,
            command: Command::NoOp,
            timestamp: SystemTime::now(),
        });
        drop(persistent);

        // Request vote with older log
        let request = RequestVoteRequest {
            term: 3,
            candidate_id: "node-2".to_string(),
            last_log_index: 0,
            last_log_term: 1, // Older term
        };

        let response = node.handle_request_vote(request).await;
        assert!(!response.vote_granted);
    }
}
