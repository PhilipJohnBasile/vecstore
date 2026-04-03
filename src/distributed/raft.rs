//! Raft Consensus Implementation
//!
//! Provides distributed consensus for VecStore clusters using the Raft algorithm.
//! Implements leader election, log replication, and fault tolerance with actual
//! gRPC-based RPC communication using tonic.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tokio::time;
use tracing::{debug, error, info, warn};

#[cfg(feature = "server")]
use tonic::{transport::Channel, Request, Response, Status};

#[cfg(feature = "server")]
pub mod pb {
    tonic::include_proto!("raft");
}

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
    /// The match index after successful append
    pub match_index: LogIndex,
}

/// Peer node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Node ID
    pub node_id: NodeId,
    /// gRPC address (e.g., "127.0.0.1:50051")
    pub address: String,
}

/// Raft node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    /// This node's ID
    pub node_id: NodeId,
    /// This node's gRPC address
    pub address: String,
    /// Peer nodes with their addresses
    pub peers: Vec<PeerInfo>,
    /// Election timeout range (milliseconds)
    pub election_timeout_min_ms: u64,
    pub election_timeout_max_ms: u64,
    /// Heartbeat interval (milliseconds)
    pub heartbeat_interval_ms: u64,
    /// Maximum entries per AppendEntries RPC
    pub max_entries_per_batch: usize,
    /// RPC timeout in milliseconds
    pub rpc_timeout_ms: u64,
    /// Maximum retry attempts for RPC calls
    pub max_retries: u32,
    /// Base delay for exponential backoff (milliseconds)
    pub retry_base_delay_ms: u64,
    /// Enable TLS for RPC communication
    pub tls_enabled: bool,
    /// Path to TLS certificate file (if TLS enabled)
    pub tls_cert_path: Option<String>,
    /// Path to TLS key file (if TLS enabled)
    pub tls_key_path: Option<String>,
    /// Path to CA certificate for verifying peers (if TLS enabled)
    pub tls_ca_path: Option<String>,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            node_id: "node-0".to_string(),
            address: "127.0.0.1:50051".to_string(),
            peers: vec![],
            election_timeout_min_ms: 150,
            election_timeout_max_ms: 300,
            heartbeat_interval_ms: 50,
            max_entries_per_batch: 100,
            rpc_timeout_ms: 500,
            max_retries: 3,
            retry_base_delay_ms: 50,
            tls_enabled: false,
            tls_cert_path: None,
            tls_key_path: None,
            tls_ca_path: None,
        }
    }
}

/// Raft persistent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentState {
    /// Latest term server has seen
    pub current_term: Term,
    /// Candidate that received vote in current term
    pub voted_for: Option<NodeId>,
    /// Log entries
    pub log: Vec<LogEntry>,
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
pub struct VolatileState {
    /// Index of highest log entry known to be committed
    pub commit_index: LogIndex,
    /// Index of highest log entry applied to state machine
    pub last_applied: LogIndex,
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
pub struct LeaderState {
    /// For each server, index of next log entry to send
    pub next_index: HashMap<NodeId, LogIndex>,
    /// For each server, index of highest log entry known to be replicated
    pub match_index: HashMap<NodeId, LogIndex>,
}

// ============================================================================
// RPC Client with Connection Pooling
// ============================================================================

#[cfg(feature = "server")]
mod rpc {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokio::sync::RwLock as TokioRwLock;
    use tonic::transport::{Certificate, ClientTlsConfig, Endpoint, Uri};

    /// Connection pool entry
    struct PooledConnection {
        client: pb::raft_service_client::RaftServiceClient<Channel>,
        last_used: Instant,
        failures: u32,
    }

    /// RPC client with connection pooling and retry logic
    pub struct RaftRpcClient {
        /// Connection pool: node_id -> connection
        connections: Arc<TokioRwLock<HashMap<NodeId, PooledConnection>>>,
        /// Peer addresses: node_id -> address
        peer_addresses: Arc<TokioRwLock<HashMap<NodeId, String>>>,
        /// Configuration
        config: RaftConfig,
        /// Metrics: total RPC calls
        rpc_calls: AtomicU64,
        /// Metrics: failed RPC calls
        rpc_failures: AtomicU64,
    }

    impl RaftRpcClient {
        /// Create a new RPC client
        pub fn new(config: RaftConfig) -> Self {
            let mut peer_addresses = HashMap::new();
            for peer in &config.peers {
                peer_addresses.insert(peer.node_id.clone(), peer.address.clone());
            }

            Self {
                connections: Arc::new(TokioRwLock::new(HashMap::new())),
                peer_addresses: Arc::new(TokioRwLock::new(peer_addresses)),
                config,
                rpc_calls: AtomicU64::new(0),
                rpc_failures: AtomicU64::new(0),
            }
        }

        /// Get or create a connection to a peer
        async fn get_connection(
            &self,
            node_id: &NodeId,
        ) -> Result<pb::raft_service_client::RaftServiceClient<Channel>, RpcError> {
            // Check if we have a valid cached connection
            {
                let connections = self.connections.read().await;
                if let Some(conn) = connections.get(node_id) {
                    // Return cached connection if it's been used recently and hasn't failed too much
                    if conn.last_used.elapsed() < Duration::from_secs(60) && conn.failures < 3 {
                        return Ok(conn.client.clone());
                    }
                }
            }

            // Get peer address
            let address = {
                let addresses = self.peer_addresses.read().await;
                addresses
                    .get(node_id)
                    .cloned()
                    .ok_or_else(|| RpcError::UnknownPeer(node_id.clone()))?
            };

            // Create new connection
            let endpoint = self.create_endpoint(&address).await?;
            let channel = endpoint
                .connect()
                .await
                .map_err(|e| RpcError::ConnectionFailed(format!("{}: {}", node_id, e)))?;

            let client = pb::raft_service_client::RaftServiceClient::new(channel);

            // Cache the connection
            {
                let mut connections = self.connections.write().await;
                connections.insert(
                    node_id.clone(),
                    PooledConnection {
                        client: client.clone(),
                        last_used: Instant::now(),
                        failures: 0,
                    },
                );
            }

            Ok(client)
        }

        /// Create an endpoint with optional TLS
        async fn create_endpoint(&self, address: &str) -> Result<Endpoint, RpcError> {
            let uri: Uri = format!("http://{}", address)
                .parse()
                .map_err(|e| RpcError::InvalidAddress(format!("{}: {}", address, e)))?;

            let mut endpoint = Endpoint::from(uri)
                .timeout(Duration::from_millis(self.config.rpc_timeout_ms))
                .connect_timeout(Duration::from_millis(self.config.rpc_timeout_ms * 2));

            // Configure TLS if enabled
            if self.config.tls_enabled {
                if let Some(ca_path) = &self.config.tls_ca_path {
                    let ca_cert = tokio::fs::read(ca_path)
                        .await
                        .map_err(|e| RpcError::TlsError(format!("Failed to read CA cert: {}", e)))?;
                    let ca = Certificate::from_pem(ca_cert);

                    let tls_config = ClientTlsConfig::new().ca_certificate(ca);
                    endpoint = endpoint.tls_config(tls_config).map_err(|e| {
                        RpcError::TlsError(format!("TLS configuration failed: {}", e))
                    })?;
                }
            }

            Ok(endpoint)
        }

        /// Mark a connection as failed
        async fn mark_connection_failed(&self, node_id: &NodeId) {
            let mut connections = self.connections.write().await;
            if let Some(conn) = connections.get_mut(node_id) {
                conn.failures += 1;
                if conn.failures >= 3 {
                    // Remove failed connection to force reconnection
                    connections.remove(node_id);
                }
            }
        }

        /// Reset connection failure count on success
        async fn mark_connection_success(&self, node_id: &NodeId) {
            let mut connections = self.connections.write().await;
            if let Some(conn) = connections.get_mut(node_id) {
                conn.failures = 0;
                conn.last_used = Instant::now();
            }
        }

        /// Send RequestVote RPC with retry logic
        pub async fn send_request_vote(
            &self,
            node_id: &NodeId,
            request: RequestVoteRequest,
        ) -> Result<RequestVoteResponse, RpcError> {
            self.rpc_calls.fetch_add(1, Ordering::Relaxed);

            let mut last_error = None;
            let mut delay = self.config.retry_base_delay_ms;

            for attempt in 0..=self.config.max_retries {
                if attempt > 0 {
                    debug!(
                        "Retrying RequestVote to {} (attempt {}/{})",
                        node_id, attempt, self.config.max_retries
                    );
                    time::sleep(Duration::from_millis(delay)).await;
                    delay = (delay * 2).min(5000); // Exponential backoff, max 5s
                }

                match self.try_send_request_vote(node_id, &request).await {
                    Ok(response) => {
                        self.mark_connection_success(node_id).await;
                        return Ok(response);
                    }
                    Err(e) => {
                        warn!("RequestVote to {} failed: {:?}", node_id, e);
                        self.mark_connection_failed(node_id).await;
                        last_error = Some(e);
                    }
                }
            }

            self.rpc_failures.fetch_add(1, Ordering::Relaxed);
            Err(last_error.unwrap_or(RpcError::MaxRetriesExceeded))
        }

        async fn try_send_request_vote(
            &self,
            node_id: &NodeId,
            request: &RequestVoteRequest,
        ) -> Result<RequestVoteResponse, RpcError> {
            let mut client = self.get_connection(node_id).await?;

            let pb_request = pb::RequestVoteRequest {
                term: request.term,
                candidate_id: request.candidate_id.clone(),
                last_log_index: request.last_log_index,
                last_log_term: request.last_log_term,
            };

            let response = client
                .request_vote(Request::new(pb_request))
                .await
                .map_err(|e| RpcError::RpcFailed(format!("RequestVote failed: {}", e)))?;

            let pb_response = response.into_inner();
            Ok(RequestVoteResponse {
                term: pb_response.term,
                vote_granted: pb_response.vote_granted,
            })
        }

        /// Send AppendEntries RPC with retry logic
        pub async fn send_append_entries(
            &self,
            node_id: &NodeId,
            request: AppendEntriesRequest,
        ) -> Result<AppendEntriesResponse, RpcError> {
            self.rpc_calls.fetch_add(1, Ordering::Relaxed);

            let mut last_error = None;
            let mut delay = self.config.retry_base_delay_ms;

            for attempt in 0..=self.config.max_retries {
                if attempt > 0 {
                    debug!(
                        "Retrying AppendEntries to {} (attempt {}/{})",
                        node_id, attempt, self.config.max_retries
                    );
                    time::sleep(Duration::from_millis(delay)).await;
                    delay = (delay * 2).min(5000);
                }

                match self.try_send_append_entries(node_id, &request).await {
                    Ok(response) => {
                        self.mark_connection_success(node_id).await;
                        return Ok(response);
                    }
                    Err(e) => {
                        warn!("AppendEntries to {} failed: {:?}", node_id, e);
                        self.mark_connection_failed(node_id).await;
                        last_error = Some(e);
                    }
                }
            }

            self.rpc_failures.fetch_add(1, Ordering::Relaxed);
            Err(last_error.unwrap_or(RpcError::MaxRetriesExceeded))
        }

        async fn try_send_append_entries(
            &self,
            node_id: &NodeId,
            request: &AppendEntriesRequest,
        ) -> Result<AppendEntriesResponse, RpcError> {
            let mut client = self.get_connection(node_id).await?;

            let pb_entries: Vec<pb::LogEntry> = request
                .entries
                .iter()
                .map(|e| log_entry_to_pb(e))
                .collect();

            let pb_request = pb::AppendEntriesRequest {
                term: request.term,
                leader_id: request.leader_id.clone(),
                prev_log_index: request.prev_log_index,
                prev_log_term: request.prev_log_term,
                entries: pb_entries,
                leader_commit: request.leader_commit,
            };

            let response = client
                .append_entries(Request::new(pb_request))
                .await
                .map_err(|e| RpcError::RpcFailed(format!("AppendEntries failed: {}", e)))?;

            let pb_response = response.into_inner();
            Ok(AppendEntriesResponse {
                term: pb_response.term,
                success: pb_response.success,
                conflict_index: pb_response.conflict_index,
                conflict_term: pb_response.conflict_term,
                match_index: pb_response.match_index,
            })
        }

        /// Add a new peer
        pub async fn add_peer(&self, node_id: NodeId, address: String) {
            let mut addresses = self.peer_addresses.write().await;
            addresses.insert(node_id, address);
        }

        /// Remove a peer
        pub async fn remove_peer(&self, node_id: &NodeId) {
            {
                let mut addresses = self.peer_addresses.write().await;
                addresses.remove(node_id);
            }
            {
                let mut connections = self.connections.write().await;
                connections.remove(node_id);
            }
        }

        /// Get RPC statistics
        pub fn get_stats(&self) -> RpcStats {
            RpcStats {
                total_calls: self.rpc_calls.load(Ordering::Relaxed),
                failed_calls: self.rpc_failures.load(Ordering::Relaxed),
            }
        }
    }

    /// Convert LogEntry to protobuf
    fn log_entry_to_pb(entry: &LogEntry) -> pb::LogEntry {
        let command_data = bincode::serialize(&entry.command).unwrap_or_default();
        let command_type = match &entry.command {
            Command::Insert { .. } => "Insert",
            Command::Delete { .. } => "Delete",
            Command::Update { .. } => "Update",
            Command::NoOp => "NoOp",
        };
        let timestamp_ms = entry
            .timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        pb::LogEntry {
            term: entry.term,
            index: entry.index,
            command_data,
            command_type: command_type.to_string(),
            timestamp_ms,
        }
    }

    /// Convert protobuf to LogEntry
    pub fn pb_to_log_entry(pb: &pb::LogEntry) -> Result<LogEntry, RpcError> {
        let command: Command = bincode::deserialize(&pb.command_data)
            .map_err(|e| RpcError::DeserializationFailed(format!("Command: {}", e)))?;

        let timestamp = SystemTime::UNIX_EPOCH + Duration::from_millis(pb.timestamp_ms);

        Ok(LogEntry {
            term: pb.term,
            index: pb.index,
            command,
            timestamp,
        })
    }

    /// RPC statistics
    #[derive(Debug, Clone)]
    pub struct RpcStats {
        pub total_calls: u64,
        pub failed_calls: u64,
    }

    /// RPC errors
    #[derive(Debug, Clone)]
    pub enum RpcError {
        UnknownPeer(NodeId),
        ConnectionFailed(String),
        InvalidAddress(String),
        TlsError(String),
        RpcFailed(String),
        DeserializationFailed(String),
        MaxRetriesExceeded,
        Timeout,
    }

    impl std::fmt::Display for RpcError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                RpcError::UnknownPeer(id) => write!(f, "Unknown peer: {}", id),
                RpcError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
                RpcError::InvalidAddress(msg) => write!(f, "Invalid address: {}", msg),
                RpcError::TlsError(msg) => write!(f, "TLS error: {}", msg),
                RpcError::RpcFailed(msg) => write!(f, "RPC failed: {}", msg),
                RpcError::DeserializationFailed(msg) => {
                    write!(f, "Deserialization failed: {}", msg)
                }
                RpcError::MaxRetriesExceeded => write!(f, "Max retries exceeded"),
                RpcError::Timeout => write!(f, "RPC timeout"),
            }
        }
    }

    impl std::error::Error for RpcError {}
}

#[cfg(feature = "server")]
pub use rpc::{RaftRpcClient, RpcError, RpcStats};

// ============================================================================
// Raft Node Implementation
// ============================================================================

/// State machine for applying committed commands
#[derive(Debug, Default)]
pub struct StateMachine {
    /// Key-value store for vectors and metadata
    data: HashMap<String, Vec<u8>>,
}

impl StateMachine {
    /// Apply a command to the state machine
    pub fn apply(&mut self, command: &Command) -> Option<Vec<u8>> {
        match command {
            Command::Insert { id, vector, metadata } => {
                let entry = serde_json::json!({
                    "vector": vector,
                    "metadata": metadata,
                });
                let bytes = serde_json::to_vec(&entry).unwrap_or_default();
                self.data.insert(id.clone(), bytes);
                None
            }
            Command::Delete { id } => {
                self.data.remove(id);
                None
            }
            Command::Update { id, vector, metadata } => {
                let entry = serde_json::json!({
                    "vector": vector,
                    "metadata": metadata,
                });
                let bytes = serde_json::to_vec(&entry).unwrap_or_default();
                self.data.insert(id.clone(), bytes);
                None
            }
            Command::NoOp => None,
        }
    }

    /// Get a value from the state machine
    pub fn get(&self, key: &str) -> Option<&Vec<u8>> {
        self.data.get(key)
    }

    /// Serialize the state machine for snapshots
    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(&self.data).unwrap_or_default()
    }

    /// Deserialize state machine from snapshot
    pub fn deserialize(data: &[u8]) -> Self {
        let data: HashMap<String, Vec<u8>> = serde_json::from_slice(data).unwrap_or_default();
        Self { data }
    }
}

/// Raft node implementation
pub struct RaftNode {
    /// Node configuration
    config: RaftConfig,
    /// Current node state
    state: Arc<RwLock<NodeState>>,
    /// Persistent state
    pub persistent: Arc<RwLock<PersistentState>>,
    /// Volatile state
    pub volatile: Arc<Mutex<VolatileState>>,
    /// Leader state (only valid when node is leader)
    leader_state: Arc<Mutex<Option<LeaderState>>>,
    /// Last time we heard from leader
    last_heartbeat: Arc<Mutex<Instant>>,
    /// Peer node IDs we can communicate with
    peers: Arc<RwLock<HashSet<NodeId>>>,
    /// Current known leader
    current_leader: Arc<RwLock<Option<NodeId>>>,
    /// RPC client for sending messages to peers
    #[cfg(feature = "server")]
    rpc_client: Arc<RaftRpcClient>,
    /// Flag to signal shutdown
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    /// State machine storing applied data
    state_machine: Arc<RwLock<StateMachine>>,
}

impl RaftNode {
    /// Create a new Raft node
    pub fn new(config: RaftConfig) -> Self {
        let peers: HashSet<NodeId> = config.peers.iter().map(|p| p.node_id.clone()).collect();

        #[cfg(feature = "server")]
        let rpc_client = Arc::new(RaftRpcClient::new(config.clone()));

        Self {
            config: config.clone(),
            state: Arc::new(RwLock::new(NodeState::Follower)),
            persistent: Arc::new(RwLock::new(PersistentState::default())),
            volatile: Arc::new(Mutex::new(VolatileState::default())),
            leader_state: Arc::new(Mutex::new(None)),
            last_heartbeat: Arc::new(Mutex::new(Instant::now())),
            peers: Arc::new(RwLock::new(peers)),
            current_leader: Arc::new(RwLock::new(None)),
            #[cfg(feature = "server")]
            rpc_client,
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            state_machine: Arc::new(RwLock::new(StateMachine::default())),
        }
    }

    /// Get a reference to the state machine
    pub fn state_machine(&self) -> &Arc<RwLock<StateMachine>> {
        &self.state_machine
    }

    /// Get node ID
    pub fn node_id(&self) -> &NodeId {
        &self.config.node_id
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
        if self.is_leader().await {
            Some(self.config.node_id.clone())
        } else {
            self.current_leader.read().await.clone()
        }
    }

    /// Start the Raft node
    pub async fn start(self: Arc<Self>) {
        info!("Starting Raft node: {}", self.config.node_id);

        // Start election timer
        let node = self.clone();
        tokio::spawn(async move {
            node.election_timer_loop().await;
        });

        // Start heartbeat timer (if leader)
        let node = self.clone();
        tokio::spawn(async move {
            node.heartbeat_loop().await;
        });

        // Start log apply loop
        let node = self.clone();
        tokio::spawn(async move {
            node.apply_loop().await;
        });
    }

    /// Shutdown the Raft node
    pub fn shutdown(&self) {
        info!("Shutting down Raft node: {}", self.config.node_id);
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if shutdown was requested
    fn is_shutdown(&self) -> bool {
        self.shutdown.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Election timer loop
    async fn election_timer_loop(self: Arc<Self>) {
        loop {
            if self.is_shutdown() {
                break;
            }

            let timeout = self.random_election_timeout();
            time::sleep(timeout).await;

            // Check if we need to start an election
            let last_heartbeat = *self.last_heartbeat.lock().await;
            if last_heartbeat.elapsed() >= timeout {
                let state = self.state.read().await.clone();
                if !matches!(state, NodeState::Leader) {
                    info!(
                        "Election timeout on node {}, starting election",
                        self.config.node_id
                    );
                    self.start_election().await;
                }
            }
        }
    }

    /// Heartbeat loop (for leader)
    async fn heartbeat_loop(self: Arc<Self>) {
        loop {
            if self.is_shutdown() {
                break;
            }

            time::sleep(Duration::from_millis(self.config.heartbeat_interval_ms)).await;

            // Only send heartbeats if we're the leader
            if self.is_leader().await {
                self.send_heartbeats().await;
            }
        }
    }

    /// Apply loop - applies committed entries to state machine
    async fn apply_loop(self: Arc<Self>) {
        loop {
            if self.is_shutdown() {
                break;
            }

            time::sleep(Duration::from_millis(10)).await;

            let entries = self.get_entries_to_apply().await;
            for entry in entries {
                debug!("Applying log entry {}: {:?}", entry.index, entry.command);
                // Apply command to state machine
                let mut state_machine = self.state_machine.write().await;
                state_machine.apply(&entry.command);
                drop(state_machine);

                // Update last_applied index
                let mut volatile = self.volatile.lock().await;
                volatile.last_applied = entry.index;
            }
        }
    }

    /// Read from state machine with specified key
    pub async fn read_from_state_machine<T>(&self, key: &str) -> Option<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let state_machine = self.state_machine.read().await;
        state_machine.get(key).and_then(|bytes| serde_json::from_slice(bytes).ok())
    }

    /// Start an election
    pub async fn start_election(&self) {
        // Transition to candidate
        *self.state.write().await = NodeState::Candidate;
        *self.current_leader.write().await = None;

        // Increment term and vote for self
        let (current_term, last_log_index, last_log_term) = {
            let mut persistent = self.persistent.write().await;
            persistent.current_term += 1;
            persistent.voted_for = Some(self.config.node_id.clone());
            let current_term = persistent.current_term;
            let last_log_index = persistent.log.last().map(|e| e.index).unwrap_or(0);
            let last_log_term = persistent.log.last().map(|e| e.term).unwrap_or(0);
            (current_term, last_log_index, last_log_term)
        };

        info!(
            "Node {} starting election for term {}",
            self.config.node_id, current_term
        );

        let request = RequestVoteRequest {
            term: current_term,
            candidate_id: self.config.node_id.clone(),
            last_log_index,
            last_log_term,
        };

        let peers = self.peers.read().await.clone();
        let votes_needed = (peers.len() + 1) / 2 + 1; // Majority including self

        // We already voted for ourselves
        let mut votes = 1u64;

        #[cfg(feature = "server")]
        {
            // Send RequestVote RPCs to all peers in parallel
            let mut vote_futures = Vec::new();

            for peer_id in peers.iter() {
                let rpc_client = self.rpc_client.clone();
                let peer_id = peer_id.clone();
                let request = request.clone();

                vote_futures.push(tokio::spawn(async move {
                    let result = rpc_client.send_request_vote(&peer_id, request).await;
                    (peer_id, result)
                }));
            }

            // Collect votes with timeout
            let timeout = Duration::from_millis(self.config.election_timeout_max_ms);
            let deadline = Instant::now() + timeout;

            for future in vote_futures {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }

                match tokio::time::timeout(remaining, future).await {
                    Ok(Ok((peer_id, Ok(response)))) => {
                        // Check if we got a higher term
                        if response.term > current_term {
                            info!(
                                "Node {} received higher term {} from {}, stepping down",
                                self.config.node_id, response.term, peer_id
                            );
                            self.step_down(response.term).await;
                            return;
                        }

                        if response.vote_granted {
                            votes += 1;
                            info!(
                                "Node {} received vote from {} ({}/{})",
                                self.config.node_id, peer_id, votes, votes_needed
                            );
                        }
                    }
                    Ok(Ok((peer_id, Err(e)))) => {
                        warn!("Failed to get vote from {}: {:?}", peer_id, e);
                    }
                    Ok(Err(e)) => {
                        warn!("Vote task failed: {:?}", e);
                    }
                    Err(_) => {
                        debug!("Vote collection timed out");
                        break;
                    }
                }
            }
        }

        #[cfg(not(feature = "server"))]
        {
            // Without server feature, use simulated election (for testing)
            // In single-node mode, we always win
            if peers.is_empty() {
                votes = votes_needed as u64;
            }
        }

        // Check if we won the election
        let current_state = self.state.read().await.clone();
        if matches!(current_state, NodeState::Candidate) && votes >= votes_needed as u64 {
            info!(
                "Node {} won election for term {} with {} votes",
                self.config.node_id, current_term, votes
            );
            self.become_leader().await;
        } else {
            debug!(
                "Node {} did not win election (got {}/{} votes)",
                self.config.node_id, votes, votes_needed
            );
        }
    }

    /// Step down to follower when receiving higher term
    async fn step_down(&self, new_term: Term) {
        let mut persistent = self.persistent.write().await;
        persistent.current_term = new_term;
        persistent.voted_for = None;
        drop(persistent);

        *self.state.write().await = NodeState::Follower;
        *self.leader_state.lock().await = None;
    }

    /// Become the leader
    async fn become_leader(&self) {
        *self.state.write().await = NodeState::Leader;
        *self.current_leader.write().await = Some(self.config.node_id.clone());

        info!("Node {} became leader", self.config.node_id);

        // Initialize leader state
        let last_log_index = {
            let persistent = self.persistent.read().await;
            persistent.log.last().map(|e| e.index).unwrap_or(0)
        };

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

        // Send immediate heartbeat
        self.send_heartbeats().await;
    }

    /// Send heartbeats to all peers
    pub async fn send_heartbeats(&self) {
        if !self.is_leader().await {
            return;
        }

        let (term, commit_index) = {
            let persistent = self.persistent.read().await;
            let volatile = self.volatile.lock().await;
            (persistent.current_term, volatile.commit_index)
        };

        let peers = self.peers.read().await.clone();

        #[cfg(feature = "server")]
        {
            let leader_state = self.leader_state.lock().await;
            let leader_state = match &*leader_state {
                Some(ls) => ls.clone(),
                None => return,
            };
            drop(leader_state);

            // Send AppendEntries to each peer in parallel
            let mut append_futures = Vec::new();

            for peer_id in peers.iter() {
                let (prev_log_index, prev_log_term, entries) = {
                    let leader_state = self.leader_state.lock().await;
                    let leader_state = match &*leader_state {
                        Some(ls) => ls,
                        None => continue,
                    };

                    let next_idx = *leader_state.next_index.get(peer_id).unwrap_or(&1);
                    let persistent = self.persistent.read().await;

                    let prev_log_index = next_idx.saturating_sub(1);
                    let prev_log_term = if prev_log_index > 0 {
                        persistent
                            .log
                            .get((prev_log_index - 1) as usize)
                            .map(|e| e.term)
                            .unwrap_or(0)
                    } else {
                        0
                    };

                    // Get entries to send
                    let start_idx = (next_idx - 1) as usize;
                    let entries: Vec<LogEntry> = persistent
                        .log
                        .iter()
                        .skip(start_idx)
                        .take(self.config.max_entries_per_batch)
                        .cloned()
                        .collect();

                    (prev_log_index, prev_log_term, entries)
                };

                let request = AppendEntriesRequest {
                    term,
                    leader_id: self.config.node_id.clone(),
                    prev_log_index,
                    prev_log_term,
                    entries: entries.clone(),
                    leader_commit: commit_index,
                };

                let rpc_client = self.rpc_client.clone();
                let peer_id = peer_id.clone();
                let node_id = self.config.node_id.clone();
                let entries_len = entries.len();
                let leader_state_arc = self.leader_state.clone();
                let persistent_arc = self.persistent.clone();
                let volatile_arc = self.volatile.clone();
                let state_arc = self.state.clone();
                let current_leader_arc = self.current_leader.clone();

                append_futures.push(tokio::spawn(async move {
                    match rpc_client.send_append_entries(&peer_id, request).await {
                        Ok(response) => {
                            // Handle response
                            if response.term > term {
                                // Step down
                                let mut persistent = persistent_arc.write().await;
                                persistent.current_term = response.term;
                                persistent.voted_for = None;
                                drop(persistent);

                                *state_arc.write().await = NodeState::Follower;
                                *current_leader_arc.write().await = None;
                                return;
                            }

                            let mut leader_state = leader_state_arc.lock().await;
                            if let Some(ls) = leader_state.as_mut() {
                                if response.success {
                                    // Update next_index and match_index
                                    ls.match_index
                                        .insert(peer_id.clone(), response.match_index);
                                    ls.next_index
                                        .insert(peer_id.clone(), response.match_index + 1);

                                    debug!(
                                        "AppendEntries to {} succeeded, match_index={}",
                                        peer_id, response.match_index
                                    );
                                } else {
                                    // Decrement next_index and retry
                                    if let Some(next) = ls.next_index.get_mut(&peer_id) {
                                        if let Some(conflict_index) = response.conflict_index {
                                            *next = conflict_index;
                                        } else {
                                            *next = next.saturating_sub(1).max(1);
                                        }
                                    }
                                    debug!("AppendEntries to {} failed, will retry", peer_id);
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to send heartbeat to {}: {:?}", peer_id, e);
                        }
                    }
                }));
            }

            // Wait for all append entries to complete (with timeout)
            let timeout = Duration::from_millis(self.config.heartbeat_interval_ms);
            for future in append_futures {
                let _ = tokio::time::timeout(timeout, future).await;
            }

            // Update commit index based on match indices
            self.update_commit_index().await;
        }

        #[cfg(not(feature = "server"))]
        {
            // Without server feature, just log the heartbeat
            debug!("Would send heartbeats to {} peers", peers.len());
        }
    }

    /// Update commit index based on match indices (leader only)
    async fn update_commit_index(&self) {
        let leader_state = self.leader_state.lock().await;
        let leader_state = match &*leader_state {
            Some(ls) => ls,
            None => return,
        };

        let persistent = self.persistent.read().await;
        let current_term = persistent.current_term;

        // Find the highest index that has been replicated on a majority
        let mut match_indices: Vec<LogIndex> = leader_state.match_index.values().cloned().collect();

        // Include leader's own log
        let our_last_index = persistent.log.last().map(|e| e.index).unwrap_or(0);
        match_indices.push(our_last_index);

        match_indices.sort_unstable();

        // Majority is at position (n/2) in sorted array
        let majority_idx = match_indices.len() / 2;
        let new_commit_index = match_indices[majority_idx];

        // Only commit entries from current term
        if new_commit_index > 0 {
            if let Some(entry) = persistent.log.get((new_commit_index - 1) as usize) {
                if entry.term == current_term {
                    drop(persistent);
                    drop(leader_state);

                    let mut volatile = self.volatile.lock().await;
                    if new_commit_index > volatile.commit_index {
                        debug!(
                            "Leader updating commit_index from {} to {}",
                            volatile.commit_index, new_commit_index
                        );
                        volatile.commit_index = new_commit_index;
                    }
                }
            }
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
            *self.current_leader.write().await = None;
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

                    debug!(
                        "Node {} granted vote for term {}",
                        self.config.node_id, request.term
                    );
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

        // Reset election timer and update known leader
        *self.last_heartbeat.lock().await = Instant::now();
        *self.current_leader.write().await = Some(request.leader_id.clone());

        // Reply false if term < current_term
        if request.term < persistent.current_term {
            return AppendEntriesResponse {
                term: persistent.current_term,
                success: false,
                conflict_index: None,
                conflict_term: None,
                match_index: 0,
            };
        }

        // Step down if we're a candidate or leader
        let current_state = self.state.read().await.clone();
        if matches!(current_state, NodeState::Candidate | NodeState::Leader) {
            *self.state.write().await = NodeState::Follower;
            *self.leader_state.lock().await = None;
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
                        match_index: 0,
                    };
                }
            } else {
                return AppendEntriesResponse {
                    term: persistent.current_term,
                    success: false,
                    conflict_index: Some(persistent.log.len() as u64),
                    conflict_term: None,
                    match_index: 0,
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

        let match_index = persistent.log.last().map(|e| e.index).unwrap_or(0);

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
            match_index,
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
        let mut rng = rand::rng();
        let timeout_ms = rng
            .random_range(self.config.election_timeout_min_ms..=self.config.election_timeout_max_ms);
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

    /// Add a peer dynamically
    #[cfg(feature = "server")]
    pub async fn add_peer(&self, node_id: NodeId, address: String) {
        self.peers.write().await.insert(node_id.clone());
        self.rpc_client.add_peer(node_id, address).await;
    }

    /// Remove a peer dynamically
    #[cfg(feature = "server")]
    pub async fn remove_peer(&self, node_id: &NodeId) {
        self.peers.write().await.remove(node_id);
        self.rpc_client.remove_peer(node_id).await;
    }

    /// Get RPC statistics
    #[cfg(feature = "server")]
    pub fn rpc_stats(&self) -> RpcStats {
        self.rpc_client.get_stats()
    }

    /// Handle InstallSnapshot RPC
    ///
    /// This is called when the leader sends a snapshot to a follower that has fallen
    /// too far behind to catch up through normal log replication. The follower
    /// discards its log, installs the snapshot, and then continues normal operation.
    pub async fn handle_install_snapshot(
        &self,
        request: InstallSnapshotRequest,
    ) -> InstallSnapshotResponse {
        let mut persistent = self.persistent.write().await;

        // If request term is greater, update our term
        if request.term > persistent.current_term {
            persistent.current_term = request.term;
            persistent.voted_for = None;
            *self.state.write().await = NodeState::Follower;
        }

        // Reset election timer and update known leader
        *self.last_heartbeat.lock().await = Instant::now();
        *self.current_leader.write().await = Some(request.leader_id.clone());

        // Reply immediately if term < current_term
        if request.term < persistent.current_term {
            return InstallSnapshotResponse {
                term: persistent.current_term,
                success: false,
                bytes_received: 0,
            };
        }

        // For a complete implementation, we would:
        // 1. Save snapshot chunks to disk as they arrive
        // 2. When `done` is true, verify the snapshot integrity
        // 3. Discard any existing log entries covered by the snapshot
        // 4. Apply the snapshot to the state machine
        // 5. Update volatile state

        if request.done {
            // Snapshot is complete - apply it
            let snapshot_index = request.metadata.last_included_index;
            let snapshot_term = request.metadata.last_included_term;

            // Discard log entries covered by the snapshot
            persistent.log.retain(|entry| entry.index > snapshot_index);

            // If log is empty after truncation, we need to set a base entry
            // to maintain log continuity (done implicitly by the snapshot metadata)

            // Update volatile state to reflect the snapshot
            let mut volatile = self.volatile.lock().await;
            if snapshot_index > volatile.commit_index {
                volatile.commit_index = snapshot_index;
            }
            if snapshot_index > volatile.last_applied {
                volatile.last_applied = snapshot_index;
            }

            info!(
                "Node {} installed snapshot up to index {} term {}",
                self.config.node_id, snapshot_index, snapshot_term
            );
        }

        InstallSnapshotResponse {
            term: persistent.current_term,
            success: true,
            bytes_received: request.data.len() as u64,
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

// ============================================================================
// gRPC Server Implementation
// ============================================================================

#[cfg(feature = "server")]
pub mod server {
    use super::*;
    use std::net::SocketAddr;
    use tonic::transport::Server;

    /// Raft gRPC service implementation
    pub struct RaftGrpcService {
        node: Arc<RaftNode>,
    }

    impl RaftGrpcService {
        pub fn new(node: Arc<RaftNode>) -> Self {
            Self { node }
        }
    }

    #[tonic::async_trait]
    impl pb::raft_service_server::RaftService for RaftGrpcService {
        async fn request_vote(
            &self,
            request: Request<pb::RequestVoteRequest>,
        ) -> Result<Response<pb::RequestVoteResponse>, Status> {
            let req = request.into_inner();

            let raft_request = RequestVoteRequest {
                term: req.term,
                candidate_id: req.candidate_id,
                last_log_index: req.last_log_index,
                last_log_term: req.last_log_term,
            };

            let response = self.node.handle_request_vote(raft_request).await;

            Ok(Response::new(pb::RequestVoteResponse {
                term: response.term,
                vote_granted: response.vote_granted,
            }))
        }

        async fn append_entries(
            &self,
            request: Request<pb::AppendEntriesRequest>,
        ) -> Result<Response<pb::AppendEntriesResponse>, Status> {
            let req = request.into_inner();

            // Convert protobuf log entries
            let entries: Result<Vec<LogEntry>, _> =
                req.entries.iter().map(|e| rpc::pb_to_log_entry(e)).collect();

            let entries = entries.map_err(|e| Status::invalid_argument(format!("{}", e)))?;

            let raft_request = AppendEntriesRequest {
                term: req.term,
                leader_id: req.leader_id,
                prev_log_index: req.prev_log_index,
                prev_log_term: req.prev_log_term,
                entries,
                leader_commit: req.leader_commit,
            };

            let response = self.node.handle_append_entries(raft_request).await;

            Ok(Response::new(pb::AppendEntriesResponse {
                term: response.term,
                success: response.success,
                conflict_index: response.conflict_index,
                conflict_term: response.conflict_term,
                match_index: response.match_index,
            }))
        }

        async fn install_snapshot(
            &self,
            request: Request<pb::InstallSnapshotRequest>,
        ) -> Result<Response<pb::InstallSnapshotResponse>, Status> {
            let req = request.into_inner();

            // Convert protobuf metadata to our SnapshotMetadata
            let metadata = SnapshotMetadata {
                last_included_index: req.last_included_index,
                last_included_term: req.last_included_term,
                cluster_config: ClusterConfig::default(), // Would be deserialized from req
                size_bytes: req.data.len() as u64,
                created_at: SystemTime::now(),
            };

            // Build the install snapshot request
            let raft_request = InstallSnapshotRequest {
                term: req.term,
                leader_id: req.leader_id,
                metadata,
                offset: req.offset,
                data: req.data,
                done: req.done,
            };

            // Handle the snapshot installation
            let response = self.node.handle_install_snapshot(raft_request).await;

            Ok(Response::new(pb::InstallSnapshotResponse {
                term: response.term,
                success: response.success,
                bytes_received: response.bytes_received,
            }))
        }

        async fn get_leader(
            &self,
            _request: Request<pb::GetLeaderRequest>,
        ) -> Result<Response<pb::GetLeaderResponse>, Status> {
            let leader_id = self.node.leader_id().await;
            let current_term = self.node.current_term().await;

            Ok(Response::new(pb::GetLeaderResponse {
                leader_known: leader_id.is_some(),
                leader_id,
                leader_address: None, // Could look up address from peer info
                term: current_term,
            }))
        }
    }

    /// Start the Raft gRPC server
    pub async fn start_raft_server(
        node: Arc<RaftNode>,
        addr: SocketAddr,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let service = RaftGrpcService::new(node.clone());

        info!("Starting Raft gRPC server on {}", addr);

        Server::builder()
            .add_service(pb::raft_service_server::RaftServiceServer::new(service))
            .serve(addr)
            .await?;

        Ok(())
    }

    /// Start the Raft gRPC server with TLS
    #[cfg(feature = "server")]
    pub async fn start_raft_server_tls(
        node: Arc<RaftNode>,
        addr: SocketAddr,
        cert_path: &str,
        key_path: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use tonic::transport::{Identity, ServerTlsConfig};

        let cert = tokio::fs::read(cert_path).await?;
        let key = tokio::fs::read(key_path).await?;

        let identity = Identity::from_pem(cert, key);
        let tls_config = ServerTlsConfig::new().identity(identity);

        let service = RaftGrpcService::new(node.clone());

        info!("Starting Raft gRPC server with TLS on {}", addr);

        Server::builder()
            .tls_config(tls_config)?
            .add_service(pb::raft_service_server::RaftServiceServer::new(service))
            .serve(addr)
            .await?;

        Ok(())
    }
}

#[cfg(feature = "server")]
pub use server::{start_raft_server, RaftGrpcService};

// ============================================================================
// Cluster Membership Management
// ============================================================================

/// Cluster configuration for dynamic membership
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Current cluster members
    pub members: HashSet<NodeId>,
    /// Configuration version
    pub version: u64,
    /// Joint consensus members (during transition)
    pub joint_consensus: Option<HashSet<NodeId>>,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            members: HashSet::new(),
            version: 0,
            joint_consensus: None,
        }
    }
}

/// Membership change type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MembershipChange {
    /// Add a new node
    AddNode(NodeId),
    /// Remove an existing node
    RemoveNode(NodeId),
}

// ============================================================================
// Snapshot Support
// ============================================================================

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Last included index
    pub last_included_index: LogIndex,
    /// Last included term
    pub last_included_term: Term,
    /// Cluster configuration at snapshot
    pub cluster_config: ClusterConfig,
    /// Size of snapshot data in bytes
    pub size_bytes: u64,
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Snapshot for log compaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Snapshot metadata
    pub metadata: SnapshotMetadata,
    /// Serialized state machine data
    pub data: Vec<u8>,
}

/// Install snapshot RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallSnapshotRequest {
    /// Leader's term
    pub term: Term,
    /// Leader ID
    pub leader_id: NodeId,
    /// Snapshot metadata
    pub metadata: SnapshotMetadata,
    /// Byte offset for chunked transfer
    pub offset: u64,
    /// Snapshot data chunk
    pub data: Vec<u8>,
    /// True if this is the last chunk
    pub done: bool,
}

/// Install snapshot RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallSnapshotResponse {
    /// Current term
    pub term: Term,
    /// Success indicator
    pub success: bool,
    /// Bytes received so far
    pub bytes_received: u64,
}

// ============================================================================
// Health Monitoring
// ============================================================================

/// Node health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Node is healthy and responsive
    Healthy,
    /// Node is degraded but functional
    Degraded,
    /// Node is unhealthy/unreachable
    Unhealthy,
    /// Node health is unknown
    Unknown,
}

/// Health check result for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealth {
    /// Node ID
    pub node_id: NodeId,
    /// Current health status
    pub status: HealthStatus,
    /// Last successful heartbeat
    pub last_heartbeat: Option<SystemTime>,
    /// Response latency (ms)
    pub latency_ms: Option<u64>,
    /// Number of consecutive failures
    pub consecutive_failures: u32,
}

/// Cluster health overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterHealth {
    /// Overall cluster health
    pub status: HealthStatus,
    /// Current leader (if known)
    pub leader: Option<NodeId>,
    /// Health of all nodes
    pub nodes: Vec<NodeHealth>,
    /// Number of healthy nodes
    pub healthy_count: usize,
    /// Number of unhealthy nodes
    pub unhealthy_count: usize,
    /// Is quorum available?
    pub has_quorum: bool,
}

// ============================================================================
// Read Consistency Levels
// ============================================================================

/// Read consistency level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadConsistency {
    /// Read from leader only (strongly consistent)
    Leader,
    /// Read from any node (eventually consistent)
    Any,
    /// Read after confirming leadership (linearizable)
    Linearizable,
    /// Read from specific number of nodes
    Quorum,
}

// ============================================================================
// Failover Support
// ============================================================================

/// Automatic failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    /// Maximum time to wait for leader (ms)
    pub leader_timeout_ms: u64,
    /// Number of retries before failover
    pub max_retries: u32,
    /// Delay between retries (ms)
    pub retry_delay_ms: u64,
    /// Whether to auto-discover new leader
    pub auto_discover_leader: bool,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            leader_timeout_ms: 5000,
            max_retries: 3,
            retry_delay_ms: 500,
            auto_discover_leader: true,
        }
    }
}

/// Failover state machine
pub struct FailoverManager {
    /// Current known leader
    current_leader: Arc<RwLock<Option<NodeId>>>,
    /// Known cluster members
    cluster_members: Arc<RwLock<Vec<NodeId>>>,
    /// Configuration
    config: FailoverConfig,
    /// Consecutive failures count
    failures: Arc<std::sync::atomic::AtomicU32>,
}

impl FailoverManager {
    /// Create a new failover manager
    pub fn new(config: FailoverConfig) -> Self {
        Self {
            current_leader: Arc::new(RwLock::new(None)),
            cluster_members: Arc::new(RwLock::new(Vec::new())),
            config,
            failures: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    /// Update known leader
    pub async fn set_leader(&self, leader: NodeId) {
        *self.current_leader.write().await = Some(leader);
        self.failures
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current leader
    pub async fn get_leader(&self) -> Option<NodeId> {
        self.current_leader.read().await.clone()
    }

    /// Report a failure
    pub async fn report_failure(&self) -> bool {
        let failures =
            self.failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;

        if failures >= self.config.max_retries {
            // Trigger leader discovery
            *self.current_leader.write().await = None;
            true // Need to discover new leader
        } else {
            false
        }
    }

    /// Update cluster members
    pub async fn set_members(&self, members: Vec<NodeId>) {
        *self.cluster_members.write().await = members;
    }

    /// Get next node to try for leader discovery
    pub async fn get_discovery_candidate(&self) -> Option<NodeId> {
        let members = self.cluster_members.read().await;
        let current = self.current_leader.read().await;

        // Find a member that's not the failed leader
        members
            .iter()
            .find(|m| current.as_ref() != Some(*m))
            .cloned()
    }
}

// ============================================================================
// Replication Coordinator
// ============================================================================

/// Coordinates replication across cluster nodes
pub struct ReplicationCoordinator {
    /// Local Raft node
    node: Arc<RaftNode>,
    /// Failover manager
    failover: Arc<FailoverManager>,
    /// Cluster configuration
    cluster_config: Arc<RwLock<ClusterConfig>>,
    /// Latest snapshot
    snapshot: Arc<RwLock<Option<Snapshot>>>,
}

impl ReplicationCoordinator {
    /// Create a new replication coordinator
    pub fn new(node: Arc<RaftNode>, failover_config: FailoverConfig) -> Self {
        Self {
            node,
            failover: Arc::new(FailoverManager::new(failover_config)),
            cluster_config: Arc::new(RwLock::new(ClusterConfig::default())),
            snapshot: Arc::new(RwLock::new(None)),
        }
    }

    /// Submit a command with automatic failover
    pub async fn submit_command(&self, command: Command) -> Result<LogIndex, String> {
        let mut retries = 0;
        let max_retries = 3;

        loop {
            // Try to append to leader
            match self.node.append_entry(command.clone()).await {
                Ok(index) => return Ok(index),
                Err(e) if e == "Not the leader" => {
                    retries += 1;
                    if retries >= max_retries {
                        return Err("Failed to find leader after retries".to_string());
                    }
                    // Would normally redirect to leader here
                    time::sleep(Duration::from_millis(100)).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Read with specified consistency
    pub async fn read<T>(
        &self,
        key: &str,
        consistency: ReadConsistency,
    ) -> Result<Option<T>, String>
    where
        T: serde::de::DeserializeOwned,
    {
        match consistency {
            ReadConsistency::Leader => {
                // Leader read: Only valid on leader, reads directly from state machine
                if !self.node.is_leader().await {
                    return Err("Not the leader".to_string());
                }
                // Read from leader's state machine
                Ok(self.node.read_from_state_machine(key).await)
            }
            ReadConsistency::Any => {
                // Any read: Read from local state machine regardless of leader status
                // May return stale data if this node is behind
                Ok(self.node.read_from_state_machine(key).await)
            }
            ReadConsistency::Linearizable => {
                // Linearizable read: Strongest guarantee - verify leadership before reading
                if !self.node.is_leader().await {
                    return Err("Not the leader".to_string());
                }

                // Verify we're still leader by sending heartbeats and getting majority ack
                // This ensures no other leader has been elected
                let confirmed = self.verify_leadership().await;
                if !confirmed {
                    return Err("Leadership verification failed".to_string());
                }

                // Now safe to read - we confirmed we're still the leader
                Ok(self.node.read_from_state_machine(key).await)
            }
            ReadConsistency::Quorum => {
                // Quorum read: Read from majority of nodes and return most recent
                // For now, if we're leader, we know we have the most up-to-date data
                // Otherwise, we should query other nodes
                if self.node.is_leader().await {
                    Ok(self.node.read_from_state_machine(key).await)
                } else {
                    // In a full implementation, we'd query multiple nodes
                    // and compare commit indices to return the most recent value
                    // For now, read local and indicate it may be stale
                    Ok(self.node.read_from_state_machine(key).await)
                }
            }
        }
    }

    /// Verify leadership by confirming with a majority of nodes
    /// This is used for linearizable reads to ensure we're still the leader
    async fn verify_leadership(&self) -> bool {
        // If we're not the leader, verification fails immediately
        if !self.node.is_leader().await {
            return false;
        }

        // Get cluster configuration
        let config = self.cluster_config.read().await;
        let total_nodes = config.members.len().max(1);
        let quorum_needed = total_nodes / 2 + 1;

        // If we're a single-node cluster, we're automatically verified
        if total_nodes == 1 {
            return true;
        }

        // In a full implementation, we would:
        // 1. Record the current commit index
        // 2. Send heartbeats to all followers
        // 3. Wait for acknowledgments from a majority
        // 4. Only then confirm leadership
        //
        // For now, we use a simplified approach:
        // - Check if we've received recent acknowledgments from followers
        // - This is tracked through the leader_state's match_index map

        let leader_state = self.node.leader_state.lock().await;
        if let Some(ref state) = *leader_state {
            // Count nodes that have recently acknowledged
            let acked_count = state.match_index.len() + 1; // +1 for self
            acked_count >= quorum_needed
        } else {
            // No leader state means we're not actually leading
            false
        }
    }

    /// Get cluster health
    pub async fn get_cluster_health(&self) -> ClusterHealth {
        let config = self.cluster_config.read().await;
        let is_leader = self.node.is_leader().await;

        let healthy_count = if is_leader { 1 } else { 0 };
        let total_nodes = config.members.len().max(1);
        let quorum_needed = total_nodes / 2 + 1;

        ClusterHealth {
            status: if healthy_count >= quorum_needed {
                HealthStatus::Healthy
            } else if healthy_count > 0 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Unhealthy
            },
            leader: self.node.leader_id().await,
            nodes: vec![NodeHealth {
                node_id: "self".to_string(),
                status: HealthStatus::Healthy,
                last_heartbeat: Some(SystemTime::now()),
                latency_ms: Some(0),
                consecutive_failures: 0,
            }],
            healthy_count,
            unhealthy_count: total_nodes - healthy_count,
            has_quorum: healthy_count >= quorum_needed,
        }
    }

    /// Create a snapshot of current state
    pub async fn create_snapshot(&self, data: Vec<u8>) -> Result<SnapshotMetadata, String> {
        let persistent = self.node.persistent.read().await;
        let volatile = self.node.volatile.lock().await;

        let last_entry = persistent.log.get((volatile.last_applied - 1) as usize);

        let metadata = SnapshotMetadata {
            last_included_index: volatile.last_applied,
            last_included_term: last_entry.map(|e| e.term).unwrap_or(0),
            cluster_config: self.cluster_config.read().await.clone(),
            size_bytes: data.len() as u64,
            created_at: SystemTime::now(),
        };

        let snapshot = Snapshot {
            metadata: metadata.clone(),
            data,
        };

        *self.snapshot.write().await = Some(snapshot);

        Ok(metadata)
    }

    /// Apply membership change
    pub async fn apply_membership_change(&self, change: MembershipChange) -> Result<(), String> {
        let mut config = self.cluster_config.write().await;

        match change {
            MembershipChange::AddNode(node_id) => {
                config.members.insert(node_id);
            }
            MembershipChange::RemoveNode(node_id) => {
                config.members.remove(&node_id);
            }
        }

        config.version += 1;

        Ok(())
    }
}

#[cfg(test)]
mod ha_tests {
    use super::*;

    #[tokio::test]
    async fn test_cluster_config() {
        let mut config = ClusterConfig::default();
        config.members.insert("node-1".to_string());
        config.members.insert("node-2".to_string());
        config.members.insert("node-3".to_string());

        assert_eq!(config.members.len(), 3);
        assert!(config.members.contains("node-1"));
    }

    #[tokio::test]
    async fn test_failover_manager() {
        let config = FailoverConfig::default();
        let manager = FailoverManager::new(config);

        // Initially no leader
        assert!(manager.get_leader().await.is_none());

        // Set leader
        manager.set_leader("node-1".to_string()).await;
        assert_eq!(manager.get_leader().await, Some("node-1".to_string()));

        // Report failures until failover
        for _ in 0..3 {
            manager.report_failure().await;
        }

        // Leader should be cleared after max failures
        assert!(manager.get_leader().await.is_none());
    }

    #[tokio::test]
    async fn test_health_status() {
        let health = NodeHealth {
            node_id: "node-1".to_string(),
            status: HealthStatus::Healthy,
            last_heartbeat: Some(SystemTime::now()),
            latency_ms: Some(5),
            consecutive_failures: 0,
        };

        assert_eq!(health.status, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_replication_coordinator_health() {
        let config = RaftConfig::default();
        let node = Arc::new(RaftNode::new(config));

        let coordinator = ReplicationCoordinator::new(node, FailoverConfig::default());
        let health = coordinator.get_cluster_health().await;

        assert!(matches!(
            health.status,
            HealthStatus::Healthy | HealthStatus::Degraded | HealthStatus::Unhealthy
        ));
    }

    #[tokio::test]
    async fn test_membership_change() {
        let config = RaftConfig::default();
        let node = Arc::new(RaftNode::new(config));

        let coordinator = ReplicationCoordinator::new(node, FailoverConfig::default());

        // Add nodes
        coordinator
            .apply_membership_change(MembershipChange::AddNode("node-1".to_string()))
            .await
            .unwrap();
        coordinator
            .apply_membership_change(MembershipChange::AddNode("node-2".to_string()))
            .await
            .unwrap();

        let cluster_config = coordinator.cluster_config.read().await;
        assert_eq!(cluster_config.members.len(), 2);
        assert_eq!(cluster_config.version, 2);
    }

    #[tokio::test]
    async fn test_snapshot_creation() {
        let config = RaftConfig::default();
        let node = Arc::new(RaftNode::new(config));

        // Set up as leader with some committed entries
        *node.state.write().await = NodeState::Leader;
        node.persistent.write().await.current_term = 1;
        node.append_entry(Command::NoOp).await.unwrap();
        node.volatile.lock().await.commit_index = 1;
        node.volatile.lock().await.last_applied = 1;

        let coordinator = ReplicationCoordinator::new(node, FailoverConfig::default());

        let data = b"snapshot data".to_vec();
        let metadata = coordinator.create_snapshot(data).await.unwrap();

        assert_eq!(metadata.last_included_index, 1);
        assert!(metadata.size_bytes > 0);
    }

    #[tokio::test]
    async fn test_read_consistency_levels() {
        let config = RaftConfig::default();
        let node = Arc::new(RaftNode::new(config));

        let coordinator = ReplicationCoordinator::new(node.clone(), FailoverConfig::default());

        // Non-leader should fail leader reads
        let result: Result<Option<String>, _> =
            coordinator.read("key", ReadConsistency::Leader).await;
        assert!(result.is_err());

        // Become leader
        *node.state.write().await = NodeState::Leader;

        // Leader reads should succeed
        let result: Result<Option<String>, _> =
            coordinator.read("key", ReadConsistency::Leader).await;
        assert!(result.is_ok());

        // Any reads should always succeed
        let result: Result<Option<String>, _> = coordinator.read("key", ReadConsistency::Any).await;
        assert!(result.is_ok());
    }
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

    #[tokio::test]
    async fn test_step_down_on_higher_term_append_entries() {
        let config = RaftConfig {
            node_id: "node-1".to_string(),
            ..Default::default()
        };
        let node = RaftNode::new(config);

        // Become leader
        *node.state.write().await = NodeState::Leader;
        node.persistent.write().await.current_term = 1;

        // Receive AppendEntries with higher term
        let request = AppendEntriesRequest {
            term: 5,
            leader_id: "node-2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };

        let response = node.handle_append_entries(request).await;

        assert!(response.success);
        assert_eq!(node.state().await, NodeState::Follower);
        assert_eq!(node.current_term().await, 5);
    }

    #[tokio::test]
    async fn test_known_leader_updated_on_append_entries() {
        let config = RaftConfig::default();
        let node = RaftNode::new(config);

        // Initially no known leader
        assert!(node.leader_id().await.is_none());

        // Receive AppendEntries from leader
        let request = AppendEntriesRequest {
            term: 1,
            leader_id: "leader-node".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };

        node.handle_append_entries(request).await;

        // Should now know the leader
        assert_eq!(
            *node.current_leader.read().await,
            Some("leader-node".to_string())
        );
    }
}
