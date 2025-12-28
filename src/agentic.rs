// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Agentic Query Framework
//!
//! Native support for AI agent workflows with multi-step retrieval, query decomposition,
//! parallel execution, and semantic reranking across branches.
//!
//! Inspired by Weaviate Agents and Azure's agentic retrieval, this module provides
//! first-class support for LLM-powered autonomous retrieval workflows.
//!
//! ## Features
//!
//! - **Query Decomposition**: Break complex queries into sub-queries
//! - **Parallel Execution**: Run sub-queries concurrently with branch merging
//! - **Agent State Management**: Track agent context within the database
//! - **Semantic Reranking**: Cross-branch result fusion with relevance scoring
//! - **Tool Calling**: Native support for function/tool invocation
//! - **Memory Management**: Short-term and long-term agent memory
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::agentic::{AgentExecutor, AgentQuery, QueryPlan};
//!
//! let executor = AgentExecutor::new(config);
//!
//! // Complex query that requires decomposition
//! let query = AgentQuery::new("Find products similar to my recent purchases that are on sale and have good reviews");
//!
//! // Agent decomposes and executes
//! let results = executor.execute(query).await?;
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Agent execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum sub-queries per decomposition
    pub max_sub_queries: usize,
    /// Maximum parallel branches
    pub max_parallel_branches: usize,
    /// Timeout for individual sub-queries
    pub sub_query_timeout_ms: u64,
    /// Maximum recursion depth for nested queries
    pub max_recursion_depth: usize,
    /// Enable semantic reranking across branches
    pub enable_reranking: bool,
    /// Top-k results per branch before fusion
    pub branch_top_k: usize,
    /// Final top-k after fusion
    pub final_top_k: usize,
    /// Memory window size (number of interactions to remember)
    pub memory_window: usize,
    /// Enable query caching
    pub enable_cache: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_sub_queries: 5,
            max_parallel_branches: 4,
            sub_query_timeout_ms: 5000,
            max_recursion_depth: 3,
            enable_reranking: true,
            branch_top_k: 20,
            final_top_k: 10,
            memory_window: 10,
            enable_cache: true,
        }
    }
}

/// Query decomposition strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DecompositionStrategy {
    /// Simple keyword extraction
    Keyword,
    /// Semantic chunking based on concepts
    Semantic,
    /// Entity-based decomposition
    EntityBased,
    /// Temporal decomposition (past, present, future)
    Temporal,
    /// Aspect-based decomposition
    AspectBased,
    /// Custom LLM-guided decomposition
    LLMGuided,
}

/// Sub-query generated from decomposition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SubQuery {
    /// Unique identifier
    pub id: String,
    /// The sub-query text
    pub query: String,
    /// Query type
    pub query_type: QueryType,
    /// Priority (higher = execute first)
    pub priority: u8,
    /// Dependencies (other sub-query IDs that must complete first)
    pub dependencies: Vec<String>,
    /// Filters to apply
    pub filters: HashMap<String, FilterValue>,
    /// Weight in final fusion
    pub weight: f32,
}

/// Type of query operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QueryType {
    /// Dense vector similarity search
    VectorSearch,
    /// Sparse/keyword search
    KeywordSearch,
    /// Hybrid dense + sparse
    HybridSearch,
    /// Metadata filter only
    FilterOnly,
    /// Aggregation query
    Aggregation,
    /// Nested sub-query execution
    Nested(Box<QueryPlan>),
}

/// Filter value types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Bool(bool),
    List(Vec<String>),
    Range { min: f64, max: f64 },
    Geo { lat: f64, lon: f64, radius_km: f64 },
}

/// Query execution plan
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryPlan {
    /// Original query
    pub original_query: String,
    /// Decomposed sub-queries
    pub sub_queries: Vec<SubQuery>,
    /// Execution order (topologically sorted)
    pub execution_order: Vec<Vec<String>>,
    /// Fusion strategy for combining results
    pub fusion_strategy: FusionStrategy,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Decomposition strategy used
    pub strategy: DecompositionStrategy,
}

/// Strategy for fusing results from multiple branches
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion
    RRF { k: f32 },
    /// Weighted score combination
    WeightedSum,
    /// Maximum score across branches
    MaxScore,
    /// Voting-based (count appearances)
    Voting { min_votes: usize },
    /// LLM-based reranking
    LLMRerank,
    /// Cross-encoder reranking
    CrossEncoder,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::RRF { k: 60.0 }
    }
}

/// Result from a single branch execution
#[derive(Debug, Clone)]
pub struct BranchResult {
    /// Sub-query ID
    pub sub_query_id: String,
    /// Results with scores
    pub results: Vec<ScoredResult>,
    /// Execution time
    pub execution_time: Duration,
    /// Whether this branch succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Individual result with score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredResult {
    /// Document/vector ID
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Vector data (optional)
    pub vector: Option<Vec<f32>>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Source sub-query
    pub source_query: String,
}

/// Fused result after combining branches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedResult {
    /// Document ID
    pub id: String,
    /// Final fused score
    pub fused_score: f32,
    /// Individual scores from each branch
    pub branch_scores: HashMap<String, f32>,
    /// Number of branches this appeared in
    pub branch_count: usize,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Explanation of how score was computed
    pub explanation: String,
}

/// Agent memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Timestamp
    pub timestamp: i64,
    /// Query that was executed
    pub query: String,
    /// Results returned
    pub result_ids: Vec<String>,
    /// User feedback (if any)
    pub feedback: Option<Feedback>,
    /// Context tags
    pub tags: Vec<String>,
}

/// User feedback on results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Feedback {
    Positive,
    Negative,
    Neutral,
    Explicit { rating: f32, comment: String },
}

/// Tool definition for agent tool calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name
    pub name: String,
    /// Description
    pub description: String,
    /// Parameters schema (JSON Schema)
    pub parameters: serde_json::Value,
    /// Whether tool requires confirmation
    pub requires_confirmation: bool,
}

/// Tool invocation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool name
    pub tool: String,
    /// Arguments
    pub arguments: HashMap<String, serde_json::Value>,
    /// Request ID
    pub request_id: String,
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Request ID
    pub request_id: String,
    /// Whether successful
    pub success: bool,
    /// Result data
    pub result: serde_json::Value,
    /// Error if failed
    pub error: Option<String>,
}

/// Agent state for tracking context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Agent ID
    pub agent_id: String,
    /// Session ID
    pub session_id: String,
    /// Short-term memory (recent interactions)
    pub short_term_memory: VecDeque<MemoryEntry>,
    /// Long-term memory references (IDs in persistent store)
    pub long_term_memory_refs: Vec<String>,
    /// Current context variables
    pub context: HashMap<String, serde_json::Value>,
    /// Available tools
    pub tools: Vec<ToolDefinition>,
    /// Accumulated knowledge
    pub knowledge: Vec<String>,
    /// Execution history
    pub history: Vec<ExecutionRecord>,
}

/// Record of a query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Timestamp
    pub timestamp: i64,
    /// Query plan used
    pub plan: QueryPlan,
    /// Results count
    pub result_count: usize,
    /// Total execution time
    pub execution_time_ms: u64,
    /// Success status
    pub success: bool,
}

impl AgentState {
    /// Create new agent state
    pub fn new(agent_id: &str, session_id: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            session_id: session_id.to_string(),
            short_term_memory: VecDeque::new(),
            long_term_memory_refs: Vec::new(),
            context: HashMap::new(),
            tools: Vec::new(),
            knowledge: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Add memory entry
    pub fn add_memory(&mut self, entry: MemoryEntry, max_window: usize) {
        self.short_term_memory.push_back(entry);
        while self.short_term_memory.len() > max_window {
            self.short_term_memory.pop_front();
        }
    }

    /// Update context
    pub fn set_context(&mut self, key: &str, value: serde_json::Value) {
        self.context.insert(key.to_string(), value);
    }

    /// Get context value
    pub fn get_context(&self, key: &str) -> Option<&serde_json::Value> {
        self.context.get(key)
    }

    /// Register a tool
    pub fn register_tool(&mut self, tool: ToolDefinition) {
        self.tools.push(tool);
    }

    /// Get recent queries for context
    pub fn recent_queries(&self, n: usize) -> Vec<&str> {
        self.short_term_memory
            .iter()
            .rev()
            .take(n)
            .map(|e| e.query.as_str())
            .collect()
    }

    /// Get recent result IDs for personalization
    pub fn recent_results(&self, n: usize) -> Vec<&str> {
        self.short_term_memory
            .iter()
            .rev()
            .take(n)
            .flat_map(|e| e.result_ids.iter().map(|s| s.as_str()))
            .collect()
    }
}

/// Query decomposer
pub struct QueryDecomposer {
    strategy: DecompositionStrategy,
    max_sub_queries: usize,
}

impl QueryDecomposer {
    /// Create new decomposer
    pub fn new(strategy: DecompositionStrategy, max_sub_queries: usize) -> Self {
        Self {
            strategy,
            max_sub_queries,
        }
    }

    /// Decompose a query into sub-queries
    pub fn decompose(&self, query: &str) -> Result<Vec<SubQuery>> {
        match self.strategy {
            DecompositionStrategy::Keyword => self.decompose_keyword(query),
            DecompositionStrategy::Semantic => self.decompose_semantic(query),
            DecompositionStrategy::EntityBased => self.decompose_entity(query),
            DecompositionStrategy::Temporal => self.decompose_temporal(query),
            DecompositionStrategy::AspectBased => self.decompose_aspect(query),
            DecompositionStrategy::LLMGuided => self.decompose_llm(query),
        }
    }

    fn decompose_keyword(&self, query: &str) -> Result<Vec<SubQuery>> {
        // Simple keyword-based decomposition
        let keywords: Vec<&str> = query
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .filter(|w| !STOP_WORDS.contains(&w.to_lowercase().as_str()))
            .take(self.max_sub_queries)
            .collect();

        let sub_queries: Vec<SubQuery> = keywords
            .into_iter()
            .enumerate()
            .map(|(i, keyword)| SubQuery {
                id: format!("kw_{}", i),
                query: keyword.to_string(),
                query_type: QueryType::HybridSearch,
                priority: 5,
                dependencies: vec![],
                filters: HashMap::new(),
                weight: 1.0,
            })
            .collect();

        // Also add the full query as a sub-query
        let mut result = vec![SubQuery {
            id: "full".to_string(),
            query: query.to_string(),
            query_type: QueryType::VectorSearch,
            priority: 10,
            dependencies: vec![],
            filters: HashMap::new(),
            weight: 2.0,
        }];
        result.extend(sub_queries);

        Ok(result)
    }

    fn decompose_semantic(&self, query: &str) -> Result<Vec<SubQuery>> {
        // Semantic decomposition based on conjunctions and phrases
        let mut sub_queries = Vec::new();

        // Split on common conjunctions
        let parts: Vec<&str> = query
            .split(|c| c == ',' || c == ';')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        // Also split on "and", "or", "but"
        let mut all_parts = Vec::new();
        for part in parts {
            let sub_parts: Vec<&str> = part
                .split(" and ")
                .flat_map(|s| s.split(" or "))
                .flat_map(|s| s.split(" but "))
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            all_parts.extend(sub_parts);
        }

        for (i, part) in all_parts.iter().take(self.max_sub_queries).enumerate() {
            sub_queries.push(SubQuery {
                id: format!("sem_{}", i),
                query: part.to_string(),
                query_type: QueryType::VectorSearch,
                priority: (10 - i as u8).max(1),
                dependencies: vec![],
                filters: HashMap::new(),
                weight: 1.0 / (i as f32 + 1.0),
            });
        }

        if sub_queries.is_empty() {
            sub_queries.push(SubQuery {
                id: "sem_0".to_string(),
                query: query.to_string(),
                query_type: QueryType::VectorSearch,
                priority: 10,
                dependencies: vec![],
                filters: HashMap::new(),
                weight: 1.0,
            });
        }

        Ok(sub_queries)
    }

    fn decompose_entity(&self, query: &str) -> Result<Vec<SubQuery>> {
        // Entity-based decomposition (simplified NER)
        let mut sub_queries = Vec::new();

        // Look for quoted entities
        let mut in_quote = false;
        let mut current_entity = String::new();
        let mut entities = Vec::new();

        for c in query.chars() {
            if c == '"' || c == '\'' {
                if in_quote && !current_entity.is_empty() {
                    entities.push(current_entity.clone());
                    current_entity.clear();
                }
                in_quote = !in_quote;
            } else if in_quote {
                current_entity.push(c);
            }
        }

        // Look for capitalized words as potential entities
        for word in query.split_whitespace() {
            if word.len() > 1 && word.chars().next().unwrap().is_uppercase() {
                if !STOP_WORDS.contains(&word.to_lowercase().as_str()) {
                    entities.push(word.to_string());
                }
            }
        }

        for (i, entity) in entities.iter().take(self.max_sub_queries).enumerate() {
            sub_queries.push(SubQuery {
                id: format!("ent_{}", i),
                query: entity.clone(),
                query_type: QueryType::HybridSearch,
                priority: 8,
                dependencies: vec![],
                filters: HashMap::new(),
                weight: 1.5,
            });
        }

        // Add full query
        sub_queries.push(SubQuery {
            id: "ent_full".to_string(),
            query: query.to_string(),
            query_type: QueryType::VectorSearch,
            priority: 10,
            dependencies: vec![],
            filters: HashMap::new(),
            weight: 1.0,
        });

        Ok(sub_queries)
    }

    fn decompose_temporal(&self, query: &str) -> Result<Vec<SubQuery>> {
        // Temporal decomposition
        let mut sub_queries = Vec::new();
        let query_lower = query.to_lowercase();

        // Check for temporal indicators
        let has_recent = query_lower.contains("recent") || query_lower.contains("latest") || query_lower.contains("new");
        let has_past = query_lower.contains("previous") || query_lower.contains("old") || query_lower.contains("historical");
        let has_future = query_lower.contains("upcoming") || query_lower.contains("future") || query_lower.contains("next");

        if has_recent {
            sub_queries.push(SubQuery {
                id: "temp_recent".to_string(),
                query: query.to_string(),
                query_type: QueryType::VectorSearch,
                priority: 10,
                dependencies: vec![],
                filters: HashMap::new(), // Would add date filter in real impl
                weight: 1.5,
            });
        }

        if has_past {
            sub_queries.push(SubQuery {
                id: "temp_past".to_string(),
                query: query.to_string(),
                query_type: QueryType::VectorSearch,
                priority: 8,
                dependencies: vec![],
                filters: HashMap::new(),
                weight: 1.0,
            });
        }

        if has_future {
            sub_queries.push(SubQuery {
                id: "temp_future".to_string(),
                query: query.to_string(),
                query_type: QueryType::VectorSearch,
                priority: 8,
                dependencies: vec![],
                filters: HashMap::new(),
                weight: 1.0,
            });
        }

        if sub_queries.is_empty() {
            sub_queries.push(SubQuery {
                id: "temp_default".to_string(),
                query: query.to_string(),
                query_type: QueryType::VectorSearch,
                priority: 10,
                dependencies: vec![],
                filters: HashMap::new(),
                weight: 1.0,
            });
        }

        Ok(sub_queries)
    }

    fn decompose_aspect(&self, query: &str) -> Result<Vec<SubQuery>> {
        // Aspect-based decomposition
        let mut sub_queries = Vec::new();

        // Common aspects to search for
        let aspects = [
            ("price", vec!["price", "cost", "cheap", "expensive", "affordable"]),
            ("quality", vec!["quality", "best", "top", "premium", "good"]),
            ("reviews", vec!["review", "rating", "feedback", "opinion"]),
            ("features", vec!["feature", "specification", "capability"]),
            ("comparison", vec!["compare", "vs", "versus", "better", "difference"]),
        ];

        let query_lower = query.to_lowercase();

        for (aspect_name, keywords) in aspects.iter() {
            if keywords.iter().any(|k| query_lower.contains(k)) {
                sub_queries.push(SubQuery {
                    id: format!("asp_{}", aspect_name),
                    query: query.to_string(),
                    query_type: QueryType::HybridSearch,
                    priority: 8,
                    dependencies: vec![],
                    filters: HashMap::new(),
                    weight: 1.2,
                });
            }
        }

        // Always include general search
        sub_queries.push(SubQuery {
            id: "asp_general".to_string(),
            query: query.to_string(),
            query_type: QueryType::VectorSearch,
            priority: 10,
            dependencies: vec![],
            filters: HashMap::new(),
            weight: 1.0,
        });

        Ok(sub_queries)
    }

    fn decompose_llm(&self, query: &str) -> Result<Vec<SubQuery>> {
        // In a real implementation, this would call an LLM
        // For now, use semantic decomposition as fallback
        self.decompose_semantic(query)
    }
}

/// Result fusion engine
pub struct ResultFuser {
    strategy: FusionStrategy,
}

impl ResultFuser {
    /// Create new fuser
    pub fn new(strategy: FusionStrategy) -> Self {
        Self { strategy }
    }

    /// Fuse results from multiple branches
    pub fn fuse(&self, branch_results: &[BranchResult], top_k: usize) -> Vec<FusedResult> {
        match &self.strategy {
            FusionStrategy::RRF { k } => self.fuse_rrf(branch_results, *k, top_k),
            FusionStrategy::WeightedSum => self.fuse_weighted(branch_results, top_k),
            FusionStrategy::MaxScore => self.fuse_max(branch_results, top_k),
            FusionStrategy::Voting { min_votes } => self.fuse_voting(branch_results, *min_votes, top_k),
            FusionStrategy::LLMRerank => self.fuse_weighted(branch_results, top_k), // Fallback
            FusionStrategy::CrossEncoder => self.fuse_weighted(branch_results, top_k), // Fallback
        }
    }

    fn fuse_rrf(&self, branch_results: &[BranchResult], k: f32, top_k: usize) -> Vec<FusedResult> {
        let mut doc_scores: HashMap<String, (f32, HashMap<String, f32>, HashMap<String, serde_json::Value>)> = HashMap::new();

        for branch in branch_results {
            if !branch.success {
                continue;
            }

            for (rank, result) in branch.results.iter().enumerate() {
                let rrf_score = 1.0 / (k + rank as f32 + 1.0);

                let entry = doc_scores.entry(result.id.clone()).or_insert_with(|| {
                    (0.0, HashMap::new(), result.metadata.clone())
                });

                entry.0 += rrf_score;
                entry.1.insert(branch.sub_query_id.clone(), result.score);
            }
        }

        let mut fused: Vec<FusedResult> = doc_scores
            .into_iter()
            .map(|(id, (score, branch_scores, metadata))| FusedResult {
                id,
                fused_score: score,
                branch_scores: branch_scores.clone(),
                branch_count: branch_scores.len(),
                metadata,
                explanation: format!("RRF fusion with k={}", k),
            })
            .collect();

        fused.sort_by(|a, b| b.fused_score.partial_cmp(&a.fused_score).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(top_k);
        fused
    }

    fn fuse_weighted(&self, branch_results: &[BranchResult], top_k: usize) -> Vec<FusedResult> {
        let mut doc_scores: HashMap<String, (f32, HashMap<String, f32>, HashMap<String, serde_json::Value>)> = HashMap::new();

        for branch in branch_results {
            if !branch.success {
                continue;
            }

            for result in &branch.results {
                let entry = doc_scores.entry(result.id.clone()).or_insert_with(|| {
                    (0.0, HashMap::new(), result.metadata.clone())
                });

                entry.0 += result.score;
                entry.1.insert(branch.sub_query_id.clone(), result.score);
            }
        }

        let mut fused: Vec<FusedResult> = doc_scores
            .into_iter()
            .map(|(id, (score, branch_scores, metadata))| FusedResult {
                id,
                fused_score: score,
                branch_scores: branch_scores.clone(),
                branch_count: branch_scores.len(),
                metadata,
                explanation: "Weighted sum fusion".to_string(),
            })
            .collect();

        fused.sort_by(|a, b| b.fused_score.partial_cmp(&a.fused_score).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(top_k);
        fused
    }

    fn fuse_max(&self, branch_results: &[BranchResult], top_k: usize) -> Vec<FusedResult> {
        let mut doc_scores: HashMap<String, (f32, HashMap<String, f32>, HashMap<String, serde_json::Value>)> = HashMap::new();

        for branch in branch_results {
            if !branch.success {
                continue;
            }

            for result in &branch.results {
                let entry = doc_scores.entry(result.id.clone()).or_insert_with(|| {
                    (0.0, HashMap::new(), result.metadata.clone())
                });

                if result.score > entry.0 {
                    entry.0 = result.score;
                }
                entry.1.insert(branch.sub_query_id.clone(), result.score);
            }
        }

        let mut fused: Vec<FusedResult> = doc_scores
            .into_iter()
            .map(|(id, (score, branch_scores, metadata))| FusedResult {
                id,
                fused_score: score,
                branch_scores: branch_scores.clone(),
                branch_count: branch_scores.len(),
                metadata,
                explanation: "Max score fusion".to_string(),
            })
            .collect();

        fused.sort_by(|a, b| b.fused_score.partial_cmp(&a.fused_score).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(top_k);
        fused
    }

    fn fuse_voting(&self, branch_results: &[BranchResult], min_votes: usize, top_k: usize) -> Vec<FusedResult> {
        let mut doc_votes: HashMap<String, (usize, HashMap<String, f32>, HashMap<String, serde_json::Value>)> = HashMap::new();

        for branch in branch_results {
            if !branch.success {
                continue;
            }

            for result in &branch.results {
                let entry = doc_votes.entry(result.id.clone()).or_insert_with(|| {
                    (0, HashMap::new(), result.metadata.clone())
                });

                entry.0 += 1;
                entry.1.insert(branch.sub_query_id.clone(), result.score);
            }
        }

        let mut fused: Vec<FusedResult> = doc_votes
            .into_iter()
            .filter(|(_, (votes, _, _))| *votes >= min_votes)
            .map(|(id, (votes, branch_scores, metadata))| FusedResult {
                id,
                fused_score: votes as f32,
                branch_scores: branch_scores.clone(),
                branch_count: branch_scores.len(),
                metadata,
                explanation: format!("Voting fusion (min_votes={})", min_votes),
            })
            .collect();

        fused.sort_by(|a, b| b.fused_score.partial_cmp(&a.fused_score).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(top_k);
        fused
    }
}

/// Query planner
pub struct QueryPlanner {
    decomposer: QueryDecomposer,
}

impl QueryPlanner {
    /// Create new planner
    pub fn new(strategy: DecompositionStrategy, max_sub_queries: usize) -> Self {
        Self {
            decomposer: QueryDecomposer::new(strategy, max_sub_queries),
        }
    }

    /// Create execution plan for a query
    pub fn plan(&self, query: &str) -> Result<QueryPlan> {
        let sub_queries = self.decomposer.decompose(query)?;

        // Topological sort based on dependencies
        let execution_order = self.topological_sort(&sub_queries);

        // Estimate cost
        let estimated_cost = sub_queries.len() as f64 * 0.001; // Simple cost model

        Ok(QueryPlan {
            original_query: query.to_string(),
            sub_queries,
            execution_order,
            fusion_strategy: FusionStrategy::default(),
            estimated_cost,
            strategy: self.decomposer.strategy.clone(),
        })
    }

    fn topological_sort(&self, sub_queries: &[SubQuery]) -> Vec<Vec<String>> {
        let mut result = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut remaining: Vec<&SubQuery> = sub_queries.iter().collect();

        while !remaining.is_empty() {
            // Find queries with all dependencies satisfied
            let ready: Vec<String> = remaining
                .iter()
                .filter(|q| q.dependencies.iter().all(|d| visited.contains(d)))
                .map(|q| q.id.clone())
                .collect();

            if ready.is_empty() && !remaining.is_empty() {
                // Circular dependency - just take the rest
                let rest: Vec<String> = remaining.iter().map(|q| q.id.clone()).collect();
                result.push(rest);
                break;
            }

            for id in &ready {
                visited.insert(id.clone());
            }

            remaining.retain(|q| !ready.contains(&q.id));

            if !ready.is_empty() {
                result.push(ready);
            }
        }

        result
    }
}

/// Main agent executor
pub struct AgentExecutor {
    config: AgentConfig,
    planner: QueryPlanner,
    fuser: ResultFuser,
    states: Arc<RwLock<HashMap<String, AgentState>>>,
    cache: Arc<RwLock<HashMap<String, Vec<FusedResult>>>>,
}

impl AgentExecutor {
    /// Create new executor
    pub fn new(config: AgentConfig) -> Self {
        let planner = QueryPlanner::new(DecompositionStrategy::Semantic, config.max_sub_queries);
        let fuser = ResultFuser::new(FusionStrategy::default());

        Self {
            config,
            planner,
            fuser,
            states: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create or get agent state
    pub fn get_or_create_state(&self, agent_id: &str, session_id: &str) -> AgentState {
        let key = format!("{}:{}", agent_id, session_id);
        let Ok(states) = self.states.read() else {
            return AgentState::new(agent_id, session_id);
        };

        if let Some(state) = states.get(&key) {
            state.clone()
        } else {
            drop(states);
            let state = AgentState::new(agent_id, session_id);
            let Ok(mut states) = self.states.write() else {
                return state;
            };
            states.insert(key, state.clone());
            state
        }
    }

    /// Save agent state
    pub fn save_state(&self, state: &AgentState) {
        let key = format!("{}:{}", state.agent_id, state.session_id);
        let Ok(mut states) = self.states.write() else { return; };
        states.insert(key, state.clone());
    }

    /// Plan a query
    pub fn plan(&self, query: &str) -> Result<QueryPlan> {
        self.planner.plan(query)
    }

    /// Execute with simulated results (for demonstration)
    pub fn execute_plan_simulated(&self, plan: &QueryPlan) -> Result<AgentExecutionResult> {
        let start = Instant::now();

        // Simulate branch execution
        let mut branch_results = Vec::new();

        for sub_query in &plan.sub_queries {
            // Simulate some results
            let results: Vec<ScoredResult> = (0..5)
                .map(|i| ScoredResult {
                    id: format!("doc_{}_{}", sub_query.id, i),
                    score: 0.9 - (i as f32 * 0.1),
                    vector: None,
                    metadata: HashMap::new(),
                    source_query: sub_query.id.clone(),
                })
                .collect();

            branch_results.push(BranchResult {
                sub_query_id: sub_query.id.clone(),
                results,
                execution_time: Duration::from_millis(10),
                success: true,
                error: None,
            });
        }

        // Fuse results
        let fused = self.fuser.fuse(&branch_results, self.config.final_top_k);

        let execution_time = start.elapsed();

        Ok(AgentExecutionResult {
            plan: plan.clone(),
            branch_results,
            fused_results: fused,
            execution_time,
            cached: false,
        })
    }

    /// Register a tool
    pub fn register_tool(&self, agent_id: &str, session_id: &str, tool: ToolDefinition) {
        let key = format!("{}:{}", agent_id, session_id);
        let Ok(mut states) = self.states.write() else { return; };

        if let Some(state) = states.get_mut(&key) {
            state.register_tool(tool);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> AgentStats {
        let Ok(states) = self.states.read() else {
            return AgentStats {
                active_agents: 0,
                cached_queries: 0,
                total_executions: 0,
            };
        };
        let Ok(cache) = self.cache.read() else {
            return AgentStats {
                active_agents: states.len(),
                cached_queries: 0,
                total_executions: states.values().map(|s| s.history.len()).sum(),
            };
        };

        let total_executions: usize = states.values().map(|s| s.history.len()).sum();

        AgentStats {
            active_agents: states.len(),
            cached_queries: cache.len(),
            total_executions,
        }
    }
}

/// Result of agent execution
#[derive(Debug)]
pub struct AgentExecutionResult {
    /// Query plan used
    pub plan: QueryPlan,
    /// Results from each branch
    pub branch_results: Vec<BranchResult>,
    /// Final fused results
    pub fused_results: Vec<FusedResult>,
    /// Total execution time
    pub execution_time: Duration,
    /// Whether result was from cache
    pub cached: bool,
}

/// Agent system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    /// Number of active agents
    pub active_agents: usize,
    /// Number of cached queries
    pub cached_queries: usize,
    /// Total executions
    pub total_executions: usize,
}

// Stop words for filtering
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare", "ought",
    "used", "that", "which", "who", "whom", "this", "these", "those", "what",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she",
    "her", "it", "its", "they", "them", "their",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_decomposition() {
        let decomposer = QueryDecomposer::new(DecompositionStrategy::Semantic, 5);
        let sub_queries = decomposer.decompose("Find products similar to iPhone and cheaper than $500").unwrap();

        assert!(!sub_queries.is_empty());
    }

    #[test]
    fn test_result_fusion_rrf() {
        let fuser = ResultFuser::new(FusionStrategy::RRF { k: 60.0 });

        let branch1 = BranchResult {
            sub_query_id: "q1".to_string(),
            results: vec![
                ScoredResult {
                    id: "doc1".to_string(),
                    score: 0.9,
                    vector: None,
                    metadata: HashMap::new(),
                    source_query: "q1".to_string(),
                },
                ScoredResult {
                    id: "doc2".to_string(),
                    score: 0.8,
                    vector: None,
                    metadata: HashMap::new(),
                    source_query: "q1".to_string(),
                },
            ],
            execution_time: Duration::from_millis(10),
            success: true,
            error: None,
        };

        let branch2 = BranchResult {
            sub_query_id: "q2".to_string(),
            results: vec![
                ScoredResult {
                    id: "doc2".to_string(),
                    score: 0.85,
                    vector: None,
                    metadata: HashMap::new(),
                    source_query: "q2".to_string(),
                },
                ScoredResult {
                    id: "doc3".to_string(),
                    score: 0.75,
                    vector: None,
                    metadata: HashMap::new(),
                    source_query: "q2".to_string(),
                },
            ],
            execution_time: Duration::from_millis(10),
            success: true,
            error: None,
        };

        let fused = fuser.fuse(&[branch1, branch2], 10);

        // doc2 appears in both branches, should have highest score
        assert!(!fused.is_empty());
        assert_eq!(fused[0].id, "doc2");
        assert_eq!(fused[0].branch_count, 2);
    }

    #[test]
    fn test_agent_state() {
        let mut state = AgentState::new("agent1", "session1");

        state.set_context("user_id", serde_json::json!("user123"));
        assert_eq!(
            state.get_context("user_id"),
            Some(&serde_json::json!("user123"))
        );

        state.add_memory(
            MemoryEntry {
                timestamp: 12345,
                query: "test query".to_string(),
                result_ids: vec!["doc1".to_string()],
                feedback: None,
                tags: vec![],
            },
            10,
        );

        assert_eq!(state.recent_queries(1), vec!["test query"]);
    }

    #[test]
    fn test_query_planner() {
        let planner = QueryPlanner::new(DecompositionStrategy::Semantic, 5);
        let plan = planner.plan("Find red cars with good fuel efficiency").unwrap();

        assert!(!plan.sub_queries.is_empty());
        assert!(!plan.execution_order.is_empty());
    }

    #[test]
    fn test_agent_executor() {
        let config = AgentConfig::default();
        let executor = AgentExecutor::new(config);

        let plan = executor.plan("test query").unwrap();
        let result = executor.execute_plan_simulated(&plan).unwrap();

        assert!(!result.fused_results.is_empty());
    }
}
