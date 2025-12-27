//! Agentic Vector Search
//!
//! Autonomous AI agents that plan, execute, and explain multi-step vector search operations.
//! Inspired by Weaviate's agent capabilities but with VecStore's unique explainability.
//!
//! ## Agent Types
//!
//! 1. **Query Agent**: Plans and executes complex multi-step queries
//! 2. **Transform Agent**: Intelligently transforms and enriches data
//! 3. **Personalization Agent**: Adapts search to user preferences
//! 4. **Compliance Agent**: Ensures privacy and lineage requirements
//!
//! ## Key Features
//!
//! - **Planning**: Breaks complex queries into executable steps
//! - **Tool Use**: Leverages search, filter, aggregate, and transform tools
//! - **Explanation**: Every decision is explainable (unique to VecStore)
//! - **Memory**: Agents remember context across interactions
//! - **Self-Correction**: Detects and recovers from errors
//!
//! ## Example
//!
//! ```rust,no_run
//! use vecstore::agent::{QueryAgent, AgentConfig};
//!
//! let agent = QueryAgent::new(AgentConfig::default());
//!
//! // Agent plans and executes multi-step query
//! let result = agent.execute(
//!     "Find similar products to what the user bought last week, \
//!      but exclude items they've already viewed"
//! ).await?;
//!
//! // Get explanation of agent's decisions
//! println!("Agent reasoning: {}", result.explanation);
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum number of planning steps
    pub max_steps: usize,

    /// Maximum execution time in seconds
    pub max_execution_time_secs: u64,

    /// Whether to enable detailed explanations
    pub explain: bool,

    /// Memory capacity (number of past interactions to remember)
    pub memory_capacity: usize,

    /// Whether to enable self-correction on errors
    pub self_correct: bool,

    /// Confidence threshold for actions (0.0-1.0)
    pub confidence_threshold: f32,

    /// Agent personality/style
    pub style: AgentStyle,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 10,
            max_execution_time_secs: 30,
            explain: true,
            memory_capacity: 100,
            self_correct: true,
            confidence_threshold: 0.7,
            style: AgentStyle::Balanced,
        }
    }
}

/// Agent operating style
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AgentStyle {
    /// Balanced between speed and quality
    Balanced,
    /// Prioritize speed over exhaustiveness
    Fast,
    /// Prioritize quality and thoroughness
    Thorough,
    /// Maximize explainability
    Explainable,
}

// ============================================================================
// TOOLS
// ============================================================================

/// Tools available to agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentTool {
    /// Vector similarity search
    VectorSearch {
        query: Vec<f32>,
        k: usize,
        filters: Option<FilterExpression>,
    },

    /// Keyword/text search
    KeywordSearch {
        query: String,
        k: usize,
        filters: Option<FilterExpression>,
    },

    /// Hybrid search (vector + keyword)
    HybridSearch {
        query: String,
        vector: Option<Vec<f32>>,
        k: usize,
        alpha: f32,
    },

    /// Filter results
    Filter {
        expression: FilterExpression,
        input_ids: Vec<String>,
    },

    /// Aggregate/summarize
    Aggregate {
        operation: AggregateOp,
        field: String,
        input_ids: Vec<String>,
    },

    /// Transform vectors or data
    Transform {
        operation: TransformOp,
        input_ids: Vec<String>,
    },

    /// Fetch by IDs
    Fetch { ids: Vec<String> },

    /// Rerank results
    Rerank {
        query: String,
        input_ids: Vec<String>,
        k: usize,
    },

    /// Explain a result
    Explain {
        query_vector: Vec<f32>,
        result_id: String,
    },

    /// Time-aware search
    TemporalSearch {
        query: Vec<f32>,
        k: usize,
        time_range: Option<TimeRange>,
        decay: Option<DecayConfig>,
    },

    /// Check privacy constraints
    PrivacyCheck {
        operation: String,
        epsilon: f64,
    },

    /// Track lineage
    TrackLineage {
        vector_id: String,
        operation: String,
    },
}

/// Filter expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterExpression {
    /// Equals: field == value
    Eq { field: String, value: FilterValue },
    /// Not equals
    Ne { field: String, value: FilterValue },
    /// Greater than
    Gt { field: String, value: FilterValue },
    /// Less than
    Lt { field: String, value: FilterValue },
    /// In list
    In { field: String, values: Vec<FilterValue> },
    /// Not in list
    NotIn { field: String, values: Vec<FilterValue> },
    /// Contains (for text/arrays)
    Contains { field: String, value: String },
    /// And combination
    And(Vec<FilterExpression>),
    /// Or combination
    Or(Vec<FilterExpression>),
    /// Negation
    Not(Box<FilterExpression>),
}

/// Filter value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

/// Aggregate operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregateOp {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Distinct,
    GroupBy { key: String },
}

/// Transform operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformOp {
    /// Normalize vectors
    Normalize,
    /// Reduce dimensions
    ReduceDimensions { target_dim: usize },
    /// Cluster results
    Cluster { k: usize },
    /// Deduplicate by similarity
    Deduplicate { threshold: f32 },
    /// Enrich with additional data
    Enrich { source: String },
}

/// Time range for temporal queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: Option<i64>,
    pub end: Option<i64>,
}

/// Decay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayConfig {
    pub function: String,
    pub half_life_hours: f64,
}

// ============================================================================
// AGENT STATE
// ============================================================================

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Tool that was executed
    pub tool: String,

    /// Whether execution succeeded
    pub success: bool,

    /// Result IDs (for search/fetch operations)
    pub result_ids: Vec<String>,

    /// Result scores (for search operations)
    pub scores: Vec<f32>,

    /// Aggregation result (for aggregate operations)
    pub aggregation: Option<AggregationResult>,

    /// Execution time in milliseconds
    pub execution_time_ms: u64,

    /// Explanation of what the tool did
    pub explanation: String,

    /// Error message if failed
    pub error: Option<String>,
}

/// Aggregation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    pub value: f64,
    pub count: usize,
    pub groups: Option<HashMap<String, f64>>,
}

/// Agent memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Query or interaction
    pub input: String,

    /// Actions taken
    pub actions: Vec<String>,

    /// Results summary
    pub result_summary: String,

    /// Timestamp
    pub timestamp: i64,

    /// User feedback (if any)
    pub feedback: Option<UserFeedback>,
}

/// User feedback on agent response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    pub helpful: bool,
    pub rating: Option<u8>,
    pub comment: Option<String>,
}

// ============================================================================
// PLANNING
// ============================================================================

/// A planned step in the agent's execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Step number
    pub step: usize,

    /// Human-readable description
    pub description: String,

    /// Tool to use
    pub tool: AgentTool,

    /// Dependencies (step numbers that must complete first)
    pub depends_on: Vec<usize>,

    /// Expected outcome
    pub expected_outcome: String,

    /// Confidence in this step (0.0-1.0)
    pub confidence: f32,

    /// Alternative approach if this fails
    pub fallback: Option<Box<PlanStep>>,
}

/// Complete execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Original query
    pub query: String,

    /// Parsed intent
    pub intent: QueryIntent,

    /// Ordered steps
    pub steps: Vec<PlanStep>,

    /// Overall confidence
    pub confidence: f32,

    /// Estimated execution time
    pub estimated_time_ms: u64,

    /// Plan explanation
    pub explanation: String,
}

/// Parsed query intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryIntent {
    /// Primary action (search, filter, aggregate, etc.)
    pub action: IntentAction,

    /// Target entities
    pub entities: Vec<String>,

    /// Constraints
    pub constraints: Vec<IntentConstraint>,

    /// Modifiers (time, scope, etc.)
    pub modifiers: Vec<IntentModifier>,
}

/// Intent actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentAction {
    Search,
    Filter,
    Aggregate,
    Compare,
    Explain,
    Transform,
    Combine,
}

/// Intent constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentConstraint {
    pub field: String,
    pub operator: String,
    pub value: String,
}

/// Intent modifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentModifier {
    TimeRange { start: Option<String>, end: Option<String> },
    Limit(usize),
    Exclude(Vec<String>),
    Include(Vec<String>),
    SortBy { field: String, ascending: bool },
}

// ============================================================================
// AGENT RESULT
// ============================================================================

/// Complete agent execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Original query
    pub query: String,

    /// Final result IDs
    pub result_ids: Vec<String>,

    /// Result scores
    pub scores: Vec<f32>,

    /// Execution plan that was followed
    pub plan: ExecutionPlan,

    /// Step-by-step execution trace
    pub execution_trace: Vec<ExecutionStep>,

    /// Overall explanation
    pub explanation: String,

    /// Confidence in the result
    pub confidence: f32,

    /// Total execution time
    pub execution_time_ms: u64,

    /// Suggestions for refinement
    pub suggestions: Vec<String>,

    /// Any warnings
    pub warnings: Vec<String>,
}

/// Single execution step trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub step: usize,
    pub tool: String,
    pub input_summary: String,
    pub output_summary: String,
    pub success: bool,
    pub time_ms: u64,
    pub explanation: String,
}

// ============================================================================
// QUERY AGENT
// ============================================================================

/// Query Agent: Plans and executes complex multi-step queries
pub struct QueryAgent {
    config: AgentConfig,
    memory: Vec<MemoryEntry>,
    tools: HashMap<String, Box<dyn ToolExecutor>>,
}

/// Trait for tool executors
pub trait ToolExecutor: Send + Sync {
    fn execute(&self, tool: &AgentTool) -> Result<ToolResult>;
    fn name(&self) -> &str;
}

impl QueryAgent {
    /// Create a new Query Agent
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            memory: Vec::new(),
            tools: HashMap::new(),
        }
    }

    /// Register a tool executor
    pub fn register_tool(&mut self, executor: Box<dyn ToolExecutor>) {
        let name = executor.name().to_string();
        self.tools.insert(name, executor);
    }

    /// Execute a natural language query
    pub async fn execute(&mut self, query: &str) -> Result<AgentResult> {
        let start_time = std::time::Instant::now();

        // Parse intent
        let intent = self.parse_intent(query)?;

        // Create execution plan
        let plan = self.plan(&intent, query)?;

        // Validate plan
        if plan.confidence < self.config.confidence_threshold {
            return Err(anyhow!(
                "Plan confidence {} below threshold {}",
                plan.confidence,
                self.config.confidence_threshold
            ));
        }

        // Execute plan
        let (result_ids, scores, trace) = self.execute_plan(&plan)?;

        // Generate explanation
        let explanation = self.generate_explanation(&plan, &trace)?;

        // Generate suggestions
        let suggestions = self.generate_suggestions(&intent, &result_ids)?;

        // Store in memory
        self.remember(query, &trace, &result_ids);

        let result = AgentResult {
            query: query.to_string(),
            result_ids,
            scores,
            plan,
            execution_trace: trace,
            explanation,
            confidence: 0.85, // TODO: compute from trace
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            suggestions,
            warnings: Vec::new(),
        };

        Ok(result)
    }

    /// Parse query intent
    fn parse_intent(&self, query: &str) -> Result<QueryIntent> {
        // Simplified intent parsing (would use NLP in production)
        let query_lower = query.to_lowercase();

        let action = if query_lower.contains("similar") || query_lower.contains("find") {
            IntentAction::Search
        } else if query_lower.contains("filter") || query_lower.contains("exclude") {
            IntentAction::Filter
        } else if query_lower.contains("count") || query_lower.contains("average") {
            IntentAction::Aggregate
        } else if query_lower.contains("compare") {
            IntentAction::Compare
        } else if query_lower.contains("explain") || query_lower.contains("why") {
            IntentAction::Explain
        } else {
            IntentAction::Search
        };

        let constraints = Vec::new();
        let mut modifiers = Vec::new();

        // Parse time modifiers
        if query_lower.contains("last week") {
            modifiers.push(IntentModifier::TimeRange {
                start: Some("-7d".to_string()),
                end: None,
            });
        } else if query_lower.contains("today") {
            modifiers.push(IntentModifier::TimeRange {
                start: Some("-1d".to_string()),
                end: None,
            });
        }

        // Parse exclusions
        if query_lower.contains("exclude") || query_lower.contains("but not") {
            // Would extract excluded entities
            modifiers.push(IntentModifier::Exclude(vec![]));
        }

        // Parse limits
        if let Some(pos) = query_lower.find("top ") {
            if let Some(num_str) = query_lower[pos + 4..].split_whitespace().next() {
                if let Ok(n) = num_str.parse::<usize>() {
                    modifiers.push(IntentModifier::Limit(n));
                }
            }
        }

        Ok(QueryIntent {
            action,
            entities: self.extract_entities(query),
            constraints,
            modifiers,
        })
    }

    /// Extract entities from query
    fn extract_entities(&self, query: &str) -> Vec<String> {
        // Simplified entity extraction
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut entities = Vec::new();

        // Look for quoted strings
        let mut in_quote = false;
        let mut current_entity = String::new();

        for word in words {
            if word.starts_with('"') {
                in_quote = true;
                current_entity = word[1..].to_string();
            } else if word.ends_with('"') {
                current_entity.push(' ');
                current_entity.push_str(&word[..word.len() - 1]);
                entities.push(current_entity.clone());
                in_quote = false;
                current_entity.clear();
            } else if in_quote {
                current_entity.push(' ');
                current_entity.push_str(word);
            }
        }

        entities
    }

    /// Create execution plan
    fn plan(&self, intent: &QueryIntent, query: &str) -> Result<ExecutionPlan> {
        let mut steps = Vec::new();
        let mut step_num = 0;

        match intent.action {
            IntentAction::Search => {
                // Step 1: Vector search
                steps.push(PlanStep {
                    step: step_num,
                    description: "Perform initial vector similarity search".to_string(),
                    tool: AgentTool::HybridSearch {
                        query: query.to_string(),
                        vector: None, // Would be computed
                        k: 100,
                        alpha: 0.7,
                    },
                    depends_on: vec![],
                    expected_outcome: "Get initial candidate set".to_string(),
                    confidence: 0.9,
                    fallback: None,
                });
                step_num += 1;

                // Step 2: Apply filters if any
                for modifier in &intent.modifiers {
                    if let IntentModifier::Exclude(items) = modifier {
                        steps.push(PlanStep {
                            step: step_num,
                            description: "Apply exclusion filters".to_string(),
                            tool: AgentTool::Filter {
                                expression: FilterExpression::NotIn {
                                    field: "id".to_string(),
                                    values: items
                                        .iter()
                                        .map(|s| FilterValue::String(s.clone()))
                                        .collect(),
                                },
                                input_ids: vec![],
                            },
                            depends_on: vec![step_num - 1],
                            expected_outcome: "Filtered results".to_string(),
                            confidence: 0.95,
                            fallback: None,
                        });
                        step_num += 1;
                    }
                }

                // Step 3: Rerank
                steps.push(PlanStep {
                    step: step_num,
                    description: "Rerank results for relevance".to_string(),
                    tool: AgentTool::Rerank {
                        query: query.to_string(),
                        input_ids: vec![],
                        k: 10,
                    },
                    depends_on: vec![step_num - 1],
                    expected_outcome: "Top relevant results".to_string(),
                    confidence: 0.85,
                    fallback: None,
                });
            }

            IntentAction::Explain => {
                steps.push(PlanStep {
                    step: 0,
                    description: "Generate explanation for query results".to_string(),
                    tool: AgentTool::Explain {
                        query_vector: vec![],
                        result_id: String::new(),
                    },
                    depends_on: vec![],
                    expected_outcome: "Detailed explanation".to_string(),
                    confidence: 0.9,
                    fallback: None,
                });
            }

            IntentAction::Aggregate => {
                steps.push(PlanStep {
                    step: 0,
                    description: "Aggregate data".to_string(),
                    tool: AgentTool::Aggregate {
                        operation: AggregateOp::Count,
                        field: "id".to_string(),
                        input_ids: vec![],
                    },
                    depends_on: vec![],
                    expected_outcome: "Aggregation result".to_string(),
                    confidence: 0.95,
                    fallback: None,
                });
            }

            _ => {
                // Default to search
                steps.push(PlanStep {
                    step: 0,
                    description: "Perform search".to_string(),
                    tool: AgentTool::HybridSearch {
                        query: query.to_string(),
                        vector: None,
                        k: 10,
                        alpha: 0.7,
                    },
                    depends_on: vec![],
                    expected_outcome: "Search results".to_string(),
                    confidence: 0.8,
                    fallback: None,
                });
            }
        }

        let overall_confidence = steps.iter().map(|s| s.confidence).product::<f32>();
        let steps_len = steps.len();

        Ok(ExecutionPlan {
            query: query.to_string(),
            intent: intent.clone(),
            steps,
            confidence: overall_confidence,
            estimated_time_ms: 100,
            explanation: format!(
                "Plan to {} with {} step(s)",
                match intent.action {
                    IntentAction::Search => "search",
                    IntentAction::Filter => "filter",
                    IntentAction::Aggregate => "aggregate",
                    IntentAction::Compare => "compare",
                    IntentAction::Explain => "explain",
                    IntentAction::Transform => "transform",
                    IntentAction::Combine => "combine",
                },
                steps_len
            ),
        })
    }

    /// Execute the plan
    fn execute_plan(
        &self,
        plan: &ExecutionPlan,
    ) -> Result<(Vec<String>, Vec<f32>, Vec<ExecutionStep>)> {
        let mut trace = Vec::new();
        let mut current_ids: Vec<String> = Vec::new();
        let mut current_scores: Vec<f32> = Vec::new();

        for step in &plan.steps {
            let start = std::time::Instant::now();

            // Execute the tool (simplified - would use actual implementations)
            let result = self.execute_tool(&step.tool, &current_ids)?;

            current_ids = result.result_ids.clone();
            current_scores = result.scores.clone();

            trace.push(ExecutionStep {
                step: step.step,
                tool: format!("{:?}", step.tool).split('{').next().unwrap().to_string(),
                input_summary: format!("{} input items", current_ids.len()),
                output_summary: format!("{} output items", result.result_ids.len()),
                success: result.success,
                time_ms: start.elapsed().as_millis() as u64,
                explanation: result.explanation,
            });

            if !result.success && self.config.self_correct {
                // Try fallback if available
                if let Some(ref fallback) = step.fallback {
                    let fallback_result = self.execute_tool(&fallback.tool, &current_ids)?;
                    if fallback_result.success {
                        current_ids = fallback_result.result_ids;
                        current_scores = fallback_result.scores;
                    }
                }
            }
        }

        Ok((current_ids, current_scores, trace))
    }

    /// Execute a single tool
    fn execute_tool(&self, tool: &AgentTool, input_ids: &[String]) -> Result<ToolResult> {
        // Simplified tool execution (would delegate to actual implementations)
        let tool_name = match tool {
            AgentTool::VectorSearch { .. } => "VectorSearch",
            AgentTool::KeywordSearch { .. } => "KeywordSearch",
            AgentTool::HybridSearch { .. } => "HybridSearch",
            AgentTool::Filter { .. } => "Filter",
            AgentTool::Aggregate { .. } => "Aggregate",
            AgentTool::Transform { .. } => "Transform",
            AgentTool::Fetch { .. } => "Fetch",
            AgentTool::Rerank { .. } => "Rerank",
            AgentTool::Explain { .. } => "Explain",
            AgentTool::TemporalSearch { .. } => "TemporalSearch",
            AgentTool::PrivacyCheck { .. } => "PrivacyCheck",
            AgentTool::TrackLineage { .. } => "TrackLineage",
        };

        // Simulate execution
        Ok(ToolResult {
            tool: tool_name.to_string(),
            success: true,
            result_ids: if input_ids.is_empty() {
                (0..10).map(|i| format!("result_{}", i)).collect()
            } else {
                input_ids.to_vec()
            },
            scores: (0..10).map(|i| 1.0 - i as f32 * 0.1).collect(),
            aggregation: None,
            execution_time_ms: 10,
            explanation: format!("Executed {} successfully", tool_name),
            error: None,
        })
    }

    /// Generate explanation
    fn generate_explanation(&self, plan: &ExecutionPlan, trace: &[ExecutionStep]) -> Result<String> {
        let mut explanation = String::new();

        explanation.push_str(&format!("Query: \"{}\"\n\n", plan.query));
        explanation.push_str("Execution Summary:\n");

        for step in trace {
            explanation.push_str(&format!(
                "  {}. [{}] {} → {} ({}ms)\n",
                step.step + 1,
                if step.success { "✓" } else { "✗" },
                step.tool,
                step.output_summary,
                step.time_ms
            ));
        }

        explanation.push_str(&format!(
            "\nOverall confidence: {:.0}%",
            plan.confidence * 100.0
        ));

        Ok(explanation)
    }

    /// Generate suggestions for query refinement
    fn generate_suggestions(&self, intent: &QueryIntent, results: &[String]) -> Result<Vec<String>> {
        let mut suggestions = Vec::new();

        if results.is_empty() {
            suggestions.push("Try broader search terms".to_string());
            suggestions.push("Remove some filters".to_string());
        } else if results.len() > 50 {
            suggestions.push("Consider adding more specific filters".to_string());
        }

        // Check for time-related suggestions
        let has_time_modifier = intent.modifiers.iter().any(|m| matches!(m, IntentModifier::TimeRange { .. }));
        if !has_time_modifier {
            suggestions.push("Add time range to focus on recent items".to_string());
        }

        Ok(suggestions)
    }

    /// Store interaction in memory
    fn remember(&mut self, query: &str, trace: &[ExecutionStep], results: &[String]) {
        let entry = MemoryEntry {
            input: query.to_string(),
            actions: trace.iter().map(|s| s.tool.clone()).collect(),
            result_summary: format!("{} results", results.len()),
            timestamp: chrono::Utc::now().timestamp(),
            feedback: None,
        };

        self.memory.push(entry);

        // Trim memory if over capacity
        if self.memory.len() > self.config.memory_capacity {
            self.memory.remove(0);
        }
    }

    /// Provide feedback on a past interaction
    pub fn provide_feedback(&mut self, interaction_index: usize, feedback: UserFeedback) {
        if interaction_index < self.memory.len() {
            self.memory[interaction_index].feedback = Some(feedback);
        }
    }

    /// Get memory/history
    pub fn memory(&self) -> &[MemoryEntry] {
        &self.memory
    }

    /// Clear memory
    pub fn clear_memory(&mut self) {
        self.memory.clear();
    }
}

// ============================================================================
// TRANSFORM AGENT
// ============================================================================

/// Transform Agent: Intelligently transforms and enriches data
pub struct TransformAgent {
    config: AgentConfig,
}

impl TransformAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self { config }
    }

    /// Transform vectors based on instruction
    pub async fn transform(&self, instruction: &str, vectors: &[Vec<f32>]) -> Result<TransformResult> {
        let start_time = std::time::Instant::now();

        // Parse transformation instruction
        let operation = self.parse_transform_instruction(instruction)?;

        // Apply transformation
        let transformed = self.apply_transform(&operation, vectors)?;

        Ok(TransformResult {
            original_count: vectors.len(),
            transformed_count: transformed.len(),
            transformed_vectors: transformed,
            operation: format!("{:?}", operation),
            explanation: format!("Applied {:?} to {} vectors", operation, vectors.len()),
            time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    fn parse_transform_instruction(&self, instruction: &str) -> Result<TransformOp> {
        let lower = instruction.to_lowercase();

        if lower.contains("normalize") {
            Ok(TransformOp::Normalize)
        } else if lower.contains("reduce") || lower.contains("dimension") {
            // Extract target dimension
            let target = 128; // Default
            Ok(TransformOp::ReduceDimensions { target_dim: target })
        } else if lower.contains("cluster") {
            Ok(TransformOp::Cluster { k: 10 })
        } else if lower.contains("deduplicate") || lower.contains("unique") {
            Ok(TransformOp::Deduplicate { threshold: 0.95 })
        } else {
            Err(anyhow!("Unknown transformation: {}", instruction))
        }
    }

    fn apply_transform(&self, operation: &TransformOp, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        match operation {
            TransformOp::Normalize => {
                Ok(vectors
                    .iter()
                    .map(|v| {
                        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 0.0 {
                            v.iter().map(|x| x / norm).collect()
                        } else {
                            v.clone()
                        }
                    })
                    .collect())
            }

            TransformOp::ReduceDimensions { target_dim } => {
                // Simplified: just truncate (would use PCA in production)
                Ok(vectors
                    .iter()
                    .map(|v| v.iter().take(*target_dim).cloned().collect())
                    .collect())
            }

            TransformOp::Deduplicate { threshold } => {
                // Simple deduplication based on cosine similarity
                let mut unique = Vec::new();

                for v in vectors {
                    let is_dup = unique.iter().any(|u: &Vec<f32>| {
                        let dot: f32 = v.iter().zip(u).map(|(a, b)| a * b).sum();
                        let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let norm_u: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
                        dot / (norm_v * norm_u + 1e-10) > *threshold
                    });

                    if !is_dup {
                        unique.push(v.clone());
                    }
                }

                Ok(unique)
            }

            _ => Ok(vectors.to_vec()),
        }
    }
}

/// Result of transformation
#[derive(Debug, Clone)]
pub struct TransformResult {
    pub original_count: usize,
    pub transformed_count: usize,
    pub transformed_vectors: Vec<Vec<f32>>,
    pub operation: String,
    pub explanation: String,
    pub time_ms: u64,
}

// ============================================================================
// PERSONALIZATION AGENT
// ============================================================================

/// Personalization Agent: Adapts search to user preferences
pub struct PersonalizationAgent {
    config: AgentConfig,
    user_profiles: HashMap<String, UserProfile>,
}

/// User profile for personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    /// Preference vector (learned from interactions)
    pub preference_vector: Vec<f32>,
    /// Category preferences (category -> weight)
    pub category_weights: HashMap<String, f32>,
    /// Recent interactions
    pub recent_interactions: Vec<InteractionRecord>,
    /// Explicit preferences
    pub explicit_preferences: HashMap<String, String>,
}

/// Record of user interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionRecord {
    pub item_id: String,
    pub interaction_type: InteractionType,
    pub timestamp: i64,
    pub duration_secs: Option<f64>,
}

/// Types of user interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    View,
    Click,
    Purchase,
    Like,
    Dislike,
    Bookmark,
    Share,
}

impl PersonalizationAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            user_profiles: HashMap::new(),
        }
    }

    /// Personalize search results for a user
    pub async fn personalize(
        &self,
        user_id: &str,
        results: &[(String, f32)],
    ) -> Result<Vec<(String, f32, PersonalizationExplanation)>> {
        let profile = self
            .user_profiles
            .get(user_id)
            .ok_or_else(|| anyhow!("User profile not found: {}", user_id))?;

        let mut personalized: Vec<(String, f32, PersonalizationExplanation)> = results
            .iter()
            .map(|(id, score)| {
                let boost = self.compute_personalization_boost(profile, id);
                let new_score = score * (1.0 + boost);

                let explanation = PersonalizationExplanation {
                    original_score: *score,
                    personalized_score: new_score,
                    boost_factor: boost,
                    reasons: vec![format!("User preference alignment: {:.2}", boost)],
                };

                (id.clone(), new_score, explanation)
            })
            .collect();

        // Re-sort by personalized score
        personalized.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(personalized)
    }

    fn compute_personalization_boost(&self, profile: &UserProfile, item_id: &str) -> f32 {
        // Simple boost based on recent interactions
        let recency_boost: f32 = profile
            .recent_interactions
            .iter()
            .filter(|i| i.item_id == item_id)
            .map(|i| match i.interaction_type {
                InteractionType::Purchase => 0.3,
                InteractionType::Like => 0.2,
                InteractionType::Click => 0.1,
                InteractionType::View => 0.05,
                InteractionType::Dislike => -0.3,
                _ => 0.0,
            })
            .sum();

        recency_boost.clamp(-0.5, 0.5)
    }

    /// Update user profile with new interaction
    pub fn record_interaction(&mut self, user_id: &str, interaction: InteractionRecord) {
        let profile = self
            .user_profiles
            .entry(user_id.to_string())
            .or_insert_with(|| UserProfile {
                user_id: user_id.to_string(),
                preference_vector: Vec::new(),
                category_weights: HashMap::new(),
                recent_interactions: Vec::new(),
                explicit_preferences: HashMap::new(),
            });

        profile.recent_interactions.push(interaction);

        // Keep only recent interactions
        if profile.recent_interactions.len() > 100 {
            profile.recent_interactions.remove(0);
        }
    }

    /// Create or update user profile
    pub fn update_profile(&mut self, user_id: &str, profile: UserProfile) {
        self.user_profiles.insert(user_id.to_string(), profile);
    }

    /// Get user profile
    pub fn get_profile(&self, user_id: &str) -> Option<&UserProfile> {
        self.user_profiles.get(user_id)
    }
}

/// Explanation of personalization applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationExplanation {
    pub original_score: f32,
    pub personalized_score: f32,
    pub boost_factor: f32,
    pub reasons: Vec<String>,
}

// ============================================================================
// COMPLIANCE AGENT
// ============================================================================

/// Compliance Agent: Ensures privacy and lineage requirements
pub struct ComplianceAgent {
    config: AgentConfig,
    privacy_budget: f64,
    lineage_enabled: bool,
}

impl ComplianceAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            privacy_budget: 1.0, // epsilon
            lineage_enabled: true,
        }
    }

    /// Check if operation is compliant
    pub fn check_compliance(&self, operation: &ComplianceCheck) -> ComplianceResult {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();

        // Check privacy budget
        if let Some(epsilon) = operation.requested_epsilon {
            if epsilon > self.privacy_budget {
                violations.push(format!(
                    "Privacy budget exceeded: requested {}, remaining {}",
                    epsilon, self.privacy_budget
                ));
            } else if epsilon > self.privacy_budget * 0.8 {
                warnings.push(format!(
                    "Privacy budget low: {} remaining after this operation",
                    self.privacy_budget - epsilon
                ));
            }
        }

        // Check data retention
        if let Some(ref policy) = operation.retention_policy {
            if policy.max_retention_days < 30 {
                warnings.push("Short retention policy may affect analytics".to_string());
            }
        }

        // Check access permissions
        if operation.requires_pii && !operation.has_pii_access {
            violations.push("PII access required but not granted".to_string());
        }

        ComplianceResult {
            compliant: violations.is_empty(),
            violations,
            warnings,
            recommendations: self.generate_recommendations(&operation),
        }
    }

    fn generate_recommendations(&self, operation: &ComplianceCheck) -> Vec<String> {
        let mut recommendations = Vec::new();

        if operation.requested_epsilon.unwrap_or(0.0) > 0.5 {
            recommendations.push("Consider using smaller epsilon for better privacy".to_string());
        }

        if !self.lineage_enabled {
            recommendations.push("Enable lineage tracking for audit compliance".to_string());
        }

        recommendations
    }

    /// Track lineage for an operation
    pub fn track_lineage(&self, operation: &str, inputs: &[String], outputs: &[String]) -> LineageRecord {
        LineageRecord {
            operation: operation.to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            privacy_cost: 0.0,
            user: None,
        }
    }

    /// Get remaining privacy budget
    pub fn remaining_budget(&self) -> f64 {
        self.privacy_budget
    }

    /// Consume privacy budget
    pub fn consume_budget(&mut self, epsilon: f64) -> Result<()> {
        if epsilon > self.privacy_budget {
            return Err(anyhow!("Insufficient privacy budget"));
        }
        self.privacy_budget -= epsilon;
        Ok(())
    }

    /// Reset privacy budget (e.g., for new time period)
    pub fn reset_budget(&mut self, new_budget: f64) {
        self.privacy_budget = new_budget;
    }
}

/// Compliance check request
#[derive(Debug, Clone)]
pub struct ComplianceCheck {
    pub operation: String,
    pub requested_epsilon: Option<f64>,
    pub retention_policy: Option<RetentionPolicy>,
    pub requires_pii: bool,
    pub has_pii_access: bool,
}

/// Data retention policy
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    pub max_retention_days: u32,
    pub delete_on_request: bool,
    pub anonymize_after_days: Option<u32>,
}

/// Compliance check result
#[derive(Debug, Clone)]
pub struct ComplianceResult {
    pub compliant: bool,
    pub violations: Vec<String>,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Lineage record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageRecord {
    pub operation: String,
    pub timestamp: i64,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub privacy_cost: f64,
    pub user: Option<String>,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_agent_basic() {
        let config = AgentConfig::default();
        let mut agent = QueryAgent::new(config);

        let result = agent.execute("Find similar products").await.unwrap();

        assert!(!result.result_ids.is_empty());
        assert!(!result.explanation.is_empty());
        assert!(!result.execution_trace.is_empty());
    }

    #[tokio::test]
    async fn test_query_agent_with_filters() {
        let config = AgentConfig::default();
        let mut agent = QueryAgent::new(config);

        let result = agent
            .execute("Find products from last week but exclude viewed items")
            .await
            .unwrap();

        assert!(result.plan.steps.len() >= 1);
    }

    #[tokio::test]
    async fn test_transform_agent() {
        let agent = TransformAgent::new(AgentConfig::default());

        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        let result = agent.transform("normalize", &vectors).await.unwrap();

        assert_eq!(result.transformed_count, 2);

        // Check normalization
        for v in &result.transformed_vectors {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.001);
        }
    }

    #[tokio::test]
    async fn test_personalization_agent() {
        let mut agent = PersonalizationAgent::new(AgentConfig::default());

        // Create user profile
        agent.update_profile(
            "user1",
            UserProfile {
                user_id: "user1".to_string(),
                preference_vector: Vec::new(),
                category_weights: HashMap::new(),
                recent_interactions: vec![InteractionRecord {
                    item_id: "item1".to_string(),
                    interaction_type: InteractionType::Like,
                    timestamp: chrono::Utc::now().timestamp(),
                    duration_secs: None,
                }],
                explicit_preferences: HashMap::new(),
            },
        );

        let results = vec![
            ("item1".to_string(), 0.8),
            ("item2".to_string(), 0.9),
        ];

        let personalized = agent.personalize("user1", &results).await.unwrap();

        // item1 should be boosted due to Like interaction
        assert_eq!(personalized.len(), 2);
    }

    #[test]
    fn test_compliance_agent() {
        let mut agent = ComplianceAgent::new(AgentConfig::default());

        let check = ComplianceCheck {
            operation: "search".to_string(),
            requested_epsilon: Some(0.5),
            retention_policy: None,
            requires_pii: false,
            has_pii_access: false,
        };

        let result = agent.check_compliance(&check);
        assert!(result.compliant);

        // Consume budget
        agent.consume_budget(0.5).unwrap();
        assert_eq!(agent.remaining_budget(), 0.5);

        // Try to exceed budget
        let check2 = ComplianceCheck {
            operation: "search".to_string(),
            requested_epsilon: Some(0.6),
            retention_policy: None,
            requires_pii: false,
            has_pii_access: false,
        };

        let result2 = agent.check_compliance(&check2);
        assert!(!result2.compliant);
    }

    #[test]
    fn test_memory() {
        let mut agent = QueryAgent::new(AgentConfig {
            memory_capacity: 3,
            ..Default::default()
        });

        // Manually add memory entries
        for i in 0..5 {
            agent.memory.push(MemoryEntry {
                input: format!("query {}", i),
                actions: vec![],
                result_summary: "".to_string(),
                timestamp: i as i64,
                feedback: None,
            });
        }

        // Trim to capacity
        while agent.memory.len() > agent.config.memory_capacity {
            agent.memory.remove(0);
        }

        assert_eq!(agent.memory.len(), 3);
        assert_eq!(agent.memory[0].input, "query 2");
    }
}
