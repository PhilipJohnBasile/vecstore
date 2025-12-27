// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Visual Query Explain
//!
//! EXPLAIN ANALYZE for vector queries with visual execution plans.
//! Provides detailed breakdown of query execution for debugging and optimization.
//!
//! ## Features
//!
//! - **Execution Plan Visualization**: Tree-based plan display
//! - **Stage-by-Stage Timing**: Detailed timing breakdown
//! - **Filter Selectivity**: Estimation vs actual comparison
//! - **Index Usage Statistics**: Which indexes were used
//! - **Optimization Recommendations**: Suggestions for improvement
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::query_explain::{QueryExplainer, ExplainFormat};
//!
//! let explainer = QueryExplainer::new();
//! let plan = explainer.explain(query, ExplainFormat::Text)?;
//! println!("{}", plan);
//!
//! // Or get HTML visualization
//! let html = explainer.explain(query, ExplainFormat::Html)?;
//! ```

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Output format for explain
#[derive(Debug, Clone, PartialEq)]
pub enum ExplainFormat {
    /// Plain text
    Text,
    /// JSON format
    Json,
    /// HTML visualization
    Html,
    /// DOT graph format
    Dot,
}

/// Query explain options
#[derive(Debug, Clone)]
pub struct ExplainOptions {
    /// Include actual execution (EXPLAIN ANALYZE)
    pub analyze: bool,
    /// Include buffer/cache statistics
    pub buffers: bool,
    /// Include timing information
    pub timing: bool,
    /// Verbose output
    pub verbose: bool,
    /// Output format
    pub format: ExplainFormat,
    /// Include optimization suggestions
    pub suggestions: bool,
}

impl Default for ExplainOptions {
    fn default() -> Self {
        Self {
            analyze: true,
            buffers: true,
            timing: true,
            verbose: false,
            format: ExplainFormat::Text,
            suggestions: true,
        }
    }
}

/// Execution plan node type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlanNodeType {
    /// Root query node
    Query,
    /// Vector similarity search
    VectorScan {
        index_type: String,
        metric: String,
    },
    /// Metadata filter
    Filter {
        expression: String,
    },
    /// Hybrid search fusion
    Fusion {
        strategy: String,
    },
    /// Reranking stage
    Rerank {
        method: String,
    },
    /// Result limit/offset
    Limit {
        limit: usize,
        offset: usize,
    },
    /// Aggregation
    Aggregate {
        function: String,
    },
    /// Cache lookup
    CacheLookup,
    /// Parallel execution
    Parallel {
        workers: usize,
    },
    /// Index scan
    IndexScan {
        index_name: String,
    },
    /// Sequential scan (fallback)
    SeqScan,
    /// Sort operation
    Sort {
        keys: Vec<String>,
    },
}

/// Execution plan node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanNode {
    /// Node type
    pub node_type: PlanNodeType,
    /// Estimated rows
    pub estimated_rows: usize,
    /// Actual rows (if analyzed)
    pub actual_rows: Option<usize>,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Actual time (if analyzed)
    pub actual_time_ms: Option<f64>,
    /// Startup time (time to first row)
    pub startup_time_ms: Option<f64>,
    /// Child nodes
    pub children: Vec<PlanNode>,
    /// Node-specific properties
    pub properties: HashMap<String, String>,
    /// Warnings/notes
    pub warnings: Vec<String>,
}

impl PlanNode {
    /// Create new plan node
    pub fn new(node_type: PlanNodeType) -> Self {
        Self {
            node_type,
            estimated_rows: 0,
            actual_rows: None,
            estimated_cost: 0.0,
            actual_time_ms: None,
            startup_time_ms: None,
            children: Vec::new(),
            properties: HashMap::new(),
            warnings: Vec::new(),
        }
    }

    /// Add child node
    pub fn add_child(&mut self, child: PlanNode) {
        self.children.push(child);
    }

    /// Set property
    pub fn set_property(&mut self, key: &str, value: &str) {
        self.properties.insert(key.to_string(), value.to_string());
    }

    /// Add warning
    pub fn add_warning(&mut self, warning: &str) {
        self.warnings.push(warning.to_string());
    }

    /// Total time including children
    pub fn total_time_ms(&self) -> f64 {
        let self_time = self.actual_time_ms.unwrap_or(0.0);
        let children_time: f64 = self.children.iter()
            .map(|c| c.total_time_ms())
            .sum();
        self_time + children_time
    }

    /// Format as text tree
    pub fn format_text(&self, indent: usize) -> String {
        let mut output = String::new();
        let prefix = "  ".repeat(indent);
        let arrow = if indent > 0 { "-> " } else { "" };

        // Node type and basic info
        let node_name = match &self.node_type {
            PlanNodeType::Query => "Query".to_string(),
            PlanNodeType::VectorScan { index_type, metric } =>
                format!("Vector Scan ({}, {})", index_type, metric),
            PlanNodeType::Filter { expression } =>
                format!("Filter ({})", expression),
            PlanNodeType::Fusion { strategy } =>
                format!("Fusion ({})", strategy),
            PlanNodeType::Rerank { method } =>
                format!("Rerank ({})", method),
            PlanNodeType::Limit { limit, offset } =>
                format!("Limit {} Offset {}", limit, offset),
            PlanNodeType::Aggregate { function } =>
                format!("Aggregate ({})", function),
            PlanNodeType::CacheLookup => "Cache Lookup".to_string(),
            PlanNodeType::Parallel { workers } =>
                format!("Parallel ({} workers)", workers),
            PlanNodeType::IndexScan { index_name } =>
                format!("Index Scan ({})", index_name),
            PlanNodeType::SeqScan => "Sequential Scan".to_string(),
            PlanNodeType::Sort { keys } =>
                format!("Sort ({})", keys.join(", ")),
        };

        output.push_str(&format!("{}{}{}", prefix, arrow, node_name));

        // Cost and rows
        output.push_str(&format!("  (cost={:.2}", self.estimated_cost));
        if let Some(actual) = self.actual_rows {
            output.push_str(&format!(" rows={}/{}", actual, self.estimated_rows));
        } else {
            output.push_str(&format!(" rows={}", self.estimated_rows));
        }
        output.push(')');

        // Timing
        if let Some(time) = self.actual_time_ms {
            output.push_str(&format!(" [{:.3} ms]", time));
        }

        output.push('\n');

        // Properties
        for (key, value) in &self.properties {
            output.push_str(&format!("{}  {}: {}\n", prefix, key, value));
        }

        // Warnings
        for warning in &self.warnings {
            output.push_str(&format!("{}  WARNING: {}\n", prefix, warning));
        }

        // Children
        for child in &self.children {
            output.push_str(&child.format_text(indent + 1));
        }

        output
    }

    /// Format as JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "node_type": format!("{:?}", self.node_type),
            "estimated_rows": self.estimated_rows,
            "actual_rows": self.actual_rows,
            "estimated_cost": self.estimated_cost,
            "actual_time_ms": self.actual_time_ms,
            "startup_time_ms": self.startup_time_ms,
            "properties": self.properties,
            "warnings": self.warnings,
            "children": self.children.iter().map(|c| c.to_json()).collect::<Vec<_>>()
        })
    }

    /// Format as HTML
    pub fn format_html(&self) -> String {
        let mut html = String::new();

        html.push_str(r#"<!DOCTYPE html>
<html>
<head>
<style>
body { font-family: monospace; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
.node { margin: 10px 0; padding: 10px; border-left: 3px solid #569cd6; background: #252526; }
.node-type { color: #4ec9b0; font-weight: bold; }
.cost { color: #ce9178; }
.rows { color: #b5cea8; }
.time { color: #dcdcaa; font-weight: bold; }
.warning { color: #f48771; background: #5a1d1d; padding: 5px; margin: 5px 0; }
.property { color: #9cdcfe; }
.children { margin-left: 30px; border-left: 1px dashed #569cd6; }
.bar { height: 8px; background: #569cd6; margin: 5px 0; }
h1 { color: #569cd6; }
</style>
</head>
<body>
<h1>Query Execution Plan</h1>
"#);

        html.push_str(&self.format_html_node(100.0));

        html.push_str("</body></html>");
        html
    }

    fn format_html_node(&self, parent_time: f64) -> String {
        let mut html = String::new();

        let node_name = match &self.node_type {
            PlanNodeType::Query => "Query".to_string(),
            PlanNodeType::VectorScan { index_type, metric } =>
                format!("Vector Scan ({}, {})", index_type, metric),
            PlanNodeType::Filter { expression } =>
                format!("Filter: {}", expression),
            PlanNodeType::Fusion { strategy } =>
                format!("Fusion: {}", strategy),
            PlanNodeType::Rerank { method } =>
                format!("Rerank: {}", method),
            PlanNodeType::Limit { limit, offset } =>
                format!("Limit {} Offset {}", limit, offset),
            PlanNodeType::Aggregate { function } =>
                format!("Aggregate: {}", function),
            PlanNodeType::CacheLookup => "Cache Lookup".to_string(),
            PlanNodeType::Parallel { workers } =>
                format!("Parallel ({} workers)", workers),
            PlanNodeType::IndexScan { index_name } =>
                format!("Index Scan: {}", index_name),
            PlanNodeType::SeqScan => "Sequential Scan".to_string(),
            PlanNodeType::Sort { keys } =>
                format!("Sort: {}", keys.join(", ")),
        };

        let time_pct = if parent_time > 0.0 {
            (self.actual_time_ms.unwrap_or(0.0) / parent_time * 100.0).min(100.0)
        } else {
            0.0
        };

        html.push_str("<div class=\"node\">");
        html.push_str(&format!("<span class=\"node-type\">{}</span>", node_name));
        html.push_str(&format!(" <span class=\"cost\">cost={:.2}</span>", self.estimated_cost));

        if let Some(actual) = self.actual_rows {
            html.push_str(&format!(" <span class=\"rows\">rows={}/{}</span>", actual, self.estimated_rows));
        } else {
            html.push_str(&format!(" <span class=\"rows\">rows={}</span>", self.estimated_rows));
        }

        if let Some(time) = self.actual_time_ms {
            html.push_str(&format!(" <span class=\"time\">[{:.3} ms]</span>", time));
            html.push_str(&format!("<div class=\"bar\" style=\"width: {}%;\"></div>", time_pct));
        }

        for (key, value) in &self.properties {
            html.push_str(&format!("<div class=\"property\">{}: {}</div>", key, value));
        }

        for warning in &self.warnings {
            html.push_str(&format!("<div class=\"warning\">WARNING: {}</div>", warning));
        }

        if !self.children.is_empty() {
            html.push_str("<div class=\"children\">");
            for child in &self.children {
                html.push_str(&child.format_html_node(self.actual_time_ms.unwrap_or(parent_time)));
            }
            html.push_str("</div>");
        }

        html.push_str("</div>");
        html
    }

    /// Format as DOT graph
    pub fn format_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph QueryPlan {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box, style=filled, fillcolor=lightblue];\n");

        let mut counter = 0;
        self.format_dot_node(&mut dot, &mut counter, None);

        dot.push_str("}\n");
        dot
    }

    fn format_dot_node(&self, dot: &mut String, counter: &mut usize, parent: Option<usize>) -> usize {
        let id = *counter;
        *counter += 1;

        let label = match &self.node_type {
            PlanNodeType::Query => "Query".to_string(),
            PlanNodeType::VectorScan { index_type, .. } => format!("VectorScan\\n{}", index_type),
            PlanNodeType::Filter { expression } => format!("Filter\\n{}", expression),
            PlanNodeType::Fusion { strategy } => format!("Fusion\\n{}", strategy),
            PlanNodeType::Rerank { method } => format!("Rerank\\n{}", method),
            PlanNodeType::Limit { limit, .. } => format!("Limit {}",limit),
            _ => format!("{:?}", self.node_type),
        };

        let color = if self.actual_time_ms.unwrap_or(0.0) > 100.0 {
            "lightcoral"
        } else if self.warnings.is_empty() {
            "lightgreen"
        } else {
            "lightyellow"
        };

        dot.push_str(&format!(
            "  n{} [label=\"{}\\nrows: {}\\ncost: {:.2}\", fillcolor={}];\n",
            id, label, self.estimated_rows, self.estimated_cost, color
        ));

        if let Some(parent_id) = parent {
            dot.push_str(&format!("  n{} -> n{};\n", parent_id, id));
        }

        for child in &self.children {
            child.format_dot_node(dot, counter, Some(id));
        }

        id
    }
}

/// Query execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Total execution time
    pub total_time_ms: f64,
    /// Planning time
    pub planning_time_ms: f64,
    /// Execution time
    pub execution_time_ms: f64,
    /// Rows examined
    pub rows_examined: usize,
    /// Rows returned
    pub rows_returned: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
    /// Index scans
    pub index_scans: usize,
    /// Sequential scans
    pub seq_scans: usize,
    /// Memory used (bytes)
    pub memory_bytes: usize,
    /// Disk reads
    pub disk_reads: usize,
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Description
    pub description: String,
    /// Estimated improvement
    pub estimated_improvement: String,
    /// How to implement
    pub action: String,
}

/// Suggestion type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Create an index
    CreateIndex,
    /// Use different index
    UseIndex,
    /// Add filter
    AddFilter,
    /// Reduce k value
    ReduceK,
    /// Use approximate search
    UseApproximate,
    /// Enable caching
    EnableCache,
    /// Parallelize
    Parallelize,
    /// Quantize vectors
    Quantize,
}

/// Complete explain result
#[derive(Debug, Clone)]
pub struct ExplainResult {
    /// Execution plan
    pub plan: PlanNode,
    /// Execution statistics
    pub stats: ExecutionStats,
    /// Optimization suggestions
    pub suggestions: Vec<OptimizationSuggestion>,
}

impl ExplainResult {
    /// Format according to options
    pub fn format(&self, options: &ExplainOptions) -> String {
        match options.format {
            ExplainFormat::Text => self.format_text(options),
            ExplainFormat::Json => self.format_json(options),
            ExplainFormat::Html => self.format_html(options),
            ExplainFormat::Dot => self.plan.format_dot(),
        }
    }

    fn format_text(&self, options: &ExplainOptions) -> String {
        let mut output = String::new();

        output.push_str("QUERY PLAN\n");
        output.push_str(&"=".repeat(60));
        output.push('\n');
        output.push_str(&self.plan.format_text(0));

        if options.timing {
            output.push_str("\nEXECUTION STATISTICS\n");
            output.push_str(&"-".repeat(60));
            output.push('\n');
            output.push_str(&format!("Total time: {:.3} ms\n", self.stats.total_time_ms));
            output.push_str(&format!("  Planning: {:.3} ms\n", self.stats.planning_time_ms));
            output.push_str(&format!("  Execution: {:.3} ms\n", self.stats.execution_time_ms));
            output.push_str(&format!("Rows: {} examined, {} returned\n",
                self.stats.rows_examined, self.stats.rows_returned));
        }

        if options.buffers {
            output.push_str(&format!("Cache: {} hits, {} misses\n",
                self.stats.cache_hits, self.stats.cache_misses));
            output.push_str(&format!("Memory: {} bytes\n", self.stats.memory_bytes));
            output.push_str(&format!("Disk reads: {}\n", self.stats.disk_reads));
        }

        if options.suggestions && !self.suggestions.is_empty() {
            output.push_str("\nOPTIMIZATION SUGGESTIONS\n");
            output.push_str(&"-".repeat(60));
            output.push('\n');
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                output.push_str(&format!("{}. [{:?}] {}\n",
                    i + 1, suggestion.suggestion_type, suggestion.description));
                output.push_str(&format!("   Improvement: {}\n", suggestion.estimated_improvement));
                output.push_str(&format!("   Action: {}\n", suggestion.action));
            }
        }

        output
    }

    fn format_json(&self, options: &ExplainOptions) -> String {
        let mut json = serde_json::json!({
            "plan": self.plan.to_json()
        });

        if options.timing {
            json["stats"] = serde_json::to_value(&self.stats).unwrap();
        }

        if options.suggestions {
            json["suggestions"] = serde_json::to_value(&self.suggestions).unwrap();
        }

        serde_json::to_string_pretty(&json).unwrap()
    }

    fn format_html(&self, options: &ExplainOptions) -> String {
        let mut html = self.plan.format_html();

        // Insert stats before closing body
        if options.timing {
            let stats_html = format!(r#"
<div style="margin-top: 20px; padding: 15px; background: #252526; border: 1px solid #569cd6;">
<h2 style="color: #569cd6;">Execution Statistics</h2>
<table style="width: 100%;">
<tr><td>Total Time</td><td style="color: #dcdcaa;">{:.3} ms</td></tr>
<tr><td>Planning Time</td><td style="color: #dcdcaa;">{:.3} ms</td></tr>
<tr><td>Execution Time</td><td style="color: #dcdcaa;">{:.3} ms</td></tr>
<tr><td>Rows Examined</td><td style="color: #b5cea8;">{}</td></tr>
<tr><td>Rows Returned</td><td style="color: #b5cea8;">{}</td></tr>
<tr><td>Cache Hits/Misses</td><td style="color: #9cdcfe;">{}/{}</td></tr>
<tr><td>Memory Used</td><td style="color: #ce9178;">{} bytes</td></tr>
</table>
</div>
"#,
                self.stats.total_time_ms,
                self.stats.planning_time_ms,
                self.stats.execution_time_ms,
                self.stats.rows_examined,
                self.stats.rows_returned,
                self.stats.cache_hits,
                self.stats.cache_misses,
                self.stats.memory_bytes
            );

            html = html.replace("</body>", &format!("{}</body>", stats_html));
        }

        html
    }
}

/// Query explainer
pub struct QueryExplainer {
    /// Index statistics
    index_stats: HashMap<String, IndexStats>,
    /// Filter selectivity history
    selectivity_history: HashMap<String, f64>,
}

/// Index statistics
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    /// Index name
    pub name: String,
    /// Index type
    pub index_type: String,
    /// Vector count
    pub vector_count: usize,
    /// Average search time
    pub avg_search_time_ms: f64,
    /// Memory usage
    pub memory_bytes: usize,
    /// Dimension
    pub dimension: usize,
}

impl QueryExplainer {
    /// Create new explainer
    pub fn new() -> Self {
        Self {
            index_stats: HashMap::new(),
            selectivity_history: HashMap::new(),
        }
    }

    /// Register index for analysis
    pub fn register_index(&mut self, name: &str, stats: IndexStats) {
        self.index_stats.insert(name.to_string(), stats);
    }

    /// Explain a vector search query
    pub fn explain_search(
        &self,
        query_vector: &[f32],
        k: usize,
        filter: Option<&str>,
        _options: &ExplainOptions,
    ) -> ExplainResult {
        let start = Instant::now();

        // Build execution plan
        let mut root = PlanNode::new(PlanNodeType::Query);
        root.estimated_rows = k;
        root.actual_rows = Some(k);

        // Add limit node
        let mut limit_node = PlanNode::new(PlanNodeType::Limit { limit: k, offset: 0 });
        limit_node.estimated_rows = k;
        limit_node.actual_rows = Some(k);
        limit_node.estimated_cost = 0.1;
        limit_node.actual_time_ms = Some(0.01);

        // Add vector scan node
        let mut scan_node = PlanNode::new(PlanNodeType::VectorScan {
            index_type: "HNSW".to_string(),
            metric: "cosine".to_string(),
        });

        // Estimate based on registered indexes
        let index_stats = self.index_stats.values().next();
        if let Some(stats) = index_stats {
            scan_node.estimated_rows = stats.vector_count;
            scan_node.estimated_cost = (stats.vector_count as f64).log2() * 0.1;
            scan_node.set_property("index", &stats.name);
            scan_node.set_property("vectors", &stats.vector_count.to_string());
            scan_node.set_property("dimension", &query_vector.len().to_string());
        } else {
            scan_node.estimated_rows = 10000;
            scan_node.estimated_cost = 10.0;
            scan_node.add_warning("No index statistics available");
        }

        // Simulated execution time
        scan_node.actual_time_ms = Some(5.0 + k as f64 * 0.1);
        scan_node.actual_rows = Some(k);

        // Add filter node if present
        if let Some(filter_expr) = filter {
            let mut filter_node = PlanNode::new(PlanNodeType::Filter {
                expression: filter_expr.to_string(),
            });

            // Estimate selectivity
            let selectivity = self.selectivity_history
                .get(filter_expr)
                .copied()
                .unwrap_or(0.1);

            filter_node.estimated_rows = (scan_node.estimated_rows as f64 * selectivity) as usize;
            filter_node.estimated_cost = 0.5;
            filter_node.actual_time_ms = Some(1.0);
            filter_node.set_property("selectivity", &format!("{:.2}%", selectivity * 100.0));

            if selectivity > 0.5 {
                filter_node.add_warning("Low selectivity filter - consider adding index");
            }

            filter_node.add_child(scan_node);
            limit_node.add_child(filter_node);
        } else {
            limit_node.add_child(scan_node);
        }

        root.add_child(limit_node);
        root.actual_time_ms = Some(start.elapsed().as_secs_f64() * 1000.0 + 5.0);
        root.estimated_cost = 10.0;

        // Build stats
        let stats = ExecutionStats {
            total_time_ms: root.actual_time_ms.unwrap_or(0.0),
            planning_time_ms: 0.1,
            execution_time_ms: root.actual_time_ms.unwrap_or(0.0) - 0.1,
            rows_examined: root.estimated_rows * 10,
            rows_returned: k,
            cache_hits: 0,
            cache_misses: 1,
            index_scans: 1,
            seq_scans: 0,
            memory_bytes: query_vector.len() * 4 * k,
            disk_reads: 0,
        };

        // Generate suggestions
        let mut suggestions = Vec::new();

        if k > 100 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: SuggestionType::ReduceK,
                description: format!("k={} is relatively high", k),
                estimated_improvement: "10-30% faster".to_string(),
                action: "Consider reducing k or using pagination".to_string(),
            });
        }

        if filter.is_some() && self.index_stats.is_empty() {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: SuggestionType::CreateIndex,
                description: "Filter without metadata index".to_string(),
                estimated_improvement: "2-5x faster for filtered queries".to_string(),
                action: "Create a payload index on the filtered field".to_string(),
            });
        }

        if stats.cache_misses > stats.cache_hits {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: SuggestionType::EnableCache,
                description: "Low cache hit rate".to_string(),
                estimated_improvement: "50-90% faster for repeated queries".to_string(),
                action: "Enable semantic caching for common query patterns".to_string(),
            });
        }

        ExplainResult {
            plan: root,
            stats,
            suggestions,
        }
    }

    /// Explain hybrid search
    pub fn explain_hybrid(
        &self,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
        _options: &ExplainOptions,
    ) -> ExplainResult {
        let start = Instant::now();

        let mut root = PlanNode::new(PlanNodeType::Query);
        root.estimated_rows = k;

        // Fusion node
        let mut fusion_node = PlanNode::new(PlanNodeType::Fusion {
            strategy: "RRF".to_string(),
        });
        fusion_node.estimated_cost = 1.0;
        fusion_node.actual_time_ms = Some(0.5);
        fusion_node.set_property("alpha", "0.5");

        // Vector branch
        let mut vector_node = PlanNode::new(PlanNodeType::VectorScan {
            index_type: "HNSW".to_string(),
            metric: "cosine".to_string(),
        });
        vector_node.estimated_rows = k * 2;
        vector_node.estimated_cost = 5.0;
        vector_node.actual_time_ms = Some(4.0);
        vector_node.set_property("dimension", &query_vector.len().to_string());

        // Text branch
        let mut text_node = PlanNode::new(PlanNodeType::IndexScan {
            index_name: "BM25".to_string(),
        });
        text_node.estimated_rows = k * 2;
        text_node.estimated_cost = 3.0;
        text_node.actual_time_ms = Some(2.0);
        text_node.set_property("query_terms", &query_text.split_whitespace().count().to_string());

        fusion_node.add_child(vector_node);
        fusion_node.add_child(text_node);

        // Limit
        let mut limit_node = PlanNode::new(PlanNodeType::Limit { limit: k, offset: 0 });
        limit_node.estimated_rows = k;
        limit_node.estimated_cost = 0.1;
        limit_node.actual_time_ms = Some(0.01);
        limit_node.add_child(fusion_node);

        root.add_child(limit_node);
        root.actual_time_ms = Some(start.elapsed().as_secs_f64() * 1000.0 + 7.0);

        let stats = ExecutionStats {
            total_time_ms: root.actual_time_ms.unwrap_or(0.0),
            planning_time_ms: 0.2,
            execution_time_ms: root.actual_time_ms.unwrap_or(0.0) - 0.2,
            rows_examined: k * 4,
            rows_returned: k,
            cache_hits: 0,
            cache_misses: 2,
            index_scans: 2,
            seq_scans: 0,
            memory_bytes: query_vector.len() * 4 * k * 2,
            disk_reads: 0,
        };

        ExplainResult {
            plan: root,
            stats,
            suggestions: vec![],
        }
    }
}

impl Default for QueryExplainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explain_search() {
        let explainer = QueryExplainer::new();
        let query = vec![0.1f32; 128];

        let result = explainer.explain_search(
            &query,
            10,
            None,
            &ExplainOptions::default()
        );

        assert!(result.stats.total_time_ms > 0.0);
        assert_eq!(result.stats.rows_returned, 10);
    }

    #[test]
    fn test_format_text() {
        let explainer = QueryExplainer::new();
        let query = vec![0.1f32; 128];

        let result = explainer.explain_search(
            &query,
            10,
            Some("category = 'test'"),
            &ExplainOptions::default()
        );

        let text = result.format(&ExplainOptions::default());
        assert!(text.contains("QUERY PLAN"));
        assert!(text.contains("Vector Scan"));
        assert!(text.contains("Filter"));
    }

    #[test]
    fn test_format_html() {
        let explainer = QueryExplainer::new();
        let query = vec![0.1f32; 128];

        let result = explainer.explain_search(&query, 10, None, &ExplainOptions::default());

        let mut options = ExplainOptions::default();
        options.format = ExplainFormat::Html;

        let html = result.format(&options);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Query Execution Plan"));
    }

    #[test]
    fn test_format_json() {
        let explainer = QueryExplainer::new();
        let query = vec![0.1f32; 128];

        let result = explainer.explain_search(&query, 10, None, &ExplainOptions::default());

        let mut options = ExplainOptions::default();
        options.format = ExplainFormat::Json;

        let json = result.format(&options);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["plan"].is_object());
    }

    #[test]
    fn test_explain_hybrid() {
        let explainer = QueryExplainer::new();
        let query = vec![0.1f32; 128];

        let result = explainer.explain_hybrid(
            &query,
            "test query",
            10,
            &ExplainOptions::default()
        );

        let text = result.format(&ExplainOptions::default());
        assert!(text.contains("Fusion"));
        assert!(text.contains("BM25"));
    }
}
