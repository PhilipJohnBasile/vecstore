// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! Tests for the Agent module (agentic vector search)

use vecstore::agent::{
    AgentConfig, AgentResult, AgentStyle, AgentTool, ExecutionPlan, ExecutionStep,
    FilterExpression, FilterValue, IntentAction, IntentConstraint, IntentModifier, PlanStep,
    QueryAgent, QueryIntent,
};

/// Test agent configuration defaults
#[test]
fn test_agent_config_defaults() {
    let config = AgentConfig::default();

    assert!(config.max_steps > 0);
    assert!(config.max_execution_time_secs > 0);
    assert!(config.memory_capacity > 0);
    assert!(config.explain);
    assert!(config.self_correct);
    assert!(config.confidence_threshold > 0.0 && config.confidence_threshold <= 1.0);
    assert!(matches!(config.style, AgentStyle::Balanced));
}

/// Test agent configuration with custom values
#[test]
fn test_agent_config_custom() {
    let config = AgentConfig {
        max_steps: 5,
        max_execution_time_secs: 60,
        explain: false,
        memory_capacity: 50,
        self_correct: false,
        confidence_threshold: 0.9,
        style: AgentStyle::Fast,
    };

    assert_eq!(config.max_steps, 5);
    assert_eq!(config.max_execution_time_secs, 60);
    assert!(!config.explain);
    assert_eq!(config.memory_capacity, 50);
    assert!(!config.self_correct);
    assert!((config.confidence_threshold - 0.9).abs() < 0.001);
    assert!(matches!(config.style, AgentStyle::Fast));
}

/// Test agent style variants
#[test]
fn test_agent_styles() {
    let styles = vec![
        AgentStyle::Balanced,
        AgentStyle::Fast,
        AgentStyle::Thorough,
        AgentStyle::Explainable,
    ];

    for style in styles {
        let config = AgentConfig {
            style,
            ..AgentConfig::default()
        };
        match config.style {
            AgentStyle::Balanced
            | AgentStyle::Fast
            | AgentStyle::Thorough
            | AgentStyle::Explainable => {}
        }
    }
}

/// Test execution plan creation
#[test]
fn test_execution_plan_creation() {
    let intent = QueryIntent {
        action: IntentAction::Search,
        entities: vec!["documents".to_string()],
        constraints: vec![],
        modifiers: vec![],
    };

    let steps = vec![
        PlanStep {
            step: 1,
            description: "Search for similar vectors".to_string(),
            tool: AgentTool::VectorSearch {
                query: vec![0.1, 0.2, 0.3],
                k: 10,
                filters: None,
            },
            depends_on: vec![],
            expected_outcome: "List of similar documents".to_string(),
            confidence: 0.9,
            fallback: None,
        },
        PlanStep {
            step: 2,
            description: "Filter results by metadata".to_string(),
            tool: AgentTool::Filter {
                expression: FilterExpression::Eq {
                    field: "category".to_string(),
                    value: FilterValue::String("tech".to_string()),
                },
                input_ids: vec!["step1_results".to_string()],
            },
            depends_on: vec![1],
            expected_outcome: "Filtered results".to_string(),
            confidence: 0.85,
            fallback: None,
        },
    ];

    let plan = ExecutionPlan {
        query: "Find similar documents".to_string(),
        intent,
        steps: steps.clone(),
        confidence: steps.iter().map(|s| s.confidence).product(),
        estimated_time_ms: 100,
        explanation: "Search then filter".to_string(),
    };

    assert_eq!(plan.steps.len(), 2);
    assert!((plan.confidence - 0.765).abs() < 0.001); // 0.9 * 0.85 = 0.765
}

/// Test execution step tracking
#[test]
fn test_execution_step() {
    let step = ExecutionStep {
        step: 1,
        tool: "vector_search".to_string(),
        input_summary: "Query: find similar documents".to_string(),
        output_summary: "Found 10 results".to_string(),
        success: true,
        time_ms: 150,
        explanation: "Performed HNSW search".to_string(),
    };

    assert!(step.success);
    assert_eq!(step.time_ms, 150);
}

/// Test agent result structure
#[test]
fn test_agent_result_structure() {
    let intent = QueryIntent {
        action: IntentAction::Search,
        entities: vec![],
        constraints: vec![],
        modifiers: vec![],
    };

    let plan = ExecutionPlan {
        query: "test query".to_string(),
        intent,
        steps: vec![],
        confidence: 0.9,
        estimated_time_ms: 50,
        explanation: "Test plan".to_string(),
    };

    let result = AgentResult {
        query: "test query".to_string(),
        result_ids: vec!["id1".to_string(), "id2".to_string()],
        scores: vec![0.95, 0.87],
        plan,
        execution_trace: vec![],
        explanation: "Test explanation".to_string(),
        confidence: 0.85,
        execution_time_ms: 100,
        suggestions: vec!["Try narrowing search".to_string()],
        warnings: vec![],
    };

    assert_eq!(result.result_ids.len(), 2);
    assert_eq!(result.scores.len(), 2);
    assert!(result.confidence > 0.0 && result.confidence <= 1.0);
}

/// Test query agent creation
#[test]
fn test_query_agent_creation() {
    let config = AgentConfig::default();
    let agent = QueryAgent::new(config);

    // Agent should have empty memory initially
    assert!(agent.memory().is_empty());
}

/// Test agent memory management
#[test]
fn test_agent_memory_capacity() {
    let config = AgentConfig {
        memory_capacity: 3,
        ..AgentConfig::default()
    };
    let mut agent = QueryAgent::new(config);

    // Memory should be empty initially
    assert!(agent.memory().is_empty());

    // Clear memory should work on empty
    agent.clear_memory();
    assert!(agent.memory().is_empty());
}

/// Test confidence value bounds
#[test]
fn test_confidence_bounds() {
    let intent = QueryIntent {
        action: IntentAction::Transform,
        entities: vec![],
        constraints: vec![],
        modifiers: vec![],
    };

    // High confidence plan
    let high_plan = ExecutionPlan {
        query: "test".to_string(),
        intent: intent.clone(),
        steps: vec![PlanStep {
            step: 1,
            description: "Test step".to_string(),
            tool: AgentTool::Rerank {
                query: "test query".to_string(),
                input_ids: vec!["doc1".to_string()],
                k: 10,
            },
            depends_on: vec![],
            expected_outcome: "Reranked".to_string(),
            confidence: 1.0,
            fallback: None,
        }],
        confidence: 1.0,
        estimated_time_ms: 50,
        explanation: "High confidence test".to_string(),
    };

    assert!(high_plan.confidence >= 0.0 && high_plan.confidence <= 1.0);

    // Low confidence plan
    let low_plan = ExecutionPlan {
        query: "test".to_string(),
        intent,
        steps: vec![PlanStep {
            step: 1,
            description: "Test step".to_string(),
            tool: AgentTool::Explain {
                query_vector: vec![0.1, 0.2, 0.3],
                result_id: "doc1".to_string(),
            },
            depends_on: vec![],
            expected_outcome: "Explanation".to_string(),
            confidence: 0.1,
            fallback: None,
        }],
        confidence: 0.1,
        estimated_time_ms: 50,
        explanation: "Low confidence test".to_string(),
    };

    assert!(low_plan.confidence >= 0.0 && low_plan.confidence <= 1.0);
}

/// Test multiple execution steps
#[test]
fn test_execution_trace() {
    let trace = vec![
        ExecutionStep {
            step: 1,
            tool: "embed".to_string(),
            input_summary: "Text input".to_string(),
            output_summary: "384-dim vector".to_string(),
            success: true,
            time_ms: 50,
            explanation: "Generated embedding".to_string(),
        },
        ExecutionStep {
            step: 2,
            tool: "search".to_string(),
            input_summary: "Query vector".to_string(),
            output_summary: "10 neighbors".to_string(),
            success: true,
            time_ms: 25,
            explanation: "HNSW search".to_string(),
        },
        ExecutionStep {
            step: 3,
            tool: "filter".to_string(),
            input_summary: "10 candidates".to_string(),
            output_summary: "5 filtered".to_string(),
            success: true,
            time_ms: 5,
            explanation: "Applied metadata filter".to_string(),
        },
    ];

    // All steps should be successful
    assert!(trace.iter().all(|s| s.success));

    // Total time should be sum of steps
    let total_time: u64 = trace.iter().map(|s| s.time_ms).sum();
    assert_eq!(total_time, 80);

    // Steps should be in order
    for (i, step) in trace.iter().enumerate() {
        assert_eq!(step.step, i + 1);
    }
}

/// Test failed execution step
#[test]
fn test_failed_execution_step() {
    let failed_step = ExecutionStep {
        step: 1,
        tool: "external_api".to_string(),
        input_summary: "Request".to_string(),
        output_summary: "Error: timeout".to_string(),
        success: false,
        time_ms: 5000,
        explanation: "External API timed out".to_string(),
    };

    assert!(!failed_step.success);
    assert!(failed_step.time_ms >= 5000); // Timeout duration
}

/// Test AgentTool variants
#[test]
fn test_agent_tools() {
    // VectorSearch
    let vs = AgentTool::VectorSearch {
        query: vec![0.1, 0.2, 0.3],
        k: 10,
        filters: None,
    };
    assert!(matches!(vs, AgentTool::VectorSearch { .. }));

    // KeywordSearch
    let ks = AgentTool::KeywordSearch {
        query: "test query".to_string(),
        k: 10,
        filters: None,
    };
    assert!(matches!(ks, AgentTool::KeywordSearch { .. }));

    // HybridSearch
    let hs = AgentTool::HybridSearch {
        query: "test".to_string(),
        vector: Some(vec![0.1, 0.2]),
        k: 10,
        alpha: 0.5,
    };
    assert!(matches!(hs, AgentTool::HybridSearch { .. }));

    // Filter
    let filter = AgentTool::Filter {
        expression: FilterExpression::Eq {
            field: "status".to_string(),
            value: FilterValue::String("active".to_string()),
        },
        input_ids: vec!["doc1".to_string()],
    };
    assert!(matches!(filter, AgentTool::Filter { .. }));

    // Rerank
    let rerank = AgentTool::Rerank {
        query: "test query".to_string(),
        input_ids: vec!["doc1".to_string()],
        k: 5,
    };
    assert!(matches!(rerank, AgentTool::Rerank { .. }));

    // Explain
    let explain = AgentTool::Explain {
        query_vector: vec![0.1, 0.2, 0.3],
        result_id: "doc1".to_string(),
    };
    assert!(matches!(explain, AgentTool::Explain { .. }));

    // Fetch
    let fetch = AgentTool::Fetch {
        ids: vec!["id1".to_string(), "id2".to_string()],
    };
    assert!(matches!(fetch, AgentTool::Fetch { .. }));
}

/// Test plan step dependencies
#[test]
fn test_plan_step_dependencies() {
    let step1 = PlanStep {
        step: 1,
        description: "First step".to_string(),
        tool: AgentTool::VectorSearch {
            query: vec![0.1],
            k: 10,
            filters: None,
        },
        depends_on: vec![], // No dependencies
        expected_outcome: "Results".to_string(),
        confidence: 0.9,
        fallback: None,
    };

    let step2 = PlanStep {
        step: 2,
        description: "Second step".to_string(),
        tool: AgentTool::Filter {
            expression: FilterExpression::Gt {
                field: "score".to_string(),
                value: FilterValue::Float(0.5),
            },
            input_ids: vec!["step1".to_string()],
        },
        depends_on: vec![1], // Depends on step 1
        expected_outcome: "Filtered".to_string(),
        confidence: 0.85,
        fallback: None,
    };

    let step3 = PlanStep {
        step: 3,
        description: "Third step".to_string(),
        tool: AgentTool::Rerank {
            query: "test".to_string(),
            input_ids: vec!["step2".to_string()],
            k: 5,
        },
        depends_on: vec![2], // Depends on step 2
        expected_outcome: "Reranked".to_string(),
        confidence: 0.95,
        fallback: None,
    };

    assert!(step1.depends_on.is_empty());
    assert_eq!(step2.depends_on, vec![1]);
    assert_eq!(step3.depends_on, vec![2]);
}

/// Test plan step with fallback
#[test]
fn test_plan_step_fallback() {
    let fallback_step = PlanStep {
        step: 1,
        description: "Fallback to keyword search".to_string(),
        tool: AgentTool::KeywordSearch {
            query: "backup search".to_string(),
            k: 10,
            filters: None,
        },
        depends_on: vec![],
        expected_outcome: "Fallback results".to_string(),
        confidence: 0.7,
        fallback: None, // No further fallback
    };

    let primary_step = PlanStep {
        step: 1,
        description: "Primary vector search".to_string(),
        tool: AgentTool::VectorSearch {
            query: vec![0.1, 0.2, 0.3],
            k: 10,
            filters: None,
        },
        depends_on: vec![],
        expected_outcome: "Primary results".to_string(),
        confidence: 0.9,
        fallback: Some(Box::new(fallback_step)),
    };

    assert!(primary_step.fallback.is_some());
    let fb = primary_step.fallback.as_ref().unwrap();
    assert!(fb.fallback.is_none());
    assert_eq!(fb.description, "Fallback to keyword search");
}

/// Test filter expressions
#[test]
fn test_filter_expressions() {
    // Eq
    let eq = FilterExpression::Eq {
        field: "category".to_string(),
        value: FilterValue::String("electronics".to_string()),
    };
    assert!(matches!(eq, FilterExpression::Eq { .. }));

    // Ne
    let ne = FilterExpression::Ne {
        field: "status".to_string(),
        value: FilterValue::String("deleted".to_string()),
    };
    assert!(matches!(ne, FilterExpression::Ne { .. }));

    // Gt
    let gt = FilterExpression::Gt {
        field: "price".to_string(),
        value: FilterValue::Float(100.0),
    };
    assert!(matches!(gt, FilterExpression::Gt { .. }));

    // Lt
    let lt = FilterExpression::Lt {
        field: "count".to_string(),
        value: FilterValue::Int(50),
    };
    assert!(matches!(lt, FilterExpression::Lt { .. }));

    // In
    let in_expr = FilterExpression::In {
        field: "color".to_string(),
        values: vec![
            FilterValue::String("red".to_string()),
            FilterValue::String("blue".to_string()),
        ],
    };
    assert!(matches!(in_expr, FilterExpression::In { .. }));

    // Contains
    let contains = FilterExpression::Contains {
        field: "description".to_string(),
        value: "wireless".to_string(),
    };
    assert!(matches!(contains, FilterExpression::Contains { .. }));

    // And
    let and_expr = FilterExpression::And(vec![eq, gt]);
    assert!(matches!(and_expr, FilterExpression::And(_)));

    // Or
    let or_expr = FilterExpression::Or(vec![ne, lt]);
    assert!(matches!(or_expr, FilterExpression::Or(_)));

    // Not
    let not_expr = FilterExpression::Not(Box::new(in_expr));
    assert!(matches!(not_expr, FilterExpression::Not(_)));
}

/// Test filter value types
#[test]
fn test_filter_values() {
    let string_val = FilterValue::String("test".to_string());
    assert!(matches!(string_val, FilterValue::String(_)));

    let int_val = FilterValue::Int(42);
    assert!(matches!(int_val, FilterValue::Int(42)));

    let float_val = FilterValue::Float(3.14);
    assert!(matches!(float_val, FilterValue::Float(_)));

    let bool_val = FilterValue::Bool(true);
    assert!(matches!(bool_val, FilterValue::Bool(true)));

    let null_val = FilterValue::Null;
    assert!(matches!(null_val, FilterValue::Null));
}

/// Test query intent
#[test]
fn test_query_intent() {
    let intent = QueryIntent {
        action: IntentAction::Search,
        entities: vec!["products".to_string(), "reviews".to_string()],
        constraints: vec![IntentConstraint {
            field: "category".to_string(),
            operator: "eq".to_string(),
            value: "electronics".to_string(),
        }],
        modifiers: vec![
            IntentModifier::Limit(10),
            IntentModifier::SortBy {
                field: "score".to_string(),
                ascending: false,
            },
        ],
    };

    assert!(matches!(intent.action, IntentAction::Search));
    assert_eq!(intent.entities.len(), 2);
    assert_eq!(intent.constraints.len(), 1);
    assert_eq!(intent.modifiers.len(), 2);
}

/// Test intent actions
#[test]
fn test_intent_actions() {
    let actions = vec![
        IntentAction::Search,
        IntentAction::Filter,
        IntentAction::Aggregate,
        IntentAction::Compare,
        IntentAction::Explain,
        IntentAction::Transform,
        IntentAction::Combine,
    ];

    for action in actions {
        match action {
            IntentAction::Search
            | IntentAction::Filter
            | IntentAction::Aggregate
            | IntentAction::Compare
            | IntentAction::Explain
            | IntentAction::Transform
            | IntentAction::Combine => {}
        }
    }
}

/// Test intent modifiers
#[test]
fn test_intent_modifiers() {
    let modifiers = vec![
        IntentModifier::TimeRange {
            start: Some("-7d".to_string()),
            end: None,
        },
        IntentModifier::Limit(100),
        IntentModifier::Exclude(vec!["deleted".to_string()]),
        IntentModifier::Include(vec!["active".to_string()]),
        IntentModifier::SortBy {
            field: "date".to_string(),
            ascending: true,
        },
    ];

    assert_eq!(modifiers.len(), 5);
    assert!(matches!(modifiers[0], IntentModifier::TimeRange { .. }));
    assert!(matches!(modifiers[1], IntentModifier::Limit(100)));
    assert!(matches!(modifiers[2], IntentModifier::Exclude(_)));
    assert!(matches!(modifiers[3], IntentModifier::Include(_)));
    assert!(matches!(modifiers[4], IntentModifier::SortBy { .. }));
}

/// Test serialization of agent structures
#[test]
fn test_agent_serialization() {
    let step = ExecutionStep {
        step: 1,
        tool: "search".to_string(),
        input_summary: "query".to_string(),
        output_summary: "results".to_string(),
        success: true,
        time_ms: 100,
        explanation: "test".to_string(),
    };

    let json = serde_json::to_string(&step).unwrap();
    let deserialized: ExecutionStep = serde_json::from_str(&json).unwrap();

    assert_eq!(step.step, deserialized.step);
    assert_eq!(step.success, deserialized.success);
    assert_eq!(step.time_ms, deserialized.time_ms);
}

/// Test agent config serialization
#[test]
fn test_agent_config_serialization() {
    let config = AgentConfig {
        max_steps: 15,
        max_execution_time_secs: 45,
        explain: true,
        memory_capacity: 200,
        self_correct: true,
        confidence_threshold: 0.8,
        style: AgentStyle::Thorough,
    };

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.max_steps, deserialized.max_steps);
    assert_eq!(
        config.max_execution_time_secs,
        deserialized.max_execution_time_secs
    );
    assert_eq!(config.memory_capacity, deserialized.memory_capacity);
    assert!((config.confidence_threshold - deserialized.confidence_threshold).abs() < 0.001);
}
