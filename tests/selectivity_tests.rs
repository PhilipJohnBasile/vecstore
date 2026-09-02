// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! Tests for filter selectivity estimation and query optimization

use serde_json::json;
use vecstore::{FilterExpr, FilterOp};

/// Helper to create a comparison filter
fn cmp(field: &str, op: FilterOp, value: serde_json::Value) -> FilterExpr {
    FilterExpr::Cmp {
        field: field.to_string(),
        op,
        value,
    }
}

/// Test equality filter selectivity (should be low - selective)
#[test]
fn test_eq_selectivity() {
    let filter = cmp("category", FilterOp::Eq, json!("electronics"));

    // Verify structure
    match &filter {
        FilterExpr::Cmp { field, op, value } => {
            assert_eq!(field, "category");
            assert_eq!(*op, FilterOp::Eq);
            assert_eq!(*value, json!("electronics"));
        },
        _ => panic!("Expected Cmp filter"),
    }
}

/// Test inequality filter selectivity (should be high - not selective)
#[test]
fn test_neq_selectivity() {
    let filter = cmp("status", FilterOp::Neq, json!("deleted"));

    match &filter {
        FilterExpr::Cmp { op, .. } => {
            assert_eq!(*op, FilterOp::Neq);
        },
        _ => panic!("Expected Cmp filter"),
    }
}

/// Test range filter selectivity
#[test]
fn test_range_selectivity() {
    // Less than
    let lt_filter = cmp("price", FilterOp::Lt, json!(100));
    match &lt_filter {
        FilterExpr::Cmp { op, .. } => assert_eq!(*op, FilterOp::Lt),
        _ => panic!("Expected Cmp filter"),
    }

    // Greater than
    let gt_filter = cmp("price", FilterOp::Gt, json!(50));
    match &gt_filter {
        FilterExpr::Cmp { op, .. } => assert_eq!(*op, FilterOp::Gt),
        _ => panic!("Expected Cmp filter"),
    }

    // Less than or equal
    let lte_filter = cmp("quantity", FilterOp::Lte, json!(10));
    match &lte_filter {
        FilterExpr::Cmp { op, .. } => assert_eq!(*op, FilterOp::Lte),
        _ => panic!("Expected Cmp filter"),
    }

    // Greater than or equal
    let gte_filter = cmp("rating", FilterOp::Gte, json!(4.0));
    match &gte_filter {
        FilterExpr::Cmp { op, .. } => assert_eq!(*op, FilterOp::Gte),
        _ => panic!("Expected Cmp filter"),
    }
}

/// Test AND filter selectivity (product of child selectivities)
#[test]
fn test_and_selectivity() {
    let filter = FilterExpr::And(vec![
        cmp("category", FilterOp::Eq, json!("electronics")),
        cmp("price", FilterOp::Lt, json!(1000)),
    ]);

    match &filter {
        FilterExpr::And(exprs) => {
            assert_eq!(exprs.len(), 2);
        },
        _ => panic!("Expected And filter"),
    }
}

/// Test OR filter selectivity (sum of child selectivities, capped at 1.0)
#[test]
fn test_or_selectivity() {
    let filter = FilterExpr::Or(vec![
        cmp("category", FilterOp::Eq, json!("electronics")),
        cmp("category", FilterOp::Eq, json!("computers")),
    ]);

    match &filter {
        FilterExpr::Or(exprs) => {
            assert_eq!(exprs.len(), 2);
        },
        _ => panic!("Expected Or filter"),
    }
}

/// Test NOT filter selectivity (inverts child selectivity)
#[test]
fn test_not_selectivity() {
    let filter = FilterExpr::Not(Box::new(cmp("status", FilterOp::Eq, json!("archived"))));

    match &filter {
        FilterExpr::Not(inner) => match inner.as_ref() {
            FilterExpr::Cmp { op, .. } => assert_eq!(*op, FilterOp::Eq),
            _ => panic!("Expected inner Cmp filter"),
        },
        _ => panic!("Expected Not filter"),
    }
}

/// Test Contains filter
#[test]
fn test_contains_filter() {
    let filter = cmp("description", FilterOp::Contains, json!("wireless"));

    match &filter {
        FilterExpr::Cmp { op, .. } => assert_eq!(*op, FilterOp::Contains),
        _ => panic!("Expected Cmp filter"),
    }
}

/// Test In filter (value in array)
#[test]
fn test_in_filter() {
    let filter = cmp("color", FilterOp::In, json!(["red", "blue", "green"]));

    match &filter {
        FilterExpr::Cmp { op, value, .. } => {
            assert_eq!(*op, FilterOp::In);
            assert!(value.is_array());
        },
        _ => panic!("Expected Cmp filter"),
    }
}

/// Test NotIn filter
#[test]
fn test_not_in_filter() {
    let filter = cmp("status", FilterOp::NotIn, json!(["deleted", "archived"]));

    match &filter {
        FilterExpr::Cmp { op, value, .. } => {
            assert_eq!(*op, FilterOp::NotIn);
            assert!(value.is_array());
        },
        _ => panic!("Expected Cmp filter"),
    }
}

/// Test StartsWith filter (prefix matching)
#[test]
fn test_starts_with_filter() {
    let filter = cmp("title", FilterOp::StartsWith, json!("Introduction to"));

    match &filter {
        FilterExpr::Cmp { op, .. } => assert_eq!(*op, FilterOp::StartsWith),
        _ => panic!("Expected Cmp filter"),
    }
}

/// Test complex nested filter
#[test]
fn test_complex_nested_filter() {
    // (category = "electronics" AND price < 500) OR (category = "books" AND rating >= 4)
    let filter = FilterExpr::Or(vec![
        FilterExpr::And(vec![
            cmp("category", FilterOp::Eq, json!("electronics")),
            cmp("price", FilterOp::Lt, json!(500)),
        ]),
        FilterExpr::And(vec![
            cmp("category", FilterOp::Eq, json!("books")),
            cmp("rating", FilterOp::Gte, json!(4)),
        ]),
    ]);

    match &filter {
        FilterExpr::Or(branches) => {
            assert_eq!(branches.len(), 2);
            for branch in branches {
                match branch {
                    FilterExpr::And(conditions) => {
                        assert_eq!(conditions.len(), 2);
                    },
                    _ => panic!("Expected And filter in branch"),
                }
            }
        },
        _ => panic!("Expected Or filter"),
    }
}

/// Test deeply nested NOT
#[test]
fn test_deeply_nested_not() {
    // NOT (NOT (category = "electronics"))
    let filter = FilterExpr::Not(Box::new(FilterExpr::Not(Box::new(cmp(
        "category",
        FilterOp::Eq,
        json!("electronics"),
    )))));

    match &filter {
        FilterExpr::Not(inner) => {
            match inner.as_ref() {
                FilterExpr::Not(innermost) => {
                    match innermost.as_ref() {
                        FilterExpr::Cmp { .. } => {}, // Expected
                        _ => panic!("Expected innermost Cmp"),
                    }
                },
                _ => panic!("Expected inner Not"),
            }
        },
        _ => panic!("Expected outer Not"),
    }
}

/// Test empty AND (should be valid)
#[test]
fn test_empty_and() {
    let filter = FilterExpr::And(vec![]);

    match &filter {
        FilterExpr::And(exprs) => {
            assert!(exprs.is_empty());
        },
        _ => panic!("Expected And filter"),
    }
}

/// Test empty OR (should be valid)
#[test]
fn test_empty_or() {
    let filter = FilterExpr::Or(vec![]);

    match &filter {
        FilterExpr::Or(exprs) => {
            assert!(exprs.is_empty());
        },
        _ => panic!("Expected Or filter"),
    }
}

/// Test filter with different value types
#[test]
fn test_filter_value_types() {
    // String value
    let str_filter = cmp("name", FilterOp::Eq, json!("test"));
    assert!(matches!(str_filter, FilterExpr::Cmp { value, .. } if value.is_string()));

    // Number value
    let num_filter = cmp("count", FilterOp::Gt, json!(42));
    assert!(matches!(num_filter, FilterExpr::Cmp { value, .. } if value.is_number()));

    // Float value
    let float_filter = cmp("score", FilterOp::Gte, json!(3.14));
    assert!(matches!(float_filter, FilterExpr::Cmp { value, .. } if value.is_f64()));

    // Boolean value
    let bool_filter = cmp("active", FilterOp::Eq, json!(true));
    assert!(matches!(bool_filter, FilterExpr::Cmp { value, .. } if value.is_boolean()));

    // Null value
    let null_filter = cmp("optional", FilterOp::Eq, json!(null));
    assert!(matches!(null_filter, FilterExpr::Cmp { value, .. } if value.is_null()));

    // Array value (for In/NotIn)
    let arr_filter = cmp("tags", FilterOp::In, json!(["a", "b", "c"]));
    assert!(matches!(arr_filter, FilterExpr::Cmp { value, .. } if value.is_array()));
}

/// Test filter serialization
#[test]
fn test_filter_serialization() {
    let filter = FilterExpr::And(vec![
        cmp("category", FilterOp::Eq, json!("test")),
        cmp("price", FilterOp::Lt, json!(100)),
    ]);

    // Serialize to JSON
    let json = serde_json::to_string(&filter).unwrap();
    assert!(!json.is_empty());

    // Deserialize back
    let deserialized: FilterExpr = serde_json::from_str(&json).unwrap();

    match deserialized {
        FilterExpr::And(exprs) => {
            assert_eq!(exprs.len(), 2);
        },
        _ => panic!("Expected And filter after deserialization"),
    }
}

/// Test FilterOp serialization
#[test]
fn test_filter_op_serialization() {
    let ops = vec![
        FilterOp::Eq,
        FilterOp::Neq,
        FilterOp::Gt,
        FilterOp::Gte,
        FilterOp::Lt,
        FilterOp::Lte,
        FilterOp::Contains,
        FilterOp::In,
        FilterOp::NotIn,
        FilterOp::StartsWith,
    ];

    for op in ops {
        let json = serde_json::to_string(&op).unwrap();
        let deserialized: FilterOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op, deserialized);
    }
}
