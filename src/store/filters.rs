use super::types::{FilterExpr, FilterOp, Metadata};
use serde_json::Value;

pub fn evaluate_filter(filter: &FilterExpr, metadata: &Metadata) -> bool {
    match filter {
        FilterExpr::And(exprs) => exprs.iter().all(|e| evaluate_filter(e, metadata)),
        FilterExpr::Or(exprs) => exprs.iter().any(|e| evaluate_filter(e, metadata)),
        FilterExpr::Not(expr) => !evaluate_filter(expr, metadata),
        FilterExpr::Cmp { field, op, value } => {
            let field_value = metadata.fields.get(field);
            match field_value {
                Some(fv) => evaluate_comparison(fv, op, value),
                None => false,
            }
        }
    }
}

fn evaluate_comparison(field_value: &Value, op: &FilterOp, target: &Value) -> bool {
    match op {
        FilterOp::Eq => values_equal(field_value, target),
        FilterOp::Neq => !values_equal(field_value, target),
        FilterOp::Gt => compare_numeric(field_value, target, |a, b| a > b),
        FilterOp::Gte => compare_numeric(field_value, target, |a, b| a >= b),
        FilterOp::Lt => compare_numeric(field_value, target, |a, b| a < b),
        FilterOp::Lte => compare_numeric(field_value, target, |a, b| a <= b),
        FilterOp::Contains => {
            // For strings, check substring; for arrays, check element presence
            match (field_value, target) {
                (Value::String(s), Value::String(pattern)) => s.contains(pattern.as_str()),
                (Value::Array(arr), val) => arr.contains(val),
                _ => false,
            }
        }
        FilterOp::In => {
            // Check if field_value is in the target array
            match target {
                Value::Array(arr) => arr.iter().any(|v| values_equal(field_value, v)),
                _ => false,
            }
        }
        FilterOp::NotIn => {
            // Check if field_value is NOT in the target array
            match target {
                Value::Array(arr) => !arr.iter().any(|v| values_equal(field_value, v)),
                _ => true, // If target is not an array, field is "not in" it
            }
        }
        FilterOp::StartsWith => {
            // Check if string field starts with prefix (Major Issue #13 fix)
            match (field_value, target) {
                (Value::String(s), Value::String(prefix)) => s.starts_with(prefix.as_str()),
                _ => false,
            }
        }
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    // Direct equality
    if a == b {
        return true;
    }

    // Try numeric coercion
    if let (Some(a_num), Some(b_num)) = (as_f64(a), as_f64(b)) {
        return (a_num - b_num).abs() < f64::EPSILON;
    }

    false
}

fn compare_numeric<F>(a: &Value, b: &Value, cmp: F) -> bool
where
    F: Fn(f64, f64) -> bool,
{
    match (as_f64(a), as_f64(b)) {
        (Some(a_num), Some(b_num)) => cmp(a_num, b_num),
        _ => false,
    }
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_metadata(fields: Vec<(&str, Value)>) -> Metadata {
        let mut map = HashMap::new();
        for (k, v) in fields {
            map.insert(k.to_string(), v);
        }
        Metadata { fields: map }
    }

    #[test]
    fn test_eq_filter() {
        let meta = make_metadata(vec![("topic", Value::String("rust".into()))]);
        let filter = FilterExpr::Cmp {
            field: "topic".into(),
            op: FilterOp::Eq,
            value: Value::String("rust".into()),
        };
        assert!(evaluate_filter(&filter, &meta));
    }

    #[test]
    fn test_gt_filter() {
        let meta = make_metadata(vec![("score", serde_json::json!(10))]);
        let filter = FilterExpr::Cmp {
            field: "score".into(),
            op: FilterOp::Gt,
            value: serde_json::json!(5),
        };
        assert!(evaluate_filter(&filter, &meta));
    }

    #[test]
    fn test_and_filter() {
        let meta = make_metadata(vec![
            ("topic", Value::String("rust".into())),
            ("score", serde_json::json!(10)),
        ]);
        let filter = FilterExpr::And(vec![
            FilterExpr::Cmp {
                field: "topic".into(),
                op: FilterOp::Eq,
                value: Value::String("rust".into()),
            },
            FilterExpr::Cmp {
                field: "score".into(),
                op: FilterOp::Gt,
                value: serde_json::json!(5),
            },
        ]);
        assert!(evaluate_filter(&filter, &meta));
    }

    #[test]
    fn test_not_filter() {
        let meta = make_metadata(vec![("topic", Value::String("rust".into()))]);
        let filter = FilterExpr::Not(Box::new(FilterExpr::Cmp {
            field: "topic".into(),
            op: FilterOp::Eq,
            value: Value::String("python".into()),
        }));
        assert!(evaluate_filter(&filter, &meta));
    }

    #[test]
    fn test_missing_field() {
        let meta = make_metadata(vec![]);
        let filter = FilterExpr::Cmp {
            field: "missing".into(),
            op: FilterOp::Eq,
            value: Value::String("value".into()),
        };
        assert!(!evaluate_filter(&filter, &meta));
    }

    #[test]
    fn test_in_operator_string() {
        let meta = make_metadata(vec![("category", Value::String("AI".into()))]);
        let filter = FilterExpr::Cmp {
            field: "category".into(),
            op: FilterOp::In,
            value: serde_json::json!(["AI", "ML", "NLP"]),
        };
        assert!(evaluate_filter(&filter, &meta));
    }

    #[test]
    fn test_in_operator_number() {
        let meta = make_metadata(vec![("priority", serde_json::json!(1))]);
        let filter = FilterExpr::Cmp {
            field: "priority".into(),
            op: FilterOp::In,
            value: serde_json::json!([1, 2, 3]),
        };
        assert!(evaluate_filter(&filter, &meta));
    }

    #[test]
    fn test_in_operator_not_found() {
        let meta = make_metadata(vec![("category", Value::String("Python".into()))]);
        let filter = FilterExpr::Cmp {
            field: "category".into(),
            op: FilterOp::In,
            value: serde_json::json!(["AI", "ML", "NLP"]),
        };
        assert!(!evaluate_filter(&filter, &meta));
    }

    #[test]
    fn test_not_in_operator() {
        let meta = make_metadata(vec![("status", Value::String("active".into()))]);
        let filter = FilterExpr::Cmp {
            field: "status".into(),
            op: FilterOp::NotIn,
            value: serde_json::json!(["deprecated", "deleted", "archived"]),
        };
        assert!(evaluate_filter(&filter, &meta));
    }

    #[test]
    fn test_not_in_operator_found() {
        let meta = make_metadata(vec![("status", Value::String("deprecated".into()))]);
        let filter = FilterExpr::Cmp {
            field: "status".into(),
            op: FilterOp::NotIn,
            value: serde_json::json!(["deprecated", "deleted", "archived"]),
        };
        assert!(!evaluate_filter(&filter, &meta));
    }

    #[test]
    fn test_complex_filter_with_in() {
        let meta = make_metadata(vec![
            ("category", Value::String("AI".into())),
            ("priority", serde_json::json!(1)),
        ]);
        let filter = FilterExpr::And(vec![
            FilterExpr::Cmp {
                field: "category".into(),
                op: FilterOp::In,
                value: serde_json::json!(["AI", "ML"]),
            },
            FilterExpr::Cmp {
                field: "priority".into(),
                op: FilterOp::In,
                value: serde_json::json!([1, 2, 3]),
            },
        ]);
        assert!(evaluate_filter(&filter, &meta));
    }
}
