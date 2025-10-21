//! Advanced Metadata Filtering
//!
//! This module provides enhanced filtering capabilities beyond the basic
//! comparison operators, including:
//!
//! - **JSON Path queries**: Navigate nested JSON structures (`user.address.city = 'NYC'`)
//! - **Regular expressions**: Pattern matching (`name ~= '^John'`)
//! - **Array operations**: Contains, length, any/all (`tags contains 'rust'`)
//! - **Composite filters**: AND/OR/NOT with complex expressions
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::advanced_filter::{AdvancedFilter, FilterBuilder};
//!
//! # fn main() -> anyhow::Result<()> {
//! let filter = FilterBuilder::new()
//!     .json_path("user.profile.age", ">=", 18)?
//!     .and()
//!     .regex("email", r".*@example\.com$")?
//!     .and()
//!     .array_contains("tags", "premium")?
//!     .build();
//!
//! // Use with VecStore query
//! // let results = store.query_with_advanced_filter(query, filter)?;
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use regex::Regex;
use serde_json::Value;
use std::collections::HashSet;

/// Advanced filter expression
#[derive(Debug, Clone)]
pub enum AdvancedFilter {
    /// JSON path query: path, operator, value
    JsonPath(String, String, Value),
    /// Regular expression: field, pattern
    Regex(String, String),
    /// Array contains: field, value
    ArrayContains(String, Value),
    /// Array length: field, operator, length
    ArrayLength(String, String, usize),
    /// All array elements match condition
    ArrayAll(String, Box<AdvancedFilter>),
    /// Any array element matches condition
    ArrayAny(String, Box<AdvancedFilter>),
    /// Logical AND
    And(Vec<AdvancedFilter>),
    /// Logical OR
    Or(Vec<AdvancedFilter>),
    /// Logical NOT
    Not(Box<AdvancedFilter>),
    /// Basic comparison (field, operator, value)
    Basic(String, String, Value),
}

impl AdvancedFilter {
    /// Evaluate filter against metadata
    pub fn matches(&self, metadata: &Value) -> Result<bool> {
        match self {
            AdvancedFilter::JsonPath(path, op, value) => {
                let extracted = extract_json_path(metadata, path)?;
                compare_values(&extracted, op, value)
            }
            AdvancedFilter::Regex(field, pattern) => {
                let field_value = extract_field(metadata, field)?;
                if let Some(s) = field_value.as_str() {
                    let re = Regex::new(pattern)?;
                    Ok(re.is_match(s))
                } else {
                    Ok(false)
                }
            }
            AdvancedFilter::ArrayContains(field, value) => {
                let field_value = extract_field(metadata, field)?;
                if let Some(arr) = field_value.as_array() {
                    Ok(arr.contains(value))
                } else {
                    Ok(false)
                }
            }
            AdvancedFilter::ArrayLength(field, op, length) => {
                let field_value = extract_field(metadata, field)?;
                if let Some(arr) = field_value.as_array() {
                    let actual_length = arr.len();
                    compare_numbers(actual_length as i64, op, *length as i64)
                } else {
                    Ok(false)
                }
            }
            AdvancedFilter::ArrayAll(field, condition) => {
                let field_value = extract_field(metadata, field)?;
                if let Some(arr) = field_value.as_array() {
                    for item in arr {
                        if !condition.matches(item)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            AdvancedFilter::ArrayAny(field, condition) => {
                let field_value = extract_field(metadata, field)?;
                if let Some(arr) = field_value.as_array() {
                    for item in arr {
                        if condition.matches(item)? {
                            return Ok(true);
                        }
                    }
                    Ok(false)
                } else {
                    Ok(false)
                }
            }
            AdvancedFilter::And(filters) => {
                for filter in filters {
                    if !filter.matches(metadata)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            AdvancedFilter::Or(filters) => {
                for filter in filters {
                    if filter.matches(metadata)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            AdvancedFilter::Not(filter) => Ok(!filter.matches(metadata)?),
            AdvancedFilter::Basic(field, op, value) => {
                let field_value = extract_field(metadata, field)?;
                compare_values(&field_value, op, value)
            }
        }
    }
}

/// Builder for advanced filters
pub struct FilterBuilder {
    filters: Vec<AdvancedFilter>,
}

impl FilterBuilder {
    /// Create a new filter builder
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Add JSON path query
    pub fn json_path(mut self, path: &str, op: &str, value: Value) -> Result<Self> {
        self.filters.push(AdvancedFilter::JsonPath(
            path.to_string(),
            op.to_string(),
            value,
        ));
        Ok(self)
    }

    /// Add regex pattern match
    pub fn regex(mut self, field: &str, pattern: &str) -> Result<Self> {
        // Validate regex
        Regex::new(pattern)?;
        self.filters.push(AdvancedFilter::Regex(
            field.to_string(),
            pattern.to_string(),
        ));
        Ok(self)
    }

    /// Add array contains check
    pub fn array_contains(mut self, field: &str, value: Value) -> Result<Self> {
        self.filters
            .push(AdvancedFilter::ArrayContains(field.to_string(), value));
        Ok(self)
    }

    /// Add array length check
    pub fn array_length(mut self, field: &str, op: &str, length: usize) -> Result<Self> {
        self.filters.push(AdvancedFilter::ArrayLength(
            field.to_string(),
            op.to_string(),
            length,
        ));
        Ok(self)
    }

    /// Add basic comparison
    pub fn basic(mut self, field: &str, op: &str, value: Value) -> Result<Self> {
        self.filters.push(AdvancedFilter::Basic(
            field.to_string(),
            op.to_string(),
            value,
        ));
        Ok(self)
    }

    /// Combine with AND
    pub fn and(self) -> Self {
        self
    }

    /// Build final filter
    pub fn build(self) -> AdvancedFilter {
        if self.filters.len() == 1 {
            self.filters.into_iter().next().unwrap()
        } else {
            AdvancedFilter::And(self.filters)
        }
    }
}

impl Default for FilterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract value from JSON path
fn extract_json_path(metadata: &Value, path: &str) -> Result<Value> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = metadata;

    for part in parts {
        // Handle array indexing like "items[0]"
        if let Some(bracket_pos) = part.find('[') {
            let field = &part[..bracket_pos];
            let index_str = &part[bracket_pos + 1..part.len() - 1];
            let index: usize = index_str.parse()?;

            current = &current[field];
            if let Some(arr) = current.as_array() {
                current = arr
                    .get(index)
                    .ok_or_else(|| anyhow!("Index out of bounds"))?;
            } else {
                return Err(anyhow!("Not an array: {}", field));
            }
        } else {
            current = &current[part];
        }

        if current.is_null() {
            return Ok(Value::Null);
        }
    }

    Ok(current.clone())
}

/// Extract field from metadata (supports dot notation for one level)
fn extract_field(metadata: &Value, field: &str) -> Result<Value> {
    if field.contains('.') {
        extract_json_path(metadata, field)
    } else {
        Ok(metadata[field].clone())
    }
}

/// Compare two JSON values
fn compare_values(a: &Value, op: &str, b: &Value) -> Result<bool> {
    match op {
        "=" | "==" => Ok(a == b),
        "!=" => Ok(a != b),
        ">" => compare_numeric(a, b, |x, y| x > y),
        ">=" => compare_numeric(a, b, |x, y| x >= y),
        "<" => compare_numeric(a, b, |x, y| x < y),
        "<=" => compare_numeric(a, b, |x, y| x <= y),
        _ => Err(anyhow!("Unknown operator: {}", op)),
    }
}

/// Compare numeric values
fn compare_numeric<F>(a: &Value, b: &Value, cmp: F) -> Result<bool>
where
    F: Fn(f64, f64) -> bool,
{
    let a_num = a.as_f64().ok_or_else(|| anyhow!("Not a number"))?;
    let b_num = b.as_f64().ok_or_else(|| anyhow!("Not a number"))?;
    Ok(cmp(a_num, b_num))
}

/// Compare integer numbers
fn compare_numbers(a: i64, op: &str, b: i64) -> Result<bool> {
    match op {
        "=" | "==" => Ok(a == b),
        "!=" => Ok(a != b),
        ">" => Ok(a > b),
        ">=" => Ok(a >= b),
        "<" => Ok(a < b),
        "<=" => Ok(a <= b),
        _ => Err(anyhow!("Unknown operator: {}", op)),
    }
}

/// Parse advanced filter from string
pub fn parse_advanced_filter(query: &str) -> Result<AdvancedFilter> {
    // Basic parsing - production version would use a proper parser
    let query = query.trim();

    // Check for regex operator
    if query.contains("~=") {
        let parts: Vec<&str> = query.split("~=").collect();
        if parts.len() == 2 {
            let field = parts[0].trim();
            let pattern = parts[1].trim().trim_matches('\'').trim_matches('"');
            return Ok(AdvancedFilter::Regex(
                field.to_string(),
                pattern.to_string(),
            ));
        }
    }

    // Check for array contains
    if query.contains(" contains ") {
        let parts: Vec<&str> = query.split(" contains ").collect();
        if parts.len() == 2 {
            let field = parts[0].trim();
            let value_str = parts[1].trim().trim_matches('\'').trim_matches('"');
            return Ok(AdvancedFilter::ArrayContains(
                field.to_string(),
                Value::String(value_str.to_string()),
            ));
        }
    }

    // Check for array.length
    if query.contains(".length ") {
        let parts: Vec<&str> = query.split(".length ").collect();
        if parts.len() == 2 {
            let field = parts[0].trim();
            let rest = parts[1].trim();
            for op in &[">=", "<=", ">", "<", "="] {
                if let Some(op_pos) = rest.find(op) {
                    let length_str = rest[op_pos + op.len()..].trim();
                    let length: usize = length_str.parse()?;
                    return Ok(AdvancedFilter::ArrayLength(
                        field.to_string(),
                        op.to_string(),
                        length,
                    ));
                }
            }
        }
    }

    // Basic comparison
    for op in &[">=", "<=", "!=", "=", ">", "<"] {
        if let Some(op_pos) = query.find(op) {
            let field = query[..op_pos].trim();
            let value_str = query[op_pos + op.len()..].trim();

            let value = if value_str.starts_with('\'') || value_str.starts_with('"') {
                Value::String(value_str.trim_matches('\'').trim_matches('"').to_string())
            } else if let Ok(num) = value_str.parse::<i64>() {
                Value::Number(num.into())
            } else if let Ok(num) = value_str.parse::<f64>() {
                Value::Number(serde_json::Number::from_f64(num).unwrap())
            } else {
                Value::String(value_str.to_string())
            };

            return Ok(AdvancedFilter::Basic(
                field.to_string(),
                op.to_string(),
                value,
            ));
        }
    }

    Err(anyhow!("Could not parse filter: {}", query))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_json_path_extraction() {
        let data = json!({
            "user": {
                "name": "Alice",
                "age": 30,
                "address": {
                    "city": "NYC"
                }
            }
        });

        let result = extract_json_path(&data, "user.name").unwrap();
        assert_eq!(result, json!("Alice"));

        let result = extract_json_path(&data, "user.address.city").unwrap();
        assert_eq!(result, json!("NYC"));
    }

    #[test]
    fn test_json_path_filter() {
        let data = json!({
            "user": {
                "age": 25
            }
        });

        let filter = AdvancedFilter::JsonPath("user.age".to_string(), ">=".to_string(), json!(18));
        assert!(filter.matches(&data).unwrap());

        let filter = AdvancedFilter::JsonPath("user.age".to_string(), "<".to_string(), json!(18));
        assert!(!filter.matches(&data).unwrap());
    }

    #[test]
    fn test_regex_filter() {
        let data = json!({
            "email": "alice@example.com"
        });

        let filter = AdvancedFilter::Regex("email".to_string(), r".*@example\.com$".to_string());
        assert!(filter.matches(&data).unwrap());

        let filter = AdvancedFilter::Regex("email".to_string(), r".*@test\.com$".to_string());
        assert!(!filter.matches(&data).unwrap());
    }

    #[test]
    fn test_array_contains() {
        let data = json!({
            "tags": ["rust", "vector", "database"]
        });

        let filter = AdvancedFilter::ArrayContains("tags".to_string(), json!("rust"));
        assert!(filter.matches(&data).unwrap());

        let filter = AdvancedFilter::ArrayContains("tags".to_string(), json!("python"));
        assert!(!filter.matches(&data).unwrap());
    }

    #[test]
    fn test_array_length() {
        let data = json!({
            "tags": ["a", "b", "c"]
        });

        let filter = AdvancedFilter::ArrayLength("tags".to_string(), ">".to_string(), 2);
        assert!(filter.matches(&data).unwrap());

        let filter = AdvancedFilter::ArrayLength("tags".to_string(), "=".to_string(), 3);
        assert!(filter.matches(&data).unwrap());

        let filter = AdvancedFilter::ArrayLength("tags".to_string(), "<".to_string(), 3);
        assert!(!filter.matches(&data).unwrap());
    }

    #[test]
    fn test_and_filter() {
        let data = json!({
            "age": 25,
            "name": "Alice"
        });

        let filter = AdvancedFilter::And(vec![
            AdvancedFilter::Basic("age".to_string(), ">=".to_string(), json!(18)),
            AdvancedFilter::Basic("name".to_string(), "=".to_string(), json!("Alice")),
        ]);

        assert!(filter.matches(&data).unwrap());
    }

    #[test]
    fn test_or_filter() {
        let data = json!({
            "status": "active"
        });

        let filter = AdvancedFilter::Or(vec![
            AdvancedFilter::Basic("status".to_string(), "=".to_string(), json!("active")),
            AdvancedFilter::Basic("status".to_string(), "=".to_string(), json!("pending")),
        ]);

        assert!(filter.matches(&data).unwrap());
    }

    #[test]
    fn test_not_filter() {
        let data = json!({
            "deleted": false
        });

        let filter = AdvancedFilter::Not(Box::new(AdvancedFilter::Basic(
            "deleted".to_string(),
            "=".to_string(),
            json!(true),
        )));

        assert!(filter.matches(&data).unwrap());
    }

    #[test]
    fn test_filter_builder() {
        let data = json!({
            "user": {
                "age": 25
            },
            "email": "test@example.com",
            "tags": ["premium"]
        });

        let filter = FilterBuilder::new()
            .json_path("user.age", ">=", json!(18))
            .unwrap()
            .and()
            .regex("email", r".*@example\.com$")
            .unwrap()
            .and()
            .array_contains("tags", json!("premium"))
            .unwrap()
            .build();

        assert!(filter.matches(&data).unwrap());
    }

    #[test]
    fn test_parse_regex_filter() {
        let filter = parse_advanced_filter("name ~= '^John'").unwrap();

        let data = json!({"name": "John Doe"});
        assert!(filter.matches(&data).unwrap());

        let data = json!({"name": "Jane Doe"});
        assert!(!filter.matches(&data).unwrap());
    }

    #[test]
    fn test_parse_array_contains() {
        let filter = parse_advanced_filter("tags contains 'rust'").unwrap();

        let data = json!({"tags": ["rust", "programming"]});
        assert!(filter.matches(&data).unwrap());

        let data = json!({"tags": ["python", "programming"]});
        assert!(!filter.matches(&data).unwrap());
    }

    #[test]
    fn test_parse_array_length() {
        let filter = parse_advanced_filter("items.length > 5").unwrap();

        let data = json!({"items": [1, 2, 3, 4, 5, 6]});
        assert!(filter.matches(&data).unwrap());

        let data = json!({"items": [1, 2, 3]});
        assert!(!filter.matches(&data).unwrap());
    }

    #[test]
    fn test_parse_basic_filter() {
        let filter = parse_advanced_filter("age >= 18").unwrap();

        let data = json!({"age": 25});
        assert!(filter.matches(&data).unwrap());

        let data = json!({"age": 15});
        assert!(!filter.matches(&data).unwrap());
    }

    #[test]
    fn test_array_indexing() {
        let data = json!({
            "items": [
                {"name": "first"},
                {"name": "second"}
            ]
        });

        let result = extract_json_path(&data, "items[0].name").unwrap();
        assert_eq!(result, json!("first"));

        let result = extract_json_path(&data, "items[1].name").unwrap();
        assert_eq!(result, json!("second"));
    }
}
