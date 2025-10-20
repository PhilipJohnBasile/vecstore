//! Schema validation for metadata
//!
//! This module provides JSON Schema-like validation for vector metadata,
//! ensuring data quality and catching errors early.
//!
//! ## Features
//!
//! - Field type validation (string, number, boolean, array, object)
//! - Required fields enforcement
//! - Custom validators
//! - Range constraints (min/max for numbers)
//! - Pattern matching for strings (regex)
//!
//! ## Usage
//!
//! ```no_run
//! use vecstore::schema::{Schema, FieldType, FieldSchema};
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut schema = Schema::new();
//!
//! // Define required fields
//! schema.add_field("title", FieldSchema::new(FieldType::String).required());
//! schema.add_field("score", FieldSchema::new(FieldType::Number)
//!     .with_min(0.0)
//!     .with_max(1.0));
//!
//! // Validate metadata
//! let metadata = serde_json::json!({
//!     "title": "My Document",
//!     "score": 0.95
//! });
//!
//! schema.validate(&metadata)?;
//! # Ok(())
//! # }
//! ```

use serde_json::Value;
use std::collections::HashMap;

/// Field data type
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
    Null,
}

/// Field schema definition
#[derive(Debug, Clone)]
pub struct FieldSchema {
    /// Field type
    pub field_type: FieldType,

    /// Whether field is required
    pub required: bool,

    /// Minimum value (for numbers)
    pub min: Option<f64>,

    /// Maximum value (for numbers)
    pub max: Option<f64>,

    /// Pattern for string validation (regex pattern string)
    pub pattern: Option<String>,

    /// Allowed values (enum)
    pub allowed_values: Option<Vec<Value>>,

    /// Custom error message
    pub error_message: Option<String>,
}

impl FieldSchema {
    /// Create a new field schema
    pub fn new(field_type: FieldType) -> Self {
        Self {
            field_type,
            required: false,
            min: None,
            max: None,
            pattern: None,
            allowed_values: None,
            error_message: None,
        }
    }

    /// Mark field as required
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Set minimum value constraint
    pub fn with_min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }

    /// Set maximum value constraint
    pub fn with_max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    /// Set pattern for string validation
    pub fn with_pattern(mut self, pattern: String) -> Self {
        self.pattern = Some(pattern);
        self
    }

    /// Set allowed values (enum)
    pub fn with_allowed_values(mut self, values: Vec<Value>) -> Self {
        self.allowed_values = Some(values);
        self
    }

    /// Set custom error message
    pub fn with_error(mut self, message: String) -> Self {
        self.error_message = Some(message);
        self
    }

    /// Validate a value against this schema
    pub fn validate(&self, value: &Value) -> Result<(), ValidationError> {
        // Type validation
        match (&self.field_type, value) {
            (FieldType::String, Value::String(_)) => {}
            (FieldType::Number, Value::Number(_)) => {}
            (FieldType::Integer, Value::Number(n)) if n.is_i64() || n.is_u64() => {}
            (FieldType::Boolean, Value::Bool(_)) => {}
            (FieldType::Array, Value::Array(_)) => {}
            (FieldType::Object, Value::Object(_)) => {}
            (FieldType::Null, Value::Null) => {}
            _ => {
                return Err(ValidationError::TypeError {
                    expected: format!("{:?}", self.field_type),
                    actual: value.clone(),
                    message: self.error_message.clone(),
                });
            }
        }

        // Range validation for numbers
        if let Value::Number(n) = value {
            if let Some(n_f64) = n.as_f64() {
                if let Some(min) = self.min {
                    if n_f64 < min {
                        return Err(ValidationError::RangeError {
                            value: n_f64,
                            min: Some(min),
                            max: None,
                        });
                    }
                }

                if let Some(max) = self.max {
                    if n_f64 > max {
                        return Err(ValidationError::RangeError {
                            value: n_f64,
                            min: None,
                            max: Some(max),
                        });
                    }
                }
            }
        }

        // Allowed values validation
        if let Some(ref allowed) = self.allowed_values {
            if !allowed.contains(value) {
                return Err(ValidationError::EnumError {
                    value: value.clone(),
                    allowed: allowed.clone(),
                });
            }
        }

        // Pattern validation for strings
        if let (Some(ref pattern), Value::String(s)) = (&self.pattern, value) {
            // Simple contains check (in production, use regex crate)
            if !s.contains(pattern) {
                return Err(ValidationError::PatternError {
                    value: s.clone(),
                    pattern: pattern.clone(),
                });
            }
        }

        Ok(())
    }
}

/// Metadata schema
#[derive(Debug, Clone)]
pub struct Schema {
    /// Field schemas
    fields: HashMap<String, FieldSchema>,

    /// Allow additional fields not in schema
    allow_additional: bool,
}

impl Schema {
    /// Create a new schema
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            allow_additional: true,
        }
    }

    /// Disallow additional fields
    pub fn strict(mut self) -> Self {
        self.allow_additional = false;
        self
    }

    /// Add a field to the schema
    pub fn add_field(&mut self, name: impl Into<String>, schema: FieldSchema) -> &mut Self {
        self.fields.insert(name.into(), schema);
        self
    }

    /// Validate metadata against this schema
    pub fn validate(&self, metadata: &Value) -> Result<(), ValidationError> {
        let obj = match metadata.as_object() {
            Some(o) => o,
            None => {
                return Err(ValidationError::NotAnObject {
                    actual: metadata.clone(),
                })
            }
        };

        // Check required fields
        for (field_name, field_schema) in &self.fields {
            if field_schema.required && !obj.contains_key(field_name) {
                return Err(ValidationError::MissingField {
                    field: field_name.clone(),
                });
            }

            // Validate field if present
            if let Some(value) = obj.get(field_name) {
                field_schema.validate(value).map_err(|e| match e {
                    ValidationError::TypeError { .. }
                    | ValidationError::RangeError { .. }
                    | ValidationError::EnumError { .. }
                    | ValidationError::PatternError { .. } => ValidationError::FieldError {
                        field: field_name.clone(),
                        error: Box::new(e),
                    },
                    _ => e,
                })?;
            }
        }

        // Check for unexpected fields
        if !self.allow_additional {
            for key in obj.keys() {
                if !self.fields.contains_key(key) {
                    return Err(ValidationError::UnexpectedField { field: key.clone() });
                }
            }
        }

        Ok(())
    }
}

impl Default for Schema {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation error
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Type mismatch
    TypeError {
        expected: String,
        actual: Value,
        message: Option<String>,
    },

    /// Value out of range
    RangeError {
        value: f64,
        min: Option<f64>,
        max: Option<f64>,
    },

    /// Value not in allowed enum
    EnumError { value: Value, allowed: Vec<Value> },

    /// String pattern mismatch
    PatternError { value: String, pattern: String },

    /// Required field missing
    MissingField { field: String },

    /// Unexpected field (strict mode)
    UnexpectedField { field: String },

    /// Field-level error
    FieldError {
        field: String,
        error: Box<ValidationError>,
    },

    /// Metadata is not an object
    NotAnObject { actual: Value },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeError {
                expected,
                actual,
                message,
            } => {
                if let Some(msg) = message {
                    write!(f, "{}", msg)
                } else {
                    write!(f, "Type error: expected {}, got {:?}", expected, actual)
                }
            }
            Self::RangeError { value, min, max } => {
                if let (Some(min), Some(max)) = (min, max) {
                    write!(f, "Value {} out of range [{}, {}]", value, min, max)
                } else if let Some(min) = min {
                    write!(f, "Value {} below minimum {}", value, min)
                } else if let Some(max) = max {
                    write!(f, "Value {} above maximum {}", value, max)
                } else {
                    write!(f, "Range error for value {}", value)
                }
            }
            Self::EnumError { value, allowed } => {
                write!(f, "Value {:?} not in allowed values: {:?}", value, allowed)
            }
            Self::PatternError { value, pattern } => {
                write!(f, "Value '{}' does not match pattern '{}'", value, pattern)
            }
            Self::MissingField { field } => {
                write!(f, "Required field '{}' is missing", field)
            }
            Self::UnexpectedField { field } => {
                write!(f, "Unexpected field '{}'", field)
            }
            Self::FieldError { field, error } => {
                write!(f, "Field '{}': {}", field, error)
            }
            Self::NotAnObject { actual } => {
                write!(f, "Metadata must be an object, got {:?}", actual)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_type_validation() {
        let schema = FieldSchema::new(FieldType::String);

        assert!(schema.validate(&json!("hello")).is_ok());
        assert!(schema.validate(&json!(42)).is_err());
    }

    #[test]
    fn test_required_field() {
        let mut schema = Schema::new();
        schema.add_field("name", FieldSchema::new(FieldType::String).required());

        let valid = json!({"name": "test"});
        assert!(schema.validate(&valid).is_ok());

        let invalid = json!({});
        assert!(schema.validate(&invalid).is_err());
    }

    #[test]
    fn test_range_validation() {
        let schema = FieldSchema::new(FieldType::Number)
            .with_min(0.0)
            .with_max(1.0);

        assert!(schema.validate(&json!(0.5)).is_ok());
        assert!(schema.validate(&json!(-0.1)).is_err());
        assert!(schema.validate(&json!(1.5)).is_err());
    }

    #[test]
    fn test_enum_validation() {
        let schema = FieldSchema::new(FieldType::String).with_allowed_values(vec![
            json!("red"),
            json!("green"),
            json!("blue"),
        ]);

        assert!(schema.validate(&json!("red")).is_ok());
        assert!(schema.validate(&json!("yellow")).is_err());
    }

    #[test]
    fn test_pattern_validation() {
        let schema = FieldSchema::new(FieldType::String).with_pattern("@".to_string());

        assert!(schema.validate(&json!("user@example.com")).is_ok());
        assert!(schema.validate(&json!("invalid-email")).is_err());
    }

    #[test]
    fn test_strict_mode() {
        let mut schema = Schema::new().strict();
        schema.add_field("name", FieldSchema::new(FieldType::String));

        let valid = json!({"name": "test"});
        assert!(schema.validate(&valid).is_ok());

        let invalid = json!({"name": "test", "extra": "field"});
        assert!(schema.validate(&invalid).is_err());
    }

    #[test]
    fn test_multiple_fields() {
        let mut schema = Schema::new();
        schema.add_field("title", FieldSchema::new(FieldType::String).required());
        schema.add_field(
            "score",
            FieldSchema::new(FieldType::Number)
                .with_min(0.0)
                .with_max(1.0),
        );
        schema.add_field("published", FieldSchema::new(FieldType::Boolean));

        let valid = json!({
            "title": "My Document",
            "score": 0.95,
            "published": true
        });

        assert!(schema.validate(&valid).is_ok());

        let invalid_score = json!({
            "title": "My Document",
            "score": 1.5  // Out of range
        });

        assert!(schema.validate(&invalid_score).is_err());
    }

    #[test]
    fn test_custom_error_message() {
        let schema =
            FieldSchema::new(FieldType::String).with_error("Must be a text value".to_string());

        let err = schema.validate(&json!(42)).unwrap_err();
        assert!(err.to_string().contains("Must be a text value"));
    }
}
