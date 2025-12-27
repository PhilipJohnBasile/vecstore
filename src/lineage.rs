// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! Vector Lineage and Provenance Tracking
//!
//! This module provides comprehensive lineage tracking for vectors, enabling:
//! - **Source Tracking**: Know where each embedding came from
//! - **Transformation History**: Track all operations applied to vectors
//! - **Model Attribution**: Which model generated each embedding
//! - **Compliance Auditing**: Full audit trail for regulatory requirements
//! - **Reproducibility**: Recreate any vector's journey from source to index
//!
//! # Example
//!
//! ```ignore
//! use vecstore::lineage::{LineageTracker, VectorOrigin, TransformationType};
//!
//! let mut tracker = LineageTracker::new();
//!
//! // Register vector origin
//! let origin = VectorOrigin::new("doc123")
//!     .from_model("openai/text-embedding-3-large")
//!     .from_source("documents/report.pdf", "page_3")
//!     .with_timestamp(Utc::now());
//!
//! tracker.register("vec_001", origin);
//!
//! // Track transformations
//! tracker.add_transformation("vec_001", TransformationType::Normalized);
//! tracker.add_transformation("vec_001", TransformationType::Quantized { bits: 8 });
//!
//! // Query lineage
//! let lineage = tracker.get_lineage("vec_001")?;
//! println!("Source: {:?}", lineage.origin);
//! println!("Transformations: {:?}", lineage.transformations);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::VecStoreError;

/// Origin information for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorOrigin {
    /// Unique identifier for the source document/data
    pub source_id: String,
    /// Type of source (document, image, audio, etc.)
    pub source_type: SourceType,
    /// Location within the source (page, timestamp, region)
    pub source_location: Option<String>,
    /// Original file path or URL
    pub source_path: Option<String>,
    /// Model used to generate the embedding
    pub embedding_model: Option<EmbeddingModel>,
    /// When the embedding was created
    pub created_at: DateTime<Utc>,
    /// Who/what created this embedding
    pub created_by: Option<String>,
    /// Additional custom metadata
    pub custom_fields: HashMap<String, serde_json::Value>,
}

impl VectorOrigin {
    /// Create a new vector origin
    pub fn new(source_id: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            source_type: SourceType::Unknown,
            source_location: None,
            source_path: None,
            embedding_model: None,
            created_at: Utc::now(),
            created_by: None,
            custom_fields: HashMap::new(),
        }
    }

    /// Set the source type
    pub fn with_type(mut self, source_type: SourceType) -> Self {
        self.source_type = source_type;
        self
    }

    /// Set the embedding model
    pub fn from_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = Some(EmbeddingModel {
            name: model.into(),
            version: None,
            provider: None,
            dimensions: None,
        });
        self
    }

    /// Set the embedding model with full details
    pub fn from_model_details(mut self, model: EmbeddingModel) -> Self {
        self.embedding_model = Some(model);
        self
    }

    /// Set the source path and location
    pub fn from_source(mut self, path: impl Into<String>, location: impl Into<String>) -> Self {
        self.source_path = Some(path.into());
        self.source_location = Some(location.into());
        self
    }

    /// Set the creation timestamp
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.created_at = timestamp;
        self
    }

    /// Set who created this
    pub fn created_by(mut self, creator: impl Into<String>) -> Self {
        self.created_by = Some(creator.into());
        self
    }

    /// Add custom field
    pub fn with_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.custom_fields.insert(key.into(), value);
        self
    }
}

/// Type of source data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SourceType {
    /// Text document (PDF, Word, plain text)
    Document,
    /// Web page or HTML content
    WebPage,
    /// Image file
    Image,
    /// Audio file or transcript
    Audio,
    /// Video file or transcript
    Video,
    /// Structured data (CSV, JSON, database)
    StructuredData,
    /// Code or source files
    Code,
    /// User-generated content
    UserContent,
    /// System-generated content
    SystemGenerated,
    /// Unknown or unspecified
    Unknown,
}

/// Information about the embedding model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModel {
    /// Model name (e.g., "text-embedding-3-large")
    pub name: String,
    /// Model version
    pub version: Option<String>,
    /// Provider (e.g., "OpenAI", "Cohere", "Local")
    pub provider: Option<String>,
    /// Output dimensions
    pub dimensions: Option<usize>,
}

impl EmbeddingModel {
    /// Create a new embedding model
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: None,
            provider: None,
            dimensions: None,
        }
    }

    /// Set the version
    pub fn version(mut self, v: impl Into<String>) -> Self {
        self.version = Some(v.into());
        self
    }

    /// Set the provider
    pub fn provider(mut self, p: impl Into<String>) -> Self {
        self.provider = Some(p.into());
        self
    }

    /// Set the dimensions
    pub fn dimensions(mut self, d: usize) -> Self {
        self.dimensions = Some(d);
        self
    }
}

/// Type of transformation applied to a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationType {
    /// Vector was normalized to unit length
    Normalized,
    /// Dimensions were reduced
    DimensionReduced {
        method: String,
        from_dims: usize,
        to_dims: usize,
    },
    /// Vector was quantized
    Quantized {
        bits: u8,
    },
    /// Product quantization was applied
    ProductQuantized {
        num_subvectors: usize,
        bits_per_subvector: u8,
    },
    /// Vector was averaged with others
    Averaged {
        source_count: usize,
    },
    /// Vector was concatenated with others
    Concatenated {
        source_ids: Vec<String>,
    },
    /// Vector was updated/modified
    Updated {
        reason: String,
    },
    /// Custom transformation
    Custom {
        name: String,
        parameters: HashMap<String, serde_json::Value>,
    },
}

/// Record of a transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationRecord {
    /// Type of transformation
    pub transformation: TransformationType,
    /// When it was applied
    pub timestamp: DateTime<Utc>,
    /// Who/what applied it
    pub applied_by: Option<String>,
    /// Vector hash before transformation
    pub before_hash: Option<u64>,
    /// Vector hash after transformation
    pub after_hash: Option<u64>,
    /// Was transformation reversible?
    pub reversible: bool,
}

/// Complete lineage for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorLineage {
    /// Vector ID
    pub vector_id: String,
    /// Origin information
    pub origin: VectorOrigin,
    /// List of transformations in order
    pub transformations: Vec<TransformationRecord>,
    /// Parent vector IDs (if derived from other vectors)
    pub parent_ids: Vec<String>,
    /// Child vector IDs (vectors derived from this one)
    pub child_ids: Vec<String>,
    /// Current version number
    pub version: u64,
    /// Last modification timestamp
    pub last_modified: DateTime<Utc>,
    /// Compliance tags
    pub compliance_tags: Vec<String>,
    /// Access history
    pub access_log: Vec<AccessRecord>,
}

/// Record of vector access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRecord {
    /// When it was accessed
    pub timestamp: DateTime<Utc>,
    /// Type of access
    pub access_type: AccessType,
    /// Who accessed it
    pub accessor: Option<String>,
    /// Query that returned this vector (if applicable)
    pub query_id: Option<String>,
}

/// Type of access
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccessType {
    /// Vector was read/retrieved
    Read,
    /// Vector was used in a search result
    SearchResult,
    /// Vector was exported
    Exported,
    /// Vector was used for training/analytics
    Analytics,
    /// Vector was modified
    Modified,
}

/// Compliance report for audit purposes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Time period covered
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    /// Total vectors tracked
    pub total_vectors: usize,
    /// Vectors by source type
    pub by_source_type: HashMap<String, usize>,
    /// Vectors by model
    pub by_model: HashMap<String, usize>,
    /// Total transformations
    pub total_transformations: usize,
    /// Access statistics
    pub access_stats: AccessStatistics,
    /// Data retention status
    pub retention_status: Vec<RetentionRecord>,
    /// Compliance issues found
    pub issues: Vec<ComplianceIssue>,
}

/// Access statistics for compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessStatistics {
    /// Total accesses
    pub total_accesses: usize,
    /// Accesses by type
    pub by_type: HashMap<String, usize>,
    /// Unique accessors
    pub unique_accessors: usize,
    /// Most accessed vectors
    pub top_accessed: Vec<(String, usize)>,
}

/// Data retention record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRecord {
    /// Vector ID
    pub vector_id: String,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Retention policy name
    pub policy: String,
    /// Expiration date
    pub expires_at: Option<DateTime<Utc>>,
    /// Is expired?
    pub is_expired: bool,
}

/// Compliance issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: String,
    /// Issue description
    pub description: String,
    /// Affected vector IDs
    pub affected_vectors: Vec<String>,
    /// Recommended action
    pub recommendation: String,
}

/// Severity levels for compliance issues
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum IssueSeverity {
    Info,
    Warning,
    Critical,
}

/// Main lineage tracker
pub struct LineageTracker {
    /// Lineage records by vector ID
    lineage: Arc<RwLock<HashMap<String, VectorLineage>>>,
    /// Enable access logging
    log_access: bool,
    /// Retention policies
    retention_policies: HashMap<String, RetentionPolicy>,
}

/// Retention policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Policy name
    pub name: String,
    /// Maximum age in days (0 = forever)
    pub max_age_days: u64,
    /// Action on expiration
    pub on_expire: RetentionAction,
    /// Tags this policy applies to
    pub applies_to_tags: Vec<String>,
}

/// Action on retention expiration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RetentionAction {
    /// Delete the vector
    Delete,
    /// Archive the vector
    Archive,
    /// Anonymize the lineage
    Anonymize,
    /// Alert only, take no action
    AlertOnly,
}

impl LineageTracker {
    /// Create a new lineage tracker
    pub fn new() -> Self {
        Self {
            lineage: Arc::new(RwLock::new(HashMap::new())),
            log_access: true,
            retention_policies: HashMap::new(),
        }
    }

    /// Create with specific configuration
    pub fn with_config(log_access: bool) -> Self {
        Self {
            lineage: Arc::new(RwLock::new(HashMap::new())),
            log_access,
            retention_policies: HashMap::new(),
        }
    }

    /// Register a new vector with its origin
    pub fn register(&self, vector_id: impl Into<String>, origin: VectorOrigin) -> Result<(), VecStoreError> {
        let id = vector_id.into();
        let lineage = VectorLineage {
            vector_id: id.clone(),
            origin,
            transformations: Vec::new(),
            parent_ids: Vec::new(),
            child_ids: Vec::new(),
            version: 1,
            last_modified: Utc::now(),
            compliance_tags: Vec::new(),
            access_log: Vec::new(),
        };

        let mut store = self.lineage.write().map_err(|_| {
            VecStoreError::Internal("Failed to acquire write lock".into())
        })?;

        store.insert(id, lineage);
        Ok(())
    }

    /// Register a derived vector (from parent vectors)
    pub fn register_derived(
        &self,
        vector_id: impl Into<String>,
        origin: VectorOrigin,
        parent_ids: Vec<String>,
    ) -> Result<(), VecStoreError> {
        let id = vector_id.into();

        let mut store = self.lineage.write().map_err(|_| {
            VecStoreError::Internal("Failed to acquire write lock".into())
        })?;

        // Update parent's child_ids
        for parent_id in &parent_ids {
            if let Some(parent) = store.get_mut(parent_id) {
                parent.child_ids.push(id.clone());
            }
        }

        let lineage = VectorLineage {
            vector_id: id.clone(),
            origin,
            transformations: Vec::new(),
            parent_ids,
            child_ids: Vec::new(),
            version: 1,
            last_modified: Utc::now(),
            compliance_tags: Vec::new(),
            access_log: Vec::new(),
        };

        store.insert(id, lineage);
        Ok(())
    }

    /// Add a transformation to a vector's history
    pub fn add_transformation(
        &self,
        vector_id: &str,
        transformation: TransformationType,
    ) -> Result<(), VecStoreError> {
        self.add_transformation_with_details(vector_id, transformation, None, None, None)
    }

    /// Add a transformation with full details
    pub fn add_transformation_with_details(
        &self,
        vector_id: &str,
        transformation: TransformationType,
        applied_by: Option<String>,
        before_hash: Option<u64>,
        after_hash: Option<u64>,
    ) -> Result<(), VecStoreError> {
        let mut store = self.lineage.write().map_err(|_| {
            VecStoreError::Internal("Failed to acquire write lock".into())
        })?;

        let lineage = store.get_mut(vector_id).ok_or_else(|| {
            VecStoreError::NotFound(format!("Vector {} not found in lineage", vector_id))
        })?;

        let reversible = matches!(
            transformation,
            TransformationType::Normalized | TransformationType::Updated { .. }
        );

        let record = TransformationRecord {
            transformation,
            timestamp: Utc::now(),
            applied_by,
            before_hash,
            after_hash,
            reversible,
        };

        lineage.transformations.push(record);
        lineage.version += 1;
        lineage.last_modified = Utc::now();

        Ok(())
    }

    /// Log an access to a vector
    pub fn log_access(
        &self,
        vector_id: &str,
        access_type: AccessType,
        accessor: Option<String>,
        query_id: Option<String>,
    ) -> Result<(), VecStoreError> {
        if !self.log_access {
            return Ok(());
        }

        let mut store = self.lineage.write().map_err(|_| {
            VecStoreError::Internal("Failed to acquire write lock".into())
        })?;

        if let Some(lineage) = store.get_mut(vector_id) {
            lineage.access_log.push(AccessRecord {
                timestamp: Utc::now(),
                access_type,
                accessor,
                query_id,
            });
        }

        Ok(())
    }

    /// Get lineage for a vector
    pub fn get_lineage(&self, vector_id: &str) -> Result<VectorLineage, VecStoreError> {
        let store = self.lineage.read().map_err(|_| {
            VecStoreError::Internal("Failed to acquire read lock".into())
        })?;

        store.get(vector_id).cloned().ok_or_else(|| {
            VecStoreError::NotFound(format!("Vector {} not found in lineage", vector_id))
        })
    }

    /// Get full lineage tree (including ancestors)
    pub fn get_lineage_tree(&self, vector_id: &str) -> Result<Vec<VectorLineage>, VecStoreError> {
        let store = self.lineage.read().map_err(|_| {
            VecStoreError::Internal("Failed to acquire read lock".into())
        })?;

        let mut result = Vec::new();
        let mut to_visit = vec![vector_id.to_string()];
        let mut visited = std::collections::HashSet::new();

        while let Some(id) = to_visit.pop() {
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id.clone());

            if let Some(lineage) = store.get(&id) {
                result.push(lineage.clone());
                for parent_id in &lineage.parent_ids {
                    if !visited.contains(parent_id) {
                        to_visit.push(parent_id.clone());
                    }
                }
            }
        }

        Ok(result)
    }

    /// Add compliance tags to a vector
    pub fn add_compliance_tags(
        &self,
        vector_id: &str,
        tags: Vec<String>,
    ) -> Result<(), VecStoreError> {
        let mut store = self.lineage.write().map_err(|_| {
            VecStoreError::Internal("Failed to acquire write lock".into())
        })?;

        let lineage = store.get_mut(vector_id).ok_or_else(|| {
            VecStoreError::NotFound(format!("Vector {} not found in lineage", vector_id))
        })?;

        lineage.compliance_tags.extend(tags);
        lineage.compliance_tags.sort();
        lineage.compliance_tags.dedup();

        Ok(())
    }

    /// Add a retention policy
    pub fn add_retention_policy(&mut self, policy: RetentionPolicy) {
        self.retention_policies.insert(policy.name.clone(), policy);
    }

    /// Check retention and get vectors that need action
    pub fn check_retention(&self) -> Result<Vec<RetentionRecord>, VecStoreError> {
        let store = self.lineage.read().map_err(|_| {
            VecStoreError::Internal("Failed to acquire read lock".into())
        })?;

        let now = Utc::now();
        let mut records = Vec::new();

        for (id, lineage) in store.iter() {
            for (policy_name, policy) in &self.retention_policies {
                // Check if policy applies to this vector
                let applies = policy.applies_to_tags.is_empty()
                    || lineage.compliance_tags.iter().any(|t| policy.applies_to_tags.contains(t));

                if !applies {
                    continue;
                }

                let age = now.signed_duration_since(lineage.origin.created_at);
                let max_age = chrono::Duration::days(policy.max_age_days as i64);

                let expires_at = if policy.max_age_days > 0 {
                    Some(lineage.origin.created_at + max_age)
                } else {
                    None
                };

                let is_expired = policy.max_age_days > 0 && age > max_age;

                records.push(RetentionRecord {
                    vector_id: id.clone(),
                    created_at: lineage.origin.created_at,
                    policy: policy_name.clone(),
                    expires_at,
                    is_expired,
                });
            }
        }

        Ok(records)
    }

    /// Generate a compliance report
    pub fn generate_compliance_report(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Result<ComplianceReport, VecStoreError> {
        let store = self.lineage.read().map_err(|_| {
            VecStoreError::Internal("Failed to acquire read lock".into())
        })?;

        let mut by_source_type: HashMap<String, usize> = HashMap::new();
        let mut by_model: HashMap<String, usize> = HashMap::new();
        let mut total_transformations = 0;
        let mut access_by_type: HashMap<String, usize> = HashMap::new();
        let mut total_accesses = 0;
        let mut accessor_set = std::collections::HashSet::new();
        let mut access_counts: HashMap<String, usize> = HashMap::new();

        for (id, lineage) in store.iter() {
            // Count by source type
            let source_type = format!("{:?}", lineage.origin.source_type);
            *by_source_type.entry(source_type).or_insert(0) += 1;

            // Count by model
            if let Some(ref model) = lineage.origin.embedding_model {
                *by_model.entry(model.name.clone()).or_insert(0) += 1;
            }

            // Count transformations
            total_transformations += lineage.transformations.len();

            // Count accesses in period
            for access in &lineage.access_log {
                if access.timestamp >= period_start && access.timestamp <= period_end {
                    total_accesses += 1;
                    let access_type = format!("{:?}", access.access_type);
                    *access_by_type.entry(access_type).or_insert(0) += 1;

                    if let Some(ref accessor) = access.accessor {
                        accessor_set.insert(accessor.clone());
                    }

                    *access_counts.entry(id.clone()).or_insert(0) += 1;
                }
            }
        }

        // Get top accessed
        let mut top_accessed: Vec<_> = access_counts.into_iter().collect();
        top_accessed.sort_by(|a, b| b.1.cmp(&a.1));
        top_accessed.truncate(10);

        // Check retention
        let retention_status = self.check_retention()?;

        // Find issues
        let mut issues = Vec::new();

        // Check for expired vectors
        let expired: Vec<_> = retention_status.iter()
            .filter(|r| r.is_expired)
            .map(|r| r.vector_id.clone())
            .collect();

        if !expired.is_empty() {
            issues.push(ComplianceIssue {
                severity: IssueSeverity::Warning,
                category: "Retention".to_string(),
                description: format!("{} vectors have exceeded their retention period", expired.len()),
                affected_vectors: expired,
                recommendation: "Review and apply retention policies".to_string(),
            });
        }

        // Check for vectors without model info
        let missing_model: Vec<_> = store.iter()
            .filter(|(_, l)| l.origin.embedding_model.is_none())
            .map(|(id, _)| id.clone())
            .collect();

        if !missing_model.is_empty() {
            issues.push(ComplianceIssue {
                severity: IssueSeverity::Info,
                category: "Provenance".to_string(),
                description: format!("{} vectors are missing model attribution", missing_model.len()),
                affected_vectors: missing_model,
                recommendation: "Update vectors with embedding model information".to_string(),
            });
        }

        Ok(ComplianceReport {
            generated_at: Utc::now(),
            period_start,
            period_end,
            total_vectors: store.len(),
            by_source_type,
            by_model,
            total_transformations,
            access_stats: AccessStatistics {
                total_accesses,
                by_type: access_by_type,
                unique_accessors: accessor_set.len(),
                top_accessed,
            },
            retention_status,
            issues,
        })
    }

    /// Find vectors by source
    pub fn find_by_source(&self, source_id: &str) -> Result<Vec<String>, VecStoreError> {
        let store = self.lineage.read().map_err(|_| {
            VecStoreError::Internal("Failed to acquire read lock".into())
        })?;

        let results: Vec<String> = store.iter()
            .filter(|(_, lineage)| lineage.origin.source_id == source_id)
            .map(|(id, _)| id.clone())
            .collect();

        Ok(results)
    }

    /// Find vectors by model
    pub fn find_by_model(&self, model_name: &str) -> Result<Vec<String>, VecStoreError> {
        let store = self.lineage.read().map_err(|_| {
            VecStoreError::Internal("Failed to acquire read lock".into())
        })?;

        let results: Vec<String> = store.iter()
            .filter(|(_, lineage)| {
                lineage.origin.embedding_model.as_ref()
                    .map(|m| m.name == model_name)
                    .unwrap_or(false)
            })
            .map(|(id, _)| id.clone())
            .collect();

        Ok(results)
    }

    /// Get vectors with specific compliance tag
    pub fn find_by_tag(&self, tag: &str) -> Result<Vec<String>, VecStoreError> {
        let store = self.lineage.read().map_err(|_| {
            VecStoreError::Internal("Failed to acquire read lock".into())
        })?;

        let results: Vec<String> = store.iter()
            .filter(|(_, lineage)| lineage.compliance_tags.contains(&tag.to_string()))
            .map(|(id, _)| id.clone())
            .collect();

        Ok(results)
    }

    /// Export lineage data for a vector
    pub fn export_lineage(&self, vector_id: &str) -> Result<String, VecStoreError> {
        let lineage = self.get_lineage(vector_id)?;
        serde_json::to_string_pretty(&lineage).map_err(|e| {
            VecStoreError::Serialization(e.to_string())
        })
    }

    /// Import lineage data
    pub fn import_lineage(&self, json: &str) -> Result<String, VecStoreError> {
        let lineage: VectorLineage = serde_json::from_str(json).map_err(|e| {
            VecStoreError::Serialization(e.to_string())
        })?;

        let id = lineage.vector_id.clone();

        let mut store = self.lineage.write().map_err(|_| {
            VecStoreError::Internal("Failed to acquire write lock".into())
        })?;

        store.insert(id.clone(), lineage);
        Ok(id)
    }

    /// Delete lineage for a vector
    pub fn delete(&self, vector_id: &str) -> Result<(), VecStoreError> {
        let mut store = self.lineage.write().map_err(|_| {
            VecStoreError::Internal("Failed to acquire write lock".into())
        })?;

        store.remove(vector_id);
        Ok(())
    }

    /// Get total count of tracked vectors
    pub fn count(&self) -> Result<usize, VecStoreError> {
        let store = self.lineage.read().map_err(|_| {
            VecStoreError::Internal("Failed to acquire read lock".into())
        })?;

        Ok(store.len())
    }
}

impl Default for LineageTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_origin_builder() {
        let origin = VectorOrigin::new("doc123")
            .with_type(SourceType::Document)
            .from_model("openai/text-embedding-3-large")
            .from_source("documents/report.pdf", "page_3")
            .created_by("indexer_v1");

        assert_eq!(origin.source_id, "doc123");
        assert_eq!(origin.source_type, SourceType::Document);
        assert!(origin.embedding_model.is_some());
        assert_eq!(origin.source_path, Some("documents/report.pdf".to_string()));
        assert_eq!(origin.source_location, Some("page_3".to_string()));
    }

    #[test]
    fn test_register_and_get_lineage() {
        let tracker = LineageTracker::new();

        let origin = VectorOrigin::new("doc1")
            .from_model("test-model");

        tracker.register("vec1", origin).unwrap();

        let lineage = tracker.get_lineage("vec1").unwrap();
        assert_eq!(lineage.vector_id, "vec1");
        assert_eq!(lineage.origin.source_id, "doc1");
        assert_eq!(lineage.version, 1);
    }

    #[test]
    fn test_add_transformation() {
        let tracker = LineageTracker::new();

        let origin = VectorOrigin::new("doc1");
        tracker.register("vec1", origin).unwrap();

        tracker.add_transformation("vec1", TransformationType::Normalized).unwrap();
        tracker.add_transformation("vec1", TransformationType::Quantized { bits: 8 }).unwrap();

        let lineage = tracker.get_lineage("vec1").unwrap();
        assert_eq!(lineage.transformations.len(), 2);
        assert_eq!(lineage.version, 3);
    }

    #[test]
    fn test_derived_vectors() {
        let tracker = LineageTracker::new();

        // Register parent vectors
        tracker.register("parent1", VectorOrigin::new("doc1")).unwrap();
        tracker.register("parent2", VectorOrigin::new("doc2")).unwrap();

        // Register derived vector
        let origin = VectorOrigin::new("derived_source");
        tracker.register_derived(
            "child1",
            origin,
            vec!["parent1".to_string(), "parent2".to_string()],
        ).unwrap();

        // Check child
        let child_lineage = tracker.get_lineage("child1").unwrap();
        assert_eq!(child_lineage.parent_ids.len(), 2);

        // Check parents have child reference
        let parent1 = tracker.get_lineage("parent1").unwrap();
        assert!(parent1.child_ids.contains(&"child1".to_string()));
    }

    #[test]
    fn test_access_logging() {
        let tracker = LineageTracker::new();

        let origin = VectorOrigin::new("doc1");
        tracker.register("vec1", origin).unwrap();

        tracker.log_access(
            "vec1",
            AccessType::SearchResult,
            Some("user123".to_string()),
            Some("query456".to_string()),
        ).unwrap();

        let lineage = tracker.get_lineage("vec1").unwrap();
        assert_eq!(lineage.access_log.len(), 1);
        assert_eq!(lineage.access_log[0].access_type, AccessType::SearchResult);
    }

    #[test]
    fn test_compliance_tags() {
        let tracker = LineageTracker::new();

        let origin = VectorOrigin::new("doc1");
        tracker.register("vec1", origin).unwrap();

        tracker.add_compliance_tags("vec1", vec![
            "GDPR".to_string(),
            "PII".to_string(),
        ]).unwrap();

        let lineage = tracker.get_lineage("vec1").unwrap();
        assert!(lineage.compliance_tags.contains(&"GDPR".to_string()));
        assert!(lineage.compliance_tags.contains(&"PII".to_string()));
    }

    #[test]
    fn test_find_by_source() {
        let tracker = LineageTracker::new();

        tracker.register("vec1", VectorOrigin::new("doc1")).unwrap();
        tracker.register("vec2", VectorOrigin::new("doc1")).unwrap();
        tracker.register("vec3", VectorOrigin::new("doc2")).unwrap();

        let results = tracker.find_by_source("doc1").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_find_by_model() {
        let tracker = LineageTracker::new();

        tracker.register("vec1", VectorOrigin::new("doc1").from_model("model-a")).unwrap();
        tracker.register("vec2", VectorOrigin::new("doc2").from_model("model-a")).unwrap();
        tracker.register("vec3", VectorOrigin::new("doc3").from_model("model-b")).unwrap();

        let results = tracker.find_by_model("model-a").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_lineage_tree() {
        let tracker = LineageTracker::new();

        // Build a tree: grandparent -> parent -> child
        tracker.register("gp", VectorOrigin::new("gp_source")).unwrap();
        tracker.register_derived("parent", VectorOrigin::new("parent_source"), vec!["gp".to_string()]).unwrap();
        tracker.register_derived("child", VectorOrigin::new("child_source"), vec!["parent".to_string()]).unwrap();

        let tree = tracker.get_lineage_tree("child").unwrap();
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_export_import() {
        let tracker = LineageTracker::new();

        let origin = VectorOrigin::new("doc1")
            .from_model("test-model")
            .with_type(SourceType::Document);
        tracker.register("vec1", origin).unwrap();
        tracker.add_transformation("vec1", TransformationType::Normalized).unwrap();

        // Export
        let json = tracker.export_lineage("vec1").unwrap();
        assert!(json.contains("vec1"));

        // Import to new tracker
        let tracker2 = LineageTracker::new();
        let imported_id = tracker2.import_lineage(&json).unwrap();
        assert_eq!(imported_id, "vec1");

        let lineage = tracker2.get_lineage("vec1").unwrap();
        assert_eq!(lineage.transformations.len(), 1);
    }

    #[test]
    fn test_compliance_report() {
        let tracker = LineageTracker::new();

        tracker.register("vec1", VectorOrigin::new("doc1").from_model("model-a")).unwrap();
        tracker.register("vec2", VectorOrigin::new("doc2").from_model("model-b")).unwrap();
        tracker.register("vec3", VectorOrigin::new("doc3")).unwrap(); // No model

        tracker.log_access("vec1", AccessType::Read, Some("user1".to_string()), None).unwrap();
        tracker.log_access("vec1", AccessType::SearchResult, Some("user2".to_string()), None).unwrap();

        let start = Utc::now() - chrono::Duration::hours(1);
        let end = Utc::now() + chrono::Duration::hours(1);

        let report = tracker.generate_compliance_report(start, end).unwrap();

        assert_eq!(report.total_vectors, 3);
        assert_eq!(report.access_stats.total_accesses, 2);
        assert_eq!(report.access_stats.unique_accessors, 2);
        // Should have one issue for missing model
        assert!(report.issues.iter().any(|i| i.category == "Provenance"));
    }
}
