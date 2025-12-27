// Data Lineage Tracking - Track vector provenance and transformations
// Provides audit trail, impact analysis, and debugging capabilities

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// Lineage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageConfig {
    /// Enable lineage tracking
    pub enabled: bool,
    /// Maximum lineage depth to track
    pub max_depth: usize,
    /// Retention period for lineage records
    pub retention_days: u32,
    /// Track transformations
    pub track_transformations: bool,
    /// Track access patterns
    pub track_access: bool,
    /// Store full vectors in lineage (expensive)
    pub store_vectors: bool,
}

impl Default for LineageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_depth: 100,
            retention_days: 90,
            track_transformations: true,
            track_access: false,
            store_vectors: false,
        }
    }
}

/// Unique identifier for a vector
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct VectorId {
    /// Collection name
    pub collection: String,
    /// Vector ID within collection
    pub id: String,
    /// Version (for versioned vectors)
    pub version: Option<u64>,
}

impl VectorId {
    pub fn new(collection: &str, id: &str) -> Self {
        Self {
            collection: collection.to_string(),
            id: id.to_string(),
            version: None,
        }
    }

    pub fn with_version(collection: &str, id: &str, version: u64) -> Self {
        Self {
            collection: collection.to_string(),
            id: id.to_string(),
            version: Some(version),
        }
    }
}

/// Source of a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorSource {
    /// Direct insertion via API
    DirectInsert {
        user: Option<String>,
        client_ip: Option<String>,
        api_version: String,
    },
    /// Imported from file
    FileImport {
        file_path: String,
        file_format: String,
        line_number: Option<u64>,
    },
    /// Generated from text embedding
    TextEmbedding {
        model: String,
        text_hash: String,
        text_length: usize,
    },
    /// Generated from image embedding
    ImageEmbedding {
        model: String,
        image_hash: String,
        image_dimensions: (u32, u32),
    },
    /// Derived from other vectors
    Derived {
        parent_ids: Vec<VectorId>,
        operation: DerivationOperation,
    },
    /// Migrated from another database
    Migration {
        source_db: String,
        source_id: String,
        migration_id: String,
    },
    /// Replicated from another node
    Replication {
        source_node: String,
        replication_timestamp: u64,
    },
    /// Unknown source
    Unknown,
}

/// Operations that derive new vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DerivationOperation {
    /// Average of multiple vectors
    Average,
    /// Concatenation of vectors
    Concatenation,
    /// Dimensionality reduction
    DimensionalityReduction { method: String, target_dims: usize },
    /// Quantization
    Quantization { method: String },
    /// Normalization
    Normalization { method: String },
    /// Custom transformation
    Custom { name: String, parameters: HashMap<String, String> },
}

/// Transformation applied to a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformation {
    /// Transformation ID
    pub id: String,
    /// Transformation type
    pub transform_type: TransformationType,
    /// Input vector IDs
    pub inputs: Vec<VectorId>,
    /// Output vector IDs
    pub outputs: Vec<VectorId>,
    /// Timestamp
    pub timestamp: u64,
    /// User who applied transformation
    pub user: Option<String>,
    /// Parameters used
    pub parameters: HashMap<String, String>,
    /// Duration of transformation
    pub duration_ms: u64,
}

/// Types of transformations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationType {
    Insert,
    Update,
    Delete,
    Normalize,
    Quantize,
    ReduceDimensions,
    Augment,
    Merge,
    Split,
    Reindex,
    Custom(String),
}

/// Lineage record for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageRecord {
    /// Vector ID
    pub vector_id: VectorId,
    /// Original source
    pub source: VectorSource,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
    /// Applied transformations (in order)
    pub transformations: Vec<String>, // Transformation IDs
    /// Metadata at creation
    pub original_metadata: Option<serde_json::Value>,
    /// Current metadata
    pub current_metadata: Option<serde_json::Value>,
    /// Original dimensions
    pub original_dimensions: usize,
    /// Current dimensions
    pub current_dimensions: usize,
    /// Original vector (if stored)
    pub original_vector: Option<Vec<f32>>,
    /// Checksum of current vector
    pub vector_checksum: String,
    /// Tags
    pub tags: HashSet<String>,
    /// Quality metrics
    pub quality: Option<QualityMetrics>,
}

/// Quality metrics for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Magnitude/norm
    pub magnitude: f32,
    /// Sparsity (% of zeros)
    pub sparsity: f32,
    /// Entropy estimate
    pub entropy: f32,
    /// Nearest neighbor distance (if known)
    pub nn_distance: Option<f32>,
    /// Cluster assignment (if clustered)
    pub cluster_id: Option<u32>,
}

/// Lineage tracker
pub struct LineageTracker {
    config: LineageConfig,
    /// Vector lineage records
    records: RwLock<HashMap<VectorId, LineageRecord>>,
    /// Transformation records
    transformations: RwLock<HashMap<String, Transformation>>,
    /// Forward dependencies: vector -> vectors derived from it
    forward_deps: RwLock<HashMap<VectorId, HashSet<VectorId>>>,
    /// Backward dependencies: vector -> vectors it was derived from
    backward_deps: RwLock<HashMap<VectorId, HashSet<VectorId>>>,
    /// Access log (if enabled)
    access_log: RwLock<VecDeque<AccessRecord>>,
    /// Metrics
    metrics: LineageMetrics,
}

/// Access record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRecord {
    pub vector_id: VectorId,
    pub access_type: AccessType,
    pub timestamp: u64,
    pub user: Option<String>,
    pub query_id: Option<String>,
}

/// Type of access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessType {
    Read,
    Search,
    Update,
    Delete,
}

/// Lineage metrics
#[derive(Debug, Default)]
struct LineageMetrics {
    records_created: std::sync::atomic::AtomicU64,
    transformations_recorded: std::sync::atomic::AtomicU64,
    lineage_queries: std::sync::atomic::AtomicU64,
    impact_analyses: std::sync::atomic::AtomicU64,
}

impl LineageTracker {
    /// Create a new lineage tracker
    pub fn new(config: LineageConfig) -> Self {
        Self {
            config,
            records: RwLock::new(HashMap::new()),
            transformations: RwLock::new(HashMap::new()),
            forward_deps: RwLock::new(HashMap::new()),
            backward_deps: RwLock::new(HashMap::new()),
            access_log: RwLock::new(VecDeque::with_capacity(10000)),
            metrics: LineageMetrics::default(),
        }
    }

    /// Record vector creation
    pub fn record_creation(
        &self,
        vector_id: VectorId,
        source: VectorSource,
        dimensions: usize,
        metadata: Option<serde_json::Value>,
        vector: Option<&[f32]>,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let now = current_timestamp();
        let checksum = vector.map(|v| compute_checksum(v)).unwrap_or_default();

        let record = LineageRecord {
            vector_id: vector_id.clone(),
            source: source.clone(),
            created_at: now,
            modified_at: now,
            transformations: Vec::new(),
            original_metadata: metadata.clone(),
            current_metadata: metadata,
            original_dimensions: dimensions,
            current_dimensions: dimensions,
            original_vector: if self.config.store_vectors {
                vector.map(|v| v.to_vec())
            } else {
                None
            },
            vector_checksum: checksum,
            tags: HashSet::new(),
            quality: vector.map(|v| compute_quality_metrics(v)),
        };

        // Handle derived sources
        if let VectorSource::Derived { ref parent_ids, .. } = source {
            let mut forward = self.forward_deps.write().unwrap();
            let mut backward = self.backward_deps.write().unwrap();

            for parent_id in parent_ids {
                forward.entry(parent_id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(vector_id.clone());
            }

            backward.insert(vector_id.clone(), parent_ids.iter().cloned().collect());
        }

        self.records.write().unwrap().insert(vector_id, record);

        use std::sync::atomic::Ordering;
        self.metrics.records_created.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Record a transformation
    pub fn record_transformation(&self, transformation: Transformation) -> Result<()> {
        if !self.config.enabled || !self.config.track_transformations {
            return Ok(());
        }

        let transform_id = transformation.id.clone();

        // Update input records
        {
            let mut records = self.records.write().unwrap();
            for input_id in &transformation.inputs {
                if let Some(record) = records.get_mut(input_id) {
                    record.transformations.push(transform_id.clone());
                    record.modified_at = current_timestamp();
                }
            }
        }

        // Track dependencies
        {
            let mut forward = self.forward_deps.write().unwrap();
            let mut backward = self.backward_deps.write().unwrap();

            for output_id in &transformation.outputs {
                backward.insert(
                    output_id.clone(),
                    transformation.inputs.iter().cloned().collect(),
                );
            }

            // Rust 1.92: Use or_default() for cleaner entry API
            for input_id in &transformation.inputs {
                let deps = forward.entry(input_id.clone()).or_default();
                deps.extend(transformation.outputs.iter().cloned());
            }
        }

        self.transformations.write().unwrap().insert(transform_id, transformation);

        use std::sync::atomic::Ordering;
        self.metrics.transformations_recorded.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Record vector access
    pub fn record_access(
        &self,
        vector_id: VectorId,
        access_type: AccessType,
        user: Option<String>,
        query_id: Option<String>,
    ) -> Result<()> {
        if !self.config.enabled || !self.config.track_access {
            return Ok(());
        }

        let record = AccessRecord {
            vector_id,
            access_type,
            timestamp: current_timestamp(),
            user,
            query_id,
        };

        let mut log = self.access_log.write().unwrap();
        log.push_back(record);

        // Limit log size
        while log.len() > 10000 {
            log.pop_front();
        }

        Ok(())
    }

    /// Get lineage for a vector
    pub fn get_lineage(&self, vector_id: &VectorId) -> Option<LineageRecord> {
        use std::sync::atomic::Ordering;
        self.metrics.lineage_queries.fetch_add(1, Ordering::Relaxed);

        self.records.read().unwrap().get(vector_id).cloned()
    }

    /// Get full ancestry (all ancestors up to max_depth)
    pub fn get_ancestry(&self, vector_id: &VectorId) -> Vec<VectorId> {
        let mut ancestors = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(vector_id.clone());

        let backward = self.backward_deps.read().unwrap();

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(parents) = backward.get(&current) {
                for parent in parents {
                    if !visited.contains(parent) && ancestors.len() < self.config.max_depth {
                        ancestors.push(parent.clone());
                        queue.push_back(parent.clone());
                    }
                }
            }
        }

        ancestors
    }

    /// Get all descendants (vectors derived from this one)
    pub fn get_descendants(&self, vector_id: &VectorId) -> Vec<VectorId> {
        let mut descendants = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(vector_id.clone());

        let forward = self.forward_deps.read().unwrap();

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(children) = forward.get(&current) {
                for child in children {
                    if !visited.contains(child) && descendants.len() < self.config.max_depth {
                        descendants.push(child.clone());
                        queue.push_back(child.clone());
                    }
                }
            }
        }

        descendants
    }

    /// Impact analysis: what would be affected if this vector changes/is deleted
    pub fn impact_analysis(&self, vector_id: &VectorId) -> ImpactReport {
        use std::sync::atomic::Ordering;
        self.metrics.impact_analyses.fetch_add(1, Ordering::Relaxed);

        let descendants = self.get_descendants(vector_id);

        let records = self.records.read().unwrap();
        let mut affected_collections: HashSet<String> = HashSet::new();
        let mut affected_by_type: HashMap<String, usize> = HashMap::new();

        for desc in &descendants {
            affected_collections.insert(desc.collection.clone());

            if let Some(record) = records.get(desc) {
                let source_type = match &record.source {
                    VectorSource::DirectInsert { .. } => "direct",
                    VectorSource::FileImport { .. } => "file_import",
                    VectorSource::TextEmbedding { .. } => "text_embedding",
                    VectorSource::ImageEmbedding { .. } => "image_embedding",
                    VectorSource::Derived { .. } => "derived",
                    VectorSource::Migration { .. } => "migration",
                    VectorSource::Replication { .. } => "replication",
                    VectorSource::Unknown => "unknown",
                };
                *affected_by_type.entry(source_type.to_string()).or_insert(0) += 1;
            }
        }

        ImpactReport {
            source_vector: vector_id.clone(),
            total_affected: descendants.len(),
            affected_vectors: descendants,
            affected_collections: affected_collections.into_iter().collect(),
            affected_by_source_type: affected_by_type,
        }
    }

    /// Find vectors by source type
    pub fn find_by_source(&self, source_type: &str) -> Vec<VectorId> {
        let records = self.records.read().unwrap();
        records.iter()
            .filter(|(_, record)| {
                let record_type = match &record.source {
                    VectorSource::DirectInsert { .. } => "direct",
                    VectorSource::FileImport { .. } => "file_import",
                    VectorSource::TextEmbedding { .. } => "text_embedding",
                    VectorSource::ImageEmbedding { .. } => "image_embedding",
                    VectorSource::Derived { .. } => "derived",
                    VectorSource::Migration { .. } => "migration",
                    VectorSource::Replication { .. } => "replication",
                    VectorSource::Unknown => "unknown",
                };
                record_type == source_type
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Find vectors by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<VectorId> {
        let records = self.records.read().unwrap();
        records.iter()
            .filter(|(_, record)| record.tags.contains(tag))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Add tag to vector
    pub fn add_tag(&self, vector_id: &VectorId, tag: &str) -> Result<()> {
        let mut records = self.records.write().unwrap();
        if let Some(record) = records.get_mut(vector_id) {
            record.tags.insert(tag.to_string());
            record.modified_at = current_timestamp();
            Ok(())
        } else {
            Err(VecStoreError::NotFound(format!("Vector {:?} not found", vector_id)))
        }
    }

    /// Get transformation history for a vector
    pub fn get_transformation_history(&self, vector_id: &VectorId) -> Vec<Transformation> {
        let records = self.records.read().unwrap();
        let transformations = self.transformations.read().unwrap();

        if let Some(record) = records.get(vector_id) {
            record.transformations.iter()
                .filter_map(|id| transformations.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Export lineage as graph (DOT format)
    pub fn export_graph(&self, root_id: &VectorId, depth: usize) -> String {
        let mut dot = String::from("digraph lineage {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box];\n\n");

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((root_id.clone(), 0));

        let backward = self.backward_deps.read().unwrap();
        let forward = self.forward_deps.read().unwrap();
        let records = self.records.read().unwrap();

        while let Some((current, current_depth)) = queue.pop_front() {
            if visited.contains(&current) || current_depth > depth {
                continue;
            }
            visited.insert(current.clone());

            // Node style based on source
            let (color, shape) = if let Some(record) = records.get(&current) {
                match &record.source {
                    VectorSource::DirectInsert { .. } => ("lightblue", "box"),
                    VectorSource::TextEmbedding { .. } => ("lightgreen", "ellipse"),
                    VectorSource::ImageEmbedding { .. } => ("lightyellow", "ellipse"),
                    VectorSource::Derived { .. } => ("lightgray", "diamond"),
                    _ => ("white", "box"),
                }
            } else {
                ("white", "box")
            };

            let node_id = format!("{}_{}", current.collection, current.id);
            dot.push_str(&format!(
                "  \"{}\" [label=\"{}\\n{}\" fillcolor={} style=filled shape={}];\n",
                node_id, current.collection, current.id, color, shape
            ));

            // Backward edges (ancestors)
            if let Some(parents) = backward.get(&current) {
                for parent in parents {
                    if current_depth < depth {
                        queue.push_back((parent.clone(), current_depth + 1));
                    }
                    let parent_node = format!("{}_{}", parent.collection, parent.id);
                    dot.push_str(&format!("  \"{}\" -> \"{}\";\n", parent_node, node_id));
                }
            }

            // Forward edges (descendants)
            if let Some(children) = forward.get(&current) {
                for child in children {
                    if current_depth < depth {
                        queue.push_back((child.clone(), current_depth + 1));
                    }
                    let child_node = format!("{}_{}", child.collection, child.id);
                    dot.push_str(&format!("  \"{}\" -> \"{}\";\n", node_id, child_node));
                }
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Get statistics
    pub fn get_stats(&self) -> LineageStats {
        use std::sync::atomic::Ordering;

        let records = self.records.read().unwrap();
        let transformations = self.transformations.read().unwrap();

        let mut source_counts: HashMap<String, usize> = HashMap::new();
        for record in records.values() {
            let source_type = match &record.source {
                VectorSource::DirectInsert { .. } => "direct",
                VectorSource::FileImport { .. } => "file_import",
                VectorSource::TextEmbedding { .. } => "text_embedding",
                VectorSource::ImageEmbedding { .. } => "image_embedding",
                VectorSource::Derived { .. } => "derived",
                VectorSource::Migration { .. } => "migration",
                VectorSource::Replication { .. } => "replication",
                VectorSource::Unknown => "unknown",
            };
            *source_counts.entry(source_type.to_string()).or_insert(0) += 1;
        }

        LineageStats {
            total_records: records.len(),
            total_transformations: transformations.len(),
            records_by_source: source_counts,
            records_created: self.metrics.records_created.load(Ordering::Relaxed),
            transformations_recorded: self.metrics.transformations_recorded.load(Ordering::Relaxed),
            lineage_queries: self.metrics.lineage_queries.load(Ordering::Relaxed),
            impact_analyses: self.metrics.impact_analyses.load(Ordering::Relaxed),
        }
    }

    /// Cleanup old records based on retention policy
    pub fn cleanup(&self) -> usize {
        let cutoff = current_timestamp() - (self.config.retention_days as u64 * 24 * 60 * 60 * 1000);

        let mut records = self.records.write().unwrap();
        let initial_count = records.len();

        records.retain(|_, record| record.created_at > cutoff);

        initial_count - records.len()
    }
}

/// Impact analysis report
#[derive(Debug, Clone, Serialize)]
pub struct ImpactReport {
    pub source_vector: VectorId,
    pub total_affected: usize,
    pub affected_vectors: Vec<VectorId>,
    pub affected_collections: Vec<String>,
    pub affected_by_source_type: HashMap<String, usize>,
}

/// Lineage statistics
#[derive(Debug, Clone, Serialize)]
pub struct LineageStats {
    pub total_records: usize,
    pub total_transformations: usize,
    pub records_by_source: HashMap<String, usize>,
    pub records_created: u64,
    pub transformations_recorded: u64,
    pub lineage_queries: u64,
    pub impact_analyses: u64,
}

/// Helper function to get current timestamp in milliseconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Compute a checksum for a vector
fn compute_checksum(vector: &[f32]) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    for v in vector {
        v.to_bits().hash(&mut hasher);
    }

    format!("{:016x}", hasher.finish())
}

/// Compute quality metrics for a vector
fn compute_quality_metrics(vector: &[f32]) -> QualityMetrics {
    let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    let zeros = vector.iter().filter(|&&x| x.abs() < 1e-10).count();
    let sparsity = zeros as f32 / vector.len() as f32;

    // Simple entropy estimate based on value distribution
    let mut value_counts: HashMap<i32, usize> = HashMap::new();
    for &v in vector {
        let bucket = (v * 100.0) as i32;
        *value_counts.entry(bucket).or_insert(0) += 1;
    }

    let n = vector.len() as f32;
    let entropy: f32 = value_counts.values()
        .map(|&count| {
            let p = count as f32 / n;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        })
        .sum();

    QualityMetrics {
        magnitude,
        sparsity,
        entropy,
        nn_distance: None,
        cluster_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lineage_creation() {
        let tracker = LineageTracker::new(LineageConfig::default());

        let vector_id = VectorId::new("test_collection", "vec1");
        let source = VectorSource::DirectInsert {
            user: Some("test_user".to_string()),
            client_ip: None,
            api_version: "v1".to_string(),
        };

        tracker.record_creation(
            vector_id.clone(),
            source,
            128,
            None,
            Some(&vec![0.1; 128]),
        ).unwrap();

        let lineage = tracker.get_lineage(&vector_id).unwrap();
        assert_eq!(lineage.original_dimensions, 128);
    }

    #[test]
    fn test_derived_vectors() {
        let tracker = LineageTracker::new(LineageConfig::default());

        // Create parent vectors
        let parent1 = VectorId::new("test", "parent1");
        let parent2 = VectorId::new("test", "parent2");

        tracker.record_creation(
            parent1.clone(),
            VectorSource::DirectInsert {
                user: None, client_ip: None, api_version: "v1".to_string(),
            },
            128, None, None,
        ).unwrap();

        tracker.record_creation(
            parent2.clone(),
            VectorSource::DirectInsert {
                user: None, client_ip: None, api_version: "v1".to_string(),
            },
            128, None, None,
        ).unwrap();

        // Create derived vector
        let child = VectorId::new("test", "child");
        tracker.record_creation(
            child.clone(),
            VectorSource::Derived {
                parent_ids: vec![parent1.clone(), parent2.clone()],
                operation: DerivationOperation::Average,
            },
            128, None, None,
        ).unwrap();

        // Check ancestry
        let ancestors = tracker.get_ancestry(&child);
        assert_eq!(ancestors.len(), 2);
        assert!(ancestors.contains(&parent1));
        assert!(ancestors.contains(&parent2));

        // Check descendants
        let descendants = tracker.get_descendants(&parent1);
        assert_eq!(descendants.len(), 1);
        assert!(descendants.contains(&child));
    }

    #[test]
    fn test_impact_analysis() {
        let tracker = LineageTracker::new(LineageConfig::default());

        let root = VectorId::new("test", "root");
        tracker.record_creation(
            root.clone(),
            VectorSource::DirectInsert {
                user: None, client_ip: None, api_version: "v1".to_string(),
            },
            128, None, None,
        ).unwrap();

        // Create chain of derived vectors
        let level1 = VectorId::new("test", "level1");
        tracker.record_creation(
            level1.clone(),
            VectorSource::Derived {
                parent_ids: vec![root.clone()],
                operation: DerivationOperation::Normalization { method: "l2".to_string() },
            },
            128, None, None,
        ).unwrap();

        let level2 = VectorId::new("test", "level2");
        tracker.record_creation(
            level2.clone(),
            VectorSource::Derived {
                parent_ids: vec![level1.clone()],
                operation: DerivationOperation::Quantization { method: "pq".to_string() },
            },
            128, None, None,
        ).unwrap();

        let report = tracker.impact_analysis(&root);
        assert_eq!(report.total_affected, 2);
    }

    #[test]
    fn test_graph_export() {
        let tracker = LineageTracker::new(LineageConfig::default());

        let parent = VectorId::new("test", "parent");
        let child = VectorId::new("test", "child");

        tracker.record_creation(
            parent.clone(),
            VectorSource::DirectInsert {
                user: None, client_ip: None, api_version: "v1".to_string(),
            },
            128, None, None,
        ).unwrap();

        tracker.record_creation(
            child.clone(),
            VectorSource::Derived {
                parent_ids: vec![parent.clone()],
                operation: DerivationOperation::Average,
            },
            128, None, None,
        ).unwrap();

        let dot = tracker.export_graph(&parent, 2);
        assert!(dot.contains("digraph"));
        assert!(dot.contains("test_parent"));
        assert!(dot.contains("test_child"));
    }
}
