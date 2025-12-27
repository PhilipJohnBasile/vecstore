// Kubernetes Operator - Cloud-native deployment and management
// Implements Custom Resource Definitions (CRDs) for VecStore clusters

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VecStoreError};

/// VecStore cluster specification (CRD spec)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecStoreClusterSpec {
    /// Number of replicas
    pub replicas: u32,
    /// Vector dimensions
    pub dimensions: usize,
    /// Storage class for persistent volumes
    pub storage_class: String,
    /// Storage size per replica
    pub storage_size: String,
    /// Resource requests/limits
    pub resources: ResourceRequirements,
    /// Index configuration
    pub index_config: IndexConfig,
    /// High availability configuration
    pub ha_config: Option<HAConfig>,
    /// Monitoring configuration
    pub monitoring: Option<MonitoringConfig>,
    /// Auto-scaling configuration
    pub autoscaling: Option<AutoscalingConfig>,
    /// Backup configuration
    pub backup: Option<BackupConfig>,
}

/// Resource requirements for pods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub requests: ResourceSpec,
    pub limits: ResourceSpec,
}

/// Resource specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSpec {
    pub cpu: String,
    pub memory: String,
    pub gpu: Option<String>,
}

/// Index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// HNSW M parameter
    pub hnsw_m: usize,
    /// HNSW ef_construction
    pub hnsw_ef_construction: usize,
    /// Enable product quantization
    pub enable_pq: bool,
    /// PQ subvectors
    pub pq_subvectors: Option<usize>,
    /// Distance metric
    pub distance_metric: DistanceMetric,
}

/// Distance metric type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

/// High availability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HAConfig {
    /// Enable multi-zone deployment
    pub multi_zone: bool,
    /// Replication factor
    pub replication_factor: u32,
    /// Enable automatic failover
    pub auto_failover: bool,
    /// Failover timeout
    pub failover_timeout_seconds: u32,
    /// Pod disruption budget
    pub pdb_min_available: u32,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable Prometheus metrics
    pub prometheus_enabled: bool,
    /// Metrics port
    pub metrics_port: u16,
    /// Enable distributed tracing
    pub tracing_enabled: bool,
    /// Jaeger endpoint
    pub jaeger_endpoint: Option<String>,
    /// Enable alerting
    pub alerting_enabled: bool,
    /// Alert rules
    pub alert_rules: Vec<AlertRule>,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub expr: String,
    pub duration: String,
    pub severity: String,
    pub annotations: HashMap<String, String>,
}

/// Autoscaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoscalingConfig {
    /// Minimum replicas
    pub min_replicas: u32,
    /// Maximum replicas
    pub max_replicas: u32,
    /// Target CPU utilization percentage
    pub target_cpu_utilization: u32,
    /// Target memory utilization percentage
    pub target_memory_utilization: Option<u32>,
    /// Target queries per second per replica
    pub target_qps_per_replica: Option<u32>,
    /// Scale down stabilization window
    pub scale_down_stabilization_seconds: u32,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable automatic backups
    pub enabled: bool,
    /// Backup schedule (cron format)
    pub schedule: String,
    /// Retention days
    pub retention_days: u32,
    /// S3/GCS bucket for backups
    pub storage_bucket: String,
    /// Storage prefix
    pub storage_prefix: String,
}

/// Cluster status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecStoreClusterStatus {
    /// Current phase
    pub phase: ClusterPhase,
    /// Ready replicas
    pub ready_replicas: u32,
    /// Total replicas
    pub total_replicas: u32,
    /// Conditions
    pub conditions: Vec<ClusterCondition>,
    /// Leader node
    pub leader: Option<String>,
    /// Last backup time
    pub last_backup: Option<String>,
    /// Vector count
    pub vector_count: u64,
    /// Index size bytes
    pub index_size_bytes: u64,
    /// Current version
    pub version: String,
}

/// Cluster phase
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClusterPhase {
    Pending,
    Creating,
    Running,
    Updating,
    Scaling,
    Failed,
    Terminating,
}

/// Cluster condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCondition {
    pub condition_type: ConditionType,
    pub status: bool,
    pub reason: String,
    pub message: String,
    pub last_transition_time: String,
}

/// Condition type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConditionType {
    Ready,
    Available,
    Progressing,
    Degraded,
    ScalingUp,
    ScalingDown,
    BackupReady,
}

/// Kubernetes operator controller
pub struct VecStoreOperator {
    /// Managed clusters
    clusters: RwLock<HashMap<String, ManagedCluster>>,
    /// Operator configuration
    config: OperatorConfig,
    /// Reconciliation queue
    reconcile_queue: RwLock<Vec<ReconcileRequest>>,
    /// Metrics
    metrics: OperatorMetrics,
}

/// Managed cluster state
#[derive(Debug, Clone)]
struct ManagedCluster {
    /// Cluster name
    name: String,
    /// Namespace
    namespace: String,
    /// Specification
    spec: VecStoreClusterSpec,
    /// Current status
    status: VecStoreClusterStatus,
    /// Last reconcile time
    last_reconcile: Instant,
    /// Generation (spec version)
    generation: u64,
    /// Observed generation
    observed_generation: u64,
}

/// Operator configuration
#[derive(Debug, Clone)]
pub struct OperatorConfig {
    /// Reconciliation interval
    pub reconcile_interval: Duration,
    /// Leader election enabled
    pub leader_election: bool,
    /// Leader lease duration
    pub leader_lease_duration: Duration,
    /// Max concurrent reconciles
    pub max_concurrent_reconciles: usize,
    /// Namespace to watch (None = all namespaces)
    pub watch_namespace: Option<String>,
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            reconcile_interval: Duration::from_secs(30),
            leader_election: true,
            leader_lease_duration: Duration::from_secs(15),
            max_concurrent_reconciles: 3,
            watch_namespace: None,
        }
    }
}

/// Reconcile request
#[derive(Debug, Clone)]
struct ReconcileRequest {
    name: String,
    namespace: String,
    trigger: ReconcileTrigger,
    queued_at: Instant,
}

/// What triggered the reconciliation
#[derive(Debug, Clone)]
enum ReconcileTrigger {
    Create,
    Update,
    Delete,
    Periodic,
    StatusChange,
    ExternalEvent(String),
}

/// Operator metrics
#[derive(Debug, Default)]
struct OperatorMetrics {
    reconcile_total: std::sync::atomic::AtomicU64,
    reconcile_errors: std::sync::atomic::AtomicU64,
    reconcile_duration_sum: std::sync::atomic::AtomicU64,
    clusters_managed: std::sync::atomic::AtomicU64,
}

impl VecStoreOperator {
    /// Create a new operator
    pub fn new(config: OperatorConfig) -> Self {
        Self {
            clusters: RwLock::new(HashMap::new()),
            config,
            reconcile_queue: RwLock::new(Vec::new()),
            metrics: OperatorMetrics::default(),
        }
    }

    /// Handle cluster creation
    pub fn handle_create(&self, name: &str, namespace: &str, spec: VecStoreClusterSpec) -> Result<()> {
        let cluster = ManagedCluster {
            name: name.to_string(),
            namespace: namespace.to_string(),
            spec,
            status: VecStoreClusterStatus {
                phase: ClusterPhase::Pending,
                ready_replicas: 0,
                total_replicas: 0,
                conditions: vec![],
                leader: None,
                last_backup: None,
                vector_count: 0,
                index_size_bytes: 0,
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            last_reconcile: Instant::now(),
            generation: 1,
            observed_generation: 0,
        };

        let key = format!("{}/{}", namespace, name);
        self.clusters.write().unwrap().insert(key.clone(), cluster);

        // Queue reconciliation
        self.queue_reconcile(name, namespace, ReconcileTrigger::Create);

        Ok(())
    }

    /// Handle cluster update
    pub fn handle_update(&self, name: &str, namespace: &str, spec: VecStoreClusterSpec) -> Result<()> {
        let key = format!("{}/{}", namespace, name);
        let mut clusters = self.clusters.write().unwrap();

        if let Some(cluster) = clusters.get_mut(&key) {
            cluster.spec = spec;
            cluster.generation += 1;
            self.queue_reconcile(name, namespace, ReconcileTrigger::Update);
            Ok(())
        } else {
            Err(VecStoreError::NotFound(format!("Cluster {} not found", key)))
        }
    }

    /// Handle cluster deletion
    pub fn handle_delete(&self, name: &str, namespace: &str) -> Result<()> {
        let key = format!("{}/{}", namespace, name);
        self.queue_reconcile(name, namespace, ReconcileTrigger::Delete);

        // Mark for deletion (actual removal after cleanup)
        if let Some(cluster) = self.clusters.write().unwrap().get_mut(&key) {
            cluster.status.phase = ClusterPhase::Terminating;
        }

        Ok(())
    }

    /// Queue a reconciliation request
    fn queue_reconcile(&self, name: &str, namespace: &str, trigger: ReconcileTrigger) {
        let request = ReconcileRequest {
            name: name.to_string(),
            namespace: namespace.to_string(),
            trigger,
            queued_at: Instant::now(),
        };
        self.reconcile_queue.write().unwrap().push(request);
    }

    /// Run the reconciliation loop
    pub fn reconcile(&self, name: &str, namespace: &str) -> Result<ReconcileResult> {
        let key = format!("{}/{}", namespace, name);
        let start = Instant::now();

        let result = self.do_reconcile(&key);

        // Update metrics
        use std::sync::atomic::Ordering;
        self.metrics.reconcile_total.fetch_add(1, Ordering::Relaxed);
        self.metrics.reconcile_duration_sum.fetch_add(
            start.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );

        if result.is_err() {
            self.metrics.reconcile_errors.fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    fn do_reconcile(&self, key: &str) -> Result<ReconcileResult> {
        let mut clusters = self.clusters.write().unwrap();
        let cluster = clusters.get_mut(key)
            .ok_or_else(|| VecStoreError::NotFound(format!("Cluster {} not found", key)))?;

        // Check if deletion is requested
        if cluster.status.phase == ClusterPhase::Terminating {
            return self.reconcile_delete(cluster);
        }

        // Reconcile the cluster
        let actions = self.compute_actions(cluster)?;

        // Execute actions
        for action in &actions {
            self.execute_action(cluster, action)?;
        }

        // Update status
        cluster.last_reconcile = Instant::now();
        cluster.observed_generation = cluster.generation;

        Ok(ReconcileResult {
            requeue: !actions.is_empty(),
            requeue_after: if actions.is_empty() {
                Some(self.config.reconcile_interval)
            } else {
                Some(Duration::from_secs(5))
            },
            actions_taken: actions,
        })
    }

    fn compute_actions(&self, cluster: &ManagedCluster) -> Result<Vec<ReconcileAction>> {
        let mut actions = Vec::new();

        // Check replica count
        if cluster.status.ready_replicas < cluster.spec.replicas {
            let needed = cluster.spec.replicas - cluster.status.ready_replicas;
            actions.push(ReconcileAction::ScaleUp { count: needed });
        } else if cluster.status.ready_replicas > cluster.spec.replicas {
            let excess = cluster.status.ready_replicas - cluster.spec.replicas;
            actions.push(ReconcileAction::ScaleDown { count: excess });
        }

        // Check if index config changed
        if cluster.generation > cluster.observed_generation {
            actions.push(ReconcileAction::UpdateConfig);
        }

        // Check HA configuration
        if let Some(ref ha) = cluster.spec.ha_config {
            if ha.auto_failover && cluster.status.leader.is_none() && cluster.status.ready_replicas > 0 {
                actions.push(ReconcileAction::ElectLeader);
            }
        }

        // Check backup schedule (Rust 1.92 if-let chain)
        if let Some(ref backup) = cluster.spec.backup
            && backup.enabled
            && self.backup_due(cluster)
        {
            actions.push(ReconcileAction::TriggerBackup);
        }

        Ok(actions)
    }

    fn execute_action(&self, cluster: &mut ManagedCluster, action: &ReconcileAction) -> Result<()> {
        match action {
            ReconcileAction::ScaleUp { count } => {
                cluster.status.phase = ClusterPhase::Scaling;
                cluster.status.total_replicas += count;
                // In real operator: create StatefulSet pods
                cluster.status.ready_replicas += count;
                cluster.status.phase = ClusterPhase::Running;

                self.add_condition(cluster, ConditionType::ScalingUp, true,
                    "ScalingUp", &format!("Scaling up by {} replicas", count));
            }
            ReconcileAction::ScaleDown { count } => {
                cluster.status.phase = ClusterPhase::Scaling;
                cluster.status.total_replicas -= count;
                cluster.status.ready_replicas -= count;
                cluster.status.phase = ClusterPhase::Running;

                self.add_condition(cluster, ConditionType::ScalingDown, true,
                    "ScalingDown", &format!("Scaling down by {} replicas", count));
            }
            ReconcileAction::UpdateConfig => {
                cluster.status.phase = ClusterPhase::Updating;
                // In real operator: rolling update of pods
                cluster.status.phase = ClusterPhase::Running;

                self.add_condition(cluster, ConditionType::Progressing, true,
                    "ConfigUpdated", "Configuration updated successfully");
            }
            ReconcileAction::ElectLeader => {
                // Simple leader election (in real operator: use Raft or lease)
                cluster.status.leader = Some(format!("{}-0", cluster.name));

                self.add_condition(cluster, ConditionType::Ready, true,
                    "LeaderElected", "Leader election completed");
            }
            ReconcileAction::TriggerBackup => {
                // In real operator: create backup job
                cluster.status.last_backup = Some(chrono::Utc::now().to_rfc3339());

                self.add_condition(cluster, ConditionType::BackupReady, true,
                    "BackupCompleted", "Backup completed successfully");
            }
            ReconcileAction::Repair { reason } => {
                cluster.status.phase = ClusterPhase::Updating;
                // In real operator: repair degraded pods
                cluster.status.phase = ClusterPhase::Running;

                self.add_condition(cluster, ConditionType::Degraded, false,
                    "Repaired", reason);
            }
        }

        Ok(())
    }

    fn reconcile_delete(&self, cluster: &mut ManagedCluster) -> Result<ReconcileResult> {
        // Cleanup resources
        // In real operator: delete StatefulSet, PVCs, Services, etc.

        cluster.status.ready_replicas = 0;
        cluster.status.total_replicas = 0;

        Ok(ReconcileResult {
            requeue: false,
            requeue_after: None,
            actions_taken: vec![ReconcileAction::Repair {
                reason: "Cluster deleted".to_string(),
            }],
        })
    }

    fn backup_due(&self, cluster: &ManagedCluster) -> bool {
        if let Some(ref last) = cluster.status.last_backup {
            if let Ok(last_time) = chrono::DateTime::parse_from_rfc3339(last) {
                let now = chrono::Utc::now();
                let elapsed = now.signed_duration_since(last_time);
                // Simple check: backup if more than 24 hours
                return elapsed.num_hours() >= 24;
            }
        }
        true // No backup yet
    }

    fn add_condition(&self, cluster: &mut ManagedCluster, ctype: ConditionType,
                     status: bool, reason: &str, message: &str) {
        // Remove existing condition of same type
        cluster.status.conditions.retain(|c| c.condition_type != ctype);

        cluster.status.conditions.push(ClusterCondition {
            condition_type: ctype,
            status,
            reason: reason.to_string(),
            message: message.to_string(),
            last_transition_time: chrono::Utc::now().to_rfc3339(),
        });
    }

    /// Get cluster status
    pub fn get_status(&self, name: &str, namespace: &str) -> Option<VecStoreClusterStatus> {
        let key = format!("{}/{}", namespace, name);
        self.clusters.read().unwrap()
            .get(&key)
            .map(|c| c.status.clone())
    }

    /// List all managed clusters
    pub fn list_clusters(&self) -> Vec<(String, String, VecStoreClusterStatus)> {
        self.clusters.read().unwrap()
            .iter()
            .map(|(_, c)| (c.namespace.clone(), c.name.clone(), c.status.clone()))
            .collect()
    }

    /// Get operator metrics
    pub fn get_metrics(&self) -> OperatorMetricsSnapshot {
        use std::sync::atomic::Ordering;
        OperatorMetricsSnapshot {
            reconcile_total: self.metrics.reconcile_total.load(Ordering::Relaxed),
            reconcile_errors: self.metrics.reconcile_errors.load(Ordering::Relaxed),
            avg_reconcile_duration_ms: {
                let total = self.metrics.reconcile_total.load(Ordering::Relaxed);
                let sum = self.metrics.reconcile_duration_sum.load(Ordering::Relaxed);
                if total > 0 { sum / total } else { 0 }
            },
            clusters_managed: self.clusters.read().unwrap().len() as u64,
        }
    }
}

/// Reconciliation result
#[derive(Debug)]
pub struct ReconcileResult {
    /// Whether to requeue
    pub requeue: bool,
    /// Requeue after duration
    pub requeue_after: Option<Duration>,
    /// Actions taken
    pub actions_taken: Vec<ReconcileAction>,
}

/// Actions that can be taken during reconciliation
#[derive(Debug, Clone)]
pub enum ReconcileAction {
    ScaleUp { count: u32 },
    ScaleDown { count: u32 },
    UpdateConfig,
    ElectLeader,
    TriggerBackup,
    Repair { reason: String },
}

/// Operator metrics snapshot
#[derive(Debug, Clone)]
pub struct OperatorMetricsSnapshot {
    pub reconcile_total: u64,
    pub reconcile_errors: u64,
    pub avg_reconcile_duration_ms: u64,
    pub clusters_managed: u64,
}

/// Generate Kubernetes manifests
pub struct ManifestGenerator;

impl ManifestGenerator {
    /// Generate CRD manifest
    pub fn generate_crd() -> String {
        r#"apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: vecstoreclusters.vecstore.io
spec:
  group: vecstore.io
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required:
                - replicas
                - dimensions
              properties:
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 100
                dimensions:
                  type: integer
                  minimum: 1
                  maximum: 65536
                storageClass:
                  type: string
                storageSize:
                  type: string
                  pattern: "^[0-9]+(Gi|Ti)$"
                resources:
                  type: object
                  properties:
                    requests:
                      type: object
                      properties:
                        cpu:
                          type: string
                        memory:
                          type: string
                    limits:
                      type: object
                      properties:
                        cpu:
                          type: string
                        memory:
                          type: string
                indexConfig:
                  type: object
                  properties:
                    hnswM:
                      type: integer
                    hnswEfConstruction:
                      type: integer
                    enablePq:
                      type: boolean
                    distanceMetric:
                      type: string
                      enum: ["Cosine", "Euclidean", "DotProduct"]
            status:
              type: object
              properties:
                phase:
                  type: string
                readyReplicas:
                  type: integer
                totalReplicas:
                  type: integer
                leader:
                  type: string
                vectorCount:
                  type: integer
      subresources:
        status: {}
      additionalPrinterColumns:
        - name: Phase
          type: string
          jsonPath: .status.phase
        - name: Ready
          type: string
          jsonPath: .status.readyReplicas
        - name: Vectors
          type: integer
          jsonPath: .status.vectorCount
  scope: Namespaced
  names:
    plural: vecstoreclusters
    singular: vecstorecluster
    kind: VecStoreCluster
    shortNames:
      - vsc
"#.to_string()
    }

    /// Generate StatefulSet manifest for a cluster
    pub fn generate_statefulset(spec: &VecStoreClusterSpec, name: &str, namespace: &str) -> String {
        format!(r#"apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: vecstore
    cluster: {name}
spec:
  serviceName: {name}-headless
  replicas: {replicas}
  selector:
    matchLabels:
      app: vecstore
      cluster: {name}
  template:
    metadata:
      labels:
        app: vecstore
        cluster: {name}
    spec:
      containers:
        - name: vecstore
          image: vecstore/vecstore:latest
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 9090
              name: grpc
            - containerPort: 9091
              name: metrics
          env:
            - name: VECSTORE_DIMENSIONS
              value: "{dimensions}"
            - name: VECSTORE_HNSW_M
              value: "{hnsw_m}"
            - name: VECSTORE_HNSW_EF
              value: "{hnsw_ef}"
          resources:
            requests:
              cpu: {cpu_request}
              memory: {memory_request}
            limits:
              cpu: {cpu_limit}
              memory: {memory_limit}
          volumeMounts:
            - name: data
              mountPath: /data
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: {storage_class}
        resources:
          requests:
            storage: {storage_size}
"#,
            name = name,
            namespace = namespace,
            replicas = spec.replicas,
            dimensions = spec.dimensions,
            hnsw_m = spec.index_config.hnsw_m,
            hnsw_ef = spec.index_config.hnsw_ef_construction,
            cpu_request = spec.resources.requests.cpu,
            memory_request = spec.resources.requests.memory,
            cpu_limit = spec.resources.limits.cpu,
            memory_limit = spec.resources.limits.memory,
            storage_class = spec.storage_class,
            storage_size = spec.storage_size,
        )
    }

    /// Generate headless service manifest
    pub fn generate_headless_service(name: &str, namespace: &str) -> String {
        format!(r#"apiVersion: v1
kind: Service
metadata:
  name: {name}-headless
  namespace: {namespace}
  labels:
    app: vecstore
    cluster: {name}
spec:
  clusterIP: None
  selector:
    app: vecstore
    cluster: {name}
  ports:
    - port: 8080
      name: http
    - port: 9090
      name: grpc
"#,
            name = name,
            namespace = namespace,
        )
    }

    /// Generate client service manifest
    pub fn generate_client_service(name: &str, namespace: &str) -> String {
        format!(r#"apiVersion: v1
kind: Service
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: vecstore
    cluster: {name}
spec:
  type: ClusterIP
  selector:
    app: vecstore
    cluster: {name}
  ports:
    - port: 8080
      name: http
      targetPort: 8080
    - port: 9090
      name: grpc
      targetPort: 9090
"#,
            name = name,
            namespace = namespace,
        )
    }

    /// Generate HPA manifest
    pub fn generate_hpa(spec: &VecStoreClusterSpec, name: &str, namespace: &str) -> Option<String> {
        spec.autoscaling.as_ref().map(|autoscaling| {
            format!(r#"apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {name}
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: {name}
  minReplicas: {min_replicas}
  maxReplicas: {max_replicas}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {cpu_target}
  behavior:
    scaleDown:
      stabilizationWindowSeconds: {stabilization}
"#,
                name = name,
                namespace = namespace,
                min_replicas = autoscaling.min_replicas,
                max_replicas = autoscaling.max_replicas,
                cpu_target = autoscaling.target_cpu_utilization,
                stabilization = autoscaling.scale_down_stabilization_seconds,
            )
        })
    }

    /// Generate PodDisruptionBudget manifest
    pub fn generate_pdb(spec: &VecStoreClusterSpec, name: &str, namespace: &str) -> Option<String> {
        spec.ha_config.as_ref().map(|ha| {
            format!(r#"apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {name}
  namespace: {namespace}
spec:
  minAvailable: {min_available}
  selector:
    matchLabels:
      app: vecstore
      cluster: {name}
"#,
                name = name,
                namespace = namespace,
                min_available = ha.pdb_min_available,
            )
        })
    }

    /// Generate ServiceMonitor for Prometheus
    pub fn generate_service_monitor(name: &str, namespace: &str) -> String {
        format!(r#"apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: vecstore
spec:
  selector:
    matchLabels:
      app: vecstore
      cluster: {name}
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
"#,
            name = name,
            namespace = namespace,
        )
    }
}

/// Helm chart generator
pub struct HelmChartGenerator;

impl HelmChartGenerator {
    /// Generate values.yaml
    pub fn generate_values() -> String {
        r#"# VecStore Helm Chart Values

# Cluster configuration
replicaCount: 3

image:
  repository: vecstore/vecstore
  tag: latest
  pullPolicy: IfNotPresent

# Vector configuration
vector:
  dimensions: 768
  distanceMetric: Cosine

# HNSW index configuration
index:
  hnswM: 16
  hnswEfConstruction: 200
  enablePQ: true
  pqSubvectors: 32

# Resource configuration
resources:
  requests:
    cpu: "1"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "16Gi"

# Storage configuration
persistence:
  enabled: true
  storageClass: ""
  size: 100Gi

# High availability
ha:
  enabled: true
  replicationFactor: 2
  autoFailover: true
  podDisruptionBudget:
    enabled: true
    minAvailable: 2

# Autoscaling
autoscaling:
  enabled: false
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilization: 70

# Monitoring
monitoring:
  prometheus:
    enabled: true
    port: 9091
  tracing:
    enabled: false
    jaegerEndpoint: ""

# Backup configuration
backup:
  enabled: false
  schedule: "0 2 * * *"
  retentionDays: 7
  s3:
    bucket: ""
    prefix: "vecstore-backups"

# Service configuration
service:
  type: ClusterIP
  httpPort: 8080
  grpcPort: 9090

# Ingress configuration
ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: vecstore.local
      paths:
        - path: /
          pathType: Prefix
  tls: []

# Security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

# Node selection
nodeSelector: {}
tolerations: []
affinity: {}
"#.to_string()
    }

    /// Generate Chart.yaml
    pub fn generate_chart() -> String {
        format!(r#"apiVersion: v2
name: vecstore
description: A Helm chart for VecStore vector database
type: application
version: 0.1.0
appVersion: "{}"
keywords:
  - vector-database
  - similarity-search
  - hnsw
  - embeddings
  - rag
home: https://github.com/vecstore/vecstore
sources:
  - https://github.com/vecstore/vecstore
maintainers:
  - name: VecStore Team
    email: team@vecstore.io
"#, env!("CARGO_PKG_VERSION"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_spec() -> VecStoreClusterSpec {
        VecStoreClusterSpec {
            replicas: 3,
            dimensions: 768,
            storage_class: "standard".to_string(),
            storage_size: "100Gi".to_string(),
            resources: ResourceRequirements {
                requests: ResourceSpec {
                    cpu: "1".to_string(),
                    memory: "4Gi".to_string(),
                    gpu: None,
                },
                limits: ResourceSpec {
                    cpu: "4".to_string(),
                    memory: "16Gi".to_string(),
                    gpu: None,
                },
            },
            index_config: IndexConfig {
                hnsw_m: 16,
                hnsw_ef_construction: 200,
                enable_pq: true,
                pq_subvectors: Some(32),
                distance_metric: DistanceMetric::Cosine,
            },
            ha_config: Some(HAConfig {
                multi_zone: true,
                replication_factor: 2,
                auto_failover: true,
                failover_timeout_seconds: 30,
                pdb_min_available: 2,
            }),
            monitoring: None,
            autoscaling: None,
            backup: None,
        }
    }

    #[test]
    fn test_operator_create() {
        let operator = VecStoreOperator::new(OperatorConfig::default());
        let spec = test_spec();

        operator.handle_create("test-cluster", "default", spec).unwrap();

        let status = operator.get_status("test-cluster", "default").unwrap();
        assert_eq!(status.phase, ClusterPhase::Pending);
    }

    #[test]
    fn test_operator_reconcile() {
        let operator = VecStoreOperator::new(OperatorConfig::default());
        let spec = test_spec();

        operator.handle_create("test-cluster", "default", spec).unwrap();

        let result = operator.reconcile("test-cluster", "default").unwrap();
        assert!(result.requeue);

        let status = operator.get_status("test-cluster", "default").unwrap();
        assert_eq!(status.ready_replicas, 3);
    }

    #[test]
    fn test_manifest_generation() {
        let crd = ManifestGenerator::generate_crd();
        assert!(crd.contains("VecStoreCluster"));
        assert!(crd.contains("vecstoreclusters.vecstore.io"));

        let spec = test_spec();
        let sts = ManifestGenerator::generate_statefulset(&spec, "test", "default");
        assert!(sts.contains("replicas: 3"));
        assert!(sts.contains("dimensions"));
    }

    #[test]
    fn test_helm_values() {
        let values = HelmChartGenerator::generate_values();
        assert!(values.contains("replicaCount: 3"));
        assert!(values.contains("dimensions: 768"));
    }
}
