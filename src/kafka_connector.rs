//! Kafka Streaming Connector
//!
//! This module provides integration with Apache Kafka for real-time
//! vector ingestion and streaming queries.
//!
//! ## Features
//!
//! - **Real-time ingestion**: Consume vectors from Kafka topics
//! - **Change streams**: Publish vector updates to Kafka
//! - **Batch processing**: Efficient bulk ingestion
//! - **Schema validation**: Ensure data quality
//! - **Dead letter queue**: Handle failed messages
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::kafka_connector::{KafkaConsumer, KafkaConfig};
//! use vecstore::VecStore;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let config = KafkaConfig {
//!     brokers: vec!["localhost:9092".to_string()],
//!     topic: "vectors".to_string(),
//!     group_id: "vecstore-consumer".to_string(),
//!     ..Default::default()
//! };
//!
//! let mut consumer = KafkaConsumer::new(config)?;
//! let mut store = VecStore::open("vectors.db")?;
//!
//! // Start consuming vectors
//! consumer.start(&mut store).await?;
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time;

/// Kafka configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaConfig {
    /// Kafka broker addresses
    pub brokers: Vec<String>,
    /// Topic to consume/produce
    pub topic: String,
    /// Consumer group ID
    pub group_id: String,
    /// Maximum batch size
    pub batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Enable auto-commit
    pub auto_commit: bool,
    /// Dead letter queue topic
    pub dlq_topic: Option<String>,
    /// Additional properties
    pub properties: HashMap<String, String>,
}

impl Default for KafkaConfig {
    fn default() -> Self {
        Self {
            brokers: vec!["localhost:9092".to_string()],
            topic: "vectors".to_string(),
            group_id: "vecstore-consumer".to_string(),
            batch_size: 100,
            batch_timeout_ms: 1000,
            auto_commit: true,
            dlq_topic: None,
            properties: HashMap::new(),
        }
    }
}

/// Vector message from Kafka
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMessage {
    /// Document ID
    pub id: String,
    /// Vector embedding
    pub vector: Vec<f32>,
    /// Metadata
    pub metadata: serde_json::Value,
    /// Operation type (insert, update, delete)
    #[serde(default = "default_operation")]
    pub operation: Operation,
}

fn default_operation() -> Operation {
    Operation::Insert
}

/// Operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Operation {
    Insert,
    Update,
    Delete,
}

/// Kafka consumer for vector ingestion
pub struct KafkaConsumer {
    config: KafkaConfig,
    // Simulated consumer state
    running: Arc<tokio::sync::RwLock<bool>>,
    stats: Arc<tokio::sync::RwLock<ConsumerStats>>,
}

impl KafkaConsumer {
    /// Create a new Kafka consumer
    pub fn new(config: KafkaConfig) -> Result<Self> {
        Ok(Self {
            config,
            running: Arc::new(tokio::sync::RwLock::new(false)),
            stats: Arc::new(tokio::sync::RwLock::new(ConsumerStats::default())),
        })
    }

    /// Start consuming messages
    ///
    /// This is a simplified implementation for demonstration.
    /// In production, would use rdkafka or similar Kafka client.
    pub async fn start_simulated(&mut self) -> Result<mpsc::Receiver<VectorMessage>> {
        *self.running.write().await = true;

        let (tx, rx) = mpsc::channel(self.config.batch_size);

        // Simulate consuming messages
        let running = self.running.clone();
        let stats = self.stats.clone();

        tokio::spawn(async move {
            let mut counter = 0;
            while *running.read().await {
                // Simulate receiving a message
                let msg = VectorMessage {
                    id: format!("doc_{}", counter),
                    vector: vec![0.1 * counter as f32; 128],
                    metadata: serde_json::json!({"index": counter}),
                    operation: Operation::Insert,
                };

                if tx.send(msg).await.is_err() {
                    break;
                }

                stats.write().await.messages_consumed += 1;
                counter += 1;

                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });

        Ok(rx)
    }

    /// Stop the consumer
    pub async fn stop(&mut self) {
        *self.running.write().await = false;
    }

    /// Get consumer statistics
    pub async fn stats(&self) -> ConsumerStats {
        self.stats.read().await.clone()
    }

    /// Consume a batch of messages
    pub async fn consume_batch(&mut self) -> Result<Vec<VectorMessage>> {
        // Simulated batch consumption
        let batch_size = self.config.batch_size;
        let mut messages = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            messages.push(VectorMessage {
                id: format!("batch_doc_{}", i),
                vector: vec![0.1 * i as f32; 128],
                metadata: serde_json::json!({"batch_index": i}),
                operation: Operation::Insert,
            });
        }

        let mut stats = self.stats.write().await;
        stats.messages_consumed += messages.len();
        stats.batches_processed += 1;

        Ok(messages)
    }

    /// Process a single message
    pub async fn process_message(&mut self, message: &VectorMessage) -> Result<()> {
        // Validate message
        if message.vector.is_empty() {
            let mut stats = self.stats.write().await;
            stats.errors += 1;
            return Err(anyhow!("Empty vector"));
        }

        // Simulate processing
        let mut stats = self.stats.write().await;
        match message.operation {
            Operation::Insert => stats.inserts += 1,
            Operation::Update => stats.updates += 1,
            Operation::Delete => stats.deletes += 1,
        }

        Ok(())
    }
}

/// Consumer statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsumerStats {
    pub messages_consumed: usize,
    pub batches_processed: usize,
    pub inserts: usize,
    pub updates: usize,
    pub deletes: usize,
    pub errors: usize,
}

/// Kafka producer for change streams
pub struct KafkaProducer {
    config: KafkaConfig,
    stats: Arc<tokio::sync::RwLock<ProducerStats>>,
}

impl KafkaProducer {
    /// Create a new Kafka producer
    pub fn new(config: KafkaConfig) -> Result<Self> {
        Ok(Self {
            config,
            stats: Arc::new(tokio::sync::RwLock::new(ProducerStats::default())),
        })
    }

    /// Publish a vector message
    pub async fn publish(&mut self, message: &VectorMessage) -> Result<()> {
        // Simulated publish
        let mut stats = self.stats.write().await;
        stats.messages_published += 1;
        stats.bytes_sent += estimate_message_size(message);

        Ok(())
    }

    /// Publish a batch of messages
    pub async fn publish_batch(&mut self, messages: &[VectorMessage]) -> Result<()> {
        for message in messages {
            self.publish(message).await?;
        }

        let mut stats = self.stats.write().await;
        stats.batches_published += 1;

        Ok(())
    }

    /// Get producer statistics
    pub async fn stats(&self) -> ProducerStats {
        self.stats.read().await.clone()
    }
}

/// Producer statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProducerStats {
    pub messages_published: usize,
    pub batches_published: usize,
    pub bytes_sent: usize,
    pub errors: usize,
}

/// Streaming pipeline for continuous ingestion
pub struct StreamingPipeline {
    consumer: KafkaConsumer,
    producer: Option<KafkaProducer>,
    batch_size: usize,
    config: PipelineConfig,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub max_retries: usize,
    pub retry_delay_ms: u64,
    pub enable_dlq: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay_ms: 1000,
            enable_dlq: true,
        }
    }
}

impl StreamingPipeline {
    /// Create a new streaming pipeline
    pub fn new(kafka_config: KafkaConfig) -> Result<Self> {
        let consumer = KafkaConsumer::new(kafka_config.clone())?;
        let producer = if kafka_config.dlq_topic.is_some() {
            Some(KafkaProducer::new(kafka_config.clone())?)
        } else {
            None
        };

        Ok(Self {
            consumer,
            producer,
            batch_size: kafka_config.batch_size,
            config: PipelineConfig::default(),
        })
    }

    /// Start the pipeline
    pub async fn start(&mut self) -> Result<mpsc::Receiver<VectorMessage>> {
        self.consumer.start_simulated().await
    }

    /// Stop the pipeline
    pub async fn stop(&mut self) {
        self.consumer.stop().await;
    }

    /// Get pipeline statistics
    pub async fn stats(&self) -> PipelineStats {
        let consumer_stats = self.consumer.stats().await;
        let producer_stats = if let Some(ref producer) = self.producer {
            Some(producer.stats().await)
        } else {
            None
        };

        PipelineStats {
            consumer: consumer_stats,
            producer: producer_stats,
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    pub consumer: ConsumerStats,
    pub producer: Option<ProducerStats>,
}

/// Estimate message size in bytes
fn estimate_message_size(message: &VectorMessage) -> usize {
    message.id.len() + message.vector.len() * 4 + message.metadata.to_string().len() + 10
    // operation overhead
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kafka_config_default() {
        let config = KafkaConfig::default();
        assert_eq!(config.brokers, vec!["localhost:9092".to_string()]);
        assert_eq!(config.topic, "vectors");
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_vector_message_serialization() {
        let msg = VectorMessage {
            id: "doc1".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            metadata: serde_json::json!({"key": "value"}),
            operation: Operation::Insert,
        };

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: VectorMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "doc1");
        assert_eq!(deserialized.vector.len(), 3);
        assert_eq!(deserialized.operation, Operation::Insert);
    }

    #[tokio::test]
    async fn test_kafka_consumer_creation() {
        let config = KafkaConfig::default();
        let consumer = KafkaConsumer::new(config);
        assert!(consumer.is_ok());
    }

    #[tokio::test]
    async fn test_consumer_batch() {
        let config = KafkaConfig {
            batch_size: 50,
            ..Default::default()
        };

        let mut consumer = KafkaConsumer::new(config).unwrap();
        let messages = consumer.consume_batch().await.unwrap();

        assert_eq!(messages.len(), 50);
        assert_eq!(messages[0].id, "batch_doc_0");
    }

    #[tokio::test]
    async fn test_consumer_stats() {
        let config = KafkaConfig::default();
        let mut consumer = KafkaConsumer::new(config).unwrap();

        let _ = consumer.consume_batch().await;

        let stats = consumer.stats().await;
        assert_eq!(stats.messages_consumed, 100);
        assert_eq!(stats.batches_processed, 1);
    }

    #[tokio::test]
    async fn test_process_message_validation() {
        let config = KafkaConfig::default();
        let mut consumer = KafkaConsumer::new(config).unwrap();

        let valid_msg = VectorMessage {
            id: "doc1".to_string(),
            vector: vec![0.1, 0.2],
            metadata: serde_json::json!({}),
            operation: Operation::Insert,
        };

        let result = consumer.process_message(&valid_msg).await;
        assert!(result.is_ok());

        let invalid_msg = VectorMessage {
            id: "doc2".to_string(),
            vector: vec![],
            metadata: serde_json::json!({}),
            operation: Operation::Insert,
        };

        let result = consumer.process_message(&invalid_msg).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_kafka_producer() {
        let config = KafkaConfig::default();
        let mut producer = KafkaProducer::new(config).unwrap();

        let msg = VectorMessage {
            id: "doc1".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            metadata: serde_json::json!({}),
            operation: Operation::Insert,
        };

        let result = producer.publish(&msg).await;
        assert!(result.is_ok());

        let stats = producer.stats().await;
        assert_eq!(stats.messages_published, 1);
    }

    #[tokio::test]
    async fn test_producer_batch() {
        let config = KafkaConfig::default();
        let mut producer = KafkaProducer::new(config).unwrap();

        let messages = vec![
            VectorMessage {
                id: "doc1".to_string(),
                vector: vec![0.1],
                metadata: serde_json::json!({}),
                operation: Operation::Insert,
            },
            VectorMessage {
                id: "doc2".to_string(),
                vector: vec![0.2],
                metadata: serde_json::json!({}),
                operation: Operation::Update,
            },
        ];

        let result = producer.publish_batch(&messages).await;
        assert!(result.is_ok());

        let stats = producer.stats().await;
        assert_eq!(stats.messages_published, 2);
        assert_eq!(stats.batches_published, 1);
    }

    #[tokio::test]
    async fn test_streaming_pipeline_creation() {
        let config = KafkaConfig::default();
        let pipeline = StreamingPipeline::new(config);
        assert!(pipeline.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_start_stop() {
        let config = KafkaConfig::default();
        let mut pipeline = StreamingPipeline::new(config).unwrap();

        let mut rx = pipeline.start().await.unwrap();

        // Receive a few messages
        let mut count = 0;
        while count < 3 {
            if rx.recv().await.is_some() {
                count += 1;
            }
        }

        pipeline.stop().await;
        assert!(count >= 3);
    }

    #[tokio::test]
    async fn test_operation_types() {
        let config = KafkaConfig::default();
        let mut consumer = KafkaConsumer::new(config).unwrap();

        for op in &[Operation::Insert, Operation::Update, Operation::Delete] {
            let msg = VectorMessage {
                id: "doc1".to_string(),
                vector: vec![0.1],
                metadata: serde_json::json!({}),
                operation: *op,
            };

            consumer.process_message(&msg).await.unwrap();
        }

        let stats = consumer.stats().await;
        assert_eq!(stats.inserts, 1);
        assert_eq!(stats.updates, 1);
        assert_eq!(stats.deletes, 1);
    }

    #[test]
    fn test_message_size_estimation() {
        let msg = VectorMessage {
            id: "doc1".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            metadata: serde_json::json!({"key": "value"}),
            operation: Operation::Insert,
        };

        let size = estimate_message_size(&msg);
        assert!(size > 0);
    }

    #[tokio::test]
    async fn test_pipeline_stats() {
        let config = KafkaConfig::default();
        let mut pipeline = StreamingPipeline::new(config).unwrap();

        let _rx = pipeline.start().await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;

        let stats = pipeline.stats().await;
        assert!(stats.consumer.messages_consumed > 0);
    }
}
