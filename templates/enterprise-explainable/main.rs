//! Enterprise Explainable Search Template
//!
//! A production-ready template for enterprise deployments requiring:
//! - Explainable search results (WHY did this match?)
//! - Audit logging and compliance tracking
//! - Vector lineage for provenance
//! - Privacy-preserving search options
//!
//! Perfect for: Healthcare, Finance, Legal, Government

use anyhow::Result;
use std::collections::HashMap;
use vecstore::{VecStore, DistanceMetric};
use vecstore::explainable::{ExplainableSearch, ExplainConfig, ExplainLevel};
use vecstore::lineage::{LineageTracker, SourceInfo, ModelInfo};
use vecstore::privacy::{DifferentialPrivacy, PrivacyConfig};

/// Enterprise RAG system with full explainability
struct EnterpriseRAG {
    store: VecStore,
    explainer: ExplainableSearch,
    lineage: LineageTracker,
    privacy: Option<DifferentialPrivacy>,
    audit_log: Vec<AuditEntry>,
}

#[derive(Debug, Clone)]
struct AuditEntry {
    timestamp: String,
    user_id: String,
    action: String,
    query: Option<String>,
    results_count: usize,
    explanation_requested: bool,
}

impl EnterpriseRAG {
    fn new(dimension: usize, enable_privacy: bool) -> Self {
        let store = VecStore::new(dimension, DistanceMetric::Cosine);

        let explainer = ExplainableSearch::new(ExplainConfig {
            level: ExplainLevel::Detailed,
            include_dimension_contributions: true,
            include_counter_examples: true,
            max_contributing_dimensions: 10,
        });

        let lineage = LineageTracker::new();

        let privacy = if enable_privacy {
            Some(DifferentialPrivacy::new(PrivacyConfig {
                epsilon: 1.0,
                delta: 1e-5,
                mechanism: "gaussian".to_string(),
            }))
        } else {
            None
        };

        Self {
            store,
            explainer,
            lineage,
            privacy,
            audit_log: Vec::new(),
        }
    }

    /// Index a document with full lineage tracking
    fn index_document(
        &mut self,
        doc_id: &str,
        content: &str,
        embedding: Vec<f32>,
        source: SourceInfo,
        model: ModelInfo,
    ) -> Result<()> {
        // Apply privacy protection if enabled
        let final_embedding = if let Some(ref privacy) = self.privacy {
            privacy.add_noise(&embedding)
        } else {
            embedding
        };

        // Track lineage
        self.lineage.track(
            doc_id,
            source,
            model,
            chrono::Utc::now(),
        );

        // Index with metadata
        self.store.add_with_metadata(
            doc_id,
            &final_embedding,
            serde_json::json!({
                "content": content,
                "indexed_at": chrono::Utc::now().to_rfc3339(),
                "privacy_protected": self.privacy.is_some(),
            }),
        )?;

        Ok(())
    }

    /// Search with full explanation
    fn search_with_explanation(
        &mut self,
        user_id: &str,
        query: &str,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<SearchResult> {
        let start = std::time::Instant::now();

        // Perform search
        let results = self.store.search(query_embedding, top_k)?;

        // Generate explanations for each result
        let mut explained_results = Vec::new();
        for result in &results {
            let explanation = self.explainer.explain(
                query_embedding,
                result.vector.as_ref().unwrap_or(&vec![]),
                result.score,
            );

            explained_results.push(ExplainedResult {
                id: result.id.clone(),
                score: result.score,
                content: result.metadata.as_ref()
                    .and_then(|m| m.get("content"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                explanation,
                lineage: self.lineage.get(&result.id),
            });
        }

        let elapsed = start.elapsed();

        // Audit logging
        self.audit_log.push(AuditEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            user_id: user_id.to_string(),
            action: "search".to_string(),
            query: Some(query.to_string()),
            results_count: explained_results.len(),
            explanation_requested: true,
        });

        Ok(SearchResult {
            results: explained_results,
            query_time_ms: elapsed.as_millis() as u64,
            total_documents: self.store.len(),
        })
    }

    /// Export audit log for compliance
    fn export_audit_log(&self) -> String {
        serde_json::to_string_pretty(&self.audit_log).unwrap_or_default()
    }
}

struct SearchResult {
    results: Vec<ExplainedResult>,
    query_time_ms: u64,
    total_documents: usize,
}

struct ExplainedResult {
    id: String,
    score: f32,
    content: String,
    explanation: Explanation,
    lineage: Option<LineageInfo>,
}

struct Explanation {
    summary: String,
    dimension_contributions: Vec<(usize, f32)>,
    key_factors: Vec<String>,
}

struct LineageInfo {
    source: String,
    model: String,
    indexed_at: String,
}

impl ExplainableSearch {
    fn new(config: ExplainConfig) -> Self {
        Self { config }
    }

    fn explain(&self, query: &[f32], document: &[f32], score: f32) -> Explanation {
        let mut contributions: Vec<(usize, f32)> = query.iter()
            .zip(document.iter())
            .enumerate()
            .map(|(i, (q, d))| (i, q * d))
            .collect();

        contributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        contributions.truncate(self.config.max_contributing_dimensions);

        let key_factors: Vec<String> = contributions.iter()
            .take(3)
            .map(|(dim, contrib)| {
                format!("Dimension {} contributed {:.4} to the score", dim, contrib)
            })
            .collect();

        Explanation {
            summary: format!(
                "This result matched with {:.2}% similarity. The match was driven by {} key dimensions.",
                score * 100.0,
                contributions.len()
            ),
            dimension_contributions: contributions,
            key_factors,
        }
    }
}

fn main() -> Result<()> {
    println!("=====================================================");
    println!("  VecStore Enterprise - Explainable Search");
    println!("  For Regulated Industries (Healthcare/Finance/Legal)");
    println!("=====================================================\n");

    // Initialize with privacy protection
    let mut rag = EnterpriseRAG::new(384, true);

    println!("[Privacy Mode: ENABLED - Differential Privacy Applied]\n");

    // Simulate indexing documents
    let documents = vec![
        ("policy-001", "Patient data must be encrypted at rest and in transit per HIPAA requirements."),
        ("policy-002", "Financial transactions require dual authorization for amounts over $10,000."),
        ("policy-003", "Legal documents must retain audit trails for 7 years minimum."),
        ("policy-004", "Access to sensitive data requires multi-factor authentication."),
        ("policy-005", "Data retention policies must comply with GDPR Article 17 right to erasure."),
    ];

    println!("Indexing {} compliance documents...\n", documents.len());

    for (doc_id, content) in &documents {
        // Generate embedding (placeholder - use real embeddings in production)
        let embedding = generate_demo_embedding(content, 384);

        rag.index_document(
            doc_id,
            content,
            embedding,
            SourceInfo {
                document_type: "policy".to_string(),
                origin: "compliance_database".to_string(),
                version: "1.0".to_string(),
            },
            ModelInfo {
                name: "text-embedding-3-small".to_string(),
                version: "2024-01".to_string(),
                provider: "openai".to_string(),
            },
        )?;
    }

    // Perform explainable search
    let query = "What are the requirements for patient data security?";
    println!("Query: {}\n", query);

    let query_embedding = generate_demo_embedding(query, 384);
    let result = rag.search_with_explanation(
        "analyst-001",
        query,
        &query_embedding,
        3,
    )?;

    println!("Results ({} ms, {} total docs):\n", result.query_time_ms, result.total_documents);

    for (i, r) in result.results.iter().enumerate() {
        println!("{}. [{}] Score: {:.4}", i + 1, r.id, r.score);
        println!("   Content: {}", &r.content[..r.content.len().min(70)]);
        println!("   Explanation: {}", r.explanation.summary);
        println!("   Key Factors:");
        for factor in &r.explanation.key_factors {
            println!("     - {}", factor);
        }
        println!();
    }

    // Export audit log
    println!("Audit Log:");
    println!("{}", rag.export_audit_log());

    Ok(())
}

// Demo embedding generator (replace with real embeddings)
fn generate_demo_embedding(text: &str, dimension: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut embedding = vec![0.0f32; dimension];
    for (i, word) in text.split_whitespace().enumerate() {
        let mut hasher = DefaultHasher::new();
        word.to_lowercase().hash(&mut hasher);
        let hash = hasher.finish();
        for j in 0..dimension {
            let idx = (hash.wrapping_add(j as u64) as usize) % dimension;
            embedding[idx] += 1.0 / ((i + 1) as f32);
        }
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    embedding
}
