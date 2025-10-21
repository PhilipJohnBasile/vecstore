// Tests for Cross-Encoder Reranking
//
// Note: Integration tests with real models require downloading ONNX models.
// Run with: cargo test --features embeddings --test cross_encoder_reranking -- --ignored

#[cfg(feature = "embeddings")]
mod cross_encoder_tests {
    use std::collections::HashMap;
    use vecstore::reranking::cross_encoder::{CrossEncoderModel, CrossEncoderReranker};
    use vecstore::reranking::Reranker;
    use vecstore::store::Neighbor;
    use vecstore::Metadata;

    fn make_neighbor(id: &str, score: f32, text: &str) -> Neighbor {
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("text".to_string(), serde_json::json!(text));

        Neighbor {
            id: id.to_string(),
            score,
            metadata,
        }
    }

    #[test]
    fn test_cross_encoder_model_enum() {
        let model = CrossEncoderModel::MiniLML6V2;
        assert_eq!(model.model_id(), "cross-encoder/ms-marco-MiniLM-L-6-v2");
        assert_eq!(model.model_dir(), "ms-marco-minilm-l6-v2");

        let model = CrossEncoderModel::MiniLML12V2;
        assert_eq!(model.model_id(), "cross-encoder/ms-marco-MiniLM-L-12-v2");
        assert_eq!(model.model_dir(), "ms-marco-minilm-l12-v2");
    }

    #[test]
    fn test_cache_dir_structure() {
        let cache_dir = CrossEncoderModel::cache_dir();
        let path_str = cache_dir.to_string_lossy();

        assert!(path_str.contains("vecstore"));
        assert!(path_str.contains("cross-encoders"));
    }

    #[test]
    fn test_from_dir_missing() {
        // Test loading from non-existent directory
        let result = CrossEncoderReranker::from_dir("/nonexistent/path/to/model");
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("not found") || err_msg.contains("does not exist"));
    }

    #[test]
    fn test_from_pretrained_not_downloaded() {
        // Test loading a model that hasn't been downloaded yet
        // This should fail with instructions
        let result = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2);

        // If the model isn't downloaded, we expect an error
        // If it IS downloaded (from previous tests), we expect success
        match result {
            Ok(_) => {
                // Model was already downloaded, which is fine
            }
            Err(e) => {
                // Model not downloaded, error should contain helpful message
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("download")
                        || err_msg.contains("not found")
                        || err_msg.contains("huggingface.co")
                );
            }
        }
    }

    // Integration tests - require actual model files

    #[test]
    #[ignore] // Run with: cargo test -- --ignored
    fn test_cross_encoder_load_model() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model. Download it first or run model download script.");

        assert_eq!(reranker.name(), "Cross-Encoder (ONNX)");
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_score_pair_basic() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        // Test basic relevance scoring
        let score = reranker
            .score_pair(
                "what is rust programming",
                "Rust is a systems programming language",
            )
            .expect("Failed to score pair");

        // Score should be a valid float
        assert!(score.is_finite());
        println!("Rust relevance score: {}", score);
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_score_pair_relevance() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        // Relevant pair
        let score_relevant = reranker
            .score_pair("what is rust", "Rust is a programming language")
            .expect("Failed to score");

        // Less relevant pair
        let score_less_relevant = reranker
            .score_pair("what is rust", "Python is a programming language")
            .expect("Failed to score");

        // Irrelevant pair
        let score_irrelevant = reranker
            .score_pair("what is rust", "I like cooking pasta with tomato sauce")
            .expect("Failed to score");

        println!("Relevant score: {}", score_relevant);
        println!("Less relevant score: {}", score_less_relevant);
        println!("Irrelevant score: {}", score_irrelevant);

        // Relevant should score higher than irrelevant
        assert!(score_relevant > score_irrelevant);
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_rerank_empty() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        let results = vec![];
        let reranked = reranker.rerank("test query", results, 10).unwrap();

        assert_eq!(reranked.len(), 0);
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_rerank_basic() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        let results = vec![
            make_neighbor("doc1", 0.5, "Rust is a systems programming language"),
            make_neighbor("doc2", 0.9, "Python is great for data science"),
            make_neighbor("doc3", 0.7, "JavaScript runs in the browser"),
        ];

        let reranked = reranker
            .rerank("rust programming language", results, 3)
            .expect("Failed to rerank");

        assert_eq!(reranked.len(), 3);

        // doc1 should be ranked first (most relevant to "rust programming language")
        println!("Reranked order:");
        for (i, neighbor) in reranked.iter().enumerate() {
            println!("  {}. {} (score: {})", i + 1, neighbor.id, neighbor.score);
        }

        assert_eq!(reranked[0].id, "doc1");
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_rerank_top_k() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        let results = vec![
            make_neighbor("doc1", 0.1, "Rust systems programming"),
            make_neighbor("doc2", 0.2, "Python data science"),
            make_neighbor("doc3", 0.3, "Rust async await"),
            make_neighbor("doc4", 0.4, "JavaScript web development"),
            make_neighbor("doc5", 0.5, "Rust memory safety"),
        ];

        // Rerank but only keep top 2
        let reranked = reranker
            .rerank("rust programming", results, 2)
            .expect("Failed to rerank");

        assert_eq!(reranked.len(), 2);

        // Top 2 should be Rust-related
        assert!(reranked.iter().all(|n| {
            let text = n.metadata.fields.get("text").unwrap().as_str().unwrap();
            text.to_lowercase().contains("rust")
        }));
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_vs_original_scores() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        // Create results where original scores are misleading
        // (e.g., embedding similarity doesn't match semantic relevance)
        let results = vec![
            // High embedding score but not very relevant
            make_neighbor(
                "doc1",
                0.95,
                "Rust is a color that forms on iron when it oxidizes",
            ),
            // Lower embedding score but very relevant
            make_neighbor("doc2", 0.60, "Rust is a systems programming language"),
            // Medium embedding score, somewhat relevant
            make_neighbor(
                "doc3",
                0.75,
                "Programming languages like Rust focus on safety",
            ),
        ];

        let query = "rust programming language";
        let reranked = reranker
            .rerank(query, results, 3)
            .expect("Failed to rerank");

        println!("Original vs Reranked:");
        for neighbor in &reranked {
            println!("  {} - score: {}", neighbor.id, neighbor.score);
        }

        // doc2 should be ranked first (most relevant to programming)
        // even though it had lower original embedding score
        assert_eq!(reranked[0].id, "doc2");
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_with_missing_text() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        // Create a neighbor without text in metadata
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata
            .fields
            .insert("other_field".to_string(), serde_json::json!("value"));

        let results = vec![Neighbor {
            id: "doc1".to_string(),
            score: 0.5,
            metadata,
        }];

        // Should not panic, should handle gracefully
        let reranked = reranker.rerank("test query", results, 1).unwrap();

        assert_eq!(reranked.len(), 1);
        // Original score should be preserved
        assert_eq!(reranked[0].score, 0.5);
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_long_documents() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        // Test with a long document (will be truncated to 512 tokens)
        let long_doc = "Rust programming language. ".repeat(100); // ~400 words

        let results = vec![
            make_neighbor("doc1", 0.5, &long_doc),
            make_neighbor("doc2", 0.6, "Short doc about Python"),
        ];

        // Should handle truncation gracefully
        let reranked = reranker
            .rerank("rust programming", results, 2)
            .expect("Failed to handle long documents");

        assert_eq!(reranked.len(), 2);
        assert_eq!(reranked[0].id, "doc1"); // Rust doc should rank first
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_special_characters() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        let results = vec![
            make_neighbor("doc1", 0.5, "Rust: A systems programming language!"),
            make_neighbor("doc2", 0.6, "Python & data science (ML/AI)"),
            make_neighbor("doc3", 0.7, "C++ programming... very fast?"),
        ];

        // Should handle special characters correctly
        let reranked = reranker
            .rerank("rust programming!", results, 3)
            .expect("Failed to handle special characters");

        assert_eq!(reranked.len(), 3);
        assert_eq!(reranked[0].id, "doc1");
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_multiple_queries() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        let results = vec![
            make_neighbor("doc1", 0.5, "Rust is a systems programming language"),
            make_neighbor("doc2", 0.6, "Python is great for data science"),
            make_neighbor("doc3", 0.7, "JavaScript for web development"),
        ];

        // Query 1: Rust
        let reranked1 = reranker
            .rerank("rust programming", results.clone(), 1)
            .expect("Failed to rerank");
        assert_eq!(reranked1[0].id, "doc1");

        // Query 2: Python
        let reranked2 = reranker
            .rerank("python data science", results.clone(), 1)
            .expect("Failed to rerank");
        assert_eq!(reranked2[0].id, "doc2");

        // Query 3: JavaScript
        let reranked3 = reranker
            .rerank("javascript web", results, 1)
            .expect("Failed to rerank");
        assert_eq!(reranked3[0].id, "doc3");
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_score_consistency() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        let query = "rust programming language";
        let document = "Rust is a systems programming language";

        // Score the same pair multiple times
        let score1 = reranker.score_pair(query, document).unwrap();
        let score2 = reranker.score_pair(query, document).unwrap();
        let score3 = reranker.score_pair(query, document).unwrap();

        // Scores should be consistent (deterministic)
        assert!((score1 - score2).abs() < 0.0001);
        assert!((score2 - score3).abs() < 0.0001);
    }

    #[test]
    #[ignore]
    fn test_cross_encoder_batch_rerank() {
        let reranker = CrossEncoderReranker::from_pretrained(CrossEncoderModel::MiniLML6V2)
            .expect("Failed to load model");

        // Test with a larger batch
        let results: Vec<Neighbor> = (0..20)
            .map(|i| {
                let text = if i % 3 == 0 {
                    format!("Rust programming document {}", i)
                } else if i % 3 == 1 {
                    format!("Python programming document {}", i)
                } else {
                    format!("JavaScript programming document {}", i)
                };
                make_neighbor(&format!("doc{}", i), 0.5, &text)
            })
            .collect();

        let reranked = reranker
            .rerank("rust programming", results, 5)
            .expect("Failed to rerank batch");

        assert_eq!(reranked.len(), 5);

        // Top results should mostly be Rust-related
        let rust_count = reranked
            .iter()
            .filter(|n| {
                let text = n.metadata.fields.get("text").unwrap().as_str().unwrap();
                text.contains("Rust")
            })
            .count();

        println!("Rust documents in top 5: {}", rust_count);
        assert!(rust_count >= 3); // At least 3 of top 5 should be Rust-related
    }
}
