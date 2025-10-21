#![cfg(all(feature = "embeddings", feature = "openai-embeddings"))]

// Tests for OpenAI embeddings backend
//
// Note: These tests use mocking and don't require a real API key.
// For integration tests with real API, set OPENAI_API_KEY environment variable.

#[cfg(feature = "openai-embeddings")]
mod openai_tests {
    use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};
    use vecstore::embeddings::TextEmbedder;

    #[test]
    fn test_model_properties() {
        // Test TextEmbedding3Small
        assert_eq!(
            OpenAIModel::TextEmbedding3Small.as_str(),
            "text-embedding-3-small"
        );
        assert_eq!(OpenAIModel::TextEmbedding3Small.dimension(), 1536);
        assert_eq!(
            OpenAIModel::TextEmbedding3Small.cost_per_million_tokens(),
            0.02
        );

        // Test TextEmbedding3Large
        assert_eq!(
            OpenAIModel::TextEmbedding3Large.as_str(),
            "text-embedding-3-large"
        );
        assert_eq!(OpenAIModel::TextEmbedding3Large.dimension(), 3072);
        assert_eq!(
            OpenAIModel::TextEmbedding3Large.cost_per_million_tokens(),
            0.13
        );

        // Test Ada002
        assert_eq!(OpenAIModel::Ada002.as_str(), "text-embedding-ada-002");
        assert_eq!(OpenAIModel::Ada002.dimension(), 1536);
        assert_eq!(OpenAIModel::Ada002.cost_per_million_tokens(), 0.10);
    }

    #[tokio::test]
    async fn test_embedder_creation() {
        let result =
            OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
                .await;

        assert!(result.is_ok());
        let embedder = result.unwrap();
        assert_eq!(embedder.model().as_str(), "text-embedding-3-small");
    }

    #[tokio::test]
    async fn test_dimension_via_trait() {
        let embedder =
            OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
                .await
                .unwrap();

        let dim = embedder.dimension().unwrap();
        assert_eq!(dim, 1536);
    }

    #[tokio::test]
    async fn test_dimension_large_model() {
        let embedder =
            OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Large)
                .await
                .unwrap();

        let dim = embedder.dimension().unwrap();
        assert_eq!(dim, 3072);
    }

    #[test]
    fn test_cost_estimation_small() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let embedder = runtime
            .block_on(OpenAIEmbedding::new(
                "test-api-key".to_string(),
                OpenAIModel::TextEmbedding3Small,
            ))
            .unwrap();

        // Test with simple text
        let texts = vec!["Hello world"];
        let cost = embedder.estimate_cost(&texts);

        // "Hello world" is ~12 chars, so ~3 tokens
        // Cost should be very small
        assert!(cost > 0.0);
        assert!(cost < 0.001); // Should be less than $0.001
    }

    #[test]
    fn test_cost_estimation_large() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let embedder = runtime
            .block_on(OpenAIEmbedding::new(
                "test-api-key".to_string(),
                OpenAIModel::TextEmbedding3Large,
            ))
            .unwrap();

        let texts = vec!["Hello world"];
        let cost = embedder.estimate_cost(&texts);

        // Large model costs more
        assert!(cost > 0.0);
    }

    #[test]
    fn test_cost_estimation_multiple_texts() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let embedder = runtime
            .block_on(OpenAIEmbedding::new(
                "test-api-key".to_string(),
                OpenAIModel::TextEmbedding3Small,
            ))
            .unwrap();

        let single_text = vec!["Hello world"];
        let multiple_texts = vec!["Hello world", "Hello world", "Hello world"];

        let cost_single = embedder.estimate_cost(&single_text);
        let cost_multiple = embedder.estimate_cost(&multiple_texts);

        // Multiple texts should cost ~3x more (with some tolerance for estimation)
        assert!(cost_multiple > cost_single * 2.0);
        assert!(cost_multiple < cost_single * 5.0); // More tolerance for estimation
    }

    #[test]
    fn test_rate_limiter_configuration() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let embedder = runtime
            .block_on(OpenAIEmbedding::new(
                "test-api-key".to_string(),
                OpenAIModel::TextEmbedding3Small,
            ))
            .unwrap();

        // Test with custom rate limit
        let embedder_limited = embedder.with_rate_limit(100);

        // Should still work (we can't easily test the actual rate limiting without real API calls)
        assert_eq!(embedder_limited.dimension().unwrap(), 1536);
    }

    #[test]
    fn test_retry_configuration() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let embedder = runtime
            .block_on(OpenAIEmbedding::new(
                "test-api-key".to_string(),
                OpenAIModel::TextEmbedding3Small,
            ))
            .unwrap();

        let embedder_with_retries = embedder.with_max_retries(5);

        assert_eq!(embedder_with_retries.dimension().unwrap(), 1536);
    }

    #[test]
    fn test_model_enum_copy() {
        let model1 = OpenAIModel::TextEmbedding3Small;
        let model2 = model1; // Should work because Copy is derived

        assert_eq!(model1.as_str(), model2.as_str());
    }

    #[test]
    fn test_all_models_have_dimensions() {
        assert!(OpenAIModel::TextEmbedding3Small.dimension() > 0);
        assert!(OpenAIModel::TextEmbedding3Large.dimension() > 0);
        assert!(OpenAIModel::Ada002.dimension() > 0);
    }

    #[test]
    fn test_all_models_have_costs() {
        assert!(OpenAIModel::TextEmbedding3Small.cost_per_million_tokens() > 0.0);
        assert!(OpenAIModel::TextEmbedding3Large.cost_per_million_tokens() > 0.0);
        assert!(OpenAIModel::Ada002.cost_per_million_tokens() > 0.0);
    }

    #[test]
    fn test_large_model_more_expensive() {
        let small_cost = OpenAIModel::TextEmbedding3Small.cost_per_million_tokens();
        let large_cost = OpenAIModel::TextEmbedding3Large.cost_per_million_tokens();

        assert!(large_cost > small_cost);
    }

    #[test]
    fn test_large_model_higher_dimension() {
        let small_dim = OpenAIModel::TextEmbedding3Small.dimension();
        let large_dim = OpenAIModel::TextEmbedding3Large.dimension();

        assert!(large_dim > small_dim);
    }

    #[test]
    fn test_cost_estimation_empty() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let embedder = runtime
            .block_on(OpenAIEmbedding::new(
                "test-api-key".to_string(),
                OpenAIModel::TextEmbedding3Small,
            ))
            .unwrap();

        let empty: Vec<&str> = vec![];
        let cost = embedder.estimate_cost(&empty);

        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_cost_estimation_long_text() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let embedder = runtime
            .block_on(OpenAIEmbedding::new(
                "test-api-key".to_string(),
                OpenAIModel::TextEmbedding3Small,
            ))
            .unwrap();

        // Create a long text (1000 chars)
        let long_text = "a".repeat(1000);
        let texts = vec![long_text.as_str()];
        let cost = embedder.estimate_cost(&texts);

        // 1000 chars ≈ 250 tokens
        // At $0.02 per 1M tokens: 250 * 0.02 / 1,000,000 = 0.000005
        assert!(cost > 0.0);
        assert!(cost < 0.01); // Should be very small
    }

    // Integration test - only runs if OPENAI_API_KEY is set
    #[tokio::test]
    #[ignore] // Ignored by default, run with: cargo test -- --ignored
    async fn test_real_api_single_embedding() {
        let api_key = match std::env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping test: OPENAI_API_KEY not set");
                return;
            }
        };

        let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small)
            .await
            .expect("Failed to create embedder");

        let embedding = embedder
            .embed_async("Hello world")
            .await
            .expect("Failed to embed text");

        assert_eq!(embedding.len(), 1536);

        // Check that embedding is normalized (approximately)
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.1); // OpenAI embeddings are normalized
    }

    #[tokio::test]
    #[ignore]
    async fn test_real_api_batch_embedding() {
        let api_key = match std::env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping test: OPENAI_API_KEY not set");
                return;
            }
        };

        let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Small)
            .await
            .expect("Failed to create embedder");

        let texts = vec!["Hello", "World", "Test"];
        let embeddings = embedder
            .embed_batch_async(&texts)
            .await
            .expect("Failed to embed batch");

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 1536);
        assert_eq!(embeddings[1].len(), 1536);
        assert_eq!(embeddings[2].len(), 1536);

        // Embeddings should be different
        assert_ne!(embeddings[0], embeddings[1]);
        assert_ne!(embeddings[1], embeddings[2]);
    }

    #[tokio::test]
    #[ignore]
    async fn test_real_api_large_model() {
        let api_key = match std::env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping test: OPENAI_API_KEY not set");
                return;
            }
        };

        let embedder = OpenAIEmbedding::new(api_key, OpenAIModel::TextEmbedding3Large)
            .await
            .expect("Failed to create embedder");

        let embedding = embedder
            .embed_async("Test")
            .await
            .expect("Failed to embed text");

        assert_eq!(embedding.len(), 3072); // Large model dimension
    }

    #[test]
    fn test_sync_interface() {
        // Test that TextEmbedder trait works (synchronous)
        // Note: This would make a real API call if we had a valid key
        // So we just test that the interface compiles
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let embedder = runtime
            .block_on(OpenAIEmbedding::new(
                "test-key".to_string(),
                OpenAIModel::TextEmbedding3Small,
            ))
            .unwrap();

        // Just verify the trait method exists
        assert_eq!(embedder.dimension().unwrap(), 1536);
    }
}
