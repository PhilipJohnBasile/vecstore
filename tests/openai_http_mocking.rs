// HTTP Mocking Tests for OpenAI Embeddings Backend
//
// This test suite uses wiremock to simulate various HTTP scenarios
// without requiring actual API calls or API keys.
//
// Run with: cargo test --features "embeddings,openai-embeddings" --test openai_http_mocking

#![cfg(all(feature = "embeddings", feature = "openai-embeddings"))]

use vecstore::embeddings::openai_backend::{OpenAIEmbedding, OpenAIModel};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_successful_single_embedding() {
    // Start mock server
    let mock_server = MockServer::start().await;

    // Mock successful response with proper embedding format
    let embedding_vec: Vec<f32> = (0..1536).map(|i| (i as f32) * 0.001).collect();
    let response_body = serde_json::json!({
        "data": [{
            "embedding": embedding_vec,
            "index": 0
        }],
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("Authorization", "Bearer test-api-key"))
        .and(header("Content-Type", "application/json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&mock_server)
        .await;

    // Create embedder with mock server URL
    let embedder = OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
        .await
        .expect("Failed to create embedder")
        .with_base_url(mock_server.uri());

    // Test actual embedding call
    let result = embedder.embed_async("Hello world").await;
    assert!(result.is_ok(), "Embedding should succeed: {:?}", result.err());

    let embedding = result.unwrap();
    assert_eq!(embedding.len(), 1536, "Embedding should have 1536 dimensions");
}

#[tokio::test]
async fn test_rate_limit_error_with_retry() {
    let mock_server = MockServer::start().await;

    // First request: rate limit error (429)
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error"
            }
        })))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    // Second request: success
    let success_response = serde_json::json!({
        "data": [{
            "embedding": vec![0.1_f32; 1536],
            "index": 0
        }],
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&success_response))
        .mount(&mock_server)
        .await;

    // Create embedder with mock server URL
    let embedder = OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
        .await
        .expect("Failed to create embedder")
        .with_base_url(mock_server.uri());

    // Test that retry succeeds after rate limit
    let result = embedder.embed_async("Hello world").await;
    assert!(result.is_ok(), "Should succeed after retry: {:?}", result.err());

    let embedding = result.unwrap();
    assert_eq!(embedding.len(), 1536, "Embedding should have 1536 dimensions");
}

#[tokio::test]
async fn test_authentication_error() {
    let mock_server = MockServer::start().await;

    // Mock authentication error
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error"
            }
        })))
        .mount(&mock_server)
        .await;

    // Create embedder with mock server URL
    let embedder = OpenAIEmbedding::new("invalid-key".to_string(), OpenAIModel::TextEmbedding3Small)
        .await
        .expect("Failed to create embedder")
        .with_base_url(mock_server.uri());

    // Test that authentication error is properly handled
    let result = embedder.embed_async("Hello world").await;
    assert!(result.is_err(), "Should fail with authentication error");

    let error = result.err().unwrap();
    assert!(
        error.to_string().contains("401") || error.to_string().contains("Unauthorized") || error.to_string().contains("API"),
        "Error should indicate authentication failure: {}",
        error
    );
}

#[tokio::test]
async fn test_server_error() {
    let mock_server = MockServer::start().await;

    // Mock server error
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
            "error": {
                "message": "Internal server error",
                "type": "server_error"
            }
        })))
        .mount(&mock_server)
        .await;

    // Create embedder with mock server URL
    let embedder = OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
        .await
        .expect("Failed to create embedder")
        .with_base_url(mock_server.uri());

    // Test that server error is properly handled
    let result = embedder.embed_async("Hello world").await;
    assert!(result.is_err(), "Should fail with server error");

    let error = result.err().unwrap();
    assert!(
        error.to_string().contains("500") || error.to_string().contains("server") || error.to_string().contains("error"),
        "Error should indicate server failure: {}",
        error
    );
}

#[tokio::test]
async fn test_malformed_response() {
    let mock_server = MockServer::start().await;

    // Mock malformed response (missing required fields)
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "invalid": "response"
        })))
        .mount(&mock_server)
        .await;

    // Create embedder with mock server URL
    let embedder = OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
        .await
        .expect("Failed to create embedder")
        .with_base_url(mock_server.uri());

    // Test that malformed response is properly handled
    let result = embedder.embed_async("Hello world").await;
    assert!(result.is_err(), "Should fail with malformed response");
}

#[tokio::test]
async fn test_batch_embedding_with_multiple_items() {
    let mock_server = MockServer::start().await;

    // Mock successful batch response
    let response_body = serde_json::json!({
        "data": [
            {
                "embedding": vec![0.1_f32; 1536],
                "index": 0
            },
            {
                "embedding": vec![0.2_f32; 1536],
                "index": 1
            },
            {
                "embedding": vec![0.3_f32; 1536],
                "index": 2
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "total_tokens": 15
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&mock_server)
        .await;

    // Create embedder with mock server URL
    let embedder = OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
        .await
        .expect("Failed to create embedder")
        .with_base_url(mock_server.uri());

    // Test batch embedding with multiple items
    let texts = vec!["Hello".to_string(), "World".to_string(), "Test".to_string()];
    let result = embedder.embed_batch_async(&texts).await;
    assert!(result.is_ok(), "Batch embedding should succeed: {:?}", result.err());

    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 3, "Should return 3 embeddings");
    for embedding in &embeddings {
        assert_eq!(embedding.len(), 1536, "Each embedding should have 1536 dimensions");
    }
}

#[tokio::test]
async fn test_network_timeout() {
    let mock_server = MockServer::start().await;

    // Mock delayed response (simulates timeout) - use 5 seconds for test
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_delay(std::time::Duration::from_secs(5)),
        )
        .mount(&mock_server)
        .await;

    // Create embedder with mock server URL and short timeout
    let embedder = OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
        .await
        .expect("Failed to create embedder")
        .with_base_url(mock_server.uri())
        .with_timeout(std::time::Duration::from_secs(1));

    // Test that timeout is properly handled
    let result = embedder.embed_async("Hello world").await;
    assert!(result.is_err(), "Should fail with timeout");

    let error = result.err().unwrap();
    assert!(
        error.to_string().to_lowercase().contains("timeout") ||
        error.to_string().to_lowercase().contains("timed out") ||
        error.to_string().to_lowercase().contains("deadline"),
        "Error should indicate timeout: {}",
        error
    );
}

#[tokio::test]
async fn test_retry_on_network_error() {
    let mock_server = MockServer::start().await;

    // First two requests: fail
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(500))
        .up_to_n_times(2)
        .mount(&mock_server)
        .await;

    // Third request: success
    let success_response = serde_json::json!({
        "data": [{
            "embedding": vec![0.1_f32; 1536],
            "index": 0
        }],
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&success_response))
        .mount(&mock_server)
        .await;

    // Create embedder with mock server URL
    let embedder = OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
        .await
        .expect("Failed to create embedder")
        .with_base_url(mock_server.uri());

    // Test that retry succeeds after failures
    let result = embedder.embed_async("Hello world").await;
    assert!(result.is_ok(), "Should succeed after retries: {:?}", result.err());

    let embedding = result.unwrap();
    assert_eq!(embedding.len(), 1536, "Embedding should have 1536 dimensions");
}

#[tokio::test]
async fn test_embedding_order_preservation() {
    let mock_server = MockServer::start().await;

    // Mock response with out-of-order indices
    let response_body = serde_json::json!({
        "data": [
            {
                "embedding": vec![0.3_f32; 1536],
                "index": 2
            },
            {
                "embedding": vec![0.1_f32; 1536],
                "index": 0
            },
            {
                "embedding": vec![0.2_f32; 1536],
                "index": 1
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "total_tokens": 15
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&mock_server)
        .await;

    // Create embedder with mock server URL
    let embedder = OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
        .await
        .expect("Failed to create embedder")
        .with_base_url(mock_server.uri());

    // Test batch embedding with out-of-order response
    let texts = vec!["First".to_string(), "Second".to_string(), "Third".to_string()];
    let result = embedder.embed_batch_async(&texts).await;
    assert!(result.is_ok(), "Batch embedding should succeed: {:?}", result.err());

    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 3, "Should return 3 embeddings");

    // Verify order is preserved (first embedding should be 0.1, second 0.2, third 0.3)
    // The mock returns out-of-order (index 2 first, then 0, then 1)
    // but the embedder should reorder them correctly
    assert!((embeddings[0][0] - 0.1).abs() < 0.001, "First embedding should be 0.1");
    assert!((embeddings[1][0] - 0.2).abs() < 0.001, "Second embedding should be 0.2");
    assert!((embeddings[2][0] - 0.3).abs() < 0.001, "Third embedding should be 0.3");
}

#[tokio::test]
async fn test_different_model_requests() {
    // Test text-embedding-3-small model
    let mock_server_small = MockServer::start().await;

    let small_embedding: Vec<f32> = (0..1536).map(|i| (i as f32) * 0.001).collect();
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{
                "embedding": small_embedding,
                "index": 0
            }],
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        })))
        .mount(&mock_server_small)
        .await;

    let embedder_small = OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Small)
        .await
        .expect("Failed to create small embedder")
        .with_base_url(mock_server_small.uri());

    let result_small = embedder_small.embed_async("test").await;
    assert!(result_small.is_ok(), "Small model should succeed: {:?}", result_small.err());
    let embedding_small = result_small.unwrap();
    assert_eq!(embedding_small.len(), 1536, "Small model should return 1536 dimensions");

    // Test text-embedding-3-large model
    let mock_server_large = MockServer::start().await;

    let large_embedding: Vec<f32> = (0..3072).map(|i| (i as f32) * 0.001).collect();
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{
                "embedding": large_embedding,
                "index": 0
            }],
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        })))
        .mount(&mock_server_large)
        .await;

    let embedder_large = OpenAIEmbedding::new("test-api-key".to_string(), OpenAIModel::TextEmbedding3Large)
        .await
        .expect("Failed to create large embedder")
        .with_base_url(mock_server_large.uri());

    let result_large = embedder_large.embed_async("test").await;
    assert!(result_large.is_ok(), "Large model should succeed: {:?}", result_large.err());
    let embedding_large = result_large.unwrap();
    assert_eq!(embedding_large.len(), 3072, "Large model should return 3072 dimensions");
}

// NOTE: These tests use wiremock to simulate OpenAI API responses.
// The OpenAIEmbedding struct supports a custom base URL via `with_base_url()`
// and custom timeout via `with_timeout()` for testing purposes.
//
// Run with: cargo test --features "embeddings,openai-embeddings" --test openai_http_mocking

#[cfg(test)]
mod test_coverage_note {
    //! HTTP Mocking Test Coverage
    //!
    //! Implemented Tests:
    //! - ✅ test_successful_single_embedding - Validates successful API response
    //! - ✅ test_rate_limit_error_with_retry - Validates retry after 429
    //! - ✅ test_authentication_error - Validates 401 error handling
    //! - ✅ test_server_error - Validates 500 error handling
    //! - ✅ test_malformed_response - Validates JSON parse error handling
    //! - ✅ test_batch_embedding_with_multiple_items - Validates batch processing
    //! - ✅ test_network_timeout - Validates timeout handling
    //! - ✅ test_retry_on_network_error - Validates retry logic after failures
    //! - ✅ test_embedding_order_preservation - Validates response ordering
    //! - ✅ test_different_model_requests - Validates model-specific dimensions
    //!
    //! Benefits:
    //! - Test HTTP error scenarios without API keys
    //! - Validate retry logic with controlled failures
    //! - Test rate limiting behavior
    //! - Verify request/response parsing
    //! - Fast test execution (no network I/O)
}
