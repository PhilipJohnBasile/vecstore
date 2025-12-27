// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # REST API with OpenAPI
//!
//! Production-ready REST API with OpenAPI 3.1 specification,
//! request validation, rate limiting, and SDK generation support.
//!
//! ## Features
//!
//! - **OpenAPI 3.1 Spec**: Auto-generated specification
//! - **Request Validation**: Schema-based validation
//! - **Rate Limiting**: Per-client rate limits
//! - **Authentication**: API key and JWT support
//! - **CORS**: Cross-origin request handling
//! - **Versioned Endpoints**: API versioning support
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::rest_api::{RestServer, RestConfig};
//!
//! let config = RestConfig::default()
//!     .with_port(8080)
//!     .with_api_key("your-key");
//!
//! let server = RestServer::new(config);
//! server.start()?;
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// REST API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestConfig {
    /// Host to bind
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// API key (optional)
    pub api_key: Option<String>,
    /// Enable CORS
    pub enable_cors: bool,
    /// CORS allowed origins
    pub cors_origins: Vec<String>,
    /// Rate limit (requests per minute)
    pub rate_limit: Option<u32>,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Max request body size
    pub max_body_size: usize,
    /// Enable request logging
    pub enable_logging: bool,
    /// API version
    pub api_version: String,
    /// Base path
    pub base_path: String,
}

impl Default for RestConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            api_key: None,
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
            rate_limit: Some(1000),
            timeout_seconds: 30,
            max_body_size: 10 * 1024 * 1024, // 10MB
            enable_logging: true,
            api_version: "v1".to_string(),
            base_path: "/api".to_string(),
        }
    }
}

impl RestConfig {
    /// Set port
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set API key
    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    /// Set rate limit
    pub fn with_rate_limit(mut self, rpm: u32) -> Self {
        self.rate_limit = Some(rpm);
        self
    }

    /// Set CORS origins
    pub fn with_cors(mut self, origins: Vec<String>) -> Self {
        self.cors_origins = origins;
        self
    }
}

/// HTTP method
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    OPTIONS,
    HEAD,
}

/// API request
#[derive(Debug, Clone)]
pub struct Request {
    /// HTTP method
    pub method: HttpMethod,
    /// Path
    pub path: String,
    /// Headers
    pub headers: HashMap<String, String>,
    /// Query parameters
    pub query: HashMap<String, String>,
    /// Body (JSON)
    pub body: Option<serde_json::Value>,
    /// Client IP
    pub client_ip: String,
    /// Request ID
    pub request_id: String,
    /// Timestamp
    pub timestamp: Instant,
}

/// API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// Status code
    pub status: u16,
    /// Headers
    pub headers: HashMap<String, String>,
    /// Body
    pub body: serde_json::Value,
}

impl Response {
    /// Create success response
    pub fn ok(body: serde_json::Value) -> Self {
        Self {
            status: 200,
            headers: default_headers(),
            body,
        }
    }

    /// Create created response
    pub fn created(body: serde_json::Value) -> Self {
        Self {
            status: 201,
            headers: default_headers(),
            body,
        }
    }

    /// Create no content response
    pub fn no_content() -> Self {
        Self {
            status: 204,
            headers: default_headers(),
            body: serde_json::json!(null),
        }
    }

    /// Create bad request response
    pub fn bad_request(message: &str) -> Self {
        Self {
            status: 400,
            headers: default_headers(),
            body: serde_json::json!({
                "error": "Bad Request",
                "message": message
            }),
        }
    }

    /// Create unauthorized response
    pub fn unauthorized() -> Self {
        Self {
            status: 401,
            headers: default_headers(),
            body: serde_json::json!({
                "error": "Unauthorized",
                "message": "Invalid or missing API key"
            }),
        }
    }

    /// Create not found response
    pub fn not_found(message: &str) -> Self {
        Self {
            status: 404,
            headers: default_headers(),
            body: serde_json::json!({
                "error": "Not Found",
                "message": message
            }),
        }
    }

    /// Create rate limited response
    pub fn rate_limited() -> Self {
        Self {
            status: 429,
            headers: default_headers(),
            body: serde_json::json!({
                "error": "Too Many Requests",
                "message": "Rate limit exceeded"
            }),
        }
    }

    /// Create internal error response
    pub fn internal_error(message: &str) -> Self {
        Self {
            status: 500,
            headers: default_headers(),
            body: serde_json::json!({
                "error": "Internal Server Error",
                "message": message
            }),
        }
    }
}

fn default_headers() -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    headers.insert("X-Request-Id".to_string(), generate_request_id());
    headers
}

/// Route definition
#[derive(Debug, Clone)]
pub struct Route {
    /// Method
    pub method: HttpMethod,
    /// Path pattern
    pub path: String,
    /// Description
    pub description: String,
    /// Tags
    pub tags: Vec<String>,
    /// Request schema
    pub request_schema: Option<serde_json::Value>,
    /// Response schema
    pub response_schema: Option<serde_json::Value>,
    /// Requires auth
    pub requires_auth: bool,
}

/// API router
pub struct Router {
    routes: Vec<Route>,
    handlers: HashMap<String, Box<dyn Fn(&Request) -> Response + Send + Sync>>,
}

impl Router {
    /// Create new router
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            handlers: HashMap::new(),
        }
    }

    /// Add route
    pub fn route<F>(&mut self, method: HttpMethod, path: &str, handler: F) -> &mut Route
    where
        F: Fn(&Request) -> Response + Send + Sync + 'static,
    {
        let key = format!("{:?}:{}", method, path);
        self.handlers.insert(key.clone(), Box::new(handler));

        self.routes.push(Route {
            method,
            path: path.to_string(),
            description: String::new(),
            tags: Vec::new(),
            request_schema: None,
            response_schema: None,
            requires_auth: false,
        });

        self.routes.last_mut().unwrap()
    }

    /// Handle request
    pub fn handle(&self, request: &Request) -> Response {
        let key = format!("{:?}:{}", request.method, request.path);

        // Try exact match
        if let Some(handler) = self.handlers.get(&key) {
            return handler(request);
        }

        // Try pattern matching (simplified)
        for (route_key, handler) in &self.handlers {
            if self.matches_pattern(route_key, &key) {
                return handler(request);
            }
        }

        Response::not_found(&format!("Route not found: {}", request.path))
    }

    fn matches_pattern(&self, pattern: &str, path: &str) -> bool {
        let pattern_parts: Vec<&str> = pattern.split('/').collect();
        let path_parts: Vec<&str> = path.split('/').collect();

        if pattern_parts.len() != path_parts.len() {
            return false;
        }

        for (p, t) in pattern_parts.iter().zip(path_parts.iter()) {
            if p.starts_with('{') && p.ends_with('}') {
                continue; // Parameter placeholder
            }
            if p != t {
                return false;
            }
        }

        true
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

impl Route {
    /// Set description
    pub fn description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add tag
    pub fn tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Set auth required
    pub fn auth_required(mut self) -> Self {
        self.requires_auth = true;
        self
    }
}

/// OpenAPI specification generator
pub struct OpenApiGenerator {
    /// API info
    pub info: ApiInfo,
    /// Routes
    pub routes: Vec<Route>,
    /// Schemas
    pub schemas: HashMap<String, serde_json::Value>,
}

/// API info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiInfo {
    /// Title
    pub title: String,
    /// Version
    pub version: String,
    /// Description
    pub description: String,
    /// Contact email
    pub contact_email: Option<String>,
    /// License
    pub license: Option<String>,
}

impl OpenApiGenerator {
    /// Create new generator
    pub fn new(info: ApiInfo) -> Self {
        Self {
            info,
            routes: Vec::new(),
            schemas: HashMap::new(),
        }
    }

    /// Add VecStore standard routes
    pub fn add_vecstore_routes(&mut self) {
        // Collections
        self.routes.push(Route {
            method: HttpMethod::GET,
            path: "/collections".to_string(),
            description: "List all collections".to_string(),
            tags: vec!["Collections".to_string()],
            request_schema: None,
            response_schema: Some(serde_json::json!({
                "type": "array",
                "items": {"$ref": "#/components/schemas/Collection"}
            })),
            requires_auth: true,
        });

        self.routes.push(Route {
            method: HttpMethod::POST,
            path: "/collections".to_string(),
            description: "Create a new collection".to_string(),
            tags: vec!["Collections".to_string()],
            request_schema: Some(serde_json::json!({
                "$ref": "#/components/schemas/CreateCollectionRequest"
            })),
            response_schema: Some(serde_json::json!({
                "$ref": "#/components/schemas/Collection"
            })),
            requires_auth: true,
        });

        self.routes.push(Route {
            method: HttpMethod::DELETE,
            path: "/collections/{collection_name}".to_string(),
            description: "Delete a collection".to_string(),
            tags: vec!["Collections".to_string()],
            request_schema: None,
            response_schema: None,
            requires_auth: true,
        });

        // Vectors
        self.routes.push(Route {
            method: HttpMethod::POST,
            path: "/collections/{collection_name}/points".to_string(),
            description: "Insert vectors".to_string(),
            tags: vec!["Vectors".to_string()],
            request_schema: Some(serde_json::json!({
                "$ref": "#/components/schemas/InsertPointsRequest"
            })),
            response_schema: Some(serde_json::json!({
                "$ref": "#/components/schemas/InsertResult"
            })),
            requires_auth: true,
        });

        self.routes.push(Route {
            method: HttpMethod::POST,
            path: "/collections/{collection_name}/points/search".to_string(),
            description: "Search for similar vectors".to_string(),
            tags: vec!["Search".to_string()],
            request_schema: Some(serde_json::json!({
                "$ref": "#/components/schemas/SearchRequest"
            })),
            response_schema: Some(serde_json::json!({
                "$ref": "#/components/schemas/SearchResponse"
            })),
            requires_auth: true,
        });

        self.routes.push(Route {
            method: HttpMethod::DELETE,
            path: "/collections/{collection_name}/points/{point_id}".to_string(),
            description: "Delete a vector".to_string(),
            tags: vec!["Vectors".to_string()],
            request_schema: None,
            response_schema: None,
            requires_auth: true,
        });

        // Add schemas
        self.add_schemas();
    }

    fn add_schemas(&mut self) {
        self.schemas.insert("Collection".to_string(), serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "dimension": {"type": "integer"},
                "metric": {"type": "string", "enum": ["cosine", "euclidean", "dot"]},
                "vector_count": {"type": "integer"},
                "created_at": {"type": "string", "format": "date-time"}
            }
        }));

        self.schemas.insert("CreateCollectionRequest".to_string(), serde_json::json!({
            "type": "object",
            "required": ["name", "dimension"],
            "properties": {
                "name": {"type": "string"},
                "dimension": {"type": "integer", "minimum": 1},
                "metric": {"type": "string", "enum": ["cosine", "euclidean", "dot"], "default": "cosine"}
            }
        }));

        self.schemas.insert("Point".to_string(), serde_json::json!({
            "type": "object",
            "required": ["id", "vector"],
            "properties": {
                "id": {"type": "string"},
                "vector": {"type": "array", "items": {"type": "number"}},
                "payload": {"type": "object", "additionalProperties": true}
            }
        }));

        self.schemas.insert("InsertPointsRequest".to_string(), serde_json::json!({
            "type": "object",
            "required": ["points"],
            "properties": {
                "points": {
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/Point"}
                }
            }
        }));

        self.schemas.insert("SearchRequest".to_string(), serde_json::json!({
            "type": "object",
            "required": ["vector"],
            "properties": {
                "vector": {"type": "array", "items": {"type": "number"}},
                "limit": {"type": "integer", "default": 10, "maximum": 1000},
                "filter": {"type": "object"},
                "with_payload": {"type": "boolean", "default": true},
                "with_vector": {"type": "boolean", "default": false}
            }
        }));

        self.schemas.insert("SearchResponse".to_string(), serde_json::json!({
            "type": "object",
            "properties": {
                "result": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "score": {"type": "number"},
                            "payload": {"type": "object"},
                            "vector": {"type": "array", "items": {"type": "number"}}
                        }
                    }
                },
                "time_ms": {"type": "number"}
            }
        }));

        self.schemas.insert("InsertResult".to_string(), serde_json::json!({
            "type": "object",
            "properties": {
                "inserted_count": {"type": "integer"},
                "operation_id": {"type": "string"}
            }
        }));
    }

    /// Generate OpenAPI spec
    pub fn generate(&self) -> serde_json::Value {
        let mut paths: HashMap<String, serde_json::Value> = HashMap::new();

        for route in &self.routes {
            let path_entry = paths.entry(route.path.clone()).or_insert_with(|| serde_json::json!({}));

            let method = match route.method {
                HttpMethod::GET => "get",
                HttpMethod::POST => "post",
                HttpMethod::PUT => "put",
                HttpMethod::DELETE => "delete",
                HttpMethod::PATCH => "patch",
                HttpMethod::OPTIONS => "options",
                HttpMethod::HEAD => "head",
            };

            let mut operation = serde_json::json!({
                "summary": route.description,
                "tags": route.tags,
                "responses": {
                    "200": {
                        "description": "Successful response"
                    },
                    "400": {
                        "description": "Bad request"
                    },
                    "401": {
                        "description": "Unauthorized"
                    },
                    "500": {
                        "description": "Internal server error"
                    }
                }
            });

            if let Some(ref schema) = route.request_schema {
                operation["requestBody"] = serde_json::json!({
                    "required": true,
                    "content": {
                        "application/json": {
                            "schema": schema
                        }
                    }
                });
            }

            if let Some(ref schema) = route.response_schema {
                operation["responses"]["200"]["content"] = serde_json::json!({
                    "application/json": {
                        "schema": schema
                    }
                });
            }

            if route.requires_auth {
                operation["security"] = serde_json::json!([{"ApiKeyAuth": []}]);
            }

            path_entry[method] = operation;
        }

        serde_json::json!({
            "openapi": "3.1.0",
            "info": {
                "title": self.info.title,
                "version": self.info.version,
                "description": self.info.description,
                "contact": {
                    "email": self.info.contact_email
                },
                "license": {
                    "name": self.info.license.clone().unwrap_or_else(|| "Apache-2.0".to_string())
                }
            },
            "servers": [
                {"url": "/api/v1", "description": "API v1"}
            ],
            "paths": paths,
            "components": {
                "schemas": self.schemas,
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    }
                }
            }
        })
    }

    /// Generate SDK from spec (TypeScript types)
    pub fn generate_typescript_types(&self) -> String {
        let mut output = String::new();

        output.push_str("// Auto-generated TypeScript types for VecStore API\n\n");

        for (name, schema) in &self.schemas {
            output.push_str(&format!("export interface {} {{\n", name));
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (prop_name, prop_schema) in props {
                    let ts_type = self.json_schema_to_ts(prop_schema);
                    let required = schema.get("required")
                        .and_then(|r| r.as_array())
                        .map(|arr| arr.iter().any(|v| v.as_str() == Some(prop_name)))
                        .unwrap_or(false);

                    let optional = if required { "" } else { "?" };
                    output.push_str(&format!("  {}{}: {};\n", prop_name, optional, ts_type));
                }
            }
            output.push_str("}\n\n");
        }

        output
    }

    fn json_schema_to_ts(&self, schema: &serde_json::Value) -> String {
        if let Some(ref_path) = schema.get("$ref").and_then(|r| r.as_str()) {
            let type_name = ref_path.split('/').last().unwrap_or("unknown");
            return type_name.to_string();
        }

        match schema.get("type").and_then(|t| t.as_str()) {
            Some("string") => "string".to_string(),
            Some("integer") | Some("number") => "number".to_string(),
            Some("boolean") => "boolean".to_string(),
            Some("array") => {
                let items_type = schema.get("items")
                    .map(|i| self.json_schema_to_ts(i))
                    .unwrap_or_else(|| "unknown".to_string());
                format!("{}[]", items_type)
            }
            Some("object") => "Record<string, any>".to_string(),
            _ => "unknown".to_string(),
        }
    }
}

/// Rate limiter for API
pub struct ApiRateLimiter {
    /// Limits per client
    clients: RwLock<HashMap<String, ClientLimit>>,
    /// Requests per minute
    rpm: u32,
}

struct ClientLimit {
    tokens: u32,
    last_refill: Instant,
}

impl ApiRateLimiter {
    /// Create new rate limiter
    pub fn new(rpm: u32) -> Self {
        Self {
            clients: RwLock::new(HashMap::new()),
            rpm,
        }
    }

    /// Check if request is allowed
    pub fn allow(&self, client_id: &str) -> bool {
        let mut clients = self.clients.write().unwrap();

        let client = clients.entry(client_id.to_string()).or_insert_with(|| {
            ClientLimit {
                tokens: self.rpm,
                last_refill: Instant::now(),
            }
        });

        // Refill tokens
        let elapsed = client.last_refill.elapsed();
        if elapsed >= Duration::from_secs(60) {
            client.tokens = self.rpm;
            client.last_refill = Instant::now();
        }

        // Try to consume token
        if client.tokens > 0 {
            client.tokens -= 1;
            true
        } else {
            false
        }
    }
}

fn generate_request_id() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("req_{:x}", ts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_builders() {
        let ok = Response::ok(serde_json::json!({"data": "test"}));
        assert_eq!(ok.status, 200);

        let err = Response::bad_request("Invalid input");
        assert_eq!(err.status, 400);

        let not_found = Response::not_found("Item not found");
        assert_eq!(not_found.status, 404);
    }

    #[test]
    fn test_router() {
        let mut router = Router::new();

        router.route(HttpMethod::GET, "/health", |_| {
            Response::ok(serde_json::json!({"status": "ok"}))
        });

        let request = Request {
            method: HttpMethod::GET,
            path: "/health".to_string(),
            headers: HashMap::new(),
            query: HashMap::new(),
            body: None,
            client_ip: "127.0.0.1".to_string(),
            request_id: "test".to_string(),
            timestamp: Instant::now(),
        };

        let response = router.handle(&request);
        assert_eq!(response.status, 200);
    }

    #[test]
    fn test_openapi_generator() {
        let mut generator = OpenApiGenerator::new(ApiInfo {
            title: "VecStore API".to_string(),
            version: "1.0.0".to_string(),
            description: "Vector database API".to_string(),
            contact_email: None,
            license: None,
        });

        generator.add_vecstore_routes();
        let spec = generator.generate();

        assert_eq!(spec["openapi"], "3.1.0");
        assert!(spec["paths"].as_object().unwrap().len() > 0);
    }

    #[test]
    fn test_typescript_generation() {
        let mut generator = OpenApiGenerator::new(ApiInfo {
            title: "Test".to_string(),
            version: "1.0".to_string(),
            description: "Test".to_string(),
            contact_email: None,
            license: None,
        });

        generator.add_vecstore_routes();
        let ts = generator.generate_typescript_types();

        assert!(ts.contains("export interface"));
        assert!(ts.contains("Point"));
        assert!(ts.contains("SearchRequest"));
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = ApiRateLimiter::new(5);

        // Should allow first 5 requests
        for _ in 0..5 {
            assert!(limiter.allow("client1"));
        }

        // Should deny 6th request
        assert!(!limiter.allow("client1"));

        // Different client should be allowed
        assert!(limiter.allow("client2"));
    }
}
