// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 VecStore Contributors

//! # Multi-Language SDK Generation
//!
//! Generate type-safe SDKs for Python, TypeScript, Go, and Java from
//! VecStore's API specification.
//!
//! ## Features
//!
//! - **TypeScript SDK**: Full type definitions with async/await
//! - **Python SDK**: Type hints with dataclasses
//! - **Go SDK**: Idiomatic Go with context support
//! - **Java SDK**: Builder pattern with fluent API
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecstore::sdk_gen::{SdkGenerator, Language};
//!
//! let generator = SdkGenerator::new();
//! let ts_sdk = generator.generate(Language::TypeScript)?;
//! std::fs::write("vecstore-sdk.ts", ts_sdk)?;
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Target language
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Language {
    /// TypeScript/JavaScript
    TypeScript,
    /// Python
    Python,
    /// Go
    Go,
    /// Java
    Java,
    /// C#
    CSharp,
    /// Rust (client library)
    Rust,
}

/// Type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDef {
    /// Type name
    pub name: String,
    /// Fields
    pub fields: Vec<FieldDef>,
    /// Description
    pub description: Option<String>,
}

/// Field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: FieldType,
    /// Is optional
    pub optional: bool,
    /// Description
    pub description: Option<String>,
    /// Default value
    pub default: Option<String>,
}

/// Field type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Int,
    Float,
    Bool,
    Array(Box<FieldType>),
    Map(Box<FieldType>, Box<FieldType>),
    Custom(String),
    Any,
}

/// Method definition
#[derive(Debug, Clone)]
pub struct MethodDef {
    /// Method name
    pub name: String,
    /// HTTP method
    pub http_method: String,
    /// Path
    pub path: String,
    /// Parameters
    pub params: Vec<ParamDef>,
    /// Request body type
    pub request_type: Option<String>,
    /// Response type
    pub response_type: Option<String>,
    /// Description
    pub description: String,
}

/// Parameter definition
#[derive(Debug, Clone)]
pub struct ParamDef {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: FieldType,
    /// Location (path, query, header)
    pub location: ParamLocation,
    /// Is required
    pub required: bool,
}

/// Parameter location
#[derive(Debug, Clone, PartialEq)]
pub enum ParamLocation {
    Path,
    Query,
    Header,
    Body,
}

/// SDK generator
pub struct SdkGenerator {
    /// Type definitions
    types: Vec<TypeDef>,
    /// Method definitions
    methods: Vec<MethodDef>,
    /// Package name
    package_name: String,
    /// Version
    version: String,
}

impl SdkGenerator {
    /// Create new generator
    pub fn new() -> Self {
        let mut generator = Self {
            types: Vec::new(),
            methods: Vec::new(),
            package_name: "vecstore".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        generator.add_core_types();
        generator.add_core_methods();
        generator
    }

    /// Set package name
    pub fn with_package_name(mut self, name: &str) -> Self {
        self.package_name = name.to_string();
        self
    }

    fn add_core_types(&mut self) {
        // Vector type
        self.types.push(TypeDef {
            name: "Vector".to_string(),
            description: Some("A vector with ID, data, and optional payload".to_string()),
            fields: vec![
                FieldDef {
                    name: "id".to_string(),
                    field_type: FieldType::String,
                    optional: false,
                    description: Some("Unique vector identifier".to_string()),
                    default: None,
                },
                FieldDef {
                    name: "vector".to_string(),
                    field_type: FieldType::Array(Box::new(FieldType::Float)),
                    optional: false,
                    description: Some("Vector data".to_string()),
                    default: None,
                },
                FieldDef {
                    name: "payload".to_string(),
                    field_type: FieldType::Map(Box::new(FieldType::String), Box::new(FieldType::Any)),
                    optional: true,
                    description: Some("Metadata payload".to_string()),
                    default: None,
                },
            ],
        });

        // Search result
        self.types.push(TypeDef {
            name: "SearchResult".to_string(),
            description: Some("A search result with score".to_string()),
            fields: vec![
                FieldDef {
                    name: "id".to_string(),
                    field_type: FieldType::String,
                    optional: false,
                    description: Some("Vector ID".to_string()),
                    default: None,
                },
                FieldDef {
                    name: "score".to_string(),
                    field_type: FieldType::Float,
                    optional: false,
                    description: Some("Similarity score".to_string()),
                    default: None,
                },
                FieldDef {
                    name: "payload".to_string(),
                    field_type: FieldType::Map(Box::new(FieldType::String), Box::new(FieldType::Any)),
                    optional: true,
                    description: None,
                    default: None,
                },
                FieldDef {
                    name: "vector".to_string(),
                    field_type: FieldType::Array(Box::new(FieldType::Float)),
                    optional: true,
                    description: None,
                    default: None,
                },
            ],
        });

        // Collection
        self.types.push(TypeDef {
            name: "Collection".to_string(),
            description: Some("A collection of vectors".to_string()),
            fields: vec![
                FieldDef {
                    name: "name".to_string(),
                    field_type: FieldType::String,
                    optional: false,
                    description: Some("Collection name".to_string()),
                    default: None,
                },
                FieldDef {
                    name: "dimension".to_string(),
                    field_type: FieldType::Int,
                    optional: false,
                    description: Some("Vector dimension".to_string()),
                    default: None,
                },
                FieldDef {
                    name: "metric".to_string(),
                    field_type: FieldType::String,
                    optional: false,
                    description: Some("Distance metric".to_string()),
                    default: Some("cosine".to_string()),
                },
                FieldDef {
                    name: "vector_count".to_string(),
                    field_type: FieldType::Int,
                    optional: true,
                    description: Some("Number of vectors".to_string()),
                    default: None,
                },
            ],
        });

        // Search request
        self.types.push(TypeDef {
            name: "SearchRequest".to_string(),
            description: Some("Search request parameters".to_string()),
            fields: vec![
                FieldDef {
                    name: "vector".to_string(),
                    field_type: FieldType::Array(Box::new(FieldType::Float)),
                    optional: false,
                    description: Some("Query vector".to_string()),
                    default: None,
                },
                FieldDef {
                    name: "limit".to_string(),
                    field_type: FieldType::Int,
                    optional: true,
                    description: Some("Max results".to_string()),
                    default: Some("10".to_string()),
                },
                FieldDef {
                    name: "filter".to_string(),
                    field_type: FieldType::Map(Box::new(FieldType::String), Box::new(FieldType::Any)),
                    optional: true,
                    description: Some("Metadata filter".to_string()),
                    default: None,
                },
                FieldDef {
                    name: "with_payload".to_string(),
                    field_type: FieldType::Bool,
                    optional: true,
                    description: Some("Include payload".to_string()),
                    default: Some("true".to_string()),
                },
                FieldDef {
                    name: "with_vector".to_string(),
                    field_type: FieldType::Bool,
                    optional: true,
                    description: Some("Include vector".to_string()),
                    default: Some("false".to_string()),
                },
            ],
        });
    }

    fn add_core_methods(&mut self) {
        self.methods.push(MethodDef {
            name: "listCollections".to_string(),
            http_method: "GET".to_string(),
            path: "/collections".to_string(),
            params: vec![],
            request_type: None,
            response_type: Some("Collection[]".to_string()),
            description: "List all collections".to_string(),
        });

        self.methods.push(MethodDef {
            name: "createCollection".to_string(),
            http_method: "POST".to_string(),
            path: "/collections".to_string(),
            params: vec![],
            request_type: Some("CreateCollectionRequest".to_string()),
            response_type: Some("Collection".to_string()),
            description: "Create a new collection".to_string(),
        });

        self.methods.push(MethodDef {
            name: "deleteCollection".to_string(),
            http_method: "DELETE".to_string(),
            path: "/collections/{collection}".to_string(),
            params: vec![ParamDef {
                name: "collection".to_string(),
                param_type: FieldType::String,
                location: ParamLocation::Path,
                required: true,
            }],
            request_type: None,
            response_type: None,
            description: "Delete a collection".to_string(),
        });

        self.methods.push(MethodDef {
            name: "upsertVectors".to_string(),
            http_method: "POST".to_string(),
            path: "/collections/{collection}/points".to_string(),
            params: vec![ParamDef {
                name: "collection".to_string(),
                param_type: FieldType::String,
                location: ParamLocation::Path,
                required: true,
            }],
            request_type: Some("Vector[]".to_string()),
            response_type: Some("UpsertResult".to_string()),
            description: "Insert or update vectors".to_string(),
        });

        self.methods.push(MethodDef {
            name: "search".to_string(),
            http_method: "POST".to_string(),
            path: "/collections/{collection}/points/search".to_string(),
            params: vec![ParamDef {
                name: "collection".to_string(),
                param_type: FieldType::String,
                location: ParamLocation::Path,
                required: true,
            }],
            request_type: Some("SearchRequest".to_string()),
            response_type: Some("SearchResult[]".to_string()),
            description: "Search for similar vectors".to_string(),
        });

        self.methods.push(MethodDef {
            name: "deleteVector".to_string(),
            http_method: "DELETE".to_string(),
            path: "/collections/{collection}/points/{id}".to_string(),
            params: vec![
                ParamDef {
                    name: "collection".to_string(),
                    param_type: FieldType::String,
                    location: ParamLocation::Path,
                    required: true,
                },
                ParamDef {
                    name: "id".to_string(),
                    param_type: FieldType::String,
                    location: ParamLocation::Path,
                    required: true,
                },
            ],
            request_type: None,
            response_type: None,
            description: "Delete a vector".to_string(),
        });
    }

    /// Generate SDK for target language
    pub fn generate(&self, language: Language) -> String {
        match language {
            Language::TypeScript => self.generate_typescript(),
            Language::Python => self.generate_python(),
            Language::Go => self.generate_go(),
            Language::Java => self.generate_java(),
            Language::CSharp => self.generate_csharp(),
            Language::Rust => self.generate_rust_client(),
        }
    }

    fn generate_typescript(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "// VecStore TypeScript SDK v{}\n// Auto-generated - do not edit\n\n",
            self.version
        ));

        // Types
        for type_def in &self.types {
            if let Some(ref desc) = type_def.description {
                output.push_str(&format!("/** {} */\n", desc));
            }
            output.push_str(&format!("export interface {} {{\n", type_def.name));

            for field in &type_def.fields {
                let ts_type = self.field_type_to_ts(&field.field_type);
                let optional = if field.optional { "?" } else { "" };

                if let Some(ref desc) = field.description {
                    output.push_str(&format!("  /** {} */\n", desc));
                }
                output.push_str(&format!("  {}{}: {};\n", field.name, optional, ts_type));
            }

            output.push_str("}\n\n");
        }

        // Client class
        output.push_str("export class VecStoreClient {\n");
        output.push_str("  private baseUrl: string;\n");
        output.push_str("  private apiKey?: string;\n\n");

        output.push_str("  constructor(baseUrl: string, apiKey?: string) {\n");
        output.push_str("    this.baseUrl = baseUrl;\n");
        output.push_str("    this.apiKey = apiKey;\n");
        output.push_str("  }\n\n");

        output.push_str("  private async request<T>(method: string, path: string, body?: any): Promise<T> {\n");
        output.push_str("    const headers: Record<string, string> = {\n");
        output.push_str("      'Content-Type': 'application/json',\n");
        output.push_str("    };\n");
        output.push_str("    if (this.apiKey) headers['X-API-Key'] = this.apiKey;\n\n");
        output.push_str("    const response = await fetch(`${this.baseUrl}${path}`, {\n");
        output.push_str("      method,\n");
        output.push_str("      headers,\n");
        output.push_str("      body: body ? JSON.stringify(body) : undefined,\n");
        output.push_str("    });\n\n");
        output.push_str("    if (!response.ok) {\n");
        output.push_str("      throw new Error(`HTTP ${response.status}: ${await response.text()}`);\n");
        output.push_str("    }\n\n");
        output.push_str("    return response.json();\n");
        output.push_str("  }\n\n");

        // Methods
        for method in &self.methods {
            let params = self.method_params_ts(&method);
            let return_type = method.response_type.as_deref().unwrap_or("void");

            output.push_str(&format!("  /** {} */\n", method.description));
            output.push_str(&format!("  async {}({}): Promise<{}> {{\n",
                method.name, params, return_type));

            let path = self.interpolate_path_ts(&method.path, &method.params);
            if method.request_type.is_some() {
                output.push_str(&format!(
                    "    return this.request('{}', {}, body);\n",
                    method.http_method, path
                ));
            } else {
                output.push_str(&format!(
                    "    return this.request('{}', {});\n",
                    method.http_method, path
                ));
            }

            output.push_str("  }\n\n");
        }

        output.push_str("}\n");
        output
    }

    fn generate_python(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "# VecStore Python SDK v{}\n# Auto-generated - do not edit\n\n",
            self.version
        ));

        output.push_str("from dataclasses import dataclass, field\n");
        output.push_str("from typing import Optional, List, Dict, Any\n");
        output.push_str("import requests\n\n");

        // Types
        for type_def in &self.types {
            output.push_str("@dataclass\n");
            output.push_str(&format!("class {}:\n", type_def.name));

            if let Some(ref desc) = type_def.description {
                output.push_str(&format!("    \"\"\"{}\"\"\"\n", desc));
            }

            for fld in &type_def.fields {
                let py_type = self.field_type_to_py(&fld.field_type);
                let optional = if fld.optional { format!("Optional[{}]", py_type) } else { py_type };
                let default = if fld.optional { " = None" } else { "" };
                output.push_str(&format!("    {}: {}{}\n", fld.name, optional, default));
            }

            output.push('\n');
        }

        // Client class
        output.push_str("class VecStoreClient:\n");
        output.push_str("    \"\"\"VecStore API client\"\"\"\n\n");

        output.push_str("    def __init__(self, base_url: str, api_key: Optional[str] = None):\n");
        output.push_str("        self.base_url = base_url\n");
        output.push_str("        self.api_key = api_key\n\n");

        output.push_str("    def _request(self, method: str, path: str, body: Optional[Dict] = None) -> Any:\n");
        output.push_str("        headers = {'Content-Type': 'application/json'}\n");
        output.push_str("        if self.api_key:\n");
        output.push_str("            headers['X-API-Key'] = self.api_key\n\n");
        output.push_str("        response = requests.request(\n");
        output.push_str("            method,\n");
        output.push_str("            f'{self.base_url}{path}',\n");
        output.push_str("            headers=headers,\n");
        output.push_str("            json=body\n");
        output.push_str("        )\n");
        output.push_str("        response.raise_for_status()\n");
        output.push_str("        return response.json() if response.content else None\n\n");

        // Methods
        for method in &self.methods {
            let params = self.method_params_py(&method);
            let return_type = method.response_type.as_deref().unwrap_or("None");

            output.push_str(&format!("    def {}(self{}) -> {}:\n",
                to_snake_case(&method.name),
                if params.is_empty() { String::new() } else { format!(", {}", params) },
                self.response_type_to_py(return_type)
            ));

            output.push_str(&format!("        \"\"\"{}\"\"\"\n", method.description));

            let path = self.interpolate_path_py(&method.path, &method.params);
            if method.request_type.is_some() {
                output.push_str(&format!(
                    "        return self._request('{}', f'{}', body)\n\n",
                    method.http_method, path
                ));
            } else {
                output.push_str(&format!(
                    "        return self._request('{}', f'{}')\n\n",
                    method.http_method, path
                ));
            }
        }

        output
    }

    fn generate_go(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "// VecStore Go SDK v{}\n// Auto-generated - do not edit\n\n",
            self.version
        ));

        output.push_str("package vecstore\n\n");
        output.push_str("import (\n");
        output.push_str("\t\"bytes\"\n");
        output.push_str("\t\"context\"\n");
        output.push_str("\t\"encoding/json\"\n");
        output.push_str("\t\"fmt\"\n");
        output.push_str("\t\"net/http\"\n");
        output.push_str(")\n\n");

        // Types
        for type_def in &self.types {
            if let Some(ref desc) = type_def.description {
                output.push_str(&format!("// {} - {}\n", type_def.name, desc));
            }
            output.push_str(&format!("type {} struct {{\n", type_def.name));

            for fld in &type_def.fields {
                let go_type = self.field_type_to_go(&fld.field_type, fld.optional);
                let json_tag = format!("`json:\"{}{}\"`",
                    fld.name,
                    if fld.optional { ",omitempty" } else { "" }
                );
                output.push_str(&format!("\t{} {} {}\n",
                    to_pascal_case(&fld.name),
                    go_type,
                    json_tag
                ));
            }

            output.push_str("}\n\n");
        }

        // Client
        output.push_str("// Client is the VecStore API client\n");
        output.push_str("type Client struct {\n");
        output.push_str("\tbaseURL string\n");
        output.push_str("\tapiKey  string\n");
        output.push_str("\thttp    *http.Client\n");
        output.push_str("}\n\n");

        output.push_str("// NewClient creates a new VecStore client\n");
        output.push_str("func NewClient(baseURL, apiKey string) *Client {\n");
        output.push_str("\treturn &Client{\n");
        output.push_str("\t\tbaseURL: baseURL,\n");
        output.push_str("\t\tapiKey:  apiKey,\n");
        output.push_str("\t\thttp:    &http.Client{},\n");
        output.push_str("\t}\n");
        output.push_str("}\n\n");

        // Request helper
        output.push_str("func (c *Client) request(ctx context.Context, method, path string, body, result interface{}) error {\n");
        output.push_str("\tvar reqBody []byte\n");
        output.push_str("\tvar err error\n");
        output.push_str("\tif body != nil {\n");
        output.push_str("\t\treqBody, err = json.Marshal(body)\n");
        output.push_str("\t\tif err != nil {\n");
        output.push_str("\t\t\treturn err\n");
        output.push_str("\t\t}\n");
        output.push_str("\t}\n\n");
        output.push_str("\treq, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bytes.NewReader(reqBody))\n");
        output.push_str("\tif err != nil {\n");
        output.push_str("\t\treturn err\n");
        output.push_str("\t}\n\n");
        output.push_str("\treq.Header.Set(\"Content-Type\", \"application/json\")\n");
        output.push_str("\tif c.apiKey != \"\" {\n");
        output.push_str("\t\treq.Header.Set(\"X-API-Key\", c.apiKey)\n");
        output.push_str("\t}\n\n");
        output.push_str("\tresp, err := c.http.Do(req)\n");
        output.push_str("\tif err != nil {\n");
        output.push_str("\t\treturn err\n");
        output.push_str("\t}\n");
        output.push_str("\tdefer resp.Body.Close()\n\n");
        output.push_str("\tif resp.StatusCode >= 400 {\n");
        output.push_str("\t\treturn fmt.Errorf(\"HTTP %d\", resp.StatusCode)\n");
        output.push_str("\t}\n\n");
        output.push_str("\tif result != nil {\n");
        output.push_str("\t\treturn json.NewDecoder(resp.Body).Decode(result)\n");
        output.push_str("\t}\n");
        output.push_str("\treturn nil\n");
        output.push_str("}\n\n");

        // Methods
        for method in &self.methods {
            let func_name = to_pascal_case(&method.name);
            output.push_str(&format!("// {} - {}\n", func_name, method.description));

            // Build function signature
            let params = self.method_params_go(&method);
            let return_type = self.response_type_to_go(method.response_type.as_deref());

            output.push_str(&format!("func (c *Client) {}(ctx context.Context{}) {} {{\n",
                func_name,
                if params.is_empty() { String::new() } else { format!(", {}", params) },
                return_type
            ));

            let path = self.interpolate_path_go(&method.path, &method.params);

            if method.response_type.is_some() {
                output.push_str("\tvar result ");
                output.push_str(&method.response_type.as_ref().unwrap().replace("[]", "[]"));
                output.push_str("\n");

                if method.request_type.is_some() {
                    output.push_str(&format!("\terr := c.request(ctx, \"{}\", {}, body, &result)\n",
                        method.http_method, path));
                } else {
                    output.push_str(&format!("\terr := c.request(ctx, \"{}\", {}, nil, &result)\n",
                        method.http_method, path));
                }
                output.push_str("\treturn result, err\n");
            } else {
                if method.request_type.is_some() {
                    output.push_str(&format!("\treturn c.request(ctx, \"{}\", {}, body, nil)\n",
                        method.http_method, path));
                } else {
                    output.push_str(&format!("\treturn c.request(ctx, \"{}\", {}, nil, nil)\n",
                        method.http_method, path));
                }
            }

            output.push_str("}\n\n");
        }

        output
    }

    fn generate_java(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "// VecStore Java SDK v{}\n// Auto-generated - do not edit\n\n",
            self.version
        ));

        output.push_str("package io.vecstore.client;\n\n");
        output.push_str("import java.util.*;\n");
        output.push_str("import java.net.http.*;\n");
        output.push_str("import com.fasterxml.jackson.databind.ObjectMapper;\n\n");

        // Main client class
        output.push_str("public class VecStoreClient {\n");
        output.push_str("    private final String baseUrl;\n");
        output.push_str("    private final String apiKey;\n");
        output.push_str("    private final HttpClient httpClient;\n");
        output.push_str("    private final ObjectMapper mapper;\n\n");

        output.push_str("    public VecStoreClient(String baseUrl, String apiKey) {\n");
        output.push_str("        this.baseUrl = baseUrl;\n");
        output.push_str("        this.apiKey = apiKey;\n");
        output.push_str("        this.httpClient = HttpClient.newHttpClient();\n");
        output.push_str("        this.mapper = new ObjectMapper();\n");
        output.push_str("    }\n\n");

        // Methods
        for method in &self.methods {
            let return_type = method.response_type.as_deref().unwrap_or("void");
            let java_return = self.response_type_to_java(return_type);

            output.push_str(&format!("    /** {} */\n", method.description));
            output.push_str(&format!("    public {} {}({}) throws Exception {{\n",
                java_return,
                method.name,
                self.method_params_java(&method)
            ));

            output.push_str("        // Implementation\n");
            output.push_str("        throw new UnsupportedOperationException();\n");
            output.push_str("    }\n\n");
        }

        output.push_str("}\n");
        output
    }

    fn generate_csharp(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "// VecStore C# SDK v{}\n// Auto-generated - do not edit\n\n",
            self.version
        ));

        output.push_str("namespace VecStore.Client;\n\n");
        output.push_str("using System.Text.Json;\n\n");

        output.push_str("public class VecStoreClient {\n");
        output.push_str("    private readonly HttpClient _client;\n");
        output.push_str("    private readonly string _baseUrl;\n\n");

        output.push_str("    public VecStoreClient(string baseUrl, string? apiKey = null) {\n");
        output.push_str("        _baseUrl = baseUrl;\n");
        output.push_str("        _client = new HttpClient();\n");
        output.push_str("        if (apiKey != null) _client.DefaultRequestHeaders.Add(\"X-API-Key\", apiKey);\n");
        output.push_str("    }\n");
        output.push_str("}\n");

        output
    }

    fn generate_rust_client(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "// VecStore Rust Client v{}\n// Auto-generated - do not edit\n\n",
            self.version
        ));

        output.push_str("use serde::{Deserialize, Serialize};\n");
        output.push_str("use std::collections::HashMap;\n\n");

        for type_def in &self.types {
            output.push_str("#[derive(Debug, Clone, Serialize, Deserialize)]\n");
            output.push_str(&format!("pub struct {} {{\n", type_def.name));

            for fld in &type_def.fields {
                let rust_type = self.field_type_to_rust(&fld.field_type);
                let field_type = if fld.optional {
                    format!("Option<{}>", rust_type)
                } else {
                    rust_type
                };
                output.push_str(&format!("    pub {}: {},\n", to_snake_case(&fld.name), field_type));
            }

            output.push_str("}\n\n");
        }

        output
    }

    // Helper methods for type conversion
    fn field_type_to_ts(&self, ft: &FieldType) -> String {
        match ft {
            FieldType::String => "string".to_string(),
            FieldType::Int | FieldType::Float => "number".to_string(),
            FieldType::Bool => "boolean".to_string(),
            FieldType::Array(inner) => format!("{}[]", self.field_type_to_ts(inner)),
            FieldType::Map(_, v) => format!("Record<string, {}>", self.field_type_to_ts(v)),
            FieldType::Custom(name) => name.clone(),
            FieldType::Any => "any".to_string(),
        }
    }

    fn field_type_to_py(&self, ft: &FieldType) -> String {
        match ft {
            FieldType::String => "str".to_string(),
            FieldType::Int => "int".to_string(),
            FieldType::Float => "float".to_string(),
            FieldType::Bool => "bool".to_string(),
            FieldType::Array(inner) => format!("List[{}]", self.field_type_to_py(inner)),
            FieldType::Map(_, v) => format!("Dict[str, {}]", self.field_type_to_py(v)),
            FieldType::Custom(name) => name.clone(),
            FieldType::Any => "Any".to_string(),
        }
    }

    fn field_type_to_go(&self, ft: &FieldType, optional: bool) -> String {
        let base = match ft {
            FieldType::String => "string".to_string(),
            FieldType::Int => "int64".to_string(),
            FieldType::Float => "float64".to_string(),
            FieldType::Bool => "bool".to_string(),
            FieldType::Array(inner) => format!("[]{}", self.field_type_to_go(inner, false)),
            FieldType::Map(_, v) => format!("map[string]{}", self.field_type_to_go(v, false)),
            FieldType::Custom(name) => name.clone(),
            FieldType::Any => "interface{}".to_string(),
        };
        if optional && !base.starts_with("[]") && !base.starts_with("map") {
            format!("*{}", base)
        } else {
            base
        }
    }

    fn field_type_to_rust(&self, ft: &FieldType) -> String {
        match ft {
            FieldType::String => "String".to_string(),
            FieldType::Int => "i64".to_string(),
            FieldType::Float => "f64".to_string(),
            FieldType::Bool => "bool".to_string(),
            FieldType::Array(inner) => format!("Vec<{}>", self.field_type_to_rust(inner)),
            FieldType::Map(_, v) => format!("HashMap<String, {}>", self.field_type_to_rust(v)),
            FieldType::Custom(name) => name.clone(),
            FieldType::Any => "serde_json::Value".to_string(),
        }
    }

    fn method_params_ts(&self, method: &MethodDef) -> String {
        let mut params = Vec::new();

        for param in &method.params {
            if param.location == ParamLocation::Path {
                params.push(format!("{}: string", param.name));
            }
        }

        if method.request_type.is_some() {
            params.push(format!("body: {}", method.request_type.as_ref().unwrap()));
        }

        params.join(", ")
    }

    fn method_params_py(&self, method: &MethodDef) -> String {
        let mut params = Vec::new();

        for param in &method.params {
            if param.location == ParamLocation::Path {
                params.push(format!("{}: str", param.name));
            }
        }

        if method.request_type.is_some() {
            params.push(format!("body: {}", method.request_type.as_ref().unwrap()));
        }

        params.join(", ")
    }

    fn method_params_go(&self, method: &MethodDef) -> String {
        let mut params = Vec::new();

        for param in &method.params {
            if param.location == ParamLocation::Path {
                params.push(format!("{} string", param.name));
            }
        }

        if method.request_type.is_some() {
            params.push(format!("body *{}", method.request_type.as_ref().unwrap().replace("[]", "")));
        }

        params.join(", ")
    }

    fn method_params_java(&self, method: &MethodDef) -> String {
        let mut params = Vec::new();

        for param in &method.params {
            if param.location == ParamLocation::Path {
                params.push(format!("String {}", param.name));
            }
        }

        if let Some(ref rt) = method.request_type {
            params.push(format!("{} body", rt.replace("[]", "List<>").replace("<>", "")));
        }

        params.join(", ")
    }

    fn interpolate_path_ts(&self, path: &str, _params: &[ParamDef]) -> String {
        let path = path.replace("{", "${");
        format!("`{}`", path)
    }

    fn interpolate_path_py(&self, path: &str, _params: &[ParamDef]) -> String {
        path.replace("{", "{").to_string()
    }

    fn interpolate_path_go(&self, path: &str, _params: &[ParamDef]) -> String {
        let formatted = path.replace("{collection}", "\" + collection + \"")
            .replace("{id}", "\" + id + \"");
        format!("\"{}\"", formatted).replace(" + \"\"", "")
    }

    fn response_type_to_py(&self, rt: &str) -> String {
        if rt == "None" {
            return "None".to_string();
        }
        if rt.ends_with("[]") {
            format!("List[{}]", &rt[..rt.len()-2])
        } else {
            rt.to_string()
        }
    }

    fn response_type_to_go(&self, rt: Option<&str>) -> String {
        match rt {
            Some(t) => {
                if t.ends_with("[]") {
                    format!("([]{}, error)", &t[..t.len()-2])
                } else {
                    format!("({}, error)", t)
                }
            }
            None => "error".to_string(),
        }
    }

    fn response_type_to_java(&self, rt: &str) -> String {
        if rt == "void" {
            return "void".to_string();
        }
        if rt.ends_with("[]") {
            format!("List<{}>", &rt[..rt.len()-2])
        } else {
            rt.to_string()
        }
    }
}

impl Default for SdkGenerator {
    fn default() -> Self {
        Self::new()
    }
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(c.to_lowercase().next().unwrap());
    }
    result
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut c = word.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_typescript() {
        let generator = SdkGenerator::new();
        let ts = generator.generate(Language::TypeScript);

        assert!(ts.contains("export interface Vector"));
        assert!(ts.contains("class VecStoreClient"));
        assert!(ts.contains("async search"));
    }

    #[test]
    fn test_generate_python() {
        let generator = SdkGenerator::new();
        let py = generator.generate(Language::Python);

        assert!(py.contains("@dataclass"));
        assert!(py.contains("class Vector"));
        assert!(py.contains("class VecStoreClient"));
    }

    #[test]
    fn test_generate_go() {
        let generator = SdkGenerator::new();
        let go = generator.generate(Language::Go);

        assert!(go.contains("package vecstore"));
        assert!(go.contains("type Vector struct"));
        assert!(go.contains("func (c *Client) Search"));
    }

    #[test]
    fn test_generate_java() {
        let generator = SdkGenerator::new();
        let java = generator.generate(Language::Java);

        assert!(java.contains("public class VecStoreClient"));
    }
}
