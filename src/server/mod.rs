//! Server mode for vecstore
//!
//! Provides gRPC and HTTP/REST APIs for remote access to vecstore.

#[cfg(feature = "server")]
pub mod admin;

#[cfg(feature = "server")]
pub mod admin_http;

#[cfg(feature = "server")]
pub mod grpc;

#[cfg(feature = "server")]
pub mod http;

#[cfg(feature = "server")]
pub mod types;

#[cfg(feature = "server")]
pub mod metrics;

#[cfg(feature = "server")]
pub use admin::AdminService;

#[cfg(feature = "server")]
pub use admin_http::AdminHttpServer;

#[cfg(feature = "server")]
pub use grpc::VecStoreGrpcServer;

#[cfg(feature = "server")]
pub use http::VecStoreHttpServer;
