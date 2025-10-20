//! VecStore Server
//!
//! Multi-protocol vector database server supporting gRPC and HTTP/REST APIs.
//!
//! ## Usage
//!
//! ```bash
//! # Start server with default settings (port 50051 for gRPC, 8080 for HTTP)
//! cargo run --bin vecstore-server --features server
//!
//! # Specify custom ports
//! cargo run --bin vecstore-server --features server -- --grpc-port 9000 --http-port 8000
//!
//! # Specify database path
//! cargo run --bin vecstore-server --features server -- --db-path /data/vectors.db
//! ```

use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Server as TonicServer;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use vecstore::namespace_manager::NamespaceManager;
use vecstore::server::{AdminHttpServer, AdminService, VecStoreGrpcServer, VecStoreHttpServer};
use vecstore::store::VecStore;

#[derive(Parser, Debug)]
#[command(name = "vecstore-server")]
#[command(about = "VecStore multi-protocol server", long_about = None)]
struct Args {
    /// Path to the vector database
    #[arg(short, long, default_value = "vecstore.db")]
    db_path: String,

    /// gRPC server port
    #[arg(long, default_value = "50051")]
    grpc_port: u16,

    /// HTTP/REST server port
    #[arg(long, default_value = "8080")]
    http_port: u16,

    /// Vector dimension (required for new databases)
    #[arg(long)]
    dimension: Option<usize>,

    /// Enable debug logging
    #[arg(long)]
    debug: bool,

    /// Disable gRPC server
    #[arg(long)]
    no_grpc: bool,

    /// Disable HTTP server
    #[arg(long)]
    no_http: bool,

    /// Enable multi-tenant namespace mode (uses NamespaceManager)
    #[arg(long)]
    namespaces: bool,

    /// Namespace root directory (only with --namespaces)
    #[arg(long, default_value = "./namespaces")]
    namespace_root: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    let log_level = if args.debug { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                format!("vecstore={},vecstore_server={}", log_level, log_level).into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("🚀 Starting VecStore Server");

    // Choose mode: single-tenant or multi-tenant
    let namespace_manager = if args.namespaces {
        info!("🏢 Multi-tenant namespace mode enabled");
        info!("Namespace root: {}", args.namespace_root);

        let manager = NamespaceManager::new(&args.namespace_root)?;
        let loaded = manager.load_namespaces()?;
        info!("Loaded {} namespaces from disk", loaded.len());

        Some(Arc::new(RwLock::new(manager)))
    } else {
        None
    };

    // Single-tenant mode (backward compatible)
    let store = if namespace_manager.is_none() {
        info!("📦 Single-tenant mode");
        info!("Database: {}", args.db_path);

        let store = if std::path::Path::new(&args.db_path).exists() {
            info!("Loading existing database from {}", args.db_path);
            VecStore::open(&args.db_path)?
        } else {
            info!("Creating new database at {}", args.db_path);
            info!("Note: Dimension will be inferred from first vector inserted");
            VecStore::open(&args.db_path)?
        };

        info!(
            "Loaded database: {} vectors, dimension {}",
            store.len(),
            store.dimension()
        );

        Some(Arc::new(RwLock::new(store)))
    } else {
        None
    };

    // Start servers
    let mut handles = vec![];

    // Start gRPC server
    if !args.no_grpc {
        let grpc_addr: SocketAddr = format!("0.0.0.0:{}", args.grpc_port).parse()?;

        info!("🔌 Starting gRPC server on {}", grpc_addr);

        let grpc_handle = if let Some(ref manager) = namespace_manager {
            // Multi-tenant mode: Admin service only (use namespaces for vector ops)
            let admin_server = AdminService::new(manager.clone());

            info!("   Admin API: grpc://{}/ (VecStoreAdminService)", grpc_addr);

            tokio::spawn(async move {
                use vecstore::server::types::pb::vec_store_admin_service_server::VecStoreAdminServiceServer;

                TonicServer::builder()
                    .add_service(VecStoreAdminServiceServer::new(admin_server))
                    .serve(grpc_addr)
                    .await
                    .map_err(|e| anyhow::anyhow!("gRPC server error: {}", e))
            })
        } else {
            // Single-tenant mode: VecStore service only
            let grpc_server = VecStoreGrpcServer::with_store(store.clone().unwrap());

            tokio::spawn(async move {
                use vecstore::server::types::pb::vec_store_service_server::VecStoreServiceServer;

                TonicServer::builder()
                    .add_service(VecStoreServiceServer::new(grpc_server))
                    .serve(grpc_addr)
                    .await
                    .map_err(|e| anyhow::anyhow!("gRPC server error: {}", e))
            })
        };

        handles.push(grpc_handle);
    } else {
        warn!("gRPC server disabled");
    }

    // Start HTTP server
    if !args.no_http {
        let http_addr: SocketAddr = format!("0.0.0.0:{}", args.http_port).parse()?;

        info!("🌐 Starting HTTP server on {}", http_addr);

        let app = if let Some(ref manager) = namespace_manager {
            // Multi-tenant mode: Admin API
            let admin_server = AdminHttpServer::new(manager.clone());

            info!("   Admin API: http://{}/admin/namespaces", http_addr);
            info!("   Stats: http://{}/admin/stats", http_addr);
            info!("   Health: http://{}/health", http_addr);

            admin_server.router()
        } else {
            // Single-tenant mode: VecStore API
            let http_server = VecStoreHttpServer::with_store(store.clone().unwrap());

            info!("   REST API: http://{}/v1/query", http_addr);
            info!("   WebSocket: ws://{}/ws/query-stream", http_addr);
            info!("   Health: http://{}/health", http_addr);
            info!("   Metrics: http://{}/metrics", http_addr);

            http_server.router()
        };

        let http_handle = tokio::spawn(async move {
            let listener = tokio::net::TcpListener::bind(http_addr)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to bind HTTP server: {}", e))?;
            axum::serve(listener, app)
                .await
                .map_err(|e| anyhow::anyhow!("HTTP server error: {}", e))
        });

        handles.push(http_handle);
    } else {
        warn!("HTTP server disabled");
    }

    if handles.is_empty() {
        anyhow::bail!("No servers enabled (at least one of --no-grpc or --no-http must be false)");
    }

    info!("✅ VecStore Server running");
    info!("   Press Ctrl+C to stop");

    // Wait for all servers
    for handle in handles {
        handle.await??;
    }

    Ok(())
}
