# Multi-stage build for VecStore Server
# This creates a minimal production image with the server binary

# ============================================================================
# Builder Stage - Compile the Rust application
# ============================================================================
FROM rustlang/rust:nightly-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./

# Copy source code
COPY src ./src
COPY proto ./proto
COPY benches ./benches

# Build for release with server feature
RUN cargo build --release --features server --bin vecstore-server

# ============================================================================
# Runtime Stage - Minimal image with just the binary
# ============================================================================
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 vecstore && \
    mkdir -p /data && \
    chown -R vecstore:vecstore /data

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/vecstore-server /app/vecstore-server

# Set ownership
RUN chown -R vecstore:vecstore /app

# Switch to non-root user
USER vecstore

# Expose ports
EXPOSE 50051 8080

# Default environment variables
ENV RUST_LOG=info
ENV DB_PATH=/data/vectors.db

# Volume for persistent data
VOLUME ["/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the server
ENTRYPOINT ["/app/vecstore-server"]
CMD ["--db-path", "/data/vectors.db", "--grpc-port", "50051", "--http-port", "8080"]
