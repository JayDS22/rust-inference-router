# syntax=docker/dockerfile:1.7

FROM rust:1.82-slim-bookworm AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock* ./
COPY build.rs ./
COPY proto ./proto
RUN mkdir -p src && echo "fn main() {}" > src/main.rs && echo "" > src/lib.rs \
    && cargo build --release || true

COPY src ./src
COPY benches ./benches
COPY tests ./tests
RUN cargo build --release --bin rust-inference-router \
    && strip target/release/rust-inference-router

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 10001 -s /sbin/nologin router
COPY --from=builder /build/target/release/rust-inference-router /usr/local/bin/rust-inference-router
USER router
EXPOSE 50051
ENV RUST_LOG=info
ENTRYPOINT ["/usr/local/bin/rust-inference-router"]
CMD ["--bind", "0.0.0.0:50051"]
