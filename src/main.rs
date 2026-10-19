use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::EnvFilter;

use rust_inference_router::health::HealthMonitor;
use rust_inference_router::router::Router;
use rust_inference_router::rpc::router_proto::router_service_server::RouterServiceServer;
use rust_inference_router::rpc::RouterRpc;

#[derive(Parser, Debug)]
#[command(name = "rust-inference-router", version)]
struct Cli {
    /// gRPC bind address, e.g. 0.0.0.0:50051
    #[arg(long, default_value = "0.0.0.0:50051")]
    bind: SocketAddr,

    /// JSON worker registry file: [{"worker_id":"w1","addr":"http://localhost:9001"}, ...]
    #[arg(long)]
    workers: Option<PathBuf>,

    /// Health check interval in seconds
    #[arg(long, default_value_t = 5)]
    health_interval: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let cli = Cli::parse();
    let router = Arc::new(Router::new());

    if let Some(path) = &cli.workers {
        let bytes = tokio::fs::read(path).await.context("read workers file")?;
        router.load_registry_json(&bytes).context("parse workers file")?;
        info!(count = router.worker_count(), "loaded worker registry");
    }

    let monitor = HealthMonitor::new(Arc::clone(&router), cli.health_interval);
    monitor.spawn();

    let svc = RouterServiceServer::new(RouterRpc::new(Arc::clone(&router)));
    info!(bind = %cli.bind, "router listening");
    Server::builder().add_service(svc).serve(cli.bind).await.context("grpc serve")?;
    Ok(())
}
