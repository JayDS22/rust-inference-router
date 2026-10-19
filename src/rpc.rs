//! gRPC surface. Thin wrapper that delegates to Router.

use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::debug;

use crate::router::Router;

pub mod router_proto {
    tonic::include_proto!("router");
}

use router_proto::router_service_server::RouterService;
use router_proto::{HealthRequest, HealthResponse, RouteRequest, RouteResponse};

pub struct RouterRpc {
    router: Arc<Router>,
}

impl RouterRpc {
    pub fn new(router: Arc<Router>) -> Self {
        Self { router }
    }
}

#[tonic::async_trait]
impl RouterService for RouterRpc {
    async fn route(&self, req: Request<RouteRequest>) -> Result<Response<RouteResponse>, Status> {
        let r = req.into_inner();
        debug!(request_id = %r.request_id, n_tokens = r.prompt_tokens.len(), "route");
        let decision = self
            .router
            .route(&r.prompt_tokens)
            .map_err(|e| Status::unavailable(e.to_string()))?;
        Ok(Response::new(RouteResponse {
            worker_id: decision.worker_id,
            worker_addr: decision.worker_addr,
            prefix_cache_hit: decision.prefix_cache_hit,
        }))
    }

    async fn health_check(&self, _req: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse { ok: true }))
    }
}
