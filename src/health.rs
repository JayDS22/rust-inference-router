//! Background health monitor. Polls each registered worker via gRPC HealthCheck
//! every N seconds. Three consecutive failures flip worker to UNHEALTHY; one
//! success clears the counter.

use std::sync::Arc;
use std::time::Duration;

use tokio::time::{interval, MissedTickBehavior};
use tonic::transport::Channel;
use tracing::{debug, info, warn};

use crate::router::Router;
use crate::rpc::router_proto::router_service_client::RouterServiceClient;
use crate::rpc::router_proto::HealthRequest;

pub struct HealthMonitor {
    router: Arc<Router>,
    interval_secs: u64,
}

impl HealthMonitor {
    pub fn new(router: Arc<Router>, interval_secs: u64) -> Self {
        Self { router, interval_secs: interval_secs.max(1) }
    }

    pub fn spawn(self) {
        tokio::spawn(async move { self.run().await });
    }

    async fn run(self) {
        let mut tick = interval(Duration::from_secs(self.interval_secs));
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        info!(interval = self.interval_secs, "health monitor started");
        loop {
            tick.tick().await;
            let targets: Vec<(String, String)> = self
                .router
                .healthy_workers()
                .into_iter()
                .filter_map(|wid| self.router.addr_of_public(&wid).map(|a| (wid, a)))
                .collect();
            for (wid, addr) in targets {
                let ok = probe(&addr).await;
                self.router.mark_health(&wid, ok);
                debug!(worker_id = %wid, ok, "health probe");
                if !ok {
                    warn!(worker_id = %wid, addr, "probe failed");
                }
            }
        }
    }
}

async fn probe(addr: &str) -> bool {
    let endpoint = match Channel::from_shared(addr.to_string()) {
        Ok(e) => e.connect_timeout(Duration::from_secs(2)).timeout(Duration::from_secs(2)),
        Err(_) => return false,
    };
    let channel = match endpoint.connect().await {
        Ok(c) => c,
        Err(_) => return false,
    };
    let mut client = RouterServiceClient::new(channel);
    client.health_check(HealthRequest {}).await.is_ok()
}
