//! Core routing logic. Three signals combined, in order:
//!   1. Longest prefix match in the radix tree (cache-locality win).
//!   2. Consistent-hash ring over healthy workers (stable fallback).
//!   3. Least pending load (break ties, avoid hot workers).
//!
//! All state is behind DashMap / parking_lot so route() stays lock-free on the
//! hot path. The radix tree uses a single RwLock internally since reads
//! dominate; see radix.rs.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::radix::RadixTree;

pub type WorkerId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: WorkerId,
    pub addr: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerHealth {
    Healthy,
    Unhealthy,
}

#[derive(Debug)]
pub struct WorkerState {
    pub addr: String,
    pub health: WorkerHealth,
    pub consecutive_failures: u32,
    pub pending: u64,
}

#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub worker_id: WorkerId,
    pub worker_addr: String,
    pub prefix_cache_hit: bool,
}

const VNODES_PER_WORKER: usize = 64;
const FAILURE_THRESHOLD: u32 = 3;

pub struct Router {
    workers: DashMap<WorkerId, WorkerState>,
    radix: Arc<RadixTree>,
    /// Consistent-hash ring: hash -> worker_id. BTreeMap for cheap successor lookup.
    /// ponytail: single RwLock, acceptable for < 10k workers. Shard if that changes.
    ring: RwLock<BTreeMap<u64, WorkerId>>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            workers: DashMap::new(),
            radix: Arc::new(RadixTree::new(100_000)),
            ring: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn radix(&self) -> Arc<RadixTree> {
        Arc::clone(&self.radix)
    }

    pub fn register_worker(&self, worker_id: WorkerId, addr: String) -> Result<()> {
        if worker_id.is_empty() {
            return Err(anyhow!("worker_id must be non-empty"));
        }
        self.workers.insert(
            worker_id.clone(),
            WorkerState {
                addr,
                health: WorkerHealth::Healthy,
                consecutive_failures: 0,
                pending: 0,
            },
        );
        let mut ring = self.ring.write();
        for v in 0..VNODES_PER_WORKER {
            let h = Self::hash_vnode(&worker_id, v);
            ring.insert(h, worker_id.clone());
        }
        debug!(worker_id, "registered worker");
        Ok(())
    }

    pub fn deregister_worker(&self, worker_id: &str) {
        self.workers.remove(worker_id);
        let mut ring = self.ring.write();
        ring.retain(|_, v| v != worker_id);
    }

    pub fn mark_health(&self, worker_id: &str, success: bool) {
        if let Some(mut w) = self.workers.get_mut(worker_id) {
            if success {
                w.consecutive_failures = 0;
                w.health = WorkerHealth::Healthy;
            } else {
                w.consecutive_failures += 1;
                if w.consecutive_failures >= FAILURE_THRESHOLD {
                    if w.health == WorkerHealth::Healthy {
                        warn!(worker_id, failures = w.consecutive_failures, "worker UNHEALTHY");
                    }
                    w.health = WorkerHealth::Unhealthy;
                }
            }
        }
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn healthy_workers(&self) -> Vec<WorkerId> {
        self.workers
            .iter()
            .filter(|e| e.value().health == WorkerHealth::Healthy)
            .map(|e| e.key().clone())
            .collect()
    }

    pub fn load_registry_json(&self, bytes: &[u8]) -> Result<()> {
        let specs: Vec<WorkerSpec> = serde_json::from_slice(bytes)?;
        for s in specs {
            self.register_worker(s.worker_id, s.addr)?;
        }
        Ok(())
    }

    /// Hot path. Three stages, each short-circuits on a hit.
    pub fn route(&self, prompt_tokens: &[u32]) -> Result<RouteDecision> {
        // Stage 1: prefix-locality.
        if let Some(wid) = self.radix.longest_prefix_match(prompt_tokens) {
            if self.is_healthy(&wid) {
                self.bump_pending(&wid);
                let addr = self.addr_of(&wid).unwrap_or_default();
                return Ok(RouteDecision { worker_id: wid, worker_addr: addr, prefix_cache_hit: true });
            }
            debug!(worker_id = %wid, "prefix match worker unhealthy, falling back");
        }

        // Stage 2: consistent-hash ring on the full token sequence.
        if let Some(wid) = self.ring_pick(prompt_tokens) {
            self.bump_pending(&wid);
            self.radix.insert(prompt_tokens, wid.clone());
            let addr = self.addr_of(&wid).unwrap_or_default();
            return Ok(RouteDecision { worker_id: wid, worker_addr: addr, prefix_cache_hit: false });
        }

        // Stage 3: least pending load across any healthy worker.
        if let Some(wid) = self.least_loaded() {
            self.bump_pending(&wid);
            self.radix.insert(prompt_tokens, wid.clone());
            let addr = self.addr_of(&wid).unwrap_or_default();
            return Ok(RouteDecision { worker_id: wid, worker_addr: addr, prefix_cache_hit: false });
        }

        Err(anyhow!("no healthy workers available"))
    }

    pub fn complete(&self, worker_id: &str) {
        if let Some(mut w) = self.workers.get_mut(worker_id) {
            w.pending = w.pending.saturating_sub(1);
        }
    }

    pub fn addr_of_public(&self, worker_id: &str) -> Option<String> {
        self.addr_of(worker_id)
    }

    fn is_healthy(&self, worker_id: &str) -> bool {
        self.workers
            .get(worker_id)
            .map(|w| w.health == WorkerHealth::Healthy)
            .unwrap_or(false)
    }

    fn addr_of(&self, worker_id: &str) -> Option<String> {
        self.workers.get(worker_id).map(|w| w.addr.clone())
    }

    fn bump_pending(&self, worker_id: &str) {
        if let Some(mut w) = self.workers.get_mut(worker_id) {
            w.pending += 1;
        }
    }

    fn least_loaded(&self) -> Option<WorkerId> {
        self.workers
            .iter()
            .filter(|e| e.value().health == WorkerHealth::Healthy)
            .min_by_key(|e| e.value().pending)
            .map(|e| e.key().clone())
    }

    fn ring_pick(&self, tokens: &[u32]) -> Option<WorkerId> {
        let ring = self.ring.read();
        if ring.is_empty() {
            return None;
        }
        let h = Self::hash_tokens(tokens);
        ring.range(h..).chain(ring.range(..h)).find_map(|(_, wid)| {
            if self.is_healthy(wid) { Some(wid.clone()) } else { None }
        })
    }

    fn hash_vnode(worker_id: &str, vnode: usize) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        worker_id.hash(&mut h);
        vnode.hash(&mut h);
        h.finish()
    }

    fn hash_tokens(tokens: &[u32]) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for t in tokens {
            t.hash(&mut h);
        }
        h.finish()
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}
