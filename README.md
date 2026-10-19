# rust-inference-router

Prefix-cache-aware request router for LLM inference workers. A focused Rust port of one subsystem of NVIDIA's [Dynamo](https://github.com/ai-dynamo/dynamo): the KV-cache-aware scheduler that routes incoming requests to the worker most likely to already hold the prompt prefix in its KV cache.

**Status:** v0.1 in-flight. Benchmarks reproducible via `make bench`.
**Author:** Jay Guwalani ([github.com/JayDS22](https://github.com/JayDS22))
**License:** Apache-2.0

---

## TL;DR

```bash
git clone https://github.com/JayDS22/rust-inference-router && cd rust-inference-router
docker compose up --build       # router + 3 mock workers on :50051
cargo run --release --bin loadgen -- --requests 10000 --prefix-skew zipf:1.1
# -> prints cache-hit-rate, P50, P99 for prefix-aware vs round-robin
```

One router process, N worker backends, one gRPC endpoint. Point your inference clients at it. It picks the worker with the longest matching prompt prefix instead of round-robin, which raises KV cache reuse and cuts prefill latency on repetitive workloads (chat, agents, RAG with shared system prompts).

---

## What this is / What this is NOT

**This IS:**
- A standalone L7 router that sits in front of N inference workers (vLLM, TGI, SGLang, llama.cpp, anything that speaks a request/response gRPC).
- A radix-tree index of token-prefixes already seen per worker, used to pick a routing target.
- A consistent-hashing fallback for cold prefixes.
- A measurable component: emits metrics for cache-hit-rate, routing-decision latency, and per-worker load.

**This is NOT:**
- A vLLM / TGI / SGLang replacement. It does zero inference. It calls workers.
- A full Dynamo port. Dynamo also owns the KV block manager, disaggregated prefill/decode orchestration, NIXL transfer, SLA planner, and more. This project ports exactly ONE subsystem.
- A production system. It's a 3-week solo build to demonstrate the routing core end-to-end. See [Known limits](#known-limits).

---

## Why prefix-aware routing

Transformer inference splits into two phases: prefill (compute the KV cache for the prompt) and decode (generate tokens). Prefill dominates TTFT and scales with prompt length. If two requests share a prefix (same system prompt, same few-shot block, same RAG context header), a worker that already holds the KV blocks for that prefix can skip most of prefill.

Round-robin and least-connections routers ignore this. They scatter identical prefixes across workers, each of which recomputes the shared prefix. Prefix-aware routing fixes that by indexing which prefixes live on which worker and routing new requests to the longest match.

Prior art:
- **SGLang RadixAttention** ([paper](https://arxiv.org/abs/2312.07104), Zheng et al. 2023): radix tree over token sequences for KV cache reuse inside a single worker.
- **NVIDIA Dynamo KV-aware router** ([docs](https://github.com/ai-dynamo/dynamo)): applies the same idea across workers in a disaggregated serving mesh.
- **vLLM PagedAttention** ([paper](https://arxiv.org/abs/2309.06180), Kwon et al. 2023): the paged KV allocator that makes prefix reuse cheap at the worker level.

This project is the cross-worker scheduling piece, written in Rust, scoped small enough to actually finish.

---

## Architecture

```
                     +---------------------------+
   client gRPC --->  |   rust-inference-router   |
                     |                           |
                     |  +---------------------+  |
                     |  | tonic gRPC server   |  |
                     |  +----------+----------+  |
                     |             |             |
                     |  +----------v----------+  |
                     |  | Router core:        |  |
                     |  |  1. Radix lookup    |  |
                     |  |  2. Consistent-hash |  |
                     |  |     ring fallback   |  |
                     |  |  3. Least-loaded    |  |
                     |  +----------+----------+  |
                     |             |             |
                     |  +----------v----------+  |
                     |  | Worker pool (tonic  |  |
                     |  | clients, health)    |  |
                     |  +----------+----------+  |
                     +-------------|-------------+
                                   |
                 +-----------------+-----------------+
                 |                 |                 |
          +------v------+   +------v------+   +------v------+
          |  worker-0   |   |  worker-1   |   |  worker-2   |
          | (vLLM/mock) |   | (vLLM/mock) |   | (vLLM/mock) |
          +-------------+   +-------------+   +-------------+
```

**Stack:** `tokio` 1.x, `tonic` + `prost` for gRPC, `dashmap` for concurrent registry, `parking_lot::RwLock` for the radix root + consistent-hash ring, `tracing` for structured logs, `anyhow` for errors. ~1200 lines of Rust excluding generated protobuf and tests.

### Routing algorithm

For each inbound request:

1. Walk the radix tree with the token sequence. Return the deepest `WorkerId` we passed through (longest matched prefix).
2. If the matched worker is healthy, route there (`prefix_cache_hit = true`).
3. Otherwise, consistent-hash the full token sequence against a ring with 64 virtual nodes per worker, pick the first healthy worker on the ring (`prefix_cache_hit = false`). Insert the prompt into the radix tree under the chosen worker so future repeat requests hit case 1.
4. If the ring is empty (no healthy workers reachable), fall back to least-pending-load.
5. If no healthy workers at all, return `Status::unavailable`.

Routing decision runs lock-free on the read path (DashMap shards + parking_lot RwLock). Updates on the write path hold a per-shard lock.

---

## API (excerpt)

```protobuf
service RouterService {
  rpc Route(RouteRequest) returns (RouteResponse);
  rpc HealthCheck(HealthRequest) returns (HealthResponse);
}

message RouteRequest {
  string request_id = 1;
  repeated uint32 prompt_tokens = 2;
}

message RouteResponse {
  string worker_id = 1;
  string worker_addr = 2;
  bool prefix_cache_hit = 3;
}
```

Full definition in `proto/router.proto`.

---

## Running it

### Local (requires Rust 1.82+, protoc)

```bash
cargo build --release
cargo run --release -- --bind 0.0.0.0:50051 --workers deploy/workers.json
```

### Docker Compose

```bash
docker compose up --build
```

Brings up: 1 router (`:50051`), 3 mock HTTP-echo backend containers. Point your inference client at `localhost:50051`.

### Config

Worker registry is JSON (`deploy/workers.json`):

```json
[
  {"worker_id": "a", "addr": "http://localhost:9001"},
  {"worker_id": "b", "addr": "http://localhost:9002"},
  {"worker_id": "c", "addr": "http://localhost:9003"}
]
```

---

## Benchmarks

### Microbench: routing decision latency

```bash
cargo bench --bench routing_bench
```

Measures `route()` call latency on a pre-populated 10K-prompt registry across 8 workers. Workload: shared 16-token head + divergent 48-token tail (modeling system-prompt + user-turn structure).

**Target: p99 < 500 microseconds at 10K indexed prefixes.**
Measured: TBD on v0.1 tag.

### Macro-sim: 10K-request stream, hit-rate + latency vs round-robin

`cargo run --release --bin loadgen -- --requests 10000 --prefix-skew zipf:1.1` (loadgen binary to be added in week 2).

Generates 10K prompts under Zipf(1.1) over 50 distinct system-prompt templates. Mock workers simulate prefill as `prefill_ms = 0.05 * (prompt_tokens - prefix_match_tokens) + base_latency` so measured gains reflect actual KV-reuse savings.

### Scorecard (v0.1, published on week-3 tag)

| Metric                                   | round-robin | consistent-hash | **prefix-aware** |
|------------------------------------------|-------------|-----------------|------------------|
| Cache hit rate (>=16 token match)        | TBD         | TBD             | TBD              |
| Mean prefix match tokens / request       | TBD         | TBD             | TBD              |
| P50 end-to-end latency (ms)              | TBD         | TBD             | TBD              |
| P99 end-to-end latency (ms)              | TBD         | TBD             | TBD              |
| Routing decision P99 (microseconds)      | n/a         | TBD             | TBD              |
| Throughput (req/s, 3 mock workers)       | TBD         | TBD             | TBD              |
| Router process RSS (MB)                  | TBD         | TBD             | TBD              |

**Target bar for v0.1 to be called shipped:**
- Cache hit rate under Zipf(1.1): >= 60% (round-robin baseline: ~15%).
- P99 end-to-end latency reduction: >= 30% vs round-robin.
- Routing decision P99: < 500 microseconds at 10K indexed prefixes.
- Router RSS: < 200 MB at 10K indexed prefixes.

Reproduce: `make bench` on a 4-core box, no GPU required.

---

## Known limits

What's deliberately NOT in v0.1 and why:

- **No real KV block tracking.** The radix tree tracks token prefixes that each worker has *seen*, not which blocks that worker still *holds*. Dynamo's full router queries workers for block residency. v0.1 approximates with LRU eviction. Upgrade path: add a `BlockReportStream` gRPC and have workers push eviction events.
- **No disaggregated prefill/decode.** Dynamo routes prefill and decode to separate pools. v0.1 treats each worker as a monolithic unit. Scope cut for a 3-week build.
- **No SLA planner, no autoscaling.** Static worker list from config, with health checks.
- **No batching across clients.** Requests forwarded 1:1. Continuous batching happens inside each worker.
- **No tokenizer in router.** Clients send pre-tokenized `prompt_tokens` on the wire. The router does not run a BPE pass; moving tokenization into the router is a reasonable v0.2 feature.
- **No auth.** gRPC endpoint is unauthenticated. Add mTLS or an API gateway in front for anything real.
- **Mock workers only in CI.** End-to-end tests against a real vLLM worker are a `scripts/e2e-vllm.sh` manual script, not CI.

---

## Project layout

```
rust-inference-router/
├── Cargo.toml
├── build.rs
├── proto/
│   └── router.proto
├── src/
│   ├── main.rs          # binary entry
│   ├── lib.rs
│   ├── router.rs        # core routing (radix + ring + least-loaded)
│   ├── radix.rs         # prefix radix tree
│   ├── health.rs        # background health monitor
│   └── rpc.rs           # tonic service impl
├── benches/
│   └── routing_bench.rs # Criterion microbench
├── tests/
│   └── integration_test.rs
├── ops/
├── deploy/
│   └── workers.json
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
└── docker-compose.yml
```

---

## Development

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
cargo bench --bench routing_bench
```

CI runs on every push to `main` and every PR via `.github/workflows/ci.yml`: fmt, clippy, test, bench (criterion report uploaded as artifact).

---

## References

- NVIDIA Dynamo: upstream inspiration. [github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo).
- SGLang RadixAttention: Zheng et al., "Efficiently Programming Large Language Models using SGLang" (2023). [arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104).
- vLLM PagedAttention: Kwon et al., SOSP 2023. [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180).
- Karger et al., "Consistent Hashing and Random Trees" (STOC 1997). The fallback ring is textbook consistent hashing with 64 virtual nodes per worker.

---

## FAQ

**Why Rust instead of Go or C++?**
Tokio + tonic give async gRPC with no runtime GC pauses. The radix tree benefits from `&[u32]` zero-copy slicing. Also, this project exists in part to close a Rust gap for data-infra roles. Honest answer.

**How does this compare to running Dynamo directly?**
It doesn't. Dynamo is a much larger system and solves more problems. This project ports one subsystem to understand it end-to-end and to have a Rust artifact that's defensible on its own terms.

**Where are the real numbers in the scorecard?**
Filled on the v0.1 tag after the week-3 benchmark run. The scorecard structure is published now so the methodology is reviewable before the numbers land.
