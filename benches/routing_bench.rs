use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rust_inference_router::router::Router;

fn seed_router(n_prompts: usize, n_workers: usize) -> (Router, Vec<Vec<u32>>) {
    let router = Router::new();
    for i in 0..n_workers {
        router
            .register_worker(format!("w{i}"), format!("http://127.0.0.1:{}", 9000 + i))
            .unwrap();
    }
    let mut prompts = Vec::with_capacity(n_prompts);
    for i in 0..n_prompts {
        let mut p = Vec::with_capacity(64);
        for j in 0..16u32 {
            p.push(j);
        }
        for j in 0..48u32 {
            p.push(((i as u32) * 1_000_003 + j) % 50_000);
        }
        router.route(&p).unwrap();
        prompts.push(p);
    }
    (router, prompts)
}

fn bench_route(c: &mut Criterion) {
    let (router, prompts) = seed_router(10_000, 8);
    let mut idx = 0usize;
    c.bench_function("route_10k_prompts_8_workers", |b| {
        b.iter_batched(
            || {
                idx = (idx + 1) % prompts.len();
                &prompts[idx]
            },
            |p| router.route(p).unwrap(),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_route);
criterion_main!(benches);
