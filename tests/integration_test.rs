use std::sync::Arc;

use rust_inference_router::router::Router;

fn mk_router(n: usize) -> Arc<Router> {
    let r = Arc::new(Router::new());
    for i in 0..n {
        r.register_worker(format!("w{i}"), format!("http://127.0.0.1:{}", 9000 + i))
            .unwrap();
    }
    r
}

#[test]
fn prefix_locality_holds_across_repeats() {
    let router = mk_router(3);
    let shared_prefix: Vec<u32> = (0..32).collect();

    let mut requests: Vec<Vec<u32>> = Vec::new();
    for i in 0..100u32 {
        let mut p = shared_prefix.clone();
        p.extend((0..16).map(|j| i * 1000 + j));
        requests.push(p);
    }

    let first: Vec<String> = requests
        .iter()
        .map(|p| router.route(p).unwrap().worker_id)
        .collect();

    for (i, p) in requests.iter().enumerate() {
        let d = router.route(p).unwrap();
        assert!(d.prefix_cache_hit, "req {i} missed cache on repeat");
        assert_eq!(d.worker_id, first[i], "req {i} drifted worker on repeat");
    }
}

#[test]
fn shared_prefix_concentrates() {
    let router = mk_router(3);
    let shared: Vec<u32> = (0..64).collect();

    let seed_wid = router.route(&shared).unwrap().worker_id;

    for i in 0..50u32 {
        let mut p = shared.clone();
        p.extend([i, i + 1, i + 2]);
        let d = router.route(&p).unwrap();
        assert_eq!(d.worker_id, seed_wid, "divergent routing at i={i}");
        assert!(d.prefix_cache_hit);
    }
}

#[test]
fn no_healthy_workers_errors() {
    let router = Router::new();
    assert!(router.route(&[1, 2, 3]).is_err());
}

#[test]
fn unhealthy_worker_falls_back() {
    let router = mk_router(2);
    let p: Vec<u32> = (0..32).collect();
    let first = router.route(&p).unwrap().worker_id;

    for _ in 0..3 {
        router.mark_health(&first, false);
    }

    let d = router.route(&p).unwrap();
    assert_ne!(d.worker_id, first);
}

#[test]
fn registry_json_roundtrip() {
    let router = Router::new();
    let json = br#"[{"worker_id":"a","addr":"http://1"},{"worker_id":"b","addr":"http://2"}]"#;
    router.load_registry_json(json).unwrap();
    assert_eq!(router.worker_count(), 2);
}
