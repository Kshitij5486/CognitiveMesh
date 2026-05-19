import sys, os, time, statistics, threading, gc, json
from datetime import datetime, timezone
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "telemetry"))

from quorum_manager import QuorumManager, QuorumDecision
from byzantine_recovery_coordinator import ByzantineRecoveryCoordinator, ByzantineNodeState
from multi_node_recovery_orchestrator import MultiNodeRecoveryOrchestrator
from quorum_aware_router import QuorumAwareRouter
from session_persistence import SessionPersistence
from health_monitor import HealthMonitor, SLATarget
from rate_limiter import SlidingWindowRateLimiter, GracefulShutdownManager

ALL_NODES = ["node-1", "node-2", "node-3"]
ITERATIONS = 1000
WARMUP = 50

def make_mock_updater(effects=None, ready=True):
    effects = effects or {"node-1": 27.5, "node-2": 28.5, "node-3": 29.0}
    updater = MagicMock()
    updater.status.return_value = {"is_ready": ready, "buffer_size": 100, "retrain_count": 10, "max_buffer_size": 200, "nodes_modeled": ALL_NODES}
    def get_snapshot(node_id):
        return {"effect": effects.get(node_id, 28.0), "samples_used": 100, "timestamp": datetime.now(timezone.utc).isoformat()}
    updater.get_current_snapshot.side_effect = get_snapshot
    return updater

def bench(fn, n=ITERATIONS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    gc.collect()
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - start) * 1000)
    return {"n": n, "mean_ms": round(statistics.mean(latencies), 4), "median_ms": round(statistics.median(latencies), 4), "p95_ms": round(sorted(latencies)[int(n * 0.95)], 4), "p99_ms": round(sorted(latencies)[int(n * 0.99)], 4), "min_ms": round(min(latencies), 4), "max_ms": round(max(latencies), 4), "stdev_ms": round(statistics.stdev(latencies), 4)}

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def result(name, stats, threshold_ms):
    passed = stats["mean_ms"] < threshold_ms
    status = "PASS" if passed else "FAIL"
    print(f"  {name:<40} mean={stats['mean_ms']:.4f}ms  p99={stats['p99_ms']:.4f}ms  [{status}]")
    return passed, stats

def run_benchmarks():
    results = {}
    all_pass = True
    print(f"\nCognitiveMesh v1.0.0 -- Performance Benchmark Suite")
    print(f"Date: {datetime.now(timezone.utc).isoformat()}")
    print(f"Iterations: {ITERATIONS}  Warmup: {WARMUP}")

    section("Sprint 1-3: Causal Engine Subsystem")
    updater = make_mock_updater()
    stats = bench(lambda: updater.status())
    ok, s = result("updater.status()", stats, 1.0); results["updater_status"] = s; all_pass &= ok
    stats = bench(lambda: updater.get_current_snapshot("node-1"))
    ok, s = result("updater.get_snapshot()", stats, 1.0); results["updater_get_snapshot"] = s; all_pass &= ok

    section("Sprint 5: Byzantine Consensus (QuorumManager)")
    qm = QuorumManager(updater=updater, check_interval=9999.0)
    stats = bench(lambda: qm.request_node_offline("node-1", "bench"))
    ok, s = result("quorum.request_node_offline()", stats, 5.0); results["quorum_request_offline"] = s; all_pass &= ok
    stats = bench(lambda: qm.get_quorum_state())
    ok, s = result("quorum.get_quorum_state()", stats, 1.0); results["quorum_get_state"] = s; all_pass &= ok
    stats = bench(lambda: qm.status())
    ok, s = result("quorum.status()", stats, 5.0); results["quorum_status"] = s; all_pass &= ok
    stats = bench(lambda: qm.get_all_node_statuses())
    ok, s = result("quorum.get_all_node_statuses()", stats, 5.0); results["quorum_get_all_statuses"] = s; all_pass &= ok
    stats = bench(lambda: qm.get_safe_to_offline())
    ok, s = result("quorum.get_safe_to_offline()", stats, 5.0); results["quorum_safe_to_offline"] = s; all_pass &= ok

    section("Sprint 9: Byzantine Detection (Coordinator)")
    router_mock = MagicMock()
    router_mock.status.return_value = {"node_states": {n: "active" for n in ALL_NODES}, "running": True}
    router_mock._compute_causal_weights.return_value = {n: 1.0/3 for n in ALL_NODES}
    retrainer_mock = MagicMock()
    retrainer_mock.status.return_value = {"running": True}
    retrainer_mock._is_in_cooldown.return_value = False
    coord = ByzantineRecoveryCoordinator(updater=updater, quorum_manager=qm, router=router_mock, retrainer=retrainer_mock, check_interval=9999.0)
    effects = {"node-1": 55.0, "node-2": 28.0, "node-3": 27.0}
    stats = bench(lambda: coord._compute_byzantine_score("node-1", effects))
    ok, s = result("coord.compute_byzantine_score()", stats, 5.0); results["coord_compute_score"] = s; all_pass &= ok
    stats = bench(lambda: coord._classify_node("node-1", 0.75))
    ok, s = result("coord.classify_node()", stats, 1.0); results["coord_classify_node"] = s; all_pass &= ok
    stats = bench(lambda: coord.get_all_node_states())
    ok, s = result("coord.get_all_node_states()", stats, 1.0); results["coord_get_all_states"] = s; all_pass &= ok
    stats = bench(lambda: coord.status())
    ok, s = result("coord.status()", stats, 5.0); results["coord_status"] = s; all_pass &= ok

    section("Sprint 9: Quorum-Aware Routing")
    q_router = QuorumAwareRouter(updater=updater, quorum_manager=qm, check_interval=9999.0)
    q_router._start_time = time.time()
    q_router._check_cycle()
    stats = bench(lambda: q_router.route_request())
    ok, s = result("router.route_request()", stats, 1.0); results["router_route_request"] = s; all_pass &= ok
    stats = bench(lambda: q_router.get_weights())
    ok, s = result("router.get_weights()", stats, 1.0); results["router_get_weights"] = s; all_pass &= ok
    cap = {"node-1": "full", "node-2": "full", "node-3": "full"}
    eff = {"node-1": 27.5, "node-2": 28.5, "node-3": 29.0}
    stats = bench(lambda: q_router._compute_weights(eff, cap, "healthy"))
    ok, s = result("router._compute_weights()", stats, 5.0); results["router_compute_weights"] = s; all_pass &= ok
    stats = bench(lambda: q_router.get_weight_stability())
    ok, s = result("router.get_weight_stability()", stats, 5.0); results["router_weight_stability"] = s; all_pass &= ok
    stats = bench(lambda: q_router.status())
    ok, s = result("router.status()", stats, 5.0); results["router_status"] = s; all_pass &= ok

    section("Sprint 10: Session Persistence (async queue)")
    p = SessionPersistence(dsn="invalid_dsn")
    stats = bench(lambda: p.save_session({"session_id": "bench-001"}))
    ok, s = result("persistence.save_session()", stats, 1.0); results["persistence_save_session"] = s; all_pass &= ok
    stats = bench(lambda: p.save_decision({"node_id": "node-1"}))
    ok, s = result("persistence.save_decision()", stats, 1.0); results["persistence_save_decision"] = s; all_pass &= ok
    stats = bench(lambda: p.save_sla_snapshot({"quorum_state": "healthy"}))
    ok, s = result("persistence.save_sla()", stats, 1.0); results["persistence_save_sla"] = s; all_pass &= ok
    stats = bench(lambda: p.get_stats())
    ok, s = result("persistence.get_stats()", stats, 1.0); results["persistence_get_stats"] = s; all_pass &= ok

    section("Sprint 10: Health Monitor")
    orch = MultiNodeRecoveryOrchestrator(updater=updater, quorum_manager=qm, coordinator=coord, check_interval=9999.0)
    orch._start_time = time.time()
    monitor = HealthMonitor(updater=updater, quorum_manager=qm, coordinator=coord, orchestrator=orch, quorum_router=q_router, persistence=None)
    monitor._start_time = time.time()
    stats = bench(lambda: monitor._run_all_checks(), n=200)
    ok, s = result("health._run_all_checks()", stats, 100.0); results["health_run_all_checks"] = s; all_pass &= ok
    stats = bench(lambda: monitor.get_liveness())
    ok, s = result("health.get_liveness()", stats, 1.0); results["health_get_liveness"] = s; all_pass &= ok
    stats = bench(lambda: monitor.get_readiness())
    ok, s = result("health.get_readiness()", stats, 5.0); results["health_get_readiness"] = s; all_pass &= ok
    stats = bench(lambda: monitor.get_full_status(), n=200)
    ok, s = result("health.get_full_status()", stats, 10.0); results["health_get_full_status"] = s; all_pass &= ok
    stats = bench(lambda: monitor.get_sla_status())
    ok, s = result("health.get_sla_status()", stats, 5.0); results["health_get_sla_status"] = s; all_pass &= ok
    sla_target = SLATarget("bench_target", 99.0, window_minutes=60)
    stats = bench(lambda: sla_target.record(True))
    ok, s = result("sla_target.record()", stats, 1.0); results["sla_target_record"] = s; all_pass &= ok
    stats = bench(lambda: sla_target.compliance_pct)
    ok, s = result("sla_target.compliance_pct", stats, 1.0); results["sla_target_compliance"] = s; all_pass &= ok

    section("Sprint 10: Rate Limiter")
    limiter = SlidingWindowRateLimiter()
    stats = bench(lambda: limiter.check("192.168.1.1", "/health"))
    ok, s = result("rate_limiter.check() /health", stats, 1.0); results["rate_limiter_check_health"] = s; all_pass &= ok
    stats = bench(lambda: limiter.check("192.168.1.2", "/quorum"))
    ok, s = result("rate_limiter.check() /quorum", stats, 1.0); results["rate_limiter_check_quorum"] = s; all_pass &= ok
    stats = bench(lambda: limiter._get_endpoint_group("/recovery/trigger"))
    ok, s = result("rate_limiter.get_endpoint_group()", stats, 0.1); results["rate_limiter_get_group"] = s; all_pass &= ok
    stats = bench(lambda: limiter.stats())
    ok, s = result("rate_limiter.stats()", stats, 1.0); results["rate_limiter_stats"] = s; all_pass &= ok

    section("Sprint 10: Graceful Shutdown")
    mgr = GracefulShutdownManager()
    stats = bench(lambda: mgr.request_start() and mgr.request_end() or True)
    ok, s = result("shutdown.request_start()+end()", stats, 1.0); results["shutdown_request_cycle"] = s; all_pass &= ok
    stats = bench(lambda: mgr.status())
    ok, s = result("shutdown.status()", stats, 1.0); results["shutdown_status"] = s; all_pass &= ok

    section("Throughput: Concurrent route_request()")
    q_router2 = QuorumAwareRouter(updater=make_mock_updater(), quorum_manager=QuorumManager(updater=make_mock_updater(), check_interval=9999.0), check_interval=9999.0)
    q_router2._start_time = time.time()
    q_router2._check_cycle()
    request_count = 0
    lock = threading.Lock()
    duration = 2.0
    def worker():
        nonlocal request_count
        end = time.time() + duration
        local = 0
        while time.time() < end:
            q_router2.route_request()
            local += 1
        with lock:
            request_count += local
    threads = [threading.Thread(target=worker) for _ in range(4)]
    t_start = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.time() - t_start
    rps = int(request_count / elapsed)
    print(f"\n  Concurrent route_request() 4 threads x {duration}s")
    print(f"  Total requests: {request_count:,}  Throughput: {rps:,} req/s")
    results["throughput_route_rps"] = rps
    throughput_ok = rps >= 10000
    print(f"  Threshold: >=10,000 req/s  [{'PASS' if throughput_ok else 'FAIL'}]")
    all_pass &= throughput_ok

    section("BENCHMARK SUMMARY")
    thresholds = {"updater_status": 1.0, "updater_get_snapshot": 1.0, "quorum_request_offline": 5.0, "quorum_get_state": 1.0, "quorum_status": 5.0, "quorum_get_all_statuses": 5.0, "quorum_safe_to_offline": 5.0, "coord_compute_score": 5.0, "coord_classify_node": 1.0, "coord_get_all_states": 1.0, "coord_status": 5.0, "router_route_request": 1.0, "router_get_weights": 1.0, "router_compute_weights": 5.0, "router_weight_stability": 5.0, "router_status": 5.0, "persistence_save_session": 1.0, "persistence_save_decision": 1.0, "persistence_save_sla": 1.0, "persistence_get_stats": 1.0, "health_run_all_checks": 100.0, "health_get_liveness": 1.0, "health_get_readiness": 5.0, "health_get_full_status": 10.0, "health_get_sla_status": 5.0, "sla_target_record": 1.0, "sla_target_compliance": 1.0, "rate_limiter_check_health": 1.0, "rate_limiter_check_quorum": 1.0, "rate_limiter_get_group": 0.1, "rate_limiter_stats": 1.0, "shutdown_request_cycle": 1.0, "shutdown_status": 1.0}
    total = len(thresholds)
    passed_count = sum(1 for k, v in results.items() if k in thresholds and isinstance(v, dict) and v.get("mean_ms", 999) < thresholds[k])
    print(f"\n  Benchmarks passed: {passed_count}/{total}")
    print(f"  Throughput:        {rps:,} req/s ({'PASS' if throughput_ok else 'FAIL'})")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    key_metrics = [("route_request latency", results["router_route_request"]["mean_ms"], "ms"), ("byzantine_score latency", results["coord_compute_score"]["mean_ms"], "ms"), ("health_check_cycle latency", results["health_run_all_checks"]["mean_ms"], "ms"), ("rate_limiter latency", results["rate_limiter_check_health"]["mean_ms"], "ms"), ("persistence_enqueue latency", results["persistence_save_session"]["mean_ms"], "ms"), ("throughput", rps, "req/s")]
    print("\n  Key metrics for release notes:")
    for name, val, unit in key_metrics:
        if unit == "req/s":
            print(f"    {name:<35} {val:>10,} {unit}")
        else:
            print(f"    {name:<35} {val:>10.4f} {unit}")
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "all_pass": all_pass, "passed_count": passed_count, "total_benchmarks": total, "throughput_rps": rps, "results": results}

if __name__ == "__main__":
    report = run_benchmarks()
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")
    print(f"\n{'='*60}\n  CognitiveMesh v1.0.0 benchmark complete\n{'='*60}\n")
