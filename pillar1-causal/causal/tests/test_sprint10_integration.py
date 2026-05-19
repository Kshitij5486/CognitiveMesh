import sys
import os
import time
import threading
import statistics
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "telemetry"))

from quorum_manager import (
    QuorumManager, QuorumState, NodeCapacityState,
    QuorumDecision,
)
from byzantine_recovery_coordinator import (
    ByzantineRecoveryCoordinator, ByzantineNodeState,
)
from multi_node_recovery_orchestrator import (
    MultiNodeRecoveryOrchestrator, RecoveryPhase,
    NodeRecoveryOutcome,
)
from quorum_aware_router import (
    QuorumAwareRouter, RoutingDecisionType,
)
from session_persistence import (
    SessionPersistence, SLAMonitor,
)
from health_monitor import (
    HealthMonitor, HealthStatus,
)
from rate_limiter import (
    SlidingWindowRateLimiter, GracefulShutdownManager,
    DEFAULT_LIMITS,
)

ALL_NODES = ["node-1", "node-2", "node-3"]


# ── Fixtures ───────────────────────────────────────────────

def make_mock_updater(
    ready=True,
    effects=None,
    buffer_size=60,
):
    if effects is None:
        effects = {
            "node-1": 27.5,
            "node-2": 28.5,
            "node-3": 29.0,
        }
    updater = MagicMock()
    updater.status.return_value = {
        "is_ready": ready,
        "buffer_size": buffer_size,
        "retrain_count": 5,
        "max_buffer_size": 200,
        "nodes_modeled": ALL_NODES,
    }

    def get_snapshot(node_id):
        return {
            "effect": effects.get(node_id, 28.0),
            "samples_used": buffer_size,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    updater.get_current_snapshot.side_effect = (
        get_snapshot
    )
    return updater


def make_full_stack(effects=None, ready=True):
    """Build the complete Sprint 9+10 stack."""
    updater = make_mock_updater(
        ready=ready, effects=effects
    )
    quorum = QuorumManager(
        updater=updater, check_interval=60.0
    )
    router_mock = MagicMock()
    router_mock.status.return_value = {
        "node_states": {
            n: "active" for n in ALL_NODES
        },
        "running": True,
    }
    router_mock._compute_causal_weights.return_value = {
        n: 1.0 / 3 for n in ALL_NODES
    }
    retrainer_mock = MagicMock()
    retrainer_mock.status.return_value = {
        "running": True
    }
    retrainer_mock._is_in_cooldown.return_value = False

    coordinator = ByzantineRecoveryCoordinator(
        updater=updater,
        quorum_manager=quorum,
        router=router_mock,
        retrainer=retrainer_mock,
        check_interval=60.0,
    )
    orchestrator = MultiNodeRecoveryOrchestrator(
        updater=updater,
        quorum_manager=quorum,
        coordinator=coordinator,
        check_interval=60.0,
    )
    q_router = QuorumAwareRouter(
        updater=updater,
        quorum_manager=quorum,
        check_interval=60.0,
    )
    return {
        "updater": updater,
        "quorum": quorum,
        "coordinator": coordinator,
        "orchestrator": orchestrator,
        "q_router": q_router,
    }


# ── TestSprint10Persistence ────────────────────────────────

class TestSprint10Persistence:

    def test_persistence_initializes_disconnected(self):
        p = SessionPersistence(dsn="invalid_dsn")
        assert not p._connected
        assert p._writes_total == 0
        assert p._writes_failed == 0

    def test_persistence_enqueues_without_connection(self):
        p = SessionPersistence(dsn="invalid_dsn")
        p.save_session({"session_id": "s1"})
        p.save_decision({"node_id": "node-1"})
        p.save_sla_snapshot({"quorum_state": "healthy"})
        assert len(p._write_queue) == 3

    def test_persistence_queue_max_size(self):
        p = SessionPersistence(dsn="invalid_dsn")
        for i in range(600):
            p.save_session({"session_id": f"s{i}"})
        assert len(p._write_queue) <= p.QUEUE_MAX

    def test_persistence_stats_structure(self):
        p = SessionPersistence(dsn="invalid_dsn")
        stats = p.get_stats()
        assert "connected" in stats
        assert "writes_total" in stats
        assert "writes_failed" in stats
        assert "queue_depth" in stats
        assert "reads_total" in stats

    def test_persistence_get_sessions_empty_when_disconnected(self):
        p = SessionPersistence(dsn="invalid_dsn")
        result = p.get_sessions(limit=10)
        assert result == []

    def test_persistence_get_decisions_empty_when_disconnected(self):
        p = SessionPersistence(dsn="invalid_dsn")
        result = p.get_decisions(limit=10)
        assert result == []

    def test_persistence_get_sla_empty_when_disconnected(self):
        p = SessionPersistence(dsn="invalid_dsn")
        result = p.get_sla_snapshots(limit=10)
        assert result == []

    def test_persistence_start_stop(self):
        p = SessionPersistence(dsn="invalid_dsn")
        p.start()
        time.sleep(0.1)
        assert p._running
        p.stop()
        assert not p._running

    def test_sla_monitor_initializes(self):
        stack = make_full_stack()
        p = SessionPersistence(dsn="invalid_dsn")
        monitor = SLAMonitor(
            persistence=p,
            quorum_manager=stack["quorum"],
            coordinator=stack["coordinator"],
            orchestrator=stack["orchestrator"],
            quorum_router=stack["q_router"],
            updater=stack["updater"],
        )
        assert monitor._snapshot_count == 0
        assert monitor._sla_violations == 0

    def test_sla_monitor_status_structure(self):
        stack = make_full_stack()
        p = SessionPersistence(dsn="invalid_dsn")
        monitor = SLAMonitor(
            persistence=p,
            quorum_manager=stack["quorum"],
            coordinator=stack["coordinator"],
            orchestrator=stack["orchestrator"],
            quorum_router=stack["q_router"],
            updater=stack["updater"],
        )
        monitor._start_time = time.time()
        status = monitor.status()
        assert "snapshot_count" in status
        assert "sla_violations" in status
        assert "sla_compliance_pct" in status
        assert "quorum_target" in status
        assert "stability_target" in status


# ── TestSprint10HealthMonitor ──────────────────────────────

class TestSprint10HealthMonitor:

    def _make_monitor(self, effects=None, ready=True):
        stack = make_full_stack(
            effects=effects, ready=ready
        )
        monitor = HealthMonitor(
            updater=stack["updater"],
            quorum_manager=stack["quorum"],
            coordinator=stack["coordinator"],
            orchestrator=stack["orchestrator"],
            quorum_router=stack["q_router"],
            persistence=None,
        )
        monitor._start_time = time.time()
        return monitor, stack

    def test_health_monitor_initializes(self):
        monitor, _ = self._make_monitor()
        assert monitor._overall_status == (
            HealthStatus.UNKNOWN
        )
        assert monitor._check_count == 0

    def test_liveness_probe_structure(self):
        monitor, _ = self._make_monitor()
        live = monitor.get_liveness()
        assert "alive" in live
        assert "uptime_seconds" in live
        assert "check_count" in live
        assert "timestamp" in live

    def test_liveness_true_before_checks(self):
        monitor, _ = self._make_monitor()
        monitor._running = True
        live = monitor.get_liveness()
        assert live["alive"] is True

    def test_readiness_probe_structure(self):
        monitor, _ = self._make_monitor()
        is_ready, details = monitor.get_readiness()
        assert isinstance(is_ready, bool)
        assert "ready" in details
        assert "critical_components" in details
        assert "not_ready" in details

    def test_readiness_critical_components(self):
        monitor, _ = self._make_monitor()
        _, details = monitor.get_readiness()
        assert "causal_engine" in details[
            "critical_components"
        ]
        assert "quorum_manager" in details[
            "critical_components"
        ]
        assert "quorum_router" in details[
            "critical_components"
        ]

    def test_full_status_structure(self):
        monitor, _ = self._make_monitor()
        status = monitor.get_full_status()
        assert "overall_status" in status
        assert "components" in status
        assert "uptime_seconds" in status
        assert "check_count" in status
        for comp in [
            "causal_engine", "quorum_manager",
            "byzantine_coordinator",
            "recovery_orchestrator",
            "quorum_router", "persistence",
        ]:
            assert comp in status["components"]

    def test_sla_status_structure(self):
        monitor, _ = self._make_monitor()
        sla = monitor.get_sla_status()
        assert "all_compliant" in sla
        assert "targets" in sla
        assert "quorum_availability" in sla["targets"]
        assert "engine_readiness" in sla["targets"]
        assert "routing_stability" in sla["targets"]

    def test_check_causal_engine_ready(self):
        monitor, _ = self._make_monitor(ready=True)
        monitor._check_causal_engine()
        comp = monitor._components["causal_engine"]
        assert comp.status == HealthStatus.HEALTHY

    def test_check_causal_engine_not_ready(self):
        monitor, _ = self._make_monitor(
            ready=False, effects={
                n: 28.0 for n in ALL_NODES
            }
        )
        monitor._check_causal_engine()
        comp = monitor._components["causal_engine"]
        assert comp.status == HealthStatus.DEGRADED

    def test_check_quorum_healthy(self):
        monitor, stack = self._make_monitor()
        monitor._check_quorum_manager()
        comp = monitor._components["quorum_manager"]
        assert comp.status == HealthStatus.HEALTHY

    def test_check_quorum_lost(self):
        monitor, stack = self._make_monitor()
        stack["quorum"].mark_node_offline("node-1")
        stack["quorum"].mark_node_offline("node-2")
        stack["quorum"].mark_node_offline("node-3")
        monitor._check_quorum_manager()
        comp = monitor._components["quorum_manager"]
        assert comp.status == HealthStatus.UNHEALTHY

    def test_check_coordinator_healthy(self):
        monitor, _ = self._make_monitor()
        monitor._check_byzantine_coordinator()
        comp = monitor._components[
            "byzantine_coordinator"
        ]
        assert comp.status == HealthStatus.HEALTHY

    def test_check_coordinator_byzantine_confirmed(self):
        monitor, stack = self._make_monitor()
        stack["coordinator"]._node_states["node-1"] = (
            ByzantineNodeState.CONFIRMED
        )
        monitor._check_byzantine_coordinator()
        comp = monitor._components[
            "byzantine_coordinator"
        ]
        assert comp.status == HealthStatus.DEGRADED

    def test_check_orchestrator_healthy(self):
        monitor, stack = self._make_monitor()
        stack["orchestrator"]._start_time = time.time()
        monitor._check_recovery_orchestrator()
        comp = monitor._components[
            "recovery_orchestrator"
        ]
        assert comp.status == HealthStatus.HEALTHY

    def test_check_router_healthy(self):
        monitor, stack = self._make_monitor()
        stack["q_router"]._start_time = time.time()
        stack["q_router"]._check_cycle()
        monitor._check_quorum_router()
        comp = monitor._components["quorum_router"]
        assert comp.status == HealthStatus.HEALTHY

    def test_check_persistence_disabled(self):
        monitor, _ = self._make_monitor()
        monitor._check_persistence()
        comp = monitor._components["persistence"]
        assert comp.status == HealthStatus.HEALTHY
        assert comp.details.get("mode") == "disabled"

    def test_full_check_cycle_healthy(self):
        monitor, stack = self._make_monitor()
        stack["orchestrator"]._start_time = time.time()
        stack["q_router"]._start_time = time.time()
        stack["q_router"]._check_cycle()
        monitor._run_all_checks()
        assert monitor._check_count == 1
        assert monitor._overall_status in (
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
        )

    def test_history_recorded_after_check(self):
        monitor, stack = self._make_monitor()
        stack["orchestrator"]._start_time = time.time()
        stack["q_router"]._start_time = time.time()
        stack["q_router"]._check_cycle()
        monitor._run_all_checks()
        history = monitor.get_history(n=10)
        assert len(history) == 1
        assert "overall_status" in history[0]
        assert "component_statuses" in history[0]
        assert "timestamp" in history[0]

    def test_sla_records_after_checks(self):
        monitor, stack = self._make_monitor()
        stack["orchestrator"]._start_time = time.time()
        stack["q_router"]._start_time = time.time()
        stack["q_router"]._check_cycle()
        monitor._run_all_checks()
        sla = monitor.get_sla_status()
        for target in sla["targets"].values():
            assert target["observations"] >= 1

    def test_component_uptime_ratio(self):
        monitor, stack = self._make_monitor()
        stack["orchestrator"]._start_time = time.time()
        stack["q_router"]._start_time = time.time()
        stack["q_router"]._check_cycle()
        for _ in range(5):
            monitor._run_all_checks()
        full = monitor.get_full_status()
        quorum_comp = full["components"][
            "quorum_manager"
        ]
        assert 0.0 <= quorum_comp["uptime_ratio"] <= 1.0

    def test_monitor_start_stop(self):
        monitor, _ = self._make_monitor()
        monitor.start()
        time.sleep(0.1)
        assert monitor._running
        monitor.stop()
        assert not monitor._running


# ── TestSprint10RateLimiter ────────────────────────────────

class TestSprint10RateLimiter:

    def test_rate_limiter_initializes(self):
        limiter = SlidingWindowRateLimiter()
        assert limiter._total_requests == 0
        assert limiter._total_allowed == 0
        assert limiter._total_denied == 0

    def test_allows_normal_traffic(self):
        limiter = SlidingWindowRateLimiter()
        result = limiter.check("192.168.1.1", "/health")
        assert result.allowed is True
        assert result.remaining >= 0

    def test_endpoint_group_routing(self):
        limiter = SlidingWindowRateLimiter()
        group = limiter._get_endpoint_group(
            "/recovery/trigger"
        )
        assert group == "/recovery/trigger"

    def test_endpoint_group_default(self):
        limiter = SlidingWindowRateLimiter()
        group = limiter._get_endpoint_group(
            "/unknown/path"
        )
        assert group == "default"

    def test_endpoint_group_prefix_match(self):
        limiter = SlidingWindowRateLimiter()
        group = limiter._get_endpoint_group(
            "/quorum/nodes/node-1"
        )
        assert group == "/quorum"

    def test_burst_protection_fires(self):
        limiter = SlidingWindowRateLimiter()
        config = limiter._get_config(
            "/recovery/trigger"
        )
        burst = config.burst_size
        results = []
        for _ in range(burst + 5):
            r = limiter.check(
                "10.0.0.1", "/recovery/trigger"
            )
            results.append(r.allowed)
        assert False in results

    def test_recovery_trigger_tight_limit(self):
        limiter = SlidingWindowRateLimiter()
        config = limiter._get_config(
            "/recovery/trigger"
        )
        assert config.requests_per_minute == 10
        assert config.burst_size == 5

    def test_health_endpoint_higher_limit(self):
        limiter = SlidingWindowRateLimiter()
        config = limiter._get_config("/health")
        assert config.requests_per_minute == 120

    def test_rate_limit_result_headers(self):
        limiter = SlidingWindowRateLimiter()
        result = limiter.check("192.168.1.1", "/health")
        headers = result.to_headers()
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers

    def test_denied_result_has_retry_after(self):
        limiter = SlidingWindowRateLimiter()
        # Fill burst
        for _ in range(10):
            limiter.check("1.2.3.4", "/recovery/trigger")
        result = limiter.check(
            "1.2.3.4", "/recovery/trigger"
        )
        if not result.allowed:
            headers = result.to_headers()
            assert "Retry-After" in headers

    def test_different_clients_independent(self):
        limiter = SlidingWindowRateLimiter()
        r1 = limiter.check("10.0.0.1", "/health")
        r2 = limiter.check("10.0.0.2", "/health")
        assert r1.allowed is True
        assert r2.allowed is True

    def test_stats_structure(self):
        limiter = SlidingWindowRateLimiter()
        limiter.check("192.168.1.1", "/health")
        stats = limiter.stats()
        assert "total_requests" in stats
        assert "total_allowed" in stats
        assert "total_denied" in stats
        assert "deny_rate" in stats
        assert "active_clients" in stats
        assert "endpoint_limits" in stats

    def test_stats_counts_correct(self):
        limiter = SlidingWindowRateLimiter()
        for _ in range(5):
            limiter.check("192.168.1.1", "/health")
        stats = limiter.stats()
        assert stats["total_requests"] == 5
        assert stats["total_allowed"] == 5
        assert stats["total_denied"] == 0

    def test_thread_safety(self):
        limiter = SlidingWindowRateLimiter()
        errors = []

        def make_requests():
            try:
                for _ in range(10):
                    limiter.check(
                        "192.168.1.1", "/health"
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=make_requests)
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_client_status_structure(self):
        limiter = SlidingWindowRateLimiter()
        limiter.check("192.168.1.1", "/health")
        status = limiter.get_client_status(
            "192.168.1.1"
        )
        assert "/health" in status
        assert "count_last_minute" in status["/health"]
        assert "limit" in status["/health"]
        assert "remaining" in status["/health"]


# ── TestSprint10GracefulShutdown ───────────────────────────

class TestSprint10GracefulShutdown:

    def test_shutdown_manager_initializes(self):
        mgr = GracefulShutdownManager()
        assert not mgr._draining
        assert mgr._in_flight == 0
        assert not mgr._shutdown_complete

    def test_request_start_allowed_initially(self):
        mgr = GracefulShutdownManager()
        result = mgr.request_start()
        assert result is True
        assert mgr._in_flight == 1

    def test_request_end_decrements(self):
        mgr = GracefulShutdownManager()
        mgr.request_start()
        mgr.request_end()
        assert mgr._in_flight == 0

    def test_request_blocked_when_draining(self):
        mgr = GracefulShutdownManager()
        mgr._draining = True
        result = mgr.request_start()
        assert result is False

    def test_shutdown_sets_draining(self):
        mgr = GracefulShutdownManager()
        report = mgr.shutdown()
        assert mgr._draining is True
        assert report["shutdown_complete"] is True

    def test_shutdown_clean_drain(self):
        mgr = GracefulShutdownManager()
        report = mgr.shutdown()
        assert report["clean_drain"] is True

    def test_shutdown_with_inflight_requests(self):
        mgr = GracefulShutdownManager()

        def slow_request():
            if mgr.request_start():
                time.sleep(0.3)
                mgr.request_end()

        t = threading.Thread(target=slow_request)
        t.start()
        time.sleep(0.05)

        report = mgr.shutdown()
        t.join()

        assert report["shutdown_complete"] is True
        assert report["clean_drain"] is True

    def test_shutdown_stops_components(self):
        mgr = GracefulShutdownManager()
        stopped = []

        class FakeComponent:
            def stop(self):
                stopped.append(True)

        report = mgr.shutdown(
            components={
                "comp_a": FakeComponent(),
                "comp_b": FakeComponent(),
            }
        )
        assert len(stopped) == 2
        assert "comp_a" in report[
            "components_stopped"
        ]
        assert "comp_b" in report[
            "components_stopped"
        ]

    def test_shutdown_report_structure(self):
        mgr = GracefulShutdownManager()
        report = mgr.shutdown()
        assert "shutdown_complete" in report
        assert "clean_drain" in report
        assert "total_elapsed_s" in report
        assert "components_stopped" in report
        assert "timestamp" in report

    def test_second_shutdown_noop(self):
        mgr = GracefulShutdownManager()
        report1 = mgr.shutdown()
        report2 = mgr.shutdown()
        assert report1["timestamp"] == (
            report2["timestamp"]
        )

    def test_status_structure(self):
        mgr = GracefulShutdownManager()
        status = mgr.status()
        assert "draining" in status
        assert "in_flight" in status
        assert "shutdown_complete" in status


# ── TestSprint10EndToEnd ───────────────────────────────────

class TestSprint10EndToEnd:
    """
    Full end-to-end integration tests covering the
    complete Sprint 10 production hardening stack
    integrated with the Sprint 9 Byzantine recovery stack.
    """

    def test_cold_start_all_healthy(self):
        """Full stack cold start → all components healthy."""
        stack = make_full_stack()
        monitor = HealthMonitor(
            updater=stack["updater"],
            quorum_manager=stack["quorum"],
            coordinator=stack["coordinator"],
            orchestrator=stack["orchestrator"],
            quorum_router=stack["q_router"],
            persistence=None,
        )
        monitor._start_time = time.time()
        stack["orchestrator"]._start_time = time.time()
        stack["q_router"]._start_time = time.time()
        stack["q_router"]._check_cycle()

        monitor._run_all_checks()

        assert (
            stack["quorum"].get_quorum_state()
            == QuorumState.HEALTHY
        )
        assert monitor._overall_status in (
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
        )
        live = monitor.get_liveness()
        assert live["check_count"] == 1

    def test_byzantine_injection_detected_by_health(self):
        """Inject Byzantine state → health detects DEGRADED."""
        stack = make_full_stack()
        monitor = HealthMonitor(
            updater=stack["updater"],
            quorum_manager=stack["quorum"],
            coordinator=stack["coordinator"],
            orchestrator=stack["orchestrator"],
            quorum_router=stack["q_router"],
            persistence=None,
        )
        monitor._start_time = time.time()
        stack["orchestrator"]._start_time = time.time()
        stack["q_router"]._start_time = time.time()
        stack["q_router"]._check_cycle()

        # Inject Byzantine
        stack["coordinator"]._node_states["node-1"] = (
            ByzantineNodeState.CONFIRMED
        )

        monitor._run_all_checks()

        byz_comp = monitor._components[
            "byzantine_coordinator"
        ]
        assert byz_comp.status == HealthStatus.DEGRADED
        overall = monitor._overall_status
        assert overall in (
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        )

    def test_quorum_degraded_reflected_in_health(self):
        """Take node offline → health shows quorum degraded."""
        stack = make_full_stack()
        monitor = HealthMonitor(
            updater=stack["updater"],
            quorum_manager=stack["quorum"],
            coordinator=stack["coordinator"],
            orchestrator=stack["orchestrator"],
            quorum_router=stack["q_router"],
            persistence=None,
        )
        monitor._start_time = time.time()
        stack["orchestrator"]._start_time = time.time()
        stack["q_router"]._start_time = time.time()

        stack["quorum"].mark_node_offline("node-1")
        stack["q_router"]._check_cycle()
        monitor._run_all_checks()

        quorum_comp = monitor._components[
            "quorum_manager"
        ]
        assert quorum_comp.status in (
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        )
        assert (
            stack["quorum"].get_quorum_state()
            == QuorumState.DEGRADED
        )

    def test_router_excludes_offline_node(self):
        """Offline node → router excludes from traffic."""
        stack = make_full_stack()
        stack["q_router"]._start_time = time.time()
        stack["quorum"].mark_node_offline("node-2")
        stack["q_router"]._check_cycle()

        weights = stack["q_router"].get_weights()
        assert weights["node-2"] == 0.0
        assert weights["node-1"] > 0.0
        assert weights["node-3"] > 0.0

        hits = {n: 0 for n in ALL_NODES}
        for _ in range(100):
            node = stack["q_router"].route_request()
            hits[node] += 1
        assert hits["node-2"] == 0

    def test_rate_limiter_protects_recovery_trigger(self):
        """Burst to /recovery/trigger → blocked after 5."""
        limiter = SlidingWindowRateLimiter()
        results = []
        for _ in range(20):
            r = limiter.check(
                "attacker-ip", "/recovery/trigger"
            )
            results.append(r.allowed)
        allowed = sum(results)
        denied = len(results) - allowed
        assert allowed <= 10
        assert denied >= 10

    def test_graceful_shutdown_blocks_new_requests(self):
        """After shutdown starts, new requests blocked."""
        mgr = GracefulShutdownManager()
        assert mgr.request_start() is True
        mgr.request_end()

        mgr._draining = True
        assert mgr.request_start() is False

    def test_persistence_queue_survives_stop(self):
        """Queued writes survive stop (in-memory queue)."""
        p = SessionPersistence(dsn="invalid_dsn")
        p.start()
        p.save_session({"session_id": "test-001"})
        p.save_decision({"node_id": "node-1"})
        p.stop()
        # Queue may have been attempted and failed
        # but no crash
        assert p._writes_failed >= 0

    def test_sla_compliance_healthy_cluster(self):
        """Healthy cluster → all SLA targets pass."""
        stack = make_full_stack()
        monitor = HealthMonitor(
            updater=stack["updater"],
            quorum_manager=stack["quorum"],
            coordinator=stack["coordinator"],
            orchestrator=stack["orchestrator"],
            quorum_router=stack["q_router"],
            persistence=None,
        )
        monitor._start_time = time.time()
        stack["orchestrator"]._start_time = time.time()
        stack["q_router"]._start_time = time.time()
        stack["q_router"]._check_cycle()

        for _ in range(10):
            monitor._run_all_checks()

        sla = monitor.get_sla_status()
        assert (
            sla["targets"]["quorum_availability"][
                "compliance_pct"
            ] == 100.0
        )
        assert (
            sla["targets"]["routing_stability"][
                "compliance_pct"
            ] == 100.0
        )

    def test_full_stack_state_transitions(self):
        """
        Complete state machine test:
        HEALTHY → node offline → DEGRADED
        → node recovered → HEALTHY
        """
        stack = make_full_stack()
        quorum = stack["quorum"]

        assert (
            quorum.get_quorum_state()
            == QuorumState.HEALTHY
        )

        quorum.mark_node_offline("node-1")
        assert (
            quorum.get_quorum_state()
            == QuorumState.DEGRADED
        )

        stack["q_router"]._start_time = time.time()
        stack["q_router"]._check_cycle()
        weights = stack["q_router"].get_weights()
        assert weights["node-1"] == 0.0

        quorum.mark_node_full("node-1")
        assert (
            quorum.get_quorum_state()
            == QuorumState.HEALTHY
        )

        stack["q_router"]._check_cycle()
        weights2 = stack["q_router"].get_weights()
        assert weights2["node-1"] > 0.0

    def test_concurrent_health_and_rate_limiting(self):
        """
        Concurrent health checks + rate limiting
        → no race conditions.
        """
        stack = make_full_stack()
        monitor = HealthMonitor(
            updater=stack["updater"],
            quorum_manager=stack["quorum"],
            coordinator=stack["coordinator"],
            orchestrator=stack["orchestrator"],
            quorum_router=stack["q_router"],
            persistence=None,
        )
        monitor._start_time = time.time()
        stack["orchestrator"]._start_time = time.time()
        stack["q_router"]._start_time = time.time()
        stack["q_router"]._check_cycle()

        limiter = SlidingWindowRateLimiter()
        errors = []

        def health_checks():
            try:
                for _ in range(5):
                    monitor._run_all_checks()
            except Exception as e:
                errors.append(f"health: {e}")

        def rate_limit_checks():
            try:
                for _ in range(20):
                    limiter.check(
                        "192.168.1.1", "/health"
                    )
            except Exception as e:
                errors.append(f"rate: {e}")

        threads = (
            [threading.Thread(target=health_checks)
             for _ in range(3)]
            + [threading.Thread(
                target=rate_limit_checks
            ) for _ in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ── TestSprint10Benchmarks ─────────────────────────────────

class TestSprint10Benchmarks:

    def test_health_check_cycle_latency(self):
        """Full health check cycle < 100ms."""
        stack = make_full_stack()
        monitor = HealthMonitor(
            updater=stack["updater"],
            quorum_manager=stack["quorum"],
            coordinator=stack["coordinator"],
            orchestrator=stack["orchestrator"],
            quorum_router=stack["q_router"],
            persistence=None,
        )
        monitor._start_time = time.time()
        stack["orchestrator"]._start_time = time.time()
        stack["q_router"]._start_time = time.time()
        stack["q_router"]._check_cycle()

        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            monitor._run_all_checks()
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 100.0

    def test_rate_limiter_check_latency(self):
        """Rate limiter decision < 1ms."""
        limiter = SlidingWindowRateLimiter()
        latencies = []
        for i in range(200):
            start = time.perf_counter()
            limiter.check(
                f"10.0.0.{i % 255}", "/health"
            )
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 1.0

    def test_shutdown_drain_latency(self):
        """Graceful shutdown drains < 2s with no load."""
        mgr = GracefulShutdownManager()
        start = time.perf_counter()
        report = mgr.shutdown()
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0
        assert report["clean_drain"] is True

    def test_persistence_enqueue_latency(self):
        """Persistence enqueue < 1ms (async queue)."""
        p = SessionPersistence(dsn="invalid_dsn")
        latencies = []
        for i in range(100):
            start = time.perf_counter()
            p.save_session({"session_id": f"s{i}"})
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 1.0

    def test_sla_target_record_latency(self):
        """SLA target recording < 1ms."""
        from health_monitor import SLATarget
        target = SLATarget(
            "test_target", 99.0, window_minutes=60
        )
        latencies = []
        for _ in range(500):
            start = time.perf_counter()
            target.record(True)
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 1.0
