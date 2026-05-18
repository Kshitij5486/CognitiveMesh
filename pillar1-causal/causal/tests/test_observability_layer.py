import sys
import os
import time
import statistics
import threading
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "telemetry"))

from prometheus_client import CollectorRegistry
from metrics_collector import MetricsCollector, MetricSnapshot
from causal_metrics_adapter import CausalMetricsAdapter
from healing_metrics_adapter import HealingMetricsAdapter


def make_mock_updater(
    ready=True,
    effect=28.5,
    load=5.0,
    buffer_size=60,
    retrain_count=5,
):
    updater = MagicMock()
    updater.status.return_value = {
        "is_ready": ready,
        "buffer_size": buffer_size,
        "retrain_count": retrain_count,
        "nodes_modeled": ["node-1", "node-2", "node-3"],
        "max_buffer_size": 200,
        "last_retrain_duration_ms": 101.0,
    }

    def get_snapshot(node_id):
        offsets = {"node-1": 0.0, "node-2": 1.5, "node-3": 3.0}
        return {
            "effect": effect + offsets.get(node_id, 0),
            "samples_used": buffer_size,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    updater.get_current_snapshot.side_effect = get_snapshot

    import pandas as pd
    df = pd.DataFrame({
        "node_1_active_queries": [load] * 30,
        "node_1_avg_query_duration_ms": [load * effect] * 30,
        "node_2_active_queries": [load] * 30,
        "node_2_avg_query_duration_ms": [load * effect] * 30,
        "node_3_active_queries": [load] * 30,
        "node_3_avg_query_duration_ms": [load * effect] * 30,
    })
    mock_buf = MagicMock()
    mock_buf.get_dataframe.return_value = df
    updater.buffer = mock_buf
    return updater


def make_mock_analyzer(observations=50, simulations=10):
    analyzer = MagicMock()
    analyzer.status.return_value = {
        "observations_collected": observations,
        "checks_run": 10,
    }
    return analyzer


def make_mock_simulator(simulations=10, worst_case=450.0):
    simulator = MagicMock()
    simulator.status.return_value = {
        "simulations_run": simulations,
        "last_worst_case_ms": worst_case,
    }
    return simulator


def make_mock_alerter(active_alerts=None, total_fired=20):
    alerter = MagicMock()
    active = active_alerts or []
    alerter.get_active_alerts.return_value = active
    alerter.status.return_value = {
        "running": True,
        "total_alerts_fired": total_fired,
        "active_alert_count": len(active),
        "checks_run": 10,
    }
    return alerter


def make_mock_engine(
    checks=10,
    total_actions=50,
    successful=50,
    failed=0,
):
    engine = MagicMock()
    engine.status.return_value = {
        "running": True,
        "auto_heal": True,
        "checks_run": checks,
        "total_actions": total_actions,
        "successful_actions": successful,
        "failed_actions": failed,
        "cooldown_seconds": 30.0,
        "active_cooldowns": {},
    }
    history = [
        {
            "action_id": f"action-{i:04d}",
            "action_type": "rebalance",
            "node_id": f"node-{(i%3)+1}",
            "status": "success",
            "duration_ms": 0.5,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        for i in range(min(total_actions, 50))
    ]
    engine.get_action_history.return_value = history
    return engine


def make_mock_router(
    checks=10,
    reroutes=0,
    recoveries=0,
    active_nodes=3,
):
    router = MagicMock()
    router.status.return_value = {
        "running": True,
        "strategy": "causal_weighted",
        "checks_run": checks,
        "total_reroutes": reroutes,
        "total_recoveries": recoveries,
        "active_decisions": 0,
        "node_states": {
            "node-1": "active",
            "node-2": "active",
            "node-3": "active",
        },
        "active_nodes": active_nodes,
        "rerouted_nodes": 0,
        "isolated_nodes": 0,
    }
    router._compute_causal_weights.return_value = {
        "node-1": 0.36,
        "node-2": 0.34,
        "node-3": 0.30,
    }
    return router


def make_mock_retrainer(
    checks=10,
    total=2,
    successful=2,
    failed=0,
):
    retrainer = MagicMock()
    retrainer.status.return_value = {
        "running": True,
        "auto_retrain": True,
        "checks_run": checks,
        "total_retrains": total,
        "successful_retrains": successful,
        "failed_retrains": failed,
        "drift_threshold_ms": 3.0,
        "cooldown_seconds": 60.0,
        "current_effects": {
            "node-1": 28.5,
            "node-2": 30.0,
            "node-3": 31.5,
        },
    }
    retrainer._is_in_cooldown.return_value = False
    return retrainer


def make_mock_orchestrator(
    checks=10,
    total=1,
    successful=1,
    failed=0,
    active=0,
):
    orchestrator = MagicMock()
    orchestrator.status.return_value = {
        "running": True,
        "auto_recover": True,
        "checks_run": checks,
        "total_sequences": total,
        "successful_recoveries": successful,
        "failed_recoveries": failed,
        "active_sequences": active,
        "active_nodes": [],
        "verify_threshold_ms": 120.0,
        "verify_wait_seconds": 15.0,
    }
    orchestrator.get_sequence_history.return_value = [
        {
            "sequence_id": "recovery-001",
            "node_id": "node-1",
            "trigger": "manual",
            "phase": "restored",
            "duration_seconds": 15.1,
            "actions_taken": [
                {"action_type": "reroute", "success": True},
                {"action_type": "retrain", "success": True},
                {"action_type": "verify", "success": True},
            ],
        }
    ]
    return orchestrator


# ── MetricSnapshot tests ───────────────────────────────────

class TestMetricSnapshot:

    def test_snapshot_initializes(self):
        snap = MetricSnapshot("2026-01-01T00:00:00+00:00")
        assert snap.timestamp == "2026-01-01T00:00:00+00:00"
        assert snap.causal_engine_ready is False
        assert snap.causal_effects == {}
        assert snap.alerts_active == 0

    def test_snapshot_to_dict(self):
        snap = MetricSnapshot("2026-01-01T00:00:00+00:00")
        snap.causal_engine_ready = True
        snap.causal_effects = {"node-1": 28.5}
        snap.alerts_active = 3
        d = snap.to_dict()
        assert "causal_engine" in d
        assert "predictive_stack" in d
        assert "healing_engine" in d
        assert "router" in d
        assert "retrainer" in d
        assert "orchestrator" in d

    def test_snapshot_causal_effects_stored(self):
        snap = MetricSnapshot("2026-01-01T00:00:00+00:00")
        snap.causal_effects = {
            "node-1": 28.5,
            "node-2": 30.0,
            "node-3": 31.5,
        }
        assert len(snap.causal_effects) == 3
        assert snap.causal_effects["node-1"] == 28.5

    def test_snapshot_alert_counts(self):
        snap = MetricSnapshot("2026-01-01T00:00:00+00:00")
        snap.alerts_active = 10
        snap.alerts_fired_total = 50
        snap.alerts_by_type = {"latency_rising": 6}
        snap.alerts_by_severity = {"warning": 8, "critical": 2}
        assert snap.alerts_active == 10
        assert snap.alerts_by_type["latency_rising"] == 6

    def test_snapshot_healing_metrics(self):
        snap = MetricSnapshot("2026-01-01T00:00:00+00:00")
        snap.healing_actions_total = 100
        snap.healing_actions_successful = 98
        snap.healing_actions_failed = 2
        snap.healing_actions_by_type = {"rebalance": 80}
        assert snap.healing_actions_total == 100

    def test_snapshot_router_metrics(self):
        snap = MetricSnapshot("2026-01-01T00:00:00+00:00")
        snap.router_active_nodes = 3
        snap.router_rerouted_nodes = 0
        snap.router_causal_weights = {
            "node-1": 0.36,
            "node-2": 0.34,
            "node-3": 0.30,
        }
        assert snap.router_active_nodes == 3
        assert abs(sum(snap.router_causal_weights.values()) - 1.0) < 0.01


# ── MetricsCollector tests ─────────────────────────────────

class TestMetricsCollector:

    def make_collector(self, **kwargs):
        updater = make_mock_updater(**kwargs)
        analyzer = make_mock_analyzer()
        simulator = make_mock_simulator()
        alerter = make_mock_alerter()
        engine = make_mock_engine()
        router = make_mock_router()
        retrainer = make_mock_retrainer()
        orchestrator = make_mock_orchestrator()
        return MetricsCollector(
            updater=updater,
            analyzer=analyzer,
            simulator=simulator,
            alerter=alerter,
            engine=engine,
            router=router,
            retrainer=retrainer,
            orchestrator=orchestrator,
            collection_interval=60.0,
        )

    def test_collector_initializes(self):
        c = self.make_collector()
        assert c is not None
        assert c._collection_count == 0
        assert c._collection_errors == 0

    def test_collect_snapshot_returns_snapshot(self):
        c = self.make_collector()
        snap = c._collect_snapshot()
        assert snap is not None
        assert isinstance(snap, MetricSnapshot)

    def test_snapshot_has_causal_effects(self):
        c = self.make_collector()
        snap = c._collect_snapshot()
        assert len(snap.causal_effects) == 3
        assert "node-1" in snap.causal_effects
        assert snap.causal_effects["node-1"] > 0

    def test_snapshot_engine_ready(self):
        c = self.make_collector(ready=True)
        snap = c._collect_snapshot()
        assert snap.causal_engine_ready is True

    def test_snapshot_engine_not_ready(self):
        c = self.make_collector(ready=False)
        snap = c._collect_snapshot()
        assert snap.causal_engine_ready is False

    def test_snapshot_buffer_size(self):
        c = self.make_collector(buffer_size=80)
        snap = c._collect_snapshot()
        assert snap.causal_buffer_size == 80

    def test_snapshot_alerts_collected(self):
        updater = make_mock_updater()
        alerter = make_mock_alerter(
            active_alerts=[
                {
                    "alert_id": "a1",
                    "node_id": "node-1",
                    "alert_type": "latency_rising",
                    "severity": "warning",
                    "acknowledged": False,
                }
            ],
            total_fired=5,
        )
        c = MetricsCollector(
            updater=updater,
            analyzer=make_mock_analyzer(),
            simulator=make_mock_simulator(),
            alerter=alerter,
            engine=make_mock_engine(),
            router=make_mock_router(),
            retrainer=make_mock_retrainer(),
            orchestrator=make_mock_orchestrator(),
            collection_interval=60.0,
        )
        snap = c._collect_snapshot()
        assert snap.alerts_active == 1
        assert snap.alerts_fired_total == 5

    def test_snapshot_router_nodes(self):
        c = self.make_collector()
        snap = c._collect_snapshot()
        assert snap.router_active_nodes == 3
        assert snap.router_rerouted_nodes == 0

    def test_get_latest_snapshot_none_before_first_collection(self):
        c = self.make_collector()
        assert c.get_latest_snapshot() is None

    def test_status_structure(self):
        c = self.make_collector()
        status = c.status()
        assert "running" in status
        assert "collection_count" in status
        assert "collection_errors" in status
        assert "uptime_seconds" in status

    def test_get_snapshot_history_empty(self):
        c = self.make_collector()
        history = c.get_snapshot_history()
        assert history == []

    def test_uptime_increases(self):
        c = self.make_collector()
        c._start_time = time.time() - 10
        uptime = c.get_uptime_seconds()
        assert uptime >= 10.0


# ── CausalMetricsAdapter tests ─────────────────────────────

class TestCausalMetricsAdapter:

    def make_adapter(self, effect=28.5, ready=True):
        updater = make_mock_updater(
            effect=effect, ready=ready
        )
        registry = CollectorRegistry()
        return CausalMetricsAdapter(
            updater=updater,
            registry=registry,
            collection_interval=60.0,
        ), registry

    def test_adapter_initializes(self):
        adapter, _ = self.make_adapter()
        assert adapter is not None
        assert adapter._collection_count == 0

    def test_collect_returns_effects(self):
        adapter, _ = self.make_adapter(effect=28.5)
        effects = adapter.collect()
        assert len(effects) == 3
        assert "node-1" in effects
        assert effects["node-1"] > 0

    def test_effects_increase_with_mock(self):
        adapter, _ = self.make_adapter(effect=30.0)
        effects = adapter.collect()
        assert effects["node-1"] == pytest.approx(30.0, abs=0.1)

    def test_stability_score_computed(self):
        adapter, _ = self.make_adapter()
        for _ in range(10):
            adapter.collect()
        report = adapter.get_stability_report()
        assert "node-1" in report
        assert "stability_score" in report["node-1"]
        assert 0.0 <= report["node-1"]["stability_score"] <= 1.0

    def test_stability_report_all_nodes(self):
        adapter, _ = self.make_adapter()
        for _ in range(5):
            adapter.collect()
        report = adapter.get_stability_report()
        assert len(report) == 3
        for node_id in ["node-1", "node-2", "node-3"]:
            assert node_id in report

    def test_stability_report_structure(self):
        adapter, _ = self.make_adapter()
        for _ in range(5):
            adapter.collect()
        report = adapter.get_stability_report()
        for node_id, data in report.items():
            assert "mean_effect_ms" in data
            assert "std_dev_ms" in data
            assert "stability_score" in data
            assert "observations" in data
            assert "effect_range_ms" in data

    def test_drift_detected_between_collections(self):
        adapter, _ = self.make_adapter(effect=25.0)
        adapter.collect()
        adapter.updater.get_current_snapshot.side_effect = (
            lambda node_id: {
                "effect": 35.0,
                "samples_used": 60,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        )
        adapter.collect()
        with adapter._lock:
            last = adapter._last_effects.get("node-1", 0)
        assert last == pytest.approx(35.0, abs=0.1)

    def test_cross_node_spread_computed(self):
        adapter, _ = self.make_adapter(effect=28.5)
        effects = adapter.collect()
        spread = max(effects.values()) - min(effects.values())
        assert spread >= 0.0

    def test_status_structure(self):
        adapter, _ = self.make_adapter()
        status = adapter.status()
        assert "running" in status
        assert "collection_count" in status
        assert "drift_threshold_ms" in status
        assert "history_sizes" in status

    def test_collection_count_increments(self):
        adapter, _ = self.make_adapter()
        adapter.collect()
        adapter.collect()
        adapter.collect()
        assert adapter._collection_count == 3

    def test_history_window_respected(self):
        adapter, _ = self.make_adapter()
        for _ in range(70):
            adapter.collect()
        with adapter._lock:
            hist_size = len(
                adapter._effect_history["node-1"]
            )
        assert hist_size <= adapter.HISTORY_WINDOW

    def test_metrics_registered_in_registry(self):
        adapter, registry = self.make_adapter()
        from prometheus_client import generate_latest
        output = generate_latest(registry).decode("utf-8")
        assert "cognitivemesh_causal_effect_distribution" in output
        assert "cognitivemesh_causal_model_stability_score" in output
        assert "cognitivemesh_causal_drift_magnitude_ms" in output


# ── HealingMetricsAdapter tests ────────────────────────────

class TestHealingMetricsAdapter:

    def make_adapter(
        self,
        engine_actions=50,
        reroutes=0,
        retrains=2,
        sequences=1,
    ):
        engine = make_mock_engine(total_actions=engine_actions)
        router = make_mock_router(reroutes=reroutes)
        retrainer = make_mock_retrainer(total=retrains)
        orchestrator = make_mock_orchestrator(total=sequences)
        registry = CollectorRegistry()
        adapter = HealingMetricsAdapter(
            engine=engine,
            router=router,
            retrainer=retrainer,
            orchestrator=orchestrator,
            registry=registry,
            collection_interval=60.0,
        )
        adapter._start_time = time.time()
        return adapter, registry

    def test_adapter_initializes(self):
        adapter, _ = self.make_adapter()
        assert adapter is not None
        assert adapter._collection_count == 0

    def test_collect_returns_overall_score(self):
        adapter, _ = self.make_adapter()
        score = adapter.collect()
        assert 0.0 <= score <= 1.0

    def test_healthy_system_score_high(self):
        adapter, _ = self.make_adapter(
            engine_actions=100,
            reroutes=0,
            retrains=2,
            sequences=1,
        )
        score = adapter.collect()
        assert score >= 0.7

    def test_health_report_structure(self):
        adapter, _ = self.make_adapter()
        report = adapter.get_health_report()
        assert "components" in report
        comps = report["components"]
        assert "healing_engine" in comps
        assert "router" in comps
        assert "retrainer" in comps
        assert "orchestrator" in comps

    def test_health_report_engine_fields(self):
        adapter, _ = self.make_adapter(engine_actions=80)
        report = adapter.get_health_report()
        eng = report["components"]["healing_engine"]
        assert "checks_run" in eng
        assert "total_actions" in eng
        assert "action_success_rate" in eng
        assert eng["total_actions"] == 80

    def test_health_report_router_fields(self):
        adapter, _ = self.make_adapter(reroutes=3)
        report = adapter.get_health_report()
        rtr = report["components"]["router"]
        assert "checks_run" in rtr
        assert "total_reroutes" in rtr
        assert "active_nodes" in rtr
        assert rtr["total_reroutes"] == 3

    def test_health_report_retrainer_fields(self):
        adapter, _ = self.make_adapter(retrains=5)
        report = adapter.get_health_report()
        ret = report["components"]["retrainer"]
        assert "total_retrains" in ret
        assert "retrain_success_rate" in ret
        assert ret["total_retrains"] == 5

    def test_health_report_orchestrator_fields(self):
        adapter, _ = self.make_adapter(sequences=2)
        report = adapter.get_health_report()
        orch = report["components"]["orchestrator"]
        assert "total_sequences" in orch
        assert "successful_recoveries" in orch
        assert orch["total_sequences"] == 2

    def test_collection_count_increments(self):
        adapter, _ = self.make_adapter()
        adapter.collect()
        adapter.collect()
        assert adapter._collection_count == 2

    def test_metrics_in_registry(self):
        adapter, registry = self.make_adapter()
        adapter.collect()
        from prometheus_client import generate_latest
        output = generate_latest(registry).decode("utf-8")
        assert "cognitivemesh_healing_engine_health_score" in output
        assert "cognitivemesh_router_health_score" in output
        assert "cognitivemesh_retrainer_health_score" in output
        assert "cognitivemesh_self_healing_overall_score" in output
        assert "cognitivemesh_self_healing_fabric_status" in output

    def test_status_structure(self):
        adapter, _ = self.make_adapter()
        status = adapter.status()
        assert "running" in status
        assert "collection_count" in status
        assert "metrics_families" in status

    def test_thread_safety_concurrent_collects(self):
        adapter, _ = self.make_adapter()
        errors = []

        def do_collect():
            try:
                for _ in range(5):
                    adapter.collect()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=do_collect)
            for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ── Integration tests ──────────────────────────────────────

class TestObservabilityIntegration:

    def test_collector_feeds_causal_adapter(self):
        updater = make_mock_updater(effect=28.5)
        analyzer = make_mock_analyzer()
        simulator = make_mock_simulator()
        alerter = make_mock_alerter()
        engine = make_mock_engine()
        router = make_mock_router()
        retrainer = make_mock_retrainer()
        orchestrator = make_mock_orchestrator()

        collector = MetricsCollector(
            updater=updater,
            analyzer=analyzer,
            simulator=simulator,
            alerter=alerter,
            engine=engine,
            router=router,
            retrainer=retrainer,
            orchestrator=orchestrator,
            collection_interval=60.0,
        )
        snap = collector._collect_snapshot()
        assert snap.causal_effects["node-1"] > 0
        assert snap.causal_effects["node-2"] > 0
        assert snap.causal_effects["node-3"] > 0

    def test_causal_adapter_and_healing_adapter_independent_registries(self):
        updater = make_mock_updater()
        engine = make_mock_engine()
        router = make_mock_router()
        retrainer = make_mock_retrainer()
        orchestrator = make_mock_orchestrator()

        registry1 = CollectorRegistry()
        registry2 = CollectorRegistry()

        causal = CausalMetricsAdapter(
            updater=updater,
            registry=registry1,
            collection_interval=60.0,
        )
        healing = HealingMetricsAdapter(
            engine=engine,
            router=router,
            retrainer=retrainer,
            orchestrator=orchestrator,
            registry=registry2,
            collection_interval=60.0,
        )

        causal.collect()
        healing.collect()

        from prometheus_client import generate_latest
        out1 = generate_latest(registry1).decode("utf-8")
        out2 = generate_latest(registry2).decode("utf-8")

        assert "causal_effect_distribution" in out1
        assert "healing_engine_health_score" in out2
        assert "healing_engine_health_score" not in out1
        assert "causal_effect_distribution" not in out2

    def test_full_pipeline_snapshot_all_fields(self):
        updater = make_mock_updater(
            effect=27.5,
            load=5.0,
            buffer_size=60,
            retrain_count=10,
        )
        collector = MetricsCollector(
            updater=updater,
            analyzer=make_mock_analyzer(observations=100),
            simulator=make_mock_simulator(
                simulations=20, worst_case=350.0
            ),
            alerter=make_mock_alerter(total_fired=50),
            engine=make_mock_engine(
                total_actions=200, successful=198
            ),
            router=make_mock_router(
                reroutes=2, recoveries=2, active_nodes=3
            ),
            retrainer=make_mock_retrainer(
                total=5, successful=5
            ),
            orchestrator=make_mock_orchestrator(
                total=2, successful=2
            ),
            collection_interval=60.0,
        )
        snap = collector._collect_snapshot()
        d = snap.to_dict()

        assert d["causal_engine"]["ready"] is True
        assert d["causal_engine"]["buffer_size"] == 60
        assert d["causal_engine"]["retrain_count"] == 10
        assert d["predictive_stack"]["simulations_run"] == 20
        assert d["predictive_stack"]["alerts_fired_total"] == 50
        assert d["healing_engine"]["actions_total"] == 200
        assert d["router"]["total_reroutes"] == 2
        assert d["router"]["active_nodes"] == 3
        assert d["retrainer"]["total"] == 5
        assert d["orchestrator"]["total_sequences"] == 2


# ── Benchmarks ──────────────────────────────────────────────

class TestObservabilityBenchmarks:

    def test_snapshot_collection_latency(self):
        updater = make_mock_updater()
        collector = MetricsCollector(
            updater=updater,
            analyzer=make_mock_analyzer(),
            simulator=make_mock_simulator(),
            alerter=make_mock_alerter(),
            engine=make_mock_engine(),
            router=make_mock_router(),
            retrainer=make_mock_retrainer(),
            orchestrator=make_mock_orchestrator(),
            collection_interval=60.0,
        )
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            collector._collect_snapshot()
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 50.0

    def test_causal_adapter_collection_latency(self):
        updater = make_mock_updater()
        registry = CollectorRegistry()
        adapter = CausalMetricsAdapter(
            updater=updater,
            registry=registry,
            collection_interval=60.0,
        )
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            adapter.collect()
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 50.0

    def test_healing_adapter_collection_latency(self):
        engine = make_mock_engine()
        router = make_mock_router()
        retrainer = make_mock_retrainer()
        orchestrator = make_mock_orchestrator()
        registry = CollectorRegistry()
        adapter = HealingMetricsAdapter(
            engine=engine,
            router=router,
            retrainer=retrainer,
            orchestrator=orchestrator,
            registry=registry,
            collection_interval=60.0,
        )
        adapter._start_time = time.time()
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            adapter.collect()
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 50.0

    def test_stability_report_latency(self):
        updater = make_mock_updater()
        registry = CollectorRegistry()
        adapter = CausalMetricsAdapter(
            updater=updater,
            registry=registry,
            collection_interval=60.0,
        )
        for _ in range(30):
            adapter.collect()
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            adapter.get_stability_report()
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 5.0

    def test_prometheus_generate_latency(self):
        updater = make_mock_updater()
        registry = CollectorRegistry()
        adapter = CausalMetricsAdapter(
            updater=updater,
            registry=registry,
            collection_interval=60.0,
        )
        adapter.collect()
        from prometheus_client import generate_latest
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            generate_latest(registry)
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 10.0