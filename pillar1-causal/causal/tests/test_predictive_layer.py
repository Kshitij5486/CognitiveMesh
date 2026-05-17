import sys
import os
import time
import statistics
import threading
from unittest import result
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "telemetry"))

from load_trend_analyzer import (
    LoadTrendAnalyzer, LoadObservation, TrendAnalysis,
    NodeLoadTracker, TrendDirection, TrendSeverity
)
from causal_simulator import (
    CausalSimulator, SimulationScenario, NodeSimulation,
    ClusterSimulation
)
from predictive_alerter import (
    PredictiveAlerter, PredictiveAlert, AlertType, AlertSeverity
)
from predictive_byzantine_bridge import ByzantinePredictiveBridge


def make_mock_updater(ready=True, effect=28.5):
    updater = MagicMock()
    updater.status.return_value = {
        "is_ready": ready,
        "buffer_size": 60,
        "retrain_count": 3,
        "nodes_modeled": ["node-1", "node-2", "node-3"],
    }
    def get_snapshot(node_id):
        return {
            "effect": effect,
            "samples_used": 60,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    updater.get_current_snapshot.side_effect = get_snapshot
    updater.get_all_snapshots.return_value = {
        n: get_snapshot(n) for n in ["node-1", "node-2", "node-3"]
    }
    return updater


def make_mock_analyzer(
    direction="stable",
    load=5.0,
    rate=0.0,
    severity="normal",
):
    analyzer = MagicMock()
    analysis = {
        "node_id": "node-1",
        "direction": direction,
        "severity": severity,
        "current_load": load,
        "baseline_load": load,
        "change_rate_per_minute": rate,
        "projected_load_5min": max(0.0, load + rate * 5),
        "projected_latency_5min": abs(28.5) * max(0.0, load + rate * 5),
        "causal_effect_ms": 28.5,
        "observations_used": 30,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "interpretation": "mock interpretation",
    }

    def get_latest():
        return {n: dict(analysis, node_id=n)
                for n in ["node-1", "node-2", "node-3"]}

    def get_analysis(node_id):
        return dict(analysis, node_id=node_id)

    def get_cluster():
        return {
            "cluster_trend": direction,
            "severity": severity,
            "nodes_analyzed": 3,
            "avg_projected_latency_5min": analysis["projected_latency_5min"],
            "timestamp": "2026-01-01T00:00:00+00:00",
        }

    analyzer.get_latest_analyses.side_effect = get_latest
    analyzer.get_analysis.side_effect = get_analysis
    analyzer.get_cluster_trend.side_effect = get_cluster
    analyzer.status.return_value = {
        "running": True,
        "observations_collected": 30,
        "analyses_run": 5,
        "nodes_tracked": ["node-1", "node-2", "node-3"],
        "tracker_sizes": {"node-1": 30, "node-2": 30, "node-3": 30},
        "latest_analyses_available": ["node-1", "node-2", "node-3"],
    }
    return analyzer


def make_mock_simulator(
    worst_latency=50.0,
    best_latency=10.0,
    has_sim=True,
):
    simulator = MagicMock()

    if has_sim:
        node_sim = {
            "node_id": "node-1",
            "current_load": 5.0,
            "causal_effect_ms": 28.5,
            "current_latency_ms": 142.5,
            "trend_direction": "stable",
            "change_rate_per_minute": 0.0,
            "worst_case_latency_ms": worst_latency,
            "best_case_latency_ms": best_latency,
            "scenarios": [
                {
                    "node_id": "node-1",
                    "scenario_name": "trend_1min",
                    "query_load": 5.0,
                    "causal_effect_ms": 28.5,
                    "projected_latency_ms": worst_latency,
                    "horizon_minutes": 1.0,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
                {
                    "node_id": "node-1",
                    "scenario_name": "trend_5min",
                    "query_load": 5.0,
                    "causal_effect_ms": 28.5,
                    "projected_latency_ms": worst_latency,
                    "horizon_minutes": 5.0,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
                {
                    "node_id": "node-1",
                    "scenario_name": "load_2.0x",
                    "query_load": 10.0,
                    "causal_effect_ms": 28.5,
                    "projected_latency_ms": worst_latency * 2,
                    "horizon_minutes": 5.0,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
            ],
            "timestamp": "2026-01-01T00:00:00+00:00",
        }

        cluster_sim = {
            "simulation_horizon_minutes": 15,
            "cluster_worst_case_latency_ms": worst_latency,
            "cluster_best_case_latency_ms": best_latency,
            "highest_risk_node": "node-1",
            "node_simulations": {
                n: dict(node_sim, node_id=n)
                for n in ["node-1", "node-2", "node-3"]
            },
            "timestamp": "2026-01-01T00:00:00+00:00",
        }

        simulator.get_latest_simulation.return_value = cluster_sim
        simulator.simulate_now.return_value = cluster_sim
        simulator.get_node_simulation.return_value = node_sim
    else:
        simulator.get_latest_simulation.return_value = None
        simulator.simulate_now.return_value = None
        simulator.get_node_simulation.return_value = None

    simulator.status.return_value = {
        "running": True,
        "simulations_run": 3,
        "has_simulation": has_sim,
        "highest_risk_node": "node-1" if has_sim else None,
        "cluster_worst_case_latency_ms": worst_latency if has_sim else 0.0,
        "simulation_horizons_minutes": [1, 5, 15],
        "load_multipliers": [1.0, 1.5, 2.0, 3.0],
        "warn_threshold_ms": 200.0,
        "critical_threshold_ms": 500.0,
    }
    return simulator


class TestNodeLoadTracker:

    def test_tracker_starts_empty(self):
        tracker = NodeLoadTracker("node-1")
        assert tracker.size() == 0

    def test_add_observation(self):
        tracker = NodeLoadTracker("node-1")
        obs = LoadObservation(
            node_id="node-1",
            active_queries=5.0,
            avg_latency_ms=100.0,
            causal_effect_ms=28.5,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        tracker.add_observation(obs)
        assert tracker.size() == 1

    def test_insufficient_observations_returns_none(self):
        tracker = NodeLoadTracker("node-1")
        for i in range(4):
            obs = LoadObservation(
                node_id="node-1",
                active_queries=float(i),
                avg_latency_ms=100.0,
                causal_effect_ms=28.5,
                timestamp="2026-01-01T00:00:00+00:00",
            )
            tracker.add_observation(obs)
        result = tracker.analyze(causal_effect_ms=28.5)
        assert result is None

    def test_sufficient_observations_returns_analysis(self):
        tracker = NodeLoadTracker("node-1")
        for i in range(10):
            obs = LoadObservation(
                node_id="node-1",
                active_queries=5.0,
                avg_latency_ms=100.0,
                causal_effect_ms=28.5,
                timestamp="2026-01-01T00:00:00+00:00",
            )
            tracker.add_observation(obs)
        result = tracker.analyze(causal_effect_ms=28.5)
        assert result is not None

    def test_rising_trend_detected(self):
        tracker = NodeLoadTracker("node-1")
        for i in range(10):
            obs = LoadObservation(
                node_id="node-1",
                active_queries=float(i * 2),
                avg_latency_ms=100.0,
                causal_effect_ms=28.5,
                timestamp="2026-01-01T00:00:00+00:00",
            )
            tracker.add_observation(obs)
        result = tracker.analyze(causal_effect_ms=28.5)
        assert result is not None
        assert result.direction == TrendDirection.RISING

    def test_falling_trend_detected(self):
        tracker = NodeLoadTracker("node-1")
        for i in range(10):
            obs = LoadObservation(
                node_id="node-1",
                active_queries=float(20 - i * 2),
                avg_latency_ms=100.0,
                causal_effect_ms=28.5,
                timestamp="2026-01-01T00:00:00+00:00",
            )
            tracker.add_observation(obs)
        result = tracker.analyze(causal_effect_ms=28.5)
        assert result is not None
        assert result.direction == TrendDirection.FALLING

    def test_stable_trend_detected(self):
        tracker = NodeLoadTracker("node-1")
        for _ in range(10):
            obs = LoadObservation(
                node_id="node-1",
                active_queries=5.0,
                avg_latency_ms=100.0,
                causal_effect_ms=28.5,
                timestamp="2026-01-01T00:00:00+00:00",
            )
            tracker.add_observation(obs)
        result = tracker.analyze(causal_effect_ms=28.5)
        assert result is not None
        assert result.direction == TrendDirection.STABLE

    def test_max_observations_capped(self):
        tracker = NodeLoadTracker("node-1")
        for i in range(80):
            obs = LoadObservation(
                node_id="node-1",
                active_queries=float(i % 10),
                avg_latency_ms=100.0,
                causal_effect_ms=28.5,
                timestamp="2026-01-01T00:00:00+00:00",
            )
            tracker.add_observation(obs)
        assert tracker.size() == NodeLoadTracker.MAX_OBSERVATIONS

    def test_critical_severity_on_large_change(self):
        tracker = NodeLoadTracker("node-1")
        for i in range(10):
            obs = LoadObservation(
                node_id="node-1",
                active_queries=float(i * 5),
                avg_latency_ms=100.0,
                causal_effect_ms=28.5,
                timestamp="2026-01-01T00:00:00+00:00",
            )
            tracker.add_observation(obs)
        result = tracker.analyze(causal_effect_ms=28.5)
        assert result is not None
        assert result.severity in (
            TrendSeverity.ELEVATED,
            TrendSeverity.HIGH,
            TrendSeverity.CRITICAL,
        )

    def test_thread_safety(self):
        tracker = NodeLoadTracker("node-1")
        errors = []

        def add_observations():
            try:
                for i in range(20):
                    obs = LoadObservation(
                        node_id="node-1",
                        active_queries=float(i),
                        avg_latency_ms=100.0,
                        causal_effect_ms=28.5,
                        timestamp="2026-01-01T00:00:00+00:00",
                    )
                    tracker.add_observation(obs)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_observations)
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestLoadObservation:

    def test_observation_creation(self):
        obs = LoadObservation(
            node_id="node-1",
            active_queries=5.0,
            avg_latency_ms=142.5,
            causal_effect_ms=28.5,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert obs.node_id == "node-1"
        assert obs.active_queries == 5.0
        assert obs.causal_effect_ms == 28.5

    def test_observation_to_dict(self):
        obs = LoadObservation(
            node_id="node-2",
            active_queries=3.0,
            avg_latency_ms=85.5,
            causal_effect_ms=28.5,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        d = obs.to_dict()
        assert "node_id" in d
        assert "active_queries" in d
        assert "causal_effect_ms" in d


class TestSimulationScenario:

    def test_scenario_creation(self):
        scenario = SimulationScenario(
            node_id="node-1",
            scenario_name="trend_5min",
            query_load=10.0,
            causal_effect_ms=28.5,
            projected_latency_ms=285.0,
            horizon_minutes=5.0,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert scenario.node_id == "node-1"
        assert scenario.projected_latency_ms == 285.0

    def test_scenario_to_dict(self):
        scenario = SimulationScenario(
            node_id="node-1",
            scenario_name="load_2.0x",
            query_load=10.0,
            causal_effect_ms=28.5,
            projected_latency_ms=285.0,
            horizon_minutes=5.0,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        d = scenario.to_dict()
        assert "scenario_name" in d
        assert "projected_latency_ms" in d
        assert "horizon_minutes" in d


class TestNodeSimulation:

    def _make_scenarios(self, latencies):
        return [
            SimulationScenario(
                node_id="node-1",
                scenario_name=f"scenario_{i}",
                query_load=5.0,
                causal_effect_ms=28.5,
                projected_latency_ms=lat,
                horizon_minutes=float(i + 1),
                timestamp="2026-01-01T00:00:00+00:00",
            )
            for i, lat in enumerate(latencies)
        ]

    def test_worst_case_latency(self):
        scenarios = self._make_scenarios([100.0, 200.0, 300.0])
        sim = NodeSimulation(
            node_id="node-1",
            current_load=5.0,
            causal_effect_ms=28.5,
            current_latency_ms=142.5,
            trend_direction="stable",
            change_rate_per_minute=0.0,
            scenarios=scenarios,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert sim.worst_case_latency() == 300.0

    def test_best_case_latency(self):
        scenarios = self._make_scenarios([100.0, 200.0, 300.0])
        sim = NodeSimulation(
            node_id="node-1",
            current_load=5.0,
            causal_effect_ms=28.5,
            current_latency_ms=142.5,
            trend_direction="stable",
            change_rate_per_minute=0.0,
            scenarios=scenarios,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert sim.best_case_latency() == 100.0

    def test_to_dict_structure(self):
        scenarios = self._make_scenarios([100.0])
        sim = NodeSimulation(
            node_id="node-1",
            current_load=5.0,
            causal_effect_ms=28.5,
            current_latency_ms=142.5,
            trend_direction="rising",
            change_rate_per_minute=1.5,
            scenarios=scenarios,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        d = sim.to_dict()
        assert "node_id" in d
        assert "worst_case_latency_ms" in d
        assert "best_case_latency_ms" in d
        assert "scenarios" in d


class TestCausalSimulator:

    def test_simulator_initializes(self):
        updater = make_mock_updater()
        analyzer = make_mock_analyzer()
        sim = CausalSimulator(
            updater=updater,
            analyzer=analyzer,
            simulation_interval_seconds=30.0,
        )
        assert sim is not None

    def test_simulate_now_with_trend_data(self):
        updater = make_mock_updater(effect=28.5)
        analyzer = make_mock_analyzer(load=5.0, rate=1.0)
        sim = CausalSimulator(
            updater=updater,
            analyzer=analyzer,
            simulation_interval_seconds=30.0,
        )
        result = sim.simulate_now()
        assert result is not None
        assert "node_simulations" in result
        assert len(result["node_simulations"]) == 3

    def test_simulation_has_correct_scenarios(self):
        updater = make_mock_updater(effect=28.5)
        analyzer = make_mock_analyzer(load=5.0, rate=0.0)
        sim = CausalSimulator(
            updater=updater,
            analyzer=analyzer,
            simulation_interval_seconds=30.0,
        )
        result = sim.simulate_now()
        assert result is not None
        node_sim = result["node_simulations"]["node-1"]
        scenario_names = [s["scenario_name"] for s in node_sim["scenarios"]]
        assert "trend_1min" in scenario_names
        assert "trend_5min" in scenario_names
        assert "trend_15min" in scenario_names
        assert "load_1.0x" in scenario_names
        assert "load_2.0x" in scenario_names

    def test_simulation_not_ready_when_engine_not_ready(self):
        updater = make_mock_updater(ready=False)
        analyzer = make_mock_analyzer()
        sim = CausalSimulator(
            updater=updater,
            analyzer=analyzer,
            simulation_interval_seconds=30.0,
        )
        result = sim.simulate_now()
        assert result is None

    def test_simulation_not_ready_when_engine_not_ready_no_trends(self):
        updater = make_mock_updater(ready=False)
        analyzer = make_mock_analyzer()
        analyzer.get_latest_analyses.return_value = {}
        sim = CausalSimulator(
            updater=updater,
            analyzer=analyzer,
            simulation_interval_seconds=30.0,
        )
        result = sim.simulate_now()
        assert result is None

    def test_highest_risk_node_identified(self):
        updater = make_mock_updater(effect=28.5)
        analyzer = make_mock_analyzer(load=10.0, rate=2.0)
        sim = CausalSimulator(
            updater=updater,
            analyzer=analyzer,
            simulation_interval_seconds=30.0,
        )
        result = sim.simulate_now()
        assert result is not None
        assert result["highest_risk_node"] is not None

    def test_cluster_worst_case_latency(self):
        updater = make_mock_updater(effect=28.5)
        analyzer = make_mock_analyzer(load=5.0, rate=1.0)
        sim = CausalSimulator(
            updater=updater,
            analyzer=analyzer,
            simulation_interval_seconds=30.0,
        )
        result = sim.simulate_now()
        assert result is not None
        assert result["cluster_worst_case_latency_ms"] >= 0.0

    def test_status_structure(self):
        updater = make_mock_updater()
        analyzer = make_mock_analyzer()
        sim = CausalSimulator(
            updater=updater,
            analyzer=analyzer,
            simulation_interval_seconds=30.0,
        )
        status = sim.status()
        assert "running" in status
        assert "simulations_run" in status
        assert "has_simulation" in status
        assert "simulation_horizons_minutes" in status


class TestPredictiveAlerter:

    def make_alerter(
        self,
        worst_latency=50.0,
        best_latency=10.0,
        has_sim=True,
        load=5.0,
        rate=0.0,
    ):
        updater = make_mock_updater()
        analyzer = make_mock_analyzer(load=load, rate=rate)
        simulator = make_mock_simulator(
            worst_latency=worst_latency,
            best_latency=best_latency,
            has_sim=has_sim,
        )
        alerter = PredictiveAlerter(
            updater=updater,
            analyzer=analyzer,
            simulator=simulator,
            check_interval=60.0,
        )
        return alerter

    def test_alerter_initializes(self):
        alerter = self.make_alerter()
        assert alerter is not None
        assert alerter._checks_run == 0
        assert alerter._total_alerts_fired == 0

    def test_no_alerts_when_latency_low(self):
        alerter = self.make_alerter(worst_latency=50.0)
        alerter._check_cycle()
        assert alerter._total_alerts_fired == 0

    def test_warning_alert_on_warn_threshold(self):
        alerter = self.make_alerter(worst_latency=160.0)
        alerter._check_cycle()
        active = alerter.get_active_alerts()
        warn_alerts = [
            a for a in active
            if a["severity"] == "warning"
            and a["alert_type"] == "latency_rising"
        ]
        assert len(warn_alerts) > 0

    def test_critical_alert_on_critical_threshold(self):
        alerter = self.make_alerter(worst_latency=350.0)
        alerter._check_cycle()
        active = alerter.get_active_alerts()
        critical_alerts = [
            a for a in active
            if a["severity"] == "critical"
        ]
        assert len(critical_alerts) > 0

    def test_trend_acceleration_alert(self):
        alerter = self.make_alerter(load=5.0, rate=8.0)
        alerter._check_cycle()
        active = alerter.get_active_alerts()
        accel_alerts = [
            a for a in active
            if a["alert_type"] == "trend_acceleration"
        ]
        assert len(accel_alerts) > 0

    def test_no_trend_alert_below_threshold(self):
        alerter = self.make_alerter(load=5.0, rate=2.0)
        alerter._check_cycle()
        active = alerter.get_active_alerts()
        accel_alerts = [
            a for a in active
            if a["alert_type"] == "trend_acceleration"
        ]
        assert len(accel_alerts) == 0

    def test_acknowledge_alert(self):
        alerter = self.make_alerter(worst_latency=160.0)
        alerter._check_cycle()
        active = alerter.get_active_alerts()
        assert len(active) > 0
        alert_id = active[0]["alert_id"]
        success = alerter.acknowledge_alert(alert_id)
        assert success is True
        active2 = alerter.get_active_alerts()
        ack = [a for a in active2 if a["alert_id"] == alert_id]
        assert ack[0]["acknowledged"] is True

    def test_clear_alert(self):
        alerter = self.make_alerter(worst_latency=160.0)
        alerter._check_cycle()
        active = alerter.get_active_alerts()
        assert len(active) > 0
        alert_id = active[0]["alert_id"]
        success = alerter.clear_alert(alert_id)
        assert success is True
        active2 = alerter.get_active_alerts()
        remaining = [a for a in active2 if a["alert_id"] == alert_id]
        assert len(remaining) == 0

    def test_alert_history_populated(self):
        alerter = self.make_alerter(worst_latency=160.0)
        alerter._check_cycle()
        history = alerter.get_alert_history()
        assert len(history) > 0

    def test_status_structure(self):
        alerter = self.make_alerter()
        status = alerter.status()
        assert "running" in status
        assert "checks_run" in status
        assert "total_alerts_fired" in status
        assert "active_alert_count" in status
        assert "thresholds" in status

    def test_no_simulation_skips_check(self):
        alerter = self.make_alerter(has_sim=False)
        alerter._check_cycle()
        assert alerter._total_alerts_fired == 0


class TestByzantinePredictiveBridge:

    def make_bridge(self):
        updater = make_mock_updater()
        analyzer = make_mock_analyzer()
        simulator = make_mock_simulator()
        alerter = PredictiveAlerter(
            updater=updater,
            analyzer=analyzer,
            simulator=simulator,
            check_interval=60.0,
        )
        bridge = ByzantinePredictiveBridge(
            alerter=alerter,
            poll_interval=60.0,
        )
        return bridge, alerter

    def test_bridge_initializes(self):
        bridge, _ = self.make_bridge()
        assert bridge is not None
        assert bridge._byzantine_api_available is False

    def test_all_nodes_trusted_when_api_unavailable(self):
        bridge, _ = self.make_bridge()
        trusted = bridge.get_trusted_nodes()
        assert set(trusted) == {"node-1", "node-2", "node-3"}

    def test_node_trusted_when_api_unavailable(self):
        bridge, _ = self.make_bridge()
        assert bridge.is_node_trusted("node-1") is True
        assert bridge.is_node_trusted("node-2") is True

    def test_enrich_simulation_without_api(self):
        bridge, _ = self.make_bridge()
        sim = {"cluster_worst_case_latency_ms": 100.0}
        enriched = bridge.enrich_simulation(sim)
        assert "byzantine_context" in enriched
        assert enriched["byzantine_context"]["available"] is False

    def test_enrich_simulation_with_api(self):
        bridge, _ = self.make_bridge()
        with bridge._lock:
            bridge._byzantine_api_available = True
            bridge._active_nodes = ["node-1", "node-2"]
            bridge._isolated_nodes = ["node-3"]
            bridge._cluster_operational = True
            bridge._node_reputations = {
                "node-1": {
                    "reputation_score": 0.85,
                    "reputation_status": "trusted",
                    "isolated": False,
                },
                "node-2": {
                    "reputation_score": 0.71,
                    "reputation_status": "trusted",
                    "isolated": False,
                },
                "node-3": {
                    "reputation_score": 0.05,
                    "reputation_status": "byzantine",
                    "isolated": True,
                },
            }
        sim = {"cluster_worst_case_latency_ms": 100.0}
        enriched = bridge.enrich_simulation(sim)
        ctx = enriched["byzantine_context"]
        assert ctx["available"] is True
        assert "node-3" in ctx["isolated_nodes"]
        assert ctx["node_trust"]["node-3"]["isolated"] is True
        assert ctx["node_trust"]["node-1"]["trusted"] is True

    def test_isolated_node_not_trusted(self):
        bridge, _ = self.make_bridge()
        with bridge._lock:
            bridge._byzantine_api_available = True
            bridge._isolated_nodes = ["node-1"]
            bridge._active_nodes = ["node-2", "node-3"]
        assert bridge.is_node_trusted("node-1") is False
        assert bridge.is_node_trusted("node-2") is True

    def test_reputation_alert_fired_for_low_score(self):
        bridge, alerter = self.make_bridge()
        nodes = {
            "node-1": {
                "reputation_score": 0.10,
                "reputation_status": "byzantine",
                "isolated": False,
            }
        }
        bridge._check_reputation_alerts(nodes)
        assert alerter._total_alerts_fired > 0

    def test_no_alert_for_healthy_reputation(self):
        bridge, alerter = self.make_bridge()
        nodes = {
            "node-1": {
                "reputation_score": 0.85,
                "reputation_status": "trusted",
                "isolated": False,
            }
        }
        bridge._check_reputation_alerts(nodes)
        assert alerter._total_alerts_fired == 0

    def test_status_structure(self):
        bridge, _ = self.make_bridge()
        status = bridge.status()
        assert "running" in status
        assert "byzantine_api_available" in status
        assert "poll_count" in status
        assert "active_nodes" in status
        assert "isolated_nodes" in status


class TestPredictiveLayerBenchmarks:

    def test_observation_add_latency(self):
        tracker = NodeLoadTracker("node-1")
        latencies = []
        for i in range(100):
            obs = LoadObservation(
                node_id="node-1",
                active_queries=float(i % 10),
                avg_latency_ms=100.0,
                causal_effect_ms=28.5,
                timestamp="2026-01-01T00:00:00+00:00",
            )
            start = time.perf_counter()
            tracker.add_observation(obs)
            latencies.append((time.perf_counter() - start) * 1000)
        assert statistics.mean(latencies) < 1.0

    def test_trend_analysis_latency(self):
        tracker = NodeLoadTracker("node-1")
        for i in range(30):
            obs = LoadObservation(
                node_id="node-1",
                active_queries=float(i % 10),
                avg_latency_ms=100.0,
                causal_effect_ms=28.5,
                timestamp="2026-01-01T00:00:00+00:00",
            )
            tracker.add_observation(obs)
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            tracker.analyze(causal_effect_ms=28.5)
            latencies.append((time.perf_counter() - start) * 1000)
        assert statistics.mean(latencies) < 5.0

    def test_simulation_latency(self):
        updater = make_mock_updater(effect=28.5)
        analyzer = make_mock_analyzer(load=5.0, rate=1.0)
        sim = CausalSimulator(
            updater=updater,
            analyzer=analyzer,
            simulation_interval_seconds=30.0,
        )
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            sim.simulate_now()
            latencies.append((time.perf_counter() - start) * 1000)
        assert statistics.mean(latencies) < 50.0

    def test_alert_check_latency(self):
        updater = make_mock_updater()
        analyzer = make_mock_analyzer(load=5.0, rate=0.0)
        simulator = make_mock_simulator(worst_latency=50.0)
        alerter = PredictiveAlerter(
            updater=updater,
            analyzer=analyzer,
            simulator=simulator,
            check_interval=60.0,
        )
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            alerter._check_cycle()
            latencies.append((time.perf_counter() - start) * 1000)
        assert statistics.mean(latencies) < 20.0

    def test_bridge_enrich_latency(self):
        bridge, _ = TestByzantinePredictiveBridge().make_bridge()
        sim = {"cluster_worst_case_latency_ms": 100.0}
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            bridge.enrich_simulation(dict(sim))
            latencies.append((time.perf_counter() - start) * 1000)
        assert statistics.mean(latencies) < 1.0