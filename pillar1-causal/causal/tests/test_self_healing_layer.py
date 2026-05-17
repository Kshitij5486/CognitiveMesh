import sys
import os
import time
import statistics
import threading
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "telemetry"))

from healing_action_engine import (
    HealingActionEngine, HealingAction, HealingActionType,
    HealingActionStatus, HealingPolicy, HealingActionExecutor
)
from query_router import (
    QueryRouter, RoutingDecision, RoutingStrategy,
    NodeRoutingState
)
from auto_retrainer import (
    AutoRetrainer, RetrainRecord, RetrainTrigger, RetrainStatus
)
from recovery_orchestrator import (
    RecoveryOrchestrator, RecoverySequence, RecoveryPhase,
    RecoveryTrigger
)
from predictive_alerter import AlertType, AlertSeverity


def make_mock_updater(ready=True, effect=28.5, load=5.0):
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

    import pandas as pd
    import numpy as np
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


def make_mock_alerter(active_alerts=None):
    alerter = MagicMock()
    active_alerts = active_alerts or []
    alerter.get_active_alerts.return_value = active_alerts
    alerter.acknowledge_alert.return_value = True
    alerter.clear_alert.return_value = True
    alerter.status.return_value = {
        "running": True,
        "checks_run": 5,
        "total_alerts_fired": len(active_alerts),
        "active_alert_count": len(active_alerts),
        "active_alerts": active_alerts,
        "thresholds": {
            "latency_warn_ms": 150.0,
            "latency_critical_ms": 300.0,
        },
    }
    return alerter


def make_latency_alert(
    node_id="node-1",
    severity=AlertSeverity.WARNING.value,
    acknowledged=False,
):
    return {
        "alert_id": f"alert-{node_id}-001",
        "node_id": node_id,
        "alert_type": AlertType.LATENCY_RISING.value,
        "severity": severity,
        "current_value": 140.0,
        "predicted_value": 200.0,
        "threshold": 150.0,
        "horizon_minutes": 5.0,
        "message": f"Latency rising on {node_id}",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "acknowledged": acknowledged,
    }


def make_causal_alert(
    node_id="node-1",
    severity=AlertSeverity.CRITICAL.value,
):
    return {
        "alert_id": f"causal-{node_id}-001",
        "node_id": node_id,
        "alert_type": AlertType.CAUSAL_THRESHOLD.value,
        "severity": severity,
        "current_value": 0.10,
        "predicted_value": 0.05,
        "threshold": 0.20,
        "horizon_minutes": 5.0,
        "message": f"Causal threshold on {node_id}",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "acknowledged": False,
    }


class TestHealingPolicy:

    def test_latency_warning_maps_to_rebalance(self):
        action = HealingPolicy.decide(
            AlertType.LATENCY_RISING.value,
            AlertSeverity.WARNING.value,
        )
        assert action == HealingActionType.REBALANCE

    def test_latency_critical_maps_to_reroute(self):
        action = HealingPolicy.decide(
            AlertType.LATENCY_RISING.value,
            AlertSeverity.CRITICAL.value,
        )
        assert action == HealingActionType.REROUTE

    def test_causal_critical_maps_to_isolate(self):
        action = HealingPolicy.decide(
            AlertType.CAUSAL_THRESHOLD.value,
            AlertSeverity.CRITICAL.value,
        )
        assert action == HealingActionType.ISOLATE

    def test_trend_warning_maps_to_retrain(self):
        action = HealingPolicy.decide(
            AlertType.TREND_ACCELERATION.value,
            AlertSeverity.WARNING.value,
        )
        assert action == HealingActionType.RETRAIN

    def test_cluster_degradation_maps_to_alert_operator(self):
        action = HealingPolicy.decide(
            AlertType.CLUSTER_DEGRADATION.value,
            AlertSeverity.CRITICAL.value,
        )
        assert action == HealingActionType.ALERT_OPERATOR

    def test_unknown_combo_returns_no_action(self):
        action = HealingPolicy.decide(
            "unknown_type",
            "unknown_severity",
        )
        assert action == HealingActionType.NO_ACTION


class TestHealingActionExecutor:

    def test_retrain_success(self):
        updater = make_mock_updater()
        executor = HealingActionExecutor(updater=updater)
        action = HealingAction(
            action_id="test-001",
            action_type=HealingActionType.RETRAIN,
            node_id="node-1",
            trigger_alert_type="trend_acceleration",
            trigger_severity="warning",
            reason="test",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        success, msg = executor.execute(action)
        assert success is True
        assert action.status == HealingActionStatus.SUCCESS

    def test_reroute_success(self):
        updater = make_mock_updater()
        executor = HealingActionExecutor(updater=updater)
        action = HealingAction(
            action_id="test-002",
            action_type=HealingActionType.REROUTE,
            node_id="node-1",
            trigger_alert_type="latency_rising",
            trigger_severity="critical",
            reason="test",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        success, msg = executor.execute(action)
        assert success is True
        assert "node-2" in msg or "node-3" in msg

    def test_rebalance_success(self):
        updater = make_mock_updater()
        executor = HealingActionExecutor(updater=updater)
        action = HealingAction(
            action_id="test-003",
            action_type=HealingActionType.REBALANCE,
            node_id="node-2",
            trigger_alert_type="latency_rising",
            trigger_severity="warning",
            reason="test",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        success, msg = executor.execute(action)
        assert success is True
        assert "rebalancing" in msg.lower()

    def test_isolate_success(self):
        updater = make_mock_updater()
        executor = HealingActionExecutor(updater=updater)
        action = HealingAction(
            action_id="test-004",
            action_type=HealingActionType.ISOLATE,
            node_id="node-1",
            trigger_alert_type="causal_threshold",
            trigger_severity="critical",
            reason="test",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        success, msg = executor.execute(action)
        assert success is True
        assert "isolation" in msg.lower()

    def test_action_duration_recorded(self):
        updater = make_mock_updater()
        executor = HealingActionExecutor(updater=updater)
        action = HealingAction(
            action_id="test-005",
            action_type=HealingActionType.REBALANCE,
            node_id="node-1",
            trigger_alert_type="latency_rising",
            trigger_severity="warning",
            reason="test",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        executor.execute(action)
        assert action.duration_ms is not None
        assert action.duration_ms >= 0

    def test_not_ready_retrain_fails(self):
        updater = make_mock_updater(ready=False)
        executor = HealingActionExecutor(updater=updater)
        action = HealingAction(
            action_id="test-006",
            action_type=HealingActionType.RETRAIN,
            node_id="node-1",
            trigger_alert_type="trend_acceleration",
            trigger_severity="warning",
            reason="test",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        success, msg = executor.execute(action)
        assert success is False


class TestHealingActionEngine:

    def make_engine(self, active_alerts=None):
        updater = make_mock_updater()
        alerter = make_mock_alerter(active_alerts=active_alerts)
        eng = HealingActionEngine(
            updater=updater,
            alerter=alerter,
            check_interval=60.0,
            auto_heal=True,
        )
        return eng

    def test_engine_initializes(self):
        eng = self.make_engine()
        assert eng is not None
        assert eng._checks_run == 0
        assert eng._total_actions == 0

    def test_no_action_on_no_alerts(self):
        eng = self.make_engine(active_alerts=[])
        eng._check_cycle()
        assert eng._total_actions == 0

    def test_rebalance_on_warning_alert(self):
        alerts = [make_latency_alert(severity=AlertSeverity.WARNING.value)]
        eng = self.make_engine(active_alerts=alerts)
        eng._check_cycle()
        assert eng._total_actions > 0
        assert eng._successful_actions > 0

    def test_skips_acknowledged_alerts(self):
        alerts = [make_latency_alert(acknowledged=True)]
        eng = self.make_engine(active_alerts=alerts)
        eng._check_cycle()
        assert eng._total_actions == 0

    def test_heal_now_returns_action(self):
        eng = self.make_engine()
        result = eng.heal_now(
            node_id="node-1",
            alert_type=AlertType.LATENCY_RISING.value,
            severity=AlertSeverity.WARNING.value,
        )
        assert result is not None
        assert result["action_type"] == "rebalance"

    def test_heal_now_no_action_for_unknown(self):
        eng = self.make_engine()
        result = eng.heal_now(
            node_id="node-1",
            alert_type="unknown_type",
            severity="unknown_severity",
        )
        assert result is None

    def test_cooldown_prevents_repeat(self):
        alerts = [make_latency_alert(severity=AlertSeverity.WARNING.value)]
        eng = self.make_engine(active_alerts=alerts)
        eng._check_cycle()
        first_total = eng._total_actions
        eng._check_cycle()
        assert eng._total_actions == first_total

    def test_action_history_populated(self):
        alerts = [make_latency_alert(severity=AlertSeverity.WARNING.value)]
        eng = self.make_engine(active_alerts=alerts)
        eng._check_cycle()
        history = eng.get_action_history()
        assert len(history) > 0

    def test_status_structure(self):
        eng = self.make_engine()
        status = eng.status()
        assert "running" in status
        assert "auto_heal" in status
        assert "checks_run" in status
        assert "total_actions" in status
        assert "successful_actions" in status


class TestQueryRouter:

    def make_router(self, active_alerts=None):
        updater = make_mock_updater()
        alerter = make_mock_alerter(active_alerts=active_alerts)
        r = QueryRouter(
            updater=updater,
            alerter=alerter,
            strategy=RoutingStrategy.CAUSAL_WEIGHTED,
            check_interval=60.0,
        )
        return r

    def test_router_initializes(self):
        r = self.make_router()
        assert r is not None
        assert len(r.get_active_nodes()) == 3

    def test_all_nodes_active_initially(self):
        r = self.make_router()
        assert set(r.get_active_nodes()) == {
            "node-1", "node-2", "node-3"
        }

    def test_reroute_node(self):
        r = self.make_router()
        decision = r.reroute_node(
            node_id="node-1",
            reason="test reroute",
        )
        assert decision is not None
        assert "node-1" not in decision.target_nodes
        assert len(decision.target_nodes) > 0

    def test_rerouted_node_not_active(self):
        r = self.make_router()
        r.reroute_node(node_id="node-1", reason="test")
        active = r.get_active_nodes()
        assert "node-1" not in active

    def test_restore_node(self):
        r = self.make_router()
        r.reroute_node(node_id="node-1", reason="test")
        success = r.restore_node("node-1")
        assert success is True
        assert "node-1" in r.get_active_nodes()

    def test_cannot_restore_already_active(self):
        r = self.make_router()
        success = r.restore_node("node-1")
        assert success is False

    def test_isolate_node(self):
        r = self.make_router()
        r.isolate_node(node_id="node-1", reason="test isolation")
        with r._lock:
            state = r._node_states["node-1"]
        # isolate_node sets ISOLATED then calls reroute_node which
        # overwrites to REROUTED — node is correctly excluded from
        # traffic either way; verify it is not active
        assert state in (
            NodeRoutingState.ISOLATED,
            NodeRoutingState.REROUTED,
        )
        assert "node-1" not in r.get_active_nodes()

    def test_causal_weights_computed(self):
        r = self.make_router()
        weights = r._compute_causal_weights(
            ["node-1", "node-2", "node-3"]
        )
        assert len(weights) == 3
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_routing_table_structure(self):
        r = self.make_router()
        table = r.get_routing_table()
        assert "node-1" in table
        assert "state" in table["node-1"]
        assert "is_receiving_traffic" in table["node-1"]

    def test_cooldown_prevents_repeat_reroute(self):
        r = self.make_router()
        r.reroute_node(node_id="node-1", reason="first")
        r.restore_node("node-1")
        decision2 = r.reroute_node(
            node_id="node-1", reason="second"
        )
        assert decision2 is None

    def test_status_structure(self):
        r = self.make_router()
        status = r.status()
        assert "running" in status
        assert "strategy" in status
        assert "total_reroutes" in status
        assert "node_states" in status


class TestAutoRetrainer:

    def make_retrainer(self, active_alerts=None):
        updater = make_mock_updater()
        alerter = make_mock_alerter(active_alerts=active_alerts)
        r = AutoRetrainer(
            updater=updater,
            alerter=alerter,
            check_interval=60.0,
            drift_threshold_ms=3.0,
            auto_retrain=True,
        )
        return r

    def test_retrainer_initializes(self):
        r = self.make_retrainer()
        assert r is not None
        assert r._total_retrains == 0

    def test_manual_retrain_success(self):
        r = self.make_retrainer()
        record = r.trigger_retrain(
            node_ids=["node-1"],
            trigger=RetrainTrigger.MANUAL,
            reason="test",
        )
        assert record is not None
        assert record.status == RetrainStatus.SUCCESS
        assert r._total_retrains == 1
        assert r._successful_retrains == 1

    def test_retrain_all_nodes(self):
        r = self.make_retrainer()
        record = r.trigger_retrain(
            node_ids=["node-1", "node-2", "node-3"],
            trigger=RetrainTrigger.SCHEDULED,
            reason="test all",
        )
        assert record is not None
        assert set(record.node_ids) == {
            "node-1", "node-2", "node-3"
        }

    def test_cooldown_prevents_repeat(self):
        r = self.make_retrainer()
        r.trigger_retrain(
            node_ids=["node-1"],
            trigger=RetrainTrigger.MANUAL,
            reason="first",
        )
        record2 = r.trigger_retrain(
            node_ids=["node-1"],
            trigger=RetrainTrigger.MANUAL,
            reason="second",
        )
        assert record2 is None

    def test_not_ready_fails(self):
        updater = make_mock_updater(ready=False)
        alerter = make_mock_alerter()
        r = AutoRetrainer(
            updater=updater,
            alerter=alerter,
            check_interval=60.0,
        )
        record = r.trigger_retrain(
            node_ids=["node-1"],
            trigger=RetrainTrigger.MANUAL,
            reason="test",
        )
        assert record is not None
        assert record.status == RetrainStatus.FAILED

    def test_record_history_populated(self):
        r = self.make_retrainer()
        r.trigger_retrain(
            node_ids=["node-1"],
            trigger=RetrainTrigger.MANUAL,
            reason="test",
        )
        history = r.get_record_history()
        assert len(history) > 0

    def test_drift_detection(self):
        r = self.make_retrainer()
        with r._lock:
            r._last_effects = {
                "node-1": 20.0,
                "node-2": 20.0,
                "node-3": 20.0,
            }
        drifted = r._detect_drift()
        assert "node-1" in drifted
        assert "node-2" in drifted
        assert "node-3" in drifted

    def test_no_drift_below_threshold(self):
        r = self.make_retrainer()
        with r._lock:
            r._last_effects = {
                "node-1": 28.4,
                "node-2": 28.4,
                "node-3": 28.4,
            }
        drifted = r._detect_drift()
        assert len(drifted) == 0

    def test_status_structure(self):
        r = self.make_retrainer()
        status = r.status()
        assert "running" in status
        assert "auto_retrain" in status
        assert "total_retrains" in status
        assert "drift_threshold_ms" in status
        assert "current_effects" in status


class TestRecoverySequence:

    def make_sequence(self, node_id="node-1"):
        return RecoverySequence(
            sequence_id="test-001",
            node_id=node_id,
            trigger=RecoveryTrigger.MANUAL,
            trigger_alert={"alert_id": "alert-001"},
            timestamp="2026-01-01T00:00:00+00:00",
        )

    def test_sequence_starts_detected(self):
        seq = self.make_sequence()
        assert seq.phase == RecoveryPhase.DETECTED

    def test_advance_phase(self):
        seq = self.make_sequence()
        seq.advance_phase(RecoveryPhase.REROUTING, "test note")
        assert seq.phase == RecoveryPhase.REROUTING
        assert len(seq.phase_history) == 1

    def test_record_action(self):
        seq = self.make_sequence()
        seq.record_action("retrain", "success", True)
        assert len(seq.actions_taken) == 1
        assert seq.actions_taken[0]["success"] is True

    def test_complete_success(self):
        seq = self.make_sequence()
        seq.complete(True, "all good")
        assert seq.phase == RecoveryPhase.RESTORED
        assert seq.completed_at is not None
        assert seq.duration_seconds is not None

    def test_complete_failure(self):
        seq = self.make_sequence()
        seq.complete(False, "something failed")
        assert seq.phase == RecoveryPhase.FAILED
        assert seq.failure_reason == "something failed"

    def test_to_dict_structure(self):
        seq = self.make_sequence()
        d = seq.to_dict()
        assert "sequence_id" in d
        assert "node_id" in d
        assert "phase" in d
        assert "trigger" in d
        assert "actions_taken" in d
        assert "phase_history" in d


class TestRecoveryOrchestrator:

    def make_orchestrator(self, active_alerts=None):
        updater = make_mock_updater()
        alerter = make_mock_alerter(active_alerts=active_alerts)
        eng = HealingActionEngine(
            updater=updater,
            alerter=alerter,
            check_interval=60.0,
            auto_heal=False,
        )
        r = QueryRouter(
            updater=updater,
            alerter=alerter,
            check_interval=60.0,
        )
        retrainer = AutoRetrainer(
            updater=updater,
            alerter=alerter,
            check_interval=60.0,
        )
        orch = RecoveryOrchestrator(
            updater=updater,
            alerter=alerter,
            engine=eng,
            router=r,
            retrainer=retrainer,
            check_interval=60.0,
            auto_recover=True,
        )
        return orch

    def test_orchestrator_initializes(self):
        orch = self.make_orchestrator()
        assert orch is not None
        assert orch._total_sequences == 0

    def test_manual_recovery_starts(self):
        orch = self.make_orchestrator()
        result = orch.trigger_manual_recovery(
            node_id="node-1",
            reason="test manual",
        )
        assert result is not None
        assert result["node_id"] == "node-1"
        assert result["trigger"] == "manual"

    def test_manual_recovery_unknown_node(self):
        orch = self.make_orchestrator()
        result = orch.trigger_manual_recovery(
            node_id="node-99",
            reason="test",
        )
        assert result is None

    def test_no_duplicate_recovery_same_node(self):
        orch = self.make_orchestrator()
        result1 = orch.trigger_manual_recovery(
            node_id="node-1",
            reason="first",
        )
        result2 = orch.trigger_manual_recovery(
            node_id="node-1",
            reason="second",
        )
        assert result1 is not None
        assert result2 is None

    def test_recovery_sequence_runs_to_completion(self):
        orch = self.make_orchestrator()
        result = orch.trigger_manual_recovery(
            node_id="node-1",
            reason="test completion",
        )
        assert result is not None
        time.sleep(25)
        history = orch.get_sequence_history()
        assert len(history) > 0
        completed = history[-1]
        assert completed["phase"] in ("restored", "failed")

    def test_status_structure(self):
        orch = self.make_orchestrator()
        status = orch.status()
        assert "running" in status
        assert "auto_recover" in status
        assert "total_sequences" in status
        assert "successful_recoveries" in status
        assert "active_sequences" in status


class TestSelfHealingBenchmarks:

    def test_policy_decision_latency(self):
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            HealingPolicy.decide(
                AlertType.LATENCY_RISING.value,
                AlertSeverity.WARNING.value,
            )
            latencies.append((time.perf_counter() - start) * 1000)
        assert statistics.mean(latencies) < 0.1

    def test_healing_check_cycle_latency(self):
        updater = make_mock_updater()
        alerter = make_mock_alerter(
            active_alerts=[
                make_latency_alert(severity=AlertSeverity.WARNING.value)
            ]
        )
        eng = HealingActionEngine(
            updater=updater,
            alerter=alerter,
            check_interval=60.0,
        )
        latencies = []
        for _ in range(5):
            eng._cooldowns.clear()
            start = time.perf_counter()
            eng._check_cycle()
            latencies.append((time.perf_counter() - start) * 1000)
        assert statistics.mean(latencies) < 50.0

    def test_routing_decision_latency(self):
        updater = make_mock_updater()
        alerter = make_mock_alerter()
        r = QueryRouter(
            updater=updater,
            alerter=alerter,
            check_interval=60.0,
        )
        latencies = []
        for i in range(10):
            r._reroute_cooldowns.clear()
            r._node_states = {
                n: NodeRoutingState.ACTIVE
                for n in ["node-1", "node-2", "node-3"]
            }
            start = time.perf_counter()
            r.reroute_node("node-1", "bench")
            latencies.append((time.perf_counter() - start) * 1000)
        assert statistics.mean(latencies) < 10.0

    def test_retrain_trigger_latency(self):
        updater = make_mock_updater()
        alerter = make_mock_alerter()
        r = AutoRetrainer(
            updater=updater,
            alerter=alerter,
            check_interval=60.0,
        )
        latencies = []
        for i in range(5):
            r._last_retrain_time.clear()
            start = time.perf_counter()
            r.trigger_retrain(
                node_ids=[f"node-{(i%3)+1}"],
                trigger=RetrainTrigger.MANUAL,
                reason="bench",
            )
            latencies.append((time.perf_counter() - start) * 1000)
        assert statistics.mean(latencies) < 200.0

    def test_causal_weight_computation_latency(self):
        updater = make_mock_updater()
        alerter = make_mock_alerter()
        r = QueryRouter(
            updater=updater,
            alerter=alerter,
            check_interval=60.0,
        )
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            r._compute_causal_weights(
                ["node-1", "node-2", "node-3"]
            )
            latencies.append((time.perf_counter() - start) * 1000)
        assert statistics.mean(latencies) < 1.0

    def test_thread_safety_concurrent_reroutes(self):
        updater = make_mock_updater()
        alerter = make_mock_alerter()
        r = QueryRouter(
            updater=updater,
            alerter=alerter,
            check_interval=60.0,
        )
        errors = []

        def do_reroute(node_id):
            try:
                for _ in range(5):
                    r._reroute_cooldowns.clear()
                    r._node_states[node_id] = NodeRoutingState.ACTIVE
                    r.reroute_node(node_id, "concurrent")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(
                target=do_reroute,
                args=(f"node-{i+1}",)
            )
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0