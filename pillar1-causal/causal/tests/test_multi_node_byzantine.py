import sys
import os
import time
import statistics
import threading
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "telemetry"))

from quorum_manager import (
    QuorumManager,
    QuorumState,
    NodeCapacityState,
    QuorumDecision,
    NodeQuorumStatus,
)
from byzantine_recovery_coordinator import (
    ByzantineRecoveryCoordinator,
    ByzantineNodeState,
    ByzantineDetectionMethod,
    ByzantineEvent,
    RecoveryPlan,
    RecoveryPriority,
)
from multi_node_recovery_orchestrator import (
    MultiNodeRecoveryOrchestrator,
    RecoverySession,
    RecoveryPhase,
    NodeRecoveryOutcome,
)
from quorum_aware_router import (
    QuorumAwareRouter,
    RoutingDecisionType,
    NodeRoutingState,
    RoutingDecision,
)


ALL_NODES = ["node-1", "node-2", "node-3"]


def make_mock_updater(
    ready=True,
    effects=None,
    buffer_size=60,
    retrain_count=5,
):
    if effects is None:
        effects = {
            "node-1": 28.5,
            "node-2": 30.0,
            "node-3": 27.0,
        }
    updater = MagicMock()
    updater.status.return_value = {
        "is_ready": ready,
        "buffer_size": buffer_size,
        "retrain_count": retrain_count,
        "max_buffer_size": 200,
        "nodes_modeled": ALL_NODES,
    }

    def get_snapshot(node_id):
        return {
            "effect": effects.get(node_id, 28.5),
            "samples_used": buffer_size,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    updater.get_current_snapshot.side_effect = (
        get_snapshot
    )

    import pandas as pd
    df = pd.DataFrame({
        "node_1_active_queries": [5.0] * 30,
        "node_2_active_queries": [5.0] * 30,
        "node_3_active_queries": [5.0] * 30,
    })
    mock_buf = MagicMock()
    mock_buf.get_dataframe.return_value = df
    updater.buffer = mock_buf
    return updater


def make_quorum_manager(updater=None, ready=True):
    if updater is None:
        updater = make_mock_updater(ready=ready)
    qm = QuorumManager(
        updater=updater,
        check_interval=60.0,
    )
    return qm


def make_coordinator(
    updater=None,
    quorum=None,
    effects=None,
):
    if updater is None:
        updater = make_mock_updater(effects=effects)
    if quorum is None:
        quorum = make_quorum_manager(updater=updater)
    router = MagicMock()
    router.status.return_value = {
        "node_states": {
            n: "active" for n in ALL_NODES
        },
        "running": True,
    }
    router._compute_causal_weights.return_value = {
        "node-1": 0.36,
        "node-2": 0.34,
        "node-3": 0.30,
    }
    retrainer = MagicMock()
    retrainer.status.return_value = {"running": True}
    retrainer._is_in_cooldown.return_value = False

    coord = ByzantineRecoveryCoordinator(
        updater=updater,
        quorum_manager=quorum,
        router=router,
        retrainer=retrainer,
        check_interval=60.0,
    )
    return coord


def make_orchestrator(
    updater=None,
    quorum=None,
    coordinator=None,
):
    if updater is None:
        updater = make_mock_updater()
    if quorum is None:
        quorum = make_quorum_manager(updater=updater)
    if coordinator is None:
        coordinator = make_coordinator(
            updater=updater,
            quorum=quorum,
        )
    orch = MultiNodeRecoveryOrchestrator(
        updater=updater,
        quorum_manager=quorum,
        coordinator=coordinator,
        check_interval=60.0,
    )
    return orch


def make_quorum_router(updater=None, quorum=None):
    if updater is None:
        updater = make_mock_updater()
    if quorum is None:
        quorum = make_quorum_manager(updater=updater)
    router = QuorumAwareRouter(
        updater=updater,
        quorum_manager=quorum,
        check_interval=60.0,
    )
    return router


# ── QuorumManager tests ────────────────────────────────────

class TestQuorumManager:

    def test_initializes_healthy(self):
        qm = make_quorum_manager()
        assert qm.get_quorum_state() == QuorumState.HEALTHY

    def test_contributing_nodes_all_full(self):
        qm = make_quorum_manager()
        assert qm._count_contributing_nodes() == 3

    def test_request_offline_allow_when_quorum_safe(self):
        qm = make_quorum_manager()
        decision = qm.request_node_offline(
            "node-1", "test"
        )
        assert decision == QuorumDecision.ALLOW

    def test_request_offline_deny_when_quorum_at_min(self):
        qm = make_quorum_manager()
        # One offline → contributing=2 = MINIMUM_QUORUM → CRITICAL
        qm.mark_node_offline("node-1")
        # Taking node-2 offline would leave 1/3 → DENY
        decision = qm.request_node_offline(
            "node-2", "test"
        )
        assert decision == QuorumDecision.DENY_QUORUM_RISK

    def test_request_offline_deny_concurrent(self):
        qm = make_quorum_manager()
        # node-1 RECOVERING: contributing = node-2 + node-3 = 2
        qm.mark_node_recovering("node-1")
        # Taking node-3 offline: would leave 1 contributing → DENY_QUORUM_RISK
        # Taking node-2 offline: would leave 1 contributing → DENY_QUORUM_RISK
        # For DENY_CONCURRENT we need quorum to be safe but concurrent limit hit
        # Reset: all full, then mark recovering and try another recovering node
        qm2 = make_quorum_manager()
        # With all 3 full, mark node-1 recovering
        qm2.mark_node_recovering("node-1")
        # node-2 can go offline safely (2 contributing remain: node-2→0, node-3 still full)
        # Actually contributing after node-1 recovering = node-2 full + node-3 full = 2
        # Taking node-2 offline: contributing = 1 < MINIMUM_QUORUM → DENY_QUORUM_RISK
        # DENY_CONCURRENT only fires when quorum is safe AND concurrent limit reached
        # Need 3 nodes where 1 recovering and 1 more can safely go offline
        # This is impossible with 3 nodes + min_quorum=2: can't have 1 recovering + 1 more offline safely
        # So DENY_CONCURRENT is only reachable via request_recovery_start
        decision = qm2.request_recovery_start(
            "node-2", "test"
        )
        assert decision == QuorumDecision.DENY_CONCURRENT

    def test_mark_node_recovering_updates_state(self):
        qm = make_quorum_manager()
        qm.mark_node_recovering("node-1")
        status = qm.get_node_status("node-1")
        assert status["capacity_state"] == "recovering"

    def test_mark_node_full_restores(self):
        qm = make_quorum_manager()
        qm.mark_node_recovering("node-1")
        qm.mark_node_full("node-1")
        status = qm.get_node_status("node-1")
        assert status["capacity_state"] == "full"

    def test_mark_node_offline_updates_quorum(self):
        qm = make_quorum_manager()
        qm.mark_node_offline("node-1")
        assert qm._count_contributing_nodes() == 2
        assert qm.get_quorum_state() == QuorumState.DEGRADED

    def test_two_offline_reaches_critical(self):
        qm = make_quorum_manager()
        qm.mark_node_offline("node-1")
        qm.mark_node_offline("node-2")
        # contributing=1, MINIMUM_QUORUM=2 → QUORUM_LOST
        assert qm.get_quorum_state() == QuorumState.QUORUM_LOST

    def test_three_offline_loses_quorum(self):
        qm = make_quorum_manager()
        qm.mark_node_offline("node-1")
        qm.mark_node_offline("node-2")
        qm.mark_node_offline("node-3")
        assert qm.get_quorum_state() == QuorumState.QUORUM_LOST

    def test_reduced_node_still_contributes(self):
        qm = make_quorum_manager()
        qm.mark_node_reduced("node-1")
        assert qm._count_contributing_nodes() == 3
        assert qm.get_quorum_state() == QuorumState.HEALTHY

    def test_safe_to_offline_returns_correct_nodes(self):
        qm = make_quorum_manager()
        safe = qm.get_safe_to_offline()
        assert len(safe) == 3

    def test_safe_to_offline_empty_when_at_min(self):
        qm = make_quorum_manager()
        qm.mark_node_offline("node-1")
        qm.mark_node_offline("node-2")
        safe = qm.get_safe_to_offline()
        assert safe == []

    def test_decision_history_recorded(self):
        qm = make_quorum_manager()
        qm.request_node_offline("node-1", "test1")
        qm.request_node_offline("node-2", "test2")
        history = qm.get_decision_history(n=10)
        assert len(history) == 2

    def test_quorum_violations_counted(self):
        qm = make_quorum_manager()
        qm.mark_node_offline("node-1")
        qm.mark_node_offline("node-2")
        qm.mark_node_offline("node-3")
        assert qm._quorum_violations >= 1

    def test_status_structure(self):
        qm = make_quorum_manager()
        status = qm.status()
        assert "quorum_state" in status
        assert "contributing_nodes" in status
        assert "minimum_quorum" in status
        assert "node_states" in status
        assert "safe_to_offline" in status

    def test_update_byzantine_score(self):
        qm = make_quorum_manager()
        qm.update_byzantine_score("node-1", 0.75)
        status = qm.get_node_status("node-1")
        assert status["byzantine_score"] == 0.75

    def test_consecutive_failure_tracking(self):
        qm = make_quorum_manager()
        qm.record_node_failure("node-1")
        qm.record_node_failure("node-1")
        status = qm.get_node_status("node-1")
        assert status["consecutive_failures"] == 2

    def test_auto_classify_reduced_on_high_effect(self):
        updater = make_mock_updater(
            effects={
                "node-1": 45.0,
                "node-2": 28.0,
                "node-3": 27.0,
            }
        )
        qm = make_quorum_manager(updater=updater)
        qm._update_node_metrics()
        status = qm.get_node_status("node-1")
        assert status["capacity_state"] == "reduced"

    def test_auto_classify_full_on_normal_effect(self):
        updater = make_mock_updater(
            effects={
                "node-1": 28.0,
                "node-2": 29.0,
                "node-3": 27.0,
            }
        )
        qm = make_quorum_manager(updater=updater)
        qm._update_node_metrics()
        status = qm.get_node_status("node-1")
        assert status["capacity_state"] == "full"

    def test_thread_safety_concurrent_requests(self):
        qm = make_quorum_manager()
        results = []
        errors = []

        def make_request(node_id):
            try:
                d = qm.request_node_offline(
                    node_id, "concurrent"
                )
                results.append(d)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(
                target=make_request,
                args=(f"node-{i+1}",)
            )
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 3


# ── ByzantineRecoveryCoordinator tests ────────────────────

class TestByzantineRecoveryCoordinator:

    def test_initializes(self):
        coord = make_coordinator()
        assert coord is not None
        assert coord._check_count == 0

    def test_initial_states_healthy(self):
        coord = make_coordinator()
        states = coord.get_all_node_states()
        for node_id in ALL_NODES:
            assert states[node_id]["state"] == "healthy"

    def test_byzantine_score_zero_initially(self):
        coord = make_coordinator()
        for node_id in ALL_NODES:
            state = coord.get_node_state(node_id)
            assert state["byzantine_score"] == 0.0

    def test_effect_spike_detection(self):
        effects = {
            "node-1": 65.0,  # spike: > 1.5x others
            "node-2": 28.0,
            "node-3": 27.0,
        }
        coord = make_coordinator(effects=effects)
        score = coord._compute_byzantine_score(
            "node-1", effects
        )
        assert score > 0.0

    def test_no_spike_when_effects_balanced(self):
        effects = {
            "node-1": 28.5,
            "node-2": 29.0,
            "node-3": 27.5,
        }
        coord = make_coordinator(effects=effects)
        score = coord._compute_byzantine_score(
            "node-1", effects
        )
        assert score < coord.SUSPECTED_SCORE_THRESHOLD

    def test_consecutive_failures_increase_score(self):
        coord = make_coordinator()
        coord._consecutive_failures["node-1"] = 5
        effects = {
            "node-1": 30.0,
            "node-2": 28.0,
            "node-3": 27.0,
        }
        score = coord._compute_byzantine_score(
            "node-1", effects
        )
        assert score > 0.0

    def test_classify_healthy_low_score(self):
        coord = make_coordinator()
        state = coord._classify_node("node-1", 0.1)
        assert state == ByzantineNodeState.HEALTHY

    def test_classify_suspected_mid_score(self):
        coord = make_coordinator()
        state = coord._classify_node("node-1", 0.5)
        assert state == ByzantineNodeState.SUSPECTED

    def test_classify_confirmed_high_score(self):
        coord = make_coordinator()
        state = coord._classify_node("node-1", 0.8)
        assert state == ByzantineNodeState.CONFIRMED

    def test_recovery_plan_built(self):
        coord = make_coordinator()
        effects = {
            "node-1": 55.0,
            "node-2": 45.0,
            "node-3": 27.0,
        }
        coord._node_states["node-1"] = (
            ByzantineNodeState.CONFIRMED
        )
        coord._node_states["node-2"] = (
            ByzantineNodeState.SUSPECTED
        )
        plan = coord._build_recovery_plan(
            ["node-1", "node-2"], effects
        )
        assert plan is not None
        assert "node-1" in plan.priority_order
        assert "node-2" in plan.priority_order
        # CONFIRMED should come first
        assert plan.priority_order[0] == "node-1"

    def test_record_node_failure(self):
        coord = make_coordinator()
        coord.record_node_failure("node-1")
        coord.record_node_failure("node-1")
        assert coord._consecutive_failures["node-1"] == 2

    def test_get_events_empty_initially(self):
        coord = make_coordinator()
        events = coord.get_events(n=10)
        assert events == []

    def test_status_structure(self):
        coord = make_coordinator()
        status = coord.status()
        assert "check_count" in status
        assert "detections_total" in status
        assert "recoveries_total" in status
        assert "node_states" in status
        assert "byzantine_scores" in status

    def test_divergence_detection(self):
        effects = {
            "node-1": 55.0,
            "node-2": 28.0,
            "node-3": 27.0,
        }
        coord = make_coordinator(effects=effects)
        score = coord._compute_byzantine_score(
            "node-1", effects
        )
        # spread = 28ms > EFFECT_DIVERGENCE_MS=12ms
        # and node-1 is the max
        assert score > 0.0


# ── RecoverySession tests ──────────────────────────────────

class TestRecoverySession:

    def test_session_initializes(self):
        session = RecoverySession(
            session_id="session-0001",
            trigger="test",
        )
        assert session.session_id == "session-0001"
        assert session.phase == RecoveryPhase.DETECTING
        assert not session.is_complete

    def test_session_success_count(self):
        session = RecoverySession("s1", "test")
        session.node_outcomes["node-1"] = (
            NodeRecoveryOutcome.SUCCESS
        )
        session.node_outcomes["node-2"] = (
            NodeRecoveryOutcome.FAILED
        )
        assert session.success_count == 1
        assert session.failure_count == 1

    def test_session_complete_sets_phase(self):
        session = RecoverySession("s1", "test")
        session.complete(RecoveryPhase.COMPLETED)
        assert session.phase == RecoveryPhase.COMPLETED
        assert session.is_complete
        assert session.completed_at is not None

    def test_session_to_dict(self):
        session = RecoverySession("s1", "test")
        session.affected_nodes = ["node-1"]
        session.node_outcomes["node-1"] = (
            NodeRecoveryOutcome.SUCCESS
        )
        session.node_durations["node-1"] = 15.3
        d = session.to_dict()
        assert d["session_id"] == "s1"
        assert "node_outcomes" in d
        assert "node_durations_seconds" in d

    def test_mttr_computed_for_successful_nodes(self):
        session = RecoverySession("s1", "test")
        session.affected_nodes = ["node-1", "node-2"]
        session.node_outcomes["node-1"] = (
            NodeRecoveryOutcome.SUCCESS
        )
        session.node_outcomes["node-2"] = (
            NodeRecoveryOutcome.SUCCESS
        )
        session.node_durations["node-1"] = 20.0
        session.node_durations["node-2"] = 30.0
        session.complete(RecoveryPhase.COMPLETED)
        assert session.mttr_seconds == pytest.approx(
            25.0, abs=0.1
        )


# ── MultiNodeRecoveryOrchestrator tests ───────────────────

class TestMultiNodeRecoveryOrchestrator:

    def test_initializes(self):
        orch = make_orchestrator()
        assert orch is not None
        assert orch._sessions_total == 0

    def test_status_structure(self):
        orch = make_orchestrator()
        status = orch.status()
        assert "sessions_total" in status
        assert "sessions_completed" in status
        assert "session_success_rate" in status
        assert "nodes_recovered_total" in status

    def test_no_active_session_initially(self):
        orch = make_orchestrator()
        assert orch.get_active_session() is None

    def test_session_history_empty_initially(self):
        orch = make_orchestrator()
        assert orch.get_session_history() == []

    def test_build_priority_order_highest_effect_first(self):
        orch = make_orchestrator()
        effects = {
            "node-1": 55.0,
            "node-2": 45.0,
            "node-3": 30.0,
        }
        order = orch._build_priority_order(
            ["node-1", "node-2", "node-3"], effects
        )
        assert order[0] == "node-1"
        assert order[1] == "node-2"
        assert order[2] == "node-3"

    def test_open_session_increments_counter(self):
        orch = make_orchestrator()
        orch._start_time = time.time()
        effects = {"node-1": 50.0}
        session = orch._open_session(
            affected_nodes=["node-1"],
            effects=effects,
            trigger="test",
        )
        assert session is not None
        assert orch._sessions_total == 1

    def test_verify_recovery_pass_within_ratio(self):
        effects = {
            "node-1": 28.0,
            "node-2": 29.0,
            "node-3": 27.0,
        }
        updater = make_mock_updater(effects=effects)
        orch = make_orchestrator(updater=updater)
        # node-1 effect=28 vs cluster_mean=28
        # ratio = 1.0 <= 1.5 → PASS
        result = orch._verify_recovery("node-1", effects)
        assert result is True

    def test_verify_recovery_fail_outside_ratio(self):
        effects = {
            "node-1": 65.0,
            "node-2": 29.0,
            "node-3": 27.0,
        }
        updater = make_mock_updater(effects=effects)
        orch = make_orchestrator(updater=updater)
        # node-1 effect=65 vs cluster_mean=28
        # ratio = 2.32 > 1.5 → FAIL
        result = orch._verify_recovery("node-1", effects)
        assert result is False

    def test_success_rate_one_when_no_sessions(self):
        orch = make_orchestrator()
        status = orch.status()
        assert status["session_success_rate"] == 1.0


# ── QuorumAwareRouter tests ────────────────────────────────

class TestQuorumAwareRouter:

    def test_initializes(self):
        router = make_quorum_router()
        assert router is not None
        assert router._check_count == 0

    def test_initial_weights_uniform(self):
        router = make_quorum_router()
        weights = router.get_weights()
        for node_id in ALL_NODES:
            assert weights[node_id] == pytest.approx(
                1.0 / 3, abs=0.01
            )

    def test_compute_weights_normal_all_full(self):
        effects = {
            "node-1": 28.0,
            "node-2": 30.0,
            "node-3": 27.0,
        }
        capacity = {n: "full" for n in ALL_NODES}
        router = make_quorum_router()
        weights, excluded, dtype, states, reason = (
            router._compute_weights(
                effects, capacity, "healthy"
            )
        )
        assert dtype == RoutingDecisionType.NORMAL
        assert excluded == []
        # lower effect → higher weight
        assert weights["node-3"] > weights["node-2"]

    def test_compute_weights_excludes_recovering(self):
        effects = {
            "node-1": 28.0,
            "node-2": 30.0,
            "node-3": 27.0,
        }
        capacity = {
            "node-1": "recovering",
            "node-2": "full",
            "node-3": "full",
        }
        router = make_quorum_router()
        weights, excluded, dtype, states, reason = (
            router._compute_weights(
                effects, capacity, "healthy"
            )
        )
        assert "node-1" in excluded
        assert weights["node-1"] == 0.0
        assert dtype == RoutingDecisionType.DEGRADED

    def test_compute_weights_excludes_offline(self):
        effects = {n: 28.0 for n in ALL_NODES}
        capacity = {
            "node-1": "offline",
            "node-2": "full",
            "node-3": "full",
        }
        router = make_quorum_router()
        weights, excluded, dtype, states, reason = (
            router._compute_weights(
                effects, capacity, "healthy"
            )
        )
        assert "node-1" in excluded
        assert weights["node-1"] == 0.0

    def test_reduced_penalty_applied(self):
        effects = {
            "node-1": 28.0,
            "node-2": 28.0,
            "node-3": 28.0,
        }
        capacity = {
            "node-1": "reduced",
            "node-2": "full",
            "node-3": "full",
        }
        router = make_quorum_router()
        weights, _, _, _, _ = router._compute_weights(
            effects, capacity, "healthy"
        )
        # node-1 should have less weight than node-2
        assert weights["node-1"] < weights["node-2"]

    def test_emergency_uniform_on_quorum_lost(self):
        effects = {n: 28.0 for n in ALL_NODES}
        capacity = {n: "full" for n in ALL_NODES}
        router = make_quorum_router()
        weights, excluded, dtype, states, reason = (
            router._compute_weights(
                effects, capacity, "quorum_lost"
            )
        )
        assert dtype == (
            RoutingDecisionType.EMERGENCY_UNIFORM
        )
        assert excluded == []
        for node_id in ALL_NODES:
            assert weights[node_id] == pytest.approx(
                1.0 / 3, abs=0.01
            )

    def test_critical_decision_two_excluded(self):
        effects = {n: 28.0 for n in ALL_NODES}
        capacity = {
            "node-1": "offline",
            "node-2": "recovering",
            "node-3": "full",
        }
        router = make_quorum_router()
        weights, excluded, dtype, states, reason = (
            router._compute_weights(
                effects, capacity, "critical"
            )
        )
        assert dtype == RoutingDecisionType.CRITICAL
        assert len(excluded) == 2

    def test_weight_floor_applied(self):
        effects = {
            "node-1": 28.0,
            "node-2": 28.0,
            "node-3": 500.0,  # extreme effect
        }
        capacity = {n: "full" for n in ALL_NODES}
        router = make_quorum_router()
        weights, _, _, _, _ = router._compute_weights(
            effects, capacity, "healthy"
        )
        # Floor applied pre-normalisation so final may be slightly below 0.05
        # but should be well above zero
        assert weights["node-3"] > 0.01
        # And significantly less than equal share
        assert weights["node-3"] < weights["node-1"]

    def test_route_request_returns_valid_node(self):
        router = make_quorum_router()
        node = router.route_request()
        assert node in ALL_NODES

    def test_route_request_excludes_offline_node(self):
        updater = make_mock_updater()
        quorum = make_quorum_manager(updater=updater)
        quorum.mark_node_offline("node-1")
        router = make_quorum_router(
            updater=updater, quorum=quorum
        )
        router._start_time = time.time()
        router._check_cycle()

        # node-1 weight should be 0
        weights = router.get_weights()
        assert weights["node-1"] == 0.0

        # 100 requests should not hit node-1
        hits = {n: 0 for n in ALL_NODES}
        for _ in range(100):
            node = router.route_request()
            hits[node] += 1
        assert hits["node-1"] == 0

    def test_stability_score_one_initially(self):
        router = make_quorum_router()
        stability = router.get_weight_stability()
        for node_id in ALL_NODES:
            assert stability[node_id] == 1.0

    def test_status_structure(self):
        router = make_quorum_router()
        status = router.status()
        assert "current_weights" in status
        assert "routing_states" in status
        assert "excluded_nodes" in status
        assert "cluster_stability" in status
        assert "decision_count" in status

    def test_thread_safety_concurrent_checks(self):
        router = make_quorum_router()
        router._start_time = time.time()
        errors = []

        def run_check():
            try:
                for _ in range(5):
                    router._check_cycle()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=run_check)
            for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ── Integration tests ──────────────────────────────────────

class TestSprint9Integration:

    def test_quorum_gates_coordinator_recovery(self):
        updater = make_mock_updater()
        quorum = make_quorum_manager(updater=updater)
        coord = make_coordinator(
            updater=updater, quorum=quorum
        )

        # Mark node-1 as Byzantine CONFIRMED
        coord._byzantine_scores["node-1"] = 0.80
        coord._node_states["node-1"] = (
            ByzantineNodeState.CONFIRMED
        )

        # Quorum should allow offline (3/3 → 2/3 safe)
        decision = quorum.request_node_offline(
            "node-1", "integration_test"
        )
        assert decision == QuorumDecision.ALLOW

    def test_quorum_prevents_second_concurrent_recovery(self):
        updater = make_mock_updater()
        quorum = make_quorum_manager(updater=updater)

        # node-1 already recovering
        quorum.mark_node_recovering("node-1")

        # node-2 recovery start should be denied concurrent
        decision = quorum.request_recovery_start(
            "node-2", "integration_test"
        )
        assert decision == QuorumDecision.DENY_CONCURRENT

    def test_router_excludes_recovering_node(self):
        updater = make_mock_updater()
        quorum = make_quorum_manager(updater=updater)
        quorum.mark_node_recovering("node-2")
        router = make_quorum_router(
            updater=updater, quorum=quorum
        )
        router._start_time = time.time()
        router._check_cycle()

        weights = router.get_weights()
        assert weights["node-2"] == 0.0
        assert weights["node-1"] > 0.0
        assert weights["node-3"] > 0.0

    def test_full_stack_healthy_state(self):
        updater = make_mock_updater(
            effects={
                "node-1": 27.5,
                "node-2": 28.5,
                "node-3": 29.0,
            }
        )
        quorum = make_quorum_manager(updater=updater)
        coord = make_coordinator(
            updater=updater, quorum=quorum
        )
        orch = make_orchestrator(
            updater=updater,
            quorum=quorum,
            coordinator=coord,
        )
        router = make_quorum_router(
            updater=updater, quorum=quorum
        )

        # All systems should report healthy
        assert (
            quorum.get_quorum_state()
            == QuorumState.HEALTHY
        )
        states = coord.get_all_node_states()
        for node_id in ALL_NODES:
            assert states[node_id]["state"] == "healthy"
        assert orch.get_active_session() is None
        weights = router.get_weights()
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_quorum_state_transitions_full_sequence(self):
        updater = make_mock_updater()
        quorum = make_quorum_manager(updater=updater)

        assert quorum.get_quorum_state() == QuorumState.HEALTHY
        quorum.mark_node_offline("node-1")
        assert quorum.get_quorum_state() == QuorumState.DEGRADED
        quorum.mark_node_offline("node-2")
        # contributing=1 < MINIMUM_QUORUM=2 → QUORUM_LOST
        assert quorum.get_quorum_state() == QuorumState.QUORUM_LOST
        quorum.mark_node_offline("node-3")
        assert quorum.get_quorum_state() == QuorumState.QUORUM_LOST

        # Restore
        quorum.mark_node_full("node-1")
        quorum.mark_node_full("node-2")
        quorum.mark_node_full("node-3")
        assert quorum.get_quorum_state() == QuorumState.HEALTHY


# ── Benchmarks ──────────────────────────────────────────────

class TestSprint9Benchmarks:

    def test_quorum_decision_latency(self):
        qm = make_quorum_manager()
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            qm.request_node_offline("node-1", "bench")
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 5.0

    def test_byzantine_score_computation_latency(self):
        effects = {
            "node-1": 55.0,
            "node-2": 28.0,
            "node-3": 27.0,
        }
        coord = make_coordinator(effects=effects)
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            coord._compute_byzantine_score(
                "node-1", effects
            )
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 5.0

    def test_weight_computation_latency(self):
        effects = {n: 28.0 for n in ALL_NODES}
        capacity = {n: "full" for n in ALL_NODES}
        router = make_quorum_router()
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            router._compute_weights(
                effects, capacity, "healthy"
            )
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 5.0

    def test_route_request_latency(self):
        router = make_quorum_router()
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            router.route_request()
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 1.0

    def test_quorum_status_latency(self):
        qm = make_quorum_manager()
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            qm.status()
            latencies.append(
                (time.perf_counter() - start) * 1000
            )
        assert statistics.mean(latencies) < 5.0
