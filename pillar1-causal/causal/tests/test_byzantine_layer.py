import sys
import os
import time
import statistics
import threading
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "telemetry"))

from byzantine_detector import (
    ByzantineDetector, ByzantineEvidence, NodeByzantineProfile,
    NodeStatus
)
from reputation_scorer import (
    ReputationScorer, NodeReputation, ReputationEvent
)
from consensus_engine import (
    ConsensusEngine, ConsensusVote, ConsensusResult
)
from isolation_mechanism import (
    IsolationMechanism, IsolationRecord, IsolationReason
)


def make_mock_updater(ready=True, effects=None):
    updater = MagicMock()
    updater.status.return_value = {
        "is_ready": ready,
        "buffer_size": 60,
        "retrain_count": 3,
        "nodes_modeled": ["node-1", "node-2", "node-3"],
        "min_samples_required": 30,
    }
    effects = effects or {
        "node-1": 28.5,
        "node-2": 29.1,
        "node-3": 27.8,
    }
    def get_snapshot(node_id):
        effect = effects.get(node_id)
        if effect is None:
            return None
        return {
            "effect": effect,
            "samples_used": 60,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    updater.get_current_snapshot.side_effect = get_snapshot
    updater.get_all_snapshots.return_value = {
        node_id: get_snapshot(node_id)
        for node_id in ["node-1", "node-2", "node-3"]
    }
    updater.explain.return_value = "mock explanation"
    return updater


def make_mock_bridge(success=True, verified=True):
    bridge = MagicMock()
    result = MagicMock()
    result.success = success
    result.verified = verified
    result.error = None
    bridge.prove.return_value = result
    return bridge


def make_mock_detector(
    node_statuses=None,
    trust_scores=None,
    violations=None,
):
    detector = MagicMock()
    node_statuses = node_statuses or {
        "node-1": NodeStatus.TRUSTED,
        "node-2": NodeStatus.TRUSTED,
        "node-3": NodeStatus.TRUSTED,
    }
    trust_scores = trust_scores or {
        "node-1": 1.0,
        "node-2": 1.0,
        "node-3": 1.0,
    }
    violations = violations or {
        "node-1": 0,
        "node-2": 0,
        "node-3": 0,
    }

    profiles = {}
    for node_id in ["node-1", "node-2", "node-3"]:
        profile = MagicMock()
        profile.status = node_statuses.get(node_id, NodeStatus.TRUSTED)
        profile.causal_violations = violations.get(node_id, 0)
        profile.proof_successes = 3
        profile.proof_failures = 0
        profile.trust_score.return_value = trust_scores.get(node_id, 1.0)
        profile.to_dict.return_value = {
            "node_id": node_id,
            "status": node_statuses.get(node_id, NodeStatus.TRUSTED).value,
            "trust_score": trust_scores.get(node_id, 1.0),
            "proof_successes": 3,
            "proof_failures": 0,
            "causal_violations": violations.get(node_id, 0),
            "checks_run": 5,
            "last_check": "2026-01-01T00:00:00+00:00",
            "flagged_at": None,
            "recent_evidence": [],
        }
        profiles[node_id] = profile

    detector.profiles = profiles
    detector.get_cluster_health.return_value = {
        "cluster_healthy": True,
        "trusted_nodes": 3,
        "suspicious_nodes": 0,
        "byzantine_nodes": 0,
        "checks_run": 5,
        "byzantine_detections": 0,
        "node_profiles": {
            node_id: profiles[node_id].to_dict()
            for node_id in ["node-1", "node-2", "node-3"]
        },
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    return detector


def make_mock_scorer(scores=None):
    scorer = MagicMock()
    scores = scores or {
        "node-1": 0.85,
        "node-2": 0.85,
        "node-3": 0.85,
    }

    def get_score(node_id):
        return scores.get(node_id, 0.5)

    def get_status(node_id):
        score = scores.get(node_id, 0.5)
        if score >= 0.36:
            return NodeStatus.TRUSTED
        elif score >= 0.16:
            return NodeStatus.SUSPICIOUS
        else:
            return NodeStatus.BYZANTINE

    scorer.get_score.side_effect = get_score
    scorer.get_status.side_effect = get_status
    scorer.cluster_summary.return_value = {
        "cluster_healthy": True,
        "trusted_nodes": 3,
        "suspicious_nodes": 0,
        "byzantine_nodes": 0,
        "average_score": sum(scores.values()) / len(scores),
        "node_scores": {
            node_id: {
                "score": score,
                "status": "trusted",
                "trend": "stable",
            }
            for node_id, score in scores.items()
        },
        "update_count": 5,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }

    reputations = {}
    for node_id, score in scores.items():
        rep = MagicMock()
        rep.get_score.return_value = score
        rep.get_trend.return_value = "stable"
        reputations[node_id] = rep

    scorer.reputations = reputations
    return scorer


class TestNodeByzantineProfile:

    def test_profile_starts_unknown(self):
        profile = NodeByzantineProfile("node-1")
        assert profile.status == NodeStatus.UNKNOWN

    def test_trust_score_starts_at_half(self):
        profile = NodeByzantineProfile("node-1")
        assert profile.trust_score() == 0.5

    def test_proof_success_moves_to_trusted(self):
        profile = NodeByzantineProfile("node-1")
        profile.record_proof_result(True)
        assert profile.status == NodeStatus.TRUSTED

    def test_proof_failure_reduces_trust(self):
        profile = NodeByzantineProfile("node-1")
        profile.record_proof_result(True)
        profile.record_proof_result(True)
        profile.record_proof_result(False)
        profile.record_proof_result(False)
        assert profile.trust_score() < 1.0

    def test_causal_violation_moves_to_suspicious(self):
        profile = NodeByzantineProfile("node-1")
        evidence = ByzantineEvidence(
            node_id="node-1",
            evidence_type="causal_effect_deviation",
            expected_value=28.0,
            observed_value=5.0,
            deviation_pct=-82.0,
            severity="CRITICAL",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        profile.add_evidence(evidence)
        assert profile.status == NodeStatus.SUSPICIOUS
        assert profile.causal_violations == 1

    def test_three_violations_moves_to_byzantine(self):
        profile = NodeByzantineProfile("node-1")
        for _ in range(3):
            evidence = ByzantineEvidence(
                node_id="node-1",
                evidence_type="causal_effect_deviation",
                expected_value=28.0,
                observed_value=5.0,
                deviation_pct=-82.0,
                severity="CRITICAL",
                timestamp="2026-01-01T00:00:00+00:00",
            )
            profile.add_evidence(evidence)
        assert profile.status == NodeStatus.BYZANTINE

    def test_trust_score_penalized_by_violations(self):
        profile = NodeByzantineProfile("node-1")
        profile.record_proof_result(True)
        trust_before = profile.trust_score()
        evidence = ByzantineEvidence(
            node_id="node-1",
            evidence_type="causal_effect_deviation",
            expected_value=28.0,
            observed_value=5.0,
            deviation_pct=-82.0,
            severity="CRITICAL",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        profile.add_evidence(evidence)
        assert profile.trust_score() < trust_before

    def test_profile_to_dict_structure(self):
        profile = NodeByzantineProfile("node-1")
        d = profile.to_dict()
        assert "node_id" in d
        assert "status" in d
        assert "trust_score" in d
        assert "proof_successes" in d
        assert "causal_violations" in d


class TestByzantineEvidence:

    def test_evidence_creation(self):
        ev = ByzantineEvidence(
            node_id="node-1",
            evidence_type="causal_effect_deviation",
            expected_value=28.0,
            observed_value=5.0,
            deviation_pct=-82.1,
            severity="CRITICAL",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert ev.node_id == "node-1"
        assert ev.severity == "CRITICAL"
        assert ev.deviation_pct == -82.1

    def test_evidence_to_dict(self):
        ev = ByzantineEvidence(
            node_id="node-2",
            evidence_type="causal_effect_deviation",
            expected_value=28.0,
            observed_value=35.0,
            deviation_pct=25.0,
            severity="WARNING",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        d = ev.to_dict()
        assert d["node_id"] == "node-2"
        assert d["severity"] == "WARNING"
        assert "description" in d


class TestNodeReputation:

    def test_initial_score(self):
        rep = NodeReputation("node-1")
        assert rep.get_score() == NodeReputation.INITIAL_SCORE

    def test_proof_success_reward(self):
        rep = NodeReputation("node-1")
        rep.reward_proof_success()
        assert rep.get_score() > NodeReputation.INITIAL_SCORE

    def test_proof_failure_penalty(self):
        rep = NodeReputation("node-1")
        rep.penalize_proof_failure()
        assert rep.get_score() < NodeReputation.INITIAL_SCORE

    def test_causal_violation_penalty(self):
        rep = NodeReputation("node-1")
        rep.penalize_causal_violation(deviation_pct=-80.0)
        assert rep.get_score() < NodeReputation.INITIAL_SCORE

    def test_honest_cycle_reward(self):
        rep = NodeReputation("node-1")
        rep.reward_honest_cycle()
        assert rep.get_score() > NodeReputation.INITIAL_SCORE

    def test_score_capped_at_max(self):
        rep = NodeReputation("node-1")
        for _ in range(20):
            rep.reward_proof_success()
        assert rep.get_score() <= NodeReputation.MAX_SCORE

    def test_score_floored_at_min(self):
        rep = NodeReputation("node-1")
        for _ in range(20):
            rep.penalize_proof_failure()
        assert rep.get_score() >= NodeReputation.MIN_SCORE

    def test_time_decay_reduces_high_score(self):
        rep = NodeReputation("node-1")
        for _ in range(5):
            rep.reward_proof_success()
        score_before = rep.get_score()
        rep.apply_time_decay()
        assert rep.get_score() <= score_before

    def test_status_trusted_above_threshold(self):
        rep = NodeReputation("node-1")
        for _ in range(10):
            rep.reward_proof_success()
        assert rep.get_status() == NodeStatus.TRUSTED

    def test_status_byzantine_below_threshold(self):
        rep = NodeReputation("node-1")
        for _ in range(10):
            rep.penalize_proof_failure()
        assert rep.get_status() == NodeStatus.BYZANTINE

    def test_trend_improving(self):
        rep = NodeReputation("node-1")
        rep.reward_proof_success()
        rep.reward_proof_success()
        rep.reward_proof_success()
        assert rep.get_trend() == "improving"

    def test_trend_degrading(self):
        rep = NodeReputation("node-1")
        rep.penalize_proof_failure()
        rep.penalize_proof_failure()
        rep.penalize_proof_failure()
        assert rep.get_trend() == "degrading"

    def test_to_dict_structure(self):
        rep = NodeReputation("node-1")
        d = rep.to_dict()
        assert "node_id" in d
        assert "score" in d
        assert "status" in d
        assert "trend" in d
        assert "total_updates" in d


class TestConsensusEngine:

    def test_engine_initializes(self):
        updater = make_mock_updater()
        detector = make_mock_detector()
        scorer = make_mock_scorer()
        eng = ConsensusEngine(
            scorer=scorer,
            detector=detector,
            updater=updater,
        )
        assert eng is not None

    def test_consensus_with_all_trusted(self):
        updater = make_mock_updater()
        detector = make_mock_detector()
        scorer = make_mock_scorer(scores={
            "node-1": 0.85,
            "node-2": 0.85,
            "node-3": 0.85,
        })
        eng = ConsensusEngine(
            scorer=scorer,
            detector=detector,
            updater=updater,
        )
        result = eng.decide_cluster_causal_effect()
        assert result is not None
        assert result.consensus_reached is True
        assert len(result.votes) == 3
        assert len(result.excluded_nodes) == 0

    def test_consensus_excludes_byzantine_node(self):
        updater = make_mock_updater()
        detector = make_mock_detector()
        scorer = make_mock_scorer(scores={
            "node-1": 0.05,
            "node-2": 0.85,
            "node-3": 0.85,
        })
        eng = ConsensusEngine(
            scorer=scorer,
            detector=detector,
            updater=updater,
        )
        result = eng.decide_cluster_causal_effect()
        assert result is not None
        assert "node-1" in result.excluded_nodes
        assert "node-1" not in result.participating_nodes
        assert len(result.votes) == 2

    def test_weighted_average_correct(self):
        updater = make_mock_updater(effects={
            "node-1": 10.0,
            "node-2": 30.0,
            "node-3": 30.0,
        })
        detector = make_mock_detector()
        scorer = make_mock_scorer(scores={
            "node-1": 0.5,
            "node-2": 1.0,
            "node-3": 1.0,
        })
        eng = ConsensusEngine(
            scorer=scorer,
            detector=detector,
            updater=updater,
        )
        result = eng.decide_cluster_causal_effect()
        assert result is not None
        assert result.decided_value > 10.0
        assert result.decided_value < 30.0

    def test_consensus_not_ready_returns_none(self):
        updater = make_mock_updater(ready=False)
        detector = make_mock_detector()
        scorer = make_mock_scorer()
        eng = ConsensusEngine(
            scorer=scorer,
            detector=detector,
            updater=updater,
        )
        result = eng.decide_cluster_causal_effect()
        assert result is None

    def test_threshold_consensus(self):
        updater = make_mock_updater()
        detector = make_mock_detector()
        scorer = make_mock_scorer()
        eng = ConsensusEngine(
            scorer=scorer,
            detector=detector,
            updater=updater,
        )
        result = eng.decide_cluster_threshold()
        assert result is not None
        assert result.decided_value > 0
        assert "threshold" in result.proposal_type

    def test_status_structure(self):
        updater = make_mock_updater()
        detector = make_mock_detector()
        scorer = make_mock_scorer()
        eng = ConsensusEngine(
            scorer=scorer,
            detector=detector,
            updater=updater,
        )
        status = eng.status()
        assert "total_decisions" in status
        assert "can_reach_consensus" in status
        assert "participating_nodes" in status
        assert "excluded_nodes" in status

    def test_recent_decisions_empty_initially(self):
        updater = make_mock_updater()
        detector = make_mock_detector()
        scorer = make_mock_scorer()
        eng = ConsensusEngine(
            scorer=scorer,
            detector=detector,
            updater=updater,
        )
        decisions = eng.get_recent_decisions()
        assert isinstance(decisions, list)
        assert len(decisions) == 0


class TestIsolationMechanism:

    def make_isolation(self, scores=None):
        scores = scores or {
            "node-1": 0.85,
            "node-2": 0.85,
            "node-3": 0.85,
        }
        updater = make_mock_updater()
        detector = make_mock_detector()
        scorer = make_mock_scorer(scores=scores)
        engine = ConsensusEngine(
            scorer=scorer,
            detector=detector,
            updater=updater,
        )
        iso = IsolationMechanism(
            scorer=scorer,
            detector=detector,
            engine=engine,
            check_interval=60.0,
        )
        return iso, scorer

    def test_all_nodes_active_initially(self):
        iso, _ = self.make_isolation()
        assert len(iso.get_active_nodes()) == 3
        assert len(iso.get_isolated_nodes()) == 0

    def test_manual_isolate(self):
        iso, _ = self.make_isolation()
        success = iso.manually_isolate("node-1")
        assert success is True
        assert "node-1" in iso.get_isolated_nodes()
        assert "node-1" not in iso.get_active_nodes()

    def test_manual_release(self):
        iso, _ = self.make_isolation()
        iso.manually_isolate("node-1")
        success = iso.manually_release("node-1")
        assert success is True
        assert "node-1" in iso.get_active_nodes()
        assert "node-1" not in iso.get_isolated_nodes()

    def test_cannot_isolate_already_isolated(self):
        iso, _ = self.make_isolation()
        iso.manually_isolate("node-1")
        success = iso.manually_isolate("node-1")
        assert success is False

    def test_cannot_release_not_isolated(self):
        iso, _ = self.make_isolation()
        success = iso.manually_release("node-1")
        assert success is False

    def test_cluster_remains_operational_with_two_active(self):
        iso, _ = self.make_isolation()
        iso.manually_isolate("node-1")
        assert iso.cluster_status()["cluster_operational"] is True

    def test_isolation_record_created(self):
        iso, _ = self.make_isolation()
        iso.manually_isolate("node-1")
        status = iso.cluster_status()
        assert "node-1" in status["isolation_records"]
        record = status["isolation_records"]["node-1"]
        assert record["currently_isolated"] is True

    def test_isolation_record_has_timestamp(self):
        iso, _ = self.make_isolation()
        iso.manually_isolate("node-1")
        status = iso.cluster_status()
        record = status["isolation_records"]["node-1"]
        assert record["isolated_at"] is not None

    def test_release_record_has_duration(self):
        iso, _ = self.make_isolation()
        iso.manually_isolate("node-1")
        time.sleep(0.1)
        iso.manually_release("node-1")
        history = iso.cluster_status()["isolation_history"]
        assert len(history) > 0
        released = [r for r in history if r["released_at"] is not None]
        assert len(released) > 0
        assert released[0]["duration_seconds"] >= 0.0

    def test_is_node_active(self):
        iso, _ = self.make_isolation()
        assert iso.is_node_active("node-1") is True
        iso.manually_isolate("node-1")
        assert iso.is_node_active("node-1") is False

    def test_status_structure(self):
        iso, _ = self.make_isolation()
        status = iso.status()
        assert "running" in status
        assert "active_nodes" in status
        assert "isolated_nodes" in status
        assert "total_isolations" in status
        assert "cluster_operational" in status


class TestByzantineLayerBenchmarks:

    def test_reputation_update_latency(self):
        rep = NodeReputation("node-1")
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            rep.reward_proof_success()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        mean_ms = statistics.mean(latencies)
        assert mean_ms < 1.0

    def test_consensus_decision_latency(self):
        updater = make_mock_updater()
        detector = make_mock_detector()
        scorer = make_mock_scorer()
        eng = ConsensusEngine(
            scorer=scorer,
            detector=detector,
            updater=updater,
        )
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            result = eng.decide_cluster_causal_effect()
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert result is not None
            latencies.append(elapsed_ms)
        mean_ms = statistics.mean(latencies)
        assert mean_ms < 10.0

    def test_isolation_check_latency(self):
        iso, _ = TestIsolationMechanism().make_isolation()
        latencies = []
        for i in range(5):
            start = time.perf_counter()
            iso.is_node_active(f"node-{(i % 3) + 1}")
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        mean_ms = statistics.mean(latencies)
        assert mean_ms < 1.0

    def test_profile_thread_safety(self):
        profile = NodeByzantineProfile("node-1")
        errors = []

        def update_profile():
            try:
                for _ in range(50):
                    profile.record_proof_result(True)
                    profile.record_check()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_profile) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert profile.proof_successes == 200