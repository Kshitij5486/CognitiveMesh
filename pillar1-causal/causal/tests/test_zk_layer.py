import sys
import os
import json
import time
import statistics
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "telemetry"))

from zk_proof_bridge import ZKProofBridge, CausalProofRequest, CausalProofResult

PROVER_PATH = r"C:\Users\KSHITIJ\mpc-network\target\debug\causal-prover.exe"


def make_bridge() -> ZKProofBridge:
    return ZKProofBridge(prover_path=PROVER_PATH)


def make_valid_request(
    node_id: str = "node-1",
    active_queries: int = 5,
    avg_latency_ms: float = 28.0,
    retrain_cycle: int = 1,
) -> CausalProofRequest:
    mac = int(avg_latency_ms) * active_queries
    return CausalProofRequest(
        node_id=node_id,
        active_queries=active_queries,
        avg_latency_ms=avg_latency_ms,
        causal_effect_ms=float(mac),
        samples_used=60,
        retrain_cycle=retrain_cycle,
    )


class TestCausalProofRequest:

    def test_job_id_auto_generated(self):
        req = make_valid_request()
        assert req.job_id is not None
        assert "node-1" in req.job_id

    def test_job_id_contains_node_id(self):
        req = make_valid_request(node_id="node-2")
        assert "node-2" in req.job_id

    def test_custom_job_id(self):
        req = CausalProofRequest(
            node_id="node-1",
            active_queries=5,
            avg_latency_ms=28.0,
            causal_effect_ms=140.0,
            samples_used=60,
            retrain_cycle=1,
            job_id="my-custom-job",
        )
        assert req.job_id == "my-custom-job"

    def test_to_rust_request_structure(self):
        req = make_valid_request()
        d = req.to_rust_request()
        assert "job_id" in d
        assert "party_id" in d
        assert "node_id" in d
        assert "active_queries" in d
        assert "avg_latency_ms" in d
        assert "causal_effect_ms" in d
        assert "samples_used" in d
        assert "retrain_cycle" in d

    def test_to_rust_request_constraint_holds(self):
        req = make_valid_request(
            active_queries=5,
            avg_latency_ms=28.0,
        )
        d = req.to_rust_request()
        assert d["causal_effect_ms"] == d["avg_latency_ms"] * d["active_queries"]

    def test_to_rust_request_values_are_integers(self):
        req = make_valid_request()
        d = req.to_rust_request()
        assert isinstance(d["active_queries"], int)
        assert isinstance(d["avg_latency_ms"], int)
        assert isinstance(d["causal_effect_ms"], int)

    def test_default_party_id(self):
        req = make_valid_request()
        assert req.party_id == 1


class TestZKProofBridge:

    def test_bridge_initializes(self):
        bridge = make_bridge()
        assert bridge is not None
        assert os.path.exists(bridge.prover_path)

    def test_bridge_missing_prover_raises(self):
        with pytest.raises(FileNotFoundError):
            ZKProofBridge(prover_path="nonexistent.exe")

    def test_prove_returns_result(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        assert isinstance(result, CausalProofResult)

    def test_prove_succeeds(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        assert result.success is True

    def test_prove_verified(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        assert result.verified is True

    def test_prove_has_proof_id(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        assert result.proof_id() is not None
        assert len(result.proof_id()) > 0

    def test_prove_has_challenge(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        assert result.challenge() is not None
        assert result.challenge() != 0

    def test_prove_constraint_satisfied(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        assert result.constraint_satisfied() is True

    def test_prove_has_timestamp(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        assert result.timestamp is not None

    def test_prove_latency_positive(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        assert result.latency_ms > 0

    def test_prove_different_nodes_different_challenges(self):
        bridge = make_bridge()
        r1 = make_valid_request(node_id="node-1")
        r2 = make_valid_request(node_id="node-2")
        p1 = bridge.prove(r1)
        p2 = bridge.prove(r2)
        assert p1.challenge() != p2.challenge()

    def test_prove_same_inputs_same_challenge(self):
        bridge = make_bridge()
        r1 = CausalProofRequest(
            node_id="node-1",
            active_queries=5,
            avg_latency_ms=28.0,
            causal_effect_ms=140.0,
            samples_used=60,
            retrain_cycle=1,
            job_id="fixed-job-id",
        )
        r2 = CausalProofRequest(
            node_id="node-1",
            active_queries=5,
            avg_latency_ms=28.0,
            causal_effect_ms=140.0,
            samples_used=60,
            retrain_cycle=1,
            job_id="fixed-job-id",
        )
        p1 = bridge.prove(r1)
        p2 = bridge.prove(r2)
        assert p1.challenge() == p2.challenge()

    def test_prove_all_three_nodes(self):
        bridge = make_bridge()
        node_effects = {
            "node-1": {"active_queries": 5, "avg_latency_ms": 28,
                       "causal_effect_ms": 140, "samples_used": 60},
            "node-2": {"active_queries": 5, "avg_latency_ms": 29,
                       "causal_effect_ms": 145, "samples_used": 60},
            "node-3": {"active_queries": 5, "avg_latency_ms": 27,
                       "causal_effect_ms": 135, "samples_used": 60},
        }
        results = bridge.prove_all_nodes(node_effects, retrain_cycle=1)
        assert len(results) == 3
        for node_id, result in results.items():
            assert result.success is True
            assert result.verified is True

    def test_prove_all_nodes_all_verified(self):
        bridge = make_bridge()
        node_effects = {
            "node-1": {"active_queries": 5, "avg_latency_ms": 28,
                       "causal_effect_ms": 140, "samples_used": 60},
            "node-2": {"active_queries": 5, "avg_latency_ms": 29,
                       "causal_effect_ms": 145, "samples_used": 60},
            "node-3": {"active_queries": 5, "avg_latency_ms": 27,
                       "causal_effect_ms": 135, "samples_used": 60},
        }
        results = bridge.prove_all_nodes(node_effects, retrain_cycle=1)
        assert all(r.verified for r in results.values())

    def test_stats_track_proof_count(self):
        bridge = make_bridge()
        req = make_valid_request()
        bridge.prove(req)
        bridge.prove(req)
        stats = bridge.stats()
        assert stats["proofs_generated"] == 2

    def test_stats_track_latency(self):
        bridge = make_bridge()
        req = make_valid_request()
        bridge.prove(req)
        stats = bridge.stats()
        assert stats["avg_latency_ms"] > 0

    def test_verify_proof_json_valid(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        proof_json = json.dumps(result.proof)
        verification = bridge.verify_proof_json(proof_json)
        assert verification["valid"] is True
        assert verification["checks_passed"] == verification["checks_total"]

    def test_verify_tampered_proof_fails(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        tampered = dict(result.proof)
        tampered["challenge"] = 99999999
        tampered["queries"] = []
        proof_json = json.dumps(tampered)
        verification = bridge.verify_proof_json(proof_json)
        assert verification["valid"] is False

    def test_result_to_dict_structure(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        d = result.to_dict()
        assert "success" in d
        assert "node_id" in d
        assert "job_id" in d
        assert "verified" in d
        assert "proof_id" in d
        assert "latency_ms" in d
        assert "timestamp" in d

    def test_result_summary_verified(self):
        bridge = make_bridge()
        req = make_valid_request()
        result = bridge.prove(req)
        summary = result.summary()
        assert "VERIFIED" in summary
        assert "node-1" in summary


class TestZKBenchmarks:

    def test_single_proof_latency_under_100ms(self):
        bridge = make_bridge()
        req = make_valid_request()
        start = time.perf_counter()
        result = bridge.prove(req)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert result.verified is True
        assert elapsed_ms < 100.0

    def test_three_node_proof_latency_under_200ms(self):
        bridge = make_bridge()
        node_effects = {
            "node-1": {"active_queries": 5, "avg_latency_ms": 28,
                       "causal_effect_ms": 140, "samples_used": 60},
            "node-2": {"active_queries": 5, "avg_latency_ms": 29,
                       "causal_effect_ms": 145, "samples_used": 60},
            "node-3": {"active_queries": 5, "avg_latency_ms": 27,
                       "causal_effect_ms": 135, "samples_used": 60},
        }
        start = time.perf_counter()
        results = bridge.prove_all_nodes(node_effects, retrain_cycle=1)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert all(r.verified for r in results.values())
        assert elapsed_ms < 200.0

    def test_proof_latency_consistency(self):
        bridge = make_bridge()
        latencies = []
        for i in range(5):
            req = CausalProofRequest(
                node_id="node-1",
                active_queries=5,
                avg_latency_ms=28.0,
                causal_effect_ms=140.0,
                samples_used=60,
                retrain_cycle=i + 1,
                job_id=f"bench-job-{i}",
            )
            result = bridge.prove(req)
            assert result.verified is True
            latencies.append(result.latency_ms)

        mean_latency = statistics.mean(latencies)
        stddev_latency = statistics.stdev(latencies)
        assert mean_latency < 50.0
        assert stddev_latency < 20.0