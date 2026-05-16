import sys
import os
import time
import threading
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "telemetry"))

from graph_builder import NodeCausalModel, DistributedCausalEngine
from cross_node_causal import CrossNodeCausalGraph, DistributedCausalCorrelator
from streaming_updater import RollingTelemetryBuffer, StreamingCausalUpdater, CausalModelSnapshot
from drift_detector import EffectHistory, DriftEvent, CausalDriftDetector


def make_loaded_dataset(n_samples: int = 60, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    data = {}
    active = np.random.randint(0, 10, n_samples).astype(float)
    for node in ["node_1", "node_2", "node_3"]:
        node_active = active + np.random.normal(0, 0.5, n_samples)
        node_active = np.clip(node_active, 0, None)
        data[f"{node}_buffers_backend"]       = np.random.randint(200, 400, n_samples).astype(float)
        data[f"{node}_buffers_alloc"]         = np.random.randint(800, 1000, n_samples).astype(float)
        data[f"{node}_checkpoints_req"]       = np.random.randint(0, 3, n_samples).astype(float)
        data[f"{node}_avg_query_duration_ms"] = (
            node_active * 28.0 + np.random.normal(0, 2.0, n_samples)
        )
        data[f"{node}_active_queries"]  = node_active
        data[f"{node}_lock_count"]      = node_active * 2 + np.random.normal(0, 0.5, n_samples)
        data[f"{node}_blocked_locks"]   = np.random.randint(0, 2, n_samples).astype(float)
    return pd.DataFrame(data)


def make_sample_dict(seed: int = 0) -> dict:
    np.random.seed(seed)
    sample = {"timestamp": "2026-01-01T00:00:00+00:00"}
    for node in ["node_1", "node_2", "node_3"]:
        sample[f"{node}_buffers_backend"]       = float(np.random.randint(200, 400))
        sample[f"{node}_buffers_alloc"]         = float(np.random.randint(800, 1000))
        sample[f"{node}_checkpoints_req"]       = float(np.random.randint(0, 3))
        sample[f"{node}_avg_query_duration_ms"] = float(np.random.uniform(50, 200))
        sample[f"{node}_active_queries"]        = float(np.random.randint(0, 10))
        sample[f"{node}_lock_count"]            = float(np.random.randint(0, 20))
        sample[f"{node}_blocked_locks"]         = float(np.random.randint(0, 2))
    return sample


class TestRollingTelemetryBuffer:

    def test_buffer_starts_empty(self):
        buf = RollingTelemetryBuffer(max_samples=100, min_samples=30)
        assert buf.size() == 0
        assert buf.total_received() == 0

    def test_buffer_appends_samples(self):
        buf = RollingTelemetryBuffer(max_samples=100, min_samples=30)
        for i in range(10):
            buf.append(make_sample_dict(i))
        assert buf.size() == 10
        assert buf.total_received() == 10

    def test_buffer_respects_max_size(self):
        buf = RollingTelemetryBuffer(max_samples=5, min_samples=2)
        for i in range(20):
            buf.append(make_sample_dict(i))
        assert buf.size() == 5
        assert buf.total_received() == 20

    def test_buffer_not_ready_below_min(self):
        buf = RollingTelemetryBuffer(max_samples=100, min_samples=30)
        for i in range(29):
            buf.append(make_sample_dict(i))
        assert buf.is_ready() is False

    def test_buffer_ready_at_min(self):
        buf = RollingTelemetryBuffer(max_samples=100, min_samples=30)
        for i in range(30):
            buf.append(make_sample_dict(i))
        assert buf.is_ready() is True

    def test_buffer_get_dataframe_none_when_not_ready(self):
        buf = RollingTelemetryBuffer(max_samples=100, min_samples=30)
        for i in range(10):
            buf.append(make_sample_dict(i))
        assert buf.get_dataframe() is None

    def test_buffer_get_dataframe_returns_df_when_ready(self):
        buf = RollingTelemetryBuffer(max_samples=100, min_samples=30)
        for i in range(40):
            buf.append(make_sample_dict(i))
        df = buf.get_dataframe()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40

    def test_buffer_dataframe_excludes_timestamp(self):
        buf = RollingTelemetryBuffer(max_samples=100, min_samples=10)
        for i in range(10):
            buf.append(make_sample_dict(i))
        df = buf.get_dataframe()
        assert "timestamp" not in df.columns

    def test_buffer_thread_safe(self):
        buf = RollingTelemetryBuffer(max_samples=200, min_samples=10)
        errors = []

        def writer():
            try:
                for i in range(50):
                    buf.append(make_sample_dict(i))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert buf.total_received() == 200


class TestCausalModelSnapshot:

    def test_snapshot_creation(self):
        snap = CausalModelSnapshot(
            node_id="node-1",
            effect=28.5,
            timestamp="2026-01-01T00:00:00+00:00",
            samples_used=60,
        )
        assert snap.node_id == "node-1"
        assert snap.effect == 28.5
        assert snap.samples_used == 60

    def test_snapshot_to_dict(self):
        snap = CausalModelSnapshot(
            node_id="node-2",
            effect=27.3,
            timestamp="2026-01-01T00:00:00+00:00",
            samples_used=45,
        )
        d = snap.to_dict()
        assert d["node_id"] == "node-2"
        assert d["effect"] == 27.3
        assert d["samples_used"] == 45
        assert "timestamp" in d


class TestEffectHistory:

    def test_history_starts_empty(self):
        h = EffectHistory("node-1")
        assert h.size() == 0

    def test_history_appends(self):
        h = EffectHistory("node-1")
        h.append(28.5, "2026-01-01T00:00:00+00:00")
        h.append(29.1, "2026-01-01T00:01:00+00:00")
        assert h.size() == 2

    def test_baseline_none_with_insufficient_history(self):
        h = EffectHistory("node-1")
        h.append(28.5, "2026-01-01T00:00:00+00:00")
        assert h.get_baseline(n=3) is None

    def test_baseline_correct(self):
        h = EffectHistory("node-1")
        for effect in [28.0, 29.0, 30.0, 31.0]:
            h.append(effect, "2026-01-01T00:00:00+00:00")
        baseline = h.get_baseline(n=3)
        assert baseline is not None
        assert abs(baseline - 29.0) < 0.1  # mean of [28, 29, 30]

    def test_get_all_effects(self):
        h = EffectHistory("node-1")
        effects = [28.0, 29.0, 30.0]
        for e in effects:
            h.append(e, "2026-01-01T00:00:00+00:00")
        result = h.get_all_effects()
        assert result == effects

    def test_max_history_respected(self):
        h = EffectHistory("node-1", max_history=5)
        for i in range(10):
            h.append(float(i), "2026-01-01T00:00:00+00:00")
        assert h.size() == 5


class TestDriftEvent:

    def test_drift_event_creation(self):
        event = DriftEvent(
            node_id="node-1",
            previous_effect=10.0,
            current_effect=25.0,
            change_pct=150.0,
            severity="CRITICAL",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert event.node_id == "node-1"
        assert event.severity == "CRITICAL"
        assert event.change_pct == 150.0

    def test_drift_event_to_dict(self):
        event = DriftEvent(
            node_id="node-2",
            previous_effect=28.0,
            current_effect=35.0,
            change_pct=25.0,
            severity="WARNING",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        d = event.to_dict()
        assert d["node_id"] == "node-2"
        assert d["severity"] == "WARNING"
        assert "description" in d
        assert "node-2" in d["description"]


class TestEndToEndCausalPipeline:

    def test_full_pipeline_build_identify_estimate(self):
        df = make_loaded_dataset(n_samples=60)
        results = {}
        for node_id in ["node-1", "node-2", "node-3"]:
            model = NodeCausalModel(node_id=node_id, dataframe=df)
            assert model.build() is True
            assert model.identify() is True
            effect = model.estimate_effect()
            assert effect is not None
            assert abs(effect) > 1.0
            results[node_id] = effect
        assert len(results) == 3

    def test_pipeline_effect_in_expected_range(self):
        df = make_loaded_dataset(n_samples=60, seed=42)
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        model.identify()
        effect = model.estimate_effect()
        assert 20.0 < abs(effect) < 40.0

    def test_cross_node_pipeline_complete(self):
        df = make_loaded_dataset(n_samples=60)
        correlator = DistributedCausalCorrelator(dataset=df)
        results = correlator.run()
        assert len(results["cross_node_effects"]) > 0
        assert results["graph_stats"]["nodes"] == 18
        assert results["graph_stats"]["edges"] == 22
        assert "node-2" in results["causal_chains"]
        assert "node-3" in results["causal_chains"]

    def test_buffer_to_model_pipeline(self):
        buf = RollingTelemetryBuffer(max_samples=100, min_samples=40)
        for i in range(60):
            buf.append(make_sample_dict(i))
        assert buf.is_ready()
        df = buf.get_dataframe()
        assert df is not None
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        built = model.build()
        assert built is True

    def test_drift_detector_no_drift_on_stable_effects(self):
        h = EffectHistory("node-1")
        stable_effects = [28.0, 28.5, 27.8, 28.2, 28.1]
        for e in stable_effects:
            h.append(e, "2026-01-01T00:00:00+00:00")
        baseline = h.get_baseline(n=3)
        current = stable_effects[-1]
        change = abs((current - baseline) / baseline)
        assert change < 0.20

    def test_drift_detector_detects_large_shift(self):
        h = EffectHistory("node-1")
        for e in [28.0, 28.5, 27.8]:
            h.append(e, "2026-01-01T00:00:00+00:00")
        h.append(90.0, "2026-01-01T00:04:00+00:00")
        baseline = h.get_baseline(n=3)
        current = 90.0
        change = abs((current - baseline) / baseline)
        assert change > 0.50

    def test_multiple_retrains_produce_consistent_effects(self):
        effects = []
        for seed in range(5):
            df = make_loaded_dataset(n_samples=60, seed=seed)
            model = NodeCausalModel(node_id="node-1", dataframe=df)
            model.build()
            model.identify()
            effect = model.estimate_effect()
            if effect is not None:
                effects.append(effect)
        assert len(effects) == 5
        mean_effect = sum(effects) / len(effects)
        for e in effects:
            assert abs(e - mean_effect) < 10.0

    def test_causal_chain_traversal(self):
        df = make_loaded_dataset(n_samples=60)
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        chains = graph.find_causal_chains("node-2", "avg_query_duration_ms")
        assert isinstance(chains, list)
        if chains:
            for chain in chains:
                assert "root_cause" in chain
                assert "symptom" in chain
                assert "path" in chain
                assert len(chain["path"]) >= 2

    def test_snapshot_lifecycle(self):
        snap = CausalModelSnapshot(
            node_id="node-3",
            effect=27.9,
            timestamp="2026-01-01T00:00:00+00:00",
            samples_used=60,
        )
        d = snap.to_dict()
        assert d["node_id"] == "node-3"
        assert d["effect"] == 27.9
        assert d["samples_used"] == 60

    def test_effect_history_baseline_excludes_current(self):
        h = EffectHistory("node-1")
        h.append(28.0, "t1")
        h.append(28.5, "t2")
        h.append(29.0, "t3")
        h.append(500.0, "t4")
        baseline = h.get_baseline(n=3)
        assert baseline is not None
        assert baseline < 100.0