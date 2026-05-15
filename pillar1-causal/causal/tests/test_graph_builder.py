import sys
import os
import pytest
import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "telemetry"))

from graph_builder import NodeCausalModel, DistributedCausalEngine
from cross_node_causal import CrossNodeCausalGraph, DistributedCausalCorrelator


def make_idle_dataset(n_samples: int = 60) -> pd.DataFrame:
    np.random.seed(42)
    data = {}
    for node in ["node_1", "node_2", "node_3"]:
        data[f"{node}_buffers_backend"]        = np.zeros(n_samples)
        data[f"{node}_buffers_alloc"]          = np.full(n_samples, 936.0)
        data[f"{node}_checkpoints_req"]        = np.ones(n_samples)
        data[f"{node}_avg_query_duration_ms"]  = np.random.normal(0.5, 0.1, n_samples)
        data[f"{node}_active_queries"]         = np.zeros(n_samples)
        data[f"{node}_lock_count"]             = np.zeros(n_samples)
        data[f"{node}_blocked_locks"]          = np.zeros(n_samples)
    return pd.DataFrame(data)


def make_loaded_dataset(n_samples: int = 60) -> pd.DataFrame:
    np.random.seed(42)
    data = {}
    active = np.random.randint(0, 10, n_samples).astype(float)
    for node in ["node_1", "node_2", "node_3"]:
        node_active = active + np.random.normal(0, 0.5, n_samples)
        node_active = np.clip(node_active, 0, None)
        data[f"{node}_buffers_backend"]       = np.random.randint(200, 400, n_samples).astype(float)
        data[f"{node}_buffers_alloc"]         = np.random.randint(800, 1000, n_samples).astype(float)
        data[f"{node}_checkpoints_req"]       = np.random.randint(0, 3, n_samples).astype(float)
        data[f"{node}_avg_query_duration_ms"] = (
            node_active * 28.0
            + np.random.normal(0, 2.0, n_samples)
        )
        data[f"{node}_active_queries"]  = node_active
        data[f"{node}_lock_count"]      = node_active * 2 + np.random.normal(0, 0.5, n_samples)
        data[f"{node}_blocked_locks"]   = np.random.randint(0, 2, n_samples).astype(float)
    return pd.DataFrame(data)


class TestNodeCausalModel:

    def test_build_succeeds_with_loaded_data(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        result = model.build()
        assert result is True

    def test_build_returns_false_when_no_columns(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        result = model.build()
        assert result is False

    def test_node_safe_replaces_hyphens(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        assert model.node_safe == "node_1"
        assert "-" not in model.node_safe

    def test_identify_succeeds_after_build(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        result = model.identify()
        assert result is True

    def test_identify_fails_without_build(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        result = model.identify()
        assert result is False

    def test_estimate_effect_returns_float(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        model.identify()
        effect = model.estimate_effect()
        assert effect is not None
        assert isinstance(effect, float)

    def test_estimate_effect_nonzero_under_load(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        model.identify()
        effect = model.estimate_effect()
        assert abs(effect) > 1.0

    def test_estimate_effect_returns_none_without_identify(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        effect = model.estimate_effect()
        assert effect is None

    def test_summary_structure(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-2", dataframe=df)
        model.build()
        model.identify()
        model.estimate_effect()
        summary = model.summary()
        assert "node_id" in summary
        assert "causal_effect" in summary
        assert "model_built" in summary
        assert "effect_identified" in summary
        assert "timestamp" in summary
        assert summary["node_id"] == "node-2"
        assert summary["model_built"] is True
        assert summary["effect_identified"] is True
        assert summary["causal_effect"] is not None

    def test_causal_graph_is_directed(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        graph = model._build_causal_graph()
        assert isinstance(graph, nx.DiGraph)

    def test_causal_graph_has_correct_edges(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        graph = model._build_causal_graph()
        assert graph.has_edge(
            "node_1_active_queries",
            "node_1_avg_query_duration_ms"
        )
        assert graph.has_edge(
            "node_1_lock_count",
            "node_1_blocked_locks"
        )

    def test_all_three_nodes_build_successfully(self):
        df = make_loaded_dataset()
        for node_id in ["node-1", "node-2", "node-3"]:
            model = NodeCausalModel(node_id=node_id, dataframe=df)
            assert model.build() is True

    def test_refute_returns_dict(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        model.identify()
        model.estimate_effect()
        result = model.refute()
        assert isinstance(result, dict)

    def test_refute_returns_empty_without_estimate(self):
        df = make_loaded_dataset()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        result = model.refute()
        assert result == {}


class TestCrossNodeCausalGraph:

    def test_build_creates_graph(self):
        df = make_loaded_dataset()
        graph = CrossNodeCausalGraph(dataframe=df)
        result = graph.build()
        assert result is True
        assert graph.graph is not None

    def test_graph_has_correct_node_count(self):
        df = make_loaded_dataset()
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        assert graph.graph.number_of_nodes() == 18

    def test_graph_has_cross_node_edges(self):
        df = make_loaded_dataset()
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        assert graph.graph.has_edge(
            "node_1_active_queries",
            "node_2_active_queries"
        )
        assert graph.graph.has_edge(
            "node_1_avg_query_duration_ms",
            "node_2_avg_query_duration_ms"
        )

    def test_estimate_cross_node_effect_returns_float(self):
        df = make_loaded_dataset()
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        effect = graph.estimate_cross_node_effect(
            treatment="node_1_active_queries",
            outcome="node_2_active_queries",
        )
        assert effect is not None
        assert isinstance(effect, float)

    def test_estimate_returns_none_for_missing_columns(self):
        df = make_loaded_dataset()
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        effect = graph.estimate_cross_node_effect(
            treatment="nonexistent_col",
            outcome="node_2_active_queries",
        )
        assert effect is None

    def test_estimate_all_effects_returns_dict(self):
        df = make_loaded_dataset()
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        effects = graph.estimate_all_cross_node_effects()
        assert isinstance(effects, dict)
        assert len(effects) > 0

    def test_find_causal_chains_returns_list(self):
        df = make_loaded_dataset()
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        chains = graph.find_causal_chains("node-2", "avg_query_duration_ms")
        assert isinstance(chains, list)

    def test_causal_chains_contain_cross_node_paths(self):
        df = make_loaded_dataset()
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        graph.estimate_all_cross_node_effects()
        chains = graph.find_causal_chains("node-2", "avg_query_duration_ms")
        cross_node_chains = [c for c in chains if c["cross_node"]]
        assert len(cross_node_chains) > 0

    def test_explain_cross_node_returns_list(self):
        df = make_loaded_dataset()
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        effects = graph.estimate_all_cross_node_effects()
        explanations = graph.explain_cross_node(effects)
        assert isinstance(explanations, list)

    def test_explain_sorted_by_effect_magnitude(self):
        df = make_loaded_dataset()
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        effects = graph.estimate_all_cross_node_effects()
        explanations = graph.explain_cross_node(effects)
        if len(explanations) >= 2:
            assert abs(explanations[0]["effect"]) >= abs(explanations[1]["effect"])


class TestDistributedCausalCorrelator:

    def test_run_returns_complete_result(self):
        df = make_loaded_dataset()
        correlator = DistributedCausalCorrelator(dataset=df)
        result = correlator.run()
        assert "cross_node_effects" in result
        assert "explanations" in result
        assert "causal_chains" in result
        assert "graph_stats" in result
        assert "timestamp" in result

    def test_graph_stats_correct(self):
        df = make_loaded_dataset()
        correlator = DistributedCausalCorrelator(dataset=df)
        result = correlator.run()
        assert result["graph_stats"]["nodes"] == 18
        assert result["graph_stats"]["edges"] == 22

    def test_causal_chains_for_both_nodes(self):
        df = make_loaded_dataset()
        correlator = DistributedCausalCorrelator(dataset=df)
        result = correlator.run()
        assert "node-2" in result["causal_chains"]
        assert "node-3" in result["causal_chains"]