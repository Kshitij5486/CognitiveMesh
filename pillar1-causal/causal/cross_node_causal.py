import logging
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
from dowhy import CausalModel

from dotenv import load_dotenv
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "telemetry")
)
sys.path.append(os.path.dirname(__file__))

from collector import TelemetryCollectorManager
from graph_builder import TelemetrySampler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.causal.cross_node")


class CrossNodeCausalGraph:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.graph: Optional[nx.DiGraph] = None
        self.models: dict = {}
        self.effects: dict = {}

    def _build_cross_node_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()

        nodes = ["node_1", "node_2", "node_3"]
        metrics = [
            "active_queries",
            "avg_query_duration_ms",
            "lock_count",
            "blocked_locks",
            "buffers_backend",
            "buffers_alloc",
        ]

        for node in nodes:
            for metric in metrics:
                g.add_node(f"{node}_{metric}")

        for node in nodes:
            n = node
            g.add_edges_from([
                (f"{n}_active_queries",         f"{n}_avg_query_duration_ms"),
                (f"{n}_active_queries",         f"{n}_lock_count"),
                (f"{n}_lock_count",             f"{n}_blocked_locks"),
                (f"{n}_blocked_locks",          f"{n}_avg_query_duration_ms"),
                (f"{n}_buffers_backend",        f"{n}_avg_query_duration_ms"),
            ])

        g.add_edges_from([
            ("node_1_active_queries", "node_2_active_queries"),
            ("node_1_active_queries", "node_3_active_queries"),
            ("node_1_lock_count",     "node_2_lock_count"),
            ("node_1_lock_count",     "node_3_lock_count"),
            ("node_2_active_queries", "node_3_active_queries"),
            ("node_1_avg_query_duration_ms", "node_2_avg_query_duration_ms"),
            ("node_2_avg_query_duration_ms", "node_3_avg_query_duration_ms"),
        ])

        return g

    def build(self) -> bool:
        logger.info("Building cross-node causal graph")
        self.graph = self._build_cross_node_graph()
        logger.info(
            "Cross-node graph built nodes=%d edges=%d",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return True

    def estimate_cross_node_effect(
        self,
        treatment: str,
        outcome: str,
    ) -> Optional[float]:
        if self.graph is None:
            logger.error("Graph not built. Call build() first.")
            return None

        if treatment not in self.df.columns or outcome not in self.df.columns:
            logger.warning(
                "treatment=%s or outcome=%s not in dataset", treatment, outcome
            )
            return None

        if self.df[treatment].std() < 0.001:
            logger.info(
                "Treatment %s has near-zero variance skipping", treatment
            )
            return None

        if self.df[outcome].std() < 0.001:
            logger.info(
                "Outcome %s has near-zero variance skipping", outcome
            )
            return None

        try:
            model = CausalModel(
                data=self.df,
                treatment=treatment,
                outcome=outcome,
                graph=self.graph,
            )
            estimand = model.identify_effect(
                proceed_when_unidentifiable=True
            )
            estimate = model.estimate_effect(
                estimand,
                method_name="backdoor.linear_regression",
            )
            effect = float(estimate.value)
            key = f"{treatment}->{outcome}"
            self.models[key] = model
            self.effects[key] = effect
            logger.info(
                "Cross-node effect %s -> %s = %.4f",
                treatment,
                outcome,
                effect,
            )
            return effect
        except Exception as e:
            logger.warning(
                "Could not estimate %s -> %s error=%s",
                treatment,
                outcome,
                e,
            )
            return None

    def estimate_all_cross_node_effects(self) -> dict:
        cross_node_pairs = [
            ("node_1_active_queries", "node_2_active_queries"),
            ("node_1_active_queries", "node_3_active_queries"),
            ("node_2_active_queries", "node_3_active_queries"),
            ("node_1_active_queries", "node_2_avg_query_duration_ms"),
            ("node_1_active_queries", "node_3_avg_query_duration_ms"),
            ("node_1_lock_count",     "node_2_lock_count"),
            ("node_1_lock_count",     "node_3_lock_count"),
            ("node_1_avg_query_duration_ms", "node_2_avg_query_duration_ms"),
            ("node_1_avg_query_duration_ms", "node_3_avg_query_duration_ms"),
            ("node_2_avg_query_duration_ms", "node_3_avg_query_duration_ms"),
        ]

        results = {}
        for treatment, outcome in cross_node_pairs:
            effect = self.estimate_cross_node_effect(treatment, outcome)
            if effect is not None:
                results[f"{treatment}->{outcome}"] = {
                    "treatment": treatment,
                    "outcome": outcome,
                    "effect": round(effect, 6),
                    "significant": abs(effect) > 0.1,
                }

        return results

    def find_causal_chains(self, symptom_node: str, symptom_metric: str) -> list:
        if self.graph is None:
            return []

        symptom = f"{symptom_node.replace('-', '_')}_{symptom_metric}"
        if symptom not in self.graph.nodes:
            logger.warning("Symptom node %s not in graph", symptom)
            return []

        chains = []
        ancestors = nx.ancestors(self.graph, symptom)

        for ancestor in ancestors:
            if ancestor == symptom:
                continue
            try:
                paths = list(nx.all_simple_paths(
                    self.graph,
                    source=ancestor,
                    target=symptom,
                    cutoff=4,
                ))
                for path in paths:
                    if len(path) >= 2:
                        key = f"{ancestor}->{symptom}"
                        effect = self.effects.get(key)
                        chains.append({
                            "root_cause": ancestor,
                            "symptom": symptom,
                            "path": path,
                            "path_length": len(path),
                            "estimated_effect": effect,
                            "cross_node": ancestor.split("_")[1] != symptom.split("_")[1],
                        })
            except Exception:
                continue

        chains.sort(key=lambda x: (
            x["cross_node"],
            x["estimated_effect"] is not None,
            abs(x["estimated_effect"]) if x["estimated_effect"] else 0,
        ), reverse=True)

        return chains[:10]

    def explain_cross_node(self, effects: dict) -> list:
        explanations = []
        for key, data in effects.items():
            if not data["significant"]:
                continue
            treatment_node = data["treatment"].split("_")[1]
            outcome_node = data["outcome"].split("_")[1]
            treatment_metric = "_".join(data["treatment"].split("_")[2:])
            outcome_metric = "_".join(data["outcome"].split("_")[2:])
            direction = "increases" if data["effect"] > 0 else "decreases"
            explanations.append({
                "finding": (
                    f"node-{treatment_node} {treatment_metric} causally "
                    f"{direction} node-{outcome_node} {outcome_metric} "
                    f"by {abs(data['effect']):.4f} units"
                ),
                "effect": data["effect"],
                "treatment": data["treatment"],
                "outcome": data["outcome"],
            })
        explanations.sort(key=lambda x: abs(x["effect"]), reverse=True)
        return explanations


class DistributedCausalCorrelator:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.cross_node_graph = CrossNodeCausalGraph(dataframe=dataset)

    def run(self) -> dict:
        logger.info("Building cross-node causal graph")
        self.cross_node_graph.build()

        logger.info("Estimating all cross-node causal effects")
        effects = self.cross_node_graph.estimate_all_cross_node_effects()

        logger.info("Finding causal chains for node-2 latency symptom")
        chains_node2 = self.cross_node_graph.find_causal_chains(
            "node-2", "avg_query_duration_ms"
        )

        logger.info("Finding causal chains for node-3 latency symptom")
        chains_node3 = self.cross_node_graph.find_causal_chains(
            "node-3", "avg_query_duration_ms"
        )

        explanations = self.cross_node_graph.explain_cross_node(effects)

        logger.info("=" * 60)
        logger.info("CROSS-NODE CAUSAL ANALYSIS RESULTS")
        logger.info("=" * 60)

        logger.info("Significant cross-node causal effects:")
        for exp in explanations:
            logger.info("  %s", exp["finding"])

        logger.info("Causal chains explaining node-2 latency:")
        for chain in chains_node2[:5]:
            logger.info(
                "  %s -> ... -> %s (length=%d cross_node=%s effect=%s)",
                chain["root_cause"],
                chain["symptom"],
                chain["path_length"],
                chain["cross_node"],
                f"{chain['estimated_effect']:.4f}"
                if chain["estimated_effect"] else "unknown",
            )

        logger.info("Causal chains explaining node-3 latency:")
        for chain in chains_node3[:5]:
            logger.info(
                "  %s -> ... -> %s (length=%d cross_node=%s effect=%s)",
                chain["root_cause"],
                chain["symptom"],
                chain["path_length"],
                chain["cross_node"],
                f"{chain['estimated_effect']:.4f}"
                if chain["estimated_effect"] else "unknown",
            )

        return {
            "cross_node_effects": effects,
            "explanations": explanations,
            "causal_chains": {
                "node-2": chains_node2,
                "node-3": chains_node3,
            },
            "graph_stats": {
                "nodes": self.cross_node_graph.graph.number_of_nodes(),
                "edges": self.cross_node_graph.graph.number_of_edges(),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


if __name__ == "__main__":
    logger.info("Collecting telemetry data for cross-node analysis")
    sampler = TelemetrySampler()
    dataset = sampler.collect_dataset(n_samples=60, interval_seconds=2.0)
    sampler.close()

    correlator = DistributedCausalCorrelator(dataset=dataset)
    results = correlator.run()

    logger.info("Cross-node effects found: %d", len(results["cross_node_effects"]))
    logger.info("Significant explanations: %d", len(results["explanations"]))
    logger.info(
        "Graph: %d nodes, %d edges",
        results["graph_stats"]["nodes"],
        results["graph_stats"]["edges"],
    )