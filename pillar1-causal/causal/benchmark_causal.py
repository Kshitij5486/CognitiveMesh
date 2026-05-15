import logging
import statistics
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from graph_builder import NodeCausalModel, DistributedCausalEngine
from cross_node_causal import CrossNodeCausalGraph, DistributedCausalCorrelator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.causal.benchmark")


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


def benchmark_model_build_time(n_runs: int = 10) -> dict:
    logger.info("Benchmarking model build time over %d runs", n_runs)
    latencies = []
    for i in range(n_runs):
        df = make_loaded_dataset(seed=i)
        start = time.perf_counter()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        logger.debug("Run %d build_time=%.3fms", i, elapsed_ms)

    result = {
        "metric": "model_build_time_ms",
        "runs": n_runs,
        "mean_ms":   round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "min_ms":    round(min(latencies), 3),
        "max_ms":    round(max(latencies), 3),
        "p95_ms":    round(sorted(latencies)[int(n_runs * 0.95)], 3),
        "stddev_ms": round(statistics.stdev(latencies), 3),
    }
    logger.info(
        "model_build mean=%.3fms p95=%.3fms",
        result["mean_ms"], result["p95_ms"]
    )
    return result


def benchmark_identify_time(n_runs: int = 10) -> dict:
    logger.info("Benchmarking causal identification time over %d runs", n_runs)
    latencies = []
    for i in range(n_runs):
        df = make_loaded_dataset(seed=i)
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        start = time.perf_counter()
        model.identify()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    result = {
        "metric": "identify_time_ms",
        "runs": n_runs,
        "mean_ms":   round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "min_ms":    round(min(latencies), 3),
        "max_ms":    round(max(latencies), 3),
        "p95_ms":    round(sorted(latencies)[int(n_runs * 0.95)], 3),
        "stddev_ms": round(statistics.stdev(latencies), 3),
    }
    logger.info(
        "identify mean=%.3fms p95=%.3fms",
        result["mean_ms"], result["p95_ms"]
    )
    return result


def benchmark_estimate_time(n_runs: int = 10) -> dict:
    logger.info("Benchmarking effect estimation time over %d runs", n_runs)
    latencies = []
    for i in range(n_runs):
        df = make_loaded_dataset(seed=i)
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        model.identify()
        start = time.perf_counter()
        model.estimate_effect()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    result = {
        "metric": "estimate_time_ms",
        "runs": n_runs,
        "mean_ms":   round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "min_ms":    round(min(latencies), 3),
        "max_ms":    round(max(latencies), 3),
        "p95_ms":    round(sorted(latencies)[int(n_runs * 0.95)], 3),
        "stddev_ms": round(statistics.stdev(latencies), 3),
    }
    logger.info(
        "estimate mean=%.3fms p95=%.3fms",
        result["mean_ms"], result["p95_ms"]
    )
    return result


def benchmark_full_pipeline_time(n_runs: int = 5) -> dict:
    logger.info(
        "Benchmarking full 3-node causal pipeline over %d runs", n_runs
    )
    latencies = []
    for i in range(n_runs):
        df = make_loaded_dataset(seed=i)
        start = time.perf_counter()
        for node_id in ["node-1", "node-2", "node-3"]:
            model = NodeCausalModel(node_id=node_id, dataframe=df)
            model.build()
            model.identify()
            model.estimate_effect()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        logger.debug("Run %d full_pipeline=%.3fms", i, elapsed_ms)

    result = {
        "metric": "full_3node_pipeline_ms",
        "runs": n_runs,
        "mean_ms":   round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "min_ms":    round(min(latencies), 3),
        "max_ms":    round(max(latencies), 3),
        "stddev_ms": round(statistics.stdev(latencies), 3),
    }
    logger.info(
        "full_pipeline mean=%.3fms max=%.3fms",
        result["mean_ms"], result["max_ms"]
    )
    return result


def benchmark_cross_node_build(n_runs: int = 5) -> dict:
    logger.info(
        "Benchmarking cross-node graph build over %d runs", n_runs
    )
    latencies = []
    for i in range(n_runs):
        df = make_loaded_dataset(seed=i)
        start = time.perf_counter()
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    result = {
        "metric": "cross_node_graph_build_ms",
        "runs": n_runs,
        "mean_ms":   round(statistics.mean(latencies), 3),
        "min_ms":    round(min(latencies), 3),
        "max_ms":    round(max(latencies), 3),
        "stddev_ms": round(statistics.stdev(latencies), 3),
    }
    logger.info(
        "cross_node_build mean=%.3fms max=%.3fms",
        result["mean_ms"], result["max_ms"]
    )
    return result


def benchmark_cross_node_estimation(n_runs: int = 3) -> dict:
    logger.info(
        "Benchmarking cross-node effect estimation over %d runs", n_runs
    )
    latencies = []
    for i in range(n_runs):
        df = make_loaded_dataset(seed=i)
        graph = CrossNodeCausalGraph(dataframe=df)
        graph.build()
        start = time.perf_counter()
        graph.estimate_all_cross_node_effects()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        logger.debug(
            "Run %d cross_node_estimation=%.3fms", i, elapsed_ms
        )

    result = {
        "metric": "cross_node_estimation_ms",
        "runs": n_runs,
        "mean_ms":   round(statistics.mean(latencies), 3),
        "min_ms":    round(min(latencies), 3),
        "max_ms":    round(max(latencies), 3),
        "stddev_ms": round(statistics.stdev(latencies) if n_runs > 1 else 0, 3),
    }
    logger.info(
        "cross_node_estimation mean=%.3fms max=%.3fms",
        result["mean_ms"], result["max_ms"]
    )
    return result


def benchmark_causal_effect_consistency(n_runs: int = 10) -> dict:
    logger.info(
        "Benchmarking causal effect consistency over %d runs", n_runs
    )
    effects = []
    for i in range(n_runs):
        df = make_loaded_dataset(seed=42)
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        model.identify()
        effect = model.estimate_effect()
        if effect is not None:
            effects.append(effect)

    result = {
        "metric": "causal_effect_consistency",
        "runs": n_runs,
        "mean_effect":   round(statistics.mean(effects), 6),
        "stddev_effect": round(statistics.stdev(effects) if len(effects) > 1 else 0, 6),
        "min_effect":    round(min(effects), 6),
        "max_effect":    round(max(effects), 6),
        "consistent":    statistics.stdev(effects) < 0.001 if len(effects) > 1 else True,
    }
    logger.info(
        "effect_consistency mean=%.6f stddev=%.6f consistent=%s",
        result["mean_effect"],
        result["stddev_effect"],
        result["consistent"],
    )
    return result


def run_full_benchmark():
    logger.info("Starting CognitiveMesh causal engine benchmark")
    logger.info("Timestamp: %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    results = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": [],
    }

    build_result = benchmark_model_build_time(n_runs=10)
    results["metrics"].append(build_result)

    identify_result = benchmark_identify_time(n_runs=10)
    results["metrics"].append(identify_result)

    estimate_result = benchmark_estimate_time(n_runs=10)
    results["metrics"].append(estimate_result)

    pipeline_result = benchmark_full_pipeline_time(n_runs=5)
    results["metrics"].append(pipeline_result)

    cross_build_result = benchmark_cross_node_build(n_runs=5)
    results["metrics"].append(cross_build_result)

    cross_est_result = benchmark_cross_node_estimation(n_runs=3)
    results["metrics"].append(cross_est_result)

    consistency_result = benchmark_causal_effect_consistency(n_runs=10)
    results["metrics"].append(consistency_result)

    logger.info("=" * 60)
    logger.info("CAUSAL ENGINE BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info(
        "model_build          mean=%-10.3fms  p95=%-10.3fms",
        build_result["mean_ms"],
        build_result["p95_ms"],
    )
    logger.info(
        "identify_effect      mean=%-10.3fms  p95=%-10.3fms",
        identify_result["mean_ms"],
        identify_result["p95_ms"],
    )
    logger.info(
        "estimate_effect      mean=%-10.3fms  p95=%-10.3fms",
        estimate_result["mean_ms"],
        estimate_result["p95_ms"],
    )
    logger.info(
        "full_3node_pipeline  mean=%-10.3fms  max=%-10.3fms",
        pipeline_result["mean_ms"],
        pipeline_result["max_ms"],
    )
    logger.info(
        "cross_node_build     mean=%-10.3fms  max=%-10.3fms",
        cross_build_result["mean_ms"],
        cross_build_result["max_ms"],
    )
    logger.info(
        "cross_node_estimate  mean=%-10.3fms  max=%-10.3fms",
        cross_est_result["mean_ms"],
        cross_est_result["max_ms"],
    )
    logger.info(
        "effect_consistency   mean=%-12.6f  stddev=%-12.6f  consistent=%s",
        consistency_result["mean_effect"],
        consistency_result["stddev_effect"],
        consistency_result["consistent"],
    )
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_full_benchmark()