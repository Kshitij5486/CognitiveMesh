import logging
import statistics
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from graph_builder import NodeCausalModel
from streaming_updater import RollingTelemetryBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.causal.optimize")


def make_loaded_dataset(n_samples: int, seed: int = 42) -> pd.DataFrame:
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


def benchmark_retrain_by_sample_size(
    sample_sizes: list,
    n_runs: int = 5,
) -> dict:
    logger.info(
        "Benchmarking retrain latency by sample size n_runs=%d", n_runs
    )
    results = {}
    for n in sample_sizes:
        latencies = []
        for i in range(n_runs):
            df = make_loaded_dataset(n_samples=n, seed=i)
            start = time.perf_counter()
            for node_id in ["node-1", "node-2", "node-3"]:
                model = NodeCausalModel(node_id=node_id, dataframe=df)
                model.build()
                model.identify()
                model.estimate_effect()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        results[n] = {
            "samples": n,
            "mean_ms": round(statistics.mean(latencies), 2),
            "min_ms":  round(min(latencies), 2),
            "max_ms":  round(max(latencies), 2),
            "p95_ms":  round(sorted(latencies)[int(n_runs * 0.95)], 2),
        }
        logger.info(
            "samples=%-4d  mean=%.2fms  min=%.2fms  max=%.2fms",
            n,
            results[n]["mean_ms"],
            results[n]["min_ms"],
            results[n]["max_ms"],
        )
    return results


def benchmark_buffer_to_dataframe(
    buffer_sizes: list,
    n_runs: int = 10,
) -> dict:
    logger.info(
        "Benchmarking buffer-to-dataframe conversion n_runs=%d", n_runs
    )
    results = {}
    for size in buffer_sizes:
        latencies = []
        for _ in range(n_runs):
            buf = RollingTelemetryBuffer(
                max_samples=size + 10,
                min_samples=size,
            )
            for i in range(size):
                buf.append(make_sample_dict(i))
            start = time.perf_counter()
            df = buf.get_dataframe()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert df is not None
            assert len(df) == size

        results[size] = {
            "buffer_size": size,
            "mean_ms": round(statistics.mean(latencies), 4),
            "min_ms":  round(min(latencies), 4),
            "max_ms":  round(max(latencies), 4),
        }
        logger.info(
            "buffer_size=%-4d  mean=%.4fms  min=%.4fms  max=%.4fms",
            size,
            results[size]["mean_ms"],
            results[size]["min_ms"],
            results[size]["max_ms"],
        )
    return results


def benchmark_single_node_vs_all_nodes(n_runs: int = 5) -> dict:
    logger.info("Benchmarking single node vs all nodes retrain")
    df = make_loaded_dataset(n_samples=60)

    single_latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model = NodeCausalModel(node_id="node-1", dataframe=df)
        model.build()
        model.identify()
        model.estimate_effect()
        elapsed_ms = (time.perf_counter() - start) * 1000
        single_latencies.append(elapsed_ms)

    all_latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for node_id in ["node-1", "node-2", "node-3"]:
            model = NodeCausalModel(node_id=node_id, dataframe=df)
            model.build()
            model.identify()
            model.estimate_effect()
        elapsed_ms = (time.perf_counter() - start) * 1000
        all_latencies.append(elapsed_ms)

    single_mean = statistics.mean(single_latencies)
    all_mean = statistics.mean(all_latencies)
    parallelization_opportunity = all_mean / single_mean

    result = {
        "single_node_mean_ms": round(single_mean, 2),
        "all_nodes_mean_ms": round(all_mean, 2),
        "parallelization_ratio": round(parallelization_opportunity, 2),
        "theoretical_parallel_speedup": round(parallelization_opportunity, 2),
        "note": (
            f"All-nodes retrain is {parallelization_opportunity:.1f}x single-node. "
            f"Parallelizing across 3 threads could reduce to ~{single_mean:.0f}ms."
        ),
    }
    logger.info(
        "single_node=%.2fms  all_nodes=%.2fms  ratio=%.2fx",
        result["single_node_mean_ms"],
        result["all_nodes_mean_ms"],
        result["parallelization_ratio"],
    )
    logger.info("  %s", result["note"])
    return result


def benchmark_parallel_retrain(n_runs: int = 5) -> dict:
    import concurrent.futures
    logger.info(
        "Benchmarking parallel retrain across 3 nodes n_runs=%d", n_runs
    )

    def train_single_node(node_id: str, df: pd.DataFrame) -> float:
        model = NodeCausalModel(node_id=node_id, dataframe=df)
        model.build()
        model.identify()
        effect = model.estimate_effect()
        return effect or 0.0

    sequential_latencies = []
    parallel_latencies = []

    for i in range(n_runs):
        df = make_loaded_dataset(n_samples=60, seed=i)

        start = time.perf_counter()
        for node_id in ["node-1", "node-2", "node-3"]:
            train_single_node(node_id, df)
        sequential_latencies.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(train_single_node, node_id, df): node_id
                for node_id in ["node-1", "node-2", "node-3"]
            }
            concurrent.futures.wait(futures)
        parallel_latencies.append((time.perf_counter() - start) * 1000)

    seq_mean = statistics.mean(sequential_latencies)
    par_mean = statistics.mean(parallel_latencies)
    speedup = seq_mean / par_mean if par_mean > 0 else 1.0

    result = {
        "sequential_mean_ms": round(seq_mean, 2),
        "parallel_mean_ms":   round(par_mean, 2),
        "speedup":            round(speedup, 2),
        "parallel_wins":      par_mean < seq_mean,
    }
    logger.info(
        "sequential=%.2fms  parallel=%.2fms  speedup=%.2fx  parallel_wins=%s",
        result["sequential_mean_ms"],
        result["parallel_mean_ms"],
        result["speedup"],
        result["parallel_wins"],
    )
    return result


def benchmark_rolling_window_effect(n_runs: int = 3) -> dict:
    logger.info("Benchmarking rolling window effect on estimate stability")
    window_sizes = [30, 60, 100, 150, 200]
    results = {}

    for window in window_sizes:
        effects = []
        for seed in range(n_runs):
            df = make_loaded_dataset(n_samples=window, seed=seed)
            model = NodeCausalModel(node_id="node-1", dataframe=df)
            model.build()
            model.identify()
            effect = model.estimate_effect()
            if effect is not None:
                effects.append(effect)

        if effects:
            mean_e = statistics.mean(effects)
            stddev_e = statistics.stdev(effects) if len(effects) > 1 else 0.0
            results[window] = {
                "window_size": window,
                "mean_effect": round(mean_e, 4),
                "stddev_effect": round(stddev_e, 4),
                "stability": "high" if stddev_e < 1.0 else "medium" if stddev_e < 5.0 else "low",
            }
            logger.info(
                "window=%-4d  mean=%.4f  stddev=%.4f  stability=%s",
                window,
                mean_e,
                stddev_e,
                results[window]["stability"],
            )

    return results


def run_full_optimization():
    logger.info("=" * 60)
    logger.info("CognitiveMesh Sprint 3 Performance Optimization")
    logger.info("Timestamp: %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    logger.info("\n--- 1. Retrain latency by sample size ---")
    retrain_results = benchmark_retrain_by_sample_size(
        sample_sizes=[30, 60, 100, 150, 200],
        n_runs=3,
    )

    logger.info("\n--- 2. Buffer-to-dataframe conversion ---")
    buffer_results = benchmark_buffer_to_dataframe(
        buffer_sizes=[30, 60, 100, 200],
        n_runs=5,
    )

    logger.info("\n--- 3. Single node vs all nodes ---")
    parallel_opportunity = benchmark_single_node_vs_all_nodes(n_runs=3)

    logger.info("\n--- 4. Sequential vs parallel retrain ---")
    parallel_results = benchmark_parallel_retrain(n_runs=3)

    logger.info("\n--- 5. Rolling window effect on stability ---")
    window_results = benchmark_rolling_window_effect(n_runs=3)

    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 60)

    logger.info("Retrain latency by sample size:")
    for n, r in retrain_results.items():
        logger.info(
            "  samples=%-4d  mean=%.2fms  max=%.2fms",
            n, r["mean_ms"], r["max_ms"]
        )

    logger.info("Buffer-to-dataframe conversion:")
    for size, r in buffer_results.items():
        logger.info(
            "  buffer=%-4d  mean=%.4fms",
            size, r["mean_ms"]
        )

    logger.info("Parallelization opportunity:")
    logger.info(
        "  sequential=%.2fms  parallel=%.2fms  speedup=%.2fx",
        parallel_results["sequential_mean_ms"],
        parallel_results["parallel_mean_ms"],
        parallel_results["speedup"],
    )

    logger.info("Rolling window stability:")
    for window, r in window_results.items():
        logger.info(
            "  window=%-4d  stddev=%.4f  stability=%s",
            window, r["stddev_effect"], r["stability"]
        )

    logger.info("=" * 60)
    logger.info("KEY FINDINGS:")

    min_retrain = min(retrain_results.values(), key=lambda x: x["mean_ms"])
    logger.info(
        "  Fastest retrain: %d samples at %.2fms",
        min_retrain["samples"], min_retrain["mean_ms"]
    )

    if parallel_results["parallel_wins"]:
        logger.info(
            "  Parallel retrain wins: %.2fx speedup",
            parallel_results["speedup"]
        )
    else:
        logger.info(
            "  Sequential retrain preferred: GIL overhead exceeds benefit"
        )

    best_window = min(window_results.values(), key=lambda x: x["stddev_effect"])
    logger.info(
        "  Most stable window: %d samples (stddev=%.4f)",
        best_window["window_size"], best_window["stddev_effect"]
    )

    logger.info("=" * 60)

    return {
        "retrain_by_sample_size": retrain_results,
        "buffer_conversion": buffer_results,
        "parallel_opportunity": parallel_opportunity,
        "parallel_retrain": parallel_results,
        "window_stability": window_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    run_full_optimization()