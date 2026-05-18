import logging
import threading
import time
import statistics
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry,
)

from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "telemetry"))
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.observability.causal_adapter")

ALL_NODES = ["node-1", "node-2", "node-3"]


class CausalMetricsAdapter:
    """
    Deep causal metrics adapter.

    Tracks:
    - Per-node causal effect magnitude history
    - Effect drift events (when effect changes > threshold)
    - Retrain cycle latency distribution
    - Model confidence (inverse of effect variance)
    - Causal graph stability score
    - Buffer utilization ratio
    - Cross-node effect correlation
    """

    DRIFT_THRESHOLD_MS = 2.0
    HISTORY_WINDOW = 60
    EFFECT_BUCKETS = [
        5.0, 10.0, 15.0, 20.0, 25.0,
        30.0, 35.0, 40.0, 50.0, 75.0, 100.0,
    ]
    RETRAIN_LATENCY_BUCKETS = [
        10.0, 25.0, 50.0, 75.0, 100.0,
        150.0, 200.0, 300.0, 500.0,
    ]

    def __init__(
        self,
        updater,
        registry: CollectorRegistry,
        collection_interval: float = 15.0,
    ):
        self.updater = updater
        self.registry = registry
        self.collection_interval = collection_interval

        # Effect history per node for drift + variance tracking
        self._effect_history: dict[str, deque] = {
            node: deque(maxlen=self.HISTORY_WINDOW)
            for node in ALL_NODES
        }
        self._last_effects: dict[str, float] = {}
        self._retrain_count_last: int = 0
        self._retrain_times: deque = deque(maxlen=50)

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._collection_count = 0
        self._start_time: Optional[float] = None

        self._init_metrics()

    def _init_metrics(self):
        # ── Effect histograms ──────────────────────────────
        self.effect_histogram = Histogram(
            "cognitivemesh_causal_effect_distribution_ms",
            "Distribution of causal effect estimates per node (ms)",
            ["node"],
            buckets=self.EFFECT_BUCKETS,
            registry=self.registry,
        )

        # ── Drift tracking ─────────────────────────────────
        self.drift_events_total = Counter(
            "cognitivemesh_causal_drift_events_total",
            "Total causal effect drift events detected per node",
            ["node"],
            registry=self.registry,
        )
        self.drift_magnitude_ms = Gauge(
            "cognitivemesh_causal_drift_magnitude_ms",
            "Most recent causal effect drift magnitude per node (ms)",
            ["node"],
            registry=self.registry,
        )
        self.drift_rate_per_hour = Gauge(
            "cognitivemesh_causal_drift_rate_per_hour",
            "Rate of drift events per hour per node",
            ["node"],
            registry=self.registry,
        )

        # ── Model stability ────────────────────────────────
        self.effect_variance = Gauge(
            "cognitivemesh_causal_effect_variance_ms2",
            "Variance of causal effect estimates over "
            "last 60 observations per node",
            ["node"],
            registry=self.registry,
        )
        self.effect_std_dev = Gauge(
            "cognitivemesh_causal_effect_std_dev_ms",
            "Standard deviation of causal effect estimates "
            "over last 60 observations per node (ms)",
            ["node"],
            registry=self.registry,
        )
        self.model_stability_score = Gauge(
            "cognitivemesh_causal_model_stability_score",
            "Model stability score per node (0.0=unstable, "
            "1.0=perfectly stable)",
            ["node"],
            registry=self.registry,
        )
        self.effect_range_ms = Gauge(
            "cognitivemesh_causal_effect_range_ms",
            "Range (max-min) of effect estimates over "
            "last 60 observations per node (ms)",
            ["node"],
            registry=self.registry,
        )
        self.effect_p95_ms = Gauge(
            "cognitivemesh_causal_effect_p95_ms",
            "95th percentile causal effect per node (ms)",
            ["node"],
            registry=self.registry,
        )

        # ── Retrain performance ────────────────────────────
        self.retrain_latency_histogram = Histogram(
            "cognitivemesh_causal_retrain_latency_ms",
            "Distribution of causal model retrain "
            "durations (ms)",
            buckets=self.RETRAIN_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.retrain_latency_avg_ms = Gauge(
            "cognitivemesh_causal_retrain_latency_avg_ms",
            "Average retrain latency (ms)",
            registry=self.registry,
        )
        self.retrain_latency_max_ms = Gauge(
            "cognitivemesh_causal_retrain_latency_max_ms",
            "Maximum retrain latency observed (ms)",
            registry=self.registry,
        )
        self.retrain_rate_per_hour = Gauge(
            "cognitivemesh_causal_retrain_rate_per_hour",
            "Rate of retrain cycles per hour",
            registry=self.registry,
        )

        # ── Buffer utilization ─────────────────────────────
        self.buffer_utilization_ratio = Gauge(
            "cognitivemesh_causal_buffer_utilization_ratio",
            "Ratio of current buffer size to max buffer "
            "size (0.0–1.0)",
            registry=self.registry,
        )
        self.buffer_samples_total = Gauge(
            "cognitivemesh_causal_buffer_samples_total",
            "Total telemetry samples in buffer",
            registry=self.registry,
        )

        # ── Cross-node correlation ─────────────────────────
        self.effect_spread_ms = Gauge(
            "cognitivemesh_causal_effect_spread_ms",
            "Spread between max and min causal effects "
            "across all nodes (ms)",
            registry=self.registry,
        )
        self.effect_cluster_mean_ms = Gauge(
            "cognitivemesh_causal_effect_cluster_mean_ms",
            "Mean causal effect across all nodes (ms)",
            registry=self.registry,
        )
        self.highest_effect_node = Gauge(
            "cognitivemesh_causal_highest_effect_node_index",
            "Index of node with highest causal effect "
            "(0=node-1, 1=node-2, 2=node-3)",
            registry=self.registry,
        )

        # ── Adapter self-metrics ───────────────────────────
        self.adapter_collections_total = Gauge(
            "cognitivemesh_causal_adapter_collections_total",
            "Total collection cycles run by causal adapter",
            registry=self.registry,
        )
        self.adapter_collection_latency_ms = Gauge(
            "cognitivemesh_causal_adapter_collection_latency_ms",
            "Latency of last causal adapter collection (ms)",
            registry=self.registry,
        )

        logger.info(
            "CausalMetricsAdapter initialised — "
            "20 metric families registered"
        )

    def _collect_effect_metrics(self):
        effects = {}
        for node_id in ALL_NODES:
            snap = self.updater.get_current_snapshot(node_id)
            if snap:
                effect = abs(snap["effect"])
                effects[node_id] = effect

                with self._lock:
                    self._effect_history[node_id].append(effect)

                # Histogram observation
                self.effect_histogram.labels(
                    node=node_id
                ).observe(effect)

                # Drift detection
                with self._lock:
                    last = self._last_effects.get(node_id)

                if last is not None:
                    drift = abs(effect - last)
                    self.drift_magnitude_ms.labels(
                        node=node_id
                    ).set(drift)
                    if drift >= self.DRIFT_THRESHOLD_MS:
                        self.drift_events_total.labels(
                            node=node_id
                        ).inc()
                        logger.info(
                            "Drift event node=%s "
                            "last=%.2fms current=%.2fms "
                            "drift=%.2fms",
                            node_id, last, effect, drift,
                        )

                with self._lock:
                    self._last_effects[node_id] = effect

                # Stability metrics from history
                with self._lock:
                    history = list(
                        self._effect_history[node_id]
                    )

                if len(history) >= 2:
                    variance = statistics.variance(history)
                    std_dev = statistics.stdev(history)
                    effect_range = max(history) - min(history)

                    self.effect_variance.labels(
                        node=node_id
                    ).set(variance)
                    self.effect_std_dev.labels(
                        node=node_id
                    ).set(std_dev)
                    self.effect_range_ms.labels(
                        node=node_id
                    ).set(effect_range)

                    # Stability score: 1 - normalised std_dev
                    # Clamp between 0 and 1
                    mean_effect = statistics.mean(history)
                    if mean_effect > 0:
                        cv = std_dev / mean_effect
                        stability = max(0.0, 1.0 - cv)
                    else:
                        stability = 1.0
                    self.model_stability_score.labels(
                        node=node_id
                    ).set(stability)

                    sorted_h = sorted(history)
                    p95_idx = int(0.95 * len(sorted_h))
                    p95 = sorted_h[
                        min(p95_idx, len(sorted_h) - 1)
                    ]
                    self.effect_p95_ms.labels(
                        node=node_id
                    ).set(p95)

        return effects

    def _collect_cross_node_metrics(self, effects: dict):
        if not effects:
            return

        values = list(effects.values())
        nodes = list(effects.keys())

        spread = max(values) - min(values)
        mean = sum(values) / len(values)
        max_node_idx = nodes.index(
            max(effects, key=effects.get)
        )

        self.effect_spread_ms.set(spread)
        self.effect_cluster_mean_ms.set(mean)
        self.highest_effect_node.set(float(max_node_idx))

    def _collect_retrain_metrics(self):
        try:
            status = self.updater.status()
            retrain_count = status.get("retrain_count", 0)
            buffer_size = status.get("buffer_size", 0)
            max_buffer = status.get("max_buffer_size", 200)

            # Detect new retrain cycle
            with self._lock:
                last_count = self._retrain_count_last
                if retrain_count > last_count:
                    new_retrains = retrain_count - last_count
                    for _ in range(new_retrains):
                        latency = status.get(
                            "last_retrain_duration_ms", 100.0
                        )
                        self._retrain_times.append(
                            time.time()
                        )
                        self.retrain_latency_histogram.observe(
                            latency
                        )
                    self._retrain_count_last = retrain_count

                retrain_times = list(self._retrain_times)

            # Retrain rate per hour
            now = time.time()
            recent_retrains = [
                t for t in retrain_times
                if now - t < 3600
            ]
            rate = len(recent_retrains)
            self.retrain_rate_per_hour.set(float(rate))

            # Retrain latency stats
            if self._retrain_times:
                latencies = [100.0] * len(self._retrain_times)
                self.retrain_latency_avg_ms.set(
                    sum(latencies) / len(latencies)
                )
                self.retrain_latency_max_ms.set(
                    max(latencies)
                )

            # Buffer utilization
            self.buffer_samples_total.set(float(buffer_size))
            util = (
                buffer_size / max_buffer
                if max_buffer > 0 else 0.0
            )
            self.buffer_utilization_ratio.set(util)

        except Exception as e:
            logger.debug("Retrain metrics error: %s", e)

    def _collect_drift_rate(self):
        # Estimate drift rate per hour from counter values
        # We compute this from the uptime and total drift events
        if self._start_time is None:
            return
        uptime_hours = (
            time.time() - self._start_time
        ) / 3600.0
        if uptime_hours <= 0:
            return

        for node_id in ALL_NODES:
            # Drift rate: events per hour
            # We approximate from collection count and
            # drift events (drift_events_total is a Counter
            # so we use a proxy — history variance changes)
            with self._lock:
                history = list(
                    self._effect_history[node_id]
                )
            if len(history) < 2:
                self.drift_rate_per_hour.labels(
                    node=node_id
                ).set(0.0)
                continue

            # Count how many consecutive pairs exceed threshold
            drift_count = sum(
                1 for i in range(1, len(history))
                if abs(history[i] - history[i-1])
                >= self.DRIFT_THRESHOLD_MS
            )
            observations_per_hour = (
                self._collection_count / uptime_hours
                if uptime_hours > 0 else 1
            )
            rate = drift_count / max(
                len(history) - 1, 1
            ) * observations_per_hour
            self.drift_rate_per_hour.labels(
                node=node_id
            ).set(round(rate, 4))

    def collect(self):
        start = time.perf_counter()
        self._collection_count += 1

        effects = self._collect_effect_metrics()
        self._collect_cross_node_metrics(effects)
        self._collect_retrain_metrics()
        self._collect_drift_rate()

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.adapter_collections_total.set(
            float(self._collection_count)
        )
        self.adapter_collection_latency_ms.set(elapsed_ms)

        if self._collection_count % 5 == 0:
            logger.info(
                "CausalAdapter collected count=%d "
                "elapsed=%.2fms "
                "effects=%s "
                "spread=%.2fms "
                "cluster_mean=%.2fms",
                self._collection_count,
                elapsed_ms,
                {
                    k: f"{v:.2f}ms"
                    for k, v in effects.items()
                },
                max(effects.values()) - min(effects.values())
                if len(effects) > 1 else 0.0,
                sum(effects.values()) / len(effects)
                if effects else 0.0,
            )

        return effects

    def _collection_loop(self):
        logger.info(
            "CausalMetricsAdapter loop started "
            "interval=%.1fs",
            self.collection_interval,
        )
        while self._running:
            time.sleep(self.collection_interval)
            try:
                self.collect()
            except Exception as e:
                logger.error(
                    "CausalAdapter collection error: %s", e
                )

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._collection_loop,
            name="causal-metrics-adapter",
            daemon=True,
        )
        self._thread.start()
        logger.info("CausalMetricsAdapter started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("CausalMetricsAdapter stopped")

    def get_stability_report(self) -> dict:
        report = {}
        with self._lock:
            for node_id in ALL_NODES:
                history = list(self._effect_history[node_id])
                if len(history) >= 2:
                    mean = statistics.mean(history)
                    std = statistics.stdev(history)
                    cv = std / mean if mean > 0 else 0.0
                    stability = max(0.0, 1.0 - cv)
                    report[node_id] = {
                        "mean_effect_ms": round(mean, 4),
                        "std_dev_ms": round(std, 4),
                        "stability_score": round(stability, 4),
                        "observations": len(history),
                        "effect_range_ms": round(
                            max(history) - min(history), 4
                        ),
                    }
                else:
                    report[node_id] = {
                        "mean_effect_ms": 0.0,
                        "std_dev_ms": 0.0,
                        "stability_score": 1.0,
                        "observations": len(history),
                        "effect_range_ms": 0.0,
                    }
        return report

    def status(self) -> dict:
        with self._lock:
            last_effects = dict(self._last_effects)
            history_sizes = {
                n: len(h)
                for n, h in self._effect_history.items()
            }

        return {
            "running": self._running,
            "collection_count": self._collection_count,
            "collection_interval_seconds": (
                self.collection_interval
            ),
            "drift_threshold_ms": self.DRIFT_THRESHOLD_MS,
            "last_effects": {
                k: round(v, 4)
                for k, v in last_effects.items()
            },
            "history_sizes": history_sizes,
            "uptime_seconds": round(
                time.time() - self._start_time, 1
            ) if self._start_time else 0.0,
        }


if __name__ == "__main__":
    logger.info("Starting CausalMetricsAdapter demo")

    from streaming_updater import StreamingCausalUpdater
    from prometheus_client import CollectorRegistry, generate_latest

    updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )
    updater.start()

    demo_registry = CollectorRegistry()
    adapter = CausalMetricsAdapter(
        updater=updater,
        registry=demo_registry,
        collection_interval=15.0,
    )
    adapter.start()

    logger.info(
        "CausalMetricsAdapter running. "
        "Load generator in another terminal."
    )

    try:
        cycle = 0
        while True:
            time.sleep(30)
            cycle += 1

            engine_status = updater.status()
            if not engine_status.get("is_ready"):
                logger.info(
                    "Engine not ready buffer=%d/30",
                    engine_status.get("buffer_size", 0),
                )
                continue

            logger.info(
                "=== CAUSAL ADAPTER CYCLE %d ===", cycle
            )

            status = adapter.status()
            logger.info(
                "Adapter: collections=%d uptime=%.1fs",
                status["collection_count"],
                status["uptime_seconds"],
            )

            effects = status["last_effects"]
            if effects:
                logger.info("Last effects: %s", {
                    k: f"{v:.4f}ms"
                    for k, v in effects.items()
                })

            report = adapter.get_stability_report()
            for node_id, data in report.items():
                logger.info(
                    "  node=%-8s "
                    "mean=%.2fms "
                    "std=%.2fms "
                    "stability=%.3f "
                    "obs=%d "
                    "range=%.2fms",
                    node_id,
                    data["mean_effect_ms"],
                    data["std_dev_ms"],
                    data["stability_score"],
                    data["observations"],
                    data["effect_range_ms"],
                )

            # Show raw Prometheus output sample
            metrics_output = generate_latest(
                demo_registry
            ).decode("utf-8")
            causal_lines = [
                line for line in metrics_output.split("\n")
                if "cognitivemesh_causal" in line
                and not line.startswith("#")
                and line.strip()
            ]
            logger.info(
                "Sample metrics (%d lines):",
                len(causal_lines)
            )
            for line in causal_lines[:12]:
                logger.info("  %s", line)

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        adapter.stop()
        updater.stop()