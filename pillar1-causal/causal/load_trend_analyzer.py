import logging
import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "telemetry"))
sys.path.insert(0, os.path.dirname(__file__))

from streaming_updater import StreamingCausalUpdater

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.predictive.trend")


class TrendDirection(Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    UNKNOWN = "unknown"


class TrendSeverity(Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class LoadObservation:
    def __init__(
        self,
        node_id: str,
        active_queries: float,
        avg_latency_ms: float,
        causal_effect_ms: float,
        timestamp: str,
    ):
        self.node_id = node_id
        self.active_queries = active_queries
        self.avg_latency_ms = avg_latency_ms
        self.causal_effect_ms = causal_effect_ms
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "active_queries": round(self.active_queries, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "causal_effect_ms": round(self.causal_effect_ms, 2),
            "timestamp": self.timestamp,
        }


class TrendAnalysis:
    def __init__(
        self,
        node_id: str,
        direction: TrendDirection,
        severity: TrendSeverity,
        current_load: float,
        baseline_load: float,
        change_rate_per_minute: float,
        projected_load_5min: float,
        projected_latency_5min: float,
        causal_effect_ms: float,
        observations_used: int,
        timestamp: str,
    ):
        self.node_id = node_id
        self.direction = direction
        self.severity = severity
        self.current_load = current_load
        self.baseline_load = baseline_load
        self.change_rate_per_minute = change_rate_per_minute
        self.projected_load_5min = projected_load_5min
        self.projected_latency_5min = projected_latency_5min
        self.causal_effect_ms = causal_effect_ms
        self.observations_used = observations_used
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "direction": self.direction.value,
            "severity": self.severity.value,
            "current_load": round(self.current_load, 2),
            "baseline_load": round(self.baseline_load, 2),
            "change_rate_per_minute": round(
                self.change_rate_per_minute, 4
            ),
            "projected_load_5min": round(self.projected_load_5min, 2),
            "projected_latency_5min": round(
                self.projected_latency_5min, 2
            ),
            "causal_effect_ms": round(self.causal_effect_ms, 2),
            "observations_used": self.observations_used,
            "timestamp": self.timestamp,
            "interpretation": self._interpret(),
        }

    def _interpret(self) -> str:
        if self.direction == TrendDirection.RISING:
            return (
                f"Load on {self.node_id} is rising at "
                f"{self.change_rate_per_minute:+.2f} queries/min. "
                f"Projected load in 5 minutes: "
                f"{self.projected_load_5min:.1f} queries "
                f"(latency: {self.projected_latency_5min:.1f}ms). "
                f"Severity: {self.severity.value.upper()}."
            )
        elif self.direction == TrendDirection.FALLING:
            return (
                f"Load on {self.node_id} is falling at "
                f"{self.change_rate_per_minute:+.2f} queries/min. "
                f"Projected load in 5 minutes: "
                f"{self.projected_load_5min:.1f} queries "
                f"(latency: {self.projected_latency_5min:.1f}ms)."
            )
        else:
            return (
                f"Load on {self.node_id} is stable at "
                f"{self.current_load:.1f} queries "
                f"(latency: ~{self.projected_latency_5min:.1f}ms)."
            )


class NodeLoadTracker:
    MAX_OBSERVATIONS = 60
    MIN_OBSERVATIONS_FOR_TREND = 5

    SEVERITY_THRESHOLDS = {
        TrendSeverity.CRITICAL: 0.50,
        TrendSeverity.HIGH: 0.30,
        TrendSeverity.ELEVATED: 0.15,
    }

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.observations: deque = deque(maxlen=self.MAX_OBSERVATIONS)
        self._lock = threading.RLock()

    def add_observation(self, obs: LoadObservation):
        with self._lock:
            self.observations.append(obs)

    def size(self) -> int:
        with self._lock:
            return len(self.observations)

    def _compute_slope(self, values: list) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum(
            (i - x_mean) * (v - y_mean)
            for i, v in enumerate(values)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator < 0.0001:
            return 0.0
        return numerator / denominator

    def analyze(
        self,
        causal_effect_ms: float,
        observation_interval_seconds: float = 3.0,
    ) -> Optional[TrendAnalysis]:
        with self._lock:
            obs_list = list(self.observations)

        if len(obs_list) < self.MIN_OBSERVATIONS_FOR_TREND:
            return None

        loads = [o.active_queries for o in obs_list]
        current_load = loads[-1]

        baseline_window = max(3, len(loads) // 3)
        baseline_load = statistics.mean(loads[:baseline_window])

        slope_per_observation = self._compute_slope(loads)
        observations_per_minute = 60.0 / observation_interval_seconds
        change_rate_per_minute = (
            slope_per_observation * observations_per_minute
        )

        projected_load_5min = max(
            0.0,
            current_load + change_rate_per_minute * 5
        )
        projected_latency_5min = abs(causal_effect_ms) * projected_load_5min

        if abs(change_rate_per_minute) < 0.05:
            direction = TrendDirection.STABLE
        elif change_rate_per_minute > 0:
            direction = TrendDirection.RISING
        else:
            direction = TrendDirection.FALLING

        if baseline_load > 0.001:
            load_change_pct = abs(
                (projected_load_5min - baseline_load) / baseline_load
            )
        else:
            load_change_pct = 0.0

        if load_change_pct >= self.SEVERITY_THRESHOLDS[TrendSeverity.CRITICAL]:
            severity = TrendSeverity.CRITICAL
        elif load_change_pct >= self.SEVERITY_THRESHOLDS[TrendSeverity.HIGH]:
            severity = TrendSeverity.HIGH
        elif load_change_pct >= self.SEVERITY_THRESHOLDS[TrendSeverity.ELEVATED]:
            severity = TrendSeverity.ELEVATED
        else:
            severity = TrendSeverity.NORMAL

        if direction == TrendDirection.STABLE:
            severity = TrendSeverity.NORMAL

        return TrendAnalysis(
            node_id=self.node_id,
            direction=direction,
            severity=severity,
            current_load=current_load,
            baseline_load=baseline_load,
            change_rate_per_minute=change_rate_per_minute,
            projected_load_5min=projected_load_5min,
            projected_latency_5min=projected_latency_5min,
            causal_effect_ms=abs(causal_effect_ms),
            observations_used=len(obs_list),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


class LoadTrendAnalyzer:
    def __init__(
        self,
        updater: StreamingCausalUpdater,
        observation_interval_seconds: float = 3.0,
        analysis_interval_seconds: float = 15.0,
    ):
        self.updater = updater
        self.observation_interval = observation_interval_seconds
        self.analysis_interval = analysis_interval_seconds

        self.trackers: dict[str, NodeLoadTracker] = {
            node_id: NodeLoadTracker(node_id)
            for node_id in ["node-1", "node-2", "node-3"]
        }

        self._latest_analyses: dict[str, TrendAnalysis] = {}
        self._analyses_lock = threading.RLock()
        self._running = False
        self._observer_thread: Optional[threading.Thread] = None
        self._analyzer_thread: Optional[threading.Thread] = None
        self._observations_collected = 0
        self._analyses_run = 0

    def _collect_observations(self):
        logger.info(
            "Load observer started interval=%.1fs",
            self.observation_interval,
        )
        while self._running:
            time.sleep(self.observation_interval)
            try:
                engine_status = self.updater.status()
                if not engine_status.get("is_ready"):
                    continue

                buf = self.updater.buffer
                if buf is None:
                    continue

                df = buf.get_dataframe()
                if df is None:
                    continue

                collected_any = False
                for node_id in ["node-1", "node-2", "node-3"]:
                    snapshot = self.updater.get_current_snapshot(node_id)
                    if not snapshot:
                        continue

                    node_safe = node_id.replace("-", "_")
                    col_queries = f"{node_safe}_active_queries"
                    col_latency = f"{node_safe}_avg_query_duration_ms"

                    if col_queries not in df.columns:
                        continue

                    active_queries = float(df[col_queries].iloc[-1])
                    avg_latency = float(
                        df[col_latency].iloc[-1]
                        if col_latency in df.columns else 0.0
                    )

                    obs = LoadObservation(
                        node_id=node_id,
                        active_queries=active_queries,
                        avg_latency_ms=avg_latency,
                        causal_effect_ms=abs(snapshot["effect"]),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    self.trackers[node_id].add_observation(obs)
                    collected_any = True

                if collected_any:
                    self._observations_collected += 1
                    if self._observations_collected % 10 == 0:
                        sizes = {
                            n: t.size()
                            for n, t in self.trackers.items()
                        }
                        logger.info(
                            "Observations collected=%d "
                            "tracker_sizes=%s",
                            self._observations_collected,
                            sizes,
                        )

            except Exception as e:
                logger.error("Observation error: %s", e)

    def _run_analyses(self):
        logger.info(
            "Load analyzer started interval=%.1fs",
            self.analysis_interval,
        )
        while self._running:
            time.sleep(self.analysis_interval)
            try:
                engine_status = self.updater.status()
                if not engine_status.get("is_ready"):
                    continue

                self._analyses_run += 1
                results = {}

                for node_id, tracker in self.trackers.items():
                    snapshot = self.updater.get_current_snapshot(node_id)
                    if not snapshot:
                        continue

                    causal_effect = abs(snapshot["effect"])
                    analysis = tracker.analyze(
                        causal_effect_ms=causal_effect,
                        observation_interval_seconds=self.observation_interval,
                    )

                    if analysis:
                        results[node_id] = analysis
                        logger.info(
                            "Trend analysis node=%-8s "
                            "direction=%-8s severity=%-10s "
                            "load=%.2f→%.2f rate=%+.3f/min "
                            "projected_latency=%.2fms",
                            node_id,
                            analysis.direction.value,
                            analysis.severity.value,
                            analysis.current_load,
                            analysis.projected_load_5min,
                            analysis.change_rate_per_minute,
                            analysis.projected_latency_5min,
                        )

                        if analysis.severity in (
                            TrendSeverity.HIGH,
                            TrendSeverity.CRITICAL,
                        ):
                            logger.warning(
                                "LOAD ALERT node=%s severity=%s "
                                "projected_load=%.1f "
                                "projected_latency=%.1fms",
                                node_id,
                                analysis.severity.value.upper(),
                                analysis.projected_load_5min,
                                analysis.projected_latency_5min,
                            )
                    else:
                        logger.info(
                            "Trend analysis node=%-8s "
                            "insufficient observations=%d/5",
                            node_id,
                            tracker.size(),
                        )

                with self._analyses_lock:
                    self._latest_analyses = results

            except Exception as e:
                logger.error("Analysis error: %s", e)

    def start(self):
        self._running = True
        self._observer_thread = threading.Thread(
            target=self._collect_observations,
            name="load-observer",
            daemon=True,
        )
        self._analyzer_thread = threading.Thread(
            target=self._run_analyses,
            name="load-analyzer",
            daemon=True,
        )
        self._observer_thread.start()
        self._analyzer_thread.start()
        logger.info("LoadTrendAnalyzer started")

    def stop(self):
        self._running = False
        if self._observer_thread:
            self._observer_thread.join(timeout=10)
        if self._analyzer_thread:
            self._analyzer_thread.join(timeout=10)
        logger.info("LoadTrendAnalyzer stopped")

    def get_latest_analyses(self) -> dict:
        with self._analyses_lock:
            return {
                node_id: analysis.to_dict()
                for node_id, analysis in self._latest_analyses.items()
            }

    def get_analysis(self, node_id: str) -> Optional[dict]:
        with self._analyses_lock:
            analysis = self._latest_analyses.get(node_id)
            return analysis.to_dict() if analysis else None

    def get_cluster_trend(self) -> dict:
        with self._analyses_lock:
            analyses = dict(self._latest_analyses)

        if not analyses:
            return {
                "cluster_trend": TrendDirection.UNKNOWN.value,
                "severity": TrendSeverity.NORMAL.value,
                "nodes_analyzed": 0,
                "avg_projected_latency_5min": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        rising = sum(
            1 for a in analyses.values()
            if a.direction == TrendDirection.RISING
        )
        falling = sum(
            1 for a in analyses.values()
            if a.direction == TrendDirection.FALLING
        )

        if rising > falling:
            cluster_trend = TrendDirection.RISING
        elif falling > rising:
            cluster_trend = TrendDirection.FALLING
        else:
            cluster_trend = TrendDirection.STABLE

        severities = [a.severity for a in analyses.values()]
        severity_order = [
            TrendSeverity.CRITICAL,
            TrendSeverity.HIGH,
            TrendSeverity.ELEVATED,
            TrendSeverity.NORMAL,
        ]
        worst_severity = TrendSeverity.NORMAL
        for s in severity_order:
            if s in severities:
                worst_severity = s
                break

        avg_projected_latency = sum(
            a.projected_latency_5min for a in analyses.values()
        ) / len(analyses)

        return {
            "cluster_trend": cluster_trend.value,
            "severity": worst_severity.value,
            "nodes_analyzed": len(analyses),
            "rising_nodes": rising,
            "falling_nodes": falling,
            "avg_projected_latency_5min": round(avg_projected_latency, 2),
            "node_trends": {
                node_id: {
                    "direction": a.direction.value,
                    "severity": a.severity.value,
                    "projected_load": round(a.projected_load_5min, 2),
                    "projected_latency": round(
                        a.projected_latency_5min, 2
                    ),
                }
                for node_id, a in analyses.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def status(self) -> dict:
        return {
            "running": self._running,
            "observations_collected": self._observations_collected,
            "analyses_run": self._analyses_run,
            "nodes_tracked": list(self.trackers.keys()),
            "tracker_sizes": {
                n: t.size() for n, t in self.trackers.items()
            },
            "latest_analyses_available": list(
                self._latest_analyses.keys()
            ),
        }


if __name__ == "__main__":
    logger.info("Starting load trend analyzer demo")

    updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )
    updater.start()

    analyzer = LoadTrendAnalyzer(
        updater=updater,
        observation_interval_seconds=3.0,
        analysis_interval_seconds=15.0,
    )
    analyzer.start()

    logger.info(
        "System running. Load generator in another terminal. "
        "Trend analysis fires every 15s after engine is ready."
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

            logger.info("=== TREND CYCLE %d ===", cycle)

            status = analyzer.status()
            logger.info(
                "Analyzer status: obs=%d analyses=%d "
                "tracker_sizes=%s",
                status["observations_collected"],
                status["analyses_run"],
                status["tracker_sizes"],
            )

            cluster = analyzer.get_cluster_trend()
            logger.info(
                "Cluster trend: %s severity=%s "
                "avg_projected_latency=%.2fms nodes=%d",
                cluster["cluster_trend"].upper(),
                cluster["severity"].upper(),
                cluster.get("avg_projected_latency_5min", 0.0),
                cluster["nodes_analyzed"],
            )

            analyses = analyzer.get_latest_analyses()
            if not analyses:
                logger.info(
                    "  No trend analyses yet — "
                    "accumulating observations"
                )
                continue

            for node_id, analysis in analyses.items():
                logger.info(
                    "  node=%-8s direction=%-8s "
                    "load=%.2f→%.2f latency=%.2fms  %s",
                    node_id,
                    analysis["direction"].upper(),
                    analysis["current_load"],
                    analysis["projected_load_5min"],
                    analysis["projected_latency_5min"],
                    analysis["severity"].upper(),
                )
                logger.info(
                    "    %s", analysis["interpretation"]
                )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        analyzer.stop()
        updater.stop()