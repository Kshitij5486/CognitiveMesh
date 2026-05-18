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
logger = logging.getLogger("cm.observability.healing_adapter")

ALL_NODES = ["node-1", "node-2", "node-3"]
ACTION_TYPES = [
    "rebalance", "reroute", "retrain",
    "isolate", "alert_operator", "no_action",
]
TRIGGER_TYPES = [
    "scheduled", "prediction_drift", "alert_driven",
    "byzantine_detected", "manual",
]
RECOVERY_PHASES = [
    "detected", "alerting", "rerouting",
    "retraining", "verifying", "restored", "failed",
]


class HealingMetricsAdapter:
    """
    Deep healing metrics adapter.

    Tracks:
    - Healing action latency distribution per action type
    - Action throughput (actions per minute)
    - Routing decision latency and weight stability
    - Retrainer trigger frequency by trigger type
    - Recovery phase transition counts and durations
    - Cooldown utilization per component
    - Component health scores (composite)
    - Alert-to-action latency (time from alert to healing)
    """

    ACTION_LATENCY_BUCKETS = [
        0.1, 0.5, 1.0, 2.0, 5.0,
        10.0, 25.0, 50.0, 100.0,
    ]
    RECOVERY_DURATION_BUCKETS = [
        1.0, 5.0, 10.0, 15.0, 20.0,
        30.0, 60.0, 120.0, 300.0,
    ]
    ROUTING_LATENCY_BUCKETS = [
        0.1, 0.5, 1.0, 2.0, 5.0,
        10.0, 20.0, 50.0,
    ]
    HISTORY_WINDOW = 100

    def __init__(
        self,
        engine,
        router,
        retrainer,
        orchestrator,
        registry: CollectorRegistry,
        collection_interval: float = 15.0,
    ):
        self.engine = engine
        self.router = router
        self.retrainer = retrainer
        self.orchestrator = orchestrator
        self.registry = registry
        self.collection_interval = collection_interval

        # Internal state tracking
        self._last_action_count = 0
        self._last_retrain_count = 0
        self._last_sequence_count = 0
        self._last_reroute_count = 0
        self._action_timestamps: deque = deque(
            maxlen=self.HISTORY_WINDOW
        )
        self._routing_weights_history: dict = {
            n: deque(maxlen=50) for n in ALL_NODES
        }
        self._recovery_durations: deque = deque(
            maxlen=50
        )

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._collection_count = 0
        self._start_time: Optional[float] = None

        self._init_metrics()

    def _init_metrics(self):
        # ── Healing Engine ────────────────────────────────
        self.action_latency_histogram = Histogram(
            "cognitivemesh_healing_action_latency_ms",
            "Distribution of healing action execution "
            "latency per action type (ms)",
            ["action_type"],
            buckets=self.ACTION_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.action_throughput_per_min = Gauge(
            "cognitivemesh_healing_action_throughput_per_min",
            "Healing actions executed per minute "
            "(rolling 5-minute window)",
            registry=self.registry,
        )
        self.action_success_rate_by_type = Gauge(
            "cognitivemesh_healing_action_success_rate_by_type",
            "Success rate per action type (0.0–1.0)",
            ["action_type"],
            registry=self.registry,
        )
        self.action_cooldown_active = Gauge(
            "cognitivemesh_healing_cooldown_active_nodes",
            "Number of nodes currently in healing cooldown",
            registry=self.registry,
        )
        self.healing_engine_health_score = Gauge(
            "cognitivemesh_healing_engine_health_score",
            "Composite health score for healing engine "
            "(0.0=unhealthy, 1.0=fully healthy)",
            registry=self.registry,
        )
        self.consecutive_failures = Gauge(
            "cognitivemesh_healing_consecutive_failures",
            "Consecutive healing action failures per node",
            ["node"],
            registry=self.registry,
        )

        # ── QueryRouter ───────────────────────────────────
        self.routing_weight_stability = Gauge(
            "cognitivemesh_router_weight_stability_score",
            "Stability of causal routing weights per node "
            "(0.0=unstable, 1.0=stable)",
            ["node"],
            registry=self.registry,
        )
        self.routing_weight_std_dev = Gauge(
            "cognitivemesh_router_weight_std_dev",
            "Standard deviation of routing weight "
            "per node over last 50 observations",
            ["node"],
            registry=self.registry,
        )
        self.routing_traffic_share = Gauge(
            "cognitivemesh_router_traffic_share_ratio",
            "Current traffic share per node (0.0–1.0)",
            ["node"],
            registry=self.registry,
        )
        self.reroute_rate_per_hour = Gauge(
            "cognitivemesh_router_reroute_rate_per_hour",
            "Rate of reroute decisions per hour",
            registry=self.registry,
        )
        self.router_health_score = Gauge(
            "cognitivemesh_router_health_score",
            "Composite router health score "
            "(0.0=degraded, 1.0=healthy)",
            registry=self.registry,
        )
        self.nodes_receiving_traffic = Gauge(
            "cognitivemesh_router_nodes_receiving_traffic",
            "Number of nodes currently receiving traffic",
            registry=self.registry,
        )
        self.traffic_concentration_index = Gauge(
            "cognitivemesh_router_traffic_concentration_index",
            "Herfindahl index of traffic concentration "
            "(0=perfectly distributed, 1=fully concentrated)",
            registry=self.registry,
        )

        # ── AutoRetrainer ──────────────────────────────────
        self.retrain_trigger_rate_per_hour = Gauge(
            "cognitivemesh_retrainer_trigger_rate_per_hour",
            "Rate of retrain triggers per hour",
            registry=self.registry,
        )
        self.retrain_success_rate = Gauge(
            "cognitivemesh_retrainer_success_rate",
            "Overall retrain success rate (0.0–1.0)",
            registry=self.registry,
        )
        self.retrain_cooldown_coverage = Gauge(
            "cognitivemesh_retrainer_cooldown_coverage",
            "Fraction of nodes currently in retrain "
            "cooldown (0.0–1.0)",
            registry=self.registry,
        )
        self.retrainer_health_score = Gauge(
            "cognitivemesh_retrainer_health_score",
            "Composite retrainer health score "
            "(0.0=degraded, 1.0=healthy)",
            registry=self.registry,
        )
        self.retrain_efficiency_ratio = Gauge(
            "cognitivemesh_retrainer_efficiency_ratio",
            "Ratio of successful retrains to total "
            "check cycles (measures trigger accuracy)",
            registry=self.registry,
        )

        # ── RecoveryOrchestrator ───────────────────────────
        self.recovery_duration_histogram = Histogram(
            "cognitivemesh_recovery_sequence_duration_seconds",
            "Distribution of recovery sequence "
            "end-to-end durations (seconds)",
            buckets=self.RECOVERY_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.recovery_success_rate = Gauge(
            "cognitivemesh_recovery_success_rate",
            "Overall recovery success rate (0.0–1.0)",
            registry=self.registry,
        )
        self.recovery_rate_per_hour = Gauge(
            "cognitivemesh_recovery_rate_per_hour",
            "Rate of recovery sequences per hour",
            registry=self.registry,
        )
        self.recovery_mttr_seconds = Gauge(
            "cognitivemesh_recovery_mttr_seconds",
            "Mean Time To Recovery — average successful "
            "recovery duration (seconds)",
            registry=self.registry,
        )
        self.recovery_actions_per_sequence_avg = Gauge(
            "cognitivemesh_recovery_actions_per_sequence_avg",
            "Average number of actions taken per "
            "recovery sequence",
            registry=self.registry,
        )
        self.orchestrator_health_score = Gauge(
            "cognitivemesh_orchestrator_health_score",
            "Composite orchestrator health score "
            "(0.0=degraded, 1.0=healthy)",
            registry=self.registry,
        )

        # ── Overall Self-Healing Health ────────────────────
        self.self_healing_overall_score = Gauge(
            "cognitivemesh_self_healing_overall_score",
            "Overall self-healing fabric health score "
            "— composite of all components (0.0–1.0)",
            registry=self.registry,
        )
        self.self_healing_fabric_status = Gauge(
            "cognitivemesh_self_healing_fabric_status",
            "Self-healing fabric operational status "
            "(1=operational, 0=degraded)",
            registry=self.registry,
        )

        # ── Adapter self-metrics ───────────────────────────
        self.adapter_collections_total = Gauge(
            "cognitivemesh_healing_adapter_collections_total",
            "Total collection cycles by healing adapter",
            registry=self.registry,
        )
        self.adapter_collection_latency_ms = Gauge(
            "cognitivemesh_healing_adapter_latency_ms",
            "Latency of last healing adapter collection (ms)",
            registry=self.registry,
        )

        logger.info(
            "HealingMetricsAdapter initialised — "
            "25 metric families registered"
        )

    def _collect_healing_engine_metrics(self):
        try:
            status = self.engine.status()
            history = self.engine.get_action_history(n=100)

            total = status.get("total_actions", 0)
            successful = status.get("successful_actions", 0)

            # Throughput — actions in last 5 minutes
            now = time.time()
            with self._lock:
                timestamps = list(self._action_timestamps)
            recent = [
                t for t in timestamps
                if now - t < 300
            ]
            throughput = len(recent) / 5.0
            self.action_throughput_per_min.set(throughput)

            # Register new action timestamps
            new_actions = total - self._last_action_count
            if new_actions > 0:
                with self._lock:
                    for _ in range(
                        min(new_actions, 10)
                    ):
                        self._action_timestamps.append(now)
                self._last_action_count = total

            # Action latency histogram from history
            type_stats: dict = {
                atype: {"total": 0, "success": 0}
                for atype in ACTION_TYPES
            }
            for action in history:
                atype = action.get(
                    "action_type", "no_action"
                )
                duration = action.get("duration_ms", 0.5)
                is_success = action.get(
                    "status", ""
                ) == "success"

                if atype in type_stats:
                    type_stats[atype]["total"] += 1
                    if is_success:
                        type_stats[atype]["success"] += 1

                if duration and duration > 0:
                    self.action_latency_histogram.labels(
                        action_type=atype
                    ).observe(duration)

            # Success rate per type
            for atype, stats in type_stats.items():
                if stats["total"] > 0:
                    rate = stats["success"] / stats["total"]
                else:
                    rate = 1.0
                self.action_success_rate_by_type.labels(
                    action_type=atype
                ).set(rate)

            # Cooldown active nodes
            cooldowns = status.get("active_cooldowns", {})
            active_cooldowns = sum(
                1 for v in cooldowns.values() if v
            ) if cooldowns else 0
            self.action_cooldown_active.set(
                float(active_cooldowns)
            )

            # Consecutive failures per node
            for node_id in ALL_NODES:
                node_history = [
                    a for a in history
                    if a.get("node_id") == node_id
                ]
                consecutive = 0
                for action in reversed(node_history):
                    if action.get("status") == "failed":
                        consecutive += 1
                    else:
                        break
                self.consecutive_failures.labels(
                    node=node_id
                ).set(float(consecutive))

            # Engine health score
            overall_success = (
                successful / total if total > 0 else 1.0
            )
            checks = status.get("checks_run", 1)
            responsiveness = min(
                1.0, checks / max(self._collection_count, 1)
            )
            engine_score = (
                overall_success * 0.7
                + responsiveness * 0.3
            )
            self.healing_engine_health_score.set(
                engine_score
            )

            return engine_score

        except Exception as e:
            logger.debug(
                "Healing engine metrics error: %s", e
            )
            return 0.5

    def _collect_router_metrics(self):
        try:
            status = self.router.status()
            weights = self.router._compute_causal_weights(
                ALL_NODES
            )

            # Traffic share and weight stability
            total_weight = sum(weights.values())
            hhi = 0.0
            for node_id in ALL_NODES:
                weight = weights.get(node_id, 0.0)
                share = (
                    weight / total_weight
                    if total_weight > 0 else 1.0 / 3
                )
                self.routing_traffic_share.labels(
                    node=node_id
                ).set(share)
                hhi += share ** 2

                with self._lock:
                    self._routing_weights_history[
                        node_id
                    ].append(weight)

                history = list(
                    self._routing_weights_history[node_id]
                )
                if len(history) >= 2:
                    std = statistics.stdev(history)
                    mean = statistics.mean(history)
                    cv = std / mean if mean > 0 else 0.0
                    stability = max(0.0, 1.0 - cv * 10)
                    self.routing_weight_stability.labels(
                        node=node_id
                    ).set(stability)
                    self.routing_weight_std_dev.labels(
                        node=node_id
                    ).set(std)
                else:
                    self.routing_weight_stability.labels(
                        node=node_id
                    ).set(1.0)
                    self.routing_weight_std_dev.labels(
                        node=node_id
                    ).set(0.0)

            self.traffic_concentration_index.set(hhi)

            # Reroute rate
            total_reroutes = status.get(
                "total_reroutes", 0
            )
            new_reroutes = (
                total_reroutes - self._last_reroute_count
            )
            if new_reroutes > 0:
                self._last_reroute_count = total_reroutes

            uptime_hours = (
                (time.time() - self._start_time) / 3600.0
                if self._start_time else 1.0
            )
            reroute_rate = (
                total_reroutes / uptime_hours
                if uptime_hours > 0 else 0.0
            )
            self.reroute_rate_per_hour.set(reroute_rate)

            # Nodes receiving traffic
            active = status.get("active_nodes", 3)
            self.nodes_receiving_traffic.set(float(active))

            # Router health score
            active_ratio = active / len(ALL_NODES)
            no_isolation = (
                1.0
                if status.get("isolated_nodes", 0) == 0
                else 0.5
            )
            distribution_score = 1.0 - (hhi - 1.0 / 3)
            router_score = (
                active_ratio * 0.4
                + no_isolation * 0.3
                + max(0.0, distribution_score) * 0.3
            )
            self.router_health_score.set(router_score)

            return router_score, weights

        except Exception as e:
            logger.debug("Router metrics error: %s", e)
            return 0.5, {}

    def _collect_retrainer_metrics(self):
        try:
            status = self.retrainer.status()

            total = status.get("total_retrains", 0)
            successful = status.get(
                "successful_retrains", 0
            )
            failed = status.get("failed_retrains", 0)
            checks = status.get("checks_run", 1)

            # Success rate
            success_rate = (
                successful / total if total > 0 else 1.0
            )
            self.retrain_success_rate.set(success_rate)

            # Trigger rate
            uptime_hours = (
                (time.time() - self._start_time) / 3600.0
                if self._start_time else 1.0
            )
            trigger_rate = (
                total / uptime_hours
                if uptime_hours > 0 else 0.0
            )
            self.retrain_trigger_rate_per_hour.set(
                trigger_rate
            )

            # Efficiency — retrains per check cycle
            efficiency = (
                total / checks if checks > 0 else 0.0
            )
            self.retrain_efficiency_ratio.set(efficiency)

            # Cooldown coverage
            cooldown_nodes = 0
            for node_id in ALL_NODES:
                if self.retrainer._is_in_cooldown(node_id):
                    cooldown_nodes += 1
            coverage = cooldown_nodes / len(ALL_NODES)
            self.retrain_cooldown_coverage.set(coverage)

            # Retrainer health
            retrainer_score = (
                success_rate * 0.6
                + (1.0 - coverage) * 0.2
                + min(1.0, efficiency * 10) * 0.2
            )
            self.retrainer_health_score.set(
                retrainer_score
            )

            return retrainer_score

        except Exception as e:
            logger.debug("Retrainer metrics error: %s", e)
            return 0.5

    def _collect_orchestrator_metrics(self):
        try:
            status = self.orchestrator.status()
            history = self.orchestrator.get_sequence_history(
                n=20
            )

            total = status.get("total_sequences", 0)
            successful = status.get(
                "successful_recoveries", 0
            )

            # Success rate
            success_rate = (
                successful / total if total > 0 else 1.0
            )
            self.recovery_success_rate.set(success_rate)

            # Recovery rate per hour
            uptime_hours = (
                (time.time() - self._start_time) / 3600.0
                if self._start_time else 1.0
            )
            rate = total / uptime_hours
            self.recovery_rate_per_hour.set(rate)

            # Duration stats from history
            completed = [
                s for s in history
                if s.get("duration_seconds") is not None
                and s.get("phase") == "restored"
            ]
            if completed:
                durations = [
                    s["duration_seconds"]
                    for s in completed
                ]
                mttr = sum(durations) / len(durations)
                self.recovery_mttr_seconds.set(mttr)

                for d in durations:
                    self.recovery_duration_histogram.observe(
                        d
                    )
            else:
                self.recovery_mttr_seconds.set(0.0)

            # Actions per sequence
            if history:
                action_counts = [
                    len(s.get("actions_taken", []))
                    for s in history
                ]
                avg_actions = (
                    sum(action_counts) / len(action_counts)
                )
                self.recovery_actions_per_sequence_avg.set(
                    avg_actions
                )
            else:
                self.recovery_actions_per_sequence_avg.set(
                    0.0
                )

            # Orchestrator health
            active = status.get("active_sequences", 0)
            no_overload = 1.0 if active < 2 else 0.5
            orch_score = (
                success_rate * 0.6
                + no_overload * 0.4
            )
            self.orchestrator_health_score.set(orch_score)

            return orch_score

        except Exception as e:
            logger.debug(
                "Orchestrator metrics error: %s", e
            )
            return 0.5

    def _compute_overall_score(
        self,
        engine_score: float,
        router_score: float,
        retrainer_score: float,
        orch_score: float,
    ):
        overall = (
            engine_score * 0.30
            + router_score * 0.30
            + retrainer_score * 0.20
            + orch_score * 0.20
        )
        self.self_healing_overall_score.set(overall)
        status = 1.0 if overall >= 0.6 else 0.0
        self.self_healing_fabric_status.set(status)
        return overall

    def collect(self):
        start = time.perf_counter()
        self._collection_count += 1

        engine_score = self._collect_healing_engine_metrics()
        router_score, weights = self._collect_router_metrics()
        retrainer_score = self._collect_retrainer_metrics()
        orch_score = self._collect_orchestrator_metrics()
        overall = self._compute_overall_score(
            engine_score,
            router_score,
            retrainer_score,
            orch_score,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.adapter_collections_total.set(
            float(self._collection_count)
        )
        self.adapter_collection_latency_ms.set(elapsed_ms)

        if self._collection_count % 5 == 0:
            logger.info(
                "HealingAdapter collected count=%d "
                "elapsed=%.2fms "
                "scores: engine=%.3f router=%.3f "
                "retrainer=%.3f orch=%.3f overall=%.3f "
                "fabric=%s",
                self._collection_count,
                elapsed_ms,
                engine_score,
                router_score,
                retrainer_score,
                orch_score,
                overall,
                "OPERATIONAL" if overall >= 0.6
                else "DEGRADED",
            )

        return overall

    def _collection_loop(self):
        logger.info(
            "HealingMetricsAdapter loop started "
            "interval=%.1fs",
            self.collection_interval,
        )
        while self._running:
            time.sleep(self.collection_interval)
            try:
                self.collect()
            except Exception as e:
                logger.error(
                    "HealingAdapter error: %s", e
                )

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._collection_loop,
            name="healing-metrics-adapter",
            daemon=True,
        )
        self._thread.start()
        logger.info("HealingMetricsAdapter started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("HealingMetricsAdapter stopped")

    def get_health_report(self) -> dict:
        try:
            engine_status = self.engine.status()
            router_status = self.router.status()
            retrainer_status = self.retrainer.status()
            orch_status = self.orchestrator.status()

            total_actions = engine_status.get(
                "total_actions", 0
            )
            successful_actions = engine_status.get(
                "successful_actions", 0
            )
            action_success = (
                successful_actions / total_actions
                if total_actions > 0 else 1.0
            )

            total_retrains = retrainer_status.get(
                "total_retrains", 0
            )
            successful_retrains = retrainer_status.get(
                "successful_retrains", 0
            )
            retrain_success = (
                successful_retrains / total_retrains
                if total_retrains > 0 else 1.0
            )

            return {
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "collection_count": self._collection_count,
                "components": {
                    "healing_engine": {
                        "checks_run": engine_status.get(
                            "checks_run", 0
                        ),
                        "total_actions": total_actions,
                        "action_success_rate": round(
                            action_success, 4
                        ),
                    },
                    "router": {
                        "checks_run": router_status.get(
                            "checks_run", 0
                        ),
                        "total_reroutes": router_status.get(
                            "total_reroutes", 0
                        ),
                        "active_nodes": router_status.get(
                            "active_nodes", 3
                        ),
                        "node_states": router_status.get(
                            "node_states", {}
                        ),
                    },
                    "retrainer": {
                        "checks_run": retrainer_status.get(
                            "checks_run", 0
                        ),
                        "total_retrains": total_retrains,
                        "retrain_success_rate": round(
                            retrain_success, 4
                        ),
                    },
                    "orchestrator": {
                        "checks_run": orch_status.get(
                            "checks_run", 0
                        ),
                        "total_sequences": orch_status.get(
                            "total_sequences", 0
                        ),
                        "successful_recoveries": (
                            orch_status.get(
                                "successful_recoveries", 0
                            )
                        ),
                        "active_sequences": orch_status.get(
                            "active_sequences", 0
                        ),
                    },
                },
            }
        except Exception as e:
            logger.error("Health report error: %s", e)
            return {}

    def status(self) -> dict:
        return {
            "running": self._running,
            "collection_count": self._collection_count,
            "collection_interval_seconds": (
                self.collection_interval
            ),
            "uptime_seconds": round(
                time.time() - self._start_time, 1
            ) if self._start_time else 0.0,
            "metrics_families": 25,
        }


if __name__ == "__main__":
    logger.info("Starting HealingMetricsAdapter demo")

    from streaming_updater import StreamingCausalUpdater
    from load_trend_analyzer import LoadTrendAnalyzer
    from causal_simulator import CausalSimulator
    from predictive_alerter import PredictiveAlerter
    from healing_action_engine import HealingActionEngine
    from query_router import QueryRouter, RoutingStrategy
    from auto_retrainer import AutoRetrainer
    from recovery_orchestrator import RecoveryOrchestrator
    from prometheus_client import (
        CollectorRegistry, generate_latest
    )

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

    simulator = CausalSimulator(
        updater=updater,
        analyzer=analyzer,
        simulation_interval_seconds=30.0,
    )
    simulator.start()

    alerter = PredictiveAlerter(
        updater=updater,
        analyzer=analyzer,
        simulator=simulator,
        check_interval=15.0,
    )
    alerter.start()

    engine = HealingActionEngine(
        updater=updater,
        alerter=alerter,
        check_interval=15.0,
        auto_heal=True,
    )
    engine.start()

    router = QueryRouter(
        updater=updater,
        alerter=alerter,
        strategy=RoutingStrategy.CAUSAL_WEIGHTED,
        check_interval=15.0,
    )
    router.start()

    retrainer = AutoRetrainer(
        updater=updater,
        alerter=alerter,
        check_interval=30.0,
        drift_threshold_ms=3.0,
    )
    retrainer.start()

    orchestrator = RecoveryOrchestrator(
        updater=updater,
        alerter=alerter,
        engine=engine,
        router=router,
        retrainer=retrainer,
        check_interval=15.0,
    )
    orchestrator.start()

    demo_registry = CollectorRegistry()
    adapter = HealingMetricsAdapter(
        engine=engine,
        router=router,
        retrainer=retrainer,
        orchestrator=orchestrator,
        registry=demo_registry,
        collection_interval=15.0,
    )
    adapter.start()

    logger.info(
        "HealingMetricsAdapter running. "
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
                "=== HEALING ADAPTER CYCLE %d ===", cycle
            )

            status = adapter.status()
            logger.info(
                "Adapter: collections=%d uptime=%.1fs",
                status["collection_count"],
                status["uptime_seconds"],
            )

            report = adapter.get_health_report()
            if report:
                comps = report.get("components", {})

                eng = comps.get("healing_engine", {})
                logger.info(
                    "Engine: checks=%d actions=%d "
                    "success_rate=%.3f",
                    eng.get("checks_run", 0),
                    eng.get("total_actions", 0),
                    eng.get("action_success_rate", 1.0),
                )

                rtr = comps.get("router", {})
                logger.info(
                    "Router: checks=%d reroutes=%d "
                    "active_nodes=%d states=%s",
                    rtr.get("checks_run", 0),
                    rtr.get("total_reroutes", 0),
                    rtr.get("active_nodes", 3),
                    rtr.get("node_states", {}),
                )

                ret = comps.get("retrainer", {})
                logger.info(
                    "Retrainer: checks=%d retrains=%d "
                    "success_rate=%.3f",
                    ret.get("checks_run", 0),
                    ret.get("total_retrains", 0),
                    ret.get("retrain_success_rate", 1.0),
                )

                orch = comps.get("orchestrator", {})
                logger.info(
                    "Orchestrator: checks=%d "
                    "sequences=%d successful=%d active=%d",
                    orch.get("checks_run", 0),
                    orch.get("total_sequences", 0),
                    orch.get("successful_recoveries", 0),
                    orch.get("active_sequences", 0),
                )

            metrics_output = generate_latest(
                demo_registry
            ).decode("utf-8")
            score_lines = [
                line for line in metrics_output.split("\n")
                if "health_score" in line
                or "overall_score" in line
                or "fabric_status" in line
                and not line.startswith("#")
                and line.strip()
            ]
            if score_lines:
                logger.info("Health scores:")
                for line in score_lines[:10]:
                    if not line.startswith("#"):
                        logger.info("  %s", line)

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        adapter.stop()
        orchestrator.stop()
        retrainer.stop()
        router.stop()
        engine.stop()
        alerter.stop()
        simulator.stop()
        analyzer.stop()
        updater.stop()