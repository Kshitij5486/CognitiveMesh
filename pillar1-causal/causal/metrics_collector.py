import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

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
logger = logging.getLogger("cm.observability.collector")


class MetricSnapshot:
    def __init__(self, timestamp: str):
        self.timestamp = timestamp

        # Causal engine metrics
        self.causal_engine_ready: bool = False
        self.causal_buffer_size: int = 0
        self.causal_retrain_count: int = 0
        self.causal_retrain_duration_ms: dict = {}
        self.causal_effects: dict = {}
        self.causal_nodes_modeled: list = []

        # Predictive stack metrics
        self.trend_observations: int = 0
        self.simulations_run: int = 0
        self.alerts_fired_total: int = 0
        self.alerts_active: int = 0
        self.alerts_by_type: dict = {}
        self.alerts_by_severity: dict = {}
        self.worst_case_latency_ms: float = 0.0

        # Healing engine metrics
        self.healing_checks_run: int = 0
        self.healing_actions_total: int = 0
        self.healing_actions_successful: int = 0
        self.healing_actions_failed: int = 0
        self.healing_actions_by_type: dict = {}
        self.healing_cooldown_seconds: float = 30.0

        # Router metrics
        self.router_checks_run: int = 0
        self.router_total_reroutes: int = 0
        self.router_total_recoveries: int = 0
        self.router_active_decisions: int = 0
        self.router_node_states: dict = {}
        self.router_active_nodes: int = 0
        self.router_rerouted_nodes: int = 0
        self.router_isolated_nodes: int = 0
        self.router_causal_weights: dict = {}

        # Retrainer metrics
        self.retrainer_checks_run: int = 0
        self.retrainer_total: int = 0
        self.retrainer_successful: int = 0
        self.retrainer_failed: int = 0
        self.retrainer_drift_threshold_ms: float = 3.0
        self.retrainer_current_effects: dict = {}

        # Orchestrator metrics
        self.orchestrator_checks_run: int = 0
        self.orchestrator_total_sequences: int = 0
        self.orchestrator_successful_recoveries: int = 0
        self.orchestrator_failed_recoveries: int = 0
        self.orchestrator_active_sequences: int = 0
        self.orchestrator_recovery_durations: list = []

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "causal_engine": {
                "ready": self.causal_engine_ready,
                "buffer_size": self.causal_buffer_size,
                "retrain_count": self.causal_retrain_count,
                "retrain_duration_ms": self.causal_retrain_duration_ms,
                "effects": self.causal_effects,
                "nodes_modeled": self.causal_nodes_modeled,
            },
            "predictive_stack": {
                "trend_observations": self.trend_observations,
                "simulations_run": self.simulations_run,
                "alerts_fired_total": self.alerts_fired_total,
                "alerts_active": self.alerts_active,
                "alerts_by_type": self.alerts_by_type,
                "alerts_by_severity": self.alerts_by_severity,
                "worst_case_latency_ms": self.worst_case_latency_ms,
            },
            "healing_engine": {
                "checks_run": self.healing_checks_run,
                "actions_total": self.healing_actions_total,
                "actions_successful": self.healing_actions_successful,
                "actions_failed": self.healing_actions_failed,
                "actions_by_type": self.healing_actions_by_type,
            },
            "router": {
                "checks_run": self.router_checks_run,
                "total_reroutes": self.router_total_reroutes,
                "total_recoveries": self.router_total_recoveries,
                "active_decisions": self.router_active_decisions,
                "node_states": self.router_node_states,
                "active_nodes": self.router_active_nodes,
                "rerouted_nodes": self.router_rerouted_nodes,
                "isolated_nodes": self.router_isolated_nodes,
                "causal_weights": self.router_causal_weights,
            },
            "retrainer": {
                "checks_run": self.retrainer_checks_run,
                "total": self.retrainer_total,
                "successful": self.retrainer_successful,
                "failed": self.retrainer_failed,
                "drift_threshold_ms": self.retrainer_drift_threshold_ms,
                "current_effects": self.retrainer_current_effects,
            },
            "orchestrator": {
                "checks_run": self.orchestrator_checks_run,
                "total_sequences": self.orchestrator_total_sequences,
                "successful_recoveries": (
                    self.orchestrator_successful_recoveries
                ),
                "failed_recoveries": self.orchestrator_failed_recoveries,
                "active_sequences": self.orchestrator_active_sequences,
                "recovery_durations": self.orchestrator_recovery_durations,
            },
        }


class MetricsCollector:
    ALL_NODES = ["node-1", "node-2", "node-3"]
    MAX_SNAPSHOT_HISTORY = 60

    def __init__(
        self,
        updater,
        analyzer,
        simulator,
        alerter,
        engine,
        router,
        retrainer,
        orchestrator,
        collection_interval: float = 15.0,
    ):
        self.updater = updater
        self.analyzer = analyzer
        self.simulator = simulator
        self.alerter = alerter
        self.engine = engine
        self.router = router
        self.retrainer = retrainer
        self.orchestrator = orchestrator
        self.collection_interval = collection_interval

        self._snapshots: deque = deque(
            maxlen=self.MAX_SNAPSHOT_HISTORY
        )
        self._latest_snapshot: Optional[MetricSnapshot] = None
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._collection_count = 0
        self._collection_errors = 0
        self._start_time: Optional[float] = None

    def _collect_causal_engine(
        self, snapshot: MetricSnapshot
    ):
        try:
            status = self.updater.status()
            snapshot.causal_engine_ready = status.get(
                "is_ready", False
            )
            snapshot.causal_buffer_size = status.get(
                "buffer_size", 0
            )
            snapshot.causal_retrain_count = status.get(
                "retrain_count", 0
            )
            snapshot.causal_nodes_modeled = status.get(
                "nodes_modeled", []
            )

            effects = {}
            for node_id in self.ALL_NODES:
                node_snapshot = self.updater.get_current_snapshot(
                    node_id
                )
                if node_snapshot:
                    effects[node_id] = round(
                        abs(node_snapshot["effect"]), 4
                    )
            snapshot.causal_effects = effects

        except Exception as e:
            logger.debug("Causal engine collection error: %s", e)

    def _collect_predictive_stack(
        self, snapshot: MetricSnapshot
    ):
        try:
            analyzer_stat = self.analyzer.status()
            snapshot.trend_observations = analyzer_stat.get(
                "observations_collected", 0
            )

            simulator_stat = self.simulator.status()
            snapshot.simulations_run = simulator_stat.get(
                "simulations_run", 0
            )
            snapshot.worst_case_latency_ms = simulator_stat.get(
                "last_worst_case_ms", 0.0
            )

            alerter_stat = self.alerter.status()
            snapshot.alerts_fired_total = alerter_stat.get(
                "total_alerts_fired", 0
            )
            snapshot.alerts_active = alerter_stat.get(
                "active_alert_count", 0
            )

            active_alerts = self.alerter.get_active_alerts()
            by_type: dict = {}
            by_severity: dict = {}
            for alert in active_alerts:
                atype = alert.get("alert_type", "unknown")
                asev = alert.get("severity", "unknown")
                by_type[atype] = by_type.get(atype, 0) + 1
                by_severity[asev] = by_severity.get(asev, 0) + 1

            snapshot.alerts_by_type = by_type
            snapshot.alerts_by_severity = by_severity

        except Exception as e:
            logger.debug("Predictive stack collection error: %s", e)

    def _collect_healing_engine(
        self, snapshot: MetricSnapshot
    ):
        try:
            status = self.engine.status()
            snapshot.healing_checks_run = status.get(
                "checks_run", 0
            )
            snapshot.healing_actions_total = status.get(
                "total_actions", 0
            )
            snapshot.healing_actions_successful = status.get(
                "successful_actions", 0
            )
            snapshot.healing_actions_failed = status.get(
                "failed_actions", 0
            )

            history = self.engine.get_action_history(n=50)
            by_type: dict = {}
            for action in history:
                atype = action.get("action_type", "unknown")
                by_type[atype] = by_type.get(atype, 0) + 1
            snapshot.healing_actions_by_type = by_type

        except Exception as e:
            logger.debug("Healing engine collection error: %s", e)

    def _collect_router(self, snapshot: MetricSnapshot):
        try:
            status = self.router.status()
            snapshot.router_checks_run = status.get(
                "checks_run", 0
            )
            snapshot.router_total_reroutes = status.get(
                "total_reroutes", 0
            )
            snapshot.router_total_recoveries = status.get(
                "total_recoveries", 0
            )
            snapshot.router_active_decisions = status.get(
                "active_decisions", 0
            )
            snapshot.router_node_states = status.get(
                "node_states", {}
            )
            snapshot.router_active_nodes = status.get(
                "active_nodes", 0
            )
            snapshot.router_rerouted_nodes = status.get(
                "rerouted_nodes", 0
            )
            snapshot.router_isolated_nodes = status.get(
                "isolated_nodes", 0
            )

            weights = self.router._compute_causal_weights(
                self.ALL_NODES
            )
            snapshot.router_causal_weights = {
                k: round(v, 4) for k, v in weights.items()
            }

        except Exception as e:
            logger.debug("Router collection error: %s", e)

    def _collect_retrainer(self, snapshot: MetricSnapshot):
        try:
            status = self.retrainer.status()
            snapshot.retrainer_checks_run = status.get(
                "checks_run", 0
            )
            snapshot.retrainer_total = status.get(
                "total_retrains", 0
            )
            snapshot.retrainer_successful = status.get(
                "successful_retrains", 0
            )
            snapshot.retrainer_failed = status.get(
                "failed_retrains", 0
            )
            snapshot.retrainer_drift_threshold_ms = status.get(
                "drift_threshold_ms", 3.0
            )
            snapshot.retrainer_current_effects = status.get(
                "current_effects", {}
            )

        except Exception as e:
            logger.debug("Retrainer collection error: %s", e)

    def _collect_orchestrator(
        self, snapshot: MetricSnapshot
    ):
        try:
            status = self.orchestrator.status()
            snapshot.orchestrator_checks_run = status.get(
                "checks_run", 0
            )
            snapshot.orchestrator_total_sequences = status.get(
                "total_sequences", 0
            )
            snapshot.orchestrator_successful_recoveries = (
                status.get("successful_recoveries", 0)
            )
            snapshot.orchestrator_failed_recoveries = status.get(
                "failed_recoveries", 0
            )
            snapshot.orchestrator_active_sequences = status.get(
                "active_sequences", 0
            )

            history = self.orchestrator.get_sequence_history(n=20)
            durations = [
                s["duration_seconds"]
                for s in history
                if s.get("duration_seconds") is not None
            ]
            snapshot.orchestrator_recovery_durations = durations

        except Exception as e:
            logger.debug(
                "Orchestrator collection error: %s", e
            )

    def _collect_snapshot(self) -> MetricSnapshot:
        ts = datetime.now(timezone.utc).isoformat()
        snapshot = MetricSnapshot(timestamp=ts)

        self._collect_causal_engine(snapshot)
        self._collect_predictive_stack(snapshot)
        self._collect_healing_engine(snapshot)
        self._collect_router(snapshot)
        self._collect_retrainer(snapshot)
        self._collect_orchestrator(snapshot)

        return snapshot

    def _collection_loop(self):
        logger.info(
            "MetricsCollector started interval=%.1fs",
            self.collection_interval,
        )
        while self._running:
            time.sleep(self.collection_interval)
            try:
                start = time.perf_counter()
                snapshot = self._collect_snapshot()
                elapsed = (time.perf_counter() - start) * 1000

                with self._lock:
                    self._snapshots.append(snapshot)
                    self._latest_snapshot = snapshot
                    self._collection_count += 1

                logger.info(
                    "Metrics collected count=%d "
                    "elapsed=%.1fms "
                    "effects=%s "
                    "alerts_active=%d "
                    "healing_actions=%d "
                    "reroutes=%d "
                    "sequences=%d",
                    self._collection_count,
                    elapsed,
                    {
                        k: f"{v:.2f}ms"
                        for k, v in snapshot.causal_effects.items()
                    },
                    snapshot.alerts_active,
                    snapshot.healing_actions_total,
                    snapshot.router_total_reroutes,
                    snapshot.orchestrator_total_sequences,
                )

            except Exception as e:
                with self._lock:
                    self._collection_errors += 1
                logger.error("Metrics collection error: %s", e)

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._collection_loop,
            name="metrics-collector",
            daemon=True,
        )
        self._thread.start()
        logger.info("MetricsCollector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("MetricsCollector stopped")

    def get_latest_snapshot(
        self,
    ) -> Optional[MetricSnapshot]:
        with self._lock:
            return self._latest_snapshot

    def get_snapshot_history(
        self, n: int = 10
    ) -> list:
        with self._lock:
            snapshots = list(self._snapshots)
        return [s.to_dict() for s in snapshots[-n:]]

    def get_uptime_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return round(time.time() - self._start_time, 1)

    def status(self) -> dict:
        with self._lock:
            count = self._collection_count
            errors = self._collection_errors
            latest = self._latest_snapshot

        return {
            "running": self._running,
            "collection_interval_seconds": self.collection_interval,
            "collection_count": count,
            "collection_errors": errors,
            "uptime_seconds": self.get_uptime_seconds(),
            "latest_timestamp": (
                latest.timestamp if latest else None
            ),
            "snapshot_history_size": len(self._snapshots),
        }


if __name__ == "__main__":
    logger.info("Starting MetricsCollector demo")

    from streaming_updater import StreamingCausalUpdater
    from load_trend_analyzer import LoadTrendAnalyzer
    from causal_simulator import CausalSimulator
    from predictive_alerter import PredictiveAlerter
    from healing_action_engine import HealingActionEngine
    from query_router import QueryRouter, RoutingStrategy
    from auto_retrainer import AutoRetrainer
    from recovery_orchestrator import RecoveryOrchestrator

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

    collector = MetricsCollector(
        updater=updater,
        analyzer=analyzer,
        simulator=simulator,
        alerter=alerter,
        engine=engine,
        router=router,
        retrainer=retrainer,
        orchestrator=orchestrator,
        collection_interval=15.0,
    )
    collector.start()

    logger.info(
        "Full stack running. Load generator in another terminal."
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
                "=== METRICS CYCLE %d ===", cycle
            )

            col_status = collector.status()
            logger.info(
                "Collector: count=%d errors=%d uptime=%.1fs",
                col_status["collection_count"],
                col_status["collection_errors"],
                col_status["uptime_seconds"],
            )

            snap = collector.get_latest_snapshot()
            if snap:
                logger.info(
                    "Snapshot: causal_effects=%s "
                    "alerts_active=%d "
                    "healing_total=%d "
                    "reroutes=%d "
                    "retrains=%d "
                    "sequences=%d",
                    {
                        k: f"{v:.2f}ms"
                        for k, v in snap.causal_effects.items()
                    },
                    snap.alerts_active,
                    snap.healing_actions_total,
                    snap.router_total_reroutes,
                    snap.retrainer_total,
                    snap.orchestrator_total_sequences,
                )
                logger.info(
                    "Router weights: %s",
                    {
                        k: f"{v:.3f}"
                        for k, v in snap.router_causal_weights.items()
                    },
                )
                logger.info(
                    "Node states: %s",
                    snap.router_node_states,
                )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        collector.stop()
        orchestrator.stop()
        retrainer.stop()
        router.stop()
        engine.stop()
        alerter.stop()
        simulator.stop()
        analyzer.stop()
        updater.stop()