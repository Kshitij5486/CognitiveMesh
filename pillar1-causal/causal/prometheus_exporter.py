import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from prometheus_client import (
    Counter, Gauge, Histogram, Info,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST,
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
logger = logging.getLogger("cm.observability.exporter")

# ── Registry ──────────────────────────────────────────────
registry = CollectorRegistry()

# ── Causal Engine ─────────────────────────────────────────
causal_engine_ready = Gauge(
    "cognitivemesh_causal_engine_ready",
    "Whether the causal engine is ready (1=ready, 0=not ready)",
    registry=registry,
)
causal_buffer_size = Gauge(
    "cognitivemesh_causal_buffer_size",
    "Current telemetry buffer size",
    registry=registry,
)
causal_retrain_total = Gauge(
    "cognitivemesh_causal_retrain_total",
    "Total number of causal model retrain cycles completed",
    registry=registry,
)
causal_effect_ms = Gauge(
    "cognitivemesh_causal_effect_ms",
    "Estimated causal effect of load on latency per node (ms)",
    ["node"],
    registry=registry,
)
causal_routing_weight = Gauge(
    "cognitivemesh_causal_routing_weight",
    "Causal routing weight per node (higher = more traffic)",
    ["node"],
    registry=registry,
)

# ── Predictive Stack ──────────────────────────────────────
predictive_trend_observations = Gauge(
    "cognitivemesh_predictive_trend_observations_total",
    "Total trend observations collected",
    registry=registry,
)
predictive_simulations_total = Gauge(
    "cognitivemesh_predictive_simulations_total",
    "Total simulation cycles run",
    registry=registry,
)
predictive_alerts_fired_total = Gauge(
    "cognitivemesh_predictive_alerts_fired_total",
    "Total predictive alerts fired since start",
    registry=registry,
)
predictive_alerts_active = Gauge(
    "cognitivemesh_predictive_alerts_active",
    "Currently active (unacknowledged) predictive alerts",
    registry=registry,
)
predictive_alerts_by_type = Gauge(
    "cognitivemesh_predictive_alerts_active_by_type",
    "Active alerts broken down by alert type",
    ["alert_type"],
    registry=registry,
)
predictive_alerts_by_severity = Gauge(
    "cognitivemesh_predictive_alerts_active_by_severity",
    "Active alerts broken down by severity",
    ["severity"],
    registry=registry,
)
predictive_worst_case_latency_ms = Gauge(
    "cognitivemesh_predictive_worst_case_latency_ms",
    "Worst-case predicted latency across cluster (ms)",
    registry=registry,
)

# ── Healing Engine ────────────────────────────────────────
healing_checks_total = Gauge(
    "cognitivemesh_healing_checks_total",
    "Total healing check cycles run",
    registry=registry,
)
healing_actions_total = Gauge(
    "cognitivemesh_healing_actions_total",
    "Total healing actions executed",
    registry=registry,
)
healing_actions_successful = Gauge(
    "cognitivemesh_healing_actions_successful_total",
    "Total successful healing actions",
    registry=registry,
)
healing_actions_failed = Gauge(
    "cognitivemesh_healing_actions_failed_total",
    "Total failed healing actions",
    registry=registry,
)
healing_actions_by_type = Gauge(
    "cognitivemesh_healing_actions_by_type_total",
    "Healing actions broken down by action type",
    ["action_type"],
    registry=registry,
)
healing_success_rate = Gauge(
    "cognitivemesh_healing_success_rate",
    "Ratio of successful healing actions to total (0.0–1.0)",
    registry=registry,
)

# ── Router ────────────────────────────────────────────────
router_checks_total = Gauge(
    "cognitivemesh_router_checks_total",
    "Total router check cycles run",
    registry=registry,
)
router_reroutes_total = Gauge(
    "cognitivemesh_router_reroutes_total",
    "Total reroute decisions made",
    registry=registry,
)
router_recoveries_total = Gauge(
    "cognitivemesh_router_recoveries_total",
    "Total node recoveries to active routing pool",
    registry=registry,
)
router_active_decisions = Gauge(
    "cognitivemesh_router_active_decisions",
    "Currently active routing decisions",
    registry=registry,
)
router_node_state = Gauge(
    "cognitivemesh_router_node_state",
    "Node routing state (1=active, 0=rerouted/isolated)",
    ["node"],
    registry=registry,
)
router_active_nodes = Gauge(
    "cognitivemesh_router_active_nodes",
    "Number of nodes currently receiving traffic",
    registry=registry,
)
router_rerouted_nodes = Gauge(
    "cognitivemesh_router_rerouted_nodes",
    "Number of nodes currently rerouted",
    registry=registry,
)
router_isolated_nodes = Gauge(
    "cognitivemesh_router_isolated_nodes",
    "Number of nodes currently isolated",
    registry=registry,
)

# ── Retrainer ─────────────────────────────────────────────
retrainer_checks_total = Gauge(
    "cognitivemesh_retrainer_checks_total",
    "Total retrainer check cycles run",
    registry=registry,
)
retrainer_total = Gauge(
    "cognitivemesh_retrainer_triggered_total",
    "Total retrain operations triggered",
    registry=registry,
)
retrainer_successful = Gauge(
    "cognitivemesh_retrainer_successful_total",
    "Total successful retrain operations",
    registry=registry,
)
retrainer_failed = Gauge(
    "cognitivemesh_retrainer_failed_total",
    "Total failed retrain operations",
    registry=registry,
)
retrainer_drift_threshold_ms = Gauge(
    "cognitivemesh_retrainer_drift_threshold_ms",
    "Configured causal effect drift threshold (ms)",
    registry=registry,
)
retrainer_current_effect_ms = Gauge(
    "cognitivemesh_retrainer_current_effect_ms",
    "Most recently observed causal effect per node (ms)",
    ["node"],
    registry=registry,
)

# ── Orchestrator ──────────────────────────────────────────
orchestrator_checks_total = Gauge(
    "cognitivemesh_orchestrator_checks_total",
    "Total orchestrator check cycles run",
    registry=registry,
)
orchestrator_sequences_total = Gauge(
    "cognitivemesh_orchestrator_sequences_total",
    "Total recovery sequences initiated",
    registry=registry,
)
orchestrator_successful_recoveries = Gauge(
    "cognitivemesh_orchestrator_successful_recoveries_total",
    "Total recovery sequences completed successfully",
    registry=registry,
)
orchestrator_failed_recoveries = Gauge(
    "cognitivemesh_orchestrator_failed_recoveries_total",
    "Total recovery sequences that failed",
    registry=registry,
)
orchestrator_active_sequences = Gauge(
    "cognitivemesh_orchestrator_active_sequences",
    "Currently running recovery sequences",
    registry=registry,
)
orchestrator_recovery_duration_avg_seconds = Gauge(
    "cognitivemesh_orchestrator_recovery_duration_avg_seconds",
    "Average recovery sequence duration (seconds)",
    registry=registry,
)
orchestrator_recovery_duration_max_seconds = Gauge(
    "cognitivemesh_orchestrator_recovery_duration_max_seconds",
    "Maximum recovery sequence duration observed (seconds)",
    registry=registry,
)

# ── Exporter self-metrics ─────────────────────────────────
exporter_collections_total = Gauge(
    "cognitivemesh_exporter_collections_total",
    "Total metric collection cycles completed by exporter",
    registry=registry,
)
exporter_collection_errors_total = Gauge(
    "cognitivemesh_exporter_collection_errors_total",
    "Total metric collection errors",
    registry=registry,
)
exporter_uptime_seconds = Gauge(
    "cognitivemesh_exporter_uptime_seconds",
    "Exporter uptime in seconds",
    registry=registry,
)
exporter_scrape_latency_ms = Gauge(
    "cognitivemesh_exporter_last_scrape_latency_ms",
    "Latency of last metric collection cycle (ms)",
    registry=registry,
)

# ── Info ──────────────────────────────────────────────────
platform_info = Info(
    "cognitivemesh_platform",
    "CognitiveMesh platform metadata",
    registry=registry,
)
platform_info.info({
    "version": "0.8.0",
    "sprint": "8",
    "component": "prometheus_exporter",
    "port": "8088",
})


class PrometheusExporter:
    def __init__(self, collector, scrape_interval: float = 15.0):
        self.collector = collector
        self.scrape_interval = scrape_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._scrape_count = 0
        self._scrape_errors = 0
        self._start_time: Optional[float] = None

    def _update_causal_metrics(self, snapshot):
        causal_engine_ready.set(
            1.0 if snapshot.causal_engine_ready else 0.0
        )
        causal_buffer_size.set(snapshot.causal_buffer_size)
        causal_retrain_total.set(snapshot.causal_retrain_count)

        for node_id, effect in snapshot.causal_effects.items():
            causal_effect_ms.labels(node=node_id).set(effect)

        for node_id, weight in snapshot.router_causal_weights.items():
            causal_routing_weight.labels(node=node_id).set(weight)

    def _update_predictive_metrics(self, snapshot):
        predictive_trend_observations.set(
            snapshot.trend_observations
        )
        predictive_simulations_total.set(snapshot.simulations_run)
        predictive_alerts_fired_total.set(
            snapshot.alerts_fired_total
        )
        predictive_alerts_active.set(snapshot.alerts_active)
        predictive_worst_case_latency_ms.set(
            snapshot.worst_case_latency_ms
        )

        known_types = [
            "latency_rising", "load_spike",
            "trend_acceleration", "causal_threshold",
            "cluster_degradation",
        ]
        for atype in known_types:
            predictive_alerts_by_type.labels(
                alert_type=atype
            ).set(snapshot.alerts_by_type.get(atype, 0))

        known_severities = [
            "warning", "critical", "emergency", "normal"
        ]
        for sev in known_severities:
            predictive_alerts_by_severity.labels(
                severity=sev
            ).set(snapshot.alerts_by_severity.get(sev, 0))

    def _update_healing_metrics(self, snapshot):
        healing_checks_total.set(snapshot.healing_checks_run)
        healing_actions_total.set(snapshot.healing_actions_total)
        healing_actions_successful.set(
            snapshot.healing_actions_successful
        )
        healing_actions_failed.set(snapshot.healing_actions_failed)

        total = snapshot.healing_actions_total
        success = snapshot.healing_actions_successful
        rate = (success / total) if total > 0 else 1.0
        healing_success_rate.set(rate)

        known_action_types = [
            "rebalance", "reroute", "retrain",
            "isolate", "alert_operator", "no_action",
        ]
        for atype in known_action_types:
            healing_actions_by_type.labels(
                action_type=atype
            ).set(snapshot.healing_actions_by_type.get(atype, 0))

    def _update_router_metrics(self, snapshot):
        router_checks_total.set(snapshot.router_checks_run)
        router_reroutes_total.set(snapshot.router_total_reroutes)
        router_recoveries_total.set(
            snapshot.router_total_recoveries
        )
        router_active_decisions.set(
            snapshot.router_active_decisions
        )
        router_active_nodes.set(snapshot.router_active_nodes)
        router_rerouted_nodes.set(snapshot.router_rerouted_nodes)
        router_isolated_nodes.set(snapshot.router_isolated_nodes)

        for node_id, state in snapshot.router_node_states.items():
            is_active = 1.0 if state == "active" else 0.0
            router_node_state.labels(node=node_id).set(is_active)

    def _update_retrainer_metrics(self, snapshot):
        retrainer_checks_total.set(snapshot.retrainer_checks_run)
        retrainer_total.set(snapshot.retrainer_total)
        retrainer_successful.set(snapshot.retrainer_successful)
        retrainer_failed.set(snapshot.retrainer_failed)
        retrainer_drift_threshold_ms.set(
            snapshot.retrainer_drift_threshold_ms
        )

        for node_id, effect in (
            snapshot.retrainer_current_effects.items()
        ):
            retrainer_current_effect_ms.labels(
                node=node_id
            ).set(effect)

    def _update_orchestrator_metrics(self, snapshot):
        orchestrator_checks_total.set(
            snapshot.orchestrator_checks_run
        )
        orchestrator_sequences_total.set(
            snapshot.orchestrator_total_sequences
        )
        orchestrator_successful_recoveries.set(
            snapshot.orchestrator_successful_recoveries
        )
        orchestrator_failed_recoveries.set(
            snapshot.orchestrator_failed_recoveries
        )
        orchestrator_active_sequences.set(
            snapshot.orchestrator_active_sequences
        )

        durations = snapshot.orchestrator_recovery_durations
        if durations:
            orchestrator_recovery_duration_avg_seconds.set(
                sum(durations) / len(durations)
            )
            orchestrator_recovery_duration_max_seconds.set(
                max(durations)
            )
        else:
            orchestrator_recovery_duration_avg_seconds.set(0.0)
            orchestrator_recovery_duration_max_seconds.set(0.0)

    def _update_exporter_metrics(self, elapsed_ms: float):
        self._scrape_count += 1
        exporter_collections_total.set(self._scrape_count)
        exporter_collection_errors_total.set(
            self._scrape_errors
        )
        exporter_scrape_latency_ms.set(elapsed_ms)
        if self._start_time:
            exporter_uptime_seconds.set(
                time.time() - self._start_time
            )

    def _scrape_and_update(self):
        start = time.perf_counter()
        snapshot = self.collector.get_latest_snapshot()

        if snapshot is None:
            self._scrape_errors += 1
            return

        try:
            self._update_causal_metrics(snapshot)
            self._update_predictive_metrics(snapshot)
            self._update_healing_metrics(snapshot)
            self._update_router_metrics(snapshot)
            self._update_retrainer_metrics(snapshot)
            self._update_orchestrator_metrics(snapshot)

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._update_exporter_metrics(elapsed_ms)

            logger.info(
                "Prometheus metrics updated scrape=%d "
                "elapsed=%.2fms "
                "active_alerts=%d "
                "healing_actions=%d "
                "causal_effects=%s",
                self._scrape_count,
                elapsed_ms,
                snapshot.alerts_active,
                snapshot.healing_actions_total,
                {
                    k: f"{v:.2f}"
                    for k, v in snapshot.causal_effects.items()
                },
            )

        except Exception as e:
            self._scrape_errors += 1
            logger.error("Prometheus update error: %s", e)

    def _exporter_loop(self):
        logger.info(
            "PrometheusExporter loop started interval=%.1fs",
            self.scrape_interval,
        )
        while self._running:
            time.sleep(self.scrape_interval)
            self._scrape_and_update()

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._exporter_loop,
            name="prometheus-exporter",
            daemon=True,
        )
        self._thread.start()
        logger.info("PrometheusExporter started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("PrometheusExporter stopped")

    def status(self) -> dict:
        return {
            "running": self._running,
            "scrape_count": self._scrape_count,
            "scrape_errors": self._scrape_errors,
            "scrape_interval_seconds": self.scrape_interval,
            "uptime_seconds": round(
                time.time() - self._start_time, 1
            ) if self._start_time else 0.0,
            "metrics_count": 35,
        }


# ── FastAPI app ───────────────────────────────────────────

exporter_instance: Optional[PrometheusExporter] = None
start_time_global: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global exporter_instance, start_time_global

    start_time_global = datetime.now(timezone.utc).isoformat()
    logger.info("Starting Prometheus Exporter API v0.8.0")

    from streaming_updater import StreamingCausalUpdater
    from load_trend_analyzer import LoadTrendAnalyzer
    from causal_simulator import CausalSimulator
    from predictive_alerter import PredictiveAlerter
    from healing_action_engine import HealingActionEngine
    from query_router import QueryRouter, RoutingStrategy
    from auto_retrainer import AutoRetrainer
    from recovery_orchestrator import RecoveryOrchestrator
    from metrics_collector import MetricsCollector

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

    exporter_instance = PrometheusExporter(
        collector=collector,
        scrape_interval=15.0,
    )
    exporter_instance.start()

    logger.info(
        "Prometheus Exporter fully started on port 8088"
    )
    yield

    logger.info("Shutting down Prometheus Exporter")
    for component in [
        exporter_instance, collector, orchestrator,
        retrainer, router, engine, alerter,
        simulator, analyzer, updater,
    ]:
        if component:
            component.stop()


app = FastAPI(
    title="CognitiveMesh Prometheus Exporter",
    description=(
        "Prometheus-compatible metrics endpoint for the "
        "CognitiveMesh self-healing distributed computing fabric. "
        "Exposes 35+ metrics across causal engine, predictive "
        "stack, healing engine, router, retrainer, and "
        "recovery orchestrator."
    ),
    version="0.8.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health")
def health():
    status = (
        exporter_instance.status()
        if exporter_instance else {}
    )
    uptime = 0.0
    if start_time_global:
        uptime = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(start_time_global)
        ).total_seconds()
    return {
        "status": "healthy",
        "version": "0.8.0",
        "port": 8088,
        "uptime_seconds": round(uptime, 1),
        "exporter": status,
        "metrics_families": 35,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics/catalog")
def metrics_catalog():
    return {
        "total_metrics": 35,
        "families": {
            "causal_engine": [
                "cognitivemesh_causal_engine_ready",
                "cognitivemesh_causal_buffer_size",
                "cognitivemesh_causal_retrain_total",
                "cognitivemesh_causal_effect_ms{node}",
                "cognitivemesh_causal_routing_weight{node}",
            ],
            "predictive_stack": [
                "cognitivemesh_predictive_trend_observations_total",
                "cognitivemesh_predictive_simulations_total",
                "cognitivemesh_predictive_alerts_fired_total",
                "cognitivemesh_predictive_alerts_active",
                "cognitivemesh_predictive_alerts_active_by_type{alert_type}",
                "cognitivemesh_predictive_alerts_active_by_severity{severity}",
                "cognitivemesh_predictive_worst_case_latency_ms",
            ],
            "healing_engine": [
                "cognitivemesh_healing_checks_total",
                "cognitivemesh_healing_actions_total",
                "cognitivemesh_healing_actions_successful_total",
                "cognitivemesh_healing_actions_failed_total",
                "cognitivemesh_healing_actions_by_type_total{action_type}",
                "cognitivemesh_healing_success_rate",
            ],
            "router": [
                "cognitivemesh_router_checks_total",
                "cognitivemesh_router_reroutes_total",
                "cognitivemesh_router_recoveries_total",
                "cognitivemesh_router_active_decisions",
                "cognitivemesh_router_node_state{node}",
                "cognitivemesh_router_active_nodes",
                "cognitivemesh_router_rerouted_nodes",
                "cognitivemesh_router_isolated_nodes",
            ],
            "retrainer": [
                "cognitivemesh_retrainer_checks_total",
                "cognitivemesh_retrainer_triggered_total",
                "cognitivemesh_retrainer_successful_total",
                "cognitivemesh_retrainer_failed_total",
                "cognitivemesh_retrainer_drift_threshold_ms",
                "cognitivemesh_retrainer_current_effect_ms{node}",
            ],
            "orchestrator": [
                "cognitivemesh_orchestrator_checks_total",
                "cognitivemesh_orchestrator_sequences_total",
                "cognitivemesh_orchestrator_successful_recoveries_total",
                "cognitivemesh_orchestrator_failed_recoveries_total",
                "cognitivemesh_orchestrator_active_sequences",
                "cognitivemesh_orchestrator_recovery_duration_avg_seconds",
                "cognitivemesh_orchestrator_recovery_duration_max_seconds",
            ],
            "exporter": [
                "cognitivemesh_exporter_collections_total",
                "cognitivemesh_exporter_collection_errors_total",
                "cognitivemesh_exporter_uptime_seconds",
                "cognitivemesh_exporter_last_scrape_latency_ms",
                "cognitivemesh_platform_info",
            ],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "prometheus_exporter:app",
        host="0.0.0.0",
        port=8088,
        reload=False,
        log_level="info",
    )