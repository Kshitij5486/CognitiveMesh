import logging
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
from load_trend_analyzer import LoadTrendAnalyzer
from causal_simulator import CausalSimulator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.predictive.alerter")


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    LATENCY_RISING = "latency_rising"
    LOAD_SPIKE = "load_spike"
    CAUSAL_THRESHOLD = "causal_threshold"
    TREND_ACCELERATION = "trend_acceleration"
    CLUSTER_DEGRADATION = "cluster_degradation"


class PredictiveAlert:
    def __init__(
        self,
        alert_id: str,
        node_id: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        current_value: float,
        predicted_value: float,
        threshold: float,
        horizon_minutes: float,
        message: str,
        timestamp: str,
    ):
        self.alert_id = alert_id
        self.node_id = node_id
        self.alert_type = alert_type
        self.severity = severity
        self.current_value = current_value
        self.predicted_value = predicted_value
        self.threshold = threshold
        self.horizon_minutes = horizon_minutes
        self.message = message
        self.timestamp = timestamp
        self.acknowledged = False

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "node_id": self.node_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "current_value": round(self.current_value, 2),
            "predicted_value": round(self.predicted_value, 2),
            "threshold": round(self.threshold, 2),
            "horizon_minutes": self.horizon_minutes,
            "message": self.message,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
        }


class PredictiveAlerter:
    LATENCY_WARN_MS = 150.0
    LATENCY_CRITICAL_MS = 300.0
    LATENCY_EMERGENCY_MS = 600.0
    LOAD_SPIKE_THRESHOLD = 2.0
    TREND_ACCELERATION_THRESHOLD = 5.0
    CHECK_INTERVAL_SECONDS = 15.0
    MAX_ALERT_HISTORY = 100

    def __init__(
        self,
        updater: StreamingCausalUpdater,
        analyzer: LoadTrendAnalyzer,
        simulator: CausalSimulator,
        check_interval: float = 15.0,
    ):
        self.updater = updater
        self.analyzer = analyzer
        self.simulator = simulator
        self.check_interval = check_interval

        self._active_alerts: dict[str, PredictiveAlert] = {}
        self._alert_history: deque = deque(maxlen=self.MAX_ALERT_HISTORY)
        self._alerts_lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._alert_counter = 0
        self._checks_run = 0
        self._total_alerts_fired = 0

    def _make_alert_id(self) -> str:
        self._alert_counter += 1
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        return f"alert-{ts}-{self._alert_counter:04d}"

    def _fire_alert(self, alert: PredictiveAlert):
        with self._alerts_lock:
            self._active_alerts[alert.alert_id] = alert
            self._alert_history.append(alert)
            self._total_alerts_fired += 1

        if alert.severity == AlertSeverity.EMERGENCY:
            logger.critical(
                "EMERGENCY ALERT [%s] node=%s %s "
                "current=%.1f predicted=%.1f threshold=%.1f "
                "in %.0fmin",
                alert.alert_type.value,
                alert.node_id,
                alert.message,
                alert.current_value,
                alert.predicted_value,
                alert.threshold,
                alert.horizon_minutes,
            )
        elif alert.severity == AlertSeverity.CRITICAL:
            logger.critical(
                "CRITICAL ALERT [%s] node=%s %s "
                "current=%.1f predicted=%.1f threshold=%.1f "
                "in %.0fmin",
                alert.alert_type.value,
                alert.node_id,
                alert.message,
                alert.current_value,
                alert.predicted_value,
                alert.threshold,
                alert.horizon_minutes,
            )
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(
                "WARNING ALERT [%s] node=%s %s "
                "current=%.1f predicted=%.1f threshold=%.1f "
                "in %.0fmin",
                alert.alert_type.value,
                alert.node_id,
                alert.message,
                alert.current_value,
                alert.predicted_value,
                alert.threshold,
                alert.horizon_minutes,
            )
        else:
            logger.info(
                "INFO ALERT [%s] node=%s %s",
                alert.alert_type.value,
                alert.node_id,
                alert.message,
            )

    def _clear_resolved_alerts(
        self, node_id: str, alert_type: AlertType
    ):
        with self._alerts_lock:
            to_remove = [
                aid for aid, alert in self._active_alerts.items()
                if alert.node_id == node_id
                and alert.alert_type == alert_type
            ]
            for aid in to_remove:
                del self._active_alerts[aid]

    def _check_latency_alerts(
        self,
        node_id: str,
        node_sim: dict,
    ):
        current_latency = node_sim["current_latency_ms"]
        worst_latency = node_sim["worst_case_latency_ms"]

        for scenario in node_sim["scenarios"]:
            predicted = scenario["projected_latency_ms"]
            horizon = scenario["horizon_minutes"]

            if predicted >= self.LATENCY_EMERGENCY_MS:
                alert = PredictiveAlert(
                    alert_id=self._make_alert_id(),
                    node_id=node_id,
                    alert_type=AlertType.LATENCY_RISING,
                    severity=AlertSeverity.EMERGENCY,
                    current_value=current_latency,
                    predicted_value=predicted,
                    threshold=self.LATENCY_EMERGENCY_MS,
                    horizon_minutes=horizon,
                    message=(
                        f"Latency predicted to reach {predicted:.1f}ms "
                        f"in {horizon:.0f} minutes — "
                        f"EMERGENCY threshold {self.LATENCY_EMERGENCY_MS:.0f}ms"
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self._fire_alert(alert)
                return

            elif predicted >= self.LATENCY_CRITICAL_MS:
                alert = PredictiveAlert(
                    alert_id=self._make_alert_id(),
                    node_id=node_id,
                    alert_type=AlertType.LATENCY_RISING,
                    severity=AlertSeverity.CRITICAL,
                    current_value=current_latency,
                    predicted_value=predicted,
                    threshold=self.LATENCY_CRITICAL_MS,
                    horizon_minutes=horizon,
                    message=(
                        f"Latency predicted to reach {predicted:.1f}ms "
                        f"in {horizon:.0f} minutes — "
                        f"exceeds critical threshold "
                        f"{self.LATENCY_CRITICAL_MS:.0f}ms"
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self._fire_alert(alert)
                return

            elif predicted >= self.LATENCY_WARN_MS:
                alert = PredictiveAlert(
                    alert_id=self._make_alert_id(),
                    node_id=node_id,
                    alert_type=AlertType.LATENCY_RISING,
                    severity=AlertSeverity.WARNING,
                    current_value=current_latency,
                    predicted_value=predicted,
                    threshold=self.LATENCY_WARN_MS,
                    horizon_minutes=horizon,
                    message=(
                        f"Latency predicted to reach {predicted:.1f}ms "
                        f"in {horizon:.0f} minutes — "
                        f"approaching warn threshold "
                        f"{self.LATENCY_WARN_MS:.0f}ms"
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self._fire_alert(alert)
                return

        self._clear_resolved_alerts(node_id, AlertType.LATENCY_RISING)

    def _check_load_spike_alerts(
        self,
        node_id: str,
        node_sim: dict,
        trend: dict,
    ):
        current_load = node_sim["current_load"]
        if current_load < 0.1:
            return

        rate = trend.get("change_rate_per_minute", 0.0)
        if rate <= 0:
            self._clear_resolved_alerts(node_id, AlertType.LOAD_SPIKE)
            return

        projected_5min = current_load + rate * 5
        ratio = projected_5min / current_load if current_load > 0 else 1.0

        if ratio >= self.LOAD_SPIKE_THRESHOLD:
            severity = (
                AlertSeverity.CRITICAL
                if ratio >= self.LOAD_SPIKE_THRESHOLD * 1.5
                else AlertSeverity.WARNING
            )
            alert = PredictiveAlert(
                alert_id=self._make_alert_id(),
                node_id=node_id,
                alert_type=AlertType.LOAD_SPIKE,
                severity=severity,
                current_value=current_load,
                predicted_value=projected_5min,
                threshold=current_load * self.LOAD_SPIKE_THRESHOLD,
                horizon_minutes=5.0,
                message=(
                    f"Load spike predicted: {current_load:.1f} → "
                    f"{projected_5min:.1f} queries in 5 minutes "
                    f"({ratio:.1f}x increase)"
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._fire_alert(alert)
        else:
            self._clear_resolved_alerts(node_id, AlertType.LOAD_SPIKE)

    def _check_trend_acceleration(
        self,
        node_id: str,
        trend: dict,
    ):
        rate = trend.get("change_rate_per_minute", 0.0)
        if abs(rate) >= self.TREND_ACCELERATION_THRESHOLD:
            direction = trend.get("direction", "unknown")
            snapshot = self.updater.get_current_snapshot(node_id)
            causal_effect = abs(snapshot["effect"]) if snapshot else 0.0

            alert = PredictiveAlert(
                alert_id=self._make_alert_id(),
                node_id=node_id,
                alert_type=AlertType.TREND_ACCELERATION,
                severity=AlertSeverity.WARNING,
                current_value=abs(rate),
                predicted_value=abs(rate) * causal_effect,
                threshold=self.TREND_ACCELERATION_THRESHOLD,
                horizon_minutes=5.0,
                message=(
                    f"Rapid load {direction} detected: "
                    f"{rate:+.2f} queries/min — "
                    f"causal effect {causal_effect:.1f}ms/query"
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._fire_alert(alert)
        else:
            self._clear_resolved_alerts(
                node_id, AlertType.TREND_ACCELERATION
            )

    def _check_cluster_degradation(self, cluster_sim: dict):
        worst = cluster_sim.get("cluster_worst_case_latency_ms", 0.0)
        highest_risk = cluster_sim.get("highest_risk_node", "unknown")

        node_sims = cluster_sim.get("node_simulations", {})
        nodes_above_warn = sum(
            1 for ns in node_sims.values()
            if ns["worst_case_latency_ms"] >= self.LATENCY_WARN_MS
        )

        if nodes_above_warn >= 2 and worst >= self.LATENCY_CRITICAL_MS:
            alert = PredictiveAlert(
                alert_id=self._make_alert_id(),
                node_id="cluster",
                alert_type=AlertType.CLUSTER_DEGRADATION,
                severity=AlertSeverity.CRITICAL,
                current_value=float(nodes_above_warn),
                predicted_value=worst,
                threshold=self.LATENCY_CRITICAL_MS,
                horizon_minutes=5.0,
                message=(
                    f"Cluster-wide degradation predicted: "
                    f"{nodes_above_warn} nodes above warn threshold, "
                    f"worst case {worst:.1f}ms on {highest_risk}"
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._fire_alert(alert)
        else:
            self._clear_resolved_alerts(
                "cluster", AlertType.CLUSTER_DEGRADATION
            )

    def _check_cycle(self):
        self._checks_run += 1

        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            return

        cluster_sim = self.simulator.get_latest_simulation()
        if not cluster_sim:
            cluster_sim = self.simulator.simulate_now()
        if not cluster_sim:
            return

        trend_analyses = self.analyzer.get_latest_analyses()

        for node_id in ["node-1", "node-2", "node-3"]:
            node_sim = cluster_sim.get(
                "node_simulations", {}
            ).get(node_id)
            if not node_sim:
                continue

            trend = trend_analyses.get(node_id, {})

            self._check_latency_alerts(node_id, node_sim)
            self._check_load_spike_alerts(node_id, node_sim, trend)
            self._check_trend_acceleration(node_id, trend)

        self._check_cluster_degradation(cluster_sim)

        with self._alerts_lock:
            active_count = len(self._active_alerts)

        if active_count > 0:
            logger.info(
                "Alert check cycle=%d active_alerts=%d "
                "total_fired=%d",
                self._checks_run,
                active_count,
                self._total_alerts_fired,
            )
        else:
            logger.info(
                "Alert check cycle=%d all_clear "
                "total_fired=%d",
                self._checks_run,
                self._total_alerts_fired,
            )

    def _alert_loop(self):
        logger.info(
            "PredictiveAlerter started check_interval=%.1fs "
            "warn=%.0fms critical=%.0fms emergency=%.0fms",
            self.check_interval,
            self.LATENCY_WARN_MS,
            self.LATENCY_CRITICAL_MS,
            self.LATENCY_EMERGENCY_MS,
        )
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._check_cycle()
            except Exception as e:
                logger.error("Alert check error: %s", e)

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._alert_loop,
            name="predictive-alerter",
            daemon=True,
        )
        self._thread.start()
        logger.info("PredictiveAlerter started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("PredictiveAlerter stopped")

    def get_active_alerts(self) -> list:
        with self._alerts_lock:
            return [
                alert.to_dict()
                for alert in self._active_alerts.values()
            ]

    def get_alert_history(self, n: int = 20) -> list:
        with self._alerts_lock:
            history = list(self._alert_history)
        return [a.to_dict() for a in history[-n:]]

    def acknowledge_alert(self, alert_id: str) -> bool:
        with self._alerts_lock:
            alert = self._active_alerts.get(alert_id)
            if alert:
                alert.acknowledged = True
                return True
        return False

    def clear_alert(self, alert_id: str) -> bool:
        with self._alerts_lock:
            if alert_id in self._active_alerts:
                del self._active_alerts[alert_id]
                return True
        return False

    def status(self) -> dict:
        with self._alerts_lock:
            active = list(self._active_alerts.values())
        return {
            "running": self._running,
            "checks_run": self._checks_run,
            "total_alerts_fired": self._total_alerts_fired,
            "active_alert_count": len(active),
            "active_alerts": [a.to_dict() for a in active],
            "thresholds": {
                "latency_warn_ms": self.LATENCY_WARN_MS,
                "latency_critical_ms": self.LATENCY_CRITICAL_MS,
                "latency_emergency_ms": self.LATENCY_EMERGENCY_MS,
                "load_spike_threshold": self.LOAD_SPIKE_THRESHOLD,
                "trend_acceleration_threshold": (
                    self.TREND_ACCELERATION_THRESHOLD
                ),
            },
        }


if __name__ == "__main__":
    logger.info("Starting predictive alerter demo")

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

    logger.info(
        "Full predictive stack running. "
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

            logger.info("=== ALERT CYCLE %d ===", cycle)

            alerter_status = alerter.status()
            logger.info(
                "Alerter: checks=%d total_fired=%d active=%d",
                alerter_status["checks_run"],
                alerter_status["total_alerts_fired"],
                alerter_status["active_alert_count"],
            )

            active = alerter.get_active_alerts()
            if active:
                for alert in active:
                    logger.info(
                        "  ACTIVE [%s] node=%s severity=%s "
                        "current=%.1f predicted=%.1f "
                        "horizon=%.0fmin",
                        alert["alert_type"],
                        alert["node_id"],
                        alert["severity"].upper(),
                        alert["current_value"],
                        alert["predicted_value"],
                        alert["horizon_minutes"],
                    )
            else:
                logger.info("  All clear — no active alerts")

            sim = simulator.get_latest_simulation()
            if sim:
                logger.info(
                    "Simulation: worst=%.1fms best=%.1fms "
                    "risk=%s",
                    sim["cluster_worst_case_latency_ms"],
                    sim["cluster_best_case_latency_ms"],
                    sim["highest_risk_node"],
                )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        alerter.stop()
        simulator.stop()
        analyzer.stop()
        updater.stop()
        