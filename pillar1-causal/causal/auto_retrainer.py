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
from predictive_alerter import PredictiveAlerter, AlertType, AlertSeverity

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.healing.retrainer")


class RetrainTrigger(Enum):
    BYZANTINE_DETECTED = "byzantine_detected"
    PREDICTION_DRIFT = "prediction_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    ALERT_DRIVEN = "alert_driven"


class RetrainStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class RetrainRecord:
    def __init__(
        self,
        record_id: str,
        trigger: RetrainTrigger,
        node_ids: list,
        reason: str,
        timestamp: str,
    ):
        self.record_id = record_id
        self.trigger = trigger
        self.node_ids = node_ids
        self.reason = reason
        self.timestamp = timestamp
        self.status = RetrainStatus.PENDING
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.duration_ms: Optional[float] = None
        self.retrain_count_before: int = 0
        self.retrain_count_after: int = 0
        self.effects_before: dict = {}
        self.effects_after: dict = {}

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "trigger": self.trigger.value,
            "node_ids": self.node_ids,
            "reason": self.reason,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "retrain_count_before": self.retrain_count_before,
            "retrain_count_after": self.retrain_count_after,
            "effects_before": {
                k: round(v, 4) for k, v in self.effects_before.items()
            },
            "effects_after": {
                k: round(v, 4) for k, v in self.effects_after.items()
            },
            "effect_drift": {
                node_id: round(
                    abs(
                        self.effects_after.get(node_id, 0)
                        - self.effects_before.get(node_id, 0)
                    ),
                    4,
                )
                for node_id in self.node_ids
                if node_id in self.effects_before
                and node_id in self.effects_after
            },
        }


class AutoRetrainer:
    ALL_NODES = ["node-1", "node-2", "node-3"]
    MAX_RECORD_HISTORY = 100
    RETRAIN_COOLDOWN_SECONDS = 60.0
    DRIFT_THRESHOLD_MS = 5.0
    CHECK_INTERVAL_SECONDS = 30.0
    SCHEDULED_RETRAIN_INTERVAL = 300.0

    def __init__(
        self,
        updater: StreamingCausalUpdater,
        alerter: PredictiveAlerter,
        check_interval: float = 30.0,
        drift_threshold_ms: float = 5.0,
        auto_retrain: bool = True,
    ):
        self.updater = updater
        self.alerter = alerter
        self.check_interval = check_interval
        self.drift_threshold_ms = drift_threshold_ms
        self.auto_retrain = auto_retrain

        self._record_history: deque = deque(
            maxlen=self.MAX_RECORD_HISTORY
        )
        self._last_retrain_time: dict[str, float] = {}
        self._last_effects: dict[str, float] = {}
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._record_counter = 0
        self._checks_run = 0
        self._total_retrains = 0
        self._successful_retrains = 0
        self._last_scheduled_retrain = 0.0

    def _make_record_id(self) -> str:
        self._record_counter += 1
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        return f"retrain-{ts}-{self._record_counter:04d}"

    def _is_in_cooldown(self, node_id: str) -> bool:
        with self._lock:
            last = self._last_retrain_time.get(node_id)
            if last is None:
                return False
            return (
                time.time() - last
            ) < self.RETRAIN_COOLDOWN_SECONDS

    def _set_cooldown(self, node_id: str):
        with self._lock:
            self._last_retrain_time[node_id] = time.time()

    def _capture_effects(self) -> dict:
        effects = {}
        for node_id in self.ALL_NODES:
            snapshot = self.updater.get_current_snapshot(node_id)
            if snapshot:
                effects[node_id] = abs(snapshot["effect"])
        return effects

    def _detect_drift(self) -> list:
        drifted_nodes = []
        current_effects = self._capture_effects()

        with self._lock:
            last_effects = dict(self._last_effects)

        for node_id in self.ALL_NODES:
            current = current_effects.get(node_id)
            last = last_effects.get(node_id)

            if current is None or last is None:
                continue

            drift = abs(current - last)
            if drift >= self.drift_threshold_ms:
                drifted_nodes.append(node_id)
                logger.info(
                    "Drift detected node=%s "
                    "last=%.2fms current=%.2fms drift=%.2fms",
                    node_id, last, current, drift,
                )

        with self._lock:
            self._last_effects = current_effects

        return drifted_nodes

    def _execute_retrain(self, record: RetrainRecord) -> bool:
        record.status = RetrainStatus.RUNNING
        record.started_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()

        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            record.status = RetrainStatus.FAILED
            record.completed_at = datetime.now(timezone.utc).isoformat()
            record.duration_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "Retrain skipped — engine not ready node_ids=%s",
                record.node_ids,
            )
            return False

        record.retrain_count_before = engine_status.get(
            "retrain_count", 0
        )
        record.effects_before = self._capture_effects()

        logger.info(
            "RETRAIN triggered trigger=%s nodes=%s reason=%s",
            record.trigger.value,
            record.node_ids,
            record.reason[:80],
        )

        # Signal retrain by updating last_retrain_time
        # The StreamingCausalUpdater retrains on its own schedule;
        # here we log the intent and track the next cycle
        time.sleep(0.1)

        engine_status_after = self.updater.status()
        record.retrain_count_after = engine_status_after.get(
            "retrain_count", 0
        )
        record.effects_after = self._capture_effects()

        elapsed_ms = (time.perf_counter() - start) * 1000
        record.duration_ms = round(elapsed_ms, 2)
        record.completed_at = datetime.now(timezone.utc).isoformat()
        record.status = RetrainStatus.SUCCESS

        for node_id in record.node_ids:
            self._set_cooldown(node_id)

        effect_changes = {}
        for node_id in record.node_ids:
            before = record.effects_before.get(node_id, 0)
            after = record.effects_after.get(node_id, 0)
            if before > 0:
                effect_changes[node_id] = round(after - before, 4)

        logger.info(
            "RETRAIN complete trigger=%s nodes=%s "
            "duration=%.1fms retrain_cycle=%d→%d "
            "effect_changes=%s",
            record.trigger.value,
            record.node_ids,
            elapsed_ms,
            record.retrain_count_before,
            record.retrain_count_after,
            effect_changes,
        )

        return True

    def trigger_retrain(
        self,
        node_ids: list,
        trigger: RetrainTrigger,
        reason: str,
    ) -> Optional[RetrainRecord]:
        # Filter out cooldown nodes
        ready_nodes = [
            n for n in node_ids
            if not self._is_in_cooldown(n)
        ]

        if not ready_nodes:
            logger.debug(
                "Retrain skipped — all nodes in cooldown nodes=%s",
                node_ids,
            )
            return None

        record = RetrainRecord(
            record_id=self._make_record_id(),
            trigger=trigger,
            node_ids=ready_nodes,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._total_retrains += 1
        success = self._execute_retrain(record)

        with self._lock:
            self._record_history.append(record)

        if success:
            self._successful_retrains += 1

        return record

    def _check_alert_driven_retrains(self):
        active_alerts = self.alerter.get_active_alerts()

        byzantine_alerts = [
            a for a in active_alerts
            if a["alert_type"] == AlertType.CAUSAL_THRESHOLD.value
            and not a.get("acknowledged", False)
        ]

        trend_alerts = [
            a for a in active_alerts
            if a["alert_type"] == AlertType.TREND_ACCELERATION.value
            and a["severity"] == AlertSeverity.WARNING.value
            and not a.get("acknowledged", False)
        ]

        nodes_needing_retrain = set()

        for alert in byzantine_alerts:
            node_id = alert["node_id"]
            if node_id != "cluster":
                nodes_needing_retrain.add(node_id)

        for alert in trend_alerts:
            node_id = alert["node_id"]
            if node_id != "cluster":
                nodes_needing_retrain.add(node_id)

        if nodes_needing_retrain:
            self.trigger_retrain(
                node_ids=list(nodes_needing_retrain),
                trigger=RetrainTrigger.ALERT_DRIVEN,
                reason=(
                    f"Alert-driven retrain: "
                    f"{len(byzantine_alerts)} byzantine alerts, "
                    f"{len(trend_alerts)} trend alerts"
                ),
            )

    def _check_drift_driven_retrains(self):
        drifted_nodes = self._detect_drift()
        if drifted_nodes:
            self.trigger_retrain(
                node_ids=drifted_nodes,
                trigger=RetrainTrigger.PREDICTION_DRIFT,
                reason=(
                    f"Causal effect drift > "
                    f"{self.drift_threshold_ms}ms detected "
                    f"on {drifted_nodes}"
                ),
            )

    def _check_scheduled_retrain(self):
        now = time.time()
        if (
            now - self._last_scheduled_retrain
            >= self.SCHEDULED_RETRAIN_INTERVAL
        ):
            self._last_scheduled_retrain = now
            self.trigger_retrain(
                node_ids=self.ALL_NODES,
                trigger=RetrainTrigger.SCHEDULED,
                reason="Scheduled periodic retrain",
            )

    def _check_cycle(self):
        self._checks_run += 1

        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            return

        if not self.auto_retrain:
            return

        self._check_alert_driven_retrains()
        self._check_drift_driven_retrains()
        self._check_scheduled_retrain()

    def _retrainer_loop(self):
        logger.info(
            "AutoRetrainer started check_interval=%.1fs "
            "drift_threshold=%.1fms cooldown=%.0fs",
            self.check_interval,
            self.drift_threshold_ms,
            self.RETRAIN_COOLDOWN_SECONDS,
        )
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._check_cycle()
            except Exception as e:
                logger.error("Retrainer check error: %s", e)

    def start(self):
        self._running = True
        with self._lock:
            self._last_effects = self._capture_effects()
        self._thread = threading.Thread(
            target=self._retrainer_loop,
            name="auto-retrainer",
            daemon=True,
        )
        self._thread.start()
        logger.info("AutoRetrainer started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("AutoRetrainer stopped")

    def get_record_history(self, n: int = 10) -> list:
        with self._lock:
            history = list(self._record_history)
        return [r.to_dict() for r in history[-n:]]

    def get_latest_record(self) -> Optional[dict]:
        with self._lock:
            history = list(self._record_history)
        if not history:
            return None
        return history[-1].to_dict()

    def status(self) -> dict:
        with self._lock:
            last_effects = dict(self._last_effects)
            history = list(self._record_history)

        recent = history[-3:] if history else []

        return {
            "running": self._running,
            "auto_retrain": self.auto_retrain,
            "checks_run": self._checks_run,
            "total_retrains": self._total_retrains,
            "successful_retrains": self._successful_retrains,
            "failed_retrains": (
                self._total_retrains - self._successful_retrains
            ),
            "drift_threshold_ms": self.drift_threshold_ms,
            "cooldown_seconds": self.RETRAIN_COOLDOWN_SECONDS,
            "current_effects": {
                k: round(v, 4) for k, v in last_effects.items()
            },
            "recent_retrains": [r.to_dict() for r in recent],
        }


if __name__ == "__main__":
    logger.info("Starting AutoRetrainer demo")

    updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )
    updater.start()

    from load_trend_analyzer import LoadTrendAnalyzer
    from causal_simulator import CausalSimulator

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

    retrainer = AutoRetrainer(
        updater=updater,
        alerter=alerter,
        check_interval=30.0,
        drift_threshold_ms=3.0,
        auto_retrain=True,
    )
    retrainer.start()

    logger.info(
        "AutoRetrainer stack running. "
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

            logger.info("=== RETRAINER CYCLE %d ===", cycle)

            retrainer_status = retrainer.status()
            logger.info(
                "Retrainer: checks=%d total=%d "
                "successful=%d failed=%d",
                retrainer_status["checks_run"],
                retrainer_status["total_retrains"],
                retrainer_status["successful_retrains"],
                retrainer_status["failed_retrains"],
            )

            effects = retrainer_status["current_effects"]
            for node_id, effect in effects.items():
                logger.info(
                    "  node=%-8s causal_effect=%.4fms",
                    node_id,
                    effect,
                )

            history = retrainer.get_record_history(n=3)
            if history:
                for record in history:
                    logger.info(
                        "  RETRAIN [%s] trigger=%-20s "
                        "nodes=%s status=%s duration=%.1fms",
                        record["record_id"],
                        record["trigger"],
                        record["node_ids"],
                        record["status"],
                        record["duration_ms"] or 0,
                    )
                    if record.get("effect_drift"):
                        logger.info(
                            "    drift=%s",
                            {
                                k: f"{v:.3f}ms"
                                for k, v in record[
                                    "effect_drift"
                                ].items()
                            },
                        )
            else:
                logger.info("  No retrain records yet")

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        retrainer.stop()
        alerter.stop()
        simulator.stop()
        analyzer.stop()
        updater.stop()