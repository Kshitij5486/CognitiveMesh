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

from streaming_updater import StreamingCausalUpdater

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.causal.drift_detector")


class EffectHistory:
    def __init__(self, node_id: str, max_history: int = 50):
        self.node_id = node_id
        self.max_history = max_history
        self._history: deque = deque(maxlen=max_history)
        self._lock = threading.RLock()

    def append(self, effect: float, timestamp: str):
        with self._lock:
            self._history.append({
                "effect": effect,
                "timestamp": timestamp,
            })

    def get_recent(self, n: int = 5) -> list:
        with self._lock:
            return list(self._history)[-n:]

    def size(self) -> int:
        with self._lock:
            return len(self._history)

    def get_baseline(self, n: int = 5) -> Optional[float]:
        with self._lock:
            if len(self._history) < n + 1:
                return None
            baseline_window = list(self._history)[-(n + 1):-1]
            effects = [e["effect"] for e in baseline_window]
            return sum(effects) / len(effects)

    def get_all_effects(self) -> list:
        with self._lock:
            return [e["effect"] for e in self._history]


class DriftEvent:
    def __init__(
        self,
        node_id: str,
        previous_effect: float,
        current_effect: float,
        change_pct: float,
        severity: str,
        timestamp: str,
    ):
        self.node_id = node_id
        self.previous_effect = previous_effect
        self.current_effect = current_effect
        self.change_pct = change_pct
        self.severity = severity
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "previous_effect_ms": round(self.previous_effect, 4),
            "current_effect_ms": round(self.current_effect, 4),
            "change_pct": round(self.change_pct, 2),
            "severity": self.severity,
            "timestamp": self.timestamp,
            "description": (
                f"Causal effect on {self.node_id} changed by "
                f"{self.change_pct:+.1f}% "
                f"({self.previous_effect:.2f}ms -> {self.current_effect:.2f}ms). "
                f"Severity: {self.severity}."
            ),
        }


class CausalDriftDetector:
    DRIFT_THRESHOLD_WARN = 0.20
    DRIFT_THRESHOLD_CRITICAL = 0.50

    def __init__(
        self,
        updater: StreamingCausalUpdater,
        check_interval_seconds: float = 10.0,
        baseline_window: int = 3,
        min_history_for_detection: int = 2,
    ):
        self.updater = updater
        self.check_interval = check_interval_seconds
        self.baseline_window = baseline_window
        self.min_history = min_history_for_detection

        self._histories: dict[str, EffectHistory] = {
            "node-1": EffectHistory("node-1"),
            "node-2": EffectHistory("node-2"),
            "node-3": EffectHistory("node-3"),
        }
        self._drift_events: deque = deque(maxlen=100)
        self._events_lock = threading.RLock()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_retrain_count = 0
        self._checks_run = 0
        self._drifts_detected = 0

    def _check_drift(self, node_id: str) -> Optional[DriftEvent]:
        history = self._histories[node_id]

        if history.size() < self.min_history:
            return None

        snapshot = self.updater.get_current_snapshot(node_id)
        if not snapshot:
            return None

        current_effect = snapshot["effect"]
        baseline = history.get_baseline(n=self.baseline_window)

        if baseline is None:
            return None

        if abs(baseline) < 0.001:
            return None

        change_pct = (current_effect - baseline) / abs(baseline)

        if abs(change_pct) >= self.DRIFT_THRESHOLD_CRITICAL:
            severity = "CRITICAL"
        elif abs(change_pct) >= self.DRIFT_THRESHOLD_WARN:
            severity = "WARNING"
        else:
            return None

        return DriftEvent(
            node_id=node_id,
            previous_effect=baseline,
            current_effect=current_effect,
            change_pct=change_pct * 100,
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _record_current_effects(self):
        for node_id in ["node-1", "node-2", "node-3"]:
            snapshot = self.updater.get_current_snapshot(node_id)
            if snapshot:
                self._histories[node_id].append(
                    effect=snapshot["effect"],
                    timestamp=snapshot["timestamp"],
                )

    def _detection_loop(self):
        logger.info(
            "Drift detector started check_interval=%.1fs "
            "warn_threshold=%.0f%% critical_threshold=%.0f%%",
            self.check_interval,
            self.DRIFT_THRESHOLD_WARN * 100,
            self.DRIFT_THRESHOLD_CRITICAL * 100,
        )

        while self._running:
            time.sleep(self.check_interval)

            engine_status = self.updater.status()
            if not engine_status["is_ready"]:
                continue

            current_retrain = engine_status["retrain_count"]
            if current_retrain == self._last_retrain_count:
                continue

            self._last_retrain_count = current_retrain
            self._record_current_effects()
            self._checks_run += 1

            logger.info(
                "Drift check #%d retrain_cycle=%d",
                self._checks_run,
                current_retrain,
            )

            for node_id in ["node-1", "node-2", "node-3"]:
                snapshot = self.updater.get_current_snapshot(node_id)
                if snapshot:
                    logger.info(
                        "  node=%-8s effect=%.4fms history_size=%d",
                        node_id,
                        snapshot["effect"],
                        self._histories[node_id].size(),
                    )

            for node_id in ["node-1", "node-2", "node-3"]:
                event = self._check_drift(node_id)
                if event:
                    self._drifts_detected += 1
                    with self._events_lock:
                        self._drift_events.append(event)
                    if event.severity == "CRITICAL":
                        logger.critical(
                            "CAUSAL DRIFT DETECTED node=%s "
                            "change=%+.1f%% "
                            "prev=%.2fms current=%.2fms",
                            node_id,
                            event.change_pct,
                            event.previous_effect,
                            event.current_effect,
                        )
                    else:
                        logger.warning(
                            "Causal drift warning node=%s "
                            "change=%+.1f%% "
                            "prev=%.2fms current=%.2fms",
                            node_id,
                            event.change_pct,
                            event.previous_effect,
                            event.current_effect,
                        )
                else:
                    history = self._histories[node_id]
                    if history.size() >= self.min_history:
                        logger.info(
                            "  node=%-8s STABLE no significant drift",
                            node_id,
                        )

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._detection_loop,
            name="drift-detector",
            daemon=True,
        )
        self._thread.start()
        logger.info("CausalDriftDetector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("CausalDriftDetector stopped")

    def get_recent_events(self, n: int = 10) -> list:
        with self._events_lock:
            events = list(self._drift_events)[-n:]
        return [e.to_dict() for e in events]

    def get_effect_history(self, node_id: str) -> list:
        if node_id not in self._histories:
            return []
        return self._histories[node_id].get_all_effects()

    def status(self) -> dict:
        histories = {
            node_id: {
                "history_size": h.size(),
                "recent_effects": [
                    round(e, 4)
                    for e in h.get_all_effects()[-5:]
                ],
            }
            for node_id, h in self._histories.items()
        }
        return {
            "running": self._running,
            "checks_run": self._checks_run,
            "drifts_detected": self._drifts_detected,
            "warn_threshold_pct": self.DRIFT_THRESHOLD_WARN * 100,
            "critical_threshold_pct": self.DRIFT_THRESHOLD_CRITICAL * 100,
            "node_histories": histories,
            "recent_events": self.get_recent_events(5),
        }


if __name__ == "__main__":
    logger.info("Starting streaming updater + drift detector")

    updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )
    updater.start()

    detector = CausalDriftDetector(
        updater=updater,
        check_interval_seconds=10.0,
        baseline_window=3,
        min_history_for_detection=2,
    )
    detector.start()

    logger.info(
        "System running. Load generator should be running in another terminal. "
        "Watching for causal drift every 10s after each retrain."
    )

    try:
        while True:
            time.sleep(30)
            updater_status = updater.status()
            detector_status = detector.status()
            logger.info(
                "SYSTEM STATUS retrains=%d checks=%d drifts=%d buffer=%d",
                updater_status["retrain_count"],
                detector_status["checks_run"],
                detector_status["drifts_detected"],
                updater_status["buffer_size"],
            )
            recent_events = detector.get_recent_events(3)
            if recent_events:
                logger.info("Recent drift events:")
                for event in recent_events:
                    logger.info("  %s", event["description"])
            else:
                logger.info("No drift events detected — system stable")

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        detector.stop()
        updater.stop()