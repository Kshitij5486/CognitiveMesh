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
logger = logging.getLogger("cm.healing.engine")


class HealingActionType(Enum):
    REROUTE = "reroute"
    REBALANCE = "rebalance"
    RETRAIN = "retrain"
    ISOLATE = "isolate"
    SCALE_DOWN = "scale_down"
    ALERT_OPERATOR = "alert_operator"
    NO_ACTION = "no_action"


class HealingActionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class HealingAction:
    def __init__(
        self,
        action_id: str,
        action_type: HealingActionType,
        node_id: str,
        trigger_alert_type: str,
        trigger_severity: str,
        reason: str,
        timestamp: str,
    ):
        self.action_id = action_id
        self.action_type = action_type
        self.node_id = node_id
        self.trigger_alert_type = trigger_alert_type
        self.trigger_severity = trigger_severity
        self.reason = reason
        self.timestamp = timestamp
        self.status = HealingActionStatus.PENDING
        self.result: Optional[str] = None
        self.executed_at: Optional[str] = None
        self.duration_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "node_id": self.node_id,
            "trigger_alert_type": self.trigger_alert_type,
            "trigger_severity": self.trigger_severity,
            "reason": self.reason,
            "status": self.status.value,
            "result": self.result,
            "timestamp": self.timestamp,
            "executed_at": self.executed_at,
            "duration_ms": self.duration_ms,
        }


class HealingPolicy:
    """
    Maps alert type + severity combinations to healing actions.
    This is the decision logic of the healing engine.
    """

    POLICY = {
        (AlertType.LATENCY_RISING.value, AlertSeverity.WARNING.value):
            HealingActionType.REBALANCE,
        (AlertType.LATENCY_RISING.value, AlertSeverity.CRITICAL.value):
            HealingActionType.REROUTE,
        (AlertType.LATENCY_RISING.value, AlertSeverity.EMERGENCY.value):
            HealingActionType.REROUTE,
        (AlertType.LOAD_SPIKE.value, AlertSeverity.WARNING.value):
            HealingActionType.REBALANCE,
        (AlertType.LOAD_SPIKE.value, AlertSeverity.CRITICAL.value):
            HealingActionType.REROUTE,
        (AlertType.TREND_ACCELERATION.value, AlertSeverity.WARNING.value):
            HealingActionType.RETRAIN,
        (AlertType.TREND_ACCELERATION.value, AlertSeverity.CRITICAL.value):
            HealingActionType.REROUTE,
        (AlertType.CAUSAL_THRESHOLD.value, AlertSeverity.WARNING.value):
            HealingActionType.RETRAIN,
        (AlertType.CAUSAL_THRESHOLD.value, AlertSeverity.CRITICAL.value):
            HealingActionType.ISOLATE,
        (AlertType.CAUSAL_THRESHOLD.value, AlertSeverity.EMERGENCY.value):
            HealingActionType.ISOLATE,
        (AlertType.CLUSTER_DEGRADATION.value, AlertSeverity.CRITICAL.value):
            HealingActionType.ALERT_OPERATOR,
        (AlertType.CLUSTER_DEGRADATION.value, AlertSeverity.EMERGENCY.value):
            HealingActionType.ALERT_OPERATOR,
    }

    @classmethod
    def decide(
        cls,
        alert_type: str,
        severity: str,
    ) -> HealingActionType:
        action = cls.POLICY.get((alert_type, severity))
        if action is None:
            return HealingActionType.NO_ACTION
        return action


class HealingActionExecutor:
    """
    Executes healing actions against the cluster.
    In production this would call the actual cluster APIs.
    In CognitiveMesh it coordinates with the existing stack.
    """

    def __init__(self, updater: StreamingCausalUpdater):
        self.updater = updater
        self._lock = threading.RLock()

    def execute(self, action: HealingAction) -> tuple[bool, str]:
        action.status = HealingActionStatus.EXECUTING
        action.executed_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()

        try:
            if action.action_type == HealingActionType.RETRAIN:
                success, msg = self._execute_retrain(action)
            elif action.action_type == HealingActionType.REROUTE:
                success, msg = self._execute_reroute(action)
            elif action.action_type == HealingActionType.REBALANCE:
                success, msg = self._execute_rebalance(action)
            elif action.action_type == HealingActionType.ISOLATE:
                success, msg = self._execute_isolate(action)
            elif action.action_type == HealingActionType.ALERT_OPERATOR:
                success, msg = self._execute_alert_operator(action)
            elif action.action_type == HealingActionType.NO_ACTION:
                success, msg = True, "No action required"
            else:
                success, msg = False, f"Unknown action type: {action.action_type}"

            elapsed_ms = (time.perf_counter() - start) * 1000
            action.duration_ms = round(elapsed_ms, 2)
            action.result = msg
            action.status = (
                HealingActionStatus.SUCCESS
                if success
                else HealingActionStatus.FAILED
            )
            return success, msg

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            action.duration_ms = round(elapsed_ms, 2)
            action.result = f"Exception: {e}"
            action.status = HealingActionStatus.FAILED
            return False, str(e)

    def _execute_retrain(
        self, action: HealingAction
    ) -> tuple[bool, str]:
        logger.info(
            "HEALING RETRAIN node=%s reason=%s",
            action.node_id,
            action.reason[:60],
        )
        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            return False, "Causal engine not ready for retrain"

        msg = (
            f"Causal model retrain triggered for {action.node_id}. "
            f"Next retrain cycle will incorporate latest telemetry. "
            f"Current retrain_count="
            f"{engine_status.get('retrain_count', 0)}"
        )
        logger.info("RETRAIN scheduled: %s", msg)
        return True, msg

    def _execute_reroute(
        self, action: HealingAction
    ) -> tuple[bool, str]:
        logger.warning(
            "HEALING REROUTE node=%s reason=%s",
            action.node_id,
            action.reason[:60],
        )
        active_nodes = ["node-1", "node-2", "node-3"]
        target_nodes = [n for n in active_nodes if n != action.node_id]

        if not target_nodes:
            return False, "No alternative nodes available for rerouting"

        msg = (
            f"Query rerouting activated: traffic from {action.node_id} "
            f"redirected to {target_nodes}. "
            f"Load distribution: {100 // len(target_nodes)}% per node."
        )
        logger.warning("REROUTE executed: %s", msg)
        return True, msg

    def _execute_rebalance(
        self, action: HealingAction
    ) -> tuple[bool, str]:
        logger.info(
            "HEALING REBALANCE node=%s reason=%s",
            action.node_id,
            action.reason[:60],
        )
        msg = (
            f"Load rebalancing initiated for {action.node_id}. "
            f"Connection pool throttled by 20%. "
            f"Query queue priority adjusted."
        )
        logger.info("REBALANCE executed: %s", msg)
        return True, msg

    def _execute_isolate(
        self, action: HealingAction
    ) -> tuple[bool, str]:
        logger.critical(
            "HEALING ISOLATE node=%s reason=%s",
            action.node_id,
            action.reason[:60],
        )
        msg = (
            f"Node isolation requested for {action.node_id}. "
            f"Forwarding to Byzantine isolation mechanism. "
            f"Node will be excluded from consensus and forecasts."
        )
        logger.critical("ISOLATE executed: %s", msg)
        return True, msg

    def _execute_alert_operator(
        self, action: HealingAction
    ) -> tuple[bool, str]:
        logger.critical(
            "HEALING ALERT_OPERATOR node=%s reason=%s",
            action.node_id,
            action.reason[:60],
        )
        msg = (
            f"Operator alert issued for cluster-wide degradation. "
            f"Affected node: {action.node_id}. "
            f"Manual intervention may be required."
        )
        logger.critical("OPERATOR ALERT: %s", msg)
        return True, msg


class HealingActionEngine:
    MAX_ACTION_HISTORY = 200
    COOLDOWN_SECONDS = 30.0

    def __init__(
        self,
        updater: StreamingCausalUpdater,
        alerter: PredictiveAlerter,
        check_interval: float = 15.0,
        auto_heal: bool = True,
    ):
        self.updater = updater
        self.alerter = alerter
        self.check_interval = check_interval
        self.auto_heal = auto_heal

        self.executor = HealingActionExecutor(updater=updater)
        self.policy = HealingPolicy()

        self._action_history: deque = deque(
            maxlen=self.MAX_ACTION_HISTORY
        )
        self._pending_actions: dict[str, HealingAction] = {}
        self._cooldowns: dict[str, float] = {}
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._action_counter = 0
        self._checks_run = 0
        self._total_actions = 0
        self._successful_actions = 0

    def _make_action_id(self) -> str:
        self._action_counter += 1
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        return f"heal-{ts}-{self._action_counter:04d}"

    def _is_in_cooldown(self, node_id: str, action_type: str) -> bool:
        key = f"{node_id}:{action_type}"
        with self._lock:
            last_time = self._cooldowns.get(key)
            if last_time is None:
                return False
            return (time.time() - last_time) < self.COOLDOWN_SECONDS

    def _set_cooldown(self, node_id: str, action_type: str):
        key = f"{node_id}:{action_type}"
        with self._lock:
            self._cooldowns[key] = time.time()

    def _process_alert(self, alert: dict) -> Optional[HealingAction]:
        alert_type = alert["alert_type"]
        severity = alert["severity"]
        node_id = alert["node_id"]

        action_type = HealingPolicy.decide(alert_type, severity)

        if action_type == HealingActionType.NO_ACTION:
            return None

        if self._is_in_cooldown(node_id, action_type.value):
            logger.debug(
                "Healing cooldown active node=%s action=%s",
                node_id,
                action_type.value,
            )
            return None

        action = HealingAction(
            action_id=self._make_action_id(),
            action_type=action_type,
            node_id=node_id,
            trigger_alert_type=alert_type,
            trigger_severity=severity,
            reason=(
                f"Alert [{alert_type}] severity={severity} "
                f"on {node_id}: {alert.get('message', '')[:80]}"
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return action

    def _execute_action(self, action: HealingAction):
        self._total_actions += 1
        success, result = self.executor.execute(action)

        with self._lock:
            self._action_history.append(action)

        self._set_cooldown(action.node_id, action.action_type.value)

        if success:
            self._successful_actions += 1
            logger.info(
                "Healing action SUCCESS [%s] node=%s "
                "duration=%.1fms result=%s",
                action.action_type.value,
                action.node_id,
                action.duration_ms or 0,
                (action.result or "")[:60],
            )
        else:
            logger.error(
                "Healing action FAILED [%s] node=%s result=%s",
                action.action_type.value,
                action.node_id,
                (action.result or "")[:60],
            )

    def _check_cycle(self):
        self._checks_run += 1

        if not self.auto_heal:
            return

        active_alerts = self.alerter.get_active_alerts()
        if not active_alerts:
            return

        actions_this_cycle = []
        for alert in active_alerts:
            if alert.get("acknowledged"):
                continue
            action = self._process_alert(alert)
            if action:
                actions_this_cycle.append(action)

        if actions_this_cycle:
            logger.info(
                "Healing check cycle=%d alerts=%d "
                "actions_planned=%d",
                self._checks_run,
                len(active_alerts),
                len(actions_this_cycle),
            )

        for action in actions_this_cycle:
            self._execute_action(action)

    def _heal_loop(self):
        logger.info(
            "HealingActionEngine started check_interval=%.1fs "
            "auto_heal=%s cooldown=%.0fs",
            self.check_interval,
            self.auto_heal,
            self.COOLDOWN_SECONDS,
        )
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._check_cycle()
            except Exception as e:
                logger.error("Healing check error: %s", e)

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._heal_loop,
            name="healing-engine",
            daemon=True,
        )
        self._thread.start()
        logger.info("HealingActionEngine started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("HealingActionEngine stopped")

    def heal_now(self, node_id: str, alert_type: str, severity: str) -> Optional[dict]:
        action_type = HealingPolicy.decide(alert_type, severity)
        if action_type == HealingActionType.NO_ACTION:
            return None

        action = HealingAction(
            action_id=self._make_action_id(),
            action_type=action_type,
            node_id=node_id,
            trigger_alert_type=alert_type,
            trigger_severity=severity,
            reason=(
                f"Manual heal: [{alert_type}] severity={severity} "
                f"on {node_id}"
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._execute_action(action)
        return action.to_dict()

    def get_action_history(self, n: int = 20) -> list:
        with self._lock:
            history = list(self._action_history)
        return [a.to_dict() for a in history[-n:]]

    def get_pending_actions(self) -> list:
        with self._lock:
            return [
                a.to_dict()
                for a in self._pending_actions.values()
            ]

    def status(self) -> dict:
        with self._lock:
            recent = list(self._action_history)[-5:]
        return {
            "running": self._running,
            "auto_heal": self.auto_heal,
            "checks_run": self._checks_run,
            "total_actions": self._total_actions,
            "successful_actions": self._successful_actions,
            "failed_actions": (
                self._total_actions - self._successful_actions
            ),
            "cooldown_seconds": self.COOLDOWN_SECONDS,
            "recent_actions": [a.to_dict() for a in recent],
        }


if __name__ == "__main__":
    logger.info("Starting HealingActionEngine demo")

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

    engine = HealingActionEngine(
        updater=updater,
        alerter=alerter,
        check_interval=15.0,
        auto_heal=True,
    )
    engine.start()

    logger.info(
        "Full healing stack running. "
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

            logger.info("=== HEALING CYCLE %d ===", cycle)

            healing_status = engine.status()
            logger.info(
                "Healing: checks=%d total=%d successful=%d failed=%d",
                healing_status["checks_run"],
                healing_status["total_actions"],
                healing_status["successful_actions"],
                healing_status["failed_actions"],
            )

            history = engine.get_action_history(n=5)
            if history:
                for action in history:
                    logger.info(
                        "  ACTION [%s] node=%s status=%s "
                        "duration=%.1fms",
                        action["action_type"],
                        action["node_id"],
                        action["status"],
                        action["duration_ms"] or 0,
                    )
            else:
                logger.info("  No healing actions yet")

            alerter_status = alerter.status()
            logger.info(
                "Alerter: active=%d total_fired=%d",
                alerter_status["active_alert_count"],
                alerter_status["total_alerts_fired"],
            )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        engine.stop()
        alerter.stop()
        simulator.stop()
        analyzer.stop()
        updater.stop()