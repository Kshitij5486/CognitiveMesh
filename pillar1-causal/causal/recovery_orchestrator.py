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
from healing_action_engine import HealingActionEngine, HealingActionType
from query_router import QueryRouter, NodeRoutingState
from auto_retrainer import AutoRetrainer, RetrainTrigger

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.healing.orchestrator")


class RecoveryPhase(Enum):
    DETECTED = "detected"
    ALERTING = "alerting"
    REROUTING = "rerouting"
    RETRAINING = "retraining"
    VERIFYING = "verifying"
    RESTORED = "restored"
    FAILED = "failed"
    ABORTED = "aborted"


class RecoveryTrigger(Enum):
    LATENCY_CRITICAL = "latency_critical"
    BYZANTINE_DETECTED = "byzantine_detected"
    CLUSTER_DEGRADATION = "cluster_degradation"
    MANUAL = "manual"


class RecoverySequence:
    def __init__(
        self,
        sequence_id: str,
        node_id: str,
        trigger: RecoveryTrigger,
        trigger_alert: dict,
        timestamp: str,
    ):
        self.sequence_id = sequence_id
        self.node_id = node_id
        self.trigger = trigger
        self.trigger_alert = trigger_alert
        self.timestamp = timestamp
        self.phase = RecoveryPhase.DETECTED
        self.phase_history: list = []
        self.started_at = timestamp
        self.completed_at: Optional[str] = None
        self.duration_seconds: Optional[float] = None
        self.actions_taken: list = []
        self.verification_result: Optional[bool] = None
        self.failure_reason: Optional[str] = None

    def advance_phase(self, new_phase: RecoveryPhase, note: str = ""):
        old_phase = self.phase
        self.phase = new_phase
        self.phase_history.append({
            "from": old_phase.value,
            "to": new_phase.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": note,
        })
        logger.info(
            "Recovery sequence=%s node=%s phase %s → %s %s",
            self.sequence_id,
            self.node_id,
            old_phase.value.upper(),
            new_phase.value.upper(),
            f"({note})" if note else "",
        )

    def record_action(self, action_type: str, result: str, success: bool):
        self.actions_taken.append({
            "action_type": action_type,
            "result": result,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def complete(self, success: bool, reason: str = ""):
        self.completed_at = datetime.now(timezone.utc).isoformat()
        start_ts = datetime.fromisoformat(self.started_at)
        end_ts = datetime.fromisoformat(self.completed_at)
        self.duration_seconds = round(
            (end_ts - start_ts).total_seconds(), 2
        )
        if success:
            self.advance_phase(RecoveryPhase.RESTORED, reason)
        else:
            self.advance_phase(RecoveryPhase.FAILED, reason)
            self.failure_reason = reason

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "node_id": self.node_id,
            "trigger": self.trigger.value,
            "phase": self.phase.value,
            "timestamp": self.timestamp,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "phase_history": self.phase_history,
            "actions_taken": self.actions_taken,
            "verification_result": self.verification_result,
            "failure_reason": self.failure_reason,
        }


class RecoveryOrchestrator:
    ALL_NODES = ["node-1", "node-2", "node-3"]
    MAX_SEQUENCE_HISTORY = 50
    CHECK_INTERVAL_SECONDS = 15.0
    VERIFY_LATENCY_THRESHOLD_MS = 120.0
    VERIFY_WAIT_SECONDS = 15.0
    MAX_RECOVERY_DURATION_SECONDS = 300.0

    def __init__(
        self,
        updater: StreamingCausalUpdater,
        alerter: PredictiveAlerter,
        engine: HealingActionEngine,
        router: QueryRouter,
        retrainer: AutoRetrainer,
        check_interval: float = 15.0,
        auto_recover: bool = True,
    ):
        self.updater = updater
        self.alerter = alerter
        self.engine = engine
        self.router = router
        self.retrainer = retrainer
        self.check_interval = check_interval
        self.auto_recover = auto_recover

        self._active_sequences: dict[str, RecoverySequence] = {}
        self._sequence_history: deque = deque(
            maxlen=self.MAX_SEQUENCE_HISTORY
        )
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sequence_counter = 0
        self._checks_run = 0
        self._total_sequences = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0

    def _make_sequence_id(self) -> str:
        self._sequence_counter += 1
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        return f"recovery-{ts}-{self._sequence_counter:04d}"

    def _get_current_latency(self, node_id: str) -> float:
        snapshot = self.updater.get_current_snapshot(node_id)
        if not snapshot:
            return 0.0
        buf = self.updater.buffer
        if not buf:
            return 0.0
        df = buf.get_dataframe()
        if df is None:
            return 0.0
        col = f"{node_id.replace('-', '_')}_active_queries"
        if col not in df.columns:
            return 0.0
        load = float(df[col].iloc[-1])
        return abs(snapshot["effect"]) * load

    def _execute_recovery_sequence(
        self, sequence: RecoverySequence
    ):
        node_id = sequence.node_id

        try:
            # Phase 1: ALERTING
            sequence.advance_phase(
                RecoveryPhase.ALERTING,
                "Acknowledging trigger alert"
            )
            active_alerts = self.alerter.get_active_alerts()
            for alert in active_alerts:
                if alert["node_id"] == node_id:
                    self.alerter.acknowledge_alert(alert["alert_id"])
            sequence.record_action(
                "acknowledge_alerts",
                f"Acknowledged alerts for {node_id}",
                True,
            )

            # Phase 2: REROUTING
            sequence.advance_phase(
                RecoveryPhase.REROUTING,
                f"Rerouting traffic from {node_id}"
            )
            decision = self.router.reroute_node(
                node_id=node_id,
                reason=(
                    f"Recovery sequence {sequence.sequence_id}: "
                    f"trigger={sequence.trigger.value}"
                ),
            )
            if decision:
                sequence.record_action(
                    "reroute",
                    f"Traffic rerouted to {decision.target_nodes} "
                    f"weights={decision.causal_weights}",
                    True,
                )
                logger.info(
                    "Recovery reroute: %s → %s",
                    node_id,
                    decision.target_nodes,
                )
            else:
                sequence.record_action(
                    "reroute",
                    "Reroute skipped (cooldown or no targets)",
                    False,
                )

            # Phase 3: RETRAINING
            sequence.advance_phase(
                RecoveryPhase.RETRAINING,
                f"Triggering causal model retrain for {node_id}"
            )
            retrain_record = self.retrainer.trigger_retrain(
                node_ids=[node_id],
                trigger=RetrainTrigger.BYZANTINE_DETECTED
                if sequence.trigger == RecoveryTrigger.BYZANTINE_DETECTED
                else RetrainTrigger.ALERT_DRIVEN,
                reason=(
                    f"Recovery sequence {sequence.sequence_id}"
                ),
            )
            if retrain_record:
                sequence.record_action(
                    "retrain",
                    f"Retrain {retrain_record.status.value} "
                    f"duration={retrain_record.duration_ms:.1f}ms",
                    retrain_record.status.value == "success",
                )
            else:
                sequence.record_action(
                    "retrain",
                    "Retrain skipped (cooldown)",
                    False,
                )

            # Phase 4: VERIFYING
            sequence.advance_phase(
                RecoveryPhase.VERIFYING,
                f"Waiting {self.VERIFY_WAIT_SECONDS}s then verifying"
            )
            time.sleep(self.VERIFY_WAIT_SECONDS)

            current_latency = self._get_current_latency(node_id)
            engine_status = self.updater.status()
            is_ready = engine_status.get("is_ready", False)

            verify_ok = (
                is_ready
                and current_latency < self.VERIFY_LATENCY_THRESHOLD_MS
            )
            sequence.verification_result = verify_ok

            sequence.record_action(
                "verify",
                f"latency={current_latency:.1f}ms "
                f"threshold={self.VERIFY_LATENCY_THRESHOLD_MS}ms "
                f"engine_ready={is_ready} "
                f"passed={verify_ok}",
                verify_ok,
            )

            if verify_ok:
                # Restore node to routing pool
                self.router.restore_node(node_id)
                sequence.complete(
                    True,
                    f"Node {node_id} verified healthy "
                    f"latency={current_latency:.1f}ms"
                )
                self._successful_recoveries += 1
                logger.info(
                    "Recovery COMPLETE sequence=%s node=%s "
                    "duration=%.1fs latency=%.1fms",
                    sequence.sequence_id,
                    node_id,
                    sequence.duration_seconds or 0,
                    current_latency,
                )
            else:
                sequence.complete(
                    False,
                    f"Verification failed: "
                    f"latency={current_latency:.1f}ms "
                    f"threshold={self.VERIFY_LATENCY_THRESHOLD_MS}ms"
                )
                self._failed_recoveries += 1
                logger.warning(
                    "Recovery FAILED sequence=%s node=%s "
                    "latency=%.1fms threshold=%.1fms",
                    sequence.sequence_id,
                    node_id,
                    current_latency,
                    self.VERIFY_LATENCY_THRESHOLD_MS,
                )

        except Exception as e:
            sequence.complete(False, f"Exception: {e}")
            self._failed_recoveries += 1
            logger.error(
                "Recovery exception sequence=%s node=%s error=%s",
                sequence.sequence_id,
                node_id,
                e,
            )

        finally:
            with self._lock:
                if sequence.sequence_id in self._active_sequences:
                    del self._active_sequences[sequence.sequence_id]
                self._sequence_history.append(sequence)

    def _start_recovery(
        self,
        node_id: str,
        trigger: RecoveryTrigger,
        trigger_alert: dict,
    ) -> Optional[RecoverySequence]:
        with self._lock:
            # Check if recovery already active for this node
            for seq in self._active_sequences.values():
                if seq.node_id == node_id:
                    logger.debug(
                        "Recovery already active for node=%s", node_id
                    )
                    return None

        sequence = RecoverySequence(
            sequence_id=self._make_sequence_id(),
            node_id=node_id,
            trigger=trigger,
            trigger_alert=trigger_alert,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._active_sequences[sequence.sequence_id] = sequence
            self._total_sequences += 1

        logger.warning(
            "Recovery sequence STARTED id=%s node=%s trigger=%s",
            sequence.sequence_id,
            node_id,
            trigger.value,
        )

        thread = threading.Thread(
            target=self._execute_recovery_sequence,
            args=(sequence,),
            name=f"recovery-{sequence.sequence_id}",
            daemon=True,
        )
        thread.start()
        return sequence

    def _check_cycle(self):
        self._checks_run += 1

        if not self.auto_recover:
            return

        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            return

        active_alerts = self.alerter.get_active_alerts()

        # Check for critical latency alerts
        for alert in active_alerts:
            if alert.get("acknowledged"):
                continue

            node_id = alert["node_id"]
            if node_id == "cluster":
                continue

            alert_type = alert["alert_type"]
            severity = alert["severity"]

            trigger = None

            if (
                alert_type == AlertType.CAUSAL_THRESHOLD.value
                and severity in (
                    AlertSeverity.CRITICAL.value,
                    AlertSeverity.EMERGENCY.value,
                )
            ):
                trigger = RecoveryTrigger.BYZANTINE_DETECTED

            elif (
                alert_type == AlertType.LATENCY_RISING.value
                and severity == AlertSeverity.EMERGENCY.value
            ):
                trigger = RecoveryTrigger.LATENCY_CRITICAL

            if trigger:
                self._start_recovery(
                    node_id=node_id,
                    trigger=trigger,
                    trigger_alert=alert,
                )

    def _orchestrator_loop(self):
        logger.info(
            "RecoveryOrchestrator started check_interval=%.1fs "
            "auto_recover=%s verify_threshold=%.0fms",
            self.check_interval,
            self.auto_recover,
            self.VERIFY_LATENCY_THRESHOLD_MS,
        )
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._check_cycle()
            except Exception as e:
                logger.error("Orchestrator check error: %s", e)

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._orchestrator_loop,
            name="recovery-orchestrator",
            daemon=True,
        )
        self._thread.start()
        logger.info("RecoveryOrchestrator started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("RecoveryOrchestrator stopped")

    def trigger_manual_recovery(
        self,
        node_id: str,
        reason: str = "Manual recovery",
    ) -> Optional[dict]:
        if node_id not in self.ALL_NODES:
            return None

        alert = {
            "alert_id": "manual",
            "node_id": node_id,
            "alert_type": "manual",
            "severity": "warning",
            "message": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "acknowledged": False,
        }
        sequence = self._start_recovery(
            node_id=node_id,
            trigger=RecoveryTrigger.MANUAL,
            trigger_alert=alert,
        )
        return sequence.to_dict() if sequence else None

    def get_active_sequences(self) -> list:
        with self._lock:
            return [
                s.to_dict()
                for s in self._active_sequences.values()
            ]

    def get_sequence_history(self, n: int = 10) -> list:
        with self._lock:
            history = list(self._sequence_history)
        return [s.to_dict() for s in history[-n:]]

    def status(self) -> dict:
        with self._lock:
            active = len(self._active_sequences)
            active_nodes = [
                s.node_id
                for s in self._active_sequences.values()
            ]

        return {
            "running": self._running,
            "auto_recover": self.auto_recover,
            "checks_run": self._checks_run,
            "total_sequences": self._total_sequences,
            "successful_recoveries": self._successful_recoveries,
            "failed_recoveries": self._failed_recoveries,
            "active_sequences": active,
            "active_nodes": active_nodes,
            "verify_threshold_ms": self.VERIFY_LATENCY_THRESHOLD_MS,
            "verify_wait_seconds": self.VERIFY_WAIT_SECONDS,
        }


if __name__ == "__main__":
    logger.info("Starting RecoveryOrchestrator demo")

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

    router = QueryRouter(
        updater=updater,
        alerter=alerter,
        check_interval=15.0,
    )
    router.start()

    retrainer = AutoRetrainer(
        updater=updater,
        alerter=alerter,
        check_interval=30.0,
        drift_threshold_ms=3.0,
        auto_retrain=True,
    )
    retrainer.start()

    orchestrator = RecoveryOrchestrator(
        updater=updater,
        alerter=alerter,
        engine=engine,
        router=router,
        retrainer=retrainer,
        check_interval=15.0,
        auto_recover=True,
    )
    orchestrator.start()

    logger.info(
        "Full self-healing stack running. "
        "Load generator in another terminal. "
        "Manual recovery available via orchestrator."
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

            logger.info("=== ORCHESTRATOR CYCLE %d ===", cycle)

            orch_status = orchestrator.status()
            logger.info(
                "Orchestrator: checks=%d total=%d "
                "successful=%d failed=%d active=%d",
                orch_status["checks_run"],
                orch_status["total_sequences"],
                orch_status["successful_recoveries"],
                orch_status["failed_recoveries"],
                orch_status["active_sequences"],
            )

            active = orchestrator.get_active_sequences()
            if active:
                for seq in active:
                    logger.info(
                        "  ACTIVE sequence=%s node=%s "
                        "phase=%s trigger=%s",
                        seq["sequence_id"],
                        seq["node_id"],
                        seq["phase"].upper(),
                        seq["trigger"],
                    )

            history = orchestrator.get_sequence_history(n=3)
            if history:
                for seq in history:
                    logger.info(
                        "  HISTORY sequence=%s node=%s "
                        "phase=%s duration=%.1fs actions=%d",
                        seq["sequence_id"],
                        seq["node_id"],
                        seq["phase"].upper(),
                        seq["duration_seconds"] or 0,
                        len(seq["actions_taken"]),
                    )
                    for action in seq["actions_taken"]:
                        logger.info(
                            "    [%s] success=%s: %s",
                            action["action_type"],
                            action["success"],
                            action["result"][:70],
                        )
            else:
                logger.info("  No recovery sequences yet")

            # Trigger a manual recovery on cycle 4
            if cycle == 4:
                logger.info(
                    "Triggering manual recovery on node-1"
                )
                result = orchestrator.trigger_manual_recovery(
                    node_id="node-1",
                    reason="Demo manual recovery cycle 4",
                )
                if result:
                    logger.info(
                        "Manual recovery started: %s",
                        result["sequence_id"],
                    )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        orchestrator.stop()
        retrainer.stop()
        router.stop()
        engine.stop()
        alerter.stop()
        simulator.stop()
        analyzer.stop()
        updater.stop()