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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.byzantine.multi_orchestrator")


class RecoveryPhase(Enum):
    IDLE          = "idle"
    DETECTING     = "detecting"
    PLANNING      = "planning"
    EXECUTING     = "executing"
    VERIFYING     = "verifying"
    COMPLETED     = "completed"
    FAILED        = "failed"
    PARTIAL       = "partial"


class NodeRecoveryOutcome(Enum):
    SUCCESS       = "success"
    FAILED        = "failed"
    SKIPPED       = "skipped"
    QUORUM_DENIED = "quorum_denied"
    TIMEOUT       = "timeout"


class RecoverySession:
    """
    Tracks one full multi-node recovery incident
    from detection through verification.
    """

    def __init__(self, session_id: str, trigger: str):
        self.session_id = session_id
        self.trigger = trigger
        self.phase = RecoveryPhase.DETECTING
        self.affected_nodes: list = []
        self.node_outcomes: dict = {}
        self.node_durations: dict = {}
        self.node_effects_at_start: dict = {}
        self.node_effects_at_end: dict = {}
        self.started_at = datetime.now(
            timezone.utc
        ).isoformat()
        self.completed_at: Optional[str] = None
        self.total_duration_seconds: Optional[float] = None
        self.mttr_seconds: Optional[float] = None
        self.quorum_state_at_start: str = "unknown"
        self.quorum_state_at_end: str = "unknown"
        self.plan_ids: list = []
        self.notes: list = []

    @property
    def is_complete(self) -> bool:
        return self.phase in (
            RecoveryPhase.COMPLETED,
            RecoveryPhase.FAILED,
            RecoveryPhase.PARTIAL,
        )

    @property
    def success_count(self) -> int:
        return sum(
            1 for v in self.node_outcomes.values()
            if v == NodeRecoveryOutcome.SUCCESS
        )

    @property
    def failure_count(self) -> int:
        return sum(
            1 for v in self.node_outcomes.values()
            if v in (
                NodeRecoveryOutcome.FAILED,
                NodeRecoveryOutcome.TIMEOUT,
            )
        )

    def complete(self, phase: RecoveryPhase):
        self.phase = phase
        self.completed_at = datetime.now(
            timezone.utc
        ).isoformat()
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            self.total_duration_seconds = (
                end - start
            ).total_seconds()

        # MTTR = mean duration of successful recoveries
        successful_durations = [
            d for node, d in self.node_durations.items()
            if self.node_outcomes.get(node)
            == NodeRecoveryOutcome.SUCCESS
        ]
        if successful_durations:
            self.mttr_seconds = sum(
                successful_durations
            ) / len(successful_durations)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "trigger": self.trigger,
            "phase": self.phase.value,
            "affected_nodes": self.affected_nodes,
            "node_outcomes": {
                k: v.value
                for k, v in self.node_outcomes.items()
            },
            "node_durations_seconds": {
                k: round(v, 2)
                for k, v in self.node_durations.items()
            },
            "node_effects_at_start": {
                k: round(v, 4)
                for k, v in self.node_effects_at_start.items()
            },
            "node_effects_at_end": {
                k: round(v, 4)
                for k, v in self.node_effects_at_end.items()
            },
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_seconds": round(
                self.total_duration_seconds, 2
            ) if self.total_duration_seconds else None,
            "mttr_seconds": round(
                self.mttr_seconds, 2
            ) if self.mttr_seconds else None,
            "quorum_state_at_start": self.quorum_state_at_start,
            "quorum_state_at_end": self.quorum_state_at_end,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "plan_ids": self.plan_ids,
            "notes": self.notes,
        }


class MultiNodeRecoveryOrchestrator:
    """
    Top-level orchestrator for multi-node Byzantine recovery.

    Responsibilities:
    1. Poll ByzantineRecoveryCoordinator for affected nodes
    2. Open a RecoverySession when >= 1 node needs recovery
    3. Sequence recovery: worst node first, one at a time
    4. Track per-node MTTR, effect before/after, outcomes
    5. Handle quorum-denied and timeout scenarios gracefully
    6. Emit structured recovery reports
    7. Enforce minimum inter-session cooldown (60s)
    8. Maintain session history for Sprint 9 benchmarks

    Architecture position:
      QuorumManager
          ↓
      ByzantineRecoveryCoordinator  (detection + single-node exec)
          ↓
      MultiNodeRecoveryOrchestrator  (session mgmt + sequencing)
    """

    ALL_NODES = ["node-1", "node-2", "node-3"]
    SESSION_COOLDOWN_SECONDS   = 60.0
    NODE_RECOVERY_TIMEOUT_S    = 150.0
    VERIFY_STABILISE_SECONDS   = 25.0
    MAX_SESSIONS_HISTORY       = 50
    MAX_RETRIES_PER_NODE       = 2
    CHECK_INTERVAL_SECONDS     = 15.0

    def __init__(
        self,
        updater,
        quorum_manager,
        coordinator,
        check_interval: float = 15.0,
    ):
        self.updater = updater
        self.quorum_manager = quorum_manager
        self.coordinator = coordinator
        self.check_interval = check_interval

        self._active_session: Optional[RecoverySession] = None
        self._session_history: deque = deque(
            maxlen=self.MAX_SESSIONS_HISTORY
        )
        self._session_counter = 0
        self._last_session_end: Optional[float] = None

        # Counters
        self._check_count = 0
        self._sessions_total = 0
        self._sessions_completed = 0
        self._sessions_partial = 0
        self._sessions_failed = 0
        self._nodes_recovered_total = 0
        self._nodes_failed_total = 0

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

    # ── Session lifecycle ──────────────────────────────────

    def _open_session(
        self,
        affected_nodes: list,
        effects: dict,
        trigger: str,
    ) -> RecoverySession:
        self._session_counter += 1
        session_id = f"session-{self._session_counter:04d}"
        session = RecoverySession(
            session_id=session_id,
            trigger=trigger,
        )
        session.affected_nodes = list(affected_nodes)
        session.node_effects_at_start = {
            n: effects.get(n, 0.0) for n in affected_nodes
        }
        session.quorum_state_at_start = (
            self.quorum_manager.get_quorum_state().value
        )
        self._sessions_total += 1

        logger.info(
            "Recovery session OPENED: id=%s trigger=%s "
            "affected=%s effects=%s quorum=%s",
            session_id,
            trigger,
            affected_nodes,
            {k: f"{v:.2f}ms" for k, v in
             session.node_effects_at_start.items()},
            session.quorum_state_at_start,
        )
        return session

    def _close_session(self, session: RecoverySession):
        # Capture final effects
        for node_id in session.affected_nodes:
            snap = self.updater.get_current_snapshot(
                node_id
            )
            if snap:
                session.node_effects_at_end[node_id] = abs(
                    snap["effect"]
                )

        session.quorum_state_at_end = (
            self.quorum_manager.get_quorum_state().value
        )

        # Determine final phase
        if session.success_count == len(
            session.affected_nodes
        ):
            final_phase = RecoveryPhase.COMPLETED
            self._sessions_completed += 1
        elif session.success_count > 0:
            final_phase = RecoveryPhase.PARTIAL
            self._sessions_partial += 1
        else:
            final_phase = RecoveryPhase.FAILED
            self._sessions_failed += 1

        session.complete(final_phase)

        with self._lock:
            self._session_history.append(session)
            self._active_session = None
            self._last_session_end = time.time()

        logger.info(
            "Recovery session CLOSED: id=%s phase=%s "
            "success=%d/%d duration=%.1fs mttr=%s "
            "quorum_end=%s",
            session.session_id,
            final_phase.value.upper(),
            session.success_count,
            len(session.affected_nodes),
            session.total_duration_seconds or 0.0,
            f"{session.mttr_seconds:.1f}s"
            if session.mttr_seconds else "N/A",
            session.quorum_state_at_end,
        )

        self._emit_recovery_report(session)

    def _emit_recovery_report(self, session: RecoverySession):
        d = session.to_dict()
        logger.info(
            "=== RECOVERY REPORT ===\n"
            "  Session:     %s\n"
            "  Trigger:     %s\n"
            "  Phase:       %s\n"
            "  Nodes:       %s\n"
            "  Outcomes:    %s\n"
            "  Durations:   %s\n"
            "  Effects before: %s\n"
            "  Effects after:  %s\n"
            "  Total time:  %s\n"
            "  MTTR:        %s\n"
            "  Quorum:      %s → %s",
            d["session_id"],
            d["trigger"],
            d["phase"].upper(),
            d["affected_nodes"],
            d["node_outcomes"],
            {k: f"{v:.1f}s" for k, v in
             d["node_durations_seconds"].items()},
            {k: f"{v:.2f}ms" for k, v in
             d["node_effects_at_start"].items()},
            {k: f"{v:.2f}ms" for k, v in
             d["node_effects_at_end"].items()},
            f"{d['total_duration_seconds']:.1f}s"
            if d["total_duration_seconds"] else "N/A",
            f"{d['mttr_seconds']:.1f}s"
            if d["mttr_seconds"] else "N/A",
            d["quorum_state_at_start"],
            d["quorum_state_at_end"],
        )

    # ── Node recovery execution ────────────────────────────

    def _recover_single_node(
        self,
        node_id: str,
        session: RecoverySession,
        effects: dict,
    ) -> NodeRecoveryOutcome:
        start_time = time.time()
        session.phase = RecoveryPhase.EXECUTING
        logger.info(
            "Session %s: recovering node=%s "
            "effect=%.2fms",
            session.session_id,
            node_id,
            effects.get(node_id, 0.0),
        )

        # Check quorum permission
        from quorum_manager import QuorumDecision
        decision = self.quorum_manager.request_node_offline(
            node_id=node_id,
            reason=f"multi_node_orchestrator "
                   f"session={session.session_id}",
        )

        if decision == QuorumDecision.DENY_QUORUM_RISK:
            logger.warning(
                "Session %s: QUORUM_DENIED for node=%s",
                session.session_id, node_id,
            )
            session.notes.append(
                f"{node_id}: quorum_denied"
            )
            return NodeRecoveryOutcome.QUORUM_DENIED

        if decision == QuorumDecision.DENY_CONCURRENT:
            logger.warning(
                "Session %s: CONCURRENT_DENIED "
                "for node=%s",
                session.session_id, node_id,
            )
            session.notes.append(
                f"{node_id}: deny_concurrent"
            )
            return NodeRecoveryOutcome.QUORUM_DENIED

        # Mark recovering in quorum
        self.quorum_manager.mark_node_recovering(node_id)

        try:
            # Attempt recovery with retry
            success = False
            for attempt in range(1, self.MAX_RETRIES_PER_NODE + 1):
                logger.info(
                    "Session %s: node=%s attempt=%d/%d",
                    session.session_id,
                    node_id,
                    attempt,
                    self.MAX_RETRIES_PER_NODE,
                )

                # Wait for stabilisation
                logger.info(
                    "Waiting %.1fs for stabilisation "
                    "node=%s attempt=%d",
                    self.VERIFY_STABILISE_SECONDS,
                    node_id, attempt,
                )
                time.sleep(self.VERIFY_STABILISE_SECONDS)

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.NODE_RECOVERY_TIMEOUT_S:
                    logger.warning(
                        "Session %s: TIMEOUT node=%s "
                        "elapsed=%.1fs",
                        session.session_id,
                        node_id, elapsed,
                    )
                    self.quorum_manager.mark_node_reduced(
                        node_id
                    )
                    duration = time.time() - start_time
                    session.node_durations[node_id] = duration
                    return NodeRecoveryOutcome.TIMEOUT

                # Verify
                session.phase = RecoveryPhase.VERIFYING
                verified = self._verify_recovery(
                    node_id, effects
                )

                if verified:
                    success = True
                    break
                else:
                    logger.info(
                        "Session %s: node=%s attempt=%d "
                        "verify FAIL — retrying",
                        session.session_id,
                        node_id, attempt,
                    )

            duration = time.time() - start_time
            session.node_durations[node_id] = duration

            if success:
                self.quorum_manager.mark_node_full(node_id)
                self._nodes_recovered_total += 1
                logger.info(
                    "Session %s: node=%s RECOVERED "
                    "duration=%.1fs",
                    session.session_id,
                    node_id, duration,
                )
                return NodeRecoveryOutcome.SUCCESS
            else:
                self.quorum_manager.mark_node_reduced(
                    node_id
                )
                self._nodes_failed_total += 1
                logger.warning(
                    "Session %s: node=%s FAILED after "
                    "%d attempts duration=%.1fs",
                    session.session_id,
                    node_id,
                    self.MAX_RETRIES_PER_NODE,
                    duration,
                )
                return NodeRecoveryOutcome.FAILED

        except Exception as e:
            duration = time.time() - start_time
            session.node_durations[node_id] = duration
            logger.error(
                "Session %s: exception during "
                "node=%s recovery: %s",
                session.session_id, node_id, e,
            )
            self.quorum_manager.mark_node_reduced(node_id)
            self._nodes_failed_total += 1
            return NodeRecoveryOutcome.FAILED

    def _verify_recovery(
        self, node_id: str, original_effects: dict
    ) -> bool:
        snap = self.updater.get_current_snapshot(node_id)
        if not snap:
            return False

        effect = abs(snap["effect"])
        others = []
        for n in self.ALL_NODES:
            if n == node_id:
                continue
            s = self.updater.get_current_snapshot(n)
            if s:
                others.append(abs(s["effect"]))

        if not others:
            return True

        cluster_mean = sum(others) / len(others)
        ratio = (
            effect / cluster_mean
            if cluster_mean > 0 else 1.0
        )
        verified = ratio <= 1.5

        logger.info(
            "Verify node=%s effect=%.2fms "
            "cluster_mean=%.2fms ratio=%.2f → %s",
            node_id, effect, cluster_mean, ratio,
            "PASS" if verified else "FAIL",
        )
        return verified

    # ── Session execution ──────────────────────────────────

    def _build_priority_order(
        self,
        affected_nodes: list,
        effects: dict,
    ) -> list:
        """Sort nodes: highest effect first (worst first)."""
        return sorted(
            affected_nodes,
            key=lambda n: effects.get(n, 0.0),
            reverse=True,
        )

    def _run_session(
        self,
        session: RecoverySession,
        effects: dict,
        priority_order: list,
    ):
        session.phase = RecoveryPhase.PLANNING
        logger.info(
            "Session %s PLANNING: priority_order=%s",
            session.session_id, priority_order,
        )

        for node_id in priority_order:
            coord_state = self.coordinator.get_node_state(
                node_id
            )
            if coord_state["byzantine_state"] == "healthy":
                logger.info(
                    "Session %s: node=%s already healthy, "
                    "skipping",
                    session.session_id, node_id,
                )
                session.node_outcomes[node_id] = (
                    NodeRecoveryOutcome.SKIPPED
                )
                session.node_durations[node_id] = 0.0
                continue

            outcome = self._recover_single_node(
                node_id, session, effects
            )
            session.node_outcomes[node_id] = outcome

            # Brief inter-node pause
            if node_id != priority_order[-1]:
                remaining = [
                    n for n in priority_order
                    if n not in session.node_outcomes
                ]
                if remaining:
                    logger.info(
                        "Session %s: 5s pause before "
                        "next node. Remaining: %s",
                        session.session_id, remaining,
                    )
                    time.sleep(5)

        self._close_session(session)

    # ── Check cycle ────────────────────────────────────────

    def _check_cycle(self):
        self._check_count += 1

        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            return

        # Skip if session active
        with self._lock:
            if self._active_session is not None:
                return

        # Enforce cooldown
        with self._lock:
            last_end = self._last_session_end

        if last_end is not None:
            elapsed = time.time() - last_end
            if elapsed < self.SESSION_COOLDOWN_SECONDS:
                return

        # Collect current effects
        effects = {}
        for node_id in self.ALL_NODES:
            snap = self.updater.get_current_snapshot(
                node_id
            )
            if snap:
                effects[node_id] = abs(snap["effect"])

        # Check coordinator for affected nodes
        affected = []
        coord_states = self.coordinator.get_all_node_states()
        for node_id, state_info in coord_states.items():
            byz_state = state_info["state"]
            if byz_state in ("suspected", "confirmed"):
                affected.append(node_id)

        if not affected:
            return

        # Open and start session
        session = self._open_session(
            affected_nodes=affected,
            effects=effects,
            trigger="coordinator_detection",
        )
        priority_order = self._build_priority_order(
            affected, effects
        )

        with self._lock:
            self._active_session = session

        session_thread = threading.Thread(
            target=self._run_session,
            args=(session, effects, priority_order),
            name=f"session-{session.session_id}",
            daemon=True,
        )
        session_thread.start()

    def _orchestrator_loop(self):
        logger.info(
            "MultiNodeRecoveryOrchestrator loop started "
            "interval=%.1fs",
            self.check_interval,
        )
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._check_cycle()
            except Exception as e:
                logger.error(
                    "Orchestrator check error: %s", e
                )

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._orchestrator_loop,
            name="multi-node-orchestrator",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "MultiNodeRecoveryOrchestrator started"
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=20)
        logger.info(
            "MultiNodeRecoveryOrchestrator stopped"
        )

    # ── Public API ─────────────────────────────────────────

    def trigger_session_manual(
        self,
        node_ids: list,
        reason: str = "manual",
    ) -> Optional[str]:
        effects = {}
        for node_id in node_ids:
            snap = self.updater.get_current_snapshot(
                node_id
            )
            if snap:
                effects[node_id] = abs(snap["effect"])

        with self._lock:
            if self._active_session is not None:
                logger.warning(
                    "Manual trigger rejected: "
                    "session already active=%s",
                    self._active_session.session_id,
                )
                return None

        session = self._open_session(
            affected_nodes=node_ids,
            effects=effects,
            trigger=reason,
        )
        priority_order = self._build_priority_order(
            node_ids, effects
        )

        with self._lock:
            self._active_session = session

        session_thread = threading.Thread(
            target=self._run_session,
            args=(session, effects, priority_order),
            name=f"manual-{session.session_id}",
            daemon=True,
        )
        session_thread.start()
        return session.session_id

    def get_active_session(self) -> Optional[dict]:
        with self._lock:
            if self._active_session:
                return self._active_session.to_dict()
        return None

    def get_session_history(self, n: int = 10) -> list:
        with self._lock:
            history = list(self._session_history)
        return [s.to_dict() for s in history[-n:]]

    def get_latest_report(self) -> Optional[dict]:
        with self._lock:
            history = list(self._session_history)
        if history:
            return history[-1].to_dict()
        return None

    def status(self) -> dict:
        with self._lock:
            active = (
                self._active_session.to_dict()
                if self._active_session else None
            )

        success_rate = (
            self._sessions_completed
            / self._sessions_total
            if self._sessions_total > 0 else 1.0
        )

        return {
            "running": self._running,
            "check_count": self._check_count,
            "sessions_total": self._sessions_total,
            "sessions_completed": self._sessions_completed,
            "sessions_partial": self._sessions_partial,
            "sessions_failed": self._sessions_failed,
            "session_success_rate": round(
                success_rate, 4
            ),
            "nodes_recovered_total": (
                self._nodes_recovered_total
            ),
            "nodes_failed_total": (
                self._nodes_failed_total
            ),
            "active_session": active,
            "sessions_in_history": len(
                list(self._session_history)
            ),
            "cooldown_seconds": (
                self.SESSION_COOLDOWN_SECONDS
            ),
            "uptime_seconds": round(
                time.time() - self._start_time, 1
            ) if self._start_time else 0.0,
        }


if __name__ == "__main__":
    logger.info(
        "Starting MultiNodeRecoveryOrchestrator demo"
    )

    from streaming_updater import StreamingCausalUpdater
    from query_router import QueryRouter, RoutingStrategy
    from auto_retrainer import AutoRetrainer
    from quorum_manager import QuorumManager
    from byzantine_recovery_coordinator import (
        ByzantineRecoveryCoordinator,
        ByzantineNodeState,
    )

    updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )
    updater.start()

    quorum = QuorumManager(
        updater=updater,
        check_interval=10.0,
    )
    quorum.start()

    router = QueryRouter(
        updater=updater,
        alerter=None,
        strategy=RoutingStrategy.CAUSAL_WEIGHTED,
        check_interval=15.0,
    )
    router.start()

    retrainer = AutoRetrainer(
        updater=updater,
        alerter=None,
        check_interval=30.0,
        drift_threshold_ms=3.0,
    )
    retrainer.start()

    coordinator = ByzantineRecoveryCoordinator(
        updater=updater,
        quorum_manager=quorum,
        router=router,
        retrainer=retrainer,
        check_interval=15.0,
    )
    coordinator.start()

    orchestrator = MultiNodeRecoveryOrchestrator(
        updater=updater,
        quorum_manager=quorum,
        coordinator=coordinator,
        check_interval=15.0,
    )
    orchestrator.start()

    logger.info(
        "MultiNodeRecoveryOrchestrator running. "
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
                "=== ORCHESTRATOR CYCLE %d ===", cycle
            )

            orch_status = orchestrator.status()
            logger.info(
                "Orchestrator: checks=%d sessions=%d "
                "completed=%d partial=%d failed=%d "
                "success_rate=%.3f",
                orch_status["check_count"],
                orch_status["sessions_total"],
                orch_status["sessions_completed"],
                orch_status["sessions_partial"],
                orch_status["sessions_failed"],
                orch_status["session_success_rate"],
            )
            logger.info(
                "  nodes_recovered=%d nodes_failed=%d",
                orch_status["nodes_recovered_total"],
                orch_status["nodes_failed_total"],
            )

            coord_status = coordinator.status()
            logger.info(
                "Coordinator: checks=%d detections=%d "
                "recoveries=%d",
                coord_status["check_count"],
                coord_status["detections_total"],
                coord_status["recoveries_total"],
            )

            quorum_status = quorum.status()
            logger.info(
                "Quorum: state=%s contributing=%d/%d "
                "violations=%d",
                quorum_status["quorum_state"],
                quorum_status["contributing_nodes"],
                quorum_status["total_nodes"],
                quorum_status["quorum_violations"],
            )

            # Cycle 2: inject multi-node Byzantine scenario
            if cycle == 2:
                logger.info(
                    "=== DEMO: Injecting 2-node "
                    "Byzantine scenario ==="
                )

                # Inject node-1 and node-2 as Byzantine
                for node_id, score in [
                    ("node-1", 0.72),
                    ("node-2", 0.68),
                ]:
                    coordinator._byzantine_scores[
                        node_id
                    ] = score
                    coordinator._node_states[node_id] = (
                        ByzantineNodeState.CONFIRMED
                    )
                    coordinator._consecutive_failures[
                        node_id
                    ] = 4
                    logger.info(
                        "Injected: %s CONFIRMED "
                        "score=%.2f failures=4",
                        node_id, score,
                    )

                # Trigger manual recovery session
                logger.info(
                    "Triggering manual recovery for "
                    "node-1 and node-2"
                )
                session_id = (
                    orchestrator.trigger_session_manual(
                        node_ids=["node-1", "node-2"],
                        reason="demo_multi_node_byzantine",
                    )
                )
                logger.info(
                    "Manual session started: %s",
                    session_id,
                )

                # Monitor session progress
                for wait in range(6):
                    time.sleep(15)
                    active = orchestrator.get_active_session()
                    if active:
                        logger.info(
                            "Session %s phase=%s "
                            "outcomes=%s",
                            active["session_id"],
                            active["phase"],
                            active["node_outcomes"],
                        )
                    else:
                        logger.info(
                            "Session %s completed",
                            session_id,
                        )
                        break

                # Show final report
                report = orchestrator.get_latest_report()
                if report:
                    logger.info(
                        "Final report: session=%s "
                        "phase=%s success=%d/%d "
                        "total_time=%s mttr=%s",
                        report["session_id"],
                        report["phase"],
                        report["success_count"],
                        len(report["affected_nodes"]),
                        f"{report['total_duration_seconds']:.1f}s"
                        if report["total_duration_seconds"]
                        else "N/A",
                        f"{report['mttr_seconds']:.1f}s"
                        if report["mttr_seconds"]
                        else "N/A",
                    )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        orchestrator.stop()
        coordinator.stop()
        retrainer.stop()
        router.stop()
        quorum.stop()
        updater.stop()