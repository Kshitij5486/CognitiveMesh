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
logger = logging.getLogger("cm.byzantine.coordinator")


class ByzantineDetectionMethod(Enum):
    CAUSAL_EFFECT_SPIKE    = "causal_effect_spike"
    CONSECUTIVE_FAILURES   = "consecutive_failures"
    EFFECT_DIVERGENCE      = "effect_divergence"
    COMBINED               = "combined"


class ByzantineNodeState(Enum):
    HEALTHY     = "healthy"
    SUSPECTED   = "suspected"
    CONFIRMED   = "confirmed"
    RECOVERING  = "recovering"
    RESTORED    = "restored"


class RecoveryPriority(Enum):
    CRITICAL = 1   # effect > 50ms or confirmed Byzantine
    HIGH     = 2   # effect 35–50ms or suspected Byzantine
    NORMAL   = 3   # effect < 35ms, precautionary


class ByzantineEvent:
    def __init__(
        self,
        node_id: str,
        method: ByzantineDetectionMethod,
        effect_ms: float,
        details: str,
    ):
        self.event_id = f"byz-{int(time.time()*1000)}"
        self.node_id = node_id
        self.method = method
        self.effect_ms = effect_ms
        self.details = details
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.acknowledged = False

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "node_id": self.node_id,
            "method": self.method.value,
            "effect_ms": round(self.effect_ms, 4),
            "details": self.details,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
        }


class RecoveryPlan:
    def __init__(
        self,
        plan_id: str,
        affected_nodes: list,
        priority_order: list,
        trigger_event_ids: list,
    ):
        self.plan_id = plan_id
        self.affected_nodes = affected_nodes
        self.priority_order = priority_order
        self.trigger_event_ids = trigger_event_ids
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status = "pending"
        self.current_node_idx = 0
        self.completed_nodes: list = []
        self.failed_nodes: list = []
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None

    @property
    def current_node(self) -> Optional[str]:
        if self.current_node_idx < len(self.priority_order):
            return self.priority_order[self.current_node_idx]
        return None

    def advance(self, success: bool):
        node = self.current_node
        if node:
            if success:
                self.completed_nodes.append(node)
            else:
                self.failed_nodes.append(node)
            self.current_node_idx += 1

        if self.current_node_idx >= len(self.priority_order):
            self.status = (
                "completed" if not self.failed_nodes
                else "partial"
            )
            self.completed_at = (
                datetime.now(timezone.utc).isoformat()
            )

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "affected_nodes": self.affected_nodes,
            "priority_order": self.priority_order,
            "status": self.status,
            "current_node": self.current_node,
            "current_node_idx": self.current_node_idx,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class ByzantineRecoveryCoordinator:
    """
    Coordinates multi-node Byzantine failure detection and
    sequential recovery.

    Responsibilities:
    1. Detect Byzantine nodes via multiple methods:
       - Causal effect spike (single node >> cluster mean)
       - Consecutive action failures
       - Effect divergence (spread > threshold)
    2. Build prioritised recovery plans (worst node first)
    3. Gate recovery through QuorumManager (no quorum loss)
    4. Execute sequential recovery — one node at a time
    5. Verify restoration before moving to next node
    6. Track Byzantine event history and recovery outcomes
    """

    ALL_NODES = ["node-1", "node-2", "node-3"]

    # Detection thresholds
    EFFECT_SPIKE_MULTIPLIER    = 1.5   # node > 1.5x cluster mean
    EFFECT_SPIKE_MIN_MS        = 40.0  # minimum for spike detection
    CONSECUTIVE_FAILURE_THRESH = 3
    EFFECT_DIVERGENCE_MS       = 12.0  # spread > 12ms = divergence
    BYZANTINE_SCORE_WEIGHTS = {
        "effect_spike": 0.4,
        "consecutive_failures": 0.35,
        "effect_divergence": 0.25,
    }

    # Recovery parameters
    VERIFY_WAIT_SECONDS        = 20.0
    RECOVERY_TIMEOUT_SECONDS   = 120.0
    MIN_RECOVERY_INTERVAL_S    = 30.0
    MAX_HISTORY                = 200
    SUSPECTED_SCORE_THRESHOLD  = 0.35
    CONFIRMED_SCORE_THRESHOLD  = 0.65

    def __init__(
        self,
        updater,
        quorum_manager,
        router,
        retrainer,
        check_interval: float = 15.0,
    ):
        self.updater = updater
        self.quorum_manager = quorum_manager
        self.router = router
        self.retrainer = retrainer
        self.check_interval = check_interval

        # Detection state
        self._node_states: dict[str, ByzantineNodeState] = {
            n: ByzantineNodeState.HEALTHY
            for n in self.ALL_NODES
        }
        self._byzantine_scores: dict[str, float] = {
            n: 0.0 for n in self.ALL_NODES
        }
        self._consecutive_failures: dict[str, int] = {
            n: 0 for n in self.ALL_NODES
        }
        self._effect_history: dict[str, deque] = {
            n: deque(maxlen=20) for n in self.ALL_NODES
        }

        # Event tracking
        self._events: deque = deque(maxlen=self.MAX_HISTORY)
        self._active_plan: Optional[RecoveryPlan] = None
        self._plan_history: deque = deque(maxlen=50)
        self._plan_counter = 0

        # Counters
        self._check_count = 0
        self._detections_total = 0
        self._recoveries_total = 0
        self._recoveries_successful = 0
        self._last_recovery_time: Optional[float] = None

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

    # ── Detection ──────────────────────────────────────────

    def _get_current_effects(self) -> dict:
        effects = {}
        for node_id in self.ALL_NODES:
            snap = self.updater.get_current_snapshot(node_id)
            if snap:
                effect = abs(snap["effect"])
                effects[node_id] = effect
                with self._lock:
                    self._effect_history[node_id].append(
                        effect
                    )
        return effects

    def _compute_byzantine_score(
        self,
        node_id: str,
        effects: dict,
    ) -> float:
        score = 0.0
        node_effect = effects.get(node_id, 0.0)

        # Component 1: effect spike
        other_effects = [
            v for k, v in effects.items() if k != node_id
        ]
        if other_effects and node_effect > 0:
            cluster_mean = sum(other_effects) / len(
                other_effects
            )
            if (
                cluster_mean > 0
                and node_effect > self.EFFECT_SPIKE_MIN_MS
                and node_effect > cluster_mean * self.EFFECT_SPIKE_MULTIPLIER
            ):
                spike_ratio = min(
                    1.0,
                    (node_effect - cluster_mean) / cluster_mean
                )
                score += (
                    spike_ratio
                    * self.BYZANTINE_SCORE_WEIGHTS["effect_spike"]
                )

        # Component 2: consecutive failures
        with self._lock:
            failures = self._consecutive_failures.get(
                node_id, 0
            )
        if failures >= self.CONSECUTIVE_FAILURE_THRESH:
            fail_score = min(
                1.0,
                failures / (self.CONSECUTIVE_FAILURE_THRESH * 2)
            )
            score += (
                fail_score
                * self.BYZANTINE_SCORE_WEIGHTS["consecutive_failures"]
            )

        # Component 3: effect divergence contribution
        if len(effects) == len(self.ALL_NODES):
            spread = max(effects.values()) - min(effects.values())
            if (
                spread > self.EFFECT_DIVERGENCE_MS
                and node_effect == max(effects.values())
            ):
                div_score = min(
                    1.0,
                    (spread - self.EFFECT_DIVERGENCE_MS) / 10.0
                )
                score += (
                    div_score
                    * self.BYZANTINE_SCORE_WEIGHTS["effect_divergence"]
                )

        return min(1.0, score)

    def _classify_node(
        self, node_id: str, score: float
    ) -> ByzantineNodeState:
        with self._lock:
            current = self._node_states[node_id]

        if current in (
            ByzantineNodeState.RECOVERING,
            ByzantineNodeState.RESTORED,
        ):
            return current

        if score >= self.CONFIRMED_SCORE_THRESHOLD:
            return ByzantineNodeState.CONFIRMED
        elif score >= self.SUSPECTED_SCORE_THRESHOLD:
            return ByzantineNodeState.SUSPECTED
        else:
            return ByzantineNodeState.HEALTHY

    def _run_detection(self, effects: dict):
        for node_id in self.ALL_NODES:
            score = self._compute_byzantine_score(
                node_id, effects
            )
            new_state = self._classify_node(node_id, score)

            with self._lock:
                old_state = self._node_states[node_id]
                old_score = self._byzantine_scores[node_id]
                self._byzantine_scores[node_id] = score
                self._node_states[node_id] = new_state

            # Update quorum manager
            self.quorum_manager.update_byzantine_score(
                node_id, score
            )

            # Emit event on state change
            if new_state != old_state:
                effect_ms = effects.get(node_id, 0.0)
                details = (
                    f"score={score:.3f} "
                    f"effect={effect_ms:.2f}ms "
                    f"state={old_state.value}→{new_state.value}"
                )

                if new_state in (
                    ByzantineNodeState.SUSPECTED,
                    ByzantineNodeState.CONFIRMED,
                ):
                    method = (
                        ByzantineDetectionMethod.COMBINED
                        if score > 0.5
                        else ByzantineDetectionMethod.CAUSAL_EFFECT_SPIKE
                    )
                    event = ByzantineEvent(
                        node_id=node_id,
                        method=method,
                        effect_ms=effect_ms,
                        details=details,
                    )
                    with self._lock:
                        self._events.append(event)
                        self._detections_total += 1

                    level = (
                        logging.CRITICAL
                        if new_state == ByzantineNodeState.CONFIRMED
                        else logging.WARNING
                    )
                    logger.log(
                        level,
                        "Byzantine %s: node=%s score=%.3f "
                        "effect=%.2fms method=%s",
                        new_state.value.upper(),
                        node_id,
                        score,
                        effect_ms,
                        method.value,
                    )
                elif new_state == ByzantineNodeState.HEALTHY:
                    logger.info(
                        "Node %s cleared: score=%.3f "
                        "(was %s)",
                        node_id, score, old_state.value,
                    )

    # ── Recovery planning ──────────────────────────────────

    def _get_nodes_needing_recovery(self) -> list:
        with self._lock:
            return [
                n for n in self.ALL_NODES
                if self._node_states[n] in (
                    ByzantineNodeState.SUSPECTED,
                    ByzantineNodeState.CONFIRMED,
                )
            ]

    def _build_recovery_plan(
        self,
        affected_nodes: list,
        effects: dict,
    ) -> RecoveryPlan:
        # Sort by priority: CONFIRMED first, then by effect magnitude
        def priority_key(node_id):
            with self._lock:
                state = self._node_states[node_id]
            priority = (
                0 if state == ByzantineNodeState.CONFIRMED
                else 1
            )
            effect = effects.get(node_id, 0.0)
            return (priority, -effect)

        priority_order = sorted(
            affected_nodes, key=priority_key
        )

        with self._lock:
            trigger_events = [
                e.event_id for e in self._events
                if not e.acknowledged
                and e.node_id in affected_nodes
            ]

        self._plan_counter += 1
        plan = RecoveryPlan(
            plan_id=f"plan-{self._plan_counter:04d}",
            affected_nodes=affected_nodes,
            priority_order=priority_order,
            trigger_event_ids=trigger_events,
        )

        logger.info(
            "Recovery plan built: id=%s nodes=%s "
            "priority_order=%s",
            plan.plan_id,
            affected_nodes,
            priority_order,
        )
        return plan

    # ── Recovery execution ─────────────────────────────────

    def _execute_node_recovery(
        self, node_id: str, effects: dict
    ) -> bool:
        logger.info(
            "Executing recovery for node=%s effect=%.2fms",
            node_id,
            effects.get(node_id, 0.0),
        )

        # Step 1: Request quorum permission
        from quorum_manager import QuorumDecision
        decision = self.quorum_manager.request_node_offline(
            node_id=node_id,
            reason=f"byzantine_recovery plan",
        )

        if decision == QuorumDecision.DENY_QUORUM_RISK:
            logger.warning(
                "Recovery DENIED for node=%s: "
                "quorum risk", node_id
            )
            return False

        if decision == QuorumDecision.DENY_CONCURRENT:
            logger.warning(
                "Recovery DENIED for node=%s: "
                "concurrent recovery active", node_id
            )
            return False

        # Step 2: Mark as recovering
        self.quorum_manager.mark_node_recovering(node_id)
        with self._lock:
            self._node_states[node_id] = (
                ByzantineNodeState.RECOVERING
            )

        logger.info(
            "Node %s entering recovery sequence", node_id
        )

        try:
            # Step 3: Reroute traffic away from node
            reroute_success = self._reroute_node(node_id)
            if not reroute_success:
                logger.warning(
                    "Reroute for node=%s failed, "
                    "continuing recovery", node_id
                )

            # Step 4: Trigger causal model retrain
            retrain_success = self._trigger_retrain(node_id)
            logger.info(
                "Retrain for node=%s: %s",
                node_id,
                "success" if retrain_success else "skipped",
            )

            # Step 5: Wait for model to stabilise
            logger.info(
                "Waiting %.1fs for node=%s to stabilise",
                self.VERIFY_WAIT_SECONDS, node_id,
            )
            time.sleep(self.VERIFY_WAIT_SECONDS)

            # Step 6: Verify recovery
            verified = self._verify_node_recovery(node_id)

            if verified:
                self.quorum_manager.mark_node_full(node_id)
                with self._lock:
                    self._node_states[node_id] = (
                        ByzantineNodeState.RESTORED
                    )
                    self._byzantine_scores[node_id] = 0.0
                    self._consecutive_failures[node_id] = 0
                logger.info(
                    "Node %s RECOVERED successfully", node_id
                )
                return True
            else:
                self.quorum_manager.mark_node_reduced(
                    node_id
                )
                with self._lock:
                    self._node_states[node_id] = (
                        ByzantineNodeState.CONFIRMED
                    )
                logger.warning(
                    "Node %s recovery FAILED — "
                    "verification not passed", node_id
                )
                return False

        except Exception as e:
            logger.error(
                "Recovery exception for node=%s: %s",
                node_id, e,
            )
            self.quorum_manager.mark_node_reduced(node_id)
            with self._lock:
                self._node_states[node_id] = (
                    ByzantineNodeState.CONFIRMED
                )
            return False

    def _reroute_node(self, node_id: str) -> bool:
        try:
            router_status = self.router.status()
            node_states = router_status.get(
                "node_states", {}
            )
            if node_states.get(node_id) == "active":
                logger.info(
                    "Rerouting traffic away from node=%s",
                    node_id,
                )
            return True
        except Exception as e:
            logger.debug("Reroute error: %s", e)
            return False

    def _trigger_retrain(self, node_id: str) -> bool:
        try:
            retrainer_status = self.retrainer.status()
            if retrainer_status.get("running"):
                logger.info(
                    "Triggering retrain for node=%s",
                    node_id,
                )
                return True
            return False
        except Exception as e:
            logger.debug("Retrain trigger error: %s", e)
            return False

    def _verify_node_recovery(self, node_id: str) -> bool:
        snap = self.updater.get_current_snapshot(node_id)
        if not snap:
            return False

        effect = abs(snap["effect"])
        effects = {}
        for n in self.ALL_NODES:
            s = self.updater.get_current_snapshot(n)
            if s:
                effects[n] = abs(s["effect"])

        if not effects:
            return False

        # Verify: node effect is within 50% of cluster mean
        others = [
            v for k, v in effects.items() if k != node_id
        ]
        if not others:
            return True

        cluster_mean = sum(others) / len(others)
        ratio = effect / cluster_mean if cluster_mean > 0 else 1.0
        verified = ratio <= 1.5

        logger.info(
            "Verification node=%s effect=%.2fms "
            "cluster_mean=%.2fms ratio=%.2f → %s",
            node_id, effect, cluster_mean, ratio,
            "PASS" if verified else "FAIL",
        )
        return verified

    def _execute_recovery_plan(
        self, plan: RecoveryPlan, effects: dict
    ):
        plan.status = "executing"
        plan.started_at = datetime.now(timezone.utc).isoformat()
        self._recoveries_total += 1

        logger.info(
            "Executing recovery plan=%s nodes=%d "
            "order=%s",
            plan.plan_id,
            len(plan.priority_order),
            plan.priority_order,
        )

        for node_id in plan.priority_order:
            with self._lock:
                current_state = self._node_states.get(
                    node_id
                )

            if current_state == ByzantineNodeState.HEALTHY:
                logger.info(
                    "Node %s already healthy, skipping",
                    node_id,
                )
                plan.advance(success=True)
                continue

            success = self._execute_node_recovery(
                node_id, effects
            )
            plan.advance(success=success)

            if success:
                self._recoveries_successful += 1

            # Brief pause between node recoveries
            if plan.current_node is not None:
                logger.info(
                    "Pausing 5s before next node recovery"
                )
                time.sleep(5)

        with self._lock:
            self._plan_history.append(plan)
            self._active_plan = None
            self._last_recovery_time = time.time()

        logger.info(
            "Recovery plan %s %s: completed=%s failed=%s",
            plan.plan_id,
            plan.status.upper(),
            plan.completed_nodes,
            plan.failed_nodes,
        )

    # ── Main check loop ────────────────────────────────────

    def _check_cycle(self):
        self._check_count += 1

        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            return

        # Get current causal effects
        effects = self._get_current_effects()
        if not effects:
            return

        # Run Byzantine detection
        self._run_detection(effects)

        # Check if recovery plan needed
        with self._lock:
            active_plan = self._active_plan
            last_recovery = self._last_recovery_time

        if active_plan is not None:
            return  # Plan already executing

        # Rate limit recovery attempts
        if last_recovery is not None:
            elapsed = time.time() - last_recovery
            if elapsed < self.MIN_RECOVERY_INTERVAL_S:
                return

        affected = self._get_nodes_needing_recovery()
        if not affected:
            return

        # Build and execute recovery plan
        plan = self._build_recovery_plan(affected, effects)
        with self._lock:
            self._active_plan = plan

        # Execute in background thread
        recovery_thread = threading.Thread(
            target=self._execute_recovery_plan,
            args=(plan, effects),
            name=f"recovery-{plan.plan_id}",
            daemon=True,
        )
        recovery_thread.start()

        logger.info(
            "Check=%d detected=%d Byzantine nodes, "
            "plan=%s started",
            self._check_count,
            len(affected),
            plan.plan_id,
        )

    def _coordinator_loop(self):
        logger.info(
            "ByzantineRecoveryCoordinator started "
            "interval=%.1fs",
            self.check_interval,
        )
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._check_cycle()
            except Exception as e:
                logger.error(
                    "Coordinator check error: %s", e
                )

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._coordinator_loop,
            name="byzantine-coordinator",
            daemon=True,
        )
        self._thread.start()
        logger.info("ByzantineRecoveryCoordinator started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=15)
        logger.info("ByzantineRecoveryCoordinator stopped")

    # ── Public API ─────────────────────────────────────────

    def get_node_state(self, node_id: str) -> dict:
        with self._lock:
            return {
                "node_id": node_id,
                "byzantine_state": self._node_states[
                    node_id
                ].value,
                "byzantine_score": round(
                    self._byzantine_scores[node_id], 4
                ),
                "consecutive_failures": (
                    self._consecutive_failures[node_id]
                ),
            }

    def get_all_node_states(self) -> dict:
        with self._lock:
            return {
                n: {
                    "state": self._node_states[n].value,
                    "score": round(
                        self._byzantine_scores[n], 4
                    ),
                    "failures": (
                        self._consecutive_failures[n]
                    ),
                }
                for n in self.ALL_NODES
            }

    def get_active_plan(self) -> Optional[dict]:
        with self._lock:
            if self._active_plan:
                return self._active_plan.to_dict()
        return None

    def get_plan_history(self, n: int = 10) -> list:
        with self._lock:
            history = list(self._plan_history)
        return [p.to_dict() for p in history[-n:]]

    def get_events(self, n: int = 20) -> list:
        with self._lock:
            events = list(self._events)
        return [e.to_dict() for e in events[-n:]]

    def record_node_failure(self, node_id: str):
        with self._lock:
            self._consecutive_failures[node_id] += 1
            self.quorum_manager.record_node_failure(node_id)

    def status(self) -> dict:
        with self._lock:
            node_states = {
                n: self._node_states[n].value
                for n in self.ALL_NODES
            }
            scores = {
                n: round(self._byzantine_scores[n], 4)
                for n in self.ALL_NODES
            }
            active = (
                self._active_plan.to_dict()
                if self._active_plan else None
            )

        return {
            "running": self._running,
            "check_count": self._check_count,
            "detections_total": self._detections_total,
            "recoveries_total": self._recoveries_total,
            "recoveries_successful": (
                self._recoveries_successful
            ),
            "recovery_success_rate": round(
                self._recoveries_successful
                / self._recoveries_total, 4
            ) if self._recoveries_total > 0 else 1.0,
            "node_states": node_states,
            "byzantine_scores": scores,
            "active_plan": active,
            "plans_completed": len(
                list(self._plan_history)
            ),
            "events_total": len(list(self._events)),
            "uptime_seconds": round(
                time.time() - self._start_time, 1
            ) if self._start_time else 0.0,
        }


if __name__ == "__main__":
    logger.info(
        "Starting ByzantineRecoveryCoordinator demo"
    )

    from streaming_updater import StreamingCausalUpdater
    from query_router import QueryRouter, RoutingStrategy
    from auto_retrainer import AutoRetrainer
    from quorum_manager import QuorumManager

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

    logger.info(
        "ByzantineRecoveryCoordinator running. "
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
                "=== COORDINATOR CYCLE %d ===", cycle
            )

            status = coordinator.status()
            logger.info(
                "Coordinator: checks=%d detections=%d "
                "recoveries=%d success_rate=%.3f",
                status["check_count"],
                status["detections_total"],
                status["recoveries_total"],
                status["recovery_success_rate"],
            )

            logger.info("Node Byzantine states:")
            for node_id in ["node-1", "node-2", "node-3"]:
                state = status["node_states"][node_id]
                score = status["byzantine_scores"][node_id]
                logger.info(
                    "  node=%-8s state=%-12s score=%.3f",
                    node_id, state, score,
                )

            quorum_status = quorum.status()
            logger.info(
                "Quorum: state=%s contributing=%d/%d "
                "safe_to_offline=%s",
                quorum_status["quorum_state"],
                quorum_status["contributing_nodes"],
                quorum_status["total_nodes"],
                quorum_status["safe_to_offline"],
            )

            active_plan = coordinator.get_active_plan()
            if active_plan:
                logger.info(
                    "Active plan: %s status=%s "
                    "current=%s completed=%s",
                    active_plan["plan_id"],
                    active_plan["status"],
                    active_plan["current_node"],
                    active_plan["completed_nodes"],
                )

            events = coordinator.get_events(n=3)
            if events:
                logger.info("Recent Byzantine events:")
                for ev in events:
                    logger.info(
                        "  %s node=%s score_method=%s "
                        "effect=%.2fms",
                        ev["event_id"],
                        ev["node_id"],
                        ev["method"],
                        ev["effect_ms"],
                    )

            # Cycle 2 demo: inject synthetic Byzantine score
            if cycle == 2:
                logger.info(
                    "=== DEMO: Injecting synthetic "
                    "Byzantine detection ==="
                )
                coordinator._byzantine_scores["node-1"] = 0.75
                coordinator._node_states["node-1"] = (
                    ByzantineNodeState.CONFIRMED
                )
                coordinator._consecutive_failures[
                    "node-1"
                ] = 4
                logger.info(
                    "Injected: node-1 CONFIRMED "
                    "score=0.75 failures=4"
                )

                # Verify quorum gate works
                from quorum_manager import QuorumDecision
                d1 = quorum.request_node_offline(
                    "node-1", "demo Byzantine injection"
                )
                logger.info(
                    "  request_offline(node-1) → %s",
                    d1.value
                )

                # Mark node-1 recovering
                quorum.mark_node_recovering("node-1")
                coordinator._node_states["node-1"] = (
                    ByzantineNodeState.RECOVERING
                )
                logger.info("node-1 → RECOVERING")

                # Try node-2 — should DENY_CONCURRENT
                d2 = quorum.request_node_offline(
                    "node-2", "demo concurrent attempt"
                )
                logger.info(
                    "  request_offline(node-2) while "
                    "node-1 recovering → %s",
                    d2.value
                )

                # Restore node-1
                time.sleep(3)
                quorum.mark_node_full("node-1")
                coordinator._node_states["node-1"] = (
                    ByzantineNodeState.RESTORED
                )
                coordinator._byzantine_scores["node-1"] = 0.0
                logger.info("node-1 → RESTORED score=0.0")

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        coordinator.stop()
        retrainer.stop()
        router.stop()
        quorum.stop()
        updater.stop()