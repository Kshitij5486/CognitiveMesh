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
logger = logging.getLogger("cm.byzantine.quorum")


class QuorumState(Enum):
    HEALTHY       = "healthy"        # all nodes active
    DEGRADED      = "degraded"       # one node out, quorum maintained
    CRITICAL      = "critical"       # at minimum quorum
    QUORUM_LOST   = "quorum_lost"    # below minimum — emergency


class NodeCapacityState(Enum):
    FULL      = "full"        # fully operational
    REDUCED   = "reduced"     # degraded but contributing
    OFFLINE   = "offline"     # not contributing to quorum
    RECOVERING = "recovering" # in active recovery sequence


class QuorumDecision(Enum):
    ALLOW            = "allow"
    DENY_QUORUM_RISK = "deny_quorum_risk"
    DENY_CONCURRENT  = "deny_concurrent"
    ALLOW_EMERGENCY  = "allow_emergency"


class NodeQuorumStatus:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.capacity_state = NodeCapacityState.FULL
        self.causal_effect_ms: float = 0.0
        self.load: float = 0.0
        self.last_updated: float = time.time()
        self.recovery_started_at: Optional[float] = None
        self.consecutive_failures: int = 0
        self.byzantine_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "capacity_state": self.capacity_state.value,
            "causal_effect_ms": round(
                self.causal_effect_ms, 4
            ),
            "load": round(self.load, 2),
            "last_updated": self.last_updated,
            "consecutive_failures": self.consecutive_failures,
            "byzantine_score": round(self.byzantine_score, 4),
            "in_recovery": (
                self.recovery_started_at is not None
            ),
        }


class QuorumManager:
    """
    Quorum-aware cluster capacity manager.

    Enforces:
    - Minimum quorum: at least 2 of 3 nodes must be FULL or REDUCED
    - Maximum concurrent recoveries: at most 1 node in RECOVERING
      state at any time (prevents split-brain recovery)
    - Byzantine node isolation only when quorum is safe
    - Emergency quorum override when all paths lead to quorum loss

    This is the safety gate for all multi-node Byzantine recovery
    decisions. No node transition happens without QuorumManager
    approval.
    """

    ALL_NODES = ["node-1", "node-2", "node-3"]
    TOTAL_NODES = 3
    MINIMUM_QUORUM = 2           # need 2/3 nodes for quorum
    MAX_CONCURRENT_RECOVERIES = 1
    MAX_HISTORY = 200

    EFFECT_DEGRADED_THRESHOLD_MS   = 35.0
    EFFECT_CRITICAL_THRESHOLD_MS   = 50.0
    BYZANTINE_SCORE_THRESHOLD      = 0.7
    CONSECUTIVE_FAILURE_THRESHOLD  = 3

    def __init__(
        self,
        updater,
        check_interval: float = 10.0,
    ):
        self.updater = updater
        self.check_interval = check_interval

        self._node_statuses: dict[str, NodeQuorumStatus] = {
            node_id: NodeQuorumStatus(node_id)
            for node_id in self.ALL_NODES
        }
        self._quorum_state = QuorumState.HEALTHY
        self._decision_history: deque = deque(
            maxlen=self.MAX_HISTORY
        )
        self._state_history: deque = deque(
            maxlen=self.MAX_HISTORY
        )
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._check_count = 0
        self._quorum_violations = 0
        self._decisions_made = 0
        self._start_time: Optional[float] = None

    # ── Quorum calculation ─────────────────────────────────

    def _count_contributing_nodes(self) -> int:
        with self._lock:
            return sum(
                1 for s in self._node_statuses.values()
                if s.capacity_state in (
                    NodeCapacityState.FULL,
                    NodeCapacityState.REDUCED,
                )
            )

    def _count_recovering_nodes(self) -> int:
        with self._lock:
            return sum(
                1 for s in self._node_statuses.values()
                if s.capacity_state
                == NodeCapacityState.RECOVERING
            )

    def _compute_quorum_state(self) -> QuorumState:
        contributing = self._count_contributing_nodes()
        if contributing == self.TOTAL_NODES:
            return QuorumState.HEALTHY
        elif contributing == self.TOTAL_NODES - 1:
            return QuorumState.DEGRADED
        elif contributing == self.MINIMUM_QUORUM:
            return QuorumState.CRITICAL
        else:
            return QuorumState.QUORUM_LOST

    def _update_quorum_state(self):
        old_state = self._quorum_state
        new_state = self._compute_quorum_state()

        if new_state != old_state:
            self._quorum_state = new_state
            ts = datetime.now(timezone.utc).isoformat()
            self._state_history.append({
                "from": old_state.value,
                "to": new_state.value,
                "timestamp": ts,
                "contributing": self._count_contributing_nodes(),
            })

            if new_state == QuorumState.QUORUM_LOST:
                logger.critical(
                    "QUORUM LOST — cluster below minimum "
                    "capacity contributing=%d/%d",
                    self._count_contributing_nodes(),
                    self.TOTAL_NODES,
                )
                self._quorum_violations += 1
            elif new_state == QuorumState.CRITICAL:
                logger.warning(
                    "QUORUM CRITICAL — at minimum capacity "
                    "contributing=%d/%d",
                    self._count_contributing_nodes(),
                    self.TOTAL_NODES,
                )
            else:
                logger.info(
                    "Quorum state: %s → %s contributing=%d/%d",
                    old_state.value,
                    new_state.value,
                    self._count_contributing_nodes(),
                    self.TOTAL_NODES,
                )

    # ── Decision gate ──────────────────────────────────────

    def request_node_offline(
        self,
        node_id: str,
        reason: str,
    ) -> QuorumDecision:
        """
        Ask permission to take a node offline (reroute/isolate).
        Returns ALLOW only if quorum will be maintained after.
        """
        with self._lock:
            current_contributing = (
                self._count_contributing_nodes()
            )
            current_recovering = (
                self._count_recovering_nodes()
            )
            node_state = self._node_statuses[
                node_id
            ].capacity_state

        # Already offline — no change needed
        if node_state in (
            NodeCapacityState.OFFLINE,
            NodeCapacityState.RECOVERING,
        ):
            return QuorumDecision.ALLOW

        # Would taking this node offline break quorum?
        would_contribute = current_contributing - 1
        if would_contribute < self.MINIMUM_QUORUM:
            # Only allow if quorum is already lost (emergency)
            if self._quorum_state == QuorumState.QUORUM_LOST:
                decision = QuorumDecision.ALLOW_EMERGENCY
                logger.warning(
                    "EMERGENCY override: quorum already lost, "
                    "allowing offline of node=%s", node_id
                )
            else:
                decision = QuorumDecision.DENY_QUORUM_RISK
                logger.warning(
                    "DENY: taking node=%s offline would break "
                    "quorum (would have %d/%d contributing)",
                    node_id, would_contribute, self.TOTAL_NODES
                )
        elif current_recovering >= self.MAX_CONCURRENT_RECOVERIES:
            decision = QuorumDecision.DENY_CONCURRENT
            logger.warning(
                "DENY: max concurrent recoveries reached "
                "(%d/%d), cannot take node=%s offline",
                current_recovering,
                self.MAX_CONCURRENT_RECOVERIES,
                node_id,
            )
        else:
            decision = QuorumDecision.ALLOW

        self._record_decision(node_id, reason, decision)
        return decision

    def request_recovery_start(
        self,
        node_id: str,
        reason: str,
    ) -> QuorumDecision:
        """
        Ask permission to start a recovery sequence for a node.
        Checks concurrent recovery limit.
        """
        with self._lock:
            current_recovering = (
                self._count_recovering_nodes()
            )

        if current_recovering >= self.MAX_CONCURRENT_RECOVERIES:
            decision = QuorumDecision.DENY_CONCURRENT
            logger.warning(
                "DENY recovery start: already %d recovery "
                "sequence(s) active, cannot start for node=%s",
                current_recovering, node_id,
            )
        else:
            decision = QuorumDecision.ALLOW
            logger.info(
                "ALLOW recovery start for node=%s "
                "concurrent_recoveries=%d",
                node_id, current_recovering,
            )

        self._record_decision(node_id, reason, decision)
        return decision

    def _record_decision(
        self,
        node_id: str,
        reason: str,
        decision: QuorumDecision,
    ):
        self._decisions_made += 1
        self._decision_history.append({
            "decision_id": self._decisions_made,
            "node_id": node_id,
            "reason": reason[:80],
            "decision": decision.value,
            "quorum_state": self._quorum_state.value,
            "contributing": self._count_contributing_nodes(),
            "recovering": self._count_recovering_nodes(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ── Node state transitions ─────────────────────────────

    def mark_node_recovering(self, node_id: str):
        with self._lock:
            self._node_statuses[
                node_id
            ].capacity_state = NodeCapacityState.RECOVERING
            self._node_statuses[
                node_id
            ].recovery_started_at = time.time()
        self._update_quorum_state()
        logger.info(
            "Node %s → RECOVERING", node_id
        )

    def mark_node_offline(self, node_id: str):
        with self._lock:
            self._node_statuses[
                node_id
            ].capacity_state = NodeCapacityState.OFFLINE
        self._update_quorum_state()
        logger.warning(
            "Node %s → OFFLINE", node_id
        )

    def mark_node_reduced(self, node_id: str):
        with self._lock:
            status = self._node_statuses[node_id]
            status.capacity_state = NodeCapacityState.REDUCED
            status.recovery_started_at = None
        self._update_quorum_state()
        logger.info(
            "Node %s → REDUCED (partial capacity)", node_id
        )

    def mark_node_full(self, node_id: str):
        with self._lock:
            status = self._node_statuses[node_id]
            status.capacity_state = NodeCapacityState.FULL
            status.recovery_started_at = None
            status.consecutive_failures = 0
        self._update_quorum_state()
        logger.info(
            "Node %s → FULL (restored)", node_id
        )

    def record_node_failure(self, node_id: str):
        with self._lock:
            self._node_statuses[
                node_id
            ].consecutive_failures += 1
            failures = self._node_statuses[
                node_id
            ].consecutive_failures

        if failures >= self.CONSECUTIVE_FAILURE_THRESHOLD:
            logger.warning(
                "Node %s has %d consecutive failures "
                "— Byzantine behavior suspected",
                node_id, failures,
            )

    def update_byzantine_score(
        self, node_id: str, score: float
    ):
        with self._lock:
            self._node_statuses[
                node_id
            ].byzantine_score = score

        if score >= self.BYZANTINE_SCORE_THRESHOLD:
            logger.warning(
                "Node %s Byzantine score=%.3f exceeds "
                "threshold=%.3f",
                node_id, score,
                self.BYZANTINE_SCORE_THRESHOLD,
            )

    # ── Live metrics update ────────────────────────────────

    def _update_node_metrics(self):
        for node_id in self.ALL_NODES:
            snap = self.updater.get_current_snapshot(node_id)
            if not snap:
                continue

            effect = abs(snap["effect"])
            buf = self.updater.buffer
            load = 0.0
            if buf:
                df = buf.get_dataframe()
                if df is not None:
                    col = (
                        f"{node_id.replace('-', '_')}"
                        f"_active_queries"
                    )
                    if col in df.columns:
                        load = float(df[col].iloc[-1])

            with self._lock:
                status = self._node_statuses[node_id]
                status.causal_effect_ms = effect
                status.load = load
                status.last_updated = time.time()

                # Auto-classify capacity based on effect
                if (
                    status.capacity_state == NodeCapacityState.FULL
                    and effect > self.EFFECT_DEGRADED_THRESHOLD_MS
                ):
                    status.capacity_state = (
                        NodeCapacityState.REDUCED
                    )
                    logger.info(
                        "Node %s auto-classified REDUCED "
                        "effect=%.2fms > threshold=%.1fms",
                        node_id, effect,
                        self.EFFECT_DEGRADED_THRESHOLD_MS,
                    )
                elif (
                    status.capacity_state
                    == NodeCapacityState.REDUCED
                    and effect <= self.EFFECT_DEGRADED_THRESHOLD_MS
                ):
                    status.capacity_state = NodeCapacityState.FULL
                    logger.info(
                        "Node %s auto-classified FULL "
                        "effect=%.2fms <= threshold=%.1fms",
                        node_id, effect,
                        self.EFFECT_DEGRADED_THRESHOLD_MS,
                    )

    def _check_cycle(self):
        self._check_count += 1
        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            return

        self._update_node_metrics()
        self._update_quorum_state()

        with self._lock:
            states = {
                n: s.capacity_state.value
                for n, s in self._node_statuses.items()
            }

        logger.info(
            "Quorum check=%d state=%s contributing=%d/%d "
            "recovering=%d node_states=%s",
            self._check_count,
            self._quorum_state.value,
            self._count_contributing_nodes(),
            self.TOTAL_NODES,
            self._count_recovering_nodes(),
            states,
        )

    def _quorum_loop(self):
        logger.info(
            "QuorumManager started interval=%.1fs "
            "min_quorum=%d/%d max_concurrent=%d",
            self.check_interval,
            self.MINIMUM_QUORUM,
            self.TOTAL_NODES,
            self.MAX_CONCURRENT_RECOVERIES,
        )
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._check_cycle()
            except Exception as e:
                logger.error("Quorum check error: %s", e)

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._quorum_loop,
            name="quorum-manager",
            daemon=True,
        )
        self._thread.start()
        logger.info("QuorumManager started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("QuorumManager stopped")

    # ── Public query API ───────────────────────────────────

    def get_quorum_state(self) -> QuorumState:
        return self._quorum_state

    def get_node_status(self, node_id: str) -> dict:
        with self._lock:
            return self._node_statuses[node_id].to_dict()

    def get_all_node_statuses(self) -> dict:
        with self._lock:
            return {
                n: s.to_dict()
                for n, s in self._node_statuses.items()
            }

    def get_safe_to_offline(self) -> list:
        """Nodes that can safely go offline without losing quorum."""
        with self._lock:
            contributing = self._count_contributing_nodes()
            recovering = self._count_recovering_nodes()
            result = []
            for node_id, status in self._node_statuses.items():
                if status.capacity_state not in (
                    NodeCapacityState.FULL,
                    NodeCapacityState.REDUCED,
                ):
                    continue
                if contributing - 1 >= self.MINIMUM_QUORUM:
                    if recovering < self.MAX_CONCURRENT_RECOVERIES:
                        result.append(node_id)
            return result

    def get_decision_history(self, n: int = 10) -> list:
        with self._lock:
            history = list(self._decision_history)
        return history[-n:]

    def status(self) -> dict:
        with self._lock:
            states = {
                n: s.capacity_state.value
                for n, s in self._node_statuses.items()
            }
            effects = {
                n: round(s.causal_effect_ms, 4)
                for n, s in self._node_statuses.items()
            }
            byzantine_scores = {
                n: round(s.byzantine_score, 4)
                for n, s in self._node_statuses.items()
            }

        return {
            "running": self._running,
            "quorum_state": self._quorum_state.value,
            "contributing_nodes": (
                self._count_contributing_nodes()
            ),
            "recovering_nodes": (
                self._count_recovering_nodes()
            ),
            "minimum_quorum": self.MINIMUM_QUORUM,
            "total_nodes": self.TOTAL_NODES,
            "check_count": self._check_count,
            "quorum_violations": self._quorum_violations,
            "decisions_made": self._decisions_made,
            "node_states": states,
            "node_effects_ms": effects,
            "byzantine_scores": byzantine_scores,
            "safe_to_offline": self.get_safe_to_offline(),
            "uptime_seconds": round(
                time.time() - self._start_time, 1
            ) if self._start_time else 0.0,
        }


if __name__ == "__main__":
    logger.info("Starting QuorumManager demo")

    from streaming_updater import StreamingCausalUpdater

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

    logger.info(
        "QuorumManager running. "
        "Load generator in another terminal."
    )

    try:
        cycle = 0
        while True:
            time.sleep(20)
            cycle += 1

            engine_status = updater.status()
            if not engine_status.get("is_ready"):
                logger.info(
                    "Engine not ready buffer=%d/30",
                    engine_status.get("buffer_size", 0),
                )
                continue

            logger.info(
                "=== QUORUM CYCLE %d ===", cycle
            )

            qs = quorum.status()
            logger.info(
                "Quorum: state=%s contributing=%d/%d "
                "recovering=%d violations=%d decisions=%d",
                qs["quorum_state"],
                qs["contributing_nodes"],
                qs["total_nodes"],
                qs["recovering_nodes"],
                qs["quorum_violations"],
                qs["decisions_made"],
            )

            for node_id, state in qs["node_states"].items():
                effect = qs["node_effects_ms"].get(
                    node_id, 0
                )
                byz = qs["byzantine_scores"].get(
                    node_id, 0
                )
                logger.info(
                    "  node=%-8s state=%-10s "
                    "effect=%.2fms byz=%.3f",
                    node_id, state.upper(),
                    effect, byz,
                )

            safe = qs["safe_to_offline"]
            logger.info(
                "Safe to offline: %s", safe
            )

            # Demo: test decision gate on cycle 3
            if cycle == 3:
                logger.info(
                    "=== DEMO: Testing decision gate ==="
                )
                for node_id in ["node-1", "node-2", "node-3"]:
                    decision = quorum.request_node_offline(
                        node_id=node_id,
                        reason=f"demo request cycle {cycle}",
                    )
                    logger.info(
                        "  request_node_offline(%s) → %s",
                        node_id, decision.value
                    )

                # Simulate taking node-1 offline
                logger.info(
                    "Simulating node-1 going RECOVERING"
                )
                quorum.mark_node_recovering("node-1")

                # Now try to take node-2 offline — should DENY
                decision = quorum.request_node_offline(
                    node_id="node-2",
                    reason="demo concurrent attempt",
                )
                logger.info(
                    "  request_node_offline(node-2) "
                    "while node-1 recovering → %s",
                    decision.value
                )

                # Restore node-1
                time.sleep(2)
                quorum.mark_node_full("node-1")
                logger.info("node-1 restored to FULL")

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        quorum.stop()
        updater.stop()