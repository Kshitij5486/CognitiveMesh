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
logger = logging.getLogger("cm.byzantine.quorum_router")


class RoutingDecisionType(Enum):
    NORMAL            = "normal"           # all nodes active
    DEGRADED          = "degraded"         # one node excluded
    CRITICAL          = "critical"         # two nodes excluded
    EMERGENCY_UNIFORM = "emergency_uniform" # quorum lost, uniform


class NodeRoutingState(Enum):
    ACTIVE     = "active"      # receiving traffic normally
    REDUCED    = "reduced"     # receiving traffic, caution
    EXCLUDED   = "excluded"    # in recovery/offline, no traffic
    EMERGENCY  = "emergency"   # quorum lost, receiving traffic anyway


class RoutingDecision:
    def __init__(
        self,
        decision_id: int,
        weights: dict,
        excluded_nodes: list,
        decision_type: RoutingDecisionType,
        quorum_state: str,
        reason: str,
    ):
        self.decision_id = decision_id
        self.weights = weights
        self.excluded_nodes = excluded_nodes
        self.decision_type = decision_type
        self.quorum_state = quorum_state
        self.reason = reason
        self.timestamp = datetime.now(
            timezone.utc
        ).isoformat()

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "weights": {
                k: round(v, 6)
                for k, v in self.weights.items()
            },
            "excluded_nodes": self.excluded_nodes,
            "decision_type": self.decision_type.value,
            "quorum_state": self.quorum_state,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class QuorumAwareRouter:
    """
    Quorum-aware traffic router for the CognitiveMesh
    Byzantine recovery cluster.

    Differences from standard QueryRouter:
    1. Consults QuorumManager for node capacity states
    2. Excludes RECOVERING/OFFLINE nodes from traffic
    3. Redistributes excluded nodes' weight proportionally
    4. Emits RoutingDecision objects with full context
    5. Handles quorum_lost with emergency uniform routing
    6. Tracks routing stability metrics (weight variance)
    7. Enforces minimum weight floor (5%) for REDUCED nodes

    Weight computation:
      Base weight = 1 / causal_effect_ms  (lower effect = more traffic)
      REDUCED nodes: weight * REDUCED_WEIGHT_PENALTY (0.5)
      RECOVERING/OFFLINE nodes: weight = 0.0
      Normalise to sum = 1.0
      Apply minimum floor MIN_WEIGHT_FLOOR for active nodes
    """

    ALL_NODES = ["node-1", "node-2", "node-3"]

    REDUCED_WEIGHT_PENALTY  = 0.5
    MIN_WEIGHT_FLOOR        = 0.05
    MAX_WEIGHT_CAP          = 0.85
    STABILITY_WINDOW        = 30
    CHECK_INTERVAL          = 10.0
    MAX_DECISION_HISTORY    = 200

    def __init__(
        self,
        updater,
        quorum_manager,
        check_interval: float = 10.0,
    ):
        self.updater = updater
        self.quorum_manager = quorum_manager
        self.check_interval = check_interval

        self._current_weights: dict = {
            n: 1.0 / len(self.ALL_NODES)
            for n in self.ALL_NODES
        }
        self._current_decision: Optional[RoutingDecision] = (
            None
        )
        self._decision_history: deque = deque(
            maxlen=self.MAX_DECISION_HISTORY
        )
        self._weight_history: dict = {
            n: deque(maxlen=self.STABILITY_WINDOW)
            for n in self.ALL_NODES
        }
        self._node_routing_states: dict = {
            n: NodeRoutingState.ACTIVE
            for n in self.ALL_NODES
        }

        # Counters
        self._check_count = 0
        self._decision_count = 0
        self._exclusion_events = 0
        self._redistribution_events = 0
        self._emergency_events = 0

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

    # ── Weight computation ─────────────────────────────────

    def _get_causal_effects(self) -> dict:
        effects = {}
        for node_id in self.ALL_NODES:
            snap = self.updater.get_current_snapshot(
                node_id
            )
            if snap:
                effects[node_id] = max(
                    0.1, abs(snap["effect"])
                )
            else:
                effects[node_id] = 30.0  # default
        return effects

    def _get_node_capacity_states(self) -> dict:
        from quorum_manager import NodeCapacityState
        all_statuses = (
            self.quorum_manager.get_all_node_statuses()
        )
        result = {}
        for node_id in self.ALL_NODES:
            status = all_statuses.get(node_id, {})
            cap_state = status.get(
                "capacity_state", "full"
            )
            result[node_id] = cap_state
        return result

    def _compute_weights(
        self,
        effects: dict,
        capacity_states: dict,
        quorum_state: str,
    ) -> tuple:
        """
        Returns (weights, excluded_nodes, decision_type,
                 routing_states, reason)
        """
        from quorum_manager import NodeCapacityState

        # Emergency: quorum lost → uniform across all
        if quorum_state == "quorum_lost":
            uniform = 1.0 / len(self.ALL_NODES)
            weights = {n: uniform for n in self.ALL_NODES}
            routing_states = {
                n: NodeRoutingState.EMERGENCY
                for n in self.ALL_NODES
            }
            return (
                weights,
                [],
                RoutingDecisionType.EMERGENCY_UNIFORM,
                routing_states,
                "quorum_lost: emergency uniform routing",
            )

        # Determine excluded nodes
        excluded = []
        for node_id in self.ALL_NODES:
            cap = capacity_states.get(node_id, "full")
            if cap in ("offline", "recovering"):
                excluded.append(node_id)

        active_nodes = [
            n for n in self.ALL_NODES
            if n not in excluded
        ]

        if not active_nodes:
            # Absolute emergency: route to all
            uniform = 1.0 / len(self.ALL_NODES)
            weights = {n: uniform for n in self.ALL_NODES}
            routing_states = {
                n: NodeRoutingState.EMERGENCY
                for n in self.ALL_NODES
            }
            return (
                weights,
                [],
                RoutingDecisionType.EMERGENCY_UNIFORM,
                routing_states,
                "no active nodes: emergency uniform",
            )

        # Compute base weights from causal effects
        raw_weights = {}
        for node_id in active_nodes:
            effect = effects.get(node_id, 30.0)
            base = 1.0 / effect
            cap = capacity_states.get(node_id, "full")
            if cap == "reduced":
                base *= self.REDUCED_WEIGHT_PENALTY
            raw_weights[node_id] = base

        # Excluded nodes get zero
        for node_id in excluded:
            raw_weights[node_id] = 0.0

        # Normalise
        total = sum(raw_weights.values())
        if total <= 0:
            equal = 1.0 / len(active_nodes)
            weights = {n: 0.0 for n in self.ALL_NODES}
            for n in active_nodes:
                weights[n] = equal
        else:
            weights = {
                n: raw_weights[n] / total
                for n in self.ALL_NODES
            }

        # Apply floor and cap for active nodes
        for node_id in active_nodes:
            weights[node_id] = max(
                self.MIN_WEIGHT_FLOOR,
                min(self.MAX_WEIGHT_CAP, weights[node_id])
            )

        # Re-normalise after floor/cap
        active_total = sum(
            weights[n] for n in active_nodes
        )
        if active_total > 0:
            for node_id in active_nodes:
                weights[node_id] = (
                    weights[node_id] / active_total
                )

        # Determine routing states
        routing_states = {}
        for node_id in self.ALL_NODES:
            if node_id in excluded:
                routing_states[node_id] = (
                    NodeRoutingState.EXCLUDED
                )
            elif capacity_states.get(node_id) == "reduced":
                routing_states[node_id] = (
                    NodeRoutingState.REDUCED
                )
            else:
                routing_states[node_id] = (
                    NodeRoutingState.ACTIVE
                )

        # Decision type
        n_excluded = len(excluded)
        if n_excluded == 0:
            decision_type = RoutingDecisionType.NORMAL
        elif n_excluded == 1:
            decision_type = RoutingDecisionType.DEGRADED
        else:
            decision_type = RoutingDecisionType.CRITICAL

        reason = (
            f"excluded={excluded} "
            f"quorum={quorum_state} "
            f"type={decision_type.value}"
        )

        return (
            weights,
            excluded,
            decision_type,
            routing_states,
            reason,
        )

    # ── Stability metrics ──────────────────────────────────

    def _compute_weight_stability(
        self, node_id: str
    ) -> float:
        with self._lock:
            history = list(self._weight_history[node_id])
        if len(history) < 2:
            return 1.0
        import statistics
        mean = statistics.mean(history)
        std = statistics.stdev(history)
        cv = std / mean if mean > 0 else 0.0
        return max(0.0, 1.0 - cv * 5)

    def _get_cluster_stability(self) -> float:
        stabilities = [
            self._compute_weight_stability(n)
            for n in self.ALL_NODES
        ]
        return sum(stabilities) / len(stabilities)

    # ── Check cycle ────────────────────────────────────────

    def _check_cycle(self):
        self._check_count += 1

        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            return

        effects = self._get_causal_effects()
        capacity_states = self._get_node_capacity_states()
        quorum_state = (
            self.quorum_manager.get_quorum_state().value
        )

        (
            weights,
            excluded,
            decision_type,
            routing_states,
            reason,
        ) = self._compute_weights(
            effects, capacity_states, quorum_state
        )

        # Detect changes
        with self._lock:
            old_excluded = [
                n for n in self.ALL_NODES
                if self._node_routing_states.get(n)
                == NodeRoutingState.EXCLUDED
            ]

        if excluded != old_excluded:
            if excluded:
                self._exclusion_events += 1
                logger.warning(
                    "Routing exclusion change: "
                    "old=%s new=%s decision=%s",
                    old_excluded, excluded,
                    decision_type.value,
                )
            if old_excluded and not excluded:
                self._redistribution_events += 1
                logger.info(
                    "Routing restored: nodes %s "
                    "returned to active pool",
                    old_excluded,
                )

        if decision_type == RoutingDecisionType.EMERGENCY_UNIFORM:
            self._emergency_events += 1

        # Update state
        with self._lock:
            self._current_weights = weights
            self._node_routing_states = routing_states
            for node_id in self.ALL_NODES:
                self._weight_history[node_id].append(
                    weights[node_id]
                )

        # Record decision
        self._decision_count += 1
        decision = RoutingDecision(
            decision_id=self._decision_count,
            weights=weights,
            excluded_nodes=excluded,
            decision_type=decision_type,
            quorum_state=quorum_state,
            reason=reason,
        )
        with self._lock:
            self._current_decision = decision
            self._decision_history.append(decision)

        # Log every 5 checks or on exclusion changes
        if (
            self._check_count % 5 == 0
            or excluded != old_excluded
        ):
            logger.info(
                "Quorum routing check=%d type=%s "
                "excluded=%s weights=%s stability=%.3f",
                self._check_count,
                decision_type.value,
                excluded,
                {
                    k: f"{v:.3f}"
                    for k, v in weights.items()
                },
                self._get_cluster_stability(),
            )

    def _router_loop(self):
        logger.info(
            "QuorumAwareRouter loop started "
            "interval=%.1fs",
            self.check_interval,
        )
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._check_cycle()
            except Exception as e:
                logger.error(
                    "QuorumAwareRouter check error: %s", e
                )

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._router_loop,
            name="quorum-aware-router",
            daemon=True,
        )
        self._thread.start()
        logger.info("QuorumAwareRouter started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("QuorumAwareRouter stopped")

    # ── Public API ─────────────────────────────────────────

    def get_weights(self) -> dict:
        with self._lock:
            return dict(self._current_weights)

    def route_request(self) -> str:
        """Select a node for a single request."""
        import random
        with self._lock:
            weights = dict(self._current_weights)

        nodes = list(weights.keys())
        w = list(weights.values())
        total = sum(w)
        if total <= 0:
            return random.choice(nodes)
        return random.choices(nodes, weights=w, k=1)[0]

    def get_current_decision(self) -> Optional[dict]:
        with self._lock:
            if self._current_decision:
                return self._current_decision.to_dict()
        return None

    def get_decision_history(
        self, n: int = 10
    ) -> list:
        with self._lock:
            history = list(self._decision_history)
        return [d.to_dict() for d in history[-n:]]

    def get_node_routing_state(
        self, node_id: str
    ) -> str:
        with self._lock:
            state = self._node_routing_states.get(
                node_id, NodeRoutingState.ACTIVE
            )
        return state.value

    def get_weight_stability(self) -> dict:
        return {
            node_id: round(
                self._compute_weight_stability(node_id), 4
            )
            for node_id in self.ALL_NODES
        }

    def status(self) -> dict:
        with self._lock:
            weights = dict(self._current_weights)
            routing_states = {
                n: s.value
                for n, s in self._node_routing_states.items()
            }
            decision = (
                self._current_decision.to_dict()
                if self._current_decision else None
            )

        excluded = [
            n for n, s in routing_states.items()
            if s == "excluded"
        ]
        active = [
            n for n, s in routing_states.items()
            if s in ("active", "reduced")
        ]

        return {
            "running": self._running,
            "check_count": self._check_count,
            "decision_count": self._decision_count,
            "exclusion_events": self._exclusion_events,
            "redistribution_events": (
                self._redistribution_events
            ),
            "emergency_events": self._emergency_events,
            "current_weights": {
                k: round(v, 6) for k, v in weights.items()
            },
            "routing_states": routing_states,
            "excluded_nodes": excluded,
            "active_nodes": active,
            "active_node_count": len(active),
            "weight_stability": self.get_weight_stability(),
            "cluster_stability": round(
                self._get_cluster_stability(), 4
            ),
            "current_decision": decision,
            "uptime_seconds": round(
                time.time() - self._start_time, 1
            ) if self._start_time else 0.0,
        }


if __name__ == "__main__":
    logger.info("Starting QuorumAwareRouter demo")

    from streaming_updater import StreamingCausalUpdater
    from quorum_manager import (
        QuorumManager,
        NodeCapacityState,
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

    router = QuorumAwareRouter(
        updater=updater,
        quorum_manager=quorum,
        check_interval=10.0,
    )
    router.start()

    logger.info(
        "QuorumAwareRouter running. "
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
                "=== ROUTER CYCLE %d ===", cycle
            )

            status = router.status()
            logger.info(
                "Router: checks=%d decisions=%d "
                "exclusions=%d redistributions=%d "
                "stability=%.3f",
                status["check_count"],
                status["decision_count"],
                status["exclusion_events"],
                status["redistribution_events"],
                status["cluster_stability"],
            )

            logger.info("Current weights:")
            for node_id in ["node-1", "node-2", "node-3"]:
                weight = status["current_weights"][node_id]
                rstate = status["routing_states"][node_id]
                stab = status["weight_stability"][node_id]
                logger.info(
                    "  node=%-8s weight=%.4f "
                    "state=%-10s stability=%.3f",
                    node_id, weight, rstate, stab,
                )

            decision = status["current_decision"]
            if decision:
                logger.info(
                    "Current decision: type=%s "
                    "excluded=%s quorum=%s",
                    decision["decision_type"],
                    decision["excluded_nodes"],
                    decision["quorum_state"],
                )

            # Cycle 2: simulate node-1 RECOVERING
            if cycle == 2:
                logger.info(
                    "=== DEMO: Simulating node-1 "
                    "RECOVERING ==="
                )
                quorum.mark_node_recovering("node-1")
                logger.info(
                    "node-1 → RECOVERING "
                    "(excluded from routing)"
                )
                time.sleep(12)

                status2 = router.status()
                logger.info(
                    "After exclusion: weights=%s "
                    "excluded=%s type=%s",
                    {
                        k: f"{v:.4f}"
                        for k, v in
                        status2["current_weights"].items()
                    },
                    status2["excluded_nodes"],
                    status2["current_decision"][
                        "decision_type"
                    ] if status2["current_decision"]
                    else "N/A",
                )

                # Simulate 100 routed requests
                counts = {n: 0 for n in router.ALL_NODES}
                for _ in range(100):
                    node = router.route_request()
                    counts[node] += 1
                logger.info(
                    "100 requests routed: %s "
                    "(node-1 should be 0)",
                    counts,
                )

                # Restore node-1
                time.sleep(5)
                quorum.mark_node_full("node-1")
                logger.info(
                    "node-1 → FULL (restored to routing)"
                )
                time.sleep(12)

                status3 = router.status()
                logger.info(
                    "After restore: weights=%s "
                    "excluded=%s redistributions=%d",
                    {
                        k: f"{v:.4f}"
                        for k, v in
                        status3["current_weights"].items()
                    },
                    status3["excluded_nodes"],
                    status3["redistribution_events"],
                )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        router.stop()
        quorum.stop()
        updater.stop()