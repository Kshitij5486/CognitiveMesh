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
logger = logging.getLogger("cm.healing.router")


class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAUSAL_WEIGHTED = "causal_weighted"
    FAILOVER = "failover"


class NodeRoutingState(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    REROUTED = "rerouted"
    ISOLATED = "isolated"


class RoutingDecision:
    def __init__(
        self,
        decision_id: str,
        source_node: str,
        target_nodes: list,
        strategy: RoutingStrategy,
        reason: str,
        causal_weights: dict,
        timestamp: str,
    ):
        self.decision_id = decision_id
        self.source_node = source_node
        self.target_nodes = target_nodes
        self.strategy = strategy
        self.reason = reason
        self.causal_weights = causal_weights
        self.timestamp = timestamp
        self.active = True
        self.reverted_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "source_node": self.source_node,
            "target_nodes": self.target_nodes,
            "strategy": self.strategy.value,
            "reason": self.reason,
            "causal_weights": {
                k: round(v, 4)
                for k, v in self.causal_weights.items()
            },
            "timestamp": self.timestamp,
            "active": self.active,
            "reverted_at": self.reverted_at,
        }


class QueryRouter:
    ALL_NODES = ["node-1", "node-2", "node-3"]
    MAX_DECISION_HISTORY = 100
    REROUTE_COOLDOWN_SECONDS = 30.0
    DEGRADED_LATENCY_THRESHOLD_MS = 150.0
    RECOVERY_LATENCY_THRESHOLD_MS = 100.0

    def __init__(
        self,
        updater: StreamingCausalUpdater,
        alerter: PredictiveAlerter,
        strategy: RoutingStrategy = RoutingStrategy.CAUSAL_WEIGHTED,
        check_interval: float = 15.0,
    ):
        self.updater = updater
        self.alerter = alerter
        self.strategy = strategy
        self.check_interval = check_interval

        self._node_states: dict[str, NodeRoutingState] = {
            node_id: NodeRoutingState.ACTIVE
            for node_id in self.ALL_NODES
        }
        self._active_decisions: dict[str, RoutingDecision] = {}
        self._decision_history: deque = deque(
            maxlen=self.MAX_DECISION_HISTORY
        )
        self._reroute_cooldowns: dict[str, float] = {}
        self._round_robin_index = 0
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._decision_counter = 0
        self._checks_run = 0
        self._total_reroutes = 0
        self._total_recoveries = 0

    def _make_decision_id(self) -> str:
        self._decision_counter += 1
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        return f"route-{ts}-{self._decision_counter:04d}"

    def _is_in_cooldown(self, node_id: str) -> bool:
        with self._lock:
            last = self._reroute_cooldowns.get(node_id)
            if last is None:
                return False
            return (time.time() - last) < self.REROUTE_COOLDOWN_SECONDS

    def _set_cooldown(self, node_id: str):
        with self._lock:
            self._reroute_cooldowns[node_id] = time.time()

    def _compute_causal_weights(
        self, candidate_nodes: list
    ) -> dict:
        weights = {}
        effects = {}

        for node_id in candidate_nodes:
            snapshot = self.updater.get_current_snapshot(node_id)
            if snapshot:
                effects[node_id] = abs(snapshot["effect"])
            else:
                effects[node_id] = 999.0

        if not effects:
            equal = 1.0 / len(candidate_nodes)
            return {n: equal for n in candidate_nodes}

        max_effect = max(effects.values())
        if max_effect == 0:
            equal = 1.0 / len(candidate_nodes)
            return {n: equal for n in candidate_nodes}

        inv_effects = {
            n: (max_effect - e + 1.0)
            for n, e in effects.items()
        }
        total = sum(inv_effects.values())
        weights = {
            n: v / total
            for n, v in inv_effects.items()
        }
        return weights

    def _select_target_nodes(
        self,
        source_node: str,
        strategy: RoutingStrategy,
    ) -> tuple[list, dict]:
        with self._lock:
            candidates = [
                n for n in self.ALL_NODES
                if n != source_node
                and self._node_states[n] == NodeRoutingState.ACTIVE
            ]

        if not candidates:
            candidates = [
                n for n in self.ALL_NODES
                if n != source_node
            ]

        if not candidates:
            return [], {}

        if strategy == RoutingStrategy.CAUSAL_WEIGHTED:
            weights = self._compute_causal_weights(candidates)
            sorted_nodes = sorted(
                candidates,
                key=lambda n: weights.get(n, 0),
                reverse=True,
            )
            return sorted_nodes, weights

        elif strategy == RoutingStrategy.LEAST_LOADED:
            loads = {}
            buf = self.updater.buffer
            if buf:
                df = buf.get_dataframe()
                if df is not None:
                    for n in candidates:
                        col = f"{n.replace('-', '_')}_active_queries"
                        if col in df.columns:
                            loads[n] = float(df[col].iloc[-1])
                        else:
                            loads[n] = 0.0
            if loads:
                sorted_nodes = sorted(
                    candidates,
                    key=lambda n: loads.get(n, 0),
                )
            else:
                sorted_nodes = candidates
            weights = {n: 1.0 / len(candidates) for n in candidates}
            return sorted_nodes, weights

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            with self._lock:
                idx = self._round_robin_index % len(candidates)
                self._round_robin_index += 1
            sorted_nodes = (
                candidates[idx:] + candidates[:idx]
            )
            weights = {n: 1.0 / len(candidates) for n in candidates}
            return sorted_nodes, weights

        else:
            weights = {n: 1.0 / len(candidates) for n in candidates}
            return candidates, weights

    def reroute_node(
        self,
        node_id: str,
        reason: str,
        strategy: Optional[RoutingStrategy] = None,
    ) -> Optional[RoutingDecision]:
        if self._is_in_cooldown(node_id):
            logger.debug(
                "Reroute cooldown active for node=%s", node_id
            )
            return None

        strategy = strategy or self.strategy
        target_nodes, weights = self._select_target_nodes(
            node_id, strategy
        )

        if not target_nodes:
            logger.warning(
                "No target nodes available for rerouting node=%s",
                node_id,
            )
            return None

        decision = RoutingDecision(
            decision_id=self._make_decision_id(),
            source_node=node_id,
            target_nodes=target_nodes,
            strategy=strategy,
            reason=reason,
            causal_weights=weights,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._node_states[node_id] = NodeRoutingState.REROUTED
            self._active_decisions[decision.decision_id] = decision
            self._decision_history.append(decision)
            self._total_reroutes += 1

        self._set_cooldown(node_id)

        load_pcts = {
            n: round(w * 100, 1)
            for n, w in weights.items()
        }
        logger.warning(
            "REROUTE node=%s → %s strategy=%s "
            "load_distribution=%s reason=%s",
            node_id,
            target_nodes,
            strategy.value,
            load_pcts,
            reason[:60],
        )

        return decision

    def restore_node(self, node_id: str) -> bool:
        with self._lock:
            current_state = self._node_states.get(node_id)
            if current_state == NodeRoutingState.ACTIVE:
                return False

            self._node_states[node_id] = NodeRoutingState.ACTIVE

            to_deactivate = [
                did for did, d in self._active_decisions.items()
                if d.source_node == node_id
            ]
            for did in to_deactivate:
                self._active_decisions[did].active = False
                self._active_decisions[did].reverted_at = (
                    datetime.now(timezone.utc).isoformat()
                )
                del self._active_decisions[did]

            self._total_recoveries += 1

        logger.info(
            "RESTORE node=%s returned to active routing "
            "decisions_reverted=%d",
            node_id,
            len(to_deactivate),
        )
        return True

    def isolate_node(self, node_id: str, reason: str):
        with self._lock:
            self._node_states[node_id] = NodeRoutingState.ISOLATED

        self.reroute_node(
            node_id=node_id,
            reason=f"ISOLATION: {reason}",
            strategy=RoutingStrategy.CAUSAL_WEIGHTED,
        )

        logger.critical(
            "ISOLATE node=%s removed from routing pool reason=%s",
            node_id,
            reason[:60],
        )

    def _check_recovery(self):
        buf = self.updater.buffer
        if not buf:
            return

        df = buf.get_dataframe()
        if df is None:
            return

        with self._lock:
            rerouted = [
                n for n, s in self._node_states.items()
                if s == NodeRoutingState.REROUTED
            ]

        for node_id in rerouted:
            node_safe = node_id.replace("-", "_")
            col_q = f"{node_safe}_active_queries"

            snapshot = self.updater.get_current_snapshot(node_id)
            if not snapshot:
                continue

            causal_effect = abs(snapshot["effect"])
            if col_q not in df.columns:
                continue

            current_load = float(df[col_q].iloc[-1])
            current_latency = causal_effect * current_load

            if current_latency < self.RECOVERY_LATENCY_THRESHOLD_MS:
                logger.info(
                    "Recovery detected node=%s "
                    "latency=%.1fms < threshold=%.1fms",
                    node_id,
                    current_latency,
                    self.RECOVERY_LATENCY_THRESHOLD_MS,
                )
                self.restore_node(node_id)

    def _check_degradation(self):
        active_alerts = self.alerter.get_active_alerts()

        reroute_alerts = [
            a for a in active_alerts
            if a["alert_type"] == AlertType.LATENCY_RISING.value
            and a["severity"] == AlertSeverity.CRITICAL.value
            and not a.get("acknowledged", False)
        ]

        isolate_alerts = [
            a for a in active_alerts
            if a["alert_type"] == AlertType.CAUSAL_THRESHOLD.value
            and a["severity"] in (
                AlertSeverity.CRITICAL.value,
                AlertSeverity.EMERGENCY.value,
            )
            and not a.get("acknowledged", False)
        ]

        for alert in reroute_alerts:
            node_id = alert["node_id"]
            if node_id == "cluster":
                continue
            with self._lock:
                state = self._node_states.get(node_id)
            if state in (
                NodeRoutingState.REROUTED,
                NodeRoutingState.ISOLATED,
            ):
                continue

            self.reroute_node(
                node_id=node_id,
                reason=alert.get("message", "latency critical")[:80],
                strategy=RoutingStrategy.CAUSAL_WEIGHTED,
            )

        for alert in isolate_alerts:
            node_id = alert["node_id"]
            if node_id == "cluster":
                continue
            with self._lock:
                state = self._node_states.get(node_id)
            if state == NodeRoutingState.ISOLATED:
                continue

            self.isolate_node(
                node_id=node_id,
                reason=alert.get("message", "causal threshold")[:80],
            )

    def _check_cycle(self):
        self._checks_run += 1

        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            return

        self._check_degradation()
        self._check_recovery()

        with self._lock:
            active_reroutes = sum(
                1 for s in self._node_states.values()
                if s == NodeRoutingState.REROUTED
            )
            isolated = sum(
                1 for s in self._node_states.values()
                if s == NodeRoutingState.ISOLATED
            )

        if active_reroutes > 0 or isolated > 0:
            logger.info(
                "Router check cycle=%d rerouted=%d isolated=%d "
                "total_reroutes=%d recoveries=%d",
                self._checks_run,
                active_reroutes,
                isolated,
                self._total_reroutes,
                self._total_recoveries,
            )

    def _router_loop(self):
        logger.info(
            "QueryRouter started strategy=%s "
            "check_interval=%.1fs",
            self.strategy.value,
            self.check_interval,
        )
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._check_cycle()
            except Exception as e:
                logger.error("Router check error: %s", e)

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._router_loop,
            name="query-router",
            daemon=True,
        )
        self._thread.start()
        logger.info("QueryRouter started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("QueryRouter stopped")

    def get_routing_table(self) -> dict:
        with self._lock:
            states = dict(self._node_states)
            decisions = {
                did: d.to_dict()
                for did, d in self._active_decisions.items()
            }

        table = {}
        for node_id in self.ALL_NODES:
            state = states.get(node_id, NodeRoutingState.ACTIVE)
            active_decision = None
            for d in decisions.values():
                if d["source_node"] == node_id and d["active"]:
                    active_decision = d
                    break

            table[node_id] = {
                "state": state.value,
                "active_decision": active_decision,
                "is_receiving_traffic": state == NodeRoutingState.ACTIVE,
            }

        return table

    def get_active_decisions(self) -> list:
        with self._lock:
            return [
                d.to_dict()
                for d in self._active_decisions.values()
            ]

    def get_decision_history(self, n: int = 10) -> list:
        with self._lock:
            history = list(self._decision_history)
        return [d.to_dict() for d in history[-n:]]

    def get_active_nodes(self) -> list:
        with self._lock:
            return [
                n for n, s in self._node_states.items()
                if s == NodeRoutingState.ACTIVE
            ]

    def status(self) -> dict:
        with self._lock:
            states = {
                n: s.value
                for n, s in self._node_states.items()
            }
            active_count = sum(
                1 for s in self._node_states.values()
                if s == NodeRoutingState.ACTIVE
            )
            rerouted_count = sum(
                1 for s in self._node_states.values()
                if s == NodeRoutingState.REROUTED
            )
            isolated_count = sum(
                1 for s in self._node_states.values()
                if s == NodeRoutingState.ISOLATED
            )

        return {
            "running": self._running,
            "strategy": self.strategy.value,
            "checks_run": self._checks_run,
            "total_reroutes": self._total_reroutes,
            "total_recoveries": self._total_recoveries,
            "active_decisions": len(self._active_decisions),
            "node_states": states,
            "active_nodes": active_count,
            "rerouted_nodes": rerouted_count,
            "isolated_nodes": isolated_count,
        }


if __name__ == "__main__":
    logger.info("Starting QueryRouter demo")

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

    router = QueryRouter(
        updater=updater,
        alerter=alerter,
        strategy=RoutingStrategy.CAUSAL_WEIGHTED,
        check_interval=15.0,
    )
    router.start()

    logger.info(
        "Full routing stack running. "
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

            logger.info("=== ROUTER CYCLE %d ===", cycle)

            router_status = router.status()
            logger.info(
                "Router: checks=%d reroutes=%d "
                "recoveries=%d active_decisions=%d",
                router_status["checks_run"],
                router_status["total_reroutes"],
                router_status["total_recoveries"],
                router_status["active_decisions"],
            )

            table = router.get_routing_table()
            for node_id, info in table.items():
                logger.info(
                    "  node=%-8s state=%-10s traffic=%s",
                    node_id,
                    info["state"].upper(),
                    "YES" if info["is_receiving_traffic"] else "NO",
                )
                if info["active_decision"]:
                    d = info["active_decision"]
                    logger.info(
                        "    → rerouted to %s "
                        "weights=%s",
                        d["target_nodes"],
                        {
                            k: round(v, 2)
                            for k, v in d["causal_weights"].items()
                        },
                    )

            decisions = router.get_active_decisions()
            if decisions:
                logger.info(
                    "Active routing decisions: %d", len(decisions)
                )
            else:
                logger.info("No active routing decisions")

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        router.stop()
        alerter.stop()
        simulator.stop()
        analyzer.stop()
        updater.stop()