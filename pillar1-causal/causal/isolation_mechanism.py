import logging
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "telemetry"))
sys.path.insert(0, os.path.dirname(__file__))

from byzantine_detector import ByzantineDetector, NodeStatus
from reputation_scorer import ReputationScorer
from consensus_engine import ConsensusEngine

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.byzantine.isolation")


class IsolationReason(Enum):
    BYZANTINE_REPUTATION = "byzantine_reputation"
    REPEATED_CAUSAL_VIOLATIONS = "repeated_causal_violations"
    PROOF_FAILURE_RATE = "proof_failure_rate"
    MANUAL_ISOLATION = "manual_isolation"


class IsolationRecord:
    def __init__(
        self,
        node_id: str,
        reason: IsolationReason,
        reputation_score: float,
        causal_violations: int,
        proof_failures: int,
        isolated_at: str,
    ):
        self.node_id = node_id
        self.reason = reason
        self.reputation_score = reputation_score
        self.causal_violations = causal_violations
        self.proof_failures = proof_failures
        self.isolated_at = isolated_at
        self.released_at: Optional[str] = None
        self.release_reason: Optional[str] = None
        self.duration_seconds: Optional[float] = None

    def release(self, reason: str):
        self.released_at = datetime.now(timezone.utc).isoformat()
        self.release_reason = reason
        isolated_dt = datetime.fromisoformat(self.isolated_at)
        released_dt = datetime.fromisoformat(self.released_at)
        self.duration_seconds = (
            released_dt - isolated_dt
        ).total_seconds()

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "reason": self.reason.value,
            "reputation_score_at_isolation": round(
                self.reputation_score, 4
            ),
            "causal_violations": self.causal_violations,
            "proof_failures": self.proof_failures,
            "isolated_at": self.isolated_at,
            "released_at": self.released_at,
            "release_reason": self.release_reason,
            "duration_seconds": self.duration_seconds,
            "currently_isolated": self.released_at is None,
        }


class IsolationMechanism:
    ISOLATION_SCORE_THRESHOLD = 0.15
    RELEASE_SCORE_THRESHOLD = 0.40
    CHECK_INTERVAL_SECONDS = 15.0
    MIN_HEALTHY_NODES = 2

    def __init__(
        self,
        scorer: ReputationScorer,
        detector: ByzantineDetector,
        engine: ConsensusEngine,
        check_interval: float = 15.0,
    ):
        self.scorer = scorer
        self.detector = detector
        self.engine = engine
        self.check_interval = check_interval

        self._isolated_nodes: dict[str, IsolationRecord] = {}
        self._isolation_history: list[IsolationRecord] = []
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._checks_run = 0
        self._total_isolations = 0
        self._total_releases = 0

    def _count_healthy_nodes(self) -> int:
        with self._lock:
            total = len(["node-1", "node-2", "node-3"])
            isolated = len(self._isolated_nodes)
            return total - isolated

    def _should_isolate(self, node_id: str) -> tuple:
        score = self.scorer.get_score(node_id)
        status = self.scorer.get_status(node_id)
        profile = self.detector.profiles.get(node_id)

        if not profile:
            return False, None

        if score <= self.ISOLATION_SCORE_THRESHOLD:
            if self._count_healthy_nodes() <= self.MIN_HEALTHY_NODES:
                logger.warning(
                    "Would isolate node=%s but cluster needs "
                    "at least %d healthy nodes",
                    node_id,
                    self.MIN_HEALTHY_NODES,
                )
                return False, None

            return True, IsolationReason.BYZANTINE_REPUTATION

        return False, None

    def _should_release(self, node_id: str) -> tuple:
        score = self.scorer.get_score(node_id)
        profile = self.detector.profiles.get(node_id)

        if not profile:
            return False, None

        if score >= self.RELEASE_SCORE_THRESHOLD:
            return True, f"Reputation recovered to {score:.4f}"

        return False, None

    def _isolate_node(self, node_id: str, reason: IsolationReason):
        with self._lock:
            if node_id in self._isolated_nodes:
                return

            profile = self.detector.profiles.get(node_id)
            score = self.scorer.get_score(node_id)

            record = IsolationRecord(
                node_id=node_id,
                reason=reason,
                reputation_score=score,
                causal_violations=profile.causal_violations if profile else 0,
                proof_failures=profile.proof_failures if profile else 0,
                isolated_at=datetime.now(timezone.utc).isoformat(),
            )

            self._isolated_nodes[node_id] = record
            self._isolation_history.append(record)
            self._total_isolations += 1

            logger.critical(
                "NODE ISOLATED node=%s reason=%s score=%.4f "
                "violations=%d proof_failures=%d",
                node_id,
                reason.value,
                score,
                record.causal_violations,
                record.proof_failures,
            )

    def _release_node(self, node_id: str, reason: str):
        with self._lock:
            record = self._isolated_nodes.pop(node_id, None)
            if not record:
                return

            record.release(reason)
            self._total_releases += 1

            logger.info(
                "NODE RELEASED node=%s reason=%s "
                "duration=%.1fs score=%.4f",
                node_id,
                reason,
                record.duration_seconds or 0,
                self.scorer.get_score(node_id),
            )

    def _check_cycle(self):
        self._checks_run += 1

        for node_id in ["node-1", "node-2", "node-3"]:
            with self._lock:
                is_isolated = node_id in self._isolated_nodes

            if is_isolated:
                should_release, release_reason = self._should_release(
                    node_id
                )
                if should_release:
                    self._release_node(node_id, release_reason)
                else:
                    score = self.scorer.get_score(node_id)
                    logger.info(
                        "Node remains isolated node=%s score=%.4f "
                        "needs=%.4f to release",
                        node_id,
                        score,
                        self.RELEASE_SCORE_THRESHOLD,
                    )
            else:
                should_isolate, reason = self._should_isolate(node_id)
                if should_isolate:
                    self._isolate_node(node_id, reason)

        self._log_cluster_state()

    def _log_cluster_state(self):
        with self._lock:
            isolated = list(self._isolated_nodes.keys())

        active = [
            n for n in ["node-1", "node-2", "node-3"]
            if n not in isolated
        ]

        scorer_summary = self.scorer.cluster_summary()
        engine_status = self.engine.status()

        logger.info(
            "CLUSTER STATE active=%s isolated=%s "
            "trusted=%d byzantine=%d checks=%d",
            active,
            isolated,
            scorer_summary["trusted_nodes"],
            scorer_summary["byzantine_nodes"],
            self._checks_run,
        )

        for node_id in ["node-1", "node-2", "node-3"]:
            score = self.scorer.get_score(node_id)
            status = self.scorer.get_status(node_id)
            is_isolated = node_id in (isolated if isolated else [])
            logger.info(
                "  node=%-8s score=%.4f status=%-12s isolated=%s",
                node_id,
                score,
                status.value,
                "YES" if is_isolated else "no",
            )

        logger.info(
            "  consensus can_reach=%s excluded=%s",
            engine_status["can_reach_consensus"],
            engine_status["excluded_nodes"],
        )

    def _isolation_loop(self):
        logger.info(
            "Isolation mechanism started check_interval=%.1fs "
            "isolate_threshold=%.2f release_threshold=%.2f",
            self.check_interval,
            self.ISOLATION_SCORE_THRESHOLD,
            self.RELEASE_SCORE_THRESHOLD,
        )
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._check_cycle()
            except Exception as e:
                logger.error("Isolation check error: %s", e)

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._isolation_loop,
            name="isolation-mechanism",
            daemon=True,
        )
        self._thread.start()
        logger.info("IsolationMechanism started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("IsolationMechanism stopped")

    def manually_isolate(self, node_id: str) -> bool:
        with self._lock:
            if node_id in self._isolated_nodes:
                logger.warning(
                    "Node already isolated node=%s", node_id
                )
                return False
            if self._count_healthy_nodes() <= self.MIN_HEALTHY_NODES:
                logger.error(
                    "Cannot manually isolate node=%s — "
                    "cluster needs at least %d healthy nodes",
                    node_id,
                    self.MIN_HEALTHY_NODES,
                )
                return False

        self._isolate_node(node_id, IsolationReason.MANUAL_ISOLATION)
        return True

    def manually_release(self, node_id: str) -> bool:
        with self._lock:
            if node_id not in self._isolated_nodes:
                logger.warning(
                    "Node not isolated node=%s", node_id
                )
                return False

        self._release_node(node_id, "Manual release by operator")
        return True

    def is_node_active(self, node_id: str) -> bool:
        with self._lock:
            return node_id not in self._isolated_nodes

    def get_active_nodes(self) -> list:
        with self._lock:
            return [
                n for n in ["node-1", "node-2", "node-3"]
                if n not in self._isolated_nodes
            ]

    def get_isolated_nodes(self) -> list:
        with self._lock:
            return list(self._isolated_nodes.keys())

    def cluster_status(self) -> dict:
        with self._lock:
            isolated_records = {
                node_id: record.to_dict()
                for node_id, record in self._isolated_nodes.items()
            }
            history = [
                r.to_dict()
                for r in self._isolation_history[-10:]
            ]

        active = self.get_active_nodes()
        scorer_summary = self.scorer.cluster_summary()

        return {
            "active_nodes": active,
            "isolated_nodes": list(isolated_records.keys()),
            "isolation_records": isolated_records,
            "cluster_operational": len(active) >= self.MIN_HEALTHY_NODES,
            "total_isolations": self._total_isolations,
            "total_releases": self._total_releases,
            "checks_run": self._checks_run,
            "reputation_summary": scorer_summary,
            "isolation_history": history,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def status(self) -> dict:
        return {
            "running": self._running,
            "checks_run": self._checks_run,
            "total_isolations": self._total_isolations,
            "total_releases": self._total_releases,
            "active_nodes": self.get_active_nodes(),
            "isolated_nodes": self.get_isolated_nodes(),
            "cluster_operational": (
                len(self.get_active_nodes()) >= self.MIN_HEALTHY_NODES
            ),
        }


if __name__ == "__main__":
    from zk_proof_bridge import ZKProofBridge
    from streaming_updater import StreamingCausalUpdater

    logger.info("Starting isolation mechanism demo")

    updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )
    updater.start()

    bridge = ZKProofBridge()

    detector = ByzantineDetector(
        updater=updater,
        bridge=bridge,
        check_interval=15.0,
    )
    detector.start()

    scorer = ReputationScorer(
        detector=detector,
        update_interval_seconds=15.0,
        decay_interval_seconds=60.0,
    )
    scorer.start()

    engine = ConsensusEngine(
        scorer=scorer,
        detector=detector,
        updater=updater,
    )

    isolation = IsolationMechanism(
        scorer=scorer,
        detector=detector,
        engine=engine,
        check_interval=15.0,
    )
    isolation.start()

    logger.info(
        "Full Byzantine consensus stack running. "
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
                    "Causal engine not ready buffer=%d/30",
                    engine_status.get("buffer_size", 0),
                )
                continue

            logger.info("=== FULL STACK CYCLE %d ===", cycle)

            status = isolation.cluster_status()
            logger.info(
                "Isolation: active=%s isolated=%s operational=%s "
                "total_isolations=%d",
                status["active_nodes"],
                status["isolated_nodes"],
                status["cluster_operational"],
                status["total_isolations"],
            )

            result = engine.decide_cluster_causal_effect()
            if result:
                logger.info(
                    "Consensus: decided=%.4fms participating=%s "
                    "excluded=%s weight=%.4f",
                    result.decided_value,
                    result.participating_nodes,
                    result.excluded_nodes,
                    result.total_weight,
                )
                for vote in result.votes:
                    logger.info(
                        "  vote node=%-8s value=%.4fms weight=%.4f",
                        vote.node_id,
                        vote.voted_value,
                        vote.reputation_weight,
                    )

            rep_summary = scorer.cluster_summary()
            logger.info(
                "Reputation: trusted=%d suspicious=%d "
                "byzantine=%d avg=%.4f",
                rep_summary["trusted_nodes"],
                rep_summary["suspicious_nodes"],
                rep_summary["byzantine_nodes"],
                rep_summary["average_score"],
            )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        isolation.stop()
        scorer.stop()
        detector.stop()
        updater.stop()