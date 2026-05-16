import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "telemetry"))
sys.path.insert(0, os.path.dirname(__file__))

from byzantine_detector import ByzantineDetector, NodeStatus
from reputation_scorer import ReputationScorer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.byzantine.consensus")


class ConsensusProposal:
    def __init__(
        self,
        proposal_id: str,
        proposal_type: str,
        proposed_value: float,
        proposing_node: str,
        context: dict,
    ):
        self.proposal_id = proposal_id
        self.proposal_type = proposal_type
        self.proposed_value = proposed_value
        self.proposing_node = proposing_node
        self.context = context
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.votes: dict = {}
        self.decided = False
        self.decision: Optional[float] = None
        self.decided_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "proposed_value": self.proposed_value,
            "proposing_node": self.proposing_node,
            "context": self.context,
            "created_at": self.created_at,
            "votes": self.votes,
            "decided": self.decided,
            "decision": self.decision,
            "decided_at": self.decided_at,
        }


class ConsensusVote:
    def __init__(
        self,
        node_id: str,
        proposal_id: str,
        voted_value: float,
        reputation_weight: float,
        timestamp: str,
    ):
        self.node_id = node_id
        self.proposal_id = proposal_id
        self.voted_value = voted_value
        self.reputation_weight = reputation_weight
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "proposal_id": self.proposal_id,
            "voted_value": round(self.voted_value, 4),
            "reputation_weight": round(self.reputation_weight, 4),
            "timestamp": self.timestamp,
        }


class ConsensusResult:
    def __init__(
        self,
        proposal_id: str,
        proposal_type: str,
        decided_value: float,
        votes: list,
        total_weight: float,
        participating_nodes: list,
        excluded_nodes: list,
        consensus_reached: bool,
        timestamp: str,
    ):
        self.proposal_id = proposal_id
        self.proposal_type = proposal_type
        self.decided_value = decided_value
        self.votes = votes
        self.total_weight = total_weight
        self.participating_nodes = participating_nodes
        self.excluded_nodes = excluded_nodes
        self.consensus_reached = consensus_reached
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "decided_value": round(self.decided_value, 4),
            "votes": [v.to_dict() for v in self.votes],
            "total_weight": round(self.total_weight, 4),
            "participating_nodes": self.participating_nodes,
            "excluded_nodes": self.excluded_nodes,
            "consensus_reached": self.consensus_reached,
            "timestamp": self.timestamp,
        }


class ConsensusEngine:
    MIN_PARTICIPATION_WEIGHT = 0.1
    BYZANTINE_EXCLUSION_THRESHOLD = 0.15
    MIN_NODES_FOR_CONSENSUS = 2

    def __init__(
        self,
        scorer: ReputationScorer,
        detector: ByzantineDetector,
        updater,
    ):
        self.scorer = scorer
        self.detector = detector
        self.updater = updater
        self._lock = threading.RLock()
        self._proposal_counter = 0
        self._consensus_history: list = []
        self._total_decisions = 0
        self._byzantine_exclusions = 0
        logger.info("ConsensusEngine initialized")

    def _get_eligible_nodes(self) -> tuple:
        participating = []
        excluded = []

        for node_id in ["node-1", "node-2", "node-3"]:
            score = self.scorer.get_score(node_id)
            status = self.scorer.get_status(node_id)

            if score <= self.BYZANTINE_EXCLUSION_THRESHOLD:
                excluded.append(node_id)
                logger.warning(
                    "Node excluded from consensus node=%s "
                    "score=%.4f status=%s",
                    node_id, score, status.value,
                )
            else:
                participating.append((node_id, score))

        return participating, excluded

    def _weighted_average(self, votes: list) -> tuple:
        if not votes:
            return 0.0, 0.0

        total_weight = sum(v.reputation_weight for v in votes)
        if total_weight < 0.001:
            return 0.0, 0.0

        weighted_sum = sum(
            v.voted_value * v.reputation_weight for v in votes
        )
        return weighted_sum / total_weight, total_weight

    def decide_cluster_causal_effect(self) -> Optional[ConsensusResult]:
        with self._lock:
            engine_status = self.updater.status()
            if not engine_status.get("is_ready"):
                logger.warning(
                    "Cannot reach consensus — causal engine not ready"
                )
                return None

            participating, excluded = self._get_eligible_nodes()

            if len(participating) < self.MIN_NODES_FOR_CONSENSUS:
                logger.error(
                    "Cannot reach consensus — insufficient nodes "
                    "participating=%d excluded=%d",
                    len(participating),
                    len(excluded),
                )
                return None

            self._proposal_counter += 1
            proposal_id = (
                f"consensus-causal-{self._proposal_counter:04d}-"
                f"{datetime.now(timezone.utc).strftime('%H%M%S')}"
            )

            votes = []
            for node_id, score in participating:
                snapshot = self.updater.get_current_snapshot(node_id)
                if not snapshot:
                    continue

                effect = abs(snapshot["effect"])
                vote = ConsensusVote(
                    node_id=node_id,
                    proposal_id=proposal_id,
                    voted_value=effect,
                    reputation_weight=score,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                votes.append(vote)

                logger.info(
                    "Vote cast node=%-8s effect=%.4fms weight=%.4f",
                    node_id, effect, score,
                )

            if not votes:
                logger.error("No votes collected")
                return None

            decided_value, total_weight = self._weighted_average(votes)
            consensus_reached = len(votes) >= self.MIN_NODES_FOR_CONSENSUS

            if excluded:
                self._byzantine_exclusions += len(excluded)

            self._total_decisions += 1

            result = ConsensusResult(
                proposal_id=proposal_id,
                proposal_type="cluster_causal_effect",
                decided_value=decided_value,
                votes=votes,
                total_weight=total_weight,
                participating_nodes=[n for n, _ in participating],
                excluded_nodes=excluded,
                consensus_reached=consensus_reached,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            self._consensus_history.append(result)
            if len(self._consensus_history) > 50:
                self._consensus_history.pop(0)

            logger.info(
                "Consensus reached proposal=%s decided=%.4fms "
                "weight=%.4f nodes=%d excluded=%d",
                proposal_id,
                decided_value,
                total_weight,
                len(votes),
                len(excluded),
            )

            return result

    def decide_cluster_threshold(
        self,
        metric: str = "avg_latency_ms",
        action: str = "alert",
    ) -> Optional[ConsensusResult]:
        with self._lock:
            participating, excluded = self._get_eligible_nodes()

            if len(participating) < self.MIN_NODES_FOR_CONSENSUS:
                return None

            self._proposal_counter += 1
            proposal_id = (
                f"consensus-threshold-{self._proposal_counter:04d}-"
                f"{datetime.now(timezone.utc).strftime('%H%M%S')}"
            )

            votes = []
            for node_id, score in participating:
                snapshot = self.updater.get_current_snapshot(node_id)
                if not snapshot:
                    continue

                effect = abs(snapshot["effect"])
                threshold_vote = effect * 1.5

                vote = ConsensusVote(
                    node_id=node_id,
                    proposal_id=proposal_id,
                    voted_value=threshold_vote,
                    reputation_weight=score,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                votes.append(vote)

            if not votes:
                return None

            decided_value, total_weight = self._weighted_average(votes)

            result = ConsensusResult(
                proposal_id=proposal_id,
                proposal_type=f"threshold_{metric}_{action}",
                decided_value=decided_value,
                votes=votes,
                total_weight=total_weight,
                participating_nodes=[n for n, _ in participating],
                excluded_nodes=excluded,
                consensus_reached=len(votes) >= self.MIN_NODES_FOR_CONSENSUS,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            self._consensus_history.append(result)
            self._total_decisions += 1

            logger.info(
                "Threshold consensus proposal=%s metric=%s "
                "threshold=%.4f nodes=%d excluded=%d",
                proposal_id, metric, decided_value,
                len(votes), len(excluded),
            )

            return result

    def get_recent_decisions(self, n: int = 5) -> list:
        with self._lock:
            return [
                r.to_dict()
                for r in self._consensus_history[-n:]
            ]

    def status(self) -> dict:
        participating, excluded = self._get_eligible_nodes()
        return {
            "total_decisions": self._total_decisions,
            "byzantine_exclusions": self._byzantine_exclusions,
            "participating_nodes": [n for n, _ in participating],
            "excluded_nodes": excluded,
            "can_reach_consensus": (
                len(participating) >= self.MIN_NODES_FOR_CONSENSUS
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


if __name__ == "__main__":
    from zk_proof_bridge import ZKProofBridge
    from streaming_updater import StreamingCausalUpdater

    logger.info("Starting consensus engine demo")

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

    logger.info(
        "Waiting for causal engine to be ready "
        "and reputation to build up..."
    )

    try:
        cycle = 0
        while True:
            time.sleep(30)
            cycle += 1

            engine_status = updater.status()
            if not engine_status.get("is_ready"):
                logger.info(
                    "Causal engine not ready yet buffer=%d/30",
                    engine_status.get("buffer_size", 0),
                )
                continue

            logger.info(
                "=== CONSENSUS CYCLE %d ===", cycle
            )

            causal_result = engine.decide_cluster_causal_effect()
            if causal_result:
                logger.info(
                    "Causal consensus: decided=%.4fms "
                    "participating=%s excluded=%s weight=%.4f",
                    causal_result.decided_value,
                    causal_result.participating_nodes,
                    causal_result.excluded_nodes,
                    causal_result.total_weight,
                )
                for vote in causal_result.votes:
                    logger.info(
                        "  vote node=%-8s value=%.4fms weight=%.4f",
                        vote.node_id,
                        vote.voted_value,
                        vote.reputation_weight,
                    )

            threshold_result = engine.decide_cluster_threshold()
            if threshold_result:
                logger.info(
                    "Threshold consensus: alert_at=%.4fms "
                    "nodes=%s excluded=%s",
                    threshold_result.decided_value,
                    threshold_result.participating_nodes,
                    threshold_result.excluded_nodes,
                )

            engine_status_summary = engine.status()
            logger.info(
                "Engine status: decisions=%d exclusions=%d "
                "can_consensus=%s",
                engine_status_summary["total_decisions"],
                engine_status_summary["byzantine_exclusions"],
                engine_status_summary["can_reach_consensus"],
            )

            scorer_summary = scorer.cluster_summary()
            logger.info(
                "Cluster: trusted=%d suspicious=%d byzantine=%d "
                "avg_score=%.4f",
                scorer_summary["trusted_nodes"],
                scorer_summary["suspicious_nodes"],
                scorer_summary["byzantine_nodes"],
                scorer_summary["average_score"],
            )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        scorer.stop()
        detector.stop()
        updater.stop()