import logging
import math
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

from byzantine_detector import (
    ByzantineDetector, NodeStatus, ByzantineEvidence, NodeByzantineProfile
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.byzantine.reputation")


class ReputationEvent:
    def __init__(
        self,
        node_id: str,
        event_type: str,
        delta: float,
        reason: str,
        timestamp: str,
    ):
        self.node_id = node_id
        self.event_type = event_type
        self.delta = delta
        self.reason = reason
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "event_type": self.event_type,
            "delta": round(self.delta, 4),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class NodeReputation:
    INITIAL_SCORE = 0.5
    MAX_SCORE = 1.0
    MIN_SCORE = 0.0

    PROOF_SUCCESS_REWARD = 0.05
    PROOF_FAILURE_PENALTY = 0.15
    CAUSAL_VIOLATION_PENALTY = 0.20
    HONEST_CYCLE_REWARD = 0.02
    DECAY_RATE = 0.005

    SUSPICIOUS_THRESHOLD = 0.35
    BYZANTINE_THRESHOLD = 0.15

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.score = self.INITIAL_SCORE
        self.history: deque = deque(maxlen=100)
        self.events: deque = deque(maxlen=50)
        self._lock = threading.RLock()
        self.total_updates = 0
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.last_updated: Optional[str] = None

    def _apply_delta(self, delta: float, reason: str, event_type: str):
        old_score = self.score
        self.score = max(
            self.MIN_SCORE,
            min(self.MAX_SCORE, self.score + delta)
        )
        self.history.append(self.score)
        self.total_updates += 1
        self.last_updated = datetime.now(timezone.utc).isoformat()

        event = ReputationEvent(
            node_id=self.node_id,
            event_type=event_type,
            delta=delta,
            reason=reason,
            timestamp=self.last_updated,
        )
        self.events.append(event)

        logger.debug(
            "Reputation update node=%s %s delta=%+.4f "
            "old=%.4f new=%.4f",
            self.node_id, event_type, delta, old_score, self.score
        )

    def reward_proof_success(self):
        with self._lock:
            self._apply_delta(
                self.PROOF_SUCCESS_REWARD,
                "ZK proof generated and verified successfully",
                "proof_success",
            )

    def penalize_proof_failure(self):
        with self._lock:
            self._apply_delta(
                -self.PROOF_FAILURE_PENALTY,
                "ZK proof generation or verification failed",
                "proof_failure",
            )

    def penalize_causal_violation(self, deviation_pct: float):
        with self._lock:
            penalty = self.CAUSAL_VIOLATION_PENALTY * min(
                abs(deviation_pct) / 100.0, 2.0
            )
            self._apply_delta(
                -penalty,
                f"Causal effect deviation of {deviation_pct:+.1f}%",
                "causal_violation",
            )

    def reward_honest_cycle(self):
        with self._lock:
            self._apply_delta(
                self.HONEST_CYCLE_REWARD,
                "Honest detection cycle — no violations",
                "honest_cycle",
            )

    def apply_time_decay(self):
        with self._lock:
            if self.score > self.INITIAL_SCORE:
                decay = self.DECAY_RATE
                self._apply_delta(
                    -decay,
                    "Time decay toward baseline",
                    "time_decay",
                )

    def get_status(self) -> NodeStatus:
        with self._lock:
            if self.score >= self.SUSPICIOUS_THRESHOLD + 0.01:
                return NodeStatus.TRUSTED
            elif self.score >= self.BYZANTINE_THRESHOLD + 0.01:
                return NodeStatus.SUSPICIOUS
            else:
                return NodeStatus.BYZANTINE

    def get_score(self) -> float:
        with self._lock:
            return round(self.score, 4)

    def get_trend(self) -> str:
        with self._lock:
            if len(self.history) < 3:
                return "stable"
            recent = list(self.history)[-3:]
            if recent[-1] > recent[0] + 0.01:
                return "improving"
            elif recent[-1] < recent[0] - 0.01:
                return "degrading"
            return "stable"

    def to_dict(self) -> dict:
        with self._lock:
            recent_events = [
                e.to_dict() for e in list(self.events)[-5:]
            ]
            return {
                "node_id": self.node_id,
                "score": self.get_score(),
                "status": self.get_status().value,
                "trend": self.get_trend(),
                "total_updates": self.total_updates,
                "created_at": self.created_at,
                "last_updated": self.last_updated,
                "recent_events": recent_events,
            }


class ReputationScorer:
    def __init__(
        self,
        detector: ByzantineDetector,
        update_interval_seconds: float = 15.0,
        decay_interval_seconds: float = 60.0,
    ):
        self.detector = detector
        self.update_interval = update_interval_seconds
        self.decay_interval = decay_interval_seconds

        self.reputations: dict[str, NodeReputation] = {
            "node-1": NodeReputation("node-1"),
            "node-2": NodeReputation("node-2"),
            "node-3": NodeReputation("node-3"),
        }

        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._decay_thread: Optional[threading.Thread] = None
        self._update_count = 0
        self._last_detector_checks = 0

    def _sync_from_detector(self):
        health = self.detector.get_cluster_health()
        current_checks = health["checks_run"]

        if current_checks == self._last_detector_checks:
            return

        self._last_detector_checks = current_checks
        self._update_count += 1

        for node_id, profile_data in health["node_profiles"].items():
            if node_id not in self.reputations:
                continue

            reputation = self.reputations[node_id]
            proof_ok = profile_data["proof_successes"] > 0
            violations = profile_data["causal_violations"]

            recent_evidence = profile_data.get("recent_evidence", [])
            new_violations = [
                e for e in recent_evidence
                if e.get("severity") == "CRITICAL"
            ]

            if new_violations:
                for ev in new_violations:
                    deviation = ev.get("deviation_pct", 60.0)
                    reputation.penalize_causal_violation(deviation)
                    logger.warning(
                        "Reputation penalty node=%s "
                        "causal_violation deviation=%.1f%%",
                        node_id,
                        deviation,
                    )
            elif violations == 0 and proof_ok:
                reputation.reward_honest_cycle()

            if profile_data["proof_successes"] > 0:
                reputation.reward_proof_success()
            if profile_data["proof_failures"] > 0:
                reputation.penalize_proof_failure()

            status = reputation.get_status()
            score = reputation.get_score()
            trend = reputation.get_trend()

            logger.info(
                "Reputation update node=%-8s score=%.4f "
                "status=%-12s trend=%s",
                node_id, score, status.value, trend,
            )

    def _update_loop(self):
        logger.info(
            "Reputation update loop started interval=%.1fs",
            self.update_interval,
        )
        while self._running:
            time.sleep(self.update_interval)
            try:
                self._sync_from_detector()
            except Exception as e:
                logger.error("Reputation update error: %s", e)

    def _decay_loop(self):
        logger.info(
            "Reputation decay loop started interval=%.1fs",
            self.decay_interval,
        )
        while self._running:
            time.sleep(self.decay_interval)
            try:
                for node_id, reputation in self.reputations.items():
                    reputation.apply_time_decay()
                    logger.debug(
                        "Time decay applied node=%s score=%.4f",
                        node_id,
                        reputation.get_score(),
                    )
            except Exception as e:
                logger.error("Decay loop error: %s", e)

    def start(self):
        self._running = True
        self._update_thread = threading.Thread(
            target=self._update_loop,
            name="reputation-updater",
            daemon=True,
        )
        self._decay_thread = threading.Thread(
            target=self._decay_loop,
            name="reputation-decayer",
            daemon=True,
        )
        self._update_thread.start()
        self._decay_thread.start()
        logger.info("ReputationScorer started")

    def stop(self):
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=10)
        if self._decay_thread:
            self._decay_thread.join(timeout=10)
        logger.info("ReputationScorer stopped")

    def get_score(self, node_id: str) -> float:
        reputation = self.reputations.get(node_id)
        return reputation.get_score() if reputation else 0.0

    def get_status(self, node_id: str) -> NodeStatus:
        reputation = self.reputations.get(node_id)
        return reputation.get_status() if reputation else NodeStatus.UNKNOWN

    def get_all_scores(self) -> dict:
        return {
            node_id: {
                "score": rep.get_score(),
                "status": rep.get_status().value,
                "trend": rep.get_trend(),
            }
            for node_id, rep in self.reputations.items()
        }

    def get_trusted_nodes(self, min_score: float = 0.35) -> list:
        return [
            node_id
            for node_id, rep in self.reputations.items()
            if rep.get_score() >= min_score
        ]

    def cluster_summary(self) -> dict:
        scores = self.get_all_scores()
        trusted = sum(
            1 for s in scores.values()
            if s["status"] == NodeStatus.TRUSTED.value
        )
        suspicious = sum(
            1 for s in scores.values()
            if s["status"] == NodeStatus.SUSPICIOUS.value
        )
        byzantine = sum(
            1 for s in scores.values()
            if s["status"] == NodeStatus.BYZANTINE.value
        )
        avg_score = sum(
            s["score"] for s in scores.values()
        ) / len(scores) if scores else 0.0

        return {
            "cluster_healthy": byzantine == 0,
            "trusted_nodes": trusted,
            "suspicious_nodes": suspicious,
            "byzantine_nodes": byzantine,
            "average_score": round(avg_score, 4),
            "node_scores": scores,
            "update_count": self._update_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def status(self) -> dict:
        return {
            "running": self._running,
            "update_count": self._update_count,
            "trusted_nodes": self.get_trusted_nodes(),
            "cluster_summary": self.cluster_summary(),
        }


if __name__ == "__main__":
    from zk_proof_bridge import ZKProofBridge
    from streaming_updater import StreamingCausalUpdater

    logger.info("Starting full reputation scoring system")

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

    logger.info(
        "System running. Load generator in another terminal. "
        "Reputation updates every 15s."
    )

    try:
        cycle = 0
        while True:
            time.sleep(30)
            cycle += 1
            summary = scorer.cluster_summary()
            logger.info(
                "REPUTATION SUMMARY cycle=%d "
                "trusted=%d suspicious=%d byzantine=%d avg=%.4f",
                cycle,
                summary["trusted_nodes"],
                summary["suspicious_nodes"],
                summary["byzantine_nodes"],
                summary["average_score"],
            )
            for node_id, data in summary["node_scores"].items():
                logger.info(
                    "  node=%-8s score=%.4f status=%-12s trend=%s",
                    node_id,
                    data["score"],
                    data["status"],
                    data["trend"],
                )
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        scorer.stop()
        detector.stop()
        updater.stop()