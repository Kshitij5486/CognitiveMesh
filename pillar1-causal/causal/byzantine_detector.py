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

from zk_proof_bridge import ZKProofBridge, CausalProofRequest
from streaming_updater import StreamingCausalUpdater

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.byzantine.detector")


class NodeStatus(Enum):
    TRUSTED = "trusted"
    SUSPICIOUS = "suspicious"
    BYZANTINE = "byzantine"
    UNKNOWN = "unknown"


class ByzantineEvidence:
    def __init__(
        self,
        node_id: str,
        evidence_type: str,
        expected_value: float,
        observed_value: float,
        deviation_pct: float,
        severity: str,
        timestamp: str,
    ):
        self.node_id = node_id
        self.evidence_type = evidence_type
        self.expected_value = expected_value
        self.observed_value = observed_value
        self.deviation_pct = deviation_pct
        self.severity = severity
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "evidence_type": self.evidence_type,
            "expected_value": round(self.expected_value, 4),
            "observed_value": round(self.observed_value, 4),
            "deviation_pct": round(self.deviation_pct, 2),
            "severity": self.severity,
            "timestamp": self.timestamp,
            "description": (
                f"{self.node_id} reported {self.evidence_type} "
                f"of {self.observed_value:.2f} but causal model "
                f"predicts {self.expected_value:.2f} "
                f"(deviation: {self.deviation_pct:+.1f}%). "
                f"Severity: {self.severity}."
            ),
        }


class NodeByzantineProfile:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.status = NodeStatus.UNKNOWN
        self.evidence_history: deque = deque(maxlen=50)
        self.proof_failures = 0
        self.proof_successes = 0
        self.causal_violations = 0
        self.checks_run = 0
        self.last_check: Optional[str] = None
        self.flagged_at: Optional[str] = None
        self._lock = threading.RLock()

    def add_evidence(self, evidence: ByzantineEvidence):
        with self._lock:
            self.evidence_history.append(evidence)
            if evidence.severity == "CRITICAL":
                self.causal_violations += 1
            self._update_status()

    def record_proof_result(self, success: bool):
        with self._lock:
            if success:
                self.proof_successes += 1
            else:
                self.proof_failures += 1
            self._update_status()

    def record_check(self):
        with self._lock:
            self.checks_run += 1
            self.last_check = datetime.now(timezone.utc).isoformat()

    def _update_status(self):
        total_proofs = self.proof_successes + self.proof_failures
        failure_rate = (
            self.proof_failures / total_proofs
            if total_proofs > 0 else 0.0
        )

        if self.causal_violations >= 3 or failure_rate >= 0.5:
            if self.status != NodeStatus.BYZANTINE:
                self.status = NodeStatus.BYZANTINE
                self.flagged_at = datetime.now(timezone.utc).isoformat()
                logger.critical(
                    "NODE FLAGGED AS BYZANTINE node=%s "
                    "causal_violations=%d proof_failure_rate=%.2f",
                    self.node_id,
                    self.causal_violations,
                    failure_rate,
                )
        elif self.causal_violations >= 1 or failure_rate >= 0.25:
            self.status = NodeStatus.SUSPICIOUS
        elif self.proof_successes > 0 and self.causal_violations == 0:
            self.status = NodeStatus.TRUSTED
        else:
            self.status = NodeStatus.UNKNOWN

    def trust_score(self) -> float:
        with self._lock:
            total_proofs = self.proof_successes + self.proof_failures
            if total_proofs == 0:
                return 0.5

            proof_score = self.proof_successes / total_proofs
            violation_penalty = min(self.causal_violations * 0.2, 0.8)
            score = max(0.0, proof_score - violation_penalty)
            return round(score, 4)

    def to_dict(self) -> dict:
        with self._lock:
            recent_evidence = [
                e.to_dict() for e in list(self.evidence_history)[-5:]
            ]
            return {
                "node_id": self.node_id,
                "status": self.status.value,
                "trust_score": self.trust_score(),
                "proof_successes": self.proof_successes,
                "proof_failures": self.proof_failures,
                "causal_violations": self.causal_violations,
                "checks_run": self.checks_run,
                "last_check": self.last_check,
                "flagged_at": self.flagged_at,
                "recent_evidence": recent_evidence,
            }


class ByzantineDetector:
    DEVIATION_WARN_THRESHOLD = 0.30
    DEVIATION_CRITICAL_THRESHOLD = 0.60
    CHECK_INTERVAL_SECONDS = 15.0

    def __init__(
        self,
        updater: StreamingCausalUpdater,
        bridge: ZKProofBridge,
        check_interval: float = 15.0,
    ):
        self.updater = updater
        self.bridge = bridge
        self.check_interval = check_interval
        self.profiles: dict[str, NodeByzantineProfile] = {
            "node-1": NodeByzantineProfile("node-1"),
            "node-2": NodeByzantineProfile("node-2"),
            "node-3": NodeByzantineProfile("node-3"),
        }
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._checks_run = 0
        self._byzantine_detections = 0

    def _get_cluster_baseline(self) -> Optional[float]:
        effects = []
        for node_id in ["node-1", "node-2", "node-3"]:
            snapshot = self.updater.get_current_snapshot(node_id)
            if snapshot:
                effects.append(abs(snapshot["effect"]))
        if len(effects) < 2:
            return None
        effects.sort()
        return effects[len(effects) // 2]

    def _check_node(
        self,
        node_id: str,
        cluster_baseline: float,
    ) -> Optional[ByzantineEvidence]:
        snapshot = self.updater.get_current_snapshot(node_id)
        if not snapshot:
            return None

        node_effect = abs(snapshot["effect"])

        if cluster_baseline < 0.001:
            return None

        deviation = (node_effect - cluster_baseline) / cluster_baseline

        if abs(deviation) >= self.DEVIATION_CRITICAL_THRESHOLD:
            severity = "CRITICAL"
        elif abs(deviation) >= self.DEVIATION_WARN_THRESHOLD:
            severity = "WARNING"
        else:
            return None

        return ByzantineEvidence(
            node_id=node_id,
            evidence_type="causal_effect_deviation",
            expected_value=cluster_baseline,
            observed_value=node_effect,
            deviation_pct=deviation * 100,
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _verify_node_proof(self, node_id: str) -> bool:
        snapshot = self.updater.get_current_snapshot(node_id)
        if not snapshot:
            return False

        engine_status = self.updater.status()
        avg_latency_ms = int(abs(snapshot["effect"]) * 100)
        active_queries = 1
        mac = avg_latency_ms * active_queries

        try:
            request = CausalProofRequest(
                node_id=node_id,
                active_queries=active_queries,
                avg_latency_ms=float(avg_latency_ms),
                causal_effect_ms=float(mac),
                samples_used=snapshot["samples_used"],
                retrain_cycle=engine_status.get("retrain_count", 1),
            )
            result = self.bridge.prove(request)
            return result.success and result.verified
        except Exception as e:
            logger.error(
                "Proof verification failed node=%s error=%s",
                node_id, e
            )
            return False

    def _detection_cycle(self):
        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            return

        self._checks_run += 1
        cluster_baseline = self._get_cluster_baseline()

        if cluster_baseline is None:
            return

        logger.info(
            "Byzantine detection cycle=%d baseline=%.4fms",
            self._checks_run,
            cluster_baseline,
        )

        for node_id, profile in self.profiles.items():
            profile.record_check()

            evidence = self._check_node(node_id, cluster_baseline)
            if evidence:
                profile.add_evidence(evidence)
                if evidence.severity == "CRITICAL":
                    self._byzantine_detections += 1
                    logger.critical(
                        "BYZANTINE EVIDENCE node=%s deviation=%+.1f%% "
                        "expected=%.2f observed=%.2f",
                        node_id,
                        evidence.deviation_pct,
                        evidence.expected_value,
                        evidence.observed_value,
                    )
                else:
                    logger.warning(
                        "Suspicious behavior node=%s deviation=%+.1f%%",
                        node_id,
                        evidence.deviation_pct,
                    )

            proof_ok = self._verify_node_proof(node_id)
            profile.record_proof_result(proof_ok)

            logger.info(
                "  node=%-8s status=%-12s trust=%.2f "
                "proof=%s violations=%d",
                node_id,
                profile.status.value,
                profile.trust_score(),
                "OK" if proof_ok else "FAIL",
                profile.causal_violations,
            )

    def _detection_loop(self):
        logger.info(
            "Byzantine detector started check_interval=%.1fs "
            "warn_threshold=%.0f%% critical_threshold=%.0f%%",
            self.check_interval,
            self.DEVIATION_WARN_THRESHOLD * 100,
            self.DEVIATION_CRITICAL_THRESHOLD * 100,
        )
        while self._running:
            time.sleep(self.check_interval)
            try:
                self._detection_cycle()
            except Exception as e:
                logger.error("Detection cycle error: %s", e)

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._detection_loop,
            name="byzantine-detector",
            daemon=True,
        )
        self._thread.start()
        logger.info("ByzantineDetector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("ByzantineDetector stopped")

    def get_cluster_health(self) -> dict:
        profiles = {
            node_id: profile.to_dict()
            for node_id, profile in self.profiles.items()
        }
        trusted = sum(
            1 for p in self.profiles.values()
            if p.status == NodeStatus.TRUSTED
        )
        suspicious = sum(
            1 for p in self.profiles.values()
            if p.status == NodeStatus.SUSPICIOUS
        )
        byzantine = sum(
            1 for p in self.profiles.values()
            if p.status == NodeStatus.BYZANTINE
        )
        return {
            "cluster_healthy": byzantine == 0,
            "trusted_nodes": trusted,
            "suspicious_nodes": suspicious,
            "byzantine_nodes": byzantine,
            "checks_run": self._checks_run,
            "byzantine_detections": self._byzantine_detections,
            "node_profiles": profiles,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def is_node_trusted(self, node_id: str) -> bool:
        profile = self.profiles.get(node_id)
        if not profile:
            return False
        return profile.status in (
            NodeStatus.TRUSTED, NodeStatus.UNKNOWN
        )

    def get_trusted_nodes(self) -> list:
        return [
            node_id for node_id, profile in self.profiles.items()
            if self.is_node_trusted(node_id)
        ]

    def status(self) -> dict:
        return {
            "running": self._running,
            "checks_run": self._checks_run,
            "byzantine_detections": self._byzantine_detections,
            "warn_threshold_pct": self.DEVIATION_WARN_THRESHOLD * 100,
            "critical_threshold_pct": self.DEVIATION_CRITICAL_THRESHOLD * 100,
            "trusted_nodes": self.get_trusted_nodes(),
        }


if __name__ == "__main__":
    logger.info("Starting Byzantine detector with live causal engine")

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

    logger.info(
        "System running. Load generator should be in another terminal. "
        "Byzantine detection fires every 15s after causal engine is ready."
    )

    try:
        while True:
            time.sleep(30)
            health = detector.get_cluster_health()
            logger.info(
                "CLUSTER HEALTH trusted=%d suspicious=%d byzantine=%d "
                "checks=%d detections=%d",
                health["trusted_nodes"],
                health["suspicious_nodes"],
                health["byzantine_nodes"],
                health["checks_run"],
                health["byzantine_detections"],
            )
            for node_id, profile in health["node_profiles"].items():
                logger.info(
                    "  node=%-8s status=%-12s trust=%.2f "
                    "proofs_ok=%d violations=%d",
                    node_id,
                    profile["status"],
                    profile["trust_score"],
                    profile["proof_successes"],
                    profile["causal_violations"],
                )
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        detector.stop()
        updater.stop()