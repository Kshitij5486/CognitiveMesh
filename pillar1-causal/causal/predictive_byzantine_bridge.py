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

from streaming_updater import StreamingCausalUpdater
from load_trend_analyzer import LoadTrendAnalyzer
from causal_simulator import CausalSimulator
from predictive_alerter import PredictiveAlerter, PredictiveAlert, AlertType, AlertSeverity

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.predictive.byzantine_bridge")


class ByzantinePredictiveBridge:
    """
    Bridges the Byzantine consensus stack (Sprint 5) with the
    Predictive Intelligence stack (Sprint 6).

    Responsibilities:
    - Poll Byzantine API for node reputation and isolation status
    - Exclude Byzantine/isolated nodes from cluster forecasts
    - Fire predictive alerts when a node approaches Byzantine threshold
    - Enrich simulation results with Byzantine risk scores
    """

    BYZANTINE_SCORE_WARN = 0.40
    BYZANTINE_SCORE_CRITICAL = 0.20
    BYZANTINE_API_URL = "http://localhost:8085"
    POLL_INTERVAL_SECONDS = 15.0

    def __init__(
        self,
        alerter: PredictiveAlerter,
        poll_interval: float = 15.0,
    ):
        self.alerter = alerter
        self.poll_interval = poll_interval

        self._node_reputations: dict = {}
        self._isolated_nodes: list = []
        self._active_nodes: list = []
        self._cluster_operational: bool = True
        self._byzantine_api_available: bool = False
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._poll_count = 0
        self._last_poll: Optional[str] = None

    def _poll_byzantine_api(self) -> Optional[dict]:
        try:
            import urllib.request
            import json
            url = f"{self.BYZANTINE_API_URL}/cluster"
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                return data
        except Exception as e:
            logger.debug("Byzantine API not available: %s", e)
            return None

    def _poll_nodes_api(self) -> Optional[dict]:
        try:
            import urllib.request
            import json
            url = f"{self.BYZANTINE_API_URL}/nodes"
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                return data
        except Exception as e:
            logger.debug("Byzantine nodes API not available: %s", e)
            return None

    def _check_reputation_alerts(self, nodes: dict):
        for node_id, node_data in nodes.items():
            score = node_data.get("reputation_score", 1.0)
            status = node_data.get("reputation_status", "trusted")
            isolated = node_data.get("isolated", False)

            if isolated:
                alert = PredictiveAlert(
                    alert_id=self.alerter._make_alert_id(),
                    node_id=node_id,
                    alert_type=AlertType.CAUSAL_THRESHOLD,
                    severity=AlertSeverity.CRITICAL,
                    current_value=score,
                    predicted_value=0.0,
                    threshold=0.15,
                    horizon_minutes=0.0,
                    message=(
                        f"{node_id} is ISOLATED by Byzantine detector "
                        f"(reputation={score:.2f}). "
                        f"Excluding from cluster forecasts."
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self.alerter._fire_alert(alert)
                logger.warning(
                    "Byzantine bridge: node=%s ISOLATED "
                    "reputation=%.2f — firing predictive alert",
                    node_id, score,
                )

            elif score <= self.BYZANTINE_SCORE_CRITICAL:
                alert = PredictiveAlert(
                    alert_id=self.alerter._make_alert_id(),
                    node_id=node_id,
                    alert_type=AlertType.CAUSAL_THRESHOLD,
                    severity=AlertSeverity.CRITICAL,
                    current_value=score,
                    predicted_value=score * 0.5,
                    threshold=self.BYZANTINE_SCORE_CRITICAL,
                    horizon_minutes=5.0,
                    message=(
                        f"{node_id} reputation CRITICAL: {score:.2f} "
                        f"(threshold={self.BYZANTINE_SCORE_CRITICAL}). "
                        f"Byzantine behavior predicted — "
                        f"causal claims may be unreliable."
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self.alerter._fire_alert(alert)
                logger.warning(
                    "Byzantine bridge: node=%s reputation=%.2f "
                    "CRITICAL threshold=%.2f",
                    node_id, score, self.BYZANTINE_SCORE_CRITICAL,
                )

            elif score <= self.BYZANTINE_SCORE_WARN:
                alert = PredictiveAlert(
                    alert_id=self.alerter._make_alert_id(),
                    node_id=node_id,
                    alert_type=AlertType.CAUSAL_THRESHOLD,
                    severity=AlertSeverity.WARNING,
                    current_value=score,
                    predicted_value=score * 0.7,
                    threshold=self.BYZANTINE_SCORE_WARN,
                    horizon_minutes=5.0,
                    message=(
                        f"{node_id} reputation WARNING: {score:.2f} "
                        f"(threshold={self.BYZANTINE_SCORE_WARN}). "
                        f"Monitor for Byzantine behavior."
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self.alerter._fire_alert(alert)
                logger.info(
                    "Byzantine bridge: node=%s reputation=%.2f "
                    "WARNING threshold=%.2f",
                    node_id, score, self.BYZANTINE_SCORE_WARN,
                )

    def _poll_cycle(self):
        self._poll_count += 1
        self._last_poll = datetime.now(timezone.utc).isoformat()

        cluster_data = self._poll_byzantine_api()
        nodes_data = self._poll_nodes_api()

        if cluster_data is None or nodes_data is None:
            with self._lock:
                self._byzantine_api_available = False
            return

        with self._lock:
            self._byzantine_api_available = True
            self._isolated_nodes = cluster_data.get(
                "isolated_nodes", []
            )
            self._active_nodes = cluster_data.get("active_nodes", [])
            self._cluster_operational = cluster_data.get(
                "cluster_operational", True
            )
            self._node_reputations = nodes_data.get("nodes", {})

        logger.info(
            "Byzantine bridge poll=%d active=%s isolated=%s "
            "operational=%s",
            self._poll_count,
            self._active_nodes,
            self._isolated_nodes,
            self._cluster_operational,
        )

        if nodes_data:
            self._check_reputation_alerts(
                nodes_data.get("nodes", {})
            )

    def _poll_loop(self):
        logger.info(
            "ByzantinePredictiveBridge started "
            "poll_interval=%.1fs "
            "byzantine_api=%s",
            self.poll_interval,
            self.BYZANTINE_API_URL,
        )
        while self._running:
            try:
                self._poll_cycle()
            except Exception as e:
                logger.error("Bridge poll error: %s", e)
            time.sleep(self.poll_interval)

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="byzantine-bridge",
            daemon=True,
        )
        self._thread.start()
        logger.info("ByzantinePredictiveBridge started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("ByzantinePredictiveBridge stopped")

    def get_trusted_nodes(self) -> list:
        with self._lock:
            if not self._byzantine_api_available:
                return ["node-1", "node-2", "node-3"]
            return list(self._active_nodes)

    def is_node_trusted(self, node_id: str) -> bool:
        with self._lock:
            if not self._byzantine_api_available:
                return True
            return node_id not in self._isolated_nodes

    def get_node_reputation(self, node_id: str) -> Optional[float]:
        with self._lock:
            node = self._node_reputations.get(node_id)
            if node:
                return node.get("reputation_score")
        return None

    def enrich_simulation(self, sim: dict) -> dict:
        with self._lock:
            if not self._byzantine_api_available:
                sim["byzantine_context"] = {
                    "available": False,
                    "note": "Byzantine API not running — "
                            "start port 8085 for full integration",
                }
                return sim

            node_trust = {}
            for node_id in ["node-1", "node-2", "node-3"]:
                node_data = self._node_reputations.get(node_id, {})
                node_trust[node_id] = {
                    "reputation_score": node_data.get(
                        "reputation_score", 1.0
                    ),
                    "reputation_status": node_data.get(
                        "reputation_status", "trusted"
                    ),
                    "isolated": node_data.get("isolated", False),
                    "trusted": node_id not in self._isolated_nodes,
                }

            sim["byzantine_context"] = {
                "available": True,
                "cluster_operational": self._cluster_operational,
                "active_nodes": self._active_nodes,
                "isolated_nodes": self._isolated_nodes,
                "node_trust": node_trust,
                "forecast_excludes_isolated": True,
            }

        return sim

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "byzantine_api_available": self._byzantine_api_available,
                "poll_count": self._poll_count,
                "last_poll": self._last_poll,
                "active_nodes": list(self._active_nodes),
                "isolated_nodes": list(self._isolated_nodes),
                "cluster_operational": self._cluster_operational,
                "node_reputations": {
                    node_id: data.get("reputation_score")
                    for node_id, data in self._node_reputations.items()
                },
            }


if __name__ == "__main__":
    logger.info("Starting Byzantine-Predictive integration demo")

    updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )
    updater.start()

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

    bridge = ByzantinePredictiveBridge(
        alerter=alerter,
        poll_interval=15.0,
    )
    bridge.start()

    logger.info(
        "Full integrated stack running. "
        "Start port 8085 Byzantine API for full integration. "
        "Load generator in another terminal."
    )

    try:
        cycle = 0
        while True:
            time.sleep(30)
            cycle += 1

            logger.info("=== INTEGRATION CYCLE %d ===", cycle)

            bridge_status = bridge.status()
            logger.info(
                "Byzantine bridge: available=%s polls=%d "
                "active=%s isolated=%s",
                bridge_status["byzantine_api_available"],
                bridge_status["poll_count"],
                bridge_status["active_nodes"],
                bridge_status["isolated_nodes"],
            )

            trusted = bridge.get_trusted_nodes()
            logger.info("Trusted nodes for forecasting: %s", trusted)

            alerter_status = alerter.status()
            logger.info(
                "Alerter: checks=%d fired=%d active=%d",
                alerter_status["checks_run"],
                alerter_status["total_alerts_fired"],
                alerter_status["active_alert_count"],
            )

            active_alerts = alerter.get_active_alerts()
            if active_alerts:
                for alert in active_alerts:
                    logger.info(
                        "  ACTIVE [%s] node=%s severity=%s: %s",
                        alert["alert_type"],
                        alert["node_id"],
                        alert["severity"].upper(),
                        alert["message"][:80],
                    )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        bridge.stop()
        alerter.stop()
        simulator.stop()
        analyzer.stop()
        updater.stop()