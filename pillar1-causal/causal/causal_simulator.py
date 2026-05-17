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
from load_trend_analyzer import LoadTrendAnalyzer, TrendDirection, TrendSeverity

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.predictive.simulator")


class SimulationScenario:
    def __init__(
        self,
        node_id: str,
        scenario_name: str,
        query_load: float,
        causal_effect_ms: float,
        projected_latency_ms: float,
        horizon_minutes: float,
        timestamp: str,
    ):
        self.node_id = node_id
        self.scenario_name = scenario_name
        self.query_load = query_load
        self.causal_effect_ms = causal_effect_ms
        self.projected_latency_ms = projected_latency_ms
        self.horizon_minutes = horizon_minutes
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "scenario_name": self.scenario_name,
            "query_load": round(self.query_load, 2),
            "causal_effect_ms": round(self.causal_effect_ms, 2),
            "projected_latency_ms": round(self.projected_latency_ms, 2),
            "horizon_minutes": self.horizon_minutes,
            "timestamp": self.timestamp,
        }


class NodeSimulation:
    def __init__(
        self,
        node_id: str,
        current_load: float,
        causal_effect_ms: float,
        current_latency_ms: float,
        trend_direction: str,
        change_rate_per_minute: float,
        scenarios: list,
        timestamp: str,
    ):
        self.node_id = node_id
        self.current_load = current_load
        self.causal_effect_ms = causal_effect_ms
        self.current_latency_ms = current_latency_ms
        self.trend_direction = trend_direction
        self.change_rate_per_minute = change_rate_per_minute
        self.scenarios = scenarios
        self.timestamp = timestamp

    def worst_case_latency(self) -> float:
        if not self.scenarios:
            return self.current_latency_ms
        return max(s.projected_latency_ms for s in self.scenarios)

    def best_case_latency(self) -> float:
        if not self.scenarios:
            return self.current_latency_ms
        return min(s.projected_latency_ms for s in self.scenarios)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "current_load": round(self.current_load, 2),
            "causal_effect_ms": round(self.causal_effect_ms, 2),
            "current_latency_ms": round(self.current_latency_ms, 2),
            "trend_direction": self.trend_direction,
            "change_rate_per_minute": round(
                self.change_rate_per_minute, 4
            ),
            "worst_case_latency_ms": round(
                self.worst_case_latency(), 2
            ),
            "best_case_latency_ms": round(
                self.best_case_latency(), 2
            ),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "timestamp": self.timestamp,
        }


class ClusterSimulation:
    def __init__(
        self,
        node_simulations: dict,
        simulation_horizon_minutes: float,
        timestamp: str,
    ):
        self.node_simulations = node_simulations
        self.simulation_horizon_minutes = simulation_horizon_minutes
        self.timestamp = timestamp

    def cluster_worst_case_latency(self) -> float:
        if not self.node_simulations:
            return 0.0
        return max(
            s.worst_case_latency()
            for s in self.node_simulations.values()
        )

    def cluster_best_case_latency(self) -> float:
        if not self.node_simulations:
            return 0.0
        return min(
            s.best_case_latency()
            for s in self.node_simulations.values()
        )

    def highest_risk_node(self) -> Optional[str]:
        if not self.node_simulations:
            return None
        return max(
            self.node_simulations.keys(),
            key=lambda n: self.node_simulations[n].worst_case_latency(),
        )

    def to_dict(self) -> dict:
        return {
            "simulation_horizon_minutes": self.simulation_horizon_minutes,
            "cluster_worst_case_latency_ms": round(
                self.cluster_worst_case_latency(), 2
            ),
            "cluster_best_case_latency_ms": round(
                self.cluster_best_case_latency(), 2
            ),
            "highest_risk_node": self.highest_risk_node(),
            "node_simulations": {
                node_id: sim.to_dict()
                for node_id, sim in self.node_simulations.items()
            },
            "timestamp": self.timestamp,
        }


class CausalSimulator:
    SIMULATION_HORIZONS = [1, 5, 15]
    LOAD_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]
    LATENCY_WARN_THRESHOLD_MS = 200.0
    LATENCY_CRITICAL_THRESHOLD_MS = 500.0

    def __init__(
        self,
        updater: StreamingCausalUpdater,
        analyzer: LoadTrendAnalyzer,
        simulation_interval_seconds: float = 30.0,
    ):
        self.updater = updater
        self.analyzer = analyzer
        self.simulation_interval = simulation_interval_seconds

        self._latest_simulation: Optional[ClusterSimulation] = None
        self._simulation_lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._simulations_run = 0

    def _simulate_node(
        self,
        node_id: str,
        trend_analysis: dict,
        snapshot: dict,
    ) -> Optional[NodeSimulation]:
        causal_effect = abs(snapshot["effect"])
        current_load = trend_analysis["current_load"]
        change_rate = trend_analysis["change_rate_per_minute"]
        trend_direction = trend_analysis["direction"]

        current_latency_ms = causal_effect * current_load

        scenarios = []

        for horizon_min in self.SIMULATION_HORIZONS:
            projected_load = max(
                0.0,
                current_load + change_rate * horizon_min
            )
            projected_latency = causal_effect * projected_load

            scenario = SimulationScenario(
                node_id=node_id,
                scenario_name=f"trend_{horizon_min}min",
                query_load=projected_load,
                causal_effect_ms=causal_effect,
                projected_latency_ms=projected_latency,
                horizon_minutes=horizon_min,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            scenarios.append(scenario)

        for multiplier in self.LOAD_MULTIPLIERS:
            scaled_load = current_load * multiplier
            scaled_latency = causal_effect * scaled_load

            scenario = SimulationScenario(
                node_id=node_id,
                scenario_name=f"load_{multiplier:.1f}x",
                query_load=scaled_load,
                causal_effect_ms=causal_effect,
                projected_latency_ms=scaled_latency,
                horizon_minutes=5.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            scenarios.append(scenario)

        return NodeSimulation(
            node_id=node_id,
            current_load=current_load,
            causal_effect_ms=causal_effect,
            current_latency_ms=current_latency_ms,
            trend_direction=trend_direction,
            change_rate_per_minute=change_rate,
            scenarios=scenarios,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _run_simulation(self) -> Optional[ClusterSimulation]:
        engine_status = self.updater.status()
        if not engine_status.get("is_ready"):
            return None

        trend_analyses = self.analyzer.get_latest_analyses()
        if not trend_analyses:
            return None

        node_simulations = {}

        for node_id in ["node-1", "node-2", "node-3"]:
            trend = trend_analyses.get(node_id)
            snapshot = self.updater.get_current_snapshot(node_id)

            if not trend or not snapshot:
                continue

            sim = self._simulate_node(node_id, trend, snapshot)
            if sim:
                node_simulations[node_id] = sim

                worst = sim.worst_case_latency()
                if worst >= self.LATENCY_CRITICAL_THRESHOLD_MS:
                    logger.critical(
                        "SIMULATION CRITICAL node=%s "
                        "worst_case=%.1fms effect=%.2fms "
                        "trend=%s rate=%+.2f/min",
                        node_id,
                        worst,
                        sim.causal_effect_ms,
                        sim.trend_direction,
                        sim.change_rate_per_minute,
                    )
                elif worst >= self.LATENCY_WARN_THRESHOLD_MS:
                    logger.warning(
                        "SIMULATION WARNING node=%s "
                        "worst_case=%.1fms effect=%.2fms",
                        node_id,
                        worst,
                        sim.causal_effect_ms,
                    )
                else:
                    logger.info(
                        "Simulation node=%-8s "
                        "current=%.1fms worst=%.1fms "
                        "trend=%-8s rate=%+.3f/min",
                        node_id,
                        sim.current_latency_ms,
                        worst,
                        sim.trend_direction,
                        sim.change_rate_per_minute,
                    )

        if not node_simulations:
            return None

        cluster_sim = ClusterSimulation(
            node_simulations=node_simulations,
            simulation_horizon_minutes=max(self.SIMULATION_HORIZONS),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        highest_risk = cluster_sim.highest_risk_node()
        logger.info(
            "Cluster simulation complete nodes=%d "
            "worst_case=%.1fms best_case=%.1fms "
            "highest_risk=%s",
            len(node_simulations),
            cluster_sim.cluster_worst_case_latency(),
            cluster_sim.cluster_best_case_latency(),
            highest_risk,
        )

        return cluster_sim

    def _simulation_loop(self):
        logger.info(
            "Causal simulator started interval=%.1fs "
            "horizons=%s multipliers=%s",
            self.simulation_interval,
            self.SIMULATION_HORIZONS,
            self.LOAD_MULTIPLIERS,
        )
        while self._running:
            time.sleep(self.simulation_interval)
            try:
                sim = self._run_simulation()
                if sim:
                    with self._simulation_lock:
                        self._latest_simulation = sim
                    self._simulations_run += 1
            except Exception as e:
                logger.error("Simulation error: %s", e)

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._simulation_loop,
            name="causal-simulator",
            daemon=True,
        )
        self._thread.start()
        logger.info("CausalSimulator started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("CausalSimulator stopped")

    def get_latest_simulation(self) -> Optional[dict]:
        with self._simulation_lock:
            if self._latest_simulation is None:
                return None
            return self._latest_simulation.to_dict()

    def simulate_now(self) -> Optional[dict]:
        sim = self._run_simulation()
        if sim:
            with self._simulation_lock:
                self._latest_simulation = sim
            self._simulations_run += 1
            return sim.to_dict()
        return None

    def get_node_simulation(self, node_id: str) -> Optional[dict]:
        with self._simulation_lock:
            if self._latest_simulation is None:
                return None
            node_sim = self._latest_simulation.node_simulations.get(
                node_id
            )
            return node_sim.to_dict() if node_sim else None

    def status(self) -> dict:
        with self._simulation_lock:
            has_sim = self._latest_simulation is not None
            highest_risk = (
                self._latest_simulation.highest_risk_node()
                if has_sim else None
            )
            worst_latency = (
                self._latest_simulation.cluster_worst_case_latency()
                if has_sim else 0.0
            )
        return {
            "running": self._running,
            "simulations_run": self._simulations_run,
            "has_simulation": has_sim,
            "highest_risk_node": highest_risk,
            "cluster_worst_case_latency_ms": round(worst_latency, 2),
            "simulation_horizons_minutes": self.SIMULATION_HORIZONS,
            "load_multipliers": self.LOAD_MULTIPLIERS,
            "warn_threshold_ms": self.LATENCY_WARN_THRESHOLD_MS,
            "critical_threshold_ms": self.LATENCY_CRITICAL_THRESHOLD_MS,
        }


if __name__ == "__main__":
    logger.info("Starting causal simulator demo")

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

    logger.info(
        "System running. Load generator in another terminal."
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

            logger.info("=== SIMULATION CYCLE %d ===", cycle)

            sim = simulator.get_latest_simulation()
            if not sim:
                sim = simulator.simulate_now()

            if not sim:
                logger.info(
                    "No simulation yet — trend analyses accumulating"
                )
                continue

            logger.info(
                "Cluster: worst_case=%.1fms best_case=%.1fms "
                "highest_risk=%s",
                sim["cluster_worst_case_latency_ms"],
                sim["cluster_best_case_latency_ms"],
                sim["highest_risk_node"],
            )

            for node_id, node_sim in sim[
                "node_simulations"
            ].items():
                logger.info(
                    "  node=%-8s current=%.1fms worst=%.1fms "
                    "trend=%-8s rate=%+.3f/min",
                    node_id,
                    node_sim["current_latency_ms"],
                    node_sim["worst_case_latency_ms"],
                    node_sim["trend_direction"],
                    node_sim["change_rate_per_minute"],
                )
                for scenario in node_sim["scenarios"][:3]:
                    logger.info(
                        "    %-20s load=%.1f "
                        "latency=%.1fms",
                        scenario["scenario_name"],
                        scenario["query_load"],
                        scenario["projected_latency_ms"],
                    )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        simulator.stop()
        analyzer.stop()
        updater.stop()