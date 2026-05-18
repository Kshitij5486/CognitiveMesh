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
logger = logging.getLogger("cm.health")


class HealthStatus(Enum):
    HEALTHY   = "healthy"
    DEGRADED  = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN   = "unknown"


class ComponentHealth:
    """Health state for a single component."""

    def __init__(self, name: str):
        self.name = name
        self.status = HealthStatus.UNKNOWN
        self.last_check: Optional[float] = None
        self.last_ok: Optional[float] = None
        self.consecutive_failures = 0
        self.total_checks = 0
        self.total_failures = 0
        self.details: dict = {}
        self.message: str = ""

    @property
    def uptime_ratio(self) -> float:
        if self.total_checks == 0:
            return 1.0
        return (
            (self.total_checks - self.total_failures)
            / self.total_checks
        )

    @property
    def seconds_since_ok(self) -> Optional[float]:
        if self.last_ok is None:
            return None
        return round(time.time() - self.last_ok, 1)

    def record_ok(self, details: dict = None):
        self.status = HealthStatus.HEALTHY
        self.last_check = time.time()
        self.last_ok = time.time()
        self.consecutive_failures = 0
        self.total_checks += 1
        self.details = details or {}
        self.message = "OK"

    def record_degraded(
        self, message: str, details: dict = None
    ):
        self.status = HealthStatus.DEGRADED
        self.last_check = time.time()
        self.consecutive_failures += 1
        self.total_checks += 1
        self.total_failures += 1
        self.details = details or {}
        self.message = message

    def record_failure(
        self, message: str, details: dict = None
    ):
        self.status = HealthStatus.UNHEALTHY
        self.last_check = time.time()
        self.consecutive_failures += 1
        self.total_checks += 1
        self.total_failures += 1
        self.details = details or {}
        self.message = message

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "uptime_ratio": round(self.uptime_ratio, 4),
            "total_checks": self.total_checks,
            "total_failures": self.total_failures,
            "consecutive_failures": (
                self.consecutive_failures
            ),
            "last_check": datetime.fromtimestamp(
                self.last_check, tz=timezone.utc
            ).isoformat() if self.last_check else None,
            "last_ok": datetime.fromtimestamp(
                self.last_ok, tz=timezone.utc
            ).isoformat() if self.last_ok else None,
            "seconds_since_ok": self.seconds_since_ok,
            "details": self.details,
        }


class SLATarget:
    """Tracks SLA compliance for one metric."""

    def __init__(
        self,
        name: str,
        target_pct: float,
        window_minutes: int = 60,
    ):
        self.name = name
        self.target_pct = target_pct
        self.window_minutes = window_minutes
        self._observations: deque = deque(
            maxlen=window_minutes * 2
        )

    def record(self, ok: bool):
        self._observations.append(
            (time.time(), ok)
        )

    @property
    def compliance_pct(self) -> float:
        if not self._observations:
            return 100.0
        window_start = (
            time.time() - self.window_minutes * 60
        )
        recent = [
            ok for ts, ok in self._observations
            if ts >= window_start
        ]
        if not recent:
            return 100.0
        return round(
            100.0 * sum(recent) / len(recent), 2
        )

    @property
    def is_compliant(self) -> bool:
        return self.compliance_pct >= self.target_pct

    @property
    def observations_count(self) -> int:
        return len(self._observations)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target_pct": self.target_pct,
            "compliance_pct": self.compliance_pct,
            "is_compliant": self.is_compliant,
            "window_minutes": self.window_minutes,
            "observations": self.observations_count,
        }


class HealthMonitor:
    """
    Central health monitoring for CognitiveMesh
    Sprint 10 production hardening.

    Checks and tracks:
    - Causal engine readiness
    - Quorum manager state
    - Byzantine coordinator activity
    - Recovery orchestrator state
    - Quorum-aware router weights
    - Persistence connectivity
    - Overall cluster health

    SLA targets tracked:
    - Quorum availability >= 99%
    - Engine readiness >= 95%
    - Routing stability >= 90%
    - Zero quorum violations per hour

    Exposes:
    - /health/live    — liveness probe (always 200 if running)
    - /health/ready   — readiness probe (200 if all critical OK)
    - /health/status  — full component breakdown
    - /health/sla     — SLA compliance per target
    - /health/history — recent health check history
    """

    CHECK_INTERVAL_S   = 15.0
    HISTORY_SIZE       = 200
    DEGRADED_THRESHOLD = 2   # consecutive failures → degraded
    UNHEALTHY_THRESHOLD = 5  # consecutive failures → unhealthy

    # SLA targets
    SLA_QUORUM_AVAILABILITY  = 99.0
    SLA_ENGINE_READINESS     = 95.0
    SLA_ROUTING_STABILITY    = 90.0

    def __init__(
        self,
        updater,
        quorum_manager,
        coordinator,
        orchestrator,
        quorum_router,
        persistence=None,
    ):
        self.updater = updater
        self.quorum_manager = quorum_manager
        self.coordinator = coordinator
        self.orchestrator = orchestrator
        self.quorum_router = quorum_router
        self.persistence = persistence

        # Component health trackers
        self._components: dict[str, ComponentHealth] = {
            "causal_engine": ComponentHealth(
                "causal_engine"
            ),
            "quorum_manager": ComponentHealth(
                "quorum_manager"
            ),
            "byzantine_coordinator": ComponentHealth(
                "byzantine_coordinator"
            ),
            "recovery_orchestrator": ComponentHealth(
                "recovery_orchestrator"
            ),
            "quorum_router": ComponentHealth(
                "quorum_router"
            ),
            "persistence": ComponentHealth(
                "persistence"
            ),
        }

        # SLA targets
        self._sla_targets: dict[str, SLATarget] = {
            "quorum_availability": SLATarget(
                "quorum_availability",
                self.SLA_QUORUM_AVAILABILITY,
            ),
            "engine_readiness": SLATarget(
                "engine_readiness",
                self.SLA_ENGINE_READINESS,
            ),
            "routing_stability": SLATarget(
                "routing_stability",
                self.SLA_ROUTING_STABILITY,
            ),
        }

        # History
        self._check_history: deque = deque(
            maxlen=self.HISTORY_SIZE
        )

        # Counters
        self._check_count = 0
        self._overall_status = HealthStatus.UNKNOWN
        self._start_time: Optional[float] = None
        self._last_check_time: Optional[float] = None

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Component checks ───────────────────────────────────

    def _check_causal_engine(self):
        comp = self._components["causal_engine"]
        try:
            status = self.updater.status()
            is_ready = status.get("is_ready", False)
            buffer_size = status.get("buffer_size", 0)
            retrain_count = status.get(
                "retrain_count", 0
            )

            details = {
                "is_ready": is_ready,
                "buffer_size": buffer_size,
                "retrain_count": retrain_count,
                "nodes_modeled": status.get(
                    "nodes_modeled", []
                ),
            }

            if not is_ready:
                comp.record_degraded(
                    f"Engine not ready buffer={buffer_size}",
                    details,
                )
            elif buffer_size < 10:
                comp.record_degraded(
                    f"Buffer low: {buffer_size}",
                    details,
                )
            else:
                comp.record_ok(details)

            self._sla_targets[
                "engine_readiness"
            ].record(is_ready)

        except Exception as e:
            comp.record_failure(
                f"Engine check error: {e}"
            )
            self._sla_targets[
                "engine_readiness"
            ].record(False)

    def _check_quorum_manager(self):
        comp = self._components["quorum_manager"]
        try:
            status = self.quorum_manager.status()
            quorum_state = status["quorum_state"]
            contributing = status["contributing_nodes"]
            violations = status["quorum_violations"]

            details = {
                "quorum_state": quorum_state,
                "contributing_nodes": contributing,
                "recovering_nodes": status[
                    "recovering_nodes"
                ],
                "violations": violations,
                "node_states": status["node_states"],
            }

            quorum_ok = contributing >= 2

            if quorum_state == "quorum_lost":
                comp.record_failure(
                    "QUORUM LOST — cluster critical",
                    details,
                )
            elif quorum_state == "critical":
                comp.record_degraded(
                    "Quorum at minimum (2/3)",
                    details,
                )
            elif violations > 0:
                comp.record_degraded(
                    f"Quorum violations: {violations}",
                    details,
                )
            else:
                comp.record_ok(details)

            self._sla_targets[
                "quorum_availability"
            ].record(quorum_ok)

        except Exception as e:
            comp.record_failure(
                f"Quorum check error: {e}"
            )
            self._sla_targets[
                "quorum_availability"
            ].record(False)

    def _check_byzantine_coordinator(self):
        comp = self._components[
            "byzantine_coordinator"
        ]
        try:
            status = self.coordinator.status()
            checks = status["check_count"]
            detections = status["detections_total"]
            active_plan = status.get("active_plan")

            details = {
                "check_count": checks,
                "detections_total": detections,
                "recoveries_total": status[
                    "recoveries_total"
                ],
                "node_states": status["node_states"],
                "has_active_plan": active_plan is not None,
            }

            confirmed_nodes = [
                n for n, s in status[
                    "node_states"
                ].items()
                if s == "confirmed"
            ]

            if len(confirmed_nodes) >= 2:
                comp.record_failure(
                    f"Multiple Byzantine confirmed: "
                    f"{confirmed_nodes}",
                    details,
                )
            elif confirmed_nodes:
                comp.record_degraded(
                    f"Byzantine confirmed: "
                    f"{confirmed_nodes}",
                    details,
                )
            else:
                comp.record_ok(details)

        except Exception as e:
            comp.record_failure(
                f"Coordinator check error: {e}"
            )

    def _check_recovery_orchestrator(self):
        comp = self._components[
            "recovery_orchestrator"
        ]
        try:
            status = self.orchestrator.status()
            sessions_total = status["sessions_total"]
            success_rate = status["session_success_rate"]
            active = status["active_session"]

            details = {
                "sessions_total": sessions_total,
                "sessions_completed": status[
                    "sessions_completed"
                ],
                "sessions_failed": status[
                    "sessions_failed"
                ],
                "success_rate": success_rate,
                "has_active_session": active is not None,
                "nodes_recovered": status[
                    "nodes_recovered_total"
                ],
            }

            if (
                sessions_total > 0
                and success_rate < 0.5
            ):
                comp.record_failure(
                    f"Recovery success rate critical: "
                    f"{success_rate:.2f}",
                    details,
                )
            elif (
                sessions_total > 0
                and success_rate < 0.8
            ):
                comp.record_degraded(
                    f"Recovery success rate low: "
                    f"{success_rate:.2f}",
                    details,
                )
            else:
                comp.record_ok(details)

        except Exception as e:
            comp.record_failure(
                f"Orchestrator check error: {e}"
            )

    def _check_quorum_router(self):
        comp = self._components["quorum_router"]
        try:
            status = self.quorum_router.status()
            stability = status["cluster_stability"]
            excluded = status["excluded_nodes"]
            decision = status.get("current_decision")
            dtype = (
                decision["decision_type"]
                if decision else "unknown"
            )

            details = {
                "cluster_stability": stability,
                "excluded_nodes": excluded,
                "decision_type": dtype,
                "active_nodes": status["active_node_count"],
                "exclusion_events": status[
                    "exclusion_events"
                ],
            }

            routing_ok = stability >= 0.80

            if dtype == "emergency_uniform":
                comp.record_failure(
                    "Emergency uniform routing active",
                    details,
                )
            elif len(excluded) >= 2:
                comp.record_degraded(
                    f"Critical routing: "
                    f"{len(excluded)} nodes excluded",
                    details,
                )
            elif stability < 0.80:
                comp.record_degraded(
                    f"Routing stability low: "
                    f"{stability:.3f}",
                    details,
                )
            else:
                comp.record_ok(details)

            self._sla_targets[
                "routing_stability"
            ].record(routing_ok)

        except Exception as e:
            comp.record_failure(
                f"Router check error: {e}"
            )
            self._sla_targets[
                "routing_stability"
            ].record(False)

    def _check_persistence(self):
        comp = self._components["persistence"]
        if self.persistence is None:
            comp.record_ok({
                "mode": "disabled",
                "note": "no persistence configured",
            })
            return
        try:
            stats = self.persistence.get_stats()
            connected = stats["connected"]
            failed = stats["writes_failed"]
            queue = stats["queue_depth"]

            details = {
                "connected": connected,
                "writes_total": stats["writes_total"],
                "writes_failed": failed,
                "queue_depth": queue,
            }

            if not connected:
                comp.record_failure(
                    "Persistence disconnected", details
                )
            elif failed > 10:
                comp.record_degraded(
                    f"High write failures: {failed}",
                    details,
                )
            elif queue > 200:
                comp.record_degraded(
                    f"Write queue backing up: {queue}",
                    details,
                )
            else:
                comp.record_ok(details)

        except Exception as e:
            comp.record_failure(
                f"Persistence check error: {e}"
            )

    # ── Overall status ─────────────────────────────────────

    def _compute_overall_status(self) -> HealthStatus:
        statuses = [
            c.status
            for c in self._components.values()
        ]
        if any(
            s == HealthStatus.UNHEALTHY
            for s in statuses
        ):
            return HealthStatus.UNHEALTHY
        if any(
            s == HealthStatus.DEGRADED
            for s in statuses
        ):
            return HealthStatus.DEGRADED
        if all(
            s == HealthStatus.HEALTHY
            for s in statuses
        ):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    def _run_all_checks(self):
        self._check_count += 1
        self._last_check_time = time.time()

        self._check_causal_engine()
        self._check_quorum_manager()
        self._check_byzantine_coordinator()
        self._check_recovery_orchestrator()
        self._check_quorum_router()
        self._check_persistence()

        with self._lock:
            self._overall_status = (
                self._compute_overall_status()
            )

        # Record to history
        snapshot = {
            "check_number": self._check_count,
            "overall_status": (
                self._overall_status.value
            ),
            "component_statuses": {
                name: comp.status.value
                for name, comp in
                self._components.items()
            },
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }
        with self._lock:
            self._check_history.append(snapshot)

        if self._check_count % 4 == 0:
            logger.info(
                "Health check #%d overall=%s "
                "components=%s",
                self._check_count,
                self._overall_status.value,
                {
                    n: c.status.value
                    for n, c in
                    self._components.items()
                },
            )

    def _monitor_loop(self):
        logger.info(
            "HealthMonitor loop started "
            "interval=%.1fs",
            self.CHECK_INTERVAL_S,
        )
        while self._running:
            time.sleep(self.CHECK_INTERVAL_S)
            try:
                self._run_all_checks()
            except Exception as e:
                logger.error(
                    "Health check error: %s", e
                )

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="health-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("HealthMonitor started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("HealthMonitor stopped")

    # ── Public API ─────────────────────────────────────────

    def get_liveness(self) -> dict:
        """Liveness probe — always alive if running."""
        uptime = (
            time.time() - self._start_time
            if self._start_time else 0.0
        )
        return {
            "alive": self._running,
            "uptime_seconds": round(uptime, 1),
            "check_count": self._check_count,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    def get_readiness(self) -> tuple:
        """
        Readiness probe.
        Returns (is_ready, details).
        Critical components: causal_engine,
        quorum_manager, quorum_router.
        """
        critical = [
            "causal_engine",
            "quorum_manager",
            "quorum_router",
        ]
        not_ready = []
        for name in critical:
            comp = self._components[name]
            if comp.status != HealthStatus.HEALTHY:
                not_ready.append(
                    f"{name}:{comp.status.value}"
                )

        is_ready = len(not_ready) == 0
        return is_ready, {
            "ready": is_ready,
            "critical_components": critical,
            "not_ready": not_ready,
            "overall_status": (
                self._overall_status.value
            ),
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    def get_full_status(self) -> dict:
        with self._lock:
            overall = self._overall_status.value

        uptime = (
            time.time() - self._start_time
            if self._start_time else 0.0
        )

        return {
            "overall_status": overall,
            "version": "1.0.0-rc",
            "uptime_seconds": round(uptime, 1),
            "check_count": self._check_count,
            "last_check": datetime.fromtimestamp(
                self._last_check_time, tz=timezone.utc
            ).isoformat()
            if self._last_check_time else None,
            "components": {
                name: comp.to_dict()
                for name, comp in
                self._components.items()
            },
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    def get_sla_status(self) -> dict:
        all_compliant = all(
            t.is_compliant
            for t in self._sla_targets.values()
        )
        uptime = (
            time.time() - self._start_time
            if self._start_time else 0.0
        )
        return {
            "all_compliant": all_compliant,
            "targets": {
                name: target.to_dict()
                for name, target in
                self._sla_targets.items()
            },
            "monitor_uptime_seconds": round(uptime, 1),
            "checks_run": self._check_count,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    def get_history(self, n: int = 20) -> list:
        with self._lock:
            history = list(self._check_history)
        return history[-n:]

    def status(self) -> dict:
        uptime = (
            time.time() - self._start_time
            if self._start_time else 0.0
        )
        return {
            "running": self._running,
            "overall_status": (
                self._overall_status.value
            ),
            "check_count": self._check_count,
            "check_interval_seconds": (
                self.CHECK_INTERVAL_S
            ),
            "uptime_seconds": round(uptime, 1),
            "sla_targets": len(self._sla_targets),
            "components_tracked": len(
                self._components
            ),
        }


if __name__ == "__main__":
    logger.info("Starting HealthMonitor demo")

    from streaming_updater import StreamingCausalUpdater
    from quorum_manager import QuorumManager
    from byzantine_recovery_coordinator import (
        ByzantineRecoveryCoordinator,
    )
    from multi_node_recovery_orchestrator import (
        MultiNodeRecoveryOrchestrator,
    )
    from quorum_aware_router import QuorumAwareRouter
    from query_router import QueryRouter, RoutingStrategy
    from auto_retrainer import AutoRetrainer

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

    base_router = QueryRouter(
        updater=updater,
        alerter=None,
        strategy=RoutingStrategy.CAUSAL_WEIGHTED,
        check_interval=15.0,
    )
    base_router.start()

    base_retrainer = AutoRetrainer(
        updater=updater,
        alerter=None,
        check_interval=30.0,
        drift_threshold_ms=3.0,
    )
    base_retrainer.start()

    coordinator = ByzantineRecoveryCoordinator(
        updater=updater,
        quorum_manager=quorum,
        router=base_router,
        retrainer=base_retrainer,
        check_interval=15.0,
    )
    coordinator.start()

    orchestrator = MultiNodeRecoveryOrchestrator(
        updater=updater,
        quorum_manager=quorum,
        coordinator=coordinator,
        check_interval=15.0,
    )
    orchestrator.start()

    q_router = QuorumAwareRouter(
        updater=updater,
        quorum_manager=quorum,
        check_interval=10.0,
    )
    q_router.start()

    monitor = HealthMonitor(
        updater=updater,
        quorum_manager=quorum,
        coordinator=coordinator,
        orchestrator=orchestrator,
        quorum_router=q_router,
        persistence=None,
    )
    monitor.start()

    logger.info(
        "HealthMonitor running. "
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
                "=== HEALTH CYCLE %d ===", cycle
            )

            # Liveness
            live = monitor.get_liveness()
            logger.info(
                "Liveness: alive=%s uptime=%.1fs "
                "checks=%d",
                live["alive"],
                live["uptime_seconds"],
                live["check_count"],
            )

            # Readiness
            is_ready, ready_details = (
                monitor.get_readiness()
            )
            logger.info(
                "Readiness: ready=%s not_ready=%s",
                is_ready,
                ready_details["not_ready"],
            )

            # Full status
            full = monitor.get_full_status()
            logger.info(
                "Overall: %s",
                full["overall_status"].upper(),
            )
            for name, comp in (
                full["components"].items()
            ):
                logger.info(
                    "  %-30s %s  uptime=%.3f  %s",
                    name,
                    comp["status"].upper(),
                    comp["uptime_ratio"],
                    comp["message"],
                )

            # SLA
            sla = monitor.get_sla_status()
            logger.info(
                "SLA all_compliant=%s",
                sla["all_compliant"],
            )
            for name, target in (
                sla["targets"].items()
            ):
                logger.info(
                    "  %-30s %.1f%% / %.1f%% target "
                    "compliant=%s obs=%d",
                    name,
                    target["compliance_pct"],
                    target["target_pct"],
                    target["is_compliant"],
                    target["observations"],
                )

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        monitor.stop()
        orchestrator.stop()
        coordinator.stop()
        base_retrainer.stop()
        base_router.stop()
        q_router.stop()
        quorum.stop()
        updater.stop()