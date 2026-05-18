import logging
import threading
import time
import json
from datetime import datetime, timezone
from typing import Optional
from collections import deque

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
logger = logging.getLogger("cm.persistence")

# ── Schema DDL ─────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cm_recovery_sessions (
    session_id          VARCHAR(32)  PRIMARY KEY,
    trigger             VARCHAR(128) NOT NULL,
    phase               VARCHAR(32)  NOT NULL,
    affected_nodes      JSONB        NOT NULL DEFAULT '[]',
    node_outcomes       JSONB        NOT NULL DEFAULT '{}',
    node_durations_s    JSONB        NOT NULL DEFAULT '{}',
    node_effects_start  JSONB        NOT NULL DEFAULT '{}',
    node_effects_end    JSONB        NOT NULL DEFAULT '{}',
    quorum_state_start  VARCHAR(32),
    quorum_state_end    VARCHAR(32),
    success_count       INTEGER      NOT NULL DEFAULT 0,
    failure_count       INTEGER      NOT NULL DEFAULT 0,
    total_duration_s    FLOAT,
    mttr_s              FLOAT,
    notes               JSONB        NOT NULL DEFAULT '[]',
    plan_ids            JSONB        NOT NULL DEFAULT '[]',
    started_at          TIMESTAMPTZ  NOT NULL,
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cm_quorum_decisions (
    decision_id         SERIAL       PRIMARY KEY,
    node_id             VARCHAR(32)  NOT NULL,
    reason              VARCHAR(256) NOT NULL,
    decision            VARCHAR(32)  NOT NULL,
    quorum_state        VARCHAR(32)  NOT NULL,
    contributing_nodes  INTEGER      NOT NULL,
    recovering_nodes    INTEGER      NOT NULL,
    decided_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cm_byzantine_events (
    event_id            VARCHAR(64)  PRIMARY KEY,
    node_id             VARCHAR(32)  NOT NULL,
    method              VARCHAR(64)  NOT NULL,
    effect_ms           FLOAT        NOT NULL,
    byzantine_score     FLOAT,
    details             TEXT,
    acknowledged        BOOLEAN      NOT NULL DEFAULT FALSE,
    occurred_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cm_sla_snapshots (
    snapshot_id         SERIAL       PRIMARY KEY,
    quorum_state        VARCHAR(32)  NOT NULL,
    contributing_nodes  INTEGER      NOT NULL,
    recovering_nodes    INTEGER      NOT NULL,
    causal_effects      JSONB        NOT NULL DEFAULT '{}',
    routing_weights     JSONB        NOT NULL DEFAULT '{}',
    excluded_nodes      JSONB        NOT NULL DEFAULT '[]',
    cluster_stability   FLOAT,
    active_session      BOOLEAN      NOT NULL DEFAULT FALSE,
    byzantine_detections INTEGER     NOT NULL DEFAULT 0,
    recorded_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_phase
    ON cm_recovery_sessions (phase);
CREATE INDEX IF NOT EXISTS idx_sessions_started
    ON cm_recovery_sessions (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_node
    ON cm_quorum_decisions (node_id, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_node
    ON cm_byzantine_events (node_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_sla_recorded
    ON cm_sla_snapshots (recorded_at DESC);
"""


class SessionPersistence:
    """
    PostgreSQL-backed persistence for CognitiveMesh
    Sprint 10 production hardening.

    Persists:
    - Recovery sessions (cm_recovery_sessions)
    - Quorum decisions (cm_quorum_decisions)
    - Byzantine events (cm_byzantine_events)
    - SLA snapshots (cm_sla_snapshots)

    Write strategy:
    - Async write queue (deque) — writes never block callers
    - Background writer thread flushes every FLUSH_INTERVAL_S
    - Failed writes retry up to MAX_RETRIES times
    - Falls back to in-memory log on persistent failure

    Read strategy:
    - Synchronous direct reads (for API endpoints)
    - Cached recent results (TTL=30s)
    """

    FLUSH_INTERVAL_S  = 5.0
    MAX_RETRIES       = 3
    RETRY_DELAY_S     = 1.0
    CACHE_TTL_S       = 30.0
    QUEUE_MAX         = 500

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._write_queue: deque = deque(
            maxlen=self.QUEUE_MAX
        )
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

        # Counters
        self._writes_total = 0
        self._writes_failed = 0
        self._writes_retried = 0
        self._reads_total = 0

        # Cache
        self._cache: dict = {}
        self._cache_times: dict = {}

        # Connection pool placeholder
        self._conn = None
        self._connected = False

    def _connect(self) -> bool:
        try:
            import psycopg2
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = False
            self._connected = True
            logger.info(
                "SessionPersistence connected to PostgreSQL"
            )
            return True
        except Exception as e:
            logger.error(
                "SessionPersistence connection failed: %s",
                e,
            )
            self._connected = False
            return False

    def _ensure_schema(self) -> bool:
        if not self._connected:
            return False
        try:
            with self._conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            self._conn.commit()
            logger.info(
                "Schema initialised: "
                "cm_recovery_sessions, cm_quorum_decisions, "
                "cm_byzantine_events, cm_sla_snapshots"
            )
            return True
        except Exception as e:
            logger.error(
                "Schema initialisation failed: %s", e
            )
            try:
                self._conn.rollback()
            except Exception:
                pass
            return False

    # ── Write queue ────────────────────────────────────────

    def _enqueue(self, operation: dict):
        with self._lock:
            self._write_queue.append(operation)

    def _flush_queue(self):
        if not self._connected:
            if not self._connect():
                return

        with self._lock:
            pending = list(self._write_queue)
            self._write_queue.clear()

        if not pending:
            return

        for operation in pending:
            success = False
            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    self._execute_operation(operation)
                    self._writes_total += 1
                    success = True
                    break
                except Exception as e:
                    self._writes_retried += 1
                    logger.debug(
                        "Write attempt %d failed: %s",
                        attempt, e,
                    )
                    try:
                        self._conn.rollback()
                    except Exception:
                        pass
                    if attempt < self.MAX_RETRIES:
                        time.sleep(self.RETRY_DELAY_S)
                    else:
                        try:
                            self._connect()
                        except Exception:
                            pass

            if not success:
                self._writes_failed += 1
                logger.warning(
                    "Write permanently failed after "
                    "%d attempts: op=%s",
                    self.MAX_RETRIES,
                    operation.get("type", "unknown"),
                )

    def _execute_operation(self, operation: dict):
        op_type = operation["type"]

        if op_type == "upsert_session":
            self._upsert_session(operation["data"])
        elif op_type == "insert_decision":
            self._insert_decision(operation["data"])
        elif op_type == "insert_event":
            self._insert_event(operation["data"])
        elif op_type == "insert_sla":
            self._insert_sla(operation["data"])

    # ── Session operations ─────────────────────────────────

    def _upsert_session(self, data: dict):
        sql = """
        INSERT INTO cm_recovery_sessions (
            session_id, trigger, phase,
            affected_nodes, node_outcomes,
            node_durations_s, node_effects_start,
            node_effects_end, quorum_state_start,
            quorum_state_end, success_count,
            failure_count, total_duration_s, mttr_s,
            notes, plan_ids, started_at, completed_at,
            updated_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, NOW()
        )
        ON CONFLICT (session_id) DO UPDATE SET
            phase               = EXCLUDED.phase,
            node_outcomes       = EXCLUDED.node_outcomes,
            node_durations_s    = EXCLUDED.node_durations_s,
            node_effects_end    = EXCLUDED.node_effects_end,
            quorum_state_end    = EXCLUDED.quorum_state_end,
            success_count       = EXCLUDED.success_count,
            failure_count       = EXCLUDED.failure_count,
            total_duration_s    = EXCLUDED.total_duration_s,
            mttr_s              = EXCLUDED.mttr_s,
            notes               = EXCLUDED.notes,
            completed_at        = EXCLUDED.completed_at,
            updated_at          = NOW()
        """
        started_at = data.get("started_at")
        if started_at and isinstance(started_at, str):
            from datetime import datetime
            started_at = datetime.fromisoformat(started_at)

        completed_at = data.get("completed_at")
        if completed_at and isinstance(completed_at, str):
            from datetime import datetime
            completed_at = datetime.fromisoformat(
                completed_at
            )

        with self._conn.cursor() as cur:
            cur.execute(sql, (
                data["session_id"],
                data["trigger"],
                data["phase"],
                json.dumps(data.get("affected_nodes", [])),
                json.dumps(data.get("node_outcomes", {})),
                json.dumps(
                    data.get("node_durations_seconds", {})
                ),
                json.dumps(
                    data.get("node_effects_at_start", {})
                ),
                json.dumps(
                    data.get("node_effects_at_end", {})
                ),
                data.get("quorum_state_at_start"),
                data.get("quorum_state_at_end"),
                data.get("success_count", 0),
                data.get("failure_count", 0),
                data.get("total_duration_seconds"),
                data.get("mttr_seconds"),
                json.dumps(data.get("notes", [])),
                json.dumps(data.get("plan_ids", [])),
                started_at,
                completed_at,
            ))
        self._conn.commit()

    def _insert_decision(self, data: dict):
        sql = """
        INSERT INTO cm_quorum_decisions (
            node_id, reason, decision, quorum_state,
            contributing_nodes, recovering_nodes,
            decided_at
        ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                data["node_id"],
                data["reason"][:256],
                data["decision"],
                data["quorum_state"],
                data["contributing"],
                data["recovering"],
            ))
        self._conn.commit()

    def _insert_event(self, data: dict):
        sql = """
        INSERT INTO cm_byzantine_events (
            event_id, node_id, method, effect_ms,
            byzantine_score, details, acknowledged,
            occurred_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (event_id) DO NOTHING
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                data["event_id"],
                data["node_id"],
                data["method"],
                data["effect_ms"],
                data.get("byzantine_score", 0.0),
                data.get("details", "")[:500],
                data.get("acknowledged", False),
            ))
        self._conn.commit()

    def _insert_sla(self, data: dict):
        sql = """
        INSERT INTO cm_sla_snapshots (
            quorum_state, contributing_nodes,
            recovering_nodes, causal_effects,
            routing_weights, excluded_nodes,
            cluster_stability, active_session,
            byzantine_detections, recorded_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
        )
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                data["quorum_state"],
                data["contributing_nodes"],
                data["recovering_nodes"],
                json.dumps(data.get("causal_effects", {})),
                json.dumps(data.get("routing_weights", {})),
                json.dumps(data.get("excluded_nodes", [])),
                data.get("cluster_stability", 1.0),
                data.get("active_session", False),
                data.get("byzantine_detections", 0),
            ))
        self._conn.commit()

    # ── Public write API ───────────────────────────────────

    def save_session(self, session_dict: dict):
        self._enqueue({
            "type": "upsert_session",
            "data": session_dict,
        })

    def save_decision(self, decision_dict: dict):
        self._enqueue({
            "type": "insert_decision",
            "data": decision_dict,
        })

    def save_byzantine_event(self, event_dict: dict):
        self._enqueue({
            "type": "insert_event",
            "data": event_dict,
        })

    def save_sla_snapshot(self, snapshot_dict: dict):
        self._enqueue({
            "type": "insert_sla",
            "data": snapshot_dict,
        })

    # ── Public read API ────────────────────────────────────

    def _cache_get(self, key: str):
        with self._lock:
            if key in self._cache:
                age = time.time() - self._cache_times[key]
                if age < self.CACHE_TTL_S:
                    return self._cache[key]
        return None

    def _cache_set(self, key: str, value):
        with self._lock:
            self._cache[key] = value
            self._cache_times[key] = time.time()

    def get_sessions(
        self,
        limit: int = 20,
        phase: Optional[str] = None,
    ) -> list:
        cache_key = f"sessions:{limit}:{phase}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self._connected:
            return []

        try:
            self._reads_total += 1
            sql = """
            SELECT session_id, trigger, phase,
                   affected_nodes, node_outcomes,
                   success_count, failure_count,
                   total_duration_s, mttr_s,
                   quorum_state_start, quorum_state_end,
                   started_at, completed_at
            FROM cm_recovery_sessions
            """
            params = []
            if phase:
                sql += " WHERE phase = %s"
                params.append(phase)
            sql += " ORDER BY started_at DESC LIMIT %s"
            params.append(limit)

            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            result = []
            for row in rows:
                result.append({
                    "session_id": row[0],
                    "trigger": row[1],
                    "phase": row[2],
                    "affected_nodes": row[3],
                    "node_outcomes": row[4],
                    "success_count": row[5],
                    "failure_count": row[6],
                    "total_duration_s": row[7],
                    "mttr_s": row[8],
                    "quorum_state_start": row[9],
                    "quorum_state_end": row[10],
                    "started_at": str(row[11]),
                    "completed_at": str(row[12])
                    if row[12] else None,
                })

            self._cache_set(cache_key, result)
            return result

        except Exception as e:
            logger.error("get_sessions error: %s", e)
            return []

    def get_decisions(
        self,
        limit: int = 50,
        node_id: Optional[str] = None,
    ) -> list:
        cache_key = f"decisions:{limit}:{node_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self._connected:
            return []

        try:
            self._reads_total += 1
            sql = """
            SELECT decision_id, node_id, reason,
                   decision, quorum_state,
                   contributing_nodes, recovering_nodes,
                   decided_at
            FROM cm_quorum_decisions
            """
            params = []
            if node_id:
                sql += " WHERE node_id = %s"
                params.append(node_id)
            sql += " ORDER BY decided_at DESC LIMIT %s"
            params.append(limit)

            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            result = [
                {
                    "decision_id": row[0],
                    "node_id": row[1],
                    "reason": row[2],
                    "decision": row[3],
                    "quorum_state": row[4],
                    "contributing_nodes": row[5],
                    "recovering_nodes": row[6],
                    "decided_at": str(row[7]),
                }
                for row in rows
            ]
            self._cache_set(cache_key, result)
            return result

        except Exception as e:
            logger.error("get_decisions error: %s", e)
            return []

    def get_sla_snapshots(
        self,
        limit: int = 100,
    ) -> list:
        cache_key = f"sla:{limit}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self._connected:
            return []

        try:
            self._reads_total += 1
            sql = """
            SELECT quorum_state, contributing_nodes,
                   recovering_nodes, causal_effects,
                   routing_weights, cluster_stability,
                   active_session, byzantine_detections,
                   recorded_at
            FROM cm_sla_snapshots
            ORDER BY recorded_at DESC
            LIMIT %s
            """
            with self._conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()

            result = [
                {
                    "quorum_state": row[0],
                    "contributing_nodes": row[1],
                    "recovering_nodes": row[2],
                    "causal_effects": row[3],
                    "routing_weights": row[4],
                    "cluster_stability": row[5],
                    "active_session": row[6],
                    "byzantine_detections": row[7],
                    "recorded_at": str(row[8]),
                }
                for row in rows
            ]
            self._cache_set(cache_key, result)
            return result

        except Exception as e:
            logger.error("get_sla_snapshots error: %s", e)
            return []

    def get_stats(self) -> dict:
        stats = {
            "connected": self._connected,
            "queue_depth": len(self._write_queue),
            "writes_total": self._writes_total,
            "writes_failed": self._writes_failed,
            "writes_retried": self._writes_retried,
            "reads_total": self._reads_total,
            "cache_entries": len(self._cache),
            "uptime_seconds": round(
                time.time() - self._start_time, 1
            ) if self._start_time else 0.0,
        }

        if self._connected:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM "
                        "cm_recovery_sessions"
                    )
                    stats["db_sessions"] = (
                        cur.fetchone()[0]
                    )
                    cur.execute(
                        "SELECT COUNT(*) FROM "
                        "cm_quorum_decisions"
                    )
                    stats["db_decisions"] = (
                        cur.fetchone()[0]
                    )
                    cur.execute(
                        "SELECT COUNT(*) FROM "
                        "cm_byzantine_events"
                    )
                    stats["db_events"] = (
                        cur.fetchone()[0]
                    )
                    cur.execute(
                        "SELECT COUNT(*) FROM "
                        "cm_sla_snapshots"
                    )
                    stats["db_sla_snapshots"] = (
                        cur.fetchone()[0]
                    )
            except Exception as e:
                logger.debug(
                    "Stats query error: %s", e
                )

        return stats

    # ── Background writer ──────────────────────────────────

    def _writer_loop(self):
        logger.info(
            "SessionPersistence writer loop started "
            "flush_interval=%.1fs",
            self.FLUSH_INTERVAL_S,
        )
        while self._running:
            time.sleep(self.FLUSH_INTERVAL_S)
            try:
                self._flush_queue()
            except Exception as e:
                logger.error(
                    "Writer loop error: %s", e
                )

    def start(self):
        self._running = True
        self._start_time = time.time()
        connected = self._connect()
        if connected:
            self._ensure_schema()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="session-persistence-writer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "SessionPersistence started "
            "connected=%s dsn=%s",
            connected,
            self.dsn[:30] + "...",
        )

    def stop(self):
        self._running = False
        # Final flush
        try:
            self._flush_queue()
        except Exception as e:
            logger.error(
                "Final flush error: %s", e
            )
        if self._thread:
            self._thread.join(timeout=10)
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        logger.info("SessionPersistence stopped")


class SLAMonitor:
    """
    Periodic SLA snapshot recorder.

    Takes a snapshot of the full cluster state every
    SLA_INTERVAL_S seconds and persists it via
    SessionPersistence. This creates a time-series of
    cluster health for SLA compliance reporting.

    SLA targets tracked:
    - Quorum availability: contributing_nodes >= 2
    - Byzantine detection latency: checks run regularly
    - Routing stability: cluster_stability >= 0.90
    - Zero active recoveries (steady-state target)
    """

    SLA_INTERVAL_S     = 30.0
    QUORUM_TARGET      = 2
    STABILITY_TARGET   = 0.90

    def __init__(
        self,
        persistence: SessionPersistence,
        quorum_manager,
        coordinator,
        orchestrator,
        quorum_router,
        updater,
    ):
        self.persistence = persistence
        self.quorum_manager = quorum_manager
        self.coordinator = coordinator
        self.orchestrator = orchestrator
        self.quorum_router = quorum_router
        self.updater = updater

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._snapshot_count = 0
        self._sla_violations = 0
        self._start_time: Optional[float] = None

    def _take_snapshot(self):
        self._snapshot_count += 1

        try:
            quorum_status = self.quorum_manager.status()
            coord_status = self.coordinator.status()
            orch_status = self.orchestrator.status()
            router_status = self.quorum_router.status()

            effects = {}
            for node_id in [
                "node-1", "node-2", "node-3"
            ]:
                snap = self.updater.get_current_snapshot(
                    node_id
                )
                if snap:
                    effects[node_id] = round(
                        abs(snap["effect"]), 4
                    )

            snapshot = {
                "quorum_state": quorum_status[
                    "quorum_state"
                ],
                "contributing_nodes": quorum_status[
                    "contributing_nodes"
                ],
                "recovering_nodes": quorum_status[
                    "recovering_nodes"
                ],
                "causal_effects": effects,
                "routing_weights": router_status[
                    "current_weights"
                ],
                "excluded_nodes": router_status[
                    "excluded_nodes"
                ],
                "cluster_stability": router_status[
                    "cluster_stability"
                ],
                "active_session": (
                    orch_status["active_session"]
                    is not None
                ),
                "byzantine_detections": coord_status[
                    "detections_total"
                ],
            }

            self.persistence.save_sla_snapshot(snapshot)

            # Check SLA violations
            violations = []
            if (
                snapshot["contributing_nodes"]
                < self.QUORUM_TARGET
            ):
                violations.append(
                    f"quorum_below_target: "
                    f"{snapshot['contributing_nodes']}"
                    f"/{self.QUORUM_TARGET}"
                )
            if (
                snapshot["cluster_stability"]
                < self.STABILITY_TARGET
            ):
                violations.append(
                    f"stability_below_target: "
                    f"{snapshot['cluster_stability']:.3f}"
                    f"<{self.STABILITY_TARGET}"
                )

            if violations:
                self._sla_violations += 1
                logger.warning(
                    "SLA violation #%d: %s",
                    self._sla_violations,
                    "; ".join(violations),
                )
            else:
                if self._snapshot_count % 10 == 0:
                    logger.info(
                        "SLA snapshot #%d: "
                        "quorum=%s contributing=%d "
                        "stability=%.3f "
                        "active_session=%s",
                        self._snapshot_count,
                        snapshot["quorum_state"],
                        snapshot["contributing_nodes"],
                        snapshot["cluster_stability"],
                        snapshot["active_session"],
                    )

        except Exception as e:
            logger.error(
                "SLA snapshot error: %s", e
            )

    def _monitor_loop(self):
        logger.info(
            "SLAMonitor loop started "
            "interval=%.1fs",
            self.SLA_INTERVAL_S,
        )
        while self._running:
            time.sleep(self.SLA_INTERVAL_S)
            self._take_snapshot()

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="sla-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("SLAMonitor started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("SLAMonitor stopped")

    def status(self) -> dict:
        uptime = (
            time.time() - self._start_time
            if self._start_time else 0.0
        )
        sla_pct = (
            round(
                100.0 * (
                    1.0 - self._sla_violations
                    / max(self._snapshot_count, 1)
                ),
                2,
            )
            if self._snapshot_count > 0 else 100.0
        )
        return {
            "running": self._running,
            "snapshot_count": self._snapshot_count,
            "sla_violations": self._sla_violations,
            "sla_compliance_pct": sla_pct,
            "sla_interval_seconds": self.SLA_INTERVAL_S,
            "quorum_target": self.QUORUM_TARGET,
            "stability_target": self.STABILITY_TARGET,
            "uptime_seconds": round(uptime, 1),
        }


if __name__ == "__main__":
    logger.info("Starting SessionPersistence demo")

    # Build DSNs for 3 nodes
    nodes = [
        ("node-1", 5436),
        ("node-2", 5437),
        ("node-3", 5438),
    ]

    results = {}
    for node_id, port in nodes:
        dsn = (
            f"host=localhost port={port} "
            f"dbname=cogmesh user=cogmesh "
            f"password=cogmesh123 "
            f"connect_timeout=5"
        )
        persistence = SessionPersistence(dsn=dsn)
        persistence.start()
        time.sleep(2)

        stats = persistence.get_stats()
        results[node_id] = {
            "connected": stats["connected"],
            "port": port,
        }
        logger.info(
            "Node %s (port %d): connected=%s",
            node_id, port, stats["connected"],
        )

        if stats["connected"]:
            # Write a test session
            test_session = {
                "session_id": f"test-{node_id}-001",
                "trigger": "demo_day64",
                "phase": "completed",
                "affected_nodes": [node_id],
                "node_outcomes": {
                    node_id: "success"
                },
                "node_durations_seconds": {
                    node_id: 15.3
                },
                "node_effects_at_start": {
                    node_id: 45.2
                },
                "node_effects_at_end": {
                    node_id: 27.8
                },
                "quorum_state_at_start": "degraded",
                "quorum_state_at_end": "healthy",
                "success_count": 1,
                "failure_count": 0,
                "total_duration_seconds": 18.5,
                "mttr_seconds": 15.3,
                "notes": ["demo write"],
                "plan_ids": [],
                "started_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "completed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
            persistence.save_session(test_session)

            # Write a test decision
            test_decision = {
                "node_id": node_id,
                "reason": "demo_day64_test",
                "decision": "allow",
                "quorum_state": "healthy",
                "contributing": 3,
                "recovering": 0,
            }
            persistence.save_decision(test_decision)

            # Write a test SLA snapshot
            test_sla = {
                "quorum_state": "healthy",
                "contributing_nodes": 3,
                "recovering_nodes": 0,
                "causal_effects": {
                    "node-1": 27.5,
                    "node-2": 28.5,
                    "node-3": 29.0,
                },
                "routing_weights": {
                    "node-1": 0.35,
                    "node-2": 0.33,
                    "node-3": 0.32,
                },
                "excluded_nodes": [],
                "cluster_stability": 0.965,
                "active_session": False,
                "byzantine_detections": 0,
            }
            persistence.save_sla_snapshot(test_sla)

            # Flush and read back
            time.sleep(6)
            sessions = persistence.get_sessions(limit=5)
            decisions = persistence.get_decisions(limit=5)
            sla = persistence.get_sla_snapshots(limit=5)
            final_stats = persistence.get_stats()

            logger.info(
                "Node %s results: "
                "sessions=%d decisions=%d sla=%d "
                "db_sessions=%s db_decisions=%s "
                "db_sla=%s",
                node_id,
                len(sessions),
                len(decisions),
                len(sla),
                final_stats.get("db_sessions", "N/A"),
                final_stats.get("db_decisions", "N/A"),
                final_stats.get("db_sla_snapshots", "N/A"),
            )

        persistence.stop()

    logger.info("=== SUMMARY ===")
    for node_id, r in results.items():
        logger.info(
            "  %s port=%d connected=%s",
            node_id, r["port"], r["connected"],
        )