import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.telemetry.collector")


@dataclass
class QueryEvent:
    event_type: str
    node_id: str
    pid: int
    query_hash: Optional[str]
    query_state: str
    duration_ms: float
    wait_event: Optional[str]
    wait_event_type: Optional[str]
    application_name: str
    timestamp: str


@dataclass
class LockEvent:
    event_type: str
    node_id: str
    pid: int
    lock_type: str
    lock_mode: str
    lock_granted: bool
    relation: Optional[str]
    timestamp: str


@dataclass
class IOEvent:
    event_type: str
    node_id: str
    checkpoints_timed: int
    checkpoints_req: int
    buffers_clean: int
    buffers_backend: int
    buffers_alloc: int
    timestamp: str


class NodeTelemetryCollector:
    def __init__(self, node_id: str, host: str, port: int, dbname: str):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.dbname = dbname
        self.connection = None
        self._connect()

    def _connect(self):
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=os.getenv("PG_USER", "cm_user"),
                password=os.getenv("PG_PASSWORD", "cm_secret"),
                connect_timeout=5
            )
            self.connection.autocommit = True
            logger.info("Connected to node %s at %s:%d", self.node_id, self.host, self.port)
        except psycopg2.OperationalError as e:
            logger.error("Failed to connect to node %s: %s", self.node_id, e)
            raise

    def _ensure_connection(self):
        try:
            with self.connection.cursor() as cur:
                cur.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            logger.warning("Connection lost to node %s, reconnecting...", self.node_id)
            self._connect()

    def collect_query_events(self) -> list[QueryEvent]:
        self._ensure_connection()
        events = []
        query = """
            SELECT
                pid,
                md5(query) AS query_hash,
                state,
                EXTRACT(EPOCH FROM (now() - query_start)) * 1000 AS duration_ms,
                wait_event,
                wait_event_type,
                application_name
            FROM pg_stat_activity
            WHERE state IS NOT NULL
              AND pid <> pg_backend_pid()
              AND query_start IS NOT NULL
        """
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()
                for row in rows:
                    events.append(QueryEvent(
                        event_type="query",
                        node_id=self.node_id,
                        pid=row["pid"],
                        query_hash=row["query_hash"],
                        query_state=row["state"],
                        duration_ms=round(float(row["duration_ms"] or 0), 3),
                        wait_event=row["wait_event"],
                        wait_event_type=row["wait_event_type"],
                        application_name=row["application_name"] or "",
                        timestamp=datetime.now(timezone.utc).isoformat()
                    ))
        except psycopg2.Error as e:
            logger.error("Failed to collect query events from node %s: %s", self.node_id, e)
        return events

    def collect_lock_events(self) -> list[LockEvent]:
        self._ensure_connection()
        events = []
        query = """
            SELECT
                l.pid,
                l.locktype,
                l.mode,
                l.granted,
                c.relname AS relation
            FROM pg_locks l
            LEFT JOIN pg_class c ON l.relation = c.oid
            WHERE l.pid <> pg_backend_pid()
        """
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()
                for row in rows:
                    events.append(LockEvent(
                        event_type="lock",
                        node_id=self.node_id,
                        pid=row["pid"],
                        lock_type=row["locktype"],
                        lock_mode=row["mode"],
                        lock_granted=row["granted"],
                        relation=row["relation"],
                        timestamp=datetime.now(timezone.utc).isoformat()
                    ))
        except psycopg2.Error as e:
            logger.error("Failed to collect lock events from node %s: %s", self.node_id, e)
        return events

    def collect_io_events(self) -> list[IOEvent]:
        self._ensure_connection()
        events = []
        query = """
            SELECT
                checkpoints_timed,
                checkpoints_req,
                buffers_clean,
                buffers_backend,
                buffers_alloc
            FROM pg_stat_bgwriter
        """
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                row = cur.fetchone()
                if row:
                    events.append(IOEvent(
                        event_type="io",
                        node_id=self.node_id,
                        checkpoints_timed=row["checkpoints_timed"],
                        checkpoints_req=row["checkpoints_req"],
                        buffers_clean=row["buffers_clean"],
                        buffers_backend=row["buffers_backend"],
                        buffers_alloc=row["buffers_alloc"],
                        timestamp=datetime.now(timezone.utc).isoformat()
                    ))
        except psycopg2.Error as e:
            logger.error("Failed to collect IO events from node %s: %s", self.node_id, e)
        return events

    def collect_all(self) -> dict:
        return {
            "query_events": [asdict(e) for e in self.collect_query_events()],
            "lock_events": [asdict(e) for e in self.collect_lock_events()],
            "io_events": [asdict(e) for e in self.collect_io_events()]
        }

    def close(self):
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from node %s", self.node_id)


class TelemetryCollectorManager:
    def __init__(self):
        self.collectors = {
            "node-1": NodeTelemetryCollector(
                node_id="node-1",
                host=os.getenv("PG_NODE1_HOST", "localhost"),
                port=int(os.getenv("PG_NODE1_PORT", "5436")),
                dbname="cm_node1"
            ),
            "node-2": NodeTelemetryCollector(
                node_id="node-2",
                host=os.getenv("PG_NODE2_HOST", "localhost"),
                port=int(os.getenv("PG_NODE2_PORT", "5437")),
                dbname="cm_node2"
            ),
            "node-3": NodeTelemetryCollector(
                node_id="node-3",
                host=os.getenv("PG_NODE3_HOST", "localhost"),
                port=int(os.getenv("PG_NODE3_PORT", "5438")),
                dbname="cm_node3"
            )
        }
        logger.info("TelemetryCollectorManager initialized with %d nodes", len(self.collectors))

    def collect_all_nodes(self) -> dict:
        results = {}
        for node_id, collector in self.collectors.items():
            results[node_id] = collector.collect_all()
        return results

    def close_all(self):
        for collector in self.collectors.values():
            collector.close()


if __name__ == "__main__":
    manager = TelemetryCollectorManager()
    logger.info("Starting telemetry collection loop...")
    try:
        while True:
            data = manager.collect_all_nodes()
            for node_id, events in data.items():
                q = len(events["query_events"])
                l = len(events["lock_events"])
                i = len(events["io_events"])
                logger.info("node=%s query_events=%d lock_events=%d io_events=%d", node_id, q, l, i)
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Shutting down telemetry collector")
        manager.close_all()