import logging
import random
import threading
import time
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
logger = logging.getLogger("cm.load.generator")


NODE_CONFIGS = [
    {
        "node_id": "node-1",
        "host": os.getenv("PG_NODE1_HOST", "127.0.0.1"),
        "port": int(os.getenv("PG_NODE1_PORT", "5436")),
        "dbname": "cm_node1",
    },
    {
        "node_id": "node-2",
        "host": os.getenv("PG_NODE2_HOST", "127.0.0.1"),
        "port": int(os.getenv("PG_NODE2_PORT", "5437")),
        "dbname": "cm_node2",
    },
    {
        "node_id": "node-3",
        "host": os.getenv("PG_NODE3_HOST", "127.0.0.1"),
        "port": int(os.getenv("PG_NODE3_PORT", "5438")),
        "dbname": "cm_node3",
    },
]

QUERY_TEMPLATES = [
    "SELECT count(*) FROM cm_workload_test WHERE node_id = '{node_id}';",
    "SELECT id, payload FROM cm_workload_test WHERE node_id = '{node_id}' ORDER BY created_at DESC LIMIT 10;",
    "SELECT node_id, count(*) FROM cm_workload_test GROUP BY node_id;",
    "SELECT * FROM cm_workload_test WHERE created_at > NOW() - INTERVAL '1 hour' LIMIT 50;",
    "SELECT pg_sleep(0.01);",
    "SELECT count(*) FROM pg_stat_activity;",
    "SELECT * FROM pg_locks LIMIT 20;",
]

INSERT_TEMPLATE = (
    "INSERT INTO cm_workload_test (node_id, payload) "
    "VALUES ('{node_id}', '{payload}');"
)


class NodeWorker:
    def __init__(self, config: dict, worker_id: int):
        self.config = config
        self.worker_id = worker_id
        self.node_id = config["node_id"]
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.queries_executed = 0
        self.errors = 0
        self._running = False

    def _connect(self):
        self.connection = psycopg2.connect(
            host=self.config["host"],
            port=self.config["port"],
            dbname=self.config["dbname"],
            user=os.getenv("PG_USER", "cm_user"),
            password=os.getenv("PG_PASSWORD", "cm_secret"),
            connect_timeout=5,
            options=f"-c application_name=cm_worker_{self.worker_id}",
        )
        self.connection.autocommit = True

    def _execute_random_query(self):
        if random.random() < 0.3:
            payload = f"load_test_{self.worker_id}_{random.randint(1000, 9999)}"
            query = INSERT_TEMPLATE.format(
                node_id=self.node_id,
                payload=payload
            )
        else:
            template = random.choice(QUERY_TEMPLATES)
            query = template.format(node_id=self.node_id)

        with self.connection.cursor() as cur:
            cur.execute(query)
            try:
                cur.fetchall()
            except psycopg2.ProgrammingError:
                pass

    def run(self, duration_seconds: int, queries_per_second: float):
        self._connect()
        self._running = True
        end_time = time.monotonic() + duration_seconds
        interval = 1.0 / queries_per_second

        logger.debug(
            "Worker %d started on %s qps=%.1f",
            self.worker_id,
            self.node_id,
            queries_per_second
        )

        while self._running and time.monotonic() < end_time:
            try:
                start = time.monotonic()
                self._execute_random_query()
                self.queries_executed += 1
                elapsed = time.monotonic() - start
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)
            except psycopg2.Error as e:
                self.errors += 1
                logger.debug("Worker %d query error: %s", self.worker_id, e)
                try:
                    self._connect()
                except Exception:
                    time.sleep(0.5)

        if self.connection:
            self.connection.close()

    def stop(self):
        self._running = False


class LoadGenerator:
    def __init__(
        self,
        workers_per_node: int = 10,
        queries_per_second_per_worker: float = 5.0,
        duration_seconds: int = 120,
    ):
        self.workers_per_node = workers_per_node
        self.qps_per_worker = queries_per_second_per_worker
        self.duration_seconds = duration_seconds
        self.workers: list[NodeWorker] = []
        self.threads: list[threading.Thread] = []

    def _create_workers(self):
        worker_id = 0
        for config in NODE_CONFIGS:
            for _ in range(self.workers_per_node):
                worker = NodeWorker(config=config, worker_id=worker_id)
                self.workers.append(worker)
                worker_id += 1

    def run(self):
        self._create_workers()
        total_workers = len(self.workers)
        total_qps = total_workers * self.qps_per_worker

        logger.info(
            "Starting load test: workers=%d qps_total=%.0f duration=%ds",
            total_workers,
            total_qps,
            self.duration_seconds
        )
        logger.info(
            "Workers per node=%d nodes=%d",
            self.workers_per_node,
            len(NODE_CONFIGS)
        )

        start_time = time.monotonic()

        for worker in self.workers:
            thread = threading.Thread(
                target=worker.run,
                args=(self.duration_seconds, self.qps_per_worker),
                daemon=True,
            )
            self.threads.append(thread)
            thread.start()

        try:
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.duration_seconds:
                    break

                total_queries = sum(w.queries_executed for w in self.workers)
                total_errors = sum(w.errors for w in self.workers)
                actual_qps = total_queries / elapsed if elapsed > 0 else 0

                per_node = {}
                for config in NODE_CONFIGS:
                    node_workers = [
                        w for w in self.workers
                        if w.node_id == config["node_id"]
                    ]
                    per_node[config["node_id"]] = sum(
                        w.queries_executed for w in node_workers
                    )

                logger.info(
                    "elapsed=%.0fs total_queries=%d actual_qps=%.0f "
                    "errors=%d per_node=%s",
                    elapsed,
                    total_queries,
                    actual_qps,
                    total_errors,
                    per_node,
                )
                time.sleep(10)

        except KeyboardInterrupt:
            logger.info("Load test interrupted")
            for worker in self.workers:
                worker.stop()

        for thread in self.threads:
            thread.join(timeout=5)

        total_queries = sum(w.queries_executed for w in self.workers)
        total_errors = sum(w.errors for w in self.workers)
        elapsed = time.monotonic() - start_time

        logger.info("Load test complete")
        logger.info(
            "total_queries=%d total_errors=%d elapsed=%.1fs avg_qps=%.0f",
            total_queries,
            total_errors,
            elapsed,
            total_queries / elapsed if elapsed > 0 else 0,
        )

        per_node_final = {}
        for config in NODE_CONFIGS:
            node_workers = [
                w for w in self.workers
                if w.node_id == config["node_id"]
            ]
            per_node_final[config["node_id"]] = sum(
                w.queries_executed for w in node_workers
            )
        logger.info("per_node_final=%s", per_node_final)


if __name__ == "__main__":
    generator = LoadGenerator(
        workers_per_node=5,
        queries_per_second_per_worker=3.0,
        duration_seconds=120,
    )
    generator.run()