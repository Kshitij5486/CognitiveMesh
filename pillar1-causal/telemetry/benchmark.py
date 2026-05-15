import logging
import statistics
import time
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.benchmark")

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


def connect(config: dict) -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=config["host"],
        port=config["port"],
        dbname=config["dbname"],
        user=os.getenv("PG_USER", "cm_user"),
        password=os.getenv("PG_PASSWORD", "cm_secret"),
        connect_timeout=5,
    )


def measure_query_latency(
    conn: psycopg2.extensions.connection,
    query: str,
    iterations: int = 100,
) -> dict:
    latencies = []
    with conn.cursor() as cur:
        for _ in range(iterations):
            start = time.perf_counter()
            cur.execute(query)
            try:
                cur.fetchall()
            except psycopg2.ProgrammingError:
                pass
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
    return {
        "min_ms":    round(min(latencies), 3),
        "max_ms":    round(max(latencies), 3),
        "mean_ms":   round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "p95_ms":    round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        "p99_ms":    round(sorted(latencies)[int(len(latencies) * 0.99)], 3),
        "stddev_ms": round(statistics.stdev(latencies), 3),
        "iterations": iterations,
    }


def measure_telemetry_overhead(
    conn: psycopg2.extensions.connection,
    node_id: str,
) -> dict:
    baseline_query = "SELECT 1;"
    telemetry_query = """
        SELECT
            pid,
            md5(query) AS query_hash,
            state,
            EXTRACT(EPOCH FROM (now() - query_start)) * 1000 AS duration_ms,
            wait_event,
            wait_event_type
        FROM pg_stat_activity
        WHERE state IS NOT NULL
          AND pid <> pg_backend_pid()
          AND query_start IS NOT NULL;
    """
    lock_query = """
        SELECT l.pid, l.locktype, l.mode, l.granted
        FROM pg_locks l
        WHERE l.pid <> pg_backend_pid();
    """
    io_query = """
        SELECT checkpoints_timed, checkpoints_req,
               buffers_clean, buffers_backend, buffers_alloc
        FROM pg_stat_bgwriter;
    """

    logger.info("Benchmarking node=%s", node_id)

    baseline = measure_query_latency(conn, baseline_query, iterations=200)
    logger.info("baseline SELECT 1: mean=%.3fms p95=%.3fms", baseline["mean_ms"], baseline["p95_ms"])

    telemetry = measure_query_latency(conn, telemetry_query, iterations=200)
    logger.info("pg_stat_activity query: mean=%.3fms p95=%.3fms", telemetry["mean_ms"], telemetry["p95_ms"])

    lock = measure_query_latency(conn, lock_query, iterations=200)
    logger.info("pg_locks query: mean=%.3fms p95=%.3fms", lock["mean_ms"], lock["p95_ms"])

    io = measure_query_latency(conn, io_query, iterations=200)
    logger.info("pg_stat_bgwriter query: mean=%.3fms p95=%.3fms", io["mean_ms"], io["p95_ms"])

    overhead_ms = (
        telemetry["mean_ms"] + lock["mean_ms"] + io["mean_ms"]
    ) - baseline["mean_ms"]

    return {
        "node_id": node_id,
        "baseline_select1": baseline,
        "telemetry_pg_stat_activity": telemetry,
        "telemetry_pg_locks": lock,
        "telemetry_pg_stat_bgwriter": io,
        "total_overhead_per_cycle_ms": round(overhead_ms, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def measure_kafka_pipeline_latency() -> dict:
    import requests
    logger.info("Measuring Kafka pipeline end-to-end latency")

    latencies = []
    for _ in range(10):
        start = time.perf_counter()
        try:
            response = requests.get(
                "http://localhost:8080/events/node-1",
                params={"event_type": "io", "last_n": 1},
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                if data["count"] > 0:
                    event = data["events"][0]
                    published_at = event.get("published_at")
                    received_at = event.get("received_at")
                    if published_at and received_at:
                        from datetime import datetime
                        pub = datetime.fromisoformat(published_at)
                        rec = datetime.fromisoformat(received_at)
                        kafka_latency_ms = (rec - pub).total_seconds() * 1000
                        latencies.append(kafka_latency_ms)
        except Exception as e:
            logger.warning("API call failed: %s", e)
        elapsed_ms = (time.perf_counter() - start) * 1000
        time.sleep(0.5)

    if latencies:
        return {
            "kafka_publish_to_consume_mean_ms": round(statistics.mean(latencies), 3),
            "kafka_publish_to_consume_min_ms":  round(min(latencies), 3),
            "kafka_publish_to_consume_max_ms":  round(max(latencies), 3),
            "samples": len(latencies),
        }
    return {"error": "no latency samples collected"}


def run_full_benchmark():
    logger.info("Starting CognitiveMesh Sprint 1 benchmark")
    logger.info("Timestamp: %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    results = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "nodes": [],
        "kafka_pipeline": {},
        "summary": {},
    }

    for config in NODE_CONFIGS:
        conn = connect(config)
        conn.autocommit = True
        node_result = measure_telemetry_overhead(conn, config["node_id"])
        results["nodes"].append(node_result)
        conn.close()

    logger.info("=" * 60)
    logger.info("Measuring Kafka pipeline latency")
    results["kafka_pipeline"] = measure_kafka_pipeline_latency()

    overhead_values = [
        n["total_overhead_per_cycle_ms"] for n in results["nodes"]
    ]
    results["summary"] = {
        "avg_telemetry_overhead_ms": round(statistics.mean(overhead_values), 3),
        "max_telemetry_overhead_ms": round(max(overhead_values), 3),
        "overhead_target_ms":        10.0,
        "overhead_within_target":    max(overhead_values) < 10.0,
        "kafka_pipeline":            results["kafka_pipeline"],
    }

    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 60)
    for node in results["nodes"]:
        logger.info(
            "node=%-8s  baseline=%.3fms  stat_activity=%.3fms  "
            "locks=%.3fms  bgwriter=%.3fms  overhead=%.3fms",
            node["node_id"],
            node["baseline_select1"]["mean_ms"],
            node["telemetry_pg_stat_activity"]["mean_ms"],
            node["telemetry_pg_locks"]["mean_ms"],
            node["telemetry_pg_stat_bgwriter"]["mean_ms"],
            node["total_overhead_per_cycle_ms"],
        )
    logger.info(
        "avg_overhead=%.3fms  max_overhead=%.3fms  within_target=%s",
        results["summary"]["avg_telemetry_overhead_ms"],
        results["summary"]["max_telemetry_overhead_ms"],
        results["summary"]["overhead_within_target"],
    )
    if "kafka_publish_to_consume_mean_ms" in results["kafka_pipeline"]:
        logger.info(
            "kafka_latency mean=%.3fms min=%.3fms max=%.3fms samples=%d",
            results["kafka_pipeline"]["kafka_publish_to_consume_mean_ms"],
            results["kafka_pipeline"]["kafka_publish_to_consume_min_ms"],
            results["kafka_pipeline"]["kafka_publish_to_consume_max_ms"],
            results["kafka_pipeline"]["samples"],
        )
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_full_benchmark()