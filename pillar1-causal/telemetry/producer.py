import json
import logging
import time
from datetime import datetime, timezone

from kafka import KafkaProducer
from dotenv import load_dotenv
import os

from collector import TelemetryCollectorManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.telemetry.producer")


class TelemetryProducer:
    def __init__(self):
        self.bootstrap_servers = os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9093"
        )
        self.topics = {
            "query": os.getenv("KAFKA_TOPIC_QUERY_EVENTS", "cm.query.events"),
            "lock":  os.getenv("KAFKA_TOPIC_LOCK_EVENTS",  "cm.lock.events"),
            "io":    os.getenv("KAFKA_TOPIC_IO_EVENTS",    "cm.io.events"),
            "node":  os.getenv("KAFKA_TOPIC_NODE_EVENTS",  "cm.node.events"),
        }
        self.producer = self._create_producer()
        logger.info(
            "TelemetryProducer connected to Kafka at %s",
            self.bootstrap_servers
        )

    def _create_producer(self) -> KafkaProducer:
        return KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
            retry_backoff_ms=500,
            request_timeout_ms=30000,
            max_block_ms=10000,
            api_version=(2, 5, 0),
        )

    def _on_send_success(self, record_metadata):
        logger.debug(
            "topic=%s partition=%d offset=%d",
            record_metadata.topic,
            record_metadata.partition,
            record_metadata.offset,
        )

    def _on_send_error(self, exception):
        logger.error("Failed to publish event error=%s", exception)

    def publish_event(self, event_type: str, node_id: str, payload: dict):
        topic = self.topics.get(event_type)
        if not topic:
            logger.warning("Unknown event_type=%s skipping", event_type)
            return

        payload["published_at"] = datetime.now(timezone.utc).isoformat()

        self.producer.send(
            topic=topic,
            key=node_id,
            value=payload,
        ).add_callback(
            self._on_send_success
        ).add_errback(
            self._on_send_error
        )

    def publish_node_heartbeat(self, node_id: str, event_counts: dict):
        payload = {
            "event_type": "heartbeat",
            "node_id": node_id,
            "query_event_count": event_counts.get("query_events", 0),
            "lock_event_count": event_counts.get("lock_events", 0),
            "io_event_count": event_counts.get("io_events", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.publish_event("node", node_id, payload)

    def flush(self):
        self.producer.flush()

    def close(self):
        self.producer.flush()
        self.producer.close()
        logger.info("TelemetryProducer closed")


class TelemetryPipeline:
    def __init__(self, collection_interval_seconds: int = 5):
        self.collection_interval = collection_interval_seconds
        self.collector_manager = TelemetryCollectorManager()
        self.producer = TelemetryProducer()
        self._published_counts = {
            "query": 0,
            "lock": 0,
            "io": 0,
            "node": 0,
        }

    def _publish_node_events(self, node_id: str, events: dict):
        for event in events.get("query_events", []):
            self.producer.publish_event("query", node_id, event)
            self._published_counts["query"] += 1

        for event in events.get("lock_events", []):
            self.producer.publish_event("lock", node_id, event)
            self._published_counts["lock"] += 1

        for event in events.get("io_events", []):
            self.producer.publish_event("io", node_id, event)
            self._published_counts["io"] += 1

        self.producer.publish_node_heartbeat(node_id, {
            "query_events": len(events.get("query_events", [])),
            "lock_events":  len(events.get("lock_events", [])),
            "io_events":    len(events.get("io_events", [])),
        })
        self._published_counts["node"] += 1

    def _log_pipeline_stats(self):
        logger.info(
            "pipeline_stats published query=%d lock=%d io=%d heartbeat=%d",
            self._published_counts["query"],
            self._published_counts["lock"],
            self._published_counts["io"],
            self._published_counts["node"],
        )

    def run(self):
        logger.info(
            "Starting telemetry pipeline with collection_interval=%ds",
            self.collection_interval
        )
        try:
            while True:
                cycle_start = time.monotonic()
                all_node_events = self.collector_manager.collect_all_nodes()
                for node_id, events in all_node_events.items():
                    self._publish_node_events(node_id, events)
                self.producer.flush()
                self._log_pipeline_stats()
                elapsed = time.monotonic() - cycle_start
                sleep_time = max(0, self.collection_interval - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted, shutting down")
        finally:
            self.producer.close()
            self.collector_manager.close_all()
            logger.info("Pipeline shut down cleanly")


if __name__ == "__main__":
    pipeline = TelemetryPipeline(collection_interval_seconds=5)
    pipeline.run()