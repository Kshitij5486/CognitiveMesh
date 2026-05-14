import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from kafka import KafkaConsumer
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.telemetry.consumer")


class EventStore:
    def __init__(self, max_events_per_node: int = 10000):
        self.max_events_per_node = max_events_per_node
        self._lock = threading.RLock()
        self._stores = {
            "query":     {},
            "lock":      {},
            "io":        {},
            "heartbeat": {},
        }

    def _ensure_node(self, store_key: str, node_id: str):
        if node_id not in self._stores[store_key]:
            self._stores[store_key][node_id] = deque(
                maxlen=self.max_events_per_node
            )

    def append(self, event_type: str, node_id: str, event: dict):
        store_key = event_type if event_type in self._stores else "query"
        with self._lock:
            self._ensure_node(store_key, node_id)
            event["received_at"] = datetime.now(timezone.utc).isoformat()
            self._stores[store_key][node_id].append(event)

    def get_events(
        self,
        event_type: str,
        node_id: str,
        last_n: Optional[int] = None
    ) -> list:
        store_key = event_type if event_type in self._stores else "query"
        with self._lock:
            if node_id not in self._stores[store_key]:
                return []
            events = list(self._stores[store_key][node_id])
            if last_n is not None:
                events = events[-last_n:]
            return events

    def get_all_nodes(self, event_type: str) -> list:
        store_key = event_type if event_type in self._stores else "query"
        with self._lock:
            return list(self._stores[store_key].keys())

    def get_counts(self) -> dict:
        with self._lock:
            return {
                event_type: {
                    node_id: len(events)
                    for node_id, events in store.items()
                }
                for event_type, store in self._stores.items()
            }


class TelemetryConsumer:
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.bootstrap_servers = os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9093"
        )
        self.topics = [
            os.getenv("KAFKA_TOPIC_QUERY_EVENTS", "cm.query.events"),
            os.getenv("KAFKA_TOPIC_LOCK_EVENTS",  "cm.lock.events"),
            os.getenv("KAFKA_TOPIC_IO_EVENTS",    "cm.io.events"),
            os.getenv("KAFKA_TOPIC_NODE_EVENTS",  "cm.node.events"),
        ]
        self._topic_to_type = {
            os.getenv("KAFKA_TOPIC_QUERY_EVENTS", "cm.query.events"): "query",
            os.getenv("KAFKA_TOPIC_LOCK_EVENTS",  "cm.lock.events"):  "lock",
            os.getenv("KAFKA_TOPIC_IO_EVENTS",    "cm.io.events"):    "io",
            os.getenv("KAFKA_TOPIC_NODE_EVENTS",  "cm.node.events"):  "heartbeat",
        }
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._consumer: Optional[KafkaConsumer] = None
        self._consumed_total = 0

    def _create_consumer(self) -> KafkaConsumer:
        return KafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            group_id="cm-telemetry-consumer-group",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            max_poll_records=500,
            api_version=(2, 5, 0),
        )

    def _consume_loop(self):
        logger.info(
            "Consumer thread started subscribing to topics=%s",
            self.topics
        )
        try:
            self._consumer = self._create_consumer()
            while self._running:
                records = self._consumer.poll(timeout_ms=1000)
                for topic_partition, messages in records.items():
                    event_type = self._topic_to_type.get(
                        topic_partition.topic, "query"
                    )
                    for message in messages:
                        node_id = message.key or "unknown"
                        event = message.value
                        self.event_store.append(event_type, node_id, event)
                        self._consumed_total += 1

                if self._consumed_total > 0 and self._consumed_total % 100 == 0:
                    logger.info(
                        "Consumer progress total_consumed=%d store_counts=%s",
                        self._consumed_total,
                        self.event_store.get_counts()
                    )
        except Exception as e:
            logger.error("Consumer thread error: %s", e)
        finally:
            if self._consumer:
                self._consumer.close()
                logger.info("Kafka consumer closed")

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._consume_loop,
            name="telemetry-consumer",
            daemon=True
        )
        self._thread.start()
        logger.info("TelemetryConsumer started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("TelemetryConsumer stopped")

    def get_total_consumed(self) -> int:
        return self._consumed_total


class EventQueryAPI:
    def __init__(self, event_store: EventStore):
        self.store = event_store

    def get_node_events(
        self,
        node_id: str,
        event_type: str = "io",
        last_n: int = 10
    ) -> dict:
        events = self.store.get_events(event_type, node_id, last_n)
        return {
            "node_id": node_id,
            "event_type": event_type,
            "count": len(events),
            "events": events,
        }

    def get_store_summary(self) -> dict:
        counts = self.store.get_counts()
        return {
            "summary": counts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_all_nodes_latest(self, event_type: str = "io") -> dict:
        nodes = self.store.get_all_nodes(event_type)
        result = {}
        for node_id in nodes:
            events = self.store.get_events(event_type, node_id, last_n=1)
            result[node_id] = events[0] if events else None
        return result


if __name__ == "__main__":
    store = EventStore(max_events_per_node=10000)
    consumer = TelemetryConsumer(event_store=store)
    query_api = EventQueryAPI(event_store=store)

    consumer.start()
    logger.info("Consumer running. Open a second terminal and run the producer.")

    try:
        cycle = 0
        while True:
            time.sleep(10)
            cycle += 1
            summary = query_api.get_store_summary()
            logger.info(
                "cycle=%d total_consumed=%d store=%s",
                cycle,
                consumer.get_total_consumed(),
                summary["summary"]
            )
            if cycle % 3 == 0:
                for node_id in ["node-1", "node-2", "node-3"]:
                    result = query_api.get_node_events(
                        node_id=node_id,
                        event_type="io",
                        last_n=2
                    )
                    if result["count"] > 0:
                        logger.info(
                            "latest io event node=%s event=%s",
                            node_id,
                            result["events"][-1]
                        )
    except KeyboardInterrupt:
        logger.info("Shutting down consumer")
    finally:
        consumer.stop()