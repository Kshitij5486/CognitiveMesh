import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "telemetry"))
sys.path.insert(0, os.path.dirname(__file__))

from collector import TelemetryCollectorManager
from graph_builder import NodeCausalModel, TelemetrySampler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.causal.streaming_updater")


class RollingTelemetryBuffer:
    def __init__(self, max_samples: int = 200, min_samples: int = 30):
        self.max_samples = max_samples
        self.min_samples = min_samples
        self._buffer: deque = deque(maxlen=max_samples)
        self._lock = threading.RLock()
        self._total_received = 0

    def append(self, sample: dict):
        with self._lock:
            self._buffer.append(sample)
            self._total_received += 1

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        with self._lock:
            if len(self._buffer) < self.min_samples:
                return None
            records = list(self._buffer)
        df = pd.DataFrame(records)
        df = df.drop(columns=["timestamp"], errors="ignore")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]
        return df

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def total_received(self) -> int:
        with self._lock:
            return self._total_received

    def is_ready(self) -> bool:
        with self._lock:
            return len(self._buffer) >= self.min_samples


class CausalModelSnapshot:
    def __init__(
        self,
        node_id: str,
        effect: float,
        timestamp: str,
        samples_used: int,
    ):
        self.node_id = node_id
        self.effect = effect
        self.timestamp = timestamp
        self.samples_used = samples_used

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "effect": self.effect,
            "timestamp": self.timestamp,
            "samples_used": self.samples_used,
        }


class StreamingCausalUpdater:
    def __init__(
        self,
        collection_interval_seconds: float = 3.0,
        retrain_interval_seconds: float = 30.0,
        min_samples_for_training: int = 30,
        max_buffer_size: int = 200,
    ):
        self.collection_interval = collection_interval_seconds
        self.retrain_interval = retrain_interval_seconds
        self.buffer = RollingTelemetryBuffer(
            max_samples=max_buffer_size,
            min_samples=min_samples_for_training,
        )
        self.sampler = TelemetrySampler()
        self._running = False
        self._collector_thread: Optional[threading.Thread] = None
        self._trainer_thread: Optional[threading.Thread] = None
        self._model_lock = threading.RLock()
        self._current_models: dict[str, NodeCausalModel] = {}
        self._current_snapshots: dict[str, CausalModelSnapshot] = {}
        self._retrain_count = 0
        self._last_retrain_time: Optional[str] = None
        self._status = "initializing"

    def _collect_loop(self):
        logger.info(
            "Collector loop started interval=%.1fs", self.collection_interval
        )
        while self._running:
            try:
                sample = self.sampler.collect_sample()
                self.buffer.append(sample)
                logger.debug(
                    "Sample collected buffer_size=%d total=%d",
                    self.buffer.size(),
                    self.buffer.total_received(),
                )
            except Exception as e:
                logger.error("Collection error: %s", e)
            time.sleep(self.collection_interval)

    def _train_models(self, df: pd.DataFrame) -> dict:
        results = {}
        for node_id in ["node-1", "node-2", "node-3"]:
            try:
                model = NodeCausalModel(node_id=node_id, dataframe=df)
                if not model.build():
                    continue
                if not model.identify():
                    continue
                effect = model.estimate_effect()
                if effect is None:
                    continue

                snapshot = CausalModelSnapshot(
                    node_id=node_id,
                    effect=effect,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    samples_used=len(df),
                )
                results[node_id] = (model, snapshot)
                logger.info(
                    "Model updated node=%s effect=%.4f samples=%d",
                    node_id,
                    effect,
                    len(df),
                )
            except Exception as e:
                logger.error(
                    "Training error node=%s error=%s", node_id, e
                )
        return results

    def _retrain_loop(self):
        logger.info(
            "Trainer loop started retrain_interval=%.1fs",
            self.retrain_interval,
        )
        while self._running:
            time.sleep(self.retrain_interval)
            if not self.buffer.is_ready():
                logger.info(
                    "Buffer not ready yet size=%d min=%d",
                    self.buffer.size(),
                    self.buffer.min_samples,
                )
                continue

            df = self.buffer.get_dataframe()
            if df is None:
                continue

            logger.info(
                "Starting retrain cycle=%d samples=%d",
                self._retrain_count + 1,
                len(df),
            )
            self._status = "retraining"

            try:
                start = time.monotonic()
                new_models = self._train_models(df)
                elapsed_ms = (time.monotonic() - start) * 1000

                with self._model_lock:
                    for node_id, (model, snapshot) in new_models.items():
                        self._current_models[node_id] = model
                        self._current_snapshots[node_id] = snapshot
                    self._retrain_count += 1
                    self._last_retrain_time = (
                        datetime.now(timezone.utc).isoformat()
                    )

                self._status = "ready"
                logger.info(
                    "Retrain complete cycle=%d nodes=%d elapsed=%.0fms",
                    self._retrain_count,
                    len(new_models),
                    elapsed_ms,
                )
            except Exception as e:
                self._status = "error"
                logger.error("Retrain cycle failed: %s", e)

    def start(self):
        self._running = True
        self._status = "collecting"

        self._collector_thread = threading.Thread(
            target=self._collect_loop,
            name="causal-collector",
            daemon=True,
        )
        self._trainer_thread = threading.Thread(
            target=self._retrain_loop,
            name="causal-trainer",
            daemon=True,
        )
        self._collector_thread.start()
        self._trainer_thread.start()
        logger.info(
            "StreamingCausalUpdater started "
            "collection_interval=%.1fs retrain_interval=%.1fs",
            self.collection_interval,
            self.retrain_interval,
        )

    def stop(self):
        self._running = False
        if self._collector_thread:
            self._collector_thread.join(timeout=10)
        if self._trainer_thread:
            self._trainer_thread.join(timeout=10)
        self.sampler.close()
        logger.info("StreamingCausalUpdater stopped")

    def get_current_effect(self, node_id: str) -> Optional[float]:
        with self._model_lock:
            snapshot = self._current_snapshots.get(node_id)
            return snapshot.effect if snapshot else None

    def get_current_snapshot(self, node_id: str) -> Optional[dict]:
        with self._model_lock:
            snapshot = self._current_snapshots.get(node_id)
            return snapshot.to_dict() if snapshot else None

    def get_all_snapshots(self) -> dict:
        with self._model_lock:
            return {
                node_id: snapshot.to_dict()
                for node_id, snapshot in self._current_snapshots.items()
            }

    def explain(self, node_id: str) -> Optional[str]:
        with self._model_lock:
            snapshot = self._current_snapshots.get(node_id)
        if snapshot is None:
            return None
        direction = "increases" if snapshot.effect > 0 else "decreases"
        return (
            f"On {node_id}: each additional concurrent active query "
            f"causally {direction} average query latency by "
            f"{abs(snapshot.effect):.4f}ms. "
            f"Model trained on {snapshot.samples_used} samples. "
            f"Last updated: {snapshot.timestamp}."
        )

    def status(self) -> dict:
        with self._model_lock:
            return {
                "status": self._status,
                "buffer_size": self.buffer.size(),
                "total_samples_received": self.buffer.total_received(),
                "retrain_count": self._retrain_count,
                "last_retrain": self._last_retrain_time,
                "nodes_modeled": list(self._current_snapshots.keys()),
                "min_samples_required": self.buffer.min_samples,
                "is_ready": self._status == "ready",
            }


if __name__ == "__main__":
    updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )

    updater.start()
    logger.info(
        "Streaming updater running. "
        "First retrain in 30s. "
        "Run load generator in another terminal."
    )

    try:
        cycle = 0
        while True:
            time.sleep(15)
            cycle += 1
            status = updater.status()
            logger.info(
                "cycle=%d status=%s buffer=%d retrains=%d nodes=%s",
                cycle,
                status["status"],
                status["buffer_size"],
                status["retrain_count"],
                status["nodes_modeled"],
            )
            if status["is_ready"]:
                for node_id in ["node-1", "node-2", "node-3"]:
                    explanation = updater.explain(node_id)
                    if explanation:
                        logger.info("  %s", explanation)
                    snapshot = updater.get_current_snapshot(node_id)
                    if snapshot:
                        logger.info(
                            "  node=%-8s effect=%.4f samples=%d",
                            node_id,
                            snapshot["effect"],
                            snapshot["samples_used"],
                        )
    except KeyboardInterrupt:
        logger.info("Shutting down streaming updater")
    finally:
        updater.stop()