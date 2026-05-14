import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from consumer import EventStore, TelemetryConsumer, EventQueryAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.telemetry.api")

event_store = EventStore(max_events_per_node=10000)
consumer = TelemetryConsumer(event_store=event_store)
query_api = EventQueryAPI(event_store=event_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting telemetry consumer")
    consumer.start()
    yield
    logger.info("Stopping telemetry consumer")
    consumer.stop()


app = FastAPI(
    title="CognitiveMesh Telemetry API",
    description="Real-time telemetry data from the 3-node PostgreSQL cluster",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_consumed": consumer.get_total_consumed(),
    }


@app.get("/store/summary")
def store_summary():
    return query_api.get_store_summary()


@app.get("/events/{node_id}")
def get_node_events(
    node_id: str,
    event_type: str = Query(
        default="io",
        description="Event type: query, lock, io, heartbeat"
    ),
    last_n: int = Query(
        default=10,
        ge=1,
        le=1000,
        description="Number of most recent events to return"
    ),
):
    valid_types = {"query", "lock", "io", "heartbeat"}
    if event_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type. Must be one of: {valid_types}"
        )
    return query_api.get_node_events(
        node_id=node_id,
        event_type=event_type,
        last_n=last_n
    )


@app.get("/events/{node_id}/latest")
def get_node_latest(
    node_id: str,
    event_type: str = Query(default="io"),
):
    result = query_api.get_node_events(
        node_id=node_id,
        event_type=event_type,
        last_n=1
    )
    if result["count"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No events found for node={node_id} event_type={event_type}"
        )
    return {
        "node_id": node_id,
        "event_type": event_type,
        "event": result["events"][0],
    }


@app.get("/nodes")
def get_all_nodes(
    event_type: str = Query(default="io"),
):
    return {
        "event_type": event_type,
        "nodes": event_store.get_all_nodes(event_type),
        "latest": query_api.get_all_nodes_latest(event_type),
    }


@app.get("/nodes/compare")
def compare_nodes(
    event_type: str = Query(default="io"),
):
    nodes = event_store.get_all_nodes(event_type)
    comparison = {}
    for node_id in nodes:
        events = event_store.get_events(event_type, node_id, last_n=10)
        if events:
            comparison[node_id] = {
                "event_count": len(events),
                "latest_timestamp": events[-1].get("timestamp"),
                "latest_event": events[-1],
            }
    return {
        "event_type": event_type,
        "node_count": len(nodes),
        "comparison": comparison,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )