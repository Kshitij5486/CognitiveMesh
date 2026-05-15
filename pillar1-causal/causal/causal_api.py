import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from graph_builder import DistributedCausalEngine, NodeCausalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.causal.api")

engine: Optional[DistributedCausalEngine] = None
engine_lock = threading.Lock()
engine_status = {
    "state": "uninitialized",
    "last_trained": None,
    "samples_collected": 0,
    "nodes_modeled": [],
    "error": None,
}


def train_engine(n_samples: int = 60, interval: float = 2.0):
    global engine, engine_status
    with engine_lock:
        engine_status["state"] = "collecting"
        engine_status["error"] = None

    try:
        new_engine = DistributedCausalEngine()
        logger.info(
            "Starting causal engine training n_samples=%d interval=%.1fs",
            n_samples,
            interval,
        )
        new_engine.collect_data(
            n_samples=n_samples,
            interval_seconds=interval,
        )

        with engine_lock:
            engine_status["state"] = "building"
            engine_status["samples_collected"] = n_samples

        results = new_engine.build_all_models()

        with engine_lock:
            engine = new_engine
            engine_status["state"] = "ready"
            engine_status["last_trained"] = datetime.now(timezone.utc).isoformat()
            engine_status["nodes_modeled"] = list(results.keys())
            engine_status["samples_collected"] = n_samples

        logger.info(
            "Causal engine training complete nodes=%s",
            list(results.keys()),
        )

    except Exception as e:
        with engine_lock:
            engine_status["state"] = "error"
            engine_status["error"] = str(e)
        logger.error("Causal engine training failed error=%s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting causal API — launching background training")
    thread = threading.Thread(
        target=train_engine,
        kwargs={"n_samples": 60, "interval": 2.0},
        daemon=True,
        name="causal-trainer",
    )
    thread.start()
    yield
    logger.info("Causal API shutting down")
    if engine:
        engine.close()


app = FastAPI(
    title="CognitiveMesh Causal Query API",
    description="Real-time causal reasoning over distributed PostgreSQL telemetry",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    with engine_lock:
        status = dict(engine_status)
    return {
        "status": "healthy",
        "engine": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/status")
def status():
    with engine_lock:
        status = dict(engine_status)
    return status


@app.post("/retrain")
def retrain(
    background_tasks: BackgroundTasks,
    n_samples: int = 60,
    interval: float = 2.0,
):
    with engine_lock:
        current_state = engine_status["state"]

    if current_state in ("collecting", "building"):
        raise HTTPException(
            status_code=409,
            detail=f"Engine is already training state={current_state}"
        )

    background_tasks.add_task(train_engine, n_samples, interval)
    return {
        "message": "Retraining started",
        "n_samples": n_samples,
        "estimated_duration_seconds": n_samples * interval,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/why/{node_id}")
def why_node_slow(node_id: str):
    with engine_lock:
        state = engine_status["state"]
        current_engine = engine

    if state != "ready" or current_engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Causal engine not ready state={state}. "
                   f"Check /status for progress."
        )

    if node_id not in current_engine.node_models:
        available = list(current_engine.node_models.keys())
        raise HTTPException(
            status_code=404,
            detail=f"No causal model for node={node_id}. "
                   f"Available nodes={available}"
        )

    model: NodeCausalModel = current_engine.node_models[node_id]
    effect = float(model.estimate.value) if model.estimate else None
    explanation = current_engine.explain_latency(node_id)

    causal_chain = []
    if effect and abs(effect) > 0.001:
        causal_chain = [
            {
                "cause": f"{model.node_safe}_active_queries",
                "effect": f"{model.node_safe}_avg_query_duration_ms",
                "magnitude_ms": round(abs(effect), 4),
                "direction": "increases" if effect > 0 else "decreases",
                "confidence": "high" if abs(effect) > 1.0 else "low",
            }
        ]

    return {
        "node_id": node_id,
        "causal_effect": effect,
        "explanation": explanation,
        "causal_chain": causal_chain,
        "treatment": f"{model.node_safe}_active_queries",
        "outcome": f"{model.node_safe}_avg_query_duration_ms",
        "model_timestamp": engine_status.get("last_trained"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/why/{node_id}/full")
def why_node_full(node_id: str):
    with engine_lock:
        state = engine_status["state"]
        current_engine = engine

    if state != "ready" or current_engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Causal engine not ready state={state}"
        )

    if node_id not in current_engine.node_models:
        raise HTTPException(
            status_code=404,
            detail=f"No causal model for node={node_id}"
        )

    model: NodeCausalModel = current_engine.node_models[node_id]

    dataset_stats = {}
    if current_engine.dataset is not None:
        node_cols = [
            c for c in current_engine.dataset.columns
            if c.startswith(model.node_safe)
        ]
        for col in node_cols:
            series = current_engine.dataset[col]
            dataset_stats[col] = {
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
            }

    return {
        "node_id": node_id,
        "summary": model.summary(),
        "dataset_stats": dataset_stats,
        "samples_used": len(current_engine.dataset)
        if current_engine.dataset is not None else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/compare")
def compare_all_nodes():
    with engine_lock:
        state = engine_status["state"]
        current_engine = engine

    if state != "ready" or current_engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Causal engine not ready state={state}"
        )

    comparison = {}
    for node_id, model in current_engine.node_models.items():
        effect = float(model.estimate.value) if model.estimate else None
        comparison[node_id] = {
            "causal_effect_ms": round(effect, 4) if effect else None,
            "treatment": f"{model.node_safe}_active_queries",
            "outcome": f"{model.node_safe}_avg_query_duration_ms",
            "explanation": current_engine.explain_latency(node_id),
        }

    if comparison:
        effects = {
            k: v["causal_effect_ms"]
            for k, v in comparison.items()
            if v["causal_effect_ms"] is not None
        }
        if effects:
            worst_node = max(effects, key=lambda k: abs(effects[k]))
            comparison["_analysis"] = {
                "worst_node": worst_node,
                "worst_effect_ms": effects[worst_node],
                "insight": f"{worst_node} has the highest causal sensitivity "
                           f"— each additional active query adds "
                           f"{abs(effects[worst_node]):.2f}ms of latency",
            }

    return {
        "nodes": comparison,
        "model_timestamp": engine_status.get("last_trained"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "causal_api:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
        log_level="info",
    )