import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "telemetry"))
sys.path.insert(0, os.path.dirname(__file__))

from streaming_updater import StreamingCausalUpdater
from cross_node_causal import CrossNodeCausalGraph, DistributedCausalCorrelator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.causal.unified_pipeline")

updater: Optional[StreamingCausalUpdater] = None
cross_node_results: Optional[dict] = None
cross_node_lock = threading.Lock()
pipeline_start_time: Optional[str] = None


def run_cross_node_analysis():
    global cross_node_results
    logger.info("Starting cross-node causal analysis")
    try:
        from graph_builder import TelemetrySampler
        sampler = TelemetrySampler()
        dataset = sampler.collect_dataset(n_samples=40, interval_seconds=2.0)
        sampler.close()
        correlator = DistributedCausalCorrelator(dataset=dataset)
        results = correlator.run()
        with cross_node_lock:
            cross_node_results = results
        logger.info(
            "Cross-node analysis complete effects=%d",
            len(results.get("cross_node_effects", {}))
        )
    except Exception as e:
        logger.error("Cross-node analysis failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global updater, pipeline_start_time
    pipeline_start_time = datetime.now(timezone.utc).isoformat()
    logger.info("Starting CognitiveMesh unified pipeline")

    updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )
    updater.start()

    cross_node_thread = threading.Thread(
        target=run_cross_node_analysis,
        name="cross-node-analyzer",
        daemon=True,
    )
    cross_node_thread.start()

    logger.info("Unified pipeline started — causal engine training in background")
    yield

    logger.info("Shutting down unified pipeline")
    if updater:
        updater.stop()


app = FastAPI(
    title="CognitiveMesh Unified Pipeline",
    description=(
        "Real-time causal reasoning over distributed PostgreSQL telemetry. "
        "Combines streaming telemetry collection, live causal model updates, "
        "and cross-node causal analysis in a single service."
    ),
    version="0.3.0",
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
    status = updater.status() if updater else {}
    return {
        "status": "healthy",
        "pipeline_start": pipeline_start_time,
        "engine": status,
        "cross_node_ready": cross_node_results is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/status")
def status():
    if not updater:
        return {"status": "not_started"}
    return updater.status()


@app.get("/why/{node_id}")
def why_node_slow(node_id: str):
    if not updater:
        raise HTTPException(status_code=503, detail="Pipeline not started")

    engine_status = updater.status()
    if not engine_status["is_ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"Causal engine not ready yet. "
                   f"Buffer: {engine_status['buffer_size']}/"
                   f"{engine_status['min_samples_required']} samples. "
                   f"Retrains: {engine_status['retrain_count']}."
        )

    snapshot = updater.get_current_snapshot(node_id)
    if not snapshot:
        available = engine_status["nodes_modeled"]
        raise HTTPException(
            status_code=404,
            detail=f"No model for node={node_id}. Available: {available}"
        )

    explanation = updater.explain(node_id)
    effect = snapshot["effect"]

    return {
        "node_id": node_id,
        "causal_effect_ms": round(effect, 4),
        "explanation": explanation,
        "causal_chain": {
            "treatment": f"{node_id.replace('-', '_')}_active_queries",
            "outcome": f"{node_id.replace('-', '_')}_avg_query_duration_ms",
            "direction": "increases" if effect > 0 else "decreases",
            "magnitude_ms": round(abs(effect), 4),
            "confidence": "high" if abs(effect) > 5.0 else "low",
        },
        "model_info": {
            "samples_used": snapshot["samples_used"],
            "last_updated": snapshot["timestamp"],
            "retrain_count": updater.status()["retrain_count"],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/why")
def why_all_nodes():
    if not updater:
        raise HTTPException(status_code=503, detail="Pipeline not started")

    engine_status = updater.status()
    if not engine_status["is_ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"Causal engine not ready. "
                   f"Buffer: {engine_status['buffer_size']}/"
                   f"{engine_status['min_samples_required']}"
        )

    snapshots = updater.get_all_snapshots()
    results = {}
    for node_id, snapshot in snapshots.items():
        effect = snapshot["effect"]
        results[node_id] = {
            "causal_effect_ms": round(effect, 4),
            "explanation": updater.explain(node_id),
            "direction": "increases" if effect > 0 else "decreases",
            "samples_used": snapshot["samples_used"],
            "last_updated": snapshot["timestamp"],
        }

    if results:
        worst = max(results, key=lambda k: abs(results[k]["causal_effect_ms"]))
        analysis = {
            "worst_node": worst,
            "worst_effect_ms": results[worst]["causal_effect_ms"],
            "insight": (
                f"{worst} is most sensitive to load — "
                f"each additional active query adds "
                f"{abs(results[worst]['causal_effect_ms']):.2f}ms latency"
            ),
        }
    else:
        analysis = {}

    return {
        "nodes": results,
        "analysis": analysis,
        "retrain_count": engine_status["retrain_count"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/cross-node")
def cross_node():
    with cross_node_lock:
        results = cross_node_results

    if results is None:
        raise HTTPException(
            status_code=503,
            detail="Cross-node analysis still running. Retry in 90 seconds."
        )

    significant = {
        k: v for k, v in results.get("cross_node_effects", {}).items()
        if v.get("significant")
    }

    return {
        "graph_stats": results.get("graph_stats", {}),
        "significant_effects": significant,
        "explanations": results.get("explanations", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    if not updater:
        raise HTTPException(status_code=503, detail="Pipeline not started")

    engine_status = updater.status()
    if engine_status["status"] == "retraining":
        raise HTTPException(
            status_code=409,
            detail="Retrain already in progress"
        )

    background_tasks.add_task(run_cross_node_analysis)
    return {
        "message": "Cross-node analysis triggered",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/pipeline/summary")
def pipeline_summary():
    if not updater:
        raise HTTPException(status_code=503, detail="Pipeline not started")

    engine_status = updater.status()
    snapshots = updater.get_all_snapshots()

    with cross_node_lock:
        cn_ready = cross_node_results is not None
        cn_effects = len(
            cross_node_results.get("cross_node_effects", {})
        ) if cn_ready else 0

    return {
        "pipeline": {
            "start_time": pipeline_start_time,
            "uptime_seconds": (
                datetime.now(timezone.utc) -
                datetime.fromisoformat(pipeline_start_time)
            ).total_seconds() if pipeline_start_time else 0,
            "version": "0.3.0",
        },
        "causal_engine": {
            "status": engine_status["status"],
            "buffer_size": engine_status["buffer_size"],
            "total_samples": engine_status["total_samples_received"],
            "retrain_count": engine_status["retrain_count"],
            "last_retrain": engine_status["last_retrain"],
            "nodes_modeled": engine_status["nodes_modeled"],
        },
        "current_effects": {
            node_id: round(snapshot["effect"], 4)
            for node_id, snapshot in snapshots.items()
        },
        "cross_node": {
            "ready": cn_ready,
            "effects_found": cn_effects,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "unified_pipeline:app",
        host="0.0.0.0",
        port=8082,
        reload=False,
        log_level="info",
    )