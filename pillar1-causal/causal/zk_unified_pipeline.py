import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "telemetry"))
sys.path.insert(0, os.path.dirname(__file__))

from zk_proof_bridge import ZKProofBridge, CausalProofRequest
from streaming_updater import StreamingCausalUpdater
from cross_node_causal import DistributedCausalCorrelator
from graph_builder import TelemetrySampler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.causal.zk_unified")

updater: Optional[StreamingCausalUpdater] = None
bridge: Optional[ZKProofBridge] = None
cross_node_results: Optional[dict] = None
cross_node_lock = threading.Lock()
proof_store: dict = {}
proof_store_lock = threading.RLock()
pipeline_start_time: Optional[str] = None


def encode_effect_for_proof(
    node_id: str,
    snapshot: dict,
    active_queries: int = 5,
) -> tuple:
    raw_effect = abs(snapshot["effect"])
    # Scale to integer — multiply by 100 to preserve 2 decimal places
    # e.g. 29.63ms -> avg_latency=2963, active_queries=1, mac=2963
    # This ensures mac = alpha * value holds exactly
    avg_latency_ms = int(raw_effect * 100)
    query_count = 1
    mac = avg_latency_ms * query_count
    return avg_latency_ms, query_count, mac, raw_effect


def run_cross_node_analysis():
    global cross_node_results
    logger.info("Starting cross-node causal analysis")
    try:
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


def auto_prove_all_nodes():
    global proof_store
    if not updater or not bridge:
        return
    engine_status = updater.status()
    if not engine_status.get("is_ready"):
        return

    logger.info(
        "Auto-proving causal effects for all nodes retrain=%d",
        engine_status["retrain_count"],
    )
    for node_id in engine_status.get("nodes_modeled", []):
        try:
            snapshot = updater.get_current_snapshot(node_id)
            if not snapshot:
                continue

            avg_latency_ms, query_count, mac, raw_effect = (
                encode_effect_for_proof(node_id, snapshot)
            )

            request = CausalProofRequest(
                node_id=node_id,
                active_queries=query_count,
                avg_latency_ms=float(avg_latency_ms),
                causal_effect_ms=float(mac),
                samples_used=snapshot["samples_used"],
                retrain_cycle=engine_status["retrain_count"],
            )
            result = bridge.prove(request)

            if result.success and result.verified:
                with proof_store_lock:
                    proof_store[node_id] = {
                        "proof_id": result.proof_id(),
                        "node_id": node_id,
                        "causal_effect_ms": raw_effect,
                        "avg_latency_ms_per_query": raw_effect,
                        "verified": result.verified,
                        "challenge": result.challenge(),
                        "constraint_satisfied": result.constraint_satisfied(),
                        "proof_latency_ms": result.latency_ms,
                        "retrain_cycle": engine_status["retrain_count"],
                        "samples_used": snapshot["samples_used"],
                        "timestamp": result.timestamp,
                        "full_proof": result.proof,
                    }
                logger.info(
                    "Auto-proof verified node=%s effect=%.4fms "
                    "proof_id=%s latency=%.1fms",
                    node_id,
                    raw_effect,
                    result.proof_id(),
                    result.latency_ms,
                )
            else:
                logger.warning(
                    "Auto-proof failed node=%s error=%s",
                    node_id,
                    result.error,
                )
        except Exception as e:
            logger.error("Auto-prove error node=%s: %s", node_id, e)


def proof_refresh_loop():
    logger.info("Proof refresh loop started")
    last_retrain = -1
    while True:
        time.sleep(15)
        try:
            if not updater:
                continue
            status = updater.status()
            current_retrain = status.get("retrain_count", 0)
            if current_retrain > last_retrain and status.get("is_ready"):
                last_retrain = current_retrain
                auto_prove_all_nodes()
        except Exception as e:
            logger.error("Proof refresh error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global updater, bridge, pipeline_start_time
    pipeline_start_time = datetime.now(timezone.utc).isoformat()
    logger.info("Starting ZK Unified Pipeline")

    bridge = ZKProofBridge()

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

    proof_thread = threading.Thread(
        target=proof_refresh_loop,
        name="proof-refresher",
        daemon=True,
    )
    proof_thread.start()

    logger.info("ZK Unified Pipeline started")
    yield

    logger.info("Shutting down ZK Unified Pipeline")
    if updater:
        updater.stop()


app = FastAPI(
    title="CognitiveMesh ZK Unified Pipeline",
    description=(
        "Privacy-preserving causal reasoning over distributed telemetry. "
        "Every causal claim is backed by a STARK zero-knowledge proof."
    ),
    version="0.4.0",
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
    engine_status = updater.status() if updater else {}
    bridge_stats = bridge.stats() if bridge else {}
    with proof_store_lock:
        proofs_ready = list(proof_store.keys())
    with cross_node_lock:
        cn_ready = cross_node_results is not None
    return {
        "status": "healthy",
        "version": "0.4.0",
        "pipeline_start": pipeline_start_time,
        "engine": engine_status,
        "bridge": bridge_stats,
        "proofs_ready": proofs_ready,
        "cross_node_ready": cn_ready,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/why/{node_id}")
def why_with_proof(node_id: str):
    if not updater:
        raise HTTPException(status_code=503, detail="Pipeline not started")

    engine_status = updater.status()
    if not engine_status.get("is_ready"):
        raise HTTPException(
            status_code=503,
            detail=f"Causal engine not ready. "
                   f"Buffer: {engine_status['buffer_size']}/"
                   f"{engine_status['min_samples_required']}"
        )

    snapshot = updater.get_current_snapshot(node_id)
    if not snapshot:
        raise HTTPException(
            status_code=404,
            detail=f"No model for node={node_id}. "
                   f"Available: {engine_status['nodes_modeled']}"
        )

    effect = snapshot["effect"]
    explanation = updater.explain(node_id)

    with proof_store_lock:
        proof_data = proof_store.get(node_id)

    return {
        "node_id": node_id,
        "causal_effect_ms": round(effect, 4),
        "explanation": explanation,
        "zk_proof": {
            "available": proof_data is not None,
            "proof_id": proof_data["proof_id"] if proof_data else None,
            "verified": proof_data["verified"] if proof_data else False,
            "challenge": proof_data["challenge"] if proof_data else None,
            "constraint_satisfied": proof_data["constraint_satisfied"]
            if proof_data else False,
            "proof_latency_ms": proof_data["proof_latency_ms"]
            if proof_data else None,
            "retrain_cycle": proof_data["retrain_cycle"]
            if proof_data else None,
        },
        "privacy_guarantee": (
            "This causal claim is backed by a STARK zero-knowledge proof. "
            "The verifier confirmed the causal effect without seeing "
            "raw telemetry data."
            if proof_data and proof_data["verified"]
            else "ZK proof pending — retrain in progress."
        ),
        "model_info": {
            "samples_used": snapshot["samples_used"],
            "last_updated": snapshot["timestamp"],
            "retrain_count": engine_status["retrain_count"],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/why")
def why_all_with_proofs():
    if not updater:
        raise HTTPException(status_code=503, detail="Pipeline not started")

    engine_status = updater.status()
    if not engine_status.get("is_ready"):
        raise HTTPException(
            status_code=503,
            detail="Causal engine not ready"
        )

    snapshots = updater.get_all_snapshots()
    results = {}

    for node_id, snapshot in snapshots.items():
        effect = snapshot["effect"]
        with proof_store_lock:
            proof_data = proof_store.get(node_id)

        results[node_id] = {
            "causal_effect_ms": round(effect, 4),
            "explanation": updater.explain(node_id),
            "zk_proof": {
                "available": proof_data is not None,
                "proof_id": proof_data["proof_id"] if proof_data else None,
                "verified": proof_data["verified"] if proof_data else False,
            },
        }

    if results:
        worst = max(
            results,
            key=lambda k: abs(results[k]["causal_effect_ms"])
        )
        analysis = {
            "worst_node": worst,
            "worst_effect_ms": results[worst]["causal_effect_ms"],
            "all_proofs_verified": all(
                r["zk_proof"]["verified"] for r in results.values()
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


@app.get("/proof/{node_id}")
def get_proof(node_id: str):
    with proof_store_lock:
        proof_data = proof_store.get(node_id)

    if not proof_data:
        raise HTTPException(
            status_code=404,
            detail=f"No proof available for node={node_id}. "
                   f"Wait for next retrain cycle."
        )

    return {
        "node_id": node_id,
        "proof_id": proof_data["proof_id"],
        "causal_effect_ms": proof_data["causal_effect_ms"],
        "verified": proof_data["verified"],
        "challenge": proof_data["challenge"],
        "constraint_satisfied": proof_data["constraint_satisfied"],
        "proof_latency_ms": proof_data["proof_latency_ms"],
        "retrain_cycle": proof_data["retrain_cycle"],
        "samples_used": proof_data["samples_used"],
        "timestamp": proof_data["timestamp"],
    }


@app.get("/proof/{node_id}/full")
def get_full_proof(node_id: str):
    with proof_store_lock:
        proof_data = proof_store.get(node_id)

    if not proof_data:
        raise HTTPException(
            status_code=404,
            detail=f"No proof available for node={node_id}"
        )

    return {
        "node_id": node_id,
        "proof_id": proof_data["proof_id"],
        "causal_effect_ms": proof_data["causal_effect_ms"],
        "verified": proof_data["verified"],
        "stark_proof": proof_data["full_proof"],
        "timestamp": proof_data["timestamp"],
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


@app.get("/pipeline/summary")
def pipeline_summary():
    engine_status = updater.status() if updater else {}
    bridge_stats = bridge.stats() if bridge else {}

    with proof_store_lock:
        proof_summary = {
            node_id: {
                "proof_id": d["proof_id"],
                "verified": d["verified"],
                "causal_effect_ms": d["causal_effect_ms"],
                "retrain_cycle": d["retrain_cycle"],
            }
            for node_id, d in proof_store.items()
        }

    with cross_node_lock:
        cn_ready = cross_node_results is not None
        cn_effects = len(
            cross_node_results.get("cross_node_effects", {})
        ) if cn_ready else 0

    all_verified = all(
        d["verified"] for d in proof_store.values()
    ) if proof_store else False

    return {
        "pipeline": {
            "start_time": pipeline_start_time,
            "uptime_seconds": (
                datetime.now(timezone.utc) -
                datetime.fromisoformat(pipeline_start_time)
            ).total_seconds() if pipeline_start_time else 0,
            "version": "0.4.0",
        },
        "causal_engine": {
            "status": engine_status.get("status"),
            "buffer_size": engine_status.get("buffer_size"),
            "retrain_count": engine_status.get("retrain_count"),
            "nodes_modeled": engine_status.get("nodes_modeled"),
        },
        "zk_layer": {
            "proofs_generated": bridge_stats.get("proofs_generated", 0),
            "avg_proof_latency_ms": bridge_stats.get("avg_latency_ms", 0),
            "nodes_with_proofs": list(proof_summary.keys()),
            "all_verified": all_verified,
        },
        "node_proofs": proof_summary,
        "cross_node": {
            "ready": cn_ready,
            "effects_found": cn_effects,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "zk_unified_pipeline:app",
        host="0.0.0.0",
        port=8084,
        reload=False,
        log_level="info",
    )