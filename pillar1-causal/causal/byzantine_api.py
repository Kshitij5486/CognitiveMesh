import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "telemetry"))
sys.path.insert(0, os.path.dirname(__file__))

from zk_proof_bridge import ZKProofBridge
from streaming_updater import StreamingCausalUpdater
from byzantine_detector import ByzantineDetector
from reputation_scorer import ReputationScorer
from consensus_engine import ConsensusEngine
from isolation_mechanism import IsolationMechanism

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.byzantine.api")

updater: Optional[StreamingCausalUpdater] = None
bridge: Optional[ZKProofBridge] = None
detector: Optional[ByzantineDetector] = None
scorer: Optional[ReputationScorer] = None
engine: Optional[ConsensusEngine] = None
isolation: Optional[IsolationMechanism] = None
api_start_time: Optional[str] = None


class ManualIsolationRequest(BaseModel):
    node_id: str
    reason: Optional[str] = "Manual operator action"


class ManualReleaseRequest(BaseModel):
    node_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global updater, bridge, detector, scorer, engine, isolation
    global api_start_time
    api_start_time = datetime.now(timezone.utc).isoformat()
    logger.info("Starting Byzantine Consensus API")

    updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )
    updater.start()

    bridge = ZKProofBridge()

    detector = ByzantineDetector(
        updater=updater,
        bridge=bridge,
        check_interval=15.0,
    )
    detector.start()

    scorer = ReputationScorer(
        detector=detector,
        update_interval_seconds=15.0,
        decay_interval_seconds=60.0,
    )
    scorer.start()

    engine = ConsensusEngine(
        scorer=scorer,
        detector=detector,
        updater=updater,
    )

    isolation = IsolationMechanism(
        scorer=scorer,
        detector=detector,
        engine=engine,
        check_interval=15.0,
    )
    isolation.start()

    logger.info("Byzantine Consensus API fully started — all 5 components running")
    yield

    logger.info("Shutting down Byzantine Consensus API")
    if isolation:
        isolation.stop()
    if scorer:
        scorer.stop()
    if detector:
        detector.stop()
    if updater:
        updater.stop()


app = FastAPI(
    title="CognitiveMesh Byzantine Consensus API",
    description=(
        "Byzantine fault detection, reputation scoring, "
        "reputation-weighted consensus, and node isolation "
        "for a self-aware distributed computing fabric."
    ),
    version="0.5.0",
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
    isolation_status = isolation.status() if isolation else {}
    return {
        "status": "healthy",
        "version": "0.5.0",
        "api_start": api_start_time,
        "causal_engine": {
            "ready": engine_status.get("is_ready", False),
            "buffer_size": engine_status.get("buffer_size", 0),
            "retrain_count": engine_status.get("retrain_count", 0),
        },
        "byzantine_stack": {
            "detector_checks": detector.status()["checks_run"]
            if detector else 0,
            "active_nodes": isolation_status.get("active_nodes", []),
            "isolated_nodes": isolation_status.get("isolated_nodes", []),
            "cluster_operational": isolation_status.get(
                "cluster_operational", False
            ),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/cluster")
def cluster_overview():
    if not all([detector, scorer, engine, isolation]):
        raise HTTPException(status_code=503, detail="Stack not ready")

    isolation_status = isolation.cluster_status()
    scorer_summary = scorer.cluster_summary()
    engine_status_data = engine.status()
    detector_status = detector.get_cluster_health()

    return {
        "cluster_operational": isolation_status["cluster_operational"],
        "active_nodes": isolation_status["active_nodes"],
        "isolated_nodes": isolation_status["isolated_nodes"],
        "reputation": scorer_summary,
        "consensus": {
            "can_reach": engine_status_data["can_reach_consensus"],
            "participating": engine_status_data["participating_nodes"],
            "excluded": engine_status_data["excluded_nodes"],
            "total_decisions": engine_status_data["total_decisions"],
        },
        "detection": {
            "checks_run": detector_status["checks_run"],
            "byzantine_detections": detector_status[
                "byzantine_detections"
            ],
            "trusted_nodes": detector_status["trusted_nodes"],
            "suspicious_nodes": detector_status["suspicious_nodes"],
            "byzantine_nodes": detector_status["byzantine_nodes"],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/nodes")
def list_nodes():
    if not all([scorer, detector, isolation]):
        raise HTTPException(status_code=503, detail="Stack not ready")

    nodes = {}
    for node_id in ["node-1", "node-2", "node-3"]:
        rep = scorer.reputations.get(node_id)
        profile = detector.profiles.get(node_id)
        is_active = isolation.is_node_active(node_id)

        nodes[node_id] = {
            "active": is_active,
            "isolated": not is_active,
            "reputation_score": scorer.get_score(node_id),
            "reputation_status": scorer.get_status(node_id).value,
            "reputation_trend": rep.get_trend() if rep else "unknown",
            "byzantine_status": profile.status.value if profile else "unknown",
            "trust_score": profile.trust_score() if profile else 0.0,
            "proof_successes": profile.proof_successes if profile else 0,
            "proof_failures": profile.proof_failures if profile else 0,
            "causal_violations": profile.causal_violations
            if profile else 0,
        }

    return {
        "nodes": nodes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/nodes/{node_id}")
def get_node(node_id: str):
    if node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {node_id}"
        )
    if not all([scorer, detector, isolation, updater]):
        raise HTTPException(status_code=503, detail="Stack not ready")

    rep = scorer.reputations.get(node_id)
    profile = detector.profiles.get(node_id)
    is_active = isolation.is_node_active(node_id)

    snapshot = None
    engine_status = updater.status()
    if engine_status.get("is_ready"):
        snapshot = updater.get_current_snapshot(node_id)

    return {
        "node_id": node_id,
        "active": is_active,
        "isolated": not is_active,
        "reputation": rep.to_dict() if rep else None,
        "byzantine_profile": profile.to_dict() if profile else None,
        "causal_snapshot": snapshot,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/consensus/causal")
def consensus_causal():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")

    engine_status = updater.status() if updater else {}
    if not engine_status.get("is_ready"):
        raise HTTPException(
            status_code=503,
            detail=f"Causal engine not ready. "
                   f"Buffer: {engine_status.get('buffer_size', 0)}/30"
        )

    result = engine.decide_cluster_causal_effect()
    if not result:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach consensus — insufficient participating nodes"
        )

    return result.to_dict()


@app.post("/consensus/threshold")
def consensus_threshold(metric: str = "avg_latency_ms"):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")

    engine_status = updater.status() if updater else {}
    if not engine_status.get("is_ready"):
        raise HTTPException(
            status_code=503,
            detail="Causal engine not ready"
        )

    result = engine.decide_cluster_threshold(metric=metric)
    if not result:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach threshold consensus"
        )

    return result.to_dict()


@app.get("/consensus/history")
def consensus_history(n: int = 5):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")

    decisions = engine.get_recent_decisions(n=n)
    return {
        "decisions": decisions,
        "total": len(decisions),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/isolation/isolate")
def isolate_node(request: ManualIsolationRequest):
    if not isolation:
        raise HTTPException(
            status_code=503, detail="Isolation not ready"
        )

    success = isolation.manually_isolate(request.node_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot isolate {request.node_id} — "
                f"already isolated or insufficient healthy nodes"
            )
        )

    return {
        "isolated": True,
        "node_id": request.node_id,
        "reason": request.reason,
        "active_nodes": isolation.get_active_nodes(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/isolation/release")
def release_node(request: ManualReleaseRequest):
    if not isolation:
        raise HTTPException(
            status_code=503, detail="Isolation not ready"
        )

    success = isolation.manually_release(request.node_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot release {request.node_id} — not isolated"
        )

    return {
        "released": True,
        "node_id": request.node_id,
        "active_nodes": isolation.get_active_nodes(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/isolation/status")
def isolation_status():
    if not isolation:
        raise HTTPException(
            status_code=503, detail="Isolation not ready"
        )
    return isolation.cluster_status()


@app.get("/reputation")
def reputation_overview():
    if not scorer:
        raise HTTPException(status_code=503, detail="Scorer not ready")
    return scorer.cluster_summary()


@app.get("/reputation/{node_id}")
def node_reputation(node_id: str):
    if node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {node_id}"
        )
    if not scorer:
        raise HTTPException(status_code=503, detail="Scorer not ready")

    rep = scorer.reputations.get(node_id)
    if not rep:
        raise HTTPException(
            status_code=404,
            detail=f"No reputation data for {node_id}"
        )
    return rep.to_dict()


@app.get("/detection")
def detection_overview():
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not ready")
    return detector.get_cluster_health()


@app.get("/detection/{node_id}")
def node_detection(node_id: str):
    if node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {node_id}"
        )
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not ready")

    profile = detector.profiles.get(node_id)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"No detection profile for {node_id}"
        )
    return profile.to_dict()


@app.get("/pipeline/summary")
def pipeline_summary():
    engine_status = updater.status() if updater else {}
    isolation_status_data = isolation.status() if isolation else {}
    scorer_summary = scorer.cluster_summary() if scorer else {}
    engine_decisions = engine.status() if engine else {}

    uptime = 0.0
    if api_start_time:
        uptime = (
            datetime.now(timezone.utc) -
            datetime.fromisoformat(api_start_time)
        ).total_seconds()

    return {
        "pipeline": {
            "version": "0.5.0",
            "start_time": api_start_time,
            "uptime_seconds": round(uptime, 1),
        },
        "causal_engine": {
            "ready": engine_status.get("is_ready", False),
            "buffer_size": engine_status.get("buffer_size", 0),
            "retrain_count": engine_status.get("retrain_count", 0),
            "nodes_modeled": engine_status.get("nodes_modeled", []),
        },
        "byzantine_detection": {
            "checks_run": detector.status()["checks_run"]
            if detector else 0,
            "byzantine_detections": detector.get_cluster_health()[
                "byzantine_detections"
            ] if detector else 0,
        },
        "reputation": {
            "trusted": scorer_summary.get("trusted_nodes", 0),
            "suspicious": scorer_summary.get("suspicious_nodes", 0),
            "byzantine": scorer_summary.get("byzantine_nodes", 0),
            "avg_score": scorer_summary.get("average_score", 0.0),
        },
        "consensus": {
            "total_decisions": engine_decisions.get(
                "total_decisions", 0
            ),
            "byzantine_exclusions": engine_decisions.get(
                "byzantine_exclusions", 0
            ),
            "can_reach": engine_decisions.get(
                "can_reach_consensus", False
            ),
        },
        "isolation": {
            "active_nodes": isolation_status_data.get(
                "active_nodes", []
            ),
            "isolated_nodes": isolation_status_data.get(
                "isolated_nodes", []
            ),
            "total_isolations": isolation_status_data.get(
                "total_isolations", 0
            ),
            "cluster_operational": isolation_status_data.get(
                "cluster_operational", False
            ),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "byzantine_api:app",
        host="0.0.0.0",
        port=8085,
        reload=False,
        log_level="info",
    )