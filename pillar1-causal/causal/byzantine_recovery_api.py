import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "telemetry"))
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.byzantine.api")


# ── Request/Response models ────────────────────────────────

class ManualRecoveryRequest(BaseModel):
    node_ids: list
    reason: str = "manual_api"


class NodeOfflineRequest(BaseModel):
    node_id: str
    reason: str = "api_request"


class ByzantineScoreRequest(BaseModel):
    node_id: str
    score: float


# ── Global component references ────────────────────────────

_updater = None
_quorum = None
_coordinator = None
_orchestrator = None
_quorum_router = None
_start_time: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _updater, _quorum, _coordinator
    global _orchestrator, _quorum_router, _start_time

    _start_time = datetime.now(timezone.utc).isoformat()
    logger.info(
        "Starting ByzantineRecoveryAPI v0.9.0 port 8089"
    )

    from streaming_updater import StreamingCausalUpdater
    from query_router import QueryRouter, RoutingStrategy
    from auto_retrainer import AutoRetrainer
    from quorum_manager import QuorumManager
    from byzantine_recovery_coordinator import (
        ByzantineRecoveryCoordinator,
    )
    from multi_node_recovery_orchestrator import (
        MultiNodeRecoveryOrchestrator,
    )
    from quorum_aware_router import QuorumAwareRouter

    _updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )
    _updater.start()

    _quorum = QuorumManager(
        updater=_updater,
        check_interval=10.0,
    )
    _quorum.start()

    _base_router = QueryRouter(
        updater=_updater,
        alerter=None,
        strategy=RoutingStrategy.CAUSAL_WEIGHTED,
        check_interval=15.0,
    )
    _base_router.start()

    _base_retrainer = AutoRetrainer(
        updater=_updater,
        alerter=None,
        check_interval=30.0,
        drift_threshold_ms=3.0,
    )
    _base_retrainer.start()

    _coordinator = ByzantineRecoveryCoordinator(
        updater=_updater,
        quorum_manager=_quorum,
        router=_base_router,
        retrainer=_base_retrainer,
        check_interval=15.0,
    )
    _coordinator.start()

    _orchestrator = MultiNodeRecoveryOrchestrator(
        updater=_updater,
        quorum_manager=_quorum,
        coordinator=_coordinator,
        check_interval=15.0,
    )
    _orchestrator.start()

    _quorum_router = QuorumAwareRouter(
        updater=_updater,
        quorum_manager=_quorum,
        check_interval=10.0,
    )
    _quorum_router.start()

    logger.info(
        "ByzantineRecoveryAPI fully started on port 8089"
    )
    yield

    logger.info("Shutting down ByzantineRecoveryAPI")
    for component in [
        _quorum_router, _orchestrator, _coordinator,
        _base_retrainer, _base_router, _quorum, _updater,
    ]:
        if component:
            component.stop()


app = FastAPI(
    title="CognitiveMesh Byzantine Recovery API",
    description=(
        "Multi-node Byzantine failure detection and "
        "recovery for the CognitiveMesh distributed "
        "computing fabric. Sprint 9 — v0.9.0"
    ),
    version="0.9.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────

@app.get("/health")
def health():
    uptime = 0.0
    if _start_time:
        uptime = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(_start_time)
        ).total_seconds()

    engine_ready = False
    if _updater:
        s = _updater.status()
        engine_ready = s.get("is_ready", False)

    return {
        "status": "healthy",
        "version": "0.9.0",
        "port": 8089,
        "sprint": 9,
        "uptime_seconds": round(uptime, 1),
        "engine_ready": engine_ready,
        "components": {
            "quorum_manager": _quorum is not None,
            "coordinator": _coordinator is not None,
            "orchestrator": _orchestrator is not None,
            "quorum_router": _quorum_router is not None,
        },
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ── Quorum endpoints ───────────────────────────────────────

@app.get("/quorum")
def get_quorum():
    if not _quorum:
        raise HTTPException(503, "Quorum manager not ready")
    return _quorum.status()


@app.get("/quorum/nodes")
def get_quorum_nodes():
    if not _quorum:
        raise HTTPException(503, "Quorum manager not ready")
    return {
        "nodes": _quorum.get_all_node_statuses(),
        "quorum_state": _quorum.get_quorum_state().value,
        "safe_to_offline": _quorum.get_safe_to_offline(),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


@app.get("/quorum/nodes/{node_id}")
def get_quorum_node(node_id: str):
    if not _quorum:
        raise HTTPException(503, "Quorum manager not ready")
    valid = ["node-1", "node-2", "node-3"]
    if node_id not in valid:
        raise HTTPException(
            404, f"Unknown node: {node_id}"
        )
    return _quorum.get_node_status(node_id)


@app.get("/quorum/decisions")
def get_quorum_decisions(n: int = 10):
    if not _quorum:
        raise HTTPException(503, "Quorum manager not ready")
    return {
        "decisions": _quorum.get_decision_history(n=n),
        "total": _quorum.status()["decisions_made"],
    }


@app.post("/quorum/nodes/{node_id}/offline")
def request_node_offline(
    node_id: str, req: NodeOfflineRequest
):
    if not _quorum:
        raise HTTPException(503, "Quorum manager not ready")
    valid = ["node-1", "node-2", "node-3"]
    if node_id not in valid:
        raise HTTPException(
            404, f"Unknown node: {node_id}"
        )
    from quorum_manager import QuorumDecision
    decision = _quorum.request_node_offline(
        node_id=node_id,
        reason=req.reason,
    )
    return {
        "node_id": node_id,
        "decision": decision.value,
        "allowed": decision in (
            QuorumDecision.ALLOW,
            QuorumDecision.ALLOW_EMERGENCY,
        ),
        "quorum_state": _quorum.get_quorum_state().value,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


@app.post("/quorum/nodes/{node_id}/restore")
def restore_node(node_id: str):
    if not _quorum:
        raise HTTPException(503, "Quorum manager not ready")
    valid = ["node-1", "node-2", "node-3"]
    if node_id not in valid:
        raise HTTPException(
            404, f"Unknown node: {node_id}"
        )
    _quorum.mark_node_full(node_id)
    return {
        "node_id": node_id,
        "action": "restored_to_full",
        "quorum_state": _quorum.get_quorum_state().value,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ── Byzantine detection endpoints ─────────────────────────

@app.get("/byzantine")
def get_byzantine_status():
    if not _coordinator:
        raise HTTPException(
            503, "Coordinator not ready"
        )
    return _coordinator.status()


@app.get("/byzantine/nodes")
def get_byzantine_nodes():
    if not _coordinator:
        raise HTTPException(
            503, "Coordinator not ready"
        )
    return {
        "nodes": _coordinator.get_all_node_states(),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


@app.get("/byzantine/nodes/{node_id}")
def get_byzantine_node(node_id: str):
    if not _coordinator:
        raise HTTPException(
            503, "Coordinator not ready"
        )
    valid = ["node-1", "node-2", "node-3"]
    if node_id not in valid:
        raise HTTPException(
            404, f"Unknown node: {node_id}"
        )
    return _coordinator.get_node_state(node_id)


@app.get("/byzantine/events")
def get_byzantine_events(n: int = 20):
    if not _coordinator:
        raise HTTPException(
            503, "Coordinator not ready"
        )
    return {
        "events": _coordinator.get_events(n=n),
        "total": _coordinator.status()["events_total"],
    }


@app.post("/byzantine/nodes/{node_id}/score")
def set_byzantine_score(
    node_id: str, req: ByzantineScoreRequest
):
    if not _coordinator:
        raise HTTPException(
            503, "Coordinator not ready"
        )
    valid = ["node-1", "node-2", "node-3"]
    if node_id not in valid:
        raise HTTPException(
            404, f"Unknown node: {node_id}"
        )
    if not 0.0 <= req.score <= 1.0:
        raise HTTPException(
            400, "Score must be between 0.0 and 1.0"
        )
    _coordinator.update_byzantine_score(
        node_id, req.score
    )
    return {
        "node_id": node_id,
        "score": req.score,
        "action": "byzantine_score_updated",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ── Recovery endpoints ─────────────────────────────────────

@app.get("/recovery")
def get_recovery_status():
    if not _orchestrator:
        raise HTTPException(
            503, "Orchestrator not ready"
        )
    return _orchestrator.status()


@app.get("/recovery/active")
def get_active_session():
    if not _orchestrator:
        raise HTTPException(
            503, "Orchestrator not ready"
        )
    active = _orchestrator.get_active_session()
    return {
        "active_session": active,
        "has_active": active is not None,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


@app.get("/recovery/history")
def get_recovery_history(n: int = 10):
    if not _orchestrator:
        raise HTTPException(
            503, "Orchestrator not ready"
        )
    return {
        "sessions": _orchestrator.get_session_history(
            n=n
        ),
        "total": _orchestrator.status()[
            "sessions_total"
        ],
    }


@app.get("/recovery/latest")
def get_latest_report():
    if not _orchestrator:
        raise HTTPException(
            503, "Orchestrator not ready"
        )
    report = _orchestrator.get_latest_report()
    if not report:
        return {
            "report": None,
            "message": "No recovery sessions yet",
        }
    return {"report": report}


@app.post("/recovery/trigger")
def trigger_recovery(req: ManualRecoveryRequest):
    if not _orchestrator:
        raise HTTPException(
            503, "Orchestrator not ready"
        )
    valid = {"node-1", "node-2", "node-3"}
    invalid = [
        n for n in req.node_ids if n not in valid
    ]
    if invalid:
        raise HTTPException(
            400, f"Unknown nodes: {invalid}"
        )
    if not req.node_ids:
        raise HTTPException(
            400, "node_ids cannot be empty"
        )

    session_id = _orchestrator.trigger_session_manual(
        node_ids=req.node_ids,
        reason=req.reason,
    )
    if session_id is None:
        active = _orchestrator.get_active_session()
        raise HTTPException(
            409,
            f"Recovery session already active: "
            f"{active['session_id'] if active else 'unknown'}",
        )
    return {
        "session_id": session_id,
        "node_ids": req.node_ids,
        "reason": req.reason,
        "status": "started",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ── Routing endpoints ──────────────────────────────────────

@app.get("/routing")
def get_routing_status():
    if not _quorum_router:
        raise HTTPException(
            503, "Quorum router not ready"
        )
    return _quorum_router.status()


@app.get("/routing/weights")
def get_routing_weights():
    if not _quorum_router:
        raise HTTPException(
            503, "Quorum router not ready"
        )
    weights = _quorum_router.get_weights()
    decision = _quorum_router.get_current_decision()
    stability = _quorum_router.get_weight_stability()
    return {
        "weights": {
            k: round(v, 6) for k, v in weights.items()
        },
        "stability": stability,
        "decision_type": decision[
            "decision_type"
        ] if decision else "unknown",
        "excluded_nodes": decision[
            "excluded_nodes"
        ] if decision else [],
        "quorum_state": decision[
            "quorum_state"
        ] if decision else "unknown",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


@app.get("/routing/route")
def route_single_request():
    if not _quorum_router:
        raise HTTPException(
            503, "Quorum router not ready"
        )
    node = _quorum_router.route_request()
    weights = _quorum_router.get_weights()
    return {
        "selected_node": node,
        "weight_used": round(
            weights.get(node, 0.0), 6
        ),
        "all_weights": {
            k: round(v, 6) for k, v in weights.items()
        },
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


@app.get("/routing/decisions")
def get_routing_decisions(n: int = 10):
    if not _quorum_router:
        raise HTTPException(
            503, "Quorum router not ready"
        )
    return {
        "decisions": _quorum_router.get_decision_history(
            n=n
        ),
        "total": _quorum_router.status()[
            "decision_count"
        ],
    }


# ── Pipeline summary ───────────────────────────────────────

@app.get("/pipeline/summary")
def get_pipeline_summary():
    if not all([
        _updater, _quorum, _coordinator,
        _orchestrator, _quorum_router,
    ]):
        raise HTTPException(
            503, "Components not ready"
        )

    engine_status = _updater.status()
    quorum_status = _quorum.status()
    coord_status = _coordinator.status()
    orch_status = _orchestrator.status()
    router_status = _quorum_router.status()

    effects = {}
    for node_id in ["node-1", "node-2", "node-3"]:
        snap = _updater.get_current_snapshot(node_id)
        if snap:
            effects[node_id] = round(
                abs(snap["effect"]), 4
            )

    return {
        "version": "0.9.0",
        "sprint": 9,
        "engine": {
            "ready": engine_status.get(
                "is_ready", False
            ),
            "buffer_size": engine_status.get(
                "buffer_size", 0
            ),
            "retrain_count": engine_status.get(
                "retrain_count", 0
            ),
            "causal_effects_ms": effects,
        },
        "quorum": {
            "state": quorum_status["quorum_state"],
            "contributing": quorum_status[
                "contributing_nodes"
            ],
            "total": quorum_status["total_nodes"],
            "violations": quorum_status[
                "quorum_violations"
            ],
            "node_states": quorum_status["node_states"],
        },
        "byzantine": {
            "checks": coord_status["check_count"],
            "detections": coord_status[
                "detections_total"
            ],
            "recoveries": coord_status["recoveries_total"],
            "node_states": coord_status["node_states"],
            "byzantine_scores": coord_status[
                "byzantine_scores"
            ],
        },
        "recovery": {
            "sessions_total": orch_status[
                "sessions_total"
            ],
            "sessions_completed": orch_status[
                "sessions_completed"
            ],
            "success_rate": orch_status[
                "session_success_rate"
            ],
            "nodes_recovered": orch_status[
                "nodes_recovered_total"
            ],
            "active_session": orch_status[
                "active_session"
            ] is not None,
        },
        "routing": {
            "checks": router_status["check_count"],
            "decisions": router_status["decision_count"],
            "exclusion_events": router_status[
                "exclusion_events"
            ],
            "cluster_stability": router_status[
                "cluster_stability"
            ],
            "current_weights": router_status[
                "current_weights"
            ],
            "excluded_nodes": router_status[
                "excluded_nodes"
            ],
        },
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "byzantine_recovery_api:app",
        host="0.0.0.0",
        port=8089,
        reload=False,
        log_level="info",
    )