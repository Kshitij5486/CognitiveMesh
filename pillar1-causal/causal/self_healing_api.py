import logging
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

from streaming_updater import StreamingCausalUpdater
from load_trend_analyzer import LoadTrendAnalyzer
from causal_simulator import CausalSimulator
from predictive_alerter import PredictiveAlerter
from healing_action_engine import HealingActionEngine, HealingActionType
from query_router import QueryRouter, RoutingStrategy
from auto_retrainer import AutoRetrainer, RetrainTrigger
from recovery_orchestrator import RecoveryOrchestrator, RecoveryTrigger

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.healing.api")

updater: Optional[StreamingCausalUpdater] = None
analyzer: Optional[LoadTrendAnalyzer] = None
simulator: Optional[CausalSimulator] = None
alerter: Optional[PredictiveAlerter] = None
engine: Optional[HealingActionEngine] = None
router: Optional[QueryRouter] = None
retrainer: Optional[AutoRetrainer] = None
orchestrator: Optional[RecoveryOrchestrator] = None
api_start_time: Optional[str] = None


class HealRequest(BaseModel):
    node_id: str
    alert_type: str
    severity: str


class RecoveryRequest(BaseModel):
    node_id: str
    reason: str = "Manual recovery via API"


class RetrainRequest(BaseModel):
    node_ids: list
    reason: str = "Manual retrain via API"


class RerouteRequest(BaseModel):
    node_id: str
    reason: str = "Manual reroute via API"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global updater, analyzer, simulator, alerter
    global engine, router, retrainer, orchestrator
    global api_start_time

    api_start_time = datetime.now(timezone.utc).isoformat()
    logger.info("Starting Self-Healing API v0.7.0")

    updater = StreamingCausalUpdater(
        collection_interval_seconds=3.0,
        retrain_interval_seconds=30.0,
        min_samples_for_training=30,
        max_buffer_size=200,
    )
    updater.start()

    analyzer = LoadTrendAnalyzer(
        updater=updater,
        observation_interval_seconds=3.0,
        analysis_interval_seconds=15.0,
    )
    analyzer.start()

    simulator = CausalSimulator(
        updater=updater,
        analyzer=analyzer,
        simulation_interval_seconds=30.0,
    )
    simulator.start()

    alerter = PredictiveAlerter(
        updater=updater,
        analyzer=analyzer,
        simulator=simulator,
        check_interval=15.0,
    )
    alerter.start()

    engine = HealingActionEngine(
        updater=updater,
        alerter=alerter,
        check_interval=15.0,
        auto_heal=True,
    )
    engine.start()

    router = QueryRouter(
        updater=updater,
        alerter=alerter,
        strategy=RoutingStrategy.CAUSAL_WEIGHTED,
        check_interval=15.0,
    )
    router.start()

    retrainer = AutoRetrainer(
        updater=updater,
        alerter=alerter,
        check_interval=30.0,
        drift_threshold_ms=3.0,
        auto_retrain=True,
    )
    retrainer.start()

    orchestrator = RecoveryOrchestrator(
        updater=updater,
        alerter=alerter,
        engine=engine,
        router=router,
        retrainer=retrainer,
        check_interval=15.0,
        auto_recover=True,
    )
    orchestrator.start()

    logger.info("Self-Healing API fully started — all components running")
    yield

    logger.info("Shutting down Self-Healing API")
    for component in [
        orchestrator, retrainer, router,
        engine, alerter, simulator, analyzer, updater
    ]:
        if component:
            component.stop()


app = FastAPI(
    title="CognitiveMesh Self-Healing API",
    description=(
        "Autonomous self-healing fabric for the CognitiveMesh "
        "distributed computing cluster. Integrates healing action "
        "engine, query router, auto-retrainer, and recovery "
        "orchestrator."
    ),
    version="0.7.0",
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
    engine_stat = engine.status() if engine else {}
    router_stat = router.status() if router else {}
    retrainer_stat = retrainer.status() if retrainer else {}
    orch_stat = orchestrator.status() if orchestrator else {}

    uptime = 0.0
    if api_start_time:
        uptime = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(api_start_time)
        ).total_seconds()

    return {
        "status": "healthy",
        "version": "0.7.0",
        "api_start": api_start_time,
        "uptime_seconds": round(uptime, 1),
        "causal_engine": {
            "ready": engine_status.get("is_ready", False),
            "buffer_size": engine_status.get("buffer_size", 0),
            "retrain_count": engine_status.get("retrain_count", 0),
        },
        "healing_engine": {
            "checks_run": engine_stat.get("checks_run", 0),
            "total_actions": engine_stat.get("total_actions", 0),
            "successful_actions": engine_stat.get(
                "successful_actions", 0
            ),
        },
        "router": {
            "strategy": router_stat.get("strategy", "unknown"),
            "total_reroutes": router_stat.get("total_reroutes", 0),
            "active_nodes": router_stat.get("active_nodes", 0),
            "rerouted_nodes": router_stat.get("rerouted_nodes", 0),
        },
        "retrainer": {
            "total_retrains": retrainer_stat.get(
                "total_retrains", 0
            ),
            "successful_retrains": retrainer_stat.get(
                "successful_retrains", 0
            ),
        },
        "orchestrator": {
            "total_sequences": orch_stat.get("total_sequences", 0),
            "successful_recoveries": orch_stat.get(
                "successful_recoveries", 0
            ),
            "active_sequences": orch_stat.get("active_sequences", 0),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/heal")
def healing_status():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    status = engine.status()
    history = engine.get_action_history(n=10)
    return {
        "status": status,
        "recent_actions": history,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/heal")
def trigger_heal(request: HealRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    if request.node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {request.node_id}"
        )
    result = engine.heal_now(
        node_id=request.node_id,
        alert_type=request.alert_type,
        severity=request.severity,
    )
    if not result:
        return {
            "triggered": False,
            "reason": "No action required for this alert type/severity",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "triggered": True,
        "action": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/heal/{node_id}")
def node_healing_history(node_id: str):
    if node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {node_id}"
        )
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    history = engine.get_action_history(n=20)
    node_history = [
        a for a in history
        if a["node_id"] == node_id
    ]
    return {
        "node_id": node_id,
        "action_count": len(node_history),
        "actions": node_history,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/router")
def routing_table():
    if not router:
        raise HTTPException(status_code=503, detail="Router not ready")
    table = router.get_routing_table()
    decisions = router.get_active_decisions()
    status = router.status()
    return {
        "routing_table": table,
        "active_decisions": decisions,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/router/reroute")
def trigger_reroute(request: RerouteRequest):
    if not router:
        raise HTTPException(status_code=503, detail="Router not ready")
    if request.node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {request.node_id}"
        )
    decision = router.reroute_node(
        node_id=request.node_id,
        reason=request.reason,
    )
    if not decision:
        return {
            "rerouted": False,
            "reason": "Cooldown active or no targets available",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "rerouted": True,
        "decision": decision.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/router/restore/{node_id}")
def restore_node(node_id: str):
    if node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {node_id}"
        )
    if not router:
        raise HTTPException(status_code=503, detail="Router not ready")
    success = router.restore_node(node_id)
    return {
        "restored": success,
        "node_id": node_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/retrain")
def retrainer_status():
    if not retrainer:
        raise HTTPException(
            status_code=503, detail="Retrainer not ready"
        )
    status = retrainer.status()
    history = retrainer.get_record_history(n=10)
    return {
        "status": status,
        "recent_retrains": history,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/retrain")
def trigger_retrain(request: RetrainRequest):
    if not retrainer:
        raise HTTPException(
            status_code=503, detail="Retrainer not ready"
        )
    valid_nodes = [
        n for n in request.node_ids
        if n in ["node-1", "node-2", "node-3"]
    ]
    if not valid_nodes:
        raise HTTPException(
            status_code=400,
            detail="No valid node IDs provided"
        )
    engine_status = updater.status() if updater else {}
    if not engine_status.get("is_ready"):
        raise HTTPException(
            status_code=503,
            detail="Causal engine not ready"
        )
    record = retrainer.trigger_retrain(
        node_ids=valid_nodes,
        trigger=RetrainTrigger.MANUAL,
        reason=request.reason,
    )
    if not record:
        return {
            "triggered": False,
            "reason": "Retrain skipped — cooldown active",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "triggered": True,
        "record": record.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/recovery")
def recovery_status():
    if not orchestrator:
        raise HTTPException(
            status_code=503, detail="Orchestrator not ready"
        )
    status = orchestrator.status()
    active = orchestrator.get_active_sequences()
    history = orchestrator.get_sequence_history(n=10)
    return {
        "status": status,
        "active_sequences": active,
        "recent_history": history,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/recovery")
def trigger_recovery(request: RecoveryRequest):
    if not orchestrator:
        raise HTTPException(
            status_code=503, detail="Orchestrator not ready"
        )
    if request.node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {request.node_id}"
        )
    result = orchestrator.trigger_manual_recovery(
        node_id=request.node_id,
        reason=request.reason,
    )
    if not result:
        return {
            "triggered": False,
            "reason": "Recovery already active for this node",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "triggered": True,
        "sequence": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/recovery/{node_id}")
def node_recovery_history(node_id: str):
    if node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {node_id}"
        )
    if not orchestrator:
        raise HTTPException(
            status_code=503, detail="Orchestrator not ready"
        )
    history = orchestrator.get_sequence_history(n=20)
    node_history = [
        s for s in history
        if s["node_id"] == node_id
    ]
    return {
        "node_id": node_id,
        "sequence_count": len(node_history),
        "sequences": node_history,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/pipeline/summary")
def pipeline_summary():
    engine_status = updater.status() if updater else {}
    analyzer_stat = analyzer.status() if analyzer else {}
    simulator_stat = simulator.status() if simulator else {}
    alerter_stat = alerter.status() if alerter else {}
    engine_stat = engine.status() if engine else {}
    router_stat = router.status() if router else {}
    retrainer_stat = retrainer.status() if retrainer else {}
    orch_stat = orchestrator.status() if orchestrator else {}

    uptime = 0.0
    if api_start_time:
        uptime = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(api_start_time)
        ).total_seconds()

    return {
        "pipeline": {
            "version": "0.7.0",
            "start_time": api_start_time,
            "uptime_seconds": round(uptime, 1),
        },
        "causal_engine": {
            "ready": engine_status.get("is_ready", False),
            "buffer_size": engine_status.get("buffer_size", 0),
            "retrain_count": engine_status.get("retrain_count", 0),
            "nodes_modeled": engine_status.get("nodes_modeled", []),
        },
        "predictive_stack": {
            "trend_observations": analyzer_stat.get(
                "observations_collected", 0
            ),
            "simulations_run": simulator_stat.get(
                "simulations_run", 0
            ),
            "alerts_fired": alerter_stat.get(
                "total_alerts_fired", 0
            ),
            "active_alerts": alerter_stat.get(
                "active_alert_count", 0
            ),
        },
        "healing_stack": {
            "total_actions": engine_stat.get("total_actions", 0),
            "successful_actions": engine_stat.get(
                "successful_actions", 0
            ),
            "total_reroutes": router_stat.get("total_reroutes", 0),
            "total_recoveries": router_stat.get(
                "total_recoveries", 0
            ),
            "total_retrains": retrainer_stat.get(
                "total_retrains", 0
            ),
        },
        "recovery_stack": {
            "total_sequences": orch_stat.get("total_sequences", 0),
            "successful_recoveries": orch_stat.get(
                "successful_recoveries", 0
            ),
            "failed_recoveries": orch_stat.get(
                "failed_recoveries", 0
            ),
            "active_sequences": orch_stat.get("active_sequences", 0),
        },
        "node_states": router_stat.get("node_states", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "self_healing_api:app",
        host="0.0.0.0",
        port=8087,
        reload=False,
        log_level="info",
    )