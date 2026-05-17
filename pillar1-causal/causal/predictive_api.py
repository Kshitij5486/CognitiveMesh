import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from predictive_byzantine_bridge import ByzantinePredictiveBridge
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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cm.predictive.api")

updater: Optional[StreamingCausalUpdater] = None
analyzer: Optional[LoadTrendAnalyzer] = None
simulator: Optional[CausalSimulator] = None
alerter: Optional[PredictiveAlerter] = None
bridge: Optional[ByzantinePredictiveBridge] = None
api_start_time: Optional[str] = None


class AcknowledgeRequest(BaseModel):
    alert_id: str


class ClearRequest(BaseModel):
    alert_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global updater, analyzer, simulator, alerter, api_start_time
    api_start_time = datetime.now(timezone.utc).isoformat()
    logger.info("Starting Predictive Intelligence API")

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
    bridge = ByzantinePredictiveBridge(
        alerter=alerter,
        poll_interval=15.0,
    )
    bridge.start()

    logger.info(
        "Predictive Intelligence API started — "
        "trend analyzer, simulator, alerter all running"
    )
    yield

    logger.info("Shutting down Predictive Intelligence API")
    if alerter:
        alerter.stop()
    if bridge:
        bridge.stop()    
    if simulator:
        simulator.stop()
    if analyzer:
        analyzer.stop()
    if updater:
        updater.stop()


app = FastAPI(
    title="CognitiveMesh Predictive Intelligence API",
    description=(
        "Predicts cluster performance degradation before it happens "
        "using causal inference, load trend analysis, and "
        "simulation-based forecasting."
    ),
    version="0.6.0",
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
    analyzer_status = analyzer.status() if analyzer else {}
    simulator_status = simulator.status() if simulator else {}
    alerter_status = alerter.status() if alerter else {}

    return {
        "status": "healthy",
        "version": "0.6.0",
        "api_start": api_start_time,
        "causal_engine": {
            "ready": engine_status.get("is_ready", False),
            "buffer_size": engine_status.get("buffer_size", 0),
            "retrain_count": engine_status.get("retrain_count", 0),
        },
        "trend_analyzer": {
            "observations": analyzer_status.get(
                "observations_collected", 0
            ),
            "analyses_run": analyzer_status.get("analyses_run", 0),
            "tracker_sizes": analyzer_status.get("tracker_sizes", {}),
        },
        "byzantine_bridge": {
            "available": bridge.status()["byzantine_api_available"] if bridge else False,
            "active_nodes": bridge.status()["active_nodes"] if bridge else [],
            "isolated_nodes": bridge.status()["isolated_nodes"] if bridge else [],
        },
        "simulator": {
            "simulations_run": simulator_status.get(
                "simulations_run", 0
            ),
            "has_simulation": simulator_status.get(
                "has_simulation", False
            ),
            "worst_latency_ms": simulator_status.get(
                "cluster_worst_case_latency_ms", 0.0
            ),
        },
        "alerter": {
            "checks_run": alerter_status.get("checks_run", 0),
            "active_alerts": alerter_status.get(
                "active_alert_count", 0
            ),
            "total_fired": alerter_status.get(
                "total_alerts_fired", 0
            ),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/trend")
def cluster_trend():
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not ready")

    cluster = analyzer.get_cluster_trend()
    analyses = analyzer.get_latest_analyses()

    return {
        "cluster": cluster,
        "nodes": analyses,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/trend/{node_id}")
def node_trend(node_id: str):
    if node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {node_id}"
        )
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not ready")

    analysis = analyzer.get_analysis(node_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"No trend analysis for {node_id} yet"
        )

    return analysis


@app.get("/simulate")
def cluster_simulation():
    if not simulator:
        raise HTTPException(
            status_code=503, detail="Simulator not ready"
        )

    sim = simulator.get_latest_simulation()
    if not sim:
        sim = simulator.simulate_now()

    if not sim:
        raise HTTPException(
            status_code=503,
            detail="No simulation available — trend analyses still accumulating"
        )

    return sim


@app.post("/simulate")
def run_simulation_now():
    if not simulator:
        raise HTTPException(
            status_code=503, detail="Simulator not ready"
        )

    engine_status = updater.status() if updater else {}
    if not engine_status.get("is_ready"):
        raise HTTPException(
            status_code=503,
            detail=f"Causal engine not ready. "
                   f"Buffer: {engine_status.get('buffer_size', 0)}/30"
        )

    sim = simulator.simulate_now()
    if not sim:
        raise HTTPException(
            status_code=503,
            detail="Simulation failed — trend analyses not available"
        )

    return sim


@app.get("/simulate/{node_id}")
def node_simulation(node_id: str):
    if node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {node_id}"
        )
    if not simulator:
        raise HTTPException(
            status_code=503, detail="Simulator not ready"
        )

    node_sim = simulator.get_node_simulation(node_id)
    if not node_sim:
        raise HTTPException(
            status_code=404,
            detail=f"No simulation for {node_id} yet"
        )

    return node_sim


@app.get("/alerts")
def list_alerts():
    if not alerter:
        raise HTTPException(
            status_code=503, detail="Alerter not ready"
        )

    active = alerter.get_active_alerts()
    history = alerter.get_alert_history(n=20)
    status = alerter.status()

    return {
        "active_count": len(active),
        "active_alerts": active,
        "recent_history": history,
        "thresholds": status["thresholds"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/alerts/active")
def active_alerts():
    if not alerter:
        raise HTTPException(
            status_code=503, detail="Alerter not ready"
        )

    active = alerter.get_active_alerts()
    return {
        "count": len(active),
        "alerts": active,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/alerts/acknowledge")
def acknowledge_alert(request: AcknowledgeRequest):
    if not alerter:
        raise HTTPException(
            status_code=503, detail="Alerter not ready"
        )

    success = alerter.acknowledge_alert(request.alert_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {request.alert_id} not found"
        )

    return {
        "acknowledged": True,
        "alert_id": request.alert_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/alerts/clear")
def clear_alert(request: ClearRequest):
    if not alerter:
        raise HTTPException(
            status_code=503, detail="Alerter not ready"
        )

    success = alerter.clear_alert(request.alert_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {request.alert_id} not found"
        )

    return {
        "cleared": True,
        "alert_id": request.alert_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/forecast/{node_id}")
def node_forecast(node_id: str, horizon_minutes: int = 5):
    if node_id not in ["node-1", "node-2", "node-3"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown node: {node_id}"
        )
    if not all([updater, analyzer, simulator]):
        raise HTTPException(status_code=503, detail="Stack not ready")

    engine_status = updater.status()
    if not engine_status.get("is_ready"):
        raise HTTPException(
            status_code=503,
            detail="Causal engine not ready"
        )

    snapshot = updater.get_current_snapshot(node_id)
    if not snapshot:
        raise HTTPException(
            status_code=404,
            detail=f"No causal model for {node_id}"
        )

    trend = analyzer.get_analysis(node_id)
    node_sim = simulator.get_node_simulation(node_id)

    causal_effect = abs(snapshot["effect"])
    current_load = trend["current_load"] if trend else 0.0
    change_rate = trend["change_rate_per_minute"] if trend else 0.0

    projected_load = max(0.0, current_load + change_rate * horizon_minutes)
    projected_latency = causal_effect * projected_load

    risk_level = "low"
    if projected_latency >= 600:
        risk_level = "emergency"
    elif projected_latency >= 300:
        risk_level = "critical"
    elif projected_latency >= 150:
        risk_level = "warning"
    elif projected_latency >= 50:
        risk_level = "elevated"

    return {
        "node_id": node_id,
        "horizon_minutes": horizon_minutes,
        "causal_effect_ms_per_query": round(causal_effect, 4),
        "current_load_queries": round(current_load, 2),
        "projected_load_queries": round(projected_load, 2),
        "projected_latency_ms": round(projected_latency, 2),
        "change_rate_per_minute": round(change_rate, 4),
        "trend_direction": trend["direction"] if trend else "unknown",
        "risk_level": risk_level,
        "interpretation": (
            f"In {horizon_minutes} minutes, {node_id} is projected to "
            f"handle {projected_load:.1f} concurrent queries at "
            f"{causal_effect:.1f}ms causal effect per query, "
            f"resulting in {projected_latency:.1f}ms total latency. "
            f"Risk: {risk_level.upper()}."
        ),
        "model_info": {
            "samples_used": snapshot["samples_used"],
            "last_updated": snapshot["timestamp"],
            "retrain_count": engine_status["retrain_count"],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/pipeline/summary")
def pipeline_summary():
    engine_status = updater.status() if updater else {}
    analyzer_status = analyzer.status() if analyzer else {}
    simulator_status = simulator.status() if simulator else {}
    alerter_status = alerter.status() if alerter else {}

    uptime = 0.0
    if api_start_time:
        uptime = (
            datetime.now(timezone.utc) -
            datetime.fromisoformat(api_start_time)
        ).total_seconds()

    return {
        "pipeline": {
            "version": "0.6.0",
            "start_time": api_start_time,
            "uptime_seconds": round(uptime, 1),
        },
        "causal_engine": {
            "ready": engine_status.get("is_ready", False),
            "buffer_size": engine_status.get("buffer_size", 0),
            "retrain_count": engine_status.get("retrain_count", 0),
            "nodes_modeled": engine_status.get("nodes_modeled", []),
        },
        "trend_analyzer": {
            "observations_collected": analyzer_status.get(
                "observations_collected", 0
            ),
            "analyses_run": analyzer_status.get("analyses_run", 0),
            "nodes_with_trends": analyzer_status.get(
                "latest_analyses_available", []
            ),
        },
        "simulator": {
            "simulations_run": simulator_status.get(
                "simulations_run", 0
            ),
            "has_simulation": simulator_status.get(
                "has_simulation", False
            ),
            "highest_risk_node": simulator_status.get(
                "highest_risk_node"
            ),
            "cluster_worst_case_ms": simulator_status.get(
                "cluster_worst_case_latency_ms", 0.0
            ),
        },
        "alerter": {
            "checks_run": alerter_status.get("checks_run", 0),
            "total_alerts_fired": alerter_status.get(
                "total_alerts_fired", 0
            ),
            "active_alert_count": alerter_status.get(
                "active_alert_count", 0
            ),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
@app.get("/byzantine")
def byzantine_status():
    if not bridge:
        raise HTTPException(status_code=503, detail="Bridge not ready")

    bridge_data = bridge.status()
    trusted = bridge.get_trusted_nodes()

    sim = simulator.get_latest_simulation() if simulator else None
    if sim and bridge:
        sim = bridge.enrich_simulation(sim)

    return {
        "bridge": bridge_data,
        "trusted_nodes_for_forecasting": trusted,
        "enriched_simulation": sim,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

if __name__ == "__main__":
    uvicorn.run(
        "predictive_api:app",
        host="0.0.0.0",
        port=8086,
        reload=False,
        log_level="info",
    )