import sys, os, time, logging
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "telemetry"))
from session_persistence import SessionPersistence
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
nodes = [("node-1", 5436), ("node-2", 5437), ("node-3", 5438)]
results = {}
for node_id, port in nodes:
    dsn = "host=127.0.0.1 port=" + str(port) + " dbname=postgres user=cm_user password=cm_secret connect_timeout=5"
    p = SessionPersistence(dsn=dsn)
    p.start()
    time.sleep(2)
    stats = p.get_stats()
    results[node_id] = {"connected": stats["connected"], "port": port}
    print("Node " + node_id + " port=" + str(port) + " connected=" + str(stats["connected"]))
    if stats["connected"]:
        p.save_session({"session_id": "test-" + node_id + "-d64", "trigger": "demo_day64", "phase": "completed", "affected_nodes": [node_id], "node_outcomes": {node_id: "success"}, "node_durations_seconds": {node_id: 15.3}, "node_effects_at_start": {node_id: 45.2}, "node_effects_at_end": {node_id: 27.8}, "quorum_state_at_start": "degraded", "quorum_state_at_end": "healthy", "success_count": 1, "failure_count": 0, "total_duration_seconds": 18.5, "mttr_seconds": 15.3, "notes": ["demo"], "plan_ids": [], "started_at": datetime.now(timezone.utc).isoformat(), "completed_at": datetime.now(timezone.utc).isoformat()})
        p.save_decision({"node_id": node_id, "reason": "demo_day64", "decision": "allow", "quorum_state": "healthy", "contributing": 3, "recovering": 0})
        p.save_sla_snapshot({"quorum_state": "healthy", "contributing_nodes": 3, "recovering_nodes": 0, "causal_effects": {"node-1": 27.5, "node-2": 28.5, "node-3": 29.0}, "routing_weights": {"node-1": 0.35, "node-2": 0.33, "node-3": 0.32}, "excluded_nodes": [], "cluster_stability": 0.965, "active_session": False, "byzantine_detections": 0})
        time.sleep(6)
        sessions = p.get_sessions(5)
        decisions = p.get_decisions(5)
        sla = p.get_sla_snapshots(5)
        fs = p.get_stats()
        print("  sessions_read=" + str(len(sessions)) + " decisions_read=" + str(len(decisions)) + " sla_read=" + str(len(sla)))
        print("  db_sessions=" + str(fs.get("db_sessions","N/A")) + " db_decisions=" + str(fs.get("db_decisions","N/A")) + " db_sla=" + str(fs.get("db_sla_snapshots","N/A")))
        print("  writes_total=" + str(fs["writes_total"]) + " writes_failed=" + str(fs["writes_failed"]))
    p.stop()
    print("")
print("=== SUMMARY ===")
for node_id, r in results.items():
    print("  " + node_id + " port=" + str(r["port"]) + " connected=" + str(r["connected"]))
