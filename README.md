# CognitiveMesh

**Distributed Byzantine-Resilient Causal Computing Fabric**

A distributed database cluster management system that uses **causal inference** to autonomously detect Byzantine failures, route traffic, and recover nodes — without ever violating quorum safety invariants.

Built in 70 days across 10 sprints by **Kshitij Srivastava** (NIT Surat, 3rd year CS).

---

## What It Does

Most distributed systems react to failures *after* they happen. CognitiveMesh is **predictive**:

- Uses **DoWhy causal inference** to model how database load *causes* query latency — not just correlates with it
- Routes traffic away from nodes that are *about to degrade* under additional load, before users feel it
- Detects **Byzantine failures** through a 3-signal composite scoring model (effect spike + consecutive failures + effect divergence)
- Recovers failed nodes sequentially without ever dropping below **2/3 quorum** — the safety invariant that prevents cascading failure
- Monitors its own **SLA compliance** in real-time and persists every decision to PostgreSQL for audit

---

## Architecture

╔══════════════════════════════════════════════════════════════════════════════╗
║                            CognitiveMesh Architecture                       ║
║          Distributed Byzantine-Resilient Causal Computing Fabric           ║
╚══════════════════════════════════════════════════════════════════════════════╝


                         ┌─────────────────────────────┐
                         │       Client Traffic        │
                         │    Queries / API Requests   │
                         └──────────────┬──────────────┘
                                        │
                                        ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║                         QUORUM-AWARE ROUTING LAYER                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  QuorumAwareRouter                                                         ║
║  • causal-weighted traffic routing                                         ║
║  • exclusion-aware balancing                                               ║
║  • proactive degradation avoidance                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
               ▼                        ▼                        ▼

        ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
        │ PostgreSQL-1 │        │ PostgreSQL-2 │        │ PostgreSQL-3 │
        │    Primary   │        │   Replica    │        │   Replica    │
        └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
               │                       │                       │
               └──────────────┬────────┴────────┬──────────────┘
                              │                 │
                              ▼                 ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║                     STREAMING CAUSAL INTELLIGENCE LAYER                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  StreamingCausalUpdater                                                    ║
║  • DoWhy causal inference engine                                           ║
║  • retrains every 30 seconds                                               ║
║  • models query load → latency effects                                     ║
║  • predicts instability before SLA degradation                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
                              │
      ┌───────────────────────┼────────────────────────┐
      │                       │                        │
      ▼                       ▼                        ▼

┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│ ByzantineCoordinator│ │   HealthMonitor     │ │     RateLimiter     │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│ • 3-signal scoring  │ │ • liveness probes   │ │ • sliding window    │
│ • effect spike      │ │ • readiness probes  │ │ • burst control     │
│ • divergence check  │ │ • SLA monitoring    │ │ • endpoint limits   │
│ • CONFIRMED state   │ │ • uptime tracking   │ │ • abuse prevention  │
└──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
           │                       │                       │
           └──────────────┬────────┴──────────────┬────────┘
                          │                       │
                          ▼                       ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║                         RECOVERY + SAFETY LAYER                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  RecoveryOrchestrator                                                      ║
║  • sequential Byzantine-safe remediation                                   ║
║  • MTTR tracking                                                           ║
║  • rollback-aware recovery lifecycle                                       ║
║                                                                              ║
║  QuorumManager                                                             ║
║  • hard quorum invariant enforcement                                       ║
║  • minimum quorum = 2/3                                                    ║
║  • max concurrent recoveries = 1                                           ║
║  • split-brain prevention                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                        │
                                        ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║                         PERSISTENCE + AUDIT LAYER                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SessionPersistence                                                        ║
║  • async PostgreSQL write queue                                            ║
║  • recovery audit logs                                                     ║
║  • zero hot-path blocking                                                  ║
║  • persistence enqueue latency = 0.001 ms                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                        │
                                        ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║                      OBSERVABILITY + OPERATIONS LAYER                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PrometheusExporter                                                        ║
║  • metrics aggregation                                                     ║
║  • real-time telemetry export                                              ║
║  • 40-panel Grafana dashboard                                              ║
║                                                                              ║
║  ByzantineRecoveryAPI                                                      ║
║  • REST interface                                                          ║
║  • operational recovery controls                                           ║
║  • port 8089                                                               ║
╚════════════════════════════════════════════════════════════════════════════

---

## Performance

Benchmarked on Windows 11, Python 3.13, 1000 iterations:

| Operation | Mean Latency | Result |
|-----------|-------------|--------|
| route_request() | 0.0023ms | PASS |
| byzantine_score() | 0.0017ms | PASS |
| quorum_decision() | 0.0072ms | PASS |
| health_check_cycle() | 0.1218ms | PASS |
| rate_limiter_check() | 0.0066ms | PASS |
| persistence_enqueue() | 0.0010ms | PASS |
| **Throughput (4 threads)** | **142,736 req/s** | **PASS** |

**33/33 benchmarks PASS**

---

## Test Suite
test_graph_builder.py         27 passed
test_integration.py           29 passed
test_zk_layer.py              30 passed
test_byzantine_layer.py       46 passed
test_predictive_layer.py      50 passed
test_self_healing_layer.py    59 passed
test_observability_layer.py   50 passed
test_multi_node_byzantine.py  73 passed
test_sprint10_integration.py  72 passed
─────────────────────────────────────────
Total                        436 passed   0 failed   64.84s

---

## Service Map

| Port | Service |
|------|---------|
| 5436-5438 | PostgreSQL nodes (node-1/2/3) |
| 8080 | Telemetry API |
| 8081 | Causal Query API |
| 8082 | Unified Pipeline |
| 8083 | ZK Causal API |
| 8084 | ZK Unified Pipeline |
| 8085 | Byzantine API |
| 8086 | Predictive Intelligence API |
| 8087 | Self-Healing API |
| 8088 | Prometheus Metrics Exporter |
| 8089 | Byzantine Recovery API |
| 9090 | Prometheus |
| 3000 | Grafana |

---

## Quick Start

**Docker Compose (recommended):**
```bash
docker compose up -d
# Health check:
curl http://localhost:8089/health
```

**Local development:**
```bash
# Terminal 1: load generator
python pillar1-causal/telemetry/load_generator.py

# Terminal 2: full API stack
cd pillar1-causal/causal
python byzantine_recovery_api.py

# Run all tests
python -m pytest pillar1-causal/causal/tests/ -v

# Run benchmarks
python pillar1-causal/causal/benchmarks/benchmark_suite.py
```

**Prerequisites:** Python 3.13, PostgreSQL 15, Docker (optional)

```bash
pip install -r requirements.txt
cp .env.docker .env   # edit PG credentials if needed
```

---

## Key Design Decisions

**1. Causal effects for routing, not raw latency**
Raw latency is reactive. A causal effect estimate (`1 unit increase in active_queries causes Xms latency`) is predictive — it quantifies how sensitive a node is to *additional* load, not just current load.

**2. Quorum gate as a single safety primitive**
Every recovery action passes through `QuorumManager.request_node_offline()`. This is the only place where the safety invariant (`MINIMUM_QUORUM=2/3`) is enforced. Centralising it prevents race conditions and makes the invariant auditable.

**3. Sequential Byzantine recovery**
With 3 nodes and `MINIMUM_QUORUM=2`, recovering two nodes simultaneously would leave 1 contributor. Any failure of that node = total cluster loss. `MAX_CONCURRENT_RECOVERIES=1` is a hard invariant, not a soft preference.

**4. Composite Byzantine scoring**
Three independent signals (effect spike, consecutive failures, effect divergence) each contribute a weighted score. A single high-effect reading alone scores at most 0.40 — below the `SUSPECTED` threshold of 0.35 + failure component. This reduces false positives dramatically in noisy real workloads.

**5. Async persistence queue**
Recovery decisions must never block waiting for database writes. The async deque + background writer adds zero latency to the hot path (enqueue = 0.001ms).

---

## Sprint History

| Version | Sprint | Key Deliverable |
|---------|--------|----------------|
| v0.1.0 | 1 | Telemetry pipeline, live PostgreSQL metrics |
| v0.2.0 | 2 | DoWhy causal graph construction + effect estimation |
| v0.3.0 | 3 | Real-time streaming causal integration |
| v0.4.0 | 4 | Zero-knowledge proof security layer |
| v0.5.0 | 5 | Byzantine consensus detection |
| v0.6.0 | 6 | Predictive intelligence (load forecasting) |
| v0.7.0 | 7 | Self-healing fabric (automated remediation) |
| v0.8.0 | 8 | Federated observability (Prometheus + Grafana) |
| v0.9.0 | 9 | Multi-node Byzantine recovery with quorum safety |
| v1.0.0 | 10 | Production hardening (persistence, health, Docker) |

---

## Tech Stack

- **Language:** Python 3.13
- **Database:** PostgreSQL 15 (3-node cluster)
- **Causal Inference:** DoWhy 0.11 (linear regression estimator)
- **API:** FastAPI + Uvicorn
- **Observability:** Prometheus + Grafana
- **Deployment:** Docker Compose
- **Testing:** pytest (436 tests)
- **Platform:** Windows 11

---

## Author

**Kshitij Srivastava**
NIT Surat — 3rd year Computer Science
CGPA: 6.14

Built entirely during summer 2026 as a placement preparation project demonstrating distributed systems, causal inference, Byzantine fault tolerance, and production engineering skills.

---

*v1.0.0 — May 2026 — 70 days — 10 sprints — 1 developer*
