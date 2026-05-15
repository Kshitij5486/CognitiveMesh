# Sprint 1 Research Notes
## CognitiveMesh — Pillar 1: Causal Consciousness Engine

**Sprint duration:** 2 weeks  
**Completed:** May 2026  
**Author:** Kshitij Srivastava, NIT Surat

---

## What was built

Sprint 1 establishes the observability foundation of CognitiveMesh. The core
deliverable is a real-time telemetry pipeline that captures, streams, stores,
and serves behavioral data from a 3-node PostgreSQL cluster.

### Components

**Telemetry Collector** (`collector.py`)  
Connects to each PostgreSQL node and queries `pg_stat_activity`, `pg_locks`,
and `pg_stat_bgwriter` on a 5-second interval. Captures query state, lock
acquisition, and background IO metrics per node.

**Kafka Producer** (`producer.py`)  
Publishes telemetry events to four Kafka topics: `cm.query.events`,
`cm.lock.events`, `cm.io.events`, `cm.node.events`. Each event carries
node_id, event_type, timestamp, and domain-specific fields. Uses a
python-ng Kafka client with `api_version=(2, 5, 0)` for compatibility
with Confluent Kafka 7.4.0.

**Kafka Consumer + EventStore** (`consumer.py`)  
A background thread subscribes to all four topics and stores events in a
thread-safe in-memory store using `collections.deque` with a configurable
per-node cap of 10,000 events. Provides a query API for retrieving events
by node, type, and recency.

**FastAPI Service** (`api.py`)  
Exposes five REST endpoints over the EventStore. Runs on port 8080. Starts
and stops the Kafka consumer via FastAPI lifespan context.

**Load Generator** (`load_generator.py`)  
Fires concurrent queries across all three nodes using Python threads. At
30 workers with 5 QPS per worker, the generator sustains approximately
85 successful queries per second across the cluster.

**Benchmark** (`benchmark.py`)  
Measures telemetry collection overhead per node by comparing baseline
`SELECT 1` latency against the three telemetry queries. Also measures
Kafka end-to-end latency from publish timestamp to consume timestamp.

---

## Infrastructure

| Component    | Image                          | Port  |
|--------------|-------------------------------|-------|
| pg-node1     | postgres:15                   | 5436  |
| pg-node2     | postgres:15                   | 5437  |
| pg-node3     | postgres:15                   | 5438  |
| zookeeper    | confluentinc/cp-zookeeper:7.4 | 2181  |
| kafka        | confluentinc/cp-kafka:7.4     | 9093  |

---

## Key technical decisions

**Mersenne prime field vs standard integers**  
Not applicable at this layer. Sprint 1 operates at the observability layer,
not the cryptographic layer. Cryptographic primitives are introduced in
Sprint 5 (Pillar 2).

**Why pg_stat_activity instead of query logging**  
Query logging writes to disk and introduces measurable latency under high
concurrency. `pg_stat_activity` is a live view of the shared memory catalog
and has negligible overhead. Measured overhead is under 5ms per collection
cycle per node.

**Why Kafka over direct database writes**  
Kafka decouples the collector from the consumer. If the consumer is slow
or offline, events buffer in Kafka and are replayed when the consumer
reconnects. This is essential for the causal graph builder in Sprint 2,
which needs to process events in order without dropping any.

**Why deque with maxlen over a database**  
For Sprint 1, the in-memory store is sufficient. The causal graph builder
in Sprint 2 will need fast random access to recent events, which a deque
provides in O(1). A database would introduce network latency on every
read. This will be revisited in Sprint 3 when persistence becomes important.

---

## Benchmark results

To be filled after running `benchmark.py`.

| Metric                          | node-1  | node-2  | node-3  |
|---------------------------------|---------|---------|---------|
| Baseline SELECT 1 mean (ms)     | 2.297   | 1.291   | 0.782   |
| pg_stat_activity mean (ms)      | 2.801   | 2.106   | 1.230   |
| pg_locks mean (ms)              | 1.463   | 0.955   | 0.768   |
| pg_stat_bgwriter mean (ms)      | 1.242   | 0.785   | 0.747   |
| Total overhead per cycle (ms)   | 3.209   | 2.555   | 1.963   |

| Kafka metric                        | Value   |
|-------------------------------------|---------|
| Publish to consume mean (ms)        | 6.421   |
| Publish to consume min (ms)         | 3.898   |
| Publish to consume max (ms)         | 11.348  |

---

## What was not built in Sprint 1

- Causal graph construction (Sprint 2)
- Cross-node causal correlation (Sprint 2)
- Persistence layer for events (Sprint 3)
- ZK proof integration (Sprint 5)

---

## Research questions opened by Sprint 1

1. At what event volume does the in-memory EventStore become a bottleneck?
2. Can the telemetry collection interval be reduced to 1 second without
   measurable impact on query latency?
3. What is the minimum set of telemetry variables needed to build a valid
   causal graph? Can we reduce the 6 variables currently collected?
4. How does Kafka consumer lag behave under sustained 500+ QPS load?

---

## Next sprint

Sprint 2 builds the causal graph engine on top of this telemetry foundation.
The goal is a live causal graph that answers the question:
"Why is node-2 experiencing high latency right now?"
using DoWhy and the live event stream from Sprint 1.