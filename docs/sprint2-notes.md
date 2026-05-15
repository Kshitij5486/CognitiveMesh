# Sprint 2 Research Notes
## CognitiveMesh — Pillar 1: Causal Consciousness Engine

**Sprint duration:** 2 weeks  
**Completed:** May 2026  
**Author:** Kshitij Srivastava, NIT Surat

---

## What was built

Sprint 2 implements the causal reasoning layer of CognitiveMesh. The system
can now answer the question "why is this node slow?" using mathematically
identified causal relationships — not correlations, not heuristics, not
rule-based alerts. Causation.

### Components

**Causal Graph Builder** (`graph_builder.py`)  
Builds a DoWhy CausalModel per node using a networkx DiGraph as the causal
structure. Treatment variable is `active_queries`. Outcome variable is
`avg_query_duration_ms`. Uses backdoor linear regression for effect
estimation. Includes placebo treatment refutation for validation.

**Cross-Node Causal Correlator** (`cross_node_causal.py`)  
Builds an 18-node, 22-edge distributed causal graph spanning all three
PostgreSQL nodes. Estimates 10 cross-node causal effects. Identifies causal
chains propagating from node-1 to node-2 and node-3. Finds that active
query load on node-1 causally propagates to downstream nodes with ~1.0
unit effect on latency.

**Causal Query API** (`causal_api.py`)  
FastAPI service on port 8081. Trains the causal engine in a background
thread on startup. Exposes `/why/{node_id}`, `/compare`, `/retrain`,
`/status` endpoints. Returns causal explanations in natural language with
quantified effect sizes.

**Test Suite** (`tests/test_graph_builder.py`)  
27 unit and integration tests covering NodeCausalModel,
CrossNodeCausalGraph, and DistributedCausalCorrelator. Zero failures.
Tests use synthetic datasets with known causal structure to verify
correctness of identification and estimation.

**Benchmark Suite** (`benchmark_causal.py`)  
7 benchmark metrics covering build time, identification time, estimation
time, full pipeline time, cross-node build time, cross-node estimation
time, and effect consistency.

---

## Key experimental results

### Per-node causal effect

Under load (15 workers, ~38 QPS per node):

| Node   | Causal Effect (ms per active query) | Placebo Effect (ms) |
|--------|--------------------------------------|---------------------|
| node-1 | 28.855                               | 0.186               |
| node-2 | 29.001                               | 0.930               |
| node-3 | 27.973                               | 0.021               |

The placebo refutation confirms the effect is causal. When the treatment
variable is randomly permuted, the effect collapses to near-zero. The
original effect of ~28ms represents a genuine causal relationship between
concurrent active queries and query latency.

### Cross-node causal effects

| Causal Path                                              | Effect  |
|----------------------------------------------------------|---------|
| node-1 active_queries -> node-2 active_queries           | ~1.000  |
| node-1 active_queries -> node-3 active_queries           | ~0.996  |
| node-2 active_queries -> node-3 active_queries           | ~0.498  |
| node-1 avg_query_duration_ms -> node-2 avg_query_duration_ms | ~1.001 |
| node-1 avg_query_duration_ms -> node-3 avg_query_duration_ms | ~1.006 |
| node-1 active_queries -> node-2 avg_query_duration_ms   | ~27-28  |
| node-1 active_queries -> node-3 avg_query_duration_ms   | ~26-29  |

The near-unity cross-node effects on active_queries confirm that the load
generator distributes work equally across nodes — a correct finding given
the experimental design.

### Performance benchmarks

| Metric                    | Mean      | p95 / max   |
|---------------------------|-----------|-------------|
| model_build               | 1.019ms   | 1.608ms     |
| identify_effect           | 3.717ms   | 7.112ms     |
| estimate_effect           | 81.322ms  | 139.829ms   |
| full_3node_pipeline       | 169.489ms | 186.067ms   |
| cross_node_graph_build    | 0.309ms   | 0.417ms     |
| cross_node_estimation     | 1357ms    | 1671ms      |
| effect_consistency stddev | 0.000000  | deterministic |

The causal graph builds in under 0.5ms. A full 3-node pipeline — build,
identify, estimate — completes in under 200ms. The effect estimate is
perfectly deterministic given fixed data (stddev=0).

---

## Key technical decisions

**Why networkx DiGraph instead of DOT string**  
DoWhy supports both DOT strings and networkx graphs as causal graph
specifications. On Windows with Python 3.13, pygraphviz has no prebuilt
wheel and pydot fails to parse variable names containing hyphens. Using
networkx directly avoids both issues and gives programmatic control over
graph construction.

**Why active_queries as treatment variable**  
The original treatment variable buffers_backend showed zero variance under
idle conditions and near-zero causal effect even under load because buffer
writes are not directly controlled by the load generator. Active queries
vary directly with the number of concurrent workers and show clear causal
signal with mean effect of 28ms per additional concurrent query.

**Why backdoor linear regression**  
The backdoor criterion is satisfied in our causal graph — the adjustment
set blocks all non-causal paths from treatment to outcome. Linear
regression is the appropriate estimator when the treatment-outcome
relationship is approximately linear, which it is here (latency scales
linearly with concurrent query count under the load conditions tested).

**Why placebo refutation**  
The placebo treatment refuter permutes the treatment variable randomly and
re-estimates the effect. If the estimated effect drops to near-zero under
random permutation, the original effect is unlikely to be spurious. All
three nodes pass this test with original effects of 27-29ms and placebo
effects of 0.02-0.93ms.

**Why 60 samples at 2-second intervals**  
60 samples gives sufficient statistical power for linear regression while
keeping collection time to 2 minutes. The 2-second interval allows the
PostgreSQL telemetry to capture meaningful variation in active query counts
between samples.

---

## Research questions opened by Sprint 2

1. The cross-node causal graph currently assumes directed edges based on
   domain knowledge. Can structure learning algorithms (PC algorithm,
   FCI algorithm) discover the causal graph from data without prior
   knowledge?

2. The effect estimate of ~28ms per active query is measured under
   controlled load. How does this change under heterogeneous workloads
   with mixed read-write ratios?

3. Can the causal model detect Byzantine node behavior — a node that is
   intentionally reporting false telemetry — by observing violations of
   expected causal relationships?

4. The current model is retrained from scratch on each call to /retrain.
   Can online causal learning update the model incrementally as new
   telemetry arrives without full retraining?

---

## What was not built in Sprint 2

- Online/incremental causal model updating
- Causal structure learning from data
- Integration with ZK proof layer (Sprint 5)
- Temporal prediction using causal model (Pillar 5)

---

## Next sprint

Sprint 3 integrates the telemetry pipeline (Pillar 1 Sprint 1) with the
causal engine (Pillar 1 Sprint 2) into a unified real-time system. The
causal model will be updated continuously as new telemetry arrives, and
the /why endpoint will serve live causal answers rather than answers based
on a training snapshot.