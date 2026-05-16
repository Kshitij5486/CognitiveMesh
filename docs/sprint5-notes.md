# Sprint 5 Research Notes
## CognitiveMesh — Pillar 3: Byzantine Consensus Engine

**Sprint duration:** 7 days (Days 29-35)  
**Completed:** May 2026  
**Author:** Kshitij Srivastava, NIT Surat

---

## What was built

Sprint 5 adds a Byzantine fault detection and consensus layer to
CognitiveMesh. The system now detects nodes whose reported telemetry
is causally inconsistent with the cluster's collective understanding,
penalizes their reputation over time, excludes them from weighted
consensus decisions, and ultimately isolates them from the cluster.
This closes the loop between causal self-awareness (Sprint 2-3),
cryptographic proof (Sprint 4), and self-defense (Sprint 5).

### Components

**ByzantineDetector** (`byzantine_detector.py`)
Monitors causal effect estimates across all three nodes every 15
seconds. Computes a cluster baseline (median of node effects) and
flags nodes whose deviation exceeds 30% (WARNING) or 60% (CRITICAL).
Generates a fresh ZK proof for each node at every detection cycle.
Maintains a NodeByzantineProfile per node tracking proof successes,
proof failures, causal violations, and status transitions.

**ReputationScorer** (`reputation_scorer.py`)
Tracks node trustworthiness over time using a score in [0.0, 1.0].
Rewards honest cycles (+0.02), proof successes (+0.05), and penalizes
causal violations (-0.20 scaled by deviation magnitude) and proof
failures (-0.15). Applies time decay every 60 seconds to prevent
scores from remaining artificially high. Syncs from ByzantineDetector
every 15 seconds.

**ConsensusEngine** (`consensus_engine.py`)
Makes cluster-wide decisions weighted by node reputation scores.
Nodes scoring below 0.15 are excluded entirely. Each participating
node votes with its current causal effect estimate, weighted by its
reputation score. The decided value is the reputation-weighted average
of all participating votes. Also supports threshold consensus for
alert level decisions.

**IsolationMechanism** (`isolation_mechanism.py`)
Quarantines nodes whose reputation drops below 0.15. Enforces a
minimum of 2 healthy nodes before isolating any node to prevent
cluster collapse. Tracks isolation records with timestamps and
durations. Supports manual operator override for both isolation and
release. Automatically releases nodes when their reputation recovers
above 0.40.

**ByzantineAPI** (`byzantine_api.py`)
FastAPI service on port 8085 integrating all five components.
Endpoints: /health, /cluster, /nodes, /nodes/{id}, /consensus/causal,
/consensus/threshold, /consensus/history, /isolation/isolate,
/isolation/release, /isolation/status, /reputation, /reputation/{id},
/detection, /detection/{id}, /pipeline/summary.

**Byzantine Test Suite** (`tests/test_byzantine_layer.py`)
46 tests covering NodeByzantineProfile state transitions, ByzantineEvidence
serialization, NodeReputation score mechanics, ConsensusEngine weighted
averaging and Byzantine exclusion, IsolationMechanism lifecycle, and
latency benchmarks. Combined with prior suites: 132 total tests, 0
failures.

---

## Key experimental results

### Byzantine detection results

| Run    | Anomalous node | Deviation   | Detection cycle | Action          |
|--------|----------------|-------------|-----------------|-----------------|
| Day 29 | node-3         | -98.0%      | cycle=1         | CRITICAL flagged|
| Day 30 | node-3         | -98.0%      | cycle=1         | score 0.50→0.05 |
| Day 31 | node-1         | -77.9%      | cycle=1         | score 0.50→0.34 |
| Day 33 | node-1         | -98.9%      | cycle=1         | ISOLATED        |

All anomalies were genuine statistical instability in the first retrain
cycle with only 30 samples. By retrain cycle 2-3, node effects
stabilized to the 27-32ms cluster range. The detector correctly
identified early-cycle instability as a deviation from the cluster
baseline.

### Reputation scoring dynamics

| Event                    | Score delta | Resulting score |
|--------------------------|-------------|-----------------|
| Initial                  | —           | 0.50            |
| CRITICAL violation (×2)  | -0.196 ×2   | 0.05 (floor)    |
| Honest cycle             | +0.02       | 0.57 (Day 29)   |
| Proof success            | +0.05       | 0.64 → 0.85     |
| Time decay               | -0.005      | gradual decline |

### Consensus results (Day 33)
Proposal: consensus-causal-0001-182211
node-2 voted 31.5351ms  weight=0.71
node-3 voted 28.9895ms  weight=0.71
decided=30.2623ms   total_weight=1.42
excluded=['node-1']   consensus_reached=true

Without reputation weighting, naive average would be
(31.54 + 28.99) / 2 = 30.27ms. With equal weights the result is
identical here since both honest nodes had the same score. The key
test (Day 31) showed node-1 with weight=0.34 vs 0.50 for honest
nodes, pulling the decided value from naive 20ms toward the honest
cluster value of 27ms.

### Full test suite

| Test file              | Tests | Passed | Failed | Runtime  |
|------------------------|-------|--------|--------|----------|
| test_graph_builder     | 27    | 27     | 0      | —        |
| test_integration       | 29    | 29     | 0      | —        |
| test_zk_layer          | 30    | 30     | 0      | —        |
| test_byzantine_layer   | 46    | 46     | 0      | 3.84s    |
| Combined               | 132   | 132    | 0      | 16.79s   |

### Latency benchmarks

| Operation                  | Mean latency | Threshold |
|----------------------------|--------------|-----------|
| Reputation update          | <1ms         | <1ms ✓   |
| Consensus decision         | <10ms        | <10ms ✓  |
| Isolation check            | <1ms         | <1ms ✓   |
| Profile update (200 concurrent) | thread-safe | —    |

---

## The research contribution

Every existing Byzantine fault tolerance system detects Byzantine
behavior by message consistency — does node A agree with node B on
the same value? CognitiveMesh detects Byzantine behavior causally —
does node A's claimed performance match what the causal model predicts
given its observed inputs?

This is a fundamentally different and stronger detection mechanism
for three reasons:

**1. It detects subtle Byzantine behavior.**
A node reporting plausible-sounding but causally inconsistent metrics
would fool message-consistency checks (since the values are in a
normal range) but not causal consistency checks (since the causal
model predicts a different relationship between queries and latency).

**2. It is cryptographically grounded.**
Every causal claim is backed by a STARK zero-knowledge proof. A
Byzantine node cannot forge a proof for an inconsistent causal
claim without violating the MAC constraint in the Rust field arithmetic.

**3. It is self-updating.**
Because the causal model retrains every 30 seconds on live telemetry,
the baseline against which Byzantine behavior is measured adapts
continuously to changing cluster conditions.

---

## Key technical decisions

**Why median rather than mean for cluster baseline**
The median is robust to a single outlier in a 3-node cluster. If
one node produces an anomalous effect (e.g. 0.5ms vs cluster 28ms),
the mean baseline would be pulled toward the outlier while the median
remains at the honest cluster value. This ensures Byzantine nodes
cannot shift the baseline to make their anomalous values appear normal.

**Why 0.15 isolation threshold and 0.40 release threshold**
The asymmetric thresholds (isolate at 0.15, release at 0.40) create
a hysteresis band that prevents oscillation. A node that briefly
recovers to 0.16 will not be released and immediately re-isolated.
The node must sustain recovery to 0.40 before returning to the cluster.

**Why minimum 2 healthy nodes before isolation**
With 3 nodes, isolating 2 would leave the cluster unable to reach
consensus. The 2-node minimum ensures the cluster remains operational
even when a Byzantine node is quarantined.

**Why reputation-weighted consensus rather than simple majority**
Simple majority voting gives equal weight to all nodes regardless of
their history. A node that has been consistently anomalous but not
yet below the isolation threshold would have the same vote weight as
a node with a perfect record. Reputation weighting continuously
reduces the influence of nodes that have shown signs of Byzantine
behavior, before they reach the isolation threshold.

---

## Research questions opened by Sprint 5

1. The detector uses fixed thresholds (30%/60%). Can adaptive
   thresholds learned from historical variance reduce false positives
   during periods of genuine workload volatility?

2. The reputation decay rate (0.005 per minute) is fixed. Can a
   Bayesian reputation model with prior beliefs and likelihood updates
   produce faster recovery for genuinely honest nodes while maintaining
   slow recovery for nodes with a history of violations?

3. The consensus engine uses reputation-weighted averaging. Can a
   more sophisticated voting mechanism — such as a Byzantine agreement
   protocol (PBFT, HotStuff) layered on top of reputation filtering —
   provide formal Byzantine fault tolerance guarantees?

4. The isolation mechanism currently operates on a per-node basis.
   Can coordinated Byzantine behavior — where two nodes collude to
   produce mutually consistent but collectively incorrect causal
   claims — be detected by the current architecture?

---

## What was not built in Sprint 5

- Adaptive detection thresholds
- Bayesian reputation model
- Formal Byzantine agreement protocol layered on reputation
- Collusion detection between multiple Byzantine nodes
- Persistent reputation state across restarts
- REST endpoint for manual threshold adjustment

---

## Sprint 6 preview

Sprint 6 begins Pillar 4: Predictive Intelligence. The system will
learn to anticipate performance degradation before it happens, using
the causal model to simulate future cluster states under different
load scenarios. A node approaching Byzantine behavior will trigger
a predictive alert before its causal effect crosses the detection
threshold.

This transforms CognitiveMesh from a system that reacts to Byzantine
behavior into one that predicts it.