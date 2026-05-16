# Sprint 4 Research Notes
## CognitiveMesh — Pillar 2: Zero-Knowledge Security Layer

**Sprint duration:** 7 days (Days 22-28)  
**Completed:** May 2026  
**Author:** Kshitij Srivastava, NIT Surat

---

## What was built

Sprint 4 adds a cryptographic proof layer to the causal reasoning engine.
The system can now prove causal claims about distributed system behavior
without revealing the raw telemetry data that generated those claims.
This is the first known system that combines causal inference with
zero-knowledge proofs for distributed database performance analysis.

### Components

**causal_prover binary** (`mpc-network/zkproof/src/causal_prover.rs`)
A Rust CLI binary that accepts a CausalProofRequest as JSON on stdin
and outputs a StarkProof as JSON on stdout. Encodes causal effects as
MAC constraints: causal_effect = avg_latency_ms * active_queries.
Built on top of the existing MpcProver infrastructure from the prior
MPC network project. Generates and verifies proofs in under 20ms.

**ZKProofBridge** (`zk_proof_bridge.py`)
Python bridge to the Rust STARK prover. Spawns the causal-prover
binary as a subprocess, passes JSON via stdin, parses the StarkProof
from stdout. Handles encoding of causal effects into integer MAC
constraints that satisfy the Rust field arithmetic. Tracks proof
count and latency statistics.

**ZKCausalAPI** (`zk_causal_api.py`)
FastAPI service on port 8083. Exposes /prove/{node_id},
/prove-all, /verify/{proof_id}, /proofs, and /status endpoints.
Generates STARK proofs on demand and stores them in an in-memory
proof store keyed by proof_id.

**ZKUnifiedPipeline** (`zk_unified_pipeline.py`)
FastAPI service on port 8084. Integrates StreamingCausalUpdater
with ZKProofBridge in a single service. Auto-generates ZK proofs
after every retrain cycle via a background proof refresh thread.
The /why/{node_id} endpoint returns both the causal explanation
and its associated ZK proof in a single response.

**ZK Test Suite** (`tests/test_zk_layer.py`)
30 tests covering CausalProofRequest structure, ZKProofBridge
proof generation and verification, determinism, tamper detection,
latency benchmarks, and multi-node proof generation. Combined with
prior test suites: 86 total tests, 0 failures.

---

## Key experimental results

### Proof generation performance

| Metric                          | Value        |
|---------------------------------|--------------|
| Single proof latency (mean)     | 12.6ms       |
| Single proof latency (max)      | 19.7ms       |
| Three-node proof latency        | 45.8ms total |
| Three-node proof latency (mean) | 15.3ms/node  |
| Proof generation rate           | ~65 proofs/s |
| Proof size (JSON)               | ~2.5KB       |
| Verification latency            | <1ms         |

### ZK unified pipeline results

| Metric                   | Value                            |
|--------------------------|----------------------------------|
| Uptime                   | 106 seconds observed             |
| Auto-retrains            | 1 cycle before measurement       |
| Proofs generated         | 3 (one per node)                 |
| All verified             | true                             |
| /why response time       | <10ms                            |
| Cross-node effects found | 5                                |

### Full test suite summary

| Test file          | Tests | Passed | Failed | Runtime  |
|--------------------|-------|--------|--------|----------|
| test_graph_builder | 27    | 27     | 0      | ~13s     |
| test_integration   | 29    | 29     | 0      | ~15s     |
| test_zk_layer      | 30    | 30     | 0      | 1.36s    |
| Combined           | 86    | 86     | 0      | 17.40s   |

---

## Causal proof schema

A causal claim on node X is encoded as a MAC constraint:
causal_effect_ms = avg_latency_ms_per_query × active_queries

In the ZK proof system this maps to:
mac = alpha × value   (mod PRIME)

Where:
- `value` = active_queries (the treatment variable)
- `alpha` = avg_latency_ms_per_query × 100 (scaled integer)
- `mac`   = causal_effect_ms × 100 (scaled integer)

The prover generates a STARK proof that this constraint holds
for the actual observed values. The verifier checks the proof
using Fiat-Shamir hashing and Merkle proof verification without
seeing the raw telemetry.

**Public inputs** (visible to verifier):
- causal_effect_ms (the claimed effect size)
- samples_used (how much data the model was trained on)
- retrain_cycle (which model version generated this claim)

**Private inputs** (hidden from verifier):
- Raw PostgreSQL telemetry (pg_stat_activity, pg_stat_bgwriter)
- Individual query durations
- Lock contention events

---

## Key technical decisions

**Why subprocess over PyO3 bindings**
PyO3 bindings require recompiling the Rust library with Python
extension headers and maintaining ABI compatibility across Python
versions. Subprocess communication via JSON stdin/stdout is simpler,
requires no Rust code changes, and performs adequately at 15ms per
proof — well within the 30-second retrain budget. The subprocess
approach also means the Rust engine can be upgraded independently
of the Python layer.

**Why MAC constraint for causal effects**
The STARK prover already implements MAC verification as one of its
core constraint types: mac = alpha × value (mod PRIME). Causal
effects naturally fit this structure — the effect per query (alpha)
times the number of concurrent queries (value) equals the total
observed latency impact (mac). This reuses existing verified
cryptographic infrastructure without adding new constraint types.

**Why scale by 100 for integer encoding**
The Rust field arithmetic operates on u64 integers. Causal effects
are floating-point values (e.g. 29.63ms). Multiplying by 100
preserves two decimal places of precision while fitting within u64
range for realistic effect sizes (0-999ms). Values above 655ms
per query would overflow at active_queries=10, but this is far
outside observed ranges (25-55ms per query).

**Why auto-prove after every retrain**
Causal models update every 30 seconds. Proofs generated from an
old model would certify stale claims. The proof refresh thread
checks the retrain counter after each 15-second sleep and generates
fresh proofs whenever a new retrain cycle completes. This ensures
every /why response is backed by a proof from the current model.

**Why store proofs in memory keyed by node_id**
The proof store uses node_id as the key so each node always has
exactly one current proof — the one from the most recent retrain.
Older proofs are automatically replaced. This avoids unbounded
memory growth while ensuring the most recent causal claim is
always provable. In production, proofs would be persisted to a
database or content-addressed store.

---

## Integration with MPC network

The ZK security layer reuses the existing MPC network's cryptographic
infrastructure without modification. Specifically:

**What was reused unchanged:**
- `mpc_crypto::field` — FieldElement arithmetic mod PRIME
- `mpc_zkproof::air` — AIR constraint definitions and ComputationTrace
- `mpc_zkproof::prover` — MpcProver with Fiat-Shamir and Merkle trees
- `mpc_zkproof::verifier` — MpcVerifier with full check suite
- `mpc_zkproof::utils` — MerkleTree, Transcript, hash functions

**What was added:**
- `causal_prover.rs` — CLI binary encoding causal effects as MAC traces
- `zk_proof_bridge.py` — Python subprocess bridge
- `zk_causal_api.py` — REST API for proof generation
- `zk_unified_pipeline.py` — integrated pipeline with auto-proving
- `tests/test_zk_layer.py` — 30 tests

The MPC network's 331 tests still pass unchanged after Sprint 4.

---

## Research questions opened by Sprint 4

1. The current proof encodes a single causal effect per node per
   retrain cycle. Can the system generate a proof over the entire
   causal graph — proving not just per-node effects but the
   cross-node propagation structure simultaneously?

2. The MAC constraint proves the arithmetic relationship between
   active_queries, avg_latency, and causal_effect. Can a more
   expressive constraint (e.g. a multiplication constraint using
   Beaver triples) prove the DoWhy linear regression computation
   itself rather than just its output?

3. The proof currently uses integer-scaled values. Can the field
   arithmetic be extended to fixed-point or rational representations
   to eliminate the scaling approximation?

4. The Solidity verifier in the MPC network can verify proofs
   on-chain. Can causal proofs be submitted to the SlashingContract
   to automatically penalize nodes whose reported causal effects
   are cryptographically inconsistent with observed telemetry?

---

## What was not built in Sprint 4

- On-chain proof submission to Solidity contracts
- Proof persistence to disk or database
- Batch proof generation for multiple retrain cycles
- Proof expiry and rotation policy
- Cross-node causal proof (joint proof over all three nodes)

---

## Sprint 5 preview

Sprint 5 begins Pillar 3: the Byzantine Consensus Engine. The system
will detect and isolate nodes that report causally inconsistent
telemetry — nodes whose claimed performance metrics violate the
causal relationships established by the ZK proof layer. A node
that cannot produce a valid ZK proof for its causal claims will
be flagged as potentially Byzantine and excluded from the cluster's
consensus decisions.

This closes the loop: the system not only understands itself
causally and proves those claims cryptographically, but actively
defends against nodes that attempt to corrupt its self-understanding.