# Phase 1: Agentic Workload Generation

## Summary

BLIS can simulate language, multimodal, and reasoning workloads — but all of these model independent requests or sequential multi-turn sessions. Real agentic inference workloads have fundamentally different structure: parent-child dependencies between LLM calls, tool call latency (external service delays), fan-out (parallel candidates), fan-in (aggregation), and iteration loops.

This issue proposes extending `WorkloadSpec` v2 with an `AgenticSpec` that defines workflow DAGs using three composable primitives — `depends_on`, `fan_out`, and `loop` — to express all major agentic patterns:

- **ReAct agents** — sequential loop: reason → tool → observe → repeat
- **MCTS / Tree-of-Thought** — tree: fan-out N candidates → evaluate → expand best
- **Parallel tool calls (fork-join)** — plan → [tool₁ ∥ tool₂ ∥ tool₃] → synthesize
- **Multi-step chains** — linear DAG with heterogeneous tool latencies

## Analysis Questions

This design enables BLIS to answer:

1. **Q1:** How does agentic loop iteration count (ReAct depth) affect cluster capacity requirements compared to equivalent independent-request workloads?
2. **Q2:** What is the impact of tool latency variance on session end-to-end latency distribution, and which tools are on the critical path?
3. **Q3:** Can routing policies optimize for fan-out parallelism in MCTS/ToT patterns, or does fan-out amplification overwhelm single-instance capacity?
4. **Q4:** What is the session-level tail latency (p95, p99) for mixed agentic + non-agentic workloads under different admission control policies?
5. **Q5:** How does context accumulation across loop iterations affect KV cache pressure and preemption rates?

## Motivation

Agentic workloads are becoming a dominant inference pattern. They differ from standard workloads in ways that matter for capacity planning:

1. **Request dependencies** — a child LLM call cannot start until its parent completes and any intermediate tool call finishes. This creates serial chains that amplify tail latency.
2. **Tool call latency** — external service calls (web search, code execution, DB queries) introduce non-LLM delays that consume wall-clock time but not GPU resources.
3. **Fan-out amplification** — MCTS/ToT patterns multiply request count per session by the branch factor. A 4-wide tree with 3 levels produces 21 LLM calls per session.
4. **Variable session length** — ReAct loops have unpredictable iteration counts, making capacity planning harder than fixed-length workloads.

Without agentic workload support, BLIS cannot model these patterns for capacity planning, policy optimization, or performance prediction.

## Modeling Decisions

| Component | Decision Status | Status | Real-System Behavior | Notes |
|-----------|----------------|--------|---------------------|-------|
| DAG structure (step dependencies) | Proposed | **Modeled** | Agents execute steps in dependency order | Exact acyclic DAG with fan-out, fan-in, and loops |
| Tool call latency (external delays) | Proposed | **Modeled** | External services have variable latency | Sampled from configurable distributions per tool type |
| Fan-out/fan-in (parallel candidates) | Proposed | **Modeled** | MCTS/ToT spawn parallel candidates, aggregate results | N parallel instances with fan-in synchronization |
| Loop iteration with max bound | Proposed | **Modeled** | ReAct loops repeat until satisfied or max rounds | Fixed max_iterations; horizon also terminates |
| Context accumulation across iterations | Proposed | **Modeled** | Agent context grows as tool outputs accumulate | Input token count grows by prior iterations' output tokens |
| GPU idle during tool execution | Proposed | **Simplified** | GPU may serve other requests during async tool calls | Tool calls are modeled as pure wall-clock delays. GPU is released — the parent request completes normally (leaves the running batch, frees KV blocks), and the child request arrives as a new independent request after tool latency elapses. In cluster mode, the child may route to any instance (not necessarily the parent's instance) — tool latency is a wall-clock delay between parent completion and child arrival, not tied to any specific instance's GPU. |
| Tool output conditioning on input | Proposed | **Simplified** | Tool output size may depend on query complexity | Output tokens sampled independently from tool's output_tokens distribution. Lost: correlation between query complexity and result size. Acceptable: first-order approximation; conditioning would require learned models from real traces. |
| Adaptive loop exit (LLM decides when to stop) | Deferred | **Omitted** | Agents decide when they have enough information | Loop runs to max_iterations or horizon. Lost: variable iteration counts driven by LLM confidence. Acceptable: capacity planning needs worst-case bounds; adaptive exit is a future phase with probabilistic edges. |
| Dynamic tool selection (LLM chooses tool at runtime) | Deferred | **Omitted** | Agents select tools based on reasoning output | Tool at each step is fixed in the workflow DAG. Lost: runtime-dependent tool selection. Acceptable: workflow DAGs capture the structural pattern; dynamic selection is a future phase. |
| Tool failures and retries | Deferred | **Omitted** | Tools may fail and require retry loops | All tool calls succeed. Lost: retry amplification and error handling overhead. Acceptable: retry semantics would require conditional edges (future phase). |
| Tool output tokenization overhead | Deferred | **Omitted** | Real systems re-tokenize tool outputs | Tokenization is assumed instant. Lost: tokenization latency (~1-5ms). Acceptable: negligible compared to tool latency (50-500ms). |

### Real-System Correspondence

| BLIS Feature | vLLM | SGLang | llm-d | LangChain/AutoGen |
|-------------|------|--------|-------|-------------------|
| DAG dependency tracking | Not native (client-side orchestration) | Not native | Gateway-level orchestration | Native DAG execution |
| Tool call latency | Async via external tools; GPU released | Similar to vLLM | Router-managed tool dispatch | Framework-managed async |
| Fan-out (parallel candidates) | Client spawns parallel requests | Client-side batching | Load-balanced across instances | Native parallel tool calls |
| Fan-in (aggregation) | Client-side join | Client-side join | Gateway-side join | Native gather/reduce |
| GPU release during tool calls | GPU freed after response; new request needed | Same | Same | N/A (no GPU management) |
| Context accumulation | KV cache retained if same session | Radix attention for shared prefix | Prefix-aware routing | Framework manages context window |

### Simplest Sufficient Version Analysis

We evaluated three design alternatives:

- **Option A (linear chains only):** Covers ReAct and multi-step chains but not MCTS/ToT. Eliminates fan-in tracking (~100 LOC savings). **Rejected:** MCTS is a major target use case (Q3); fork-join patterns (Q2) are common in real agents.
- **Option B (linear + fan-out, no fan-in):** Each parallel branch completes independently. Eliminates fan-in synchronization. **Rejected:** Fork-join patterns (plan → parallel tools → synthesize) require fan-in for the synthesis step.
- **Option C (full DAG with fan-out/fan-in/loops):** Selected. Supports all four patterns. Fan-in tracking adds ~100 LOC but is essential for expressing real agent workflows.

### Sensitivity Analysis

| Component | If Omitted | Impact on Q1-Q5 | Justification for Inclusion |
|-----------|-----------|-----------------|----------------------------|
| Fan-in synchronization | Each fan-out branch completes independently; no aggregation step | Q3 unanswerable (MCTS evaluate step requires fan-in); Q2 fork-join critical path undefined | Essential: ~100 LOC cost justified by Q2+Q3 coverage |
| Tool latency variance | Use constant tool latency (mean only) | Q2 partially unanswerable (tool variance → zero eliminates critical path analysis); Q1, Q4, Q5 unaffected | Essential for Q2; variance drives tail latency distribution |
| Context accumulation | Clamp input tokens to base distribution max | Q5 unanswerable (no KV pressure growth); Q1-Q4 <5% change (latency growth is modest for typical iteration counts) | Essential for Q5; low implementation cost (accumulator in context_growth loop) |
| Loop iteration tracking | Treat loop body as single-shot (no iteration) | Q1 unanswerable (cannot compare iteration depths); Q5 unanswerable | Essential for Q1+Q5; core to ReAct pattern |
| Fan-out expansion | No parallel candidates; sequential-only DAGs | Q3 unanswerable; Q1, Q2, Q4, Q5 unaffected | Essential for Q3; enables MCTS/ToT patterns |

**Summary:** Option A (linear only) answers Q1, Q4, Q5 but not Q2 (fork-join critical path) or Q3 (fan-out optimization). Option C adds Q2+Q3 for ~200 additional LOC (~30% increase over Option A's ~600 LOC). All five analysis questions require Option C.

## Two-Layer Arrival Time Model

Agentic workloads require two layers of timing:

**Layer 1 — Session arrivals (pre-generated).** When does each new agentic session start? Controlled by existing `ArrivalSpec` on `ClientSpec` (poisson, gamma, weibull, constant). Only root steps are pre-generated. Root steps enter the standard admission → routing pipeline (same as non-agentic requests), ensuring agentic sessions participate fully in cluster-level load balancing.

**Layer 2 — Intra-session step arrivals (deferred injection).** When does each step within a workflow execute? Governed by DAG dependencies at simulation runtime:

| Edge type | Child arrival time | Classification |
|---|---|---|
| LLM → LLM child | Parent completion time (immediate) | Endogenous |
| LLM → tool_call child | Parent completion time + sampled tool latency | Endogenous (randomness from PartitionedRNG) |
| tool_call → LLM child | Tool completion time (immediate after tool) | Endogenous |
| Fan-in (multiple parents) | max(completion times of all parents) | Endogenous (deterministic) |
| Loop iteration boundary | Last loop step's completion time | Endogenous |

All Layer 2 events are **endogenous** — driven by simulator state transitions (request completions), not external workload arrivals. Child requests are injected as standard `ArrivalEvent`s with the same priority constant as Layer 1 arrivals, ensuring consistent `(timestamp, priority, seqID)` tie-breaking. Determinism is preserved because all timing derives from seeded PartitionedRNG and deterministic completion times.

Deferred injection means intra-session step timing depends on actual completion times in the simulator. A new **SessionRegistry** component tracks active sessions and injects child requests when parent steps complete.

```
ReAct session (3 iterations):

Session arrival: t=1000

Iter 1:  reason[t=1000]──complete[t=1200]──►tool[lat=500]──►observe[t=1700]──complete[t=1850]
Iter 2:  reason[t=1850]──complete[t=2100]──►tool[lat=300]──►observe[t=2400]──complete[t=2600]
Iter 3:  reason[t=2600]──complete[t=2900]──►tool[lat=450]──►observe[t=3350]──complete[t=3500]
         final-answer[t=3500]──complete[t=3800]

(Illustrative: actual times depend on configured distributions and RNG state)
```

## Schema: AgenticSpec

New optional field on `ClientSpec`, following the same extension pattern as `multimodal` and `reasoning`:

```yaml
version: "2"
seed: 42
category: "agentic"
aggregate_rate: 10.0
horizon: 60000000

clients:
  - id: "react-agent"
    tenant_id: "team-search"
    slo_class: "standard"
    rate_fraction: 0.5
    arrival:
      process: poisson              # Layer 1: session inter-arrival
    agentic:
      workflow: "react-search"
      loop:
        over: ["reason", "act", "observe"]
        max_iterations: 5
      steps:
        - id: "reason"
          type: llm_call
          input_distribution: {type: gaussian, params: {mean: 256, std_dev: 50, min: 32, max: 1024}}
          output_distribution: {type: exponential, params: {mean: 128}}
        - id: "act"
          type: tool_call
          tool: "web_search"
          depends_on: ["reason"]
        - id: "observe"
          type: llm_call
          depends_on: ["act"]
          context_growth: accumulate
          input_distribution: {type: gaussian, params: {mean: 200, std_dev: 50, min: 32, max: 2048}}
          output_distribution: {type: exponential, params: {mean: 64}}
        - id: "final-answer"
          type: llm_call
          depends_on: ["observe"]
          input_distribution: {type: gaussian, params: {mean: 512, std_dev: 100, min: 64, max: 2048}}
          output_distribution: {type: exponential, params: {mean: 256}}
      tools:
        web_search:
          latency: {type: exponential, params: {mean: 50000}}
          output_tokens: {type: gaussian, params: {mean: 200, std_dev: 50, min: 10, max: 1000}}
```

### Structural primitives

- **Sequential:** linear `depends_on` chain (A → B → C)
- **Fan-out:** `fan_out: N` spawns N parallel instances (each gets a unique index, e.g., `generate[0]`..`generate[3]`), or multiple steps sharing the same `depends_on`
- **Fan-in:** one step with `depends_on: [A, B, C]` — waits for all parents to **complete** (not just arrive)
- **Loop:** `loop.over` names repeating steps; steps outside the loop run after loop exits

### Context Growth Semantics

When `context_growth: accumulate` is set on a loop step, the input token count for iteration N is:

```
effective_input[N] = sample(input_distribution) + sum(output_tokens[i] for i in 0..N-1)
```

If the effective input exceeds `input_distribution.params.max`, it is clamped to that maximum. This models the agent's growing context window as tool outputs accumulate across iterations. The accumulated tokens contribute to KV cache allocation and latency model calculations.

### More examples

<details>
<summary>MCTS / Tree-of-Thought</summary>

```yaml
agentic:
  workflow: "mcts-code"
  steps:
    - id: "decompose"
      type: llm_call
      input_distribution: {type: gaussian, params: {mean: 500, std_dev: 100, min: 64, max: 2048}}
      output_distribution: {type: exponential, params: {mean: 200}}
    - id: "generate"
      type: llm_call
      depends_on: ["decompose"]
      fan_out: 4
      input_distribution: {type: gaussian, params: {mean: 300, std_dev: 80, min: 32, max: 1024}}
      output_distribution: {type: exponential, params: {mean: 512}}
    - id: "evaluate"
      type: llm_call
      depends_on: ["generate"]
      input_distribution: {type: gaussian, params: {mean: 2048, std_dev: 200, min: 256, max: 4096}}
      output_distribution: {type: exponential, params: {mean: 64}}
    - id: "verify"
      type: tool_call
      tool: "code_exec"
      depends_on: ["evaluate"]
    - id: "refine"
      type: llm_call
      depends_on: ["verify"]
      input_distribution: {type: gaussian, params: {mean: 1024, std_dev: 150, min: 128, max: 4096}}
      output_distribution: {type: exponential, params: {mean: 256}}
  tools:
    code_exec:
      latency: {type: gaussian, params: {mean: 100000, std_dev: 20000, min: 10000, max: 500000}}
      output_tokens: {type: constant, params: {value: 150}}
```

</details>

<details>
<summary>Parallel tool calls (fork-join)</summary>

```yaml
agentic:
  workflow: "fork-join-research"
  steps:
    - id: "plan"
      type: llm_call
      input_distribution: {type: gaussian, params: {mean: 300, std_dev: 50, min: 32, max: 1024}}
      output_distribution: {type: exponential, params: {mean: 100}}
    - id: "search-web"
      type: tool_call
      tool: "web_search"
      depends_on: ["plan"]
    - id: "query-db"
      type: tool_call
      tool: "db_query"
      depends_on: ["plan"]
    - id: "fetch-docs"
      type: tool_call
      tool: "doc_retrieval"
      depends_on: ["plan"]
    - id: "synthesize"
      type: llm_call
      depends_on: ["search-web", "query-db", "fetch-docs"]
      input_distribution: {type: gaussian, params: {mean: 1500, std_dev: 200, min: 256, max: 4096}}
      output_distribution: {type: exponential, params: {mean: 512}}
  tools:
    web_search:
      latency: {type: exponential, params: {mean: 80000}}
      output_tokens: {type: gaussian, params: {mean: 300, std_dev: 80, min: 20, max: 1000}}
    db_query:
      latency: {type: gaussian, params: {mean: 20000, std_dev: 5000, min: 1000, max: 100000}}
      output_tokens: {type: constant, params: {value: 100}}
    doc_retrieval:
      latency: {type: exponential, params: {mean: 30000}}
      output_tokens: {type: gaussian, params: {mean: 500, std_dev: 100, min: 50, max: 2000}}
```

</details>

## SessionRegistry Module Contract

**Extension type:** Subsystem Module (design guidelines Section 5.3)

| Aspect | Description |
|--------|-------------|
| **Observes** | Request completion notifications (session ID, step ID, completion timestamp). Workflow DAG (loaded at session creation, frozen thereafter). Does NOT observe global clock, KV state, or routing state. |
| **Controls** | (1) When child requests are injected into the event queue (deferred injection timing). (2) Fan-in synchronization — whether a fan-in step's dependencies are all satisfied. (3) Loop iteration advancement and termination. |
| **Owns** | Per-session mutable state: workflow DAG reference, completion timestamps per step+iteration, fan-in dependency counters, current loop iteration, session lifecycle state (active/completed). Also owns a global injection counter for INV-1 verification. |
| **Invariants** | INV-A1 (DAG causality), INV-A2 (fan-in completeness), INV-A3 (loop termination), INV-A4 (session completeness), INV-A5 (tool latency non-negative). See Invariants section. |
| **Events** | **Consumes:** Completion notification (triggered when a request with agentic metadata completes). **Produces:** Child requests returned to caller for injection as standard `ArrivalEvent`s. No new event types — reuses existing event infrastructure. Priority: same as standard ArrivalEvents. |
| **Extension friction** | Adding a second SessionRegistry implementation (e.g., persistent/distributed): ~4 files (extract interface + new implementation + factory + CLI flag). Adding a new workflow pattern (e.g., conditional edges): ~2 files (DAG validator + SessionRegistry edge handler). |

### Randomness Declaration

SessionRegistry uses PartitionedRNG subsystem `"agentic"` to sample tool latencies and output token counts. Key properties:

- **Lifecycle:** The `"agentic"` subsystem is created once at simulation start (same as other PartitionedRNG subsystems), seeded from the global simulation seed. It is NOT created per-session — all sessions share a single deterministic stream. This ensures reproducibility: same seed → same tool latency sequence → same child injection times across all sessions.
- **Isolation guarantee:** PartitionedRNG subsystems are independent streams derived from the global seed. Changing the routing policy (which uses the `"routing"` subsystem) does NOT affect tool latency samples from the `"agentic"` subsystem. This isolation enables Common Random Numbers (CRN) experiments — comparing two routing policies with identical agentic workload timing.
- **Sampling timing:** Tool latencies are sampled **endogenously at parent completion time**, not pre-generated during spec loading. When a parent LLM step completes and its child is a tool_call, the next value from the `"agentic"` RNG stream determines that tool's latency. This ensures tool latencies depend on execution order (deterministic given the same seed).

### No-Op Default Behavior

When `ClientSpec.Agentic` is nil (all existing workload specs), no SessionRegistry is instantiated. The request completion handler checks whether agentic metadata is present; if absent, no session tracking occurs. This ensures zero overhead for non-agentic workloads — no allocations, no branching beyond a nil check.

### Failure Modes

| Condition | Handling | Rule |
|-----------|----------|------|
| Malformed DAG (cycle detected) | Return validation error during spec loading | R1 (no silent data loss) |
| Missing tool reference | Return validation error during spec loading | R1 |
| Fan-in counter goes negative | Panic with descriptive message (logic error) | R6 (library panics on invariant violation) |
| Horizon expires mid-session | Session marked as incomplete; in-flight requests remain in their current state (queued/running); conservation law still holds | INV-A4, INV-1 |
| Empty workflow DAG (zero steps) | Return validation error | R20 (degenerate inputs) |
| `fan_out: 0` or `fan_out: 1` | Return validation error (`fan_out` must be >= 2) | R3 (validate CLI flags) |
| Fan-in parent dropped/preempted before completion | Fan-in step never injects; session is incomplete at horizon. SessionRegistry does NOT wait forever — horizon termination (INV-A4) handles this case. The fan-in counter remains >0, which is a valid terminal state when clock >= horizon. | INV-A4, INV-A2 |
| SessionRegistry fails to inject expected child (logic error) | Panic with descriptive message including session ID, step ID, and satisfied dependencies. This is an internal invariant violation, not a recoverable error. | R1, R6 |

### Session Lifecycle

1. **Creation:** When a root step's request is generated during Layer 1 pre-generation, a session record is created in SessionRegistry (keyed by session ID).
2. **Mutation:** On each step completion notification, SessionRegistry updates completion timestamps, decrements fan-in counters, and advances loop iteration if applicable.
3. **Deletion:** When all terminal steps complete (or horizon expires), session state is marked completed. Completed sessions are retained until simulation end for metrics collection, then garbage collected.
4. **Cardinality:** One SessionRegistry per simulation run. In single-instance mode, owned by the Simulator. In cluster mode, owned by ClusterSimulator (cluster-wide, since sessions may span routing decisions).

### Multi-Instance Coordination

In cluster mode, SessionRegistry lives in the ClusterSimulator (not per-instance). When a step completes on any instance, the cluster event loop triggers SessionRegistry, which computes child requests. Child requests are then injected as new ClusterArrivalEvents — subject to the full admission → routing pipeline. This means:

- **No session affinity enforced.** Child steps may route to different instances than their parents. This is intentional: it lets routing policies optimize across instances.
- **Signal freshness (INV-7):** Child injections create new ClusterArrivalEvents, so routing snapshots are taken at child arrival time (not parent completion time). Each child sees fresh routing state.
- **InFlightRequests:** Incremented when child is dispatched to an instance, decremented on completion — same as non-agentic requests.

## Deferred Injection Behavior

When a request with agentic session metadata completes, SessionRegistry identifies child steps whose dependencies are now satisfied and computes their arrival times:

- **LLM → LLM child:** Arrives immediately at parent completion time
- **LLM → tool_call child:** Arrives at parent completion time + sampled tool latency. Tool latency is sampled **endogenously at parent completion time** from the tool's latency distribution using the `"agentic"` PartitionedRNG subsystem (guaranteed non-negative by DistSpec validation, see Randomness Declaration).
- **Fan-in step:** Injected exactly once, after ALL parent steps complete. Arrival time = max(all parent completion times). Observable verification: the fan-in step appears in the event queue exactly once with arrival time >= max(parent completions).
- **Loop boundary:** If current iteration < max_iterations, restart loop body from its entry steps. If iteration == max_iterations, inject post-loop steps. Entry steps in a loop are those steps named in `loop.over` that have no `depends_on` referencing other loop steps — they form the loop's entry point.
- **Tool latency non-negativity:** Enforced at YAML validation time via DistSpec parameter validation (min >= 0). No runtime clamping needed.

SessionRegistry returns a list of child requests to the caller. It does not directly interact with the event queue — the caller (simulator or cluster) handles injection. This maintains separation of concerns.

## Key Invariants

| ID | Name | Property | Verification (observable behavior) |
|----|------|----------|-----------------------------------|
| INV-A1 | DAG causality | No child step arrives before all its parents complete | For every child request injected: `child.ArrivalTime >= max(parent.CompletionTime for all parents)`. Specializes INV-5 (causality) for DAG edges. |
| INV-A2 | Fan-in completeness | Fan-in step injected exactly once per iteration, after ALL parents transition to StateCompleted | For each iteration I of each session: the fan-in step is injected exactly once, only after ALL its parent steps have transitioned to StateCompleted (not merely StateRunning). In-loop fan-in is injected once per iteration; out-of-loop fan-in (post-loop) is injected exactly once for the session. If any parent is never completed (preempted, horizon-interrupted), the fan-in step is never injected — this is a valid terminal state under INV-A4. Tested by counting injections per step+iteration across entire simulation. |
| INV-A3 | Loop termination | Every loop terminates at or before `max_iterations` | `actual_iterations <= max_iterations` for every session. Horizon may terminate earlier. |
| INV-A4 | Session completeness | Every session either completes all terminal steps or is interrupted by horizon | At simulation end: for each session, either all terminal steps have completed requests, OR `clock >= horizon` and remaining steps are in queued/running/not-yet-injected state. No "stuck" sessions (session has incomplete steps but no in-flight work and clock < horizon). |
| INV-A5 | Tool latency non-negative | Sampled tool latencies are always >= 0 | Enforced by DistSpec validation (`min` parameter >= 0). |
| INV-1 | Request conservation (extended) | `injected_total == completed + queued + running + dropped` | `injected_total` = Layer 1 root requests + Layer 2 child requests. The injection counter is owned by SessionRegistry (global, not per-session) and incremented exactly once each time SessionRegistry produces a child request for injection. The counter is exposed in JSON output under `agentic_sessions.total_injected` for verification. Example: ReAct with 3 iterations (reason+act+observe per iteration + final-answer) = 1 root + 9 children = 10 injected_total. Extends existing INV-1 to include deferred injections. |
| INV-6 | Determinism | Same seed produces identical results | Tool latencies sampled from PartitionedRNG subsystem "agentic". Same seed → same completion times → same child injection times → same results. |
| INV-7 | Signal freshness (preserved) | Routing snapshots remain fresh for agentic child requests | Child requests are injected as new ClusterArrivalEvents. Each child takes a fresh routing snapshot at its own arrival time, not the parent's completion time. InFlightRequests (synchronous) and periodic signals (QueueDepth, KVUtilization) follow the existing freshness hierarchy unchanged. |
| INV-8 | Work-conserving (preserved) | Simulator does not idle while agentic work is waiting | After child request injection, if WaitQ.Len() > 0, a StepEvent must exist in the event queue. The existing work-conserving logic in `scheduleNextStep()` handles this — child ArrivalEvents flow through QueuedEvent which triggers StepEvent scheduling if needed. No special agentic handling required. |

## Agentic Metrics

| Metric | Description | Collection |
|---|---|---|
| Session E2E latency | Root step arrival → last terminal step completion | On-demand at session completion |
| Agent throughput | LLM requests completed per second for a single agentic session. Computed as `total_llm_steps / session_E2E_latency`. Measures how efficiently the inference system processes agentic workflows end-to-end, including tool wait gaps. | On-demand at session completion |
| Total inference time | Cumulative time spent in inference-related phases (wait time in queue + actual LLM step execution), excluding tool call latencies and inter-step gaps. Computed as `sum(queue_wait + step_execution)` across all LLM steps in the session. Wait time is currently deterministic per the scheduling policy, but becomes dynamic if queuing pressure changes across iterations (e.g., due to load from other sessions). | On-demand at session completion |
| Critical path latency | Longest dependency path through DAG (wall-clock) | Computed on-demand via DAG traversal of completed timestamps |

Metrics are collected per-session and aggregated across all sessions at simulation end. In JSON output, agentic session metrics appear under a new `agentic_sessions` field in `MetricsOutput`. Non-agentic simulations omit this field.

## Validation Rules

1. DAG must be acyclic. Steps in `loop.over` may form a cycle within the loop body, but all non-loop steps must form a DAG. Acyclicity is checked via topological sort during YAML loading.
2. If no loop is present: exactly one root step (no `depends_on`). If a loop is present: zero or one root steps. If zero roots, the loop body must contain at least one step with no intra-loop `depends_on` (the loop entry step).
3. Every `tool` reference in steps must exist in the `tools` section.
4. `fan_out` must be >= 2 when specified.
5. `loop.over` steps must form a connected subgraph.
6. `llm_call` steps require `input_distribution` and `output_distribution`.
7. `tool_call` steps must not have `input_distribution` or `output_distribution` (latency comes from `tools` section).
8. `context_growth: accumulate` is only valid for steps inside `loop.over`.
9. `agentic` is mutually exclusive with `reasoning` and `multimodal` on `ClientSpec`. If both are set, validation returns an error. Enforcement: checked in `validateClient()` during spec loading.
10. Tool latency distributions must have `min >= 0` (enforced by DistSpec parameter validation).
11. All YAML parsing uses `yaml.KnownFields(true)` for strict parsing (R10: typos must cause errors).

## Configuration and CLI

Agentic mode is controlled entirely by YAML — if `ClientSpec.Agentic` is present and non-nil, that client generates agentic sessions; otherwise, it generates regular requests. No new CLI flags are required.

The `validCategories` registry adds `"agentic"` to the allowed category set. `validateClient()` is extended to handle agentic-specific validation (steps, tools, DAG structure) alongside existing multimodal/reasoning validation.

## Codebase Integration Points

The following existing code requires modification (identified via R4 construction site audit):

1. **`sim/workload/spec.go`:** Add `AgenticSpec` type, add `Agentic *AgenticSpec` field to `ClientSpec`, add `"agentic"` to `validCategories`, extend `validateClient()`.
2. **`sim/workload/generator.go`:** Add agentic client handling in `GenerateRequests()` — generate only root step requests (Layer 1). Child steps are NOT pre-generated.
3. **`sim/request.go`:** Requests carry metadata identifying which workflow step they represent and which loop iteration (if any). All construction sites must be updated per R4.
4. **`sim/event.go` or `sim/simulator.go`:** After request completion processing, check for agentic metadata and notify SessionRegistry. SessionRegistry returns child requests to inject.
5. **`sim/cluster/cluster.go`:** In the shared-clock event loop, after processing instance events that cause completions, check for agentic children and schedule new ClusterArrivalEvents.
6. **`sim/cluster/metrics.go`:** Extend `CollectRawMetrics()` and `MetricsOutput` for agentic session metrics.

## Verification and Validation Strategy

### Verification (correctness)

Each invariant (INV-A1 through INV-A5, extended INV-1, INV-6) will have:
- **Table-driven tests** with concrete scenarios: ReAct (3 iterations), MCTS (fan-out 4 with fan-in), fork-join (3-way parallel tools)
- **Golden scenario:** A canonical ReAct + MCTS combined workload that serves as regression baseline
- **Invariant companion tests** per R7: every golden test has a companion that verifies the law, not just the output values

Concrete test scenarios (implemented in A0-2 and A0-3):
- INV-A1: Parent completes at t=1000; child must arrive at t >= 1000 (or t >= 1000 + tool_latency for tool edges)
- INV-A2: 4-way fan-out completes at t=1400, 1450, 1475, 1500. Fan-in step injected exactly once at t=1500.
- INV-A3: Loop with max_iterations=5 and horizon at iteration 3: verify actual_iterations <= 3
- INV-A4: Session with 10 steps, horizon cuts at step 6: verify steps 1-5 completed, step 6 in-progress, steps 7-10 not injected
- INV-1 (extended): After simulation, `injection_counter == completed + queued + running + dropped`
- INV-6: Run same seed twice with agentic workload, verify byte-identical stdout

### Validation (fidelity)

- **Analytical baseline:** For a ReAct session with N iterations and mean tool latency T_tool under low load (utilization < 50%, where queueing delay is negligible): `E[session_E2E] ≈ N × (E[LLM_latency] + T_tool)`. Verify simulation mean is within ±15% of this formula across 3 seeds (42, 123, 456). The ±15% tolerance accounts for sampling variance in tool latency distributions and LLM step time variation.
- **Preconditions for analytical baseline validity:** (1) Single instance or lightly loaded cluster (utilization < 50%). (2) Horizon >= 2× expected session E2E (sessions can complete without truncation). (3) No fan-out (linear chain only for baseline validation). Under higher load, queueing delay becomes significant and the baseline formula underestimates — this is expected, not a bug.
- **Falsification criteria:** If simulated session E2E latency diverges >30% from analytical prediction for a simple linear chain under the preconditions above, the deferred injection model has a bug.
- **Data sources for future calibration:** Tool latency distributions will initially use literature-informed defaults (web search ~50-100ms, DB queries ~20-50ms, code execution ~100-500ms). Post-implementation, calibration against real agentic traces (LangChain production logs, OpenAI assistants API latencies) will refine these distributions. This is tracked as a separate validation effort, not a blocker for Phase 1.

## DES Design Review Checklist

| # | Question | Answer |
|---|----------|--------|
| 1 | What analysis questions does this design help answer? | Q1-Q5 above (capacity planning, tool latency impact, fan-out optimization, tail latency, KV pressure) |
| 2 | What is modeled, simplified, and deliberately omitted? | See Modeling Decisions table. Key omissions: adaptive loop exit, dynamic tool selection, tool failures. |
| 3 | What events are introduced or modified? Exogenous or endogenous? | No new event types. Deferred injections produce standard ArrivalEvents (endogenous). Tool latency sampling uses PartitionedRNG subsystem "agentic". |
| 4 | How do new events interact with existing tie-breaking rules? | Injected child ArrivalEvents use same priority as regular arrivals. `(timestamp, priority, seqID)` ordering preserved. Determinism via seqID. |
| 5 | What new state is introduced? Who owns it? | SessionRegistry owns per-session state (DAG, completion tracking, fan-in counters, loop iteration, lifecycle). Request carries step metadata. |
| 6 | What new metrics are derived? Collected incrementally or on demand? | Four agentic metrics (see table). Session E2E, agent throughput, total inference time, and critical path latency — all computed on-demand at session completion. |
| 7 | How will correctness be verified? Which invariants? | INV-A1 through INV-A5 + extended INV-1 + INV-6. Table-driven tests with ReAct, MCTS, and fork-join scenarios. |
| 8 | How will fidelity be validated? Against what data? | Analytical baseline (±15% across 3 seeds for linear chains under low load, utilization < 50%). Falsification criteria: >30% divergence under stated preconditions indicates a bug. Future calibration against real agentic traces (LangChain, OpenAI assistants). |
| 9 | Does this introduce new randomness? Which PartitionedRNG subsystem? | Yes: subsystem "agentic" for tool latency and output token sampling. Isolated from existing streams. |
| 10 | What is the simplest version that answers the same questions? | Full DAG (Option C). Linear-only (Option A) cannot express MCTS. No-fan-in (Option B) cannot express fork-join. See Simplest Sufficient Version analysis. |

## PR Series

| PR | Title | Status | Depends On |
|----|-------|--------|------------|
| A0-1 | AgenticSpec schema + YAML loading + validation | Planned | W0-1 |
| A0-2 | SessionRegistry + deferred injection | Planned | A0-1 |
| A0-3 | Agentic metrics + invariant tests | Planned | A0-2 |
| A0-4 | Built-in scenarios + documentation | Planned | A0-3 |

### Dependency DAG

```
W0-1 (spec v2, complete)
  └──> A0-1 (schema + validation)
         └──> A0-2 (session registry + deferred injection)
                └──> A0-3 (metrics + invariant tests)
                       └──> A0-4 (scenarios + docs)
```

### Interface Freeze Points

A0-1 freezes the `AgenticSpec` YAML schema. After A0-1 merges, A0-2 and documentation work can proceed in parallel.

## Out of Scope (Future Phases)

- Nested sub-workflows (hierarchical agents)
- Conditional / probabilistic edges (enables adaptive loop exit)
- Runtime-dynamic branching (LLM output determines next step)
- Multi-model routing within a workflow (different steps targeting different models)
- `blis convert` for external agentic trace formats (separate issue)
- Tool cost models conditioned on input complexity
- Fan-in timeout/deadline policies (currently waits for all parents; future phase could add configurable timeout)

## Backward Compatibility

- `agentic` field on `ClientSpec` is optional (`omitempty`) — no existing YAML breaks
- Category `"agentic"` is additive to the valid category set
- Non-agentic requests bypass SessionRegistry entirely (nil check, no allocations, no map lookups)
- All existing workload specs (language, multimodal, reasoning) are unaffected
- Agentic metrics field in JSON output is omitted for non-agentic simulations

## References

- #420 — Phase 0: Workload Unification (this feature extends WorkloadSpec v2 from that effort)
