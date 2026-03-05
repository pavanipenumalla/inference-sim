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

1. **Q1:** How does agentic loop iteration count (ReAct depth) affect cluster capacity requirements compared to equivalent independent-request workloads? Note: Phase 1 uses fixed `max_iterations` (adaptive exit is omitted), so Q1 results represent upper-bound capacity requirements — real agents that exit early will use fewer resources.
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
| DAG structure (step dependencies) | Proposed | **Modeled** | Agents execute steps in dependency order | Workflow spec may contain loop back-edges, but execution is acyclic: loops are unrolled at runtime into per-iteration step instances, producing an acyclic execution graph. Fan-out and fan-in are first-class primitives. |
| Tool call latency (external delays) | Proposed | **Modeled** | External services have variable latency | Sampled from configurable distributions per tool type |
| Fan-out/fan-in (parallel candidates) | Proposed | **Modeled** | MCTS/ToT spawn parallel candidates, aggregate results | N parallel instances with fan-in synchronization |
| Loop iteration with max bound | Proposed | **Modeled** | ReAct loops repeat until satisfied or max rounds | Fixed max_iterations; horizon also terminates |
| Context accumulation across iterations | Proposed | **Modeled** | Agent context grows as tool outputs accumulate | Input token count grows by prior iterations' output tokens |
| GPU idle during tool execution | Proposed | **Simplified** | GPU may serve other requests during async tool calls | Tool calls are modeled as pure wall-clock delays. GPU is released — the parent request completes normally (leaves the running batch, frees KV blocks), and the child request arrives as a new independent request after tool latency elapses. In cluster mode, the child may route to any instance (not necessarily the parent's instance). **What breaks if wrong:** Complete KV release forces every child request to re-prefill from scratch, overestimating prefill cost for sessions where real systems retain KV cache (vLLM same-session, SGLang radix attention). This makes Q5 KV pressure results pessimistic (overestimates churn). Expected magnitude: for a ReAct agent with N iterations and mean input I tokens, each iteration re-prefills ~I tokens instead of extending incrementally, yielding ~N× overestimate of prefill KV allocation events compared to perfect session affinity. Acceptable for Phase 1: pessimistic estimates are safe for capacity planning (over-provisioning is cheaper than under-provisioning). |
| Tool output conditioning on input | Proposed | **Simplified** | Tool output size may depend on query complexity | Output tokens sampled independently from tool's output_tokens distribution. Lost: correlation between query complexity and result size. **What breaks if wrong:** For tools with positive input-output correlation (e.g., complex web queries returning longer results), context accumulation variance is underestimated — KV pressure tail events (p99) may be more severe than simulated. Mean KV utilization impact is bounded (<10% based on unconditional vs conditional variance analysis for typical log-normal outputs), but tail KV pressure could diverge further. Acceptable for Phase 1: conditioning would require learned models from real traces. |
| Adaptive loop exit (LLM decides when to stop) | Deferred | **Omitted** | Agents decide when they have enough information | Loop runs to max_iterations or horizon. Lost: variable iteration counts driven by LLM confidence. Acceptable: capacity planning needs worst-case bounds; adaptive exit is a future phase with probabilistic edges. |
| Dynamic tool selection (LLM chooses tool at runtime) | Deferred | **Omitted** | Agents select tools based on reasoning output | Tool at each step is fixed in the workflow DAG. Lost: runtime-dependent tool selection. Acceptable: workflow DAGs capture the structural pattern; dynamic selection is a future phase. |
| Tool failures and retries | Deferred | **Omitted** | Tools may fail and require retry loops | All tool calls succeed. Lost: retry amplification and error handling overhead. Acceptable: retry semantics would require conditional edges (future phase). |
| Tool output tokenization overhead | Deferred | **Omitted** | Real systems re-tokenize tool outputs | Tokenization is assumed instant. Lost: tokenization latency (~1-5ms). Acceptable: negligible compared to tool latency (50-500ms). |
| Session affinity (KV cache locality) | Proposed | **Simplified** | SGLang radix attention retains KV for shared prefixes; llm-d uses prefix-aware routing to keep related requests on the same instance | No session affinity enforced — child steps route through the standard admission → routing pipeline and may land on any instance. Lost: KV cache reuse across session steps on the same instance. **What breaks if wrong:** Q5 KV pressure results are pessimistic (overestimate churn) because each child re-prefills. The existing prefix-affinity scorer may partially recover locality for child requests that share a prefix with the parent, but this is incidental, not guaranteed. Acceptable for Phase 1: pessimistic estimates are safe for capacity planning. Future phase: optional soft-affinity scorer weight for session continuity. |
| Agentic + reasoning/multimodal combination | Proposed | **Simplified** | Real agents use chain-of-thought within steps (reasoning) and process images via tools (multimodal) | `agentic` is mutually exclusive with `reasoning` and `multimodal` on a single `ClientSpec`. Lost: per-step reasoning ratios and multimodal input within agentic workflows. Acceptable for Phase 1: each step's token distributions already capture the aggregate effect of reasoning/multimodal content. Users modeling reasoning-within-agentic should inflate step-level output distributions to account for reasoning tokens (typically 3-10x visible output). A mixed workload spec can include separate agentic and reasoning/multimodal clients. Future phase: composable extensions per step. |
| Mid-session admission control | Proposed | **Simplified** | Real orchestrators (LangChain, AutoGen) guarantee in-flight session completion; gateways may exempt in-flight sessions from rate limiting | Child requests go through the full admission → routing pipeline and may be rejected by admission policies (e.g., TokenBucket). Lost: session completion guarantees. **What breaks if wrong:** A rejected child kills the downstream subgraph, creating partial sessions that inflate Q4 tail latency. Acceptable for Phase 1: this models a worst-case where the inference gateway has no session-awareness. The existing `AlwaysAdmit` default avoids this issue; `TokenBucket` users should be aware that mid-session rejection is possible. Future phase: session-aware admission that exempts in-flight sessions. |
| Request priority inheritance | Proposed | **Modeled** | Child requests in agentic sessions share the parent session's priority class for scheduling and admission | Child requests inherit `slo_class` and `tenant_id` from the root request (set once at session creation, immutable). This ensures all steps in a session receive consistent scheduling treatment — a `critical` session's tool-call children are not demoted to `standard`. No per-step priority overrides in Phase 1. Lost: per-step priority differentiation (e.g., evaluate step at higher priority than generate step). Acceptable: uniform session priority matches real orchestrator behavior (LangChain, AutoGen treat all steps within a task equally). |

### Real-System Correspondence

| BLIS Feature | vLLM | SGLang | llm-d | LangChain/AutoGen |
|-------------|------|--------|-------|-------------------|
| DAG dependency tracking | Not native (client-side orchestration) | Not native | Gateway-level orchestration | Native DAG execution |
| Tool call latency | Async via external tools; GPU released | Similar to vLLM | Router-managed tool dispatch | Framework-managed async |
| Fan-out (parallel candidates) | Client spawns parallel requests | Client-side batching | Load-balanced across instances | Native parallel tool calls |
| Fan-in (aggregation) | Client-side join | Client-side join | Gateway-side join | Native gather/reduce |
| GPU release during tool calls | GPU freed after response; new request needed | Same | Same | N/A (no GPU management) |
| Context accumulation | KV cache retained if same session (incremental extension) | Radix attention for shared prefix (cross-session sharing) | Prefix-aware routing | Framework manages context window |

Note: BLIS models neither vLLM's incremental KV extension nor SGLang's cross-session radix sharing — it releases KV entirely between steps. This makes Q5 results pessimistic for both systems. See GPU idle modeling decision for quantification.

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
| Tool output conditioning on input | Output tokens independent of input complexity | Q2: critical path analysis unaffected (tool latency dominates, not output size). Q5: context accumulation slightly biased if real correlation is strong, but <10% effect on KV pressure for typical distributions | Acceptable omission: weak correlation in practice; conditioning would require learned models from real traces |
| Fan-out / KV capacity interaction | Fan-out concentrates N simultaneous requests on instances | Q3: fan-out amplification can overwhelm single-instance KV capacity, causing cascading preemptions. MCTS with fan_out: 4 across 3 levels produces 21 LLM calls per session. In single-instance mode or under routing concentration, this creates transient KV pressure spikes. | Modeled implicitly: the existing KV cache, batch formation, and preemption logic handle this naturally. The interaction is emergent, not a separate component. Users should monitor preemption rates when configuring high fan-out factors. Real systems (vLLM, SGLang) handle this via client-side rate limiting, which is not modeled (deferred). |

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
- **Fan-out:** `fan_out: N` on a step spawns N parallel instances at runtime. Fan-out instances are **iteration-scoped**: within a loop, each iteration produces its own set of N instances, and fan-in counters are keyed by `(step_id, iteration_index)`. A downstream step that `depends_on` a fan-out step implicitly depends on **all N instances from the same iteration** (fan-in). For example, if `evaluate` depends on `["generate"]` and `generate` has `fan_out: 4`, then `evaluate` waits for all four generate instances from the current iteration to complete before injecting. Multiple steps sharing the same `depends_on` (without `fan_out`) run in parallel.
- **Fan-in:** one step with `depends_on` referencing one or more parents — waits for all parents (including all fan-out instances) to **complete** (not just arrive). Fan-in resolution is implicit when depending on a fan-out step.
- **Loop:** `loop.over` names repeating steps; steps outside the loop run after loop exits

### Context Growth Semantics

When `context_growth: accumulate` is set on a loop step, the effective input length for each iteration includes all prior iterations' output tokens, modeling the agent's growing context window as tool outputs accumulate. Specifically: each iteration's base input is freshly sampled from the step's `input_distribution`, then all output tokens produced in prior iterations of that step are added. The effective input is capped at the model's maximum context window. The accumulated tokens contribute to KV cache allocation and latency model calculations.

**Design choice:** `accumulate` is the simplest and most pessimistic context growth model. Real agents use additional strategies: sliding window (retain only last K iterations), summarization (compression ratio on older context), and RAG-style retrieval (selective inclusion). Phase 1 models only `accumulate` — it represents the worst-case for KV pressure analysis (Q5). Future phases may add `sliding_window` with a configurable retention count.

**Monotonicity guarantee:** For accumulate-mode steps, effective input length is non-decreasing across iterations (prior outputs only grow). This directly drives KV cache pressure growth for Q5 analysis.

**Latency model interaction:** Because GPU resources are fully released between steps (see GPU idle modeling decision), every iteration's accumulated-context request is treated as a **new prefill** of the full accumulated input — not an incremental KV extension. This means the latency model receives the full accumulated token count as its prefill input for each iteration. For the blackbox model, this affects alpha (queueing) coefficients; for roofline, it affects prefill FLOPs; for cross-model, it affects per-layer KV bandwidth. This is a direct consequence of the "no session affinity" simplification and makes KV pressure pessimistic (Q5).

**Non-loop steps (independent input):** Steps outside a loop (or loop steps without `context_growth: accumulate`) sample their input tokens independently from their `input_distribution`. Tool output tokens from parent tool_call steps do NOT automatically flow into the child LLM step's input — each step's `input_distribution` should be configured to account for expected tool output sizes. For example, in a fork-join workflow, the synthesize step's `input_distribution` should be sized to include the expected aggregate tool outputs. This is a deliberate simplification: real context assembly depends on the orchestrator's prompt template, which varies across frameworks. Users should size distributions based on observed prompt lengths from their target framework.

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
| **Observes** | Request completion notifications and drop notifications (session ID, step ID, completion/drop timestamp, terminal status). Workflow DAG (loaded at session creation, frozen thereafter). Does NOT observe global clock, KV state, or routing state. Note: preemption events are NOT observed — preempted requests remain in-flight and may eventually complete or be dropped. |
| **Controls** | (1) When child requests are injected into the event queue (deferred injection timing). (2) Fan-in synchronization — whether a fan-in step's dependencies are all satisfied. (3) Loop iteration advancement and termination. |
| **Owns** | Per-session mutable state: workflow DAG reference, completion timestamps per step+iteration, fan-in dependency counters, current loop iteration, session lifecycle state (active/completed/incomplete-unsatisfiable). The total injection count for INV-1 verification is derived on-demand from session records (sum of injected steps across all sessions), not maintained as a running counter — this keeps state and statistics cleanly separated. |
| **Invariants** | INV-A1 (DAG causality), INV-A2 (fan-in completeness), INV-A3 (loop termination), INV-A4 (session completeness), INV-A5 (tool latency non-negative). See Invariants section. |
| **Events** | **Consumes:** Completion notification (triggered when a request with agentic metadata completes). **Produces:** Child requests returned to caller for injection as standard `ArrivalEvent`s. No new event types — reuses existing event infrastructure. Priority: same as standard ArrivalEvents. |
| **Extension friction** | Adding a second SessionRegistry implementation (e.g., persistent/distributed): ~4 files (extract interface + new implementation + factory + CLI flag). Adding a new workflow pattern (e.g., conditional edges): ~2 files (DAG validator + SessionRegistry edge handler). |

**Behavioral interface:** SessionRegistry exposes two interaction points sufficient for mock-based testing: (1) **Notify completion** — accepts a completed request's agentic metadata (session ID, step ID, completion timestamp, terminal status) and returns zero or more child requests ready for injection. (2) **Query session state** — returns session lifecycle state (active/completed/incomplete) for metrics collection. These two interaction points are sufficient to test INV-A1 through INV-A5 in isolation using mock completion sequences without running a full simulation.

### Randomness Declaration

SessionRegistry uses PartitionedRNG subsystem `"agentic"` to sample tool latencies and output token counts. Key properties:

- **Lifecycle:** The `"agentic"` subsystem is created once at simulation start (same as other PartitionedRNG subsystems), seeded from the global simulation seed. It is NOT created per-session — all sessions share a single deterministic stream. This ensures reproducibility: same seed → same tool latency sequence → same child injection times across all sessions.
- **Isolation guarantee:** PartitionedRNG subsystems are independent streams derived from the global seed. Changing the routing policy (which uses the `"routing"` subsystem) does NOT affect tool latency samples from the `"agentic"` subsystem. This isolation enables Common Random Numbers (CRN) experiments — comparing two routing policies with identical agentic workload timing.
- **Sampling timing:** Tool latencies and tool output token counts are sampled **endogenously at parent completion time** from the `"agentic"` subsystem, not pre-generated during spec loading. When a parent LLM step completes and its child is a tool_call, the next value from the `"agentic"` RNG stream determines that tool's latency. LLM step output tokens are sampled from the existing `"workload-gen"` subsystem (same as non-agentic requests), preserving CRN isolation between tool timing and LLM output generation.
- **Fan-out draw ordering:** When a fan-out step spawns N children, RNG draws for child requests are made in deterministic index order (0, 1, ..., N-1). This ensures INV-6 (determinism) holds regardless of internal data structure iteration order.

### No-Op Default Behavior

When `ClientSpec.Agentic` is nil (all existing workload specs), no SessionRegistry is instantiated. The request completion handler checks whether agentic metadata is present; if absent, no session tracking occurs. This ensures zero observable overhead for non-agentic workloads — existing behavior is completely unchanged.

### Failure Modes

| Condition | Handling | Rule |
|-----------|----------|------|
| Malformed DAG (cycle detected) | Return validation error during spec loading | R1 (no silent data loss) |
| Missing tool reference | Return validation error during spec loading | R1 |
| Fan-in counter goes negative | Panic with descriptive message (logic error) | R6 exception: panic is acceptable for internal invariant violations that indicate logic bugs (not user-facing errors). This follows the existing pattern in `sim/` where `MustNewKVCacheState` and `MustNewKVStoreFromConfig` panic on nil-guard violations. These are programmer errors, not recoverable runtime conditions. |
| Horizon expires mid-session | Session marked as incomplete; in-flight requests remain in their current state (queued/running); conservation law still holds | INV-A4, INV-1 |
| Empty workflow DAG (zero steps) | Return validation error | R20 (degenerate inputs) |
| `fan_out: 0` or `fan_out: 1` | Return validation error (`fan_out` must be >= 2) | R3 (validate CLI flags) |
| Fan-in parent preempted (requeued) | Preempted requests are requeued by the batch formation logic (state reset to `StateQueued`, re-enters WaitQ). SessionRegistry takes no action on preemption — the parent is still in-flight and may eventually complete, satisfying the fan-in counter normally. Fan-in counters remain unchanged; the session continues to wait. This is correct behavior: preemption is temporary, not terminal. | INV-A2 |
| Fan-in parent dropped as unservable | Dropped requests are permanently removed (counted in `DroppedUnservable`). SessionRegistry is notified of the drop via the completion pipeline (the drop event carries agentic metadata). SessionRegistry marks the parent step as permanently failed, making downstream fan-in steps unsatisfiable. The session transitions to the incomplete-unsatisfiable terminal state under INV-A4(c). | INV-A4, INV-A2 |
| Context accumulation exceeds KV capacity | An accumulate-mode request in a late iteration may have effective input tokens exceeding KV block capacity. The request is dropped as unservable (existing `DroppedUnservable` handling). SessionRegistry is notified of the drop, marking downstream fan-in steps as permanently unsatisfiable under INV-A4(c). | INV-A4, existing KV capacity enforcement |
| Mid-session admission rejection | A child request rejected by admission control (e.g., TokenBucket). Downstream subgraph becomes dead. Session transitions to incomplete-unsatisfiable under INV-A4(c). | INV-A4; see mid-session admission control modeling decision |
| SessionRegistry fails to inject expected child (logic error) | Panic with descriptive message including session ID, step ID, and satisfied dependencies. This is an internal invariant violation, not a recoverable error. | R1; R6 exception (same justification as fan-in counter panic — logic bug, not user error) |

### Session Lifecycle

1. **Creation:** When a root step's request is generated during Layer 1 pre-generation, a session record is created in SessionRegistry (keyed by session ID).
2. **Mutation:** On each step completion notification, SessionRegistry updates completion timestamps, decrements fan-in counters, and advances loop iteration if applicable.
3. **Deletion:** When all terminal steps complete (or horizon expires), session state is marked completed. Completed sessions are retained until simulation end for metrics collection, then garbage collected.
4. **Cardinality:** One SessionRegistry per simulation run. In single-instance mode, the Simulator owns SessionRegistry and invokes it from the completion pipeline. In cluster mode, ClusterSimulator owns SessionRegistry (cluster-wide, since sessions may span routing decisions); per-instance Simulators have no SessionRegistry — the cluster coordinator intercepts completions and invokes SessionRegistry before injecting children through the admission → routing pipeline. The ownership is exclusive: exactly one layer (Simulator or ClusterSimulator) drives SessionRegistry, never both.

### Multi-Instance Coordination

In cluster mode, SessionRegistry lives in the ClusterSimulator (not per-instance). When a step completes on any instance, the cluster event loop triggers SessionRegistry, which computes child requests. Child requests are then injected as new ClusterArrivalEvents — subject to the full admission → routing pipeline. This means:

- **No session affinity enforced.** Child steps may route to different instances than their parents. This is intentional: it lets routing policies optimize across instances.
- **Signal freshness (INV-7):** Child injections create new ClusterArrivalEvents, so routing snapshots are taken at child arrival time (not parent completion time). Under `--snapshot-refresh-interval 0` (Immediate mode), each child sees fresh routing state. Under periodic refresh (`--snapshot-refresh-interval > 0`), fan-out children arriving at the same timestamp see identically stale snapshots, which may concentrate them on the same instance. This is consistent with how all burst arrivals behave under periodic refresh — not specific to agentic workloads. Users running MCTS/fan-out patterns under periodic refresh should be aware of this concentration effect.
- **InFlightRequests:** Incremented when child is dispatched to an instance, decremented on completion — same as non-agentic requests.

## Deferred Injection Behavior

When a request with agentic session metadata completes, SessionRegistry identifies child steps whose dependencies are now satisfied and computes their arrival times:

- **LLM → LLM child:** Arrives immediately at parent completion time
- **LLM → tool_call child:** Arrives at parent completion time + sampled tool latency. Tool latency is sampled **endogenously at parent completion time** from the tool's latency distribution using the `"agentic"` PartitionedRNG subsystem (guaranteed non-negative by DistSpec validation, see Randomness Declaration).
- **Fan-in step:** Injected exactly once, after ALL parent steps complete. Arrival time = max(all parent completion times). Observable verification: the fan-in step appears in the event queue exactly once with arrival time >= max(parent completions).
- **Loop boundary:** If current iteration < max_iterations, restart loop body from its entry steps. If iteration == max_iterations, inject post-loop steps. Entry steps in a loop are those steps named in `loop.over` that have no `depends_on` referencing other loop steps — they form the loop's entry point.
- **Tool latency non-negativity:** Enforced at YAML validation time via DistSpec parameter validation (min >= 0). No runtime clamping needed.

SessionRegistry returns a list of child requests to the caller. It does not directly interact with the event queue — the caller (simulator or cluster) handles injection by **enqueuing** child requests as standard events into the event queue. Children are never processed inline during the parent's completion handler; they are enqueued and processed in proper `(timestamp, priority, seqID)` order by the event loop. This maintains both separation of concerns and DES event ordering guarantees.

## Key Invariants

| ID | Name | Property | Verification (observable behavior) |
|----|------|----------|-----------------------------------|
| INV-A1 | DAG causality | No child step arrives before all its parents complete | For every child request injected: `child.ArrivalTime >= max(parent.CompletionTime for all parents)`. Specializes INV-5 (causality) for DAG edges. |
| INV-A2 | Fan-in completeness | Fan-in step injected exactly once per iteration, after ALL parents transition to StateCompleted | For each iteration I of each session: the fan-in step is injected exactly once, only after ALL its parent steps have transitioned to StateCompleted (not merely StateRunning). In-loop fan-in is injected once per iteration; out-of-loop fan-in (post-loop) is injected exactly once for the session. If any parent is never completed (e.g., horizon-interrupted or dropped as unservable), the fan-in step is never injected — this is a valid terminal state under INV-A4. Tested by counting injections per step+iteration across entire simulation. |
| INV-A3 | Loop termination | Every loop terminates at or before `max_iterations` | `actual_iterations <= max_iterations` for every session. Horizon may terminate earlier. |
| INV-A4 | Session completeness | Every session either completes all terminal steps, is interrupted by horizon, or has unsatisfiable dependencies from dropped parents | At simulation end: for each session, one of three states holds: (a) all terminal steps have completed, (b) `clock >= horizon` and remaining steps are in queued/running/not-yet-injected state, or (c) one or more dependency parents were **dropped as unservable** (not merely preempted — preempted requests are requeued and may still complete), making downstream fan-in steps permanently unsatisfiable. Note: preemption is temporary (request requeued) and does NOT trigger the unsatisfiable state; only permanent drops (e.g., input exceeds KV capacity) do. No session may have incomplete steps with all dependencies satisfiable, no in-flight work, and clock < horizon. |
| INV-A5 | Tool latency non-negative | Sampled tool latencies are always >= 0 | Enforced by DistSpec validation (`min` parameter >= 0). |
| INV-1 | Request conservation (extended) | `injected_total == completed + queued + running + dropped` | `injected_total` = Layer 1 root requests + Layer 2 child requests. The total is derived on-demand from session records (sum of injected steps across all sessions). Example: ReAct with 3 iterations (reason+act+observe per iteration + final-answer) = 1 root + 9 children = 10 injected_total. Extends existing INV-1 to include deferred injections. |
| INV-6 | Determinism | Same seed produces identical results | Tool latencies sampled from PartitionedRNG subsystem "agentic". Same seed → same completion times → same child injection times → same results. |
| INV-7 | Signal freshness (preserved) | Routing snapshots remain fresh for agentic child requests | Child requests are injected as new ClusterArrivalEvents. Each child takes a fresh routing snapshot at its own arrival time, not the parent's completion time. InFlightRequests (synchronous) and periodic signals (QueueDepth, KVUtilization) follow the existing freshness hierarchy unchanged. |
| INV-8 | Work-conserving (preserved) | Simulator does not idle while agentic work is waiting | After child request injection, if WaitQ.Len() > 0, a StepEvent must exist in the event queue. The existing work-conserving logic in `scheduleNextStep()` handles this — child ArrivalEvents flow through QueuedEvent which triggers StepEvent scheduling if needed. No special agentic handling required. |

## Agentic Metrics

| Metric | Description | Collection |
|---|---|---|
| Session E2E latency | Root step arrival → last terminal step completion | On-demand at session completion |
| Agent throughput | LLM requests completed per second for a single agentic session. Computed as `total_llm_steps / session_E2E_latency`. Measures how efficiently the inference system processes agentic workflows end-to-end, including tool wait gaps. | On-demand at session completion |
| Total inference time | Cumulative time spent in inference-related phases (wait time in queue + actual LLM step execution), excluding tool call latencies and inter-step gaps. Computed as `sum over all LLM steps, over all attempts per step (queue_wait_per_attempt + execution_time_per_attempt)`. A single LLM step may have multiple attempts due to preemption — each attempt's queue wait and execution time counts. Wait time is currently deterministic per the scheduling policy, but becomes dynamic if queuing pressure changes across iterations (e.g., due to load from other sessions). | On-demand at session completion |
| Critical path latency | Longest dependency path through DAG (wall-clock) | Computed on-demand via DAG traversal of completed timestamps |

**Incomplete sessions:** Sessions interrupted by horizon or with unsatisfiable dependencies (INV-A4 states b/c) are excluded from session E2E and agent throughput aggregation. They are reported separately under `agentic_sessions.incomplete_count` with the reason (horizon vs. dropped dependency). This avoids inflating tail latency with sentinel values while preserving visibility into session completion rates.

**Preempted requests:** Total inference time counts all queue wait time and execution time incurred by each LLM step, including wasted execution from preempted attempts and the additional queue wait from requeuing. This reflects the true GPU cost and wall-clock delay experienced by the session — preemption is not free, and capacity planning must account for the overhead.

Metrics are collected per-session and aggregated across all sessions at simulation end. In JSON output, agentic session metrics appear under a new `agentic_sessions` field in `MetricsOutput`. Non-agentic simulations omit this field.

## Validation Rules

1. DAG must be acyclic. Steps in `loop.over` may form a cycle within the loop body, but all non-loop steps must form a DAG. Acyclicity is checked via topological sort during YAML loading.
2. If no loop is present: exactly one root step (no `depends_on`). If a loop is present: zero or one root steps. If zero roots, the loop body must contain at least one step with no intra-loop `depends_on` (the loop entry step).
3. Every `tool` reference in steps must exist in the `tools` section.
4. `fan_out` must be >= 2 when specified.
4a. `max_iterations` must be >= 1 when specified (R3: validate for zero; R20: zero iterations is a degenerate input that produces a loop with no iterations, which is behaviorally equivalent to omitting the loop).
5. `loop.over` steps must form a connected subgraph in the static (pre-expansion) step graph. Fan-out expansion is a runtime concern; connectivity is validated on step IDs as declared in the workflow spec.
6. `llm_call` steps require `input_distribution` and `output_distribution`.
7. `tool_call` steps must not have `input_distribution` or `output_distribution` (latency comes from `tools` section).
8. `context_growth: accumulate` is only valid for steps inside `loop.over`.
9. `agentic` is mutually exclusive with `reasoning` and `multimodal` on `ClientSpec`. If both are set, validation returns an error. Enforcement: checked in `validateClient()` during spec loading.
10. Tool latency distributions must have an explicit `min` parameter with value >= 0. This is mandatory (not optional) for all tool latency distributions regardless of distribution type, because some types (e.g., gaussian) can sample negative values without a floor. Enforced by DistSpec parameter validation during YAML loading.
11. All YAML parsing uses `yaml.KnownFields(true)` for strict parsing (R10: typos must cause errors).

## Configuration and CLI

Agentic mode is controlled entirely by YAML — if `ClientSpec.Agentic` is present and non-nil, that client generates agentic sessions; otherwise, it generates regular requests. No new CLI flags are required.

The `validCategories` registry adds `"agentic"` to the allowed category set. `validateClient()` is extended to handle agentic-specific validation (steps, tools, DAG structure) alongside existing multimodal/reasoning validation.

## Module Boundary Interactions

SessionRegistry integrates with four existing module boundaries:

1. **Workload specification layer:** The `AgenticSpec` extends `ClientSpec` as an optional field, following the same pattern as multimodal and reasoning extensions. The workload category registry gains `"agentic"`. Client validation is extended to enforce agentic-specific structural rules (DAG acyclicity, tool references, fan-out constraints).
2. **Request generation pipeline:** Only root steps are pre-generated during Layer 1 workload generation. Child steps are deferred to Layer 2 (endogenous injection at runtime). Root and child requests carry **immutable** agentic metadata set once at creation and never modified thereafter. The metadata must be sufficient to identify the session, the step within the workflow, and the loop iteration (if applicable). This metadata is a label, not mutable state — the event loop, scheduler, and routing policies read it but never write it. SessionRegistry reads this metadata from completed requests to determine which session/step completed.
3. **Completion pipeline:** When any request completes, the completion handler checks for agentic metadata. If present, SessionRegistry is notified and returns child requests for injection. In single-instance mode, the simulator injects children as standard arrival events. In cluster mode, the cluster coordinator injects children through the full admission → routing pipeline.
4. **Metrics aggregation:** Agentic session metrics (session E2E, agent throughput, total inference time, critical path) are collected per-session and aggregated at simulation end. The metrics output gains an optional agentic section, omitted for non-agentic workloads.

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
- INV-A4(b): Session with 10 steps, horizon cuts at step 6: verify steps 1-5 completed, step 6 in-progress, steps 7-10 not injected
- INV-A4(c): Fan-out 3, one branch's child exceeds KV capacity and is dropped as unservable. Verify: fan-in step is never injected, session classified as incomplete-unsatisfiable, downstream subgraph steps (2+) are never injected
- INV-1 (extended): After simulation, `injection_counter == completed + queued + running + dropped`
- INV-6: Run same seed twice with agentic workload, verify byte-identical stdout

### Validation (fidelity)

- **Analytical baseline:** For a ReAct session with N iterations under low load (utilization < 50%, where queueing delay is negligible): `E[session_E2E] ≈ N × (E[reason_latency] + E[tool_latency] + E[observe_latency]) + E[final_answer_latency]`. Each iteration has two LLM steps (reason + observe) and one tool call; the final-answer step runs once after the loop. Verify simulation mean is within ±15% of this formula across 3 seeds (42, 123, 456) with a minimum of 50 sessions per seed (to reduce per-seed variance). The ±15% tolerance accounts for sampling variance in tool latency distributions and LLM step time variation.
- **Preconditions for analytical baseline validity:** (1) Single instance or lightly loaded cluster (utilization < 50%). (2) Horizon >= 2× expected session E2E (sessions can complete without truncation). (3) No fan-out (linear chain only for baseline validation). Under higher load, queueing delay becomes significant and the baseline formula underestimates — this is expected, not a bug.
- **Fork-join analytical baseline:** For a fork-join workflow with K parallel tool calls under low load: `E[session_E2E] ≈ E[plan_latency] + E[max(tool_1, ..., tool_K)] + E[synthesize_latency]`. The `E[max]` term has known closed forms for exponential distributions: `E[max of K exponentials with mean μ] = μ × H_K` where `H_K` is the K-th harmonic number. Verify within ±15% across 3 seeds under the same low-load preconditions.
- **MCTS analytical baseline:** For a MCTS workflow with fan_out F under low load: `E[session_E2E] ≈ E[decompose] + E[max(generate[0..F-1])] + E[evaluate] + E[tool_latency] + E[refine]`. The `E[max]` of F independent generate steps follows the order statistic for the generate step's latency distribution. Additionally, verify total LLM call count equals `1 + F + 1 + 1 = F + 3` (decompose + F generate + evaluate + refine). Verify within ±15% across 3 seeds under the same low-load preconditions.
- **Falsification criteria:** If simulated session E2E latency diverges >30% from analytical prediction for a simple linear chain, fork-join, or MCTS pattern under the preconditions above, the deferred injection model has a bug.
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
| 8 | How will fidelity be validated? Against what data? | Analytical baselines for all three primary patterns (ReAct linear chain, fork-join, MCTS) under low load (utilization < 50%, ±15% across 3 seeds with 50+ sessions per seed). Falsification criteria: >30% divergence under stated preconditions indicates a bug. Future calibration against real agentic traces (LangChain, OpenAI assistants). |
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
