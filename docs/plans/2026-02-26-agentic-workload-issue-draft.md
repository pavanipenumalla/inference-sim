# Phase 1: Agentic Workload Generation

## Summary

BLIS can simulate language, multimodal, and reasoning workloads — but all of these model independent requests or sequential multi-turn sessions. Real agentic inference workloads have fundamentally different structure: parent-child dependencies between LLM calls, tool call latency (external service delays), fan-out (parallel candidates), fan-in (aggregation), and iteration loops.

This issue proposes extending `WorkloadSpec` v2 with an `AgenticSpec` that defines workflow DAGs using three composable primitives — `depends_on`, `fan_out`, and `loop` — to express all major agentic patterns:

- **ReAct agents** — sequential loop: reason → tool → observe → repeat
- **MCTS / Tree-of-Thought** — tree: fan-out N candidates → evaluate → expand best
- **Parallel tool calls (fork-join)** — plan → [tool₁ ∥ tool₂ ∥ tool₃] → synthesize
- **Multi-step chains** — linear DAG with heterogeneous tool latencies

Design document: [`docs/plans/2026-02-26-agentic-workload-generation-design.md`](docs/plans/2026-02-26-agentic-workload-generation-design.md)

## Motivation

Agentic workloads are becoming a dominant inference pattern. They differ from standard workloads in ways that matter for capacity planning:

1. **Request dependencies** — a child LLM call cannot start until its parent completes and any intermediate tool call finishes. This creates serial chains that amplify tail latency.
2. **Tool call latency** — external service calls (web search, code execution, DB queries) introduce non-LLM delays that consume wall-clock time but not GPU resources.
3. **Fan-out amplification** — MCTS/ToT patterns multiply request count per session by the branch factor. A 4-wide tree with 3 levels produces 21 LLM calls per session.
4. **Variable session length** — ReAct loops have unpredictable iteration counts, making capacity planning harder than fixed-length workloads.

Without agentic workload support, BLIS cannot model these patterns for capacity planning, policy optimization, or performance prediction.

## Two-Layer Arrival Time Model

Agentic workloads require two layers of timing:

**Layer 1 — Session arrivals (pre-generated).** When does each new agentic session start? Controlled by existing `ArrivalSpec` on `ClientSpec` (poisson, gamma, weibull, constant). Only root steps are pre-generated.

**Layer 2 — Intra-session step arrivals (deferred injection).** When does each step within a workflow execute? Governed by DAG dependencies at simulation runtime:

| Edge type | Child arrival time |
|---|---|
| LLM → LLM child | `T_complete` (immediate) |
| LLM → tool_call child | Tool completes at `T_complete + sample(tool.latency)` |
| tool_call → LLM child | `T_tool_complete` (tool output becomes child input) |
| Fan-in (multiple parents) | `max(T_complete for all parents)` |
| Loop iteration boundary | Next iteration starts at last loop step's completion time |

Deferred injection means intra-session step timing depends on actual completion times in the simulator. A new `SessionRegistry` component tracks active sessions and injects child requests when parent steps complete.

```
ReAct session (3 iterations):

Session arrival: t=1000

Iter 1:  reason[t=1000]──complete[t=1200]──►tool[lat=500]──►observe[t=1700]──complete[t=1850]
Iter 2:  reason[t=1850]──complete[t=2100]──►tool[lat=300]──►observe[t=2400]──complete[t=2600]
Iter 3:  reason[t=2600]──complete[t=2900]──►tool[lat=450]──►observe[t=3350]──complete[t=3500]
         final-answer[t=3500]──complete[t=3800]


MCTS session (fan-out 4):

Session arrival: t=2000

decompose[t=2000]──complete[t=2300]──►┬── candidate_0[t=2300]──complete[t=2700]──┐
                                      ├── candidate_1[t=2300]──complete[t=2600]──┤
                                      ├── candidate_2[t=2300]──complete[t=2800]──┤ fan-in
                                      └── candidate_3[t=2300]──complete[t=2750]──┘
                                                                   evaluate[t=2800]──complete[t=2900]
                                                                   run-tests[t=2900,lat=1000]
                                                                   refine[t=3900]──complete[t=4200]
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
- **Fan-out:** `fan_out: N` spawns N parallel instances, or multiple steps sharing the same `depends_on`
- **Fan-in:** one step with `depends_on: [A, B, C]` — waits for all parents
- **Loop:** `loop.over` names repeating steps; steps outside the loop run after loop exits

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

## Runtime Architecture

### SessionRegistry

New component alongside WaitQueue, KVStore, etc:

```go
type SessionRegistry struct {
    sessions         map[string]*AgenticSession
    injectionCounter int64              // total requests injected (for INV-1)
    rng              *rand.Rand         // seeded, partitioned
}

type AgenticSession struct {
    WorkflowDAG    *WorkflowDAG
    CompletedSteps map[string]int64    // step_id+iteration → completion_time
    PendingFanIn   map[string]int      // step_id → remaining parent count
    LoopIteration  int
    State          SessionState        // active | completed
}
```

### Deferred injection event flow

```
Request completes (existing RequestLeftEvent)
       │
       ▼
SessionRegistry.OnStepComplete(sessionID, stepID, completionTime)
       │
       ├── Look up children in WorkflowDAG
       ├── For each child:
       │     ├── tool_call → arrival = completionTime + sample(tool.latency)
       │     ├── llm_call  → arrival = completionTime
       │     ├── fan-in    → decrement PendingFanIn; inject when count == 0
       │     │                arrival = max(all parent completion times)
       │     └── loop boundary → if iteration < max: restart loop body
       │                         else: inject post-loop steps
       ├── Generate child Request(s) with sampled input/output tokens
       ├── Increment injectionCounter
       └── Return []*Request → injected as ArrivalEvents into event queue
```

New fields on `Request`: `StepID` (string) and `LoopIteration` (int). `SessionID` already exists.

## Key Invariants

- **INV-A1 (DAG causality):** No child step arrives before all its parents complete
- **INV-A2 (Fan-in completeness):** Fan-in step injected exactly once, after ALL parents complete
- **INV-A3 (Loop termination):** Every loop terminates at or before `max_iterations`
- **INV-A4 (Session completeness):** Every session either completes all terminal steps or is interrupted by horizon
- **INV-A5 (Tool latency non-negative):** Sampled tool latencies are always >= 0
- **INV-1 (Request conservation, extended):** `injectionCounter == completed + queued + running + dropped`
- **INV-6 (Determinism):** Tool latencies sampled from seeded, partitioned RNG — same seed, same results

## Agentic Metrics

| Metric | Description |
|---|---|
| Session E2E latency | Root step arrival → terminal step completion |
| Steps per session | Actual count (loops may exit early at horizon) |
| Tool wait time | Total time spent in tool calls per session |
| Fan-out utilization | Candidates evaluated before horizon (MCTS) |
| Loop iterations | Distribution of actual iteration counts |
| Critical path latency | Longest path through DAG |

## Validation Rules

1. DAG must be acyclic (ignoring loop-back edges)
2. Exactly one root step (no `depends_on`) unless loop body has implicit entry
3. Every `tool` reference must exist in the `tools` section
4. `fan_out` must be >= 2 when specified
5. `loop.over` steps must form a connected subgraph
6. `llm_call` steps require `input_distribution` and `output_distribution`
7. `tool_call` steps must not have distributions (latency comes from `tools` section)
8. `context_growth: accumulate` only valid inside `loop.over`
9. `agentic` is mutually exclusive with `reasoning` and `multimodal` on `ClientSpec`

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

## Out of Scope (Future Phases)

- Nested sub-workflows (hierarchical agents)
- Conditional / probabilistic edges
- Runtime-dynamic branching (LLM output determines next step)
- Multi-model routing within a workflow (different steps targeting different models)
- `blis convert` for external agentic trace formats

## Backward Compatibility

- `agentic` field on `ClientSpec` is optional (`omitempty`) — no existing YAML breaks
- Category `"agentic"` is additive to the valid category set
- Non-agentic requests bypass `SessionRegistry` entirely (zero overhead)
- All existing workload specs (language, multimodal, reasoning) are unaffected

## References

- #420 — Phase 0: Workload Unification (this feature extends WorkloadSpec v2 from that effort)
- Design doc: [`docs/plans/2026-02-26-agentic-workload-generation-design.md`](docs/plans/2026-02-26-agentic-workload-generation-design.md)
