# Agentic Workload Generation

**Date:** 2026-02-26
**Status:** Proposed
**Species:** Specification

## Problem

BLIS currently supports language, multimodal, and reasoning workload categories. All of these model independent requests or sequential multi-turn sessions (linear chain of rounds). Real agentic inference workloads have fundamentally different structure:

- **ReAct agents**: LLM reasons, calls a tool, observes the result, loops until done. Sequential cycle with variable iterations and tool call latency.
- **MCTS / Tree-of-Thought**: LLM generates N candidates in parallel, evaluates them, expands the best. Tree-shaped DAG with fan-out and fan-in.
- **Parallel tool calls (fork-join)**: LLM plans, dispatches multiple tools concurrently, aggregates results. Fork-join parallelism.
- **Multi-step chains**: LLM output triggers a sequence of different tools, each feeding the next LLM call. Linear DAG with heterogeneous tool latencies.

None of these patterns can be expressed in the current `WorkloadSpec` because there is no concept of:
1. Parent-child dependencies between requests
2. Tool call latency (external service delay between LLM calls)
3. Fan-out (one step spawning multiple parallel child steps)
4. Fan-in (one step waiting for multiple parents to complete)
5. Loops with iteration limits

## Solution Overview

Extend `WorkloadSpec` v2 with an optional `AgenticSpec` on `ClientSpec` that defines workflow DAGs. Three structural primitives — `depends_on`, `fan_out`, and `loop` — compose to express all four agentic patterns.

Two-layer arrival time model:
- **Layer 1 (session arrivals):** Controlled by existing `ArrivalSpec` on `ClientSpec`. Pre-generated. Determines when each agentic workflow session starts.
- **Layer 2 (intra-session step arrivals):** Deferred injection at runtime. When a step completes, the simulator injects child steps into the event queue based on DAG dependencies and sampled tool latencies.

## YAML Schema

### AgenticSpec on ClientSpec

```yaml
clients:
  - id: "react-agent"
    tenant_id: "team-search"
    slo_class: "standard"
    rate_fraction: 0.5
    arrival:
      process: poisson            # Layer 1: session inter-arrival times
    agentic:                       # NEW: optional, mutually exclusive with reasoning/multimodal
      workflow: "react-search"     # human-readable workflow name
      loop:                        # optional: defines which steps repeat
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
          context_growth: accumulate   # input grows with prior iterations
          input_distribution: {type: gaussian, params: {mean: 200, std_dev: 50, min: 32, max: 2048}}
          output_distribution: {type: exponential, params: {mean: 64}}
        - id: "final-answer"
          type: llm_call
          depends_on: ["observe"]      # runs after loop exits
          input_distribution: {type: gaussian, params: {mean: 512, std_dev: 100, min: 64, max: 2048}}
          output_distribution: {type: exponential, params: {mean: 256}}
      tools:
        web_search:
          latency: {type: exponential, params: {mean: 50000}}
          output_tokens: {type: gaussian, params: {mean: 200, std_dev: 50, min: 10, max: 1000}}
```

### Step types

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Unique step identifier within the workflow |
| `type` | string | yes | `llm_call` or `tool_call` |
| `depends_on` | []string | no | Parent step IDs. Step waits for all parents to complete. Empty = root step. |
| `fan_out` | int | no | Spawn N parallel instances of this step. Default 1. |
| `tool` | string | conditional | Tool name (required when `type: tool_call`, references `tools` section) |
| `context_growth` | string | no | `accumulate` to carry forward prior iteration outputs. Only meaningful in loops. |
| `input_distribution` | DistSpec | conditional | Required for `llm_call` steps |
| `output_distribution` | DistSpec | conditional | Required for `llm_call` steps |

### Tool definitions

| Field | Type | Required | Description |
|---|---|---|---|
| `latency` | DistSpec | yes | Latency distribution in microseconds |
| `output_tokens` | DistSpec | yes | Token count distribution for tool output (becomes input to child LLM call) |

### Loop configuration

| Field | Type | Required | Description |
|---|---|---|---|
| `over` | []string | yes | Step IDs that form the loop body (must be a connected subgraph) |
| `max_iterations` | int | yes | Upper bound on iterations. Must be >= 1. |

### Structural primitives

- **Sequential:** Linear `depends_on` chain (A → B → C)
- **Fan-out:** Multiple steps with the same `depends_on`, or `fan_out: N` for N identical parallel instances
- **Fan-in:** One step with `depends_on: [A, B, C]` — waits for all parents
- **Loop:** `loop.over` names the steps that repeat; steps outside the loop run after loop exits

### Full examples

#### MCTS / Tree-of-Thought

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
      depends_on: ["generate"]       # fan-in: waits for all 4
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

#### Parallel tool calls (fork-join)

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

## Arrival Time Model

### Layer 1 — Session arrivals (pre-generated)

Controlled by `ClientSpec.Arrival` (existing `ArrivalSpec`). Poisson, gamma, weibull, or constant inter-arrival times determine when each agentic workflow session starts. Only root steps are pre-generated as `*sim.Request` with the session's arrival time.

### Layer 2 — Intra-session step arrivals (deferred injection)

When a step completes at time `T_complete`:

| Edge type | Child arrival time |
|---|---|
| LLM → LLM child | `T_complete` (immediate) |
| LLM → tool_call child | Tool completes at `T_complete + sample(tool.latency)` |
| tool_call → LLM child | `T_complete + sample(tool.latency)` (tool output becomes child input) |
| Fan-in (multiple parents) | `max(T_complete for all parents)` |
| Loop iteration boundary | Next iteration starts at last loop step's completion time |

### Timeline examples

```
ReAct session (3 iterations):

Session arrival (from ArrivalSpec): t=1000

Iter 1:  reason[t=1000]──complete[t=1200]──►tool[t=1200,lat=500]──►observe[t=1700]──complete[t=1850]
Iter 2:  reason[t=1850]──complete[t=2100]──►tool[t=2100,lat=300]──►observe[t=2400]──complete[t=2600]
Iter 3:  reason[t=2600]──complete[t=2900]──►tool[t=2900,lat=450]──►observe[t=3350]──complete[t=3500]
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

## Runtime Architecture

### New fields on Request

```go
StepID        string   // which step in the workflow DAG ("reason", "act", etc.)
LoopIteration int      // current iteration for loop workflows (0-based)
```

`SessionID` (already exists) links requests to their session.

### SessionRegistry

New component alongside WaitQueue, KVStore, etc:

```go
type SessionRegistry struct {
    sessions         map[string]*AgenticSession
    injectionCounter int64
    rng              *rand.Rand  // seeded, partitioned
}

type AgenticSession struct {
    WorkflowDAG    *WorkflowDAG
    CompletedSteps map[string]int64   // step_id+iteration → completion_time
    PendingFanIn   map[string]int     // step_id → remaining parent count
    LoopIteration  int
    State          SessionState       // active | completed
}
```

### Event flow

```
Request completes (existing RequestLeftEvent)
       │
       ▼
SessionRegistry.OnStepComplete(sessionID, stepID, completionTime)
       │
       ├── Look up children in WorkflowDAG
       │
       ├── For each child:
       │     ├── tool_call → child arrival = completionTime + sample(tool.latency)
       │     ├── llm_call  → child arrival = completionTime
       │     ├── fan-in    → decrement PendingFanIn; inject only when count == 0
       │     │                arrival = max(all parent completion times)
       │     └── loop boundary → increment LoopIteration
       │                         if < max_iterations: restart loop body
       │                         else: inject post-loop steps
       │
       ├── Generate child Request(s) with sampled input/output tokens
       ├── Increment injectionCounter
       └── Return []*Request → injected as ArrivalEvents
```

Hook point: `simulator.go` where `RequestLeftEvent` is processed. After a request completes, if it has a `SessionID` and `StepID`, call `SessionRegistry.OnStepComplete()` and schedule returned child requests as `ArrivalEvent`s. No new event types needed.

## Invariants

| ID | Invariant | Verification |
|---|---|---|
| INV-A1 | **DAG causality**: No child step arrives before all parents complete | Assert `child.ArrivalTime >= max(parent.CompletionTime)` |
| INV-A2 | **Fan-in completeness**: Fan-in step injected exactly once after ALL parents complete | `PendingFanIn` reaches 0 exactly once per fan-in step per iteration |
| INV-A3 | **Loop termination**: Every loop terminates at or before `max_iterations` | `LoopIteration <= max_iterations` at all times |
| INV-A4 | **Session completeness**: Every session either completes terminal steps or is interrupted by horizon | No "stuck" sessions with zero pending work at simulation end |
| INV-A5 | **Tool latency non-negative**: Sampled tool latencies >= 0 | Enforced by DistSpec `min` param or clamping |
| INV-1 | **Request conservation** (extended) | `injectionCounter == completed + queued + running + dropped` |
| INV-6 | **Determinism** | Tool latencies sampled from seeded, partitioned RNG. Same seed → same results. |

## Agentic Metrics

| Metric | Description |
|---|---|
| Session E2E latency | Root step arrival → terminal step completion |
| Steps per session | Actual count (loops may hit horizon before `max_iterations`) |
| Tool wait time | Total tool latency per session |
| Fan-out utilization | Candidates evaluated before horizon (MCTS) |
| Loop iterations | Distribution of actual iteration counts |
| Critical path latency | Longest path through DAG |

## Validation Rules

1. DAG must be acyclic (ignoring loop-back edges)
2. Exactly one root step (no `depends_on`) unless loop body has implicit entry
3. Every `tool` reference in steps must exist in `tools` section
4. `fan_out` must be >= 2 when specified
5. `loop.over` steps must form a connected subgraph
6. `llm_call` steps must have `input_distribution` and `output_distribution`
7. `tool_call` steps must not have `input_distribution` or `output_distribution`
8. `context_growth: accumulate` only valid for steps inside `loop.over`
9. `agentic` is mutually exclusive with `reasoning` and `multimodal` on `ClientSpec`

## Phased Delivery

| Phase | Title | Delivers | Dependencies |
|---|---|---|---|
| A0-1 | AgenticSpec schema + YAML loading + validation | `AgenticSpec`, `WorkflowDAG`, `ToolSpec` structs. Step/edge validation (acyclicity, fan-in, loop well-formedness). Strict YAML parsing. | W0-1 |
| A0-2 | SessionRegistry + deferred injection | `SessionRegistry`, `OnStepComplete()`, hook into `RequestLeftEvent`, `injectionCounter`, fan-in tracking, loop iteration. Root-only pre-generation. | A0-1 |
| A0-3 | Agentic metrics + invariant tests | `AgenticMetrics`, INV-A1–A5 tests, INV-1 extension. | A0-2 |
| A0-4 | Built-in scenarios + documentation | `ScenarioReActAgent()`, `ScenarioMCTS()`, `ScenarioForkJoin()` presets. Extension recipe. Example YAMLs. | A0-3 |

## Out of Scope (Future Phases)

- Nested sub-workflows (hierarchical agents)
- Conditional / probabilistic edges
- Runtime-dynamic branching
- Multi-model routing within a workflow
- `blis convert` for external agentic trace formats

## Backward Compatibility

- `agentic` field on `ClientSpec` is optional (`omitempty`)
- Category `"agentic"` is additive to the valid set
- Non-agentic requests bypass `SessionRegistry` (zero overhead)
- Existing workload specs are unaffected
