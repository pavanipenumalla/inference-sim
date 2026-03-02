# Agentix Trace Converter for Agentic Workload Generation

**Date:** 2026-03-02
**Status:** Proposed
**Species:** Specification
**Depends on:** [Agentic Workload Generation Design](2026-02-26-agentic-workload-generation-design.md) (phases A0-1, A0-2)

## Problem

BLIS has a design for agentic workload generation via YAML-defined `AgenticSpec` workflows (DAG topology, deferred injection, tool latencies). However, this requires users to manually author workflow DAGs and estimate token distributions — a labor-intensive process that may not capture the true structure and token characteristics of real agentic systems.

Real agentic traces exist (e.g., from the Agentix framework's LATS agent running HotPotQA). These traces are JSON call trees with exact per-call token counts and full parent-child structure. Bridging these traces into BLIS would provide:

1. **Empirical validation**: Run BLIS with real-world DAG topologies and token counts
2. **Zero manual authoring**: Users point at traces, get a workload spec
3. **Fidelity**: Per-program replay with exact token counts, not estimated distributions

## Trace Format Analysis

### Agentix JSON Schema

Each program is a JSON object:

```json
{
  "program_id": "201",
  "total_calls": 30,
  "total_decode_tokens": 846,
  "call_tree": { ... }
}
```

The `call_tree` is a recursive tree of two node types:

| Field | LLMCall | ToolCall |
|---|---|---|
| `type` | `"LLMCall"` | `"ToolCall"` |
| `id` | UUID string | integer or string |
| `prefill_tokens` | incremental new tokens | — |
| `decode_tokens` | output tokens generated | — |
| `total_prefill_tokens` | full prompt size (including context) | — |
| `tool` | — | tool name string |
| `input` | raw prompt text | tool input |
| `output` | LLM response text | tool output |
| `children` | child nodes | child nodes |

### Token Semantics

- **`total_prefill_tokens`** → BLIS `input_tokens`: the full context window size the GPU must process (KV cache allocation, prefill latency)
- **`decode_tokens`** → BLIS `output_tokens`: autoregressive tokens generated
- **`prefill_tokens`** (incremental new tokens): not used directly, but the difference `total_prefill_tokens - prefill_tokens` represents cacheable prefix tokens

### Tool Classification

Traces contain two categories of `ToolCall` nodes:

**Orchestration tools** (framework infrastructure, no real latency):
- `lats_search` — LATS root orchestrator
- `EvalNodeMerge` — evaluation aggregation
- `RolloutNewState` — tree expansion
- `Rollout` — rollout execution
- `LatsSearchMerge` — search result aggregation

**Real tools** (external service calls with actual latency):
- `wiki_env` — Wikipedia search API
- Any other tool not in the orchestration list

### Observed Dataset (hotpotlats_raw.json)

- 374 programs (LATS agent solving HotPotQA questions)
- 24–814 total calls per program (mean: 287)
- 578–35,993 total decode tokens per program (mean: 11,601)
- Max tree depth: 256
- Consistent structure: root `lats_search` → fan-out 10 LLM calls → sequential LLM→tool→LLM chains → deeper rollouts with fan-out 5
- All roots are `ToolCall(lats_search)` with exactly 10 children

## Solution Overview

Build a `blis convert agentix` command that transforms Agentix JSON call trees into `AgenticSpec`-compliant `WorkloadSpec` YAML. The converter is a pure JSON→YAML transformation — no changes to BLIS core formats or data structures.

### Why a converter (not a new trace format)?

Three approaches were evaluated:

| Approach | Description | Verdict |
|---|---|---|
| **A: Converter to YAML** | JSON → AgenticSpec YAML | **Chosen.** Zero core changes, reuses existing AgenticSpec design |
| B: Extend Trace v2 CSV | Add dependency columns to CSV + deferred injection in replay | More invasive; traces lack timestamps anyway |
| C: Hybrid | Both A and B | Double implementation surface for marginal benefit |

Approach A was chosen because:
1. The AgenticSpec design already handles DAG topology, deferred injection, fan-out, and tool latencies
2. The traces lack timing information, so "exact replay" via CSV is misleading — BLIS reconstructs timing regardless
3. Per-program YAML with `constant` distributions preserves exact token counts
4. Zero changes to BLIS core — the converter is an isolated addition

## Conversion Pipeline

### Step 1: Parse JSON Call Tree

Read the Agentix JSON and build an in-memory tree of `LLMCall` and `ToolCall` nodes.

### Step 2: Strip Orchestration Tools

For each `ToolCall` node whose `tool` name is in the orchestration list:
1. Re-parent its children to its own parent
2. Remove the orchestration node from the tree
3. Preserve the edge semantics (children become siblings of the removed node's siblings, maintaining order)

**Before flattening** (program_201):
```
ToolCall(lats_search) → 10× LLMCall
  LLMCall → ToolCall(wiki_env) → LLMCall
    → ToolCall(EvalNodeMerge) → ToolCall(RolloutNewState) → 5× LLMCall
      each LLMCall → ToolCall(wiki_env) → ...
```

**After flattening:**
```
10× LLMCall (root, fan-out 10)
  LLMCall → ToolCall(wiki_env) → LLMCall
    → 5× LLMCall (fan-out 5)
      each LLMCall → ToolCall(wiki_env) → ...
```

### Step 3: Build DAG with Step IDs

Assign deterministic step IDs to each node in the flattened tree:
- LLM calls: `llm-{depth}-{index}` (e.g., `llm-1-0`, `llm-3-2`)
- Tool calls: `{tool_name}-{depth}-{index}` (e.g., `wiki-env-2-0`)

Build `depends_on` edges from parent-child relationships.

### Step 4: Detect Fan-out

When a parent has N children of the same type with identical (or near-identical) token counts:
- Collapse into a single step with `fan_out: N`
- Use the token counts from any representative child (they're identical)

When children differ in token counts or type, they remain as separate named steps.

### Step 5: Emit AgenticSpec YAML

Produce a `WorkloadSpec` YAML with:
- One `ClientSpec` per program
- `agentic` block with the flattened DAG as `steps`
- Token counts as `constant` distributions (exact values)
- Tool definitions from the user-supplied tool config

### Per-Program Output Example

For program_201 (30 calls, 16 LLM calls, 14 tool calls):

```yaml
version: 2
horizon_seconds: 300
clients:
  - id: "program-201"
    tenant_id: "agentix"
    slo_class: "standard"
    rate_fraction: 1.0
    arrival:
      process: constant
      params:
        interval_us: 1000000
    agentic:
      workflow: "lats-program-201"
      steps:
        - id: "llm-1"
          type: llm_call
          fan_out: 10
          input_distribution: {type: constant, params: {value: 1588}}
          output_distribution: {type: constant, params: {value: 54}}
        - id: "wiki-env-1"
          type: tool_call
          tool: "wiki_env"
          depends_on: ["llm-1"]
        - id: "llm-2"
          type: llm_call
          depends_on: ["wiki-env-1"]
          input_distribution: {type: constant, params: {value: 2218}}
          output_distribution: {type: constant, params: {value: 61}}
        - id: "llm-3"
          type: llm_call
          depends_on: ["llm-2"]
          fan_out: 5
          input_distribution: {type: constant, params: {value: 1835}}
          output_distribution: {type: constant, params: {value: 49}}
        - id: "wiki-env-2"
          type: tool_call
          tool: "wiki_env"
          depends_on: ["llm-3"]
        # ... remaining steps
      tools:
        wiki_env:
          latency: {type: exponential, params: {mean: 50000}}
          output_tokens: {type: gaussian, params: {mean: 200, std_dev: 50, min: 10, max: 1000}}
```

## CLI Interface

```bash
# Convert a single program
blis convert agentix \
  --input program_201.json \
  --tool-config tools.yaml \
  --output workload-201.yaml

# Convert all programs in a batch file (one YAML per program)
blis convert agentix \
  --input hotpotlats_raw.json \
  --tool-config tools.yaml \
  --output-dir converted/ \
  --batch

# Combine all programs into a single multi-client YAML
blis convert agentix \
  --input hotpotlats_raw.json \
  --tool-config tools.yaml \
  --output combined-workload.yaml \
  --merge
```

### Tool Latency Config

User-supplied YAML specifying latency distributions for real tools:

```yaml
tools:
  wiki_env:
    latency: {type: exponential, params: {mean: 50000}}
    output_tokens: {type: gaussian, params: {mean: 200, std_dev: 50, min: 10, max: 1000}}

# Optional: override the default orchestration tool list
orchestration_tools:
  - EvalNodeMerge
  - RolloutNewState
  - Rollout
  - LatsSearchMerge
  - lats_search
```

**Default orchestration tools** are built-in (`EvalNodeMerge`, `RolloutNewState`, `Rollout`, `LatsSearchMerge`, `lats_search`). The `orchestration_tools` key in the tool config overrides this list entirely.

**Validation rules:**
1. Every real tool found in traces must have a matching entry in the tool config
2. Orchestration tools must NOT appear in the `tools` section
3. `latency` and `output_tokens` are required for each real tool

## What Changes to BLIS Core: None

| Aspect | Status |
|---|---|
| Trace v2 CSV format | No changes |
| `sim.Request` struct | No new fields (uses existing `StepID`, `SessionID` from AgenticSpec design) |
| Event loop | No changes (deferred injection from AgenticSpec design) |
| WorkloadSpec | No changes (uses existing `AgenticSpec` schema) |
| CLI | New `convert agentix` subcommand only |

The converter lives in its own package (e.g., `cmd/convert/agentix/`) and depends only on the `AgenticSpec` types from `sim/workload/`.

## Dependencies

| Dependency | Type | Description |
|---|---|---|
| A0-1 (AgenticSpec schema) | Hard (for BLIS consumption) | The converter produces AgenticSpec YAML. A0-1 must land for BLIS to parse the output. |
| A0-2 (SessionRegistry + deferred injection) | Hard (for BLIS simulation) | Required for BLIS to simulate the DAG at runtime. |
| Converter implementation | Independent | Pure JSON→YAML transformation. Can be built and tested without A0-1/A0-2. |

The converter can be developed in parallel with A0-1/A0-2. Its output YAML can be validated structurally before BLIS can run it.

## Edge Cases and Caveats

### Deep Trees

Some programs have tree depth up to 256. After orchestration stripping, the effective depth is reduced (orchestration tools account for ~50% of depth), but the resulting YAML may still have 50+ steps. The converter reports step counts per program so users know what they're getting.

### Variable Tree Topology

Different programs have different tree structures (different depths, different branching at intermediate levels). Each program produces its own unique `AgenticSpec` DAG. This is expected and correct — real agent executions vary.

### Fan-out Detection Heuristic

Siblings are collapsed into `fan_out: N` when they:
1. Are all the same type (`LLMCall` or `ToolCall`)
2. Have identical `total_prefill_tokens` AND `decode_tokens` (for LLM calls)
3. Have identical `tool` name (for tool calls)

If any sibling differs, all siblings become separate named steps. This is conservative — it never merges nodes that might have different behavior.

### Missing Tool Latencies

If a real tool appears in traces but not in the tool config, the converter exits with an error listing all missing tools. It does not silently default to zero latency.

## Future Extensions

- **Other trace formats**: The converter pattern (`blis convert <format>`) extends to other agent frameworks (LangChain traces, AutoGen logs, etc.) by adding new subcommands
- **Distribution mode**: A `--mode=distribution` flag could fit statistical distributions across programs instead of per-program constants (deferred — user currently prefers per-program replay)
- **Prefix cache hints**: The `prefill_tokens` vs `total_prefill_tokens` delta could populate `prefix_group` fields to test KV cache sharing across fan-out siblings (they share the same prompt prefix)

## Validation Checklist

1. Converter produces valid YAML that passes `WorkloadSpec` schema validation
2. Every LLM call in the trace maps to exactly one `llm_call` step (or part of a `fan_out` group)
3. Every real tool call maps to exactly one `tool_call` step
4. Orchestration tools produce zero steps (fully stripped)
5. `depends_on` edges match the original tree's parent-child relationships (post-flattening)
6. Total LLM call count in output matches total LLM call count in input (accounting for fan-out)
7. Token counts in `constant` distributions match source trace exactly
