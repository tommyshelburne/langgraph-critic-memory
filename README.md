# langgraph-critic-memory

[![CI](https://github.com/tommyshelburne/langgraph-critic-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/tommyshelburne/langgraph-critic-memory/actions/workflows/ci.yml)

A small, runnable LangGraph agent that pairs **long-term `Store` memory** with an
**adversarial critic gate** and a bounded **iterate loop**. It's a deliberately
compact demonstration of the production-reliability patterns the agent literature
keeps converging on — and a port of the patterns from a hand-rolled multi-agent
fleet ("OpenClaw") into the idiomatic LangGraph abstractions.

```
   START → recall → generate → critic ─┬─→ remember → END     (ACKNOWLEDGED)
                       ▲                │
                       └─── generate ◀──┤                      (ITERATE, under cap)
                                        │
                                        └─→ escalate → END     (REJECTED / cap hit)
```

## Why this shape

| Node | Pattern | Grounding in the research |
|---|---|---|
| `recall` | Retrieve cross-session memory, inject as grounding | Memory taxonomy: episodic→semantic recall (Park et al., *Generative Agents*, UIST 2023) |
| `generate` | Produce/revise conditioned on memory + prior critique | plan-then-execute / Reflexion-style revision (Shinn et al., NeurIPS 2023) |
| `critic` | Adversarial gate → `ACKNOWLEDGED / ITERATE / REJECTED` | Verification-at-handoffs beats prompt tweaks (Berkeley **MAST**, NeurIPS 2025; *AgentAsk*, 2025) |
| iterate cap | Bound the refine loop (≤ `max_iterations`) | Avoids the error-cascade / non-termination failure modes (MAST FC3) |
| `remember` | Consolidate the accepted outcome back to the Store | Episodic write-summarize-recall (cheapest semantic-memory layer that pays off) |
| `escalate` | Dead-end → optional human-in-the-loop `interrupt()` | Human-in-the-loop as the closing move; LangGraph-native HITL |

**Two memory tiers, on purpose.** The graph compiles with both a *checkpointer*
(per-thread short-term state) and a *Store* (cross-thread long-term memory). The
demo runs two tasks on different `thread_id`s but the same `user_id`: the second
run recalls what the first wrote — because the Store is cross-thread and the
checkpointer is not.

**Semantic search is opt-in.** Matching the LangGraph design (and a hard-won
result from the fleet: naive embedding RAG surfaces *semantically-similar-but-wrong*
"hard distractors" and underperforms on precise tasks), the Store's vector index
is a deliberate add-on, not a default — see `app/memory.py`.

## Run it

Runs offline out of the box (deterministic stub model + toy hash embedder — no API key):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m app.main      # or: critic-memory-demo
pytest -q               # smoke tests: end-to-end, the iterate loop, cross-thread memory
```

Expected: run 1 loops once (ITERATE → revised → ACKNOWLEDGED) and writes a memory;
run 2 shows that memory under `recalled`.

## Use a real model

Set one env var (see `.env.example`) — node code is unchanged:

```bash
pip install -e ".[anthropic]"          # or ".[openai]"
export CHAT_MODEL=anthropic:claude-sonnet-4-6
export EMBEDDINGS_MODEL=openai:text-embedding-3-small   # optional, for real semantic recall
export ANTHROPIC_API_KEY=...
python -m app.main
```

## Production swaps (same interfaces)

- **Durability:** `InMemorySaver`/`InMemoryStore` → `PostgresSaver`/`PostgresStore`. For
  *exactly-once* crash recovery (no agent framework provides it natively), drive the
  graph under a durable-execution backend (e.g. Temporal).
- **Observability:** add LangSmith tracing, or emit OpenTelemetry GenAI spans.
- **Memory consolidation:** replace `remember`'s raw write with a reflection/summarize step.

## Layout

```
app/state.py        State (TypedDict) + Critique (structured-output schema)
app/llm.py          get_model()      — real chat model or offline stub
app/embeddings.py   get_embeddings() — real embeddings or offline hash embedder
app/memory.py       build_store()    — Store with opt-in semantic index
app/nodes.py        recall / generate / critic / remember / escalate + router
app/graph.py        StateGraph wiring (checkpointer + store)
app/main.py         CLI: two-thread cross-session-memory demo
tests/test_graph.py smoke tests
```
