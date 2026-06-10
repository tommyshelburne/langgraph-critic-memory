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
| `recall` | **Multi-signal retrieval**: similarity nominates candidates, ranking is recency + importance + relevance (min-max normalized, equal weights) | Embedding similarity is only 1/3 of the canonical retrieval score (Park et al., *Generative Agents*, UIST 2023); similarity-only RAG surfaces "hard distractors" (ACL 2025) |
| `generate` | Produce/revise conditioned on memory + prior critique | plan-then-execute / Reflexion-style revision (Shinn et al., NeurIPS 2023) |
| `critic` | Adversarial gate → `ACKNOWLEDGED / ITERATE / REJECTED` | Verification-at-handoffs beats prompt tweaks (Berkeley **MAST**, NeurIPS 2025; *AgentAsk*, 2025) |
| iterate cap | Bound the refine loop (≤ `max_iterations`) | Avoids the error-cascade / non-termination failure modes (MAST FC3) |
| `remember` | Log the outcome as an episodic memory, then periodically **reflect** recent episodes into a semantic insight | Episodic→semantic consolidation (Park et al., *Generative Agents*, UIST 2023) |
| `escalate` | Dead-end → optional human-in-the-loop `interrupt()` | Human-in-the-loop as the closing move; LangGraph-native HITL |

**Two memory tiers, on purpose.** The graph compiles with both a *checkpointer*
(per-thread short-term state) and a *Store* (cross-thread long-term memory). The
demo runs two tasks on different `thread_id`s but the same `user_id`: the second
run recalls what the first wrote — because the Store is cross-thread and the
checkpointer is not.

**Similarity nominates; it doesn't decide.** In LangGraph the Store's vector index
is opt-in, and this repo opts in unconditionally (`app/memory.py`) because `recall`
rides on it twice: `store.search(query=…)` nominates the candidate pool and
`hit.score` is the relevance signal. But naive embedding RAG surfaces
*semantically-similar-but-wrong* "hard distractors" — so the defense here is
**re-ranking** (recency + importance + relevance), not gating the index: similarity
alone never determines what gets injected.

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
- **Memory consolidation & retrieval:** `remember` does episodic→semantic reflection (importance-scored, threshold-triggered) and `recall` ranks by recency + importance + relevance. Two deliberate deviations from Park et al.: recency decays on last *write* (`updated_at`), not last *access* — recall stays a pure read (no write-on-read, no re-embedding cost per recall) — and recency is used *raw* (0.995^hours is already in (0,1]) rather than pool-min-maxed, which in a same-session pool would stretch microsecond write-order deltas to full ranking weight and swamp importance. Next steps: tune `REFLECTION_THRESHOLD` / the α weights, or LLM-rate reflection importance instead of the fixed 8.

## Layout

```
app/state.py        State (TypedDict) + Critique / Importance (structured-output schemas)
app/llm.py          get_model()      — real chat model or offline stub
app/embeddings.py   get_embeddings() — real embeddings or offline hash embedder
app/memory.py       build_store()    — Store with opt-in semantic index
app/nodes.py        recall / generate / critic / remember / escalate + router
app/graph.py        StateGraph wiring (checkpointer + store)
app/main.py         CLI: two-thread cross-session-memory demo
tests/test_graph.py smoke tests
```
