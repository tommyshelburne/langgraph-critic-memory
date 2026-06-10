"""Smoke tests — run offline with the stub model/embeddings (no API key)."""

from datetime import datetime, timezone

from langgraph.store.base import SearchItem

from app.graph import build_graph
from app.memory import build_store
from app.nodes import _minmax, recall


def test_graph_runs_end_to_end():
    graph = build_graph(build_store())
    out = graph.invoke(
        {"task": "test task", "iterations": 0, "max_iterations": 2},
        {"configurable": {"thread_id": "t1", "user_id": "u1"}},
    )
    assert out.get("verdict") in {"ACKNOWLEDGED", "ITERATE", "REJECTED"}
    assert out.get("result")


def test_critic_loop_then_accepts():
    # Stub: first draft -> ITERATE -> revised draft -> ACKNOWLEDGED (2 generate passes).
    graph = build_graph(build_store())
    out = graph.invoke(
        {"task": "loop test", "iterations": 0, "max_iterations": 2},
        {"configurable": {"thread_id": "t2", "user_id": "u2"}},
    )
    assert out["verdict"] == "ACKNOWLEDGED"
    assert out["iterations"] == 2


def test_memory_persists_across_threads():
    store = build_store()
    graph = build_graph(store)
    # Write a memory under a shared user on thread 'a'...
    graph.invoke(
        {"task": "remember the alpha protocol", "iterations": 0, "max_iterations": 2},
        {"configurable": {"thread_id": "a", "user_id": "shared"}},
    )
    # ...then recall it from a *different* thread (fresh short-term state).
    out = graph.invoke(
        {"task": "alpha protocol", "iterations": 0, "max_iterations": 2},
        {"configurable": {"thread_id": "b", "user_id": "shared"}},
    )
    assert out.get("recalled"), "expected the Store to surface the memory written on thread 'a'"


def test_reflection_consolidation_after_threshold():
    # Stub importance = 5/outcome; REFLECTION_THRESHOLD = 10 → reflect after 2 episodes.
    store = build_store()
    graph = build_graph(store)
    for i in range(2):
        graph.invoke(
            {"task": f"episode {i}", "iterations": 0, "max_iterations": 2},
            {"configurable": {"thread_id": f"r{i}", "user_id": "refl"}},
        )
    items = store.search(("refl", "memories"), limit=50)
    kinds = [it.value.get("kind") for it in items]
    assert "episodic" in kinds, "raw episodes should be retained"
    assert "reflection" in kinds, "a semantic reflection should have been consolidated"

    reflection = next(it for it in items if it.value.get("kind") == "reflection")
    assert reflection.value.get("sources"), "reflection must cite the episodes it drew on"
    assert reflection.value.get("importance") == 8, (
        "reflections must carry the fixed importance that lets them outrank raw episodes"
    )


class _FakeStore:
    """Minimal stand-in whose search() returns hand-built SearchItems, so the
    ranking math can be tested with fully pinned timestamps and scores."""

    def __init__(self, items):
        self._items = items

    def search(self, namespace, *, query=None, limit=10):
        return self._items[:limit]


def test_recall_ranking_isolates_importance():
    # Identical updated_at (recency exactly equal) and identical relevance scores:
    # importance is the ONLY live signal. An implementation that drops the
    # importance term returns insertion order and fails this test.
    ts = datetime(2026, 6, 10, tzinfo=timezone.utc)
    mk = lambda key, imp: SearchItem(("u", "memories"), key, {"kind": "episodic", "text": key, "importance": imp}, ts, ts, score=0.9)
    store = _FakeStore([mk("imp-1", 1), mk("imp-5", 5), mk("imp-10", 10)])

    out = recall(
        {"task": "anything"},
        {"configurable": {"user_id": "u"}},
        store=store,
    )
    assert out["recalled"] == ["imp-10", "imp-5", "imp-1"]


def test_recall_ranks_importance_over_equal_relevance():
    # Graph-level integration check. Equal token overlap vs the query => equal
    # relevance under the hash embedder. The importance-10 memory is written FIRST,
    # so raw recency (~1e-10 spread at microsecond write gaps) slightly OPPOSES it —
    # it must win on importance alone.
    store = build_store()
    ns = ("rank", "memories")
    store.put(ns, "high", {"kind": "episodic", "text": "alpha protocol details two", "importance": 10})
    store.put(ns, "low", {"kind": "episodic", "text": "alpha protocol details one", "importance": 1})

    graph = build_graph(store)
    out = graph.invoke(
        {"task": "alpha protocol details", "iterations": 0, "max_iterations": 2},
        {"configurable": {"thread_id": "rank-t", "user_id": "rank"}},
    )
    recalled = out["recalled"]
    assert len(recalled) >= 2
    assert recalled[0] == "alpha protocol details two", (
        "the importance-10 memory should outrank the importance-1 memory "
        f"when relevance is equal; got order: {recalled}"
    )


def test_recall_prefixes_reflections_and_caps_at_top_n():
    # Reflections carry a '[reflection] ' prefix; episodic memories don't; and
    # recall injects at most RECALL_TOP_N (5) memories after re-ranking.
    store = build_store()
    ns = ("pfx", "memories")
    store.put(ns, "refl", {"kind": "reflection", "text": "alpha insight", "importance": 8})
    for i in range(7):
        store.put(ns, f"ep{i}", {"kind": "episodic", "text": f"alpha episode {i}", "importance": 5})

    graph = build_graph(store)
    out = graph.invoke(
        {"task": "alpha", "iterations": 0, "max_iterations": 2},
        {"configurable": {"thread_id": "pfx-t", "user_id": "pfx"}},
    )
    recalled = out["recalled"]
    assert len(recalled) == 5, f"expected RECALL_TOP_N=5 memories, got {len(recalled)}"
    assert "[reflection] alpha insight" in recalled, (
        f"importance-8 reflection should rank into the top 5 with its prefix; got {recalled}"
    )
    assert all(not r.startswith("[reflection]") for r in recalled if "episode" in r)


def test_recall_single_candidate_does_not_crash():
    # Degenerate min-max ranges (one candidate -> constant signals) must not div/0.
    store = build_store()
    store.put(("solo", "memories"), "only", {"kind": "episodic", "text": "lone memory"})
    graph = build_graph(store)
    out = graph.invoke(
        {"task": "lone memory", "iterations": 0, "max_iterations": 2},
        {"configurable": {"thread_id": "solo-t", "user_id": "solo"}},
    )
    assert out["recalled"] == ["lone memory"]


def test_minmax_degenerate_and_normal():
    assert _minmax([3.0, 3.0, 3.0]) == [0.5, 0.5, 0.5]
    assert _minmax([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]
