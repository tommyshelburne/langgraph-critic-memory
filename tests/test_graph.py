"""Smoke tests — run offline with the stub model/embeddings (no API key)."""

from app.graph import build_graph
from app.memory import build_store


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
