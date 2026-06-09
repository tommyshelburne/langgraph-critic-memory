"""CLI entrypoint.

Runs two tasks on DIFFERENT threads but the SAME user. Run 1 writes a memory;
Run 2 (fresh per-thread state) recalls it from the Store — demonstrating that
long-term memory is cross-thread while short-term checkpointer state is not.
"""

from app.graph import build_graph
from app.memory import build_store


def _run(graph, task: str, thread: str, user: str = "tommy") -> None:
    config = {"configurable": {"thread_id": thread, "user_id": user}}
    out = graph.invoke({"task": task, "iterations": 0, "max_iterations": 2}, config)
    print(f"\n=== thread={thread!r} user={user!r}")
    print(f"  task      : {task}")
    print(f"  recalled  : {out.get('recalled')}")
    print(f"  iterations: {out.get('iterations')}   verdict: {out.get('verdict')}")
    print(f"  result    : {out.get('result')}")
    if out.get("escalation"):
        print(f"  escalation: {out['escalation']}")


def main() -> None:
    store = build_store()
    graph = build_graph(store)

    _run(graph, "Draft a one-line summary of the LangGraph critic-memory demo.", "demo-1")
    _run(graph, "Recall the earlier summary and extend it with the memory angle.", "demo-2")


if __name__ == "__main__":
    main()
