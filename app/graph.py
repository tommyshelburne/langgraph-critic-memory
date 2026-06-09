"""Graph wiring.

    START -> recall -> generate -> critic -+-> remember -> END   (ACKNOWLEDGED)
                          ^                 |
                          +---- generate <--+                    (ITERATE, under cap)
                                            |
                                            +-> escalate -> END  (REJECTED / cap hit)

Compiled with a checkpointer (per-thread short-term state) AND a store
(cross-thread long-term memory) — the two memory tiers, side by side.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from app.nodes import critic, escalate, generate, recall, remember, route_after_critic
from app.state import State


def build_graph(store: BaseStore):
    builder = StateGraph(State)

    builder.add_node("recall", recall)
    builder.add_node("generate", generate)
    builder.add_node("critic", critic)
    builder.add_node("remember", remember)
    builder.add_node("escalate", escalate)

    builder.add_edge(START, "recall")
    builder.add_edge("recall", "generate")
    builder.add_edge("generate", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"generate": "generate", "remember": "remember", "escalate": "escalate"},
    )
    builder.add_edge("remember", END)
    builder.add_edge("escalate", END)

    return builder.compile(checkpointer=InMemorySaver(), store=store)
