"""Graph nodes.

recall   — pull relevant long-term memories from the Store and inject them (grounding)
generate — produce/revise a draft, conditioned on memory + any prior critique
critic   — adversarial gate returning ACKNOWLEDGED / ITERATE / REJECTED  (the Hermes analog)
remember — write the outcome as an episodic memory; periodically reflect recent
           episodes into an abstracted semantic insight (Generative-Agents style)
escalate — rejected or iterate-cap exhausted → optional human-in-the-loop interrupt
"""

from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import interrupt

from app.llm import get_model
from app.state import Critique, Importance, State

# Accrued importance that triggers a consolidation. Park et al. use 150 on a
# 1-10 importance scale; kept small here so the demo reflects after ~2 episodes.
REFLECTION_THRESHOLD = 10


def _namespace(config: RunnableConfig) -> tuple[str, str]:
    user = config["configurable"].get("user_id", "default")
    return (user, "memories")


def recall(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    hits = store.search(_namespace(config), query=state["task"], limit=5)
    return {"recalled": [item.value.get("text", "") for item in hits]}


def generate(state: State) -> dict:
    model = get_model()
    iters = state.get("iterations", 0)
    memory = "\n".join(f"- {m}" for m in state.get("recalled", [])) or "(none)"

    if iters >= 1 and state.get("critique"):
        instruction = f"Revise the previous draft. Address these critic asks:\n{state['critique']}"
    else:
        instruction = "Produce a first draft."

    prompt = (
        f"You are a generator. {instruction}\n\n"
        f"Task: {state['task']}\n\n"
        f"Relevant long-term memory (use if helpful):\n{memory}\n"
    )
    response = model.invoke(prompt)
    return {"draft": response.content, "iterations": iters + 1}


def critic(state: State) -> dict:
    model = get_model()
    prompt = (
        "You are an adversarial critic (the gate before any output ships). "
        "Review the draft strictly against the task and return a verdict.\n"
        "- ACKNOWLEDGED: good enough to ship.\n"
        "- ITERATE: fixable — list concrete asks.\n"
        "- REJECTED: the premise is wrong; no revision will save it.\n\n"
        f"Task: {state['task']}\n\n"
        f"Draft:\n{state.get('draft', '')}\n"
    )
    verdict: Critique = model.with_structured_output(Critique).invoke(prompt)
    feedback = "; ".join(verdict.asks) if verdict.asks else verdict.reasons
    return {"verdict": verdict.verdict, "critique": feedback}


def route_after_critic(state: State) -> str:
    """Conditional edge: accept, loop, or escalate (with an iterate cap)."""
    verdict = state.get("verdict")
    if verdict == "ACKNOWLEDGED":
        return "remember"
    if verdict == "ITERATE" and state.get("iterations", 0) < state.get("max_iterations", 2):
        return "generate"
    return "escalate"  # REJECTED, or ITERATE past the cap


def _rate_importance(task: str, draft: str) -> int:
    """LLM-rated salience (1-10) of an outcome — the reflection trigger signal."""
    prompt = (
        "Rate how important this outcome is to remember long-term, "
        "1 (mundane) to 10 (pivotal).\n\n"
        f"Task: {task}\nOutcome: {draft}"
    )
    try:
        score = get_model().with_structured_output(Importance).invoke(prompt).score
        return max(1, min(10, int(score)))
    except Exception:
        return 5


def _reflect(store: BaseStore, namespace: tuple[str, str], episodic_keys: list[str]) -> str:
    """Episodic -> semantic consolidation (Generative-Agents 'reflection').

    Pull the recent episodes, synthesize ONE higher-level insight, and store it
    back as a distinct `reflection` memory that cites the episodes it drew on.
    """
    episodes = []
    for key in episodic_keys:
        item = store.get(namespace, key)
        if item:
            episodes.append(item.value.get("text", ""))
    joined = "\n".join(f"- {e}" for e in episodes) or "(none)"

    prompt = (
        "You are consolidating memory. From these recent episodes, extract ONE "
        "higher-level insight — a reusable, de-contextualized takeaway, not a restatement.\n\n"
        f"Recent episodes:\n{joined}\n"
    )
    insight = get_model().invoke(prompt).content
    store.put(
        namespace,
        str(uuid4()),
        {"kind": "reflection", "text": insight, "sources": episodic_keys},
    )
    return insight


def remember(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Write the accepted outcome as an episodic memory, then consolidate.

    Generative-Agents style: every outcome is logged as a raw `episodic` memory
    with an importance score; once accrued importance crosses a threshold, a
    `reflection` step abstracts the recent episodes into a semantic insight and
    resets the accumulator. Raw episodes are kept; reflections coexist beside them.
    """
    namespace = _namespace(config)

    importance = _rate_importance(state["task"], state["draft"])
    episodic_key = str(uuid4())
    store.put(
        namespace,
        episodic_key,
        {
            "kind": "episodic",
            "text": f"Task: {state['task']} -> {state['draft']}",
            "task": state["task"],
            "result": state["draft"],
            "importance": importance,
        },
    )

    # The reflection accumulator must persist across sessions, so it lives in the
    # cross-thread Store (a separate 'meta' namespace), not in per-thread state.
    meta_ns = (namespace[0], "meta")
    bucket = store.get(meta_ns, "reflect")
    pending = (bucket.value["pending"] if bucket else []) + [episodic_key]
    accrued = (bucket.value["accrued"] if bucket else 0) + importance

    out: dict = {"result": state["draft"], "importance": importance}
    if accrued >= REFLECTION_THRESHOLD:
        out["reflection"] = _reflect(store, namespace, pending)
        pending, accrued = [], 0  # reset the trigger after consolidating

    store.put(meta_ns, "reflect", {"pending": pending, "accrued": accrued}, index=False)
    return out


def escalate(state: State, config: RunnableConfig) -> dict:
    """Dead-end gate: surface to a human. Mirrors OpenClaw's escalation nudge.

    HITL is opt-in via config (`human_in_the_loop: true`) so the default run
    doesn't block. When on, `interrupt()` pauses the graph until resumed with
    `Command(resume=<decision>)` — LangGraph's native human-in-the-loop.
    """
    reason = (
        f"verdict={state.get('verdict')} after {state.get('iterations')} iteration(s): "
        f"{state.get('critique')}"
    )
    if config["configurable"].get("human_in_the_loop"):
        decision = interrupt({"reason": reason, "draft": state.get("draft")})
        return {"escalation": reason, "result": f"human-decided: {decision}"}
    return {"escalation": reason, "result": "[ESCALATED — no automatic result]"}
