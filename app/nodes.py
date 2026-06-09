"""Graph nodes.

recall   — pull relevant long-term memories from the Store and inject them (grounding)
generate — produce/revise a draft, conditioned on memory + any prior critique
critic   — adversarial gate returning ACKNOWLEDGED / ITERATE / REJECTED  (the Hermes analog)
remember — consolidate the accepted outcome back into the Store (episodic write)
escalate — rejected or iterate-cap exhausted → optional human-in-the-loop interrupt
"""

from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import interrupt

from app.llm import get_model
from app.state import Critique, State


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


def remember(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Episodic consolidation: write the accepted outcome to long-term memory.

    A production version would summarize/reflect (Generative-Agents style) rather
    than store the raw draft; kept simple here.
    """
    summary = f"Task: {state['task']} -> {state['draft']}"
    store.put(
        _namespace(config),
        str(uuid4()),
        {"text": summary, "task": state["task"], "result": state["draft"]},
    )
    return {"result": state["draft"]}


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
