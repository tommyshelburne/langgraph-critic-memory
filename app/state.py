"""Graph state and the critic's structured-output schema."""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class State(TypedDict, total=False):
    """Shared graph state. `total=False` so nodes can fill fields incrementally;
    read optional fields with `.get(...)`.
    """

    task: str               # the incoming request
    recalled: list[str]     # long-term memories injected by the recall node (grounding)
    draft: str              # current generated output
    verdict: str            # ACKNOWLEDGED | ITERATE | REJECTED (from the critic)
    critique: str           # critic feedback / asks, threaded back into generate
    iterations: int         # number of generate passes so far
    max_iterations: int     # iterate cap (mirrors OpenClaw's Hermes loop, <=3)
    result: str             # final accepted output
    escalation: str         # set when rejected or the iterate cap is exhausted
    importance: int         # LLM-rated salience (1-10) of the latest stored outcome
    reflection: str         # set when an episodic->semantic consolidation fired this run


class Critique(BaseModel):
    """Structured verdict the critic node returns — the adversarial gate.

    Mirrors OpenClaw's Hermes plan-critique contract (ACKNOWLEDGED/ITERATE/REJECTED).
    """

    verdict: Literal["ACKNOWLEDGED", "ITERATE", "REJECTED"] = Field(
        description="ACKNOWLEDGED = ship it; ITERATE = fixable, see asks; REJECTED = premise is wrong."
    )
    reasons: str = Field(description="Why this verdict.")
    asks: list[str] = Field(
        default_factory=list,
        description="Concrete changes to make if verdict is ITERATE.",
    )


class Importance(BaseModel):
    """Salience score assigned to a memory at write time (Generative-Agents style).

    Used to decide when enough has accumulated to trigger a reflection.
    """

    score: int = Field(ge=1, le=10, description="1 = mundane, 10 = pivotal.")
    rationale: str = Field(default="", description="Why this score.")
