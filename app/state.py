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
