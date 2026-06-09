"""Chat model provider.

Returns a real LangChain chat model when CHAT_MODEL is set (e.g.
`anthropic:claude-sonnet-4-6` or `openai:gpt-4o`), otherwise a deterministic
offline stub so the whole graph runs with no API key — the scaffold works
out of the box, then you swap in a real model by setting one env var.
"""

import os

from langchain_core.messages import AIMessage


def get_model():
    model_id = os.environ.get("CHAT_MODEL")
    if model_id:
        # init_chat_model is provider-agnostic: "anthropic:...", "openai:...", etc.
        from langchain.chat_models import init_chat_model

        return init_chat_model(model_id)
    return _StubChat()


class _StubChat:
    """Offline stand-in. Deterministic so the demo + tests are reproducible.

    Behavior: emits a first draft, then a '[revised]' draft once asked to
    revise — which lets the stub critic ACKNOWLEDGE after exactly one iterate,
    demonstrating the full generate -> critic -> iterate -> accept loop.
    """

    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        if "Revise" in text:
            return AIMessage(content="[stub] revised draft addressing the critic's asks [revised]")
        return AIMessage(content="[stub] initial draft")

    def with_structured_output(self, schema):
        return _StructuredStub(schema)


class _StructuredStub:
    """Stub for `model.with_structured_output(Critique)` — keeps node code
    identical between the stub and a real model."""

    def __init__(self, schema):
        self.schema = schema

    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        if "[revised]" in text:
            return self.schema(
                verdict="ACKNOWLEDGED",
                reasons="stub: the revision addresses the asks",
                asks=[],
            )
        return self.schema(
            verdict="ITERATE",
            reasons="stub: first draft is too generic",
            asks=["Add a concrete, specific example"],
        )
