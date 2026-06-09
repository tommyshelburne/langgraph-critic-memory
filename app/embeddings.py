"""Embeddings provider for the Store's semantic index.

Returns a real embeddings model when EMBEDDINGS_MODEL is set (e.g.
`openai:text-embedding-3-small`), otherwise a tiny deterministic offline
hash embedding so semantic search is demonstrable with no API key.

Note: per the research, LangGraph's Store semantic index is OPT-IN. Here we
always enable it (with a toy embedder by default) so the pattern is visible;
in production you'd gate it and use a real model.
"""

import hashlib
import math
import os

from langchain_core.embeddings import Embeddings


def get_embeddings() -> Embeddings:
    model_id = os.environ.get("EMBEDDINGS_MODEL")
    if model_id:
        from langchain.embeddings import init_embeddings

        return init_embeddings(model_id)
    return LocalHashEmbeddings()


class LocalHashEmbeddings(Embeddings):
    """Deterministic, offline bag-of-words hash embedding.

    Captures lexical overlap only — NOT a real semantic model. It exists so
    the demo runs without network/keys; replace via EMBEDDINGS_MODEL.
    """

    def __init__(self, dims: int = 256):
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for tok in text.lower().split():
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dims
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
