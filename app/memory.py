"""Long-term memory Store.

The Store is namespaced by user and persists ACROSS threads — this is the key
distinction from the checkpointer (per-thread short-term state). Two runs on
different thread_ids but the same user_id share this memory; that's what the
demo in main.py shows.
"""

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from app.embeddings import get_embeddings


def build_store() -> BaseStore:
    """In-memory Store with an opt-in semantic index over the `text` field.

    Swap InMemoryStore for `langgraph.store.postgres.PostgresStore` (or the
    LangGraph Platform managed store) for durability — same BaseStore interface.
    """
    emb = get_embeddings()
    dims = len(emb.embed_query("dimension probe"))  # works for real or stub embeddings
    return InMemoryStore(index={"embed": emb, "dims": dims, "fields": ["text"]})
