"""Chapter 15 Challenge - Gold: give the Store an embedding index.

The chapter's Store files facts under a namespace and hands them back by prefix:
`search(namespace)` is a LISTING, in insertion order, and it has no idea what any
of the facts say. An embedding index changes that. Build the Store with

    InMemoryStore(index={"embed": <model>, "dims": <its width>, "fields": ["$"]})

and every item is embedded on the way in, so `search(namespace, query="...")`
embeds the question too and ranks by distance - the retrieval machinery of
Chapter 25, pointed at remembered facts instead of documents.

The point of this file is the CONTRAST, and the second half is the part worth
remembering: ask an UNINDEXED store the same question and it does not raise. It
quietly ignores `query=`, hands back every item in the namespace, and sets every
`score` to None. A ranking you never got looks exactly like a ranking that put
everything first.

Prerequisite: LM Studio serving the embedding model from Chapter 0. This is the
only Part II example that needs it - the chat model is not used here at all.

Run:  python code/answers/ch15_semantic_recall.py   (see answers/README.md)
"""

from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore

LM_BASE = "http://localhost:1234/v1"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
EMBED_DIMS = 768          # this model's width - NOT the 1536 the docs' example uses

embeddings = OpenAIEmbeddings(
    base_url=LM_BASE,
    api_key="lm-studio",                     # the local server ignores the key
    model=EMBED_MODEL,
    check_embedding_ctx_length=False,        # send text, not token ids (@sec-chroma)
)

NAMESPACE = ("memories", "alice")
# Each fact names its subject. That is not decoration: with a short sentence and
# a small local embedder, similarity scores bunch up around 0.5, and dropping the
# subject shrinks the gap between the right fact and the wrong one to ~0.03. Named,
# the query below wins by ~0.12.
FACTS = {
    "f1": {"text": "Alice is allergic to shellfish and avoids seafood restaurants"},
    "f2": {"text": "Alice works from Berlin and is two hours ahead of the rest of the team"},
    "f3": {"text": "Alice wants deploy notifications in Slack, never by email"},
    "f4": {"text": "Alice reviews every pull request before it is merged to main"},
}


def seeded(store: InMemoryStore) -> InMemoryStore:
    for key, value in FACTS.items():
        store.put(NAMESPACE, key, value)
    return store


def indexed_store() -> InMemoryStore:
    """The whole change: three keys under `index`."""
    return seeded(InMemoryStore(index={
        "embed": embeddings,      # a LangChain Embeddings instance
        "dims": EMBED_DIMS,       # must match THAT model, or you index into the wrong space
        "fields": ["$"],          # JSON path: "$" = embed the whole stored dict
    }))


def plain_store() -> InMemoryStore:
    """The chapter's Store, unchanged - no index."""
    return seeded(InMemoryStore())


QUESTION = "what timezone is Alice in?"      # no word of it appears in any fact


def ranked(store: InMemoryStore, query: str):
    """(key, score) for each hit, best first."""
    return [(i.key, i.score) for i in store.search(NAMESPACE, query=query)]


# --- tests -------------------------------------------------------------------
def test_indexed_store_ranks_the_right_fact_first():
    hits = ranked(indexed_store(), QUESTION)
    assert hits[0][0] == "f2"                       # the Berlin/timezone fact
    assert all(s is not None for _k, s in hits)     # every hit carries a score
    # Embeddings are deterministic, so this margin is stable for THIS model - but
    # it is only ~0.12, and it is the first thing a different embedder would move.
    assert hits[0][1] - hits[1][1] > 0.05


def test_unindexed_store_ignores_the_query_without_saying_so():
    hits = ranked(plain_store(), QUESTION)
    assert {k for k, _s in hits} == set(FACTS)      # the whole namespace came back
    assert all(s is None for _k, s in hits)         # and nothing was ranked


def test_neither_call_raises():
    # The failure this challenge is about is silent in both directions.
    for store in (indexed_store(), plain_store()):
        store.search(NAMESPACE, query=QUESTION)


def test_prefix_listing_is_unchanged_by_the_index():
    # Adding an index does not disturb the chapter's plain listing.
    assert {i.key for i in indexed_store().search(NAMESPACE)} == set(FACTS)


if __name__ == "__main__":
    print(f"question: {QUESTION!r}\n")
    for label, store in (("indexed", indexed_store()), ("no index", plain_store())):
        print(f"--- {label} store")
        for key, score in ranked(store, QUESTION):
            shown = "None " if score is None else f"{score:.3f}"
            print(f"    {shown}  {key}  {FACTS[key]['text']}")
        print()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
