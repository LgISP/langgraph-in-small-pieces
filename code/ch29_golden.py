"""Chapter 29 - The golden-answer layer.

Some answers are worth more than the documents they came from. A reviewed,
known-good reply to a frequently asked question is faster and more reliable than
re-running retrieval and generation every time. So we cache those blessed
answers in their own store - the golden layer - and let a query hit it FIRST.

Two knowledge bases sit side by side:

    docs    the source documents - the source of truth
    golden  curated Q&A - distilled from feedback, human-reviewed, fast to
            serve, but DERIVED, so it can go stale when the docs move on

The cascade is just a conditional edge:

    check_golden --hit (close enough + fresh)--> END        (no LLM call at all)
                 --miss / stale-------------> retrieve_docs -> generate  (normal RAG)

Two guards protect the golden layer:

    1. similarity - only a very close question counts as a hit (threshold set HIGH)
    2. freshness  - a golden answer derived from an outdated doc version is refused

Run:  python code/ch29_golden.py
"""

import os
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from typing import TypedDict

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

# Both models are served locally by LM Studio (OpenAI-compatible API). Nothing
# here talks to a paid cloud - the whole book runs on your own machine.
LM_BASE = "http://localhost:1234/v1"
GOLDEN_THRESHOLD = 0.80   # set the bar high; a false hit skips the model

llm = ChatOpenAI(
    base_url=LM_BASE,
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)
class NomicEmbeddings(OpenAIEmbeddings):
    """nomic's task prefixes -- the adapter @sec-rag-foundations introduced.
    search_document: for indexed text, search_query: for a lookup; the model card
    requires them for retrieval and the local server does not add them."""

    def embed_documents(self, texts, *args, **kwargs):
        return super().embed_documents(
            [f"search_document: {t}" for t in texts], *args, **kwargs)

    def embed_query(self, text):
        # super().embed_documents, NOT embed_query: the base routes embed_query
        # through embed_documents, which would stack both prefixes.
        return super().embed_documents([f"search_query: {text}"])[0]


emb = NomicEmbeddings(
    base_url=LM_BASE,
    api_key="lm-studio",
    model="text-embedding-nomic-embed-text-v1.5",
    check_embedding_ctx_length=False,      # the local server has no such endpoint
)


# ---------------------------------------------------------------------------
# Build the two stores (rebuilt every run - this is a demo)
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
for d in ["chroma_ch29_docs", "chroma_ch29_golden"]:
    shutil.rmtree(os.path.join(BASE, d), ignore_errors=True)

# The source-document store.
doc_store = Chroma(
    collection_name="docs",
    embedding_function=emb,
    persist_directory=os.path.join(BASE, "chroma_ch29_docs"),
    collection_metadata={"hnsw:space": "cosine"},
)
# ⚠️ Every document carries the version it was indexed at, exactly as
#    @sec-versioning's sync_index stamps it. The freshness guard below READS this
#    rather than consulting a table someone maintains by hand -- a parallel table
#    is a second source of truth, and the failure it produces is silent: the
#    golden layer keeps serving answers derived from documents that moved, because
#    nobody remembered to bump a dict.
doc_store.add_texts(
    texts=[
        "Kafka batch consumption: set spring.kafka.listener.type=batch to enable it.",
        "Redis connection pool: configure it via spring.data.redis.lettuce.pool.max-active.",
        "MySQL slow queries: a B+ tree index on frequently queried columns speeds them up a lot.",
    ],
    metadatas=[
        {"topic": "kafka", "version": "v2"},
        {"topic": "redis", "version": "v2"},
        {"topic": "mysql", "version": "v2"},
    ],
)


def current_doc_version(store: Chroma, topic: str) -> str | None:
    """What version a DOCUMENT STORE currently holds for a topic.

    One source of truth: the same metadata @sec-versioning writes on every sync.
    Re-index a topic, the stamp moves, and every golden answer derived from the
    old one goes stale on the next question -- with nothing to remember to update.
    The store is a parameter, not a global: the guard works against whichever
    index the answer was derived from.
    """
    got = store.get(where={"topic": topic}, include=["metadatas"])
    versions = {m.get("version") for m in got["metadatas"]}
    return next(iter(versions), None) if len(versions) == 1 else None

golden_store = Chroma(
    collection_name="golden",
    embedding_function=emb,
    persist_directory=os.path.join(BASE, "chroma_ch29_golden"),
    collection_metadata={"hnsw:space": "cosine"},
)
golden_store.add_texts(
    texts=[  # embed the QUESTIONS, not the answers
        "How do I configure Kafka batch consumption?",
        "How do I configure a Redis connection pool?",
    ],
    metadatas=[
        {"answer": "[Golden] Set spring.kafka.listener.type=batch, tune throughput "
                   "with max-poll-records, and take a List parameter in the batch listener.",
         "topic": "kafka", "doc_version": "v2"},   # v2 == current -> fresh
        {"answer": "[Golden (old)] Configure the connection pool via spring.redis.pool.max-active.",
         "topic": "redis", "doc_version": "v1"},    # v1 != current v2 -> stale!
    ],
)


def excerpt(text: str, width: int = 140) -> str:
    """First line of a model answer, cut at a word boundary, with '...' if cut.

    Answers are long and often contain code fences; the trace only needs
    enough to see WHAT the model decided to say, not the whole essay.
    """
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) <= width and first == text.strip():
        return first
    cut = first[:width].rsplit(" ", 1)[0] if len(first) > width else first
    return cut.rstrip(" ,;:") + " ..."


def is_stale(golden: dict, store: Chroma) -> bool:
    """Freshness guard: the doc version a golden entry (its metadata) was derived
    from no longer matches what `store` currently holds for its topic.

    On a well-formed entry the guard is its last line. Keeping entries
    well-formed (stamped with `topic` and `doc_version`) is the review process's
    job, not the guard's; the two early returns only decide what happens if that
    slips -- an unstamped entry, or a topic whose documents are gone, would compare
    None against None (EQUAL) and be called fresh, so they call it stale instead
    and the slip costs a retrieval rather than a served answer.
    """
    topic = golden.get("topic")
    golden_version = golden.get("doc_version")
    if not topic or not golden_version:
        return True                      # untagged entry: nothing to check against
    current = current_doc_version(store, topic)
    if current is None:
        return True                      # source gone, or no single current version
    return golden_version != current


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------
class State(TypedDict):
    question: str
    docs: str          # what retrieval found - NOT the answer, see retrieve_docs_node
    answer: str
    source: str        # "golden" or "generated" - which path we took
    golden_route: str  # "hit" / "miss"


def check_golden_node(state: State) -> dict:
    # One audience, so no tenant filter here -- on purpose, and said out loud. A
    # store that serves several tenants carries the caller's scope on this lookup
    # too (@sec-rag-tenant-isolation): a golden answer that leaks is still a leak.
    hits = golden_store.similarity_search_with_score(state["question"], k=1)
    if hits:
        doc, dist = hits[0]
        sim = 1 - dist                       # cosine distance -> similarity
        stale = is_stale(doc.metadata, doc_store)
        print(f"   [check_golden] closest golden question similarity={sim:.3f} stale={stale}")
        # Guard 1: close enough.  Guard 2: not stale.
        if sim >= GOLDEN_THRESHOLD and not stale:
            return {"answer": doc.metadata["answer"], "source": "golden", "golden_route": "hit"}
        if sim >= GOLDEN_THRESHOLD and stale:
            print("   [check_golden] similar enough, but stale -> guard refuses, falling back to docs")
    else:
        print("   [check_golden] no golden candidate")
    return {"golden_route": "miss"}


def route_golden(state: State) -> str:
    return state["golden_route"]


def retrieve_docs_node(state: State) -> dict:
    docs = doc_store.similarity_search(state["question"], k=2)
    print(f"   [retrieve_docs] recalled {len(docs)} document(s) from the source store")
    # ⚠️ These are DOCUMENTS, so they go in `docs`. Parking them in `answer`
    #    and overwriting it one node later would work, and would teach that a
    #    state key means whatever the last writer meant -- which is how a field
    #    ends up holding two different things and a reader of `answer` mid-run
    #    gets raw sources instead of a reply.
    return {"docs": "\n".join(d.page_content for d in docs)}


def is_transient_model_swap(exc: Exception) -> bool:
    """A model served on demand by LM Studio can be swapped out mid-run when the
    server is shared; that surfaces as a 400 while the model reloads. Those are
    transient, so we let the node retry rather than crash the whole graph."""
    msg = str(exc).lower()
    return any(s in msg for s in ("unload", "failed to load", "operation canceled", "loading"))


# Every node here calls the shared LM Studio (check_golden and retrieve_docs embed
# the query; generate runs the chat model), so any of them can catch a transient
# swap. One policy, applied to all three, keeps the demo robust without touching
# the golden-layer logic.
MODEL_SWAP_RETRY = RetryPolicy(
    max_attempts=8, initial_interval=2.0, max_interval=20.0,
    retry_on=is_transient_model_swap,
)


NOT_COVERED = "Not covered in the docs."


def generate_node(state: State) -> dict:
    # The fall-through has no similarity threshold (@sec-rag-foundations has one):
    # retrieve_docs hands back the nearest two documents whatever the question,
    # so the refusal has to be the model's, and it is a fixed sentence so that a
    # double miss -- no golden entry, no covering document -- has a checkable end.
    prompt = f"""
        Answer the question concisely, using only the sources below. If the
        sources do not cover the question, reply exactly "{NOT_COVERED}".

        Question: {state['question']}

        Sources:
        {state['docs']}

        Answer:"""
    answer = llm.invoke(prompt).content.strip()
    return {"answer": answer, "source": "generated"}


builder = StateGraph(State)
builder.add_node("check_golden", check_golden_node, retry_policy=MODEL_SWAP_RETRY)
builder.add_node("retrieve_docs", retrieve_docs_node, retry_policy=MODEL_SWAP_RETRY)
builder.add_node("generate", generate_node, retry_policy=MODEL_SWAP_RETRY)
builder.add_edge(START, "check_golden")
builder.add_conditional_edges(
    "check_golden", route_golden, {"hit": END, "miss": "retrieve_docs"}
)
builder.add_edge("retrieve_docs", "generate")
builder.add_edge("generate", END)
graph = builder.compile()


if __name__ == "__main__":
    cases = [
        ("How do I set up Kafka batch consumption?",
         "frequent question -> hits a fresh golden answer, returned directly (no LLM call)"),
        ("How do I optimize slow database queries?",
         "rare question -> not in the golden layer, falls back to document generation"),
        ("How do I configure connection pooling for Redis?",
         "matches a golden answer, but it is stale -> guard refuses, falls back to docs"),
        ("How do I rotate the TLS certificates on the gateway?",
         "double miss -> no golden entry and no covering document; generate must refuse"),
    ]
    for question, expect in cases:
        print(f"\n{'=' * 56}\nQuestion: {question}\nExpect:   {expect}\n{'=' * 56}")
        result = graph.invoke({"question": question})
        print(f"   >>> source = {result['source']}")
        print(f"   >>> answer: {excerpt(result['answer'])}")   # first line, trimmed
