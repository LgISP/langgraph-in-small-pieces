"""Chapter 25 — RAG by hand: turn search_docs from keyword-lookup into vector retrieval.

Chapter 5's search_docs returned a hard-coded stub. Here it becomes a real
retriever, and we write the whole mechanism out in the open — no vector-store
library hiding the math — so you can see what "retrieval" actually is:

    1) at startup: embed every document into a vector, and keep the vectors (the index)
    2) at query time: embed the question into a vector too
    3) score the question vector against each document vector with COSINE SIMILARITY
    4) keep the top-k whose score clears a threshold; those are the "hits"

The one idea to carry away: the graph does not change to gain retrieval. The
graph, the agent, generate, and every edge are byte-for-byte the Chapter 5 shape.
Only the BODY of search_docs changes — keyword match becomes vector search. That
is the whole point: a node is a pluggable box behind a stable edge.

Prerequisite: LM Studio is serving an embedding model (text-embedding-nomic-embed-text-v1.5).

Run:  python code/ch25_rag_cosine.py
"""

import warnings

# with_structured_output makes pydantic emit one harmless serializer warning about
# an internal field; the parsed result is correct. Silence just that one line.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

import math

import requests
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


LM_BASE = "http://localhost:1234/v1"      # your local server, not the OpenAI cloud
API_KEY = "lm-studio"                  # the local server ignores the key; a hosted one needs a real one
CHAT_MODEL = "qwen/qwen3.8-27b"   # follows instructions well; good at structured output
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"

llm = ChatOpenAI(
    base_url=LM_BASE,
    api_key="lm-studio",   # the local server ignores the key
    model=CHAT_MODEL,
    temperature=0,         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
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


# ===========================================================================
# RAG, part one: the embedding call + cosine similarity — the math core of retrieval.
# ===========================================================================
def embed(text: str) -> list[float]:
    """Ask LM Studio's embedding endpoint to turn a string into one vector (768 numbers)."""
    r = requests.post(
        f"{LM_BASE}/embeddings",
        json={"model": EMBED_MODEL, "input": text},
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60,
    )
    r.raise_for_status()      # a 4xx/5xx here is a config problem, not a vector
    return r.json()["data"][0]["embedding"]


# nomic-embed-text wants you to say WHY you are embedding: `search_document:` for
# text going INTO the index, `search_query:` for the thing you are looking up. The
# model card requires these task prefixes for retrieval, and the server does not add
# them for you. Leave them off and you still get vectors back -- which is why this is
# easy to get wrong -- but you are matching a query against documents in a space the
# model was never asked to build.
def embed_document(text: str) -> list[float]:
    return embed(f"search_document: {text}")


def embed_query(text: str) -> list[float]:
    return embed(f"search_query: {text}")


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity: the closer two vectors point, the closer to 1 (the closer in meaning)."""
    # zip pairs the vectors position by position; each sum(...) runs over those
    # pairs without building a list in between
    dot = sum(x * y for x, y in zip(a, b))    # the dot product
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


# ===========================================================================
# RAG, part two: a minimal "vector store". At startup we embed every document (build the index).
# ===========================================================================
DOCUMENTS = [
    "Kafka batch consumption: set spring.kafka.listener.type=batch to enable batch consumer mode.",
    "Kafka tuning: batch consumers need max-poll-records tuning for throughput.",
    "Redis connection pool: configure pool parameters via the spring.data.redis.* settings.",
    "Spring Boot auto-configuration: @SpringBootApplication triggers the auto-configuration mechanism.",
    "MySQL indexing: a B+ tree index on frequently queried columns greatly speeds up slow queries.",
]

THRESHOLD = 0.60   # below this similarity we treat a document as "not relevant, no hit" (calibrated by hand)


class VectorStore:
    """A bare-bones vector store: documents + their vectors, with top-k semantic search."""

    def __init__(self, docs: list[str]):
        print(f"[index] embedding {len(docs)} document(s) ...")
        self.docs = docs
        self.vectors = [embed_document(d) for d in docs]   # one vector per document
        print("[index] index built")

    def search(self, query: str, k: int = 3, threshold: float = THRESHOLD) -> list[str]:
        q_vec = embed_query(query)                 # the question becomes a vector too,
                                                   # under the OTHER task prefix
        # score the question against every document
        scored = [(cosine(q_vec, v), d) for v, d in zip(self.vectors, self.docs)]
        scored.sort(reverse=True)                  # highest similarity first
        hits = [(s, d) for s, d in scored if s >= threshold][:k]
        # print the scores so you can watch retrieval happen (in real RAG this is your observability point)
        print(f"   [retrieve] similarity for '{query}':")
        for s, d in scored[:3]:
            mark = "HIT " if s >= threshold else "skip"
            print(f"        {s:.3f} {mark}  {d}")
        return [d for s, d in hits]


# Build the index once, at startup.
VSTORE = VectorStore(DOCUMENTS)


# ===========================================================================
# The decision shape (same structured output as the earlier agent chapters).
# ===========================================================================
class Decision(BaseModel):
    # reason comes FIRST on purpose. With JSON-schema decoding the model emits the
    # fields in schema order, so it writes `reason` before it commits to `route` —
    # enough, on this pinned LM Studio stack, to stop it skipping the search rule
    # and jumping straight to web (see Chapter 14 for the same trick).
    reason: str = Field(description="Work through the rules in order, then decide.")
    route: Literal["search", "web", "generate"] = Field(description="the next action")


# Bind the shape to the model: `decider` returns a Decision object, never raw text.
decider = llm.with_structured_output(Decision, method="json_schema")


class State(TypedDict):
    question: str                          # input, set by the caller
    route: NotRequired[str]                # the agent's pick this step
    docs: NotRequired[list[str]]           # written by search_docs
    web_results: NotRequired[list[str]]    # written by web_search
    iterations: NotRequired[int]           # a loop counter, for observability
    answer: NotRequired[str]               # written by generate
    reason: NotRequired[str]               # the agent's reason for its pick


def agent_node(state: State) -> dict:
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    prompt = f"""
        You are the router for a question-answering agent. Pick the next action.

        Question: {state['question']}
        Local documents searched yet: {"yes" if "docs" in state else "no"}   (hits so far: {len(docs)})
        Web searched yet: {"yes" if "web_results" in state else "no"}   (results so far: {len(web)})

        Apply the FIRST rule that matches, in this exact priority order:
        1. If there is at least one local hit or web result -> generate
        2. Otherwise, if local documents have not been searched yet -> search
        3. Otherwise, if the web has not been searched yet -> web
        4. Otherwise -> generate

        Always try the local documents (search) before the web."""
    d = decider.invoke(prompt)
    print(f"[agent] route={d.route}  reason={excerpt(d.reason)}")
    return {"route": d.route, "reason": d.reason, "iterations": state.get("iterations", 0) + 1}


# ---------------------------------------------------------------------------
# The one node that actually changed: search_docs now does vector retrieval.
# ---------------------------------------------------------------------------
def search_docs_node(state: State) -> dict:
    hits = VSTORE.search(state["question"])
    print(f"   [search_docs] {len(hits)} hit(s)")
    return {"docs": hits}


def web_search_node(state: State) -> dict:
    print("   [web_search] (stub) fall back to a web search")
    return {"web_results": [f"[web] community discussion about '{state['question']}'"]}


def generate_node(state: State) -> dict:
    sources = state.get("docs", []) + state.get("web_results", [])
    sources_text = "\n".join(f"- {s}" for s in sources) or "(no sources)"
    prompt = f"""
        Answer the question concisely using only the sources below. Do not make things up.

        Question: {state['question']}
        Sources:
        {sources_text}

        Answer:"""
    answer = llm.invoke(prompt).content.strip()
    print("   [generate] answer generated")
    return {"answer": answer}


def route_after_agent(state: State) -> str:
    return state["route"]


# ---------------------------------------------------------------------------
# Build the graph — still byte-for-byte the Chapter 5 shape.
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {"search": "search_docs", "web": "web_search", "generate": "generate"},
)
builder.add_edge("search_docs", "agent")
builder.add_edge("web_search", "agent")
builder.add_edge("generate", END)
graph = builder.compile()


if __name__ == "__main__":
    # The third question is deliberately about something NOT in the store (Python)
    # -> zero hits -> the agent falls back to the web branch.
    for q in [
        "How do I configure Spring Boot Kafka batch consumers?",  # semantically matches the Kafka docs
        # Not one content word in common with the MySQL doc -- no keyword search
        # could connect them; only meaning can.
        "Why are lookups against my relational tables so sluggish?",
        "How do I write a Python decorator?",                     # not in the store -> fall back to web
    ]:
        print(f"\n{'=' * 50}\nQuestion: {q}\n{'=' * 50}")
        result = graph.invoke({"question": q}, config={"recursion_limit": 12})
        print("--- answer ---")
        print(excerpt(result["answer"]))
