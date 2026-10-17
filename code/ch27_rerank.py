"""Chapter 27 - Reranking: recall loosely, then let the model denoise.

A single similarity threshold on the retriever forces an impossible choice. Set
it high and you miss relevant documents (poor recall); set it low and junk
leaks into the prompt (poor precision). Reranking splits that one dial into two
honest stages:

    search_docs (recall):  pull the top-k by vector similarity with NO threshold
                           - better too many candidates than too few.
    rerank (precision):    ask the LLM to score each candidate 0-10 for how well
                           it actually answers THIS question, then keep only the
                           high scorers.

Retrieval promises "don't miss it"; reranking promises "don't use the wrong
one." Two cheap stages beat one brittle threshold.

This also changes the GRAPH, not just a node body: rerank is a NEW node spliced
between search_docs and agent (search_docs -> rerank -> agent). When every
candidate scores low - a question the knowledge base simply cannot answer -
rerank returns nothing, and the agent falls back to web search on its own.

The knowledge base is a handful of English docs embedded by the local model server
and stored in Chroma. The store is rebuilt from those docs on every run, so this
file is self-contained and never depends on a store built elsewhere.

Run:  python code/ch27_rerank.py
"""

import os
import shutil
import warnings

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph

# Local models served by LM Studio (OpenAI-compatible API). Nothing here talks
# to a paid cloud - the whole book runs on your own machine. timeout is generous
# because a local server may be busy loading a model on the first call.
LM_BASE = "http://localhost:1234/v1"

llm = ChatOpenAI(
    base_url=LM_BASE,
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",      # a capable open model that scores reliably
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
    timeout=240,
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


# check_embedding_ctx_length is off because the local server does not advertise a
# token limit the way the OpenAI cloud does.
embeddings = NomicEmbeddings(
    base_url=LM_BASE,
    api_key="lm-studio",
    model="text-embedding-nomic-embed-text-v1.5",
    check_embedding_ctx_length=False,
    timeout=240,
)

# A tiny knowledge base: five one-line docs on unrelated topics. Only the two
# Kafka lines truly answer a Kafka batch-consumption question; the rest are the
# noise that reranking has to strip out.
DOCUMENTS = [
    "Kafka batch consumption: set spring.kafka.listener.type=batch to enable batch consumption mode.",
    "Kafka tuning: batch consumers need max-poll-records tuned to raise consumer throughput.",
    "Redis connection pool: configure pool parameters through the spring.data.redis.* settings.",
    "Spring Boot auto-configuration: @SpringBootApplication triggers the auto-wiring mechanism.",
    "MySQL indexing: a B+ tree index on frequently queried columns greatly speeds up slow queries.",
]

# Build a fresh Chroma store next to this file, rebuilt from scratch on every run
# so the demo never depends on stale vectors. cosine space matches the embeddings.
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_ch27_rerank")
shutil.rmtree(DB_DIR, ignore_errors=True)
vectorstore = Chroma(
    collection_name="doc_bot_rerank",
    embedding_function=embeddings,
    persist_directory=DB_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)
vectorstore.add_texts(DOCUMENTS)
print(f"[DB] built knowledge base with {len(DOCUMENTS)} documents at {DB_DIR}")


RERANK_CUTOFF = 6   # keep a candidate only if the LLM scores it >= this (0-10)


# The agent's routing decision, as structured output so we get a clean enum back.
class Decision(BaseModel):
    route: Literal["search", "web", "generate"] = Field(description="the next action")
    reason: str = Field(description="a short justification")


decider = llm.with_structured_output(Decision, method="json_schema")


# The reranker's per-document relevance score, also structured output.
class Relevance(BaseModel):
    score: int = Field(ge=0, le=10,   # a score outside the scale is a scorer failure, not a low score
                       description="relevance of the document to the question, 0 (irrelevant) to 10 (perfect)")


scorer = llm.with_structured_output(Relevance, method="json_schema")


class State(TypedDict):
    question: str
    route: str
    docs: list
    web_results: list
    iterations: int
    answer: str
    reason: str
    rerank_degraded: bool   # the scorer failed and the shortlist is the raw recall order


def agent_node(state: State) -> dict:
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    prompt = f"""
        You are the router for a question-answering agent. The actions are:
        - "search": search the LOCAL knowledge base. Always try this FIRST, before the web.
        - "web": search the public WEB. Only allowed AFTER the local base was searched and came up empty.
        - "generate": write the final answer from what we already have.

        User question: {state['question']}
        Local documents found: {len(docs)}
        Web results found: {len(web)}
        Local knowledge base already searched: {"docs" in state}
        Web already searched: {"web_results" in state}

        Choose the FIRST action whose condition holds:
        1. If local documents found > 0 or web results found > 0 -> generate
        2. Otherwise if the local knowledge base was NOT searched yet -> search
        3. Otherwise if the web was NOT searched yet -> web
        4. Otherwise -> generate"""
    d = decider.invoke(prompt)
    print(f"[agent] route={d.route}  reason={excerpt(d.reason)}")
    return {"route": d.route, "reason": d.reason, "iterations": state.get("iterations", 0) + 1}


# ---------------------------------------------------------------------------
# search_docs: RECALL stage. Top-5 with NO threshold; rerank filters downstream.
# ---------------------------------------------------------------------------
def search_docs_node(state: State) -> dict:
    results = vectorstore.similarity_search_with_score(state["question"], k=5)
    candidates = [doc.page_content for doc, _ in results]
    print(f"   [search_docs] loosely recalled {len(candidates)} candidate(s) (rerank will filter)")
    return {"docs": candidates}


# ---------------------------------------------------------------------------
# rerank: PRECISION stage. Score each candidate 0-10 for THIS question and keep
# only the high scorers. This is the new node in the graph.
# ---------------------------------------------------------------------------
def score_candidates(question: str, candidates: list, score_fn=None, attempts: int = 2) -> list | None:
    """Score every candidate; None if the scorer FAILED.

    Two outcomes look alike from the outside and must not be confused: "the
    scorer ran and found nothing relevant" (a list of low scores, and the empty
    shortlist that follows is the agent's web fallback working) and "the scorer
    did not run" (a timeout, a reply outside the 0-10 scale). The second is an
    execution failure: retried once, and if it fails again this returns None
    so the caller can degrade on purpose instead of shipping a zero it never
    measured.
    """
    score_fn = score_fn or (lambda p: scorer.invoke(p).score)
    scored = []
    for doc in candidates:
        prompt = f"""
            Score how relevant the document is to answering the question, from 0 to 10.
            Question: {question}
            Document: {doc}"""
        for attempt in range(attempts):
            try:
                score = score_fn(prompt)
                break
            except Exception as exc:   # noqa: BLE001 - timeout, bad schema, out of range
                if attempt == attempts - 1:
                    print(f"   [rerank] scorer failed on candidate {len(scored) + 1}: "
                          f"{type(exc).__name__} -- keeping the recall order unscored")
                    return None
        scored.append((score, doc))
        print(f"        {score:>2}/10  {doc}")
    return scored


def rerank_node(state: State) -> dict:
    question = state["question"]
    candidates = state.get("docs", [])
    print(f"   [rerank] scoring {len(candidates)} candidate(s):")

    scored = score_candidates(question, candidates)
    if scored is None:                                         # execution failure, not "all irrelevant"
        return {"docs": candidates[:2], "rerank_degraded": True}   # top of the recall order, flagged

    scored.sort(reverse=True)                                  # best first
    kept = [doc for score, doc in scored if score >= RERANK_CUTOFF]
    print(f"   [rerank] kept {len(kept)} (score >= {RERANK_CUTOFF})")
    return {"docs": kept, "rerank_degraded": False}            # overwrite docs with the shortlist


def web_search_node(state: State) -> dict:
    print("   [web_search] (stub) falling back to the web")
    return {"web_results": [f"[web] community discussion about '{state['question']}'"]}


def generate_node(state: State) -> dict:
    sources = state.get("docs", []) + state.get("web_results", [])
    sources_text = "\n".join(f"- {s}" for s in sources) or "(no sources)"
    prompt = f"""
        Answer the question concisely, using only the sources below. Do not make things up.

        Question: {state['question']}
        Sources:
        {sources_text}

        Answer:"""
    answer = llm.invoke(prompt).content.strip()
    print("   [generate] wrote the answer")
    return {"answer": answer}


def route_after_agent(state: State) -> str:
    return state["route"]


# ---------------------------------------------------------------------------
# The graph. Note the changed wiring: search_docs -> rerank -> agent inserts the
# new node without touching any of the others.
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("rerank", rerank_node)               # the new node
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", route_after_agent,
    {"search": "search_docs", "web": "web_search", "generate": "generate"},
)
builder.add_edge("search_docs", "rerank")             # recall -> precision
builder.add_edge("rerank", "agent")
builder.add_edge("web_search", "agent")
builder.add_edge("generate", END)
graph = builder.compile()


if __name__ == "__main__":
    for q in [
        "How do I configure Kafka batch consumers?",   # rerank keeps the Kafka docs, drops the rest
        "How do I speed up slow database queries?",     # rerank keeps only the MySQL doc
        "How do I write a Python decorator?",           # rerank drops everything -> agent falls back to web
    ]:
        print(f"\n{'=' * 60}\nQuestion: {q}\n{'=' * 60}")
        result = graph.invoke({"question": q}, config={"recursion_limit": 15})
        print("--- answer ---")
        print(excerpt(result["answer"]))
