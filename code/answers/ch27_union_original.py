"""Chapter 27 Challenge - Bronze: union the original query back in.

The rewrite demo's "after" side retrieves with the REWRITTEN queries only, so a
rewrite that drifts can retrieve *worse* than the plain question would have - and
silently lose a document the raw query would have found. The fix from the
chapter's Gotcha, made runnable: put the original question back into the
retrieval set and union it with the rewrites -

    for sq in [q] + state["queries"]:      # the original FIRST, then the rewrites
        for sim, d in retrieve(sq):
            after[d] = max(after.get(d, 0.0), sim)

so the raw query is a recall FLOOR and the rewrites are only upside on top of it.
Rewriting can now ADD candidates but never SUBTRACT the ones the plain query
would have found.

Everything real here is real: the rewriter is the book's local model (Qwen3.8
behind LM Studio, the chapter's own rewrite prompt) and retrieval is the
chapter's Chroma store over LM Studio's nomic embeddings, threshold 0.60.

An honest report from running it against the real stack, because the staged
version of this challenge overshot: we could NOT make the real rewriter lose a
document at temperature 0. Not with the chapter's prompt, not with an
aggressively "generalize away the specifics" prompt, not even with a 7B rewriter
- semantic retrieval forgives generalized wording, and the model keeps the
topic. (The drift is real - one observed fan-out confidently rewrote an unnamed
message queue into RabbitMQ queries that missed every Kafka doc - but a sibling
query in the same fan-out always recovered the recall.) What IS reproducibly
true, and what this file demonstrates instead of a staged catastrophe:

  1. colloquial, no keywords   raw 1/2, rewrite 2/2, union 2/2  (rewrite adds)
  2. two intents in one        raw 1/2, rewrite 2/2, union 2/2  (fan-out adds)
  3. precise, keyword-rich     recall ties at 1/1, but rewriting the precise
     query TENDS to erode its similarity margin (raw ~0.88 vs best rewrite
     lower) - the drift is margin erosion, one threshold away from a miss, and
     the union keeps the raw margin at zero cost.

     NOTE: rewrite_margin comes from a live rewrite, so it is model-dependent.
     `test_precise_query_rewrite_erodes_the_margin_and_union_restores_it` is the one assertion in this file that
     can legitimately flake: a rewrite that happens to land closer to the gold
     document scores HIGHER than the raw query. That does not break the lesson -
     you cannot know at build time which fan-out is the one that drifts.

Case 3 is the floor made visible in the numbers: you cannot know at build time
which fan-out will be the one where every rewrite drifts, and unioning the
original makes that run cost nothing.

Run:  python code/answers/ch27_union_original.py   (see answers/README.md)
"""

import os
import shutil
import warnings
from typing import TypedDict

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph


# Local models served by LM Studio (OpenAI-compatible API), as in the chapter.
LM_BASE = "http://localhost:1234/v1"

llm = ChatOpenAI(
    base_url=LM_BASE,
    api_key="lm-studio",
    model="qwen/qwen3.8-27b",
    temperature=0,
    reasoning_effort="none",
    timeout=240,
)

class NomicEmbeddings(OpenAIEmbeddings):
    # nomic's task prefixes, the @sec-rag-foundations adapter the chapter uses.
    def embed_documents(self, texts, *args, **kwargs):
        return super().embed_documents(
            [f"search_document: {t}" for t in texts], *args, **kwargs)

    def embed_query(self, text):
        return super().embed_documents([f"search_query: {text}"])[0]


embeddings = NomicEmbeddings(
    base_url=LM_BASE,
    api_key="lm-studio",
    model="text-embedding-nomic-embed-text-v1.5",
    check_embedding_ctx_length=False,
    timeout=240,
)


def excerpt(text: str, width: int = 140) -> str:
    """First line of a model answer, cut at a word boundary, with '...' if cut."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) <= width and first == text.strip():
        return first
    cut = first[:width].rsplit(" ", 1)[0] if len(first) > width else first
    return cut.rstrip(" ,;:") + " ..."


# --- the tiny knowledge base (five one-line docs), embedded in Chroma --------
DOCUMENTS = [
    "Kafka batch consumption: set spring.kafka.listener.type=batch to enable batch consumption mode.",
    "Kafka tuning: batch consumers need max-poll-records tuned to raise consumer throughput.",
    "Redis connection pool: configure pool parameters through the spring.data.redis.* settings.",
    "Spring Boot auto-configuration: @SpringBootApplication triggers the auto-wiring mechanism.",
    "MySQL indexing: a B+ tree index on frequently queried columns greatly speeds up slow queries.",
]

# Rebuild a fresh store on every run (self-contained), in code/ like the chapter's.
DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_ch27_union"
)
shutil.rmtree(DB_DIR, ignore_errors=True)
vectorstore = Chroma(
    collection_name="doc_bot_union",
    embedding_function=embeddings,
    persist_directory=DB_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)
vectorstore.add_texts(DOCUMENTS)
print(f"[DB] built knowledge base with {len(DOCUMENTS)} documents at {DB_DIR}")

SIM_THRESHOLD = 0.60   # similarity (= 1 - distance) below this counts as a miss


def retrieve(query: str) -> list:
    """Return (similarity, document) pairs whose similarity clears the threshold."""
    results = vectorstore.similarity_search_with_score(query, k=5)
    return [(1 - dist, doc.page_content) for doc, dist in results
            if (1 - dist) >= SIM_THRESHOLD]


def best_sim(query: str, gold_marker: str) -> float:
    """The similarity the gold doc scored for this query (threshold ignored) —
    used to measure the MARGIN a rewrite keeps or burns."""
    results = vectorstore.similarity_search_with_score(query, k=5)
    for doc, dist in results:
        if gold_marker in doc.page_content:
            return 1 - dist
    return 0.0


def covered_gold(hit_docs: list, gold: list) -> list:
    """Which gold markers (a keyword slice of each relevant doc) got recalled."""
    return [g for g in gold if any(g in d for d in hit_docs)]


def union_retrieve(queries: list, gold: list) -> tuple:
    """Retrieve with every query, union the hits (best similarity per doc), and
    report the recalled documents and which gold markers they covered."""
    best: dict = {}
    for q in queries:
        for sim, d in retrieve(q):
            best[d] = max(best.get(d, 0.0), sim)
    docs = [d for d, _ in sorted(best.items(), key=lambda kv: -kv[1])]
    return docs, covered_gold(docs, gold)


# --- the real rewriter: the chapter's prompt, the chapter's model ------------
class Rewrite(BaseModel):
    queries: list[str] = Field(
        description="2 to 3 retrieval-friendly search queries; keyword-rich, one intent each"
    )


rewriter = llm.with_structured_output(Rewrite, method="json_schema")

REWRITE_PROMPT = """
    You rewrite a user's question into better search queries for vector retrieval.
    Raw questions are often colloquial, vague, or ask several things at once, so retrieving with
    them directly recalls poorly. Rewrite the question:
    - name the concrete technology the question most likely refers to, and the configuration
      keywords it implies, instead of generic words like "message queue" or "database"
    - if the question contains several independent intents, split it into separate sub-queries,
      one intent per query
    - produce 2 to 3 focused queries in total, no more
    Output only the search queries, no explanations and no numbering.

    Question: {q}"""


# --- the graph: rewrite -> search_docs -> generate ---------------------------
class State(TypedDict):
    question: str
    gold: list          # the documents this question SHOULD recall (to measure recall)
    queries: list       # the rewritten search queries
    docs: list          # the union retrieval set that feeds generate
    raw_docs: list      # what the plain query alone retrieved (the floor)
    raw_recall: int
    rewrite_recall: int
    union_recall: int
    raw_margin: float       # gold doc's similarity from the raw query alone
    rewrite_margin: float   # gold doc's best similarity from the rewrites alone
    answer: str


def rewrite_node(state: State) -> dict:
    r = rewriter.invoke(REWRITE_PROMPT.format(q=state["question"]))
    print("   [rewrite] rewrote the raw question into search queries:")
    for sq in r.queries:
        print(f"        - {sq}")
    return {"queries": r.queries}


def search_docs_node(state: State) -> dict:
    q, gold = state["question"], state["gold"]

    # RAW: the plain question alone - the recall floor we must never drop below.
    raw_docs = [d for _, d in retrieve(q)]
    raw_hit = covered_gold(raw_docs, gold)

    # REWRITE-ONLY: the demo's "after" side - rewrites, but WITHOUT the original.
    rewrite_docs, rewrite_hit = union_retrieve(state["queries"], gold)

    # UNION: the fix - the original query FIRST, then the rewrites, unioned.
    union_docs, union_hit = union_retrieve([q] + state["queries"], gold)

    # The margin the first gold doc keeps under each retrieval set (case 3's story).
    raw_margin = best_sim(q, gold[0])
    rewrite_margin = max((best_sim(sq, gold[0]) for sq in state["queries"]),
                         default=0.0)

    print(f"   [raw]     plain question          -> recalled {len(raw_hit)}/{len(gold)}"
          f"   (gold[0] sim {raw_margin:.3f})")
    print(f"   [rewrite] rewrites only (no orig) -> recalled {len(rewrite_hit)}/{len(gold)}"
          f"   (gold[0] sim {rewrite_margin:.3f})")
    print(f"   [union]   original + rewrites     -> recalled {len(union_hit)}/{len(gold)}")
    return {
        "docs": union_docs,
        "raw_docs": raw_docs,
        "raw_recall": len(raw_hit),
        "rewrite_recall": len(rewrite_hit),
        "union_recall": len(union_hit),
        "raw_margin": raw_margin,
        "rewrite_margin": rewrite_margin,
    }


def generate_node(state: State) -> dict:
    sources = state.get("docs", [])
    sources_text = "\n".join(f"- {s}" for s in sources) or "(no sources)"
    prompt = (f"Answer the question concisely, using only the sources below. "
              f"Do not make things up.\n\nQuestion: {state['question']}\n"
              f"Sources:\n{sources_text}")
    answer = llm.invoke(prompt).content.strip()
    print(f"   [generate] {excerpt(answer)}")
    return {"answer": answer}


builder = StateGraph(State)
builder.add_node("rewrite", rewrite_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "rewrite")
builder.add_edge("rewrite", "search_docs")
builder.add_edge("search_docs", "generate")
builder.add_edge("generate", END)
graph = builder.compile()


# The three cases. 1 and 2 are the chapter's own demo questions; 3 is the
# challenge's "precise, keyword-rich query the model is tempted to broaden".
CASES = {
    "colloquial": {
        "question": "Our message queue keeps piling up and the consumer cannot keep up. What should we do?",
        "gold": ["Kafka tuning", "Kafka batch consumption"],
    },
    "two_intents": {
        "question": "How do I turn on Kafka batch consumption, and also our reports take forever to load from the database?",
        "gold": ["Kafka batch consumption", "MySQL indexing"],
    },
    "precise": {
        "question": "add a B+ tree index on frequently queried columns to speed up slow queries",
        "gold": ["MySQL indexing"],
    },
}

# Each case runs ONCE and the result is shared by every test (real model calls
# are slow, and the assertions all describe one run).
_RUNS: dict = {}


def run(name: str) -> dict:
    if name not in _RUNS:
        _RUNS[name] = graph.invoke(CASES[name], config={"recursion_limit": 10})
    return _RUNS[name]


# --- tests -------------------------------------------------------------------
# Real-model note: recall counts are asserted only where the margin is wide;
# everything else is the union's structural guarantee, which holds whatever the
# model rewrites.
def test_union_never_falls_below_either_alone():
    # The core guarantee: union recall >= max(raw, rewrite-only) on every case.
    for name in CASES:
        r = run(name)
        assert r["union_recall"] >= max(r["raw_recall"], r["rewrite_recall"]), name


def test_original_query_is_a_floor_at_the_document_level():
    # Stronger than recall: every doc the plain query found is still in the union.
    # The rewrites can only add candidates on top; they never subtract the floor.
    # This is BY CONSTRUCTION - it holds no matter what the model rewrote.
    for name in CASES:
        r = run(name)
        assert set(r["raw_docs"]).issubset(set(r["docs"])), name


def test_rewrites_add_recall_on_the_chapter_cases():
    # The chapter's two demo questions, on the real stack: the colloquial question
    # recalls nothing raw; the two-intent question loses its database half. The
    # real rewrites recover what's missing, and the union keeps it.
    colloquial = run("colloquial")
    assert colloquial["raw_recall"] < 2                       # raw misses at least the batch doc
    assert colloquial["rewrite_recall"] == 2                  # the rewrites recover both
    assert colloquial["rewrite_recall"] > colloquial["raw_recall"]
    assert colloquial["union_recall"] == colloquial["rewrite_recall"]

    two = run("two_intents")
    assert two["raw_recall"] >= 1                             # the Kafka half hits raw
    assert two["rewrite_recall"] == 2                         # the split recovers both
    assert two["union_recall"] == 2


def test_precise_query_rewrite_erodes_the_margin_and_union_restores_it():
    # The honest version of "the rewrite drifts": at temperature 0 the real
    # rewriter never lost the doc outright, but rewriting a precise query
    # reliably BURNS similarity margin - the best rewrite scores the gold doc
    # lower than the raw query does. The union puts the raw margin back for free.
    r = run("precise")
    assert r["raw_recall"] == 1                               # the raw query nails it
    assert r["raw_margin"] > r["rewrite_margin"]              # the rewrite eroded it
    assert r["raw_margin"] >= SIM_THRESHOLD                   # raw is a genuine hit
    assert r["union_recall"] == 1                             # the union keeps the doc


if __name__ == "__main__":
    for name in CASES:
        print(f"\n{'=' * 60}\n{name}: {CASES[name]['question']}\n{'=' * 60}")
        result = run(name)
        print(f"   recall  raw {result['raw_recall']}  ->  rewrite-only "
              f"{result['rewrite_recall']}  ->  union {result['union_recall']}"
              f"   (gold {len(CASES[name]['gold'])})")
        print(f"   margin on gold[0]: raw {result['raw_margin']:.3f}  vs  "
              f"best rewrite {result['rewrite_margin']:.3f}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
