"""Chapter 27 - Query rewriting: fix the query before you retrieve.

A real user's question is often colloquial, vague, or asks several things at
once. Embedded as-is, it retrieves badly:

    - No technical keywords -> the relevant document's similarity stays low and
      falls under the threshold, so recall misses it.
    - Several intents in one sentence -> the embedding averages them into a
      blurry vector that matches none of them well, so half the question is lost.

Reranking (the sibling technique in this chapter) cleans up AFTER retrieval - it
throws out bad candidates. Query rewriting works BEFORE retrieval - it repairs a
bad query so retrieval has a chance:

    - expand synonyms and spell out the implied technical context, and
    - split a multi-intent question into separate sub-queries, one intent each,
      then retrieve for every sub-query and union the hits.

To prove it helps, search_docs runs an A/B for each question: it retrieves once
with the RAW question and once with the REWRITTEN queries, and measures recall
against a known set of relevant documents (the "gold" set). Recall before vs
after tells the story.

The knowledge base is a handful of English docs embedded by the local model server
and stored in Chroma, rebuilt on every run so this file is self-contained. To
keep the spotlight on rewriting we use the shortest possible graph - no agent
decision loop, just rewrite -> search_docs -> generate.

Run:  python code/ch27_query_rewrite.py
"""

import os
import shutil
import warnings

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from typing import TypedDict

from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph

# Local models served by LM Studio (OpenAI-compatible API). Nothing here talks to
# a paid cloud. timeout is generous because a local server may be busy.
LM_BASE = "http://localhost:1234/v1"

llm = ChatOpenAI(
    base_url=LM_BASE,
    api_key="lm-studio",
    model="qwen/qwen3.8-27b",
    temperature=0,
    reasoning_effort="none",
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


embeddings = NomicEmbeddings(
    base_url=LM_BASE,
    api_key="lm-studio",
    model="text-embedding-nomic-embed-text-v1.5",
    check_embedding_ctx_length=False,
    timeout=240,
)

DOCUMENTS = [
    "Kafka batch consumption: set spring.kafka.listener.type=batch to enable batch consumption mode.",
    "Kafka tuning: batch consumers need max-poll-records tuned to raise consumer throughput.",
    "Redis connection pool: configure pool parameters through the spring.data.redis.* settings.",
    "Spring Boot auto-configuration: @SpringBootApplication triggers the auto-wiring mechanism.",
    "MySQL indexing: a B+ tree index on frequently queried columns greatly speeds up slow queries.",
]

# Rebuild a fresh Chroma store next to this file on every run (self-contained).
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_ch27_rewrite")
shutil.rmtree(DB_DIR, ignore_errors=True)
vectorstore = Chroma(
    collection_name="doc_bot_rewrite",
    embedding_function=embeddings,
    persist_directory=DB_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)
vectorstore.add_texts(DOCUMENTS)
print(f"[DB] built knowledge base with {len(DOCUMENTS)} documents at {DB_DIR}")


SIM_THRESHOLD = 0.60   # similarity (= 1 - distance) below this counts as a miss


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


class State(TypedDict):
    question: str
    gold: list        # the documents this question SHOULD recall (to measure recall)
    queries: list     # the rewritten search queries
    docs: list        # what finally goes to generate
    answer: str


def retrieve(query: str) -> list:
    """Return (similarity, document) pairs whose similarity clears the threshold."""
    results = vectorstore.similarity_search_with_score(query, k=5)
    return [(1 - dist, doc.page_content) for doc, dist in results if (1 - dist) >= SIM_THRESHOLD]


def covered_gold(hit_docs: list, gold: list) -> list:
    """Which gold markers (a keyword slice of each relevant doc) got recalled."""
    return [g for g in gold if any(g in d for d in hit_docs)]


def rewrite_node(state: State) -> dict:
    q = state["question"]
    r = rewriter.invoke(REWRITE_PROMPT.format(q=q))
    print("   [rewrite] rewrote the raw question into search queries:")
    for sq in r.queries:
        print(f"        - {sq}")
    return {"queries": r.queries}


# ---------------------------------------------------------------------------
# search_docs: the A/B. Retrieve with the raw question, then with the rewritten
# queries (union), and compare recall against the gold set.
# ---------------------------------------------------------------------------
def search_docs_node(state: State) -> dict:
    q = state["question"]
    gold = state["gold"]

    # BEFORE: retrieve with the raw question.
    before = retrieve(q)
    before_docs = [d for _, d in before]
    before_hit = covered_gold(before_docs, gold)
    print(f"   [before] raw question, hits (sim >= {SIM_THRESHOLD}): {len(before)}")
    for sim, d in before:
        print(f"        {sim:.3f}  {d}")
    print(f"            -> recalled {len(before_hit)}/{len(gold)} relevant docs")

    # AFTER: retrieve with every rewritten query and union the hits.
    after = {}
    for sq in state["queries"]:
        for sim, d in retrieve(sq):
            after[d] = max(after.get(d, 0.0), sim)
    # after is {document: best similarity}; the minus sorts high scores first
    after_sorted = sorted(after.items(), key=lambda kv: -kv[1])
    after_docs = [d for d, _ in after_sorted]
    after_hit = covered_gold(after_docs, gold)
    print(f"   [after]  rewritten queries (union), hits (sim >= {SIM_THRESHOLD}): {len(after_sorted)}")
    for d, sim in after_sorted:
        print(f"        {sim:.3f}  {d}")
    print(f"            -> recalled {len(after_hit)}/{len(gold)} relevant docs")

    verdict = "rewrite recovered the missing docs" if len(after_hit) > len(before_hit) else ""
    print(f"   [compare] recall {len(before_hit)}/{len(gold)}  ->  {len(after_hit)}/{len(gold)}   {verdict}")
    return {"docs": after_docs}


def generate_node(state: State) -> dict:
    sources = state.get("docs", [])
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


# ---------------------------------------------------------------------------
# The graph: the shortest path that shows rewriting - rewrite -> search -> generate.
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("rewrite", rewrite_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "rewrite")
builder.add_edge("rewrite", "search_docs")
builder.add_edge("search_docs", "generate")
builder.add_edge("generate", END)
graph = builder.compile()


if __name__ == "__main__":
    cases = [
        # Colloquial, single intent: no technical keywords, so the raw question
        # misses the Kafka docs; the rewrite adds context and recovers them.
        {
            "question": "Our message queue keeps piling up and the consumer cannot keep up. What should we do?",
            "gold": ["Kafka batch consumption", "Kafka tuning"],
        },
        # One sentence, two intents: the raw embedding averages them and drops the
        # database half; splitting into sub-queries retrieves both.
        {
            "question": "How do I turn on Kafka batch consumption, and also our reports take forever to load from the database?",
            "gold": ["Kafka batch consumption", "MySQL indexing"],
        },
    ]
    for c in cases:
        print(f"\n{'=' * 60}\nQuestion: {c['question']}\n{'=' * 60}")
        result = graph.invoke(c, config={"recursion_limit": 10})
        print("--- answer ---")
        print(excerpt(result["answer"]))
