"""Chapter 10 Challenge — Silver: grade the docs, don't just count them.

The chapter's agent falls back to the web only when `search_docs` returns *zero*
hits. But a keyword hit isn't an answer: docs can mention "Redis" and still not
address the question. This uses `with_structured_output` a SECOND time to tell the
difference — the document-relevance-scoring idea RAG leans on in Part III, built
entirely from this chapter's pattern.

A `Grade` model (`relevant: Literal["yes", "no"]` + a one-line reason) and a
`grade_docs` node judge, after a non-empty search, whether the retrieved docs
actually answer the question. Grade "no" routes to `web_search` even though docs
were found; grade "yes" goes on to `generate`. Two questions whose keywords BOTH
hit the knowledge base take different paths on the strength of a *judgment*, not a
count — and we added one node and one schema, touching no edge we didn't mean to.

The judging model is the book's real local model (Qwen3.8 behind LM Studio),
the same one `code/ch10_real_llm.py` routes with — no fake stands in for it. The
tests therefore assert the *structure* of each run (the path taken, the grade's
legality) rather than any verbatim model text. One real-model lesson surfaced
immediately: the grader can only judge docs it can SEE, so the grading prompt must
actually include the retrieved documents. (An earlier offline version of this file
built a grading prompt that ended at "Retrieved documents:" with nothing after it —
a bug its fake model papered over by parsing state the real model never gets.)

Run:  python code/answers/ch10_doc_grader.py   (see answers/README.md)
"""

import warnings
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# with_structured_output makes pydantic emit one harmless serializer warning about
# an internal field; the parsed result is correct. Silence just that one line.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


# The book's real local model — the chapter's local-server setup.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


# --- the two schemas ---------------------------------------------------------
class Decision(BaseModel):
    """Choose the next action for a documentation question-answering agent."""

    route: Literal["search", "web", "generate"] = Field(
        description="Next action: search=look in the local docs, "
        "web=search the open web, generate=write the answer now."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


class Grade(BaseModel):
    """Judge whether the retrieved documents actually answer the question."""

    relevant: Literal["yes", "no"] = Field(
        description="yes = the documents answer the question; "
        "no = they only mention the topic without addressing it."
    )
    reason: str = Field(description="One short sentence explaining the judgment.")


decider = llm.with_structured_output(Decision, method="json_schema")
grader = llm.with_structured_output(Grade, method="json_schema")


def excerpt(text: str, width: int = 140) -> str:
    """First line of a model answer, cut at a word boundary, with '...' if cut."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) <= width and first == text.strip():
        return first
    cut = first[:width].rsplit(" ", 1)[0] if len(first) > width else first
    return cut.rstrip(" ,;:") + " ..."


# --- the doc-bot, with a grade_docs node added between search and the agent ---
class State(TypedDict):
    question: str                       # input, set by the caller
    route: NotRequired[str]             # written by agent AND grade_docs
    docs: NotRequired[list[str]]        # written by search_docs (may be empty)
    web_results: NotRequired[list[str]] # written by web_search
    iterations: NotRequired[int]        # loop counter the agent keeps (Ch4)
    reason: NotRequired[str]            # the agent's stated reason (observability)
    relevant: NotRequired[str]          # the grade: yes|no
    grade_reason: NotRequired[str]      # the grader's stated reason
    answer: NotRequired[str]            # written by generate


# One topic the docs genuinely answer (kafka) and one they only mention (redis):
# both keywords hit, so a count alone can't tell them apart — a grade can.
KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: enable batch consumption with spring.kafka.listener.type=batch.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ],
    "redis": [
        "Redis Manual: configure the connection pool via spring.data.redis.pool.",
    ],
}

TRACE: list[str] = []   # node visit order this run (for the tests)


def agent_node(state: State) -> dict:
    TRACE.append("agent")
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    searched_docs = "docs" in state
    searched_web = "web_results" in state

    # The chapter's hardened ordered rules (see code/ch10_real_llm.py): the real
    # model needs to be told that one source is enough and "searched" is final.
    prompt = f"""
        You are the router for a documentation question-answering agent.
        Decide the single next action from the state below.

        Question: {state['question']}
        Local documents found so far: {len(docs)}
        Web results so far: {len(web)}
        Already searched the local docs: {searched_docs}
        Already searched the web: {searched_web}

        Rules — read them in order and act on the FIRST whose condition is true; once a
        rule matches, ignore every rule below it.
        1. If you ALREADY have at least one source (local documents found > 0 OR web
           results > 0), you have enough -> generate. Do NOT gather more: one source is
           enough, even if the web has not been searched yet.
        2. else if (already searched the local docs is False)            -> search
        3. else if (already searched the web is False)                   -> web
        4. otherwise                                                     -> generate
        Treat "already searched" as final: searching again cannot change a count that is
        0, so never repeat a search that was already done."""

    decision = decider.invoke(prompt)
    print(f"[agent] docs={len(docs)} web={len(web)} -> {decision.route}"
          f"  ({excerpt(decision.reason)})")
    return {
        "route": decision.route,
        "reason": decision.reason,
        "iterations": state.get("iterations", 0) + 1,
    }


def search_docs_node(state: State) -> dict:
    TRACE.append("search_docs")
    q = state["question"].lower()
    hits: list[str] = []
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in q:
            hits.extend(docs)
    print(f"[search_docs] {len(hits)} hit(s)")
    return {"docs": hits}


def grade_docs_node(state: State) -> dict:
    # The new step: after a search, judge whether the docs answer the question. A
    # count would send any non-empty result to generate; the grade can overrule it.
    TRACE.append("grade_docs")
    docs = state.get("docs", [])
    if not docs:
        # Nothing to grade — a plain miss, handled like the chapter: go to the web.
        print("[grade_docs] 0 docs -> web")
        return {"route": "web", "relevant": "no", "grade_reason": "no docs retrieved"}

    # The grader can only judge docs it can SEE — they must be in the prompt.
    docs_text = "\n".join(f"- {d}" for d in docs)
    prompt = f"""
        Do the retrieved documents actually answer the question, or do they
        only mention the topic without addressing it?

        Question: {state['question']}
        Retrieved documents:
        {docs_text}"""

    grade = grader.invoke(prompt)
    print(f"[grade_docs] relevant={grade.relevant}  ({excerpt(grade.reason)})")
    if grade.relevant == "yes":
        return {"route": "generate", "relevant": "yes", "grade_reason": grade.reason}
    # Keyword hit but not an answer: drop the irrelevant docs and fall to the web.
    return {"route": "web", "relevant": "no", "grade_reason": grade.reason, "docs": []}


def web_search_node(state: State) -> dict:
    TRACE.append("web_search")
    print("[web_search] falling back to a web search")
    return {"web_results": [f"[web] thread about {state['question']!r}"]}


def generate_node(state: State) -> dict:
    TRACE.append("generate")
    sources = state.get("docs", []) + state.get("web_results", [])
    sources_text = "\n".join(f"- {s}" for s in sources) or "(no sources)"
    prompt = (f"Answer the user's question concisely, using only these sources:\n"
              f"{sources_text}\n\nQuestion: {state['question']}")
    answer = llm.invoke(prompt).content.strip()
    print(f"[generate] {excerpt(answer)}")
    return {"answer": answer}


def pick_route(state: State) -> str:
    return state["route"]


# The graph: Chapter 10's structured doc-bot, plus one grade_docs node wired
# between search_docs and its own generate/web branch. Every other edge is unchanged.
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("grade_docs", grade_docs_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", pick_route,
    {"search": "search_docs", "web": "web_search", "generate": "generate"},
)
builder.add_edge("search_docs", "grade_docs")          # always grade after a search
builder.add_conditional_edges(
    "grade_docs", pick_route,
    {"generate": "generate", "web": "web_search"},      # the JUDGMENT branch
)
builder.add_edge("web_search", "agent")
builder.add_edge("generate", END)
graph = builder.compile()


Q_RELEVANT = "How do I configure Kafka batch consumers?"    # docs truly answer it
Q_IRRELEVANT = "How do I monitor Redis memory usage?"       # keyword hits, docs don't answer
Q_MISS = "How do I write a Python decorator?"               # no keyword hit at all

# Each question is asked ONCE and the (result, trace) shared by every test —
# real model calls are slow, and the assertions are about one run's structure.
_RUNS: dict[str, tuple[dict, list[str]]] = {}


def run(question: str) -> tuple[dict, list[str]]:
    if question not in _RUNS:
        TRACE.clear()
        result = graph.invoke({"question": question}, {"recursion_limit": 15})
        _RUNS[question] = (result, list(TRACE))
    return _RUNS[question]


# --- tests -------------------------------------------------------------------
# Real-model note: these assert routing structure and grade legality, never
# verbatim model text — the answers vary in wording, the paths must not.
def test_relevant_docs_go_straight_to_generate():
    result, trace = run(Q_RELEVANT)
    assert trace == ["agent", "search_docs", "grade_docs", "generate"]
    assert result["relevant"] == "yes"
    assert "web_search" not in trace               # never fell through to the web
    assert result["docs"]                          # answer was built from the docs
    assert result["answer"]


def test_keyword_hit_but_irrelevant_falls_through_to_web():
    result, trace = run(Q_IRRELEVANT)
    # docs were FOUND (redis matched) but graded "no", so the path detours to the web
    assert trace == ["agent", "search_docs", "grade_docs",
                     "web_search", "agent", "generate"]
    assert result["relevant"] == "no"
    assert result["web_results"]                   # the web caught it
    assert result["docs"] == []                    # the irrelevant docs were dropped


def test_total_miss_also_uses_web():
    result, trace = run(Q_MISS)
    assert "grade_docs" in trace and "web_search" in trace
    assert result["relevant"] == "no"              # 0 docs -> nothing to grade
    assert result["web_results"]


def test_grade_is_always_a_legal_shape():
    for q in (Q_RELEVANT, Q_IRRELEVANT, Q_MISS):
        result, _ = run(q)
        assert result["relevant"] in {"yes", "no"}   # the Literal held every time


if __name__ == "__main__":
    for q in (Q_RELEVANT, Q_IRRELEVANT, Q_MISS):
        print(f"\n===== question: {q} =====")
        result, trace = run(q)
        print(f"grade={result['relevant']!r} path={trace}")
        print("answer:", excerpt(result["answer"]))

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
