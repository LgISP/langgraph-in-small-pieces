"""Chapter 27 Challenge - Silver: make the keep rule less brittle.

The chapter's rerank keeps a candidate on ONE hard cutoff (`score >= 6`). That
number is a cliff: a genuinely relevant doc scored a 5 gets dropped, and a run of
stingy scoring can empty a shortlist that should have kept something. This replaces
the lone cutoff with a two-part rule:

    FLOOR  drop any candidate scoring below 3   (is anything actually good?)
    CAP    of what clears the floor, keep at most the top 3

    def floor_and_cap(scored, floor=3, cap=3):
        ranked = sorted(scored, reverse=True)
        above = [(s, d) for s, d in ranked if s >= floor]
        return [d for s, d in above[:cap]]

The floor and the cap answer different questions, and which one fires is the whole
lesson:

  - Kafka question    -> two docs score high, both clear the floor, cap doesn't bite
                         -> keep 2.
  - database question -> only MySQL scores high, the rest sit below the floor
                         -> keep 1.
  - decorator question-> EVERY candidate scores 0-1, the FLOOR fires and keeps 0,
                         so the agent's empty-shortlist web fallback still triggers.

The decorator case is why the floor must exist. A cap-only "keep the top 3" rule
has no notion of "good": it would keep the three least-irrelevant junk docs, hand
the agent a non-empty shortlist, and rob it of the fallback - the base can't answer
the question, but the agent would never find out. `floor_and_cap` degrades to an
empty list exactly when nothing is good enough, which is the behaviour the fallback
depends on. And "keep the best few, but only if any are actually good" survives the
score wobble a lone `>= 6` does not: two Kafka docs that slip from 8 to 5 still
clear a floor of 3, where a `>= 6` cutoff would drop both.

The router and the scorer are the book's real local model (Qwen3.8 behind
LM Studio), exactly as in code/ch27_rerank.py; retrieval is the chapter's
"over-fetch everything" recall stage over an in-memory store. Real scores wobble
where a fake's cannot, so the tests assert the *behaviour* of the keep rule —
which docs survive, which path the agent then takes — plus the rule's pure
function on synthetic scores; they never assert an exact model-issued integer.
(For the record, one observed run scored kafka [10,8,0,0,0], database
[0,0,0,0,9], decorator [0,0,0,0,0] — the same shape the chapter's transcript
shows.)

Run:  python code/answers/ch27_floor_and_cap.py   (see answers/README.md)
"""

import warnings
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


# The book's real local model — the chapter's local-server setup.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


def excerpt(text: str, width: int = 140) -> str:
    """First line of a model answer, cut at a word boundary, with '...' if cut."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) <= width and first == text.strip():
        return first
    cut = first[:width].rsplit(" ", 1)[0] if len(first) > width else first
    return cut.rstrip(" ,;:") + " ..."


# --- the keep rule, as pure functions so both halves are directly testable ---
FLOOR = 3   # drop any candidate scoring below this
CAP = 3     # of what clears the floor, keep at most this many


def floor_and_cap(scored: list, floor: int = FLOOR, cap: int = CAP) -> list:
    """The two-part keep rule: filter by the floor, then cap the survivors."""
    ranked = sorted(scored, key=lambda sd: -sd[0])
    above = [(s, d) for s, d in ranked if s >= floor]
    return [d for _, d in above[:cap]]


def cap_only(scored: list, cap: int = CAP) -> list:
    """The brittle alternative the challenge warns against: a top-N rule with no
    floor. It always keeps `cap` docs when that many exist - even pure junk."""
    ranked = sorted(scored, key=lambda sd: -sd[0])
    return [d for _, d in ranked[:cap]]


def hard_cutoff(scored: list, cutoff: int = 6) -> list:
    """The chapter's original single-cutoff rule, kept here for contrast."""
    ranked = sorted(scored, key=lambda sd: -sd[0])
    return [d for s, d in ranked if s >= cutoff]


# --- the in-memory knowledge base (five one-line docs) -----------------------
DOCUMENTS = [
    "Kafka batch consumption: set spring.kafka.listener.type=batch to enable batch consumption mode.",
    "Kafka tuning: batch consumers need max-poll-records tuned to raise consumer throughput.",
    "Redis connection pool: configure pool parameters through the spring.data.redis.* settings.",
    "Spring Boot auto-configuration: @SpringBootApplication triggers the auto-wiring mechanism.",
    "MySQL indexing: a B+ tree index on frequently queried columns greatly speeds up slow queries.",
]

class Decision(BaseModel):
    route: Literal["search", "web", "generate"] = Field(description="the next action")
    reason: str = Field(description="a short justification")


class Relevance(BaseModel):
    score: int = Field(description="relevance to the question, 0 (irrelevant) to 10 (perfect)")


decider = llm.with_structured_output(Decision, method="json_schema")
scorer = llm.with_structured_output(Relevance, method="json_schema")


# --- the graph: agent -> search_docs -> rerank -> agent, with a web fallback --
class State(TypedDict):
    question: str
    route: NotRequired[str]
    docs: NotRequired[list]
    web_results: NotRequired[list]
    iterations: NotRequired[int]
    answer: NotRequired[str]
    reason: NotRequired[str]


TRACE: list = []        # node visit order this run (for the tests)
LAST_SCORED: list = []  # the (score, doc) pairs the last rerank produced


def agent_node(state: State) -> dict:
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    prompt = f"""
        You are the router for a question-answering agent.

        User question: {state['question']}
        Local documents found: {len(docs)}
        Web results found: {len(web)}
        Local knowledge base already searched: {"docs" in state}
        Web already searched: {"web_results" in state}

        Rules — read them in order and act on the FIRST whose condition is true; once a
        rule matches, ignore every rule below it.
        1. If you ALREADY have at least one source (local documents found > 0 OR web
           results found > 0), you have enough -> generate. Do NOT gather more: one
           source is enough, even if the web has not been searched yet.
        2. else if the local knowledge base was NOT searched yet -> search
        3. else if the web was NOT searched yet -> web
        4. otherwise -> generate
        Treat "already searched" as final: searching again cannot change a count that is
        0, so never repeat a search that was already done."""
    TRACE.append("agent")
    d = decider.invoke(prompt)
    print(f"[agent] route={d.route}  ({excerpt(d.reason)})")
    return {"route": d.route, "reason": d.reason, "iterations": state.get("iterations", 0) + 1}


def search_docs_node(state: State) -> dict:
    # RECALL: over-fetch on purpose - hand every doc to rerank, no threshold.
    TRACE.append("search_docs")
    candidates = list(DOCUMENTS)
    print(f"   [search_docs] loosely recalled {len(candidates)} candidate(s) (rerank will filter)")
    return {"docs": candidates}


def rerank_node(state: State) -> dict:
    # PRECISION: score each candidate, then keep by FLOOR + CAP (not one cutoff).
    TRACE.append("rerank")
    question = state["question"]
    candidates = state.get("docs", [])
    print(f"   [rerank] scoring {len(candidates)} candidate(s):")

    scored = []
    for doc in candidates:
        prompt = f"""
            Score how relevant the document is to answering the question, from 0 to 10.
            Question: {question}
            Document: {doc}"""
        r = scorer.invoke(prompt)
        scored.append((r.score, doc))
        print(f"        {r.score:>2}/10  {doc[:48]}")

    LAST_SCORED[:] = scored
    kept = floor_and_cap(scored)
    print(f"   [rerank] kept {len(kept)} (score >= {FLOOR}, top {CAP})")
    return {"docs": kept}   # overwrite docs with the shortlist (may be empty)


def web_search_node(state: State) -> dict:
    TRACE.append("web_search")
    print("   [web_search] (stub) falling back to the web")
    return {"web_results": [f"[web] community discussion about '{state['question']}'"]}


def generate_node(state: State) -> dict:
    TRACE.append("generate")
    sources = state.get("docs", []) + state.get("web_results", [])
    sources_text = "\n".join(f"- {s}" for s in sources) or "(no sources)"
    prompt = (f"Answer the user's question concisely, using only these sources:\n"
              f"{sources_text}\n\nQuestion: {state['question']}")
    answer = llm.invoke(prompt).content.strip()
    print(f"   [generate] {excerpt(answer)}")
    return {"answer": answer}


def route_after_agent(state: State) -> str:
    return state["route"]


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("rerank", rerank_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", route_after_agent,
    {"search": "search_docs", "web": "web_search", "generate": "generate"},
)
builder.add_edge("search_docs", "rerank")     # recall -> precision
builder.add_edge("rerank", "agent")           # hand the shortlist back to the agent
builder.add_edge("web_search", "agent")
builder.add_edge("generate", END)
graph = builder.compile()


Q_KAFKA = "How do I configure Kafka batch consumers?"
Q_DATABASE = "How do I speed up slow database queries?"
Q_DECORATOR = "How do I write a Python decorator?"


# Each question is asked ONCE and the (result, trace, scores) shared by every
# test — real model calls are slow, and the assertions describe one run.
_RUNS: dict = {}


def run(question: str):
    if question not in _RUNS:
        TRACE.clear()
        result = graph.invoke({"question": question}, config={"recursion_limit": 15})
        _RUNS[question] = (result, list(TRACE), list(LAST_SCORED))
    return _RUNS[question]


# --- tests -------------------------------------------------------------------
# Real-model note: the model issues the scores, so no test pins an exact integer
# from it. The graph tests assert which docs SURVIVED the keep rule and which
# path the agent then took; the keep rule itself is pinned by the synthetic-score
# tests below.
def test_kafka_keeps_the_two_kafka_docs():
    result, trace, scored = run(Q_KAFKA)
    # Both Kafka docs cleared the floor, nothing else did; the cap didn't bite.
    assert set(result["docs"]) == {DOCUMENTS[0], DOCUMENTS[1]}
    # And the graph answers from the base without ever touching the web.
    assert trace == ["agent", "search_docs", "rerank", "agent", "generate"]
    assert "web_search" not in trace


def test_database_keeps_mysql_on_top_and_skips_the_web():
    result, trace, scored = run(Q_DATABASE)
    kept = result["docs"]
    assert DOCUMENTS[4] in kept                       # MySQL cleared the floor
    assert kept[0] == DOCUMENTS[4]                    # ...and ranked first
    assert DOCUMENTS[0] not in kept and DOCUMENTS[1] not in kept   # kafka docs didn't
    assert "web_search" not in trace                  # a good doc is enough


def test_decorator_keeps_zero_so_the_web_fallback_still_triggers():
    result, trace, scored = run(Q_DECORATOR)
    # The FLOOR fires: no candidate was good, the shortlist is empty...
    assert all(score < FLOOR for score, _ in scored)
    # ...so the agent's existing routing falls back to the web.
    assert "web_search" in trace
    assert result["web_results"]
    assert trace == ["agent", "search_docs", "rerank", "agent",
                     "web_search", "agent", "generate"]


def test_the_floor_not_the_cap_is_what_saves_the_fallback():
    # On the decorator run's REAL scores, cap-only would keep 3 junk docs and hand
    # the agent a non-empty shortlist - killing the fallback. floor_and_cap keeps 0.
    _, _, scored = run(Q_DECORATOR)
    assert len(cap_only(scored)) == 3                 # top-N alone keeps junk...
    assert floor_and_cap(scored) == []                # ...the floor throws it all out


def test_cap_trims_when_more_than_three_clear_the_floor():
    # Five docs all comfortably above the floor: the CAP is what bites now.
    scored = [(9, "a"), (8, "b"), (7, "c"), (6, "d"), (5, "e")]
    kept = floor_and_cap(scored)
    assert kept == ["a", "b", "c"]                    # top 3 by score, the rest dropped


def test_floor_and_cap_survives_score_wobble_a_hard_cutoff_does_not():
    # Two genuinely relevant docs slip from 7 to 5 on a stingy run. A >= 6 cutoff
    # drops BOTH; the floor of 3 keeps them - robust where a lone cutoff is brittle.
    wobbled = [(5, "kafka_batch"), (5, "kafka_tuning"), (0, "x"), (1, "y"), (0, "z")]
    assert hard_cutoff(wobbled, cutoff=6) == []       # the cliff swallows the good docs
    assert set(floor_and_cap(wobbled)) == {"kafka_batch", "kafka_tuning"}


if __name__ == "__main__":
    for q in (Q_KAFKA, Q_DATABASE, Q_DECORATOR):
        print(f"\n{'=' * 60}\nQuestion: {q}\n{'=' * 60}")
        result, trace, _ = run(q)
        print(f"--- kept {len(result['docs'])} doc(s); path {trace}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
