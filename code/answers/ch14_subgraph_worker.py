"""Chapter 14 Challenge — Bronze: turn a worker into a real isolated sub-agent.

In the chapter, `docs_agent` was a plain node sharing the parent `State`. Here it
becomes a genuine **subgraph** (Chapter 9): a two-step pipeline — `rewrite` the
query, then `search` — with its own `RagState` that carries a private `rewritten`
field the parent never declares. An adapter node (the Chapter 9 schema-mapping
pattern) maps the supervisor's `question` INTO the subgraph and folds the subgraph's
`docs` back OUT as one `[docs_agent] ...` report.

The payoff is the isolation the chapter's Deep Dive described: the sub-agent grinds
through its own messy middle (`rewritten`) in its own state space, and only the
conclusion crosses back. We prove it — `rewritten` never appears in the supervisor's
state. That is the on-demand-sub-agent shape for real, not by analogy.

The supervisor routes with the chapter's REAL model (`openai/gpt-oss-20b` behind
LM Studio) and the chapter's exact prompt — including the load-bearing "repeat the
two Already-delegated values" line that keeps a reasoning model from bailing (see
ch14_scratchpad_line.py). More of that prompt is load-bearing than the chapter
let on: a first draft of this file shortened only the INTRO sentence ("You
delegate to ONE worker at a time, read its report, then decide again" became "of
a documentation team") and gpt-oss started re-delegating to docs_agent at
reports=1 while *claiming* rule 1 applied. `finish` stitches the reports together
without a model.
The tests assert the run's structure — which reports exist, and that the private
field never crossed — not the router's wording.

Run:  python code/answers/ch14_subgraph_worker.py   (see answers/README.md)
"""

import operator
import warnings
from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


# The chapter's real local model — gpt-oss, as in code/ch14_supervisor.py.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="openai/gpt-oss-20b",            # gpt-oss on purpose — the chapter's pick
    temperature=0,                         # 0 = as reproducible as a real model gets
)


def excerpt(text: str, width: int = 140) -> str:
    """First line of a model answer, cut at a word boundary, with '...' if cut."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) <= width and first == text.strip():
        return first
    cut = first[:width].rsplit(" ", 1)[0] if len(first) > width else first
    return cut.rstrip(" ,;:") + " ..."


# --- the docs_agent SUBGRAPH: its own state, a private `rewritten` field -------
class RagState(TypedDict):
    question: str                   # input, mapped in from the parent's question
    rewritten: NotRequired[str]     # PRIVATE: the parent never declares this key
    docs: NotRequired[list[str]]    # output, folded back out as a report


KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ],
    "redis": [
        "Redis Manual: set the connection pool via spring.data.redis.*.",
    ],
}


def rewrite_node(state: RagState) -> dict:
    # Step 1: expand the query. This scratch work must NOT leak to the supervisor.
    rewritten = f"{state['question'].lower()} (expanded: + synonyms/context)"
    print(f"      [rag/rewrite] rewrote query: {rewritten}")
    return {"rewritten": rewritten}


def search_node(state: RagState) -> dict:
    # Step 2: search using the rewritten query.
    rewritten = state["rewritten"]
    hits: list[str] = []
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in rewritten:
            hits.extend(docs)
    print(f"      [rag/search] searched with rewritten query: {len(hits)} hit(s)")
    return {"docs": hits}


def _build_docs_subgraph():
    b = StateGraph(RagState)
    b.add_node("rewrite", rewrite_node)
    b.add_node("search", search_node)
    b.add_edge(START, "rewrite")
    b.add_edge("rewrite", "search")
    b.add_edge("search", END)
    return b.compile()


docs_subgraph = _build_docs_subgraph()


# --- the routing model: the chapter's typed Delegate over the real model ------
class Delegate(BaseModel):
    """The supervisor's typed decision — shape guaranteed by the Literal `route`."""

    # reason comes FIRST: fields are emitted in schema order, so this is the only
    # place gpt-oss gets to think under LM Studio's JSON grammar (see the chapter).
    reason: str = Field(
        description="Think step by step about which rule applies, then decide."
    )
    route: Literal["docs_agent", "web_agent", "finish"] = Field(
        description="Delegate to docs_agent / web_agent, or finish."
    )


router = llm.with_structured_output(Delegate, method="json_schema")


# --- the parent supervisor graph ----------------------------------------------
class State(TypedDict):
    question: str                                # note: no `rewritten` key here
    route: NotRequired[str]
    reason: NotRequired[str]
    reports: Annotated[list[str], operator.add]
    answer: NotRequired[str]


# Captured at finish so a test can prove the private field never crossed back.
FINISH_STATE_KEYS: list[str] = []


def supervisor_node(state: State) -> dict:
    reports = state.get("reports", [])
    searched_docs = any(r.startswith("[docs_agent]") for r in reports)
    searched_web = any(r.startswith("[web_agent]") for r in reports)
    reports_text = "\n".join(f"- {r}" for r in reports) or "(none yet)"
    prompt = f"""
        You are the SUPERVISOR of a documentation question-answering team
        with two workers. You delegate to ONE worker at a time, read its report, then
        decide again.

        Question: {state['question']}
        Already delegated to docs_agent: {searched_docs}
        Already delegated to web_agent: {searched_web}

        Worker reports so far:
        {reports_text}

        Read the rules in order and choose the FIRST whose condition is true:
        1. If docs_agent has NOT run yet                                  -> docs_agent
        2. else if the docs came back empty AND web_agent has NOT run yet -> web_agent
        3. otherwise                                                      -> finish

        In `reason`, first repeat the two "Already delegated" values, then state which
        rule applied."""
    decision = router.invoke(prompt)
    print(f"[supervisor] reports={len(reports)} -> {decision.route}  ({excerpt(decision.reason)})")
    return {"route": decision.route, "reason": decision.reason}


def docs_agent_node(state: State) -> dict:
    # The ADAPTER (Chapter 9): map parent -> child in, invoke the subgraph, map
    # child -> parent out. The subgraph's private `rewritten` never touches State.
    child_out = docs_subgraph.invoke({"question": state["question"]})
    docs = child_out.get("docs", [])
    if docs:
        report = f"[docs_agent] {len(docs)} internal doc(s): " + "; ".join(docs)
    else:
        report = "[docs_agent] found nothing in the internal docs"
    print(f"   [docs_agent] subgraph returned {len(docs)} doc(s); "
          f"child keys were {sorted(child_out)}")
    return {"reports": [report]}       # only the conclusion crosses back


def web_agent_node(state: State) -> dict:
    print("   [web_agent] searched the open web")
    report = f"[web_agent] 1 web result: community thread about '{state['question']}'"
    return {"reports": [report]}


def finish_node(state: State) -> dict:
    FINISH_STATE_KEYS[:] = sorted(state)          # snapshot the supervisor's keys
    print(f"[finish] supervisor state keys: {FINISH_STATE_KEYS}")
    reports = state.get("reports", [])
    answer = "Answer from reports:\n" + "\n".join(reports) if reports else "(no reports)"
    return {"answer": answer}


def route_after_supervisor(state: State) -> str:
    return state["route"]


builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("docs_agent", docs_agent_node)
builder.add_node("web_agent", web_agent_node)
builder.add_node("finish", finish_node)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_after_supervisor, {
    "docs_agent": "docs_agent", "web_agent": "web_agent", "finish": "finish",
})
builder.add_edge("docs_agent", "supervisor")
builder.add_edge("web_agent", "supervisor")
builder.add_edge("finish", END)
graph = builder.compile()


# Each question runs ONCE; the (result, finish-keys) pair is shared by the tests.
_RUNS: dict = {}


def run(question: str):
    if question not in _RUNS:
        result = graph.invoke({"question": question}, {"recursion_limit": 12})
        _RUNS[question] = (result, list(FINISH_STATE_KEYS))
    return _RUNS[question]


# --- tests -------------------------------------------------------------------
# Real-model note: the router's reasons vary in wording; the reports and the
# state keys — the structure — must not.
def test_subgraph_worker_answers_a_docs_hit():
    result, _ = run("How do I configure Kafka batch consumers?")
    assert len(result["reports"]) == 1
    assert result["reports"][0].startswith("[docs_agent] 2 internal doc(s):")
    assert "answer" in result


def test_private_rewritten_field_never_leaks():
    result, finish_keys = run("How do I configure Kafka batch consumers?")
    # The Deep Dive's promise: the sub-agent's scratch work stays sealed.
    assert "rewritten" not in result                  # not in the returned state
    assert "rewritten" not in finish_keys             # not seen by the supervisor
    assert "docs" not in result                       # the child's output key, too


def test_web_fallback_when_docs_subgraph_finds_nothing():
    result, _ = run("How do I write a Python decorator?")
    assert result["reports"][0] == "[docs_agent] found nothing in the internal docs"
    assert result["reports"][1].startswith("[web_agent] 1 web result")
    assert len(result["reports"]) == 2


if __name__ == "__main__":
    result, finish_keys = run("How do I configure Kafka batch consumers?")
    print("--- answer ---")
    print(result["answer"])
    print("rewritten leaked to supervisor?", "rewritten" in finish_keys)

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
