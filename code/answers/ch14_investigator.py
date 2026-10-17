"""Chapter 14 Challenge — Silver: a supervisor that investigates.

Combine the two halves of the chapter into a supervisor that plans, delegates,
reads, and re-plans until it is satisfied or hits a budget. It gets a COMPOUND
question, a `max_delegations` budget carried in the state, and a richer decision:
each turn it chooses *which* worker AND *what* sub-question to hand off, appending a
running summary so the next decision is informed by every prior report.

The budget is a **programmatic** guard, not a line in the prompt — a number the model
is merely told about is a suggestion a confused model can ignore. It is checked in
code at the top of `supervisor_node`, before the model is ever consulted:

    if len(state.get("reports", [])) >= state["max_delegations"]:
        return {"route": "finish", "reason": "delegation budget reached"}

That is the same real fuse as Chapter 4's `recursion_limit`: enforced by the machine,
not by the model's goodwill.

The planner is a REAL model. Two things the fake version hid, both worth knowing:
the chapter's `openai/gpt-oss-20b` handled the two-field `Delegate` fine but
mis-routed this richer three-field plan (it sent the Kafka sub-question to
web_agent, and half-degenerated the second sub-question), so this file uses the
book's default `qwen/qwen3.8-27b` — worker choice plus sub-question decomposition
needs the stronger router. And the planner needs to be TOLD what the internal
docs cover; a real model cannot smell the knowledge base's contents the way a
keyword-matching fake could. `finish` stitches the reports together without a
model. Tests assert the plan's structure — which workers ran, in what order,
under which budget — never the model's wording.

Run:  python code/answers/ch14_investigator.py   (see answers/README.md)
"""

import operator
import warnings
from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


# The book's real local model — qwen here, not gpt-oss: see the docstring.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch to enable batch consumption.",
    ],
    "redis": [
        "Redis Manual: set the connection pool via spring.data.redis.*.",
    ],
}


# --- the planner: a real model behind a typed three-field plan ----------------
class Delegate(BaseModel):
    """The supervisor's typed plan for one turn — shape fixed by the Literals."""

    reason: str = Field(
        description="First list which parts of the question already have a report "
        "and which do not, then state which rule applied."
    )
    route: Literal["docs_agent", "web_agent", "finish"] = Field(
        description="Which worker to delegate to next, or finish."
    )
    subquestion: str = Field(
        description="The focused sub-question to hand off; empty when finishing."
    )


planner = llm.with_structured_output(Delegate, method="json_schema")


def excerpt(text: str, width: int = 140) -> str:
    """First line of a model answer, cut at a word boundary, with '...' if cut."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) <= width and first == text.strip():
        return first
    cut = first[:width].rsplit(" ", 1)[0] if len(first) > width else first
    return cut.rstrip(" ,;:") + " ..."


# --- state: note the `max_delegations` budget the caller sets -----------------
class State(TypedDict):
    question: str
    max_delegations: int                         # the programmatic budget
    route: NotRequired[str]
    reason: NotRequired[str]
    subquestion: NotRequired[str]
    reports: Annotated[list[str], operator.add]
    answer: NotRequired[str]


def supervisor_node(state: State) -> dict:
    reports = state.get("reports", [])

    # THE PROGRAMMATIC GUARD — checked before the model is ever consulted. Same real
    # fuse as Chapter 4's recursion_limit: the machine stops the loop, not the model.
    if len(reports) >= state["max_delegations"]:
        print(f"[supervisor] budget {state['max_delegations']} reached -> finish")
        return {"route": "finish", "reason": "delegation budget reached"}

    summary = "\n".join(f"- {r}" for r in reports) or "(nothing yet)"
    prompt = f"""
        You are the SUPERVISOR investigating a compound question. Each turn you
        delegate ONE focused sub-question to ONE worker, read the report, then decide again.

        The workers:
        - docs_agent searches the INTERNAL documentation. It covers ONLY these topics:
          kafka, redis. It knows nothing else.
        - web_agent searches the open web. Use it for every topic not in the internal docs.

        Question: {state['question']}

        Learned so far:
        {summary}

        Read the rules in order and choose the FIRST whose condition is true:
        1. If a part of the question is about an internal-docs topic (kafka, redis) and
           has no report yet -> route to docs_agent with that part as the subquestion.
        2. else if a part of the question has no report yet -> route to web_agent with
           that part as the subquestion.
        3. otherwise (every part has a report) -> finish.

        In `reason`, first list which parts already have a report and which do not,
        then state which rule applied."""
    plan = planner.invoke(prompt)
    print(f"[supervisor] reports={len(reports)}/{state['max_delegations']} "
          f"-> {plan.route}  subq={plan.subquestion!r}  ({excerpt(plan.reason)})")
    return {"route": plan.route, "reason": plan.reason, "subquestion": plan.subquestion}


def docs_agent_node(state: State) -> dict:
    subq = state.get("subquestion", state["question"]).lower()
    hits: list[str] = []
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in subq:
            hits.extend(docs)
    if hits:
        report = f"[docs_agent] {state['subquestion']}: " + "; ".join(hits)
    else:
        report = f"[docs_agent] {state['subquestion']}: found nothing in the internal docs"
    print(f"   [docs_agent] searched docs for {state['subquestion']!r}: {len(hits)} hit(s)")
    return {"reports": [report]}


def web_agent_node(state: State) -> dict:
    print(f"   [web_agent] searched the web for {state['subquestion']!r}")
    report = f"[web_agent] {state['subquestion']}: community thread with an example"
    return {"reports": [report]}


def finish_node(state: State) -> dict:
    reports = state.get("reports", [])
    answer = "Investigation summary:\n" + "\n".join(reports) if reports else "(no reports)"
    print(f"[finish] wrote the answer from {len(reports)} report(s)")
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


# A compound question that needs BOTH workers: Kafka is in the docs, the Python
# decorator is not, so the investigator must delegate to docs_agent then web_agent.
COMPOUND_Q = "How do I configure Kafka batch consumers and write a Python decorator?"

# Each (question, budget) pair runs ONCE and is shared by the tests.
_RUNS: dict = {}


def run(budget: int) -> dict:
    if budget not in _RUNS:
        _RUNS[budget] = graph.invoke(
            {"question": COMPOUND_Q, "max_delegations": budget},
            {"recursion_limit": 20})
    return _RUNS[budget]


# --- tests -------------------------------------------------------------------
# Real-model note: the assertions pin the investigation's STRUCTURE — which
# worker wrote which report, how many, and that the guard (not the model) ended
# the run — never the planner's phrasing or the sub-question's exact words.
def test_investigates_both_workers_within_budget():
    result = run(2)
    # Both sub-questions answered by the right worker, in order, then a clean finish.
    assert len(result["reports"]) == 2
    assert result["reports"][0].startswith("[docs_agent]")
    assert "kafka" in result["reports"][0].lower()          # the docs half, first
    assert "Kafka Manual" in result["reports"][0]           # ...and it found the doc
    assert result["reports"][1].startswith("[web_agent]")   # the web half, second
    assert "Investigation summary" in result["answer"]


def test_planner_finishes_when_satisfied_not_at_budget():
    # The evidence that the loop reflects, not just spends: give it MORE budget than
    # the work needs (4, for a 2-part question). After both reports the guard has not
    # fired (2 < 4), so the supervisor consults the planner again and IT chooses to
    # finish because every part now has a report — reason is the model's own, not the
    # fixed budget string.
    result = run(4)
    assert len(result["reports"]) == 2                       # exactly the two it needed
    assert result["reason"] != "delegation budget reached"   # planner finished, not the guard
    assert "Investigation summary" in result["answer"]


def test_budget_guard_forces_an_earlier_finish():
    # Same question and same planner, but a tighter budget. The planner WOULD keep
    # going (the Python half is still open), yet the guard finishes after one.
    result = run(1)
    assert len(result["reports"]) == 1                       # cut off early
    assert result["reports"][0].startswith("[docs_agent]")   # only the first ran
    assert not any(r.startswith("[web_agent]") for r in result["reports"])
    assert result["reason"] == "delegation budget reached"   # the guard fired, not the model


def test_guard_beats_the_prompt_zero_budget():
    # The extreme: budget 0 finishes before any worker runs at all — proof the guard
    # is enforced in code, not merely suggested to the model.
    result = run(0)
    assert result.get("reports", []) == []
    assert result["reason"] == "delegation budget reached"


if __name__ == "__main__":
    print("=== surplus budget (planner finishes when satisfied, not at the guard) ===")
    r = run(4)
    print(f"reports={len(r['reports'])}, finish reason={r['reason']!r} "
          f"(not the budget string)")
    print("--- answer ---")
    print(r["answer"])

    print("\n=== exact budget (needs both workers) ===")
    r = run(2)
    print("--- answer ---")
    print(r["answer"])

    print("\n=== tight budget (guard forces early finish) ===")
    r = run(1)
    print("--- answer ---")
    print(r["answer"])

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
