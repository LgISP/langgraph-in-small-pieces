"""Chapter 14 — Multi-agent supervisor: one agent whose "tools" are other agents.

A `supervisor` node reads the question, delegates to ONE worker at a time, reads
what that worker reports back, and decides whether to delegate again or finish.
This is the Chapter 5 agent-tool loop (a hub node with back-edges), except the
"tools" are worker agents and the hub is a supervisor.

Two things to keep straight:
    - This is SERIAL delegation (Chapter 14): one worker per step, the supervisor
      re-decides after each report. It is NOT Chapter 8's `Send` parallel fan-out,
      which spawns N identical workers in a single superstep.
    - The route is chosen by structured output (Chapter 10): a `Delegate` whose
      `route` is a `Literal`, so the pick is always a legal worker name.

Both workers are real agents, not stubs: each has its own prompt, its own model
call, and its own private pile of raw material. Only a one-sentence conclusion
crosses back into `reports`, which is the isolation a sub-agent buys you — the
`read ... / returned ...` line each worker prints is the two ends of it.

`web_agent` searches DuckDuckGo, so this is the one file in the book that needs a
network connection (no API key, no bill). Its results — and so this script's
output — change over time.

Built from primitives — add_conditional_edges + back-edges — no supervisor library.

Run:  pip install ddgs && python code/ch14_supervisor.py
"""

import operator
import warnings
from typing import Annotated, Literal, NotRequired, TypedDict

from ddgs import DDGS
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# with_structured_output makes pydantic emit one harmless serializer warning about
# an internal field; the parsed result is correct. Silence just that one line.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


# A real model behind a local OpenAI-compatible endpoint. Nothing here
# talks to a paid cloud — the whole book runs on your own machine.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",  # your local server, not the OpenAI cloud
    api_key="lm-studio",                   # the local server ignores the key
    model="openai/gpt-oss-20b",            # gpt-oss on purpose here — see "Look at what it's thinking"
    temperature=0,                         # 0 = as reproducible as a real model gets
)


# The supervisor's decision as a typed shape. `route` is a Literal, so the pick is
# always one of the three legal values — never a stray sentence (see Chapter 10).
class Delegate(BaseModel):
    """Pick the next worker for a documentation supervisor, or finish."""

    # reason comes FIRST: fields are emitted in schema order, so this is the only
    # place gpt-oss gets to think under LM Studio's JSON grammar. Move it after
    # `route` and the supervisor mis-routes (see "Look at what it's thinking").
    reason: str = Field(
        description="Think step by step about which rule applies, then decide."
    )
    route: Literal["docs_agent", "web_agent", "finish"] = Field(
        description="Which worker to delegate to next: "
        "docs_agent=search the internal documentation, "
        "web_agent=search the open web, "
        "finish=stop delegating and write the final answer."
    )


# Bind the shape to the model: `router` returns a Delegate object, never raw text.
router = llm.with_structured_output(Delegate, method="json_schema")


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


class State(TypedDict):
    question: str                                # input, set by the caller
    route: NotRequired[str]                      # supervisor's pick this step
    reason: NotRequired[str]                     # the supervisor's stated reason
    reports: Annotated[list[str], operator.add]  # workers append; supervisor reads
    answer: NotRequired[str]                     # written by finish


# The same internal knowledge base as Chapters 5 and 10 — the doc-bot universe is
# unchanged. There is deliberately no "python" topic, so one question misses the
# docs and the supervisor has to fall back to the web agent.
KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ],
    "redis": [
        "Redis Manual: set the connection pool via spring.data.redis.*.",
    ],
}


# The supervisor: read the question and every report so far, pick the next worker
# (or finish). This is Chapter 5's `agent` — except its "tools" are worker agents.
def supervisor_node(state: State) -> dict:
    reports = state.get("reports", [])
    # Which workers have already reported? Read it off the tagged reports.
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
    print(f"[supervisor] reports={len(reports)} -> {decision.route}"
          f"  ({decision.reason})")
    return {"route": decision.route, "reason": decision.reason}


# What makes a worker an AGENT and not a tool: it reads its raw material and
# decides what is worth reporting. Both workers are the same three steps — fetch
# material, have the model read it, return one sentence — so the model call lives
# in one helper and only the fetching differs.
WORKER_PROMPT = """You are the {name}. Answer ONLY from the material below.
Write one sentence stating what it says about the question. If the material does
not answer it, reply exactly: nothing useful found.

Question: {q}

Material:
{material}"""


def run_worker(name: str, tag: str, q: str, material: str) -> dict:
    """One worker turn: read `material`, hand back a single tagged conclusion.

    The two printed numbers are the point of the sub-agent boundary: the worker
    reads all of `material`, and only `conclusion` ever reaches the supervisor,
    because the supervisor's prompt is built from `reports`.
    """
    conclusion = llm.invoke(
        WORKER_PROMPT.format(name=name, q=q, material=material)
    ).content.strip()
    print(f"   [{tag}] read {len(material)} chars of raw material, "
          f"returned {len(conclusion)}")
    return {"reports": [f"[{tag}] {conclusion}"]}


# Worker: search the internal docs, report one conclusion, return to the
# supervisor. A worker reports ONLY to the supervisor — never to the other worker.
def docs_agent_node(state: State) -> dict:
    q = state["question"]
    hits = [d for t, docs in KNOWLEDGE_BASE.items() if t in q.lower() for d in docs]
    print(f"   [docs_agent] searched internal docs: {len(hits)} hit(s)")
    material = "\n".join(f"- {h}" for h in hits) or "(no internal documents matched)"
    return run_worker("internal-documentation agent", "docs_agent", q, material)


# Worker: the web fallback — a real DuckDuckGo search (no API key). This is the
# one line in the book that leaves your machine, and the reason this script's
# output drifts: the web changes, so the snippets and the conclusion change too.
def web_agent_node(state: State) -> dict:
    q = state["question"]
    results = list(DDGS().text(q, max_results=4))
    print(f"   [web_agent] searched the open web: {len(results)} result(s)")
    material = "\n".join(f"- {r['title']}: {r['body']}" for r in results)
    return run_worker("web-search agent", "web_agent", q, material)


# The supervisor finishes here: the model writes the final answer from the reports.
def finish_node(state: State) -> dict:
    reports = state.get("reports", [])
    reports_text = "\n".join(f"- {r}" for r in reports) or "(no reports)"
    prompt = f"""
        Answer the user's question concisely, using only the worker
        reports below. Do not invent anything.

        Question: {state['question']}

        Worker reports:
        {reports_text}

        Answer:"""
    answer = llm.invoke(prompt).content.strip()
    print("[finish] the supervisor wrote the final answer")
    return {"answer": answer}


# The supervisor's router: read the route it wrote and return it — the same
# one-line helper as Chapters 5 and 10.
def route_after_supervisor(state: State) -> str:
    return state["route"]


# The graph is Chapter 5's hub-and-back-edges loop, with a supervisor at the hub:
# the supervisor routes to a worker, each worker returns to the supervisor, and
# `finish` is the one exit.
builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("docs_agent", docs_agent_node)
builder.add_node("web_agent", web_agent_node)
builder.add_node("finish", finish_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "docs_agent": "docs_agent",
        "web_agent": "web_agent",
        "finish": "finish",
    },
)
builder.add_edge("docs_agent", "supervisor")
builder.add_edge("web_agent", "supervisor")
builder.add_edge("finish", END)

graph = builder.compile()


if __name__ == "__main__":
    # Two questions take two paths, just as in Chapters 5 and 10:
    #   Kafka  is in the knowledge base -> docs_agent answers -> finish
    #   Python is not                   -> docs_agent misses  -> web_agent -> finish
    for q in [
        "How do I configure Kafka batch consumers?",
        "How do I write a Python decorator?",
    ]:
        print(f"\n===== question: {q} =====")
        result = graph.invoke({"question": q}, {"recursion_limit": 12})
        print("--- answer ---")
        print(excerpt(result["answer"]))
