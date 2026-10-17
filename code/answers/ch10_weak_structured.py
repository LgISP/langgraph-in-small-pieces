"""Chapter 10 Challenge — Bronze: does the schema save the weak model?

Point the structured doc-bot (code/ch10_real_llm.py) at a weak model and watch
what the `Literal` route can and cannot buy you. Structured output enforces the
SHAPE — every `route` that comes back is one of search / web / generate, never a
parse error, never garbage. It does nothing for the CHOICE: a weak model that
ignores the state still picks `search` on a full result and loops until Chapter 4's
recursion fuse trips with `GraphRecursionError`. Shape is not judgment.

Both models are REAL, served by LM Studio: the capable one is the book's default
`qwen/qwen3.8-27b`, the weak one is `qwen2.5-vl-7b-instruct` — an actual
small model, not a stand-in scripted to fail. Same schema, same prompt, same
graph, same guaranteed-legal enum; only the judgment differs. On our machine the
7B fails exactly as the chapter warned: shown two retrieved documents and rule 1
("you already have a source -> generate"), it answers `search` again — its stated
reason even *quotes* a rule whose condition is false — and loops until the fuse
blows. That last part is a MODEL OBSERVATION, not a law: a stronger 7B might
terminate, and the chapter tells you to record whatever yours does. So the tests
assert only the framework INVARIANTS — every route is a legal enum, and the run
is bounded (an answer, or Chapter 4's recursion fuse) — never that a given model
must loop.

Run:  python code/answers/ch10_weak_structured.py   (see answers/README.md)
"""

import warnings
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


# Two REAL models behind the same local server. The only difference is capability.
capable_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",      # the book's default: follows the rules
    temperature=0,
    reasoning_effort="none",
)
weak_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen2.5-vl-7b-instruct",        # a real 7B: the schema can't fix its judgment
    temperature=0,
)


class Decision(BaseModel):
    """Choose the next action for a documentation question-answering agent."""

    route: Literal["search", "web", "generate"] = Field(
        description="Next action: search=look in the local docs, "
        "web=search the open web, generate=write the answer now."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


def excerpt(text: str, width: int = 140) -> str:
    """First line of a model answer, cut at a word boundary, with '...' if cut."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) <= width and first == text.strip():
        return first
    cut = first[:width].rsplit(" ", 1)[0] if len(first) > width else first
    return cut.rstrip(" ,;:") + " ..."


# --- the structured doc-bot, unchanged in shape from code/ch10_real_llm.py ----
class State(TypedDict):
    question: str                       # input, set by the caller
    route: NotRequired[str]             # written by agent: search|web|generate
    docs: NotRequired[list[str]]        # written by search_docs (may be empty)
    web_results: NotRequired[list[str]] # written by web_search
    iterations: NotRequired[int]        # loop counter the agent keeps (Ch4)
    reason: NotRequired[str]            # the model's stated reason (observability)
    answer: NotRequired[str]            # written by generate


KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ],
}

ROUTES: list[str] = []   # every route the agent produced this run (for the tests)


def build(model):
    """Build Chapter 10's structured doc-bot around a given REAL model. The graph
    topology never changes — only which model the `agent` and `generate` bodies use."""
    decider = model.with_structured_output(Decision, method="json_schema")

    def agent_node(state: State) -> dict:
        docs = state.get("docs", [])
        web = state.get("web_results", [])
        searched_docs = "docs" in state
        searched_web = "web_results" in state

        # The chapter's hardened ordered rules, verbatim from code/ch10_real_llm.py.
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

        decision = decider.invoke(prompt)     # a Decision; route is a legal enum
        ROUTES.append(decision.route)
        print(f"[agent] docs={len(docs)} web={len(web)} -> {decision.route}"
              f"  ({excerpt(decision.reason)})")
        return {
            "route": decision.route,
            "reason": decision.reason,
            "iterations": state.get("iterations", 0) + 1,
        }

    def search_docs_node(state: State) -> dict:
        q = state["question"].lower()
        hits: list[str] = []
        for topic, docs in KNOWLEDGE_BASE.items():
            if topic in q:
                hits.extend(docs)
        print(f"[search_docs] {len(hits)} hit(s)")
        return {"docs": hits}

    def web_search_node(state: State) -> dict:
        print("[web_search] falling back to a web search")
        return {"web_results": [f"[web] thread about {state['question']!r}"]}

    def generate_node(state: State) -> dict:
        sources = state.get("docs", []) + state.get("web_results", [])
        sources_text = "\n".join(f"- {s}" for s in sources) or "(no sources)"
        prompt = (f"Answer the user's question concisely, using only these sources:\n"
                  f"{sources_text}\n\nQuestion: {state['question']}")
        answer = model.invoke(prompt).content.strip()   # a REAL model call
        print(f"[generate] {excerpt(answer)}")
        return {"answer": answer}

    def route_after_agent(state: State) -> str:
        return state["route"]

    b = StateGraph(State)
    b.add_node("agent", agent_node)
    b.add_node("search_docs", search_docs_node)
    b.add_node("web_search", web_search_node)
    b.add_node("generate", generate_node)
    b.add_edge(START, "agent")
    b.add_conditional_edges(
        "agent",
        route_after_agent,
        {"search": "search_docs", "web": "web_search", "generate": "generate"},
    )
    b.add_edge("search_docs", "agent")
    b.add_edge("web_search", "agent")
    b.add_edge("generate", END)
    return b.compile()


capable_graph = build(capable_llm)
weak_graph = build(weak_llm)

QUESTION = "How do I configure Kafka batch consumers?"
LEGAL_ROUTES = {"search", "web", "generate"}

# Each graph runs ONCE; the (outcome, routes) pair is shared by every test.
# (Real model calls are slow, and the two models evict each other from the
# server's memory — one run each is the honest budget.)
_RUNS: dict = {}


def run_capable():
    if "capable" not in _RUNS:
        ROUTES.clear()
        result = capable_graph.invoke({"question": QUESTION}, {"recursion_limit": 8})
        _RUNS["capable"] = (result, list(ROUTES))
    return _RUNS["capable"]


def run_weak():
    if "weak" not in _RUNS:
        ROUTES.clear()
        raised = False
        result = None
        try:
            result = weak_graph.invoke({"question": QUESTION}, {"recursion_limit": 8})
        except GraphRecursionError:
            raised = True                     # the fuse blew — one possible outcome
        _RUNS["weak"] = (raised, list(ROUTES), result)
    return _RUNS["weak"]


# --- tests -------------------------------------------------------------------
def test_capable_model_terminates():
    result, routes = run_capable()
    assert result["answer"]                          # it produced an answer
    assert all(r in LEGAL_ROUTES for r in routes)    # every route legal
    assert routes[-1] == "generate"                  # a terminating run ends by generating


def test_shape_is_always_legal():
    # The invariant structured output *does* buy: every route that comes back is a
    # legal enum member, whatever the model's judgment. Never a parse error, never
    # garbage — that holds for any model behind this schema.
    _, routes, _ = run_weak()
    assert routes                                    # it did decide, at least once
    assert all(r in LEGAL_ROUTES for r in routes)    # shape: every route legal


def test_weak_run_is_bounded():
    # The other invariant: the run cannot wander off. It either finishes with an
    # answer or trips Chapter 4's recursion fuse — never unbounded, never an illegal
    # shape. WHICH of the two happens is a model observation, not a guarantee: this
    # 7B loops to the fuse on our machine; a stronger one might answer.
    raised, _, result = run_weak()
    assert raised or (result and result.get("answer"))


def _warm(model_id: str) -> None:
    """LM Studio sometimes fails to JIT-load the 7B; one throwaway prompt fixes it."""
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    for _ in range(3):
        try:
            client.chat.completions.create(model=model_id,
                messages=[{"role": "user", "content": "hi"}], max_tokens=1)
            return
        except Exception:
            import time; time.sleep(2)


if __name__ == "__main__":
    _warm("qwen2.5-vl-7b-instruct")
    print("===== capable real model (follows the rules) =====")
    result, routes = run_capable()
    print("routes:", routes)
    print("answer:", excerpt(result["answer"]))

    print("\n===== weak real model (ignores the state) =====")
    raised, routes, weak_result = run_weak()
    if raised:
        print("!! GraphRecursionError: the agent looped until the fuse blew")
    else:
        print(f"(terminated with an answer: {excerpt(weak_result['answer'])})")
    print(f"routes the weak model produced: {routes}")
    print(f"every route legal (shape held): {all(r in LEGAL_ROUTES for r in routes)}")
    # An OBSERVATION on our machine (yours may differ): this 7B chose the same
    # route every pass even holding two docs — shape without judgment.
    print(f"distinct routes chosen: {sorted(set(routes))}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
