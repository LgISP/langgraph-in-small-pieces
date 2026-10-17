"""Chapter 11 Challenge — Silver: reach vs. control, made concrete.

The chapter's lesson was that native tool-calling buys *reach* at the cost of
*control*: the model may skip a tool you'd have mandated, or answer with no tool
at all. This file makes that trade runnable and testable, in three steps:

1. FAILURE. A `create_agent` bot, asked "What is the capital of France?",
   calls NO tool and answers from its own knowledge — the loss-of-control moment
   the chapter's Gotcha flagged.
2. THE NUDGE. Pass a `system_prompt=` message ordering it to call `search_docs`
   before answering. Compliance goes UP — but nothing *enforces* it: it is an
   empirical outcome that can shift with the model, the wording, the question.
3. THE STRUCTURAL FIX. The @sec-tool-nodes graph, where an edge — not the model —
   makes `search_docs` run first. Now it runs on EVERY question, 100%, because the
   guarantee lives in the wiring, not the model's mood.

The model is the chapter's real one — `openai/gpt-oss-20b` behind LM Studio,
driven by the real `create_agent` loop. Two honest notes from running it for
real rather than scripting it. First, on THESE three questions the strict prompt
happened to reach full compliance (gpt-oss searched the docs every time — and
then chained into `web_search` too when the docs came up empty); the difference
from the graph is not this run's score but the *kind* of promise: a prompt gave
us 3/3 today, the edge gives 3/3 by construction. Second, the no-prompt bot's
choices are the model's own — it searched docs for Kafka, went straight to the
web for the Python question, and called nothing for France — so the tests pin
the structure (which tools were requested, that a ToolMessage came back, that
the final message answers) and never the model's prose.

Run:  python code/answers/ch11_forced_tool.py   (see answers/README.md)
"""

import warnings
from typing import NotRequired, TypedDict

# create_agent lives in the langchain package (pip install langchain==1.3.15); it
# is where langgraph.prebuilt's create_react_agent moved in 1.0. Same loop.
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


# The chapter's real local model — gpt-oss, as in code/ch11_react_agent.py.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="openai/gpt-oss-20b",
    temperature=0,                         # 0 = as reproducible as a real model gets
)


def excerpt(text: str, width: int = 140) -> str:
    """First line of a model answer, cut at a word boundary, with '...' if cut."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) <= width and first == text.strip():
        return first
    cut = first[:width].rsplit(" ", 1)[0] if len(first) > width else first
    return cut.rstrip(" ,;:") + " ..."


# --- the doc-bot tools, verbatim from the chapter ----------------------------
KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch "
        "to enable batch consumption.",
    ],
    "redis": [
        "Redis Manual: set the connection pool via spring.data.redis.*.",
    ],
}


@tool
def search_docs(query: str) -> str:
    """Search the internal technical-documentation knowledge base. Use this
    whenever the user asks how to configure something, about an error message, or
    how to do a technical task."""
    q = query.lower()
    hits: list[str] = []
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in q:
            hits.extend(docs)
    return "\n".join(hits) if hits else "No matching local documents."


@tool
def web_search(query: str) -> str:
    """Search the open web. Use this when the local docs come up empty, or the
    question is about something recent."""
    # PLACEHOLDER: not a real web search — a canned string so the loop has something
    # to run, and it stays a stub for the whole book.
    return f"[web] community thread about {query!r} (example result)"


TOOLS = [search_docs, web_search]

STRICT_PROMPT = (
    "You must call search_docs before answering any question. "
    "Never answer from your own knowledge."
)


# --- the two tool-calling bots: bare, and nudged with a system prompt ---------
# (create_agent has no "force this tool" switch: system_prompt= is the nudge the
# challenge asks for, and the structural fix below is the guarantee.)
no_prompt_agent = create_agent(llm, TOOLS)
prompt_agent = create_agent(llm, TOOLS, system_prompt=STRICT_PROMPT)


# --- the structural fix: the @sec-tool-nodes graph ----------------------------
# Here NO model decides whether to search. The edge START -> search does, so
# search_docs runs before generate can, on every question, unconditionally.
class DocBotState(TypedDict):
    question: str
    docs: NotRequired[list[str]]
    answer: NotRequired[str]


def force_search_node(state: DocBotState) -> dict:
    result = search_docs.invoke({"query": state["question"]})   # code runs it, always
    return {"docs": [result]}


def generate_node(state: DocBotState) -> dict:
    docs = state.get("docs", [])
    docs_text = "\n".join(f"- {d}" for d in docs) or "(no sources)"
    prompt = (f"Answer the user's question concisely. Retrieved documents "
              f"(may be empty):\n{docs_text}\n\nQuestion: {state['question']}")
    return {"answer": llm.invoke(prompt).content.strip()}


_b = StateGraph(DocBotState)
_b.add_node("search", force_search_node)
_b.add_node("generate", generate_node)
_b.add_edge(START, "search")
_b.add_edge("search", "generate")   # generate is UNREACHABLE until search has run
_b.add_edge("generate", END)
forced_graph = _b.compile()


# --- questions and small trace helpers ---------------------------------------
Q_FRANCE = "What is the capital of France?"    # general knowledge, no docs
Q_KAFKA = "How do I configure Kafka batch consumers?"   # docs cover it
Q_PYTHON = "How do I write a Python decorator?"          # docs do NOT cover it
QUESTIONS = [Q_FRANCE, Q_KAFKA, Q_PYTHON]


def tool_calls_in(result: dict) -> list[str]:
    """Every tool name the model asked for across a create_agent run."""
    names: list[str] = []
    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            names.append(tc["name"])
    return names


# Each (agent, question) pair runs ONCE and is shared by the tests — real model
# calls are slow, and every assertion describes the same run.
_RUNS: dict = {}


def react_run(agent, label: str, question: str) -> dict:
    key = (label, question)
    if key not in _RUNS:
        # recursion_limit is the loop fuse, same as ch11_react_agent.py.
        _RUNS[key] = agent.invoke(
            {"messages": [("user", question)]}, {"recursion_limit": 10}
        )
    return _RUNS[key]


def search_ran_react(agent, label: str, question: str) -> bool:
    return "search_docs" in tool_calls_in(react_run(agent, label, question))


def search_ran_forced(question: str) -> bool:
    # The forced graph never emits tool_calls; search_docs runs inside the node,
    # and its output is the ONLY source of `docs`, so docs-present <=> search ran.
    key = ("forced", question)
    if key not in _RUNS:
        _RUNS[key] = forced_graph.invoke({"question": question})
    return bool(_RUNS[key].get("docs"))


def compliance(agent, label: str) -> int:
    return sum(search_ran_react(agent, label, q) for q in QUESTIONS)


# --- tests -------------------------------------------------------------------
# Real-model note: the assertions pin which tools were REQUESTED and the shape
# of the loop's transcript, never the model's wording. The one content check
# (Paris) is a fact, not a phrasing.
def test_no_prompt_bot_answers_france_from_thin_air():
    # The loss-of-control failure in one run: no tool at all, a parametric answer.
    result = react_run(no_prompt_agent, "bare", Q_FRANCE)
    assert tool_calls_in(result) == []                     # called NOTHING
    assert "paris" in result["messages"][-1].content.lower()  # answered from knowledge


def test_no_prompt_search_compliance_is_partial():
    # Left to its own judgment the model searches the docs only where IT sees the
    # point: the Kafka question. France gets no tool at all; the model may well
    # skip straight past the docs elsewhere (for Python it went to web_search).
    assert search_ran_react(no_prompt_agent, "bare", Q_KAFKA) is True
    assert search_ran_react(no_prompt_agent, "bare", Q_FRANCE) is False
    assert compliance(no_prompt_agent, "bare") < len(QUESTIONS)


def test_system_prompt_raises_compliance_but_cannot_enforce_it():
    # The prompt RAISES compliance — on this run it even reached 3/3 (see the
    # docstring). What it cannot do is ENFORCE: nothing in the mechanism refuses
    # to answer when the model skips the tool, so the ceiling is the model's
    # obedience, not the system's wiring.
    assert compliance(prompt_agent, "strict") > compliance(no_prompt_agent, "bare")
    # France flips to compliant under the prompt — the failure case is cured by
    # words alone, which is exactly why it can regress by words alone.
    assert search_ran_react(prompt_agent, "strict", Q_FRANCE) is True


def test_graph_node_bot_forces_search_first_structurally():
    # The edge, not the model, decides. search_docs runs on every question.
    assert all(search_ran_forced(q) for q in QUESTIONS)
    # And generate only ever sees docs that search produced.
    result = _RUNS[("forced", Q_FRANCE)]
    assert result["docs"] and "answer" in result


def test_control_ladder_is_monotonic():
    # The whole point in one line: reach (no prompt) < nudge (prompt) <= guarantee.
    no_prompt = compliance(no_prompt_agent, "bare")
    with_prompt = compliance(prompt_agent, "strict")
    structural = sum(search_ran_forced(q) for q in QUESTIONS)
    assert no_prompt < with_prompt <= structural == len(QUESTIONS)


def test_real_loop_runs_tool_as_data_then_code_executes():
    # The mechanic on the Kafka run: the model emits a search_docs tool_call
    # (DATA), the agent's ToolNode runs it and appends a ToolMessage (CODE ran
    # it), and the model answers only after reading that result.
    result = react_run(no_prompt_agent, "bare", Q_KAFKA)
    msgs = result["messages"]
    assert isinstance(msgs[0], HumanMessage)
    assert tool_calls_in(result) == ["search_docs"]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "spring.kafka.listener.type=batch" in tool_msgs[0].content
    assert isinstance(msgs[-1], AIMessage) and msgs[-1].content.strip()


if __name__ == "__main__":
    print("########## tool-calling bot, NO prompt: reach, little control ##########")
    for q in QUESTIONS:
        called = tool_calls_in(react_run(no_prompt_agent, "bare", q)) or ["(no tool)"]
        print(f"  {q!r:52} -> {', '.join(called)}")

    print("\n########## tool-calling bot, strict prompt: a stronger nudge ##########")
    for q in QUESTIONS:
        called = tool_calls_in(react_run(prompt_agent, "strict", q)) or ["(no tool)"]
        print(f"  {q!r:52} -> {', '.join(called)}")

    print("\n########## graph-node bot: search_docs forced by an edge ##########")
    for q in QUESTIONS:
        ran = "search_docs (forced)" if search_ran_forced(q) else "(skipped!)"
        print(f"  {q!r:52} -> {ran}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
