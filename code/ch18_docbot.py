"""Chapter 18 - The doc-bot, packaged as a system under test.

Every graph so far was built and run in one file. A graph you intend to TEST has
to be importable without running: `build_docbot()` returns the uncompiled
builder, so each test compiles its own copy with its own checkpointer, and
nothing is decided at import time.

The one design choice that makes the rest of the chapter possible is `Context`:
the model arrives as `llm: BaseChatModel` rather than being constructed inside
`model()`. That is the seam @sec-runtime-context describes -- a test hands in a
scripted model and not one node changes.

Run:  python code/ch18_docbot.py        (a real run, needs LM Studio)
Test: pytest code/ch18_test_docbot.py   (no model server needed)
"""

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

KNOWLEDGE_BASE = {
    "kafka": "Kafka Manual: use spring.kafka.listener.type=batch to enable",
    "redis": "Redis Manual: configure spring.data.redis.lettuce.pool.max-active",
}


@tool
def search_docs(query: str) -> str:
    """Search the internal technical-documentation knowledge base. Use this
    whenever the user asks how to configure something."""
    q = query.lower()
    hits = [text for topic, text in KNOWLEDGE_BASE.items() if topic in q]
    return "\n".join(hits) if hits else "No matching local documents."


@tool
def web_search(query: str) -> str:
    """Search the open web. Use this when the local docs come up empty."""
    return f"[web] community thread about “{query}” (example result)"


TOOLS = [search_docs, web_search]
BY_NAME = {t.name: t for t in TOOLS}


@dataclass
class Context:
    """Per-run values. The model is one of them -- that is the testing seam."""

    llm: BaseChatModel


class State(TypedDict):
    messages: Annotated[list, add_messages]
    tools_used: Annotated[list[str], operator.add]


def model(state: State, runtime: Runtime[Context]) -> dict:
    """Ask the model what to do next. Returns ONLY a messages delta."""
    reply = runtime.context.llm.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [reply]}


def tools(state: State) -> dict:
    """Run whatever the model asked for. Needs no context, so a test can reach
    it through `graph.nodes["tools"]`."""
    calls = state["messages"][-1].tool_calls
    results, used = [], []
    for call in calls:
        output = BY_NAME[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=output, tool_call_id=call["id"]))
        used.append(call["name"])
    return {"messages": results, "tools_used": used}


def route(state: State) -> str:
    """Tool calls pending -> run them; otherwise the answer is final."""
    return "tools" if state["messages"][-1].tool_calls else END


def build_docbot() -> StateGraph:
    """The UNCOMPILED builder. Tests compile their own copy."""
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("model", model)
    builder.add_node("tools", tools)
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", route, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "model")
    return builder


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="openai/gpt-oss-20b",
        temperature=0,
    )
    graph = build_docbot().compile()
    out = graph.invoke(
        {"messages": [("user", "How do I configure Kafka batch consumers?")],
         "tools_used": []},
        context=Context(llm=llm),
    )
    print("tools_used:", out["tools_used"])
    print("answer    :", out["messages"][-1].content[:80], "...")
