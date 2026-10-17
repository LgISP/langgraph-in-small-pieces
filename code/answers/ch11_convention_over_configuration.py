"""Chapter 11 Challenge — Bronze: convention over configuration.

`builder.add_conditional_edges("model", tools_condition)` never mentions the
`ToolNode`, yet the loop closes. This file exposes the mechanism in three steps:

1. THE SPELL. The chapter's wiring, verbatim: a node registered as "tools" and
   the prebuilt router. It works because `tools_condition` returns the STRING
   "tools", and a routing function's return value is a node name.
2. BREAK IT. Register the same ToolNode as "toolbox" instead. `compile()` —
   not even `invoke()` — fails: LangGraph reads the router's return annotation
   `Literal["tools", "__end__"]` to learn the possible targets, and "tools" no
   longer names a node.
3. BUY IT BACK. Keep the name "toolbox" and pay for it with a path map
   `{"tools": "toolbox", END: END}` — configuration overriding convention, at
   the price of one explicit mapping you now maintain.

`tools_condition` assumes a second name, too: it reads `state["messages"]`
(default `messages_key="messages"`) — the convention `MessagesState` satisfies.

The model is the chapter's real one (gpt-oss-20b via LM Studio, bound to the
tool): the wiring lesson is about names, but the tool call that exercises the
wiring is the model's own.

Run:  python code/answers/ch11_convention_over_configuration.py
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def search_docs(query: str) -> str:
    """Search the internal docs for a query."""
    return f"[docs] indexing chapter for {query!r}"


TOOLS = [search_docs]


llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
                 model="openai/gpt-oss-20b", temperature=0).bind_tools(TOOLS)

QUESTION = ("What does the internal documentation say about indexing? "
            "Use search_docs, then answer in one sentence.")


def model_node(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def build(tool_node_name: str, path_map: dict | None):
    """The chapter's loop, with the ToolNode's registered name as a knob."""
    builder = StateGraph(MessagesState)
    builder.add_node("model", model_node)
    builder.add_node(tool_node_name, ToolNode(TOOLS))
    builder.add_edge(START, "model")
    if path_map is None:
        builder.add_conditional_edges("model", tools_condition)
    else:
        builder.add_conditional_edges("model", tools_condition, path_map)
    builder.add_edge(tool_node_name, "model")
    return builder.compile()


def test_convention_works():
    # The name "tools" is the entire wiring: router string == node name.
    graph = build("tools", path_map=None)
    out = graph.invoke({"messages": [("user", QUESTION)]})
    assert any(m.type == "tool" and "[docs]" in str(m.content) for m in out["messages"])
    assert out["messages"][-1].content            # the model answered after the tool ran


def test_breaking_the_name_fails_at_compile():
    # Same graph, node renamed. The router still returns "tools" — and compile()
    # already knows, from Literal["tools", "__end__"], that nothing answers to it.
    try:
        build("toolbox", path_map=None)
    except ValueError as e:
        assert "unknown target 'tools'" in str(e)
    else:
        raise AssertionError("expected compile() to reject the dangling 'tools' target")


def test_path_map_buys_the_name_back():
    # Configuration overriding convention: one explicit mapping, and any name works.
    graph = build("toolbox", path_map={"tools": "toolbox", END: END})
    out = graph.invoke({"messages": [("user", QUESTION)]})
    assert any(m.type == "tool" and "[docs]" in str(m.content) for m in out["messages"])
    assert out["messages"][-1].content            # the model answered after the tool ran


if __name__ == "__main__":
    print("convention:", build("tools", None).invoke({"messages": [("user", QUESTION)]})["messages"][-1].content[:80])
    try:
        build("toolbox", None)
    except ValueError as e:
        print("renamed:   ", f"ValueError: {e}")
    print("path map:  ", build("toolbox", {"tools": "toolbox", END: END})
          .invoke({"messages": [("user", QUESTION)]})["messages"][-1].content[:80])

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
