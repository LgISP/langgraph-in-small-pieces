"""Chapter 23 - The same MCP tools, in a graph you wired yourself.

Between the raw client (you call the tool) and `create_agent` (one line does
everything) sits the shape this book has been building since @sec-tool-nodes: a
`model` node, a `ToolNode`, and a router between them. Running MCP tools through
it is the point of this file, because the alternative reading of Chapter 23 --
"crossing a process boundary means giving up your hand-built graph" -- is wrong,
and only code says so convincingly.

What `client.get_tools()` hands back are **ordinary LangChain tools**. Not
proxies you have to treat carefully, not a special MCP-flavoured object: the same
`BaseTool` interface `@tool` produces, with a name, a description, an args schema,
and an `ainvoke`. So everything downstream is unchanged. `bind_tools` binds them.
`ToolNode` dispatches them. The router reads `tool_calls` exactly as before. The
process boundary is real and it is *behind* the tool object, which is what a good
abstraction looks like.

One thing does change, and it is worth meeting here rather than in production:
**MCP tools are async.** The adapter builds them around an async client session,
so `ToolNode` has to be reached through `ainvoke`, and a `model` node that calls
`llm.invoke` sits in an async graph doing blocking work. This file uses `ainvoke`
throughout; @sec-async-vs-threads is where that choice gets taken apart.

Run:  python code/ch23_mcp_toolnode.py    (needs the local model from Ch10)
"""

import asyncio
import os
import sys
import warnings
from typing import Annotated, TypedDict

warnings.filterwarnings("ignore", message=".*incomplete definition.*")

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch23_mcp_server.py")

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen/qwen3.8-27b",
    temperature=0,
    reasoning_effort="none",
)


class State(TypedDict):
    messages: Annotated[list, add_messages]



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

def build(tools) -> StateGraph:
    """@sec-tool-nodes' three pieces, with MCP tools in the ToolNode.

    Read the wiring and notice what is NOT here: no mention of stdio, no session,
    no `call_tool`. The graph does not know its tools live in another process.
    """
    bound = llm.bind_tools(tools)

    async def model(state: State) -> dict:
        return {"messages": [await bound.ainvoke(state["messages"])]}

    def route(state: State) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    g = StateGraph(State)
    g.add_node("model", model)
    g.add_node("tools", ToolNode(tools))     # <- the only MCP-aware line, and it isn't
    g.add_edge(START, "model")
    g.add_conditional_edges("model", route, {"tools": "tools", END: END})
    g.add_edge("tools", "model")
    return g


async def main() -> None:
    client = MultiServerMCPClient({"doc-bot-kb": {
        "command": sys.executable, "args": [SERVER], "transport": "stdio"}})
    tools = await client.get_tools()

    print(f"Loaded {len(tools)} tool(s) over stdio: {[t.name for t in tools]}")
    print(f"Type of the first one: {type(tools[0]).__mro__[1].__name__}"
          f"  <- a plain LangChain tool, not an MCP object\n")

    graph = build(tools).compile()
    question = "How do I configure a Spring Boot Kafka batch consumer?"
    print(f"Question: {question}")

    out = await graph.ainvoke({"messages": [("user", question)]})
    for m in out["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            print(f"  [model] chose {tc['name']}({tc['args']})")
        if type(m).__name__ == "ToolMessage":
            blocks = m.content if isinstance(m.content, list) else [m.content]
            text = ", ".join(b["text"] if isinstance(b, dict) else str(b) for b in blocks)
            print(f"  [tools] returned {text}")
    print(f"  [answer] {excerpt(out['messages'][-1].content)}")


if __name__ == "__main__":
    asyncio.run(main())
