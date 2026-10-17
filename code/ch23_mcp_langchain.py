"""Chapter 23 - Load MCP tools into a LangGraph agent with langchain-mcp-adapters.

This wires three pieces into the production shape of "a knowledge base exposed
to an agent through MCP":

    ch23_mcp_server.py (exposes search_docs / list_topics)
        |  loaded by langchain-mcp-adapters
        v
    ordinary LangChain tools
        |  handed to create_agent (tool calling)
        v
    a LangGraph agent whose LLM decides on its own to call these MCP tools

The contrast worth noticing:
    the @tool functions in Part II are defined IN THIS process; the tools here
    come from ANOTHER process's MCP server -- yet to the agent they are
    identical. It tool-calls them exactly the same way and cannot tell which is
    which.

Run:  python code/ch23_mcp_langchain.py
"""

import asyncio
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

# The server file lives next to this one; reference it by absolute path so the
# working directory does not matter.
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch23_mcp_server.py")

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen/qwen3.8-27b",  # a capable open model that tool-calls well
    temperature=0,
    reasoning_effort="none",
)


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


async def main():
    # (1) where the tools live; sys.executable for the same reason as in the raw client
    client = MultiServerMCPClient(
        {
            "doc-bot-kb": {
                "command": sys.executable,
                "args": [SERVER],
                "transport": "stdio",
            }
        }
    )

    # (2) load the server's tools as LangChain tools
    tools = await client.get_tools()
    print("Tools loaded from the MCP server:", [t.name for t in tools])
    print("(Note: these come from ANOTHER process's server, not this file)\n")

    # (3) the same one-liner as for in-process @tool functions
    agent = create_agent(llm, tools)

    for q in ["How do I configure a Spring Boot Kafka batch consumer?",
              "What topics does the knowledge base cover?"]:
        print(f"{'=' * 56}\nQuestion: {q}\n{'=' * 56}")
        result = await agent.ainvoke({"messages": [("user", q)]})
        for m in result["messages"]:
            role = type(m).__name__.replace("Message", "")
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    print(f"  [LLM] decided to call MCP tool -> {tc['name']}({tc['args']})")
            elif role == "Tool":
                # An MCP tool's ToolMessage.content is a LIST of content blocks,
                # one per item, not a string; join their text before printing.
                blocks = m.content if isinstance(m.content, list) else [m.content]
                text = ", ".join(b["text"] if isinstance(b, dict) else str(b) for b in blocks)
                print(f"  [MCP tool returned] {text}")
            elif role == "AI" and m.content:
                print(f"  [final answer] {excerpt(m.text)}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
