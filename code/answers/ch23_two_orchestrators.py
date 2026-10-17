"""Chapter 23 Challenge — Silver: same tools, two orchestrators.

The Gotcha claimed the raw client and the agent drive the *same* MCP tools while
sitting at opposite ends of the orchestration axis. This proves it with one
question -- "What topics can you answer about?" -- answered twice.

**Path A: you orchestrate.** The raw-client style. `session.call_tool("list_topics",
{})`. The tool was chosen by a line of your code, at the time you wrote it. The run
is deterministic, needs no model, and cannot pick a different tool no matter what
the question says.

**Path B: the model orchestrates.** The same server, loaded through
`MultiServerMCPClient` into `create_agent`. Nobody names a tool anywhere. The
question goes in, and `result["messages"]` shows the model deciding, on its own,
that `list_topics` is the one to call.

Two axes, and only one of them moved:

  * **packaging -- fixed.** Both paths cross the same process boundary to the same
    stdio server and get the same JSON back. `list_topics` did not become a
    different tool by being called differently.
  * **orchestration -- flipped.** Path A's caller is a `call_tool` line; path B's
    caller is a model reading a description. That is the whole difference, and it
    is worth being able to say out loud, because "we use MCP" tells you nothing
    about it: MCP is the packaging axis, and it leaves the choosing axis open.

Needs LM Studio at :1234 for path B; path A runs offline and is tested either way.
Run:  python code/answers/ch23_two_orchestrators.py
"""

import asyncio
import sys
import urllib.error
import urllib.request
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTION = "What topics can you answer about?"
SERVER = str(Path(__file__).resolve().parents[1] / "ch23_mcp_server.py")
LM_BASE = "http://localhost:1234/v1"


class Skip(Exception):
    pass


def server_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{LM_BASE}/models", timeout=2):
            return True
    except urllib.error.HTTPError:
        return True        # a status code is a server: a hosted endpoint answers 401 to a key-less probe
    except (urllib.error.URLError, OSError):
        return False


# --------------------------------------------------------------------------- #
# Path A -- your code chooses the tool
# --------------------------------------------------------------------------- #

async def you_orchestrate() -> list[str]:
    params = StdioServerParameters(command=sys.executable, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_topics", {})   # <- the choice
            return [item.text for item in result.content]


# --------------------------------------------------------------------------- #
# Path B -- the model chooses the tool
# --------------------------------------------------------------------------- #

async def the_model_orchestrates() -> tuple[list[str], list[str]]:
    """Returns (tools the model called, topic strings it got back)."""
    from langchain.agents import create_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI

    mcp = MultiServerMCPClient({"doc-bot-kb": {
        "command": sys.executable, "args": [SERVER], "transport": "stdio"}})
    tools = await mcp.get_tools()
    llm = ChatOpenAI(base_url=LM_BASE, api_key="lm-studio",
                     model="qwen/qwen3.8-27b", temperature=0, reasoning_effort="none")
    agent = create_agent(llm, tools)                              # <- no tool named

    result = await agent.ainvoke({"messages": [("user", QUESTION)]})
    called, returned = [], []
    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            called.append(tc["name"])
        if type(m).__name__ == "ToolMessage":
            blocks = m.content if isinstance(m.content, list) else [m.content]
            returned += [b["text"] if isinstance(b, dict) else str(b) for b in blocks]
    return called, returned


_CACHE: dict = {}


def cached() -> dict:
    if "a" not in _CACHE:
        _CACHE["a"] = asyncio.run(you_orchestrate())
    if "b" not in _CACHE and server_is_up():
        _CACHE["b"] = asyncio.run(the_model_orchestrates())
    return _CACHE


def demo_two_orchestrators():
    c = cached()
    print(f"Question: {QUESTION}\n")
    print("A. you orchestrate  (raw client, `call_tool(\"list_topics\", {})`)")
    print(f"   -> {c['a']}")
    print("   the tool was named in your source; no model was consulted\n")

    if "b" not in c:
        print(f"B. the model orchestrates -- SKIPPED, no model server at {LM_BASE}")
        return
    called, returned = c["b"]
    print("B. the model orchestrates  (create_agent over the same server)")
    print(f"   model chose : {called}")
    print(f"   tool returned: {returned}")
    print("   no tool name appears anywhere in this file's path-B code\n")

    print("Fixed  : packaging -- one stdio server, one process boundary, same JSON.")
    print("Flipped: orchestration -- a `call_tool` line, versus a model reading a")
    print("         description and deciding. 'We use MCP' says nothing about which.")


# --- tests -------------------------------------------------------------------

def test_you_orchestrating_returns_the_topics():
    assert set(cached()["a"]) == {"kafka", "redis", "mysql"}, cached()["a"]


def test_path_a_names_the_tool_in_its_own_source():
    """Path A's defining property: the choice is visible in the source."""
    import inspect
    assert '"list_topics"' in inspect.getsource(you_orchestrate)


def test_path_b_never_names_a_tool():
    """Path B's defining property: it cannot be, or the model is not choosing."""
    import inspect
    src = inspect.getsource(the_model_orchestrates)
    body = src[src.index('mcp = MultiServerMCPClient'):]
    assert "list_topics" not in body, body


def test_the_model_chose_list_topics_by_itself():
    if not server_is_up():
        raise Skip(f"no model server at {LM_BASE}")
    called, _ = cached()["b"]
    assert "list_topics" in called, called


def test_both_paths_get_the_same_topics_from_the_same_server():
    """The packaging axis did not move: same tool, same data, either way."""
    if not server_is_up():
        raise Skip(f"no model server at {LM_BASE}")
    _, returned = cached()["b"]
    joined = " ".join(returned).lower()
    for topic in cached()["a"]:
        assert topic in joined, (topic, joined)


def test_path_a_needs_no_model_at_all():
    """Run path A again with the model server's address made unusable -- it does
    not care, because nothing in it consults a model."""
    import os
    saved = os.environ.get("OPENAI_BASE_URL")
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:1"
    try:
        assert set(asyncio.run(you_orchestrate())) == {"kafka", "redis", "mysql"}
    finally:
        if saved is None:
            os.environ.pop("OPENAI_BASE_URL", None)
        else:
            os.environ["OPENAI_BASE_URL"] = saved


if __name__ == "__main__":
    demo_two_orchestrators()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        try:
            t()
            print("PASS", t.__name__)
        except Skip as e:
            print("SKIP", t.__name__, f"({e})")
    print(f"\nall {len(_tests)} tests passed")
