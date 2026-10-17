"""Chapter 23 Challenge — Bronze: a tool the client never heard of.

The claim under test is MCP's whole selling point: the toolset is a **run-time
fact**. So the client's source is not touched. Not one line -- this file imports
`ch23_mcp_client` and runs its `main()`. The only thing that changes is which
server it launches, and that is a path, not code.

Predict before running:

  * **How many tools does the discovery block print?** Three. `list_tools` asks
    the server, and the server now registers three.
  * **What does `count_docs`'s line look like?** `- count_docs[]  -- Total number
    of documentation snippets across the whole knowledge base.` The empty bracket
    is the point: the client prints `list(inputSchema["properties"])`, and a
    zero-argument function produces an empty properties object. The description is
    the docstring's first line, lifted by FastMCP with no help from us -- which is
    why a tool's docstring is not a comment, it is the API the other side reads.

Then the tool is called with `call_tool("count_docs", {})` and the number is
checked against the knowledge base the server was built from: 2 kafka + 1 redis +
1 mysql = 4.

Run:  python code/answers/ch23_a_tool_the_client_never_heard_of.py
"""

import asyncio
import contextlib
import io
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ch23_mcp_client as client                  # noqa: E402
from ch23_mcp_server import KNOWLEDGE_BASE        # noqa: E402

SERVER = str(Path(__file__).with_name("ch23_server_plus_count.py"))
EXPECTED_TOTAL = sum(len(v) for v in KNOWLEDGE_BASE.values())


async def discover_and_call() -> tuple[list, int]:
    """Talk to the extended server: what is there, and what does the new tool say."""
    params = StdioServerParameters(command=sys.executable, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            result = await session.call_tool("count_docs", {})
            return tools, int(result.content[0].text)


def run_the_untouched_client() -> str:
    """Run `ch23_mcp_client.main()` verbatim, pointed at the extended server.

    Patching the module's SERVER constant is the file-level equivalent of typing
    a different path -- the client's *logic* is what must stay untouched, and it
    does: it still learns the toolset by asking.
    """
    original, client.SERVER = client.SERVER, SERVER
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            asyncio.run(client.main())
    finally:
        client.SERVER = original
    return buf.getvalue()


_CACHE: dict = {}


def cached():
    if not _CACHE:
        _CACHE["tools"], _CACHE["count"] = asyncio.run(discover_and_call())
        _CACHE["client_output"] = run_the_untouched_client()
    return _CACHE


def demo_a_tool_the_client_never_heard_of():
    c = cached()
    print("The chapter's client, unedited, pointed at a server with one more tool:")
    print("-" * 70)
    print(c["client_output"].rstrip())
    print("-" * 70)
    print(f"\ncount_docs() -> {c['count']}   "
          f"(knowledge base holds {EXPECTED_TOTAL} snippets)")
    print("\nThe client was never told count_docs exists. It asked.")


# --- tests -------------------------------------------------------------------

def test_the_server_now_exposes_three_tools():
    assert len(cached()["tools"]) == 3, [t.name for t in cached()["tools"]]


def test_the_new_tool_is_among_them():
    assert "count_docs" in {t.name for t in cached()["tools"]}


def test_the_new_tool_takes_no_parameters():
    """The empty bracket in the client's output, asserted."""
    tool = next(t for t in cached()["tools"] if t.name == "count_docs")
    assert list(tool.inputSchema.get("properties", {})) == []


def test_the_description_is_the_docstring():
    """FastMCP publishes the docstring as the tool's description -- so a docstring
    here is not documentation for humans, it is the contract the caller reads."""
    tool = next(t for t in cached()["tools"] if t.name == "count_docs")
    assert tool.description.splitlines()[0] == (
        "Total number of documentation snippets across the whole knowledge base.")


def test_the_count_matches_the_knowledge_base():
    assert cached()["count"] == EXPECTED_TOTAL == 4


def test_the_untouched_client_discovered_it_too():
    """The claim that matters: not that WE can call the new tool, but that the
    chapter's client -- whose source has not changed -- lists it."""
    out = cached()["client_output"]
    assert "count_docs[]" in out, out
    assert "search_docs" in out and "list_topics" in out


def test_the_clients_source_was_not_edited():
    """Guard against the cheat this file would be if it quietly rewrote the client.

    Hash the client's source, run everything, hash it again. Comparing against
    `git show HEAD` was the first version and it was wrong for a reason worth
    keeping: it failed whenever the client had uncommitted edits of its own,
    which tests the repository's state rather than this file's behaviour."""
    import hashlib
    src = Path(client.__file__)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    cached()                                     # everything this file does
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before


if __name__ == "__main__":
    demo_a_tool_the_client_never_heard_of()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
