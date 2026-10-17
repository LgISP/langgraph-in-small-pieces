"""Chapter 23 - MCP Client: discover and call the tools a server exposes.

This shows the full MCP round trip:
    1) the client LAUNCHES the server as a child process (stdio transport)
    2) initialize -- the handshake
    3) list_tools -- DISCOVER which tools the server has (the client does not
                     need to know this in advance!)
    4) call_tool  -- CALL one of them and get the result back

The key point: this client does NOT hardcode what search_docs looks like. It
learns about the tools at run time by asking the server. That is the value of
MCP's decoupling: point the client at a different server and it automatically
sees the new tools, with no code change here.

Run:  python code/ch23_mcp_client.py
      (it launches ch23_mcp_server.py itself over stdio)
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Absolute path so we do not depend on the current working directory.
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch23_mcp_server.py")


async def main():
    # sys.executable: the interpreter running this client, which is the one
    # known to have mcp installed. A bare "python" may resolve to another venv.
    server_params = StdioServerParameters(command=sys.executable, args=[SERVER])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                      # (1) handshake
            print("Connected to the MCP server\n")

            tools = await session.list_tools()              # (2) discover
            print("Discovered tools (not hardcoded here; learned from the server):")
            for t in tools.tools:
                params = list(t.inputSchema.get("properties", {}).keys())
                print(f"  - {t.name}{params}  -- {t.description.splitlines()[0]}")

            # (3) call. A list[str] result arrives as one content block per element.
            # A tool call crosses a process boundary, so it gets a deadline. The
            # chapter's operational rule says "treat an MCP tool the way you treat
            # the network"; this is the line that makes the rule true of the code.
            # `asyncio.timeout` abandons the wait -- the child process may still be
            # working, which is the honest shape of a timeout across a boundary you
            # do not own.
            print("\nCall list_topics():")
            async with asyncio.timeout(10):
                r1 = await session.call_tool("list_topics", {})
            print("  ->", [item.text for item in r1.content])

            # (4) a query that hits the knowledge base ...
            print("\nCall search_docs(query='how do I configure kafka batch consumers'):")
            r2 = await session.call_tool(
                "search_docs", {"query": "how do I configure kafka batch consumers"}
            )
            for item in r2.content:
                print("  ->", item.text)

            # ... and one that misses, to show the empty-result path.
            print("\nCall search_docs(query='python decorators'):")
            r3 = await session.call_tool("search_docs", {"query": "python decorators"})
            for item in r3.content:
                print("  ->", item.text)


if __name__ == "__main__":
    asyncio.run(main())
