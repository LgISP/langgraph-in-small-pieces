"""Chapter 23 Challenge — Bronze, part 1: the chapter's server plus one tool.

The challenge says to add `count_docs()` to `ch23_mcp_server.py`. An answer file
must not edit the chapter's code, so this imports that server and registers one
more tool on the same `FastMCP` instance -- which is the same thing the decorator
does when you type it into the original file, and it keeps the two definitions
next to each other where the diff is easy to read.

Run indirectly: ch23_a_tool_the_client_never_heard_of.py launches this over stdio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ch23_mcp_server import KNOWLEDGE_BASE, mcp    # noqa: E402


@mcp.tool()
def count_docs() -> int:
    """Total number of documentation snippets across the whole knowledge base."""
    return sum(len(snippets) for snippets in KNOWLEDGE_BASE.values())


if __name__ == "__main__":
    mcp.run()
