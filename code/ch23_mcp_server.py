"""Chapter 23 - MCP Server: wrap a knowledge base as standard tools.

MCP (Model Context Protocol) is a standard that DECOUPLES the side that
provides capability (the server) from the side that uses it (an LLM app, the
client). You package your capability as an MCP server, and any MCP client -
Claude Desktop, Claude Code, or your own LangGraph agent - can DISCOVER and
CALL it, with no per-agent adapter code.

The analogy: MCP is to "an agent plugging in tools" what USB is to "a computer
plugging in peripherals" - one uniform connector.

An MCP server can expose three kinds of things:
    - tools     : callable functions (the focus of this file)
    - resources : readable data (like a GET, read-only)
    - prompts   : reusable prompt templates

Here FastMCP exposes two tools over a tiny dict-backed knowledge base. In
production you swap the body of search_docs for a real vector store (Chroma,
pgvector, ...) - the tool's signature, and therefore every client, stays the
same.

Run:  usually you do NOT run this by hand; the client launches it (see
      ch23_mcp_client.py). To sanity-check it starts, you can run this file and
      it will wait on stdin.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("doc-bot-kb")   # the name a connecting client sees

KNOWLEDGE_BASE = {
    "kafka": [
        "Kafka batch consumption: set spring.kafka.listener.type=batch to enable it.",
        "Kafka tuning: pair a batch consumer with max-poll-records to raise throughput.",
    ],
    "redis": ["Redis connection pool: configure it via spring.data.redis.lettuce.pool.*"],
    "mysql": ["MySQL indexing: add a B+ tree index on hot query columns to speed up slow queries."],
}


# ---------------------------------------------------------------------------
# @mcp.tool() registers an ordinary function AS an MCP tool:
#   - the function name  -> the tool name
#   - the argument hints -> the tool's inputSchema (how a client calls it)
#   - the return hint     -> its outputSchema; every result also carries
#                            structuredContent alongside the text blocks
#   - the docstring      -> the tool description (how a client/LLM decides
#                           whether and how to use it)
# ---------------------------------------------------------------------------
@mcp.tool()
def search_docs(query: str) -> list[str]:
    """Search the technical-docs knowledge base and return snippets relevant to query.

    Args:
        query: the user's question or keywords
    """
    q = query.lower()
    hits = []
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in q:
            hits.extend(docs)
    return hits or ["(no relevant documents found)"]


@mcp.tool()
def list_topics() -> list[str]:
    """List which topics the knowledge base currently covers."""
    return list(KNOWLEDGE_BASE.keys())


if __name__ == "__main__":
    mcp.run()   # stdio by default; SSE / HTTP transports are also available
