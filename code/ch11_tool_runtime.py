"""Chapter 11 - ToolRuntime: what a tool may see that the model may not.

The model writes a tool's arguments. Anything the model must NOT choose --
who is asking, what the conversation already said, where to write progress --
has to reach the tool by another road. That road is `ToolRuntime`: annotate a
parameter with it and the framework injects it at call time. The annotation is
what does the work -- `runtime` is only the conventional name -- and the
parameter never appears in the schema the model is shown.

Run:  python code/ch11_tool_runtime.py

Needs LM Studio on http://localhost:1234/v1 with openai/gpt-oss-20b loaded
(same connection block as the rest of the chapter).
"""

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

KNOWLEDGE_BASE = {
    "kafka": ["Kafka Manual: use spring.kafka.listener.type=batch to enable"],
    "redis": ["Redis Manual: configure spring.data.redis.lettuce.pool.max-active"],
}


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


@tool
def search_docs(query: str) -> str:
    """Search the internal technical-documentation knowledge base. Use this
    whenever the user asks how to configure something."""
    q = query.lower()
    hits = [d for topic, docs in KNOWLEDGE_BASE.items() if topic in q for d in docs]
    return "\n".join(hits) if hits else "No matching local documents."


@tool
def conversation_so_far(runtime: ToolRuntime) -> str:
    """Report the questions the user has already asked in this conversation.
    Use this when the user refers back to something they asked earlier."""
    asked = [m.content for m in runtime.state["messages"] if m.type == "human"]
    return " | ".join(asked) if asked else "(nothing yet)"


def show_schemas() -> None:
    """What the model is told about each tool -- and what it is not told."""
    print("search_docs.args         :", search_docs.args)
    print("conversation_so_far.args :", conversation_so_far.args)
    # The runtime parameter survives on the raw args_schema; that is plumbing,
    # not the wire format. `.args` is the surface that answers "what will the
    # model be shown?"
    print("raw args_schema fields   :", list(conversation_so_far.args_schema.model_fields))


def main() -> None:
    show_schemas()

    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="openai/gpt-oss-20b",
        temperature=0,
    )
    agent = create_agent(llm, [search_docs, conversation_so_far])

    out = agent.invoke({"messages": [
        ("user", "How do I configure Kafka batch consumers?"),
        ("assistant", "Set spring.kafka.listener.type=batch."),
        ("user", "What have I asked you so far?"),
    ]})

    print()
    for m in out["messages"]:
        if m.type == "ai" and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  [model] calls {tc['name']}({tc['args']})")
        elif m.type == "tool":
            print(f"  [tools] returned {m.content!r}")
    print(f"  [model] final: {excerpt(out['messages'][-1].content)}")


if __name__ == "__main__":
    main()
