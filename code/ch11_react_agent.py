"""Chapter 11 — the framework runs the loop: ToolNode, tools_condition, and then
create_agent in one line.

The hand-written loop in ch11_handwritten_loop.py showed the mechanic. Here the
framework does the execute-and-append for us, but you still wire the model<->tools
loop yourself: a `model` node that may emit tool calls, a `ToolNode` that runs
them, `tools_condition` to route between them, and one back-edge to close the loop.
Then create_agent (from the langchain package, pip install langchain==1.4.4)
collapses that whole graph into a single call.

Same doc-bot tools as Chapter 5/10: search_docs and web_search.

Run:  python code/ch11_react_agent.py
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from langchain.agents import create_agent   # the prebuilt agent's home since 1.0
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# The same local model as every Part II chapter (LM Studio, no paid cloud).
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="openai/gpt-oss-20b",
    temperature=0,
)


KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch "
        "to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ],
    "redis": [
        "Redis Manual: set the connection pool via spring.data.redis.*.",
    ],
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
    whenever the user asks how to configure something, about an error message, or
    how to do a technical task."""
    q = query.lower()
    hits: list[str] = []
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in q:
            hits.extend(docs)
    return "\n".join(hits) if hits else "No matching local documents."


@tool
def web_search(query: str) -> str:
    """Search the open web. Use this when the local docs come up empty, or the
    question is about something recent."""
    # PLACEHOLDER: not a real web search — a canned string so the loop has something
    # to run, and it stays a stub for the whole book.
    return f"[web] community thread about “{query}” (example result)"


tools = [search_docs, web_search]

# bind_tools lets the model EMIT tool calls; running them is ToolNode's job below.
llm_with_tools = llm.bind_tools(tools)


def call_model(state: MessagesState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# MessagesState: one `messages` field with the add_messages reducer (Chapter 12).
builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")           # the back-edge: results return to model
graph = builder.compile()


# The same graph from one call; it binds the tools internally, so hand it the bare
# llm. (langgraph.prebuilt.create_react_agent is the deprecated older name.)
react = create_agent(llm, tools)


def show_trace(app, question: str) -> None:
    print(f"\n===== question: {question} =====")
    # recursion_limit is the loop's fuse (the graph cousin of the hand loop's
    # max_tool_turns, Ch 4/10): cap total model<->tools super-steps per run.
    result = app.invoke(
        {"messages": [("user", question)]},
        {"recursion_limit": 10},
    )
    for m in result["messages"]:
        kind = type(m).__name__
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"  [model] calls {tc['name']}(query={tc['args'].get('query')!r})")
        elif kind == "ToolMessage":
            print(f"  [tools] returned {m.content!r}")
        elif kind == "AIMessage" and m.content:
            print(f"  [model] final: {excerpt(m.content)}")


if __name__ == "__main__":
    print("########## hand-wired graph: ToolNode + tools_condition ##########")
    show_trace(graph, "How do I configure Kafka batch consumers?")
    show_trace(graph, "How do I write a Python decorator?")

    print("\n########## create_agent: the same loop, one line ##########")
    show_trace(react, "How do I configure Kafka batch consumers?")
    show_trace(react, "How do I write a Python decorator?")
