"""Chapter 11 — Tool-calling, the mechanic laid bare: a hand-written loop.

In Chapters 5 and 10 YOU decided which tool ran: the agent wrote a `route` and a
conditional edge dispatched it. Native tool-calling moves that decision into the
model. `@tool` + `bind_tools` teaches the model what it can call; the model then
*emits* a tool call — a name plus args, as data — and your code runs it and hands
the result back for the model to keep going.

This file writes that loop by hand, with no LangGraph, so you can see the one fact
the frameworks hide: the LLM never EXECUTES a tool. It only decides to. Every real
search_docs(query=...) call is CODE calling `tool.invoke(args)`.

Run:  python code/ch11_handwritten_loop.py
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# A real model behind a local OpenAI-compatible endpoint. Nothing here
# talks to a paid cloud — the whole book runs on your own machine.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",  # your local server, not the OpenAI cloud
    api_key="lm-studio",                   # the local server ignores the key
    model="openai/gpt-oss-20b",            # gpt-oss on purpose here — see the Gotcha on tool-choice
    temperature=0,                         # 0 = as reproducible as a real model gets
)


# The same small keyed knowledge base as the Chapter 5 / 10 doc-bot.
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


# `@tool` turns a plain function into a tool the LLM can call:
#   - the function name        -> the tool's name
#   - the type hints           -> the argument schema (so the model knows what to send)
#   - the docstring            -> the tool's description, which the model reads to
#                                 decide WHETHER and WHEN to call it  <- this is prompt

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


# bind_tools attaches the tools' schemas to the model. It does NOT let the model
# run anything — it only lets the model EMIT a tool call. Execution is still ours.
llm_with_tools = llm.bind_tools([search_docs, web_search])

tools_by_name = {"search_docs": search_docs, "web_search": web_search}


def run(question: str, max_tool_turns: int = 5) -> None:
    print(f"\n===== question: {question} =====")
    messages: list = [("user", question)]

    turns = 0
    while True:
        ai = llm_with_tools.invoke(messages)
        messages.append(ai)

        if not ai.tool_calls:
            print(f"  [model] final answer: {excerpt(ai.content)}")
            return

        # A fuse: a tool-happy model could keep asking forever, so bound the loop.
        # This is the hand-written cousin of the graph's recursion_limit (Ch 4/10).
        turns += 1
        if turns > max_tool_turns:
            print(f"  [code]  hit max_tool_turns={max_tool_turns}; stopping")
            return

        for tc in ai.tool_calls:
            print(f"  [model] calls {tc['name']}(query={tc['args'].get('query')!r})")
            tool = tools_by_name[tc["name"]]
            result = tool.invoke(tc["args"])
            print(f"  [code]  ran {tc['name']} -> {result!r}")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))


if __name__ == "__main__":
    run("How do I configure Kafka batch consumers?")   # docs cover this
    run("How do I write a Python decorator?")           # docs do NOT cover this
    run("What is the capital of France?")               # no tool needed at all
