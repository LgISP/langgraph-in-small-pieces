"""Chapter 15 - Memory & Long Conversations: multi-turn chat + summarization.

Two demos on the same one-node doc-bot:

  demo_multiturn() - a checkpointer keyed by thread_id IS the conversation
                     memory. Send only the NEW message each turn; the history is
                     stored by the checkpointer and appended by add_messages.
  demo_summary()   - when the history grows too long, compress the older turns
                     into a running summary so the context stays bounded.

Runs against your local model server (see Chapter 10); nothing here talks to a paid
cloud.
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",  # your local server, not the OpenAI cloud
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",      # a capable open model that follows instructions
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
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


# ===========================================================================
# 1. Multi-turn chat: the checkpointer IS the conversation memory
# ===========================================================================
class ChatState(TypedDict):
    # the built-in message reducer: appends (merging by id) instead of overwriting
    messages: Annotated[list, add_messages]


def chat_node(state: ChatState) -> dict:
    system = SystemMessage(
        content="You are a documentation assistant. Answer concisely, "
        "in no more than three sentences."
    )
    reply = llm.invoke([system] + state["messages"])
    return {"messages": [reply]}


def build_chat_graph():
    builder = StateGraph(ChatState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=InMemorySaver())


def send(graph, thread_id: str, text: str) -> None:
    """Send one message to a conversation and print the reply.

    Each turn passes ONLY the new user message; the checkpointer supplies the
    history and add_messages appends to it.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    reply = result["messages"][-1].content   # the last message is this turn's reply
    print(f"  [{thread_id}] user: {text}")
    print(f"  [{thread_id}] bot:  {excerpt(reply)}\n")


def demo_multiturn() -> None:
    graph = build_chat_graph()

    # Conversation A (thread_id = conv-1): several turns that depend on context.
    print("===== conversation conv-1 =====")
    send(graph, "conv-1", "How do I configure Kafka batch consumers in Spring Boot?")
    send(graph, "conv-1", "What are the pitfalls?")       # "the pitfalls" of WHAT? -> tests memory
    send(graph, "conv-1", "What did I ask about first?")  # tests recall of the whole history

    # Conversation B (thread_id = conv-2): a brand-new thread, no shared context.
    print("===== conversation conv-2 (a fresh thread - no shared context) =====")
    send(graph, "conv-2", "What are the pitfalls?")       # no earlier turn -> "it" is undefined

    # The full history conv-1 accumulated, stored by the checkpointer.
    print("===== history accumulated on conv-1 =====")
    state = graph.get_state({"configurable": {"thread_id": "conv-1"}})
    for m in state.values["messages"]:
        who = "user" if isinstance(m, HumanMessage) else "bot "
        print(f"  {who}: {m.content[:48]}")
    print(f"  {len(state.values['messages'])} messages (3 turns = 3 questions + 3 answers)")


# ===========================================================================
# 2. Long conversation: summarize older turns to keep the context bounded
# ===========================================================================
KEEP_RECENT = 4   # keep roughly the last 4 messages verbatim (rounded up to a
                  # whole turn); fold older ones into summary


class SummaryState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: NotRequired[str]   # a running summary of the turns we dropped (absent on turn 1)


def summarize_node(state: SummaryState) -> dict:
    """When the history is long, fold the older messages into `summary` and
    drop them from the state with RemoveMessage, so the state stays small."""
    msgs = state["messages"]
    # Fold older turns once the history outgrows the recent window. Walk the
    # cutoff back to a turn boundary (a HumanMessage) so we only ever fold
    # COMPLETE user/assistant turns and never orphan an answer in the window.
    cutoff = max(0, len(msgs) - KEEP_RECENT)
    while cutoff > 0 and not isinstance(msgs[cutoff], HumanMessage):
        cutoff -= 1
    old = msgs[:cutoff]   # whole turns only, ready to compress away
    if not old:
        return {}   # nothing complete to fold yet

    old_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in old
    )
    prev = state.get("summary", "")
    prompt = (
        "Compress the conversation below into a short summary that keeps the key "
        "facts and conclusions.\n"
        f"Existing summary: {prev or '(none)'}\n"
        f"New turns to fold in:\n{old_text}\n\n"
        "Updated summary:"
    )
    new_summary = llm.invoke(prompt).content.strip()

    removals = [RemoveMessage(id=m.id) for m in old]
    print(f"      (summarize: folded {len(old)} old message(s) into the summary; "
          f"kept the recent turns verbatim)")
    return {"summary": new_summary, "messages": removals}


def summary_chat_node(state: SummaryState) -> dict:
    system_text = "You are a documentation assistant. Answer concisely, in no more than two sentences."
    if state.get("summary"):
        # The summary carries the dropped turns forward as background.
        system_text += f"\n\nEarlier conversation summary: {state['summary']}"
    reply = llm.invoke([SystemMessage(content=system_text)] + state["messages"])
    return {"messages": [reply]}


def build_summary_graph():
    builder = StateGraph(SummaryState)
    builder.add_node("summarize", summarize_node)
    builder.add_node("chat", summary_chat_node)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=InMemorySaver())


def demo_summary() -> None:
    graph = build_summary_graph()
    turns = [
        "How do I configure Kafka batch consumers?",
        "What are the pitfalls?",
        "How do I configure a Redis connection pool?",
        "How do I optimize a slow MySQL query?",
        "What was the very first topic we discussed?",   # tests recall after compression
    ]
    config = {"configurable": {"thread_id": "long-1"}}
    for text in turns:
        result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
        print(f"  user: {text}")
        print(f"  bot:  {excerpt(result['messages'][-1].content)}")
        kept = len(result["messages"])
        summ = result.get("summary", "")
        tail = f" | summary: {excerpt(summ)}" if summ else ""
        print(f"  state now holds {kept} message(s){tail}\n")


if __name__ == "__main__":
    print("########## multi-turn chat (checkpointer = conversation memory) ##########\n")
    demo_multiturn()
    print("\n########## long conversation (summarize older turns) ##########\n")
    demo_summary()
