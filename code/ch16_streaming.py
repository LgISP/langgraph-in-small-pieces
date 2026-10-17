"""Chapter 16 - Streaming.

Streaming is just choosing WHAT the graph yields as it runs, by passing a
`stream_mode`. The graph and the node bodies never change; only the caller
decides whether to watch whole-state snapshots, per-node updates, LLM tokens,
or the node's own progress events.

Run:  python code/ch16_streaming.py
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from typing import NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",  # your local server, not the OpenAI cloud
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",      # a capable open model that follows instructions
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


class State(TypedDict):
    question: str                     # input, set by the caller
    docs: NotRequired[list[str]]      # written by search_docs
    answer: NotRequired[str]          # written by generate


def search_docs_node(state: State) -> dict:
    # writer() broadcasts on the `custom` stream only: nothing it sends lands in
    # State or the checkpoint (see the chapter's Gotcha).
    writer = get_stream_writer()
    writer({"progress": "searching the knowledge base"})

    docs = [
        "Kafka Manual: set spring.kafka.listener.type=batch to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ]

    writer({"progress": f"found {len(docs)} document(s)"})
    return {"docs": docs}   # lands in State; with a checkpointer, also in a checkpoint


def generate_node(state: State) -> dict:
    sources = "\n".join(f"- {d}" for d in state.get("docs", []))
    prompt = f"""
        Answer the question concisely, using only the sources below.

        Question: {state['question']}

        Sources:
        {sources}

        Answer:"""
    answer = llm.invoke(prompt).content.strip()
    return {"answer": answer}


builder = StateGraph(State)
builder.add_node("search_docs", search_docs_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "search_docs")
builder.add_edge("search_docs", "generate")
builder.add_edge("generate", END)
graph = builder.compile()


QUESTION = "How do I configure Kafka batch consumers?"


def demo_updates() -> None:
    print("\n===== stream_mode='updates' (per-node delta) =====")
    for chunk in graph.stream({"question": QUESTION}, stream_mode="updates"):
        for node, delta in chunk.items():
            print(f"  [{node}] wrote fields {list(delta.keys())}")


def demo_messages() -> None:
    print("\n===== stream_mode='messages' (token by token) =====")
    # end="" keeps tokens on one line; flush=True prints each one immediately
    # instead of waiting for a full buffer, which is what makes it look live
    print("  ", end="", flush=True)
    for msg_chunk, metadata in graph.stream(
        {"question": QUESTION}, stream_mode="messages"
    ):
        if metadata.get("langgraph_node") == "generate" and msg_chunk.content:
            print(msg_chunk.content, end="", flush=True)
    print()


def demo_custom() -> None:
    print("\n===== stream_mode='custom' (progress events you emit) =====")
    for event in graph.stream({"question": QUESTION}, stream_mode="custom"):
        print(f"  progress -> {event['progress']}")


def demo_combined() -> None:
    # A node's `updates` delta arrives once it FINISHES; its `messages` tokens
    # arrive WHILE it runs, so the answer streams out between the two lines.
    print("\n===== stream_mode=['updates', 'messages'] ((mode, data) tuples) =====")
    for mode, data in graph.stream(
        {"question": QUESTION}, stream_mode=["updates", "messages"]
    ):
        if mode == "updates":
            for node, delta in data.items():
                print(f"\n  [{node}] finished, wrote {list(delta.keys())}")
        elif mode == "messages":
            msg_chunk, metadata = data
            if metadata.get("langgraph_node") == "generate" and msg_chunk.content:
                print(msg_chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    demo_updates()
    demo_messages()
    demo_custom()
    demo_combined()
