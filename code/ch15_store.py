"""Chapter 15 - Long-term memory with a Store: facts that survive across threads.

A checkpoint remembers ONE conversation (per thread_id). A Store remembers a
USER across conversations: distilled facts, filed under a namespace, that any
future thread can read back.

  demo_store_api()    - the Store on its own: put / get / search, and how a
                        namespace scopes facts to a user.
  demo_cross_thread() - wire the Store into the doc-bot. Record facts about a
                        user once; recall them from a brand-new thread whose
                        conversation history is empty.

Runs against your local model server (see Chapter 10); nothing here talks to a paid
cloud.
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

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
# 1. The Store on its own: put / get / search, scoped by namespace
# ===========================================================================
def demo_store_api() -> None:
    store = InMemoryStore()

    # A namespace is a tuple that scopes facts - here, one per user.
    alice = ("memories", "u_alice")
    bob = ("memories", "u_bob")

    # put(namespace, key, value): value is a dict you choose the shape of.
    store.put(alice, "stack", {"text": "Works in Python, never Java"})
    store.put(alice, "level", {"text": "New to LangGraph"})
    store.put(bob, "stack", {"text": "Works in Go"})

    # get(namespace, key) returns one Item (or None); the payload is .value.
    print("get (alice, stack) ->", store.get(alice, "stack").value)

    # search(namespace) returns Items under this namespace prefix (up to 10 by
    # default; pass limit/offset to page through more).
    print("everything known about u_alice:")
    for item in store.search(alice):
        print(f"    {item.key}: {item.value['text']}")

    # A different namespace is a different drawer - Bob's facts are separate.
    print("everything known about u_bob:")
    for item in store.search(bob):
        print(f"    {item.key}: {item.value['text']}")


# ===========================================================================
# 2. The Store inside the doc-bot: recall a user across threads
# ===========================================================================
# Who this run is for is a per-run DEPENDENCY, not conversation data, so it rides
# runtime context (Chapter 10), never State.
@dataclass
class Context:
    user_id: str   # who this conversation is with


class State(TypedDict):
    messages: Annotated[list, add_messages]


def remember_fact(store: BaseStore, user_id: str, key: str, text: str) -> None:
    """Deliberately record a durable fact about a user. Unlike a checkpoint,
    which saves the whole state automatically, you write the Store ON PURPOSE."""
    store.put(("memories", user_id), key, {"text": text})


def chat_node(state: State, runtime: Runtime[Context]) -> dict:
    # facts are keyed by THIS user (runtime context), not this thread
    store = runtime.store
    user_id = runtime.context.user_id
    namespace = ("memories", user_id)
    facts = [item.value["text"] for item in store.search(namespace)]
    print(f"  [chat] recalled {len(facts)} fact(s) about {user_id}: {facts}")

    system_text = "You are a documentation assistant. Answer concisely, in no more than two sentences."
    if facts:
        # The recalled facts steer the answer, even on a brand-new thread.
        system_text += "\n\nKnown facts about this user: " + "; ".join(facts)
    reply = llm.invoke([SystemMessage(content=system_text)] + state["messages"])
    return {"messages": [reply]}


def build_store_graph(store: BaseStore):
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    # Two separate memories on the compiled graph: the checkpointer (per-thread
    # conversation history) and the Store (cross-thread facts).
    return builder.compile(checkpointer=InMemorySaver(), store=store)


def ask(graph, thread_id: str, user_id: str, text: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    # thread_id (which conversation) rides config; user_id (which user) rides
    # runtime context - both are per-run, neither is conversation data.
    result = graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config=config,
        context=Context(user_id=user_id),
    )
    print(f"  [{thread_id}] user: {text}")
    print(f"  [{thread_id}] bot:  {excerpt(result['messages'][-1].content)}\n")


def demo_cross_thread() -> None:
    store = InMemoryStore()
    graph = build_store_graph(store)

    # Deliberately record two durable facts about Alice - the Store's whole job.
    remember_fact(store, "u_alice", "stack", "Works in Python, never Java")
    remember_fact(store, "u_alice", "level", "New to LangGraph")

    # One conversation on one thread.
    print("----- conv-A: Alice -----")
    ask(graph, "conv-A", "u_alice", "How should I read a config file?")

    # A BRAND-NEW thread for the SAME user. Its checkpoint is empty - the bot has
    # zero conversation history here - yet the Store still knows Alice.
    print("----- conv-B: Alice again, a different thread -----")
    ask(graph, "conv-B", "u_alice", "How do I parse JSON?")

    # A different user: same namespace scheme, a different namespace -> no facts.
    print("----- conv-C: Bob, a different user -----")
    ask(graph, "conv-C", "u_bob", "How do I parse JSON?")


if __name__ == "__main__":
    print("########## the Store on its own (put / get / search) ##########\n")
    demo_store_api()
    print("\n########## the Store inside the doc-bot (recall across threads) ##########\n")
    demo_cross_thread()
