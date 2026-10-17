"""Chapter 15 Challenge - Silver: remember a fact the model was TOLD, not one you
hard-coded.

In `code/ch15_store.py` the facts are written from *outside* the conversation by
`remember_fact`. This closes the loop: a `remember` node watches each turn, and
when the user *states* a durable preference ("from now on, answer in Python") it
writes that fact to the Store itself with `store.put`. A later thread - a fresh
`thread_id` whose checkpoint is empty - then recalls a fact the user gave in an
*earlier* conversation, because the fact lives in the Store (keyed by user), not
in any one thread's checkpoint.

The extraction uses **structured output** (@sec-structured-output): the model is
asked for a small `Fact` schema, so the code branches on typed fields instead of
scraping prose. The schema guarantees the *shape*; whether a sentence is worth
saving is the model's *judgment* - the shape-versus-choice line from
@sec-real-llm. Both the chat turns and the extraction are REAL calls to the
book's local model (Qwen3.8 behind LM Studio), so the wording of every
answer and every distilled fact varies; the tests assert structure - what got
stored, under which key, recalled on which thread - never verbatim text. One
real-model consequence shaped the schema: a fake could promise to always phrase a
fact as "Prefers answers in X", so the original design keyed the Store on that
prefix. A real model phrases the fact differently every time, so the schema now
carries an explicit `kind` field and the *kind* is the stable key - restating a
language preference updates one entry instead of piling up near-duplicates.

Run:  python code/answers/ch15_learned_facts.py   (see answers/README.md)
"""

import re
import warnings
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


# The book's real local model — the chapter's local-server setup.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


# --- the structured-output schema --------------------------------------------
# Typed fields to branch on. `worth_remembering` is the judgment; `kind` is the
# stable identity a Store key can hang on; `text` is the distilled fact to file.
# The type guarantees the shape, not the judgment.
class Fact(BaseModel):
    worth_remembering: bool = Field(
        description="true ONLY if the user stated a durable preference that should "
        "shape future conversations; a one-off question is false."
    )
    kind: Literal["language_preference", "other"] = Field(
        description="language_preference if the fact is about which programming "
        "language the user wants answers in; otherwise other."
    )
    text: str = Field(
        description="The distilled fact, one short sentence; empty if not worth "
        "remembering. A Store holds conclusions, not transcripts."
    )


extractor = llm.with_structured_output(Fact, method="json_schema")

# One extraction prompt, shared by the node and the tests — the judgment is
# prompt-sensitive, so they must ask the model the same way.
EXTRACT_SYSTEM = ("Extract a durable user preference worth remembering "
                  "across future conversations, if the message states one.")


def _last_human_content(messages) -> str:
    if isinstance(messages, str):
        return messages
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


# --- the graph: remember (write learned facts) -> chat (recall them) ---------
@dataclass
class Context:
    user_id: str   # who this run is for - a per-run dependency, not State


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _fact_key(fact: Fact) -> str:
    """A stable key so re-stating a preference UPDATES it instead of piling up a
    duplicate. The model's `kind` field carries the identity: every language
    preference shares one key, whatever wording the model chose for `text`."""
    if fact.kind == "language_preference":
        return "language_preference"
    slug = re.sub(r"[^a-z0-9]+", "-", fact.text.lower()).strip("-")
    return f"pref-{slug[:32]}"


# How many facts `chat` recalled on each call, keyed by (thread ignored) run order —
# lets a test check recall structurally without parsing the model's prose.
LAST_RECALL: dict[str, int] = {}


def remember_node(state: State, runtime: Runtime[Context]) -> dict:
    """Watch the newest user turn; if it STATES a durable preference, extract it
    with structured output and file it in the Store under this user's namespace.
    This is the loop `ch15_store.py` left open: the fact now comes from the
    conversation, not from a hard-coded `remember_fact` call outside it."""
    store = runtime.store
    user_id = runtime.context.user_id
    last_user = state["messages"][-1]   # add_messages just appended this turn's input

    fact = extractor.invoke([SystemMessage(content=EXTRACT_SYSTEM), last_user])
    if fact.worth_remembering:
        namespace = ("memories", user_id)
        store.put(namespace, _fact_key(fact), {"text": fact.text})
        print(f"  [remember] learned about {user_id}: {fact.text!r} "
              f"(kind={fact.kind})")
    # The Store is written as a side effect; nothing about it flows through State.
    return {}


def chat_node(state: State, runtime: Runtime[Context]) -> dict:
    store = runtime.store
    user_id = runtime.context.user_id
    namespace = ("memories", user_id)
    facts = [item.value["text"] for item in store.search(namespace)]
    LAST_RECALL[user_id] = len(facts)
    print(f"  [chat] recalled {len(facts)} fact(s) about {user_id}: {facts}")

    system_text = ("You are a documentation assistant. Answer concisely, "
                   "in no more than two sentences.")
    if facts:
        # The recalled facts steer the answer - even on a brand-new thread.
        system_text += "\n\nKnown facts about this user: " + "; ".join(facts)
    reply = llm.invoke([SystemMessage(content=system_text)] + state["messages"])
    return {"messages": [reply]}


def build_graph(store: BaseStore):
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("remember", remember_node)   # first: learn from this turn
    builder.add_node("chat", chat_node)            # then: answer, honoring facts
    builder.add_edge(START, "remember")
    builder.add_edge("remember", "chat")
    builder.add_edge("chat", END)
    # Two memories on one graph: the checkpointer (per-thread history) and the
    # Store (cross-thread facts).
    return builder.compile(checkpointer=InMemorySaver(), store=store)


def ask(graph, thread_id: str, user_id: str, text: str) -> str:
    """Send one turn and return the bot's reply. thread_id (which conversation)
    rides config; user_id (which user) rides runtime context."""
    result = graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread_id}},
        context=Context(user_id=user_id),
    )
    reply = result["messages"][-1].content
    print(f"  [{thread_id}] user: {text}")
    print(f"  [{thread_id}] bot:  {reply}\n")
    return reply


# --- tests -------------------------------------------------------------------
# Real-model note: every assertion is structural (what was stored, under which
# key, recalled where) or a loose keyword check — never verbatim model text.
def test_structured_output_is_typed_and_parseable():
    # The extraction comes back as a typed Fact, not prose to scrape.
    fact = extractor.invoke([
        SystemMessage(content=EXTRACT_SYSTEM),
        HumanMessage(content="From now on, answer in Python."),
    ])
    assert isinstance(fact, Fact)
    assert fact.worth_remembering is True
    assert fact.kind == "language_preference"
    assert "python" in fact.text.lower()


def test_plain_question_is_not_remembered():
    # A normal question states no durable preference -> nothing worth saving.
    fact = extractor.invoke([
        SystemMessage(content=EXTRACT_SYSTEM),
        HumanMessage(content="How do I parse JSON?"),
    ])
    assert fact.worth_remembering is False


def test_stated_preference_is_written_to_the_store():
    store = InMemoryStore()
    graph = build_graph(store)
    ask(graph, "conv-A", "u_alice", "From now on, answer in Python.")
    facts = [i.value["text"] for i in store.search(("memories", "u_alice"))]
    assert len(facts) == 1                            # the node wrote it, not us
    assert "python" in facts[0].lower()


def test_fact_the_user_gave_crosses_threads():
    store = InMemoryStore()
    graph = build_graph(store)
    # Thread A: the user STATES a preference (never hard-coded by us).
    ask(graph, "conv-A", "u_alice", "From now on, answer in Python.")
    # Thread B: a BRAND-NEW thread, empty checkpoint. The bot still honors it.
    reply = ask(graph, "conv-B", "u_alice", "How do I read a config file?")
    assert LAST_RECALL["u_alice"] == 1                # the Store fact was recalled
    assert "python" in reply.lower()                  # ...and it steered the answer
    # Prove the recall came from the Store, not a shared checkpoint: conv-B's
    # history holds only this single turn (1 user + 1 bot).
    state = graph.get_state({"configurable": {"thread_id": "conv-B"}})
    assert len(state.values["messages"]) == 2


def test_namespace_isolates_a_different_user():
    store = InMemoryStore()
    graph = build_graph(store)
    ask(graph, "conv-A", "u_alice", "From now on, answer in Python.")
    ask(graph, "conv-C", "u_bob", "How do I read a config file?")
    # Bob's turn recalled nothing: his namespace is empty, Alice's fact is hers.
    # (We assert the recall, not Bob's prose — an unconstrained model may well
    # pick Python for a generic answer all on its own.)
    assert LAST_RECALL["u_bob"] == 0
    assert store.search(("memories", "u_bob")) == []  # Bob's drawer is empty


def test_restating_updates_the_same_fact():
    store = InMemoryStore()
    graph = build_graph(store)
    ask(graph, "conv-A", "u_alice", "From now on, answer in Python.")
    ask(graph, "conv-A", "u_alice", "Actually, from now on use Rust.")
    items = list(store.search(("memories", "u_alice")))
    # The stable `kind`-derived key means the new preference REPLACES the old
    # one, not duplicates — whatever wording the model chose for `text`.
    assert len(items) == 1
    assert items[0].key == "language_preference"
    assert "rust" in items[0].value["text"].lower()


if __name__ == "__main__":
    store = InMemoryStore()
    graph = build_graph(store)

    print("----- conv-A: Alice states a durable preference -----")
    ask(graph, "conv-A", "u_alice", "From now on, answer in Python.")

    print("----- conv-B: Alice again, a BRAND-NEW thread -----")
    ask(graph, "conv-B", "u_alice", "How do I read a config file?")

    print("----- conv-C: Bob, a different user (namespace isolates him) -----")
    ask(graph, "conv-C", "u_bob", "How do I read a config file?")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
