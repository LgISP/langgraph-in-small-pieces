"""Chapter 16 Challenge — Silver: bullhorn vs logbook, proven in one script.

The Gotcha claimed `custom` progress vanishes with the run while `State` stays
inspectable. This proves the second half. `search_docs` reports each progress
step **twice**: it `writer(...)`s it onto the `custom` stream (the bullhorn) and
it *also* `return`s it into a `progress` field carrying an `operator.add` reducer
(the logbook). We compile with an `InMemorySaver` and a `thread_id`, stream the
run, and then — after the stream is over — call `graph.get_state(config)` and
find the `progress` list sitting in `State`, while the `custom` events we streamed
are nowhere in that state, because we never stored them.

To make the proof unmistakable, the bullhorn shout carries a live-only ` (live)`
tag that the logbook copy does not. So we can check the exact bullhorn text landed
on the `custom` stream *and* is absent from the persisted `State` — the two
channels really are independent, and only the logbook (the return) survives.

Honest scope: `InMemorySaver` keeps state in *this process's* memory, so this
shows *inspectability*, not crash survival — kill the process and the saved state
dies too. A durable checkpointer (SQLite/Postgres, per the save-resume chapter)
is what lets the same `progress` list survive a real restart.

The model in `generate` is the book's real local model (Qwen3.8 behind LM
Studio). The bullhorn/logbook proof is about the two channels, not the model, so
the tests assert the progress entries — which the nodes write in code — and
never the answer's wording.

Run:  python code/answers/ch16_durable_progress.py   (see answers/README.md)
"""

import itertools
import operator
from typing import Annotated, NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph


# The book's real local model — the chapter's local-server setup.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)

LIVE_TAG = " (live)"   # rides only on the bullhorn, never on the logbook copy


# --- State with a progress logbook: operator.add appends across nodes ---------
class State(TypedDict):
    question: str
    docs: NotRequired[list[str]]
    answer: NotRequired[str]
    progress: Annotated[list[str], operator.add]   # the logbook: reducer appends


def search_docs_node(state: State) -> dict:
    writer = get_stream_writer()
    docs = [
        "Kafka Manual: set spring.kafka.listener.type=batch to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ]
    logged: list[str] = []
    for msg in ["searching the knowledge base", f"found {len(docs)} document(s)"]:
        writer({"progress": msg + LIVE_TAG})   # bullhorn: shouts now, stores nothing
        logged.append(msg)                     # logbook: the canonical, clean fact
    # Only THIS dict reaches State (the bullhorn shouts above do not):
    return {"docs": docs, "progress": logged}


def generate_node(state: State) -> dict:
    sources = "\n".join(f"- {d}" for d in state.get("docs", []))
    prompt = (f"Answer the user's question concisely, using only:\n"
              f"{sources}\n\nQ: {state['question']}")
    answer = llm.invoke(prompt).content.strip()
    # generate contributes to the logbook too, so you can see operator.add append.
    return {"answer": answer, "progress": ["answer generated"]}


builder = StateGraph(State)
builder.add_node("search_docs", search_docs_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "search_docs")
builder.add_edge("search_docs", "generate")
builder.add_edge("generate", END)
graph = builder.compile(checkpointer=InMemorySaver())


QUESTION = "How do I configure Kafka batch consumers?"

# A fresh thread per run: reusing a thread_id would let operator.add keep
# appending across runs and make the counts non-deterministic.
_thread_seq = itertools.count()


_CACHED: list = []


def run_once():
    """Stream the run on its own thread, then read State back. Returns
    (bullhorn_events, persisted_progress). One real run is shared by the tests —
    the assertions all describe the same run's two channels."""
    if _CACHED:
        return _CACHED[0]
    config = {"configurable": {"thread_id": f"silver-{next(_thread_seq)}"}}
    # Consume the CUSTOM stream live — this is the only place these events exist.
    bullhorn = [e["progress"] for e in graph.stream(
        {"question": QUESTION}, config, stream_mode="custom")]
    # The stream is over. The logbook is still readable from the checkpoint.
    persisted = graph.get_state(config).values["progress"]
    _CACHED.append((bullhorn, persisted))
    return bullhorn, persisted


def demo_durable_progress() -> None:
    print("\n===== bullhorn (custom stream) vs logbook (State) =====")
    bullhorn, persisted = run_once()
    print("  bullhorn — streamed live off `custom`, then gone:")
    for m in bullhorn:
        print(f"      shout -> {m}")
    print("  logbook — read back from State AFTER the stream ended:")
    for m in persisted:
        print(f"      state -> {m}")
    print("  Note the ` (live)` tag: it rode the bullhorn but never reached State.")


# --- tests -------------------------------------------------------------------
def test_logbook_persists_in_state_after_the_stream():
    _, persisted = run_once()
    # search_docs logged 2 (clean, no tag), generate appended 1 — reducer joined them.
    assert persisted == [
        "searching the knowledge base",
        "found 2 document(s)",
        "answer generated",
    ]


def test_bullhorn_was_delivered_on_the_custom_stream():
    bullhorn, _ = run_once()
    # The same two facts traveled the custom channel, each wearing the live tag.
    assert bullhorn == [
        "searching the knowledge base" + LIVE_TAG,
        "found 2 document(s)" + LIVE_TAG,
    ]


def test_bullhorn_shouts_never_reached_state():
    # The live-only tag is our fingerprint: if any bullhorn text had been stored,
    # the tag would show up somewhere in State. It does not.
    _, persisted = run_once()
    assert all(LIVE_TAG not in entry for entry in persisted)


def test_custom_payloads_are_not_a_state_field():
    # The raw {"progress": ...} objects we streamed are not stored anywhere in the
    # checkpointed State — `custom` is a broadcast, not a channel of State.
    config = {"configurable": {"thread_id": f"silver-{next(_thread_seq)}"}}
    streamed = list(graph.stream({"question": QUESTION}, config, stream_mode="custom"))
    values = graph.get_state(config).values
    assert "custom" not in values                      # no hidden custom store
    assert all(evt not in values.values() for evt in streamed)   # payloads unstored


if __name__ == "__main__":
    demo_durable_progress()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
