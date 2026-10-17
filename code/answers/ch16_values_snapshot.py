"""Chapter 16 Challenge — Bronze: the snapshot view, `stream_mode="values"`.

The chapter streamed `updates`, `messages`, and `custom` but skipped `values`.
This adds `demo_values()`. `updates` gave *two* chunks (one delta per node), so
you might predict `values` gives two as well — but it gives **three**. `values`
yields the *whole* `State` after every step, and it counts the initial input as a
step: one snapshot for the bare input, then one after `search_docs`, then one
after `generate`. Each yield is a full cumulative snapshot, not a delta.

Reach for `values` over `updates` when a consumer needs the entire `State` at
each step — re-rendering a full UI from scratch, or debugging accumulated state —
rather than just the single field a node happened to change.

The model in `generate` is the book's real local model (Qwen3.8 behind LM
Studio) — the snapshot counts this Bronze is about are a property of the graph,
not of the model, so the tests assert keys and counts and never the answer's
wording.

Run:  python code/answers/ch16_values_snapshot.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


# The book's real local model — the chapter's local-server setup.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


# --- the chapter's linear pipeline: search_docs -> generate -------------------
class State(TypedDict):
    question: str                     # input, set by the caller
    docs: NotRequired[list[str]]      # written by search_docs
    answer: NotRequired[str]          # written by generate


def search_docs_node(state: State) -> dict:
    # Stub lookup, exactly as in the chapter (custom writer calls omitted here —
    # this Bronze is only about the values view, and they don't affect it).
    docs = [
        "Kafka Manual: set spring.kafka.listener.type=batch to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ]
    return {"docs": docs}


def generate_node(state: State) -> dict:
    sources = "\n".join(f"- {d}" for d in state.get("docs", []))
    prompt = (f"Answer the user's question concisely, using only these sources:\n"
              f"{sources}\n\nQ: {state['question']}")
    return {"answer": llm.invoke(prompt).content.strip()}


builder = StateGraph(State)
builder.add_node("search_docs", search_docs_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "search_docs")
builder.add_edge("search_docs", "generate")
builder.add_edge("generate", END)
graph = builder.compile()


QUESTION = "How do I configure Kafka batch consumers?"


_CACHE: dict[str, list] = {}


def run_values() -> list[dict]:
    # One real run per mode is plenty: every test asserts the same run's shape.
    if "values" not in _CACHE:
        _CACHE["values"] = list(graph.stream({"question": QUESTION}, stream_mode="values"))
    return _CACHE["values"]


def run_updates() -> list[dict]:
    if "updates" not in _CACHE:
        _CACHE["updates"] = list(graph.stream({"question": QUESTION}, stream_mode="updates"))
    return _CACHE["updates"]


def demo_values() -> None:
    # stream_mode="values": yield the FULL State after every step. Print the keys
    # of each snapshot so you can watch State fill up, snapshot by snapshot.
    print("\n===== stream_mode='values' (whole-state snapshot each step) =====")
    for i, snapshot in enumerate(run_values()):
        print(f"  snapshot {i}: keys = {sorted(snapshot)}")
    print("  (updates gave 2 chunks; values gives 3 — the extra one is the "
          "initial input, before any node ran.)")


# --- tests -------------------------------------------------------------------
def test_values_yields_one_snapshot_per_step_plus_initial():
    # Two nodes ran, but values also emits the bare input snapshot first -> 3.
    assert len(run_values()) == 3


def test_first_snapshot_is_the_bare_input():
    first = run_values()[0]
    assert sorted(first) == ["question"]          # nothing written yet


def test_snapshots_are_cumulative_not_deltas():
    snaps = run_values()
    # Each snapshot is the whole State so far; keys only grow, never shrink.
    assert sorted(snaps[1]) == ["docs", "question"]              # after search_docs
    assert sorted(snaps[2]) == ["answer", "docs", "question"]    # after generate


def test_values_yields_exactly_one_more_than_updates():
    # The headline contrast: values = updates + 1, because of the input snapshot.
    assert len(run_updates()) == 2
    assert len(run_values()) == len(run_updates()) + 1


if __name__ == "__main__":
    demo_values()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
