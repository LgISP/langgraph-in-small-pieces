"""Chapter 12 Challenge — Silver: three windows, one state.

`StateGraph(InternalState, input_schema=InputState, output_schema=OutputState)`
splits the public shape from the internal one. This file runs the four
predictions from the challenge and shows that all four answers follow from one
sentence: THESE SCHEMAS ARE WINDOWS, NOT GUARDS.

1. RETURN VALUE. `invoke({"question": ...})` hands back ONLY the
   `output_schema` fields — `answer` — never `question` or `scratch`.
2. SMUGGLED INPUT. An extra key not in `input_schema` is dropped silently:
   no error, and no node ever sees it.
3. NO-SCHEMA WRITE. A node returning a key that appears in no schema loses
   that value just as silently — it lands in no channel and is gone.
4. MISSING INPUT. Leaving `question` out fails NOT at the door: the run
   starts, and the first node to read the missing key dies with a plain
   `KeyError`. A TypedDict window validates nothing — that early rejection
   is the Bronze's `BaseModel` job, not the window's.

Run:  python code/answers/ch12_schema_windows.py
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    answer: str


class InternalState(TypedDict):
    question: str
    scratch: str      # never leaves the graph
    answer: str


SEEN_BY_NODES: list[set] = []   # what each node actually received, per run


def think(state: InternalState) -> dict:
    SEEN_BY_NODES.append(set(state.keys()))
    return {"scratch": f"thinking about {state['question']}"}


def answer(state: InternalState) -> dict:
    SEEN_BY_NODES.append(set(state.keys()))
    return {
        "answer": f"answered: {state['scratch']}",
        "uninvited": "no schema lists me",   # prediction 3: written into the void
    }


builder = StateGraph(InternalState, input_schema=InputState, output_schema=OutputState)
builder.add_node("think", think)
builder.add_node("answer", answer)
builder.add_edge(START, "think")
builder.add_edge("think", "answer")
builder.add_edge("answer", END)
graph = builder.compile()


def test_1_output_window_filters_the_return():
    result = graph.invoke({"question": "kafka"})
    assert result == {"answer": "answered: thinking about kafka"}   # nothing else


def test_2_smuggled_input_vanishes_silently():
    SEEN_BY_NODES.clear()
    result = graph.invoke({"question": "kafka", "smuggled": 42})    # no error
    assert result == {"answer": "answered: thinking about kafka"}
    assert all("smuggled" not in seen for seen in SEEN_BY_NODES)    # no node saw it


def test_3_no_schema_write_vanishes_silently():
    result = graph.invoke({"question": "kafka"})                    # no error either
    assert "uninvited" not in result
    # and it isn't lurking in the internal state: a fresh run's think-node
    # never receives it back
    SEEN_BY_NODES.clear()
    graph.invoke({"question": "again"})
    assert all("uninvited" not in seen for seen in SEEN_BY_NODES)


def test_4_missing_input_fails_in_the_node_not_at_the_door():
    try:
        graph.invoke({})
    except KeyError as e:                   # raised inside `think`, mid-run
        assert e.args[0] == "question"
    else:
        raise AssertionError("expected a KeyError from the first read")


if __name__ == "__main__":
    print("returns:", graph.invoke({"question": "kafka"}))
    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
