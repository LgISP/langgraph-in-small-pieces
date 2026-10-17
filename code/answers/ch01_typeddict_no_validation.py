"""Chapter 1 Challenge — Bronze: TypedDict does no runtime validation.

The chapter claims a `TypedDict` state is *just annotations* — LangGraph never
checks it at runtime. This proves it by feeding the single-node graph three
inputs a strict schema would reject, and observing what actually happens:

  * an EXTRA key not in the schema      -> silently dropped (kept only declared channels)
  * `question` as the WRONG type (int)  -> accepted, no complaint
  * a MISSING required key              -> NOT rejected upfront; it fails only when
                                           a node *reads* the absent key (KeyError)

The one-sentence takeaway: with a TypedDict the schema is documentation, not a
gate — problems surface at the point of use, not the point of entry.

Run:  python code/answers/ch01_typeddict_no_validation.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    answer: NotRequired[str]


def answer_node(state: State) -> dict:
    # Reads `question`. If it isn't there, THIS is where things break — not at
    # invoke() time, but here, the moment the key is actually needed.
    return {"answer": f"You asked: {state['question']!r}"}


def build_graph():
    b = StateGraph(State)
    b.add_node("answer", answer_node)
    b.add_edge(START, "answer")
    b.add_edge("answer", END)
    return b.compile()


graph = build_graph()


# --- tests -------------------------------------------------------------------
def test_extra_key_is_dropped():
    # A key not declared in State never reaches the final state.
    result = graph.invoke({"question": "hi", "nonsense": 999})
    assert "nonsense" not in result
    assert result["question"] == "hi"


def test_wrong_type_is_accepted():
    # TypedDict is annotation-only: an int where a str is declared runs fine.
    result = graph.invoke({"question": 123})  # type: ignore[dict-item]
    assert result["answer"] == "You asked: 123"


def test_missing_key_fails_only_on_read():
    # No validation at invoke; the KeyError happens inside state["question"].
    try:
        graph.invoke({})
    except KeyError as e:
        assert "question" in str(e)
    else:
        raise AssertionError("expected KeyError for the missing 'question'")


if __name__ == "__main__":
    print("extra key ->", graph.invoke({"question": "hi", "nonsense": 999}))
    print("wrong type ->", graph.invoke({"question": 123}))  # type: ignore[dict-item]
    try:
        graph.invoke({})
    except KeyError as e:
        print("missing key -> KeyError:", e)

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
