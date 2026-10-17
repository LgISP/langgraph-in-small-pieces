"""Chapter 1 Challenge — Silver: opt into validation with a Pydantic state.

Bronze showed a `TypedDict` state does no runtime checking. Swap the schema for a
Pydantic `BaseModel` and the SAME bad inputs are now rejected at `invoke()` time,
*before any node runs*. Nothing about the wiring changes — only the state's type,
and how the node reads it.

Two things to notice:
  * A `BaseModel` state is read with ATTRIBUTES (`state.question`), not subscripts
    (`state["question"]`). That's the one change inside the node.
  * `graph.invoke(...)` still hands you back a plain dict, even though the node
    received a validated model.
  * Validation checks the state HANDED TO A NODE, not the update a node returns.
    `bad_write_node` returning `{"answer": 12345}` lands unchanged because this
    one-node graph has no later node to receive it, and the final output is not
    re-validated. Add a second node and that node's input IS rebuilt from the
    schema, so the bad `answer` is rejected there — the asymmetry the Silver
    teaches, and the reason Chapter 12 exists.

This is a first taste of Chapter 12, "State Schemas".

Run:  python code/answers/ch01_pydantic_state.py   (see answers/README.md)
"""

from pydantic import BaseModel, ValidationError

from langgraph.graph import StateGraph, START, END


class State(BaseModel):
    question: str
    answer: str | None = None      # optional output, like NotRequired before


def answer_node(state: State) -> dict:
    # Attribute access, because `state` is a validated model instance here.
    return {"answer": f"You asked: {state.question!r}"}


def bad_write_node(state: State) -> dict:
    # An int into a `str | None` field. Nothing checks a node's RETURN...
    return {"answer": 12345}


def read_after_node(state: State) -> dict:
    # ...but a plain second node forces the issue. To run this node, LangGraph
    # rebuilds a State from what `bad_write_node` wrote, and THAT is validated.
    return {"answer": f"seen: {state.answer!r}"}


def build_graph():
    b = StateGraph(State)
    b.add_node("answer", answer_node)
    b.add_edge(START, "answer")
    b.add_edge("answer", END)
    return b.compile()


graph = build_graph()


def build_bad_write_graph():
    b = StateGraph(State)
    b.add_node("answer", bad_write_node)
    b.add_edge(START, "answer")
    b.add_edge("answer", END)
    return b.compile()


bad_write_graph = build_bad_write_graph()


def build_bad_write_then_read_graph():
    b = StateGraph(State)
    b.add_node("bad", bad_write_node)
    b.add_node("after", read_after_node)
    b.add_edge(START, "bad")
    b.add_edge("bad", "after")
    b.add_edge("after", END)
    return b.compile()


bad_write_then_read_graph = build_bad_write_then_read_graph()


# --- tests -------------------------------------------------------------------
def test_good_input_runs():
    result = graph.invoke({"question": "hi"})
    # invoke() returns a dict even though the node saw a model.
    assert result["answer"] == "You asked: 'hi'"


def test_wrong_type_is_rejected_upfront():
    try:
        graph.invoke({"question": 123})  # int, not str
    except ValidationError as e:
        assert "question" in str(e)
    else:
        raise AssertionError("expected ValidationError for the wrong type")


def test_missing_key_is_rejected_upfront():
    try:
        graph.invoke({})
    except ValidationError:
        pass
    else:
        raise AssertionError("expected ValidationError for the missing 'question'")


def test_node_write_is_not_validated_in_a_one_node_graph():
    # Half the punchline: the same schema that rejected {"question": 123} at the
    # door accepts an int written INTO `answer` by the graph's last node, because
    # nothing downstream receives it and the output is not re-validated.
    result = bad_write_graph.invoke({"question": "hi"})
    assert result["answer"] == 12345
    assert isinstance(result["answer"], int)


def test_node_write_is_validated_by_the_next_node():
    # The other half: add a node that receives the int, and the same schema
    # rejects `answer` when it rebuilds that node's input.
    try:
        bad_write_then_read_graph.invoke({"question": "hi"})
    except ValidationError as e:
        assert "answer" in str(e)
    else:
        raise AssertionError("expected ValidationError for the int 'answer'")


if __name__ == "__main__":
    print("good input ->", graph.invoke({"question": "hi"}))
    print("last node writes an int ->", bad_write_graph.invoke({"question": "hi"}))
    try:
        bad_write_then_read_graph.invoke({"question": "hi"})
    except ValidationError as e:
        print("...but a second node rejects it ->", str(e).splitlines()[0])
    for bad in ({"question": 123}, {}):
        try:
            graph.invoke(bad)
        except ValidationError as e:
            first = str(e).splitlines()[0]
            print(f"{bad!r:20} -> ValidationError: {first}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
