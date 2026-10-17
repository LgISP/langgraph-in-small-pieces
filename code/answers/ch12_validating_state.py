"""Chapter 12 Challenge — Bronze: make the door reject bad input.

Take a real slice of the doc-bot's state — question / route / iterations — and
write it two ways: a plain `@dataclass` and a Pydantic `BaseModel`. The BaseModel
puts a floor on iterations with `Field(ge=0)`, so a negative count is illegal.

Feed each the SAME broken inputs and watch WHERE the failure lands:

  * BaseModel — invoke() raises ValidationError at the door, before any node
    runs, naming the offending field. A wrong type (question=123) and an
    out-of-range value (iterations=-5) are both caught up front.
  * dataclass — invoke() constructs the object without a murmur. iterations=-5
    sails straight through (a negative int is a perfectly good int) and is never
    caught at all; question=123 is only caught when the node calls a *string*
    method on it and blows up with an AttributeError deep inside the node.

That gap — front door vs. deep in a node, named field vs. symptom — is the whole
reason BaseModel exists.

Run:  python code/answers/ch12_validating_state.py   (see answers/README.md)
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from langgraph.graph import StateGraph, START, END


# --- the same doc-bot state slice, written two ways --------------------------
@dataclass
class DataclassState:
    question: str
    route: str
    iterations: int
    summary: str = ""          # output field; default keeps callers from setting it


class ValidatedState(BaseModel):
    question: str
    route: str
    iterations: int = Field(ge=0)   # a floor: a negative pass count is illegal
    summary: str = ""


# One node, reused by both graphs. Both states are objects, so it reads with
# ATTRIBUTES, and it calls a *string* method on `question` — the operation a
# wrong-typed question cannot survive.
def summarize(state) -> dict:
    return {"summary": f"{state.question.strip()} via {state.route} (pass {state.iterations})"}


def build(state_type):
    b = StateGraph(state_type)
    b.add_node("summarize", summarize)
    b.add_edge(START, "summarize")
    b.add_edge("summarize", END)
    return b.compile()


dataclass_graph = build(DataclassState)
validated_graph = build(ValidatedState)


GOOD = {"question": "  how do I configure Kafka?  ", "route": "retrieve", "iterations": 1}
BAD_TYPE = {"question": 123, "route": "retrieve", "iterations": 1}      # question wrong type
BAD_RANGE = {"question": "hi", "route": "retrieve", "iterations": -5}   # illegal count


# --- tests -------------------------------------------------------------------
def test_basemodel_good_input_runs():
    result = validated_graph.invoke(GOOD)
    assert result["summary"] == "how do I configure Kafka? via retrieve (pass 1)"


def test_basemodel_rejects_wrong_type_at_door():
    try:
        validated_graph.invoke(BAD_TYPE)      # question=123
    except ValidationError as e:
        assert "question" in str(e)           # the door names the field
    else:
        raise AssertionError("expected ValidationError for the wrong type")


def test_basemodel_rejects_out_of_range_at_door():
    try:
        validated_graph.invoke(BAD_RANGE)     # iterations=-5, below the Field(ge=0) floor
    except ValidationError as e:
        assert "iterations" in str(e)         # named, before any node runs
    else:
        raise AssertionError("expected ValidationError for the illegal count")


def test_dataclass_accepts_negative_silently():
    # No validation: the illegal count is never caught. It flows straight through
    # the node and into the result unchallenged.
    result = dataclass_graph.invoke(BAD_RANGE)
    assert result["iterations"] == -5
    assert "pass -5" in result["summary"]


def test_dataclass_wrong_type_detonates_inside_node():
    # The dataclass waves question=123 through the door; the failure surfaces only
    # when the node calls .strip() on an int — an AttributeError, not a
    # ValidationError, and pointing at a symptom rather than the cause.
    try:
        dataclass_graph.invoke(BAD_TYPE)
    except ValidationError:
        raise AssertionError("dataclass should NOT validate at the door")
    except AttributeError as e:
        assert "strip" in str(e)              # symptom, deep in the node
    else:
        raise AssertionError("expected AttributeError from inside the node")


if __name__ == "__main__":
    print("BaseModel, good input ->", validated_graph.invoke(GOOD)["summary"])

    print("\n-- BaseModel: the bad value bounces off the front door --")
    for bad in (BAD_TYPE, BAD_RANGE):
        try:
            validated_graph.invoke(bad)
        except ValidationError as e:
            loc = ".".join(str(p) for p in e.errors()[0]["loc"])
            print(f"  {bad}  ->  ValidationError at invoke(), field {loc!r}")

    print("\n-- dataclass: no door; the bad value walks in --")
    walked_in = dataclass_graph.invoke(BAD_RANGE)   # never caught at all
    print(f"  {BAD_RANGE}  ->  accepted; summary = {walked_in['summary']!r}")
    try:
        dataclass_graph.invoke(BAD_TYPE)            # detonates inside the node
    except AttributeError as e:
        print(f"  {BAD_TYPE}  ->  {type(e).__name__} inside the node: {e}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
