"""Chapter 18 Challenge — Silver: test the doc-bot NOT working.

The chapter's suite tests the happy path at four altitudes. Two failure paths are
untested, and they fail in very different ways.

**1. A tool that does not exist.** `tools()` does `BY_NAME[call["name"]]`, and the
model can emit any name it likes. Today that is a bare `KeyError` out of a
dictionary lookup -- which is not a tool failure the agent can reason about, it is
the node dying. @sec-error-handling's `RetryPolicy` would not save it either:
retrying a deterministic `KeyError` just raises it again, because a fault that cannot
pass is not transient.

    The decision this file pins: an unknown tool comes back as a ToolMessage
    saying so. The model asked for something that is not there; that is
    information, and the loop's own back-edge is what gets it to the model. The
    run stays inside the graph instead of ending in a traceback.

**2. A search that finds nothing.** `search_docs` already returns "No matching
local documents." -- a real sentence, not an empty string. Untested, though, and
"" would be an easy accident: `"\\n".join([])` is `""`, so the fallback is one
`if` away from vanishing.

Both tests are level-1: they call the node directly, with no model server and no
graph, which is why they run in milliseconds and say exactly which node was wrong.

Run:  python code/answers/ch18_failure_paths.py   (see answers/README.md)
"""

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ch18_docbot import (BY_NAME, Context, State, build_docbot,   # noqa: E402
                         search_docs)
from ch18_test_docbot import ScriptedModel                        # noqa: E402


# --------------------------------------------------------------------------- #
# The fix: dispatch that answers instead of dying
# --------------------------------------------------------------------------- #

def tools_tolerant(state: State) -> dict:
    """`tools`, with the decision made: an unknown name is a ToolMessage.

    Note what does NOT change -- the return is still a delta, still carries both
    channels, and `tools_used` still records the attempt. A tool the model
    invented was still a tool call the model made, and a trajectory that hides it
    is a trajectory that lies (@sec-trajectory).
    """
    results, used = [], []
    for call in state["messages"][-1].tool_calls:
        tool = BY_NAME.get(call["name"])
        if tool is None:
            output = (f"No tool named {call['name']!r}. "
                      f"Available tools: {', '.join(sorted(BY_NAME))}.")
        else:
            output = tool.invoke(call["args"])
        results.append(ToolMessage(content=output, tool_call_id=call["id"]))
        used.append(call["name"])
    return {"messages": results, "tools_used": used}


def _state_asking_for(name: str, args: dict) -> State:
    return {
        "messages": [HumanMessage(content="q"),
                     AIMessage(content="", tool_calls=[
                         {"name": name, "args": args, "id": "c1"}])],
        "tools_used": [],
    }


def demo_failure_paths():
    print("1. the model asks for a tool that does not exist")
    from ch18_docbot import tools as tools_today
    try:
        tools_today(_state_asking_for("delete_everything", {}))
        print("   today: returned normally (?)")
    except KeyError as e:
        print(f"   today   : KeyError({e}) -- the node dies, mid-run")
    out = tools_tolerant(_state_asking_for("delete_everything", {}))
    print(f"   decided : {out['messages'][0].content}")
    print(f"   recorded: tools_used={out['tools_used']}")

    print("\n2. search_docs finds nothing")
    print(f"   returns : {search_docs.invoke({'query': 'quantum tunnelling'})!r}")
    out = tools_tolerant(_state_asking_for("search_docs", {"query": "quantum tunnelling"}))
    print(f"   as a ToolMessage: {out['messages'][0].content!r}")

    print("\n3. the whole loop survives an invented tool")
    script = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "delete_everything", "args": {}, "id": "c1"}]),
        AIMessage(content="I cannot do that, but I can search the docs."),
    ])
    # In your own repo you would edit `tools()` in ch18_docbot.py. This file must
    # leave the chapter's code alone, so it swaps the node function in the builder
    # it got back -- same graph, same wiring, one node's body replaced.
    builder = build_docbot()
    builder.nodes["tools"].runnable.func = tools_tolerant
    graph = builder.compile()
    result = graph.invoke({"messages": [("user", "delete everything")], "tools_used": []},
                          context=Context(llm=script))
    print(f"   tools_used: {result['tools_used']}")
    print(f"   final     : {result['messages'][-1].content}")


def _tolerant_graph():
    """The chapter's builder with the decision swapped in (see demo above)."""
    builder = build_docbot()
    builder.nodes["tools"].runnable.func = tools_tolerant
    return builder.compile()


# --- tests -------------------------------------------------------------------

def test_todays_dispatch_dies_on_an_unknown_tool():
    """Pinning the CURRENT behaviour first: you cannot decide to change what you
    have not written down."""
    from ch18_docbot import tools as tools_today
    try:
        tools_today(_state_asking_for("delete_everything", {}))
    except KeyError as e:
        assert "delete_everything" in str(e)
    else:
        raise AssertionError("expected a KeyError from the dictionary lookup")


def test_an_unknown_tool_comes_back_as_a_tool_message():
    out = tools_tolerant(_state_asking_for("delete_everything", {}))
    assert len(out["messages"]) == 1
    assert isinstance(out["messages"][0], ToolMessage)
    assert "delete_everything" in out["messages"][0].content


def test_the_unknown_tool_message_names_what_does_exist():
    """A refusal the model cannot act on just burns another turn."""
    content = tools_tolerant(_state_asking_for("delete_everything", {}))["messages"][0].content
    assert "search_docs" in content and "web_search" in content


def test_an_invented_tool_is_still_recorded_in_the_trajectory():
    assert tools_tolerant(_state_asking_for("delete_everything", {}))["tools_used"] \
        == ["delete_everything"]


def test_the_tool_call_id_is_echoed_back():
    """A ToolMessage without its call id is an orphan; the next model call is
    what would break, one node later and much harder to read."""
    out = tools_tolerant(_state_asking_for("delete_everything", {}))
    assert out["messages"][0].tool_call_id == "c1"


def test_a_miss_returns_the_sentence_not_an_empty_string():
    got = search_docs.invoke({"query": "quantum tunnelling"})
    assert got == "No matching local documents."
    assert got.strip() != ""


def test_a_miss_reaches_the_model_as_that_sentence():
    out = tools_tolerant(_state_asking_for("search_docs", {"query": "quantum tunnelling"}))
    assert out["messages"][0].content == "No matching local documents."


def test_a_hit_still_works_after_the_change():
    """The failure path must not cost the happy path."""
    out = tools_tolerant(_state_asking_for("search_docs", {"query": "kafka batch"}))
    assert "spring.kafka.listener.type=batch" in out["messages"][0].content


def test_the_run_terminates_after_an_unknown_tool():
    """Level 4: the graph gets the bad news to the model and finishes."""
    script = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "delete_everything", "args": {}, "id": "c1"}]),
        AIMessage(content="I cannot do that."),
    ])
    out = _tolerant_graph().invoke(
        {"messages": [("user", "delete everything")], "tools_used": []},
        context=Context(llm=script))
    assert out["tools_used"] == ["delete_everything"]
    assert out["messages"][-1].content == "I cannot do that."


def test_the_run_terminates_after_an_empty_search():
    script = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "search_docs", "args": {"query": "quantum"}, "id": "c1"}]),
        AIMessage(content="The local docs do not cover that."),
    ])
    out = _tolerant_graph().invoke(
        {"messages": [("user", "quantum?")], "tools_used": []},
        context=Context(llm=script))
    assert out["tools_used"] == ["search_docs"]
    assert "No matching local documents." in [
        m.content for m in out["messages"] if isinstance(m, ToolMessage)]


if __name__ == "__main__":
    demo_failure_paths()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
