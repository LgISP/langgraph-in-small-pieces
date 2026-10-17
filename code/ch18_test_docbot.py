"""Chapter 18 - Tests for the doc-bot, at four altitudes.

Nothing here talks to a model server. Every test runs in milliseconds and gives
the same answer on every machine, which is what separates a test from the
evaluation harness of @sec-evaluation: a test asks "did my code do what I wired
it to do", and the answer is yes or no.

Run:  pytest code/ch18_test_docbot.py -v
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END
from langgraph.runtime import Runtime

from ch18_docbot import Context, State, build_docbot, model, route

# --------------------------------------------------------------------------- #
# The scripted model
# --------------------------------------------------------------------------- #


class ScriptedModel(FakeMessagesListChatModel):
    """A model that says exactly what the test tells it to.

    `FakeMessagesListChatModel` returns its `responses` in order, one per call.
    It does not implement `bind_tools`, and `model()` calls that -- so without
    this two-line override every test dies on `NotImplementedError`. Ignoring
    the tools is correct here: the script, not the schema, decides the reply.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def searches_then_answers() -> ScriptedModel:
    """Turn 1: call search_docs. Turn 2: answer from what came back."""
    return ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "search_docs", "args": {"query": "kafka batch"}, "id": "c1"},
        ]),
        AIMessage(content="Set spring.kafka.listener.type=batch."),
    ])


@pytest.fixture
def compiled():
    """A fresh graph AND a fresh checkpointer for every test that asks."""
    return build_docbot().compile(checkpointer=InMemorySaver())


# --------------------------------------------------------------------------- #
# Level 1 - one node, no graph
# --------------------------------------------------------------------------- #


def test_tools_node_returns_a_delta_not_a_state():
    """`graph.nodes[name]` runs one node with no graph around it."""
    graph = build_docbot().compile()
    ai = AIMessage(content="", tool_calls=[
        {"name": "search_docs", "args": {"query": "kafka"}, "id": "c1"},
    ])

    # A non-empty history, or the test cannot tell a delta from a merged state:
    # with tools_used=[] a node that wrongly returned state["tools_used"] + used
    # produces the identical result, and this test would pass on it.
    prior = {"messages": [ai], "tools_used": ["web_search"]}
    out = graph.nodes["tools"].invoke(prior)

    # What comes back is the node's RETURN VALUE -- the delta. Nothing merged
    # it into a state, because there is no graph here to do the merging.
    assert set(out) == {"messages", "tools_used"}
    assert out["tools_used"] == ["search_docs"]          # this turn only, not the history
    assert len(out["messages"]) == 1                     # likewise: one ToolMessage, not the transcript
    assert isinstance(out["messages"][0], ToolMessage)
    assert "spring.kafka.listener.type=batch" in out["messages"][0].content


def test_model_node_is_a_plain_function_and_takes_a_scripted_model():
    """`model` reads `runtime.context`, so it cannot go through
    `graph.nodes[...]` -- see test_nodes_dict_cannot_supply_context. Call the
    function. It is a function."""
    out = model(
        {"messages": [HumanMessage("How do I configure Kafka?")], "tools_used": []},
        Runtime(context=Context(llm=searches_then_answers())),
    )

    # ONE key: `model` never touches tools_used, and the delta proves it.
    assert set(out) == {"messages"}
    assert out["messages"][0].tool_calls[0]["name"] == "search_docs"


def test_nodes_dict_cannot_supply_context():
    """The limit of `graph.nodes[...]`, pinned so it cannot surprise anyone.

    There is no `context=` on this call, and no config key that stands in for
    one. The error names a config key, which is misleading -- the missing thing
    is the runtime context.
    """
    graph = build_docbot().compile()
    with pytest.raises(ValueError, match="Missing required config key"):
        graph.nodes["model"].invoke({"messages": [], "tools_used": []})


# --------------------------------------------------------------------------- #
# Level 2 - the router, as the function it is
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("last, expected", [
    (AIMessage(content="", tool_calls=[
        {"name": "search_docs", "args": {"query": "k"}, "id": "c1"}]), "tools"),
    (AIMessage(content="Here is the answer."), END),
])
def test_route_picks_the_edge(last, expected):
    """No graph, no model, no checkpointer: a conditional edge is a function
    from state to a node name, and four lines cover both of its branches."""
    assert route({"messages": [last], "tools_used": []}) == expected


# --------------------------------------------------------------------------- #
# Level 3 - a partial path: start in the middle
# --------------------------------------------------------------------------- #


def test_partial_path_from_a_planted_state(compiled):
    """Plant the state a node would have produced, and run from there.

    This is the one that changes how you test. Reaching "the model has just
    asked for search_docs" the honest way means running the model; here
    `update_state(as_node="model")` writes that state directly and the router
    then decides from it, exactly as it would mid-run.
    """
    config = {"configurable": {"thread_id": "partial"}}
    planted = AIMessage(content="", tool_calls=[
        {"name": "search_docs", "args": {"query": "redis pool"}, "id": "c9"},
    ])

    compiled.update_state(config, {"messages": [planted]}, as_node="model")

    # The router has already run on the planted state: `tools` is up next.
    assert compiled.get_state(config).next == ("tools",)

    # Resume with None -- "continue from the checkpoint", not "start over".
    out = compiled.invoke(None, config, context=Context(llm=ScriptedModel(
        responses=[AIMessage(content="Use lettuce.pool.max-active.")])))

    assert out["tools_used"] == ["search_docs"]
    assert "lettuce.pool.max-active" in out["messages"][-1].content


def test_interrupt_after_stops_the_run_for_inspection(compiled):
    """`interrupt_after` is a test instrument here, not a human gate: it stops
    the run at a node boundary so the test can look at what is on the thread."""
    config = {"configurable": {"thread_id": "halted"}}

    compiled.invoke(
        {"messages": [HumanMessage("How do I configure Kafka?")], "tools_used": []},
        config,
        context=Context(llm=searches_then_answers()),
        interrupt_after=["model"],
    )

    snapshot = compiled.get_state(config)
    assert snapshot.next == ("tools",)          # stopped before running them
    assert snapshot.values["tools_used"] == []  # so nothing ran yet


# --------------------------------------------------------------------------- #
# Level 4 - the whole graph
# --------------------------------------------------------------------------- #


def test_the_whole_loop_searches_then_answers(compiled):
    """One turn through the real topology, with the model scripted.

    The assertions are about STRUCTURE -- which tool ran, how many turns, what
    the last message is. Nothing asserts the model's wording, because the only
    reason the wording is knowable here is that a test wrote it.
    """
    out = compiled.invoke(
        {"messages": [HumanMessage("How do I configure Kafka?")], "tools_used": []},
        {"configurable": {"thread_id": "whole"}},
        context=Context(llm=searches_then_answers()),
    )

    assert out["tools_used"] == ["search_docs"]
    assert [m.type for m in out["messages"]] == ["human", "ai", "tool", "ai"]
    assert not out["messages"][-1].tool_calls   # the loop actually ended


def test_no_tool_call_means_no_tools_node(compiled):
    """The other branch of `route`, end to end: the model answers directly and
    the tools node never runs."""
    out = compiled.invoke(
        {"messages": [HumanMessage("Hello!")], "tools_used": []},
        {"configurable": {"thread_id": "direct"}},
        context=Context(llm=ScriptedModel(responses=[AIMessage(content="Hi.")])),
    )

    assert out["tools_used"] == []
    assert [m.type for m in out["messages"]] == ["human", "ai"]


def test_the_loop_cannot_run_away(compiled):
    """The only test here that proves the doc-bot cannot hang.

    Every other test runs a cooperative script, which proves the loop ends when
    the model lets it. This one scripts a model that never stops asking for
    tools -- the pathological case a real model reaches on a bad day -- and
    pins that @sec-looping's fuse is what stops it.
    """
    class NeverStops(ScriptedModel):
        def invoke(self, *args, **kwargs):
            return AIMessage(content="", tool_calls=[
                {"name": "search_docs", "args": {"query": "x"}, "id": "c"}])

    with pytest.raises(GraphRecursionError):
        compiled.invoke(
            {"messages": [HumanMessage("Hello!")], "tools_used": []},
            {"configurable": {"thread_id": "runaway"}, "recursion_limit": 6},
            context=Context(llm=NeverStops(responses=[])),
        )


def test_each_test_gets_its_own_checkpointer():
    """Why the fixture builds a new saver every time.

    Both halves use the SAME thread_id. With one shared `InMemorySaver` the
    second run would resume the first one's conversation; with a fresh saver
    each run starts empty, and the second assertion is what proves it.
    """
    config = {"configurable": {"thread_id": "same-id-both-times"}}

    for _ in range(2):
        graph = build_docbot().compile(checkpointer=InMemorySaver())
        out = graph.invoke(
            {"messages": [HumanMessage("Hello!")], "tools_used": []},
            config,
            context=Context(llm=ScriptedModel(responses=[AIMessage(content="Hi.")])),
        )
        assert len(out["messages"]) == 2   # never 4, never 6
