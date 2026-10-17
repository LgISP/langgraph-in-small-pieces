"""Chapter 31 - Refusing the call, instead of noticing it afterwards.

ch31_trajectory.py shows that a right answer can come from a path you did not
allow. This file is the other half: what it takes to actually STOP that call, and
why the obvious fix is not enough.

Three runs of Ch18's doc-bot, same scripted model asking for web_search every time:

    1) shipped      - the graph as Ch18 wrote it. The call runs.
    2) unadvertised - web_search removed from the list bound to the model, i.e. the
                      model is no longer TOLD about it. The call STILL RUNS.
    3) authorized   - the executor checks an allow-list before dispatching. The call
                      is refused, and the refusal is recorded.

Run 2 is the point. `bind_tools(TOOLS)` is a DESCRIPTION handed to the model; the
executor is a separate dict lookup (`BY_NAME[call["name"]]`) with no permission
check in it. Hiding a tool from the description only makes the model less likely to
ask -- and "less likely to ask" is not a control, because the model is exactly the
component you cannot make promises on behalf of. Anything that decides what a run may
do has to sit where the model cannot reach it, and has to run BEFORE the side effect.

What run 3 produces is also the shape of an audit answer: not "we handle that", but a
request that appears in the trajectory, an execution count of zero, and a dated line
saying who refused what.

Offline: the model is scripted, so no server and no network.

Run:  python code/ch31_refuse.py
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import ch18_docbot as db
from ch18_docbot import Context, build_docbot
from ch18_test_docbot import ScriptedModel
from ch31_trajectory import requested

QUESTION = "How do I configure Kafka batch consumers?"
ANSWER = "Set spring.kafka.listener.type=batch."

# What THIS run is allowed to execute. Not what the model was told about -- the two
# are different things, and only this one is a control.
ALLOWED = {"search_docs"}


def call(name: str, query: str, cid: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": {"query": query}, "id": cid}])


# The model asks for the forbidden tool first, whatever it was told about.
SCRIPT = [call("web_search", "kafka batch consumers", "c1"),
          call("search_docs", "kafka batch", "c2"),
          AIMessage(content=ANSWER)]

REFUSALS: list[str] = []      # the audit trail run 3 leaves behind


def authorizing_tools(state):
    """Ch18's `tools` node with one thing added: authorize BEFORE dispatch.

    The refusal goes back to the model as a ToolMessage, so the run continues and
    can still answer from what it is allowed to use -- a denied call is not a
    crashed graph.
    """
    results, used = [], []
    for c in state["messages"][-1].tool_calls:
        if c["name"] not in ALLOWED:
            REFUSALS.append(f"refused {c['name']}: not in this run's allow-list")
            results.append(ToolMessage(
                content=f"[denied] {c['name']} is not available to this run",
                tool_call_id=c["id"]))
            continue                      # <- the side effect never happens
        output = db.BY_NAME[c["name"]].invoke(c["args"])
        results.append(ToolMessage(content=output, tool_call_id=c["id"]))
        used.append(c["name"])
    return {"messages": results, "tools_used": used}


def run(*, advertise_web: bool, authorize: bool) -> dict:
    """One run. `advertise_web` controls what the MODEL is told; `authorize`
    controls what the EXECUTOR permits."""
    original_tools, original_node = db.TOOLS, db.tools
    REFUSALS.clear()
    try:
        db.TOOLS = list(original_tools) if advertise_web else [
            t for t in original_tools if t.name in ALLOWED]
        # build_docbot() wires whatever `tools` names at build time, so swapping it
        # here gives the SAME graph with one node replaced -- no second graph to
        # keep in sync.
        if authorize:
            db.tools = authorizing_tools
        out = build_docbot().compile().invoke(
            {"messages": [HumanMessage(QUESTION)], "tools_used": []},
            context=Context(llm=ScriptedModel(responses=list(SCRIPT))))
    finally:
        db.TOOLS, db.tools = original_tools, original_node
    return {
        "advertised": sorted(t.name for t in (
            original_tools if advertise_web else
            [t for t in original_tools if t.name in ALLOWED])),
        "requested": requested(out),
        "executed": out["tools_used"],
        "refusals": list(REFUSALS),
        "answer": out["messages"][-1].content,
    }


CASES = {
    "1 shipped":      dict(advertise_web=True,  authorize=False),
    "2 unadvertised": dict(advertise_web=False, authorize=False),
    "3 authorized":   dict(advertise_web=False, authorize=True),
}

_CACHE: dict = {}


def cased(label: str) -> dict:
    if label not in _CACHE:
        _CACHE[label] = run(**CASES[label])
    return _CACHE[label]


# --- tests -------------------------------------------------------------------
# The model is scripted, so these are exact on every machine.

def test_the_shipped_graph_executes_the_forbidden_call():
    assert "web_search" in cased("1 shipped")["executed"]


def test_hiding_the_tool_from_the_model_does_not_stop_execution():
    """The counterexample this file exists for. web_search is no longer bound, so
    the model was never told about it -- and the executor runs it anyway, because
    `BY_NAME` is a lookup, not a permission check."""
    c = cased("2 unadvertised")
    assert "web_search" not in c["advertised"]      # not described to the model
    assert "web_search" in c["executed"]            # ...executed all the same


def test_authorizing_the_executor_refuses_the_call():
    c = cased("3 authorized")
    assert "web_search" in c["requested"]           # the model still asked
    assert c["executed"] == ["search_docs"]         # the side effect did not happen
    assert c["refusals"] == [
        "refused web_search: not in this run's allow-list"]


def test_a_refused_run_still_answers():
    """A denied call is not a crashed graph: the run continues on what it may use."""
    assert cased("3 authorized")["answer"] == ANSWER


def test_the_request_trajectory_cannot_tell_the_three_apart():
    """Every run REQUESTED the same two tools. Only the execution record differs --
    which is why a trajectory read after the fact is a detector, not a control."""
    assert len({tuple(cased(l)["requested"]) for l in CASES}) == 1
    assert len({tuple(cased(l)["executed"]) for l in CASES}) == 2


if __name__ == "__main__":
    print(f"question: {QUESTION}\n")
    print(f"{'run':<16}{'advertised':<28}{'requested':<28}{'executed'}")
    for label in CASES:
        c = cased(label)
        print(f"{label:<16}{str(c['advertised']):<28}"
              f"{str(c['requested']):<28}{c['executed']}")

    print("\nrefusals recorded:", cased("3 authorized")["refusals"] or "none")
    print(f"answer after the refusal: {cased('3 authorized')['answer']!r}")

    print("\nRun 2 is the one to sit with. web_search was NOT bound, so the model was")
    print("never told it exists -- and it ran. `bind_tools` writes a description;")
    print("the executor is `BY_NAME[name]` with nothing in between. Only run 3 has a")
    print("control, and it is a control because it decides BEFORE the call and the")
    print("model cannot reach it.")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
