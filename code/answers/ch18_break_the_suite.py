"""Chapter 18 Challenge — Bronze: make a green suite go red.

A test you have never seen fail is a test you have not written yet. So this file
breaks the doc-bot three ways, one at a time, and runs the chapter's REAL suite
(`ch18_test_docbot.py`) against each broken build — no re-implementation, no
stand-in. What comes back is the thing the challenge asks for: which test caught
which break, and how quickly the failure names an address.

The three breaks, from the challenge:

  1. delete the `tools -> model` back-edge
  2. `route` returns "tools" whenever `messages` is non-empty
  3. the `tools` node returns `{"tools_used": used}` and drops the messages

Each is applied by patching `ch18_docbot` in place, because that is what the
graph is built from: `build_docbot()` looks its node functions up in the module's
globals at build time, so a patched `route` is the `route` the next compiled copy
gets. Two bindings need patching twice. The test module did
`from ch18_docbot import build_docbot, route` at ITS import time, so it holds its
own references: patch only the `ch18_docbot` side and `test_route_picks_the_edge`
keeps testing the original function, and -- the one that actually bit while this
file was written -- the `compiled` fixture keeps building the UNBROKEN graph, so
break 1 comes back "caught by nothing" and looks like a hole in the suite.

Run:  python code/answers/ch18_break_the_suite.py   (see answers/README.md)
"""

import contextlib
import io
import sys
from pathlib import Path

import pytest
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ch18_docbot as bot                      # noqa: E402
import ch18_test_docbot as suite               # noqa: E402

TEST_FILE = str(Path(bot.__file__).with_name("ch18_test_docbot.py"))


# --------------------------------------------------------------------------- #
# Running the real suite, and remembering the verdict per test
# --------------------------------------------------------------------------- #

class _Verdicts:
    """A three-line pytest plugin: keep the outcome of each test's call phase."""

    def __init__(self):
        self.outcome: dict[str, str] = {}

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.outcome[report.nodeid.split("::")[-1]] = report.outcome


def run_suite() -> dict[str, str]:
    """Run ch18_test_docbot.py in-process and return {test name: outcome}."""
    v = _Verdicts()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        pytest.main(["-q", "-p", "no:cacheprovider", "--no-header", TEST_FILE], plugins=[v])
    return v.outcome


def reds(outcome: dict[str, str]) -> list[str]:
    return sorted(n for n, o in outcome.items() if o != "passed")


# --------------------------------------------------------------------------- #
# The three breaks. Each returns an undo, so the next break starts from green.
# --------------------------------------------------------------------------- #

def break_1_no_back_edge():
    """Delete `tools -> model`. The loop becomes a one-way street."""
    original = bot.build_docbot

    def crippled() -> StateGraph:
        builder = StateGraph(bot.State, context_schema=bot.Context)
        builder.add_node("model", bot.model)
        builder.add_node("tools", bot.tools)
        builder.add_edge(START, "model")
        builder.add_conditional_edges("model", bot.route, {"tools": "tools", END: END})
        # builder.add_edge("tools", "model")   <-- the break
        return builder

    bot.build_docbot = crippled
    suite.build_docbot = crippled       # the fixture imported it by name too
    return lambda: (setattr(bot, "build_docbot", original),
                    setattr(suite, "build_docbot", original))


def break_2_route_always_tools():
    """`route` stops reading tool_calls and just checks that messages exist."""
    original = bot.route

    def always_tools(state) -> str:
        return "tools" if state["messages"] else END

    bot.route = always_tools
    suite.route = always_tools          # the test module imported it by name
    return lambda: (setattr(bot, "route", original),
                    setattr(suite, "route", original))


def break_3_tools_drops_messages():
    """The tools node keeps its bookkeeping and forgets its actual output."""
    original = bot.tools

    def forgetful(state) -> dict:
        used = [c["name"] for c in state["messages"][-1].tool_calls]
        for call in state["messages"][-1].tool_calls:
            bot.BY_NAME[call["name"]].invoke(call["args"])   # side effect still happens
        return {"tools_used": used}                          # ...but no ToolMessage

    bot.tools = forgetful
    return lambda: setattr(bot, "tools", original)


BREAKS = [
    ("1. delete the tools -> model back-edge", break_1_no_back_edge),
    ("2. route returns 'tools' whenever messages is non-empty", break_2_route_always_tools),
    ("3. tools returns only tools_used, dropping the messages", break_3_tools_drops_messages),
]


def demo_break_the_suite():
    green = run_suite()
    print(f"baseline: {len(green)} tests, {len(reds(green))} red\n")

    for label, make_break in BREAKS:
        undo = make_break()
        try:
            failed = reds(run_suite())
        finally:
            undo()
        print(f"{label}\n   {len(failed)} red:")
        for name in failed:
            print(f"     - {name}")
        print()

    got = _collect()
    worst = max(((label, len(got[label])) for label, _ in BREAKS), key=lambda x: x[1])
    print(f"Widest blast radius: break {worst[0][0]} -- {worst[1]} tests at once.")
    print("Breadth is not an address, though. Read the red list of break 3: the")
    print("level-1 test names the node (tools) while the whole-graph tests only")
    print("say the run came out wrong. That is what the altitudes buy you.")


# --- tests -------------------------------------------------------------------
# These assert the SHAPE of the answer -- that each break is caught, and that the
# altitudes behave as the chapter claims -- not the exact test names, which would
# make this file brittle against the suite it is reading.

_CACHE: dict = {}


def _collect() -> dict[str, list[str]]:
    if "collected" not in _CACHE:
        out = {"baseline": reds(run_suite())}
        for label, make_break in BREAKS:
            undo = make_break()
            try:
                out[label] = reds(run_suite())
            finally:
                undo()
        _CACHE["collected"] = out
    return _CACHE["collected"]


def test_the_suite_is_green_before_anything_is_broken():
    assert _collect()["baseline"] == []


def test_every_break_turns_at_least_one_test_red():
    got = _collect()
    for label, _ in BREAKS:
        assert got[label], f"{label} was caught by nothing -- a gap in the suite"


def test_one_break_reddens_four_or_more_tests_at_once():
    """The chapter promises one break with a wide blast radius. It is the router:
    a router that always says "tools" breaks every altitude that runs the graph,
    and the route unit test on top."""
    got = _collect()
    counts = {label: len(got[label]) for label, _ in BREAKS}
    assert max(counts.values()) >= 4, counts


def test_a_level_1_test_names_the_broken_node():
    """Break 3 is a bug inside one node, and a level-1 test says so by name."""
    got = _collect()
    third = [label for label, _ in BREAKS if label.startswith("3.")][0]
    assert any("tools_node" in n for n in got[third]), got[third]


def test_the_breaks_are_undone():
    """Every break restores what it patched -- otherwise the file's own tests
    would be running against a broken doc-bot, which is the joke that writes
    itself."""
    _collect()
    assert reds(run_suite()) == []


if __name__ == "__main__":
    demo_break_the_suite()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
