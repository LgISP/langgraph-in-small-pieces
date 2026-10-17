"""Chapter 22 Challenge — Silver: break the tracer two ways, and price each break.

The chapter's `wrapper` records in a `finally` and re-raises. Both halves are
load-bearing; this file removes them one at a time and shows what each removal
costs, because a guarantee you have never seen fail is a guarantee you are taking
on trust.

**Break 1 — record after the call instead of in `finally`.** The step that raised
is the one step missing from the trace. Everything before it is there, the run
died at a node the waterfall does not mention, and the gap is exactly where you
would look first. Losing visibility at the moment of failure is not a small hole
in an observability tool; it is the hole.

**Break 2 — keep `finally`, swap `raise` for `return {}`.** This is the dangerous
one, and it is the one that ships, because it looks like defensive programming:
the node "handled" its error. What actually happens is that the tracer produces a
complete, tidy, entirely misleading waterfall of a run that carried on over a state
no node ever wrote. The graph never hears about the failure, `generate` reads a
`docs` key that was never set, and the only signal left is the answer being wrong
later, somewhere else, for reasons the trace actively denies.

The two failures are worth naming apart: break 1 loses information, break 2
manufactures it. A tool that says nothing is a tool you stop trusting; a tool that
says the wrong thing with confidence is one you keep trusting.

And the Gotcha's point, which neither break touches: the correct tracer watched the
crash in full detail and that still does not let you resume from it. A trace is a
story about a run; a checkpoint is the run's state. Reading the story back does not
put you inside it -- that is @sec-save-resume's checkpointer, and
@sec-inside-persistence shows the `__error__` row the crashed task leaves in the
writes table while its output never reaches the checkpoint.

Run:  python code/answers/ch22_trace_the_failure.py   (see answers/README.md)
"""

import functools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ch22_observability as ch21    # noqa: E402


def traced_correct(fn):
    """The chapter's tracer, for reference: record in `finally`, re-raise."""
    @functools.wraps(fn)
    def wrapper(state):
        start = time.perf_counter()
        result = error = None
        try:
            result = fn(state)
            return result
        except BaseException as exc:                  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _record(fn, start, result, error)
    return wrapper


def traced_break1_no_finally(fn):
    """BREAK 1: append after the call. A raising node never reaches the append."""
    @functools.wraps(fn)
    def wrapper(state):
        start = time.perf_counter()
        result = fn(state)
        _record(fn, start, result, None)
        return result
    return wrapper


def traced_break2_swallows(fn):
    """BREAK 2: record everything, tell the graph nothing."""
    @functools.wraps(fn)
    def wrapper(state):
        start = time.perf_counter()
        result = error = None
        try:
            result = fn(state)
            return result
        except BaseException as exc:                  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            return {}                                 # <- the graph never hears
        finally:
            _record(fn, start, result, error)
    return wrapper


def _record(fn, start, result, error):
    ch21.TRACE.append({
        "node": fn.__name__,
        "start_ms": (start - ch21._T0) * 1000,
        "dur_ms": (time.perf_counter() - start) * 1000,
        "output_keys": list(result.keys()) if result else [],
        "error": error,
    })


def print_waterfall(trace: list[dict]) -> None:
    """The chapter's waterfall, with a lane for the step that failed."""
    print("\n" + "=" * 70)
    print("TRACE waterfall (failures included)")
    print("=" * 70)
    total = max(t["start_ms"] + t["dur_ms"] for t in trace)
    scale = 40 / total
    for i, t in enumerate(trace):
        bar = " " * int(t["start_ms"] * scale) + "#" * max(1, int(t["dur_ms"] * scale))
        tail = (f"raised {t['error']}" if t.get("error")
                else f"wrote {t['output_keys']}")
        flag = "!!" if t.get("error") else "  "
        print(f"{i:>2}{flag}{t['node']:<16} {bar}  {t['dur_ms']:>6.1f}ms  {tail}")
    print("-" * 70)
    failed = [t for t in trace if t.get("error")]
    print(f"Total: {total:.1f}ms   steps: {len(trace)}   failed: {len(failed)}")


NODES = {"agent": "agent_node", "search_docs": "search_docs_node",
         "web_search": "web_search_node", "generate": "generate_node"}
BOOM = "generate"


def build_graph(wrapper) -> object:
    """The chapter's graph, wrapped with the tracer variant under test, with
    `generate` armed to raise.

    `__wrapped__` is what `functools.wraps` leaves behind: it hands back the
    undecorated node so we put OUR wrapper on it rather than stacking one on top
    of the chapter's.
    """
    from langgraph.graph import END, START, StateGraph

    def crash(state):
        raise RuntimeError("model provider returned 503")

    builder = StateGraph(ch21.State)
    for node_name, fn_name in NODES.items():
        fn = getattr(ch21, fn_name).__wrapped__
        if node_name == BOOM:
            fn = functools.wraps(fn)(crash)
        builder.add_node(node_name, wrapper(fn))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", ch21.route_after_agent,
        {"search": "search_docs", "web": "web_search", "generate": "generate"},
    )
    builder.add_edge("search_docs", "agent")
    builder.add_edge("web_search", "agent")
    builder.add_edge("generate", END)
    return builder.compile()


def run_with(wrapper) -> tuple[list[dict], BaseException | None, dict | None]:
    """Returns (trace, exception that escaped or None, final state or None)."""
    ch21.TRACE.clear()
    caught = out = None
    try:
        out = build_graph(wrapper).invoke({"question": "How do I configure Kafka?"})
    except BaseException as exc:                       # noqa: BLE001
        caught = exc
    return list(ch21.TRACE), caught, out


_CACHE: dict = {}


def cached() -> dict:
    if not _CACHE:
        for name, w in (("correct", traced_correct),
                        ("break1", traced_break1_no_finally),
                        ("break2", traced_break2_swallows)):
            _CACHE[name] = run_with(w)
    return _CACHE


def demo_trace_the_failure():
    c = cached()

    trace, caught, _ = c["correct"]
    print("=== the chapter's tracer " + "=" * 46)
    print_waterfall(trace)
    print(f"escaped the graph: {type(caught).__name__}: {caught}")

    trace, caught, _ = c["break1"]
    print("\n=== break 1: append after the call, not in finally " + "=" * 20)
    print_waterfall(trace)
    print(f"escaped the graph: {type(caught).__name__}: {caught}")
    print(f"steps recorded: {len(trace)} (the correct tracer recorded "
          f"{len(c['correct'][0])})")
    print("The run died at a node the waterfall does not mention. It lost information.")

    trace, caught, out = c["break2"]
    print("\n=== break 2: finally kept, raise swapped for return {} " + "=" * 16)
    print_waterfall(trace)
    print(f"escaped the graph: {caught}")
    print(f"the run 'succeeded' and returned keys: {sorted(out) if out else out}")
    print(f"answer present? {bool(out and out.get('answer'))}")
    print("A complete, tidy, entirely misleading trace. It manufactured information.")

    print("\nBreak 1 loses information; break 2 makes it up. A tool that says nothing")
    print("is one you stop trusting. A tool that says the wrong thing with confidence")
    print("is one you keep trusting.")


# --- tests -------------------------------------------------------------------

def test_the_correct_tracer_records_the_failing_step():
    trace, _, _ = cached()["correct"]
    assert [t for t in trace if t.get("error")], "the crash left no trace at all"


def test_the_correct_tracer_names_what_was_raised():
    trace, _, _ = cached()["correct"]
    bad = [t for t in trace if t.get("error")][0]
    assert bad["node"] == NODES[BOOM]
    assert "RuntimeError" in bad["error"] and "503" in bad["error"]


def test_the_correct_tracer_does_not_swallow_the_failure():
    _, caught, _ = cached()["correct"]
    assert isinstance(caught, RuntimeError) and "503" in str(caught)


def test_break1_loses_exactly_the_step_that_failed():
    """The whole cost of appending outside `finally`."""
    good, _, _ = cached()["correct"]
    trace, caught, _ = cached()["break1"]
    assert isinstance(caught, RuntimeError)            # the graph still dies
    assert len(trace) == len(good) - 1                 # ...one step quieter
    assert not [t for t in trace if t.get("error")]


def test_break1_still_records_everything_before_the_failure():
    """It is not that the tracer stopped working -- it worked right up to the
    one step you needed, which is why the gap is easy to misread as 'the run
    never got there'."""
    trace, _, _ = cached()["break1"]
    assert [t["node"] for t in trace] and NODES[BOOM] not in [t["node"] for t in trace]


def test_break2_hides_the_failure_from_the_graph():
    _, caught, out = cached()["break2"]
    assert caught is None, "the exception escaped; break 2 did not swallow it"
    assert out is not None


def test_break2_produces_a_run_that_looks_finished_and_is_not():
    """The dangerous half: the graph reports success and the answer never exists."""
    _, _, out = cached()["break2"]
    assert not out.get("answer")


def test_break2_records_the_error_it_hid():
    """And the trace says so -- which is the trap. The record is accurate and the
    conclusion a reader draws from a 'complete' waterfall is not."""
    trace, _, _ = cached()["break2"]
    assert [t for t in trace if t.get("error")]
    assert len(trace) == len(cached()["correct"][0])


def test_the_two_breaks_fail_in_opposite_directions():
    """Break 1 loses information; break 2 manufactures it."""
    b1, caught1, _ = cached()["break1"]
    b2, caught2, out2 = cached()["break2"]
    assert len(b1) < len(b2)                  # one is quieter than the truth
    assert caught1 is not None                # ...and honest about the run dying
    assert caught2 is None and not out2.get("answer")   # the other is neither


if __name__ == "__main__":
    demo_trace_the_failure()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
