"""Chapter 19 Challenge - Gold: all three recovery shapes in one graph.

The chapter kept the patterns in separate demos for clarity. A real production
graph wears all three at once - one per node, each matched to the kind of failure
that node can have:

  fetch   RetryPolicy         - transient blips (a dropped connection); let
                                LangGraph re-run the node until it catches.
  web     graceful degradation - a non-critical enrichment tool times out; skip
                                it, mark the run degraded, and keep going.
  parse   catch + route        - a permanent, fatal error (malformed payload);
                                catch it, write it into State, and take the
                                fallback branch instead of crashing.

Then we run the SAME compiled graph twice:
  - an input that trips the transient retry AND the graceful degradation but NOT
    the fatal parse -> the run completes with a degraded-but-real answer;
  - one flipped flag so parse fails -> the same graph takes the fallback branch.

The order below - fetch -> web -> parse - is a teaching arrangement, not a
production topology: it runs the web enrichment BEFORE the fatal parse check, so a
doomed request still pays for the web call. A real graph validates first
(fetch -> parse -> web) and falls back before spending on enrichment; this order is
chosen only so one run exercises all three mechanisms in sequence.

Everything is deterministic: the failures are a counter and raised exceptions -
no LLM.

Run:  python code/answers/ch19_combined_recovery.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


class S(TypedDict):
    request_id: str                   # input
    parse_ok: bool                    # input: does the fetched payload parse? (drives the fatal path)
    data: NotRequired[str]            # written by fetch
    web_results: NotRequired[list]    # written by web
    degraded: NotRequired[bool]       # written by web on failure
    error: NotRequired[str]           # written by parse on the fatal path
    answer: NotRequired[str]          # written by generate or fallback


# fetch fails transiently twice per request id, then succeeds - a simulated blip.
# A request id in _outage_forever never succeeds: that is the case the retry
# policy cannot fix, and what happens after the third attempt is the point.
_fetch_attempts: dict[str, int] = {}
_outage_forever: set[str] = set()


# --- node 1: fetch - RetryPolicy handles a transient blip ---------------------
def fetch_node(state: S) -> dict:
    rid = state["request_id"]
    _fetch_attempts[rid] = _fetch_attempts.get(rid, 0) + 1
    n = _fetch_attempts[rid]
    if rid in _outage_forever:
        print(f"      fetch attempt {n} -> ConnectionError (outage, not a blip)")
        raise ConnectionError("upstream is down")
    if n < 3:
        print(f"      fetch attempt {n} -> ConnectionError (transient); RetryPolicy will re-run")
        raise ConnectionError("transient blip on fetch")
    print(f"      fetch attempt {n} -> ok")
    return {"data": "raw payload"}


# --- node 2: web - graceful degradation for a non-critical tool ---------------
def web_node(state: S) -> dict:
    try:
        raise TimeoutError("web enrichment timed out")   # non-critical: nice-to-have
    except Exception as e:
        print(f"      web enrichment failed, degrading gracefully: {e}")
        # Do NOT re-raise. Mark degraded and hand back empty results so the flow
        # continues on local data alone.
        return {"web_results": [], "degraded": True}


# --- node 3: parse - catch a permanent fault and route to a fallback ----------
def parse_node(state: S) -> dict:
    try:
        if not state["parse_ok"]:
            # A permanent fault retrying will never fix (malformed payload).
            raise ValueError("payload is permanently malformed")
    except Exception as e:
        print(f"      parse failed permanently and caught: {e}")
        return {"error": str(e)}
    # Success: compose the answer, noting the degradation if web was down.
    note = "(note: web enrichment unavailable) " if state.get("degraded") else ""
    return {"answer": f"{note}parsed answer from {state['data']}"}


def route_after_parse(state: S) -> str:
    return "fallback" if state.get("error") else "done"


def fallback_node(state: S) -> dict:
    return {"answer": f"Sorry, we could not process your request (reason: {state['error']}). Please try again later."}


def _build():
    b = StateGraph(S)
    b.add_node(
        "fetch",
        fetch_node,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.01,
            backoff_factor=2.0,
            retry_on=(ConnectionError,),   # retry ONLY the transient fault
        ),
    )
    b.add_node("web", web_node)
    b.add_node("parse", parse_node)
    b.add_node("fallback", fallback_node)
    b.add_edge(START, "fetch")
    b.add_edge("fetch", "web")
    b.add_edge("web", "parse")
    b.add_conditional_edges("parse", route_after_parse, {"done": END, "fallback": "fallback"})
    b.add_edge("fallback", END)
    return b.compile()


graph = _build()   # ONE compiled graph, used for both runs below


def run(request_id: str, parse_ok: bool) -> dict:
    _fetch_attempts.pop(request_id, None)   # fresh transient sequence for this run
    return graph.invoke({"request_id": request_id, "parse_ok": parse_ok})


def run_outage(request_id: str) -> Exception:
    """The case the retry policy cannot fix. RetryPolicy re-runs fetch three
    times, then the ConnectionError propagates out of invoke(): no fallback node
    ran, no answer was written. That is a TERMINAL failure, and it is the right
    outcome -- the fallback edge handles a fault a node *caught*, and an
    exhausted retry is one nobody caught. Whatever owns the call (the HTTP
    handler in @sec-serving, a job runner) is where an exhausted retry is turned
    into a status, not a node that swallows it and cancels the retry."""
    _fetch_attempts.pop(request_id, None)
    _outage_forever.add(request_id)
    try:
        graph.invoke({"request_id": request_id, "parse_ok": True})
    except ConnectionError as exc:
        return exc
    finally:
        _outage_forever.discard(request_id)
    raise AssertionError("the outage should have propagated")


# --- tests -------------------------------------------------------------------
def test_degraded_answer_when_parse_survives():
    # Transient retry + graceful degradation both fire; parse survives.
    out = run("req-ok", parse_ok=True)
    assert _fetch_attempts["req-ok"] == 3       # fetch was retried past the two blips
    assert out["degraded"] is True              # web died and we noted it
    assert "error" not in out                   # parse did NOT fail
    assert out["answer"].startswith("(note: web enrichment unavailable)")
    assert "parsed answer from raw payload" in out["answer"]


def test_fallback_branch_when_parse_fails():
    # Same graph, one flipped flag: the fatal parse routes to the fallback node.
    out = run("req-bad", parse_ok=False)
    assert _fetch_attempts["req-bad"] == 3      # fetch still retried away its blips
    assert out["degraded"] is True              # web still degraded gracefully
    assert out["error"] == "payload is permanently malformed"
    assert out["answer"].startswith("Sorry, we could not process your request")


def test_same_graph_takes_both_paths():
    # The two outcomes come from ONE compiled graph, differing only by input.
    ok = run("a", parse_ok=True)
    bad = run("b", parse_ok=False)
    assert "error" not in ok and "error" in bad          # one path per input
    assert ok["answer"] != bad["answer"]
    assert ok["degraded"] is True and bad["degraded"] is True   # web degrades on both


def test_exhausted_retry_is_a_terminal_failure_not_a_fallback():
    # Three attempts (max_attempts=3), then the error leaves the graph. Nothing
    # downstream ran: the fallback branch is for faults parse CATCHES, and an
    # exhausted retry is not one of them.
    exc = run_outage("req-outage")
    assert isinstance(exc, ConnectionError)
    assert _fetch_attempts["req-outage"] == 3
    assert "req-outage" not in _outage_forever          # cleaned up
    # and the graph is not wedged: the next request on a healthy fetch still works
    assert "parsed answer" in run("req-after-outage", parse_ok=True)["answer"]


def test_transient_retry_is_invisible_in_the_result():
    # RetryPolicy swallows the two ConnectionErrors: the final State has no error
    # from fetch - only the answer.
    out = run("req-quiet", parse_ok=True)
    assert out.get("error", "") == ""
    assert out["data"] == "raw payload"


if __name__ == "__main__":
    print("########## run 1: transient blip + web degradation, parse survives ##########")
    ok = run("demo-ok", parse_ok=True)
    print(f"  answer: {ok['answer']}")
    print(f"  degraded={ok['degraded']}  error={'error' in ok}")

    print("\n########## run 2: same graph, parse fails -> fallback branch ##########")
    bad = run("demo-bad", parse_ok=False)
    print(f"  answer: {bad['answer']}")
    print(f"  degraded={bad['degraded']}  error={bad.get('error')!r}")

    print("\n########## run 3: an outage the retry cannot fix -> terminal failure, no fallback ##########")
    exc = run_outage("demo-outage")
    print(f"  invoke() raised {type(exc).__name__}: {exc}  after {_fetch_attempts['demo-outage']} attempts; no answer written")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
