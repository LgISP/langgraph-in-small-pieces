"""Chapter 19 Challenge - Silver: make the retried node idempotent.

The retry Gotcha warned that a node body runs once per attempt, side effects and
all. This file makes it bite, then fixes it.

  naive_charge_node   appends to a module-level `charges` list *before* the point
                      where it might raise. Under a RetryPolicy that re-runs the
                      node three times, a single "successful" call charges the
                      customer THREE times - the side effect replayed once per
                      attempt.

  idempotent_charge_node   guards the same append with an idempotency key: it
                      only charges if the request id is not already in a `seen`
                      set. No matter how many times the node retries, the effect
                      happens exactly once.

This is the human-in-the-loop lesson (a replayed node must be safe to replay)
meeting the retry loop. Everything is deterministic: the "network blip" is a
counter that fails twice and succeeds on the third attempt - no LLM.

The in-process `_seen` set here is a teaching stand-in, NOT a production idempotency
store. It dies with the process, is not shared across replicas, and its
`add`-then-charge is not atomic - a crash between the two leaves "recorded as paid,
never charged." A real system sends an idempotency key to a payment/API that
dedupes server-side, or records the effect in the SAME durable transaction that
performs it. The point to take from this file is the SHAPE (guard the effect with a
key so a replay is a no-op), not this particular store.

Run:  python code/answers/ch19_idempotent_retry.py   (see answers/README.md)
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


class S(TypedDict):
    request_id: str
    result: str


# --- the broken version: the side effect replays once per attempt -------------
naive_charges: list[str] = []      # the fake side effect (imagine a payment API)
_naive_attempt = {"n": 0}          # simulate "fails twice, succeeds on the third try"


def naive_charge_node(state: S) -> dict:
    _naive_attempt["n"] += 1
    # Side effect FIRST, exactly as the Gotcha warns - so it runs on every attempt.
    naive_charges.append(f"charge for {state['request_id']}")
    if _naive_attempt["n"] < 3:
        raise ConnectionError("network blip")   # transient: RetryPolicy re-runs the node
    return {"result": "charged"}


def _build(node) -> "StateGraph":
    b = StateGraph(S)
    b.add_node(
        "charge",
        node,
        # Retry the transient blip up to 3 times. The node body - side effect and
        # all - re-runs on each attempt; that is exactly what we are studying.
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.01,
            backoff_factor=2.0,
            retry_on=(ConnectionError,),
        ),
    )
    b.add_edge(START, "charge")
    b.add_edge("charge", END)
    return b.compile()


naive_graph = _build(naive_charge_node)


def run_naive() -> dict:
    naive_charges.clear()
    _naive_attempt["n"] = 0
    result = naive_graph.invoke({"request_id": "req-1"})
    return {"charges": len(naive_charges), "attempts": _naive_attempt["n"], "result": result["result"]}


# --- the fix: an idempotency key makes the effect happen exactly once ----------
idem_charges: list[str] = []
_seen: set[str] = set()            # idempotency keys we have already applied
_idem_attempt = {"n": 0}


def idempotent_charge_node(state: S) -> dict:
    _idem_attempt["n"] += 1
    rid = state["request_id"]
    # Guard the side effect with the idempotency key: charge only if this request
    # id has never been charged. A replayed attempt finds rid already in `_seen`
    # and skips the effect - safe to replay.
    if rid not in _seen:
        _seen.add(rid)
        idem_charges.append(f"charge for {rid}")
    if _idem_attempt["n"] < 3:
        raise ConnectionError("network blip")   # same transient fault, same 3 attempts
    return {"result": "charged"}


idem_graph = _build(idempotent_charge_node)


def run_idempotent() -> dict:
    idem_charges.clear()
    _seen.clear()
    _idem_attempt["n"] = 0
    result = idem_graph.invoke({"request_id": "req-1"})
    return {"charges": len(idem_charges), "attempts": _idem_attempt["n"], "result": result["result"]}


# --- tests -------------------------------------------------------------------
def test_naive_node_double_charges_under_retry():
    r = run_naive()
    assert r["result"] == "charged"        # the call did ultimately succeed
    assert r["attempts"] == 3              # RetryPolicy re-ran the node three times
    assert r["charges"] == 3               # ...and the side effect fired every time - the bug


def test_idempotent_node_charges_exactly_once():
    r = run_idempotent()
    assert r["result"] == "charged"        # same success
    assert r["attempts"] == 3              # same three attempts (same transient fault)
    assert r["charges"] == 1               # but the guarded effect fired ONCE - the fix


def test_only_difference_is_the_guard_not_the_retry_count():
    # Both nodes retry identically; the effect count is the only thing that moves.
    naive = run_naive()
    idem = run_idempotent()
    assert naive["attempts"] == idem["attempts"] == 3
    assert naive["charges"] == 3 and idem["charges"] == 1


def test_replayed_request_id_is_never_charged_twice():
    # The idempotency key is per request id: replays of the SAME id are no-ops.
    run_idempotent()                       # charges req-1 once, leaves it in _seen
    before = len(idem_charges)
    _idem_attempt["n"] = 2                 # next invoke will succeed on its first body run
    idem_graph.invoke({"request_id": "req-1"})   # same id again
    assert len(idem_charges) == before     # no second charge for req-1


if __name__ == "__main__":
    naive = run_naive()
    print(f"naive charge node:      result={naive['result']!r}  attempts={naive['attempts']}  charges={naive['charges']}  <- billed 3x for one call")
    idem = run_idempotent()
    print(f"idempotent charge node: result={idem['result']!r}  attempts={idem['attempts']}  charges={idem['charges']}  <- billed once, the fix")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
