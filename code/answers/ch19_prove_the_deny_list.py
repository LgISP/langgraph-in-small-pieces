"""Chapter 19 Challenge — Bronze: prove the deny-list, in both directions.

`RetryPolicy(retry_on=(ConnectionError,))` is an allow-list, so it is also a
deny-list, and getting it wrong costs you either way. This runs the node twice --
once raising a `ValueError` the policy does not cover, once with the policy
widened -- and counts the attempts.

Predict first:

  * **narrow policy, `ValueError` raised.** The node runs **once**, and `invoke`
    **raises**. Not "retries and then raises": the policy looks at the exception's
    type before it decides to sleep, and a type it was not given is not its
    business. The backoff never happens, and the error travels up unchanged.
  * **widened to `(ConnectionError, ValueError)`.** Now the same node runs three
    times and `invoke` **returns**, because the third attempt is the one the
    chapter's stub was written to succeed on.

The two failure modes named in the challenge, made concrete by the numbers this
file prints:

  * **too narrow** -- a genuinely transient fault crashes a run that one retry
    would have saved.
  * **too wide** -- a deterministic bug (a `KeyError` in your own dict, a
    `ValidationError` on a malformed payload) is retried three times, sleeping
    between attempts, arriving at exactly the same failure with the latency
    tripled and any side effects done three times over.

The second one is the one people get wrong: retrying is not free and not
harmless. `retry_on` is where you write down which faults you believe are about
the world rather than about your code.

Run:  python code/answers/ch19_prove_the_deny_list.py   (see answers/README.md)
"""

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

attempts = {"n": 0}


class S(TypedDict, total=False):
    result: str


def flaky(raises: type[Exception]):
    """The chapter's `flaky_api_node`, with the exception type as a parameter."""
    def node(state: S) -> dict:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise raises("first two attempts fail")
        return {"result": "call succeeded"}
    return node


def run(raises: type[Exception], retry_on: tuple) -> tuple[int, str]:
    """Returns (attempts made, "returned" | "raised: ...")."""
    attempts["n"] = 0
    b = StateGraph(S)
    b.add_node("api", flaky(raises),
               retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.05,
                                        backoff_factor=2.0, retry_on=retry_on))
    b.add_edge(START, "api")
    b.add_edge("api", END)
    graph = b.compile()
    try:
        graph.invoke({})
        return attempts["n"], "returned"
    except BaseException as exc:                      # noqa: BLE001
        return attempts["n"], f"raised {type(exc).__name__}"


CASES = [
    ("ValueError, narrow policy", ValueError, (ConnectionError,)),
    ("ValueError, widened policy", ValueError, (ConnectionError, ValueError)),
    ("ConnectionError, narrow policy", ConnectionError, (ConnectionError,)),
]

_CACHE: dict = {}


def cached() -> dict:
    if not _CACHE:
        for label, raises, retry_on in CASES:
            _CACHE[label] = run(raises, retry_on)
    return _CACHE


def demo_deny_list():
    c = cached()
    print(f"{'case':<34}{'attempts':>10}   outcome")
    print("-" * 66)
    for label, _, _ in CASES:
        n, outcome = c[label]
        print(f"{label:<34}{n:>10}   {outcome}")
    print("\nThe first two lines are the same node raising the same exception. The")
    print("only difference is one tuple, and it decides whether the run survives.")
    print("\nRead the first line again: ONE attempt. The policy did not retry and")
    print("give up -- it never retried. A type outside retry_on is not a fault the")
    print("policy has an opinion about, so there is no backoff to observe.")


# --- tests -------------------------------------------------------------------

def test_an_uncovered_exception_is_not_retried_at_all():
    n, outcome = cached()["ValueError, narrow policy"]
    assert n == 1, n
    assert outcome == "raised ValueError", outcome


def test_widening_the_policy_flips_the_behaviour():
    n, outcome = cached()["ValueError, widened policy"]
    assert n == 3, n
    assert outcome == "returned", outcome


def test_the_covered_exception_still_behaves_as_the_chapter_says():
    """The control: nothing about the widening changed the original case."""
    n, outcome = cached()["ConnectionError, narrow policy"]
    assert (n, outcome) == (3, "returned")


def test_the_same_node_and_the_same_exception_give_both_outcomes():
    """The claim in one assertion: the tuple is the whole difference."""
    narrow = cached()["ValueError, narrow policy"]
    wide = cached()["ValueError, widened policy"]
    assert narrow[1].startswith("raised") and wide[1] == "returned"
    assert narrow[0] < wide[0]


def test_retrying_costs_attempts_which_is_the_price_of_being_too_wide():
    """The second failure mode, priced: a deterministic bug under a wide policy
    does its side effects three times before failing identically."""
    attempts["n"] = 0

    def always_fails(state):
        attempts["n"] += 1
        raise ValueError("a bug, not a blip")

    b = StateGraph(S)
    b.add_node("api", always_fails,
               retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01,
                                        retry_on=(ValueError,)))
    b.add_edge(START, "api")
    b.add_edge("api", END)
    try:
        b.compile().invoke({})
    except BaseException:                              # noqa: BLE001
        pass
    assert attempts["n"] == 3, attempts["n"]


if __name__ == "__main__":
    demo_deny_list()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
