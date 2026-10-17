"""Chapter 27 - Try it: a scorer that fails is not a scorer that said 0.

The rerank stage calls the model once per candidate. Two very different things
can come back looking the same from outside the node: "I scored them all and
nothing is relevant" (a legitimate empty shortlist, and the agent's web fallback
firing is the design working) and "I never got a score" (a timeout on the second
candidate, a reply outside the 0-10 scale). `score_candidates` keeps them apart:
an execution failure is retried once and then reported as None, and rerank_node
degrades on purpose -- the top of the recall order, flagged -- instead of
shipping a zero it never measured.

Three cases, offline, against the real `score_candidates` / `rerank_node` with a
fake scorer injected:
  1. the scorer times out on the second candidate  -> degraded, recall order kept
  2. the scorer answers 11                          -> a schema failure, same path
  3. every candidate legitimately scores 0          -> empty shortlist, not degraded

Run:  python code/ch27_scorer_failure.py
"""

# ch27_rerank builds its Chroma store at import (it embeds the corpus through LM
# Studio). These cases never retrieve, so stand a no-op store in first; the rest of
# the module -- the schema, score_candidates, rerank_node -- is the real thing.
import langchain_chroma


class _NoStore:
    def __init__(self, *a, **k): pass
    def add_texts(self, *a, **k): pass


langchain_chroma.Chroma = _NoStore   # must precede the import below

import ch27_rerank as R                      # noqa: E402

CANDIDATES = ["Kafka batch consumption: set spring.kafka.listener.type=batch ...",
              "Kafka tuning: batch consumers need max-poll-records tuned ...",
              "Redis connection pool: configure spring.data.redis.* ..."]


def flaky_scorer(calls: list):
    """Scores 9 for the first candidate, then times out every time."""
    def score(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return 9
        raise TimeoutError("scorer timed out")
    return score


def out_of_range_scorer(prompt):
    return R.Relevance(score=11).score        # pydantic rejects it: ge=0, le=10


def all_zero_scorer(prompt):
    return 0


# --- tests -------------------------------------------------------------------
def test_timeout_mid_shortlist_degrades_to_the_recall_order():
    calls: list = []
    scored = R.score_candidates("q", CANDIDATES, score_fn=flaky_scorer(calls))
    assert scored is None                                  # reported as a failure ...
    assert len(calls) == 3                                 # ... after one retry (1 ok + 2 attempts)
    out = _rerank_with(flaky_scorer([]))
    assert out["rerank_degraded"] is True
    assert out["docs"] == CANDIDATES[:2]                   # recall order, unscored, still non-empty


def test_a_score_outside_the_scale_is_a_failure_not_a_low_score():
    try:
        R.Relevance(score=11)
        raise AssertionError("11 should not validate")
    except ValueError:
        pass
    out = _rerank_with(out_of_range_scorer)
    assert out["rerank_degraded"] is True
    assert out["docs"] == CANDIDATES[:2]


def test_a_legitimate_all_zero_shortlist_is_empty_and_not_degraded():
    out = _rerank_with(all_zero_scorer)
    assert out["rerank_degraded"] is False
    assert out["docs"] == []                               # the agent's web fallback fires from here


def _rerank_with(score_fn):
    """rerank_node with the real scorer swapped for a fake, restored afterwards."""
    original = R.score_candidates
    R.score_candidates = lambda q, c, score_fn=None, attempts=2: original(q, c, score_fn=score_fn_holder[0], attempts=attempts)
    score_fn_holder[0] = score_fn
    try:
        return R.rerank_node({"question": "q", "docs": list(CANDIDATES)})
    finally:
        R.score_candidates = original


score_fn_holder = [None]


if __name__ == "__main__":
    print("Case 1: the scorer times out on the second candidate")
    print(f"   -> {_rerank_with(flaky_scorer([]))}")
    print("Case 2: the scorer answers 11")
    print(f"   -> {_rerank_with(out_of_range_scorer)}")
    print("Case 3: every candidate scores 0")
    print(f"   -> {_rerank_with(all_zero_scorer)}")
    print()
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
