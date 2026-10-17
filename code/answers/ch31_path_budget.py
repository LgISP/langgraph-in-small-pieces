"""Chapter 31 Challenge — Bronze: the budget fails while the answer stays right.

Ch31 showed two runs with identical answers and different paths. This adds a third:
a run that never goes off-policy -- every call is `search_docs`, the tool it IS
allowed to use -- but dithers, calling it three times before answering. A model that
wasted money rather than one that went somewhere forbidden.

Three transcripts, three checks, and the point is that they disagree:

    run            answer   allow-list            budget (<= 2)
    A straight     right    ok                    ok
    B detour       right    FAIL  (web_search)    ok
    C dithering    right    ok                    FAIL  (3 calls)

The answer check passes all three and is therefore useless for telling them apart.
The allow-list catches B and is blind to C. The budget catches C and is blind to B.
None of the three is a superset of another, which is the whole reason to run more
than one: they are answering three different questions about the same transcript.

Gate or report? The two failures are not the same kind of thing:

  * B is a POLICY violation. Rarity is no defence -- a tool the bot may never call is
    one it may never call once. But the durable fix is not this assertion, and it is
    not un-advertising the tool either: ch31_refuse.py shows the executor running
    web_search even when the model was never told it exists. The fix is an
    authorization check in front of the dispatch. A detector that reads a finished
    transcript cannot un-make the call it just read.
  * C depends on which claim you are making, and "budget" covers two different ones.
    A hard per-run ceiling ("never more than N calls") is a constraint: exceeding it
    is a violation, and the runtime should stop before the next call rather than let
    a check notice afterwards. A typical-cost expectation ("this usually takes two
    calls") is statistical: one dithering run means nothing, a shifted distribution
    means something, and gating it over a live model fails builds at random.

So: over a SCRIPTED model both are exact functions of the script and both can gate a
merge (that is what this file's tests do). Over a LIVE model, scripted-vs-live changes
how much your evidence covers -- it does not decide whether a rule is enforceable.
Enforce the allow-list and any hard ceiling in the runtime; report the distribution.

Offline: the model is Ch18's ScriptedModel, so this runs with no server and no network.

Run:  python code/answers/ch31_path_budget.py   (see answers/README.md)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

from ch18_docbot import Context, build_docbot
from ch18_test_docbot import ScriptedModel
from ch31_trajectory import ANSWER, QUESTION, DETOUR, STRAIGHT, requested

ALLOWED = {"search_docs"}   # a docs bot answers from the docs
BUDGET = 2                  # at most two tool calls to answer one question


def call(name: str, query: str, cid: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": {"query": query}, "id": cid}])


# The new script: allowed tool every time, same final answer, three times the calls.
DITHERING = [
    call("search_docs", "kafka batch", "d1"),
    call("search_docs", "kafka batch consumption", "d2"),
    call("search_docs", "spring kafka listener type", "d3"),
    AIMessage(content=ANSWER),
]

SCRIPTS = {"A straight": STRAIGHT, "B detour": DETOUR, "C dithering": DITHERING}


def run(script) -> dict:
    graph = build_docbot().compile()
    return graph.invoke(
        {"messages": [HumanMessage(QUESTION)], "tools_used": []},
        context=Context(llm=ScriptedModel(responses=list(script))),
    )


def score(script) -> dict:
    """The three checks over one run, side by side."""
    out = run(script)
    path = requested(out)
    answer = out["messages"][-1].content
    return {
        "answer": answer,
        "answer_ok": answer == ANSWER,          # what Ch30's harness would score
        "path": path,
        "off_policy": sorted(set(path) - ALLOWED),
        "allow_ok": set(path) <= ALLOWED,
        "calls": len(path),
        "budget_ok": len(path) <= BUDGET,
    }


_CACHE: dict = {}


def scored(label: str) -> dict:
    if label not in _CACHE:
        _CACHE[label] = score(SCRIPTS[label])
    return _CACHE[label]


# --- tests -------------------------------------------------------------------
# The model is scripted, so every assertion here is exact on every machine: over a
# scripted model these ARE tests, not evals (@sec-test-not-eval).

def test_every_run_produces_the_same_answer():
    """The premise: answer-scoring cannot separate any of the three."""
    for label in SCRIPTS:
        assert scored(label)["answer_ok"], label
    assert len({scored(l)["answer"] for l in SCRIPTS}) == 1


def test_the_allow_list_catches_only_the_detour():
    assert scored("B detour")["allow_ok"] is False
    assert scored("B detour")["off_policy"] == ["web_search"]
    assert scored("A straight")["allow_ok"] is True
    assert scored("C dithering")["allow_ok"] is True      # blind to dithering


def test_the_budget_catches_only_the_dithering():
    assert scored("C dithering")["budget_ok"] is False
    assert scored("C dithering")["calls"] == 3
    assert scored("A straight")["budget_ok"] is True
    assert scored("B detour")["budget_ok"] is True        # blind to the detour


def test_neither_check_subsumes_the_other():
    """Why you run both: each one's failure is the other's blind spot."""
    b, c = scored("B detour"), scored("C dithering")
    assert (b["allow_ok"], b["budget_ok"]) == (False, True)
    assert (c["allow_ok"], c["budget_ok"]) == (True, False)


def test_the_dithering_run_never_leaves_the_allowed_tool():
    """It is expensive, not off-policy -- the distinction the two checks encode."""
    assert set(scored("C dithering")["path"]) == {"search_docs"}


if __name__ == "__main__":
    print(f"question: {QUESTION}\n")
    print(f"{'run':<14}{'answer':<9}{'allow-list':<22}{'budget (<= %d)' % BUDGET}")
    for label in SCRIPTS:
        s = scored(label)
        allow = "ok" if s["allow_ok"] else f"FAIL  ({', '.join(s['off_policy'])})"
        budget = "ok" if s["budget_ok"] else f"FAIL  ({s['calls']} calls)"
        print(f"{label:<14}{'right' if s['answer_ok'] else 'wrong':<9}{allow:<22}{budget}")

    print("\nThree checks, three different questions about the same transcript:")
    print("  the answer check passes all three and separates nothing;")
    print("  the allow-list sees B and is blind to C;")
    print("  the budget sees C and is blind to B.")
    print("\nB is a policy violation -- enforce it where the tool is granted, not in a")
    print("detector that reads the call after it happened. C is a cost signal -- report")
    print("it and watch the trend, or a live model's ordinary variance will fail builds")
    print("at random until you stop believing the gate.")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
