"""Chapter 18 Challenge — Gold: one test that needs a real model, and is marked.

A scripted model proves the wiring. It cannot prove the thing a scripted model
stands in for: that `search_docs`'s NAME and DESCRIPTION are good enough that a
real model reaches for it. That claim needs the real model, so it needs a test
that is slow, needs a server, and -- the uncomfortable part -- can fail on a day
nothing in your repository changed.

So it is marked, and the default run skips it:

    pytest code/answers/ch18_real_model_test.py            # fast suite only
    pytest code/answers/ch18_real_model_test.py -m slow    # the real-model one

The marker is declared in the repo's `pytest.ini`, so `--strict-markers` stays
usable and a typo'd marker is an error rather than a silently-skipped test:

    [pytest]
    markers =
        slow: needs the local model server (LM Studio at :1234)

**What it asserts, and what it must not.** Only this:

    "search_docs" in out["tools_used"]

Not the answer's wording, not the number of turns, not which snippet came back.
Every one of those moves when the model's temperature, version or mood moves, and
a test that fails for those reasons teaches you to ignore it. The tool CHOICE is
the weakest claim that still says something about the tool descriptions -- and it
is the strongest one that belongs in a test at all.

**And then the honest part.** Even this can go red on an unchanged repository. That
does not make it a flaky test to be fixed with a retry; it makes it a MEASUREMENT
wearing a test's clothes. Measurements belong on a dashboard with a trend line and
a threshold you chose on purpose -- @sec-evaluation -- not on the gate that blocks
a merge. Keeping it here, marked and out of the default run, is the compromise:
you can ask it a question when you change a tool description, without it ever
blocking someone else's unrelated commit.

Run:  python code/answers/ch18_real_model_test.py   (see answers/README.md)
"""

import inspect
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ch18_docbot import Context, build_docbot    # noqa: E402

LM_BASE = "http://localhost:1234/v1"
QUESTION = "How do I configure Kafka batch consumers?"   # the local docs plainly cover this


def server_is_up(base: str = LM_BASE) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=2):
            return True
    except urllib.error.HTTPError:
        return True        # a status code is a server: a hosted endpoint answers 401 to a key-less probe
    except (urllib.error.URLError, OSError):
        return False


def real_model():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(base_url=LM_BASE, api_key="lm-studio",
                      model="openai/gpt-oss-20b", temperature=0)


def ask_with_a_real_model(question: str = QUESTION) -> dict:
    graph = build_docbot().compile()
    return graph.invoke({"messages": [("user", question)], "tools_used": []},
                        context=Context(llm=real_model()))


# --------------------------------------------------------------------------- #
# The fast tests: they run everywhere, and they are ABOUT the slow one
# --------------------------------------------------------------------------- #

def test_the_slow_test_carries_the_marker():
    """The point of the challenge is the marking, so the marking is tested."""
    marks = {m.name for m in test_a_real_model_reaches_for_search_docs.pytestmark}
    assert "slow" in marks


def test_the_marker_is_declared_in_pytest_ini():
    """An undeclared marker is a warning today and an error under
    --strict-markers; either way the test silently stops selecting."""
    ini = Path(__file__).resolve().parents[2] / "pytest.ini"
    assert ini.exists(), f"missing {ini}"
    assert "slow:" in ini.read_text()


def test_the_assertion_is_about_the_path_not_the_prose():
    """A guard against the version of this test everyone writes first: asserting
    the answer's wording. Read the slow test's own source and confirm every
    assertion in it is about the tool CHOICE -- the one thing that does not move
    when the model's temperature, version or mood moves.

    (Written twice: the first attempt sliced the file by searching for the slow
    test's name, and the first match was the search string itself, one line up.
    `inspect.getsource` knows where a function starts without being told.)"""
    body = inspect.getsource(test_a_real_model_reaches_for_search_docs)
    asserts = [l.strip() for l in body.splitlines() if l.strip().startswith("assert ")]
    assert asserts == ['assert "search_docs" in out["tools_used"]'], asserts


# --------------------------------------------------------------------------- #
# The slow one
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_a_real_model_reaches_for_search_docs():
    """The only claim a scripted model cannot make for us."""
    if not server_is_up():
        pytest.skip(f"no model server at {LM_BASE}")
    out = ask_with_a_real_model()
    assert "search_docs" in out["tools_used"]


def demo_real_model_test():
    print(f"model server at {LM_BASE}: {'up' if server_is_up() else 'down'}")
    if not server_is_up():
        print("skipping the real-model run -- start LM Studio to see it")
        return
    out = ask_with_a_real_model()
    print(f"question  : {QUESTION}")
    print(f"tools_used: {out['tools_used']}")
    print(f"answer    : {out['messages'][-1].content[:70]} ...")
    print("\nThe assertion is the first line only. The answer is printed so you can")
    print("see what we deliberately did NOT assert.")


if __name__ == "__main__":
    demo_real_model_test()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        try:
            t()
            print("PASS", t.__name__)
        except BaseException as e:                      # pytest.skip raises here
            if type(e).__name__ == "Skipped":
                print("SKIP", t.__name__, f"({e})")
            else:
                raise
    print(f"\nall {len(_tests)} tests passed")
