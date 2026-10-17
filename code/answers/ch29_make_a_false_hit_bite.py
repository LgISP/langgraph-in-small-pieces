"""Chapter 29 Challenge — Silver: make a false hit bite.

Drop `GOLDEN_THRESHOLD` to something reckless -- 0.40 -- and ask a question that
lands near a curated entry but asks something it does not answer: "How do I handle
Kafka consumer failures?". Its nearest golden question is the FRESH batch-consumption
entry (~0.67), so only the threshold -- not the freshness guard -- decides its fate.
Then restore 0.80 and ask it again.

What the two runs show:

  * **at 0.40** -- `source=golden`, and the curated answer that comes back is the
    batch-consumption entry (`set spring.kafka.listener.type=batch...`), which says
    nothing about handling failures. No `[retrieve_docs]` line, no model call,
    nothing in the run that could have noticed. The reply is fast, confident, and
    answering a different question.
  * **at 0.80** -- the same ~0.67 falls short, the run continues to retrieval and
    generation, and the answer is about the question.

The lesson the challenge asks you to write down: **on a path that bypasses the
model, the similarity threshold is the only thing standing between "fast and
right" and "fast and wrong."** Everywhere else in this book a bad retrieval is
survivable, because a model reads the documents and can decline to use them. Here
the retrieval IS the answer. There is no reader.

And the Deep Dive's claim, put under load: is anything left in the loop to notice?
No. Walk the graph: `check_golden` compares two numbers, and on a hit it returns
the stored answer and routes to END. Nothing downstream sees the question and the
answer together. The number is not the first line of defence, it is the only one
-- which is why 0.80 is not tuning, it is the safety margin, and why the honest
move when you cannot pick a number you trust is to not ship the bypass.

**Order independence.** Ch29's module wipes both stores at import, so a fresh
import gives exactly the chapter's two golden entries. If the Bronze answer file
ran first *in the same process*, its mysql entry is still there and would change
which entry the false hit lands on -- so this file removes it and says so, rather
than quietly producing different numbers depending on collection order.

Needs LM Studio at :1234 for embeddings.
Run:  python code/answers/ch29_make_a_false_hit_bite.py   (see answers/README.md)
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LM_BASE = "http://localhost:1234/v1"
# A question that lands nearest the FRESH kafka golden ("How do I configure Kafka
# batch consumption?") but asks something its answer does not cover -- failure
# handling, not batch config. Nearest is fresh, so nothing but the threshold stands
# between the question and a wrong curated answer (a stale nearest would be refused
# by the freshness guard, which is a different lesson).
QUESTION = "How do I handle Kafka consumer failures?"
RECKLESS = 0.40


class Skip(Exception):
    pass


def server_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{LM_BASE}/models", timeout=2):
            return True
    except urllib.error.HTTPError:
        return True        # a status code is a server: a hosted endpoint answers 401 to a key-less probe
    except (urllib.error.URLError, OSError):
        return False


def _ch28():
    if not server_is_up():
        raise Skip(f"no model server at {LM_BASE} (embeddings are needed)")
    import ch29_golden as ch28

    # See "Order independence" above.
    extra = ch28.golden_store.get(where={"topic": "mysql"})
    if extra["ids"]:
        ch28.golden_store.delete(ids=extra["ids"])
    return ch28


def ask_at(ch28, threshold: float) -> dict:
    saved, ch28.GOLDEN_THRESHOLD = ch28.GOLDEN_THRESHOLD, threshold
    try:
        return ch28.graph.invoke({"question": QUESTION})
    finally:
        ch28.GOLDEN_THRESHOLD = saved


_CACHE: dict = {}


def cached() -> dict:
    if not _CACHE:
        ch28 = _ch28()
        doc, dist = ch28.golden_store.similarity_search_with_score(QUESTION, k=1)[0]
        _CACHE.update(
            ch28=ch28,
            sim=1 - dist,
            nearest_topic=doc.metadata["topic"],
            nearest_question=doc.page_content,
            reckless=ask_at(ch28, RECKLESS),
            careful=ask_at(ch28, 0.80),
        )
    return _CACHE


def demo_false_hit():
    if not server_is_up():
        print(f"no model server at {LM_BASE} -- embeddings are needed; skipping")
        return
    c = cached()
    print(f"question       : {QUESTION!r}")
    print(f"nearest golden : {c['nearest_question']!r}  (topic={c['nearest_topic']})")
    print(f"similarity     : {c['sim']:.3f}\n")

    print(f"--- threshold {RECKLESS} " + "-" * 40)
    print(f"source : {c['reckless']['source']}")
    print(f"answer : {c['reckless']['answer'][:78]} ...")
    print(f"   ^ the batch-consumption answer, served to a question about failures "
          f"-- a different ask\n")

    print("--- threshold 0.80 " + "-" * 40)
    print(f"source : {c['careful']['source']}")
    print(f"answer : {c['careful']['answer'][:78]} ...\n")

    print("Nothing in the 0.40 run was in a position to object. check_golden")
    print("compares two numbers and routes to END on a hit; no later node ever")
    print("sees the question and the answer in the same place. The threshold is")
    print("not the first line of defence on this path -- it is the only one.")


# --- tests -------------------------------------------------------------------

def test_the_question_genuinely_misses_at_the_real_threshold():
    """The premise: ~0.67, over the reckless 0.40 but under 0.80. If this drifts, the
    challenge is demonstrating something else and the prose needs rewriting."""
    c = cached()
    assert 0.40 < c["sim"] < 0.80, c["sim"]


def test_a_reckless_threshold_serves_it_from_the_golden_layer():
    assert cached()["reckless"]["source"] == "golden"
    assert cached()["reckless"]["golden_route"] == "hit"


def test_the_false_hit_answers_a_different_question():
    """The bite: the served answer is the batch-consumption golden entry, which
    says nothing about handling failures -- a curated answer to a question it was
    never written for."""
    c = cached()
    assert c["reckless"]["answer"].startswith("[Golden")
    assert "listener.type=batch" in c["reckless"]["answer"]      # the batch-config answer...
    assert "fail" in QUESTION.lower()                            # ...served to a failure question
    assert "DefaultErrorHandler" not in c["reckless"]["answer"]  # which it does not address


def test_the_real_threshold_falls_through_to_generation():
    assert cached()["careful"]["source"] == "generated"
    assert cached()["careful"]["golden_route"] == "miss"


def test_the_two_runs_disagree_about_the_same_question():
    """Same question, same store, same embeddings -- one number apart."""
    c = cached()
    assert c["reckless"]["answer"] != c["careful"]["answer"]


def test_nothing_downstream_of_a_golden_hit_could_have_noticed():
    """The Deep Dive's claim, read off the graph rather than asserted: on a hit,
    `route_golden` sends the run to END, so no node that sees the answer also
    sees the question."""
    ch28 = cached()["ch28"]
    import inspect
    src = inspect.getsource(ch28)
    edges = src[src.index("builder.add_conditional_edges"):]
    edges = edges[:edges.index("builder.add_edge(\"retrieve_docs\"")]
    assert "END" in edges and "hit" in edges, edges
    assert cached()["reckless"]["source"] == "golden"


if __name__ == "__main__":
    demo_false_hit()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        try:
            t()
            print("PASS", t.__name__)
        except Skip as e:
            print("SKIP", t.__name__, f"({e})")
    print(f"\nall {len(_tests)} tests passed")
