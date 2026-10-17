"""Chapter 29 Challenge — Bronze: a curated answer serving a phrasing it never saw.

The doc store knows about MySQL indexing; the golden store does not, so a MySQL
question falls through to generation. Add a third golden entry -- question "How do
I add a database index in MySQL?", a curated answer, `topic="mysql"`,
`doc_version` matching what the document store holds for mysql -- then ask
a **paraphrase the store has never seen**: "How do I create an index on a MySQL
table?"

Predict first: does it clear the 0.80 bar?

It does, comfortably. Nothing about the two sentences matches as text -- "add"
against "create", "in MySQL" against "on a MySQL table" -- and that is the whole
demonstration: the entry is stored by embedding **the question**, so the lookup is
in meaning-space and a rephrasing lands next to the original. A golden layer keyed
by exact string would serve this question exactly never; keyed by embedding it
serves it on the first try, with `retrieve_docs` never printing and no model call
anywhere in the run.

Which is also the bill. Every question that hits here costs one embedding
(~20ms, @sec-observability measured it) instead of a retrieval plus a generation
(~20 seconds on the same machine). That is the ~1000x the chapter is really
selling, and the reason the 0.80 bar has to be as high as it is: the saving is
paid for by skipping the only component that could have noticed a wrong answer.

Needs LM Studio at :1234 (embeddings only -- no chat model is called on the golden
path, which is the point).
Run:  python code/answers/ch29_add_a_golden_entry.py   (see answers/README.md)
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LM_BASE = "http://localhost:1234/v1"

NEW_QUESTION = "How do I add a database index in MySQL?"
NEW_ANSWER = ("[Golden] Add a B+ tree index on the columns your slow queries filter "
              "or join on: CREATE INDEX idx_name ON table_name (col1, col2). Check it "
              "is used with EXPLAIN before and after.")
PARAPHRASE = "How do I create an index on a MySQL table?"


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
    return ch28


def seed_the_third_entry(ch28) -> None:
    """Add the mysql entry once. `add_texts` on a persistent collection appends,
    so a guard keeps a re-run from stacking duplicates -- worth noticing, because
    a golden store that silently doubles is a golden store whose similarity
    numbers stop meaning anything."""
    existing = ch28.golden_store.get(where={"topic": "mysql"})
    if existing["ids"]:
        return
    ch28.golden_store.add_texts(
        texts=[NEW_QUESTION],
        metadatas=[{"answer": NEW_ANSWER, "topic": "mysql",
                    "doc_version": ch28.current_doc_version(ch28.doc_store, "mysql")}],
    )


_CACHE: dict = {}


def cached() -> dict:
    if not _CACHE:
        ch28 = _ch28()
        seed_the_third_entry(ch28)
        hits = ch28.golden_store.similarity_search_with_score(PARAPHRASE, k=1)
        doc, dist = hits[0]
        _CACHE.update(
            ch28=ch28,
            sim=1 - dist,
            matched=doc.page_content,
            stale=ch28.is_stale(doc.metadata, ch28.doc_store),
            result=ch28.graph.invoke({"question": PARAPHRASE}),
            threshold=ch28.GOLDEN_THRESHOLD,
        )
    return _CACHE


def demo_add_a_golden_entry():
    if not server_is_up():
        print(f"no model server at {LM_BASE} -- embeddings are needed; skipping")
        return
    c = cached()
    print(f"stored question : {NEW_QUESTION!r}")
    print(f"asked instead   : {PARAPHRASE!r}")
    print("shared words    : 'a', 'MySQL', 'index' -- no phrase in common\n")
    print(f"closest golden  : {c['matched']!r}")
    print(f"similarity      : {c['sim']:.3f}   (bar is {c['threshold']})")
    print(f"stale           : {c['stale']}")
    print(f"source          : {c['result']['source']}")
    print(f"answer          : {c['result']['answer'][:70]} ...")
    print("\nNo `[retrieve_docs]` line above, and no model call: the run ended at")
    print("the first node. The question was never seen; its meaning was.")


# --- tests -------------------------------------------------------------------

def test_the_paraphrase_clears_the_bar():
    c = cached()
    assert c["sim"] >= c["threshold"], (c["sim"], c["threshold"])


def test_it_matched_the_entry_we_added():
    assert cached()["matched"] == NEW_QUESTION


def test_the_entry_is_fresh_not_stale():
    """The entry's doc_version equals what the DOCUMENT STORE stamps for mysql,
    read from its metadata rather than from a hand-kept table, so guard 2 lets it
    past. Re-index that topic and this entry goes stale on its own."""
    assert cached()["stale"] is False


def test_the_run_is_served_from_the_golden_layer():
    assert cached()["result"]["source"] == "golden"
    assert cached()["result"]["golden_route"] == "hit"


def test_the_curated_answer_is_returned_verbatim():
    """Not summarised, not regenerated -- returned. That is what makes it cheap
    and what makes a wrong entry dangerous."""
    assert cached()["result"]["answer"] == NEW_ANSWER


def test_no_text_overlap_carried_the_match():
    """Guard against the demonstration accidentally being a keyword match: strip
    the shared tokens and the two questions have nothing left in common."""
    a = {w.strip("?,.").lower() for w in NEW_QUESTION.split()}
    b = {w.strip("?,.").lower() for w in PARAPHRASE.split()}
    content = (a & b) - {"how", "do", "i", "a", "on", "in", "the", "an", "to"}
    assert content <= {"index", "mysql"}, content


def test_the_stale_redis_entry_still_gets_refused():
    """The new entry must not have moved the guard: Ch29's stale case still
    falls back."""
    c = cached()
    out = c["ch28"].graph.invoke({"question": "How do I configure connection pooling for Redis?"})
    assert out["source"] != "golden"


if __name__ == "__main__":
    demo_add_a_golden_entry()

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
