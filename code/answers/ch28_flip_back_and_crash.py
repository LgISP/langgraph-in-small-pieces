"""Chapter 28 Challenge — Silver: redeploy onto a collection you deleted.

The chapter's script ends by deleting `docs_v1` after the deployment has moved to
`docs_v2`. This adds the sixth step: redeploy with `ACTIVE_INDEX=docs_v1` and query.

The prediction most people make is "the process fails to start: no such
collection". It is wrong, and the reason is one default. `get_collection()` is a
`Chroma(collection_name=...)` call, and Chroma's constructor takes
`create_collection_if_not_exists=True`. So the deploy succeeds, `list_collections()`
shows `docs_v1` resurrected, and what the new process is holding is a **brand-new
empty collection wearing the name of the one you deleted**.

Then the query blows up on `[0]` of an empty result list -- and that is luck, not a
safety net. Ask for `k=2` in a loop, or write `hits[:1]`, and there is no crash at
all: the bot answers every question with nothing, cheerfully, at full speed. The
chapter's own `App.query()` would be one small edit away from being that bot.

**The crash on paper.** The process building v2 dies in step 2, halfway through
`build_index("2", ...)`. What is the serving process on, and what does a query
return? Its config still says `docs_v1` -- the build never touched it, and a
cut-over is a redeploy that has not happened -- so queries keep being served,
correctly, from the complete old index while a half-built `docs_v2` sits on disk
being nobody's problem. **That is the property blue-green is bought for**: the
thing that decides what is live is owned by the deployment, and the build has no
way to write it.

What blue-green does NOT protect you from is the failure this file demonstrates:
a deployment whose config names something that no longer exists, or never did.
`deploy()` validates nothing. A production version checks readiness before it
routes traffic -- count the documents in the target and refuse to come up if it is
zero, which is a four-line check and the difference between "the deploy failed"
and "the deploy succeeded and the bot has amnesia".

Needs LM Studio at :1234 for embeddings.
Run:  python code/answers/ch28_flip_back_and_crash.py   (see answers/README.md)
"""

import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LM_BASE = "http://localhost:1234/v1"

V1 = {"kafka": "Kafka batch consumption: spring.kafka.listener.type=batch",
      "redis": "Redis connection pool: configure via spring.data.redis.*"}
V2 = {"kafka": "Kafka batch consumption [v2]: listener.type=batch + max-poll-records tuning",
      "redis": "Redis connection pool [v2]: lettuce.pool.max-active / max-idle explained",
      "pg": "PostgreSQL [new]: pgvector extension for vector search"}


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


def _ch27():
    if not server_is_up():
        raise Skip(f"no model server at {LM_BASE}")
    import ch28_bluegreen as ch27
    shutil.rmtree(ch27.DB_DIR, ignore_errors=True)
    return ch27


_CACHE: dict = {}


def cached() -> dict:
    if not _CACHE:
        ch27 = _ch27()
        import chromadb

        ch27.build_index("1", V1)
        ch27.deploy("docs_v1")
        ch27.build_index("2", V2)
        ch27.deploy("docs_v2")
        chromadb.PersistentClient(path=ch27.DB_DIR).delete_collection("docs_v1")

        names_after_delete = sorted(
            c.name for c in chromadb.PersistentClient(path=ch27.DB_DIR).list_collections())

        # Step six: redeploy onto the collection that is gone.
        app = ch27.deploy("docs_v1")
        resurrected = app.col                 # the handle the new process is holding
        count = chromadb.PersistentClient(path=ch27.DB_DIR).get_collection("docs_v1").count()
        names_after_flip = sorted(
            c.name for c in chromadb.PersistentClient(path=ch27.DB_DIR).list_collections())

        hits_k1 = resurrected.similarity_search("kafka batch", k=1)
        crash = None
        try:
            app.query("kafka batch")
        except BaseException as exc:                   # noqa: BLE001
            crash = f"{type(exc).__name__}: {exc}"

        _CACHE.update(ch27=ch27, names_after_delete=names_after_delete,
                      names_after_flip=names_after_flip, count=count,
                      hits_k1=hits_k1, crash=crash, active=app.active)
    return _CACHE


def demo_flip_back():
    if not server_is_up():
        print(f"no model server at {LM_BASE} -- embeddings are needed; skipping")
        return
    c = cached()
    print(f"after deleting docs_v1 : {c['names_after_delete']}")
    print(f"deploy('docs_v1')      : new process serving {c['active']!r}")
    print(f"after the deploy       : {c['names_after_flip']}   <- docs_v1 is back")
    print(f"documents inside it    : {c['count']}\n")

    print(f"the chapter's query()  : {c['crash']}")
    print(f"   ^ an IndexError on [0] of an empty list -- the crash is the shape of")
    print(f"     `hits[0]`, not a guard anybody wrote.\n")
    print(f"same store, k=1, read without [0]: {c['hits_k1']}")
    print("   ^ no exception. A bot written this way answers everything with")
    print("     nothing, at full speed, and reports no error at all.")


# --- tests -------------------------------------------------------------------

def test_the_deleted_collection_is_really_gone_first():
    assert "docs_v1" not in cached()["names_after_delete"]


def test_pointing_at_a_deleted_collection_recreates_it_empty():
    c = cached()
    assert "docs_v1" in c["names_after_flip"]
    assert c["count"] == 0


def test_the_deploy_accepts_a_config_naming_nothing():
    """`deploy` validates nothing -- the process comes up either way."""
    assert cached()["active"] == "docs_v1"


def test_the_crash_is_an_indexerror_not_a_missing_collection():
    """The prediction people get wrong, pinned: the failure is downstream of a
    successful connection to an empty collection."""
    assert cached()["crash"] is not None
    assert cached()["crash"].startswith("IndexError"), cached()["crash"]


def test_without_the_index_zero_there_is_no_crash_at_all():
    """Why the IndexError is luck: the same state, read defensively, is silent."""
    assert cached()["hits_k1"] == []


def test_a_document_count_check_would_have_caught_it():
    """The four-line fix the docstring proposes, demonstrated."""
    c = cached()
    import chromadb
    target = chromadb.PersistentClient(path=c["ch27"].DB_DIR).get_collection("docs_v1")
    assert target.count() == 0                    # a startup check would refuse to come up here


def test_a_healthy_collection_passes_the_same_check():
    """...and does not refuse the good case, which is the other half of a guard
    being useful."""
    c = cached()
    import chromadb
    assert chromadb.PersistentClient(path=c["ch27"].DB_DIR).get_collection("docs_v2").count() == len(V2)


if __name__ == "__main__":
    try:
        demo_flip_back()
    except Skip as e:
        print(f"skipping: {e}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    passed = skipped = 0
    for t in _tests:
        try:
            t()
            print("PASS", t.__name__)
            passed += 1
        except Skip as e:
            print("SKIP", t.__name__, f"({e})")
            skipped += 1
    if skipped:
        print(f"\n{passed} passed, {skipped} SKIPPED (no model server) -- nothing was verified")
    else:
        print(f"\nall {passed} tests passed")
