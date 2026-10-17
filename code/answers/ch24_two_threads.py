"""Chapter 24 Challenge — Bronze: two threads, two filing cabinets.

The chapter inspected one `thread_id`. Run the same compiled graph on two, and the
`checkpoints` table shows what the schema is really shaped like: **thread first**.
Its primary key is `(thread_id, checkpoint_ns, checkpoint_id)` -- you can read that
line straight out of the `CREATE TABLE`. Every row a thread owns carries its
`thread_id`, so one thread's history is a prefix of the key, never tangled with the
other's.

Each thread recovers its own `[-1, 0, 1, 2, 3]` timeline independently, and no
checkpoint id appears under both -- ids are UUIDv6, so collisions are not prevented
by the key, they are merely astronomically unlikely.

**Why is the table keyed by `thread_id` first, before namespace and id?** Because
every question the runtime actually asks starts with a thread. "Resume this
conversation" means *give me the latest checkpoint of thread T*; "time-travel"
means *give me checkpoint C of thread T*; "delete this user's history" means
*drop thread T*. None of them is "find checkpoint C, wherever it lives". Putting
the thread first makes the common query a prefix lookup instead of a scan -- which
is exactly why the `PRIMARY KEY` lists `thread_id` before `checkpoint_ns` and
`checkpoint_id`. The id is unique on its own; the ordering is not about uniqueness,
it is about which prefix you have in your hand when you ask.

Run:  python code/answers/ch24_two_threads.py   (see answers/README.md)
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ch24_inside_persistence as ch23    # noqa: E402

QUESTIONS = {
    "inspect-A": "How do I configure Kafka?",
    "inspect-B": "How do I tune Redis?",
}

_CACHE: dict = {}


def run_both() -> str:
    # One db file, two threads through the chapter's graph. Return the path so the
    # reads below open it with a stock sqlite3 connection, the way anyone could.
    db_path = os.path.join(tempfile.mkdtemp(), "two_threads.db")
    with SqliteSaver.from_conn_string(db_path) as saver:
        graph = ch23.build_graph(saver)
        for thread, question in QUESTIONS.items():
            graph.invoke({"question": question, "steps": []},
                         {"configurable": {"thread_id": thread}})
    return db_path


def cached() -> str:
    if "db" not in _CACHE:
        _CACHE["db"] = run_both()
    return _CACHE["db"]


def db_rows(db_path: str, sql: str, params=()) -> list:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql, params).fetchall()


def timeline(db_path: str, thread: str) -> list:
    # One thread's checkpoints, in id order, mapped to their step numbers. The
    # WHERE clause is a prefix lookup on the thread_id half of the primary key.
    with SqliteSaver.from_conn_string(db_path) as saver:
        cfg = {"configurable": {"thread_id": thread}}
        step = {t.checkpoint["id"]: t.metadata["step"] for t in saver.list(cfg)}
    ids = [r[0] for r in db_rows(db_path,
        "SELECT checkpoint_id FROM checkpoints WHERE thread_id=? ORDER BY checkpoint_id",
        (thread,))]
    return [step[cid] for cid in ids]


def ids_of(db_path: str, thread: str) -> set:
    return {r[0] for r in db_rows(db_path,
        "SELECT checkpoint_id FROM checkpoints WHERE thread_id=?", (thread,))}


def question_kept(db_path: str, thread: str):
    with SqliteSaver.from_conn_string(db_path) as saver:
        return saver.get_tuple({"configurable": {"thread_id": thread}})


def primary_key_line(db_path: str) -> str:
    sql = db_rows(db_path,
        "SELECT sql FROM sqlite_master WHERE name='checkpoints'")[0][0]
    return next(ln.strip() for ln in sql.splitlines() if "PRIMARY KEY" in ln)


def demo_two_threads():
    db_path = cached()
    threads = [r[0] for r in db_rows(db_path,
        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")]
    print(f"distinct thread_id in checkpoints: {threads}")
    print("  one per thread -- the first column of the primary key\n")

    for thread in QUESTIONS:
        n = db_rows(db_path,
            "SELECT count(*) FROM checkpoints WHERE thread_id=?", (thread,))[0][0]
        print(f"{thread}: {n} checkpoints, steps {timeline(db_path, thread)}")
        latest = question_kept(db_path, thread)
        print(f"   question kept: {latest.checkpoint['channel_values']['question']!r}")
        print(f"   steps kept   : {latest.checkpoint['channel_values']['steps']}")

    a, b = ids_of(db_path, "inspect-A"), ids_of(db_path, "inspect-B")
    print(f"\nshared checkpoint ids: {a & b or '{}'}  "
          f"({len(a)} + {len(b)} ids, none in common)")
    print("\nthe schema itself, thread first:")
    print(f"  {primary_key_line(db_path)}")
    print("\nSorting each thread's ids recovers that thread's own timeline, because")
    print("UUIDv6 is time-ordered -- and it recovers it INDEPENDENTLY, because the")
    print("WHERE thread_id=? never leaves the thread's own rows.")


# --- tests -------------------------------------------------------------------

def test_each_thread_is_its_own_thread_id():
    db = cached()
    threads = [r[0] for r in db_rows(db,
        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")]
    assert threads == sorted(QUESTIONS)


def test_each_thread_recovers_the_same_five_step_timeline():
    db = cached()
    for thread in QUESTIONS:
        assert timeline(db, thread) == [-1, 0, 1, 2, 3], (thread, timeline(db, thread))


def test_no_checkpoint_id_appears_under_both_threads():
    db = cached()
    assert not ids_of(db, "inspect-A") & ids_of(db, "inspect-B")


def test_each_thread_kept_its_own_question():
    db = cached()
    for thread, question in QUESTIONS.items():
        got = question_kept(db, thread)
        assert got.checkpoint["channel_values"]["question"] == question


def test_the_reducer_did_not_leak_across_threads():
    """`steps` is Annotated[..., operator.add]. If threads shared a cabinet this
    is where it would show: B would have eight entries, not four."""
    db = cached()
    for thread in QUESTIONS:
        got = question_kept(db, thread)
        assert got.checkpoint["channel_values"]["steps"] == [
            "plan", "check_docs", "search_web", "generate"]


def test_the_primary_key_lists_thread_id_first():
    """The point of the key order: the lookup is a prefix, not a search."""
    assert primary_key_line(cached()).startswith("PRIMARY KEY (thread_id,")


def test_reading_a_thread_needs_only_its_own_id():
    db = cached()
    assert question_kept(db, "inspect-B") is not None
    assert question_kept(db, "never-ran") is None


if __name__ == "__main__":
    demo_two_threads()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
