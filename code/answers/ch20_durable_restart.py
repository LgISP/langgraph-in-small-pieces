"""Chapter 20 Challenge - Silver: make a run survive a restart.

The Gotcha claims `InMemorySaver` dies with its process. This proves it and
then fixes it, without needing two terminals: a *process* restart is simulated
faithfully by throwing the saver away and building a brand-new graph over the
same backing store -- which is exactly what a new process would do.

With `InMemorySaver` the "new process" starts empty and the run is gone (the
`404` the challenge describes). With `SqliteSaver` pointed at a file, the second
graph reads the first one's thread straight off disk.

One API shape to notice: `SqliteSaver.from_conn_string()` is a **context
manager**, not a constructor. Outside a `with` block you get an iterator, not a
saver -- the most common way this swap goes wrong on the first try. A long-lived
server holds the connection for its whole life, which is what `ENTER`/`EXIT`
below stands in for.

Needs one package that is pinned in requirements.txt but only the persistence
answers import -- install it now if you kept your install to the main line:

    pip install langgraph-checkpoint-sqlite

Run:  python code/answers/ch20_durable_restart.py   (see answers/README.md)
"""

import os
import sqlite3
import tempfile
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

THREAD = {"configurable": {"thread_id": "demo"}}


class State(TypedDict):
    question: str
    answer: str


def build() -> StateGraph:
    """The smallest stand-in for @sec-serving's graph: one node, one answer."""
    builder = StateGraph(State)
    builder.add_node("generate", lambda s: {"answer": f"answer to {s['question']}"})
    builder.add_edge(START, "generate")
    builder.add_edge("generate", END)
    return builder


# --------------------------------------------------------------------------- #
# the two halves of the ritual
# --------------------------------------------------------------------------- #

def in_memory_loses_the_run() -> None:
    """POST /ask, GET /run/demo, restart, GET /run/demo -> gone."""
    saver = InMemorySaver()
    build().compile(checkpointer=saver).invoke({"question": "kafka?"}, THREAD)

    same_process = build().compile(checkpointer=saver)
    print("  before restart :", same_process.get_state(THREAD).values or "(empty)")

    # A new process gets a new InMemorySaver. Nothing carries over.
    after_restart = build().compile(checkpointer=InMemorySaver())
    values = after_restart.get_state(THREAD).values
    print("  after  restart :", values or "(empty)  <- the 404")
    assert values == {}


def sqlite_keeps_it(db_path: str) -> None:
    """The same ritual, one argument different."""
    with SqliteSaver.from_conn_string(db_path) as saver:   # ENTER
        saver.setup()                                      # one-time DDL
        build().compile(checkpointer=saver).invoke({"question": "kafka?"}, THREAD)
        print("  before restart :", build().compile(
            checkpointer=saver).get_state(THREAD).values)
    # EXIT: connection closed, "process" gone. The file remains.

    with SqliteSaver.from_conn_string(db_path) as saver:   # a NEW process
        saver.setup()                                      # idempotent
        values = build().compile(checkpointer=saver).get_state(THREAD).values
        print("  after  restart :", values, " <- read off the file")
        assert values["answer"] == "answer to kafka?"


def what_landed_on_disk(db_path: str) -> None:
    """Two tables, and everything in State written into them in the clear."""
    con = sqlite3.connect(db_path)
    tables = [r[0] for r in con.execute(
        "select name from sqlite_master where type='table' order by name")]
    print("  tables         :", tables)
    print("  file size      :", os.path.getsize(db_path), "bytes")
    con.close()


if __name__ == "__main__":
    db_path = os.path.join(tempfile.mkdtemp(), "serving.sqlite")

    print("InMemorySaver -- the run dies with the process")
    in_memory_loses_the_run()

    print("\nSqliteSaver -- the run outlives it")
    sqlite_keeps_it(db_path)

    print("\nwhat is actually on disk")
    what_landed_on_disk(db_path)

    print("\nDurability across a RESTART, on one machine. The other half of the")
    print("Gotcha -- a second replica answering for a run it never executed --")
    print("needs a store every replica can reach, which is why the ladder does")
    print("not stop at a file.")
