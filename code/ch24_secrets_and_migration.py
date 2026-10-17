"""Chapter 24 - Two things the storage layer will not warn you about.

The first example in this chapter reads the tables. This reads what is IN them, and what happens when the
graph changes while a thread is parked between two of them. Both halves fail
silently on the pinned runtime, which is the only reason they are worth a
chapter's space.

1. Everything in State is written to the checkpoint in the clear. A secret you
   put in State is a secret in your database, greppable with `strings`.
   `EncryptedSerializer` fixes the storage half; nothing fixes the fact that a
   secret in State is also a secret in every trace and every `get_state()`.

2. A graph that changes under a parked thread does not raise. Rename the node a
   thread is parked before and the resume SKIPS it. Rename a State key and the
   old value stays on disk, invisible to the new schema, reading back as None.

Needs two packages the book does not install by default:

    pip install langgraph-checkpoint-sqlite pycryptodome

Run:  python code/ch24_secrets_and_migration.py
"""

import os
import sqlite3
import tempfile
from typing import TypedDict

from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

SECRET = "sk-live-DO-NOT-LEAK"
CFG = {"configurable": {"thread_id": "t1"}}


# --------------------------------------------------------------------------- #
# 1. what a checkpoint keeps, and in what form
# --------------------------------------------------------------------------- #

class Creds(TypedDict):
    api_key: str
    answer: str


def creds_graph() -> StateGraph:
    builder = StateGraph(Creds)
    builder.add_node("call", lambda s: {"answer": "ok"})
    builder.add_edge(START, "call")
    builder.add_edge("call", END)
    return builder


def write_once(path: str, serde=None) -> bytes:
    with SqliteSaver.from_conn_string(path) as saver:
        if serde is not None:
            saver.serde = serde          # swap the serializer, keep the store
        saver.setup()
        creds_graph().compile(checkpointer=saver).invoke(
            {"api_key": SECRET, "answer": ""}, CFG)
    return open(path, "rb").read()


def secrets_demo(workdir: str) -> None:
    plain = write_once(os.path.join(workdir, "plain.sqlite"))
    print(f"  plain file     : secret readable with `strings` -> "
          f"{SECRET.encode() in plain}")

    os.environ.setdefault("LANGGRAPH_AES_KEY", "0" * 32)   # 16, 24 or 32 bytes
    serde = EncryptedSerializer.from_pycryptodome_aes()

    enc_path = os.path.join(workdir, "encrypted.sqlite")
    encrypted = write_once(enc_path, serde=serde)
    print(f"  encrypted file : secret readable with `strings` -> "
          f"{SECRET.encode() in encrypted}")

    with SqliteSaver.from_conn_string(enc_path) as saver:
        saver.serde = serde
        values = creds_graph().compile(checkpointer=saver).get_state(CFG).values
        print(f"  with the key   : round-trips -> {values['api_key'] == SECRET}")

    with SqliteSaver.from_conn_string(enc_path) as saver:   # no serde: no key
        try:
            creds_graph().compile(checkpointer=saver).get_state(CFG)
            print("  without it     : read succeeded (unexpected)")
        except NotImplementedError as err:
            print(f"  without it     : {type(err).__name__}: {err}")


# --------------------------------------------------------------------------- #
# 2. changing the graph under a parked thread
# --------------------------------------------------------------------------- #

class V1(TypedDict):
    question: str
    answer: str


def v1() -> StateGraph:
    builder = StateGraph(V1)
    builder.add_node("plan", lambda s: {"question": s["question"] + " (planned)"})
    builder.add_node("generate", lambda s: {"answer": "v1 answer"})
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "generate")
    builder.add_edge("generate", END)
    return builder


def v2_node_renamed() -> StateGraph:
    """`generate` is now called `respond`. Nothing else changed."""
    builder = StateGraph(V1)
    builder.add_node("plan", lambda s: {"question": s["question"]})
    builder.add_node("respond", lambda s: {"answer": "v2 answer"})
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "respond")
    builder.add_edge("respond", END)
    return builder


class V3(TypedDict):
    query: str          # was `question`
    answer: str


def v3_key_renamed() -> StateGraph:
    builder = StateGraph(V3)
    builder.add_node("plan", lambda s: {"query": s.get("query", "<missing>")})
    builder.add_node("generate", lambda s: {"answer": f"saw query={s.get('query')!r}"})
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "generate")
    builder.add_edge("generate", END)
    return builder


def park(path: str) -> None:
    with SqliteSaver.from_conn_string(path) as saver:
        saver.setup()
        graph = v1().compile(checkpointer=saver)
        graph.invoke({"question": "kafka?", "answer": ""}, CFG,
                     interrupt_before=["generate"])
        print(f"  parked at      : next={graph.get_state(CFG).next} "
              f"values={graph.get_state(CFG).values}")


def migration_demo(workdir: str) -> None:
    node_db = os.path.join(workdir, "node_rename.sqlite")
    park(node_db)
    with SqliteSaver.from_conn_string(node_db) as saver:
        out = v2_node_renamed().compile(checkpointer=saver).invoke(None, CFG)
        print(f"  after rename   : {out}")
        print(f"                   answer still empty -> "
              f"{out['answer'] == ''}   (the node was SKIPPED, no error)")

    key_db = os.path.join(workdir, "key_rename.sqlite")
    park(key_db)
    with SqliteSaver.from_conn_string(key_db) as saver:
        graph = v3_key_renamed().compile(checkpointer=saver)
        print(f"  v3 sees state  : {graph.get_state(CFG).values}"
              f"   (`question` is on disk, invisible to this schema)")
        print(f"  after resume   : {graph.invoke(None, CFG)}")


if __name__ == "__main__":
    workdir = tempfile.mkdtemp()

    print("1. what a checkpoint keeps")
    secrets_demo(workdir)

    print("\n2. changing the graph under a parked thread")
    migration_demo(workdir)

    print("\nNeither migration raised. That is the finding.")
