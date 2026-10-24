"""Chapter 15 - a thread that keeps everything, stored two ways.

Chapter 15's `summarize` node keeps the conversation short by deleting old
turns. Sometimes you cannot: an audit trail, a run you must be able to replay
exactly, anything where the record is the point. Then the history stays, and
the cost of keeping it shows up on disk rather than in the context window.

This measures that cost. Sixty turns on one thread, nothing deleted, the same
graph twice -- once with the usual `add_messages` reducer, once with the same
messages behind a `DeltaChannel`. The final conversation is identical; what
differs is how many bytes the checkpointer wrote to get there.

`DeltaChannel` is beta in LangGraph 1.2.16: the API and the on-disk shape may
both change. Read the numbers, not the format.

No model: the replies are canned, because what is being measured is the
checkpointer, not the model. Message ids are fixed rather than random so the
byte counts come out the same on every run; the totals are printed in KB,
which rounds off the last few bytes of checkpoint-id jitter.

Run:  python code/ch15_delta_channel.py
"""

import os
import sqlite3
import tempfile
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

TURNS = 60
REPLY = "Here is a paragraph of answer. " * 8        # a reply of ordinary length


def add_messages_batch(state, writes):
    """`add_messages`, but taking a batch of writes at a time.

    A DeltaChannel reducer is handed several writes in one call, because it
    replays history in larger batches than the run produced them. Folding them
    one at a time, as here, satisfies that: the result does not depend on how
    the writes were grouped.
    """
    out = state or []
    for w in writes:
        out = add_messages(out, w)
    return out


class Plain(TypedDict):
    messages: Annotated[list, add_messages]


class Delta(TypedDict):
    messages: Annotated[list, DeltaChannel(add_messages_batch)]


def one_thread(state_schema) -> tuple[int, int, int]:
    """Sixty turns on one thread. Returns the message count and the bytes written."""
    conn = sqlite3.connect(os.path.join(tempfile.mkdtemp(), "t.sqlite"), check_same_thread=False)
    b = StateGraph(state_schema)
    turn = {"n": 0}

    def chat(state):
        turn["n"] += 1
        return {"messages": [AIMessage(REPLY, id=f"a{turn['n']}")]}   # fixed ids: the bytes repeat

    b.add_node("chat", chat)
    b.add_edge(START, "chat")
    b.add_edge("chat", END)
    graph = b.compile(checkpointer=SqliteSaver(conn))

    config = {"configurable": {"thread_id": "t"}}
    for i in range(TURNS):
        graph.invoke({"messages": [HumanMessage(f"question {i}", id=f"h{i}")]}, config)

    kept = graph.get_state(config).values["messages"]
    checkpoints = conn.execute("select sum(length(checkpoint)) from checkpoints").fetchone()[0] or 0
    writes = conn.execute("select sum(length(value)) from writes").fetchone()[0] or 0
    conn.close()
    return len(kept), checkpoints, writes


if __name__ == "__main__":
    print(f"\n===== {TURNS} turns on one thread, nothing deleted =====")
    print(f"   {'reducer':<14}{'#msg':>6}{'checkpoints':>14}{'writes':>10}{'total':>10}")
    print(f"   {'-' * 12:<14}{'-' * 4:>6}{'-' * 12:>14}{'-' * 8:>10}{'-' * 8:>10}")
    results = {}
    for label, schema in (("add_messages", Plain), ("DeltaChannel", Delta)):
        kept, ck, wr = one_thread(schema)
        results[label] = (kept, ck + wr)
        print(f"   {label:<14}{kept:>6}{ck // 1024:>11,} KB{wr // 1024:>7,} KB{(ck + wr) // 1024:>7,} KB")
    print()

    a, b = results["add_messages"], results["DeltaChannel"]
    assert a[0] == b[0] == TURNS * 2        # same conversation, both ways
    assert b[1] * 4 < a[1]                  # and it is not a near thing
    print(f"   same {a[0]} messages either way; the delta thread wrote {a[1] / b[1]:.0f}x less")
