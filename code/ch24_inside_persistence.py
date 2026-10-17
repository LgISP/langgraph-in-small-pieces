"""Chapter 24 - Inside Persistence (a deep-dive reading).

Chapter 6 handed the graph a checkpointer and the runtime started auto-saving
state after every superstep; Chapter 9 leaned on the same machinery for
subgraphs. This chapter opens the box. There is no new stage and no LLM - we
build a tiny graph, run it under a `SqliteSaver`, and then read the checkpointer's
OWN two SQL tables to see how a checkpoint is actually stored.

The graph fans out on purpose: plan -> {check_docs, search_web} -> generate. A
straight line would never let the writes table earn its keep -- with one node per
superstep the writes table and the next checkpoint move in lockstep. Put TWO nodes
in a superstep and the writes table has a job a checkpoint cannot do: when one
sibling finishes and the other crashes, the superstep never completes, so no
checkpoint is written -- yet the finished sibling's output survives, because it
reached the writes table. Section 2 stages exactly that.

We use a `SqliteSaver`, not an `InMemorySaver`, for two reasons. First, a real
database is the honest object for a chapter called "Inside Persistence": its
storage is two tables with named, readable columns. Second, it survives a process
death - so section 2 can KILL the process mid-superstep and let a fresh one resume
from the file, which an in-memory store can never demonstrate.

The one idea to carry away: a checkpointer keeps TWO tables, not one.

    checkpoints   one row per superstep snapshot (+ a parent_checkpoint_id
                  column that chains each row back to the one before it)
    writes        the per-task outputs that turn one checkpoint into the next,
                  each row tagged with the task_id that produced it

(`SqliteSaver` keeps exactly these two tables and folds each channel's value
into the checkpoint row. `PostgresSaver` splits the values out into a third
table, `checkpoint_blobs`, to store each value once per version; the two-table
core is the same. An `InMemorySaver` holds the same two as plain dicts.)

Around those two tables we look at five more things:

    2. crash      kill the process mid-superstep; a fan-out sibling's output
                  survives in the writes table, and a fresh process resumes it
    3. reserved   errors and interrupts ride the writes table at negative indices
    4. timeline   checkpoint ids are UUIDv6 = time-ordered, so sorting the
                  keys reconstructs the run's timeline; parent_checkpoint_id
                  is that same timeline as a linked list
    5. history    get_state_history() is that timeline, newest-first
    6. values     what a stored channel value + its monotonic version look like

Everything is deterministic EXCEPT the raw ids/versions/task-ids, which embed a
timestamp or a random tie-breaker and therefore differ every run. Those few
lines are tagged "[varies]"; we prove the STRUCTURE around them (ordering,
parent links, integer version prefixes) with values that are stable run to run.

Needs one package the book does not install by default:

    pip install langgraph-checkpoint-sqlite

Run:  python code/ch24_inside_persistence.py
"""

import operator
import os
import sqlite3
import subprocess
import sys
import tempfile
from typing import Annotated, NotRequired, TypedDict

from langgraph.checkpoint.base import WRITES_IDX_MAP
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

THREAD = {"configurable": {"thread_id": "inspect-001"}}
CRASH_THREAD = {"configurable": {"thread_id": "crash-001"}}


class State(TypedDict):
    question: str                              # input, set by the caller
    steps: Annotated[list[str], operator.add]  # each node appends its own name
    answer: NotRequired[str]                   # written by generate


# Pure stub nodes - no LLM, no I/O, like the Part I graphs. The `steps` reducer
# (operator.add) makes each node APPEND, so the accumulating list is something
# concrete to watch land in the checkpoints. `check_docs` and `search_web` run in
# the SAME superstep (plan fans out to both); that shared superstep is what gives
# the writes table something a checkpoint cannot do (section 2).
def plan_node(state: State) -> dict:
    return {"steps": ["plan"]}


def check_docs_node(state: State) -> dict:
    return {"steps": ["check_docs"]}


def search_web_node(state: State) -> dict:
    return {"steps": ["search_web"]}


def search_web_boom(state: State) -> dict:
    # The same node, rigged to die - the sibling that crashes mid-superstep while
    # check_docs finishes. Nothing catches it: the whole process goes down.
    raise RuntimeError("web search backend died mid-superstep")


def generate_node(state: State) -> dict:
    return {"steps": ["generate"], "answer": "final answer"}


def build_graph(saver: SqliteSaver, crash: bool = False):
    # One builder, used three ways: the main inspection, the crash child
    # (crash=True swaps search_web for the node that raises), and the resume.
    # plan fans out to check_docs + search_web (one superstep, two tasks); both
    # feed generate, which joins them.
    builder = StateGraph(State)
    builder.add_node("plan", plan_node)
    builder.add_node("check_docs", check_docs_node)
    builder.add_node("search_web", search_web_boom if crash else search_web_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "check_docs")
    builder.add_edge("plan", "search_web")
    builder.add_edge("check_docs", "generate")
    builder.add_edge("search_web", "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=saver)


# --------------------------------------------------------------------------- #
# reading the file: a plain sqlite3 connection, the way anyone would.
# --------------------------------------------------------------------------- #

def db_rows(db_path: str, sql: str) -> list:
    # Open the checkpoint file with a stock sqlite3 connection - no LangGraph -
    # to make the point that the storage is an ordinary database anyone can read.
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql).fetchall()


def step_labels(saver: SqliteSaver, cfg=THREAD) -> dict:
    # Each checkpoint_id -> its stable step number, via the saver's own list()
    # (it deserializes the metadata blob for us). Used only to LABEL varying ids.
    return {t.checkpoint["id"]: t.metadata["step"] for t in saver.list(cfg)}


def show_two_tables(db_path: str, saver: SqliteSaver) -> None:
    # The headline: a checkpointer is TWO tables. `checkpoints` is one row per
    # superstep snapshot; `writes` is a separate table of per-task outputs, each
    # row keyed to the checkpoint it was computed FROM. We print the SELECTs and
    # their results the way a sqlite3 session would, so the reads are plain to see.
    print("\n===== 1. Two real SQL tables: checkpoints and writes =====")
    print("sqlite> .tables")
    print("  ".join(r[0] for r in db_rows(db_path,
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")))
    print("\ncolumns:")
    print("  checkpoints:", ", ".join(
        r[1] for r in db_rows(db_path, "PRAGMA table_info(checkpoints)")))
    print("  writes     :", ", ".join(
        r[1] for r in db_rows(db_path, "PRAGMA table_info(writes)")))

    step = step_labels(saver)                   # checkpoint_id -> step, for labels

    # checkpoints: checkpoint_id + parent_checkpoint_id ARE the linked list, shown
    # as their real UUIDv6 values -- each row's parent is the id of the row above.
    # The trailing step is our own label (from the metadata blob), not a column.
    print("\nsqlite> SELECT checkpoint_id, parent_checkpoint_id FROM checkpoints"
          " ORDER BY checkpoint_id;")
    print(f"  [varies] the ids are UUIDv6, different every run; each parent = the row above")
    print(f"  {'checkpoint_id':<36}  {'parent_checkpoint_id':<36}  step")
    print(f"  {'-'*36}  {'-'*36}  ----")
    for cid, parent in db_rows(db_path, "SELECT checkpoint_id, parent_checkpoint_id "
                               "FROM checkpoints ORDER BY checkpoint_id"):
        print(f"  {cid:<36}  {(parent or ''):<36}  {step[cid]:>4}")

    # writes: one row per channel a task wrote. A stock SELECT returns the value as
    # bytes; we decode it (via the saver's serde) so you can read what each write
    # carried. Step 1's bundle has TWO tasks -- the fan-out siblings check_docs and
    # search_web -- which is the whole reason the writes table is a separate store.
    # We label them t1/t2 within the step (raw task_ids vary per run) and sort by
    # content, so the order is stable.
    print("\nsqlite> SELECT checkpoint_id, task_id, channel, value FROM writes;")
    print("        (checkpoint_id as its step; value decoded, '-' = a control write")
    print("         with no value; five tasks t1..t5 -- step 1 holds two, the fan-out)")
    W = [(step[cid], task, chan, saver.serde.loads_typed((typ, val)))
         for cid, task, chan, typ, val in db_rows(db_path,
            "SELECT checkpoint_id, task_id, channel, type, value FROM writes")]

    def _task_key(s, t):
        return sorted(f"{c}={v!r}" for (ss, tt, c, v) in W if ss == s and tt == t)

    # ONE stable label per task across the whole run (not per step): there are five
    # tasks in all -- the input, plan, the two fan-out siblings, and generate -- so
    # the labels run t1..t5 in execution order, and step 1's two of them sit adjacent.
    tasks = sorted({(s, t) for (s, t, c, v) in W}, key=lambda st: (st[0], _task_key(*st)))
    label = {st: f"t{i + 1}" for i, st in enumerate(tasks)}
    W.sort(key=lambda x: (x[0], label[(x[0], x[1])], x[2]))

    print(f"  {'step':>4}  task  {'channel':<20}  value (the delta written)")
    print(f"  {'----':>4}  ----  {'-' * 20}  -------------------------")
    for s, t, chan, val in W:
        # control-channel writes (branch:to:*) carry no value -- show '-', not None
        shown = "-" if val is None else repr(val)
        print(f"  {s:>4}  {label[(s, t)]:<4}  {chan:<20}  {shown}")


def show_timeline(db_path: str, saver: SqliteSaver) -> None:
    # Checkpoint ids are UUIDv6: the high bits are a timestamp, so the ids are
    # "unique AND monotonically increasing" (the Checkpoint docstring's words).
    # Consequence: SELECT ... ORDER BY checkpoint_id reconstructs chronological
    # order, and parent_checkpoint_id is that same order as a linked list.
    print("\n===== 4. Checkpoint ids sort as a timeline (UUIDv6 = monotonic) =====")
    step = step_labels(saver)
    ids_sorted = [r[0] for r in db_rows(db_path,
        "SELECT checkpoint_id FROM checkpoints ORDER BY checkpoint_id")]

    # Prove it WITHOUT printing the varying ids: map each sorted id back to its
    # stable step number. A monotonic key order must yield monotonic steps.
    steps_in_key_order = [step[cid] for cid in ids_sorted]
    increasing = all(a < b for a, b in zip(steps_in_key_order, steps_in_key_order[1:]))
    print(f"steps, in checkpoint_id sort order : {steps_in_key_order}")
    print(f"strictly increasing                : {increasing}")

    # the parent_checkpoint_id chain is the same timeline, walked backward
    parents = dict(db_rows(db_path,
        "SELECT checkpoint_id, parent_checkpoint_id FROM checkpoints"))
    walk, cur = [], ids_sorted[-1]              # start at the newest
    while cur is not None:
        walk.append(step[cur])
        cur = parents[cur]
    print(f"parent_checkpoint_id chain (newest->oldest): {walk}")

    # One raw id, shown only to reveal the shape; this exact string differs every
    # run. The 3rd group starting with '6' is the UUIDv6 version nibble.
    print(f"[varies] one raw checkpoint_id     : {ids_sorted[-1]}")


def show_history(graph) -> None:
    # get_state_history() is the PUBLIC read of that same timeline: it yields
    # StateSnapshots newest-first. step/next/source are fully deterministic;
    # this is the API Chapter 6 used for time travel, now explained from below.
    # The fan-out step's `next` lists BOTH siblings.
    print("\n===== 5. get_state_history(): the timeline, newest-first =====")
    for snap in graph.get_state_history(THREAD):
        step = snap.metadata["step"]
        source = snap.metadata["source"]
        nxt = ", ".join(snap.next) if snap.next else "(done)"
        print(f"  step {step:>2}  source={source:<6}  next={nxt}")


def show_channel_values_and_versions(saver: SqliteSaver) -> None:
    # get_tuple() rehydrates ONE checkpoint by deserializing its blob. The three
    # maps below all live INSIDE that one `checkpoint` blob -- they are NOT SQL
    # tables, which is why section 1's `.tables` listed only checkpoints and writes.
    print("\n===== 6. Inside the checkpoint blob: values, versions, versions_seen =====")
    latest = saver.get_tuple(THREAD)
    print("channel_values -- the live state (each channel's current value):")
    for chan, val in latest.checkpoint["channel_values"].items():
        print(f"  {chan:<12} = {val!r}")

    # version string "<counter>.<random>", two fields (the 0 after the dot is the
    # random float's leading digit): only the counter prefix orders
    versions = latest.checkpoint["channel_versions"]
    prefixes = {ch: int(v.split(".")[0]) for ch, v in versions.items()}
    print("\nchannel_versions -- each channel's version (a run-wide counter, +1 per superstep):")
    for chan, pref in sorted(prefixes.items(), key=lambda kv: (kv[1], kv[0])):
        tag = "  <- internal routing channel" if chan.startswith(("branch:", "__")) else ""
        print(f"  {chan:<22} v{pref}{tag}")
    sample = next(iter(versions.values()))
    print(f"[varies] one raw version string: {sample!r}")
    print("         (the '.<random>' suffix is a per-write tie-breaker)")

    # versions_seen: per node, the trigger versions it has already consumed. This is
    # Pregel's scheduler, not a resume-only check -- on every superstep it compares
    # these against channel_versions: a node runs when a trigger it subscribes to is
    # newer than what it last saw, and is left alone once it has already seen it.
    print("\nversions_seen -- per node, the trigger version it last consumed:")
    for node, seen in latest.checkpoint["versions_seen"].items():
        pref = {c: int(v.split(".")[0]) for c, v in seen.items()}
        print(f"  {node:<12} {pref}")


def show_reserved_channels() -> None:
    # Special writes never collide with regular ones: they live at NEGATIVE
    # indices in the writes table, reserved by channel name. This is how an
    # error or an interrupt is stored alongside a task's ordinary output.
    print("\n===== 3. Reserved special-write channels (negative indices) =====")
    for channel, idx in WRITES_IDX_MAP.items():
        print(f"  {channel:<14} -> index {idx}")


def run_crash_child(crash_db: str) -> None:
    # The child role. build_graph(crash=True) swaps search_web for the node that
    # raises. plan commits its checkpoint; then the fan-out superstep runs both
    # siblings -- check_docs finishes and writes to the file, search_web raises and
    # takes the whole process down. Nothing here catches it - that is the crash.
    with SqliteSaver.from_conn_string(crash_db) as saver:
        graph = build_graph(saver, crash=True)
        graph.invoke({"question": "How do I configure Kafka?", "steps": []},
                     CRASH_THREAD)


def dump_writes(crash_db: str, saver: SqliteSaver, label: str) -> None:
    # Dump the whole writes table of the crashed file, decoded, so the crash state
    # and the resumed state can be compared side by side. Sorted by (step, channel,
    # value) so the order is stable even where a channel repeats.
    step = step_labels(saver, CRASH_THREAD)
    n_ck = db_rows(crash_db, "SELECT count(*) FROM checkpoints")[0][0]
    rows = [(step[cid], task, channel, saver.serde.loads_typed((typ, val)))
            for cid, task, channel, typ, val in db_rows(crash_db,
                "SELECT checkpoint_id, task_id, channel, type, value FROM writes")]
    print(f"\n--- {label} (checkpoints: {n_ck}) ---")
    print(f"  {'step':>4}  {'task_id':<8}  {'channel':<20}  value")
    print(f"  {'----':>4}  {'-' * 8}  {'-' * 20}  -----")
    for s, task, channel, dec in sorted(rows, key=lambda r: (r[0], r[2], repr(r[3]))):
        print(f"  {s:>4}  {task[:8]:<8}  {channel:<20}  {'-' if dec is None else repr(dec)}")


def show_crash_and_resume(crash_db: str) -> None:
    # Kill a superstep mid-flight in a SEPARATE process, then read the writes table
    # straight from the crashed file with plain SQL -- and again after resuming, so
    # the two states line up. check_docs finished and search_web crashed in the SAME
    # superstep, so that superstep never completed and NO new checkpoint was written;
    # yet check_docs' output is in the writes table, next to search_web's __error__.
    # On resume, search_web reruns beside the lingering __error__ and the run finishes.
    print("\n===== 2. A real crash: the writes table before and after resume =====")
    child = subprocess.run(
        [sys.executable, __file__, "--crash-child", crash_db],
        capture_output=True, text=True,
    )
    print("child (check_docs finishes; search_web raises, then dies):"
          f" exit code {child.returncode}")
    print("\nsqlite> SELECT checkpoint_id, task_id, channel, value FROM writes"
          " ORDER BY checkpoint_id, channel;")
    print("  [varies] checkpoint_id shown as step; task_id abbreviated; value decoded"
          " ('-' = none)")

    with SqliteSaver.from_conn_string(crash_db) as saver:
        dump_writes(crash_db, saver, "after the crash")
        # neither sibling is in versions_seen (the fan-out step never checkpointed),
        # so check_docs is skipped by WRITE-REPLAY, not by versions_seen -- read it
        # off the crash checkpoint before the resume advances past it.
        seen = list(saver.get_tuple(CRASH_THREAD).checkpoint["versions_seen"])
        out = build_graph(saver).invoke(None, CRASH_THREAD)   # resume from the file
        dump_writes(crash_db, saver, "after resume")
    print(f"\nfinal steps: {out['steps']}")
    print(f"versions_seen at the crash checkpoint: {seen}  (stops at plan; neither sibling)")


if __name__ == "__main__":
    # Child role: build the rigged graph, crash, and let the process die. The
    # parent (show_crash_and_resume) spawned us with this flag and a db path.
    if len(sys.argv) >= 3 and sys.argv[1] == "--crash-child":
        run_crash_child(sys.argv[2])
        sys.exit(0)                             # unreachable: the invoke raises

    workdir = tempfile.mkdtemp()
    db_path = os.path.join(workdir, "ch23.db")
    crash_db = os.path.join(workdir, "ch24_crash.db")

    # One tiny run leaves a full trail behind it; the sections just read it back.
    # Section 2 (the crash) spawns its own process and its own file; it reads
    # nothing off `saver`, so it slots in here without disturbing the others.
    with SqliteSaver.from_conn_string(db_path) as saver:
        graph = build_graph(saver)
        result = graph.invoke(
            {"question": "How do I configure Kafka?", "steps": []}, THREAD)
        print("run finished. final steps:", result["steps"])
        # Point the reader at the actual file (a temp dir, fresh each run) without
        # polluting the stdout the book mirrors -- send it to stderr.
        print(f"checkpoint file: {db_path}\n  read it with:  sqlite3 {db_path}",
              file=sys.stderr)

        show_two_tables(db_path, saver)              # 1
        show_crash_and_resume(crash_db)              # 2
        show_reserved_channels()                     # 3
        show_timeline(db_path, saver)                # 4
        show_history(graph)                          # 5
        show_channel_values_and_versions(saver)      # 6
