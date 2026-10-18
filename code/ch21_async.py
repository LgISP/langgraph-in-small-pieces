"""Chapter 21 - Async at Scale: waiting without paying for it.

The doc-bot's spine from Chapter 20 - search_docs -> generate - with a new
node, `research`, between them. `research` asks the model three questions
about the user's question and looks one thing up on the web (a stand-in that
sleeps), and under async it can wait on all four at once.

Every section runs this graph through `ainvoke` and measures one thing:
  1. the cost of waiting    research written three ways: one wait after another,
                            the web lookup alongside the model, everything at once
  2. infectious ainvoke     one async node closes `invoke` for the whole graph
  3. threads                which thread a def node and an async def node run on
  3b. node timeouts         what a node timeout can cancel, and what it cannot
  4. four runs              four questions in series, then in parallel
  5. a semaphore            at most two model calls in flight, four runs at once
  6. common mistakes        a blocking call, a sync checkpointer, a missing await

Timings are printed as seconds with ratios beside them. The seconds depend on
your machine and on how many requests your model server runs at once (LM
Studio: 4 by default); the ratios are what the chapter is about.

Run:  python code/ch21_async.py
"""

import asyncio
import gc
import inspect
import sqlite3
import threading
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from typing import NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeTimeoutError
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import TimeoutPolicy

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen/qwen3.8-27b",
    temperature=0,
    reasoning_effort="none",
    max_tokens=120,          # short answers: the chapter measures waiting, not writing
    timeout=240,
)

DOCS = {
    "kafka": "Kafka Manual: set spring.kafka.listener.type=batch and tune max-poll-records.",
    "redis": "Redis Guide: use a connection pool; configure max-active and max-idle.",
    "mysql": "MySQL Notes: add a B+ tree index on frequently queried columns; use EXPLAIN.",
}

WAIT_TOO_LONG = 5          # section 3b: longer than the one-second node timeout it is given

ANGLES = ["what it is", "how to configure it", "the most common mistake"]

QUESTION = "How do I configure Kafka batch consumers?"
QUESTIONS = [
    "How do I configure Kafka batch consumers?",
    "How should I size a Redis connection pool?",
    "Why is my MySQL query slow?",
    "What does Kafka's max-poll-records do?",
]


# ===========================================================================
# The graph. Three nodes; the middle one waits on four things.
# ===========================================================================
class State(TypedDict):
    question: str
    docs: NotRequired[list[str]]        # written by search_docs
    notes: NotRequired[list[str]]       # written by research: one note per angle, plus the web lookup
    answer: NotRequired[str]            # written by generate


def search_docs_node(state: State) -> dict:
    """Plain `def`: no I/O to wait on, nothing to gain from async."""
    q = state["question"].lower()
    docs = [text for key, text in DOCS.items() if key in q]
    return {"docs": docs}


async def ask(angle: str, question: str, docs: list[str]) -> str:
    """One model call, one angle. The unit `research` is built from."""
    prompt = (f"In two sentences, {angle} for: {question}\n"
              f"Use only these notes: {' '.join(docs) or '(none)'}")
    reply = await llm.ainvoke(prompt)
    return f"[{angle}] {reply.content.strip()}"


async def web_lookup() -> str:
    """A stand-in for a tool that waits on the network: a web search, a database.

    It sleeps on the loop, which is exactly what an `httpx` call does between
    sending the request and the first byte of the reply.
    """
    await asyncio.sleep(1.5)
    return "[web] Kafka batch listeners deliver a List<ConsumerRecord> per poll."


async def research_node(state: State) -> dict:
    """Three model calls and the web lookup, all in flight at once.

    `gather` hands the four coroutines to the loop; each yields at its own
    `await`, and the loop waits on four sockets instead of one. The two slower
    ways to write this node are in section 1.
    """
    *notes, web = await asyncio.gather(*(ask(a, state["question"], state["docs"]) for a in ANGLES),
                                       web_lookup())
    return {"notes": [*notes, web]}


async def generate_node(state: State) -> dict:
    prompt = ("Answer the question in three sentences from these notes.\n"
              f"Question: {state['question']}\nNotes:\n" + "\n".join(state["notes"]))
    reply = await llm.ainvoke(prompt)
    return {"answer": reply.content.strip()}


def build(research=research_node, checkpointer=None, wrap=lambda node: node, timeout=None):
    """The graph, with `research` swappable: every section swaps in the version it measures."""
    b = StateGraph(State)
    b.add_node("search_docs", wrap(search_docs_node))
    b.add_node("research", wrap(research), timeout=timeout)
    b.add_node("generate", wrap(generate_node))
    b.add_edge(START, "search_docs")
    b.add_edge("search_docs", "research")
    b.add_edge("research", "generate")
    b.add_edge("generate", END)
    return b.compile(checkpointer=checkpointer)


graph = build()


# ===========================================================================
# 1. The cost of waiting: research written three ways.
# ===========================================================================
async def research_in_series(state: State) -> dict:
    """Every wait finishes before the next one starts: the time is the sum."""
    notes = [await ask(a, state["question"], state["docs"]) for a in ANGLES]
    notes.append(await web_lookup())
    return {"notes": notes}


async def research_web_alongside(state: State) -> dict:
    """The web lookup waits alongside the model; the three model calls still take turns."""
    async def angles_in_series() -> list[str]:
        return [await ask(a, state["question"], state["docs"]) for a in ANGLES]
    notes, web = await asyncio.gather(angles_in_series(), web_lookup())
    return {"notes": [*notes, web]}


async def the_cost_of_waiting() -> None:
    print("\n===== 1. the cost of waiting: research written three ways =====")
    with timed("every wait one after another") as in_series:
        await build(research_in_series).ainvoke({"question": QUESTION})
    with timed("web lookup alongside the model", compare_to=in_series):
        await build(research_web_alongside).ainvoke({"question": QUESTION})
    with timed("everything at once", compare_to=in_series):
        await graph.ainvoke({"question": QUESTION})
    print("   the first saving is the whole web wait, the loop's; the second is the model server's")


# ===========================================================================
# 2. Infectious ainvoke: one async node closes invoke for the whole graph.
# ===========================================================================
async def two_doors() -> None:
    print("\n===== 2. infectious ainvoke =====")
    result = await graph.ainvoke({"question": QUESTION})
    print(f"   ainvoke : notes={len(result['notes'])}  answer starts: {result['answer'][:60]!r}")
    try:
        graph.invoke({"question": QUESTION})     # the sync door, on a graph with async nodes
    except TypeError as exc:
        lines = str(exc).splitlines()            # two lines: what is wrong, and what to do
        print("   invoke  : TypeError: " + "\n             ".join(lines))
    print("   a sync node answers both doors; an async node is reachable through ainvoke only")


# ===========================================================================
# 3. Threads: where a def node and an async def node run under ainvoke.
# ===========================================================================
def reporting_thread(node):
    """Wrap a node so that it prints the thread it runs on, then runs as before."""
    name = node.__name__.removesuffix("_node")
    if inspect.iscoroutinefunction(node):
        async def wrapped(state: State) -> dict:
            print(f"   {name:<12} async def  ran on thread: {threading.current_thread().name}")
            return await node(state)
    else:
        def wrapped(state: State) -> dict:
            print(f"   {name:<12} def        ran on thread: {threading.current_thread().name}")
            return node(state)
    return wrapped


async def the_threads() -> None:
    print("\n===== 3. threads: the same graph, each node reporting where it ran =====")
    await build(wrap=reporting_thread).ainvoke({"question": QUESTION})
    print("   a plain def is pushed to a worker thread so it cannot stall the loop; an async def runs on the loop itself")


async def slow_research(state: State) -> dict:
    """An async research that waits too long: the timeout is meant to cut it off."""
    await asyncio.sleep(WAIT_TOO_LONG)
    return {"notes": []}


async def blocking_research(state: State) -> dict:
    """The same wait, taken on the loop's thread instead of behind an await."""
    time.sleep(WAIT_TOO_LONG)
    return {"notes": []}


def sync_research(state: State) -> dict:
    return {"notes": []}


async def node_timeouts() -> None:
    """What a node timeout can cut off: only a node that gives the loop a chance."""
    print("\n===== 3b. node timeouts: only what yields can be cancelled =====")
    try:
        build(research=sync_research, timeout=TimeoutPolicy(run_timeout=1))
    except ValueError as exc:
        print(f"   {'def node':<22} -> {type(exc).__name__} at compile: {str(exc).splitlines()[0]}")

    for label, node in (("async def that awaits", slow_research),
                        ("async def that blocks", blocking_research)):
        g = build(research=node, timeout=TimeoutPolicy(run_timeout=1))
        try:
            await g.ainvoke({"question": QUESTION})
            print(f"   {label:<22} -> no error: the node ran its full {WAIT_TOO_LONG}s and returned")
        except NodeTimeoutError as exc:
            print(f"   {label:<22} -> {type(exc).__name__}: {str(exc).splitlines()[0]}")
    print("   a timeout is a cancellation, and only an await is a place to cancel")


# ===========================================================================
# 4. Four runs: in series, then in parallel.
# ===========================================================================
async def across_runs() -> None:
    print(f"\n===== 4. four runs: {len(QUESTIONS)} questions =====")
    with timed("one after another") as in_series:
        for q in QUESTIONS:
            await graph.ainvoke({"question": q})
    with timed("at once (asyncio.gather)", compare_to=in_series):
        await asyncio.gather(*(graph.ainvoke({"question": q}) for q in QUESTIONS))
    print("   the gain is bounded by how many requests the model server runs at once, not by this process")


# ===========================================================================
# 5. A semaphore: never more than N model calls in flight, across runs.
# ===========================================================================
async def a_ceiling(limit: int = 2) -> None:
    print(f"\n===== 5. a semaphore: at most {limit} model calls in flight, {len(QUESTIONS)} runs at once =====")
    gate = asyncio.Semaphore(limit)

    async def guarded(angle: str, question: str, docs: list[str]) -> str:
        who = f"question {QUESTIONS.index(question)}, {angle}"
        t0 = time.perf_counter()
        async with gate:                       # takes a permit, or waits here for one
            waited = time.perf_counter() - t0
            if waited > 0.01:
                print(f"   waited {waited:4.1f}s     : {who}")
            else:
                print(f"   went straight in : {who}")
            return await ask(angle, question, docs)

    async def research_capped(state: State) -> dict:
        """research_node with the gate around each model call."""
        *notes, web = await asyncio.gather(*(guarded(a, state["question"], state["docs"]) for a in ANGLES),
                                           web_lookup())
        return {"notes": [*notes, web]}

    g = build(research_capped)
    with timed(f"{len(QUESTIONS) * len(ANGLES)} research calls asked to start"):
        await asyncio.gather(*(g.ainvoke({"question": q}) for q in QUESTIONS))


# ===========================================================================
# 6. Common mistakes: three that look fine in the editor.
# ===========================================================================
async def research_blocking(state: State) -> dict:
    time.sleep(2)                              # BUG: blocks the loop; nothing else runs meanwhile
    return {"notes": []}


async def research_yielding(state: State) -> dict:
    await asyncio.sleep(2)                     # yields; the loop runs the other runs meanwhile
    return {"notes": []}


async def research_forgot_await(state: State) -> dict:
    notes = [ask(a, state["question"], state["docs"]) for a in ANGLES]   # BUG: no await
    return {"notes": notes}                    # three unstarted coroutines go into state


async def common_mistakes() -> None:
    print("\n===== 6. common mistakes =====")

    # (a) a blocking call inside an async node stalls every run on the loop
    with timed("(a) 4 runs, time.sleep(2) in research"):
        await asyncio.gather(*(build(research_blocking).ainvoke({"question": q}) for q in QUESTIONS))
    with timed("(a) 4 runs, await asyncio.sleep(2)"):
        await asyncio.gather(*(build(research_yielding).ainvoke({"question": q}) for q in QUESTIONS))

    # (b) a sync checkpointer behind the async door
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    g = build(checkpointer=SqliteSaver(conn))
    try:
        await g.ainvoke({"question": QUESTION}, {"configurable": {"thread_id": "t1"}})
    except NotImplementedError as exc:
        print(f"   (b) SqliteSaver + ainvoke      -> NotImplementedError: {str(exc).splitlines()[0]}")   # the rest is the install hint
    conn.close()
    g = build(checkpointer=InMemorySaver())      # InMemorySaver serves both doors
    await g.ainvoke({"question": QUESTION}, {"configurable": {"thread_id": "t1"}})
    print("   (b) InMemorySaver + ainvoke    -> fine; it serves both doors")
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:   # the file-backed fix
        g = build(checkpointer=saver)
        await g.ainvoke({"question": QUESTION}, {"configurable": {"thread_id": "t1"}})
        saved = await g.aget_state({"configurable": {"thread_id": "t1"}})
    print(f"   (b) AsyncSqliteSaver + ainvoke -> fine; checkpoint holds {len(saved.values['notes'])} notes")

    # (c) a coroutine you forgot to await never runs, and the error shows up somewhere else
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        try:
            await build(research_forgot_await).ainvoke({"question": QUESTION})
        except TypeError as exc:
            print(f"   (c) research forgot its awaits -> TypeError, raised in generate: {exc}")
        gc.collect()                           # the dropped coroutines are collected now ...
        print(f"       ... and only now Python warns: {caught[-1].message}")


# ===========================================================================
# A small utility: time a block and print it. Used by every section above.
# ===========================================================================
class timed:
    """`with timed("label"):` times the block and prints the label and the seconds.

    `compare_to=` names an earlier `timed` block, and the line then ends with the
    ratio: how many times faster this block was than that one.
    """

    def __init__(self, label: str, compare_to: "timed | None" = None):
        self.label, self.compare_to = label, compare_to

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        ratio = f"   ({self.compare_to.seconds / self.seconds:.1f}x faster than the first)" if self.compare_to else ""
        print(f"   {self.label:<40} {self.seconds:6.1f}s{ratio}")


async def main() -> None:
    await the_cost_of_waiting()    # 1. how much time waiting costs
    await two_doors()              # 2. infectious ainvoke
    await the_threads()            # 3. threads for sync and async nodes
    await node_timeouts()          # 3b. what a node timeout can cut off
    await across_runs()            # 4. runs in parallel versus in series
    await a_ceiling()              # 5. capping model calls with a semaphore
    await common_mistakes()        # 6. common async mistakes


if __name__ == "__main__":
    asyncio.run(main())
