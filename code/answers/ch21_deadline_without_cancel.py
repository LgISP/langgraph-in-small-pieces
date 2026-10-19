"""Chapter 21 Challenge - Silver: a deadline for the request, not for the run.

A caller will not hold a connection open for thirty seconds, but they still
want the answer. Those are two different limits, and `asyncio.wait_for` only
knows how to serve the first: it cancels the task it wraps, so the deadline
that was meant to free the connection destroys the run as well.

`asyncio.wait` is the other half of the pair. It never raises on a timeout --
whatever has not finished comes back in `pending`, still running -- so the
handler can answer `202 Accepted` with a run id and leave the graph to finish
on the event loop. The answer is read back later from the checkpointer, which
is why this only works with an async-capable saver: `AsyncSqliteSaver` here,
`SqliteSaver` being the one that raises under `ainvoke` (Section 21.12).

A cancelled run leaves no mark of its own: the thread's last checkpoint is
whatever the cancel interrupted, which is exactly what a run still in progress
looks like. `GET /run/{id}` answers "running" for both, forever.

Three endpoints, exercised in-process through httpx's ASGI transport, so the
whole thing runs in one file with no server to start:

    POST /ask            200 with the answer, or 202 with a run id
    GET  /run/{id}       404 unknown, 202 still running, 200 with the answer
    POST /ask-wait-for   the trap, kept so the tests can show what it costs

Needs one package that is pinned in requirements.txt but only the persistence
answers import -- install it now if you kept your install to the main line:

    pip install langgraph-checkpoint-sqlite

Run:  python code/answers/ch21_deadline_without_cancel.py   (see answers/README.md)
"""

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from typing import NotRequired, TypedDict

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

DEADLINE = 0.3             # how long a request may hold the connection
SLOW = 1.2                 # how long the graph takes: comfortably past it
FAST = 0.05                # ... and a question the graph answers inside it


# ---------------------------------------------------------------------------
# The graph. Two nodes, both async, both waiting on nothing in particular.
# ---------------------------------------------------------------------------
class State(TypedDict):
    question: str
    wait: float
    docs: NotRequired[list[str]]
    answer: NotRequired[str]


async def search_docs(state: State) -> dict:
    await asyncio.sleep(state["wait"] / 2)
    return {"docs": [f"docs for {state['question']}"]}


async def generate(state: State) -> dict:
    await asyncio.sleep(state["wait"] / 2)
    return {"answer": f"answer from {state['docs'][0]}"}


def build(checkpointer):
    b = StateGraph(State)
    b.add_node("search_docs", search_docs)
    b.add_node("generate", generate)
    b.add_edge(START, "search_docs")
    b.add_edge("search_docs", "generate")
    b.add_edge("generate", END)
    return b.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# The server. One saver for the life of the process, as a real one would hold it.
# ---------------------------------------------------------------------------
GRAPH = None
IN_FLIGHT: dict[str, asyncio.Task] = {}        # a reference, or the loop may collect the task


@asynccontextmanager
async def serving():
    """What a real server does in its lifespan handler: one saver, held open."""
    global GRAPH
    path = os.path.join(tempfile.mkdtemp(), "runs.sqlite")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        GRAPH = build(saver)
        yield
    GRAPH = None


app = FastAPI()


class AskRequest(BaseModel):
    question: str
    thread_id: str
    wait: float = SLOW


def cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@app.post("/ask")
async def ask(req: AskRequest):
    """Start the run, give it `DEADLINE` to finish, hand back a run id if it does not."""
    task = asyncio.create_task(
        GRAPH.ainvoke({"question": req.question, "wait": req.wait}, cfg(req.thread_id))
    )
    IN_FLIGHT[req.thread_id] = task
    done, pending = await asyncio.wait([task], timeout=DEADLINE)   # never raises; leaves it running
    if task in done:
        IN_FLIGHT.pop(req.thread_id, None)
        return {"status": "done", "answer": task.result()["answer"]}
    return JSONResponse(
        status_code=202,
        content={"status": "running", "run_id": req.thread_id},
    )


@app.post("/ask-wait-for")
async def ask_wait_for(req: AskRequest):
    """The trap: the same deadline written with `wait_for`, which cancels the run."""
    try:
        result = await asyncio.wait_for(
            GRAPH.ainvoke({"question": req.question, "wait": req.wait}, cfg(req.thread_id)),
            timeout=DEADLINE,
        )
    except TimeoutError:
        return JSONResponse(status_code=202, content={"status": "running", "run_id": req.thread_id})
    return {"status": "done", "answer": result["answer"]}


@app.get("/run/{run_id}")
async def get_run(run_id: str):
    """Read the thread back. The checkpointer is the only thing that remembers."""
    snap = await GRAPH.aget_state(cfg(run_id))
    if not snap.values:
        return JSONResponse(status_code=404, content={"status": "unknown"})
    if "answer" not in snap.values:
        return JSONResponse(
            status_code=202,
            content={"status": "running", "next": list(snap.next)},
        )
    return {"status": "done", "answer": snap.values["answer"]}


# ---------------------------------------------------------------------------
# The demo, through the ASGI transport: a real request/response, no socket.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def client():
    """httpx talks to the app object directly; `serving()` stands in for the lifespan."""
    async with serving():
        async with httpx.ASGITransport(app=app) as transport:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                yield c


async def scenario() -> dict:
    out: dict = {}
    async with client() as c:
        # 1. a question the graph answers inside the deadline: one request, done.
        r = await c.post("/ask", json={"question": "fast", "thread_id": "t-fast", "wait": FAST})
        out["fast_status"] = r.status_code
        out["fast_body"] = r.json()

        # 2. a slow one: the connection is freed, the run is not.
        r = await c.post("/ask", json={"question": "slow", "thread_id": "t-slow", "wait": SLOW})
        out["slow_status"] = r.status_code
        out["slow_body"] = r.json()

        r = await c.get("/run/t-slow")                      # asked too early, on purpose
        out["early_status"] = r.status_code
        out["early_body"] = r.json()

        await IN_FLIGHT.pop("t-slow")                       # let it finish, as the loop would
        r = await c.get("/run/t-slow")
        out["late_status"] = r.status_code
        out["late_body"] = r.json()

        # 3. the trap: the same deadline, written with wait_for.
        r = await c.post("/ask-wait-for",
                         json={"question": "slow", "thread_id": "t-killed", "wait": SLOW})
        out["trap_status"] = r.status_code
        await asyncio.sleep(SLOW * 1.5)                     # wait longer than the run needed
        r = await c.get("/run/t-killed")
        out["trap_later_status"] = r.status_code
        out["trap_later_body"] = r.json()

        # 4. a run nobody started.
        r = await c.get("/run/t-nothing")
        out["unknown_status"] = r.status_code
    return out


def test_a_run_inside_the_deadline_answers_in_one_request(r):
    assert r["fast_status"] == 200
    assert r["fast_body"]["answer"] == "answer from docs for fast"


def test_a_slow_run_frees_the_connection_and_keeps_going(r):
    assert r["slow_status"] == 202                          # the caller is let go ...
    assert r["slow_body"]["run_id"] == "t-slow"
    assert r["early_status"] == 202                         # ... and the run is still on its way
    assert r["early_body"]["next"] != []


def test_the_answer_is_there_when_the_caller_comes_back(r):
    assert r["late_status"] == 200
    assert r["late_body"]["answer"] == "answer from docs for slow"


def test_wait_for_frees_the_connection_by_destroying_the_run(r):
    assert r["trap_status"] == 202                          # the same 202 the caller sees ...
    assert r["trap_later_status"] == 202                    # ... but it never becomes an answer
    assert r["trap_later_body"]["next"] == ["search_docs"]  # cancelled inside the first node


def test_a_thread_nobody_started_is_a_404(r):
    assert r["unknown_status"] == 404


if __name__ == "__main__":
    results = asyncio.run(scenario())

    def line(what, status, tail):
        print(f"   {what:<34}-> {status}  {tail}")

    print(f"\n=== a deadline of {DEADLINE}s for the request, not for the run ===")
    line(f"POST /ask          ({FAST}s run)", results["fast_status"],
         repr(results["fast_body"]["answer"]))
    line(f"POST /ask          ({SLOW}s run)", results["slow_status"],
         f"run_id={results['slow_body']['run_id']!r}")
    line("GET  /run/t-slow   (too early)", results["early_status"],
         f"next={results['early_body']['next']}")
    line("GET  /run/t-slow   (after it ran)", results["late_status"],
         repr(results["late_body"]["answer"]))

    print("\n=== the same deadline written with wait_for ===")
    line(f"POST /ask-wait-for ({SLOW}s run)", results["trap_status"], "run_id='t-killed'")
    line("GET  /run/t-killed (long after)", results["trap_later_status"],
         f"next={results['trap_later_body']['next']}")
    print("   the status endpoint cannot tell a cancelled run from a slow one: "
          "the checkpoint looks the same")

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(results)
    print(f"\nall {len(tests)} tests passed ✅")
