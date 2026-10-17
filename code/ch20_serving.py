"""Chapter 20 - Serving over HTTP (synchronous).

The web layer is thin: FastAPI takes a question, runs the compiled graph, and
hands the answer back. The request's `thread_id` becomes the graph's `thread_id`,
so a checkpointer lets you come back later with that id and pull the run's state
out of storage. The runId you handed the client IS the thread_id.

Two paths, and the second is why that id matters:

  - the local docs cover the question -> one request, one run, answer returned.
  - they do not -> the graph parks at `approve_web` and asks permission to go
    outside. `/ask` comes back with a runId and NO answer. The only way to get
    one is to return with that id and approve (or refuse) the web search.

That second path is what a cache cannot do for you. A cache hands back an answer
you already had; here there is no answer yet, and the id is the only handle on a
run that is sitting half-finished in the checkpointer waiting for a human.

Run:  python code/ch20_serving.py
      (then POST to http://localhost:8000/ask - see the curl notes at the bottom)
"""

import threading
import uuid
from contextlib import contextmanager
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from typing import NotRequired, Optional, TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",  # your local server, not the OpenAI cloud
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",      # a capable open model that follows instructions
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


# ===========================================================================
# Part 1 - the graph. Retrieve, and if the local docs come up short, ask a
# human for permission before going out to the web. Serving does not change
# the graph; the gate is what gives the runId a job a cache cannot do.
# ===========================================================================
class State(TypedDict):
    question: str                          # input, set by the caller
    docs: NotRequired[list[str]]           # written by search_docs
    approved: NotRequired[bool]            # written by approve_web, from the human
    web_results: NotRequired[list[str]]    # written by web_search, if allowed
    answer: NotRequired[str]               # written by generate


def search_docs_node(state: State) -> dict:
    # Stub lookup - a real vector store is the subject of other chapters. What
    # matters here is that it can come up EMPTY, because that is the branch that
    # sends the run to a human.
    if "kafka" in state["question"].lower():
        return {"docs": [
            "Kafka Manual: set spring.kafka.listener.type=batch to enable batch consumption.",
            "Kafka FAQ: batch consumers need max-poll-records tuning.",
        ]}
    return {"docs": []}


def route_after_docs(state: State) -> str:
    # Ch3's conditional edge, deciding whether a human needs to be involved.
    return "generate" if state.get("docs") else "approve_web"


def approve_web_node(state: State) -> dict:
    """Stop the run and wait for a person. This is Ch7's gate, served over HTTP.

    `interrupt(...)` freezes the graph right here and surfaces its payload. The
    run is now sitting in the checkpointer with `next == ("approve_web",)` and no
    answer in `State`. Nothing resumes it except a later request carrying the
    same thread_id -- which is the runId the caller is holding.
    """
    decision = interrupt({
        "node": "approve_web",
        "ask": "The local docs do not cover this. Search the web?",
        "question": state["question"],
    })
    return {"approved": bool(decision)}


def route_after_approval(state: State) -> str:
    # Refusing is an answer too: fall through to generate and say what is known.
    return "web_search" if state.get("approved") else "generate"


def web_search_node(state: State) -> dict:
    # Stub again - no network in this chapter. The point is that it runs ONLY
    # after a human said so.
    return {"web_results": [
        f"web result for {state['question']!r}: raise the pool ceiling and set a wait timeout.",
    ]}


def generate_node(state: State) -> dict:
    # An ordinary llm.invoke - the node has no idea it is being served over HTTP,
    # nor that a human approved its sources.
    sources = "\n".join(f"- {d}" for d in state.get("docs", []) + state.get("web_results", []))
    if not sources:
        sources = "- (none: no local documents, and the web search was refused)"
    prompt = f"""
        Answer the question concisely, using only the sources below.
        If the sources do not contain the answer, say so in one sentence.

        Question: {state['question']}

        Sources:
        {sources}

        Answer:"""
    answer = llm.invoke(prompt).content.strip()
    return {"answer": answer}


builder = StateGraph(State)
builder.add_node("search_docs", search_docs_node)
builder.add_node("approve_web", approve_web_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "search_docs")
builder.add_conditional_edges("search_docs", route_after_docs, ["approve_web", "generate"])
builder.add_conditional_edges("approve_web", route_after_approval, ["web_search", "generate"])
builder.add_edge("web_search", "generate")
builder.add_edge("generate", END)

# Without a checkpointer the answer lives only in the POST response - and a run
# that parks at `approve_web` could never be picked up again at all.
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ===========================================================================
# Part 2 - the HTTP layer.
# ===========================================================================
app = FastAPI(title="doc-bot")


# One run per thread at a time. Every handler below does a read-then-act on the
# checkpoint -- read the snapshot, decide, invoke -- and two of those interleaved
# on the same thread_id both pass the check and both invoke: a double-clicked
# "approve" runs the approved side effect twice. The shared saver gives every
# replica the same state; it does not serialise their access to it. So a request
# claims the thread for the duration of its run, and a second request on a
# claimed thread is refused with 409 rather than queued -- the caller already
# holds a runId and can GET it. This registry lives in the process: with several
# replicas the claim has to live where the checkpoints do (the Gold challenge).
_in_flight: set[str] = set()
_in_flight_lock = threading.Lock()


@contextmanager
def claim(run_id: str):
    with _in_flight_lock:
        if run_id in _in_flight:
            raise HTTPException(status_code=409, detail="a run is already in progress on this thread")
        _in_flight.add(run_id)
    try:
        yield
    finally:
        with _in_flight_lock:
            _in_flight.discard(run_id)


# The shape of the POST /ask body: a required question, and an optional
# thread_id the caller may supply to name (or resume) the run themselves.
class AskRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


@app.post("/ask")
def ask(req: AskRequest):
    """Start a run. It either finishes, or it parks at the approval gate."""
    # runId IS the graph's thread_id: the caller's if they sent one, else fresh
    run_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": run_id}}

    # A new question is a new run: reset THIS run's derived channels. Without a
    # reducer, a channel keeps its last value, so reusing a thread_id that already
    # held a run would leak its `web_results`/`answer`/`approved` into this one --
    # e.g. an earlier question that approved the web would make a later, web-refused
    # question still generate from the stale web sources. Passing `question` alone
    # overwrites only `question`; the rest must be cleared explicitly.
    fresh_run = {"question": req.question, "docs": [], "web_results": [],
                 "approved": None, "answer": None}

    # A node that raises propagates straight out of invoke. Catch it so the caller
    # gets the runId back instead of a bare 500 with a stack trace; the failure is
    # checkpointed under that id, so GET /run/{id} still reports status "failed".
    with claim(run_id):
        try:
            result = graph.invoke(fresh_run, config=config)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"runId": run_id, "error": str(exc)},
            )

    # A parked run surfaces its interrupt payload under `__interrupt__` instead
    # of an answer. Say so plainly: a status field the caller can branch on, and
    # the question the graph is waiting on.
    if "__interrupt__" in result:
        return {
            "runId": run_id,
            "status": "paused",
            "awaiting": result["__interrupt__"][0].value,
        }

    return {"runId": run_id, "status": "done", "answer": result.get("answer")}


class ResumeRequest(BaseModel):
    approve: bool


@app.post("/run/{run_id}/resume")
def resume(run_id: str, req: ResumeRequest):
    """Answer the gate and let the run finish.

    This is the endpoint that makes the runId worth handing out. The caller has
    no answer yet; the half-finished run is in the checkpointer, and the id is
    the only way back to it. `Command(resume=...)` delivers the human's decision
    as the return value of the `interrupt()` call that stopped the node, and the
    graph carries on from there -- a different process, a different day, the same
    checkpoint.
    """
    config = {"configurable": {"thread_id": run_id}}
    # The claim comes BEFORE the snapshot is read: check-then-resume is the
    # window two concurrent approvals slip through, and the claim closes it.
    with claim(run_id):
        snapshot = graph.get_state(config)
        if snapshot.created_at is None:                                 # <1>
            raise HTTPException(status_code=404, detail="unknown runId")

        # Two ways to be un-resumable, and `next` only catches one. It is empty on a
        # thread that reached END -- but a run that stopped because a node RAISED also
        # has work pending and no gate to answer, and resuming it just re-runs the
        # failure. Ask for the interrupt, which is the thing this endpoint answers.
        if not snapshot.next:
            raise HTTPException(status_code=409, detail="run is already finished")
        if not snapshot.interrupts:
            raise HTTPException(status_code=409, detail="run is not waiting for input")

        result = graph.invoke(Command(resume=req.approve), config=config)
    return {
        "runId": run_id,
        "status": "done",
        "answer": result.get("answer"),
        "webSearched": bool(result.get("web_results")),
    }


@app.get("/run/{run_id}")
def get_run(run_id: str):
    """Take the runId (= thread_id) and pull this run's latest snapshot."""
    # Saver-neutral existence check. An id that never ran comes back with
    # `created_at is None` -- and every BaseCheckpointSaver answers `get_state`
    # that way, so this survives the swap from InMemorySaver to SqliteSaver or
    # RedisSaver. (A `checkpointer.storage` membership test would not: only the
    # InMemorySaver has that dict, so it AttributeErrors on any durable saver.)
    #    One InMemorySaver-only footnote: its `storage` is a defaultdict, so this
    # read materializes one empty entry per unknown id. That is a toy-saver quirk
    # that vanishes on the durable saver you would actually serve on -- there the
    # read touches nothing and leaks nothing.
    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)
    if snapshot.created_at is None:
        raise HTTPException(status_code=404, detail="unknown runId")

    # Four states, not two. `next` is non-empty in three different situations --
    # still going, stopped because a task raised, and waiting for a human -- and
    # a status endpoint that calls all three "running" is lying about the two
    # cases it exists for. Both distinctions are on the snapshot: the error is on
    # a task, the pending gate is in `interrupts`.
    error = next((t.error for t in snapshot.tasks if t.error), None)
    if error:
        status = "failed"
    elif snapshot.interrupts:
        status = "paused"
    elif snapshot.next:
        status = "running"
    else:
        status = "done"

    return {
        "runId": run_id,
        "status": status,
        "question": snapshot.values.get("question"),
        "answer": snapshot.values.get("answer"),
        **({"awaiting": snapshot.interrupts[0].value} if snapshot.interrupts else {}),
        # Present only when it means something. A field that is always there and
        # usually null invites callers to test the wrong thing.
        **({"error": str(error), "parkedAt": list(snapshot.next)} if error else {}),
    }


# ===========================================================================
# How to test (in another terminal):
#
#   # 1) Start the server
#   python code/ch20_serving.py
#
#   # 2) Ask a question; get back a runId and the answer
#   curl -X POST localhost:8000/ask \
#        -H "Content-Type: application/json" \
#        -d '{"question":"How do I configure Kafka batch consumers?"}'
#
#   # 3) Use the returned runId to fetch this run's state
#   curl localhost:8000/run/<paste-the-runId-here>
#
#   # Or open http://localhost:8000/docs for the interactive UI.
# ===========================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
