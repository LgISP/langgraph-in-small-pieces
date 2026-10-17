"""Chapter 20 - Serving over HTTP (async + streaming).

The synchronous server made the client wait until the whole answer was ready.
A real answer can take many seconds of LLM generation, and a browser wants to
show words as they arrive - the ChatGPT typewriter. So instead of returning one
JSON blob, this server streams the answer token by token.

Two things change versus the sync version, and nothing else does:
  - the endpoint is `async def` and returns a `StreamingResponse`;
  - it drives the graph with `graph.astream(..., stream_mode="messages")`, which
    yields (message_chunk, metadata) for every LLM token as it is produced.
The graph and the node bodies are untouched. Streaming is a decision made at the
HTTP layer, exactly as Chapter 16 argued for `graph.stream`.

The `thread_id` still rides in from the request, and the checkpointer still
stores each run - so after the stream ends the client can GET /run/{run_id} and
retrieve the finished answer by the same id.

Run:  python code/ch20_serving_async.py
      (then POST to http://localhost:8001/ask - see the curl notes at the bottom)
"""

import json
import uuid
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from typing import AsyncIterator, NotRequired, Optional, TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
# The graph - identical to the sync server, approval gate and all. Serving
# choices never touch it. One consequence worth knowing before you run this:
# a question the local docs do NOT cover parks at `approve_web` before
# `generate` ever runs, so the stream closes with no tokens and GET /run/{id}
# reports "paused". Answering the gate is `POST /run/{id}/resume` below -- the
# async twin of the sync server's, driving the same checkpoint from this process.
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
# HTTP layer.
# ===========================================================================
app = FastAPI(title="doc-bot (async)")


# One run at a time per thread, exactly as in the sync server -- and for the same
# reason, which async makes worse rather than better. Read the gate, decide it is
# still open, then resume: two requests that interleave between the decision and
# the resume both approve the same interrupt, and "approve" runs the approved
# side effect twice. Under `def` endpoints that window is a thread scheduling
# accident. Under `async def` it is not an accident at all: `await` IS the yield
# point, so the second request is guaranteed the chance to read the same snapshot
# while the first is suspended. Measured on this file before the claim existed --
# two concurrent resumes, both HTTP 200, both generating, start times equal to
# the millisecond.
#
# No `threading.Lock` here, unlike the sync twin. One event loop never switches
# between two statements that do not `await`, so `test, then add` is already
# indivisible -- provided nothing awaits BETWEEN them. That proviso is the whole
# design: the claim has to be taken before the first `await`, not after it.
_in_flight: set[str] = set()


def take(run_id: str) -> None:
    """Claim the thread, or refuse. Nothing may `await` between these two lines."""
    if run_id in _in_flight:
        raise HTTPException(status_code=409,
                            detail="a run is already in progress on this thread")
    _in_flight.add(run_id)


@contextmanager
def claim(run_id: str):
    """For a request that does its work before it responds."""
    take(run_id)
    try:
        yield
    finally:
        _in_flight.discard(run_id)


class AskRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


async def stream_tokens(question: str, run_id: str) -> AsyncIterator[str]:
    """Drive the graph and yield Server-Sent Events, one per LLM token.

    stream_mode="messages" surfaces (message_chunk, metadata) for every token.
    We keep only the tokens emitted by the `generate` node and forward each as an
    SSE `data:` line; a final [DONE] marks the end of the stream.
    """
    config = {"configurable": {"thread_id": run_id}}
    # A new question is a new run: reset this run's derived channels so a reused
    # thread_id cannot leak an earlier run's web_results/answer/approved into it
    # (see ch20_serving.py for the full explanation).
    fresh_run = {"question": question, "docs": [], "web_results": [],
                 "approved": None, "answer": None}
    # A stream has three ways to end, and the client must be able to tell them
    # apart: completed (tokens, then [DONE]), paused (a `paused` event, then
    # [DONE]), and failed (a `failed` event, then [DONE]). [DONE] is the only
    # thing that says "the server finished what it had to say" -- a stream that
    # ends without it was cut off, and its partial text is not an answer.
    # The claim this generator releases was taken in `ask`, before the response
    # existed. `finally` and not a plain trailing line: the client can disconnect
    # mid-stream, which closes the generator here rather than running it to the
    # end, and a claim that is only released on the happy path locks the thread
    # for the life of the process.
    try:
        try:
            async for msg_chunk, metadata in graph.astream(
                fresh_run, config=config, stream_mode="messages"
            ):
                if metadata.get("langgraph_node") == "generate" and msg_chunk.content:
                    # JSON-encode so tokens with newlines cannot break SSE framing.
                    yield f"data: {json.dumps({'token': msg_chunk.content})}\n\n"
        except Exception as exc:   # noqa: BLE001 - a node raised mid-stream
            # The failure is checkpointed under this run id, so GET /run/{id}
            # reports it with where it stopped; here the client just needs to know
            # the tokens it has are not an answer. Redact: it may quote internals.
            yield f"data: {json.dumps({'failed': type(exc).__name__})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Missed the local docs and parked at the gate: no tokens went out, so say
        # so. (messages mode doesn't surface the interrupt; read it off the state.)
        snapshot = await graph.aget_state(config)
        if snapshot.interrupts:
            yield f"data: {json.dumps({'paused': snapshot.interrupts[0].value})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        _in_flight.discard(run_id)


@app.post("/ask")
async def ask(req: AskRequest):
    """Stream the answer token by token as it is generated."""
    run_id = req.thread_id or str(uuid.uuid4())
    # Taken here and released in the generator, which is the one place the sync
    # twin's `with claim(run_id):` cannot be copied. A StreamingResponse hands
    # back a generator: the endpoint returns BEFORE the work starts, so a `with`
    # around this return would release the claim before the first token. Taking
    # it here is also what makes a duplicate a real 409 instead of an error event
    # inside a 200 stream.
    take(run_id)
    return StreamingResponse(
        stream_tokens(req.question, run_id),
        media_type="text/event-stream",
        headers={"X-Run-Id": run_id},
    )


@app.get("/run/{run_id}")
async def get_run(run_id: str):
    """After the stream ends, the finished answer is still in the checkpoint."""
    config = {"configurable": {"thread_id": run_id}}
    snapshot = await graph.aget_state(config)   # async handler, async read
    # Saver-neutral existence check: an id that never ran comes back with
    # `created_at is None`. Every BaseCheckpointSaver answers `aget_state` this
    # way -- unlike `checkpointer.storage`, which only the InMemorySaver has, so
    # the moment you swap in SqliteSaver/RedisSaver a `.storage` probe would
    # AttributeError. The same read serves the 404 and the status below.
    if snapshot.created_at is None:
        raise HTTPException(status_code=404, detail="unknown runId")

    error = next((t.error for t in snapshot.tasks if t.error), None)
    if error:
        status = "failed"
    elif snapshot.interrupts:      # parked at approve_web, waiting for a human
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
        # Same contract as the sync server: while paused, echo what the run is
        # waiting on so a client need not re-read the stream to learn the gate.
        **({"awaiting": snapshot.interrupts[0].value} if snapshot.interrupts else {}),
        **({"error": str(error), "parkedAt": list(snapshot.next)} if error else {}),
    }


class ResumeRequest(BaseModel):
    approve: bool


@app.post("/run/{run_id}/resume")
async def resume(run_id: str, req: ResumeRequest):
    """Answer the gate and let the parked run finish -- the async twin of the
    sync server's /resume.

    The stream closed with a `paused` event and no answer; the half-finished run
    is in the checkpointer under this id, and this endpoint is the only way to
    produce one. `Command(resume=...)` delivers the human's decision as the return
    value of the `interrupt()` that stopped the node, and the graph carries on.
    """
    config = {"configurable": {"thread_id": run_id}}
    # Before the first `await`, and around all three steps. Claiming after the
    # snapshot is read would leave the window exactly where it was.
    with claim(run_id):
        snapshot = await graph.aget_state(config)
        if snapshot.created_at is None:
            raise HTTPException(status_code=404, detail="unknown runId")
        # Two ways to be un-resumable, and `next` only catches one -- a finished
        # run. A run that stopped because a node RAISED also has pending work and
        # no gate; resuming it just re-runs the failure. Ask for the interrupt
        # this endpoint answers.
        if not snapshot.next:
            raise HTTPException(status_code=409, detail="run is already finished")
        if not snapshot.interrupts:
            raise HTTPException(status_code=409, detail="run is not waiting for input")

        result = await graph.ainvoke(Command(resume=req.approve), config=config)
    return {
        "runId": run_id,
        "status": "done",
        "answer": result.get("answer"),
        "webSearched": bool(result.get("web_results")),
    }


# ===========================================================================
# How to test (in another terminal):
#
#   # 1) Start the server
#   python code/ch20_serving_async.py
#
#   # 2) Stream the answer. -N tells curl not to buffer, so tokens print live.
#   curl -N -X POST localhost:8001/ask \
#        -H "Content-Type: application/json" \
#        -d '{"question":"How do I configure Kafka batch consumers?"}'
#   # -> data: {"token": "To"}
#   #    data: {"token": " configure"}
#   #    ... one event per token ...
#   #    data: [DONE]
#
#   # 3) Supply your own thread_id, then fetch the stored run by that id
#   curl -N -X POST localhost:8001/ask \
#        -H "Content-Type: application/json" \
#        -d '{"question":"How do I configure Kafka batch consumers?","thread_id":"demo-2"}'
#   curl localhost:8001/run/demo-2
#
#   # 4) A question the local docs miss parks at the gate: the stream ends with a
#   #    `paused` event and no tokens. Resume it by the same id to get an answer.
#   curl -N -X POST localhost:8001/ask \
#        -H "Content-Type: application/json" \
#        -d '{"question":"How do I tune Redis connection pools?","thread_id":"demo-3"}'
#   # -> data: {"paused": {...}}  /  data: [DONE]
#   curl localhost:8001/run/demo-3                 # status: paused, awaiting: {...}
#   curl -X POST localhost:8001/run/demo-3/resume \
#        -H "Content-Type: application/json" \
#        -d '{"approve":true}'                     # status: done, webSearched: true
# ===========================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
