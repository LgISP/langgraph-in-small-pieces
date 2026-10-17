"""Chapter 20 - Try it: a run that fails is a status, not a stack trace.

Two endpoints already carry the failure branch; this shows it, offline. The
serving module's graph is swapped for one whose `generate` node raises, and the
HTTP layer is driven in-process with FastAPI's TestClient -- no uvicorn, no
model. Then the same failing graph is streamed through the async endpoint's
generator, to see the SSE stream's third ending.

  POST /ask               -> 500, and the body still carries the runId
  GET  /run/{id}          -> 200 status "failed", with the error and where it stopped
  POST /run/{id}/resume   -> 409: there is no gate to answer, only a failure to re-run
  stream_tokens(...)      -> a `failed` event, then [DONE]

Run:  python code/ch20_failed_run.py
"""

import asyncio
import json

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

import ch20_serving as sync_srv
import ch20_serving_async as async_srv


def failing_graph(state_cls):
    """The chapter's graph with one difference: generate raises."""
    def search_docs(state):
        return {"docs": ["Kafka Manual: set spring.kafka.listener.type=batch."]}

    def generate(state):
        raise RuntimeError("model endpoint returned 502")

    b = StateGraph(state_cls)
    b.add_node("search_docs", search_docs)
    b.add_node("generate", generate)
    b.add_edge(START, "search_docs")
    b.add_edge("search_docs", "generate")
    b.add_edge("generate", END)
    return b.compile(checkpointer=InMemorySaver())


def show_http() -> dict:
    sync_srv.graph = failing_graph(sync_srv.State)      # the handlers read the module global
    client = TestClient(sync_srv.app)

    r = client.post("/ask", json={"question": "How do I configure Kafka batch consumers?"})
    body = r.json()
    run_id = body["detail"]["runId"]
    print(f"POST /ask -> {r.status_code}  runId={run_id}  error={body['detail']['error']!r}")

    g = client.get(f"/run/{run_id}")
    got = g.json()
    print(f"GET  /run/{{runId}} -> {g.status_code}  status={got['status']}  "
          f"parkedAt={got.get('parkedAt')}  error={got.get('error')!r}")

    rs = client.post(f"/run/{run_id}/resume", json={"approve": True})
    print(f"POST /run/{{runId}}/resume -> {rs.status_code}  {rs.json()['detail']!r}")
    return {"ask": r.status_code, "get": g.status_code, "status": got["status"],
            "parked_at": got.get("parkedAt"), "resume": rs.status_code, "run_id": run_id}


def show_stream() -> list:
    async_srv.graph = failing_graph(async_srv.State)
    async def collect():
        return [ev async for ev in async_srv.stream_tokens("How do I configure Kafka batch consumers?", "demo-failed")]
    events = asyncio.run(collect())
    print("stream_tokens -> " + " ".join(e.strip() for e in events))
    return events


# --- tests -------------------------------------------------------------------
def test_ask_reports_the_failure_with_a_run_id():
    out = show_http()
    assert out["ask"] == 500 and out["run_id"]


def test_the_failed_run_is_readable_and_says_where_it_stopped():
    out = show_http()
    assert out["get"] == 200 and out["status"] == "failed" and out["parked_at"] == ["generate"]


def test_a_failed_run_cannot_be_resumed_like_a_paused_one():
    assert show_http()["resume"] == 409


def test_the_stream_ends_with_failed_then_done():
    events = show_stream()
    assert json.loads(events[-2].split("data:", 1)[1])["failed"] == "RuntimeError"
    assert events[-1].strip() == "data: [DONE]"


if __name__ == "__main__":
    show_http()
    show_stream()
    print()
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
