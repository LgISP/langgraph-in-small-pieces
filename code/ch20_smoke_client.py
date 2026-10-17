"""Chapter 20 - a smoke client: start the server, drive it, check it, stop it.

Every other chapter's file is a closed loop -- `python code/chNN_*.py` runs the
idea and prints what happened. The two serving files cannot be: they are servers,
so they start and wait. This is their missing other half. It launches
`ch20_serving.py` as a child process, drives the two endpoints over real HTTP,
asserts the chapter's claims, and shuts the server down again.

What it checks is the *structure* the chapter promises, never the model's wording
(the Chapter 18 rule, applied to a web layer):

    1. a question the local docs cover: POST /ask returns a runId and an answer
    2. the runId IS the thread_id -- send your own, get the same one back
    3. GET /run/{runId} replays that run out of the checkpoint: same question,
       same answer, status "done"
    4. GET /run/<never-ran> is a 404, not an empty 200
    5. a question they do NOT cover parks at the approval gate: /ask returns a
       runId, status "paused", and no answer at all
    6. the parked run is still there afterwards -- GET says "paused" and repeats
       what it is waiting for
    7. resuming with that runId is the only thing that produces an answer
    8. refusing is also an answer: the run finishes without a web search

It needs a model, like everything else from Part II on: the /ask path runs the
graph, and the graph calls LM Studio. Start LM Studio first (@sec-setup).

Run:  python code/ch20_smoke_client.py
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "ch20_serving.py")
BASE = "http://127.0.0.1:8000"
QUESTION = "How do I configure Kafka batch consumers?"   # local docs cover it
UNCOVERED = "How do I tune Redis connection pools?"       # they do not



def excerpt(text: str, width: int = 140) -> str:
    """First line of a model answer, cut at a word boundary, with '...' if cut.

    Answers are long and often contain code fences; the trace only needs
    enough to see WHAT the model decided to say, not the whole essay.
    """
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) <= width and first == text.strip():
        return first
    cut = first[:width].rsplit(" ", 1)[0] if len(first) > width else first
    return cut.rstrip(" ,;:") + " ..."

def get(path: str) -> tuple[int, dict]:
    """GET, returning (status, body). A 404 is an answer here, not a failure."""
    try:
        with urllib.request.urlopen(BASE + path, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def post(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.status, json.loads(r.read())


def post_status(path: str, payload: dict) -> int:
    """Like post(), but only the status code -- for the check where a 409 is the point."""
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status
    except urllib.error.HTTPError as err:
        return err.code


def wait_for_server(proc: subprocess.Popen, seconds: int = 60) -> None:
    """Poll until the server answers, or the child dies, or we give up."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"the server exited before it was ready (code {proc.returncode}); "
                "run `python code/ch20_serving.py` on its own to see why"
            )
        try:
            # An id that never ran: a 404 proves the app is up and routing.
            get("/run/probe-not-a-real-run")
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"the server did not come up within {seconds}s")


def main() -> int:
    print(f"starting {os.path.basename(SERVER)} ...")
    proc = subprocess.Popen(
        [sys.executable, SERVER],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,      # its own group, so we can stop uvicorn cleanly
    )
    try:
        wait_for_server(proc)
        print(f"server is up on {BASE}\n")

        # 1) One request = one graph execution.
        status, asked = post("/ask", {"question": QUESTION})
        print(f"POST /ask -> {status}  runId={asked['runId']}")
        print(f"  answer: {excerpt(asked['answer'])}")

        # 2) The runId is the caller's thread_id when the caller supplies one.
        mine = "smoke-client-thread"
        _, named = post("/ask", {"question": QUESTION, "thread_id": mine})
        print(f"POST /ask with thread_id={mine!r} -> runId={named['runId']}")

        # 3) The finished run is readable afterwards, out of the checkpoint.
        fetch_status, fetched = get(f"/run/{asked['runId']}")
        print(f"GET  /run/{{runId}} -> {fetch_status}  status={fetched['status']}")
        print(f"  question: {fetched['question']}")

        # 4) An id that never ran is a 404, not an empty 200.
        missing_status, missing = get("/run/never-happened")
        print(f"GET  /run/never-happened -> {missing_status}  {missing.get('detail')!r}")

        print()
        # 5) A question the docs do not cover stops for a human. No answer yet.
        _, parked = post("/ask", {"question": UNCOVERED})
        print(f"POST /ask (uncovered) -> status={parked['status']}  runId={parked['runId']}")
        print(f"  awaiting: {parked['awaiting']['ask']}")

        # 6) The half-finished run is readable, and still says what it wants.
        _, looked = get(f"/run/{parked['runId']}")
        print(f"GET  /run/{{runId}} -> status={looked['status']}  answer={looked['answer']!r}")

        # 7) Only coming back with the runId produces an answer.
        _, approved = post(f"/run/{parked['runId']}/resume", {"approve": True})
        print(f"POST /run/{{runId}}/resume approve=True -> webSearched={approved['webSearched']}")
        print(f"  answer: {excerpt(approved['answer'])}")

        # 8) Refusing finishes the run too, without the web search.
        _, parked2 = post("/ask", {"question": UNCOVERED, "thread_id": "smoke-refused"})
        _, refused = post("/run/smoke-refused/resume", {"approve": False})
        print(f"POST /run/{{runId}}/resume approve=False -> webSearched={refused['webSearched']}")
        print(f"  answer: {excerpt(refused['answer'])}")

        # 9) The double click: two approvals of the same parked run at once. One
        #    runs, the other is refused -- the approved side effect happens once.
        _, parked3 = post("/ask", {"question": UNCOVERED, "thread_id": "smoke-double"})
        statuses: list[int] = []
        def approve_once():
            statuses.append(post_status("/run/smoke-double/resume", {"approve": True}))
        pair = [threading.Thread(target=approve_once) for _ in range(2)]
        for th in pair: th.start()
        for th in pair: th.join()
        print(f"POST /run/{{runId}}/resume x2 concurrently -> {sorted(statuses)}")

        print()
        checks = [
            ("ask returns a runId", bool(asked["runId"])),
            ("ask returns a non-empty answer", bool(asked["answer"].strip())),
            ("the caller's thread_id comes back as the runId", named["runId"] == mine),
            ("the run is readable afterwards", fetch_status == 200
                                               and fetched["runId"] == asked["runId"]),
            ("the stored question is the one we asked", fetched["question"] == QUESTION),
            ("the stored answer is the one we got", fetched["answer"] == asked["answer"]),
            ("a finished run reports done", fetched["status"] == "done"),
            ("an id that never ran is a 404", missing_status == 404),
            ("an uncovered question parks instead of answering",
                parked["status"] == "paused" and "answer" not in parked),
            ("the parked run says what it is waiting for",
                parked["awaiting"]["node"] == "approve_web"),
            ("a parked run still reads as paused, with no answer",
                looked["status"] == "paused" and looked["answer"] is None),
            ("resuming with the runId is what produces the answer",
                bool(approved["answer"].strip())),
            ("approving actually ran the web search", approved["webSearched"] is True),
            ("refusing still answers, without the web search",
                bool(refused["answer"].strip()) and refused["webSearched"] is False),
            ("two concurrent approvals: one runs, one is a 409",
                sorted(statuses) == [200, 409]),
        ]
        failed = [name for name, ok in checks if not ok]
        for name, ok in checks:
            print(f"  {'PASS' if ok else 'FAIL'} {name}")
        print()
        if failed:
            print(f"{len(failed)} check(s) FAILED: {', '.join(failed)}")
            return 1
        print(f"all {len(checks)} checks passed")
        return 0
    finally:
        # uvicorn traps SIGINT; kill the group so no listener is left on 8000.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        print("server stopped.")


if __name__ == "__main__":
    sys.exit(main())
