"""Ch21: one run at a time per thread, on the async server.

The regression these pin was real and silent. Before the claim existed, two
concurrent `POST /run/{id}/resume` on one paused run both returned HTTP 200 and
both generated -- measured with start times equal to the millisecond. The gated
side effect in the chapter is a stub, so nothing in the response, the log or the
final checkpoint showed it. That is what makes it worth a test: the failure mode
is invisible in the example and expensive in the thing a reader builds from it.

No model and no server here: the graph is replaced by a stub whose `ainvoke`
waits, which is what forces the two requests to overlap. Each test drives its
own loop with `asyncio.run` rather than reaching for pytest-asyncio -- one
regression is not worth a new pin in the book's frozen requirements.

    python -m pytest code/test_ch20_concurrent_resume.py -q
"""
import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ch20_serving_async as mod  # noqa: E402


class _Snapshot:
    """What the endpoint reads off a run parked at the approval gate."""
    created_at = "2026-01-01T00:00:00Z"
    next = ("web_search",)
    interrupts = ({"ask": "search the web?"},)


class _SlowGraph:
    """A graph that is still resuming when the second request arrives."""

    def __init__(self):
        self.resumes = 0
        self.started = asyncio.Event()

    async def aget_state(self, config):
        await asyncio.sleep(0)            # the yield point the race needs
        return _Snapshot()

    async def ainvoke(self, command, config):
        self.resumes += 1
        self.started.set()
        await asyncio.sleep(0.05)         # long enough for the other to arrive
        return {"answer": "done", "web_results": ["w"]}


@pytest.fixture
def stub(monkeypatch):
    g = _SlowGraph()
    monkeypatch.setattr(mod, "graph", g)
    mod._in_flight.clear()
    yield g
    mod._in_flight.clear()


async def _resume(run_id):
    """One request, as its outcome: the body, or the HTTP status it refused with."""
    try:
        return await mod.resume(run_id, mod.ResumeRequest(approve=True))
    except HTTPException as e:
        return e.status_code


def test_two_concurrent_resumes_produce_one_run(stub):
    asyncio.run(_test_two_concurrent_resumes_produce_one_run(stub))


async def _test_two_concurrent_resumes_produce_one_run(stub):
    a, b = await asyncio.gather(_resume("t"), _resume("t"))
    outcomes = [a, b]
    assert 409 in outcomes, "both resumes were accepted: the claim is not holding"
    assert sum(isinstance(o, dict) for o in outcomes) == 1
    # The one that matters. An approval the human gave once must not resume twice.
    assert stub.resumes == 1


def test_the_claim_is_released_when_the_run_finishes(stub):
    asyncio.run(_test_the_claim_is_released_when_the_run_finishes(stub))


async def _test_the_claim_is_released_when_the_run_finishes(stub):
    assert isinstance(await _resume("t"), dict)
    assert "t" not in mod._in_flight
    assert isinstance(await _resume("t"), dict)      # sequential is fine
    assert stub.resumes == 2


def test_a_refused_duplicate_does_not_release_the_holder_s_claim(stub):
    """The 409 path must not run the context manager's `finally` for someone else."""
    asyncio.run(_test_a_refused_duplicate_does_not_release_the_holder_s_claim(stub))


async def _test_a_refused_duplicate_does_not_release_the_holder_s_claim(stub):
    async def late():
        await stub.started.wait()
        return await _resume("t")
    first, second = await asyncio.gather(_resume("t"), late())
    assert second == 409
    assert isinstance(first, dict)
    assert "t" not in mod._in_flight                 # released by the holder, once


def test_other_threads_are_not_blocked(stub):
    """The claim is per run_id. Two different conversations still run together."""
    asyncio.run(_test_other_threads_are_not_blocked(stub))


async def _test_other_threads_are_not_blocked(stub):
    a, b = await asyncio.gather(_resume("one"), _resume("two"))
    assert isinstance(a, dict) and isinstance(b, dict)
    assert stub.resumes == 2


def test_the_claim_survives_a_failing_run(stub):
    """A node that raises must still give the thread back, or one failure locks
    the conversation for the life of the process."""
    asyncio.run(_test_the_claim_survives_a_failing_run(stub))


async def _test_the_claim_survives_a_failing_run(stub):
    async def boom(command, config):
        raise RuntimeError("node failed")
    stub.ainvoke = boom
    with pytest.raises(RuntimeError):
        await mod.resume("t", mod.ResumeRequest(approve=True))
    assert "t" not in mod._in_flight
