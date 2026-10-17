"""Chapter 20 Challenge - Gold: a checkpointer every replica shares.

`InMemorySaver` keeps runs in one process's heap, so a second replica behind the
load balancer returns 404 for a run the first replica executed (the chapter's
scale-out Gotcha). A shared, networked checkpointer fixes it. This file proves the
difference with Redis: two savers built against the SAME Redis - the second
standing in for a second replica - both see the run, while two separate
`InMemorySaver`s do not.

The graph here has no LLM: the challenge is about the checkpointer, not the model,
so a stub `answer` node keeps it deterministic and offline (only Redis is external).

Needs a Redis with the RediSearch + RedisJSON modules (Redis 8+ or Redis Stack):
    docker run -p 6379:6379 redis:8
    pip install langgraph-checkpoint-redis

Run:  ../.venv1x/bin/python code/answers/ch20_shared_checkpointer_redis.py
"""

import os
from typing import NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, START, StateGraph

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


class State(TypedDict):
    question: str
    answer: NotRequired[str]


def answer_node(state: State) -> dict:
    # No LLM on purpose: this challenge is about where the checkpoint lives, not
    # about generation, so keep the node deterministic.
    return {"answer": f"answer to: {state['question']}"}


def build(checkpointer):
    """The doc-bot's shape, reduced to the one thing this challenge tests: a graph
    that writes an answer and checkpoints it under the request's thread_id."""
    b = StateGraph(State)
    b.add_node("answer", answer_node)
    b.add_edge(START, "answer")
    b.add_edge("answer", END)
    return b.compile(checkpointer=checkpointer)


def run_on(graph, question, thread_id):
    """A replica handling POST /ask: run the graph, checkpoint under thread_id."""
    return graph.invoke({"question": question}, {"configurable": {"thread_id": thread_id}})


def read_from(graph, thread_id):
    """A replica handling GET /run/{id}: the latest snapshot, or None if this
    checkpointer has never heard of the thread (the 404 case)."""
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    return snap.values.get("answer") if snap.values else None


def redis_reachable() -> bool:
    try:
        with RedisSaver.from_conn_string(REDIS_URL) as saver:
            saver.setup()
        return True
    except Exception:
        return False


# --- tests -------------------------------------------------------------------
def test_two_in_memory_savers_do_not_share():
    # Two replicas, each with its own InMemorySaver: the run written by A is
    # invisible to B. This is the scale-out 404.
    replica_a = build(InMemorySaver())
    replica_b = build(InMemorySaver())
    run_on(replica_a, "How do I configure Kafka batch consumers?", "demo")
    assert read_from(replica_a, "demo") is not None        # A wrote it
    assert read_from(replica_b, "demo") is None            # B never saw it -> 404


def test_shared_redis_saver_serves_the_other_replica():
    # Two savers against the SAME Redis: B answers for a run only A executed.
    with RedisSaver.from_conn_string(REDIS_URL) as saver_a:
        saver_a.setup()
        replica_a = build(saver_a)
        run_on(replica_a, "How do I configure Kafka batch consumers?", "demo-redis")

    # A fresh saver + fresh graph = a second replica that shares nothing but Redis.
    with RedisSaver.from_conn_string(REDIS_URL) as saver_b:
        saver_b.setup()
        replica_b = build(saver_b)
        answer = read_from(replica_b, "demo-redis")

    assert answer == "answer to: How do I configure Kafka batch consumers?"


if __name__ == "__main__":
    print("two InMemorySavers (separate heaps):")
    a, b = build(InMemorySaver()), build(InMemorySaver())
    run_on(a, "How do I configure Kafka batch consumers?", "demo")
    print(f"  replica A GET /run/demo -> {read_from(a, 'demo')!r}")
    print(f"  replica B GET /run/demo -> {read_from(b, 'demo')!r}   <- 404 in the server")

    tests = [test_two_in_memory_savers_do_not_share]
    if redis_reachable():
        print(f"\none shared Redis at {REDIS_URL}:")
        with RedisSaver.from_conn_string(REDIS_URL) as sa:
            sa.setup()
            run_on(build(sa), "How do I configure Kafka batch consumers?", "demo-redis")
        with RedisSaver.from_conn_string(REDIS_URL) as sb:
            sb.setup()
            print(f"  replica A wrote it, replica B GET /run/demo-redis -> "
                  f"{read_from(build(sb), 'demo-redis')!r}")
        tests.append(test_shared_redis_saver_serves_the_other_replica)
    else:
        print(f"\nSKIP Redis tests: no Redis at {REDIS_URL}. Start one with "
              f"`docker run -p 6379:6379 redis:8`.")

    print()
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(tests)} tests passed"
          + ("" if len(tests) == 2 else " (Redis test skipped)"))
