"""Chapter 13 Challenge — Bronze: where a @task is allowed to run.

`@task` and `@entrypoint` look like ordinary decorators, but they draw a boundary:
a task is a unit of work the RUNTIME schedules and checkpoints, so it needs a live
run to belong to. Call one from module scope and there is no run to attach it to.

The error says so almost literally:

    RuntimeError: Called get_config outside of a runnable context

`get_config` is how the task finds the run it is part of — the thread, the
checkpoint, the position in the entrypoint's body that keys its cached result
(@sec-functional-api). Outside an entrypoint none of that exists.

It raises at the CALL, not at `.result()`: no future is ever handed back.

There is a second way to get the boundary wrong, and it fails differently. An
`@entrypoint` is a compiled `Pregel` object, not a function — call it like one
(`doc_bot("kafka")`) and you get `TypeError: 'Pregel' object is not callable`. You
drive it with `.invoke(...)`, same as any compiled graph. Two mistakes, two
distinct errors: a task needs a run to live in (RuntimeError), and an entrypoint is
invoked, not called (TypeError).

Offline by design: nothing here calls a model.

Run:  python code/answers/ch13_task_boundary.py   (see answers/README.md)
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task


@task
def search_docs(question: str) -> list[str]:
    """The same shape as the chapter's task: a unit of work the runtime owns."""
    return [f"doc about {question}"]


@entrypoint(checkpointer=InMemorySaver())
def doc_bot(question: str) -> dict:
    docs = search_docs(question).result()      # legal: we are inside a run
    return {"question": question, "docs": docs}


def call_task_from_module_scope() -> str:
    """Reproduce the failure and hand back its message.

    Note where it raises: on the CALL. There is no future to wait on, so
    `.result()` is never reached — the requirement belongs to invoking the task,
    not to collecting its result.
    """
    try:
        search_docs("kafka")
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("expected a RuntimeError outside an entrypoint")


def call_entrypoint_directly() -> str:
    """The other mistake: call the entrypoint like a plain function.

    A compiled `@entrypoint` is a `Pregel`, not a callable function, so `doc_bot(x)`
    raises `TypeError` — a different failure from the task's RuntimeError. Drive it
    with `.invoke(...)` instead.
    """
    try:
        doc_bot("kafka")
    except TypeError as exc:
        return str(exc)
    raise AssertionError("expected a TypeError from calling the entrypoint directly")


# --- tests -------------------------------------------------------------------
def test_task_outside_an_entrypoint_raises():
    msg = call_task_from_module_scope()
    # The message names what is MISSING (a runnable context), not what you did.
    assert "outside of a runnable context" in msg


def test_the_same_task_is_fine_inside_the_entrypoint():
    out = doc_bot.invoke("kafka", {"configurable": {"thread_id": "b1"}})
    assert out["docs"] == ["doc about kafka"]


def test_calling_the_entrypoint_directly_raises_typeerror():
    # The contrast the Bronze asks for: the entrypoint is a Pregel, not a function.
    msg = call_entrypoint_directly()
    assert "not callable" in msg


def test_the_entrypoint_runs_via_invoke():
    # The right way to drive it: `.invoke(...)`, from module scope, no wrapper —
    # the entrypoint is the thing that creates the runnable context.
    out = doc_bot.invoke("redis", {"configurable": {"thread_id": "b2"}})
    assert out["question"] == "redis"


if __name__ == "__main__":
    print("task called from module scope ->")
    print("   RuntimeError:", call_task_from_module_scope())
    print("entrypoint called directly like a function ->")
    print("   TypeError:", call_entrypoint_directly())
    print("same task inside the entrypoint, driven with .invoke() ->")
    print("  ", doc_bot.invoke("kafka", {"configurable": {"thread_id": "demo"}}))

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
