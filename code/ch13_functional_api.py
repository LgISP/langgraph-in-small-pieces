"""Chapter 13 — The Functional API: @entrypoint / @task.

The same doc-bot as Chapter 10 (search -> decide -> generate), written with plain
Python control flow instead of a StateGraph. Two decorators do the work:

    @task        wraps a unit of work; in a checkpointed workflow its return value
                 is stored with the checkpoint (so it must be serializable; the
                 arguments need not be), and once it has completed a deterministic replay
                 reuses the cached result instead of re-running. That dedupes a side
                 effect on resume, but it is NOT exactly-once (a crash before the
                 result is checkpointed re-runs it), so external effects must stay
                 idempotent.
    @entrypoint  marks the start of a flow; its body is ordinary if/for/function
                 calls, and the runtime checkpoints every step for you.

No StateGraph, no nodes, no edges, no router, no recursion_limit — the routing
Chapter 10 built as an `agent` node plus a conditional edge is, here, one `if`.

Run:  python code/ch13_functional_api.py
"""

from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task          # the two decorators live here
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",  # your local server, not the OpenAI cloud
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",      # a capable open model that follows instructions
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch "
        "to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ],
    "redis": [
        "Redis Manual: set the connection pool via spring.data.redis.*.",
    ],
    # Note: deliberately no "python" topic — used below to show a miss.
}


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


# Each unit of work is a @task. Its body is ordinary Python, but call it only from
# inside an @entrypoint (or a graph node), where the call returns a future; .result()
# blocks for the value. Calling it anywhere else raises RuntimeError. In a checkpointed
# workflow its return value is stored with the checkpoint, so that must be serializable
# (a list, a str — fine); its inputs are not stored, so they need not be. Once completed,
# the task replays from cache on resume instead of running again.
@task
def search_docs(question: str) -> list[str]:
    q = question.lower()
    hits: list[str] = []
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in q:
            hits.extend(docs)
    print(f"   [search_docs] {len(hits)} hit(s)")   # a side effect — deduped on replay
    return hits                                      # may be empty — a real answer too


@task
def web_search(question: str) -> list[str]:
    print("   [web_search] falling back to a web search")
    return [f"[web] community thread about “{question}”"]


@task
def generate(question: str, docs: list[str], web: list[str]) -> str:
    sources = docs + web
    sources_text = "\n".join(f"- {s}" for s in sources) or "(no sources)"
    prompt = f"""
        Answer the user's question concisely, using only the sources
        below. Do not invent anything.

        Question: {question}

        Sources:
        {sources_text}

        Answer:"""
    answer = llm.invoke(prompt).content.strip()      # a real model call
    print("   [generate] the model wrote an answer")
    return answer


@entrypoint(checkpointer=InMemorySaver())
def doc_bot_session(question: str, *, previous: list[str] | None = None):
    """The same flow, but the thread remembers the questions rather than the answer.

    `previous` is whatever the last run on this thread_id saved -- None on the
    first. Returning `entrypoint.final` splits the two audiences: `value` goes to
    the caller, `save` goes to the checkpoint and comes back as `previous`.
    """
    print(f"   [session] previous = {previous}")
    asked = (previous or []) + [question]
    docs = search_docs(question).result()
    answer = generate(question, docs, []).result()
    return entrypoint.final(value=answer, save=asked)


# The whole flow is an @entrypoint whose body is plain Python. Compare Chapter 10:
# an `agent` node, a `route_after_agent` router, three conditional edges, two
# back-edges, and a recursion_limit fuse expressed this same plan — search first, fall
# back to the web on a miss, then generate. Here it is three task calls and one `if`.
@entrypoint(checkpointer=InMemorySaver())
def doc_bot(question: str) -> str:
    docs = search_docs(question).result()
    web = web_search(question).result() if not docs else []
    return generate(question, docs, web).result()


# An approval gate is the same interrupt()/Command from Chapter 7, with no graph. On resume the body replays from the top, but
# search_docs returns its cached result (it does NOT print again) — only generate,
# which had not run before the pause, runs for the first time.
@entrypoint(checkpointer=InMemorySaver())
def doc_bot_with_approval(question: str) -> str:
    docs = search_docs(question).result()
    web = web_search(question).result() if not docs else []

    approved = interrupt(f"Generate an answer from {len(docs + web)} source(s)? (yes/no)")
    if approved != "yes":
        return "Declined by a human — no answer generated."

    return generate(question, docs, web).result()


if __name__ == "__main__":
    # Demo A — two questions take the two paths, as in Chapter 10. Each
    # question is its own thread_id, the thread or timeline identifier from Chapter 6.
    for i, q in enumerate(
        ["How do I configure Kafka batch consumers?",
         "How do I write a Python decorator?"],
        start=1,
    ):
        print(f"\n===== question: {q} =====")
        config = {"configurable": {"thread_id": f"a{i}"}}
        answer = doc_bot.invoke(q, config)   # no __interrupt__ -> returns the answer
        print("--- answer ---")
        print(excerpt(answer))               # first line only; the full text is in answer

    # Demo B — the same flow with a human-approval gate, to show HITL + resume, and
    # to prove task caching: after resume, search_docs does NOT print again.
    print("\n\n===== with an approval gate =====")
    q = "How do I configure Kafka batch consumers?"
    config = {"configurable": {"thread_id": "b1"}}
    paused = doc_bot_with_approval.invoke(q, config)
    print("paused, waiting for a human:", paused["__interrupt__"][0].value)
    print("--- human approves; resuming (search_docs is silent = cache hit) ---")
    answer = doc_bot_with_approval.invoke(Command(resume="yes"), config)
    print("--- answer ---")
    print(excerpt(answer))

    # Demo D — what to return vs what to remember. A `previous` parameter asks the
    # checkpointer for whatever the last run on this thread SAVED, which by default
    # is the return value. `entrypoint.final` is how the two stop being the same
    # thing: the caller gets `value`, the next turn gets `save` as `previous`.
    print("\n\n===== entrypoint.final: return one thing, remember another =====")
    config = {"configurable": {"thread_id": "f1"}}
    for q in ("How do I configure Kafka batch consumers?",
              "And how do I handle a failure in one?",
              "What about max-poll-records?"):
        answer = doc_bot_session.invoke(q, config)
        print(f"  returned: {excerpt(answer)}")
    # Watch the `previous` line above grow while `returned` never mentions it. The
    # saved value is not the state you can read back: `get_state(config).values` is
    # the RETURN value, `answer`. `save` reaches exactly one place -- the `previous`
    # of the next call on this thread.
    print(f"  get_state().values is still just the answer: "
          f"{excerpt(doc_bot_session.get_state(config).values)}")


    # Demo C — streaming: one chunk per task as it completes, keyed by the task's
    # name, then a final chunk keyed by the entrypoint. A fresh thread_id, because a
    # replayed thread would serve every task from cache and stay silent.
    print("\n\n===== streaming the steps =====")
    config = {"configurable": {"thread_id": "s1"}}
    for chunk in doc_bot.stream("How do I configure Kafka batch consumers?", config):
        # each chunk is {name: value}; shorten the model's prose, keep the lists whole
        print({name: excerpt(v) if isinstance(v, str) else v for name, v in chunk.items()})
