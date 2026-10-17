"""Chapter 13 Challenge — Silver: port a Graph feature into the function.

Give the `generate` step an *output-based* retry: if the model returns an empty
or suspiciously short answer, call it again, up to twice. This is a real
difference from the Graph API — a `RetryPolicy` on a node only fires when the node
*raises*, never on a short-but-successful return. In the Functional API a plain
`for` loop inside the entrypoint calls the `generate` task again, and each call is
a fresh task invocation that genuinely re-runs rather than replaying the last
result.

Then it closes the loop with the chapter's Gotcha and makes the cache do the
proving. `generate` calls the book's REAL local model (Qwen3.8 behind LM
Studio) on every attempt; the failure injection is a SWITCH (`INJECT_FAILURE`) so
both paths are reachable — on, attempt 0 is truncated to a stub (forcing one
retry) and attempt 1 returns whole; off, a good first answer stops the loop after
a single call. The model call is real; only the truncation is code. A module-level
counter bumps each time the task actually runs. An `interrupt()` sits *after* the
retry loop, so the pause lands with both completed attempts already checkpointed —
and it is a REAL gate: we read the decision and withhold the answer on "no". On
resume the counter must NOT increase — proof that both attempts replayed from cache
instead of re-running (no model call happens on resume either: a replayed task
never re-enters its body). That is the cache-replay claim made checkable: a real
number you can assert on, not a trace you eyeball.

The last test checks the two INTENDED routes from Chapter 10 — Kafka to the docs,
Python to the web — here navigated by an `if` instead of a model router. It asserts
the Functional version's routes against those intended paths; it does not run the
Ch10 graph, so it is not a two-implementation comparison.

Run:  python code/answers/ch13_functional_retry.py   (see answers/README.md)
"""

from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# The book's real local model — the chapter's local-server setup.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ],
    "redis": [
        "Redis Manual: set the connection pool via spring.data.redis.*.",
    ],
    # Deliberately no "python" topic — that question misses and falls back to the web.
}

MIN_ANSWER_LEN = 20   # answers shorter than this are "suspiciously short" -> retry

# Fault injection is a SWITCH, so both paths the Silver asks about are reachable:
# with it on, attempt 0 is truncated to force one retry; with it off, a good first
# answer stops the loop after a single call. Tests flip it per scenario.
INJECT_FAILURE = True

# Module-level call counters. `generate` is the one the cache-replay proof watches;
# `search_docs`/`web_search` record the route taken, to prove the ch10 path match.
CALLS = {"search_docs": 0, "web_search": 0, "generate": 0}


def reset_counts() -> None:
    for k in CALLS:
        CALLS[k] = 0


# --- the three tasks (search + web are the chapter's stubs) -------------------
@task
def search_docs(question: str) -> list[str]:
    CALLS["search_docs"] += 1
    q = question.lower()
    hits: list[str] = []
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in q:
            hits.extend(docs)
    print(f"   [search_docs] {len(hits)} hit(s)")
    return hits


@task
def web_search(question: str) -> list[str]:
    CALLS["web_search"] += 1
    print("   [web_search] falling back to a web search")
    return [f"[web] community thread about “{question}”"]


@task
def generate(question: str, docs: list[str], web: list[str], attempt: int) -> str:
    # A REAL model call on every attempt. The two-phase behavior the retry needs
    # is injected around it: attempt 0 truncates the real answer to a stub (as if
    # the model had trailed off), attempt 1 lets it through whole. The counter
    # bumps on every REAL run — so the resume test can prove that a replay from
    # cache does NOT bump it (a replayed task never re-enters this body at all).
    CALLS["generate"] += 1
    sources = docs + web
    sources_text = "\n".join(f"- {s}" for s in sources) or "(no sources)"
    prompt = (f"Answer the user's question concisely, using only the sources below. "
              f"Do not invent anything.\n{sources_text}\n\nQuestion: {question}")
    answer = llm.invoke(prompt).content.strip()
    if attempt == 0 and INJECT_FAILURE:
        answer = answer[:3]                             # injected failure: a stub
    print(f"   [generate] attempt {attempt} -> {len(answer)} char(s)")
    return answer


# The flow: search -> (maybe) web -> retry generate up to twice -> interrupt -> return.
# The interrupt lands AFTER the loop, so the good answer (and both attempts) are
# already checkpointed when we pause.
@entrypoint(checkpointer=InMemorySaver())
def doc_bot_retry(question: str) -> str:
    docs = search_docs(question).result()
    web = web_search(question).result() if not docs else []

    answer = ""
    for attempt in range(3):                    # 1 initial call + up to 2 retries
        answer = generate(question, docs, web, attempt).result()
        if len(answer) >= MIN_ANSWER_LEN:       # good enough — stop retrying
            break

    # Gate AFTER the retry loop: the completed attempts are checkpointed before we
    # pause. This is a REAL approval — we read the decision and honor a rejection,
    # rather than prompting "yes/no" and ignoring the answer.
    decision = interrupt(f"Answer ready ({len(answer)} char(s)). Approve? (yes/no)")
    if str(decision).strip().lower() not in ("yes", "y", "approve"):
        return "(answer withheld: not approved)"
    return answer


# --- helpers -----------------------------------------------------------------
def run_to_interrupt(question: str, thread_id: str):
    """Invoke up to the post-retry interrupt; return (paused_result, calls_snapshot)."""
    config = {"configurable": {"thread_id": thread_id}}
    paused = doc_bot_retry.invoke(question, config)
    return paused, dict(CALLS), config


# --- tests -------------------------------------------------------------------
def test_output_retry_recovers_from_short_answer():
    # A short first answer forces exactly one retry; the second attempt is good.
    global INJECT_FAILURE
    INJECT_FAILURE = True
    reset_counts()
    paused, at_pause, _ = run_to_interrupt(
        "How do I configure Kafka batch consumers?", "retry-recover")
    assert "__interrupt__" in paused                    # paused at the post-loop gate
    assert at_pause["generate"] == 2                    # ran twice: short, then good


def test_good_first_answer_calls_generate_once():
    # Silver step 1: with the injected failure OFF, a good first answer stops the
    # loop after a single call — no retry.
    global INJECT_FAILURE
    INJECT_FAILURE = False
    try:
        reset_counts()
        _, at_pause, _ = run_to_interrupt(
            "How do I configure Kafka batch consumers?", "retry-once")
        assert at_pause["generate"] == 1                # one call, no retry
    finally:
        INJECT_FAILURE = True


def test_rejection_withholds_the_answer():
    # The gate is real: resuming with "no" returns the withheld sentinel, not the
    # answer. (Both attempts still replayed from cache — the pause is post-loop.)
    global INJECT_FAILURE
    INJECT_FAILURE = True
    reset_counts()
    _, _, config = run_to_interrupt(
        "How do I configure Kafka batch consumers?", "retry-reject")
    result = doc_bot_retry.invoke(Command(resume="no"), config)
    assert result == "(answer withheld: not approved)"


def test_cached_attempts_do_not_rerun_on_resume():
    # THE cache-replay proof: resume replays the whole body, but every completed
    # `generate` attempt comes from cache, so the counter must not move.
    reset_counts()
    paused, at_pause, config = run_to_interrupt(
        "How do I configure Kafka batch consumers?", "retry-cache")
    calls_at_pause = at_pause["generate"]               # == 2
    answer = doc_bot_retry.invoke(Command(resume="yes"), config)
    assert CALLS["generate"] == calls_at_pause          # NOT 4 — both attempts replayed
    assert len(answer) >= MIN_ANSWER_LEN                # the good answer came back


def test_paths_match_the_intended_routes():
    # The two INTENDED routes from Chapter 10, here navigated by the `if` in the
    # entrypoint rather than a model router: Kafka hits the docs (no web); Python
    # misses and falls back to the web. This asserts the Functional version's routes
    # against those intended paths — it does not import or run the Ch10 graph, so it
    # is not a two-implementation comparison.
    reset_counts()
    run_to_interrupt("How do I configure Kafka batch consumers?", "path-kafka")
    assert CALLS["search_docs"] == 1 and CALLS["web_search"] == 0   # docs path

    reset_counts()
    run_to_interrupt("How do I write a Python decorator?", "path-python")
    assert CALLS["search_docs"] == 1 and CALLS["web_search"] == 1   # web-fallback path


if __name__ == "__main__":
    reset_counts()
    paused, at_pause, config = run_to_interrupt(
        "How do I configure Kafka batch consumers?", "demo")
    print("paused:", paused["__interrupt__"][0].value)
    print(f"generate ran {at_pause['generate']}x before the pause (short, then good)")
    answer = doc_bot_retry.invoke(Command(resume="yes"), config)
    print(f"generate ran {CALLS['generate']}x after resume "
          f"(unchanged = both attempts replayed from cache)")
    print("--- answer ---")
    print(answer)

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
