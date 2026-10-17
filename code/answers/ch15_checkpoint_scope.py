"""Chapter 15 Challenge — Bronze: the checkpoint's scope is one conversation.

Tell the bot a fact about yourself on one `thread_id`, ask a question that depends
on it, and watch it honour the fact. Then ask the *same question* on a brand-new
`thread_id` and watch it know nothing.

That is not a bug and it is not a small memory. It is the checkpointer's contract:
a checkpoint is keyed by thread, so it can only ever hold what happened in that
thread. Ch24 shows the shape directly -- `saver.storage[thread_id][...]`, thread
first. Nothing about "the user" appears anywhere in that key, so nothing about the
user can survive a change of thread.

Which is exactly the gap the Store is built to fill, and the reason the chapter
introduces both: they are keyed by different things and therefore answer different
questions.

    checkpointer   keyed by thread   -> "what did we say in THIS conversation?"
    store          keyed by namespace -> "what do I know about THIS USER, ever?"

A bot with only a checkpointer forgets you between sessions. A bot with only a
store remembers your preferences and loses your place mid-sentence. The pairing
is not belt-and-braces; each one is unable to do the other's job.

Needs LM Studio at :1234.
Run:  python code/answers/ch15_checkpoint_scope.py   (see answers/README.md)
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LM_BASE = "http://localhost:1234/v1"
FACT = "For any code you show me, I only write Rust. Never suggest another language."
QUESTION = "How do I read a file line by line?"


class Skip(Exception):
    pass


def server_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{LM_BASE}/models", timeout=2):
            return True
    except urllib.error.HTTPError:
        return True        # a status code is a server: a hosted endpoint answers 401 to a key-less probe
    except (urllib.error.URLError, OSError):
        return False


def say(graph, thread: str, text: str) -> str:
    out = graph.invoke({"messages": [HumanMessage(content=text)]},
                       config={"configurable": {"thread_id": thread}})
    return out["messages"][-1].content


_CACHE: dict = {}


def cached() -> dict:
    if not _CACHE:
        if not server_is_up():
            raise Skip(f"no model server at {LM_BASE}")
        import ch15_memory as ch15
        graph = ch15.build_chat_graph()

        _CACHE["ack"] = say(graph, "conv-rust", FACT)
        _CACHE["with_context"] = say(graph, "conv-rust", QUESTION)
        _CACHE["fresh_thread"] = say(graph, "conv-brand-new", QUESTION)

        saver = graph.checkpointer
        _CACHE["threads"] = sorted(saver.storage)
        _CACHE["graph"] = graph
    return _CACHE


def mentions_rust(text: str) -> bool:
    # The fact is honoured when the answer is Rust -- whether it says the word or
    # just writes the code (`std::fs::File`, `fn main`). A run that answered in
    # Rust without naming it used to count as forgetting.
    t = text.lower()
    return "rust" in t or "std::" in t or "fn main" in t or "cargo" in t


def demo_checkpoint_scope():
    if not server_is_up():
        print(f"no model server at {LM_BASE} -- skipping")
        return
    c = cached()
    print(f"[conv-rust]      user: {FACT}")
    print(f"[conv-rust]      bot : {c['ack'][:100]}\n")
    print(f"[conv-rust]      user: {QUESTION}")
    print(f"[conv-rust]      bot : {c['with_context'][:120]}")
    print(f"   mentions Rust? {mentions_rust(c['with_context'])}\n")
    print(f"[conv-brand-new] user: {QUESTION}     <- same question, new thread")
    print(f"[conv-brand-new] bot : {c['fresh_thread'][:120]}")
    print(f"   mentions Rust? {mentions_rust(c['fresh_thread'])}\n")
    print(f"threads in the checkpointer: {c['threads']}")
    print("\nTwo cabinets, no shared drawer. The fact lives in one conversation")
    print("because a checkpoint is keyed by thread and by nothing else -- which is")
    print("the gap the Store exists to fill.")


# --- tests -------------------------------------------------------------------
# A real model, so the assertions are about the PATH -- which thread saw which
# message -- never the model's word choice. A fresh model COULD volunteer Rust on
# its own; that would not mean the fact crossed threads, so the proof is the
# history each thread carries, not whether "rust" appears in a reply.

def test_the_fact_survives_within_its_own_thread():
    # Same-thread continuity: the fact is in this thread's history, so the model
    # can honour it. (In-context instruction-following, model-dependent wording.)
    assert mentions_rust(cached()["with_context"])


def test_the_new_thread_was_never_told_the_fact():
    # The robust "knows nothing" proof: the fact never entered this thread's
    # history. Whether the model happens to mention Rust anyway is beside the point.
    c = cached()
    state = c["graph"].get_state({"configurable": {"thread_id": "conv-brand-new"}})
    texts = [m.content for m in state.values["messages"]]
    assert FACT not in texts


def test_the_two_threads_are_separate_keys_in_the_checkpointer():
    assert cached()["threads"] == ["conv-brand-new", "conv-rust"]


def test_the_new_thread_saw_exactly_one_user_message():
    """The mechanism, not the wording: the fresh thread's history is one turn, so
    the model could not have known -- it was never told."""
    c = cached()
    state = c["graph"].get_state({"configurable": {"thread_id": "conv-brand-new"}})
    humans = [m for m in state.values["messages"] if m.type == "human"]
    assert len(humans) == 1 and humans[0].content == QUESTION


def test_the_old_thread_still_holds_the_whole_conversation():
    c = cached()
    state = c["graph"].get_state({"configurable": {"thread_id": "conv-rust"}})
    humans = [m.content for m in state.values["messages"] if m.type == "human"]
    assert humans == [FACT, QUESTION], humans


def test_nothing_in_the_checkpoint_key_mentions_a_user():
    """Why it cannot be otherwise: the key is the thread. There is no field the
    fact could have been filed under to make it cross."""
    saver = cached()["graph"].checkpointer
    for thread, namespaces in saver.storage.items():
        assert thread.startswith("conv-")
        assert list(namespaces) == [""], namespaces


if __name__ == "__main__":
    try:
        demo_checkpoint_scope()
    except Skip as e:
        print(f"skipping: {e}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    passed = skipped = failed = 0
    print()
    for t in _tests:
        try:
            t()
            print("PASS", t.__name__); passed += 1
        except Skip as e:
            print("SKIP", t.__name__, f"({e})"); skipped += 1
        except AssertionError as e:
            print("FAIL", t.__name__, f"({e})"); failed += 1
    # Report the REAL tally: skips are not passes. All-skipped (no server) prints
    # "0 passed, 6 skipped", never "all passed".
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    if failed:
        sys.exit(1)
