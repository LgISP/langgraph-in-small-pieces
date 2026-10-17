"""Chapter 19 - Error Handling: how an agent survives failure.

A production agent hits real trouble: the LLM service hiccups, a tool times
out, retrieval falls over. There are three standard responses, and each one is
a different graph shape - not a different try/except style:

  Pattern 1  Automatic retry (RetryPolicy) - for TRANSIENT faults (a network
             blip, a transient 5xx): let LangGraph re-run the node for you.
  Pattern 2  Catch + route to a fallback node - for PERMANENT faults (this call
             will always fail): write the error into State and take a
             degraded branch instead of crashing the whole graph.
  Pattern 3  Graceful degradation - for a NON-critical tool: skip it and answer
             with whatever information you already have.

There is a fourth response - a node crashes and you recover from a checkpoint -
but that was Chapter 6: a checkpoint restores STATE, not the call stack.

Everything here is deterministic. The failures are simulated (a counter, a
raised exception), so this file prints the same thing every run - no LLM.

Run:  python code/ch19_error_handling.py
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from typing import NotRequired, TypedDict

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


# ===========================================================================
# Pattern 1: Automatic retry (RetryPolicy) - for TRANSIENT faults
# ===========================================================================
# Scenario: an external API occasionally blips (a dropped connection, a
# transient 5xx) and simply retrying fixes it. You do NOT hand-write a for-loop
# with sleeps: LangGraph has retry-with-exponential-backoff built in.

_attempt = {"n": 0}  # simulated flaky service: fails twice, succeeds on the third call


class S1(TypedDict):
    result: NotRequired[str]   # written by the node; absent in the initial input


def flaky_api_node(state: S1) -> dict:
    _attempt["n"] += 1
    if _attempt["n"] < 3:
        print(f"      attempt {_attempt['n']} -> raising ConnectionError (simulated network blip)")
        raise ConnectionError("network blip")
    print(f"      attempt {_attempt['n']} -> success")
    return {"result": "call succeeded"}


def demo_retry() -> None:
    print("\n########## Pattern 1: RetryPolicy auto-retries transient faults ##########")
    b = StateGraph(S1)
    b.add_node(
        "api",
        flaky_api_node,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.1,
            backoff_factor=2.0,
            retry_on=(ConnectionError,),
        ),
    )
    b.add_edge(START, "api")
    b.add_edge("api", END)
    graph = b.compile()

    _attempt["n"] = 0
    result = graph.invoke({})
    print(f"  final: {result['result']} (the first two failures were retried away)")


# ===========================================================================
# Pattern 2: Catch + route to a fallback node - for PERMANENT faults
# ===========================================================================
# Scenario: an error that retrying will not fix (bad input, a downstream that is
# permanently down). We must NOT let the whole graph crash. Instead: catch the
# error, write it into State, and let a conditional edge steer to a "fallback"
# node that gives the user a graceful reply.


class S2(TypedDict):
    question: str                  # input, set by the caller
    answer: NotRequired[str]       # written by normal_answer or fallback
    error: NotRequired[str]        # written by risky_node when it fails


def risky_node(state: S2) -> dict:
    try:
        raise ValueError("downstream service permanently unavailable")  # always fails
    except ValueError as e:   # narrow: catch the fault you expect, not every exception
        print(f"      risky_node failed and caught: {e}")
        return {"error": str(e)}


def normal_answer_node(state: S2) -> dict:
    return {"answer": "this is the normal, successful answer"}


def fallback_node(state: S2) -> dict:
    # This demo echoes the raw reason so you can watch the error travel through
    # State. In production the caller gets a generic message and the raw reason
    # stays in logs and traces - never surface a downstream exception to the user.
    return {"answer": f"Sorry, we cannot handle this right now (reason: {state['error']}). Please try again later."}


def route_on_error(state: S2) -> str:
    return "fallback" if state.get("error") else "ok"


def demo_catch_and_route() -> None:
    print("\n########## Pattern 2: catch the error + conditional edge to a fallback ##########")
    b = StateGraph(S2)
    b.add_node("risky", risky_node)
    b.add_node("normal", normal_answer_node)
    b.add_node("fallback", fallback_node)
    b.add_edge(START, "risky")
    b.add_conditional_edges("risky", route_on_error, {"ok": "normal", "fallback": "fallback"})
    b.add_edge("normal", END)
    b.add_edge("fallback", END)
    graph = b.compile()

    result = graph.invoke({"question": "ask me anything"})
    print(f"  user receives: {result['answer']}")
    print("  -> the graph did not crash; it returned a graceful, degraded reply")


# ===========================================================================
# Pattern 2, declaratively: `add_node(error_handler=...)` (LangGraph 1.2)
# ===========================================================================
# The same outcome as Pattern 2 with none of the wiring: no `error` field, no
# router, no conditional edge. The runtime catches what the node raises and calls
# the handler instead. Worth building the long way first -- this is what that
# machinery does for you, and knowing which is why the edge cases below read as
# edge cases rather than surprises.


def raising_node(state: S2) -> dict:
    # No try/except. Pattern 2's `risky_node` had to catch its own failure to put
    # it in `State`; here the runtime is the one that catches.
    raise ValueError("downstream service permanently unavailable")


def fallback_handler(state: S2, error: NodeError) -> dict:
    # A parameter annotated `NodeError` is how the handler asks for the failure:
    # the runtime injects it. `.node` is the name that raised, `.error` the
    # exception. An ordinary node otherwise -- it may return a state update, or a
    # `Command` if it wants to choose where the run goes next.
    print(f"  [handler] {error.node} raised {type(error.error).__name__}")
    return {"answer": f"Sorry, we cannot handle this right now (reason: {error.error})."}


def demo_error_handler() -> None:
    print("\n########## Pattern 2, declaratively: add_node(error_handler=...) ##########")
    b = StateGraph(S2)
    b.add_node("risky", raising_node, error_handler=fallback_handler)
    b.add_edge(START, "risky")
    b.add_edge("risky", END)
    graph = b.compile()

    result = graph.invoke({"question": "ask me anything"})
    print(f"  user receives: {result['answer']}")
    print("  -> same apology as Pattern 2, without the error field or the router")


# ===========================================================================
# Pattern 3: Graceful degradation - a non-critical tool dies, skip it and go on
# ===========================================================================
# Scenario: web_search is a nice-to-have. If it dies it must NOT fail the whole
# Q&A. Catch the failure, mark "web unavailable", and generate the answer from
# the local docs anyway.


class S3(TypedDict):
    question: str                     # input, set by the caller
    docs: NotRequired[list]           # written by local_search
    web_results: NotRequired[list]    # written by web_search (empty when it degrades)
    degraded: NotRequired[bool]       # written by web_search on failure
    answer: NotRequired[str]          # written by generate


def local_search_node(state: S3) -> dict:
    return {"docs": ["local doc: Kafka configuration guide"]}


def web_search_node(state: S3) -> dict:
    try:
        raise TimeoutError("web search API timed out")
    except Exception as e:
        print(f"      web_search failed, degrading gracefully: {e}")
        # Do NOT re-raise: mark the run degraded and return empty results.
        return {"web_results": [], "degraded": True}


def generate_node(state: S3) -> dict:
    sources = state.get("docs", []) + state.get("web_results", [])
    note = "(note: web retrieval is unavailable; answering from local docs only) " if state.get("degraded") else ""
    return {"answer": f"{note}answer based on {len(sources)} source(s)"}


def demo_graceful() -> None:
    print("\n########## Pattern 3: graceful degradation (a non-critical tool dies, keep going) ##########")
    b = StateGraph(S3)
    b.add_node("local", local_search_node)
    b.add_node("web", web_search_node)
    b.add_node("generate", generate_node)
    b.add_edge(START, "local")
    b.add_edge("local", "web")
    b.add_edge("web", "generate")
    b.add_edge("generate", END)
    graph = b.compile()

    result = graph.invoke({"question": "How do I configure Kafka?"})
    print(f"  answer: {result['answer']}")
    print("  -> web died, but the local docs still produced an answer (degraded, not failed)")


if __name__ == "__main__":
    demo_retry()
    demo_catch_and_route()
    demo_error_handler()
    demo_graceful()
