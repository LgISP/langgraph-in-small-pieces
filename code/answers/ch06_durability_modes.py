"""Chapter 6 Challenge -- Gold: when does a checkpoint actually land?

`invoke` takes an optional `durability` argument that decides *when* the
checkpointer writes. Three modes; the way to compare them is to run the same
graph under each and count what was saved:

    "sync"   persist before the next superstep starts   (safest, slowest)
    "async"  persist while the next superstep runs      (the default)
    "exit"   persist once, when the run exits           (fastest, thinnest trail)

We use the chapter's loop graph (agent -> search_docs -> agent -> web_search ->
agent -> generate) with quiet stub nodes, run it on a fresh thread per mode, and
print how many snapshots `get_state_history` holds afterwards. A fourth run
stages the chapter's crash in `generate` under `"exit"`: a Python exception is
not a kill, the runtime still flushes a checkpoint on the way out, so the run is
parked at `generate` as in the chapter. Only a hard process kill (kill -9, OOM,
a power cut) leaves nothing, and that cannot be shown in-process.

Pre-1.0 code passes `checkpoint_during=True/False` instead; it still works but
emits `LangGraphDeprecatedSinceV10` and is slated for removal in 2.0.

Run:  python code/answers/ch06_durability_modes.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str
    route: NotRequired[str]
    docs: NotRequired[list[str]]
    web_results: NotRequired[list[str]]
    iterations: NotRequired[int]
    answer: NotRequired[str]


# Flipped by the __main__ block to stage the chapter's crash in `generate`.
SIMULATE_OUTAGE = False


# The same four node shapes as ch06_save_resume.py, minus the print lines, so
# the output below is only about checkpoints.
def agent_node(state: State) -> dict:
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    route = "search" if not docs else ("web" if not web else "generate")
    return {"route": route, "iterations": state.get("iterations", 0) + 1}


def search_docs_node(state: State) -> dict:
    return {"docs": ["doc-1", "doc-2"]}


def web_search_node(state: State) -> dict:
    return {"web_results": ["web-A"]}


def generate_node(state: State) -> dict:
    if SIMULATE_OUTAGE:
        raise RuntimeError("downstream model timed out")
    return {"answer": "Final answer"}


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    lambda s: s["route"],
    {"search": "search_docs", "web": "web_search", "generate": "generate"},
)
builder.add_edge("search_docs", "agent")
builder.add_edge("web_search", "agent")
builder.add_edge("generate", END)

graph = builder.compile(checkpointer=InMemorySaver())


def fmt_next(nxt: tuple) -> str:
    return ", ".join(nxt) if nxt else "(done)"


def run_and_count(label: str, thread_id: str, **kwargs) -> None:
    # One fresh thread per mode, so the counts don't mix. `**kwargs` is either
    # empty (the default) or `durability=...`.
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"question": "How do I configure Kafka?"}, config=config, **kwargs)
    history = list(graph.get_state_history(config))
    final = graph.get_state(config)
    print(f"{label:<20} snapshots={len(history)}  next={fmt_next(final.next)}")


if __name__ == "__main__":
    print("===== 1. Same run, three durability modes =====")
    run_and_count('durability="sync"', "dur-sync", durability="sync")
    run_and_count('durability="async"', "dur-async", durability="async")
    run_and_count("(unset = async)", "dur-default")
    run_and_count('durability="exit"', "dur-exit", durability="exit")

    print('\n===== 2. Crash in generate under durability="exit" =====')
    crash_cfg = {"configurable": {"thread_id": "dur-exit-crash"}}
    SIMULATE_OUTAGE = True
    try:
        graph.invoke(
            {"question": "How do I configure Kafka?"},
            config=crash_cfg,
            durability="exit",
        )
    except RuntimeError as err:
        print("caught:", err)
    SIMULATE_OUTAGE = False

    # The exception unwound through the runtime, and the runtime flushed a
    # checkpoint on the way out -- one snapshot, parked at the failed node.
    history = list(graph.get_state_history(crash_cfg))
    parked = graph.get_state(crash_cfg)
    print(f"snapshots after the crash: {len(history)}")
    print(f"parked at: next={fmt_next(parked.next)} | "
          f"docs={len(parked.values.get('docs', []))} "
          f"web={len(parked.values.get('web_results', []))} "
          f"iterations={parked.values.get('iterations')}")

    # And it resumes like any other parked run: only `generate` runs.
    resumed = graph.invoke(None, config=crash_cfg, durability="exit")
    print(f"resumed: answer={resumed['answer']!r} iterations={resumed['iterations']}")
    print("(a hard process kill mid-run would have left no snapshot at all)")
