"""Chapter 6 — Save & Resume (checkpointing + time travel).

The doc-bot forgets everything the moment `invoke` returns. Here we make it
*durable* by handing the graph a **checkpointer**. From then on the runtime
auto-saves the full state after every superstep, filed under a `thread_id`, so
a run can resume from the last saved step — and we can even rewind to an older
checkpoint and fork a new run from it. (`InMemorySaver` here resumes within one
process; surviving a real restart needs a persistent saver — see the chapter.)

The graph loops: the agent looks at what it has gathered so far and routes to
search_docs, then web_search, then generate — revisiting `agent` between each.
That loop gives us several supersteps, and therefore several checkpoints, to
inspect, resume from, and travel back to.

    1. autosave      one run leaves a trail of snapshots on its thread
    2. crash+resume  a node blows up mid-run; invoke(None) continues from it
    3. time travel   rewind to an earlier checkpoint, change a field, fork

Nothing about the nodes changed to make this work — durability is a property of
the *compiled graph*, switched on with one keyword.

Run:  python code/ch06_save_resume.py
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    question: str  # input, set by the caller
    route: NotRequired[str]  # written by agent: search / web / generate
    docs: NotRequired[list[str]]  # written by search_docs
    web_results: NotRequired[list[str]]  # written by web_search
    iterations: NotRequired[int]  # bumped by agent each time it runs
    answer: NotRequired[str]  # written by generate


# Flip to simulate a flaky downstream in `generate`. The __main__ block toggles
# it to stage a crash, then a resume — the state itself never records this.
SIMULATE_OUTAGE = False


def agent_node(state: State) -> dict:
    # Look at what's gathered so far and decide the next step. With nothing yet,
    # search the docs; with docs but no web, search the web; once both are in,
    # generate. Each visit bumps the iteration count.
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    if not docs:
        route = "search"
    elif not web:
        route = "web"
    else:
        route = "generate"
    print(f"[agent] docs={len(docs)} web={len(web)} -> {route}")
    return {"route": route, "iterations": state.get("iterations", 0) + 1}


def search_docs_node(state: State) -> dict:
    # Pretend this is a slow, expensive retrieval call.
    print("[search_docs] searching internal docs (expensive)")
    return {"docs": ["doc-1", "doc-2"]}


def web_search_node(state: State) -> dict:
    # Also slow and expensive — the kind of work you don't want to redo.
    print("[web_search] searching the web (expensive)")
    return {"web_results": ["web-A"]}


def generate_node(state: State) -> dict:
    print("[generate] composing the final answer")
    if SIMULATE_OUTAGE:
        # The kind of transient failure a real LLM/API call can throw.
        raise RuntimeError("downstream model timed out")
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    answer = (
        "Final answer synthesized from the docs and web results "
        f"(docs={len(docs)}, web={len(web)})"
    )
    return {"answer": answer}


def route_after_agent(state: State) -> str:
    return state["route"]


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {"search": "search_docs", "web": "web_search", "generate": "generate"},
)
builder.add_edge("search_docs", "agent")   # loop back to the agent
builder.add_edge("web_search", "agent")     # loop back to the agent
builder.add_edge("generate", END)

# The one new line: hand the compiled graph a checkpointer. In production this
# would be a Sqlite/Postgres saver; in-memory is fine for learning.
graph = builder.compile(checkpointer=InMemorySaver())


def fmt_next(nxt: tuple) -> str:
    # snap.next is a tuple of node names still to run; empty means "done".
    return ", ".join(nxt) if nxt else "(done)"


def show_history(config: dict) -> None:
    # get_state_history yields snapshots newest-first; reverse for a timeline.
    history = list(graph.get_state_history(config))
    print(f"snapshots saved on this thread: {len(history)}")
    for snap in reversed(history):
        step = snap.metadata.get("step")
        vals = snap.values
        it = vals.get("iterations", "-")
        has_answer = "yes" if "answer" in vals else "no"
        print(
            f"  step {step:>2}  next={fmt_next(snap.next):<12}"
            f"  docs={len(vals.get('docs', []))}"
            f" web={len(vals.get('web_results', []))}"
            f" iter={it} answer={has_answer}"
        )


if __name__ == "__main__":
    # 1) Autosave: one run leaves a trail of snapshots under its thread_id.
    run1_cfg = {"configurable": {"thread_id": "run-001"}}
    print("===== 1. Autosave: one run, snapshots persist (thread_id=run-001) =====")
    run1 = graph.invoke(
        {"question": "How do I configure Kafka?"}, config=run1_cfg
    )
    print("answer:", run1["answer"])

    now = graph.get_state(run1_cfg)
    original_final_cfg = now.config   # remember this exact checkpoint for later
    print(f"\nlatest snapshot: next={fmt_next(now.next)}  "
          f"iterations={now.values['iterations']}")
    show_history(run1_cfg)

    # 2) Crash & resume: the same thread survives a mid-run failure.
    crash_cfg = {"configurable": {"thread_id": "run-crash-002"}}
    print("\n===== 2. Crash & resume on the same thread (thread_id=run-crash-002) =====")
    SIMULATE_OUTAGE = True
    print("-- first attempt (downstream is down) --")
    try:
        graph.invoke(
            {"question": "How do I configure Kafka?"}, config=crash_cfg
        )
    except RuntimeError as err:
        print("caught:", err)

    parked = graph.get_state(crash_cfg)
    print(f"parked at: next={fmt_next(parked.next)} | "
          f"docs={len(parked.values.get('docs', []))} "
          f"web={len(parked.values.get('web_results', []))} "
          f"iterations={parked.values.get('iterations')}")

    print("-- resume (downstream recovered) --")
    SIMULATE_OUTAGE = False
    # invoke(None) means: no new input, just continue from the saved point.
    resumed = graph.invoke(None, config=crash_cfg)
    print("resumed answer:", resumed["answer"])
    print(f"iterations={resumed['iterations']}  "
          f"(not restarted from 0 — it resumed mid-run)")
    print("note: no new [search_docs]/[web_search] lines — the expensive work "
          "came from the checkpoint")

    # 3) Time travel: rewind the old run, change a field, fork a new line.
    print("\n===== 3. Time travel: rewind, change a field, fork (thread_id=run-001 again) =====")
    print(f"original run: web_results={run1['web_results']} "
          f"iterations={run1['iterations']}")
    print(f"              answer={run1['answer']}")

    # Find the snapshot taken just before `generate` ran on the first run.
    before_generate = next(
        s for s in graph.get_state_history(run1_cfg)
        if s.next == ("generate",)
    )
    # Overwrite web_results at that point, attributing the edit to `agent` (the
    # step that had already decided to generate), so resuming re-runs ONLY
    # generate.
    forked_cfg = graph.update_state(
        before_generate.config,
        {"web_results": ["web-A", "web-B", "web-C"]},
        as_node="agent",
    )
    print("rewound to the pre-generate snapshot, injected 3 web results…")
    fork_run = graph.invoke(None, config=forked_cfg)
    print(f"forked run:   web_results={fork_run['web_results']} "
          f"iterations={fork_run['iterations']}")
    print(f"              answer={fork_run['answer']}")

    print(f"\nanswers differ after the fork: "
          f"{fork_run['answer'] != run1['answer']}")
    # The original checkpoint is still there, reachable by its own id.
    original_again = graph.get_state(original_final_cfg)
    print(f"original run still intact: "
          f"{original_again.values['answer'] == run1['answer']}")
    show_history(run1_cfg)
