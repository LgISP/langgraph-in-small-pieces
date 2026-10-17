"""Chapter 7 — Human in the Loop.

The bot is about to do something it can't take back — delete the production
knowledge base. Before it does, the graph *pauses itself* and hands control to
a human:

    plan → human_approval  ──interrupt()──▶  (human answers: yes / no)
                  │                                       │
                  └──────────── Command(resume=…) ────────┘
                                       ▼
                            execute (or cancel) → END

`interrupt()` stops the run mid-node and surfaces a payload as `__interrupt__`.
A human looks at it and calls `graph.invoke(Command(resume=decision), …)`,
which continues from exactly where it stopped. None of this works without a
checkpointer — the pause *is* a saved checkpoint, the same machinery as the
previous chapter.

Run:  python code/ch07_human_in_the_loop.py

The first two scenarios script the answer so the yes/no split is reproducible.
The third has no scripted answer: run it in a real terminal and the graph stops
and waits for you to type. (It is skipped when stdin is not a TTY, so an
automated run doesn't hang on the prompt.)
"""

import sys
from typing import NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


class State(TypedDict):
    question: str                # input, set by the caller
    approved: NotRequired[bool]  # written by human_approval, from the human's answer
    result: NotRequired[str]     # written by execute: what actually happened


def plan_node(state: State) -> dict:
    # Stand-in for the bot's judgment: it decides to run a destructive action.
    print("[plan] the bot decided to delete the production knowledge base")
    return {}


def human_approval_node(state: State) -> dict:
    # This line runs EVERY time the node runs — including on resume, when the
    # node replays from the top. It only describes, so repeating it is harmless.
    print("[human_approval] preparing the approval request")

    # interrupt(...) stops the graph right here:
    #   1) the payload is thrown out of the graph, for a human to look at
    #   2) a checkpoint is saved and invoke() returns, carrying __interrupt__
    #   3) on resume, interrupt() RETURNS the value passed to Command(resume=…)
    decision = interrupt(
        {
            "action": "delete the production knowledge base",
            "question": "Confirm? reply 'yes' or 'no'",
        }
    )

    # Not reached on the first pass; runs only after a human resumes.
    print(f"[human_approval] human decided: {decision}")
    return {"approved": decision == "yes"}


def execute_node(state: State) -> dict:
    # The irreversible work lives in its OWN node, downstream of the interrupt,
    # so it runs exactly once — never during the human_approval replay.
    if state["approved"]:
        print("[execute] approved → deleting the knowledge base")
        return {"result": "knowledge base deleted"}
    print("[execute] rejected → operation cancelled")
    return {"result": "operation cancelled"}


builder = StateGraph(State)
builder.add_node("plan", plan_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("execute", execute_node)

builder.add_edge(START, "plan")
builder.add_edge("plan", "human_approval")
builder.add_edge("human_approval", "execute")
builder.add_edge("execute", END)

# HITL REQUIRES a checkpointer: the pause is a saved checkpoint, and resume
# reads it back. (InMemorySaver is fine for a demo; use a durable backend in
# production — see Chapter 6.)
graph = builder.compile(checkpointer=InMemorySaver())


def run_demo(thread_id: str, human_decision: str | None = None) -> None:
    # The thread_id names this run's checkpoint lane, exactly as in the previous
    # chapter. Resume must reuse the SAME thread_id to land on the paused
    # checkpoint.
    config = {"configurable": {"thread_id": thread_id}}

    if human_decision is None:
        print("\n===== scenario: waiting for a real human at the keyboard =====")
    else:
        print(f"\n===== scenario: the human will answer '{human_decision}' =====")

    # 1) Start the run. It flows until human_approval calls interrupt(), then
    #    pauses. invoke() returns the state so far — plus __interrupt__.
    result = graph.invoke({"question": "Clean up old data"}, config=config)

    # 2) The pause surfaces as __interrupt__: a list of Interrupt objects. Read
    #    .value to see what the graph is asking a human to decide.
    pending = result["__interrupt__"][0]
    print("[paused] the graph stopped and is asking:")
    print("        ", pending.value)

    # 3) Inspect where it parked. .next is the node it will resume into.
    snapshot = graph.get_state(config)
    print("[paused] parked at node:", snapshot.next)

    # 4) Get the human's decision. A scripted answer resumes immediately; with
    #    none, this is a REAL pause — the run is sitting on a saved checkpoint and
    #    we block at the keyboard until a person types. You could kill the process
    #    right here, restart, and resume from this same interrupt: the pause
    #    outlives the program.
    if human_decision is None:
        decision = input("        your answer ('yes'/'no'): ").strip()
    else:
        decision = human_decision

    # 5) Resume from exactly the interrupt point; the decision becomes the return
    #    value of interrupt() inside the node.
    final = graph.invoke(Command(resume=decision), config=config)
    print("[done] result:", final["result"])


if __name__ == "__main__":
    # Two scripted threads make the yes/no split reproducible.
    run_demo(thread_id="approve-case", human_decision="yes")
    run_demo(thread_id="reject-case", human_decision="no")

    # A third thread with NO scripted answer: the graph really stops and waits
    # for you to type. Guarded by isatty() so an automated run (piped stdin, CI)
    # skips the blocking prompt instead of hanging on it.
    if sys.stdin.isatty():
        run_demo(thread_id="interactive-case")
    else:
        print("\n(skipping interactive scenario: stdin is not a terminal)")
