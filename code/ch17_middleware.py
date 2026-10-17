"""Chapter 17 - Middleware: hooks on the agent harness.

Chapter 11's `create_agent` hides the model-tools loop; middleware is the sanctioned
way back in. A middleware object hangs functions on the loop's seams - before and
after the model call, around the model call, around each tool call - the same
boundaries you wired by hand as nodes and edges in Part I.

Two demos:
  1) The shipped HumanInTheLoopMiddleware: one line turns a destructive tool into
     an approval gate - Chapter 7's interrupt(), relocated into the harness.
  2) A hand-built CallLimitMiddleware: Chapter 4's fuse as a before_model hook,
     with the counter living in an extended agent state (Chapter 12's move).

Run:  python code/ch17_middleware.py
"""

from typing import Any, NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    HumanInTheLoopMiddleware,
    hook_config,
)
from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

MODEL = "qwen/qwen3.8-27b"
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
                 model=MODEL, temperature=0, reasoning_effort="none")


@tool
def search_docs(query: str) -> str:
    """Search the local documentation for configuration answers."""
    return "No relevant document found. Try a different phrasing."


@tool
def delete_index(name: str) -> str:
    """Delete a document index permanently. Destructive."""
    return f"index '{name}' deleted"


# ---------- 1) The shipped gate: HumanInTheLoopMiddleware ----------
def demo_hitl() -> None:
    agent = create_agent(
        llm,
        [search_docs, delete_index],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"delete_index": True})],
        checkpointer=InMemorySaver(),      # an interrupt needs somewhere to park
    )

    cfg = {"configurable": {"thread_id": "hitl-1"}}
    result = agent.invoke(
        {"messages": [("user", "Delete the index called stale-docs-v1.")]},
        config=cfg,
    )
    req = result["__interrupt__"][0].value["action_requests"][0]
    print(f"[paused]   the agent wants: {req['name']}({req['args']})")

    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=cfg)
    print(f"[approved] {result['messages'][-1].content}")

    # Same request on a fresh thread, rejected this time.
    cfg2 = {"configurable": {"thread_id": "hitl-2"}}
    agent.invoke({"messages": [("user", "Delete the index called stale-docs-v1.")]}, config=cfg2)
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "reject",
                                       "message": "Rejected: that index is still serving traffic."}]}),
        config=cfg2,
    )
    print(f"[rejected] {result['messages'][-1].content}")


# ---------- 2) A middleware by hand: Chapter 4's fuse as a hook ----------
class CallCounterState(AgentState):
    model_calls: NotRequired[int]          # the middleware's own channel


class CallLimitMiddleware(AgentMiddleware):
    state_schema = CallCounterState

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime) -> dict[str, Any] | None:
        if state.get("model_calls", 0) >= self.limit:
            return {"jump_to": "end",
                    "messages": [AIMessage(f"Stopped: model call limit of {self.limit} reached.")]}
        return None                        # under the limit: change nothing

    def after_model(self, state, runtime) -> dict[str, Any] | None:
        return {"model_calls": state.get("model_calls", 0) + 1}


def demo_call_limit() -> None:
    agent = create_agent(llm, [search_docs], middleware=[CallLimitMiddleware(limit=3)])
    result = agent.invoke({"messages": [("user",
        "Find the exact config key for enabling Kafka batch mode. "
        "Keep searching the docs until you find it.")]})
    calls = result.get("model_calls", 0)        # the middleware's own count
    print(f"model calls made: {calls}")
    print(f"final message:    {result['messages'][-1].content}")


if __name__ == "__main__":
    print(f"===== 1. HumanInTheLoopMiddleware (model: {MODEL}) =====")
    demo_hitl()
    print("\n===== 2. A hand-built CallLimitMiddleware, limit=3 =====")
    demo_call_limit()
