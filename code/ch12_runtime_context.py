"""Chapter 12 — runtime context: pass ambient dependencies via context, not State.

State is the DATA flowing through the graph (question, docs, answer). The model
handle, the user id, a database connection are ambient DEPENDENCIES — they steer a
run but they are not data the graph produces. LangGraph 1.x gives them their own
typed channel: declare a `context_schema` on the graph, and every node can read a
`Runtime[Context]` whose `.context` holds those per-run values. State never carries
them, and because the model HANDLE rides this channel, a test can inject a fake
model by passing a different context — with no change to the graph.

Run:  python code/ch12_runtime_context.py
"""

from dataclasses import dataclass
from typing import NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime


# The per-run context: ambient DEPENDENCIES, not graph data. A db connection or a
# feature flag would live here too. It holds the model HANDLE itself, so a test can
# pass a fake model through the very same channel.
@dataclass
class Context:
    user_id: str
    llm: BaseChatModel      # the model handle itself — a dependency, not data
    model_label: str        # a display name for the run; see below


class State(TypedDict):
    question: str
    answer: NotRequired[str]    # the model/user/db are NOT here


# A node's optional SECOND parameter is the Runtime; LangGraph injects it.
def answer_node(state: State, runtime: Runtime[Context]) -> dict:
    user_id = runtime.context.user_id
    llm = runtime.context.llm
    # Read the label from context, NOT `llm.model_name`: that attribute is
    # provider-specific (ChatOpenAI has it, FakeListChatModel does not), and
    # touching it here would silently narrow the dependency from BaseChatModel to
    # "an OpenAI handle" — breaking the fake-injection the whole pattern promises.
    # The node uses `llm` only through the BaseChatModel interface (e.g. .invoke()).
    print(f"[answer] user={user_id} using model={runtime.context.model_label}")
    return {"answer": f"question from {user_id}, to be answered by {runtime.context.model_label}"}


builder = StateGraph(State, context_schema=Context)
builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)
graph = builder.compile()


if __name__ == "__main__":
    # Build the model handles once (constructing them touches no network); hand a
    # different one to each run through context.
    big = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
                     model="openai/gpt-oss-20b", temperature=0)
    small = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
                       model="qwen2.5-vl-7b-instruct", temperature=0)

    # Same compiled graph, two runs, different ambient context — State is identical,
    # only the context differs.
    for ctx in [
        Context(user_id="u_123", llm=big, model_label="openai/gpt-oss-20b"),
        Context(user_id="u_777", llm=small, model_label="qwen2.5-vl-7b-instruct"),
    ]:
        result = graph.invoke({"question": "ping"}, context=ctx)
        print("  ->", result["answer"])

    # The payoff: because the node reads `llm` only through the BaseChatModel
    # interface, ANY BaseChatModel drops in through the same channel — including a
    # fake, with no change to the graph. (This is what Chapter 18 leans on.)
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    fake_ctx = Context(user_id="u_test", llm=FakeListChatModel(responses=["ok"]),
                       model_label="fake")
    result = graph.invoke({"question": "ping"}, context=fake_ctx)
    assert result["answer"], "the fake model should drive the graph unchanged"
    print("  -> fake injected cleanly:", result["answer"])
