"""Chapter 12 — State Schemas.

The doc-bot's State has been a TypedDict since Chapter 1. It doesn't have to be.
This file writes the same tiny question/answer graph four ways — TypedDict,
Pydantic BaseModel, @dataclass, and the prebuilt MessagesState — so you can see
what each choice buys you (validation, defaults, access style) and what it costs.

No LLM here: unlike the rest of Part II, every run is fully deterministic, because
the whole point is the shape of State, not a model's judgment.
"""

from dataclasses import dataclass
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ValidationError


# The one piece of work every version does, so only the STATE differs below.
def make_answer(question: str) -> str:
    # `.strip()` is an ordinary string operation — it only works on a real str.
    # That matters in the dataclass section, where a bad value gets this far.
    return f"You asked: {question.strip()}"


# ---------------------------------------------------------------------------
# 1. TypedDict — the book's default, and the lightest option.
# ---------------------------------------------------------------------------
# At runtime this is just a dict; the annotations are for the type checker only,
# so nothing is validated. Nodes read with subscripts: state["question"].
class State(TypedDict):
    question: str
    answer: NotRequired[str]


def answer_typeddict(state: State) -> dict:
    print('  [typeddict]  read state["question"] (subscript)')
    return {"answer": make_answer(state["question"])}


def run_typeddict() -> None:
    print("===== 1. TypedDict (the default, lightest) =====")
    builder = StateGraph(State)
    builder.add_node("answer", answer_typeddict)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile()
    print("  result:", graph.invoke({"question": " how do I configure Kafka? "}))


# ---------------------------------------------------------------------------
# 2. Pydantic BaseModel — runtime validation + coercion + defaults.
# ---------------------------------------------------------------------------
# The heavy option. LangGraph validates the input against the model at invoke()
# time, so a bad type fails EARLY, before any node runs. Nodes read with
# attributes: state.question. `answer` has a default, so callers may omit it.
class ValidatedState(BaseModel):
    question: str
    answer: str = ""


def answer_model(state: ValidatedState) -> dict:
    print("  [basemodel]  read state.question (attribute)")
    return {"answer": make_answer(state.question)}


def run_basemodel() -> None:
    print("\n===== 2. Pydantic BaseModel (runtime validation) =====")
    builder = StateGraph(ValidatedState)
    builder.add_node("answer", answer_model)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile()
    print("  result:", graph.invoke({"question": " how do I configure Kafka? "}))

    # Now feed it a bad value: question should be a str, we pass an int.
    print("  --- feeding it a bad value: question=123 ---")
    try:
        graph.invoke({"question": 123})
    except ValidationError as e:
        print("  ValidationError raised at invoke(), before any node ran:")
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            print(f"    {loc}: {err['msg']} (got {err['input']!r})")


# ---------------------------------------------------------------------------
# 3. dataclass — defaults + attribute access, but NO validation.
# ---------------------------------------------------------------------------
# The middle option: like BaseModel you get defaults and state.question access,
# but like TypedDict nothing checks the types. A bad value slips through invoke()
# and only detonates later, deep inside a node.
@dataclass
class DataclassState:
    question: str
    answer: str = ""


def answer_dataclass(state: DataclassState) -> dict:
    print("  [dataclass]  read state.question (attribute)")
    return {"answer": make_answer(state.question)}


def run_dataclass() -> None:
    print("\n===== 3. dataclass (defaults, attribute access, no validation) =====")
    builder = StateGraph(DataclassState)
    builder.add_node("answer", answer_dataclass)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile()
    print("  result:", graph.invoke({"question": " how do I configure Kafka? "}))

    # Feed it the SAME bad value. Unlike BaseModel, invoke() does not complain.
    print("  --- feeding it the SAME bad value: question=123 ---")
    try:
        graph.invoke({"question": 123})
    except Exception as e:
        print("  input passed schema construction; the node then raised:")
        print(f"    {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 4. MessagesState — the prebuilt TypedDict for chat / tool-calling.
# ---------------------------------------------------------------------------
# It is a TypedDict with one field baked in: messages: Annotated[list,
# add_messages]. The add_messages reducer APPENDS instead of overwriting — the
# same Annotated-reducer machinery from Chapter 8, prebuilt for you. Subclass it
# to add your own fields. This is the shape ToolNode / create_agent expect
# (Chapter 11). Access is by subscript, like any TypedDict: state["messages"].
class ChatState(MessagesState):
    user_id: str


def chat_node(state: ChatState) -> dict:
    incoming = state["messages"][-1].content
    print(f'  [messages]   user={state["user_id"]} said {incoming!r}; appending a reply')
    # Return ONE message; the add_messages reducer appends it to the list.
    return {"messages": [AIMessage(content=f"You asked: {incoming.strip()}")]}


def run_messagesstate() -> None:
    print("\n===== 4. MessagesState (prebuilt, for chat / tool-calling) =====")
    builder = StateGraph(ChatState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    graph = builder.compile()
    result = graph.invoke(
        {"messages": [HumanMessage(content="how do I configure Kafka?")], "user_id": "u_123"}
    )
    print(f"  result: {len(result['messages'])} message(s), user_id={result['user_id']!r}")
    for msg in result["messages"]:
        print(f"    {type(msg).__name__}: {msg.content!r}")


# ---------------------------------------------------------------------------
# 5. Input / output schema separation — decouple the public shape from State.
# ---------------------------------------------------------------------------
# The graph runs on a wide internal State, but the caller only supplies InputState
# and only sees OutputState. The internal `scratch` field never leaves the graph.
class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    answer: str


class InternalState(TypedDict):
    question: str
    scratch: str  # never leaves the graph
    answer: str


def io_node(state: InternalState) -> dict:
    return {"scratch": "internal notes", "answer": make_answer(state["question"])}


def run_io_schema() -> None:
    print("\n===== 5. Input / output schema separation =====")
    builder = StateGraph(
        InternalState, input_schema=InputState, output_schema=OutputState
    )
    builder.add_node("answer", io_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile()
    # Caller passes only `question`; the returned dict carries only `answer`.
    result = graph.invoke({"question": " how do I configure Kafka? "})
    print("  invoke saw only {question}; internal 'scratch' is hidden from output")
    print("  result:", result)


# A reducer rides Annotated the same way for ALL four schema types. MessagesState's
# `messages` field is exactly this pattern, prebuilt (Chapter 8):
class ReducerState(TypedDict):
    messages: Annotated[list, add_messages]  # append-on-write, not overwrite


if __name__ == "__main__":
    run_typeddict()
    run_basemodel()
    run_dataclass()
    run_messagesstate()
    run_io_schema()
