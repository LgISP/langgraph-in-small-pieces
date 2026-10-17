"""Chapter 1 — The Smallest Graph.

A graph with a single state and a single node. No LLM yet; the answer is
hard-coded. The point is to watch the machine turn over: state in, update out,
runtime merges.

Run:  python code/ch01_minimal_graph.py
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str              # input:  the caller provides this
    answer: NotRequired[str]   # output: a node fills this in


def answer_node(state: State) -> dict:
    question = state["question"]

    # Hard-coded for now; a real LLM replaces this in Part II.
    reply = f"You asked: “{question}”. (This answer is hard-coded.)"

    # Return only the fields we want to change, not the whole state.
    return {"answer": reply}


builder = StateGraph(State)          # this graph's state has shape `State`
builder.add_node("answer", answer_node)

builder.add_edge(START, "answer")    # enter → run `answer`
builder.add_edge("answer", END)      # then → done

graph = builder.compile()            # turn the design into a runnable graph


if __name__ == "__main__":
    result = graph.invoke({"question": "How do I configure Kafka in Spring Boot?"})

    print("final state:", result)
    print("answer:", result["answer"])
