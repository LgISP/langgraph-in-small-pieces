"""Chapter 22 - LangSmith reports token usage on its own.

The @traced demo (ch22_observability.py) uses stub nodes with fake sleeps, so
there are no tokens to see. Here `generate` makes a real model call -- and with
LangSmith tracing on (the three env vars from .env), the `generate` span carries
the model's `usage_metadata`, so token counts show up in the trace with no
counting code of ours. A node-boundary decorator could never report them: the
tokens ride in on the model's reply, not on the node's return value.

Run:  set -a; source .env; set +a          # load LANGSMITH_* into the environment
      python code/ch22_langsmith.py
"""

import os
import sys
from typing import NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen/qwen3.8-27b",
    temperature=0,
    reasoning_effort="none",
)


class State(TypedDict):
    question: str
    docs: NotRequired[list[str]]
    answer: NotRequired[str]


def search_docs_node(state: State) -> dict:
    # Stub, exactly as in the earlier examples: no model, no tokens.
    return {"docs": ["Kafka Manual: set spring.kafka.listener.type=batch to enable batch consumption."]}


def generate_node(state: State) -> dict:
    # The one real model call -- this is where the tokens are.
    sources = "\n".join(f"- {d}" for d in state.get("docs", []))
    prompt = (
        "Answer the question concisely, using only the sources below.\n\n"
        f"Question: {state['question']}\n\nSources:\n{sources}\n\nAnswer:"
    )
    return {"answer": llm.invoke(prompt).content.strip()}


builder = StateGraph(State)
builder.add_node("search_docs", search_docs_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "search_docs")
builder.add_edge("search_docs", "generate")
builder.add_edge("generate", END)
graph = builder.compile()


if __name__ == "__main__":
    if not os.environ.get("LANGSMITH_TRACING"):
        sys.exit("LANGSMITH_TRACING is not set -- run `set -a; source .env; set +a` first.")

    from langchain_core.tracers.context import collect_runs
    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langsmith import Client

    with collect_runs() as cb:
        out = graph.invoke({"question": "How do I configure Kafka batch consumers?"})
    print("answer:", out["answer"])

    # LangSmith rolls the model's token usage up to the run. Read it back to prove
    # it landed -- the same numbers you would see on the trace's span in the UI.
    # Ingestion lags the send by a beat, so retry until the run is queryable.
    wait_for_all_tracers()
    client = Client()
    run_id = cb.traced_runs[0].id
    run = None
    for _ in range(15):
        try:
            run = client.read_run(run_id)
            break
        except Exception:
            list(client.list_runs(project_name=os.environ.get("LANGSMITH_PROJECT", "default"), limit=1))

    if run is None:
        print("\nrun not queryable yet; open it in the LangSmith UI:", run_id)
    else:
        print("\nToken usage, straight from LangSmith (no counting code of ours):")
        print(f"  prompt     : {run.prompt_tokens}")
        print(f"  completion : {run.completion_tokens}")
        print(f"  total      : {run.total_tokens}")
        print("\ntrace:", run.url)
