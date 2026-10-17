"""Chapter 16 Challenge — Gold: one token stream, two nodes, routed by name.

The chapter's `if metadata.get("langgraph_node") == "generate"` line hinted at
routing; here it does real work. We add a `critique` node after `generate` — a
*second* model call that asks for a one-line critique of the answer. Now two nodes
pour tokens onto the **same** `messages` stream, interleaved in run order. We use
`metadata["langgraph_node"]` to split them back apart: `generate`'s tokens print
under an "Answer:" heading, `critique`'s under a "Critique:" heading, live, as
each streams.

Both calls go to the book's real local model (Qwen3.8 behind LM Studio), so
the token text differs run to run. The tests therefore never compare against
canned strings: they stream `messages` and `updates` together in ONE run and
assert that the tokens routed to each node reassemble into exactly the text that
node returned into `State` — which is also the proof the two token streams never
bled into each other.

Run:  python code/answers/ch16_two_streams.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


# The book's real local model — the chapter's local-server setup.
# One model instance serves both nodes: the routing works the same whether the
# tokens come from one shared model or two — `langgraph_node` tags the NODE that
# made the call, not the model object.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


class State(TypedDict):
    question: str
    docs: NotRequired[list[str]]
    answer: NotRequired[str]
    critique: NotRequired[str]


def search_docs_node(state: State) -> dict:
    return {"docs": [
        "Kafka Manual: set spring.kafka.listener.type=batch to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ]}


def generate_node(state: State) -> dict:
    sources = "\n".join(f"- {d}" for d in state.get("docs", []))
    prompt = (f"Answer the user's question in one or two sentences, using only:\n"
              f"{sources}\n\nQ: {state['question']}")
    return {"answer": llm.invoke(prompt).content.strip()}


def critique_node(state: State) -> dict:
    prompt = f"Give a one-line critique of this answer:\n{state['answer']}"
    return {"critique": llm.invoke(prompt).content.strip()}


builder = StateGraph(State)
builder.add_node("search_docs", search_docs_node)
builder.add_node("generate", generate_node)
builder.add_node("critique", critique_node)
builder.add_edge(START, "search_docs")
builder.add_edge("search_docs", "generate")
builder.add_edge("generate", "critique")
builder.add_edge("critique", END)
graph = builder.compile()


QUESTION = "How do I configure Kafka batch consumers?"

_HEADINGS = {"generate": "Answer:", "critique": "Critique:"}


def demo_two_streams() -> None:
    # One messages stream, two model nodes. Route each token by its source node
    # and print a heading the moment a new node starts to speak.
    print("\n===== stream_mode='messages', two nodes routed by node name =====")
    current = None
    for msg_chunk, metadata in graph.stream(
        {"question": QUESTION}, stream_mode="messages"
    ):
        node = metadata.get("langgraph_node")
        if node not in _HEADINGS or not msg_chunk.content:
            continue
        if node != current:                       # the speaker changed
            print(f"\n  {_HEADINGS[node]} ", end="", flush=True)
            current = node
        print(msg_chunk.content, end="", flush=True)
    print()


def collect_run() -> tuple[dict[str, str], dict[str, str]]:
    """ONE run, two stream modes at once: reassemble each node's tokens from the
    interleaved `messages` stream, and capture what each node returned to State
    from the `updates` stream. Returns (routed_tokens, state_updates)."""
    routed: dict[str, str] = {"generate": "", "critique": ""}
    updates: dict[str, str] = {}
    for mode, payload in graph.stream(
        {"question": QUESTION}, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            msg_chunk, metadata = payload
            node = metadata.get("langgraph_node")
            if node in routed and msg_chunk.content:
                routed[node] += msg_chunk.content
        elif mode == "updates":
            for node, delta in payload.items():
                if node == "generate":
                    updates["generate"] = delta["answer"]
                elif node == "critique":
                    updates["critique"] = delta["critique"]
    return routed, updates


_CACHE: list = []


def cached_run():
    if not _CACHE:
        _CACHE.append(collect_run())
    return _CACHE[0]


# --- tests -------------------------------------------------------------------
# Real-model note: no canned text anywhere. Each node's routed tokens must equal
# the text that same node returned into State on the SAME run — structure, not
# wording.
def test_generate_tokens_reassemble_into_the_answer():
    routed, updates = cached_run()
    assert routed["generate"].strip() == updates["generate"].strip()


def test_critique_tokens_reassemble_into_the_critique():
    routed, updates = cached_run()
    assert routed["critique"].strip() == updates["critique"].strip()


def test_the_two_streams_do_not_bleed_into_each_other():
    routed, updates = cached_run()
    # Both nodes really streamed, and each reassembled EXACTLY into its own
    # node's State text — one stray token in the wrong bucket would break the
    # equality above and the length accounting here.
    assert routed["generate"] and routed["critique"]
    total = len(routed["generate"]) + len(routed["critique"])
    assert total == len(updates["generate"]) + len(updates["critique"])


def test_both_nodes_land_their_result_in_state():
    _, updates = cached_run()
    assert updates["generate"]                      # the answer reached State
    assert updates["critique"]                      # ...and so did the critique


if __name__ == "__main__":
    demo_two_streams()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
