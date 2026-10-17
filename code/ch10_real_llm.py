"""Chapter 10 — A real LLM, the right way: structured output.

This is the Chapter 5 doc-bot with real-model node bodies, but the `agent` no
longer parses free text with `if "SEARCH" in reply`. It asks the model for a
`Decision` — a Pydantic model whose `route` is a `Literal` — via
`llm.with_structured_output(...)`. Now the route is a guaranteed-legal enum, not a
string we hope to match.

Two independent things (only the first is a guarantee):
    shape  -- `route` is one of search / web / generate, never garbage (the Literal)
    choice -- the ordered rules in the prompt still steer WHICH one it picks; only
              as good as the prompt and the model

The node names, routes, and compiled graph topology are Chapter 5's; only the
`agent` and `generate` bodies changed, plus one added `reason` field on State.

Run:  python code/ch10_real_llm.py
"""

import json
import warnings
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# with_structured_output makes pydantic emit one harmless serializer warning about
# an internal field; the parsed result is correct. Silence just that one line.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model="qwen/qwen3.8-27b",
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


# The agent's decision as a typed shape. LangChain turns this class into JSON
# Schema and sends it to the model, so every `description` here is really a prompt:
# the model reads them and fills the fields accordingly (see the Deep Dive).
class Decision(BaseModel):
    """Choose the next action for a documentation question-answering agent."""

    route: Literal["search", "web", "generate"] = Field(
        description="Next action: search=look in the local docs, "
        "web=search the open web, generate=write the answer now."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


# method="json_schema" asks LM Studio to enforce the schema with response_format.
decider = llm.with_structured_output(Decision, method="json_schema")


class State(TypedDict):
    question: str                       # input, set by the caller
    route: NotRequired[str]             # written by agent: search|web|generate
    docs: NotRequired[list[str]]        # written by search_docs (may be empty)
    web_results: NotRequired[list[str]] # written by web_search
    iterations: NotRequired[int]        # loop counter the agent keeps (Ch4)
    reason: NotRequired[str]            # the model's stated reason (observability)
    answer: NotRequired[str]            # written by generate


KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch "
        "to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ],
    "redis": [
        "Redis Manual: set the connection pool via spring.data.redis.*.",
    ],
    # Note: deliberately no "python" topic — used below to show a miss.
}


def excerpt(text: str, width: int = 140) -> str:
    """First line of a model answer, cut at a word boundary, with '...' if cut.

    Answers are long and often contain code fences; the trace only needs
    enough to see WHAT the model decided to say, not the whole essay.
    """
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) <= width and first == text.strip():
        return first
    cut = first[:width].rsplit(" ", 1)[0] if len(first) > width else first
    return cut.rstrip(" ,;:") + " ..."


# The agent, deciding with structured output. Same job as Chapter 5 — read the
# state, write a route — but the route arrives as a legal enum, no parsing.
def agent_node(state: State) -> dict:
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    searched_docs = "docs" in state
    searched_web = "web_results" in state

    # Same ordered rules as before. The Literal guarantees the shape (route is a
    # legal enum); these rules guide the choice. Note rule 2/3: the model has to
    # be told, in words, that "already searched" is final — a distinction the
    # Chapter 5 code got for free from `"docs" in state` (see the Gotcha).
    prompt = f"""
        You are the router for a documentation question-answering agent.
        Decide the single next action from the state below.

        Question: {state['question']}
        Local documents found so far: {len(docs)}
        Web results so far: {len(web)}
        Already searched the local docs: {searched_docs}
        Already searched the web: {searched_web}

        Rules — read them in order and act on the FIRST whose condition is true; once a
        rule matches, ignore every rule below it.
        1. If you ALREADY have at least one source (local documents found > 0 OR web
           results > 0), you have enough -> generate. Do NOT gather more: one source is
           enough, even if the web has not been searched yet.
        2. else if (already searched the local docs is False)            -> search
        3. else if (already searched the web is False)                   -> web
        4. otherwise                                                     -> generate
        Treat "already searched" as final: searching again cannot change a count that is
        0, so never repeat a search that was already done."""

    decision = decider.invoke(prompt)
    print(f"[agent] docs={len(docs)} web={len(web)} -> {decision.route}"
          f"  ({excerpt(decision.reason)})")   # reason trimmed for the trace
    return {
        "route": decision.route,
        "reason": decision.reason,
        "iterations": state.get("iterations", 0) + 1,
    }


def search_docs_node(state: State) -> dict:
    question = state["question"].lower()
    hits: list[str] = []
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in question:
            hits.extend(docs)
    print(f"[search_docs] {len(hits)} hit(s)")
    return {"docs": hits}


def web_search_node(state: State) -> dict:
    print("[web_search] falling back to a web search")
    return {"web_results": [f"[web] community thread about “{state['question']}”"]}


def generate_node(state: State) -> dict:
    sources = state.get("docs", []) + state.get("web_results", [])
    sources_text = "\n".join(f"- {s}" for s in sources) or "(no sources)"
    prompt = f"""
        Answer the user's question concisely, using only the sources
        below. Do not invent anything.

        Question: {state['question']}

        Sources:
        {sources_text}

        Answer:"""
    answer = llm.invoke(prompt).content.strip()
    print("[generate] the model wrote an answer")
    return {"answer": answer}


def route_after_agent(state: State) -> str:
    return state["route"]


# The graph is Chapter 5's, unchanged — same nodes, same edges, same router.
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
builder.add_edge("search_docs", "agent")
builder.add_edge("web_search", "agent")
builder.add_edge("generate", END)
graph = builder.compile()


if __name__ == "__main__":
    # What the model actually sees: the Pydantic class as JSON Schema. Every
    # description string below travels to the model verbatim (see the Deep Dive).
    print("===== Decision JSON schema =====")
    print(json.dumps(Decision.model_json_schema(), indent=2))

    # Two questions take the two paths, exactly as in Chapter 5:
    #   "Kafka"  is in the knowledge base -> local hit -> generate
    #   "Python" is not                   -> local miss -> fall back to web
    for q in [
        "How do I configure Kafka batch consumers?",
        "How do I write a Python decorator?",
    ]:
        print(f"\n===== question: {q} =====")
        result = graph.invoke({"question": q}, {"recursion_limit": 10})
        print("--- answer ---")
        print(excerpt(result["answer"]))   # first line, trimmed: answers run long
