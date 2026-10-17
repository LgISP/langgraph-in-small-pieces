"""Chapter 31 - The path an agent took, which answer-scoring cannot see.

Two runs of the SAME graph (Ch18's doc-bot, imported unchanged) that end with
byte-identical answers. Ch30's harness scores the answer, so it scores both runs
the same. One of them CALLED web_search on a question the internal docs plainly
cover -- the promise-breaking behaviour @sec-tool-calling warned about -- and
nothing in an answer-quality score can tell you.

Read the evidence boundary honestly: the model is Ch18's ScriptedModel and that
doc-bot's web_search returns a canned string, so nothing leaves the machine. What
this demonstrates is that a PATH difference survives an identical-answer check --
not an exfiltration. In a real deployment that same call is a real one.

No new agent is built here. The graph is Ch18's; only the script differs.

Run:  python code/ch31_trajectory.py
"""

from langchain_core.messages import AIMessage, HumanMessage
from ch18_docbot import Context, build_docbot
from ch18_test_docbot import ScriptedModel

ANSWER = "Set spring.kafka.listener.type=batch."
QUESTION = "How do I configure Kafka batch consumers?"


def call(name: str, query: str, cid: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": {"query": query}, "id": cid}])


# Both scripts end on the same sentence. Only the route differs.
STRAIGHT = [call("search_docs", "kafka batch", "c1"), AIMessage(content=ANSWER)]
DETOUR = [call("web_search", "kafka batch consumers", "c1"),
          call("search_docs", "kafka batch", "c2"),
          AIMessage(content=ANSWER)]


def requested(state) -> list[str]:
    """The tools a run ASKED for, in order -- attempts included.

    `invalid_tool_calls` is where a call whose arguments failed to parse lands.
    An audit wants it: a model that reached for a forbidden tool and fumbled the
    arguments still reached for it.
    """
    return [tc["name"] for m in state["messages"]
            for tc in (getattr(m, "tool_calls", None) or [])
            + (getattr(m, "invalid_tool_calls", None) or [])]


def run(script) -> dict:
    graph = build_docbot().compile()
    return graph.invoke(
        {"messages": [HumanMessage(QUESTION)], "tools_used": []},
        context=Context(llm=ScriptedModel(responses=list(script))),
    )


ALLOWED = {"search_docs"}   # a docs bot answers from the docs

if __name__ == "__main__":
    for label, script in (("A", STRAIGHT), ("B", DETOUR)):
        out = run(script)
        path = requested(out)
        stray = sorted(set(path) - ALLOWED)
        print(f"run {label}")
        print(f"  answer     : {out['messages'][-1].content}")
        print(f"  requested  : {path}")
        print(f"  off-policy : {stray or 'none'}")

    a, b = run(STRAIGHT), run(DETOUR)
    print()
    print("same answer      :", a["messages"][-1].content == b["messages"][-1].content)
    print("same requests    :", requested(a) == requested(b))
