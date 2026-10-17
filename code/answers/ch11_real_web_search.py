"""Chapter 11 Challenge - Gold: give the agent a REAL web search.

The chapter's `web_search` is a placeholder that returns a canned string, so the
"recent" answers in the trace came from the model's own memory, not the web -- that
is the Gotcha in the flesh. This makes it real: the backend is DuckDuckGo via the
`ddgs` package (no API key), the model is the chapter's gpt-oss-20b, and the agent
loop is the chapter's `create_agent` one-liner.

The packaging-axis point (the Deep Dive) survives *only if nothing else moves*: the
`@tool` contract, the loop, and the OTHER tool (`search_docs`, restored here to the
chapter's real keyword lookup) are all untouched -- the *body* of `web_search` is the
only casualty. Same `@tool` name, same one-string-arg schema, same `create_agent`
call. Swapping the body is what repairs the Gotcha: the "latest release" answer is
now grounded in retrieved titles and URLs instead of training data.

We run BOTH questions the Gold asks for -- the Python-decorator one and a recent one
-- so the regression is real, not just the recent question. Citing the URL in the
final prose is an OPTIONAL extra (a 20b needs a nudge, see `agent_citing` below); it
is not part of the drop-in swap, so the grounding test checks the URL in the tool
output, where it provably came from the search, not the model.

NOTE - THIS FILE GOES ONLINE. It needs `pip install ddgs` and a working network
connection (the only file in answers/ that does; everything else stays on
localhost). The tests assert structure -- a tool call happened, real URLs reached
the tool transcript, `search_docs` still does keyword lookup -- never specific
search results, which change daily.

Run:  python code/answers/ch11_real_web_search.py   (see answers/README.md)
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
                 model="openai/gpt-oss-20b", temperature=0)


def ddgs_backend(query: str) -> list:
    """Live DuckDuckGo search: query -> [{title, snippet, url}]. Network required."""
    from ddgs import DDGS
    with DDGS() as d:
        return [{"title": r["title"], "snippet": r["body"], "url": r["href"]}
                for r in d.text(query, max_results=3)]


SEARCH_BACKEND = ddgs_backend    # the swap point: the stub's body was the ONLY casualty


def format_results(results: list) -> str:
    """Turn raw hits into the plain-text block the model reads back."""
    if not results:
        return "No web results found."
    return "\n".join(f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results)


@tool
def web_search(query: str) -> str:
    """Search the open web. Use this when the local docs come up empty, or the
    question is about something recent."""
    return format_results(SEARCH_BACKEND(query))


# search_docs is the chapter's tool, verbatim -- restored so that ONLY web_search's
# body changed. (An earlier draft short-circuited this to a canned Kafka string,
# which quietly broke the "nothing else moved" point the Gold is making.)
KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ],
    "redis": [
        "Redis Manual: set the connection pool via spring.data.redis.*.",
    ],
}


@tool
def search_docs(query: str) -> str:
    """Search the internal technical-documentation knowledge base. Use this
    whenever the user asks how to configure something, about an error message, or
    how to do a technical task."""
    q = query.lower()
    hits: list[str] = []
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in q:
            hits.extend(docs)
    return "\n".join(hits) if hits else "No matching local documents."


# The drop-in agent: the chapter's one-liner, unchanged. Nothing here knows the
# web_search body is now live.
agent = create_agent(llm, [search_docs, web_search])

# OPTIONAL extra, not part of the swap: a 20b will ground its answer in the tool
# output but often won't quote the URL in its prose unless told to. This variant
# pins that habit; use it if you want the citation to show up in the final text.
agent_citing = create_agent(
    llm, [search_docs, web_search],
    system_prompt="When you used web results, always cite the source URL "
                  "(from the tool output) in your final answer.",
)

QUESTIONS = {
    "decorator": "How do I write a Python decorator?",
    "recent": "What is the latest release of LangGraph? Search the web for it.",
}


def run_agent(question: str, which=agent):
    return which.invoke({"messages": [("user", question)]})["messages"]


# --- tests -------------------------------------------------------------------
def test_web_search_is_still_a_dropin_tool():
    # Same registration the stub had: bind_tools sees no difference.
    assert web_search.name == "web_search"
    assert "query" in web_search.args
    assert "Search the open web" in web_search.description


def test_search_docs_is_unchanged():
    # The other tool still does real keyword lookup -- proof that only web_search's
    # body moved. This is offline; it needs no network or model.
    assert "spring.kafka" in search_docs.invoke({"query": "kafka batch"})
    assert search_docs.invoke({"query": "python"}) == "No matching local documents."


def test_empty_results_are_handled():
    global SEARCH_BACKEND
    original = SEARCH_BACKEND
    SEARCH_BACKEND = lambda q: []       # a backend that finds nothing
    try:
        assert web_search.invoke({"query": "obscure"}) == "No web results found."
    finally:
        SEARCH_BACKEND = original


def test_recent_answer_grounded_in_live_results(messages=None):
    # The recent question must reach web_search, and the tool output must carry a
    # real URL (not the stub's canned "(example result)"). We check the URL in the
    # TOOL message, where it provably came from the search -- asserting the final
    # prose contains a URL would test the model's citing habit, not the grounding.
    messages = messages or run_agent(QUESTIONS["recent"])
    tool_msgs = [m for m in messages if m.type == "tool" and m.name == "web_search"]
    assert tool_msgs, "the model never called web_search"
    assert "http" in str(tool_msgs[0].content)          # real URLs, not a canned string
    assert "(example result)" not in str(tool_msgs[0].content)
    assert messages[-1].content                          # and it answered afterwards


if __name__ == "__main__":
    print("Live sample (network):")
    print(web_search.invoke({"query": "LangGraph latest release"}), "\n")

    # Both questions the Gold asks for, through the untouched one-liner.
    recent_msgs = None
    for label, q in QUESTIONS.items():
        print(f"===== {label}: {q} =====")
        msgs = run_agent(q)
        for m in msgs:
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    print(f"[model] -> {tc['name']}({tc['args']})")
            elif m.type == "tool":
                print(f"[{m.name}] {str(m.content)[:76]}")
        print("answer:", msgs[-1].content, "\n")
        if label == "recent":
            recent_msgs = msgs

    test_web_search_is_still_a_dropin_tool(); print("PASS test_web_search_is_still_a_dropin_tool")
    test_search_docs_is_unchanged(); print("PASS test_search_docs_is_unchanged")
    test_empty_results_are_handled(); print("PASS test_empty_results_are_handled")
    test_recent_answer_grounded_in_live_results(recent_msgs); print("PASS test_recent_answer_grounded_in_live_results")
    print("\nall 4 tests passed")
