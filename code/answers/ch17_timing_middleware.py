"""Chapter 17 Challenge -- Silver: a timing middleware.

Chapter 22's `traced()` decorator timed *nodes*. The harness version stands on
the two wrap hooks: `wrap_model_call` and `wrap_tool_call` receive the request
plus a `handler` to forward to, so a timer is a clock read on either side of
`handler(request)`. Rows collect on the middleware instance itself: timing is
bookkeeping about the run, not agent state the model should ever see.

Run:  python code/answers/ch17_timing_middleware.py   (see answers/README.md)
"""

import time

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
                 model="qwen/qwen3.8-27b", temperature=0, reasoning_effort="none")


@tool
def search_docs(query: str) -> str:
    """Search the local documentation for configuration answers."""
    return "Kafka batch consumption: set spring.kafka.listener.type=batch."


class TimingMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.rows: list[tuple[str, str, float]] = []

    def _timed(self, kind: str, name: str, request, handler):
        t0 = time.perf_counter()
        try:
            return handler(request)
        finally:
            self.rows.append((kind, name, (time.perf_counter() - t0) * 1000))

    def wrap_model_call(self, request, handler):
        return self._timed("model", getattr(request.model, "model_name", "model"),
                           request, handler)

    def wrap_tool_call(self, request, handler):
        return self._timed("tool", request.tool_call["name"], request, handler)


def test_every_call_got_a_row():
    timing = TimingMiddleware()
    agent = create_agent(llm, [search_docs], middleware=[timing])
    result = agent.invoke({"messages": [("user",
        "What is the config key for Kafka batch mode? Search the docs.")]})
    kinds = [k for k, _, _ in timing.rows]
    assert kinds.count("tool") >= 1                    # the search ran, timed
    assert kinds.count("model") == kinds.count("tool") + 1   # one extra call to answer
    assert all(ms > 0 for _, _, ms in timing.rows)
    return timing, result


if __name__ == "__main__":
    timing, result = test_every_call_got_a_row()
    print(f"{'kind':6} {'name':34} {'ms':>8}")
    for kind, name, ms in timing.rows:
        print(f"{kind:6} {name:34} {ms:8.1f}")
    print("\nPASS test_every_call_got_a_row")
