"""Chapter 14 — Look at what the supervisor is thinking.

The supervisor's decision is a structured output. When it picks the wrong worker,
the temptation is to reword the prompt until it stops. This script does the other
thing the book keeps asking for: look. It sends the supervisor prompt to the model
the way the chapter does, then reads the model's REASONING back out and prints it
next to the answer — under six different output constraints.

gpt-oss is a *reasoning model*: it writes in channels (the "Harmony" format),
`analysis` for its thinking and `final` for the answer. LM Studio returns the
analysis text in a non-standard `reasoning` key on the message (for Qwen3.x it is
`reasoning_content`). LangChain's `ChatOpenAI` drops that key, so we go one level
down to the OpenAI client it wraps (`llm.root_client`) and read it from
`message.model_extra`.

The last two rows use the book's default model, Qwen3.8, a *hybrid*: thinking is
a per-request switch. The catalogue install (`lms get qwen/qwen3.8-27b`) is what
makes `reasoning_effort="none"` turn the switch off; a plain Hugging Face download
of the same weights has no switch.

Run:  python code/ch14_peek_reasoning.py
"""

import json

import httpx
from langchain_openai import ChatOpenAI

# Same connection block as ch14_supervisor.py. We only need its raw client.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="openai/gpt-oss-20b",
    temperature=0,
)
# The openai.OpenAI object ChatOpenAI talks through -- with a timeout, because
# one of the states below can stop coming back. Under the json_schema grammar
# gpt-oss-20b answers three of the four decision states in under two seconds and
# never returns on the fourth (observed on this local build: 3 trials at 180s,
# 3 more at 120s). ChatOpenAI builds this client with timeout=None (a bare
# OpenAI() would default to ~10 min), so without an explicit one the probe hangs
# with no output and nothing to read. A timeout turns that into a result you can see.
client = llm.root_client.with_options(
    timeout=httpx.Timeout(90.0), max_retries=0)

GPT_OSS = "openai/gpt-oss-20b"           # reasoning model (Harmony channels)
QWEN = "qwen/qwen3.8-27b"                # the book's default: a hybrid, thinking on/off per request


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


# The chapter's Delegate schema, field descriptions included, as the JSON Schema
# with_structured_output sends it. `reason` first, `route` second — the order the
# model fills them in.
DELEGATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "Think step by step about which rule applies, then decide.",
        },
        "route": {
            "type": "string",
            "enum": ["docs_agent", "web_agent", "finish"],
            "description": "Which worker to delegate to next: "
            "docs_agent=search the internal documentation, "
            "web_agent=search the open web, "
            "finish=stop delegating and write the final answer.",
        },
    },
    "required": ["reason", "route"],
    "additionalProperties": False,
}

# The same schema packaged as a tool. With tools, the grammar constrains only the
# tool ARGUMENTS; the model's thinking channel runs before it, unconstrained.
DELEGATE_TOOL = {
    "type": "function",
    "function": {
        "name": "Delegate",
        "description": "Pick the next worker for a documentation supervisor, or finish.",
        "parameters": DELEGATE_SCHEMA,
    },
}


def peek(prompt: str, *, model: str = GPT_OSS, schema: dict | None = None,
         tools: list | None = None, reasoning_effort: str | None = None) -> tuple[str, str]:
    """One raw chat call. Returns (reasoning, answer) — reasoning is '<none>' if absent.

    schema -> response_format=json_schema, what with_structured_output(method=
             "json_schema") sends; the server applies a grammar to the output.
    tools  -> tool calling with tool_choice="required": the model MUST call the one
             tool, so the route still can't be a stray sentence. (LM Studio rejects
             the {"type": "function", ...} form of tool_choice, which is why the
             string form is used here.)
    reasoning_effort -> "none" switches a hybrid model's thinking off for this
             request; leaving it unset keeps the model's default (on).
    """
    kwargs: dict = {"model": model, "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}]}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "Delegate", "strict": True, "schema": schema},
        }
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "required"
    try:
        msg = client.chat.completions.create(**kwargs).choices[0].message
    except httpx.TimeoutException:
        return "<none>", "<no response within the timeout>"
    except Exception as exc:                      # APITimeoutError wraps the above
        if "Timeout" not in type(exc).__name__:
            raise
        return "<none>", "<no response within the timeout>"
    # `reasoning` is not in the OpenAI spec, so the SDK parks it in model_extra.
    # gpt-oss comes back under "reasoning", Qwen3.x under "reasoning_content".
    extra = msg.model_extra or {}
    reasoning = extra.get("reasoning") or extra.get("reasoning_content") or "<none>"
    if msg.tool_calls:
        answer = msg.tool_calls[0].function.arguments
    else:
        answer = msg.content or ""
    return reasoning.strip(), answer.strip()


# The four decision states the supervisor passes through in the chapter's run,
# each with the route the ordered rules say is right.
PY_Q = "How do I write a Python decorator?"
KAFKA_Q = "How do I configure Kafka batch consumers?"
# Reports are quoted verbatim from a real ch14_supervisor.py run, so this probe
# sees exactly what the supervisor sees. (The web one drifts with the web.)
DOCS_EMPTY = "- [docs_agent] nothing useful found"
WEB_HIT = (
    "- [web_agent] To write a Python decorator, you define an outer function that "
    "takes the target function as an argument, create an inner wrapper function "
    "inside it (which typically calls the original function and can modify its "
    "behavior), and return that wrapper."
)
DOCS_HIT = (
    "- [docs_agent] To configure Kafka batch consumers, set "
    "`spring.kafka.listener.type=batch` and tune the `max-poll-records` setting "
    "as needed."
)
STATES = [
    ("Python, nothing run", PY_Q, False, False, "(none yet)", "docs_agent"),
    ("Python, docs empty", PY_Q, True, False, DOCS_EMPTY, "web_agent"),
    ("Python, both run", PY_Q, True, True, DOCS_EMPTY + "\n        " + WEB_HIT, "finish"),
    ("Kafka, docs hit", KAFKA_Q, True, False, DOCS_HIT, "finish"),
]


# The chapter's supervisor prompt WITHOUT its last line — the original wording.
def original_prompt(question: str, docs: bool, web: bool, reports_text: str) -> str:
    return f"""
        You are the SUPERVISOR of a documentation question-answering team
        with two workers. You delegate to ONE worker at a time, read its report, then
        decide again.

        Question: {question}
        Already delegated to docs_agent: {docs}
        Already delegated to web_agent: {web}

        Worker reports so far:
        {reports_text}

        Read the rules in order and choose the FIRST whose condition is true:
        1. If docs_agent has NOT run yet                                  -> docs_agent
        2. else if the docs came back empty AND web_agent has NOT run yet -> web_agent
        3. otherwise                                                      -> finish
        """


SCRATCHPAD_LINE = """
        In `reason`, first repeat the two "Already delegated" values, then state which
        rule applied."""

UNCONSTRAINED_LINE = """
        Answer with one JSON object with keys "reason" and "route"."""


def run(title: str, *, extra: str = "", **peek_kwargs) -> None:
    """Run the three states under one constraint and tally the routes."""
    print(f"\n== {title} ==")
    ok = 0
    for label, q, docs, web, reports_text, expected in STATES:
        reasoning, answer = peek(original_prompt(q, docs, web, reports_text) + extra,
                                 **peek_kwargs)
        if not answer:                         # nothing in `content` at all
            answer = "<empty>"
        try:                                   # parse the JSON the model returned
            parsed = json.loads(answer)
            answer = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
            route = parsed.get("route", "?")
        except ValueError:                     # not JSON at all: prose, no route
            route = "?"
        mark = "OK" if route == expected else "WRONG"
        ok += mark == "OK"
        note = ""
        if answer == "<empty>":                # did the JSON land in the thinking channel?
            try:
                note = f"  (JSON is in the reasoning channel; route there: {json.loads(reasoning).get('route', '?')})"
            except ValueError:
                pass
        print(f"  {label:<20} reasoning: {excerpt(reasoning, 60)}")
        print(f"  {'':<20} answer:    {excerpt(answer, 60)}  -> {route} [{mark}]{note}")
    print(f"  {ok}/{len(STATES)} routes correct")


if __name__ == "__main__":
    run("(a) gpt-oss, no constraint", extra=UNCONSTRAINED_LINE)
    run("(b) gpt-oss, json_schema", schema=DELEGATE_SCHEMA)
    run("(c) gpt-oss, json_schema + scratchpad line", schema=DELEGATE_SCHEMA,
        extra=SCRATCHPAD_LINE)
    run("(d) gpt-oss, tool calling, tool_choice=required", tools=[DELEGATE_TOOL])
    run("(e) qwen3.8-27b, json_schema, thinking off", model=QWEN,
        schema=DELEGATE_SCHEMA, reasoning_effort="none")
    run("(f) qwen3.8-27b, json_schema, thinking on", model=QWEN,
        schema=DELEGATE_SCHEMA)
