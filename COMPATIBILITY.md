# Compatibility

*LangGraph in Small Pieces* reports what the library **does**, measured on one
frozen set of packages, not what its documentation says it does. This file is
the list of places where those two differ, or where the book leans on something
the library does not promise -- so that a reader on a later release knows what
to re-check, and so that the book's claims can be dated.

The book is a snapshot; this file is the living part of it. When a later release
changes one of these rows, the row gets a note and a date, and the book's next
edition absorbs it.

## The frozen set

Measured against `requirements.txt` as frozen on **2026-10-22** (re-frozen from the 2026-10-12 set; see the last section):

| Package | Version |
|---|---|
| `langgraph` | 1.2.16 |
| `langchain-core` | 1.6.9 |
| `langchain` | 1.4.4 |
| `langchain-openai` | 1.7.0 |

Chapter 0's `code/ch00_setup.py` prints the installed versions; `code/preflight.py`
checks the local model setup. Neither asserts a version -- read what they print.

## Where the book and the documentation differ

| Where | The documentation says | What 1.2.16 does | Book |
|---|---|---|---|
| Default `recursion_limit` | `1000`, "starting in version 1.0.6" (Graph API guide) | `10007`, read from `langgraph._internal._config`, and overridable by the `LANGGRAPH_DEFAULT_RECURSION_LIMIT` environment variable | Ch4 Gotcha |
| `create_react_agent` | — | Still imports from `langgraph.prebuilt`; calling it warns `LangGraphDeprecatedSinceV10` ("moved to `langchain.agents`"), removal slated for LangGraph 2.0. The book uses `create_agent` | Ch11 Deep Dive |
| `SummarizationMiddleware` parameters | Many examples online still use `max_tokens_before_summary=` / `messages_to_keep=` | Those spellings raise `DeprecationWarning` and run on a fallback path; the 1.x forms are `trigger=("tokens", N)` and `keep=("messages", N)` | Ch17 |
| `FastMCP` | The SDK's current README (its 2.x line) shows `from mcp.server import MCPServer`; `FastMCP` is gone from it | On the pinned `mcp==1.29.0` the class is still `mcp.server.fastmcp.FastMCP`, and that is what the book imports. On `mcp>=2.0` the same class is `MCPServer` in `mcp.server.mcpserver`, with the same signatures, and the old path raises `ModuleNotFoundError` | Ch23 Deep Dive |

## Where the book leans on something not promised

These are implementation details of 1.2.16, shown because they explain behaviour
you will otherwise meet as a surprise. None is a public contract. Code that
depends on them is the reader's risk, and the book says so in place.

| What | Where it lives | Book |
|---|---|---|
| `RemainingSteps` is read-only: a node that returns it is ignored, with a debug log line ("wrote to unknown channel remaining_steps, ignoring it") | `langgraph.managed` | Ch4 Deep Dive |
| `default_retry_on`, the predicate `RetryPolicy` uses when `retry_on` is left off | `langgraph._internal._retry`; it is reachable as `langgraph.types.default_retry_on` only because `types` imports it, and it is in no module's `__all__` (still so on 1.2.16) | Ch19 |
| `WRITES_IDX_MAP`: the reserved write channels and their negative indices in the `writes` table (`__error__` -1, `__scheduled__` -2, `__interrupt__` -3, `__resume__` -4) | `langgraph.checkpoint.base` | Ch24 |
| The shape of the two `SqliteSaver` tables (`checkpoints`, `writes`) and the msgpack checkpoint blob | `langgraph.checkpoint.sqlite` | Ch24 |
| `checkpoint_id` is a UUIDv6, so ids sort as a timeline | `langgraph.checkpoint.base` | Ch24 |
| `Command(goto=…)` is checked declaratively only when destinations are annotated; a runtime `goto` string is not | `langgraph.types` | Ch8 |

## Behaviour that depends on the model server, not on LangGraph

| What | Observed on | Book |
|---|---|---|
| `with_structured_output(method="json_schema")` is the only method LM Studio enforces; `json_mode` and `function_calling` are rejected by the server | LM Studio 0.4.x | Ch10 |
| A reasoning model under a JSON grammar can write its reasoning into the constrained channel and leave `content` empty | LM Studio 0.4.x, gpt-oss-20b and Qwen3.8 with thinking on | Ch14 |
| `reasoning_effort="none"` turns Qwen3.8's thinking off only for the catalogue install (`lms get qwen/qwen3.8-27b`); a Hugging Face download of the same weights ignores it | LM Studio 0.4.x | Ch0, Ch10 |
| LM Studio unloads the previously JIT-loaded model when another is requested (`unloadPreviousJITModelOnLoad`); a model loaded explicitly with `lms load` is not evicted | LM Studio 0.4.24 | Ch10 |

## The 2026-10-12 re-freeze: what moved between 1.2.11 and 1.2.14

The book's set was upgraded from `langgraph 1.2.11 / langchain-core 1.5.5 /
langchain 1.3.15 / langchain-openai 1.5.1` to the versions above, and every
chapter program and every challenge answer was run again (55 model-facing
programs and 63 offline ones). Nothing the book shows changed its behaviour. What did change:

| What | Where | Effect on the book |
|---|---|---|
| A connection failure now surfaces as `OpenAIConnectionError`, a subclass of both OpenAI's `APIConnectionError` and core's new `ModelConnectionError` (core 1.6.0, "standard model exception types") | `langchain-core`, `langchain-openai` | Same failure, same ending for a turn; nothing in the book prints the class name |
| `interrupt()` takes a `response_schema` (langgraph 1.2.12) | `langgraph.types` | New capability; the book's approvals do not use it yet |
| `get_state` no longer reports interrupts that have been answered (langgraph 1.2.13), plus several `update_state` / fork fixes | `langgraph` | The book's observations about pending approvals hold; the semantics are cleaner |
| Subgraphs are detected from bytecode rather than source (langgraph 1.2.12) | `langgraph` | No change seen in the chapters that stream subgraphs |
| `langchain.mcp` namespace and `MCPAdapter` (langchain 1.4.0); importing it needs the separate `fastmcp` package, which is not in the pinned set | `langchain` | Ch23 uses `langchain-mcp-adapters`; the first-party entry point is the newer road |
| `InMemoryChatMessageHistory` deprecated (core 1.6.4, removal in 2.0; the docs point at short-term memory instead) | `langchain-core` | Not used; listed in the Class Map as deliberately skipped |
| The default `recursion_limit` is still `10007`; `WRITES_IDX_MAP`, `RemainingSteps` and `default_retry_on` are where the book says | measured | — |

## Re-checked 2026-10-19

Every row above that can be checked without a model was run again on the frozen
set (Python 3.13 this time; the numbers are the same): the recursion limit and its
environment variable, the `create_react_agent` warning text ("Deprecated in
LangGraph V1.0 to be removed in V2.0"), both `SummarizationMiddleware` spellings,
`WRITES_IDX_MAP`, the two SQLite tables and their msgpack blob, the UUIDv6
`checkpoint_id`, `interrupt(response_schema=)`, the `Command(goto=)` check with
and without the annotation, and the ignored write to `RemainingSteps`. Against LM
Studio 0.4.x with the catalogue Qwen3.8-27B: `json_mode` is refused with
`'response_format.type' must be 'json_schema' or 'text'`, `function_calling` with
`Invalid tool_choice type: 'object'`, and `json_schema` with thinking on returns
`content=''` and no parse, which `reasoning_effort="none"` fixes. Four rows were
corrected in the process: the `FastMCP` row had described the SDK's 2.x line as if
it were the pinned 1.29; `default_retry_on` is reachable through `langgraph.types`
after all, just not exported; `langchain.mcp` needs `fastmcp`; and the chapter
numbers for MCP and persistence had not followed the renumbering (Ch22 -> Ch23,
Ch23 -> Ch24).

## Changes since the freeze

*(none yet)*

## The 2026-10-22 re-freeze: what moved between 1.2.14 and 1.2.16

`langgraph` went 1.2.14 -> 1.2.16 and `langgraph-checkpoint` 4.2.0 -> 4.3.0.
Nothing else in the set moved, and nothing the book shows changed.

Both releases are bug fixes only -- no new API, nothing deprecated. Nineteen
fixes between them, and the honest count of how many land on a path this book
walks is four, with one more half-way. The test applied was not "does the book
mention this area" but "does any chapter or companion program call this API":

| Fix | The API it is about | Used in |
|---|---|---|
| A resumed `Command`'s update is applied once; an invalid resume value no longer takes the other tasks down with it (and `ToolNode` handles one properly) | `Command(resume=...)` | Ch7, Ch13, Ch17, Ch20 |
| An `update_state` input's writes land on the checkpoint it updates | `update_state` | Ch6, Ch18 |
| A cached node's writes are saved like a node that ran | `CachePolicy` | Ch19's Deep Dive |
| Four fixes to exit-mode durability: pending writes saved once, delta writes never stored under the null task id, replayed in live order, and an exit run kept out of a branch the thread moved past | `durability="exit"` | Ch6, Ch7, Ch20, Ch21, Ch24 -- though only Ch24 times the modes |
| A subgraph no longer re-runs its finished tasks on resume | subgraphs plus `interrupt()` | half: Ch9 builds subgraphs and Ch7 interrupts, but no chapter does both at once |

What the other fixes touch, the book does not:

- **`DeltaChannel`** -- six of the nineteen. The book names it once, in the
  index. It teaches channels through `BinaryOperatorAggregate` reducers.
- **`astream_events`** -- the v1/v2 control-and-interrupt fix. Neither the
  chapters nor the companion programs ever call it; Ch16's Deep Dive names it
  only to say that `graph.stream(stream_mode=...)` is what to build on.
- **`bulk_update_state`** -- zero uses anywhere.
- **`interrupt_before` under a requested drain** -- the two halves never meet:
  `interrupt_before` appears in Ch7 and Ch24, `RunControl` only in Ch20, and
  Ch20 says in place that its drain snippet is not wired into its servers.

The book does not demonstrate any of the broken behaviours, so no output block
moved. What was re-run on the new set: the 23 offline chapter programs and both
pytest suites (0 failures), `ch21_max_concurrency.py` byte-for-byte against its
recorded block, Ch21's whole program end to end, and the version-bound claims
the chapters make by name -- the `10007` recursion limit, `create_react_agent`
still importing, `default_retry_on`'s private home, `astream_events` v2 raising
no deprecation, a node `RetryPolicy` surfacing a retry as a repeated
`on_chain_start` rather than an `on_retry`, and `InjectedState`/`InjectedStore`
importing unwarned. All unchanged.
