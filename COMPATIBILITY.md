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

Measured against `requirements.txt` as frozen on **2026-10-12** (re-frozen from the 2026-10-08 set; see the last section):

| Package | Version |
|---|---|
| `langgraph` | 1.2.14 |
| `langchain-core` | 1.6.9 |
| `langchain` | 1.4.4 |
| `langchain-openai` | 1.7.0 |

Chapter 0's `code/ch00_setup.py` prints the installed versions; `code/preflight.py`
checks the local model setup. Neither asserts a version -- read what they print.

## Where the book and the documentation differ

| Where | The documentation says | What 1.2.14 does | Book |
|---|---|---|---|
| Default `recursion_limit` | `1000`, "starting in version 1.0.6" (Graph API guide) | `10007`, read from `langgraph._internal._config`, and overridable by the `LANGGRAPH_DEFAULT_RECURSION_LIMIT` environment variable | Ch4 Gotcha |
| `create_react_agent` | — | Still imports from `langgraph.prebuilt`; calling it warns `LangGraphDeprecatedSinceV10` ("moved to `langchain.agents`"), removal slated for LangGraph 2.0. The book uses `create_agent` | Ch11 Deep Dive |
| `SummarizationMiddleware` parameters | Many examples online still use `max_tokens_before_summary=` / `messages_to_keep=` | Those spellings raise `DeprecationWarning` and run on a fallback path; the 1.x forms are `trigger=("tokens", N)` and `keep=("messages", N)` | Ch17 |
| `FastMCP` | Older material imports `mcp.server.fastmcp.FastMCP` | Renamed `MCPServer`, moved to `mcp.server.mcpserver`; the old path raises `ModuleNotFoundError` pointing at a migration guide | Ch22 |

## Where the book leans on something not promised

These are implementation details of 1.2.14, shown because they explain behaviour
you will otherwise meet as a surprise. None is a public contract. Code that
depends on them is the reader's risk, and the book says so in place.

| What | Where it lives | Book |
|---|---|---|
| `RemainingSteps` is read-only: writing it is ignored and logged | `langgraph.managed` | Ch4 Deep Dive |
| `default_retry_on`, the predicate `RetryPolicy` uses when `retry_on` is left off | `langgraph._internal._retry` -- nothing public exports it (still there on 1.2.14) | Ch19 |
| `WRITES_IDX_MAP`: the reserved write channels (`__error__` = -1, `__interrupt__`, …) and their negative indices in the `writes` table | `langgraph.checkpoint.base` | Ch23 |
| The shape of the two `SqliteSaver` tables (`checkpoints`, `writes`) and the msgpack checkpoint blob | `langgraph.checkpoint.sqlite` | Ch23 |
| `checkpoint_id` is a UUIDv6, so ids sort as a timeline | `langgraph.checkpoint.base` | Ch23 |
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
| A connection failure now surfaces as `OpenAIConnectionError`; on core 1.5 it was the provider's `APIConnectionError` (core 1.6.0, "standard model exception types") | `langchain-core`, `langchain-openai` | Same failure, same ending for a turn; one transcript line in Ch31 shows the old name |
| `interrupt()` takes a `response_schema` (langgraph 1.2.12) | `langgraph.types` | New capability; the book's approvals do not use it yet |
| `get_state` no longer reports interrupts that have been answered (langgraph 1.2.13), plus several `update_state` / fork fixes | `langgraph` | The book's observations about pending approvals hold; the semantics are cleaner |
| Subgraphs are detected from bytecode rather than source (langgraph 1.2.12) | `langgraph` | No change seen in the chapters that stream subgraphs |
| `langchain.mcp` namespace and `MCPAdapter` (langchain 1.4.0) | `langchain` | Ch22 uses `langchain-mcp-adapters`; the first-party entry point is the newer road |
| `ChatMessageHistory` deprecated (core 1.6.4) | `langchain-core` | Not used; listed in the Class Map as deliberately skipped |
| The default `recursion_limit` is still `10007`; `WRITES_IDX_MAP`, `RemainingSteps` and `default_retry_on` are where the book says | measured | — |

## Changes since the freeze

*(none yet)*
