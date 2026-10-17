# Challenge answers

Worked solutions to the end-of-chapter **Challenge** sections. Each file is
self-contained: it holds the solution *and* its tests, so running the file both
demonstrates the idea and checks that it still works.

## How to run

From the repo root, in the companion virtualenv the [root
README](../../README.md#setup) sets up (`.venv` — Python 3.11, LangGraph 1.2.11):

```sh
python code/answers/ch01_typeddict_no_validation.py
```

Every file runs a short demo, then executes its `test_*` assertions and prints
`all N tests passed ✅`. No test framework is required — the assertions are plain
Python, and **running the file is how an answer proves itself**.

`pytest code/answers/` is not that. pytest collects from `test_*.py` and
`*_test.py`, and these are named `chNN_*.py`, so it finds one file out of
forty-odd and reports a confident green for the rest having never imported them.
Widening pytest's pattern would be worse than leaving it: most answers here
reach for a local model server, so a broadened offline run would fail for
reasons that say nothing about the answers. Run the file you want, or run them
all through `python tools/verify_samples.py --only answers`, which knows which
ones need a model and says so instead of skipping quietly.

Where a chapter's solution involves a model, the answer file calls the **same
real, locally-served models as the chapter** (LM Studio at `localhost:1234`;
`qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, the nomic embeddings, and —
for the "weak model" Bronze — `qwen2.5-vl-7b-instruct`). Start the server and
load nothing in advance; it JIT-loads models per request. Because the model text
varies run to run, those files assert *structure* — paths taken, counts, what
was stored where — never verbatim wording. Files from the model-free chapters
still run fully offline.

## Index

| File | Chapter | Tier | Demonstrates |
|------|---------|------|--------------|
| `ch01_typeddict_no_validation.py` | 1 | 🥉 | `TypedDict` state is annotation-only: no runtime validation |
| `ch01_pydantic_state.py`          | 1 | 🥈 | A Pydantic state validates inputs at `invoke()` |
| `ch02_break_the_order.py`         | 2 | 🥈 | Edges encode data dependencies; a bad order fails only at read time |
| `ch03_three_way_route.py`         | 3 | 🥉 | Add a third branch to a conditional edge |
| `ch03_safe_router.py`             | 3 | 🥈 | An unmapped route raises `KeyError`; guard it with a default |
| `ch04_trajectory_log.py`          | 4 | 🥉 | Record the route taken each pass (read-and-append, no reducer) |
| `ch04_budget_degrade.py`          | 4 | 🥈 | A pass budget with a graceful, labelled partial answer |
| `ch05_tool_priority.py`           | 5 | 🥈 | Data-driven tool fallback: walk a priority list, not an if/elif ladder |
| `ch06_thread_isolation.py`        | 6 | 🥉 | Two `thread_id`s stay isolated across runs |
| `ch06_resume_no_redo.py`          | 6 | 🥈 | Resume doesn't re-run completed nodes (proven with call counters) |
| `ch06_durability_modes.py`        | 6 | 🥇 | Snapshot counts under `durability="sync"/"async"/"exit"`, and what an exception leaves under `"exit"` |
| `ch07_rich_resume.py`             | 7 | 🥉 | `Command(resume=…)` can carry a structured decision, not just a string |
| `ch07_serial_gates.py`            | 7 | 🥈 | Two serial `interrupt()` gates on one thread |
| `ch08_defer_join.py`              | 8 | 🥉 | A plain join fires `merge` twice; both declared joins fix it — and only one survives a runtime-chosen fan-out |
| `ch08_custom_reducer.py`          | 8 | 🥈 | A custom `union` reducer deduplicates parallel writes |
| `ch09_see_inside_node.py`         | 9 | 🥉 | `stream(..., subgraphs=True)` exposes the inner steps of a subgraph node |
| `ch09_schema_adapter.py`          | 9 | 🥈 | A wrapper node bridges mismatched parent/child schemas |
| `ch10_weak_structured.py`         | 10 | 🥉 | Structured output guarantees *shape* (a legal enum) but not *choice*: a real 7B ignores a full result and loops into the recursion fuse |
| `ch10_doc_grader.py`              | 10 | 🥈 | A second `with_structured_output` grades doc relevance (docs in the prompt, judged by the real model) and routes on judgment, not a count |
| `ch11_convention_over_configuration.py` | 11 | 🥉 | tools_condition routes by returning the string "tools": rename the node to break it at compile(), buy the name back with a path map |
| `ch11_forced_tool.py`             | 11 | 🥈 | Reach vs. control: a system prompt raises tool compliance but only a graph-node edge guarantees it |
| `ch11_real_web_search.py`         | 11 | 🥇 | Live DuckDuckGo behind the same @tool; the contract and the loop never change. **Goes online** (`pip install ddgs`) — the one networked file here |
| `ch12_validating_state.py`        | 12 | 🥉 | A `BaseModel` rejects bad input at the door; a `@dataclass` waves it through to fail deep in a node |
| `ch12_schema_windows.py`          | 12 | 🥈 | input/output schemas are windows, not guards: the return is filtered, smuggled keys and no-schema writes vanish silently, a missing key fails mid-run |
| `ch13_task_boundary.py`           | 13 | 🥉 | A `@task` called from module scope raises; the message names the missing runnable context |
| `ch13_functional_retry.py`        | 13 | 🥈 | Output-based `@task` retry; a call counter proves attempts replay from cache on resume |
| `ch14_subgraph_worker.py`         | 14 | 🥉 | A worker becomes a real isolated subgraph; its private `rewritten` never reaches the supervisor |
| `ch14_investigator.py`            | 14 | 🥈 | A budgeted investigation loop with a programmatic delegation guard (not a prompt suggestion); the richer 3-field plan needs the qwen router — gpt-oss mis-routes it |
| `ch15_learned_facts.py`           | 15 | 🥈 | A `remember` node writes a user-stated preference to the Store; a later thread recalls it |
| `ch15_semantic_recall.py`         | 15 | 🥇 | An embedding index makes the Store rank by meaning; without one `query=` is ignored in silence |
| `ch16_values_snapshot.py`         | 16 | 🥉 | `stream_mode="values"` yields whole-state snapshots — one more than `updates` |
| `ch16_durable_progress.py`        | 16 | 🥈 | Bullhorn vs logbook: `custom` vanishes, `State` progress survives the stream |
| `ch16_two_streams.py`             | 16 | 🥇 | Route one `messages` stream across two model nodes by `langgraph_node` |
| `ch19_prove_the_deny_list.py`     | 19 | 🥉 | `RetryPolicy(retry_on=…)` is an allow-list: a narrow policy runs a `ValueError` once and raises; widening it lets the third attempt succeed |
| `ch19_idempotent_retry.py`        | 19 | 🥈 | A guard flag makes a node's side effect fire once even when `RetryPolicy` retries it |
| `ch19_combined_recovery.py`       | 19 | 🥇 | One graph composes `RetryPolicy` + catch-and-fallback + graceful degradation |
| `ch17_timing_middleware.py`       | 17 | 🥈 | `wrap_model_call`/`wrap_tool_call` time every model and tool call |
| `ch20_durable_restart.py`         | 20 | 🥈 | A run survives a process restart once `InMemorySaver` becomes a `SqliteSaver` on a file; needs `langgraph-checkpoint-sqlite` |
| `ch20_shared_checkpointer_redis.py` | 20 | 🥇 | Two replicas share one run: separate `InMemorySaver`s 404, a shared Redis saver serves it; needs `langgraph-checkpoint-redis` + a Redis 8/Stack |
| `ch21_serial_to_concurrent.py`    | 21 | 🥉 | One node, three waits: awaited in turn they add up, handed to `gather` they take the longest; asserted as a ratio |
| `ch21_unblock_the_tool.py`        | 21 | 🥈 | A blocking legacy tool stalls every run on the loop; `asyncio.to_thread` gives the overlap back, and a `Semaphore` caps calls in flight |
| `ch21_cancel_and_resume.py`       | 21 | 🥇 | `asyncio.wait_for` cancels a run mid-node; `AsyncSqliteSaver` kept the finished steps, and `ainvoke(None, config)` finishes without repeating them |
| `ch25_prove_semantic_retrieval.py` | 25 | 🥈 | One added document, no rule changes: the question flips from web-fallback to a local hit — and phrasings sharing no content word reach it, while ones with no lexical hook at all fall just short |
| `ch25_prompt_injection.py`        | 25 | 🥇 | Plant a hostile document, score the injection with a canary, and grade four prompt defenses — the finding is the variance, not the winner |
| `ch26_structural_retrieval.py`    | 26 | 🥈 | Markdown structural chunks carry a heading path, so retrieval returns a citable source |
| `ch27_union_original.py`          | 27 | 🥉 | Union the original query with the rewrites: a zero-cost recall floor — on the real stack the drift shows up as similarity-margin erosion, and the union keeps the raw margin |
| `ch27_floor_and_cap.py`           | 27 | 🥈 | A rerank keep-rule with a floor and a cap; the floor is what empties a junk shortlist |
| `ch30_catch_regression.py`        | 30 | 🥉 | A one-line retrieval regression drops the hit-rate under the gate, which fails the build |
| `ch30_judge_wobble.py`            | 30 | 🥈 | Measure the LLM-judge's run-to-run wobble; a larger eval set shrinks the spread |
| `ch30_independent_judge.py`       | 30 | 🥇 | Swap the judge to an independent model (Gemini); it splits from the self-judge only on borderline answers, catching invented keys the self-judge waves through |
| `ch31_path_budget.py`             | 31 | 🥉 | Same answer, three verdicts: the allow-list catches the detour, the budget catches the dithering, each blind to the other |

(Chapter 2's 🥉 Bronze is a one-line exploration — `graph.get_graph().draw_mermaid()`
— so it has no answer file.)
