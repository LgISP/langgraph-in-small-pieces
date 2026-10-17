"""Chapter 22 Challenge — Gold: take the choosing away.

Every duration in the chapter was picked by hand. `web_search` is the widest bar
because its stub sleeps half a second, and it sleeps half a second because that is
what a web search feels like. Fine for learning to read a waterfall, and circular
as evidence: the advice is "optimise the fattest bar" on a graph whose fattest bar
was chosen by the author.

So this file keeps the tracer and the wiring exactly as they are and replaces the
four stubs with the real thing -- `agent` and `generate` call the local model from
@sec-real-llm, `search_docs` queries the Chroma store from @sec-chroma, every
`time.sleep` deleted -- then runs the same question several times and reads the
waterfalls side by side.

**What comes out**, on the machine this was written on (LM Studio, Qwen3.8,
nomic embeddings, four runs):

    generate     ~8000 ms    spread   0%
    agent        ~1170 ms    spread   3%
    search_docs   ~260 ms    spread 335%   (one cold call at ~900 ms, then ~40 ms)

Three things, and two of them are not what the chapter's stubs would have led you
to expect.

**1. The ranking stops being a matter of taste.** `generate` beats `search_docs`
by about 30x. The hand-picked graph had its widest bar ahead by about 2x. "Optimise
the fattest bar" is cheap advice on a real graph because there is only one bar --
the interesting question is not which node to optimise but whether to call the
model at all, which is what @sec-golden-layer is about.

**2. The "local vector search" is mostly a model call.** A 40 ms warm call
sounds like a database; it is not. Timed separately (this file does it),
embedding the query takes ~11ms over HTTP and the actual vector math takes
~0.6ms: **95% of a warm "local vector search" is a round-trip to the embedding
model**, and the cold first call — the 900 ms one — is the embedding model
warming up. A retrieval node's cost is an embedding model's cost, and no amount
of index tuning touches it.

**3. Relative spread runs backwards.** The eight-second node is the *stable* one
(0%); the quarter-second node swings 335%. There is nothing paradoxical about it: jitter is
roughly a fixed number of milliseconds -- scheduling, HTTP, a model warming up --
so it is a rounding error on a big bar and most of a small one. The practical
consequence is about alerting: a percentage threshold ("warn if this node is 50%
slower than usual") is a noise generator on your cheap nodes and nearly blind on
your expensive one. Small bars want absolute thresholds; big bars want percentages.

Needs: LM Studio at :1234, and the Chroma store built by ch25_rag_chroma.py.
Run:  python code/answers/ch22_real_waterfall.py   (see answers/README.md)
"""

import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ch22_observability as ch21    # noqa: E402

LM_BASE = "http://localhost:1234/v1"
DB_DIR = str(Path(__file__).resolve().parents[1] / "chroma_ch25")
QUESTION = "How do I configure Kafka batch consumers?"
RUNS = 4


def server_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{LM_BASE}/models", timeout=2):
            return True
    except urllib.error.HTTPError:
        return True        # a status code is a server: a hosted endpoint answers 401 to a key-less probe
    except (urllib.error.URLError, OSError):
        return False


def store_exists() -> bool:
    return Path(DB_DIR).exists()


# --------------------------------------------------------------------------- #
# The same tracer, the same wiring, real work inside
# --------------------------------------------------------------------------- #

# Reuse the chapter's own tracer rather than redefining a weaker copy. `ch21.traced`
# records into `ch21.TRACE` inside a `finally`, with an `error` field, so a node that
# RAISES still leaves a trace -- the failure blind spot an earlier draft of this file
# reintroduced by wrapping without a `finally`.
traced = ch21.traced


def build_real_graph():
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langgraph.graph import END, START, StateGraph

    llm = ChatOpenAI(base_url=LM_BASE, api_key="lm-studio",
                     model="qwen/qwen3.8-27b", temperature=0, reasoning_effort="none")
    store = Chroma(collection_name="doc_bot", persist_directory=DB_DIR,
                   embedding_function=OpenAIEmbeddings(
                       base_url=LM_BASE, api_key="lm-studio",
                       model="text-embedding-nomic-embed-text-v1.5",
                       check_embedding_ctx_length=False))

    def agent_node(state):
        """One real model call to decide the route -- no stub, no coin flip."""
        verdict = llm.invoke(
            "Answer with one word, 'search' or 'generate'. Does this question need "
            f"the internal technical docs? Question: {state['question']}").content
        route = "search" if "search" in verdict.lower() else "generate"
        return {"route": route if state.get("iterations", 0) == 0 else "generate",
                "iterations": state.get("iterations", 0) + 1}

    def search_docs_node(state):
        hits = store.similarity_search(state["question"], k=2)
        return {"docs": [d.page_content for d in hits]}

    def web_search_node(state):
        return {"web_results": []}          # unreachable here; kept for wiring parity

    def generate_node(state):
        docs = "\n".join(state.get("docs", []))
        answer = llm.invoke(f"Context:\n{docs}\n\nQuestion: {state['question']}").content
        return {"answer": answer}

    builder = StateGraph(ch21.State)
    for fn in (agent_node, search_docs_node, web_search_node, generate_node):
        builder.add_node(fn.__name__.removesuffix("_node"), traced(fn))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", ch21.route_after_agent,
        {"search": "search_docs", "web": "web_search", "generate": "generate"},
    )
    builder.add_edge("search_docs", "agent")
    builder.add_edge("web_search", "agent")
    builder.add_edge("generate", END)
    return builder.compile()


def one_run(graph) -> dict[str, float]:
    """Total ms per node for a single run."""
    ch21.TRACE.clear()
    graph.invoke({"question": QUESTION})
    totals: dict[str, float] = {}
    for t in ch21.TRACE:
        totals[t["node"]] = totals.get(t["node"], 0.0) + t["dur_ms"]
    return totals


def time_the_search_split() -> tuple[float, float]:
    """How much of the search node is the embedding call, warm, measured."""
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    emb = OpenAIEmbeddings(base_url=LM_BASE, api_key="lm-studio",
                           model="text-embedding-nomic-embed-text-v1.5",
                           check_embedding_ctx_length=False)
    store = Chroma(collection_name="doc_bot", persist_directory=DB_DIR,
                   embedding_function=emb)
    for _ in range(2):                                   # warm both sides
        store.similarity_search_by_vector(emb.embed_query(QUESTION), k=2)
    es, vs = [], []
    for _ in range(5):
        t = time.perf_counter(); vec = emb.embed_query(QUESTION)
        es.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter(); store.similarity_search_by_vector(vec, k=2)
        vs.append((time.perf_counter() - t) * 1000)
    return statistics.mean(es), statistics.mean(vs)


_CACHE: list = []


def cached_runs() -> list[dict[str, float]]:
    if not _CACHE:
        graph = build_real_graph()
        _CACHE.append([one_run(graph) for _ in range(RUNS)])
    return _CACHE[0]


def demo_real_waterfall():
    if not (server_is_up() and store_exists()):
        print(f"model server at {LM_BASE}: {'up' if server_is_up() else 'DOWN'}")
        print(f"chroma store {DB_DIR}: {'present' if store_exists() else 'MISSING'}")
        print("\nBoth are needed. Start LM Studio, and run ch25_rag_chroma.py once")
        print("to build the store. Nothing here is faked in their absence -- the")
        print("whole point of this challenge is that the numbers are not chosen.")
        return

    runs = cached_runs()
    nodes = sorted({n for r in runs for n in r}, key=lambda n: -statistics.mean(
        [r.get(n, 0.0) for r in runs]))

    print(f"\n{'node':<14}" + "".join(f"{'run ' + str(i + 1):>11}" for i in range(len(runs)))
          + f"{'mean':>11}{'spread':>9}")
    print("-" * (14 + 11 * len(runs) + 20))
    for n in nodes:
        vals = [r.get(n, 0.0) for r in runs]
        mean = statistics.mean(vals)
        spread = (max(vals) - min(vals)) / mean if mean else 0
        print(f"{n:<14}" + "".join(f"{v:>10.1f}m" for v in vals)
              + f"{mean:>10.1f}m{spread:>8.0%}")

    embed_ms, vec_ms = time_the_search_split()
    print(f"\nInside search_docs: embedding the query {embed_ms:.1f}ms, "
          f"vector math {vec_ms:.1f}ms")
    print(f"  -> {embed_ms / (embed_ms + vec_ms):.0%} of a warm 'local vector search'")
    print("     is a round-trip to a model. The index is not what costs you.")

    slow, fast = nodes[0], nodes[-1]
    ratio = statistics.mean([r.get(slow, 0) for r in runs]) / max(
        1e-9, statistics.mean([r.get(fast, 0) for r in runs]))
    print(f"\nWidest: {slow}. It beats {fast} by {ratio:,.0f}x -- the chapter's")
    print("hand-picked graph had the widest bar ahead by about 2x.")
    print("\nNow read the spread column backwards: the biggest node is the steady")
    print("one and the smallest node swings wildly. Jitter is a fixed number of")
    print("milliseconds, so it is a rounding error on a big bar and most of a small")
    print("one -- which is why percentage alerts belong on your expensive nodes and")
    print("absolute ones on your cheap nodes.")


# --- tests -------------------------------------------------------------------
# Skipped (not failed) when the server or the store is absent: an answer file
# that cannot reach its dependencies has nothing to say, and saying it loudly
# would be worse than saying nothing.

def _need_env():
    if not server_is_up():
        raise Skip(f"no model server at {LM_BASE}")
    if not store_exists():
        raise Skip(f"no chroma store at {DB_DIR} (run ch25_rag_chroma.py once)")


class Skip(Exception):
    pass


# The waterfall RATIOS (a model call vs a vector search, the spread being worse on
# the cheap node, embedding vs vector math) are the point of this exercise, but they
# are MEASUREMENTS on one machine, not invariants: a faster box or a different
# provider can reorder them without anything being wrong. `demo_real_waterfall()`
# above prints them as observations. The tests below assert only what must hold on
# every machine -- the structure -- so a reader whose numbers differ still passes.

def test_no_sleep_survives_in_the_real_graph():
    """The stubs are gone, not shortened."""
    import inspect
    src = inspect.getsource(build_real_graph)
    assert "sleep(" not in src        # the word survives in a docstring; the call does not


def test_every_node_still_reports_into_the_same_logbook():
    """The tracer is the chapter's own: same six fields, real work inside."""
    _need_env()
    cached_runs()
    assert ch21.TRACE
    assert set(ch21.TRACE[0]) == {"node", "thread_id", "start_ms", "dur_ms", "output_keys", "error"}


def test_a_failing_node_still_leaves_a_trace():
    """The reason to reuse ch21.traced: its `finally` records a step that RAISED.
    Offline -- no server needed."""
    ch21.TRACE.clear()

    @traced
    def boom_node(state):
        raise ValueError("kaboom")

    try:
        boom_node({"question": "x"})
    except ValueError:
        pass
    assert ch21.TRACE and ch21.TRACE[-1]["node"] == "boom_node"
    assert ch21.TRACE[-1]["error"] and "ValueError" in ch21.TRACE[-1]["error"]


if __name__ == "__main__":
    demo_real_waterfall()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    passed = skipped = failed = 0
    print()
    for t in _tests:
        try:
            t()
            print("PASS", t.__name__); passed += 1
        except Skip as e:
            print("SKIP", t.__name__, f"({e})"); skipped += 1
        except AssertionError as e:
            print("FAIL", t.__name__, f"({e})"); failed += 1
    # Real tally: skipped tests are not passed tests (an absent server must not read
    # as green), and the machine-specific ratios are observations, not assertions.
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    if failed:
        sys.exit(1)
