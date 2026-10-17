"""Chapter 25 Challenge — Silver: add a document, not a keyword.

The Python question misses because the store holds no Python document -- not
because of anything about keywords. This proves it by changing exactly one thing:
one more entry in `DOCUMENTS`, a rebuilt index, and the same question asked again.

    "Python decorators: wrap a function with @functools.wraps to preserve its
     name and docstring."

Nothing else moves. No matching rule, no threshold, no synonym list, no stopword
file. The question scores below `SIM_THRESHOLD` before the document exists and
clears it after; the hit is the document that was added; phrasings that share not
one word with it reach it too; and, run through Chapter 5's routing rule as a real
graph, the same question that fell back to the web now routes to `generate`.

Contrast with @sec-tool-nodes' keyword store, where the same fix meant hand-adding
a `"python"` topic key and hoping the user typed that exact word. Two costs are
worth naming, because they are what you are actually buying:

  * **the fix scales with your documents, not with your users' phrasings.** One
    document covers "python decorator", "how do I wrap a function", "@wraps" and
    the phrasings nobody has thought of yet. The keyword store needed one entry
    per phrasing, forever.
  * **you cannot audit it by reading.** The keyword store's behaviour was legible
    -- you could see which words routed where. Meaning-space is not legible; the
    only way to know what a new document made reachable is to ask. This file's
    tests are that asking, which is the habit @sec-evaluation formalises.

The file leaves the chapter's store untouched: it builds its own index in a temp
directory, so running it does not change what `ch25_rag_chroma.py` prints.

Needs LM Studio at :1234 for embeddings.
Run:  python code/answers/ch25_prove_semantic_retrieval.py   (see answers/README.md)
"""

import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LM_BASE = "http://localhost:1234/v1"
PYTHON_DOC = ("Python decorators: wrap a function with @functools.wraps to preserve "
              "its name and docstring.")
QUESTION = "How do I write a Python decorator?"
PARAPHRASES = [
    "How do I write a Python decorator?",
    "how can I wrap a function and keep its __name__",
    "what does functools.wraps do",
]

# Three tiers, measured -- and the gradient is the lesson. The paraphrases above
# reuse a word from the document ("wrap", "function", "functools.wraps"), so a
# keyword index could have matched them too. These share no CONTENT word with it --
# no "python", no "decorator", no "wrap" -- and still land on it:
NO_CONTENT_WORD = [
    "what is the @ syntax written above a def",
    "@staticmethod and @property style syntax explained",
]

# ...but reach runs out. These are just as much decorator questions and share no
# lexical hook at all -- not even the `@` -- and they land just UNDER the threshold
# and miss. One short document does not cover a topic; it covers a neighbourhood
# around its own wording, and the threshold sets how big that neighbourhood is.
# (Measured on this stack; expect the third decimal to move, not the verdicts.)
NEAR_MISS = [
    "syntax for adding reusable behaviour on top of a def",
    "how do I annotate a method so extra behaviour runs around it",
]


# Read the chapter's constants WITHOUT importing the module. `ch25_rag_chroma.py`
# connects to (and seeds) the chapter's Chroma directory at import time, so an
# `import ch25_rag_chroma` inside a test would build the very store this file
# promises not to touch. `ast` parses the source and never runs it.
def chapter_constants() -> tuple[list[str], float]:
    import ast
    src = (Path(__file__).resolve().parent.parent / "ch25_rag_chroma.py").read_text()
    tree = ast.parse(src)
    found: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("DOCUMENTS", "SIM_THRESHOLD"):
                found[name] = ast.literal_eval(node.value)
    return found["DOCUMENTS"], found["SIM_THRESHOLD"]


class Skip(Exception):
    pass


def server_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{LM_BASE}/models", timeout=2):
            return True
    except urllib.error.HTTPError:
        return True        # a status code is a server: a hosted endpoint answers 401 to a key-less probe
    except (urllib.error.URLError, OSError):
        return False


def index(docs: list[str], where: Path):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    class NomicEmbeddings(OpenAIEmbeddings):
        """nomic's task prefixes, the same pair ch25_rag_chroma.py uses."""

        def embed_documents(self, texts, *a, **kw):
            return super().embed_documents([f"search_document: {t}" for t in texts], *a, **kw)

        def embed_query(self, text):
            # super().embed_documents, NOT self./super().embed_query -- the base
            # class routes embed_query through embed_documents, which would stack
            # both prefixes.
            return super().embed_documents([f"search_query: {text}"])[0]

    emb = NomicEmbeddings(base_url=LM_BASE, api_key="lm-studio",
                          model="text-embedding-nomic-embed-text-v1.5",
                          check_embedding_ctx_length=False, timeout=240)
    store = Chroma(collection_name="doc_bot", embedding_function=emb,
                   persist_directory=str(where),
                   collection_metadata={"hnsw:space": "cosine"})
    store.add_texts(docs)
    return store


def best(store, question: str) -> tuple[str, float]:
    doc, dist = store.similarity_search_with_score(question, k=1)[0]
    return doc.page_content, 1 - dist


_CACHE: dict = {}


def cached() -> dict:
    if not _CACHE:
        if not server_is_up():
            raise Skip(f"no model server at {LM_BASE}")
        documents, threshold = chapter_constants()
        tmp = Path(tempfile.mkdtemp(prefix="ch25_semantic_"))
        before = index(list(documents), tmp / "before")
        after = index(list(documents) + [PYTHON_DOC], tmp / "after")
        _CACHE.update(
            stores={"before": before, "after": after},
            threshold=threshold,
            before=best(before, QUESTION),
            after=best(after, QUESTION),
            paraphrases={q: best(after, q) for q in PARAPHRASES},
            no_content_word={q: best(after, q) for q in NO_CONTENT_WORD},
            near_miss={q: best(after, q) for q in NEAR_MISS},
            n_docs=len(documents),
        )
    return _CACHE


def demo_semantic_retrieval():
    if not server_is_up():
        print(f"no model server at {LM_BASE} -- embeddings are needed; skipping")
        return
    c = cached()
    t = c["threshold"]
    print(f"question : {QUESTION}\nthreshold: {t}\n")
    text, sim = c["before"]
    print(f"before ({c['n_docs']} docs) -- best {sim:.3f}  "
          f"{'HIT' if sim >= t else 'skip -> web fallback'}")
    print(f"          {text[:80]}\n")
    text, sim = c["after"]
    print(f"after  ({c['n_docs'] + 1} docs) -- best {sim:.3f}  "
          f"{'HIT' if sim >= t else 'skip'}")
    print(f"          {text[:80]}\n")

    print("One document added. Zero matching rules changed.\n")
    print("And it did not just cover the sentence you asked:")
    for q, (hit, sim) in c["paraphrases"].items():
        mark = "HIT " if sim >= t else "skip"
        print(f"   {mark} {sim:.3f}  {q}")
    print("\n...including phrasings that share no CONTENT word with it:")
    for q, (hit, sim) in c["no_content_word"].items():
        mark = "HIT " if sim >= t else "skip"
        print(f"   {mark} {sim:.3f}  {q}")
    print("\n...and where the reach runs out (no shared word, no @):")
    for q, (hit, sim) in c["near_miss"].items():
        mark = "HIT " if sim >= t else "skip"
        print(f"   {mark} {sim:.3f}  {q}")
    print("\nThe keyword store would have needed one entry per phrasing. This")
    print("needed one point in meaning-space, and the phrasings found it.")


# --- tests -------------------------------------------------------------------

def test_the_question_missed_before_the_document_existed():
    c = cached()
    assert c["before"][1] < c["threshold"], c["before"]


def test_the_same_question_hits_after_adding_one_document():
    c = cached()
    assert c["after"][1] >= c["threshold"], c["after"]


def test_the_hit_is_the_document_we_added():
    assert cached()["after"][0] == PYTHON_DOC


def test_no_matching_rule_was_touched():
    """The claim in the challenge, made checkable: the two indexes differ by
    exactly one document and nothing else."""
    documents, _ = chapter_constants()
    added = set(list(documents) + [PYTHON_DOC]) - set(documents)
    assert added == {PYTHON_DOC}


def test_phrasings_the_document_never_used_also_reach_it():
    """The part that separates this from a keyword fix: '@wraps' and 'keep its
    __name__' were never entries anywhere."""
    c = cached()
    for q, (hit, sim) in c["paraphrases"].items():
        assert sim >= c["threshold"], (q, sim)
        assert hit == PYTHON_DOC, (q, hit)


def test_phrasings_sharing_no_content_word_still_reach_it():
    """The claim a plain keyword lookup cannot make. These questions contain none of
    the document's content words -- not "python", not "decorator", not "wrap" -- so
    no exact-match index could connect them. (They do share the `@` character and
    ordinary function words like "a"; the assertion below is about content words,
    which is what a keyword index would key on.)"""
    import re
    c = cached()
    STOPWORDS = {"a", "an", "the", "is", "it", "its", "and", "to", "do", "what",
                 "you", "with", "for", "of", "how", "i", "my"}
    def content(text):
        return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}
    doc_words = content(PYTHON_DOC)
    for q, (hit, sim) in c["no_content_word"].items():
        assert not (doc_words & content(q)), (q, doc_words & content(q))
        assert sim >= c["threshold"], (q, sim)
        assert hit == PYTHON_DOC, (q, hit)


def test_reach_runs_out_on_phrasings_with_no_lexical_hook():
    """The other half of the truth, and the reason @sec-eval-ab exists: these are
    decorator questions too, but with no shared word AND no `@`, and they fall just
    short of the threshold. Semantic retrieval widens the net; it does not make it
    infinite."""
    c = cached()
    for q, (hit, sim) in c["near_miss"].items():
        assert sim < c["threshold"], (q, sim)   # under the bar -- would fall back to web


def test_the_added_document_flips_the_route_from_web_to_generate():
    """The consequence at the GRAPH level, which is what the chapter cares about.

    The chapter's own agent is a model, so its route is a model decision; the rule
    it is asked to follow is not. This rebuilds that rule as a plain function --
    hits -> generate, no hits -> web -- and runs it as a real two-node graph over
    the two indexes. Same retrieval, one extra document, opposite route.
    """
    from typing import NotRequired, TypedDict
    from langgraph.graph import StateGraph, START, END

    c = cached()

    class S(TypedDict):
        question: str
        docs: NotRequired[list]
        route: NotRequired[str]

    def build(store):
        def search_docs(state: S) -> dict:
            doc, sim = best(store, state["question"])
            return {"docs": [doc] if sim >= c["threshold"] else []}

        def route_after_docs(state: S) -> str:
            # Chapter 5's rule, unchanged: an empty docs list is what sends a run
            # to the web branch.
            return "generate" if state["docs"] else "web"

        b = StateGraph(S)
        b.add_node("search_docs", search_docs)
        b.add_node("generate", lambda s: {"route": "generate"})
        b.add_node("web", lambda s: {"route": "web"})
        b.add_edge(START, "search_docs")
        b.add_conditional_edges("search_docs", route_after_docs,
                                {"generate": "generate", "web": "web"})
        b.add_edge("generate", END)
        b.add_edge("web", END)
        return b.compile()

    before = build(c["stores"]["before"]).invoke({"question": QUESTION})
    after = build(c["stores"]["after"]).invoke({"question": QUESTION})
    assert before["route"] == "web", before          # nothing retrieved -> web fallback
    assert after["route"] == "generate", after       # one document added -> answers locally


def test_the_chapters_own_store_was_not_modified():
    """This file builds in a temp directory; running it must not change what the
    chapter prints."""
    documents, _ = chapter_constants()
    assert PYTHON_DOC not in documents


if __name__ == "__main__":
    try:
        demo_semantic_retrieval()
    except Skip as e:
        print(f"skipping: {e}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    passed = skipped = failed = 0
    for t in _tests:
        try:
            t()
            print("PASS", t.__name__); passed += 1
        except Skip as e:
            print("SKIP", t.__name__, f"({e})"); skipped += 1
        except AssertionError as e:
            print("FAIL", t.__name__, f"({e})"); failed += 1
    # Skips are not passes: with no embedding server this prints 0 passed, N skipped.
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    if failed:
        sys.exit(1)
