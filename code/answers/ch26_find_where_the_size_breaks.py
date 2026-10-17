"""Chapter 26 Challenge — Bronze: push chunk_size to both extremes.

The good index used `chunk_size=300`, comfortably larger than any paragraph in the
sample, so each topic stayed whole. This runs the chapter's Part C comparison at
three sizes -- 60, 300 and 2000 -- against the same document, the same question and
the same embeddings, and prints what comes back on top.

Predict before reading the output:

  * **60** -- the batch-consumption topic no longer fits in one chunk, so the top
    hit is a *fragment*. On this document it is the bare heading, "Kafka batch
    consumption" -- 23 characters, no setting, no answer. It scores **0.759** against
    the good split's **0.801**: the score does rank the real answer higher, but the
    fragment still clears the one bar you have met so far, the 0.60 similarity filter
    from Ch25. A short, entirely-on-topic chunk is a confident HIT with no answer in
    it.

    Be precise about what that does and does not prove. It proves that a
    similarity threshold -- a check that sees the SCORE and nothing else -- cannot
    distinguish "about the topic" from "answers the question". It does NOT prove the
    later stages are equally blind, and they are not built the same way: Ch27's
    reranker hands the candidate's TEXT to a model and scores it afresh, and Ch29's
    golden layer matches the question and checks freshness before it trusts a cached
    answer. Neither is demonstrated blind here. The lesson that generalises is
    narrower and more useful: a scalar similarity is a statement about proximity, so
    any consumer that reads only the scalar inherits exactly this failure.
  * **300** -- the chapter's setting. One chunk per topic.
  * **2000** -- larger than the whole document, so the split collapses back into a
    single blob and the result becomes indistinguishable from the "bad" index the
    chapter opened with. Byte-for-byte identical, in fact, which the file asserts.

Both extremes hurt, and the Gotcha's symmetry is the reason: retrieval returns
*chunks*, so a chunk is simultaneously the unit of matching and the unit of
answering. Too small optimises matching at the cost of answering; too large does
the reverse. There is no setting that is safe in the abstract -- only one that
fits the shape of your documents, which is why the chapter's 300 is described
relative to the paragraphs rather than as a number.

Needs LM Studio at :1234 for embeddings.
Run:  python code/answers/ch26_find_where_the_size_breaks.py   (see answers/README.md)
"""

import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LM_BASE = "http://localhost:1234/v1"
QUESTION = "How do I configure Kafka batch consumers?"
SIZES = [60, 300, 2000]


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


def top_hit_at(size: int, sample: str, tmp: Path) -> tuple[str, float, list]:
    """Split at `size`, index, ask the question.

    Returns (top-hit text, similarity, the full chunk list) -- the chunk list so the
    tests can assert on what was indexed, not only on what came back.
    """
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    class NomicEmbeddings(OpenAIEmbeddings):
        # nomic's task prefixes, the @sec-rag-foundations adapter the chapter uses.
        def embed_documents(self, texts, *args, **kwargs):
            return super().embed_documents(
                [f"search_document: {t}" for t in texts], *args, **kwargs)

        def embed_query(self, text):
            return super().embed_documents([f"search_query: {text}"])[0]

    emb = NomicEmbeddings(base_url=LM_BASE, api_key="lm-studio",
                          model="text-embedding-nomic-embed-text-v1.5",
                          check_embedding_ctx_length=False, timeout=240)
    # The chapter's separator list, including the " " and "" fallbacks -- without
    # them `chunk_size` is a target rather than a ceiling and size=60 emits a
    # 170-char chunk (@sec-chunk-size-not-a-ceiling). chunk_overlap is held FIXED at
    # 40 across all three sizes so that chunk_size is the only variable; at size=60
    # that is a 67% overlap, which is itself part of what "too small" costs you.
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=40,
        separators=["\n\n", "\n", ". ", " ", ""]).split_text(sample)

    db = tmp / f"chunk_{size}"
    shutil.rmtree(db, ignore_errors=True)
    store = Chroma(collection_name=f"s{size}", embedding_function=emb,
                   persist_directory=str(db),
                   collection_metadata={"hnsw:space": "cosine"})
    store.add_texts(chunks)
    doc, dist = store.similarity_search_with_score(QUESTION, k=1)[0]
    return doc.page_content, 1 - dist, chunks


_CACHE: dict = {}


def cached() -> dict:
    if not _CACHE:
        if not server_is_up():
            raise Skip(f"no model server at {LM_BASE}")
        import ch26_chunking as ch25
        tmp = Path(tempfile.mkdtemp(prefix="ch26_sizes_"))
        _CACHE["sample"] = ch25.SAMPLE
        for size in SIZES:
            _CACHE[size] = top_hit_at(size, ch25.SAMPLE, tmp)
        # the chapter's "bad" index: the whole document, unsplit
        _CACHE["bad"] = top_hit_at(10_000, ch25.SAMPLE, tmp)
    return _CACHE


def demo_where_the_size_breaks():
    if not server_is_up():
        print(f"no model server at {LM_BASE} -- embeddings are needed; skipping")
        return
    c = cached()
    print(f"question: {QUESTION}\n")
    for size in SIZES:
        text, sim, chunks = c[size]
        print(f"--- chunk_size={size}  ({len(chunks)} chunk(s), longest "
              f"{max(len(x) for x in chunks)} chars, top hit {len(text)} chars, "
              f"similarity {sim:.3f})")
        print(f"    {text.strip()[:150]}{'...' if len(text) > 150 else ''}\n")

    small_sim, mid_sim = c[60][1], c[300][1]
    print(f"Scores: {small_sim:.3f} at size 60, {mid_sim:.3f} at size 300 -- "
          f"{abs(small_sim - mid_sim):.3f} apart,")
    print(f"for a {len(c[60][0].strip())}-char fragment against a "
          f"{len(c[300][0].strip())}-char answer. The score ranks the")
    print(f"answer higher, but {small_sim:.3f} still clears the 0.60 similarity filter")
    print("from Ch25 -- so a check that reads only the score takes a chunk with no")
    print("config key in it for a hit. That is a fact about SCALAR thresholds, not")
    print("about every later stage: Ch27's reranker reads the candidate's text and")
    print("Ch29's golden layer checks the question and the stamp. Neither is shown")
    print("blind here; what generalises is that proximity is not answerability.")
    print(f"\nAt 2000 the split produced {len(c[2000][2])} chunk(s): the whole document, which is")
    print("the 'bad' index from the chapter's Part C, arrived at from the other side.")


# --- tests -------------------------------------------------------------------

def test_a_tiny_chunk_size_fragments_the_topic():
    c = cached()
    assert len(c[60][2]) > len(c[300][2]), (len(c[60][2]), len(c[300][2]))
    assert len(c[60][0]) < len(c[300][0])


def test_the_top_hit_at_60_is_exactly_the_bare_heading():
    """The narrated fragment, asserted verbatim rather than by length."""
    c = cached()
    assert c[60][0].strip() == "Kafka batch consumption", repr(c[60][0])
    assert len(c[60][0].strip()) == 23


def test_the_fragment_is_too_short_to_answer_with():
    """The concrete harm: the setting the chapter's answer depends on is no
    longer inside the retrieved text."""
    c = cached()
    assert "spring.kafka.listener.type=batch" in c[300][0]
    assert "spring.kafka.listener.type=batch" not in c[60][0]


def test_the_fragment_still_clears_a_scalar_threshold():
    """Proximity is not answerability. The score DOES rank the real answer higher,
    yet the 23-char heading still clears Ch25's 0.60 filter -- so a check reading
    only the scalar admits a chunk with no config key in it. Scoped deliberately to
    scalar thresholds: Ch27 rescores the text and Ch29 checks the question, and
    nothing here shows either of them blind."""
    c = cached()
    frag_sim, ans_sim = c[60][1], c[300][1]
    assert frag_sim >= 0.60, frag_sim
    assert ans_sim > frag_sim, (ans_sim, frag_sim)
    assert len(c[60][0].strip()) < len(c[300][0].strip()) / 3


def test_the_middle_setting_is_exactly_one_chunk_per_paragraph():
    """Not 'about one per topic' -- the three chunks ARE the three paragraphs."""
    c = cached()
    assert c[300][2] == cached()["sample"].split("\n\n")
    assert [len(x) for x in c[300][2]] == [237, 275, 239]


def test_a_huge_chunk_size_collapses_into_one_blob():
    c = cached()
    assert len(c[2000][2]) == 1, len(c[2000][2])
    assert c[2000][2][0] == cached()["sample"]      # byte for byte, not stripped


def test_the_huge_setting_is_the_chapters_bad_index():
    """Not merely 'similar to': the same text comes back."""
    c = cached()
    assert c[2000][0] == c["bad"][0]


def test_chunk_size_is_a_real_ceiling_with_the_fallback_separators():
    """Why the separator list keeps ' ' and '': without them size=60 emits a
    170-char chunk. With them, 60 means 60."""
    c = cached()
    for size in SIZES:
        assert max(len(x) for x in c[size][2]) <= size, (size, c[size][2])


if __name__ == "__main__":
    try:
        demo_where_the_size_breaks()
    except Skip as e:
        print(f"skipping: {e}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    passed = skipped = 0
    for t in _tests:
        try:
            t()
            print("PASS", t.__name__)
            passed += 1
        except Skip as e:
            print("SKIP", t.__name__, f"({e})")
            skipped += 1
    # Report what actually happened. A run with no model server verifies nothing,
    # and saying "all tests passed" there would be a lie the reader cannot see.
    if skipped:
        print(f"\n{passed} passed, {skipped} SKIPPED (no model server) "
              f"-- nothing was verified")
    else:
        print(f"\nall {passed} tests passed")
