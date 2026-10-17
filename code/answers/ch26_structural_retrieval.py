"""Chapter 26 Challenge — Silver: chunk by structure, retrieve with the breadcrumb.

Part B produced structural chunks with a heading path; Part C fed retrieval with
plain paragraph chunks. This is the join the challenge asks for, made runnable:
split the `MARKDOWN_DOC` with `MarkdownHeaderTextSplitter`, prepend each chunk's
heading path to the text before indexing (so `Kafka > Error handling` becomes part
of what gets embedded), AND stash that path in Chroma's per-document metadata. Then
ask *"how do I stop a bad message from failing the whole batch?"* and show two
things: the returned chunk is the error-handling one (not batch consumption, not the
RabbitMQ ack), and its heading path can be printed from metadata as the answer's
source.

To run offline with no model server, the embedding here is a deterministic fake that
mirrors `OpenAIEmbeddings`'s shape - it exposes `embed_documents`/`embed_query` and
Chroma drives it exactly like the LM Studio model in `code/ch26_chunking.py`. Instead
of learned semantics it scores text against a small concept lexicon (one axis per
section topic).

Be clear about what that fake can and cannot demonstrate, because it is easy to
read more into a green test than is there. The lexicon has one axis PER SECTION and
the query's phrases were put on the error-handling axis by hand, so the ranking
below is a property of this file, not evidence about any embedding model. What the
fake genuinely exercises is the WIRING: that the breadcrumb reaches the embedder and
the store, that metadata survives retrieval, and that a path can be cited from a
hit. Whether a real embedder also prefers the error-handling chunk over the bare
word "batch" is a question only a real embedder can answer - point this file at the
chapter's `NomicEmbeddings` (see the bottom of the file) to ask it.

The point to land: a good chunk is not only well-sized text - it can carry where it
came from, and that breadcrumb is the difference between "here is an answer" and
"here is an answer, from Kafka > Error handling." A naive character split of the same
document throws that breadcrumb away (see the last test), so even when it retrieves
the right text it cannot cite the section.

Run:  python code/answers/ch26_structural_retrieval.py   (see answers/README.md)
"""

from typing import List

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# --- the structured document, verbatim from the chapter (Part B) --------------
MARKDOWN_DOC = """# Message Queues
## Kafka
### Batch consumption
Set spring.kafka.listener.type=batch to enable it, then tune max-poll-records.
### Error handling
Use a DefaultErrorHandler with a dead-letter queue to isolate bad messages.
## RabbitMQ
### Acknowledgements
Manual ack keeps messages from being lost."""

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def heading_path(metadata: dict) -> str:
    # Join the enclosing h1/h2/h3 titles into one breadcrumb, e.g.
    # "Message Queues > Kafka > Error handling". This is what we both embed into
    # the text and stash in metadata as the answer's citable source.
    return " > ".join(metadata.get(k, "") for k in ("h1", "h2", "h3") if metadata.get(k))


# --- a deterministic stand-in for the LM Studio embedding model ---------------
# Same interface Chroma expects from OpenAIEmbeddings (embed_documents/embed_query),
# but instead of a learned model it projects text onto one axis per section topic by
# counting concept phrases. That is enough to disambiguate the challenge query the
# way a real embedding would - by what the text is ABOUT ("isolate a bad message"),
# not by the surface word "batch" it happens to contain.
CONCEPT_AXES = [
    ("batch consumption", ["batch consumption", "max-poll-records", "listener.type",
                           "poll", "receives a list", "configure kafka batch"]),
    ("error handling",    ["error handling", "bad message", "stop a bad", "dead-letter",
                           "defaulterrorhandler", "duplicate", "retried", "isolate",
                           "failing", "fails", "whole batch is retried"]),
    ("rabbitmq ack",      ["rabbitmq", "acknowledgement", "manual ack",
                           "keeps messages from being lost", "being lost"]),
]


class ConceptEmbeddings(Embeddings):
    """Deterministic keyword-concept embedding; offline and stable across runs."""

    def _vec(self, text: str) -> List[float]:
        low = text.lower()
        return [float(sum(low.count(phrase) for phrase in phrases))
                for _label, phrases in CONCEPT_AXES]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


# --- build the structural index (breadcrumb embedded AND in metadata) ---------
def structural_chunks():
    # MarkdownHeaderTextSplitter cuts on the headings and remembers the enclosing
    # h1/h2/h3 titles as metadata - the boundaries the author already chose.
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS)
    return splitter.split_text(MARKDOWN_DOC)


class RecordingEmbeddings(ConceptEmbeddings):
    """The same stand-in, but it keeps every text Chroma actually handed it.

    Used by one test to check what was really indexed, rather than re-deriving the
    string the test expects and asserting that it matches itself.
    """

    def __init__(self):
        self.seen: List[str] = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.seen.extend(texts)
        return super().embed_documents(texts)


def build_index(embeddings=None, collection: str = "ch26_structural"):
    chunks = structural_chunks()
    texts, metadatas = [], []
    for doc in chunks:
        path = heading_path(doc.metadata)
        # Prepend the breadcrumb to the text so the SECTION a chunk lives in becomes
        # part of what gets embedded, and stash the same path in metadata so we can
        # cite it back as the source.
        texts.append(f"{path}\n{doc.page_content.strip()}")
        metadatas.append({"heading_path": path})
    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings or ConceptEmbeddings(),
        collection_metadata={"hnsw:space": "cosine"},   # cosine, as in the chapter
    )
    store.add_texts(texts, metadatas=metadatas)
    return store


INDEX = build_index()
QUESTION = "how do I stop a bad message from failing the whole batch?"


def retrieve(question: str, k: int = 1):
    # Chroma returns a cosine DISTANCE; similarity is 1 - distance, exactly as in
    # code/ch26_chunking.py. Return (Document, similarity) pairs, closest first.
    hits = INDEX.similarity_search_with_score(question, k=k)
    return [(doc, 1.0 - dist) for doc, dist in hits]


# --- tests -------------------------------------------------------------------
def test_structural_chunks_carry_heading_metadata():
    # Six heading lines -> three chunks: the splitter emits a chunk only where text
    # follows a heading, so "# Message Queues" and "## Kafka" shape the PATH without
    # producing chunks of their own. Each chunk remembers its full h1/h2/h3 path.
    chunks = structural_chunks()
    assert sum(1 for line in MARKDOWN_DOC.splitlines() if line.startswith("#")) == 6
    assert len(chunks) == 3
    paths = [heading_path(c.metadata) for c in chunks]
    assert paths == [
        "Message Queues > Kafka > Batch consumption",
        "Message Queues > Kafka > Error handling",
        "Message Queues > RabbitMQ > Acknowledgements",
    ]
    # The metadata really holds the split-out heading levels, not just a joined string.
    err = chunks[1].metadata
    assert err["h1"] == "Message Queues" and err["h2"] == "Kafka" and err["h3"] == "Error handling"


def test_breadcrumb_reaches_the_embedder_and_the_stored_document():
    """The claim is that the breadcrumb is part of what gets EMBEDDED, not metadata
    the vector never sees -- so check the texts the embedder was actually handed,
    and the text Chroma actually stored. (Asserting that `path + text` starts with
    `path` would only test the test.)"""
    spy = RecordingEmbeddings()
    store = build_index(embeddings=spy, collection="ch26_structural_spy")

    expected = ["Message Queues > Kafka > Batch consumption",
                "Message Queues > Kafka > Error handling",
                "Message Queues > RabbitMQ > Acknowledgements"]
    assert len(spy.seen) == 3, spy.seen
    for text, path in zip(spy.seen, expected):
        assert text.startswith(path + "\n"), text     # the embedder saw the breadcrumb

    # ...and it survived into the store, alongside the metadata copy.
    stored = store.get(include=["documents", "metadatas"])
    pairs = sorted(zip(stored["documents"], stored["metadatas"]),
                   key=lambda p: p[1]["heading_path"])
    assert [m["heading_path"] for _d, m in pairs] == sorted(expected)
    for doc_text, meta in pairs:
        assert doc_text.startswith(meta["heading_path"] + "\n"), doc_text


def test_targeted_chunk_is_retrieved_with_its_source():
    # The whole point: the "bad message / dead-letter" question returns the
    # error-handling chunk, and we can print WHERE it came from from metadata.
    (doc, sim), = retrieve(QUESTION, k=1)
    assert doc.metadata["heading_path"] == "Message Queues > Kafka > Error handling"
    assert "dead-letter" in doc.page_content.lower()
    # Not the batch-consumption chunk (the lexical "batch" trap) and not RabbitMQ.
    assert "batch consumption" not in doc.page_content.lower().split("\n", 1)[1]
    assert "rabbitmq" not in doc.metadata["heading_path"].lower()
    # Exactly on the error-handling axis -- true of the hand-built lexicon, and NOT a
    # number to expect from a real embedder. Do not carry this assertion across.
    assert sim >= 0.99


def test_error_handling_outranks_the_batch_chunk():
    # Ranked retrieval, not just top-1: error handling beats both other sections,
    # even though the query literally contains the word "batch". True of the
    # stand-in's lexicon by construction; run it against NomicEmbeddings to find out
    # whether a learned model agrees.
    ranked = retrieve(QUESTION, k=3)
    top_path, top_sim = ranked[0][0].metadata["heading_path"], ranked[0][1]
    assert top_path == "Message Queues > Kafka > Error handling"
    others = [sim for doc, sim in ranked[1:]]
    assert all(top_sim > other for other in others)   # strictly ahead of the rest


def test_naive_split_throws_the_breadcrumb_away():
    # A naive character split of the SAME document keeps no section metadata, so even
    # when it retrieves the right text it cannot cite where the text came from. That
    # is the concrete win of structural chunking over counting characters.
    naive = RecursiveCharacterTextSplitter(
        chunk_size=80, chunk_overlap=0, separators=["\n\n", "\n", ". ", " ", ""],
    ).create_documents([MARKDOWN_DOC])
    assert naive, "expected at least one naive chunk"
    assert all(heading_path(c.metadata) == "" for c in naive)   # no breadcrumb at all


if __name__ == "__main__":
    print("===== structural chunks (each carries its heading path) =====")
    for doc in structural_chunks():
        print(f"  [{heading_path(doc.metadata)}]")
        print(f"      {doc.page_content.strip()}")

    print(f"\n===== retrieval - question: {QUESTION} =====")
    (doc, sim), = retrieve(QUESTION, k=1)
    print(f"  top hit, similarity {sim:.3f}")
    print(f"  source: {doc.metadata['heading_path']}")
    print(f"  {doc.page_content.strip()}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
