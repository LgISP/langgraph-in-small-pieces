"""Chapter 26 - Chunking.

Chunking is the single choice that most decides retrieval quality in a RAG
pipeline. Before a document ever reaches an embedding model it has to be cut
into pieces, and HOW you cut it decides what a query can find later. This file
makes that concrete in three parts:

    A) Fixed vs recursive splitting - what "cutting badly" actually looks like.
    B) Markdown structural splitting - cut on headings, keep the heading path
       as metadata so each chunk remembers where it came from.
    C) Retrieval comparison - the same question against two indexes: the whole
       document as one blob vs the document split by topic.

Part C embeds text with a local model served by LM Studio (OpenAI-compatible
API) and stores vectors in Chroma. Nothing here talks to a paid cloud - the
whole book runs on your own machine. The two Chroma directories are rebuilt
from the seed document on every run, so this file is self-contained.

Run:  python code/ch26_chunking.py
"""

import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

SAMPLE = """Kafka batch consumption. To turn on batch consumption, set \
spring.kafka.listener.type=batch. Once enabled, the listener method receives a \
List and pulls many messages in one poll. To bound each batch, tune the \
max-poll-records parameter.

Kafka error handling. Under batch consumption, if one message fails the whole \
batch is retried by default, which easily causes duplicate consumption. \
Configure a DefaultErrorHandler with a dead-letter queue to isolate the bad \
message so it does not drag down the whole batch.

Redis connection pool. Spring Boot uses Lettuce as its default client. Configure \
the pool through spring.data.redis.lettuce.pool.*, for example max-active for the \
maximum number of connections and max-idle for the maximum idle connections."""


def show_chunks(title, chunks):
    # chunks is either a list[str] (from the character/recursive splitters) or a
    # list[Document] (from the markdown splitter); handle both so the callers
    # stay simple.
    print(f"\n===== {title} =====")
    for i, chunk in enumerate(chunks):
        text = chunk if isinstance(chunk, str) else chunk.page_content
        cut = "..." if len(text) > 60 else ""          # the listing shows each chunk's first 60 chars
        print(f"  [chunk {i}] ({len(text)} chars) {text[:60].strip()}{cut}")
    print(f"  {len(chunks)} chunk(s) total")


# ===========================================================================
# A) Fixed splitting vs recursive splitting
# ===========================================================================
def part_a():
    # separator="" means "just count characters"
    fixed = CharacterTextSplitter(separator="", chunk_size=120, chunk_overlap=0)
    show_chunks("A1) fixed split (every 120 chars) - sentences cut mid-word", fixed.split_text(SAMPLE))

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],  # paragraph -> line -> sentence -> word
    )
    show_chunks("A2) recursive split (prefer paragraph/sentence) - no word is cut in half", recursive.split_text(SAMPLE))

    # Compare the final chunk here against A2's.
    overlap = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    show_chunks("A3) recursive split + 30-char overlap - neighbors share text at the seam", overlap.split_text(SAMPLE))


# ===========================================================================
# B) Markdown structural splitting (cut on headings, keep the heading path)
# ===========================================================================
MARKDOWN_DOC = """# Message Queues
## Kafka
### Batch consumption
Set spring.kafka.listener.type=batch to enable it, then tune max-poll-records.
### Error handling
Use a DefaultErrorHandler with a dead-letter queue to isolate bad messages.
## RabbitMQ
### Acknowledgements
Manual ack keeps messages from being lost."""


def part_b():
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "h1"), ("##", "h2"), ("###", "h3"),
    ])
    docs = splitter.split_text(MARKDOWN_DOC)
    print("\n===== B) markdown structural split - each chunk carries its heading path =====")
    for doc in docs:
        # the heading path: store it next to the vector, cite it as the answer's source
        path = " > ".join(doc.metadata.get(k, "") for k in ["h1", "h2", "h3"] if doc.metadata.get(k))
        print(f"  [{path}]")
        print(f"      {doc.page_content.strip()}")


# ===========================================================================
# C) Retrieval comparison: whole document as one blob vs split by topic
# ===========================================================================
def part_c():
    import shutil

    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    class NomicEmbeddings(OpenAIEmbeddings):
        """nomic's task prefixes -- the same adapter @sec-rag-foundations introduced.
        search_document: for text going INTO the index, search_query: for a lookup.
        The model card requires them for retrieval and the local server does not add
        them, so a query embedded without them lands in a space the model was never
        asked to build."""

        def embed_documents(self, texts, *args, **kwargs):
            return super().embed_documents(
                [f"search_document: {t}" for t in texts], *args, **kwargs)

        def embed_query(self, text):
            # super().embed_documents, NOT embed_query: the base class routes
            # embed_query through embed_documents, which would stack both prefixes.
            return super().embed_documents([f"search_query: {text}"])[0]

    # check_embedding_ctx_length is off because the local server does not report
    # a token limit the way the OpenAI cloud does; timeout is generous since the
    # server may be busy.
    embeddings = NomicEmbeddings(
        base_url="http://localhost:1234/v1",  # your local server, not the OpenAI cloud
        api_key="lm-studio",                   # the local server ignores the key
        model="text-embedding-nomic-embed-text-v1.5",
        check_embedding_ctx_length=False,
        timeout=240,
    )

    # Two Chroma stores next to this file. Both are rebuilt from scratch on every
    # run, so the demo never depends on stale vectors from a previous run.
    here = os.path.dirname(os.path.abspath(__file__))
    db_bad = os.path.join(here, "chroma_chunk_bad")
    db_good = os.path.join(here, "chroma_chunk_good")
    shutil.rmtree(db_bad, ignore_errors=True)
    shutil.rmtree(db_good, ignore_errors=True)

    # BAD: the whole document as one chunk
    bad = Chroma(collection_name="bad", embedding_function=embeddings,
                 persist_directory=db_bad, collection_metadata={"hnsw:space": "cosine"})
    bad.add_texts([SAMPLE], ids=["sample:0"], metadatas=[{"doc_id": "sample", "chunk": 0}])

    # GOOD: one chunk per paragraph (chunk_size exceeds any single paragraph).
    # Keep the " " and "" fallbacks: without them chunk_size is a target rather than
    # a ceiling, which bites the moment you turn the dial down (the Bronze challenge
    # does exactly that). At 300 they never fire -- every paragraph already fits.
    good_chunks = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=40,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_text(SAMPLE)
    good = Chroma(collection_name="good", embedding_function=embeddings,
                  persist_directory=db_good, collection_metadata={"hnsw:space": "cosine"})
    # A chunk's identity is its document plus its place in it: the same two
    # fields @sec-rag-foundations gave whole documents, one level down.
    good.add_texts(good_chunks,
                   ids=[f"sample:{i}" for i in range(len(good_chunks))],
                   metadatas=[{"doc_id": "sample", "chunk": i} for i in range(len(good_chunks))])

    question = "How do I configure Kafka batch consumers?"
    print(f"\n===== C) retrieval comparison - question: {question} =====")

    doc_bad, dist_bad = bad.similarity_search_with_score(question, k=1)[0]
    print(f"\n  BAD  index (whole doc as one chunk) - top hit, similarity {1 - dist_bad:.3f}:")
    print(f"       {doc_bad.page_content[:120].strip()}...")
    print(f"       -> returns all {len(doc_bad.page_content)} chars: batch consumption plus error handling plus Redis, diluting the answer.")

    doc_good, dist_good = good.similarity_search_with_score(question, k=1)[0]
    print(f"\n  GOOD index (split by topic) - top hit, similarity {1 - dist_good:.3f}:")
    print(f"       {doc_good.page_content.strip()}")
    print(f"       -> returns only {len(doc_good.page_content)} chars, hitting the batch-consumption topic exactly.")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
