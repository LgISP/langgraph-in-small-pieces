"""Chapter 28 - Versioning the index (incremental rebuilds).

Documents change. A production RAG index has to keep up without re-embedding the
world every time. This example shows the three things that matter:

    1) Re-embed only the documents that actually changed - skip the rest and
       save the compute.
    2) Delete the vectors whose source document was removed upstream.
    3) Make every answer traceable: which VERSION of a document produced it.

The mechanism is a stable id (the doc_id, never a random uuid) plus a content
hash stored in each vector's metadata. On every sync we diff the store against
the source of truth and reconcile each document: add / change (re-embed) /
unchanged (skip) / delete.

Run:  python code/ch28_versioning.py
"""

import hashlib
import os
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

LM_BASE = "http://localhost:1234/v1"        # your local server, not the OpenAI cloud
# A dedicated on-disk collection directory, kept next to this script.
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_versioned")

# Start from a clean directory so the add/skip/change/delete story shows every
# run. chromadb caches one client per path, so this MUST happen before the
# Chroma object below is created - otherwise the client would keep a stale
# handle to the deleted database and every write would fail read-only.
shutil.rmtree(DB_DIR, ignore_errors=True)

class NomicEmbeddings(OpenAIEmbeddings):
    """nomic's task prefixes -- the adapter @sec-rag-foundations introduced.
    search_document: for indexed text, search_query: for a lookup; the model card
    requires them for retrieval and the local server does not add them."""

    def embed_documents(self, texts, *args, **kwargs):
        return super().embed_documents(
            [f"search_document: {t}" for t in texts], *args, **kwargs)

    def embed_query(self, text):
        # super().embed_documents, NOT embed_query: the base routes embed_query
        # through embed_documents, which would stack both prefixes.
        return super().embed_documents([f"search_query: {text}"])[0]


embeddings = NomicEmbeddings(
    base_url=LM_BASE,
    api_key="lm-studio",                     # the local server ignores the key
    model="text-embedding-nomic-embed-text-v1.5",
    check_embedding_ctx_length=False,        # skip the tokenizer length probe local servers do not serve
)

vectorstore = Chroma(
    collection_name="versioned",
    embedding_function=embeddings,
    persist_directory=DB_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)


def content_hash(text: str) -> str:
    """Content hash: same content -> same hash; change one character -> the whole hash changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def sync_index(docs: dict, version: str) -> None:
    """Bring the vector store in line with `docs`, incrementally.

    docs: {doc_id: content} - this is the SOURCE OF TRUTH.
    version: the tag stamped on everything written this pass (stored in metadata
             so an answer can later be traced back to a document version).
    """
    print(f"\n===== sync index -> version {version} =====")

    # 1) Read what the store currently holds: doc_id -> the hash it has on file.
    existing = vectorstore.get()   # returns ids / documents / metadatas
    stored = {}
    for meta in existing["metadatas"]:
        stored[meta["doc_id"]] = meta["content_hash"]

    # 2) Reconcile document by document.
    for doc_id, content in docs.items():
        h = content_hash(content)
        if doc_id not in stored:
            # New: embed and write.
            vectorstore.add_texts(
                [content], ids=[doc_id],
                metadatas=[{"doc_id": doc_id, "content_hash": h, "version": version}],
            )
            print(f"  [new]       {doc_id}  -> embedding")
        elif stored[doc_id] != h:
            # Changed: write over the SAME id. Chroma overwrites a row whose id
            # already exists, so this is the upsert -- and doing it without a
            # pre-delete is what keeps a failure survivable: if the embedding call
            # raises, the old vector is still there and the next sync retries. A
            # delete-then-add would already have thrown the old row away.
            vectorstore.add_texts(
                [content], ids=[doc_id],
                metadatas=[{"doc_id": doc_id, "content_hash": h, "version": version}],
            )
            print(f"  [changed]   {doc_id}  -> re-embedding (hash {stored[doc_id]}->{h})")
        else:
            # Unchanged: skip; no model call.
            print(f"  [unchanged] {doc_id}  -> skip (hash {h} unchanged, no recompute)")

    # 3) Delete: anything the store holds that the source of truth no longer has.
    for doc_id in set(stored) - set(docs):
        vectorstore.delete(ids=[doc_id])
        print(f"  [deleted]   {doc_id}  -> removed from store")


def show_store() -> None:
    """Print the store contents with their version/hash - the traceability made visible."""
    data = vectorstore.get()
    print("  store contents:")
    for doc_id, content, meta in sorted(zip(data["ids"], data["documents"], data["metadatas"])):
        print(f"    {doc_id:8} v{meta['version']} #{meta['content_hash']}  {content}")


def query(q: str) -> None:
    """On a query, carry the hit's version/hash back out - the answer is traceable to a version."""
    hits = vectorstore.similarity_search(q, k=1)
    print(f"  query \"{q}\" hit:")
    for doc in hits:
        m = doc.metadata
        print(f"    {doc.page_content}  <- doc_id={m['doc_id']} v{m['version']} #{m['content_hash']}")


if __name__ == "__main__":
    # ---------- Version 1: first build ----------
    V1 = {
        "kafka": "Kafka batch consumption: set spring.kafka.listener.type=batch",
        "redis": "Redis connection pool: configure via spring.data.redis.*",
        "mysql": "MySQL indexing: add a B+ tree index to speed up queries",
    }
    sync_index(V1, version="1")
    show_store()

    # ---------- Sync the SAME content again: everything should be unchanged/skip ----------
    sync_index(V1, version="1")   # content did not move, so not a single embedding fires

    # ---------- Version 2: changed kafka, deleted redis, added pg ----------
    V2 = {
        "kafka": "Kafka batch consumption: spring.kafka.listener.type=batch, plus max-poll-records",  # changed
        # "redis" removed
        "mysql": "MySQL indexing: add a B+ tree index to speed up queries",   # untouched
        "pg": "PostgreSQL: use the pgvector extension for vector search",      # new
    }
    sync_index(V2, version="2")
    show_store()

    # ---------- Query: the hit carries its version, so the answer is traceable ----------
    print("\n===== query (answer traceable to a version) =====")
    query("how to configure kafka batch consumption")
    query("postgres vector search")
