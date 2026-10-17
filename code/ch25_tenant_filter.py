"""Chapter 25 - Two things a retriever must not do: obey, and over-share.

The doc-bot's index has been single-tenant and fully trusted since Ch5. Neither
survives a real corpus. This file shows the second problem and its fix, because
that one has a mechanical answer; the first (a document that talks to the model)
does not, and the chapter says so.

Retrieval must be scoped WHERE THE SEARCH HAPPENS, not after. Filtering
afterwards spends the top-k budget on documents the user may not see, so the
answer degrades exactly when the corpus grows -- and any code path that forgets
to filter has already put someone else's document in a prompt.

Run:  python code/ch25_tenant_filter.py

Needs LM Studio on http://localhost:1234/v1 with the embedding model loaded.
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

LM_BASE = "http://localhost:1234/v1"

# Two tenants, one collection, one topic. Only the metadata separates them.
DOCS = [
    ("Acme Kafka runbook: set spring.kafka.listener.type=batch.", "acme"),
    ("Globex Kafka runbook: Globex runs a bespoke batch listener; "
     "do not copy this configuration.", "globex"),
]


class NomicEmbeddings(OpenAIEmbeddings):
    """nomic's task prefixes -- the same adapter ch25_rag_chroma.py uses.

    Defined here rather than imported, because importing that module would connect
    to (and seed) the chapter's own Chroma directory as a side effect.
    """

    def embed_documents(self, texts, *args, **kwargs):
        return super().embed_documents(
            [f"search_document: {t}" for t in texts], *args, **kwargs)

    def embed_query(self, text):
        # super().embed_documents, NOT self./super().embed_query: the base class
        # routes embed_query through embed_documents, which would stack both prefixes.
        return super().embed_documents([f"search_query: {text}"])[0]


def build_store() -> Chroma:
    embeddings = NomicEmbeddings(
        base_url=LM_BASE, api_key="lm-studio",
        model="text-embedding-nomic-embed-text-v1.5",
        check_embedding_ctx_length=False,
    )
    store = Chroma(
        collection_name="tenant_demo",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_texts([text for text, _ in DOCS],
                    metadatas=[{"tenant": t} for _, t in DOCS])
    return store


def search_docs(store: Chroma, query: str, tenant: str) -> list:
    """Scoped retrieval. `tenant` comes from the RUN, never from the model.

    In the doc-bot this is `runtime.context.tenant` (@sec-tool-runtime): if it
    were a tool argument the model would fill it in, and the model would be
    choosing whose documents to read.

    The other half is write-side and not shown here because this file has no
    ingestion path: `tenant` must be stamped from the same authenticated
    identity at index time. A metadata field a tenant can choose is a metadata
    field a tenant can choose to be someone else's.

    `filter=` is equality in its flat form; Chroma's operator dicts also work
    (`{"tenant": {"$eq": t}}`, `$in`, `$and`) despite the `dict[str, str]` hint.
    """
    return store.similarity_search(query, k=2, filter={"tenant": tenant})


if __name__ == "__main__":
    store = build_store()
    question = "how do I configure kafka batch consumers?"

    print("unscoped search -- every tenant's runbook is a candidate:")
    for doc in store.similarity_search(question, k=2):
        print(f"  [{doc.metadata['tenant']:6}] {doc.page_content[:52]}")

    print("\nscoped to acme:")
    for doc in search_docs(store, question, tenant="acme"):
        print(f"  [{doc.metadata['tenant']:6}] {doc.page_content[:52]}")

    print("\nscoped to a tenant with nothing indexed:")
    hits = search_docs(store, question, tenant="initech")
    print(f"  {hits} <- empty, not an error. Same empty result the similarity")
    print("     threshold produces, so it takes the same route out: the router")
    print("     falls back to the web branch rather than raising.")
