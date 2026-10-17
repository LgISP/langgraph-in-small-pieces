"""Chapter 28 - Blue-green index deployment: the single-process stand-in.

The chapter's Part 2 runs on a Kubernetes cluster (deploy/ch28-bluegreen/): the
store is a Service, the index build is a Job, the doc-bot is a Deployment, and
which collection is live is one env var on that Deployment, ACTIVE_INDEX. This
file is the same five steps in one Python process, for reading the mechanism
without a cluster:

  * ACTIVE_INDEX is an environment variable, read ONCE into Settings at startup.
  * An App resolves its collection handle at startup and keeps it.
  * deploy(name) is the cut-over: set the variable, construct a fresh App. The
    same call with the old value is the rollback. There is no setter anywhere.

What it cannot show is the cluster's step 7 -- staging a colour onto a collection
that does not exist, which never becomes Ready while the other colour keeps serving.
Here get_collection() uses the LangChain default create_collection_if_not_exists
=True, so a dead name comes back as an empty collection; the Silver answer file
pins exactly that, which is why the real service passes False.

Run:  python code/ch28_bluegreen.py
"""

import os
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

LM_BASE = "http://localhost:1234/v1"        # your local server, not the OpenAI cloud
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_bluegreen")

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


def get_collection(name: str) -> Chroma:
    """Connect to a collection by name (creating it if it does not exist)."""
    return Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


# --------- Deployment config: which collection this process serves ---------
@dataclass(frozen=True)
class Settings:
    """Everything a process needs to know that is decided by the deployment,
    not by the code. Read once, at startup, from the environment."""
    active_index: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(active_index=os.environ["ACTIVE_INDEX"])


class App:
    """One running process. It resolves its collection handle at startup and
    keeps it: nothing in the read path re-decides which version is live."""

    def __init__(self, settings: Settings):
        self.active = settings.active_index
        self.col = get_collection(self.active)

    def query(self, q: str) -> None:
        hit = self.col.similarity_search(q, k=1)[0]
        print(f"  query \"{q}\" -> [{self.active}] {hit.page_content[:34]}")


def deploy(active: str) -> App:
    """Stand-in for a deployment: set the config, start a fresh process.

    The cut-over IS this: no process ever changes its mind about which index
    it serves, so every process is on exactly one whole version. During a
    rolling deploy old and new processes coexist for a moment -- each of them
    whole -- which is the property blue-green is bought for.
    """
    os.environ["ACTIVE_INDEX"] = active
    print(f"  >>> deploy: ACTIVE_INDEX={active}")
    return App(Settings.from_env())


# --------- Build an index (a new collection for a given version) ---------
def build_index(version: str, docs: dict) -> str:
    name = f"docs_v{version}"
    print(f"  [build] building new index collection={name} (embedding {len(docs)} docs)")
    col = get_collection(name)
    col.add_texts(
        list(docs.values()),
        ids=list(docs.keys()),
        metadatas=[{"version": version} for _ in docs],
    )
    return name


def list_collections() -> None:
    client = chromadb.PersistentClient(path=DB_DIR)
    names = sorted(c.name for c in client.list_collections())   # Chroma returns them in no fixed order
    print(f"  collections on disk: {names}")


if __name__ == "__main__":
    # Clean start.
    import shutil
    shutil.rmtree(DB_DIR, ignore_errors=True)

    V1 = {
        "kafka": "Kafka batch consumption: spring.kafka.listener.type=batch",
        "redis": "Redis connection pool: configure via spring.data.redis.*",
    }
    # Version 2 re-chunks / rewrites the content (simulating "docs updated, re-split + re-embed").
    V2 = {
        "kafka": "Kafka batch consumption [v2]: listener.type=batch + max-poll-records tuning + manual ack",
        "redis": "Redis connection pool [v2]: lettuce.pool.max-active / max-idle explained",
        "pg": "PostgreSQL [new]: pgvector extension for vector search",
    }

    print("========== 1) Ship v1: deploy with ACTIVE_INDEX=docs_v1 ==========")
    build_index("1", V1)
    app = deploy("docs_v1")
    list_collections()
    app.query("how to configure kafka batch consumption")

    print("\n========== 2) Build v2 in the background (v1 still serving!) ==========")
    build_index("2", V2)          # build the new version without touching the old one
    list_collections()            # both collections now live on disk at once
    print("  note: the running process still serves v1 (its config has not changed) -")
    app.query("how to configure kafka batch consumption")   # still v1's content

    print("\n========== 3) Once v2 checks out, redeploy with ACTIVE_INDEX=docs_v2 ==========")
    app = deploy("docs_v2")       # cut over: a new process, on the new config
    app.query("how to configure kafka batch consumption")   # now v2's content
    app.query("postgres vector search")                     # v2's new document is reachable too

    print("\n========== 4) Something wrong? Roll back: redeploy with the old value ==========")
    app = deploy("docs_v1")
    app.query("how to configure kafka batch consumption")   # back on v1

    print("\n========== 5) Once the new version is stable, delete the old one to free space ==========")
    app = deploy("docs_v2")
    client = chromadb.PersistentClient(path=DB_DIR)
    client.delete_collection("docs_v1")
    print("  deleted docs_v1")
    list_collections()
