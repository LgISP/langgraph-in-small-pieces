"""One place that knows how to reach the Chroma server."""

import chromadb
from langchain_chroma import Chroma

from .embeddings import make_embeddings
from .settings import Settings


def client(s: Settings) -> chromadb.ClientAPI:
    return chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)


def open_collection(s: Settings, name: str, must_exist: bool = True) -> Chroma:
    """Attach to an EXISTING collection. `must_exist=True` is the whole point: a
    name with nothing behind it must fail here, at startup, where the readiness
    probe can see it -- not answer with nothing. `must_exist=False` is the
    wrapper's default behaviour, create-on-connect; the Silver challenge uses it
    to build the version of the service that has no safety net."""
    return Chroma(
        client=client(s),
        collection_name=name,
        embedding_function=make_embeddings(s.lm_base, s.embed_model),
        collection_metadata={"hnsw:space": "cosine"},
        create_collection_if_not_exists=not must_exist,
    )
