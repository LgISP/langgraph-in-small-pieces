"""Build one index version as its own collection. Runs as a Kubernetes Job.

    INDEX_VERSION=2 DOCS_FILE=/docs/docs.json python -m app.build_index

Two rules, both enforced here rather than by convention:

  * A build is immutable. It creates `docs_v{INDEX_VERSION}` and refuses to run
    if that collection already exists -- a published collection is never written
    to again, and a version number is never reused. A failed build leaves a
    partial collection behind; retire it (`make retire V=N`) and build again.

  * A build says when it is complete. The collection is created with the number
    of documents it will hold, and marked `build=complete` only after the last
    write. The serving process checks both before it accepts traffic, so a
    collection that a crash left half-written is never served.

It cannot change which collection the serving Deployment uses: that is the
Deployment's env, and this Job has no permission to edit Deployments. The build
produces a collection and reports its name; a person or a pipeline deploys it.
"""

import json
import os
import sys

from chromadb.errors import NotFoundError
from langchain_chroma import Chroma

from .embeddings import make_embeddings
from .settings import Settings
from .store import client


def main() -> int:
    s = Settings.from_env() if "ACTIVE_INDEX" in os.environ else Settings(
        active_index="", chroma_host=os.environ.get("CHROMA_HOST", "chroma"),
        chroma_port=int(os.environ.get("CHROMA_PORT", "8000")),
        lm_base=os.environ.get("LM_BASE", "http://host.docker.internal:1234/v1"),
        embed_model=os.environ.get("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"))
    version = os.environ["INDEX_VERSION"]
    with open(os.environ.get("DOCS_FILE", "/docs/docs.json")) as f:
        docs: dict[str, str] = json.load(f)
    name = f"docs_v{version}"
    c = client(s)

    try:
        c.get_collection(name)
    except NotFoundError:
        pass
    else:
        print(f"[build] {name} already exists. Builds are immutable: retire it or pick a new version.")
        return 1

    print(f"[build] {name}: embedding {len(docs)} documents")
    col = Chroma(client=c, collection_name=name,
                 embedding_function=make_embeddings(s.lm_base, s.embed_model),
                 collection_metadata={"hnsw:space": "cosine", "expected_docs": len(docs)})
    col.add_texts(list(docs.values()), ids=list(docs.keys()),
                  metadatas=[{"version": version} for _ in docs])
    raw = c.get_collection(name)                      # the chromadb handle: state, not search
    n = raw.count()
    if n != len(docs):
        print(f"[build] {name}: wrote {n} of {len(docs)} documents -- NOT marking complete")
        return 1
    raw.modify(metadata={"expected_docs": len(docs), "build": "complete"})
    print(f"[build] {name}: done, {n} documents, marked complete. Deploy it with:")
    print(f"        make stage V={version} && make cutover    # the idle colour gets {name}; once it is Ready, the Service moves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
