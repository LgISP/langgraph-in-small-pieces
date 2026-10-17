"""The doc-bot's serving process, blue-green edition.

Everything that can go wrong with "which index am I serving" goes wrong here, at
import time, before uvicorn opens a port: the collection ACTIVE_INDEX names must
exist, must be marked complete by the build that made it, and must hold the
number of documents that build said it would. Otherwise the process exits.
Kubernetes turns "the process exits" into a pod that is never Ready, and a
Deployment whose new pods are never Ready keeps the old ones serving and never
retires them. No handler, no retry, no 503 to craft -- fail at startup and let
the platform do what it is for.
"""

import os

from fastapi import FastAPI, HTTPException

from .settings import Settings
from .store import client, open_collection

SETTINGS = Settings.from_env()                      # ACTIVE_INDEX and friends, read once
POD = os.environ.get("HOSTNAME", "local")

# Two handles to one collection. The raw chromadb one answers "what state is this
# collection in" -- it raises NotFoundError for a dead name, and exposes the
# metadata and count the build left behind. The LangChain wrapper is only for
# retrieval, and is opened once the raw handle has said the collection is fit.
raw = client(SETTINGS).get_collection(SETTINGS.active_index)
meta = raw.metadata or {}
if meta.get("build") != "complete":
    raise SystemExit(f"collection {SETTINGS.active_index!r} is not marked complete: a build did not finish")
if raw.count() != meta.get("expected_docs"):
    raise SystemExit(f"collection {SETTINGS.active_index!r} holds {raw.count()} documents, "
                     f"build expected {meta.get('expected_docs')}")
print(f"[startup] {POD} checked {SETTINGS.active_index}: {raw.count()} documents, build complete")
col = open_collection(SETTINGS, SETTINGS.active_index)

app = FastAPI(title="doc-bot (blue-green)")


@app.get("/healthz")
def healthz():
    return {"ok": True, "pod": POD}


@app.get("/readyz")
def readyz():
    # If this handler is running at all, startup succeeded. Readiness is "the
    # process is up".
    return {"ready": True, "pod": POD, "active_index": SETTINGS.active_index}


@app.get("/ask")
def ask(q: str):
    hits = col.similarity_search(q, k=1)
    if not hits:
        raise HTTPException(status_code=404, detail="no documents")
    h = hits[0]
    return {"pod": POD, "active_index": SETTINGS.active_index,
            "version": h.metadata.get("version"), "hit": h.page_content}
