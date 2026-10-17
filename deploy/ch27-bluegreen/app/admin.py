"""Collection housekeeping against the Chroma server.

    python -m app.admin list                 # every collection, with its build status
    python -m app.admin delete docs_v1       # step 5: retire the old version
    python -m app.admin partial docs_v7 3    # a half-written collection, to test the startup checks
"""

import os
import sys

import chromadb


def main(argv: list[str]) -> int:
    c = chromadb.HttpClient(host=os.environ.get("CHROMA_HOST", "chroma"),
                            port=int(os.environ.get("CHROMA_PORT", "8000")))
    cmd = argv[1] if len(argv) > 1 else "list"
    if cmd == "list":
        for col in sorted(c.list_collections(), key=lambda x: x.name):
            m = col.metadata or {}
            status = "complete" if m.get("build") == "complete" else "NOT complete"
            print(f"  {col.name:10} {col.count()} documents, build {status}")
        return 0
    if cmd == "delete":
        c.delete_collection(argv[2])
        print(f"  deleted {argv[2]}")
        return 0
    if cmd == "partial":
        name, expected = argv[2], int(argv[3])
        col = c.create_collection(name, metadata={"hnsw:space": "cosine", "expected_docs": expected})
        col.add(ids=["only"], embeddings=[[0.1] * 768], documents=["half-written"])
        print(f"  {name}: expected {expected}, holds {col.count()}, no complete mark")
        return 0
    print("usage: admin list | admin delete <collection> | admin partial <collection> <expected>")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
