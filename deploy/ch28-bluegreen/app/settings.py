"""Everything the deployment decides and the code does not.

Read once, at process start, from the environment. There is no setter: the
only way a running process's settings change is that the process is replaced.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    active_index: str        # which Chroma collection this process serves
    chroma_host: str
    chroma_port: int
    lm_base: str             # the embedding server (LM Studio on the host, in the demo)
    embed_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            active_index=os.environ["ACTIVE_INDEX"],
            chroma_host=os.environ.get("CHROMA_HOST", "chroma"),
            chroma_port=int(os.environ.get("CHROMA_PORT", "8000")),
            lm_base=os.environ.get("LM_BASE", "http://host.docker.internal:1234/v1"),
            embed_model=os.environ.get("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
        )
