#!/usr/bin/env python3
"""Point every chapter program at a different model provider.

    python tools/use_llm.py openai      # OpenAI: gpt-4.1-mini + text-embedding-3-small
    python tools/use_llm.py open        # the book's gpt-oss-20b, hosted (DeepInfra), + an open embedder
    python tools/use_llm.py local       # back to LM Studio on this machine (the default)

The book's code names its model in three places, the same three lines in every
file that talks to a model:

    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen/qwen3.8-27b"                # or "openai/gpt-oss-20b", or the embedding model

plus, on the Qwen constructors, the switch that turns the hybrid model's
thinking off:

    reasoning_effort="none"

This script rewrites exactly those lines, under code/ and code/answers/, and
nothing else -- run it with `--diff` to see what it would change. Hosted
non-reasoning models reject `reasoning_effort`, so the switch becomes
`reasoning_effort=None` (which the client omits) and comes back with `local`.
It is idempotent, and `local` puts the original lines back, so the companion
can be switched back and forth without touching git.

Appendix A of the book ("Using Other LLMs") is the guide to this file.
"""
import argparse, os, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "code"

LOCAL_BASE = "http://localhost:1234/v1"
LOCAL_CHAT = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b", "qwen2.5-vl-7b-instruct")
LOCAL_EMBED = "text-embedding-nomic-embed-text-v1.5"

# One entry per provider. `chat` replaces every chat model the book uses (the
# book uses two locally: Qwen for most chapters, gpt-oss for the tool-calling
# ones -- one hosted model serves both roles); `embed` replaces the embedding
# model. A None base_url means "the provider's default", and a None api_key
# means "the library reads it from OPENAI_API_KEY".
PROVIDERS = {
    "local": dict(base_url=LOCAL_BASE, api_key='"lm-studio"', chat=None, embed=LOCAL_EMBED,
                  key_env=None),
    # gpt-4.1-mini, not gpt-5-mini: the book sets temperature=0 everywhere for
    # reproducibility, and the GPT-5 family accepts only the default temperature.
    # Explicit endpoint and key rather than the library defaults: some files
    # build a probe URL from the base, and an explicit None is not "unset".
    "openai": dict(base_url="https://api.openai.com/v1", api_key='os.environ["OPENAI_API_KEY"]',
                   chat="gpt-4.1-mini", embed="text-embedding-3-small", key_env="OPENAI_API_KEY"),
    # The book's own open chat model, gpt-oss-20b, hosted -- with tool calling --
    # by DeepInfra through an OpenAI-compatible endpoint, plus an open embedding
    # model from the same provider (nomic-embed-text-v1.5 is not offered there;
    # Fireworks has it, without tool calling on gpt-oss). One key for both.
    "open": dict(base_url="https://api.deepinfra.com/v1/openai",
                 api_key='os.environ["DEEPINFRA_API_KEY"]',
                 chat="openai/gpt-oss-20b", embed="BAAI/bge-base-en-v1.5",
                 key_env="DEEPINFRA_API_KEY"),
}

ALL_BASES = {LOCAL_BASE} | {p["base_url"] for p in PROVIDERS.values() if p["base_url"]}
BASE_OVERRIDE = None                    # --base-url: any OpenAI-compatible endpoint (a proxy, a gateway)
ALL_KEYS = {'"lm-studio"', "None"} | {p["api_key"] for p in PROVIDERS.values() if p["api_key"]}
KNOWN_CHAT = set(LOCAL_CHAT) | {p["chat"] for p in PROVIDERS.values() if p["chat"]}
KNOWN_EMBED = {LOCAL_EMBED} | {p["embed"] for p in PROVIDERS.values()}

# The book's comments on the two local lines; leaving local swaps them for a
# note and coming back restores them, so `local` is byte-for-byte the original.
LOCAL_COMMENTS = {"# your local server, not the OpenAI cloud": "# use_llm.py: the provider's endpoint",
                  "# the local server ignores the key": "# use_llm.py: key from the environment"}
MEMO = "  # use_llm.py: was "
IMPORT_OS = "import os  # use_llm.py"

_base_re = re.compile(r'(\bbase_url=|\b(?:LM_BASE|BASE_URL)\s*=\s*)(None|"[^"]*")')
_key_re = re.compile(r'(\bapi_key=|\bAPI_KEY\s*=\s*)(None|"[^"]*"|os\.environ\[[^\]]+\])')
_model_re = re.compile(r'(\bmodel=|\b(?:CHAT_MODEL|MODEL|JUDGE_MODEL|WEAK_MODEL|CAPABLE_MODEL|EMBED_MODEL)\s*=\s*)"([^"]+)"')
_reason_re = re.compile(r'(\breasoning_effort=)("none"|None)')


def rewrite(text: str, prov: dict) -> str:
    local = prov["chat"] is None
    out = []
    for line in text.split("\n"):
        if line == IMPORT_OS:
            continue                          # ours; re-added below if still needed
        # a remembered local chat model, from an earlier switch
        was = None
        if MEMO.strip() in line:
            line, memo = line.split(MEMO, 1)
            was = memo.strip()

        def base(m):
            v = m.group(2)
            if v.strip('"') not in ALL_BASES and v != "None" and v.strip('"') != BASE_OVERRIDE:
                return m.group(0)             # some other URL: not ours to touch
            new = f'"{prov["base_url"]}"' if local else ("None" if prov["base_url"] is None else f'"{prov["base_url"]}"')
            return m.group(1) + new
        def key(m):
            v = m.group(2)
            if v not in ALL_KEYS:
                return m.group(0)
            new = '"lm-studio"' if local else ("None" if prov["api_key"] is None else prov["api_key"])
            return m.group(1) + new
        chat_seen = []
        def model(m):
            name = m.group(2)
            if name in KNOWN_EMBED or (m.group(1).startswith("EMBED_MODEL")):
                return m.group(1) + f'"{prov["embed"]}"'
            if name in KNOWN_CHAT or was:
                original = was or name
                chat_seen.append(original)
                return m.group(1) + f'"{original if local else prov["chat"]}"'
            return m.group(0)
        before = line
        line = _base_re.sub(base, line)
        if local:
            # a --base-url endpoint this script never listed: recognised by the
            # note or marker it left, and restored to the local server
            if line == before and "use_llm.py" in line and _base_re.search(line):
                line = _base_re.sub(lambda m: m.group(1) + f'"{prov["base_url"]}"', line)
            if line.endswith("  # use_llm.py"):
                line = line[: -len("  # use_llm.py")]
        for orig, note in LOCAL_COMMENTS.items():
            line = line.replace(note, orig) if local else line.replace(orig, note)
        # a base line we changed carries a marker (unless its note already says so)
        if not local and line != before and "use_llm.py" not in line:
            line += "  # use_llm.py"
        line = _key_re.sub(key, line)
        line = _model_re.sub(model, line)
        # the thinking switch: only a local hybrid model has one
        line = _reason_re.sub(lambda m: m.group(1) + ('"none"' if local else "None"), line)
        for orig, note in LOCAL_COMMENTS.items():
            line = line.replace(note, orig) if local else line.replace(orig, note)
        if chat_seen and not local:
            line += MEMO + chat_seen[0]
        out.append(line)
    text2 = "\n".join(out)
    if prov["api_key"] and "os.environ[" in prov["api_key"] and "os.environ[" in text2 \
            and not re.search(r"^import os\b|^import .*\bos\b|^from os\b", text2, re.M):
        # before the first real import statement (a docstring can contain
        # example imports, so this is found by parsing, not by pattern)
        import ast
        tree = ast.parse(text2)
        first = next((n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))), None)
        at = (first.lineno if first else (tree.body[0].lineno if tree.body else 1)) - 1
        lines = text2.split("\n"); lines.insert(at, IMPORT_OS); text2 = "\n".join(lines)
    return text2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("provider", choices=sorted(PROVIDERS))
    ap.add_argument("--diff", action="store_true", help="show the changes, write nothing")
    ap.add_argument("--root", default=str(ROOT), help="code directory (default: the companion's code/)")
    ap.add_argument("--base-url", help="use this endpoint instead of the provider's (a proxy or gateway)")
    a = ap.parse_args()
    prov = dict(PROVIDERS[a.provider])
    global BASE_OVERRIDE
    if a.base_url:
        BASE_OVERRIDE = a.base_url; prov["base_url"] = a.base_url
    changed = 0
    for path in sorted(pathlib.Path(a.root).rglob("*.py")):
        old = path.read_text()
        new = rewrite(old, prov)
        if new == old:
            continue
        changed += 1
        if a.diff:
            import difflib
            sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), str(path), str(path)))
        else:
            path.write_text(new)
    print(f"{'would change' if a.diff else 'changed'} {changed} file(s) -> {a.provider}"
          + (f"; set {prov['key_env']}" if prov["key_env"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
