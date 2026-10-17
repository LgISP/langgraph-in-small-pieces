"""Chapter 30 Challenge - Bronze: make the harness catch a real regression.

The gate is green because the system is good. This proves the gate turns red the
moment retrieval breaks. We run the SAME frozen eval set over two versions of the
knowledge base:

  * GOOD      - the full corpus (the shipping system)
  * REGRESSED - the same corpus with the Redis document DELETED (the sabotage)

Deleting the document a case depends on is a retrieval regression you can cause on
ANY retriever: with the doc gone, nothing similar to the Redis question remains, so
its answer can no longer reach the context. (Re-ordering the doc list would NOT do
this - a similarity search ranks by relevance, not insertion order, so the Redis doc
stays the top hit for the Redis question until you actually remove it. There's a test
for that below.)

The chapter's gate watches BOTH metrics - hit-rate AND pass-rate - because a
retrieval regression should block a merge even when the judge's average still looks
fine. That is the whole Bronze lesson: on the regressed run the DETERMINISTIC hit-rate
falls to 75% and trips the retrieval gate, while the judged pass-rate is still sitting
at exactly 80% and would have shipped on its own. The cheap, stable metric fingers the
retrieval break first - read the two metrics in order.

Offline by design: retrieval is a deterministic keyword-overlap score (no embeddings,
no vector store) and the "LLM judge" is a deterministic fake whose PASS/FAIL we can
predict exactly. No LM Studio, no network.

Run:  python code/answers/ch30_catch_regression.py   (see answers/README.md)
"""

import re
import sys
from collections import namedtuple

# --- tokenizing, the shared primitive both metrics lean on -------------------
STOPWORDS = {
    "how", "do", "i", "the", "a", "an", "to", "in", "for", "of", "is", "and",
    "my", "with", "use", "using", "set", "get", "up", "via", "it", "on", "at",
    "by", "or", "if", "as", "does", "what", "when", "that", "this", "from",
    "into", "out", "are", "be", "can", "configure", "one", "you", "your",
}


def content_words(text: str) -> set:
    """Lowercase content tokens: drop punctuation, short words, and stopwords."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 2 and w not in STOPWORDS}


def overlap(a: str, b: str) -> int:
    return len(content_words(a) & content_words(b))


# ===========================================================================
# System under test: a deterministic RAG (retrieve + generate). The regression
# we cause is upstream, in the knowledge base: we DELETE the doc a case needs.
#
# DOCS[2] (Redis pool) is the document eval-0004 depends on - delete it and that
# case's answer can no longer be retrieved.
# ===========================================================================
NOT_COVERED = "Not covered in the docs, please ask the platform team"

DOCS = [
    "Kafka batch consumption: enable spring.kafka.listener.type=batch and take a "
    "List parameter in the listener method; tune throughput with max-poll-records.",
    "Kafka error handling: a failed message in a batch consumer retries the whole "
    "batch and risks duplicates; configure a DefaultErrorHandler with a dead-letter "
    "queue to isolate the bad message.",
    "Redis connection pool: configure it through spring.data.redis.lettuce.pool with "
    "max-active and max-idle to cap the open connections.",
    "MySQL slow query optimization: add a B+ tree index on frequently queried columns "
    "to speed up a slow query, and run EXPLAIN to inspect the execution plan.",
]
REDIS_DOC = DOCS[2]   # the document eval-0004 depends on; deleting it is the sabotage

RELEVANCE_FLOOR = 3   # top doc must clear this overlap, else the bot says NOT_COVERED
JUDGE_MIN = 4         # the judge needs at least this much overlap with the good answer
TOP_K = 2             # retrieval depth, unchanged between the two runs


def rank(question: str, docs: list) -> list:
    """Deterministic stand-in for a similarity search: score each doc by content-word
    overlap with the question, best first. Stable tie-break by position."""
    scored = [(overlap(question, d), idx, d) for idx, d in enumerate(docs)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored


def retrieve(question: str, docs: list) -> list:
    """Return the top-K (score, idx, doc)."""
    return rank(question, docs)[:TOP_K]


def rag_answer(question: str, retrieved: list) -> str:
    """The current system's answer: grounded in whatever retrieval surfaced. If the
    best hit is too weak, refuse honestly instead of inventing config keys."""
    if not retrieved or retrieved[0][0] < RELEVANCE_FLOOR:
        return NOT_COVERED
    return "Based on the docs: " + " ".join(doc for _, _, doc in retrieved)


# ===========================================================================
# The judge: a DETERMINISTIC fake standing in for the chapter's LLM-as-judge.
# Same job the real prompt asks for - does the candidate reach the good answer
# and avoid resembling the bad one - decided here by content-word overlap so the
# verdict is predictable and the tests run offline.
# ===========================================================================
Verdict = namedtuple("Verdict", "verdict reason")   # mirrors the chapter's structured Verdict


def judge(good: str, bad: str, candidate: str) -> Verdict:
    g = overlap(candidate, good)
    b = overlap(candidate, bad or "")
    if g >= JUDGE_MIN and g > b:
        return Verdict("PASS", f"covers the good answer (overlap {g} > bad {b})")
    return Verdict("FAIL", f"misses the good answer (overlap {g} < min {JUDGE_MIN} or <= bad {b})")


# ===========================================================================
# The frozen eval set: data you edit, not labels you maintain. Same five cases as
# the chapter's EVAL_SET, same ids, same order, same `expect` values.
#
# The QUESTIONS are worded a little differently on purpose. The chapter retrieves
# with embeddings; this file's stand-in scores literal content-word overlap, which
# has no idea that "optimize" and "optimization" are the same word. Ask eval-0005
# the chapter's way and the top doc scores 2 against a floor of 3, so the bot
# refuses a question it can plainly answer. The wording here is what that scorer
# needs; the case ids and the jobs they do are the chapter's.
#
#   expect None  -> honesty case (no relevant doc), out of the hit-rate denominator
# ===========================================================================
EVAL_SET = [
    {
        "id": "eval-0001",
        "question": "How do I configure Kafka batch consumption in Spring Boot?",
        "good_answer": "Enable it with spring.kafka.listener.type=batch and take a List "
        "parameter in the listener method; tune throughput with max-poll-records.",
        "bad_answer": "Just turn on batch mode in the Kafka config file.",
        "expect": "listener.type=batch",
    },
    {
        "id": "eval-0002",
        "question": "What happens when one message in a batch consumer fails to process?",
        "good_answer": "By default the whole batch is retried and risks duplicates; "
        "configure a DefaultErrorHandler with a dead-letter queue to isolate the bad "
        "message.",
        "bad_answer": "The failed message is retried on its own until it succeeds.",
        "expect": "DefaultErrorHandler",
    },
    {
        # Honesty case: nothing in DOCS covers rate limiting, so a good bot refuses.
        # `expect is None` keeps it out of the hit-rate denominator.
        "id": "eval-0003",
        "question": "How do I add rate limiting to a Kafka consumer?",
        "good_answer": "This is not covered in the docs; please ask the platform team "
        "and do not invent config keys.",
        "bad_answer": "Set spring.kafka.consumer.rate-limit=100 to add rate limiting.",
        "expect": None,
    },
    {
        # The case we regress: its answer lives ONLY in the Redis doc (DOCS[2]).
        # Delete that doc and this case can no longer retrieve its evidence.
        "id": "eval-0004",
        "question": "How do I limit the number of open connections to Redis?",
        "good_answer": "Cap the open connections with the Redis pool settings "
        "spring.data.redis.lettuce.pool max-active and max-idle.",
        "bad_answer": "Set spring.redis.pool.max-connections=100 to limit Redis "
        "connections.",
        "expect": "lettuce.pool",
    },
    {
        "id": "eval-0005",
        "question": "My database query is slow, how do I speed it up?",
        "good_answer": "Add a B+ tree index on the queried columns to speed up the slow "
        "query and run EXPLAIN to inspect the execution plan.",
        "bad_answer": "Just add more RAM to the database server.",
        "expect": "EXPLAIN",
    },
]

# The gate: BOTH metrics must clear their threshold to ship. Gating the deterministic
# hit-rate, and not the judged pass-rate alone, is precisely what makes it catch a
# retrieval regression the judged average would wave through.
HIT_THRESHOLD = 0.80
PASS_THRESHOLD = 0.80

Config = namedtuple("Config", "name docs")
GOOD = Config("good", DOCS)                                            # the shipping system
REGRESSED = Config("regressed", [d for d in DOCS if d != REDIS_DOC])  # Redis doc deleted

Report = namedtuple(
    "Report",
    "name hits hit_total hit_rate passed total pass_rate retrieval_misses fails gate_ok",
)


def run_harness(config: Config) -> Report:
    """Score the current system over the frozen eval set. Pure: returns a Report."""
    hits = hit_total = passed = 0
    retrieval_misses, fails = [], []
    for c in EVAL_SET:
        retrieved = retrieve(c["question"], config.docs)             # (1) retrieve
        answer = rag_answer(c["question"], retrieved)                # (2) answer
        v = judge(c["good_answer"], c.get("bad_answer"), answer)     # (3) judge

        if c["expect"] is not None:
            hit_total += 1
            got = any(c["expect"] in doc for _, _, doc in retrieved)
            hits += got
            if not got:
                retrieval_misses.append(c["id"])

        if v.verdict == "PASS":
            passed += 1
        else:
            fails.append(c["id"])

    total = len(EVAL_SET)
    hit_rate = hits / hit_total if hit_total else 1.0
    pass_rate = passed / total if total else 0.0
    gate_ok = hit_rate >= HIT_THRESHOLD and pass_rate >= PASS_THRESHOLD
    return Report(config.name, hits, hit_total, hit_rate, passed, total, pass_rate,
                  retrieval_misses, fails, gate_ok)


def gate_exit_code(report: Report) -> int:
    """The one line CI reads: 0 to ship, non-zero to block the merge."""
    return 0 if report.gate_ok else 1


def print_report(report: Report) -> None:
    print(f"--- {report.name} system " + "-" * (48 - len(report.name)))
    print(f"Retrieval hit-rate:  {report.hits}/{report.hit_total} = "
          f"{report.hit_rate * 100:.0f}%   (deterministic; stable across runs)")
    print(f"Answer-quality pass: {report.passed}/{report.total} = "
          f"{report.pass_rate * 100:.0f}%   (LLM-judged)")
    if report.retrieval_misses:
        print(f"Retrieval MISSes:    {report.retrieval_misses}")
    if report.fails:
        print(f"Judge FAILs:         {report.fails}")
    gate = "PASS" if report.gate_ok else "FAIL"
    print(f"Regression gate:     {gate}  (exit {gate_exit_code(report)})")


# --- tests -------------------------------------------------------------------
def test_good_system_passes_the_gate():
    r = run_harness(GOOD)
    assert r.hit_rate == 1.0 and r.pass_rate == 1.0     # 4/4 and 5/5
    assert r.gate_ok is True
    assert gate_exit_code(r) == 0                        # CI would ship


def test_regression_is_deleting_the_target_document():
    # The ONLY difference between the two systems is the deleted Redis doc.
    assert REDIS_DOC in GOOD.docs
    assert REDIS_DOC not in REGRESSED.docs
    assert len(REGRESSED.docs) == len(GOOD.docs) - 1


def test_regression_drops_hit_rate_and_misses_the_redis_doc():
    r = run_harness(REGRESSED)
    assert r.retrieval_misses == ["eval-0004"]          # the Redis doc no longer reaches context
    assert r.hit_rate == 0.75                           # 3/4, down from 4/4
    assert r.hit_rate < HIT_THRESHOLD                   # below the gate's bar


def test_regression_fails_the_gate_and_exits_nonzero():
    r = run_harness(REGRESSED)
    assert r.gate_ok is False
    assert gate_exit_code(r) == 1                        # CI would block the merge


def test_hit_rate_is_the_leading_signal():
    # The heart of Bronze: on the regressed run the JUDGED pass-rate still clears the
    # threshold (it would have shipped), yet the DETERMINISTIC hit-rate does not - so
    # the gate flipped because of hit-rate, caught before the judge's average moved.
    r = run_harness(REGRESSED)
    assert r.pass_rate >= PASS_THRESHOLD                # judge alone: gate would be green
    assert r.hit_rate < HIT_THRESHOLD                   # hit-rate alone: gate is red
    assert r.gate_ok is False                            # the red metric wins


def test_reordering_docs_would_not_regress_retrieval():
    # Why the sabotage is a DELETE, not a shuffle: reordering the corpus changes
    # nothing, because retrieval ranks by relevance, not position.
    import random
    shuffled = list(GOOD.docs)
    random.Random(0).shuffle(shuffled)
    r = run_harness(Config("shuffled", shuffled))
    assert r.hit_rate == 1.0 and r.gate_ok is True       # order is irrelevant to ranking


def test_hit_rate_is_deterministic_across_runs():
    a = run_harness(REGRESSED)
    b = run_harness(REGRESSED)
    assert a.hit_rate == b.hit_rate == 0.75             # a substring check never wobbles


if __name__ == "__main__":
    good, regressed = run_harness(GOOD), run_harness(REGRESSED)
    print_report(good)
    print()
    print_report(regressed)
    print()
    print("Why hit-rate caught it first: the regression is upstream, in retrieval, so "
          "the deterministic hit-rate")
    print("fell to 75% and tripped the gate while the judged pass-rate still read 80% "
          "and would have shipped -")
    print("read the cheap, stable metric first and it fingers a retrieval break before "
          "the judge's average does.")
    print()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
