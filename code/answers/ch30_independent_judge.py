"""Chapter 30 Challenge - Gold: judge with a different (independent) model.

The chapter judges every answer with the SAME model that produced it. This is the
weak link: a model grading its own output tends to rubber-stamp its own style and
repeat its own blind spots. The challenge is to point JUDGE_MODEL at a DIFFERENT
(ideally stronger) model and see what changes.

What actually changes -- measured, not guessed, twice. First when the book's local
model was Qwen2.5-32B (that model as the self-judge, gemini-flash-latest as the
independent judge, 2026-08-29); the split below is that pair's. Swapping the judge
to Gemini (via Google AI Studio's OpenAI-compatible endpoint) and re-judging the
chapter's clean eval set changed NOTHING: both judges agreed on
every case, because every answer there is clearly good or clearly bad. Judge choice
only starts to matter on BORDERLINE answers -- fluent but subtly wrong. Feeding the
two judges six hand-crafted borderline candidates, they split on four of them, and
the direction is the point:

  * fluent-hallucination      self=PASS  indep=FAIL   (one invented key, one real
                                                       but unsupported key)
  * correct-but-hedged        self=FAIL  indep=PASS   (penalized the hedging, not a fact)
  * correct-plus-false-addon  self=PASS  indep=FAIL   (missed a false claim)
  * helpful-but-off-instr.    self=PASS  indep=FAIL   (answered instead of refusing)

Three of the four are "the weak self-judge PASSES what the independent judge FAILS":
it waves through invented keys and a false claim that fact-checking catches. The one
reverse (hedged) shows the self-judge is also just noisier -- it confuses an
uncertain TONE with a wrong ANSWER. That is the argument for an independent judge,
and the argument for keeping the eval set sharp: a clean set can't show you any of
this.

Re-measured 2026-09-14 with the book's current model, Qwen3.8-27B, as the
self-judge (Gemini reached through Vertex AI's OpenAI-compatible endpoint,
google/gemini-flash-latest): the clean set still agrees 5/5, Gemini passes the
chapter's judge exam 9/9, and the self-judge now FAILS the fluent hallucination
and the false add-on -- the split moved to ONE candidate, honest-plus-suggestion:
Gemini PASS (the refusal is there, the hint names a documented setting), Qwen3.8
FAIL (the documents call max-poll-records a throughput knob, not a rate limit).
The offline policies below model the FIRST pair; a stronger self-judge closes
the surface-vs-facts gap and leaves the judges arguing about what counts as a
refusal -- which is the point: two judges disagreeing settles nothing, a human
label does.

This file reproduces the first finding deterministically. build_independent_judge() is
the literal swap the challenge asks for -- one ChatOpenAI, three fields, and no
reasoning_effort line, which only the local hybrid model understands (a real Gemini
judge; needs a key and network, so the tests do not call it). Run the chapter's
`check-judge` with the new grader before comparing verdicts. The demonstration below models the two
judges as two transparent POLICIES over each candidate's properties -- an
independent judge that checks FACTS, a self-judge that checks SURFACE -- so the
split is predictable and the tests run offline.

Run:  python code/answers/ch30_independent_judge.py   (see answers/README.md)
"""

import os
from collections import namedtuple

Verdict = namedtuple("Verdict", "verdict reason")   # mirrors the chapter's structured Verdict


# ===========================================================================
# The literal challenge deliverable: point the judge at a DIFFERENT model.
#
# Google AI Studio speaks the OpenAI protocol, so this is the SAME ChatOpenAI the
# chapter already uses -- base_url, model and key change, and the local model's
# reasoning_effort="none" goes away; the judge() prompt is untouched. That is the
# whole swap: the judge stops being the model under test and becomes an independent
# grader.
#
# Needs GEMINI_KEY in the env and network access, so the offline tests below do NOT
# call this -- they use the deterministic policies further down.
#
# No AI Studio key? A Google Cloud project with Vertex AI enabled serves the same
# models through an OpenAI-compatible endpoint, authenticated with an OAuth token
# (`gcloud auth print-access-token`, valid for an hour):
#   base_url = "https://aiplatform.googleapis.com/v1/projects/<project>/locations/global/endpoints/openapi"
#   model    = "google/gemini-flash-latest"
# ===========================================================================
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def build_independent_judge(model: str = "gemini-flash-latest"):
    """Return a real, independent judge_llm pointed at Google AI Studio.

    Verify the current model id against the endpoint's /models list -- the Gemini
    ids churn (gemini-2.5-flash / 2.0-flash already 404 with "no longer available"),
    so prefer the floating alias gemini-flash-latest.
    """
    from langchain_openai import ChatOpenAI   # local import: only needed for the live path

    from pydantic import BaseModel, Field
    from typing import Literal

    class Verdict(BaseModel):            # the chapter's structured schema
        verdict: Literal["PASS", "FAIL"]
        reason: str = Field(description="one-sentence justification")

    return ChatOpenAI(
        base_url=GEMINI_BASE,
        api_key=os.environ["GEMINI_KEY"],   # never hard-code the key
        model=model,
        temperature=0,
    ).with_structured_output(Verdict, method="json_schema")


# ===========================================================================
# Offline reproduction. Each candidate declares the properties a judge would react
# to -- exactly the hand-labeled data the eval set already is (good_answer /
# bad_answer are author labels too). The two judges are two POLICIES over those
# properties, modelling the two real judges' behavior (validated against a live
# qwen2.5-32b-vs-gemini-flash-latest run, September 2026, before the book moved to
# Qwen3.8):
#
#   independent (fact-checking): PASS only when the FACTS hold -- correct keys, no
#       invented key or false claim, and (on an honesty case) an honest refusal.
#   self (surface): blind to facts. It grades STYLE -- it fails a hedged answer or
#       one that pattern-matches the provided bad example, and passes anything else.
#
# The split falls out of the two policies disagreeing on candidates where facts and
# surface point opposite ways (a fluent hallucination; a correct-but-hedged answer).
# ===========================================================================
Candidate = namedtuple(
    "Candidate",
    "tag text honesty facts_correct invents refuses hedged resembles_bad",
)


def independent_judge(c: Candidate) -> Verdict:
    """Fact-checking policy: does the ANSWER actually hold up?"""
    if c.invents:
        return Verdict("FAIL", "states an invented config key or a false claim as fact")
    if c.honesty:
        if c.refuses:
            return Verdict("PASS", "honestly defers instead of inventing an answer")
        return Verdict("FAIL", "should say it is not covered, not answer anyway")
    if c.facts_correct:
        return Verdict("PASS", "key facts / config keys correct")
    return Verdict("FAIL", "missing or wrong key facts")


def self_judge(c: Candidate) -> Verdict:
    """Surface policy: does the answer LOOK confident and unlike the bad example?
    Deliberately blind to whether the facts are right -- that is the blind spot."""
    if c.hedged:
        return Verdict("FAIL", "uncertain / hedged wording")
    if c.resembles_bad:
        return Verdict("FAIL", "reads like the bad example")
    return Verdict("PASS", "fluent, on-topic, and not the bad example")


# The two anchors both judges must agree on (a clean eval set is all anchors), then
# the six borderline candidates that make the judges diverge. Properties below were
# confirmed against the live run; see the module docstring.
CLEARCUT = [
    Candidate("clear-good", "Set spring.kafka.listener.type=batch and take a List "
              "parameter; tune throughput with max-poll-records.",
              honesty=False, facts_correct=True, invents=False, refuses=False,
              hedged=False, resembles_bad=False),
    Candidate("clear-bad", "Just turn on batch mode in the Kafka config file.",
              honesty=False, facts_correct=False, invents=False, refuses=False,
              hedged=False, resembles_bad=True),
]

BORDERLINE = [
    # Right SHAPE and vocabulary, wrong facts -- in two different ways.
    # `spring.kafka.listener.batch-mode` is invented (the real one is
    # listener.type=batch); `fetch.max.bytes` is a REAL Kafka consumer setting that
    # simply is not what bounds a batch and is not in the retrieved material. The
    # second is the harder one to catch: looking it up confirms it exists.
    Candidate("fluent-hallucination",
              "Enable batch consumption with spring.kafka.listener.batch-mode=true, "
              "accept a List<> parameter, and control throughput with fetch.max.bytes.",
              honesty=False, facts_correct=False, invents=True, refuses=False,
              hedged=False, resembles_bad=False),
    # correct key, just omits the throughput knob
    Candidate("correct-but-incomplete",
              "Set spring.kafka.listener.type=batch and take a List parameter in the "
              "listener method.",
              honesty=False, facts_correct=True, invents=False, refuses=False,
              hedged=False, resembles_bad=False),
    # correct facts, wrapped in hedging language
    Candidate("correct-but-hedged",
              "I think you set spring.kafka.listener.type=batch and use a List "
              "parameter, though I'm not totally sure about the throughput setting.",
              honesty=False, facts_correct=True, invents=False, refuses=False,
              hedged=True, resembles_bad=False),
    # correct answer plus one confidently-stated FALSE add-on
    Candidate("correct-plus-false-addon",
              "Set spring.kafka.listener.type=batch, take a List parameter, and tune "
              "max-poll-records. Batch mode also requires enable.auto.commit=true.",
              honesty=False, facts_correct=True, invents=True, refuses=False,
              hedged=False, resembles_bad=False),
    # honesty case: HELPFUL but off-instruction (a real workaround, not "not covered")
    Candidate("helpful-but-off-instruction",
              "Spring Kafka has no built-in rate-limit property, but you can throttle "
              "by lowering max-poll-records and increasing the poll interval.",
              honesty=True, facts_correct=False, invents=False, refuses=False,
              hedged=False, resembles_bad=False),
    # honesty case: defers, and only then hints at a workaround
    Candidate("honest-plus-suggestion",
              "I don't see rate limiting in the reference material, so ask the "
              "platform team; you could explore max-poll-records as a rough workaround.",
              honesty=True, facts_correct=False, invents=False, refuses=True,
              hedged=False, resembles_bad=False),
]


def disagreements(candidates):
    """The tags where the two judges return different verdicts."""
    return [c.tag for c in candidates
            if self_judge(c).verdict != independent_judge(c).verdict]


# --- tests -------------------------------------------------------------------
def test_clearcut_cases_agree():
    # A clean eval set is all clear-cut -> the two judges never diverge, which is
    # exactly why swapping the judge on the chapter's set changed nothing.
    for c in CLEARCUT:
        assert self_judge(c).verdict == independent_judge(c).verdict
    assert disagreements(CLEARCUT) == []


def test_fluent_hallucination_is_where_the_self_judge_fails():
    # The headline: the self-judge rubber-stamps invented config keys because they
    # LOOK right; the independent judge fact-checks them and fails.
    c = next(c for c in BORDERLINE if c.tag == "fluent-hallucination")
    assert self_judge(c).verdict == "PASS"
    assert independent_judge(c).verdict == "FAIL"


def test_hedge_splits_the_other_way():
    # The one reverse: a correct answer that HEDGES. The self-judge penalizes the
    # tone; the independent judge looks past it to the (correct) facts.
    c = next(c for c in BORDERLINE if c.tag == "correct-but-hedged")
    assert self_judge(c).verdict == "FAIL"
    assert independent_judge(c).verdict == "PASS"


def test_false_addon_and_off_instruction_slip_past_the_self_judge():
    # Both slip past the self-judge (fluent, not the bad example) and are caught only
    # by fact-checking: a false claim, and answering where it should refuse.
    for tag in ("correct-plus-false-addon", "helpful-but-off-instruction"):
        c = next(c for c in BORDERLINE if c.tag == tag)
        assert self_judge(c).verdict == "PASS"
        assert independent_judge(c).verdict == "FAIL"


def test_borderline_produces_four_disagreements():
    assert set(disagreements(BORDERLINE)) == {
        "fluent-hallucination",
        "correct-but-hedged",
        "correct-plus-false-addon",
        "helpful-but-off-instruction",
    }


def test_independent_judge_is_the_stricter_one_on_facts():
    # Across every disagreement, count the direction. The majority is
    # self=PASS / indep=FAIL -- the independent judge catches fluent errors the
    # self-judge waves through -- with exactly one reverse (the hedge).
    self_pass_indep_fail = indep_pass_self_fail = 0
    for c in BORDERLINE:
        sv, iv = self_judge(c).verdict, independent_judge(c).verdict
        if sv == "PASS" and iv == "FAIL":
            self_pass_indep_fail += 1
        elif sv == "FAIL" and iv == "PASS":
            indep_pass_self_fail += 1
    assert self_pass_indep_fail == 3    # invented keys, false claim, off-instruction
    assert indep_pass_self_fail == 1    # the hedge, the lone reverse


def test_the_swap_is_only_config():
    # The challenge's actual deliverable: pointing at an independent model is just a
    # different endpoint + model id on the same OpenAI-compatible client.
    assert GEMINI_BASE.endswith("/v1beta/openai/")
    assert "generativelanguage.googleapis.com" in GEMINI_BASE


def _print_table(title, candidates):
    print(title)
    print(f"  {'candidate seam':<28} {'self':<6} {'indep':<6} split?")
    for c in candidates:
        sv, iv = self_judge(c).verdict, independent_judge(c).verdict
        split = "<<< DISAGREE" if sv != iv else "agree"
        print(f"  {c.tag:<28} {sv:<6} {iv:<6} {split}")


if __name__ == "__main__":
    _print_table("Clear-cut (a clean eval set) -- judges agree:", CLEARCUT)
    print()
    _print_table("Borderline (fluent but subtly wrong) -- judges diverge:", BORDERLINE)
    print()
    print(f"Judges disagree on: {disagreements(BORDERLINE)}")
    print("Three of four are self=PASS/indep=FAIL: the independent judge catches "
          "invented keys, a false")
    print("claim, and answering-instead-of-refusing that the self-judge waves "
          "through. Swap the judge, and")
    print("keep the eval set sharp -- a clean set of clear-cut answers can't show "
          "you any of this.")
    print()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
