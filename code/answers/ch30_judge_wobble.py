"""Chapter 30 Challenge - Silver: quantify the judge's wobble.

The Gotcha claims the judged pass-rate is noisy and the hit-rate is not. This
measures the claim instead of trusting it -- and measures the JUDGE, which means
holding everything else still. The chapter's harness is run once and its per-case
answers are kept; then only the judge is called again, N times, on those fixed
candidates. Two things fall out:

  * per case, a FLIP RATE: how often the verdict on the same answer changed;
  * per run, a pass-rate, whose spread across the runs is the flip rates' shadow
    on the summary line.

The retrieval hit-rate cannot move here at all: nothing is retrieved again. That
is the point of re-judging rather than rerunning -- a full rerun mixes the judge's
noise with retrieval's and the generator's, and you cannot say which one wobbled.

Two further readings, both about the 0.80 gate:

  * a NOISIER judge (more borderline candidates) can push the worst re-judging
    below 0.80, which would force you to lower the bar;
  * the tempting fix -- a BIGGER set -- only shrinks the spread if the new cases
    are as borderline as the old. Pad the set with easy questions and the MEAN
    climbs while a case that fails for good is still failing, now a smaller
    fraction of a healthier-looking number. There is a test for that.

Offline by design: there is no LM Studio and no live model, so the judge's noise
is a deterministic FAKE. A clear candidate is graded the same way every run; a
"borderline" candidate is a seeded coin, PASS with a fixed probability. The coin
is seeded per (case, run), so the WHOLE measurement is reproducible: same seed in,
same verdicts out. That is what lets the tests assert a moving number without
being flaky.

Run:  python code/answers/ch30_judge_wobble.py   (see answers/README.md)
"""

import random
import statistics
from collections import namedtuple

N_RUNS = 5
SEED = 7            # the run's seed; change it and the verdicts change, the hit-rate does not
THRESHOLD = 0.80    # the chapter's regression gate on the judged pass-rate


# ===========================================================================
# The cached candidates. One harness run produced, per case, a retrieved context
# and an answer; that is all a re-judging needs, so that is all a case holds:
#   expect / context -> the deterministic hit-rate (substring in retrieved text)
#   kind             -> how the FAKE judge grades this case's (fixed) answer:
#                       "pass"       clear pass, same verdict every run
#                       "fail"       clear fail, same verdict every run
#                       "borderline" a coin the judge flips -> this is the wobble
#   p_pass           -> for a borderline candidate, the probability of PASS
# ===========================================================================
def clear(cid, expect, context):
    return {"id": cid, "expect": expect, "context": context, "kind": "pass", "p_pass": 1.0}


def broken(cid, expect, context):
    return {"id": cid, "expect": expect, "context": context, "kind": "fail", "p_pass": 0.0}


def borderline(cid, expect, context, p_pass):
    return {"id": cid, "expect": expect, "context": context,
            "kind": "borderline", "p_pass": p_pass}


def honesty(cid):
    # No relevant doc -> out of the hit-rate denominator (expect None); a good bot
    # refuses, which the judge passes every run.
    return {"id": cid, "expect": None, "context": "", "kind": "pass", "p_pass": 1.0}


# BASE: four rock-solid candidates + one the judge is genuinely unsure about.
# hit-rate is frozen at 4/4; pass-rate rides on that single coin.
BASE = [
    clear("w-0001", "listener.type=batch",
          "Kafka batch: set spring.kafka.listener.type=batch and tune max-poll-records."),
    clear("w-0002", "DefaultErrorHandler",
          "Kafka errors: configure a DefaultErrorHandler with a dead-letter queue."),
    clear("w-0003", "lettuce.pool",
          "Redis pool: configure spring.data.redis.lettuce.pool max-active/max-idle."),
    borderline("w-0004", "EXPLAIN",
               "MySQL: add a B+ tree index and run EXPLAIN to inspect the plan.", 0.60),
    honesty("w-0005"),
]

# NOISIER: same size, but TWO borderline candidates. Its worst possible re-judging
# (both coins land FAIL) is 3/5 = 60%, below the gate - a noisier judge forces the
# bar down.
NOISIER = [
    clear("n-0001", "listener.type=batch",
          "Kafka batch: set spring.kafka.listener.type=batch and tune max-poll-records."),
    clear("n-0002", "DefaultErrorHandler",
          "Kafka errors: configure a DefaultErrorHandler with a dead-letter queue."),
    borderline("n-0003", "lettuce.pool",
               "Redis pool: configure spring.data.redis.lettuce.pool max-active.", 0.50),
    borderline("n-0004", "EXPLAIN",
               "MySQL: add a B+ tree index and run EXPLAIN to inspect the plan.", 0.50),
    honesty("n-0005"),
]

# REGRESSED: BASE with its borderline case turned into a candidate that fails for
# good. This is the set the padding test pads.
REGRESSED = BASE[:3] + [
    broken("w-0004", "EXPLAIN",
           "MySQL: add a B+ tree index and run EXPLAIN to inspect the plan."),
] + [BASE[4]]


def padded(base: list, extra_clear: int) -> list:
    """The tempting fix: the same cases plus `extra_clear` easy ones. The spread
    shrinks and the mean climbs -- whether or not anything was fixed."""
    pad = [clear(f"pad-{i:02d}", "index", f"Doc {i}: build an index for lookups.")
           for i in range(extra_clear)]
    return base + pad


# ===========================================================================
# The FAKE judge, re-invoked on a cached candidate. Deterministic given the seed:
# a borderline candidate is a seeded coin keyed by (seed, case, run), so the
# sequence of verdicts is reproducible and independent of how many other cases
# were graded first.
# ===========================================================================
def judge_verdict(case: dict, run_idx: int, seed: int) -> str:
    if case["kind"] == "pass":
        return "PASS"
    if case["kind"] == "fail":
        return "FAIL"
    coin = random.Random(f"{seed}|{case['id']}|{run_idx}").random()
    return "PASS" if coin < case["p_pass"] else "FAIL"


def hit_rate(candidates: list) -> float:
    """The retrieval score of the cached run. It takes no run index on purpose:
    re-judging never retrieves, so there is nothing here that could move."""
    scored = [c for c in candidates if c["expect"] is not None]
    hits = sum(c["expect"] in c["context"] for c in scored)
    return hits / len(scored) if scored else 0.0


def rejudge(candidates: list, run_idx: int, seed: int) -> dict:
    """One re-judging pass: {case id: verdict}."""
    return {c["id"]: judge_verdict(c, run_idx, seed) for c in candidates}


Measurement = namedtuple("Measurement", "verdicts pass_rates hit_rate")


def measure(candidates: list, seed: int = SEED, n_runs: int = N_RUNS) -> Measurement:
    """The whole experiment: N re-judgings of the same cached answers."""
    verdicts = [rejudge(candidates, r, seed) for r in range(n_runs)]
    pass_rates = [sum(v == "PASS" for v in run.values()) / len(candidates) for run in verdicts]
    return Measurement(verdicts, pass_rates, hit_rate(candidates))


# --- reading the measurement ------------------------------------------------
def flip_rate(m: Measurement, cid: str) -> float:
    """How often this case's verdict disagreed with its own majority: 0 for a
    case the judge is sure about, up to 0.5 for a coin."""
    vs = [run[cid] for run in m.verdicts]
    return min(vs.count("PASS"), vs.count("FAIL")) / len(vs)


def worst_possible_pass(candidates: list) -> float:
    """Pass-rate if EVERY borderline coin lands FAIL - the judge's true floor."""
    clear_pass = sum(1 for c in candidates if c["kind"] == "pass")
    return clear_pass / len(candidates)


def margin_is_safe(candidates: list, seed: int = SEED, n_runs: int = N_RUNS,
                   threshold: float = THRESHOLD) -> bool:
    """Does the LOWEST observed pass-rate still clear the gate? That is the margin."""
    return min(measure(candidates, seed, n_runs).pass_rates) >= threshold


def print_table(name: str, candidates: list, m: Measurement) -> None:
    print(f"--- {name}: {len(m.verdicts)} re-judgings of the same answers, seed={SEED} " + "-" * 8)
    print(f"{'case':<10}" + "".join(f"{'run ' + str(i + 1):>8}" for i in range(len(m.verdicts))) + "   flip rate")
    for c in candidates:
        row = "".join(f"{run[c['id']]:>8}" for run in m.verdicts)
        print(f"{c['id']:<10}{row}   {flip_rate(m, c['id']):.0%}")
    print(f"{'pass-rate':<10}" + "".join(f"{p:>8.0%}" for p in m.pass_rates))
    print(f"hit-rate: {m.hit_rate:.0%} on every run — nothing was retrieved again; "
          f"pass-rate: {min(m.pass_rates):.0%}-{max(m.pass_rates):.0%} "
          f"(stdev {statistics.pstdev(m.pass_rates) * 100:.1f} pts)")


# --- tests -------------------------------------------------------------------
def test_hit_rate_cannot_move_when_only_the_judge_reruns():
    m = measure(BASE)
    assert m.hit_rate == 1.0                              # 4/4, honesty case is n/a
    assert measure(BASE, seed=123).hit_rate == m.hit_rate  # the seed drives the judge only


def test_pass_rate_wobbles_across_rejudgings():
    m = measure(BASE)
    assert len(set(m.pass_rates)) > 1                     # the judge is not stable


def test_flip_rate_is_a_per_case_number():
    m = measure(BASE)
    assert all(flip_rate(m, c["id"]) == 0.0 for c in BASE if c["kind"] == "pass")
    assert flip_rate(m, "w-0004") > 0.0                   # only the borderline case flips


def test_measurement_is_reproducible_given_the_seed():
    # Same seed in -> exactly the same verdicts out. This is what makes a moving
    # number safe to assert on: the wobble is seeded, not truly random.
    assert measure(BASE, seed=SEED) == measure(BASE, seed=SEED)
    assert measure(BASE, seed=SEED).pass_rates == [1.0, 1.0, 0.8, 0.8, 1.0]


def test_seed_moves_the_judge_not_the_retrieval():
    a = measure(BASE, seed=7)
    b = measure(BASE, seed=123)
    assert a.hit_rate == b.hit_rate                       # seed cannot move it
    assert a.pass_rates != b.pass_rates                   # seed moves the judge


def test_padding_with_easy_cases_raises_the_mean_but_not_the_failure():
    # REGRESSED has one candidate that fails every time: 4/5 = 80%, on the bar.
    # Pad it with ten easy cases and the mean reads 93% -- and w-0004 still fails
    # on every run. A bigger set shrinks the spread only when the new cases are as
    # borderline as the old; it never fixes a case, it hides one.
    small = measure(REGRESSED)
    big = measure(padded(REGRESSED, extra_clear=10))
    assert min(small.pass_rates) == 0.8
    assert min(big.pass_rates) > min(small.pass_rates)   # the summary got healthier ...
    assert all(run["w-0004"] == "FAIL" for run in big.verdicts)   # ... and nothing was fixed


def test_noisier_judge_can_breach_the_gate_but_base_cannot():
    # BASE's floor (its one coin fails) is exactly 0.80 - the margin is real but thin.
    # NOISIER's floor (two coins fail at once) is 0.60 - below the gate.
    assert worst_possible_pass(BASE) >= THRESHOLD                 # 4/5 = 0.80
    assert worst_possible_pass(NOISIER) < THRESHOLD               # 3/5 = 0.60


def test_margin_recommendation_matches_the_observed_spread():
    # BASE never dips below 0.80 across the re-judgings, so 0.80 is (just) defensible.
    assert margin_is_safe(BASE) is True
    assert min(measure(BASE).pass_rates) == THRESHOLD             # sat exactly on the line


if __name__ == "__main__":
    base = measure(BASE)
    print_table("BASE", BASE, base)
    print()
    print(f"Defending the {THRESHOLD:.0%} gate:")
    print(f"  Across {N_RUNS} re-judgings the hit-rate never moved off {base.hit_rate:.0%}; "
          f"the pass-rate ranged {min(base.pass_rates):.0%}-{max(base.pass_rates):.0%},")
    print(f"  and every point of that range is one case, w-0004, flipping "
          f"{flip_rate(base, 'w-0004'):.0%} of the time.")
    print(f"  A 100% gate would have failed {sum(p < 1.0 for p in base.pass_rates)} of "
          f"{N_RUNS} re-judgings on judge noise alone; {THRESHOLD:.0%} absorbed the wobble.")
    print(f"  The worst re-judging sat at {min(base.pass_rates):.0%}, right on the line: "
          "the margin is thin.")
    print()
    reg = measure(REGRESSED)
    big = measure(padded(REGRESSED, extra_clear=10))
    print("The tempting fix, measured: REGRESSED (w-0004 now fails every time) and the")
    print("same set padded with ten easy cases.")
    print(f"  pass-rate  regressed: {min(reg.pass_rates):.0%}-{max(reg.pass_rates):.0%}"
          f"   padded: {min(big.pass_rates):.0%}-{max(big.pass_rates):.0%}"
          f"   w-0004 in the padded set: {big.verdicts[0]['w-0004']} on every run")
    print("  The mean climbed; nothing was fixed. Grow the set with cases as hard as the")
    print("  ones you have, and read the per-case flips before the mean.")
    print()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
