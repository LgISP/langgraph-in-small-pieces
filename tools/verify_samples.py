#!/usr/bin/env python3
"""Run every sample that can run without a model, and make it prove itself.

Three populations, three standards of proof:

  code/*.py          the listings the book prints. Must exit 0.
  code/test_*.py     pytest suites -- any code/ file that defines test_* and has
                     no __main__ block. Running one as a script imports it,
                     defines some functions and exits 0 WITHOUT RUNNING A TEST,
                     which this tool used to count as verified. They go to
                     pytest, and collecting nothing is a failure, not a pass.
  code/answers/*.py  the worked solutions. Must exit 0, and any that defines
                     test_* functions must also report "all N tests passed" --
                     exiting 0 is not enough there, because a solution whose
                     asserts silently stopped running still exits 0. Two of the
                     43 are demonstrations with no test functions; the book links
                     those as "Worked solution:" rather than "with tests", so they
                     are held only to exiting 0.

Anything that reaches for a model server is skipped by inspection, not by trying
it and catching the failure -- a skip must be a decision, not an accident.
"""
import argparse, os, re, subprocess, sys, time

NEEDS_MODEL = re.compile(r"ChatOpenAI|localhost:1234|OPENAI_API_KEY")

# A file can reach a model *transitively*, which the regex above cannot see:
# nothing in its own source names ChatOpenAI, yet running it reaches a server.
# The regex is a heuristic, not a proof, so every known transitive dependency is
# DECLARED here by name -- a skip must be a decision, not an accident, and a file
# whose inline strings later change must not silently start running unguarded.
INDIRECT = {
    "code/ch21_async.py":
        "times model calls against the local server; six sections, about two minutes",
    "code/ch20_smoke_client.py":
        "launches ch20_serving.py, whose /ask path needs a model server",
    # ch22_callback_tracer's Part 2 builds a real model graph; it also names
    # ChatOpenAI inline (so the regex catches it too), but declare it regardless.
    "code/ch22_callback_tracer.py":
        "Part 2 builds a real model graph (ChatOpenAI on localhost:1234)",
    # Not a script. Its own docstring says so: another answer launches it over
    # stdio. Run directly it waits on stdin for a client that never connects,
    # and the only thing that ended it was this tool's 300s timeout -- five
    # minutes of a "failure" that was the harness misreading what the file is.
    "code/answers/ch23_server_plus_count.py":
        "an MCP server, launched over stdio by ch23_a_tool_the_client_never_heard_of.py",
    # langgraph-checkpoint-redis is deliberately NOT in requirements.txt: the
    # freeze is what Chapter 0 tells the reader to install, and a running Redis
    # is not. Declared rather than pinned, so the offline run says why instead
    # of failing on an import that was never promised.
    "code/answers/ch20_shared_checkpointer_redis.py":
        "needs a running Redis and langgraph-checkpoint-redis (outside the pinned freeze)",
}
DEFINES_TESTS = re.compile(r"^def test_", re.M)
RUNS_ITSELF = re.compile(r"__main__")


def is_suite(src):
    """A pytest suite, as opposed to a listing that happens to carry assertions.

    The distinction is whether the file runs its own tests when executed. The
    answers and a few listings define test_* AND call them from __main__, so
    they prove themselves as scripts. A file with no __main__ cannot: it exits 0
    having done nothing, and exit 0 was the whole bar here.
    """
    return bool(DEFINES_TESTS.search(src)) and not RUNS_ITSELF.search(src)
INTERACTIVE = re.compile(r"isatty")
PASSED = re.compile(r"all (\d+) tests? passed")


def classify(path, rel):
    if rel in INDIRECT:
        return INDIRECT[rel]
    src = open(path, encoding="utf-8").read()
    if NEEDS_MODEL.search(src):
        return "needs a model server"
    if INTERACTIVE.search(src):
        return "waits for a person at a terminal"
    return None


def run(paths, root, py, timeout, expect_tests):
    ok = failed = skipped = 0
    problems = []
    for p in sorted(paths):
        rel = os.path.relpath(p, root)
        why = classify(p, rel)
        if why:
            skipped += 1
            print("  SKIP %-46s %s" % (rel, why))
            continue
        t0 = time.time()
        try:
            r = subprocess.run([py, rel], cwd=root, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            failed += 1
            problems.append((rel, "timed out after %ds" % timeout))
            print("  FAIL %-46s timed out" % rel)
            continue
        dt = time.time() - t0
        if r.returncode != 0:
            failed += 1
            tail = (r.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
            problems.append((rel, "exit %d: %s" % (r.returncode, tail[0])))
            print("  FAIL %-46s exit %d" % (rel, r.returncode))
            continue
        wants_tests = expect_tests and DEFINES_TESTS.search(open(p, encoding="utf-8").read())
        if wants_tests:
            m = PASSED.search(r.stdout)
            if not m:
                failed += 1
                problems.append((rel, "ran clean but never reported its tests passing"))
                print("  FAIL %-46s no 'all N tests passed'" % rel)
                continue
            print("  ok   %-46s %s tests  %4.1fs" % (rel, m.group(1), dt))
        elif expect_tests:
            print("  ok   %-46s (demo, no tests) %4.1fs" % (rel, dt))
        else:
            print("  ok   %-46s %4.1fs" % (rel, dt))
        ok += 1
    return ok, failed, skipped, problems


def run_suites(paths, root, py, timeout):
    """pytest, with `collected nothing` treated as the failure it is.

    pytest already distinguishes these; the harness just has to stop accepting
    them. 5 is "no tests ran" and 4 is "the path is wrong" -- the second is how
    a CI step spent months green while pointing at a file that did not exist.
    """
    ok = failed = skipped = 0
    problems = []
    for p in sorted(paths):
        rel = os.path.relpath(p, root)
        why = classify(p, rel)
        if why:
            skipped += 1
            print("  SKIP %-46s %s" % (rel, why))
            continue
        t0 = time.time()
        try:
            r = subprocess.run([py, "-m", "pytest", rel, "-q"], cwd=root,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            failed += 1
            problems.append((rel, "timed out after %ds" % timeout))
            print("  FAIL %-46s timed out" % rel)
            continue
        dt = time.time() - t0
        note = {4: "bad path or usage error -- nothing was collected",
                5: "collected no tests"}.get(r.returncode)
        if r.returncode != 0:
            tail = (r.stdout or r.stderr or "").strip().splitlines()[-1:] or ["(no output)"]
            problems.append((rel, note or "exit %d: %s" % (r.returncode, tail[0])))
            failed += 1
            print("  FAIL %-46s %s" % (rel, note or "exit %d" % r.returncode))
            continue
        m = re.search(r"(\d+) passed", r.stdout)
        if not m:
            failed += 1
            problems.append((rel, "exited 0 but reported no passing tests"))
            print("  FAIL %-46s exit 0 with nothing passing" % rel)
            continue
        print("  ok   %-46s %s passed  %4.1fs" % (rel, m.group(1), dt))
        ok += 1
    return ok, failed, skipped, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--only", choices=["code", "answers"])
    a = ap.parse_args()

    import glob
    tot_ok = tot_fail = tot_skip = 0
    allp = []
    if a.only != "answers":
        every = glob.glob(os.path.join(a.root, "code", "*.py"))
        suites = [p for p in every if is_suite(open(p, encoding="utf-8").read())]
        listings = [p for p in every if p not in suites]
        print("listings printed in the book (must exit 0):")
        o, f, s, pr = run(listings, a.root, a.python, a.timeout, expect_tests=False)
        tot_ok += o; tot_fail += f; tot_skip += s; allp += pr
        print()
        print("pytest suites (must collect tests AND pass them):")
        o, f, s, pr = run_suites(suites, a.root, a.python, a.timeout)
        tot_ok += o; tot_fail += f; tot_skip += s; allp += pr
        print()
    if a.only != "code":
        print("worked solutions (must exit 0 AND report their tests passing):")
        ps = glob.glob(os.path.join(a.root, "code", "answers", "*.py"))
        o, f, s, pr = run(ps, a.root, a.python, a.timeout, expect_tests=True)
        tot_ok += o; tot_fail += f; tot_skip += s; allp += pr
        print()

    print("%d verified, %d FAILED, %d skipped" % (tot_ok, tot_fail, tot_skip))
    if allp:
        print("\nFAILURES:")
        for rel, why in allp:
            print("  %-46s %s" % (rel, why))
    return 1 if tot_fail else 0


if __name__ == "__main__":
    sys.exit(main())
