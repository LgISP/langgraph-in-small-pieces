"""Chapter 22 Challenge — Bronze: bill the loop, not the step.

`print_waterfall()` ends by naming the slowest single step. That is one useful
question and not the only one: `agent_node` appears three times, each visit cheap,
and a cheap node visited often is the classic hidden cost. This adds the other
rollup -- total time per node across the whole run, slowest-first -- and the two
answers do not agree.

The prediction the challenge asks for, before you scroll: summed over its three
visits, does `agent` still rank last?

**It does.** ~58ms a visit times three is ~175ms, and the next node up spends
~307ms in a single visit. Three turns of the loop are not enough to overtake one
web search. That is the honest result on this graph, and it is worth more than the
tidy one: the rollup did not change the ranking here, so this run does not get to
be the story about the hidden cost of loops.

What it does buy you is the number. The file prints the break-even -- how many
visits `agent` would need to pass the node above it -- and on these durations it
is about five (about nine to top the table). That is not a hypothetical: a ReAct
agent taking nine turns on a hard question is an ordinary Tuesday, and at that
point the cheapest node in the graph is the most expensive line in the bill,
with the waterfall still drawing it as nine thin slivers nobody reads.

So: compute the rollup, and believe it even when it agrees with the waterfall.
Which of the two to trust depends on what you are about to do:

  * **slowest single step** answers "where does one request stall?" -- the
    latency question, the one a user feels.
  * **most time overall** answers "where does the run spend its life?" -- the
    throughput and cost question, the one your bill feels.

Optimising the widest bar when the fattest total is elsewhere is how you make a
node 30% faster and the run 2% faster. The loop is what makes them capable of
differing: a node inside a cycle is billed once per visit and drawn once per
visit, and only the rollup adds them up. Whether they actually differ is a fact
about your graph and your inputs, which is exactly why it is measured rather than
assumed.

One more number the rollup makes visible: summed node time is a little under the
waterfall's total. The gap is the runtime's own work between nodes -- the part no
node-level tracer can bill to anybody.

Run:  python code/answers/ch22_bill_the_loop.py   (see answers/README.md)
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ch22_observability as ch21    # noqa: E402


def rollup(trace: list[dict]) -> list[tuple[str, int, float]]:
    """Group the logbook by node: (node, visits, total ms), slowest-first.

    Note what is NOT summed: `start_ms`. Durations add up across visits; start
    times do not add up to anything at all. A rollup that quietly sums every
    numeric column is a report that looks precise and means nothing.
    """
    visits: dict[str, list[float]] = defaultdict(list)
    for t in trace:
        visits[t["node"]].append(t["dur_ms"])
    rows = [(node, len(ds), sum(ds)) for node, ds in visits.items()]
    return sorted(rows, key=lambda r: -r[2])


def print_rollup(trace: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("TRACE rollup (total time per node across the whole run)")
    print("=" * 70)
    total = sum(t["dur_ms"] for t in trace)
    for node, n, ms in rollup(trace):
        bar = "#" * max(1, int(40 * ms / total))
        print(f"{node:<16} {n} visit(s)  {ms:>7.1f}ms  {ms / total:>5.1%}  {bar}")
    print("-" * 70)
    print(f"Summed node time: {total:.1f}ms")


def run_once() -> list[dict]:
    ch21.TRACE.clear()
    ch21.graph.invoke({"question": "How do I configure Kafka?"})
    return list(ch21.TRACE)


_CACHE: list = []


def cached_trace() -> list[dict]:
    if not _CACHE:
        _CACHE.append(run_once())
    return _CACHE[0]


def demo_bill_the_loop():
    trace = cached_trace()
    ch21.print_waterfall()
    print_rollup(trace)

    rows = rollup(trace)
    slowest_step = max(trace, key=lambda t: t["dur_ms"])["node"]
    agent_rank = [r[0] for r in rows].index("agent_node") + 1
    visits, agent_total = next((n, ms) for node, n, ms in rows if node == "agent_node")
    per_visit = agent_total / visits
    above = [ms for node, _, ms in rows if node != "agent_node" and ms > agent_total]

    print(f"\nSlowest single step : {slowest_step}")
    print(f"Most time overall   : {rows[0][0]}")
    print(f"agent_node          : rank {agent_rank} of {len(rows)}, "
          f"{visits} visits x {per_visit:.1f}ms = {agent_total:.1f}ms")
    if above:
        need = min(above) / per_visit
        print(f"\nSo the prediction resolves the boring way: three turns are not enough,")
        print(f"and agent stays last. Break-even is ~{need:.0f} visits -- past that the")
        print(f"cheapest node in the graph tops the bill, while the waterfall keeps")
        print(f"drawing it as {need:.0f} thin slivers nobody reads.")
    else:
        print("\nOn this run the loop node did overtake the others -- rerun and watch")
        print("how much of the ranking is the loop and how much is the sleeps.")

    node_ms = sum(t["dur_ms"] for t in trace)
    wall = max(t["start_ms"] + t["dur_ms"] for t in trace)
    print(f"\nSummed node time {node_ms:.1f}ms vs waterfall total {wall:.1f}ms -- the")
    print(f"{wall - node_ms:.1f}ms gap is the runtime between nodes, billed to no node.")


# --- tests -------------------------------------------------------------------

def test_the_rollup_accounts_for_every_recorded_step():
    trace = cached_trace()
    assert sum(n for _, n, _ in rollup(trace)) == len(trace)


def test_the_rollup_conserves_total_time():
    trace = cached_trace()
    assert abs(sum(ms for _, _, ms in rollup(trace))
               - sum(t["dur_ms"] for t in trace)) < 1e-9


def test_the_rollup_is_sorted_slowest_first():
    totals = [ms for _, _, ms in rollup(cached_trace())]
    assert totals == sorted(totals, reverse=True)


def test_the_loop_node_really_is_visited_more_than_once():
    """Without a repeated visit the challenge has nothing to show."""
    by_node = {node: n for node, n, _ in rollup(cached_trace())}
    assert by_node["agent_node"] >= 3, by_node


def test_agent_is_the_cheapest_node_on_every_single_visit():
    trace = cached_trace()
    agent_steps = [t["dur_ms"] for t in trace if t["node"] == "agent_node"]
    others = [t["dur_ms"] for t in trace if t["node"] != "agent_node"]
    assert max(agent_steps) < min(others)


def test_three_visits_are_not_enough_to_change_the_ranking():
    """The result that actually came out. If a future edit to the chapter's
    sleeps flips this, the test fails and the file's prose gets rewritten --
    which is the point of writing the answer down instead of asserting it."""
    rows = rollup(cached_trace())
    assert rows[-1][0] == "agent_node", [(r[0], round(r[2], 1)) for r in rows]


def test_the_break_even_visit_count_is_reachable_in_a_real_run():
    """The claim the file makes in place of the tidy one: agent overtakes on
    visit count alone, and the count is an ordinary number of ReAct turns rather
    than a thought experiment."""
    rows = rollup(cached_trace())
    visits, agent_total = next((n, ms) for node, n, ms in rows if node == "agent_node")
    runner_up = min(ms for node, _, ms in rows if node != "agent_node")
    need = runner_up / (agent_total / visits)
    assert 4 < need < 25, need


def test_summed_node_time_is_less_than_the_waterfall_total():
    """The runtime's own work between nodes is real and belongs to no node."""
    trace = cached_trace()
    node_ms = sum(t["dur_ms"] for t in trace)
    wall = max(t["start_ms"] + t["dur_ms"] for t in trace)
    assert node_ms < wall


if __name__ == "__main__":
    demo_bill_the_loop()

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
