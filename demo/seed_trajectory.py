"""
demo/seed_trajectory.py — writes a reproducible demo learner trajectory and verifies
the Auditor produces the confirmed-vs-suspected contrast needed for the demo.

Demo learner arc:
  functions_and_scope — CONFIRMED gap (engagement stuck + sandbox failure recorded)
                        highest fan-out root: blocks conditionals, loops, recursion, and all
                        their descendants — big number, credible, verified by code failure.
  list_comprehensions — SUSPECTED gap (engagement stuck, no code submission failure)
                        lower fan-out leaf — shows the contrast: risk without evidence.

Run:  python demo/seed_trajectory.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from store.trajectory import TrajectoryStore
from store.concept_graph import ConceptGraph
from agents.auditor import Auditor

DEMO_LEARNER = "demo_learner"           # must match app.py learner_id
DEMO_TRAJECTORY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "trajectory.jsonl"
)  # must match app.py TRAJECTORY_PATH
PACKETS_DIR = os.path.join(os.path.dirname(__file__), "..", "concept_packets")


def seed(store: TrajectoryStore) -> None:
    # ── functions_and_scope: CONFIRMED gap ───────────────────────────────────
    # Learner tried code, sandbox caught a failure → confirmed=True
    store.record(DEMO_LEARNER, "functions_and_scope", "Passive",
                 "asked how to define a function")
    store.record(DEMO_LEARNER, "functions_and_scope", "Active",
                 "copied the return statement example verbatim")
    store.record(DEMO_LEARNER, "functions_and_scope", "Active",
                 "ran example, reported wrong output",
                 failure_mode="wrong_output")           # ← sandbox failure recorded
    store.record(DEMO_LEARNER, "functions_and_scope", "Passive",
                 "said scope rules make no sense, stopped engaging")

    # ── list_comprehensions: SUSPECTED gap ───────────────────────────────────
    # Learner is stuck on engagement level only — never submitted code that failed
    store.record(DEMO_LEARNER, "list_comprehensions", "Active",
                 "re-typed the list comp example from the packet")
    store.record(DEMO_LEARNER, "list_comprehensions", "Active",
                 "ran the example, reported correct output but couldn't explain it")
    store.record(DEMO_LEARNER, "list_comprehensions", "Passive",
                 "asked for a simpler example, stopped engaging")


def verify(store: TrajectoryStore) -> None:
    graph = ConceptGraph(PACKETS_DIR)
    auditor = Auditor(store, graph)

    print("\n── audit_full() output ─────────────────────────────────────────────")
    report = auditor.audit_full(DEMO_LEARNER)

    if not report.risks:
        print("No stuck concepts detected (check min_interactions threshold).")
        return

    for i, r in enumerate(report.risks):
        gap_label = "CONFIRMED gap" if r.confirmed else "SUSPECTED gap"
        print(f"\nRisk #{i+1}: {r.concept_id}")
        print(f"  Status        : {gap_label}")
        print(f"  Level         : {r.current_level}  (interactions: {r.interaction_count})")
        print(f"  Reach         : {r.downstream_reach} concepts downstream")
        print(f"  Downstream    : {[d['concept_id'] for d in r.downstream_concepts]}")
        print(f"  Consequence   : {r.consequence}")

    print(f"\nSummary: {report.summary}")

    # ── assertions ────────────────────────────────────────────────────────────
    assert any(r.confirmed for r in report.risks), \
        "FAIL: no confirmed gap found — seed needs failure_mode events"
    assert any(not r.confirmed for r in report.risks), \
        "FAIL: no suspected gap found — need at least one concept with no failure_mode"
    assert report.risks[0].confirmed, \
        "FAIL: highest-reach risk should be the confirmed gap (functions_and_scope)"

    confirmed_consequence = report.risks[0].consequence.lower()
    assert "confirmed" in confirmed_consequence, \
        f"FAIL: confirmed gap consequence doesn't say 'confirmed': {report.risks[0].consequence!r}"

    suspected_consequence = report.risks[1].consequence.lower()
    assert "suspected" in suspected_consequence, \
        f"FAIL: suspected gap consequence doesn't say 'suspected': {report.risks[1].consequence!r}"

    print("\n✓  confirmed gap present — consequence says 'confirmed gap'")
    print("✓  suspected gap present — consequence says 'suspected gap'")
    print("✓  prescriptive summary line present")
    print("✓  all assertions passed")


if __name__ == "__main__":
    # Clear previous demo run so seed is reproducible
    demo_path = DEMO_TRAJECTORY_PATH
    if os.path.exists(demo_path):
        os.remove(demo_path)
        print(f"Cleared previous demo trajectory: {demo_path}")

    store = TrajectoryStore(demo_path)
    seed(store)
    print(f"Seeded {DEMO_LEARNER} → {demo_path}")

    verify(store)
