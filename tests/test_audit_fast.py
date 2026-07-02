"""
Regression guard: audit_fast() must make ZERO Qwen calls.

Run: python tests/test_audit_fast.py   (or: python -m unittest tests.test_audit_fast)
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test-placeholder")


# ── Minimal stub stores ──────────────────────────────────────────────────────

class _FakeTrajectory:
    def stuck_concepts(self, learner_id):
        return [
            {
                "concept_id": "functions_and_scope",
                "current_level": "Passive",
                "interaction_count": 4,
                "level_history": ["Passive"] * 4,
                "last_timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "concept_id": "list_comprehensions",
                "current_level": "Active",
                "interaction_count": 3,
                "level_history": ["Active"] * 3,
                "last_timestamp": "2026-01-01T00:00:00+00:00",
            },
        ]

    def events_for(self, learner_id):
        return [
            # functions_and_scope: has failure_mode → confirmed
            {"concept_id": "functions_and_scope", "icap_level": "Passive",
             "evidence": "asked how to start", "failure_mode": None},
            {"concept_id": "functions_and_scope", "icap_level": "Active",
             "evidence": "copied example", "failure_mode": "wrong_output"},
            # list_comprehensions: no failure_mode → suspected
            {"concept_id": "list_comprehensions", "icap_level": "Active",
             "evidence": "re-typed example", "failure_mode": None},
        ]


class _FakeGraph:
    def rank_stuck_concepts(self, stuck):
        downstream = {
            "functions_and_scope": [
                {"concept_id": "loops_iteration",   "hop_distance": 1, "est_sessions_blocked": 2},
                {"concept_id": "recursion_basics",  "hop_distance": 1, "est_sessions_blocked": 3},
                {"concept_id": "conditionals",      "hop_distance": 1, "est_sessions_blocked": 2},
            ],
            "list_comprehensions": [
                {"concept_id": "generator_expressions", "hop_distance": 1, "est_sessions_blocked": 2},
            ],
        }
        result = []
        for s in stuck:
            cid = s["concept_id"]
            dc = downstream.get(cid, [])
            result.append({**s, "downstream_reach": len(dc), "downstream_concepts": dc})
        result.sort(key=lambda x: -x["downstream_reach"])
        return result


# ── Test case ────────────────────────────────────────────────────────────────

class TestAuditFastZeroQwenCalls(unittest.TestCase):

    def setUp(self):
        # Patch OpenAI at module level before importing Auditor
        patcher = patch("openai.OpenAI")
        patcher.start()
        self.addCleanup(patcher.stop)

        import importlib
        import agents.auditor as auditor_mod
        importlib.reload(auditor_mod)   # ensure patched client is used

        self.mock_client = MagicMock()
        auditor_mod.client = self.mock_client

        from agents.auditor import Auditor
        self.auditor = Auditor(_FakeTrajectory(), _FakeGraph())

    def test_zero_qwen_calls(self):
        report = self.auditor.audit_fast("learner_test")
        calls = self.mock_client.chat.completions.create.call_count
        self.assertEqual(calls, 0, f"audit_fast() made {calls} Qwen call(s) — expected 0")

    def test_risks_present(self):
        report = self.auditor.audit_fast("learner_test")
        self.assertEqual(len(report.risks), 2)

    def test_ranked_by_reach(self):
        report = self.auditor.audit_fast("learner_test")
        self.assertEqual(report.risks[0].concept_id, "functions_and_scope")
        self.assertEqual(report.risks[0].downstream_reach, 3)

    def test_confirmed_derived_from_failure_mode(self):
        report = self.auditor.audit_fast("learner_test")
        # functions_and_scope has a failure_mode event → confirmed
        self.assertTrue(report.risks[0].confirmed)
        # list_comprehensions has no failure_mode → suspected
        self.assertFalse(report.risks[1].confirmed)

    def test_consequence_empty_in_fast_path(self):
        report = self.auditor.audit_fast("learner_test")
        for risk in report.risks:
            self.assertEqual(risk.consequence, "",
                             f"{risk.concept_id}.consequence should be empty in fast path")


if __name__ == "__main__":
    unittest.main(verbosity=2)
