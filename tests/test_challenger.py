"""
tests/test_challenger.py

Tests for the Challenger agent — all run with a mocked Qwen client so
no API calls are made and no DASHSCOPE_API_KEY is required.

Covers:
  1. Valid tool call → passes validation, correct text + starter_code
  2. starter_code with eval() → safety check blocks it, fallback skeleton used
  3. starter_code with 'import os' → safety check blocks it
  4. Syntactically broken starter_code → syntax check blocks it, fallback used
  5. Qwen returns no tool call → ValueError raised
  6. _validate_tool_args: no function def → validation fails
  7. Audit log written on every call (pass and fail)
  8. Manual flow guide printed (not a test, just a reminder)
"""

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── allow import from repo root ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Provide a dummy API key so the module-level OpenAI() call doesn't crash
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")

from agents.challenger import (
    Challenger,
    ChallengeResponse,
    _check_safety,
    _minimal_skeleton,
    _validate_tool_args,
    _audit,
    AUDIT_LOG_PATH,
)
from agents.classifier import IcapResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_icap(level="Passive", conf=0.8):
    return IcapResult(icap_level=level, confidence=conf, evidence="test evidence")


def _fake_tool_response(args: dict):
    """Build a mock OpenAI response that looks like a Qwen tool call."""
    tool_call = MagicMock()
    tool_call.function.arguments = json.dumps(args)
    msg = MagicMock()
    msg.tool_calls = [tool_call]
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


_VALID_ARGS = {
    "intro":            "Recursion is all about a function calling itself with a smaller problem.",
    "task_description": "Write count_down(n) that prints n, n-1, ... 1 using recursion.",
    "starter_code":     "def count_down(n):\n    \"\"\"Print n down to 1 recursively.\"\"\"\n    pass\n",
    "constraint":       "Your base case is n == 0, which is different from the factorial example.",
}


# ── Test cases ────────────────────────────────────────────────────────────────

class TestChallengerValidation(unittest.TestCase):

    def test_valid_args_pass(self):
        passed, reason, code = _validate_tool_args(_VALID_ARGS)
        self.assertTrue(passed, f"expected pass but got: {reason}")
        self.assertIn("def count_down", code)

    def test_eval_in_starter_blocked(self):
        args = {**_VALID_ARGS, "starter_code": "def f(x):\n    return eval(x)\n"}
        passed, reason, code = _validate_tool_args(args)
        self.assertFalse(passed)
        self.assertIn("banned", reason)
        # fallback skeleton still has a def
        self.assertIn("def", code)

    def test_import_os_blocked(self):
        args = {**_VALID_ARGS, "starter_code": "import os\ndef f(x):\n    pass\n"}
        passed, reason, code = _validate_tool_args(args)
        self.assertFalse(passed)
        self.assertIn("banned", reason)

    def test_syntax_error_blocked(self):
        args = {**_VALID_ARGS, "starter_code": "def f(x\n    pass\n"}
        passed, reason, code = _validate_tool_args(args)
        self.assertFalse(passed)
        self.assertIn("syntax", reason)
        self.assertIn("def", code)  # fallback

    def test_no_function_def_blocked(self):
        args = {**_VALID_ARGS, "starter_code": "x = 1 + 1\n"}
        passed, reason, code = _validate_tool_args(args)
        self.assertFalse(passed)
        self.assertIn("function definition", reason)

    def test_comment_with_import_not_blocked(self):
        """A comment containing 'import os' should not trigger the safety check."""
        args = {**_VALID_ARGS, "starter_code": "def f(x):\n    # import os — do not use this\n    pass\n"}
        passed, reason, _ = _validate_tool_args(args)
        self.assertTrue(passed, f"comment should not trigger safety: {reason}")


class TestChallengerIntegration(unittest.TestCase):
    """Full generate() flow with mocked Qwen client."""

    def _make_challenger_with_mock(self, tool_args):
        c = Challenger.__new__(Challenger)
        c._client = MagicMock()
        c._client.chat.completions.create.return_value = _fake_tool_response(tool_args)
        return c

    def test_valid_tool_call_produces_response(self):
        c = self._make_challenger_with_mock(_VALID_ARGS)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp_path = Path(f.name)
        with patch("agents.challenger.AUDIT_LOG_PATH", tmp_path):
            result = c.generate(
                concept_id="recursion_basics",
                icap_result=_make_icap(),
                content_chunk="factorial example",
                session_history=[],
                learner_message="I think I understand",
            )
        self.assertIsInstance(result, ChallengeResponse)
        self.assertIn("count_down", result.starter_code)
        self.assertIn("Constraint", result.text)
        self.assertTrue(result.meta["validation_passed"])
        # Audit log was written
        entries = [json.loads(l) for l in tmp_path.read_text().splitlines() if l]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tool"], "create_challenge")
        self.assertTrue(entries[0]["validation_passed"])
        tmp_path.unlink()

    def test_unsafe_tool_call_uses_fallback(self):
        bad_args = {**_VALID_ARGS, "starter_code": "def f(x):\n    exec(x)\n"}
        c = self._make_challenger_with_mock(bad_args)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp_path = Path(f.name)
        with patch("agents.challenger.AUDIT_LOG_PATH", tmp_path):
            result = c.generate(
                concept_id="recursion_basics",
                icap_result=_make_icap(),
                content_chunk="factorial",
                session_history=[],
                learner_message="ok",
            )
        # fallback skeleton used, session not blocked
        self.assertIn("def", result.starter_code)
        self.assertNotIn("exec", result.starter_code)
        self.assertFalse(result.meta["validation_passed"])
        # Audit logged the failure
        entries = [json.loads(l) for l in tmp_path.read_text().splitlines() if l]
        self.assertFalse(entries[0]["validation_passed"])
        tmp_path.unlink()

    def test_no_tool_call_raises(self):
        c = Challenger.__new__(Challenger)
        c._client = MagicMock()
        no_tool_msg = MagicMock()
        no_tool_msg.tool_calls = []
        no_tool_choice = MagicMock()
        no_tool_choice.message = no_tool_msg
        c._client.chat.completions.create.return_value = MagicMock(choices=[no_tool_choice])
        with self.assertRaises(ValueError, msg="should raise when no tool call returned"):
            c.generate("recursion_basics", _make_icap(), "content", [], "message")


class TestSafetyCheck(unittest.TestCase):

    def test_exec_blocked(self):
        safe, reason = _check_safety("exec('rm -rf /')")
        self.assertFalse(safe)

    def test_subprocess_blocked(self):
        safe, reason = _check_safety("import subprocess\nsubprocess.run(['ls'])")
        self.assertFalse(safe)

    def test_open_blocked(self):
        safe, reason = _check_safety("f = open('/etc/passwd')")
        self.assertFalse(safe)

    def test_builtins_blocked(self):
        safe, reason = _check_safety("x = __builtins__")
        self.assertFalse(safe)

    def test_clean_code_passes(self):
        code = "def add(a, b):\n    return a + b\n"
        safe, reason = _check_safety(code)
        self.assertTrue(safe, reason)

    def test_json_import_allowed(self):
        code = "import json\ndef f(s):\n    return json.loads(s)\n"
        safe, reason = _check_safety(code)
        self.assertTrue(safe, reason)


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("MANUAL TEST GUIDE — how to exercise the Challenger live")
    print("=" * 60)
    print("""
1. Clear any existing trajectory:
     rm -f data/trajectory.jsonl

2. Start the app:
     python app.py

3. Select topic: recursion_basics

4. Turn 1 — First contact (triggers deliver_concept):
     Type: "What is recursion?"
     Expected: Tutor teaches IDEA + EXAMPLE + DO-step
     Trace panel: ① Passive → ② deliver_concept → ③ Tutor

5. Turn 2 — Follow-up (triggers Challenger):
     Type: "Ok I think I get it"
     Expected:
       - Challenger generates a new problem (NOT the factorial example)
       - Code accordion opens automatically with a function skeleton
       - Trace panel shows:
           ① Passive/Active
           ② challenge (post-teach)
           ③ 🧩 Challenger
              Tool called: create_challenge
              Constraint: <what Qwen generated>
              ✓ validated

6. Turn 3 — Submit a solution:
     Fill in the starter code, paste in code box, hit Send
     Expected: Classified as Constructive or Active
               If Constructive → advance
               If Active → expose_gap (Provocateur)

7. Check the audit log:
     cat data/tool_audit.jsonl | python -m json.tool
     Should show: tool=create_challenge, validation_passed=true

8. Check the safety check works:
     python -m unittest tests.test_challenger -v
""")
    print("=" * 60)
    unittest.main(verbosity=2)
