"""
Challenger: drives learners to Constructive level by generating calibrated problems.

Uses Qwen function calling (tool use) to produce STRUCTURED challenge output:
  - text        : what the learner sees in the chatbot
  - starter_code: Python skeleton pre-populated in the UI code editor
  - meta        : full tool call arguments + validation result (audit + UI trace)

Why tool use instead of response_format JSON?
  Tool use forces Qwen to commit to a named schema with typed parameters.
  The schema communicates intent to the model: each field name is a contract,
  not a hint. It also gives us a clean audit surface — we log the exact
  arguments the model generated before we act on them.

GRC controls (see _validate_tool_args):
  1. Syntax check    — starter_code must parse as valid Python
  2. Safety check    — no dangerous builtins, imports, or exec-family calls
  3. Audit log       — every tool call written to data/tool_audit.jsonl
  4. Fallback        — if validation fails, a safe minimal skeleton is used
                       so the session is never blocked by a bad generation
"""

import ast
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from agents.classifier import IcapResult
from models import QWEN_BASE_URL, QWEN_CODE as QWEN_MODEL

_REPO_ROOT     = Path(__file__).resolve().parents[1]
AUDIT_LOG_PATH = Path(os.environ.get("ICAP_DATA_DIR", str(_REPO_ROOT / "data"))) / "tool_audit.jsonl"

# ── Safety pattern ─────────────────────────────────────────────────────────────
# Applied line-by-line (skipping comments) to every model-generated code string.
# Blocks: exec/eval/compile, dangerous imports, file ops, builtins access.

_SAFETY_RE = re.compile(
    r"\b(eval|exec|compile|__import__)\s*\("
    r"|^\s*(import|from)\s+(os|sys|subprocess|socket|urllib|requests|shutil|builtins|ctypes|pickle)\b"
    r"|\bopen\s*\("
    r"|\b__builtins__\b"
    r"|\bos\.(system|popen|exec|fork|remove|unlink|rmdir|listdir)\b"
    r"|\bsubprocess\."
    r"|\bgetattr\s*\(.+__",
    re.MULTILINE,
)


# ── Tool schema ────────────────────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_challenge",
            "description": (
                "Create a calibrated coding challenge that pushes the learner to "
                "Constructive ICAP level — they must generate, not reproduce."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "2-3 sentences shown to the learner: one warm sentence "
                            "connecting to what they just learned, then one sentence "
                            "stating exactly what the function should do. "
                            "Never use the word 'challenge'."
                        ),
                    },
                    "starter_code": {
                        "type": "string",
                        "description": (
                            "A valid Python function skeleton: def line + docstring + pass. "
                            "Body must be pass only — no logic, no solution. "
                            "No imports. No eval/exec/open. Syntactically valid."
                        ),
                    },
                    "constraint": {
                        "type": "string",
                        "description": (
                            "One rule making copy-paste of the worked example impossible. "
                            "E.g. 'Use a while loop, not for' or 'Start counting from 1, not 0'."
                        ),
                    },
                },
                "required": ["prompt", "starter_code", "constraint"],
            },
        },
    }
]

_SYSTEM_PROMPT = """\
You are a challenge designer for an ICAP Python tutoring system.

Teaching already happened. Your job is to create a coding problem that forces the learner
to GENERATE their own solution (Constructive ICAP level) — not reproduce the worked example.

The challenge must:
1. Use the SAME CONCEPT but a DIFFERENT scenario than the worked example in content_chunk
2. Be small enough to attempt in 5 minutes — one function, one clear task
3. Have a constraint that makes copy-pasting the example impossible

starter_code rules (these are safety requirements, not style preferences):
- Valid Python only: def + docstring + pass. Nothing else in the body.
- No imports of any kind in starter_code
- No eval(), exec(), or open() calls
- The scenario must differ from any example in content_chunk

You MUST call create_challenge. No other output.\
"""


# ── Response dataclass ─────────────────────────────────────────────────────────

@dataclass
class ChallengeResponse:
    text: str                         # shown in chatbot
    starter_code: str                 # pre-populated in code editor
    meta: dict = field(default_factory=dict)
    # meta shape: {tool, concept_id, intro, task_description, constraint,
    #              validation_passed, validation_reason}


# ── Validation ─────────────────────────────────────────────────────────────────

def _check_safety(code: str) -> tuple[bool, str]:
    """Line-by-line scan; skips comment lines. Returns (safe, reason)."""
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if _SAFETY_RE.search(stripped):
            return False, f"banned pattern: {stripped[:80]!r}"
    return True, ""


def _validate_tool_args(args: dict) -> tuple[bool, str, str]:
    """
    Validate model-generated tool arguments before acting on them.
    Returns (passed, reason, safe_starter_code).
    On failure, returns a minimal safe skeleton so the session continues.
    """
    raw_code = args.get("starter_code", "").strip()

    # 1. Syntax
    try:
        ast.parse(raw_code)
    except SyntaxError as e:
        return False, f"syntax error: {e}", _minimal_skeleton(raw_code)

    # 2. Safety
    safe, reason = _check_safety(raw_code)
    if not safe:
        return False, reason, _minimal_skeleton(raw_code)

    # 3. Must have a def — a skeleton without a function definition is useless
    has_def = any(
        isinstance(node, ast.FunctionDef)
        for node in ast.walk(ast.parse(raw_code))
    )
    if not has_def:
        return False, "no function definition found", _minimal_skeleton(raw_code)

    # 4. Body must be skeleton only — no executable logic (solution code leaked in)
    if not _body_is_skeleton(raw_code):
        return False, "starter_code body contains logic — must be pass or docstring only", _minimal_skeleton(raw_code)

    return True, "", raw_code


def _body_is_skeleton(code: str) -> bool:
    """Reject any function whose body contains executable logic (only pass and docstrings allowed)."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Pass):
                    continue
                if (isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)):
                    continue  # docstring
                return False
    return True


def _minimal_skeleton(raw: str) -> str:
    """Best-effort safe fallback: preserve the def line, drop everything else."""
    for line in (raw or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and "(" in stripped:
            return f"{stripped}\n    # TODO: implement this\n    pass\n"
    return "def solution():\n    # TODO: implement this\n    pass\n"


# ── Audit log ──────────────────────────────────────────────────────────────────

def _audit(concept_id: str, args: dict, passed: bool, reason: str) -> None:
    """
    Append one record to data/tool_audit.jsonl.
    Captures: what tool was called, what arguments the model generated,
    whether validation passed, and why it failed if not.
    This is the explainability trail for every model-driven tool call.
    """
    entry = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "agent":              "challenger",
        "tool":               "create_challenge",
        "concept_id":         concept_id,
        "model":              QWEN_MODEL,
        "args_prompt":        args.get("prompt", "")[:300],
        "args_constraint":    args.get("constraint", "")[:200],
        "args_starter_code":  args.get("starter_code", "")[:500],
        "validation_passed":  passed,
        "validation_reason":  reason,
    }
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


# ── Agent class ────────────────────────────────────────────────────────────────

class Challenger:
    def __init__(self):
        self._client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=QWEN_BASE_URL,
        )

    def generate(
        self,
        concept_id: str,
        icap_result: IcapResult,
        content_chunk: str,
        session_history: list[dict],
        learner_message: str,
    ) -> ChallengeResponse:
        # session_history intentionally excluded — Challenger only needs what was taught,
        # not the full conversation. Keeping context small reduces tool-call latency.
        context = (
            f"concept_id: {concept_id}\n"
            f"icap_level: {icap_result.icap_level}\n"
            f"content_chunk (what was just taught):\n{content_chunk}\n\n"
            f"learner_message: {learner_message}"
        )

        resp = self._client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": context},
            ],
            tools=_TOOLS,
            tool_choice={"type": "function", "function": {"name": "create_challenge"}},
            temperature=0.65,
            max_tokens=400,
            timeout=30,
        )

        msg = resp.choices[0].message
        if not msg.tool_calls:
            raise ValueError(
                "Challenger: Qwen did not call create_challenge. "
                "Check that tool_choice is supported by this model endpoint."
            )

        raw_args = json.loads(msg.tool_calls[0].function.arguments)

        # ── Validate before acting ─────────────────────────────────────────
        passed, reason, starter_code = _validate_tool_args(raw_args)
        _audit(concept_id, raw_args, passed, reason)

        meta = {
            "tool":              "create_challenge",
            "concept_id":        concept_id,
            "prompt":            raw_args.get("prompt", ""),
            "constraint":        raw_args.get("constraint", ""),
            "validation_passed": passed,
            "validation_reason": reason,
        }

        if passed:
            text = (
                f"{raw_args['prompt']}\n\n"
                f"⚠️ **Constraint:** {raw_args['constraint']}\n\n"
                f"Complete the function below and hit Send."
            )
        else:
            text = (
                f"Here's a problem to try — write a function using what you just learned.\n\n"
                f"Complete the function below and hit Send."
            )

        return ChallengeResponse(text=text, starter_code=starter_code, meta=meta)
