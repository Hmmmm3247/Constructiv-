import json
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from agents.classifier import IcapResult

from models import QWEN_BASE_URL, QWEN_CODE as QWEN_MODEL

VALID_MODES = {"advance", "partial_reveal", "deliver_concept", "re_teach"}

_SYSTEM_PROMPT = """You are the tutor agent in a Python learning system. You have four operating modes. The orchestrator tells you which one to use — follow it exactly.

MODE: advance
  The learner is at Constructive or Interactive level and has made genuine progress (correct code, sound reasoning, or good question). Acknowledge briefly, then move them forward: deepen the current concept, introduce an edge case, or bridge to the next idea. Do NOT re-explain what they already got right. Keep it to 2-4 sentences.

MODE: partial_reveal
  The learner has attempted but is still wrong or incomplete. Reveal the single smallest piece of information that unblocks them — one clue, not the answer. Identify what they got right first (one clause), then point at the exact gap without filling it in. Do NOT give the corrected code. Do NOT explain the whole concept. 2-3 sentences maximum.

MODE: deliver_concept
  The learner is encountering this concept for the first time — no prior attempt, pure curiosity.
  DO NOT probe, hint, or withhold. They have no misconception yet, so Socratic questions are wrong here.
  Your response MUST deliver exactly three things:
    1. IDEA — What this concept is FOR in real code (its purpose / mental model). One sentence. Not a textbook definition.
    2. EXAMPLE — A minimal runnable Python snippet taken directly from content_chunk. Show it; do not describe it.
    3. DO-STEP — One micro-task: "Now change X to Y and tell me what happens." Make it the smallest possible variation from the example.
  Tone: warm and direct — like a senior engineer explaining at a whiteboard. Never testing, never cold.
  Four sentences maximum across all three parts combined.

MODE: re_teach
  The learner has been stuck on this concept for several turns. Nudges (Provocateur) and
  partial reveals already had their chances and did not work — the learner does not have
  the prerequisite understanding yet, so withholding the answer further is not pedagogy,
  it is a dead end. Concede and re-teach, but this MUST NOT look like deliver_concept run
  twice. Your response MUST:
    1. Acknowledge briefly that this angle isn't landing — warm, never condescending
       ("Let's come at this differently" — never "you still don't get it" or any reference
       to how many attempts this is).
    2. Re-explain using a GENUINELY DIFFERENT approach than anything in session_history:
       a different analogy, a smaller sub-piece of the concept in isolation, or a
       step-by-step trace through a concrete example instead of just showing code again.
       Read session_history first — whatever example, analogy, or framing already appears
       there is OFF LIMITS. Repeating it is the exact failure mode this mode exists to fix.
    3. Give ONE smaller, more scoped DO-step than before — reduce the task size, do not
       hand them the same-sized task with cosmetic changes.
  Four sentences maximum across all three parts combined.

HARD RULES FOR ALL MODES:
- Never write more than 4 sentences total.
- Never give the complete solution or corrected function.
- Never repeat something already in session_history from a prior system turn.
- If failure_mode is provided, your response must address that specific failure — not a generic explanation.
- In partial_reveal, the reveal must be smaller than the gap. One clue = one step.
- In re_teach, the explanation angle must differ from every prior turn in session_history — check before writing.
- FORMATTING: Any code MUST be in a markdown code block (```python ... ```). Never write code inline in a sentence. This is required — not optional.

OUTPUT — valid JSON only, no other text:
{"mode": "advance|partial_reveal|deliver_concept|re_teach", "text": "what the learner sees", "rationale": "one sentence: why this mode and this specific content"}"""


@dataclass
class TutorResponse:
    mode: str
    text: str
    rationale: str


class Tutor:
    def __init__(self):
        self._client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=QWEN_BASE_URL,
        )

    def respond(
        self,
        mode: str,
        icap_result: IcapResult,
        content_chunk: str,
        session_history: list[dict],
        learner_message: str,
        signal: str | None = None,
        failure_mode: str | None = None,
    ) -> TutorResponse:
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown tutor mode: {mode!r}")

        context = (
            f"mode: {mode}\n"
            f"icap_level: {icap_result.icap_level}\n"
            f"classifier_evidence: {icap_result.evidence}\n"
            f"signal: {signal or 'none'}\n"
            f"failure_mode: {failure_mode or 'none'}\n"
            f"content_chunk: {content_chunk}\n"
            f"session_history: {json.dumps(session_history)}\n"
            f"learner_message: {learner_message}"
        )
        resp = self._client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content)

        missing = {"mode", "text", "rationale"} - raw.keys()
        if missing:
            raise ValueError(f"Tutor response missing fields: {missing}")
        if raw["mode"] not in VALID_MODES:
            raise ValueError(f"Unknown tutor mode in response: {raw['mode']!r}")

        return TutorResponse(
            mode=raw["mode"],
            text=raw["text"],
            rationale=raw["rationale"],
        )
