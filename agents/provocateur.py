import json
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from agents.classifier import IcapResult

from models import QWEN_BASE_URL, QWEN_GENERAL as QWEN_MODEL

VALID_TYPES = {"forcing_question", "error_plant", "variant_problem", "reduce_scope"}

_SYSTEM_PROMPT = """You are a surgical interventionist for a Python tutoring system. A classifier has identified the learner's engagement level. Your job is to push them up exactly one level with a single, precise action.

INTERVENTION TYPES — pick exactly one based on ICAP level:

forcing_question (Passive learner):
  The learner is consuming without generating. Give them one tiny, specific task they must produce — not a broad question.
  Good: "Without looking at the example, write just the base case: what should factorial(1) return?"
  Bad: "Can you explain recursion in your own words?" (too open, too broad)

error_plant (Active learner):
  The learner is reproducing but not reasoning. Show them a short snippet with one deliberate bug and ask them to find it.
  The bug must be subtle enough to require reasoning, not just reading. State the bug exists — don't make them guess if one exists.
  Good: "I wrote this — it has one bug. Find it: def factorial(n):\n    if n == 1: return 1\n    return n * factorial(n)"
  Bad: Asking them to write new code (that's a variant_problem, not an error_plant)

variant_problem (Active or mild Constructive learner who needs transfer):
  Same concept, completely different surface. Forces transfer rather than recall.
  Good: "Now write a recursive sum(lst) that adds all numbers in a list."
  Bad: Asking them to explain the same example again

reduce_scope (Struggling learner):
  Strip the problem to the single smallest step they can succeed at. Do NOT ask them to solve the full problem.
  Good: "Forget the whole function. What's the one value where you already know the answer without recursion?"
  Bad: "What have you tried?" or re-explaining the concept

HARD RULES:
- Output must be 1–3 sentences. Never a lecture. Never bullet points.
- Do NOT give the answer. Do NOT explain the concept. Intervene and stop.
- If execution_error is provided, the intervention MUST address that specific error — not a generic provocation.
- Choose the type that matches the ICAP level. Forcing_question for Passive. Error_plant or variant for Active. Reduce_scope for Struggling.

OUTPUT — valid JSON only, no other text:
{"type": "forcing_question|error_plant|variant_problem|reduce_scope", "text": "what the learner sees", "rationale": "one sentence: why this specific intervention"}"""


@dataclass
class Provocation:
    type: str
    text: str
    rationale: str


class Provocateur:
    def __init__(self):
        self._client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=QWEN_BASE_URL,
        )

    def provoke(
        self,
        icap_result: IcapResult,
        content_chunk: str,
        session_history: list[dict],
        learner_message: str,
        execution_error: str | None = None,
    ) -> Provocation:
        context = (
            f"icap_level: {icap_result.icap_level}\n"
            f"classifier_evidence: {icap_result.evidence}\n"
            f"content_chunk: {content_chunk}\n"
            f"session_history: {json.dumps(session_history)}\n"
            f"learner_message: {learner_message}\n"
            f"execution_error: {execution_error or 'none'}"
        )
        resp = self._client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content)

        missing = {"type", "text", "rationale"} - raw.keys()
        if missing:
            raise ValueError(f"Provocateur response missing fields: {missing}")
        if raw["type"] not in VALID_TYPES:
            raise ValueError(f"Unknown provocation type: {raw['type']!r}")

        return Provocation(
            type=raw["type"],
            text=raw["text"],
            rationale=raw["rationale"],
        )
