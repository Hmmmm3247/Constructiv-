import json
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from models import QWEN_BASE_URL, QWEN_CODE as QWEN_MODEL

VALID_LEVELS = {"Passive", "Active", "Constructive", "Interactive", "Struggling"}

_SYSTEM_PROMPT = """You are an ICAP engagement classifier for a Python tutoring system. Your ONLY job is to classify the learner's most recent interaction into exactly one engagement state. You do NOT tutor. You output JSON only.

You are given the CURRENT concept the learner is working on. This matters: only engagement WITH THE CURRENT CONCEPT counts.

ICAP states, lowest to highest cognitive engagement:

PASSIVE - receiving, or asking to begin/receive, without producing anything.
  Includes: "ok", "I read it", restating a definition, AND asking how to start a concept
  they have not yet attempted ("how do I start?", "where do I begin?", "can you show me?").
  A fresh start is PASSIVE, never Struggling.

ACTIVE - manipulating existing material: reproducing example code (even if reformatted from
  a one-liner to multi-line Python), running given code and reporting output, quoting the
  slide, trivial variable renames without explanation. If the learner's code implements the
  same function or algorithm already present in content_chunk, it is ACTIVE regardless of
  formatting — expanding a one-liner to full syntax is reproduction, not generation.

CONSTRUCTIVE - generating beyond the source: original code, explaining WHY in own words,
  predicting output before running, reasoning about a self-found error, posing an edge case.

INTERACTIVE - constructive engagement that explicitly builds on the system's immediately
  preceding turn (challenges it, extends it, answers its specific question with synthesis).

STRUGGLING - stuck, frustrated, or disengaging WITHIN THE CURRENT CONCEPT. Requires
  evidence IN THIS CONCEPT: repeated failed attempts at the current concept, explicit
  defeat about the current concept ("this makes no sense", "I give up"), or short
  disengaged replies after prior attempts AT THIS CONCEPT.

CRITICAL RULES:
- RULE F (fresh start): If the learner is asking to BEGIN the current concept and has NO
  prior attempt at THIS concept in the history, classify PASSIVE. Do not classify
  Struggling on a first contact with a concept, no matter how they phrase it.
- RULE C (concept-scoped history): Only history entries for the CURRENT concept inform the
  label. If the learner struggled with a DIFFERENT concept earlier, that does NOT make this
  turn Struggling. Frustration does not carry across concepts.
- Paraphrase is not generation: reproducing/restating the source is ACTIVE, never Constructive.
- INTERACTIVE vs CONSTRUCTIVE tiebreaker: if the last entry in session_history is a system
  turn, ask "Does the learner's message directly answer, challenge, or build on it?" If yes,
  classify INTERACTIVE even if the response contains original reasoning. A learner who
  corrects a wrong claim the system made is INTERACTIVE. Only choose CONSTRUCTIVE if the
  learner ignores the system's last message entirely and starts fresh on a new idea.
- STRUGGLING pattern: if session_history shows 2+ consecutive learner turns with incorrect
  or incomplete code AT THIS CONCEPT, classify STRUGGLING even if the latest looks Active.
- Surrender override: any message with "I give up", "I quit", "I can't do this" about the
  CURRENT concept is STRUGGLING regardless of other content.
- When torn between two levels, choose the LOWER one. Under-crediting is safe.

OUTPUT - JSON only, no other text:
{"icap_level": "Passive|Active|Constructive|Interactive|Struggling", "confidence": 0.0-1.0, "evidence": "one short clause citing the specific signal IN THE CURRENT CONCEPT"}"""


@dataclass
class IcapResult:
    icap_level: str
    confidence: float
    evidence: str


class Classifier:
    def __init__(self):
        self._client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=QWEN_BASE_URL,
        )

    def classify(
        self,
        current_concept: str,
        content_chunk: str,
        session_history: list[dict],
        learner_message: str,
    ) -> IcapResult:
        user_prompt = (
            f"current_concept: {current_concept}\n"
            f"content_chunk: {content_chunk}\n"
            f"session_history: {json.dumps(session_history)}\n"
            f"learner_message: {learner_message}"
        )
        resp = self._client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content)

        missing = {"icap_level", "confidence", "evidence"} - raw.keys()
        if missing:
            raise ValueError(f"Classifier response missing fields: {missing}")
        if raw["icap_level"] not in VALID_LEVELS:
            raise ValueError(f"Unknown icap_level: {raw['icap_level']!r}")

        return IcapResult(
            icap_level=raw["icap_level"],
            confidence=float(raw["confidence"]),
            evidence=raw["evidence"],
        )
