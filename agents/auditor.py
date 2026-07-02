"""
Auditor agent: reads the learner's ICAP trajectory and concept graph, then produces
a ranked risk report — which stuck concepts will block the most downstream learning.

Two audit paths:
  audit_fast(learner_id)  — pure Python, zero Qwen calls. Safe every turn.
  audit_full(learner_id)  — calls Qwen for consequence narration + summary.
                            Runs only when the stuck concept set has changed since
                            the last full audit (or on explicit UI demand).
"""

import json
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from store.trajectory import TrajectoryStore
from store.concept_graph import ConceptGraph

from models import QWEN_BASE_URL, QWEN_GENERAL as QWEN_MODEL

client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=QWEN_BASE_URL)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ConceptRisk:
    concept_id: str
    current_level: str               # Passive | Active | Struggling
    interaction_count: int
    downstream_reach: int            # count of blocked downstream concepts
    downstream_concepts: list[dict]  # [{concept_id, hop_distance, est_sessions_blocked}]
    consequence: str = ""            # one Qwen sentence; empty in fast path
    confirmed: bool = False          # True = sandbox-verified failure; False = suspected gap


@dataclass
class AuditReport:
    learner_id: str
    risks: list[ConceptRisk] = field(default_factory=list)  # ranked: highest reach first
    summary: str = ""


# ── Qwen helpers ──────────────────────────────────────────────────────────────

def _consequence_for(
    concept_id: str,
    current_level: str,
    downstream_concepts: list[dict],
    confirmed: bool = False,
) -> str:
    """Call Qwen to generate ONE sentence describing the consequence of being stuck.

    The graph is authoritative. Qwen formats a sentence from the given facts;
    it must not reason about, rename, or invent any concept.
    """
    downstream_names = [d["concept_id"] for d in downstream_concepts]
    reach = len(downstream_names)
    total_sessions = sum(d.get("est_sessions_blocked", 3) for d in downstream_concepts)
    gap_label = "confirmed gap" if confirmed else "suspected gap"

    system_prompt = (
        "You are a learning-risk analyst for a Python tutoring system. "
        "You will be given a stuck concept, its engagement level, and an EXACT list of "
        "downstream concepts it blocks. Generate exactly ONE sentence describing the consequence.\n\n"
        "CRITICAL CONSTRAINTS — failure to follow these is wrong output:\n"
        "- Use ONLY the exact concept names listed in blocked_concepts. Verbatim. "
        "Do not add, rename, infer, or invent any concept.\n"
        "- Do not change the count. The blocked_count you receive is authoritative.\n"
        "- You are formatting a sentence from given facts, not reasoning about dependencies.\n"
        "- Use the exact gap_label provided. Do not upgrade a 'suspected gap' to 'confirmed gap'.\n\n"
        "Example shape (FAKE names — never substitute these for real concept names): "
        "\"This is a suspected gap in CONCEPT_A; if unresolved it blocks CONCEPT_B and CONCEPT_C — "
        "approximately N more sessions before those become reachable.\"\n\n"
        "Output valid JSON only: {\"consequence\": \"one sentence\"}"
    )

    user_prompt = (
        f"stuck_concept: {concept_id}\n"
        f"current_level: {current_level}\n"
        f"gap_label: {gap_label}\n"
        f"blocked_count: {reach}\n"
        f"blocked_concepts: {downstream_names if downstream_names else ['none']}\n"
        f"estimated_total_sessions_blocked: {total_sessions}\n\n"
        "Generate ONE consequence sentence using ONLY these facts."
    )

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    data = json.loads(response.choices[0].message.content)
    return data["consequence"]


def _overall_summary(risks: list[ConceptRisk]) -> str:
    """Call Qwen to generate a two-sentence summary with a prescriptive intervention.

    Uses deduplicated union of downstream concept sets so overlapping downstreams
    (e.g. list_comprehensions is inside functions_and_scope's fan-out) are counted
    once, not twice. Both the concept count and the session estimate use the union.
    """
    if not risks:
        return "Learner is on track — no stuck concepts detected."

    top_risk = risks[0]  # already ranked: highest downstream_reach first

    # Deduplicated union — first occurrence of each concept_id wins for sessions
    seen: dict[str, int] = {}
    for r in risks:
        for d in r.downstream_concepts:
            cid = d["concept_id"]
            if cid not in seen:
                seen[cid] = d.get("est_sessions_blocked", 3)

    total_unique_downstream = len(seen)
    total_sessions_at_risk  = sum(seen.values())

    system_prompt = (
        "You are a learning-risk analyst for a Python tutoring system. "
        "Generate exactly TWO sentences:\n"
        "1. Overall risk: state how many UNIQUE downstream concepts are at risk "
        "(use total_unique_downstream — this is already deduplicated so overlapping "
        "downstreams are not double-counted), and the approximate sessions blocked "
        "(use total_sessions_at_risk). Name the top-risk concept.\n"
        "2. Prescriptive action: name the highest-reach concept as THE recommended "
        "intervention point. Reason: fixing it unblocks the most downstream learning "
        "per unit effort.\n\n"
        "CRITICAL CONSTRAINTS:\n"
        "- Use ONLY the exact concept names provided in all_risks. Verbatim. "
        "Do not add, rename, or invent any concept.\n"
        "- Use total_unique_downstream (not the sum of per-gap reaches) as the "
        "total concept count. The deduplication is already done — trust the number.\n"
        "- You are formatting sentences from given facts.\n\n"
        "Example shape (FAKE names — never substitute real concept names from here): "
        "\"The biggest risk is CONCEPT_A — across all gaps, N unique downstream concepts "
        "are at risk (~S sessions of blocked learning). "
        "Highest-leverage action: resolve CONCEPT_A first — it is the root blocking "
        "the most downstream learning per unit effort.\"\n\n"
        "Output valid JSON only: {\"summary\": \"two sentences\"}"
    )

    risks_payload = [
        {
            "concept_id": r.concept_id,
            "current_level": r.current_level,
            "downstream_reach": r.downstream_reach,
        }
        for r in risks
    ]

    user_prompt = (
        f"top_risk_concept: {top_risk.concept_id}\n"
        f"top_risk_level: {top_risk.current_level}\n"
        f"top_risk_downstream_reach: {top_risk.downstream_reach}\n"
        f"total_unique_downstream: {total_unique_downstream}\n"
        f"total_sessions_at_risk: {total_sessions_at_risk}\n"
        f"all_risks: {json.dumps(risks_payload)}\n\n"
        "Generate the two-sentence summary with intervention recommendation."
    )

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    data = json.loads(response.choices[0].message.content)
    return data["summary"]


# ── Auditor ───────────────────────────────────────────────────────────────────

class Auditor:
    def __init__(self, trajectory_store: TrajectoryStore, concept_graph: ConceptGraph):
        self.trajectory_store = trajectory_store
        self.concept_graph = concept_graph
        # (concept_id, current_level, confirmed) → consequence sentence
        # confirmed is included so a suspected→confirmed flip regenerates the sentence.
        self._consequence_cache: dict[tuple[str, str, bool], str] = {}
        # Stuck-set snapshot from the last full audit — used to short-circuit re-narration
        self._last_audited_stuck_set: frozenset[str] = frozenset()
        self._last_full_report: AuditReport | None = None

    def _confirmed_map(self, learner_id: str) -> dict[str, bool]:
        """One file read → {concept_id: confirmed} for all concepts in trajectory."""
        result: dict[str, bool] = {}
        for e in self.trajectory_store.events_for(learner_id):
            cid = e["concept_id"]
            # Any event with a sandbox failure_mode confirms the gap
            if e.get("failure_mode") is not None:
                result[cid] = True
            elif cid not in result:
                result[cid] = False
        return result

    def _build_risks(
        self,
        ranked: list[dict],
        confirmed_map: dict[str, bool],
        with_consequence: bool = False,
    ) -> list[ConceptRisk]:
        risks = []
        for item in ranked:
            cid = item["concept_id"]
            confirmed = confirmed_map.get(cid, False)
            consequence = ""
            if with_consequence:
                cache_key = (cid, item["current_level"], confirmed)
                if cache_key not in self._consequence_cache:
                    self._consequence_cache[cache_key] = _consequence_for(
                        concept_id=cid,
                        current_level=item["current_level"],
                        downstream_concepts=item["downstream_concepts"],
                        confirmed=confirmed,
                    )
                consequence = self._consequence_cache[cache_key]
            risks.append(ConceptRisk(
                concept_id=cid,
                current_level=item["current_level"],
                interaction_count=item["interaction_count"],
                downstream_reach=item["downstream_reach"],
                downstream_concepts=item["downstream_concepts"],
                consequence=consequence,
                confirmed=confirmed,
            ))
        return risks

    def audit_fast(self, learner_id: str) -> AuditReport:
        """
        Pure Python — graph traversal + reach ranking only. Zero Qwen calls.
        Consequence strings are empty; call audit_full() to fill them.
        confirmed is derived from trajectory failure_mode flags (no LLM needed).
        """
        stuck = self.trajectory_store.stuck_concepts(learner_id)
        if not stuck:
            return AuditReport(
                learner_id=learner_id,
                risks=[],
                summary="Learner is on track — no stuck concepts detected.",
            )

        ranked = self.concept_graph.rank_stuck_concepts(stuck)
        confirmed_map = self._confirmed_map(learner_id)
        risks = self._build_risks(ranked, confirmed_map, with_consequence=False)
        return AuditReport(learner_id=learner_id, risks=risks, summary="")

    def audit_full(self, learner_id: str) -> AuditReport:
        """
        Calls Qwen for consequence narration + prescriptive summary.
        Short-circuits entirely if the stuck concept set is unchanged since last call.
        Caches consequence sentences by (concept_id, current_level).
        """
        stuck = self.trajectory_store.stuck_concepts(learner_id)
        if not stuck:
            return AuditReport(
                learner_id=learner_id,
                risks=[],
                summary="Learner is on track — no stuck concepts detected.",
            )

        current_stuck_set = frozenset(s["concept_id"] for s in stuck)

        if (
            self._last_full_report is not None
            and current_stuck_set == self._last_audited_stuck_set
        ):
            return self._last_full_report

        ranked = self.concept_graph.rank_stuck_concepts(stuck)
        confirmed_map = self._confirmed_map(learner_id)
        risks = self._build_risks(ranked, confirmed_map, with_consequence=True)
        summary = _overall_summary(risks)

        self._last_audited_stuck_set = current_stuck_set
        self._last_full_report = AuditReport(
            learner_id=learner_id, risks=risks, summary=summary
        )
        return self._last_full_report

    def audit(self, learner_id: str) -> AuditReport:
        """Backwards-compatible alias — delegates to audit_fast()."""
        return self.audit_fast(learner_id)
