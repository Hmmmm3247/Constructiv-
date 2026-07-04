"""
orchestrator.py — main pipeline coordinator for the ICAP Python tutoring system.

Given a learner message (and optional code), this module:
  1. Loads the concept packet yaml
  2. Runs the Classifier on the learner message
  3. Runs the sandbox if code was submitted
  4. Calls the router to decide which agent speaks and in what mode
  5. Calls Provocateur or Tutor accordingly
  6. Records the interaction to TrajectoryStore
  7. Runs the Auditor to get a risk report
  8. Returns a structured OrchestratorResponse
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

import yaml

from agents.auditor import Auditor, AuditReport
from agents.challenger import Challenger, ChallengeResponse
from agents.classifier import Classifier, IcapResult
from agents.provocateur import Provocateur
from agents.tutor import Tutor
from router import route
from store.concept_graph import ConceptGraph
from store.sandbox import evaluate
from store.trajectory import TrajectoryStore


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResponse:
    agent_used: str          # "tutor" | "provocateur" | "challenger"
    mode: str                # "advance" | "partial_reveal" | "expose_gap" | "challenge" | ...
    response_text: str       # what the learner sees
    icap_level: str
    confidence: float
    evidence: str
    signal: str | None       # from sandbox, None if no code submitted
    failure_mode: str | None
    route_reason:         str
    starter_code:         str | None
    challenge_meta:       dict | None
    audit_report:         Any
    recommended_concepts: list[dict]  # [{concept_id, status, unlocked_by, downstream_reach}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_content_chunk(packet: dict) -> str:
    obj = packet.get("learning_objective", "")
    ex = packet.get("worked_examples", [{}])[0]
    return f"{obj}\nExample ({ex.get('id', '')}):\n{ex.get('code', '')}"


def _build_session_history(events: list[dict]) -> list[dict]:
    # Include concept tag so the classifier can apply RULE C (cross-concept isolation).
    return [
        {
            "role": "learner",
            "concept": e["concept_id"],
            "text": f"[{e['icap_level']}] {e['evidence']}",
        }
        for e in events[-6:]
    ]


def _detect_func_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except SyntaxError:
        pass
    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, packets_dir: str, trajectory_path: str):
        # Instantiate agents
        self._classifier = Classifier()
        self._provocateur = Provocateur()
        self._tutor = Tutor()
        self._challenger = Challenger()

        # Instantiate stores
        self._trajectory = TrajectoryStore(trajectory_path)
        self._concept_graph = ConceptGraph(packets_dir)

        # Instantiate auditor
        self._auditor = Auditor(self._trajectory, self._concept_graph)

        # Load all packets keyed by concept_id
        self._packets: dict[str, dict] = {}
        for p in Path(packets_dir).glob("*.yaml"):
            packet = yaml.safe_load(p.read_text())
            self._packets[packet["concept_id"]] = packet

    def run(
        self,
        learner_id: str,
        concept_id: str,
        learner_message: str,
        code: str | None = None,
    ) -> OrchestratorResponse:
        # ------------------------------------------------------------------
        # 1. Load packet
        # ------------------------------------------------------------------
        if concept_id not in self._packets:
            raise ValueError(f"Unknown concept_id: {concept_id!r}")

        packet = self._packets[concept_id]
        content_chunk = _build_content_chunk(packet)
        test_cases = packet.get("worked_examples", [{}])[0].get("test_cases", [])

        # ------------------------------------------------------------------
        # 2. Classify
        # ------------------------------------------------------------------
        # Pass ALL recent events (not concept-filtered) so the classifier
        # can apply RULE C: cross-concept history is tagged and ignored.
        all_events = self._trajectory.events_for(learner_id)
        session_history = _build_session_history(all_events)
        icap_result: IcapResult = self._classifier.classify(
            concept_id, content_chunk, session_history, learner_message
        )

        # ------------------------------------------------------------------
        # 3. Sandbox (only if code was submitted)
        # ------------------------------------------------------------------
        signal: str | None = None
        failure_mode: str | None = None
        stderr_text: str | None = None

        if code and code.strip():
            func_name = _detect_func_name(code)
            if func_name is not None:
                try:
                    result = evaluate(code, func_name, test_cases)
                    signal = result["signal"]
                    failure_mode = result["failure_mode"]
                    stderr_text = result["run"].get("stderr") or None
                except Exception:
                    signal = "Error"
                    failure_mode = None
            # If func_name can't be detected: skip sandbox, leave signal/failure_mode None

        # ------------------------------------------------------------------
        # 4. Route
        # ------------------------------------------------------------------
        # All three must be computed BEFORE trajectory.record() so the current turn isn't counted.
        concept_events   = [e for e in all_events if e["concept_id"] == concept_id]
        no_prior_attempt = len(concept_events) == 0
        prior_count      = len(concept_events)
        stuck_streak     = self._trajectory.consecutive_stuck(learner_id, concept_id)
        route_decision = route(
            icap_result.icap_level,
            icap_result.confidence,
            failure_mode,
            no_prior_attempt=no_prior_attempt,
            stuck_streak=stuck_streak,
            prior_count=prior_count,
        )

        # ------------------------------------------------------------------
        # 5. Off-topic early return — no agent call, no trajectory record
        # ------------------------------------------------------------------
        if route_decision.mode == "redirect":
            concept_name = concept_id.replace("_", " ").replace("-", " ")
            redirect_text = (
                f"I'm a Python coding tutor, so that one's outside my lane! "
                f"Happy to help you with **{concept_name}** though — "
                f"what would you like to explore or try?"
            )
            traj_summary = self._trajectory.summary(learner_id)
            return OrchestratorResponse(
                agent_used="tutor", mode="redirect",
                response_text=redirect_text,
                icap_level="Off-topic",
                confidence=icap_result.confidence,
                evidence=icap_result.evidence,
                signal=None, failure_mode=None,
                route_reason=route_decision.reason,
                starter_code=None, challenge_meta=None,
                audit_report=self._auditor.audit_fast(learner_id),
                recommended_concepts=self._concept_graph.recommend(traj_summary, top_n=2),
            )

        # ------------------------------------------------------------------
        # 6. Dispatch
        # ------------------------------------------------------------------
        starter_code:   str | None  = None
        challenge_meta: dict | None = None

        if route_decision.agent == "provocateur":
            provocation = self._provocateur.provoke(
                icap_result,
                content_chunk,
                session_history,
                learner_message,
                execution_error=stderr_text,
            )
            response_text = provocation.text

        elif route_decision.agent == "challenger":
            try:
                # Pass only this concept's events — cross-concept history is irrelevant
                # and bloats the tool-call context. At prior_count==1 this is 1 event.
                concept_history = [
                    {"role": "learner", "text": e["evidence"]}
                    for e in concept_events
                ]
                challenge = self._challenger.generate(
                    concept_id,
                    icap_result,
                    content_chunk,
                    concept_history,
                    learner_message,
                )
                response_text  = challenge.text
                starter_code   = challenge.starter_code
                challenge_meta = challenge.meta
            except Exception:
                # Challenger timed out or tool call failed — fall back to tutor advance
                tutor_response = self._tutor.respond(
                    "advance",
                    icap_result,
                    content_chunk,
                    session_history,
                    learner_message,
                    signal,
                    failure_mode,
                )
                response_text  = tutor_response.text
                route_decision.agent = "tutor"

        else:  # "tutor"
            tutor_response = self._tutor.respond(
                route_decision.mode,
                icap_result,
                content_chunk,
                session_history,
                learner_message,
                signal,
                failure_mode,
            )
            response_text = tutor_response.text

        # ------------------------------------------------------------------
        # 7. Record
        # ------------------------------------------------------------------
        self._trajectory.record(
            learner_id, concept_id, icap_result.icap_level, icap_result.evidence,
            failure_mode=failure_mode,
        )

        # ------------------------------------------------------------------
        # 7. Audit (fast path — pure Python, zero Qwen calls per turn)
        # ------------------------------------------------------------------
        audit_report: AuditReport = self._auditor.audit_fast(learner_id)

        # ------------------------------------------------------------------
        # 8. Recommendation engine
        # ------------------------------------------------------------------
        traj_summary = self._trajectory.summary(learner_id)
        recommendations = self._concept_graph.recommend(traj_summary, top_n=2)

        # ------------------------------------------------------------------
        # 9. Return
        # ------------------------------------------------------------------
        return OrchestratorResponse(
            agent_used=route_decision.agent,
            mode=route_decision.mode,
            response_text=response_text,
            icap_level=icap_result.icap_level,
            confidence=icap_result.confidence,
            evidence=icap_result.evidence,
            signal=signal,
            failure_mode=failure_mode,
            route_reason=route_decision.reason,
            starter_code=starter_code,
            challenge_meta=challenge_meta,
            audit_report=audit_report,
            recommended_concepts=recommendations,
        )
