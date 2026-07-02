"""
Append-only event log for learner ICAP interactions.

Every interaction is recorded as a JSON line — nothing is ever overwritten.
Current state (level, count) is always derived from the log, not stored separately.
This preserves the trajectory arc the Auditor and demo both need.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

STUCK_LEVELS = {"Passive", "Active"}
VALID_LEVELS = {"Passive", "Active", "Constructive", "Interactive", "Struggling"}


class TrajectoryStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ── Write ──────────────────────────────────────────────────────────────

    def record(
        self,
        learner_id: str,
        concept_id: str,
        icap_level: str,
        evidence: str,
        failure_mode: str | None = None,
    ) -> None:
        if icap_level not in VALID_LEVELS:
            raise ValueError(f"Invalid icap_level: {icap_level!r}")
        event = {
            "learner_id": learner_id,
            "concept_id": concept_id,
            "icap_level": icap_level,
            "evidence": evidence,
            "failure_mode": failure_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")

    # ── Read (all derived from the raw log) ───────────────────────────────

    def events_for(self, learner_id: str) -> list[dict]:
        if not self.path.exists():
            return []
        events = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    if e["learner_id"] == learner_id:
                        events.append(e)
        return events

    def _by_concept(self, learner_id: str) -> dict[str, list[dict]]:
        grouped = defaultdict(list)
        for e in self.events_for(learner_id):
            grouped[e["concept_id"]].append(e)
        return grouped

    def current_level(self, learner_id: str, concept_id: str) -> str | None:
        """Most recent ICAP level for this concept. None if no interactions yet."""
        events = [e for e in self.events_for(learner_id) if e["concept_id"] == concept_id]
        return events[-1]["icap_level"] if events else None

    def interaction_count(self, learner_id: str, concept_id: str) -> int:
        return sum(1 for e in self.events_for(learner_id) if e["concept_id"] == concept_id)

    def consecutive_stuck(self, learner_id: str, concept_id: str) -> int:
        """
        Count consecutive most-recent turns on this concept where the learner has NOT
        advanced (Passive, Active, or Struggling). Walking backward, the count resets
        to 0 at the first Constructive/Interactive turn, or at the start of history.
        Used by the router to detect when nudging/provoking has stopped working and a
        genuine re-teach (a different explanation, not another nudge) is needed.
        """
        events = [e for e in self.events_for(learner_id) if e["concept_id"] == concept_id]
        count = 0
        for e in reversed(events):
            if e["icap_level"] in {"Passive", "Active", "Struggling"}:
                count += 1
            else:
                break
        return count

    def trajectory(self, learner_id: str, concept_id: str) -> list[dict]:
        """Ordered interaction history for one concept. Powers the demo arc view."""
        return [e for e in self.events_for(learner_id) if e["concept_id"] == concept_id]

    def stuck_concepts(self, learner_id: str, min_interactions: int = 3) -> list[dict]:
        """
        Concepts where the learner is stuck: Passive or Active after >= min_interactions.
        Returned with interaction_count so the Auditor can rank by severity.
        """
        stuck = []
        for concept_id, events in self._by_concept(learner_id).items():
            if len(events) >= min_interactions and events[-1]["icap_level"] in STUCK_LEVELS:
                stuck.append({
                    "concept_id": concept_id,
                    "current_level": events[-1]["icap_level"],
                    "interaction_count": len(events),
                    "level_history": [e["icap_level"] for e in events],
                    "last_timestamp": events[-1]["timestamp"],
                })
        return stuck

    def summary(self, learner_id: str) -> dict[str, dict]:
        """
        Full per-concept summary for the Auditor's report generation.
        Keys: current_level, interaction_count, level_history, last_timestamp.
        """
        result = {}
        for concept_id, events in self._by_concept(learner_id).items():
            result[concept_id] = {
                "current_level": events[-1]["icap_level"],
                "interaction_count": len(events),
                "level_history": [e["icap_level"] for e in events],
                "last_timestamp": events[-1]["timestamp"],
            }
        return result
