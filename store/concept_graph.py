"""
Builds a directed graph from concept packets and answers the Auditor's questions:
  - What concepts are transitively reachable from a stuck concept? (advance_to edges)
  - What is the downstream reach (fan-out count) of a stuck concept?
  - How should stuck concepts be ranked by severity?

Reach, not hop-distance, is the ranking metric — a foundational gap that silently
blocks twenty downstream concepts outranks a leaf-node gap that blocks one.
"""

import yaml
from collections import deque
from pathlib import Path


class ConceptGraph:
    def __init__(self, packets_dir: str):
        self._graph: dict[str, list[str]] = {}     # concept_id -> advance_to list
        self._sessions: dict[str, int] = {}        # concept_id -> typical_sessions_to_master
        self._load(Path(packets_dir))

    def _load(self, packets_dir: Path) -> None:
        for p in packets_dir.glob("*.yaml"):
            with open(p) as f:
                packet = yaml.safe_load(f)
            cid = packet["concept_id"]
            self._graph[cid] = packet.get("advance_to", [])
            self._sessions[cid] = packet.get("typical_sessions_to_master", 3)

    def reachable(self, concept_id: str) -> list[dict]:
        """
        BFS from concept_id along advance_to edges.
        Returns [{concept_id, hop_distance, est_sessions_blocked}] for all descendants,
        sorted by hop_distance then concept_id for stable ordering.
        """
        if concept_id not in self._graph:
            return []

        visited = {}       # concept_id -> hop_distance
        queue = deque([(concept_id, 0)])
        while queue:
            node, depth = queue.popleft()
            for neighbor in self._graph.get(node, []):
                if neighbor not in visited:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

        return sorted(
            [
                {
                    "concept_id": cid,
                    "hop_distance": hop,
                    "est_sessions_blocked": self._sessions.get(cid, 3),
                }
                for cid, hop in visited.items()
            ],
            key=lambda x: (x["hop_distance"], x["concept_id"]),
        )

    def downstream_reach(self, concept_id: str) -> int:
        """Count of concepts transitively reachable from concept_id (fan-out)."""
        return len(self.reachable(concept_id))

    def recommend(self, trajectory_summary: dict, top_n: int = 2) -> list[dict]:
        """
        Recommend what to study next given a trajectory summary
        (from TrajectoryStore.summary()).

        Priority:
          1. In-progress concepts (started but not Constructive/Interactive)
             that are direct advance_to neighbours of mastered concepts.
          2. Unlocked concepts (never attempted) that are neighbours of mastered.
          3. Fallback when nothing is mastered: highest-reach unmastered concept
             in the entire graph (steers the learner to the best starting point).

        Returns up to top_n dicts:
          {concept_id, status, unlocked_by, downstream_reach}
        """
        mastered  = {cid for cid, s in trajectory_summary.items()
                     if s["current_level"] in {"Constructive", "Interactive"}}
        attempted = set(trajectory_summary.keys())

        candidates: dict[str, dict] = {}

        for mc in mastered:
            for neighbour in self._graph.get(mc, []):
                if neighbour in mastered or neighbour in candidates:
                    continue
                status = "in_progress" if neighbour in attempted else "unlocked"
                candidates[neighbour] = {
                    "concept_id":       neighbour,
                    "status":           status,
                    "unlocked_by":      mc,
                    "downstream_reach": self.downstream_reach(neighbour),
                }

        # Fallback: nothing mastered yet — recommend highest-reach unmastered concept
        if not candidates:
            for cid in self._graph:
                if cid not in mastered and cid not in candidates:
                    candidates[cid] = {
                        "concept_id":       cid,
                        "status":           "in_progress" if cid in attempted else "suggested",
                        "unlocked_by":      None,
                        "downstream_reach": self.downstream_reach(cid),
                    }

        order = {"in_progress": 0, "unlocked": 1, "suggested": 2}
        return sorted(
            candidates.values(),
            key=lambda x: (order.get(x["status"], 9), -x["downstream_reach"]),
        )[:top_n]

    def rank_stuck_concepts(self, stuck: list[dict]) -> list[dict]:
        """
        Takes stuck_concepts() output from TrajectoryStore, adds reach and ranks.
        Primary sort: downstream_reach descending (foundational gaps first).
        Secondary sort: interaction_count descending (longest-stuck first).
        """
        ranked = []
        for s in stuck:
            reach = self.downstream_reach(s["concept_id"])
            downstream = self.reachable(s["concept_id"])
            ranked.append({**s, "downstream_reach": reach, "downstream_concepts": downstream})

        ranked.sort(key=lambda x: (-x["downstream_reach"], -x["interaction_count"]))
        return ranked
