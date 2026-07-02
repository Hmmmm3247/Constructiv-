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
