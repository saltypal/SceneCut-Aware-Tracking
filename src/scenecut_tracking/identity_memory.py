"""Long-term appearance memory and conservative cross-cut identity recovery."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import RecoveryDecision


@dataclass
class IdentityEntry:
    global_id: int
    gallery_size: int
    embeddings: deque[np.ndarray] = field(init=False)
    last_seen_frame: int = -1

    def __post_init__(self) -> None:
        self.embeddings = deque(maxlen=self.gallery_size)

    def add(self, embedding: np.ndarray, frame_index: int) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.size == 0 or not np.isfinite(norm) or norm <= 1e-12:
            return
        self.embeddings.append(vector / norm)
        self.last_seen_frame = frame_index

    @property
    def prototype(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        average = np.mean(np.vstack(self.embeddings), axis=0)
        norm = float(np.linalg.norm(average))
        return average / max(norm, 1e-12)


class IdentityMemory:
    """Map short-lived tracker IDs to persistent IDs, invoking ReID only after cuts."""

    def __init__(self, similarity_threshold: float, recovery_window_frames: int, gallery_size: int):
        self.similarity_threshold = float(similarity_threshold)
        self.recovery_window_frames = int(recovery_window_frames)
        self.gallery_size = int(gallery_size)
        self.entries: dict[int, IdentityEntry] = {}
        self.local_to_global: dict[int, int] = {}
        self.next_global_id = 1
        self.recovery_candidates: set[int] = set()
        self.claimed_candidates: set[int] = set()
        self.recovery_until_frame = -1

    def begin_cut(self, frame_index: int) -> None:
        self.recovery_candidates = {
            global_id for global_id, entry in self.entries.items() if entry.prototype is not None
        }
        self.claimed_candidates.clear()
        self.local_to_global.clear()
        self.recovery_until_frame = frame_index + self.recovery_window_frames - 1

    def _allocate(self) -> int:
        global_id = self.next_global_id
        self.next_global_id += 1
        self.entries[global_id] = IdentityEntry(global_id, self.gallery_size)
        return global_id

    def assign(
        self,
        frame_index: int,
        local_ids: list[int],
        embeddings: np.ndarray,
    ) -> tuple[list[int], list[RecoveryDecision]]:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(local_ids) != len(embeddings):
            raise ValueError("Each local track must have exactly one embedding row")

        assigned: list[int | None] = [None] * len(local_ids)
        decisions: list[RecoveryDecision] = []
        new_indices: list[int] = []

        for index, local_id in enumerate(local_ids):
            if local_id in self.local_to_global:
                assigned[index] = self.local_to_global[local_id]
            else:
                new_indices.append(index)

        recovery_active = frame_index <= self.recovery_until_frame
        available_ids = sorted(self.recovery_candidates - self.claimed_candidates)
        valid_candidates = [
            global_id for global_id in available_ids if self.entries[global_id].prototype is not None
        ]

        if recovery_active and new_indices and valid_candidates:
            candidate_matrix = np.vstack([self.entries[global_id].prototype for global_id in valid_candidates])
            query_matrix = embeddings[new_indices]
            query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
            query_matrix = query_matrix / np.maximum(query_norms, 1e-12)
            similarities = np.nan_to_num(
                query_matrix @ candidate_matrix.T, nan=-1.0, posinf=-1.0, neginf=-1.0
            )
            query_rows, candidate_columns = linear_sum_assignment(-similarities)
            for query_row, candidate_column in zip(query_rows, candidate_columns):
                item_index = new_indices[int(query_row)]
                similarity = float(similarities[query_row, candidate_column])
                candidate_id = valid_candidates[int(candidate_column)]
                if np.isfinite(similarity) and similarity >= self.similarity_threshold:
                    assigned[item_index] = candidate_id
                    self.claimed_candidates.add(candidate_id)
                    decisions.append(
                        RecoveryDecision(
                            frame=frame_index,
                            local_id=local_ids[item_index],
                            global_id=candidate_id,
                            decision="recovered",
                            similarity=similarity,
                        )
                    )

        for index in new_indices:
            local_id = local_ids[index]
            if assigned[index] is None:
                global_id = self._allocate()
                assigned[index] = global_id
                decisions.append(
                    RecoveryDecision(
                        frame=frame_index,
                        local_id=local_id,
                        global_id=global_id,
                        decision="new_identity",
                        similarity=None,
                    )
                )
            self.local_to_global[local_id] = int(assigned[index])

        final_ids = [int(global_id) for global_id in assigned]
        for index, global_id in enumerate(final_ids):
            self.entries[global_id].add(embeddings[index], frame_index)
        return final_ids, decisions
