"""Phase 2 module: note_candidates"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class NoteCandidate:
    bbox_xywh: tuple[int, int, int, int]
    area_px: int


def detect_note_candidate_mask(
    seed_mask: np.ndarray,
    max_width_fraction: float = 0.18,
    max_height_fraction: float = 0.12,
    min_area_px: int = 10,
) -> tuple[np.ndarray, list[NoteCandidate]]:
    h, w = seed_mask.shape[:2]
    max_w = int(round(w * max_width_fraction))
    max_h = int(round(h * max_height_fraction))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)

    note_mask = np.zeros_like(seed_mask)
    candidates: list[NoteCandidate] = []

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])

        if area < min_area_px:
            continue
        if bw > max_w or bh > max_h:
            continue

        note_mask[labels == idx] = 255
        candidates.append(NoteCandidate(bbox_xywh=(x, y, bw, bh), area_px=area))

    return note_mask, candidates
