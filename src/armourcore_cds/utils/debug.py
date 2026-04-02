from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from armourcore_cds.utils.image_ops import ensure_uint8_bgr, save_image


class DebugWriter:
    def __init__(self, debug_dir: Path, enabled: bool = True) -> None:
        self.debug_dir = debug_dir
        self.enabled = enabled
        if self.enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def image(self, name: str, image: np.ndarray) -> Path | None:
        if not self.enabled:
            return None
        path = self.debug_dir / f'{name}.png'
        save_image(path, image)
        return path

    def text_overlay(self, image: np.ndarray, lines: list[str]) -> np.ndarray:
        canvas = ensure_uint8_bgr(image).copy()
        y = 30
        for line in lines:
            cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            y += 30
        return canvas
