from __future__ import annotations


def mm_to_pixels(mm: float, dpi: int) -> float:
    return (mm / 25.4) * dpi
