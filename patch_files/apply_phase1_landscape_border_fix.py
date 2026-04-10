from pathlib import Path

ROOT = Path.cwd()

# 1) Flip x-large template to landscape
yaml_path = ROOT / 'configs/templates/cds_xlarge_500x900.yaml'
text = yaml_path.read_text(encoding='utf-8')
old = '  width: 500\n  height: 900'
new = '  width: 900\n  height: 500'
if old in text:
    text = text.replace(old, new)
yaml_path.write_text(text, encoding='utf-8')

boundary_code = '''"""Phase 1 border detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from armourcore_cds.utils.image_ops import to_gray


@dataclass
class BorderDetectionResult:
    contour: np.ndarray
    ordered_corners: np.ndarray
    preview_edges: np.ndarray
    preview_mask: np.ndarray
    contour_area_px: float
    score: float
    candidate_count: int


def _order_points_clockwise(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _edge_dark_ratio(gray: np.ndarray, rect: np.ndarray) -> float:
    mask = np.zeros_like(gray)
    thickness = max(3, int(round(min(gray.shape[:2]) * 0.0045)))
    cv2.polylines(mask, [rect.astype(np.int32)], isClosed=True, color=255, thickness=thickness)
    pixels = gray[mask > 0]
    if pixels.size == 0:
        return 0.0
    dark_threshold = float(np.percentile(gray, 35))
    return float(np.mean(pixels <= dark_threshold))


def _contour_score(
    contour: np.ndarray,
    image_shape: tuple[int, int],
    expected_aspect_ratio: float,
    gray: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None

    area = cv2.contourArea(approx)
    if area <= 0:
        return None

    h, w = image_shape[:2]
    image_area = float(h * w)
    area_ratio = area / image_area
    if area_ratio < 0.08:
        return None

    rect = _order_points_clockwise(approx.reshape(4, 2))
    top_w = float(np.linalg.norm(rect[1] - rect[0]))
    bottom_w = float(np.linalg.norm(rect[2] - rect[3]))
    left_h = float(np.linalg.norm(rect[3] - rect[0]))
    right_h = float(np.linalg.norm(rect[2] - rect[1]))
    mean_w = max((top_w + bottom_w) / 2.0, 1.0)
    mean_h = max((left_h + right_h) / 2.0, 1.0)
    aspect_ratio = mean_w / mean_h
    aspect_error = abs(np.log(aspect_ratio / expected_aspect_ratio))

    dark_ratio = _edge_dark_ratio(gray, rect)

    contour_pts = approx.reshape(4, 2).astype(np.float32)
    contour_center = contour_pts.mean(axis=0)
    image_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    centre_distance = float(np.linalg.norm(contour_center - image_center))
    max_centre_distance = float(np.linalg.norm(image_center))
    centre_penalty = centre_distance / max(max_centre_distance, 1.0)

    min_rect = cv2.minAreaRect(contour)
    box_area = max(cv2.contourArea(cv2.boxPoints(min_rect).astype(np.float32)), 1.0)
    extent = area / box_area

    x, y, bw, bh = cv2.boundingRect(approx)
    margin = max(6, int(round(min(h, w) * 0.01)))
    touches_image_edge = x <= margin or y <= margin or (x + bw) >= (w - margin) or (y + bh) >= (h - margin)
    edge_touch_penalty = 0.25 if touches_image_edge else 0.0

    score = (
        (area_ratio * 3.0)
        + (extent * 1.0)
        + (dark_ratio * 2.5)
        - (aspect_error * 3.5)
        - (centre_penalty * 0.4)
        - edge_touch_penalty
    )
    return score, rect


def detect_outer_border(image: np.ndarray, expected_aspect_ratio: float) -> BorderDetectionResult:
    gray = to_gray(image)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_score = None
    best_rect = None
    best_contour = None

    for contour in contours:
        scored = _contour_score(contour, image.shape, expected_aspect_ratio, gray)
        if scored is None:
            continue
        score, rect = scored
        if best_score is None or score > best_score:
            best_score = score
            best_rect = rect
            best_contour = contour

    if best_rect is None or best_contour is None or best_score is None:
        raise RuntimeError('Could not detect CDS outer border candidate.')

    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, best_rect.astype(np.int32), 255)

    return BorderDetectionResult(
        contour=best_contour,
        ordered_corners=best_rect,
        preview_edges=dilated,
        preview_mask=mask,
        contour_area_px=float(cv2.contourArea(best_contour)),
        score=float(best_score),
        candidate_count=len(contours),
    )
'''

crop_code = '''"""Phase 1 inner-border crop recovery."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from armourcore_cds.utils.image_ops import to_gray


@dataclass
class InnerBorderCropResult:
    cropped_image: np.ndarray
    inner_rect_xyxy: tuple[int, int, int, int]
    border_samples_px: dict[str, int]
    preview_binary: np.ndarray


def _find_peak_run(profile: np.ndarray, prefer_start: bool) -> tuple[int, int]:
    if profile.size == 0:
        return 0, 0

    threshold = max(float(np.percentile(profile, 92)) * 0.55, 0.10)
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0

    for idx, value in enumerate(profile):
        if value >= threshold:
            if not in_run:
                start = idx
                in_run = True
        else:
            if in_run:
                runs.append((start, idx - 1))
                in_run = False

    if in_run:
        runs.append((start, len(profile) - 1))

    if not runs:
        peak = int(np.argmax(profile))
        return peak, peak

    best_run = runs[0]
    best_score = float('-inf')
    length = max(len(profile), 1)
    for run_start, run_end in runs:
        run_slice = profile[run_start:run_end + 1]
        run_strength = float(run_slice.mean())
        run_len = run_end - run_start + 1
        edge_distance = run_start if prefer_start else (length - 1 - run_end)
        score = (run_strength * run_len) - ((edge_distance / length) * 0.5)
        if score > best_score:
            best_score = score
            best_run = (run_start, run_end)
    return best_run


def find_inner_border_crop(rectified_image: np.ndarray) -> InnerBorderCropResult:
    gray = to_gray(rectified_image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        9,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    h, w = rectified_image.shape[:2]
    search_y = max(20, int(round(h * 0.18)))
    search_x = max(20, int(round(w * 0.12)))

    centre_x0 = int(round(w * 0.12))
    centre_x1 = int(round(w * 0.88))
    centre_y0 = int(round(h * 0.12))
    centre_y1 = int(round(h * 0.88))

    top_strength = (binary[:search_y, centre_x0:centre_x1] > 0).mean(axis=1)
    bottom_strength = (binary[h - search_y:, centre_x0:centre_x1] > 0).mean(axis=1)
    left_strength = (binary[centre_y0:centre_y1, :search_x] > 0).mean(axis=0)
    right_strength = (binary[centre_y0:centre_y1, w - search_x:] > 0).mean(axis=0)

    _, top_end = _find_peak_run(top_strength, prefer_start=True)
    bottom_start, _ = _find_peak_run(bottom_strength, prefer_start=False)
    _, left_end = _find_peak_run(left_strength, prefer_start=True)
    right_start, _ = _find_peak_run(right_strength, prefer_start=False)

    y0 = int(np.clip(top_end + 1, 0, h - 2))
    y1 = int(np.clip((h - search_y) + bottom_start - 1, y0 + 1, h - 1))
    x0 = int(np.clip(left_end + 1, 0, w - 2))
    x1 = int(np.clip((w - search_x) + right_start - 1, x0 + 1, w - 1))

    if (x1 - x0) < (w * 0.50) or (y1 - y0) < (h * 0.50):
        fallback_pad_x = max(2, int(round(w * 0.01)))
        fallback_pad_y = max(2, int(round(h * 0.01)))
        x0 = fallback_pad_x
        x1 = w - fallback_pad_x
        y0 = fallback_pad_y
        y1 = h - fallback_pad_y

    cropped = rectified_image[y0:y1, x0:x1].copy()
    return InnerBorderCropResult(
        cropped_image=cropped,
        inner_rect_xyxy=(x0, y0, x1, y1),
        border_samples_px={
            'top': int(y0),
            'bottom': int(h - y1),
            'left': int(x0),
            'right': int(w - x1),
        },
        preview_binary=binary,
    )
'''

(ROOT / 'src/armourcore_cds/phase1/boundary_detection.py').write_text(boundary_code, encoding='utf-8')
(ROOT / 'src/armourcore_cds/phase1/crop.py').write_text(crop_code, encoding='utf-8')

print('Patched 3 files:')
print(' - configs/templates/cds_xlarge_500x900.yaml')
print(' - src/armourcore_cds/phase1/boundary_detection.py')
print(' - src/armourcore_cds/phase1/crop.py')
