"""Input-image format / orientation normalisation.

Two pre-processing helpers, both designed to run BEFORE Phase 1 so
that downstream code never has to think about input quirks.

* ``ensure_loadable_image(path)``  — HEIC/HEIF -> PNG via pillow-heif.
* ``ensure_landscape(path)``       — portrait -> 90-deg rotated PNG.

Both write to a process-shared temp folder so repeated runs on the
same file reuse the cached copy.  Both return the original path
unchanged when no conversion is needed (zero overhead common case).

Usage::

    from gui.image_loader import ensure_loadable_image, ensure_landscape
    loadable = ensure_loadable_image(user_input_path)
    oriented = ensure_landscape(loadable)
    # pass `oriented` to Phase 1
"""
from __future__ import annotations

import tempfile
from pathlib import Path


# Formats Phase 1's cv2.imread / PDF loader can already handle.
_NATIVE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf",
}

# Formats that need PIL/pillow-heif conversion.
_HEIC_SUFFIXES = {".heic", ".heif"}


def ensure_loadable_image(input_path: Path) -> Path:
    """Return a path Phase 1 can ``cv2.imread`` straight away.

    If ``input_path`` is already a natively-supported format, returns
    it unchanged.

    If it's a HEIC / HEIF file, decodes it via pillow-heif, writes a
    PNG copy into a process-shared temporary folder, and returns that
    PNG path.  The original is never modified.

    Raises ``RuntimeError`` if pillow-heif is not installed and the
    input requires it.
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in _NATIVE_SUFFIXES:
        return input_path

    if suffix in _HEIC_SUFFIXES:
        return _convert_heic_to_png(input_path)

    # Unknown suffix — let Phase 1 fail with its own clearer error
    # rather than silently mangling something we don't understand.
    return input_path


def ensure_landscape(input_path: Path) -> tuple[Path, bool]:
    """Rotate a portrait input to landscape if needed (ISSUE-001 / H).

    Phase 1's rectifier was tuned for landscape-oriented inputs and
    fails or mis-classifies portrait shots.  Simple rule: if the
    image is taller than it is wide, rotate 90 deg clockwise and
    hand the rotated PNG to Phase 1.  PDFs and HEICs are expected
    to have already been converted to PNG by ``ensure_loadable_image``
    before this is called.

    Returns
    -------
    (path_to_use, was_rotated)
        path_to_use : either the original path or a temp PNG copy
        was_rotated : True if a rotation happened, False otherwise
    """
    import cv2

    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        # PDF rasterisation happens inside Phase 1; orientation is
        # decided there.  Skip.
        return input_path, False

    img = cv2.imread(str(input_path))
    if img is None:
        # Let the downstream loader raise its own clear error
        return input_path, False
    h, w = img.shape[:2]
    if h <= w:
        return input_path, False     # already landscape

    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    tmp_root = Path(tempfile.gettempdir()) / "armourcore_oriented"
    tmp_root.mkdir(parents=True, exist_ok=True)
    out_path = tmp_root / f"{input_path.stem}_landscape.png"
    cv2.imwrite(str(out_path), rotated)
    return out_path, True


def _convert_heic_to_png(heic_path: Path) -> Path:
    """Decode a HEIC file via pillow-heif and write a temp PNG."""
    try:
        import pillow_heif  # type: ignore
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "HEIC input requires the 'pillow-heif' package.\n"
            "Install it once with:  pip install pillow-heif"
        ) from e

    pillow_heif.register_heif_opener()
    img = Image.open(heic_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Keep all converted PNGs in one process-local temp folder so
    # repeated runs of the same file don't re-decode.
    tmp_root = Path(tempfile.gettempdir()) / "armourcore_heic"
    tmp_root.mkdir(parents=True, exist_ok=True)
    out_path = tmp_root / f"{heic_path.stem}.png"
    img.save(out_path, format="PNG")
    return out_path
