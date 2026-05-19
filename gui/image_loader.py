"""Input-image format normalisation.

The downstream Phase 1 pipeline calls ``cv2.imread`` which doesn't
understand HEIC / HEIF (Apple's iOS-default photo format).  This
module detects HEIC inputs and pre-converts them to PNG in a
temporary folder so Phase 1 sees an image it can load.

For all other formats it returns the original path unchanged, so
the conversion has zero overhead on the common case.

Usage::

    from gui.image_loader import ensure_loadable_image
    path_for_phase1 = ensure_loadable_image(user_input_path)
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
