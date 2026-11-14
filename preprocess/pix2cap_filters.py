"""Caption-validity and crop-visibility helpers used by the training-set filter."""

import re
from typing import Iterable, Optional, Sequence, Tuple

_REFUSAL_PATTERNS: Sequence[Tuple[str, re.Pattern]] = (
    ("apology", re.compile(r"^(?:i(?:'m| am)\s+)?sorry\b", re.IGNORECASE)),
    (
        "cannot_describe",
        re.compile(
            r"\b(?:cannot|can't|could not|couldn't|unable to)\s+"
            r"(?:find|identify|see|locate|detect|describe|generate)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "object_not_visible",
        re.compile(
            r"\b(?:there (?:is|are) no|no visible|not visible|does(?:n't| not) "
            r"seem to be|do(?:es)? not appear)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_leak",
        re.compile(
            r"\b(?:bounding box|query object|queried object|specified object|"
            r"highlighted object|current frame|provide (?:a )?(?:different|another) "
            r"(?:image|query))\b",
            re.IGNORECASE,
        ),
    ),
)


def clean_caption(value: object) -> str:
    """Normalize whitespace while preserving the released sentence."""
    return " ".join(str(value or "").strip().split())


def invalid_caption_reason(caption: str) -> Optional[str]:
    if not caption:
        return "empty_caption"
    for name, pattern in _REFUSAL_PATTERNS:
        if pattern.search(caption):
            return name
    return None


def crop_window(width: int, height: int, crop_size: int) -> Tuple[float, float, float, float, float]:
    scale = max(float(crop_size) / float(height), float(crop_size) / float(width))
    resized_w = float(round(width * scale))
    resized_h = float(round(height * scale))
    left = max((resized_w - crop_size) // 2, 0.0)
    top = max((resized_h - crop_size) // 2, 0.0)
    return scale, left, top, left + crop_size, top + crop_size


def bbox_crop_intersection(
    bbox: Iterable[float], width: int, height: int, crop_size: int
) -> Tuple[bool, float, float, float]:
    """Return visibility, approximate visible area, crop-x, and crop-y."""
    x, y, w, h = (float(v) for v in bbox)
    if w <= 0 or h <= 0:
        return False, 0.0, 0.0, 0.0
    scale, left, top, right, bottom = crop_window(width, height, crop_size)
    x0, y0 = x * scale, y * scale
    x1, y1 = (x + w) * scale, (y + h) * scale
    ix0, iy0 = max(x0, left), max(y0, top)
    ix1, iy1 = min(x1, right), min(y1, bottom)
    iw, ih = max(ix1 - ix0, 0.0), max(iy1 - iy0, 0.0)
    return iw > 0 and ih > 0, iw * ih, ix0 - left, iy0 - top
