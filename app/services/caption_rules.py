"""Conservative caption rules for low-confidence visual classifications."""

from __future__ import annotations

import re

LOW_CONFIDENCE_THRESHOLD = 0.75
RULES = {
    "tem": (r"\btem\b", r"\bhrtem\b", r"\bhaadf\b", r"\bstem\b", r"transmission electron microscopy", r"\bsaed\b"),
    "sem": (r"\bsem\b", r"scanning electron microscopy", r"scanning electron micrograph"),
    "xrd": (r"\bxrd\b", r"x-ray diffraction", r"x ray diffraction", r"\b2θ\b", r"\b2theta\b"),
    "bar_chart": (r"bar chart", r"bar graph", r"histogram"),
    "line_chart": (r"line chart", r"line plot", r"line graph", r"\bcurves?\b"),
}


def caption_override(caption: str | None, visual_confidence: float) -> str | None:
    """Return a single unambiguous caption label for a low-confidence image."""
    if not caption or visual_confidence >= LOW_CONFIDENCE_THRESHOLD:
        return None
    normalized = caption.casefold()
    matches = {
        label for label, patterns in RULES.items()
        if any(re.search(pattern, normalized) for pattern in patterns)
    }
    return matches.pop() if len(matches) == 1 else None
