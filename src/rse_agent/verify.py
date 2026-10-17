"""Answer verification: every number in a final answer must come from the data.

A number counts as verified if it appears (allowing rounding) in a successful tool result, in the
user's question, or in the data dictionary (band labels such as '1 cr - 5 cr'). Anything else
is treated as possibly invented: the agent gets one chance to re-query, after which the answer
carries a visible warning.
"""
from __future__ import annotations

import json
import re

_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.M)


def numbers(text: str) -> list[float]:
    text = _LIST_MARKER.sub(" ", text)                 # '1. ' list numbering is not data
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def successful(content: str) -> bool:
    """A tool result counts only if it did not fail."""
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return bool(content)
    return not (isinstance(obj, dict) and "error" in obj)


def _matches(n: float, sources: set[float]) -> bool:
    for s in sources:
        if abs(n - s) <= 0.005 * max(1.0, abs(s)):
            return True
        if any(round(s, k) == n for k in (0, 1, 2)):
            return True
    return False


def unverified(answer: str, source_texts: list[str]) -> list[float]:
    sources = {v for t in source_texts for v in numbers(t)}
    seen, bad = set(), []
    for n in numbers(answer):
        if n in seen:
            continue
        seen.add(n)
        if not _matches(n, sources):
            bad.append(n)
    return bad


def fmt(nums: list[float]) -> str:
    return ", ".join(f"{n:g}" for n in nums[:8])
