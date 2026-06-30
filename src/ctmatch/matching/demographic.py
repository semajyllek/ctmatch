"""
Pure-function helpers for demographic pre-filtering (age extraction and exclusion).

Imported by pipeline.py and tested directly — no model loading, no heavy deps.
"""

import re
from typing import Optional


_AGE_NUM_YR    = r'(\d+(?:\.\d+)?)\s*(years?|yrs?)'
_AGE_NUM_YR_MO = r'(\d+(?:\.\d+)?)\s*(years?|yrs?|months?)'


def _parse_age_years(n: str, unit: str) -> float:
    v = float(n)
    u = unit.lower()
    if 'month' in u: return v / 12.0
    if 'week'  in u: return v / 52.0
    if 'day'   in u: return v / 365.0
    return v


def extract_patient_age(topic_text: str) -> Optional[float]:
    """Extract patient age in fractional years from a TREC clinical vignette.
    Anchors to the first age mention — vignettes always open with the patient."""
    t = topic_text.strip()
    for pat, divisor in [
        (r'\b(\d+)[\s-]*day[\s-]*old',  365.0),
        (r'\b(\d+)[\s-]*week[\s-]*old',  52.0),
        (r'\b(\d+)[\s-]*month[\s-]*old', 12.0),
    ]:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            return int(m.group(1)) / divisor
    m = re.search(r'\b(\d+)\s*[-–]?\s*(?:year[\s-]*old|yo\b|y\.o\.?)', t, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def mask_nonchrono_age(text: str) -> str:
    """Mask gestational/postnatal/corrected age references so age regexes don't false-fire."""
    text = re.sub(
        r'\b\d+(?:\s*[-–/]\s*\d+(?:\s*\d+/\d+)?)?\s*(?:weeks?|days?)\s+'
        r'(?:gestational|postnatal|post-natal|corrected|conceptional)\s+age\b',
        'NONCHRONO', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b(?:gestational|postnatal|post-natal|corrected|conceptional)\s+age\b'
        r'(?:\s*(?:of\s+)?\s*(?:>=|≥|<=|≤|>|<|=))?\s*\d*\s*(?:weeks?|days?)?',
        'NONCHRONO', text, flags=re.IGNORECASE)
    return text


def trial_age_range(doc_text: str) -> tuple:
    """Return (min_age_years, max_age_years) from trial eligibility text.

    Conservative: requires 'years' for ambiguous patterns; returns (None, None)
    when min > max (parsing contradiction)."""
    doc = mask_nonchrono_age(doc_text)

    exc_m = re.search(r'\bExclusion\s+Criteria\b', doc, re.IGNORECASE)
    inc_text = doc[:exc_m.start()] if exc_m else doc
    exc_text = doc[exc_m.start():] if exc_m else ''

    min_age: Optional[float] = None
    max_age: Optional[float] = None

    # ── Inclusion: minimum age ──────────────────────────────────────────────

    m = re.search(r'\bage\s*(?:>=|≥)\s*' + _AGE_NUM_YR_MO, inc_text, re.IGNORECASE)
    if m:
        min_age = _parse_age_years(m.group(1), m.group(2))

    if min_age is None:
        m = re.search(r'\bage\s*>(?!=)\s*' + _AGE_NUM_YR_MO, inc_text, re.IGNORECASE)
        if m:
            nearby = inc_text[m.start(): min(m.end() + 60, len(inc_text))]
            if not re.search(r'or\s*<', nearby, re.IGNORECASE):
                min_age = _parse_age_years(m.group(1), m.group(2))

    if min_age is None:
        m = re.search(r'at\s+least\s+' + _AGE_NUM_YR, inc_text, re.IGNORECASE)
        if m:
            min_age = float(m.group(1))

    if min_age is None:
        m = re.search(_AGE_NUM_YR + r'\s+(?:of\s+age\s+)?or\s+older', inc_text, re.IGNORECASE)
        if m:
            min_age = float(m.group(1))

    if min_age is None:
        m = re.search(r'\bage[:\s]+' + _AGE_NUM_YR + r'\s*(?:\+|and\s+older)', inc_text, re.IGNORECASE)
        if m:
            min_age = float(m.group(1))

    if min_age is None:
        m = re.search(r'\bage[:\s]+(\d+)\s*\+', inc_text, re.IGNORECASE)
        if m:
            min_age = float(m.group(1))

    # ── Inclusion: maximum age ──────────────────────────────────────────────

    m = re.search(r'\bage\s*(?:<=|≤)\s*' + _AGE_NUM_YR, inc_text, re.IGNORECASE)
    if m:
        max_age = float(m.group(1))

    if max_age is None:
        m = re.search(r'\bage\s*<(?!=)\s*' + _AGE_NUM_YR, inc_text, re.IGNORECASE)
        if m:
            max_age = float(m.group(1))

    # ── Exclusion: under-N → minimum age ───────────────────────────────────

    if exc_text:
        m = re.search(r'\bage\s*<(?!=)\s*' + _AGE_NUM_YR, exc_text, re.IGNORECASE)
        if m:
            c = float(m.group(1))
            min_age = max(min_age, c) if min_age is not None else c

        m = re.search(r'(?:less\s+than|under|younger\s+than)\s+' + _AGE_NUM_YR,
                      exc_text, re.IGNORECASE)
        if m:
            c = float(m.group(1))
            min_age = max(min_age, c) if min_age is not None else c

        m = re.search(r'\bage\s*>(?!=)\s*' + _AGE_NUM_YR, exc_text, re.IGNORECASE)
        if m:
            c = float(m.group(1))
            max_age = min(max_age, c) if max_age is not None else c

        m = re.search(r'(?:older\s+than|over)\s+' + _AGE_NUM_YR, exc_text, re.IGNORECASE)
        if m:
            c = float(m.group(1))
            max_age = min(max_age, c) if max_age is not None else c

    if min_age is not None and max_age is not None and min_age > max_age:
        return None, None

    return min_age, max_age


def trial_excludes_age(doc_text: str, patient_age: float) -> bool:
    """Return True only when confident the patient age is outside the trial range.
    0.5-year buffer absorbs rounding from months/weeks conversions."""
    min_age, max_age = trial_age_range(doc_text)
    if min_age is not None and patient_age < min_age - 0.5:
        return True
    if max_age is not None and patient_age > max_age + 0.5:
        return True
    return False
