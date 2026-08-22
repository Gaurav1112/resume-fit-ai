"""Date parsing and normalisation.

Two consumers with different needs, and the split matters:

* The **writer** normalises every date to one display format ("Mar 2021"), which
  is what makes the ATS date-consistency check pass by construction.
* The **truth validator** compares dates *semantically* — (year, month) — not as
  strings. Reformatting "03/2021" to "Mar 2021" changes the presentation, not the
  fact, so it must not be flagged as a fabrication. Changing 2021 to 2019 must be.
"""

from __future__ import annotations

import re
from datetime import date

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

PRESENT = {"present", "current", "now", "ongoing", "till date", "to date", "date"}

_MON_YEAR = re.compile(r"^([A-Za-z]{3,9})\.?\s*[,']?\s*(\d{4})$")
_YEAR_MON = re.compile(r"^(\d{4})[-/](\d{1,2})$")
_MON_NUM_YEAR = re.compile(r"^(\d{1,2})[-/](\d{4})$")
_MON_DAY_YEAR = re.compile(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})$")
_YEAR_ONLY = re.compile(r"^(\d{4})$")

# A date *range*, used to find role boundaries when parsing a resume.
RANGE = re.compile(
    r"(?P<start>(?:[A-Za-z]{3,9}\.?\s*,?\s*\d{4})|(?:\d{1,2}[-/]\d{4})|(?:\d{4}[-/]\d{1,2})|(?:\d{4}))"
    r"\s*(?:-|–|—|to|until|through)\s*"
    r"(?P<end>(?:[A-Za-z]{3,9}\.?\s*,?\s*\d{4})|(?:\d{1,2}[-/]\d{4})|(?:\d{4}[-/]\d{1,2})|(?:\d{4})"
    r"|(?:[Pp]resent|[Cc]urrent|[Nn]ow|[Oo]ngoing|PRESENT|CURRENT))",
    re.I,
)


def parse(value: str) -> tuple[int, int] | None:
    """Return (year, month) — month 0 when only a year is known. None if unparseable.

    `(9999, 12)` is the sentinel for "Present", so ordering works naturally.
    """
    v = (value or "").strip().strip(",.")
    if not v:
        return None
    if v.lower() in PRESENT:
        return (9999, 12)

    m = _MON_YEAR.match(v)
    if m and m.group(1).lower() in MONTHS:
        return (int(m.group(2)), MONTHS[m.group(1).lower()])

    m = _YEAR_MON.match(v)
    if m:
        month = int(m.group(2))
        if 1 <= month <= 12:
            return (int(m.group(1)), month)

    m = _MON_NUM_YEAR.match(v)
    if m:
        month = int(m.group(1))
        if 1 <= month <= 12:
            return (int(m.group(2)), month)

    m = _MON_DAY_YEAR.match(v)
    if m:
        month, year = int(m.group(1)), int(m.group(3))
        if year < 100:
            year += 2000 if year < 50 else 1900
        if 1 <= month <= 12:
            return (year, month)

    m = _YEAR_ONLY.match(v)
    if m:
        year = int(m.group(1))
        if 1900 <= year <= 2100:
            return (year, 0)
    return None


def normalise(value: str) -> str:
    """Render a date in the single format the generated resume uses.

    Unparseable input is returned untouched — silently dropping a date the parser
    doesn't understand would be worse than an inconsistent one.
    """
    parsed = parse(value)
    if parsed is None:
        return (value or "").strip()
    year, month = parsed
    if year == 9999:
        return "Present"
    if month == 0:
        return str(year)
    return f"{MONTH_NAMES[month]} {year}"


def same(a: str, b: str) -> bool:
    """Semantic equality — the comparison the truth gate uses."""
    pa, pb = parse(a), parse(b)
    if pa is None or pb is None:
        return (a or "").strip().lower() == (b or "").strip().lower()
    if pa[1] == 0 or pb[1] == 0:      # one side is year-only; compare years
        return pa[0] == pb[0]
    return pa == pb


def months_between(start: str, end: str, *, today: date | None = None) -> int:
    """Duration in months. 'Present' resolves against today."""
    s, e = parse(start), parse(end)
    if s is None:
        return 0
    now = today or date.today()
    sy, sm = s[0], s[1] or 1
    if e is None:
        ey, em = now.year, now.month
    elif e[0] == 9999:
        ey, em = now.year, now.month
    else:
        ey, em = e[0], e[1] or 12
    return max(0, (ey - sy) * 12 + (em - sm) + 1)


def total_experience_years(
    ranges: list[tuple[str, str]], *, today: date | None = None
) -> float | None:
    """Total professional experience, merging overlapping roles.

    Summing role durations double-counts concurrent or overlapping employment and
    inflates the figure — which the truth gate would then reject. Merging the
    intervals gives the number a resume can actually defend.
    """
    intervals: list[tuple[int, int]] = []
    now = today or date.today()
    now_index = now.year * 12 + now.month

    for start, end in ranges:
        s = parse(start)
        if s is None:
            continue
        e = parse(end)
        s_index = s[0] * 12 + (s[1] or 1)
        if e is None or e[0] == 9999:
            e_index = now_index
        else:
            e_index = e[0] * 12 + (e[1] or 12)
        if e_index >= s_index:
            intervals.append((s_index, e_index + 1))   # inclusive of the end month

    if not intervals:
        return None

    intervals.sort()
    merged: list[list[int]] = [list(intervals[0])]
    for start_i, end_i in intervals[1:]:
        if start_i <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end_i)
        else:
            merged.append([start_i, end_i])

    months = sum(end_i - start_i for start_i, end_i in merged)
    return round(months / 12.0, 1) if months else None
