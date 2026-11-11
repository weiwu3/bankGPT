"""Turn the structured parts of a page into rows: one row = one published value with the
qualifiers that identify it. Four shapes are read:

    table rows            | Share Savings | $3,000 - $11,999.99 | 0.27% | 0.27% | ...
    labelled values       Routing Number: 123456789          6-month certificate APY: 2.95%
    fee-list lines        Paper statement (mailed) $2.00 per month
    rates in a sentence   Summer special: 6-month certificate APY 2.95% (weak evidence)

A row keeps where it came from (page, table label, section heading) and the "effective as
of <date>" line printed under its table, the best evidence of which copy is current.
Choosing between rows that disagree is not done here (see lookup.RowIndex).
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from .pages import DOCUMENT, for_business
from .text import (DAY, HEADING, INF, Quals, clean, content_tokens, effective_date, find_dates, fix_footnotes,
                   money_of, parse_amount_range, parse_term, parse_years, percent_of, strip_markdown)

# --------------------------------------------------------------------------- records


@dataclass
class Value:
    """One published value of a row, as printed and as a number."""
    metric: str            # apy, dividend, apr, rate, points, payment, min_open, min_earn, fee, value
    header: str            # the column header as printed
    text: str              # the value as printed, footnote marks removed
    num: Optional[float]
    unit: str              # '%', '$' or ''


@dataclass(eq=False)
class Row:
    """One published fact: its values, the qualifiers that tell it from its neighbours, and where
    it was printed. Compared by identity."""
    page: object                       # PageVersion
    table: int                         # index of the table on the page (-1 for labelled values)
    label: str                         # table label: 'Promo Certificates'
    section: str                       # headings above the table
    name: str                          # the row's own words: '19 Month Bump-Up Certificate Special'
    quals: Quals
    values: list[Value]
    effective: Optional[date] = None
    order: int = 0
    kind: str = "table"                # table | labelled | fee | claim
    expired: Optional[date] = None     # a promotion the page says has ended: on the row's line or above it
    # Where `effective` came from: "page" (printed on the page), "history" (printed on another
    # download of the same table) or "change" (the download before the table changed: a lower
    # bound). See Store.
    dated_by: str = "page"
    maintained: bool = False           # this page has changed this row over the downloads (see Store)
    name_tokens: frozenset = frozenset()
    label_tokens: frozenset = frozenset()
    section_tokens: frozenset = frozenset()
    variant: int = 0                   # 1, 2 ...: an earlier row of this page reads the same but says another value

    @property
    def business(self) -> bool:
        """A business product: 'Business Share Savings' is not the same fact as 'Share Savings'."""
        return "business" in self.name_tokens or "business" in self.label_tokens or "business" in self.section_tokens \
            or for_business(self.page.key)

    @property
    def slot(self) -> tuple:
        """Two rows with the same slot state the same fact (on different pages or dates)."""
        return (self.name_tokens, self.label_tokens if not self.name_tokens else frozenset(), self.quals, self.business,
                self.variant)

    def primary(self) -> Optional[Value]:
        """The value the row is about: its APY, else APR, rate, dividend, fee or plain value; else the first."""
        for metric in ("apy", "apr", "rate", "dividend", "fee", "value"):
            for v in self.values:
                if v.metric == metric:
                    return v
        return self.values[0] if self.values else None

    def signature(self) -> tuple:
        """The published values, compared as numbers so '$25' and '$25.00' are the same."""
        return tuple((v.metric, round(v.num, 4) if v.num is not None else v.text.lower()) for v in self.values)


# --------------------------------------------------------------------------- cell reading

_EMPTY = {"", "-", "--", "n/a", "na", "none", "free", "varies"}
_SINGLE_VALUE = re.compile(
    r"^(?:as low as\s*)?(?:\$\s?[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?\s?%(?:\s?-\s?\d+(?:\.\d+)?\s?%)?|\d+(?:\.\d+)?)"
    r"\*?\s*(?:apr|apy|intro apr)?\*?$", re.I)


_LEADING_VALUE = re.compile(r"^(?:as low as\s*)?(?:\$\s?[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?\s?%)\*?\s+[A-Za-z]", re.I)


def _cell(s: str) -> str:
    # Tables read from PDFs arrive HTML-escaped, some cells twice: "Non&amp;#45;member" is "Non-member".
    while html.unescape(s) != s:
        s = html.unescape(s)
    s = re.sub(r"<br\s*/?>", " ", s)
    s = fix_footnotes(strip_markdown(s))
    return re.sub(r"(?<=[a-z][.!?])\s\d$", "", s)          # a footnote mark after a sentence: "... will apply. 4"


def _is_value(cell: str) -> bool:
    c = cell.lower().strip()
    if c in _EMPTY:
        return True
    if not _SINGLE_VALUE.match(c):
        return False
    rng = parse_amount_range(cell)
    return rng is None or rng[0] == rng[1]


def _value_with_terms(cell: str) -> bool:
    """A value followed by its conditions: '2.49% intro APR for 9 months, then ...'."""
    return bool(_LEADING_VALUE.match(cell.strip())) and len(cell) > 25


def _metric_of(header: str, cell: str = "") -> str:
    """What a value measures, from its cell ('6.84% APR') or else its column header."""
    h, c = header.lower(), cell.lower()
    if "apr" in c:
        return "apr"
    if "apy" in c:
        return "apy"
    if "apy" in h or "yield" in h:
        return "apy"
    if "dividend" in h:
        return "dividend"
    if "apr" in h or "annual percentage rate" in h:
        return "apr"
    if "point" in h:
        return "points"
    if "payment" in h:
        return "payment"
    if "min" in h and "open" in h:
        return "min_open"
    if "min" in h and ("earn" in h or "balance" in h):
        return "min_earn"
    if "fee" in h or "charge" in h or "cost" in h or "price" in h:
        return "fee"
    if "rate" in h:
        return "rate"
    return "value"


def _value(header: str, cell: str) -> Optional[Value]:
    c = cell.lower().strip()
    if c in _EMPTY and c != "free":
        return None
    pct, money = percent_of(cell), money_of(cell)
    if pct is not None:
        return Value(_metric_of(header, cell), header, cell, pct, "%")
    if money is not None:
        return Value(_metric_of(header, cell), header, cell, money, "$")
    try:
        return Value(_metric_of(header, cell), header, cell, float(cell.replace(",", "")), "")
    except ValueError:
        return Value(_metric_of(header, cell), header, cell, None, "")


_QUAL_WORDS = re.compile(
    r"\$\s?[\d,]+(?:\.\d+)?\+?|\b\d[\d,]*(?:\.\d+)?\b|\b(months?|mos?|years?|yrs?|or less|or more|and older|and up)\b|[-+()]",
    re.I)


def _name_tokens(text: str) -> frozenset:
    """Product words of a row name with the qualifier text (terms, amounts, years) removed."""
    return frozenset(content_tokens(_QUAL_WORDS.sub(" ", text)))


def _read_quals(key_cells: list[tuple[str, str]]) -> Quals:
    """Qualifiers from the row's key cells, given as (header, cell)."""
    term = amount = years = None
    for header, cell in key_cells:
        if term is None:
            term = parse_term(cell)
        if amount is None:
            amount = parse_amount_range(cell)
        # A cell under a "Model Year" column is a model year: "2024 and newer". Not under a plain "Year": the
        # year of a car for sale is no tier of a rate.
        dated = re.search(r"model|year|\b(19|20)\d\d\s?-\s?\d{2,4}\b", cell, re.I) or re.search(r"model", header, re.I)
        if years is None and dated and not parse_term(cell):
            years = parse_years(cell)
    return Quals(term, amount, years)


# --------------------------------------------------------------------------- page walking

_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")


def _label_like(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 70 or s[0] in "-*|[!>•" or s.endswith((".", ":", ",", ";")):
        return False
    if re.search(r"effective|subject to change|https?://", s, re.I) or s.lower() in ("table", "go!", "---"):
        return False
    return bool(re.search(r"[A-Za-z]{3}", s))


def _split_row(line: str) -> list[str]:
    return [_cell(c) for c in line.strip().strip("|").split("|")]


def extract(page) -> list[Row]:
    """All rows on a page, in page order."""
    lines = page.markdown.split("\n")
    rows: list[Row] = []
    headings: list[tuple[int, str]] = []
    doc_effective = effective_date(page.markdown[:600])      # "Schedule of Charges, Effective March 9, 2026"
    table_no = 0
    i = 0
    group = ""            # a plain label line that heads several labelled tables: "Certificates"
    plain = ""            # the latest plain label line: names the fee lines under it in documents
    while i < len(lines):
        line = lines[i]
        h = HEADING.match(line.strip())
        if h:
            level = len(h.group(1))
            headings = [x for x in headings if x[0] < level] + [(level, strip_markdown(h.group(2)))]
            group = plain = ""
            i += 1
            continue
        # A table: a header line, then a line of dashes. Written with or without its outer pipes
        # ("| Term | APY |" or "Term | APY"); without them, each of its lines holds a pipe.
        piped = line.lstrip().startswith("|")
        if "|" in line and i + 1 < len(lines) and "|" in lines[i + 1] and _SEPARATOR.match(lines[i + 1].strip()):
            j = i + 2
            while j < len(lines) and (lines[j].lstrip().startswith("|") if piped else "|" in lines[j]):
                j += 1
            rows.extend(_table_rows(page, table_no, lines, i, j, headings, group,
                                    doc_effective if page.kind == DOCUMENT else None))
            table_no += 1
            i = j
            continue
        if _label_like(line):
            k = i + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines) and _label_like(lines[k]) and not _FEE_LINE.match(strip_markdown(lines[k])):
                group = strip_markdown(line)
            # A fee line whose fee is a word ("Withdrawals at our ATMs Free") is a fee line too, not
            # the title of the lines under it.
            text = fix_footnotes(strip_markdown(line))
            if not _FEE_LINE.match(text) and not _WORD_FEE.search(text):
                plain = strip_markdown(line)
        rows.extend(_labelled(page, line, i, lines, headings, doc_effective, plain))
        i += 1
    # Rows printed side by side under one title (a table, tables that share a title, a page's labelled
    # values) are different facts even when they read alike once their numbers are set aside: box
    # sizes, two cards' "Standard Rate", two programs' "Awards available". They are told
    # apart by the order their values first appear in; a row printed twice with the same values stays one.
    values: dict[tuple, list] = {}
    for n, r in enumerate(rows):
        r.order = n
        seen = values.setdefault((r.slot, r.label if r.kind == "table" else ""), [])
        if r.signature() not in seen:
            seen.append(r.signature())
        r.variant = seen.index(r.signature())
    return rows


def _table_rows(page, table_no: int, lines: list[str], start: int, end: int, headings, group: str = "",
                doc_effective=None) -> list[Row]:
    header = _split_row(lines[start])
    body = [_split_row(line) for line in lines[start + 2:end]]
    body = [r for r in body if any(c for c in r)]
    if not body:
        return []
    ncol = len(header)
    body = [(r + [""] * ncol)[:ncol] for r in body]

    # Which columns hold values and which identify the row.
    is_value_col = []
    for c in range(ncol):
        cells = [r[c] for r in body if r[c]]
        share = sum(_is_value(x) for x in cells) / len(cells) if cells else 0.0
        # A rate column may hold one value that comes with its conditions spelt out.
        if 0.4 <= share < 0.6:
            share += sum(_value_with_terms(x) for x in cells) / len(cells)
        is_value_col.append(share >= 0.6 and c > 0)
    if not any(is_value_col) or all(is_value_col):
        return []                                    # not a table of published values
    value_cells = [r[c] for r in body for c in range(ncol) if is_value_col[c]]
    has_percent = any(percent_of(x) is not None for x in value_cells)
    has_money = any(money_of(x) is not None for x in value_cells)
    priced = re.search(r"fee|charge|cost|price|minimum|min\.|deposit|rate|apr|apy", " ".join(header), re.I)
    if not (has_percent or (has_money and priced)):
        return []                                    # chart data or a form, not a table of published values

    # Label: the nearest short line above, else the nearest heading.
    label = ""
    k = start - 1
    while k >= 0 and not lines[k].strip():
        k -= 1
    if k >= 0 and _label_like(lines[k]) and not HEADING.match(lines[k].strip()):
        label = strip_markdown(lines[k])
    elif headings:
        label = headings[-1][1]
    section = " / ".join([t for _, t in headings[-2:]] + ([group] if group and group != label else []))

    eff = _date_under(lines, end) or doc_effective
    ended = _ended_above(headings, lines, start)

    out = []
    for r in body:
        keys = [(header[c], r[c]) for c in range(ncol) if not is_value_col[c] and r[c]]
        vals = [v for c in range(ncol) if is_value_col[c] for v in [_value(header[c], r[c])] if v]
        name = ", ".join(c for _, c in keys)
        if not vals or not re.search(r"[A-Za-z]", name + label):
            continue
        quals = _read_quals(keys)
        if not re.search(r"[A-Za-z]{2}", name) and quals.is_empty():
            continue                                 # chart data ("1 | $0 | $25,000"), not a published value
        out.append(Row(page=page, table=table_no, label=label, section=section, name=name,
                       quals=quals, values=vals, effective=eff, expired=ended,
                       name_tokens=_name_tokens(name), label_tokens=frozenset(content_tokens(label)),
                       section_tokens=frozenset(content_tokens(section))))
    _nest_or_less_tiers(out)
    return out


# How far under a table its "effective as of" line may sit: so many lines of text, blank lines not
# counted (some pages put a dozen blank lines between a table and its date), and never past the
# next table or heading, whose date it would be.
_DATE_TEXT_LINES = 8
_DATE_ANY_LINES = 40


def _date_under(lines: list[str], end: int) -> Optional[date]:
    """The effective date printed under the table that ends on line `end`, or None. For a document
    the caller then falls back to the date printed at its top ("Schedule of Charges, Effective March 9, 2026")."""
    text_lines = 0
    for line in lines[end:end + _DATE_ANY_LINES]:
        if line.lstrip().startswith("|") or HEADING.match(line.strip()):
            return None
        if not line.strip():
            continue
        eff = effective_date(line)
        if eff:
            return eff
        text_lines += 1
        if text_lines >= _DATE_TEXT_LINES:
            return None
    return None


def _nest_or_less_tiers(rows: list[Row]) -> None:
    """'$120,000 or less', '$240,000 or less', ... are steps: the second covers 120,000.01-240,000."""
    groups: dict[tuple, list[Row]] = {}
    for r in rows:
        if r.quals.amount and r.quals.amount[0] == 0.0 and r.quals.amount[1] != INF:
            groups.setdefault((r.name_tokens, r.quals.term), []).append(r)
    for members in groups.values():
        if len(members) < 2:
            continue
        prev = 0.0
        for r in sorted(members, key=lambda x: x.quals.amount[1]):
            r.quals = Quals(r.quals.term, (prev, r.quals.amount[1]), r.quals.years)
            prev = r.quals.amount[1] + 0.01


# --------------------------------------------------------------------------- labelled values

_LABELLED = re.compile(r"^\s*(?:[-*•]\s+)?\**([A-Za-z0-9][^:|*]{2,80}?)\**\s*:\**\s*(.{1,120})$")
_WORD_FEE = re.compile(r"\sfree\*?$", re.I)
_FEE_LINE = re.compile(
    r"^\s*([A-Za-z(][^$|]{3,110}?)\s+(\$\d[\d,]*(?:\.\d+)?\*?(?:\s?(?:per|each|a|an|for)\b[A-Za-z /-]{0,40})?)\s*$")
_DAY = re.compile(DAY, re.I)
# A value, not "6-month ..." or "12-month ...": the number is read to its end before the unit is looked for.
_VALUE_START = re.compile(r"^\(?\[?\$?\d[\d,.]*(?![\d,.])(?!\s?-?\s?(?:months?|years?|days?|mos?\b|yrs?\b))", re.I)
_ENDED = re.compile(r"(?:offer|promotion|promo|special|rate)s?\s+(?:has\s+)?(?:ended|expired|ends|expires|closed)\s*(?:on\s+)?", re.I)


def _ended_on(text: str) -> Optional[date]:
    """The date a line says its offer ended (or ends): "Summer special ... (offer ended 2025-09-30)"."""
    e = _ENDED.search(text)
    if not e:
        return None
    found = find_dates(text[e.end():e.end() + 30])
    return found[0][1] if found else None


def _ended_above(headings, lines: Optional[list] = None, start: int = 0) -> Optional[date]:
    """An end date that covers a whole table or section: in a heading above it ("# Winter special
    (offer ended January 31, 2026)"), or in the two lines of text right above a table."""
    texts = [t for _, t in headings]
    k = start - 1
    while lines is not None and k >= 0 and len(texts) < len(headings) + 2:
        line = lines[k].strip()
        if line.startswith(("|", "#")):
            break
        if line:
            texts.append(fix_footnotes(strip_markdown(line)))
        k -= 1
    return next((d for d in map(_ended_on, texts) if d), None)


def _labelled(page, line: str, index: int, lines: list[str], headings, doc_effective, plain: str = "") -> list[Row]:
    raw = line.strip()
    if not raw or len(raw) > 240 or raw.startswith(("|", "![", "#")):
        return []
    text = fix_footnotes(strip_markdown(raw))
    section = " / ".join(t for _, t in headings[-2:])
    label = headings[-1][1] if headings else ""
    if page.kind == DOCUMENT and plain:
        label = plain                          # fee schedules have no headings, only plain section lines

    kind = name = value_text = None
    labelled = _LABELLED.match(text)
    if labelled and not _DAY.match(labelled.group(1)) and _VALUE_START.match(labelled.group(2).strip()):
        kind, name, value_text = "labelled", clean(labelled.group(1)), clean(labelled.group(2)).rstrip(". ")
    else:
        fee = _FEE_LINE.match(text)
        if fee and page.kind == DOCUMENT:
            kind, name, value_text = "fee", clean(fee.group(1)), clean(fee.group(2))
            if re.match(r"^[(a-z]", name) and index > 0 and lines[index - 1].strip() and "$" not in lines[index - 1]:
                name = clean(strip_markdown(lines[index - 1])) + " " + name      # the name wrapped onto two lines
        else:
            # "Summer special: 6-month certificate APY 2.95% (offer ended 2025-09-30)"
            claim = re.match(r"^(?:[-*•]\s+)?(?:[^:]{2,60}:\s*)?([A-Za-z0-9][^%$:]{3,90}?(?:apy|apr|rate|yield|fee)s?)\s*:?\s+"
                         r"((?:is|of|at)\s+)?(\$?\d+(?:\.\d+)?\s?%?)", text, re.I)
            if claim and (percent_of(claim.group(3)) is not None or "$" in claim.group(3)):
                kind, name, value_text = "claim", clean(claim.group(1)), clean(claim.group(3))
    if not kind:
        return []

    pct, money = percent_of(value_text), money_of(value_text)
    digits = re.sub(r"\D", "", value_text)
    if pct is not None:
        num, unit = pct, "%"
    elif money is not None:
        num, unit = money, "$"
    elif len(digits) >= 7 and re.fullmatch(r"[\d\s().+-]+", value_text):
        num, unit = None, ""                      # phone numbers, routing numbers: text identity
    else:
        return []
    metric = _metric_of(name, value_text) if kind != "fee" else "fee"
    expired = _ended_on(text) or _ended_above(headings)
    return [Row(page=page, table=-1, label=label, section=section, name=name,
                quals=_read_quals([("", name)]), values=[Value(metric, name, value_text, num, unit)],
                effective=doc_effective if page.kind == DOCUMENT else None, kind=kind, expired=expired,
                name_tokens=_name_tokens(name), label_tokens=frozenset(content_tokens(label)),
                section_tokens=frozenset(content_tokens(section)))]
