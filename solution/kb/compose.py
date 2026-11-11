"""Word an answer from rows, without a language model.

The text carries only the rows that won: the current value of the tier that was asked for. A
value that lost (an old rate, a stale copy) is never written into the answer; it goes into
`conflict_note`, where a reviewer can see what was set aside and why. A copy that lost but states
only figures the answer gives is not set aside: its page is cited with the answer.
"""
from __future__ import annotations

import re
from typing import Optional

from .lookup import RATE_ORDER, RATES, Match, Query, RowIndex, family, newer, printed
from .reply import dated, figures
from .tables import Row, Value
from .text import content_tokens, fmt_date

_COMPARES = re.compile(r"\b(?:more|less|fewer|higher|lower|above|below|over|under|least|most|than|exceeds?|between|up to)\b",
                       re.I)
_METRIC_NAME = {
    "dividend": "dividend rate", "apy": "APY", "apr": "APR", "rate": "rate", "points": "points",
    "payment": "payment example", "min_open": "minimum to open", "min_earn": "minimum to earn", "fee": "", "value": "",
}
MAX_ROWS = 24


def _shown_values(q: Query, row: Row) -> list[Value]:
    wanted = [v for v in row.values if v.metric in RATE_ORDER]
    if "min" in q.metrics:
        wanted = [v for v in row.values if v.metric in ("min_open", "min_earn")] or wanted
    for extra in ("points", "payment", "fee"):
        if extra in q.metrics:
            wanted += [v for v in row.values if v.metric == extra and v not in wanted]
    return wanted or list(row.values)


def _value_text(v: Value, repeated: bool = False) -> str:
    text = v.text.strip()
    if repeated:                         # two APY columns: only the printed header tells them apart
        header = re.sub(r"[*\\]|\d+$", "", v.header).strip()
        return f"{text} ({header})"
    name = _METRIC_NAME.get(v.metric, "")
    if not name or re.search(rf"\b{re.escape(name)}\b", text, re.I):
        return text
    return f"{text} {name}"


def describe(row: Row) -> str:
    """'Classic Certificates, 60 Month' or 'Share Savings, $3,000 - $11,999.99'."""
    name = row.name.strip()
    if row.kind != "table":
        return name or row.label
    if row.label and not (row.label_tokens and row.label_tokens <= row.name_tokens):
        # The name's opening words, when they repeat the title, are said once: "Mortgages - Buying" and
        # "Buying: 30-Year Fixed".
        head = re.match(r"([^:,\d]+)[:,]\s*(?=\S)", name)
        words = set(content_tokens(head.group(1))) if head else set()
        if words and words <= row.label_tokens:
            name = name[head.end():]
        return f"{row.label}, {name}" if name else row.label
    return name or row.label


def in_order(q: Query, rows: list[Row]) -> list[Row]:
    """The rows in the site's order, except that the loan-amount or balance tiers of one product are
    listed from the smallest amount up when the question names no amount: the first tier reads as
    the headline, and the site often lists the largest first. Terms and model years keep their order."""
    if q.quals.amount is not None:
        return rows
    def product_key(r: Row) -> tuple:
        return (r.label, r.effective, r.business, r.page.key, r.quals.term, r.quals.years, r.quals.amount is None)
    out = []
    for r in rows:
        at = len(out)
        while (r.quals.amount is not None and at and product_key(out[at - 1]) == product_key(r)
               and out[at - 1].quals.amount > r.quals.amount):
            at -= 1
        out.insert(at, r)
    return out


def line(q: Query, row: Row) -> str:
    """One row as a line of the answer: 'Its name: its values'."""
    shown = _shown_values(q, row)
    metrics = [v.metric for v in shown]
    name = describe(row)
    values = ", ".join(_value_text(v, metrics.count(v.metric) > 1) for v in shown)
    if len(shown) == 1 and re.search(rf"\b{re.escape(_METRIC_NAME.get(shown[0].metric) or '~')}\b", name, re.I):
        values = shown[0].text.strip()          # "6-month certificate APY: 3.50%", not "... APY: 3.50% APY"
    return f"{name}: {values}"


def yes_or_no(q: Query, rows: list[Row]) -> str:
    """"Yes. " or "No. " for "Do any savings tiers pay 0.35% APY?", by whether a row pays that rate in
    the measure named (any rate when none is), so the rates that follow answer the question as asked.
    Nothing for a comparison ("more than 4%"), or when no row states the measure named."""
    asked = {round(float(n), 4) for n in re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", q.text)}
    if not asked or not re.match(r"\W*(?:do|does|is|are)\b", q.text) or _COMPARES.search(q.text):
        return ""
    measures = q.metrics & (RATES - {"rate"}) or RATES
    paid = {round(v.num, 4) for r in rows for v in r.values if v.metric in measures and v.num is not None}
    return "" if not paid else "Yes. " if asked & paid else "No. "


def _pick_extreme(q: Query, rows: list[Row]) -> list[Row]:
    """'highest' / 'lowest': keep the row(s) with the extreme value of the measure asked for. With
    two measures named ("the highest dividend or APY"), the first in RATE_ORDER decides: a fixed
    order, never the order a set happens to iterate in."""
    asked = [m for m in RATE_ORDER if m in q.metrics]

    def key(r: Row) -> Optional[float]:
        for metric in asked + list(RATE_ORDER):
            for v in r.values:
                if v.metric == metric and v.num is not None:
                    return v.num
        return None
    valued = [(key(r), r) for r in rows if key(r) is not None]
    if not valued:
        return rows
    best = (max if q.superlative == "max" else min)(k for k, _ in valued)
    return [r for k, r in valued if k == best]


def answer_from_rows(q: Query, match: Match, index: RowIndex, as_of: Optional[str]) -> tuple[str, list, Optional[str]]:
    """The answer's text, its source pages (most trusted first) and the note on the copies set aside."""
    rows = match.rows
    # A value stated both in a table and in a sentence under the same heading: the table speaks.
    table_labels = {r.label_tokens for r in rows if r.kind == "table"}
    rows = [r for r in rows if r.kind == "table" or r.label_tokens not in table_labels] or rows
    if q.superlative:
        rows = _pick_extreme(q, rows)
    view = index.view
    several_headings = len({r.label for r in rows if r.kind != "table"}) > 1

    # Group by table so each table's effective date is said once.
    groups: list[tuple[str, list[Row]]] = []
    for r in in_order(q, rows[:MAX_ROWS]):
        key = (r.label, r.effective)
        if groups and groups[-1][0] == key:
            groups[-1][1].append(r)
        else:
            groups.append((key, [r]))
    parts = []
    for (_, eff), members in groups:
        body = "; ".join((f"{r.label}: " if several_headings and r.kind != "table" and r.label else "") + line(q, r)
                         for r in members)
        if eff and any(r.dated_by != "change" for r in members):     # a change date is a lower bound, not printed
            body += f" (rates effective {fmt_date(eff)})"
        parts.append(body)
    text = yes_or_no(q, rows) + ". ".join(parts) + "."
    if len(rows) > MAX_ROWS:
        text += f" ({len(rows) - MAX_ROWS} more rows are listed on the rates page.)"
    if as_of:
        text = dated(text, as_of, view)
    elif any(r.page.key in view.gone for r in rows):
        text += " (This page is no longer on the website; this is the last version that was published.)"

    # Sources: every page that prints the winning rows or a copy that repeats them, most trusted first.
    repeats = [x for _, x in match.rivals if _repeats(x, rows[:MAX_ROWS])]
    pages, seen = [], set()
    for r in rows + repeats:
        for p in match.sources.get(id(r), [r.page]):
            if p.key not in seen:
                seen.add(p.key)
                pages.append(p)
    words = set(q.product)
    pages.sort(key=lambda p: (-view.authority(p), -len(words & index.page_tokens.get(id(p), frozenset())), len(p.key)))
    return text, pages, _conflict_note(index, match, [(w, x) for w, x in match.rivals if x not in repeats])


def _repeats(lose: Row, rows: list[Row]) -> bool:
    """Whether a copy that lost states no figure the answer's row does not give, so it is no rival.
    One answer row must give all its figures and be the same item: the same term and tier, in a copy
    of the row or of its product (an older edition may name its model years otherwise, but its years
    must overlap: 2019-20 models are not 2025-26 models). Figures pooled from two rows would hide an
    old rate that another tier pays today. Each value is compared with the row's values of the same
    measure; a fee and an unnamed amount stand for each other (a fee read from a column headed
    "Amount" in one copy and from a line of text in another), but never a minimum balance.
    "11.25%-17.25% APR" repeats "11.25% - 17.25%", and "2.10%" repeats a row giving "3.90%" and
    "2.10%"; "2.15%" does not repeat "2.12%"; a value in words only ("Free") never repeats."""
    def covered(r: Row) -> bool:
        def given(metric: str) -> set:
            same = [w for w in r.values if w.metric == metric] or [w for w in r.values if w.metric in ("fee", "value")]
            return {f for w in same for f in figures(w.text)}
        return bool(lose.values) and all(figures(v.text) and figures(v.text) <= given(v.metric) for v in lose.values)
    def overlap(a, b) -> bool:
        return not a or not b or (a[0] <= b[1] and b[0] <= a[1])
    return any((r.quals.term, r.quals.amount) == (lose.quals.term, lose.quals.amount) and overlap(r.quals.years, lose.quals.years)
               and (r.slot == lose.slot or family(r, lose)) and covered(r) for r in rows)


def _conflict_note(index: RowIndex, match: Match, rivals: list) -> Optional[str]:
    """The values that lost to the answer's rows, where they are printed, and why the winner won."""
    if not rivals:
        return None
    seen, items = set(), []
    for _, lose in rivals:
        v = lose.primary()
        if not v:
            continue
        key = (lose.page.key, v.text)
        if key in seen:
            continue
        seen.add(key)
        when = f", effective {lose.effective.isoformat()}" if printed(lose) else ""      # a change date is not printed
        items.append(f"{v.text} on {lose.page.url} ({lose.page.label()}{when})")
        if len(items) >= 4:
            break
    if not items:
        return None
    # Explain a pair that really competed, won by a row the answer gives (rows[0] may be another product).
    win, lose = next(((w, x) for w, x in rivals if w in match.rows), rivals[0])
    why = reason(index, win, lose)
    return "Other copies of this item show different values: " + "; ".join(items) + f". Chosen because {why}."


def reason(index: RowIndex, win: Row, lose: Row) -> str:
    """Why `win` beats `lose`, in words: the first rule of RowIndex.better, in its order, that
    tells them apart."""
    view = index.view
    if index.expired(lose) and not index.expired(win):
        return f"the other value is a promotion that ended {lose.expired.isoformat()}"
    if view.authority(win.page) > view.authority(lose.page):
        if lose.page.key in view.gone:
            return "the other copy is on a page that is no longer on the website"
        if lose.page.key in view.stale:
            dates = (f" (its copy is dated {lose.effective.isoformat()}, this one {win.effective.isoformat()})"
                     if printed(win) and printed(lose) and win.effective > lose.effective else "")
            return f"the other copy is on a page that was left behind: its rate tables are older than the same tables elsewhere{dates}"
        return f"{_a(win.page.label())} outranks {_a(lose.page.label())}"
    if newer(win, lose) > 0:
        if not printed(win):
            return (f"its table changed after the download of {win.effective.isoformat()}, later than the other "
                    f"copy's {'date' if printed(lose) else 'change'} ({lose.effective.isoformat()})")
        return (f"its rates are effective {win.effective.isoformat()}, newer than the other copy's "
                f"{lose.effective.isoformat()}")
    if win.maintained and not lose.maintained:
        return "its page keeps updating this copy; the other copy never changes"
    if win.slot == lose.slot:                  # copies of one row: how many pages show each value
        shown = index.support(index.slots.get(win.slot, []))
        if len(shown[win.signature()]) > len(shown[lose.signature()]):
            return "more pages show this value"
    if printed(win) and not printed(lose):
        return f"it carries an effective date ({win.effective.isoformat()}) and the other copy does not"
    if win.page.fetched > lose.page.fetched:
        return "it is on the more recently downloaded page"
    if win.kind == "table" and lose.kind != "table":
        return "it is printed in a table and the other copy in a sentence"
    if len(win.page.key) < len(lose.page.key):
        return "nothing else tells the copies apart, and its page has the shorter address"
    return "nothing else tells the copies apart, and its page's address comes first in alphabetical order"


def _a(noun: str) -> str:
    """'a product page', 'an article': the page kinds all start with a letter that is said as written."""
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun
