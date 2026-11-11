"""Every place the website contradicts itself, which value stands, and why.

Four kinds of entry, in this order, each found by its own detector below: rates and fees
(_rate_conflicts), restated facts and running counts (_restated_facts), closed places a branch
list still offers (_closed_places) and edited statements (_edited_statements). Each fact is one
entry, and the list is capped: a report that lists every changed digit hides the contradictions
that matter. Which copy is current is decided as the answers decide it (kb.lookup).
"""
from __future__ import annotations

import re
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Optional

from . import pages as P
from .compose import describe, reason
from .lookup import RATES, RowIndex, family
from .tables import Row
from .text import STOPWORDS, content_tokens, find_dates, strip_markdown

_REPORTED_METRICS = RATES | {"fee"}
_METRIC = {"apy": "APY", "apr": "APR", "rate": "rate", "dividend": "dividend rate", "fee": "fee"}
# Other names members use for the same product, so a topic can be found by any of them.
_ALSO = {"mortgage": "mortgage, home loan", "certificate": "certificate, CD", "auto": "auto loan, car loan, vehicle",
         "heloc": "HELOC, home equity line of credit", "creditcard": "credit card", "moneymarket": "money market"}
MAX_CONFLICTS = 30


@dataclass
class Found:
    """One entry of the report, before solution.py turns it into the contract's Conflict."""
    topic: str
    values: list
    sources: list = field(default_factory=list)       # (url, snapshot_id, quote)
    resolution: Optional[str] = None
    reason: Optional[str] = None


def find_conflicts(store) -> list[Found]:
    """The report: the four kinds of entry in order, the same contradiction once, at most MAX_CONFLICTS."""
    if not store.views:
        return []
    candidates = []
    for detector in (_rate_conflicts, _restated_facts, _closed_places, _edited_statements):
        try:
            candidates += detector(store)
        except Exception:               # a page that breaks one detector must not take the report down
            traceback.print_exc()
    found, known = [], []
    for c in candidates:
        nums = {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", " ".join(c.values))}
        # The same contradiction again: a business or IRA table that mirrors the personal one, or a
        # sentence that repeats what a table already reports.
        if nums and any(len(nums & k) >= 0.7 * len(nums | k) or nums <= k for k in known):
            continue
        found.append(c)
        known.append(nums)
    return found[:MAX_CONFLICTS]


_DATELINE = re.compile(r"[A-Z][a-z]+\.? \d{1,2}, \d{4}|\d{1,2}/\d{1,2}/\d{2,4}")


@lru_cache(maxsize=4096)                 # three detectors read the newest download's pages
def _sentences(markdown: str) -> tuple:
    """The page's text cut into sentences. Headings and table rows are left out (a changed row is a rate),
    and so are datelines: glued to the first sentence, a month would split one statement in two."""
    text = " ".join(line for line in markdown.split("\n")
                    if not line.lstrip().startswith(("|", "#")) and not _DATELINE.fullmatch(line.strip()))
    return tuple(re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])|\s+-\s+(?=[A-Z])", strip_markdown(text)))      # never inside "Oct. 1, 2024"


# --------------------------------------------------------------------------- 1. rates and fees


def _rate_values(row: Row) -> list:
    return [v for v in row.values if v.metric in _REPORTED_METRICS and v.num is not None]


def _figure(v) -> str:
    """'2.50% APY': the figure with its measure, unless it already says it ('5.39% APR')."""
    text, measure = v.text.strip(), _METRIC[v.metric]
    return text if re.search(rf"\b{measure}\b", text, re.I) else f"{text} {measure}"


def _printed(row: Row) -> str:
    return ", ".join(_figure(v) for v in _rate_values(row))


def _share(a, b) -> float:
    """How much two tiers overlap: the shared part of their ranges over both (1.0: the same tier)."""
    for x, y in ((a.amount, b.amount), (a.years, b.years)):
        if x and y:
            # An open end ("$25,000+", "2016 and older") is infinite; capped, so the share is a number.
            (x0, x1), (y0, y1) = [(max(lo, -1e12), min(hi, 1e12)) for lo, hi in (x, y)]
            lo, hi = max(x0, y0), min(x1, y1)
            if lo > hi:
                return 0.0
            return (hi - lo + 1) / (max(x1, y1) - min(x0, y0) + 1)
    return 1.0


def _mirror(r: Row) -> bool:
    """A business or IRA copy of a personal table: never a newer edition of the personal one."""
    return r.business or bool({"ira", "roth", "traditional"} & (r.name_tokens | r.label_tokens))


def _current_rows(store) -> tuple[list[Row], RowIndex]:
    """The rows in force in the newest download, after the same resolution the answers use."""
    latest = store.latest
    index: RowIndex = latest.cache["rows"]
    current: list[Row] = []
    for copies in index.slots.values():
        # Every copy competes, as in the answers (lookup.resolve), so that "shown on more pages"
        # counts the same pages and both name the same winner. Only then are the winners that
        # this report leaves out set aside: an article's or a calculator's copy, a sentence.
        win = index.winner(copies)
        if (not index.expired(win) and win.kind in ("table", "labelled")
                and win.page.kind not in (P.ARTICLE, P.INDEX, P.CALCULATOR) and _rate_values(win)):
            current.append(win)
    current = [r for r in current
               if not any((o.page.key, o.table) != (r.page.key, r.table) and family(r, o) and _mirror(r) == _mirror(o)
                          and index.generation(o, r) > 0 for o in current)]
    return [r for r in current if latest.authority(r.page) >= P.AUTHORITY[P.DOCUMENT]], index


def _older_copy_of(old: Row, seen: int, rows: list, index: RowIndex) -> Optional[Row]:
    """The current row that `old` is an older copy of: the closest tier of the same product and term
    that beats it as an edition (lookup.resolve). The same tier also counts in an earlier download,
    where the table may have had another title. Another tier counts only in the newest download: a
    tier that exists only in an earlier download was rolled over (last year's model years), not
    contradicted. A row without a tier has no other edition: "HELOC" and "HELOC Intro Rate"."""
    def share(r: Row) -> float:
        if r.quals.is_empty() or not family(r, old):
            return 0.0
        newer = index.generation(r, old) > 0
        if old.quals != r.quals:
            ok = newer and seen == index.view.index        # another tier: only while it is still printed
        else:
            ok = newer or seen < index.view.index          # the same tier: also from an earlier download
        if not ok:
            return 0.0
        return _share(r.quals, old.quals)
    best = max(rows, key=share, default=None)
    return best if best is not None and share(best) > 0 else None


def _rate_conflicts(store) -> list[Found]:
    """Rates and fees. For each product and term, the first row (in table order) that the site has
    shown with another value: an older copy still printed on a less trusted page, or the value an
    earlier download showed. The entry holds that row's own history only, never a sibling tier's
    figure, and its resolution also gives the current figure of every sibling tier."""
    latest = store.latest
    current, index = _current_rows(store)
    by_slot = {r.slot: r for r in current}
    by_term: dict[tuple, list] = defaultdict(list)
    for r in current:
        by_term[(r.business, r.quals.term)].append(r)
    history: dict[int, list] = defaultdict(list)          # id(current row) -> [(older copy, last download seen)]
    for versions in store.versions.values():
        for version in versions:
            for old in version.tables:
                if not _rate_values(old):
                    continue
                cur = by_slot.get(old.slot) or _older_copy_of(old, version.last_seen,
                                                               by_term[(old.business, old.quals.term)], index)
                if cur is not None and cur is not old and cur.signature() != old.signature():
                    history[id(cur)].append((old, version.last_seen))

    def stated(rows) -> set:
        return {(v.metric, v.num) for r in rows for v in _rate_values(r)}

    # The headline of a product and term: its first row (in table order) whose history shows a figure it does not.
    pos = {key: n for n, key in enumerate(latest.pages)}
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for r in current:
        groups[(r.business, r.name_tokens | r.label_tokens, r.quals.term)].append(r)
    heads = [head for rows in groups.values() for head in
             [next((r for r in rows if stated(o for o, _ in history[id(r)]) - stated([r])), None)] if head]
    heads.sort(key=lambda r: (_mirror(r), pos[r.page.key], r.table, len(r.name_tokens), r.order))
    # Products whose tables are set alike (purchase and refinance rates of one term) are one entry.
    partners: dict[tuple, list] = {}
    for head in heads:
        notes = frozenset((v.metric, v.num) for v in _rate_values(head) if v.metric in ("rate", "dividend"))
        key = (head.business, head.quals, notes) if notes and not _mirror(head) else id(head)
        partners.setdefault(key, []).append(head)
    out = []
    for head, *others in partners.values():
        olds = sorted(history[id(head)], key=lambda x: (-x[1], -(x[0].effective or date.min).toordinal()))
        values, listed = [], set()
        for row in [head] + [o for o, _ in olds]:
            new = [v for v in _rate_values(row) if (v.unit, v.num) not in listed]
            if len(values) + len(new) > 8:
                break
            values += [_figure(v) for v in new]
            listed |= {(v.unit, v.num) for v in new}
        if len(values) < 2:
            continue
        tiers = [r for h in [head] + others for r in current if r is not head and r.business == h.business
                 and r.quals.term == h.quals.term and family(h, r)]
        name = describe(head) if not others else f"{' / '.join(dict.fromkeys(h.label for h in [head] + others))}, {head.name}"
        measures = " and ".join(dict.fromkeys(_METRIC[v.metric] for v in _rate_values(head)))
        also = sorted({_ALSO[w] for w in head.name_tokens | head.label_tokens if w in _ALSO})
        topic = ("Business " if head.business and "business" not in name.lower() else "") + f"{name}: {measures}" \
            + (f" [{'; '.join(also)}]" if also else "")
        live = [o for o, seen in olds if seen == latest.index]          # a stale copy still printed today
        # An old copy under other words may be another product's table, so its title is said.
        last = olds[0][0]
        titled = (f' in a table titled "{last.label}"'
                  if last.name_tokens | last.label_tokens != head.name_tokens | head.label_tokens else "")
        because = f"Current because {reason(index, head, live[0])}." if live else (
            f"The newest download ({latest.taken_on.isoformat()}) shows {_printed(head)}; the other values are from "
            f"earlier downloads, the last of them from {store.views[olds[0][1]].taken_on.isoformat()}{titled}.")
        out.append(Found(
            topic=topic, values=values, reason=because,
            sources=[(head.page.url, latest.snapshot_id, f"{describe(head)}: {_printed(head)}")]
            + [(o.page.url, store.views[seen].snapshot_id, f"{describe(o)}: {_printed(o)}") for o, seen in olds[:5]],
            resolution=f"{_printed(head)} ({describe(head)})"
            + "".join(f"; {describe(r)}: {_printed(r)}" for r in list(dict.fromkeys(tiers))[:10])))
    return out


# --------------------------------------------------------------------------- 2. restated facts

_NUMBER = re.compile(r"\$?\b\d[\d,]*(?:\.\d+)?\b(?:\s(?:million|billion))?%?")
# Words that end the phrase after a figure: it names nothing counted ("$300 million this spring").
_COUNT_FILLER = STOPWORDS | frozenset("nearly almost approximately exceeding exceeds grown total totaling".split())
_BEFORE = re.compile(r"([a-z]+)\s+(?:has grown to|have grown to|exceed(?:ing|s)?|totall?ing)\s+"
                     r"(?:more than\s+|over\s+|nearly\s+|about\s+)?$", re.I)
_QUALIFIER = re.compile(r"(?:more than|over|nearly|about|almost|approximately)\s+$", re.I)


def _counted(sentence: str, m: re.Match) -> tuple[str, str]:
    """What a figure counts, and the words that state it: ('employees', 'over 600 employees'),
    ('assets', 'assets exceeding $1.2 billion'). ('', '') when it counts nothing named."""
    start = max(0, m.start() - 60)
    before = _BEFORE.search(sentence[start:m.start()])
    if before:
        return before.group(1).lower(), sentence[start + before.start():m.end()]
    qualifier = _QUALIFIER.search(sentence[start:m.start()])
    words = []
    for w in re.finditer(r"[A-Za-z]+", sentence[m.end():m.end() + 45]):
        if len(words) == 3 or w.group().lower() in _COUNT_FILLER:
            break
        words.append(w.group().lower())
        if words[-1].endswith("s"):                      # a plural noun closes the phrase: "... centers"
            return " ".join(words), sentence[start + qualifier.start() if qualifier else m.start():m.end() + w.end()]
    return "", ""


def _is_plain_count(value: str, sentence: str, m: re.Match) -> bool:
    if re.fullmatch(r"(19|20)\d\d|0\d", value):
        return False                                            # a year, or a day of the month
    if re.search(r"[-.(/]$", sentence[max(0, m.start() - 1):m.start()]) or re.match(r"[-./)]\d", sentence[m.end():]):
        return False                                            # part of a phone number or a date
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s$", sentence[max(0, m.start() - 12):m.start()], re.I):
        return False
    return bool(re.search(r"\d", value))


def _restated_facts(store) -> list[Found]:
    """Restated facts. A count that many pages repeat with different numbers ("... and 31 service
    centers"): the most recently published statement stands. Then the running counts (_running_counts)."""
    latest = store.latest
    families: dict[str, dict] = defaultdict(dict)     # the sentence with its figures masked -> {page: (page, sentence)}
    for page in latest.pages.values():
        for sentence in _sentences(page.markdown) if page.kind in (P.ARTICLE, P.OFFICIAL) else []:
            if 40 <= len(sentence) <= 400 and _NUMBER.search(sentence) and "min read" not in sentence:
                families[_NUMBER.sub("#", sentence.lower())].setdefault(page.key, (page, sentence))
    facts: dict[str, dict] = defaultdict(dict)        # what is counted -> {value: (date, page, sentence, span)}
    spread: dict[str, set] = defaultdict(set)
    for members in families.values():
        if len(members) < 5:
            continue
        for page, sentence in members.values():
            when = page.published or (latest.taken_on if page.kind == P.OFFICIAL else date.min)
            for m in _NUMBER.finditer(sentence):
                noun, span = _counted(sentence, m)
                if not noun or not _is_plain_count(m.group(0), sentence, m):
                    continue
                spread[noun].add(page.key)
                held = facts[noun].get(m.group(0))
                if held is None or when > held[0]:
                    facts[noun][m.group(0)] = (when, page, sentence, span)
    out = []
    for noun, values in facts.items():
        if len(values) < 3 or len(spread[noun]) < 8:
            continue                                      # a standing fact: on many pages, with several values
        ranked = sorted(values.values(), key=lambda x: x[0], reverse=True)
        when, page, sentence, best = ranked[0]
        if when == date.min:
            continue
        shown = [span for _, _, _, span in ranked][:8]
        sources = [(p.url, latest.snapshot_id, s[:200]) for _, p, s, _ in ranked[:4]]
        because = (f"The most recently published statement ({when.isoformat()}) stands; the other figures were true "
                   f"when their pages were published.")
        # A figure that carries its own date ("As of: June 30, 2026 ... Members: 57,409") and is
        # newer than every press release is the better statement of the same fact.
        dated = _dated_figure(latest, noun, after=when)
        if dated is not None:
            figure, as_of, page, quote = dated
            shown = [figure] + shown[:7]
            sources.insert(0, (page.url, latest.snapshot_id, quote[:200]))
            because = (f"Stated with its own date ({as_of.isoformat()}) on {page.url}, later than the newest press "
                       f"release ({when.isoformat()}); the other figures were true when their pages were published.")
            best = f"{figure} (as of {as_of.isoformat()})"
        also = _also_counted(noun)
        out.append(Found(topic=f"{noun.capitalize()}{' (' + also + ')' if also else ''}: a figure restated across pages",
                         values=shown, sources=sources[:5], resolution=best, reason=because))
    return out + _running_counts(latest)


# Another name for what a count counts, by its last word; any phrase with "branch" in it counts branches.
_ALSO_COUNTED = {"membership": "members", "employees": "staff"}


def _also_counted(noun: str) -> str:
    """The other name for a counted phrase ('branch sites' -> 'branches'), or '' when it has none."""
    return _ALSO_COUNTED.get(noun.split()[-1]) or ("branches" if "branch" in noun.split() else "")


def _dated_figure(view, noun: str, after: date) -> Optional[tuple]:
    """A table-like line that states the figure under an 'as of' date: (figure, date, page, quote)."""
    # The line names what the phrase counts, its last word: "membership area counties" are counties,
    # not members. Six letters, so that "membership" finds "Members:".
    word = re.escape(noun.split()[-1][:6])
    figure = re.compile(rf"\b{word}[a-z]*\s*:\s*(\$?\d[\d,]*(?:\.\d+)?(?:\s(?:million|billion))?)", re.I)
    best = None
    for page in view.pages.values():
        if page.kind not in (P.OFFICIAL, P.DOCUMENT):
            continue
        for m in re.finditer(r"\bas of:?\s+([A-Za-z]+\.? \d{1,2},? \d{4}|\d{1,2}/\d{1,2}/\d{4})", page.markdown, re.I):
            found = find_dates(m.group(1))
            hit = figure.search(page.markdown[m.end():m.end() + 400])
            if found and hit and after < found[0][1] <= view.taken_on and (best is None or found[0][1] > best[1]):
                quote = strip_markdown(page.markdown[m.start():m.end() + hit.end()]).replace("|", " ")
                best = (hit.group(0), found[0][1], page, " ".join(quote.split())[:200])     # a table's cells as one line
    return best


_ORDINALS = "first second third fourth fifth sixth seventh eighth ninth tenth".split()
_NTH_TIME = re.compile(rf"\b({'|'.join(_ORDINALS)}|\d{{1,2}}(?:st|nd|rd|th))\s+time\s+(\w+(?:\s+\w+){{0,3}})", re.I)


def _running_counts(view) -> list[Found]:
    """'The fourth time X' in one release and 'the fifth time X' in a later one: a running count,
    each true when published. Reported when the newest release holds the highest count."""
    said: dict[str, dict] = defaultdict(dict)          # the words after "time" -> {count: (published, page, ordinal, words)}
    for page in view.pages.values():
        for m in _NTH_TIME.finditer(" ".join(_sentences(page.markdown)) if page.kind == P.ARTICLE and page.published else ""):
            word, key = m.group(1).lower(), m.group(2)
            n = _ORDINALS.index(word) + 1 if word in _ORDINALS else int(word[:-2])
            if n not in said[key.lower()] or page.published > said[key.lower()][n][0]:
                said[key.lower()][n] = (page.published, page, m.group(1), key)
    out = []
    for by_count in said.values():
        ranked = [x for _, x in sorted(by_count.items(), reverse=True)]
        if len(ranked) < 2 or len({p.key for _, p, _, _ in ranked}) < 2 or ranked[0][0] < max(x[0] for x in ranked):
            continue
        when, page, top, key = ranked[0]
        out.append(Found(
            topic=f"How many times {key} (a running count restated across releases)",
            values=[f"{w} time" for _, _, w, _ in ranked], resolution=f"{top} time ({page.label()})",
            sources=[(p.url, view.snapshot_id, f"the {w} time {k}") for _, p, w, k in ranked[:5]],
            reason=f"A running count only grows: the newest release ({when.isoformat()}) has the highest; each older "
                   f"figure was true when its release was published."))
    return out


# --------------------------------------------------------------------------- 3. closed places

_CLOSES = re.compile(r"\b(?:will close|to close|last day of (?:service|business|operation))\b", re.I)
_PLACE = re.compile(r"\b(?:branch|location|office)", re.I)
# One option of a list run together on one line: "Alder: Main St.Birch: Oak Ave.Cedarton: 1st St.".
_OPTION = re.compile(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?: .+?(?=[A-Z][a-z]+(?: [A-Z][a-z]+)?: |$)")


def _closed_places(store) -> list[Found]:
    """Closed places. A place a release says closed (or will close, on a date already past) that a
    page of the newest download still offers in a list of branches."""
    view = store.latest
    closures = []                                       # (release, sentence, date as printed, words naming the place)
    for page in view.pages.values():
        for sentence in _sentences(page.markdown) if page.kind == P.ARTICLE else []:
            closes = _CLOSES.search(sentence)
            dates = find_dates(sentence[closes.end():]) if closes else []
            # A closing that "reopens" is a holiday or a move, not a place that is gone.
            if dates and dates[0][1] < view.taken_on and _PLACE.search(page.title + sentence) \
                    and not re.search(r"\breopen", sentence, re.I):
                tail, year = sentence[closes.end() + dates[0][0]:], str(dates[0][1].year)
                printed = tail[:tail.index(year) + 4] if year in tail else dates[0][1].isoformat()
                words = set(content_tokens(page.title + " " + sentence)) - {"branch", "location", "office"}
                closures.append((page, sentence, printed, words))
    out = {}
    for page in view.pages.values() if closures else []:
        for line in strip_markdown(page.markdown).split("\n"):
            options = [o.strip() for o in _OPTION.findall(line)]
            if len(options) < 8:
                continue                                # a list of branches to choose from, not a sentence
            words = [set(content_tokens(o)) for o in options]
            for option, own in zip(options, words):
                # The words of this option alone: a town that has several branches names none of them.
                distinct = own - set().union(*(w for w in words if w is not own))
                for release, sentence, printed, place in closures:
                    if distinct & place and option not in out:
                        out[option] = Found(
                            topic=f"Branch closure: {option} is still offered in a branch list after it closed",
                            values=[printed, option], resolution=f"{printed}: closed, no longer a branch",
                            sources=[(release.url, view.snapshot_id, sentence[:200]), (page.url, view.snapshot_id, option)],
                            reason=f"{release.url} ({release.label()}) says it closes on {printed}, yet {page.url} "
                                   f"still offers it in the newest download ({view.taken_on.isoformat()}).")
    return list(out.values())


# --------------------------------------------------------------------------- 4. edited statements

_WORKED_EXAMPLE = re.compile(r"\bexample\b|\billustrat|\bfor instance\b|\byou (?:will|would) pay\b|\bestimated?\b", re.I)
_MONEY_OR_PERCENT = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s(?:million|billion))?|\b\d+(?:\.\d+)?\s?%")


def _edited_statements(store) -> list[Found]:
    """Edited statements. A sentence on a product or campaign page whose amount changed between two
    downloads of that page ("up to $1,200" became "up to $1,750"). The newest download stands."""
    latest = store.latest
    out = []
    for key, history in store.versions.items():
        if len(history) < 2 or key not in latest.pages or history[-1].kind not in (P.OFFICIAL, P.LANDING):
            continue
        said: dict[str, dict] = defaultdict(dict)        # skeleton -> {version index: [(amounts, sentence)]}
        for n, version in enumerate(history):
            for sentence in _sentences(version.markdown):
                if not 30 <= len(sentence) <= 400 or _WORKED_EXAMPLE.search(sentence):
                    continue                              # a payment example follows the rate; it is not a second fact
                amounts = tuple(_MONEY_OR_PERCENT.findall(sentence))
                if amounts:
                    # Every figure and date masked: a sentence whose date changed with its rate still pairs.
                    said[re.sub(r"\d[\d,./]*", "#", sentence.lower())].setdefault(n, []).append((amounts, sentence))
        for by_version in said.values():
            last = len(history) - 1
            # One such sentence per download (two bullets of the same shape are two products), still on the page.
            if last not in by_version or any(len({a for a, _ in v}) != 1 for v in by_version.values()):
                continue
            now, sentence = by_version[last][0]
            earlier = [(n, v[0][0], v[0][1]) for n, v in sorted(by_version.items(), reverse=True) if v[0][0] != now]
            if not earlier:
                continue
            page = history[-1]
            # Only the amounts that changed: a sentence can carry other figures that stayed the same.
            moved = [i for i in range(len(now)) if any(len(a) == len(now) and a[i] != now[i] for _, a, _ in earlier)]
            if not moved:
                continue
            values = list(dict.fromkeys([now[i] for i in moved]
                                        + [a[i] for _, a, _ in earlier if len(a) == len(now) for i in moved]))
            if any(a != b and (a.startswith(b) or b.startswith(a)) for a in values for b in values):
                continue                                  # "$85" and "$852": a footnote mark, not a change
            around = re.search(rf"(?<!\S)(?:\S+\s+){{0,12}}{re.escape(now[moved[0]])}(?:\s+\S+){{0,6}}", sentence).group(0)
            text = ("..." if not sentence.startswith(around) else "") + re.sub(r"\s+", " ", around)
            n = earlier[0][0]
            out.append(Found(
                topic=f"{page.title}: {text}", values=values[:8],
                sources=[(page.url, latest.snapshot_id, sentence[:200])]
                + [(page.url, store.views[history[k].last_seen].snapshot_id, s[:200]) for k, _, s in earlier[:3]],
                resolution=", ".join(now[i] for i in moved),
                reason=(f"The page changed between the downloads of {store.views[history[n].last_seen].taken_on.isoformat()} "
                        f"and {store.views[history[n + 1].first_seen].taken_on.isoformat()}; the newest download shows "
                        f"this value.")))
    return out[:4]                                        # a few: most edits are a rate a table already reports
