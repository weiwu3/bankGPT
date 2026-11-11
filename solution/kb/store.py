"""History, not just the latest copy.

The store keeps every distinct version of every page and, for each download, a View: the
website exactly as it stood on that date. Questions about "now" use the newest View; "as of"
questions use the View of the last download on or before the date.

A page that was downloaded before but is missing from a later download is not forgotten: its
last version stays in later Views, marked `gone`, and is trusted a little less than a live page
of the same kind: a vanished product page still outranks a live document or campaign page.
as_of_day reads the date a question is asked as of.
"""
from __future__ import annotations

import calendar
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterator, Optional

from . import pages as P
from .tables import extract
from .text import MONTHS, clean, forget_site, learn_names, learn_site_name


@dataclass(eq=False)
class View:
    """The website as one download saw it: its pages, the pages gone since, and their rows."""
    index: int
    snapshot_id: str
    taken_on: date
    pages: dict                                   # url key -> PageVersion present in this download
    gone: dict = field(default_factory=dict)      # url key -> last known PageVersion of a vanished page
    stale: set = field(default_factory=set)       # url keys whose dated tables are behind the same tables elsewhere
    rows: list = field(default_factory=list)      # every Row of every page in the view
    cache: dict = field(default_factory=dict)     # indexes built on demand by other modules

    def all_pages(self) -> Iterator[P.PageVersion]:
        """The live pages, then the gone ones."""
        yield from self.pages.values()
        yield from self.gone.values()

    def authority(self, page) -> float:
        """How much a page is trusted in this view when pages disagree."""
        a = float(P.AUTHORITY[page.kind])
        if page.key in self.stale:
            a = min(a, float(P.AUTHORITY[P.LANDING]))
        if page.key in self.gone:
            a -= 0.5
        return a

    def pages_printing(self, value: str, exclude=()) -> list:
        """The live pages that print `value`, except those in `exclude` (by address), most trusted first."""
        have = {p.key for p in exclude}
        return sorted((p for p in self.pages.values() if p.key not in have and value in p.markdown),
                      key=lambda p: (-self.authority(p), len(p.key)))


class Store:
    """Every version of every page, and one View per download, oldest first."""

    def __init__(self) -> None:
        self.versions: dict[str, list[P.PageVersion]] = defaultdict(list)
        self.views: list[View] = []

    # ------------------------------------------------------------------ ingest

    def add_snapshot(self, snapshot) -> View:
        """Read one download into a new View, then relearn what the page histories tell."""
        # Snapshots arrive oldest first (the scorer sorts them by date); every index below assumes that.
        index = len(self.views)
        taken = _to_date(snapshot.taken_on)
        present: dict[str, P.PageVersion] = {}
        if not any(view.pages for view in self.views):
            # Vocabulary taken from the site rather than written into the code: its own name
            # (not a product word) and its coined names. Learned once, from the first download
            # that has pages, before anything is tokenised, so that every download is read with
            # the same vocabulary. The vocabulary is process-wide (kb.text), so what an earlier
            # site taught is forgotten first.
            forget_site()
            hosts = Counter(P.url_key(page.url).split("/")[0] for page in snapshot.pages)
            if hosts:
                learn_site_name(hosts.most_common(1)[0][0])
            learn_names(page.markdown for page in snapshot.pages)
        for page in snapshot.pages:
            key = P.url_key(page.url)
            history = self.versions[key]
            if history and history[-1].markdown == page.markdown:
                version = history[-1]
                version.last_seen = index
            else:
                version = P.PageVersion(
                    url=page.url, key=key, title=P.page_title(page.title, page.markdown), markdown=page.markdown,
                    fetched=page.fetched_at or snapshot.taken_on, first_seen=index, last_seen=index,
                    kind=P.classify(page.url, page.markdown))
                if version.kind == P.ARTICLE:
                    version.published = P.published_on(page.url, page.markdown)
                version.tables = extract(version)
                history.append(version)
            present[key] = version
        gone = {key: history[-1] for key, history in self.versions.items() if history and key not in present}
        view = View(index=index, snapshot_id=snapshot.snapshot_id, taken_on=taken, pages=present, gone=gone)
        view.rows = [row for page in view.all_pages() for row in page.tables]
        self.views.append(view)
        self._learn_from_history()
        return view

    def _learn_from_history(self) -> None:
        """What only the history of a page can tell.

        0. Labels. The same table (same row names on the same page) takes its fullest title.
        1. Dates. Early downloads print no "effective as of" line. When the same page later
           shows the same table with a date, that date belongs to the earlier copy too.
        2. Maintenance. A table this page has changed over the downloads is being kept up to
           date. A table that never moves while other copies of it do is a forgotten copy.
           This matters when two undated copies disagree.
        3. Change dates. An undated table that the page changed between two downloads took
           effect after the earlier download, so it is dated to that download (a lower bound;
           see lookup.newer for what it may decide). Later versions of the page that keep the
           table as it is keep that date.
        """
        for history in self.versions.values():
            # 0. Labels. Early downloads title a table "Classic"; later ones "Classic Certificates".
            titles: dict[tuple, object] = {}
            for version in history:
                for rows in _tables_of(version).values():
                    names = tuple(r.name.lower() for r in rows)
                    best = titles.get(names)
                    if best is None or len(rows[0].label_tokens) > len(best.label_tokens):
                        titles[names] = rows[0]
            for version in history:
                for rows in _tables_of(version).values():
                    best = titles[tuple(r.name.lower() for r in rows)]
                    if best.label_tokens > rows[0].label_tokens:
                        for r in rows:
                            r.label, r.label_tokens = best.label, best.label_tokens
            # 1-3 work on whole tables: every row of a table shares its date and its upkeep.
            # They run after every download, so the dates they gave before are taken back first:
            # a date printed in a later download (rule 1) must win over a change date (rule 3).
            for version in history:
                for r in version.tables:
                    if r.dated_by in ("history", "change"):
                        r.effective, r.dated_by = None, "page"
            dated: dict[tuple, date] = {}
            contents: dict[frozenset, set] = defaultdict(set)      # table title -> the contents it has had
            for version in history:
                for content, rows in _tables_of(version).items():
                    eff = next((r.effective for r in rows if r.effective and r.dated_by == "page"), None)
                    if eff and (content not in dated or eff < dated[content]):
                        dated[content] = eff
                    contents[rows[0].label_tokens].add(content)
            previous: dict[frozenset, tuple] = {}
            since: dict[frozenset, Optional[date]] = {}            # table title -> change date of its contents
            for n, version in enumerate(history):
                current = {}
                for content, rows in _tables_of(version).items():
                    title = rows[0].label_tokens
                    current[title] = content
                    # A later version of the page with the same table (only other text was edited)
                    # keeps the date of the change that brought these contents.
                    if previous.get(title) != content:
                        changed = n > 0 and version.first_seen > 0
                        since[title] = self.views[version.first_seen - 1].taken_on if changed else None
                    for r in rows:
                        r.maintained = len(contents[title]) > 1
                        if r.effective is None and content in dated:
                            r.effective, r.dated_by = dated[content], "history"
                        elif r.effective is None and since.get(title):
                            r.effective, r.dated_by = since[title], "change"
                previous = current

    # ------------------------------------------------------------------ views

    @property
    def latest(self) -> Optional[View]:
        """The newest View, or None before the first download."""
        return self.views[-1] if self.views else None

    def view_at(self, as_of: object) -> Optional[View]:
        """The website as it stood on `as_of` (see as_of_day for how it may be written); the
        newest view when no date is given or it cannot be read."""
        if not self.views:
            return None
        day = as_of_day(as_of)
        if day is None:                           # no date, or not one we can read: answer for today
            return self.views[-1]
        chosen = None
        for view in self.views:
            if view.taken_on <= day:
                chosen = view
        return chosen or self.views[0]            # a date before the first download: the first download


def _tables_of(version) -> dict:
    """The tables of one page version, keyed by their content (row names and values)."""
    groups: dict[int, list] = defaultdict(list)
    for r in version.tables:
        if r.kind == "table":
            groups[r.table].append(r)
    return {tuple((r.name.lower(), r.signature()) for r in rows): rows for rows in groups.values()}


def _to_date(s) -> date:
    if isinstance(s, date):
        return s
    return date.fromisoformat(str(s)[:10])


# The ways a question may write its date, each matched against the whole of the cleaned,
# lower-case text. Whatever follows a date written year first (a time, a time zone) is ignored.
# Without a day, the month's last day is meant ("as of May 2026" is 2026-05-31); without a
# month, the year's last day.
_AS_OF_FORMS = [re.compile(p) for p in (
    r"(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})(?:\D.*)?",   # 2026-05-31, 2026/05/31, 2026.05.31
    r"(?P<y>\d{4})(?:-(?P<m>\d{1,2}))?",                              # 2026-05, 2026
    r"(?P<m>\d{1,2})/(?:(?P<d>\d{1,2})/)?(?P<y>\d{4})",               # 05/31/2026, 05/2026: month first, the US order
    r"(?P<m>[a-z]+)\.?(?: (?P<d>\d{1,2}),?)? (?P<y>\d{4})",           # May 2026, May 31, 2026, May 31 2026
    r"(?P<d>\d{1,2}) (?P<m>[a-z]+)\.? (?P<y>\d{4})",                  # 31 May 2026
)]


def as_of_day(as_of) -> Optional[date]:
    """The day a question's `as_of` names, or None when it names none that can be read.

    The contract asks for an ISO date, but a question may write it another way, and a date that
    is not read is answered from the newest download: for a past date, the wrong one. A wording
    not listed above, an impossible date and a value that is neither text nor a date give None.
    """
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    if not isinstance(as_of, str):
        return None
    text = re.sub(r"^as of ", "", clean(as_of).lower())
    for form in _AS_OF_FORMS:
        found = form.fullmatch(text)
        if found:
            parts = found.groupdict()
            try:
                year, month = int(parts["y"]), parts["m"] or "12"
                month = int(month) if month.isdigit() else MONTHS.get(month, 0)
                return date(year, month, int(parts.get("d") or calendar.monthrange(year, month)[1]))
            except ValueError:                    # month 13, February 30, a word that is not a month
                return None
    return None
