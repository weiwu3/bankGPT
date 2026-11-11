"""One downloaded page at one point in time, and what kind of page it is.

The kind decides how much a page is trusted when pages disagree (AUTHORITY): a product or rate
page (official) states the current value; a published file such as a fee schedule (document)
comes next; a campaign page (landing) was accurate when written; an article was true on its
publication date; a calculator's numbers illustrate; an index only lists links. Which page wins
a disagreement is decided elsewhere (kb.lookup, kb.passages).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from urllib.parse import urlparse

from .text import clean, find_dates

OFFICIAL, DOCUMENT, LANDING, ARTICLE, CALCULATOR, INDEX = "official", "document", "landing", "article", "calculator", "index"

# Trust when two pages state different values for the same thing. Higher wins.
AUTHORITY = {OFFICIAL: 4, DOCUMENT: 3, LANDING: 2, ARTICLE: 1, CALCULATOR: 1, INDEX: 0}

_ARTICLE_PATH = re.compile(r"/(blog|news|newsroom|press|press-releases?|articles?|stories|posts?|resources)/[^/]+", re.I)
_DATED_PATH = re.compile(r"/(19|20)\d\d/\d{1,2}/")
_INDEX_PATH = re.compile(r"/(tag|tags|author|authors|category|categories|topic|topics)/", re.I)
_DOC_PATH = re.compile(r"\.(pdf|xlsx?|docx?|pptx?|csv)$", re.I)
_LANDING_PATH = re.compile(r"(-lp$|-lp/|landing|promo|offer|campaign|giveaway|sweepstakes)", re.I)
_FIRST_HEADING = re.compile(r"^#\s+(.+)$", re.M)


def url_key(url: str) -> str:
    """Same page, same key: host without www, path without trailing slash, no fragment."""
    parts = urlparse(url if "//" in url else "https://" + url)
    key = parts.netloc.lower().removeprefix("www.") + parts.path.rstrip("/")
    return key + ("?" + parts.query if parts.query else "")


def classify(url: str, markdown: str) -> str:
    """The page's kind (OFFICIAL, DOCUMENT, ...), from its address and the top of its text."""
    path = urlparse(url).path
    head = markdown[:400]
    # /hubfs/ is where a common hosted CMS keeps uploaded files, whatever their extension.
    if _DOC_PATH.search(path) or "/hubfs/" in path:
        return DOCUMENT
    if "calculator" in path.lower():
        return CALCULATOR
    if _INDEX_PATH.search(path) or "category=" in url:
        return INDEX
    if _ARTICLE_PATH.search(path) or _DATED_PATH.search(path) or re.search(r"\b\d+ min read\b", head):
        return ARTICLE
    if _LANDING_PATH.search(path):
        return LANDING
    return OFFICIAL


def published_on(url: str, markdown: str) -> Optional[date]:
    """Publication date of an article: the first date near the top, else a /YYYY/MM/DD/ URL."""
    found = find_dates(markdown[:700])
    if found:
        return found[0][1]
    m = re.search(r"/((?:19|20)\d\d)/(\d{1,2})/(\d{1,2})/", url)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


@dataclass(eq=False)
class PageVersion:
    """One page's text as it was in one or more consecutive downloads; compared by identity."""
    url: str
    key: str
    title: str
    markdown: str
    fetched: str                      # ISO timestamp of the download
    first_seen: int                   # index of the first download that has this exact text
    last_seen: int                    # index of the last download that has it
    kind: str = OFFICIAL
    published: Optional[date] = None  # articles only
    tables: list = field(default_factory=list)
    passages: list = field(default_factory=list)

    def label(self) -> str:
        """Short human description used in reasons: 'calculator page', 'article published 2026-03-12'."""
        if self.kind == ARTICLE and self.published:
            return f"article published {self.published.isoformat()}"
        return {OFFICIAL: "product page", DOCUMENT: "published document", LANDING: "campaign page",
                ARTICLE: "article", CALCULATOR: "calculator page", INDEX: "index page"}[self.kind]


def first_heading(markdown: str) -> str:
    """The page's first top-level heading as written, or ''."""
    m = _FIRST_HEADING.search(markdown)
    return m.group(1) if m else ""


def page_title(title: str, markdown: str) -> str:
    """The page's own name without the site suffix ('Auto Loans | Example Credit Union' -> 'Auto Loans')."""
    t = clean(title or "")
    t = re.split(r"\s[|\-–]\s", t)[0] if t else ""
    if not t:
        t = clean(first_heading(markdown))
    return t


def for_business(key: str) -> bool:
    """A page for businesses, by its address: '/business/wren-checking' or '/business-loans'."""
    return bool(re.search(r"/business(?:/|-|$)", key))
