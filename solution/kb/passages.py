"""Find the text that answers a question that is not a table lookup.

Each page is cut into passages (a heading and the paragraphs under it), ranked by BM25 and then by
who is speaking (a product page outranks a campaign page, an article or a calculator) and when
(among dated articles the newer one wins, so a 2021 press release does not answer for today).
Identical passages on many pages (footers, every press release's "About us") are collapsed to the
copy from the newest and most trusted page. It also finds the site's links, a passage's footnotes
and the contact paragraph. Wording an answer is not done here (kb.quote, kb.reader). The search
leaves out the "Apr" of a date (kb.lookup) and adds the postal code of a state named in full (kb.hours).
"""
from __future__ import annotations

import difflib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Optional

from . import hours
from . import pages as P
from .lookup import without_april
from .text import HEADING, IMAGE, RELATIVE_YEAR, SITE_NAMES, find_dates, strip_markdown, tokens, year_offset

_FILLER = frozenset("right now currently current today present latest please tell give".split())
_JUNK = re.compile(r"^(!\[|\* \* \*$|✕$|\[Continue\]|Cancel$|data:image|Loading|\d+ Results?$)")
_BYLINE = re.compile(r"(?:written by\b|\d+ min(?:ute)? read\b)", re.I)     # "1 min read Jane Doe July 9, 2025"
# A label and what follows it: "Eligibility: members only".
_LABEL_AND_VALUE = re.compile(r"^[^:]{2,40}:\s+(\S.*)$")
# What follows the colon of a title can ask instead of answering, with or without the question mark:
# "Stall or table: does it matter?", "Choosing the pitches: how we decide".
_ASKS = re.compile(r"\?|^(?:what|how|why|when|where|which|who|whose)\b", re.I)
# A link that leads to another page. A number to call or an address to write to is not one: its
# words are the fact itself.
_PAGE_LINK = re.compile(r"\[[^\]]*\]\((?!\s*(?:tel|mailto|sms):)[^)]*\)", re.I)
MAX_CHARS = 900
COUNTING = re.compile(r"\bhow many\b|\bnumber of\b", re.I)      # a question that asks for a count
# "... earns the same dividends as a basic share account": a rate stated by pointing to another product's.
SAME_RATE = r"\bsame (?:[a-z]+ ){0,2}(?:rates?|dividends?|apy|apr)"
EVIDENCE_K = 14            # passages a search returns for a question; the reader shows them all, then adds more


@dataclass(eq=False)
class Passage:
    """A heading and the text under it, the unit a search returns; compared by identity."""
    page: object
    heading: str
    text: str            # plain text (links reduced to their words)
    raw: str             # markdown, links kept: needed when the answer is a link
    order: int
    toks: tuple = ()
    leaf: str = ""       # the nearest heading on its own ("Recent news - <headline>" -> "<headline>")


def split(page) -> list[Passage]:
    """Cut a page into passages of at most MAX_CHARS, never across a heading.

    A heading is normally the label of the lines under it. One that states something and that no
    passage carries, neither under it nor in a label, would be lost ("Application period: March 3 -
    April 14, 2027" with the next heading right under it): it becomes a passage of its own, labelled
    by the headings above it. One that is only a link to another page is not this page's to state."""
    out: list[Passage] = []
    heading = leaf = page.title
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    # Headings on the stack that state something and that no passage has carried yet, by their
    # depth in the stack: (their place among the passages, the label above them, text, markdown).
    unsaid: dict[int, tuple] = {}

    def flush() -> None:
        if not buf:
            return
        raw = "\n".join(buf).strip()
        buf.clear()
        text = strip_markdown(raw.replace("\n", " "))
        if _thin(text):
            return
        out.append(Passage(page=page, heading=heading, text=text, raw=raw, order=len(out), leaf=leaf))
        for depth in (len(stack) - 1, len(stack) - 2):      # its heading, and the one above when in its label
            if depth in unsaid and stack[depth][1] in heading:
                del unsaid[depth]

    def keep_unsaid(depth: int) -> None:
        """The headings leaving the stack from `depth` on that no passage carried, each put in at
        its own place: the deepest first, so that the places of the ones above it do not move."""
        for d in sorted((d for d in unsaid if d >= depth), reverse=True):
            at, label, text, raw = unsaid.pop(d)
            out.insert(at, Passage(page=page, heading=label, text=text, raw=raw, order=at, leaf=text))

    size = 0
    for line in page.markdown.split("\n"):
        s = line.strip()
        if not s or _JUNK.match(s) or re.search(r"\S{1500,}", s):      # empty, decoration, or an embedded blob
            continue
        h = HEADING.match(s)
        if h:
            flush()
            size = 0
            text = strip_markdown(h.group(2))
            if text:
                # A name in one heading and a title in the next ("Rosa Lindqvist" / "Chair of the Board")
                # belong together: keep the heading above when this one is nested under it.
                level = len(h.group(1))
                above = [x for x in stack if x[0] < level]
                keep_unsaid(len(above))                   # the headings this one closes
                if _states(text) and not _thin(text) and not _only_a_link(h.group(2)):
                    unsaid[len(above)] = (len(out), _label(above, page.title), text, h.group(2).strip())
                stack = above + [(level, text)]
                leaf = text
                heading = _label(stack, page.title)
            continue
        for piece in _pieces(s):
            if size + len(piece) > MAX_CHARS and buf:
                flush()
                size = 0
            buf.append(piece)
            size += len(piece)
    flush()
    keep_unsaid(0)
    for n, p in enumerate(out):                       # number them again: kept headings were put in between
        p.order = n
    return out


def _label(stack: list, default: str) -> str:
    """The label of what stands under the last heading of `stack`: that heading, after the one
    above it when that one is short."""
    if not stack:
        return default
    if len(stack) > 1 and len(stack[-2][1]) < 60:
        return " - ".join(t for _, t in stack[-2:])
    return stack[-1][1]


def _states(text: str) -> bool:
    """A heading that states something instead of naming a section: it carries a figure
    ("Application period: March 3 - April 14, 2027") or gives a value after a label ("Eligibility:
    members only"). A value is at most three words and answers: after the colon of a title comes
    more title ("Selling at night: a season of summer evenings by the river") or a question."""
    if re.search(r"\d", text):
        return True
    m = _LABEL_AND_VALUE.match(text)
    return bool(m) and len(m.group(1).split()) <= 3 and not _ASKS.search(m.group(1))


def _only_a_link(raw: str) -> bool:
    """A heading that is nothing but a link to another page: a headline under "Recent news". What
    it announces is the article's to state, and the article may not even be in the download."""
    rest = IMAGE.sub(" ", raw)                # images first: a link can wrap one
    return not re.search(r"[A-Za-z0-9]", _PAGE_LINK.sub(" ", rest))


def _thin(text: str) -> bool:
    """Too little to stand as a passage: fewer than 20 letters and digits, or only a byline."""
    return len(re.sub(r"[^A-Za-z0-9]", "", text)) < 20 or (len(text) < 90 and bool(_BYLINE.match(text)))


def _pieces(line: str) -> list[str]:
    """A very long line (a page of a PDF, a paragraph of fine print) is cut at sentence ends so
    that no part of it is out of reach. Lines that are long only because of their links stay whole."""
    if len(line) <= 1.5 * MAX_CHARS or len(strip_markdown(line)) <= 1.5 * MAX_CHARS:
        return [line]
    out, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'$])", line):
        if cur and len(cur) + len(sentence) > MAX_CHARS:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    return out + [cur] if cur else out


def _recency(page, today: date) -> float:
    """Articles fade with age; everything else speaks in the present."""
    if page.kind != P.ARTICLE:
        return 1.0
    if not page.published:
        return 0.8
    years = max(0.0, (today - page.published).days / 365.25)
    return max(0.55, 1.0 - 0.09 * years)


_KIND_WEIGHT = {P.OFFICIAL: 1.0, P.DOCUMENT: 0.9, P.LANDING: 0.85, P.ARTICLE: 0.85, P.CALCULATOR: 0.6, P.INDEX: 0.45}


# What opening hours are written with: "9 a.m.", "5:30 pm", "noon". Not a time to the second, which dates
# an article ("Mar 3, 2024 10:41:07 AM").
CLOCK_TIME = re.compile(r"(?<![\d:])\d{1,2}(?::\d\d)?(?!:\d)\s?[ap]\.?m\b|\bnoon\b", re.I)
# What kind of thing would answer the question, and what that looks like in text. A passage that
# contains the right kind of thing (a clock time for "what are the hours") is a better bet than
# one that only shares the question's words.
_ANSWER_TYPES = [   # (the question asks for, a passage that could answer contains, required?)
    (re.compile(r"^(?!.*\b(?:how many|volunteer\w*|per week|a week)\b).*(?:\bhours?\b|\bopen(?:s|ing)?\b (?:on|at|until|till|late|early|today|tomorrow|sat|sun|week)"
                r"|\bclos(?:e|es|ing) (?:at|on|early)|what time)", re.I | re.S),
     CLOCK_TIME, True),
    (re.compile(r"\bphone\b|\btelephone\b|number (?:to|for)\b|\bhow (?:do|can) (?:i|we) call\b", re.I),
     re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"), True),
    (re.compile(r"\baddress\b|\bwhere is\b|\bwhere's\b|\blocated\b|\blocation of\b", re.I),
     re.compile(r"\b\d{2,5} [A-Z0-9][\w.]*(?: [A-Z][\w.]*){0,3},? .{0,30}\b[A-Z]{2},? \d{5}\b"), False),
    (re.compile(r"\brates?\b|\bapr\b|\bapy\b|\bdividend\b|\binterest\b|\byield\b|\bpercent", re.I),
     re.compile(r"\d\s?%"), False),
    (re.compile(r"how much|\bcost\b|\bfees?\b|\bprice\b|\bcharge\b|\bamount\b|\bgrant\b|\blimit\b|\bup to\b", re.I),
     re.compile(r"\$\s?\d"), False),
    (re.compile(r"\bwhen\b|what date|\bdeadline\b|\bby when\b|what day", re.I),
     re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2},? (?:19|20)\d\d\b", re.I), False),
]


def answer_shapes(question: str) -> list:
    """(pattern, required) for each kind of answer the question asks for."""
    return [(shape, required) for asks, shape, required in _ANSWER_TYPES if asks.search(question)]


def _has_count(text: str, query_tokens: list[str]) -> bool:
    """True when a number sits next to one of the question's words: '31 service centers',
    'assets have grown to $1.2 billion'."""
    want = set(query_tokens)
    text = re.sub(r"[|:]+", " ", text)                 # "Members: 57,409" in a table cell
    for m in re.finditer(r"((?:[A-Za-z-]+\s+){0,6})\$?\d[\d,.]*\+?\s+((?:[A-Za-z-]+\s+){0,2}[A-Za-z-]+)", text):
        if set(tokens(m.group(2))) & want or set(tokens(m.group(1))) & want:
            return True
    return False


@dataclass(eq=False)
class Link:
    """A link on a page, with the words that can find it; compared by identity."""
    text: str            # the words the site puts on the link
    url: str
    heading: str         # the section it appears in
    page: object
    toks: frozenset = frozenset()


_LINK = re.compile(r"(?<!!)\[([^\]]{2,120})\]\((https?://[^)\s]+)\)")


def _links_of(page) -> list[Link]:
    """Every link on a page to another page (not to an image), with its section and its line's words."""
    out, heading = [], page.title
    for line in page.markdown.split("\n"):
        h = HEADING.match(line.strip())
        if h:
            heading = strip_markdown(h.group(2)) or heading
            continue
        for m in _LINK.finditer(line):
            text, url = strip_markdown(m.group(1)), m.group(2)
            if re.search(r"\.(png|jpe?g|svg|gif|webp)(\?|$)", url, re.I) or not re.search(r"[A-Za-z]{2}", text):
                continue
            context = strip_markdown(_LINK.sub(lambda x: x.group(1), line))[:200]
            path = re.sub(r"https?://[^/]+", "", url).replace("-", " ").replace("/", " ").replace("#", " ")
            out.append(Link(text, url, heading, page, frozenset(tokens(f"{text} {heading} {context} {path}"))))
    return out


# A footnote mark glued to a word or a value ("account1.", "APY2", "4.85%1"), and the line it points to
# ("1 One bonus per member ...", "_2APY = Annual Percentage Yield ..."). A numbered list
# item ("1. Open the app") is not a footnote.
_NOTE_MARK = re.compile(r"(?:(?<=[a-z%)])|(?<=AP[RY]))([1-9])(?=[\s.,;:)*_|]|$)")
_NOTE_LINE = re.compile(r"^[\s_*>]*([1-9])\s?(?=[A-Z$\"'])([^|]{60,})$")


def _headline(page) -> str:
    return strip_markdown(P.first_heading(page.markdown))


def _squash(s: str) -> str:
    return re.sub(r"\W+", "", s.lower())


class PassageIndex:
    """BM25 over the passages of one view."""

    K1, B = 1.4, 0.7

    def __init__(self, view) -> None:
        self.view = view
        # One statement, many copies. Passages that are identical once their digits are masked
        # are versions of the same statement (the "About us" paragraph of every press release,
        # a rate table repeated on another page). Only the most trusted, newest version is kept,
        # so a superseded number cannot be retrieved at all.
        best: dict[str, Passage] = {}
        copies: dict[str, int] = defaultdict(int)
        group: dict[str, list] = defaultdict(list)
        self.teasers: set[int] = set()
        # A teaser ("Recent news", then a headline and the first lines of that article) is a copy
        # of another page's opening. The article itself is indexed under its own date, so the
        # teaser is left out: otherwise a 2024 figure turns up dated by the page that links to it.
        headlines: dict[str, set] = defaultdict(set)
        names = {}
        for page in view.all_pages():
            names[page.key] = {n for n in (_squash(page.title), _squash(_headline(page))) if len(n) >= 15}
            if page.kind == P.ARTICLE:
                for name in names[page.key]:
                    headlines[name].add(page.key)
        for page in view.all_pages():
            if not page.passages:
                page.passages = split(page)
            for p in page.passages:
                leaf = _squash(p.leaf)
                # Under another article's headline; or under its own headline but inside a
                # section ("Recent news - <headline>"): two articles can share a headline.
                if (leaf not in names[page.key] or p.heading != p.leaf) and headlines.get(leaf, set()) - {page.key}:
                    self.teasers.add(id(p))
                    continue
                text = _squash(p.text)
                # Long passages: mask the digits and compare the first 500 letters and digits (the
                # "About us" paragraph ends differently from one release to the next). Short ones
                # (a locator entry) are distinct facts.
                key = re.sub(r"\d", "#", text)[:500] if len(text) >= 200 else p.heading.lower() + "|" + text
                copies[key] += 1
                group[key].append((p, text[:500]))
                old = best.get(key)
                if old is None or self._freshness(page) > self._freshness(old.page):
                    best[key] = p
        # Other numbers in the same words, on a page trusted and dated alike, are another fact and not
        # an older version of this one: one desk's phone extension and another desk's.
        for key, win in list(best.items()):
            for p, said in group[key]:
                other = key + "|" + said
                if said != _squash(win.text)[:500] \
                        and self._freshness(p.page)[:2] == self._freshness(win.page)[:2] \
                        and (other not in best or self._freshness(p.page) > self._freshness(best[other].page)):
                    best[other] = p
        self.passages = list(best.values())
        # An article's copy that lost to another page is an older version of the statement (an older
        # press release's "About us" figures): the reader's page walks leave it out too. Other pages
        # keep theirs: the digits masked above can be a page's own (a department's phone extension).
        kept = {id(p) for p in self.passages}
        self.superseded = {id(p) for page in view.all_pages() if page.kind == P.ARTICLE
                           for p in page.passages if id(p) not in kept}
        # How many pages repeat each statement. The organisation's standing facts ("... members,
        # ... service centers") are the ones every press release repeats.
        self.repeated = {id(p): copies[k] for k, p in best.items() if copies[k] > 1}
        self.lead = {}
        for p in self.passages:
            if p.page.kind == P.ARTICLE and len(p.text) > 200:
                cur = self.lead.get(p.page.key)
                if cur is None or p.order < cur.order:
                    self.lead[p.page.key] = p
        self.df: dict[str, int] = defaultdict(int)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        total = 0
        for i, p in enumerate(self.passages):
            if not p.toks:
                p.toks = tuple(tokens(p.heading + " " + p.heading + " " + p.page.title + " " + p.text))
            total += len(p.toks)
            counts: dict[str, int] = defaultdict(int)
            for t in p.toks:
                counts[t] += 1
            for t, c in counts.items():
                self.df[t] += 1
                self.postings[t].append((i, c))
        self.avg = total / max(1, len(self.passages))
        self.n = len(self.passages)
        self._footnotes_by_page: dict[str, dict] = {}         # page key -> {footnote mark: its text}
        self._by_initial = None                          # vocabulary by first letter, built on first use
        # What the whole page is about: its title, headings and opening lines. A passage that says
        # "Our hours are ..." belongs to the Elmwood branch because its page opens "ELMWOOD, Kan.".
        self.about: dict[str, frozenset] = {}
        for page in view.all_pages():
            heads = " ".join(p.heading for p in page.passages[:40])
            opening = " ".join(p.text for p in page.passages[:2])[:700]
            self.about[page.key] = frozenset(tokens(f"{page.title} {heads} {opening}"))
        self.links = [link for page in view.pages.values() if page.kind != P.INDEX for link in _links_of(page)]
        self.live = {page.key.split("?")[0] for page in view.pages.values()}
        hosts = defaultdict(int)
        for key in self.live:
            hosts[key.split("/")[0]] += 1
        self.host = max(hosts, key=hosts.get) if hosts else ""

    def find_links(self, query: str, k: int = 6) -> list[Link]:
        """Links whose words, section and target match the question. A link to a page that is
        not in the download is a dead link: it scores a quarter of a live one. A link found on a
        stale or old page ranks below the same link on a current page."""
        q = [t for t in dict.fromkeys(tokens(query))
             if t not in ("link", "url", "website", "webpage", "page", "site", "web", "address", "visit", "view", "see",
                          "use", "go", "find", "current", "straight", "take") and self.df.get(t, 0) < 0.25 * self.n]
        if not q:
            return []
        best: dict[str, tuple] = {}
        for link in self.links:
            hit = sum(self.idf(t) for t in q if t in link.toks)
            if hit <= 0:
                continue
            target = P.url_key(link.url).split("?")[0]
            dead = target.split("/")[0] == self.host and target not in self.live
            score = hit * self._prior(link.page) * (0.25 if dead else 1.0)
            key = link.url.rstrip("/")
            if key not in best or score > best[key][0]:
                best[key] = (score, link)
        return [link for _, link in sorted(best.values(), key=lambda x: -x[0])[:k]]

    def defining(self, term: str, exclude: list) -> Optional[Passage]:
        """The passage that best says what a term covers: it contains the term followed by a list
        or a definition. Product pages first."""
        rx = re.compile(re.escape(term) + r"s?\b\s*(?::|\bis\b|\bare\b|\bincludes?\b|\bcovers?\b|\bmeans\b|\(|-\s)", re.I)
        found = [p for p in self.passages if rx.search(p.text) and all(p is not x for x in exclude)]
        if not found:
            return None
        return max(found, key=lambda p: (self._prior(p.page), -len(p.text)))

    def contact(self) -> Optional[Passage]:
        """The site's own 'how to reach us' paragraph: on a contact page, with a phone number."""
        if "contact" not in self.view.cache:
            found = None
            for p in self.passages:
                if p.page.kind == P.OFFICIAL and re.search(r"contact", p.page.key.split("/")[-1] + " " + p.page.title, re.I) \
                        and re.search(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b", p.text):
                    if found is None or (len(p.page.key), p.order) < (len(found.page.key), found.order):
                        found = p
            self.view.cache["contact"] = found
        return self.view.cache["contact"]

    def _prior(self, page) -> float:
        w = _KIND_WEIGHT[page.kind] * _recency(page, self.view.taken_on)
        if page.key in self.view.stale:
            w *= 0.8
        if page.key in self.view.gone:
            w *= 0.8
        return w

    def _freshness(self, page) -> tuple:
        return (round(self._prior(page), 3), str(page.published or ""), page.fetched)

    def idf(self, t: str) -> float:
        """BM25's weight for a word: high for a rare one, near zero for one in every passage."""
        return math.log(1.0 + (self.n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))

    def search(self, query: str, k: int = 8, per_page: int = 2, held_back: Optional[list] = None
               ) -> list[tuple[float, Passage]]:
        """The k best passages, at most `per_page` from one page so that several pages are heard.
        Passages that lost their place only to that cap are appended to `held_back` when given."""
        q = self.query_tokens(query)
        scores: dict[int, float] = defaultdict(float)
        for t in q:
            if t not in self.postings:
                continue
            idf = self.idf(t)
            for i, c in self.postings[t]:
                n = len(self.passages[i].toks)
                scores[i] += idf * c * (self.K1 + 1) / (c + self.K1 * (1 - self.B + self.B * n / self.avg))
        # A passage that covers more of the question beats one that repeats a single word of it.
        weight = {t: self.idf(t) for t in q if t in self.postings}
        total = sum(weight.values()) or 1.0
        covered: dict[int, float] = defaultdict(float)
        for t, w in weight.items():
            for i, _ in self.postings[t]:
                covered[i] += w
        for i in scores:                                 # words the page is about count half
            p = self.passages[i]
            about = self.about.get(p.page.key, frozenset())
            have = set(p.toks)
            for t, w in weight.items():
                if t in about and t not in have:
                    covered[i] += 0.5 * w
                    scores[i] += 0.5 * w
        counting = COUNTING.search(query.lower())
        shapes = answer_shapes(query)
        # A member asking about "checking" means their own account: pages for businesses step back
        # unless the question is about a business.
        personal = 1.0 if "business" in q else 0.85
        first = sorted(((s * self._prior(self.passages[i].page) * (0.3 + 0.7 * min(1.0, covered[i] / total))
                         * (personal if P.for_business(self.passages[i].page.key) else 1.0), i)
                        for i, s in scores.items()), reverse=True)
        ranked = []
        for s, i in first[:400]:                         # the answer-shape checks are regex work: best 400 only
            p = self.passages[i]
            if counting:
                s *= 1.6 if _has_count(p.text, q) else 0.5
                s *= 1.0 + 0.35 * math.log(self.repeated.get(id(p), 1))
                if self._dated_recently(p.text):
                    s *= 2.5                             # a figure that carries its own recent "as of" date
            for shape, required in shapes:
                if shape.search(p.text):
                    s *= 1.5
                elif required:
                    s *= 0.4              # an hours question cannot be answered by text with no clock time
            ranked.append((s, i))
        ranked.sort(reverse=True)
        out, per = [], defaultdict(int)
        for s, i in ranked:
            p = self.passages[i]
            if per[p.page.key] >= per_page:
                if held_back is not None:
                    held_back.append((s, p))
                continue
            per[p.page.key] += 1
            out.append((s, p))
            if len(out) >= k:
                break
        return out

    def query_tokens(self, query: str) -> list[str]:
        """The question's words as the index knows them: filler dropped, spelling and endings matched.
        A year in words ("last year", "a year ago") also brings the year it means on the date of this
        download, as the pages print it ("In 2025, ..."): an "as of" question about an older download
        asks for that download's years. A state's name may bring its postal code (see below). "Apr" in a
        date is the month, not the APR."""
        words = [self.spell(self.inflect(t)) for t in tokens(without_april(query))
                 if t not in _FILLER and t not in SITE_NAMES]
        words += [str(self.view.taken_on.year + year_offset(m.group())) for m in RELATIVE_YEAR.finditer(query)]
        # A state named in full ("any ATM in North Dakota?") also brings the postal code an address
        # gives ("Fargo, ND"), but only for a state the site seldom names: where the name is everywhere
        # (its home state) the code is everywhere too, and would only dilute the other words.
        # tokens() leaves out a code that is a common word ("OR", "IN").
        words += [t for code, name in hours.states_by_name(query).items() for t in tokens(code)
                  if min(self.df.get(w, 0) for w in name) < 0.01 * self.n]
        return list(dict.fromkeys(words))

    def more_from(self, page, query: str, n: int = 2) -> list[Passage]:
        """The passages of one page that cover most of the question, best first."""
        weight = {t: self.idf(t) for t in self.query_tokens(query) if t in self.df}
        scored = []
        for p in page.passages:
            if id(p) in self.teasers or id(p) in self.superseded:
                continue
            have = set(tokens(p.heading + " " + p.text))
            score = sum(w for t, w in weight.items() if t in have)
            if score > 0:
                scored.append((score, -p.order, p))
        scored.sort(key=lambda x: x[:2], reverse=True)
        return [p for _, _, p in scored[:n]]

    def footnotes(self, p: Passage) -> list[str]:
        """The fine print a passage points to: "get $25 when you open a youth account1" has its
        limits in the line that starts "1 One bonus per member ...", lower on the page."""
        marks = set(_NOTE_MARK.findall(p.raw))
        if not marks:
            return []
        notes = self._footnotes_by_page.get(p.page.key)
        if notes is None:
            notes = self._footnotes_by_page[p.page.key] = {}
            for line in p.page.markdown.split("\n"):
                for part in re.split(r"(?<=[.!?])_\s+_(?=\d)", line.strip()):     # "_1 ..._ _2 ..._" on one line
                    m = _NOTE_LINE.match(part)
                    if m:                                 # the last such line wins: fine print sits at the bottom
                        notes[m.group(1)] = strip_markdown(m.group(2))[:420]
        return [f"Footnote {n} on that page: {notes[n]}" for n in sorted(marks) if n in notes]

    def _dated_recently(self, text: str) -> bool:
        m = re.search(r"\bas of:?\s+", text, re.I)
        if not m:
            return False
        found = find_dates(text[m.end():m.end() + 30])
        return bool(found) and 0 <= (self.view.taken_on - found[0][1]).days <= 200

    def known(self, token: str) -> bool:
        """The word appears in some passage of this view."""
        return token in self.df

    def spell(self, t: str) -> str:
        """A question word the site never uses, one slip from a word it does: 'mortage' -> 'mortgage'."""
        if t in self.df or len(t) < 5 or t.isdigit():
            return t
        if self._by_initial is None:
            self._by_initial = defaultdict(list)
            for w, n in self.df.items():
                if n >= 3 and len(w) >= 5 and w.isalpha():
                    self._by_initial[w[0]].append(w)
        close = difflib.get_close_matches(t, [w for w in self._by_initial[t[0]] if abs(len(w) - len(t)) <= 2], n=1, cutoff=0.84)
        return close[0] if close else t

    def inflect(self, t: str) -> str:
        """'chairing' -> 'chair', 'closed' -> 'close' when the plainer form is the common one here."""
        for suffix, add in (("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("d", "")):
            if len(t) > len(suffix) + 3 and t.endswith(suffix):
                base = t[:-len(suffix)] + add
                if self.df.get(base, 0) > 3 * self.df.get(t, 0):
                    return base
        return t


def add_contact(pages: list, passages: PassageIndex) -> None:
    """Add the page of the site's contact paragraph to `pages`, unless it is there already."""
    contact = passages.contact()
    if contact is not None and contact.page not in pages:
        pages.append(contact.page)
