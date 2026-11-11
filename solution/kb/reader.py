"""The reader: the code that has the model word an answer from evidence the rules selected, and
checks its reply.

The model never decides which table is current. By the time it is called, the rows have been
resolved to the winning copy, copies of one statement have been reduced to the newest, and the
passages are laid out in the order they are trusted: pages that state the present, then
articles from the newest to the oldest. Its job is reading: pick the tier that was asked for,
combine a few passages, quote a link, or say the website does not state it.

BANKGPT_DEBUG=1 prints each raw reply to stderr; dev/outage.py counts them.
"""
from __future__ import annotations

import os
import re
import sys
from typing import NamedTuple, Optional

from . import pages as P
from .compose import in_order, line
from .lookup import Match, Query, RowIndex, asks_rate, nearby_rows, parse_question
from .passages import EVIDENCE_K, MAX_CHARS, SAME_RATE, PassageIndex, add_contact
from .reply import Reply, dated, figures, refusal
from .text import IMAGE, RELATIVE_YEAR, clean, content_tokens, digits_beside_words, year_offset
from .transport import AnthropicBackend

SYSTEM = """You answer members' questions for a credit union. Your only source is the evidence in \
the message: excerpts of the credit union's own website as it was downloaded on the date given.

How the evidence was prepared, so you know what to trust:
- "Published values" are rows from the site's rate and fee tables. Out-of-date copies have \
already been removed: every row listed is in force. Each row is one tier (a term, a balance or \
loan-amount range, a model-year range) of one product. Rows marked [business] are business products.
- "Passages" are tagged with the kind of page and its date. Product pages and documents come \
first, then articles from the newest to the oldest. A product page states the present. An \
article or press release was true on its publication date: when two disagree, the newer one is \
right, and a product page beats both, however many older articles or documents repeat the other \
value. Each passage's "Section:" line names the section it sits under; that heading is part of \
what the page says.
- "Links" are links the site itself gives, with the words it puts on them. "General contact" is \
the site's own contact paragraph: its address, phone number and ways to reach it.

Rules:
1. Use only the evidence. Never add a figure, name, date, link or fact from memory.
2. Answer exactly what was asked: the term, tier, balance, model year, product and branch named. \
If the question names no tier, give every tier of that product briefly, first-listed first. Keep \
personal and business products apart unless the question is about everything on offer.
3. The most recent statement is the current one. State it plainly, with its date when that \
helps. Do not hedge or decline because it was published some time ago, and never mention a \
superseded value, not even to compare. A newer statement replaces an older one only \
when both are about the same thing: an article about one branch, product or programme says \
nothing about another. For "last year", "this year", "latest" or "most recent", work from \
today's date ("last year" is the calendar year before today's): use the figure for that period \
if any passage gives it, however briefly, even when another passage dwells on an earlier year. \
A figure reported for a named year (a year's count or total) is not superseded by a later \
year's: it stays the figure for its year. So when a question names a past year (or says last \
year, the past year or a year ago) that no passage gives a count or total for, say so and give \
the two most recent years' figures, each with its year and nothing more, as an answer, not a \
decline: "No count is published for 2025. The latest are 41 in 2024 and 37 in 2023." A standing fact (how many members \
or branches, a total since some year, a rate in force, a date to come) is not a year's figure: \
there the newest value replaces the older ones.
4. Be specific. Keep the evidence's wording for the fact itself (if it says "the third time", \
write "the third time", not "three times"), and copy figures, names, places, dates, phone numbers \
and links exactly as written. If the evidence says what a term in your \
answer covers (which area, which accounts, which branches), say that too. When asked for a link \
or how to do something online, give one full URL: the first entry under "Links" that fits (they \
are ordered best first), in preference to the address of the page a passage came from.
5. Never ask the member to clarify. If the question is general (no branch, tier or product \
named), answer for what the evidence covers. When a fact differs by branch (hours, phone, \
services), give each distinct schedule or detail once, say which branch it belongs to, and say \
it varies by branch.
6. Be direct: one to three short sentences, no preamble, nothing about "the evidence" or "passages".

Choose one verdict:
- answer: the evidence states what was asked, for the thing asked about or, when the question \
is general, for the cases it covers. A yes-or-no question ("do you have...", "is there...", \
"does any...") is answered yes or no: when the evidence shows what is on offer and it does not \
include what was asked about, say no and state what is listed.
- decline: the question asks for a fact (a rate, fee, amount, hours, address, date, count, name) \
about some thing, and the evidence has no such fact about that thing. That includes a product \
that is described but has no published rate, and a named branch, term, tier or product that \
does not appear in the evidence. Never substitute a different product, tier or branch, and \
never guess. But when the evidence does speak to the question for the thing asked about, the \
verdict is answer even if it is not in the exact form asked: in other words (the question says \
"credit unions", the evidence says "participating branches"); under another reading (asked \
when a branch closes, the evidence gives the date it closed for good); or for an earlier period \
(asked about "the most recent round" when the newest round is not finished: give the newest \
completed one, with its period; asked for a past year's count or total that no passage gives for \
that year: give the two latest years' figures, as rule 3 says).
- contact: the member asks how to get something done with their own account, and the evidence \
has no specific steps for it. Give the contact route from "General contact", and the online \
route the evidence links to, if any, without dwelling on the steps it lacks.

Reply in exactly this form and nothing else:
QUOTES: the one or two sentences that decide the answer, the shortest that do, each copied word \
for word from a passage and followed by its passage number in brackets; or "none"
VERDICT: answer, decline or contact
SOURCES: passage numbers used, comma separated, or none
ANSWER: the answer for the member (leave empty when declining)"""


# The Unicode spaces clean() leaves alone (it changes only the no-break ones). A model may put any of
# them where a plain space belongs, so the reply's become plain spaces; clean() itself, which every
# page goes through, stays as it is.
_SPACES = re.compile(r"[\u2000-\u200a\u205f\u3000]")
# "Can you tell me the rate ..." and "Do you know the rate ..." ask for a fact, as "What is the
# rate ..." does, and are declined like it when the site does not state it; "... tell me if (or
# whether) ..." still asks yes or no.
_YES_NO = re.compile(r"^\W*(?!(?:can|could|would|will) you (?:please )?"
                     r"(?:tell|give|show|share|list|provide|send|repeat|let me know)\b(?!(?: me| us)? (?:if|whether)\b)"
                     r"|do you know\b(?! (?:if|whether)\b))"
                     r"(do|does|did|is|are|was|were|can|could|will|would|has|have|had|should|may)\b", re.I)
# No \b before "n't": there is none inside "isn't", "wasn't" or "aren't". "Does not include",
# "provide" or "give" says as often what a product lacks ("it does not include a debit card") as
# what the site leaves out, so it counts only after a word for the source earlier in the sentence
# ("the evidence ... does not provide its lobby hours"). The span is bounded so the scan stays linear.
_NOT_STATED = re.compile(
    r"(?:\bnot|\bno|n't)\b(?: \w+){0,3} (?:listed|stated|mentioned|specified|published|provided|given|shown|disclosed)\b"
    r"|\b(?:does not|doesn't|do not|don't) (?:state|say|specify|list|mention|publish)\b"
    r"|\b(?:evidence|information|site|website|page|table|passage|source|document)s?\b[^.;!?]{0,150}?"
    r"\b(?:does not|doesn't|do not|don't) (?:give|provide|include)\b"
    r"|\bno (?:information|mention|details?)\b|\b(?:cannot|can't|unable to) (?:confirm|determine|find)\b", re.I)
# "The special is no longer listed" says the thing asked about was discontinued, which answers the
# question when the answer goes on to say what there is now; it does not say the site leaves the
# question open. So the phrase is set aside before the check above, but only in an answer that
# gives something: a figure the question does not already have (the test a refusal is held to: it
# may repeat the question's numbers and no others). "Its fee is no longer listed" on its own states
# nothing and stays a refusal, and anything else in the answer that _NOT_STATED finds still makes
# it one.
_DISCONTINUED = re.compile(r"\bno longer (?:be |being )?(?:listed|offered|available)\b", re.I)
# "No count is published for 2025. The latest are 41 in 2024 and 37 in 2023." (SYSTEM rule 3) says
# the site has no figure for the year asked, and answers with the years it has: see _by_period.
# A year a question asks about, in its own words: "2025", or a phrase such as "the past year" or
# "a year ago" (text.RELATIVE_YEAR). A year after "since" starts a total, which is a standing fact.
_ASKED_YEAR = re.compile(rf"\b(since )?(?:(20\d\d)\b|({RELATIVE_YEAR.pattern}))", re.I)
_YEAR_IN_FIGURES = re.compile(r"\b20\d\d\b")
# A date ("held March 3, 2025") or a rank ("its 12th meeting") is no year's count or total.
_DATE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{1,2}\b|\b\d{4}-\d\d-\d\d\b"
                   r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d+(?:st|nd|rd|th)\b", re.I)
# "No count of members is published for 2025" (more words before the verb than _NOT_STATED allows)
# and "There is no count for 2025" (or "for the past year": _by_period checks the year it names).
_NO_FIGURE = re.compile(r"\bno\b[^.!?;]{0,60}?\b(?:is|are|was|were|has been|have been) (?:published|listed|given|reported|stated)\b"
                        r"|\bno (?:\w+ ){0,3}for\b", re.I)
# A how-to is shown the site's contact paragraph and its links. "I'd like to know ..." (or "find out")
# asks for a fact, as "Can you tell me ..." does: it is a how-to only when what it wants to know is
# one: "... how to ...", "... where I can ...", "... who to call", "... what to do", "... what number
# to call", "... the best way to ...".
_HOW_TO_KNOW = (r"(?:how|where|who|what) (?:i|do|can|could|should|would|to)\b|what \w+ to\b|what steps\b|"
                r"the (?:best |easiest |right |quickest )?(?:way|process|procedure) (?:to|for)\b")
_HOW_TO = re.compile(r"^\W*(how (do|can|could|should|would|to)\b|can i\b|could i\b|may i\b|where (do|can|should) i\b|"
                    r"what do i (do|need)|what should i do|who (do|should|can) i\b|"
                    rf"i(?: need| want| would like|'d like) to\b(?! (?:know|find out)\b(?!,? (?:{_HOW_TO_KNOW})))|"
                    r"help me\b|is there a way|what(?:'s| is|s) the (?:best |easiest |right |quickest )?(?:way|process|procedure) (?:to|for)\b|"
                    r"what steps)", re.I)


_SAME_RATE_AS = re.compile(SAME_RATE + r" as (?:our |the |an? )?((?:[a-z'-]+ ?){1,4})", re.I)


def _page_tag(page, view, newest=None) -> str:
    if page.kind == P.ARTICLE:
        tag = f"article published {page.published.isoformat()}" if page.published else "article, undated"
        if newest is not None and page.published == newest:
            tag += " (the newest article here)"
    else:
        tag = page.label() + (" (illustrative figures)" if page.kind == P.CALCULATOR else "")
        year = re.findall(r"(?<!\d)(20\d\d)(?!\d)", page.key.split("/")[-1]) if page.kind == P.DOCUMENT else []
        if year:
            tag += f" ({year[-1]} in its file name)"
    if page.key in view.gone:
        tag += ", no longer on the site"
    return tag


def passage_text(p) -> str:
    """A passage as the model is shown it: its markdown with links, without images or embedded data;
    only a very long one is cut."""
    raw = IMAGE.sub("", p.raw)
    raw = re.sub(r"\(data:[^)]*\)", "()", raw)
    raw = re.sub(r"[ \t]+", " ", raw).strip()
    # split() keeps a line whole while its text is at most 1.5 * MAX_CHARS (a paragraph of fine
    # print; longer in markdown when it has links): show all of it. Only a longer passage, a line
    # with no sentence end to cut it at, is cut.
    return raw if len(p.text) <= 1.5 * MAX_CHARS else raw[:1100]


class Prompt(NamedTuple):
    """What Reader.prompt builds: the message for the model (user), the passages in the order they are
    numbered in it, as (score, passage), the published values shown, as (row, the pages that print it,
    most trusted first), whether the rate tables lack what was asked for, and the site's own words in
    the message (passages, their notes, the contact paragraph): a quote must come from those, not from
    the question or from what the message says itself."""
    user: str
    passages: list
    shown: list
    rate_gap: bool
    site: str


class Reader:
    """Builds the message for the model from the evidence, sends it through `backend` and checks the reply."""

    def __init__(self, backend) -> None:
        self.backend = backend

    def prompt(self, question: str, as_of: Optional[str], view, q: Query, match: Optional[Match], ranked: list,
               rows: RowIndex, held_back=()) -> Prompt:
        """The message for the model, and what reading its reply needs from it (see Prompt)."""
        passages = view.cache["passages"]

        row_lines, shown = _row_lines(q, rows)
        usable, newest = _gather(question, passages, ranked, held_back)
        notes = _notes(usable, passages)

        # Pages that state the present first, the most trusted kind first (product pages, documents,
        # campaign pages, calculators), then articles from the newest to the oldest: the order the
        # evidence is read in is the order it is trusted in.
        def place(item) -> tuple:
            page = item[1].page
            if page.kind != P.ARTICLE:
                return (0, -P.AUTHORITY[page.kind])
            return (1, -page.published.toordinal() if page.published else 0)
        usable.sort(key=place)                        # stable: best match first within each kind
        newest = newest.published if newest is not None else None
        blocks = [f"[{n}] {_page_tag(p.page, view, newest)} | {p.page.url}\nSection: {p.heading}\n{passage_text(p)}"
                  + "".join("\n" + note for note in notes.get(id(p), []))
                  for n, (_, p) in enumerate(usable, 1)]

        how_to = bool(_HOW_TO.search(question))
        link_lines = []
        if q.wants_link or how_to or re.search(r"\bonline\b|\bapply\b|\bsign up\b|\blog ?in\b", question, re.I):
            link_lines = [f'- "{link.text}" -> {link.url}  (on {link.page.url}, section: {link.heading})'
                          for link in passages.find_links(question, 3)]
        about_reaching = re.search(r"\bcontact\b|\baddress\b|\bphone\b|\bmail\b|\bemail\b|\bcall\b", q.text)
        contact = passages.contact() if how_to or about_reaching else None

        missing = _rate_gap(question, q, match, rows, usable, row_lines, shown)
        when = (f"The question asks about {as_of}. The evidence is the website as downloaded on "
                f"{view.taken_on.isoformat()}, the last download on or before that date; treat that date as today."
                if as_of else f"Today is {view.taken_on.isoformat()}, the date of the latest download.")
        user = (f"{when}\n\nQuestion: {question}{missing}\n\n"
                + ("Published values:\n" + "\n".join(row_lines) + "\n\n" if row_lines else "")
                + "Passages:\n" + ("\n\n".join(blocks) if blocks else "(none found)")
                + ("\n\nLinks:\n" + "\n".join(link_lines) if link_lines else "")
                + (f"\n\nGeneral contact ({contact.page.url}):\n{passage_text(contact)}" if contact else ""))
        site = "\n\n".join(blocks) + (f"\n\n{passage_text(contact)}" if contact else "")
        return Prompt(user, usable, shown, bool(missing), site)

    def read(self, question: str, as_of: Optional[str], view, q: Query, match: Optional[Match], ranked: list,
             rows: RowIndex, held_back=()) -> Optional[Reply]:
        """The model's answer, checked against the evidence and the question, or a refusal; None when
        the model gave no reply or one not in the form asked (the caller then quotes the site)."""
        prompt = self.prompt(question, as_of, view, q, match, ranked, rows, held_back)
        raw = self.backend.complete(SYSTEM, prompt.user)
        if os.environ.get("BANKGPT_DEBUG"):
            print(f"--- {question}\n{raw}\n", file=sys.stderr)
        if not raw:
            return None
        # The reply gets the normalisation every page gets (dashes, quotes, spaces), and its other
        # Unicode spaces become plain ones, so the checks below, written with plain apostrophes,
        # hyphens and spaces ("isn't listed"), also catch a curly one and a thin space. Its quotes are
        # compared with the evidence through _plain, which normalises both.
        raw = clean(_SPACES.sub(" ", raw))
        m = re.search(r"VERDICT:\s*(answer|decline|contact)\b.*?SOURCES:\s*([^\n]*)\n+ANSWER:\s*(.*)", raw, re.S | re.I)
        if not m:
            return None
        verdict, cited, text = m.group(1).lower(), m.group(2), m.group(3).strip()
        today = view.taken_on.year
        text = re.sub(r"\*\*|__", "", text)
        quotes = self._verified_quotes(raw, prompt.site)
        text = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", text)             # passage numbers mean nothing to a member
        verdict, text = _settle(verdict, text, question, today, prompt.rate_gap)
        text = re.sub(r"^(?:The|This) evidence (?:shows|states|indicates|says|confirms)(?: that)?\s+", "", text)
        text = re.sub(r"\s+in the evidence\b", "", text)
        text = text[:1].upper() + text[1:]
        if verdict == "decline" or not text:
            # SYSTEM asks for no text with a decline (asking for a sentence on what the site does not
            # state makes the model decline questions it otherwise answers), but the model often gives
            # one, and refusal() says it when it states nothing else. (_settle has already read a
            # decline's text for the latest years' figures.)
            return refusal(question, text)
        pages = _cited_pages(prompt, cited, verdict, text, question, view, ranked)
        text, quotes = _quotes_beside(text, quotes, verdict)
        text = dated(text, as_of, view)
        # Each quote goes with the listed page whose passage it was copied from, not with the first page listed.
        quoted = {}
        for qt in quotes:
            home = next((p.page for _, p in prompt.passages
                         if p.page in pages[:10] and _plain(qt) in _plain(passage_text(p))), None)
            if home is not None:
                quoted.setdefault(home.key, qt)
        return Reply(text, pages[:10], True, path="reader", quotes=quoted or None)

    @staticmethod
    def _verified_quotes(raw: str, evidence: str) -> list[str]:
        """Quotes the model gave, kept only when they really are in the evidence, word for word."""
        block = re.search(r"QUOTES:\s*(.*?)\n\s*VERDICT:", raw, re.S | re.I)
        if not block:
            return []
        haystack = _plain(evidence)
        out = []
        for part in re.split(r"\[\d+\]|\n", block.group(1)):
            quote = part.strip(" -\"'“”•*")
            if 25 <= len(quote) <= 320 and _plain(quote) in haystack and quote not in out:
                out.append(quote)
        return out[:3]


def _row_lines(q, rows) -> tuple[list, list]:
    """The published values nearest the question, one line each, and each row with the pages that print it."""
    row_lines, shown = [], []
    found, sources = nearby_rows(rows, q)
    for r in in_order(q, found):
        # A date worked out from a change between downloads (see Store) is a lower bound, not a printed date.
        eff = f" (effective {r.effective.isoformat()})" if r.effective and r.dated_by != "change" else ""
        row_lines.append(f"- {'[business] ' if r.business else ''}{line(q, r)}{eff}")
        shown.append((r, sources.get(id(r), [r.page])))
    return row_lines, shown


def _gather(question: str, passages: PassageIndex, ranked: list, held_back) -> tuple[list, Optional[P.PageVersion]]:
    """The passages for the message, best first: the ranked ones, then each article's opening paragraph,
    passages of the best pages kept out by the two-per-page limit, the rest of the best page, and the
    newest article's opening and best passages. Also that newest article's page, or None when the
    evidence has articles from fewer than two pages."""
    usable = list(ranked[:EVIDENCE_K])
    shown = {id(p) for _, p in usable}

    def add(score: float, p) -> int:
        if id(p) in shown or id(p) in passages.teasers or id(p) in passages.superseded or len(p.text) <= 60:
            return 0
        shown.add(id(p))
        usable.append((score, p))
        return len(p.text)

    for s, p in list(usable[:4]):           # an article states its main fact in its opening paragraph
        lead = passages.lead.get(p.page.key)
        if lead is not None:
            add(s, lead)
    # The page that answers a question often answers it in more than one place. Passages of the
    # best pages that only the two-per-page limit kept out are brought back.
    best_pages = {p.page.key for _, p in usable[:5]}
    for s, p in [x for x in held_back if x[1].page.key in best_pages][:4]:
        add(s, p)
    # A list or table cut in two continues in the next passage; then the rest of the best page:
    # the sentence that completes the answer is often a section away.
    budget = 3000
    for _, p in list(usable[:2]):
        for x in p.page.passages:
            if x.order == p.order + 1:
                budget -= add(0.0, x)
    if usable:
        for p in sorted(usable[0][1].page.passages, key=lambda x: x.order):
            if budget <= 0:
                break
            budget -= add(0.0, p)

    # When articles disagree the newest one is right, so the newest article in the evidence
    # is shown in full where it touches the question: its opening and its best passages.
    articles = [p.page for _, p in usable if p.page.kind == P.ARTICLE and p.page.published]
    newest = max(articles, key=lambda page: page.published) if len({page.key for page in articles}) > 1 else None
    if newest is not None:
        lead = passages.lead.get(newest.key)
        if lead is not None:
            add(0.0, lead)
        for p in passages.more_from(newest, question, 2):
            add(0.0, p)
    return usable, newest


def _notes(usable: list, passages: PassageIndex) -> dict[int, list[str]]:
    """Lines to show under each passage, by id(passage)."""
    # A term the best passages link ("membership area") is explained somewhere on the site:
    # bring the passage that says what it covers. Fine print the passage points to comes too.
    notes: dict[int, list[str]] = {}
    for _, p in list(usable[:2]):
        for term in re.findall(r"(?<!!)\[([a-z][a-z ]{4,40})\]\(", p.raw):
            hop = passages.defining(term, exclude=[x for _, x in usable])
            if hop is not None:
                at = re.search(re.escape(term), hop.text, re.I)
                said = hop.text[at.start():at.start() + 420].strip()
                notes.setdefault(id(p), []).append(
                    f'Definition of "{term}", from {hop.page.url}: {said}')
    for _, p in list(usable[:6]):
        notes.setdefault(id(p), []).extend(passages.footnotes(p))
    return notes


def _rate_gap(question: str, q, match, rows, usable: list, row_lines: list, shown: list) -> str:
    """What the rate tables do not have is evidence too: a note that says so, so a description of the
    product is not mistaken for its rate; "" when the tables have it. When a passage says the product
    has another's rates, the note says that instead, and that product's rows join row_lines and shown."""
    missing = ""
    if asks_rate(q):
        absent = sorted((match.unmatched & q.modifiers) if match else q.modifiers)
        if absent or match is None or match.tier_missing:
            what = " ".join(absent) or "this product, term or tier"
            missing = (f"\n\nNote: the site's rate tables have no row for \"{what}\". Answer only if a passage "
                       f"states the specific rate asked about; a description of the product is not its rate.")
            # "... earns the same dividends as a basic share account": the rate is published, under another name.
            same = next((m for _, p in usable[:8] if set(p.toks) & set(absent)
                         for m in [_SAME_RATE_AS.search(p.text)] if m), None)
            if same:
                other = parse_question(f"{same.group(1)} rate {' '.join(re.findall(r'[$][0-9][0-9,.]*', question))}")
                found, sources = nearby_rows(rows, other)
                for r in found[:8]:
                    entry = f"- {'[business] ' if r.business else ''}{line(other, r)}"
                    if entry not in row_lines:
                        row_lines.append(entry)
                        shown.append((r, sources.get(id(r), [r.page])))
                missing = (f"\n\nNote: the site's rate tables have no row named \"{what}\", but a passage says it has "
                           f"the same rates as \"{same.group(1).strip()}\". Find that product among the published "
                           f"values and give its rate for the balance or term asked about.")
    return missing


def _settle(verdict: str, text: str, question: str, today: int, rate_gap: bool) -> tuple[str, str]:
    """The verdict the reply's text bears out, and the text to give. The model's own verdict is
    overridden two ways, each against one failure:
    - answer becomes decline when the answer says the site does not state it and the question is not
      a yes-or-no one: that answer is a refusal. Not when it first states a figure the question does
      not have and the gap is a caveat about something else (_fact_before_gap), unless the rate
      tables lack what was asked: there a figure before the gap is likely a neighbouring product's;
    - decline becomes answer when the reply gives the latest years' figures for a past year that has
      none (SYSTEM rule 3).
    Each was measured on recorded replies, by replaying them with it switched off. Otherwise the
    model's verdict stands: contact gives the contact route, decline a refusal. An answer about such a
    past year is given as those latest years' figures."""
    period = _by_period(text, question, today)
    if verdict == "answer" and _not_stated(text, question, today) and not _YES_NO.search(question) \
            and (rate_gap or not _fact_before_gap(text, question)):
        verdict = "decline"                                       # "the site does not list a fee" is a refusal
    elif verdict == "decline" and period and not _not_stated(text, question, today):
        verdict = "answer"                                        # the latest years' figures (SYSTEM rule 3)
    if verdict == "answer" and period:
        text = period[1]
    return verdict, text


def _cited_pages(prompt: Prompt, cited: str, verdict: str, text: str, question: str, view, ranked: list) -> list:
    """The pages to cite: the passages the model named, the contact page for a contact reply, every
    page that prints a row whose value the answer gives (most trusted first, as the rows' own answers
    cite them), every page that prints a number it gives, the best-ranked pages, a page it links to,
    and the contact page for a question about branches or hours."""
    pages = []
    for n in re.findall(r"\d+", cited):
        i = int(n) - 1
        if 0 <= i < len(prompt.passages) and prompt.passages[i][1].page not in pages:
            pages.append(prompt.passages[i][1].page)
    if verdict == "contact":
        add_contact(pages, view.cache["passages"])
    given = {f for f in figures(text) if f[0]}                # a rate or an amount, not a bare count or term
    used = [p for r, printing in prompt.shown if any(figures(v.text) & given for v in r.values) for p in printing]
    for page in used or (list(dict.fromkeys(r.page for r, _ in prompt.shown))[:3] if not pages else []):
        if page not in pages:
            pages.append(page)
    for number in re.findall(r"\b\d[\d-]{6,}\d\b", text):      # a routing or phone number: cite the pages that print it
        pages.extend(view.pages_printing(number, pages)[:6])
    for _, passage in ranked[:2]:                                # the best-ranked pages were the grounds as well
        if passage.page not in pages:
            pages.append(passage.page)
    for url in re.findall(r"https?://[^\s)\"'>]+", text):      # a link given as the answer is a source too
        hit = view.pages.get(P.url_key(url.rstrip(".,;")))
        if hit is not None and hit not in pages:
            pages.append(hit)
    # Hours, addresses and branch details: the contact page is the place to confirm them.
    if re.search(r"\bhours?\b|\bbranch|\blobby\b|\bdrive[- ]?(up|thru|through)\b|\batm\b|\blocation|\baddress\b", question, re.I):
        add_contact(pages, view.cache["passages"])
    return pages


def _quotes_beside(text: str, quotes: list, verdict: str) -> tuple[str, list]:
    """Keep the site's own wording next to the answer when the answer rephrased it: the answer with
    up to two quotes added, and the quotes as they are shown (without link targets and emphasis)."""
    # The answer's own counts get their digits first, so a quote is compared with them and is then
    # shown word for word.
    text = digits_beside_words(text)
    said = _plain(text)
    quotes = [re.sub(r"\s+", " ", re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", qt).replace("*", "")).strip() for qt in quotes]
    numbers = set(re.findall(r"\d[\d,.]*\d|\d", said))
    # A quote is added for its wording, never for new figures: every number in it must already
    # be in the answer, so a superseded value cannot ride in on a quote.
    extra = [qt for qt in quotes if _plain(qt) not in said
             and any(w not in said for w in re.findall(r"[a-z]{4,}", _plain(qt)))
             and set(re.findall(r"\d[\d,.]*\d|\d", _plain(qt))) <= numbers][:2]
    if extra and verdict == "answer":
        text += " The site says: " + " ".join(f'"{qt}"' for qt in extra)
    return text, quotes


def _not_stated(answer: str, question: str, today: int = 0) -> bool:
    """Whether an answer says the site does not state what was asked (see _DISCONTINUED and _by_period)."""
    if figures(answer) - figures(question):
        period = _by_period(answer, question, today)
        answer = _DISCONTINUED.sub("", period[0] if period else answer)
    return bool(_NOT_STATED.search(answer))


def _new_figure(s: str, question: str) -> bool:
    """s gives a figure the question does not have, years and dates aside: "in 2024" or "March 3" says
    when, not how much."""
    return bool(figures(_YEAR_IN_FIGURES.sub("", _DATE.sub("", s))) - figures(question))


# The words a sentence about a gap uses for the source, for saying it is silent, for the missing fact
# and for what was asked, whatever that was: "The price of a box is not given on the site" names
# nothing beyond "a box". ("doesn" is how tokens() leaves "doesn't".)
_GAP_WORDS = frozenset("site website page document table source passage evidence state stated mention "
                       "mentioned specify specified provided given shown disclosed doesn don isn aren cannot confirm "
                       "determine price figure date time day yet exact specific his her".split())
_RATE_WORD = re.compile(r"\b(?:rates?|aprs?|apys?|dividends?|yields?)\b", re.I)


def _inflected(a: str, b: str) -> bool:
    """b starts with a's stem (a less a final "e", 3 letters at least): "move" and "moving", "transfer"
    and "transferred". Loose ("car" and "card" pass too); a false match only keeps the reply a refusal."""
    a, b = sorted((a, b), key=len)
    stem = a[:-1] if a.endswith("e") else a
    return a == b or len(stem) >= 3 and b.startswith(stem)


def _fact_before_gap(answer: str, question: str) -> bool:
    """The answer gives a figure (see _new_figure) before its first sentence that says the site does
    not state something, and that sentence is a caveat, not the question's own subject: it names a
    word the question does not have ("Two options: $1,200 and $8,000. No business minimum is published"
    on "what are the minimums"), or a rate when the question asks for none. "The cutoff time is not
    published", after another fact, restates what was asked and leaves it unanswered."""
    stated = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        if _NOT_STATED.search(_DISCONTINUED.sub("", sentence)):
            break
        stated.append(sentence)
    else:
        return False
    asked = content_tokens(question)
    caveat = any(not any(_inflected(w, a) for a in asked)
                 for w in content_tokens(sentence) if w not in _GAP_WORDS) \
        or (bool(_RATE_WORD.search(sentence)) and not asks_rate(parse_question(question)))
    return caveat and _new_figure(" ".join(stated), question)


def asked_years(question: str, today: int) -> list[tuple]:
    """(year, words, since) for each year the question names (see _ASKED_YEAR): the year as a number,
    the question's words for it, and whether it follows "since". `today` is the current year. Beside
    a year in figures, "the previous year" is the one before that year, so it is left out."""
    found = [m.groups() for m in _ASKED_YEAR.finditer(question)]
    in_figures = any(number for _, number, _ in found)
    out = []
    for since, number, words in found:
        if number:
            out.append((int(number), number, bool(since)))
        elif not (in_figures and words.lower().startswith(("previous", "prior"))):
            out.append((today + year_offset(words), words, bool(since)))
    return out


def _by_period(answer: str, question: str, today: int) -> Optional[tuple]:
    """For a question about a past year (`today` is the current year; see asked_years), an answer
    that says that year has no figure and gives the latest years' instead (SYSTEM rule 3): the answer
    without its sentences about the gap, and the answer to give. Otherwise None. What is left must
    give a figure the question does not have beside a year, and no sentence may name a year later
    than the one asked: a later figure means the site has a newer value, which for a standing fact
    (a member count) replaces the older ones. Another sentence about the asked year itself (its
    hours, when its count was asked) is no part of that answer, and is dropped from it."""
    found = asked_years(question, today)
    asked = {year for year, _, _ in found}
    if not asked or max(asked) >= today or any(since for _, _, since in found):
        return None
    # Not after "Oct."; "(For 2025, ...)" is a sentence of its own.
    sentences = re.split(r"(?:(?<=[.!?])|(?<=[.!?]\)))\s+(?=\(?[A-Z])|;\s*", answer)
    named = re.compile("|".join(map(re.escape, sorted({str(y) for y in asked} | {w for _, w, _ in found}))), re.I)
    gaps = [s for s in sentences if (_NOT_STATED.search(s) or _NO_FIGURE.search(s)) and named.search(s)]
    others = [s for s in sentences if s not in gaps]
    if not gaps or any(int(y) > min(asked) for s in others for y in _YEAR_IN_FIGURES.findall(s) if int(y) not in asked):
        return None
    rest = [s for s in others if not named.search(s)]
    if not any(_new_figure(s, question) for s in rest if _YEAR_IN_FIGURES.search(s)):
        return None
    given = answer
    for s in others:
        if named.search(s):
            given = re.sub(re.escape(s) + r"(?:\s*;)?\s*", "", given, count=1)
    return " ".join(rest), given.strip(" ;")


def _plain(s: str) -> str:
    """s for comparing wording: normalised as the pages are (clean), without link targets, quote
    marks, emphasis and case. The reply is normalised and the evidence is not, so both sides of a
    comparison go through here."""
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", clean(s))
    s = re.sub(r"[*_`\\\"']", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def make_reader() -> Optional[Reader]:
    """The reader for this run, or None for rules only. BANKGPT_LLM=off (or 0, none, false) forces rules
    only; otherwise the API is used when a key is set."""
    mode = os.environ.get("BANKGPT_LLM", "auto").lower()
    if mode in ("off", "0", "none", "false"):
        return None
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        try:
            return Reader(AnthropicBackend())
        except ImportError:
            print("[bankgpt] ANTHROPIC_API_KEY is set but the `anthropic` package is missing "
                  "(pip install -r requirements.txt); answering from rules", file=sys.stderr)
            return None
    print("[bankgpt] no ANTHROPIC_API_KEY: answering from rules only (free-text questions will be weaker)",
          file=sys.stderr)
    return None
