"""The knowledge system: ingest snapshots, answer questions, list contradictions.

One question goes through these steps, in order (Engine._answer):
    1. Pick the view: the newest download, or the one in force on the `as_of` date.
    2. The winning rows (kb.lookup), when they carry the published value asked (_rows_suffice).
    3. Opening hours from the published schedules (kb.hours).
    4. A rate for a kind of product no table lists and no passage states: not published, by rule.
    5. Otherwise the reader words an answer from the nearest rows and best passages, or declines;
       without a reader, or when the model gives no usable reply, kb.quote quotes the site or declines.
Steps 2 to 4 use no model, so those answers are exact, fast and the same every time. What is
current is decided by rules in every step; the model only reads.
"""
from __future__ import annotations

import re
from typing import Optional

from . import hours, quote
from . import pages as P
from .compose import answer_from_rows
from .conflicts import find_conflicts
from .lookup import HEAD_NOUNS, RATE_METRICS, Match, Query, RowIndex, asks_rate, find, parse_question
from .passages import EVIDENCE_K, PassageIndex, add_contact
from .reply import Reply, dated, refusal
from .store import Store, as_of_day
from .text import SITE_NAMES, clean, tokens


def _asked_kinds(q: Query) -> set:
    """What kind of value the question asks for: a rate, a fee or a minimum."""
    return {"fee" if m in ("fee", "payment") else "min" if m == "min" else "rate" for m in q.metrics}


def _value_kinds(row) -> set:
    """What kind of value a row states."""
    return {"rate" if v.metric in RATE_METRICS else
            "min" if v.metric.startswith("min") else "fee" for v in row.values}


# What a product is called: "intro" in "intro rate" names no kind of product, and no branch has a rate.
_PRODUCT_HEADS = HEAD_NOUNS - {"rate", "fee", "branch", "office", "location"}


def _kind_words(question: str) -> set:
    """The word that names the kind of product: the one before its head nouns ("youth" in "a youth
    savings account", "construction" in "a home construction loan")."""
    words = tokens(question, keep_stop=True)
    return {a for a, b in zip(words, words[1:]) if b in _PRODUCT_HEADS and a not in HEAD_NOUNS}


def _no_rate(question: str, missing: set, rows: RowIndex, passages: PassageIndex) -> str:
    """The sentence "The website does not publish a rate for construction loans.": the question's own words, from
    the first one no rate table has (and the figure before it: "7/1") to the kind of product it
    qualifies. "" when there are none, when they run long (a typo can be "missing"), or when the rows
    do list a rate for them: then what is missing is a tier ("a 13-month certificate"), and naming
    only the product would be false. "" too when a passage found by those words states a rate for
    them: the question's other words may have kept it out of the evidence, and the claim would be false."""
    words = re.findall(r"[\w/'-]+", re.sub(r"'s\b", "", question))          # "the loan's rate": the loan
    first = next((i for i, w in enumerate(words) if w[0].isalpha() and set(tokens(w)) & missing), None)
    for end in range(first, len(words)) if first is not None else ():
        if set(tokens(words[end])) & _PRODUCT_HEADS:
            if end - first > 2 or not all(w[0].isalpha() and tokens(w) for w in words[first:end]):
                return ""
            noun = words[end]
            if not noun.endswith(("s", "ing")):          # "checking" and "savings" stay as they are
                noun += "es" if noun.endswith(("ch", "sh", "x")) else "s"
            named = f"What is the rate for {' '.join(words[first:end] + [noun])}?"
            listed = find(rows, parse_question(named))
            if listed is not None and listed.rows and not listed.unmatched or \
                    any(quote.states_rate(p, missing) for _, p in passages.search(named)):
                return ""
            start = first - 1 if first and re.search(r"\d", words[first - 1]) else first
            return f"The website does not publish a rate for {' '.join(words[start:end] + [noun])}."
    return ""


class Engine:
    """One knowledge system: the store of downloads, its indexes per view, and the optional reader."""

    def __init__(self, reader=None) -> None:
        self.store = Store()
        self.reader = reader                      # kb.reader.Reader or None
        self._answers: dict[tuple, Reply] = {}

    # ------------------------------------------------------------------ ingest

    def ingest(self, snapshot) -> None:
        """Add one download and rebuild every view's indexes."""
        self.store.add_snapshot(snapshot)
        # RowIndex first: it marks view.stale, which PassageIndex and View.authority read.
        # A new download teaches something about the earlier ones (dates, labels, which copies
        # were left behind), so every view's indexes are rebuilt, not only the newest.
        for view in self.store.views:
            view.cache.clear()
            view.cache["rows"] = RowIndex(view)
            view.cache["passages"] = passages = PassageIndex(view)
            view.cache["hours"] = found = [s for page in view.pages.values() for s in hours.extract(page)]
            # Which branch a schedule belongs to.
            view.cache["hours_about"] = {s.page.key: hours.opening_words(s.page) for s in found}
            # Words in a third of all passages (the organisation's own name) say nothing about which
            # branch is meant. Only on a site large enough for "a third" to mean something.
            view.cache["common"] = SITE_NAMES | ({t for t, n in passages.df.items() if n > 0.3 * passages.n}
                                                 if passages.n >= 200 else set())
        self._answers.clear()

    # ------------------------------------------------------------------ answer

    def answer(self, question: str, as_of: Optional[str] = None) -> Reply:
        """The answer for the site as it stood on `as_of` (the newest download when None), cached."""
        # Normalised once, here, the way the pages are: a curly apostrophe, a non-breaking space or a
        # doubled space in a question then changes nothing that reads it, the reader's checks included.
        question = clean(question)
        # The date is read once, here, and written in one form from then on. One that cannot be read is no
        # date at all: the answer is for the newest download and claims no date.
        day = as_of_day(as_of)
        as_of = day.isoformat() if day else None
        # Keyed by the question as written: a name in capitals is read differently from one in small letters.
        key = (" ".join(question.split()), as_of)
        if key not in self._answers:
            self._answers[key] = self._answer(question, as_of)
        return self._answers[key]

    def _answer(self, question: str, as_of: Optional[str]) -> Reply:
        view = self.store.view_at(as_of)
        if view is None:
            return refusal()
        rows: RowIndex = view.cache["rows"]
        passages: PassageIndex = view.cache["passages"]
        q = parse_question(question)
        match = find(rows, q)

        if self._rows_suffice(q, match):
            # A rate question is answered by the rows that state what was asked: a car's minimum bid
            # is no answer to an APR question, though its make ("Buick") makes it an "auto" row too.
            if "rate" in _asked_kinds(q):
                match.rows = [r for r in match.rows if _asked_kinds(q) & _value_kinds(r)]
            text, pages, note = answer_from_rows(q, match, rows, as_of)
            pages = self._also_stating(view, match, pages)
            return Reply(text, pages[:10], True, note, path="rows")

        # Opening hours are read from the published schedules, the same for every wording: for
        # the branch the question names, or every distinct schedule when it names none.
        if view.cache["hours"]:
            found = hours.for_place(question, passages.query_tokens(question), view.cache["hours"],
                                    view.cache["hours_about"], view.cache["common"])
            if found is None and hours.is_general(question, view.cache["common"]):
                found = hours.answer(question, view.cache["hours"])
            if found is not None:
                text, pages = found
                add_contact(pages, passages)
                return Reply(dated(text, as_of, view), pages[:10], True, path="hours")

        held_back: list = []
        ranked = passages.search(question, k=EVIDENCE_K, held_back=held_back)
        ranked = [(s, p) for s, p in ranked if p.page.key not in view.stale] or ranked
        missing = self._rate_not_published(q, match, ranked, passages)
        # The rows may still list what is asked: tied rows can each match other words ("used" one, "boat"
        # another), or the missing word only describes the kind they list ("personal" in "an unsecured personal
        # loan"; in "a home construction loan" the kind is construction, whatever "home" matches). Then the
        # reader, which sees the rows, decides. Without one nothing tells which row is meant, nor can it be
        # said that no rate is published: the plain refusal. ("business" names no kind.)
        covered = match.covered if match is not None else set()
        listed = bool(missing) and (missing <= covered or bool(_kind_words(q.text) & (covered - {"business"})))
        if missing and not (listed and self.reader is not None):
            return refusal(question, "" if listed else _no_rate(question, missing, rows, passages))
        if self.reader is not None:
            reply = self.reader.read(question, as_of, view, q, match, ranked, rows, held_back)
            if reply is not None:
                return reply
        return quote.answer(question, as_of, view, q, match, ranked, passages)

    @staticmethod
    def _rows_suffice(q: Query, match: Optional[Match]) -> bool:
        """The rows answer alone when they carry the measure asked, leave no modifier of the product
        unmatched, state a rate to rank them by when the highest or lowest is asked, and match at least
        half (tables, and a fee schedule's lines, which are its table read as text) or three quarters
        (other rows) of the product words."""
        if not match or not match.rows or match.tier_missing or not q.wants_number:
            return False
        if match.unmatched & q.modifiers:
            return False                      # "construction loan": a kind of product the rows do not list
        if not (match.matched - {"business"}):
            return False                      # nothing but "business" matched: that names no product
        if q.metrics and not any(_asked_kinds(q) & _value_kinds(r) for r in match.rows):
            return False                      # asked what something costs; these rows only give rates
        if q.superlative and not any("rate" in _value_kinds(r) for r in match.rows):
            return False                      # "the smallest locker": no rate to rank them by, so it needs reading
        share = len(match.matched) / max(1, len(q.product))
        if any(r.page.kind in (P.ARTICLE, P.INDEX, P.CALCULATOR) for r in match.rows):
            return False                      # a number inside an article needs reading, not lookup
        if not q.metrics:
            # "how much ...", "what number ...": only a near-exact match, and not a table of rates
            # ("how much did members save in the Snowball Club" is not asking for its APY).
            return share >= 0.75 and all(v.unit != "%" for r in match.rows for v in r.values[:1])
        if all(r.kind in ("table", "fee") for r in match.rows):
            return share >= 0.5
        return share >= 0.75

    @staticmethod
    def _rate_not_published(q: Query, match: Optional[Match], ranked: list, passages: PassageIndex) -> set:
        """A rate is asked for a named kind of product ("construction loan") that no rate table
        lists, and no passage about it states a rate either: the site does not publish it. Returns
        the words that name it ("construction"), or nothing when the rate may be published.
        Only words the site itself uses count; an unknown word may be a typo, and is left to be read."""
        if not (asks_rate(q) and q.modifiers):
            return set()
        missing = (match.unmatched & q.modifiers) if match is not None and not match.tier_missing else set(q.modifiers)
        missing = {t for t in missing if passages.known(t)}
        return set() if any(quote.states_rate(p, missing) for _, p in ranked[:8]) else missing

    @staticmethod
    def _also_stating(view, match: Match, pages: list) -> list:
        """For a single labelled value (a routing number), cite every trusted page that prints it."""
        if len(match.rows) != 1 or match.rows[0].kind == "table":
            return pages
        value = match.rows[0].values[0].text
        if len(re.sub(r"\D", "", value)) < 5:
            return pages
        return pages + view.pages_printing(value, pages)

    # ------------------------------------------------------------------ conflicts

    def conflicts(self) -> list:
        """The contradictions across the downloads (see kb.conflicts)."""
        return find_conflicts(self.store)
