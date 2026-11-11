"""Answer "what is the value of X" from the rows, and decide which copy of X is current.

Two steps, both plain rules:

1. Match. Read the question into product words ("share savings"), qualifiers (60 months,
   $3,000-$11,999, model year 2025) and the measure asked for (APY, APR, fee). Keep the rows
   whose product words match best and whose qualifiers fit. A row for another term, tier or
   model year is a different fact and is never returned.

2. Resolve. The same row is usually printed on several pages. When the copies disagree, the
   winner is chosen by, in order:
       a promotion that has ended loses;
       the more trusted kind of page wins (product page > document > campaign page >
       article or calculator; a page caught showing out-of-date tables is demoted);
       the newer "effective as of" date wins (see newer(): a date worked out from a change
       between downloads is a lower bound, and beats only an older date);
       a copy its page keeps updating beats a copy that never changes;
       the value shown on more pages wins, counting only copies that no newer date has beaten;
       a copy with a printed effective date beats one without;
       the more recently downloaded page wins;
       a table row beats any other kind of row (a labelled value, a fee line, a sentence);
       finally, the page with the shorter address (nearer the top of the site) wins.
   Dates only rank dated copies, so the copies with a printed date are settled among
   themselves, the others (undated, or dated only by a change) among themselves, and the two
   survivors then meet under the same rules. When the newest dated copy meets an undated one
   and trust, dates and upkeep tie, an older dated copy that agrees with the undated one does
   not count for it: the undated value wins on pages only if more current pages show it, and
   otherwise the printed date wins.
   The losing values are kept for the conflict report, never for the answer.
"""
from __future__ import annotations

import difflib
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Optional

from . import pages as P
from .tables import Row
from .text import DESCRIBING, INF, MONTHS, STOPWORDS, Quals, canon, content_tokens, parse_term, scale, tokens

# --------------------------------------------------------------------------- the question

# The parts the date patterns below are built from.
_YEAR4 = r"(?:19|20)\d\d"                                    # 2026
_DAY_NO = r"\d{1,2}(?:st|nd|rd|th)?"                          # 1, 1st, 21st
_LEAD = r"(?:as of|back in|in|during|since|by|at the end of|at the start of|around|before|after|for)"
_PART = r"(?:early|mid|late|the end of|the start of)"        # of a month
_DAY = r"(?:[12]?\d|3[01])(?:st|nd|rd|th)?"                   # a day of the month, 1 to 31
_TERM_UNIT = r"\s*-?\s*(?:months?|mos?|years?|yrs?)\b"         # a term's unit: "5 years", "36-month"
# "for 2024" is a date; "for 2024 motorcycles" is a model year (canon has already made cars "auto").
_YEAR_WORDS = (rf"\b(?:in|during|since|of|year|for|by)\s+({_YEAR4})\b(?!\s*(?:model|auto|or newer|or older|"
               r"(?:motorcycles?|atvs?|snowmobiles?|jet[- ]?skis?|boats?|rvs?|campers?|trailers?)\b))")
_MONTHS_BUT_APR = "|".join(sorted((m for m in MONTHS if m != "apr"), key=len, reverse=True))
# A year that dates the question ("May 2026", "June 1, 2026", "in 2026"), so no model year.
_DATED_YEAR = rf"\b(?:{_MONTHS_BUT_APR})\.?\s+(?:{_DAY_NO},?\s+)?{_YEAR4}\b|{_YEAR_WORDS}"
# "APR 36 months" and "APR 2025 model" ask for a rate: "apr" is April only before a full date ("Apr 5,
# 2026"), before a year that the question's own model year comes after ("In Apr 2026, ... a 2021 car"),
# or after a date word and before a day that ends the phrase ("on Apr 5?", "since Apr 5 to 9"). Not in
# "the HELOC APR 6, or 7?", "APR 6, 2024 model", "APR 5 to 7 years", "APR 5.5%" or "APR 7/1 ARM". A
# later year that is itself a date is no model year: "APR 2025 auto loan as of May 2026" asks for the
# 2025 rate. Only the next year is looked at, and the later year is found by looking ahead to the first
# one: looking through all the rest of the text, at every "apr", is quadratic ("apr 2026" x 30,000).
_DAY_ENDS = rf"(?=\s*(?:[,?!]|\.(?!\d)|$)|\s+(?:{_YEAR4}\b|(?:to|through|until|and|-)\s+(?!\d+{_TERM_UNIT})))"
_MONTH_RX = _MONTHS_BUT_APR + rf"|apr(?=\.?\s+(?:{_DAY_NO},?\s+{_YEAR4}\b(?!\s*model)" \
    rf"|(?:of\s+)?{_YEAR4}\b(?!(?:(?!\b{_YEAR4}\b).)*?(?:" + _DATED_YEAR + rf")).*?\b{_YEAR4}\b))"
_DATE_PHRASE = re.compile(
    rf"\b{_LEAD}?\s*{_PART}?[- ]?(?:{_MONTH_RX})\.?"
    rf"(?:\s+{_DAY_NO}(?!{_TERM_UNIT}))?,?(?:\s+(?:of\s+)?{_YEAR4})?\b"
    rf"|\b(?:{_LEAD}|on|until|from|between)\s+{_PART}?[- ]?apr\.?\s+{_DAY}{_DAY_ENDS},?(?:\s+(?:of\s+)?{_YEAR4})?\b"
    # "in mid-Apr 2026": a date word before "apr" and a year after it make it April; so do a date word
    # and a part of a month before it ("in mid-Apr", "by late Apr"; "the late APR" is a fee's rate).
    rf"|\b{_LEAD}\s+{_PART}?[- ]?apr\.?\s+(?:of\s+)?{_YEAR4}\b(?!\s*model)"
    rf"|\b{_LEAD}\s+{_PART}[- ]?apr\b\.?(?:\s+(?:of\s+)?{_YEAR4}\b)?"
    rf"|\b{_YEAR4}-\d\d-\d\d\b", re.I)


def without_april(text: str) -> str:
    """text with "Apr" left out where it is the month (see _DATE_PHRASE), so that a search for its
    words does not take "in mid-Apr 2026" for the APR."""
    return _DATE_PHRASE.sub(lambda m: re.sub(r"\bapr\b\.?", " ", m.group(0), flags=re.I), text)


# Read before canon(), which keeps "7/1" as one word ("7x1") and so would turn 05/31/2026 into "05x31/2026".
_SLASH_DATE = re.compile(r"\b(?:\d{1,2}/){1,2}(?:19|20)\d\d\b|\b\d{1,2}/\d{1,2}/\d\d\b|\b(?:19|20)\d\d/\d{1,2}/\d{1,2}\b")
_YEAR_CONTEXT = re.compile(_YEAR_WORDS, re.I)
_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?"
_MONEY = re.compile(r"\$\s?" + _NUM + r"\s?(k|thousand|grand|m|million)?\b|\b" + _NUM + r"\s?(k|thousand|grand|million)\b"
                    r"|\b(\d{1,3}(?:,\d{3})+)\b|\b" + _NUM + r"\s?dollars?\b", re.I)
# A plain number of three or more digits is an amount only right after a word for holding, putting or
# borrowing money: "I have 8000 in it", "a balance of 2500", "borrow 15000"; not "with 720 credit score",
# "with 5000 members" or a down payment ("put 3000 down", "6000 toward a boat").
_BARE_AMOUNT = re.compile(r"\b(?:have|having|with|deposit\w*|put\w*|balance of|borrow\w*|invest\w*|save\w*)\s+"
                          r"(?:about |around |roughly |like |only |just )?(\d{3,7})\b(?![.,]\d|\s*(?:%|percent|points?|"
                          r"credit|scores?|fico|days?|weeks?|miles?|sq|square|people|members?|payments?|down|towards?)"
                          r"\b)", re.I)
# "(?<!\d)": a percentage starts where its digits do, so a long run of digits is not tried from each one.
# A down payment is not the amount borrowed: "$2000 down", "a $5,000 down payment", "down payment of $20k".
_NOT_AN_AMOUNT = re.compile(r"\b\d{1,3}\s?-?\s?(?:months?|mos?|years?|yrs?)\b|(?<!\$)(?<!\$ )\b(?:19|20)\d\d\b"
                            r"|(?<!\d)\d+(?:\.\d+)?\s?%|\b\d+x\d+\b|\b40[13]\s?k\b"
                            r"|\$\s?[\d,.]+\s?(?:k|thousand|grand)?\s+down\b|\bdown ?payment (?:of |is )?\$\s?[\d,.]+")
_MEASURE = {
    "apy": "apy", "yield": "apy", "apr": "apr", "dividend": "dividend", "point": "points", "payment": "payment",
    "fee": "fee", "cost": "fee", "charge": "fee", "price": "fee", "rate": "rate", "interest": "rate",
}
RATE_ORDER = ("dividend", "apy", "rate", "apr")           # the measures a rate question asks, in the order
RATES = frozenset(RATE_ORDER)                             # an answer looks for them in a row
RATE_METRICS = RATES | {"points"}                          # and the ones a row states as a rate
HEAD_NOUNS = frozenset("loan mortgage certificate account card creditcard branch checking saving moneymarket "
                       "heloc ira office location rate special fee".split())
# A loan is not a deposit: what a product is called and the measure it states say which it is.
LENDING = frozenset("loan mortgage heloc card creditcard".split())
SAVING = frozenset("certificate account checking saving moneymarket ira hsa".split())
_NEGATION = re.compile(r"\b(?:non[- ]|not (?:the |a |an )?|no |without (?:the |a |an )?|excluding (?:the )?|"
                       r"other than (?:the )?)([a-z0-9]+)")
# The unit of a term or an amount, on its own or glued to its number ("mos", "6mo", "5k"): not a product word.
# A 401k is a plan, not $401,000 (as in _NOT_AN_AMOUNT).
_UNIT_WORDS = re.compile(r"(?!40[13]k)\d*(?:month|year|yr|mo|day|week|k|grand|thousand|million|dollar)s?")
# "shorter than 12 months", "at least 24 months" and "24 months or longer" bound the term; they do not name
# it. A bare "over" is not a bound: "spread it over 72 months" names the term.
_A_TERM = r"(\d{1,3}\s?-?\s?(?:months?|mos?|years?|yrs?))"
_TERM_BOUND = re.compile(rf"\b(under|below|less than|shorter than|fewer than|at most|up to|no (?:more|longer) than"
                         rf"|more than|longer than|greater than|at least|no (?:less|shorter) than)\s+{_A_TERM}\b"
                         rf"|\b{_A_TERM}\s+(or (?:less|shorter|fewer|more|longer|greater)|and (?:up|over|longer))\b")
_UPPER = re.compile(r"under|below|less|shorter|fewer|at most|up to|no (?:more|longer)|or (?:less|shorter|fewer)")


@dataclass
class Query:
    """A question read into what it asks about: the product words, the qualifiers and the measures."""
    text: str                                     # the question, lower-cased with synonyms folded
    product: list                                 # words that name the thing asked about
    quals: Quals                                  # the term, amount and model year asked about
    metrics: set                                  # the measures named: apy, apr, rate, fee, min ...
    business: bool
    superlative: Optional[str]                    # 'max' ("highest") or 'min' ("lowest")
    wants_number: bool                            # asks for a published value at all
    wants_link: bool                              # "what link should I use", "what's the URL"
    modifiers: set = field(default_factory=set)   # product words right before a head noun: "construction loan"
    excluded: set = field(default_factory=set)    # "non-introductory": rows with this word are not wanted
    spelled: bool = False                         # typos already corrected against the rows' vocabulary
    terms: Optional[tuple] = None                 # "under 12 months": the terms that qualify, (0, 11)


# A question that mentions a rate but asks for something else: a fee or a minimum as the thing asked
# ("is there a monthly fee on the checking account that pays interest?"; not the verb "charge", which
# asks a rate in "what rate do you charge", nor a fee or minimum that describes the product: "the APR
# on a card with no fee", "a no-fee card", "a certificate with a $1,000 minimum"), or how a rate
# changes, with the rate as what changes ("how much can my rate go up", "how often does the rate on
# my HELOC adjust", "what's the most my rate could ever go", "rate caps"; not "the APR if rates go
# down" or "the rate on a bridge loan, and can it change?", which ask the rate too). The verbs are
# whole words: "adjustable" in "a 7/1 adjustable-rate mortgage" asks a rate. "Fall" is left out: it is
# also the season ("rates this fall"). Spans are bounded so the scan stays linear.
_CHANGES = r"(?:go(?:es)?|change[sd]?|adjust(?:s|ed)?|increase[sd]?|rise[sn]?|drop|reset)\b"
_A_RATE = r"(?:rates?|aprs?|apys?)"
_ASKED = r"\b(?:what|which|how much|is there|are there|any)\b(?:(?!\b(?:no|with|without)\b)[^?.]){0,40}?(?<!-)"
_NOT_A_RATE = re.compile(
    rf"{_ASKED}\b(?:fees?|min(?:imum)?)\b(?! (?:apr|apy|rate|interest|dividend)\b)"
    rf"|\b(?:how (?:much|high|far|often|low)|ever|most|max(?:imum)?|can|could|will|would|may|might)\b[^?.]{{0,80}}?"
    rf"(?<!\bif )(?<!\bwhen )\b{_A_RATE}(?:\s+(?!if\b)\w+){{0,4}}?\s+{_CHANGES}"
    rf"|\b{_A_RATE} (?:can|could|will|would|may|might) (?:ever )?{_CHANGES}"
    rf"|\b{_A_RATE} (?:caps?|adjustments?|changes?)\b")
# A question can ask a rate in one clause and a fee in the next ("the rate on a VA loan, and what fees
# apply?"), so each clause is read on its own.
_CLAUSE = re.compile(r"[?;,(&]|\b(?:and|plus)\b")


def asks_rate(q: Query) -> bool:
    """Whether the question asks for a published rate value: it wants a number, and one of its clauses
    names a rate measure and asks no fee, minimum or change of rate instead (see _NOT_A_RATE)."""
    return q.wants_number and bool(q.metrics & RATES) and any(
        not _NOT_A_RATE.search(c) and any(_MEASURE.get(w) in RATES for w in tokens(c))
        for c in _CLAUSE.split(_DATE_PHRASE.sub(" ", q.text)))


def _lending_or_saving(words: set, metrics: set = frozenset()) -> set:
    """'lend' for a loan, mortgage or card, 'save' for an account or certificate; both or neither when
    the words do not say. A row's APR is charged on what is lent; its APY or dividend is paid on savings."""
    out = set()
    if words & LENDING or "apr" in metrics:
        out.add("lend")
    if words & SAVING or metrics & {"apy", "dividend"}:
        out.add("save")
    return out


def parse_question(question: str) -> Query:
    """Read a question into a Query. Dates are taken out first, so a year in one is no model year."""
    text = canon(_SLASH_DATE.sub(" ", question))
    # Dates name the time asked about, not a model year or a term.
    undated = _YEAR_CONTEXT.sub(" ", _DATE_PHRASE.sub(" ", text))

    # A year that is not part of a date is taken as a model year ("a 2021 sedan"); it only
    # ever matters for rows that are keyed by model year. After a dollar sign it is money ("$2000").
    year = re.search(r"(?<!\$)(?<!\$ )\b((?:19|20)\d\d)\b", undated)
    years = (float(year.group(1)), float(year.group(1))) if year else None
    term, terms = parse_term(undated), None
    for bound in _TERM_BOUND.finditer(undated):                          # "at least 12 months but under 36 months"
        word, n = bound.group(1) or bound.group(4), parse_term(bound.group(2) or bound.group(3))
        strict = not re.match(r"at |up |no |or |and ", word)            # "under 12": 11 at most; "at most 12": 12
        lo, hi = (0, n - strict) if _UPPER.match(word) else (n + strict, INF)
        term, terms = None, (max(lo, terms[0]), min(hi, terms[1])) if terms else (lo, hi)
    quals = Quals(term, _asked_amount(undated), years)

    excluded = {t for m in _NEGATION.finditer(undated) for t in content_tokens(m.group(1))}
    date_words = set(tokens(" ".join(m.group(0) for m in _DATE_PHRASE.finditer(text))))
    product = []
    for t in content_tokens(_TERM_BOUND.sub(" ", undated)):          # "less than 60 months" names no product
        if t in _MEASURE or t in date_words or t in excluded or t == "non" or _UNIT_WORDS.fullmatch(t):
            continue
        if t not in product:
            product.append(t)

    words = tokens(text, keep_stop=True)
    metrics = {_MEASURE[w] for w in tokens(_DATE_PHRASE.sub(" ", text)) if w in _MEASURE}      # "in Apr 2026" asks no APR
    if not re.search(r"\bpoints\b", text):
        metrics.discard("points")                       # only "points" asks them: "Harbor Point Road" is a street
    if re.search(r"\bmin(?:imum)?\b", text) and re.search(r"open|deposit|balance|earn|required|need", text):
        metrics.add("min")
    # A product word directly before a head noun names a specific kind: "construction loan". A word
    # after "to" is a verb: "to start saving" names no kind of savings.
    after = ["", *words]
    modifiers = {a for z, a, b in zip(after, words, words[1:]) if b in HEAD_NOUNS and a in product and z != "to"}
    modifiers |= {a for z, a, b, c in zip(after, words, words[1:], words[2:])
                  if c in HEAD_NOUNS and a in product and (b in product or b in _MEASURE) and z != "to"}

    # "at least 24 months" is a bound, not the least of anything.
    superlative = "max" if re.search(r"\b(highest|maximum|max|best|top|biggest|largest|(?<!at )most)\b", text) else \
        "min" if re.search(r"\b(lowest|cheapest|smallest|(?<!at )least)\b", text) else None
    wants_link = bool(
        re.search(r"\b(link|url|web address|which (?:web ?)?page|what (?:web ?)?page|"
                  r"(?:what|which)(?:'s| is)? (?:the |your )?web ?site)\b", text)
        and not re.search(r"\blink(?:ing|ed)?\b (?:my|an|a|the|your|up|to)\b", text)
        and not re.search(r"\b[A-Z][a-z]+ Link\b", question))                 # a name: "Dana Link" is a person
    wants_number = bool(metrics) or bool(
        re.search(r"how much|what does .* cost|what do .* cost|\bnumber\b|\bearn\b|\bpays?\b", text))
    return Query(text=text, product=product, quals=quals, metrics=metrics,
                 business="business" in words, superlative=superlative,      # "businesses" too
                 wants_number=wants_number and not wants_link, wants_link=wants_link,
                 modifiers=modifiers, excluded=excluded, terms=terms)


def _asked_amount(text: str) -> Optional[tuple]:
    """The balance or loan amount in the question, as a range: '$12,000', '12k', 'between $3,000 and $11,999'."""
    scrub = _NOT_AN_AMOUNT.sub(" ", text)
    amounts = []
    for m in _MONEY.finditer(scrub):
        if m.group(1):
            v, unit = float(m.group(1).replace(",", "") + (m.group(2) or "")), (m.group(3) or "")
        elif m.group(4):
            v, unit = float(m.group(4).replace(",", "") + (m.group(5) or "")), m.group(6)
        elif m.group(7):
            v, unit = float(m.group(7).replace(",", "")), ""
        else:
            v, unit = float(m.group(8).replace(",", "") + (m.group(9) or "")), ""
        amounts.append((m.start(), scale(v, unit)))
    if not amounts:
        bare = list(_BARE_AMOUNT.finditer(scrub))
        bare = [m for m in bare if m.group(0).startswith("borrow")] or bare      # "have 8000 ... borrow 20000"
        amounts = [(m.start(1), float(m.group(1))) for m in bare[:1]]
    if not amounts:
        return None
    if len(amounts) > 1 and not re.search(r"between|from .* to ", scrub):
        # "the $150 offer ... on a $12,000 balance": the amount next to a balance word is the tier, and
        # one that the word follows ("a $12,000 balance") before one that follows a word ("with the $150").
        named = [a for a in amounts if re.match(r"\S+\s+(balance|deposit|loan|mortgage)", scrub[a[0]:a[0] + 30])]
        near = [a for a in amounts
                if re.search(r"(balance|deposit|loan|borrow|invest|sav|put|park|have|with|of)\w*\W+(\w+\W+){0,2}$",
                             scrub[max(0, a[0] - 30):a[0]])]
        amounts = (named or near)[:1] or [max(amounts, key=lambda a: a[1])]
    at, value = amounts[0]
    before = scrub[max(0, at - 24):at]
    if len(amounts) >= 2 and re.search(r"between|from|-|to\b", scrub[at:amounts[1][0] + 1] + before):
        return tuple(sorted((value, amounts[1][1])))
    if re.search(r"(over|above|more than|exceed\w*|greater than)\s*\$?$", before):
        value += 0.01
    elif re.search(r"(under|below|less than)\s*\$?$", before):
        value -= 0.01
    return (value, value)


# --------------------------------------------------------------------------- dates


def printed(row: Row) -> bool:
    """The row's effective date is printed: on its page, or on another download of the same table."""
    return bool(row.effective) and row.dated_by != "change"


def newer(a: Row, b: Row) -> int:
    """Whether the dates show which copy is newer. Positive: a; 0: they do not tell.

    A printed date says when the rates took effect. A change date (see Store) says only that the
    table changed after that download, so it is a lower bound. Two dates of the same kind compare
    as dates. A change date beats an older printed date: the change came after that date. A newer
    printed date does not beat a change date, because the change may have come later still; then
    the rules after the date decide."""
    if not (a.effective and b.effective) or a.effective == b.effective:
        return 0
    if printed(a) == printed(b):
        return 1 if a.effective > b.effective else -1
    change = b if printed(a) else a
    other = a if change is b else b
    if change.effective > other.effective:
        return 1 if change is a else -1
    return 0


# --------------------------------------------------------------------------- the index

_FIELD_WEIGHT = {"name": 1.0, "label": 1.0, "section": 0.5, "page": 0.5}


def _page_tokens(page) -> frozenset:
    tail = " ".join(page.key.split("/")[1:]).replace("-", " ")
    return frozenset(content_tokens(page.title + " " + tail))


class RowIndex:
    """All rows of one view, grouped into slots (copies of the same row), and which pages are behind."""

    def __init__(self, view) -> None:
        self.view = view
        self.rows: list[Row] = view.rows
        self.page_tokens = {id(p): _page_tokens(p) for p in view.all_pages()}
        # The words of the rows' own names and titles; with their sections' and pages', every word the rows know.
        self.vocabulary = vocabulary = {t for r in self.rows for t in r.name_tokens | r.label_tokens}
        self._known = vocabulary | {t for r in self.rows for t in r.section_tokens | self.page_tokens[id(r.page)]}
        self._words = sorted(t for t in vocabulary if len(t) >= 5 and not t.isdigit())
        self.slots: dict[tuple, list[Row]] = defaultdict(list)
        # Slots that some table prints side by side: "Auto Loan" and "Auto Loan Intro Rate" in one table
        # are two different facts, never an old and a new edition of one.
        self.beside: dict[tuple, set] = defaultdict(set)
        together: dict[tuple, set] = defaultdict(set)
        for r in self.rows:
            self.slots[r.slot].append(r)
            if r.kind == "table":
                together[(r.page.key, r.table)].add(r.slot)
        for slots in together.values():
            for slot in slots:
                self.beside[slot] |= slots - {slot}
        # Product names that contain a measure word in the middle: "Gold Rate Savings". Only between two
        # words of a name: in a fee line ("Courier fee for a lost card", "Coin counting fee 2% of the total")
        # the measure is what the row states, and a question that uses it is asking for it.
        self.named_measures = set()
        for name in {r.name for r in self.rows} | {r.label for r in self.rows}:
            words = tokens(name, keep_stop=True)
            self.named_measures.update((a, b, c) for a, b, c in zip(words, words[1:], words[2:])
                                       if b in _MEASURE and not {a, c} & STOPWORDS and not (a.isdigit() or c.isdigit()))
        self._mark_stale_pages()
        self._sides: dict[int, set] = {}

    def side(self, row: Row) -> set:
        """Whether a row is about lending or saving (see _lending_or_saving): its measure, its words and its titles'."""
        if id(row) not in self._sides:
            self._sides[id(row)] = _lending_or_saving(set(tokens(f"{row.name} {row.label} {row.section}")),
                                                     {v.metric for v in row.values})
        return self._sides[id(row)]

    def _mark_stale_pages(self) -> None:
        """A page that prints an older copy of a table that is newer elsewhere has not been maintained.
        "Older" is judged by the effective dates, as newer() compares them. Equal values under an
        older date count only when the date is more than a year behind: a table still labelled 2024
        next to the same table labelled 2026 was not looked at since."""
        for _ in range(2):                    # the second pass uses the trust levels the first one set
            wins: dict[str, int] = defaultdict(int)
            losses: dict[str, int] = defaultdict(int)
            for rows in self.slots.values():
                rows = [r for r in rows if r.kind == "table"]
                if len({r.page.key for r in rows}) < 2:
                    continue
                if len({r.signature() for r in rows}) < 2 and len({r.effective for r in rows}) < 2:
                    continue
                best = self.winner(rows)
                for r in rows:
                    behind = newer(best, r) > 0
                    same = r.signature() == best.signature()
                    # A year apart is measured on printed dates only: a change date is a lower bound.
                    if behind and (not same or (printed(r) and printed(best)
                                                and (best.effective - r.effective).days > 365)):
                        losses[r.page.key] += 1
                    elif same:
                        wins[r.page.key] += 1
            self.view.stale = {k for k, n in losses.items() if n > wins.get(k, 0)}

    def expired(self, row: Row) -> bool:
        """The row is a promotion that had ended by this view's date."""
        return bool(row.expired and row.expired <= self.view.taken_on)

    def better(self, a: Row, b: Row, support: dict) -> int:
        """Which of two copies of the same row is current? Positive: a. The order is the rule."""
        if self.expired(a) != self.expired(b):
            return -1 if self.expired(a) else 1                    # a promotion that has ended loses
        trust_a, trust_b = self.view.authority(a.page), self.view.authority(b.page)
        if trust_a != trust_b:
            return 1 if trust_a > trust_b else -1                  # the more trusted kind of page wins
        dates = newer(a, b)
        if dates:
            return dates                                           # the newer "effective as of" date wins
        if a.maintained != b.maintained:
            return 1 if a.maintained else -1                       # a copy the page keeps updating beats a frozen one
        pages_a, pages_b = len(support[a.signature()]), len(support[b.signature()])
        if pages_a != pages_b:
            return 1 if pages_a > pages_b else -1                  # the value shown on more pages wins
        if printed(a) != printed(b):
            return 1 if printed(a) else -1                         # a copy with a printed date beats one without
        # `fetched` is when this exact text was first downloaded, so this rule prefers the copy whose
        # page changed most recently; every page in a view was downloaded for it.
        if a.page.fetched != b.page.fetched:
            return 1 if a.page.fetched > b.page.fetched else -1    # the more recently downloaded page wins
        if a.kind != b.kind:
            return (a.kind == "table") - (b.kind == "table")       # a table row beats any other kind
        # Nothing the site says tells them apart. The page nearer the top of the site (the shorter address:
        # "rates" before "rates/guide/wren") is its own general statement, and the address does not depend
        # on the order the pages were downloaded in, as "the first one listed" would.
        where_a, where_b = (len(a.page.key), a.page.key), (len(b.page.key), b.page.key)
        return (where_a < where_b) - (where_a > where_b)

    def generation(self, a: Row, b: Row) -> int:
        """Compare two rows as editions of a table: trust, then date, then upkeep. 0 = same edition."""
        # The first three rules of better() (ended, trust, date), then upkeep unless both copies print a date.
        if self.expired(a) != self.expired(b):
            return -1 if self.expired(a) else 1
        trust_a, trust_b = self.view.authority(a.page), self.view.authority(b.page)
        if trust_a != trust_b:
            return 1 if trust_a > trust_b else -1
        dates = newer(a, b)
        if dates:
            return dates
        if a.maintained != b.maintained and not (printed(a) and printed(b)):
            return 1 if a.maintained else -1
        return 0

    @staticmethod
    def support(rows: list[Row]) -> dict:
        """The pages that show each value, for "the value shown on more pages wins". A copy that a
        newer date has beaten is out of date and counts for no page: otherwise an undated copy
        that repeats an old edition borrows that edition's page and beats the newest dated copy."""
        out: dict[tuple, set] = defaultdict(set)
        for r in rows:
            if not any(newer(o, r) > 0 for o in rows):
                out[r.signature()].add(r.page.key)
        return out

    def winner(self, rows: list[Row]) -> Row:
        """The current copy, whatever order the pages were downloaded in.

        better() compares dates only when both copies have one, so with dated and undated copies it
        can go round in a circle: a frozen copy dated April 22 beats a kept-up copy dated April 2,
        which beats a kept-up undated copy, which beats the frozen one. So the copies with a printed
        date are ranked among themselves, the others (undated, or dated only by a change) among
        themselves, and the winners of the two groups then meet."""
        support = self.support(rows)
        best = None
        for group in ([r for r in rows if printed(r)], [r for r in rows if not printed(r)]):
            if not group:
                continue
            top = group[0]
            for r in group[1:]:
                if self.better(r, top, support) > 0:
                    top = r
            if best is None or self.better(top, best, support) > 0:
                best = top
        return best

    def spell(self, token: str) -> str:
        """A question word the site never uses, one slip away from a word the rows do use: 'yeild' -> 'yield'."""
        if token in self._known or len(token) < 5 or token.isdigit():
            return token
        passages = self.view.cache.get("passages")
        if passages is not None and passages.known(token):
            return token                      # a real word on this site: leave it alone
        close = difflib.get_close_matches(token, self._words, n=1, cutoff=0.82)
        return close[0] if close else token

    def score(self, q: Query, row: Row) -> tuple[float, set]:
        """How many of the question's product words this row accounts for, and which.
        A word in the row's own name or its table's title counts 1; in the section or page title, half."""
        matched, total = set(), 0.0
        page = self.page_tokens[id(row.page)]
        for t in q.product:
            if t in row.name_tokens:
                w = _FIELD_WEIGHT["name"]
            elif t in row.label_tokens:
                w = _FIELD_WEIGHT["label"]
            elif t in row.section_tokens:
                w = _FIELD_WEIGHT["section"]
            elif t in page:
                w = _FIELD_WEIGHT["page"]
            else:
                continue
            matched.add(t)
            total += w * (0.3 if row.kind == "claim" else 1.0)     # a rate mentioned in a sentence is weak evidence
        return total, matched


# --------------------------------------------------------------------------- matching


def _fits(q: Query, row: Row) -> Optional[bool]:
    """True: the qualifiers asked for fit this row. False: this row is another tier. None: nothing to compare."""
    compared = False
    if (q.quals.term is not None or q.terms is not None) and row.quals.term is not None:
        lo, hi = q.terms or (q.quals.term, q.quals.term)
        if not lo <= row.quals.term <= hi:
            return False
        compared = True
    if q.quals.years is not None and row.quals.years is not None:
        lo, hi = row.quals.years
        if not lo <= q.quals.years[0] <= hi:
            return False
        compared = True
    if q.quals.amount is not None and row.quals.amount is not None:
        lo, hi = row.quals.amount
        a, b = q.quals.amount
        if not (lo - 0.005 <= a <= hi + 0.005 and (b == a or lo - 0.005 <= b <= hi + 1.0)):
            return False
        compared = True
    elif q.quals.amount is not None:
        # No balance tier, but a stated minimum: $5,000 does not buy a certificate that needs $40,000.
        least = _minimum(row)
        if least is not None:
            if q.quals.amount[1] < least:
                return False
            compared = True
    return True if compared else None


def _minimum(row: Row) -> Optional[float]:
    """The deposit a row asks for ("Minimum balance: $50,000"), when it states one."""
    found = [v.num for v in row.values if v.metric in ("min_earn", "min_open") and v.num]
    return max(found) if found else None


def _tier_reached(q: Query, rows: list[Row]) -> list[Row]:
    """$60,000 meets the $500 minimum of a 9-month special and the $50,000 minimum of the 9-month
    jumbo special: it buys the jumbo. Of one product at one term, the highest minimum the whole
    amount reaches is its tier, and a row with a lower minimum is another tier."""
    if q.quals.amount is None:
        return rows

    def outranked(r: Row) -> bool:
        least = _minimum(r)
        return least is not None and r.quals.amount is None and any(
            o.quals == r.quals and family(r, o) and least < (_minimum(o) or 0.0) <= q.quals.amount[0] for o in rows)
    return [r for r in rows if not outranked(r)]


_DIGITS = re.compile(r"\d+(?:\.\d+)?")


def _sized(q: Query, rows: list[Row]) -> list[Row]:
    """Rows that read alike but for the numbers in their names ("Storage locker 2 X 4 X 18" and
    "Storage locker 4 X 4 X 18") are told apart by those numbers: a question that gives them in
    that order ("a 2 x 4 locker") is about the row whose name carries them. Rows that read alike
    but for a small word ("Wire in", "Wire out") are told apart by that word."""
    undated = _DATE_PHRASE.sub(" ", q.text)                  # "in January 2026" holds no word of a row
    asked = _DIGITS.findall(_YEAR_CONTEXT.sub(" ", undated))
    alike: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        alike[r.slot[:-1]].append(r)                     # the slot without the variant number

    def says(r: Row, nums: list) -> bool:
        have = _DIGITS.findall(r.name)
        return any(have[i:i + len(nums)] == nums for i in range(len(have) - len(nums) + 1))
    out = []
    for r in rows:
        group = alike[r.slot[:-1]]
        nums = [n for n in asked if any(n in _DIGITS.findall(o.name) for o in group)]
        if len(group) > 1 and nums and any(says(o, nums) for o in group) and not says(r, nums):
            continue
        out.append(r)
    if q.excluded:
        return out                                       # "a check with no stub": its "with" tells nothing apart
    said = set(tokens(undated, keep_stop=True))
    words = {id(r): {w for w in tokens(r.name, keep_stop=True) if not w.isdigit()} for r in out}
    kept = []
    for r in out:
        group = [words[id(o)] for o in alike[r.slot[:-1]] if id(o) in words]
        told = set.union(*group) - set.intersection(*group)          # the words not every one of them has
        if len(told & said & words[id(r)]) < max(len(told & said & w) for w in group):
            continue
        kept.append(r)
    return kept


@dataclass
class Match:
    """What find() returns: the current rows, the pages behind them, and the copies they beat."""
    rows: list                                            # the current rows that answer the question
    sources: dict = field(default_factory=dict)           # id(row) -> pages that print the same row
    rivals: list = field(default_factory=list)            # (winning row, losing row) pairs
    matched: set = field(default_factory=set)             # question words every returned row accounts for
    unmatched: set = field(default_factory=set)
    covered: set = field(default_factory=set)             # question words some returned row accounts for
    tier_missing: bool = False                            # the product is listed, but not with this term or tier


def find(index: RowIndex, q: Query) -> Optional[Match]:
    """The current rows that answer q, or None when it names no product or no row fits. First q is
    completed against the rows (_settle_query: the spelling, measure words inside product names, a
    business or a general row); then every row is scored, the best are kept and resolved to their
    current copies. A Match with no rows and tier_missing means the product is listed, but not with
    the term or tier asked."""
    if not q.product:
        return None
    _settle_query(index, q)

    # A question that names a family only through its page or section ("what credit cards do you
    # offer") is about everything on offer, business products included.
    personal_best = max((index.score(q, r)[0] for r in index.rows if not r.business), default=0.0)
    family_wide = not q.business and 0 < personal_best < 1.0
    # A loan's rate is not a certificate's of the same name, nor the other way round. Only the nouns say which
    # is asked about: "the Gold card APY" asks a card's rate in the wrong word. A fee is not so bound: the
    # fee for paying by card is charged on a loan payment too.
    side = _lending_or_saving(set(tokens(q.text))) if q.metrics & RATES else set()

    # A question that asks only a rate is about rows that state one: a car for sale is no tier of a loan,
    # though its model year and make match.
    only_rate = bool(q.metrics) and q.metrics <= RATES

    def not_asked(row: Row) -> bool:
        return only_rate and not any(v.metric in RATE_METRICS for v in row.values) \
            or len(side) == 1 and len(index.side(row)) == 1 and side != index.side(row)
    scored, other_tier = [], []
    for row in index.rows:
        if row.business != q.business and not (family_wide and row.business):
            continue
        s, matched = index.score(q, row)
        if s > 0 and not_asked(row):
            continue
        if row.business and not q.business:
            matched, s = matched - {"business"}, s - (1.0 if "business" in matched else 0.0)
        if s <= 0:
            continue
        fit = _fits(q, row)
        if fit is False:
            other_tier.append((s, row))
        else:
            scored.append((s, matched, row, fit))
    # The kind of product named ("jumbo mortgage" is a mortgage) must be what the row is: a jumbo
    # certificate does not answer a question about a jumbo mortgage.
    heads = {t for t in q.product if t in HEAD_NOUNS and t in index._known}
    if heads:
        scored = [x for x in scored if x[1] & heads]
        other_tier = [(s, r) for s, r in other_tier if index.score(q, r)[1] & heads]
    if not scored:
        return Match(rows=[], tier_missing=True) if other_tier else None

    if q.excluded:
        scored = [x for x in scored if not q.excluded & (x[2].name_tokens | x[2].label_tokens)] or scored
    best = max(s for s, _, _, _ in scored)
    # A question that gives a term, amount or model year is about rows that carry one: those
    # come first even when another row happens to share one more word with the question.
    fitting = [x for x in scored if x[3]]
    if fitting and max(s for s, _, _, _ in fitting) >= 0.6 * best:
        scored = fitting
        best = max(s for s, _, _, _ in scored)
    top = [x for x in scored if x[0] >= best * 0.98]
    # A row for another tier that matches the product better means the asked tier is not published.
    if any(s > best * 1.02 for s, _ in other_tier) and not any(fit for _, _, _, fit in top):
        return Match(rows=[], tier_missing=True)

    # Rows that tie on score but match different words answer different questions: keep only
    # what every one of them explains, so a low share sends the question on to be read.
    matched = set.intersection(*(set(m) for _, m, _, _ in top))
    # Every edition of the chosen tables competes, not only the copy that scored best: a stale
    # copy must not win because its page happens to share an extra word with the question.
    chosen = [r for _, _, r, _ in top]
    beside = set().union(*(index.beside.get(r.slot, set()) for r in chosen))
    editions = [o for o in index.rows if o not in chosen and o.slot not in beside and _fits(q, o) is not False
                and not q.excluded & (o.name_tokens | o.label_tokens) and not not_asked(o)
                and any(family(r, o) for r in chosen)]
    rows, sources, rivals = resolve(index, chosen + editions)
    rows = _sized(q, _tier_reached(q, rows))
    if not rows:
        return None
    rows.sort(key=lambda r: (r.business, r.page.key, r.order))
    # The rows that survive must themselves account for the kind of product asked about: a
    # plain "Auto Loan" row that outlived an "Auto Loan Intro Rate" row does not answer for "intro".
    lost = {t for t in q.modifiers & matched if any(t not in index.score(q, r)[1] for r in rows)}
    matched -= lost
    return Match(rows=rows, sources=sources, rivals=rivals, matched=matched, unmatched=set(q.product) - matched,
                 covered=set().union(*(index.score(q, r)[1] for r in rows)))


def _settle_query(index: RowIndex, q: Query) -> None:
    """Complete q against the rows, in place: its words joined and spelled as the rows write them, a
    describing word dropped, a measure word that is part of a product's name not read as the measure
    asked, and whether a question that says "business" is about a business row or about a fee every
    member pays. All but the last are done once per query (q.spelled)."""
    if not q.spelled:
        # Two words of the question that the rows write as one are that word: "tree house savings" and
        # "tree-house savings" ask for "Treehouse Savings".
        words = tokens(q.text, keep_stop=True)
        for a, b in zip(words, words[1:]):
            if a.isalpha() and b.isalpha() and a + b in index.vocabulary and content_tokens(a + b) == [a + b]:
                q.product = [a + b if t in (a, b) else t for t in q.product]
                q.modifiers = {a + b if t in (a, b) else t for t in q.modifiers}
        q.product = list(dict.fromkeys(index.spell(t) for t in q.product))
        # A describing word ("a regular credit card", text.DESCRIBING) names no kind of product.
        q.modifiers = {index.spell(t) for t in q.modifiers if t not in DESCRIBING}
        # A measure word inside a product's name is not the measure asked: "gold rate savings hit me with
        # a fee" asks a fee.
        inside = {i + 1 for i, trio in enumerate(zip(words, words[1:], words[2:])) if trio in index.named_measures}
        asked = {_MEASURE[w] for i, w in enumerate(words) if w in _MEASURE and i not in inside}
        if inside:
            q.metrics = (q.metrics - set(_MEASURE.values())) | asked | (q.metrics & {"min"})
            # As parse_question decides it, less its "what does ... cost": "cost" is a measure word,
            # counted in asked above unless it is part of a product's name.
            q.wants_number = not q.wants_link and (
                bool(q.metrics) or bool(re.search(r"how much|\bnumber\b|\bearn\b|\bpays?\b", q.text)))
        q.spelled = True

    # "The night drop key fee for businesses": a fee the schedule charges every member has no business
    # row. When a general row names more of what is asked than any business row, the question is
    # about that fee. A rate is never borrowed this way: a business product's rate is its own.
    if q.business and not q.metrics & RATES:
        plain = replace(q, product=[t for t in q.product if t != "business"])
        best = {b: max((index.score(plain, r)[0] for r in index.rows if r.business == b), default=0.0)
                for b in (False, True)}
        q.business = best[True] >= best[False]


def family(a: Row, b: Row) -> bool:
    """Two rows that are tiers of the same product, possibly under different tier schemes.

    'Camper Loans, 2019-22 Model Year' and 'Camper & Trailer Loans, 2020-23 Model Year' are one
    family: one set of product words contains the other, and both are keyed by the same kinds
    of qualifier. 'Promo Certificates' and 'Classic Certificates' are not.
    """
    if a.business != b.business:
        return False
    if (a.quals.term is None, a.quals.amount is None, a.quals.years is None) != \
            (b.quals.term is None, b.quals.amount is None, b.quals.years is None):
        return False
    words_a, words_b = a.name_tokens | a.label_tokens, b.name_tokens | b.label_tokens
    return bool(words_a) and bool(words_b) and (words_a <= words_b or words_b <= words_a)


def resolve(index: RowIndex, rows: list[Row]) -> tuple[list[Row], dict, list]:
    """Reduce matching rows to the current ones. Three steps:

    1. Copies of the same row (same slot) compete under RowIndex.better.
    2. Editions of the same table compete: when a page shows a product's tiers under an older
       scheme (other model years, an older effective date), the newer edition speaks alone.
    3. The most trusted kind of page speaks alone: a calculator's table is dropped when a
       product page has one too.
    Returns the surviving rows, the pages that print each of them, and (winner, loser) pairs.
    """
    by_slot: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        by_slot[r.slot].append(r)
    winners, sources, rivals = [], {}, []
    for slot, members in by_slot.items():
        copies = index.slots.get(slot, members)
        win = index.winner(copies)
        if index.expired(win):
            continue
        winners.append(win)
        sources[id(win)] = _pages_agreeing(index, win, copies)
        rivals.extend((win, c) for c in copies if c.signature() != win.signature())
    if not winners:
        return [], {}, []

    kept = []
    for r in winners:
        beaten_by = [o for o in winners if (o.page.key, o.table) != (r.page.key, r.table)
                     and family(r, o) and index.generation(o, r) > 0]
        if beaten_by:
            rivals.append((beaten_by[0], r))
        else:
            kept.append(r)
    top_trust = max(index.view.authority(r.page) for r in kept)
    trusted = [r for r in kept if index.view.authority(r.page) >= top_trust]
    rivals.extend((trusted[0], r) for r in kept if r not in trusted)
    trusted.sort(key=lambda r: (r.page.key, r.order))

    # The same rows printed on two pages (or under two titles) are one answer.
    unique: list[Row] = []
    for r in trusted:
        ident = (r.name_tokens, r.quals, r.business, r.signature())
        twin = next((u for u in unique if (u.name_tokens, u.quals, u.business, u.signature()) == ident
                     and (u.slot == r.slot or family(u, r))), None)
        if twin is None:
            unique.append(r)
        else:
            known = sources.setdefault(id(twin), [])
            known.extend(p for p in sources.get(id(r), [r.page]) if p not in known)
    return unique, sources, rivals


def nearby_rows(index: RowIndex, q: Query, limit: int = 36) -> tuple[list[Row], dict]:
    """Evidence for a reader: every tier of the tables closest to the question, current copies only,
    and the pages that print each of them, most trusted first (by id(row), as Match.sources).

    Unlike find(), other terms and tiers are kept, so the reader can see what is and is not
    published. Business rows are included; they are marked as such in the prompt. A calculator's
    figures illustrate and are not in force (see engine._rows_suffice), so its rows are left out.
    """
    if not q.product:
        return [], {}
    tables: dict[tuple, list] = defaultdict(list)
    best_of: dict[tuple, float] = defaultdict(float)
    for row in index.rows:
        if row.kind == "claim" or row.page.kind in (P.ARTICLE, P.INDEX, P.CALCULATOR):
            continue
        s, _ = index.score(q, row)
        if s <= 0:
            continue
        key = (row.page.key, row.table, row.label)
        tables[key].append(row)
        best_of[key] = max(best_of[key], s + (0.25 if _fits(q, row) else 0.0))
    if not tables:
        return [], {}
    top = max(best_of.values())
    chosen = [r for key, members in tables.items() if best_of[key] >= max(0.5, top * 0.6) for r in members]
    rows, sources, _ = resolve(index, chosen)
    rows.sort(key=lambda r: (-index.score(q, r)[0], r.business, r.page.key, r.order))
    return rows[:limit], sources


def _pages_agreeing(index: RowIndex, win: Row, copies: list[Row]) -> list:
    pages = sorted((c.page for c in copies if c.signature() == win.signature()),
                   key=lambda p: (-index.view.authority(p), p.key))
    return list({p.key: p for p in pages}.values())
