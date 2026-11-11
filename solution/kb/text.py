"""Text helpers shared by every module: cleaning, tokens, and the qualifiers that make two
numbers different facts (term, balance or loan amount, model year).

Everything here is deliberately plain: regular expressions and small word lists. The word
lists are general credit-union vocabulary ("CD" means certificate, "car" means auto), not
facts about any one website.

Also numbers in words, dates, and the site's own vocabulary (learn_names, learn_site_name),
learned at ingest and held process-wide: one site per process (forget_site clears it).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

# --------------------------------------------------------------------------- cleaning

_TRANSLATE = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",   # dashes
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',                                # quotes
    "\u00a0": " ", "\u202f": " ", "\u200b": "", "\u00ae": "", "\u2122": "",                    # spaces, marks
})
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")         # a markdown heading line: its level and its text
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")         # a markdown image


def clean(s: str) -> str:
    """Unify dashes, quotes and spaces so the same text always compares equal."""
    return re.sub(r"[ \t]+", " ", (s or "").translate(_TRANSLATE)).strip()


def strip_markdown(s: str) -> str:
    """Drop images, keep link text, drop emphasis marks and escapes."""
    s = IMAGE.sub(" ", s)
    s = re.sub(r"\[\d{1,2}\]\([^)]*\)", "", s)            # footnote links: [1](...#fine-print)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"[*_`]{1,3}", "", s)
    s = s.replace("\\", "")
    return clean(s)


# --------------------------------------------------------------------------- numbers in words

_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_NUMWORD = re.compile(
    r"\b(" + "|".join(_TENS) + r")(?:[ -](" + "|".join(list(_ONES)[1:10]) + r"))?\b|\b(" + "|".join(_ONES) + r")\b"
)


def _words_to_digits(s: str) -> str:
    """'sixty month' -> '60 month', 'thirty-six' -> '36'. Lower-case input expected."""
    def repl(m: re.Match) -> str:
        if m.group(1):
            return str(_TENS[m.group(1)] + (_ONES[m.group(2)] if m.group(2) else 0))
        word = m.group(3)
        # "one" is usually an article ("which one"), so only convert it next to a unit.
        if word == "one" and not re.match(r"[ -](month|year|yr|day|week)", s[m.end():m.end() + 7]):
            return word
        return str(_ONES[word])
    return _NUMWORD.sub(repl, s)


# --------------------------------------------------------------------------- vocabulary

# Phrases that mean the same product or measure. Applied before tokenising, on lower-case text.
_PHRASES = [
    (r"\bcertificates? of deposits?\b", "certificate"),
    (r"\b(?:term[- ]share|share|term) certificates?\b", "certificate"),
    (r"\bcds?\b", "certificate"),
    (r"\bhome equity lines?(?: of credit)?\b", "heloc"),
    (r"\bhelocs?\b", "heloc"),
    (r"\bhome loans?\b", "mortgage"),
    (r"\bmortgages?\b", "mortgage"),
    (r"\bmoney[- ]markets?\b|\bmmas?\b|\bmm(?=[- ](?:accounts?|accts?|savings)\b)", "moneymarket"),   # not "5 mm"
    (r"\bcredit[- ]cards?\b", "creditcard"),
    (r"\bdebit[- ]cards?\b", "debitcard"),
    (r"\bdrive[- ]?(?:ups?|thrus?|throughs?)\b", "driveup"),
    (r"\brecreational vehicles?\b", "rv"),
    (r"\b(?:cars?|vehicles?|automobiles?|autos?|trucks?|pickups?|suvs?|minivans?|sedans?)\b", "auto"),
    (r"\b(?:honda|toyota|ford|chevy|chevrolet|nissan|jeep|subaru|hyundai|kia|tesla|bmw|audi|volkswagen|vw|mazda|"
     r"dodge|ram|gmc|buick|lexus|acura|volvo|chrysler|cadillac|mercedes)\b", "auto"),          # "a used 2023 Honda"
    (r"\boverdr(?:aw|awn|aws|awing|ew|afts?|afting)\b", "overdraft"),
    (r"\bannual percentage yields?\b", "apy"),
    (r"\bannual percentage rates?\b", "apr"),
    (r"\bhealth savings accounts?\b", "hsa"),
    (r"\bhsas?\b", "hsa"),
    (r"\bindividual retirement accounts?\b", "ira"),
    (r"\biras?\b", "ira"),
    (r"\brefi(?:nanc(?:e|ing|ed))?\b", "refinance"),
    (r"\barms?\b", "adjustable"),
    (r"\b(\d+)/(\d+)\b", r"\1x\2"),          # "7/1 ARM" stays one token: 7x1
    (r"\be[- ]?statements?\b", "estatement"),
    (r"\bmemberships?\b", "member"),
    (r"\baba(?: routing)?(?: transit)?(?: number| #| no\.?)?(?=\W|$)|\brouting (?:#|no\.?|num\b)", "routing number"),
    (r"\bhigh[- ]y(?:ie|ei)ld\b", "highyield"),      # a product name, not a request for a yield (and its usual typo)
    (r"\b(?:introductory|intro|promotional|promo|limited[- ]time)\b", "intro"),
    (r"\baccts?\b", "account"),
    # "a fixed-rate mortgage" names a product, not a rate question; "the 30-year fixed rate" asks a rate.
    (r"\b(fixed|variable|adjustable)[- ]rates?(?=[- ](?:mortgage|loan|heloc|home|equity|certificate|creditcard|"
     r"account|option|product|plan)s?\b)", r"\1"),
    (r"\bmodel-years?\b", "model year"),             # a qualifier (the year is read from the question), not a name
    (r"\bbiz\b|\bcommercial\b|\bsmall[- ]business\b", "business"),
    (r"\b(?:get|getting|keep|stay) in touch\b|\breach (?:out|you|us|someone)\b|\bspeak (?:to|with)\b|\btalk to\b", "contact"),
    (r"\bco[- ]?op\b", "coop"),
    (r"\b(?:a\.?t\.?m\.?s?)\b", "atm"),
    (r"\bzip ?codes?\b", "zipcode"),
    (r"\bsign[- ]?up\b", "signup"),
    (r"\b1st\b", "first"),
    (r"\bhrs\b", "hours"),
    (r"\borgs\b", "organizations"),
    (r"\bstaff members?\b|\bstaff\b|\bteam members?\b|\bworkers?\b"
     r"|\b(?:people|folks|persons) (?:who )?work(?:s|ing)?(?= (?:at|for|there|here)\b)", "employee"),
    (r"\breporters?\b|\bjournalists?\b|\bthe press\b(?= *(?:[?.!,;]|$)| (?:about|for|with)\b)"
     r"|\bpress(?= (?:inquir|contacts?\b|requests?\b))", "media"),
    # Joining the credit union, not an account or a programme: "who can join?", "can I join Birch?",
    # "qualify for membership" (already "member"); not "can someone join my account", "membership rewards",
    # "join the credit union board".
    (r"\b(?:qualify to|who (?:can|may)|can (?:i|someone|somebody|anyone)) join(?=(?: \w+)? *(?:[?.!,;]|$)"
     r"|(?: \w+){0,3} credit union\b(?= *(?:[?.!,;]|$))| as an? member\b)"
     r"|\bqualify for member(?= *(?:[?.!,;]|$)| (?:at|with|in)\b)",
     "eligible to join"),
    # A bank check is the thing bought, not the bank checking something: "a bank check", "bank check fee";
    # not "will the bank check my credit".
    (r"\b(?:cashier|teller)'?s? checks?\b|(?:(?<=\ba )|(?<=\ban ))bank check\b"
     r"|\bbank checks?(?= (?:fees?|costs?|prices?|charges?)\b| *(?:[?.!,;]|$))", "official check"),
    (r"\boverseas\b|\babroad\b|\bout of the country\b", "international"),
    (r"\b(?:service|maintenance) (?:charges?|fees?)\b", "fee"),
]
_PHRASES = [(re.compile(p), r) for p, r in _PHRASES]

STOPWORDS = frozenset("""
a an the and or but if of to in on at by for from with without about as into than then that this these those
is are was were be been being am do does did doing have has had having can could will would shall should may might must
i me my mine we us our ours you your yours he she it its they them their there here what which who whom whose when where why how
tell give show get got find know say see let need want like please thanks thank hi hello hey just also still really
any some all each every both either neither much many more most other another such own same so too very not no nor only
up down out off over under again further once s t d ll re ve m o y
whats hows wheres whos whens im ive id dont doesnt didnt cant couldnt wont wouldnt isnt arent u ur r rn thing things stuff
""".split())

# Words that describe the question, not the thing asked about.
_GENERIC = frozenset("""
rate rates current currently today now right present latest new offer offers offered offering available
account accounts loan loans option options kind kinds type types product products service services
credit union cu member members membership bank banking info information detail details question
interest percent percentage number amount value much cost costs
earn earns earning pay pays paid paying charge charges charged provide provides get gets give gives make makes take takes
use uses using open opens opening apply applies qualify borrow deposit deposits keep hold put receive tell
balance balances between within tier tiers level levels range model typical typically general generally usual usually
term terms length long average recent recently
highest lowest maximum minimum max min best top cheapest smallest most least higher lower
dollar dollars money funds fund
back putting send sending moment pls plz holding charging offering looking wondering thinking trying going want wanted
quote quoted quoting listed list show showing advertised advertise published publish posted post
""".split())

# Plain English words of quality or size: "a regular credit card", "basic checking", "a small loan"
# describe the product asked about and name no kind of it (see lookup._settle_query).
DESCRIBING = frozenset("regular normal standard basic plain ordinary typical usual simple small big large little cheap "
                       "general".split())


# Names learned from the site itself (see learn_names): applied like the phrases above.
_LEARNED: list[tuple[re.Pattern, str]] = []
_NAME = re.compile(r"(?<=[a-z,] )([A-Z][a-z]+) ([A-Z][a-z]+)(?=[ ,.;:!?)]|$)")


def learn_names(texts, min_pages: int = 5) -> None:
    """A two-word name the site writes in capitals in the middle of sentences, made only of words
    that are otherwise ignored, is kept as one word. A programme called "Give More" would vanish
    from every question ('give' and 'more' say nothing on their own); as 'givemore' it is findable.
    One of the two must be a stop word: two generic words ("Product Details", "Rate Details") form
    an ordinary phrase, and joining them would hide both words from every question."""
    seen: dict[tuple, int] = {}
    for text in texts:
        found = set()
        for m in _NAME.finditer(strip_markdown(text)):
            pair = (m.group(1).lower(), m.group(2).lower())
            if all(w in STOPWORDS or w in _GENERIC or _stem(w) in _GENERIC for w in pair) and any(w in STOPWORDS for w in pair):
                found.add(pair)
        for pair in found:
            seen[pair] = seen.get(pair, 0) + 1
    known = {rep for _, rep in _LEARNED}
    for (a, b), n in sorted(seen.items()):
        if n >= min_pages and a + b not in known:
            _LEARNED.append((re.compile(rf"\b{re.escape(a)} {re.escape(b)}\b"), a + b))


def canon(s: str) -> str:
    """Lower-case, clean, turn number words into digits and fold synonyms."""
    s = clean(strip_markdown(s)).lower()
    s = re.sub(r"(?<=[a-z])\$(?=[a-z])", "s", s)          # brand spellings: "Kid$ave" is "kidsave"
    s = _words_to_digits(s)
    for rx, rep in _PHRASES:
        s = rx.sub(rep, s)
    for rx, rep in _LEARNED:
        s = rx.sub(rep, s)
    return s


def _stem(w: str) -> str:
    """A plain-English plural made singular: 'rates' -> 'rate', 'branches' -> 'branch'."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokens(s: str, keep_stop: bool = False) -> list[str]:
    """Canonical word tokens. Numbers are kept (they matter: '19 month', '2025')."""
    out = []
    text = canon(s)
    for w in _WORD.findall(text):
        w = w.split("'")[0]
        if not w or (not keep_stop and w in STOPWORDS):
            continue
        out.append(_stem(w))
    # A hyphenated word is also findable written as one: "Sun-Mart" as "sunmart", "e-receipt" as "ereceipt".
    for a, b in re.findall(r"\b([a-z]{1,12})-([a-z]{2,12})\b", text):
        out.append(_stem(a + b))
    return out


# The organisation's own name ("acme" for acme.com) says nothing about which product is meant.
# It is learned from the address of the pages when they are ingested, not written down here.
SITE_NAMES: set[str] = set()


def forget_site() -> None:
    """Drop the names learned from another site: they belong to the process, not to one system,
    and a second site read in the same process must not ignore the first one's name."""
    SITE_NAMES.clear()
    _LEARNED.clear()


def learn_site_name(host: str) -> None:
    """'www.acme.com' -> 'acme' becomes a word that names nothing in particular."""
    labels = [x for x in host.lower().split(":")[0].split(".") if x and x != "www"]
    if len(labels) >= 2 and len(labels[-2]) > 2:
        SITE_NAMES.add(_stem(labels[-2]))


def content_tokens(s: str) -> list[str]:
    """Tokens that can name a product or a thing: no stopwords, no generic words, no bare numbers."""
    return [t for t in tokens(s) if t not in _GENERIC and t not in SITE_NAMES and not t.isdigit()]


# --------------------------------------------------------------------------- values

_PCT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_DOLLARS = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")


def fix_footnotes(s: str) -> str:
    """Tables print footnote marks glued to the value: '0.42%1', '6.84% APR1'. Remove them."""
    s = re.sub(r"(%)\s?\d{1,2}(?=\s|$|[),;|])", r"\1", s)
    s = re.sub(r"\b(APR|APY)\*?\d{1,2}\b", r"\1", s)
    return s


def percent_of(s: str) -> Optional[float]:
    """The first percentage in the text: '2.50% APY' -> 2.5."""
    m = _PCT.search(s or "")
    return float(m.group(1)) if m else None


def money_of(s: str) -> Optional[float]:
    """The first dollar amount in the text: '$1,234.50' -> 1234.5."""
    m = _DOLLARS.search(s or "")
    return float(m.group(1).replace(",", "") + (m.group(2) or "")) if m else None


# --------------------------------------------------------------------------- qualifiers

INF = math.inf


@dataclass(frozen=True)
class Quals:
    """What makes one row of a rate table different from its neighbours."""
    term: Optional[int] = None                       # months
    amount: Optional[tuple[float, float]] = None     # balance or loan-amount range, dollars
    years: Optional[tuple[float, float]] = None      # model-year range

    def is_empty(self) -> bool:
        """No term, amount or model year."""
        return self.term is None and self.amount is None and self.years is None


_TERM = re.compile(r"\b(\d{1,3})\s?-?\s?(months?|mos?\b|years?|yrs?)\b", re.I)
_AMOUNT_PATTERN = r"\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?(k|thousand|grand|m|million)?\b"
_AMOUNT = re.compile(_AMOUNT_PATTERN, re.I)
_RANGE = re.compile(_AMOUNT_PATTERN + r"\s?(?:-|to|and|through)\s?" + _AMOUNT_PATTERN, re.I)
_YEAR = r"((?:19|20)\d\d)"
_YEAR_RANGE = re.compile(_YEAR + r"\s?-\s?(\d{2,4})\b")


def parse_term(s: str) -> Optional[int]:
    """A term in months: '18 Month' -> 18, '2 years' -> 24."""
    m = _TERM.search(s or "")
    if not m:
        return None
    n = int(m.group(1))
    return n * 12 if m.group(2).lower().startswith(("y",)) else n


def scale(value: float, unit: str) -> float:
    """An amount with its unit word: 12 with "k", "thousand" or "grand" is 12,000; with "m" or "million", 12,000,000."""
    unit = unit.lower()
    if unit in ("k", "thousand", "grand"):
        return value * 1_000
    if unit in ("m", "million"):
        return value * 1_000_000
    return value


def _amount(m: re.Match, g: int = 1) -> float:
    return scale(float(m.group(g).replace(",", "") + (m.group(g + 1) or "")), m.group(g + 2) or "")


def parse_amount_range(s: str) -> Optional[tuple[float, float]]:
    """'$3,000 - $11,999.99' -> (3000, 11999.99); '$12,000+' -> (12000, inf);
    '$640,000 or less' -> (0, 640000). Needs a dollar sign so '60 Months' is not an amount."""
    s = clean(s or "")
    if "$" not in s:
        return None
    m = _RANGE.search(s)
    if m and "$" in m.group(0):
        lo, hi = _amount(m, 1), _amount(m, 4)
        if hi >= lo:
            return (lo, hi)
    m = re.search(r"\$\s?[\d,]+(?:\.\d+)?", s)
    if not m:
        return None
    a = _AMOUNT.search(s, m.start())
    if not a:                          # "$," or "$1MM": a dollar sign, but no amount _AMOUNT can read
        return None
    v = _amount(a)
    tail = s[a.end():a.end() + 16].lower()
    head = s[max(0, m.start() - 22):m.start()].lower()
    if tail.startswith("+") or re.match(r"\s?(or more|and (up|over|above)|or greater|or above)", tail) or \
            re.search(r"(over|above|at least|more than|minimum of)\s?$", head):
        return (v, INF)
    if re.match(r"\s?(or less|or below|or under|and (under|below))", tail) or \
            re.search(r"(up to|under|below|less than|maximum of)\s?$", head):
        return (0.0, v)
    return (v, v)


def parse_years(s: str) -> Optional[tuple[float, float]]:
    """'2020-23 Model Year' -> (2020, 2023); 'Model Year 2025-21' -> (2021, 2025);
    '2012 Model Year (and older)' -> (-inf, 2012)."""
    s = clean(s or "")
    m = _YEAR_RANGE.search(s)
    if m:
        a = int(m.group(1))
        b = m.group(2)
        b = int(b) if len(b) == 4 else int(str(a)[:2] + b)
        return (float(min(a, b)), float(max(a, b)))
    m = re.search(_YEAR, s)
    if not m:
        return None
    y = float(m.group(1))
    low = s.lower()
    if re.search(r"(and|or) (older|earlier|prior|before)", low):
        return (-INF, y)
    if re.search(r"(and|or) (newer|later|after)", low):
        return (y, INF)
    return (y, y)


# --------------------------------------------------------------------------- dates

# A year named in words: "last year", "the past year", "a year ago", "this year", "next year". The
# search and the reader both read it here, as the year it means from a current year (year_offset).
RELATIVE_YEAR = re.compile(r"\b(?:(?:last|past|previous|prior|this|current|next)\s+(?:year|yr)|(?:a|one)\s+(?:year|yr)\s+ago)\b",
                           re.I)


def year_offset(words: str) -> int:
    """How many years after the current one a RELATIVE_YEAR phrase means: -1, 0 or 1."""
    first = words.lower().split()[0]
    return 0 if first in ("this", "current") else 1 if first == "next" else -1


MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
_MONTH_NAME = "|".join(sorted(MONTHS, key=len, reverse=True))
_DATE_WORDS = re.compile(rf"\b({_MONTH_NAME})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d\d)\b", re.I)
_DATE_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b")
_DATE_ISO = re.compile(r"\b((?:19|20)\d\d)-(\d{2})-(\d{2})\b")


def _mk(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def find_dates(s: str) -> list[tuple[int, date]]:
    """All dates in the text with their position: 'March 12, 2026', '07/21/2026', '2026-08-19'."""
    out = []
    for m in _DATE_WORDS.finditer(s):
        d = _mk(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
        if d:
            out.append((m.start(), d))
    for m in _DATE_SLASH.finditer(s):
        y = int(m.group(3))
        d = _mk(y + 2000 if y < 100 else y, int(m.group(1)), int(m.group(2)))
        if d:
            out.append((m.start(), d))
    for m in _DATE_ISO.finditer(s):
        d = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            out.append((m.start(), d))
    return sorted(out)


# A day of the week as schedules write it: "Mon", "Tues", "Thursdays". Not the start of "Monthly".
DAY = r"(?:mon|tue|tues|wed|wednes|thu|thur|thurs|fri|sat|satur|sun)(?:day)?s?\b"

_EFFECTIVE = re.compile(r"effective(?:\s+as\s+of|\s+on|\s+from|\s+date)?:?\s+", re.I)


def effective_date(s: str) -> Optional[date]:
    """The date after the word 'effective': 'Rates effective as of 07/21/2026'."""
    for m in _EFFECTIVE.finditer(s or ""):
        found = find_dates(s[m.end():m.end() + 40])
        if found and found[0][0] <= 2:
            return found[0][1]
    return None


def fmt_date(d: Optional[date]) -> str:
    """'March 4, 2031', or '' for no date."""
    return f"{d:%B} {d.day}, {d.year}" if d else ""


_SPELLED = re.compile(
    r"\b(?:" + "|".join(_TENS) + r")(?:[ -](?:" + "|".join(list(_ONES)[1:10]) + r"))?\b|\b(?:"
    + "|".join(list(_ONES)[10:]) + r")\b", re.I)


def digits_beside_words(s: str) -> str:
    """'Forty-one volunteers' -> 'Forty-one (41) volunteers'. A count written out in words is
    also given in digits, so the figure is unmistakable. Ten and above only; ordinals untouched."""
    def repl(m: re.Match) -> str:
        word, tail = m.group(0), s[m.end():m.end() + 12]
        # Left alone too: a word that only starts a larger number ("ten thousand", "twenty-first"),
        # where its own digits would be the wrong figure.
        if re.match(r"\s?\(\d|[ -](?:hundred|thousand|million|billion)\b", tail, re.I) or word.lower() in _TENS \
                and re.match(r"-(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)\b", tail, re.I):
            return word
        return f"{word} ({_words_to_digits(word.lower())})"
    return _SPELLED.sub(repl, s)
