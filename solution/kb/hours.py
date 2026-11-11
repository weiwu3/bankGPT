"""Opening hours as structured facts.

Hours are printed as a service name followed by day lines ("**Lobby**", then "Monday-Friday:
8:40 a.m.-5:20 p.m."). Each block is read into a Schedule, and two kinds of question are answered
from them without a search, so every wording gets the same reply: with no branch named, all the
distinct schedules, most recent first; with a branch named, the schedule from the page about that
branch (every word of the question that is not about hours is in that page's title or opening
lines; see is_about). A named branch with no such page is left to the normal route, which
declines when the site says nothing about its hours.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .text import DAY, clean, strip_markdown, tokens

_DAY_LINE = re.compile(rf"^\s*(?:\*\*([A-Za-z][A-Za-z -]{{2,24}})\*\*\s*)?({DAY}(?:\s?(?:-|to|&|and)\s?{DAY})?)\s*:\s*(.{{3,60}})$", re.I)
_SERVICE = re.compile(r"^\s*\*\*([A-Za-z][A-Za-z -]{2,24})\*\*\s*:?\s*$|^\s*([A-Za-z][A-Za-z -]{2,24}) hours\s*:?\s*$", re.I)
# What a schedule's day line gives after its colon: a time, "closed" or "by appointment only".
_TIMES = re.compile(r"\d\s?[ap]\.?m|closed|noon|only|appointment", re.I)
# Words a general question about hours may contain. Anything else names something specific.
_GENERAL = frozenset("""hour lobby driveup branch open close time day weekday weekend monday tuesday wednesday thursday
friday saturday sunday general typical normal regular usual same all keep kept different vary business office location
bank credit union operating operation opening closing standard teller window lane service
typically normally regularly usually generally identical differ across everywhere depend each every one another
schedule time late early week today tomorrow tonight morning afternoon evening night now right member customer public
during even""".split())             # "even": "evening" as the search index inflects it (kb.passages.inflect)


@dataclass(eq=False)
class Schedule:
    """One service's day lines as one page prints them."""
    service: str                 # "lobby", "drive-up"
    lines: tuple                 # ("Monday-Thursday: 9:15 a.m.-4:45 p.m.", ...)
    page: object


def extract(page) -> list[Schedule]:
    """Every schedule on a page: a service name, then its day lines; two other lines in a row end it."""
    out, service, lines = [], "", []

    def flush() -> None:
        nonlocal lines
        if service and lines:
            out.append(Schedule(service.lower(), tuple(lines), page))
        lines = []

    gap = 0
    for raw in page.markdown.split("\n"):
        line = clean(raw)
        if not line:
            continue
        day = _DAY_LINE.match(line)
        if day:
            if day.group(1):                         # "**Lobby** Monday-Thursday: ..." on one line
                flush()
                service = strip_markdown(day.group(1))
            if _TIMES.search(day.group(3)):
                lines.append(f"{day.group(2)}: {clean(day.group(3))}")
                gap = 0
            continue
        label = _SERVICE.match(line)
        if label:
            flush()
            service = strip_markdown(label.group(1) or label.group(2))
            gap = 0
            continue
        gap += 1
        if gap > 1:                                  # the block has ended
            flush()
            service = ""
    flush()
    return out


def is_general(question: str, common: set) -> bool:
    """True when the question asks about hours without naming a branch, town or anything specific."""
    words = set(tokens(question))
    if not words & {"hour", "open", "close"}:
        return False
    return all(w in _GENERAL or w in common or w.isdigit() for w in words)


def answer(question: str, schedules: list[Schedule]) -> Optional[tuple[str, list]]:
    """The distinct schedules for the service asked about, newest page first."""
    words = set(tokens(question))
    wanted = [s for s in schedules if set(tokens(s.service)) & words] or schedules
    if not wanted:
        return None
    by_service: dict[str, dict] = {}
    for s in sorted(wanted, key=lambda s: str(s.page.published or ""), reverse=True):
        by_service.setdefault(s.service, {}).setdefault(_key(s.lines), []).append(s)
    parts, pages = [], []
    for service, variants in by_service.items():
        described = []
        for group in variants.values():
            where = f" ({len(group)} branches)" if len(group) > 1 else ""
            described.append(", ".join(group[0].lines) + where)
            pages.extend(s.page for s in group if s.page not in pages)
        name = service.capitalize()
        if len(described) > 1:
            parts.append(f"{name} hours vary by branch. Published schedules: " + "; or ".join(described) + ".")
        else:
            parts.append(f"{name} hours: {described[0]}.")
    return " ".join(parts) + " Tell me which branch you mean and I can check whether its hours are listed.", pages


_ASKS_HOURS = re.compile(
    r"\bhours?\b|\bhrs\b|\bwhat time\b|\bhow (?:late|early)\b"
    r"|\b(?:open|close)[sd]?\b[^.?!]{0,40}\b(?:(?:mon|tues|wednes|thurs|fri|satur|sun)days?|weekends?|weekdays?|today|tomorrow|tonight)\b"
    r"|\b(?:open|close)s? (?:at|until|till|late|early)\b", re.I)
_NOT_OPENING_HOURS = re.compile(r"\bhow many\b|\bvolunteer|\bper (?:hour|week|month|year)\b|\bhourly\b", re.I)
# "Is the Elmwood branch open this afternoon?", "Are you open now?": answered by its hours too, when
# what is open is a place or the credit union ("is the application open now?" is not about hours).
# Not "still open?" or "when does it close?" without a time: that may ask whether it closed for good.
_OPEN_NOW = re.compile(r"(?<!still )\b(?:open|close)s? (?:right now|now|today|tonight|tomorrow|this \w+)\W*$", re.I)
_A_PLACE = re.compile(r"\b(?:branch|office|location|lobby|drive|you|building|atm|teller)", re.I)

# The ways a state is written: postal code, newspaper abbreviation, name. "Pa." or "Vt." is the
# postal code once tokenised. "N.C." and "W.Va." leave two words, read together; "N.Y." and "S.C."
# are not listed, as the tokeniser leaves one letter of them, the letter a street's "N." leaves.
_STATE_FORMS = {
    "AL": "Ala., Alabama", "AK": "Alaska", "AZ": "Ariz., Arizona", "AR": "Ark., Arkansas", "CA": "Calif., California",
    "CO": "Colo., Colorado", "CT": "Conn., Connecticut", "DE": "Del., Delaware", "DC": "District of Columbia",
    "FL": "Fla., Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Ill., Illinois", "IN": "Ind., Indiana",
    "IA": "Iowa", "KS": "Kan., Kans., Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Mass., Massachusetts", "MI": "Mich., Michigan", "MN": "Minn., Minnesota", "MS": "Miss., Mississippi",
    "MO": "Missouri", "MT": "Mont., Montana", "NE": "Neb., Nebr., Nebraska", "NV": "Nev., Nevada",
    "NH": "N.H., New Hampshire", "NJ": "N.J., New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "N.C., North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Okla., Oklahoma",
    "OR": "Ore., Oreg., Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tenn., Tennessee", "TX": "Tex., Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Wash., Washington", "WV": "W.Va., West Virginia", "WI": "Wis., Wisc., Wisconsin",
    "WY": "Wyo., Wyoming",
}


def _state_words() -> tuple[dict, dict]:
    """Each form as kb.text.tokens leaves it, the way questions and pages reach this rule: "Kans." is
    "kan", "Texas" is "texa", "New York" is two words, and "IN" or "OR" is a stop word, never seen here."""
    words, pairs = {}, {}
    for code, forms in _STATE_FORMS.items():
        for form in [code] + forms.split(", "):
            toks = tuple(tokens(form))
            if len(toks) == 1 and len(toks[0]) > 1:
                words[toks[0]] = code
            elif len(toks) == 2:
                pairs[toks] = code
    return words, pairs


_STATE_WORDS, _STATE_PAIRS = _state_words()
# The word for the kind of street, as tokens leaves it. A state's name right before one is a street's
# name ("Ohio Street", "New York Ave."): it tells branches apart, as the state does not.
STREET_WORDS = frozenset(tokens("street st avenue ave av road rd boulevard blvd drive dr lane ln way highway hwy "
                                "parkway pkwy court ct place pl"))
_HIGHWAY = frozenset(tokens("highway hwy"))     # the number after it is the road's name


def _streets_in(words: list) -> set:
    """The question's words that name a street after a state: the state's name, in one word or two,
    and the street word right after it. "ohio street" and "new york ave" are streets, and so is
    "oregon ct", where "ct" is a court, not Connecticut. `words` keep the question's order, so
    "right after" is read on what reaches this rule: without punctuation and stop words."""
    out = set()
    for i in range(1, len(words)):
        if words[i] not in STREET_WORDS:
            continue
        if i >= 2 and (words[i - 2], words[i - 1]) in _STATE_PAIRS:
            out |= set(words[i - 2:i + 1])
        elif words[i - 1] in _STATE_WORDS:
            out |= set(words[i - 1:i + 1])
    return out


def _states_in(words: list, place: set) -> tuple[set, set]:
    """The states among a question's place words, as postal codes, and the place words left:
    "springfield il" gives ({"IL"}, {"springfield"}). A two-word name counts only side by side.
    A street named after a state is no state: "ohio street" gives (set(), {"ohio", "street"})."""
    street = _streets_in(words)
    states = set()
    for pair in zip(words, words[1:]):
        if pair in _STATE_PAIRS and set(pair) <= place - street:
            states.add(_STATE_PAIRS[pair])
            place = place - set(pair)
    states |= {_STATE_WORDS[w] for w in place - street if w in _STATE_WORDS}
    return states, {w for w in place if w in street or w not in _STATE_WORDS}


def states_by_name(text: str) -> dict:
    """The states a text names as where something is, by name and not by postal code, as {code: the
    words of the name}: "any ATM in North Dakota?" gives {"ND": ("north", "dakota")}. Only right after
    "in", so "the Nevada Avenue branch" names none ("in Nevada Avenue" still does)."""
    words = tokens(text, keep_stop=True)
    found = {}
    for at in (i + 1 for i, w in enumerate(words) if w == "in"):
        pair, word = tuple(words[at:at + 2]), words[at] if at < len(words) else ""
        if pair in _STATE_PAIRS:
            found[_STATE_PAIRS[pair]] = pair
        elif word in _STATE_WORDS and word != _STATE_WORDS[word].lower():
            found[_STATE_WORDS[word]] = (word,)
    return found


def _states_named(have: frozenset) -> set:
    """The states a page's title and opening lines name, however written."""
    return ({_STATE_WORDS[w] for w in have if w in _STATE_WORDS}
            | {code for pair, code in _STATE_PAIRS.items() if set(pair) <= have})


# A site shortens what a member writes out: "Linden Hill Drive" is "Linden Hill Dr." on the page,
# "East Elm Road" is "E. Elm Road" and "Elmwood, Kansas" is "Elmwood, KS". A street's kind, a
# direction and a state are compared in one form, in the question and on the page alike.
_LONG = {"av": "avenue", "ave": "avenue", "blvd": "boulevard", "ct": "court", "dr": "drive", "hwy": "highway",
         "ln": "lane", "pkwy": "parkway", "pl": "place", "rd": "road", "st": "street",
         "e": "east", "w": "west", "n": "north", "s": "south"}


def long_form(words, states: bool = False) -> set:
    """The words with a street's kind written out, and with states=True every state as its code.
    (The hours rule reads states on their own, so a street named after one stays a street.)"""
    return {_LONG.get(w) or (_STATE_WORDS.get(w, w) if states else w) for w in words}


def place_of(words: list, common: set) -> tuple[set, set]:
    """The states and the other place words a question about opening hours names: every word that
    is not about hours, not the site's own name and not a number (except a highway's: "Highway 52"
    is not "Highway 88"). `words` are its tokens with the spelling matched to the site."""
    place = {w for i, w in enumerate(words) if w not in _GENERAL and w not in common
             and (not w.isdigit() or i > 0 and words[i - 1] in _HIGHWAY)}
    return _states_in(words, place)


def asks_hours(question: str) -> bool:
    """The question asks when something is open: its hours, or whether it is open now. Not
    "which accounts can I open at ...", which asks about opening an account."""
    return bool(_ASKS_HOURS.search(question) or (_OPEN_NOW.search(question) and _A_PLACE.search(question))) \
        and not _NOT_OPENING_HOURS.search(question) and not re.search(r"\baccounts?\b", question, re.I)


def opening_words(page) -> frozenset:
    """The words of a page's title and opening lines, which tell what place the page is about."""
    return frozenset(tokens(page.title + " " + " ".join(p.text for p in page.passages[:2])[:700]))


def is_about(have: frozenset, place: set, states: set) -> bool:
    """Whether a page with the words `have` in its title and opening lines is about the place asked.
    The state does not tell branches apart and pages write it their own way ("Ill." for "IL"), so it
    need not be there. But a page that names states must name the one asked: "Elmwood, KS" is not the
    Elmwood in Oregon, and "Vermont Terrace", a kind of street STREET_WORDS does not list, must not reach
    every page with a "terrace"."""
    named = _states_named(have)
    if not place:                                    # the question names nothing but a state
        return states <= named
    return long_form(place) <= long_form(have) and (states <= named or not named)


# The branch's street address after "located at": the house number, then words that start with a
# capital or a digit. An initial or a "Mt." goes on to the next word; a street-type abbreviation
# ("St.", "Blvd.") ends the address and keeps its period. A lower-case word ("at the corner of",
# "in Elmwood", "with"), a comma, the end of the line or the period that ends the sentence ends it too.
# An initial with a word after it is never taken for that period, so a teaser cut off further on
# ("located at 9 N. Lake Road...") gives no address instead of "9 N". A hyphen goes on only to a capital
# or a digit ("Birch-Hollow"): clean() writes a dash as "-", and in "Road-across from" the sentence goes on.
# "\d\w*" matches what "\d+\w*" would, without trying every split of a long run of digits.
_DOTTED_ABBREV = r"(?:[A-Z]|Mt|Ft|St|Hwy|Rte)\.(?= [A-Z0-9])"
_ADDRESS = re.compile(
    r"\blocated at (\d\w*"
    rf"(?: (?:{_DOTTED_ABBREV}|(?!{_DOTTED_ABBREV})[A-Z0-9][\w']*(?:-[A-Z0-9][\w']*)*))+?"
    r"(?: (?:St|Ave|Blvd|Rd|Dr|Ln|Ct|Pl|Pkwy|Cir|Ter|Trl|Sq)\.(?: [NS][EW])?(?![\w.])"
    r"|(?=\.(?![\w.])|[^\w\s.]| (?![A-Z0-9])|\n|$)))")


def _before_hours(markdown: str) -> str:
    """The page up to its first schedule line: a day line that gives a time."""
    lines = markdown.split("\n")
    for i, raw in enumerate(lines):
        day = _DAY_LINE.match(clean(raw))
        if day and _TIMES.search(day.group(3)):
            return "\n".join(lines[:i])
    return markdown


def _address(markdown: str) -> Optional[str]:
    """'located at 1200 E. Maple Road at the corner of ...' gives '1200 E. Maple Road'. A cut-off
    teaser ('located at 9 Lake...') gives nothing. Only the part before the hours is read: a branch's
    page says where it is before it prints its hours, and below them it often points to another
    branch ("our Oakdale office, located at ..."), whose address is not this schedule's."""
    m = _ADDRESS.search(clean(_before_hours(markdown)))
    return m.group(1) if m else None


def for_place(question: str, words: list, schedules: list[Schedule], about: dict, common: set) -> Optional[tuple[str, list]]:
    """The schedule of the branch a question names. `words` are the question's tokens with its
    spelling matched to the site; `about` maps a page to the words of its title and opening lines.
    Returns None unless some page that prints a schedule is about every place word in the question."""
    if not asks_hours(question):
        return None
    states, place = place_of(words, common)
    if not place and not states:
        return None
    found = [s for s in schedules if is_about(about.get(s.page.key, frozenset()), place, states)]
    wanted = [s for s in found if set(tokens(s.service)) & set(words)] or found
    if not wanted:
        return None
    by_page: dict = {}
    for s in sorted(wanted, key=lambda s: str(s.page.published or ""), reverse=True):
        by_page.setdefault(s.page, []).append(s)
    parts = []
    for page, group in by_page.items():
        where = _address(page.markdown)
        label = f"the branch at {where}" if where else page.title
        when = f", as announced {page.published.isoformat()}" if page.published else ""
        parts.append(f"For {label}{when}: " + "; ".join(f"{s.service} hours: " + ", ".join(s.lines) for s in group) + ".")
    return " ".join(parts), list(by_page)


def _key(lines: tuple) -> tuple:
    """Day lines compared without case, spaces or periods, and with "to" as "-"."""
    return tuple(re.sub(r"[\s.]", "", line.lower()).replace("to", "-") for line in lines)
