"""What every route returns, and the one place every refusal is worded."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

ASK = "Please contact the credit union for details."
REFUSAL = "I don't know: that isn't listed on the website. " + ASK
# A figure is a number with its unit ("$" before it, "%" after it, or none), compared by value, as the
# organisers' scorer reads one: "3.5%" and "3.50 %" are one figure, and "$41" is not the 41 of
# "41-month". Space before the number is looked for only after a "$": a leading \s* is tried at
# every position, which makes the search quadratic on a long run of spaces (20,000 took four seconds).
_FIGURE = re.compile(r"(?:(\$)\s*)?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%|percent)?", re.I)


@dataclass(eq=False)
class Reply:
    """What the engine returns; solution.py turns it into the scorer's Answer."""
    text: str
    pages: list = field(default_factory=list)
    confident: bool = True
    note: Optional[str] = None
    quotes: dict = field(default_factory=dict)       # page key -> the words quoted from it
    path: str = ""              # which route produced it: rows, hours, link, contact, quote, reader or declined

    def __post_init__(self) -> None:
        self.pages = list(self.pages)
        self.quotes = self.quotes or {}


def dated(text: str, as_of: Optional[str], view) -> str:
    """An answer for an "as of" date says which download it was read from; one for now says nothing."""
    if as_of:
        return f"As of {as_of} (the website as downloaded on {view.taken_on.isoformat()}): " + text
    return text


def figures(s: str) -> set:
    """The figures in s, as (unit, value) pairs."""
    return {("$" if m.group(1) else "%" if m.group(4) else "",
             round(float(m.group(2).replace(",", "") + "." + (m.group(3) or "0")), 4)) for m in _FIGURE.finditer(s)}


# A refusal's reason says only what the website does not state: "The website does not list ..." or
# "No rate is published for ...", one sentence with no second clause. "Wire transfers are free and
# no fee is listed" states a fact, and the fixed REFUSAL is given instead.
_UNSTATED = re.compile(r"(?:The (?:web)?site (?:does not|doesn't) (?:list|publish|state|say|give|show|mention|specify)"
                       r"|No\b[^.]{0,60}?\b(?:is|are) (?:published|listed|stated|given))\b[^.!?,;:–—]*\.")
_SECOND_CLAUSE = re.compile(r"\b(?:and|but|though|although|while|whereas|because|so|which|since)\b|\s-\s", re.I)


def refusal(question: str = "", why: str = "") -> Reply:
    """Every refusal: the rules', the reader's and solution.py's. `why` is said when it is a short
    _UNSTATED sentence with no figure the question does not have (the scorer passes a refusal only
    when it states no number but the question's own); otherwise the fixed REFUSAL."""
    why = " ".join(why.split())
    if (len(why) <= 150 and _UNSTATED.fullmatch(why) and not _SECOND_CLAUSE.search(why)
            and not figures(why) - figures(question)):
        return Reply(f"{why} {ASK}", confident=False, path="declined")
    return Reply(REFUSAL, confident=False, path="declined")
