"""Answer by quoting the site when no model reply is available: a passage, a link, or decline.

The rows and the opening-hours rules have already had their turn (kb.engine). What is left is
quoted: a link the site gives, or the stretch of the best passage that covers the question and holds
the kind of answer asked for. A question that asks for steps no passage gives gets the site's
contact paragraph. It declines when the question names something the site does not mention, when
no passage is about the place or the thing asked, and when a rate is asked for and no passage
states one.
"""
from __future__ import annotations

import re

from . import hours
from .lookup import LENDING, RATES, SAVING, Query
from .passages import CLOCK_TIME, COUNTING, SAME_RATE, PassageIndex, answer_shapes
from .reply import Reply, dated, refusal
from .text import MONTHS, STOPWORDS, content_tokens, tokens

# A percentage stated as a rate: "7.40% APR", "a rate of 5.15%", "as low as 2.49% intro APR".
_RATE_STATEMENT = re.compile(
    r"\d(?:\.\d+)?\s?%\*?\s?(?:[A-Za-z]+\s){0,2}(?:apr|apy|annual percentage|dividend|interest rate|rate\b)"
    r"|(?:apr|apy|rate)s?\*?\s(?:of|is|are|at|as low as|from|starting at)\s(?:[a-z]+\s){0,2}\d(?:\.\d+)?\s?%", re.I)
_RATE_BY_REFERENCE = re.compile(SAME_RATE + r"\b", re.I)
# A sentence that goes on about what the one before it named.
_CONTINUES_SENTENCE = re.compile(r"(?:it|its|this|these|they|their)\b", re.I)
_PRODUCT_NOUNS = LENDING | SAVING
_MONTHS = frozenset(tokens(" ".join(MONTHS)))
_STREET_ADDRESS = re.compile(r"\b\d{1,5}(?: [NSEW]\.?)? [A-Z0-9][\w.]*(?: [A-Z][\w.]*){0,3},? ")      # "2245 Elm Hill Dr., "
# A question that asks for steps ("how do I", not "how much"). Narrower than the reader's _HOW_TO on purpose:
# here it only decides whether a passage is quoted as the steps.
_HOW_TO_STEPS = re.compile(r"\W*(how (?:do|can|could|would|should|to)|can i|where (?:do|can) i|what(?:'s| is) the "
                           r"(?:best )?way)\b", re.I)


def states_rate(p, words: set) -> bool:
    """The passage states a rate for one of `words`: under a heading or on a page named for it, or in a
    sentence about it. A sentence is about it when it names it, or goes on from one that is without naming
    another kind of product: "A Wren Loan builds credit. With terms up to a year and a low 6.10% APR, ..."
    prices the Wren Loan; "... construction loans. Auto loans are as low as 4.99% APR." prices auto loans.
    "... earns the same dividends as a basic share account" states the rate by pointing to another."""
    def rate(text: str) -> bool:
        return bool(_RATE_STATEMENT.search(text) or _RATE_BY_REFERENCE.search(text))
    if set(tokens(f"{p.heading} {p.page.title}")) & words:
        return rate(p.text)
    about = False
    for sentence in re.split(r"(?<=[.!?])\s+", p.text):
        if set(tokens(sentence)) & words:
            about = True
        elif about and not _CONTINUES_SENTENCE.match(sentence):
            said, named = tokens(sentence, keep_stop=True), set(content_tokens(sentence))
            about = not any(b in _PRODUCT_NOUNS and a in named and a not in _PRODUCT_NOUNS for a, b in zip(said, said[1:]))
        if about and rate(sentence):
            return True
    return False


def answer(question: str, as_of, view, q: Query, match, ranked: list, passages: PassageIndex) -> Reply:
    """Answer without a language model: quote the site, or decline."""
    if q.wants_number and q.metrics & RATES:
        # A rate was asked for and the rows do not settle it. Quote a passage that states a
        # rate for the thing asked about, if there is one; otherwise the site does not say.
        named = set(q.modifiers) or set(q.product)
        stated = [(s, p) for s, p in ranked[:8] if states_rate(p, named)]
        if not stated or (match is not None and match.tier_missing):
            return refusal()
        ranked = stated
    if q.wants_link:
        links = passages.find_links(question, 1)
        if links:
            return Reply(f'Use {links[0].url} (the site\'s "{links[0].text}" link).', [links[0].page], True, path="link")
    # A name the website never uses, or uses only in passing, cannot be answered from it.
    names, unknown = _names(question, passages)
    if unknown:
        return refusal()
    shapes = answer_shapes(question)
    # Opening hours for a named place come from a passage about that place: every word of the
    # place's name that is not about hours (nor a month) is in the passage, or in its page's
    # title or opening lines, as in kb.hours. The contact paragraph's hours are when messages
    # are answered, whatever places it lists. With no such passage, decline: another branch's
    # hours are no answer. The name may also open the question: "Elm Road lobby hours?".
    asks_hours = hours.asks_hours(question)
    if asks_hours and not any(shape is CLOCK_TIME for shape, _ in shapes):
        shapes = shapes + [(CLOCK_TIME, True)]    # "is it open today?" is answered by its hours
    first = re.findall(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)+|^[A-Z][a-z]+(?=\s+(?:branch|office|location|lobby|drive|hours?|hrs)\b)",
                       question)
    # The question's words as the hours rule reads them (spelling matched to the site), kept
    # where they belong to the name, and the number after a highway ("Highway 52"). With no
    # name found ("WHEN IS THE ELMWOOD LOBBY OPEN?"), every word counts, as in kb.hours.
    named = {passages.spell(t) for t in tokens(" ".join(names + first))}
    words = [passages.spell(t) for t in tokens(question)]         # not inflected: "evening" is no "even"
    if named:
        words = [w for i, w in enumerate(words) if w in named or w.isdigit() and i > 0 and words[i - 1] in named]
    states, place = hours.place_of(words, view.cache["common"])
    place -= _MONTHS
    asks_where = bool(names) and bool(re.search(r"\baddress\b|\bwhere is\b|\bwhere's\b|\blocated\b|\bdirections\b",
                                                  question, re.I))
    # The steps are in a passage about the thing: one word in common ("change" in "rates can change") is a
    # coincidence. Two of the question's words, or its only one.
    how_to = _HOW_TO_STEPS.match(question)
    topic = set(passages.query_tokens(question[how_to.end():])) if how_to else set()
    usable = _usable(ranked, passages, shapes, names, states, place, asks_hours, asks_where, how_to, topic)
    if not usable:
        contact = passages.contact()
        if how_to and contact is not None and not (asks_hours and (place or states)):    # "how late is ... open"
            return Reply("The website doesn't give specific steps for that. " + _window(contact.text, set(), 320),
                         [contact.page], True, path="contact")
        return refusal()
    want = set(tokens(question))
    general = not any(len(n.split()) > 1 for n in names)
    picks = usable[:2] if (general and shapes) else usable[:1]
    looks = [shape for shape, _ in shapes]
    if COUNTING.search(question):      # a count: a number next to the thing counted
        nouns = "|".join(re.escape(w[:max(4, len(w) - 3)]) for w in re.findall(r"[a-z]{4,}", question.lower())
                         if set(tokens(w)) & want)
        if nouns:
            looks.append(re.compile(rf"\d[\d,.]*\+?\s+(?:[A-Za-z-]+\s+){{0,2}}(?:{nouns})", re.I))
    text = dated(" ".join(_window(p.text, want, 300, looks) for _, p in picks), as_of, view)
    return Reply(text[:900], list(dict.fromkeys(p.page for _, p in picks)), True, path="quote")


def _names(question: str, passages: PassageIndex) -> tuple[list, bool]:
    """The names the question gives (words in capitals, or failing those the words before "branch"),
    and whether one of them is a word the site never uses, or uses only in passing."""
    names = re.findall(r"(?<!^)(?<![.?!]\s)\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", question)
    # A name in capitals is read as written: "Harvesta" is a town the site never names, not "harvest". (A slip
    # in a branch whose hours are published was matched by kb.hours.) Only small letters are spell-checked.
    spell = (lambda t: t) if names else passages.spell
    # None written in capitals ("the birchwood avenue branch"): the words before "branch" name it. "\b": a
    # run of letters is tried from its first letter only, not from each (quadratic in a very long word).
    if not names:
        names = [n for n in (" ".join(w for w in m.split() if w not in STOPWORDS) for m in
                             re.findall(r"\b((?:[a-z]+ ){1,2})(?:branch|office|location)\b", question.lower())) if n]
    # A misspelt word the site does use ("elmwod" for Elmwood) is that word, not an unknown name;
    # a street's kind ("boulevard", which a site may write only as "Blvd.") is no name at all.
    unknown = [t for t in content_tokens(question) if t not in hours.STREET_WORDS and not passages.known(spell(t))]
    return names, any(re.search(rf"\b{re.escape(t[:5])}", w, re.I) for t in unknown for w in names)


def _usable(ranked: list, passages: PassageIndex, shapes: list, names: list, states: set, place: set,
            asks_hours: bool, asks_where: bool, how_to, topic: set) -> list:
    """The best passages that can be quoted as the answer: they hold the kind of answer asked for, are
    about the place or the thing named, give a street address when asked where something is, and
    share the topic of a question that asks for steps."""
    usable = []
    for score, p in ranked[:6]:
        if any(required and not shape.search(p.text) for shape, required in shapes):
            continue                      # an hours question needs text with a clock time in it
        if asks_hours and (place or states):
            if p is passages.contact() or not hours.is_about(hours.opening_words(p.page).union(p.toks), place, states):
                continue
        # Any other answer for "Birchwood Avenue" must come from a page that names Birchwood Avenue,
        # and where such a place is, from a passage with a street address in it: a menu that lists
        # "Birchwood Ave." among the branches says nothing about where it is.
        about = hours.long_form(passages.about.get(p.page.key, frozenset()) | set(p.toks), states=True)
        if any(not hours.long_form(tokens(name), states=True) <= about for name in names if len(name.split()) > 1):
            continue
        if asks_where and not _STREET_ADDRESS.search(p.text):
            continue
        if how_to and len(topic & about) < min(2, len(topic)):
            continue
        usable.append((score, p))
    return usable


def _window(text: str, want: set, size: int, shapes: tuple = ()) -> str:
    """The stretch of a passage that covers most of the question and holds the kind of answer asked
    for (a clock time, an amount, a count). Starts at a word, ends at a sentence where it can."""
    if len(text) <= size:
        return text.strip()
    best, best_at = -1.0, 0
    for m in re.finditer(r"\S+", text):
        at = m.start()
        chunk = text[at:at + size]
        score = len(want & set(tokens(chunk))) + sum(2.5 * min(2, len(shape.findall(chunk))) for shape in shapes)
        if score > best + 1e-9:
            best, best_at = score, at
        if at + size >= len(text):
            break
    chunk = text[best_at:best_at + size]
    end = max(chunk.rfind(". "), chunk.rfind("! "), chunk.rfind("? "))
    return (chunk[:end + 1] if end > 0.6 * size else chunk).strip()
