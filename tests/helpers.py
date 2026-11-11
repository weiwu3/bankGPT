"""Shared helpers for the tests written on made-up sites.

Every site here is invented: its towns, products, numbers and wording. Nothing in this folder is
taken from the credit union's website.

build() feeds pages to the solution exactly as the scorer does (Page and Snapshot from the
organisers' contract, oldest download first). conftest.py has already taken the language model
out whatever the shell says, so a key in the environment can neither cost money nor change an
answer; a test that wants a model plugs in a scripted one.
"""
from pathlib import Path

from contract import Page, Snapshot
import solution
from kb.reader import Reader
from kb.reply import ASK, REFUSAL, refusal  # noqa: F401  (for the test files)

ROOT = Path(__file__).resolve().parents[1]
SELFTEST = ROOT / "harness" / "selftest"
SITE = "https://cu.test"

# A credit union whose 41-month certificate special of the spring is gone; two others run now.
SPRING = ("2026-04-02", {"specials": "# Certificate specials\n\nOur spring specials are the 41-month and the "
                                     "14-month certificates.\n"})
NOW = ("2026-06-10", {"specials": "# Certificate specials\n\nOur current specials are the 14-month and the "
                                  "22-month certificates. Open one at any Maple Falls branch.\n"})
GONE = "In the spring you had a 41-month certificate special. Which specials do you have now?"


def refused(answer, question=""):
    """A refusal as engine.refusal words one: the fixed sentence, or the one sentence on what the site
    does not state that it would say for this question, then ASK. Not confident either way."""
    why = answer.text[:-len(ASK)].strip() if answer.text.endswith(" " + ASK) else ""
    return not answer.confident and answer.text in (REFUSAL, refusal(question, why).text)


def snapshot(n, day, pages, site=SITE, reverse=False, title=None):
    """One download: pages is {path: markdown}. The title of each page is its path, or title(markdown)."""
    items = list(pages.items())[::-1] if reverse else list(pages.items())
    return Snapshot(f"{n:02d}_{day}", day, [Page(f"{site}/{path}", title(md) if title else path, md, f"{day}T00:00:00Z")
                                            for path, md in items])


def build(*downloads, reverse=False, reader=None, title=None):
    """downloads: (day, {path: markdown}) oldest first. reverse=True lists each download's pages backwards.
    reader: a kb.reader.Reader to answer free-text questions, or None for rules only. title: as for snapshot."""
    ks = solution.build()
    ks.engine.reader = reader
    for n, (day, pages) in enumerate(downloads, 1):
        ks.ingest(snapshot(n, day, pages, reverse=reverse, title=title))
    return ks


def table(label, rows, effective=None, header="| Term | APY |", heading="## Rates"):
    """A rate table under a heading and a label line, with its "effective as of" line when given."""
    lines = [f"{heading}\n\n{label}\n", header, "| --- " * (header.count("|") - 1) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if effective:
        lines.append(f"\nRates effective as of {effective}; they can change at any time.")
    return "\n".join(lines) + "\n"


def wren(apy, effective=None, label="Wren Certificates"):
    """A one-row certificate table: the 6-month APY."""
    return table(label, [("6 Month", apy)], effective)


# A certificate table, and a site with it and a fact only a sentence states.
CERTIFICATES = table("Certificates", [("6 Month", "3.50%"), ("12 Month", "3.10%")], "03/01/2026")
SMALL_SITE = ("2026-03-05", {"rates": CERTIFICATES, "about": "# About us\n\nOur lobby is open Monday to Friday. "
                                                            "We were founded in 1957 by twelve teachers.\n"})


def path_of(ks, question, as_of=None):
    """Which route answered: rows, hours, link, contact, quote, reader or declined."""
    return ks.engine.answer(question, as_of).path


class Scripted:
    """Stands in for the language model: returns a fixed reply and keeps the prompts it was sent."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    @property
    def calls(self):
        """How many times the model was asked."""
        return len(self.prompts)

    def complete(self, system, user):
        self.prompts.append(user)
        return self.reply


def scripted_reader(reply):
    """A reader whose model always replies `reply`."""
    return Reader(Scripted(reply))


def with_reader(reply, *downloads):
    """build(*downloads) with a reader whose model always replies `reply`."""
    return build(*downloads, reader=scripted_reader(reply))


def model_reply(answer, verdict="answer", sources="1", quotes="none"):
    """A reply in the form the reader asks the model for."""
    return f"QUOTES: {quotes}\nVERDICT: {verdict}\nSOURCES: {sources}\nANSWER: {answer}"


def has_digit(text):
    return any(ch.isdigit() for ch in text)
