"""Properties that hold for every question, checked over many generated questions on a generated site.

The site is made up and drawn from a seeded random generator, so every run sees the same site and the
same questions: four products with term tables, balance tiers, a business twin and model-year loans,
over three downloads in which some values change, plus the copies that lose (a calculator, a campaign
page and an article showing older values). Every published percentage on it is unique, so a value in
an answer says exactly which product, tier and download it came from.

Plain seeded `random` rather than a property-testing library: the cases are cheap to enumerate and a
failure names its question. No downloads, no model.
"""
import json
import os
import random
import re
import subprocess
import sys

import pytest

from helpers import REFUSAL, ROOT, build

SEED = 20261110
DAYS = ["2026-01-12", "2026-03-09", "2026-05-11"]
NAMES = ["Wren", "Heron", "Plover", "Kestrel"]
TERMS = [3, 6, 12, 18, 24, 36, 48, 60]
TIERS = [(0, 999.99), (1000, 9999.99), (10000, 49999.99), (50000, None)]
YEARS = [((2024, 2026), "2024-26 Model Year"), ((2019, 2023), "2019-23 Model Year")]


def make_site(seed=SEED):
    """(downloads, facts). downloads: [(day, {path: markdown})]. facts: {(n, key): value} for the value of each
    tier in download n, where key is ("cert", name, term), ("sav", name, tier), ("biz", name, tier) or
    ("auto", name, years)."""
    rng = random.Random(seed)
    used = set()

    def value():
        while True:
            v = f"{rng.uniform(0.2, 9.8):.2f}%"
            if v not in used:
                used.add(v)
                return v

    terms = {name: sorted(rng.sample(TERMS, 4)) for name in NAMES}
    facts, current = {}, {}
    for name in NAMES:
        for t in terms[name]:
            current[("cert", name, t)] = value()
        for i in range(len(TIERS)):
            current[("sav", name, i)] = value()
            current[("biz", name, i)] = value()
        for y, _ in YEARS:
            current[("auto", name, y)] = value()
    downloads = []
    for n, day in enumerate(DAYS):
        if n:                                           # some values change in each new download
            for key in rng.sample(sorted(current, key=str), 12):
                current[key] = value()
        facts.update({(n, key): v for key, v in current.items()})
        stamp = f"{int(day[5:7]):02d}/01/{day[:4]}"
        pages = {}
        for name in NAMES:
            low = name.lower()
            cert = [(f"{t} Month", current[("cert", name, t)]) for t in terms[name]]
            sav = [(_tier_text(i), current[("sav", name, i)]) for i in range(len(TIERS))]
            biz = [(_tier_text(i), current[("biz", name, i)]) for i in range(len(TIERS))]
            auto = [(label, "60 Months", current[("auto", name, y)]) for y, label in YEARS]
            pages[f"{low}/certificates"] = _table(f"{name} Certificates", "| Term | APY |", cert, stamp)
            pages[f"{low}/savings"] = _table(f"{name} Savings", "| Balance | APY |", sav, stamp)
            pages[f"business/{low}/savings"] = _table(f"Business {name} Savings", "| Balance | APY |", biz, stamp)
            pages[f"{low}/auto-loans"] = _table(f"{name} Auto Loans", "| Model Year | Term | APR |", auto, stamp)
        if n:                                           # copies left behind, on pages trusted less
            first = {k: v for (m, k), v in facts.items() if m == 0}
            name = NAMES[n % len(NAMES)]
            cert = [(f"{t} Month", first[("cert", name, t)]) for t in terms[name]]
            pages[f"tools/{name.lower()}-calculator"] = _table(f"{name} Certificates", "| Term | APY |", cert)
            pages[f"{name.lower()}-promo-lp"] = _table(f"{name} Certificates", "| Term | APY |", cert)
            pages[f"news/{name.lower()}-story"] = (f"1 min read\n\nJanuary 20, {day[:4]}\n\n# {name} news\n\n"
                                                   + _table(f"{name} Certificates", "| Term | APY |", cert))
        pages["about"] = "# About\n\nThe Larkspur Valley credit union has served orchard towns since 1961.\n"
        downloads.append((day, pages))
    return downloads, facts, terms


def _tier_text(i):
    lo, hi = TIERS[i]
    return f"${lo:,.2f}+" if hi is None else f"${lo:,.2f} - ${hi:,.2f}"


def _table(label, header, rows, stamp=None):
    lines = ["## Rates", "", label, "", header, "| --- " * (header.count("|") - 1) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if stamp:
        lines += ["", f"Rates effective as of {stamp}."]
    return "\n".join(lines) + "\n"


# The ways a question for one certificate term or one savings balance is worded.
CERT_WORDINGS = ["What is the APY on a {t}-month {n} certificate?", "{n} certificate APY for {t} months?",
                 "APY on {n} certificates, {t} month term?", "Current {t} month {n} certificate APY"]
SAVINGS_WORDINGS = ["{n} Savings APY on {a}?", "What APY does {n} Savings pay on a {a} balance?"]


def make_questions(terms, seed=SEED):
    """[(question, key or None)]: key names the one fact the question asks for; None: not published."""
    rng = random.Random(seed + 1)
    out = []
    for name in NAMES:
        for t in terms[name]:
            out.append((rng.choice(CERT_WORDINGS).format(t=t, n=name), ("cert", name, t)))
        for t in sorted(set(TERMS) - set(terms[name]))[:2]:
            out.append((f"What is the APY on a {t}-month {name} certificate?", None))
        for i, (lo, hi) in enumerate(TIERS):
            amount = rng.choice([lo, (hi or lo * 3) - 0.01 if hi else lo + 1, rng.randint(int(lo), int(hi or lo * 3))])
            asked = f"${amount:,.2f}" if amount != int(amount) else f"${int(amount):,}"
            out.append((rng.choice(SAVINGS_WORDINGS).format(n=name, a=asked), ("sav", name, i)))
            out.append((f"Business {name} Savings APY on {asked}?", ("biz", name, i)))
        for y, _ in YEARS:
            year = rng.randint(*y)
            out.append((f"{name} auto loan APR for a {year} model year, 60 months?", ("auto", name, y)))
        out.append((f"{name} auto loan APR for a 2031 model year, 60 months?", None))
    out += [("What is the APY on a 12-month Zorblex certificate?", None),
            ("What rate do you charge on a construction loan?", None)]
    return out


DOWNLOADS, FACTS, CERT_TERMS = make_site()
QUESTIONS = make_questions(CERT_TERMS)
ALL_VALUES = set(FACTS.values())
# Dates to ask as of: each download's day, a day before the next one, and today.
AS_OF = [None, "2026-01-12", "2026-03-08", "2026-03-09", "2026-05-10", "2026-05-11"]


def view_index(as_of):
    if as_of is None:
        return len(DAYS) - 1
    return max(n for n, day in enumerate(DAYS) if day <= as_of)


def values_in(text):
    return {v for v in re.findall(r"\d+\.\d\d%", text) if v in ALL_VALUES}


@pytest.fixture(scope="module")
def ks():
    return build(*DOWNLOADS)


def test_the_generated_site_and_questions_are_what_the_tests_expect():
    assert len(QUESTIONS) >= 60 and sum(k is None for _, k in QUESTIONS) >= 10
    assert len(ALL_VALUES) == len(set(FACTS.values()))
    changed = {k for (n, k), v in FACTS.items() if n == 2 and FACTS[(0, k)] != v}
    assert len(changed) >= 12


@pytest.mark.parametrize("as_of", AS_OF)
def test_a_question_for_one_tier_gets_that_tier_s_current_value_and_no_other(ks, as_of):
    n = view_index(as_of)
    wrong = []
    for question, key in QUESTIONS:
        if key is None:
            continue
        got = values_in(ks.answer(question, as_of).text)
        if got != {FACTS[(n, key)]}:
            wrong.append((question, sorted(got), FACTS[(n, key)]))
    assert not wrong, wrong[:5]


@pytest.mark.parametrize("as_of", AS_OF)
def test_every_refusal_is_the_fixed_sentence_with_no_digit(ks, as_of):
    for question, key in QUESTIONS:
        answer = ks.answer(question, as_of)
        if not answer.confident or key is None:
            assert answer.text == REFUSAL and not answer.confident, question
            assert not re.search(r"\d", answer.text) and answer.sources == []


@pytest.mark.parametrize("as_of", AS_OF)
def test_an_as_of_answer_never_holds_a_value_that_only_a_later_download_has(ks, as_of):
    n = view_index(as_of)
    then = {v for (m, _), v in FACTS.items() if m <= n}
    for question, _ in QUESTIONS:
        assert values_in(ks.answer(question, as_of).text) <= then, (question, as_of)


@pytest.mark.parametrize("as_of", AS_OF)
def test_every_figure_in_a_rules_answer_is_printed_in_the_download_in_force(ks, as_of):
    pages = "\n".join(DOWNLOADS[view_index(as_of)][1].values())
    printed = set(re.findall(r"\$?[\d,]+(?:\.\d+)?%?", pages))
    for question, _ in QUESTIONS:
        reply = ks.engine.answer(question, as_of)
        if reply.path != "rows":
            continue
        body = re.sub(r"^As of .*?\): |\(rates effective [^)]*\)", "", reply.text)
        figures = set(re.findall(r"\$[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?%", body))
        assert figures <= printed, (question, figures - printed)


def test_asking_again_in_a_new_system_gives_the_same_answer(ks):
    again = build(*DOWNLOADS)
    for question, _ in QUESTIONS:
        for as_of in (None, "2026-03-09"):
            a, b = ks.answer(question, as_of), again.answer(question, as_of)
            assert (a.text, a.confident, [s.url for s in a.sources]) == (b.text, b.confident, [s.url for s in b.sources])


def test_every_wording_of_a_fact_gets_the_same_answer(ks):
    for as_of in (None, "2026-03-09"):
        for name in NAMES:
            asked = [[w.format(t=t, n=name) for w in CERT_WORDINGS] for t in CERT_TERMS[name]]
            asked += [[w.format(n=name, a=f"${int(lo) + 1:,}") for w in SAVINGS_WORDINGS] for lo, _ in TIERS]
            for wordings in asked:
                texts = {ks.answer(q, as_of).text for q in wordings}
                assert len(texts) == 1, (as_of, wordings, texts)


def variants(question):
    """The same question written other ways: case, punctuation, apostrophes, spaces."""
    yield question.lower()
    yield question.upper()
    yield question.rstrip("?") + "??"
    yield question.rstrip("?") + "."
    yield "  " + question.replace(" ", "  ") + "  "
    yield question.replace(" ", " ")
    yield question.replace("What is", "What’s").replace("What's", "What’s")
    yield question.replace("What is", "What's")
    yield question.replace(", ", " ").replace("?", "")


def test_case_punctuation_apostrophes_and_spacing_do_not_change_a_rules_answer(ks):
    for question, _ in QUESTIONS:
        reply = ks.engine.answer(question)
        if reply.path != "rows":
            continue
        for other in variants(question):
            ks.engine._answers.clear()           # each wording read afresh, not served from the cache
            assert ks.engine.answer(other).text == reply.text, (question, other)


def test_the_order_of_the_pages_in_a_download_changes_no_answer(ks):
    backwards = build(*DOWNLOADS, reverse=True)
    for question, _ in QUESTIONS:
        for as_of in (None, "2026-03-09"):
            assert backwards.answer(question, as_of).text == ks.answer(question, as_of).text, question
    # The same entries (listed in another order: entries of equal weight keep the order the pages came in).
    assert sorted((c.topic, c.values, c.resolution) for c in backwards.conflicts()) == \
           sorted((c.topic, c.values, c.resolution) for c in ks.conflicts())


CHILD = """
import json, sys
sys.path[:0] = ["tests", "harness", "solution"]
from helpers import build
from test_invariants import DOWNLOADS, QUESTIONS
ks = build(*DOWNLOADS)
out = [[ks.answer(q, a).text for a in (None, "2026-03-09")] for q, _ in QUESTIONS]
out.append([[c.topic, c.values, c.resolution, c.reason, [s.url for s in c.sources]] for c in ks.conflicts()])
print(json.dumps(out))
"""


def test_two_fresh_processes_with_different_hash_seeds_give_identical_answers():
    runs = []
    for seed in ("1", "2"):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("BANKGPT_", "ANTHROPIC_"))}
        env.update(PYTHONHASHSEED=seed, BANKGPT_LLM="off")
        out = subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr[-2000:]
        runs.append(json.loads(out.stdout))
    assert runs[0] == runs[1]
