"""The contract with the organisers' scorer, end to end.

build() gives an object with ingest(), answer() and conflicts(); answers come back as the scorer's
Answer shape, never raise, and come back in time whatever the question or the date. The scorer
itself (harness/run_eval.py) runs on its self-test site in a fresh process. Made-up sites and the
kit's own self-test site; no downloads, no key, no network.
"""
import datetime
import json
import os
import re
import subprocess
import sys
import time

import pytest

from helpers import REFUSAL, ROOT, SELFTEST, build, snapshot, table
from contract import Answer, Conflict, KnowledgeSystem, Source
from run_eval import load_snapshots, to_answer
import solution

QUESTION = "What is the APY on a 6-month Wren certificate?"
SITE = {"rates": table("Wren Certificates", [("6 Month", "1.65%"), ("12 Month", "2.05%")], "02/02/2026"),
        "contact-us": "# Contact us\n\nCall us at 555-010-4477 or visit any branch during business hours.\n",
        "about": "# About\n\nWe have served the Larkspur valley since 1961 from three branches.\n"}
FAST_S = 2.0          # the scorer's speed pillar is full at a p95 of 2 s; it gives up on an answer after 60 s


def clean_env():
    """The shell's settings for the solution taken out, and the model off."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BANKGPT_", "ANTHROPIC_"))}
    env["BANKGPT_LLM"] = "off"
    return env


def check_shape(answer):
    assert isinstance(answer, Answer)
    assert isinstance(answer.text, str) and isinstance(answer.confident, bool)
    assert isinstance(answer.sources, list) and all(isinstance(s, Source) and s.url.startswith("https://") for s in answer.sources)
    assert answer.conflict_note is None or isinstance(answer.conflict_note, str)
    if not answer.confident:
        assert answer.text == REFUSAL


@pytest.fixture(scope="module")
def ks():
    return build(("2026-01-10", SITE), ("2026-02-10", SITE))


def test_build_gives_a_knowledge_system_the_scorer_accepts():
    ks = solution.build()
    assert isinstance(ks, KnowledgeSystem)
    assert all(callable(getattr(ks, m)) for m in ("ingest", "answer", "conflicts"))


def test_the_scorer_loads_the_submission_folder_in_a_fresh_process():
    code = ("import sys; sys.path.insert(0, 'harness'); from pathlib import Path; import run_eval; "
            "ks = run_eval.load_submission(Path('solution')); print(type(ks).__name__)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=clean_env(), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == "KnowledgeSystem", out.stderr


def test_the_scorer_scores_its_own_selftest_site_in_full(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "harness" / "run_eval.py"), "--submission", str(ROOT / "solution"),
                          "--snapshots", str(SELFTEST / "snapshots"), "--eval", str(SELFTEST / "eval.jsonl"),
                          "--conflicts-gold", str(SELFTEST / "conflicts_gold.json"), "--out", str(tmp_path)],
                         cwd=tmp_path, env=clean_env(), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["errors"] == 0 and report["stale_trap_hits"] == 0
    assert report["pillars"] == {"accuracy": 1.0, "consistency": 1.0, "speed": 1.0, "conflicts": 1.0}
    assert report["final_score"] == 1.0
    lines = (tmp_path / "per_question.jsonl").read_text().splitlines()
    assert len(lines) == 5 and all(json.loads(line)["score"] == 1.0 for line in lines)


def test_ingest_oldest_first_gives_one_view_per_download_in_order():
    ks = solution.build()
    for snap in load_snapshots(SELFTEST / "snapshots"):
        ks.ingest(snap)
    assert [v.snapshot_id for v in ks.engine.store.views] == ["01_2026-01-05_baseline", "02_2026-03-01_update"]


def test_an_answer_has_the_contract_s_shape_and_cites_the_download_in_force(ks):
    now, then = ks.answer(QUESTION), ks.answer(QUESTION, "2026-01-20")
    for answer in (now, then):
        check_shape(answer)
        assert to_answer(answer) == answer                  # the scorer's reading of it keeps every field
        assert answer.confident and answer.sources[0].url == "https://cu.test/rates"
    assert {s.snapshot_id for s in now.sources} == {"02_2026-02-10"}
    assert {s.snapshot_id for s in then.sources} == {"01_2026-01-10"}


def test_conflicts_entries_have_every_field():
    ks = build(("2026-01-05", {"rates": table("Wren Certificates", [("6 Month", "2.10%")])}),
               ("2026-03-05", {"rates": table("Wren Certificates", [("6 Month", "2.60%")])}))
    [entry] = ks.conflicts()
    assert isinstance(entry, Conflict) and entry.topic and entry.resolution and entry.reason
    assert len(entry.values) >= 2 and all(isinstance(v, str) for v in entry.values)
    assert all(isinstance(s, Source) and s.url and s.snapshot_id and s.quote for s in entry.sources)


def test_before_any_download_every_call_answers_honestly():
    ks = solution.build()
    ks.engine.reader = None
    answer = ks.answer(QUESTION)
    check_shape(answer)
    assert not answer.confident and answer.sources == [] and ks.conflicts() == []


def test_a_question_that_breaks_the_code_gets_the_fixed_refusal(ks, monkeypatch):
    def broken(question, as_of):
        raise ValueError("a question no rule was written for")
    monkeypatch.setattr(ks.engine, "answer", broken)
    answer = ks.answer(QUESTION)
    check_shape(answer)
    assert (answer.text, answer.confident, answer.sources) == (REFUSAL, False, [])


@pytest.mark.parametrize("first", [True, False])
def test_an_empty_download_breaks_nothing(first):
    ks = solution.build()
    ks.engine.reader = None
    days = ["2026-01-05", "2026-03-05"]
    order = [{}, SITE] if first else [SITE, {}]
    for n, (day, pages) in enumerate(zip(days, order), 1):
        ks.ingest(snapshot(n, day, pages))
    for as_of in (None, "2026-01-10"):
        check_shape(ks.answer(QUESTION, as_of))
    assert isinstance(ks.conflicts(), list)


ODD_QUESTIONS = [
    None, "", "   ", "\t\n", "?", "\x00\x07\x1b[31m", "\ud800", "​" * 50, "利率是多少", "ما هو سعر الفائدة؟",
    "Какая ставка?", "💰💰 apy?", b"6 month wren certificate apy", 42, 3.5, ["a list"], {"a": "dict"}, object(),
    "apy " * 25_000, "6 month " * 12_000, "$" * 50_000, "9" * 100_000, "-" * 100_000, "%" * 100_000,
    QUESTION + "​" * 1000, QUESTION.upper(), "<script>alert(1)</script>", "'; DROP TABLE rates; --",
    "{question}", "%s %d", "\\x00",
]


@pytest.mark.parametrize("question", ODD_QUESTIONS, ids=lambda q: repr(q)[:30])
def test_any_question_gets_an_answer_of_the_contract_s_shape_in_time(ks, question):
    start = time.perf_counter()
    answer = ks.answer(question)
    assert time.perf_counter() - start < FAST_S
    check_shape(answer)


ODD_DATES = [None, "", "2026-01-15", datetime.date(2026, 1, 15), datetime.datetime(2026, 1, 15, 9, 30), 20260115, 2026.0,
             b"2026-01-15", ["2026-01-15"], {"day": 15}, object(), "nonsense", "2026-13-45", "1900-01-01", "9999-12-31",
             "x" * 100_000, "\x00", "15/01/2026"]


@pytest.mark.parametrize("as_of", ODD_DATES, ids=lambda d: repr(d)[:30])
def test_any_date_gets_an_answer_of_the_contract_s_shape_in_time(ks, as_of):
    start = time.perf_counter()
    answer = ks.answer(QUESTION, as_of)
    assert time.perf_counter() - start < FAST_S
    check_shape(answer)
    assert "1.65%" in answer.text and "2.05%" not in answer.text      # a date never changes which tier


@pytest.mark.parametrize("as_of", ODD_DATES, ids=lambda d: repr(d)[:30])
def test_a_date_is_said_in_one_form_and_one_that_cannot_be_read_is_not_said(ks, as_of):
    # Read: printed as an ISO date ("1900-01-01" is read, and answered from the oldest download). Not read
    # (an object, a list, 100,000 letters): the answer is for the newest download and claims no date.
    readable = [("2026-01-15", "2026-01-15"), (datetime.date(2026, 1, 15), "2026-01-15"),
                (datetime.datetime(2026, 1, 15, 9, 30), "2026-01-15"), ("1900-01-01", "1900-01-01"),
                ("9999-12-31", "9999-12-31")]
    for question in (QUESTION, "Since when have you served the valley?"):
        said = re.findall(r"^As of (.*?) \(", ks.answer(question, as_of).text)
        assert said == [iso for day, iso in readable if day == as_of], question


@pytest.mark.parametrize("as_of, download", [("2026-01-15", "01_2026-01-10"), (datetime.date(2026, 1, 15), "01_2026-01-10"),
                                              ("1900-01-01", "01_2026-01-10"), ("nonsense", "02_2026-02-10"),
                                              (None, "02_2026-02-10")])
def test_the_sources_name_the_download_the_answer_came_from(ks, as_of, download):
    assert {s.snapshot_id for s in ks.answer(QUESTION, as_of).sources} == {download}


@pytest.mark.parametrize("unit", ["x", "7", " ", "$", "apr 2026 "])
def test_a_very_long_question_takes_time_in_proportion_to_its_length(unit):
    # 300,000 characters take about four times as long as 75,000 when every pattern applied to the
    # question is linear; a pattern that backtracks over the rest of the text at each position takes
    # sixteen times as long. The faster of two runs, so that a busy machine does not decide. In a fresh
    # process, so that a runaway is stopped instead of holding up the suite.
    code = ("import sys, time; sys.path[:0] = ['tests', 'harness', 'solution']; from helpers import build, table; "
            "ks = build(('2026-02-10', {'rates': table('Wren Certificates', [('6 Month', '1.65%')])})); "
            f"q = {unit!r} * 300_000\n"
            "def took(n):\n"
            "    ks.engine._answers.clear(); t = time.perf_counter(); ks.answer(q[:n]); return time.perf_counter() - t\n"
            "print(min(took(75_000) for _ in range(2)), min(took(300_000) for _ in range(2)))")
    try:
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=clean_env(), capture_output=True, text=True,
                             timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("no answer in time")
    assert out.returncode == 0, out.stderr
    short, long = map(float, out.stdout.split())
    assert long < 6 * max(short, 0.05) and long < 10

