"""Checks on the five real downloads, without a model.

The downloads are not in the repository: put them (or links to them) in snapshots/ as the README
says. Without them every test here is skipped. Loading them takes most of a minute, so the tests are
marked slow: `python -m pytest tests -m "not slow"` leaves them out.

Every expected value comes from the organisers' own practice file, eval/public_eval.jsonl, and is
read when the test runs: no text of the website is written here.
"""
import json
import time

import pytest

from helpers import ASK, ROOT
from run_eval import load_snapshots
from scoring import contains, score_consistency, score_item, summarize
import solution

DOWNLOADS = ROOT / "snapshots"
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not any(DOWNLOADS.glob("0*/manifest.json")), reason="the real downloads are not in snapshots/")]
ITEMS = [json.loads(line) for line in (ROOT / "eval" / "public_eval.jsonl").read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def loaded():
    """The system with every download ingested, oldest first, and how long that took."""
    ks = solution.build()
    ks.engine.reader = None
    start = time.perf_counter()
    for snap in load_snapshots(DOWNLOADS):
        ks.ingest(snap)
    return ks, time.perf_counter() - start


@pytest.fixture(scope="module")
def answered(loaded):
    """Each practice item with its answer, the route it took and how long it took."""
    ks, _ = loaded
    out = []
    for item in ITEMS:
        start = time.perf_counter()
        answer = ks.answer(item["question"], item.get("as_of"))
        out.append((item, answer, ks.engine.answer(item["question"], item.get("as_of")).path, time.perf_counter() - start))
    return out


def test_the_five_downloads_load_within_two_minutes(loaded):
    ks, seconds = loaded
    assert len(ks.engine.store.views) == 5 and seconds < 120


def test_no_answer_takes_more_than_a_second(answered):
    slow = [(item["id"], round(s, 2)) for item, _, _, s in answered if s > 1.0]
    assert not slow


def test_the_practice_set_without_a_model_scores_at_least_three_quarters(answered):
    results = []
    for item, answer, _, seconds in answered:
        result = score_item(item, answer)
        result.update(latency_s=seconds, answer={"text": answer.text})
        results.append(result)
    report = summarize(results, None, [{"id": r["id"], "text": r["answer"]["text"]} for r in results])
    assert report["qa_score"] >= 0.75 and report["errors"] == 0
    assert report["consistency"]["score"] == 1.0 and report["speed"]["score"] == 1.0


def test_each_practice_question_the_rules_answer_keeps_its_values_and_none_it_must_not(answered):
    by_rule = [(item, answer) for item, answer, path, _ in answered if path in ("rows", "hours")]
    assert len(by_rule) >= 15
    for item, answer in by_rule:
        assert answer.confident, item["id"]
        assert all(contains(answer.text, v) for v in item.get("must_include", [])), (item["id"], answer.text)
        assert not any(contains(answer.text, v) for v in item.get("must_not_include", [])), (item["id"], answer.text)


def test_every_practice_question_to_refuse_is_refused_with_no_number_of_its_own(answered):
    refusals = [(item, answer) for item, answer, _, _ in answered if item.get("category") == "should_refuse"]
    assert refusals
    for item, answer in refusals:
        assert score_item(item, answer)["score"] == 1.0 and answer.text.endswith(ASK), item["id"]


def test_reworded_practice_questions_agree(answered):
    results = [dict(score_item(item, answer), answer={"text": answer.text}) for item, answer, _, _ in answered]
    assert score_consistency(results, [])["paraphrase_consistency"] == 1.0


def test_the_contradiction_report_has_between_15_and_45_well_formed_entries(loaded):
    ks, _ = loaded
    found = ks.conflicts()
    assert 15 <= len(found) <= 45
    snapshot_ids = {v.snapshot_id for v in ks.engine.store.views}
    for entry in found:
        assert entry.topic.strip() and entry.reason.strip() and entry.resolution, entry.topic
        assert len(entry.values) >= 2 and len(set(entry.values)) == len(entry.values), entry.topic
        assert any(v in entry.resolution for v in entry.values), entry.topic
        assert entry.sources and all(s.url.startswith("http") and s.snapshot_id in snapshot_ids for s in entry.sources)
    assert len({e.topic for e in found}) == len(found)
