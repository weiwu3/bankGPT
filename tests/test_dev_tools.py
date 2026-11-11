"""The developer's tools in dev/ and the repository's .gitignore, checked on made-up data.

Each tool is loaded from its file as a module, or run as a script; no downloads, no key, no network.
"""
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import types

import pytest

from contract import Page, Snapshot
from helpers import CERTIFICATES, ROOT, SITE, table
from kb import transport

CERTS_OLD = table("Certificates", [("6 Month", "2.95%"), ("12 Month", "3.10%")], "01/02/2026")


def dev_tool(name):
    """dev/<name>.py, loaded as a module."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "dev" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_dev_tools_stop_when_there_are_no_downloads(tmp_path):
    # A fresh clone has no snapshot folders until snapshots_zip/ is unpacked: every dev tool read
    # nothing and reported a score of nothing.
    load = dev_tool("load")
    (tmp_path / "README.md").write_text("The snapshot folders go here.\n")
    with pytest.raises(SystemExit) as stop:
        load.snapshots(tmp_path)
    assert str(stop.value).startswith(f"no downloads in {tmp_path}")


def test_dev_run_stops_at_once_when_no_question_matches():
    out = subprocess.run([sys.executable, "-B", str(ROOT / "dev" / "run.py"), str(ROOT / "eval" / "public_eval.jsonl"),
                          "--id", "no-such-question"], capture_output=True, text=True, timeout=300,
                         env=dict(os.environ, BANKGPT_LLM="off"))
    assert out.returncode != 0 and "no question to run" in out.stderr and "ingest" not in out.stdout


def test_dev_spend_adds_up_ledger_lines_with_and_without_call_times(monkeypatch, tmp_path, capsys):
    spend = dev_tool("spend")
    ledger = tmp_path / "spend.jsonl"
    before = {"calls": 2, "input_tokens": 1_000, "output_tokens": 100, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0, "usd": 0.0015}            # a line from before call times were kept
    ledger.write_text(json.dumps({"time": 1, "label": "before", "models": {"claude-haiku-4-5": before}}) + "\n")
    for tokens, seconds in (((2_000, 300), 1.25), ((1_000, 100), 2.75)):      # two replies, as the meter reads them
        usage = types.SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1])
        transport.USAGE.add("claude-sonnet-5-5", types.SimpleNamespace(usage=usage), seconds)
    transport.USAGE.log(str(ledger), "after")                                    # and one written now, with them
    lines = [json.loads(x)["models"] for x in ledger.read_text().splitlines()]
    assert "seconds" not in lines[0]["claude-haiku-4-5"]
    assert lines[1]["claude-sonnet-5-5"]["seconds"] == {"median": 2.0, "p95": 2.75, "max": 2.75}

    assert spend.read(ledger) == ([("before", [("claude-haiku-4-5", 2, 1_000, 100, 0.0015)]),
                                   ("after", [("claude-sonnet-5-5", 2, 3_000, 400, 0.01)])], 0)
    monkeypatch.setattr(sys, "argv", ["spend.py", str(ledger)])
    assert spend.main() == 0
    assert "total: $0.011500 over 2 runs and 4 calls" in capsys.readouterr().out


def dev_conflicts(monkeypatch, tree, *args):
    """dev/conflicts.py copied to <tree>/dev and set up to run over two made-up downloads whose
    certificate rates differ. <tree> has no dev/_local/, so no gold file is found unless one is
    given. Returns the script's path and a list that grows each time the downloads are read."""
    snaps = [Snapshot(f"{n}_{day}", day, [Page(f"{SITE}/rates", "rates", md, f"{day}T00:00:00Z")])
             for n, (day, md) in enumerate([("2026-01-05", CERTS_OLD), ("2026-03-05", CERTIFICATES)], 1)]
    reads = []
    (tree / "dev").mkdir(exist_ok=True)
    script = shutil.copy(ROOT / "dev" / "conflicts.py", tree / "dev" / "conflicts.py")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "load", types.SimpleNamespace(snapshots=lambda: reads.append(1) or snaps))
    monkeypatch.setattr(sys, "argv", ["conflicts.py", *map(str, args)])
    return str(script), reads


def test_dev_conflicts_lists_the_gold_entries_the_score_counts_as_missed(tmp_path, monkeypatch, capsys):
    # The scorer pairs reported and gold entries one to one: two gold entries for the one reported
    # contradiction leave one of them unmatched, and the list under the score must name it.
    gold = [{"id": f"g-{n}", "topic_keywords": ["certificate"], "values": ["3.50%", "2.95%"], "correct_value": "3.50%"}
            for n in (1, 2)]
    (tmp_path / "gold.json").write_text(json.dumps(gold))
    script, _ = dev_conflicts(monkeypatch, tmp_path, tmp_path / "gold.json")
    runpy.run_path(script, run_name="__main__")
    out = capsys.readouterr().out
    score, _, missed = out.partition("gold entries with no match")
    first = out.splitlines()[0]
    assert first.startswith("1 conflicts in ") and "; score vs gold.json: {" in first
    assert "'matched': 1" in score and missed.count("g-") == 1


def test_dev_conflicts_prints_the_report_and_how_to_score_it_without_an_answer_list(tmp_path, monkeypatch, capsys):
    # The gold file is private and not in a fresh clone: the tool printed only an error then.
    script, _ = dev_conflicts(monkeypatch, tmp_path)
    runpy.run_path(script, run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("1 conflicts in ") and "score" not in lines[0]
    assert lines[1].startswith("- ") and "2.95%" in lines[1]
    assert lines[-1] == "no score: scoring needs a gold file, given as the first argument"


def test_dev_conflicts_stops_before_the_ingest_on_an_answer_list_that_is_not_there(tmp_path, monkeypatch):
    script, reads = dev_conflicts(monkeypatch, tmp_path, tmp_path / "missing.json")
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(script, run_name="__main__")
    assert str(stop.value) == f"no gold file at {tmp_path / 'missing.json'}" and reads == []


def baseline_record(variant, qid, **fields):
    """One line of dev/baseline.py's records.jsonl, as its scoring loop writes it."""
    excerpt = {"snapshot": "01_2026-05-12", "url": f"{SITE}/rates", "heading": "Rates", "score": 1.0, "text": "3.50% APY"}
    record = {"variant": variant, "file": "public_eval.jsonl", "id": qid, "category": "current_value", "group": None,
              "weight": 1.0, "as_of": None, "question": "What is the APY?", "must_include": ["3.50%"], "must_not_include": [],
              "answer": "3.50% APY", "confident": True, "sources": [], "score": 1.0, "correctness": 1.0, "trap_hits": [],
              "missed": [], "latency_s": 0.1, "jobs": 1,
              "evidence": {"path": "rows"} if variant == "D" else {"excerpts": [excerpt]}}
    record.update(fields)
    return record


def test_dev_baseline_summarizes_a_solution_run_that_has_not_finished(tmp_path, capsys):
    # A run of the solution (D) cut short, or run on fewer files, covers fewer questions than A.
    for variant, ids in (("A", ["q-1", "q-2"]), ("D", ["q-1"])):
        (tmp_path / variant).mkdir()
        (tmp_path / variant / "records.jsonl").write_text("".join(json.dumps(baseline_record(variant, i)) + "\n" for i in ids))
    dev_tool("baseline").summarize([str(tmp_path / "A"), str(tmp_path / "D")])
    assert "D-route rows     n=  1" in capsys.readouterr().out


def test_dev_baseline_says_where_a_trap_value_came_from(tmp_path, capsys):
    # As of June 15 the June 9 download is in force; the trap value is in the May and the September ones.
    excerpts = [{"snapshot": s, "url": f"{SITE}/rates", "heading": "Rates", "score": 1.0, "text": f"{apy} APY"}
                for s, apy in (("01_2026-05-12", "4.00%"), ("03_2026-06-09", "4.25%"), ("05_2026-09-28", "4.00%"))]
    record = baseline_record("A", "q-1", as_of="2026-06-15", answer="4.00% APY", score=0.0, correctness=0.0,
                             must_include=["4.25%"], must_not_include=["4.00%"], trap_hits=["4.00%"],
                             evidence={"excerpts": excerpts})
    (tmp_path / "records.jsonl").write_text(json.dumps(record) + "\n")
    dev_tool("baseline").summarize([str(tmp_path)])
    assert "A: where the trap values in its answers came from: in older and newer downloads 1" in capsys.readouterr().out


def test_dev_baseline_asks_the_model_once_per_question():
    # The API transport asks again by itself, within a question's 40 s. A second call from the
    # baseline would let one answer take twice that, over the scorer's 60 s.
    baseline = dev_tool("baseline")
    asked = []

    class Silent:
        def complete(self, system, user):
            asked.append(user)
            return None
    ks = baseline.SearchAndAsk("A", backend=Silent())
    day = "2026-06-10"
    ks.ingest(Snapshot(f"01_{day}", day, [Page(f"{SITE}/lobby", "lobby", "# Lobby\n\nThe Elm Hollow lobby opens at "
                                                                         "nine.\n", f"{day}T00:00:00Z")]))
    answer = ks.answer("When does the Elm Hollow lobby open?")
    assert answer.text == "I don't know." and len(asked) == 1 and ks.failures == 1


def test_dev_outage_shows_nothing_of_the_last_run_when_the_scorer_fails(tmp_path):
    false = shutil.which("false")                       # stands in for an interpreter whose scorer run fails
    if false is None:
        pytest.skip("no `false` command here")
    outage = dev_tool("outage")
    scenario = outage.SCENARIOS[0]
    folder = tmp_path / scenario[0]
    folder.mkdir()
    kept = ("questions.jsonl", "per_question.jsonl", "report.json", "answer_times.jsonl")
    for name in kept:
        (folder / name).write_text('{"from": "the last run"}\n')
    assert outage.run(scenario, false, tmp_path, "claude-haiku-4-5")["exit"] != 0
    assert [name for name in kept if (folder / name).exists()] == []


def test_snapshot_folders_are_kept_out_of_git_whether_unpacked_or_linked(tmp_path):
    # dev/outage.py says to "link or unpack the five snapshot folders": a link is no folder to git.
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    (tmp_path / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    (tmp_path / "snapshots" / "02_unpacked").mkdir(parents=True)
    (tmp_path / "snapshots" / "02_unpacked" / "manifest.json").write_text("{}\n")
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", tmp_path / "snapshots" / "01_linked")
    env = dict(os.environ, HOME=str(tmp_path), GIT_CONFIG_NOSYSTEM="1")       # no one's own git settings

    def git_run(*args):
        return subprocess.run([git, "-c", f"core.excludesFile={os.devnull}", *args], cwd=tmp_path, env=env,
                              capture_output=True, text=True, check=True).stdout
    git_run("init", "-q")
    assert "snapshots" not in git_run("status", "--porcelain", "--untracked-files=all")
