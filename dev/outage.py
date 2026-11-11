#!/usr/bin/env python3
"""Development helper: run the organisers' scorer while the API fails in set ways, and see what
each failure costs the free-text answers and the latencies.

    python dev/outage.py                                  # every scenario, one after another
    python dev/outage.py --only 529-once --only blip
    python dev/outage.py --python path/to/python --out path/to/folder > table.txt

Each scenario starts the fake API in tests/fake_api.py with its own rules, then runs
harness/run_eval.py on eval/public_eval.jsonl in a fresh process pointed at it, with the default
reader: no ANTHROPIC_ or BANKGPT_ variable from this shell (so BANKGPT_LLM, BANKGPT_MODEL and the
spend controls are unset), ANTHROPIC_BASE_URL set to the fake, ANTHROPIC_API_KEY=fake-key, and
BANKGPT_DEBUG=1, which prints one block to stderr each time the model call returns and changes
nothing else. The scorer runs unchanged, except that each of its calls to answer() is also timed
on the monotonic clock (TIMED below). Loading the downloads takes about 35 s per run. The
scenarios run one after another, so their timings do not disturb each other. The interpreter
given by --python needs the `anthropic` package.

A call no rule fails is answered after 0.3 s with REPLY, which the reader accepts as an answer and
which carries MARKER into the answer text, so the answers that came from a model reply can be
counted. Accuracy against a canned reply means nothing, so it is not shown. The table:

    free-text   questions that reached the reader's complete(): it returned (a BANKGPT_DEBUG block)
                or it raised after the question reached the fake (the prompt's "Question:" line)
    model       answers that carry the marker: the model's reply was used
    fallback    free-text questions whose complete() returned no usable reply: the rules answered
    raised      free-text questions whose complete() raised: solution.py answered "I don't know"
    p50 p95 max the scorer's own latency figures (report.json; answers that failed are left out)
    longest     the longest answer on the monotonic clock. The scorer's own figures are on the wall
                clock, so on a machine whose wall clock jumps (see clock) they can be a few
                seconds more than the answer took
    >60s/err    answers over the scorer's 60 s limit, or that failed
    total       the 40 per-question latencies added up
    calls       requests that reached the fake, retries and the scorer's repeated questions included
    clock       how far the wall clock moved beyond the monotonic clock during the run. The scorer
                times answers with the wall clock; on a WSL2 machine it steps forward about 2 s at
                a time, and each step lands in whichever answer is running. 0.0 on a sound clock.

Each scenario's folder under --out keeps the scorer's output (per_question.jsonl, report.json), its
stdout and stderr, the fake's request log (fake_requests.jsonl), the monotonic time of each call
to answer() (answer_times.jsonl) and one line per question (questions.jsonl).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "dev"))
from fake_api import FakeAnthropic  # noqa: E402
from load import folders  # noqa: E402

EVAL = "eval/public_eval.jsonl"
MARKER = "FAKE-API-ANSWER"
# A reply in the form the reader asks for (SYSTEM in solution/kb/reader.py). Reader.read keeps it as an
# answer: nothing in it says "not stated", and it has no figure, link or number word to rewrite.
REPLY = f"QUOTES: none\nVERDICT: answer\nSOURCES: 1\nANSWER: {MARKER} (a canned reply from the local fake API)."
NORMAL = {"do": "reply", "delay": 0.3}          # every call no rule covers
FIRST = "<first model>"                         # stands for the model the reader tries first
# A 200 that is not a message: what a proxy or a captive portal sends.
PORTAL = {"do": "reply", "body": "<html><body>Sign in to use this network</body></html>",
          "headers": {"Content-Type": "text/html"}}
QUESTION = re.compile(r"^Question: (.*)$", re.M)   # the question's line in the reader's prompt

# One line per scenario: name, what happens, the fake's rules (see tests/fake_api.py). Calls are
# numbered in the order they reach the fake, retries included; the first rule that covers a call wins.
SCENARIOS = [
    ("control", "every call answered after 0.3 s", []),
    ("529-once", "one 529 on call 3", [{"calls": 3, "do": "status", "status": 529}]),
    ("429-once", "one 429 with retry-after: 1 on call 3", [{"calls": 3, "do": "status", "status": 429, "retry_after": 1}]),
    ("500-once", "one 500 on call 3", [{"calls": 3, "do": "status", "status": 500}]),
    ("blip", "calls 3 to 5 closed without an answer, then normal", [{"calls": [3, 5], "do": "close"}]),
    ("blip-6", "calls 3 to 8 closed without an answer, then normal", [{"calls": [3, 8], "do": "close"}]),
    ("hang-once", "call 3 hangs for 30 s, then normal", [{"calls": 3, "do": "hang", "seconds": 30}]),
    ("429-first-model", "every call to the first model gets 429 with retry-after: 2; the others answer",
     [{"model": FIRST, "do": "status", "status": 429, "retry_after": 2}]),
    ("429-all", "every call gets 429 with retry-after: 3", [{"do": "status", "status": 429, "retry_after": 3}]),
    ("gone", "every call from call 5 on closed without an answer", [{"calls": [5, None], "do": "close"}]),
    ("black-hole", "every call from call 5 on hangs for 60 s", [{"calls": [5, None], "do": "hang", "seconds": 60}]),
    ("bad-key", "every call gets 401", [{"do": "status", "status": 401}]),
    ("portal", "every call from call 3 on gets a 200 with an HTML page", [dict(PORTAL, calls=[3, None])]),
    ("cut-off", "call 3 gets a 200 whose body is cut off", [{"calls": 3, "do": "reply", "body": '{"id": "msg_cut", "ty'}]),
    ("400-once", "one 400 on call 3, after the model has answered", [{"calls": 3, "do": "status", "status": 400}]),
    ("404-first-model", "every call to the first model gets 404; the others answer",
     [{"model": FIRST, "do": "status", "status": 404}]),
    ("529-all", "every call gets 529 without retry-after", [{"do": "status", "status": 529}]),
    ("trickle", "every call from call 5 on gets its reply a byte at a time over 60 s",
     [{"calls": [5, None], "do": "trickle", "seconds": 60}]),
    ("hang-first-model", "every call to the first model hangs for 60 s; the others answer",
     [{"model": FIRST, "do": "hang", "seconds": 60}]),
    ("slow-first-model", "every call to the first model is answered after 27 s; the others answer at once",
     [{"model": FIRST, "do": "reply", "delay": 27}]),
    ("gone-15s", "every call from 1 s to 16 s after the first one closed without an answer, then normal",
     [{"from_s": 1, "to_s": 16, "do": "close"}]),
    ("403-after-first", "the first model answers call 1, then gets 403 on every call; the others answer",
     [{"calls": [2, None], "model": FIRST, "do": "status", "status": 403}]),
    ("400-question-1", "calls 1 to 3, the first question to every model, get 400; then normal",
     [{"calls": [1, 3], "do": "status", "status": 400}]),
]

# Runs the organisers' scorer as it is, with the submission's answer() also timed on the monotonic
# clock: one line per call to the file named first (the main pass, then the repeated questions).
TIMED = """
import json, sys, time
log, sys.argv = open(sys.argv[1], "w", encoding="utf-8"), ["harness/run_eval.py"] + sys.argv[2:]
sys.path.insert(0, "harness")
import run_eval
load = run_eval.load_submission

def load_timed(folder):
    ks = load(folder)
    answer = ks.answer

    def timed(question, as_of=None):
        start = time.monotonic()
        try:
            return answer(question, as_of)
        finally:
            log.write(json.dumps({"question": question, "seconds": time.monotonic() - start}) + "\\n")
            log.flush()
    ks.answer = timed
    return ks

run_eval.load_submission = load_timed
run_eval.main()
"""

# Asked of the chosen interpreter before any run: can it load the reader, and which model comes first.
PROBE = ("import json, sys; sys.path.insert(0, 'solution'); import anthropic; from kb import transport; "
         "print(json.dumps([anthropic.__version__, transport.AnthropicBackend.CHOICES]))")


def scorer_env(url: str) -> dict:
    """This environment without any ANTHROPIC_ or BANKGPT_ setting, pointed at the fake."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "BANKGPT_"))}
    env.update(ANTHROPIC_BASE_URL=url, ANTHROPIC_API_KEY="fake-key", BANKGPT_DEBUG="1",
               NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
    return env


def reader_models(python: str) -> list:
    """The models the reader tries, in order; exits when the interpreter cannot load the reader."""
    out = subprocess.run([python, "-c", PROBE], cwd=ROOT, env=scorer_env("http://127.0.0.1:9"),
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        sys.exit(f"{python} cannot load the reader (it needs the anthropic package):\n{out.stderr.strip()[-800:]}")
    sdk, models = json.loads(out.stdout.strip().splitlines()[-1])
    print(f"{python}: anthropic {sdk}; the reader tries {', '.join(models)}", file=sys.stderr)
    return models


def returned(stderr: str) -> dict:
    """{question: whether complete() returned a reply} for each call that returned, from the
    BANKGPT_DEBUG blocks: '--- <question>', the reply (or None), and a blank line."""
    lines = stderr.splitlines()
    return {line[4:].strip(): i + 1 < len(lines) and lines[i + 1] != "None"
            for i, line in enumerate(lines) if line.startswith("--- ")}


def jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def run(scenario: tuple, python: str, out: Path, first_model: str) -> dict:
    """Runs the scorer once against the fake with the scenario's rules; returns the table row."""
    name, what, rules = scenario
    rules = [dict(r, model=first_model) if r.get("model") == FIRST else r for r in rules]
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    # a failed run must not show the last one
    for old in ("per_question.jsonl", "report.json", "answer_times.jsonl", "questions.jsonl"):
        (folder / old).unlink(missing_ok=True)
    cmd = [python, "-c", TIMED, str(folder / "answer_times.jsonl"), "--submission", "solution", "--eval", EVAL,
           "--snapshots", "snapshots", "--out", str(folder)]
    with FakeAnthropic(rules, default=NORMAL, text=REPLY) as fake:
        with open(folder / "scorer.out", "w", encoding="utf-8") as stdout:
            with open(folder / "scorer.err", "w", encoding="utf-8") as stderr:
                start, wall_start = time.monotonic(), time.time()
                code = subprocess.run(cmd, cwd=ROOT, env=scorer_env(fake.url), stdout=stdout, stderr=stderr,
                                      timeout=3600).returncode
                seconds, wall = time.monotonic() - start, time.time() - wall_start
    calls = fake.calls
    for r in calls:
        found = QUESTION.search(r.pop("prompt") or "")
        r.update(question=found.group(1).strip() if found else None, since_start=round(r["t"] - start, 3))
    write_jsonl(folder / "fake_requests.jsonl", calls)
    row = {"name": name, "what": what, "calls": len(calls), "seconds": seconds, "clock": wall - seconds, "exit": code}
    if not (folder / "per_question.jsonl").exists():
        return row                                         # the scorer failed: see scorer.err

    came_back = returned((folder / "scorer.err").read_text(encoding="utf-8", errors="replace"))
    asked = {r["question"] for r in calls}
    lines = []
    timed = jsonl(folder / "answer_times.jsonl")
    for it, r, t in zip(jsonl(ROOT / EVAL), jsonl(folder / "per_question.jsonl"), timed):
        q = it["question"].strip()
        lines.append({"id": r["id"], "latency_s": r["latency_s"], "seconds": round(t["seconds"], 3), "error": r["error"],
                      "reached_reader": q in came_back or q in asked, "reader_returned": q in came_back,
                      "reader_got_reply": came_back.get(q, False),
                      "model_answer": MARKER in (r["answer"] or {}).get("text", "")})
    write_jsonl(folder / "questions.jsonl", lines)
    speed = json.loads((folder / "report.json").read_text(encoding="utf-8")).get("speed", {})
    row.update(free=sum(line["reached_reader"] for line in lines), model=sum(line["model_answer"] for line in lines),
               fallback=sum(line["reader_returned"] and not line["model_answer"] for line in lines),
               raised=sum(line["reached_reader"] and not line["reader_returned"] for line in lines),
               p50=speed.get("p50_s"), p95=speed.get("p95_s"), max=speed.get("max_s"),
               longest=max((line["seconds"] for line in lines), default=0.0),
               bad=sum(1 for line in lines if line["error"] or line["latency_s"] > 60),
               total=sum(line["latency_s"] for line in lines))
    return row


HEADER = (f"{'scenario':16s} {'free-text':>9s} {'model':>6s} {'fallback':>8s} {'raised':>6s}  "
          f"{'p50 / p95 / max (s)':>20s} {'longest':>7s} {'>60s/err':>8s} {'total (s)':>9s} {'calls':>6s} {'clock':>6s}"
          "  what happens")


def line(row: dict) -> str:
    if "free" not in row:
        return f"{row['name']:16s} the scorer failed (exit {row['exit']}): see its scorer.err  | {row['what']}"
    times = "/".join("  -  " if row[k] is None else f"{row[k]:5.2f}" for k in ("p50", "p95", "max"))
    return (f"{row['name']:16s} {row['free']:9d} {row['model']:6d} {row['fallback']:8d} {row['raised']:6d}  "
            f"{times:>20s} {row['longest']:7.2f} {row['bad']:8d} {row['total']:9.1f} {row['calls']:6d} {row['clock']:+6.1f}"
            f"  {row['what']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the scorer against a fake API that fails on cue.")
    ap.add_argument("--only", action="append", metavar="NAME",
                    help="run this scenario only (may be given more than once): " + ", ".join(s[0] for s in SCENARIOS))
    ap.add_argument("--python", default=sys.executable, help="the interpreter that runs the scorer (default: this one)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "outage", help="one folder per scenario goes here")
    args = ap.parse_args()

    unknown = sorted(set(args.only or ()) - {s[0] for s in SCENARIOS})
    if unknown:
        print(f"no scenario named {', '.join(unknown)}; there are: {', '.join(s[0] for s in SCENARIOS)}", file=sys.stderr)
        return 2
    folders(ROOT / "snapshots")                 # stops when there are none, as every dev tool does
    first_model = reader_models(args.python)[0]
    rows = []
    for scenario in SCENARIOS:
        if args.only and scenario[0] not in args.only:
            continue
        print(f"{scenario[0]}: {scenario[1]} ...", file=sys.stderr, flush=True)
        row = run(scenario, args.python, args.out.resolve(), first_model)
        rows.append(row)
        failed = f"   (the scorer exited with {row['exit']})" if row["exit"] else ""
        print(f"  {row['seconds']:.0f} s   {line(row)}{failed}", file=sys.stderr, flush=True)
    print(HEADER)
    for row in rows:
        print(line(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
