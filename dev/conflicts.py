#!/usr/bin/env python3
"""Development helper: list the contradictions the solution reports and, given a gold file, score them.

    python dev/conflicts.py                       # dev/_local/conflicts.json if it is there
    python dev/conflicts.py path/to/gold.json -n 80

The gold file is private and not in the repository. Without one the report is still printed;
only the score is left out.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BANKGPT_LLM", "off")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "dev" / "_local" / "conflicts.json"
sys.path.insert(0, str(ROOT / "dev"))
from load import snapshots  # noqa: E402
from scoring import contains, score_conflicts, _norm  # noqa: E402
import solution  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="List the solution's conflicts and, given a gold file, score them.")
    ap.add_argument("gold", nargs="?",
                    help="the gold file to score against (default: dev/_local/conflicts.json when it is "
                         "there; it is kept out of git, so without it only the report is printed)")
    ap.add_argument("-n", type=int, default=60, help="how many reported conflicts to list (default 60)")
    args = ap.parse_args()
    if args.gold is None:
        gold_path = DEFAULT_GOLD if DEFAULT_GOLD.is_file() else None
    else:
        gold_path = Path(args.gold)
        if not gold_path.is_file():              # before the 35 s of ingest, not after
            sys.exit(f"no gold file at {gold_path}")

    ks = solution.build()
    for s in snapshots():
        ks.ingest(s)
    t = time.time()
    found = ks.conflicts()
    head = f"{len(found)} conflicts in {time.time() - t:.2f}s"
    if gold_path is None:
        print(head)
    else:
        gold = json.loads(gold_path.read_text())
        print(f"{head}; score vs {gold_path.name}: {score_conflicts(gold, found)}")
    for c in found[:args.n]:
        print(f"- {c.topic[:70]:70s} | {c.values[:8]} | => {str(c.resolution)[:60]}")
    if gold_path is None:
        print("\nno score: scoring needs a gold file, given as the first argument")
        return
    print("\ngold entries with no match (as in the score, a reported conflict counts for one gold entry):")
    # The same one-to-one pairing as harness/scoring.py's score_conflicts, written out again because
    # that returns only the counts, not which gold entries went unmatched.
    used = set()
    for g in gold:
        hit = False
        for i, c in enumerate(found):
            if i in used:
                continue
            vals = " | ".join(c.values)
            v = sum(contains(vals, x) for x in g["values"])
            k = any(_norm(x) in _norm(c.topic) for x in g.get("topic_keywords", []))
            if v >= 2 or (v >= 1 and k):
                hit = True
                used.add(i)
                break
        if not hit:
            print(f"  {g['id']:10s} {g['topic_keywords']} {g['values']} -> {g.get('correct_value')}")


if __name__ == "__main__":
    main()
