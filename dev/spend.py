#!/usr/bin/env python3
"""Development helper: add up the spend ledger that BANKGPT_SPEND_LOG writes.

    python dev/spend.py                          # results/spend.jsonl in this repository
    python dev/spend.py path/to/spend.jsonl

Each run that called the model appended one line at exit (see solution/kb/transport.py). This prints
the totals per label (BANKGPT_SPEND_LABEL), per model, and the grand total in USD. Calls to a
model without a price are counted but cannot be added to the dollars: they are shown apart.
Newer lines also give each model's call times ("seconds": median, 95th percentile, longest).
Those of separate runs do not add up, so they are not totalled here, and a line is read the
same with or without them.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "results" / "spend.jsonl"


def read(path: Path) -> tuple:
    """(runs, lines that could not be read): each run is (label, [(model, calls, input, output, usd or None)])."""
    runs, bad = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            models = [(str(model), int(m["calls"]), int(m["input_tokens"]), int(m["output_tokens"]),
                       None if m.get("usd") is None else float(m["usd"]))
                      for model, m in entry["models"].items()]
            runs.append((str(entry.get("label") or ""), models))
        except (ValueError, TypeError, KeyError, AttributeError):
            bad += 1                     # a line cut short or edited by hand: skipped whole, never half counted
    return runs, bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Add up the spend ledger that BANKGPT_SPEND_LOG writes.")
    ap.add_argument("ledger", nargs="?", default=str(DEFAULT), help="default: results/spend.jsonl")
    path = Path(ap.parse_args().ledger).expanduser()
    if not path.is_file():
        print(f"no ledger at {path}")
        return 1
    runs, bad = read(path)
    by_label = defaultdict(lambda: {"runs": 0, "calls": 0, "usd": 0.0, "unpriced": 0})
    by_model = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0, "usd": 0.0, "unpriced": 0})
    for label, models in runs:
        by_label[label]["runs"] += 1
        for model, calls, inp, out, cost in models:
            for row in (by_label[label], by_model[model]):
                row["calls"] += calls
                if cost is None:
                    row["unpriced"] += calls
                else:
                    row["usd"] += cost
            by_model[model]["input"] += inp
            by_model[model]["output"] += out

    def runs_(n) -> str:
        return f"{n} run{'' if n == 1 else 's'}"

    def unpriced(row) -> str:
        n = row["unpriced"]
        return f"  ({n} call{'' if n == 1 else 's'} with no price)" if n else ""

    def usd(row) -> str:
        return f"{'unknown':>12s}" if row["unpriced"] == row["calls"] > 0 else f"{row['usd']:12.6f}"

    print(f"{path}: {runs_(len(runs))}" + (f", {bad} unreadable lines skipped" if bad else ""))
    print(f"\n{'label':32s} {'runs':>6s} {'calls':>7s} {'USD':>12s}")
    for label, row in sorted(by_label.items(), key=lambda kv: -kv[1]["usd"]):
        print(f"{label or '(no label)':32s} {row['runs']:6d} {row['calls']:7d} {usd(row)}{unpriced(row)}")
    print(f"\n{'model':32s} {'calls':>7s} {'input tokens':>14s} {'output tokens':>14s} {'USD':>12s}")
    for model, row in sorted(by_model.items(), key=lambda kv: -kv[1]["usd"]):
        print(f"{model:32s} {row['calls']:7d} {row['input']:14,d} {row['output']:14,d} {usd(row)}{unpriced(row)}")
    total = {k: sum(row[k] for row in by_model.values()) for k in ("calls", "usd", "unpriced")}
    print(f"\ntotal: ${total['usd']:.6f} over {runs_(len(runs))} and {total['calls']} calls{unpriced(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
