#!/usr/bin/env python3
"""Development helper: run a question file through the solution and show each answer.

    python dev/run.py eval/public_eval.jsonl            # a line per question, three more for one that loses points
    python dev/run.py eval/public_eval.jsonl --fail     # only the ones that lose points
    python dev/run.py eval/public_eval.jsonl --id dup-016 -v

Scores with the organisers' own scoring.py. `harness/run_eval.py` remains the official run.
The reader is used when ANTHROPIC_API_KEY is set; BANKGPT_LLM=off forces rules only.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dev"))
import load  # noqa: E402  (puts harness/ and solution/ on the path)
from scoring import score_item  # noqa: E402
import solution  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--fail", action="store_true")
    ap.add_argument("--id", action="append")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--width", type=int, default=230)
    ap.add_argument("--half", choices=["a", "b"], help="a = the half used while developing, b = the half kept unseen")
    ap.add_argument("--jobs", type=int, default=1, help="answer in parallel first (accuracy only; timings are not meaningful)")
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    args = ap.parse_args()

    import zlib
    from concurrent.futures import ThreadPoolExecutor
    items = [it for it in load.items(args.files, args.id)
             if not args.half or "ab"[zlib.crc32(str(it.get("group") or it["id"]).encode()) % 2] == args.half]
    if not items:                            # before the 35 s of ingest, not after
        sys.exit("no question to run: the files are empty or none matches --id/--half")

    ks = solution.build()
    t = time.time()
    for s in load.snapshots():
        ks.ingest(s)
    print(f"ingest {time.time() - t:.1f}s  reader={'on' if ks.engine.reader else 'off'}")

    if args.jobs > 1:
        with ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(lambda it: ks.answer(it["question"], it.get("as_of")), items))

    total = weight = 0.0
    by_cat, lat = {}, []
    paths = {}
    for it in items:
        t = time.perf_counter()              # not time.time(): the wall clock can jump
        ans = ks.answer(it["question"], it.get("as_of"))
        dt = time.perf_counter() - t
        lat.append(dt)
        r = score_item(it, ans)
        w = it.get("weight", 1.0)
        total += r["score"] * w
        weight += w
        c = by_cat.setdefault(it.get("category", "?"), [0.0, 0])
        c[0] += r["score"]                   # by category a plain mean, as the scorer's report gives it
        c[1] += 1
        path = ks.engine.answer(it["question"], it.get("as_of")).path
        pp = paths.setdefault(path, [0.0, 0.0])
        pp[0] += r["score"] * w
        pp[1] += w
        if args.quiet or (args.fail and r["score"] >= 0.999):
            continue
        print(f"{load.flag(r['score'])}{it['id']:14s} {r['score']:.2f} {it.get('category', '')[:12]:12s} {path:8s} {dt:5.2f}s "
              f"as_of={it.get('as_of') or '-':10s} Q: {it['question'][:args.width]}")
        if r["score"] < 0.999 or args.verbose:
            print(f"      missed={r['missed']} traps={r['trap_hits']} src={r['source_hit']} conf={ans.confident}")
            print(f"      A: {ans.text[:args.width * 2]}")
            print(f"      want: {it.get('expected_answer', '')[:args.width]}")
            if args.verbose:
                print(f"      sources: {[s.url for s in ans.sources]}")
                if ans.conflict_note:
                    print(f"      note: {ans.conflict_note[:300]}")
    lat.sort()
    print(f"\naccuracy {total / max(weight, 1e-9):.4f} over {len(lat)} questions; "
          f"p50 {lat[len(lat) // 2]:.2f}s p95 {lat[min(len(lat) - 1, int(0.95 * len(lat)))]:.2f}s")
    print("  ".join(f"{k}={v[0] / v[1]:.2f}" for k, v in sorted(by_cat.items())))
    print("by route: " + "  ".join(f"{k}={v[0] / v[1]:.2f} (w{v[1]:.0f})" for k, v in sorted(paths.items())))


if __name__ == "__main__":
    main()
