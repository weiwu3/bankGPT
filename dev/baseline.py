#!/usr/bin/env python3
"""Ablation baselines for the rules-versus-model question: search and ask, nothing else.

    A  every passage of every download in one pool; `as_of` is ignored.
    B  only the download in force: the latest, or the last one on or before `as_of`.

Both cut pages with kb.passages.split, rank passages with a plain BM25 written here (no page-kind
or recency priors, no de-duplication, no teaser removal, no synonyms), and give the top passages
to the model the solution would use (kb.reader.make_reader: ANTHROPIC_API_KEY),
shown as the solution's reader shows a passage, with the date of the download in force and a
short instruction: answer from these excerpts, giving only the value current on that date (a
rule the solution's reader has too), or say you don't know. conflicts() returns an empty list.
Nothing here is part of the submission.

Scoring loop: one record per question (answer, score, latency, the excerpts the model was given)
is appended to <out>/records.jsonl as soon as it is answered; --summarize prints the tables from
one or more such files, counting each question once, by its newest record. A question that got
no answer (the model unavailable or switched off for the run, an error) is recorded as a miss.
Running the same command again continues where the last run stopped: it skips the recorded
questions, except those misses, which it asks again.

    ANTHROPIC_API_KEY=... python3 dev/baseline.py --variant A eval/public_eval.jsonl dev/_local/questions/*.jsonl --out DIR
    ANTHROPIC_API_KEY=... python3 dev/baseline.py --variant B ... --out DIR
    ANTHROPIC_API_KEY=... python3 dev/baseline.py --variant D ... --out DIR   # the solution as it is
    python3 dev/baseline.py --variant C ... --out DIR                         # the solution, rules only
    python3 dev/baseline.py --summarize DIR_A DIR_B ...

Questions run one at a time by default, so latencies are those of a lone caller; --part k/n
runs the k-th of n slices, for runs longer than one sitting. Under the organisers' scorer (which
counts a question the model did not answer as a refusal, full marks where one is expected; the
loop here counts it as a miss):

    BASELINE_VARIANT=B ANTHROPIC_API_KEY=... python3 harness/run_eval.py --submission dev/baseline_submission \\
        --eval eval/public_eval.jsonl --out DIR
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "harness", ROOT / "solution"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from contract import Answer, Source  # noqa: E402  (the scorer's own shapes)
from kb.reader import passage_text  # noqa: E402
from kb.passages import split  # noqa: E402


K = 20                    # excerpts given to the model (the solution gives it about twenty passages)
DONT_KNOW = "I don't know."

SYSTEM = """You answer questions from members of a credit union. Use only the excerpts from the \
credit union's website given in the message. If the excerpts do not say, answer "I don't know." \
Give only the value that is current on the date given: never mention an older or superseded value, \
not even to compare. Keep the answer short: one to three sentences.

Reply in exactly this form:
ANSWER: the answer
SOURCES: the numbers of the excerpts you used, comma separated, or none"""

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""
a an the and or but if of to in on at by for from with about as into than then that this these those
is are was were be been being am do does did have has had can could will would shall should may might must
i me my we us our you your he she it its they them their there here what which who whom whose when where why how
not no so just also any some all each other such only very
""".split())


def _tokens(s: str) -> list:
    """Lower-case words and numbers, stop words out, a plural 's' off. Nothing else."""
    out = []
    for w in _WORD.findall(s.lower()):
        if w in _STOP or (len(w) == 1 and w.isalpha()):
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _shown(passage) -> str:
    """The excerpt as the model sees it, by the solution's own function, so that both are shown the
    same text: links kept, images dropped, line breaks kept (a table stays a table), the whole passage."""
    return passage_text(passage)


class Index:
    """Plain BM25 (k1 1.2, b 0.75) over passages; a passage is its page title, heading and text."""

    K1, B = 1.2, 0.75

    def __init__(self) -> None:
        self.docs: list = []                    # (passage, snapshot_id)
        self.lengths: list = []
        self.postings: dict = defaultdict(list)  # term -> [(doc number, count)]
        self.total = 0

    def add(self, passage, snapshot_id: str) -> None:
        i = len(self.docs)
        toks = _tokens(f"{passage.page.title} {passage.heading} {passage.text}")
        self.docs.append((passage, snapshot_id))
        self.lengths.append(len(toks))
        self.total += len(toks)
        for t, c in Counter(toks).items():
            self.postings[t].append((i, c))

    def search(self, query: str, k: int) -> list:
        n = len(self.docs)
        avg = self.total / max(1, n)
        scores: dict = defaultdict(float)
        for t in set(_tokens(query)):
            post = self.postings.get(t)
            if not post:
                continue
            idf = math.log(1.0 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for i, c in post:
                scores[i] += idf * c * (self.K1 + 1) / (c + self.K1 * (1 - self.B + self.B * self.lengths[i] / avg))
        best = heapq.nlargest(k, scores.items(), key=lambda x: (x[1], -x[0]))   # ties: the order of ingest
        return [(s, self.docs[i][0], self.docs[i][1]) for i, s in best]


def _backend() -> object:
    """The transport the solution would use, chosen by the solution's own make_reader() from the same
    environment (ANTHROPIC_API_KEY), so that a run through the API goes through
    the same client and is counted by the same spend meter. Without a model there is nothing to
    measure: say so and stop."""
    from kb.reader import make_reader
    reader = make_reader()
    if reader is None:
        raise SystemExit("the search-and-ask baseline needs a model: set ANTHROPIC_API_KEY")
    return reader.backend


def _day(s) -> date:
    return s if isinstance(s, date) else date.fromisoformat(str(s)[:10])


class SearchAndAsk:
    """A: one pool of every download.  B: the download in force only."""

    def __init__(self, variant: str = "A", backend=None) -> None:
        self.variant = variant.upper()
        if self.variant not in ("A", "B"):
            raise ValueError("variant is A or B")
        self.backend = backend if backend is not None else _backend()
        self.pool = Index()                     # A
        # (taken_on, snapshot_id, Index), oldest first. B searches one of them; A's all hold the pool,
        # and both variants tell the model the date of the one in force.
        self.downloads: list = []
        self.failures = 0                       # model calls that returned nothing

    def ingest(self, snapshot) -> None:
        index = Index() if self.variant == "B" else self.pool
        for page in snapshot.pages:
            for p in split(page):
                index.add(p, snapshot.snapshot_id)
        self.downloads.append((_day(snapshot.taken_on), snapshot.snapshot_id, index))
        self.downloads.sort(key=lambda d: d[0])

    def _index(self, as_of: Optional[str]) -> tuple:
        """(date, index) of the download in force. A: the pool, dated by the latest download;
        `as_of` is not used. B: the latest download, or the last one on or before `as_of`."""
        if self.variant == "A" or not as_of:
            return self.downloads[-1][0], self.downloads[-1][2]
        try:
            day = _day(as_of)
        except ValueError:
            return self.downloads[-1][0], self.downloads[-1][2]
        chosen = [d for d in self.downloads if d[0] <= day]
        d = chosen[-1] if chosen else self.downloads[0]
        return d[0], d[2]

    def answer(self, question: str, as_of: Optional[str] = None):
        return self.answer_with_evidence(question, as_of)[0]

    def answer_with_evidence(self, question: str, as_of: Optional[str] = None) -> tuple:
        day, index = self._index(as_of)
        hits = index.search(str(question or ""), K)
        excerpts = [f"[{n}] {p.page.url}\n{p.heading}\n{_shown(p)}" for n, (_, p, _) in enumerate(hits, 1)]
        # The date line the solution's reader gives. Without it the model cannot tell when the
        # excerpts were true, and it declines dated questions for that reason alone.
        when = (f"The question asks about {as_of}. The excerpts are from the website as downloaded on "
                f"{day.isoformat()}, the last download on or before that date; treat that date as today."
                if as_of and self.variant == "B" else f"Today is {day.isoformat()}, the date of the latest download.")
        user = (f"{when}\n\nQuestion: {question}\n\nExcerpts:\n\n"
                + ("\n\n".join(excerpts) if excerpts else "(none found)"))
        # One call: the API transport asks again by itself, within the time one question may take,
        # and a question that still gets no answer is asked again by the next run (see above).
        try:
            raw, error = self.backend.complete(SYSTEM, user), None
        except Exception as e:                                 # a transport that raises did not answer either
            raw, error = None, f"{type(e).__name__}: {e}"
        evidence = {"excerpts": [{"snapshot": sid, "url": p.page.url, "heading": p.heading, "score": round(s, 3),
                                  "text": _shown(p)} for s, p, sid in hits],
                    "when": when, "raw": raw, "prompt_chars": len(SYSTEM) + len(user)}
        if os.environ.get("BANKGPT_DEBUG"):
            print(f"--- {question}\n{raw}\n", file=sys.stderr)
        if not raw:
            self.failures += 1
            evidence["error"] = error or "the model returned nothing"
            return Answer(text=DONT_KNOW, confident=False), evidence
        a = re.search(r"ANSWER:\s*(.*?)(?=\n\s*SOURCES:|\Z)", raw, re.S | re.I)
        s = re.search(r"SOURCES:\s*([^\n]*)", raw, re.I)
        text = (a.group(1) if a else raw).strip()
        text = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", text)          # excerpt numbers mean nothing to a member
        cited = [int(n) for n in re.findall(r"\d+", s.group(1))] if s else []
        confident = not re.match(r"\W*i (?:do not|don't) know\b", text, re.I)
        chosen = [hits[n - 1] for n in cited if 1 <= n <= len(hits)] or (hits[:3] if confident else [])
        sources, seen = [], set()
        for _, p, sid in chosen:
            if p.page.url not in seen:
                seen.add(p.page.url)
                sources.append(Source(url=p.page.url, snapshot_id=sid))
        return Answer(text=text or DONT_KNOW, sources=sources, confident=confident and bool(text)), evidence

    def conflicts(self) -> list:
        return []


def build() -> "SearchAndAsk":
    return SearchAndAsk(os.environ.get("BASELINE_VARIANT", "A"))


# --------------------------------------------------------------------------- scoring loop


def _records(path: Path) -> list:
    """The records in one records.jsonl, oldest first. A line that does not read as a record (the
    last one of a run killed while writing it) is skipped, so the run can still be resumed and
    summarized; that question is asked again."""
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                print(f"skipped a line of {path} that is not a whole record", file=sys.stderr)
    return out


def _failed(record: dict) -> bool:
    """A question recorded as a miss because no answer came back (model unavailable, an error)."""
    return bool(record.get("evidence", {}).get("error"))


def run(args) -> None:
    sys.path.insert(0, str(ROOT / "dev"))
    import load
    from scoring import score_item

    items = load.items(args.files, args.id)
    if args.part:
        k, n = (int(x) for x in args.part.split("/"))
        items = items[(k - 1) * len(items) // n: k * len(items) // n]
    path = args.out / "records.jsonl"
    newest = {(r["file"], r["id"]): r for r in (_records(path) if path.exists() else [])
              if r["variant"] == args.variant}
    # A run cut short continues where it stopped. A question that got no answer is asked again:
    # its miss counts until a later run records an answer.
    failed_before = {key for key, r in newest.items() if _failed(r)}
    done_before = newest.keys() - failed_before
    again = sum(1 for it in items if (it["_file"], it["id"]) in failed_before)
    items = [it for it in items if (it["_file"], it["id"]) not in done_before]
    if not items:
        print("every question is already recorded in", args.out)
        return
    if args.variant == "C":
        os.environ["BANKGPT_LLM"] = "off"
    if args.variant in ("A", "B"):
        ks = SearchAndAsk(args.variant)
    else:
        import solution
        ks = solution.build()
        if args.variant == "D" and ks.engine.reader is None:
            raise SystemExit("variant D needs the model: set ANTHROPIC_API_KEY")
    t = time.time()
    for snap in load.snapshots():
        ks.ingest(snap)
    size = len(ks.pool.docs) if args.variant == "A" else (
        [len(d[2].docs) for d in ks.downloads] if args.variant == "B" else "-")
    print(f"variant {args.variant}: ingest {time.time() - t:.1f}s, passages {size}, {len(items)} questions to ask "
          f"({len(done_before)} recorded before, {again} asked again after getting no answer), "
          f"load average {os.getloadavg()[0]:.1f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    sink = path.open("a")
    if path.read_bytes()[-1:] not in (b"", b"\n"):
        sink.write("\n")                    # the last run was killed mid-record: the next one starts on its own line

    def one(it):
        t0 = time.time()
        try:
            if args.variant in ("A", "B"):
                ans, evidence = ks.answer_with_evidence(it["question"], it.get("as_of"))
            else:
                ans = ks.answer(it["question"], it.get("as_of"))
                evidence = {"path": ks.engine.answer(it["question"], it.get("as_of")).path}
        except Exception as e:              # one question failing is a miss; the run goes on
            ans = Answer(text="", confident=False)
            # with the keys --summarize reads from A/B records (excerpts) and from D records (path)
            evidence = {"excerpts": [], "path": "failed", "error": f"{type(e).__name__}: {e}"}
        return ans, evidence, time.time() - t0

    pool = None
    if args.jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(args.jobs)
        done = pool.map(one, items)         # in order, each as soon as it is answered: a run cut short keeps them
    else:
        done = (one(it) for it in items)
    missed = 0
    try:
        for it, (ans, evidence, dt) in zip(items, done):
            r = score_item(it, ans)
            if evidence.get("error"):
                # No answer came back: a miss, even where the "I don't know" standing in for it
                # would pass as a refusal.
                r.update(score=0.0, correctness=0.0)
                missed += 1
            rec = {"variant": args.variant, "file": it["_file"], "id": it["id"], "category": it.get("category"),
                   "group": it.get("group"), "weight": it.get("weight", 1.0), "as_of": it.get("as_of"),
                   "question": it["question"], "must_include": it.get("must_include", []),
                   "must_not_include": it.get("must_not_include", []), "answer": ans.text, "confident": ans.confident,
                   "sources": [s.url for s in ans.sources], "score": r["score"], "correctness": r["correctness"],
                   "trap_hits": r["trap_hits"], "missed": r["missed"], "latency_s": round(dt, 3),
                   "jobs": args.jobs, "evidence": evidence}
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            print(f"{load.flag(r['score'])}{it['id']:16s} {r['score']:.2f} {dt:5.2f}s traps={r['trap_hits']} A: {ans.text[:160]!r}"
                  + (f"  [no answer: {evidence['error']}]" if evidence.get("error") else ""), flush=True)
    finally:
        sink.close()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)   # a run cut short does not wait for the rest
    if args.variant in ("A", "B"):
        print(f"model calls that returned nothing: {ks.failures}")
    if missed:
        print(f"{missed} question(s) got no answer and are recorded as misses; the same command asks them again")


def _pct(xs: list, q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def _acc(rs: list) -> float:
    w = sum(r["weight"] for r in rs)
    return sum(r["score"] * r["weight"] for r in rs) / w if w else float("nan")


def summarize(dirs) -> None:
    """Accuracy (weighted, as the scorer computes it), by category, traps and latency, per variant
    and per question set (practice = eval/public_eval.jsonl, own = dev/_local/questions/*); then, from the
    same records: consistency across rewordings, accuracy on the questions with an `as_of` date,
    accuracy by the route the solution (D) took, and for A and B where the trap values came from
    and whether the excerpts held the expected values at all."""
    from scoring import contains, score_consistency
    recs = []
    for d in dirs:
        p = Path(d) / "records.jsonl" if Path(d).is_dir() else Path(d)
        recs += _records(p)
    sets = {"practice": lambda r: r["file"] == "public_eval.jsonl", "own": lambda r: r["file"] != "public_eval.jsonl"}
    variants = sorted({r["variant"] for r in recs})
    of = {v: {(r["file"], r["id"]): r for r in recs if r["variant"] == v} for v in variants}
    for variant in variants:
        for name, keep in sets.items():
            rs = [r for r in of[variant].values() if keep(r)]
            if not rs:
                continue
            by = defaultdict(list)
            for r in rs:
                by[r["category"]].append(r)
            lat = [r["latency_s"] for r in rs]
            traps = [r for r in rs if r["trap_hits"]]
            fails = sum(1 for r in rs if _failed(r))
            declined = sum(1 for r in rs if not r["confident"] and not _failed(r))    # a failure is not a decline
            print(f"{variant} {name:8s} n={len(rs):3d} accuracy {_acc(rs):.4f}  trap answers {len(traps)}"
                  f" (of {sum(1 for r in rs if r['must_not_include'])} with a trap list)  declined "
                  f"{declined}  p50 {_pct(lat, 0.5):.2f}s p95 {_pct(lat, 0.95):.2f}s"
                  f"  model failures {fails}")
            print("    " + "  ".join(f"{c}={sum(x['score'] for x in v) / len(v):.2f}(n{len(v)})"
                                     for c, v in sorted(by.items())))

    print("\nrewordings (groups with two or more questions) and questions with an as_of date:")
    for v in variants:
        groups = defaultdict(list)
        for r in of[v].values():
            if r["group"]:
                groups[r["group"]].append(r)
        rs = [dict(id=r["id"], group=r["group"], correctness=r["correctness"], answer={"text": r["answer"]})
              for m in groups.values() if len(m) >= 2 for r in m]
        c = score_consistency(rs, [])
        dated = [r for r in of[v].values() if r["as_of"]]
        print(f"  {v}: paraphrase consistency {c['paraphrase_consistency']} over {c['paraphrase_groups']} groups;"
              f" as_of questions n={len(dated)} accuracy {_acc(dated):.3f}, trap answers"
              f" {sum(1 for r in dated if r['trap_hits'])},"
              f" declined {sum(1 for r in dated if not r['confident'] and not _failed(r))}")

    if "D" in of:
        print("\naccuracy on the questions the solution answers by each route (route taken by D):")
        for name, keep in sets.items():
            routes = defaultdict(list)
            for k, r in of["D"].items():
                if keep(r):
                    routes[r["evidence"]["path"]].append(k)
            for route, keys in sorted(routes.items()):
                print(f"  {name:8s} {route:8s} n={len(keys):3d}  " + "  ".join(
                    f"{v}={_acc([of[v][k] for k in keys if k in of[v]]):.3f}" for v in variants))

    snaps = sorted({e["snapshot"] for r in recs for e in r.get("evidence", {}).get("excerpts", [])})

    def in_force(as_of):
        days = [(re.search(r"\d{4}-\d\d-\d\d", s).group(0), s) for s in snaps]
        chosen = [s for d, s in days if not as_of or d <= as_of[:10]]
        return chosen[-1] if chosen else snaps[0]

    for v in [v for v in variants if v in ("A", "B")]:
        origin = defaultdict(list)
        for r in of[v].values():
            if not r["trap_hits"]:
                continue
            carriers = {e["snapshot"] for e in r["evidence"]["excerpts"]
                        if any(contains(e["text"], t) for t in r["trap_hits"])}
            cur = in_force(r["as_of"])
            kind = ("in no excerpt" if not carriers else "also in the download in force" if cur in carriers
                    else "only in older downloads" if max(carriers) < cur
                    else "only in newer downloads" if min(carriers) > cur else "in older and newer downloads")
            origin[kind].append(r["id"])
        print(f"\n{v}: where the trap values in its answers came from: "
              + "; ".join(f"{k} {len(ids)}" for k, ids in sorted(origin.items())))
        for k, ids in sorted(origin.items()):
            print(f"    {k}: {' '.join(ids)}")
        distinct = sorted(len({(e["url"], e["text"]) for e in r["evidence"]["excerpts"]}) for r in of[v].values())
        print(f"  distinct passages among the {K} excerpts: median {distinct[len(distinct) // 2]},"
              f" range {distinct[0]}-{distinct[-1]}")
        print("  is every expected value somewhere in the excerpts? (an upper bound: numbers can match by chance)")
        for route in (["all"] + sorted({r["evidence"]["path"] for r in of["D"].values()}) if "D" in of else ["all"]):
            keys = [k for k, r in of[v].items() if r["must_include"]
                    and (route == "all" or (k in of.get("D", {}) and of["D"][k]["evidence"]["path"] == route))]
            if not keys:
                continue
            present = [k for k in keys if all(any(contains(e["text"], t) for e in of[v][k]["evidence"]["excerpts"])
                                              for t in of[v][k]["must_include"])]
            absent = [k for k in keys if k not in present]
            print(f"    D-route {route:8s} n={len(keys):3d}: present in {len(present)} ({len(present) / len(keys):.0%});"
                  f" accuracy when present {_acc([of[v][k] for k in present]):.3f},"
                  f" when absent {_acc([of[v][k] for k in absent]):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--variant", choices=list("ABCD"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--id", action="append")
    ap.add_argument("--part", help="k/n: only the k-th of n slices of the questions")
    ap.add_argument("--jobs", type=int, default=1, help="answer in parallel (latencies then include contention)")
    ap.add_argument("--summarize", nargs="+", metavar="DIR")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if not (args.files and args.variant and args.out):
        ap.error("give question files, --variant and --out (or --summarize DIR...)")
    run(args)


if __name__ == "__main__":
    main()
