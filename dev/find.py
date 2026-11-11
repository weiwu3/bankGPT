#!/usr/bin/env python3
"""Search the website snapshots with a regular expression.

    python dev/find.py "12 Month" --snap 05          # only the latest snapshot
    python dev/find.py "routing number" -c 60 -n 20
    python dev/find.py --url savings --snap 03      # print the pages whose URL contains "savings"

Each hit shows which snapshots (01..05) contain that exact context, the page URL and the text
around the match. A development aid only; the solution does not use it.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import snapshots  # noqa: E402


def load(only=None) -> dict:
    """{snapshot id: {url: page}} for every download, or only those whose number is in `only`."""
    return {s.snapshot_id: {page.url: page for page in s.pages}
            for s in snapshots() if not only or s.snapshot_id[:2] in only}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="?")
    ap.add_argument("--snap", action="append", help="snapshot number(s), e.g. 05; repeatable")
    ap.add_argument("--url", help="print the page whose URL contains this text instead of searching")
    ap.add_argument("-c", "--context", type=int, default=100)
    ap.add_argument("-n", "--max", type=int, default=30)
    ap.add_argument("--in-url", help="only search pages whose URL contains this text")
    args = ap.parse_args()
    snaps = load(args.snap)
    if args.url:
        for name, pages in snaps.items():
            for url, page in pages.items():
                if args.url in url:
                    print(f"===== {name}  {url}  (fetched {page.fetched_at[:10]})\n{page.markdown}")
        return
    if not args.pattern:
        ap.error("give a pattern or --url")
    rx = re.compile(args.pattern, re.I)
    hits = {}
    for name, pages in snaps.items():
        for url, page in pages.items():
            if args.in_url and args.in_url not in url:
                continue
            md = page.markdown
            for m in rx.finditer(md):
                a, b = max(0, m.start() - args.context), min(len(md), m.end() + args.context)
                hits.setdefault((url, re.sub(r"\s+", " ", md[a:b])), []).append(name[:2])
    for i, ((url, text), where) in enumerate(sorted(hits.items())):
        if i >= args.max:
            print(f"... {len(hits) - i} more hits (raise -n)")
            break
        print(f"[{','.join(sorted(set(where)))}] {url}\n      …{text}…")
    if not hits:
        print("no hits", file=sys.stderr)


if __name__ == "__main__":
    main()
