"""Development helper shared by the tools in dev/: the downloads as contract.Snapshot objects, read by
harness/run_eval.py's own load_snapshots, and the question files as the tools read them. Imported by the dev tools; not run on its own."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "solution"))
from run_eval import ensure_snapshots, load_snapshots  # noqa: E402


def folders(root=ROOT / "snapshots") -> list:
    """The snapshot folders, after unpacking snapshots_zip/ as the scorer does. Stops when there are
    none: every dev tool would otherwise run on nothing and report it as a result."""
    ensure_snapshots(Path(root))
    found = sorted(p for p in Path(root).iterdir() if (p / "manifest.json").exists())
    if not found:
        sys.exit(f"no downloads in {root}: put snapshots_zip/ at the top level or link the five snapshot folders there")
    return found


def snapshots(root=ROOT / "snapshots") -> list:
    """The downloads, oldest first, as the scorer loads them."""
    folders(root)
    return load_snapshots(Path(root))


def items(files, ids=None) -> list:
    """The questions in these .jsonl files, in order, each with "_file" set to its file's name; with
    ids, only the questions with one of those ids."""
    out = []
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                it = json.loads(line)
                if not ids or it["id"] in ids:
                    it["_file"] = Path(f).name
                    out.append(it)
    return out


def flag(score: float) -> str:
    """The mark a tool prints before a scored question: full marks, part marks, or none."""
    return "ok " if score >= 0.999 else ("~  " if score > 0 else "XX ")
