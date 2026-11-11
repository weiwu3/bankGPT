"""The search-and-ask baseline (dev/baseline.py) as a submission folder for harness/run_eval.py.

    BASELINE_VARIANT=A  every download in one pool (default)
    BASELINE_VARIANT=B  the download in force only
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline import build  # noqa: E402,F401
