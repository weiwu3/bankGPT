"""Set-up shared by every test file: the import paths, the model off, and no spend carried over.

Run the tests with `python -m pytest tests`.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("harness", "solution"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))
os.environ["BANKGPT_LLM"] = "off"            # no reader from the environment: a test plugs in its own


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the five real downloads (about a minute); deselect with -m 'not slow'")


@pytest.fixture(autouse=True)
def no_spend(monkeypatch):
    """Every test starts and ends with no model usage counted and without the developer's own spend
    settings, so no count leaks into the next test or into the summary printed when pytest exits."""
    from kb import transport
    for name in ("BANKGPT_BUDGET_USD", "BANKGPT_SPEND_LOG", "BANKGPT_SPEND_LABEL"):
        monkeypatch.delenv(name, raising=False)
    transport.USAGE.reset()
    yield
    transport.USAGE.reset()
